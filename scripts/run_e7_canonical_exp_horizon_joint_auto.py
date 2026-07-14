#!/usr/bin/env python3
"""Opt-in canonical E7 runner with measured CPU/RAM resource selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from drpo import e7_canonical_exp_horizon_grid as joint
from drpo import e7_canonical_sweep as base
from drpo.runtime_resource_adapters import (
    revalidate_e7_runtime,
    select_e7_runtime,
)
from drpo.runtime_resource_autotune import (
    RuntimeResourceError,
    discover_machine,
    load_json,
)

_MINIMUM_EVAL_WINDOWS_FOR_PROBE = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run"))
    parser.add_argument("--contract", required=True)
    parser.add_argument("--run-spec", required=True)
    parser.add_argument("--grid", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fallback-workers", type=int, default=60)
    parser.add_argument("--probe-steps", type=int, default=20_000)
    parser.add_argument("--probe-seed", type=int, default=990_001)
    parser.add_argument("--probe-seconds", type=float, default=120.0)
    parser.add_argument("--cpu-fraction", type=float, default=0.85)
    parser.add_argument("--memory-headroom-fraction", type=float, default=0.15)
    parser.add_argument("--per-worker-safety-factor", type=float, default=1.20)
    parser.add_argument(
        "--per-worker-cpu-safety-factor", type=float, default=1.25
    )
    parser.add_argument("--minimum-cpu-cores-per-worker", type=float, default=1.0)
    parser.add_argument("--max-workers", type=int)
    parser.add_argument("--max-growth-factor", type=float, default=3.0)
    parser.add_argument("--minimum-branches-for-probe", type=int, default=8)
    parser.add_argument("--revalidation-samples", type=int, default=3)
    parser.add_argument("--revalidation-sample-seconds", type=float, default=1.0)
    parser.add_argument("--meminfo", default="/proc/meminfo")
    parser.add_argument("--loadavg", default="/proc/loadavg")
    parser.add_argument("--cgroup-root", default="/sys/fs/cgroup")
    parser.add_argument("--proc-self-cgroup", default="/proc/self/cgroup")
    parser.add_argument("--proc-stat", default="/proc/stat")
    parser.add_argument("--proc-root", default="/proc")
    return parser


def _positive_int_cli_option(
    argv: Sequence[str], option_names: Sequence[str]
) -> int | None:
    """Return the last positive integer value for one CLI option."""
    result: int | None = None
    for index, token in enumerate(argv):
        raw: str | None = None
        matched_name: str | None = None
        for name in option_names:
            if token == name:
                matched_name = name
                if index + 1 >= len(argv):
                    raise RuntimeResourceError(f"{name} is missing its value")
                raw = str(argv[index + 1])
                break
            prefix = f"{name}="
            if token.startswith(prefix):
                matched_name = name
                raw = token[len(prefix) :]
                break
        if raw is None:
            continue
        try:
            parsed = int(raw)
        except ValueError as exc:
            raise RuntimeResourceError(
                f"{matched_name} must be a literal positive integer for the E7 probe"
            ) from exc
        if parsed < 1:
            raise RuntimeResourceError(
                f"{matched_name} must be positive for the E7 probe"
            )
        result = parsed
    return result


def _resolve_effective_probe_steps(
    run_spec_path: str | Path, requested_probe_steps: int
) -> int:
    """Keep the bounded probe alive past the trainer's first evaluation window."""
    if requested_probe_steps < 1:
        raise RuntimeResourceError("probe_steps must be positive")
    run_spec, _ = joint.load_exp_horizon_run_spec(str(Path(run_spec_path).resolve()))
    template = [str(item) for item in run_spec["trainer_argv_template"]]
    eval_interval = _positive_int_cli_option(
        template, ("--eval_interval", "--eval-interval")
    )
    if eval_interval is None:
        return requested_probe_steps
    return max(requested_probe_steps, _MINIMUM_EVAL_WINDOWS_FOR_PROBE * eval_interval)


def _run_with_joint_hooks(argv: list[str]) -> int:
    previous_grid = base.load_grid
    previous_run_spec = base.load_run_spec
    previous_builder = base.build_branches
    base.load_grid = joint.load_exp_horizon_grid
    base.load_run_spec = joint.load_exp_horizon_run_spec
    base.build_branches = joint.build_exp_horizon_branches
    try:
        return base.main(argv)
    finally:
        base.load_grid = previous_grid
        base.load_run_spec = previous_run_spec
        base.build_branches = previous_builder


def _reject_legacy_work_dir(work_dir: Path) -> None:
    selection = work_dir / "RUNTIME_SELECTION.json"
    legacy_evidence = (
        work_dir / "EXECUTION_PLAN.json",
        work_dir / "RUN_IDENTITY.json",
        work_dir / "RUN_SUMMARY.json",
    )
    if any(path.exists() for path in legacy_evidence) and not selection.is_file():
        raise RuntimeResourceError(
            "refusing to attach the auto runner to a pre-existing fixed E7 work directory"
        )


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _validate_existing_run_identity(
    work_dir: Path, selected_workers: int, selection_digest: str
) -> None:
    identity_path = work_dir / "RUN_IDENTITY.json"
    if not identity_path.is_file():
        raise RuntimeResourceError("run requires RUN_IDENTITY.json created by plan")
    identity = load_json(identity_path)
    plan = identity.get("plan")
    existing = plan.get("max_workers") if isinstance(plan, dict) else None
    binding = identity.get("runtime_resource_selection")
    existing_digest = binding.get("selection_digest") if isinstance(binding, dict) else None
    existing_workers = binding.get("selected_workers") if isinstance(binding, dict) else None
    if existing != selected_workers or existing_workers != selected_workers:
        raise RuntimeResourceError(
            "existing canonical E7 identity fixes a different worker count"
        )
    if existing_digest != selection_digest:
        raise RuntimeResourceError(
            "existing canonical E7 identity is not bound to the immutable selection"
        )


def _bind_selection_to_run_identity(
    work_dir: Path, *, selected_workers: int, selection_digest: str
) -> None:
    identity_path = work_dir / "RUN_IDENTITY.json"
    if not identity_path.is_file():
        raise RuntimeResourceError("plan completed without RUN_IDENTITY.json")
    identity = load_json(identity_path)
    plan = identity.get("plan")
    existing = plan.get("max_workers") if isinstance(plan, dict) else None
    if existing != selected_workers:
        raise RuntimeResourceError(
            "generated canonical E7 identity does not match selected workers"
        )
    identity["runtime_resource_selection"] = {
        "selection_digest": selection_digest,
        "selected_workers": selected_workers,
        "path": str(work_dir / "RUNTIME_SELECTION.json"),
        "scientific_matrix_changed": False,
    }
    _atomic_json(identity_path, identity)


def _load_planned_selection(work_dir: Path) -> tuple[int, str]:
    path = work_dir / "RUNTIME_SELECTION.json"
    if not path.is_file():
        raise RuntimeResourceError("run requires RUNTIME_SELECTION.json created by plan")
    document = load_json(path)
    selection = document.get("selection")
    if not isinstance(selection, dict):
        raise RuntimeResourceError("runtime selection payload is missing")
    workers = int(selection.get("selected_workers", 0) or 0)
    digest = document.get("selection_digest")
    if workers < 1 or not isinstance(digest, str) or not digest:
        raise RuntimeResourceError("runtime selection identity is malformed")
    return workers, digest


def _runtime_kwargs(
    args: argparse.Namespace,
    *,
    machine: Any,
    repo: Path,
    work_dir: Path,
    effective_probe_steps: int,
) -> dict[str, Any]:
    return {
        "machine": machine,
        "repo_root": repo,
        "contract_path": args.contract,
        "run_spec_path": args.run_spec,
        "grid_path": args.grid,
        "work_dir": work_dir,
        "fallback_workers": args.fallback_workers,
        "probe_steps": effective_probe_steps,
        "probe_seed": args.probe_seed,
        "probe_seconds": args.probe_seconds,
        "cpu_fraction": args.cpu_fraction,
        "memory_headroom_fraction": args.memory_headroom_fraction,
        "per_worker_safety_factor": args.per_worker_safety_factor,
        "per_worker_cpu_safety_factor": args.per_worker_cpu_safety_factor,
        "minimum_cpu_cores_per_worker": args.minimum_cpu_cores_per_worker,
        "max_workers": args.max_workers,
        "max_growth_factor": args.max_growth_factor,
        "minimum_branches_for_probe": args.minimum_branches_for_probe,
        "cgroup_root": args.cgroup_root,
        "proc_self_cgroup_path": args.proc_self_cgroup,
        "proc_stat_path": args.proc_stat,
        "revalidation_samples": args.revalidation_samples,
        "revalidation_sample_seconds": args.revalidation_sample_seconds,
        "throughput_retention_fraction": 1.0,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "plan" and args.resume:
        raise RuntimeResourceError("--resume is valid only for the run command")
    repo = Path(__file__).resolve().parents[1]
    work_dir = Path(args.work_dir).resolve()
    _reject_legacy_work_dir(work_dir)
    if args.command == "plan" and (work_dir / "RUN_IDENTITY.json").is_file():
        raise RuntimeResourceError(
            "cannot re-plan an E7 work directory that already has a run identity"
        )
    effective_probe_steps = _resolve_effective_probe_steps(
        args.run_spec, args.probe_steps
    )
    machine = discover_machine(
        meminfo_path=args.meminfo,
        loadavg_path=args.loadavg,
        cgroup_root=args.cgroup_root,
    )
    kwargs = _runtime_kwargs(
        args,
        machine=machine,
        repo=repo,
        work_dir=work_dir,
        effective_probe_steps=effective_probe_steps,
    )
    if args.command == "plan":
        document = select_e7_runtime(**kwargs)
        workers = int(document["selection"]["selected_workers"])
        selection_digest = str(document["selection_digest"])
    else:
        planned_workers, planned_digest = _load_planned_selection(work_dir)
        _validate_existing_run_identity(work_dir, planned_workers, planned_digest)
        document = revalidate_e7_runtime(**kwargs, proc_root=args.proc_root)
        workers = int(document["selection"]["selected_workers"])
        selection_digest = str(document["selection_digest"])
        if workers != planned_workers or selection_digest != planned_digest:
            raise RuntimeResourceError("run revalidation changed immutable selection identity")
    print(
        json.dumps(
            {
                "runtime_selection": str(work_dir / "RUNTIME_SELECTION.json"),
                "mode": document["mode"],
                "selected_workers": workers,
                "selection_digest": selection_digest,
                "revalidation": document.get("revalidation"),
                "requested_probe_steps": args.probe_steps,
                "effective_probe_steps": effective_probe_steps,
                "probe_steps_adjusted": effective_probe_steps != args.probe_steps,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    delegated = [
        args.command,
        "--contract",
        args.contract,
        "--run-spec",
        args.run_spec,
        "--grid",
        args.grid,
        "--work-dir",
        str(work_dir),
        "--max-workers",
        str(workers),
    ]
    if args.command == "run" and args.resume:
        delegated.append("--resume")
    returncode = _run_with_joint_hooks(delegated)
    if args.command == "plan" and returncode == 0:
        _bind_selection_to_run_identity(
            work_dir,
            selected_workers=workers,
            selection_digest=selection_digest,
        )
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
