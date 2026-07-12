#!/usr/bin/env python3
"""Opt-in E7 runner with CPU/RAM capacity selection before execution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from drpo import e7_canonical_exp_horizon_grid as joint
from drpo import e7_canonical_sweep as base
from drpo.runtime_resource_adapters import select_e7_runtime
from drpo.runtime_resource_autotune import (
    RuntimeResourceError,
    discover_machine,
    load_json,
)


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
    parser.add_argument("--max-workers", type=int)
    parser.add_argument("--max-growth-factor", type=float, default=3.0)
    parser.add_argument("--minimum-branches-for-probe", type=int, default=8)
    parser.add_argument("--meminfo", default="/proc/meminfo")
    parser.add_argument("--loadavg", default="/proc/loadavg")
    parser.add_argument("--cgroup-root", default="/sys/fs/cgroup")
    return parser


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


def _validate_existing_run_identity(work_dir: Path, selected_workers: int) -> None:
    identity_path = work_dir / "RUN_IDENTITY.json"
    if not identity_path.is_file():
        return
    identity = load_json(identity_path)
    plan = identity.get("plan")
    existing = plan.get("max_workers") if isinstance(plan, dict) else None
    if existing != selected_workers:
        raise RuntimeResourceError(
            "existing E7 run identity fixes max_workers="
            f"{existing}, but the current safe selection is {selected_workers}; "
            "resume is blocked until the original schedule is safe again"
        )


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
    machine = discover_machine(
        meminfo_path=args.meminfo,
        loadavg_path=args.loadavg,
        cgroup_root=args.cgroup_root,
    )
    document = select_e7_runtime(
        machine=machine,
        repo_root=repo,
        contract_path=args.contract,
        run_spec_path=args.run_spec,
        grid_path=args.grid,
        work_dir=args.work_dir,
        fallback_workers=args.fallback_workers,
        probe_steps=args.probe_steps,
        probe_seed=args.probe_seed,
        probe_seconds=args.probe_seconds,
        cpu_fraction=args.cpu_fraction,
        memory_headroom_fraction=args.memory_headroom_fraction,
        per_worker_safety_factor=args.per_worker_safety_factor,
        max_workers=args.max_workers,
        max_growth_factor=args.max_growth_factor,
        minimum_branches_for_probe=args.minimum_branches_for_probe,
    )
    workers = int(document["selection"]["selected_workers"])
    _validate_existing_run_identity(work_dir, workers)
    print(
        json.dumps(
            {
                "runtime_selection": str(
                    Path(args.work_dir).resolve() / "RUNTIME_SELECTION.json"
                ),
                "mode": document["mode"],
                "selected_workers": workers,
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
        args.work_dir,
        "--max-workers",
        str(workers),
    ]
    if args.command == "run" and args.resume:
        delegated.append("--resume")
    return _run_with_joint_hooks(delegated)


if __name__ == "__main__":
    raise SystemExit(main())
