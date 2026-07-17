#!/usr/bin/env python3
"""Plan or run the shared E7 squared-remoteness execution pipeline."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from drpo import e7_squared_exp_night as pilot
from drpo.e7_squared_exp_night_runtime_autotune import (
    revalidate_runtime,
    select_runtime,
)
from drpo.runtime_resource_autotune import (
    RuntimeResourceError,
    canonical_json_sha256,
    discover_machine,
    load_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run"))
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--contract", required=True)
    parser.add_argument("--run-spec", required=True)
    parser.add_argument("--grid", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--matched-gae-pair",
        action="store_true",
        help="run only the registered Hopper seed-200 TD/GAE engineering pair",
    )
    parser.add_argument("--fallback-workers", type=int, default=60)
    parser.add_argument("--probe-steps", type=int, default=5_000)
    parser.add_argument("--probe-seed", type=int, default=993_001)
    parser.add_argument("--probe-seconds", type=float, default=120.0)
    parser.add_argument(
        "--throughput-retention-fraction", type=float, default=0.97
    )
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


def _reject_legacy_work_dir(work_dir: Path) -> None:
    selection = work_dir / "RUNTIME_SELECTION.json"
    existing = (
        work_dir / "EXECUTION_PLAN.json",
        work_dir / "RUN_IDENTITY.json",
        work_dir / "RUN_SUMMARY.json",
    )
    if any(path.exists() for path in existing) and not selection.is_file():
        raise RuntimeResourceError(
            "refusing to attach the squared-EXP auto runner to a pre-existing "
            "fixed work directory"
        )


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
            "existing night-suite identity fixes a different worker count"
        )
    if existing_digest != selection_digest:
        raise RuntimeResourceError(
            "existing night-suite identity is not bound to the immutable selection"
        )


def _load_or_materialize_run_identity(work_dir: Path) -> dict[str, Any]:
    identity_path = work_dir / "RUN_IDENTITY.json"
    if identity_path.is_file():
        return load_json(identity_path)
    plan_path = work_dir / "EXECUTION_PLAN.json"
    if not plan_path.is_file():
        raise RuntimeResourceError(
            "plan completed without EXECUTION_PLAN.json or RUN_IDENTITY.json"
        )
    plan = load_json(plan_path)
    stable_plan = {key: value for key, value in plan.items() if key != "created_utc"}
    identity = {
        "run_identity_sha256": canonical_json_sha256(stable_plan),
        "plan": plan,
    }
    _atomic_json(identity_path, identity)
    return identity


def _bind_selection_to_run_identity(
    work_dir: Path, *, selected_workers: int, selection_digest: str
) -> None:
    identity_path = work_dir / "RUN_IDENTITY.json"
    identity = _load_or_materialize_run_identity(work_dir)
    plan = identity.get("plan")
    existing = plan.get("max_workers") if isinstance(plan, dict) else None
    if existing != selected_workers:
        raise RuntimeResourceError(
            "generated run identity does not match selected runtime workers"
        )
    identity["runtime_resource_selection"] = {
        "selection_digest": selection_digest,
        "selected_workers": selected_workers,
        "path": str(work_dir / "RUNTIME_SELECTION.json"),
        "scientific_matrix_changed": False,
    }
    _atomic_json(identity_path, identity)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _git_value(repo: Path, *args: str) -> str | None:
    try:
        value = subprocess.check_output(
            ["git", *args],
            cwd=repo,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return value or None


def _registration_snapshot(repo: Path) -> dict[str, Any]:
    experiment_id = pilot.active_experiment_id()
    protocol_path = repo / "docs" / "experiments" / f"{experiment_id}.md"
    registry_path = repo / "experiments" / "registry.yaml"
    registry_contains_id = False
    if registry_path.is_file():
        registry_contains_id = experiment_id in registry_path.read_text(encoding="utf-8")
    protocol_present = protocol_path.is_file()
    registered = protocol_present and registry_contains_id
    return {
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": experiment_id,
        "repo_commit": _git_value(repo, "rev-parse", "HEAD"),
        "repo_branch": _git_value(repo, "branch", "--show-current"),
        "protocol_document_present": protocol_present,
        "registry_contains_experiment_id": registry_contains_id,
        "authoritative_registration_complete": registered,
        "launch_mode": "registered" if registered else "code_first_pre_registration",
        "launch_blocked_by_registration": False,
        "formal_evidence_allowed_at_launch": False,
    }


def _write_launch_registration_status(repo: Path, work_dir: Path) -> Path:
    path = work_dir / "LAUNCH_REGISTRATION_STATUS.json"
    snapshot = _registration_snapshot(repo)
    if path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
    else:
        payload = {}
    if not isinstance(payload, dict) or "initial_check" not in payload:
        payload = {
            "schema_version": 1,
            "initial_check": snapshot,
            "history": [snapshot],
        }
    else:
        history = payload.get("history")
        if not isinstance(history, list):
            history = []
        latest = payload.get("latest_check")
        if not isinstance(latest, dict) or any(
            latest.get(key) != snapshot.get(key)
            for key in (
                "repo_commit",
                "repo_branch",
                "protocol_document_present",
                "registry_contains_experiment_id",
                "authoritative_registration_complete",
                "launch_mode",
            )
        ):
            history.append(snapshot)
        payload["history"] = history
    payload["latest_check"] = snapshot
    _atomic_json(path, payload)
    return path


def _write_stage_status(work_dir: Path, *, matched_pair: bool) -> Path:
    historical = pilot.active_experiment_id() == pilot.EXPERIMENT_ID
    payload = {
        "status": "BLOCKED" if historical else "READY",
        "experiment_id": pilot.active_experiment_id(),
        "stage": "stage_c_gae",
        "gae_lambda": 0.95,
        "reason": (
            "verified ordered-trajectory and terminal/truncation contract unavailable"
            if historical
            else "ordered-trajectory contract is implemented in the existing pipeline"
        ),
        "branches_started": 0,
        "planned_branch_count": pilot.active_expected_branch_count(),
        "matched_liveness_pair": bool(matched_pair),
        "scientific_result_available": False,
    }
    path = work_dir / "GAE_STAGE_STATUS.json"
    _atomic_json(path, payload)
    return path


def _write_failed_terminal_audit(work_dir: Path, error: BaseException) -> Path:
    summary_path = work_dir / "RUN_SUMMARY.json"
    summary: dict[str, Any] = {}
    if summary_path.is_file():
        try:
            loaded = json.loads(summary_path.read_text())
            if isinstance(loaded, dict):
                summary = loaded
        except (OSError, json.JSONDecodeError):
            summary = {}
    completed = int(summary.get("completed", 0) or 0)
    failed = int(summary.get("failed", 0) or 0)
    branch_count = int(summary.get("branch_count", completed + failed) or 0)
    audit = {
        "status": "FAIL",
        "experiment_id": pilot.active_experiment_id(),
        "scientific_status": pilot.active_scientific_status(),
        "raw_complete": False,
        "branch_count_observed": branch_count,
        "expected_branch_count": pilot.active_expected_branch_count(),
        "completed_or_skipped": completed,
        "failed_branches": failed,
        "error_type": type(error).__name__,
        "error": str(error),
        "task_performance_collapse_separate": True,
        "support_or_variance_boundary_separate": True,
        "nan_inf_separate": True,
        "convergence_claim_allowed": False,
        "method_ranking_claim_allowed": False,
        "actor_update_causal_claim_allowed": False,
        "gae_claim_allowed": False,
        "held_out_seeds_touched": False,
    }
    path = work_dir / "aggregate" / "terminal_audit.json"
    _atomic_json(path, audit)
    return path


def _runtime_kwargs(
    args: argparse.Namespace, *, machine: Any, repo: Path, work_dir: Path
) -> dict[str, Any]:
    return {
        "machine": machine,
        "repo_root": repo,
        "contract_path": args.contract,
        "run_spec_path": args.run_spec,
        "grid_path": args.grid,
        "work_dir": work_dir,
        "fallback_workers": args.fallback_workers,
        "probe_steps": args.probe_steps,
        "probe_seed": args.probe_seed,
        "probe_seconds": args.probe_seconds,
        "throughput_retention_fraction": args.throughput_retention_fraction,
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
    }


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


def _matched_pair_steps(run_spec_path: str | Path) -> int:
    run_spec, _ = pilot.load_run_spec(run_spec_path)
    matches = [
        item
        for item in run_spec["datasets"]
        if str(item["id"]) == pilot.GAE_LIVENESS_DATASET
    ]
    if len(matches) != 1:
        raise RuntimeResourceError(
            "GAE matched pair requires exactly one Hopper medium-expert dataset"
        )
    source = Path(str(matches[0]["path"])).expanduser().resolve()
    required = (
        "observations",
        "actions",
        "rewards",
        "terminals",
        "timeouts",
        "next_observations",
    )
    with h5py.File(source, "r") as handle:
        missing = [name for name in required if name not in handle]
        if missing:
            raise RuntimeResourceError(
                f"GAE matched pair dataset is missing fields: {missing}"
            )
        lengths = {int(handle[name].shape[0]) for name in required}
        if lengths == {0} or len(lengths) != 1:
            raise RuntimeResourceError(
                "GAE matched pair dataset fields must be non-empty and aligned"
            )
        terminals = np.asarray(handle["terminals"][:], dtype=np.bool_).reshape(-1)
        timeouts = np.asarray(handle["timeouts"][:], dtype=np.bool_).reshape(-1)
        if bool((terminals & timeouts).any()):
            raise RuntimeResourceError("terminal and timeout flags overlap")
    transition_count = lengths.pop()
    return math.ceil(transition_count / pilot.GAE_CANONICAL_BATCH_SIZE) + 1


def _configure_profile(args: argparse.Namespace) -> int | None:
    pilot.configure_execution(args.grid)
    if not args.matched_gae_pair:
        os.environ.pop("DRPO_E7_GAE_LIVENESS_PAIR", None)
        os.environ.pop("DRPO_E7_GAE_LIVENESS_STEPS", None)
        return None
    if pilot.active_experiment_id() != pilot.GAE_EXPERIMENT_ID:
        raise RuntimeResourceError(
            "--matched-gae-pair requires the GAE successor grid"
        )
    steps = _matched_pair_steps(args.run_spec)
    pilot.configure_execution(
        args.grid,
        liveness_pair=True,
        liveness_steps=steps,
    )
    os.environ["DRPO_E7_GAE_LIVENESS_PAIR"] = "1"
    os.environ["DRPO_E7_GAE_LIVENESS_STEPS"] = str(steps)
    return steps


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "plan" and args.resume:
        raise RuntimeResourceError("--resume is valid only for run")
    liveness_steps = _configure_profile(args)
    repo = Path(args.repo_root).resolve()
    work_dir = Path(args.work_dir).resolve()
    _reject_legacy_work_dir(work_dir)
    if args.command == "plan" and (work_dir / "RUN_IDENTITY.json").is_file():
        raise RuntimeResourceError(
            "cannot re-plan a work directory with a run identity"
        )
    registration_status = _write_launch_registration_status(repo, work_dir)
    _write_stage_status(work_dir, matched_pair=args.matched_gae_pair)
    machine = discover_machine(
        meminfo_path=args.meminfo,
        loadavg_path=args.loadavg,
        cgroup_root=args.cgroup_root,
    )
    kwargs = _runtime_kwargs(args, machine=machine, repo=repo, work_dir=work_dir)
    if args.command == "plan":
        document = select_runtime(**kwargs)
        workers = int(document["selection"]["selected_workers"])
        selection_digest = str(document["selection_digest"])
    else:
        planned_workers, planned_digest = _load_planned_selection(work_dir)
        _validate_existing_run_identity(work_dir, planned_workers, planned_digest)
        document = revalidate_runtime(**kwargs, proc_root=args.proc_root)
        workers = int(document["selection"]["selected_workers"])
        selection_digest = str(document["selection_digest"])
        if workers != planned_workers or selection_digest != planned_digest:
            raise RuntimeResourceError(
                "run revalidation changed immutable selection identity"
            )
    registration = json.loads(registration_status.read_text(encoding="utf-8"))[
        "latest_check"
    ]
    profile = pilot.active_runtime_profile()
    print(
        json.dumps(
            {
                "runtime_selection": str(work_dir / "RUNTIME_SELECTION.json"),
                "selection_mode": document["mode"],
                "selected_workers": workers,
                "selection_digest": selection_digest,
                "revalidation": document.get("revalidation"),
                "experiment_id": pilot.active_experiment_id(),
                "branch_count": pilot.active_expected_branch_count(),
                "matched_gae_pair": args.matched_gae_pair,
                "matched_pair_steps": liveness_steps,
                "registration_state": registration["launch_mode"],
                "registration_blocks_launch": False,
                "registration_status_file": str(registration_status),
                "selection_scope": profile,
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
    try:
        returncode = pilot.main(delegated)
        if args.command == "plan" and returncode == 0:
            _bind_selection_to_run_identity(
                work_dir,
                selected_workers=workers,
                selection_digest=selection_digest,
            )
        return int(returncode)
    except BaseException as exc:
        if args.command == "run":
            _write_failed_terminal_audit(work_dir, exc)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
