#!/usr/bin/env python3
"""Run the 186-branch E7 PPO w(0)-by-c pilot with automatic resource selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from drpo import e7_ppo_w0_grid_pilot as pilot
from drpo.e7_ppo_w0_runtime_autotune import select_runtime
from drpo.runtime_resource_autotune import (
    RuntimeResourceError,
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
    parser.add_argument("--fallback-workers", type=int, default=60)
    parser.add_argument("--probe-steps", type=int, default=5_000)
    parser.add_argument("--probe-seed", type=int, default=991_001)
    parser.add_argument("--probe-seconds", type=float, default=120.0)
    parser.add_argument(
        "--throughput-retention-fraction", type=float, default=0.97
    )
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


def _reject_legacy_work_dir(work_dir: Path) -> None:
    selection = work_dir / "RUNTIME_SELECTION.json"
    legacy_evidence = (
        work_dir / "EXECUTION_PLAN.json",
        work_dir / "RUN_IDENTITY.json",
        work_dir / "RUN_SUMMARY.json",
    )
    if any(path.exists() for path in legacy_evidence) and not selection.is_file():
        raise RuntimeResourceError(
            "refusing to attach the w(0) auto runner to a pre-existing fixed work directory"
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
            "existing w(0) run identity fixes max_workers="
            f"{existing}, but the current safe selection is {selected_workers}; "
            "resume is blocked until the original schedule is safe again"
        )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "plan" and args.resume:
        raise RuntimeResourceError("--resume is valid only for the run command")

    repo = Path(args.repo_root).resolve()
    work_dir = Path(args.work_dir).resolve()
    _reject_legacy_work_dir(work_dir)
    if args.command == "plan" and (work_dir / "RUN_IDENTITY.json").is_file():
        raise RuntimeResourceError(
            "cannot re-plan a w(0) work directory that already has a run identity"
        )

    machine = discover_machine(
        meminfo_path=args.meminfo,
        loadavg_path=args.loadavg,
        cgroup_root=args.cgroup_root,
    )
    document = select_runtime(
        machine=machine,
        repo_root=repo,
        contract_path=args.contract,
        run_spec_path=args.run_spec,
        grid_path=args.grid,
        work_dir=work_dir,
        fallback_workers=args.fallback_workers,
        probe_steps=args.probe_steps,
        probe_seed=args.probe_seed,
        probe_seconds=args.probe_seconds,
        throughput_retention_fraction=args.throughput_retention_fraction,
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
                "runtime_selection": str(work_dir / "RUNTIME_SELECTION.json"),
                "selection_mode": document["mode"],
                "selected_workers": workers,
                "scientific_branch_count": pilot.EXPECTED_TOTAL_BRANCHES,
                "selection_scope": (
                    "short empirical candidate grid under a CPU/RAM safety ceiling"
                ),
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
    return pilot.main(delegated)


if __name__ == "__main__":
    raise SystemExit(main())
