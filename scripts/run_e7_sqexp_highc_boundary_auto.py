#!/usr/bin/env python3
"""Plan or run the 48-branch E7 squared-EXP high-c boundary pilot."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from drpo import e7_sqexp_highc_boundary as pilot
from drpo.e7_sqexp_highc_boundary_aggregate import aggregate
from drpo.e7_sqexp_highc_boundary_runtime_autotune import select_runtime
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
    parser.add_argument("--fallback-workers", type=int, default=48)
    parser.add_argument("--probe-steps", type=int, default=5_000)
    parser.add_argument("--probe-seed", type=int, default=993_301)
    parser.add_argument("--probe-seconds", type=float, default=120.0)
    parser.add_argument("--throughput-retention-fraction", type=float, default=0.97)
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


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _git_value(repo: Path, *args: str) -> str | None:
    try:
        value = subprocess.check_output(
            ["git", *args], cwd=repo, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return value or None


def _write_registration_status(repo: Path, work_dir: Path) -> Path:
    protocol = repo / "docs" / "experiments" / f"{pilot.EXPERIMENT_ID}.md"
    registry = repo / "experiments" / "registry.yaml"
    registry_contains_id = registry.is_file() and pilot.EXPERIMENT_ID in registry.read_text(
        encoding="utf-8"
    )
    protocol_present = protocol.is_file()
    registered = protocol_present and registry_contains_id
    snapshot = {
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": pilot.EXPERIMENT_ID,
        "repo_commit": _git_value(repo, "rev-parse", "HEAD"),
        "repo_branch": _git_value(repo, "branch", "--show-current"),
        "protocol_document_present": protocol_present,
        "registry_contains_experiment_id": registry_contains_id,
        "authoritative_registration_complete": registered,
        "launch_mode": "registered" if registered else "code_first_pre_registration",
        "launch_blocked_by_registration": False,
        "formal_evidence_allowed_at_launch": False,
        "predecessor_implementation_commit": pilot.PREDECESSOR_IMPLEMENTATION_COMMIT,
    }
    path = work_dir / "LAUNCH_REGISTRATION_STATUS.json"
    _atomic_json(path, {"schema_version": 1, "latest_check": snapshot})
    return path


def _selection_path(work_dir: Path) -> Path:
    return work_dir / "RUNTIME_SELECTION.json"


def _planned_workers(work_dir: Path) -> int:
    path = _selection_path(work_dir)
    if not path.is_file():
        raise RuntimeResourceError(
            "run requires an existing RUNTIME_SELECTION.json; execute plan first"
        )
    document = load_json(path)
    selection = document.get("selection")
    if not isinstance(selection, dict):
        raise RuntimeResourceError("runtime selection has no selection object")
    workers = int(selection.get("selected_workers", 0))
    if workers <= 0:
        raise RuntimeResourceError("runtime selection has invalid selected_workers")
    return workers


def _plan(args: argparse.Namespace, repo: Path, work_dir: Path) -> int:
    if (work_dir / "RUN_IDENTITY.json").exists():
        raise RuntimeResourceError("cannot re-plan a work directory with a run identity")
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
    print(
        json.dumps(
            {
                "command": "plan",
                "runtime_selection": str(_selection_path(work_dir)),
                "selected_workers": workers,
                "scientific_branch_count": pilot.EXPECTED_TOTAL_BRANCHES,
                "selection_is_immutable_for_run": True,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return pilot.main(
        [
            "plan",
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
    )


def _run(args: argparse.Namespace, work_dir: Path) -> int:
    workers = _planned_workers(work_dir)
    identity = work_dir / "RUN_IDENTITY.json"
    if identity.is_file():
        plan = load_json(identity).get("plan")
        existing = plan.get("max_workers") if isinstance(plan, dict) else None
        if int(existing) != workers:
            raise RuntimeResourceError(
                f"run identity fixes max_workers={existing}, selection fixes {workers}"
            )
    print(
        json.dumps(
            {
                "command": "run",
                "selected_workers": workers,
                "selection_recomputed": False,
                "scientific_branch_count": pilot.EXPECTED_TOTAL_BRANCHES,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    delegated = [
        "run",
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
    if args.resume:
        delegated.append("--resume")
    try:
        return pilot.main(delegated)
    except BaseException:
        if (work_dir / "branches").is_dir():
            aggregate(work_dir)
        raise


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "plan" and args.resume:
        raise RuntimeResourceError("--resume is valid only for run")
    repo = Path(args.repo_root).resolve()
    work_dir = Path(args.work_dir).resolve()
    _write_registration_status(repo, work_dir)
    if args.command == "plan":
        return _plan(args, repo, work_dir)
    return _run(args, work_dir)


if __name__ == "__main__":
    raise SystemExit(main())
