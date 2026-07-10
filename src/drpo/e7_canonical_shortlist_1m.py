"""Fail-closed runner for the fixed E7 canonical two-dataset 1M shortlist."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import threading
from pathlib import Path
from typing import Any, Mapping, Sequence

import drpo.e7_canonical_sweep as sweep
from drpo.e7_canonical_injection import CanonicalContract
from drpo.e7_canonical_shortlist_audit import build_terminal_audit
from drpo.e7_canonical_shortlist_protocol import (
    EXPERIMENT_ID,
    RUNNER_VERSION,
    SCIENTIFIC_STATUS,
    apply_reporting_aliases,
    atomic_write_json,
    capture_repository_provenance,
    validate_fixed_protocol,
)

__all__ = [
    "apply_reporting_aliases",
    "build_terminal_audit",
    "capture_repository_provenance",
    "validate_fixed_protocol",
]


def _prepare(args: argparse.Namespace) -> tuple[
    Path,
    CanonicalContract,
    dict[str, Any],
    dict[str, Any],
    str,
    str,
    list[sweep.Branch],
    Path,
    dict[str, Any],
    dict[str, Any],
]:
    contract_path = Path(args.contract).expanduser().resolve()
    contract = CanonicalContract.load(contract_path)
    contract.verify_runtime()
    run_spec, run_spec_sha256 = sweep.load_run_spec(args.run_spec)
    grid, grid_sha256 = sweep.load_grid(args.grid)
    protocol_validation = validate_fixed_protocol(contract, run_spec, grid)

    sweep.SCIENTIFIC_STATUS = SCIENTIFIC_STATUS
    sweep.RUNNER_VERSION = RUNNER_VERSION
    generic_branches = sweep.build_branches(contract, run_spec, grid)
    branches = apply_reporting_aliases(generic_branches, grid["reporting_aliases"])
    if len(branches) != int(grid["expected_total_branches"]) or len(branches) != 56:
        raise ValueError(
            f"fixed shortlist must expand to 56 branches, got {len(branches)}"
        )
    for dataset in {branch.dataset for branch in branches}:
        dataset.verify()

    work_dir = Path(args.work_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    provenance = capture_repository_provenance(
        require_clean_main=bool(args.require_clean_main)
    )
    atomic_write_json(work_dir / "REPOSITORY_PROVENANCE_START.json", provenance)
    return (
        contract_path,
        contract,
        run_spec,
        grid,
        grid_sha256,
        run_spec_sha256,
        branches,
        work_dir,
        protocol_validation,
        provenance,
    )


def _write_plan(
    *,
    contract: CanonicalContract,
    branches: Sequence[sweep.Branch],
    grid_sha256: str,
    run_spec_sha256: str,
    work_dir: Path,
    max_workers: int,
    protocol_validation: Mapping[str, Any],
    provenance: Mapping[str, Any],
    grid: Mapping[str, Any],
) -> dict[str, Any]:
    plan = sweep.write_plan(
        contract=contract,
        branches=branches,
        grid_sha256=grid_sha256,
        run_spec_sha256=run_spec_sha256,
        work_dir=work_dir,
        max_workers=max_workers,
    )
    plan.update(
        {
            "scientific_status": SCIENTIFIC_STATUS,
            "dedicated_runner_version": RUNNER_VERSION,
            "protocol_validation": dict(protocol_validation),
            "reporting_aliases": dict(grid["reporting_aliases"]),
            "repository_provenance": dict(provenance),
            "formal_evidence_allowed": False,
            "formal_nine_task_benchmark_unlocked": False,
        }
    )
    atomic_write_json(work_dir / "EXECUTION_PLAN.json", plan)
    return plan


def _run_branches(
    *,
    args: argparse.Namespace,
    contract_path: Path,
    contract: CanonicalContract,
    run_spec: Mapping[str, Any],
    grid_sha256: str,
    run_spec_sha256: str,
    branches: Sequence[sweep.Branch],
    work_dir: Path,
) -> list[dict[str, Any]]:
    environment = {
        str(key): str(value) for key, value in run_spec.get("environment", {}).items()
    }
    trainer_template = [str(item) for item in run_spec["trainer_argv_template"]]
    results: list[dict[str, Any]] = []
    print_lock = threading.Lock()

    def run_one(branch: sweep.Branch) -> dict[str, Any]:
        result = sweep.execute_branch(
            contract_path=contract_path,
            contract=contract,
            branch=branch,
            work_dir=work_dir,
            grid_sha256=grid_sha256,
            run_spec_sha256=run_spec_sha256,
            trainer_argv_template=trainer_template,
            base_environment=environment,
            resume=args.resume,
        )
        with print_lock:
            print(json.dumps(result, sort_keys=True), flush=True)
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = [pool.submit(run_one, branch) for branch in branches]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda row: str(row["branch_id"]))
    return results


def cmd_run(args: argparse.Namespace) -> int:
    (
        contract_path,
        contract,
        run_spec,
        grid,
        grid_sha256,
        run_spec_sha256,
        branches,
        work_dir,
        protocol_validation,
        provenance_start,
    ) = _prepare(args)

    plan = _write_plan(
        contract=contract,
        branches=branches,
        grid_sha256=grid_sha256,
        run_spec_sha256=run_spec_sha256,
        work_dir=work_dir,
        max_workers=args.max_workers,
        protocol_validation=protocol_validation,
        provenance=provenance_start,
        grid=grid,
    )
    stable_plan = {key: value for key, value in plan.items() if key != "created_utc"}
    run_identity = sweep.canonical_json_sha256(stable_plan)
    run_identity_path = work_dir / "RUN_IDENTITY.json"
    if run_identity_path.is_file():
        existing = json.loads(run_identity_path.read_text())
        if existing.get("run_identity_sha256") != run_identity:
            raise RuntimeError("work directory belongs to another fixed shortlist run")
        if not args.resume:
            raise RuntimeError("work directory exists; pass --resume or use a new path")
    else:
        atomic_write_json(
            run_identity_path,
            {
                "run_identity_sha256": run_identity,
                "experiment_id": EXPERIMENT_ID,
                "scientific_status": SCIENTIFIC_STATUS,
                "plan": plan,
            },
        )

    results = _run_branches(
        args=args,
        contract_path=contract_path,
        contract=contract,
        run_spec=run_spec,
        grid_sha256=grid_sha256,
        run_spec_sha256=run_spec_sha256,
        branches=branches,
        work_dir=work_dir,
    )
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "scientific_status": SCIENTIFIC_STATUS,
        "finished_utc": sweep.utc_now(),
        "branch_count": len(results),
        "completed": sum(row["status"] in {"completed", "skipped"} for row in results),
        "failed": sum(row["status"] == "failed" for row in results),
        "results": results,
    }
    atomic_write_json(work_dir / "RUN_SUMMARY.json", summary)
    if summary["failed"]:
        raise RuntimeError(f"{summary['failed']} fixed shortlist branches failed")

    audit = build_terminal_audit(
        work_dir=work_dir,
        branches=branches,
        grid=grid,
        repository_provenance=provenance_start,
    )
    provenance_end = capture_repository_provenance(
        require_clean_main=bool(args.require_clean_main)
    )
    provenance_end["same_head_as_start"] = (
        provenance_end["head_commit"] == provenance_start["head_commit"]
    )
    if not provenance_end["same_head_as_start"]:
        raise RuntimeError("repository HEAD changed during the fixed shortlist run")
    atomic_write_json(work_dir / "REPOSITORY_PROVENANCE_END.json", provenance_end)
    summary["terminal_audit"] = "TERMINAL_AUDIT.json"
    summary["terminal_audit_status"] = audit["status"]
    summary["repository_provenance_end"] = "REPOSITORY_PROVENANCE_END.json"
    atomic_write_json(work_dir / "RUN_SUMMARY.json", summary)
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    _, _, _, grid, _, _, branches, work_dir, _, provenance = _prepare(args)
    audit = build_terminal_audit(
        work_dir=work_dir,
        branches=branches,
        grid=grid,
        repository_provenance=provenance,
    )
    print(json.dumps({"status": audit["status"], "path": "TERMINAL_AUDIT.json"}))
    return 0


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--contract", required=True)
    parser.add_argument("--run-spec", required=True)
    parser.add_argument("--grid", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--max-workers", type=int, default=40)
    parser.add_argument("--require-clean-main", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run the frozen shortlist and audit it")
    add_common_args(run)
    run.add_argument("--resume", action="store_true")
    run.set_defaults(func=cmd_run)
    audit = subparsers.add_parser("audit", help="audit an already completed shortlist")
    add_common_args(audit)
    audit.set_defaults(func=cmd_audit, resume=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_workers < 2:
        raise ValueError(
            "max_workers must be at least 2 for the registered parallel pilot"
        )
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
