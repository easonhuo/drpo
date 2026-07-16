"""Run canonical E7 branches under a pre-launch admitted worker count.

The canonical execution plan and run identity retain the reviewed planned upper bound.
Only the ThreadPool executor width is reduced to the current attempt-local admission.
Branch identities and the scientific matrix are unchanged.
"""
from __future__ import annotations

import concurrent.futures
import json
import threading
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator


def _run_with_admitted_workers(
    base: ModuleType,
    args: Any,
    *,
    admitted_workers: int,
) -> int:
    if admitted_workers < 1:
        raise ValueError("admitted_workers must be positive")
    if admitted_workers > int(args.max_workers):
        raise ValueError("admitted_workers cannot exceed planned max_workers")

    (
        contract_path,
        contract,
        run_spec,
        _,
        grid_sha256,
        run_spec_sha256,
        branches,
        work_dir,
    ) = base._prepare(args)  # noqa: SLF001
    plan = base.write_plan(
        contract=contract,
        branches=branches,
        grid_sha256=grid_sha256,
        run_spec_sha256=run_spec_sha256,
        work_dir=work_dir,
        max_workers=args.max_workers,
    )
    stable_plan = {key: value for key, value in plan.items() if key != "created_utc"}
    run_identity = base.canonical_json_sha256(stable_plan)
    run_identity_path = work_dir / "RUN_IDENTITY.json"
    if run_identity_path.is_file():
        existing = json.loads(run_identity_path.read_text())
        if existing.get("run_identity_sha256") != run_identity:
            raise RuntimeError(
                "work directory belongs to another canonical sweep; use a new path"
            )
        if not args.resume:
            raise RuntimeError("work directory exists; pass --resume or use a new path")
    else:
        base.atomic_write_json(
            run_identity_path,
            {"run_identity_sha256": run_identity, "plan": plan},
        )

    environment = {
        str(key): str(value)
        for key, value in run_spec.get("environment", {}).items()
    }
    trainer_template = [str(item) for item in run_spec["trainer_argv_template"]]
    results: list[dict[str, Any]] = []
    print_lock = threading.Lock()

    def run_one(branch: Any) -> dict[str, Any]:
        result = base.execute_branch(
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

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=admitted_workers
    ) as pool:
        futures = [pool.submit(run_one, branch) for branch in branches]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda row: row["branch_id"])
    summary = {
        "finished_utc": base.utc_now(),
        "branch_count": len(results),
        "completed": sum(
            row["status"] in {"completed", "skipped"} for row in results
        ),
        "failed": sum(row["status"] == "failed" for row in results),
        "planned_max_workers": int(args.max_workers),
        "runtime_admitted_workers": admitted_workers,
        "scientific_matrix_changed": False,
        "results": results,
    }
    base.atomic_write_json(work_dir / "RUN_SUMMARY.json", summary)
    if summary["failed"]:
        raise RuntimeError(f"{summary['failed']} canonical sweep branches failed")
    return 0


@contextmanager
def installed_admitted_workers(
    pilot_module: ModuleType,
    *,
    admitted_workers: int,
) -> Iterator[None]:
    """Temporarily replace the shared canonical run function for one invocation."""

    base = pilot_module.base
    original = base.cmd_run

    def cmd_run(args: Any) -> int:
        return _run_with_admitted_workers(
            base,
            args,
            admitted_workers=admitted_workers,
        )

    base.cmd_run = cmd_run
    try:
        yield
    finally:
        base.cmd_run = original
