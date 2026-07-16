from __future__ import annotations

import hashlib
import json
import threading
import time
from argparse import Namespace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

from drpo.runtime_worker_admission_runner import installed_admitted_workers


def _digest(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def test_admitted_executor_width_preserves_planned_identity(tmp_path: Path) -> None:
    base = ModuleType("fake_canonical_base")
    work = tmp_path / "work"
    branches = [SimpleNamespace(branch_id=f"branch-{index}") for index in range(8)]
    active = 0
    maximum_active = 0
    lock = threading.Lock()

    def prepare(args: Namespace):
        del args
        work.mkdir(parents=True, exist_ok=True)
        return (
            tmp_path / "contract.json",
            object(),
            {"environment": {}, "trainer_argv_template": []},
            {},
            "grid-digest",
            "run-spec-digest",
            branches,
            work,
        )

    def write_plan(**kwargs: Any) -> dict[str, Any]:
        plan = {
            "created_utc": "2026-07-16T00:00:00+00:00",
            "max_workers": int(kwargs["max_workers"]),
            "branch_count": len(kwargs["branches"]),
        }
        _write(work / "EXECUTION_PLAN.json", plan)
        return plan

    def execute_branch(**kwargs: Any) -> dict[str, Any]:
        nonlocal active, maximum_active
        branch = kwargs["branch"]
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.01)
        with lock:
            active -= 1
        return {"branch_id": branch.branch_id, "status": "completed"}

    def original_cmd_run(args: Namespace) -> int:
        del args
        return 99

    base._prepare = prepare
    base.write_plan = write_plan
    base.canonical_json_sha256 = _digest
    base.atomic_write_json = _write
    base.execute_branch = execute_branch
    base.utc_now = lambda: "2026-07-16T00:00:01+00:00"
    base.cmd_run = original_cmd_run
    pilot = SimpleNamespace(base=base)
    args = Namespace(max_workers=5, resume=False)

    with installed_admitted_workers(pilot, admitted_workers=2):
        assert base.cmd_run(args) == 0
        assert base.cmd_run is not original_cmd_run

    assert base.cmd_run is original_cmd_run
    assert maximum_active == 2
    identity = json.loads((work / "RUN_IDENTITY.json").read_text())
    assert identity["plan"]["max_workers"] == 5
    summary = json.loads((work / "RUN_SUMMARY.json").read_text())
    assert summary["planned_max_workers"] == 5
    assert summary["runtime_admitted_workers"] == 2
    assert summary["scientific_matrix_changed"] is False
    assert summary["completed"] == 8
    assert summary["failed"] == 0
