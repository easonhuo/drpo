from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_e7_ppo_w0_grid_pilot_auto.py"
    )
    spec = importlib.util.spec_from_file_location("e7_ppo_w0_auto", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_failed_run_writes_separate_terminal_audit(tmp_path: Path) -> None:
    module = _module()
    work = tmp_path / "run"
    work.mkdir()
    (work / "RUN_SUMMARY.json").write_text(
        json.dumps(
            {
                "branch_count": 186,
                "completed": 185,
                "failed": 1,
            }
        )
    )

    path = module._write_failed_terminal_audit(  # noqa: SLF001
        work,
        RuntimeError("one branch failed"),
    )
    audit = json.loads(path.read_text())
    assert audit["status"] == "FAIL"
    assert audit["raw_complete"] is False
    assert audit["branch_count_observed"] == 186
    assert audit["completed_or_skipped"] == 185
    assert audit["failed_branches"] == 1
    assert audit["task_performance_collapse_separate"] is True
    assert audit["support_or_variance_boundary_separate"] is True
    assert audit["nan_inf_separate"] is True
    assert audit["convergence_claim_allowed"] is False
    assert audit["held_out_seeds_touched"] is False
