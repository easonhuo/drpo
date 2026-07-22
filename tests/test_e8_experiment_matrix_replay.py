from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPLAY = ROOT / "scripts" / "run_e8_experiment_matrix_replay_ab.sh"


def test_full_family_config_driven_replay() -> None:
    completed = subprocess.run(
        ["bash", str(REPLAY)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(completed.stdout)
    assert summary["status"] == "PASS"
    assert len(summary["historical_replays"]) == 7
    assert set(summary["new_grid_candidates"]) == {
        "legacy_exp",
        "reciprocal_screen",
        "asymre_scan",
    }
    assert all(
        candidate["runtime_plan_passed"] is True
        and candidate["python_profile_edit_required"] is False
        for candidate in summary["new_grid_candidates"].values()
    )
