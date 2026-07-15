from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
AGENT_SCRIPTS = ROOT / "scripts" / "agent"
if str(AGENT_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(AGENT_SCRIPTS))

from runspec_lib import validate_runspec  # noqa: E402
from runspec_recovery import validate_recovery_policy  # noqa: E402


PILOT_RUNSPEC = (
    ROOT / "runspecs" / "templates" / "E7_PPO_W0_EXP_GRID_PILOT_20260713_01.yaml"
)
LIVENESS_RUNSPEC = (
    ROOT
    / "runspecs"
    / "templates"
    / "E7_PPO_W0_EXP_GRID_LIVENESS_20260713_01.yaml"
)
LIVENESS_SCRIPT = ROOT / "scripts" / "run_e7_ppo_w0_grid_liveness_one_click.sh"


def test_pilot_runspec_is_structurally_valid_and_recovery_is_bounded() -> None:
    spec = validate_runspec(ROOT, PILOT_RUNSPEC, require_registry=False)
    assert spec["experiment_id"] == "EXT-H-E7-PPO-W0-EXP-GRID-01"
    assert spec["policy"]["formal_evidence_allowed"] is False
    assert "186" in " ".join(spec["success_criteria"])
    assert spec["outputs"]["run_dir"].endswith("ppo_w0_exp_grid_pilot_001")

    recovery = validate_recovery_policy(ROOT, spec)
    assert recovery is not None
    assert recovery["max_attempts"] == 2
    assert recovery["retryable_exit_codes"] == [137, 143]
    assert recovery["resume_command"] == (
        "bash scripts/run_e7_ppo_w0_grid_pilot_resume_one_click.sh"
    )


def test_liveness_runspec_is_structurally_valid_and_non_scientific() -> None:
    spec = validate_runspec(ROOT, LIVENESS_RUNSPEC, require_registry=False)
    assert spec["experiment_id"] == "EXT-H-E7-PPO-W0-EXP-GRID-01"
    assert spec["policy"]["scientific_aggregation_allowed"] is False
    assert spec["policy"]["formal_evidence_allowed"] is False
    assert spec["entrypoint"]["command"] == (
        "bash scripts/run_e7_ppo_w0_grid_liveness_one_click.sh"
    )
    script = LIVENESS_SCRIPT.read_text()
    assert 'PROBE_STEPS="${E7_PPO_W0_LIVENESS_PROBE_STEPS:-500}"' in script
    assert 'MAX_WORKERS="${E7_PPO_W0_LIVENESS_MAX_WORKERS:-2}"' in script
    assert validate_recovery_policy(ROOT, spec) is None


def test_templates_pin_protected_path_ancestors() -> None:
    pilot = yaml.safe_load(PILOT_RUNSPEC.read_text())
    liveness = yaml.safe_load(LIVENESS_RUNSPEC.read_text())
    assert len(pilot["repo_commit"]) == 40
    assert len(liveness["repo_commit"]) == 40
    assert pilot["provenance"]["commit_policy"] == "protected_paths_unchanged"
    assert liveness["provenance"]["commit_policy"] == "protected_paths_unchanged"
    pilot_protected = set(pilot["provenance"]["protected_paths"])
    liveness_protected = set(liveness["provenance"]["protected_paths"])
    assert "configs/e7_ppo_w0_exp_grid_pilot_v1.json" in pilot_protected
    assert "scripts/run_e7_ppo_w0_grid_pilot_resume_one_click.sh" in pilot_protected
    assert "scripts/run_e7_ppo_w0_grid_liveness_one_click.sh" in liveness_protected
