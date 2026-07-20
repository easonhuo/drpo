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


PINNED_IMPLEMENTATION = "d4cdc176562db1fa439afa9f9a0cbd9a9c8d0259"
PILOT_RUNSPEC = (
    ROOT
    / "runspecs"
    / "templates"
    / "E7_SQUARED_EXP_NIGHT_PILOT_20260714_01.yaml"
)
LIVENESS_RUNSPEC = (
    ROOT
    / "runspecs"
    / "templates"
    / "E7_SQUARED_EXP_NIGHT_LIVENESS_20260714_01.yaml"
)
AUTO_SCRIPT = ROOT / "scripts" / "run_e7_squared_exp_night_auto.py"
RUN_SCRIPT = ROOT / "scripts" / "run_e7_squared_exp_night_one_click.sh"
RESUME_SCRIPT = ROOT / "scripts" / "run_e7_squared_exp_night_resume_one_click.sh"
LIVENESS_SCRIPT = ROOT / "scripts" / "run_e7_squared_exp_night_liveness_one_click.sh"


def test_pilot_runspec_is_valid_and_recovery_is_bounded() -> None:
    spec = validate_runspec(ROOT, PILOT_RUNSPEC, require_registry=False)
    assert spec["experiment_id"] == "EXT-H-E7-SQUARED-EXP-NIGHT-01"
    assert spec["repo_commit"] == PINNED_IMPLEMENTATION
    assert spec["policy"]["formal_evidence_allowed"] is False
    criteria = " ".join(spec["success_criteria"])
    assert "126" in criteria
    assert "1000000" in criteria
    assert "500000" in criteria
    assert "KL-refresh diagnostics" in criteria
    assert "GAE_STAGE_STATUS.json" in criteria
    assert "LAUNCH_REGISTRATION_STATUS.json" in criteria
    assert spec["outputs"]["run_dir"].endswith("squared_exp_night_1m_001")

    recovery = validate_recovery_policy(ROOT, spec)
    assert recovery is not None
    assert recovery["max_attempts"] == 2
    assert recovery["retryable_exit_codes"] == [137, 143]
    assert recovery["resume_command"] == (
        "bash scripts/run_e7_squared_exp_night_resume_one_click.sh"
    )


def test_liveness_runspec_is_valid_and_non_scientific() -> None:
    spec = validate_runspec(ROOT, LIVENESS_RUNSPEC, require_registry=False)
    assert spec["experiment_id"] == "EXT-H-E7-SQUARED-EXP-NIGHT-01"
    assert spec["repo_commit"] == PINNED_IMPLEMENTATION
    assert spec["policy"]["scientific_aggregation_allowed"] is False
    assert spec["policy"]["formal_evidence_allowed"] is False
    assert spec["entrypoint"]["command"] == (
        "bash scripts/run_e7_squared_exp_night_liveness_one_click.sh"
    )
    script = LIVENESS_SCRIPT.read_text()
    assert 'PROBE_STEPS="${E7_SQUARED_EXP_LIVENESS_PROBE_STEPS:-500}"' in script
    assert 'MAX_WORKERS="${E7_SQUARED_EXP_LIVENESS_MAX_WORKERS:-2}"' in script
    assert validate_recovery_policy(ROOT, spec) is None


def test_code_first_launch_is_not_blocked_by_registration() -> None:
    forbidden_messages = (
        "experiment is not authoritatively registered",
        "experiment is absent from experiments/registry.yaml",
    )
    for path in (RUN_SCRIPT, RESUME_SCRIPT, LIVENESS_SCRIPT):
        text = path.read_text()
        for message in forbidden_messages:
            assert message not in text
    auto = AUTO_SCRIPT.read_text()
    assert "LAUNCH_REGISTRATION_STATUS.json" in auto
    assert '"launch_blocked_by_registration": False' in auto
    assert '"code_first_pre_registration"' in auto


def test_templates_pin_all_scientific_and_execution_paths() -> None:
    pilot = yaml.safe_load(PILOT_RUNSPEC.read_text())
    liveness = yaml.safe_load(LIVENESS_RUNSPEC.read_text())
    assert pilot["repo_commit"] == PINNED_IMPLEMENTATION
    assert liveness["repo_commit"] == PINNED_IMPLEMENTATION
    assert pilot["provenance"]["commit_policy"] == "protected_paths_unchanged"
    assert liveness["provenance"]["commit_policy"] == "protected_paths_unchanged"

    required_common = {
        "configs/e7_squared_exp_night_v1.json",
        "src/drpo/e7_squared_exp_kernel.py",
        "src/drpo/e7_ppo_kl_refresh.py",
        "src/drpo/e7_squared_exp_night.py",
        "src/drpo/e7_squared_exp_night_bootstrap.py",
        "src/drpo/e7_squared_exp_night_aggregate.py",
        "src/drpo/e7_squared_exp_night_runtime_autotune.py",
        "scripts/run_e7_squared_exp_night_auto.py",
    }
    pilot_protected = set(pilot["provenance"]["protected_paths"])
    liveness_protected = set(liveness["provenance"]["protected_paths"])
    assert required_common <= pilot_protected
    assert required_common <= liveness_protected
    assert "scripts/run_e7_squared_exp_night_one_click.sh" in pilot_protected
    assert "scripts/run_e7_squared_exp_night_resume_one_click.sh" in pilot_protected
    assert (
        "scripts/run_e7_squared_exp_night_liveness_one_click.sh"
        in liveness_protected
    )


def test_one_click_inherits_unified_worker_cap_and_is_resume_safe() -> None:
    run_text = RUN_SCRIPT.read_text()
    resume_text = RESUME_SCRIPT.read_text()
    expected_cap = (
        'MAX_WORKERS="${E7_SQUARED_EXP_MAX_WORKERS:-${DRPO_RUNTIME_MAX_WORKERS:-}}"'
    )
    assert expected_cap in run_text
    assert expected_cap in resume_text
    assert 'SELECTION_PATH="${WORK_DIR}/RUNTIME_SELECTION.json"' in run_text
    assert 'IDENTITY_PATH="${WORK_DIR}/RUN_IDENTITY.json"' in run_text
    assert 'python scripts/run_e7_squared_exp_night_auto.py run "${COMMON_ARGS[@]}" --resume' in run_text
    assert "partial runtime identity" in run_text
