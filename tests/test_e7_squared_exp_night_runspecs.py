from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
AGENT_SCRIPTS = ROOT / "scripts" / "agent"
if str(AGENT_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(AGENT_SCRIPTS))

from runspec_delivery_policy import validate_simple_size_policy  # noqa: E402
from runspec_lib import validate_runspec  # noqa: E402
from runspec_recovery import validate_recovery_policy  # noqa: E402


PINNED_IMPLEMENTATION = "d4cdc176562db1fa439afa9f9a0cbd9a9c8d0259"
P1_PINNED_IMPLEMENTATION = "f7241a1084527f6b15be35a73a4304ba47147458"
P2_PINNED_IMPLEMENTATION = "909249875c190a75301ceb2dc2c2062ca0efcb16"
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
P1_BLOCKED_RUNSPEC = (
    ROOT
    / "runspecs"
    / "templates"
    / "E7_BENCH_JOINT_GAE_P1_FULL_20260719_01.yaml"
)
P1_FULL_RUNSPEC = (
    ROOT
    / "runspecs"
    / "templates"
    / "E7_BENCH_JOINT_GAE_P1_FULL_20260719_02.yaml"
)
P2_FULL_RUNSPEC = (
    ROOT
    / "runspecs"
    / "templates"
    / "E7_BENCH_JOINT_GAE_P2_LEFT_FULL_20260721_01.yaml"
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


def test_blocked_p1_run_id_is_retained_but_not_reusable() -> None:
    spec = validate_runspec(ROOT, P1_BLOCKED_RUNSPEC, require_registry=False)
    assert spec["run_id"] == "E7_BENCH_JOINT_GAE_P1_FULL_20260719_01"
    assert spec["repo_commit"] == P1_PINNED_IMPLEMENTATION
    assert "must not be promoted or reused" in spec["purpose"]
    assert validate_simple_size_policy(spec) == {
        "max_total_size_mb": 30,
        "max_file_size_mb": 10,
    }


def test_p1_full_run_uses_standard_v1_delivery_channel() -> None:
    spec = validate_runspec(ROOT, P1_FULL_RUNSPEC, require_registry=False)
    assert spec["run_id"] == "E7_BENCH_JOINT_GAE_P1_FULL_20260719_02"
    assert spec["experiment_id"] == "EXT-H-E7-SQEXP-GAE-01"
    assert spec["repo_commit"] == P1_PINNED_IMPLEMENTATION
    assert spec["registration"] == {"mode": "deferred", "closure_required": True}
    assert spec["policy"]["formal_evidence_allowed"] is False
    assert spec["delivery"] == {
        "enabled": True,
        "auto": True,
        "mode": "results_repo",
        "repository": "easonhuo/drpo-results",
        "branch": "ingest/e7",
        "export_profile": "manifest_text_v1",
        "max_total_size_mb": 30,
        "max_file_size_mb": 10,
    }
    assert validate_simple_size_policy(spec) == {
        "max_total_size_mb": 30,
        "max_file_size_mb": 10,
    }
    assert spec["publish"] == {"enabled": False, "auto": False}
    command = spec["entrypoint"]["command"]
    assert "DRPO_E7_P1_FULL_RUN=1" in command
    assert "E7_SQUARED_EXP_MODE=p1" in command
    assert "bench_joint_gae_p1_full_002" in command
    assert "scripts/run_e7_squared_exp_night_one_click.sh" in command
    assert "198" in " ".join(spec["success_criteria"])

    include = spec["artifacts"]["include"]
    assert spec["artifacts"]["max_package_size_mb"] == 30
    assert (
        "outputs/e7/bench_joint_gae_p1_full_002/aggregate/*.csv"
        in include
    )
    assert any(path.endswith("/GEOMETRY_DIAGNOSTICS_LATEST.json") for path in include)
    assert not any("geometry_diagnostics.jsonl" in path for path in include)
    assert not any("stdout_stderr.log" in path for path in include)
    assert not any(path.endswith("/*.log") for path in include)


def test_p2_left_full_run_uses_standard_v1_delivery_channel() -> None:
    spec = validate_runspec(ROOT, P2_FULL_RUNSPEC, require_registry=False)
    assert spec["run_id"] == "E7_BENCH_JOINT_GAE_P2_LEFT_FULL_20260721_01"
    assert spec["experiment_id"] == "EXT-H-E7-SQEXP-GAE-01"
    assert spec["repo_commit"] == P2_PINNED_IMPLEMENTATION
    assert spec["registration"] == {"mode": "deferred", "closure_required": True}
    assert spec["policy"]["formal_evidence_allowed"] is False
    assert spec["delivery"] == {
        "enabled": True,
        "auto": True,
        "mode": "results_repo",
        "repository": "easonhuo/drpo-results",
        "branch": "ingest/e7",
        "export_profile": "manifest_text_v1",
        "max_total_size_mb": 30,
        "max_file_size_mb": 10,
    }
    assert validate_simple_size_policy(spec) == {
        "max_total_size_mb": 30,
        "max_file_size_mb": 10,
    }
    assert spec["publish"] == {"enabled": False, "auto": False}

    command = spec["entrypoint"]["command"]
    assert "DRPO_E7_P2_LEFT_FULL_RUN=1" in command
    assert "E7_SQUARED_EXP_MODE=p2_left" in command
    assert "e7_bench_joint_gae_tuning_p2_left_c.json" in command
    assert "bench_joint_gae_p2_left_full_001" in command
    assert "scripts/run_e7_squared_exp_night_one_click.sh" in command

    criteria = " ".join(spec["success_criteria"])
    assert "180" in criteria
    assert "0.015625" in criteria
    assert "c 0.25 and Uncontrolled are absent" in criteria
    assert "cross-run reference" in criteria
    assert "selected_control remains null" in criteria

    recovery = validate_recovery_policy(ROOT, spec)
    assert recovery is not None
    assert recovery["max_attempts"] == 2
    assert recovery["retryable_exit_codes"] == [137, 143]
    assert "DRPO_E7_P2_LEFT_FULL_RUN=1" in recovery["resume_command"]

    include = spec["artifacts"]["include"]
    assert spec["artifacts"]["max_package_size_mb"] == 30
    assert (
        "outputs/e7/bench_joint_gae_p2_left_full_001/aggregate/*.csv"
        in include
    )
    assert any(path.endswith("/GEOMETRY_DIAGNOSTICS_LATEST.json") for path in include)
    assert not any("geometry_diagnostics.jsonl" in path for path in include)
    assert not any("stdout_stderr.log" in path for path in include)
    assert not any(path.endswith("/*.log") for path in include)


def test_p2_left_one_click_requires_standard_runspec_authorization() -> None:
    script = RUN_SCRIPT.read_text()
    assert "export DRPO_E7_P2_LEFT_FULL_RUN=1" not in script
    assert '[[ "${DRPO_E7_P2_LEFT_FULL_RUN:-0}" != "1" ]]' in script
    assert "P2-left mode is authorized only by the standard RunSpec entrypoint" in script
