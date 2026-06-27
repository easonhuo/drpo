from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = REPO_ROOT / "outputs" / "du1_e6_semantic_longrun"
RUN_COMMIT = "eb5e12626026854f44f4698dbc8ed8829e74e0b0"
CLOSURE_BASE = "a1672d95653139964debdd5c1baf00173722c071"
RAW_SHA256 = "e098d4dd0483a661468db0cb1c4b67e4e563e2426a6aa078fe7b808f7ac691fa"
APPLIED_COMMIT = "ff2afe443167154eae5de7871cda83f3aba9a89e"


def _registry() -> dict:
    value = yaml.safe_load((REPO_ROOT / "experiments" / "registry.yaml").read_text())
    assert isinstance(value, dict)
    return value


def _experiments() -> dict[str, dict]:
    return {row["id"]: row for row in _registry()["experiments"]}


def _development() -> dict[str, dict]:
    return {
        row["id"]: row
        for row in _registry()["development_experiment_registrations"]
    }


def test_registry_closes_formal_longrun_without_reusing_held_out_seeds() -> None:
    entry = _experiments()["D-U1-E6-SEMANTIC-LONGRUN-01"]
    assert entry["status"] == "long_run_validated"
    assert entry["execution_gate"]["state"] == "blocked"
    assert entry["formal_execution"]["activation_state"] == "blocked"
    assert entry["execution"]["state"] == "delivered"
    assert entry["execution"]["process_exit_code"] == 0
    assert entry["evidence"]["actual_runs"] == 360
    assert entry["evidence"]["expected_runs"] == 360
    assert entry["evidence"]["terminal_audit_all_checks_passed"] is True
    assert entry["evidence"]["formal_two_x_extension_performed"] is True
    assert entry["evidence"]["task_performance_collapse_events"] == 0
    assert entry["evidence"]["support_or_temperature_boundary_events"] == 120
    assert entry["evidence"]["nan_inf_numerical_events"] == 0
    assert entry["evidence"]["raw_complete_package_sha256"] == RAW_SHA256
    assert entry["evidence"]["repository_applied"] is True
    assert entry["evidence"]["applied_commit"] == APPLIED_COMMIT
    assert entry["evidence"]["final_repository_closure_package_sha256"] is None
    assert entry["evidence"]["final_repository_closure_package_sha256_status"] == (
        "not_recorded_in_repository_evidence"
    )
    assert entry["provenance"]["run_commit"] == RUN_COMMIT
    assert entry["provenance"]["repository_closure_base_commit"] == CLOSURE_BASE
    assert entry["provenance"]["raw_artifact_is_drpo_update_input"] is False
    assert entry["paper_use"]["suitable_for_universal_method_ranking"] is False


def test_e6_a_formal_paired_result_is_non_monotonic() -> None:
    payload = json.loads((RESULT_ROOT / "paired_comparisons.json").read_text())
    comparisons = payload["comparisons"]
    alpha_025 = comparisons[
        "E6-A local_only alpha=0.25 minus positive_only reward"
    ]
    alpha_050 = comparisons[
        "E6-A local_only alpha=0.5 minus positive_only reward"
    ]
    alpha_075 = comparisons[
        "E6-A local_only alpha=0.75 minus positive_only reward"
    ]
    assert alpha_025["wins"] == 20
    assert alpha_025["mean_difference"] == pytest.approx(0.021537724137306213)
    assert alpha_025["bootstrap_ci95"][0] > 0
    assert alpha_050["wins"] == 20
    assert alpha_050["mean_difference"] == pytest.approx(0.01598736345767975)
    assert alpha_050["bootstrap_ci95"][0] > 0
    assert alpha_075["losses"] == 20
    assert alpha_075["mean_difference"] == pytest.approx(-0.031028494238853455)
    assert alpha_075["bootstrap_ci95"][1] < 0


def test_e6_b_separates_reward_support_and_terminal_classification() -> None:
    with (RESULT_ROOT / "condition_summary.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_key = {
        (row["protocol"], row["embedding_mode"], row["method"]): row
        for row in rows
    }
    far_zero = by_key[("E6-B", "aligned", "far_zero")]
    uncontrolled = by_key[("E6-B", "aligned", "uncontrolled")]
    far_cap = by_key[("E6-B", "aligned", "far_cap")]
    global_control = by_key[("E6-B", "aligned", "budget_matched_global")]
    assert int(far_zero["support_or_temperature_boundary_count"]) == 0
    assert int(far_zero["terminal_plateau_count"]) == 5
    assert int(far_zero["terminal_drift_or_inconclusive_count"]) == 15
    assert int(uncontrolled["support_or_temperature_boundary_count"]) == 20
    assert int(far_cap["support_or_temperature_boundary_count"]) == 20
    assert int(global_control["support_or_temperature_boundary_count"]) == 20

    comparisons = json.loads((RESULT_ROOT / "paired_comparisons.json").read_text())[
        "comparisons"
    ]
    cap = comparisons["E6-B far_cap minus uncontrolled reward"]
    budget = comparisons["E6-B budget_matched_global minus far_cap reward"]
    assert cap["bootstrap_ci95"][0] < 0 < cap["bootstrap_ci95"][1]
    assert budget["bootstrap_ci95"][0] < 0 < budget["bootstrap_ci95"][1]


def test_e6_c_alignment_control_wins_all_paired_seeds() -> None:
    comparisons = json.loads((RESULT_ROOT / "paired_comparisons.json").read_text())[
        "comparisons"
    ]
    for method in ("positive_only", "far_zero", "uncontrolled", "far_cap"):
        reward = comparisons[f"E6-C aligned minus shuffled {method} reward"]
        hidden = comparisons[
            f"E6-C aligned minus shuffled {method} hidden_optimal_probability"
        ]
        assert reward["wins"] == 20
        assert reward["bootstrap_ci95"][0] > 0
        assert hidden["wins"] == 20
        assert hidden["bootstrap_ci95"][0] > 0


def test_artifact_index_hashes_compact_evidence_and_handoff_boundaries() -> None:
    index = json.loads((RESULT_ROOT / "ARTIFACT_INDEX.json").read_text())
    assert index["experiment_id"] == "D-U1-E6-SEMANTIC-LONGRUN-01"
    assert index["scientific_run_commit"] == RUN_COMMIT
    assert index["repository_closure_base_commit"] == CLOSURE_BASE
    assert index["scientific_status"] == "long_run_validated"
    assert index["raw_complete_artifact"]["sha256"] == RAW_SHA256
    assert index["raw_complete_artifact"]["applicable_by_drpo_update"] is False
    for filename, metadata in index["compact_repository_files"].items():
        path = RESULT_ROOT / filename
        assert path.is_file()
        assert path.stat().st_size == metadata["size_bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == metadata["sha256"]

    handoff = (REPO_ROOT / "docs" / "handoff.md").read_text()
    assert "v47（D-U1 E6 长程结果闭环与 raw-complete 包类型修复版）" in handoff
    assert "`360/360` runs" in handoff
    assert "task-performance collapse `0/360`" in handoff
    assert "support/temperature boundary `120/360`" in handoff
    assert "NaN/Inf numerical failure `0/360`" in handoff
    assert "不能直接交给 `drpo-update`" in handoff
    assert "不是 OOD generalization" in handoff
    assert "review-required, not runnable" in handoff


def test_e6_taper_predecessor_is_satisfied_but_execution_stays_blocked() -> None:
    taper = _development()["D-U1-E6-TAPER-01"]
    assert taper["predecessor_delivery_satisfied"] is True
    assert "delivered_D-U1-E6-SEMANTIC-LONGRUN-01" not in taper["blocked_by"]
    assert taper["implementation_state"] == "not_implemented"
    assert taper["formal_execution"]["activation_state"] == "blocked"
    assert taper["formal_execution"]["entrypoint_status"] == "planned"
    assert taper["distance_rule"]["exact_semantic_coordinate"] == (
        "pending_separate_E6_TAPER_protocol_freeze"
    )
