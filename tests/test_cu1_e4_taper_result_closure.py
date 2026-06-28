from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = REPO_ROOT / "outputs" / "cu1_e4_taper"
RUN_COMMIT = "054c2e275cfd36e07e8883cb65d0b8df00460361"
RAW_SHA256 = "18ce26dfd9762f645095035ec24d544e4ec832e05e167402db04acb972c20b16"


def _experiments() -> dict[str, dict]:
    registry = yaml.safe_load((REPO_ROOT / "experiments" / "registry.yaml").read_text())
    return {row["id"]: row for row in registry["experiments"]}


def test_registry_records_finite_step_taper_result_without_terminal_overclaim() -> None:
    taper = _experiments()["C-U1-E4-TAPER-01"]
    assert taper["status"] == "finite_step_validated"
    assert taper["formal_run_status"] == "delivered"
    assert taper["execution"]["run_id"] == "cu1_e4_taper_054c2e2_run002"
    assert taper["execution"]["process_exit_code"] == 0
    assert taper["evidence"]["actual_runs"] == 220
    assert taper["evidence"]["actual_primary_pairs"] == 20
    assert taper["evidence"]["maximum_steps_unresolved_runs"] == 200
    assert taper["evidence"]["support_or_variance_boundary_events"] == 20
    assert taper["evidence"]["task_performance_collapse_events"] == 10
    assert taper["evidence"]["nan_inf_numerical_events"] == 0
    assert taper["evidence"]["stable_plateau_2x_confirmed_runs"] == 0
    assert taper["evidence"]["terminal_audit_all_checks_passed"] is False
    assert taper["evidence"]["raw_complete_package_sha256"] == RAW_SHA256
    assert taper["provenance"]["run_commit"] == RUN_COMMIT
    assert taper["provenance"]["provenance_compromised"] is False
    assert taper["paper_use"]["suitable_for_terminally_stable_method_ranking"] is False


def test_primary_paired_summary_matches_registered_claim() -> None:
    summary = json.loads((RESULT_ROOT / "paired_primary_summary.json").read_text())
    assert summary["rho"] == 0.25
    assert summary["paired_seeds"] == 20
    assert summary["quadratic_suppression_wins"] == 20
    assert summary["quadratic_reward_wins"] == 20
    mean, low, high = summary["reward_difference_mean_ci95"]
    assert mean == pytest.approx(0.011371958255767822)
    assert low > 0
    assert high > low
    mean, low, high = summary["far_near_ratio_difference_mean_ci95"]
    assert mean == pytest.approx(-1.6013766842192854)
    assert high < 0
    assert low < high


def test_compact_aggregate_separates_events_and_preserves_no_universal_winner() -> None:
    with (RESULT_ROOT / "aggregate.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_key = {(row["family"], float(row["rho"])): row for row in rows}
    linear = by_key[("reciprocal_linear", 0.25)]
    quadratic = by_key[("reciprocal_quadratic", 0.25)]
    exponential = by_key[("exponential", 0.25)]
    unweighted = by_key[("unweighted", 1.0)]
    assert float(quadratic["terminal_far_near_gradient_ratio_mean"]) < float(
        linear["terminal_far_near_gradient_ratio_mean"]
    )
    assert float(quadratic["reward_mean"]) > float(linear["reward_mean"])
    assert float(exponential["reward_mean"]) > float(quadratic["reward_mean"])
    assert int(unweighted["task_performance_collapse_events"]) == 10
    assert int(unweighted["support_or_variance_boundary_events"]) == 20
    assert int(unweighted["nan_inf_numerical_events"]) == 0

    audit = json.loads((RESULT_ROOT / "terminal_audit.json").read_text())
    assert audit["scientific_status"] == "finite-step validated / 有限训练步数验证"
    assert audit["all_checks_passed"] is False
    unresolved = next(
        row for row in audit["checks"] if row["name"] == "all_runs_terminally_resolved"
    )
    assert unresolved["value"]["maximum_steps"] == 200
    assert unresolved["value"]["support_or_variance_boundary_event"] == 20


def test_artifact_index_hashes_and_handoff_boundary() -> None:
    index = json.loads((RESULT_ROOT / "ARTIFACT_INDEX.json").read_text())
    assert index["experiment_id"] == "C-U1-E4-TAPER-01"
    assert index["run_commit"] == RUN_COMMIT
    assert index["repository_closure_base_commit"] == RUN_COMMIT
    assert index["scientific_status"] == "finite_step_validated"
    assert index["terminal_audit_all_checks_passed"] is False
    assert index["raw_complete_artifact"]["sha256"] == RAW_SHA256
    for filename, metadata in index["compact_repository_files"].items():
        path = RESULT_ROOT / filename
        assert path.is_file()
        assert path.stat().st_size == metadata["size_bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == metadata["sha256"]

    handoff = (REPO_ROOT / "docs" / "handoff.md").read_text()
    assert "v45（E4-TAPER 结果闭环、环境识别与公平性边界版）" in handoff
    assert "`220/220` runs" in handoff
    assert "quadratic-minus-linear" in handoff
    assert "有限训练步数验证" in handoff
    assert "不得写成 Exp、Quadratic 或其他方法的 universal winner" in handoff
    assert "同分布 held-out-context generalization" in handoff
    assert "连续环境与有限离线支持必须区分" in handoff
    assert "质量解耦不等于方向效用解耦" in handoff
    assert "超参数不能改变尾部阶数" in handoff


def test_registry_records_geometry_and_fairness_boundaries() -> None:
    taper = _experiments()["C-U1-E4-TAPER-01"]
    boundary = taper["environment_identification_boundary"]
    assert boundary["action_space_continuous"] is True
    assert boundary["reward_function_continuous"] is True
    assert boundary["equal_reward_and_advantage_are_controlled_by_design"] is True
    assert (
        boundary[
            "quality_magnitude_decoupled_from_policy_relative_distance_within_negative_set"
        ]
        is True
    )
    assert boundary["directional_utility_decoupled_from_distance"] is False
    assert boundary["not_an_ood_protocol"] is True

    fairness = taper["comparison_fairness_boundary"]
    assert fairness["allowed_claim"] == "mechanism_order_under_anchor_normalization"
    assert "reference_weight_at_d_ref" in fairness["matched"]
    assert "mean_near_negative_retention" in fairness["not_matched"]
    assert "total_negative_gradient_norm" in fairness["not_matched"]
    assert "exponential_is_universally_best" in fairness["forbidden_claims"]

    followup = taper["followup_evidence_requirements"]
    assert followup["authorization_state"] == "user_approved_registered_not_runnable"
    assert followup["no_automatic_execution"] is True
    assert "continuous_angle_sampling" in followup["geometry_robustness"]
    assert "matched_near_negative_retention" in followup["fair_family_comparison"]
    assert "new_confirmatory_seeds" in followup["fair_family_comparison"]


def test_design_and_fairness_note_is_indexed() -> None:
    note = RESULT_ROOT / "DESIGN_AND_FAIRNESS_NOTE.md"
    assert note.is_file()
    content = note.read_text()
    assert "continuous two-dimensional action space" in content
    assert "quality magnitude is distance-matched while directional utility is not" in content
    assert "anchor-normalized mechanism-order result" in content
    assert "p=2" in content

    index = json.loads((RESULT_ROOT / "ARTIFACT_INDEX.json").read_text())
    assert "DESIGN_AND_FAIRNESS_NOTE.md" in index["compact_repository_files"]
