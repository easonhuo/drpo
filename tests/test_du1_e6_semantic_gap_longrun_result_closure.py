from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = REPO_ROOT / "outputs" / "du1_e6_semantic_gap_longrun"
EXPERIMENT_ID = "D-U1-E6-SEMANTIC-GAP-LONGRUN-01"
RUN_COMMIT = "0907c3c0e76fc836c2bf2b752abf554c17f79f22"
CLOSURE_BASE = "fa225510e3e3e4616f36d8f586611aa6af79bf6e"
RAW_SHA256 = "65630159ef85c665a3a0ac0eba5cbf751ecb77a929f267423f7a6d9a8e5c4fbf"


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


def test_registry_closes_semantic_gap_as_finite_step_evidence() -> None:
    entry = _experiments()[EXPERIMENT_ID]
    assert entry["status"] == "finite_step_validated"
    assert entry["scientific_status"] == "finite_step_validated"
    assert entry["execution_gate"]["state"] == "blocked"
    assert entry["formal_execution"]["activation_state"] == "blocked"
    assert entry["execution"]["state"] == "delivered"
    assert entry["execution"]["process_exit_code"] == 0
    assert entry["evidence"]["actual_runs"] == 100
    assert entry["evidence"]["expected_runs"] == 100
    assert entry["evidence"]["terminal_plateau_runs"] == 45
    assert entry["evidence"]["persistent_drift_or_inconclusive_runs"] == 55
    assert entry["evidence"]["all_terminal_audits_accepted"] is True
    assert entry["evidence"]["formal_two_x_extension_performed"] is True
    assert entry["evidence"]["task_performance_collapse_events"] == 0
    assert entry["evidence"]["support_or_temperature_boundary_events"] == 0
    assert entry["evidence"]["nan_inf_numerical_events"] == 0
    assert entry["evidence"]["raw_complete_package_sha256"] == RAW_SHA256
    assert entry["evidence"]["repository_applied"] is False
    assert entry["evidence"]["applied_commit"] is None
    assert entry["provenance"]["run_commit"] == RUN_COMMIT
    assert entry["provenance"]["repository_closure_base_commit"] == CLOSURE_BASE
    assert entry["provenance"]["raw_artifact_package_kind"] == "experiment-raw-complete"
    assert entry["provenance"]["raw_artifact_is_drpo_update_input"] is False
    assert entry["provenance"]["diagnostic_rejection_phase"] == "package_extract"
    assert entry["result_summary"]["terminal"]["formal_method_ranking_allowed"] is False
    assert entry["paper_use"]["suitable_for_steady_state_method_ranking"] is False
    assert "state_distribution_OOD_generalization" in entry["paper_use"]["prohibited_claims"]


def test_final_paired_results_support_moderate_alpha_but_not_alpha_075() -> None:
    comparisons = json.loads((RESULT_ROOT / "paired_comparisons.json").read_text())[
        "comparisons"
    ]
    alpha_025 = comparisons["alpha_0_25_minus_positive_only"]
    alpha_050 = comparisons["alpha_0_5_minus_positive_only"]
    alpha_075 = comparisons["alpha_0_75_minus_positive_only"]
    alpha_100 = comparisons["alpha_1_0_minus_positive_only"]

    assert alpha_025["wins"] == 20
    assert alpha_025["mean_difference"] == pytest.approx(0.024960073828697204)
    assert alpha_025["bootstrap_ci95"][0] > 0
    assert alpha_050["wins"] == 20
    assert alpha_050["mean_difference"] == pytest.approx(0.024666464328765868)
    assert alpha_050["bootstrap_ci95"][0] > 0

    assert alpha_075["wins"] == 9
    assert alpha_075["losses"] == 11
    assert alpha_075["bootstrap_ci95"][0] < 0 < alpha_075["bootstrap_ci95"][1]

    assert alpha_100["wins"] == 0
    assert alpha_100["losses"] == 20
    assert alpha_100["mean_difference"] == pytest.approx(-0.061084982752799985)
    assert alpha_100["bootstrap_ci95"][1] < 0


def test_alpha_one_gap_worsens_across_registered_horizons() -> None:
    with (RESULT_ROOT / "horizon_summary.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    alpha_one = {
        int(row["step"]): float(row["paired_difference_vs_positive_only_mean"])
        for row in rows
        if float(row["alpha"]) == 1.0
    }
    assert alpha_one[4000] == pytest.approx(0.0039426833391189575)
    assert alpha_one[8000] == pytest.approx(-0.01374114751815796)
    assert alpha_one[16000] == pytest.approx(-0.03916723430156708)
    assert alpha_one[24000] == pytest.approx(-0.05322670340538025)
    assert alpha_one[32000] == pytest.approx(-0.061084982752799985)
    assert alpha_one[8000] > alpha_one[16000] > alpha_one[24000] > alpha_one[32000]


def test_terminal_state_counts_and_events_remain_separate() -> None:
    with (RESULT_ROOT / "terminal_state_counts.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_alpha = {float(row["alpha"]): row for row in rows}
    expected = {
        0.0: (20, 0),
        0.25: (20, 0),
        0.5: (5, 15),
        0.75: (0, 20),
        1.0: (0, 20),
    }
    for alpha, (plateau, drifting) in expected.items():
        row = by_alpha[alpha]
        assert int(row["terminal_plateau_count"]) == plateau
        assert int(row["persistent_drift_or_inconclusive_count"]) == drifting
        assert int(row["task_performance_collapse_count"]) == 0
        assert int(row["support_or_temperature_boundary_count"]) == 0
        assert int(row["nan_inf_numerical_failure_count"]) == 0


def test_artifact_index_hashes_all_compact_repository_evidence() -> None:
    index = json.loads((RESULT_ROOT / "ARTIFACT_INDEX.json").read_text())
    assert index["experiment_id"] == EXPERIMENT_ID
    assert index["scientific_run_commit"] == RUN_COMMIT
    assert index["repository_closure_base_commit"] == CLOSURE_BASE
    assert index["scientific_status"] == "finite_step_validated"
    assert index["raw_complete_artifact"]["sha256"] == RAW_SHA256
    assert index["raw_complete_artifact"]["package_kind"] == "experiment-raw-complete"
    assert index["raw_complete_artifact"]["applicable_by_drpo_update"] is False
    for filename, metadata in index["compact_repository_files"].items():
        path = RESULT_ROOT / filename
        assert path.is_file()
        assert path.stat().st_size == metadata["size_bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == metadata["sha256"]


def test_handoff_records_finite_step_boundary_and_correct_terminology() -> None:
    handoff = (REPO_ROOT / "docs" / "handoff.md").read_text()
    assert "v55（D-U1 E6 Semantic-Gap 正式结果闭环版）" in handoff
    assert "`100/100` method-seed runs" in handoff
    assert "`45/100` runs" in handoff
    assert "`55/100` 为 persistent-drift-or-inconclusive" in handoff
    assert "task-performance collapse `0/100`" in handoff
    assert "support/temperature boundary `0/100`" in handoff
    assert "NaN/Inf numerical failure `0/100`" in handoff
    assert "不得形成全 alpha 稳态方法排名" in handoff
    assert "same-distribution held-out-context generalization" in handoff
    assert "不得称 state-distribution OOD generalization" in handoff
    assert "不是 repository update" in handoff


def test_taper_successor_delivery_is_satisfied_but_four_gates_remain() -> None:
    taper = _development()["D-U1-E6-TAPER-01"]
    assert taper["semantic_gap_successor_delivery_satisfied"] is True
    assert "D-U1-E6-SEMANTIC-GAP-LONGRUN-01_delivery" not in taper["blocked_by"]
    assert taper["blocked_by"] == [
        "frozen_semantic_remoteness_coordinate",
        "frozen_paired_method_protocol",
        "frozen_untouched_held_out_seeds",
        "separately_implemented_formal_runner",
    ]
    assert taper["implementation_state"] == "not_implemented"
    assert taper["formal_execution"]["activation_state"] == "blocked"
    assert taper["formal_execution"]["entrypoint_status"] == "planned"
    assert taper["evidence"]["semantic_gap_successor_raw_complete"] is True
    assert taper["evidence"]["semantic_gap_successor_terminal_audited"] is True
    assert taper["evidence"]["semantic_gap_successor_delivered_to_user"] is True
    assert taper["evidence"]["semantic_gap_successor_scientific_status"] == (
        "finite_step_validated"
    )
