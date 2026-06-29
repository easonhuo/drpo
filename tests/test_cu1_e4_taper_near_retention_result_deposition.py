from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "outputs" / "cu1_e4_taper_near_retention"


def test_compact_deposition_records_exact_finite_step_result_and_missing_raw_boundary() -> None:
    payload = json.loads((RESULT / "RESULT_SUMMARY.json").read_text())
    assert payload["experiment_id"] == "C-U1-E4-TAPER-NEAR-RETENTION-01"
    assert payload["run_commit"] == "69c8f532570b5c4377a0cd35ff42f0bcb77afef0"
    assert payload["scientific_status"] == "finite_step_validated"
    assert payload["formal_matrix"]["completed_runs"] == 280
    assert payload["terminal_audit"]["unresolved_at_8000_steps"] == 260
    assert payload["terminal_audit"]["steady_state_ranking_allowed"] is False
    assert payload["events"]["task_performance_collapse"] == 13
    assert payload["events"]["support_or_variance_boundary"] == 20
    assert payload["events"]["nan_inf_numerical_failure"] == 0
    provenance = payload["artifact_provenance"]
    assert provenance["raw_rows_embedded_in_repository"] is False
    assert provenance["raw_complete_artifact_available_in_current_session"] is False
    assert provenance["raw_artifact_hash_known"] is False


def test_primary_paired_summary_and_harmful_far_numbers_match_deposition() -> None:
    payload = json.loads((RESULT / "RESULT_SUMMARY.json").read_text())
    primary = payload["primary_paired_results_at_near_retention_0_75"]
    assert primary["reciprocal_quadratic"]["mean_held_out_context_reward_delta"] == pytest.approx(0.012002)
    assert primary["current_exponential"]["mean_held_out_context_reward_delta"] == pytest.approx(0.015619)
    assert primary["squared_distance_exponential"]["mean_held_out_context_reward_delta"] == pytest.approx(0.036134)
    for candidate in ("reciprocal_quadratic", "current_exponential", "squared_distance_exponential"):
        assert primary[candidate]["positive_delta_seeds"] == 20
        assert primary[candidate]["paired_seeds"] == 20
    mechanism = payload["mechanism_diagnostics_at_near_retention_0_75"]
    assert mechanism["reciprocal_linear_harmful_far_retention"] == pytest.approx(0.055886)
    assert mechanism["squared_distance_exponential_harmful_far_retention"] == pytest.approx(0.010382)

    with (RESULT / "PAIRED_PRIMARY_SUMMARY.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert {int(row["positive_delta_seeds"]) for row in rows} == {20}


def test_handoff_uses_held_out_context_and_refuses_steady_state_overclaim() -> None:
    handoff = (ROOT / "docs" / "handoff.md").read_text()
    assert "E4-TAPER Near-Retention 结果沉淀与闭环协议版" in handoff
    assert "260/280" in handoff
    assert "held-out-context reward" in handoff
    assert "禁止稳态、普遍方法排名或 OOD 表述" in handoff
    assert "原始 280-run raw-complete artifact" in handoff
