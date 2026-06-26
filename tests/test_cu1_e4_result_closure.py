from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = REPO_ROOT / "outputs" / "cu1_e4_adam"
EXPECTED_BASE_COMMIT = "d699bb6b1d0093d8a9b935fd6c67f049fc3c3df0"
EXPECTED_ARTIFACT_SHA256 = (
    "c2fbc594891b594652338b8937d02d4b283e75caa7cd475572ca7307f6f08673"
)


def _experiments() -> dict[str, dict]:
    registry = yaml.safe_load(
        (REPO_ROOT / "experiments" / "registry.yaml").read_text()
    )
    return {row["id"]: row for row in registry["experiments"]}


def _rows(filename: str) -> list[dict[str, str]]:
    with (RESULT_ROOT / filename).open(newline="") as handle:
        return list(csv.DictReader(handle))


def _alpha_rows(filename: str, branch: str | None = None) -> dict[float, dict[str, str]]:
    rows = _rows(filename)
    if branch is not None:
        rows = [row for row in rows if row.get("branch") == branch]
    return {float(row["alpha"]): row for row in rows}


def test_e4_registry_records_finite_step_evidence_without_terminal_overclaim() -> None:
    experiments = _experiments()
    e4 = experiments["C-U1-E4-ADAM-RERUN"]
    taper = experiments["C-U1-E4-TAPER-01"]

    assert e4["status"] == "finite_step_validated"
    assert e4["scientific_status"] == "finite_step_validated"
    assert e4["formal_run_status"] == "delivered"
    assert e4["held_out_seeds"] == list(range(50, 70))
    assert e4["data"]["terminology"] == "held_out_context_generalization"
    assert e4["optimizer"]["name"] == "Adam"

    audit = e4["terminal_audit"]
    assert audit["integrity_checks_all_passed"] is True
    assert audit["scientific_terminal_acceptance_passed"] is False
    assert "3/20" in audit["failure_reason"]

    evidence = e4["evidence"]
    assert evidence["actual_fixed_rows"] == 160
    assert evidence["actual_learnable_rows"] == 160
    assert evidence["actual_control_rows"] == 60
    assert evidence["actual_variance_robustness_rows"] == 45
    assert evidence["missing_required_files"] == 0
    assert evidence["package_sha256"] == EXPECTED_ARTIFACT_SHA256

    assert e4["provenance"]["run_commit"] == EXPECTED_BASE_COMMIT
    assert e4["provenance"]["source_mode"] == "exact_git_bundle_checkout"
    assert e4["provenance"]["provenance_compromised"] is False

    assert taper["status"] == "not_run"
    assert taper["execution_gate"]["state"] == "ready"
    assert taper["execution_gate"]["closure_decision"] == "user_confirmed_on_2026_06_26"


def test_fixed_variance_phase_scan_and_terminal_state_counts_match() -> None:
    fixed = _alpha_rows("fixed_variance_aggregate.csv")
    counts = _alpha_rows("terminal_state_counts.csv", branch="fixed_variance")

    assert sorted(fixed) == [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
    for row in fixed.values():
        assert int(row["n"]) == 20
        assert float(row["log_sigma_output_finite"]) == 1.0
        assert float(row["sigma_output_finite"]) == 1.0

    assert float(fixed[0.0]["reward"]) == pytest.approx(0.646987646818161)
    assert float(fixed[1.0]["reward"]) == pytest.approx(0.9917025655508042)
    assert float(fixed[1.0]["normalized_extrapolation_displacement"]) == pytest.approx(
        1.007807719707489, abs=3e-7
    )

    for alpha in (0.25, 0.5, 0.75, 1.0):
        assert float(fixed[alpha]["reward"]) > float(fixed[0.0]["reward"])

    assert int(counts[1.0]["stationary_audit_both_success_count"]) == 3
    assert int(counts[1.5]["task_performance_collapse_count"]) == 20
    assert int(counts[1.75]["task_performance_collapse_count"]) == 20
    assert int(counts[1.75]["finite_continuing_drift_or_runaway_count"]) == 20
    assert int(counts[1.5]["nan_inf_count"]) == 0
    assert int(counts[1.75]["nan_inf_count"]) == 0


def test_learnable_variance_support_events_are_separate_from_numerical_failure() -> None:
    counts = _alpha_rows("terminal_state_counts.csv", branch="learnable_variance")
    robustness = _rows("variance_robustness_aggregate.csv")

    assert int(counts[0.38]["support_contraction_count"]) == 0
    assert int(counts[0.4]["support_contraction_count"]) == 18
    assert int(counts[0.5]["support_contraction_count"]) == 20
    for row in counts.values():
        assert int(row["unexpected_support_expansion_count"]) == 0
        assert int(row["nan_inf_count"]) == 0

    by_alpha: dict[float, list[dict[str, str]]] = {}
    for row in robustness:
        by_alpha.setdefault(float(row["alpha"]), []).append(row)
    assert sum(int(row["cross_minus_8_count"]) for row in by_alpha[0.38]) == 0
    assert sum(int(row["cross_minus_8_count"]) for row in by_alpha[0.4]) == 15
    assert sum(int(row["cross_minus_12_count"]) for row in by_alpha[0.4]) == 11
    assert sum(int(row["cross_minus_14_count"]) for row in by_alpha[0.5]) == 15
    assert sum(int(row["unexpected_positive_crossing_count"]) for row in robustness) == 0


def test_long_run_controls_are_descriptive_not_universal_ranking() -> None:
    controls = {row["method"]: row for row in _rows("control_aggregate.csv")}
    assert set(controls) == {"uncontrolled_all", "far_cap", "budget_matched_global"}
    for row in controls.values():
        assert int(row["n"]) == 20
        assert float(row["finite_parameters"]) == 1.0
        assert float(row["log_sigma_output_finite"]) == 1.0
        assert float(row["sigma_output_finite"]) == 1.0
        assert float(row["nonfinite_onset_event_rate"]) == 0.0

    assert float(controls["uncontrolled_all"]["reward"]) == 0.0
    assert float(controls["uncontrolled_all"]["task_failure_onset_event_rate"]) == 1.0
    assert float(controls["far_cap"]["reward"]) == pytest.approx(0.9952240824699402)
    assert float(controls["far_cap"]["task_failure_onset_event_rate"]) == 0.0
    assert float(controls["budget_matched_global"]["reward"]) == pytest.approx(
        0.502925130724907
    )
    assert float(controls["budget_matched_global"]["task_failure_onset_event_rate"]) == 0.0

    summary = (RESULT_ROOT / "RESULT_SUMMARY.md").read_text()
    assert "Method ranking was not pre-registered" in summary
    assert "raw-gradient matching is not Adam-update matching" in summary


def test_artifact_index_hashes_and_handoff_boundary_are_closed() -> None:
    index = json.loads((RESULT_ROOT / "ARTIFACT_INDEX.json").read_text())
    assert index["experiment_id"] == "C-U1-E4-ADAM-RERUN"
    assert index["repository_closure_base_commit"] == EXPECTED_BASE_COMMIT
    assert index["scientific_status"] == "finite_step_validated"
    assert index["external_artifact"]["sha256"] == EXPECTED_ARTIFACT_SHA256
    assert index["external_artifact"]["scientific_terminal_acceptance_passed"] is False

    for filename, metadata in index["compact_repository_files"].items():
        path = RESULT_ROOT / filename
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == metadata["sha256"]
        assert path.stat().st_size == metadata["size_bytes"]

    handoff = (REPO_ROOT / "docs" / "handoff.md").read_text()
    assert "v32（C-U1 E4 统一 Adam 有限步相变证据与终态门禁审计版）" in handoff
    assert "同分布 held-out-context generalization" in handoff
    assert "受益分支未通过冻结的终态残差门禁" in handoff
    assert "NaN/Inf 数值崩溃" in handoff
    assert "C-U1-E4-TAPER-01` **继续阻塞**" in handoff  # preserved v32 history
    assert "`C-U1-E4-TAPER-01` 的 E4 前置门禁解除" in handoff
