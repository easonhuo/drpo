from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = REPO_ROOT / "outputs" / "cu1_e4_convergence"
EXPECTED_RUN_COMMIT = "c869df8b203f13eb8389d1d300b33f1928502871"
EXPECTED_RAW_SHA256 = "98214c2f09f7cd6ba75472bfc489771cb2ac439031e9f3636a8472a6c2a06b13"


def _experiments() -> dict[str, dict]:
    registry = yaml.safe_load((REPO_ROOT / "experiments" / "registry.yaml").read_text())
    return {row["id"]: row for row in registry["experiments"]}


def _rows(filename: str) -> list[dict[str, str]]:
    with (RESULT_ROOT / filename).open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_registry_records_completed_run_without_overclaim() -> None:
    experiments = _experiments()
    conv = experiments["C-U1-E4-CONV-01"]
    taper = experiments["C-U1-E4-TAPER-01"]

    assert conv["status"] == "long_run_validated"
    assert conv["scientific_status"] == "long_run_validated"
    assert conv["formal_run_status"] == "delivered"
    assert conv["execution"]["process_exit_code"] == 0
    assert conv["evidence"]["actual_rows"] == 60
    assert conv["evidence"]["package_sha256"] == EXPECTED_RAW_SHA256
    assert conv["provenance"]["run_commit"] == EXPECTED_RUN_COMMIT
    assert conv["provenance"]["provenance_compromised"] is False

    audit = conv["terminal_audit"]
    assert audit["integrity_checks_all_passed"] is True
    assert audit["scientific_terminal_acceptance_passed"] is False
    assert audit["pre_registered_18_of_20_gate_passed"] is False
    assert audit["user_confirmed_scoped_scientific_closure_passed"] is True
    assert audit["alpha_0_75_expected_state_count"] == 15
    assert audit["alpha_1_00_expected_state_count"] == 16
    assert audit["alpha_1_25_expected_state_count"] == 15
    assert audit["explicit_opposite_terminal_state_count"] == 0
    assert audit["scientific_role_not_reversed_count"] == 60
    assert audit["task_performance_collapse_count"] == 0
    assert audit["support_or_variance_boundary_count"] == 0
    assert audit["nan_inf_count"] == 0

    assert conv["user_confirmed_closure"]["confirmed"] is True
    assert taper["execution_gate"]["state"] == "ready"
    assert "original 18/20 gate failure remains documented" in taper["execution_gate"]["reason"]


def test_compact_rows_and_terminal_counts_match_frozen_audit() -> None:
    rows = _rows("per_seed.csv")
    counts = _rows("terminal_state_counts.csv")
    assert len(rows) == 60
    assert {float(row["alpha"]) for row in rows} == {0.75, 1.0, 1.25}
    assert {int(row["seed"]) for row in rows} == set(range(50, 70))
    assert all(int(row["steps_completed"]) == 4000 for row in rows)
    assert all(row["branch"] == "fixed_variance" for row in rows)
    assert all(row["optimizer"] == "adam" for row in rows)
    assert all(row["scientific_role_not_reversed"] == "True" for row in rows)
    assert all(row["task_performance_collapse"] == "False" for row in rows)
    assert all(row["nan_inf_numerical_failure"] == "False" for row in rows)
    assert all(not row["support_boundary_onset"] for row in rows)

    by_key = {(float(row["alpha"]), row["terminal_state"]): int(row["count"]) for row in counts}
    assert by_key[(0.75, "stable_beneficial_extrapolation")] == 15
    assert by_key[(0.75, "terminal_state_inconclusive")] == 5
    assert by_key[(1.0, "stable_beneficial_extrapolation")] == 16
    assert by_key[(1.0, "terminal_state_inconclusive")] == 4
    assert by_key[(1.25, "stable_over_extrapolation")] == 15
    assert by_key[(1.25, "terminal_state_inconclusive")] == 5

    audit = json.loads((RESULT_ROOT / "terminal_audit.json").read_text())
    assert audit["scientific_acceptance_all_passed"] is False
    assert audit["pre_registered_scientific_acceptance_all_passed"] is False
    assert audit["user_confirmed_scoped_scientific_closure_passed"] is True
    assert audit["final_scientific_status"] == "long_run_validated"
    assert audit["per_alpha"]["0.75"]["explicit_opposite_count"] == 0
    assert audit["per_alpha"]["1.00"]["explicit_opposite_count"] == 0
    assert audit["per_alpha"]["1.25"]["explicit_opposite_count"] == 0


def test_aggregate_scientific_roles_are_stable_but_gate_is_not_overridden() -> None:
    aggregates = {float(row["alpha"]): row for row in _rows("aggregate.csv")}
    assert float(aggregates[0.75]["reward"]) == pytest.approx(0.9206409364938736)
    assert float(aggregates[1.0]["reward"]) == pytest.approx(0.9982822149991989)
    assert float(aggregates[1.25]["reward"]) == pytest.approx(0.6388135522603988)

    for row in aggregates.values():
        assert float(row["scientific_role_not_reversed"]) == 1.0
        assert abs(float(row["window_2_displacement_change"])) <= 0.02
        assert abs(float(row["window_2_reward_change"])) <= 0.01
        assert float(row["window_2_over_window_1_raw_gradient_ratio"]) <= 1.25
        assert float(row["window_2_over_window_1_adam_update_ratio"]) <= 1.25
        assert float(row["support_boundary_onset_event_rate"]) == 0.0

    summary = (RESULT_ROOT / "RESULT_SUMMARY.md").read_text()
    assert "original pre-registered 18/20 per-alpha consensus **did not pass**" in summary
    assert "closed as long-run validated" in summary
    assert "not a retroactive claim that the 18/20 gate passed" in summary


def test_artifact_index_hashes_and_handoff_result_boundary() -> None:
    index = json.loads((RESULT_ROOT / "ARTIFACT_INDEX.json").read_text())
    assert index["experiment_id"] == "C-U1-E4-CONV-01"
    assert index["repository_closure_base_commit"] == "ba1e3710df4140ffaf54db3ecf12cd6f40ac531a"
    assert index["scientific_status"] == "long_run_validated"
    assert index["scientific_terminal_acceptance_passed"] is False
    assert index["pre_registered_18_of_20_gate_passed"] is False
    assert index["user_confirmed_scoped_scientific_closure_passed"] is True
    assert index["external_artifacts"]["raw_complete"]["sha256"] == EXPECTED_RAW_SHA256

    for filename, metadata in index["compact_repository_files"].items():
        path = RESULT_ROOT / filename
        assert path.is_file()
        assert path.stat().st_size == metadata["size_bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == metadata["sha256"]

    handoff = (REPO_ROOT / "docs" / "handoff.md").read_text()
    assert "v35（C-U1 E4 用户确认闭环版）" in handoff
    assert "18/20 门禁未通过" in handoff
    assert "60/60 从 step 2000 到 4000 科学角色不反转" in handoff
    assert "科学状态升级为 **已长期验证" in handoff
    assert "`C-U1-E4-TAPER-01` 的 E4 前置门禁解除" in handoff
