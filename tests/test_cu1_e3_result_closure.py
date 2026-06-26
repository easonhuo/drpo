from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = REPO_ROOT / "outputs" / "cu1_e3_adam"
EXPECTED_ARTIFACT_SHA256 = (
    "2b8bfdbe6f33ed1db9dc1e59f6e9fbdb6c224c7b31b1326a7f2fbaeeaaaf522b"
)


def _experiment() -> dict:
    registry = yaml.safe_load(
        (REPO_ROOT / "experiments" / "registry.yaml").read_text()
    )
    experiments = {row["id"]: row for row in registry["experiments"]}
    return experiments["C-U1-E3-ADAM-RERUN"]


def _aggregate_rows(filename: str) -> dict[str, dict[str, str]]:
    with (RESULT_ROOT / filename).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["method"]: row for row in rows}


def test_e3_registry_is_long_run_validated_and_delivered() -> None:
    e3 = _experiment()
    assert e3["status"] == "long_run_validated"
    assert e3["formal_run_status"] == "delivered"
    assert e3["optimizer"]["name"] == "Adam"
    assert e3["held_out_seeds"] == list(range(30, 50))
    assert e3["data"]["terminology"] == "held_out_context_generalization"

    evidence = e3["evidence"]
    assert evidence["raw_complete"] is True
    assert evidence["terminal_audited"] is True
    assert evidence["terminal_audit_all_checks_passed"] is True
    assert evidence["missing_required_files"] == 0
    assert evidence["package_sha256"] == EXPECTED_ARTIFACT_SHA256
    assert evidence["delivered_to_user"] is True


def test_compact_aggregates_match_registered_causal_counts() -> None:
    fixed = _aggregate_rows("fixed_variance_aggregate.csv")
    learnable = _aggregate_rows("learnable_variance_aggregate.csv")

    assert int(fixed["baseline"]["n"]) == 20
    assert float(fixed["baseline"]["task_failure_onset_event_rate"]) == 1.0
    assert float(fixed["near_zero"]["task_failure_onset_event_rate"]) == 1.0
    assert float(fixed["far_zero"]["task_failure_onset_event_rate"]) == 0.0
    assert float(fixed["far_cap"]["task_failure_onset_event_rate"]) == 0.0

    assert int(learnable["baseline"]["n"]) == 20
    assert float(learnable["baseline"]["support_boundary_onset_event_rate"]) == 1.0
    assert float(learnable["near_zero"]["support_boundary_onset_event_rate"]) == 1.0
    assert float(learnable["far_zero"]["support_boundary_onset_event_rate"]) == 0.0
    assert float(learnable["far_cap"]["support_boundary_onset_event_rate"]) == 0.0
    assert float(learnable["global_scale"]["support_boundary_onset_event_rate"]) == 0.0

    assert float(learnable["baseline"]["support_boundary_onset"]) == 72.9
    assert float(learnable["near_zero"]["support_boundary_onset"]) == 73.1
    for row in learnable.values():
        assert float(row["unexpected_support_expansion"]) == 0.0
        assert float(row["finite_parameters"]) == 1.0
        assert float(row["log_sigma_output_finite"]) == 1.0
        assert float(row["sigma_output_finite"]) == 1.0


def test_artifact_index_and_handoff_close_e3_without_overclaiming() -> None:
    index_path = RESULT_ROOT / "ARTIFACT_INDEX.json"
    index = json.loads(index_path.read_text())
    assert index["experiment_id"] == "C-U1-E3-ADAM-RERUN"
    assert index["scientific_status"] == "long_run_validated"
    assert index["external_artifact"]["sha256"] == EXPECTED_ARTIFACT_SHA256

    for filename, metadata in index["compact_repository_files"].items():
        path = RESULT_ROOT / filename
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == metadata["sha256"]
        assert path.stat().st_size == metadata["size_bytes"]

    handoff = (REPO_ROOT / "docs" / "handoff.md").read_text()
    assert "v31（C-U1 E3 统一 Adam 因果闭环与论文结果版）" in handoff
    assert "C-U1-E3-ADAM-RERUN` 已完成 20-seed" in handoff
    assert "Baseline 与 Near-zero 均为 20/20 任务性能崩溃" in handoff
    assert "中位 onset 都为 step 73" in handoff
    assert "不得称 OOD" in handoff
    assert "不得写成“方差爆炸”" in handoff

    registry = yaml.safe_load(
        (REPO_ROOT / "experiments" / "registry.yaml").read_text()
    )
    experiments = {row["id"]: row for row in registry["experiments"]}
    assert experiments["C-U1-E4-ADAM-RERUN"]["status"] == "not_run"
    assert experiments["C-U1-E4-TAPER-01"]["status"] == "not_run"


def test_reported_confidence_intervals_match_compact_aggregates() -> None:
    fixed = _aggregate_rows("fixed_variance_aggregate.csv")
    learnable = _aggregate_rows("learnable_variance_aggregate.csv")
    summary = (RESULT_ROOT / "RESULT_SUMMARY.md").read_text()
    handoff = (REPO_ROOT / "docs" / "handoff.md").read_text()

    for rows in (fixed, learnable):
        for row in rows.values():
            snippet = (
                f"{float(row['reward']):.6f} "
                f"[{float(row['reward_ci_low']):.6f}, "
                f"{float(row['reward_ci_high']):.6f}]"
            )
            assert snippet in summary
            assert snippet in handoff
