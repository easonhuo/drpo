from __future__ import annotations

import json
from pathlib import Path

from drpo.countdown_e8_oracle_bank_audit import (
    AuditThresholds,
    run_standard_audit,
    thresholds_from_config,
)
from drpo.countdown_e8_oracle_bank_v2 import (
    NEGATIVE_BINS,
    audit_oracle_corpus,
    build_oracle_corpus,
)


def small_config() -> dict:
    return {
        "data": {
            "generation_seed": 1234,
            "train_rows": 60,
            "validation_rows": 12,
            "test_rows": 12,
            "corpus_train_rows": 24,
        },
        "negative_generation": {
            "seed": 2026070901,
            "random_candidates_per_prompt": 96,
            "per_bin_target": {
                "detail_wrong": 2,
                "near_value_wrong": 1,
                "mid_value_wrong": 1,
                "far_value_wrong": 1,
            },
            "near_value_error_max": 5.0,
            "mid_value_error_max": 25.0,
            "min_total_negatives_per_prompt": 3,
            "min_bins_per_prompt": 2,
            "drop_rows_with_incomplete_bins": False,
        },
    }


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_build_oracle_corpus_is_model_independent_and_oracle_anchored(tmp_path: Path) -> None:
    outputs = build_oracle_corpus(small_config(), tmp_path / "run")
    rows = read_jsonl(outputs.train_corpus)
    assert rows
    assert outputs.audit_json.exists()
    assert outputs.run_manifest.exists()
    for row in rows:
        assert row["positives"][0]["source"] == "oracle_positive"
        assert row["audit_flags"]["has_oracle_positive"] is True
        assert row["audit_flags"]["model_sampled_negatives_used"] is False
        assert row["audit_flags"]["model_scoring_used"] is False
        assert row["negatives"]
        assert {item["negative_bin"] for item in row["negatives"]} <= set(NEGATIVE_BINS)
        assert all(item["correct"] is False for item in row["negatives"])


def test_audit_reports_difficulty_and_negative_bins(tmp_path: Path) -> None:
    outputs = build_oracle_corpus(small_config(), tmp_path / "run")
    rows = read_jsonl(outputs.train_corpus)
    audit = audit_oracle_corpus(rows)
    assert audit["rows"] == len(rows)
    assert sum(audit["difficulty_counts"].values()) == len(rows)
    assert audit["model_dependency"]["positives_are_oracle_anchored"] is True
    assert audit["model_dependency"]["distance_quality_axes_decoupled"] is True
    assert audit["total_negatives"] > 0
    assert audit["negative_bin_counts_by_difficulty"]


def test_reusing_existing_outputs_without_force(tmp_path: Path) -> None:
    cfg = small_config()
    outputs1 = build_oracle_corpus(cfg, tmp_path / "run")
    manifest_mtime = outputs1.run_manifest.stat().st_mtime_ns
    outputs2 = build_oracle_corpus(cfg, tmp_path / "run")
    assert outputs2.run_manifest == outputs1.run_manifest
    assert outputs2.run_manifest.stat().st_mtime_ns == manifest_mtime


def test_standard_audit_writes_report_tables_figures_and_samples(tmp_path: Path) -> None:
    cfg = small_config()
    outputs = build_oracle_corpus(cfg, tmp_path / "run")
    rows = read_jsonl(outputs.train_corpus)
    thresholds = AuditThresholds(
        expected_rows=len(rows),
        min_negatives_per_prompt=3,
        min_bins_per_prompt=2,
        max_duplicate_negative_rate=1.0,
        max_incomplete_row_rate=1.0,
        spotcheck_per_difficulty=2,
        spotcheck_per_negative_bin=2,
    )
    summary = run_standard_audit(outputs.train_corpus, tmp_path / "audit", thresholds=thresholds)
    assert summary["row_count"] == len(rows)
    assert summary["status"] in {"PASS", "WARN", "FAIL"}
    assert (tmp_path / "audit" / "REPORT.md").exists()
    assert (tmp_path / "audit" / "summary.json").exists()
    assert (tmp_path / "audit" / "tables" / "positive_negative_coverage.csv").exists()
    assert (tmp_path / "audit" / "tables" / "negative_type_distribution.csv").exists()
    assert (tmp_path / "audit" / "figures" / "difficulty_distribution.png").exists()
    assert (tmp_path / "audit" / "figures" / "easy_medium_hard_quality_matrix.png").exists()
    assert (tmp_path / "audit" / "samples" / "suspicious_rows.jsonl").exists()
    assert (tmp_path / "audit.zip").exists()


def test_audit_thresholds_follow_config() -> None:
    thresholds = thresholds_from_config(small_config())
    assert thresholds.expected_rows == 24
    assert thresholds.min_negatives_per_prompt == 3
    assert thresholds.min_bins_per_prompt == 2
