from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "du1_e5_longrun"


def _experiments() -> dict[str, dict]:
    registry = yaml.safe_load((ROOT / "experiments" / "registry.yaml").read_text())
    return {row["id"]: row for row in registry["experiments"]}


def test_e5_registry_is_long_run_closed_without_overclaim() -> None:
    e5 = _experiments()["D-U1-E5-LONGRUN-RERUN"]
    assert e5["status"] == "long_run_validated"
    assert e5["evidence"]["scientific_status"] == "long_run_validated"
    assert e5["evidence"]["raw_complete"] is True
    assert e5["evidence"]["terminal_audited"] is True
    assert e5["terminal_audit"]["actual_method_seed_runs"] == 120
    assert e5["terminal_audit"]["historical_joint_class_matches"] == 120
    assert e5["terminal_audit"]["total_nan_inf_count"] == 0
    assert e5["historical_provenance"]["exact_legacy_code_reproduction_claimed"] is False
    assert "E6" in e5["does_not_replace"]


def test_e5_compact_results_reproduce_registered_counts() -> None:
    aggregate = json.loads((OUT / "aggregate_summary.json").read_text())
    expected = {
        "positive_only": (0, 0, 20),
        "baseline": (20, 20, 0),
        "near_zero": (20, 20, 0),
        "far_zero": (0, 0, 20),
        "far_cap": (0, 0, 20),
        "global_scale": (0, 20, 0),
    }
    assert aggregate["total_runs"] == 120
    assert aggregate["all_runs_classified"] is True
    assert aggregate["all_historical_classes_match"] is True
    assert aggregate["total_nan_inf_count"] == 0
    for method, (task, support, stable) in expected.items():
        row = aggregate["methods"][method]
        assert row["task_collapse_count"] == task
        assert row["support_collapse_count"] == support
        assert row["terminal_class_counts"]["stable_bounded"] == stable
        assert row["historical_joint_match_count"] == 20


def test_direct_softmax_is_bounded_but_suppresses_support() -> None:
    direct = json.loads((OUT / "direct_summary.json").read_text())
    high = direct["high_probability_negative"]
    low = direct["low_probability_negative"]
    assert high["score_bound_pass"] is True
    assert low["score_bound_pass"] is True
    assert high["max_score"] <= 2 ** 0.5 + 1e-12
    assert low["max_score"] <= 2 ** 0.5 + 1e-12
    assert high["terminal_probability"] < 1e-11
    assert low["terminal_probability"] < 1e-19
    assert high["tail_surprisal_slope_per_step"] > 0
    assert low["tail_surprisal_slope_per_step"] > 0


def test_terminal_state_counts_are_complete() -> None:
    with (OUT / "terminal_state_counts.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 6
    assert sum(int(row["runs"]) for row in rows) == 120
    assert sum(int(row["nan_inf_numerical_failure"]) for row in rows) == 0


def test_handoff_records_e5_v37_closure_and_boundaries() -> None:
    handoff = (ROOT / "docs" / "handoff.md").read_text()
    assert "v37（D-U1 E5 长程复核闭环版）" in handoff
    assert "120/120 historical joint class match" in handoff
    assert "不声称旧未提交 runner 的 byte-identical 复现" in handoff
    assert "E6 仍是独立未完成实验" in handoff
