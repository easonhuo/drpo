"""Terminal audit for the fixed E7 canonical two-dataset shortlist."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

import drpo.e7_canonical_sweep as sweep
from drpo.e7_canonical_shortlist_protocol import (
    EXPECTED_DATASETS,
    EXPECTED_REPORTING_ALIASES,
    EXPECTED_SEEDS,
    EXPERIMENT_ID,
    RUNNER_VERSION,
    SCIENTIFIC_STATUS,
    atomic_write_json,
)


def _mean(values: Sequence[float]) -> float:
    return float(statistics.fmean(values))


def _sample_std(values: Sequence[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def _population_std(values: Sequence[float]) -> float:
    return float(statistics.pstdev(values)) if len(values) > 1 else 0.0


def _least_squares_slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    x_mean = _mean(xs)
    y_mean = _mean(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if denominator == 0.0:
        return 0.0
    return float(
        sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
        / denominator
    )


def _read_trainer_summary(branch_dir: Path) -> tuple[Path, dict[str, Any]]:
    matches = sorted((branch_dir / "trainer_output").glob("*_summary.json"))
    if len(matches) != 1:
        raise RuntimeError(
            "expected exactly one trainer summary below "
            f"{branch_dir}, found {len(matches)}"
        )
    return matches[0], json.loads(matches[0].read_text())


def audit_branch(
    work_dir: Path,
    branch: sweep.Branch,
    late_window_steps: Sequence[int],
) -> dict[str, Any]:
    branch_dir = work_dir / "branches" / branch.branch_id
    completed_path = branch_dir / "COMPLETED.json"
    if not completed_path.is_file():
        raise RuntimeError(f"branch is not complete: {branch.branch_id}")
    summary_path, summary = _read_trainer_summary(branch_dir)
    history = summary.get("history")
    if not isinstance(history, Mapping):
        raise RuntimeError(f"trainer summary has no history: {summary_path}")
    steps = [int(value) for value in history.get("steps", [])]
    metric_keys = [str(key) for key in history if key != "steps"]
    if len(metric_keys) != 1:
        raise RuntimeError(
            "trainer history must contain exactly one score series: "
            f"{summary_path}"
        )
    scores = [float(value) for value in history[metric_keys[0]]]
    if len(steps) != len(scores) or not steps:
        raise RuntimeError(f"trainer history shape mismatch: {summary_path}")
    if any(right <= left for left, right in zip(steps, steps[1:])):
        raise RuntimeError(
            "trainer history steps are not strictly increasing: "
            f"{summary_path}"
        )
    if any(not math.isfinite(value) for value in scores):
        raise RuntimeError(f"non-finite evaluation score in {summary_path}")

    score_by_step = dict(zip(steps, scores))
    missing_late = [step for step in late_window_steps if step not in score_by_step]
    if missing_late:
        raise RuntimeError(
            f"branch {branch.branch_id} is missing late-window evaluations: "
            f"{missing_late}"
        )
    late_scores = [score_by_step[int(step)] for step in late_window_steps]
    best_index = max(range(len(scores)), key=scores.__getitem__)
    best_score = scores[best_index]
    final_score = scores[-1]
    late_mean = _mean(late_scores)
    slope_per_step = _least_squares_slope(
        [float(step) for step in late_window_steps], late_scores
    )
    reporting_id = branch.branch_id.split("__", 2)[2]
    return {
        "branch_id": branch.branch_id,
        "dataset_id": branch.dataset.id,
        "seed": branch.seed,
        "reporting_id": reporting_id,
        "trainer_summary": str(summary_path.relative_to(work_dir)),
        "evaluation_count": len(scores),
        "late_window_steps": [int(step) for step in late_window_steps],
        "late_window_mean": late_mean,
        "late_window_std": _population_std(late_scores),
        "late_window_min": min(late_scores),
        "late_window_max": max(late_scores),
        "final_score": final_score,
        "best_score": best_score,
        "best_step": steps[best_index],
        "best_to_final_drop": best_score - final_score,
        "best_to_late_mean_drop": best_score - late_mean,
        "terminal_slope_per_step": slope_per_step,
        "terminal_slope_per_100k_steps": slope_per_step * 100_000.0,
        "late_fraction_above_registered_threshold": None,
        "registered_score_threshold": None,
        "terminal_classification": "fixed_horizon_inconclusive",
        "terminal_classification_reason": (
            "No stationarity tolerance or score threshold is registered for "
            "this pilot; the terminal slope is diagnostic only."
        ),
        "event_separation": {
            "task_performance_collapse": {
                "status": "not_classified",
                "reason": "no task-collapse threshold is registered for this pilot",
            },
            "support_or_variance_boundary_event": {
                "status": "not_available",
                "reason": (
                    "the unchanged canonical trainer summary does not expose "
                    "a boundary metric"
                ),
            },
            "nan_inf_numerical_failure": {
                "status": "absent",
                "nonfinite_evaluation_count": 0,
            },
        },
    }


def _group_rows(branch_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in branch_rows:
        key = (str(row["dataset_id"]), str(row["reporting_id"]))
        grouped.setdefault(key, []).append(row)

    result: list[dict[str, Any]] = []
    for (dataset_id, reporting_id), rows in sorted(grouped.items()):
        rows.sort(key=lambda item: int(item["seed"]))
        late_means = [float(item["late_window_mean"]) for item in rows]
        finals = [float(item["final_score"]) for item in rows]
        bests = [float(item["best_score"]) for item in rows]
        slopes = [float(item["terminal_slope_per_100k_steps"]) for item in rows]
        result.append(
            {
                "dataset_id": dataset_id,
                "reporting_id": reporting_id,
                "seeds": [int(item["seed"]) for item in rows],
                "seed_count": len(rows),
                "late_window_mean_across_seeds": _mean(late_means),
                "late_window_std_across_seeds": _sample_std(late_means),
                "late_window_min_across_seeds": min(late_means),
                "late_window_max_across_seeds": max(late_means),
                "final_mean_across_seeds": _mean(finals),
                "best_mean_across_seeds": _mean(bests),
                "best_to_final_drop_mean": _mean(
                    [float(item["best_to_final_drop"]) for item in rows]
                ),
                "best_to_late_mean_drop_mean": _mean(
                    [float(item["best_to_late_mean_drop"]) for item in rows]
                ),
                "terminal_slope_per_100k_mean": _mean(slopes),
                "terminal_classification": "fixed_horizon_inconclusive",
            }
        )
    return result


def _paired_rows(branch_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    late_by_cell = {
        (str(row["dataset_id"]), str(row["reporting_id"]), int(row["seed"])): float(
            row["late_window_mean"]
        )
        for row in branch_rows
    }
    result: list[dict[str, Any]] = []
    for dataset_id in EXPECTED_DATASETS:
        positive = {
            seed: late_by_cell[(dataset_id, "positive_only", seed)]
            for seed in EXPECTED_SEEDS
        }
        for reporting_id in EXPECTED_REPORTING_ALIASES.values():
            differences = [
                late_by_cell[(dataset_id, reporting_id, seed)] - positive[seed]
                for seed in EXPECTED_SEEDS
            ]
            result.append(
                {
                    "dataset_id": dataset_id,
                    "reporting_id": reporting_id,
                    "reference_reporting_id": "positive_only",
                    "paired_seeds": list(EXPECTED_SEEDS),
                    "late_window_difference_mean": _mean(differences),
                    "late_window_difference_std": _sample_std(differences),
                    "late_window_differences": differences,
                }
            )
    return result


def build_terminal_audit(
    *,
    work_dir: Path,
    branches: Sequence[sweep.Branch],
    grid: Mapping[str, Any],
    repository_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    late_steps = [
        int(value)
        for value in grid["fixed_protocol"]["primary_late_window_steps"]
    ]
    branch_rows = [audit_branch(work_dir, branch, late_steps) for branch in branches]
    if len(branch_rows) != 56:
        raise RuntimeError(
            f"terminal audit expected 56 branches, found {len(branch_rows)}"
        )

    payload = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "scientific_status": SCIENTIFIC_STATUS,
        "runner_version": RUNNER_VERSION,
        "status": "PASS",
        "expected_branch_count": 56,
        "audited_branch_count": len(branch_rows),
        "late_window_steps": late_steps,
        "primary_metric": "late_window_mean_750k_to_1m",
        "ranking_allowed": False,
        "steady_state_claim_allowed": False,
        "fixed_horizon_is_not_convergence": True,
        "registered_score_threshold": None,
        "stationarity_rule_registered": False,
        "repository_provenance": dict(repository_provenance),
        "branches": branch_rows,
        "groups": _group_rows(branch_rows),
        "paired_differences_vs_positive_only": _paired_rows(branch_rows),
        "event_separation_summary": {
            "task_performance_collapse": "not_classified_without_registered_threshold",
            "support_or_variance_boundary_event": (
                "not_available_in_unchanged_trainer_summary"
            ),
            "nan_inf_numerical_failure_count": 0,
        },
    }
    atomic_write_json(work_dir / "TERMINAL_AUDIT.json", payload)
    return payload
