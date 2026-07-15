"""Aggregate EXT-H-E7-SQEXP-GAE-01 without failed-cell imputation."""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from drpo.e7_squared_exp_night_aggregate import (
    _aggregate_geometry,
    _aggregate_ppo,
    _atomic_json,
    _mean,
    _only,
    _sample_std,
    _score_at,
    _slope_per_100k,
    _write_csv,
)


EXPERIMENT_ID = "EXT-H-E7-SQEXP-GAE-01"
EXPECTED_BRANCHES = 192
EXPECTED_FINAL_STEP = 1_000_000
INTERMEDIATE_STEP = 500_000
LATE_WINDOW_START = 800_000
EXPECTED_SEEDS = (200, 201, 202, 203)
ACTOR_MODES = ("a2c", "ppo_clip_k4")
ADVANTAGE_ESTIMATORS = ("td", "gae")


def _history(summary: dict[str, Any]) -> tuple[list[int], list[float], dict[str, list[float]]]:
    history = summary.get("history")
    if not isinstance(history, dict):
        raise RuntimeError("trainer summary has no history mapping")
    steps = [int(value) for value in history.get("steps", [])]
    scores = [float(value) for value in history.get("score", [])]
    if not steps or len(steps) != len(scores):
        raise RuntimeError("trainer history steps/score length mismatch")
    if steps != sorted(steps) or len(steps) != len(set(steps)):
        raise RuntimeError("trainer history steps must be unique and sorted")
    support_names = (
        "actor_log_std_min",
        "actor_log_std_max",
        "actor_sigma_min",
        "actor_sigma_max",
        "lower_log_std_boundary_fraction",
        "upper_log_std_boundary_fraction",
    )
    support: dict[str, list[float]] = {}
    for name in support_names:
        values = [float(value) for value in history.get(name, [])]
        if len(values) != len(steps):
            raise RuntimeError(f"support history length mismatch: {name}")
        if not all(math.isfinite(value) for value in values):
            raise FloatingPointError(f"support history contains NaN/Inf: {name}")
        support[name] = values
    return steps, scores, support


def _coefficient(control: dict[str, Any]) -> float | None:
    return None if control["method"] == "positive_only" else float(control["exp_coefficient"])


def _control_label(value: float | None) -> str:
    return "positive_only" if value is None else f"c={value:g}"


def aggregate(work_dir: str | Path) -> dict[str, Any]:
    work = Path(work_dir).expanduser().resolve()
    branch_root = work / "branches"
    if not branch_root.is_dir():
        raise RuntimeError("GAE branch directory is missing")
    branch_dirs = sorted(path for path in branch_root.iterdir() if path.is_dir())

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    numerical_failures = 0
    support_events = 0
    for branch_dir in branch_dirs:
        config_path = branch_dir / "branch_config.json"
        if not config_path.is_file():
            failures.append({"branch_id": branch_dir.name, "reason": "missing_branch_config"})
            continue
        branch = json.loads(config_path.read_text())
        if branch.get("experiment_id") != EXPERIMENT_ID:
            raise RuntimeError(f"branch experiment mismatch: {branch_dir.name}")
        if not (branch_dir / "COMPLETED.json").is_file():
            failed_path = branch_dir / "FAILED.json"
            failed = json.loads(failed_path.read_text()) if failed_path.is_file() else {}
            failures.append(
                {
                    "branch_id": branch.get("branch_id", branch_dir.name),
                    "dataset": branch.get("dataset_id"),
                    "seed": branch.get("seed"),
                    "return_code": failed.get("return_code"),
                    "reason": "branch_not_completed",
                }
            )
            continue

        values = branch["template_values"]
        seed = int(branch["seed"])
        estimator = str(values["advantage_estimator"])
        actor_mode = str(values["actor_update_mode"])
        if seed not in EXPECTED_SEEDS:
            raise RuntimeError(f"forbidden seed in branch: {branch_dir.name}")
        if estimator not in ADVANTAGE_ESTIMATORS or actor_mode not in ACTOR_MODES:
            raise RuntimeError(f"unknown branch mode: {branch_dir.name}")
        control = branch["weight_control"]
        if control.get("formula") != "w(d)=w(0)*exp(-c*(d/2)^2)":
            raise RuntimeError(f"branch formula mismatch: {branch_dir.name}")

        summary_path = _only(
            (branch_dir / "trainer_output").glob("*_summary.json"),
            "trainer summary",
        )
        summary = json.loads(summary_path.read_text())
        steps, scores, support = _history(summary)
        if steps[-1] != EXPECTED_FINAL_STEP:
            raise RuntimeError(f"branch final step mismatch: {branch_dir.name}")
        finite_scores = all(math.isfinite(score) for score in scores)
        if not finite_scores:
            numerical_failures += 1
        finite_late = [
            score
            for step, score in zip(steps, scores, strict=True)
            if step >= LATE_WINDOW_START and math.isfinite(score)
        ]
        if not finite_late:
            raise RuntimeError(f"branch has no finite late-window score: {branch_dir.name}")
        lower_max = max(support["lower_log_std_boundary_fraction"])
        upper_max = max(support["upper_log_std_boundary_fraction"])
        support_event = lower_max > 0.0 or upper_max > 0.0
        support_events += int(support_event)
        best_index = max(range(len(scores)), key=scores.__getitem__)
        coefficient = _coefficient(control)
        row: dict[str, Any] = {
            "branch_id": branch["branch_id"],
            "dataset": branch["dataset_id"],
            "seed": seed,
            "advantage_estimator": estimator,
            "actor_update_mode": actor_mode,
            "control": _control_label(coefficient),
            "exp_coefficient": coefficient,
            "score_at_500k": _score_at(steps, scores, INTERMEDIATE_STEP),
            "best_score": scores[best_index],
            "best_step": steps[best_index],
            "final_score": scores[-1],
            "best_to_final_drop": scores[best_index] - scores[-1],
            "late_window_mean_800k_1m": _mean(finite_late),
            "late_window_std_800k_1m": _sample_std(finite_late),
            "late_slope_per_100k": _slope_per_100k(steps, scores, LATE_WINDOW_START),
            "task_performance_collapse_event": "not_adjudicated_no_registered_threshold",
            "support_or_variance_boundary_event": support_event,
            "lower_log_std_boundary_fraction_max": lower_max,
            "upper_log_std_boundary_fraction_max": upper_max,
            "terminal_actor_log_std_min": support["actor_log_std_min"][-1],
            "terminal_actor_log_std_max": support["actor_log_std_max"][-1],
            "nan_inf_numerical_failure": not finite_scores,
            "critic_frozen": bool(summary.get("critic_frozen")),
        }
        geometry = _aggregate_geometry(branch_dir / "geometry_diagnostics.jsonl")
        row.update({f"geometry_{key}": value for key, value in geometry.items()})
        if actor_mode == "ppo_clip_k4":
            ppo = _aggregate_ppo(branch_dir / "ppo_diagnostics.jsonl")
            row.update({f"ppo_{key}": value for key, value in ppo.items()})
        rows.append(row)

    grouped: dict[tuple[str, str, str, float | None], list[dict[str, Any]]] = defaultdict(list)
    paired: dict[tuple[str, int, str, float | None, str], dict[str, Any]] = {}
    for row in rows:
        group_key = (
            row["dataset"],
            row["advantage_estimator"],
            row["actor_update_mode"],
            row["exp_coefficient"],
        )
        grouped[group_key].append(row)
        paired[
            (
                row["dataset"],
                int(row["seed"]),
                row["actor_update_mode"],
                row["exp_coefficient"],
                row["advantage_estimator"],
            )
        ] = row

    groups: list[dict[str, Any]] = []
    for (dataset, estimator, actor_mode, coefficient), values in sorted(
        grouped.items(),
        key=lambda item: (
            item[0][0],
            item[0][1],
            item[0][2],
            -1.0 if item[0][3] is None else item[0][3],
        ),
    ):
        seeds = tuple(sorted(int(row["seed"]) for row in values))
        groups.append(
            {
                "dataset": dataset,
                "advantage_estimator": estimator,
                "actor_update_mode": actor_mode,
                "control": _control_label(coefficient),
                "exp_coefficient": coefficient,
                "seeds_observed": list(seeds),
                "paired_seed_set_complete": seeds == EXPECTED_SEEDS,
                "score_at_500k_mean": _mean([float(row["score_at_500k"]) for row in values]),
                "best_mean": _mean([float(row["best_score"]) for row in values]),
                "final_mean": _mean([float(row["final_score"]) for row in values]),
                "final_seed_std": _sample_std([float(row["final_score"]) for row in values]),
                "late_window_mean_800k_1m": _mean(
                    [float(row["late_window_mean_800k_1m"]) for row in values]
                ),
                "late_window_seed_std": _sample_std(
                    [float(row["late_window_mean_800k_1m"]) for row in values]
                ),
                "support_or_variance_boundary_events": sum(
                    bool(row["support_or_variance_boundary_event"]) for row in values
                ),
                "nan_inf_numerical_failures": sum(
                    bool(row["nan_inf_numerical_failure"]) for row in values
                ),
            }
        )

    comparisons: list[dict[str, Any]] = []
    datasets = sorted({row["dataset"] for row in rows})
    coefficients = sorted(
        {row["exp_coefficient"] for row in rows},
        key=lambda value: -1.0 if value is None else value,
    )
    for dataset in datasets:
        for actor_mode in ACTOR_MODES:
            for coefficient in coefficients:
                differences: list[tuple[float, float, float]] = []
                observed: list[int] = []
                for seed in EXPECTED_SEEDS:
                    td = paired.get((dataset, seed, actor_mode, coefficient, "td"))
                    gae = paired.get((dataset, seed, actor_mode, coefficient, "gae"))
                    if td is None or gae is None:
                        continue
                    observed.append(seed)
                    differences.append(
                        (
                            float(gae["late_window_mean_800k_1m"])
                            - float(td["late_window_mean_800k_1m"]),
                            float(gae["final_score"]) - float(td["final_score"]),
                            float(gae["score_at_500k"]) - float(td["score_at_500k"]),
                        )
                    )
                comparisons.append(
                    {
                        "dataset": dataset,
                        "actor_update_mode": actor_mode,
                        "control": _control_label(coefficient),
                        "exp_coefficient": coefficient,
                        "paired_seeds_observed": observed,
                        "paired_seed_set_complete": tuple(observed) == EXPECTED_SEEDS,
                        "gae_minus_td_late_mean": (
                            statistics.fmean(item[0] for item in differences)
                            if differences
                            else None
                        ),
                        "gae_minus_td_final_mean": (
                            statistics.fmean(item[1] for item in differences)
                            if differences
                            else None
                        ),
                        "gae_minus_td_500k_mean": (
                            statistics.fmean(item[2] for item in differences)
                            if differences
                            else None
                        ),
                        "gae_late_wins": sum(item[0] > 0.0 for item in differences),
                        "paired_cells": len(differences),
                        "failed_cell_imputation": False,
                    }
                )

    aggregate_dir = work / "aggregate"
    _write_csv(aggregate_dir / "branch_results.csv", rows)
    _write_csv(aggregate_dir / "group_summary.csv", groups)
    _write_csv(aggregate_dir / "gae_vs_td.csv", comparisons)
    _write_csv(aggregate_dir / "failed_branches.csv", failures)

    complete = (
        len(branch_dirs) == EXPECTED_BRANCHES
        and len(rows) == EXPECTED_BRANCHES
        and not failures
        and numerical_failures == 0
        and all(group["paired_seed_set_complete"] for group in groups)
        and all(item["paired_seed_set_complete"] for item in comparisons)
    )
    status = "PASS" if complete else "FAIL"
    audit = {
        "status": status,
        "experiment_id": EXPERIMENT_ID,
        "raw_complete": len(rows) == EXPECTED_BRANCHES and not failures,
        "branch_directories_observed": len(branch_dirs),
        "branches_completed": len(rows),
        "expected_branch_count": EXPECTED_BRANCHES,
        "failed_branches": len(failures),
        "support_or_variance_boundary_events": support_events,
        "nan_inf_numerical_failures": numerical_failures,
        "failed_cell_imputation": False,
        "task_performance_collapse_separate": True,
        "support_or_variance_boundary_separate": True,
        "nan_inf_separate": True,
        "held_out_seeds_touched": False,
        "fixed_1m_is_convergence": False,
        "steady_state_ranking_allowed": False,
        "universal_gae_claim_allowed": False,
        "universal_actor_claim_allowed": False,
        "formal_evidence_allowed": False,
    }
    _atomic_json(aggregate_dir / "terminal_audit.json", audit)
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "status": status,
        "branch_count": len(rows),
        "failure_count": len(failures),
        "group_count": len(groups),
        "comparison_count": len(comparisons),
        "terminal_audit": str(aggregate_dir / "terminal_audit.json"),
        "formal_result": False,
    }
    _atomic_json(aggregate_dir / "aggregate_summary.json", summary)
    if status != "PASS":
        raise RuntimeError("GAE pilot terminal audit failed; partial evidence preserved")
    return summary
