"""Paired TD-versus-GAE summaries using the existing E7 aggregation helpers."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from drpo.e7_sqexp_gae_protocol import (
    ACTOR_MODES,
    COEFFICIENTS,
    ESTIMATORS,
    EXPECTED_BRANCHES,
    EXPECTED_DATASETS,
    EXPECTED_SEEDS,
    EXPERIMENT_ID,
)
from drpo.e7_squared_exp_night_aggregate import (
    _atomic_json,
    _mean,
    _only,
    _read_history,
    _sample_std,
    _score_at,
    _write_csv,
)

INTERMEDIATE_STEP = 500_000
LATE_WINDOW_START = 800_000
FINAL_STEP = 1_000_000
EXPECTED_PAIRS = EXPECTED_BRANCHES // 2


def _control_key(branch: dict[str, Any]) -> float | None:
    control = branch["weight_control"]
    return None if control["method"] == "positive_only" else float(control["exp_coefficient"])


def _pair_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index = {
        (
            row["dataset"],
            int(row["seed"]),
            row["actor_update_mode"],
            row["exp_coefficient"],
            row["advantage_estimator"],
        ): row
        for row in rows
    }
    paired: list[dict[str, Any]] = []
    controls: tuple[float | None, ...] = (None, *COEFFICIENTS)
    for dataset in EXPECTED_DATASETS:
        for seed in EXPECTED_SEEDS:
            for actor_mode in ACTOR_MODES:
                for coefficient in controls:
                    key = (dataset, seed, actor_mode, coefficient)
                    td = index.get((*key, "td"))
                    gae = index.get((*key, "gae"))
                    if td is None or gae is None:
                        raise RuntimeError(f"missing paired TD/GAE cell: {key}")
                    paired.append(
                        {
                            "dataset": dataset,
                            "seed": seed,
                            "actor_update_mode": actor_mode,
                            "exp_coefficient": coefficient,
                            "control": "positive_only" if coefficient is None else f"c={coefficient:g}",
                            "gae_minus_td_score_at_500k": gae["score_at_500k"] - td["score_at_500k"],
                            "gae_minus_td_late_mean": gae["late_window_mean_800k_1m"]
                            - td["late_window_mean_800k_1m"],
                            "gae_minus_td_final_score": gae["final_score"] - td["final_score"],
                            "gae_minus_td_best_score": gae["best_score"] - td["best_score"],
                        }
                    )
    if len(paired) != EXPECTED_PAIRS:
        raise RuntimeError(f"expected {EXPECTED_PAIRS} paired cells, got {len(paired)}")
    return paired


def aggregate(work_dir: str | Path) -> dict[str, Any]:
    work = Path(work_dir).expanduser().resolve()
    branch_root = work / "branches"
    branch_dirs = sorted(path for path in branch_root.iterdir() if path.is_dir())
    if len(branch_dirs) != EXPECTED_BRANCHES:
        raise RuntimeError(f"expected {EXPECTED_BRANCHES} branch directories")
    rows: list[dict[str, Any]] = []
    for branch_dir in branch_dirs:
        if not (branch_dir / "COMPLETED.json").is_file():
            raise RuntimeError(f"branch is not complete: {branch_dir.name}")
        branch = json.loads((branch_dir / "branch_config.json").read_text())
        manifest = json.loads((branch_dir / "branch_manifest.json").read_text())
        if branch.get("experiment_id") != EXPERIMENT_ID:
            raise RuntimeError(f"branch experiment mismatch: {branch_dir.name}")
        values = branch["template_values"]
        estimator = str(values["advantage_estimator"])
        actor_mode = str(values["actor_update_mode"])
        if estimator not in ESTIMATORS or actor_mode not in ACTOR_MODES:
            raise RuntimeError(f"branch matrix mismatch: {branch_dir.name}")
        if manifest.get("critic_immutability_verified") is not True:
            raise RuntimeError(f"critic audit failed: {branch_dir.name}")
        summary_path = _only(
            (branch_dir / "trainer_output").glob("*_summary.json"), "trainer summary"
        )
        steps, scores = _read_history(json.loads(summary_path.read_text()))
        if steps[-1] != FINAL_STEP or not all(math.isfinite(score) for score in scores):
            raise RuntimeError(f"incomplete or non-finite task history: {branch_dir.name}")
        late = [score for step, score in zip(steps, scores, strict=True) if step >= LATE_WINDOW_START]
        if not late:
            raise RuntimeError(f"missing late window: {branch_dir.name}")
        best = max(scores)
        coefficient = _control_key(branch)
        rows.append(
            {
                "branch_id": branch["branch_id"],
                "dataset": branch["dataset_id"],
                "seed": int(branch["seed"]),
                "advantage_estimator": estimator,
                "actor_update_mode": actor_mode,
                "exp_coefficient": coefficient,
                "control": "positive_only" if coefficient is None else f"c={coefficient:g}",
                "score_at_500k": _score_at(steps, scores, INTERMEDIATE_STEP),
                "late_window_mean_800k_1m": _mean(late),
                "final_score": scores[-1],
                "best_score": best,
            }
        )
    paired = _pair_rows(rows)
    grouped: dict[tuple[str, str, float | None], list[dict[str, Any]]] = defaultdict(list)
    for row in paired:
        grouped[(row["dataset"], row["actor_update_mode"], row["exp_coefficient"])].append(row)
    summaries: list[dict[str, Any]] = []
    for (dataset, actor_mode, coefficient), values in sorted(
        grouped.items(),
        key=lambda item: (
            item[0][0],
            item[0][1],
            -1 if item[0][2] is None else item[0][2],
        ),
    ):
        seeds = tuple(sorted(int(row["seed"]) for row in values))
        if seeds != EXPECTED_SEEDS:
            raise RuntimeError(f"incomplete paired seeds: {dataset},{actor_mode},{coefficient}")
        late_differences = [float(row["gae_minus_td_late_mean"]) for row in values]
        summaries.append(
            {
                "dataset": dataset,
                "actor_update_mode": actor_mode,
                "exp_coefficient": coefficient,
                "control": "positive_only" if coefficient is None else f"c={coefficient:g}",
                "seeds": list(seeds),
                "gae_minus_td_late_mean": _mean(late_differences),
                "gae_minus_td_late_seed_std": _sample_std(late_differences),
                "gae_minus_td_final_mean": _mean(
                    [float(row["gae_minus_td_final_score"]) for row in values]
                ),
                "gae_minus_td_best_mean": _mean(
                    [float(row["gae_minus_td_best_score"]) for row in values]
                ),
            }
        )
    aggregate_dir = work / "aggregate"
    _write_csv(aggregate_dir / "branch_results.csv", rows)
    _write_csv(aggregate_dir / "gae_vs_td.csv", paired)
    _write_csv(aggregate_dir / "gae_vs_td_summary.csv", summaries)
    payload = {
        "status": "PASS",
        "experiment_id": EXPERIMENT_ID,
        "branch_rows": len(rows),
        "paired_cells": len(paired),
        "paired_groups": len(summaries),
        "failed_cell_imputation": False,
        "fixed_1m_endpoint_is_convergence": False,
        "method_ranking_allowed": False,
    }
    _atomic_json(
        aggregate_dir / "gae_vs_td_summary.json",
        {**payload, "groups": summaries},
    )
    return payload
