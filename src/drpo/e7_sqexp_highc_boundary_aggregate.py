"""Aggregate and audit the 48-branch E7 squared-EXP high-c boundary pilot."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from drpo import e7_sqexp_actor_decision_aggregate as predecessor


EXPERIMENT_ID = "EXT-H-E7-SQEXP-HIGHC-BOUNDARY-01"
EXPECTED_BRANCHES = 48
EXPECTED_SEEDS = (200, 201, 202, 203)
EXPECTED_DATASETS = (
    "hopper-medium-expert-v2",
    "walker2d-medium-v2",
    "walker2d-medium-replay-v2",
)
EXPECTED_ACTORS = ("a2c", "ppo_clip_kl_k4")
EXPECTED_CONTROLS = ("squared_c256", "squared_c512")
EXPECTED_GROUPS = 12
EXPECTED_ACTOR_COMPARISONS = 6
EXPECTED_HIGHC_PAIRS = 24


def _comparison_rows(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index = {
        (str(group["dataset"]), str(group["actor_update_mode"]), str(group["control"])): group
        for group in groups
        if group["full_four_seed_group"]
    }
    comparisons: list[dict[str, Any]] = []
    for dataset in EXPECTED_DATASETS:
        for control in EXPECTED_CONTROLS:
            a2c = index.get((dataset, "a2c", control))
            ppo = index.get((dataset, "ppo_clip_kl_k4", control))
            if a2c is None or ppo is None:
                continue
            a2c_sd = float(a2c["late_window_seed_sd"])
            ppo_sd = float(ppo["late_window_seed_sd"])
            if a2c_sd == 0.0:
                ratio = 0.0 if ppo_sd == 0.0 else math.inf
            else:
                ratio = ppo_sd / a2c_sd
            comparisons.append(
                {
                    "dataset": dataset,
                    "control": control,
                    "exp_coefficient": a2c["exp_coefficient"],
                    "ppo_minus_a2c_late": float(ppo["late_window_mean_800k_1m"])
                    - float(a2c["late_window_mean_800k_1m"]),
                    "a2c_late_seed_sd": a2c_sd,
                    "ppo_late_seed_sd": ppo_sd,
                    "ppo_over_a2c_seed_sd_ratio": ratio,
                    "ppo_has_lower_seed_sd": ppo_sd < a2c_sd,
                    "ppo_minus_a2c_final": float(ppo["final_mean"])
                    - float(a2c["final_mean"]),
                    "ppo_minus_a2c_best_to_final_drop": float(
                        ppo["best_to_final_drop_mean"]
                    )
                    - float(a2c["best_to_final_drop_mean"]),
                    "ppo_minus_a2c_absolute_late_slope": float(
                        ppo["absolute_late_slope_mean"]
                    )
                    - float(a2c["absolute_late_slope_mean"]),
                    "effective_negative_mass_fraction_a2c": a2c[
                        "effective_negative_mass_fraction_mean"
                    ],
                    "effective_negative_mass_fraction_ppo": ppo[
                        "effective_negative_mass_fraction_mean"
                    ],
                }
            )
    return comparisons


def _highc_pair_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    complete = [row for row in rows if row["branch_status"] == "complete"]
    index = {
        (
            str(row["dataset"]),
            str(row["actor_update_mode"]),
            str(row["control"]),
            int(row["seed"]),
        ): row
        for row in complete
    }
    pairs: list[dict[str, Any]] = []
    for dataset in EXPECTED_DATASETS:
        for actor in EXPECTED_ACTORS:
            for seed in EXPECTED_SEEDS:
                c256 = index.get((dataset, actor, "squared_c256", seed))
                c512 = index.get((dataset, actor, "squared_c512", seed))
                if c256 is None or c512 is None:
                    continue
                pairs.append(
                    {
                        "dataset": dataset,
                        "actor_update_mode": actor,
                        "seed": seed,
                        "c256_late": float(c256["late_window_mean_800k_1m"]),
                        "c512_late": float(c512["late_window_mean_800k_1m"]),
                        "c512_minus_c256_late": float(
                            c512["late_window_mean_800k_1m"]
                        )
                        - float(c256["late_window_mean_800k_1m"]),
                        "c256_final": float(c256["final_score"]),
                        "c512_final": float(c512["final_score"]),
                        "c512_minus_c256_final": float(c512["final_score"])
                        - float(c256["final_score"]),
                        "c512_minus_c256_best_to_final_drop": float(
                            c512["best_to_final_drop"]
                        )
                        - float(c256["best_to_final_drop"]),
                        "c512_minus_c256_absolute_late_slope": abs(
                            float(c512["late_slope_per_100k"])
                        )
                        - abs(float(c256["late_slope_per_100k"])),
                        "c256_effective_negative_mass_fraction": float(
                            c256["geometry_effective_negative_mass_fraction"]
                        ),
                        "c512_effective_negative_mass_fraction": float(
                            c512["geometry_effective_negative_mass_fraction"]
                        ),
                    }
                )
    return pairs


def _pair_summary(values: list[dict[str, Any]]) -> dict[str, Any]:
    late_differences = [float(row["c512_minus_c256_late"]) for row in values]
    final_differences = [float(row["c512_minus_c256_final"]) for row in values]
    drop_differences = [
        float(row["c512_minus_c256_best_to_final_drop"]) for row in values
    ]
    slope_differences = [
        float(row["c512_minus_c256_absolute_late_slope"]) for row in values
    ]
    mass_256 = [
        float(row["c256_effective_negative_mass_fraction"]) for row in values
    ]
    mass_512 = [
        float(row["c512_effective_negative_mass_fraction"]) for row in values
    ]
    return {
        "pair_count": len(values),
        "c512_minus_c256_late_mean": statistics.fmean(late_differences),
        "c512_minus_c256_late_median": statistics.median(late_differences),
        "c512_late_wins": sum(value > 0.0 for value in late_differences),
        "c512_late_losses": sum(value < 0.0 for value in late_differences),
        "c512_late_ties": sum(value == 0.0 for value in late_differences),
        "c512_minus_c256_final_mean": statistics.fmean(final_differences),
        "c512_minus_c256_best_to_final_drop_mean": statistics.fmean(
            drop_differences
        ),
        "c512_minus_c256_absolute_late_slope_mean": statistics.fmean(
            slope_differences
        ),
        "c256_effective_negative_mass_fraction_mean": statistics.fmean(mass_256),
        "c512_effective_negative_mass_fraction_mean": statistics.fmean(mass_512),
        "c512_over_c256_effective_negative_mass_ratio": (
            statistics.fmean(mass_512) / statistics.fmean(mass_256)
            if statistics.fmean(mass_256) > 0.0
            else None
        ),
    }


def _boundary_summary(
    groups: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
) -> dict[str, Any]:
    grouped_pairs: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in pairs:
        grouped_pairs[(str(row["dataset"]), str(row["actor_update_mode"]))].append(row)
    by_dataset_actor = {
        f"{dataset}::{actor}": _pair_summary(values)
        for (dataset, actor), values in sorted(grouped_pairs.items())
    }
    by_actor = {
        actor: _pair_summary(
            [row for row in pairs if row["actor_update_mode"] == actor]
        )
        for actor in EXPECTED_ACTORS
        if any(row["actor_update_mode"] == actor for row in pairs)
    }
    overall = _pair_summary(pairs) if pairs else None
    finite_sd_ratios = [
        float(row["ppo_over_a2c_seed_sd_ratio"])
        for row in comparisons
        if math.isfinite(float(row["ppo_over_a2c_seed_sd_ratio"]))
    ]
    actor_descriptive = {
        "paired_cells": len(comparisons),
        "pooled_ppo_minus_a2c_late_mean": (
            statistics.fmean(float(row["ppo_minus_a2c_late"]) for row in comparisons)
            if comparisons
            else None
        ),
        "cells_with_lower_ppo_seed_sd": sum(
            bool(row["ppo_has_lower_seed_sd"]) for row in comparisons
        ),
        "median_ppo_over_a2c_seed_sd_ratio": (
            statistics.median(finite_sd_ratios) if finite_sd_ratios else None
        ),
        "selection_authorized": False,
        "note": "This focused boundary pilot does not reopen or replace the predecessor PPO retention gate.",
    }
    return {
        "experiment_id": EXPERIMENT_ID,
        "status": "descriptive_only",
        "primary_comparison": "squared_c512_minus_squared_c256",
        "group_count": len(groups),
        "highc_pair_count": len(pairs),
        "highc_pair_count_expected": EXPECTED_HIGHC_PAIRS,
        "overall": overall,
        "by_actor": by_actor,
        "by_dataset_actor": by_dataset_actor,
        "actor_descriptive": actor_descriptive,
        "automatic_common_c_selection_authorized": False,
        "automatic_ppo_selection_authorized": False,
        "c1024_extension_authorized": False,
        "required_interpretation": (
            "Report c256 and c512 by dataset, actor, and seed; join predecessor c128 and "
            "Positive-only only as separately identified historical paired context."
        ),
    }


def aggregate(work_dir: str | Path) -> dict[str, Any]:
    work = Path(work_dir).expanduser().resolve()
    branch_root = work / "branches"
    if not branch_root.is_dir():
        raise RuntimeError("branch directory is missing")
    branch_dirs = sorted(path for path in branch_root.iterdir() if path.is_dir())
    rows: list[dict[str, Any]] = []
    for branch_dir in branch_dirs:
        config_path = branch_dir / "branch_config.json"
        if not config_path.is_file():
            continue
        branch = predecessor._read_json(config_path)  # noqa: SLF001
        if branch.get("experiment_id") != EXPERIMENT_ID:
            raise RuntimeError(f"experiment mismatch: {branch_dir.name}")
        if (branch_dir / "COMPLETED.json").is_file():
            try:
                rows.append(predecessor._complete_row(branch_dir, branch))  # noqa: SLF001
            except BaseException as exc:
                failure = predecessor._failure_row(branch_dir, branch)  # noqa: SLF001
                failure["error_type"] = type(exc).__name__
                failure["error"] = str(exc)
                failure["nan_inf_numerical_failure"] = isinstance(
                    exc, (FloatingPointError, OverflowError)
                )
                rows.append(failure)
        else:
            rows.append(predecessor._failure_row(branch_dir, branch))  # noqa: SLF001

    groups = predecessor._group_rows(rows)  # noqa: SLF001
    comparisons = _comparison_rows(groups)
    highc_pairs = _highc_pair_rows(rows)
    boundary = _boundary_summary(groups, highc_pairs, comparisons)

    complete_count = sum(row["branch_status"] == "complete" for row in rows)
    failed_count = sum(row["branch_status"] != "complete" for row in rows)
    missing_count = EXPECTED_BRANCHES - len(rows)
    nan_inf_count = sum(bool(row.get("nan_inf_numerical_failure")) for row in rows)
    terminal_status = (
        "PASS"
        if len(rows) == EXPECTED_BRANCHES
        and complete_count == EXPECTED_BRANCHES
        and failed_count == 0
        and nan_inf_count == 0
        and len(groups) == EXPECTED_GROUPS
        and len(comparisons) == EXPECTED_ACTOR_COMPARISONS
        and len(highc_pairs) == EXPECTED_HIGHC_PAIRS
        else "FAIL"
    )

    aggregate_dir = work / "aggregate"
    predecessor._write_csv(aggregate_dir / "branch_results.csv", rows)  # noqa: SLF001
    predecessor._write_csv(aggregate_dir / "group_summary.csv", groups)  # noqa: SLF001
    predecessor._write_csv(  # noqa: SLF001
        aggregate_dir / "actor_comparisons.csv", comparisons
    )
    predecessor._write_csv(aggregate_dir / "highc_pairwise.csv", highc_pairs)  # noqa: SLF001
    predecessor._atomic_json(  # noqa: SLF001
        aggregate_dir / "highc_boundary_summary.json", boundary
    )
    audit = {
        "status": terminal_status,
        "experiment_id": EXPERIMENT_ID,
        "raw_complete": terminal_status == "PASS",
        "expected_branch_count": EXPECTED_BRANCHES,
        "branch_records_observed": len(rows),
        "completed_branches": complete_count,
        "failed_branches": failed_count,
        "missing_branches": missing_count,
        "nan_inf_numerical_failures": nan_inf_count,
        "group_count": len(groups),
        "actor_comparison_count": len(comparisons),
        "highc_pair_count": len(highc_pairs),
        "held_out_seeds_touched": any(
            int(row.get("seed", -1)) in {204, 205, 206, 207} for row in rows
        ),
        "task_performance_collapse_separate": True,
        "support_or_variance_boundary_separate": True,
        "nan_inf_separate": True,
        "convergence_claim_allowed": False,
        "steady_state_ranking_allowed": False,
        "common_c_selection_allowed": False,
        "ppo_selection_allowed": False,
        "gae_claim_allowed": False,
        "formal_evidence_allowed": False,
    }
    predecessor._atomic_json(aggregate_dir / "terminal_audit.json", audit)  # noqa: SLF001
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "status": terminal_status,
        "branch_count": len(rows),
        "completed": complete_count,
        "failed": failed_count,
        "missing": missing_count,
        "group_count": len(groups),
        "actor_comparison_count": len(comparisons),
        "highc_pair_count": len(highc_pairs),
        "highc_boundary_summary": boundary,
        "files": {
            "branch_results": str(aggregate_dir / "branch_results.csv"),
            "group_summary": str(aggregate_dir / "group_summary.csv"),
            "actor_comparisons": str(aggregate_dir / "actor_comparisons.csv"),
            "highc_pairwise": str(aggregate_dir / "highc_pairwise.csv"),
            "highc_boundary_summary": str(
                aggregate_dir / "highc_boundary_summary.json"
            ),
            "terminal_audit": str(aggregate_dir / "terminal_audit.json"),
        },
    }
    predecessor._atomic_json(  # noqa: SLF001
        aggregate_dir / "aggregate_summary.json", summary
    )
    return summary
