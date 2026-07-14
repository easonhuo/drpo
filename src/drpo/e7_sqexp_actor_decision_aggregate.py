"""Aggregate and audit the 192-branch E7 actor/high-c decision pilot."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from drpo import e7_squared_exp_night_aggregate as common


EXPERIMENT_ID = "EXT-H-E7-SQEXP-ACTOR-DECISION-01"
EXPECTED_BRANCHES = 192
EXPECTED_FINAL_STEP = 1_000_000
INTERMEDIATE_STEP = 500_000
LATE_WINDOW_START = 800_000
EXPECTED_SEEDS = (200, 201, 202, 203)
EXPECTED_DATASETS = (
    "hopper-medium-expert-v2",
    "walker2d-medium-v2",
    "walker2d-medium-replay-v2",
)
EXPECTED_ACTORS = ("a2c", "ppo_clip_kl_k4")
EXPECTED_CONTROLS = (
    "positive_only",
    "linear_c12",
    "squared_c4",
    "squared_c8",
    "squared_c16",
    "squared_c32",
    "squared_c64",
    "squared_c128",
)
EXPECTED_CELLS = 24


def _atomic_json(path: Path, payload: Any) -> None:
    common._atomic_json(path, payload)  # noqa: SLF001


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _only(paths: Iterable[Path], label: str) -> Path:
    return common._only(paths, label)  # noqa: SLF001


def _control_fields(control: dict[str, Any]) -> dict[str, Any]:
    return {
        "control": str(control["id"]),
        "weight_family": str(control["family"]),
        "weight_at_zero": float(control["weight_at_zero"]),
        "exp_coefficient": float(control["exp_coefficient"]),
        "formula": str(control["formula"]),
    }


def _failure_row(branch_dir: Path, branch: dict[str, Any]) -> dict[str, Any]:
    manifest_path = branch_dir / "branch_manifest.json"
    manifest = _read_json(manifest_path) if manifest_path.is_file() else {}
    return {
        "branch_id": str(branch.get("branch_id", branch_dir.name)),
        "dataset": str(branch.get("dataset_id")),
        "seed": int(branch.get("seed", -1)),
        "actor_update_mode": str(branch.get("actor_update", {}).get("id")),
        **_control_fields(branch.get("weight_control", {})),
        "branch_status": "failed",
        "error_type": manifest.get("error_type"),
        "error": manifest.get("error"),
        "nan_inf_numerical_failure": manifest.get("error_type")
        in {"FloatingPointError", "OverflowError"},
        "task_performance_collapse_event": "not_adjudicated_no_registered_threshold",
        "support_or_variance_boundary_event": "not_instrumented_in_this_pilot",
    }


def _complete_row(branch_dir: Path, branch: dict[str, Any]) -> dict[str, Any]:
    summary_path = _only(
        (branch_dir / "trainer_output").glob("*_summary.json"),
        "trainer summary",
    )
    summary = _read_json(summary_path)
    steps, scores = common._read_history(summary)  # noqa: SLF001
    if steps[-1] != EXPECTED_FINAL_STEP:
        raise RuntimeError(f"final step mismatch: {branch_dir.name}")
    if not all(math.isfinite(score) for score in scores):
        raise FloatingPointError(f"non-finite task score: {branch_dir.name}")
    best_index = max(range(len(scores)), key=scores.__getitem__)
    late_scores = [
        score
        for step, score in zip(steps, scores, strict=True)
        if step >= LATE_WINDOW_START
    ]
    actor_id = str(branch["actor_update"]["id"])
    row: dict[str, Any] = {
        "branch_id": str(branch["branch_id"]),
        "dataset": str(branch["dataset_id"]),
        "seed": int(branch["seed"]),
        "actor_update_mode": actor_id,
        **_control_fields(branch["weight_control"]),
        "branch_status": "complete",
        "score_at_500k": common._score_at(steps, scores, INTERMEDIATE_STEP),  # noqa: SLF001
        "best_score": scores[best_index],
        "best_step": steps[best_index],
        "final_score": scores[-1],
        "best_to_final_drop": scores[best_index] - scores[-1],
        "late_window_mean_800k_1m": common._mean(late_scores),  # noqa: SLF001
        "late_window_temporal_std": common._sample_std(late_scores),  # noqa: SLF001
        "late_slope_per_100k": common._slope_per_100k(  # noqa: SLF001
            steps, scores, LATE_WINDOW_START
        ),
        "nan_inf_numerical_failure": False,
        "task_performance_collapse_event": "not_adjudicated_no_registered_threshold",
        "support_or_variance_boundary_event": "not_instrumented_in_this_pilot",
    }
    geometry = common._aggregate_geometry(  # noqa: SLF001
        branch_dir / "geometry_diagnostics.jsonl"
    )
    row.update({f"geometry_{key}": value for key, value in geometry.items()})
    if actor_id == "ppo_clip_kl_k4":
        ppo = common._aggregate_ppo(branch_dir / "ppo_diagnostics.jsonl")  # noqa: SLF001
        row.update({f"ppo_{key}": value for key, value in ppo.items()})
        kl_latest = _read_json(branch_dir / "PPO_KL_DIAGNOSTICS_LATEST.json")
        row.update(
            {
                "kl_target": float(kl_latest["target_kl"]),
                "kl_analytic_kl_current_batch": float(
                    kl_latest["analytic_kl_current_batch"]
                ),
                "kl_triggered_refresh_count": int(
                    kl_latest["kl_triggered_refresh_count"]
                ),
                "kl_old_policy_refresh_count": int(
                    kl_latest["old_policy_refresh_count"]
                ),
                "kl_updates_since_old_policy_refresh": int(
                    kl_latest["updates_since_old_policy_refresh"]
                ),
                "kl_max_updates_per_old_policy": int(
                    kl_latest["max_updates_per_old_policy"]
                ),
            }
        )
    return row


def _group_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    complete = [row for row in rows if row["branch_status"] == "complete"]
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in complete:
        grouped[
            (
                str(row["dataset"]),
                str(row["actor_update_mode"]),
                str(row["control"]),
            )
        ].append(row)
    groups: list[dict[str, Any]] = []
    for (dataset, actor, control), values in sorted(grouped.items()):
        seeds = tuple(sorted(int(row["seed"]) for row in values))
        late = [float(row["late_window_mean_800k_1m"]) for row in values]
        final = [float(row["final_score"]) for row in values]
        drops = [float(row["best_to_final_drop"]) for row in values]
        slopes = [float(row["late_slope_per_100k"]) for row in values]
        group = {
            "dataset": dataset,
            "actor_update_mode": actor,
            "control": control,
            "weight_family": values[0]["weight_family"],
            "exp_coefficient": values[0]["exp_coefficient"],
            "complete_seeds": list(seeds),
            "complete_seed_count": len(seeds),
            "full_four_seed_group": seeds == EXPECTED_SEEDS,
            "score_at_500k_mean": statistics.fmean(
                float(row["score_at_500k"]) for row in values
            ),
            "late_window_mean_800k_1m": statistics.fmean(late),
            "late_window_seed_sd": statistics.stdev(late) if len(late) > 1 else None,
            "final_mean": statistics.fmean(final),
            "final_seed_sd": statistics.stdev(final) if len(final) > 1 else None,
            "best_mean": statistics.fmean(float(row["best_score"]) for row in values),
            "best_to_final_drop_mean": statistics.fmean(drops),
            "late_slope_per_100k_mean": statistics.fmean(slopes),
            "absolute_late_slope_mean": statistics.fmean(abs(value) for value in slopes),
            "late_temporal_std_mean": statistics.fmean(
                float(row["late_window_temporal_std"]) for row in values
            ),
            "effective_negative_mass_fraction_mean": statistics.fmean(
                float(row["geometry_effective_negative_mass_fraction"])
                for row in values
            ),
        }
        if actor == "ppo_clip_kl_k4":
            group.update(
                {
                    "ppo_ratio_outside_fraction_mean": statistics.fmean(
                        float(row["ppo_ratio_outside_fraction"]) for row in values
                    ),
                    "ppo_objective_clip_fraction_mean": statistics.fmean(
                        float(row["ppo_objective_clip_fraction"]) for row in values
                    ),
                    "kl_triggered_refresh_count_mean": statistics.fmean(
                        float(row["kl_triggered_refresh_count"]) for row in values
                    ),
                    "kl_old_policy_refresh_count_mean": statistics.fmean(
                        float(row["kl_old_policy_refresh_count"]) for row in values
                    ),
                }
            )
        groups.append(group)
    return groups


def _comparison_rows(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index = {
        (str(group["dataset"]), str(group["actor_update_mode"]), str(group["control"])): group
        for group in groups
        if group["full_four_seed_group"]
    }
    rows: list[dict[str, Any]] = []
    for dataset in EXPECTED_DATASETS:
        for control in EXPECTED_CONTROLS:
            a2c = index.get((dataset, "a2c", control))
            ppo = index.get((dataset, "ppo_clip_kl_k4", control))
            if a2c is None or ppo is None:
                continue
            a2c_sd = float(a2c["late_window_seed_sd"])
            ppo_sd = float(ppo["late_window_seed_sd"])
            ratio = 0.0 if a2c_sd == 0.0 and ppo_sd == 0.0 else (
                math.inf if a2c_sd == 0.0 else ppo_sd / a2c_sd
            )
            rows.append(
                {
                    "dataset": dataset,
                    "control": control,
                    "weight_family": a2c["weight_family"],
                    "exp_coefficient": a2c["exp_coefficient"],
                    "ppo_minus_a2c_late": float(
                        ppo["late_window_mean_800k_1m"]
                    )
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
                    "ppo_minus_a2c_temporal_std": float(
                        ppo["late_temporal_std_mean"]
                    )
                    - float(a2c["late_temporal_std_mean"]),
                    "effective_negative_mass_fraction_a2c": a2c[
                        "effective_negative_mass_fraction_mean"
                    ],
                    "effective_negative_mass_fraction_ppo": ppo[
                        "effective_negative_mass_fraction_mean"
                    ],
                }
            )
    return rows


def _decision_gate(comparisons: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    finite_ratios = [
        float(row["ppo_over_a2c_seed_sd_ratio"])
        for row in comparisons
        if math.isfinite(float(row["ppo_over_a2c_seed_sd_ratio"]))
    ]
    lower_sd_cells = sum(bool(row["ppo_has_lower_seed_sd"]) for row in comparisons)
    pooled_late = (
        statistics.fmean(float(row["ppo_minus_a2c_late"]) for row in comparisons)
        if comparisons
        else None
    )
    median_ratio = statistics.median(finite_ratios) if finite_ratios else None
    median_drop = (
        statistics.median(
            float(row["ppo_minus_a2c_best_to_final_drop"])
            for row in comparisons
        )
        if comparisons
        else None
    )
    median_slope = (
        statistics.median(
            float(row["ppo_minus_a2c_absolute_late_slope"])
            for row in comparisons
        )
        if comparisons
        else None
    )
    dataset_differences = {
        dataset: statistics.fmean(
            float(row["ppo_minus_a2c_late"])
            for row in comparisons
            if row["dataset"] == dataset
        )
        for dataset in EXPECTED_DATASETS
        if any(row["dataset"] == dataset for row in comparisons)
    }
    failures = {
        actor: sum(
            row["branch_status"] != "complete"
            and bool(row.get("nan_inf_numerical_failure"))
            for row in rows
            if row.get("actor_update_mode") == actor
        )
        for actor in EXPECTED_ACTORS
    }
    conditions = {
        "all_24_cells_complete": len(comparisons) == EXPECTED_CELLS,
        "lower_seed_sd_in_at_least_16_cells": lower_sd_cells >= 16,
        "median_seed_sd_ratio_at_most_0p8": median_ratio is not None and median_ratio <= 0.8,
        "pooled_late_mean_not_worse_than_minus_3": pooled_late is not None and pooled_late >= -3.0,
        "lower_median_best_to_final_drop": median_drop is not None and median_drop < 0.0,
        "lower_median_absolute_late_slope": median_slope is not None and median_slope < 0.0,
        "no_more_nan_inf_failures_than_a2c": failures["ppo_clip_kl_k4"] <= failures["a2c"],
        "no_dataset_mean_worse_than_minus_3": len(dataset_differences) == 3 and min(dataset_differences.values()) >= -3.0,
    }
    recommendation = all(conditions.values())
    return {
        "experiment_id": EXPERIMENT_ID,
        "status": "PASS" if recommendation else "FAIL",
        "decision": (
            "retain_ppo_for_e7_mainline"
            if recommendation
            else "do_not_retain_ppo_for_e7_mainline_under_this_backbone"
        ),
        "scientific_scope": "Current canonical E7 offline backbone and one-step TD only.",
        "paired_cells_observed": len(comparisons),
        "paired_cells_expected": EXPECTED_CELLS,
        "cells_with_lower_ppo_seed_sd": lower_sd_cells,
        "median_ppo_over_a2c_seed_sd_ratio": median_ratio,
        "pooled_ppo_minus_a2c_late_mean": pooled_late,
        "median_ppo_minus_a2c_best_to_final_drop": median_drop,
        "median_ppo_minus_a2c_absolute_late_slope": median_slope,
        "dataset_mean_late_differences": dataset_differences,
        "nan_inf_numerical_failures_by_actor": failures,
        "conditions": conditions,
        "automatic_merge_or_method_deletion_authorized": False,
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
        branch = _read_json(config_path)
        if branch.get("experiment_id") != EXPERIMENT_ID:
            raise RuntimeError(f"experiment mismatch: {branch_dir.name}")
        if (branch_dir / "COMPLETED.json").is_file():
            try:
                rows.append(_complete_row(branch_dir, branch))
            except BaseException as exc:
                failure = _failure_row(branch_dir, branch)
                failure["error_type"] = type(exc).__name__
                failure["error"] = str(exc)
                failure["nan_inf_numerical_failure"] = isinstance(
                    exc, (FloatingPointError, OverflowError)
                )
                rows.append(failure)
        else:
            rows.append(_failure_row(branch_dir, branch))

    groups = _group_rows(rows)
    comparisons = _comparison_rows(groups)
    gate = _decision_gate(comparisons, rows)
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
        else "FAIL"
    )

    aggregate_dir = work / "aggregate"
    _write_csv(aggregate_dir / "branch_results.csv", rows)
    _write_csv(aggregate_dir / "group_summary.csv", groups)
    _write_csv(aggregate_dir / "actor_comparisons.csv", comparisons)
    _atomic_json(aggregate_dir / "ppo_retention_gate.json", gate)
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
        "held_out_seeds_touched": any(
            int(row.get("seed", -1)) in {204, 205, 206, 207} for row in rows
        ),
        "task_performance_collapse_separate": True,
        "support_or_variance_boundary_separate": True,
        "nan_inf_separate": True,
        "convergence_claim_allowed": False,
        "steady_state_ranking_allowed": False,
        "controlled_causal_actor_update_claim_allowed": False,
        "gae_claim_allowed": False,
        "formal_evidence_allowed": False,
        "ppo_retention_gate_status": gate["status"],
    }
    _atomic_json(aggregate_dir / "terminal_audit.json", audit)
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "status": terminal_status,
        "branch_count": len(rows),
        "completed": complete_count,
        "failed": failed_count,
        "missing": missing_count,
        "group_count": len(groups),
        "comparison_count": len(comparisons),
        "ppo_retention_gate": gate,
        "files": {
            "branch_results": str(aggregate_dir / "branch_results.csv"),
            "group_summary": str(aggregate_dir / "group_summary.csv"),
            "actor_comparisons": str(aggregate_dir / "actor_comparisons.csv"),
            "ppo_retention_gate": str(aggregate_dir / "ppo_retention_gate.json"),
            "terminal_audit": str(aggregate_dir / "terminal_audit.json"),
        },
    }
    _atomic_json(aggregate_dir / "aggregate_summary.json", summary)
    return summary
