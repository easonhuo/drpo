"""Aggregate the fixed 150-branch Stage A KL-threshold tuning suite."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from drpo import e7_squared_exp_night_aggregate as common


EXPERIMENT_ID = "EXT-H-E7-SQUARED-EXP-KL-TUNE-01"
STAGE_ID = "stage_a_kl_threshold_and_reference_lifecycle_screen"
EXPECTED_BRANCHES = 150
EXPECTED_FINAL_STEP = 1_000_000
INTERMEDIATE_STEP = 500_000
LATE_WINDOW_START = 800_000
EXPECTED_SEEDS = (200, 201)
EXPECTED_DATASETS = (
    "hopper-medium-expert-v2",
    "walker2d-medium-v2",
    "walker2d-medium-replay-v2",
)
EXPECTED_COEFFICIENTS = (None, 4.0, 8.0, 16.0, 32.0)
BASELINE_LIFECYCLE = "ppo_clip_k4"
FIXED_K16_LIFECYCLE = "ppo_clip_k16"
ADAPTIVE_LIFECYCLES = (
    "ppo_clip_kl_k16_t0p003",
    "ppo_clip_kl_k16_t0p01",
    "ppo_clip_kl_k16_t0p03",
)
EXPECTED_LIFECYCLES = (
    BASELINE_LIFECYCLE,
    FIXED_K16_LIFECYCLE,
    *ADAPTIVE_LIFECYCLES,
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _lifecycle_target(lifecycle_id: str) -> float | None:
    targets = {
        "ppo_clip_kl_k16_t0p003": 0.003,
        "ppo_clip_kl_k16_t0p01": 0.01,
        "ppo_clip_kl_k16_t0p03": 0.03,
    }
    return targets.get(lifecycle_id)


def _control_name(coefficient: float | None) -> str:
    return "positive_only" if coefficient is None else f"c={coefficient:g}"


def _only(paths: Iterable[Path], label: str) -> Path:
    return common._only(paths, label)  # noqa: SLF001


def _branch_row(branch_dir: Path) -> dict[str, Any]:
    if not (branch_dir / "COMPLETED.json").is_file():
        raise RuntimeError(f"branch is not complete: {branch_dir.name}")
    branch = _read_json(branch_dir / "branch_config.json")
    if branch.get("experiment_id") != EXPERIMENT_ID:
        raise RuntimeError(f"branch experiment mismatch: {branch_dir.name}")
    if branch.get("stage_id") != STAGE_ID:
        raise RuntimeError(f"branch stage mismatch: {branch_dir.name}")
    dataset = str(branch["dataset_id"])
    seed = int(branch["seed"])
    if dataset not in EXPECTED_DATASETS:
        raise RuntimeError(f"unexpected dataset in branch: {branch_dir.name}")
    if seed not in EXPECTED_SEEDS:
        raise RuntimeError(f"forbidden seed in branch: {branch_dir.name}")

    control = branch["weight_control"]
    if control.get("formula") != "w(d)=w(0)*exp(-c*(d/2)^2)":
        raise RuntimeError(f"branch is not squared-remoteness EXP: {branch_dir.name}")
    coefficient = (
        None
        if control["method"] == "positive_only"
        else float(control["exp_coefficient"])
    )
    if coefficient not in EXPECTED_COEFFICIENTS:
        raise RuntimeError(f"unexpected coefficient in branch: {branch_dir.name}")

    lifecycle = branch["reference_lifecycle"]
    lifecycle_id = str(lifecycle["id"])
    if lifecycle_id not in EXPECTED_LIFECYCLES:
        raise RuntimeError(f"unexpected lifecycle in branch: {branch_dir.name}")
    target_kl = lifecycle.get("target_kl")
    expected_target = _lifecycle_target(lifecycle_id)
    if expected_target is None:
        if target_kl is not None:
            raise RuntimeError(f"fixed lifecycle has target_kl: {branch_dir.name}")
    elif target_kl is None or not math.isclose(
        float(target_kl), expected_target, abs_tol=1e-12
    ):
        raise RuntimeError(f"adaptive lifecycle target mismatch: {branch_dir.name}")

    summary_path = _only(
        (branch_dir / "trainer_output").glob("*_summary.json"),
        "trainer summary",
    )
    summary = _read_json(summary_path)
    steps, scores = common._read_history(summary)  # noqa: SLF001
    if steps[-1] != EXPECTED_FINAL_STEP:
        raise RuntimeError(f"branch final step mismatch: {branch_dir.name}")
    finite_scores = all(math.isfinite(score) for score in scores)
    best_index = max(range(len(scores)), key=scores.__getitem__)
    late_scores = [
        score
        for step, score in zip(steps, scores, strict=True)
        if step >= LATE_WINDOW_START and math.isfinite(score)
    ]

    row: dict[str, Any] = {
        "branch_id": branch["branch_id"],
        "dataset": dataset,
        "seed": seed,
        "stage": STAGE_ID,
        "reference_lifecycle": lifecycle_id,
        "target_kl": expected_target,
        "control": _control_name(coefficient),
        "weight_at_zero": float(control["weight_at_zero"]),
        "exp_coefficient": coefficient,
        "score_at_500k": common._score_at(  # noqa: SLF001
            steps, scores, INTERMEDIATE_STEP
        ),
        "best_score": scores[best_index],
        "best_step": steps[best_index],
        "final_score": scores[-1],
        "best_to_final_drop": scores[best_index] - scores[-1],
        "late_window_mean_800k_1m": common._mean(late_scores),  # noqa: SLF001
        "late_window_std_800k_1m": common._sample_std(late_scores),  # noqa: SLF001
        "late_slope_per_100k": common._slope_per_100k(  # noqa: SLF001
            steps, scores, LATE_WINDOW_START
        ),
        "finite_task_scores": finite_scores,
        "task_performance_collapse_event": (
            "not_adjudicated_no_registered_threshold"
        ),
        "support_or_variance_boundary_event": "not_instrumented_in_this_pilot",
        "nan_inf_numerical_failure": not finite_scores,
    }
    geometry = common._aggregate_geometry(  # noqa: SLF001
        branch_dir / "geometry_diagnostics.jsonl"
    )
    row.update({f"geometry_{key}": value for key, value in geometry.items()})
    ppo = common._aggregate_ppo(branch_dir / "ppo_diagnostics.jsonl")  # noqa: SLF001
    row.update({f"ppo_{key}": value for key, value in ppo.items()})
    if lifecycle_id in ADAPTIVE_LIFECYCLES:
        kl = common._aggregate_kl(  # noqa: SLF001
            branch_dir / "ppo_kl_diagnostics.jsonl"
        )
        if not math.isclose(float(kl["target_kl"]), float(expected_target), abs_tol=1e-12):
            raise RuntimeError(f"aggregated target_kl mismatch: {branch_dir.name}")
        row.update({f"kl_{key}": value for key, value in kl.items()})
    return row


def _group_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, float | None], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["dataset"]),
            str(row["reference_lifecycle"]),
            row["exp_coefficient"],
        )
        grouped[key].append(row)

    groups: list[dict[str, Any]] = []
    for (dataset, lifecycle_id, coefficient), values in sorted(
        grouped.items(),
        key=lambda item: (
            item[0][0],
            item[0][1],
            -1.0 if item[0][2] is None else float(item[0][2]),
        ),
    ):
        seeds = tuple(sorted(int(row["seed"]) for row in values))
        if seeds != EXPECTED_SEEDS:
            raise RuntimeError(
                f"expected paired seeds for {dataset},{lifecycle_id},"
                f"c={coefficient}; got {seeds}"
            )
        groups.append(
            {
                "dataset": dataset,
                "reference_lifecycle": lifecycle_id,
                "target_kl": _lifecycle_target(lifecycle_id),
                "exp_coefficient": coefficient,
                "control": _control_name(coefficient),
                "seeds": list(seeds),
                "score_at_500k_mean": common._mean(  # noqa: SLF001
                    [float(row["score_at_500k"]) for row in values]
                ),
                "best_mean": common._mean(  # noqa: SLF001
                    [float(row["best_score"]) for row in values]
                ),
                "final_mean": common._mean(  # noqa: SLF001
                    [float(row["final_score"]) for row in values]
                ),
                "final_seed_std": common._sample_std(  # noqa: SLF001
                    [float(row["final_score"]) for row in values]
                ),
                "late_window_mean_800k_1m": common._mean(  # noqa: SLF001
                    [float(row["late_window_mean_800k_1m"]) for row in values]
                ),
                "late_window_seed_std": common._sample_std(  # noqa: SLF001
                    [float(row["late_window_mean_800k_1m"]) for row in values]
                ),
                "late_slope_per_100k_mean": common._mean(  # noqa: SLF001
                    [float(row["late_slope_per_100k"]) for row in values]
                ),
                "effective_negative_mass_fraction_mean": common._mean(  # noqa: SLF001
                    [
                        float(row["geometry_effective_negative_mass_fraction"])
                        for row in values
                    ]
                ),
                "ppo_ratio_outside_fraction_mean": common._mean(  # noqa: SLF001
                    [float(row["ppo_ratio_outside_fraction"]) for row in values]
                ),
                "ppo_objective_clip_fraction_mean": common._mean(  # noqa: SLF001
                    [float(row["ppo_objective_clip_fraction"]) for row in values]
                ),
                "kl_analytic_kl_mean": (
                    common._mean(  # noqa: SLF001
                        [float(row["kl_analytic_kl_mean"]) for row in values]
                    )
                    if lifecycle_id in ADAPTIVE_LIFECYCLES
                    else None
                ),
                "kl_triggered_refresh_count_mean": (
                    common._mean(  # noqa: SLF001
                        [
                            float(row["kl_kl_triggered_refresh_count"])
                            for row in values
                        ]
                    )
                    if lifecycle_id in ADAPTIVE_LIFECYCLES
                    else None
                ),
                "nan_inf_numerical_failures": sum(
                    bool(row["nan_inf_numerical_failure"]) for row in values
                ),
            }
        )
    expected_groups = len(EXPECTED_DATASETS) * len(EXPECTED_LIFECYCLES) * len(
        EXPECTED_COEFFICIENTS
    )
    if len(groups) != expected_groups:
        raise RuntimeError(f"expected {expected_groups} groups, found {len(groups)}")
    return groups


def _comparison_rows(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index = {
        (
            str(group["dataset"]),
            str(group["reference_lifecycle"]),
            group["exp_coefficient"],
        ): group
        for group in groups
    }
    comparisons: list[dict[str, Any]] = []
    for dataset in EXPECTED_DATASETS:
        for coefficient in EXPECTED_COEFFICIENTS:
            baseline = index[(dataset, BASELINE_LIFECYCLE, coefficient)]
            baseline_late = float(baseline["late_window_mean_800k_1m"])
            for lifecycle_id in EXPECTED_LIFECYCLES[1:]:
                candidate = index[(dataset, lifecycle_id, coefficient)]
                comparisons.append(
                    {
                        "dataset": dataset,
                        "exp_coefficient": coefficient,
                        "control": _control_name(coefficient),
                        "baseline_lifecycle": BASELINE_LIFECYCLE,
                        "candidate_lifecycle": lifecycle_id,
                        "target_kl": _lifecycle_target(lifecycle_id),
                        "candidate_minus_k4_late": (
                            float(candidate["late_window_mean_800k_1m"])
                            - baseline_late
                        ),
                        "candidate_late_seed_std": candidate[
                            "late_window_seed_std"
                        ],
                        "k4_late_seed_std": baseline["late_window_seed_std"],
                    }
                )
    return comparisons


def _qualification(comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for lifecycle_id in ADAPTIVE_LIFECYCLES:
        selected = [
            row
            for row in comparisons
            if row["candidate_lifecycle"] == lifecycle_id
        ]
        expected_cells = len(EXPECTED_DATASETS) * len(EXPECTED_COEFFICIENTS)
        if len(selected) != expected_cells:
            raise RuntimeError(
                f"expected {expected_cells} cells for {lifecycle_id}, "
                f"found {len(selected)}"
            )
        differences = [float(row["candidate_minus_k4_late"]) for row in selected]
        positive_only = [
            float(row["candidate_minus_k4_late"])
            for row in selected
            if row["exp_coefficient"] is None
        ]
        squared_exp = [
            float(row["candidate_minus_k4_late"])
            for row in selected
            if row["exp_coefficient"] is not None
        ]
        mean_difference = statistics.fmean(differences)
        median_difference = statistics.median(differences)
        wins = sum(value > 0.0 for value in differences)
        qualifies = (
            mean_difference > 0.0
            and median_difference > 0.0
            and wins > len(differences) / 2.0
        )
        candidates.append(
            {
                "reference_lifecycle": lifecycle_id,
                "target_kl": _lifecycle_target(lifecycle_id),
                "paired_cells": len(differences),
                "mean_late_difference_vs_k4": mean_difference,
                "median_late_difference_vs_k4": median_difference,
                "paired_cell_wins_vs_k4": wins,
                "positive_only_mean_difference_vs_k4": statistics.fmean(
                    positive_only
                ),
                "squared_exp_mean_difference_vs_k4": statistics.fmean(
                    squared_exp
                ),
                "qualifies": qualifies,
            }
        )
    qualified = [candidate for candidate in candidates if candidate["qualifies"]]
    qualified.sort(
        key=lambda item: (
            float(item["mean_late_difference_vs_k4"]),
            float(item["median_late_difference_vs_k4"]),
            int(item["paired_cell_wins_vs_k4"]),
        ),
        reverse=True,
    )
    selected = qualified[0]["reference_lifecycle"] if qualified else None
    return {
        "status": "QUALIFIED" if selected is not None else "NO_COMMON_THRESHOLD_QUALIFIED",
        "experiment_id": EXPERIMENT_ID,
        "stage_id": STAGE_ID,
        "baseline_lifecycle": BASELINE_LIFECYCLE,
        "selection_rule": {
            "mean_late_difference_positive": True,
            "median_late_difference_positive": True,
            "paired_cell_wins_more_than_half": True,
            "tie_break_order": [
                "mean_late_difference",
                "median_late_difference",
                "paired_cell_wins",
            ],
        },
        "candidates": candidates,
        "selected_reference_lifecycle": selected,
        "selected_target_kl": (
            _lifecycle_target(str(selected)) if selected is not None else None
        ),
        "stage_b_launch_authorized": False,
        "stage_b_requires_registration_and_explicit_launch": True,
    }


def aggregate(work_dir: str | Path) -> dict[str, Any]:
    work = Path(work_dir).expanduser().resolve()
    branch_root = work / "branches"
    if not branch_root.is_dir():
        raise RuntimeError("Stage A branch directory is missing")
    branch_dirs = sorted(path for path in branch_root.iterdir() if path.is_dir())
    if len(branch_dirs) != EXPECTED_BRANCHES:
        raise RuntimeError(
            f"expected {EXPECTED_BRANCHES} branch directories, found {len(branch_dirs)}"
        )

    rows = [_branch_row(branch_dir) for branch_dir in branch_dirs]
    numerical_failures = sum(
        bool(row["nan_inf_numerical_failure"]) for row in rows
    )
    groups = _group_rows(rows)
    comparisons = _comparison_rows(groups)
    qualification = _qualification(comparisons)

    aggregate_dir = work / "aggregate"
    _write_csv(aggregate_dir / "branch_results.csv", rows)
    _write_csv(aggregate_dir / "group_summary.csv", groups)
    _write_csv(aggregate_dir / "lifecycle_comparisons.csv", comparisons)
    common._atomic_json(  # noqa: SLF001
        aggregate_dir / "stage_a_qualification.json", qualification
    )

    terminal_status = "PASS" if numerical_failures == 0 else "FAIL"
    audit = {
        "status": terminal_status,
        "experiment_id": EXPERIMENT_ID,
        "stage_id": STAGE_ID,
        "raw_complete": True,
        "branch_count_observed": len(rows),
        "expected_branch_count": EXPECTED_BRANCHES,
        "reference_lifecycles": list(EXPECTED_LIFECYCLES),
        "controls_per_lifecycle": len(EXPECTED_COEFFICIENTS),
        "nan_inf_numerical_failures": numerical_failures,
        "task_performance_collapse_separate": True,
        "support_or_variance_boundary_separate": True,
        "nan_inf_separate": True,
        "held_out_seeds_touched": False,
        "convergence_claim_allowed": False,
        "steady_state_ranking_allowed": False,
        "causal_actor_update_claim_allowed": False,
        "gae_claim_allowed": False,
        "formal_evidence_allowed": False,
        "stage_b_launch_authorized": False,
    }
    common._atomic_json(aggregate_dir / "terminal_audit.json", audit)  # noqa: SLF001
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "stage_id": STAGE_ID,
        "status": terminal_status,
        "branch_count": len(rows),
        "group_count": len(groups),
        "comparison_count": len(comparisons),
        "qualification": qualification,
        "files": {
            "branch_results": str(aggregate_dir / "branch_results.csv"),
            "group_summary": str(aggregate_dir / "group_summary.csv"),
            "lifecycle_comparisons": str(
                aggregate_dir / "lifecycle_comparisons.csv"
            ),
            "stage_a_qualification": str(
                aggregate_dir / "stage_a_qualification.json"
            ),
            "terminal_audit": str(aggregate_dir / "terminal_audit.json"),
        },
    }
    common._atomic_json(  # noqa: SLF001
        aggregate_dir / "aggregate_summary.json", summary
    )
    if terminal_status != "PASS":
        raise RuntimeError("Stage A terminal audit failed")
    return summary
