"""Aggregate the fixed 126-branch E7 squared-remoteness night suite."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


EXPERIMENT_ID = "EXT-H-E7-SQUARED-EXP-NIGHT-01"
EXPECTED_BRANCHES = 126
EXPECTED_FINAL_STEP = 1_000_000
INTERMEDIATE_STEP = 500_000
LATE_WINDOW_START = 800_000
EXPECTED_SEEDS = (200, 201)
ACTOR_MODES = ("a2c", "ppo_clip_k4", "ppo_clip_kl_k16")


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _only(paths: Iterable[Path], label: str) -> Path:
    values = list(paths)
    if len(values) != 1:
        raise RuntimeError(f"expected exactly one {label}, found {len(values)}")
    return values[0]


def _mean(values: list[float]) -> float:
    if not values:
        raise RuntimeError("cannot average an empty list")
    return statistics.fmean(values)


def _sample_std(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) >= 2 else None


def _read_history(summary: Mapping[str, Any]) -> tuple[list[int], list[float]]:
    history = summary.get("history")
    if not isinstance(history, dict):
        raise RuntimeError("trainer summary has no history mapping")
    steps = [int(value) for value in history.get("steps", [])]
    score_keys = [key for key in history if key != "steps"]
    if len(score_keys) != 1:
        raise RuntimeError("trainer history must contain exactly one score series")
    scores = [float(value) for value in history[score_keys[0]]]
    if not steps or len(steps) != len(scores):
        raise RuntimeError("trainer history steps/scores length mismatch")
    if len(set(steps)) != len(steps) or steps != sorted(steps):
        raise RuntimeError("trainer history steps must be unique and sorted")
    return steps, scores


def _score_at(steps: list[int], scores: list[float], target: int) -> float:
    try:
        index = steps.index(target)
    except ValueError as exc:
        raise RuntimeError(f"history is missing required step {target}") from exc
    return scores[index]


def _slope_per_100k(steps: list[int], scores: list[float], start: int) -> float | None:
    pairs = [
        (float(step), float(score))
        for step, score in zip(steps, scores, strict=True)
        if step >= start and math.isfinite(score)
    ]
    if len(pairs) < 2:
        return None
    x_mean = _mean([x for x, _ in pairs])
    y_mean = _mean([y for _, y in pairs])
    denominator = sum((x - x_mean) ** 2 for x, _ in pairs)
    if denominator <= 0.0:
        return None
    slope = sum((x - x_mean) * (y - y_mean) for x, y in pairs) / denominator
    return slope * 100_000.0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    if not records:
        raise RuntimeError(f"empty diagnostics file: {path}")
    return records


def _aggregate_geometry(path: Path) -> dict[str, Any]:
    records = _read_jsonl(path)
    final = records[-1]
    if final.get("status") != "complete" or int(final.get("update", -1)) != EXPECTED_FINAL_STEP:
        raise RuntimeError(f"geometry diagnostics did not complete at 1M: {path}")
    negative_samples = 0
    distance_sum = 0.0
    weight_sum = 0.0
    abs_advantage = 0.0
    weighted_abs_advantage = 0.0
    for record in records:
        if "negative_control" in record or "weight_control" not in record:
            raise RuntimeError(f"geometry diagnostics violate public control contract: {path}")
        count = int(record.get("negative_samples", 0))
        negative_samples += count
        if record.get("negative_distance_mean") is not None:
            distance_sum += float(record["negative_distance_mean"]) * count
        if record.get("negative_weight_mean") is not None:
            weight_sum += float(record["negative_weight_mean"]) * count
        abs_advantage += float(record.get("negative_abs_advantage_sum", 0.0))
        weighted_abs_advantage += float(
            record.get("weighted_negative_abs_advantage_sum", 0.0)
        )
    return {
        "records": len(records),
        "negative_samples": negative_samples,
        "negative_distance_mean": (
            distance_sum / negative_samples if negative_samples else None
        ),
        "negative_weight_mean": (
            weight_sum / negative_samples if negative_samples else None
        ),
        "effective_negative_mass_fraction": (
            weighted_abs_advantage / abs_advantage if abs_advantage > 0.0 else None
        ),
        "terminal_negative_distance_p50": final.get("negative_distance_p50"),
        "terminal_negative_distance_p90": final.get("negative_distance_p90"),
        "terminal_negative_distance_p99": final.get("negative_distance_p99"),
        "terminal_negative_weight_p50": final.get("negative_weight_p50"),
        "terminal_negative_weight_p90": final.get("negative_weight_p90"),
        "terminal_negative_weight_p99": final.get("negative_weight_p99"),
    }


def _aggregate_ppo(path: Path) -> dict[str, Any]:
    records = _read_jsonl(path)
    final = records[-1]
    if final.get("status") != "complete" or int(final.get("update", -1)) != EXPECTED_FINAL_STEP:
        raise RuntimeError(f"PPO diagnostics did not complete at 1M: {path}")
    samples = 0
    outside = 0.0
    clipped = 0.0
    positive_samples = 0
    positive_clipped = 0.0
    negative_samples = 0
    negative_clipped = 0.0
    for record in records:
        if "negative_control" in record or "weight_control" not in record:
            raise RuntimeError(f"PPO diagnostics violate public control contract: {path}")
        pre = record["pre_update"]
        count = int(pre["samples"])
        samples += count
        outside += float(pre["ratio_outside_fraction"]) * count
        clipped += float(pre["objective_clip_fraction"]) * count
        pos = int(pre["positive_samples"])
        neg = int(pre["negative_samples"])
        positive_samples += pos
        negative_samples += neg
        if pre.get("positive_objective_clip_fraction") is not None:
            positive_clipped += float(pre["positive_objective_clip_fraction"]) * pos
        if pre.get("negative_objective_clip_fraction") is not None:
            negative_clipped += float(pre["negative_objective_clip_fraction"]) * neg
    return {
        "records": len(records),
        "ratio_outside_fraction": outside / samples,
        "objective_clip_fraction": clipped / samples,
        "positive_objective_clip_fraction": (
            positive_clipped / positive_samples if positive_samples else None
        ),
        "negative_objective_clip_fraction": (
            negative_clipped / negative_samples if negative_samples else None
        ),
        "old_policy_refresh_count": int(final["old_policy_refresh_count"]),
        "terminal_abs_log_ratio_max": float(final["pre_update"]["abs_log_ratio_max"]),
        "terminal_ratio_max": float(final["pre_update"]["ratio_max"]),
    }


def _aggregate_kl(path: Path) -> dict[str, Any]:
    records = _read_jsonl(path)
    final = records[-1]
    if final.get("status") != "complete" or int(final.get("update", -1)) != EXPECTED_FINAL_STEP:
        raise RuntimeError(f"KL diagnostics did not complete at 1M: {path}")
    weighted_sum = 0.0
    updates = 0
    maximum = 0.0
    for record in records:
        interval_updates = int(record.get("interval_updates", 1))
        weighted_sum += float(record["interval_analytic_kl_mean"]) * interval_updates
        updates += interval_updates
        maximum = max(maximum, float(record["interval_analytic_kl_max"]))
    return {
        "records": len(records),
        "analytic_kl_mean": weighted_sum / updates,
        "analytic_kl_max": maximum,
        "kl_triggered_refresh_count": int(final["kl_triggered_refresh_count"]),
        "old_policy_refresh_count": int(final["old_policy_refresh_count"]),
        "target_kl": float(final["target_kl"]),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(work_dir: str | Path) -> dict[str, Any]:
    work = Path(work_dir).expanduser().resolve()
    branch_root = work / "branches"
    if not branch_root.is_dir():
        raise RuntimeError("night-suite branch directory is missing")
    branch_dirs = sorted(path for path in branch_root.iterdir() if path.is_dir())
    if len(branch_dirs) != EXPECTED_BRANCHES:
        raise RuntimeError(
            f"expected {EXPECTED_BRANCHES} branch directories, found {len(branch_dirs)}"
        )

    rows: list[dict[str, Any]] = []
    numerical_failures = 0
    for branch_dir in branch_dirs:
        if not (branch_dir / "COMPLETED.json").is_file():
            raise RuntimeError(f"branch is not complete: {branch_dir.name}")
        branch = json.loads((branch_dir / "branch_config.json").read_text())
        if branch.get("experiment_id") != EXPERIMENT_ID:
            raise RuntimeError(f"branch experiment mismatch: {branch_dir.name}")
        if int(branch["seed"]) not in EXPECTED_SEEDS:
            raise RuntimeError(f"forbidden seed in branch: {branch_dir.name}")
        control = branch["weight_control"]
        if control.get("formula") != "w(d)=w(0)*exp(-c*(d/2)^2)":
            raise RuntimeError(f"branch is not squared-remoteness EXP: {branch_dir.name}")
        values = branch["template_values"]
        actor_mode = str(values["actor_update_mode"])
        if actor_mode not in ACTOR_MODES:
            raise RuntimeError(f"unknown actor mode: {actor_mode}")

        summary_path = _only(
            (branch_dir / "trainer_output").glob("*_summary.json"),
            "trainer summary",
        )
        summary = json.loads(summary_path.read_text())
        steps, scores = _read_history(summary)
        if steps[-1] != EXPECTED_FINAL_STEP:
            raise RuntimeError(f"branch final step mismatch: {branch_dir.name}")
        finite_scores = all(math.isfinite(score) for score in scores)
        if not finite_scores:
            numerical_failures += 1
        best_index = max(range(len(scores)), key=scores.__getitem__)
        late_scores = [
            score
            for step, score in zip(steps, scores, strict=True)
            if step >= LATE_WINDOW_START and math.isfinite(score)
        ]
        coefficient = (
            None
            if control["method"] == "positive_only"
            else float(control["exp_coefficient"])
        )
        row: dict[str, Any] = {
            "branch_id": branch["branch_id"],
            "dataset": branch["dataset_id"],
            "seed": int(branch["seed"]),
            "stage": str(values["stage"]),
            "actor_update_mode": actor_mode,
            "control": (
                "positive_only" if coefficient is None else f"w0=1,c={coefficient:g}"
            ),
            "weight_at_zero": float(control["weight_at_zero"]),
            "exp_coefficient": coefficient,
            "score_at_500k": _score_at(steps, scores, INTERMEDIATE_STEP),
            "best_score": scores[best_index],
            "best_step": steps[best_index],
            "final_score": scores[-1],
            "best_to_final_drop": scores[best_index] - scores[-1],
            "late_window_mean_800k_1m": _mean(late_scores),
            "late_window_std_800k_1m": _sample_std(late_scores),
            "late_slope_per_100k": _slope_per_100k(steps, scores, LATE_WINDOW_START),
            "finite_task_scores": finite_scores,
            "task_performance_collapse_event": "not_adjudicated_no_registered_threshold",
            "support_or_variance_boundary_event": "not_instrumented_in_this_pilot",
            "nan_inf_numerical_failure": not finite_scores,
        }
        geometry = _aggregate_geometry(branch_dir / "geometry_diagnostics.jsonl")
        row.update({f"geometry_{key}": value for key, value in geometry.items()})
        if actor_mode != "a2c":
            ppo = _aggregate_ppo(branch_dir / "ppo_diagnostics.jsonl")
            row.update({f"ppo_{key}": value for key, value in ppo.items()})
        if actor_mode == "ppo_clip_kl_k16":
            kl = _aggregate_kl(branch_dir / "ppo_kl_diagnostics.jsonl")
            row.update({f"kl_{key}": value for key, value in kl.items()})
        rows.append(row)

    grouped: dict[tuple[str, str, float | None], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["actor_update_mode"], row["exp_coefficient"])].append(row)
    groups: list[dict[str, Any]] = []
    for (dataset, actor_mode, coefficient), values in sorted(
        grouped.items(),
        key=lambda item: (
            item[0][0],
            item[0][1],
            -1.0 if item[0][2] is None else item[0][2],
        ),
    ):
        seeds = tuple(sorted(int(row["seed"]) for row in values))
        if seeds != EXPECTED_SEEDS:
            raise RuntimeError(
                f"expected paired seeds for {dataset},{actor_mode},c={coefficient}; got {seeds}"
            )
        groups.append(
            {
                "dataset": dataset,
                "actor_update_mode": actor_mode,
                "exp_coefficient": coefficient,
                "control": "positive_only" if coefficient is None else f"c={coefficient:g}",
                "seeds": list(seeds),
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
                "late_slope_per_100k_mean": _mean(
                    [float(row["late_slope_per_100k"]) for row in values]
                ),
                "effective_negative_mass_fraction_mean": _mean(
                    [float(row["geometry_effective_negative_mass_fraction"]) for row in values]
                ),
                "nan_inf_numerical_failures": sum(
                    bool(row["nan_inf_numerical_failure"]) for row in values
                ),
                "ppo_objective_clip_fraction_mean": (
                    _mean([float(row["ppo_objective_clip_fraction"]) for row in values])
                    if actor_mode != "a2c"
                    else None
                ),
                "kl_analytic_kl_mean": (
                    _mean([float(row["kl_analytic_kl_mean"]) for row in values])
                    if actor_mode == "ppo_clip_kl_k16"
                    else None
                ),
                "kl_triggered_refresh_count_mean": (
                    _mean([float(row["kl_kl_triggered_refresh_count"]) for row in values])
                    if actor_mode == "ppo_clip_kl_k16"
                    else None
                ),
            }
        )

    group_index = {
        (group["dataset"], group["actor_update_mode"], group["exp_coefficient"]): group
        for group in groups
    }
    comparisons: list[dict[str, Any]] = []
    for dataset in sorted({row["dataset"] for row in rows}):
        coefficients = sorted(
            {row["exp_coefficient"] for row in rows if row["dataset"] == dataset},
            key=lambda value: -1.0 if value is None else value,
        )
        for coefficient in coefficients:
            a2c = group_index[(dataset, "a2c", coefficient)]
            ppo4 = group_index[(dataset, "ppo_clip_k4", coefficient)]
            ppo16 = group_index[(dataset, "ppo_clip_kl_k16", coefficient)]
            comparisons.append(
                {
                    "dataset": dataset,
                    "exp_coefficient": coefficient,
                    "control": "positive_only" if coefficient is None else f"c={coefficient:g}",
                    "ppo_k4_minus_a2c_late": (
                        float(ppo4["late_window_mean_800k_1m"])
                        - float(a2c["late_window_mean_800k_1m"])
                    ),
                    "ppo_kl_k16_minus_ppo_k4_late": (
                        float(ppo16["late_window_mean_800k_1m"])
                        - float(ppo4["late_window_mean_800k_1m"])
                    ),
                    "ppo_kl_k16_minus_a2c_late": (
                        float(ppo16["late_window_mean_800k_1m"])
                        - float(a2c["late_window_mean_800k_1m"])
                    ),
                }
            )

    aggregate_dir = work / "aggregate"
    _write_csv(aggregate_dir / "branch_results.csv", rows)
    _write_csv(aggregate_dir / "group_summary.csv", groups)
    _write_csv(aggregate_dir / "actor_comparisons.csv", comparisons)
    gae_status = {
        "status": "BLOCKED",
        "stage": "stage_c_gae",
        "gae_lambda": 0.95,
        "reason": "verified ordered-trajectory and terminal/truncation contract unavailable",
        "branches_started": 0,
        "scientific_result_available": False,
    }
    _atomic_json(aggregate_dir / "GAE_STAGE_STATUS.json", gae_status)

    terminal_status = "PASS" if numerical_failures == 0 else "FAIL"
    audit = {
        "status": terminal_status,
        "experiment_id": EXPERIMENT_ID,
        "raw_complete": True,
        "branch_count_observed": len(rows),
        "expected_branch_count": EXPECTED_BRANCHES,
        "stage_a_branches": 84,
        "stage_b_branches": 42,
        "stage_c_status": "blocked_before_execution",
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
    }
    _atomic_json(aggregate_dir / "terminal_audit.json", audit)
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "status": terminal_status,
        "branch_count": len(rows),
        "group_count": len(groups),
        "comparison_count": len(comparisons),
        "stage_c": gae_status,
        "files": {
            "branch_results": str(aggregate_dir / "branch_results.csv"),
            "group_summary": str(aggregate_dir / "group_summary.csv"),
            "actor_comparisons": str(aggregate_dir / "actor_comparisons.csv"),
            "terminal_audit": str(aggregate_dir / "terminal_audit.json"),
        },
    }
    _atomic_json(aggregate_dir / "aggregate_summary.json", summary)
    if terminal_status != "PASS":
        raise RuntimeError("night-suite terminal audit failed")
    return summary
