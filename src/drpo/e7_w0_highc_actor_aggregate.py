"""Aggregate the fixed 84-branch E7 high-c A2C/PPO screening pilot."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

EXPECTED_BRANCHES = 84
EXPECTED_FINAL_STEP = 500_000
LATE_WINDOW_START = 400_000


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
    return steps, scores


def _late_slope_per_100k(steps: list[int], scores: list[float]) -> float | None:
    pairs = [
        (float(step), float(score))
        for step, score in zip(steps, scores, strict=True)
        if step >= LATE_WINDOW_START and math.isfinite(score)
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


def _aggregate_geometry(path: Path) -> dict[str, Any]:
    records = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    if not records:
        raise RuntimeError(f"empty geometry diagnostics: {path}")
    final = records[-1]
    if final.get("status") != "complete":
        raise RuntimeError(f"geometry diagnostics not complete: {path}")
    if int(final.get("update", -1)) != EXPECTED_FINAL_STEP:
        raise RuntimeError(f"geometry diagnostics final update mismatch: {path}")
    for record in records:
        if "negative_control" in record:
            raise RuntimeError(f"legacy negative control leaked into geometry diagnostics: {path}")
        if "weight_control" not in record:
            raise RuntimeError(f"geometry diagnostics missing public weight control: {path}")
    negative_samples = sum(int(record.get("negative_samples", 0)) for record in records)
    distance_sum = sum(
        float(record["negative_distance_mean"]) * int(record["negative_samples"])
        for record in records
        if record.get("negative_distance_mean") is not None
    )
    weight_sum = sum(
        float(record["negative_weight_mean"]) * int(record["negative_samples"])
        for record in records
        if record.get("negative_weight_mean") is not None
    )
    abs_advantage = sum(float(record["negative_abs_advantage_sum"]) for record in records)
    weighted_abs_advantage = sum(
        float(record["weighted_negative_abs_advantage_sum"])
        for record in records
    )
    payload: dict[str, Any] = {
        "records": len(records),
        "negative_samples": negative_samples,
        "negative_distance_mean": distance_sum / negative_samples if negative_samples else None,
        "negative_weight_mean": weight_sum / negative_samples if negative_samples else None,
        "effective_negative_mass_fraction": (
            weighted_abs_advantage / abs_advantage if abs_advantage > 0.0 else None
        ),
    }
    for threshold in ("0p5", "0p1", "0p05", "0p01"):
        key = f"negative_weight_gt_{threshold}_fraction"
        numerator = sum(
            float(record[key]) * int(record["negative_samples"])
            for record in records
            if record.get(key) is not None
        )
        payload[key] = numerator / negative_samples if negative_samples else None
    for prefix in ("negative_distance", "negative_weight"):
        for suffix in ("p10", "p25", "p50", "p75", "p90", "p99"):
            payload[f"terminal_{prefix}_{suffix}"] = final.get(f"{prefix}_{suffix}")
    return payload


def _aggregate_ppo(path: Path) -> dict[str, Any]:
    records = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    if not records:
        raise RuntimeError(f"empty PPO diagnostics: {path}")
    final = records[-1]
    if final.get("status") != "complete":
        raise RuntimeError(f"PPO diagnostics not complete: {path}")
    if int(final.get("update", -1)) != EXPECTED_FINAL_STEP:
        raise RuntimeError(f"PPO diagnostics final update mismatch: {path}")
    total_samples = 0
    outside = 0.0
    objective_clip = 0.0
    negative_samples = 0
    negative_clip = 0.0
    for record in records:
        if "negative_control" in record or "weight_control" not in record:
            raise RuntimeError(f"PPO diagnostics violate direct-w(0) contract: {path}")
        pre = record["pre_update"]
        samples = int(pre["samples"])
        total_samples += samples
        outside += float(pre["ratio_outside_fraction"]) * samples
        objective_clip += float(pre["objective_clip_fraction"]) * samples
        negative = int(pre["negative_samples"])
        negative_samples += negative
        fraction = pre.get("negative_objective_clip_fraction")
        if fraction is not None:
            negative_clip += float(fraction) * negative
    return {
        "records": len(records),
        "ratio_outside_fraction": outside / total_samples,
        "objective_clip_fraction": objective_clip / total_samples,
        "negative_objective_clip_fraction": (
            negative_clip / negative_samples if negative_samples else None
        ),
        "old_policy_refresh_count": int(final["old_policy_refresh_count"]),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(work_dir: str | Path) -> dict[str, Any]:
    work = Path(work_dir).expanduser().resolve()
    branch_dirs = sorted(path for path in (work / "branches").iterdir() if path.is_dir())
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
        if "negative_control" in branch:
            raise RuntimeError(f"legacy negative control leaked into branch: {branch_dir.name}")
        control = branch["weight_control"]
        values = branch["template_values"]
        actor_update_mode = str(values["actor_update_mode"])
        w0 = float(control["weight_at_zero"])
        coefficient = (
            None if control["method"] == "positive_only" else float(control["exp_coefficient"])
        )
        summary_path = _only(
            (branch_dir / "trainer_output").glob("*_summary.json"),
            "trainer summary",
        )
        summary = json.loads(summary_path.read_text())
        steps, scores = _read_history(summary)
        if steps[-1] != EXPECTED_FINAL_STEP:
            raise RuntimeError(f"branch final step mismatch: {branch_dir.name}: {steps[-1]}")
        finite_scores = all(math.isfinite(score) for score in scores)
        if not finite_scores:
            numerical_failures += 1
        best_index = max(range(len(scores)), key=scores.__getitem__)
        late_scores = [
            score
            for step, score in zip(steps, scores, strict=True)
            if step >= LATE_WINDOW_START and math.isfinite(score)
        ]
        row: dict[str, Any] = {
            "branch_id": branch["branch_id"],
            "dataset": branch["dataset_id"],
            "seed": int(branch["seed"]),
            "actor_update_mode": actor_update_mode,
            "weight_at_zero": w0,
            "exp_coefficient": coefficient,
            "control": "positive_only" if coefficient is None else f"w0={w0:g},c={coefficient:g}",
            "best_score": scores[best_index],
            "best_step": steps[best_index],
            "final_score": scores[-1],
            "best_to_final_drop": scores[best_index] - scores[-1],
            "late_window_mean_400k_500k": _mean(late_scores),
            "late_window_std_400k_500k": _sample_std(late_scores),
            "late_slope_per_100k": _late_slope_per_100k(steps, scores),
            "finite_task_scores": finite_scores,
            "task_performance_collapse_event": "not_adjudicated_no_registered_threshold",
            "support_or_variance_boundary_event": "not_instrumented_in_this_pilot",
            "nan_inf_numerical_failure": not finite_scores,
        }
        geometry = _aggregate_geometry(branch_dir / "geometry_diagnostics.jsonl")
        row.update({f"geometry_{key}": value for key, value in geometry.items()})
        if actor_update_mode == "ppo_clip":
            ppo = _aggregate_ppo(branch_dir / "ppo_diagnostics.jsonl")
            row.update({f"ppo_{key}": value for key, value in ppo.items()})
        else:
            row.update(
                {
                    "ppo_records": None,
                    "ppo_ratio_outside_fraction": None,
                    "ppo_objective_clip_fraction": None,
                    "ppo_negative_objective_clip_fraction": None,
                    "ppo_old_policy_refresh_count": None,
                }
            )
        rows.append(row)

    grouped: dict[tuple[str, str, float, float | None], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["dataset"],
                row["actor_update_mode"],
                float(row["weight_at_zero"]),
                row["exp_coefficient"],
            )
        ].append(row)
    groups: list[dict[str, Any]] = []
    for (dataset, actor_mode, w0, coefficient), values in sorted(
        grouped.items(),
        key=lambda item: (
            item[0][0],
            item[0][1],
            item[0][2],
            -1.0 if item[0][3] is None else item[0][3],
        ),
    ):
        if len(values) != 2:
            raise RuntimeError(
                f"expected two seeds for {dataset},{actor_mode},w0={w0},c={coefficient}"
            )
        groups.append(
            {
                "dataset": dataset,
                "actor_update_mode": actor_mode,
                "weight_at_zero": w0,
                "exp_coefficient": coefficient,
                "seeds": sorted(int(row["seed"]) for row in values),
                "best_mean": _mean([float(row["best_score"]) for row in values]),
                "final_mean": _mean([float(row["final_score"]) for row in values]),
                "final_seed_std": _sample_std([float(row["final_score"]) for row in values]),
                "best_to_final_drop_mean": _mean(
                    [float(row["best_to_final_drop"]) for row in values]
                ),
                "late_window_mean_400k_500k": _mean(
                    [float(row["late_window_mean_400k_500k"]) for row in values]
                ),
                "late_slope_per_100k_mean": _mean(
                    [float(row["late_slope_per_100k"]) for row in values]
                ),
                "geometry_effective_negative_mass_fraction_mean": _mean(
                    [float(row["geometry_effective_negative_mass_fraction"]) for row in values]
                ),
                "geometry_negative_distance_mean": _mean(
                    [float(row["geometry_negative_distance_mean"]) for row in values]
                ),
                "geometry_negative_weight_mean": _mean(
                    [float(row["geometry_negative_weight_mean"]) for row in values]
                ),
                "nan_inf_numerical_failures": sum(
                    bool(row["nan_inf_numerical_failure"]) for row in values
                ),
            }
        )

    by_pair: dict[tuple[str, float, float | None], dict[str, dict[str, Any]]] = defaultdict(dict)
    for group in groups:
        by_pair[
            (
                group["dataset"],
                float(group["weight_at_zero"]),
                group["exp_coefficient"],
            )
        ][group["actor_update_mode"]] = group
    actor_pairs: list[dict[str, Any]] = []
    for (dataset, w0, coefficient), modes in sorted(
        by_pair.items(),
        key=lambda item: (item[0][0], item[0][1], -1.0 if item[0][2] is None else item[0][2]),
    ):
        if set(modes) != {"a2c", "ppo_clip"}:
            raise RuntimeError(f"missing paired actor mode for {dataset},w0={w0},c={coefficient}")
        a2c = modes["a2c"]
        ppo = modes["ppo_clip"]
        actor_pairs.append(
            {
                "dataset": dataset,
                "weight_at_zero": w0,
                "exp_coefficient": coefficient,
                "ppo_minus_a2c_late_mean": (
                    float(ppo["late_window_mean_400k_500k"])
                    - float(a2c["late_window_mean_400k_500k"])
                ),
                "ppo_minus_a2c_final_mean": float(ppo["final_mean"]) - float(a2c["final_mean"]),
                "a2c_late_mean": a2c["late_window_mean_400k_500k"],
                "ppo_late_mean": ppo["late_window_mean_400k_500k"],
                "a2c_effective_negative_mass": a2c[
                    "geometry_effective_negative_mass_fraction_mean"
                ],
                "ppo_effective_negative_mass": ppo[
                    "geometry_effective_negative_mass_fraction_mean"
                ],
            }
        )

    output = work / "aggregate"
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "per_branch_summary.csv", rows)
    _write_csv(output / "grid_summary.csv", groups)
    _write_csv(output / "actor_pair_summary.csv", actor_pairs)
    summary = {
        "schema_version": 1,
        "experiment_id": "EXT-H-E7-W0-HIGHC-ACTOR-01",
        "scientific_status": "w0_highc_actor_update_ablation_pilot_only",
        "branch_count": len(rows),
        "group_count": len(groups),
        "actor_pair_count": len(actor_pairs),
        "datasets": sorted({row["dataset"] for row in rows}),
        "development_seeds": sorted({int(row["seed"]) for row in rows}),
        "held_out_seeds_touched": False,
        "final_step": EXPECTED_FINAL_STEP,
        "numerical_failures": numerical_failures,
        "task_performance_collapse_reporting": "raw trajectories and drops only; no unregistered threshold applied",
        "support_or_variance_boundary_reporting": "not instrumented; reported separately as unavailable",
        "nan_inf_reporting": "counted independently from task performance",
        "fixed_500k_is_convergence": False,
        "formal_evidence_allowed": False,
        "groups": groups,
        "actor_pairs": actor_pairs,
    }
    terminal_audit = {
        "status": "PASS" if numerical_failures == 0 else "PASS_WITH_NUMERICAL_FAILURES_REPORTED",
        "experiment_id": summary["experiment_id"],
        "raw_complete": len(rows) == EXPECTED_BRANCHES,
        "branch_count": len(rows),
        "expected_branch_count": EXPECTED_BRANCHES,
        "numerical_failures": numerical_failures,
        "task_performance_collapse_separate": True,
        "support_or_variance_boundary_separate": True,
        "nan_inf_separate": True,
        "convergence_claim_allowed": False,
        "method_ranking_claim_allowed": False,
        "actor_update_causal_claim_allowed": False,
        "held_out_seeds_touched": False,
    }
    _atomic_json(output / "aggregate_summary.json", summary)
    _atomic_json(output / "terminal_audit.json", terminal_audit)
    return summary
