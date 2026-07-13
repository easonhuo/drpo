"""Aggregate the fixed 186-branch E7 PPO direct-w(0) screening pilot."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

EXPECTED_BRANCHES = 186
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


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


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
    x_mean = _mean([pair[0] for pair in pairs])
    y_mean = _mean([pair[1] for pair in pairs])
    denominator = sum((x - x_mean) ** 2 for x, _ in pairs)
    if denominator <= 0.0:
        return None
    slope_per_step = sum((x - x_mean) * (y - y_mean) for x, y in pairs) / denominator
    return slope_per_step * 100_000.0


def _aggregate_ppo_jsonl(path: Path) -> dict[str, Any]:
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
    if "negative_control" in final or "weight_control" not in final:
        raise RuntimeError(f"diagnostics violate direct-w(0) public contract: {path}")

    total_samples = 0
    ratio_outside_count = 0.0
    objective_clip_count = 0.0
    negative_samples = 0
    negative_clip_count = 0.0
    gradient_norms: list[float] = []
    update_norms: list[float] = []
    for record in records:
        if "negative_control" in record:
            raise RuntimeError(f"legacy scale leaked into diagnostics: {path}")
        pre = record["pre_update"]
        samples = int(pre["samples"])
        total_samples += samples
        ratio_outside_count += float(pre["ratio_outside_fraction"]) * samples
        objective_clip_count += float(pre["objective_clip_fraction"]) * samples
        negative = int(pre["negative_samples"])
        negative_samples += negative
        fraction = pre.get("negative_objective_clip_fraction")
        if fraction is not None:
            negative_clip_count += float(fraction) * negative
        for field, destination in (
            ("actor_gradient_norm", gradient_norms),
            ("actor_parameter_update_norm", update_norms),
        ):
            value = record.get(field)
            if _finite(value):
                destination.append(float(value))
    if total_samples <= 0:
        raise RuntimeError(f"PPO diagnostics contain no samples: {path}")
    return {
        "diagnostic_records": len(records),
        "ratio_outside_fraction": ratio_outside_count / total_samples,
        "objective_clip_fraction": objective_clip_count / total_samples,
        "negative_objective_clip_fraction": (
            negative_clip_count / negative_samples if negative_samples else None
        ),
        "actor_gradient_norm_mean_sampled": (
            _mean(gradient_norms) if gradient_norms else None
        ),
        "actor_parameter_update_norm_mean_sampled": (
            _mean(update_norms) if update_norms else None
        ),
        "old_policy_refresh_count": int(final["old_policy_refresh_count"]),
    }


def aggregate(work_dir: str | Path) -> dict[str, Any]:
    work = Path(work_dir).expanduser().resolve()
    branches_root = work / "branches"
    branch_dirs = sorted(path for path in branches_root.iterdir() if path.is_dir())
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
            raise RuntimeError(f"legacy scale leaked into branch config: {branch_dir.name}")
        control = branch["weight_control"]
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
            raise RuntimeError(
                f"branch final step mismatch: {branch_dir.name}: {steps[-1]}"
            )
        finite_scores = all(math.isfinite(value) for value in scores)
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
            "weight_at_zero": w0,
            "exp_coefficient": coefficient,
            "control": (
                "positive_only" if coefficient is None else f"w0={w0:g},c={coefficient:g}"
            ),
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
        diagnostics = _aggregate_ppo_jsonl(branch_dir / "ppo_diagnostics.jsonl")
        row.update({f"ppo_{key}": value for key, value in diagnostics.items()})
        rows.append(row)

    groups_by_key: dict[tuple[str, float, float | None], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups_by_key[
            (row["dataset"], float(row["weight_at_zero"]), row["exp_coefficient"])
        ].append(row)
    groups: list[dict[str, Any]] = []
    for (dataset, w0, coefficient), values in sorted(
        groups_by_key.items(),
        key=lambda item: (item[0][0], item[0][1], -1.0 if item[0][2] is None else item[0][2]),
    ):
        if len(values) != 2:
            raise RuntimeError(
                f"expected two development seeds for {dataset}, w0={w0}, c={coefficient}"
            )
        finals = [float(row["final_score"]) for row in values]
        bests = [float(row["best_score"]) for row in values]
        drops = [float(row["best_to_final_drop"]) for row in values]
        late_means = [float(row["late_window_mean_400k_500k"]) for row in values]
        slopes = [
            float(row["late_slope_per_100k"])
            for row in values
            if row["late_slope_per_100k"] is not None
        ]
        negative_clips = [
            float(row["ppo_negative_objective_clip_fraction"])
            for row in values
            if row["ppo_negative_objective_clip_fraction"] is not None
        ]
        groups.append(
            {
                "dataset": dataset,
                "weight_at_zero": w0,
                "exp_coefficient": coefficient,
                "seeds": sorted(int(row["seed"]) for row in values),
                "best_mean": _mean(bests),
                "final_mean": _mean(finals),
                "final_seed_std": _sample_std(finals),
                "best_to_final_drop_mean": _mean(drops),
                "late_window_mean_400k_500k": _mean(late_means),
                "late_slope_per_100k_mean": _mean(slopes) if slopes else None,
                "ppo_objective_clip_fraction_mean": _mean(
                    [float(row["ppo_objective_clip_fraction"]) for row in values]
                ),
                "ppo_negative_objective_clip_fraction_mean": (
                    _mean(negative_clips) if negative_clips else None
                ),
                "nan_inf_numerical_failures": sum(
                    bool(row["nan_inf_numerical_failure"]) for row in values
                ),
            }
        )

    output = work / "aggregate"
    output.mkdir(parents=True, exist_ok=True)
    for path, values in (
        (output / "per_branch_summary.csv", rows),
        (output / "grid_summary.csv", groups),
    ):
        fieldnames = sorted({key for row in values for key in row})
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(values)

    summary = {
        "schema_version": 1,
        "experiment_id": "EXT-H-E7-PPO-W0-EXP-GRID-01",
        "scientific_status": "ppo_w0_exp_grid_screening_pilot_only",
        "branch_count": len(rows),
        "group_count": len(groups),
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
        "held_out_seeds_touched": False,
    }
    _atomic_json(output / "aggregate_summary.json", summary)
    _atomic_json(output / "terminal_audit.json", terminal_audit)
    return summary
