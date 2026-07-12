"""Aggregate the fixed E7 PPO-stability pilot without external dependencies."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

EXPECTED_BRANCHES = 96
EXPECTED_PPO_BRANCHES = 48
EXPECTED_FINAL_STEP = 1_000_000


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


def _sample_std(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) >= 2 else None


def _mean(values: list[float]) -> float:
    if not values:
        raise RuntimeError("cannot average an empty list")
    return statistics.fmean(values)


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _control_label(control: Mapping[str, Any]) -> str:
    method = str(control["method"])
    if method == "positive_only":
        return "positive_only"
    if method == "exponential":
        coefficient = float(control["exponential_coefficient"])
        return f"exp_c{coefficient:g}"
    raise RuntimeError(f"unexpected pilot method: {method}")


def _read_history(summary: Mapping[str, Any]) -> tuple[list[int], list[float]]:
    history = summary.get("history")
    if not isinstance(history, dict):
        raise RuntimeError("trainer summary has no history mapping")
    steps = [int(value) for value in history.get("steps", [])]
    score_keys = [key for key in history if key != "steps"]
    if len(score_keys) != 1:
        raise RuntimeError("trainer history must contain exactly one score series")
    scores = [float(value) for value in history[score_keys[0]]]
    if len(steps) != len(scores) or not steps:
        raise RuntimeError("trainer history steps/scores length mismatch")
    return steps, scores


def _aggregate_ppo_jsonl(path: Path) -> dict[str, Any]:
    records = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    if not records:
        raise RuntimeError(f"empty PPO diagnostics: {path}")
    if records[-1].get("status") != "complete":
        raise RuntimeError(f"PPO diagnostics not complete: {path}")
    if int(records[-1].get("update", -1)) != EXPECTED_FINAL_STEP:
        raise RuntimeError(f"PPO diagnostics final update mismatch: {path}")

    total_samples = 0
    ratio_sum = 0.0
    abs_log_ratio_sum = 0.0
    ratio_min = float("inf")
    ratio_max = float("-inf")
    outside_count = 0.0
    objective_clip_count = 0.0
    positive_samples = 0
    positive_clip_count = 0.0
    negative_samples = 0
    negative_clip_count = 0.0
    gradient_norms: list[float] = []
    update_norms: list[float] = []
    relative_update_norms: list[float] = []
    sampled_post_outside: list[float] = []
    sampled_step_ratio_max: list[float] = []

    for record in records:
        pre = record["pre_update"]
        samples = int(pre["samples"])
        total_samples += samples
        ratio_sum += float(pre["ratio_mean"]) * samples
        abs_log_ratio_sum += float(pre["abs_log_ratio_mean"]) * samples
        ratio_min = min(ratio_min, float(pre["ratio_min"]))
        ratio_max = max(ratio_max, float(pre["ratio_max"]))
        outside_count += float(pre["ratio_outside_fraction"]) * samples
        objective_clip_count += float(pre["objective_clip_fraction"]) * samples

        positive = int(pre["positive_samples"])
        positive_samples += positive
        if pre["positive_objective_clip_fraction"] is not None:
            positive_clip_count += (
                float(pre["positive_objective_clip_fraction"]) * positive
            )
        negative = int(pre["negative_samples"])
        negative_samples += negative
        if pre["negative_objective_clip_fraction"] is not None:
            negative_clip_count += (
                float(pre["negative_objective_clip_fraction"]) * negative
            )

        for field, destination in (
            ("actor_gradient_norm", gradient_norms),
            ("actor_parameter_update_norm", update_norms),
            ("actor_relative_parameter_update_norm", relative_update_norms),
        ):
            value = record.get(field)
            if _finite(value):
                destination.append(float(value))
        post = record.get("sampled_post_update")
        if isinstance(post, dict):
            value = post.get("ratio_to_old_outside_fraction")
            if _finite(value):
                sampled_post_outside.append(float(value))
            value = post.get("single_step_ratio_max")
            if _finite(value):
                sampled_step_ratio_max.append(float(value))

    if total_samples <= 0:
        raise RuntimeError(f"PPO diagnostics contain no samples: {path}")
    return {
        "diagnostic_records": len(records),
        "total_samples": total_samples,
        "ratio_mean": ratio_sum / total_samples,
        "ratio_min": ratio_min,
        "ratio_max": ratio_max,
        "abs_log_ratio_mean": abs_log_ratio_sum / total_samples,
        "ratio_outside_fraction": outside_count / total_samples,
        "objective_clip_fraction": objective_clip_count / total_samples,
        "positive_objective_clip_fraction": (
            positive_clip_count / positive_samples if positive_samples else None
        ),
        "negative_objective_clip_fraction": (
            negative_clip_count / negative_samples if negative_samples else None
        ),
        "actor_gradient_norm_mean_sampled": (
            _mean(gradient_norms) if gradient_norms else None
        ),
        "actor_parameter_update_norm_mean_sampled": (
            _mean(update_norms) if update_norms else None
        ),
        "actor_relative_parameter_update_norm_mean_sampled": (
            _mean(relative_update_norms) if relative_update_norms else None
        ),
        "post_update_ratio_outside_fraction_mean_sampled": (
            _mean(sampled_post_outside) if sampled_post_outside else None
        ),
        "single_step_ratio_max_max_sampled": (
            max(sampled_step_ratio_max) if sampled_step_ratio_max else None
        ),
        "old_policy_refresh_count": int(records[-1]["old_policy_refresh_count"]),
    }


def aggregate(work_dir: str | Path) -> dict[str, Any]:
    work = Path(work_dir).expanduser().resolve()
    branches_root = work / "branches"
    branch_dirs = sorted(path for path in branches_root.iterdir() if path.is_dir())
    if len(branch_dirs) != EXPECTED_BRANCHES:
        raise RuntimeError(
            f"expected {EXPECTED_BRANCHES} branch directories, "
            f"found {len(branch_dirs)}"
        )

    rows: list[dict[str, Any]] = []
    numerical_failures = 0
    ppo_complete = 0
    for branch_dir in branch_dirs:
        if not (branch_dir / "COMPLETED.json").is_file():
            raise RuntimeError(f"branch is not complete: {branch_dir.name}")
        branch = json.loads((branch_dir / "branch_config.json").read_text())
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
        if not all(math.isfinite(value) for value in scores):
            numerical_failures += 1
        best_index = max(range(len(scores)), key=scores.__getitem__)
        late_scores = [
            score
            for step, score in zip(steps, scores, strict=True)
            if step >= 750_000
        ]
        actor_update_mode = str(
            branch["template_values"]["actor_update_mode"]
        )
        control_label = _control_label(branch["negative_control"])
        row: dict[str, Any] = {
            "branch_id": branch["branch_id"],
            "dataset": branch["dataset_id"],
            "seed": int(branch["seed"]),
            "control": control_label,
            "actor_update_mode": actor_update_mode,
            "best_score": scores[best_index],
            "best_step": steps[best_index],
            "final_score": scores[-1],
            "best_to_final_drop": scores[best_index] - scores[-1],
            "late_window_mean": _mean(late_scores),
            "late_window_std": _sample_std(late_scores),
        }
        if actor_update_mode == "ppo_clip":
            ppo_path = branch_dir / "ppo_diagnostics.jsonl"
            if not ppo_path.is_file():
                raise RuntimeError(f"missing PPO diagnostics: {branch_dir.name}")
            row.update(
                {
                    f"ppo_{key}": value
                    for key, value in _aggregate_ppo_jsonl(ppo_path).items()
                }
            )
            ppo_complete += 1
        rows.append(row)

    if ppo_complete != EXPECTED_PPO_BRANCHES:
        raise RuntimeError(
            f"expected {EXPECTED_PPO_BRANCHES} PPO branches, "
            f"found {ppo_complete}"
        )

    group_values: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        group_values[
            (row["dataset"], row["control"], row["actor_update_mode"])
        ].append(row)
    groups: list[dict[str, Any]] = []
    for (dataset, control, mode), values in sorted(group_values.items()):
        finals = [float(row["final_score"]) for row in values]
        bests = [float(row["best_score"]) for row in values]
        drops = [float(row["best_to_final_drop"]) for row in values]
        group = {
            "dataset": dataset,
            "control": control,
            "actor_update_mode": mode,
            "seeds": sorted(int(row["seed"]) for row in values),
            "best_mean": _mean(bests),
            "final_mean": _mean(finals),
            "final_seed_std": _sample_std(finals),
            "best_to_final_drop_mean": _mean(drops),
        }
        if mode == "ppo_clip":
            group["ppo_objective_clip_fraction_mean"] = _mean(
                [
                    float(row["ppo_objective_clip_fraction"])
                    for row in values
                ]
            )
            group["ppo_ratio_outside_fraction_mean"] = _mean(
                [float(row["ppo_ratio_outside_fraction"]) for row in values]
            )
            negative_clip = [
                row["ppo_negative_objective_clip_fraction"]
                for row in values
                if row["ppo_negative_objective_clip_fraction"] is not None
            ]
            group["ppo_negative_objective_clip_fraction_mean"] = (
                _mean([float(value) for value in negative_clip])
                if negative_clip
                else None
            )
        groups.append(group)

    keyed = {
        (
            row["dataset"],
            row["seed"],
            row["control"],
            row["actor_update_mode"],
        ): row
        for row in rows
    }
    paired: list[dict[str, Any]] = []
    for dataset in sorted({row["dataset"] for row in rows}):
        for control in sorted({row["control"] for row in rows}):
            differences_final: list[float] = []
            differences_drop: list[float] = []
            differences_best: list[float] = []
            seeds: list[int] = []
            for seed in sorted({int(row["seed"]) for row in rows}):
                a2c = keyed.get((dataset, seed, control, "a2c"))
                ppo = keyed.get((dataset, seed, control, "ppo_clip"))
                if a2c is None or ppo is None:
                    continue
                seeds.append(seed)
                differences_final.append(
                    float(ppo["final_score"]) - float(a2c["final_score"])
                )
                differences_best.append(
                    float(ppo["best_score"]) - float(a2c["best_score"])
                )
                differences_drop.append(
                    float(ppo["best_to_final_drop"])
                    - float(a2c["best_to_final_drop"])
                )
            if seeds:
                paired.append(
                    {
                        "dataset": dataset,
                        "control": control,
                        "seeds": seeds,
                        "ppo_minus_a2c_final_mean": _mean(differences_final),
                        "ppo_minus_a2c_best_mean": _mean(differences_best),
                        "ppo_minus_a2c_best_to_final_drop_mean": _mean(
                            differences_drop
                        ),
                    }
                )

    output = work / "aggregate"
    output.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with (output / "per_branch_summary.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "experiment_id": "EXT-H-E7-PPO-STABILITY-01",
        "scientific_status": "pilot",
        "branch_count": len(rows),
        "ppo_branch_count": ppo_complete,
        "groups": groups,
        "paired_ppo_vs_a2c": paired,
        "interpretation_boundary": (
            "Clip activity is mechanism evidence only. Stability support requires "
            "smaller BEST-to-FINAL drop and seed variability while retaining BEST."
        ),
    }
    _atomic_json(output / "aggregate_summary.json", summary)
    audit = {
        "status": "pass" if numerical_failures == 0 else "fail",
        "expected_branches": EXPECTED_BRANCHES,
        "completed_branches": len(rows),
        "expected_ppo_branches": EXPECTED_PPO_BRANCHES,
        "completed_ppo_diagnostics": ppo_complete,
        "nan_inf_numerical_failure_branches": numerical_failures,
        "task_performance_degradation_events": None,
        "task_performance_degradation_reason": (
            "No threshold was preregistered for this development stability pilot."
        ),
        "support_or_variance_boundary_events": None,
        "support_or_variance_boundary_reason": (
            "The unchanged canonical trainer summary does not expose a registered "
            "support/variance boundary field; absence is not reported as zero."
        ),
        "fixed_1m_is_convergence": False,
    }
    _atomic_json(output / "terminal_audit.json", audit)
    return {"summary": summary, "terminal_audit": audit}
