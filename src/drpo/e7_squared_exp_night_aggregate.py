"""Aggregate the historical squared-night suite or its TD/GAE successor."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


EXPERIMENT_ID = "EXT-H-E7-SQUARED-EXP-NIGHT-01"
GAE_EXPERIMENT_ID = "EXT-H-E7-SQEXP-GAE-01"
EXPECTED_BRANCHES = 126
GAE_EXPECTED_BRANCHES = 96
EXPECTED_FINAL_STEP = 1_000_000
INTERMEDIATE_STEP = 500_000
LATE_WINDOW_START = 800_000
EXPECTED_SEEDS = (200, 201)
GAE_EXPECTED_SEEDS = (200, 201, 202, 203)
ACTOR_MODES = ("a2c", "ppo_clip_k4", "ppo_clip_kl_k16")
GAE_ESTIMATORS = ("td", "gae")
GAE_COEFFICIENTS = (64.0, 128.0, 256.0)
TUNING_PROFILE_ID = "d4rl9_common_c_p2_left"
TUNING_EXPECTED_BRANCHES = 180
TUNING_DATASETS = (
    "hopper-medium-v2",
    "hopper-medium-replay-v2",
    "hopper-medium-expert-v2",
    "walker2d-medium-v2",
    "walker2d-medium-replay-v2",
    "walker2d-medium-expert-v2",
    "halfcheetah-medium-v2",
    "halfcheetah-medium-replay-v2",
    "halfcheetah-medium-expert-v2",
)
TUNING_SEEDS = (200, 201)
TUNING_SCALES = (0.2, 0.16, 0.125, 0.1, 0.08, 0.0625, 0.04, 0.025, 0.015625)
P3_PROFILE_ID = "d4rl9_common_c_p3_left_saturation"
P3_EXPECTED_BRANCHES = 180
P3_SCALES = tuple(10.0 ** (-2.0 - index / 4.0) for index in range(9))
P3_LIVENESS_SCALE = 0.001
TUNING_FORMULA = (
    "w(D)=w(0)*exp(-taper_lambda*relu((D-remoteness_threshold)/remoteness_scale))"
)


def _is_tuning_profile(profile: Any) -> bool:
    return profile in {TUNING_PROFILE_ID, P3_PROFILE_ID}


def _profile_scales(profile: str) -> tuple[float, ...]:
    return P3_SCALES if profile == P3_PROFILE_ID else TUNING_SCALES


def _profile_expected_branches(profile: str) -> int:
    return P3_EXPECTED_BRANCHES if profile == P3_PROFILE_ID else TUNING_EXPECTED_BRANCHES


def _profile_prefix(profile: str) -> str:
    return "p3" if profile == P3_PROFILE_ID else "p2"


def _profile_liveness_scale(profile: str) -> float:
    return P3_LIVENESS_SCALE if profile == P3_PROFILE_ID else 0.1


def _profile_name(profile: str) -> str:
    return "P3 left saturation" if profile == P3_PROFILE_ID else "P2 left"


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


def _slope_per_100k(
    steps: list[int], scores: list[float], start: int
) -> float | None:
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


def _aggregate_geometry(
    path: Path, *, expected_final_step: int = EXPECTED_FINAL_STEP
) -> dict[str, Any]:
    records = _read_jsonl(path)
    final = records[-1]
    if final.get("status") != "complete" or int(final.get("update", -1)) != expected_final_step:
        raise RuntimeError(f"geometry diagnostics did not complete at expected step: {path}")
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


def _branch_dirs(work: Path) -> tuple[list[Path], str]:
    branch_root = work / "branches"
    if not branch_root.is_dir():
        raise RuntimeError("night-suite branch directory is missing")
    branch_dirs = sorted(path for path in branch_root.iterdir() if path.is_dir())
    if not branch_dirs:
        raise RuntimeError("night-suite has no branch directories")
    experiment_ids = {
        str(json.loads((path / "branch_config.json").read_text()).get("experiment_id"))
        for path in branch_dirs
    }
    if len(experiment_ids) != 1:
        raise RuntimeError(f"mixed experiment IDs in one work directory: {experiment_ids}")
    return branch_dirs, experiment_ids.pop()


def _legacy_aggregate(work: Path, branch_dirs: list[Path]) -> dict[str, Any]:
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
        steps, scores = _read_history(json.loads(summary_path.read_text()))
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
            "late_slope_per_100k": _slope_per_100k(
                steps, scores, LATE_WINDOW_START
            ),
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
                    "ppo_k4_minus_a2c_late": float(ppo4["late_window_mean_800k_1m"])
                    - float(a2c["late_window_mean_800k_1m"]),
                    "ppo_kl_k16_minus_ppo_k4_late": float(
                        ppo16["late_window_mean_800k_1m"]
                    )
                    - float(ppo4["late_window_mean_800k_1m"]),
                    "ppo_kl_k16_minus_a2c_late": float(
                        ppo16["late_window_mean_800k_1m"]
                    )
                    - float(a2c["late_window_mean_800k_1m"]),
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


def _gae_branch_row(branch_dir: Path) -> dict[str, Any]:
    if not (branch_dir / "COMPLETED.json").is_file():
        raise RuntimeError(f"incomplete GAE branch: {branch_dir.name}")
    branch = json.loads((branch_dir / "branch_config.json").read_text())
    manifest = json.loads((branch_dir / "branch_manifest.json").read_text())
    if branch.get("experiment_id") != GAE_EXPERIMENT_ID:
        raise RuntimeError(f"GAE branch experiment mismatch: {branch_dir.name}")
    profile = branch.get("profile_id")
    if profile is not None and not _is_tuning_profile(profile):
        raise RuntimeError(f"unknown GAE profile: {profile}")
    seed = int(branch["seed"])
    allowed_seeds = TUNING_SEEDS if _is_tuning_profile(profile) else GAE_EXPECTED_SEEDS
    if seed not in allowed_seeds:
        raise RuntimeError(f"forbidden GAE seed: {branch_dir.name}")
    values = branch["template_values"]
    estimator = str(values.get("advantage_estimator"))
    if _is_tuning_profile(profile) and values.get("actor_update_mode") != "a2c":
        raise RuntimeError(f"{_profile_name(str(profile))} requires canonical A2C")
    allowed_estimators = ("gae",) if _is_tuning_profile(profile) else GAE_ESTIMATORS
    if estimator not in allowed_estimators:
        raise RuntimeError(f"unknown advantage estimator: {estimator}")
    expected_step = int(values["steps"])
    summary_path = _only(
        (branch_dir / "trainer_output").glob("*_summary.json"), "trainer summary"
    )
    steps, scores = _read_history(json.loads(summary_path.read_text()))
    if steps[-1] != expected_step or not all(math.isfinite(score) for score in scores):
        raise RuntimeError(f"non-finite or incomplete GAE branch: {branch_dir.name}")
    snapshot = manifest.get("trajectory_snapshot", {})
    hashes = [str(value) for value in snapshot.get("snapshot_hashes", [])]
    if snapshot.get("critic_evolution_observed") is not True or len(hashes) < 2:
        raise RuntimeError(f"critic evolution or snapshots missing: {branch_dir.name}")
    control = branch["weight_control"]
    method = str(control["method"])
    if _is_tuning_profile(profile):
        if (
            control.get("formula") != TUNING_FORMULA
            or control.get("coordinate") != "normalized_squared_standardized_distance"
            or float(control.get("reference_distance", -1.0)) != 2.0
            or float(control.get("remoteness_threshold", -1.0)) != 0.0
            or float(control.get("taper_lambda", -1.0)) != 1.0
        ):
            raise RuntimeError(f"{_profile_name(str(profile))} public taper contract changed")
        if method not in {"positive_only", "thresholded_exponential"}:
            raise RuntimeError(f"unknown {_profile_name(str(profile))} control: {method}")
        scale = (
            float(control["remoteness_scale"])
            if method == "thresholded_exponential"
            else None
        )
        coefficient = (
            float(control["derived_exp_coefficient"])
            if method == "thresholded_exponential"
            else None
        )
    else:
        scale = None
        coefficient = (
            None if method == "positive_only" else float(control["exp_coefficient"])
        )
    late = [
        score
        for step, score in zip(steps, scores, strict=True)
        if step >= LATE_WINDOW_START
    ]
    best = max(range(len(scores)), key=scores.__getitem__)
    geometry = _aggregate_geometry(
        branch_dir / "geometry_diagnostics.jsonl", expected_final_step=expected_step
    )
    return {
        "branch_id": branch["branch_id"],
        "profile_id": profile,
        "dataset": branch["dataset_id"],
        "seed": seed,
        "execution_mode": str(values.get("execution_mode", "full")),
        "trainer_steps": expected_step,
        "advantage_estimator": estimator,
        "weight_method": method,
        "remoteness_scale": scale,
        "exp_coefficient": coefficient,
        "best_score": scores[best],
        "best_step": steps[best],
        "final_score": scores[-1],
        "best_to_final_drop": scores[best] - scores[-1],
        "late_window_mean_800k_1m": _mean(late) if late else None,
        "late_window_std_800k_1m": _sample_std(late) if late else None,
        "late_slope_per_100k": (
            _slope_per_100k(steps, scores, LATE_WINDOW_START) if late else None
        ),
        "snapshot_count": int(snapshot["snapshot_count"]),
        "snapshot_refresh_interval": int(snapshot["snapshot_refresh_interval"]),
        "snapshot_hashes": hashes,
        "first_snapshot_critic_sha256": snapshot["first_snapshot_critic_sha256"],
        "latest_snapshot_critic_sha256": snapshot["latest_snapshot_critic_sha256"],
        "final_critic_sha256": snapshot["final_critic_sha256"],
        "critic_evolution_observed": True,
        "geometry_effective_negative_mass_fraction": geometry[
            "effective_negative_mass_fraction"
        ],
        "task_performance_collapse_event": "not_adjudicated_no_registered_threshold",
        "support_or_variance_boundary_event": "not_instrumented_in_this_pilot",
        "rollout_failure_event": False,
        "nan_inf_numerical_failure": False,
    }


def _tuning_label(row: Mapping[str, Any]) -> str:
    method = str(row["weight_method"])
    if method == "thresholded_exponential":
        return f"drpo_c{float(row['remoteness_scale']):g}"
    return method


def _dataset_parts(dataset: str) -> tuple[str, str]:
    environment, remainder = dataset.split("-", 1)
    return environment, remainder.removesuffix("-v2")


def _validate_td_gae_pair(td: Mapping[str, Any], gae: Mapping[str, Any]) -> None:
    for field in ("dataset", "seed", "exp_coefficient", "execution_mode"):
        if td[field] != gae[field]:
            raise RuntimeError(f"TD/GAE pair mismatch in {field}")
    if td["advantage_estimator"] != "td" or gae["advantage_estimator"] != "gae":
        raise RuntimeError("TD/GAE pair estimator labels changed")
    if td["snapshot_refresh_interval"] != gae["snapshot_refresh_interval"]:
        raise RuntimeError("TD/GAE snapshot cadence diverged")
    if td["snapshot_count"] != gae["snapshot_count"]:
        raise RuntimeError("TD/GAE snapshot counts diverged")
    if td["snapshot_hashes"] != gae["snapshot_hashes"]:
        raise RuntimeError("TD/GAE critic snapshot trajectories diverged")


def _tuning_aggregate(
    work: Path, rows: list[dict[str, Any]], mode: str
) -> dict[str, Any]:
    profiles = {str(row["profile_id"]) for row in rows}
    if len(profiles) != 1:
        raise RuntimeError(f"mixed tuning profiles: {profiles}")
    profile = profiles.pop()
    if not _is_tuning_profile(profile):
        raise RuntimeError(f"unsupported tuning profile: {profile}")
    profile_name = _profile_name(profile)
    prefix = _profile_prefix(profile)
    scales = _profile_scales(profile)
    expected = 2 if mode == "liveness" else _profile_expected_branches(profile)
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} {profile_name} branches, found {len(rows)}")
    for row in rows:
        row["control"] = _tuning_label(row)
        if row["remoteness_scale"] is not None:
            row["log10_remoteness_scale"] = math.log10(float(row["remoteness_scale"]))
        if mode == "full" and int(row["trainer_steps"]) != EXPECTED_FINAL_STEP:
            raise RuntimeError(f"{profile_name} full branch did not reach one million steps")
    aggregate_dir = work / "aggregate"
    _write_csv(aggregate_dir / "branch_results.csv", rows)

    if mode == "liveness":
        observed = {row["control"] for row in rows}
        required = {
            "positive_only",
            f"drpo_c{_profile_liveness_scale(profile):g}",
        }
        if observed != required:
            raise RuntimeError(
                f"{profile_name} liveness controls changed: {sorted(observed)}"
            )
        audit = {
            "status": "PASS",
            "experiment_id": GAE_EXPERIMENT_ID,
            "profile_id": profile,
            "execution_mode": "liveness",
            "engineering_evidence_only": True,
            "scientific_aggregation_allowed": False,
            "branch_count_observed": len(rows),
            "expected_branch_count": 2,
            "critic_evolution_observed": True,
            "prepared_advantage_artifact_used": False,
            "held_out_seeds_touched": False,
            "formal_evidence_allowed": False,
        }
        _atomic_json(aggregate_dir / "LIVENESS_AUDIT.json", audit)
        summary = {
            "experiment_id": GAE_EXPERIMENT_ID,
            "profile_id": profile,
            "status": "PASS",
            "execution_mode": "liveness",
            "branch_count": len(rows),
            "audit": str(aggregate_dir / "LIVENESS_AUDIT.json"),
        }
        _atomic_json(aggregate_dir / "aggregate_summary.json", summary)
        return summary

    expected_controls = {"positive_only", *(f"drpo_c{x:g}" for x in scales)}
    cell_index: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        cell_index[(str(row["dataset"]), int(row["seed"]))].append(row)
    expected_cells = {
        (dataset, seed) for dataset in TUNING_DATASETS for seed in TUNING_SEEDS
    }
    if set(cell_index) != expected_cells:
        raise RuntimeError(f"{profile_name} task-seed matrix changed")
    for cell, values in cell_index.items():
        controls = {str(row["control"]) for row in values}
        if controls != expected_controls or len(values) != len(expected_controls):
            raise RuntimeError(
                f"{profile_name} control matrix changed for {cell}: {sorted(controls)}"
            )

    po = {
        (str(row["dataset"]), int(row["seed"])): row
        for row in rows
        if row["control"] == "positive_only"
    }
    paired: list[dict[str, Any]] = []
    for row in rows:
        if row["control"] == "positive_only":
            continue
        anchor = po[(str(row["dataset"]), int(row["seed"]))]
        paired.append(
            {
                "dataset": row["dataset"],
                "seed": row["seed"],
                "control": row["control"],
                "remoteness_scale": row["remoteness_scale"],
                "log10_remoteness_scale": row["log10_remoteness_scale"],
                "late_delta_vs_positive_only": float(row["late_window_mean_800k_1m"])
                - float(anchor["late_window_mean_800k_1m"]),
                "final_delta_vs_positive_only": float(row["final_score"])
                - float(anchor["final_score"]),
            }
        )

    task_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        task_groups[(str(row["control"]), str(row["dataset"]))].append(row)
    task_rows: list[dict[str, Any]] = []
    for (control, dataset), values in sorted(task_groups.items()):
        seeds = tuple(sorted(int(row["seed"]) for row in values))
        if seeds != TUNING_SEEDS:
            raise RuntimeError(
                f"{profile_name} seed set changed for {control},{dataset}: {seeds}"
            )
        environment, tier = _dataset_parts(dataset)
        scale = values[0]["remoteness_scale"]
        task_rows.append(
            {
                "control": control,
                "remoteness_scale": scale,
                "log10_remoteness_scale": (
                    None if scale is None else math.log10(float(scale))
                ),
                "dataset": dataset,
                "environment": environment,
                "tier": tier,
                "seeds": list(seeds),
                "late_mean": _mean(
                    [float(row["late_window_mean_800k_1m"]) for row in values]
                ),
                "final_mean": _mean([float(row["final_score"]) for row in values]),
                "best_mean": _mean([float(row["best_score"]) for row in values]),
                "best_to_final_drop_mean": _mean(
                    [float(row["best_to_final_drop"]) for row in values]
                ),
                "late_slope_per_100k_mean": _mean(
                    [float(row["late_slope_per_100k"]) for row in values]
                ),
            }
        )

    paired_by_control: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in paired:
        paired_by_control[str(row["control"])].append(row)
    task_by_control: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in task_rows:
        task_by_control[str(row["control"])].append(row)
    po_task = {
        str(row["dataset"]): row
        for row in task_rows
        if row["control"] == "positive_only"
    }
    controls: list[dict[str, Any]] = []
    for control, values in sorted(task_by_control.items()):
        if len(values) != len(TUNING_DATASETS):
            raise RuntimeError(f"{profile_name} task coverage changed for {control}")
        deltas = paired_by_control.get(control, [])
        scale = values[0]["remoteness_scale"]
        controls.append(
            {
                "control": control,
                "remoteness_scale": scale,
                "log10_remoteness_scale": (
                    None if scale is None else math.log10(float(scale))
                ),
                "task_count": len(values),
                "equal_task_weighted_late_mean": _mean(
                    [float(row["late_mean"]) for row in values]
                ),
                "equal_task_weighted_late_delta_vs_positive_only": (
                    None
                    if control == "positive_only"
                    else _mean(
                        [float(row["late_delta_vs_positive_only"]) for row in deltas]
                    )
                ),
                "median_task_late_mean": statistics.median(
                    float(row["late_mean"]) for row in values
                ),
                "minimum_task_late_mean": min(
                    float(row["late_mean"]) for row in values
                ),
                "equal_task_weighted_final_mean": _mean(
                    [float(row["final_mean"]) for row in values]
                ),
                "equal_task_weighted_best_mean": _mean(
                    [float(row["best_mean"]) for row in values]
                ),
                "equal_task_weighted_best_to_final_drop": _mean(
                    [float(row["best_to_final_drop_mean"]) for row in values]
                ),
                "equal_task_weighted_late_slope_per_100k": _mean(
                    [float(row["late_slope_per_100k_mean"]) for row in values]
                ),
                "paired_seed_win_count_vs_positive_only": sum(
                    float(row["late_delta_vs_positive_only"]) > 0 for row in deltas
                ),
                "paired_seed_count": len(deltas),
                "paired_seed_win_rate_vs_positive_only": (
                    sum(
                        float(row["late_delta_vs_positive_only"]) > 0
                        for row in deltas
                    )
                    / len(deltas)
                    if deltas
                    else None
                ),
                "task_win_count_vs_positive_only": (
                    None
                    if control == "positive_only"
                    else sum(
                        float(row["late_mean"])
                        > float(po_task[str(row["dataset"])]["late_mean"])
                        for row in values
                    )
                ),
            }
        )

    strata: list[dict[str, Any]] = []
    for dimension in ("environment", "tier"):
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in task_rows:
            grouped[(str(row["control"]), str(row[dimension]))].append(row)
        for (control, value), values in sorted(grouped.items()):
            scale = values[0]["remoteness_scale"]
            anchor_values = [
                po_task[str(row["dataset"])] for row in values
            ]
            strata.append(
                {
                    "control": control,
                    "remoteness_scale": scale,
                    "log10_remoteness_scale": (
                        None if scale is None else math.log10(float(scale))
                    ),
                    "dimension": dimension,
                    "value": value,
                    "task_count": len(values),
                    "late_mean": _mean([float(row["late_mean"]) for row in values]),
                    "late_delta_vs_positive_only": (
                        None
                        if control == "positive_only"
                        else _mean(
                            [
                                float(row["late_mean"]) - float(anchor["late_mean"])
                                for row, anchor in zip(
                                    values, anchor_values, strict=True
                                )
                            ]
                        )
                    ),
                    "final_mean": _mean(
                        [float(row["final_mean"]) for row in values]
                    ),
                }
            )

    _write_csv(aggregate_dir / f"{prefix}_paired_deltas.csv", paired)
    _write_csv(aggregate_dir / f"{prefix}_task_summary.csv", task_rows)
    _write_csv(aggregate_dir / f"{prefix}_control_summary.csv", controls)
    _write_csv(aggregate_dir / f"{prefix}_stratum_summary.csv", strata)
    selection_status = (
        "left_saturation_curve_only_no_best_c_selection"
        if profile == P3_PROFILE_ID
        else "response_curve_only_pending_protocol_freeze"
    )
    audit = {
        "status": "PASS",
        "experiment_id": GAE_EXPERIMENT_ID,
        "profile_id": profile,
        "raw_complete": True,
        "branch_count_observed": len(rows),
        "expected_branch_count": _profile_expected_branches(profile),
        "task_count": len(TUNING_DATASETS),
        "development_seeds": list(TUNING_SEEDS),
        "held_out_seeds_touched": False,
        "critic_updated_during_actor_training": True,
        "prepared_advantage_artifact_used": False,
        "task_performance_collapse_status": (
            "not_adjudicated_no_registered_threshold"
        ),
        "support_or_variance_boundary_status": "not_instrumented_in_this_pilot",
        "rollout_failure_count": 0,
        "nan_inf_numerical_failure_count": 0,
        "task_performance_collapse_separate": True,
        "support_or_variance_boundary_separate": True,
        "rollout_failure_separate": True,
        "nan_inf_separate": True,
        "fixed_horizon_is_not_convergence": True,
        "selected_control": None,
        "selection_status": selection_status,
        "curve_shape_claim_allowed": profile == P3_PROFILE_ID,
        "best_c_claim_allowed": False,
        "method_ranking_claim_allowed": False,
        "steady_state_ranking_allowed": False,
        "formal_evidence_allowed": False,
    }
    _atomic_json(aggregate_dir / "terminal_audit.json", audit)
    summary = {
        "experiment_id": GAE_EXPERIMENT_ID,
        "profile_id": profile,
        "status": "PASS",
        "branch_count": len(rows),
        "task_count": len(TUNING_DATASETS),
        "control_count": len(controls),
        "selection_status": selection_status,
        "terminal_audit": str(aggregate_dir / "terminal_audit.json"),
    }
    _atomic_json(aggregate_dir / "aggregate_summary.json", summary)
    return summary


def _gae_aggregate(work: Path, branch_dirs: list[Path]) -> dict[str, Any]:
    rows = [_gae_branch_row(path) for path in branch_dirs]
    modes = {str(row["execution_mode"]) for row in rows}
    if len(modes) != 1:
        raise RuntimeError(f"mixed GAE execution modes: {modes}")
    mode = modes.pop()
    profiles = {row["profile_id"] for row in rows}
    if len(profiles) != 1:
        raise RuntimeError(f"mixed GAE profiles: {profiles}")
    profile = profiles.pop()
    if _is_tuning_profile(profile):
        return _tuning_aggregate(work, rows, mode)
    expected = 2 if mode == "liveness" else GAE_EXPECTED_BRANCHES
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} GAE branches, found {len(rows)}")

    by_pair: dict[tuple[str, int, float | None], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (row["dataset"], int(row["seed"]), row["exp_coefficient"])
        estimator = str(row["advantage_estimator"])
        if estimator in by_pair[key]:
            raise RuntimeError(f"duplicate estimator in TD/GAE pair: {key},{estimator}")
        by_pair[key][estimator] = row
    pair_rows: list[dict[str, Any]] = []
    for key, pair in sorted(by_pair.items(), key=lambda item: repr(item[0])):
        if set(pair) != set(GAE_ESTIMATORS):
            raise RuntimeError(f"incomplete TD/GAE pair for {key}: {sorted(pair)}")
        td, gae = pair["td"], pair["gae"]
        _validate_td_gae_pair(td, gae)
        pair_rows.append(
            {
                "dataset": key[0],
                "seed": key[1],
                "exp_coefficient": key[2],
                "gae_minus_td_final": float(gae["final_score"])
                - float(td["final_score"]),
                "gae_minus_td_late": (
                    None
                    if gae["late_window_mean_800k_1m"] is None
                    else float(gae["late_window_mean_800k_1m"])
                    - float(td["late_window_mean_800k_1m"])
                ),
                "snapshot_count": int(td["snapshot_count"]),
                "snapshot_refresh_interval": int(td["snapshot_refresh_interval"]),
                "snapshot_hashes_match": True,
            }
        )

    aggregate_dir = work / "aggregate"
    _write_csv(aggregate_dir / "branch_results.csv", rows)
    _write_csv(aggregate_dir / "gae_td_paired_results.csv", pair_rows)

    if mode == "liveness":
        audit = {
            "status": "PASS",
            "experiment_id": GAE_EXPERIMENT_ID,
            "execution_mode": "liveness",
            "engineering_evidence_only": True,
            "scientific_aggregation_allowed": False,
            "branch_count_observed": len(rows),
            "expected_branch_count": 2,
            "matched_pair_count": 1,
            "snapshot_hash_trajectories_match": True,
            "critic_evolution_observed": True,
            "prepared_advantage_artifact_used": False,
            "held_out_seeds_touched": False,
            "formal_evidence_allowed": False,
        }
        _atomic_json(aggregate_dir / "LIVENESS_AUDIT.json", audit)
        summary = {
            "experiment_id": GAE_EXPERIMENT_ID,
            "status": "PASS",
            "execution_mode": "liveness",
            "branch_count": len(rows),
            "pair_count": len(pair_rows),
            "audit": str(aggregate_dir / "LIVENESS_AUDIT.json"),
        }
        _atomic_json(aggregate_dir / "aggregate_summary.json", summary)
        return summary

    grouped: dict[tuple[str, str, float | None], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["advantage_estimator"], row["exp_coefficient"])].append(row)
    groups: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items(), key=lambda item: repr(item[0])):
        seeds = tuple(sorted(int(row["seed"]) for row in values))
        if seeds != GAE_EXPECTED_SEEDS:
            raise RuntimeError(f"paired seed set changed for {key}: {seeds}")
        groups.append(
            {
                "dataset": key[0],
                "advantage_estimator": key[1],
                "exp_coefficient": key[2],
                "seeds": list(seeds),
                "best_mean": _mean([float(row["best_score"]) for row in values]),
                "final_mean": _mean([float(row["final_score"]) for row in values]),
                "final_seed_std": _sample_std([float(row["final_score"]) for row in values]),
                "late_mean": _mean(
                    [float(row["late_window_mean_800k_1m"]) for row in values]
                ),
                "late_seed_std": _sample_std(
                    [float(row["late_window_mean_800k_1m"]) for row in values]
                ),
            }
        )
    index = {
        (group["dataset"], group["advantage_estimator"], group["exp_coefficient"]): group
        for group in groups
    }
    comparisons = [
        {
            "dataset": dataset,
            "exp_coefficient": coefficient,
            "gae_minus_td_final": index[(dataset, "gae", coefficient)]["final_mean"]
            - index[(dataset, "td", coefficient)]["final_mean"],
            "gae_minus_td_late": index[(dataset, "gae", coefficient)]["late_mean"]
            - index[(dataset, "td", coefficient)]["late_mean"],
        }
        for dataset in sorted({row["dataset"] for row in rows})
        for coefficient in (None, *GAE_COEFFICIENTS)
    ]
    _write_csv(aggregate_dir / "group_summary.csv", groups)
    _write_csv(aggregate_dir / "gae_td_comparisons.csv", comparisons)
    audit = {
        "status": "PASS",
        "experiment_id": GAE_EXPERIMENT_ID,
        "raw_complete": True,
        "branch_count_observed": len(rows),
        "expected_branch_count": GAE_EXPECTED_BRANCHES,
        "matched_pair_count": len(pair_rows),
        "snapshot_hash_trajectories_match": True,
        "critic_updated_during_actor_training": True,
        "prepared_advantage_artifact_used": False,
        "held_out_seeds_touched": False,
        "task_performance_collapse_separate": True,
        "support_or_variance_boundary_separate": True,
        "nan_inf_separate": True,
        "convergence_claim_allowed": False,
        "steady_state_ranking_allowed": False,
        "universal_gae_superiority_claim_allowed": False,
        "formal_evidence_allowed": False,
    }
    _atomic_json(aggregate_dir / "terminal_audit.json", audit)
    summary = {
        "experiment_id": GAE_EXPERIMENT_ID,
        "status": "PASS",
        "branch_count": len(rows),
        "group_count": len(groups),
        "paired_comparison_count": len(comparisons),
        "terminal_audit": str(aggregate_dir / "terminal_audit.json"),
    }
    _atomic_json(aggregate_dir / "aggregate_summary.json", summary)
    return summary


def aggregate(work_dir: str | Path) -> dict[str, Any]:
    work = Path(work_dir).expanduser().resolve()
    branch_dirs, experiment_id = _branch_dirs(work)
    if experiment_id == EXPERIMENT_ID:
        return _legacy_aggregate(work, branch_dirs)
    if experiment_id == GAE_EXPERIMENT_ID:
        return _gae_aggregate(work, branch_dirs)
    raise RuntimeError(f"unsupported aggregate experiment_id={experiment_id!r}")
