#!/usr/bin/env python3
"""Formal C-U1 stepwise negative-gradient-budget-matched taper comparison.

Experiment ID: C-U1-E4-TAPER-BUDGET-MATCH-01.

The primary fairness coordinate is the L2 norm of the *raw negative-gradient
vector before Adam*.  For every seed and minibatch step, an independently run
reciprocal-linear reference produces the frozen target norm.  Reciprocal-
quadratic, exponential, squared-distance exponential, and a non-selective
unweighted direction are rescaled by one detached scalar so that their raw
negative-gradient norm matches the reference target at that exact step.

This does not claim that Adam's parameter-update norm is matched.  The realized
Adam total parameter-update norm is logged as a secondary diagnostic.  The
experiment is finite-horizon fairness evidence; long-run terminal resolution is
owned by C-U1-E4-TAPER-CONV-01.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import shutil
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

try:
    from . import cu1_core
    from . import cu1_distance_taper_formal as taper_base
    from . import cu1_taper_near_retention_formal as near_base
    from . import drpo_cu1_e1_e4_oneclick as experiment
except ImportError:  # direct script execution
    import cu1_core
    import cu1_distance_taper_formal as taper_base
    import cu1_taper_near_retention_formal as near_base
    import drpo_cu1_e1_e4_oneclick as experiment


EXPERIMENT_ID = "C-U1-E4-TAPER-BUDGET-MATCH-01"
SCRIPT_VERSION = "2026.06.29-stepwise-negative-gradient-l2-v1"
EPS = 1e-12


@dataclass(frozen=True)
class BudgetMatchProtocol:
    development_seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    formal_seeds: tuple[int, ...] = tuple(range(110, 130))
    target_retention: float = 0.75
    reference_distance: float = 5.0
    near_region_boundary: float = 5.0
    selective_families: tuple[str, ...] = (
        "reciprocal_linear",
        "reciprocal_quadratic",
        "exponential",
        "squared_distance_exponential",
    )
    matched_methods: tuple[str, ...] = (
        "reciprocal_linear",
        "reciprocal_quadratic",
        "exponential",
        "squared_distance_exponential",
        "global_stepwise_scale",
    )
    boundary_controls: tuple[str, ...] = ("positive_only", "unweighted_boundary")
    learning_rate: float = 5e-4
    batch_states: int = 256
    evaluation_interval: int = 100
    maximum_steps: int = 8000
    minimum_steps_before_stationarity: int = 1000
    stable_windows: int = 10
    normalized_slope_threshold: float = 1e-4
    normalized_field_residual_threshold: float = 2e-3
    positive_absolute_gradient_threshold: float = 1e-3
    task_failure_retention: float = 0.45
    log_sigma_event_boundary: float = 12.0
    calibration_tolerance: float = 1e-6
    budget_relative_tolerance: float = 1e-6
    bootstrap_samples: int = 4000
    checkpoint_every_formal_seeds: int = 5


PROTOCOL = BudgetMatchProtocol()


def atomic_json(path: Path, value: Any) -> None:
    taper_base.atomic_json(path, value)


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    taper_base.write_csv(path, rows)


def log_message(output_root: Path, message: str) -> None:
    taper_base.log_message(output_root, message)


def method_names(protocol: BudgetMatchProtocol) -> tuple[str, ...]:
    return (*protocol.boundary_controls, *protocol.matched_methods)


def family_for_method(method: str) -> str:
    if method in {
        "reciprocal_linear",
        "reciprocal_quadratic",
        "exponential",
        "squared_distance_exponential",
    }:
        return method
    if method in {"global_stepwise_scale", "unweighted_boundary"}:
        return "unweighted"
    if method == "positive_only":
        return "positive_only"
    raise ValueError(f"unknown method: {method}")


def coefficient_for_method(
    method: str,
    coefficients: dict[tuple[str, float], float],
    protocol: BudgetMatchProtocol,
) -> float:
    family = family_for_method(method)
    if family in {"positive_only", "unweighted"}:
        return 0.0
    return coefficients[(family, protocol.target_retention)]


def calibration_protocol(protocol: BudgetMatchProtocol) -> near_base.NearRetentionProtocol:
    return near_base.NearRetentionProtocol(
        development_seeds=protocol.development_seeds,
        formal_seeds=protocol.formal_seeds,
        primary_retention=protocol.target_retention,
        sensitivity_retentions=(),
        families=protocol.selective_families,
        reference_distance=protocol.reference_distance,
        near_region_boundary=protocol.near_region_boundary,
        learning_rate=protocol.learning_rate,
        batch_states=protocol.batch_states,
        evaluation_interval=protocol.evaluation_interval,
        minimum_steps=protocol.minimum_steps_before_stationarity,
        maximum_steps=protocol.maximum_steps,
        stable_windows=protocol.stable_windows,
        normalized_slope_threshold=protocol.normalized_slope_threshold,
        normalized_field_residual_threshold=protocol.normalized_field_residual_threshold,
        positive_absolute_gradient_threshold=protocol.positive_absolute_gradient_threshold,
        task_failure_retention=protocol.task_failure_retention,
        log_sigma_event_boundary=protocol.log_sigma_event_boundary,
        calibration_tolerance=protocol.calibration_tolerance,
        bootstrap_samples=protocol.bootstrap_samples,
        checkpoint_every_formal_seeds=protocol.checkpoint_every_formal_seeds,
    )


def weighted_negative_loss(
    actor: experiment.GaussianActor,
    split: experiment.Split,
    ids: torch.Tensor,
    method: str,
    coefficient: float,
    protocol: BudgetMatchProtocol,
) -> torch.Tensor:
    family = family_for_method(method)
    if family == "positive_only":
        return torch.zeros((), device=experiment.DEVICE)
    mu, log_std = actor(split.s[ids])
    actions = split.negative_actions[ids]
    advantages = split.negative_advantages[ids]
    log_prob = cu1_core.gaussian_log_prob(mu, log_std, actions, experiment.P.action_dim)
    distance = cu1_core.standardized_distance(mu, log_std, actions)
    weight = near_base.taper_weight_from_coefficient(
        distance.detach(), family, coefficient, protocol.reference_distance
    )
    return -(advantages * weight * log_prob).mean()


def _parameter_update_norm(before: list[torch.Tensor], actor: experiment.GaussianActor) -> float:
    pieces = []
    for old, parameter in zip(before, actor.parameters()):
        pieces.append((parameter.detach() - old).reshape(-1))
    return float(torch.linalg.vector_norm(torch.cat(pieces)).item()) if pieces else 0.0


def _set_combined_gradient(
    actor: experiment.GaussianActor,
    positive_grad: tuple[torch.Tensor | None, ...],
    negative_grad: tuple[torch.Tensor | None, ...],
    scale: float,
) -> None:
    combined = experiment.add_tuples(
        positive_grad,
        negative_grad,
        scales=(1.0, scale),
    )
    experiment.set_parameter_grads(list(actor.parameters()), combined)


def _scaled_retention(
    actor: experiment.GaussianActor,
    split: experiment.Split,
    method: str,
    coefficient: float,
    scale: float,
    protocol: BudgetMatchProtocol,
) -> dict[str, float]:
    near_protocol = calibration_protocol(protocol)
    values = near_base.retention_and_harm_diagnostics(
        actor,
        split,
        family_for_method(method),
        coefficient,
        near_protocol,
    )
    linear_fields = {
        "near_region_weight_mean",
        "far_region_weight_mean",
        "near_useful_weight_mean",
        "near_useful_gradient_retention",
        "far_harmful_influence_retention",
        "far_harmful_weighted_projection_mean",
        "far_harmful_weighted_update_norm_mean",
    }
    return {
        key: (float(value) * scale if key in linear_fields else float(value))
        for key, value in values.items()
    }


def evaluate_state(
    actor: experiment.GaussianActor,
    environment: experiment.Environment,
    method: str,
    coefficient: float,
    scale: float,
    protocol: BudgetMatchProtocol,
    initial_reward: float,
) -> dict[str, Any]:
    task = experiment.evaluation(actor, environment.test)
    positive = experiment.positive_loss(actor, environment.train)
    positive_grad = taper_base.gradient_tuple(positive, actor, retain_graph=method != "positive_only")
    positive_norm = float(experiment.norm_tuple(positive_grad).item())
    if method == "positive_only":
        negative_norm = 0.0
        total_norm = positive_norm
        residual = float("nan")
        stationarity = positive_norm
        stationarity_kind = "absolute_positive_gradient_norm"
        retention = _scaled_retention(
            actor, environment.train, method, coefficient, 0.0, protocol
        )
    else:
        all_ids = torch.arange(experiment.P.n_train_states, device=experiment.DEVICE)
        negative = weighted_negative_loss(
            actor, environment.train, all_ids, method, coefficient, protocol
        )
        negative_grad = taper_base.gradient_tuple(negative, actor, retain_graph=False)
        scaled_negative = experiment.scale_tuple(negative_grad, scale)
        total_grad = experiment.add_tuples(positive_grad, scaled_negative)
        negative_norm = float(experiment.norm_tuple(scaled_negative).item())
        total_norm = float(experiment.norm_tuple(total_grad).item())
        residual = total_norm / (positive_norm + negative_norm + EPS)
        stationarity = residual
        stationarity_kind = "normalized_signed_field_residual"
        retention = _scaled_retention(
            actor, environment.train, method, coefficient, scale, protocol
        )
    finite = experiment.finite_model(actor) and all(
        math.isfinite(float(task[key]))
        for key in ("reward", "sigma_mean", "log_sigma_min", "log_sigma_max")
    )
    reward_retention = float(task["reward"]) / max(initial_reward, EPS)
    return {
        **task,
        "reward_retention": reward_retention,
        "positive_gradient_norm": positive_norm,
        "negative_gradient_norm": negative_norm,
        "total_gradient_norm": total_norm,
        "normalized_field_residual": residual,
        "stationarity_residual": stationarity,
        "stationarity_residual_kind": stationarity_kind,
        "effective_negative_scale": scale,
        "task_performance_collapse_event": reward_retention < protocol.task_failure_retention,
        "support_or_variance_boundary_event": max(
            abs(float(task["log_sigma_min"])), abs(float(task["log_sigma_max"]))
        ) >= protocol.log_sigma_event_boundary,
        "nan_inf_numerical_event": not finite,
        **retention,
    }


def _train_one(
    *,
    seed: int,
    method: str,
    initial_state: dict[str, torch.Tensor],
    environment: experiment.Environment,
    coefficient: float,
    protocol: BudgetMatchProtocol,
    output_root: Path,
    target_schedule: list[float] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[float]]:
    summary_path = output_root / "runs" / method / f"seed_{seed}.json"
    trajectory_path = output_root / "runs" / method / f"seed_{seed}_trajectory.csv"
    schedule_path = output_root / "runs" / method / f"seed_{seed}_negative_budget.json"
    if summary_path.exists() and trajectory_path.exists() and schedule_path.exists():
        return (
            json.loads(summary_path.read_text(encoding="utf-8")),
            experiment.read_csv(trajectory_path),
            json.loads(schedule_path.read_text(encoding="utf-8"))["realized_negative_norms"],
        )

    experiment.seed_all(seed + 2_300_000)
    actor = experiment.GaussianActor().to(experiment.DEVICE)
    actor.load_state_dict(copy.deepcopy(initial_state))
    optimizer = torch.optim.Adam(actor.parameters(), lr=protocol.learning_rate)
    index_generator = torch.Generator(device="cpu").manual_seed(seed + 1_700_003)
    initial_reward = float(experiment.evaluation(actor, environment.test)["reward"])

    trajectory: list[dict[str, Any]] = []
    realized_negative_norms: list[float] = []
    applied_scales: list[float] = []
    relative_errors: list[float] = []
    parameter_update_norms: list[float] = []
    stable_candidate_step: int | None = None
    candidate_classification: tuple[bool, bool, bool] | None = None
    classification_at_2x: tuple[bool, bool, bool] | None = None
    stop_reason = "maximum_steps"
    last_scale = 0.0 if method == "positive_only" else 1.0
    started = time.perf_counter()

    for step in range(1, protocol.maximum_steps + 1):
        ids = torch.randint(
            0,
            experiment.P.n_train_states,
            (protocol.batch_states,),
            generator=index_generator,
        ).to(experiment.DEVICE)
        positive = experiment.positive_loss(actor, environment.train, ids)
        positive_grad = taper_base.gradient_tuple(
            positive, actor, retain_graph=method != "positive_only"
        )
        raw_negative_norm = 0.0
        target_norm = 0.0
        scale = 0.0
        relative_error = 0.0
        negative_loss_value = 0.0
        if method != "positive_only":
            negative = weighted_negative_loss(
                actor, environment.train, ids, method, coefficient, protocol
            )
            negative_grad = taper_base.gradient_tuple(negative, actor, retain_graph=False)
            raw_negative_norm = float(experiment.norm_tuple(negative_grad).item())
            if method in protocol.matched_methods:
                if method == "reciprocal_linear" and target_schedule is None:
                    target_norm = raw_negative_norm
                    scale = 1.0
                else:
                    if target_schedule is None or len(target_schedule) < step:
                        raise RuntimeError("reference negative-gradient budget schedule is incomplete")
                    target_norm = float(target_schedule[step - 1])
                    if raw_negative_norm <= EPS:
                        if target_norm > protocol.budget_relative_tolerance:
                            raise RuntimeError(
                                f"zero candidate negative gradient at seed={seed} step={step}"
                            )
                        scale = 0.0
                    else:
                        scale = target_norm / raw_negative_norm
                realized_norm = raw_negative_norm * scale
                relative_error = abs(realized_norm - target_norm) / max(target_norm, EPS)
            else:  # unweighted boundary control is deliberately unmatched
                target_norm = raw_negative_norm
                scale = 1.0
                realized_norm = raw_negative_norm
            negative_loss_value = float(negative.detach().item())
            _set_combined_gradient(actor, positive_grad, negative_grad, scale)
        else:
            experiment.set_parameter_grads(list(actor.parameters()), positive_grad)
            realized_norm = 0.0

        before = [parameter.detach().clone() for parameter in actor.parameters()]
        optimizer.step()
        update_norm = _parameter_update_norm(before, actor)
        realized_negative_norms.append(float(realized_norm))
        applied_scales.append(float(scale))
        relative_errors.append(float(relative_error))
        parameter_update_norms.append(update_norm)
        last_scale = float(scale)

        should_evaluate = (
            step == 1
            or step % protocol.evaluation_interval == 0
            or step == protocol.maximum_steps
        )
        if not should_evaluate:
            continue
        state = evaluate_state(
            actor,
            environment,
            method,
            coefficient,
            last_scale,
            protocol,
            initial_reward,
        )
        row = {
            "seed": seed,
            "step": step,
            "method": method,
            "coefficient": coefficient,
            "positive_loss": float(positive.detach().item()),
            "negative_loss": negative_loss_value,
            "target_negative_gradient_norm": target_norm,
            "raw_negative_gradient_norm": raw_negative_norm,
            "realized_negative_gradient_norm": realized_norm,
            "budget_relative_error": relative_error,
            "negative_scale": scale,
            "total_parameter_update_norm": update_norm,
            "cumulative_target_negative_gradient_norm": float(sum(
                target_schedule[:step] if target_schedule is not None else realized_negative_norms
            )),
            "cumulative_realized_negative_gradient_norm": float(sum(realized_negative_norms)),
            "cumulative_total_parameter_update_norm": float(sum(parameter_update_norms)),
            **state,
        }
        trajectory.append(row)
        write_csv(trajectory_path, trajectory)

        classification = (
            bool(state["task_performance_collapse_event"]),
            bool(state["support_or_variance_boundary_event"]),
            bool(state["nan_inf_numerical_event"]),
        )
        if state["nan_inf_numerical_event"]:
            stop_reason = "nan_inf_numerical_event"
            break
        if state["support_or_variance_boundary_event"]:
            stop_reason = "support_or_variance_boundary_event"
            break
        if (
            stable_candidate_step is None
            and step >= protocol.minimum_steps_before_stationarity
            and len(trajectory) >= protocol.stable_windows
        ):
            window = trajectory[-protocol.stable_windows :]
            max_slope = max(
                taper_base.normalized_slope(window, field)
                for field in ("reward", "normalized_extrapolation_displacement", "sigma_mean")
            )
            threshold = (
                protocol.positive_absolute_gradient_threshold
                if method == "positive_only"
                else protocol.normalized_field_residual_threshold
            )
            if max_slope < protocol.normalized_slope_threshold and float(
                state["stationarity_residual"]
            ) < threshold:
                stable_candidate_step = step
                candidate_classification = classification
        if stable_candidate_step is not None and step == 2 * stable_candidate_step:
            classification_at_2x = classification

    elapsed = time.perf_counter() - started
    final_state = evaluate_state(
        actor,
        environment,
        method,
        coefficient,
        last_scale,
        protocol,
        initial_reward,
    )
    completed_steps = len(realized_negative_norms)
    checkpoint_path = output_root / "checkpoints" / method / f"seed_{seed}_terminal.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "experiment_id": EXPERIMENT_ID,
            "seed": seed,
            "method": method,
            "steps_completed": completed_steps,
            "actor_state_dict": actor.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "coefficient": coefficient,
            "last_negative_scale": last_scale,
        },
        checkpoint_path,
    )
    schedule_payload = {
        "experiment_id": EXPERIMENT_ID,
        "seed": seed,
        "method": method,
        "primary_budget_coordinate": "stepwise_raw_negative_gradient_l2_before_Adam",
        "target_schedule_source": (
            "self_reciprocal_linear_reference"
            if method == "reciprocal_linear" and target_schedule is None
            else "paired_reciprocal_linear_reference"
            if method in protocol.matched_methods
            else "not_matched_boundary_control"
        ),
        "realized_negative_norms": realized_negative_norms,
        "applied_scales": applied_scales,
        "relative_errors": relative_errors,
        "total_parameter_update_norms": parameter_update_norms,
    }
    atomic_json(schedule_path, schedule_payload)
    summary = {
        "seed": seed,
        "method": method,
        "coefficient": coefficient,
        "steps_completed": completed_steps,
        "stop_reason": stop_reason,
        "stable_candidate_step": stable_candidate_step,
        "candidate_classification": candidate_classification,
        "classification_at_2x": classification_at_2x,
        "elapsed_seconds": elapsed,
        "initial_reward": initial_reward,
        "maximum_budget_relative_error": max(relative_errors, default=0.0),
        "mean_budget_relative_error": float(np.mean(relative_errors)) if relative_errors else 0.0,
        "cumulative_realized_negative_gradient_norm": float(sum(realized_negative_norms)),
        "cumulative_total_parameter_update_norm": float(sum(parameter_update_norms)),
        "mean_negative_scale": float(np.mean(applied_scales)) if applied_scales else 0.0,
        "terminal_checkpoint": str(checkpoint_path),
        **final_state,
    }
    atomic_json(summary_path, summary)
    return summary, trajectory, realized_negative_norms


def bootstrap_ci(values: list[float], protocol: BudgetMatchProtocol, seed: int) -> tuple[float, float, float]:
    return experiment.mean_ci(values, seed=seed, n_boot=protocol.bootstrap_samples)


def aggregate_results(
    summaries: list[dict[str, Any]],
    output_root: Path,
    protocol: BudgetMatchProtocol,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    aggregate_rows: list[dict[str, Any]] = []
    numeric_fields = (
        "reward",
        "reward_retention",
        "near_useful_gradient_retention",
        "far_harmful_influence_retention",
        "cumulative_realized_negative_gradient_norm",
        "cumulative_total_parameter_update_norm",
        "maximum_budget_relative_error",
    )
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in summaries:
        grouped.setdefault(str(row["method"]), []).append(row)
    for method, rows in sorted(grouped.items()):
        item: dict[str, Any] = {
            "method": method,
            "n": len(rows),
            "task_performance_collapse_events": sum(bool(r["task_performance_collapse_event"]) for r in rows),
            "support_or_variance_boundary_events": sum(bool(r["support_or_variance_boundary_event"]) for r in rows),
            "nan_inf_numerical_events": sum(bool(r["nan_inf_numerical_event"]) for r in rows),
        }
        for index, field in enumerate(numeric_fields):
            values = [float(row[field]) for row in rows if math.isfinite(float(row[field]))]
            mean, low, high = bootstrap_ci(values, protocol, 2026062900 + index)
            item[f"{field}_mean"] = mean
            item[f"{field}_ci95_low"] = low
            item[f"{field}_ci95_high"] = high
        aggregate_rows.append(item)
    write_csv(output_root / "aggregate.csv", aggregate_rows)

    reference = {int(row["seed"]): row for row in grouped["reciprocal_linear"]}
    paired_rows: list[dict[str, Any]] = []
    paired_summary: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "reference_method": "reciprocal_linear",
        "primary_budget_coordinate": "stepwise_raw_negative_gradient_l2_before_Adam",
        "comparisons": {},
    }
    for method in (
        "reciprocal_quadratic",
        "exponential",
        "squared_distance_exponential",
        "global_stepwise_scale",
    ):
        candidate = {int(row["seed"]): row for row in grouped[method]}
        reward_delta: list[float] = []
        harmful_delta: list[float] = []
        for seed in protocol.formal_seeds:
            left, right = reference[seed], candidate[seed]
            row = {
                "seed": seed,
                "candidate": method,
                "reward_delta_candidate_minus_linear": float(right["reward"]) - float(left["reward"]),
                "far_harmful_retention_delta_candidate_minus_linear": float(right["far_harmful_influence_retention"]) - float(left["far_harmful_influence_retention"]),
            }
            paired_rows.append(row)
            reward_delta.append(row["reward_delta_candidate_minus_linear"])
            harmful_delta.append(row["far_harmful_retention_delta_candidate_minus_linear"])
        rmean, rlow, rhigh = bootstrap_ci(reward_delta, protocol, 2026062911)
        hmean, hlow, hhigh = bootstrap_ci(harmful_delta, protocol, 2026062912)
        paired_summary["comparisons"][method] = {
            "reward_delta_mean": rmean,
            "reward_delta_ci95": [rlow, rhigh],
            "reward_positive_seeds": sum(value > 0 for value in reward_delta),
            "far_harmful_retention_delta_mean": hmean,
            "far_harmful_retention_delta_ci95": [hlow, hhigh],
            "far_harmful_lower_seeds": sum(value < 0 for value in harmful_delta),
        }
    write_csv(output_root / "paired_comparisons.csv", paired_rows)
    atomic_json(output_root / "paired_summary.json", paired_summary)
    return aggregate_rows, paired_rows, paired_summary


def build_terminal_audit(
    summaries: list[dict[str, Any]],
    protocol: BudgetMatchProtocol,
    base_commit: str,
    smoke: bool,
) -> dict[str, Any]:
    expected = len(protocol.formal_seeds) * len(method_names(protocol))
    matched = [row for row in summaries if row["method"] in protocol.matched_methods]
    max_error = max((float(row["maximum_budget_relative_error"]) for row in matched), default=0.0)
    checks = [
        {"name": "complete_run_matrix", "passed": len(summaries) == expected, "actual": len(summaries), "expected": expected},
        {"name": "formal_seed_coverage", "passed": sorted({int(r["seed"]) for r in summaries}) == sorted(protocol.formal_seeds), "actual": sorted({int(r["seed"]) for r in summaries}), "expected": list(protocol.formal_seeds)},
        {"name": "method_coverage", "passed": sorted({str(r["method"]) for r in summaries}) == sorted(method_names(protocol)), "actual": sorted({str(r["method"]) for r in summaries}), "expected": sorted(method_names(protocol))},
        {"name": "stepwise_negative_gradient_budget_match", "passed": max_error <= protocol.budget_relative_tolerance, "actual": max_error, "expected_maximum": protocol.budget_relative_tolerance},
        {"name": "no_nan_inf_numerical_failure", "passed": not any(bool(r["nan_inf_numerical_event"]) for r in summaries), "actual": sum(bool(r["nan_inf_numerical_event"]) for r in summaries), "expected": 0},
    ]
    coverage = all(bool(check["passed"]) for check in checks[:-1])
    return {
        "experiment_id": EXPERIMENT_ID,
        "base_commit": base_commit,
        "execution_status": "engineering_smoke" if smoke else "formal_run",
        "scientific_status": "not run / 尚未运行" if smoke else ("finite-step validated / 有限训练步数验证" if coverage else "not run / 尚未运行"),
        "coverage_checks_passed": coverage,
        "all_checks_passed": all(bool(check["passed"]) for check in checks),
        "checks": checks,
        "event_counts": {
            "task_performance_collapse": sum(bool(r["task_performance_collapse_event"]) for r in summaries),
            "support_or_variance_boundary": sum(bool(r["support_or_variance_boundary_event"]) for r in summaries),
            "nan_inf_numerical_failure": sum(bool(r["nan_inf_numerical_event"]) for r in summaries),
        },
        "interpretation_boundary": (
            "The primary matched coordinate is per-step raw negative-gradient L2 "
            "before Adam. Adam parameter-update norms are observed, not matched. "
            "This 8000-step experiment cannot establish a steady-state ranking, "
            "a universal winner, OOD generalization, or external-task validity."
        ),
    }


def write_protocol_freeze(
    output_root: Path,
    protocol: BudgetMatchProtocol,
    coefficients: dict[tuple[str, float], float],
) -> None:
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "script_version": SCRIPT_VERSION,
        "protocol": asdict(protocol),
        "primary_budget_coordinate": "stepwise_raw_negative_gradient_l2_before_Adam",
        "reference_schedule": "paired reciprocal-linear actor, same seed and minibatch indices",
        "matching_rule": "detached scalar target_norm/current_method_negative_gradient_norm",
        "global_control": "unweighted negative-gradient direction with the same per-step target norm",
        "Adam_parameter_update_norm": "logged_secondary_not_matched",
        "coefficient_source": "development seeds 0-4, near-retention target 0.75",
        "calibrated_coefficients": [
            {"family": family, "target_retention": retention, "coefficient": value}
            for (family, retention), value in sorted(coefficients.items())
        ],
        "held_out_context_is_same_distribution": True,
        "OOD_claim_allowed": False,
        "long_run_ranking_allowed": False,
        "no_method_winner_assumed": True,
    }
    atomic_json(output_root / "formal_protocol_freeze.json", payload)


def checkpoint_manifest(output_root: Path, completed: list[int], protocol: BudgetMatchProtocol) -> None:
    atomic_json(
        output_root / "checkpoints" / f"seed_block_{completed[-1]}.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "completed_formal_seeds": completed,
            "remaining_formal_seeds": [seed for seed in protocol.formal_seeds if seed not in completed],
            "checkpoint_kind": "runner_progress_index_not_final_scientific_result",
            "partial_summary": "per_seed_runs_partial.csv",
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-commit", required=True)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = args.output_dir.resolve()
    supervisor_owned = {"heartbeat.json", "logs", "provenance_launch", "run_manifest.json", "scientific_run_manifest.json"}
    if output_root.exists():
        unexpected = sorted(path.name for path in output_root.iterdir() if path.name not in supervisor_owned)
        if unexpected:
            raise SystemExit("output directory contains non-supervisor files: " + ", ".join(unexpected))
    output_root.mkdir(parents=True, exist_ok=True)

    global PROTOCOL
    if args.smoke:
        experiment.P = replace(
            experiment.P,
            n_train_states=64,
            n_test_states=64,
            hidden_dim=16,
            positive_batch_states=32,
            positive_steps=4,
            positive_continuation_steps=2,
            lbfgs_max_iter=1,
            positive_polish_min_steps=1,
            positive_polish_max_steps=1,
            positive_polish_check_every=1,
            probe_states=2,
        )
        PROTOCOL = BudgetMatchProtocol(
            development_seeds=(0,),
            formal_seeds=(110,),
            batch_states=32,
            evaluation_interval=1,
            maximum_steps=3,
            minimum_steps_before_stationarity=1,
            stable_windows=2,
            bootstrap_samples=20,
            checkpoint_every_formal_seeds=1,
        )

    experiment.ROOT = output_root
    started = time.time()
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "script_version": SCRIPT_VERSION,
        "base_commit": args.base_commit,
        "result_status": "engineering_smoke" if args.smoke else "running",
        "cu1_protocol": asdict(experiment.P),
        "budget_match_protocol": asdict(PROTOCOL),
        "device": str(experiment.DEVICE),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "start_unix": started,
        "scope": "same-distribution held-out-context; stepwise raw negative-gradient L2 matching; not OOD; not long-run",
    }
    atomic_json(output_root / "experiment_manifest.json", manifest)
    source_snapshot = output_root / "source_snapshot"
    source_snapshot.mkdir(parents=True, exist_ok=True)
    for source in (Path(__file__), Path(near_base.__file__), Path(taper_base.__file__), Path(experiment.__file__), Path(cu1_core.__file__)):
        shutil.copy2(source, source_snapshot / source.name)

    audit = experiment.audit_environment(experiment.make_environment(PROTOCOL.formal_seeds[0]))
    atomic_json(output_root / "environment_audit.json", audit)
    if not audit["passed"]:
        raise SystemExit("shared C-U1 environment audit failed")

    near_protocol = calibration_protocol(PROTOCOL)
    coefficients, calibration = near_base.calibrate_families(output_root, near_protocol)
    write_protocol_freeze(output_root, PROTOCOL, coefficients)

    summaries: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []
    positive_rows: list[dict[str, Any]] = []
    completed: list[int] = []
    for index, seed in enumerate(PROTOCOL.formal_seeds, start=1):
        log_message(output_root, f"formal seed={seed} positive-only initialization")
        _, environment, _, positive_summary = experiment.train_positive(seed)
        positive_rows.append({**positive_summary, "method_initialization_source": "positive_only_adam_2000_step_checkpoint"})
        initial_state = copy.deepcopy(experiment.load_initialization_state(seed))

        reference_coefficient = coefficient_for_method("reciprocal_linear", coefficients, PROTOCOL)
        reference_summary, reference_trajectory, target_schedule = _train_one(
            seed=seed,
            method="reciprocal_linear",
            initial_state=initial_state,
            environment=environment,
            coefficient=reference_coefficient,
            protocol=PROTOCOL,
            output_root=output_root,
            target_schedule=None,
        )
        summaries.append(reference_summary)
        trajectories.extend(reference_trajectory)
        for method in method_names(PROTOCOL):
            if method == "reciprocal_linear":
                continue
            coefficient = coefficient_for_method(method, coefficients, PROTOCOL)
            summary, rows, _ = _train_one(
                seed=seed,
                method=method,
                initial_state=initial_state,
                environment=environment,
                coefficient=coefficient,
                protocol=PROTOCOL,
                output_root=output_root,
                target_schedule=(target_schedule if method in PROTOCOL.matched_methods else None),
            )
            summaries.append(summary)
            trajectories.extend(rows)
            write_csv(output_root / "per_seed_runs_partial.csv", summaries)
        completed.append(seed)
        if index % PROTOCOL.checkpoint_every_formal_seeds == 0 or index == len(PROTOCOL.formal_seeds):
            checkpoint_manifest(output_root, completed, PROTOCOL)

    write_csv(output_root / "positive_summary.csv", positive_rows)
    write_csv(output_root / "per_seed_runs.csv", summaries)
    write_csv(output_root / "all_trajectories.csv", trajectories)
    aggregate_rows, paired_rows, paired_summary = aggregate_results(summaries, output_root, PROTOCOL)
    terminal_audit = build_terminal_audit(summaries, PROTOCOL, args.base_commit, args.smoke)
    atomic_json(output_root / "terminal_audit.json", terminal_audit)
    atomic_json(
        output_root / "budget_audit.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "primary_budget_coordinate": "stepwise_raw_negative_gradient_l2_before_Adam",
            "maximum_relative_error": max(
                (float(row["maximum_budget_relative_error"]) for row in summaries if row["method"] in PROTOCOL.matched_methods),
                default=0.0,
            ),
            "tolerance": PROTOCOL.budget_relative_tolerance,
            "Adam_parameter_update_norm_matched": False,
            "Adam_parameter_update_norm_logged": True,
        },
    )
    manifest.update(
        {
            "result_status": "engineering_smoke" if args.smoke else terminal_audit["scientific_status"],
            "end_unix": time.time(),
            "elapsed_seconds": time.time() - started,
            "runs_completed": len(summaries),
            "paired_rows": len(paired_rows),
        }
    )
    atomic_json(output_root / "experiment_manifest.json", manifest)
    atomic_json(
        output_root / "RUN_COMPLETE.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "base_commit": args.base_commit,
            "result_status": manifest["result_status"],
            "coverage_checks_passed": terminal_audit["coverage_checks_passed"],
            "aggregate_rows": len(aggregate_rows),
            "formal_run_started": not args.smoke,
        },
    )
    log_message(output_root, "run complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
