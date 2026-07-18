"""Hopper E7-Q2 correlation, mechanism, and terminal-classification helpers."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from torch import nn

from ..common.io import atomic_json, write_csv
from .hopper_models import SquashedGaussianPolicy

EPS = 1.0e-6


class ActorAuditProtocol(Protocol):
    audit_windows: int
    actor_state_drift_tolerance: float
    actor_update_tolerance: float
    support_boundary_fraction: float
    task_return_drop_threshold: float


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if denominator <= EPS:
        return float("nan")
    return 1.0 - float(np.sum((y_true - y_pred) ** 2)) / denominator


def pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.std(y_true) <= EPS or np.std(y_pred) <= EPS:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def relative_slope(
    rows: Sequence[dict[str, Any]],
    key: str,
    windows: int,
) -> float:
    if len(rows) < windows:
        return float("inf")
    tail = rows[-windows:]
    steps = np.asarray([float(row["step"]) for row in tail], dtype=np.float64)
    values = np.asarray([float(row[key]) for row in tail], dtype=np.float64)
    if not np.all(np.isfinite(values)) or steps[-1] == steps[0]:
        return float("inf")
    slope = float(np.polyfit(steps, values, 1)[0])
    scale = max(float(np.mean(np.abs(values))), 1.0)
    return abs(slope) / scale


def normalized_window_drift(
    rows: Sequence[dict[str, Any]],
    key: str,
    windows: int,
) -> float:
    if len(rows) < windows:
        return float("inf")
    tail = rows[-windows:]
    steps = np.asarray([float(row["step"]) for row in tail], dtype=np.float64)
    values = np.asarray([float(row[key]) for row in tail], dtype=np.float64)
    if (
        not np.all(np.isfinite(steps))
        or not np.all(np.isfinite(values))
        or steps[-1] <= steps[0]
    ):
        return float("inf")
    slope = float(np.polyfit(steps - steps[0], values, 1)[0])
    span = float(steps[-1] - steps[0])
    scale = max(float(np.median(np.abs(values))), 1.0e-3)
    return abs(slope) * span / scale


def _tensor(array: np.ndarray, device: torch.device | str) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.float32, device=device)


def _full_gradient_norm(
    loss: torch.Tensor,
    parameters: Iterable[nn.Parameter],
) -> float:
    parameter_list = list(parameters)
    gradients = torch.autograd.grad(
        loss,
        parameter_list,
        retain_graph=False,
        allow_unused=True,
    )
    total_square = 0.0
    for gradient in gradients:
        if gradient is not None:
            total_square += float(gradient.detach().square().sum().cpu())
    return math.sqrt(total_square)


def match_near_far_indices(
    advantages: np.ndarray,
    distances: np.ndarray,
    negative_indices: np.ndarray,
    near_quantile: float,
    far_quantile: float,
    bins: int,
    max_pairs: int,
    relative_tolerance: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Pair near and far negatives while matching absolute advantage magnitude."""

    negative_distances = distances[negative_indices]
    near_cut = float(np.quantile(negative_distances, near_quantile))
    far_cut = float(np.quantile(negative_distances, far_quantile))
    near_pool = negative_indices[negative_distances <= near_cut]
    far_pool = negative_indices[negative_distances >= far_cut]
    magnitude = np.abs(advantages)
    all_magnitude = magnitude[negative_indices]
    edges = np.unique(
        np.quantile(all_magnitude, np.linspace(0.0, 1.0, bins + 1))
    )
    generator = np.random.default_rng(seed)
    pairs: list[tuple[int, int, float]] = []
    used_near: set[int] = set()
    used_far: set[int] = set()
    for left, right in zip(edges[:-1], edges[1:]):
        near = near_pool[
            (magnitude[near_pool] >= left) & (magnitude[near_pool] <= right)
        ]
        far = far_pool[
            (magnitude[far_pool] >= left) & (magnitude[far_pool] <= right)
        ]
        if len(near) == 0 or len(far) == 0:
            continue
        generator.shuffle(near)
        far_sorted = far[np.argsort(magnitude[far])]
        for near_index in near:
            near_index = int(near_index)
            if near_index in used_near:
                continue
            position = int(
                np.searchsorted(magnitude[far_sorted], magnitude[near_index])
            )
            candidates = far_sorted[
                max(0, position - 4) : min(len(far_sorted), position + 5)
            ]
            candidates = np.asarray(
                [
                    int(value)
                    for value in candidates
                    if int(value) not in used_far
                ],
                dtype=np.int64,
            )
            if len(candidates) == 0:
                continue
            far_index = int(
                candidates[
                    np.argmin(
                        np.abs(magnitude[candidates] - magnitude[near_index])
                    )
                ]
            )
            relative_error = abs(
                float(magnitude[far_index] - magnitude[near_index])
            ) / max(float(magnitude[near_index]), 1.0e-8)
            if relative_error <= relative_tolerance:
                pairs.append((near_index, far_index, relative_error))
                used_near.add(near_index)
                used_far.add(far_index)
    if not pairs:
        raise RuntimeError("No advantage-matched near/far pairs were found")
    generator.shuffle(pairs)
    pairs = pairs[:max_pairs]
    near_indices = np.asarray([pair[0] for pair in pairs], dtype=np.int64)
    far_indices = np.asarray([pair[1] for pair in pairs], dtype=np.int64)
    summary = {
        "near_cut": near_cut,
        "far_cut": far_cut,
        "pairs": len(pairs),
        "mean_relative_advantage_error": float(
            np.mean([pair[2] for pair in pairs])
        ),
        "advantage_magnitude_far_near_ratio": float(
            np.mean(magnitude[far_indices])
            / max(np.mean(magnitude[near_indices]), EPS)
        ),
        "distance_far_near_ratio": float(
            np.mean(distances[far_indices])
            / max(np.mean(distances[near_indices]), EPS)
        ),
    }
    return near_indices, far_indices, summary


def per_sample_gradient_norm(
    policy: SquashedGaussianPolicy,
    observations: np.ndarray,
    actions: np.ndarray,
    advantages: np.ndarray,
    indices: np.ndarray,
    device: torch.device | str,
) -> np.ndarray:
    norms: list[float] = []
    policy.eval()
    parameters = [
        parameter
        for parameter in policy.parameters()
        if parameter.requires_grad
    ]
    for index in indices:
        observation = _tensor(observations[index : index + 1], device)
        action = _tensor(actions[index : index + 1], device)
        advantage = _tensor(advantages[index : index + 1], device)
        loss = -(advantage * policy.log_prob(observation, action)).mean()
        norms.append(_full_gradient_norm(loss, parameters))
    return np.asarray(norms, dtype=np.float64)


def loglog_slope(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if int(mask.sum()) < 3:
        return float("nan")
    return float(np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)[0])


def analytic_output_autograd_relative_error(
    policy: SquashedGaussianPolicy,
    observations: np.ndarray,
    actions: np.ndarray,
    indices: np.ndarray,
    device: torch.device | str,
    max_samples: int = 8,
) -> float:
    errors: list[float] = []
    for index in indices[:max_samples]:
        observation = _tensor(observations[index : index + 1], device)
        action = _tensor(actions[index : index + 1], device)
        latent = policy.inverse_action(action).detach()
        with torch.no_grad():
            mean_initial, log_std_initial = policy.latent_parameters(observation)
        mean = mean_initial.detach().clone().requires_grad_(True)
        log_std = log_std_initial.detach().clone().requires_grad_(True)
        standard_deviation = torch.exp(log_std)
        standardized = (latent - mean) / standard_deviation
        base_log_probability = -0.5 * (
            standardized.square()
            + 2.0 * log_std
            + math.log(2.0 * math.pi)
        ).sum()
        gradient_mean, gradient_log_std = torch.autograd.grad(
            base_log_probability,
            [mean, log_std],
        )
        analytic_mean = (
            latent - mean.detach()
        ) / standard_deviation.detach().square()
        analytic_log_std = standardized.detach().square() - 1.0
        numerator = torch.sqrt(
            (gradient_mean - analytic_mean).square().sum()
            + (gradient_log_std - analytic_log_std).square().sum()
        )
        denominator = torch.sqrt(
            analytic_mean.square().sum() + analytic_log_std.square().sum()
        ).clamp_min(EPS)
        errors.append(float((numerator / denominator).cpu()))
    return float(max(errors)) if errors else float("nan")


def _numpy_score_components(
    policy: SquashedGaussianPolicy,
    observations: np.ndarray,
    actions: np.ndarray,
    indices: np.ndarray,
    device: torch.device | str,
) -> dict[str, np.ndarray]:
    with torch.no_grad():
        values = policy.score_components(
            _tensor(observations[indices], device),
            _tensor(actions[indices], device),
        )
    keys = (
        "radius",
        "mean_score_norm",
        "raw_log_scale_score_norm",
        "corrected_q_xi",
        "joint_output_score_norm",
        "log_scale_to_mean_ratio",
        "raw_action_distance",
        "pre_squash_distance",
    )
    return {
        key: values[key].detach().cpu().numpy()
        for key in keys
    }


def create_gradient_probe(
    *,
    policy: SquashedGaussianPolicy,
    observations: np.ndarray,
    actions: np.ndarray,
    advantages: np.ndarray,
    near_indices: np.ndarray,
    far_indices: np.ndarray,
    population_indices: np.ndarray,
    max_gradient_pairs: int,
    distance_bins: int,
    device: torch.device | str,
    output_dir: Path,
) -> dict[str, Any]:
    """Write matched-pair score and full-parameter gradient diagnostics."""

    output_dir.mkdir(parents=True, exist_ok=True)
    near = _numpy_score_components(
        policy,
        observations,
        actions,
        near_indices,
        device,
    )
    far = _numpy_score_components(
        policy,
        observations,
        actions,
        far_indices,
        device,
    )
    population = _numpy_score_components(
        policy,
        observations,
        actions,
        population_indices,
        device,
    )
    gradient_pairs = min(len(near_indices), max_gradient_pairs)
    near_gradient = per_sample_gradient_norm(
        policy,
        observations,
        actions,
        advantages,
        near_indices[:gradient_pairs],
        device,
    )
    far_gradient = per_sample_gradient_norm(
        policy,
        observations,
        actions,
        advantages,
        far_indices[:gradient_pairs],
        device,
    )
    pair_rows: list[dict[str, Any]] = []
    for pair_id, (near_index, far_index) in enumerate(
        zip(near_indices, far_indices)
    ):
        row: dict[str, Any] = {
            "pair_id": pair_id,
            "near_index": int(near_index),
            "far_index": int(far_index),
            "near_advantage": float(advantages[near_index]),
            "far_advantage": float(advantages[far_index]),
            "near_abs_advantage": float(abs(advantages[near_index])),
            "far_abs_advantage": float(abs(advantages[far_index])),
        }
        for key in near:
            row[f"near_{key}"] = float(near[key][pair_id])
            row[f"far_{key}"] = float(far[key][pair_id])
        if pair_id < gradient_pairs:
            row["near_full_parameter_gradient_norm"] = float(
                near_gradient[pair_id]
            )
            row["far_full_parameter_gradient_norm"] = float(
                far_gradient[pair_id]
            )
        pair_rows.append(row)
    write_csv(output_dir / "matched_near_far_components.csv", pair_rows)

    radius = population["radius"]
    edges = np.unique(
        np.quantile(radius, np.linspace(0.0, 1.0, distance_bins + 1))
    )
    bin_rows: list[dict[str, Any]] = []
    gradient_indices = np.concatenate(
        [near_indices[:gradient_pairs], far_indices[:gradient_pairs]]
    )
    gradient_values = np.concatenate([near_gradient, far_gradient])
    with torch.no_grad():
        gradient_radius = policy.standardized_distance(
            _tensor(observations[gradient_indices], device),
            _tensor(actions[gradient_indices], device),
        ).cpu().numpy()
    for bin_id, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        population_mask = (radius >= left) & (
            radius <= right
            if bin_id == len(edges) - 2
            else radius < right
        )
        if not bool(population_mask.any()):
            continue
        gradient_mask = (gradient_radius >= left) & (
            gradient_radius <= right
            if bin_id == len(edges) - 2
            else gradient_radius < right
        )
        row = {
            "bin": bin_id,
            "radius_left": float(left),
            "radius_right": float(right),
            "count": int(population_mask.sum()),
        }
        for key, values in population.items():
            row[f"{key}_mean"] = float(np.mean(values[population_mask]))
            row[f"{key}_median"] = float(np.median(values[population_mask]))
        row["full_parameter_gradient_norm_mean"] = (
            float(np.mean(gradient_values[gradient_mask]))
            if bool(gradient_mask.any())
            else float("nan")
        )
        row["full_parameter_gradient_count"] = int(gradient_mask.sum())
        bin_rows.append(row)
    write_csv(output_dir / "component_distance_bins.csv", bin_rows)

    autograd_error = analytic_output_autograd_relative_error(
        policy,
        observations,
        actions,
        np.concatenate([near_indices, far_indices]),
        device,
    )
    corrected_slope = loglog_slope(
        radius,
        population["corrected_q_xi"],
    )
    mean_score_slope = loglog_slope(
        radius,
        population["mean_score_norm"],
    )
    action_dimension = int(actions.shape[1])
    natural_far_threshold = math.sqrt(2.0 * action_dimension)
    far_median_radius = float(np.median(far["radius"]))
    summary = {
        "pairs": len(pair_rows),
        "gradient_pairs": gradient_pairs,
        "abs_advantage_far_near_ratio": float(
            np.mean(np.abs(advantages[far_indices]))
            / max(np.mean(np.abs(advantages[near_indices])), EPS)
        ),
        "standardized_distance_far_near_ratio": float(
            np.mean(far["radius"]) / max(np.mean(near["radius"]), EPS)
        ),
        "mean_output_score_far_near_ratio": float(
            np.mean(far["mean_score_norm"])
            / max(np.mean(near["mean_score_norm"]), EPS)
        ),
        "raw_log_scale_output_score_far_near_ratio": float(
            np.mean(far["raw_log_scale_score_norm"])
            / max(np.mean(near["raw_log_scale_score_norm"]), EPS)
        ),
        "corrected_q_xi_far_near_ratio": float(
            np.mean(far["corrected_q_xi"])
            / max(np.mean(near["corrected_q_xi"]), EPS)
        ),
        "joint_output_score_far_near_ratio": float(
            np.mean(far["joint_output_score_norm"])
            / max(np.mean(near["joint_output_score_norm"]), EPS)
        ),
        "log_scale_to_mean_far_near_ratio": float(
            np.mean(far["log_scale_to_mean_ratio"])
            / max(np.mean(near["log_scale_to_mean_ratio"]), EPS)
        ),
        "full_parameter_gradient_far_near_ratio": float(
            np.mean(far_gradient) / max(np.mean(near_gradient), EPS)
        ),
        "mean_score_loglog_slope_vs_radius": mean_score_slope,
        "corrected_q_xi_loglog_slope_vs_radius": corrected_slope,
        "analytic_autograd_relative_error_max": autograd_error,
        "natural_far_threshold_sqrt_2d": natural_far_threshold,
        "far_median_radius": far_median_radius,
        "natural_far_field_present": bool(
            far_median_radius >= natural_far_threshold
        ),
    }
    atomic_json(output_dir / "gradient_probe_summary.json", summary)
    return summary


def aggregate_negative_gradient_norm(
    policy: SquashedGaussianPolicy,
    observations: np.ndarray,
    actions: np.ndarray,
    advantages: np.ndarray,
    indices: np.ndarray,
    device: torch.device | str,
) -> float:
    observation = _tensor(observations[indices], device)
    action = _tensor(actions[indices], device)
    advantage = _tensor(advantages[indices], device)
    loss = -(advantage * policy.log_prob(observation, action)).mean()
    return _full_gradient_norm(loss, policy.parameters())


def resolve_global_scale(
    *,
    policy: SquashedGaussianPolicy,
    observations: np.ndarray,
    actions: np.ndarray,
    advantages: np.ndarray,
    negative_indices: np.ndarray,
    far_threshold: float,
    far_cap_score: float,
    audit_size: int,
    seed: int,
    device: torch.device | str,
) -> dict[str, Any]:
    """Match the initial per-sample negative-influence budget of Far-cap."""

    generator = np.random.default_rng(seed + 991)
    chosen = generator.choice(
        negative_indices,
        size=min(audit_size, len(negative_indices)),
        replace=False,
    )
    with torch.no_grad():
        components = policy.score_components(
            _tensor(observations[chosen], device),
            _tensor(actions[chosen], device),
        )
        distances = components["radius"].cpu().numpy()
        joint_scores = components["joint_output_score_norm"].cpu().numpy()
    per_sample = per_sample_gradient_norm(
        policy,
        observations,
        actions,
        advantages,
        chosen,
        device,
    )
    far_mask = distances > far_threshold
    cap_factor = np.ones_like(per_sample)
    cap_factor[far_mask] = np.minimum(
        1.0,
        far_cap_score / np.maximum(joint_scores[far_mask], EPS),
    )
    all_budget = float(np.sum(per_sample))
    far_cap_budget = float(np.sum(per_sample * cap_factor))
    scale = far_cap_budget / max(all_budget, EPS)
    return {
        "audit_samples": len(chosen),
        "far_samples": int(np.sum(far_mask)),
        "all_negative_per_sample_gradient_norm_sum": all_budget,
        "far_cap_per_sample_gradient_norm_sum": far_cap_budget,
        "far_cap_score": far_cap_score,
        "global_scale": float(np.clip(scale, 0.0, 1.0)),
        "matching_rule": (
            "match initial sum of per-sample full-parameter negative-gradient "
            "norms retained by detached Far-cap; no aggregate-vector cancellation"
        ),
    }


def classify_actor_terminal(
    rows: Sequence[dict[str, Any]],
    protocol: ActorAuditProtocol,
    candidate_step: int | None,
    extension_complete: bool,
    fixed_budget_completed: bool = False,
) -> dict[str, Any]:
    last = rows[-1]
    relative_update_norm = float(
        last.get(
            "relative_update_norm",
            last.get("update_norm", float("inf")),
        )
    )
    nonfinite = any(
        not math.isfinite(float(last[key]))
        for key in (
            "loss",
            "positive_nll",
            "gradient_norm",
            "update_norm",
            "sigma_mean",
        )
    ) or not math.isfinite(relative_update_norm)
    support_event = bool(
        float(last["mean_boundary_fraction"])
        >= protocol.support_boundary_fraction
        or float(last["log_std_min_fraction"]) > 0.0
        or float(last["log_std_max_fraction"]) > 0.0
    )
    slopes = {
        key: relative_slope(rows, key, protocol.audit_windows)
        for key in (
            "positive_nll",
            "mean_abs",
            "sigma_mean",
            "phantom_distance_mean",
        )
    }
    state_drifts = {
        key: normalized_window_drift(rows, key, protocol.audit_windows)
        for key in (
            "mean_abs",
            "sigma_mean",
            "phantom_distance_mean",
        )
    }
    stable = bool(
        candidate_step is not None
        and extension_complete
        and all(
            value <= protocol.actor_state_drift_tolerance
            for value in state_drifts.values()
        )
        and relative_update_norm <= protocol.actor_update_tolerance
        and not nonfinite
    )
    rollout_values = [
        float(row.get("normalized_return", float("nan")))
        for row in rows
    ]
    finite_rollouts = [
        value for value in rollout_values if math.isfinite(value)
    ]
    rollout_statuses = {
        str(row.get("rollout_status", "not_evaluated"))
        for row in rows
    }
    initial_return = finite_rollouts[0] if finite_rollouts else float("nan")
    final_return = finite_rollouts[-1] if finite_rollouts else float("nan")
    if finite_rollouts:
        task_status = "available"
        task_collapse: bool | None = bool(
            initial_return - final_return
            >= protocol.task_return_drop_threshold
        )
    elif "unavailable" in rollout_statuses:
        task_status = "unavailable"
        task_collapse = None
    elif rollout_statuses == {"disabled"}:
        task_status = "disabled"
        task_collapse = None
    else:
        task_status = "not_evaluated"
        task_collapse = None

    if nonfinite:
        state = "nan_inf_numerical_collapse"
    elif stable and support_event:
        state = "finite_terminal_with_support_boundary_event"
    elif stable:
        state = "finite_terminal"
    elif support_event:
        state = (
            "support_or_variance_boundary_event_without_terminal_convergence"
        )
    elif len(rows) >= protocol.audit_windows and any(
        value > protocol.actor_state_drift_tolerance
        for value in state_drifts.values()
    ):
        state = "persistent_or_slow_drift"
    elif fixed_budget_completed:
        state = "fixed_horizon_inconclusive"
    else:
        state = "training_incomplete_without_terminal_classification"
    explicit = state != "training_incomplete_without_terminal_classification"
    return {
        "state": state,
        "candidate_step": candidate_step,
        "extension_complete": extension_complete,
        "fixed_budget_completed": fixed_budget_completed,
        "terminal_audit_controls_stopping": False,
        "slopes": slopes,
        "state_drifts": state_drifts,
        "state_drift_tolerance": protocol.actor_state_drift_tolerance,
        "relative_update_norm": relative_update_norm,
        "support_boundary_event": support_event,
        "numerical_nonfinite": nonfinite,
        "task_performance_status": task_status,
        "task_performance_collapse": task_collapse,
        "normalized_return_available": task_status == "available",
        "initial_normalized_return": initial_return,
        "final_normalized_return": final_return,
        "task_return_drop_threshold": protocol.task_return_drop_threshold,
        "explicit_terminal_classification": explicit,
        "reporting_separation": [
            "task_performance_status_and_collapse",
            "support_or_variance_boundary_event",
            "nan_inf_numerical_collapse",
        ],
    }
