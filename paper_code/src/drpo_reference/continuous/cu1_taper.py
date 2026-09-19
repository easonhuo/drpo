"""C-U1 distance-taper comparison and terminal audit.

The implementation keeps the C-U1 environment and initialization,
advantages, optimizer, and minibatch stream fixed. Methods differ only in the
detached negative-sample weight applied to standardized action distance.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from drpo_reference.common import seed_all
from drpo_reference.controls import (
    TaperFamily,
    point_retention_coefficient,
    taper_weight,
)

from .cu1 import (
    CU1Protocol,
    Environment,
    Split,
    actor_log_prob,
    evaluation,
    positive_loss,
)
from .cu1_training import (
    CU1PositiveProtocol,
    field_diagnostics,
    finite_model,
    initialized_actor,
    make_adam,
    sample_ids,
)
from .gaussian import GaussianActor, standardized_distance


@dataclass(frozen=True)
class CU1TaperProtocol:
    """C-U1 taper settings."""

    seeds: tuple[int, ...] = tuple(range(70, 90))
    reference_distance: float = 5.0
    primary_retention: float = 0.25
    sensitivity_retentions: tuple[float, ...] = (0.50, 0.75)
    negative_alpha: float = 1.0
    learning_rate: float = 5e-4
    batch_states: int = 256
    evaluation_interval: int = 100
    minimum_steps: int = 1000
    maximum_steps: int = 8000
    stable_windows: int = 10
    normalized_slope_threshold: float = 1e-4
    normalized_field_residual_threshold: float = 2e-3
    positive_absolute_gradient_threshold: float = 1e-3
    task_failure_retention: float = 0.45



_TAPER_FAMILIES = {
    "positive_only": TaperFamily.POSITIVE_ONLY,
    "unweighted": TaperFamily.UNCONTROLLED,
    "reciprocal_linear": TaperFamily.RECIPROCAL_LINEAR,
    "reciprocal_quadratic": TaperFamily.RECIPROCAL_QUADRATIC,
    "exponential": TaperFamily.EXPONENTIAL_LINEAR,
}


def method_configs(protocol: CU1TaperProtocol) -> list[tuple[str, float]]:
    families = ("reciprocal_linear", "reciprocal_quadratic", "exponential")
    retentions = (protocol.primary_retention, *protocol.sensitivity_retentions)
    return [
        ("positive_only", 1.0),
        ("unweighted", 1.0),
        *((family, retention) for retention in retentions for family in families),
    ]


def weighted_negative_loss(
    actor: GaussianActor,
    split: Split,
    protocol: CU1Protocol,
    taper: CU1TaperProtocol,
    ids: torch.Tensor | None,
    *,
    family: str,
    retention: float,
) -> torch.Tensor:
    states = split.s if ids is None else split.s[ids]
    actions = split.negative_actions if ids is None else split.negative_actions[ids]
    advantages = split.negative_advantages if ids is None else split.negative_advantages[ids]
    log_probability, mu, log_std = actor_log_prob(actor, states, actions, protocol)
    distance = standardized_distance(mu, log_std, actions)
    canonical = _TAPER_FAMILIES[family]
    if canonical in {TaperFamily.POSITIVE_ONLY, TaperFamily.UNCONTROLLED}:
        weight = taper_weight(distance, family=canonical, detach_distance=True)
    else:
        coefficient = point_retention_coefficient(
            canonical,
            retention=retention,
            reference_distance=taper.reference_distance,
        )
        weight = taper_weight(
            distance,
            family=canonical,
            coefficient=coefficient,
            detach_distance=True,
        )
    return -(advantages * weight * log_probability).mean()


def full_field_diagnostics(
    actor: GaussianActor,
    split: Split,
    protocol: CU1Protocol,
    taper: CU1TaperProtocol,
    *,
    family: str,
    retention: float,
) -> dict[str, Any]:
    negative = (
        None
        if family == "positive_only"
        else weighted_negative_loss(
            actor,
            split,
            protocol,
            taper,
            None,
            family=family,
            retention=retention,
        )
    )
    return field_diagnostics(
        positive_loss(actor, split, protocol),
        negative,
        actor.all_parameters(),
        alpha=taper.negative_alpha,
    )


def max_normalized_slope(
    rows: deque[tuple[float, float, float, float]],
) -> float:
    values = np.asarray(rows, dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        return float("inf")
    slopes = np.polyfit(values[:, 0], values[:, 1:], 1)[0]
    scales = np.maximum(np.mean(np.abs(values[:, 1:]), axis=0), 1e-8)
    return float(np.max(np.abs(slopes) / scales))


def evaluate_taper_state(
    actor: GaussianActor,
    environment: Environment,
    protocol: CU1Protocol,
    taper: CU1TaperProtocol,
    *,
    family: str,
    retention: float,
    initial_reward: float,
) -> dict[str, Any]:
    task = evaluation(actor, environment.test, protocol)
    field = full_field_diagnostics(
        actor,
        environment.train,
        protocol,
        taper,
        family=family,
        retention=retention,
    )
    finite_parameters = finite_model(actor)
    support = evaluation(actor, environment.train, protocol)
    numerical = bool(
        not finite_parameters
        or not support["log_sigma_output_finite"]
        or not support["sigma_output_finite"]
    )
    boundary = bool(
        support["support_contraction_boundary"] or support["unexpected_support_expansion_boundary"]
    )
    task_failure = bool(float(task["reward"]) < taper.task_failure_retention * initial_reward)
    return {
        **task,
        **field,
        "task_performance_collapse_event": task_failure,
        "support_or_variance_boundary_event": boundary,
        "nan_inf_numerical_event": numerical,
    }


def run_taper_method(
    *,
    seed: int,
    initialization_state: dict[str, torch.Tensor],
    environment: Environment,
    protocol: CU1Protocol,
    positive_training: CU1PositiveProtocol | None = None,
    taper: CU1TaperProtocol | None = None,
    family: str,
    retention: float,
) -> dict[str, Any]:
    """Run one taper branch from the exact positive-only Adam checkpoint."""

    positive_training = CU1PositiveProtocol() if positive_training is None else positive_training
    taper = CU1TaperProtocol() if taper is None else taper
    seed_all(seed + 900_000)
    actor = initialized_actor(protocol, environment, initialization_state)
    optimizer = make_adam(
        actor.all_parameters(),
        learning_rate=taper.learning_rate,
        training=positive_training,
    )
    index_generator = torch.Generator(device="cpu").manual_seed(seed + 700_003)
    initial_reward = float(evaluation(actor, environment.test, protocol)["reward"])
    window: deque[tuple[float, float, float, float]] = deque(maxlen=taper.stable_windows)
    completed_steps = 0
    stable_candidate_step: int | None = None
    audit_target_step: int | None = None
    candidate_classification: tuple[bool, bool, bool] | None = None
    stop_reason = "maximum_steps"

    for step in range(1, taper.maximum_steps + 1):
        ids = sample_ids(index_generator, environment.train, taper.batch_states)
        positive = positive_loss(actor, environment.train, protocol, ids)
        if family == "positive_only":
            loss = positive
        else:
            negative = weighted_negative_loss(
                actor,
                environment.train,
                protocol,
                taper,
                ids,
                family=family,
                retention=retention,
            )
            loss = positive + taper.negative_alpha * negative
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        should_evaluate = (
            step == 1
            or step % taper.evaluation_interval == 0
            or step == taper.maximum_steps
            or (audit_target_step is not None and step >= audit_target_step)
        )
        if not should_evaluate:
            continue
        state = evaluate_taper_state(
            actor,
            environment,
            protocol,
            taper,
            family=family,
            retention=retention,
            initial_reward=initial_reward,
        )
        completed_steps = step
        window.append(
            (
                float(step),
                float(state["reward"]),
                float(state["normalized_extrapolation_displacement"]),
                float(state["sigma_mean"]),
            )
        )
        if state["nan_inf_numerical_event"]:
            stop_reason = "nan_inf_numerical_event"
            break
        if state["support_or_variance_boundary_event"]:
            stop_reason = "support_or_variance_boundary_event"
            break
        if (
            stable_candidate_step is None
            and step >= taper.minimum_steps
            and len(window) >= taper.stable_windows
        ):
            max_slope = max_normalized_slope(window)
            residual = float(state["stationarity_residual"])
            residual_threshold = (
                taper.positive_absolute_gradient_threshold
                if family == "positive_only"
                else taper.normalized_field_residual_threshold
            )
            if max_slope < taper.normalized_slope_threshold and residual < residual_threshold:
                stable_candidate_step = step
                target = 2 * step
                audit_target_step = target if target <= taper.maximum_steps else None
                if audit_target_step is not None:
                    candidate_classification = (
                        bool(state["task_performance_collapse_event"]),
                        bool(state["support_or_variance_boundary_event"]),
                        bool(state["nan_inf_numerical_event"]),
                    )
        if audit_target_step is not None and step >= audit_target_step:
            terminal_classification = (
                bool(state["task_performance_collapse_event"]),
                bool(state["support_or_variance_boundary_event"]),
                bool(state["nan_inf_numerical_event"]),
            )
            stop_reason = (
                "stable_plateau_2x_confirmed"
                if terminal_classification == candidate_classification
                else "terminal_classification_reversed"
            )
            break

    final_state = evaluate_taper_state(
        actor,
        environment,
        protocol,
        taper,
        family=family,
        retention=retention,
        initial_reward=initial_reward,
    )
    summary: dict[str, Any] = {
        "seed": seed,
        "family": family,
        "rho": retention,
        "steps_completed": completed_steps,
        "stop_reason": stop_reason,
        "stable_candidate_step": stable_candidate_step,
        "audit_target_step": audit_target_step,
        "initial_reward": initial_reward,
        **final_state,
    }
    return summary
