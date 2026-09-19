"""C-U1 strength scans and stationary audits."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from .cu1 import (
    CU1Protocol,
    Environment,
    evaluation,
    local_negative_loss,
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


@dataclass(frozen=True)
class CU1PhaseProtocol:
    """C-U1 strength-scan settings."""

    fixed_alphas: tuple[float, ...] = (
        0.0,
        0.25,
        0.50,
        0.75,
        1.00,
        1.25,
        1.50,
        1.75,
    )
    learnable_alphas: tuple[float, ...] = (
        0.0,
        0.10,
        0.20,
        0.30,
        0.35,
        0.38,
        0.40,
        0.50,
    )
    learning_rate: float = 5e-4
    warm_steps: int = 200
    continuation_steps: int = 200
    runaway_steps: int = 4000
    normalized_residual_threshold: float = 2e-3
    absolute_residual_threshold_alpha_zero: float = 1e-3
    control_alpha_local: float = 1.0
    control_lambda_far: float = 1.0
    control_far_cap_ratio: float = 0.05
    control_learning_rate: float = 5e-4
    control_steps: int = 4000
    seeds: tuple[int, ...] = tuple(range(50, 70))

def analytic_positive_sigma(protocol: CU1Protocol) -> float:
    residual_second_moment = protocol.positive_contour_radius**2 - protocol.gap_to_unseen_optimum**2
    return math.sqrt(residual_second_moment / protocol.action_dim)

def analytic_local_solution(
    protocol: CU1Protocol,
    alpha: float,
) -> dict[str, float | bool]:
    positive = (
        math.exp(-0.5 * (protocol.positive_contour_radius / protocol.reward_width) ** 2)
        - protocol.baseline
    )
    negative = alpha * abs(
        math.exp(-0.5 * (protocol.negative_contour_radius / protocol.reward_width) ** 2)
        - protocol.baseline
    )
    if negative >= positive:
        return {"finite_mean_fixed_point": False}
    displacement = negative * protocol.negative_offset_from_positive / (positive - negative)
    normalized = displacement / protocol.gap_to_unseen_optimum
    reward = math.exp(
        -0.5 * ((protocol.gap_to_unseen_optimum - displacement) / protocol.reward_width) ** 2
    )
    residual = protocol.positive_contour_radius**2 - protocol.gap_to_unseen_optimum**2
    positive_moment = residual + displacement**2
    negative_moment = (protocol.negative_offset_from_positive + displacement) ** 2
    sigma_squared = (positive * positive_moment - negative * negative_moment) / (
        protocol.action_dim * (positive - negative)
    )
    return {
        "finite_mean_fixed_point": True,
        "analytic_normalized_extrapolation_displacement": normalized,
        "analytic_reward": reward,
        "analytic_sigma": (math.sqrt(sigma_squared) if sigma_squared > 0.0 else float("nan")),
        "finite_variance_fixed_point": sigma_squared > 0.0,
    }

def run_phase_scan(
    *,
    seed: int,
    initialization_state: dict[str, torch.Tensor],
    environment: Environment,
    protocol: CU1Protocol,
    positive_training: CU1PositiveProtocol | None = None,
    phase: CU1PhaseProtocol | None = None,
    alpha: float,
    fixed_sigma: float | None,
    branch: str,
) -> dict[str, Any]:
    """Run one C-U1 local-strength branch and both stationary checks."""

    positive_training = CU1PositiveProtocol() if positive_training is None else positive_training
    phase = CU1PhaseProtocol() if phase is None else phase
    actor = initialized_actor(protocol, environment, initialization_state)
    parameters = actor.mean_parameters() if fixed_sigma is not None else actor.all_parameters()
    optimizer = make_adam(
        parameters,
        learning_rate=phase.learning_rate,
        training=positive_training,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed + 400009)
    analytic = analytic_local_solution(protocol, alpha)
    finite_internal = bool(analytic.get("finite_mean_fixed_point", False)) and (
        fixed_sigma is not None or bool(analytic.get("finite_variance_fixed_point", False))
    )
    first_phase_steps = phase.warm_steps if finite_internal else phase.runaway_steps
    support_onset: int | None = None
    first_support_event_type: str | None = None
    stop_reason = "completed"

    def adam_phase(number_of_steps: int, start_step: int) -> int:
        nonlocal support_onset, first_support_event_type, stop_reason
        completed = 0
        for offset in range(1, number_of_steps + 1):
            step = start_step + offset
            ids = sample_ids(
                generator,
                environment.train,
                positive_training.positive_batch_states,
            )
            loss = positive_loss(
                actor,
                environment.train,
                protocol,
                ids,
                fixed_sigma,
            ) + alpha * local_negative_loss(
                actor,
                environment.train,
                protocol,
                ids,
                fixed_sigma,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            completed = offset
            finite = finite_model(actor)
            post_support = (
                evaluation(actor, environment.train, protocol) if fixed_sigma is None else {}
            )
            event_type = post_support["support_event_type"] if fixed_sigma is None else None
            if event_type is not None and support_onset is None:
                support_onset = step
                first_support_event_type = event_type
            if not finite:
                stop_reason = "non_finite_parameter"
            elif event_type is not None:
                stop_reason = f"{event_type}_boundary_event"
            if not finite or event_type is not None:
                break
        return completed

    completed_first = adam_phase(first_phase_steps, 0)
    audit_1_ok = False
    audit_2_ok = False
    audit_1_residual = float("nan")
    audit_2_residual = float("nan")
    residual_key = "total_gradient_norm" if alpha == 0.0 else "normalized_field_residual"
    threshold = (
        phase.absolute_residual_threshold_alpha_zero
        if alpha == 0.0
        else phase.normalized_residual_threshold
    )

    def current_field() -> dict[str, float]:
        return field_diagnostics(
            positive_loss(actor, environment.train, protocol, fixed_sigma=fixed_sigma),
            local_negative_loss(actor, environment.train, protocol, fixed_sigma=fixed_sigma),
            parameters,
            alpha=alpha,
        )

    def stationary_residual() -> float:
        return float(current_field()[residual_key])

    if finite_internal and finite_model(actor) and support_onset is None:
        audit_1_residual = stationary_residual()
        audit_1_ok = audit_1_residual < threshold
        adam_phase(
            phase.continuation_steps,
            completed_first,
        )
        if finite_model(actor) and support_onset is None:
            audit_2_residual = stationary_residual()
            audit_2_ok = audit_2_residual < threshold

    final = evaluation(actor, environment.test, protocol, fixed_sigma)
    field = current_field()
    stable = (
        finite_internal
        and finite_model(actor)
        and audit_1_ok
        and audit_2_ok
        and support_onset is None
    )
    positive_ceiling_reward = math.exp(
        -0.5 * (protocol.gap_to_unseen_optimum / protocol.reward_width) ** 2
    )
    if stable:
        displacement = float(final["normalized_extrapolation_displacement"])
        reward_gain = float(final["reward"]) - positive_ceiling_reward
        if abs(displacement) <= 0.05:
            state = "stable_imitation_ceiling"
        elif reward_gain > 0.01 and displacement <= 1.25:
            state = "stable_beneficial_extrapolation"
        elif float(final["reward"]) < protocol.task_failure_retention * positive_ceiling_reward:
            state = "stable_bad_fixed_point"
        else:
            state = "stable_over_extrapolated_fixed_point"
    elif stop_reason == "non_finite_parameter" or stop_reason.endswith("_boundary_event"):
        state = stop_reason
    else:
        state = "finite_continuing_drift_or_runaway"

    summary: dict[str, Any] = {
        "seed": seed,
        "alpha": alpha,
        "branch": branch,
        **analytic,
        **final,
        **field,
        "stationary_audit_succeeded": audit_1_ok and audit_2_ok,
        "stationary_audit_1_residual": audit_1_residual,
        "stationary_audit_2_residual": audit_2_residual,
        "state_class": state,
        "support_boundary_onset": support_onset,
        "support_event_type": first_support_event_type,
        "stop_reason": stop_reason,
    }
    return summary
