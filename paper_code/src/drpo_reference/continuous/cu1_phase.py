"""C-U1 strength scans and stationary audits."""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .cu1 import (
    CU1Protocol,
    Environment,
    Split,
    evaluation,
    local_negative_loss,
    positive_loss,
    support_diagnostics,
)
from .cu1_mechanism import _support_event_type as support_event_type
from .cu1_training import (
    CU1PositiveProtocol,
    finite_model,
    make_adam,
    normalized_field_residual,
    optimizer_step_with_norm,
)
from .gaussian import GaussianActor


@dataclass(frozen=True)
class CU1PhaseProtocol:
    """Frozen E4 strength-scan settings."""

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
    evaluation_interval: int = 100
    normalized_residual_threshold: float = 2e-3
    absolute_residual_threshold_alpha_zero: float = 1e-3
    formal_seeds: tuple[int, ...] = tuple(range(50, 70))

    def __post_init__(self) -> None:
        for name in (
            "warm_steps",
            "continuation_steps",
            "runaway_steps",
            "evaluation_interval",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass
class PhaseRun:
    actor: GaussianActor
    trajectory: list[dict[str, Any]]
    summary: dict[str, Any]


def positive_advantage_value(protocol: CU1Protocol) -> float:
    return (
        math.exp(-0.5 * (protocol.positive_contour_radius / protocol.reward_width) ** 2)
        - protocol.baseline
    )


def negative_advantage_value(protocol: CU1Protocol) -> float:
    return (
        math.exp(-0.5 * (protocol.negative_contour_radius / protocol.reward_width) ** 2)
        - protocol.baseline
    )


def analytic_positive_sigma(protocol: CU1Protocol) -> float:
    residual_second_moment = protocol.positive_contour_radius**2 - protocol.gap_to_unseen_optimum**2
    return math.sqrt(residual_second_moment / protocol.action_dim)


def analytic_mean_critical_alpha(protocol: CU1Protocol) -> float:
    return positive_advantage_value(protocol) / abs(negative_advantage_value(protocol))


def analytic_variance_boundary_alpha(protocol: CU1Protocol) -> float:
    positive = positive_advantage_value(protocol)
    negative = abs(negative_advantage_value(protocol))
    residual = protocol.positive_contour_radius**2 - protocol.gap_to_unseen_optimum**2

    def field(alpha: float) -> float:
        weighted_negative = alpha * negative
        displacement = (
            weighted_negative
            * protocol.negative_offset_from_positive
            / (positive - weighted_negative)
        )
        positive_moment = residual + displacement**2
        negative_moment = (protocol.negative_offset_from_positive + displacement) ** 2
        return positive * positive_moment - weighted_negative * negative_moment

    lower = 0.0
    upper = min(analytic_mean_critical_alpha(protocol) - 1e-8, 1.0)
    for _ in range(100):
        middle = 0.5 * (lower + upper)
        if field(middle) > 0.0:
            lower = middle
        else:
            upper = middle
    return 0.5 * (lower + upper)


def analytic_local_solution(
    protocol: CU1Protocol,
    alpha: float,
) -> dict[str, float | bool]:
    positive = positive_advantage_value(protocol)
    negative = alpha * abs(negative_advantage_value(protocol))
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


def evaluation_from_geometry(
    distance_to_star: float,
    protocol: CU1Protocol,
) -> float:
    return math.exp(-0.5 * (distance_to_star / protocol.reward_width) ** 2)


def local_objective(
    actor: GaussianActor,
    split: Split,
    protocol: CU1Protocol,
    ids: torch.Tensor | None,
    *,
    alpha: float,
    fixed_sigma: float | None,
) -> torch.Tensor:
    return positive_loss(
        actor,
        split,
        protocol,
        ids,
        fixed_sigma,
    ) + alpha * local_negative_loss(
        actor,
        split,
        protocol,
        ids,
        fixed_sigma,
    )


def policy_distance_diagnostics(
    actor: GaussianActor,
    split: Split,
    protocol: CU1Protocol,
    fixed_sigma: float | None,
) -> dict[str, float]:
    actor.eval()
    with torch.no_grad():
        mu, predicted = actor(split.s)
        log_std = (
            predicted if fixed_sigma is None else torch.full_like(predicted, math.log(fixed_sigma))
        )
        raw = torch.linalg.vector_norm(
            split.negative_actions - mu[:, None, :],
            dim=-1,
        )
        standardized = raw / torch.exp(log_std)[:, None]
        near = standardized <= protocol.near_far_standardized_threshold
        return {
            "negative_raw_distance_mean": raw.mean().item(),
            "negative_standardized_distance_mean": standardized.mean().item(),
            "dynamic_near_occupancy": near.float().mean().item(),
            "dynamic_far_occupancy": (~near).float().mean().item(),
            "local_negative_raw_distance": raw[:, 0].mean().item(),
            "local_negative_standardized_distance": (standardized[:, 0].mean().item()),
            "farthest_negative_raw_distance": raw[:, 4].mean().item(),
            "farthest_negative_standardized_distance": (standardized[:, 4].mean().item()),
        }


def _parameter_gradient_norm(parameters: Sequence[torch.nn.Parameter]) -> float:
    gradients = [
        parameter.grad.reshape(-1) for parameter in parameters if parameter.grad is not None
    ]
    if not gradients:
        return 0.0
    return torch.linalg.vector_norm(torch.cat(gradients)).item()


def _new_actor(
    protocol: CU1Protocol,
    environment: Environment,
    initialization_state: dict[str, torch.Tensor],
) -> GaussianActor:
    actor = GaussianActor(
        state_dim=protocol.state_dim,
        action_dim=protocol.action_dim,
        hidden_dim=protocol.hidden_dim,
        initial_sigma=protocol.initial_sigma,
    ).to(environment.train.s.device, dtype=environment.train.s.dtype)
    actor.load_state_dict(copy.deepcopy(initialization_state))
    return actor


def run_phase_scan(
    *,
    seed: int,
    initialization_state: dict[str, torch.Tensor],
    environment: Environment,
    protocol: CU1Protocol,
    positive_training: CU1PositiveProtocol = CU1PositiveProtocol(),
    phase: CU1PhaseProtocol = CU1PhaseProtocol(),
    alpha: float,
    fixed_sigma: float | None,
    branch: str,
) -> PhaseRun:
    """Run one frozen E4 local-strength branch and both stationary audits."""

    actor = _new_actor(protocol, environment, initialization_state)
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
    trajectory: list[dict[str, Any]] = []
    support_onset: int | None = None
    first_support_event_type: str | None = None
    stop_reason = "completed"

    def record(
        step: int,
        stage: str,
        extra: dict[str, Any] | None = None,
    ) -> None:
        row: dict[str, Any] = {
            "step": step,
            "stage": stage,
            "optimizer": "adam",
            **evaluation(actor, environment.test, protocol, fixed_sigma),
            **normalized_field_residual(
                actor,
                environment.train,
                protocol,
                alpha=alpha,
                fixed_sigma=fixed_sigma,
                local_negative=True,
            ),
            **policy_distance_diagnostics(
                actor,
                environment.train,
                protocol,
                fixed_sigma,
            ),
        }
        if fixed_sigma is None:
            row.update(support_diagnostics(actor, environment.train, protocol))
        if extra:
            row.update(extra)
        trajectory.append(row)

    record(0, "initial")

    def adam_phase(number_of_steps: int, start_step: int, stage: str) -> int:
        nonlocal support_onset, first_support_event_type, stop_reason
        completed = 0
        for offset in range(1, number_of_steps + 1):
            step = start_step + offset
            pre_support = (
                support_diagnostics(actor, environment.train, protocol)
                if fixed_sigma is None
                else {}
            )
            ids = torch.randint(
                0,
                protocol.n_train_states,
                (positive_training.positive_batch_states,),
                generator=generator,
            ).to(environment.train.s.device)
            loss = local_objective(
                actor,
                environment.train,
                protocol,
                ids,
                alpha=alpha,
                fixed_sigma=fixed_sigma,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            raw_gradient_norm = _parameter_gradient_norm(parameters)
            parameter_update_norm = optimizer_step_with_norm(
                optimizer,
                parameters,
            )
            completed = offset
            finite = finite_model(actor)
            post_support = (
                support_diagnostics(actor, environment.train, protocol)
                if fixed_sigma is None
                else {}
            )
            event_type = support_event_type(post_support) if fixed_sigma is None else None
            if event_type is not None and support_onset is None:
                support_onset = step
                first_support_event_type = event_type
            extra = {
                "raw_total_gradient_norm": raw_gradient_norm,
                "parameter_update_norm": parameter_update_norm,
            }
            if fixed_sigma is None:
                extra.update({f"pre_{key}": value for key, value in pre_support.items()})
                extra.update({f"post_{key}": value for key, value in post_support.items()})
            if not finite:
                stop_reason = "non_finite_parameter"
            elif event_type is not None:
                stop_reason = f"{event_type}_boundary_event"
            if (
                offset % phase.evaluation_interval == 0
                or offset == number_of_steps
                or event_type is not None
                or not finite
            ):
                record(step, stage, extra)
            if not finite or event_type is not None:
                break
        return completed

    completed_first = adam_phase(first_phase_steps, 0, "adam_phase_1")
    audit_1_ok = False
    audit_2_ok = False
    audit_1_residual = float("nan")
    audit_2_residual = float("nan")
    if finite_internal and finite_model(actor) and support_onset is None:
        first_field = normalized_field_residual(
            actor,
            environment.train,
            protocol,
            alpha=alpha,
            fixed_sigma=fixed_sigma,
            local_negative=True,
        )
        audit_1_residual = (
            first_field["total_gradient_norm"]
            if alpha == 0.0
            else first_field["normalized_field_residual"]
        )
        threshold = (
            phase.absolute_residual_threshold_alpha_zero
            if alpha == 0.0
            else phase.normalized_residual_threshold
        )
        audit_1_ok = audit_1_residual < threshold
        record(
            completed_first,
            "full_data_residual_audit_1",
            {"audit_residual": audit_1_residual},
        )
        completed_second = adam_phase(
            phase.continuation_steps,
            completed_first,
            "adam_continuation",
        )
        if finite_model(actor) and support_onset is None:
            second_field = normalized_field_residual(
                actor,
                environment.train,
                protocol,
                alpha=alpha,
                fixed_sigma=fixed_sigma,
                local_negative=True,
            )
            audit_2_residual = (
                second_field["total_gradient_norm"]
                if alpha == 0.0
                else second_field["normalized_field_residual"]
            )
            audit_2_ok = audit_2_residual < threshold
            record(
                completed_first + completed_second,
                "full_data_residual_audit_2",
                {"audit_residual": audit_2_residual},
            )

    final = evaluation(actor, environment.test, protocol, fixed_sigma)
    field = normalized_field_residual(
        actor,
        environment.train,
        protocol,
        alpha=alpha,
        fixed_sigma=fixed_sigma,
        local_negative=True,
    )
    stable = (
        finite_internal
        and finite_model(actor)
        and audit_1_ok
        and audit_2_ok
        and support_onset is None
    )
    positive_ceiling_reward = evaluation_from_geometry(
        protocol.gap_to_unseen_optimum,
        protocol,
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

    displacement_slope = float("nan")
    log_sigma_slope = float("nan")
    dynamic_rows = [
        row for row in trajectory if row.get("stage") in {"adam_phase_1", "adam_continuation"}
    ]
    by_step = {int(row["step"]): row for row in dynamic_rows}
    ordered = [by_step[step] for step in sorted(by_step)]
    if len(ordered) >= 3:
        tail = ordered[-min(5, len(ordered)) :]
        steps_array = np.asarray([float(row["step"]) for row in tail])
        if np.ptp(steps_array) > 0.0:
            displacement_slope = float(
                np.polyfit(
                    steps_array,
                    np.asarray(
                        [float(row["normalized_extrapolation_displacement"]) for row in tail]
                    ),
                    1,
                )[0]
            )
            log_sigma_slope = float(
                np.polyfit(
                    steps_array,
                    np.log(
                        np.maximum(
                            np.asarray([float(row["sigma_mean"]) for row in tail]),
                            1e-30,
                        )
                    ),
                    1,
                )[0]
            )

    summary: dict[str, Any] = {
        "seed": seed,
        "alpha": alpha,
        "branch": branch,
        "optimizer": "adam",
        **analytic,
        **final,
        **field,
        **policy_distance_diagnostics(
            actor,
            environment.train,
            protocol,
            fixed_sigma,
        ),
        "stationary_audit_attempted": finite_internal,
        "stationary_audit_1_succeeded": audit_1_ok,
        "stationary_audit_2_succeeded": audit_2_ok,
        "stationary_audit_succeeded": audit_1_ok and audit_2_ok,
        "stationary_audit_1_residual": audit_1_residual,
        "stationary_audit_2_residual": audit_2_residual,
        "state_class": state,
        "support_boundary_onset": support_onset,
        "support_event_type": first_support_event_type,
        "unexpected_support_expansion": (
            first_support_event_type == "unexpected_support_expansion"
        ),
        "stop_reason": stop_reason,
        "normalized_extrapolation_displacement_window_slope": (displacement_slope),
        "log_sigma_window_slope": log_sigma_slope,
    }
    return PhaseRun(actor=actor, trajectory=trajectory, summary=summary)
