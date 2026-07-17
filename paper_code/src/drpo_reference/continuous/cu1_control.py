"""C-U1 far-pressure controls for the phase experiment."""

from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass
from typing import Any

import torch

from .cu1 import (
    CU1Protocol,
    Environment,
    Split,
    actor_log_prob,
    evaluation,
    local_negative_loss,
    positive_loss,
)
from .cu1_phase import analytic_positive_sigma
from .cu1_training import (
    EPS,
    CU1PositiveProtocol,
    add_gradients,
    finite_model,
    gradient_norm,
    make_adam,
    optimizer_step_with_norm,
    scale_gradients,
    set_parameter_gradients,
)
from .gaussian import GaussianActor

GradientTuple = tuple[torch.Tensor | None, ...]


@dataclass(frozen=True)
class CU1ControlProtocol:
    """Frozen E4 local/far pressure control settings."""

    alpha_local: float = 1.0
    lambda_far: float = 1.0
    far_cap_ratio: float = 0.05
    learning_rate: float = 5e-4
    steps: int = 4000
    evaluation_interval: int = 100
    formal_seeds: tuple[int, ...] = tuple(range(50, 70))

    def __post_init__(self) -> None:
        if self.steps <= 0 or self.evaluation_interval <= 0:
            raise ValueError("control budgets must be positive")
        if self.alpha_local < 0.0 or self.lambda_far < 0.0:
            raise ValueError("control strengths must be non-negative")
        if not 0.0 <= self.far_cap_ratio <= 1.0:
            raise ValueError("far_cap_ratio must lie in [0, 1]")


@dataclass
class ControlRun:
    actor: GaussianActor
    trajectory: list[dict[str, Any]]
    summary: dict[str, Any]


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


def control_gradients(
    actor: GaussianActor,
    split: Split,
    protocol: CU1Protocol,
    control: CU1ControlProtocol,
    ids: torch.Tensor,
    *,
    method: str,
    fixed_sigma: float,
) -> tuple[GradientTuple, dict[str, float]]:
    """Return the registered E4 local/far control gradient."""

    parameters = actor.mean_parameters()
    positive = positive_loss(actor, split, protocol, ids, fixed_sigma)
    local = local_negative_loss(actor, split, protocol, ids, fixed_sigma)
    states = split.s[ids]
    actions = split.negative_actions[ids, 1:]
    advantages = split.negative_advantages[ids, 1:]
    far_log_probability, _, _ = actor_log_prob(
        actor,
        states,
        actions,
        protocol,
        fixed_sigma,
    )
    far = -(advantages * far_log_probability).mean()
    positive_gradient = torch.autograd.grad(
        positive,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    local_gradient = torch.autograd.grad(
        local,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    far_gradient = torch.autograd.grad(
        far,
        parameters,
        allow_unused=True,
    )
    weighted_local = scale_gradients(
        local_gradient,
        control.alpha_local,
    )
    weighted_far = scale_gradients(
        far_gradient,
        control.lambda_far,
    )
    raw_negative = add_gradients(weighted_local, weighted_far)
    local_norm = gradient_norm(weighted_local).item()
    far_norm = gradient_norm(weighted_far).item()
    raw_norm = gradient_norm(raw_negative).item()
    far_scale = min(
        1.0,
        control.far_cap_ratio * local_norm / (far_norm + EPS),
    )
    capped_negative = add_gradients(
        weighted_local,
        scale_gradients(weighted_far, far_scale),
    )
    capped_norm = gradient_norm(capped_negative).item()
    if method == "uncontrolled_all":
        controlled_negative = raw_negative
    elif method == "far_cap":
        controlled_negative = capped_negative
    elif method == "budget_matched_global":
        controlled_negative = scale_gradients(
            raw_negative,
            capped_norm / (raw_norm + EPS),
        )
    else:
        raise ValueError(f"unknown E4 control method: {method}")
    total = add_gradients(positive_gradient, controlled_negative)
    return total, {
        "positive_gradient_norm": gradient_norm(positive_gradient).item(),
        "local_negative_gradient_norm": local_norm,
        "far_negative_gradient_norm": far_norm,
        "raw_negative_gradient_norm": raw_norm,
        "post_control_negative_gradient_norm": gradient_norm(
            controlled_negative
        ).item(),
        "far_scale": far_scale,
        "total_update_norm": gradient_norm(total).item(),
    }


def run_far_pressure_control(
    *,
    seed: int,
    initialization_state: dict[str, torch.Tensor],
    environment: Environment,
    protocol: CU1Protocol,
    positive_training: CU1PositiveProtocol = CU1PositiveProtocol(),
    control: CU1ControlProtocol = CU1ControlProtocol(),
    method: str,
) -> ControlRun:
    """Run one registered E4 far-pressure control branch."""

    actor = _new_actor(protocol, environment, initialization_state)
    parameters = actor.mean_parameters()
    optimizer = make_adam(
        parameters,
        learning_rate=control.learning_rate,
        training=positive_training,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed + 500009)
    fixed_sigma = analytic_positive_sigma(protocol)
    trajectory: list[dict[str, Any]] = []
    task_threshold = protocol.task_failure_retention * float(
        evaluation(actor, environment.test, protocol, fixed_sigma)["reward"]
    )
    below_threshold: deque[int] = deque(
        maxlen=protocol.task_failure_consecutive_evals
    )
    task_onset: int | None = None
    nonfinite_onset: int | None = None

    for step in range(1, control.steps + 1):
        ids = torch.randint(
            0,
            protocol.n_train_states,
            (positive_training.positive_batch_states,),
            generator=generator,
        ).to(environment.train.s.device)
        gradients, diagnostics = control_gradients(
            actor,
            environment.train,
            protocol,
            control,
            ids,
            method=method,
            fixed_sigma=fixed_sigma,
        )
        optimizer.zero_grad(set_to_none=True)
        set_parameter_gradients(parameters, gradients)
        diagnostics["raw_total_gradient_norm"] = gradient_norm(gradients).item()
        diagnostics["parameter_update_norm"] = optimizer_step_with_norm(
            optimizer,
            parameters,
        )
        diagnostics["optimizer"] = "adam"
        if not finite_model(actor):
            nonfinite_onset = step
            break
        if (
            step % control.evaluation_interval == 0
            or step == 1
            or step == control.steps
        ):
            metrics = evaluation(
                actor,
                environment.test,
                protocol,
                fixed_sigma,
            )
            if float(metrics["reward"]) < task_threshold:
                below_threshold.append(step)
            else:
                below_threshold.clear()
            if (
                len(below_threshold)
                == protocol.task_failure_consecutive_evals
                and task_onset is None
            ):
                task_onset = below_threshold[0]
            trajectory.append(
                {
                    "step": step,
                    "method": method,
                    **metrics,
                    **diagnostics,
                    "task_threshold": task_threshold,
                }
            )

    if finite_model(actor):
        final = evaluation(actor, environment.test, protocol, fixed_sigma)
    else:
        final = {
            key: float("nan")
            for key in (
                "reward",
                "normalized_extrapolation_displacement",
                "distance_to_a_plus",
                "distance_to_a_star",
                "sigma_mean",
                "sigma_min",
                "sigma_max",
                "log_sigma_min",
                "log_sigma_max",
            )
        }
    summary: dict[str, Any] = {
        "seed": seed,
        "method": method,
        "optimizer": "adam",
        **final,
        "task_failure_threshold": task_threshold,
        "task_failure_onset": task_onset,
        "nonfinite_onset": nonfinite_onset,
        "finite_parameters": finite_model(actor),
        "steps_completed": trajectory[-1]["step"] if trajectory else 0,
    }
    return ControlRun(actor=actor, trajectory=trajectory, summary=summary)
