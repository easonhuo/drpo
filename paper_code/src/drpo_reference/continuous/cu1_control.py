"""C-U1 far-pressure controls for the phase experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .cu1 import (
    CU1Protocol,
    Environment,
    Split,
    evaluation,
    local_negative_loss,
    negative_loss,
    positive_loss,
)
from .cu1_phase import analytic_positive_sigma
from .cu1_training import (
    EPS,
    CU1PositiveProtocol,
    add_gradients,
    finite_model,
    gradients,
    initialized_actor,
    gradient_norm,
    make_adam,
    sample_ids,
    scale_gradients,
    set_parameter_gradients,
)
from .gaussian import GaussianActor

GradientTuple = tuple[torch.Tensor | None, ...]


@dataclass(frozen=True)
class CU1ControlProtocol:
    """C-U1 local/far pressure control settings."""

    alpha_local: float = 1.0
    lambda_far: float = 1.0
    far_cap_ratio: float = 0.05
    learning_rate: float = 5e-4
    steps: int = 4000
    seeds: tuple[int, ...] = tuple(range(50, 70))



def control_gradients(
    actor: GaussianActor,
    split: Split,
    protocol: CU1Protocol,
    control: CU1ControlProtocol,
    ids: torch.Tensor,
    *,
    method: str,
    fixed_sigma: float,
) -> GradientTuple:
    """Return the E4 local/far control gradient."""

    parameters = actor.mean_parameters()
    positive = positive_loss(actor, split, protocol, ids, fixed_sigma)
    local = local_negative_loss(actor, split, protocol, ids, fixed_sigma)
    far = negative_loss(actor, split, protocol, ids, fixed_sigma, slice(1, None))
    positive_gradient = gradients(positive, parameters, retain_graph=True)
    local_gradient = gradients(local, parameters, retain_graph=True)
    far_gradient = gradients(far, parameters)
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
    return add_gradients(positive_gradient, controlled_negative)


def run_far_pressure_control(
    *,
    seed: int,
    initialization_state: dict[str, torch.Tensor],
    environment: Environment,
    protocol: CU1Protocol,
    positive_training: CU1PositiveProtocol | None = None,
    control: CU1ControlProtocol | None = None,
    method: str,
) -> dict[str, Any]:
    """Run one E4 far-pressure control branch."""

    positive_training = CU1PositiveProtocol() if positive_training is None else positive_training
    control = CU1ControlProtocol() if control is None else control
    actor = initialized_actor(protocol, environment, initialization_state)
    parameters = actor.mean_parameters()
    optimizer = make_adam(
        parameters,
        learning_rate=control.learning_rate,
        training=positive_training,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed + 500009)
    fixed_sigma = analytic_positive_sigma(protocol)
    for step in range(1, control.steps + 1):
        ids = sample_ids(
            generator,
            environment.train,
            positive_training.positive_batch_states,
        )
        gradients = control_gradients(
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
        optimizer.step()
        if not finite_model(actor):
            break

    final = evaluation(actor, environment.test, protocol, fixed_sigma)
    summary: dict[str, Any] = {
        "seed": seed,
        "method": method,
        **final,
        "finite_parameters": finite_model(actor),
        "steps_completed": step,
    }
    return summary
