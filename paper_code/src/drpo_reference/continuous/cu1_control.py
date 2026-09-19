"""C-U1 far-pressure controls for the phase experiment."""

from __future__ import annotations

import copy
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
    make_actor,
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
    return add_gradients(positive_gradient, controlled_negative)


def run_far_pressure_control(
    *,
    seed: int,
    initialization_state: dict[str, torch.Tensor],
    environment: Environment,
    protocol: CU1Protocol,
    positive_training: CU1PositiveProtocol = CU1PositiveProtocol(),
    control: CU1ControlProtocol = CU1ControlProtocol(),
    method: str,
) -> dict[str, Any]:
    """Run one E4 far-pressure control branch."""

    actor = make_actor(protocol).to(
        environment.train.s.device,
        dtype=environment.train.s.dtype,
    )
    actor.load_state_dict(copy.deepcopy(initialization_state))
    parameters = actor.mean_parameters()
    optimizer = make_adam(
        parameters,
        learning_rate=control.learning_rate,
        training=positive_training,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed + 500009)
    fixed_sigma = analytic_positive_sigma(protocol)
    for step in range(1, control.steps + 1):
        ids = torch.randint(
            0,
            protocol.n_train_states,
            (positive_training.positive_batch_states,),
            generator=generator,
        ).to(environment.train.s.device)
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
