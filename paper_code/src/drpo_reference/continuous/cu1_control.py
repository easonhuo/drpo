"""C-U1 far-pressure controls for the phase experiment."""

from __future__ import annotations

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
from .cu1_mechanism import controlled_negative_gradients
from .cu1_phase import CU1PhaseProtocol, analytic_positive_sigma
from .cu1_training import (
    CU1PositiveProtocol,
    add_gradients,
    finite_model,
    gradients,
    initialized_actor,
    make_adam,
    sample_ids,
    set_parameter_gradients,
)
from .gaussian import GaussianActor

GradientTuple = tuple[torch.Tensor | None, ...]


def control_gradients(
    actor: GaussianActor,
    split: Split,
    protocol: CU1Protocol,
    phase: CU1PhaseProtocol,
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
    controlled_negative = controlled_negative_gradients(
        local_gradient,
        far_gradient,
        near_scale=phase.control_alpha_local,
        far_scale=phase.control_lambda_far,
        cap_ratio=phase.control_far_cap_ratio,
        method=method,
    )
    return add_gradients(positive_gradient, controlled_negative)


def run_far_pressure_control(
    *,
    seed: int,
    initialization_state: dict[str, torch.Tensor],
    environment: Environment,
    protocol: CU1Protocol,
    positive_training: CU1PositiveProtocol | None = None,
    phase: CU1PhaseProtocol | None = None,
    method: str,
) -> dict[str, Any]:
    """Run one E4 far-pressure control branch."""

    positive_training = CU1PositiveProtocol() if positive_training is None else positive_training
    phase = CU1PhaseProtocol() if phase is None else phase
    actor = initialized_actor(protocol, environment, initialization_state)
    parameters = actor.mean_parameters()
    optimizer = make_adam(
        parameters,
        learning_rate=phase.control_learning_rate,
        training=positive_training,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed + 500009)
    fixed_sigma = analytic_positive_sigma(protocol)
    for step in range(1, phase.control_steps + 1):
        ids = sample_ids(
            generator,
            environment.train,
            positive_training.positive_batch_states,
        )
        gradients = control_gradients(
            actor,
            environment.train,
            protocol,
            phase,
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
