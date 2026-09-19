"""Positive-only training and deterministic gradient utilities for C-U1."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn

from drpo_reference.common import cpu_generator, seed_all

from .cu1 import (
    CU1Protocol,
    Environment,
    Split,
    make_actor,
    make_environment,
    positive_loss,
)
from .gaussian import GaussianActor

EPS = 1.0e-12
GradientTuple = tuple[torch.Tensor | None, ...]


@dataclass(frozen=True)
class CU1PositiveProtocol:
    """C-U1 positive-training settings."""

    positive_adam_lr: float = 1e-3
    positive_batch_states: int = 256
    positive_steps: int = 2000
    positive_continuation_steps: int = 2000
    lbfgs_lr: float = 0.25
    lbfgs_max_iter: int = 120
    positive_polish_min_steps: int = 100
    positive_polish_max_steps: int = 500
    positive_polish_check_every: int = 25
    positive_polish_lr: float = 1e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    absolute_residual_threshold_alpha_zero: float = 1e-3



@dataclass
class PositiveRun:
    """In-memory result of one positive-only C-U1 run."""

    actor: GaussianActor
    environment: Environment
    initialization_state: dict[str, torch.Tensor]


def gradient_norm(gradients: Sequence[torch.Tensor | None]) -> torch.Tensor:
    present = [gradient.reshape(-1) for gradient in gradients if gradient is not None]
    if not present:
        return torch.zeros(())
    return torch.linalg.vector_norm(torch.cat(present))


def add_gradients(
    *groups: Sequence[torch.Tensor | None],
    scales: Sequence[float] | None = None,
) -> GradientTuple:
    if not groups:
        return ()
    if scales is None:
        scales = [1.0] * len(groups)
    result: list[torch.Tensor | None] = []
    for components in zip(*groups):
        value: torch.Tensor | None = None
        for gradient, scale in zip(components, scales):
            if gradient is not None:
                value = gradient * scale if value is None else value + gradient * scale
        result.append(value)
    return tuple(result)


def scale_gradients(
    gradients: Sequence[torch.Tensor | None],
    scale: float | torch.Tensor,
) -> GradientTuple:
    return tuple(None if gradient is None else gradient * scale for gradient in gradients)


def set_parameter_gradients(
    parameters: Sequence[nn.Parameter],
    gradients: Sequence[torch.Tensor | None],
) -> None:
    for parameter, gradient in zip(parameters, gradients):
        parameter.grad = None if gradient is None else gradient.detach().clone()


def finite_model(model: nn.Module) -> bool:
    return all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters())


def make_adam(
    parameters: Sequence[nn.Parameter],
    *,
    learning_rate: float,
    training: CU1PositiveProtocol,
) -> torch.optim.Adam:
    return torch.optim.Adam(
        parameters,
        lr=learning_rate,
        betas=(training.adam_beta1, training.adam_beta2),
        eps=training.adam_eps,
    )


def normalized_field_residual(
    actor: GaussianActor,
    split: Split,
    protocol: CU1Protocol,
    *,
    alpha: float,
    fixed_sigma: float | None,
) -> dict[str, float]:
    from .cu1 import local_negative_loss

    parameters = actor.mean_parameters() if fixed_sigma is not None else actor.all_parameters()
    positive = positive_loss(actor, split, protocol, fixed_sigma=fixed_sigma)
    negative = local_negative_loss(actor, split, protocol, fixed_sigma=fixed_sigma)
    positive_gradient = torch.autograd.grad(
        positive,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    negative_gradient = torch.autograd.grad(
        negative,
        parameters,
        allow_unused=True,
    )
    total_gradient = add_gradients(
        positive_gradient,
        negative_gradient,
        scales=(1.0, alpha),
    )
    positive_norm = gradient_norm(positive_gradient).item()
    negative_norm = gradient_norm(scale_gradients(negative_gradient, alpha)).item()
    total_norm = gradient_norm(total_gradient).item()
    return {
        "positive_gradient_norm": positive_norm,
        "negative_gradient_norm": negative_norm,
        "total_gradient_norm": total_norm,
        "normalized_field_residual": total_norm / (positive_norm + negative_norm + EPS),
    }


def _copy_state(actor: GaussianActor) -> dict[str, torch.Tensor]:
    return copy.deepcopy(actor.state_dict())


def train_positive(
    *,
    seed: int,
    protocol: CU1Protocol,
    training: CU1PositiveProtocol = CU1PositiveProtocol(),
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> PositiveRun:
    """Run positive-only training and preserve the E3/E4 initialization state."""

    target = torch.device(device)
    environment = make_environment(seed, protocol, target, dtype)
    seed_all(seed)
    actor = make_actor(protocol).to(device=target, dtype=dtype)
    optimizer = make_adam(
        actor.all_parameters(),
        learning_rate=training.positive_adam_lr,
        training=training,
    )
    generator = cpu_generator(seed + 100003)

    for _ in range(training.positive_steps):
        ids = torch.randint(
            0,
            protocol.n_train_states,
            (training.positive_batch_states,),
            generator=generator,
        ).to(target)
        loss = positive_loss(actor, environment.train, protocol, ids)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    initialization_state = _copy_state(actor)

    lbfgs = torch.optim.LBFGS(
        actor.parameters(),
        lr=training.lbfgs_lr,
        max_iter=training.lbfgs_max_iter,
        history_size=50,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        lbfgs.zero_grad(set_to_none=True)
        loss = positive_loss(actor, environment.train, protocol)
        loss.backward()
        return loss

    lbfgs.step(closure)

    continuation = make_adam(
        actor.all_parameters(),
        learning_rate=training.positive_adam_lr * 0.25,
        training=training,
    )
    for _ in range(training.positive_continuation_steps):
        ids = torch.randint(
            0,
            protocol.n_train_states,
            (training.positive_batch_states,),
            generator=generator,
        ).to(target)
        loss = positive_loss(actor, environment.train, protocol, ids)
        continuation.zero_grad(set_to_none=True)
        loss.backward()
        continuation.step()

    final_lbfgs = torch.optim.LBFGS(
        actor.parameters(),
        lr=training.lbfgs_lr,
        max_iter=training.lbfgs_max_iter,
        history_size=50,
        line_search_fn="strong_wolfe",
    )

    def final_closure() -> torch.Tensor:
        final_lbfgs.zero_grad(set_to_none=True)
        loss = positive_loss(actor, environment.train, protocol)
        loss.backward()
        return loss

    final_lbfgs.step(final_closure)

    polish = make_adam(
        actor.all_parameters(),
        learning_rate=training.positive_polish_lr,
        training=training,
    )
    for polish_step in range(1, training.positive_polish_max_steps + 1):
        loss = positive_loss(actor, environment.train, protocol)
        polish.zero_grad(set_to_none=True)
        loss.backward()
        polish.step()
        should_check = polish_step >= training.positive_polish_min_steps and (
            polish_step % training.positive_polish_check_every == 0
            or polish_step == training.positive_polish_max_steps
        )
        if should_check:
            field = normalized_field_residual(
                actor,
                environment.train,
                protocol,
                alpha=0.0,
                fixed_sigma=None,
            )
            if field["total_gradient_norm"] < training.absolute_residual_threshold_alpha_zero:
                break

    return PositiveRun(
        actor=actor,
        environment=environment,
        initialization_state=initialization_state,
    )
