"""Positive-only training and deterministic gradient utilities for C-U1."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

from drpo_reference.common import cpu_generator, seed_all

from .cu1 import (
    CU1Protocol,
    Environment,
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


def gradients(
    loss: torch.Tensor,
    parameters: Sequence[nn.Parameter],
    *,
    retain_graph: bool = False,
) -> GradientTuple:
    return tuple(
        torch.autograd.grad(
            loss,
            parameters,
            retain_graph=retain_graph,
            allow_unused=True,
        )
    )


def add_gradients(*groups: Sequence[torch.Tensor | None]) -> GradientTuple:
    result: list[torch.Tensor | None] = []
    for components in zip(*groups):
        value: torch.Tensor | None = None
        for gradient in components:
            if gradient is not None:
                value = gradient if value is None else value + gradient
        result.append(value)
    return tuple(result)


def scale_gradients(
    gradients: Sequence[torch.Tensor | None],
    scale: float | torch.Tensor,
) -> GradientTuple:
    return tuple(None if gradient is None else gradient * scale for gradient in gradients)


def finite_model(model: nn.Module) -> bool:
    return all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters())


def initialized_actor(
    protocol: CU1Protocol,
    environment: Environment,
    state: dict[str, torch.Tensor],
) -> GaussianActor:
    actor = make_actor(protocol).to(
        environment.train.s.device,
        dtype=environment.train.s.dtype,
    )
    actor.load_state_dict(copy.deepcopy(state))
    return actor


def sample_ids(generator: torch.Generator, split, batch_size: int) -> torch.Tensor:
    return torch.randint(
        len(split.s),
        (batch_size,),
        generator=generator,
    ).to(split.s.device)


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


def field_diagnostics(
    positive: torch.Tensor,
    negative: torch.Tensor | None,
    parameters: Sequence[nn.Parameter],
    *,
    alpha: float = 1.0,
) -> dict[str, float]:
    positive_gradient = gradients(
        positive,
        parameters,
        retain_graph=negative is not None,
    )
    positive_norm = float(gradient_norm(positive_gradient).item())
    if negative is None:
        return {
            "total_gradient_norm": positive_norm,
            "normalized_field_residual": float("nan"),
            "stationarity_residual": positive_norm,
        }
    negative_gradient = gradients(negative, parameters)
    weighted_negative = scale_gradients(negative_gradient, alpha)
    total_gradient = add_gradients(positive_gradient, weighted_negative)
    negative_norm = float(gradient_norm(weighted_negative).item())
    total_norm = float(gradient_norm(total_gradient).item())
    residual = total_norm / (positive_norm + negative_norm + EPS)
    return {
        "total_gradient_norm": total_norm,
        "normalized_field_residual": residual,
        "stationarity_residual": residual,
    }


def train_positive(
    *,
    seed: int,
    protocol: CU1Protocol,
    training: CU1PositiveProtocol | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> PositiveRun:
    """Run positive-only training and preserve the E3/E4 initialization state."""

    training = CU1PositiveProtocol() if training is None else training
    target = torch.device(device)
    environment = make_environment(seed, protocol, target, dtype)
    seed_all(seed)
    actor = make_actor(protocol).to(device=target, dtype=dtype)
    generator = cpu_generator(seed + 100003)

    def minibatch_adam(learning_rate: float, steps: int) -> None:
        optimizer = make_adam(
            actor.all_parameters(),
            learning_rate=learning_rate,
            training=training,
        )
        for _ in range(steps):
            ids = sample_ids(generator, environment.train, training.positive_batch_states)
            loss = positive_loss(actor, environment.train, protocol, ids)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    def lbfgs_refine() -> None:
        optimizer = torch.optim.LBFGS(
            actor.parameters(),
            lr=training.lbfgs_lr,
            max_iter=training.lbfgs_max_iter,
            history_size=50,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            loss = positive_loss(actor, environment.train, protocol)
            loss.backward()
            return loss

        optimizer.step(closure)

    minibatch_adam(training.positive_adam_lr, training.positive_steps)
    initialization_state = copy.deepcopy(actor.state_dict())
    lbfgs_refine()
    minibatch_adam(
        training.positive_adam_lr * 0.25,
        training.positive_continuation_steps,
    )
    lbfgs_refine()

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
            field = field_diagnostics(
                positive_loss(actor, environment.train, protocol),
                None,
                actor.all_parameters(),
            )
            if field["total_gradient_norm"] < training.absolute_residual_threshold_alpha_zero:
                break

    return PositiveRun(
        actor=actor,
        environment=environment,
        initialization_state=initialization_state,
    )
