"""Positive-only training and deterministic gradient utilities for C-U1."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from drpo_reference.common import atomic_json, cpu_generator, seed_all, write_csv

from .cu1 import (
    CU1Protocol,
    Environment,
    Split,
    all_negative_loss,
    evaluation,
    make_actor,
    make_environment,
    positive_loss,
)
from .gaussian import GaussianActor

EPS = 1.0e-12
GradientTuple = tuple[torch.Tensor | None, ...]


@dataclass(frozen=True)
class CU1PositiveProtocol:
    """Frozen optimizer, budget, audit, and seed settings for C-U1 E1/E2."""

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
    eval_every: int = 100
    probe_states: int = 128
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    absolute_residual_threshold_alpha_zero: float = 1e-3
    formal_seeds: tuple[int, ...] = tuple(range(10, 30))

    def __post_init__(self) -> None:
        if self.positive_steps <= 0 or self.positive_continuation_steps < 0:
            raise ValueError("positive training budgets must be non-negative")
        if self.positive_batch_states <= 0:
            raise ValueError("positive_batch_states must be positive")
        if self.eval_every <= 0:
            raise ValueError("eval_every must be positive")
        if self.positive_polish_min_steps < 0:
            raise ValueError("positive_polish_min_steps must be non-negative")
        if self.positive_polish_max_steps < self.positive_polish_min_steps:
            raise ValueError("polish maximum must not precede its minimum")
        if self.positive_polish_check_every <= 0:
            raise ValueError("positive_polish_check_every must be positive")


@dataclass
class PositiveRun:
    """In-memory result of one positive-only C-U1 run."""

    actor: GaussianActor
    environment: Environment
    initialization_state: dict[str, torch.Tensor]
    trajectory: list[dict[str, Any]]
    summary: dict[str, Any]


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
    if len(scales) != len(groups):
        raise ValueError("scales must match gradient groups")
    if any(len(group) != len(groups[0]) for group in groups):
        raise ValueError("gradient groups must have equal length")
    result: list[torch.Tensor | None] = []
    for components in zip(*groups):
        value: torch.Tensor | None = None
        for gradient, scale in zip(components, scales):
            if gradient is not None:
                value = (
                    gradient * scale
                    if value is None
                    else value + gradient * scale
                )
        result.append(value)
    return tuple(result)


def scale_gradients(
    gradients: Sequence[torch.Tensor | None],
    scale: float | torch.Tensor,
) -> GradientTuple:
    return tuple(
        None if gradient is None else gradient * scale
        for gradient in gradients
    )


def set_parameter_gradients(
    parameters: Sequence[nn.Parameter],
    gradients: Sequence[torch.Tensor | None],
) -> None:
    if len(parameters) != len(gradients):
        raise ValueError("parameters and gradients must have equal length")
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


def optimizer_step_with_norm(
    optimizer: torch.optim.Optimizer,
    parameters: Sequence[nn.Parameter],
) -> float:
    before = [parameter.detach().clone() for parameter in parameters]
    optimizer.step()
    changes = [
        (parameter.detach() - previous).reshape(-1)
        for parameter, previous in zip(parameters, before)
    ]
    return torch.linalg.vector_norm(torch.cat(changes)).item() if changes else 0.0


def phantom_metrics(
    actor: GaussianActor,
    split: Split,
    protocol: CU1Protocol,
    probe_count: int | None = None,
) -> dict[str, float]:
    count = len(split.s) if probe_count is None else min(probe_count, len(split.s))
    ids = torch.arange(count, device=split.s.device)
    parameters = actor.all_parameters()
    loss = all_negative_loss(actor, split, protocol, ids)
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
    with torch.no_grad():
        mu, log_std = actor(split.s[ids])
        actions = split.negative_actions[ids]
        sigma = torch.exp(log_std)
        raw = torch.linalg.vector_norm(actions - mu[:, None, :], dim=-1)
        standardized = raw / sigma[:, None]
    return {
        "aggregate_phantom_negative_gradient_norm": gradient_norm(gradients).item(),
        "negative_raw_distance_mean": raw.mean().item(),
        "negative_standardized_distance_mean": standardized.mean().item(),
        "near_standardized_distance": standardized[:, 0].mean().item(),
        "far_standardized_distance": standardized[:, 4].mean().item(),
    }


def normalized_field_residual(
    actor: GaussianActor,
    split: Split,
    protocol: CU1Protocol,
    *,
    alpha: float,
    fixed_sigma: float | None,
    local_negative: bool = True,
) -> dict[str, float]:
    from .cu1 import local_negative_loss

    parameters = (
        actor.mean_parameters()
        if fixed_sigma is not None
        else actor.all_parameters()
    )
    positive = positive_loss(actor, split, protocol, fixed_sigma=fixed_sigma)
    negative = (
        local_negative_loss(actor, split, protocol, fixed_sigma=fixed_sigma)
        if local_negative
        else all_negative_loss(actor, split, protocol, fixed_sigma=fixed_sigma)
    )
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
        "normalized_field_residual": total_norm
        / (positive_norm + negative_norm + EPS),
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
    """Run the frozen E2 positive-only training and terminal audits in memory."""

    target = torch.device(device)
    environment = make_environment(seed, protocol, target, dtype)
    seed_all(seed)
    actor = make_actor(protocol).to(device=target, dtype=dtype)
    initial_phantom = phantom_metrics(
        actor,
        environment.train,
        protocol,
        training.probe_states,
    )
    optimizer = make_adam(
        actor.all_parameters(),
        learning_rate=training.positive_adam_lr,
        training=training,
    )
    generator = cpu_generator(seed + 100003)
    trajectory: list[dict[str, Any]] = []

    def record(step: int, stage: str) -> None:
        row: dict[str, Any] = {
            "step": step,
            "stage": stage,
            **evaluation(actor, environment.test, protocol),
        }
        if step % max(training.eval_every, 1) == 0 or stage != "adam":
            row.update(
                phantom_metrics(
                    actor,
                    environment.train,
                    protocol,
                    training.probe_states,
                )
            )
        trajectory.append(row)

    record(0, "initial")
    actor.train()
    for step in range(1, training.positive_steps + 1):
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
        if step % training.eval_every == 0 or step == training.positive_steps:
            record(step, "adam")

    # E1/E3/E4 branch from this exact Adam checkpoint, before E2-only audits.
    initialization_state = _copy_state(actor)

    snapshot = _copy_state(actor)
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

    try:
        lbfgs.step(closure)
        if not finite_model(actor):
            raise FloatingPointError("non-finite LBFGS state")
    except Exception:
        actor.load_state_dict(snapshot)
    record(training.positive_steps, "stationary_audit")

    continuation = make_adam(
        actor.all_parameters(),
        learning_rate=training.positive_adam_lr * 0.25,
        training=training,
    )
    for extra in range(1, training.positive_continuation_steps + 1):
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
        step = training.positive_steps + extra
        if (
            extra % training.eval_every == 0
            or extra == training.positive_continuation_steps
        ):
            record(step, "continuation")

    post_continuation = _copy_state(actor)
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

    final_audit_succeeded = False
    try:
        final_lbfgs.step(final_closure)
        if not finite_model(actor):
            raise FloatingPointError("non-finite final LBFGS state")
        final_audit_succeeded = True
    except Exception:
        actor.load_state_dict(post_continuation)

    polish = make_adam(
        actor.all_parameters(),
        learning_rate=training.positive_polish_lr,
        training=training,
    )
    polish_steps_used = 0
    for polish_step in range(1, training.positive_polish_max_steps + 1):
        loss = positive_loss(actor, environment.train, protocol)
        polish.zero_grad(set_to_none=True)
        loss.backward()
        polish.step()
        polish_steps_used = polish_step
        should_check = (
            polish_step >= training.positive_polish_min_steps
            and (
                polish_step % training.positive_polish_check_every == 0
                or polish_step == training.positive_polish_max_steps
            )
        )
        if should_check:
            field = normalized_field_residual(
                actor,
                environment.train,
                protocol,
                alpha=0.0,
                fixed_sigma=None,
            )
            if field["total_gradient_norm"] < (
                training.absolute_residual_threshold_alpha_zero
            ):
                break

    record(
        training.positive_steps + training.positive_continuation_steps,
        "final_stationary_audit_and_adaptive_polish",
    )
    final_phantom = phantom_metrics(actor, environment.train, protocol)
    final_evaluation = evaluation(actor, environment.test, protocol)
    field = normalized_field_residual(
        actor,
        environment.train,
        protocol,
        alpha=0.0,
        fixed_sigma=None,
    )
    initial_norm = initial_phantom["aggregate_phantom_negative_gradient_norm"]
    summary: dict[str, Any] = {
        "seed": seed,
        **final_evaluation,
        **final_phantom,
        **field,
        "initial_probe_phantom_gradient_norm": initial_norm,
        "probe_phantom_growth": trajectory[-1].get(
            "aggregate_phantom_negative_gradient_norm",
            float("nan"),
        )
        / (initial_norm + EPS),
        "final_stationary_audit_succeeded": final_audit_succeeded,
        "full_data_polish_steps": polish_steps_used,
        "full_data_polish_max_steps": training.positive_polish_max_steps,
        "status": (
            "stable_plateau_2x_confirmed"
            if final_audit_succeeded
            and field["total_gradient_norm"]
            < training.absolute_residual_threshold_alpha_zero
            else "finite_but_residual_above_strict_threshold"
        ),
    }
    return PositiveRun(
        actor=actor,
        environment=environment,
        initialization_state=initialization_state,
        trajectory=trajectory,
        summary=summary,
    )


def write_positive_run(output_root: Path, run: PositiveRun) -> None:
    """Write the two checkpoints and E2 trajectory/summary used downstream."""

    seed = int(run.summary["seed"])
    checkpoint_dir = output_root / "positive_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        run.initialization_state,
        checkpoint_dir / f"seed_{seed}_adam2000_initialization.pt",
    )
    torch.save(
        run.actor.state_dict(),
        checkpoint_dir / f"seed_{seed}.pt",
    )
    write_csv(output_root / "e2" / f"seed_{seed}_trajectory.csv", run.trajectory)
    atomic_json(output_root / "e2" / f"seed_{seed}.json", run.summary)
