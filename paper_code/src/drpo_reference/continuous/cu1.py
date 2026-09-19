"""C-U1 environment, objectives, evaluation, and event diagnostics.

Train and held-out states are independent draws from the same ``Normal(0, I)``
distribution. The held-out split therefore measures same-distribution
held-out-context generalization, not OOD generalization.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from drpo_reference.controls import near_mask

from .gaussian import GaussianActor, gaussian_log_prob, standardized_distance


@dataclass(frozen=True)
class CU1Protocol:
    """C-U1 state, action, contour, and policy settings."""

    state_dim: int = 6
    action_dim: int = 2
    n_train_states: int = 4096
    n_test_states: int = 4096
    gap_to_unseen_optimum: float = 0.70
    negative_offset_from_positive: float = 0.50
    positive_contour_radius: float = 0.75
    negative_contour_radius: float = 1.20
    reward_width: float = 0.75
    baseline: float = 0.40
    positive_angle_1: float = 0.20
    hidden_dim: int = 64
    initial_sigma: float = 0.60
    near_far_standardized_threshold: float = 5.0
    task_failure_retention: float = 0.45
    task_failure_consecutive_evals: int = 3
    log_sigma_event_boundary: float = 12.0



@dataclass
class Split:
    s: torch.Tensor
    a_plus: torch.Tensor
    a_star: torch.Tensor
    direction: torch.Tensor
    positive_actions: torch.Tensor
    positive_advantages: torch.Tensor
    negative_actions: torch.Tensor
    negative_advantages: torch.Tensor


@dataclass
class Environment:
    train: Split
    test: Split


def state_geometry(
    states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    plus = torch.stack(
        [
            0.70
            * torch.tanh(
                0.85 * states[:, 0]
                - 0.30 * states[:, 1] * states[:, 2]
                + 0.20 * torch.sin(1.6 * states[:, 3])
            ),
            0.65
            * torch.tanh(
                -0.50 * states[:, 1]
                + 0.35 * torch.cos(1.1 * states[:, 4])
                + 0.22 * states[:, 0] * states[:, 5]
            ),
        ],
        dim=1,
    )
    angle = (
        1.15 * torch.tanh(0.75 * states[:, 0] + 0.50 * states[:, 2] - 0.30 * states[:, 5])
        + 0.30 * torch.sin(1.35 * states[:, 1])
    )
    direction = torch.stack([torch.cos(angle), torch.sin(angle)], dim=1)
    perpendicular = torch.stack([-direction[:, 1], direction[:, 0]], dim=1)
    return plus, direction, perpendicular


def reward_from_optimum(
    action: torch.Tensor,
    optimum: torch.Tensor,
    reward_width: float,
) -> torch.Tensor:
    distance = torch.linalg.vector_norm(action - optimum, dim=-1)
    return torch.exp(-0.5 * (distance / reward_width).square())


def contour_angles(
    protocol: CU1Protocol,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    theta_1 = protocol.positive_angle_1
    theta_2 = math.acos(
        2.0 * protocol.gap_to_unseen_optimum / protocol.positive_contour_radius
        - math.cos(theta_1)
    )
    positive = (math.pi - theta_1, math.pi + theta_1, math.pi - theta_2, math.pi + theta_2)
    negative = (
        math.pi,
        3.0 * math.pi / 4.0,
        math.pi / 2.0,
        math.pi / 4.0,
        0.0,
        -math.pi / 4.0,
        -math.pi / 2.0,
        -3.0 * math.pi / 4.0,
    )
    return (
        torch.tensor(positive, dtype=dtype, device=device),
        torch.tensor(negative, dtype=dtype, device=device),
    )


def make_split(states: torch.Tensor, protocol: CU1Protocol) -> Split:
    plus, direction, perpendicular = state_geometry(states)
    star = plus + protocol.gap_to_unseen_optimum * direction
    positive_theta, negative_theta = contour_angles(protocol, states.dtype, states.device)

    def contour_actions(theta: torch.Tensor, radius: float) -> torch.Tensor:
        contour_direction = (
            torch.cos(theta)[None, :, None] * direction[:, None, :]
            + torch.sin(theta)[None, :, None] * perpendicular[:, None, :]
        )
        return star[:, None, :] + radius * contour_direction

    positive_actions = contour_actions(positive_theta, protocol.positive_contour_radius)
    negative_actions = contour_actions(negative_theta, protocol.negative_contour_radius)
    positive_advantages = (
        reward_from_optimum(positive_actions, star[:, None, :], protocol.reward_width)
        - protocol.baseline
    )
    negative_advantages = (
        reward_from_optimum(negative_actions, star[:, None, :], protocol.reward_width)
        - protocol.baseline
    )
    return Split(
        s=states,
        a_plus=plus,
        a_star=star,
        direction=direction,
        positive_actions=positive_actions,
        positive_advantages=positive_advantages,
        negative_actions=negative_actions,
        negative_advantages=negative_advantages,
    )


def make_environment(
    seed: int,
    protocol: CU1Protocol,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Environment:
    target = torch.device(device)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    train_states = torch.randn(
        protocol.n_train_states,
        protocol.state_dim,
        generator=generator,
        dtype=dtype,
    ).to(target)
    test_states = torch.randn(
        protocol.n_test_states,
        protocol.state_dim,
        generator=generator,
        dtype=dtype,
    ).to(target)
    return Environment(
        train=make_split(train_states, protocol),
        test=make_split(test_states, protocol),
    )


def make_actor(protocol: CU1Protocol) -> GaussianActor:
    return GaussianActor(
        state_dim=protocol.state_dim,
        action_dim=protocol.action_dim,
        hidden_dim=protocol.hidden_dim,
        initial_sigma=protocol.initial_sigma,
    )


def actor_log_prob(
    actor: GaussianActor,
    states: torch.Tensor,
    actions: torch.Tensor,
    protocol: CU1Protocol,
    fixed_sigma: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mu, predicted_log_std = actor(states)
    log_std = (
        predicted_log_std
        if fixed_sigma is None
        else torch.full_like(predicted_log_std, math.log(fixed_sigma))
    )
    return (
        gaussian_log_prob(mu, log_std, actions, protocol.action_dim),
        mu,
        log_std,
    )


def positive_loss(
    actor: GaussianActor,
    split: Split,
    protocol: CU1Protocol,
    ids: torch.Tensor | None = None,
    fixed_sigma: float | None = None,
) -> torch.Tensor:
    states = split.s if ids is None else split.s[ids]
    actions = split.positive_actions if ids is None else split.positive_actions[ids]
    advantages = split.positive_advantages if ids is None else split.positive_advantages[ids]
    log_probability, _, _ = actor_log_prob(actor, states, actions, protocol, fixed_sigma)
    return -(advantages * log_probability).mean()


def negative_loss(
    actor: GaussianActor,
    split: Split,
    protocol: CU1Protocol,
    ids: torch.Tensor | None = None,
    fixed_sigma: float | None = None,
    columns: slice = slice(None),
) -> torch.Tensor:
    states = split.s if ids is None else split.s[ids]
    actions = split.negative_actions if ids is None else split.negative_actions[ids]
    advantages = split.negative_advantages if ids is None else split.negative_advantages[ids]
    log_probability, _, _ = actor_log_prob(
        actor,
        states,
        actions[:, columns],
        protocol,
        fixed_sigma,
    )
    return -(advantages[:, columns] * log_probability).mean()


def local_negative_loss(
    actor: GaussianActor,
    split: Split,
    protocol: CU1Protocol,
    ids: torch.Tensor | None = None,
    fixed_sigma: float | None = None,
) -> torch.Tensor:
    return negative_loss(actor, split, protocol, ids, fixed_sigma, slice(0, 1))


def near_far_losses(
    actor: GaussianActor,
    split: Split,
    protocol: CU1Protocol,
    ids: torch.Tensor,
    fixed_sigma: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    states = split.s[ids]
    actions = split.negative_actions[ids]
    advantages = split.negative_advantages[ids]
    log_probability, mu, log_std = actor_log_prob(
        actor,
        states,
        actions,
        protocol,
        fixed_sigma,
    )
    standardized = standardized_distance(mu, log_std, actions)
    near = near_mask(
        standardized,
        threshold=protocol.near_far_standardized_threshold,
    ).to(log_probability.dtype)
    far = (~near.bool()).to(log_probability.dtype)
    denominator = float(log_probability.numel())
    near_loss = -(advantages * log_probability * near).sum() / denominator
    far_loss = -(advantages * log_probability * far).sum() / denominator
    return near_loss, far_loss


def evaluation(
    actor: GaussianActor,
    split: Split,
    protocol: CU1Protocol,
    fixed_sigma: float | None = None,
) -> dict[str, float | bool]:
    actor.eval()
    with torch.no_grad():
        mu, predicted = actor(split.s)
        log_std = (
            predicted if fixed_sigma is None else torch.full_like(predicted, math.log(fixed_sigma))
        )
        reward = reward_from_optimum(mu, split.a_star, protocol.reward_width)
        axis = ((mu - split.a_plus) * split.direction).sum(-1)
        normalized = axis / protocol.gap_to_unseen_optimum
        sigma = torch.exp(log_std)
        return {
            "reward": reward.mean().item(),
            "normalized_extrapolation_displacement": normalized.mean().item(),
            "sigma_mean": sigma.mean().item(),
            "log_sigma_min": log_std.min().item(),
            "log_sigma_max": log_std.max().item(),
            "log_sigma_output_finite": bool(torch.isfinite(log_std).all().item()),
            "sigma_output_finite": bool(torch.isfinite(sigma).all().item()),
        }


def support_diagnostics(
    actor: GaussianActor,
    split: Split,
    protocol: CU1Protocol,
) -> dict[str, Any]:
    metrics = evaluation(actor, split, protocol)
    finite_log_sigma = bool(metrics["log_sigma_output_finite"])
    finite_sigma = bool(metrics["sigma_output_finite"])
    log_min = float(metrics["log_sigma_min"]) if finite_log_sigma else float("nan")
    log_max = float(metrics["log_sigma_max"]) if finite_log_sigma else float("nan")
    contraction = finite_log_sigma and log_min < -protocol.log_sigma_event_boundary
    expansion = finite_log_sigma and log_max > protocol.log_sigma_event_boundary
    event_type = (
        "nonfinite_log_sigma_output"
        if not finite_log_sigma
        else "nonfinite_sigma_output"
        if not finite_sigma
        else "support_contraction"
        if contraction
        else "unexpected_support_expansion"
        if expansion
        else None
    )
    return {
        "log_sigma_min_all_states": log_min,
        "log_sigma_max_all_states": log_max,
        "sigma_output_finite_all_states": finite_sigma,
        "log_sigma_output_finite_all_states": finite_log_sigma,
        "support_contraction_boundary": contraction,
        "unexpected_support_expansion_boundary": expansion,
        "event_type": event_type,
    }
