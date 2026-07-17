"""Frozen C-U1 environment, objectives, evaluation, and event diagnostics.

Train and held-out states are independent draws from the same ``Normal(0, I)``
distribution. The held-out split therefore measures same-distribution
held-out-context generalization, not OOD generalization.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from drpo_reference.common import EventFlags
from drpo_reference.controls import far_mask, near_mask

from .gaussian import GaussianActor, gaussian_log_prob, standardized_distance


@dataclass(frozen=True)
class CU1Protocol:
    """Frozen C-U1 state, action, contour, policy, and audit constants."""

    state_dim: int = 6
    action_dim: int = 2
    n_train_states: int = 4096
    n_test_states: int = 4096
    positive_samples_per_state: int = 4
    negative_samples_per_state: int = 8
    gap_to_unseen_optimum: float = 0.70
    negative_offset_from_positive: float = 0.50
    positive_contour_radius: float = 0.75
    negative_contour_radius: float = 1.20
    reward_width: float = 0.75
    baseline: float = 0.40
    positive_angle_1: float = 0.20
    hidden_dim: int = 64
    hidden_layers: int = 2
    initial_sigma: float = 0.60
    near_far_standardized_threshold: float = 5.0
    task_failure_retention: float = 0.45
    task_failure_consecutive_evals: int = 3
    log_sigma_event_boundary: float = 12.0

    def __post_init__(self) -> None:
        if self.action_dim != 2:
            raise ValueError("the frozen C-U1 protocol requires action_dim=2")
        if self.hidden_layers != 2:
            raise ValueError("the frozen C-U1 protocol requires two hidden layers")
        if (
            self.positive_samples_per_state != 4
            or self.negative_samples_per_state != 8
        ):
            raise ValueError(
                "the frozen C-U1 contour cardinalities are 4 positive and 8 negative"
            )


@dataclass
class Split:
    s: torch.Tensor
    a_plus: torch.Tensor
    a_star: torch.Tensor
    a_minus: torch.Tensor
    direction: torch.Tensor
    orthogonal: torch.Tensor
    positive_actions: torch.Tensor
    positive_rewards: torch.Tensor
    positive_advantages: torch.Tensor
    negative_actions: torch.Tensor
    negative_rewards: torch.Tensor
    negative_advantages: torch.Tensor


@dataclass
class Environment:
    train: Split
    test: Split


def base_from_state(states: torch.Tensor) -> torch.Tensor:
    first = 0.70 * torch.tanh(
        0.85 * states[:, 0]
        - 0.30 * states[:, 1] * states[:, 2]
        + 0.20 * torch.sin(1.6 * states[:, 3])
    )
    second = 0.65 * torch.tanh(
        -0.50 * states[:, 1]
        + 0.35 * torch.cos(1.1 * states[:, 4])
        + 0.22 * states[:, 0] * states[:, 5]
    )
    return torch.stack([first, second], dim=1)


def task_direction_from_state(states: torch.Tensor) -> torch.Tensor:
    angle = 1.15 * torch.tanh(
        0.75 * states[:, 0]
        + 0.50 * states[:, 2]
        - 0.30 * states[:, 5]
    )
    angle = angle + 0.30 * torch.sin(1.35 * states[:, 1])
    return torch.stack([torch.cos(angle), torch.sin(angle)], dim=1)


def orthogonal(direction: torch.Tensor) -> torch.Tensor:
    return torch.stack([-direction[:, 1], direction[:, 0]], dim=1)


def reward_from_optimum(
    action: torch.Tensor,
    optimum: torch.Tensor,
    reward_width: float,
) -> torch.Tensor:
    distance = torch.linalg.vector_norm(action - optimum, dim=-1)
    return torch.exp(-0.5 * (distance / reward_width).square())


def positive_angles(protocol: CU1Protocol, dtype: torch.dtype) -> torch.Tensor:
    theta_1 = protocol.positive_angle_1
    cos_theta_2 = (
        2.0 * protocol.gap_to_unseen_optimum / protocol.positive_contour_radius
        - math.cos(theta_1)
    )
    if not (-1.0 <= cos_theta_2 <= 1.0):
        raise RuntimeError("invalid positive contour geometry")
    theta_2 = math.acos(cos_theta_2)
    return torch.tensor(
        [
            math.pi - theta_1,
            math.pi + theta_1,
            math.pi - theta_2,
            math.pi + theta_2,
        ],
        dtype=dtype,
    )


def negative_angles(dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(
        [
            math.pi,
            3.0 * math.pi / 4.0,
            math.pi / 2.0,
            math.pi / 4.0,
            0.0,
            -math.pi / 4.0,
            -math.pi / 2.0,
            -3.0 * math.pi / 4.0,
        ],
        dtype=dtype,
    )


def make_split(states: torch.Tensor, protocol: CU1Protocol) -> Split:
    plus = base_from_state(states)
    direction = task_direction_from_state(states)
    perpendicular = orthogonal(direction)
    star = plus + protocol.gap_to_unseen_optimum * direction
    minus = plus - protocol.negative_offset_from_positive * direction

    positive_theta = positive_angles(protocol, states.dtype).to(states.device)
    positive_direction = (
        torch.cos(positive_theta)[None, :, None] * direction[:, None, :]
        + torch.sin(positive_theta)[None, :, None] * perpendicular[:, None, :]
    )
    positive_actions = (
        star[:, None, :] + protocol.positive_contour_radius * positive_direction
    )
    positive_rewards = reward_from_optimum(
        positive_actions,
        star[:, None, :],
        protocol.reward_width,
    )
    positive_advantages = positive_rewards - protocol.baseline

    negative_theta = negative_angles(states.dtype).to(states.device)
    negative_direction = (
        torch.cos(negative_theta)[None, :, None] * direction[:, None, :]
        + torch.sin(negative_theta)[None, :, None] * perpendicular[:, None, :]
    )
    negative_actions = (
        star[:, None, :] + protocol.negative_contour_radius * negative_direction
    )
    negative_rewards = reward_from_optimum(
        negative_actions,
        star[:, None, :],
        protocol.reward_width,
    )
    negative_advantages = negative_rewards - protocol.baseline

    return Split(
        s=states,
        a_plus=plus,
        a_star=star,
        a_minus=minus,
        direction=direction,
        orthogonal=perpendicular,
        positive_actions=positive_actions,
        positive_rewards=positive_rewards,
        positive_advantages=positive_advantages,
        negative_actions=negative_actions,
        negative_rewards=negative_rewards,
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


def audit_environment(
    environment: Environment,
    protocol: CU1Protocol,
) -> dict[str, Any]:
    train = environment.train
    positive_centroid_error = torch.max(
        torch.abs(train.positive_actions.mean(dim=1) - train.a_plus)
    ).item()
    negative_reward_range = (
        train.negative_rewards.max(1).values
        - train.negative_rewards.min(1).values
    ).abs().max().item()
    negative_advantage_range = (
        train.negative_advantages.max(1).values
        - train.negative_advantages.min(1).values
    ).abs().max().item()
    positive_reward_range = (
        train.positive_rewards.max(1).values
        - train.positive_rewards.min(1).values
    ).abs().max().item()
    distances = torch.linalg.vector_norm(
        train.negative_actions - train.a_plus[:, None, :],
        dim=-1,
    )
    nearest = distances[:, 0].mean().item()
    farthest = distances[:, 4].mean().item()
    result: dict[str, Any] = {
        "state_distribution": f"Normal(0,I_{protocol.state_dim})",
        "n_train_states": protocol.n_train_states,
        "n_test_states": protocol.n_test_states,
        "positive_actions_per_state": int(train.positive_actions.shape[1]),
        "negative_actions_per_state": int(train.negative_actions.shape[1]),
        "positive_centroid_max_error": positive_centroid_error,
        "positive_reward_max_range_per_state": positive_reward_range,
        "negative_reward_max_range_per_state": negative_reward_range,
        "negative_advantage_max_range_per_state": negative_advantage_range,
        "positive_advantage_fraction": (
            (train.positive_advantages > 0).float().mean().item()
        ),
        "negative_advantage_fraction": (
            (train.negative_advantages < 0).float().mean().item()
        ),
        "nearest_negative_distance": nearest,
        "farthest_negative_distance": farthest,
        "farthest_nearest_distance_ratio": farthest / nearest,
    }
    result["passed"] = bool(
        positive_centroid_error < 2e-6
        and positive_reward_range < 2e-6
        and negative_reward_range < 2e-6
        and negative_advantage_range < 2e-6
        and result["positive_advantage_fraction"] == 1.0
        and result["negative_advantage_fraction"] == 1.0
    )
    return result


def actor_log_prob(
    actor: GaussianActor,
    states: torch.Tensor,
    actions: torch.Tensor,
    protocol: CU1Protocol,
    fixed_sigma: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mu, predicted_log_std = actor(states)
    if fixed_sigma is None:
        log_std = predicted_log_std
    else:
        if not math.isfinite(fixed_sigma) or fixed_sigma <= 0.0:
            raise ValueError("fixed_sigma must be finite and positive")
        log_std = torch.full_like(predicted_log_std, math.log(fixed_sigma))
    return (
        gaussian_log_prob(mu, log_std, actions, protocol.action_dim),
        mu,
        log_std,
    )


def _selected(
    split: Split,
    ids: torch.Tensor | None,
    *,
    negative: bool,
    local_only: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    states = split.s if ids is None else split.s[ids]
    if negative:
        actions = split.negative_actions if ids is None else split.negative_actions[ids]
        advantages = (
            split.negative_advantages
            if ids is None
            else split.negative_advantages[ids]
        )
        if local_only:
            actions = actions[:, :1]
            advantages = advantages[:, :1]
    else:
        actions = split.positive_actions if ids is None else split.positive_actions[ids]
        advantages = (
            split.positive_advantages
            if ids is None
            else split.positive_advantages[ids]
        )
    return states, actions, advantages


def positive_loss(
    actor: GaussianActor,
    split: Split,
    protocol: CU1Protocol,
    ids: torch.Tensor | None = None,
    fixed_sigma: float | None = None,
) -> torch.Tensor:
    states, actions, advantages = _selected(split, ids, negative=False)
    log_probability, _, _ = actor_log_prob(
        actor,
        states,
        actions,
        protocol,
        fixed_sigma,
    )
    return -(advantages * log_probability).mean()


def local_negative_loss(
    actor: GaussianActor,
    split: Split,
    protocol: CU1Protocol,
    ids: torch.Tensor | None = None,
    fixed_sigma: float | None = None,
) -> torch.Tensor:
    states, actions, advantages = _selected(
        split,
        ids,
        negative=True,
        local_only=True,
    )
    log_probability, _, _ = actor_log_prob(
        actor,
        states,
        actions,
        protocol,
        fixed_sigma,
    )
    return -(advantages * log_probability).mean()


def all_negative_loss(
    actor: GaussianActor,
    split: Split,
    protocol: CU1Protocol,
    ids: torch.Tensor | None = None,
    fixed_sigma: float | None = None,
) -> torch.Tensor:
    states, actions, advantages = _selected(split, ids, negative=True)
    log_probability, _, _ = actor_log_prob(
        actor,
        states,
        actions,
        protocol,
        fixed_sigma,
    )
    return -(advantages * log_probability).mean()


def near_far_losses(
    actor: GaussianActor,
    split: Split,
    protocol: CU1Protocol,
    ids: torch.Tensor,
    fixed_sigma: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    states, actions, advantages = _selected(split, ids, negative=True)
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
    far = far_mask(
        standardized,
        threshold=protocol.near_far_standardized_threshold,
    ).to(log_probability.dtype)
    denominator = float(log_probability.numel())
    near_loss = -(advantages * log_probability * near).sum() / denominator
    far_loss = -(advantages * log_probability * far).sum() / denominator
    raw = torch.linalg.vector_norm(actions - mu[:, None, :], dim=-1)
    diagnostics = {
        "near_occupancy": near.mean().item(),
        "far_occupancy": far.mean().item(),
        "raw_distance_mean": raw.mean().item(),
        "standardized_distance_mean": standardized.mean().item(),
    }
    return near_loss, far_loss, diagnostics


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
            predicted
            if fixed_sigma is None
            else torch.full_like(predicted, math.log(fixed_sigma))
        )
        reward = reward_from_optimum(mu, split.a_star, protocol.reward_width)
        axis = ((mu - split.a_plus) * split.direction).sum(-1)
        normalized = axis / protocol.gap_to_unseen_optimum
        sigma = torch.exp(log_std)
        return {
            "reward": reward.mean().item(),
            "normalized_extrapolation_displacement": normalized.mean().item(),
            "distance_to_a_plus": torch.linalg.vector_norm(
                mu - split.a_plus,
                dim=-1,
            ).mean().item(),
            "distance_to_a_star": torch.linalg.vector_norm(
                mu - split.a_star,
                dim=-1,
            ).mean().item(),
            "sigma_mean": sigma.mean().item(),
            "sigma_min": sigma.min().item(),
            "sigma_max": sigma.max().item(),
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
    with torch.no_grad():
        _, log_sigma = actor(split.s)
        sigma = torch.exp(log_sigma)
    finite_log_sigma = bool(torch.isfinite(log_sigma).all().item())
    finite_sigma = bool(torch.isfinite(sigma).all().item())
    log_min = float(log_sigma.min().item()) if finite_log_sigma else float("nan")
    log_max = float(log_sigma.max().item()) if finite_log_sigma else float("nan")
    return {
        "log_sigma_min_all_states": log_min,
        "log_sigma_max_all_states": log_max,
        "sigma_output_finite_all_states": finite_sigma,
        "log_sigma_output_finite_all_states": finite_log_sigma,
        "support_contraction_boundary": (
            finite_log_sigma and log_min < -protocol.log_sigma_event_boundary
        ),
        "unexpected_support_expansion_boundary": (
            finite_log_sigma and log_max > protocol.log_sigma_event_boundary
        ),
    }


def event_flags(
    *,
    task_performance_collapse: bool,
    support: dict[str, Any],
    finite_parameters: bool,
    environment_valid: bool = True,
) -> EventFlags:
    numerical = bool(
        not finite_parameters
        or not support["log_sigma_output_finite_all_states"]
        or not support["sigma_output_finite_all_states"]
    )
    boundary = bool(
        support["support_contraction_boundary"]
        or support["unexpected_support_expansion_boundary"]
    )
    return EventFlags(
        task_performance_collapse=task_performance_collapse,
        support_or_probability_boundary=boundary,
        nan_inf_numerical_failure=numerical,
        environment_invalid=not environment_valid,
    )
