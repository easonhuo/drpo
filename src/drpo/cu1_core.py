"""Shared C-U1 environment, Gaussian actor, and distance primitives.

This module is the single implementation source for the C-U1 state/action
geometry and policy architecture. Experiment runners keep their own protocols,
training loops, controls, and reporting, but they must construct environments
and actors through this module so that a new experiment cannot silently fork
the frozen C-U1 definition.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol

import torch
import torch.nn as nn


class CU1Protocol(Protocol):
    state_dim: int
    action_dim: int
    n_train_states: int
    n_test_states: int
    gap_to_unseen_optimum: float
    negative_offset_from_positive: float
    positive_contour_radius: float
    negative_contour_radius: float
    reward_width: float
    baseline: float
    positive_angle_1: float
    hidden_dim: int
    initial_sigma: float


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


def base_from_state(s: torch.Tensor) -> torch.Tensor:
    b1 = 0.70 * torch.tanh(
        0.85 * s[:, 0]
        - 0.30 * s[:, 1] * s[:, 2]
        + 0.20 * torch.sin(1.6 * s[:, 3])
    )
    b2 = 0.65 * torch.tanh(
        -0.50 * s[:, 1]
        + 0.35 * torch.cos(1.1 * s[:, 4])
        + 0.22 * s[:, 0] * s[:, 5]
    )
    return torch.stack([b1, b2], dim=1)


def task_direction_from_state(s: torch.Tensor) -> torch.Tensor:
    angle = 1.15 * torch.tanh(
        0.75 * s[:, 0] + 0.50 * s[:, 2] - 0.30 * s[:, 5]
    )
    angle = angle + 0.30 * torch.sin(1.35 * s[:, 1])
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
    """Return the four equal-reward angles whose centroid is ``a_plus``."""
    theta1 = protocol.positive_angle_1
    cos_theta2 = (
        2.0 * protocol.gap_to_unseen_optimum / protocol.positive_contour_radius
        - math.cos(theta1)
    )
    if not (-1.0 <= cos_theta2 <= 1.0):
        raise RuntimeError("Invalid positive contour geometry")
    theta2 = math.acos(cos_theta2)
    return torch.tensor(
        [
            math.pi - theta1,
            math.pi + theta1,
            math.pi - theta2,
            math.pi + theta2,
        ],
        dtype=dtype,
    )


def negative_angles(dtype: torch.dtype) -> torch.Tensor:
    """Return the frozen eight-point negative contour.

    Index 0 is ``a_minus`` and index 4 is the opposite/farthest contour copy
    at the positive-only initialization.
    """
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


def make_split(s: torch.Tensor, protocol: CU1Protocol) -> Split:
    plus = base_from_state(s)
    direction = task_direction_from_state(s)
    ortho = orthogonal(direction)
    star = plus + protocol.gap_to_unseen_optimum * direction
    minus = plus - protocol.negative_offset_from_positive * direction

    pos_angles = positive_angles(protocol, s.dtype).to(s.device)
    pos_direction = (
        torch.cos(pos_angles)[None, :, None] * direction[:, None, :]
        + torch.sin(pos_angles)[None, :, None] * ortho[:, None, :]
    )
    positive_actions = (
        star[:, None, :] + protocol.positive_contour_radius * pos_direction
    )
    positive_rewards = reward_from_optimum(
        positive_actions,
        star[:, None, :],
        protocol.reward_width,
    )
    positive_advantages = positive_rewards - protocol.baseline

    neg_angles = negative_angles(s.dtype).to(s.device)
    neg_direction = (
        torch.cos(neg_angles)[None, :, None] * direction[:, None, :]
        + torch.sin(neg_angles)[None, :, None] * ortho[:, None, :]
    )
    negative_actions = (
        star[:, None, :] + protocol.negative_contour_radius * neg_direction
    )
    negative_rewards = reward_from_optimum(
        negative_actions,
        star[:, None, :],
        protocol.reward_width,
    )
    negative_advantages = negative_rewards - protocol.baseline

    return Split(
        s=s,
        a_plus=plus,
        a_star=star,
        a_minus=minus,
        direction=direction,
        orthogonal=ortho,
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
    device: torch.device,
    dtype: torch.dtype,
) -> Environment:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    train_states = torch.randn(
        protocol.n_train_states,
        protocol.state_dim,
        generator=generator,
        dtype=dtype,
    ).to(device)
    test_states = torch.randn(
        protocol.n_test_states,
        protocol.state_dim,
        generator=generator,
        dtype=dtype,
    ).to(device)
    return Environment(
        train=make_split(train_states, protocol),
        test=make_split(test_states, protocol),
    )


def audit_environment(environment: Environment, protocol: CU1Protocol) -> dict[str, Any]:
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
        train.negative_actions - train.a_plus[:, None, :], dim=-1
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


class GaussianActor(nn.Module):
    """Frozen C-U1 two-layer MLP with isotropic state-conditioned log scale."""

    def __init__(
        self,
        *,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        initial_sigma: float,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.log_std_head.weight)
        nn.init.constant_(self.log_std_head.bias, math.log(initial_sigma))

    def features(self, s: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.fc2(torch.relu(self.fc1(s))))

    def forward(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features(s)
        return self.mu_head(features), self.log_std_head(features).squeeze(-1)

    def mean_parameters(self) -> list[nn.Parameter]:
        return (
            list(self.fc1.parameters())
            + list(self.fc2.parameters())
            + list(self.mu_head.parameters())
        )

    def all_parameters(self) -> list[nn.Parameter]:
        return list(self.parameters())


def gaussian_log_prob(
    mu: torch.Tensor,
    log_std: torch.Tensor,
    actions: torch.Tensor,
    action_dim: int,
) -> torch.Tensor:
    """Isotropic Gaussian log probability for ``actions=[B,K,D]``."""
    inverse_std = torch.exp(-log_std)[:, None, None]
    standardized = (actions - mu[:, None, :]) * inverse_std
    return (
        -0.5 * standardized.square().sum(-1)
        - action_dim * log_std[:, None]
        - 0.5 * action_dim * math.log(2.0 * math.pi)
    )


def standardized_distance(
    mu: torch.Tensor,
    log_std: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    raw = torch.linalg.vector_norm(actions - mu[:, None, :], dim=-1)
    return raw / torch.exp(log_std)[:, None]


def gaussian_output_components(
    mu: torch.Tensor,
    log_std: torch.Tensor,
    actions: torch.Tensor,
    action_dim: int,
) -> dict[str, torch.Tensor]:
    """Return exact isotropic-Gaussian output-score components.

    The helper operates in policy-output coordinates ``(mu, log_sigma)`` and
    intentionally excludes the neural-network Jacobian/pullback.
    """
    if mu.ndim != 2 or actions.ndim != 3:
        raise ValueError("expected mu=[B,D] and actions=[B,K,D]")
    if mu.shape[0] != actions.shape[0] or mu.shape[1] != actions.shape[2]:
        raise ValueError("mu/actions shape mismatch")
    if int(mu.shape[1]) != int(action_dim):
        raise ValueError("action_dim does not match tensor width")

    log_std_flat = log_std.reshape(mu.shape[0])
    sigma = torch.exp(log_std_flat)
    sigma2 = sigma.square()[:, None]
    delta = actions - mu[:, None, :]
    raw_distance = torch.linalg.vector_norm(delta, dim=-1)
    standardized2 = raw_distance.square() / sigma2
    mean_score = raw_distance / sigma2
    log_scale_score = standardized2 - action_dim
    corrected_log_scale = log_scale_score + action_dim
    joint_score = torch.sqrt(mean_score.square() + log_scale_score.square())
    return {
        "sigma": sigma,
        "sigma2": sigma2,
        "raw_distance": raw_distance,
        "standardized_distance": torch.sqrt(standardized2),
        "standardized2": standardized2,
        "mean_score": mean_score,
        "log_scale_score": log_scale_score,
        "corrected_log_scale": corrected_log_scale,
        "joint_score": joint_score,
        "normalized_mean": mean_score * sigma2,
        "normalized_quadratic": corrected_log_scale * sigma2,
    }


def distance_taper_weight(
    distance: torch.Tensor,
    *,
    family: str,
    rho: float,
    reference_distance: float,
) -> torch.Tensor:
    """Return an aligned positive taper on one standardized distance.

    ``rho`` fixes the common reference attenuation ``w(reference_distance)``.
    The caller must detach ``distance`` before this function when the taper is
    intended as pure sample reweighting rather than a differentiable distance
    regularizer.
    """
    if not (0.0 < rho <= 1.0):
        raise ValueError("rho must lie in (0, 1]")
    if reference_distance <= 0.0:
        raise ValueError("reference_distance must be positive")
    normalized = distance / reference_distance
    if family == "unweighted":
        return torch.ones_like(normalized)
    if family == "reciprocal_linear":
        coefficient = 1.0 / rho - 1.0
        return 1.0 / (1.0 + coefficient * normalized)
    if family == "reciprocal_quadratic":
        coefficient = 1.0 / rho - 1.0
        return 1.0 / (1.0 + coefficient * normalized.square())
    if family == "exponential":
        coefficient = -math.log(rho)
        return torch.exp(-coefficient * normalized)
    raise ValueError(f"unknown taper family: {family}")
