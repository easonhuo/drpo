"""Canonical Hopper E7-Q2 value and squashed-Gaussian networks."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn

EPS = 1.0e-6


def activation_layer(name: str) -> nn.Module:
    normalized = name.strip().lower()
    if normalized == "tanh":
        return nn.Tanh()
    if normalized == "relu":
        return nn.ReLU()
    raise ValueError(f"unsupported E7 network activation: {name}")


def apply_orthogonal_init(
    module: nn.Module,
    gain: float,
) -> None:
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(
                layer.weight,
                gain=float(gain),
            )
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


def make_mlp(
    input_dim: int,
    output_dim: int,
    hidden_sizes: Sequence[int],
    activation: str = "tanh",
    init_scheme: str = "default",
    init_gain: float = 1.0,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    width = input_dim
    for hidden in hidden_sizes:
        layers.extend(
            [
                nn.Linear(width, hidden),
                activation_layer(activation),
            ]
        )
        width = hidden
    layers.append(nn.Linear(width, output_dim))
    network = nn.Sequential(*layers)
    init_name = init_scheme.strip().lower()
    if init_name == "orthogonal":
        apply_orthogonal_init(network, init_gain)
    elif init_name != "default":
        raise ValueError(f"unsupported E7 init scheme: {init_scheme}")
    return network


class ValueNetwork(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: Sequence[int],
        activation: str = "tanh",
        init_scheme: str = "default",
        init_gain: float = 1.0,
    ) -> None:
        super().__init__()
        self.net = make_mlp(
            obs_dim,
            1,
            hidden_sizes,
            activation,
            init_scheme,
            init_gain,
        )

    def forward(
        self,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(observations).squeeze(-1)


class SquashedGaussianPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int],
        log_std_min: float,
        log_std_max: float,
        action_clip_epsilon: float,
        activation: str = "tanh",
        init_scheme: str = "default",
        init_gain: float = 1.0,
    ) -> None:
        super().__init__()
        self.mean_net = make_mlp(
            obs_dim,
            action_dim,
            hidden_sizes,
            activation,
            init_scheme,
            init_gain,
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.action_clip_epsilon = float(action_clip_epsilon)

    def latent_parameters(
        self,
        observations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.mean_net(observations)
        log_std = self.log_std.clamp(
            self.log_std_min,
            self.log_std_max,
        )
        return mean, log_std.expand_as(mean)

    def action_mean(
        self,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        mean, _ = self.latent_parameters(observations)
        return torch.tanh(mean)

    def inverse_action(
        self,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        epsilon = self.action_clip_epsilon
        clipped = actions.clamp(
            -1.0 + epsilon,
            1.0 - epsilon,
        )
        return torch.atanh(clipped)

    def log_prob(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        latent = self.inverse_action(actions)
        mean, log_std = self.latent_parameters(observations)
        inverse_variance = torch.exp(-2.0 * log_std)
        gaussian = -0.5 * (
            (latent - mean).square() * inverse_variance + 2.0 * log_std + math.log(2.0 * math.pi)
        )
        bounded_actions = actions.clamp(
            -1.0 + EPS,
            1.0 - EPS,
        )
        jacobian = torch.log(
            torch.clamp(
                1.0 - bounded_actions.square(),
                min=EPS,
            )
        )
        return (gaussian - jacobian).sum(dim=-1)

    def score_components(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        latent = self.inverse_action(actions)
        mean, log_std = self.latent_parameters(observations)
        standard_deviation = torch.exp(log_std)
        standardized = (latent - mean) / standard_deviation
        mean_score = (latent - mean) / standard_deviation.square()
        log_scale_score = standardized.square() - 1.0
        mean_norm = torch.linalg.vector_norm(
            mean_score,
            dim=-1,
        )
        log_scale_norm = torch.linalg.vector_norm(
            log_scale_score,
            dim=-1,
        )
        joint_norm = torch.sqrt(mean_norm.square() + log_scale_norm.square())
        radius = torch.linalg.vector_norm(
            standardized,
            dim=-1,
        )
        corrected_quadratic = standardized.square().sum(dim=-1)
        action_mean = torch.tanh(mean)
        return {
            "latent": latent,
            "mean": mean,
            "log_std": log_std,
            "z": standardized,
            "radius": radius,
            "mean_score": mean_score,
            "mean_score_norm": mean_norm,
            "log_scale_score": log_scale_score,
            "raw_log_scale_score_norm": (log_scale_norm),
            "corrected_q_xi": corrected_quadratic,
            "joint_output_score_norm": joint_norm,
            "log_scale_to_mean_ratio": (log_scale_norm / mean_norm.clamp_min(EPS)),
            "raw_action_distance": (
                torch.linalg.vector_norm(
                    actions - action_mean,
                    dim=-1,
                )
            ),
            "pre_squash_distance": (
                torch.linalg.vector_norm(
                    latent - mean,
                    dim=-1,
                )
            ),
        }

    def standardized_distance(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        return self.score_components(
            observations,
            actions,
        )["radius"]

    def output_score_norm(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        return self.score_components(
            observations,
            actions,
        )["joint_output_score_norm"]
