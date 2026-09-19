"""Isotropic Gaussian policy primitives used by the continuous paper experiments."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class GaussianActor(nn.Module):
    """Two-hidden-layer MLP with a state-conditioned isotropic log scale."""

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

    def features(self, states: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.fc2(torch.relu(self.fc1(states))))

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features(states)
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
    """Return log density for ``mu=[B,D]``, ``log_std=[B]``, actions ``[B,K,D]``."""

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
    """Return exact isotropic-Gaussian output-score components."""

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
