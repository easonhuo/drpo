"""Isotropic Gaussian policy primitives used by the continuous paper experiments."""

from __future__ import annotations

import math

import torch
from torch import nn


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
