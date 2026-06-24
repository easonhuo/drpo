from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def make_mlp(
    input_dim: int,
    output_dim: int,
    hidden_sizes: Sequence[int],
    activation: type[nn.Module] = nn.Tanh,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev_dim = input_dim
    for hidden_dim in hidden_sizes:
        layers.extend([nn.Linear(prev_dim, hidden_dim), activation()])
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


class GaussianMLPPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (256, 256),
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ) -> None:
        super().__init__()
        self.mean_net = make_mlp(obs_dim, action_dim, hidden_sizes)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, obs: torch.Tensor) -> torch.distributions.Normal:
        mean = self.mean_net(obs)
        log_std = self.log_std.clamp(self.log_std_min, self.log_std_max)
        std = log_std.exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def log_prob(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        dist = self(obs)
        return dist.log_prob(actions).sum(dim=-1)
