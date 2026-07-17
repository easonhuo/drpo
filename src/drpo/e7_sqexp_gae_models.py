"""Canonical E7 actor/critic geometry and actor objectives."""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
from torch import nn

from drpo.e7_sqexp_gae_contract import (
    CONTROL_IDS, EXPECTILE_TAU, PPO_CLIP_EPSILON, PPO_OLD_POLICY_CADENCE,
    REFERENCE_DISTANCE,
)

class CanonicalCritic(nn.Module):
    """Recovered E7 2x256 ReLU value network with orthogonal initialization."""

    def __init__(self, observation_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        _orthogonal(self.net)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations).squeeze(-1)


class CanonicalActor(nn.Module):
    """Recovered E7 separate Gaussian actor; mean is tanh-bounded."""

    def __init__(self, observation_dim: int, action_dim: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(observation_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))
        _orthogonal(self.trunk)
        _orthogonal(self.mean)

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(observations)
        mean = torch.tanh(self.mean(features))
        log_std = self.log_std.clamp(-5.0, 2.0).expand_as(mean)
        return mean, log_std

    def log_prob(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        mean, log_std = self(observations)
        return torch.distributions.Normal(mean, log_std.exp()).log_prob(actions).sum(dim=-1)

    def standardized_rms_distance(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        mean, log_std = self(observations)
        with torch.no_grad():
            z = (actions - mean.detach()) / log_std.detach().exp().clamp_min(1e-8)
            return z.square().mean(dim=-1).sqrt()


def _orthogonal(module: nn.Module) -> None:
    gain = math.sqrt(2.0)
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, gain=gain)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


def squared_exponential_factor(
    distance: torch.Tensor,
    coefficient: float,
    reference_distance: float = REFERENCE_DISTANCE,
) -> torch.Tensor:
    if coefficient < 0.0 or reference_distance <= 0.0:
        raise ValueError("coefficient must be non-negative and reference_distance positive")
    if not bool(torch.isfinite(distance).all()):
        raise FloatingPointError("non-finite standardized distance")
    exponent = torch.clamp(
        -float(coefficient) * (distance / float(reference_distance)).square(),
        min=-40.0,
        max=0.0,
    )
    return torch.exp(exponent)


def controlled_advantage(
    advantage: torch.Tensor,
    distance: torch.Tensor,
    control_id: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if control_id not in CONTROL_IDS:
        raise ValueError(f"unknown control: {control_id}")
    negative = advantage < 0
    factor = torch.ones_like(advantage)
    if control_id == "positive_only":
        factor = torch.where(negative, torch.zeros_like(factor), factor)
    else:
        coefficient = float(control_id.rsplit("c", 1)[1])
        shape = squared_exponential_factor(distance, coefficient)
        factor = torch.where(negative, shape, factor)
    return advantage * factor, factor


def actor_objective(
    *,
    actor: CanonicalActor,
    old_actor: CanonicalActor | None,
    observations: torch.Tensor,
    actions: torch.Tensor,
    advantages: torch.Tensor,
    actor_mode: str,
    control_id: str,
    clip_epsilon: float = PPO_CLIP_EPSILON,
) -> tuple[torch.Tensor, dict[str, float]]:
    distance = actor.standardized_rms_distance(observations, actions)
    weighted, factor = controlled_advantage(advantages, distance, control_id)
    current_log_prob = actor.log_prob(observations, actions)
    if actor_mode == "a2c":
        objective = current_log_prob * weighted
        ratio = torch.ones_like(objective)
        clipped = torch.zeros_like(objective, dtype=torch.bool)
    elif actor_mode == "ppo_clip_k4":
        if old_actor is None:
            raise ValueError("PPO requires an old-policy reference")
        with torch.no_grad():
            old_log_prob = old_actor.log_prob(observations, actions)
        ratio = torch.exp(torch.clamp(current_log_prob - old_log_prob, min=-20.0, max=20.0))
        clipped_ratio = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
        unclipped_objective = ratio * weighted
        clipped_objective = clipped_ratio * weighted
        objective = torch.minimum(unclipped_objective, clipped_objective)
        clipped = unclipped_objective > clipped_objective
    else:
        raise ValueError(f"unknown actor mode: {actor_mode}")
    loss = -objective.mean()
    negative = advantages < 0
    diagnostics = {
        "loss": float(loss.detach().cpu()),
        "positive_fraction": float((~negative).float().mean().detach().cpu()),
        "negative_fraction": float(negative.float().mean().detach().cpu()),
        "negative_factor_mean": (
            float(factor[negative].mean().detach().cpu()) if bool(negative.any()) else float("nan")
        ),
        "negative_distance_mean": (
            float(distance[negative].mean().detach().cpu()) if bool(negative.any()) else float("nan")
        ),
        "ratio_mean": float(ratio.mean().detach().cpu()),
        "ratio_outside_fraction": float(
            ((ratio < 1.0 - clip_epsilon) | (ratio > 1.0 + clip_epsilon))
            .float()
            .mean()
            .detach()
            .cpu()
        ),
        "objective_clip_fraction": float(clipped.float().mean().detach().cpu()),
    }
    return loss, diagnostics


class OldPolicyCadence:
    def __init__(self, cadence: int = PPO_OLD_POLICY_CADENCE):
        if cadence <= 0:
            raise ValueError("old-policy cadence must be positive")
        self.cadence = int(cadence)
        self.refresh_count = 0
        self.first_refresh_step: int | None = None
        self.last_refresh_step: int | None = None

    def should_refresh_before(self, step: int) -> bool:
        if step <= 0:
            raise ValueError("optimizer step must be positive")
        return (step - 1) % self.cadence == 0

    def refresh(self, old_actor: nn.Module, actor: nn.Module, step: int) -> None:
        if not self.should_refresh_before(step):
            raise ValueError(f"step {step} is not a scheduled K={self.cadence} refresh")
        old_actor.load_state_dict(copy.deepcopy(actor.state_dict()))
        old_actor.eval()
        for parameter in old_actor.parameters():
            parameter.requires_grad_(False)
        self.refresh_count += 1
        if self.first_refresh_step is None:
            self.first_refresh_step = int(step)
        self.last_refresh_step = int(step)


def expectile_critic_loss(error: torch.Tensor, tau: float = EXPECTILE_TAU) -> torch.Tensor:
    if not 0.5 <= tau < 1.0:
        raise ValueError("expectile tau must be in [0.5,1)")
    weight = torch.where(error > 0, torch.full_like(error, tau), torch.full_like(error, 1.0 - tau))
    return (weight * error.square()).mean()


def fit_observation_normalizer(observations: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    selected = np.asarray(observations, dtype=np.float32)[np.asarray(indices, dtype=np.int64)]
    mean = selected.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = selected.std(axis=0, dtype=np.float64).astype(np.float32)
    return mean, np.maximum(std, 1e-6)


def normalize_observations(array: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((np.asarray(array, dtype=np.float32) - mean) / std).astype(np.float32)


