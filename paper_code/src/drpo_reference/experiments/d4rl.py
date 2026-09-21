"""D4RL-9 SNA2C-IQLV training implementation.

One actor/critic/optimizer lifecycle serves all nine locomotion tasks. The public
path uses the manuscript DRPO negative taper; historical ExpRank support remains
internal only for compatibility with the recovered trainer lineage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn

D4RL_METHODS = (
    "drpo",
    "positive_only",
    "global",
    "reciprocal_linear",
    "reciprocal_quadratic",
)
CANONICAL_ALPHA = 0.11
REFERENCE_DISTANCE = 2.0
RECIPROCAL_LINEAR_COEFFICIENT = 0.4362580032734791
RECIPROCAL_QUADRATIC_COEFFICIENT = 0.5520268617673281
EXPONENTIAL_COEFFICIENT = 0.374162511054291
DRPO_EXPONENTIAL_MULTIPLIER = 1.0


ENVIRONMENTS = ("halfcheetah", "hopper", "walker2d")
DATASET_TIERS = ("medium", "medium-replay", "medium-expert")
_REFERENCE_SCORES = {
    "halfcheetah": (-280.178953, 12135.0),
    "hopper": (-20.272305, 3234.3),
    "walker2d": (1.629008, 4592.3),
}
_ENV_IDS = {
    "halfcheetah": "HalfCheetah-v4",
    "hopper": "Hopper-v4",
    "walker2d": "Walker2d-v4",
}


@dataclass(frozen=True)
class D4RLTaskSpec:
    task_id: str
    dataset_basename: str
    env_id: str
    normalized_score_reference_min: float
    normalized_score_reference_max: float


def _make_task(environment: str, dataset_tier: str) -> D4RLTaskSpec:
    minimum, maximum = _REFERENCE_SCORES[environment]
    return D4RLTaskSpec(
        task_id=f"{environment}-{dataset_tier}-v2",
        dataset_basename=f"{environment}_{dataset_tier.replace('-', '_')}-v2.hdf5",
        env_id=_ENV_IDS[environment],
        normalized_score_reference_min=minimum,
        normalized_score_reference_max=maximum,
    )


D4RL9_TASKS = tuple(
    _make_task(environment, dataset_tier)
    for dataset_tier in DATASET_TIERS
    for environment in ENVIRONMENTS
)
D4RL9_BY_ID = {task.task_id: task for task in D4RL9_TASKS}


def resolve_d4rl_task(task_id: str) -> D4RLTaskSpec:
    return D4RL9_BY_ID[task_id]


@dataclass(frozen=True)
class OfflineData:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    terminals: np.ndarray
    timeouts: np.ndarray


def load_d4rl_hdf5(path: str | Path) -> OfflineData:
    """Load the arrays used by the compact D4RL runner."""

    with h5py.File(Path(path), "r") as handle:
        observations = np.asarray(handle["observations"], dtype=np.float32)
        actions = np.asarray(handle["actions"], dtype=np.float32)
        rewards = np.asarray(handle["rewards"], dtype=np.float32).reshape(-1)
        terminals = np.asarray(handle["terminals"], dtype=np.bool_).reshape(-1)
        timeouts = (
            np.asarray(handle["timeouts"], dtype=np.bool_).reshape(-1)
            if "timeouts" in handle
            else np.zeros(len(rewards), dtype=np.bool_)
        )
        next_observations = (
            np.asarray(handle["next_observations"], dtype=np.float32)
            if "next_observations" in handle
            else np.concatenate([observations[1:], observations[-1:]], axis=0)
        )
    return OfflineData(
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        terminals=terminals,
        timeouts=timeouts,
    )


@dataclass(frozen=True)
class CanonicalExpRankTrainingConfig:
    steps: int
    batch_size: int
    learning_rate: float = 3.0e-4
    gamma: float = 0.99
    alpha: float = CANONICAL_ALPHA
    tau: float = 0.7
    temperature: float = 1.0


@dataclass(frozen=True)
class CanonicalD4RLDataset:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    terminals: np.ndarray

    @property
    def size(self) -> int:
        return int(self.observations.shape[0])

    @property
    def observation_dim(self) -> int:
        return int(self.observations.shape[1])

    @property
    def action_dim(self) -> int:
        return int(self.actions.shape[1])


def _orthogonal_init(module: nn.Module) -> None:
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, gain=math.sqrt(2.0))
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


class CanonicalActor(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_size: int = 256,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_size, action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim) * 1.0e-3)
        _orthogonal_init(self.net)
        _orthogonal_init(self.mu)

    def forward(
        self,
        observations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean = torch.tanh(self.mu(self.net(observations)))
        log_std = torch.clamp(self.log_std, -5.0, 2.0)
        return mean, log_std.expand_as(mean)


class CanonicalCritic(nn.Module):
    def __init__(self, observation_dim: int, hidden_size: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        _orthogonal_init(self.net)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


def canonical_exprank_negative_weights(
    negative_advantages: torch.Tensor,
    *,
    alpha: float,
    temperature: float,
) -> torch.Tensor:
    count = int(negative_advantages.numel())
    if count == 0:
        return negative_advantages.clone()
    if count == 1:
        return torch.tensor(
            [float(alpha)],
            device=negative_advantages.device,
            dtype=negative_advantages.dtype,
        )
    order = negative_advantages.argsort()
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(count, device=negative_advantages.device)
    score = 1.0 - ranks.float() / float(count - 1)
    return float(alpha) * torch.exp(torch.clamp(-float(temperature) * score, min=-20.0))


def canonical_standardized_action_remoteness(
    mean: torch.Tensor,
    log_std: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    """Detached dimension-normalized squared standardized remoteness."""

    if log_std.shape != mean.shape:
        log_std = log_std.expand_as(mean)
    safe_log_std = torch.clamp(log_std, min=-20.0, max=5.0)
    with torch.no_grad():
        standardized = (actions.detach() - mean.detach()) / safe_log_std.detach().exp().clamp_min(
            1.0e-8
        )
        return standardized.square().mean(dim=-1)


def canonical_method_negative_factors(
    negative_advantages: torch.Tensor,
    negative_remoteness: torch.Tensor,
    *,
    method: str,
    exprank_temperature: float,
    exprank_alpha: float = CANONICAL_ALPHA,
) -> torch.Tensor:
    """Return detached negative-side factors for one method.

    The manuscript coordinate is D=mean(((a-mu)/sigma)^2). With tau=0 and
    scale=REFERENCE_DISTANCE**2, x=D/scale and r=sqrt(x). DRPO therefore uses
    exp(-lambda*x), i.e. the squared-remoteness exponential kernel exercised by
    the E7 implementation, with unit near-field multiplier.
    """

    negative_advantages = negative_advantages.detach()
    negative_remoteness = negative_remoteness.detach()
    if method == "exprank":
        return canonical_exprank_negative_weights(
            negative_advantages,
            alpha=exprank_alpha,
            temperature=exprank_temperature,
        )
    if method == "positive_only":
        return torch.zeros_like(negative_advantages)
    if method == "signed":
        return torch.ones_like(negative_advantages)
    if method == "global":
        return torch.full_like(negative_advantages, float(exprank_alpha))

    normalized_excess = torch.relu(
        negative_remoteness / float(REFERENCE_DISTANCE**2)
    )
    radial = torch.sqrt(normalized_excess)
    if method == "reciprocal_linear":
        return 1.0 / (1.0 + RECIPROCAL_LINEAR_COEFFICIENT * radial)
    if method == "reciprocal_quadratic":
        return 1.0 / (1.0 + RECIPROCAL_QUADRATIC_COEFFICIENT * normalized_excess)
    if method in {"drpo", "exponential"}:
        return DRPO_EXPONENTIAL_MULTIPLIER * torch.exp(
            torch.clamp(
                -EXPONENTIAL_COEFFICIENT * normalized_excess,
                min=-40.0,
                max=0.0,
            )
        )
    raise AssertionError(f"unreachable D4RL method: {method}")


class SNA2CIQLVExpRankAgent:
    """SNA2C-IQLV actor/critic update with manuscript DRPO weighting.

    The historical class name is retained to avoid replacing the surrounding
    recovered trainer structure.
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        *,
        learning_rate: float = 3.0e-4,
        gamma: float = 0.99,
        alpha: float = CANONICAL_ALPHA,
        tau: float = 0.7,
        temperature: float = 1.0,
        device: torch.device | str = "cpu",
        method: str = "drpo",
    ) -> None:
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.tau = float(tau)
        self.temperature = float(temperature)
        self.method = method
        self.device = torch.device(device)
        self.actor = CanonicalActor(observation_dim, action_dim).to(self.device)
        self.critic = CanonicalCritic(observation_dim).to(self.device)
        self.a_opt = torch.optim.Adam(
            self.actor.parameters(),
            lr=float(learning_rate),
        )
        self.c_opt = torch.optim.Adam(
            self.critic.parameters(),
            lr=float(learning_rate),
        )

    @torch.no_grad()
    def get_action(
        self,
        observation: np.ndarray | torch.Tensor,
    ) -> np.ndarray:
        tensor = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        mean, _ = self.actor(tensor)
        return mean.squeeze(0).cpu().numpy()

    def update(
        self,
        observations: np.ndarray | torch.Tensor,
        actions: np.ndarray | torch.Tensor,
        rewards: np.ndarray | torch.Tensor,
        next_observations: np.ndarray | torch.Tensor,
        dones: np.ndarray | torch.Tensor,
    ) -> float:
        states = _float_tensor(observations, self.device)
        action_tensor = _float_tensor(actions, self.device)
        reward_tensor = _float_tensor(rewards, self.device)
        next_states = _float_tensor(next_observations, self.device)
        done_tensor = torch.as_tensor(dones, dtype=torch.bool, device=self.device)

        with torch.no_grad():
            next_value = self.critic(next_states).squeeze(-1)
            target = reward_tensor + self.gamma * next_value * (~done_tensor).float()
        value = self.critic(states).squeeze(-1)
        advantage = target - value.detach()

        mean, log_std = self.actor(states)
        transformed = advantage.clone()
        negative = advantage < 0
        if negative.any():
            remoteness = canonical_standardized_action_remoteness(
                mean,
                log_std,
                action_tensor,
            )
            with torch.no_grad():
                factor = canonical_method_negative_factors(
                    advantage[negative],
                    remoteness[negative],
                    method=self.method,
                    exprank_temperature=self.temperature,
                    exprank_alpha=self.alpha,
                )
            transformed[negative] = advantage[negative] * factor

        distribution = torch.distributions.Normal(mean, log_std.exp())
        log_probability = distribution.log_prob(action_tensor).sum(dim=-1)
        actor_loss = -(log_probability * transformed).mean()

        value_error = target - value
        expectile_weight = torch.where(
            value_error > 0,
            self.tau,
            1.0 - self.tau,
        )
        critic_loss = (expectile_weight * value_error.square()).mean()

        self.a_opt.zero_grad()
        actor_loss.backward()
        self.a_opt.step()
        self.c_opt.zero_grad()
        critic_loss.backward()
        self.c_opt.step()
        return float(actor_loss.item() + 0.5 * critic_loss.item())


def _float_tensor(
    value: np.ndarray | torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.float32)
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def reward_norm_locomotion(
    rewards: np.ndarray,
    terminals: np.ndarray,
    timeouts: np.ndarray,
) -> np.ndarray:
    returns: list[float] = []
    start = 0
    for index in range(len(rewards)):
        if terminals[index] or timeouts[index] or index == len(rewards) - 1:
            returns.append(float(rewards[start : index + 1].sum()))
            start = index + 1
    if len(returns) < 2:
        return rewards
    span = max(returns) - min(returns)
    if span < 1.0e-8:
        return rewards
    return rewards / span * 1000.0


def prepare_canonical_locomotion_dataset(
    data: OfflineData,
    *,
    gamma: float = 0.99,
) -> CanonicalD4RLDataset:
    actions = np.clip(data.actions, -1.0 + 1.0e-5, 1.0 - 1.0e-5)
    rewards = reward_norm_locomotion(
        data.rewards,
        data.terminals,
        data.timeouts,
    ).astype(np.float32)
    return CanonicalD4RLDataset(
        observations=data.observations,
        actions=actions,
        rewards=rewards,
        next_observations=data.next_observations,
        terminals=data.terminals,
    )


def train_canonical_method(
    *,
    dataset: CanonicalD4RLDataset,
    seed: int,
    config: CanonicalExpRankTrainingConfig,
    method: str,
    device: torch.device | str = "cpu",
) -> SNA2CIQLVExpRankAgent:
    """Train one method with the canonical D4RL update sequence."""

    torch.manual_seed(int(seed))
    resolved_device = torch.device(device)
    agent = SNA2CIQLVExpRankAgent(
        dataset.observation_dim,
        dataset.action_dim,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        alpha=config.alpha,
        tau=config.tau,
        temperature=config.temperature,
        device=resolved_device,
        method=method,
    )
    tensors = {
        "s": torch.from_numpy(dataset.observations).to(resolved_device),
        "a": torch.from_numpy(dataset.actions).to(resolved_device),
        "r": torch.from_numpy(dataset.rewards).to(resolved_device),
        "ns": torch.from_numpy(dataset.next_observations).to(resolved_device),
        "d": torch.from_numpy(dataset.terminals).to(resolved_device),
    }
    generator = torch.Generator(device=resolved_device)
    generator.manual_seed(int(seed))
    for step in range(1, config.steps + 1):
        indices = torch.randint(
            0,
            dataset.size,
            (config.batch_size,),
            generator=generator,
            device=resolved_device,
        )
        agent.update(
            tensors["s"].index_select(0, indices),
            tensors["a"].index_select(0, indices),
            tensors["r"].index_select(0, indices),
            tensors["ns"].index_select(0, indices),
            tensors["d"].index_select(0, indices),
        )
    return agent
