"""D4RL-9 SNA2C-IQLV-ExpRank performance implementation.

One migrated trainer serves all nine locomotion tasks. The Hopper E7-Q2
frozen-advantage mechanism trainer remains scientifically distinct.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from drpo_reference.common.io import atomic_json
from drpo_reference.external.d4rl_tasks import (
    D4RL9_TASKS,
    D4RLTaskSpec,
    validate_d4rl9_matrix,
    validate_dataset_path,
)
from drpo_reference.external.hopper_data import OfflineData

D4RL9_EXPERIMENT_ID = "EXT-H-E7-BENCH-01"
D4RL9_RUNNER_VERSION = "0.4.0-canonical-exprank-migrated"
TaskRunner = Callable[..., dict[str, Any]]
Evaluator = Callable[["SNA2CIQLVExpRankAgent", int], Mapping[str, float]]


@dataclass(frozen=True)
class D4RLPerformanceBackendSpec:
    backend_id: str
    algorithm_family: str
    source_paths: tuple[str, ...]
    implementation_selected: bool
    implementation_migrated: bool
    protocol_status: str
    protocol_frozen: bool
    formal_task_matrix_eligible: bool
    mechanism_runner_reusable: bool
    shared_contracts: tuple[str, ...]
    distinct_contracts: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.backend_id or not self.algorithm_family:
            raise ValueError("D4RL backend identity must be non-empty")
        if not self.source_paths:
            raise ValueError("D4RL backend source provenance is required")
        if self.implementation_migrated and not self.implementation_selected:
            raise ValueError("a migrated backend must first be selected")
        if self.protocol_frozen and not self.formal_task_matrix_eligible:
            raise ValueError(
                "a frozen backend protocol must be formal-matrix eligible"
            )
        if self.mechanism_runner_reusable:
            raise ValueError("Hopper mechanism runner is not this backend")
        if set(self.shared_contracts) & set(self.distinct_contracts):
            raise ValueError("D4RL shared and distinct contracts overlap")


CANONICAL_EXPRANK_BACKEND = D4RLPerformanceBackendSpec(
    backend_id="canonical_sna2c_iqlv_exprank",
    algorithm_family="SNA2C_IQLV_ExpRank",
    source_paths=(
        "src/drpo/e7_canonical_vendor/d4rl/agents.py",
        "src/drpo/e7_canonical_vendor/d4rl/train_sna2c_variant.py",
        "src/drpo/e7_canonical_vendor/d4rl/d4rl_common/train_loop.py",
        "src/drpo/e7_canonical_vendor/d4rl/d4rl_common/normalize.py",
    ),
    implementation_selected=True,
    implementation_migrated=True,
    protocol_status="selected_backend_code_migrated_protocol_unfrozen",
    protocol_frozen=False,
    formal_task_matrix_eligible=False,
    mechanism_runner_reusable=False,
    shared_contracts=(
        "d4rl_v2_dataset_identity",
        "locomotion_task_catalog",
        "gymnasium_mujoco_rollout_boundary",
        "d4rl_reference_score_normalization",
        "event_taxonomy",
    ),
    distinct_contracts=(
        "actor_likelihood_contract",
        "critic_update_lifecycle",
        "advantage_lifecycle",
        "optimizer_schedule",
        "method_matrix",
        "terminal_audit_protocol",
    ),
)
LEGACY_CANONICAL_BACKEND_CANDIDATE = CANONICAL_EXPRANK_BACKEND


@dataclass(frozen=True)
class CanonicalExpRankTrainingConfig:
    steps: int
    batch_size: int
    learning_rate: float = 3.0e-4
    gamma: float = 0.99
    alpha: float = 0.11
    tau: float = 0.7
    temperature: float = 1.0
    eval_interval: int = 10_000
    checkpoint_interval: int = 10_000
    checkpoint_last_fraction: float = 0.10

    def __post_init__(self) -> None:
        if min(
            self.steps,
            self.batch_size,
            self.eval_interval,
            self.checkpoint_interval,
        ) <= 0:
            raise ValueError("canonical ExpRank integer controls must be positive")
        if self.learning_rate <= 0.0 or self.alpha < 0.0:
            raise ValueError("canonical ExpRank lr/alpha are invalid")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("canonical ExpRank gamma must be in [0, 1]")
        if not 0.5 <= self.tau < 1.0:
            raise ValueError("canonical ExpRank tau must be in [0.5, 1)")
        if self.temperature < 0.0:
            raise ValueError("canonical ExpRank temperature is invalid")
        if not 0.0 < self.checkpoint_last_fraction <= 1.0:
            raise ValueError("checkpoint_last_fraction must be in (0, 1]")


@dataclass(frozen=True)
class CanonicalD4RLDataset:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    terminals: np.ndarray
    mc_returns: np.ndarray

    def __post_init__(self) -> None:
        size = int(self.observations.shape[0])
        arrays = (
            self.actions,
            self.rewards,
            self.next_observations,
            self.terminals,
            self.mc_returns,
        )
        if self.observations.ndim != 2 or self.actions.ndim != 2:
            raise ValueError("D4RL observations and actions must be rank-2")
        if self.next_observations.shape != self.observations.shape:
            raise ValueError("next observations must match observations")
        if size <= 0 or any(int(array.shape[0]) != size for array in arrays):
            raise ValueError("D4RL arrays must share a non-empty first axis")

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
    if negative_advantages.ndim != 1:
        raise ValueError("negative advantages must be rank-1")
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
    return float(alpha) * torch.exp(
        torch.clamp(-float(temperature) * score, min=-20.0)
    )


class SNA2CIQLVExpRankAgent:
    """Migration of the canonical ``SNA2C_IQLV_ExpRankAgent``."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        *,
        learning_rate: float = 3.0e-4,
        gamma: float = 0.99,
        alpha: float = 0.11,
        tau: float = 0.7,
        temperature: float = 1.0,
        device: torch.device | str = "cpu",
    ) -> None:
        if not 0.5 <= tau < 1.0:
            raise ValueError("expectile tau must be in [0.5, 1)")
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.tau = float(tau)
        self.temperature = float(temperature)
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
    ) -> tuple[np.ndarray, float]:
        tensor = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        mean, _ = self.actor(tensor)
        return mean.squeeze(0).cpu().numpy(), 0.0

    def loss_components(
        self,
        observations: np.ndarray | torch.Tensor,
        actions: np.ndarray | torch.Tensor,
        rewards: np.ndarray | torch.Tensor,
        next_observations: np.ndarray | torch.Tensor,
        dones: np.ndarray | torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        states = _float_tensor(observations, self.device)
        action_tensor = _float_tensor(actions, self.device)
        reward_tensor = _float_tensor(rewards, self.device)
        next_states = _float_tensor(next_observations, self.device)
        done_tensor = torch.as_tensor(
            dones,
            dtype=torch.bool,
            device=self.device,
        )
        with torch.no_grad():
            next_value = self.critic(next_states).squeeze(-1)
            target = reward_tensor + self.gamma * next_value * (
                ~done_tensor
            ).float()
        value = self.critic(states).squeeze(-1)
        advantage = target - value.detach()
        transformed = advantage.clone()
        negative_mask = advantage < 0
        if negative_mask.any():
            with torch.no_grad():
                weights = canonical_exprank_negative_weights(
                    advantage[negative_mask],
                    alpha=self.alpha,
                    temperature=self.temperature,
                )
            transformed[negative_mask] = advantage[negative_mask] * weights
        mean, log_std = self.actor(states)
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
        return {
            "target": target,
            "value": value,
            "advantage": advantage,
            "transformed_advantage": transformed,
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
        }

    def update(
        self,
        observations: np.ndarray | torch.Tensor,
        actions: np.ndarray | torch.Tensor,
        rewards: np.ndarray | torch.Tensor,
        next_observations: np.ndarray | torch.Tensor,
        dones: np.ndarray | torch.Tensor,
        episode_returns: np.ndarray | torch.Tensor | None = None,
    ) -> float:
        del episode_returns
        components = self.loss_components(
            observations,
            actions,
            rewards,
            next_observations,
            dones,
        )
        actor_loss = components["actor_loss"]
        critic_loss = components["critic_loss"]
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


def compute_canonical_mc_returns(
    rewards: np.ndarray,
    terminals: np.ndarray,
    timeouts: np.ndarray,
    gamma: float = 0.99,
) -> np.ndarray:
    output = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        if terminals[index] or timeouts[index]:
            running = 0.0
        running = float(rewards[index]) + float(gamma) * running
        output[index] = running
    return output


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
        mc_returns=compute_canonical_mc_returns(
            rewards,
            data.terminals,
            data.timeouts,
            gamma=gamma,
        ),
    )


def train_canonical_exprank(
    *,
    dataset: CanonicalD4RLDataset,
    seed: int,
    config: CanonicalExpRankTrainingConfig,
    device: torch.device | str = "cpu",
    evaluator: Evaluator | None = None,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    np.random.seed(int(seed))
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
    )
    tensors = {
        "s": torch.from_numpy(dataset.observations).to(resolved_device),
        "a": torch.from_numpy(dataset.actions).to(resolved_device),
        "r": torch.from_numpy(dataset.rewards).to(resolved_device),
        "ns": torch.from_numpy(dataset.next_observations).to(resolved_device),
        "d": torch.from_numpy(dataset.terminals).to(resolved_device),
        "ret": torch.from_numpy(dataset.mc_returns).to(resolved_device),
    }
    generator = torch.Generator(device=resolved_device)
    generator.manual_seed(int(seed))
    output = (
        None
        if output_root is None
        else Path(output_root).expanduser().resolve()
    )
    if output is not None:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"D4RL output must be new or empty: {output}")
        output.mkdir(parents=True, exist_ok=True)
    checkpoint_start = int(
        config.steps * (1.0 - config.checkpoint_last_fraction)
    )
    losses: list[dict[str, float | int]] = []
    evaluations: list[dict[str, float | int]] = []
    checkpoints: list[str] = []
    for step in range(1, config.steps + 1):
        indices = torch.randint(
            0,
            dataset.size,
            (config.batch_size,),
            generator=generator,
            device=resolved_device,
        )
        loss = agent.update(
            tensors["s"].index_select(0, indices),
            tensors["a"].index_select(0, indices),
            tensors["r"].index_select(0, indices),
            tensors["ns"].index_select(0, indices),
            tensors["d"].index_select(0, indices),
            tensors["ret"].index_select(0, indices),
        )
        if step == 1 or step == config.steps:
            losses.append({"step": step, "loss": loss})
        if evaluator is not None and step % config.eval_interval == 0:
            evaluations.append(
                {
                    "step": step,
                    **{
                        str(name): float(value)
                        for name, value in evaluator(agent, step).items()
                    },
                }
            )
        if (
            output is not None
            and step >= checkpoint_start
            and step % config.checkpoint_interval == 0
        ):
            checkpoint_dir = output / "ckpts"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = checkpoint_dir / f"step_{step:07d}.pt"
            torch.save(
                {
                    "actor": agent.actor.state_dict(),
                    "sdim": dataset.observation_dim,
                    "adim": dataset.action_dim,
                    "network_preset": "default",
                    "preset_actor_h": -1,
                    "preset_critic_h": -1,
                    "preset_actor_depth": -1,
                    "preset_critic_depth": -1,
                    "step": step,
                    "variant": "iqlv_exp_rank",
                    "alpha": config.alpha,
                    "p": 0.5,
                    "shape": "linear",
                    "tau": config.tau,
                    "seed": int(seed),
                },
                checkpoint,
            )
            checkpoints.append(str(checkpoint))
    record = {
        "backend": asdict(CANONICAL_EXPRANK_BACKEND),
        "seed": int(seed),
        "config": asdict(config),
        "transition_count": dataset.size,
        "loss_records": losses,
        "evaluations": evaluations,
        "checkpoints": checkpoints,
        "formal_result_claim": False,
        "method_ranking_claim_allowed": False,
    }
    if output is not None:
        atomic_json(output / "TRAINING_RESULT.json", record)
        atomic_json(
            output / "COMPLETED.json",
            {
                "status": "completed_non_formal",
                "seed": int(seed),
                "steps": config.steps,
                "formal_result_claim": False,
                "method_ranking_claim_allowed": False,
            },
        )
    return {**record, "agent": agent}


@dataclass(frozen=True)
class D4RL9ExecutionPlan:
    tasks: tuple[D4RLTaskSpec, ...]
    dataset_paths: dict[str, Path]
    seeds: tuple[int, ...]
    backend: D4RLPerformanceBackendSpec
    execution_kind: str
    dataset_identity_complete: bool
    performance_protocol_frozen: bool
    backend_protocol_complete: bool
    formal_evidence_eligible: bool
    method_ranking_claim_allowed: bool
    blocked_reasons: tuple[str, ...]

    def as_manifest(self) -> dict[str, Any]:
        return {
            "experiment_id": D4RL9_EXPERIMENT_ID,
            "runner_version": D4RL9_RUNNER_VERSION,
            "execution_kind": self.execution_kind,
            "tasks": [asdict(task) for task in self.tasks],
            "dataset_paths": {
                task_id: str(path)
                for task_id, path in self.dataset_paths.items()
            },
            "seeds": list(self.seeds),
            "backend": asdict(self.backend),
            "dataset_identity_complete": self.dataset_identity_complete,
            "performance_protocol_frozen": self.performance_protocol_frozen,
            "backend_protocol_complete": self.backend_protocol_complete,
            "formal_evidence_eligible": self.formal_evidence_eligible,
            "method_ranking_claim_allowed": self.method_ranking_claim_allowed,
            "blocked_reasons": list(self.blocked_reasons),
            "single_migrated_trainer_across_d4rl9_tasks": True,
            "shared_training_engine_with_hopper_mechanism": False,
            "separate_per_task_trainers_allowed": False,
        }


def resolve_d4rl9_execution(
    *,
    dataset_paths: Mapping[str, str | Path],
    seeds: Sequence[int],
    tasks: Sequence[D4RLTaskSpec] = D4RL9_TASKS,
    backend: D4RLPerformanceBackendSpec = CANONICAL_EXPRANK_BACKEND,
    performance_protocol_frozen: bool = False,
    smoke: bool = False,
) -> D4RL9ExecutionPlan:
    resolved_tasks = validate_d4rl9_matrix(tasks)
    resolved_seeds = tuple(int(seed) for seed in seeds)
    if not resolved_seeds:
        raise ValueError("at least one D4RL-9 seed is required")
    if len(set(resolved_seeds)) != len(resolved_seeds):
        raise ValueError("D4RL-9 seed list contains duplicates")
    expected_ids = {task.task_id for task in resolved_tasks}
    actual_ids = set(dataset_paths)
    missing = sorted(expected_ids - actual_ids)
    extra = sorted(actual_ids - expected_ids)
    if missing or extra:
        raise ValueError(
            "D4RL-9 dataset mapping mismatch; "
            f"missing={missing}, extra={extra}"
        )
    resolved_paths = {
        task.task_id: Path(dataset_paths[task.task_id]).resolve()
        for task in resolved_tasks
    }
    unresolved = tuple(
        task.task_id
        for task in resolved_tasks
        if not task.dataset_identity_verified
    )
    blocked: list[str] = []
    if unresolved:
        blocked.append(
            "unresolved_dataset_sha256:" + ",".join(unresolved)
        )
    if not backend.implementation_selected:
        blocked.append("d4rl9_performance_backend_not_selected")
    if not backend.implementation_migrated:
        blocked.append("d4rl9_performance_backend_not_migrated")
    if not backend.protocol_frozen:
        blocked.append(
            "d4rl9_performance_backend_protocol_not_frozen"
        )
    if not backend.formal_task_matrix_eligible:
        blocked.append("d4rl9_backend_not_formal_matrix_eligible")
    if not performance_protocol_frozen:
        blocked.append("d4rl9_performance_protocol_not_frozen")
    if len(resolved_seeds) != 10:
        blocked.append("manuscript_ten_run_coordinate_not_complete")
    if smoke:
        blocked.append("smoke_is_not_scientific_evidence")
    dataset_complete = not unresolved
    backend_complete = bool(
        backend.implementation_selected
        and backend.implementation_migrated
        and backend.protocol_frozen
        and backend.formal_task_matrix_eligible
    )
    formal_eligible = bool(
        not smoke
        and dataset_complete
        and backend_complete
        and performance_protocol_frozen
        and len(resolved_seeds) == 10
    )
    return D4RL9ExecutionPlan(
        tasks=resolved_tasks,
        dataset_paths=resolved_paths,
        seeds=resolved_seeds,
        backend=backend,
        execution_kind=(
            "formal_candidate"
            if formal_eligible
            else "blocked_or_non_evidence"
        ),
        dataset_identity_complete=dataset_complete,
        performance_protocol_frozen=performance_protocol_frozen,
        backend_protocol_complete=backend_complete,
        formal_evidence_eligible=formal_eligible,
        method_ranking_claim_allowed=False,
        blocked_reasons=tuple(blocked),
    )


def validate_d4rl9_datasets(
    plan: D4RL9ExecutionPlan,
    *,
    require_formal_identity: bool,
) -> dict[str, dict[str, object]]:
    return {
        task.task_id: validate_dataset_path(
            plan.dataset_paths[task.task_id],
            task,
            require_verified_sha=require_formal_identity,
        )
        for task in plan.tasks
    }


def dispatch_d4rl9(
    *,
    plan: D4RL9ExecutionPlan,
    output_root: str | Path,
    task_runner: TaskRunner,
    allow_non_evidence: bool = False,
) -> dict[str, Any]:
    if plan.blocked_reasons and not allow_non_evidence:
        raise RuntimeError(
            "D4RL-9 dispatch is blocked: "
            + "; ".join(plan.blocked_reasons)
        )
    output = Path(output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"D4RL-9 output must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    results = {
        task.task_id: task_runner(
            task=task,
            backend=plan.backend,
            dataset_path=plan.dataset_paths[task.task_id],
            output_root=output / task.task_id,
            seeds=plan.seeds,
            formal_evidence_eligible=plan.formal_evidence_eligible,
            method_ranking_claim_allowed=False,
        )
        for task in plan.tasks
    }
    return {
        "plan": plan.as_manifest(),
        "tasks": results,
        "formal_result_claim": False,
        "method_ranking_claim_allowed": False,
        "single_migrated_trainer_across_d4rl9_tasks": True,
        "shared_training_engine_with_hopper_mechanism": False,
    }
