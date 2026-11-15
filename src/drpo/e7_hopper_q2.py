#!/usr/bin/env python3
"""EXT-H-E7-Q2 Hopper Gaussian log-scale external-validation runner.

The scientific runner trains an episode-split value critic, freezes critic-derived
advantages, takes a tanh-squashed Gaussian actor through a Positive-only terminal
audit, constructs advantage-matched near/far negative probes, decomposes Gaussian
mean and log-scale output scores, and branches the preregistered signed controls.
The explicit ``rebuttal-pq-is`` profile reuses components with cold-start actors,
frozen/refreshed advantages and raw IS; the original Q2 defaults are unchanged.

This file never creates ZIP/TAR artifacts. Formal supervision, recovery packaging,
checksums, and uploadable result archives are owned by the canonical hardened
execution channel in ``scripts/artifact_protocol_hardened.py``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import traceback
from importlib import metadata as importlib_metadata
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

try:
    import yaml
except ImportError as exc:  # pragma: no cover - dependency declared by project
    raise SystemExit("PyYAML is required. Install the project with pip install -e .") from exc

EXPERIMENT_ID = "EXT-H-E7-Q2"
RUNNER_VERSION = "4.3.1-network-profile-config"
EPS = 1e-6
METHODS = (
    "positive_only",
    "signed",
    "near_zero",
    "far_zero",
    "far_cap",
    "dynamic_budget_matched_global",
)
_EVENT_LOG_PATH: Path | None = None


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def emit_event(payload: dict[str, Any]) -> None:
    line = json.dumps(payload, sort_keys=True, default=_json_default)
    print(line, flush=True)
    if _EVENT_LOG_PATH is not None:
        _EVENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _EVENT_LOG_PATH.open("a") as handle:
            handle.write(line + "\n")


def atomic_write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n")
    os.replace(temp, path)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_csv(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def git_output(repo_root: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=repo_root, text=True, capture_output=True, check=False
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def collect_git_state(repo_root: Path) -> dict[str, Any]:
    state: dict[str, Any] = {"available": False}
    try:
        state.update(
            {
                "available": True,
                "head": git_output(repo_root, "rev-parse", "HEAD"),
                "branch": git_output(repo_root, "rev-parse", "--abbrev-ref", "HEAD"),
                "status_porcelain": git_output(repo_root, "status", "--porcelain=v1"),
            }
        )
    except Exception as exc:  # pragma: no cover - exercised only outside git
        state["error"] = str(exc)
    return state


def environment_manifest() -> dict[str, Any]:
    gpu: dict[str, Any] = {"available": bool(torch.cuda.is_available())}
    if torch.cuda.is_available():
        gpu.update(
            {
                "count": torch.cuda.device_count(),
                "name": torch.cuda.get_device_name(0),
                "capability": torch.cuda.get_device_capability(0),
                "memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            }
        )
    return {
        "created_utc": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": gpu,
    }


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModeConfig:
    max_transitions: int | None
    seeds: tuple[int, ...]
    canonical_critic_seed: int
    critic_max_steps: int
    critic_min_steps: int
    critic_eval_interval: int
    positive_max_steps: int
    positive_min_steps: int
    actor_eval_interval: int
    branch_max_steps: int
    branch_min_steps: int
    matched_pairs: int
    audit_sample_size: int
    rollout_episodes: int
    final_rollout_episodes: int
    rollout_eval_interval: int


@dataclass(frozen=True)
class E7Config:
    experiment_id: str
    dataset_basename: str
    dataset_sha256: str
    env_backend: str
    rollout_dataset_id: str
    env_id: str
    normalized_score_percent: bool
    normalized_score_reference_min: float
    normalized_score_reference_max: float
    process_isolated_preflight: bool
    rollout_preflight_timeout_seconds: int
    formal_rollout_required: bool
    pilot_rollout_required: bool
    rollout_preflight_max_steps: int
    gamma: float
    train_fraction: float
    validation_fraction: float
    hidden_sizes: tuple[int, ...]
    activation: str
    init_scheme: str
    init_gain: float
    critic_lr: float
    actor_lr: float
    critic_batch_size: int
    actor_batch_size: int
    weight_decay: float
    max_gradient_norm: float
    log_std_min: float
    log_std_max: float
    action_clip_epsilon: float
    advantage_standardize: bool
    near_quantile: float
    far_quantile: float
    advantage_bins: int
    advantage_match_relative_tolerance: float
    gradient_probe_pairs: int
    distance_bins: int
    far_cap_reference_quantile: float
    global_budget_audit_size: int
    qxi_slope_target: float
    qxi_slope_tolerance: float
    analytic_autograd_error_max: float
    full_parameter_ratio_min: float
    log_scale_to_mean_ratio_min: float
    critic_validation_r2_min: float
    critic_validation_pearson_min: float
    critic_max_final_to_best_validation_mse_ratio: float
    critic_advantage_sign_agreement_min: float
    critic_advantage_pearson_min: float
    critic_advantage_spearman_min: float
    critic_negative_set_jaccard_min: float
    audit_windows: int
    critic_relative_slope_tolerance: float
    critic_gradient_tolerance: float
    critic_update_tolerance: float
    actor_relative_slope_tolerance: float
    actor_state_drift_tolerance: float
    actor_gradient_tolerance: float
    actor_update_tolerance: float
    support_boundary_fraction: float
    support_boundary_threshold: float
    task_return_drop_threshold: float
    checkpoint_every_formal_seeds: int
    pilot: ModeConfig
    formal: ModeConfig


def _mode_from_dict(raw: dict[str, Any]) -> ModeConfig:
    return ModeConfig(
        max_transitions=(
            None if raw.get("max_transitions") in (None, 0) else int(raw["max_transitions"])
        ),
        seeds=tuple(int(x) for x in raw["seeds"]),
        canonical_critic_seed=int(raw["canonical_critic_seed"]),
        critic_max_steps=int(raw["critic_max_steps"]),
        critic_min_steps=int(raw["critic_min_steps"]),
        critic_eval_interval=int(raw["critic_eval_interval"]),
        positive_max_steps=int(raw["positive_max_steps"]),
        positive_min_steps=int(raw["positive_min_steps"]),
        actor_eval_interval=int(raw["actor_eval_interval"]),
        branch_max_steps=int(raw["branch_max_steps"]),
        branch_min_steps=int(raw["branch_min_steps"]),
        matched_pairs=int(raw["matched_pairs"]),
        audit_sample_size=int(raw["audit_sample_size"]),
        rollout_episodes=int(raw.get("rollout_episodes", 0)),
        final_rollout_episodes=int(
            raw.get("final_rollout_episodes", raw.get("rollout_episodes", 0))
        ),
        rollout_eval_interval=int(raw.get("rollout_eval_interval", 0)),
    )


def load_config(path: str | Path) -> E7Config:
    raw = yaml.safe_load(Path(path).read_text())
    if raw.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError(
            f"Config experiment_id must be {EXPERIMENT_ID}, got {raw.get('experiment_id')}"
        )
    env = raw["environment_evaluation"]
    gate = raw["independent_validation_gate"]
    return E7Config(
        experiment_id=raw["experiment_id"],
        dataset_basename=str(raw["dataset"]["basename"]),
        dataset_sha256=str(raw["dataset"]["sha256"]),
        env_backend=str(env["backend"]),
        rollout_dataset_id=str(env["dataset_id"]),
        env_id=str(env["env_id"]),
        normalized_score_percent=bool(env["normalized_score_percent"]),
        normalized_score_reference_min=float(env["reference_min_score"]),
        normalized_score_reference_max=float(env["reference_max_score"]),
        process_isolated_preflight=bool(env["process_isolated_preflight"]),
        rollout_preflight_timeout_seconds=int(env["preflight_timeout_seconds"]),
        formal_rollout_required=bool(env["formal_required"]),
        pilot_rollout_required=bool(env["pilot_required"]),
        rollout_preflight_max_steps=int(env["preflight_max_steps"]),
        gamma=float(raw["critic"]["gamma"]),
        train_fraction=float(raw["critic"]["episode_split"]["train"]),
        validation_fraction=float(raw["critic"]["episode_split"]["validation"]),
        hidden_sizes=tuple(int(x) for x in raw["model"]["hidden_sizes"]),
        activation=str(raw["model"].get("activation", "tanh")),
        init_scheme=str(raw["model"].get("init", "default")),
        init_gain=float(raw["model"].get("init_gain", 1.0)),
        critic_lr=float(raw["critic"]["learning_rate"]),
        actor_lr=float(raw["actor"]["learning_rate"]),
        critic_batch_size=int(raw["critic"]["batch_size"]),
        actor_batch_size=int(raw["actor"]["batch_size"]),
        weight_decay=float(raw["optimization"]["weight_decay"]),
        max_gradient_norm=float(raw["optimization"]["max_gradient_norm"]),
        log_std_min=float(raw["actor"]["log_std_min"]),
        log_std_max=float(raw["actor"]["log_std_max"]),
        action_clip_epsilon=float(raw["actor"]["action_clip_epsilon"]),
        advantage_standardize=bool(raw["advantage"]["standardize_once"]),
        near_quantile=float(raw["matching"]["near_quantile"]),
        far_quantile=float(raw["matching"]["far_quantile"]),
        advantage_bins=int(raw["matching"]["advantage_bins"]),
        advantage_match_relative_tolerance=float(
            raw["matching"]["relative_advantage_tolerance"]
        ),
        gradient_probe_pairs=int(raw["matching"]["gradient_probe_pairs"]),
        distance_bins=int(raw["matching"]["distance_bins"]),
        far_cap_reference_quantile=float(
            raw["interventions"]["far_cap_reference_quantile"]
        ),
        global_budget_audit_size=int(raw["interventions"]["global_budget_audit_size"]),
        qxi_slope_target=float(gate["corrected_Q_xi_loglog_slope_target"]),
        qxi_slope_tolerance=float(gate["corrected_Q_xi_loglog_slope_tolerance"]),
        analytic_autograd_error_max=float(gate["analytic_autograd_relative_error_max"]),
        full_parameter_ratio_min=float(gate["full_parameter_far_near_ratio_min"]),
        log_scale_to_mean_ratio_min=float(
            gate["log_scale_to_mean_far_near_ratio_min"]
        ),
        critic_validation_r2_min=float(raw["critic_acceptance"]["validation_r2_min"]),
        critic_validation_pearson_min=float(raw["critic_acceptance"]["validation_pearson_min"]),
        critic_max_final_to_best_validation_mse_ratio=float(
            raw["critic_acceptance"]["max_final_to_best_validation_mse_ratio"]
        ),
        critic_advantage_sign_agreement_min=float(
            raw["critic_acceptance"]["advantage_sign_agreement_min"]
        ),
        critic_advantage_pearson_min=float(
            raw["critic_acceptance"]["advantage_pearson_min"]
        ),
        critic_advantage_spearman_min=float(
            raw["critic_acceptance"]["advantage_spearman_min"]
        ),
        critic_negative_set_jaccard_min=float(
            raw["critic_acceptance"]["negative_set_jaccard_min"]
        ),
        audit_windows=int(raw["terminal_audit"]["windows"]),
        critic_relative_slope_tolerance=float(
            raw["terminal_audit"]["critic_relative_slope_tolerance"]
        ),
        critic_gradient_tolerance=float(
            raw["terminal_audit"].get("critic_gradient_tolerance_diagnostic", 0.01)
        ),
        critic_update_tolerance=float(
            raw["terminal_audit"].get(
                "critic_relative_update_tolerance",
                raw["terminal_audit"].get("critic_update_tolerance", 0.0001),
            )
        ),
        actor_relative_slope_tolerance=float(
            raw["terminal_audit"]["actor_relative_slope_tolerance"]
        ),
        actor_state_drift_tolerance=float(
            raw["terminal_audit"].get(
                "actor_state_drift_tolerance",
                raw["terminal_audit"]["actor_relative_slope_tolerance"],
            )
        ),
        actor_gradient_tolerance=float(
            raw["terminal_audit"].get("actor_gradient_tolerance_diagnostic", 0.01)
        ),
        actor_update_tolerance=float(
            raw["terminal_audit"].get(
                "actor_relative_update_tolerance",
                raw["terminal_audit"].get("actor_update_tolerance", 0.0001),
            )
        ),
        support_boundary_fraction=float(
            raw["terminal_audit"]["support_boundary_fraction"]
        ),
        support_boundary_threshold=float(
            raw["terminal_audit"]["support_boundary_threshold"]
        ),
        task_return_drop_threshold=float(
            raw["terminal_audit"]["task_normalized_return_drop_threshold"]
        ),
        checkpoint_every_formal_seeds=int(
            raw["artifact"]["checkpoint_every_formal_seeds"]
        ),
        pilot=_mode_from_dict(raw["modes"]["pilot"]),
        formal=_mode_from_dict(raw["modes"]["formal"]),
    )


# ---------------------------------------------------------------------------
# Data and episode logic
# ---------------------------------------------------------------------------


@dataclass
class OfflineData:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    terminals: np.ndarray
    timeouts: np.ndarray
    episode_ids: np.ndarray
    behavior_log_prob: np.ndarray | None = None
    transition_ids: np.ndarray | None = None

    @property
    def size(self) -> int:
        return int(self.observations.shape[0])


def load_hopper_hdf5(path: str | Path, max_transitions: int | None) -> OfflineData:
    path = Path(path)
    with h5py.File(path, "r") as handle:
        required = ("observations", "actions", "rewards", "terminals")
        missing = [key for key in required if key not in handle]
        if missing:
            raise ValueError(f"Missing legacy D4RL arrays: {missing}")
        total = int(handle["observations"].shape[0])
        limit = total if max_transitions is None else min(total, int(max_transitions))
        observations = np.asarray(handle["observations"][:limit], dtype=np.float32)
        actions = np.asarray(handle["actions"][:limit], dtype=np.float32)
        behavior_log_prob = (
            np.asarray(handle["infos/action_log_probs"][:limit], dtype=np.float32).reshape(-1)
            if "infos/action_log_probs" in handle else None
        )
        rewards = np.asarray(handle["rewards"][:limit], dtype=np.float32).reshape(-1)
        terminals = np.asarray(handle["terminals"][:limit], dtype=np.bool_).reshape(-1)
        if "timeouts" in handle:
            timeouts = np.asarray(handle["timeouts"][:limit], dtype=np.bool_).reshape(-1)
        else:
            timeouts = np.zeros(limit, dtype=np.bool_)
        if "next_observations" in handle:
            next_obs = np.asarray(handle["next_observations"][:limit], dtype=np.float32)
        else:
            next_obs = np.concatenate([observations[1:], observations[-1:]], axis=0)
    if observations.ndim != 2 or actions.ndim != 2:
        raise ValueError("observations and actions must be rank-2 arrays")
    if len(observations) < 2:
        raise ValueError("dataset must contain at least two transitions")
    episode_ids = build_episode_ids(terminals, timeouts)
    return OfflineData(
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_obs,
        terminals=terminals,
        timeouts=timeouts,
        episode_ids=episode_ids,
        behavior_log_prob=behavior_log_prob,
        transition_ids=np.arange(limit, dtype=np.int64),
    )


def build_episode_ids(terminals: np.ndarray, timeouts: np.ndarray) -> np.ndarray:
    terminals = np.asarray(terminals, dtype=np.bool_).reshape(-1)
    timeouts = np.asarray(timeouts, dtype=np.bool_).reshape(-1)
    if terminals.shape != timeouts.shape:
        raise ValueError("terminals and timeouts must have the same shape")
    out = np.empty(len(terminals), dtype=np.int64)
    episode = 0
    for idx in range(len(terminals)):
        out[idx] = episode
        if terminals[idx] or timeouts[idx]:
            episode += 1
    return out


def discounted_returns(
    rewards: np.ndarray,
    terminals: np.ndarray,
    timeouts: np.ndarray,
    gamma: float,
) -> np.ndarray:
    rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
    terminals = np.asarray(terminals, dtype=np.bool_).reshape(-1)
    timeouts = np.asarray(timeouts, dtype=np.bool_).reshape(-1)
    returns = np.empty_like(rewards, dtype=np.float32)
    running = 0.0
    for idx in range(len(rewards) - 1, -1, -1):
        if idx == len(rewards) - 1 or terminals[idx] or timeouts[idx]:
            running = float(rewards[idx])
        else:
            running = float(rewards[idx]) + gamma * running
        returns[idx] = running
    return returns


def split_episode_indices(
    episode_ids: np.ndarray,
    seed: int,
    train_fraction: float,
    validation_fraction: float,
) -> dict[str, np.ndarray]:
    episodes = np.unique(episode_ids)
    if len(episodes) < 3:
        raise ValueError("At least three episodes are required for train/validation/test split")
    rng = np.random.default_rng(seed)
    shuffled = episodes.copy()
    rng.shuffle(shuffled)
    n_train = max(1, int(round(len(shuffled) * train_fraction)))
    n_val = max(1, int(round(len(shuffled) * validation_fraction)))
    if n_train + n_val >= len(shuffled):
        n_train = max(1, len(shuffled) - 2)
        n_val = 1
    groups = {
        "train": shuffled[:n_train],
        "validation": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }
    return {
        name: np.flatnonzero(np.isin(episode_ids, eps)).astype(np.int64)
        for name, eps in groups.items()
    }


@dataclass(frozen=True)
class Normalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, array: np.ndarray) -> "Normalizer":
        mean = np.mean(array, axis=0, dtype=np.float64).astype(np.float32)
        std = np.std(array, axis=0, dtype=np.float64).astype(np.float32)
        std = np.maximum(std, 1e-6)
        return cls(mean=mean, std=std)

    def transform(self, array: np.ndarray) -> np.ndarray:
        return ((array - self.mean) / self.std).astype(np.float32)


# ---------------------------------------------------------------------------
# Networks and policy geometry
# ---------------------------------------------------------------------------


def activation_layer(name: str) -> nn.Module:
    normalized = name.strip().lower()
    if normalized == "tanh":
        return nn.Tanh()
    if normalized == "relu":
        return nn.ReLU()
    raise ValueError(f"unsupported E7 network activation: {name}")


def apply_orthogonal_init(module: nn.Module, gain: float) -> None:
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, gain=float(gain))
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
        layers.extend([nn.Linear(width, hidden), activation_layer(activation)])
        width = hidden
    layers.append(nn.Linear(width, output_dim))
    net = nn.Sequential(*layers)
    init_name = init_scheme.strip().lower()
    if init_name == "orthogonal":
        apply_orthogonal_init(net, init_gain)
    elif init_name != "default":
        raise ValueError(f"unsupported E7 init scheme: {init_scheme}")
    return net


class ValueNetwork(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: Sequence[int],
        activation: str = "tanh",
        init_scheme: str = "default",
        init_gain: float = 1.0,
    ):
        super().__init__()
        self.net = make_mlp(obs_dim, 1, hidden_sizes, activation, init_scheme, init_gain)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


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
    ):
        super().__init__()
        self.mean_net = make_mlp(
            obs_dim, action_dim, hidden_sizes, activation, init_scheme, init_gain
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.action_clip_epsilon = float(action_clip_epsilon)

    def latent_parameters(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.mean_net(obs)
        log_std = self.log_std.clamp(self.log_std_min, self.log_std_max)
        return mean, log_std.expand_as(mean)

    def action_mean(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self.latent_parameters(obs)
        return torch.tanh(mean)

    def inverse_action(self, actions: torch.Tensor) -> torch.Tensor:
        eps = self.action_clip_epsilon
        clipped = actions.clamp(-1.0 + eps, 1.0 - eps)
        return torch.atanh(clipped)

    def log_prob(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        latent = self.inverse_action(actions)
        mean, log_std = self.latent_parameters(obs)
        inv_var = torch.exp(-2.0 * log_std)
        gaussian = -0.5 * ((latent - mean).square() * inv_var + 2.0 * log_std + math.log(2.0 * math.pi))
        jacobian = torch.log(torch.clamp(1.0 - actions.clamp(-1.0 + EPS, 1.0 - EPS).square(), min=EPS))
        return (gaussian - jacobian).sum(dim=-1)

    def score_components(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        latent = self.inverse_action(actions)
        mean, log_std = self.latent_parameters(obs)
        std = torch.exp(log_std)
        z = (latent - mean) / std
        mean_score = (latent - mean) / std.square()
        log_scale_score = z.square() - 1.0
        mean_norm = torch.linalg.vector_norm(mean_score, dim=-1)
        log_scale_norm = torch.linalg.vector_norm(log_scale_score, dim=-1)
        joint_norm = torch.sqrt(mean_norm.square() + log_scale_norm.square())
        radius = torch.linalg.vector_norm(z, dim=-1)
        q_xi = z.square().sum(dim=-1)
        action_mean = torch.tanh(mean)
        return {
            "latent": latent,
            "mean": mean,
            "log_std": log_std,
            "z": z,
            "radius": radius,
            "mean_score": mean_score,
            "mean_score_norm": mean_norm,
            "log_scale_score": log_scale_score,
            "raw_log_scale_score_norm": log_scale_norm,
            "corrected_q_xi": q_xi,
            "joint_output_score_norm": joint_norm,
            "log_scale_to_mean_ratio": log_scale_norm / mean_norm.clamp_min(EPS),
            "raw_action_distance": torch.linalg.vector_norm(actions - action_mean, dim=-1),
            "pre_squash_distance": torch.linalg.vector_norm(latent - mean, dim=-1),
        }

    def standardized_distance(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.score_components(obs, actions)["radius"]

    def output_score_norm(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.score_components(obs, actions)["joint_output_score_norm"]


# ---------------------------------------------------------------------------
# Metrics and terminal audit
# ---------------------------------------------------------------------------


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if denom <= EPS:
        return float("nan")
    return 1.0 - float(np.sum((y_true - y_pred) ** 2)) / denom


def pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.std(y_true) <= EPS or np.std(y_pred) <= EPS:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def relative_slope(rows: Sequence[dict[str, Any]], key: str, windows: int) -> float:
    if len(rows) < windows:
        return float("inf")
    tail = rows[-windows:]
    x = np.asarray([float(row["step"]) for row in tail], dtype=np.float64)
    y = np.asarray([float(row[key]) for row in tail], dtype=np.float64)
    if not np.all(np.isfinite(y)) or x[-1] == x[0]:
        return float("inf")
    slope = float(np.polyfit(x, y, 1)[0])
    scale = max(float(np.mean(np.abs(y))), 1.0)
    return abs(slope) / scale


def normalized_window_drift(
    rows: Sequence[dict[str, Any]], key: str, windows: int
) -> float:
    """Fitted end-to-end state change over an audit window, scale-normalized.

    The x-axis is the recorded training step, while the normalization uses the
    median absolute state magnitude. This is stable under stochastic minibatch
    noise and does not become ill-conditioned when a likelihood crosses zero.
    """
    if len(rows) < windows:
        return float("inf")
    tail = rows[-windows:]
    x = np.asarray([float(row["step"]) for row in tail], dtype=np.float64)
    y = np.asarray([float(row[key]) for row in tail], dtype=np.float64)
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or x[-1] <= x[0]:
        return float("inf")
    slope = float(np.polyfit(x - x[0], y, 1)[0])
    span = float(x[-1] - x[0])
    scale = max(float(np.median(np.abs(y))), 1e-3)
    return abs(slope) * span / scale


def classify_actor_terminal(
    rows: Sequence[dict[str, Any]],
    config: E7Config,
    candidate_step: int | None,
    extension_complete: bool,
    fixed_budget_completed: bool = False,
) -> dict[str, Any]:
    last = rows[-1]
    relative_update_norm = float(
        last.get("relative_update_norm", last.get("update_norm", float("inf")))
    )
    nonfinite = any(
        not math.isfinite(float(last[key]))
        for key in (
            "loss",
            "positive_nll",
            "gradient_norm",
            "update_norm",
            "sigma_mean",
        )
    ) or not math.isfinite(relative_update_norm)
    support_event = (
        float(last["mean_boundary_fraction"]) >= config.support_boundary_fraction
        or float(last["log_std_min_fraction"]) > 0.0
        or float(last["log_std_max_fraction"]) > 0.0
    )
    slopes = {
        key: relative_slope(rows, key, config.audit_windows)
        for key in ("positive_nll", "mean_abs", "sigma_mean", "phantom_distance_mean")
    }
    state_drifts = {
        key: normalized_window_drift(rows, key, config.audit_windows)
        for key in ("mean_abs", "sigma_mean", "phantom_distance_mean")
    }
    stable = (
        candidate_step is not None
        and extension_complete
        and all(
            value <= config.actor_state_drift_tolerance
            for value in state_drifts.values()
        )
        and relative_update_norm <= config.actor_update_tolerance
        and not nonfinite
    )
    rollout_values = [float(row.get("normalized_return", float("nan"))) for row in rows]
    finite_rollouts = [value for value in rollout_values if math.isfinite(value)]
    rollout_statuses = {
        str(row.get("rollout_status", "not_evaluated")) for row in rows
    }
    initial_return = finite_rollouts[0] if finite_rollouts else float("nan")
    final_return = finite_rollouts[-1] if finite_rollouts else float("nan")
    if finite_rollouts:
        task_status = "available"
        task_collapse: bool | None = bool(
            initial_return - final_return >= config.task_return_drop_threshold
        )
    elif "unavailable" in rollout_statuses:
        task_status = "unavailable"
        task_collapse = None
    elif rollout_statuses == {"disabled"}:
        task_status = "disabled"
        task_collapse = None
    else:
        task_status = "not_evaluated"
        task_collapse = None

    if nonfinite:
        state = "nan_inf_numerical_collapse"
    elif stable and support_event:
        state = "finite_terminal_with_support_boundary_event"
    elif stable:
        state = "finite_terminal"
    elif support_event:
        state = "support_or_variance_boundary_event_without_terminal_convergence"
    elif len(rows) >= config.audit_windows and any(
        value > config.actor_state_drift_tolerance for value in state_drifts.values()
    ):
        state = "persistent_or_slow_drift"
    elif fixed_budget_completed:
        state = "fixed_horizon_inconclusive"
    else:
        state = "training_incomplete_without_terminal_classification"
    explicit_terminal_classification = (
        state != "training_incomplete_without_terminal_classification"
    )
    return {
        "state": state,
        "candidate_step": candidate_step,
        "extension_complete": extension_complete,
        "fixed_budget_completed": fixed_budget_completed,
        "terminal_audit_controls_stopping": False,
        "slopes": slopes,
        "state_drifts": state_drifts,
        "state_drift_tolerance": config.actor_state_drift_tolerance,
        "relative_update_norm": relative_update_norm,
        "support_boundary_event": support_event,
        "numerical_nonfinite": nonfinite,
        "task_performance_status": task_status,
        "task_performance_collapse": task_collapse,
        "normalized_return_available": task_status == "available",
        "initial_normalized_return": initial_return,
        "final_normalized_return": final_return,
        "task_return_drop_threshold": config.task_return_drop_threshold,
        "explicit_terminal_classification": explicit_terminal_classification,
        "reporting_separation": [
            "task_performance_status_and_collapse",
            "support_or_variance_boundary_event",
            "nan_inf_numerical_collapse",
        ],
    }


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def normalize_d4rl_reference_score(
    raw_return: float,
    reference_min_score: float,
    reference_max_score: float,
    *,
    percent: bool,
) -> float:
    """Normalize a raw return with frozen D4RL-v2 reference scores.

    The rollout simulator is intentionally decoupled from the offline dataset
    registration stack.  This function reproduces the public D4RL normalization
    convention without importing ``d4rl`` or ``mujoco_py``.
    """
    raw = float(raw_return)
    lo = float(reference_min_score)
    hi = float(reference_max_score)
    if not all(math.isfinite(value) for value in (raw, lo, hi)):
        raise ValueError("Raw return and D4RL reference scores must be finite")
    if hi <= lo:
        raise ValueError("D4RL reference_max_score must exceed reference_min_score")
    normalized = (raw - lo) / (hi - lo)
    return normalized * 100.0 if percent else normalized


def _open_gymnasium_mujoco_env(env_id: str) -> tuple[Any, dict[str, Any]]:
    """Open the explicit Gymnasium/MuJoCo compatibility environment.

    There is deliberately no legacy D4RL fallback.  Importing ``d4rl`` can load
    ``mujoco_py`` and terminate the process with SIGSEGV before Python exception
    handling runs.  Dataset identity and score normalization are handled
    separately from simulator construction.
    """
    legacy_modules_before = {
        name: name in sys.modules for name in ("d4rl", "mujoco_py")
    }
    gymnasium = importlib.import_module("gymnasium")
    env = gymnasium.make(env_id)
    legacy_modules_after = {
        name: name in sys.modules for name in ("d4rl", "mujoco_py")
    }
    if any(legacy_modules_after.values()):
        env.close()
        raise RuntimeError(
            "Gymnasium rollout process contains a legacy D4RL/mujoco_py module"
        )
    metadata = {
        "backend": "gymnasium_mujoco",
        "gym_backend": "gymnasium",
        "gym_module": getattr(gymnasium, "__name__", "gymnasium"),
        "evaluation_env_id": env_id,
        "legacy_d4rl_fallback": "forbidden",
        "legacy_modules_before": legacy_modules_before,
        "legacy_modules_after": legacy_modules_after,
    }
    return env, metadata


def _reset_env(env: Any, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        result = env.reset(seed=int(seed))
        reset_mode = "reset_seed_kwarg"
    except TypeError:
        if hasattr(env, "seed"):
            env.seed(int(seed))
        result = env.reset()
        reset_mode = "legacy_env_seed"
    if isinstance(result, tuple):
        observation = result[0]
        info = result[1] if len(result) > 1 and isinstance(result[1], dict) else {}
    else:
        observation = result
        info = {}
    return np.asarray(observation, dtype=np.float32), {
        "reset_mode": reset_mode,
        "info_keys": sorted(str(key) for key in info),
    }


def _step_env(env: Any, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
    result = env.step(action)
    if not isinstance(result, tuple):
        raise RuntimeError(f"env.step returned {type(result).__name__}, expected tuple")
    if len(result) == 5:
        observation, reward, terminated, truncated, info = result
        done = bool(terminated or truncated)
        api = "new_five_tuple"
    elif len(result) == 4:
        observation, reward, done, info = result
        done = bool(done)
        api = "legacy_four_tuple"
    else:
        raise RuntimeError(f"env.step returned {len(result)} values, expected 4 or 5")
    return (
        np.asarray(observation, dtype=np.float32),
        float(reward),
        done,
        {
            "step_api": api,
            "info_keys": sorted(str(key) for key in info) if isinstance(info, dict) else [],
        },
    )


def _clip_action_to_space(env: Any, action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32)
    space = getattr(env, "action_space", None)
    low = getattr(space, "low", None)
    high = getattr(space, "high", None)
    if low is not None and high is not None:
        action = np.clip(action, np.asarray(low), np.asarray(high))
    return action.astype(np.float32, copy=False)


def _max_episode_steps(env: Any, fallback: int) -> int:
    spec = getattr(env, "spec", None)
    value = getattr(spec, "max_episode_steps", None)
    if isinstance(value, int) and value > 0:
        return min(value, fallback)
    value = getattr(env, "_max_episode_steps", None)
    if isinstance(value, int) and value > 0:
        return min(value, fallback)
    return fallback


def _rollout_environment_versions() -> dict[str, Any]:
    return {
        "python": sys.version,
        "numpy": np.__version__,
        "packages": {
            name: _package_version(name)
            for name in ("gymnasium", "mujoco", "gym", "d4rl", "mujoco-py")
        },
        "legacy_modules_imported": {
            name: name in sys.modules for name in ("d4rl", "mujoco_py")
        },
    }


def _run_rollout_preflight_worker(
    *,
    backend: str,
    dataset_id: str,
    env_id: str,
    expected_observation_dim: int,
    expected_action_dim: int,
    seed: int,
    max_steps: int,
    normalized_score_percent: bool,
    reference_min_score: float,
    reference_max_score: float,
    output_report: Path,
) -> dict[str, Any]:
    """Run one preflight inside the current process and persist its report."""
    output_report.parent.mkdir(parents=True, exist_ok=True)
    versions = _rollout_environment_versions()
    atomic_write_json(output_report.parent / "environment_versions.json", versions)
    report: dict[str, Any] = {
        "status": "running",
        "started_utc": utc_now(),
        "backend": backend,
        "dataset_id": dataset_id,
        "evaluation_env_id": env_id,
        "expected_observation_dim": expected_observation_dim,
        "expected_action_dim": expected_action_dim,
        "max_steps": max_steps,
        "versions": versions,
        "normalization": {
            "protocol": "d4rl_v2_reference",
            "reference_dataset_id": dataset_id,
            "reference_min_score": float(reference_min_score),
            "reference_max_score": float(reference_max_score),
            "percent": bool(normalized_score_percent),
        },
    }
    env = None
    try:
        if backend != "gymnasium_mujoco":
            raise ValueError(f"Unsupported rollout backend: {backend!r}")
        env, open_metadata = _open_gymnasium_mujoco_env(env_id)
        report["open"] = open_metadata
        observation, reset_metadata = _reset_env(env, seed)
        report["reset"] = {
            **reset_metadata,
            "observation_shape": list(observation.shape),
            "observation_finite": bool(np.all(np.isfinite(observation))),
        }
        if observation.size != expected_observation_dim:
            raise RuntimeError(
                f"Environment observation size {observation.size} does not match "
                f"dataset observation dimension {expected_observation_dim}"
            )
        action_space = getattr(env, "action_space", None)
        if action_space is None or not hasattr(action_space, "sample"):
            raise RuntimeError("Environment does not expose a sampleable action_space")
        if hasattr(action_space, "seed"):
            action_space.seed(int(seed))
        sample_action = _clip_action_to_space(env, action_space.sample())
        if sample_action.size != expected_action_dim:
            raise RuntimeError(
                f"Environment action size {sample_action.size} does not match "
                f"dataset action dimension {expected_action_dim}"
            )
        next_observation, reward, done, step_metadata = _step_env(env, sample_action)
        report["single_step"] = {
            **step_metadata,
            "action_shape": list(sample_action.shape),
            "action_finite": bool(np.all(np.isfinite(sample_action))),
            "next_observation_shape": list(next_observation.shape),
            "next_observation_finite": bool(np.all(np.isfinite(next_observation))),
            "reward": reward,
            "reward_finite": math.isfinite(reward),
            "done": done,
        }
        observation, _ = _reset_env(env, seed + 1)
        total = 0.0
        limit = _max_episode_steps(env, max_steps)
        episode_steps = 0
        done = False
        last_api = None
        while not done and episode_steps < limit:
            action = _clip_action_to_space(env, action_space.sample())
            observation, reward, done, meta = _step_env(env, action)
            total += reward
            episode_steps += 1
            last_api = meta["step_api"]
        normalized = normalize_d4rl_reference_score(
            total,
            reference_min_score,
            reference_max_score,
            percent=normalized_score_percent,
        )
        report["random_episode"] = {
            "steps": episode_steps,
            "step_limit": limit,
            "terminated_or_truncated": done,
            "return": total,
            "normalized_return": normalized,
            "normalized_return_available": math.isfinite(normalized),
            "last_step_api": last_api,
        }
        if episode_steps <= 0:
            raise RuntimeError("Random rollout completed zero steps")
        if not math.isfinite(total):
            raise RuntimeError("Random rollout return is non-finite")
        if not math.isfinite(normalized):
            raise RuntimeError("D4RL-reference normalized score is non-finite")
        report.update(
            {
                "status": "passed",
                "completed_utc": utc_now(),
                "interaction_verified": True,
                "normalized_score_verified": True,
            }
        )
    except Exception as exc:
        report.update(
            {
                "status": "failed",
                "failed_utc": utc_now(),
                "interaction_verified": False,
                "normalized_score_verified": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        atomic_write_json(output_report, report)
    return report


def _signal_name(returncode: int) -> str | None:
    if returncode >= 0:
        return None
    try:
        import signal

        return signal.Signals(-returncode).name
    except (ValueError, OSError):
        return f"SIGNAL_{-returncode}"




def _diagnostic_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)

def preflight_rollout_environment(
    *,
    backend: str,
    dataset_id: str,
    env_id: str,
    expected_observation_dim: int,
    expected_action_dim: int,
    seed: int,
    max_steps: int,
    normalized_score_percent: bool,
    reference_min_score: float,
    reference_max_score: float,
    output_dir: Path,
    required: bool,
    process_isolated: bool,
    timeout_seconds: int,
) -> dict[str, Any]:
    """Run the Gymnasium rollout preflight, isolating native crashes by default."""
    output_dir.mkdir(parents=True, exist_ok=True)
    worker_report = output_dir / "rollout_preflight_worker.json"
    canonical_report = output_dir / "rollout_preflight.json"

    if process_isolated:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "rollout-preflight-worker",
            "--backend",
            backend,
            "--dataset-id",
            dataset_id,
            "--env-id",
            env_id,
            "--expected-observation-dim",
            str(expected_observation_dim),
            "--expected-action-dim",
            str(expected_action_dim),
            "--seed",
            str(seed),
            "--max-steps",
            str(max_steps),
            "--reference-min-score",
            repr(float(reference_min_score)),
            "--reference-max-score",
            repr(float(reference_max_score)),
            "--output-report",
            str(worker_report),
        ]
        if normalized_score_percent:
            command.append("--normalized-score-percent")
        try:
            completed = subprocess.run(
                command,
                text=True,
                capture_output=True,
                timeout=int(timeout_seconds),
                check=False,
            )
            if worker_report.is_file():
                report = json.loads(worker_report.read_text())
            else:
                report = {
                    "status": "failed",
                    "interaction_verified": False,
                    "normalized_score_verified": False,
                    "error_type": (
                        "NativeProcessSignal"
                        if completed.returncode < 0
                        else "WorkerReportMissing"
                    ),
                    "error": (
                        f"rollout worker terminated by {_signal_name(completed.returncode)}"
                        if completed.returncode < 0
                        else "rollout worker exited without writing its report"
                    ),
                }
            report["subprocess_isolation"] = {
                "enabled": True,
                "command": command,
                "returncode": completed.returncode,
                "signal_name": _signal_name(completed.returncode),
                "stdout": _diagnostic_text(completed.stdout),
                "stderr": _diagnostic_text(completed.stderr),
                "timeout_seconds": int(timeout_seconds),
            }
            if completed.returncode != 0:
                report["status"] = "failed"
                report["interaction_verified"] = False
                report["normalized_score_verified"] = False
                if completed.returncode < 0:
                    report["error_type"] = "NativeProcessSignal"
                    report["error"] = (
                        f"rollout worker terminated by {_signal_name(completed.returncode)}"
                    )
        except subprocess.TimeoutExpired as exc:
            report = {
                "status": "failed",
                "interaction_verified": False,
                "normalized_score_verified": False,
                "error_type": "RolloutPreflightTimeout",
                "error": f"rollout worker exceeded {timeout_seconds} seconds",
                "subprocess_isolation": {
                    "enabled": True,
                    "command": command,
                    "timeout_seconds": int(timeout_seconds),
                    "stdout": _diagnostic_text(exc.stdout),
                    "stderr": _diagnostic_text(exc.stderr),
                },
            }
        except Exception as exc:
            report = {
                "status": "failed",
                "interaction_verified": False,
                "normalized_score_verified": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "subprocess_isolation": {"enabled": True, "command": command},
            }
    else:
        report = _run_rollout_preflight_worker(
            backend=backend,
            dataset_id=dataset_id,
            env_id=env_id,
            expected_observation_dim=expected_observation_dim,
            expected_action_dim=expected_action_dim,
            seed=seed,
            max_steps=max_steps,
            normalized_score_percent=normalized_score_percent,
            reference_min_score=reference_min_score,
            reference_max_score=reference_max_score,
            output_report=worker_report,
        )
        report["subprocess_isolation"] = {"enabled": False}

    report["required"] = bool(required)
    report["legacy_d4rl_fallback"] = "forbidden"
    atomic_write_json(canonical_report, report)
    if required and report.get("status") != "passed":
        raise RuntimeError(
            f"Gymnasium rollout preflight failed for {env_id!r}: "
            f"{report.get('error', 'unknown failure')}. See {canonical_report}"
        )
    return report


def evaluate_d4rl_rollouts(
    *,
    policy: SquashedGaussianPolicy,
    obs_norm: Normalizer,
    backend: str,
    dataset_id: str,
    env_id: str,
    episodes: int,
    seed: int,
    device: torch.device,
    normalized_score_percent: bool,
    reference_min_score: float,
    reference_max_score: float,
    required: bool,
    diagnostics_path: Path | None = None,
) -> dict[str, Any]:
    if episodes <= 0:
        return {
            "rollout_status": "disabled",
            "rollout_return_mean": float("nan"),
            "rollout_return_std": float("nan"),
            "normalized_return": float("nan"),
            "normalized_return_available": False,
            "rollout_episodes": 0,
        }
    env = None
    try:
        if backend != "gymnasium_mujoco":
            raise ValueError(f"Unsupported rollout backend: {backend!r}")
        env, open_metadata = _open_gymnasium_mujoco_env(env_id)
        returns: list[float] = []
        episode_steps: list[int] = []
        for episode in range(episodes):
            episode_seed = int(seed + episode)
            observation, _ = _reset_env(env, episode_seed)
            total = 0.0
            done = False
            steps = 0
            limit = _max_episode_steps(env, 10000)
            while not done and steps < limit:
                normalized_obs = obs_norm.transform(observation.reshape(1, -1))
                with torch.no_grad():
                    action = (
                        policy.action_mean(tensor(normalized_obs, device))[0]
                        .detach()
                        .cpu()
                        .numpy()
                    )
                action = _clip_action_to_space(env, action)
                observation, reward, done, _ = _step_env(env, action)
                total += reward
                steps += 1
            if steps >= limit and not done:
                raise RuntimeError(
                    f"Episode reached safety limit {limit} without termination"
                )
            returns.append(total)
            episode_steps.append(steps)
        mean_return = float(np.mean(returns))
        normalized = normalize_d4rl_reference_score(
            mean_return,
            reference_min_score,
            reference_max_score,
            percent=normalized_score_percent,
        )
        result: dict[str, Any] = {
            "rollout_status": "available",
            "rollout_return_mean": mean_return,
            "rollout_return_std": float(np.std(returns)),
            "normalized_return": normalized,
            "normalized_return_available": math.isfinite(normalized),
            "rollout_episodes": int(episodes),
            "rollout_episode_steps_mean": float(np.mean(episode_steps)),
            "rollout_open_metadata": open_metadata,
            "rollout_backend": backend,
            "evaluation_env_id": env_id,
            "offline_dataset_id": dataset_id,
            "normalization": {
                "protocol": "d4rl_v2_reference",
                "reference_min_score": float(reference_min_score),
                "reference_max_score": float(reference_max_score),
                "percent": bool(normalized_score_percent),
            },
        }
        if required and not math.isfinite(normalized):
            raise RuntimeError("Required normalized return is unavailable or non-finite")
        if diagnostics_path is not None:
            atomic_write_json(diagnostics_path, result)
        return result
    except Exception as exc:
        failure: dict[str, Any] = {
            "rollout_status": "unavailable",
            "rollout_return_mean": float("nan"),
            "rollout_return_std": float("nan"),
            "normalized_return": float("nan"),
            "normalized_return_available": False,
            "rollout_episodes": 0,
            "rollout_backend": backend,
            "evaluation_env_id": env_id,
            "offline_dataset_id": dataset_id,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        if diagnostics_path is not None:
            atomic_write_json(diagnostics_path, failure)
        if required:
            raise RuntimeError(
                f"Required rollout evaluation failed for {env_id!r}: {exc}"
            ) from exc
        return failure
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

def sample_indices(rng: np.random.Generator, pool: np.ndarray, batch_size: int) -> np.ndarray:
    replace = len(pool) < batch_size
    return rng.choice(pool, size=batch_size, replace=replace)


def tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.float32, device=device)


def parameter_norm(parameters: Iterable[nn.Parameter]) -> float:
    total = 0.0
    for parameter in parameters:
        total += float(parameter.detach().square().sum().cpu())
    return math.sqrt(total)


def full_gradient_statistics(
    loss: torch.Tensor, parameters: Iterable[nn.Parameter]
) -> dict[str, float]:
    parameter_list = list(parameters)
    grads = torch.autograd.grad(
        loss, parameter_list, retain_graph=False, allow_unused=True
    )
    total_sq = 0.0
    elements = 0
    for grad in grads:
        if grad is not None:
            total_sq += float(grad.detach().square().sum().cpu())
            elements += int(grad.numel())
    raw = math.sqrt(total_sq)
    rms = raw / math.sqrt(max(elements, 1))
    relative = raw / max(parameter_norm(parameter_list), EPS)
    return {
        "raw": raw,
        "rms": rms,
        "relative_to_parameter_norm": relative,
        "elements": float(elements),
    }


def full_gradient_norm(loss: torch.Tensor, parameters: Iterable[nn.Parameter]) -> float:
    return full_gradient_statistics(loss, parameters)["raw"]


def parameter_update_statistics(
    previous: Sequence[torch.Tensor],
    parameters: Iterable[nn.Parameter],
    elapsed_steps: int,
) -> dict[str, float]:
    parameter_list = list(parameters)
    delta_sq = 0.0
    elements = 0
    for old, current in zip(previous, parameter_list):
        delta_sq += float((current.detach() - old).square().sum().cpu())
        elements += int(current.numel())
    elapsed = max(int(elapsed_steps), 1)
    delta = math.sqrt(delta_sq)
    raw_per_step = delta / elapsed
    rms_per_step = delta / math.sqrt(max(elements, 1)) / elapsed
    relative_per_step = delta / max(parameter_norm(parameter_list), EPS) / elapsed
    return {
        "raw_per_step": raw_per_step,
        "rms_per_step": rms_per_step,
        "relative_per_step": relative_per_step,
        "elements": float(elements),
    }


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    position = 0
    while position < len(values):
        stop = position + 1
        while stop < len(values) and values[order[stop]] == values[order[position]]:
            stop += 1
        average_rank = 0.5 * (position + stop - 1) + 1.0
        ranks[order[position:stop]] = average_rank
        position = stop
    return ranks


def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return pearson(_rankdata(y_true), _rankdata(y_pred))


def critic_advantage_arrays(
    *,
    critic: ValueNetwork,
    data: OfflineData,
    obs_norm: Normalizer,
    target_norm: Normalizer,
    gamma: float,
    standardize: bool,
    standardization_indices: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    obs = obs_norm.transform(data.observations)
    next_obs = obs_norm.transform(data.next_observations)
    values: list[np.ndarray] = []
    next_values: list[np.ndarray] = []
    critic.eval()
    with torch.no_grad():
        for offset in range(0, data.size, 65536):
            stop = min(data.size, offset + 65536)
            values.append(critic(tensor(obs[offset:stop], device)).cpu().numpy())
            next_values.append(
                critic(tensor(next_obs[offset:stop], device)).cpu().numpy()
            )
    value_norm = np.concatenate(values).astype(np.float32)
    next_value_norm = np.concatenate(next_values).astype(np.float32)
    value = value_norm * float(target_norm.std[0]) + float(target_norm.mean[0])
    next_value = (
        next_value_norm * float(target_norm.std[0]) + float(target_norm.mean[0])
    )
    bootstrap_mask = (~(data.terminals | data.timeouts)).astype(np.float32)
    raw = data.rewards + gamma * bootstrap_mask * next_value - value
    center = float(np.mean(raw[standardization_indices]))
    scale = float(np.std(raw[standardization_indices]))
    if standardize:
        advantage = ((raw - center) / max(scale, 1e-8)).astype(np.float32)
    else:
        advantage = raw.astype(np.float32)
        center, scale = 0.0, 1.0
    return {
        "advantage": advantage,
        "raw_advantage": raw.astype(np.float32),
        "value": value.astype(np.float32),
        "next_value": next_value.astype(np.float32),
        "center": center,
        "scale": scale,
    }

def train_critic(
    *,
    data: OfflineData,
    split: dict[str, np.ndarray],
    obs_norm: Normalizer,
    returns: np.ndarray,
    config: E7Config,
    mode: ModeConfig,
    seed: int,
    device: torch.device,
    output_dir: Path,
    heartbeat: Callable[[str, int], None] | None = None,
) -> tuple[ValueNetwork, Normalizer, dict[str, Any]]:
    """Train and accept a canonical frozen-advantage critic.

    The critic always consumes the configured fixed optimizer-step budget unless a
    non-finite loss or gradient makes continuation invalid. Validation MSE selects
    the canonical checkpoint after the full budget. Optimization-stationarity,
    predictive-quality, and best/final frozen-advantage-stability diagnostics remain
    recorded, but they neither stop training nor choose the checkpoint. Formal use
    requires a completed fixed budget and finite selected metrics; no acceptance
    field is forced to ``True``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    obs = obs_norm.transform(data.observations)
    target_norm = Normalizer.fit(returns[split["train"]].reshape(-1, 1))
    normalized_targets = target_norm.transform(returns.reshape(-1, 1)).reshape(-1)
    model = ValueNetwork(
        obs.shape[1],
        config.hidden_sizes,
        config.activation,
        config.init_scheme,
        config.init_gain,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.critic_lr, weight_decay=config.weight_decay
    )
    rng = np.random.default_rng(seed + 1000)
    rows: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_step = 0
    best_path = output_dir / "best_validation_critic.pt"
    final_path = output_dir / "final_training_critic.pt"
    selected_path = output_dir / "canonical_critic.pt"
    candidate_step: int | None = None
    extension_target: int | None = None
    early_stop_reason: str | None = None
    train_audit = rng.choice(
        split["train"],
        size=min(mode.audit_sample_size, len(split["train"])),
        replace=False,
    )
    validation_audit = rng.choice(
        split["validation"],
        size=min(mode.audit_sample_size, len(split["validation"])),
        replace=False,
    )
    eval_snapshot = [parameter.detach().clone() for parameter in model.parameters()]
    last_eval_step = 0

    def evaluate(step: int, update_stats: dict[str, float]) -> dict[str, Any]:
        model.eval()
        result: dict[str, Any] = {
            "step": step,
            "update_norm_per_step": update_stats["raw_per_step"],
            "update_rms_per_step": update_stats["rms_per_step"],
            "relative_update_norm_per_step": update_stats["relative_per_step"],
        }
        with torch.no_grad():
            for name in ("train", "validation", "test"):
                idx = split[name]
                predictions: list[np.ndarray] = []
                for offset in range(0, len(idx), 65536):
                    chunk = idx[offset : offset + 65536]
                    predictions.append(model(tensor(obs[chunk], device)).cpu().numpy())
                pred_norm = np.concatenate(predictions)
                pred = pred_norm * float(target_norm.std[0]) + float(target_norm.mean[0])
                truth = returns[idx]
                result[f"{name}_mse"] = float(np.mean((truth - pred) ** 2))
                result[f"{name}_r2"] = r2_score(truth, pred)
                result[f"{name}_pearson"] = pearson(truth, pred)
        for name, audit_indices in (
            ("train", train_audit),
            ("validation", validation_audit),
        ):
            audit_pred = model(tensor(obs[audit_indices], device))
            audit_target = tensor(normalized_targets[audit_indices], device)
            audit_loss = F.mse_loss(audit_pred, audit_target)
            gradient = full_gradient_statistics(audit_loss, model.parameters())
            result[f"{name}_audit_loss_normalized"] = float(audit_loss.detach().cpu())
            result[f"{name}_gradient_norm"] = gradient["raw"]
            result[f"{name}_gradient_rms"] = gradient["rms"]
            result[f"{name}_relative_gradient_norm"] = gradient[
                "relative_to_parameter_norm"
            ]
        model.train()
        return result

    model.train()
    for step in range(1, mode.critic_max_steps + 1):
        idx = sample_indices(rng, split["train"], config.critic_batch_size)
        pred = model(tensor(obs[idx], device))
        loss = F.mse_loss(pred, tensor(normalized_targets[idx], device))
        optimizer.zero_grad(set_to_none=True)
        loss_value = float(loss.detach().cpu())
        if not math.isfinite(loss_value):
            early_stop_reason = "nonfinite_train_loss"
            emit_event(
                {
                    "stage": "canonical_critic",
                    "step": step,
                    "train_batch_loss_normalized": loss_value,
                    "numerical_nonfinite": True,
                }
            )
            break
        loss.backward()
        train_gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0).detach().cpu()
        )
        if not math.isfinite(train_gradient_norm):
            early_stop_reason = "nonfinite_train_gradient"
            emit_event(
                {
                    "stage": "canonical_critic",
                    "step": step,
                    "train_batch_gradient_norm": train_gradient_norm,
                    "numerical_nonfinite": True,
                }
            )
            break
        optimizer.step()
        if step % mode.critic_eval_interval == 0 or step == mode.critic_max_steps:
            update_stats = parameter_update_statistics(
                eval_snapshot, model.parameters(), step - last_eval_step
            )
            eval_snapshot = [parameter.detach().clone() for parameter in model.parameters()]
            last_eval_step = step
            row = evaluate(step, update_stats)
            row["train_batch_loss_normalized"] = loss_value
            row["train_batch_gradient_norm"] = train_gradient_norm
            rows.append(row)
            if heartbeat is not None:
                heartbeat("canonical_critic", step)
            emit_event(
                {
                    "stage": "canonical_critic",
                    "step": step,
                    "validation_mse": row["validation_mse"],
                    "test_r2": row["test_r2"],
                    "train_gradient_norm_diagnostic": row["train_gradient_norm"],
                    "relative_update_norm_per_step": row[
                        "relative_update_norm_per_step"
                    ],
                }
            )
            val = float(row["validation_mse"])
            if val < best_loss:
                best_loss = val
                best_step = step
                torch.save(
                    {
                        "model": model.state_dict(),
                        "obs_mean": obs_norm.mean,
                        "obs_std": obs_norm.std,
                        "target_mean": target_norm.mean,
                        "target_std": target_norm.std,
                        "step": step,
                        "checkpoint_role": "best_validation_candidate",
                    },
                    best_path,
                )
            validation_slope = relative_slope(
                rows, "validation_mse", config.audit_windows
            )
            train_audit_slope = relative_slope(
                rows, "train_audit_loss_normalized", config.audit_windows
            )
            extension_possible = 2 * step <= mode.critic_max_steps
            if (
                candidate_step is None
                and step >= mode.critic_min_steps
                and extension_possible
                and validation_slope <= config.critic_relative_slope_tolerance
                and train_audit_slope <= config.critic_relative_slope_tolerance
                and float(row["relative_update_norm_per_step"])
                <= config.critic_update_tolerance
            ):
                candidate_step = step
                extension_target = 2 * step
            # Candidate/extension fields are post-hoc diagnostics only. Fixed-budget
            # training never stops because a short window appears stationary.

    if not rows or not best_path.is_file():
        raise RuntimeError("Critic produced no auditable checkpoint")
    final_step = int(rows[-1]["step"])
    fixed_budget_completed = bool(
        final_step == mode.critic_max_steps and early_stop_reason is None
    )
    final_metrics = dict(rows[-1])
    final_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save(
        {
            "model": final_state,
            "obs_mean": obs_norm.mean,
            "obs_std": obs_norm.std,
            "target_mean": target_norm.mean,
            "target_std": target_norm.std,
            "step": final_step,
            "checkpoint_role": "final_training_checkpoint",
        },
        final_path,
    )
    extension_complete = bool(
        candidate_step is not None and final_step >= 2 * candidate_step
    )
    final_validation_slope = relative_slope(
        rows, "validation_mse", config.audit_windows
    )
    final_train_audit_slope = relative_slope(
        rows, "train_audit_loss_normalized", config.audit_windows
    )
    final_stationarity_reconfirmed = bool(
        final_validation_slope <= config.critic_relative_slope_tolerance
        and final_train_audit_slope <= config.critic_relative_slope_tolerance
        and float(rows[-1]["relative_update_norm_per_step"])
        <= config.critic_update_tolerance
    )
    optimization_terminal = bool(
        candidate_step is not None
        and extension_complete
        and final_stationarity_reconfirmed
    )

    final_advantage = critic_advantage_arrays(
        critic=model,
        data=data,
        obs_norm=obs_norm,
        target_norm=target_norm,
        gamma=config.gamma,
        standardize=config.advantage_standardize,
        standardization_indices=split["train"],
        device=device,
    )["advantage"]
    best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model"])
    model.eval()
    selected_metrics = next(
        dict(row) for row in rows if int(row["step"]) == best_step
    )
    model.eval()
    best_advantage = critic_advantage_arrays(
        critic=model,
        data=data,
        obs_norm=obs_norm,
        target_norm=target_norm,
        gamma=config.gamma,
        standardize=config.advantage_standardize,
        standardization_indices=split["train"],
        device=device,
    )["advantage"]
    stability_indices = split["train"]
    best_advantage_stability = best_advantage[stability_indices]
    final_advantage_stability = final_advantage[stability_indices]
    sign_agreement = float(
        np.mean(
            np.sign(best_advantage_stability)
            == np.sign(final_advantage_stability)
        )
    )
    advantage_pearson = pearson(
        best_advantage_stability, final_advantage_stability
    )
    advantage_spearman = spearman(
        best_advantage_stability, final_advantage_stability
    )
    best_negative = best_advantage_stability < 0
    final_negative = final_advantage_stability < 0
    union = int(np.sum(best_negative | final_negative))
    negative_set_jaccard = (
        float(np.sum(best_negative & final_negative)) / union if union else 1.0
    )
    final_to_best_ratio = float(final_metrics["validation_mse"]) / max(
        float(selected_metrics["validation_mse"]), EPS
    )
    terminal_checkpoint_eligible = False
    selected_step = best_step
    selected_role_base = "best_validation_checkpoint"

    operational_acceptance_checks = {
        "fixed_budget_completed": fixed_budget_completed,
        "finite_selected_metrics": bool(
            all(
                math.isfinite(float(selected_metrics[key]))
                for key in ("validation_mse", "validation_r2", "validation_pearson")
            )
        ),
    }
    quality_audit_checks = {
        "validation_r2": bool(
            float(selected_metrics["validation_r2"])
            >= config.critic_validation_r2_min
        ),
        "validation_pearson": bool(
            float(selected_metrics["validation_pearson"])
            >= config.critic_validation_pearson_min
        ),
        "final_to_best_validation_mse_ratio": bool(
            final_to_best_ratio
            <= config.critic_max_final_to_best_validation_mse_ratio
        ),
        "advantage_sign_agreement": bool(
            sign_agreement >= config.critic_advantage_sign_agreement_min
        ),
        "advantage_pearson": bool(
            advantage_pearson >= config.critic_advantage_pearson_min
        ),
        "advantage_spearman": bool(
            advantage_spearman >= config.critic_advantage_spearman_min
        ),
        "negative_set_jaccard": bool(
            negative_set_jaccard >= config.critic_negative_set_jaccard_min
        ),
    }
    critic_accepted = all(operational_acceptance_checks.values())
    critic_quality_audit_passed = all(quality_audit_checks.values())
    selected_role = (
        selected_role_base
        if critic_accepted
        else f"{selected_role_base}_for_pilot_diagnostics"
    )
    torch.save(
        {
            "model": model.state_dict(),
            "obs_mean": obs_norm.mean,
            "obs_std": obs_norm.std,
            "target_mean": target_norm.mean,
            "target_std": target_norm.std,
            "step": selected_step,
            "candidate_step": candidate_step,
            "extension_complete": extension_complete,
            "fixed_budget_steps": mode.critic_max_steps,
            "fixed_budget_completed": fixed_budget_completed,
            "stopping_rule": "fixed_optimizer_steps",
            "optimization_terminal": optimization_terminal,
            "critic_accepted_for_frozen_advantage": critic_accepted,
            "critic_quality_audit_passed": critic_quality_audit_passed,
            "checkpoint_role": selected_role,
        },
        selected_path,
    )
    audit = {
        "best_step": best_step,
        "best_validation_mse": best_loss,
        "stopping_rule": "fixed_optimizer_steps",
        "fixed_budget_steps": mode.critic_max_steps,
        "fixed_budget_completed": fixed_budget_completed,
        "early_stop_reason": early_stop_reason,
        "terminal_audit_controls_stopping": False,
        "candidate_step": candidate_step,
        "extension_target": extension_target,
        "extension_complete": extension_complete,
        "final_stationarity_reconfirmed": final_stationarity_reconfirmed,
        "validation_mse_relative_slope": final_validation_slope,
        "train_audit_loss_relative_slope": final_train_audit_slope,
        "final_train_gradient_norm_diagnostic": rows[-1]["train_gradient_norm"],
        "final_validation_gradient_norm_diagnostic": rows[-1][
            "validation_gradient_norm"
        ],
        "final_update_norm_per_step_raw": rows[-1]["update_norm_per_step"],
        "final_relative_update_norm_per_step": rows[-1][
            "relative_update_norm_per_step"
        ],
        "optimization_terminal": optimization_terminal,
        "critic_accepted_for_frozen_advantage": critic_accepted,
        "operational_acceptance_checks": operational_acceptance_checks,
        "critic_quality_audit_passed": critic_quality_audit_passed,
        "quality_audit_checks": quality_audit_checks,
        "acceptance_checks": {
            **operational_acceptance_checks,
            **quality_audit_checks,
        },
        "acceptance_metrics": {
            "final_to_best_validation_mse_ratio": final_to_best_ratio,
            "advantage_sign_agreement": sign_agreement,
            "advantage_pearson": advantage_pearson,
            "advantage_spearman": advantage_spearman,
            "negative_set_jaccard": negative_set_jaccard,
            "stability_scope": "actor_training_split",
            "stability_sample_count": int(len(stability_indices)),
            "test_r2_report_only": float(selected_metrics["test_r2"]),
            "test_pearson_report_only": float(selected_metrics["test_pearson"]),
        },
        "selected_checkpoint_role": selected_role,
        "selected_checkpoint_step": selected_step,
        "terminal_checkpoint_eligible": terminal_checkpoint_eligible,
        "selected_checkpoint_metrics": selected_metrics,
        "statistical_note": (
            "The fixed optimizer-step budget controls critic stopping. Validation "
            "MSE always chooses the canonical checkpoint after that budget. "
            "Optimization terminality and thresholded predictive/advantage-stability "
            "checks are report-only diagnostics. Formal execution requires only a "
            "completed finite fixed-budget critic artifact; test metrics remain "
            "final-report-only."
        ),
        "checkpoint": {
            "path": str(selected_path),
            "sha256": sha256_file(selected_path),
            "size_bytes": selected_path.stat().st_size,
        },
        "best_validation_checkpoint": {
            "path": str(best_path),
            "sha256": sha256_file(best_path),
            "size_bytes": best_path.stat().st_size,
            "role": "canonical_selection_after_fixed_budget",
        },
        "final_training_checkpoint": {
            "path": str(final_path),
            "sha256": sha256_file(final_path),
            "size_bytes": final_path.stat().st_size,
            "role": "fixed_budget_final_and_advantage_stability_comparator",
        },
        "final_training_metrics": final_metrics,
    }
    write_csv(output_dir / "critic_metrics.csv", rows)
    atomic_write_json(output_dir / "critic_terminal_audit.json", audit)
    atomic_write_json(
        output_dir / "critic_normalizers.json",
        {
            "observation_mean": obs_norm.mean,
            "observation_std": obs_norm.std,
            "return_target_mean": target_norm.mean,
            "return_target_std": target_norm.std,
        },
    )
    return model, target_norm, audit

def freeze_advantages(
    *,
    critic: ValueNetwork,
    data: OfflineData,
    obs_norm: Normalizer,
    target_norm: Normalizer,
    gamma: float,
    standardize: bool,
    standardization_indices: np.ndarray,
    device: torch.device,
    output_dir: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = critic_advantage_arrays(
        critic=critic,
        data=data,
        obs_norm=obs_norm,
        target_norm=target_norm,
        gamma=gamma,
        standardize=standardize,
        standardization_indices=standardization_indices,
        device=device,
    )
    advantage = arrays["advantage"]
    path = output_dir / "frozen_advantages.npz"
    np.savez_compressed(
        path,
        advantage=advantage,
        raw_advantage=arrays["raw_advantage"],
        value=arrays["value"],
        next_value=arrays["next_value"],
        center=np.asarray(arrays["center"], dtype=np.float32),
        scale=np.asarray(arrays["scale"], dtype=np.float32),
    )
    manifest = {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "count": len(advantage),
        "standardized_once": standardize,
        "standardization_population": "critic_train_episode_split_only",
        "standardization_count": int(len(standardization_indices)),
        "center": arrays["center"],
        "scale": arrays["scale"],
        "positive_fraction": float(np.mean(advantage > 0)),
        "negative_fraction": float(np.mean(advantage < 0)),
        "zero_fraction": float(np.mean(advantage == 0)),
        "frozen": True,
        "min": float(np.min(advantage)),
        "max": float(np.max(advantage)),
        "mean": float(np.mean(advantage)),
        "std": float(np.std(advantage)),
    }
    atomic_write_json(output_dir / "advantage_manifest.json", manifest)
    return advantage, manifest



@dataclass
class CanonicalCriticContext:
    root: Path
    split: dict[str, np.ndarray]
    obs_norm: Normalizer
    target_norm: Normalizer
    critic: ValueNetwork
    advantages: np.ndarray
    critic_audit: dict[str, Any]
    advantage_manifest: dict[str, Any]
    artifact_manifest: dict[str, Any]
    reused: bool


def _artifact_file_record(path: Path, root: Path) -> dict[str, Any]:
    relative = path.resolve().relative_to(root.resolve())
    return {
        "path": relative.as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _artifact_path(root: Path, record: dict[str, Any]) -> Path:
    relative = Path(str(record["path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"Unsafe canonical critic artifact path: {relative}")
    return root / relative


def _verify_artifact_file(root: Path, record: dict[str, Any], label: str) -> Path:
    path = _artifact_path(root, record)
    if not path.is_file():
        raise RuntimeError(f"Canonical critic artifact is missing {label}: {path}")
    actual = sha256_file(path)
    if actual != record.get("sha256"):
        raise RuntimeError(
            f"Canonical critic artifact hash mismatch for {label}: "
            f"expected {record.get('sha256')}, got {actual}"
        )
    if int(record.get("size_bytes", -1)) != path.stat().st_size:
        raise RuntimeError(f"Canonical critic artifact size mismatch for {label}: {path}")
    return path


def _canonical_identity(
    *,
    config_path: Path,
    dataset_manifest: dict[str, Any],
    data: OfflineData,
    mode_name: str,
    mode: ModeConfig,
) -> dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "runner_version": RUNNER_VERSION,
        "mode": mode_name,
        "config_sha256": sha256_file(config_path),
        "dataset_basename": dataset_manifest["basename"],
        "dataset_sha256": dataset_manifest["sha256"],
        "dataset_size_bytes": dataset_manifest["size_bytes"],
        "transitions_loaded": data.size,
        "max_transitions": mode.max_transitions,
        "observation_dim": int(data.observations.shape[1]),
        "action_dim": int(data.actions.shape[1]),
        "canonical_critic_seed": mode.canonical_critic_seed,
    }


def _load_canonical_critic_context(
    *,
    root: Path,
    expected_identity: dict[str, Any],
    config: E7Config,
    data: OfflineData,
    device: torch.device,
    require_accepted: bool,
) -> CanonicalCriticContext:
    manifest_path = root / "canonical_critic_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(
            f"Canonical critic artifact does not contain {manifest_path.name}: {root}"
        )
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 3 or not manifest.get("complete"):
        raise RuntimeError("Canonical critic artifact is incomplete or has an unsupported schema")
    if manifest.get("identity") != expected_identity:
        raise RuntimeError(
            "Canonical critic artifact identity does not match this run. "
            f"Expected {expected_identity}, got {manifest.get('identity')}"
        )
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise RuntimeError("Canonical critic manifest is missing its file inventory")
    checkpoint_path = _verify_artifact_file(root, files["critic_checkpoint"], "critic checkpoint")
    split_path = _verify_artifact_file(root, files["episode_split"], "episode split")
    advantage_path = _verify_artifact_file(root, files["frozen_advantages"], "frozen advantages")
    critic_audit_path = _verify_artifact_file(root, files["critic_audit"], "critic audit")
    advantage_manifest_path = _verify_artifact_file(
        root, files["advantage_manifest"], "advantage manifest"
    )

    critic_audit = json.loads(critic_audit_path.read_text())
    if require_accepted and not bool(
        critic_audit.get("critic_accepted_for_frozen_advantage")
    ):
        raise RuntimeError(
            "Formal E7 requires a canonical critic accepted for frozen-advantage use"
        )
    split_payload = np.load(split_path)
    split = {
        name: np.asarray(split_payload[name], dtype=np.int64)
        for name in ("train", "validation", "test")
    }
    all_indices = np.concatenate(list(split.values()))
    if len(all_indices) != data.size or len(np.unique(all_indices)) != data.size:
        raise RuntimeError("Canonical critic episode split does not partition the loaded dataset")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    obs_norm = Normalizer(
        mean=np.asarray(checkpoint["obs_mean"], dtype=np.float32),
        std=np.asarray(checkpoint["obs_std"], dtype=np.float32),
    )
    target_norm = Normalizer(
        mean=np.asarray(checkpoint["target_mean"], dtype=np.float32),
        std=np.asarray(checkpoint["target_std"], dtype=np.float32),
    )
    critic = ValueNetwork(
        data.observations.shape[1],
        config.hidden_sizes,
        config.activation,
        config.init_scheme,
        config.init_gain,
    ).to(device)
    critic.load_state_dict(checkpoint["model"])
    critic.eval()
    for parameter in critic.parameters():
        parameter.requires_grad_(False)

    advantage_payload = np.load(advantage_path)
    advantages = np.asarray(advantage_payload["advantage"], dtype=np.float32)
    if len(advantages) != data.size:
        raise RuntimeError(
            f"Frozen advantage count {len(advantages)} does not match data size {data.size}"
        )
    advantage_manifest = json.loads(advantage_manifest_path.read_text())
    return CanonicalCriticContext(
        root=root,
        split=split,
        obs_norm=obs_norm,
        target_norm=target_norm,
        critic=critic,
        advantages=advantages,
        critic_audit=critic_audit,
        advantage_manifest=advantage_manifest,
        artifact_manifest=manifest,
        reused=True,
    )


def prepare_canonical_critic_context(
    *,
    data: OfflineData,
    config: E7Config,
    mode: ModeConfig,
    mode_name: str,
    config_path: Path,
    dataset_manifest: dict[str, Any],
    device: torch.device,
    artifact_root: Path,
    reuse_root: Path | None,
    heartbeat: Callable[[str, int], None] | None = None,
) -> CanonicalCriticContext:
    """Prepare or strictly reuse the one critic/advantage artifact for all seeds."""
    expected_identity = _canonical_identity(
        config_path=config_path,
        dataset_manifest=dataset_manifest,
        data=data,
        mode_name=mode_name,
        mode=mode,
    )
    require_accepted = mode_name == "formal"
    if reuse_root is not None:
        return _load_canonical_critic_context(
            root=reuse_root.resolve(),
            expected_identity=expected_identity,
            config=config,
            data=data,
            device=device,
            require_accepted=require_accepted,
        )
    manifest_path = artifact_root / "canonical_critic_manifest.json"
    if manifest_path.is_file():
        return _load_canonical_critic_context(
            root=artifact_root,
            expected_identity=expected_identity,
            config=config,
            data=data,
            device=device,
            require_accepted=require_accepted,
        )
    if artifact_root.exists() and any(artifact_root.iterdir()):
        raise RuntimeError(
            f"Canonical critic directory is non-empty but incomplete: {artifact_root}"
        )
    artifact_root.mkdir(parents=True, exist_ok=True)
    seed_everything(mode.canonical_critic_seed)
    split = split_episode_indices(
        data.episode_ids,
        mode.canonical_critic_seed,
        config.train_fraction,
        config.validation_fraction,
    )
    obs_norm = Normalizer.fit(data.observations[split["train"]])
    returns = discounted_returns(data.rewards, data.terminals, data.timeouts, config.gamma)
    training_dir = artifact_root / "training"
    critic, target_norm, critic_audit = train_critic(
        data=data,
        split=split,
        obs_norm=obs_norm,
        returns=returns,
        config=config,
        mode=mode,
        seed=mode.canonical_critic_seed,
        device=device,
        output_dir=training_dir,
        heartbeat=heartbeat,
    )
    split_path = artifact_root / "episode_split.npz"
    np.savez_compressed(split_path, **split)
    advantages, advantage_manifest = freeze_advantages(
        critic=critic,
        data=data,
        obs_norm=obs_norm,
        target_norm=target_norm,
        gamma=config.gamma,
        standardize=config.advantage_standardize,
        standardization_indices=split["train"],
        device=device,
        output_dir=artifact_root / "frozen_advantage",
    )
    for parameter in critic.parameters():
        parameter.requires_grad_(False)
    if require_accepted and not bool(
        critic_audit.get("critic_accepted_for_frozen_advantage")
    ):
        failure_manifest = {
            "schema_version": 3,
            "complete": False,
            "identity": expected_identity,
            "created_utc": utc_now(),
            "reason": "critic_failed_frozen_advantage_acceptance",
            "critic_fixed_budget_completed": bool(
                critic_audit.get("fixed_budget_completed")
            ),
            "critic_optimization_terminal": bool(
                critic_audit.get("optimization_terminal")
            ),
            "critic_accepted_for_frozen_advantage": False,
            "operational_acceptance_checks": critic_audit.get(
                "operational_acceptance_checks", {}
            ),
            "quality_audit_checks": critic_audit.get("quality_audit_checks", {}),
        }
        atomic_write_json(
            artifact_root / "canonical_critic_incomplete.json", failure_manifest
        )
        raise RuntimeError(
            "Formal E7 gate failed: the canonical critic did not complete its "
            "fixed optimizer-step budget with finite selected metrics"
        )
    checkpoint_path = Path(critic_audit["checkpoint"]["path"])
    critic_audit_path = training_dir / "critic_terminal_audit.json"
    advantage_path = artifact_root / "frozen_advantage" / "frozen_advantages.npz"
    advantage_manifest_path = artifact_root / "frozen_advantage" / "advantage_manifest.json"
    manifest = {
        "schema_version": 3,
        "complete": True,
        "created_utc": utc_now(),
        "identity": expected_identity,
        "critic_training_count": 1,
        "shared_across_all_actor_seeds": True,
        "critic_updates_during_actor_training": False,
        "advantage_recomputation_during_actor_training": False,
        "critic_fixed_budget_completed": bool(
            critic_audit["fixed_budget_completed"]
        ),
        "critic_optimization_terminal": bool(critic_audit["optimization_terminal"]),
        "critic_accepted_for_frozen_advantage": bool(
            critic_audit["critic_accepted_for_frozen_advantage"]
        ),
        "critic_quality_audit_passed": bool(
            critic_audit["critic_quality_audit_passed"]
        ),
        "checkpoint_role": critic_audit["selected_checkpoint_role"],
        "files": {
            "critic_checkpoint": _artifact_file_record(checkpoint_path, artifact_root),
            "episode_split": _artifact_file_record(split_path, artifact_root),
            "frozen_advantages": _artifact_file_record(advantage_path, artifact_root),
            "critic_audit": _artifact_file_record(critic_audit_path, artifact_root),
            "advantage_manifest": _artifact_file_record(
                advantage_manifest_path, artifact_root
            ),
        },
    }
    atomic_write_json(manifest_path, manifest)
    return CanonicalCriticContext(
        root=artifact_root,
        split=split,
        obs_norm=obs_norm,
        target_norm=target_norm,
        critic=critic,
        advantages=advantages,
        critic_audit=critic_audit,
        advantage_manifest=advantage_manifest,
        artifact_manifest=manifest,
        reused=False,
    )

def actor_eval_metrics(
    *,
    policy: SquashedGaussianPolicy,
    obs: np.ndarray,
    actions: np.ndarray,
    advantages: np.ndarray,
    audit_indices: np.ndarray,
    fixed_negative_indices: np.ndarray,
    device: torch.device,
    loss_value: float,
    gradient_norm: float,
    gradient_rms: float,
    relative_gradient_norm: float,
    update_norm: float,
    relative_update_norm: float,
    step: int,
    boundary_threshold: float,
    rollout_metrics: dict[str, float] | None = None,
) -> dict[str, Any]:
    policy.eval()
    with torch.no_grad():
        idx = audit_indices
        obs_t = tensor(obs[idx], device)
        action_t = tensor(actions[idx], device)
        lp = policy.log_prob(obs_t, action_t)
        adv_t = tensor(advantages[idx], device)
        pos = adv_t > 0
        positive_nll = float((-lp[pos]).mean().cpu()) if bool(pos.any()) else float("nan")
        mean_latent, log_std = policy.latent_parameters(obs_t)
        action_mean = torch.tanh(mean_latent)
        boundary = (action_mean.abs() >= boundary_threshold).any(dim=-1)
        mean_boundary_fraction = float(boundary.float().mean().cpu())
        sigma = torch.exp(log_std)
        log_std_vector = policy.log_std.detach().clamp(policy.log_std_min, policy.log_std_max)
        min_fraction = float((log_std_vector <= policy.log_std_min + 1e-7).float().mean().cpu())
        max_fraction = float((log_std_vector >= policy.log_std_max - 1e-7).float().mean().cpu())
        neg_obs = tensor(obs[fixed_negative_indices], device)
        neg_actions = tensor(actions[fixed_negative_indices], device)
        components = policy.score_components(neg_obs, neg_actions)
        metrics: dict[str, Any] = {
            "step": step,
            "loss": loss_value,
            "positive_nll": positive_nll,
            "gradient_norm": gradient_norm,
            "gradient_rms": gradient_rms,
            "relative_gradient_norm": relative_gradient_norm,
            "update_norm": update_norm,
            "relative_update_norm": relative_update_norm,
            "mean_abs": float(action_mean.abs().mean().cpu()),
            "mean_boundary_fraction": mean_boundary_fraction,
            "sigma_mean": float(sigma.mean().cpu()),
            "sigma_min": float(sigma.min().cpu()),
            "sigma_max": float(sigma.max().cpu()),
            "log_std_min_fraction": min_fraction,
            "log_std_max_fraction": max_fraction,
            "phantom_distance_mean": float(components["radius"].mean().cpu()),
            "phantom_mean_score_norm": float(components["mean_score_norm"].mean().cpu()),
            "phantom_raw_log_scale_score_norm": float(
                components["raw_log_scale_score_norm"].mean().cpu()
            ),
            "phantom_corrected_q_xi_mean": float(components["corrected_q_xi"].mean().cpu()),
            "phantom_joint_output_score_mean": float(
                components["joint_output_score_norm"].mean().cpu()
            ),
            "phantom_log_scale_to_mean_ratio": float(
                components["log_scale_to_mean_ratio"].mean().cpu()
            ),
        }
        # Compatibility field retained for historical plotting utilities.
        metrics["phantom_score_mean"] = metrics["phantom_joint_output_score_mean"]
    if rollout_metrics:
        metrics.update(rollout_metrics)
    else:
        metrics.update(
            {
                "rollout_status": "not_evaluated",
                "rollout_return_mean": float("nan"),
                "rollout_return_std": float("nan"),
                "normalized_return": float("nan"),
                "normalized_return_available": False,
                "rollout_episodes": 0,
            }
        )
    policy.train()
    return metrics


def actor_batch_loss(
    policy: SquashedGaussianPolicy,
    obs_t: torch.Tensor,
    actions_t: torch.Tensor,
    advantages_t: torch.Tensor,
    method: str,
    far_threshold: float,
    global_scale: float,
    far_cap_score: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    log_prob = policy.log_prob(obs_t, actions_t)
    weights = advantages_t.clone()
    components = policy.score_components(obs_t, actions_t)
    distance = components["radius"].detach()
    joint_score = components["joint_output_score_norm"].detach()
    negative = weights < 0
    far = negative & (distance > far_threshold)
    near = negative & ~far
    cap_factor = torch.ones_like(weights)
    dynamic_scale = 1.0
    proxy_before = 0.0
    proxy_target = 0.0
    proxy_after = 0.0
    if method == "positive_only":
        weights = torch.where(weights > 0, weights, torch.zeros_like(weights))
    elif method == "near_zero":
        weights = torch.where(near, torch.zeros_like(weights), weights)
    elif method == "far_zero":
        weights = torch.where(far, torch.zeros_like(weights), weights)
    elif method == "far_cap":
        cap_factor = torch.minimum(
            torch.ones_like(weights),
            torch.full_like(weights, far_cap_score) / joint_score.clamp_min(EPS),
        )
        weights = torch.where(far, weights * cap_factor, weights)
    elif method == "dynamic_budget_matched_global":
        if bool(negative.any()):
            magnitude = (-weights[negative]).detach()
            score = joint_score[negative]
            negative_far = far[negative]
            target_factor = torch.ones_like(magnitude)
            target_factor = torch.where(
                negative_far,
                torch.minimum(
                    torch.ones_like(magnitude),
                    torch.full_like(magnitude, far_cap_score)
                    / score.clamp_min(EPS),
                ),
                target_factor,
            )
            proxy_before_t = torch.sum(magnitude * score)
            proxy_target_t = torch.sum(magnitude * score * target_factor)
            dynamic_scale = float(
                torch.clamp(proxy_target_t / proxy_before_t.clamp_min(EPS), 0.0, 1.0)
                .detach()
                .cpu()
            )
            weights = torch.where(negative, weights * dynamic_scale, weights)
            proxy_before = float(proxy_before_t.detach().cpu())
            proxy_target = float(proxy_target_t.detach().cpu())
            proxy_after = proxy_before * dynamic_scale
    elif method == "budget_matched_global":
        # Compatibility alias for pre-v4.2 tests/artifacts.  It is not a formal method.
        weights = torch.where(negative, weights * global_scale, weights)
        dynamic_scale = float(global_scale)
    elif method != "signed":
        raise ValueError(f"Unknown method: {method}")
    active = weights.ne(0)
    if not bool(active.any()):
        raise RuntimeError(f"Method {method} produced an empty active batch")
    loss = -(weights[active] * log_prob[active]).mean()
    diagnostics = {
        "active_fraction": float(active.float().mean().detach().cpu()),
        "negative_fraction": float(negative.float().mean().detach().cpu()),
        "far_negative_fraction": float(far.float().mean().detach().cpu()),
        "far_cap_factor_mean": float(cap_factor[far].mean().detach().cpu())
        if bool(far.any())
        else 1.0,
        "dynamic_global_scale": dynamic_scale,
        "negative_influence_proxy_before": proxy_before,
        "negative_influence_proxy_target": proxy_target,
        "negative_influence_proxy_after": proxy_after,
    }
    return loss, diagnostics

def train_actor_stage(
    *,
    policy: SquashedGaussianPolicy,
    method: str,
    obs: np.ndarray,
    actions: np.ndarray,
    advantages: np.ndarray,
    train_indices: np.ndarray,
    audit_indices: np.ndarray,
    fixed_negative_indices: np.ndarray,
    config: E7Config,
    min_steps: int,
    max_steps: int,
    eval_interval: int,
    seed: int,
    device: torch.device,
    output_dir: Path,
    far_threshold: float = float("inf"),
    global_scale: float = 1.0,
    far_cap_score: float = float("inf"),
    rollout_evaluator: Callable[[SquashedGaussianPolicy, int, str], dict[str, Any]] | None = None,
    rollout_eval_interval: int = 0,
    heartbeat: Callable[[str, int], None] | None = None,
) -> tuple[SquashedGaussianPolicy, dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=config.actor_lr, weight_decay=config.weight_decay
    )
    rng = np.random.default_rng(seed + 2000)
    rows: list[dict[str, Any]] = []
    candidate_step: int | None = None
    extension_target: int | None = None
    train_batch_loss = float("nan")
    train_batch_gradient = float("inf")
    early_stop_reason: str | None = None
    last_diag: dict[str, float] = {}
    eval_snapshot = [parameter.detach().clone() for parameter in policy.parameters()]
    last_eval_step = 0

    def evaluate(step: int, update_norm_per_step: dict[str, float]) -> dict[str, Any]:
        audit_obs = tensor(obs[audit_indices], device)
        audit_actions = tensor(actions[audit_indices], device)
        audit_advantages = tensor(advantages[audit_indices], device)
        audit_loss, audit_diag = actor_batch_loss(
            policy,
            audit_obs,
            audit_actions,
            audit_advantages,
            method,
            far_threshold,
            global_scale,
            far_cap_score,
        )
        audit_gradient = full_gradient_statistics(audit_loss, policy.parameters())
        row = actor_eval_metrics(
            policy=policy,
            obs=obs,
            actions=actions,
            advantages=advantages,
            audit_indices=audit_indices,
            fixed_negative_indices=fixed_negative_indices,
            device=device,
            loss_value=float(audit_loss.detach().cpu()),
            gradient_norm=audit_gradient["raw"],
            gradient_rms=audit_gradient["rms"],
            relative_gradient_norm=audit_gradient["relative_to_parameter_norm"],
            update_norm=update_norm_per_step["raw_per_step"],
            relative_update_norm=update_norm_per_step["relative_per_step"],
            step=step,
            boundary_threshold=config.support_boundary_threshold,
            rollout_metrics=(
                rollout_evaluator(policy, step, method)
                if rollout_evaluator
                and (step == 0 or rollout_eval_interval <= 0 or step % rollout_eval_interval == 0)
                else None
            ),
        )
        row.update({f"audit_{key}": value for key, value in audit_diag.items()})
        row.update(
            {
                "train_batch_loss": train_batch_loss,
                "train_batch_gradient_norm": train_batch_gradient,
            }
        )
        return row

    policy.train()
    initial_row = evaluate(
        0, {"raw_per_step": 0.0, "rms_per_step": 0.0, "relative_per_step": 0.0}
    )
    rows.append(initial_row)
    if heartbeat is not None:
        heartbeat(f"actor:{method}", 0)
    for step in range(1, max_steps + 1):
        idx = sample_indices(rng, train_indices, config.actor_batch_size)
        obs_t = tensor(obs[idx], device)
        actions_t = tensor(actions[idx], device)
        advantages_t = tensor(advantages[idx], device)
        loss, last_diag = actor_batch_loss(
            policy,
            obs_t,
            actions_t,
            advantages_t,
            method,
            far_threshold,
            global_scale,
            far_cap_score,
        )
        optimizer.zero_grad(set_to_none=True)
        train_batch_loss = float(loss.detach().cpu())
        if not math.isfinite(train_batch_loss):
            early_stop_reason = "nonfinite_train_loss"
            train_batch_gradient = float("nan")
        else:
            loss.backward()
            train_batch_gradient = float(
                torch.nn.utils.clip_grad_norm_(
                    policy.parameters(), config.max_gradient_norm
                ).cpu()
            )
            if not math.isfinite(train_batch_gradient):
                early_stop_reason = "nonfinite_train_gradient"
            else:
                optimizer.step()
        if early_stop_reason is not None:
            # Do not apply an optimizer step after a non-finite loss/gradient.  The
            # last finite policy remains available for post-mortem diagnostics.
            optimizer.zero_grad(set_to_none=True)
            row = actor_eval_metrics(
                policy=policy,
                obs=obs,
                actions=actions,
                advantages=advantages,
                audit_indices=audit_indices,
                fixed_negative_indices=fixed_negative_indices,
                device=device,
                loss_value=train_batch_loss,
                gradient_norm=train_batch_gradient,
                gradient_rms=float("nan"),
                relative_gradient_norm=float("nan"),
                update_norm=float("nan"),
                relative_update_norm=float("nan"),
                step=step,
                boundary_threshold=config.support_boundary_threshold,
            )
            row.update(last_diag)
            row["numerical_failure_reason"] = early_stop_reason
            rows.append(row)
            if heartbeat is not None:
                heartbeat(f"actor:{method}", step)
            emit_event(
                {
                    "stage": f"actor:{method}",
                    "step": step,
                    "train_batch_loss": train_batch_loss,
                    "train_batch_gradient_norm": train_batch_gradient,
                    "numerical_nonfinite": True,
                    "numerical_failure_reason": early_stop_reason,
                }
            )
            break
        if step % eval_interval == 0 or step == max_steps:
            update_stats = parameter_update_statistics(
                eval_snapshot, policy.parameters(), step - last_eval_step
            )
            eval_snapshot = [parameter.detach().clone() for parameter in policy.parameters()]
            last_eval_step = step
            row = evaluate(step, update_stats)
            rows.append(row)
            if heartbeat is not None:
                heartbeat(f"actor:{method}", step)
            emit_event(
                {
                    "stage": f"actor:{method}",
                    "step": step,
                    "audit_loss": row["loss"],
                    "positive_nll": row["positive_nll"],
                    "audit_gradient_norm": row["gradient_norm"],
                    "update_norm_per_step_raw": row["update_norm"],
                    "relative_update_norm_per_step": row["relative_update_norm"],
                }
            )
            state_drifts = [
                normalized_window_drift(rows, key, config.audit_windows)
                for key in ("mean_abs", "sigma_mean", "phantom_distance_mean")
            ]
            if (
                candidate_step is None
                and step >= min_steps
                and 2 * step <= max_steps
                and all(
                    value <= config.actor_state_drift_tolerance
                    for value in state_drifts
                )
                and float(row["relative_update_norm"])
                <= config.actor_update_tolerance
            ):
                candidate_step = step
                extension_target = 2 * step
            # Candidate/extension fields classify the fixed-budget endpoint only.
            # They never shorten the registered horizon.

    parameters_finite = all(
        bool(torch.isfinite(parameter).all()) for parameter in policy.parameters()
    )
    if (
        rollout_evaluator
        and parameters_finite
        and not math.isfinite(float(rows[-1].get("normalized_return", float("nan"))))
    ):
        rows[-1].update(
            rollout_evaluator(policy, int(rows[-1]["step"]), method)
        )

    final_step = int(rows[-1]["step"])
    fixed_budget_completed = bool(
        final_step == max_steps
        and early_stop_reason is None
        and parameters_finite
        and math.isfinite(float(rows[-1].get("loss", float("nan"))))
    )
    if not fixed_budget_completed and early_stop_reason is None:
        early_stop_reason = "incomplete_fixed_budget_unknown_reason"
    checkpoint_path = output_dir / "terminal_actor.pt"
    torch.save(
        {
            "model": policy.state_dict(),
            "method": method,
            "step": final_step,
            "checkpoint_role": "fixed_budget_final_checkpoint",
            "fixed_budget_steps": max_steps,
            "fixed_budget_completed": fixed_budget_completed,
            "far_threshold": far_threshold,
            "global_scale": global_scale,
            "global_scale_semantics": (
                "dynamic_per_batch_detached_output_score_proxy"
                if method == "dynamic_budget_matched_global"
                else "fixed_compatibility_or_unused"
            ),
            "far_cap_score": far_cap_score,
        },
        checkpoint_path,
    )
    extension_complete = bool(
        candidate_step is not None and rows[-1]["step"] >= 2 * candidate_step
    )
    audit = classify_actor_terminal(
        rows,
        config,
        candidate_step,
        extension_complete,
        fixed_budget_completed=fixed_budget_completed,
    )
    audit.update(
        {
            "method": method,
            "final_step": final_step,
            "max_steps": max_steps,
            "fixed_budget_steps": max_steps,
            "fixed_budget_completed": fixed_budget_completed,
            "reached_max_steps": fixed_budget_completed,
            "stopping_rule": "fixed_optimizer_steps",
            "early_stop_reason": early_stop_reason,
            "terminal_audit_controls_stopping": False,
            "extension_target": extension_target,
            "far_threshold": far_threshold,
            "global_scale": global_scale,
            "global_scale_semantics": (
                "dynamic_per_batch_detached_output_score_proxy"
                if method == "dynamic_budget_matched_global"
                else "fixed_compatibility_or_unused"
            ),
            "far_cap_score": far_cap_score,
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": sha256_file(checkpoint_path),
                "size_bytes": checkpoint_path.stat().st_size,
            },
            "final_metrics": rows[-1],
        }
    )
    audit["terminal_audit_complete"] = bool(
        audit["fixed_budget_completed"] or audit["numerical_nonfinite"]
    )
    write_csv(output_dir / "curves.csv", rows)
    atomic_write_json(output_dir / "terminal_audit.json", audit)
    return policy, audit


# ---------------------------------------------------------------------------
# Near/far matching and gradient probes
# ---------------------------------------------------------------------------


def match_near_far_indices(
    advantages: np.ndarray,
    distances: np.ndarray,
    negative_indices: np.ndarray,
    near_quantile: float,
    far_quantile: float,
    bins: int,
    max_pairs: int,
    relative_tolerance: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    neg_dist = distances[negative_indices]
    near_cut = float(np.quantile(neg_dist, near_quantile))
    far_cut = float(np.quantile(neg_dist, far_quantile))
    near_pool = negative_indices[neg_dist <= near_cut]
    far_pool = negative_indices[neg_dist >= far_cut]
    magnitude = np.abs(advantages)
    all_mag = magnitude[negative_indices]
    edges = np.unique(np.quantile(all_mag, np.linspace(0.0, 1.0, bins + 1)))
    rng = np.random.default_rng(seed)
    pairs: list[tuple[int, int, float]] = []
    used_near: set[int] = set()
    used_far: set[int] = set()
    for left, right in zip(edges[:-1], edges[1:]):
        near = near_pool[(magnitude[near_pool] >= left) & (magnitude[near_pool] <= right)]
        far = far_pool[(magnitude[far_pool] >= left) & (magnitude[far_pool] <= right)]
        if len(near) == 0 or len(far) == 0:
            continue
        rng.shuffle(near)
        far_sorted = far[np.argsort(magnitude[far])]
        for near_idx in near:
            near_idx = int(near_idx)
            if near_idx in used_near:
                continue
            position = int(np.searchsorted(magnitude[far_sorted], magnitude[near_idx]))
            candidates = far_sorted[max(0, position - 4) : min(len(far_sorted), position + 5)]
            candidates = np.asarray([int(x) for x in candidates if int(x) not in used_far], dtype=np.int64)
            if len(candidates) == 0:
                continue
            far_idx = int(candidates[np.argmin(np.abs(magnitude[candidates] - magnitude[near_idx]))])
            rel = abs(float(magnitude[far_idx] - magnitude[near_idx])) / max(
                float(magnitude[near_idx]), 1e-8
            )
            if rel <= relative_tolerance:
                pairs.append((near_idx, far_idx, rel))
                used_near.add(near_idx)
                used_far.add(far_idx)
    if not pairs:
        raise RuntimeError("No advantage-matched near/far pairs were found")
    rng.shuffle(pairs)
    pairs = pairs[:max_pairs]
    near_indices = np.asarray([x[0] for x in pairs], dtype=np.int64)
    far_indices = np.asarray([x[1] for x in pairs], dtype=np.int64)
    summary = {
        "near_cut": near_cut,
        "far_cut": far_cut,
        "pairs": len(pairs),
        "mean_relative_advantage_error": float(np.mean([x[2] for x in pairs])),
        "advantage_magnitude_far_near_ratio": float(
            np.mean(magnitude[far_indices]) / max(np.mean(magnitude[near_indices]), EPS)
        ),
        "distance_far_near_ratio": float(
            np.mean(distances[far_indices]) / max(np.mean(distances[near_indices]), EPS)
        ),
    }
    return near_indices, far_indices, summary


def per_sample_gradient_norm(
    policy: SquashedGaussianPolicy,
    obs: np.ndarray,
    actions: np.ndarray,
    advantages: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    norms: list[float] = []
    policy.eval()
    parameters = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    for idx in indices:
        obs_t = tensor(obs[idx : idx + 1], device)
        action_t = tensor(actions[idx : idx + 1], device)
        advantage_t = tensor(advantages[idx : idx + 1], device)
        loss = -(advantage_t * policy.log_prob(obs_t, action_t)).mean()
        norms.append(full_gradient_norm(loss, parameters))
    return np.asarray(norms, dtype=np.float64)


def loglog_slope(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if int(mask.sum()) < 3:
        return float("nan")
    return float(np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)[0])


def analytic_output_autograd_relative_error(
    policy: SquashedGaussianPolicy,
    obs: np.ndarray,
    actions: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
    max_samples: int = 8,
) -> float:
    errors: list[float] = []
    for index in indices[:max_samples]:
        obs_t = tensor(obs[index : index + 1], device)
        action_t = tensor(actions[index : index + 1], device)
        latent = policy.inverse_action(action_t).detach()
        with torch.no_grad():
            mean0, log_std0 = policy.latent_parameters(obs_t)
        mean = mean0.detach().clone().requires_grad_(True)
        log_std = log_std0.detach().clone().requires_grad_(True)
        std = torch.exp(log_std)
        z = (latent - mean) / std
        base_log_prob = -0.5 * (
            z.square() + 2.0 * log_std + math.log(2.0 * math.pi)
        ).sum()
        grad_mean, grad_log_std = torch.autograd.grad(
            base_log_prob, [mean, log_std]
        )
        analytic_mean = (latent - mean.detach()) / std.detach().square()
        analytic_log_std = z.detach().square() - 1.0
        numerator = torch.sqrt(
            (grad_mean - analytic_mean).square().sum()
            + (grad_log_std - analytic_log_std).square().sum()
        )
        denominator = torch.sqrt(
            analytic_mean.square().sum() + analytic_log_std.square().sum()
        ).clamp_min(EPS)
        errors.append(float((numerator / denominator).cpu()))
    return float(max(errors)) if errors else float("nan")


def create_gradient_probe(
    *,
    policy: SquashedGaussianPolicy,
    obs: np.ndarray,
    actions: np.ndarray,
    advantages: np.ndarray,
    near_indices: np.ndarray,
    far_indices: np.ndarray,
    population_indices: np.ndarray,
    max_gradient_pairs: int,
    distance_bins: int,
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    def numpy_components(indices: np.ndarray) -> dict[str, np.ndarray]:
        with torch.no_grad():
            values = policy.score_components(
                tensor(obs[indices], device), tensor(actions[indices], device)
            )
        keys = (
            "radius",
            "mean_score_norm",
            "raw_log_scale_score_norm",
            "corrected_q_xi",
            "joint_output_score_norm",
            "log_scale_to_mean_ratio",
            "raw_action_distance",
            "pre_squash_distance",
        )
        return {key: values[key].detach().cpu().numpy() for key in keys}

    near = numpy_components(near_indices)
    far = numpy_components(far_indices)
    population = numpy_components(population_indices)
    n_grad = min(len(near_indices), max_gradient_pairs)
    near_grad = per_sample_gradient_norm(
        policy, obs, actions, advantages, near_indices[:n_grad], device
    )
    far_grad = per_sample_gradient_norm(
        policy, obs, actions, advantages, far_indices[:n_grad], device
    )
    rows: list[dict[str, Any]] = []
    for pair_id, (near_idx, far_idx) in enumerate(zip(near_indices, far_indices)):
        row: dict[str, Any] = {
            "pair_id": pair_id,
            "near_index": int(near_idx),
            "far_index": int(far_idx),
            "near_advantage": float(advantages[near_idx]),
            "far_advantage": float(advantages[far_idx]),
            "near_abs_advantage": float(abs(advantages[near_idx])),
            "far_abs_advantage": float(abs(advantages[far_idx])),
        }
        for key in near:
            row[f"near_{key}"] = float(near[key][pair_id])
            row[f"far_{key}"] = float(far[key][pair_id])
        if pair_id < n_grad:
            row["near_full_parameter_gradient_norm"] = float(near_grad[pair_id])
            row["far_full_parameter_gradient_norm"] = float(far_grad[pair_id])
        rows.append(row)
    write_csv(output_dir / "matched_near_far_components.csv", rows)

    radius = population["radius"]
    edges = np.unique(np.quantile(radius, np.linspace(0.0, 1.0, distance_bins + 1)))
    bin_rows: list[dict[str, Any]] = []
    grad_indices = np.concatenate([near_indices[:n_grad], far_indices[:n_grad]])
    grad_values = np.concatenate([near_grad, far_grad])
    with torch.no_grad():
        grad_radius = policy.standardized_distance(
            tensor(obs[grad_indices], device), tensor(actions[grad_indices], device)
        ).cpu().numpy()
    for bin_id, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (radius >= left) & (radius <= right if bin_id == len(edges) - 2 else radius < right)
        if not bool(mask.any()):
            continue
        grad_mask = (grad_radius >= left) & (
            grad_radius <= right if bin_id == len(edges) - 2 else grad_radius < right
        )
        row = {
            "bin": bin_id,
            "radius_left": float(left),
            "radius_right": float(right),
            "count": int(mask.sum()),
        }
        for key, values in population.items():
            row[f"{key}_mean"] = float(np.mean(values[mask]))
            row[f"{key}_median"] = float(np.median(values[mask]))
        row["full_parameter_gradient_norm_mean"] = (
            float(np.mean(grad_values[grad_mask])) if bool(grad_mask.any()) else float("nan")
        )
        row["full_parameter_gradient_count"] = int(grad_mask.sum())
        bin_rows.append(row)
    write_csv(output_dir / "component_distance_bins.csv", bin_rows)

    autograd_error = analytic_output_autograd_relative_error(
        policy, obs, actions, np.concatenate([near_indices, far_indices]), device
    )
    qxi_slope = loglog_slope(radius, population["corrected_q_xi"])
    mean_score_slope = loglog_slope(radius, population["mean_score_norm"])
    action_dim = int(actions.shape[1])
    natural_far_threshold = math.sqrt(2.0 * action_dim)
    far_median_radius = float(np.median(far["radius"]))
    summary = {
        "pairs": len(rows),
        "gradient_pairs": n_grad,
        "abs_advantage_far_near_ratio": float(
            np.mean(np.abs(advantages[far_indices]))
            / max(np.mean(np.abs(advantages[near_indices])), EPS)
        ),
        "standardized_distance_far_near_ratio": float(
            np.mean(far["radius"]) / max(np.mean(near["radius"]), EPS)
        ),
        "mean_output_score_far_near_ratio": float(
            np.mean(far["mean_score_norm"])
            / max(np.mean(near["mean_score_norm"]), EPS)
        ),
        "raw_log_scale_output_score_far_near_ratio": float(
            np.mean(far["raw_log_scale_score_norm"])
            / max(np.mean(near["raw_log_scale_score_norm"]), EPS)
        ),
        "corrected_q_xi_far_near_ratio": float(
            np.mean(far["corrected_q_xi"])
            / max(np.mean(near["corrected_q_xi"]), EPS)
        ),
        "joint_output_score_far_near_ratio": float(
            np.mean(far["joint_output_score_norm"])
            / max(np.mean(near["joint_output_score_norm"]), EPS)
        ),
        "log_scale_to_mean_far_near_ratio": float(
            np.mean(far["log_scale_to_mean_ratio"])
            / max(np.mean(near["log_scale_to_mean_ratio"]), EPS)
        ),
        "full_parameter_gradient_far_near_ratio": float(
            np.mean(far_grad) / max(np.mean(near_grad), EPS)
        ),
        "mean_score_loglog_slope_vs_radius": mean_score_slope,
        "corrected_q_xi_loglog_slope_vs_radius": qxi_slope,
        "analytic_autograd_relative_error_max": autograd_error,
        "natural_far_threshold_sqrt_2d": natural_far_threshold,
        "far_median_radius": far_median_radius,
        "natural_far_field_present": bool(far_median_radius >= natural_far_threshold),
    }
    atomic_write_json(output_dir / "gradient_probe_summary.json", summary)
    return summary


def aggregate_negative_gradient_norm(
    policy: SquashedGaussianPolicy,
    obs: np.ndarray,
    actions: np.ndarray,
    advantages: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
) -> float:
    obs_t = tensor(obs[indices], device)
    action_t = tensor(actions[indices], device)
    advantage_t = tensor(advantages[indices], device)
    loss = -(advantage_t * policy.log_prob(obs_t, action_t)).mean()
    return full_gradient_norm(loss, policy.parameters())


def resolve_global_scale(
    *,
    policy: SquashedGaussianPolicy,
    obs: np.ndarray,
    actions: np.ndarray,
    advantages: np.ndarray,
    negative_indices: np.ndarray,
    far_threshold: float,
    far_cap_score: float,
    audit_size: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    """Match the initial per-sample negative-influence budget of Far-cap."""
    rng = np.random.default_rng(seed + 991)
    chosen = rng.choice(
        negative_indices,
        size=min(audit_size, len(negative_indices)),
        replace=False,
    )
    with torch.no_grad():
        components = policy.score_components(
            tensor(obs[chosen], device), tensor(actions[chosen], device)
        )
        distances = components["radius"].cpu().numpy()
        joint_scores = components["joint_output_score_norm"].cpu().numpy()
    per_sample = per_sample_gradient_norm(
        policy, obs, actions, advantages, chosen, device
    )
    far_mask = distances > far_threshold
    cap_factor = np.ones_like(per_sample)
    cap_factor[far_mask] = np.minimum(
        1.0, far_cap_score / np.maximum(joint_scores[far_mask], EPS)
    )
    all_budget = float(np.sum(per_sample))
    far_cap_budget = float(np.sum(per_sample * cap_factor))
    scale = far_cap_budget / max(all_budget, EPS)
    return {
        "audit_samples": len(chosen),
        "far_samples": int(np.sum(far_mask)),
        "all_negative_per_sample_gradient_norm_sum": all_budget,
        "far_cap_per_sample_gradient_norm_sum": far_cap_budget,
        "far_cap_score": far_cap_score,
        "global_scale": float(np.clip(scale, 0.0, 1.0)),
        "matching_rule": (
            "match initial sum of per-sample full-parameter negative-gradient norms "
            "retained by detached Far-cap; no aggregate-vector cancellation"
        ),
    }


# ---------------------------------------------------------------------------
# Per-seed and full run orchestration
# ---------------------------------------------------------------------------


def copy_policy(policy: SquashedGaussianPolicy, config: E7Config, obs_dim: int, action_dim: int, device: torch.device) -> SquashedGaussianPolicy:
    clone = SquashedGaussianPolicy(
        obs_dim,
        action_dim,
        config.hidden_sizes,
        config.log_std_min,
        config.log_std_max,
        config.action_clip_epsilon,
        config.activation,
        config.init_scheme,
        config.init_gain,
    ).to(device)
    clone.load_state_dict(policy.state_dict())
    return clone


def run_seed(
    *,
    seed: int,
    data: OfflineData,
    canonical: CanonicalCriticContext,
    config: E7Config,
    mode: ModeConfig,
    device: torch.device,
    seed_dir: Path,
    heartbeat: Callable[[str, int], None] | None = None,
    formal_mode: bool = False,
) -> dict[str, Any]:
    seed_everything(seed)
    seed_dir.mkdir(parents=True, exist_ok=True)
    split = canonical.split
    obs_norm = canonical.obs_norm
    advantages = canonical.advantages
    rollout_required = (
        config.formal_rollout_required if formal_mode else config.pilot_rollout_required
    )

    def rollout_evaluator(
        policy: SquashedGaussianPolicy, step: int, stage: str
    ) -> dict[str, Any]:
        diagnostics = seed_dir / "rollouts" / stage / f"step_{step:08d}.json"
        registered_final_step = (
            mode.positive_max_steps
            if stage == "positive_only"
            else mode.branch_max_steps
        )
        episodes = (
            mode.final_rollout_episodes
            if step >= registered_final_step
            else mode.rollout_episodes
        )
        return evaluate_d4rl_rollouts(
            policy=policy,
            obs_norm=obs_norm,
            backend=config.env_backend,
            dataset_id=config.rollout_dataset_id,
            env_id=config.env_id,
            episodes=episodes,
            seed=seed * 100000 + step,
            device=device,
            normalized_score_percent=config.normalized_score_percent,
            reference_min_score=config.normalized_score_reference_min,
            reference_max_score=config.normalized_score_reference_max,
            required=rollout_required,
            diagnostics_path=diagnostics,
        )

    obs = obs_norm.transform(data.observations)
    train_indices = split["train"]
    positive_train = train_indices[advantages[train_indices] > 0]
    negative_train = train_indices[advantages[train_indices] < 0]
    if len(positive_train) < 10 or len(negative_train) < 10:
        raise RuntimeError("Frozen advantages do not yield enough positive and negative samples")
    rng = np.random.default_rng(seed + 321)
    validation_positive = split["validation"][advantages[split["validation"]] > 0]
    validation_negative = split["validation"][advantages[split["validation"]] < 0]
    if len(validation_positive) == 0 or len(validation_negative) == 0:
        raise RuntimeError("Validation split must contain both positive and negative frozen advantages")
    half = max(1, mode.audit_sample_size // 2)
    audit_positive = rng.choice(
        validation_positive, size=min(half, len(validation_positive)), replace=False
    )
    audit_negative = rng.choice(
        validation_negative, size=min(half, len(validation_negative)), replace=False
    )
    audit_indices = np.concatenate([audit_positive, audit_negative]).astype(np.int64)
    rng.shuffle(audit_indices)
    fixed_negative_indices = rng.choice(
        negative_train,
        size=min(mode.audit_sample_size, len(negative_train)),
        replace=False,
    )
    base_policy = SquashedGaussianPolicy(
        obs.shape[1],
        data.actions.shape[1],
        config.hidden_sizes,
        config.log_std_min,
        config.log_std_max,
        config.action_clip_epsilon,
        config.activation,
        config.init_scheme,
        config.init_gain,
    ).to(device)
    positive_policy, positive_audit = train_actor_stage(
        policy=base_policy,
        method="positive_only",
        obs=obs,
        actions=data.actions,
        advantages=advantages,
        train_indices=positive_train,
        audit_indices=audit_indices,
        fixed_negative_indices=fixed_negative_indices,
        config=config,
        min_steps=mode.positive_min_steps,
        max_steps=mode.positive_max_steps,
        eval_interval=mode.actor_eval_interval,
        seed=seed + 500000,
        device=device,
        output_dir=seed_dir / "positive_only_initialization",
        rollout_evaluator=rollout_evaluator,
        rollout_eval_interval=mode.rollout_eval_interval,
        heartbeat=heartbeat,
    )
    if formal_mode and (
        positive_audit["numerical_nonfinite"]
        or not positive_audit["fixed_budget_completed"]
    ):
        raise RuntimeError(
            "Formal E7 gate failed: Positive-only did not complete its fixed "
            "optimizer-step budget with finite parameters"
        )

    with torch.no_grad():
        all_negative_distances = np.full(data.size, np.nan, dtype=np.float32)
        for offset in range(0, len(negative_train), 65536):
            idx = negative_train[offset : offset + 65536]
            all_negative_distances[idx] = positive_policy.standardized_distance(
                tensor(obs[idx], device), tensor(data.actions[idx], device)
            ).cpu().numpy()
    near_idx, far_idx, matching_summary = match_near_far_indices(
        advantages,
        all_negative_distances,
        negative_train,
        config.near_quantile,
        config.far_quantile,
        config.advantage_bins,
        mode.matched_pairs,
        config.advantage_match_relative_tolerance,
        seed,
    )
    pair_rows = [
        {
            "pair_id": i,
            "near_index": int(n),
            "far_index": int(f),
            "near_abs_advantage": float(abs(advantages[n])),
            "far_abs_advantage": float(abs(advantages[f])),
            "near_distance": float(all_negative_distances[n]),
            "far_distance": float(all_negative_distances[f]),
        }
        for i, (n, f) in enumerate(zip(near_idx, far_idx))
    ]
    probe_dir = seed_dir / "probes"
    probe_dir.mkdir(exist_ok=True)
    write_csv(probe_dir / "matching_pairs.csv", pair_rows)
    atomic_write_json(probe_dir / "matching_summary.json", matching_summary)
    gradient_summary = create_gradient_probe(
        policy=positive_policy,
        obs=obs,
        actions=data.actions,
        advantages=advantages,
        near_indices=near_idx,
        far_indices=far_idx,
        population_indices=fixed_negative_indices,
        max_gradient_pairs=min(config.gradient_probe_pairs, len(near_idx)),
        distance_bins=config.distance_bins,
        device=device,
        output_dir=probe_dir,
    )
    far_threshold = float(
        (matching_summary["near_cut"] + matching_summary["far_cut"]) / 2.0
    )
    near_negative_pool = negative_train[
        all_negative_distances[negative_train] <= matching_summary["near_cut"]
    ]
    if len(near_negative_pool) == 0:
        raise RuntimeError("No near-negative samples available to define Far-cap")
    with torch.no_grad():
        near_joint_scores = positive_policy.output_score_norm(
            tensor(obs[near_negative_pool], device),
            tensor(data.actions[near_negative_pool], device),
        ).cpu().numpy()
    far_cap_score = float(
        np.quantile(near_joint_scores, config.far_cap_reference_quantile)
    )
    atomic_write_json(
        probe_dir / "far_cap_definition.json",
        {
            "reference_population": "near_negative_pool",
            "reference_quantile": config.far_cap_reference_quantile,
            "far_cap_score": far_cap_score,
            "far_threshold": far_threshold,
            "detached": True,
        },
    )
    budget = resolve_global_scale(
        policy=positive_policy,
        obs=obs,
        actions=data.actions,
        advantages=advantages,
        negative_indices=negative_train,
        far_threshold=far_threshold,
        far_cap_score=far_cap_score,
        audit_size=config.global_budget_audit_size,
        seed=seed,
        device=device,
    )
    atomic_write_json(probe_dir / "global_budget_match.json", budget)
    branch_audits: dict[str, Any] = {}
    for method in METHODS:
        policy = copy_policy(
            positive_policy, config, obs.shape[1], data.actions.shape[1], device
        )
        _, audit = train_actor_stage(
            policy=policy,
            method=method,
            obs=obs,
            actions=data.actions,
            advantages=advantages,
            train_indices=train_indices,
            audit_indices=audit_indices,
            fixed_negative_indices=fixed_negative_indices,
            config=config,
            min_steps=mode.branch_min_steps,
            max_steps=mode.branch_max_steps,
            eval_interval=mode.actor_eval_interval,
            seed=seed,
            device=device,
            output_dir=seed_dir / "methods" / method,
            far_threshold=far_threshold,
            global_scale=float(budget["global_scale"]),
            far_cap_score=far_cap_score,
            rollout_evaluator=rollout_evaluator,
            rollout_eval_interval=mode.rollout_eval_interval,
            heartbeat=heartbeat,
        )
        branch_audits[method] = audit

    slope = float(gradient_summary["corrected_q_xi_loglog_slope_vs_radius"])
    signed_audit = branch_audits["signed"]
    control_outcomes: dict[str, dict[str, bool]] = {}
    for method in ("far_zero", "far_cap", "dynamic_budget_matched_global"):
        control = branch_audits[method]
        score_reduced = float(
            control["final_metrics"]["phantom_joint_output_score_mean"]
        ) < 0.95 * float(
            signed_audit["final_metrics"]["phantom_joint_output_score_mean"]
        )
        support_rescued = bool(signed_audit["support_boundary_event"]) and not bool(
            control["support_boundary_event"]
        )
        signed_task_collapse = signed_audit["task_performance_collapse"] is True
        control_task_collapse = control["task_performance_collapse"] is True
        task_rescued = signed_task_collapse and not control_task_collapse
        finite_terminal_rescued = (
            signed_audit["state"] != "finite_terminal"
            and control["state"] == "finite_terminal"
        )
        control_outcomes[method] = {
            "diagnostic_score_mitigation": bool(score_reduced),
            "support_boundary_rescue": bool(support_rescued),
            "task_performance_rescue": bool(task_rescued),
            "finite_terminal_rescue": bool(finite_terminal_rescued),
            "any_mitigation_observed": bool(
                score_reduced
                or support_rescued
                or task_rescued
                or finite_terminal_rescued
            ),
        }
    terminal_records_complete = all(
        bool(audit.get("terminal_audit_complete")) for audit in branch_audits.values()
    )
    terminal_classification_complete = all(
        bool(audit.get("explicit_terminal_classification"))
        for audit in branch_audits.values()
    )
    rollout_available = all(
        audit.get("task_performance_status") == "available"
        for audit in branch_audits.values()
    )
    seed_gate = {
        "natural_far_field_present": bool(gradient_summary["natural_far_field_present"]),
        "corrected_quadratic_branch_empirically_active": bool(
            math.isfinite(slope)
            and abs(slope - config.qxi_slope_target) <= config.qxi_slope_tolerance
            and float(gradient_summary["analytic_autograd_relative_error_max"])
            <= config.analytic_autograd_error_max
        ),
        "log_scale_relative_dominance_observed": bool(
            float(gradient_summary["log_scale_to_mean_far_near_ratio"])
            >= config.log_scale_to_mean_ratio_min
        ),
        "measurable_full_parameter_contribution": bool(
            float(gradient_summary["full_parameter_gradient_far_near_ratio"])
            >= config.full_parameter_ratio_min
        ),
        "control_outcomes": control_outcomes,
        "targeted_control_outcomes_reported": True,
        "terminal_audit_records_complete": terminal_records_complete,
        "terminal_state_classification_complete": terminal_classification_complete,
        "rollout_available_for_all_methods": rollout_available,
    }
    core_keys = (
        "natural_far_field_present",
        "corrected_quadratic_branch_empirically_active",
        "measurable_full_parameter_contribution",
    )
    seed_gate["all_mechanism_subchecks_passed"] = all(
        bool(seed_gate[key]) for key in core_keys
    )
    summary = {
        "seed": seed,
        "canonical_critic_artifact": {
            "root": str(canonical.root),
            "reused": canonical.reused,
            "identity": canonical.artifact_manifest["identity"],
            "critic_training_count": canonical.artifact_manifest["critic_training_count"],
            "shared_across_all_actor_seeds": True,
        },
        "critic": canonical.critic_audit,
        "advantage": canonical.advantage_manifest,
        "positive_only_initialization": positive_audit,
        "matching": matching_summary,
        "gradient_probe": gradient_summary,
        "initial_global_budget_diagnostic": budget,
        "global_budget": budget,
        "mechanism_subchecks": seed_gate,
        # Compatibility alias for pre-v4 result readers.  Root-level audit semantics
        # no longer treat this alias as a formal scientific gate in pilot runs.
        "independent_validation_gate": seed_gate,
        "methods": branch_audits,
    }
    atomic_write_json(seed_dir / "seed_summary.json", summary)
    return summary


def flatten_seed_summary(summary: dict[str, Any]) -> dict[str, Any]:
    probe = summary["gradient_probe"]
    row: dict[str, Any] = {
        "seed": summary["seed"],
        "canonical_critic_seed": summary["canonical_critic_artifact"]["identity"][
            "canonical_critic_seed"
        ],
        "critic_test_r2": summary["critic"]["selected_checkpoint_metrics"]["test_r2"],
        "critic_test_pearson": summary["critic"]["selected_checkpoint_metrics"][
            "test_pearson"
        ],
        "critic_fixed_budget_completed": summary["critic"][
            "fixed_budget_completed"
        ],
        "critic_optimization_terminal": summary["critic"]["optimization_terminal"],
        "critic_accepted_for_frozen_advantage": summary["critic"][
            "critic_accepted_for_frozen_advantage"
        ],
        "positive_fraction": summary["advantage"]["positive_fraction"],
        "matched_pairs": summary["matching"]["pairs"],
        "abs_advantage_far_near_ratio": probe["abs_advantage_far_near_ratio"],
        "standardized_distance_far_near_ratio": probe[
            "standardized_distance_far_near_ratio"
        ],
        "mean_output_score_far_near_ratio": probe[
            "mean_output_score_far_near_ratio"
        ],
        "raw_log_scale_output_score_far_near_ratio": probe[
            "raw_log_scale_output_score_far_near_ratio"
        ],
        "corrected_q_xi_far_near_ratio": probe["corrected_q_xi_far_near_ratio"],
        "joint_output_score_far_near_ratio": probe[
            "joint_output_score_far_near_ratio"
        ],
        "log_scale_to_mean_far_near_ratio": probe[
            "log_scale_to_mean_far_near_ratio"
        ],
        "full_parameter_gradient_far_near_ratio": probe[
            "full_parameter_gradient_far_near_ratio"
        ],
        "mean_score_loglog_slope_vs_radius": probe[
            "mean_score_loglog_slope_vs_radius"
        ],
        "corrected_q_xi_loglog_slope_vs_radius": probe[
            "corrected_q_xi_loglog_slope_vs_radius"
        ],
        "analytic_autograd_relative_error_max": probe[
            "analytic_autograd_relative_error_max"
        ],
        "natural_far_field_present": probe["natural_far_field_present"],
        "global_scale": summary["global_budget"]["global_scale"],
        "mechanism_subchecks_passed": summary["mechanism_subchecks"][
            "all_mechanism_subchecks_passed"
        ],
        "terminal_audit_records_complete": summary["mechanism_subchecks"][
            "terminal_audit_records_complete"
        ],
        "terminal_state_classification_complete": summary["mechanism_subchecks"][
            "terminal_state_classification_complete"
        ],
        "rollout_available_for_all_methods": summary["mechanism_subchecks"][
            "rollout_available_for_all_methods"
        ],
        "positive_only_fixed_budget_completed": summary[
            "positive_only_initialization"
        ]["fixed_budget_completed"],
        "positive_only_terminal_state": summary["positive_only_initialization"]["state"],
        "positive_only_terminal_audit_complete": summary["positive_only_initialization"][
            "terminal_audit_complete"
        ],
        "positive_only_task_performance_status": summary["positive_only_initialization"][
            "task_performance_status"
        ],
    }
    for method, audit in summary["methods"].items():
        final = audit["final_metrics"]
        row[f"{method}_fixed_budget_completed"] = audit["fixed_budget_completed"]
        row[f"{method}_terminal_state"] = audit["state"]
        row[f"{method}_terminal_audit_complete"] = audit["terminal_audit_complete"]
        row[f"{method}_task_performance_status"] = audit["task_performance_status"]
        row[f"{method}_task_performance_collapse"] = audit[
            "task_performance_collapse"
        ]
        row[f"{method}_normalized_return_available"] = audit[
            "normalized_return_available"
        ]
        row[f"{method}_support_boundary_event"] = audit["support_boundary_event"]
        row[f"{method}_nan_inf_event"] = audit["numerical_nonfinite"]
        row[f"{method}_normalized_return"] = audit["final_normalized_return"]
        row[f"{method}_positive_nll"] = final["positive_nll"]
        row[f"{method}_mean_boundary_fraction"] = final["mean_boundary_fraction"]
        row[f"{method}_sigma_mean"] = final["sigma_mean"]
        row[f"{method}_phantom_joint_output_score_mean"] = final[
            "phantom_joint_output_score_mean"
        ]
        row[f"{method}_phantom_log_scale_to_mean_ratio"] = final[
            "phantom_log_scale_to_mean_ratio"
        ]
    return row


def aggregate_seed_summaries(summaries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not summaries:
        return {"seeds_completed": 0}
    rows = [flatten_seed_summary(item) for item in summaries]
    numeric_keys = [
        key
        for key, value in rows[0].items()
        if isinstance(value, (int, float, np.integer, np.floating, bool))
        and key not in {"seed", "canonical_critic_seed"}
    ]
    aggregate: dict[str, Any] = {
        "seeds_completed": len(rows),
        "seed_ids": [row["seed"] for row in rows],
        "canonical_critic_seed": rows[0]["canonical_critic_seed"],
        "one_shared_canonical_critic": len(
            {row["canonical_critic_seed"] for row in rows}
        )
        == 1,
    }
    for key in numeric_keys:
        values = np.asarray(
            [float(row[key]) for row in rows if row[key] is not None],
            dtype=np.float64,
        )
        finite = values[np.isfinite(values)]
        if len(finite):
            aggregate[key] = {
                "mean": float(np.mean(finite)),
                "std": float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0,
                "min": float(np.min(finite)),
                "max": float(np.max(finite)),
            }
    aggregate["terminal_state_counts"] = {
        method: {
            state: sum(
                1
                for summary in summaries
                if summary["methods"][method]["state"] == state
            )
            for state in sorted(
                {summary["methods"][method]["state"] for summary in summaries}
            )
        }
        for method in METHODS
    }
    aggregate["reporting_separation"] = {
        method: {
            "task_performance_available_count": sum(
                summary["methods"][method]["task_performance_status"] == "available"
                for summary in summaries
            ),
            "task_performance_unavailable_count": sum(
                summary["methods"][method]["task_performance_status"] != "available"
                for summary in summaries
            ),
            "task_performance_collapse_count": sum(
                summary["methods"][method]["task_performance_collapse"] is True
                for summary in summaries
            ),
            "support_or_variance_boundary_count": sum(
                bool(summary["methods"][method]["support_boundary_event"])
                for summary in summaries
            ),
            "nan_inf_numerical_count": sum(
                bool(summary["methods"][method]["numerical_nonfinite"])
                for summary in summaries
            ),
        }
        for method in METHODS
    }
    gates = [summary["mechanism_subchecks"] for summary in summaries]
    gate_names = (
        "natural_far_field_present",
        "corrected_quadratic_branch_empirically_active",
        "measurable_full_parameter_contribution",
        "log_scale_relative_dominance_observed",
        "targeted_control_outcomes_reported",
        "terminal_audit_records_complete",
        "terminal_state_classification_complete",
        "rollout_available_for_all_methods",
        "all_mechanism_subchecks_passed",
    )
    aggregate["mechanism_subcheck_counts"] = {
        name: int(sum(bool(gate[name]) for gate in gates)) for name in gate_names
    }
    # Compatibility alias for pre-v4 readers.  The root terminal audit carries the
    # authoritative pilot/formal semantics.
    aggregate["independent_validation_gate_counts"] = aggregate[
        "mechanism_subcheck_counts"
    ]
    aggregate["method_ranking_claim_allowed"] = False
    return aggregate


def build_terminal_audit(
    *,
    summaries: Sequence[dict[str, Any]],
    mode_name: str,
    expected_seed_count: int,
    canonical: CanonicalCriticContext,
    rollout_preflight: dict[str, Any],
    rollout_required: bool,
) -> dict[str, Any]:
    aggregate = aggregate_seed_summaries(summaries)
    completed = len(summaries)
    counts = aggregate.get("mechanism_subcheck_counts", {})
    seed_count_complete = completed == expected_seed_count and expected_seed_count > 0
    mechanism_subchecks_passed = bool(
        completed > 0
        and all(
            int(counts.get(name, 0)) == completed
            for name in (
                "natural_far_field_present",
                "corrected_quadratic_branch_empirically_active",
                "measurable_full_parameter_contribution",
            )
        )
    )
    terminal_audit_records_complete = bool(
        completed > 0
        and int(counts.get("terminal_audit_records_complete", 0)) == completed
    )
    terminal_state_classification_complete = bool(
        completed > 0
        and int(counts.get("terminal_state_classification_complete", 0)) == completed
    )
    paired_seed_evidence_complete = bool(
        mode_name == "formal" and expected_seed_count > 1 and seed_count_complete
    )
    positive_only_fixed_budget_all_seeds = bool(
        completed > 0
        and all(
            summary["positive_only_initialization"]["fixed_budget_completed"]
            and summary["positive_only_initialization"]["terminal_audit_complete"]
            for summary in summaries
        )
    )
    all_actor_fixed_budgets_completed = bool(
        completed > 0
        and all(
            all(
                audit["fixed_budget_completed"]
                for audit in summary["methods"].values()
            )
            for summary in summaries
        )
    )
    positive_only_terminal_all_seeds = bool(
        completed > 0
        and all(
            summary["positive_only_initialization"]["state"] == "finite_terminal"
            for summary in summaries
        )
    )
    rollout_available_all_required_checkpoints = bool(
        rollout_preflight.get("status") == "passed"
        and (
            not rollout_required
            or all(
                summary["positive_only_initialization"]["task_performance_status"]
                == "available"
                and all(
                    audit["task_performance_status"] == "available"
                    for audit in summary["methods"].values()
                )
                for summary in summaries
            )
        )
    )
    critic_fixed_budget_completed = bool(
        canonical.critic_audit.get("fixed_budget_completed")
    )
    critic_artifact_terminal = bool(
        canonical.critic_audit.get("optimization_terminal")
    )
    critic_artifact_accepted = bool(
        canonical.critic_audit.get("critic_accepted_for_frozen_advantage")
    )
    critic_quality_audit_passed = bool(
        canonical.critic_audit.get("critic_quality_audit_passed")
    )
    engineering_pipeline_complete = bool(
        seed_count_complete
        and canonical.artifact_manifest.get("complete")
        and rollout_preflight.get("status") == "passed"
        and critic_fixed_budget_completed
        and positive_only_fixed_budget_all_seeds
        and all_actor_fixed_budgets_completed
        and terminal_audit_records_complete
    )
    formal_evidence_prerequisites_complete = bool(
        mode_name == "formal"
        and engineering_pipeline_complete
        and critic_artifact_accepted
        and positive_only_fixed_budget_all_seeds
        and all_actor_fixed_budgets_completed
        and paired_seed_evidence_complete
        and rollout_available_all_required_checkpoints
        and terminal_state_classification_complete
        and mechanism_subchecks_passed
    )
    return {
        "experiment_id": EXPERIMENT_ID,
        "mode": mode_name,
        "seeds_expected": expected_seed_count,
        "seeds_audited": completed,
        "scientific_status": (
            "pilot" if mode_name == "pilot" else "raw_complete_pending_review"
        ),
        "engineering_pipeline_complete": engineering_pipeline_complete,
        "mechanism_subchecks_passed_for_completed_seeds": mechanism_subchecks_passed,
        "critic_fixed_budget_completed": critic_fixed_budget_completed,
        "critic_artifact_terminal": critic_artifact_terminal,
        "critic_optimization_terminal_role": "diagnostic_only",
        "critic_artifact_accepted_for_frozen_advantage": critic_artifact_accepted,
        "critic_quality_audit_passed_diagnostic": critic_quality_audit_passed,
        "critic_artifact_shared_across_all_actor_seeds": True,
        "positive_only_fixed_budget_all_seeds": positive_only_fixed_budget_all_seeds,
        "all_actor_fixed_budgets_completed": all_actor_fixed_budgets_completed,
        "positive_only_terminal_all_seeds_diagnostic": positive_only_terminal_all_seeds,
        "terminal_audit_records_complete": terminal_audit_records_complete,
        "terminal_state_classification_complete": terminal_state_classification_complete,
        "rollout_preflight_passed": rollout_preflight.get("status") == "passed",
        "rollout_available_all_required_checkpoints": rollout_available_all_required_checkpoints,
        "paired_seed_evidence_complete": paired_seed_evidence_complete,
        "formal_evidence_prerequisites_complete": formal_evidence_prerequisites_complete,
        "formal_scientific_gate_passed": False,
        "formal_scientific_gate_reason": "post_run_scientific_review_required",
        "independent_validation_gate_all_seeds": formal_evidence_prerequisites_complete,
        "independent_validation_gate_all_seeds_deprecated_semantics": (
            "Compatibility field. It is false for pilot runs and only mirrors "
            "formal_evidence_prerequisites_complete; it is not a final scientific claim."
        ),
        "gate_counts": counts,
        "reporting_separation": aggregate.get("reporting_separation", {}),
        "method_ranking_claim_allowed": False,
        "formal_claim_requires_post_run_review": True,
    }


def validate_formal_preflight(
    *,
    mode_name: str,
    allow_dirty: bool,
    repo_root: Path,
    git_state: dict[str, Any],
    dataset_path: Path,
    config: E7Config,
) -> dict[str, Any]:
    if dataset_path.name != config.dataset_basename:
        raise ValueError(
            f"Dataset basename mismatch: expected {config.dataset_basename}, got {dataset_path.name}"
        )
    digest = sha256_file(dataset_path)
    if digest != config.dataset_sha256:
        raise ValueError(
            f"Dataset SHA-256 mismatch: expected {config.dataset_sha256}, got {digest}"
        )
    if mode_name == "formal":
        if not git_state.get("available"):
            raise RuntimeError("Formal mode requires a Git checkout")
        if git_state.get("status_porcelain"):
            raise RuntimeError("Formal mode requires a clean Git worktree")
    elif git_state.get("status_porcelain") and not allow_dirty:
        raise RuntimeError("Dirty pilot requires --allow-dirty")
    return {
        "path": str(dataset_path.resolve()),
        "basename": dataset_path.name,
        "sha256": digest,
        "size_bytes": dataset_path.stat().st_size,
    }


def run_experiment(args: argparse.Namespace) -> int:
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    mode = config.formal if args.mode == "formal" else config.pilot
    dataset_path = Path(args.dataset_path).expanduser().resolve()
    work_dir = Path(args.work_dir).expanduser().resolve()
    repo_root = Path(args.repo_root).expanduser().resolve()
    reuse_critic_root = (
        Path(args.critic_artifact).expanduser().resolve()
        if args.critic_artifact
        else None
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    global _EVENT_LOG_PATH
    _EVENT_LOG_PATH = work_dir / "events.jsonl"
    run_id = args.run_id or (
        f"{args.mode}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    git_start = collect_git_state(repo_root)
    atomic_write_json(work_dir / "GIT_STATE_START.json", git_start)
    atomic_write_json(work_dir / "ENVIRONMENT.json", environment_manifest())
    dataset_manifest = validate_formal_preflight(
        mode_name=args.mode,
        allow_dirty=args.allow_dirty,
        repo_root=repo_root,
        git_state=git_start,
        dataset_path=dataset_path,
        config=config,
    )
    atomic_write_json(work_dir / "DATASET_MANIFEST.json", dataset_manifest)
    shutil.copy2(config_path, work_dir / "resolved_config.yaml")
    run_manifest: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "runner_version": RUNNER_VERSION,
        "run_id": run_id,
        "mode": args.mode,
        "started_utc": utc_now(),
        "state": "running",
        "repo_root": str(repo_root),
        "run_commit": git_start.get("head"),
        "dataset": dataset_manifest,
        "seeds": list(mode.seeds),
        "canonical_critic_seed": mode.canonical_critic_seed,
        "canonical_critic_reuse_request": (
            str(reuse_critic_root) if reuse_critic_root else None
        ),
        "device_request": args.device,
        "claim_boundary": (
            "External mechanism validation with diagnostic normalized-return trajectories; "
            "not a standard offline-RL method-ranking table."
        ),
    }
    atomic_write_json(work_dir / "scientific_run_manifest.json", run_manifest)
    summaries: list[dict[str, Any]] = []
    canonical: CanonicalCriticContext | None = None
    rollout_preflight: dict[str, Any] = {"status": "not_started"}

    def root_heartbeat(stage: str, step: int) -> None:
        atomic_write_json(
            work_dir / "scientific_heartbeat.json",
            {
                "utc": utc_now(),
                "run_id": run_id,
                "state": "running",
                "stage": stage,
                "step": step,
                "seeds_completed": len(summaries),
            },
        )

    try:
        device = choose_device(args.device)
        data = load_hopper_hdf5(dataset_path, mode.max_transitions)
        runtime_summary = {
            "transitions_loaded": data.size,
            "episodes_loaded": int(np.max(data.episode_ids) + 1),
            "observation_dim": data.observations.shape[1],
            "action_dim": data.actions.shape[1],
            "reward_mean": float(np.mean(data.rewards)),
            "reward_std": float(np.std(data.rewards)),
            "terminal_fraction": float(np.mean(data.terminals)),
            "timeout_fraction": float(np.mean(data.timeouts)),
        }
        atomic_write_json(work_dir / "DATASET_RUNTIME_SUMMARY.json", runtime_summary)
        rollout_required = (
            config.formal_rollout_required
            if args.mode == "formal"
            else config.pilot_rollout_required
        )
        root_heartbeat("rollout_preflight", 0)
        rollout_preflight_dir = work_dir / "rollout_preflight"
        try:
            rollout_preflight = preflight_rollout_environment(
                backend=config.env_backend,
                dataset_id=config.rollout_dataset_id,
                env_id=config.env_id,
                expected_observation_dim=data.observations.shape[1],
                expected_action_dim=data.actions.shape[1],
                seed=mode.canonical_critic_seed,
                max_steps=config.rollout_preflight_max_steps,
                normalized_score_percent=config.normalized_score_percent,
                reference_min_score=config.normalized_score_reference_min,
                reference_max_score=config.normalized_score_reference_max,
                output_dir=rollout_preflight_dir,
                required=rollout_required,
                process_isolated=config.process_isolated_preflight,
                timeout_seconds=config.rollout_preflight_timeout_seconds,
            )
        except Exception:
            failure_path = rollout_preflight_dir / "rollout_preflight.json"
            if failure_path.is_file():
                rollout_preflight = json.loads(failure_path.read_text())
                atomic_write_json(
                    work_dir / "ROLLOUT_PREFLIGHT.json", rollout_preflight
                )
            raise
        atomic_write_json(work_dir / "ROLLOUT_PREFLIGHT.json", rollout_preflight)
        root_heartbeat("canonical_critic_prepare", 0)
        canonical = prepare_canonical_critic_context(
            data=data,
            config=config,
            mode=mode,
            mode_name=args.mode,
            config_path=config_path,
            dataset_manifest=dataset_manifest,
            device=device,
            artifact_root=work_dir / "canonical_critic",
            reuse_root=reuse_critic_root,
            heartbeat=root_heartbeat,
        )
        canonical_reference = {
            "root": str(canonical.root),
            "reused": canonical.reused,
            "identity": canonical.artifact_manifest["identity"],
            "critic_fixed_budget_completed": canonical.critic_audit[
                "fixed_budget_completed"
            ],
            "critic_optimization_terminal": canonical.critic_audit[
                "optimization_terminal"
            ],
            "critic_accepted_for_frozen_advantage": canonical.critic_audit[
                "critic_accepted_for_frozen_advantage"
            ],
            "critic_training_count": canonical.artifact_manifest[
                "critic_training_count"
            ],
            "shared_across_all_actor_seeds": True,
            "files": canonical.artifact_manifest["files"],
        }
        atomic_write_json(
            work_dir / "CANONICAL_CRITIC_REFERENCE.json", canonical_reference
        )
        run_manifest.update(
            {
                "rollout_preflight_status": rollout_preflight["status"],
                "canonical_critic_root": str(canonical.root),
                "canonical_critic_reused": canonical.reused,
                "canonical_critic_fixed_budget_completed": canonical.critic_audit[
                    "fixed_budget_completed"
                ],
                "canonical_critic_optimization_terminal": canonical.critic_audit[
                    "optimization_terminal"
                ],
                "canonical_critic_accepted_for_frozen_advantage": canonical.critic_audit[
                    "critic_accepted_for_frozen_advantage"
                ],
            }
        )
        atomic_write_json(work_dir / "scientific_run_manifest.json", run_manifest)

        for index, seed in enumerate(mode.seeds, start=1):
            atomic_write_json(
                work_dir / "scientific_heartbeat.json",
                {
                    "utc": utc_now(),
                    "run_id": run_id,
                    "state": "running",
                    "seed": seed,
                    "seed_index": index,
                    "seed_total": len(mode.seeds),
                },
            )

            def seed_heartbeat(stage: str, step: int) -> None:
                atomic_write_json(
                    work_dir / "scientific_heartbeat.json",
                    {
                        "utc": utc_now(),
                        "run_id": run_id,
                        "state": "running",
                        "seed": seed,
                        "seed_index": index,
                        "seed_total": len(mode.seeds),
                        "stage": stage,
                        "step": step,
                    },
                )

            summary = run_seed(
                seed=seed,
                data=data,
                canonical=canonical,
                config=config,
                mode=mode,
                device=device,
                seed_dir=work_dir / "seeds" / f"seed_{seed}",
                heartbeat=seed_heartbeat,
                formal_mode=args.mode == "formal",
            )
            summaries.append(summary)
            write_csv(
                work_dir / "per_seed_summary.csv",
                [flatten_seed_summary(item) for item in summaries],
            )
            aggregate = aggregate_seed_summaries(summaries)
            aggregate["scientific_status"] = (
                "pilot_partial"
                if args.mode == "pilot"
                else "formal_partial_pending_completion"
            )
            atomic_write_json(work_dir / "aggregate_summary.json", aggregate)
            if args.mode == "formal" and (
                index % config.checkpoint_every_formal_seeds == 0
                or index == len(mode.seeds)
            ):
                run_manifest["state"] = "checkpoint_ready"
                run_manifest["seeds_completed"] = index
                atomic_write_json(work_dir / "scientific_run_manifest.json", run_manifest)

        git_end = collect_git_state(repo_root)
        atomic_write_json(work_dir / "GIT_STATE_END.json", git_end)
        provenance_compromised = bool(
            args.mode == "formal"
            and (
                git_end.get("head") != git_start.get("head")
                or git_end.get("status_porcelain")
            )
        )
        aggregate = aggregate_seed_summaries(summaries)
        aggregate["provenance_compromised"] = provenance_compromised
        aggregate["scientific_status"] = (
            "pilot" if args.mode == "pilot" else "raw_complete_pending_scientific_review"
        )
        aggregate["paired_seed_evidence_complete"] = bool(
            args.mode == "formal" and len(summaries) == len(mode.seeds) and len(mode.seeds) > 1
        )
        atomic_write_json(work_dir / "aggregate_summary.json", aggregate)
        terminal_audit = build_terminal_audit(
            summaries=summaries,
            mode_name=args.mode,
            expected_seed_count=len(mode.seeds),
            canonical=canonical,
            rollout_preflight=rollout_preflight,
            rollout_required=rollout_required,
        )
        terminal_audit["provenance_compromised"] = provenance_compromised
        atomic_write_json(work_dir / "terminal_audit.json", terminal_audit)
        complete = {
            "experiment_id": EXPERIMENT_ID,
            "run_id": run_id,
            "mode": args.mode,
            "completed_utc": utc_now(),
            "process_exit_code": 0 if not provenance_compromised else 3,
            "seeds_completed": len(summaries),
            "provenance_compromised": provenance_compromised,
            "result_status": (
                "pilot" if args.mode == "pilot" else "raw_complete_pending_review"
            ),
            "formal_result_claim": False,
            "formal_evidence_prerequisites_complete": terminal_audit[
                "formal_evidence_prerequisites_complete"
            ],
        }
        atomic_write_json(work_dir / "RUN_COMPLETE.json", complete)
        atomic_write_json(
            work_dir / "scientific_heartbeat.json",
            {
                "utc": utc_now(),
                "run_id": run_id,
                "state": "completed",
                "seeds_completed": len(summaries),
            },
        )
        run_manifest.update(
            {
                "state": (
                    "raw_complete" if not provenance_compromised else "failed_provenance"
                ),
                "completed_utc": utc_now(),
                "seeds_completed": len(summaries),
            }
        )
        atomic_write_json(work_dir / "scientific_run_manifest.json", run_manifest)
        emit_event({"event": "scientific_runner_complete", **complete})
        return int(complete["process_exit_code"])
    except Exception as exc:
        failure = {
            "experiment_id": EXPERIMENT_ID,
            "run_id": run_id,
            "mode": args.mode,
            "failed_utc": utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "seeds_completed": len(summaries),
            "rollout_preflight_status": rollout_preflight.get("status"),
            "canonical_critic_available": canonical is not None,
            "result_status": (
                "failed_pilot" if args.mode == "pilot" else "failed_formal_attempt"
            ),
        }
        atomic_write_json(work_dir / "SCIENTIFIC_RUN_FAILED.json", failure)
        atomic_write_json(
            work_dir / "scientific_heartbeat.json",
            {
                "utc": utc_now(),
                "run_id": run_id,
                "state": "failed",
                "seeds_completed": len(summaries),
            },
        )
        emit_event(
            {"event": "run_failed", "error_type": type(exc).__name__, "error": str(exc)}
        )
        run_manifest.update({"state": "failed", "failed_utc": utc_now()})
        atomic_write_json(work_dir / "scientific_run_manifest.json", run_manifest)
        raise


# ---------------------------------------------------------------------------
# Isolated Hopper p/q / refreshed-advantage / raw-IS profile
# ---------------------------------------------------------------------------

REBUTTAL_EXPERIMENT_ID = "EXT-H-E7-REBUTTAL-PQ-IS-01"
REBUTTAL_METHODS = ("positive_only", "signed", "drpo", "is")
REBUTTAL_MODES = ("frozen", "refreshed")
Q2_PAIRS = (("far_zero", "global_matched_far_zero"), ("far_cap", "global_matched_far_cap"))
Q2_METHODS = tuple(method for pair in Q2_PAIRS for method in pair)


def load_rebuttal_config(path: str | Path, smoke: bool = False) -> tuple[dict, E7Config]:
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text())
    if raw["experiment_id"] != REBUTTAL_EXPERIMENT_ID:
        raise ValueError("The rebuttal profile requires its own experiment config")
    settings = dict(raw)
    smoke_settings = settings.pop("smoke")
    if smoke:
        settings.update(smoke_settings)
    settings["smoke"] = smoke
    base_path = path.parent / raw["base_config"]
    base = load_config(base_path)
    settings["config_sha256"] = sha256_file(path)
    settings["base_config_sha256"] = sha256_file(base_path)
    settings["base_config_path"] = str(base_path.resolve())
    identity = {k: v for k, v in settings.items() if k != "base_config_path"}
    settings["identity"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()
    ).hexdigest()
    base = replace(
        base, actor_batch_size=settings["actor_batch_size"],
        critic_batch_size=settings["critic_batch_size"],
        actor_lr=settings["learning_rate"], critic_lr=settings["learning_rate"],
        max_gradient_norm=settings["actor_gradient_clip"],
    )
    return settings, base


def rebuttal_matrix(seeds: Sequence[int], methods_by_mode: dict | None = None) -> list[dict[str, Any]]:
    methods_by_mode = methods_by_mode or {"frozen": REBUTTAL_METHODS + Q2_METHODS, "refreshed": REBUTTAL_METHODS}
    return [
        {"seed": int(seed), "advantage_mode": mode, "method": method}
        for seed in seeds for mode in REBUTTAL_MODES for method in methods_by_mode[mode]
    ]


def rebuttal_jobs(settings: dict) -> list[dict[str, Any]]:
    matrix = rebuttal_matrix(settings["seeds"], settings["methods_by_advantage_mode"])
    jobs = [{**key, "job_kind": "independent", "cells": [key]} for key in matrix if key["method"] in REBUTTAL_METHODS]
    for seed in settings["seeds"]:
        for intervention, control in settings["q2"]["pairs"]:
            jobs.append({"seed": int(seed), "advantage_mode": "frozen", "method": intervention,
                         "job_kind": "q2_pair", "cells": [
                             {"seed": int(seed), "advantage_mode": "frozen", "method": method}
                             for method in (intervention, control)],
                         "checkpoint_index": f"pairs/seed_{seed}/{intervention}/checkpoint_index.json"})
    return jobs


def _rebuttal_json(value: Any) -> Any:
    """JSON-safe observations; undefined ratios never become arbitrary epsilons."""
    if isinstance(value, dict):
        return {k: _rebuttal_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_rebuttal_json(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_rebuttal_json(path: Path, value: Any) -> None:
    atomic_write_json(path, _rebuttal_json(value))


def _rebuttal_nonnegative_mean(values: np.ndarray) -> float:
    """Reduce diagnostic mass without overflowing a finite nonnegative mean."""
    if values.size == 0:
        return float("nan")
    maximum = float(np.max(values))
    if maximum == 0 or not math.isfinite(maximum):
        return maximum
    return float(np.mean(values / maximum)) * maximum


def rebuttal_mass(coefficient: np.ndarray) -> dict[str, Any]:
    coefficient = np.asarray(coefficient, dtype=np.float64)
    p = _rebuttal_nonnegative_mean(np.maximum(coefficient, 0))
    q = _rebuttal_nonnegative_mean(np.maximum(-coefficient, 0))
    if not (math.isfinite(p) and math.isfinite(q)):
        ratio, status = None, "nonfinite_mass"
    elif p > 0:
        ratio, status = q / p, "finite"
        if not math.isfinite(ratio):
            ratio, status = None, "ratio_overflow"
    else:
        ratio, status = None, "p_zero_q_positive" if q > 0 else "both_zero"
    return {"p": p, "q": q, "rho": ratio, "ratio_status": status}


def current_action_log_prob(
    policy: SquashedGaussianPolicy, obs: torch.Tensor, actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """New-profile action density and RMS-squared native Gaussian remoteness.

    Use one safe action for both inverse and Jacobian. Double arithmetic helps
    evaluate raw IS; it does not clamp its log-ratio or alter old Q2 log_prob.
    """
    safe = actions.double().clamp(
        -1.0 + policy.action_clip_epsilon, 1.0 - policy.action_clip_epsilon
    )
    latent = torch.atanh(safe)
    mean, log_std = policy.latent_parameters(obs)
    z = (latent - mean.double()) * torch.exp(-log_std.double())
    gaussian = -0.5 * (z.square() + 2 * log_std.double() + math.log(2 * math.pi))
    logp = (gaussian - torch.log1p(-safe.square())).sum(-1)
    boundary = (safe != actions.double()).any(-1)
    return logp, z.square().mean(-1).detach(), boundary


def rebuttal_actor_coefficients(
    advantage: torch.Tensor, logp: torch.Tensor, distance_squared: torch.Tensor,
    behavior_logp: torch.Tensor, method: str, settings: dict,
) -> torch.Tensor:
    a = advantage.detach().double()
    if method == "positive_only":
        return a.clamp_min(0)
    if method == "signed":
        return a
    if method == "drpo":
        exponent = -settings["drpo_exponent_coefficient"] * (
            distance_squared.detach().double() / settings["drpo_reference_distance"] ** 2
        )
        weight = settings["drpo_scale"] * torch.exp(exponent)
        return a.clamp_min(0) - weight * (-a).clamp_min(0)
    if method == "is":
        return torch.exp(logp.detach().double() - behavior_logp.double()) * a
    raise ValueError(f"Unknown rebuttal method: {method}")


def rebuttal_actor_loss(logp: torch.Tensor, coefficient: torch.Tensor) -> torch.Tensor:
    return -(coefficient.detach() * logp).mean()


def _clip_rebuttal_gradients(parameters: Iterable[nn.Parameter], limit: float) -> torch.Tensor:
    """Same global norm rule, with float64 reduction of finite raw gradients."""
    gradients = [p.grad for p in parameters if p.grad is not None]
    if not gradients:
        return torch.tensor(0.0, dtype=torch.float64)
    norm = torch.sqrt(sum(g.detach().double().square().sum() for g in gradients))
    if bool(torch.isfinite(norm)):
        factor = (limit / (norm + 1e-6)).clamp(max=1.0)
        for gradient in gradients:
            gradient.mul_(factor.to(gradient.dtype))
    return norm


def _q2_force_from_graph(policy: SquashedGaussianPolicy, logp: torch.Tensor, coefficient: torch.Tensor) -> torch.Tensor:
    """Full-parameter ascent force, full B denominator, no higher derivatives."""
    gradients = torch.autograd.grad(
        (logp * coefficient.detach()).mean(), list(policy.parameters()), retain_graph=True,
    )
    return torch.cat([gradient.detach().double().flatten() for gradient in gradients])


def q2_cap_factor(far_norm: torch.Tensor, cap: float) -> torch.Tensor:
    denominator = torch.where(far_norm > 0, far_norm, torch.ones_like(far_norm))
    return torch.where(far_norm > 0, (cap / denominator).clamp(max=1), torch.ones_like(far_norm)).detach()


def q2_global_scale(target: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
    denominator = torch.where(raw > 0, raw, torch.ones_like(raw))
    return torch.where((target > 0) & (raw > 0), target / denominator, torch.zeros_like(target)).detach()


def q2_pair_coefficients(
    actors: Sequence[SquashedGaussianPolicy], logps: Sequence[torch.Tensor],
    distances: Sequence[torch.Tensor], advantage: torch.Tensor, calibration: dict,
    intervention: str,
) -> tuple[list[torch.Tensor], dict[str, torch.Tensor]]:
    """Budgets come from the two actual pre-update actors, never a proxy."""
    a = advantage.detach().double()
    positive, negative = a.clamp_min(0), (-a).clamp_min(0)
    far = ((a < 0) & (distances[0].detach() > calibration["far_threshold"])).detach()
    near = (a < 0) & ~far
    g_near = _q2_force_from_graph(actors[0], logps[0], -negative * near)
    g_far = _q2_force_from_graph(actors[0], logps[0], -negative * far)
    near_norm, far_norm = g_near.norm(), g_far.norm()
    kappa = q2_cap_factor(far_norm, calibration["far_cap"]) if intervention == "far_cap" else far_norm.new_zeros(())
    c_intervention = positive - negative * (near.double() + kappa * far.double())
    # Measure the actual weighted backward, including parameter-dtype rounding.
    g_intervention = (g_near if intervention == "far_zero" else
                      _q2_force_from_graph(actors[0], logps[0], c_intervention - positive))
    target = g_intervention.norm()
    g_global = _q2_force_from_graph(actors[1], logps[1], -negative)
    raw_global = g_global.norm()
    alpha = q2_global_scale(target, raw_global)
    c_global = positive - alpha * negative
    actual = _q2_force_from_graph(actors[1], logps[1], c_global - positive).norm()
    error = (actual - target).abs()
    nonzero = target > 0
    unmatched = nonzero & (raw_global == 0)
    statistics = {
        "target_budget": target, "raw_global_budget": raw_global, "actual_global_budget": actual,
        "absolute_error": error,
        "relative_error_nonzero_target": torch.where(nonzero, error / torch.where(nonzero, target, torch.ones_like(target)), error.new_zeros(())),
        "nonzero_target": nonzero, "zero_target": ~nonzero,
        "matchable": ~unmatched, "unmatched_zero_raw_gradient": unmatched,
        "alpha": alpha, "alpha_gt_one": alpha > 1, "kappa": kappa,
        "cap_triggered": (kappa < 1) if intervention == "far_cap" else far_norm.new_zeros(()),
        "far_count": far.sum(), "far_fraction": far.double().mean(),
        "near_count": near.sum(), "near_negative_norm": near_norm,
        "far_negative_norm_before": far_norm, "far_negative_norm_after": kappa * far_norm,
        "negative_norm_before": (g_near + g_far).norm(), "negative_norm_after": target,
        "intervention_p": c_intervention.clamp_min(0).mean(), "intervention_q": (-c_intervention).clamp_min(0).mean(),
        "global_p": c_global.clamp_min(0).mean(), "global_q": (-c_global).clamp_min(0).mean(),
    }
    return [c_intervention.detach(), c_global.detach()], {k: v.detach().double() for k, v in statistics.items()}


def calibrate_rebuttal_q2(policy: SquashedGaussianPolicy, context: dict, seed: int, device: torch.device) -> dict:
    settings, data = context["settings"], context["data"]
    q2, ids = settings["q2"], context["diagnostic_ids"]
    negative = context["initial_advantage"][ids] < 0
    record = {"seed": seed, "settings_identity": settings["identity"],
              "diagnostic_ids_sha256": context["diagnostic_ids_sha256"],
              "negative_count": int(negative.sum()), "quantile": q2["far_threshold_quantile"],
              "calibration_batches": q2["cap_calibration_batches"],
              "calibration_batch_size": q2["cap_calibration_batch_size"],
              "rng_seed_sequence": [seed, q2["calibration_rng_role"]]}
    if not negative.any():
        return {**record, "status": "undefined_no_initial_negative", "far_threshold": None, "far_cap": None}
    distances = []
    with torch.no_grad():
        for start in range(0, len(ids), settings["diagnostic_batch_size"]):
            ix = ids[start:start + settings["diagnostic_batch_size"]]
            _, d, _ = current_action_log_prob(policy, tensor(context["obs"][ix], device), tensor(data.actions[ix], device))
            distances.append(d.cpu().numpy())
    threshold = float(np.quantile(np.concatenate(distances)[negative], q2["far_threshold_quantile"]))
    rng = np.random.default_rng(np.random.SeedSequence(record["rng_seed_sequence"]))
    norms, batch_hash = [], hashlib.sha256()
    for _ in range(q2["cap_calibration_batches"]):
        ix = sample_indices(rng, context["train_ids"], q2["cap_calibration_batch_size"])
        batch_hash.update(data.transition_ids[ix].astype("<i8").tobytes())
        lp, d, _ = current_action_log_prob(policy, tensor(context["obs"][ix], device), tensor(data.actions[ix], device))
        a = torch.as_tensor(context["initial_advantage"][ix], device=device)
        force = _q2_force_from_graph(policy, lp, a.double().clamp(max=0) * (d > threshold))
        norms.append(float(force.norm().cpu()))
    cap = float(np.median(norms))
    return {**record, "status": "defined" if math.isfinite(threshold) and math.isfinite(cap) else "nonfinite_calibration",
            "far_threshold": threshold, "far_cap": cap, "zero_cap_degeneracy": cap == 0,
            "initial_far_block_norms": norms, "calibration_transition_ids_sha256": batch_hash.hexdigest()}


class _Q2BudgetBlock:
    """Constant-memory device reductions; no million-row per-step output."""

    def __init__(self, device: torch.device, state: dict | None = None):
        self.device, self.count, self.keys = device, 0, []
        self.sums = self.maxima = None
        if state:
            self.count, self.keys = state["count"], state["keys"]
            self.sums, self.maxima = state["sums"].to(device), state["maxima"].to(device)

    def add(self, values: dict[str, torch.Tensor]) -> None:
        vector = torch.stack(list(values.values()))
        if self.count == 0:
            self.keys, self.sums, self.maxima = list(values), vector.clone(), vector.clone()
        else:
            self.sums.add_(vector)
            self.maxima = torch.maximum(self.maxima, vector)
        self.count += 1

    def state_dict(self) -> dict | None:
        return {"count": self.count, "keys": self.keys, "sums": self.sums.cpu(), "maxima": self.maxima.cpu()} if self.count else None

    def finish(self, step: int) -> dict:
        sums, maxima = self.sums.cpu().tolist(), self.maxima.cpu().tolist()
        row = {"step": step, "start_step": step - self.count + 1, "updates": self.count}
        totals = dict(zip(self.keys, sums))
        for key, total, maximum in zip(self.keys, sums, maxima):
            denominator = totals["nonzero_target"] if key == "relative_error_nonzero_target" else self.count
            row[f"q2_{key}_mean"] = total / denominator if denominator else None
            row[f"q2_{key}_max"] = maximum if denominator else None
            if key in ("nonzero_target", "zero_target", "matchable", "unmatched_zero_raw_gradient", "alpha_gt_one", "cap_triggered"):
                row[f"q2_{key}_steps"] = int(total)
        self.count = 0
        return row


def current_rebuttal_advantage(
    critic: ValueNetwork, obs: torch.Tensor, next_obs: torch.Tensor,
    reward: torch.Tensor, done: torch.Tensor, target_norm: Normalizer,
    gamma: float, center: float, scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        m, s = float(target_norm.mean[0]), float(target_norm.std[0])
        value = critic(obs).double() * s + m
        next_value = critic(next_obs).double() * s + m
        delta = reward.double() + gamma * (~done).double() * next_value - value
        return ((delta - center) / scale).detach(), delta.detach()


def update_rebuttal_critic(
    critic: ValueNetwork, optimizer: torch.optim.Optimizer,
    obs: torch.Tensor, next_obs: torch.Tensor, reward: torch.Tensor,
    done: torch.Tensor, target_norm: Normalizer, gamma: float, settings: dict,
) -> dict[str, Any]:
    m, s = float(target_norm.mean[0]), float(target_norm.std[0])
    value = critic(obs)
    with torch.no_grad():
        target = reward.double() + gamma * (~done).double() * (
            critic(next_obs).double() * s + m
        )
        error_sign = target > value.detach().double() * s + m
        eta = torch.where(
            error_sign, settings["critic_expectile"], 1 - settings["critic_expectile"]
        )
        normalized_target = (target - m) / s
    loss = (eta * (value - normalized_target).square()).mean()
    optimizer.zero_grad(set_to_none=True)
    if not bool(torch.isfinite(loss)):
        return {"loss": float("nan"), "finite": False}
    loss.backward()
    norm = _clip_rebuttal_gradients(critic.parameters(), settings["critic_gradient_clip"])
    finite = bool(torch.isfinite(norm))
    if finite:
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return {"loss": loss.detach(), "gradient_norm": norm.detach(), "finite": finite}


def _rebuttal_weight_statistics(log_ratio: np.ndarray) -> dict[str, Any]:
    x = np.asarray(log_ratio, dtype=np.float64)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        weights = np.exp(x)
    result = {
        "ratio_nonfinite_fraction": float(np.mean(~np.isfinite(weights))),
        "ratio_zero_fraction": float(np.mean(weights == 0)),
    }
    for prefix, values in (("log_ratio", x), ("ratio", weights)):
        with np.errstate(over="ignore", invalid="ignore"):
            mean = _rebuttal_nonnegative_mean(values) if prefix == "ratio" else np.mean(values)
            summaries = [mean, *np.quantile(values, [0.5, 0.9, 0.99]), np.max(values)]
        for name, val in zip(("mean", "median", "p90", "p99", "max"), summaries):
            result[f"{prefix}_{name}"] = float(val)
    if np.all(np.isfinite(x)):
        scaled = np.exp(x - np.max(x))
        total = float(scaled.sum())
        result["ess"] = total ** 2 / float(np.square(scaled).sum())
        result["ess_fraction"] = result["ess"] / len(x)
        k = max(1, math.ceil(len(x) * 0.01))
        result["top_1pct_weight_share"] = float(np.sort(scaled)[-k:].sum()) / total
    else:
        result.update(ess=None, ess_fraction=None, top_1pct_weight_share=None)
    return result


def _rebuttal_force(
    policy: SquashedGaussianPolicy, obs: np.ndarray, actions: np.ndarray,
    coefficient: np.ndarray, denominator: int, batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    params = list(policy.parameters())
    force = torch.zeros(sum(p.numel() for p in params), dtype=torch.float64, device=device)
    if not np.any(coefficient):
        return force
    for start in range(0, len(obs), batch_size):
        stop = start + batch_size
        logp, _, _ = current_action_log_prob(
            policy, tensor(obs[start:stop], device), tensor(actions[start:stop], device)
        )
        c = torch.as_tensor(coefficient[start:stop], dtype=torch.float64, device=device)
        grads = torch.autograd.grad(-(c * logp).sum() / denominator, params)
        force -= torch.cat([g.detach().reshape(-1).double() for g in grads])
    return force


def _rebuttal_advantage_arrays(
    critic: ValueNetwork, context: dict, indices: np.ndarray, device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    chunks, deltas = [], []
    data, base = context["data"], context["base"]
    for start in range(0, len(indices), 4096):
        ids = indices[start:start + 4096]
        a, delta = current_rebuttal_advantage(
            critic, tensor(context["obs"][ids], device),
            tensor(context["next_obs"][ids], device), tensor(data.rewards[ids], device),
            torch.as_tensor(data.terminals[ids] | data.timeouts[ids], device=device),
            context["target_norm"], base.gamma, context["center"], context["scale"],
        )
        chunks.append(a.cpu().numpy())
        deltas.append(delta.cpu().numpy())
    return np.concatenate(chunks), np.concatenate(deltas)


def collect_rebuttal_diagnostics(
    policy: SquashedGaussianPolicy, critic: ValueNetwork, context: dict,
    method: str, advantage_mode: str, step: int, previous: np.ndarray | None,
    device: torch.device, effective_coefficient: np.ndarray | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    settings, data = context["settings"], context["data"]
    ids = context["diagnostic_ids"]
    initial = context["initial_advantage"][ids]
    if advantage_mode == "frozen":
        a, delta = initial.copy(), context["initial_delta"][ids]
    else:
        a, delta = _rebuttal_advantage_arrays(critic, context, ids, device)
    logps, distances, boundaries = [], [], []
    with torch.no_grad():
        for start in range(0, len(ids), settings["diagnostic_batch_size"]):
            ix = ids[start:start + settings["diagnostic_batch_size"]]
            lp, d, boundary = current_action_log_prob(
                policy, tensor(context["obs"][ix], device), tensor(data.actions[ix], device)
            )
            logps.append(lp.cpu())
            distances.append(d.cpu())
            boundaries.append(boundary.cpu())
    logp, d = torch.cat(logps), torch.cat(distances)
    boundary = torch.cat(boundaries).numpy()
    coefficient = effective_coefficient if effective_coefficient is not None else rebuttal_actor_coefficients(
        torch.from_numpy(a), logp, d, torch.from_numpy(data.behavior_log_prob[ids]),
        method, settings,
    ).numpy()
    row: dict[str, Any] = {
        "step": step, "sample_count": len(ids),
        "diagnostic_ids_sha256": context["diagnostic_ids_sha256"],
        "raw_delta_mean": float(delta.mean()), "raw_delta_std": float(delta.std()),
        "advantage_mean": float(a.mean()), "advantage_std": float(a.std()),
        "positive_fraction": float(np.mean(a > 0)),
        "negative_fraction": float(np.mean(a < 0)), "zero_fraction": float(np.mean(a == 0)),
        "boundary_clip_count": int(boundary.sum()),
    }
    for label, values in (("raw", a), ("effective", coefficient), ("td", delta)):
        for key, val in rebuttal_mass(values).items():
            row[f"{key}_{label}"] = val
    for label, reference in (("initial", initial), ("previous", previous)):
        if reference is None:
            reference = a
        difference = a - reference
        row.update({
            f"{label}_mean_abs_change": float(np.mean(np.abs(difference))),
            f"{label}_rms_change": float(np.sqrt(np.mean(difference ** 2))),
            f"{label}_pos_to_neg_fraction": float(np.mean((reference > 0) & (a < 0))),
            f"{label}_neg_to_pos_fraction": float(np.mean((reference < 0) & (a > 0))),
            f"{label}_zero_to_nonzero_fraction": float(np.mean((reference == 0) & (a != 0))),
            f"{label}_nonzero_to_zero_fraction": float(np.mean((reference != 0) & (a == 0))),
        })
    distance = d.numpy()
    negative = a < 0
    cuts = np.quantile(distance[negative], [0.25, 0.75]) if negative.any() else [None, None]
    row["near_distance_cut"], row["far_distance_cut"] = cuts
    if negative.any():
        row["negative_distance_mean"] = float(distance[negative].mean())
        for quantile in (0.25, 0.5, 0.75, 0.9, 0.99):
            row[f"negative_D_q{int(100 * quantile)}"] = float(np.quantile(distance[negative], quantile))
    groups = {
        "positive": a > 0, "negative": negative,
        "near_negative": negative & (distance <= cuts[0]) if negative.any() else negative,
        "far_negative": negative & (distance >= cuts[1]) & (distance > cuts[0]) if negative.any() else negative,
        "boundary": boundary,
    }
    for name, mask in groups.items():
        row[f"{name}_count"] = int(mask.sum())
        row[f"{name}_abs_advantage_mean"] = float(np.abs(a[mask]).mean()) if mask.any() else None
        row[f"{name}_abs_coefficient_mean"] = float(np.abs(coefficient[mask]).mean()) if mask.any() else None
        raw_norm = 0.0
        for label, values in (("raw", a), ("effective", coefficient)):
            if label == "effective" and np.array_equal(a[mask], coefficient[mask]):
                row[f"{label}_{name}_force_norm"] = raw_norm
                continue
            if mask.any():
                force = _rebuttal_force(
                    policy, context["obs"][ids[mask]], data.actions[ids[mask]],
                    values[mask], len(ids), settings["diagnostic_batch_size"], device,
                )
                norm = float(torch.linalg.vector_norm(force).cpu())
            else:
                norm = 0.0
            row[f"{label}_{name}_force_norm"] = norm
            if label == "raw":
                raw_norm = norm
    for label in ("raw", "effective"):
        far, near = (row[f"{label}_{name}_negative_force_norm"] for name in ("far", "near"))
        row[f"{label}_far_vs_near_norm_share"] = far / (far + near) if far + near > 0 else None
    log_ratio = logp.numpy() - data.behavior_log_prob[ids].astype(np.float64)
    row.update(_rebuttal_weight_statistics(log_ratio))
    scaled = np.exp(log_ratio - np.max(log_ratio)) if np.all(np.isfinite(log_ratio)) else None
    row["boundary_ratio_mass_share"] = float(scaled[boundary].sum() / scaled.sum()) if scaled is not None else None
    abs_c = np.abs(coefficient)
    row["boundary_coefficient_mass_share"] = float(abs_c[boundary].sum() / abs_c.sum()) if abs_c.sum() > 0 else None
    with torch.no_grad():
        row["log_std_min_fraction"] = float((policy.log_std <= policy.log_std_min).double().mean())
        row["log_std_max_fraction"] = float((policy.log_std >= policy.log_std_max).double().mean())
        means = policy.action_mean(tensor(context["obs"][ids], device))
        row["action_mean_boundary_fraction"] = float((means.abs() >= context["base"].support_boundary_threshold).double().mean())
    if effective_coefficient is not None:
        row = {k.replace("effective", "audit_effective"): v for k, v in row.items()}
        row["effective_coefficient_scope"] = "fixed_population_audit_not_training_batch"
    return _rebuttal_json(row), a.copy()


def collect_rebuttal_gradient_variance(
    policy: SquashedGaussianPolicy, critic: ValueNetwork, context: dict,
    advantage_mode: str, source_method: str, step: int, seed: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    settings, data = context["settings"], context["data"]
    rng = np.random.default_rng(np.random.SeedSequence([settings["diagnostic_seed"], seed, step, 19]))
    statistics: dict[str, dict] = {rule: {"n": 0, "m2": 0.0, "norm_sum": 0.0, "nonfinite": False}
                                    for rule in ("signed", "drpo", "is")}
    log_ratios = []
    for _ in range(settings["variance_batches"]):
        ids = sample_indices(rng, context["train_ids"], settings["variance_batch_size"])
        a = context["initial_advantage"][ids] if advantage_mode == "frozen" else _rebuttal_advantage_arrays(critic, context, ids, device)[0]
        with torch.no_grad():
            logp, d, _ = current_action_log_prob(policy, tensor(context["obs"][ids], device), tensor(data.actions[ids], device))
        behavior = torch.as_tensor(data.behavior_log_prob[ids], device=device)
        log_ratios.append(logp.cpu().numpy() - data.behavior_log_prob[ids])
        for rule, stats in statistics.items():
            c = rebuttal_actor_coefficients(torch.as_tensor(a, device=device), logp, d, behavior, rule, settings)
            force = _rebuttal_force(policy, context["obs"][ids], data.actions[ids], c.cpu().numpy(), len(ids), len(ids), device)
            if not bool(torch.isfinite(force).all()):
                stats["nonfinite"] = True
                continue
            stats["n"] += 1
            stats["norm_sum"] += float(force.norm().cpu())
            if stats["n"] == 1:
                stats["mean"] = force.clone()
            else:
                difference = force - stats["mean"]
                stats["mean"] += difference / stats["n"]
                stats["m2"] += float(torch.dot(difference, force - stats["mean"]).cpu())
    weights = _rebuttal_weight_statistics(np.concatenate(log_ratios))
    rows = []
    for rule, stats in statistics.items():
        complete = not stats["nonfinite"] and stats["n"] == settings["variance_batches"]
        rows.append(_rebuttal_json({
            "step": step, "rule": rule, "checkpoint_source_method": source_method,
            "variance_trace": stats["m2"] / (stats["n"] - 1) if complete and stats["n"] > 1 else None,
            "mean_gradient_norm": float(stats["mean"].norm().cpu()) if complete else None,
            "mean_batch_gradient_norm": stats["norm_sum"] / stats["n"] if complete else None,
            "batch_size": settings["variance_batch_size"], "M": settings["variance_batches"],
            "valid_batches": stats["n"], "status": "finite" if complete else "nonfinite_estimator",
            **weights,
        }))
    return rows


def _new_rebuttal_policy(base: E7Config, obs_dim: int, action_dim: int) -> SquashedGaussianPolicy:
    return SquashedGaussianPolicy(
        obs_dim, action_dim, base.hidden_sizes, base.log_std_min, base.log_std_max,
        base.action_clip_epsilon, base.activation, base.init_scheme, base.init_gain,
    )


def _atomic_rebuttal_checkpoint(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def _rebuttal_rng_state(device: torch.device) -> dict:
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def _restore_rebuttal_rng(state: dict, device: torch.device) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if device.type == "cuda" and state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"].cpu(), device)


def prepare_rebuttal_context(
    dataset: Path, root: Path, settings: dict, base: E7Config,
    device: torch.device, source_commit: str,
) -> dict:
    """Prepare once; old Q2 run_seed/warmstart/active-mean are never called."""
    root.mkdir(parents=True, exist_ok=True)
    seed_everything(settings["critic_seed"])
    data = load_hopper_hdf5(dataset, settings["max_transitions"])
    if data.behavior_log_prob is None or len(data.behavior_log_prob) != data.size:
        raise ValueError("Raw IS requires aligned infos/action_log_probs in this input")
    if not np.isfinite(data.behavior_log_prob).all():
        raise ValueError("Input behavior log probabilities contain non-finite values")
    split = split_episode_indices(data.episode_ids, settings["critic_seed"], base.train_fraction, base.validation_fraction)
    obs_norm = Normalizer.fit(data.observations[split["train"]])
    returns = discounted_returns(data.rewards, data.terminals, data.timeouts, base.gamma)
    mode = replace(
        base.formal, max_transitions=settings["max_transitions"],
        critic_max_steps=settings["critic_steps"], critic_min_steps=settings["critic_steps"],
        critic_eval_interval=settings["critic_eval_interval"],
        audit_sample_size=settings["diagnostic_size"],
    )
    critic, target_norm, audit = train_critic(
        data=data, split=split, obs_norm=obs_norm, returns=returns, config=base,
        mode=mode, seed=settings["critic_seed"], device=device,
        output_dir=root / "critic_training",
    )
    if not audit["fixed_budget_completed"]:
        raise RuntimeError("The shared critic did not complete its fixed pretraining budget")
    context = {
        "data": data, "base": base, "obs": obs_norm.transform(data.observations),
        "next_obs": obs_norm.transform(data.next_observations), "target_norm": target_norm,
        "center": 0.0, "scale": 1.0,
    }
    _, delta = _rebuttal_advantage_arrays(critic, context, np.arange(data.size), device)
    center = float(delta[split["train"]].mean())
    scale = max(float(delta[split["train"]].std()), 1e-8)
    advantage = (delta - center) / scale
    diagnostic_ids = np.random.default_rng(settings["diagnostic_seed"]).choice(
        split["train"], min(settings["diagnostic_size"], len(split["train"])), replace=False
    )
    np.savez_compressed(root / "split_indices.npz", **split)
    np.save(root / "diagnostic_transition_ids.npy", data.transition_ids[diagnostic_ids])
    np.savez_compressed(root / "initial_advantages.npz", advantage=advantage, delta=delta)
    prepared = {
        "settings_identity": settings["identity"], "source_commit": source_commit,
        "data_sha256": sha256_file(dataset), "transitions": data.size,
        "obs_mean": obs_norm.mean, "obs_std": obs_norm.std,
        "target_mean": target_norm.mean, "target_std": target_norm.std,
        "advantage_center": center, "advantage_scale": scale,
        "critic": {k: v.detach().cpu() for k, v in critic.state_dict().items()},
        "diagnostic_ids": diagnostic_ids,
        "diagnostic_ids_sha256": sha256_file(root / "diagnostic_transition_ids.npy"),
    }
    for seed in settings["seeds"]:
        seed_everything(seed)
        actor = _new_rebuttal_policy(base, data.observations.shape[1], data.actions.shape[1])
        _atomic_rebuttal_checkpoint(root / f"actor_initial_seed_{seed}.pt", {"actor": actor.state_dict()})
        calibration = calibrate_rebuttal_q2(actor.to(device), {
            **context, "settings": settings, "initial_advantage": advantage,
            "train_ids": split["train"], "diagnostic_ids": diagnostic_ids,
            "diagnostic_ids_sha256": prepared["diagnostic_ids_sha256"],
        }, seed, device)
        calibration.update(source_commit=source_commit, actor_initial_sha256=sha256_file(root / f"actor_initial_seed_{seed}.pt"))
        _write_rebuttal_json(root / f"q2_calibration_seed_{seed}.json", calibration)
    _atomic_rebuttal_checkpoint(root / "context.pt", prepared)
    _write_rebuttal_json(root / "normalization.json", {k: v for k, v in prepared.items() if k in (
        "obs_mean", "obs_std", "target_mean", "target_std", "advantage_center", "advantage_scale"
    )})
    _write_rebuttal_json(root / "data_manifest.json", {
        "path": str(dataset), "sha256": prepared["data_sha256"], "transitions": data.size,
        "behavior_logp_source": "D4RL replay env_infos/action_log_prob -> infos/action_log_probs",
        "behavior_logp_min": float(data.behavior_log_prob.min()),
        "behavior_logp_max": float(data.behavior_log_prob.max()),
        "boundary_exact_count": int(np.any(np.abs(data.actions) == 1, axis=1).sum()),
        "boundary_clip_count": int(np.any(np.abs(data.actions) > 1 - base.action_clip_epsilon, axis=1).sum()),
        "density_processing": "shared a_safe for atanh/Jacobian; finite-precision approximation at boundaries",
    })
    _write_rebuttal_json(root / "initial_advantage_summary.json", {
        "center": center, "scale": scale, "train_population": len(split["train"]),
        **rebuttal_mass(advantage[split["train"]]),
    })
    shutil.copy2(root / "critic_training" / "critic_metrics.csv", root / "critic_pretrain_metrics.csv")
    _write_rebuttal_json(root / "critic_checkpoint_index.json", audit["checkpoint"])
    inputs = ["context.pt", "initial_advantages.npz", "split_indices.npz", "diagnostic_transition_ids.npy"]
    inputs.extend(f"actor_initial_seed_{seed}.pt" for seed in settings["seeds"])
    inputs.extend(f"q2_calibration_seed_{seed}.json" for seed in settings["seeds"])
    _write_rebuttal_json(root / "PREPARE_COMPLETE.json", {
        "source_commit": source_commit, "settings_identity": settings["identity"],
        "context_sha256": sha256_file(root / "context.pt"), "data_sha256": prepared["data_sha256"],
        "prepared_input_hashes": {name: sha256_file(root / name) for name in inputs},
    })
    return prepared


def _load_rebuttal_context(root: Path, dataset: Path, settings: dict, base: E7Config) -> dict:
    prepared = torch.load(root / "context.pt", map_location="cpu", weights_only=False)
    if prepared["settings_identity"] != settings["identity"]:
        raise ValueError("Preparation/config identity mismatch")
    data = load_hopper_hdf5(dataset, settings["max_transitions"])
    obs_norm = Normalizer(prepared["obs_mean"], prepared["obs_std"])
    target_norm = Normalizer(prepared["target_mean"], prepared["target_std"])
    with np.load(root / "initial_advantages.npz") as arrays:
        initial, initial_delta = arrays["advantage"].copy(), arrays["delta"].copy()
    with np.load(root / "split_indices.npz") as arrays:
        split = {k: arrays[k].copy() for k in arrays.files}
    return {
        "prepared": prepared, "settings": settings, "base": base, "data": data,
        "obs_norm": obs_norm, "target_norm": target_norm,
        "obs": obs_norm.transform(data.observations), "next_obs": obs_norm.transform(data.next_observations),
        "train_ids": split["train"], "validation_ids": split["validation"],
        "diagnostic_ids": prepared["diagnostic_ids"],
        "diagnostic_ids_sha256": prepared["diagnostic_ids_sha256"],
        "initial_advantage": initial, "initial_delta": initial_delta,
        "center": prepared["advantage_center"], "scale": prepared["advantage_scale"],
        "returns": discounted_returns(data.rewards, data.terminals, data.timeouts, base.gamma),
    }


def _rebuttal_mc_reference_mse(critic: ValueNetwork, context: dict, device: torch.device) -> float:
    ids = context["validation_ids"]
    error = 0.0
    norm = context["target_norm"]
    with torch.no_grad():
        for start in range(0, len(ids), 4096):
            ix = ids[start:start + 4096]
            value = critic(tensor(context["obs"][ix], device)).double()
            value = value * float(norm.std[0]) + float(norm.mean[0])
            error += float((value - torch.as_tensor(context["returns"][ix], device=device)).square().sum().cpu())
    return error / len(ids)


def _rebuttal_cell(root: Path, seed: int, mode: str, method: str) -> Path:
    return root / "cells" / f"seed_{seed}" / mode / method


def _rebuttal_rollout_row(actor, context: dict, cell: Path, step: int, seed: int, device: torch.device) -> dict:
    settings, base = context["settings"], context["base"]
    # Environment setup may consume global RNG in third-party libraries.
    state = _rebuttal_rng_state(device)
    try:
        result = evaluate_d4rl_rollouts(
            policy=actor, obs_norm=context["obs_norm"], backend=base.env_backend,
            dataset_id=base.rollout_dataset_id, env_id=base.env_id,
            episodes=settings["terminal_evaluation_episodes"] if step == settings["actor_steps"] else settings["evaluation_episodes"],
            seed=settings["evaluation_seed"] + seed * 100000 + step,
            device=device, normalized_score_percent=base.normalized_score_percent,
            reference_min_score=base.normalized_score_reference_min,
            reference_max_score=base.normalized_score_reference_max, required=True,
            diagnostics_path=cell / "rollouts" / f"step_{step:08d}.json",
        )
    finally:
        _restore_rebuttal_rng(state, device)
    return {
        "step": step, "raw_return": result["rollout_return_mean"],
        "episode_return_sd": result["rollout_return_std"],
        "normalized_return": result["normalized_return"],
        "episodes": result["rollout_episodes"], "status": result["rollout_status"],
    }


def _rebuttal_branch_summary(identity: dict, context: dict, history: dict, completed: int, critic_updates: int, failure_record: dict | None) -> dict:
    settings, base = context["settings"], context["base"]
    failure = failure_record["reason"] if failure_record else None
    rollouts = history["rollouts"]
    late = [r for r in rollouts if r["step"] in settings["late_steps"]]
    complete_late = len(late) == len(settings["late_steps"]) and failure is None
    terminal = next((r["normalized_return"] for r in rollouts if r["step"] == settings["actor_steps"]), None)
    best = max(r["normalized_return"] for r in rollouts) if rollouts else None
    boundary = any(r["action_mean_boundary_fraction"] >= base.support_boundary_fraction or r["log_std_min_fraction"] > 0 or r["log_std_max_fraction"] > 0 for r in history["diagnostics"])
    summary = {
        **identity, "status": "numerical_failure" if failure else "completed",
        "completed_actor_steps": completed, "scheduled_actor_steps": settings["actor_steps"],
        "critic_update_count": critic_updates, "numerical_failure": failure,
        "numerical_failure_observed_actor_steps": failure_record["completed_actor_steps"] if failure_record else None,
        "support_or_variance_boundary_event": boundary, "task_collapse": None,
        "task_collapse_status": "not_adjudicated_no_coldstart_threshold",
        "terminal_classification": "numerical_failure" if failure else "fixed_horizon_inconclusive",
        "late_count": len(late), "late_expected_count": len(settings["late_steps"]),
        "late_mean": float(np.mean([r["normalized_return"] for r in late])) if complete_late else None,
        "terminal": terminal, "best": best,
        "best_to_terminal_drop": best - terminal if terminal is not None else None,
        "late_slope_per_update": float(np.polyfit([r["step"] for r in late], [r["normalized_return"] for r in late], 1)[0]) if complete_late and len(late) > 1 else None,
        "scientific_status": "smoke" if settings["smoke"] else "finite_budget_diagnostic",
    }
    return summary


def train_rebuttal_branch(args: argparse.Namespace) -> int:
    settings, base = load_rebuttal_config(args.config, args.smoke)
    torch.set_num_threads(settings["worker_threads"])
    device = choose_device(args.device)
    seed, mode, method = args.seed, args.advantage_mode, args.method
    root = Path(args.work_dir).resolve()
    cell = _rebuttal_cell(root, seed, mode, method)
    cell.mkdir(parents=True, exist_ok=True)
    context = _load_rebuttal_context(root / "prepare", Path(args.dataset_path), settings, base)
    data = context["data"]
    seed_everything(seed)
    actor = _new_rebuttal_policy(base, data.observations.shape[1], data.actions.shape[1]).to(device)
    actor.load_state_dict(torch.load(root / "prepare" / f"actor_initial_seed_{seed}.pt", map_location=device, weights_only=False)["actor"])
    critic = ValueNetwork(data.observations.shape[1], base.hidden_sizes, base.activation, base.init_scheme, base.init_gain).to(device)
    critic.load_state_dict(context["prepared"]["critic"])
    actor_opt = torch.optim.AdamW(actor.parameters(), lr=base.actor_lr, weight_decay=base.weight_decay)
    critic_opt = torch.optim.AdamW(critic.parameters(), lr=base.critic_lr, weight_decay=base.weight_decay)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 41591]))
    identity = {"settings_identity": settings["identity"], "seed": seed, "advantage_mode": mode,
                "method": method, "source_commit": context["prepared"]["source_commit"],
                "context_sha256": sha256_file(root / "prepare" / "context.pt")}
    history: dict[str, list] = {k: [] for k in ("training_metrics", "diagnostics", "rollouts", "gradient_variance")}
    start, completed, critic_updates, clipped, previous = 0, 0, 0, 0, None
    critic_reference = torch.cat([p.detach().cpu().flatten() for p in critic.parameters()])
    failure_record = None
    if args.resume_from:
        previous_cell = _rebuttal_cell(Path(args.resume_from), seed, mode, method)
        old_checkpoint = previous_cell / "checkpoint.pt"
        if old_checkpoint.is_file():
            state = torch.load(old_checkpoint, map_location="cpu", weights_only=False)
            if state["identity"] != identity:
                raise ValueError("Checkpoint does not belong to this exact branch/source/config")
            actor.load_state_dict(state["actor"])
            critic.load_state_dict(state["critic"])
            actor_opt.load_state_dict(state["actor_optimizer"])
            critic_opt.load_state_dict(state["critic_optimizer"])
            rng.bit_generator.state = state["training_rng"]
            _restore_rebuttal_rng(state["global_rng"], device)
            start = completed = state["step"]
            critic_updates, clipped = state["critic_updates"], state["clipped"]
            history, previous = state["history"], state["previous_advantage"]
            critic_reference = state["critic_reference"].cpu()
            failure_record = state.get("numerical_failure")
        marker = previous_cell / "numerical_failure.json"
        if marker.is_file():
            failure_record = json.loads(marker.read_text())
    last_critic: dict[str, Any] = {"loss": None, "gradient_norm": None}
    failure = failure_record["reason"] if failure_record else None

    def flush() -> None:
        for name, rows in history.items():
            write_csv(cell / f"{name}.csv", rows)
        _write_rebuttal_json(cell / "status.json", {
            **identity, "status": "running", "step": completed, "utc": utc_now(),
            "critic_update_count": critic_updates,
        })

    def checkpoint() -> None:
        flush()
        _atomic_rebuttal_checkpoint(cell / "checkpoint.pt", {
            "identity": identity, "step": completed, "critic_updates": critic_updates,
            "clipped": clipped, "actor": actor.state_dict(), "critic": critic.state_dict(),
            "actor_optimizer": actor_opt.state_dict(), "critic_optimizer": critic_opt.state_dict(),
            "training_rng": rng.bit_generator.state, "global_rng": _rebuttal_rng_state(device),
            "history": history, "previous_advantage": previous, "critic_reference": critic_reference,
            "numerical_failure": failure_record,
        })
        _write_rebuttal_json(cell / "checkpoint_index.json", {
            "path": str(cell / "checkpoint.pt"), "sha256": sha256_file(cell / "checkpoint.pt"),
            "size_bytes": (cell / "checkpoint.pt").stat().st_size, "step": completed,
            "persistence": getattr(args, "large_file_persistence", "persistent_local"), **identity,
        })

    def observe(step: int) -> None:
        nonlocal previous, critic_reference
        row, previous = collect_rebuttal_diagnostics(actor, critic, context, method, mode, step, previous, device)
        current = torch.cat([p.detach().cpu().flatten() for p in critic.parameters()])
        row.update({
            "critic_update_count": critic_updates,
            "critic_parameter_change_since_diagnostic": float((current - critic_reference).double().norm()),
            "critic_mc_validation_reference_mse": _rebuttal_mc_reference_mse(critic, context, device),
            "critic_td_loss": float(last_critic["loss"]) if last_critic["loss"] is not None else None,
        })
        critic_reference = current
        history["diagnostics"].append(row)
        if step in settings["variance_steps"]:
            history["gradient_variance"].extend(collect_rebuttal_gradient_variance(actor, critic, context, mode, method, step, seed, device))

    def rollout(step: int) -> None:
        history["rollouts"].append(_rebuttal_rollout_row(actor, context, cell, step, seed, device))

    try:
        if start == 0 and not history["diagnostics"]:
            observe(0)
            rollout(0)
            checkpoint()
        steps = range(start + 1, settings["actor_steps"] + 1) if failure is None else ()
        for step in steps:
            ids = sample_indices(rng, context["train_ids"], settings["actor_batch_size"])
            obs, next_obs = tensor(context["obs"][ids], device), tensor(context["next_obs"][ids], device)
            rewards = tensor(data.rewards[ids], device)
            done = torch.as_tensor(data.terminals[ids] | data.timeouts[ids], device=device)
            if mode == "frozen":
                advantage = torch.as_tensor(context["initial_advantage"][ids], device=device)
            else:
                advantage, _ = current_rebuttal_advantage(critic, obs, next_obs, rewards, done, context["target_norm"], base.gamma, context["center"], context["scale"])
            logp, d, _ = current_action_log_prob(actor, obs, tensor(data.actions[ids], device))
            coefficient = rebuttal_actor_coefficients(advantage, logp, d, torch.as_tensor(data.behavior_log_prob[ids], device=device), method, settings)
            loss = rebuttal_actor_loss(logp, coefficient)
            actor_opt.zero_grad(set_to_none=True)
            if not bool(torch.isfinite(loss)):
                failure = "nonfinite_actor_loss"
                break
            loss.backward()
            norm = _clip_rebuttal_gradients(actor.parameters(), settings["actor_gradient_clip"])
            if not bool(torch.isfinite(norm)):
                failure = "nonfinite_actor_gradient"
                actor_opt.zero_grad(set_to_none=True)
                break
            clipped += int(norm > settings["actor_gradient_clip"])
            post_clip_norm = (
                float(torch.sqrt(sum(p.grad.detach().double().square().sum() for p in actor.parameters())))
                if step % settings["log_interval"] == 0 or step == settings["actor_steps"] else None
            )
            actor_opt.step()
            actor_opt.zero_grad(set_to_none=True)
            completed = step
            if mode == "refreshed":
                last_critic = update_rebuttal_critic(critic, critic_opt, obs, next_obs, rewards, done, context["target_norm"], base.gamma, settings)
                if not last_critic["finite"]:
                    failure = "nonfinite_critic_update"
                    break
                critic_updates += 1
            if step % settings["log_interval"] == 0 or step == settings["actor_steps"]:
                history["training_metrics"].append(_rebuttal_json({
                    "step": step, "loss": float(loss.detach()),
                    "gradient_norm_pre_clip": float(norm),
                    "gradient_norm_post_clip": post_clip_norm,
                    "clipped_updates": clipped, "clipping_fraction": clipped / step,
                    "critic_update_count": critic_updates,
                    "critic_td_loss": float(last_critic["loss"]) if last_critic["loss"] is not None else None,
                }))
                flush()
                print(json.dumps({"seed": seed, "advantage_mode": mode, "method": method, "step": step}), flush=True)
            if step % settings["diagnostic_interval"] == 0 or step == settings["actor_steps"]:
                observe(step)
            if step % settings["evaluation_interval"] == 0 or step == settings["actor_steps"]:
                rollout(step)
            if step % settings["checkpoint_interval"] == 0 or step == settings["actor_steps"]:
                checkpoint()
        if failure:
            if failure_record is None:
                failure_record = {"reason": failure, "completed_actor_steps": completed, "attempted_step": step}
            failure_record = {**failure_record, "checkpoint_actor_steps": completed}
            _write_rebuttal_json(cell / "numerical_failure.json", failure_record)
            # Last finite model and all completed observations remain available.
            if history["diagnostics"][-1]["step"] != completed:
                observe(completed)
            if history["rollouts"][-1]["step"] != completed:
                rollout(completed)
            checkpoint()
        if completed == settings["actor_steps"] and failure is None:
            if start == completed:
                # Recover an interruption after the final checkpoint but before
                # summary export: publish its histories/index in this new root.
                checkpoint()
            # Full train-pool moments supplement the fixed-subset observations.
            full_a = context["initial_advantage"][context["train_ids"]] if mode == "frozen" else _rebuttal_advantage_arrays(critic, context, context["train_ids"], device)[0]
            _write_rebuttal_json(cell / "terminal_full_pool_mass.json", rebuttal_mass(full_a))
        summary = _rebuttal_branch_summary(identity, context, history, completed, critic_updates, failure_record)
        _write_rebuttal_json(cell / "summary.json", summary)
        _write_rebuttal_json(cell / "status.json", summary)
        return 0
    except BaseException:
        flush()
        record = {
            **identity, "status": "runtime_failure", "completed_actor_steps": completed,
            "traceback": traceback.format_exc(),
        }
        _write_rebuttal_json(cell / "failure.json", record)
        _write_rebuttal_json(cell / "status.json", record)
        raise


def _q2_pair_path(root: Path, seed: int, intervention: str) -> Path:
    return root / "pairs" / f"seed_{seed}" / intervention


def train_rebuttal_q2_pair(args: argparse.Namespace) -> int:
    settings, base = load_rebuttal_config(args.config, args.smoke)
    torch.set_num_threads(settings["worker_threads"])
    device, root, seed = choose_device(args.device), Path(args.work_dir).resolve(), args.seed
    methods = next(tuple(pair) for pair in settings["q2"]["pairs"] if pair[0] == args.method)
    pair_root = _q2_pair_path(root, seed, methods[0])
    pair_root.mkdir(parents=True, exist_ok=True)
    cells = [_rebuttal_cell(root, seed, "frozen", method) for method in methods]
    for cell in cells:
        cell.mkdir(parents=True, exist_ok=True)
    context = _load_rebuttal_context(root / "prepare", Path(args.dataset_path), settings, base)
    calibration_path = root / "prepare" / f"q2_calibration_seed_{seed}.json"
    calibration = json.loads(calibration_path.read_text())
    identity = {"seed": seed, "advantage_mode": "frozen", "method": methods[0], "global_method": methods[1],
                "source_commit": context["prepared"]["source_commit"], "settings_identity": settings["identity"],
                "context_sha256": sha256_file(root / "prepare/context.pt"),
                "calibration_sha256": sha256_file(calibration_path)}
    identities = [{**identity, "method": method, "paired_intervention": methods[0]} for method in methods]
    seed_everything(seed)
    data = context["data"]
    initial = torch.load(root / "prepare" / f"actor_initial_seed_{seed}.pt", map_location=device, weights_only=False)["actor"]
    actors = [_new_rebuttal_policy(base, data.observations.shape[1], data.actions.shape[1]).to(device) for _ in methods]
    for actor in actors:
        actor.load_state_dict(initial)
    optimizers = [torch.optim.AdamW(actor.parameters(), lr=base.actor_lr, weight_decay=base.weight_decay) for actor in actors]
    critic = ValueNetwork(data.observations.shape[1], base.hidden_sizes, base.activation, base.init_scheme, base.init_gain).to(device)
    critic.load_state_dict(context["prepared"]["critic"])
    rng = np.random.default_rng(np.random.SeedSequence([seed, 41591]))
    histories = [{k: [] for k in ("training_metrics", "diagnostics", "rollouts", "gradient_variance")} for _ in methods]
    previous, clipped, completed, budgets, failure = [None, None], [0, 0], 0, [], None
    block = _Q2BudgetBlock(device)
    if args.resume_from:
        old_root = _q2_pair_path(Path(args.resume_from), seed, methods[0])
        checkpoint_path = old_root / "checkpoint.pt"
        if not checkpoint_path.is_file() and (old_root / "checkpoint_index.json").is_file():
            checkpoint_path = Path(json.loads((old_root / "checkpoint_index.json").read_text())["path"])
        if checkpoint_path.is_file():
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if state["identity"] != identity or state["calibration"] != calibration:
                raise ValueError("Paired checkpoint/source/config/calibration identity mismatch")
            for index, (actor, optimizer) in enumerate(zip(actors, optimizers)):
                actor.load_state_dict(state["actors"][index])
                optimizer.load_state_dict(state["optimizers"][index])
            completed, clipped, histories, previous = state["step"], state["clipped"], state["histories"], state["previous"]
            budgets, failure = state["budgets"], state["numerical_failure"]
            block = _Q2BudgetBlock(device, state["budget_block"])
            rng.bit_generator.state = state["training_rng"]
            _restore_rebuttal_rng(state["global_rng"], device)
        marker = old_root / "numerical_failure.json"
        if marker.is_file():
            failure = json.loads(marker.read_text())
        for cell, method in zip(cells, methods):
            old_rollouts = _rebuttal_cell(Path(args.resume_from), seed, "frozen", method) / "rollouts"
            if old_rollouts.is_dir():
                shutil.copytree(old_rollouts, cell / "rollouts", dirs_exist_ok=True)

    def flush() -> None:
        write_csv(pair_root / "budget_matching.csv", budgets)
        for cell, cell_identity, history in zip(cells, identities, histories):
            for name, rows in history.items():
                write_csv(cell / f"{name}.csv", rows)
            _write_rebuttal_json(cell / "status.json", {**cell_identity, "status": "running", "step": completed, "critic_update_count": 0})
        _write_rebuttal_json(pair_root / "status.json", {**identity, "status": "running", "step": completed})

    def checkpoint() -> None:
        flush()
        path = pair_root / "checkpoint.pt"
        _atomic_rebuttal_checkpoint(path, {
            "identity": identity, "step": completed, "actors": [actor.state_dict() for actor in actors],
            "optimizers": [optimizer.state_dict() for optimizer in optimizers], "clipped": clipped,
            "histories": histories, "previous": previous, "budgets": budgets, "budget_block": block.state_dict(),
            "training_rng": rng.bit_generator.state, "global_rng": _rebuttal_rng_state(device),
            "calibration": calibration, "numerical_failure": failure,
        })
        index = {**identity, "path": str(path), "sha256": sha256_file(path), "step": completed,
                 "size_bytes": path.stat().st_size, "persistence": getattr(args, "large_file_persistence", "persistent_local"),
                 "methods": list(methods), "role": "synchronized_q2_pair"}
        _write_rebuttal_json(pair_root / "checkpoint_index.json", index)
        for cell, method in zip(cells, methods):
            _write_rebuttal_json(cell / "checkpoint_index.json", {**index, "actor_key": method})

    def observe(step: int) -> None:
        ids = context["diagnostic_ids"]
        obs, actions = tensor(context["obs"][ids], device), tensor(data.actions[ids], device)
        a = torch.as_tensor(context["initial_advantage"][ids], device=device)
        outputs = [current_action_log_prob(actor, obs, actions) for actor in actors]
        coefficients, statistics = q2_pair_coefficients(actors, [out[0] for out in outputs], [out[1] for out in outputs], a, calibration, methods[0])
        for i, (actor, method, history, output) in enumerate(zip(actors, methods, histories, outputs)):
            row, previous[i] = collect_rebuttal_diagnostics(
                actor, critic, context, method, "frozen", step, previous[i], device,
                effective_coefficient=coefficients[i].cpu().numpy(),
            )
            distance = output[1]
            far = (a < 0) & (distance > calibration["far_threshold"])
            row.update({"critic_update_count": 0, "q2_far_threshold": calibration["far_threshold"],
                        "q2_far_cap": calibration["far_cap"], "q2_fixed_threshold_far_count": int(far.sum()),
                        "q2_fixed_threshold_far_fraction": float(far.double().mean()),
                        "q2_audit_budget_source_method": methods[0],
                        **{f"q2_audit_{k}": float(v.cpu()) for k, v in statistics.items()}})
            near = (a < 0) & ~far
            for group, mask in (("near", near), ("far", far)):
                selected = mask.cpu().numpy()
                for label, values in (("raw", a.cpu().numpy()), ("audit_effective", coefficients[i].cpu().numpy())):
                    force = _rebuttal_force(actor, context["obs"][ids[selected]], data.actions[ids[selected]],
                                            values[selected], len(ids), settings["diagnostic_batch_size"], device)
                    row[f"q2_{label}_{group}_negative_force_norm"] = float(force.norm().cpu())
            history["diagnostics"].append(_rebuttal_json(row))
            if step in settings["variance_steps"]:
                history["gradient_variance"].extend(collect_rebuttal_gradient_variance(actor, critic, context, "frozen", method, step, seed, device))

    def rollout(step: int) -> None:
        for actor, cell, history in zip(actors, cells, histories):
            history["rollouts"].append(_rebuttal_rollout_row(actor, context, cell, step, seed, device))

    def emit_budget(last: list[dict]) -> None:
        if not block.count:
            return
        row = block.finish(completed)
        budgets.append(row)
        for i, history in enumerate(histories):
            prefix = "intervention" if i == 0 else "global"
            history["training_metrics"].append({"step": completed, "critic_update_count": 0,
                "coefficient_scope": "actual_training_batches_block_mean", "p_effective": row[f"q2_{prefix}_p_mean"],
                "q_effective": row[f"q2_{prefix}_q_mean"], "clipped_updates": clipped[i],
                "clipping_fraction": clipped[i] / completed, **last[i]})

    try:
        if calibration["status"] == "defined":
            if not histories[0]["diagnostics"]:
                observe(0)
                rollout(0)
                checkpoint()
            last = [{}, {}]
            steps = range(completed + 1, settings["actor_steps"] + 1) if failure is None else ()
            for step in steps:
                ids = sample_indices(rng, context["train_ids"], settings["actor_batch_size"])
                obs, actions = tensor(context["obs"][ids], device), tensor(data.actions[ids], device)
                a = torch.as_tensor(context["initial_advantage"][ids], device=device)
                outputs = [current_action_log_prob(actor, obs, actions) for actor in actors]
                coefficients, statistics = q2_pair_coefficients(actors, [out[0] for out in outputs], [out[1] for out in outputs], a, calibration, methods[0])
                if not bool(torch.isfinite(torch.stack(list(statistics.values()))).all()):
                    failure = {"reason": "nonfinite_q2_budget", "affected_methods": [methods[0] if not bool(torch.isfinite(statistics["target_budget"])) else methods[1]]}
                    break
                losses = [rebuttal_actor_loss(out[0], c) for out, c in zip(outputs, coefficients)]
                bad = [method for method, loss in zip(methods, losses) if not bool(torch.isfinite(loss))]
                if bad:
                    failure = {"reason": "nonfinite_actor_loss", "affected_methods": bad}
                    break
                norms = []
                for actor, optimizer, loss in zip(actors, optimizers, losses):
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    norms.append(_clip_rebuttal_gradients(actor.parameters(), settings["actor_gradient_clip"]))
                bad = [method for method, norm in zip(methods, norms) if not bool(torch.isfinite(norm))]
                if bad:
                    failure = {"reason": "nonfinite_actor_gradient", "affected_methods": bad}
                    break
                log_step = step % settings["log_interval"] == 0 or step == settings["actor_steps"]
                for i, (actor, optimizer, norm, loss) in enumerate(zip(actors, optimizers, norms, losses)):
                    clipped[i] += int(norm > settings["actor_gradient_clip"])
                    if log_step:
                        last[i] = {"loss": float(loss.detach()), "gradient_norm_pre_clip": float(norm),
                                   "gradient_norm_post_clip": float(torch.sqrt(sum(p.grad.detach().double().square().sum() for p in actor.parameters())))}
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                completed = step
                block.add(statistics)
                if log_step:
                    emit_budget(last)
                    flush()
                    print(json.dumps({**identity, "step": step, "event": "pair_progress"}), flush=True)
                if step % settings["diagnostic_interval"] == 0 or step == settings["actor_steps"]:
                    observe(step)
                if step % settings["evaluation_interval"] == 0 or step == settings["actor_steps"]:
                    rollout(step)
                if step % settings["checkpoint_interval"] == 0 or step == settings["actor_steps"]:
                    checkpoint()
            if failure:
                failure = {"observed_completed_actor_steps": completed, "attempted_step": completed + 1, **failure,
                           "checkpoint_actor_steps": completed}
                _write_rebuttal_json(pair_root / "numerical_failure.json", failure)
                emit_budget(last)
                if histories[0]["diagnostics"][-1]["step"] != completed:
                    observe(completed)
                if histories[0]["rollouts"][-1]["step"] != completed:
                    rollout(completed)
                checkpoint()
            elif completed == settings["actor_steps"] and not (pair_root / "checkpoint.pt").is_file():
                checkpoint()
        else:
            _write_rebuttal_json(pair_root / "calibration_status.json", calibration)
            if calibration["status"] == "nonfinite_calibration":
                failure = {"reason": "nonfinite_q2_calibration", "affected_methods": list(methods),
                           "observed_completed_actor_steps": 0, "checkpoint_actor_steps": 0, "attempted_step": 0}
                _write_rebuttal_json(pair_root / "numerical_failure.json", failure)
                checkpoint()
        statuses = []
        for cell, method, cell_identity, history in zip(cells, methods, identities, histories):
            failed = failure and method in failure["affected_methods"]
            record = {"reason": failure["reason"], "completed_actor_steps": failure["observed_completed_actor_steps"]} if failed else None
            summary = _rebuttal_branch_summary(cell_identity, context, history, completed, 0, record)
            if failure and not failed:
                summary.update(status="paired_stop", late_mean=None, terminal_classification="paired_stop")
            if calibration["status"] == "undefined_no_initial_negative":
                summary.update(status="calibration_undefined", terminal_classification="calibration_undefined")
            summary.update(pair_failure=failure, pair_checkpoint_index=str(pair_root / "checkpoint_index.json"))
            if completed == settings["actor_steps"] and failure is None:
                _write_rebuttal_json(cell / "terminal_full_pool_mass.json", rebuttal_mass(context["initial_advantage"][context["train_ids"]]))
            _write_rebuttal_json(cell / "summary.json", summary)
            _write_rebuttal_json(cell / "status.json", summary)
            if failed:
                _write_rebuttal_json(cell / "numerical_failure.json", failure)
            statuses.append(summary["status"])
        _write_rebuttal_json(pair_root / "summary.json", {**identity, "step": completed, "cell_statuses": statuses,
                            "status": "completed" if all(status == "completed" for status in statuses) else "stopped",
                            "numerical_failure": failure, "calibration": calibration})
        return 0
    except BaseException:
        flush()
        record = {**identity, "status": "runtime_failure", "completed_actor_steps": completed, "traceback": traceback.format_exc()}
        _write_rebuttal_json(pair_root / "failure.json", record)
        for cell, cell_identity in zip(cells, identities):
            cell_record = {**record, **cell_identity}
            _write_rebuttal_json(cell / "failure.json", cell_record)
            _write_rebuttal_json(cell / "status.json", cell_record)
        raise


def _read_rebuttal_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _plot_rebuttal(root: Path, tables: dict[str, list[dict]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = root / "figures"
    output.mkdir(exist_ok=True)
    specifications = {
        "pq_raw_and_effective": ("diagnostics", ["p_raw", "q_raw", "p_effective", "q_effective"]),
        "advantage_change_and_sign_flips": ("diagnostics", ["initial_rms_change", "initial_pos_to_neg_fraction", "initial_neg_to_pos_fraction"]),
        "far_negative_force_and_remoteness": ("diagnostics", ["raw_far_negative_force_norm", "effective_far_negative_force_norm", "negative_distance_mean"]),
        "is_weights_ess_and_gradient_variance": ("diagnostics", ["ess_fraction", "log_ratio_p99", "variance_trace"]),
        "hopper_return_curves": ("rollouts", ["normalized_return"]),
        "q2_return_curves": ("rollouts", ["normalized_return"]),
        "q2_budget_matching": ("q2_budgets", ["q2_target_budget_mean", "q2_actual_global_budget_mean", "q2_absolute_error_max"]),
        "q2_far_negative_force": ("diagnostics", ["q2_raw_far_negative_force_norm", "q2_audit_effective_far_negative_force_norm"]),
    }
    for name, (table, metrics) in specifications.items():
        fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 3.1 * len(metrics)), squeeze=False)
        for ax, metric in zip(axes[:, 0], metrics):
            rows = tables["gradient_variance"] if metric == "variance_trace" else tables[table]
            for mode in (("frozen",) if name.startswith("q2_") else REBUTTAL_MODES):
                methods = REBUTTAL_METHODS + Q2_METHODS if name.startswith("q2_") else REBUTTAL_METHODS
                for method in methods:
                    buckets: dict[int, list[float]] = {}
                    for row in rows:
                        if row["advantage_mode"] != mode:
                            continue
                        if metric == "variance_trace":
                            # Same IS checkpoint, three offline weighting rules.
                            if row["method"] != "is" or row["rule"] != method:
                                continue
                        elif row["method"] != method:
                            continue
                        raw = row.get(metric)
                        if raw in (None, ""):
                            continue
                        value = float(raw)
                        if math.isfinite(value):
                            buckets.setdefault(int(row["step"]), []).append(value)
                    if not buckets:
                        continue
                    steps = sorted(buckets)
                    mean = np.array([np.mean(buckets[s]) for s in steps])
                    color = f"C{methods.index(method)}"
                    ax.plot(np.array(steps) / 1e6, mean, color=color,
                            linestyle="-" if mode == "frozen" else "--", label=f"{mode}/{method}")
            ax.set(xlabel="Actor updates (millions)", ylabel=metric.replace("_", " "))
            if metric == "variance_trace":
                ax.set_title("Rules compared at the same trained IS checkpoints; pre-clipping")
            ax.grid(alpha=0.2)
            if ax.lines:
                ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(output / f"{name}.svg")
        plt.close(fig)


def _aggregate_rebuttal_q2(root: Path, settings: dict, matrix: list[dict], tables: dict) -> None:
    output = root / "summary"
    budgets = []
    for seed in settings["seeds"]:
        for intervention, control in settings["q2"]["pairs"]:
            budgets.extend({**row, "seed": seed, "advantage_mode": "frozen", "method": intervention,
                            "global_method": control}
                           for row in _read_rebuttal_csv(_q2_pair_path(root, seed, intervention) / "budget_matching.csv"))
    tables["q2_budgets"] = budgets
    write_csv(output / "Q2_BUDGET_MATCHING.csv", budgets)
    common = {"seed", "advantage_mode", "method", "step", "sample_count", "effective_coefficient_scope"}
    write_csv(output / "Q2_INTERVENTION_DIAGNOSTICS.csv", [
        {k: v for k, v in row.items() if k in common or k.startswith(("q2_", "p_", "q_", "rho_", "ratio_status_"))}
        for row in tables["diagnostics"] if row["method"] in Q2_METHODS
    ])
    rows = []
    for method in REBUTTAL_METHODS + Q2_METHODS:
        cells = [r for r in matrix if r["advantage_mode"] == "frozen" and r["method"] == method]
        row = {"method": method, "advantage_mode": "frozen", "planned_n": len(cells),
               "completed_n": sum(r["status"] == "completed" for r in cells),
               "numerical_failure_n": sum(bool(r.get("numerical_failure")) or r["status"] == "numerical_failure" for r in cells),
               "paired_stop_n": sum(r["status"] == "paired_stop" for r in cells),
               "runtime_failure_n": sum(r["status"] == "runtime_failure" for r in cells),
               "calibration_undefined_n": sum(r["status"] == "calibration_undefined" for r in cells),
               "source": "existing_frozen_cell" if method in REBUTTAL_METHODS else "q2_paired_cell"}
        for metric in ("late_mean", "terminal"):
            values = [r[metric] for r in cells if r["status"] == "completed" and r.get(metric) is not None]
            row.update({f"{metric}_n": len(values), f"{metric}_mean": float(np.mean(values)) if values else None,
                        f"{metric}_seed_sample_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None})
        pair_rows = [r for r in budgets if method in (r["method"], r["global_method"])]
        if pair_rows:
            total = sum(int(r["updates"]) for r in pair_rows)
            row["q2_budget_observed_steps"] = total
            for field in ("matchable", "unmatched_zero_raw_gradient", "zero_target", "alpha_gt_one", "cap_triggered"):
                row[f"q2_{field}_fraction"] = sum(int(r[f"q2_{field}_steps"]) for r in pair_rows) / total
            row["q2_absolute_error_max"] = max(float(r["q2_absolute_error_max"]) for r in pair_rows)
            relative_errors = [float(r["q2_relative_error_nonzero_target_max"]) for r in pair_rows if r["q2_relative_error_nonzero_target_max"] not in (None, "")]
            row["q2_relative_error_nonzero_target_max"] = max(relative_errors) if relative_errors else None
        rows.append(row)
    write_csv(output / "Q2_METHOD_SUMMARY.csv", rows)
    comparisons = [(intervention, reference) for intervention, control in settings["q2"]["pairs"]
                   for reference in ("signed", control)]
    paired = []
    by_key = {(r["seed"], r["method"]): r for r in matrix if r["advantage_mode"] == "frozen"}
    for method, reference in comparisons:
        for metric in ("late_mean", "terminal"):
            differences = []
            for seed in settings["seeds"]:
                cell, baseline = by_key[seed, method], by_key[seed, reference]
                difference = cell[metric] - baseline[metric] if cell["status"] == baseline["status"] == "completed" and cell.get(metric) is not None and baseline.get(metric) is not None else None
                if difference is not None:
                    differences.append(difference)
                paired.append({"row_type": "seed", "seed": seed, "method": method, "reference": reference,
                               "metric": metric, "difference": difference, "method_status": cell["status"], "reference_status": baseline["status"]})
            paired.append({"row_type": "aggregate", "method": method, "reference": reference, "metric": metric,
                           "paired_n": len(differences), "mean_difference": float(np.mean(differences)) if differences else None,
                           "difference_sample_sd": float(np.std(differences, ddof=1)) if len(differences) > 1 else None})
    write_csv(output / "Q2_PAIRED_COMPARISONS.csv", paired)


def aggregate_rebuttal(root: Path, settings: dict) -> dict:
    summary_root = root / "summary"
    summary_root.mkdir(exist_ok=True)
    matrix, tables = [], {name: [] for name in ("diagnostics", "rollouts", "gradient_variance")}
    for key in rebuttal_matrix(settings["seeds"], settings["methods_by_advantage_mode"]):
        cell = _rebuttal_cell(root, key["seed"], key["advantage_mode"], key["method"])
        path = cell / "summary.json"
        row = json.loads(path.read_text()) if path.is_file() else {**key, "status": "runtime_failure", "late_mean": None, "terminal": None}
        failure_path = cell / "failure.json"
        if failure_path.is_file():
            failure = json.loads(failure_path.read_text())
            row = {**row, "status": "runtime_failure", "worker_returncode": failure.get("returncode"),
                   "runtime_failure_log": failure.get("log")}
        matrix.append(row)
        for name, rows in tables.items():
            rows.extend({**r, **key} for r in _read_rebuttal_csv(cell / f"{name}.csv"))
    write_csv(summary_root / "MATRIX.csv", matrix)
    write_csv(summary_root / "PER_RUN_SUMMARY.csv", matrix)
    method_rows, paired = [], []
    for mode in REBUTTAL_MODES:
        for method in REBUTTAL_METHODS:
            cells = [r for r in matrix if r["advantage_mode"] == mode and r["method"] == method]
            row = {"advantage_mode": mode, "method": method, "planned_n": len(cells),
                   "completed_n": sum(r["status"] == "completed" for r in cells),
                   "numerical_failure_n": sum(r["status"] == "numerical_failure" or bool(r.get("numerical_failure")) for r in cells),
                   "runtime_failure_n": sum(r["status"] == "runtime_failure" for r in cells)}
            for metric in ("late_mean", "terminal"):
                values = [r[metric] for r in cells if r.get(metric) is not None and r["status"] == "completed"]
                row[f"{metric}_n"] = len(values)
                row[f"{metric}_mean"] = float(np.mean(values)) if values else None
                row[f"{metric}_seed_sample_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
            method_rows.append(row)
            if method == "positive_only":
                continue
            for metric in ("late_mean", "terminal"):
                differences = []
                for cell in cells:
                    reference = next(r for r in matrix if r["advantage_mode"] == mode and r["method"] == "positive_only" and r["seed"] == cell["seed"])
                    if cell.get(metric) is not None and reference.get(metric) is not None and cell["status"] == reference["status"] == "completed":
                        differences.append(cell[metric] - reference[metric])
                paired.append({"advantage_mode": mode, "method": method, "reference": "positive_only", "metric": metric,
                               "paired_n": len(differences), "mean_difference": float(np.mean(differences)) if differences else None,
                               "difference_sample_sd": float(np.std(differences, ddof=1)) if len(differences) > 1 else None})
    write_csv(summary_root / "METHOD_SUMMARY.csv", method_rows)
    write_csv(summary_root / "PAIRED_COMPARISONS.csv", paired)
    field_sets = {
        "PQ_TRAJECTORIES": lambda k: k.startswith(("p_", "q_", "rho_", "ratio_status_")),
        "ADVANTAGE_REFRESH": lambda k: k.startswith(("initial_", "previous_", "advantage_", "raw_delta_", "critic_")) or k in ("positive_fraction", "negative_fraction", "zero_fraction"),
        "GRADIENT_GEOMETRY": lambda k: "force_norm" in k or "norm_share" in k or "distance" in k or "_D_" in k or "abs_advantage" in k or "abs_coefficient" in k or k.endswith("_count") or "log_std" in k or "action_mean_boundary" in k,
        "IS_WEIGHT_STATS": lambda k: k.startswith(("log_ratio_", "ratio_", "ess", "top_", "boundary_")),
    }
    common = {"seed", "advantage_mode", "method", "step", "sample_count", "diagnostic_ids_sha256"}
    for name, include in field_sets.items():
        write_csv(summary_root / f"{name}.csv", [{k: v for k, v in r.items() if k in common or include(k)} for r in tables["diagnostics"]])
    write_csv(summary_root / "GRADIENT_VARIANCE.csv", tables["gradient_variance"])
    _aggregate_rebuttal_q2(root, settings, matrix, tables)
    terminal = sum(r["status"] in ("completed", "numerical_failure", "paired_stop", "calibration_undefined") for r in matrix)
    audit = {
        "experiment_id": REBUTTAL_EXPERIMENT_ID, "expected_cells": len(matrix),
        "terminal_cells": terminal, "completed_cells": sum(r["status"] == "completed" for r in matrix),
        "numerical_failure_cells": sum(r["status"] == "numerical_failure" or bool(r.get("numerical_failure")) for r in matrix),
        "runtime_failure_cells": len(matrix) - terminal,
        "paired_stop_cells": sum(r["status"] == "paired_stop" for r in matrix),
        "calibration_undefined_cells": sum(r["status"] == "calibration_undefined" for r in matrix),
        "support_or_variance_boundary_cells": sum(bool(r.get("support_or_variance_boundary_event")) for r in matrix),
        "task_collapse_status": "not_adjudicated_no_coldstart_threshold",
        "all_full_budgets_completed": all(r["status"] == "completed" and r["completed_actor_steps"] == settings["actor_steps"] for r in matrix),
        "all_late_windows_complete": all(r.get("late_count", 0) == len(settings["late_steps"]) for r in matrix),
        "terminal_classification": "fixed_horizon_inconclusive_no_steady_state_claim",
        "status": "PASS" if terminal == len(matrix) else "PARTIAL",
        "status_meaning": "matrix/evidence completeness only, not convergence or method superiority",
        "smoke": settings["smoke"],
    }
    _write_rebuttal_json(summary_root / "TERMINAL_AUDIT.json", audit)
    _write_rebuttal_json(root / "terminal_audit.json", audit)
    _plot_rebuttal(root, tables)
    return audit


def _reuse_rebuttal_job(resume: Path, root: Path, job: dict, settings: dict, source_commit: str) -> bool:
    old_cells = [_rebuttal_cell(resume, key["seed"], key["advantage_mode"], key["method"]) for key in job["cells"]]
    paired = job["job_kind"] == "q2_pair"
    old_pair = _q2_pair_path(resume, job["seed"], job["method"])
    if paired and ((old_pair / "failure.json").is_file() or not (old_pair / "summary.json").is_file()):
        return False
    for old, key in zip(old_cells, job["cells"]):
        if not (old / "summary.json").is_file() or (old / "failure.json").is_file():
            return False
        row = json.loads((old / "summary.json").read_text())
        if row["status"] not in (("completed",) if paired else ("completed", "numerical_failure")):
            return False
        if (row["settings_identity"] != settings["identity"] or row["source_commit"] != source_commit
                or any(row[k] != v for k, v in key.items())):
            raise ValueError("Completed resume cell identity mismatch")
    for old, key in zip(old_cells, job["cells"]):
        cell = _rebuttal_cell(root, key["seed"], key["advantage_mode"], key["method"])
        cell.mkdir(parents=True, exist_ok=True)
        for path in old.iterdir():
            if path.suffix in (".csv", ".json"):
                shutil.copy2(path, cell / path.name)
            elif path.is_dir() and path.name == "rollouts":
                shutil.copytree(path, cell / path.name)
    if paired:
        pair = _q2_pair_path(root, job["seed"], job["method"])
        pair.mkdir(parents=True, exist_ok=True)
        for path in old_pair.iterdir():
            if path.suffix in (".csv", ".json"):
                shutil.copy2(path, pair / path.name)
    return True


def run_rebuttal_experiment(args: argparse.Namespace) -> int:
    import time

    settings, base = load_rebuttal_config(args.config, args.smoke)
    torch.set_num_threads(settings["worker_threads"])
    root, repo = Path(args.work_dir).resolve(), Path(args.repo_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    dataset = Path(args.dataset_path).resolve()
    source_commit = git_output(repo, "rev-parse", "HEAD")
    digest = sha256_file(dataset)
    if not args.smoke and digest != base.dataset_sha256:
        raise ValueError("This fixed dataset protocol requires the registered Hopper-replay HDF5")
    matrix = rebuttal_matrix(settings["seeds"], settings["methods_by_advantage_mode"])
    jobs = rebuttal_jobs(settings)
    manifest = {
        "experiment_id": REBUTTAL_EXPERIMENT_ID, "base_commit": source_commit,
        "source_commit": source_commit, "source_branch": git_output(repo, "branch", "--show-current"),
        "run_id": root.name, "settings_identity": settings["identity"], "settings": settings,
        "dataset": {"path": str(dataset), "sha256": digest}, "matrix": matrix,
        "jobs": jobs, "cell_count": len(matrix), "job_count": len(jobs),
        "started_utc": utc_now(), "smoke": args.smoke,
        "scientific_status": "not_run" if args.smoke else "pilot",
        "density_processing": "tanh action-space ratio with one a_safe; no ratio clipping/normalization",
        "behavior_logp_source": "infos/action_log_probs, per collection row",
        "state_occupancy_correction": False, "actor_warmstart_steps": 0,
        "environment": environment_manifest(),
        "large_file_persistence": args.large_file_persistence,
    }
    _write_rebuttal_json(root / "scientific_run_manifest.json", manifest)
    owns_run_manifest = not (root / "run_manifest.json").exists()
    if owns_run_manifest:
        _write_rebuttal_json(root / "run_manifest.json", manifest)
    _write_rebuttal_json(root / "resolved_config.json", settings)
    shutil.copy2(args.config, root / "resolved_config.yaml")
    resume = Path(args.resume_from).resolve() if args.resume_from else None
    if resume:
        old = json.loads((resume / "scientific_run_manifest.json").read_text())
        if any(old[k] != manifest[k] for k in ("source_commit", "settings_identity")) or old["dataset"]["sha256"] != digest:
            raise ValueError("Resume requires the same exact source/config/dataset; use the original commit")
        done = resume / "prepare" / "PREPARE_COMPLETE.json"
        if done.is_file():
            record = json.loads(done.read_text())
            if record["context_sha256"] != sha256_file(resume / "prepare" / "context.pt"):
                raise ValueError("The prepared recovery context checksum changed")
            for name, expected in record["prepared_input_hashes"].items():
                if sha256_file(resume / "prepare" / name) != expected:
                    raise ValueError(f"Prepared recovery input checksum changed: {name}")
            shutil.copytree(resume / "prepare", root / "prepare")
    device = choose_device(args.device)
    if device.type == "cuda":
        gpu_ids = [int(v) for v in args.gpu_ids.split(",") if v.strip()]
        if args.device.startswith("cuda:"):
            gpu_ids = [device.index]
        devices = [f"cuda:{i}" for i in gpu_ids if i < torch.cuda.device_count()]
        if not devices:
            raise ValueError("No requested CUDA slot is visible")
    else:
        devices = ["cpu"] * args.max_workers
    devices = devices[:args.max_workers]
    preflight = preflight_rollout_environment(
        backend=base.env_backend, dataset_id=base.rollout_dataset_id, env_id=base.env_id,
        expected_observation_dim=11, expected_action_dim=3, seed=settings["evaluation_seed"],
        max_steps=base.rollout_preflight_max_steps, normalized_score_percent=base.normalized_score_percent,
        reference_min_score=base.normalized_score_reference_min,
        reference_max_score=base.normalized_score_reference_max,
        output_dir=root / "rollout_preflight", required=True,
        process_isolated=True, timeout_seconds=base.rollout_preflight_timeout_seconds,
    )
    _write_rebuttal_json(root / "ROLLOUT_PREFLIGHT.json", preflight)
    if not (root / "prepare" / "PREPARE_COMPLETE.json").is_file():
        prepare_rebuttal_context(dataset, root / "prepare", settings, base, torch.device(devices[0]), source_commit)
    manifest["preparation"] = json.loads((root / "prepare/normalization.json").read_text())
    manifest["q2_calibrations"] = [json.loads((root / "prepare" / f"q2_calibration_seed_{seed}.json").read_text()) for seed in settings["seeds"]]
    _write_rebuttal_json(root / "scientific_run_manifest.json", manifest)
    if owns_run_manifest:
        _write_rebuttal_json(root / "run_manifest.json", manifest)
    pending = [job for job in jobs if not (resume and _reuse_rebuttal_job(resume, root, job, settings, source_commit))]
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    active: dict[int, tuple[subprocess.Popen, Any, dict]] = {}
    try:
        while pending or active:
            for slot, worker_device in enumerate(devices):
                if slot in active or not pending:
                    continue
                key = pending.pop(0)
                command = [
                    sys.executable, str(Path(__file__).resolve()), "rebuttal-pq-is-worker",
                    "--config", str(Path(args.config).resolve()), "--dataset-path", str(dataset),
                    "--work-dir", str(root), "--device", worker_device, "--seed", str(key["seed"]),
                    "--advantage-mode", key["advantage_mode"], "--method", key["method"],
                    "--job-kind", key["job_kind"],
                    "--large-file-persistence", args.large_file_persistence,
                ]
                if args.smoke:
                    command.append("--smoke")
                if resume:
                    command.extend(["--resume-from", str(resume)])
                log = (logs / f"seed_{key['seed']}_{key['advantage_mode']}_{key['method']}.log").open("w")
                env = os.environ.copy()
                env.update(OMP_NUM_THREADS=str(settings["worker_threads"]), MKL_NUM_THREADS=str(settings["worker_threads"]))
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
                active[slot] = (process, log, key)
                print(json.dumps({"event": "worker_started", **key, "device": worker_device}), flush=True)
            for slot, (process, log, key) in list(active.items()):
                code = process.poll()
                if code is not None:
                    log.close()
                    if code != 0:
                        failure_targets = [
                            (_rebuttal_cell(root, cell["seed"], cell["advantage_mode"], cell["method"]), {**key, **cell})
                            for cell in key["cells"]
                        ]
                        if key["job_kind"] == "q2_pair":
                            failure_targets.append((_q2_pair_path(root, key["seed"], key["method"]), key))
                        for directory, failure_identity in failure_targets:
                            failure_path = directory / "failure.json"
                            failure = json.loads(failure_path.read_text()) if failure_path.is_file() else {}
                            failure.update({**failure_identity, "status": "runtime_failure", "returncode": code, "log": str(log.name)})
                            _write_rebuttal_json(failure_path, failure)
                            _write_rebuttal_json(directory / "status.json", failure)
                    print(json.dumps({"event": "worker_exited", **key, "returncode": code}), flush=True)
                    del active[slot]
            _write_rebuttal_json(root / "scientific_heartbeat.json", {
                "utc": utc_now(), "pending_cells": len(pending), "active_cells": [key for _, _, key in active.values()],
            })
            if pending or active:
                time.sleep(0.25)
    finally:
        for process, log, _ in active.values():
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            log.close()
    audit = aggregate_rebuttal(root, settings)
    if audit["runtime_failure_cells"]:
        _write_rebuttal_json(root / "RUN_PARTIAL.json", {"experiment_id": REBUTTAL_EXPERIMENT_ID, "base_commit": source_commit, **audit})
        return 2
    _write_rebuttal_json(root / "RUN_COMPLETE.json", {
        "experiment_id": REBUTTAL_EXPERIMENT_ID, "base_commit": source_commit,
        "run_id": root.name, "execution_state": "raw_complete", "smoke": args.smoke,
        "scientific_status": "not_run" if args.smoke else "pilot", "finished_utc": utc_now(),
        "matrix_terminal_cells": audit["terminal_cells"], "numerical_failures": audit["numerical_failure_cells"],
    })
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("rebuttal-pq-is", "rebuttal-pq-is-worker"):
        profile = sub.add_parser(name, help="isolated Hopper p/q and raw-IS profile")
        profile.add_argument("--config", default="configs/e7_hopper_pq_is_rebuttal_v1.yaml")
        profile.add_argument("--dataset-path", required=True)
        profile.add_argument("--work-dir", required=True)
        profile.add_argument("--device", default="auto")
        profile.add_argument("--smoke", action="store_true")
        profile.add_argument("--resume-from")
        profile.add_argument("--large-file-persistence", choices=("persistent_local", "ephemeral", "unknown"), default="persistent_local")
        if name.endswith("-worker"):
            profile.add_argument("--seed", type=int, required=True)
            profile.add_argument("--advantage-mode", choices=REBUTTAL_MODES, required=True)
            profile.add_argument("--method", choices=REBUTTAL_METHODS + Q2_METHODS, required=True)
            profile.add_argument("--job-kind", choices=("independent", "q2_pair"), default="independent")
        else:
            profile.add_argument("--repo-root", default=".")
            profile.add_argument("--gpu-ids", default="0,1,2,3,4,5,6,7")
            profile.add_argument("--max-workers", type=int, default=8)
    run = sub.add_parser("run", help="run the automatic E7-Q2 pipeline")
    run.add_argument("--mode", choices=("pilot", "formal"), required=True)
    run.add_argument("--dataset-path", required=True)
    run.add_argument("--work-dir", required=True)
    run.add_argument(
        "--config", default="configs/e7_hopper_q2_medium_replay_v2.yaml"
    )
    run.add_argument("--repo-root", default=".")
    run.add_argument("--device", default="auto")
    run.add_argument("--run-id", default=None)
    run.add_argument(
        "--critic-artifact",
        default=None,
        help="Optional exact canonical critic artifact directory to verify and reuse",
    )
    run.add_argument("--allow-dirty", action="store_true")
    inspect = sub.add_parser("inspect", help="validate config and dataset only")
    inspect.add_argument("--dataset-path", required=True)
    inspect.add_argument(
        "--config", default="configs/e7_hopper_q2_medium_replay_v2.yaml"
    )
    inspect.add_argument("--max-transitions", type=int, default=1000)
    worker = sub.add_parser(
        "rollout-preflight-worker",
        help=argparse.SUPPRESS,
    )
    worker.add_argument("--backend", required=True)
    worker.add_argument("--dataset-id", required=True)
    worker.add_argument("--env-id", required=True)
    worker.add_argument("--expected-observation-dim", type=int, required=True)
    worker.add_argument("--expected-action-dim", type=int, required=True)
    worker.add_argument("--seed", type=int, required=True)
    worker.add_argument("--max-steps", type=int, required=True)
    worker.add_argument("--reference-min-score", type=float, required=True)
    worker.add_argument("--reference-max-score", type=float, required=True)
    worker.add_argument("--normalized-score-percent", action="store_true")
    worker.add_argument("--output-report", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "rebuttal-pq-is":
        if args.max_workers < 1:
            raise ValueError("--max-workers must be positive")
        return run_rebuttal_experiment(args)
    if args.command == "rebuttal-pq-is-worker":
        return train_rebuttal_q2_pair(args) if args.job_kind == "q2_pair" else train_rebuttal_branch(args)
    if args.command == "rollout-preflight-worker":
        report = _run_rollout_preflight_worker(
            backend=args.backend,
            dataset_id=args.dataset_id,
            env_id=args.env_id,
            expected_observation_dim=args.expected_observation_dim,
            expected_action_dim=args.expected_action_dim,
            seed=args.seed,
            max_steps=args.max_steps,
            normalized_score_percent=args.normalized_score_percent,
            reference_min_score=args.reference_min_score,
            reference_max_score=args.reference_max_score,
            output_report=Path(args.output_report).resolve(),
        )
        return 0 if report.get("status") == "passed" else 2
    if args.command == "inspect":
        config = load_config(args.config)
        path = Path(args.dataset_path).resolve()
        manifest = {
            "basename": path.name,
            "sha256": sha256_file(path),
            "expected_basename": config.dataset_basename,
            "expected_sha256": config.dataset_sha256,
        }
        data = load_hopper_hdf5(path, args.max_transitions)
        manifest.update(
            {
                "transitions_loaded": data.size,
                "episodes_loaded": int(np.max(data.episode_ids) + 1),
                "observation_dim": data.observations.shape[1],
                "action_dim": data.actions.shape[1],
            }
        )
        print(json.dumps(manifest, indent=2))
        return 0
    return run_experiment(args)


if __name__ == "__main__":
    raise SystemExit(main())
