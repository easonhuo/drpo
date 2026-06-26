#!/usr/bin/env python3
"""EXT-H-E7-Q2 Hopper Gaussian log-scale external-validation runner.

The scientific runner trains an episode-split value critic, freezes critic-derived
advantages, takes a tanh-squashed Gaussian actor through a Positive-only terminal
audit, constructs advantage-matched near/far negative probes, decomposes Gaussian
mean and log-scale output scores, and branches the preregistered signed controls.

This file never creates ZIP/TAR artifacts. Formal supervision, recovery packaging,
checksums, and uploadable result archives are owned by the canonical hardened
execution channel in ``scripts/artifact_protocol_hardened.py``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import traceback
from dataclasses import dataclass
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
RUNNER_VERSION = "3.0.0-hardened-q2"
EPS = 1e-6
METHODS = (
    "positive_only",
    "signed",
    "near_zero",
    "far_zero",
    "far_cap",
    "budget_matched_global",
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
    rollout_eval_interval: int


@dataclass(frozen=True)
class E7Config:
    experiment_id: str
    dataset_basename: str
    dataset_sha256: str
    env_id: str
    env_registration_import: str
    normalized_score_percent: bool
    formal_rollout_required: bool
    pilot_rollout_required: bool
    gamma: float
    train_fraction: float
    validation_fraction: float
    hidden_sizes: tuple[int, ...]
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
    audit_windows: int
    critic_relative_slope_tolerance: float
    critic_gradient_tolerance: float
    critic_update_tolerance: float
    actor_relative_slope_tolerance: float
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
        env_id=str(env["env_id"]),
        env_registration_import=str(env["registration_import"]),
        normalized_score_percent=bool(env["normalized_score_percent"]),
        formal_rollout_required=bool(env["formal_required"]),
        pilot_rollout_required=bool(env["pilot_required"]),
        gamma=float(raw["critic"]["gamma"]),
        train_fraction=float(raw["critic"]["episode_split"]["train"]),
        validation_fraction=float(raw["critic"]["episode_split"]["validation"]),
        hidden_sizes=tuple(int(x) for x in raw["model"]["hidden_sizes"]),
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
        audit_windows=int(raw["terminal_audit"]["windows"]),
        critic_relative_slope_tolerance=float(
            raw["terminal_audit"]["critic_relative_slope_tolerance"]
        ),
        critic_gradient_tolerance=float(raw["terminal_audit"]["critic_gradient_tolerance"]),
        critic_update_tolerance=float(raw["terminal_audit"]["critic_update_tolerance"]),
        actor_relative_slope_tolerance=float(
            raw["terminal_audit"]["actor_relative_slope_tolerance"]
        ),
        actor_gradient_tolerance=float(raw["terminal_audit"]["actor_gradient_tolerance"]),
        actor_update_tolerance=float(raw["terminal_audit"]["actor_update_tolerance"]),
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


def make_mlp(input_dim: int, output_dim: int, hidden_sizes: Sequence[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    width = input_dim
    for hidden in hidden_sizes:
        layers.extend([nn.Linear(width, hidden), nn.Tanh()])
        width = hidden
    layers.append(nn.Linear(width, output_dim))
    return nn.Sequential(*layers)


class ValueNetwork(nn.Module):
    def __init__(self, obs_dim: int, hidden_sizes: Sequence[int]):
        super().__init__()
        self.net = make_mlp(obs_dim, 1, hidden_sizes)

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
    ):
        super().__init__()
        self.mean_net = make_mlp(obs_dim, action_dim, hidden_sizes)
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


def classify_actor_terminal(
    rows: Sequence[dict[str, Any]],
    config: E7Config,
    candidate_step: int | None,
    extension_complete: bool,
) -> dict[str, Any]:
    last = rows[-1]
    nonfinite = any(
        not math.isfinite(float(last[key]))
        for key in ("loss", "positive_nll", "gradient_norm", "update_norm", "sigma_mean")
    )
    support_event = (
        float(last["mean_boundary_fraction"]) >= config.support_boundary_fraction
        or float(last["log_std_min_fraction"]) > 0.0
        or float(last["log_std_max_fraction"]) > 0.0
    )
    slopes = {
        key: relative_slope(rows, key, config.audit_windows)
        for key in ("positive_nll", "mean_abs", "sigma_mean", "phantom_distance_mean")
    }
    stable = (
        candidate_step is not None
        and extension_complete
        and all(value <= config.actor_relative_slope_tolerance for value in slopes.values())
        and float(last["gradient_norm"]) <= config.actor_gradient_tolerance
        and float(last["update_norm"]) <= config.actor_update_tolerance
        and not nonfinite
    )
    rollout_values = [
        float(row.get("normalized_return", float("nan"))) for row in rows
    ]
    finite_rollouts = [value for value in rollout_values if math.isfinite(value)]
    initial_return = finite_rollouts[0] if finite_rollouts else float("nan")
    final_return = finite_rollouts[-1] if finite_rollouts else float("nan")
    task_collapse = bool(
        math.isfinite(initial_return)
        and math.isfinite(final_return)
        and initial_return - final_return >= config.task_return_drop_threshold
    )
    if nonfinite:
        state = "nan_inf_numerical_collapse"
    elif stable and support_event:
        state = "finite_terminal_with_support_boundary_event"
    elif stable:
        state = "finite_terminal"
    elif support_event:
        state = "support_or_variance_boundary_event_without_terminal_convergence"
    elif len(rows) >= config.audit_windows and slopes["mean_abs"] > config.actor_relative_slope_tolerance:
        state = "persistent_or_slow_drift"
    else:
        state = "max_horizon_without_terminal_classification"
    return {
        "state": state,
        "candidate_step": candidate_step,
        "extension_complete": extension_complete,
        "slopes": slopes,
        "support_boundary_event": support_event,
        "numerical_nonfinite": nonfinite,
        "task_performance_collapse": task_collapse,
        "initial_normalized_return": initial_return,
        "final_normalized_return": final_return,
        "task_return_drop_threshold": config.task_return_drop_threshold,
        "reporting_separation": [
            "task_performance_collapse",
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


def evaluate_d4rl_rollouts(
    *,
    policy: SquashedGaussianPolicy,
    obs_norm: Normalizer,
    env_id: str,
    registration_import: str,
    episodes: int,
    seed: int,
    device: torch.device,
    normalized_score_percent: bool,
    required: bool,
) -> dict[str, float]:
    if episodes <= 0:
        return {
            "rollout_return_mean": float("nan"),
            "rollout_return_std": float("nan"),
            "normalized_return": float("nan"),
            "rollout_episodes": 0,
        }
    try:
        __import__(registration_import)
        try:
            import gym  # type: ignore
        except ImportError:
            import gymnasium as gym  # type: ignore
        env = gym.make(env_id)
    except Exception as exc:
        if required:
            raise RuntimeError(
                f"Rollout evaluation requires {registration_import!r} and env {env_id!r}: {exc}"
            ) from exc
        return {
            "rollout_return_mean": float("nan"),
            "rollout_return_std": float("nan"),
            "normalized_return": float("nan"),
            "rollout_episodes": 0,
            "rollout_unavailable": 1.0,
        }

    returns: list[float] = []
    try:
        for episode in range(episodes):
            episode_seed = int(seed + episode)
            try:
                reset_out = env.reset(seed=episode_seed)
            except TypeError:
                if hasattr(env, "seed"):
                    env.seed(episode_seed)
                reset_out = env.reset()
            observation = reset_out[0] if isinstance(reset_out, tuple) else reset_out
            total = 0.0
            done = False
            while not done:
                normalized = obs_norm.transform(
                    np.asarray(observation, dtype=np.float32).reshape(1, -1)
                )
                with torch.no_grad():
                    action = (
                        policy.action_mean(tensor(normalized, device))[0]
                        .detach()
                        .cpu()
                        .numpy()
                    )
                step_out = env.step(action)
                if len(step_out) == 5:
                    observation, reward, terminated, truncated, _ = step_out
                    done = bool(terminated or truncated)
                else:
                    observation, reward, done, _ = step_out
                    done = bool(done)
                total += float(reward)
            returns.append(total)
        mean_return = float(np.mean(returns))
        normalized = float("nan")
        scorer = getattr(env, "get_normalized_score", None)
        if callable(scorer):
            normalized = float(scorer(mean_return))
            if normalized_score_percent:
                normalized *= 100.0
        elif required:
            raise RuntimeError(
                f"Environment {env_id!r} does not expose get_normalized_score"
            )
        return {
            "rollout_return_mean": mean_return,
            "rollout_return_std": float(np.std(returns)),
            "normalized_return": normalized,
            "rollout_episodes": int(episodes),
        }
    finally:
        env.close()


def sample_indices(rng: np.random.Generator, pool: np.ndarray, batch_size: int) -> np.ndarray:
    replace = len(pool) < batch_size
    return rng.choice(pool, size=batch_size, replace=replace)


def tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.float32, device=device)


def full_gradient_norm(loss: torch.Tensor, parameters: Iterable[nn.Parameter]) -> float:
    grads = torch.autograd.grad(loss, list(parameters), retain_graph=False, allow_unused=True)
    total = torch.zeros((), device=loss.device)
    for grad in grads:
        if grad is not None:
            total = total + grad.detach().square().sum()
    return float(torch.sqrt(total).cpu())


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
    output_dir.mkdir(parents=True, exist_ok=True)
    obs = obs_norm.transform(data.observations)
    target_norm = Normalizer.fit(returns[split["train"]].reshape(-1, 1))
    normalized_targets = target_norm.transform(returns.reshape(-1, 1)).reshape(-1)
    model = ValueNetwork(obs.shape[1], config.hidden_sizes).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.critic_lr, weight_decay=config.weight_decay
    )
    rng = np.random.default_rng(seed + 1000)
    rows: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_step = 0
    best_path = output_dir / "best_critic.pt"
    candidate_step: int | None = None
    extension_target: int | None = None
    validation_audit = rng.choice(
        split["validation"],
        size=min(mode.audit_sample_size, len(split["validation"])),
        replace=False,
    )
    eval_snapshot = [parameter.detach().clone() for parameter in model.parameters()]
    last_eval_step = 0

    def evaluate(step: int, update_norm: float) -> dict[str, Any]:
        model.eval()
        result: dict[str, Any] = {"step": step, "update_norm_per_step": update_norm}
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
        audit_pred = model(tensor(obs[validation_audit], device))
        audit_target = tensor(normalized_targets[validation_audit], device)
        audit_loss = F.mse_loss(audit_pred, audit_target)
        result["validation_audit_loss_normalized"] = float(audit_loss.detach().cpu())
        result["validation_gradient_norm"] = full_gradient_norm(
            audit_loss, model.parameters()
        )
        model.train()
        return result

    model.train()
    for step in range(1, mode.critic_max_steps + 1):
        idx = sample_indices(rng, split["train"], config.critic_batch_size)
        pred = model(tensor(obs[idx], device))
        loss = F.mse_loss(pred, tensor(normalized_targets[idx], device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        if step % mode.critic_eval_interval == 0 or step == mode.critic_max_steps:
            update_sq = 0.0
            with torch.no_grad():
                for previous, current in zip(eval_snapshot, model.parameters()):
                    update_sq += float((current - previous).square().sum().cpu())
                eval_snapshot = [parameter.detach().clone() for parameter in model.parameters()]
            update_norm = math.sqrt(update_sq) / max(step - last_eval_step, 1)
            last_eval_step = step
            row = evaluate(step, update_norm)
            row["train_batch_loss_normalized"] = float(loss.detach().cpu())
            rows.append(row)
            if heartbeat is not None:
                heartbeat("critic", step)
            emit_event(
                {
                    "stage": "critic",
                    "step": step,
                    "validation_mse": row["validation_mse"],
                    "test_r2": row["test_r2"],
                    "validation_gradient_norm": row["validation_gradient_norm"],
                    "update_norm_per_step": row["update_norm_per_step"],
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
                    },
                    best_path,
                )
            slope = relative_slope(rows, "validation_mse", config.audit_windows)
            if (
                candidate_step is None
                and step >= mode.critic_min_steps
                and slope <= config.critic_relative_slope_tolerance
                and float(row["validation_gradient_norm"])
                <= config.critic_gradient_tolerance
                and float(row["update_norm_per_step"])
                <= config.critic_update_tolerance
            ):
                candidate_step = step
                extension_target = min(mode.critic_max_steps, 2 * step)
            if extension_target is not None and step >= extension_target:
                break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    selected_metrics = evaluate(best_step, 0.0)
    model.eval()
    extension_complete = bool(
        candidate_step is not None and rows[-1]["step"] >= 2 * candidate_step
    )
    audit = {
        "best_step": best_step,
        "best_validation_mse": best_loss,
        "candidate_step": candidate_step,
        "extension_target": extension_target,
        "extension_complete": extension_complete,
        "validation_mse_relative_slope": relative_slope(
            rows, "validation_mse", config.audit_windows
        ),
        "final_validation_gradient_norm": rows[-1]["validation_gradient_norm"],
        "final_update_norm_per_step": rows[-1]["update_norm_per_step"],
        "optimization_terminal": bool(candidate_step is not None and extension_complete),
        "selected_checkpoint_metrics": selected_metrics,
        "statistical_note": (
            "Optimization convergence does not imply a ground-truth value function; "
            "held-out R2/Pearson are reported to expose critic error."
        ),
        "checkpoint": {
            "path": str(best_path),
            "sha256": sha256_file(best_path),
            "size_bytes": best_path.stat().st_size,
        },
        "final_training_metrics": rows[-1],
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
    obs = obs_norm.transform(data.observations)
    next_obs = obs_norm.transform(data.next_observations)
    values: list[np.ndarray] = []
    next_values: list[np.ndarray] = []
    critic.eval()
    with torch.no_grad():
        for start in range(0, data.size, 65536):
            stop = min(data.size, start + 65536)
            values.append(critic(tensor(obs[start:stop], device)).cpu().numpy())
            next_values.append(critic(tensor(next_obs[start:stop], device)).cpu().numpy())
    value_norm = np.concatenate(values).astype(np.float32)
    next_value_norm = np.concatenate(next_values).astype(np.float32)
    value = value_norm * float(target_norm.std[0]) + float(target_norm.mean[0])
    next_value = next_value_norm * float(target_norm.std[0]) + float(target_norm.mean[0])
    bootstrap_mask = (~(data.terminals | data.timeouts)).astype(np.float32)
    raw = data.rewards + gamma * bootstrap_mask * next_value - value
    center = float(np.mean(raw[standardization_indices]))
    scale = float(np.std(raw[standardization_indices]))
    if standardize:
        advantage = ((raw - center) / max(scale, 1e-8)).astype(np.float32)
    else:
        advantage = raw.astype(np.float32)
        center, scale = 0.0, 1.0
    path = output_dir / "frozen_advantages.npz"
    np.savez_compressed(
        path,
        advantage=advantage,
        raw_advantage=raw.astype(np.float32),
        value=value,
        next_value=next_value,
        center=np.asarray(center, dtype=np.float32),
        scale=np.asarray(scale, dtype=np.float32),
    )
    manifest = {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "count": len(advantage),
        "standardized_once": standardize,
        "standardization_population": "critic_train_episode_split_only",
        "standardization_count": int(len(standardization_indices)),
        "center": center,
        "scale": scale,
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
    update_norm: float,
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
            "update_norm": update_norm,
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
                "rollout_return_mean": float("nan"),
                "rollout_return_std": float("nan"),
                "normalized_return": float("nan"),
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
    elif method == "budget_matched_global":
        weights = torch.where(negative, weights * global_scale, weights)
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
    rollout_evaluator: Callable[[SquashedGaussianPolicy, int], dict[str, float]] | None = None,
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
    last_diag: dict[str, float] = {}
    eval_snapshot = [parameter.detach().clone() for parameter in policy.parameters()]
    last_eval_step = 0

    def evaluate(step: int, update_norm_per_step: float) -> dict[str, Any]:
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
        audit_gradient = full_gradient_norm(audit_loss, policy.parameters())
        row = actor_eval_metrics(
            policy=policy,
            obs=obs,
            actions=actions,
            advantages=advantages,
            audit_indices=audit_indices,
            fixed_negative_indices=fixed_negative_indices,
            device=device,
            loss_value=float(audit_loss.detach().cpu()),
            gradient_norm=audit_gradient,
            update_norm=update_norm_per_step,
            step=step,
            boundary_threshold=config.support_boundary_threshold,
            rollout_metrics=(
                rollout_evaluator(policy, step)
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
    initial_row = evaluate(0, 0.0)
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
        loss.backward()
        train_batch_gradient = float(
            torch.nn.utils.clip_grad_norm_(
                policy.parameters(), config.max_gradient_norm
            ).cpu()
        )
        optimizer.step()
        train_batch_loss = float(loss.detach().cpu())
        if not math.isfinite(train_batch_loss) or not math.isfinite(train_batch_gradient):
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
                update_norm=float("nan"),
                step=step,
                boundary_threshold=config.support_boundary_threshold,
            )
            row.update(last_diag)
            rows.append(row)
            if heartbeat is not None:
                heartbeat(f"actor:{method}", step)
            emit_event(
                {
                    "stage": f"actor:{method}",
                    "step": step,
                    "train_batch_loss": train_batch_loss,
                    "numerical_nonfinite": True,
                }
            )
            break
        if step % eval_interval == 0 or step == max_steps:
            update_sq = 0.0
            with torch.no_grad():
                for previous, current in zip(eval_snapshot, policy.parameters()):
                    update_sq += float((current - previous).square().sum().cpu())
                eval_snapshot = [parameter.detach().clone() for parameter in policy.parameters()]
            update_norm = math.sqrt(update_sq) / max(step - last_eval_step, 1)
            last_eval_step = step
            row = evaluate(step, update_norm)
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
                    "update_norm_per_step": row["update_norm"],
                }
            )
            slopes = [
                relative_slope(rows, key, config.audit_windows)
                for key in ("positive_nll", "mean_abs", "sigma_mean", "phantom_distance_mean")
            ]
            if (
                candidate_step is None
                and step >= min_steps
                and all(x <= config.actor_relative_slope_tolerance for x in slopes)
                and float(row["gradient_norm"]) <= config.actor_gradient_tolerance
                and float(row["update_norm"]) <= config.actor_update_tolerance
            ):
                candidate_step = step
                extension_target = min(max_steps, 2 * step)
            if extension_target is not None and step >= extension_target:
                break

    if rollout_evaluator and not math.isfinite(float(rows[-1].get("normalized_return", float("nan")))):
        rows[-1].update(rollout_evaluator(policy, int(rows[-1]["step"])))

    checkpoint_path = output_dir / "terminal_actor.pt"
    torch.save(
        {
            "model": policy.state_dict(),
            "method": method,
            "step": rows[-1]["step"],
            "far_threshold": far_threshold,
            "global_scale": global_scale,
            "far_cap_score": far_cap_score,
        },
        checkpoint_path,
    )
    extension_complete = bool(
        candidate_step is not None and rows[-1]["step"] >= 2 * candidate_step
    )
    audit = classify_actor_terminal(rows, config, candidate_step, extension_complete)
    audit.update(
        {
            "method": method,
            "final_step": rows[-1]["step"],
            "extension_target": extension_target,
            "far_threshold": far_threshold,
            "global_scale": global_scale,
            "far_cap_score": far_cap_score,
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": sha256_file(checkpoint_path),
                "size_bytes": checkpoint_path.stat().st_size,
            },
            "final_metrics": rows[-1],
        }
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
    ).to(device)
    clone.load_state_dict(policy.state_dict())
    return clone


def run_seed(
    *,
    seed: int,
    data: OfflineData,
    config: E7Config,
    mode: ModeConfig,
    device: torch.device,
    seed_dir: Path,
    heartbeat: Callable[[str, int], None] | None = None,
    formal_mode: bool = False,
) -> dict[str, Any]:
    seed_everything(seed)
    seed_dir.mkdir(parents=True, exist_ok=True)
    split = split_episode_indices(
        data.episode_ids, seed, config.train_fraction, config.validation_fraction
    )
    obs_norm = Normalizer.fit(data.observations[split["train"]])
    rollout_required = config.formal_rollout_required if formal_mode else config.pilot_rollout_required

    def rollout_evaluator(policy: SquashedGaussianPolicy, step: int) -> dict[str, float]:
        return evaluate_d4rl_rollouts(
            policy=policy,
            obs_norm=obs_norm,
            env_id=config.env_id,
            registration_import=config.env_registration_import,
            episodes=mode.rollout_episodes,
            seed=seed * 100000 + step,
            device=device,
            normalized_score_percent=config.normalized_score_percent,
            required=rollout_required,
        )

    returns = discounted_returns(data.rewards, data.terminals, data.timeouts, config.gamma)
    critic, critic_target_norm, critic_audit = train_critic(
        data=data,
        split=split,
        obs_norm=obs_norm,
        returns=returns,
        config=config,
        mode=mode,
        seed=seed,
        device=device,
        output_dir=seed_dir / "critic",
        heartbeat=heartbeat,
    )
    if formal_mode and not critic_audit["optimization_terminal"]:
        raise RuntimeError(
            "Formal E7 gate failed: critic did not pass the registered optimization terminal audit"
        )

    advantages, advantage_manifest = freeze_advantages(
        critic=critic,
        data=data,
        obs_norm=obs_norm,
        target_norm=critic_target_norm,
        gamma=config.gamma,
        standardize=config.advantage_standardize,
        standardization_indices=split["train"],
        device=device,
        output_dir=seed_dir / "frozen_advantage",
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
    if formal_mode and positive_audit["state"] != "finite_terminal":
        raise RuntimeError(
            "Formal E7 gate failed: Positive-only did not reach a finite terminal without a support boundary event"
        )

    with torch.no_grad():
        all_negative_distances = np.full(data.size, np.nan, dtype=np.float32)
        for start in range(0, len(negative_train), 65536):
            idx = negative_train[start : start + 65536]
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
    far_threshold = float((matching_summary["near_cut"] + matching_summary["far_cut"]) / 2.0)
    near_negative_pool = negative_train[all_negative_distances[negative_train] <= matching_summary["near_cut"]]
    if len(near_negative_pool) == 0:
        raise RuntimeError("No near-negative samples available to define Far-cap")
    with torch.no_grad():
        near_joint_scores = positive_policy.output_score_norm(
            tensor(obs[near_negative_pool], device),
            tensor(data.actions[near_negative_pool], device),
        ).cpu().numpy()
    far_cap_score = float(np.quantile(near_joint_scores, config.far_cap_reference_quantile))
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
    mitigation_details: dict[str, bool] = {}
    for method in ("far_zero", "far_cap", "budget_matched_global"):
        control = branch_audits[method]
        score_reduced = float(
            control["final_metrics"]["phantom_joint_output_score_mean"]
        ) < 0.95 * float(
            signed_audit["final_metrics"]["phantom_joint_output_score_mean"]
        )
        support_rescued = bool(signed_audit["support_boundary_event"]) and not bool(
            control["support_boundary_event"]
        )
        task_rescued = bool(signed_audit["task_performance_collapse"]) and not bool(
            control["task_performance_collapse"]
        )
        mitigation_details[method] = bool(score_reduced or support_rescued or task_rescued)
    seed_gate = {
        "natural_far_field_present": bool(gradient_summary["natural_far_field_present"]),
        "corrected_quadratic_branch_empirically_active": bool(
            math.isfinite(slope)
            and abs(slope - config.qxi_slope_target) <= config.qxi_slope_tolerance
            and float(gradient_summary["analytic_autograd_relative_error_max"])
            <= config.analytic_autograd_error_max
            and float(gradient_summary["log_scale_to_mean_far_near_ratio"])
            >= config.log_scale_to_mean_ratio_min
        ),
        "measurable_full_parameter_contribution": bool(
            float(gradient_summary["full_parameter_gradient_far_near_ratio"])
            >= config.full_parameter_ratio_min
        ),
        "targeted_far_control_mitigates_dynamics": bool(
            mitigation_details.get("far_zero", False)
            and mitigation_details.get("far_cap", False)
        ),
        "control_details": mitigation_details,
        "terminal_state_audit_complete": bool(
            all(audit.get("state") for audit in branch_audits.values())
        ),
    }
    seed_gate["all_seed_level_checks_passed"] = bool(
        all(value for key, value in seed_gate.items() if key not in {"control_details", "all_seed_level_checks_passed"})
    )
    summary = {
        "seed": seed,
        "critic": critic_audit,
        "advantage": advantage_manifest,
        "positive_only_initialization": positive_audit,
        "matching": matching_summary,
        "gradient_probe": gradient_summary,
        "global_budget": budget,
        "independent_validation_gate": seed_gate,
        "methods": branch_audits,
    }
    atomic_write_json(seed_dir / "seed_summary.json", summary)
    return summary


def flatten_seed_summary(summary: dict[str, Any]) -> dict[str, Any]:
    probe = summary["gradient_probe"]
    row: dict[str, Any] = {
        "seed": summary["seed"],
        "critic_test_r2": summary["critic"]["selected_checkpoint_metrics"]["test_r2"],
        "critic_test_pearson": summary["critic"]["selected_checkpoint_metrics"]["test_pearson"],
        "critic_optimization_terminal": summary["critic"]["optimization_terminal"],
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
        "seed_gate_passed": summary["independent_validation_gate"][
            "all_seed_level_checks_passed"
        ],
    }
    for method, audit in summary["methods"].items():
        final = audit["final_metrics"]
        row[f"{method}_terminal_state"] = audit["state"]
        row[f"{method}_task_performance_collapse"] = audit[
            "task_performance_collapse"
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
        if isinstance(value, (int, float, np.integer, np.floating, bool)) and key != "seed"
    ]
    aggregate: dict[str, Any] = {
        "seeds_completed": len(rows),
        "seed_ids": [row["seed"] for row in rows],
    }
    for key in numeric_keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
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
            "task_performance_collapse_count": sum(
                bool(summary["methods"][method]["task_performance_collapse"])
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
    gates = [summary["independent_validation_gate"] for summary in summaries]
    gate_names = (
        "natural_far_field_present",
        "corrected_quadratic_branch_empirically_active",
        "measurable_full_parameter_contribution",
        "targeted_far_control_mitigates_dynamics",
        "terminal_state_audit_complete",
        "all_seed_level_checks_passed",
    )
    aggregate["independent_validation_gate_counts"] = {
        name: int(sum(bool(gate[name]) for gate in gates)) for name in gate_names
    }
    aggregate["paired_seed_evidence_complete"] = len(rows) > 1
    aggregate["scientific_status"] = (
        "pilot" if len(rows) == 1 else "raw_complete_pending_scientific_review"
    )
    aggregate["method_ranking_claim_allowed"] = False
    return aggregate


def build_terminal_audit(
    summaries: Sequence[dict[str, Any]], mode_name: str
) -> dict[str, Any]:
    aggregate = aggregate_seed_summaries(summaries)
    expected = len(summaries)
    counts = aggregate.get("independent_validation_gate_counts", {})
    all_gates = bool(
        expected > 0
        and counts
        and all(int(counts.get(name, 0)) == expected for name in (
            "natural_far_field_present",
            "corrected_quadratic_branch_empirically_active",
            "measurable_full_parameter_contribution",
            "targeted_far_control_mitigates_dynamics",
            "terminal_state_audit_complete",
        ))
    )
    return {
        "experiment_id": EXPERIMENT_ID,
        "mode": mode_name,
        "seeds_audited": expected,
        "scientific_status": "pilot" if mode_name == "pilot" else "raw_complete_pending_review",
        "independent_validation_gate_all_seeds": all_gates,
        "gate_counts": counts,
        "paired_seed_evidence_complete": bool(expected > 1),
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
    work_dir.mkdir(parents=True, exist_ok=True)
    global _EVENT_LOG_PATH
    _EVENT_LOG_PATH = work_dir / "events.jsonl"
    run_id = args.run_id or f"{args.mode}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
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
        "device_request": args.device,
        "claim_boundary": (
            "External mechanism validation with diagnostic normalized-return trajectories; "
            "not a standard offline-RL method-ranking table."
        ),
    }
    atomic_write_json(work_dir / "scientific_run_manifest.json", run_manifest)
    device = choose_device(args.device)
    data = load_hopper_hdf5(dataset_path, mode.max_transitions)
    atomic_write_json(
        work_dir / "DATASET_RUNTIME_SUMMARY.json",
        {
            "transitions_loaded": data.size,
            "episodes_loaded": int(np.max(data.episode_ids) + 1),
            "observation_dim": data.observations.shape[1],
            "action_dim": data.actions.shape[1],
            "reward_mean": float(np.mean(data.rewards)),
            "reward_std": float(np.std(data.rewards)),
            "terminal_fraction": float(np.mean(data.terminals)),
            "timeout_fraction": float(np.mean(data.timeouts)),
        },
    )
    summaries: list[dict[str, Any]] = []
    try:
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
                config=config,
                mode=mode,
                device=device,
                seed_dir=work_dir / "seeds" / f"seed_{seed}",
                heartbeat=seed_heartbeat,
                formal_mode=args.mode == "formal",
            )
            summaries.append(summary)
            flat = [flatten_seed_summary(item) for item in summaries]
            write_csv(work_dir / "per_seed_summary.csv", flat)
            aggregate = aggregate_seed_summaries(summaries)
            atomic_write_json(work_dir / "aggregate_summary.json", aggregate)
            if args.mode == "formal" and (
                index % config.checkpoint_every_formal_seeds == 0 or index == len(mode.seeds)
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
        atomic_write_json(work_dir / "aggregate_summary.json", aggregate)
        terminal_audit = build_terminal_audit(summaries, args.mode)
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
            "result_status": "pilot" if args.mode == "pilot" else "raw_complete_pending_review",
            "formal_result_claim": False,
        }
        atomic_write_json(work_dir / "RUN_COMPLETE.json", complete)
        atomic_write_json(
            work_dir / "scientific_heartbeat.json",
            {"utc": utc_now(), "run_id": run_id, "state": "completed", "seeds_completed": len(summaries)},
        )
        run_manifest.update(
            {
                "state": "raw_complete" if not provenance_compromised else "failed_provenance",
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
            "result_status": "failed_pilot" if args.mode == "pilot" else "failed_formal_attempt",
        }
        atomic_write_json(work_dir / "SCIENTIFIC_RUN_FAILED.json", failure)
        atomic_write_json(
            work_dir / "scientific_heartbeat.json",
            {"utc": utc_now(), "run_id": run_id, "state": "failed", "seeds_completed": len(summaries)},
        )
        emit_event({"event": "run_failed", "error_type": type(exc).__name__, "error": str(exc)})
        run_manifest.update({"state": "failed", "failed_utc": utc_now()})
        atomic_write_json(work_dir / "scientific_run_manifest.json", run_manifest)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
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
    run.add_argument("--allow-dirty", action="store_true")
    inspect = sub.add_parser("inspect", help="validate config and dataset only")
    inspect.add_argument("--dataset-path", required=True)
    inspect.add_argument(
        "--config", default="configs/e7_hopper_q2_medium_replay_v2.yaml"
    )
    inspect.add_argument("--max-transitions", type=int, default=1000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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
