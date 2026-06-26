#!/usr/bin/env python3
"""D-U1 E6 shared-semantic categorical pilot.

This runner prepares the controlled E6 environment and executes development-only
invariant, smoke, and pilot stages. It intentionally fails closed for formal use
until the protocol has been frozen after development seeds 0--4.

Scientific boundaries
---------------------
* E5 already establishes repeated categorical suppression and support boundaries.
* E6 asks a different question: can shared semantic structure let controlled local
  negative gradients improve probability on a hidden optimal action for unseen
  contexts, and can far-negative pressure destroy that benefit?
* Train and test contexts are sampled independently from the same distribution.
  Results are held-out-context generalization, not OOD generalization.
* Task collapse, support/temperature boundary events, and NaN/Inf numerical failure
  are reported separately.
* Smoke tests and pilot runs are not formal results or method rankings.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


EXPERIMENT_ID = "D-U1-E6-SEMANTIC-PILOT-01"
FORMAL_EXPERIMENT_ID = "D-U1-E6-SEMANTIC-LONGRUN-01"
EPS = 1.0e-12


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def unit(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, p=2, dim=-1, eps=1.0e-12)


def finite_scalar(value: float | int) -> bool:
    return math.isfinite(float(value))


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def yaml_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_text(*args: str) -> str | None:
    try:
        proc = subprocess.run(
            ["git", *args],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return proc.stdout.strip()


def ensure_new_or_empty(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f"output root must be new or empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def nested(config: Mapping[str, Any], *keys: str) -> Any:
    value: Any = config
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            raise KeyError(".".join(keys))
        value = value[key]
    return value


def validate_config(config: Mapping[str, Any], stage: str) -> None:
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError(f"experiment_id must be {EXPERIMENT_ID}")
    if config.get("scientific_status") != "pilot":
        raise ValueError("E6 preparation runner must remain status=pilot")
    if bool(config.get("formal_parameter_freeze")):
        raise ValueError("pilot config must not claim a formal parameter freeze")
    if stage == "formal":
        formal = nested(config, "formal_gate")
        required = (
            bool(formal.get("enabled")),
            bool(formal.get("approval_record")),
            bool(formal.get("frozen_protocol_path")),
            bool(formal.get("held_out_seeds")),
        )
        if not all(required):
            raise RuntimeError(
                f"{FORMAL_EXPERIMENT_ID} is blocked: pilot review and an explicit "
                "formal parameter-freeze record are required"
            )
        raise RuntimeError(
            "formal execution is intentionally not implemented in the pilot runner; "
            "create a separately registered formal runner after protocol freeze"
        )
    data = nested(config, "data")
    if int(data["state_dim"]) != 6 or int(data["semantic_dim"]) != 4:
        raise ValueError("locked E6 development geometry is 6D state / 4D semantics")
    if int(data["positive_actions_per_state"]) != 4:
        raise ValueError("E6 development geometry requires four positive actions per state")
    if int(data["far_negative_actions_per_state"]) != 4:
        raise ValueError("E6 development geometry requires four far negatives per state")
    if nested(config, "geometry", "negative_advantage") >= 0:
        raise ValueError("negative_advantage must be negative")
    if nested(config, "geometry", "positive_advantage") <= 0:
        raise ValueError("positive_advantage must be positive")
    if not bool(nested(config, "geometry", "fixed_advantage")):
        raise ValueError("E6 pilot is a fixed-advantage mechanism experiment")
    seeds = list(nested(config, "seeds", "development"))
    if stage in {"pilot", "all", "invariants"} and seeds != [0, 1, 2, 3, 4]:
        raise ValueError("development seeds are locked to 0--4 for this pilot")
    if list(nested(config, "seeds", "held_out_formal")):
        raise ValueError("pilot config must not consume held-out formal seeds")


def smoke_config(config: Mapping[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(dict(config))
    out["execution_mode"] = "engineering_smoke"
    out["seeds"]["development"] = [0]
    out["data"]["train_states"] = 24
    out["data"]["test_states"] = 24
    out["data"]["action_count"] = 16
    out["policy"]["hidden_dim"] = 8
    out["optimization"]["batch_size"] = 8
    out["optimization"]["maximum_steps"] = 2
    out["optimization"]["evaluation_interval_steps"] = 2
    out["optimization"]["audit_states"] = 8
    out["protocol_a"]["local_alpha_grid"] = [0.0, 0.5]
    out["protocol_b"]["methods"] = [
        "positive_only",
        "far_cap",
        "budget_matched_global",
    ]
    out["protocol_c"]["methods"] = ["positive_only", "local_only"]
    out["terminal_audit"]["trailing_evaluations_per_window"] = 2
    return out


@dataclass(frozen=True)
class RunSpec:
    protocol: str
    method: str
    alpha: float
    far_lambda: float
    concentration_mode: str
    embedding_mode: str

    @property
    def key(self) -> str:
        alpha = f"{self.alpha:.4f}".rstrip("0").rstrip(".")
        far = f"{self.far_lambda:.4f}".rstrip("0").rstrip(".")
        return (
            f"{self.protocol}__{self.embedding_mode}__{self.concentration_mode}__"
            f"{self.method}__a{alpha}__f{far}"
        )


class SemanticEnvironment:
    """Frozen D-U1 semantic catalogue and state-conditioned training targets."""

    def __init__(self, config: Mapping[str, Any], seed: int, embedding_mode: str):
        self.config = config
        self.seed = int(seed)
        self.embedding_mode = embedding_mode
        data = nested(config, "data")
        self.state_dim = int(data["state_dim"])
        self.semantic_dim = int(data["semantic_dim"])
        self.action_count = int(data["action_count"])
        self.n_positive = int(data["positive_actions_per_state"])
        self.n_far = int(data["far_negative_actions_per_state"])
        self.train_count = int(data["train_states"])
        self.test_count = int(data["test_states"])
        self.target_offset = float(nested(config, "geometry", "target_offset"))
        self.reward_scale = float(nested(config, "geometry", "reward_scale"))
        self.positive_advantage = float(nested(config, "geometry", "positive_advantage"))
        self.negative_advantage = float(nested(config, "geometry", "negative_advantage"))

        catalogue_generator = torch.Generator(device="cpu").manual_seed(100_003 + self.seed)
        base_embeddings = unit(
            torch.randn(
                self.action_count,
                self.semantic_dim,
                generator=catalogue_generator,
            )
        )
        action_permutation = torch.randperm(self.action_count, generator=catalogue_generator)
        self.reward_embeddings = base_embeddings[action_permutation].contiguous()
        self.action_permutation = action_permutation
        if embedding_mode == "aligned":
            self.policy_embeddings = self.reward_embeddings.clone()
            self.policy_permutation = torch.arange(self.action_count)
        elif embedding_mode == "shuffled":
            policy_generator = torch.Generator(device="cpu").manual_seed(150_003 + self.seed)
            shuffle = torch.randperm(self.action_count, generator=policy_generator)
            if torch.equal(shuffle, torch.arange(self.action_count)):
                shuffle = torch.roll(shuffle, 1)
            self.policy_embeddings = self.reward_embeddings[shuffle].clone()
            self.policy_permutation = shuffle
        else:
            raise ValueError(f"unknown embedding_mode: {embedding_mode}")

        geometry_generator = torch.Generator(device="cpu").manual_seed(180_003 + self.seed)
        self.w_plus = torch.randn(self.state_dim, self.semantic_dim, generator=geometry_generator)
        self.w_direction = torch.randn(
            self.state_dim, self.semantic_dim, generator=geometry_generator
        )
        self.train = self._build_split(self.train_count, 200_003 + self.seed)
        self.test = self._build_split(self.test_count, 300_003 + self.seed)

    def _build_split(self, count: int, split_seed: int) -> dict[str, torch.Tensor]:
        generator = torch.Generator(device="cpu").manual_seed(split_seed)
        states = torch.randn(count, self.state_dim, generator=generator)
        t_plus = unit(states @ self.w_plus)
        raw_direction = states @ self.w_direction
        raw_direction = raw_direction - (raw_direction * t_plus).sum(-1, keepdim=True) * t_plus
        weak = raw_direction.norm(dim=-1) < 1.0e-6
        if bool(weak.any()):
            fallback = torch.zeros_like(raw_direction)
            fallback[:, 0] = 1.0
            fallback = fallback - (fallback * t_plus).sum(-1, keepdim=True) * t_plus
            raw_direction[weak] = fallback[weak]
        direction = unit(raw_direction)
        t_star = unit(t_plus + self.target_offset * direction)
        t_minus = unit(t_plus - self.target_offset * direction)
        reward_similarity = t_star @ self.reward_embeddings.T
        rewards = self.reward_scale * (1.0 + reward_similarity)
        hidden = reward_similarity.argmax(dim=1)

        plus_similarity = t_plus @ self.reward_embeddings.T
        minus_similarity = t_minus @ self.reward_embeddings.T
        far_similarity = (-t_plus) @ self.reward_embeddings.T
        banned = torch.zeros(count, self.action_count, dtype=torch.bool)
        banned.scatter_(1, hidden[:, None], True)
        positive = self._topk_excluding(plus_similarity, banned, self.n_positive)
        banned.scatter_(1, positive, True)
        local = self._topk_excluding(minus_similarity, banned, 1).squeeze(1)
        banned.scatter_(1, local[:, None], True)
        far = self._topk_excluding(far_similarity, banned, self.n_far)
        return {
            "states": states,
            "t_plus": t_plus,
            "direction": direction,
            "t_star": t_star,
            "t_minus": t_minus,
            "reward_matrix": rewards,
            "hidden": hidden,
            "positive": positive,
            "local": local,
            "far": far,
            "positive_advantage": torch.full((count, self.n_positive), self.positive_advantage),
            "local_advantage": torch.full((count,), self.negative_advantage),
            "far_advantage": torch.full((count, self.n_far), self.negative_advantage),
        }

    @staticmethod
    def _topk_excluding(scores: torch.Tensor, banned: torch.Tensor, k: int) -> torch.Tensor:
        masked = scores.masked_fill(banned, -torch.inf)
        selected = masked.topk(k, dim=1).indices
        values = masked.gather(1, selected)
        if not bool(torch.isfinite(values).all()):
            raise RuntimeError("catalogue too small for disjoint E6 action sets")
        return selected

    def audit(self) -> dict[str, Any]:
        split_audits: dict[str, Any] = {}
        passed = True
        for split_name, split in (("train", self.train), ("test", self.test)):
            hidden = split["hidden"]
            positive = split["positive"]
            local = split["local"]
            far = split["far"]
            rows = torch.arange(hidden.numel())
            hidden_reward = split["reward_matrix"][rows, hidden]
            max_reward = split["reward_matrix"].max(dim=1).values
            positive_reward = split["reward_matrix"].gather(1, positive).mean(1)
            local_reward = split["reward_matrix"][rows, local]
            far_reward = split["reward_matrix"].gather(1, far).mean(1)
            overlap_hidden_positive = int((positive == hidden[:, None]).sum())
            overlap_hidden_local = int((local == hidden).sum())
            overlap_hidden_far = int((far == hidden[:, None]).sum())
            overlap_positive_local = int((positive == local[:, None]).sum())
            overlap_positive_far = int((positive[:, :, None] == far[:, None, :]).sum())
            overlap_local_far = int((far == local[:, None]).sum())
            negative_advantage_range = float(
                torch.cat([split["local_advantage"][:, None], split["far_advantage"]], dim=1)
                .max(1)
                .values.sub(
                    torch.cat(
                        [split["local_advantage"][:, None], split["far_advantage"]],
                        dim=1,
                    )
                    .min(1)
                    .values
                )
                .abs()
                .max()
            )
            orthogonality_error = float((split["t_plus"] * split["direction"]).sum(-1).abs().max())
            split_passed = all(
                [
                    overlap_hidden_positive == 0,
                    overlap_hidden_local == 0,
                    overlap_hidden_far == 0,
                    overlap_positive_local == 0,
                    overlap_positive_far == 0,
                    overlap_local_far == 0,
                    negative_advantage_range <= 1.0e-12,
                    orthogonality_error <= 1.0e-5,
                    float((hidden_reward - max_reward).abs().max()) <= 1.0e-7,
                ]
            )
            passed = passed and split_passed
            split_audits[split_name] = {
                "passed": split_passed,
                "hidden_is_reward_argmax_max_error": float(
                    (hidden_reward - max_reward).abs().max()
                ),
                "t_plus_direction_orthogonality_max_error": orthogonality_error,
                "negative_advantage_range_max": negative_advantage_range,
                "overlap_counts": {
                    "hidden_positive": overlap_hidden_positive,
                    "hidden_local": overlap_hidden_local,
                    "hidden_far": overlap_hidden_far,
                    "positive_local": overlap_positive_local,
                    "positive_far": overlap_positive_far,
                    "local_far": overlap_local_far,
                },
                "mean_rewards": {
                    "hidden": float(hidden_reward.mean()),
                    "positive": float(positive_reward.mean()),
                    "local_negative": float(local_reward.mean()),
                    "far_negative": float(far_reward.mean()),
                },
            }
        reward_norm_error = float(self.reward_embeddings.norm(dim=1).sub(1).abs().max())
        policy_norm_error = float(self.policy_embeddings.norm(dim=1).sub(1).abs().max())
        identity = torch.arange(self.action_count)
        shuffle_changed = bool(
            self.embedding_mode == "aligned" or not torch.equal(self.policy_permutation, identity)
        )
        passed = passed and reward_norm_error <= 1.0e-6 and policy_norm_error <= 1.0e-6
        passed = passed and shuffle_changed
        return {
            "seed": self.seed,
            "embedding_mode": self.embedding_mode,
            "passed": passed,
            "reward_embedding_norm_max_error": reward_norm_error,
            "policy_embedding_norm_max_error": policy_norm_error,
            "policy_mapping_changed_when_shuffled": shuffle_changed,
            "reward_policy_diagonal_cosine_mean": float(
                (self.reward_embeddings * self.policy_embeddings).sum(-1).mean()
            ),
            "action_id_permutation_checksum": hashlib.sha256(
                self.action_permutation.numpy().tobytes()
            ).hexdigest(),
            "policy_permutation_checksum": hashlib.sha256(
                self.policy_permutation.numpy().tobytes()
            ).hexdigest(),
            "splits": split_audits,
        }


class SemanticPolicy(nn.Module):
    def __init__(self, config: Mapping[str, Any], concentration_mode: str):
        super().__init__()
        state_dim = int(nested(config, "data", "state_dim"))
        semantic_dim = int(nested(config, "data", "semantic_dim"))
        hidden_dim = int(nested(config, "policy", "hidden_dim"))
        self.concentration_mode = concentration_mode
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.direction_head = nn.Linear(hidden_dim, semantic_dim)
        self.fixed_concentration = float(nested(config, "policy", "fixed_concentration"))
        self.concentration_floor = float(nested(config, "policy", "learnable_concentration_floor"))
        if concentration_mode == "learnable":
            self.concentration_head: nn.Linear | None = nn.Linear(hidden_dim, 1)
            target = float(nested(config, "policy", "initial_learnable_concentration"))
            raw = math.log(math.expm1(max(target - self.concentration_floor, 1.0e-4)))
            nn.init.zeros_(self.concentration_head.weight)
            nn.init.constant_(self.concentration_head.bias, raw)
        elif concentration_mode == "fixed":
            self.concentration_head = None
        else:
            raise ValueError(f"unknown concentration_mode: {concentration_mode}")

    def forward(
        self, states: torch.Tensor, action_embeddings: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.trunk(states)
        direction = unit(self.direction_head(hidden))
        if self.concentration_head is None:
            concentration = torch.full(
                (states.shape[0], 1),
                self.fixed_concentration,
                dtype=states.dtype,
                device=states.device,
            )
        else:
            concentration = F.softplus(self.concentration_head(hidden)) + self.concentration_floor
        logits = concentration * (direction @ action_embeddings.T)
        return logits, direction, concentration.squeeze(-1)


def flat_norm(grads: Sequence[torch.Tensor | None]) -> float:
    total = torch.zeros((), dtype=torch.float64)
    for grad in grads:
        if grad is not None:
            total = total + grad.detach().double().square().sum().cpu()
    return float(torch.sqrt(total))


def scale_grads(
    grads: Sequence[torch.Tensor | None], scale: float
) -> tuple[torch.Tensor | None, ...]:
    return tuple(None if grad is None else grad * scale for grad in grads)


def add_grads(
    *groups: Sequence[torch.Tensor | None],
) -> tuple[torch.Tensor | None, ...]:
    if not groups:
        return tuple()
    result: list[torch.Tensor | None] = []
    for pieces in zip(*groups):
        present = [piece for piece in pieces if piece is not None]
        result.append(None if not present else sum(present[1:], present[0].clone()))
    return tuple(result)


def gradient_branches(
    model: SemanticPolicy,
    states: torch.Tensor,
    positive: torch.Tensor,
    local: torch.Tensor,
    far: torch.Tensor,
    action_embeddings: torch.Tensor,
) -> tuple[
    tuple[torch.Tensor | None, ...],
    tuple[torch.Tensor | None, ...],
    tuple[torch.Tensor | None, ...],
    dict[str, float],
]:
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    logits, _, _ = model(states, action_embeddings)
    log_probs = F.log_softmax(logits, dim=-1)
    loss_positive = -log_probs.gather(1, positive).mean()
    loss_local = log_probs.gather(1, local[:, None]).mean()
    loss_far = log_probs.gather(1, far).mean()
    positive_grad = torch.autograd.grad(
        loss_positive, parameters, retain_graph=True, allow_unused=True
    )
    local_grad = torch.autograd.grad(loss_local, parameters, retain_graph=True, allow_unused=True)
    far_grad = torch.autograd.grad(loss_far, parameters, allow_unused=True)
    losses = {
        "loss_positive": float(loss_positive.detach()),
        "loss_local": float(loss_local.detach()),
        "loss_far": float(loss_far.detach()),
    }
    return positive_grad, local_grad, far_grad, losses


def controlled_gradient(
    method: str,
    positive_grad: Sequence[torch.Tensor | None],
    local_grad: Sequence[torch.Tensor | None],
    far_grad: Sequence[torch.Tensor | None],
    alpha: float,
    far_lambda: float,
    far_cap_ratio: float,
) -> tuple[tuple[torch.Tensor | None, ...], dict[str, float]]:
    weighted_local = scale_grads(local_grad, alpha)
    weighted_far = scale_grads(far_grad, far_lambda)
    raw_negative = add_grads(weighted_local, weighted_far)
    local_norm = flat_norm(weighted_local)
    far_norm = flat_norm(weighted_far)
    raw_negative_norm = flat_norm(raw_negative)
    cap_target = far_cap_ratio * local_norm
    cap_scale = 0.0 if far_norm <= EPS else min(1.0, cap_target / far_norm)
    capped_far = scale_grads(weighted_far, cap_scale)
    capped_negative = add_grads(weighted_local, capped_far)
    capped_negative_norm = flat_norm(capped_negative)

    if method == "positive_only":
        controlled_negative = scale_grads(raw_negative, 0.0)
    elif method in {"local_only", "far_zero"}:
        controlled_negative = weighted_local
    elif method == "near_zero":
        controlled_negative = weighted_far
    elif method == "uncontrolled":
        controlled_negative = raw_negative
    elif method == "far_cap":
        controlled_negative = capped_negative
    elif method == "budget_matched_global":
        global_scale = 0.0 if raw_negative_norm <= EPS else capped_negative_norm / raw_negative_norm
        controlled_negative = scale_grads(raw_negative, global_scale)
    else:
        raise ValueError(f"unknown method: {method}")
    total = add_grads(positive_grad, controlled_negative)
    diagnostics = {
        "raw_positive_gradient_norm": flat_norm(positive_grad),
        "raw_local_gradient_norm": flat_norm(local_grad),
        "raw_far_gradient_norm": flat_norm(far_grad),
        "weighted_local_gradient_norm": local_norm,
        "weighted_far_gradient_norm": far_norm,
        "raw_negative_gradient_norm": raw_negative_norm,
        "far_cap_scale": cap_scale,
        "far_cap_target_norm": cap_target,
        "far_cap_controlled_negative_norm": capped_negative_norm,
        "controlled_negative_gradient_norm": flat_norm(controlled_negative),
        "raw_total_gradient_norm": flat_norm(total),
    }
    diagnostics["global_budget_match_error"] = (
        abs(diagnostics["controlled_negative_gradient_norm"] - capped_negative_norm)
        if method == "budget_matched_global"
        else 0.0
    )
    return total, diagnostics


def assign_grads(model: nn.Module, grads: Sequence[torch.Tensor | None]) -> None:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if len(parameters) != len(grads):
        raise RuntimeError("gradient/parameter length mismatch")
    for parameter, grad in zip(parameters, grads):
        parameter.grad = None if grad is None else grad.detach().clone()


def parameter_vector(model: nn.Module) -> torch.Tensor:
    parts = [parameter.detach().reshape(-1).cpu() for parameter in model.parameters()]
    return torch.cat(parts) if parts else torch.empty(0)


def run_specs(config: Mapping[str, Any]) -> list[RunSpec]:
    specs: list[RunSpec] = []
    protocol_a = nested(config, "protocol_a")
    for alpha in protocol_a["local_alpha_grid"]:
        alpha_value = float(alpha)
        method = "positive_only" if alpha_value == 0.0 else "local_only"
        specs.append(
            RunSpec(
                protocol="E6-A",
                method=method,
                alpha=alpha_value,
                far_lambda=float(protocol_a["far_pressure_lambda"]),
                concentration_mode=str(protocol_a["concentration_mode"]),
                embedding_mode="aligned",
            )
        )
    protocol_b = nested(config, "protocol_b")
    for method in protocol_b["methods"]:
        specs.append(
            RunSpec(
                protocol="E6-B",
                method=str(method),
                alpha=float(protocol_b["local_alpha"]),
                far_lambda=float(protocol_b["far_pressure_lambda"]),
                concentration_mode=str(protocol_b["concentration_mode"]),
                embedding_mode="aligned",
            )
        )
    protocol_c = nested(config, "protocol_c")
    for embedding_mode in protocol_c["embedding_modes"]:
        for method in protocol_c["methods"]:
            specs.append(
                RunSpec(
                    protocol="E6-C",
                    method=str(method),
                    alpha=float(protocol_c["local_alpha"]),
                    far_lambda=float(protocol_c["far_pressure_lambda"]),
                    concentration_mode=str(protocol_c["concentration_mode"]),
                    embedding_mode=str(embedding_mode),
                )
            )
    keys = [spec.key for spec in specs]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate E6 run specification")
    return specs


def shared_batch_stream(
    seed: int, train_count: int, batch_size: int, steps: int
) -> list[torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(900_003 + seed)
    return [torch.randint(0, train_count, (batch_size,), generator=generator) for _ in range(steps)]


@torch.no_grad()
def evaluate(
    model: SemanticPolicy,
    environment: SemanticEnvironment,
    split_name: str,
    device: torch.device,
) -> dict[str, float]:
    split = environment.train if split_name == "train" else environment.test
    states = split["states"].to(device)
    action_embeddings = environment.policy_embeddings.to(device)
    logits, direction, concentration = model(states, action_embeddings)
    probabilities = F.softmax(logits, dim=-1)
    rows = torch.arange(states.shape[0], device=device)
    hidden = split["hidden"].to(device)
    positive = split["positive"].to(device)
    reward_matrix = split["reward_matrix"].to(device)
    hidden_probability = probabilities[rows, hidden]
    positive_probability = probabilities.gather(1, positive).sum(1)
    expected_reward = (probabilities * reward_matrix).sum(1)
    entropy = -(probabilities * F.log_softmax(logits, dim=-1)).sum(1)
    effective_support = entropy.exp()
    t_plus = split["t_plus"].to(device)
    t_star = split["t_star"].to(device)
    improvement = split["direction"].to(device)
    numerator = ((direction - t_plus) * improvement).sum(-1)
    denominator = ((t_star - t_plus) * improvement).sum(-1).clamp_min(1.0e-8)
    extrapolation = numerator / denominator
    return {
        "hidden_optimal_probability": float(hidden_probability.mean()),
        "positive_support_probability": float(positive_probability.mean()),
        "expected_semantic_reward": float(expected_reward.mean()),
        "normalized_semantic_extrapolation": float(extrapolation.mean()),
        "entropy_mean": float(entropy.mean()),
        "effective_support_mean": float(effective_support.mean()),
        "effective_support_p05": float(torch.quantile(effective_support, 0.05)),
        "concentration_mean": float(concentration.mean()),
        "concentration_min": float(concentration.min()),
        "concentration_max": float(concentration.max()),
    }


def audit_subset_gradients(
    model: SemanticPolicy,
    environment: SemanticEnvironment,
    spec: RunSpec,
    device: torch.device,
    config: Mapping[str, Any],
) -> dict[str, float]:
    count = min(
        int(nested(config, "optimization", "audit_states")),
        int(environment.train["states"].shape[0]),
    )
    split = environment.train
    states = split["states"][:count].to(device)
    positive = split["positive"][:count].to(device)
    local = split["local"][:count].to(device)
    far = split["far"][:count].to(device)
    positive_grad, local_grad, far_grad, losses = gradient_branches(
        model,
        states,
        positive,
        local,
        far,
        environment.policy_embeddings.to(device),
    )
    _, diagnostics = controlled_gradient(
        spec.method,
        positive_grad,
        local_grad,
        far_grad,
        spec.alpha,
        spec.far_lambda,
        float(
            nested(
                config,
                "optimization",
                "far_cap_ratio_to_weighted_local_gradient",
            )
        ),
    )
    return {**losses, **diagnostics}


def normalized_change(values: Sequence[float]) -> float:
    if len(values) < 2:
        return float("inf")
    scale = max(abs(float(np.mean(values))), 1.0e-8)
    return abs(float(values[-1]) - float(values[0])) / scale


def terminal_classification(
    trajectory: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    if not trajectory:
        return {"class": "inconclusive", "reason": "empty_trajectory"}
    if any(bool(row.get("nan_inf_numerical_failure")) for row in trajectory):
        return {"class": "numerical_failure", "reason": "nonfinite_value_observed"}
    if any(bool(row.get("support_or_temperature_boundary")) for row in trajectory):
        return {
            "class": "support_or_temperature_boundary",
            "reason": "effective_support_boundary_reached",
        }
    width = int(nested(config, "terminal_audit", "trailing_evaluations_per_window"))
    if len(trajectory) < 2 * width:
        return {
            "class": "inconclusive",
            "reason": "insufficient_two_window_history",
            "required_evaluations": 2 * width,
            "actual_evaluations": len(trajectory),
        }
    tolerance = float(nested(config, "terminal_audit", "normalized_metric_change_tolerance"))
    grad_tolerance = float(nested(config, "terminal_audit", "raw_total_gradient_median_tolerance"))
    windows = [trajectory[-2 * width : -width], trajectory[-width:]]
    metrics = (
        "test_expected_semantic_reward",
        "test_hidden_optimal_probability",
        "test_normalized_semantic_extrapolation",
        "test_entropy_mean",
    )
    window_checks: list[dict[str, Any]] = []
    for window in windows:
        changes = {
            metric: normalized_change([float(row[metric]) for row in window]) for metric in metrics
        }
        grad_median = float(
            np.median([float(row["audit_raw_total_gradient_norm"]) for row in window])
        )
        window_checks.append(
            {
                "normalized_changes": changes,
                "raw_total_gradient_median": grad_median,
                "passed": all(value <= tolerance for value in changes.values())
                and grad_median <= grad_tolerance,
            }
        )
    classification = (
        "provisional_plateau"
        if all(item["passed"] for item in window_checks)
        else "persistent_drift_or_inconclusive"
    )
    return {
        "class": classification,
        "two_window_checks": window_checks,
        "formal_two_x_extension_performed": False,
        "formal_acceptance": False,
        "note": "Pilot terminal classification cannot establish long-run validation or ranking.",
    }


def write_trajectory_rows(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("a") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def run_one(
    config: Mapping[str, Any],
    seed: int,
    spec: RunSpec,
    environment: SemanticEnvironment,
    initial_state: Mapping[str, torch.Tensor],
    batches: Sequence[torch.Tensor],
    output_root: Path,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    seed_all(seed + 40_000)
    model = SemanticPolicy(config, spec.concentration_mode).to(device)
    model.load_state_dict(initial_state, strict=True)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(nested(config, "optimization", "learning_rate")),
        betas=tuple(float(x) for x in nested(config, "optimization", "betas")),
        eps=float(nested(config, "optimization", "eps")),
    )
    maximum_steps = int(nested(config, "optimization", "maximum_steps"))
    eval_interval = int(nested(config, "optimization", "evaluation_interval_steps"))
    support_threshold = float(nested(config, "events", "effective_support_boundary"))
    concentration_warning = float(nested(config, "events", "concentration_warning"))
    split = environment.train
    action_embeddings = environment.policy_embeddings.to(device)
    trajectory: list[dict[str, Any]] = []
    numerical_failure = False

    def add_evaluation(step: int, update_norm: float) -> None:
        nonlocal numerical_failure
        train_metrics = evaluate(model, environment, "train", device)
        test_metrics = evaluate(model, environment, "test", device)
        gradient_metrics = audit_subset_gradients(model, environment, spec, device, config)
        numeric_values = [
            *train_metrics.values(),
            *test_metrics.values(),
            *gradient_metrics.values(),
        ]
        numerical_failure = numerical_failure or not all(
            finite_scalar(value) for value in numeric_values
        )
        row: dict[str, Any] = {
            "experiment_id": EXPERIMENT_ID,
            "seed": seed,
            "run_key": spec.key,
            "protocol": spec.protocol,
            "method": spec.method,
            "alpha": spec.alpha,
            "far_lambda": spec.far_lambda,
            "concentration_mode": spec.concentration_mode,
            "embedding_mode": spec.embedding_mode,
            "step": step,
            "adam_parameter_update_norm": update_norm,
            "nan_inf_numerical_failure": numerical_failure,
        }
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"test_{key}": value for key, value in test_metrics.items()})
        row.update({f"audit_{key}": value for key, value in gradient_metrics.items()})
        row["support_or_temperature_boundary"] = bool(
            test_metrics["effective_support_p05"] <= support_threshold
        )
        row["concentration_warning"] = bool(
            test_metrics["concentration_max"] >= concentration_warning
        )
        trajectory.append(row)

    add_evaluation(0, 0.0)
    for step in range(1, maximum_steps + 1):
        index = batches[step - 1]
        states = split["states"][index].to(device)
        positive = split["positive"][index].to(device)
        local = split["local"][index].to(device)
        far = split["far"][index].to(device)
        positive_grad, local_grad, far_grad, _ = gradient_branches(
            model, states, positive, local, far, action_embeddings
        )
        total_grad, _ = controlled_gradient(
            spec.method,
            positive_grad,
            local_grad,
            far_grad,
            spec.alpha,
            spec.far_lambda,
            float(
                nested(
                    config,
                    "optimization",
                    "far_cap_ratio_to_weighted_local_gradient",
                )
            ),
        )
        optimizer.zero_grad(set_to_none=True)
        assign_grads(model, total_grad)
        if not all(grad is None or bool(torch.isfinite(grad).all()) for grad in total_grad):
            numerical_failure = True
            add_evaluation(step, float("nan"))
            break
        before = parameter_vector(model)
        optimizer.step()
        after = parameter_vector(model)
        update_norm = float(torch.linalg.vector_norm(after - before))
        if not finite_scalar(update_norm) or not all(
            bool(torch.isfinite(parameter).all()) for parameter in model.parameters()
        ):
            numerical_failure = True
        if step % eval_interval == 0 or step == maximum_steps or numerical_failure:
            add_evaluation(step, update_norm)
        if numerical_failure:
            break

    terminal = terminal_classification(trajectory, config)
    final = trajectory[-1]
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "scientific_status": "pilot",
        "seed": seed,
        "run_key": spec.key,
        "protocol": spec.protocol,
        "method": spec.method,
        "alpha": spec.alpha,
        "far_lambda": spec.far_lambda,
        "concentration_mode": spec.concentration_mode,
        "embedding_mode": spec.embedding_mode,
        "completed_steps": int(final["step"]),
        "terminal_class": terminal["class"],
        "terminal_audit": terminal,
        "task_performance_collapse": None,
        "support_or_temperature_boundary": bool(
            any(row["support_or_temperature_boundary"] for row in trajectory)
        ),
        "nan_inf_numerical_failure": bool(
            any(row["nan_inf_numerical_failure"] for row in trajectory)
        ),
        "final": {
            key.removeprefix("test_"): value
            for key, value in final.items()
            if key.startswith("test_")
        },
    }
    per_run_dir = output_root / "runs" / f"seed_{seed:03d}"
    per_run_dir.mkdir(parents=True, exist_ok=True)
    json_dump(per_run_dir / f"{spec.key}.summary.json", summary)
    return summary, trajectory


def apply_task_collapse_labels(summaries: list[dict[str, Any]], config: Mapping[str, Any]) -> None:
    ratio = float(nested(config, "events", "task_collapse_ratio_to_paired_positive_only"))
    reference: dict[tuple[int, str, str], float] = {}
    for summary in summaries:
        if summary["method"] == "positive_only":
            key = (summary["seed"], summary["protocol"], summary["embedding_mode"])
            reference[key] = float(summary["final"]["expected_semantic_reward"])
    for summary in summaries:
        key = (summary["seed"], summary["protocol"], summary["embedding_mode"])
        baseline = reference.get(key)
        if baseline is None:
            summary["task_performance_collapse"] = None
            summary["task_collapse_reference_missing"] = True
        else:
            reward = float(summary["final"]["expected_semantic_reward"])
            summary["paired_positive_only_reward"] = baseline
            summary["task_performance_collapse"] = bool(reward <= ratio * baseline)
            summary["task_collapse_reference_missing"] = False


def aggregate_summaries(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for summary in summaries:
        group = (
            f"{summary['protocol']}|{summary['embedding_mode']}|"
            f"{summary['concentration_mode']}|{summary['method']}|"
            f"alpha={summary['alpha']}|far={summary['far_lambda']}"
        )
        groups.setdefault(group, []).append(summary)
    aggregate: dict[str, Any] = {}
    for group, rows in sorted(groups.items()):
        metric_names = (
            "hidden_optimal_probability",
            "positive_support_probability",
            "expected_semantic_reward",
            "normalized_semantic_extrapolation",
            "entropy_mean",
            "effective_support_mean",
            "effective_support_p05",
            "concentration_mean",
            "concentration_max",
        )
        metrics: dict[str, Any] = {}
        for metric in metric_names:
            values = np.asarray([float(row["final"][metric]) for row in rows], dtype=float)
            metrics[metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                "min": float(values.min()),
                "max": float(values.max()),
            }
        aggregate[group] = {
            "n": len(rows),
            "metrics": metrics,
            "terminal_class_counts": dict(
                sorted(
                    {
                        name: sum(row["terminal_class"] == name for row in rows)
                        for name in {str(row["terminal_class"]) for row in rows}
                    }.items()
                )
            ),
            "task_performance_collapse_count": sum(
                row.get("task_performance_collapse") is True for row in rows
            ),
            "support_or_temperature_boundary_count": sum(
                bool(row["support_or_temperature_boundary"]) for row in rows
            ),
            "nan_inf_numerical_failure_count": sum(
                bool(row["nan_inf_numerical_failure"]) for row in rows
            ),
        }
    return aggregate


def pilot_freeze_recommendation(
    summaries: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    protocol_a = [row for row in summaries if row["protocol"] == "E6-A"]
    by_alpha: dict[float, list[Mapping[str, Any]]] = {}
    for row in protocol_a:
        by_alpha.setdefault(float(row["alpha"]), []).append(row)
    reference_rows = by_alpha.get(0.0, [])
    reference_hidden = (
        float(np.mean([row["final"]["hidden_optimal_probability"] for row in reference_rows]))
        if reference_rows
        else None
    )
    candidates: list[dict[str, Any]] = []
    for alpha, rows in sorted(by_alpha.items()):
        hidden = float(np.mean([row["final"]["hidden_optimal_probability"] for row in rows]))
        reward = float(np.mean([row["final"]["expected_semantic_reward"] for row in rows]))
        support_events = sum(bool(row["support_or_temperature_boundary"]) for row in rows)
        numerical_events = sum(bool(row["nan_inf_numerical_failure"]) for row in rows)
        candidates.append(
            {
                "alpha": alpha,
                "hidden_optimal_probability_mean": hidden,
                "expected_semantic_reward_mean": reward,
                "hidden_probability_delta_vs_positive_only": (
                    None if reference_hidden is None else hidden - reference_hidden
                ),
                "support_event_count": support_events,
                "numerical_failure_count": numerical_events,
                "development_candidate": bool(
                    alpha > 0
                    and reference_hidden is not None
                    and hidden > reference_hidden
                    and support_events == 0
                    and numerical_events == 0
                ),
            }
        )
    return {
        "experiment_id": EXPERIMENT_ID,
        "scientific_status": "pilot",
        "automatic_freeze_allowed": False,
        "user_review_required": True,
        "protocol_a_development_summary": candidates,
        "must_freeze_before_formal": [
            "local alpha grid or selected alphas",
            "fixed and learnable concentration settings",
            "learning rate and optimizer",
            "maximum steps and evaluation interval",
            "two-window and 2x-extension stopping criteria",
            "task and support boundary thresholds",
            "untouched held-out formal seeds",
            "formal method matrix and primary comparisons",
        ],
        "formal_experiment_id_reserved": FORMAL_EXPERIMENT_ID,
        "formal_gate_enabled": bool(nested(config, "formal_gate", "enabled")),
    }


def write_summary_csv(path: Path, summaries: Sequence[Mapping[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        row = {
            "seed": summary["seed"],
            "protocol": summary["protocol"],
            "embedding_mode": summary["embedding_mode"],
            "concentration_mode": summary["concentration_mode"],
            "method": summary["method"],
            "alpha": summary["alpha"],
            "far_lambda": summary["far_lambda"],
            "completed_steps": summary["completed_steps"],
            "terminal_class": summary["terminal_class"],
            "task_performance_collapse": summary["task_performance_collapse"],
            "support_or_temperature_boundary": summary["support_or_temperature_boundary"],
            "nan_inf_numerical_failure": summary["nan_inf_numerical_failure"],
        }
        row.update({f"final_{key}": value for key, value in summary["final"].items()})
        rows.append(row)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def source_manifest(output_root: Path) -> dict[str, Any]:
    tracked = [
        Path("src/drpo/du1_e6_semantic.py"),
        Path("configs/du1_e6_semantic_pilot.yaml"),
        Path("docs/handoff.md"),
        Path("experiments/registry.yaml"),
    ]
    files: list[dict[str, Any]] = []
    for path in tracked:
        if path.exists():
            files.append(
                {"path": path.as_posix(), "size": path.stat().st_size, "sha256": sha256_file(path)}
            )
    return {
        "git_head": git_text("rev-parse", "HEAD"),
        "git_status_porcelain": git_text("status", "--porcelain"),
        "source_files": files,
        "output_root": str(output_root),
    }


def execute(config: dict[str, Any], stage: str, output_root: Path, device: torch.device) -> None:
    ensure_new_or_empty(output_root)
    started = time.time()
    yaml_dump(output_root / "resolved_config.yaml", config)
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "scientific_status": "pilot",
        "execution_mode": config.get("execution_mode"),
        "requested_stage": stage,
        "formal_result": False,
        "method_ranking_allowed": False,
        "started_unix": started,
        "device": str(device),
        "source": source_manifest(output_root),
    }
    json_dump(output_root / "run_manifest.json", manifest)

    seeds = [int(seed) for seed in nested(config, "seeds", "development")]
    embedding_modes = sorted(
        set(["aligned", *list(nested(config, "protocol_c", "embedding_modes"))])
    )
    environments: dict[tuple[int, str], SemanticEnvironment] = {}
    audits: list[dict[str, Any]] = []
    for seed in seeds:
        for embedding_mode in embedding_modes:
            environment = SemanticEnvironment(config, seed, embedding_mode)
            audit = environment.audit()
            environments[(seed, embedding_mode)] = environment
            audits.append(audit)
    json_dump(output_root / "environment_audits.json", audits)
    if not all(bool(audit["passed"]) for audit in audits):
        json_dump(
            output_root / "RUN_FAILED.json",
            {
                "experiment_id": EXPERIMENT_ID,
                "reason": "environment_invariant_failure",
                "scientific_status": "pilot",
            },
        )
        raise RuntimeError("E6 environment invariant audit failed")

    if stage == "invariants":
        terminal_audit = {
            "experiment_id": EXPERIMENT_ID,
            "stage": "invariants",
            "all_environment_invariants_passed": True,
            "scientific_result": False,
            "formal_acceptance": False,
        }
        json_dump(output_root / "terminal_audit.json", terminal_audit)
        json_dump(output_root / "per_run_summary.json", [])
        json_dump(output_root / "aggregate_summary.json", {})
        json_dump(
            output_root / "pilot_freeze_recommendation.json",
            {
                "automatic_freeze_allowed": False,
                "reason": "invariants-only stage contains no training evidence",
            },
        )
        json_dump(
            output_root / "RUN_COMPLETE.json",
            {
                "experiment_id": EXPERIMENT_ID,
                "stage": "invariants",
                "scientific_status": "pilot",
                "formal_result": False,
                "completed": True,
            },
        )
        return

    specs = run_specs(config)
    summaries: list[dict[str, Any]] = []
    trajectory_path = output_root / "trajectories.jsonl"
    maximum_steps = int(nested(config, "optimization", "maximum_steps"))
    batch_size = int(nested(config, "optimization", "batch_size"))
    initial_states: dict[tuple[int, str], dict[str, torch.Tensor]] = {}
    for seed in seeds:
        for mode in {spec.concentration_mode for spec in specs}:
            seed_all(seed + 10_000)
            initial_model = SemanticPolicy(config, mode)
            initial_states[(seed, mode)] = {
                key: value.detach().clone() for key, value in initial_model.state_dict().items()
            }
        batches = shared_batch_stream(
            seed,
            int(nested(config, "data", "train_states")),
            batch_size,
            maximum_steps,
        )
        for spec in specs:
            summary, trajectory = run_one(
                config,
                seed,
                spec,
                environments[(seed, spec.embedding_mode)],
                initial_states[(seed, spec.concentration_mode)],
                batches,
                output_root,
                device,
            )
            summaries.append(summary)
            write_trajectory_rows(trajectory_path, trajectory)

    apply_task_collapse_labels(summaries, config)
    json_dump(output_root / "per_run_summary.json", summaries)
    write_summary_csv(output_root / "per_run_summary.csv", summaries)
    aggregate = aggregate_summaries(summaries)
    json_dump(output_root / "aggregate_summary.json", aggregate)
    freeze = pilot_freeze_recommendation(summaries, config)
    json_dump(output_root / "pilot_freeze_recommendation.json", freeze)
    expected_runs = len(seeds) * len(specs)
    all_runs_present = len(summaries) == expected_runs
    no_numerical_failures = all(not row["nan_inf_numerical_failure"] for row in summaries)
    terminal_audit = {
        "experiment_id": EXPERIMENT_ID,
        "scientific_status": "pilot",
        "expected_runs": expected_runs,
        "actual_runs": len(summaries),
        "all_runs_present": all_runs_present,
        "environment_invariants_passed": all(bool(audit["passed"]) for audit in audits),
        "nan_inf_numerical_failure_count": sum(
            bool(row["nan_inf_numerical_failure"]) for row in summaries
        ),
        "support_or_temperature_boundary_count": sum(
            bool(row["support_or_temperature_boundary"]) for row in summaries
        ),
        "task_performance_collapse_count": sum(
            row.get("task_performance_collapse") is True for row in summaries
        ),
        "formal_two_x_extension_performed": False,
        "formal_scientific_acceptance": False,
        "formal_method_ranking_allowed": False,
        "pilot_integrity_passed": all_runs_present and no_numerical_failures,
        "remaining_gate": nested(config, "formal_gate", "reason"),
    }
    json_dump(output_root / "terminal_audit.json", terminal_audit)
    completed = all_runs_present and no_numerical_failures
    json_dump(
        output_root / "RUN_COMPLETE.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "scientific_status": "pilot",
            "execution_mode": config.get("execution_mode"),
            "completed": completed,
            "formal_result": False,
            "method_ranking_allowed": False,
            "expected_runs": expected_runs,
            "actual_runs": len(summaries),
            "elapsed_seconds": time.time() - started,
        },
    )
    if not completed:
        raise RuntimeError("E6 pilot did not complete all finite method-seed runs")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--stage",
        required=True,
        choices=["invariants", "smoke", "pilot", "all", "formal"],
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = yaml.safe_load(args.config.read_text())
    if not isinstance(config, dict):
        raise ValueError("config root must be a mapping")
    validate_config(config, args.stage)
    if args.stage == "smoke":
        config = smoke_config(config)
        torch.set_num_threads(1)
    stage = "pilot" if args.stage in {"smoke", "pilot", "all"} else args.stage
    device = resolve_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(1)
    execute(config, stage, args.output_root, device)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"E6 runner failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
