#!/usr/bin/env python3
"""D-U1 E6 utility x surprisal Cartesian successor and taper comparison.

This formal successor fixes the confound in the historical shared-semantic E6:
semantic directional utility and learner-relative rarity are represented by two
independent coordinates.  Each semantic negative prototype is duplicated into a
common and a rare categorical action with identical reward/utility semantics and
an exactly controlled initial logit-bias gap.  The same 2x2 environment supports
both the corrected E6 mechanism block and the E6-TAPER method block.

Scientific boundaries
---------------------
* The four negative cells have equal fixed advantage and equal sample count.
* Utility is a ground-truth semantic direction label; rarity is current surprisal.
* Train/test contexts are i.i.d. from the same distribution.  Results are
  held-out-context generalization, never OOD generalization.
* Categorical direct-logit scores are bounded; the experiment studies support
  suppression and shared-semantic transfer, not Gaussian unbounded gradients.
* Task-performance collapse, support boundary, and NaN/Inf failure are separate.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import csv
import hashlib
import json
import math
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


EXPERIMENT_ID = "D-U1-E6-CARTESIAN-TAPER-01"
EPS = 1.0e-12
CELL_NAMES = (
    "useful_common",
    "useful_rare",
    "unhelpful_common",
    "unhelpful_rare",
)
MECHANISM_METHODS = (
    "positive_only",
    "useful_common_only",
    "useful_rare_only",
    "unhelpful_common_only",
    "unhelpful_rare_only",
    "useful_all",
    "unhelpful_all",
    "common_all",
    "rare_all",
    "all_negative",
)
TAPER_METHODS = (
    "global_matched",
    "reciprocal_linear",
    "reciprocal_quadratic",
    "exponential",
)
ALL_METHODS = MECHANISM_METHODS + TAPER_METHODS


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def unit(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, p=2, dim=-1, eps=1.0e-12)


def nested(config: Mapping[str, Any], *keys: str) -> Any:
    value: Any = config
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            raise KeyError(".".join(keys))
        value = value[key]
    return value


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def yaml_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_text(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def ensure_new_or_empty(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f"output root must be new or empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def prepare_output_root(output_root: Path, stage: str) -> Path:
    if stage != "formal":
        ensure_new_or_empty(output_root)
        return output_root / "run_manifest.json"
    if not output_root.is_dir():
        raise RuntimeError("formal run must execute inside the hardened guard output root")
    manifest = output_root / "run_manifest.json"
    if not manifest.is_file():
        raise RuntimeError("formal run requires guard-owned run_manifest.json")
    data = json.loads(manifest.read_text())
    required = {
        "experiment_id": EXPERIMENT_ID,
        "run_class": "formal",
        "execution_state": "running",
    }
    for key, expected in required.items():
        if data.get(key) != expected:
            raise RuntimeError(f"guard manifest {key} must be {expected!r}")
    stale = [
        name
        for name in (
            "resolved_config.yaml",
            "scientific_run_manifest.json",
            "environment_audits.json",
            "coordinate_calibration.json",
            "trajectories.jsonl",
            "per_run_summary.json",
            "per_run_summary.csv",
            "aggregate_summary.json",
            "mechanism_summary.json",
            "taper_summary.json",
            "terminal_audit.json",
            "formal_protocol_freeze.json",
            "RUN_COMPLETE.json",
        )
        if (output_root / name).exists()
    ]
    if stale:
        raise RuntimeError("formal output root contains stale scientific files: " + ", ".join(stale))
    return output_root / "scientific_run_manifest.json"


def load_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("config must be a mapping")
    return data


def formal_expected() -> dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "data.state_dim": 6,
        "data.semantic_dim": 4,
        "data.semantic_prototypes": 32,
        "data.rarity_replicas": 2,
        "data.action_count": 64,
        "data.train_states": 2048,
        "data.test_states": 2048,
        "data.positive_prototypes_per_state": 4,
        "geometry.positive_advantage": 1.0,
        "geometry.negative_advantage": -1.0,
        "geometry.target_offset": 0.45,
        "geometry.reward_scale": 0.5,
        "geometry.fixed_advantage": True,
        "policy.hidden_dim": 64,
        "policy.fixed_concentration": 8.0,
        "policy.activation": "tanh",
        "policy.trainable_action_bias": True,
        "policy.initial_rarity_logit_gap": 4.0,
        "policy.frozen_initial_semantic_reference": True,
        "policy.initial_semantic_logit_residual_zero": True,
        "optimization.optimizer": "Adam",
        "optimization.learning_rate": 0.001,
        "optimization.betas": [0.9, 0.999],
        "optimization.eps": 1.0e-8,
        "optimization.batch_size": 128,
        "optimization.maximum_steps": 8000,
        "optimization.evaluation_interval_steps": 100,
        "optimization.audit_states": 512,
        "optimization.cpu_threads_per_run": 1,
        "optimization.parallel_workers": 8,
        "optimization.negative_alpha": 0.25,
        "taper.coordinate": "normalized_excess_current_surprisal",
        "taper.threshold_rule": "pretraining_common_median",
        "taper.scale_rule": "pretraining_rare_minus_common_median",
        "taper.minimum_calibration_gap": 1.0,
        "taper.initial_cartesian_tolerance": 1.0e-6,
        "taper.reference_normalized_rare_coordinate": 1.0,
        "taper.reference_rare_retention": 0.25,
        "taper.global_control": "initial_raw_negative_gradient_norm_matched_to_exponential",
        "taper.detach_surprisal_weight": True,
        "taper.dynamic_rarity_role_assignment": True,
        "events.task_collapse_ratio_to_paired_positive_only": 0.2,
        "events.effective_support_boundary": 1.5,
        "terminal_audit.mode": "formal_two_x_windows",
        "terminal_audit.formal_horizon_steps": 8000,
        "terminal_audit.window_1_steps": [4000, 6000],
        "terminal_audit.window_2_steps": [6000, 8000],
        "terminal_audit.metric_window_mean_abs_tolerances": {
            "expected_semantic_reward": 0.01,
            "hidden_optimal_family_probability": 0.02,
            "entropy_mean": 0.08,
            "action_bias_gap_mean": 0.20,
        },
        "seeds.held_out_formal": list(range(200, 220)),
        "methods": list(ALL_METHODS),
    }


def _config_value(config: Mapping[str, Any], dotted: str) -> Any:
    parts = dotted.split(".")
    return nested(config, *parts) if len(parts) > 1 else config[parts[0]]


def validate_config(config: Mapping[str, Any], stage: str) -> None:
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError(f"experiment_id must be {EXPERIMENT_ID}")
    if int(nested(config, "data", "semantic_prototypes")) * int(
        nested(config, "data", "rarity_replicas")
    ) != int(nested(config, "data", "action_count")):
        raise ValueError("semantic_prototypes * rarity_replicas must equal action_count")
    if int(nested(config, "data", "rarity_replicas")) != 2:
        raise ValueError("this protocol requires exactly common/rare replicas")
    if list(config.get("methods", [])) != list(ALL_METHODS):
        raise ValueError("method order must match the frozen protocol")
    if stage == "formal":
        if not bool(config.get("formal_parameter_freeze")):
            raise RuntimeError("formal_parameter_freeze must be true")
        for key, expected in formal_expected().items():
            actual = _config_value(config, key)
            if actual != expected:
                raise RuntimeError(f"formal config mismatch for {key}: {actual!r} != {expected!r}")
        if not bool(nested(config, "approval", "user_approved")):
            raise RuntimeError("formal protocol lacks user approval")


def smoke_config(config: Mapping[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(dict(config))
    out["formal_parameter_freeze"] = False
    out["scientific_status"] = "pilot"
    out["seeds"]["held_out_formal"] = [0]
    out["data"]["train_states"] = 32
    out["data"]["test_states"] = 32
    out["optimization"]["maximum_steps"] = 4
    out["optimization"]["evaluation_interval_steps"] = 2
    out["optimization"]["audit_states"] = 16
    out["terminal_audit"]["window_1_steps"] = [0, 2]
    out["terminal_audit"]["window_2_steps"] = [2, 4]
    return out


@dataclass(frozen=True)
class MethodSpec:
    method: str
    active_cells: tuple[str, ...]
    taper_family: str | None = None


def method_specs() -> list[MethodSpec]:
    return [
        MethodSpec("positive_only", ()),
        MethodSpec("useful_common_only", ("useful_common",)),
        MethodSpec("useful_rare_only", ("useful_rare",)),
        MethodSpec("unhelpful_common_only", ("unhelpful_common",)),
        MethodSpec("unhelpful_rare_only", ("unhelpful_rare",)),
        MethodSpec("useful_all", ("useful_common", "useful_rare")),
        MethodSpec("unhelpful_all", ("unhelpful_common", "unhelpful_rare")),
        MethodSpec("common_all", ("useful_common", "unhelpful_common")),
        MethodSpec("rare_all", ("useful_rare", "unhelpful_rare")),
        MethodSpec("all_negative", CELL_NAMES),
        MethodSpec("global_matched", CELL_NAMES, "global"),
        MethodSpec("reciprocal_linear", CELL_NAMES, "reciprocal_linear"),
        MethodSpec("reciprocal_quadratic", CELL_NAMES, "reciprocal_quadratic"),
        MethodSpec("exponential", CELL_NAMES, "exponential"),
    ]


class CartesianSemanticEnvironment:
    """Semantic utility x initial-rarity Cartesian categorical environment."""

    def __init__(self, config: Mapping[str, Any], seed: int):
        self.config = config
        self.seed = int(seed)
        data = nested(config, "data")
        self.state_dim = int(data["state_dim"])
        self.semantic_dim = int(data["semantic_dim"])
        self.prototype_count = int(data["semantic_prototypes"])
        self.rarity_replicas = int(data["rarity_replicas"])
        self.action_count = int(data["action_count"])
        self.n_positive = int(data["positive_prototypes_per_state"])
        self.train_count = int(data["train_states"])
        self.test_count = int(data["test_states"])
        self.target_offset = float(nested(config, "geometry", "target_offset"))
        self.reward_scale = float(nested(config, "geometry", "reward_scale"))
        self.positive_advantage = float(nested(config, "geometry", "positive_advantage"))
        self.negative_advantage = float(nested(config, "geometry", "negative_advantage"))

        gen = torch.Generator(device="cpu").manual_seed(410_003 + self.seed)
        half = self.prototype_count // 2
        base = unit(torch.randn(half, self.semantic_dim, generator=gen))
        prototypes = torch.cat([base, -base], dim=0)
        permutation = torch.randperm(self.prototype_count, generator=gen)
        self.prototype_embeddings = prototypes[permutation].contiguous()
        self.action_prototype = torch.arange(self.prototype_count).repeat_interleave(2)
        self.action_rarity = torch.tensor([0, 1], dtype=torch.long).repeat(self.prototype_count)
        self.action_embeddings = self.prototype_embeddings[self.action_prototype]

        geom = torch.Generator(device="cpu").manual_seed(420_003 + self.seed)
        self.w_plus = torch.randn(self.state_dim, self.semantic_dim, generator=geom)
        self.w_direction = torch.randn(self.state_dim, self.semantic_dim, generator=geom)
        self.train = self._build_split(self.train_count, 430_003 + self.seed)
        self.test = self._build_split(self.test_count, 440_003 + self.seed)

    @staticmethod
    def action_id(prototype: torch.Tensor, rarity: int) -> torch.Tensor:
        return prototype * 2 + int(rarity)

    @staticmethod
    def _topk_excluding(scores: torch.Tensor, banned: torch.Tensor, k: int) -> torch.Tensor:
        masked = scores.masked_fill(banned, -torch.inf)
        values, indices = masked.topk(k, dim=1)
        if not bool(torch.isfinite(values).all()):
            raise RuntimeError("insufficient admissible prototypes")
        return indices

    def _build_split(self, count: int, split_seed: int) -> dict[str, torch.Tensor]:
        gen = torch.Generator(device="cpu").manual_seed(split_seed)
        states = torch.randn(count, self.state_dim, generator=gen)
        t_plus = unit(states @ self.w_plus)
        raw = states @ self.w_direction
        raw = raw - (raw * t_plus).sum(-1, keepdim=True) * t_plus
        weak = raw.norm(dim=-1) < 1.0e-6
        if bool(weak.any()):
            fallback = torch.zeros_like(raw)
            fallback[:, 0] = 1.0
            fallback = fallback - (fallback * t_plus).sum(-1, keepdim=True) * t_plus
            raw[weak] = fallback[weak]
        direction = unit(raw)
        t_star = unit(t_plus + self.target_offset * direction)

        reward_similarity = t_star @ self.prototype_embeddings.T
        prototype_rewards = self.reward_scale * (1.0 + reward_similarity)
        hidden_proto = reward_similarity.argmax(dim=1)
        banned = torch.zeros(count, self.prototype_count, dtype=torch.bool)
        banned.scatter_(1, hidden_proto[:, None], True)
        positive_proto = self._topk_excluding(t_plus @ self.prototype_embeddings.T, banned, self.n_positive)
        banned.scatter_(1, positive_proto, True)

        # Repelling prototype e from a policy direction near t_plus moves along
        # t_plus - e.  Its projection on the true t_plus -> t_star direction is
        # the ground-truth directional utility used for the first Cartesian axis.
        utility = ((t_plus[:, None, :] - self.prototype_embeddings[None, :, :]) * direction[:, None, :]).sum(-1)
        useful_proto = self._topk_excluding(utility, banned, 1).squeeze(1)
        banned.scatter_(1, useful_proto[:, None], True)
        unhelpful_proto = self._topk_excluding(-utility, banned, 1).squeeze(1)

        positive = self.action_id(positive_proto, 0)  # positive demonstrations use common replicas only
        hidden_actions = torch.stack(
            [self.action_id(hidden_proto, 0), self.action_id(hidden_proto, 1)], dim=1
        )
        cells = {
            "useful_common": self.action_id(useful_proto, 0),
            "useful_rare": self.action_id(useful_proto, 1),
            "unhelpful_common": self.action_id(unhelpful_proto, 0),
            "unhelpful_rare": self.action_id(unhelpful_proto, 1),
        }
        useful_pair = torch.stack(
            [cells["useful_common"], cells["useful_rare"]], dim=1
        )
        unhelpful_pair = torch.stack(
            [cells["unhelpful_common"], cells["unhelpful_rare"]], dim=1
        )
        reward_matrix = prototype_rewards[:, self.action_prototype]
        return {
            "states": states,
            "t_plus": t_plus,
            "direction": direction,
            "t_star": t_star,
            "prototype_reward_matrix": prototype_rewards,
            "reward_matrix": reward_matrix,
            "hidden_proto": hidden_proto,
            "hidden_actions": hidden_actions,
            "positive": positive,
            "positive_advantage": torch.full((count, self.n_positive), self.positive_advantage),
            "useful_proto": useful_proto,
            "unhelpful_proto": unhelpful_proto,
            "utility_matrix": utility,
            "useful_utility": utility.gather(1, useful_proto[:, None]).squeeze(1),
            "unhelpful_utility": utility.gather(1, unhelpful_proto[:, None]).squeeze(1),
            "useful_pair": useful_pair,
            "unhelpful_pair": unhelpful_pair,
            **cells,
            **{f"{name}_advantage": torch.full((count,), self.negative_advantage) for name in CELL_NAMES},
        }

    def initial_action_bias(self) -> torch.Tensor:
        gap = float(nested(self.config, "policy", "initial_rarity_logit_gap"))
        common = gap / 2.0
        rare = -gap / 2.0
        return torch.where(self.action_rarity == 0, torch.tensor(common), torch.tensor(rare)).float()

    def audit(self) -> dict[str, Any]:
        result: dict[str, Any] = {"seed": self.seed, "splits": {}}
        passed = True
        for name, split in (("train", self.train), ("test", self.test)):
            uc, ur = split["useful_common"], split["useful_rare"]
            nc, nr = split["unhelpful_common"], split["unhelpful_rare"]
            useful_same = bool(torch.equal(uc // 2, ur // 2))
            unhelpful_same = bool(torch.equal(nc // 2, nr // 2))
            rarity_exact = bool(
                torch.all(uc % 2 == 0)
                and torch.all(ur % 2 == 1)
                and torch.all(nc % 2 == 0)
                and torch.all(nr % 2 == 1)
            )
            advantages = torch.stack([split[f"{cell}_advantage"] for cell in CELL_NAMES], dim=1)
            advantage_range = float((advantages.max(1).values - advantages.min(1).values).abs().max())
            utility_gap_min = float((split["useful_utility"] - split["unhelpful_utility"]).min())
            utility_sign_fraction = float(
                ((split["useful_utility"] > 0) & (split["unhelpful_utility"] < 0)).float().mean()
            )
            reward_pair_error = max(
                float((split["reward_matrix"][torch.arange(len(uc)), uc] - split["reward_matrix"][torch.arange(len(ur)), ur]).abs().max()),
                float((split["reward_matrix"][torch.arange(len(nc)), nc] - split["reward_matrix"][torch.arange(len(nr)), nr]).abs().max()),
            )
            overlap_hidden = sum(
                int((split[cell][:, None] == split["hidden_actions"]).sum()) for cell in CELL_NAMES
            )
            split_passed = all(
                [
                    useful_same,
                    unhelpful_same,
                    rarity_exact,
                    advantage_range <= 1.0e-12,
                    utility_gap_min > 0.0,
                    utility_sign_fraction >= 1.0 - 1.0e-12,
                    reward_pair_error <= 1.0e-12,
                    overlap_hidden == 0,
                ]
            )
            passed = passed and split_passed
            result["splits"][name] = {
                "passed": split_passed,
                "useful_replica_same_semantic_prototype": useful_same,
                "unhelpful_replica_same_semantic_prototype": unhelpful_same,
                "rarity_replica_identity_exact": rarity_exact,
                "negative_advantage_range_max": advantage_range,
                "utility_gap_min": utility_gap_min,
                "utility_sign_fraction": utility_sign_fraction,
                "rarity_pair_reward_max_error": reward_pair_error,
                "negative_hidden_overlap_count": overlap_hidden,
                "mean_useful_utility": float(split["useful_utility"].mean()),
                "mean_unhelpful_utility": float(split["unhelpful_utility"].mean()),
            }
        bias = self.initial_action_bias()
        pair_gap = bias[0::2] - bias[1::2]
        expected_gap = float(nested(self.config, "policy", "initial_rarity_logit_gap"))
        bias_error = float((pair_gap - expected_gap).abs().max())
        passed = passed and bias_error <= 1.0e-7
        result.update(
            {
                "passed": passed,
                "prototype_count": self.prototype_count,
                "action_count": self.action_count,
                "rarity_bias_gap_max_error": bias_error,
                "expected_rarity_logit_gap": expected_gap,
            }
        )
        return result


class CartesianPolicy(nn.Module):
    def __init__(self, config: Mapping[str, Any], environment: CartesianSemanticEnvironment):
        super().__init__()
        hidden_dim = int(nested(config, "policy", "hidden_dim"))
        self.trunk = nn.Sequential(
            nn.Linear(environment.state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.direction_head = nn.Linear(hidden_dim, environment.semantic_dim)
        self.reference_trunk = copy.deepcopy(self.trunk)
        self.reference_direction_head = copy.deepcopy(self.direction_head)
        for parameter in self.reference_trunk.parameters():
            parameter.requires_grad_(False)
        for parameter in self.reference_direction_head.parameters():
            parameter.requires_grad_(False)
        self.fixed_concentration = float(nested(config, "policy", "fixed_concentration"))
        self.action_bias = nn.Parameter(environment.initial_action_bias())

    def forward(self, states: torch.Tensor, action_embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        direction = unit(self.direction_head(self.trunk(states)))
        with torch.no_grad():
            reference_direction = unit(
                self.reference_direction_head(self.reference_trunk(states))
            )
        semantic_residual = (direction - reference_direction) @ action_embeddings.T
        logits = self.fixed_concentration * semantic_residual + self.action_bias
        return logits, direction


def trainable_parameters(model: nn.Module) -> tuple[nn.Parameter, ...]:
    return tuple(parameter for parameter in model.parameters() if parameter.requires_grad)


def batch_indices(seed: int, step: int, count: int, batch_size: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(900_000_003 + int(seed) * 100_003 + int(step))
    return torch.randint(0, count, (batch_size,), generator=gen)


def gather_log_probs(log_probs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    if actions.ndim == 1:
        return log_probs.gather(1, actions[:, None]).squeeze(1)
    return log_probs.gather(1, actions)


def taper_coefficients(rho: float) -> dict[str, float]:
    if not 0.0 < rho < 1.0:
        raise ValueError("reference retention must be in (0,1)")
    return {
        "reciprocal_linear": 1.0 / rho - 1.0,
        "reciprocal_quadratic": 1.0 / rho - 1.0,
        "exponential": -math.log(rho),
    }


def taper_weight(u: torch.Tensor, family: str, coefficient: float) -> torch.Tensor:
    u = torch.clamp(u.detach(), min=0.0)
    if family == "reciprocal_linear":
        return 1.0 / (1.0 + coefficient * u)
    if family == "reciprocal_quadratic":
        return 1.0 / (1.0 + coefficient * u.square())
    if family == "exponential":
        return torch.exp(-coefficient * u)
    raise ValueError(f"unknown taper family: {family}")


def cell_log_probs(
    model: CartesianPolicy,
    environment: CartesianSemanticEnvironment,
    split: Mapping[str, torch.Tensor],
    index: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    states = split["states"][index]
    logits, _ = model(states, environment.action_embeddings)
    log_probs = F.log_softmax(logits, dim=-1)
    positive = gather_log_probs(log_probs, split["positive"][index]).mean(1)
    useful_pair = gather_log_probs(log_probs, split["useful_pair"][index])
    unhelpful_pair = gather_log_probs(log_probs, split["unhelpful_pair"][index])
    # Rarity is defined relative to the current learner. Reassign common/rare
    # within each utility-matched replica pair at every forward pass. The
    # max/min index choice is discrete; gradients only flow through the chosen
    # action log-probability.
    cells = {
        "useful_common": useful_pair.max(dim=1).values,
        "useful_rare": useful_pair.min(dim=1).values,
        "unhelpful_common": unhelpful_pair.max(dim=1).values,
        "unhelpful_rare": unhelpful_pair.min(dim=1).values,
    }
    return positive, cells, logits


def coordinate_calibration(
    model: CartesianPolicy,
    environment: CartesianSemanticEnvironment,
    config: Mapping[str, Any],
) -> dict[str, float]:
    count = min(int(nested(config, "optimization", "audit_states")), environment.train_count)
    index = torch.arange(count)
    with torch.no_grad():
        _, cells, _ = cell_log_probs(model, environment, environment.train, index)
    common = torch.cat([-cells["useful_common"], -cells["unhelpful_common"]])
    rare = torch.cat([-cells["useful_rare"], -cells["unhelpful_rare"]])
    common_center = float(common.median())
    rare_center = float(rare.median())
    scale = rare_center - common_center
    if scale <= float(nested(config, "taper", "minimum_calibration_gap")):
        raise RuntimeError(f"initial common/rare surprisal gap too small: {scale}")
    useful_common = -cells["useful_common"]
    useful_rare = -cells["useful_rare"]
    unhelpful_common = -cells["unhelpful_common"]
    unhelpful_rare = -cells["unhelpful_rare"]
    common_utility_axis_error = float((useful_common - unhelpful_common).abs().max())
    rare_utility_axis_error = float((useful_rare - unhelpful_rare).abs().max())
    useful_rarity_gap_error = float(
        ((useful_rare - useful_common) - scale).abs().max()
    )
    unhelpful_rarity_gap_error = float(
        ((unhelpful_rare - unhelpful_common) - scale).abs().max()
    )
    exact_tolerance = float(nested(config, "taper", "initial_cartesian_tolerance"))
    exact_cartesian = all(
        value <= exact_tolerance
        for value in (
            common_utility_axis_error,
            rare_utility_axis_error,
            useful_rarity_gap_error,
            unhelpful_rarity_gap_error,
        )
    )
    if not exact_cartesian:
        raise RuntimeError(
            "initial utility x rarity probability grid is not exact: "
            f"common={common_utility_axis_error}, rare={rare_utility_axis_error}, "
            f"useful_gap={useful_rarity_gap_error}, "
            f"unhelpful_gap={unhelpful_rarity_gap_error}"
        )
    return {
        "threshold": common_center,
        "scale": scale,
        "common_surprisal_median": common_center,
        "rare_surprisal_median": rare_center,
        "rare_minus_common_median": scale,
        "initial_common_surprisal_utility_axis_max_error": common_utility_axis_error,
        "initial_rare_surprisal_utility_axis_max_error": rare_utility_axis_error,
        "initial_useful_rarity_gap_max_error": useful_rarity_gap_error,
        "initial_unhelpful_rarity_gap_max_error": unhelpful_rarity_gap_error,
        "initial_cartesian_exact": exact_cartesian,
    }


def normalized_excess_surprisal(log_prob: torch.Tensor, calibration: Mapping[str, float]) -> torch.Tensor:
    surprisal = -log_prob
    return F.relu((surprisal - float(calibration["threshold"])) / float(calibration["scale"]))


def active_cell_loss(
    cells: Mapping[str, torch.Tensor],
    spec: MethodSpec,
    calibration: Mapping[str, float],
    coefficients: Mapping[str, float],
    global_scale: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if not spec.active_cells:
        zero = next(iter(cells.values())).sum() * 0.0
        return zero, {f"weight_{cell}": 0.0 for cell in CELL_NAMES}
    pieces: list[torch.Tensor] = []
    diagnostics: dict[str, float] = {}
    for cell in spec.active_cells:
        lp = cells[cell]
        if spec.taper_family is None:
            weight = torch.ones_like(lp)
        elif spec.taper_family == "global":
            weight = torch.full_like(lp, float(global_scale))
        else:
            u = normalized_excess_surprisal(lp, calibration)
            weight = taper_weight(u, spec.taper_family, float(coefficients[spec.taper_family]))
        pieces.append((weight * lp).mean())
        diagnostics[f"weight_{cell}"] = float(weight.detach().mean())
    for cell in CELL_NAMES:
        diagnostics.setdefault(f"weight_{cell}", 0.0)
    # Deleting cells must not rescale the cells that remain. Every cell keeps
    # the same 1/4 coefficient it has in all_negative.
    return torch.stack(pieces).sum() / float(len(CELL_NAMES)), diagnostics


def flat_grad_norm(loss: torch.Tensor, parameters: Sequence[nn.Parameter], retain_graph: bool = True) -> float:
    grads = torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True)
    total = torch.zeros((), dtype=torch.float64)
    for grad in grads:
        if grad is not None:
            total += grad.detach().double().square().sum().cpu()
    return float(torch.sqrt(total))


def calibrate_global_match(
    model: CartesianPolicy,
    environment: CartesianSemanticEnvironment,
    config: Mapping[str, Any],
    calibration: Mapping[str, float],
    coefficients: Mapping[str, float],
) -> dict[str, float]:
    count = min(int(nested(config, "optimization", "audit_states")), environment.train_count)
    index = torch.arange(count)
    _, cells, _ = cell_log_probs(model, environment, environment.train, index)
    params = trainable_parameters(model)
    all_spec = MethodSpec("all_negative", CELL_NAMES)
    exp_spec = MethodSpec("exponential", CELL_NAMES, "exponential")
    raw_loss, _ = active_cell_loss(cells, all_spec, calibration, coefficients, 1.0)
    exp_loss, _ = active_cell_loss(cells, exp_spec, calibration, coefficients, 1.0)
    raw_norm = flat_grad_norm(raw_loss, params, retain_graph=True)
    exp_norm = flat_grad_norm(exp_loss, params, retain_graph=False)
    scale = exp_norm / max(raw_norm, EPS)
    return {
        "raw_negative_gradient_norm": raw_norm,
        "exponential_negative_gradient_norm": exp_norm,
        "global_scale": scale,
        "matched_norm_error": abs(scale * raw_norm - exp_norm),
    }


def evaluate(
    model: CartesianPolicy,
    environment: CartesianSemanticEnvironment,
    split: Mapping[str, torch.Tensor],
    calibration: Mapping[str, float],
) -> dict[str, float]:
    with torch.no_grad():
        logits, _ = model(split["states"], environment.action_embeddings)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        expected_reward = (probs * split["reward_matrix"]).sum(1).mean()
        hidden_prob = probs.gather(1, split["hidden_actions"]).sum(1).mean()
        positive_prob = probs.gather(1, split["positive"]).sum(1).mean()
        entropy = -(probs * log_probs).sum(1).mean()
        effective_support = entropy.exp()
        result = {
            "expected_semantic_reward": float(expected_reward),
            "hidden_optimal_family_probability": float(hidden_prob),
            "positive_support_probability": float(positive_prob),
            "entropy_mean": float(entropy),
            "effective_support": float(effective_support),
            "action_bias_common_mean": float(model.action_bias[0::2].mean()),
            "action_bias_rare_mean": float(model.action_bias[1::2].mean()),
            "action_bias_gap_mean": float((model.action_bias[0::2] - model.action_bias[1::2]).mean()),
        }
        useful_pair_lp = gather_log_probs(log_probs, split["useful_pair"])
        unhelpful_pair_lp = gather_log_probs(log_probs, split["unhelpful_pair"])
        dynamic_cells = {
            "useful_common": useful_pair_lp.max(dim=1).values,
            "useful_rare": useful_pair_lp.min(dim=1).values,
            "unhelpful_common": unhelpful_pair_lp.max(dim=1).values,
            "unhelpful_rare": unhelpful_pair_lp.min(dim=1).values,
        }
        for cell, lp in dynamic_cells.items():
            result[f"{cell}_surprisal_mean"] = float((-lp).mean())
            result[f"{cell}_probability_mean"] = float(lp.exp().mean())
            result[f"{cell}_normalized_excess_mean"] = float(
                normalized_excess_surprisal(lp, calibration).mean()
            )
        result["useful_rarity_role_swap_fraction"] = float(
            (useful_pair_lp[:, 1] > useful_pair_lp[:, 0]).float().mean()
        )
        result["unhelpful_rarity_role_swap_fraction"] = float(
            (unhelpful_pair_lp[:, 1] > unhelpful_pair_lp[:, 0]).float().mean()
        )
        return result


def parameter_vector(model: nn.Module) -> torch.Tensor:
    return torch.cat(
        [parameter.detach().reshape(-1).cpu() for parameter in trainable_parameters(model)]
    )


def terminal_classification(
    trajectory: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    if any(bool(row["nan_inf_numerical_failure"]) for row in trajectory):
        return {"class": "nan_inf_numerical_failure", "formal_acceptance": True}
    if any(bool(row["support_boundary_event"]) for row in trajectory):
        return {"class": "support_boundary", "formal_acceptance": True}
    w1 = list(nested(config, "terminal_audit", "window_1_steps"))
    w2 = list(nested(config, "terminal_audit", "window_2_steps"))
    rows1 = [row for row in trajectory if w1[0] <= int(row["step"]) <= w1[1]]
    rows2 = [row for row in trajectory if w2[0] <= int(row["step"]) <= w2[1]]
    if not rows1 or not rows2:
        return {"class": "incomplete_terminal_windows", "formal_acceptance": False}
    tolerances = nested(config, "terminal_audit", "metric_window_mean_abs_tolerances")
    deltas: dict[str, float] = {}
    passed = True
    for metric, tolerance in tolerances.items():
        first = float(np.mean([float(row[metric]) for row in rows1]))
        second = float(np.mean([float(row[metric]) for row in rows2]))
        delta = abs(second - first)
        deltas[metric] = delta
        passed = passed and delta <= float(tolerance)
    return {
        "class": "terminal_plateau" if passed else "persistent_drift_or_inconclusive",
        "formal_acceptance": passed,
        "window_mean_abs_deltas": deltas,
    }


def run_one(
    config: Mapping[str, Any],
    seed: int,
    spec: MethodSpec,
    base_state: Mapping[str, torch.Tensor],
    calibration: Mapping[str, float],
    global_match: Mapping[str, float],
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seed_all(seed)
    environment = CartesianSemanticEnvironment(config, seed)
    model = CartesianPolicy(config, environment).to(device)
    model.load_state_dict(base_state)
    environment.action_embeddings = environment.action_embeddings.to(device)
    for split in (environment.train, environment.test):
        for key, value in list(split.items()):
            if isinstance(value, torch.Tensor):
                split[key] = value.to(device)
    optimizer = torch.optim.Adam(
        trainable_parameters(model),
        lr=float(nested(config, "optimization", "learning_rate")),
        betas=tuple(float(x) for x in nested(config, "optimization", "betas")),
        eps=float(nested(config, "optimization", "eps")),
    )
    maximum_steps = int(nested(config, "optimization", "maximum_steps"))
    eval_every = int(nested(config, "optimization", "evaluation_interval_steps"))
    batch_size = int(nested(config, "optimization", "batch_size"))
    alpha = float(nested(config, "optimization", "negative_alpha"))
    coefficients = taper_coefficients(float(nested(config, "taper", "reference_rare_retention")))
    support_threshold = float(nested(config, "events", "effective_support_boundary"))
    trajectory: list[dict[str, Any]] = []
    last_update_norm = 0.0
    last_diag = {f"weight_{cell}": 0.0 for cell in CELL_NAMES}

    def record(step: int, numerical_failure: bool = False) -> None:
        metrics = evaluate(model, environment, environment.test, calibration)
        row: dict[str, Any] = {
            "seed": seed,
            "method": spec.method,
            "step": step,
            **metrics,
            **last_diag,
            "adam_parameter_update_norm": last_update_norm,
            "support_boundary_event": bool(metrics["effective_support"] < support_threshold),
            "nan_inf_numerical_failure": numerical_failure,
        }
        trajectory.append(row)

    record(0)
    numerical_failure = False
    for step in range(1, maximum_steps + 1):
        index = batch_indices(seed, step, environment.train_count, batch_size).to(device)
        measure_update = step % eval_every == 0 or step == maximum_steps
        before = parameter_vector(model) if measure_update else None
        positive_lp, cells, _ = cell_log_probs(model, environment, environment.train, index)
        positive_loss = -positive_lp.mean()
        negative_loss, last_diag = active_cell_loss(
            cells,
            spec,
            calibration,
            coefficients,
            float(global_match["global_scale"]),
        )
        loss = positive_loss + alpha * negative_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        finite = bool(torch.isfinite(loss.detach())) and all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        )
        if not finite:
            numerical_failure = True
            record(step, True)
            break
        optimizer.step()
        if measure_update:
            after = parameter_vector(model)
            assert before is not None
            last_update_norm = float((after - before).norm())
        if not all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
            numerical_failure = True
            record(step, True)
            break
        if step % eval_every == 0 or step == maximum_steps:
            record(step)

    terminal = terminal_classification(trajectory, config)
    final = trajectory[-1]
    summary = {
        "seed": seed,
        "method": spec.method,
        "active_cells": list(spec.active_cells),
        "taper_family": spec.taper_family,
        "steps_completed": int(final["step"]),
        "terminal_class": terminal["class"],
        "terminal_formal_acceptance": bool(terminal["formal_acceptance"]),
        "task_performance_collapse": False,  # paired label is assigned after all methods finish
        "support_boundary_event": any(bool(row["support_boundary_event"]) for row in trajectory),
        "nan_inf_numerical_failure": numerical_failure,
        "final_expected_semantic_reward": float(final["expected_semantic_reward"]),
        "final_hidden_optimal_family_probability": float(final["hidden_optimal_family_probability"]),
        "final_effective_support": float(final["effective_support"]),
        "final_action_bias_gap_mean": float(final["action_bias_gap_mean"]),
        "coordinate_calibration": dict(calibration),
        "global_match": dict(global_match),
        "terminal_audit": terminal,
    }
    return trajectory, summary



def run_seed_bundle(
    config: Mapping[str, Any], seed: int, specs: Sequence[MethodSpec], device_name: str
) -> dict[str, Any]:
    device = resolve_device(device_name)
    if device.type == "cpu":
        torch.set_num_threads(int(nested(config, "optimization", "cpu_threads_per_run")))
    seed_all(seed)
    environment = CartesianSemanticEnvironment(config, seed)
    audit = environment.audit()
    if not audit["passed"]:
        raise RuntimeError(f"environment audit failed for seed {seed}")
    model = CartesianPolicy(config, environment).to(device)
    environment.action_embeddings = environment.action_embeddings.to(device)
    for split in (environment.train, environment.test):
        for key, value in list(split.items()):
            if isinstance(value, torch.Tensor):
                split[key] = value.to(device)
    base_state = copy.deepcopy(model.state_dict())
    calibration = coordinate_calibration(model, environment, config)
    coefficients = taper_coefficients(float(nested(config, "taper", "reference_rare_retention")))
    match = calibrate_global_match(model, environment, config, calibration, coefficients)
    trajectories: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for spec in specs:
        trajectory, summary = run_one(
            config, seed, spec, base_state, calibration, match, device
        )
        trajectories.extend(trajectory)
        summaries.append(summary)
    return {
        "seed": seed,
        "audit": audit,
        "calibration": {
            "coordinate": calibration,
            "coefficients": coefficients,
            "global_match": match,
        },
        "trajectories": trajectories,
        "summaries": summaries,
    }

def assign_task_collapse(summaries: list[dict[str, Any]], config: Mapping[str, Any]) -> None:
    ratio = float(nested(config, "events", "task_collapse_ratio_to_paired_positive_only"))
    reference = {
        int(row["seed"]): float(row["final_expected_semantic_reward"])
        for row in summaries
        if row["method"] == "positive_only"
    }
    for row in summaries:
        ref = reference[int(row["seed"])]
        row["task_performance_collapse"] = bool(
            float(row["final_expected_semantic_reward"]) < ratio * ref
        )


def paired_effect(values: Sequence[float], seed: int = 12345) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    if array.size == 0:
        return {"mean": None, "ci95": [None, None], "wins": 0}
    draws = rng.choice(array, size=(5000, array.size), replace=True).mean(axis=1)
    return {
        "mean": float(array.mean()),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "wins": int((array > 0).sum()),
    }


def aggregate(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in summaries:
        grouped.setdefault(str(row["method"]), []).append(row)
    out: dict[str, Any] = {}
    positive = {int(row["seed"]): row for row in grouped.get("positive_only", [])}
    for method, rows in grouped.items():
        reward_values = [float(row["final_expected_semantic_reward"]) for row in rows]
        paired = [
            float(row["final_expected_semantic_reward"])
            - float(positive[int(row["seed"])]["final_expected_semantic_reward"])
            for row in rows
            if int(row["seed"]) in positive
        ]
        out[method] = {
            "runs": len(rows),
            "reward_mean": float(np.mean(reward_values)),
            "reward_delta_vs_positive_only": paired_effect(paired),
            "hidden_probability_mean": float(
                np.mean([float(row["final_hidden_optimal_family_probability"]) for row in rows])
            ),
            "effective_support_mean": float(np.mean([float(row["final_effective_support"]) for row in rows])),
            "task_performance_collapse_events": sum(bool(row["task_performance_collapse"]) for row in rows),
            "support_boundary_events": sum(bool(row["support_boundary_event"]) for row in rows),
            "nan_inf_numerical_failures": sum(bool(row["nan_inf_numerical_failure"]) for row in rows),
            "terminal_plateaus": sum(row["terminal_class"] == "terminal_plateau" for row in rows),
        }
    return out


def _summary_index(
    summaries: Sequence[Mapping[str, Any]],
) -> dict[str, dict[int, Mapping[str, Any]]]:
    out: dict[str, dict[int, Mapping[str, Any]]] = {}
    for row in summaries:
        out.setdefault(str(row["method"]), {})[int(row["seed"])] = row
    return out


def paired_metric_effect(
    index: Mapping[str, Mapping[int, Mapping[str, Any]]],
    lhs: str,
    rhs: str,
    metric: str,
) -> dict[str, Any]:
    common_seeds = sorted(set(index.get(lhs, {})) & set(index.get(rhs, {})))
    values = [
        float(index[lhs][seed][metric]) - float(index[rhs][seed][metric])
        for seed in common_seeds
    ]
    result = paired_effect(values)
    result.update({"lhs": lhs, "rhs": rhs, "metric": metric, "seeds": common_seeds})
    return result


def mechanism_report(
    summaries: Sequence[Mapping[str, Any]], aggregate_summary: Mapping[str, Any]
) -> dict[str, Any]:
    index = _summary_index(summaries)
    metrics = (
        "final_expected_semantic_reward",
        "final_hidden_optimal_family_probability",
        "final_effective_support",
    )
    pairs = {
        "rarity_effect_at_useful": ("useful_rare_only", "useful_common_only"),
        "rarity_effect_at_unhelpful": ("unhelpful_rare_only", "unhelpful_common_only"),
        "utility_effect_at_common": ("useful_common_only", "unhelpful_common_only"),
        "utility_effect_at_rare": ("useful_rare_only", "unhelpful_rare_only"),
    }
    contrasts = {
        label: {
            metric: paired_metric_effect(index, lhs, rhs, metric)
            for metric in metrics
        }
        for label, (lhs, rhs) in pairs.items()
    }
    common_seeds = sorted(
        set(index.get("useful_rare_only", {}))
        & set(index.get("useful_common_only", {}))
        & set(index.get("unhelpful_rare_only", {}))
        & set(index.get("unhelpful_common_only", {}))
    )
    interaction: dict[str, Any] = {}
    for metric in metrics:
        values = [
            (
                float(index["useful_rare_only"][seed][metric])
                - float(index["useful_common_only"][seed][metric])
            )
            - (
                float(index["unhelpful_rare_only"][seed][metric])
                - float(index["unhelpful_common_only"][seed][metric])
            )
            for seed in common_seeds
        ]
        interaction[metric] = {
            **paired_effect(values),
            "metric": metric,
            "seeds": common_seeds,
            "definition": "(useful_rare-useful_common)-(unhelpful_rare-unhelpful_common)",
        }
    return {
        "experiment_id": EXPERIMENT_ID,
        "block": "E6_CARTESIAN_MECHANISM",
        "methods": {
            method: aggregate_summary[method]
            for method in MECHANISM_METHODS
            if method in aggregate_summary
        },
        "paired_contrasts": contrasts,
        "utility_x_rarity_interaction": interaction,
        "interpretation_gate": "descriptive_until_terminal_audit_and_repository_closure",
    }


def taper_report(
    summaries: Sequence[Mapping[str, Any]], aggregate_summary: Mapping[str, Any]
) -> dict[str, Any]:
    index = _summary_index(summaries)
    report_methods = (
        "positive_only",
        "all_negative",
        "global_matched",
        "reciprocal_linear",
        "reciprocal_quadratic",
        "exponential",
    )
    metrics = (
        "final_expected_semantic_reward",
        "final_hidden_optimal_family_probability",
        "final_effective_support",
    )
    contrasts: dict[str, Any] = {}
    for candidate in TAPER_METHODS[1:]:
        for control in ("all_negative", "global_matched", "positive_only"):
            label = f"{candidate}_minus_{control}"
            contrasts[label] = {
                metric: paired_metric_effect(index, candidate, control, metric)
                for metric in metrics
            }
    return {
        "experiment_id": EXPERIMENT_ID,
        "block": "E6_TAPER_METHOD_COMPARISON",
        "methods": {
            method: aggregate_summary[method]
            for method in report_methods
            if method in aggregate_summary
        },
        "paired_contrasts": contrasts,
        "no_method_winner_assumed": True,
        "interpretation_gate": "no_ranking_unless_terminal_audit_method_ranking_allowed",
    }


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row.keys() if not isinstance(row[key], (dict, list))})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def execute(config: Mapping[str, Any], output_root: Path, stage: str, device: torch.device) -> int:
    validate_config(config, stage)
    if stage == "formal" and device.type != "cpu":
        raise RuntimeError("the frozen formal protocol requires CPU execution")
    if device.type == "cpu":
        torch.set_num_threads(int(nested(config, "optimization", "cpu_threads_per_run")))
    manifest_path = prepare_output_root(output_root, stage)
    effective = smoke_config(config) if stage == "smoke" else copy.deepcopy(dict(config))
    yaml_dump(output_root / "resolved_config.yaml", effective)
    seeds = [int(x) for x in nested(effective, "seeds", "held_out_formal")]
    specs = method_specs()
    if stage == "smoke":
        smoke_methods = {
            "positive_only", "useful_common_only", "useful_rare_only",
            "all_negative", "global_matched", "reciprocal_linear",
            "reciprocal_quadratic", "exponential",
        }
        specs = [spec for spec in specs if spec.method in smoke_methods]
    expected_runs = len(seeds) * len(specs)
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "stage": stage,
        "registered_scientific_status": effective.get("scientific_status"),
        "scientific_status": "not_run" if stage == "formal" else "pilot",
        "formal_result": stage == "formal",
        "git_commit": git_text("rev-parse", "HEAD"),
        "git_status_porcelain": git_text("status", "--porcelain"),
        "device": str(device),
        "seeds": seeds,
        "methods": [spec.method for spec in specs],
        "expected_runs": expected_runs,
        "source_sha256": sha256_file(Path(__file__)),
    }
    json_dump(manifest_path, manifest)

    audits: list[dict[str, Any]] = []
    calibrations: dict[str, Any] = {}
    all_trajectories: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    coefficients = taper_coefficients(float(nested(effective, "taper", "reference_rare_retention")))

    def accept_bundle(bundle: Mapping[str, Any]) -> None:
        seed = int(bundle["seed"])
        audits.append(dict(bundle["audit"]))
        calibrations[str(seed)] = dict(bundle["calibration"])
        all_trajectories.extend(list(bundle["trajectories"]))
        summaries.extend(list(bundle["summaries"]))
        checkpoint_root = output_root / "checkpoints" / f"seed_{seed}"
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        write_jsonl(checkpoint_root / "trajectories.jsonl", bundle["trajectories"])
        json_dump(checkpoint_root / "per_run_summary.json", bundle["summaries"])
        json_dump(checkpoint_root / "environment_audit.json", bundle["audit"])
        json_dump(checkpoint_root / "coordinate_calibration.json", bundle["calibration"])
        checkpoint = checkpoint_root / "CHECKPOINT_COMPLETE.json"
        json_dump(
            checkpoint,
            {
                "experiment_id": EXPERIMENT_ID,
                "seed": seed,
                "methods_completed": [row["method"] for row in bundle["summaries"]],
                "run_count": len(bundle["summaries"]),
                "scientific_status": "not_run" if stage == "formal" else "pilot",
                "payload_files": [
                    "trajectories.jsonl",
                    "per_run_summary.json",
                    "environment_audit.json",
                    "coordinate_calibration.json",
                ],
            },
        )

    if stage == "formal":
        workers = int(nested(effective, "optimization", "parallel_workers"))
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(run_seed_bundle, effective, seed, specs, str(device)): seed
                for seed in seeds
            }
            for future in concurrent.futures.as_completed(futures):
                accept_bundle(future.result())
    else:
        for seed in seeds:
            accept_bundle(run_seed_bundle(effective, seed, specs, str(device)))

    audits.sort(key=lambda row: int(row["seed"]))
    summaries.sort(key=lambda row: (int(row["seed"]), str(row["method"])))
    all_trajectories.sort(key=lambda row: (int(row["seed"]), str(row["method"]), int(row["step"])))
    assign_task_collapse(summaries, effective)
    aggregate_summary = aggregate(summaries)
    mechanism_summary = mechanism_report(summaries, aggregate_summary)
    taper_summary = taper_report(summaries, aggregate_summary)
    terminal = {
        "experiment_id": EXPERIMENT_ID,
        "expected_runs": expected_runs,
        "actual_runs": len(summaries),
        "mechanism_block_runs": sum(
            row["method"] in MECHANISM_METHODS for row in summaries
        ),
        "taper_block_runs": sum(row["method"] in TAPER_METHODS for row in summaries),
        "all_registered_runs_present": len(summaries) == expected_runs,
        "terminal_class_counts": {
            label: sum(row["terminal_class"] == label for row in summaries)
            for label in sorted({row["terminal_class"] for row in summaries})
        },
        "task_performance_collapse_events": sum(bool(row["task_performance_collapse"]) for row in summaries),
        "support_boundary_events": sum(bool(row["support_boundary_event"]) for row in summaries),
        "nan_inf_numerical_failures": sum(bool(row["nan_inf_numerical_failure"]) for row in summaries),
        "formal_scientific_acceptance": bool(
            stage == "formal"
            and len(summaries) == expected_runs
            and all(bool(row["terminal_formal_acceptance"]) for row in summaries)
        ),
        "method_ranking_allowed": bool(
            stage == "formal"
            and len(summaries) == expected_runs
            and all(bool(row["terminal_formal_acceptance"]) for row in summaries)
        ),
    }
    protocol_freeze = {
        "experiment_id": EXPERIMENT_ID,
        "cartesian_axes": {
            "utility": "ground_truth_directional_utility_of_repulsion",
            "rarity": "current_policy_surprisal_with_exact_common_rare_logit_bias_replica",
        },
        "initial_semantic_logit_rule": "subtract_frozen_initialized_semantic_reference",
        "initial_probability_matching": "useful_equals_unhelpful_within_each_rarity_level_per_context",
        "dynamic_rarity_role_assignment": "higher_pair_probability_is_common_lower_is_rare_each_forward_stop_gradient",
        "subset_intervention_normalization": "inactive_cells_zeroed_remaining_cells_keep_one_quarter_coefficient",
        "four_cells": list(CELL_NAMES),
        "equal_negative_advantage": float(nested(effective, "geometry", "negative_advantage")),
        "equal_samples_per_cell_per_context": 1,
        "coordinate_calibration_rule": "per-seed pre-training common median threshold and rare-minus-common median scale; frozen across all paired methods",
        "taper_reference_rare_retention": float(nested(effective, "taper", "reference_rare_retention")),
        "taper_coefficients": coefficients,
        "negative_alpha": float(nested(effective, "optimization", "negative_alpha")),
        "held_out_seeds": seeds,
        "mechanism_methods": list(MECHANISM_METHODS),
        "taper_methods": list(TAPER_METHODS),
        "formal_device": "cpu",
        "parallel_seed_workers": int(nested(effective, "optimization", "parallel_workers")),
        "no_method_winner_assumed": True,
        "terminology": "same_distribution_held_out_context_generalization",
    }
    json_dump(output_root / "environment_audits.json", audits)
    json_dump(output_root / "coordinate_calibration.json", calibrations)
    write_jsonl(output_root / "trajectories.jsonl", all_trajectories)
    json_dump(output_root / "per_run_summary.json", summaries)
    write_csv(output_root / "per_run_summary.csv", summaries)
    json_dump(output_root / "aggregate_summary.json", aggregate_summary)
    json_dump(output_root / "mechanism_summary.json", mechanism_summary)
    json_dump(output_root / "taper_summary.json", taper_summary)
    json_dump(output_root / "terminal_audit.json", terminal)
    json_dump(output_root / "formal_protocol_freeze.json", protocol_freeze)
    complete = {
        "experiment_id": EXPERIMENT_ID,
        "completed": True,
        "stage": stage,
        "formal_result": stage == "formal",
        "scientific_status": "finite_step_validated" if stage == "formal" else "pilot",
        "expected_runs": expected_runs,
        "actual_runs": len(summaries),
        "terminal_audit_all_checks_passed": terminal["formal_scientific_acceptance"],
        "task_performance_collapse_events": terminal["task_performance_collapse_events"],
        "support_boundary_events": terminal["support_boundary_events"],
        "nan_inf_numerical_failures": terminal["nan_inf_numerical_failures"],
    }
    json_dump(output_root / "RUN_COMPLETE.json", complete)
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("smoke", "formal"), default="smoke")
    parser.add_argument("--device", default="cpu")
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
    config = load_config(args.config)
    return execute(config, args.output_root, args.stage, resolve_device(args.device))


if __name__ == "__main__":
    raise SystemExit(main())
