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
PROTOCOL_REVISION = 2
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
    "reciprocal_linear_distance",
    "reciprocal_quadratic_distance",
    "reciprocal_quartic_distance",
    "exponential_quadratic_distance",
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
        "policy.trainable_action_bias": False,
        "policy.rarity_parameterization": "shared_contextual_residual_head",
        "policy.initial_rarity_logit_gap": 4.0,
        "policy.rarity_residual_head_zero_initialized": True,
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
        "optimization.rarity_logit_anchor_coefficient": 0.25,
        "taper.coordinate": "normalized_excess_current_surprisal",
        "taper.threshold_rule": "pretraining_common_median",
        "taper.scale_rule": "pretraining_rare_minus_common_median",
        "taper.minimum_calibration_gap": 1.0,
        "taper.initial_cartesian_tolerance": 1.0e-6,
        "taper.reference_normalized_rare_coordinate": 1.0,
        "taper.reference_rare_retention": 0.25,
        "taper.global_control": "stepwise_raw_negative_gradient_norm_matched_to_exponential",
        "taper.detach_surprisal_weight": True,
        "taper.dynamic_rarity_role_assignment": True,
        "events.task_collapse_ratio_to_paired_positive_only": 0.2,
        "events.prototype_effective_support_boundary": 1.5,
        "events.rarity_mass_boundary": 1.0e-4,
        "terminal_audit.mode": "formal_two_x_windows",
        "terminal_audit.formal_horizon_steps": 8000,
        "terminal_audit.window_1_steps": [4000, 6000],
        "terminal_audit.window_2_steps": [6000, 8000],
        "terminal_audit.metric_window_mean_abs_tolerances": {
            "expected_semantic_reward": 0.01,
            "hidden_optimal_family_probability": 0.02,
            "prototype_entropy_mean": 0.08,
            "rarity_logit_gap_mean": 0.20,
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
    if float(nested(config, "optimization", "rarity_logit_anchor_coefficient")) <= 0.0:
        raise ValueError("rarity_logit_anchor_coefficient must be positive")
    if stage == "formal":
        if not bool(config.get("formal_parameter_freeze")):
            raise RuntimeError("formal_parameter_freeze must be true")
        if not bool(nested(config, "formal_gate", "enabled")):
            raise RuntimeError("formal_gate.enabled must be true")
        if not bool(nested(config, "approval", "formal_hyperparameters_approved")):
            raise RuntimeError("formal hyperparameters require explicit approval")
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
        MethodSpec("reciprocal_linear_distance", CELL_NAMES, "reciprocal_linear_distance"),
        MethodSpec("reciprocal_quadratic_distance", CELL_NAMES, "reciprocal_quadratic_distance"),
        MethodSpec("reciprocal_quartic_distance", CELL_NAMES, "reciprocal_quartic_distance"),
        MethodSpec("exponential_quadratic_distance", CELL_NAMES, "exponential_quadratic_distance"),
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
        # Policy-side rarity is an orthogonal product coordinate.  Common and
        # rare replicas share task semantics/reward but have opposite signs on
        # one shared contextual rarity axis.  No per-action trainable bias is
        # used.
        self.action_rarity_sign = torch.where(
            self.action_rarity == 0, torch.tensor(1.0), torch.tensor(-1.0)
        )
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

        # Positive evidence is defined at semantic-family level.  The policy
        # objective sums the common/rare replica probabilities for each positive
        # prototype, so Positive-only is exactly neutral to the rarity axis.
        positive_pairs = torch.stack(
            [self.action_id(positive_proto, 0), self.action_id(positive_proto, 1)], dim=-1
        )
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
            "positive_proto": positive_proto,
            "positive_pairs": positive_pairs,
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

    def initial_rarity_half_gap(self) -> float:
        return float(nested(self.config, "policy", "initial_rarity_logit_gap")) / 2.0

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
        expected_gap = float(nested(self.config, "policy", "initial_rarity_logit_gap"))
        result.update(
            {
                "passed": passed,
                "prototype_count": self.prototype_count,
                "action_count": self.action_count,
                "rarity_coordinate_source": "shared_contextual_residual_head",
                "trainable_per_action_bias": False,
                "expected_rarity_logit_gap": expected_gap,
            }
        )
        return result


class CartesianPolicy(nn.Module):
    """Shared categorical policy with orthogonal semantic and rarity coordinates.

    The frozen half-gap creates the initial common/rare probability separation.
    A zero-initialized contextual rarity residual is shared by every action and
    context; action-specific negative updates therefore have collateral effects
    on the whole common/rare partition rather than only on one private bias.
    """

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
        self.rarity_residual_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.rarity_residual_head.weight)
        nn.init.zeros_(self.rarity_residual_head.bias)

        self.reference_trunk = copy.deepcopy(self.trunk)
        self.reference_direction_head = copy.deepcopy(self.direction_head)
        for parameter in self.reference_trunk.parameters():
            parameter.requires_grad_(False)
        for parameter in self.reference_direction_head.parameters():
            parameter.requires_grad_(False)

        self.fixed_concentration = float(nested(config, "policy", "fixed_concentration"))
        self.initial_rarity_half_gap = environment.initial_rarity_half_gap()
        self.register_buffer(
            "action_rarity_sign", environment.action_rarity_sign.clone().float()
        )

    def semantic_residual(
        self, states: torch.Tensor, action_embeddings: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.trunk(states)
        direction = unit(self.direction_head(features))
        with torch.no_grad():
            reference_direction = unit(
                self.reference_direction_head(self.reference_trunk(states))
            )
        residual = self.fixed_concentration * (
            (direction - reference_direction) @ action_embeddings.T
        )
        return residual, direction, features

    def forward(
        self, states: torch.Tensor, action_embeddings: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        semantic_logits, direction, features = self.semantic_residual(
            states, action_embeddings
        )
        rarity_residual = self.rarity_residual_head(features).squeeze(-1)
        rarity_coordinate = self.initial_rarity_half_gap + rarity_residual
        logits = semantic_logits + rarity_coordinate[:, None] * self.action_rarity_sign[None, :]
        return logits, direction

    def rarity_coordinate(self, states: torch.Tensor) -> torch.Tensor:
        features = self.trunk(states)
        return self.initial_rarity_half_gap + self.rarity_residual_head(features).squeeze(-1)


def trainable_parameters(model: nn.Module) -> tuple[nn.Parameter, ...]:
    return tuple(parameter for parameter in model.parameters() if parameter.requires_grad)


def batch_indices(seed: int, step: int, count: int, batch_size: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(900_000_003 + int(seed) * 100_003 + int(step))
    return torch.randint(0, count, (batch_size,), generator=gen)


def gather_log_probs(log_probs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    if actions.ndim == 1:
        return log_probs.gather(1, actions[:, None]).squeeze(1)
    flat = actions.reshape(actions.shape[0], -1)
    gathered = log_probs.gather(1, flat)
    return gathered.reshape(actions.shape)


def taper_coefficients(rho: float) -> dict[str, float]:
    if not 0.0 < rho < 1.0:
        raise ValueError("reference retention must be in (0,1)")
    return {
        "reciprocal_linear_distance": 1.0 / rho - 1.0,
        "reciprocal_quadratic_distance": 1.0 / rho - 1.0,
        "reciprocal_quartic_distance": 1.0 / rho - 1.0,
        "exponential_quadratic_distance": -math.log(rho),
    }


def taper_weight(u: torch.Tensor, family: str, coefficient: float) -> torch.Tensor:
    u = torch.clamp(u.detach(), min=0.0)
    if family == "reciprocal_linear_distance":
        return 1.0 / (1.0 + coefficient * torch.sqrt(u))
    if family == "reciprocal_quadratic_distance":
        return 1.0 / (1.0 + coefficient * u)
    if family == "reciprocal_quartic_distance":
        return 1.0 / (1.0 + coefficient * u.square())
    if family == "exponential_quadratic_distance":
        return torch.exp(-coefficient * u)
    raise ValueError(f"unknown taper family: {family}")


def cell_log_probs(
    model: CartesianPolicy,
    environment: CartesianSemanticEnvironment,
    split: Mapping[str, torch.Tensor],
    index: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    states = split["states"][index]
    semantic_logits, _, _ = model.semantic_residual(states, environment.action_embeddings)
    logits, _ = model(states, environment.action_embeddings)
    log_probs = F.log_softmax(logits, dim=-1)
    # Compute the positive semantic-family objective directly in prototype
    # space.  Algebraically this equals the sum of the two replica
    # probabilities, but the direct form removes the rarity head from the
    # autograd graph exactly instead of relying on floating-point cancellation
    # through action-level softmax/logsumexp.
    prototype_logits = semantic_logits[:, 0::2]
    prototype_log_probs = F.log_softmax(prototype_logits, dim=-1)
    positive = gather_log_probs(
        prototype_log_probs, split["positive_proto"][index]
    ).mean(1)
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


def _grad_norm_from_tensors(grads: Sequence[torch.Tensor | None]) -> float:
    total = torch.zeros((), dtype=torch.float64)
    for grad in grads:
        if grad is not None:
            total += grad.detach().double().square().sum().cpu()
    return float(torch.sqrt(total))


def policy_geometry_audit(
    model: CartesianPolicy,
    environment: CartesianSemanticEnvironment,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail-closed checks for the repaired rarity coordinate.

    The audit verifies that Positive-only is neutral to within-family rarity,
    while rare/common negatives differ through a shared contextual head rather
    than private action biases.
    """

    count = min(int(nested(config, "optimization", "audit_states")), environment.train_count)
    index = torch.arange(count)
    positive_lp, cells, _ = cell_log_probs(model, environment, environment.train, index)
    rarity_params = tuple(model.rarity_residual_head.parameters())
    full_params = trainable_parameters(model)

    positive_grads = torch.autograd.grad(
        -positive_lp.mean(), rarity_params, retain_graph=True, allow_unused=True
    )
    positive_rarity_grad_norm = _grad_norm_from_tensors(positive_grads)

    cell_rarity_norms: dict[str, float] = {}
    cell_full_norms: dict[str, float] = {}
    for idx, cell in enumerate(CELL_NAMES):
        retain = idx < len(CELL_NAMES) - 1
        loss = cells[cell].mean()
        rarity_grads = torch.autograd.grad(
            loss, rarity_params, retain_graph=True, allow_unused=True
        )
        full_grads = torch.autograd.grad(
            loss, full_params, retain_graph=retain, allow_unused=True
        )
        cell_rarity_norms[cell] = _grad_norm_from_tensors(rarity_grads)
        cell_full_norms[cell] = _grad_norm_from_tensors(full_grads)

    useful_ratio = cell_rarity_norms["useful_rare"] / max(
        cell_rarity_norms["useful_common"], EPS
    )
    unhelpful_ratio = cell_rarity_norms["unhelpful_rare"] / max(
        cell_rarity_norms["unhelpful_common"], EPS
    )

    # The semantic-family positive likelihood must be invariant to a global
    # within-pair rarity shift.
    with torch.no_grad():
        original_bias = model.rarity_residual_head.bias.detach().clone()
        baseline_positive, _, _ = cell_log_probs(
            model, environment, environment.train, index
        )
        model.rarity_residual_head.bias.add_(0.75)
        shifted_positive, _, _ = cell_log_probs(
            model, environment, environment.train, index
        )
        model.rarity_residual_head.bias.copy_(original_bias)
    positive_family_shift_error = float(
        (baseline_positive - shifted_positive).abs().max()
    )

    passed = bool(
        positive_rarity_grad_norm <= 1.0e-6
        and positive_family_shift_error <= 2.0e-6
        and useful_ratio >= 5.0
        and unhelpful_ratio >= 5.0
        and cell_rarity_norms["useful_rare"] > 0.0
        and cell_rarity_norms["unhelpful_rare"] > 0.0
    )
    return {
        "passed": passed,
        "positive_rarity_gradient_norm": positive_rarity_grad_norm,
        "positive_family_rarity_shift_max_error": positive_family_shift_error,
        "cell_shared_rarity_gradient_norms": cell_rarity_norms,
        "cell_full_parameter_gradient_norms": cell_full_norms,
        "useful_rare_to_common_shared_rarity_gradient_ratio": useful_ratio,
        "unhelpful_rare_to_common_shared_rarity_gradient_ratio": unhelpful_ratio,
        "trainable_per_action_bias": False,
    }


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


def rarity_logit_anchor_loss(
    model: CartesianPolicy,
    states: torch.Tensor,
) -> torch.Tensor:
    """Quadratic trust region around the frozen initial rarity logit gap.

    The old repair draft used forward KL from the already-low-probability
    reference replica.  Its asymptotic restoring gradient is proportional to
    the tiny reference rare mass and can be weaker than the negative objective,
    so the combined loss can remain unbounded.  Penalizing the shared rarity
    residual directly is zero at initialization, leaves Positive-only exactly
    rarity-neutral, and grows quadratically while the negative log-probability
    objective grows only linearly in the rarity coordinate.  Any positive
    coefficient therefore yields a finite output-level optimum.
    """

    residual = model.rarity_coordinate(states) - model.initial_rarity_half_gap
    return 0.5 * residual.square().mean()


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
    exp_spec = MethodSpec(
        "exponential_quadratic_distance", CELL_NAMES, "exponential_quadratic_distance"
    )
    raw_loss, _ = active_cell_loss(cells, all_spec, calibration, coefficients, 1.0)
    exp_loss, _ = active_cell_loss(cells, exp_spec, calibration, coefficients, 1.0)
    raw_norm = flat_grad_norm(raw_loss, params, retain_graph=True)
    exp_norm = flat_grad_norm(exp_loss, params, retain_graph=False)
    scale = exp_norm / max(raw_norm, EPS)
    return {
        "raw_negative_gradient_norm": raw_norm,
        "exponential_quadratic_distance_negative_gradient_norm": exp_norm,
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
        positive_pair_probs = gather_log_probs(probs.log(), split["positive_pairs"]).exp()
        positive_prob = positive_pair_probs.sum(-1).sum(-1).mean()

        action_entropy = -(probs * log_probs).sum(1)
        action_effective_support = action_entropy.exp()
        prototype_probs = probs.reshape(
            probs.shape[0], environment.prototype_count, environment.rarity_replicas
        ).sum(-1)
        prototype_log_probs = prototype_probs.clamp_min(EPS).log()
        prototype_entropy = -(prototype_probs * prototype_log_probs).sum(1)
        prototype_effective_support = prototype_entropy.exp()

        common_mass = probs[:, 0::2].sum(1)
        rare_mass = probs[:, 1::2].sum(1)
        rarity_coordinate = model.rarity_coordinate(split["states"])
        rarity_gap = 2.0 * rarity_coordinate.abs()

        result = {
            "expected_semantic_reward": float(expected_reward),
            "hidden_optimal_family_probability": float(hidden_prob),
            "positive_support_probability": float(positive_prob),
            "action_entropy_mean": float(action_entropy.mean()),
            "action_effective_support": float(action_effective_support.mean()),
            "prototype_entropy_mean": float(prototype_entropy.mean()),
            "prototype_effective_support": float(prototype_effective_support.mean()),
            "common_total_probability": float(common_mass.mean()),
            "rare_total_probability": float(rare_mass.mean()),
            "rarity_mass_gap_mean": float((common_mass - rare_mass).abs().mean()),
            "rarity_coordinate_mean": float(rarity_coordinate.mean()),
            "rarity_coordinate_abs_mean": float(rarity_coordinate.abs().mean()),
            "rarity_logit_gap_mean": float(rarity_gap.mean()),
            "rarity_residual_head_weight_norm": float(
                model.rarity_residual_head.weight.detach().norm()
            ),
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
    params = trainable_parameters(model)
    optimizer = torch.optim.Adam(
        params,
        lr=float(nested(config, "optimization", "learning_rate")),
        betas=tuple(float(x) for x in nested(config, "optimization", "betas")),
        eps=float(nested(config, "optimization", "eps")),
    )
    maximum_steps = int(nested(config, "optimization", "maximum_steps"))
    eval_every = int(nested(config, "optimization", "evaluation_interval_steps"))
    batch_size = int(nested(config, "optimization", "batch_size"))
    alpha = float(nested(config, "optimization", "negative_alpha"))
    anchor_coefficient = float(
        nested(config, "optimization", "rarity_logit_anchor_coefficient")
    )
    coefficients = taper_coefficients(float(nested(config, "taper", "reference_rare_retention")))
    prototype_support_threshold = float(
        nested(config, "events", "prototype_effective_support_boundary")
    )
    rarity_mass_boundary = float(nested(config, "events", "rarity_mass_boundary"))
    trajectory: list[dict[str, Any]] = []
    last_update_norm = 0.0
    last_diag = {f"weight_{cell}": 0.0 for cell in CELL_NAMES}
    last_diag.update(
        {
            "rarity_logit_anchor_loss": 0.0,
            "negative_raw_gradient_norm": 0.0,
            "negative_target_gradient_norm": 0.0,
            "negative_applied_gradient_norm": 0.0,
            "stepwise_budget_match_error": 0.0,
            "stepwise_global_scale": 0.0,
        }
    )

    def record(step: int, numerical_failure: bool = False) -> None:
        metrics = evaluate(model, environment, environment.test, calibration)
        prototype_boundary = bool(
            metrics["prototype_effective_support"] < prototype_support_threshold
        )
        rarity_boundary = bool(
            min(metrics["common_total_probability"], metrics["rare_total_probability"])
            < rarity_mass_boundary
        )
        row: dict[str, Any] = {
            "seed": seed,
            "method": spec.method,
            "step": step,
            **metrics,
            **last_diag,
            "adam_parameter_update_norm": last_update_norm,
            "prototype_support_boundary_event": prototype_boundary,
            "rarity_mass_boundary_event": rarity_boundary,
            "support_boundary_event": bool(prototype_boundary or rarity_boundary),
            "nan_inf_numerical_failure": numerical_failure,
        }
        trajectory.append(row)

    record(0)
    numerical_failure = False
    raw_spec = MethodSpec("all_negative", CELL_NAMES)
    exp_spec = MethodSpec(
        "exponential_quadratic_distance",
        CELL_NAMES,
        "exponential_quadratic_distance",
    )
    for step in range(1, maximum_steps + 1):
        index = batch_indices(seed, step, environment.train_count, batch_size).to(device)
        measure_update = step % eval_every == 0 or step == maximum_steps
        before = parameter_vector(model) if measure_update else None
        states = environment.train["states"][index]
        positive_lp, cells, _ = cell_log_probs(
            model, environment, environment.train, index
        )
        positive_loss = -positive_lp.mean()

        if spec.taper_family == "global":
            raw_loss, _ = active_cell_loss(
                cells, raw_spec, calibration, coefficients, 1.0
            )
            target_loss, _ = active_cell_loss(
                cells, exp_spec, calibration, coefficients, 1.0
            )
            raw_norm = flat_grad_norm(raw_loss, params, retain_graph=True)
            target_norm = flat_grad_norm(target_loss, params, retain_graph=True)
            scale = target_norm / max(raw_norm, EPS)
            negative_loss, cell_diag = active_cell_loss(
                cells, spec, calibration, coefficients, scale
            )
            applied_norm = scale * raw_norm
            budget_error = abs(applied_norm - target_norm)
            last_diag = {
                **cell_diag,
                "negative_raw_gradient_norm": raw_norm,
                "negative_target_gradient_norm": target_norm,
                "negative_applied_gradient_norm": applied_norm,
                "stepwise_budget_match_error": budget_error,
                "stepwise_global_scale": scale,
            }
        else:
            negative_loss, cell_diag = active_cell_loss(
                cells, spec, calibration, coefficients, 1.0
            )
            last_diag = {
                **cell_diag,
                "negative_raw_gradient_norm": 0.0,
                "negative_target_gradient_norm": 0.0,
                "negative_applied_gradient_norm": 0.0,
                "stepwise_budget_match_error": 0.0,
                "stepwise_global_scale": 1.0 if spec.active_cells else 0.0,
            }

        anchor_loss = rarity_logit_anchor_loss(model, states)
        last_diag["rarity_logit_anchor_loss"] = float(anchor_loss.detach())
        loss = positive_loss + alpha * negative_loss + anchor_coefficient * anchor_loss
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
        "task_performance_collapse": False,
        "prototype_support_boundary_event": any(
            bool(row["prototype_support_boundary_event"]) for row in trajectory
        ),
        "rarity_mass_boundary_event": any(
            bool(row["rarity_mass_boundary_event"]) for row in trajectory
        ),
        "support_boundary_event": any(
            bool(row["support_boundary_event"]) for row in trajectory
        ),
        "nan_inf_numerical_failure": numerical_failure,
        "final_expected_semantic_reward": float(final["expected_semantic_reward"]),
        "final_hidden_optimal_family_probability": float(
            final["hidden_optimal_family_probability"]
        ),
        "final_action_effective_support": float(final["action_effective_support"]),
        "final_prototype_effective_support": float(
            final["prototype_effective_support"]
        ),
        "final_rare_total_probability": float(final["rare_total_probability"]),
        "final_rarity_logit_gap_mean": float(final["rarity_logit_gap_mean"]),
        "max_stepwise_budget_match_error": float(
            max(float(row["stepwise_budget_match_error"]) for row in trajectory)
        ),
        "coordinate_calibration": dict(calibration),
        "global_match_initial_audit": dict(global_match),
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
    model = CartesianPolicy(config, environment)
    geometry_audit = policy_geometry_audit(model, environment, config)
    audit["policy_geometry"] = geometry_audit
    audit["passed"] = bool(audit["passed"] and geometry_audit["passed"])
    if not audit["passed"]:
        raise RuntimeError(f"environment audit failed for seed {seed}")
    model = model.to(device)
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
            "action_effective_support_mean": float(
                np.mean([float(row["final_action_effective_support"]) for row in rows])
            ),
            "prototype_effective_support_mean": float(
                np.mean([float(row["final_prototype_effective_support"]) for row in rows])
            ),
            "rare_total_probability_mean": float(
                np.mean([float(row["final_rare_total_probability"]) for row in rows])
            ),
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
        "final_action_effective_support",
        "final_prototype_effective_support",
        "final_rare_total_probability",
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
        "reciprocal_linear_distance",
        "reciprocal_quadratic_distance",
        "reciprocal_quartic_distance",
        "exponential_quadratic_distance",
    )
    metrics = (
        "final_expected_semantic_reward",
        "final_hidden_optimal_family_probability",
        "final_action_effective_support",
        "final_prototype_effective_support",
        "final_rare_total_probability",
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
            "all_negative", "global_matched", "reciprocal_linear_distance",
            "reciprocal_quadratic_distance", "reciprocal_quartic_distance",
            "exponential_quadratic_distance",
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
        "prototype_support_boundary_events": sum(
            bool(row["prototype_support_boundary_event"]) for row in summaries
        ),
        "rarity_mass_boundary_events": sum(
            bool(row["rarity_mass_boundary_event"]) for row in summaries
        ),
        "support_boundary_events": sum(
            bool(row["support_boundary_event"]) for row in summaries
        ),
        "nan_inf_numerical_failures": sum(
            bool(row["nan_inf_numerical_failure"]) for row in summaries
        ),
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
        "protocol_revision": PROTOCOL_REVISION,
        "cartesian_axes": {
            "utility": "ground_truth_directional_utility_of_repulsion",
            "rarity": "current_policy_surprisal_from_shared_contextual_rarity_coordinate",
        },
        "initial_semantic_logit_rule": "subtract_frozen_initialized_semantic_reference",
        "rarity_parameterization": "frozen_initial_half_gap_plus_zero_initialized_shared_contextual_residual_head",
        "positive_objective": "semantic_family_log_probability_neutral_to_rarity_axis",
        "rarity_logit_anchor_coefficient": float(
            nested(effective, "optimization", "rarity_logit_anchor_coefficient")
        ),
        "global_budget_control": "stepwise_raw_negative_gradient_norm_matched_to_exponential",
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
        "prototype_support_boundary_events": terminal[
            "prototype_support_boundary_events"
        ],
        "rarity_mass_boundary_events": terminal["rarity_mass_boundary_events"],
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
