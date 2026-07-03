#!/usr/bin/env python3
"""D-U1 E6 protocol-revision-4 formal runner.

This runner freezes the scientifically repaired categorical environment after
revision-3 development calibration.  The observed 2x2 utility x rarity lattice
is augmented by task-visible, evaluation-only hidden rare actions so that
shared-rarity support contraction has a real task cost.  All formal methods use
the same model initialization, Adam state, minibatch stream, negative alpha,
rarity anchor, and reference-retention calibration.

Scientific boundaries
---------------------
* Train and test contexts are independently sampled from the same distribution.
* Results are same-distribution held-out-context generalization, not OOD.
* Categorical direct-logit scores are bounded; this is not a Gaussian
  unbounded-gradient experiment.
* Task collapse, support boundary, numerical failure, and environment-invalid
  events are reported separately.
* No method winner is assumed before held-out formal execution.
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
PROTOCOL_REVISION = 4
EPS = 1.0e-12
CELL_NAMES = (
    "useful_common",
    "useful_rare",
    "unhelpful_common",
    "unhelpful_rare",
)
FORMAL_METHODS = (
    "positive_only",
    "all_negative",
    "global_matched",
    "reciprocal_linear_distance",
    "reciprocal_quadratic_distance",
    "exponential_quadratic_distance",
)
TAPER_METHODS = (
    "global_matched",
    "reciprocal_linear_distance",
    "reciprocal_quadratic_distance",
    "exponential_quadratic_distance",
)
ALL_METHODS = FORMAL_METHODS
HISTORICAL_EXCLUDED_METHODS = ("reciprocal_quartic_distance",)


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
        "protocol_revision": PROTOCOL_REVISION,
        "data.state_dim": 6,
        "data.semantic_dim": 4,
        "data.semantic_prototypes": 32,
        "data.hidden_semantic_prototypes": 16,
        "data.rarity_replicas": 2,
        "data.observed_action_count": 64,
        "data.hidden_action_count": 16,
        "data.action_count": 80,
        "data.hidden_optimal_actions_per_state": 4,
        "data.train_states": 2048,
        "data.test_states": 2048,
        "data.positive_prototypes_per_state": 4,
        "geometry.positive_advantage": 1.0,
        "geometry.negative_advantage": -1.0,
        "geometry.target_offset": 0.45,
        "geometry.fixed_advantage": True,
        "geometry.utility_definition": "oracle_expected_reward_derivative_sign",
        "geometry.neutral_observed_reward": 0.4,
        "geometry.positive_observed_reward": 0.7,
        "geometry.useful_negative_reward": -1.0,
        "geometry.unhelpful_negative_reward": 2.2,
        "geometry.hidden_reward_min": 1.5,
        "geometry.hidden_reward_max": 2.2,
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
        "optimization.negative_alpha": 0.5,
        "optimization.rarity_logit_anchor_coefficient": 0.25,
        "optimization.positive_warm_start_steps": 0,
        "taper.coordinate": "normalized_excess_current_surprisal",
        "taper.threshold_rule": "pretraining_common_median",
        "taper.scale_rule": "pretraining_rare_minus_common_median",
        "taper.minimum_calibration_gap": 1.0,
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
        "methods": list(FORMAL_METHODS),
    }


def _config_value(config: Mapping[str, Any], dotted: str) -> Any:
    parts = dotted.split(".")
    return nested(config, *parts) if len(parts) > 1 else config[parts[0]]


def validate_config(config: Mapping[str, Any], stage: str) -> None:
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError(f"experiment_id must be {EXPERIMENT_ID}")
    if int(config.get("protocol_revision", 0)) != PROTOCOL_REVISION:
        raise ValueError(f"protocol_revision must be {PROTOCOL_REVISION}")
    if list(config.get("methods", [])) != list(FORMAL_METHODS):
        raise ValueError("method order must match the frozen six-method formal protocol")
    if any(method in config.get("methods", []) for method in HISTORICAL_EXCLUDED_METHODS):
        raise ValueError("Quartic is historical-only and excluded from the active formal matrix")
    if int(nested(config, "data", "observed_action_count")) != 2 * int(
        nested(config, "data", "semantic_prototypes")
    ):
        raise ValueError("observed_action_count must equal semantic_prototypes * 2")
    if int(nested(config, "data", "action_count")) != int(
        nested(config, "data", "observed_action_count")
    ) + int(nested(config, "data", "hidden_action_count")):
        raise ValueError("action_count must equal observed plus hidden actions")
    if float(nested(config, "optimization", "rarity_logit_anchor_coefficient")) <= 0.0:
        raise ValueError("rarity_logit_anchor_coefficient must be positive")
    if stage == "formal":
        if not bool(config.get("formal_parameter_freeze")):
            raise RuntimeError("formal_parameter_freeze must be true")
        if not bool(nested(config, "formal_gate", "enabled")):
            raise RuntimeError("formal_gate.enabled must be true")
        if not bool(nested(config, "approval", "formal_hyperparameters_approved")):
            raise RuntimeError("formal hyperparameters require explicit approval")
        if bool(nested(config, "development_calibration", "formal_seed_access_allowed")) is not True:
            raise RuntimeError("formal seed access must be explicitly enabled after freeze")
        for key, expected in formal_expected().items():
            actual = _config_value(config, key)
            if actual != expected:
                raise RuntimeError(f"formal config mismatch for {key}: {actual!r} != {expected!r}")


def smoke_config(config: Mapping[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(dict(config))
    out["formal_parameter_freeze"] = False
    out["scientific_status"] = "pilot"
    out["seeds"]["held_out_formal"] = [0]
    out["data"]["train_states"] = 64
    out["data"]["test_states"] = 64
    out["optimization"]["maximum_steps"] = 4
    out["optimization"]["evaluation_interval_steps"] = 2
    out["optimization"]["audit_states"] = 32
    out["terminal_audit"]["window_1_steps"] = [0, 2]
    out["terminal_audit"]["window_2_steps"] = [2, 4]
    return out


@dataclass(frozen=True)
class MethodSpec:
    method: str
    active_cells: tuple[str, ...]
    taper_family: str | None = None


def method_specs(method_names: Sequence[str] | None = None) -> list[MethodSpec]:
    specs = {
        "positive_only": MethodSpec("positive_only", ()),
        "all_negative": MethodSpec("all_negative", CELL_NAMES),
        "global_matched": MethodSpec("global_matched", CELL_NAMES, "global"),
        "reciprocal_linear_distance": MethodSpec(
            "reciprocal_linear_distance", CELL_NAMES, "reciprocal_linear_distance"
        ),
        "reciprocal_quadratic_distance": MethodSpec(
            "reciprocal_quadratic_distance", CELL_NAMES, "reciprocal_quadratic_distance"
        ),
        "exponential_quadratic_distance": MethodSpec(
            "exponential_quadratic_distance", CELL_NAMES, "exponential_quadratic_distance"
        ),
    }
    names = list(FORMAL_METHODS if method_names is None else method_names)
    unknown = [name for name in names if name not in specs]
    if unknown:
        raise ValueError(f"unknown active methods: {unknown}")
    return [specs[name] for name in names]



class CartesianSemanticEnvironment:
    """Exact observed Cartesian lattice plus task-visible hidden rare actions."""

    def __init__(self, config: Mapping[str, Any], seed: int):
        self.config = config
        self.seed = int(seed)
        data = nested(config, "data")
        geometry = nested(config, "geometry")
        self.state_dim = int(data["state_dim"])
        self.semantic_dim = int(data["semantic_dim"])
        self.prototype_count = int(data["semantic_prototypes"])
        self.hidden_prototype_count = int(data.get("hidden_semantic_prototypes", 16))
        self.semantic_family_count = self.prototype_count + self.hidden_prototype_count
        self.rarity_replicas = int(data["rarity_replicas"])
        self.observed_action_count = int(data.get("observed_action_count", self.prototype_count * 2))
        self.hidden_action_count = int(data.get("hidden_action_count", self.hidden_prototype_count))
        self.action_count = int(data.get("action_count", self.observed_action_count + self.hidden_action_count))
        self.hidden_optimal_count = int(data.get("hidden_optimal_actions_per_state", 4))
        self.n_positive = int(data["positive_prototypes_per_state"])
        self.train_count = int(data["train_states"])
        self.test_count = int(data["test_states"])
        self.target_offset = float(geometry["target_offset"])
        self.positive_advantage = float(geometry["positive_advantage"])
        self.negative_advantage = float(geometry["negative_advantage"])
        self.neutral_reward = float(geometry.get("neutral_observed_reward", 0.4))
        self.positive_reward = float(geometry.get("positive_observed_reward", 0.7))
        self.useful_reward = float(geometry.get("useful_negative_reward", 0.0))
        self.unhelpful_reward = float(geometry.get("unhelpful_negative_reward", 1.1))
        self.hidden_reward_min = float(geometry.get("hidden_reward_min", 0.9))
        self.hidden_reward_max = float(geometry.get("hidden_reward_max", 1.4))

        if self.observed_action_count != self.prototype_count * 2:
            raise ValueError("observed_action_count must equal semantic_prototypes * 2")
        if self.hidden_action_count != self.hidden_prototype_count:
            raise ValueError("hidden action/prototype counts must match")
        if self.action_count != self.observed_action_count + self.hidden_action_count:
            raise ValueError("action_count mismatch")

        gen = torch.Generator(device="cpu").manual_seed(410_003 + self.seed)
        half = self.prototype_count // 2
        base = unit(torch.randn(half, self.semantic_dim, generator=gen))
        observed = torch.cat([base, -base], dim=0)
        observed = observed[torch.randperm(self.prototype_count, generator=gen)].contiguous()
        hidden = unit(torch.randn(self.hidden_prototype_count, self.semantic_dim, generator=gen))
        self.prototype_embeddings = torch.cat([observed, hidden], dim=0)

        observed_action_proto = torch.arange(self.prototype_count).repeat_interleave(2)
        hidden_action_proto = torch.arange(self.prototype_count, self.semantic_family_count)
        self.action_prototype = torch.cat([observed_action_proto, hidden_action_proto])
        observed_rarity = torch.tensor([0, 1], dtype=torch.long).repeat(self.prototype_count)
        hidden_rarity = torch.ones(self.hidden_action_count, dtype=torch.long)
        self.action_rarity = torch.cat([observed_rarity, hidden_rarity])
        self.action_rarity_sign = torch.where(
            self.action_rarity == 0,
            torch.tensor(1.0),
            torch.tensor(-1.0),
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

        observed_embeddings = self.prototype_embeddings[: self.prototype_count]
        hidden_embeddings = self.prototype_embeddings[self.prototype_count :]
        positive_proto = (t_plus @ observed_embeddings.T).topk(self.n_positive, dim=1).indices
        banned = torch.zeros(count, self.prototype_count, dtype=torch.bool)
        banned.scatter_(1, positive_proto, True)
        utility_geometry = (
            (t_plus[:, None, :] - observed_embeddings[None, :, :])
            * direction[:, None, :]
        ).sum(-1)
        useful_proto = self._topk_excluding(utility_geometry, banned, 1).squeeze(1)
        banned.scatter_(1, useful_proto[:, None], True)
        unhelpful_proto = self._topk_excluding(-utility_geometry, banned, 1).squeeze(1)

        positive_pairs = torch.stack(
            [self.action_id(positive_proto, 0), self.action_id(positive_proto, 1)], dim=-1
        )
        cells = {
            "useful_common": self.action_id(useful_proto, 0),
            "useful_rare": self.action_id(useful_proto, 1),
            "unhelpful_common": self.action_id(unhelpful_proto, 0),
            "unhelpful_rare": self.action_id(unhelpful_proto, 1),
        }
        useful_pair = torch.stack([cells["useful_common"], cells["useful_rare"]], dim=1)
        unhelpful_pair = torch.stack([cells["unhelpful_common"], cells["unhelpful_rare"]], dim=1)

        reward_matrix = torch.full((count, self.action_count), self.neutral_reward)
        reward_matrix.scatter_(
            1,
            positive_pairs.reshape(count, -1),
            torch.full((count, self.n_positive * 2), self.positive_reward),
        )
        for cell in ("useful_common", "useful_rare"):
            reward_matrix.scatter_(1, cells[cell][:, None], torch.full((count, 1), self.useful_reward))
        for cell in ("unhelpful_common", "unhelpful_rare"):
            reward_matrix.scatter_(1, cells[cell][:, None], torch.full((count, 1), self.unhelpful_reward))

        hidden_similarity = t_star @ hidden_embeddings.T
        hidden_rewards = self.hidden_reward_min + (
            self.hidden_reward_max - self.hidden_reward_min
        ) * (hidden_similarity + 1.0) * 0.5
        reward_matrix[:, self.observed_action_count :] = hidden_rewards
        hidden_optimal_actions = (
            hidden_rewards.topk(self.hidden_optimal_count, dim=1).indices
            + self.observed_action_count
        )

        return {
            "states": states,
            "t_plus": t_plus,
            "direction": direction,
            "t_star": t_star,
            "reward_matrix": reward_matrix,
            "hidden_optimal_actions": hidden_optimal_actions,
            "positive_proto": positive_proto,
            "positive_pairs": positive_pairs,
            "useful_proto": useful_proto,
            "unhelpful_proto": unhelpful_proto,
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
            rows = torch.arange(len(split["states"]))
            uc, ur = split["useful_common"], split["useful_rare"]
            nc, nr = split["unhelpful_common"], split["unhelpful_rare"]
            advantages = torch.stack([split[f"{cell}_advantage"] for cell in CELL_NAMES], dim=1)
            checks = {
                "useful_replica_same_semantic_prototype": bool(torch.equal(uc // 2, ur // 2)),
                "unhelpful_replica_same_semantic_prototype": bool(torch.equal(nc // 2, nr // 2)),
                "rarity_replica_identity_exact": bool(
                    torch.all(uc % 2 == 0) and torch.all(ur % 2 == 1)
                    and torch.all(nc % 2 == 0) and torch.all(nr % 2 == 1)
                ),
                "negative_advantage_equal": bool(torch.all(advantages == self.negative_advantage)),
                "useful_reward_exact": bool(
                    torch.allclose(split["reward_matrix"][rows, uc], torch.full((len(rows),), self.useful_reward))
                    and torch.allclose(split["reward_matrix"][rows, ur], torch.full((len(rows),), self.useful_reward))
                ),
                "unhelpful_reward_exact": bool(
                    torch.allclose(split["reward_matrix"][rows, nc], torch.full((len(rows),), self.unhelpful_reward))
                    and torch.allclose(split["reward_matrix"][rows, nr], torch.full((len(rows),), self.unhelpful_reward))
                ),
                "hidden_ids_valid": bool(
                    torch.all(split["hidden_optimal_actions"] >= self.observed_action_count)
                    and torch.all(split["hidden_optimal_actions"] < self.action_count)
                ),
            }
            split_passed = all(checks.values())
            passed = passed and split_passed
            result["splits"][name] = {"passed": split_passed, **checks}
        result.update(
            {
                "passed": passed,
                "protocol_revision": PROTOCOL_REVISION,
                "observed_action_count": self.observed_action_count,
                "hidden_action_count": self.hidden_action_count,
                "action_count": self.action_count,
                "hidden_actions_share_rare_coordinate": bool(
                    torch.all(self.action_rarity_sign[self.observed_action_count :] < 0)
                ),
                "trainable_per_action_bias": False,
            }
        )
        return result


class CartesianPolicy(nn.Module):
    def __init__(self, config: Mapping[str, Any], environment: CartesianSemanticEnvironment):
        super().__init__()
        hidden_dim = int(nested(config, "policy", "hidden_dim"))
        self.trunk = nn.Sequential(
            nn.Linear(environment.state_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
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
        self.register_buffer("action_rarity_sign", environment.action_rarity_sign.clone().float())

    def semantic_residual(
        self,
        states: torch.Tensor,
        action_embeddings: torch.Tensor,
        reference_direction: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.trunk(states)
        direction = unit(self.direction_head(features))
        if reference_direction is None:
            with torch.no_grad():
                reference_direction = unit(
                    self.reference_direction_head(self.reference_trunk(states))
                )
        residual = self.fixed_concentration * ((direction - reference_direction) @ action_embeddings.T)
        return residual, direction, features

    def forward(self, states: torch.Tensor, action_embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        semantic_logits, direction, features = self.semantic_residual(states, action_embeddings)
        rarity_coordinate = self.initial_rarity_half_gap + self.rarity_residual_head(features).squeeze(-1)
        logits = semantic_logits + rarity_coordinate[:, None] * self.action_rarity_sign[None, :]
        return logits, direction

    def rarity_coordinate(self, states: torch.Tensor) -> torch.Tensor:
        features = self.trunk(states)
        return self.initial_rarity_half_gap + self.rarity_residual_head(features).squeeze(-1)


def trainable_parameters(model: nn.Module) -> tuple[nn.Parameter, ...]:
    return tuple(parameter for parameter in model.parameters() if parameter.requires_grad)


def cache_reference_directions(model: CartesianPolicy, environment: CartesianSemanticEnvironment) -> None:
    with torch.no_grad():
        for split in (environment.train, environment.test):
            split["reference_direction"] = unit(
                model.reference_direction_head(model.reference_trunk(split["states"]))
            )


def batch_indices(seed: int, step: int, count: int, batch_size: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(900_000_003 + int(seed) * 100_003 + int(step))
    return torch.randint(0, count, (batch_size,), generator=gen)


def gather_log_probs(log_probs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    if actions.ndim == 1:
        return log_probs.gather(1, actions[:, None]).squeeze(1)
    flat = actions.reshape(actions.shape[0], -1)
    return log_probs.gather(1, flat).reshape(actions.shape)


def taper_coefficients(rho: float) -> dict[str, float]:
    if not 0.0 < rho < 1.0:
        raise ValueError("reference retention must be in (0,1)")
    return {
        "reciprocal_linear_distance": 1.0 / rho - 1.0,
        "reciprocal_quadratic_distance": 1.0 / rho - 1.0,
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
    reference = split.get("reference_direction")
    reference_batch = None if reference is None else reference[index]
    semantic_logits, _, features = model.semantic_residual(states, environment.action_embeddings, reference_batch)
    rarity_residual = model.rarity_residual_head(features).squeeze(-1)
    rarity_coordinate = model.initial_rarity_half_gap + rarity_residual
    logits = semantic_logits + rarity_coordinate[:, None] * model.action_rarity_sign[None, :]
    log_probs = F.log_softmax(logits, dim=-1)
    prototype_logits = semantic_logits[:, : environment.observed_action_count : 2]
    prototype_log_probs = F.log_softmax(prototype_logits, dim=-1)
    positive = gather_log_probs(prototype_log_probs, split["positive_proto"][index]).mean(1)
    useful_pair = gather_log_probs(log_probs, split["useful_pair"][index])
    unhelpful_pair = gather_log_probs(log_probs, split["unhelpful_pair"][index])
    cells = {
        "useful_common": useful_pair.max(dim=1).values,
        "useful_rare": useful_pair.min(dim=1).values,
        "unhelpful_common": unhelpful_pair.max(dim=1).values,
        "unhelpful_rare": unhelpful_pair.min(dim=1).values,
    }
    return positive, cells, rarity_residual


def normalized_excess_surprisal(log_prob: torch.Tensor, calibration: Mapping[str, float]) -> torch.Tensor:
    return F.relu((-log_prob - float(calibration["threshold"])) / float(calibration["scale"]))


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
    threshold = float(common.median())
    rare_median = float(rare.median())
    scale = rare_median - threshold
    minimum_gap = float(nested(config, "taper", "minimum_calibration_gap"))
    if scale <= minimum_gap:
        raise RuntimeError(f"initial common/rare surprisal gap too small: {scale}")
    return {
        "threshold": threshold,
        "scale": scale,
        "common_surprisal_median": threshold,
        "rare_surprisal_median": rare_median,
        "rare_minus_common_median": scale,
        "initial_cartesian_exact": True,
    }


def rarity_logit_anchor_loss(model: CartesianPolicy, states: torch.Tensor) -> torch.Tensor:
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
    return torch.stack(pieces).sum() / float(len(CELL_NAMES)), diagnostics


def flat_grad_norm(loss: torch.Tensor, parameters: Sequence[nn.Parameter], retain_graph: bool = True) -> float:
    grads = torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True)
    total = torch.zeros((), dtype=torch.float64)
    for grad in grads:
        if grad is not None:
            total += grad.detach().double().square().sum().cpu()
    return float(torch.sqrt(total))


def _oracle_metrics(
    model: CartesianPolicy,
    environment: CartesianSemanticEnvironment,
    split: Mapping[str, torch.Tensor],
) -> dict[str, float]:
    logits, _ = model(split["states"], environment.action_embeddings)
    probs = F.softmax(logits, dim=-1)
    per_state_reward = (probs * split["reward_matrix"]).sum(1)
    effects: dict[str, torch.Tensor] = {}
    valid: list[float] = []
    for cell in CELL_NAMES:
        pair = split["useful_pair"] if cell.startswith("useful") else split["unhelpful_pair"]
        pair_probs = probs.gather(1, pair)
        dynamic_action = pair.gather(1, pair_probs.argmax(dim=1, keepdim=True)).squeeze(1) if cell.endswith("common") else pair.gather(1, pair_probs.argmin(dim=1, keepdim=True)).squeeze(1)
        action_prob = probs.gather(1, dynamic_action[:, None]).squeeze(1)
        action_reward = split["reward_matrix"].gather(1, dynamic_action[:, None]).squeeze(1)
        effect = action_prob * (per_state_reward - action_reward)
        effects[cell] = effect
        valid.append(float(((effect > 0) if cell.startswith("useful") else (effect < 0)).float().mean()))
    probe = float(nested(environment.config, "diagnostics", "rarity_shift_probe"))
    shifted_probs = F.softmax(logits + probe * model.action_rarity_sign[None, :], dim=-1)
    base_reward = per_state_reward.mean()
    shifted_reward = (shifted_probs * split["reward_matrix"]).sum(1).mean()
    hidden = split["hidden_optimal_actions"]
    base_hidden = probs.gather(1, hidden).sum(1).mean()
    shifted_hidden = shifted_probs.gather(1, hidden).sum(1).mean()
    return {
        "utility_oracle_sign_valid_fraction": min(valid),
        "counterfactual_common_shift_reward_delta": float(shifted_reward - base_reward),
        "counterfactual_common_shift_hidden_probability_delta": float(shifted_hidden - base_hidden),
    }


def policy_geometry_audit(
    model: CartesianPolicy,
    environment: CartesianSemanticEnvironment,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    count = min(int(nested(config, "optimization", "audit_states")), environment.train_count)
    index = torch.arange(count)
    positive_lp, cells, _ = cell_log_probs(model, environment, environment.train, index)
    rarity_params = tuple(model.rarity_residual_head.parameters())
    positive_grads = torch.autograd.grad(-positive_lp.mean(), rarity_params, retain_graph=True, allow_unused=True)
    positive_norm = math.sqrt(sum(float(g.detach().double().square().sum()) for g in positive_grads if g is not None))
    norms: dict[str, float] = {}
    for i, cell in enumerate(CELL_NAMES):
        grads = torch.autograd.grad(cells[cell].mean(), rarity_params, retain_graph=i < len(CELL_NAMES)-1, allow_unused=True)
        norms[cell] = math.sqrt(sum(float(g.detach().double().square().sum()) for g in grads if g is not None))
    useful_ratio = norms["useful_rare"] / max(norms["useful_common"], EPS)
    unhelpful_ratio = norms["unhelpful_rare"] / max(norms["unhelpful_common"], EPS)
    with torch.no_grad():
        oracle = _oracle_metrics(model, environment, {k: (v[:count] if isinstance(v, torch.Tensor) and v.shape[0] == environment.train_count else v) for k,v in environment.train.items()})
    min_sign = float(nested(config, "diagnostics", "utility_sign_fraction_min"))
    min_reward_drop = float(nested(config, "diagnostics", "minimum_rarity_shift_reward_drop"))
    min_hidden_drop = float(nested(config, "diagnostics", "minimum_rarity_shift_hidden_probability_drop"))
    passed = bool(
        positive_norm <= 1.0e-6
        and useful_ratio >= 5.0
        and unhelpful_ratio >= 5.0
        and oracle["utility_oracle_sign_valid_fraction"] >= min_sign
        and -oracle["counterfactual_common_shift_reward_delta"] >= min_reward_drop
        and -oracle["counterfactual_common_shift_hidden_probability_delta"] >= min_hidden_drop
    )
    return {
        "passed": passed,
        "positive_rarity_gradient_norm": positive_norm,
        "cell_shared_rarity_gradient_norms": norms,
        "useful_rare_to_common_shared_rarity_gradient_ratio": useful_ratio,
        "unhelpful_rare_to_common_shared_rarity_gradient_ratio": unhelpful_ratio,
        "utility_oracle_sign_valid_fraction": oracle["utility_oracle_sign_valid_fraction"],
        "rarity_shift_reward_drop": -oracle["counterfactual_common_shift_reward_delta"],
        "rarity_shift_hidden_probability_drop": -oracle["counterfactual_common_shift_hidden_probability_delta"],
        "trainable_per_action_bias": False,
    }


def parameter_vector(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1).cpu() for p in trainable_parameters(model)])


def build_positive_warm_start(
    config: Mapping[str, Any],
    seed: int,
    model: CartesianPolicy,
    environment: CartesianSemanticEnvironment,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
    params = trainable_parameters(model)
    optimizer = torch.optim.Adam(
        params,
        lr=float(nested(config, "optimization", "learning_rate")),
        betas=tuple(float(x) for x in nested(config, "optimization", "betas")),
        eps=float(nested(config, "optimization", "eps")),
    )
    steps = int(nested(config, "optimization", "positive_warm_start_steps"))
    batch_size = int(nested(config, "optimization", "batch_size"))
    anchor_coeff = float(nested(config, "optimization", "rarity_logit_anchor_coefficient"))
    initial_gap = float((2.0 * model.rarity_coordinate(environment.train["states"]).abs()).mean().detach())
    initial_vec = parameter_vector(model)
    final_loss = 0.0
    for step in range(1, steps + 1):
        index = batch_indices(seed + 50_000, step, environment.train_count, batch_size).to(device)
        states = environment.train["states"][index]
        positive_lp, _, _ = cell_log_probs(model, environment, environment.train, index)
        loss = -positive_lp.mean() + anchor_coeff * rarity_logit_anchor_loss(model, states)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if not bool(torch.isfinite(loss)) or any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in params):
            raise RuntimeError(f"positive warm-start numerical failure for seed {seed}")
        optimizer.step()
        final_loss = float(loss.detach())
    final_gap = float((2.0 * model.rarity_coordinate(environment.train["states"]).abs()).mean().detach())
    gap_drift = abs(final_gap - initial_gap)
    max_drift = float(nested(config, "diagnostics", "maximum_positive_only_rarity_gap_drift"))
    if gap_drift > max_drift:
        raise RuntimeError(f"positive warm-start changed rarity gap for seed {seed}: {gap_drift}")
    return (
        copy.deepcopy(model.state_dict()),
        copy.deepcopy(optimizer.state_dict()),
        {
            "steps": steps,
            "final_loss": final_loss,
            "parameter_delta_norm": float((parameter_vector(model)-initial_vec).norm()),
            "initial_rarity_logit_gap_mean": initial_gap,
            "final_rarity_logit_gap_mean": final_gap,
            "rarity_logit_gap_abs_drift": gap_drift,
            "numerical_failure": False,
        },
    )


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
        hidden_prob = probs.gather(1, split["hidden_optimal_actions"]).sum(1).mean()
        positive_probs = gather_log_probs(log_probs, split["positive_pairs"]).exp()
        positive_prob = positive_probs.sum(-1).sum(-1).mean()
        entropy = -(probs * log_probs).sum(1)
        action_support = entropy.exp()
        observed_probs = probs[:, : environment.observed_action_count]
        observed_family = observed_probs.reshape(-1, environment.prototype_count, 2).sum(-1)
        hidden_family = probs[:, environment.observed_action_count :]
        family_probs = torch.cat([observed_family, hidden_family], dim=1)
        family_entropy = -(family_probs * family_probs.clamp_min(EPS).log()).sum(1)
        prototype_support = family_entropy.exp()
        common_mass = observed_probs[:, 0::2].sum(1)
        rare_mass = observed_probs[:, 1::2].sum(1) + hidden_family.sum(1)
        rarity_coordinate = model.rarity_coordinate(split["states"])
        result = {
            "expected_semantic_reward": float(expected_reward),
            "hidden_optimal_family_probability": float(hidden_prob),
            "positive_support_probability": float(positive_prob),
            "action_entropy_mean": float(entropy.mean()),
            "action_effective_support": float(action_support.mean()),
            "prototype_entropy_mean": float(family_entropy.mean()),
            "prototype_effective_support": float(prototype_support.mean()),
            "common_total_probability": float(common_mass.mean()),
            "rare_total_probability": float(rare_mass.mean()),
            "rarity_logit_gap_mean": float((2.0 * rarity_coordinate.abs()).mean()),
            "rarity_coordinate_positive_fraction": float((rarity_coordinate > 0).float().mean()),
        }
        useful_pair_lp = gather_log_probs(log_probs, split["useful_pair"])
        unhelpful_pair_lp = gather_log_probs(log_probs, split["unhelpful_pair"])
        dynamic = {
            "useful_common": useful_pair_lp.max(1).values,
            "useful_rare": useful_pair_lp.min(1).values,
            "unhelpful_common": unhelpful_pair_lp.max(1).values,
            "unhelpful_rare": unhelpful_pair_lp.min(1).values,
        }
        for cell, lp in dynamic.items():
            result[f"{cell}_surprisal_mean"] = float((-lp).mean())
            result[f"{cell}_probability_mean"] = float(lp.exp().mean())
            result[f"{cell}_normalized_excess_mean"] = float(normalized_excess_surprisal(lp, calibration).mean())
        result.update(_oracle_metrics(model, environment, split))
        return result


def terminal_classification(trajectory: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    if any(not bool(row.get("environment_valid", True)) for row in trajectory):
        return {"class": "environment_invalid", "formal_acceptance": False}
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
    deltas: dict[str, float] = {}
    passed = True
    for metric, tolerance in nested(config, "terminal_audit", "metric_window_mean_abs_tolerances").items():
        first = float(np.mean([float(row[metric]) for row in rows1]))
        second = float(np.mean([float(row[metric]) for row in rows2]))
        delta = abs(second-first)
        deltas[metric] = delta
        passed = passed and delta <= float(tolerance)
    return {"class": "terminal_plateau" if passed else "persistent_drift_or_inconclusive", "formal_acceptance": passed, "window_mean_abs_deltas": deltas}


def run_one(
    config: Mapping[str, Any],
    seed: int,
    spec: MethodSpec,
    base_state: Mapping[str, torch.Tensor],
    base_optimizer_state: Mapping[str, Any],
    calibration: Mapping[str, float],
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
    cache_reference_directions(model, environment)
    params = trainable_parameters(model)
    optimizer = torch.optim.Adam(
        params,
        lr=float(nested(config, "optimization", "learning_rate")),
        betas=tuple(float(x) for x in nested(config, "optimization", "betas")),
        eps=float(nested(config, "optimization", "eps")),
    )
    optimizer.load_state_dict(copy.deepcopy(base_optimizer_state))
    maximum_steps = int(nested(config, "optimization", "maximum_steps"))
    eval_every = int(nested(config, "optimization", "evaluation_interval_steps"))
    batch_size = int(nested(config, "optimization", "batch_size"))
    alpha = float(nested(config, "optimization", "negative_alpha"))
    anchor_coeff = float(nested(config, "optimization", "rarity_logit_anchor_coefficient"))
    coefficients = taper_coefficients(float(nested(config, "taper", "reference_rare_retention")))
    prototype_threshold = float(nested(config, "events", "prototype_effective_support_boundary"))
    rarity_boundary = float(nested(config, "events", "rarity_mass_boundary"))
    min_sign = float(nested(config, "diagnostics", "utility_sign_fraction_min"))
    trajectory: list[dict[str, Any]] = []
    last_diag = {f"weight_{cell}": 0.0 for cell in CELL_NAMES}
    last_diag.update({"negative_raw_gradient_norm": 0.0, "negative_target_gradient_norm": 0.0, "negative_applied_gradient_norm": 0.0, "stepwise_budget_match_error": 0.0, "stepwise_global_scale": 0.0, "rarity_logit_anchor_loss": 0.0})
    environment_failure = False
    last_update_norm = 0.0

    def record(step: int, numerical_failure: bool = False) -> None:
        nonlocal environment_failure
        metrics = evaluate(model, environment, environment.test, calibration)
        utility_valid = metrics["utility_oracle_sign_valid_fraction"] >= min_sign
        rarity_valid = metrics["rarity_coordinate_positive_fraction"] >= min_sign
        environment_valid = bool(utility_valid and rarity_valid)
        environment_failure = environment_failure or not environment_valid
        prototype_event = metrics["prototype_effective_support"] < prototype_threshold
        rarity_event = min(metrics["common_total_probability"], metrics["rare_total_probability"]) < rarity_boundary
        trajectory.append({
            "seed": seed,
            "method": spec.method,
            "step": step,
            **metrics,
            **last_diag,
            "adam_parameter_update_norm": last_update_norm,
            "environment_utility_sign_valid": bool(utility_valid),
            "environment_rarity_role_valid": bool(rarity_valid),
            "environment_valid": environment_valid,
            "prototype_support_boundary_event": bool(prototype_event),
            "rarity_mass_boundary_event": bool(rarity_event),
            "support_boundary_event": bool(prototype_event or rarity_event),
            "nan_inf_numerical_failure": numerical_failure,
        })

    record(0)
    numerical_failure = False
    raw_spec = MethodSpec("all_negative", CELL_NAMES)
    exp_spec = MethodSpec("exponential_quadratic_distance", CELL_NAMES, "exponential_quadratic_distance")
    for step in range(1, maximum_steps+1):
        index = batch_indices(seed, step, environment.train_count, batch_size).to(device)
        states = environment.train["states"][index]
        positive_lp, cells, _ = cell_log_probs(model, environment, environment.train, index)
        positive_loss = -positive_lp.mean()
        if spec.taper_family == "global":
            raw_loss, _ = active_cell_loss(cells, raw_spec, calibration, coefficients, 1.0)
            target_loss, _ = active_cell_loss(cells, exp_spec, calibration, coefficients, 1.0)
            raw_norm = flat_grad_norm(raw_loss, params, retain_graph=True)
            target_norm = flat_grad_norm(target_loss, params, retain_graph=True)
            scale = target_norm / max(raw_norm, EPS)
            negative_loss, diag = active_cell_loss(cells, spec, calibration, coefficients, scale)
            last_diag = {**diag, "negative_raw_gradient_norm": raw_norm, "negative_target_gradient_norm": target_norm, "negative_applied_gradient_norm": scale*raw_norm, "stepwise_budget_match_error": abs(scale*raw_norm-target_norm), "stepwise_global_scale": scale}
        else:
            negative_loss, diag = active_cell_loss(cells, spec, calibration, coefficients, 1.0)
            last_diag = {**diag, "negative_raw_gradient_norm": 0.0, "negative_target_gradient_norm": 0.0, "negative_applied_gradient_norm": 0.0, "stepwise_budget_match_error": 0.0, "stepwise_global_scale": 1.0 if spec.active_cells else 0.0}
        anchor = rarity_logit_anchor_loss(model, states)
        last_diag["rarity_logit_anchor_loss"] = float(anchor.detach())
        loss = positive_loss + alpha * negative_loss + anchor_coeff * anchor
        measure = step % eval_every == 0 or step == maximum_steps
        before = parameter_vector(model) if measure else None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        finite = bool(torch.isfinite(loss)) and all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in params)
        if not finite:
            numerical_failure = True
            record(step, True)
            break
        optimizer.step()
        if measure:
            assert before is not None
            last_update_norm = float((parameter_vector(model)-before).norm())
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
        "terminal_audit": terminal,
        "task_performance_collapse": False,
        "prototype_support_boundary_event": any(bool(r["prototype_support_boundary_event"]) for r in trajectory),
        "rarity_mass_boundary_event": any(bool(r["rarity_mass_boundary_event"]) for r in trajectory),
        "support_boundary_event": any(bool(r["support_boundary_event"]) for r in trajectory),
        "nan_inf_numerical_failure": numerical_failure,
        "environment_validity_failure": environment_failure,
        "minimum_utility_oracle_sign_valid_fraction": min(float(r["utility_oracle_sign_valid_fraction"]) for r in trajectory),
        "final_expected_semantic_reward": float(final["expected_semantic_reward"]),
        "final_hidden_optimal_family_probability": float(final["hidden_optimal_family_probability"]),
        "final_action_effective_support": float(final["action_effective_support"]),
        "final_prototype_effective_support": float(final["prototype_effective_support"]),
        "final_rare_total_probability": float(final["rare_total_probability"]),
        "final_rarity_logit_gap_mean": float(final["rarity_logit_gap_mean"]),
        "max_stepwise_budget_match_error": max(float(r["stepwise_budget_match_error"]) for r in trajectory),
        "coordinate_calibration": dict(calibration),
    }
    return trajectory, summary


def run_seed_bundle(
    config: Mapping[str, Any], seed: int, specs: Sequence[MethodSpec], device_name: str
) -> dict[str, Any]:
    device = torch.device(device_name)
    if device.type == "cpu":
        torch.set_num_threads(int(nested(config, "optimization", "cpu_threads_per_run")))
    seed_all(seed)
    environment = CartesianSemanticEnvironment(config, seed)
    audit = environment.audit()
    model = CartesianPolicy(config, environment).to(device)
    environment.action_embeddings = environment.action_embeddings.to(device)
    for split in (environment.train, environment.test):
        for key, value in list(split.items()):
            if isinstance(value, torch.Tensor):
                split[key] = value.to(device)
    cache_reference_directions(model, environment)
    initial_geometry = policy_geometry_audit(model, environment, config)
    audit["policy_geometry_initial"] = initial_geometry
    audit["passed"] = bool(audit["passed"] and initial_geometry["passed"])
    if not audit["passed"]:
        raise RuntimeError(f"initial environment audit failed for seed {seed}: {audit}")
    base_state, base_optimizer_state, warm_summary = build_positive_warm_start(config, seed, model, environment, device)
    warm_geometry = policy_geometry_audit(model, environment, config)
    audit["positive_warm_start"] = warm_summary
    audit["policy_geometry_after_warm_start"] = warm_geometry
    audit["passed"] = bool(audit["passed"] and warm_geometry["passed"])
    if not audit["passed"]:
        raise RuntimeError(f"post-warm-start environment audit failed for seed {seed}: {audit}")
    calibration = coordinate_calibration(model, environment, config)
    coefficients = taper_coefficients(float(nested(config, "taper", "reference_rare_retention")))
    trajectories: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for spec in specs:
        trajectory, summary = run_one(config, seed, spec, base_state, base_optimizer_state, calibration, device)
        trajectories.extend(trajectory)
        summaries.append(summary)
    return {
        "seed": seed,
        "audit": audit,
        "calibration": {"coordinate": calibration, "coefficients": coefficients},
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
    if set(reference) != {int(row["seed"]) for row in summaries}:
        raise RuntimeError("paired Positive-only reference missing for one or more seeds")
    for row in summaries:
        ref = reference[int(row["seed"])]
        row["paired_positive_only_reward"] = ref
        row["task_performance_collapse"] = bool(
            float(row["final_expected_semantic_reward"]) < ratio * ref
        )


def paired_effect(values: Sequence[float], seed: int = 12345) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    if array.size == 0:
        return {"mean": None, "ci95": [None, None], "wins": 0, "ties": 0, "losses": 0}
    draws = rng.choice(array, size=(5000, array.size), replace=True).mean(axis=1)
    return {
        "mean": float(array.mean()),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "wins": int((array > 0).sum()),
        "ties": int((array == 0).sum()),
        "losses": int((array < 0).sum()),
    }


def aggregate(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in summaries:
        grouped.setdefault(str(row["method"]), []).append(row)
    positive = {int(row["seed"]): row for row in grouped.get("positive_only", [])}
    out: dict[str, Any] = {}
    for method in FORMAL_METHODS:
        rows = grouped.get(method, [])
        if not rows:
            continue
        paired = [
            float(row["final_expected_semantic_reward"])
            - float(positive[int(row["seed"])]["final_expected_semantic_reward"])
            for row in rows
        ]
        out[method] = {
            "runs": len(rows),
            "reward_mean": float(np.mean([float(row["final_expected_semantic_reward"]) for row in rows])),
            "reward_delta_vs_positive_only": paired_effect(paired),
            "hidden_probability_mean": float(np.mean([float(row["final_hidden_optimal_family_probability"]) for row in rows])),
            "action_effective_support_mean": float(np.mean([float(row["final_action_effective_support"]) for row in rows])),
            "prototype_effective_support_mean": float(np.mean([float(row["final_prototype_effective_support"]) for row in rows])),
            "rare_total_probability_mean": float(np.mean([float(row["final_rare_total_probability"]) for row in rows])),
            "task_performance_collapse_events": sum(bool(row["task_performance_collapse"]) for row in rows),
            "support_boundary_events": sum(bool(row["support_boundary_event"]) for row in rows),
            "nan_inf_numerical_failures": sum(bool(row["nan_inf_numerical_failure"]) for row in rows),
            "environment_validity_failures": sum(bool(row["environment_validity_failure"]) for row in rows),
            "terminal_plateaus": sum(row["terminal_class"] == "terminal_plateau" for row in rows),
        }
    return out


def _summary_index(summaries: Sequence[Mapping[str, Any]]) -> dict[str, dict[int, Mapping[str, Any]]]:
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
    return {
        **paired_effect(values),
        "lhs": lhs,
        "rhs": rhs,
        "metric": metric,
        "seeds": common_seeds,
    }


def mechanism_report(
    audits: Sequence[Mapping[str, Any]], summaries: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    initial = [dict(audit["policy_geometry_initial"]) for audit in audits]
    after = [dict(audit["policy_geometry_after_warm_start"]) for audit in audits]
    return {
        "experiment_id": EXPERIMENT_ID,
        "block": "E6_REV4_ENVIRONMENT_IDENTIFICATION_AUDIT",
        "protocol_revision": PROTOCOL_REVISION,
        "seeds": [int(audit["seed"]) for audit in audits],
        "all_environment_audits_passed": all(bool(audit["passed"]) for audit in audits),
        "minimum_initial_utility_oracle_sign_valid_fraction": min(float(row["utility_oracle_sign_valid_fraction"]) for row in initial),
        "minimum_runtime_utility_oracle_sign_valid_fraction": min(float(row["minimum_utility_oracle_sign_valid_fraction"]) for row in summaries),
        "minimum_initial_rarity_shift_reward_drop": min(float(row["rarity_shift_reward_drop"]) for row in initial),
        "minimum_initial_rarity_shift_hidden_probability_drop": min(float(row["rarity_shift_hidden_probability_drop"]) for row in initial),
        "minimum_rare_common_shared_rarity_gradient_ratio": min(
            min(
                float(row["useful_rare_to_common_shared_rarity_gradient_ratio"]),
                float(row["unhelpful_rare_to_common_shared_rarity_gradient_ratio"]),
            )
            for row in initial + after
        ),
        "hidden_rare_channel_is_evaluation_only": True,
        "interpretation": "environment validity and causal support-cost audit; not a separate method ranking",
    }


def taper_report(
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
    contrasts: dict[str, Any] = {}
    for candidate in (
        "reciprocal_linear_distance",
        "reciprocal_quadratic_distance",
        "exponential_quadratic_distance",
    ):
        for control in ("all_negative", "global_matched", "positive_only"):
            label = f"{candidate}_minus_{control}"
            contrasts[label] = {
                metric: paired_metric_effect(index, candidate, control, metric)
                for metric in metrics
            }
    contrasts["exponential_quadratic_distance_minus_reciprocal_quadratic_distance"] = {
        metric: paired_metric_effect(
            index,
            "exponential_quadratic_distance",
            "reciprocal_quadratic_distance",
            metric,
        )
        for metric in metrics
    }
    return {
        "experiment_id": EXPERIMENT_ID,
        "block": "E6_REV4_FORMAL_TAPER_COMPARISON",
        "methods": {method: aggregate_summary[method] for method in FORMAL_METHODS if method in aggregate_summary},
        "paired_contrasts": contrasts,
        "quartic_active": False,
        "no_method_winner_assumed": True,
        "interpretation_gate": "no ranking unless all formal runs and terminal audits are complete",
    }


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\
")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted(
        {key for row in rows for key in row if not isinstance(row[key], (dict, list))}
    )
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
    seeds = [int(value) for value in nested(effective, "seeds", "held_out_formal")]
    specs = method_specs(effective["methods"])
    expected_runs = len(seeds) * len(specs)
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_revision": PROTOCOL_REVISION,
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
        "quartic_excluded_from_active_matrix": True,
    }
    json_dump(manifest_path, manifest)

    audits: list[dict[str, Any]] = []
    calibrations: dict[str, Any] = {}
    trajectories: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    def accept_bundle(bundle: Mapping[str, Any]) -> None:
        seed = int(bundle["seed"])
        audits.append(dict(bundle["audit"]))
        calibrations[str(seed)] = dict(bundle["calibration"])
        trajectories.extend(list(bundle["trajectories"]))
        summaries.extend(list(bundle["summaries"]))
        checkpoint_root = output_root / "checkpoints" / f"seed_{seed}"
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        write_jsonl(checkpoint_root / "trajectories.jsonl", bundle["trajectories"])
        json_dump(checkpoint_root / "per_run_summary.json", bundle["summaries"])
        json_dump(checkpoint_root / "environment_audit.json", bundle["audit"])
        json_dump(checkpoint_root / "coordinate_calibration.json", bundle["calibration"])
        json_dump(
            checkpoint_root / "CHECKPOINT_COMPLETE.json",
            {
                "experiment_id": EXPERIMENT_ID,
                "protocol_revision": PROTOCOL_REVISION,
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
    trajectories.sort(key=lambda row: (int(row["seed"]), str(row["method"]), int(row["step"])))
    assign_task_collapse(summaries, effective)
    aggregate_summary = aggregate(summaries)
    mechanism_summary = mechanism_report(audits, summaries)
    taper_summary = taper_report(summaries, aggregate_summary)
    terminal = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_revision": PROTOCOL_REVISION,
        "expected_runs": expected_runs,
        "actual_runs": len(summaries),
        "all_registered_runs_present": len(summaries) == expected_runs,
        "terminal_class_counts": {
            label: sum(row["terminal_class"] == label for row in summaries)
            for label in sorted({row["terminal_class"] for row in summaries})
        },
        "environment_validity_failures": sum(bool(row["environment_validity_failure"]) for row in summaries),
        "task_performance_collapse_events": sum(bool(row["task_performance_collapse"]) for row in summaries),
        "prototype_support_boundary_events": sum(bool(row["prototype_support_boundary_event"]) for row in summaries),
        "rarity_mass_boundary_events": sum(bool(row["rarity_mass_boundary_event"]) for row in summaries),
        "support_boundary_events": sum(bool(row["support_boundary_event"]) for row in summaries),
        "nan_inf_numerical_failures": sum(bool(row["nan_inf_numerical_failure"]) for row in summaries),
    }
    terminal["formal_scientific_acceptance"] = bool(
        stage == "formal"
        and terminal["all_registered_runs_present"]
        and terminal["environment_validity_failures"] == 0
        and all(bool(row["terminal_formal_acceptance"]) for row in summaries)
    )
    terminal["method_ranking_allowed"] = terminal["formal_scientific_acceptance"]

    coefficients = taper_coefficients(float(nested(effective, "taper", "reference_rare_retention")))
    protocol_freeze = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_revision": PROTOCOL_REVISION,
        "development_calibration_source": "revision_3_development_grid_seeds_0_4",
        "selection_rule": "strongest_registered_negative_pressure_that_passed_environment_support_numerical_and_terminal_gates",
        "selection_not_conditioned_on_exponential_winning": True,
        "observed_cartesian_cells": list(CELL_NAMES),
        "hidden_rare_actions": int(nested(effective, "data", "hidden_action_count")),
        "hidden_rare_channel_training_role": "evaluation_only",
        "positive_objective": "observed_semantic_family_log_probability_neutral_to_rarity_axis",
        "shared_start_steps": int(nested(effective, "optimization", "positive_warm_start_steps")),
        "negative_alpha": float(nested(effective, "optimization", "negative_alpha")),
        "rarity_logit_anchor_coefficient": float(nested(effective, "optimization", "rarity_logit_anchor_coefficient")),
        "taper_reference_rare_retention": float(nested(effective, "taper", "reference_rare_retention")),
        "taper_coefficients": coefficients,
        "global_budget_control": "stepwise_raw_negative_gradient_norm_matched_to_exponential",
        "formal_methods": list(FORMAL_METHODS),
        "historical_excluded_methods": list(HISTORICAL_EXCLUDED_METHODS),
        "held_out_seeds": seeds,
        "maximum_steps": int(nested(effective, "optimization", "maximum_steps")),
        "terminal_windows": [
            list(nested(effective, "terminal_audit", "window_1_steps")),
            list(nested(effective, "terminal_audit", "window_2_steps")),
        ],
        "formal_device": "cpu",
        "parallel_seed_workers": int(nested(effective, "optimization", "parallel_workers")),
        "no_method_winner_assumed": True,
        "terminology": "same_distribution_held_out_context_generalization",
    }
    json_dump(output_root / "environment_audits.json", audits)
    json_dump(output_root / "coordinate_calibration.json", calibrations)
    write_jsonl(output_root / "trajectories.jsonl", trajectories)
    json_dump(output_root / "per_run_summary.json", summaries)
    write_csv(output_root / "per_run_summary.csv", summaries)
    json_dump(output_root / "aggregate_summary.json", aggregate_summary)
    json_dump(output_root / "mechanism_summary.json", mechanism_summary)
    json_dump(output_root / "taper_summary.json", taper_summary)
    json_dump(output_root / "terminal_audit.json", terminal)
    json_dump(output_root / "formal_protocol_freeze.json", protocol_freeze)
    json_dump(
        output_root / "RUN_COMPLETE.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_revision": PROTOCOL_REVISION,
            "completed": True,
            "stage": stage,
            "formal_result": stage == "formal",
            "scientific_status": "finite_step_validated" if stage == "formal" else "pilot",
            "expected_runs": expected_runs,
            "actual_runs": len(summaries),
            "terminal_audit_all_checks_passed": terminal["formal_scientific_acceptance"],
            "environment_validity_failures": terminal["environment_validity_failures"],
            "task_performance_collapse_events": terminal["task_performance_collapse_events"],
            "support_boundary_events": terminal["support_boundary_events"],
            "nan_inf_numerical_failures": terminal["nan_inf_numerical_failures"],
        },
    )
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
    return execute(load_config(args.config), args.output_root, args.stage, resolve_device(args.device))


if __name__ == "__main__":
    raise SystemExit(main())
