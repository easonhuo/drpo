#!/usr/bin/env python3
"""D-U1 E6 structured conditional-support-gap categorical experiment.

This is a new E6 protocol; it does not overwrite the completed dense semantic
long-run.  Train and test states remain independent draws from the same marginal
state distribution.  The intervention is on conditional state-action coverage:
for exactly half of states, the entire optimal action group is absent from the
logged training roles while that action group remains observed in other states.

The experiment distinguishes:
* task-performance collapse toward/below a random-policy reference;
* support/concentration boundary events; and
* NaN/Inf numerical failure.

Development runs and smoke tests are never formal results.  Formal execution is
accepted only through the dedicated frozen long-run entrypoint and hardened guard.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drpo.du1_e6_semantic import (
    EPS,
    SemanticPolicy,
    assign_grads,
    controlled_gradient,
    parameter_vector,
    seed_all,
    unit,
)

DEVELOPMENT_EXPERIMENT_ID = "D-U1-E6-CONDITIONAL-GAP-DEV-01"
FORMAL_EXPERIMENT_ID = "D-U1-E6-CONDITIONAL-GAP-01"
ALLOWED_EXPERIMENT_IDS = {DEVELOPMENT_EXPERIMENT_ID, FORMAL_EXPERIMENT_ID}


def nested(config: Mapping[str, Any], *keys: str) -> Any:
    value: Any = config
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            raise KeyError(".".join(keys))
        value = value[key]
    return value


def experiment_id(config: Mapping[str, Any]) -> str:
    value = str(config.get("experiment_id", ""))
    if value not in ALLOWED_EXPERIMENT_IDS:
        raise ValueError(f"unsupported experiment_id: {value!r}")
    return value


def is_formal(config: Mapping[str, Any]) -> bool:
    return experiment_id(config) == FORMAL_EXPERIMENT_ID


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def yaml_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def git_text(*args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def prepare_output_root(root: Path, formal: bool) -> Path:
    if not formal:
        if root.exists() and any(root.iterdir()):
            raise RuntimeError(f"development output root must be new or empty: {root}")
        root.mkdir(parents=True, exist_ok=True)
        return root / "run_manifest.json"
    if not root.is_dir():
        raise RuntimeError("formal conditional-gap E6 requires the hardened guard output root")
    guard_path = root / "run_manifest.json"
    if not guard_path.is_file():
        raise RuntimeError("formal run requires guard-owned run_manifest.json")
    guard = json.loads(guard_path.read_text())
    required = {
        "experiment_id": FORMAL_EXPERIMENT_ID,
        "run_class": "formal",
        "execution_state": "running",
    }
    for key, expected in required.items():
        if guard.get(key) != expected:
            raise RuntimeError(f"guard manifest {key} must be {expected!r}, got {guard.get(key)!r}")
    stale = [
        name
        for name in (
            "resolved_config.yaml",
            "scientific_run_manifest.json",
            "environment_audits.json",
            "trajectories.jsonl",
            "per_run_summary.json",
            "per_run_summary.csv",
            "aggregate_summary.json",
            "terminal_audit.json",
            "formal_protocol_freeze.json",
            "RUN_COMPLETE.json",
            "RUN_FAILED.json",
            "runs",
            "checkpoints",
        )
        if (root / name).exists()
    ]
    if stale:
        raise RuntimeError(
            "formal output root contains stale scientific files: " + ", ".join(stale)
        )
    return root / "scientific_run_manifest.json"


def _require_exact(value: Any, expected: Any, label: str) -> None:
    if value != expected:
        raise ValueError(f"{label} is frozen to {expected!r}, got {value!r}")


def validate_common(config: Mapping[str, Any]) -> None:
    experiment_id(config)
    data = nested(config, "data")
    if int(data["state_dim"]) != 6:
        raise ValueError("conditional-gap E6 state_dim is frozen to 6")
    if int(data["semantic_dim"]) != 4:
        raise ValueError("conditional-gap E6 semantic_dim is frozen to 4")
    groups = int(data["action_groups"])
    per_group = int(data["actions_per_group"])
    if groups != 8 or int(data["action_count"]) != groups * per_group:
        raise ValueError("action_count must equal 8 * actions_per_group")
    if int(data["train_states"]) % 2 or int(data["test_states"]) % 2:
        raise ValueError("paired covered/gap construction requires even split sizes")
    _require_exact(float(data["gap_state_fraction"]), 0.5, "data.gap_state_fraction")
    _require_exact(
        float(data["conditional_block_gap_fraction"]),
        0.625,
        "data.conditional_block_gap_fraction",
    )
    _require_exact(
        int(data["observed_action_groups_per_state"]), 3, "data.observed_action_groups_per_state"
    )
    _require_exact(
        data["train_test_relation"],
        "independent_same_distribution",
        "data.train_test_relation",
    )
    _require_exact(
        data["terminology"],
        "same_distribution_structured_state_action_support_gap",
        "data.terminology",
    )
    geometry = nested(config, "geometry")
    _require_exact(bool(geometry["fixed_advantage"]), True, "geometry.fixed_advantage")
    if float(geometry["positive_advantage"]) <= 0 or float(geometry["negative_advantage"]) >= 0:
        raise ValueError("positive/negative advantage signs are invalid")
    _require_exact(int(geometry["correct_group_offset"]), 0, "geometry.correct_group_offset")
    _require_exact(
        int(geometry["proxy_positive_group_offset"]), -1, "geometry.proxy_positive_group_offset"
    )
    _require_exact(
        int(geometry["local_negative_group_offset"]), -2, "geometry.local_negative_group_offset"
    )
    _require_exact(
        int(geometry["far_negative_group_offset"]), -3, "geometry.far_negative_group_offset"
    )
    _require_exact(int(geometry["trap_group_offset"]), 1, "geometry.trap_group_offset")
    _require_exact(float(geometry["correct_group_reward"]), 1.0, "geometry.correct_group_reward")
    _require_exact(float(geometry["proxy_group_reward"]), 0.65, "geometry.proxy_group_reward")
    _require_exact(float(geometry["other_group_reward"]), 0.0, "geometry.other_group_reward")
    if float(geometry["within_group_reward_floor"]) <= 0:
        raise ValueError("within_group_reward_floor must be positive")
    methods = {str(row["method"]) for row in nested(config, "run_matrix")}
    allowed = {
        "positive_only",
        "local_only",
        "uncontrolled",
        "near_zero",
        "far_cap",
        "budget_matched_global",
    }
    if not methods <= allowed:
        raise ValueError(f"unknown methods in run_matrix: {sorted(methods - allowed)}")
    keys = [run_spec_from_mapping(row).key for row in nested(config, "run_matrix")]
    if len(keys) != len(set(keys)):
        raise ValueError("run_matrix contains duplicate specifications")


def validate_formal_config(config: Mapping[str, Any], stage: str) -> None:
    if stage != "formal":
        raise ValueError("formal config may only run with --stage formal")
    _require_exact(config.get("scientific_status"), "not_run", "scientific_status")
    _require_exact(config.get("execution_mode"), "formal_longrun", "execution_mode")
    _require_exact(bool(config.get("formal_parameter_freeze")), True, "formal_parameter_freeze")
    _require_exact(config.get("predecessor"), "D-U1-E6-SEMANTIC-LONGRUN-01", "predecessor")
    _require_exact(
        dict(config.get("approval", {})),
        {
            "user_approved": True,
            "approval_date": "2026-06-27",
            "approval_scope": "large_structured_conditional_support_gap_rerun",
        },
        "approval",
    )
    _require_exact(list(nested(config, "seeds", "development")), [], "seeds.development")
    _require_exact(
        list(nested(config, "seeds", "held_out_formal")),
        list(range(130, 150)),
        "seeds.held_out_formal",
    )
    expected_data = {
        "state_dim": 6,
        "semantic_dim": 4,
        "action_groups": 8,
        "actions_per_group": 32,
        "action_count": 256,
        "train_states": 4096,
        "test_states": 4096,
        "positive_actions_per_state": 4,
        "local_negative_actions_per_state": 1,
        "far_negative_actions_per_state": 4,
        "gap_state_fraction": 0.5,
        "observed_action_groups_per_state": 3,
        "conditional_block_gap_fraction": 0.625,
        "state_distribution": "paired_standard_normal_marginal",
        "train_test_relation": "independent_same_distribution",
        "terminology": "same_distribution_structured_state_action_support_gap",
    }
    _require_exact(dict(nested(config, "data")), expected_data, "data")
    expected_geometry = {
        "correct_group_offset": 0,
        "proxy_positive_group_offset": -1,
        "local_negative_group_offset": -2,
        "far_negative_group_offset": -3,
        "trap_group_offset": 1,
        "correct_group_reward": 1.0,
        "proxy_group_reward": 0.65,
        "other_group_reward": 0.0,
        "within_group_reward_floor": 0.85,
        "positive_advantage": 1.0,
        "negative_advantage": -1.0,
        "fixed_advantage": True,
        "random_action_id_permutation": True,
    }
    _require_exact(dict(nested(config, "geometry")), expected_geometry, "geometry")
    expected_policy = {
        "hidden_dim": 64,
        "fixed_concentration": 8.0,
        "learnable_concentration_floor": 0.05,
        "initial_learnable_concentration": 8.0,
        "learnable_concentration_upper_clamp": False,
        "activation": "tanh",
        "concentration_mode": "fixed",
    }
    _require_exact(dict(nested(config, "policy")), expected_policy, "policy")
    expected_optimization = {
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "betas": [0.9, 0.999],
        "eps": 1.0e-8,
        "batch_size": 128,
        "maximum_steps": 8000,
        "evaluation_interval_steps": 50,
        "audit_states": 512,
        "parallel_workers": 8,
        "far_cap_ratio_to_weighted_local_gradient": 1.0,
    }
    _require_exact(dict(nested(config, "optimization")), expected_optimization, "optimization")
    expected_matrix = [
        {
            "coverage_mode": "optimal_group_covered",
            "method": "positive_only",
            "local_alpha": 0.0,
            "far_lambda": 0.0,
        },
        {
            "coverage_mode": "optimal_group_covered",
            "method": "local_only",
            "local_alpha": 0.5,
            "far_lambda": 0.0,
        },
        {
            "coverage_mode": "optimal_group_covered",
            "method": "local_only",
            "local_alpha": 1.5,
            "far_lambda": 0.0,
        },
        {
            "coverage_mode": "structured_gap",
            "method": "positive_only",
            "local_alpha": 0.0,
            "far_lambda": 0.0,
        },
        {
            "coverage_mode": "structured_gap",
            "method": "local_only",
            "local_alpha": 0.5,
            "far_lambda": 0.0,
        },
        {
            "coverage_mode": "structured_gap",
            "method": "local_only",
            "local_alpha": 1.5,
            "far_lambda": 0.0,
        },
        {
            "coverage_mode": "structured_gap",
            "method": "uncontrolled",
            "local_alpha": 0.5,
            "far_lambda": 4.0,
        },
        {
            "coverage_mode": "structured_gap",
            "method": "near_zero",
            "local_alpha": 0.5,
            "far_lambda": 4.0,
        },
        {
            "coverage_mode": "structured_gap",
            "method": "far_cap",
            "local_alpha": 0.5,
            "far_lambda": 4.0,
        },
        {
            "coverage_mode": "structured_gap",
            "method": "budget_matched_global",
            "local_alpha": 0.5,
            "far_lambda": 4.0,
        },
    ]
    _require_exact(list(nested(config, "run_matrix")), expected_matrix, "run_matrix")
    _require_exact(
        dict(nested(config, "events")),
        {
            "task_collapse_normalized_margin_to_random": 0.2,
            "below_random_is_collapse": True,
            "effective_support_boundary": 1.5,
            "concentration_warning": 80.0,
        },
        "events",
    )
    _require_exact(
        dict(nested(config, "terminal_audit")),
        {
            "mode": "formal_extension_windows",
            "development_reference_horizon_steps": 1000,
            "formal_horizon_steps": 8000,
            "formal_extension_factor": 8.0,
            "window_1_steps": [4000, 6000],
            "window_2_steps": [6000, 8000],
            "metric_window_mean_abs_tolerances": {
                "test_gap_expected_reward": 0.015,
                "test_gap_correct_group_probability": 0.03,
                "test_gap_trap_group_probability": 0.03,
                "test_entropy_mean": 0.08,
            },
            "raw_total_gradient_median_ratio_max": 1.25,
            "adam_update_median_ratio_max": 1.25,
            "require_all_registered_runs": True,
            "allow_scientific_failure_outcomes": True,
        },
        "terminal_audit",
    )
    expected_checkpointing = {
        "seed_block_size": 5,
        "seed_blocks": [
            [130, 131, 132, 133, 134],
            [135, 136, 137, 138, 139],
            [140, 141, 142, 143, 144],
            [145, 146, 147, 148, 149],
        ],
        "persistence": "persistent_local",
        "write_compact_manifest_after_each_block": True,
    }
    _require_exact(dict(nested(config, "checkpointing")), expected_checkpointing, "checkpointing")
    _require_exact(
        dict(nested(config, "formal_gate")),
        {
            "enabled": True,
            "approval_record": "user_approved_2026-06-27_large_structured_gap",
            "frozen_protocol_path": "configs/du1_e6_conditional_gap_longrun.yaml",
            "held_out_seeds": list(range(130, 150)),
        },
        "formal_gate",
    )


def validate_config(config: Mapping[str, Any], stage: str) -> None:
    validate_common(config)
    if is_formal(config):
        validate_formal_config(config, stage)
        return
    if config.get("scientific_status") != "pilot":
        raise ValueError("development config must remain scientific_status=pilot")
    if bool(config.get("formal_parameter_freeze")):
        raise ValueError("development config cannot claim formal parameter freeze")
    if stage == "formal":
        raise RuntimeError("development conditional-gap config cannot launch formally")
    _require_exact(list(nested(config, "seeds", "development")), [0, 1], "seeds.development")
    _require_exact(list(nested(config, "seeds", "held_out_formal")), [], "seeds.held_out_formal")


def smoke_config(config: Mapping[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(dict(config))
    out["execution_mode"] = "engineering_smoke"
    out["seeds"]["development"] = [0]
    out["data"]["train_states"] = 256
    out["data"]["test_states"] = 256
    out["data"]["actions_per_group"] = 8
    out["data"]["action_count"] = 64
    out["policy"]["hidden_dim"] = 8
    out["optimization"]["batch_size"] = 8
    out["optimization"]["maximum_steps"] = 2
    out["optimization"]["evaluation_interval_steps"] = 2
    out["optimization"]["audit_states"] = 8
    out["optimization"]["parallel_workers"] = 1
    out["run_matrix"] = out["run_matrix"][:2]
    out["terminal_audit"] = {
        "mode": "pilot_trailing_windows",
        "trailing_evaluations_per_window": 1,
        "normalized_metric_change_tolerance": 10.0,
        "raw_total_gradient_median_tolerance": 1.0e9,
    }
    return out


@dataclass(frozen=True)
class RunSpec:
    coverage_mode: str
    method: str
    local_alpha: float
    far_lambda: float

    @property
    def key(self) -> str:
        alpha = f"{self.local_alpha:.4f}".rstrip("0").rstrip(".")
        far = f"{self.far_lambda:.4f}".rstrip("0").rstrip(".")
        return f"{self.coverage_mode}__{self.method}__a{alpha}__f{far}"


def run_spec_from_mapping(row: Mapping[str, Any]) -> RunSpec:
    coverage = str(row["coverage_mode"])
    if coverage not in {"optimal_group_covered", "structured_gap"}:
        raise ValueError(f"unknown coverage_mode: {coverage}")
    return RunSpec(
        coverage_mode=coverage,
        method=str(row["method"]),
        local_alpha=float(row["local_alpha"]),
        far_lambda=float(row["far_lambda"]),
    )


def run_specs(config: Mapping[str, Any]) -> list[RunSpec]:
    return [run_spec_from_mapping(row) for row in nested(config, "run_matrix")]


class ConditionalGapEnvironment:
    """Paired same-distribution states with a large structured conditional gap."""

    def __init__(self, config: Mapping[str, Any], seed: int, coverage_mode: str):
        self.config = config
        self.seed = int(seed)
        self.coverage_mode = coverage_mode
        data = nested(config, "data")
        self.state_dim = int(data["state_dim"])
        self.semantic_dim = int(data["semantic_dim"])
        self.action_groups = int(data["action_groups"])
        self.actions_per_group = int(data["actions_per_group"])
        self.action_count = int(data["action_count"])
        self.n_positive = int(data["positive_actions_per_state"])
        self.n_far = int(data["far_negative_actions_per_state"])
        self.train_count = int(data["train_states"])
        self.test_count = int(data["test_states"])
        geometry = nested(config, "geometry")
        self.proxy_offset = int(geometry["proxy_positive_group_offset"])
        self.local_offset = int(geometry["local_negative_group_offset"])
        self.far_offset = int(geometry["far_negative_group_offset"])
        self.trap_offset = int(geometry["trap_group_offset"])
        self.correct_reward = float(geometry["correct_group_reward"])
        self.proxy_reward = float(geometry["proxy_group_reward"])
        self.other_reward = float(geometry["other_group_reward"])
        self.within_floor = float(geometry["within_group_reward_floor"])
        self.positive_advantage = float(geometry["positive_advantage"])
        self.negative_advantage = float(geometry["negative_advantage"])
        self._build_catalogue()
        self.train = self._build_split(self.train_count, 200_003 + self.seed)
        self.test = self._build_split(self.test_count, 300_003 + self.seed)

    def _build_catalogue(self) -> None:
        rows: list[list[float]] = []
        canonical_groups: list[int] = []
        canonical_variants: list[int] = []
        for group in range(self.action_groups):
            theta = 2.0 * math.pi * group / self.action_groups
            for variant in range(self.actions_per_group):
                phi = 2.0 * math.pi * variant / self.actions_per_group
                rows.append(
                    [
                        1.2 * math.cos(theta),
                        1.2 * math.sin(theta),
                        0.35 * math.cos(phi),
                        0.35 * math.sin(phi),
                    ]
                )
                canonical_groups.append(group)
                canonical_variants.append(variant)
        embeddings = unit(torch.tensor(rows, dtype=torch.float32))
        group_tensor = torch.tensor(canonical_groups, dtype=torch.long)
        variant_tensor = torch.tensor(canonical_variants, dtype=torch.long)
        generator = torch.Generator(device="cpu").manual_seed(100_003 + self.seed)
        permutation = torch.randperm(self.action_count, generator=generator)
        self.policy_embeddings = embeddings[permutation].contiguous()
        self.reward_embeddings = self.policy_embeddings.clone()
        self.action_group = group_tensor[permutation].contiguous()
        self.action_variant = variant_tensor[permutation].contiguous()
        self.action_permutation = permutation

    def _paired_states(
        self, count: int, generator: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        half = count // 2
        base = torch.randn(half, self.state_dim, generator=generator)
        magnitude = base[:, 3].abs().clamp_min(1.0e-6)
        covered = base.clone()
        covered[:, 3] = -magnitude
        gap = base.clone()
        gap[:, 3] = magnitude
        states = torch.cat([covered, gap], dim=0)
        gap_mask = torch.cat(
            [torch.zeros(half, dtype=torch.bool), torch.ones(half, dtype=torch.bool)]
        )
        pair_index = torch.arange(half, dtype=torch.long).repeat(2)
        return states, gap_mask, pair_index

    def _build_split(self, count: int, split_seed: int) -> dict[str, torch.Tensor]:
        generator = torch.Generator(device="cpu").manual_seed(split_seed)
        states, gap_mask, pair_index = self._paired_states(count, generator)
        bits = (states[:, :3] >= 0).long()
        target_group = bits[:, 0] + 2 * bits[:, 1] + 4 * bits[:, 2]
        target_angle = torch.atan2(states[:, 5], states[:, 4]).remainder(2.0 * math.pi)
        target_variant = (
            torch.round(target_angle / (2.0 * math.pi) * self.actions_per_group)
            .long()
            .remainder(self.actions_per_group)
        )

        signed_offset = (self.action_group[None, :] - target_group[:, None]).remainder(
            self.action_groups
        )
        group_reward = torch.full(signed_offset.shape, self.other_reward, dtype=torch.float32)
        group_reward[signed_offset == 0] = self.correct_reward
        group_reward[signed_offset == self.action_groups - 1] = self.proxy_reward
        variant_distance = (self.action_variant[None, :] - target_variant[:, None]).abs()
        variant_distance = torch.minimum(
            variant_distance, self.actions_per_group - variant_distance
        )
        within = self.within_floor + (1.0 - self.within_floor) * 0.5 * (
            1.0 + torch.cos(2.0 * math.pi * variant_distance / self.actions_per_group)
        )
        reward_matrix = group_reward * within
        hidden = reward_matrix.argmax(dim=1)

        if self.coverage_mode == "optimal_group_covered":
            positive_group = target_group
        elif self.coverage_mode == "structured_gap":
            positive_group = torch.where(
                gap_mask,
                (target_group + self.proxy_offset).remainder(self.action_groups),
                target_group,
            )
        else:
            raise ValueError(f"unknown coverage_mode: {self.coverage_mode}")
        local_group = (target_group + self.local_offset).remainder(self.action_groups)
        far_group = (target_group + self.far_offset).remainder(self.action_groups)
        trap_group = (target_group + self.trap_offset).remainder(self.action_groups)

        def select(group: torch.Tensor, k: int, *, exclude_hidden: bool = False) -> torch.Tensor:
            mask = self.action_group[None, :] == group[:, None]
            scores = within.masked_fill(~mask, -torch.inf)
            if exclude_hidden:
                scores = scores.clone()
                scores.scatter_(1, hidden[:, None], -torch.inf)
            selected = scores.topk(k, dim=1).indices
            if not bool(torch.isfinite(scores.gather(1, selected)).all()):
                raise RuntimeError("action group is too small for the requested role set")
            return selected

        positive = select(
            positive_group,
            self.n_positive,
            exclude_hidden=True,
        )
        local = select(local_group, 1).squeeze(1)
        far = select(far_group, self.n_far)
        observed_group_mask = torch.zeros(count, self.action_groups, dtype=torch.bool)
        observed_group_mask.scatter_(1, positive_group[:, None], True)
        observed_group_mask.scatter_(1, local_group[:, None], True)
        observed_group_mask.scatter_(1, far_group[:, None], True)
        return {
            "states": states,
            "gap_mask": gap_mask,
            "pair_index": pair_index,
            "target_group": target_group,
            "target_variant": target_variant,
            "positive_group": positive_group,
            "local_group": local_group,
            "far_group": far_group,
            "trap_group": trap_group,
            "reward_matrix": reward_matrix,
            "random_policy_reward": reward_matrix.mean(dim=1),
            "hidden": hidden,
            "positive": positive,
            "local": local,
            "far": far,
            "observed_group_mask": observed_group_mask,
            "positive_advantage": torch.full((count, self.n_positive), self.positive_advantage),
            "local_advantage": torch.full((count,), self.negative_advantage),
            "far_advantage": torch.full((count, self.n_far), self.negative_advantage),
        }

    def audit(self) -> dict[str, Any]:
        split_reports: dict[str, Any] = {}
        passed = True
        for name, split in (("train", self.train), ("test", self.test)):
            rows = torch.arange(split["states"].shape[0])
            gap = split["gap_mask"]
            target = split["target_group"]
            hidden = split["hidden"]
            positive = split["positive"]
            hidden_reward = split["reward_matrix"][rows, hidden]
            max_reward = split["reward_matrix"].max(dim=1).values
            correct_group_mask = self.action_group[None, :] == target[:, None]
            correct_group_rewards = split["reward_matrix"][correct_group_mask]
            correct_group_reward_min = float(correct_group_rewards.min())
            correct_group_reward_max = float(correct_group_rewards.max())
            observed_counts = split["observed_group_mask"].sum(dim=1)
            correct_observed = split["observed_group_mask"].gather(1, target[:, None]).squeeze(1)
            paired_target_equal = bool(
                torch.equal(target[: target.shape[0] // 2], target[target.shape[0] // 2 :])
            )
            paired_nuisance_equal = float(
                (
                    split["states"][: target.shape[0] // 2, [0, 1, 2, 4, 5]]
                    - split["states"][target.shape[0] // 2 :, [0, 1, 2, 4, 5]]
                )
                .abs()
                .max()
            )
            gap_fraction = float(gap.float().mean())
            block_gap_fraction = float((~split["observed_group_mask"]).float().mean())
            gap_correct_observed = int(correct_observed[gap].sum())
            covered_correct_observed = int(correct_observed[~gap].sum())
            hidden_positive_overlap = int((positive == hidden[:, None]).sum())
            all_groups_positive = int(torch.unique(split["positive_group"]).numel())
            split_passed = all(
                [
                    abs(gap_fraction - 0.5) <= 1.0e-12,
                    abs(block_gap_fraction - 0.625) <= 1.0e-12,
                    bool((observed_counts == 3).all()),
                    paired_target_equal,
                    paired_nuisance_equal <= 1.0e-7,
                    float((hidden_reward - max_reward).abs().max()) <= 1.0e-7,
                    abs(correct_group_reward_min - self.within_floor) <= 1.0e-6,
                    abs(correct_group_reward_max - self.correct_reward) <= 1.0e-6,
                    all_groups_positive == self.action_groups,
                    hidden_positive_overlap == 0,
                    covered_correct_observed == int((~gap).sum()),
                    (
                        gap_correct_observed == 0
                        if self.coverage_mode == "structured_gap"
                        else gap_correct_observed == int(gap.sum())
                    ),
                ]
            )
            passed = passed and split_passed
            split_reports[name] = {
                "passed": split_passed,
                "gap_state_fraction": gap_fraction,
                "conditional_block_gap_fraction": block_gap_fraction,
                "observed_action_groups_per_state_unique": sorted(
                    set(int(x) for x in observed_counts.tolist())
                ),
                "gap_correct_group_observed_count": gap_correct_observed,
                "covered_correct_group_observed_count": covered_correct_observed,
                "hidden_positive_overlap_count": hidden_positive_overlap,
                "positive_role_action_groups_covered": all_groups_positive,
                "paired_target_group_equal": paired_target_equal,
                "paired_non_gap_state_coordinates_max_error": paired_nuisance_equal,
                "hidden_is_reward_argmax_max_error": float(
                    (hidden_reward - max_reward).abs().max()
                ),
                "correct_group_reward_min": correct_group_reward_min,
                "correct_group_reward_max": correct_group_reward_max,
                "mean_random_policy_reward": float(split["random_policy_reward"].mean()),
            }
        norm_error = float(self.policy_embeddings.norm(dim=1).sub(1).abs().max())
        permutation_changed = not torch.equal(
            self.action_permutation, torch.arange(self.action_count)
        )
        passed = passed and norm_error <= 1.0e-6 and permutation_changed
        return {
            "seed": self.seed,
            "coverage_mode": self.coverage_mode,
            "passed": passed,
            "policy_embedding_norm_max_error": norm_error,
            "action_id_permutation_changed": permutation_changed,
            "action_id_permutation_checksum": hashlib.sha256(
                self.action_permutation.numpy().tobytes()
            ).hexdigest(),
            "splits": split_reports,
        }


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
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    logits, _, _ = model(states, action_embeddings)
    log_probs = F.log_softmax(logits, dim=-1)
    positive_loss = -log_probs.gather(1, positive).mean()
    local_loss = log_probs.gather(1, local[:, None]).mean()
    far_loss = log_probs.gather(1, far).mean()
    positive_grad = torch.autograd.grad(
        positive_loss, parameters, retain_graph=True, allow_unused=True
    )
    local_grad = torch.autograd.grad(local_loss, parameters, retain_graph=True, allow_unused=True)
    far_grad = torch.autograd.grad(far_loss, parameters, allow_unused=True)
    return (
        positive_grad,
        local_grad,
        far_grad,
        {
            "loss_positive": float(positive_loss.detach()),
            "loss_local": float(local_loss.detach()),
            "loss_far": float(far_loss.detach()),
        },
    )


def training_gradient(
    model: SemanticPolicy,
    states: torch.Tensor,
    positive: torch.Tensor,
    local: torch.Tensor,
    far: torch.Tensor,
    action_embeddings: torch.Tensor,
    spec: RunSpec,
    far_cap_ratio: float,
) -> tuple[torch.Tensor | None, ...]:
    """Return the exact training gradient with a single backward when possible.

    Positive-only, local-only, near-zero, and uncontrolled objectives are linear
    combinations of branch losses, so one autograd call is exactly equivalent
    to summing three separately materialized gradients.  Far-cap and
    budget-matched-global require branch norms and retain the explicit branch
    path.
    """
    if spec.method in {"far_cap", "budget_matched_global"}:
        positive_grad, local_grad, far_grad, _ = gradient_branches(
            model, states, positive, local, far, action_embeddings
        )
        total, _ = controlled_gradient(
            spec.method,
            positive_grad,
            local_grad,
            far_grad,
            spec.local_alpha,
            spec.far_lambda,
            far_cap_ratio,
        )
        return total

    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    logits, _, _ = model(states, action_embeddings)
    log_probs = F.log_softmax(logits, dim=-1)
    positive_loss = -log_probs.gather(1, positive).mean()
    local_loss = log_probs.gather(1, local[:, None]).mean()
    far_loss = log_probs.gather(1, far).mean()
    if spec.method == "positive_only":
        loss = positive_loss
    elif spec.method in {"local_only", "far_zero"}:
        loss = positive_loss + spec.local_alpha * local_loss
    elif spec.method == "near_zero":
        loss = positive_loss + spec.far_lambda * far_loss
    elif spec.method == "uncontrolled":
        loss = positive_loss + spec.local_alpha * local_loss + spec.far_lambda * far_loss
    else:
        raise ValueError(f"unknown method: {spec.method}")
    return torch.autograd.grad(loss, parameters, allow_unused=True)


@torch.no_grad()
def evaluate(
    model: SemanticPolicy,
    environment: ConditionalGapEnvironment,
    split_name: str,
    device: torch.device,
) -> dict[str, float]:
    split = environment.train if split_name == "train" else environment.test
    states = split["states"].to(device)
    logits, _, concentration = model(states, environment.policy_embeddings.to(device))
    probabilities = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    rewards = split["reward_matrix"].to(device)
    gap = split["gap_mask"].to(device)
    covered = ~gap
    entropy = -(probabilities * log_probs).sum(dim=1)
    effective_support = entropy.exp()
    expected_reward = (probabilities * rewards).sum(dim=1)
    groups = environment.action_group.to(device)
    group_probabilities = torch.zeros(states.shape[0], environment.action_groups, device=device)
    group_probabilities.scatter_add_(1, groups[None, :].expand(states.shape[0], -1), probabilities)
    rows = torch.arange(states.shape[0], device=device)
    target = split["target_group"].to(device)
    proxy = (target + environment.proxy_offset).remainder(environment.action_groups)
    trap = split["trap_group"].to(device)
    correct_mass = group_probabilities[rows, target]
    proxy_mass = group_probabilities[rows, proxy]
    trap_mass = group_probabilities[rows, trap]
    top_group = group_probabilities.argmax(dim=1)
    hidden = split["hidden"].to(device)
    hidden_probability = probabilities[rows, hidden]
    half = states.shape[0] // 2
    paired_l1 = (group_probabilities[:half] - group_probabilities[half:]).abs().sum(dim=1)

    def mean_on(values: torch.Tensor, mask: torch.Tensor) -> float:
        return float(values[mask].mean())

    return {
        "expected_reward": float(expected_reward.mean()),
        "covered_expected_reward": mean_on(expected_reward, covered),
        "gap_expected_reward": mean_on(expected_reward, gap),
        "random_policy_reward": float(split["random_policy_reward"].mean()),
        "gap_random_policy_reward": float(split["random_policy_reward"][split["gap_mask"]].mean()),
        "hidden_optimal_probability": float(hidden_probability.mean()),
        "gap_hidden_optimal_probability": mean_on(hidden_probability, gap),
        "gap_correct_group_probability": mean_on(correct_mass, gap),
        "gap_proxy_group_probability": mean_on(proxy_mass, gap),
        "gap_trap_group_probability": mean_on(trap_mass, gap),
        "covered_correct_group_probability": mean_on(correct_mass, covered),
        "gap_top1_group_accuracy": mean_on((top_group == target).float(), gap),
        "covered_top1_group_accuracy": mean_on((top_group == target).float(), covered),
        "paired_group_distribution_l1": float(paired_l1.mean()),
        "entropy_mean": float(entropy.mean()),
        "effective_support_mean": float(effective_support.mean()),
        "effective_support_p05": float(torch.quantile(effective_support, 0.05)),
        "concentration_mean": float(concentration.mean()),
        "concentration_max": float(concentration.max()),
    }


def shared_batches(seed: int, train_count: int, batch_size: int, steps: int) -> list[torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(900_003 + seed)
    return [torch.randint(0, train_count, (batch_size,), generator=generator) for _ in range(steps)]


def audit_gradients(
    model: SemanticPolicy,
    environment: ConditionalGapEnvironment,
    spec: RunSpec,
    config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, float]:
    count = min(
        int(nested(config, "optimization", "audit_states")),
        int(environment.train["states"].shape[0]),
    )
    split = environment.train
    positive, local, far, losses = gradient_branches(
        model,
        split["states"][:count].to(device),
        split["positive"][:count].to(device),
        split["local"][:count].to(device),
        split["far"][:count].to(device),
        environment.policy_embeddings.to(device),
    )
    _, diagnostics = controlled_gradient(
        spec.method,
        positive,
        local,
        far,
        spec.local_alpha,
        spec.far_lambda,
        float(nested(config, "optimization", "far_cap_ratio_to_weighted_local_gradient")),
    )
    return {**losses, **diagnostics}


def terminal_classification(
    trajectory: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    if not trajectory:
        return {"class": "inconclusive", "reason": "empty_trajectory", "formal_acceptance": False}
    final_step = int(trajectory[-1]["step"])
    maximum_steps = int(nested(config, "optimization", "maximum_steps"))
    full_horizon = final_step >= maximum_steps
    if any(bool(row["nan_inf_numerical_failure"]) for row in trajectory):
        return {
            "class": "numerical_failure",
            "reason": "nonfinite_value_observed",
            "completed_steps": final_step,
            "formal_two_x_extension_performed": False,
            "formal_acceptance": is_formal(config),
        }
    audit = nested(config, "terminal_audit")
    if audit["mode"] == "formal_extension_windows":
        bounds = [audit["window_1_steps"], audit["window_2_steps"]]
        windows = [
            [row for row in trajectory if int(lo) <= int(row["step"]) <= int(hi)]
            for lo, hi in bounds
        ]
        if any(not window for window in windows):
            return {
                "class": "formal_terminal_inconclusive",
                "reason": "missing_registered_terminal_window",
                "window_sizes": [len(x) for x in windows],
                "formal_two_x_extension_performed": full_horizon,
                "formal_acceptance": False,
            }
        tolerances = {
            str(k): float(v) for k, v in audit["metric_window_mean_abs_tolerances"].items()
        }
        means = [
            {
                metric: float(np.mean([float(row[metric]) for row in window]))
                for metric in tolerances
            }
            for window in windows
        ]
        deltas = {metric: abs(means[1][metric] - means[0][metric]) for metric in tolerances}
        grad_medians = [
            float(np.median([float(row["audit_raw_total_gradient_norm"]) for row in window]))
            for window in windows
        ]
        update_medians = [
            float(np.median([float(row["adam_parameter_update_norm"]) for row in window]))
            for window in windows
        ]
        grad_ratio = grad_medians[1] / max(grad_medians[0], EPS)
        update_ratio = update_medians[1] / max(update_medians[0], EPS)
        passed = (
            all(deltas[k] <= tolerances[k] for k in tolerances)
            and grad_ratio <= float(audit["raw_total_gradient_median_ratio_max"])
            and update_ratio <= float(audit["adam_update_median_ratio_max"])
        )
        return {
            "class": "formal_terminal_plateau"
            if passed
            else "formal_persistent_drift_or_inconclusive",
            "window_bounds": bounds,
            "metric_window_means": means,
            "metric_window_mean_abs_deltas": deltas,
            "raw_total_gradient_medians": grad_medians,
            "raw_total_gradient_median_ratio": grad_ratio,
            "adam_update_medians": update_medians,
            "adam_update_median_ratio": update_ratio,
            "full_horizon_reached": full_horizon,
            "formal_two_x_extension_performed": full_horizon,
            "formal_acceptance": full_horizon,
        }
    width = int(audit["trailing_evaluations_per_window"])
    if len(trajectory) < 2 * width:
        return {
            "class": "pilot_inconclusive",
            "reason": "insufficient_two_window_history",
            "formal_acceptance": False,
        }
    windows = [trajectory[-2 * width : -width], trajectory[-width:]]
    metrics = (
        "test_gap_expected_reward",
        "test_gap_correct_group_probability",
        "test_gap_trap_group_probability",
        "test_entropy_mean",
    )
    checks = []
    for window in windows:
        changes = {}
        for metric in metrics:
            values = [float(row[metric]) for row in window]
            scale = max(abs(float(np.mean(values))), 1.0e-8)
            changes[metric] = abs(values[-1] - values[0]) / scale
        grad_median = float(
            np.median([float(row["audit_raw_total_gradient_norm"]) for row in window])
        )
        checks.append(
            {
                "normalized_changes": changes,
                "raw_total_gradient_median": grad_median,
                "passed": all(
                    value <= float(audit["normalized_metric_change_tolerance"])
                    for value in changes.values()
                )
                and grad_median <= float(audit["raw_total_gradient_median_tolerance"]),
            }
        )
    return {
        "class": "pilot_provisional_plateau"
        if all(x["passed"] for x in checks)
        else "pilot_persistent_drift_or_inconclusive",
        "two_window_checks": checks,
        "formal_two_x_extension_performed": False,
        "formal_acceptance": False,
    }


def write_trajectory_rows(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("a") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def run_one(
    config: Mapping[str, Any],
    seed: int,
    spec: RunSpec,
    output_root: Path,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    environment = ConditionalGapEnvironment(config, seed, spec.coverage_mode)
    seed_all(seed + 10_000)
    model = SemanticPolicy(config, str(nested(config, "policy", "concentration_mode"))).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(nested(config, "optimization", "learning_rate")),
        betas=tuple(float(x) for x in nested(config, "optimization", "betas")),
        eps=float(nested(config, "optimization", "eps")),
    )
    maximum_steps = int(nested(config, "optimization", "maximum_steps"))
    eval_interval = int(nested(config, "optimization", "evaluation_interval_steps"))
    batches = shared_batches(
        seed,
        int(nested(config, "data", "train_states")),
        int(nested(config, "optimization", "batch_size")),
        maximum_steps,
    )
    trajectory: list[dict[str, Any]] = []
    numerical_failure = False

    def add_evaluation(step: int, update_norm: float) -> None:
        nonlocal numerical_failure
        train_metrics = evaluate(model, environment, "train", device)
        test_metrics = evaluate(model, environment, "test", device)
        grad_metrics = audit_gradients(model, environment, spec, config, device)
        values = [*train_metrics.values(), *test_metrics.values(), *grad_metrics.values()]
        numerical_failure = numerical_failure or not all(math.isfinite(float(x)) for x in values)
        row: dict[str, Any] = {
            "experiment_id": experiment_id(config),
            "seed": seed,
            "run_key": spec.key,
            "coverage_mode": spec.coverage_mode,
            "method": spec.method,
            "local_alpha": spec.local_alpha,
            "far_lambda": spec.far_lambda,
            "step": step,
            "adam_parameter_update_norm": update_norm,
            "nan_inf_numerical_failure": numerical_failure,
        }
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"test_{k}": v for k, v in test_metrics.items()})
        row.update({f"audit_{k}": v for k, v in grad_metrics.items()})
        row["support_or_temperature_boundary"] = bool(
            test_metrics["effective_support_p05"]
            <= float(nested(config, "events", "effective_support_boundary"))
            or test_metrics["concentration_max"]
            >= float(nested(config, "events", "concentration_warning"))
        )
        trajectory.append(row)

    add_evaluation(0, 0.0)
    embeddings = environment.policy_embeddings.to(device)
    split = environment.train
    for step in range(1, maximum_steps + 1):
        idx = batches[step - 1]
        total = training_gradient(
            model,
            split["states"][idx].to(device),
            split["positive"][idx].to(device),
            split["local"][idx].to(device),
            split["far"][idx].to(device),
            embeddings,
            spec,
            float(nested(config, "optimization", "far_cap_ratio_to_weighted_local_gradient")),
        )
        optimizer.zero_grad(set_to_none=True)
        assign_grads(model, total)
        if not all(g is None or bool(torch.isfinite(g).all()) for g in total):
            numerical_failure = True
            add_evaluation(step, float("nan"))
            break
        before = parameter_vector(model)
        optimizer.step()
        after = parameter_vector(model)
        update_norm = float(torch.linalg.vector_norm(after - before))
        if not math.isfinite(update_norm) or not all(
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
        "experiment_id": experiment_id(config),
        "scientific_status": "not_run" if is_formal(config) else "pilot",
        "seed": seed,
        "run_key": spec.key,
        "coverage_mode": spec.coverage_mode,
        "method": spec.method,
        "local_alpha": spec.local_alpha,
        "far_lambda": spec.far_lambda,
        "completed_steps": int(final["step"]),
        "terminal_class": terminal["class"],
        "terminal_audit": terminal,
        "task_performance_collapse": None,
        "support_or_temperature_boundary": bool(
            any(bool(row["support_or_temperature_boundary"]) for row in trajectory)
        ),
        "nan_inf_numerical_failure": bool(
            any(bool(row["nan_inf_numerical_failure"]) for row in trajectory)
        ),
        "final": {
            key.removeprefix("test_"): value
            for key, value in final.items()
            if key.startswith("test_")
        },
    }
    per_run = output_root / "runs" / f"seed_{seed:03d}"
    per_run.mkdir(parents=True, exist_ok=True)
    json_dump(per_run / f"{spec.key}.summary.json", summary)
    return summary, trajectory


def run_one_worker(
    config: Mapping[str, Any], seed: int, spec: RunSpec, output_root: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    torch.set_num_threads(1)
    return run_one(config, seed, spec, output_root, torch.device("cpu"))


def apply_task_labels(summaries: list[dict[str, Any]], config: Mapping[str, Any]) -> None:
    references: dict[tuple[int, str], float] = {}
    for row in summaries:
        if row["method"] == "positive_only":
            references[(int(row["seed"]), str(row["coverage_mode"]))] = float(
                row["final"]["gap_expected_reward"]
            )
    threshold = float(nested(config, "events", "task_collapse_normalized_margin_to_random"))
    below_random = bool(nested(config, "events", "below_random_is_collapse"))
    for row in summaries:
        key = (int(row["seed"]), str(row["coverage_mode"]))
        reference = references.get(key)
        random_reward = float(row["final"]["gap_random_policy_reward"])
        reward = float(row["final"]["gap_expected_reward"])
        if reference is None or reference <= random_reward + 1.0e-10:
            row["task_performance_collapse"] = None
            row["task_collapse_reference_missing"] = True
            continue
        normalized = (reward - random_reward) / (reference - random_reward)
        row["paired_positive_only_gap_reward"] = reference
        row["gap_reward_margin_over_random_normalized"] = normalized
        row["task_performance_collapse"] = bool(
            normalized <= threshold or (below_random and reward <= random_reward)
        )
        row["task_collapse_reference_missing"] = False


def aggregate_summaries(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in summaries:
        key = (
            f"{row['coverage_mode']}|{row['method']}|"
            f"alpha={row['local_alpha']}|far={row['far_lambda']}"
        )
        groups.setdefault(key, []).append(row)
    metrics = (
        "expected_reward",
        "covered_expected_reward",
        "gap_expected_reward",
        "gap_correct_group_probability",
        "gap_proxy_group_probability",
        "gap_trap_group_probability",
        "gap_top1_group_accuracy",
        "paired_group_distribution_l1",
        "entropy_mean",
        "effective_support_p05",
    )
    output: dict[str, Any] = {}
    for key, rows in sorted(groups.items()):
        values: dict[str, Any] = {}
        for metric in metrics:
            array = np.asarray([float(row["final"][metric]) for row in rows])
            values[metric] = {
                "mean": float(array.mean()),
                "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
                "min": float(array.min()),
                "max": float(array.max()),
            }
        output[key] = {
            "n": len(rows),
            "metrics": values,
            "task_performance_collapse_count": sum(
                row.get("task_performance_collapse") is True for row in rows
            ),
            "support_or_temperature_boundary_count": sum(
                bool(row["support_or_temperature_boundary"]) for row in rows
            ),
            "nan_inf_numerical_failure_count": sum(
                bool(row["nan_inf_numerical_failure"]) for row in rows
            ),
            "terminal_class_counts": {
                name: sum(str(row["terminal_class"]) == name for row in rows)
                for name in sorted({str(row["terminal_class"]) for row in rows})
            },
        }
    return output


def write_summary_csv(path: Path, summaries: Sequence[Mapping[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        row = {
            key: value for key, value in summary.items() if key not in {"final", "terminal_audit"}
        }
        row.update({f"final_{k}": v for k, v in summary["final"].items()})
        rows.append(row)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def source_manifest() -> dict[str, Any]:
    return {
        "git_commit": git_text("rev-parse", "HEAD"),
        "git_branch": git_text("rev-parse", "--abbrev-ref", "HEAD"),
        "git_status_porcelain": git_text("status", "--porcelain"),
    }


def write_checkpoint(
    root: Path,
    block_index: int,
    block_seeds: Sequence[int],
    completed_seeds: Sequence[int],
    summaries: Sequence[Mapping[str, Any]],
    expected_runs: int,
) -> None:
    checkpoint = (
        root
        / "checkpoints"
        / (f"block_{block_index + 1:02d}_seeds_{min(block_seeds)}_{max(block_seeds)}")
    )
    checkpoint.mkdir(parents=True, exist_ok=True)
    json_dump(
        checkpoint / "CHECKPOINT_COMPLETE.json",
        {
            "experiment_id": FORMAL_EXPERIMENT_ID,
            "block_index": block_index + 1,
            "block_seeds": list(block_seeds),
            "completed_seeds": list(completed_seeds),
            "completed_runs": len(summaries),
            "expected_total_runs": expected_runs,
        },
    )


def formal_protocol_freeze(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "experiment_id": FORMAL_EXPERIMENT_ID,
        "claim": config["claim"],
        "data": copy.deepcopy(config["data"]),
        "geometry": copy.deepcopy(config["geometry"]),
        "policy": copy.deepcopy(config["policy"]),
        "optimization": copy.deepcopy(config["optimization"]),
        "run_matrix": copy.deepcopy(config["run_matrix"]),
        "events": copy.deepcopy(config["events"]),
        "terminal_audit": copy.deepcopy(config["terminal_audit"]),
        "checkpointing": copy.deepcopy(config["checkpointing"]),
        "formal_gate": copy.deepcopy(config["formal_gate"]),
        "automatic_retuning_allowed": False,
        "development_seeds_forbidden": [0, 1],
    }


def execute(config: dict[str, Any], stage: str, output_root: Path, device: torch.device) -> None:
    started = time.time()
    formal = is_formal(config)
    manifest_path = prepare_output_root(output_root, formal)
    yaml_dump(output_root / "resolved_config.yaml", config)
    json_dump(
        manifest_path,
        {
            "experiment_id": experiment_id(config),
            "registered_scientific_status": config.get("scientific_status"),
            "execution_mode": config.get("execution_mode"),
            "requested_stage": stage,
            "run_class": "formal" if formal else "pilot",
            "formal_result": formal,
            "started_unix": started,
            "device": str(device),
            "source": source_manifest(),
        },
    )
    seeds = (
        [int(x) for x in nested(config, "seeds", "held_out_formal")]
        if formal
        else [int(x) for x in nested(config, "seeds", "development")]
    )
    specs = run_specs(config)
    coverage_modes = sorted({spec.coverage_mode for spec in specs})
    audits = [
        ConditionalGapEnvironment(config, seed, mode).audit()
        for seed in seeds
        for mode in coverage_modes
    ]
    json_dump(output_root / "environment_audits.json", audits)
    invariants_passed = all(bool(row["passed"]) for row in audits)
    if not invariants_passed:
        json_dump(output_root / "RUN_FAILED.json", {"reason": "environment_invariant_failure"})
        raise RuntimeError("conditional-gap environment invariant audit failed")
    if stage == "invariants":
        if formal:
            raise RuntimeError("formal config cannot run invariants-only")
        json_dump(output_root / "per_run_summary.json", [])
        json_dump(output_root / "aggregate_summary.json", {})
        json_dump(
            output_root / "terminal_audit.json",
            {
                "experiment_id": experiment_id(config),
                "all_environment_invariants_passed": True,
                "scientific_result": False,
                "formal_acceptance": False,
            },
        )
        json_dump(
            output_root / "RUN_COMPLETE.json",
            {
                "experiment_id": experiment_id(config),
                "completed": True,
                "formal_result": False,
                "scientific_status": "pilot",
                "stage": "invariants",
            },
        )
        return

    workers = int(nested(config, "optimization", "parallel_workers")) if device.type == "cpu" else 1
    seed_blocks = (
        [list(x) for x in nested(config, "checkpointing", "seed_blocks")] if formal else [seeds]
    )
    summaries: list[dict[str, Any]] = []
    trajectory_path = output_root / "trajectories.jsonl"
    expected_runs = len(seeds) * len(specs)
    completed_seeds: list[int] = []
    for block_index, block in enumerate(seed_blocks):
        tasks = [(config, seed, spec, output_root) for seed in block for spec in specs]
        block_results: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
        if workers > 1:
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(run_one_worker, *task) for task in tasks]
                for future in futures:
                    block_results.append(future.result())
        else:
            for config_i, seed, spec, root in tasks:
                block_results.append(run_one(config_i, seed, spec, root, device))
        for summary, trajectory in block_results:
            summaries.append(summary)
            write_trajectory_rows(trajectory_path, trajectory)
        completed_seeds.extend(block)
        apply_task_labels(summaries, config)
        json_dump(output_root / "per_run_summary.json", summaries)
        write_summary_csv(output_root / "per_run_summary.csv", summaries)
        json_dump(output_root / "aggregate_summary.json", aggregate_summaries(summaries))
        if formal:
            write_checkpoint(
                output_root,
                block_index,
                block,
                completed_seeds,
                summaries,
                expected_runs,
            )

    apply_task_labels(summaries, config)
    json_dump(output_root / "per_run_summary.json", summaries)
    write_summary_csv(output_root / "per_run_summary.csv", summaries)
    json_dump(output_root / "aggregate_summary.json", aggregate_summaries(summaries))
    if formal:
        json_dump(output_root / "formal_protocol_freeze.json", formal_protocol_freeze(config))
    else:
        json_dump(
            output_root / "pilot_freeze_recommendation.json",
            {
                "automatic_freeze_allowed": False,
                "reason": "development evidence requires explicit user-approved formal freeze",
                "candidate_run_matrix": copy.deepcopy(config["run_matrix"]),
            },
        )

    all_runs = len(summaries) == expected_runs
    missing_refs = sum(bool(row.get("task_collapse_reference_missing")) for row in summaries)
    numerical = sum(bool(row["nan_inf_numerical_failure"]) for row in summaries)
    all_terminal = all(bool(row.get("terminal_audit")) for row in summaries)
    formal_two_x = bool(
        formal
        and all_terminal
        and all(
            bool(row["terminal_audit"].get("formal_two_x_extension_performed"))
            or row["terminal_class"] == "numerical_failure"
            for row in summaries
        )
    )
    formal_acceptance = bool(
        formal
        and all_runs
        and invariants_passed
        and missing_refs == 0
        and formal_two_x
        and all(bool(row["terminal_audit"].get("formal_acceptance")) for row in summaries)
    )
    terminal = {
        "experiment_id": experiment_id(config),
        "scientific_status": "not_run" if formal else "pilot",
        "run_class": "formal" if formal else "pilot",
        "expected_runs": expected_runs,
        "actual_runs": len(summaries),
        "all_runs_present": all_runs,
        "environment_invariants_passed": invariants_passed,
        "missing_task_collapse_reference_count": missing_refs,
        "task_performance_collapse_count": sum(
            row.get("task_performance_collapse") is True for row in summaries
        ),
        "support_or_temperature_boundary_count": sum(
            bool(row["support_or_temperature_boundary"]) for row in summaries
        ),
        "nan_inf_numerical_failure_count": numerical,
        "formal_two_x_extension_performed": formal_two_x,
        "formal_scientific_acceptance": formal_acceptance,
        "scientific_failure_outcomes_preserved": True,
        "pilot_integrity_passed": bool((not formal) and all_runs and numerical == 0),
    }
    json_dump(output_root / "terminal_audit.json", terminal)
    completed = formal_acceptance if formal else bool(all_runs and numerical == 0)
    json_dump(
        output_root / "RUN_COMPLETE.json",
        {
            "experiment_id": experiment_id(config),
            "scientific_status": "not_run" if formal else "pilot",
            "execution_mode": config.get("execution_mode"),
            "completed": completed,
            "formal_result": formal,
            "expected_runs": expected_runs,
            "actual_runs": len(summaries),
            "elapsed_seconds": time.time() - started,
        },
    )
    if not completed:
        raise RuntimeError("conditional-gap E6 did not complete its registered audit")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=["invariants", "smoke", "pilot", "formal"], required=True
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def resolve_device(name: str) -> torch.device:
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
    device = resolve_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(1)
    stage = "pilot" if args.stage == "smoke" else args.stage
    execute(config, stage, args.output_root, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
