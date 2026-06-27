#!/usr/bin/env python3
"""D-U1 E6 shared-semantic categorical mechanism implementation.

This module owns the shared environment, training loop, diagnostics, and terminal
audits used by the development pilot, focused blocker-resolution runs, and the
separately frozen formal long-run. Formal execution is accepted only for the exact
registered config and untouched held-out seeds 10--29.

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
* Smoke tests and development runs are not formal results or method rankings.
* Formal execution must use the separate long-run entrypoint and hardened guard.
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


PILOT_EXPERIMENT_ID = "D-U1-E6-SEMANTIC-PILOT-01"
FOCUSED_EXPERIMENT_ID = "D-U1-E6-SEMANTIC-FOCUSED-DEV-01"
EXPERIMENT_ID = PILOT_EXPERIMENT_ID  # Backward-compatible public constant.
FORMAL_EXPERIMENT_ID = "D-U1-E6-SEMANTIC-LONGRUN-01"
SEMANTIC_GAP_FORMAL_EXPERIMENT_ID = "D-U1-E6-SEMANTIC-GAP-LONGRUN-01"
ALLOWED_DEVELOPMENT_EXPERIMENT_IDS = {PILOT_EXPERIMENT_ID, FOCUSED_EXPERIMENT_ID}
ALLOWED_EXPERIMENT_IDS = {
    *ALLOWED_DEVELOPMENT_EXPERIMENT_IDS,
    FORMAL_EXPERIMENT_ID,
    SEMANTIC_GAP_FORMAL_EXPERIMENT_ID,
}
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


def prepare_output_manifest_path(
    output_root: Path, *, formal: bool, experiment_id: str = FORMAL_EXPERIMENT_ID
) -> Path:
    """Prepare an output root without clobbering the hardened guard manifest.

    The canonical guard creates ``run_manifest.json`` and ``logs/`` before the
    scientific child starts. Development runs still require a new or empty root,
    while formal runs fail closed unless that guard-owned manifest is present and
    identifies the active E6 formal experiment.
    """
    if not formal:
        ensure_new_or_empty(output_root)
        return output_root / "run_manifest.json"

    if not output_root.is_dir():
        raise RuntimeError("formal E6 must run inside the canonical hardened guard output root")
    guard_manifest_path = output_root / "run_manifest.json"
    if not guard_manifest_path.is_file():
        raise RuntimeError(
            "formal E6 requires the guard-owned run_manifest.json; direct formal "
            "execution is forbidden"
        )
    try:
        guard_manifest = json.loads(guard_manifest_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeError("cannot read the hardened guard manifest") from exc
    required_identity = {
        "experiment_id": experiment_id,
        "run_class": "formal",
        "execution_state": "running",
    }
    for key, expected in required_identity.items():
        if guard_manifest.get(key) != expected:
            raise RuntimeError(
                f"guard manifest {key} must be {expected!r}, got {guard_manifest.get(key)!r}"
            )

    science_owned = [
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
    ]
    stale = [name for name in science_owned if (output_root / name).exists()]
    if stale:
        raise RuntimeError(
            "formal E6 output root contains stale scientific files: " + ", ".join(stale)
        )
    return output_root / "scientific_run_manifest.json"


def nested(config: Mapping[str, Any], *keys: str) -> Any:
    value: Any = config
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            raise KeyError(".".join(keys))
        value = value[key]
    return value


def configured_experiment_id(config: Mapping[str, Any]) -> str:
    value = str(config.get("experiment_id", ""))
    if value not in ALLOWED_EXPERIMENT_IDS:
        allowed = ", ".join(sorted(ALLOWED_EXPERIMENT_IDS))
        raise ValueError(f"experiment_id must be one of: {allowed}")
    return value


def is_formal_config(config: Mapping[str, Any]) -> bool:
    return configured_experiment_id(config) in {
        FORMAL_EXPERIMENT_ID,
        SEMANTIC_GAP_FORMAL_EXPERIMENT_ID,
    }


def run_seeds(config: Mapping[str, Any]) -> list[int]:
    key = "held_out_formal" if is_formal_config(config) else "development"
    return [int(seed) for seed in nested(config, "seeds", key)]


def result_scientific_status(config: Mapping[str, Any]) -> str:
    experiment_id = configured_experiment_id(config)
    if experiment_id == FORMAL_EXPERIMENT_ID:
        return "long_run_validated"
    if experiment_id == SEMANTIC_GAP_FORMAL_EXPERIMENT_ID:
        return "finite_step_validated"
    return "pilot"


def _require_exact(value: Any, expected: Any, label: str) -> None:
    if value != expected:
        raise ValueError(f"{label} is frozen to {expected!r}, got {value!r}")


def validate_formal_config(config: Mapping[str, Any], stage: str) -> None:
    """Fail closed unless every registered E6 formal field matches the freeze."""
    if stage != "formal":
        raise ValueError("formal E6 config may only run with --stage formal")

    _require_exact(config.get("scientific_status"), "not_run", "scientific_status")
    _require_exact(config.get("execution_mode"), "formal_longrun", "execution_mode")
    _require_exact(bool(config.get("formal_parameter_freeze")), True, "formal_parameter_freeze")
    _require_exact(
        config.get("predecessor"),
        FOCUSED_EXPERIMENT_ID,
        "predecessor",
    )
    _require_exact(
        dict(config.get("approval", {})),
        {
            "user_approved": True,
            "approval_date": "2026-06-27",
            "approval_scope": "exact_focused_development_freeze_recommendation",
        },
        "approval",
    )

    held_out = list(range(10, 30))
    _require_exact(list(nested(config, "seeds", "development")), [], "seeds.development")
    _require_exact(
        list(nested(config, "seeds", "held_out_formal")),
        held_out,
        "seeds.held_out_formal",
    )

    expected_data = {
        "state_dim": 6,
        "semantic_dim": 4,
        "action_count": 64,
        "train_states": 2048,
        "test_states": 2048,
        "positive_actions_per_state": 4,
        "far_negative_actions_per_state": 4,
        "hidden_optimal_actions_per_state": 1,
        "local_negative_actions_per_state": 1,
        "state_distribution": "standard_normal",
        "train_test_relation": "independent_same_distribution",
        "terminology": "held_out_context_generalization",
    }
    _require_exact(dict(nested(config, "data")), expected_data, "data")

    expected_geometry = {
        "target_offset": 0.45,
        "reward_scale": 0.5,
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
    _require_exact(
        dict(nested(config, "optimization")),
        expected_optimization,
        "optimization",
    )

    expected_protocol_a = {
        "responsibility": "fixed_concentration_positive_only_ceiling_and_alpha_transition",
        "concentration_mode": "fixed",
        "local_alpha_grid": [0.0, 0.25, 0.5, 0.75],
        "far_pressure_lambda": 0.0,
    }
    _require_exact(dict(nested(config, "protocol_a")), expected_protocol_a, "protocol_a")

    expected_protocol_b = {
        "responsibility": "learnable_concentration_near_far_causal_and_control_matrix",
        "concentration_mode": "learnable",
        "settings": [
            {
                "local_alpha": 0.0,
                "far_pressure_lambda": 0.0,
                "methods": ["positive_only"],
            },
            {
                "local_alpha": 0.1,
                "far_pressure_lambda": 0.05,
                "methods": [
                    "far_zero",
                    "uncontrolled",
                    "near_zero",
                    "far_cap",
                    "budget_matched_global",
                ],
            },
        ],
    }
    _require_exact(dict(nested(config, "protocol_b")), expected_protocol_b, "protocol_b")

    expected_protocol_c = {
        "responsibility": "policy_side_semantic_alignment_exclusion_control",
        "concentration_mode": "learnable",
        "local_alpha": 0.1,
        "far_pressure_lambda": 0.05,
        "embedding_modes": ["aligned", "shuffled"],
        "methods": ["positive_only", "far_zero", "uncontrolled", "far_cap"],
    }
    _require_exact(dict(nested(config, "protocol_c")), expected_protocol_c, "protocol_c")

    _require_exact(
        list(config.get("reporting_separation", [])),
        [
            "task_performance_collapse",
            "support_or_temperature_boundary",
            "nan_inf_numerical_failure",
        ],
        "reporting_separation",
    )
    _require_exact(
        dict(nested(config, "events")),
        {
            "task_collapse_ratio_to_paired_positive_only": 0.2,
            "effective_support_boundary": 1.5,
            "concentration_warning": 80.0,
        },
        "events",
    )

    expected_terminal_audit = {
        "mode": "formal_two_x_windows",
        "development_reference_horizon_steps": 4000,
        "formal_horizon_steps": 8000,
        "formal_extension_factor": 2.0,
        "window_1_steps": [4000, 6000],
        "window_2_steps": [6000, 8000],
        "metric_window_mean_abs_tolerances": {
            "test_expected_semantic_reward": 0.01,
            "test_hidden_optimal_probability": 0.02,
            "test_normalized_semantic_extrapolation": 0.08,
            "test_entropy_mean": 0.08,
        },
        "raw_total_gradient_median_ratio_max": 1.25,
        "adam_update_median_ratio_max": 1.25,
        "require_all_registered_runs": True,
        "allow_scientific_failure_outcomes": True,
    }
    _require_exact(
        dict(nested(config, "terminal_audit")),
        expected_terminal_audit,
        "terminal_audit",
    )

    expected_checkpointing = {
        "seed_block_size": 5,
        "seed_blocks": [
            [10, 11, 12, 13, 14],
            [15, 16, 17, 18, 19],
            [20, 21, 22, 23, 24],
            [25, 26, 27, 28, 29],
        ],
        "persistence": "persistent_local",
        "write_compact_manifest_after_each_block": True,
    }
    _require_exact(
        dict(nested(config, "checkpointing")),
        expected_checkpointing,
        "checkpointing",
    )

    expected_gate = {
        "enabled": True,
        "approval_record": "user_approved_2026-06-27_exact_focused_dev_freeze",
        "frozen_protocol_path": "configs/du1_e6_semantic_longrun.yaml",
        "held_out_seeds": held_out,
    }
    _require_exact(dict(nested(config, "formal_gate")), expected_gate, "formal_gate")

    expected_outputs = [
        "resolved_config.yaml",
        "scientific_run_manifest.json",
        "environment_audits.json",
        "trajectories.jsonl",
        "per_run_summary.json",
        "per_run_summary.csv",
        "aggregate_summary.json",
        "terminal_audit.json",
        "formal_protocol_freeze.json",
        "run_manifest.json",
        "RUN_COMPLETE.json",
    ]
    _require_exact(
        list(nested(config, "outputs", "required")),
        expected_outputs,
        "outputs.required",
    )


def validate_config(config: Mapping[str, Any], stage: str) -> None:
    experiment_id = configured_experiment_id(config)
    formal = is_formal_config(config)
    if experiment_id == SEMANTIC_GAP_FORMAL_EXPERIMENT_ID:
        # Import lazily to avoid a module-level cycle: the successor validator
        # deliberately reuses the original E6 implementation.
        from drpo.du1_e6_semantic_gap import validate_formal_config as validate_gap_config

        validate_gap_config(config, stage)
    elif formal:
        validate_formal_config(config, stage)
    else:
        if config.get("scientific_status") != "pilot":
            raise ValueError("E6 development runner must remain status=pilot")
        if bool(config.get("formal_parameter_freeze")):
            raise ValueError("pilot config must not claim a formal parameter freeze")
        if stage == "formal":
            raise RuntimeError(
                f"{FORMAL_EXPERIMENT_ID} is blocked for development configs; use the "
                "separately frozen formal config and formal entrypoint"
            )

    data = nested(config, "data")
    if int(data["state_dim"]) != 6 or int(data["semantic_dim"]) != 4:
        raise ValueError("locked E6 geometry is 6D state / 4D semantics")
    if int(data["positive_actions_per_state"]) != 4:
        raise ValueError("E6 geometry requires four positive actions per state")
    if int(data["far_negative_actions_per_state"]) != 4:
        raise ValueError("E6 geometry requires four far negatives per state")
    if nested(config, "geometry", "negative_advantage") >= 0:
        raise ValueError("negative_advantage must be negative")
    if nested(config, "geometry", "positive_advantage") <= 0:
        raise ValueError("positive_advantage must be positive")
    if not bool(nested(config, "geometry", "fixed_advantage")):
        raise ValueError("E6 is a fixed-advantage mechanism experiment")

    if not formal:
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
    out["optimization"]["parallel_workers"] = 1
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
        coverage = config.get("conditional_coverage", {})
        if coverage is None:
            coverage = {}
        if not isinstance(coverage, Mapping):
            raise ValueError("conditional_coverage must be a mapping when provided")
        self.coverage_mode = str(coverage.get("mode", "dense"))
        self.gap_state_fraction = float(coverage.get("gap_state_fraction", 0.0))
        self.withheld_action_fraction = float(coverage.get("withheld_action_fraction", 0.0))
        self.require_global_action_coverage = bool(
            coverage.get("require_global_action_coverage", False)
        )
        if self.coverage_mode not in {"dense", "structured_semantic_neighbourhood_gap"}:
            raise ValueError(f"unknown conditional coverage mode: {self.coverage_mode}")
        if not 0.0 <= self.gap_state_fraction <= 1.0:
            raise ValueError("gap_state_fraction must be in [0, 1]")
        if not 0.0 <= self.withheld_action_fraction < 1.0:
            raise ValueError("withheld_action_fraction must be in [0, 1)")
        if self.coverage_mode == "dense" and (
            self.gap_state_fraction != 0.0 or self.withheld_action_fraction != 0.0
        ):
            raise ValueError("dense coverage cannot specify a nonzero structured gap")

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

        gap_mask = torch.zeros(count, dtype=torch.bool)
        conditional_gap = torch.zeros(count, self.action_count, dtype=torch.bool)
        target_neighbourhood = torch.empty(count, 0, dtype=torch.long)
        if self.coverage_mode == "structured_semantic_neighbourhood_gap":
            gap_count = int(round(hidden.numel() * self.gap_state_fraction))
            if gap_count > 0:
                state_score = states[:, 0] + 0.37 * states[:, 1]
                gap_rows = state_score.argsort(descending=True)[:gap_count]
                gap_mask[gap_rows] = True
            withheld_count = int(round(self.action_count * self.withheld_action_fraction))
            withheld_count = max(1, withheld_count)
            max_withheld = self.action_count - (self.n_positive + self.n_far + 1)
            withheld_count = min(withheld_count, max_withheld)
            target_neighbourhood = reward_similarity.topk(withheld_count, dim=1).indices
            if bool(gap_mask.any()):
                rows = gap_mask.nonzero(as_tuple=False).squeeze(1)
                conditional_gap[rows[:, None], target_neighbourhood[rows]] = True

        banned = conditional_gap.clone()
        banned.scatter_(1, hidden[:, None], True)
        positive = self._topk_excluding(plus_similarity, banned, self.n_positive)
        banned.scatter_(1, positive, True)
        local = self._topk_excluding(minus_similarity, banned, 1).squeeze(1)
        banned.scatter_(1, local[:, None], True)
        far = self._topk_excluding(far_similarity, banned, self.n_far)

        if self.require_global_action_coverage:
            roles = torch.cat([positive.reshape(-1), local.reshape(-1), far.reshape(-1)])
            counts = torch.bincount(roles, minlength=self.action_count)
            missing = (counts == 0).nonzero(as_tuple=False).squeeze(1).tolist()
            covered_rows = (~gap_mask).nonzero(as_tuple=False).squeeze(1)
            for action in missing:
                admissible = covered_rows[
                    (hidden[covered_rows] != action)
                    & ~(positive[covered_rows] == action).any(dim=1)
                    & (local[covered_rows] != action)
                    & ~(far[covered_rows] == action).any(dim=1)
                ]
                if admissible.numel() == 0:
                    raise RuntimeError(f"cannot repair global coverage for action {action}")
                candidate_scores = far_similarity[admissible, action]
                repaired = False
                for row in admissible[candidate_scores.argsort(descending=True)].tolist():
                    replaceable = [
                        slot
                        for slot in range(self.n_far)
                        if int(counts[int(far[row, slot])]) > 1
                    ]
                    if not replaceable:
                        continue
                    slot = min(
                        replaceable,
                        key=lambda index: float(far_similarity[row, far[row, index]]),
                    )
                    old_action = int(far[row, slot])
                    far[row, slot] = int(action)
                    counts[old_action] -= 1
                    counts[action] += 1
                    repaired = True
                    break
                if not repaired:
                    raise RuntimeError(f"cannot find replaceable far slot for action {action}")

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
            "gap_mask": gap_mask,
            "conditional_gap_mask": conditional_gap,
            "target_neighbourhood": target_neighbourhood,
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
            gap_mask = split["gap_mask"]
            conditional_gap = split["conditional_gap_mask"]
            role_gap_violations = 0
            hidden_gap_violations = 0
            if bool(gap_mask.any()):
                gap_rows = gap_mask.nonzero(as_tuple=False).squeeze(1)
                role_gap_violations += int(
                    conditional_gap[gap_rows[:, None], positive[gap_mask]].sum()
                )
                role_gap_violations += int(
                    conditional_gap[gap_rows, local[gap_mask]].sum()
                )
                role_gap_violations += int(
                    conditional_gap[gap_rows[:, None], far[gap_mask]].sum()
                )
                hidden_gap_violations = int(
                    (~conditional_gap[gap_rows, hidden[gap_mask]]).sum()
                )
            logged_roles = torch.cat(
                [positive.reshape(-1), local.reshape(-1), far.reshape(-1)]
            )
            global_action_counts = torch.bincount(
                logged_roles, minlength=self.action_count
            )
            globally_unobserved_actions = int((global_action_counts == 0).sum())
            expected_gap_states = (
                int(round(hidden.numel() * self.gap_state_fraction))
                if self.coverage_mode == "structured_semantic_neighbourhood_gap"
                else 0
            )
            expected_withheld = (
                int(round(self.action_count * self.withheld_action_fraction))
                if self.coverage_mode == "structured_semantic_neighbourhood_gap"
                else 0
            )
            actual_withheld_values = conditional_gap[gap_mask].sum(1)
            actual_withheld_min = (
                int(actual_withheld_values.min()) if actual_withheld_values.numel() else 0
            )
            actual_withheld_max = (
                int(actual_withheld_values.max()) if actual_withheld_values.numel() else 0
            )
            coverage_passed = all(
                [
                    int(gap_mask.sum()) == expected_gap_states,
                    actual_withheld_min == expected_withheld,
                    actual_withheld_max == expected_withheld,
                    role_gap_violations == 0,
                    hidden_gap_violations == 0,
                    (not self.require_global_action_coverage)
                    or globally_unobserved_actions == 0,
                ]
            )
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
                    coverage_passed,
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
                "conditional_coverage": {
                    "mode": self.coverage_mode,
                    "gap_state_count": int(gap_mask.sum()),
                    "expected_gap_state_count": expected_gap_states,
                    "withheld_action_count_min": actual_withheld_min,
                    "withheld_action_count_max": actual_withheld_max,
                    "expected_withheld_action_count": expected_withheld,
                    "logged_role_gap_violations": role_gap_violations,
                    "hidden_not_withheld_violations": hidden_gap_violations,
                    "globally_unobserved_action_count": globally_unobserved_actions,
                    "passed": coverage_passed,
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
    settings = protocol_b.get("settings")
    if settings is None:
        settings = [
            {
                "local_alpha": protocol_b["local_alpha"],
                "far_pressure_lambda": protocol_b["far_pressure_lambda"],
                "methods": protocol_b["methods"],
            }
        ]
    if not isinstance(settings, Sequence):
        raise ValueError("protocol_b.settings must be a sequence")
    for setting in settings:
        if not isinstance(setting, Mapping):
            raise ValueError("each protocol_b setting must be a mapping")
        for method in setting["methods"]:
            specs.append(
                RunSpec(
                    protocol="E6-B",
                    method=str(method),
                    alpha=float(setting["local_alpha"]),
                    far_lambda=float(setting["far_pressure_lambda"]),
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
    formal_mode = is_formal_config(config)
    final_step = int(trajectory[-1].get("step", -1))
    maximum_steps = int(nested(config, "optimization", "maximum_steps"))
    full_horizon_reached = final_step >= maximum_steps
    if any(bool(row.get("nan_inf_numerical_failure")) for row in trajectory):
        return {
            "class": "numerical_failure",
            "reason": "nonfinite_value_observed",
            "completed_steps": final_step,
            "formal_two_x_extension_performed": False,
            "formal_acceptance": formal_mode,
        }
    if any(bool(row.get("support_or_temperature_boundary")) for row in trajectory):
        return {
            "class": "support_or_temperature_boundary",
            "reason": "effective_support_boundary_reached",
            "completed_steps": final_step,
            "full_horizon_reached": full_horizon_reached,
            "formal_two_x_extension_performed": formal_mode and full_horizon_reached,
            "formal_acceptance": formal_mode and full_horizon_reached,
        }

    audit = nested(config, "terminal_audit")
    mode = audit.get("mode")
    if mode in {"focused_two_x_windows", "formal_two_x_windows"}:
        formal_mode = mode == "formal_two_x_windows"
        first_bounds = [int(x) for x in audit["window_1_steps"]]
        second_bounds = [int(x) for x in audit["window_2_steps"]]
        if len(first_bounds) != 2 or len(second_bounds) != 2:
            raise ValueError("registered terminal windows must each have two endpoints")
        windows = [
            [row for row in trajectory if first_bounds[0] <= int(row["step"]) <= first_bounds[1]],
            [row for row in trajectory if second_bounds[0] < int(row["step"]) <= second_bounds[1]],
        ]
        if any(not window for window in windows):
            prefix = "formal" if formal_mode else "focused"
            return {
                "class": f"{prefix}_terminal_inconclusive",
                "reason": "missing_registered_terminal_window",
                "window_sizes": [len(window) for window in windows],
                "formal_two_x_extension_performed": formal_mode,
                "formal_acceptance": False,
            }
        tolerances = {
            str(key): float(value)
            for key, value in audit["metric_window_mean_abs_tolerances"].items()
        }
        means = [
            {
                metric: float(np.mean([float(row[metric]) for row in window]))
                for metric in tolerances
            }
            for window in windows
        ]
        deltas = {metric: abs(means[1][metric] - means[0][metric]) for metric in tolerances}
        gradient_medians = [
            float(np.median([float(row["audit_raw_total_gradient_norm"]) for row in window]))
            for window in windows
        ]
        update_medians = [
            float(np.median([float(row["adam_parameter_update_norm"]) for row in window]))
            for window in windows
        ]
        gradient_ratio = gradient_medians[1] / max(gradient_medians[0], EPS)
        update_ratio = update_medians[1] / max(update_medians[0], EPS)
        metric_pass = all(deltas[name] <= tolerances[name] for name in tolerances)
        gradient_pass = gradient_ratio <= float(audit["raw_total_gradient_median_ratio_max"])
        update_pass = update_ratio <= float(audit["adam_update_median_ratio_max"])
        passed = metric_pass and gradient_pass and update_pass
        if formal_mode:
            classification = (
                "formal_terminal_plateau" if passed else "formal_persistent_drift_or_inconclusive"
            )
            note = (
                "Formal terminal classification is scientific evidence; task, "
                "support, and numerical events remain separately reported."
            )
        else:
            classification = (
                "focused_terminal_plateau" if passed else "focused_terminal_drift_or_inconclusive"
            )
            note = "Focused development terminal evidence cannot establish formal ranking."
        return {
            "class": classification,
            "window_bounds": [first_bounds, second_bounds],
            "window_sizes": [len(window) for window in windows],
            "metric_window_means": means,
            "metric_window_mean_abs_deltas": deltas,
            "metric_tolerances": tolerances,
            "raw_total_gradient_medians": gradient_medians,
            "raw_total_gradient_median_ratio": gradient_ratio,
            "adam_update_medians": update_medians,
            "adam_update_median_ratio": update_ratio,
            "metric_pass": metric_pass,
            "gradient_pass": gradient_pass,
            "update_pass": update_pass,
            "development_two_x_horizon_performed": not formal_mode,
            "formal_two_x_extension_performed": formal_mode,
            "formal_acceptance": formal_mode,
            "note": note,
        }

    width = int(audit["trailing_evaluations_per_window"])
    if len(trajectory) < 2 * width:
        return {
            "class": "inconclusive",
            "reason": "insufficient_two_window_history",
            "required_evaluations": 2 * width,
            "actual_evaluations": len(trajectory),
        }
    tolerance = float(audit["normalized_metric_change_tolerance"])
    grad_tolerance = float(audit["raw_total_gradient_median_tolerance"])
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
            "experiment_id": configured_experiment_id(config),
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
        "experiment_id": configured_experiment_id(config),
        "scientific_status": result_scientific_status(config),
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


def run_one_reconstructed(
    config: Mapping[str, Any],
    seed: int,
    spec: RunSpec,
    output_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run one deterministic CPU branch in a worker process.

    Reconstructing the environment, initialization, and minibatch stream from the
    registered seed preserves paired scientific inputs while avoiding transfer of
    large mutable objects between processes.
    """
    torch.set_num_threads(1)
    device = torch.device("cpu")
    environment = SemanticEnvironment(config, seed, spec.embedding_mode)
    seed_all(seed + 10_000)
    initial_model = SemanticPolicy(config, spec.concentration_mode)
    initial_state = {
        key: value.detach().clone() for key, value in initial_model.state_dict().items()
    }
    batches = shared_batch_stream(
        seed,
        int(nested(config, "data", "train_states")),
        int(nested(config, "optimization", "batch_size")),
        int(nested(config, "optimization", "maximum_steps")),
    )
    return run_one(
        config,
        seed,
        spec,
        environment,
        initial_state,
        batches,
        output_root,
        device,
    )


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
        "experiment_id": configured_experiment_id(config),
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


def write_formal_checkpoint(
    output_root: Path,
    *,
    experiment_id: str,
    block_index: int,
    block_seeds: Sequence[int],
    completed_seeds: Sequence[int],
    summaries: Sequence[Mapping[str, Any]],
    expected_total_runs: int,
) -> None:
    """Write a compact persistent-local checkpoint after each five-seed block."""
    checkpoint_root = (
        output_root
        / "checkpoints"
        / (f"block_{block_index + 1:02d}_seeds_{block_seeds[0]}_{block_seeds[-1]}")
    )
    checkpoint_root.mkdir(parents=True, exist_ok=False)
    tracked = [
        output_root / "trajectories.jsonl",
        output_root / "per_run_summary.json",
        output_root / "per_run_summary.csv",
        output_root / "aggregate_summary.json",
    ]
    files = []
    for path in tracked:
        if not path.is_file():
            raise RuntimeError(f"formal checkpoint source is missing: {path}")
        files.append(
            {
                "path": path.relative_to(output_root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    json_dump(
        checkpoint_root / "CHECKPOINT_COMPLETE.json",
        {
            "schema_version": 1,
            "experiment_id": experiment_id,
            "scientific_status": "not_run",
            "checkpoint_only": True,
            "method_ranking_allowed": False,
            "persistence": "persistent_local",
            "block_index": block_index,
            "block_seeds": list(block_seeds),
            "completed_seeds": list(completed_seeds),
            "completed_run_count": len(summaries),
            "expected_total_run_count": expected_total_runs,
            "git_head": git_text("rev-parse", "HEAD"),
            "files": files,
        },
    )


def source_manifest(output_root: Path) -> dict[str, Any]:
    tracked = [
        Path("src/drpo/du1_e6_semantic.py"),
        Path("src/drpo/du1_e6_semantic_longrun.py"),
        Path("scripts/run_du1_e6_semantic_longrun.py"),
        Path("configs/du1_e6_semantic_pilot.yaml"),
        Path("configs/du1_e6_semantic_focused_dev.yaml"),
        Path("configs/du1_e6_semantic_focused_dev_phase2.yaml"),
        Path("configs/du1_e6_semantic_longrun.yaml"),
        Path("src/drpo/du1_e6_semantic_gap.py"),
        Path("src/drpo/du1_e6_semantic_gap_longrun.py"),
        Path("scripts/run_du1_e6_semantic_gap_longrun.py"),
        Path("configs/du1_e6_semantic_gap_longrun.yaml"),
        Path("docs/handoff.md"),
        Path("experiments/registry.yaml"),
    ]
    files: list[dict[str, Any]] = []
    for path in tracked:
        if path.exists():
            files.append(
                {
                    "path": path.as_posix(),
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return {
        "git_head": git_text("rev-parse", "HEAD"),
        "git_status_porcelain": git_text("status", "--porcelain"),
        "source_files": files,
        "output_root": str(output_root),
    }


def write_registered_horizon_summary(
    trajectory_path: Path,
    config: Mapping[str, Any],
    output_root: Path,
) -> None:
    """Aggregate the preregistered overall-reward checkpoints, when present."""
    checkpoints = [int(value) for value in config.get("registered_horizon_checkpoints", [])]
    if not checkpoints:
        return
    rows: list[dict[str, Any]] = []
    with trajectory_path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if int(row.get("step", -1)) in checkpoints:
                rows.append(row)
    expected = len(run_seeds(config)) * len(run_specs(config)) * len(checkpoints)
    if len(rows) != expected:
        raise RuntimeError(
            f"registered horizon rows mismatch: expected {expected}, got {len(rows)}"
        )
    grouped: dict[tuple[int, float], list[dict[str, Any]]] = {}
    by_seed: dict[tuple[int, int, float], float] = {}
    for row in rows:
        step = int(row["step"])
        alpha = float(row["alpha"])
        reward = float(row["test_expected_semantic_reward"])
        grouped.setdefault((step, alpha), []).append(row)
        by_seed[(int(row["seed"]), step, alpha)] = reward
    summary_rows: list[dict[str, Any]] = []
    for step in checkpoints:
        baseline = {
            seed: reward
            for (seed, row_step, alpha), reward in by_seed.items()
            if row_step == step and alpha == 0.0
        }
        for alpha in sorted({key[1] for key in grouped if key[0] == step}):
            values = np.asarray(
                [float(row["test_expected_semantic_reward"]) for row in grouped[(step, alpha)]],
                dtype=float,
            )
            paired = np.asarray(
                [
                    by_seed[(seed, step, alpha)] - baseline[seed]
                    for seed in sorted(baseline)
                ],
                dtype=float,
            )
            summary_rows.append(
                {
                    "step": step,
                    "alpha": alpha,
                    "n": int(values.size),
                    "overall_reward_mean": float(values.mean()),
                    "overall_reward_std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                    "paired_difference_vs_positive_only_mean": float(paired.mean()),
                    "paired_wins_vs_positive_only": int((paired > 0).sum()),
                    "paired_losses_vs_positive_only": int((paired < 0).sum()),
                }
            )
    json_dump(
        output_root / "horizon_summary.json",
        {
            "experiment_id": configured_experiment_id(config),
            "primary_metric": "overall_expected_semantic_reward",
            "registered_steps": checkpoints,
            "rows": summary_rows,
        },
    )
    with (output_root / "horizon_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)


def formal_protocol_freeze(config: Mapping[str, Any]) -> dict[str, Any]:
    if not is_formal_config(config):
        raise ValueError("formal protocol freeze is only defined for the formal config")
    return {
        "experiment_id": configured_experiment_id(config),
        "user_approval": copy.deepcopy(config["approval"]),
        "held_out_seeds": list(nested(config, "seeds", "held_out_formal")),
        "data": copy.deepcopy(config["data"]),
        "conditional_coverage": copy.deepcopy(config.get("conditional_coverage")),
        "geometry": copy.deepcopy(config["geometry"]),
        "optimizer": copy.deepcopy(config["optimization"]),
        "policy": copy.deepcopy(config["policy"]),
        "primary_metrics": copy.deepcopy(config.get("primary_metrics")),
        "registered_horizon_checkpoints": copy.deepcopy(
            config.get("registered_horizon_checkpoints")
        ),
        "protocol_a": copy.deepcopy(config["protocol_a"]),
        "protocol_b": copy.deepcopy(config["protocol_b"]),
        "protocol_c": copy.deepcopy(config["protocol_c"]),
        "events": copy.deepcopy(config["events"]),
        "checkpointing": copy.deepcopy(config["checkpointing"]),
        "terminal_audit": copy.deepcopy(config["terminal_audit"]),
        "formal_gate": copy.deepcopy(config["formal_gate"]),
        "automatic_retuning_allowed": False,
        "development_seeds_forbidden": list(
            config.get("development_seeds_forbidden_in_formal_aggregation", [0, 1, 2, 3, 4])
        ),
    }


def execute(
    config: dict[str, Any],
    stage: str,
    output_root: Path,
    device: torch.device,
) -> None:
    started = time.time()
    formal = is_formal_config(config)
    scientific_manifest_path = prepare_output_manifest_path(
        output_root, formal=formal, experiment_id=configured_experiment_id(config)
    )
    scientific_status = result_scientific_status(config)
    yaml_dump(output_root / "resolved_config.yaml", config)
    manifest = {
        "experiment_id": configured_experiment_id(config),
        "registered_scientific_status": config.get("scientific_status"),
        "execution_mode": config.get("execution_mode"),
        "requested_stage": stage,
        "run_class": "formal" if formal else "pilot",
        "formal_result": formal,
        "method_ranking_allowed_after_complete_audit": formal,
        "started_unix": started,
        "device": str(device),
        "source": source_manifest(output_root),
    }
    json_dump(scientific_manifest_path, manifest)

    seeds = run_seeds(config)
    requested_workers = int(nested(config, "optimization").get("parallel_workers", 1))
    workers = requested_workers if device.type == "cpu" else 1
    embedding_modes = sorted(
        set(["aligned", *list(nested(config, "protocol_c", "embedding_modes"))])
    )
    environments: dict[tuple[int, str], SemanticEnvironment] = {}
    audits: list[dict[str, Any]] = []
    for seed in seeds:
        for embedding_mode in embedding_modes:
            environment = SemanticEnvironment(config, seed, embedding_mode)
            audit = environment.audit()
            if workers == 1:
                environments[(seed, embedding_mode)] = environment
            audits.append(audit)
    json_dump(output_root / "environment_audits.json", audits)
    environment_invariants_passed = all(bool(audit["passed"]) for audit in audits)
    if not environment_invariants_passed:
        json_dump(
            output_root / "RUN_FAILED.json",
            {
                "experiment_id": configured_experiment_id(config),
                "reason": "environment_invariant_failure",
                "run_class": "formal" if formal else "pilot",
            },
        )
        raise RuntimeError("E6 environment invariant audit failed")

    if stage == "invariants":
        if formal:
            raise RuntimeError("formal config does not support an invariants-only launch")
        terminal_audit = {
            "experiment_id": configured_experiment_id(config),
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
                "experiment_id": configured_experiment_id(config),
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
    seed_blocks = (
        [list(block) for block in nested(config, "checkpointing", "seed_blocks")]
        if formal
        else [seeds]
    )
    expected_runs = len(seeds) * len(specs)
    completed_seeds: list[int] = []
    for block_index, seed_block in enumerate(seed_blocks):
        block_summaries: list[dict[str, Any]] = []
        tasks = [(config, seed, spec, output_root) for seed in seed_block for spec in specs]
        if workers > 1:
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
                results = executor.map(
                    run_one_reconstructed,
                    *(list(values) for values in zip(*tasks)),
                )
                for summary, trajectory in results:
                    block_summaries.append(summary)
                    write_trajectory_rows(trajectory_path, trajectory)
        else:
            initial_states: dict[tuple[int, str], dict[str, torch.Tensor]] = {}
            for seed in seed_block:
                for mode in {spec.concentration_mode for spec in specs}:
                    seed_all(seed + 10_000)
                    initial_model = SemanticPolicy(config, mode)
                    initial_states[(seed, mode)] = {
                        key: value.detach().clone()
                        for key, value in initial_model.state_dict().items()
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
                    block_summaries.append(summary)
                    write_trajectory_rows(trajectory_path, trajectory)
        summaries.extend(block_summaries)
        completed_seeds.extend(seed_block)
        if formal:
            apply_task_collapse_labels(summaries, config)
            json_dump(output_root / "per_run_summary.json", summaries)
            write_summary_csv(output_root / "per_run_summary.csv", summaries)
            json_dump(
                output_root / "aggregate_summary.json",
                aggregate_summaries(summaries),
            )
            write_formal_checkpoint(
                output_root,
                experiment_id=configured_experiment_id(config),
                block_index=block_index,
                block_seeds=seed_block,
                completed_seeds=completed_seeds,
                summaries=summaries,
                expected_total_runs=expected_runs,
            )

    apply_task_collapse_labels(summaries, config)
    json_dump(output_root / "per_run_summary.json", summaries)
    write_summary_csv(output_root / "per_run_summary.csv", summaries)
    aggregate = aggregate_summaries(summaries)
    json_dump(output_root / "aggregate_summary.json", aggregate)
    write_registered_horizon_summary(trajectory_path, config, output_root)
    if formal:
        json_dump(output_root / "formal_protocol_freeze.json", formal_protocol_freeze(config))
    else:
        freeze = pilot_freeze_recommendation(summaries, config)
        json_dump(output_root / "pilot_freeze_recommendation.json", freeze)

    all_runs_present = len(summaries) == expected_runs
    numerical_failure_count = sum(bool(row["nan_inf_numerical_failure"]) for row in summaries)
    missing_task_references = sum(
        bool(row.get("task_collapse_reference_missing")) for row in summaries
    )
    all_terminal_audits_present = all(bool(row.get("terminal_audit")) for row in summaries)
    all_terminal_audits_accepted = bool(
        all_terminal_audits_present
        and all(bool(row["terminal_audit"].get("formal_acceptance")) for row in summaries)
    )
    formal_two_x_for_all_non_numerical = bool(
        formal
        and all_terminal_audits_present
        and all(
            bool(row["terminal_audit"].get("formal_two_x_extension_performed"))
            or row["terminal_class"] == "numerical_failure"
            for row in summaries
        )
    )
    formal_audit_complete = bool(
        formal
        and all_runs_present
        and environment_invariants_passed
        and missing_task_references == 0
        and all_terminal_audits_accepted
        and formal_two_x_for_all_non_numerical
    )
    all_formal_terminal_plateau = bool(
        formal
        and all_runs_present
        and all(
            row.get("terminal_class") == "formal_terminal_plateau"
            for row in summaries
        )
    )
    stable_method_ranking_allowed = bool(
        formal_audit_complete and all_formal_terminal_plateau
    )
    terminal_audit = {
        "experiment_id": configured_experiment_id(config),
        "scientific_status": scientific_status if formal_audit_complete else "pilot",
        "run_class": "formal" if formal else "pilot",
        "expected_runs": expected_runs,
        "actual_runs": len(summaries),
        "all_runs_present": all_runs_present,
        "environment_invariants_passed": environment_invariants_passed,
        "all_terminal_audits_present": all_terminal_audits_present,
        "all_terminal_audits_accepted": all_terminal_audits_accepted,
        "missing_task_collapse_reference_count": missing_task_references,
        "nan_inf_numerical_failure_count": numerical_failure_count,
        "support_or_temperature_boundary_count": sum(
            bool(row["support_or_temperature_boundary"]) for row in summaries
        ),
        "task_performance_collapse_count": sum(
            row.get("task_performance_collapse") is True for row in summaries
        ),
        "development_two_x_horizon_performed": bool(
            nested(config, "terminal_audit").get("mode") == "focused_two_x_windows"
        ),
        "formal_two_x_extension_performed": formal_two_x_for_all_non_numerical,
        "formal_scientific_acceptance": formal_audit_complete,
        "formal_method_ranking_allowed": stable_method_ranking_allowed,
        "all_formal_terminal_plateau": all_formal_terminal_plateau,
        "scientific_failure_outcomes_preserved": True,
        "pilot_integrity_passed": bool(
            (not formal) and all_runs_present and numerical_failure_count == 0
        ),
    }
    if not formal:
        terminal_audit["remaining_gate"] = nested(config, "formal_gate", "reason")
    json_dump(output_root / "terminal_audit.json", terminal_audit)

    completed = (
        formal_audit_complete if formal else bool(all_runs_present and numerical_failure_count == 0)
    )
    completed_status = scientific_status if completed else "pilot"
    json_dump(
        output_root / "RUN_COMPLETE.json",
        {
            "experiment_id": configured_experiment_id(config),
            "scientific_status": completed_status,
            "execution_mode": config.get("execution_mode"),
            "completed": completed,
            "formal_result": formal,
            "method_ranking_allowed": stable_method_ranking_allowed,
            "expected_runs": expected_runs,
            "actual_runs": len(summaries),
            "elapsed_seconds": time.time() - started,
        },
    )
    if not completed:
        label = "formal long-run" if formal else "pilot"
        raise RuntimeError(f"E6 {label} did not complete its registered audit")


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
