#!/usr/bin/env python3
"""Registered minimum-change E6 semantic conditional-gap protocol.

The scientific environment remains the original D-U1 E6 shared-semantic
categorical environment.  The only environment intervention is a deterministic
state-conditioned mask that removes the reward-optimal semantic neighbourhood
from logged positive/local/far roles on half of the contexts.  Reward geometry,
policy, optimiser, advantages, action count, and the original alpha anchors are
inherited from ``D-U1-E6-SEMANTIC-LONGRUN-01``.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

from drpo.du1_e6_semantic import (
    SEMANTIC_GAP_FORMAL_EXPERIMENT_ID,
    _require_exact,
    execute,
    nested,
)

EXPERIMENT_ID = SEMANTIC_GAP_FORMAL_EXPERIMENT_ID
PREDECESSOR_ID = "D-U1-E6-SEMANTIC-LONGRUN-01"
CONFIG_PATH = "configs/du1_e6_semantic_gap_longrun.yaml"
DEVELOPMENT_SEEDS = list(range(900, 910))
HELD_OUT_SEEDS = list(range(150, 170))
ALPHA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]


def validate_formal_config(config: Mapping[str, Any], stage: str = "formal") -> None:
    """Fail closed unless the exact user-approved successor protocol is present."""
    if stage != "formal":
        raise ValueError("semantic-gap formal config may only run with stage=formal")
    _require_exact(config.get("experiment_id"), EXPERIMENT_ID, "experiment_id")
    _require_exact(config.get("scientific_status"), "not_run", "scientific_status")
    _require_exact(config.get("execution_mode"), "formal_longrun", "execution_mode")
    _require_exact(bool(config.get("formal_parameter_freeze")), True, "formal_parameter_freeze")
    _require_exact(config.get("predecessor"), PREDECESSOR_ID, "predecessor")
    _require_exact(
        dict(config.get("approval", {})),
        {
            "user_approved": True,
            "approval_date": "2026-06-27",
            "approval_scope": "minimum_change_semantic_gap_formal_promotion",
            "automatic_retuning_allowed": False,
        },
        "approval",
    )
    _require_exact(list(nested(config, "seeds", "development")), [], "seeds.development")
    _require_exact(
        list(nested(config, "seeds", "held_out_formal")),
        HELD_OUT_SEEDS,
        "seeds.held_out_formal",
    )
    _require_exact(
        list(config.get("development_seeds_forbidden_in_formal_aggregation", [])),
        DEVELOPMENT_SEEDS,
        "development_seeds_forbidden_in_formal_aggregation",
    )
    _require_exact(
        dict(nested(config, "data")),
        {
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
        },
        "data",
    )
    _require_exact(
        dict(nested(config, "conditional_coverage")),
        {
            "mode": "structured_semantic_neighbourhood_gap",
            "gap_state_fraction": 0.5,
            "withheld_action_fraction": 0.25,
            "state_partition": "top_half_of_state_0_plus_0_37_state_1",
            "withhold_basis": "per_state_reward_similarity_top_fraction",
            "apply_to_logged_roles": ["positive", "local_negative", "far_negative"],
            "evaluation_oracle_remains_complete": True,
            "require_global_action_coverage": True,
        },
        "conditional_coverage",
    )
    _require_exact(
        dict(nested(config, "geometry")),
        {
            "target_offset": 0.45,
            "reward_scale": 0.5,
            "positive_advantage": 1.0,
            "negative_advantage": -1.0,
            "fixed_advantage": True,
            "random_action_id_permutation": True,
        },
        "geometry",
    )
    _require_exact(
        dict(nested(config, "policy")),
        {
            "hidden_dim": 64,
            "fixed_concentration": 8.0,
            "learnable_concentration_floor": 0.05,
            "initial_learnable_concentration": 8.0,
            "learnable_concentration_upper_clamp": False,
            "activation": "tanh",
        },
        "policy",
    )
    _require_exact(
        dict(nested(config, "optimization")),
        {
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "betas": [0.9, 0.999],
            "eps": 1.0e-8,
            "batch_size": 128,
            "maximum_steps": 32000,
            "evaluation_interval_steps": 200,
            "audit_states": 512,
            "parallel_workers": 8,
            "far_cap_ratio_to_weighted_local_gradient": 1.0,
        },
        "optimization",
    )
    _require_exact(
        dict(nested(config, "protocol_a")),
        {
            "responsibility": "minimum_change_gap_alpha_utility_and_uncontrolled_degradation",
            "concentration_mode": "fixed",
            "local_alpha_grid": ALPHA_GRID,
            "far_pressure_lambda": 0.0,
        },
        "protocol_a",
    )
    _require_exact(
        dict(nested(config, "protocol_b")),
        {"responsibility": "not_in_scope", "concentration_mode": "fixed", "settings": []},
        "protocol_b",
    )
    _require_exact(
        dict(nested(config, "protocol_c")),
        {
            "responsibility": "not_in_scope",
            "concentration_mode": "fixed",
            "local_alpha": 0.0,
            "far_pressure_lambda": 0.0,
            "embedding_modes": [],
            "methods": [],
        },
        "protocol_c",
    )
    _require_exact(
        list(config.get("primary_metrics", [])),
        [
            "overall_expected_semantic_reward",
            "paired_reward_difference_vs_positive_only",
            "reward_trajectory_across_registered_horizons",
        ],
        "primary_metrics",
    )
    _require_exact(
        list(config.get("registered_horizon_checkpoints", [])),
        [4000, 8000, 16000, 24000, 32000],
        "registered_horizon_checkpoints",
    )
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
    _require_exact(
        dict(nested(config, "terminal_audit")),
        {
            "mode": "formal_two_x_windows",
            "development_reference_horizon_steps": 16000,
            "formal_horizon_steps": 32000,
            "formal_extension_factor": 2.0,
            "window_1_steps": [16000, 24000],
            "window_2_steps": [24000, 32000],
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
        },
        "terminal_audit",
    )
    blocks = [list(range(start, start + 5)) for start in (150, 155, 160, 165)]
    _require_exact(
        dict(nested(config, "checkpointing")),
        {
            "seed_block_size": 5,
            "seed_blocks": blocks,
            "persistence": "persistent_local",
            "write_compact_manifest_after_each_block": True,
        },
        "checkpointing",
    )
    _require_exact(
        dict(nested(config, "formal_gate")),
        {
            "enabled": True,
            "approval_record": "user_approved_2026-06-27_minimum_change_semantic_gap",
            "frozen_protocol_path": CONFIG_PATH,
            "held_out_seeds": HELD_OUT_SEEDS,
        },
        "formal_gate",
    )
    _require_exact(
        list(nested(config, "outputs", "required")),
        [
            "resolved_config.yaml",
            "scientific_run_manifest.json",
            "environment_audits.json",
            "trajectories.jsonl",
            "per_run_summary.json",
            "per_run_summary.csv",
            "aggregate_summary.json",
            "horizon_summary.json",
            "horizon_summary.csv",
            "terminal_audit.json",
            "formal_protocol_freeze.json",
            "run_manifest.json",
            "RUN_COMPLETE.json",
        ],
        "outputs.required",
    )


def smoke_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return an engineering-only shrink that cannot be mistaken for formal evidence."""
    out = copy.deepcopy(dict(config))
    out["experiment_id"] = "D-U1-E6-SEMANTIC-PILOT-01"
    out["scientific_status"] = "pilot"
    out["execution_mode"] = "engineering_smoke"
    out["formal_parameter_freeze"] = False
    out["seeds"] = {"development": [0], "held_out_formal": []}
    out["data"]["train_states"] = 32
    out["data"]["test_states"] = 32
    out["optimization"]["batch_size"] = 8
    out["optimization"]["maximum_steps"] = 2
    out["optimization"]["evaluation_interval_steps"] = 2
    out["optimization"]["audit_states"] = 8
    out["optimization"]["parallel_workers"] = 1
    out["protocol_a"]["local_alpha_grid"] = [0.0, 0.5, 1.0]
    out["registered_horizon_checkpoints"] = []
    out["terminal_audit"] = {
        "mode": "sandbox_trailing",
        "trailing_evaluations_per_window": 1,
        "normalized_metric_change_tolerance": 1.0,
        "raw_total_gradient_median_tolerance": 1.0e12,
    }
    out["checkpointing"] = {
        "seed_block_size": 1,
        "seed_blocks": [[0]],
        "persistence": "temporary",
        "write_compact_manifest_after_each_block": False,
    }
    out["formal_gate"] = {"enabled": False, "reason": "engineering_smoke_only"}
    return out


def run_formal(config: dict[str, Any], output_root: Path, device: Any) -> None:
    validate_formal_config(config, "formal")
    execute(config, "formal", output_root, device)
