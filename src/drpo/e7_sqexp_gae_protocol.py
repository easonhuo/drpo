"""Frozen scientific matrix for EXT-H-E7-SQEXP-GAE-01."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from drpo import e7_squared_exp_night as night
from drpo.e7_canonical_injection import sha256_file
from drpo.e7_squared_exp_kernel import FORMULA

EXPERIMENT_ID = "EXT-H-E7-SQEXP-GAE-01"
SCIENTIFIC_STATUS = "frozen_critic_trajectory_gae_development_pilot_only"
RUNNER_VERSION = "2.0.0-minimal-canonical-wrapper"
EXPECTED_DATASETS = night.EXPECTED_DATASETS
EXPECTED_SEEDS = (200, 201, 202, 203)
HELD_OUT_SEEDS = night.HELD_OUT_SEEDS
ACTOR_MODES = ("a2c", "ppo_clip_k4")
ESTIMATORS = ("td", "gae")
COEFFICIENTS = (64.0, 128.0, 256.0)
EXPECTED_BRANCHES = 192
STEPS = 1_000_000


def load_grid(path: str | Path) -> tuple[dict[str, Any], str]:
    source = Path(path)
    raw = json.loads(source.read_text())
    expected = {
        "experiment_id": EXPERIMENT_ID,
        "run_kind": "pilot",
        "status": "not_run",
        "scientific_status": SCIENTIFIC_STATUS,
        "predecessor_experiment_id": "EXT-H-E7-SQUARED-EXP-NIGHT-01",
        "datasets": list(EXPECTED_DATASETS),
        "development_seeds": list(EXPECTED_SEEDS),
        "held_out_seeds": list(HELD_OUT_SEEDS),
        "steps": STEPS,
        "evaluation_interval": 50_000,
        "evaluation_episodes": 10,
        "actor_update_modes": list(ACTOR_MODES),
        "advantage_modes": ["one_step_td", "gae_lambda_0p95"],
        "shared_frozen_critic": {
            "steps": 100_000,
            "batch": 256,
            "gamma": 0.99,
            "tau": 0.5,
            "lr": 3e-4,
            "temperature": 5.0,
            "device": "cpu",
            "shared_per_dataset_seed": True,
            "updated_during_actor_training": False,
        },
        "trajectory_advantage": {
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "ordered_behavior_trajectory": True,
            "terminal_bootstrap": False,
            "timeout_bootstrap": True,
            "terminal_stops_recursion": True,
            "timeout_stops_recursion": True,
            "tail_bootstrap_and_stop_recursion": True,
            "lambda_zero_must_equal_one_step": True,
            "normalization": "none",
            "clipping": "none",
        },
        "weight_control": {
            "weight_at_zero": 1.0,
            "positive_only_anchor": True,
            "reference_distance": night.REFERENCE_DISTANCE,
            "formula": FORMULA,
            "exp_coefficients": list(COEFFICIENTS),
        },
        "ppo": {
            "clip_epsilon": 0.2,
            "updates_per_old_policy": 4,
            "analytic_kl_early_refresh": False,
            "kl_penalty": False,
            "entropy_bonus": False,
            "actor_gradient_clip": False,
            "value_clip": False,
        },
        "diagnostics": {
            "interval": 1000,
            "sampled_values_per_update": 16,
            "record_500k_intermediate": True,
            "late_window_start": 800_000,
            "separate_task_support_numerical_events": True,
        },
        "expected_controls_per_actor_advantage_cell": 4,
        "expected_total_branches": EXPECTED_BRANCHES,
        "screening_only": True,
        "formal_evidence_allowed": False,
        "non_claims": [
            "convergence_from_fixed_1m_horizon",
            "steady_state_method_ranking",
            "universal_gae_superiority",
            "universal_ppo_or_a2c_superiority",
            "causal_actor_update_identification",
            "ood_generalization",
            "replacement_of_controlled_causal_evidence",
        ],
    }
    if raw != expected:
        changed = sorted(
            key for key in set(raw) | set(expected) if raw.get(key) != expected.get(key)
        )
        raise ValueError(f"frozen GAE grid changed: {changed}")
    return raw, sha256_file(source)


def load_run_spec(path: str | Path) -> tuple[dict[str, Any], str]:
    run_spec, digest = night.load_run_spec(path)
    run_spec["seeds"] = list(EXPECTED_SEEDS)
    return run_spec, digest
