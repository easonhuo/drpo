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
        "datasets": list(EXPECTED_DATASETS),
        "development_seeds": list(EXPECTED_SEEDS),
        "held_out_seeds": list(HELD_OUT_SEEDS),
        "actor_update_modes": list(ACTOR_MODES),
        "advantage_estimators": list(ESTIMATORS),
        "steps": STEPS,
        "evaluation_interval": 50_000,
        "evaluation_episodes": 10,
        "expected_total_branches": EXPECTED_BRANCHES,
        "formal_evidence_allowed": False,
    }
    for key, value in expected.items():
        if raw.get(key) != value:
            raise ValueError(f"GAE grid field changed: {key}")
    if raw.get("shared_frozen_critic") != {
        "steps": 100_000,
        "batch": 256,
        "gamma": 0.99,
        "tau": 0.5,
        "lr": 3e-4,
        "temperature": 5.0,
        "shared_per_dataset_seed": True,
        "updated_during_actor_training": False,
    }:
        raise ValueError("shared frozen critic contract changed")
    if raw.get("trajectory_advantage") != {
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "terminal_bootstrap": False,
        "timeout_bootstrap": True,
        "terminal_stops_recursion": True,
        "timeout_stops_recursion": True,
        "tail_stops_recursion": True,
        "normalization": "none",
        "clipping": "none",
    }:
        raise ValueError("trajectory GAE contract changed")
    control = raw.get("weight_control", {})
    if (
        control.get("formula") != FORMULA
        or float(control.get("weight_at_zero", -1)) != 1.0
        or control.get("positive_only_anchor") is not True
        or float(control.get("reference_distance", -1)) != night.REFERENCE_DISTANCE
        or tuple(float(value) for value in control.get("exp_coefficients", ()))
        != COEFFICIENTS
    ):
        raise ValueError("squared-EXP control contract changed")
    return raw, sha256_file(source)


def load_run_spec(path: str | Path) -> tuple[dict[str, Any], str]:
    run_spec, digest = night.load_run_spec(path)
    run_spec["seeds"] = list(EXPECTED_SEEDS)
    return run_spec, digest
