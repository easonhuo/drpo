#!/usr/bin/env python3
"""Frozen six-cell E8 reproducibility and evaluation-RNG audit adapter."""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

from drpo import countdown_e8_alpha1_c_scan_common as _base

EXPERIMENT_ID = "EXT-C-E8-ORACLE-OFFLINE-V2-REPRO-RNG-AUDIT-0.5B-01"
DEFAULT_GRID_CONFIG = (
    "configs/countdown_e8_oracle_offline_v2_repro_rng_audit_0p5b.yaml"
)
PROTOCOLS = ("legacy_contaminated_v1", "rng_isolated_v2")
AUDIT_CELL_SPECS = (
    (0.5, 1.0, 3000),
    (0.5, 1.0, 4000),
    (0.5, 1.0, 13000),
    (0.5, 1.0, 16000),
    (1.0, 8.0, 13000),
    (1.0, 8.0, 16000),
)
PARAMETER_POINTS = ((0.5, 1.0), (1.0, 8.0))
SEED_OFFSETS = (3000, 4000, 13000, 16000)
EXPECTED_POINTS = 2
EXPECTED_CELLS = 6

Cell = _base.Cell
sha256_file = _base.sha256_file
atomic_json = _base.atomic_json
load_yaml = _base.load_yaml
continuous_remoteness = _base.continuous_remoteness
continuous_exp_weights = _base.continuous_exp_weights
mean_unique_negative_term = _base.mean_unique_negative_term
unique_negative_expressions = _base.unique_negative_expressions
ContinuousUniqueBankDataset = _base.ContinuousUniqueBankDataset
make_continuous_unique_bank_collator = _base.make_continuous_unique_bank_collator
weight_diagnostics = _base.weight_diagnostics
git_state = _base.git_state

_ACTIVE_PROTOCOL = PROTOCOLS[0]
_PATCH_KEYS = (
    "EXPERIMENT_ID",
    "VERSION",
    "DEFAULT_GRID_CONFIG",
    "PARAMETER_POINTS",
    "SEED_OFFSETS",
    "EXPECTED_POINTS",
    "EXPECTED_CELLS",
    "validate_grid_config",
    "parameter_points",
    "build_cells",
    "_identity",
)


def version_for(protocol: str) -> str:
    if protocol not in PROTOCOLS:
        raise ValueError(f"Unknown RNG protocol: {protocol}")
    return f"0.1.0-dev-code-first-{protocol}"


def validate_grid_config(config: Mapping[str, Any]) -> None:
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("Repro RNG audit experiment_id mismatch")
    if config.get("result_status") != "pilot":
        raise ValueError("Repro RNG audit must remain a pilot")
    if config.get("registration_state") != "dev_code_first_unregistered":
        raise ValueError("Code-first audit must remain unregistered until SHA freeze")

    bank = config.get("bank", {})
    if bank.get("use_all_unique_negatives") is not True:
        raise ValueError("Every first-occurrence unique negative must participate")
    if bank.get("explicit_near_far_training_classes") is not False:
        raise ValueError("Near/far training classes are forbidden")
    if bank.get("extreme_selection_forbidden") is not True:
        raise ValueError("Extreme negative selection must remain forbidden")

    remoteness = config.get("remoteness", {})
    if remoteness.get("coordinate") != "u=d/2":
        raise ValueError("Frozen coordinate must remain u=d/2")
    if remoteness.get("weight") != "alpha*exp(-c*u^2)":
        raise ValueError("Frozen weight must remain alpha*exp(-c*u^2)")
    if float(remoteness.get("reference_distance", -1.0)) != 2.0:
        raise ValueError("reference_distance must remain 2.0")
    if remoteness.get("detached") is not True:
        raise ValueError("Remoteness weights must remain detached")

    sweep = config.get("sweep", {})
    configured = tuple(
        (float(item["alpha"]), float(item["c"]), int(item["seed_offset"]))
        for item in sweep.get("audit_cells", ())
    )
    if configured != AUDIT_CELL_SPECS:
        raise ValueError(f"Audit cells changed: {configured}")
    if tuple(sweep.get("protocols", ())) != PROTOCOLS:
        raise ValueError("Both frozen RNG protocols are required in fixed order")
    if int(sweep.get("cells_per_protocol", -1)) != EXPECTED_CELLS:
        raise ValueError("Each protocol must contain exactly six cells")
    if int(sweep.get("total_cells", -1)) != 2 * EXPECTED_CELLS:
        raise ValueError("The complete audit must contain exactly twelve cells")

    training = config.get("training", {})
    required_training = {
        "steps": 1200,
        "eval_every": 100,
        "pass64_every": 200,
        "early_stop": False,
        "denominator": "unique_negative_count_per_prompt",
        "normalize_by_weight_sum": False,
        "hidden_negative_scale": False,
        "gradient_budget_matching": False,
    }
    for key, expected in required_training.items():
        if training.get(key) != expected:
            raise ValueError(f"Frozen training field changed: {key}")

    execution = config.get("execution", {})
    if int(execution.get("parallel_cells_per_gpu", -1)) != 1:
        raise ValueError("Reproduction audit requires one cell per GPU")
    if execution.get("sequential_protocol_phases") is not True:
        raise ValueError("Legacy and isolated phases must run sequentially")
    if execution.get("same_gpu_pool_for_both_phases") is not True:
        raise ValueError("Both phases must use the same GPU pool")
    if execution.get("identity_checked_resume") is not True:
        raise ValueError("Identity-checked resume is required")

    evaluation = config.get("evaluation", {})
    if evaluation.get("validation_only_during_audit") is not True:
        raise ValueError("Audit must remain validation-only")
    if evaluation.get("test_access_forbidden") is not True:
        raise ValueError("Test access must remain forbidden")
    if evaluation.get("best_checkpoint_selects_protocol") is not False:
        raise ValueError("Best checkpoint may not select an RNG protocol")


def parameter_points(config: Mapping[str, Any]) -> tuple[tuple[float, float], ...]:
    validate_grid_config(config)
    return PARAMETER_POINTS


def build_cells(config: Mapping[str, Any]) -> tuple[Cell, ...]:
    validate_grid_config(config)
    cells = tuple(
        Cell(alpha=alpha, c=coefficient, seed_offset=seed_offset)
        for alpha, coefficient, seed_offset in AUDIT_CELL_SPECS
    )
    if (
        len(cells) != EXPECTED_CELLS
        or len({cell.name for cell in cells}) != EXPECTED_CELLS
    ):
        raise AssertionError("Repro RNG audit must produce six unique cells")
    return cells


def _identity(
    *,
    repo: Path,
    model_path: Path,
    bank: Path,
    val: Path,
    base_config: Path,
    grid_config: Path,
    cell: Cell,
    smoke: bool,
) -> dict[str, Any]:
    package_dir = Path(__file__).resolve().parent
    source_paths = {
        "audit_common": Path(__file__).resolve(),
        "rng_isolation": package_dir / "countdown_e8_rng_isolation.py",
        "legacy_runtime": package_dir / "countdown_e8_repro_legacy_runtime.py",
        "isolated_runtime": package_dir
        / "countdown_e8_repro_rng_isolated_runtime.py",
        "base_common": Path(_base.__file__).resolve(),
        "base_trainer": package_dir / "countdown_e8_alpha1_c_scan_trainer.py",
        "base_runtime": package_dir / "countdown_e8_alpha1_c_scan_runtime.py",
        "audit_launcher": repo
        / "scripts"
        / "run_countdown_e8_oracle_offline_v2_repro_rng_audit.py",
    }
    missing = [name for name, path in source_paths.items() if not path.is_file()]
    if missing:
        raise RuntimeError(
            "Repro RNG identity is missing protected sources: "
            + ", ".join(sorted(missing))
        )
    return {
        "experiment_id": EXPERIMENT_ID,
        "version": version_for(_ACTIVE_PROTOCOL),
        "rng_protocol": _ACTIVE_PROTOCOL,
        "source": _base.git_state(repo),
        "source_sha256": {
            name: _base.sha256_file(path)
            for name, path in sorted(source_paths.items())
        },
        "model_path": str(model_path),
        "bank_sha256": _base.sha256_file(bank),
        "validation_sha256": _base.sha256_file(val),
        "base_config_sha256": _base.sha256_file(base_config),
        "grid_config_sha256": _base.sha256_file(grid_config),
        "cell": {
            "name": cell.name,
            "method": cell.method,
            "alpha": cell.alpha,
            "c": cell.c,
            "seed_offset": cell.seed_offset,
        },
        "smoke": bool(smoke),
        "test_data_used": False,
    }


def _apply(protocol: str) -> None:
    global _ACTIVE_PROTOCOL
    if protocol not in PROTOCOLS:
        raise ValueError(f"Unknown RNG protocol: {protocol}")
    _ACTIVE_PROTOCOL = protocol
    values = {
        "EXPERIMENT_ID": EXPERIMENT_ID,
        "VERSION": version_for(protocol),
        "DEFAULT_GRID_CONFIG": DEFAULT_GRID_CONFIG,
        "PARAMETER_POINTS": PARAMETER_POINTS,
        "SEED_OFFSETS": SEED_OFFSETS,
        "EXPECTED_POINTS": EXPECTED_POINTS,
        "EXPECTED_CELLS": EXPECTED_CELLS,
        "validate_grid_config": validate_grid_config,
        "parameter_points": parameter_points,
        "build_cells": build_cells,
        "_identity": _identity,
    }
    for name, value in values.items():
        setattr(_base, name, value)


@contextmanager
def activated(protocol: str) -> Iterator[None]:
    previous = {name: getattr(_base, name) for name in _PATCH_KEYS}
    previous_protocol = _ACTIVE_PROTOCOL
    _apply(protocol)
    try:
        yield
    finally:
        globals()["_ACTIVE_PROTOCOL"] = previous_protocol
        for name, value in previous.items():
            setattr(_base, name, value)


def activate(protocol: str) -> None:
    _apply(protocol)


def identity_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)
