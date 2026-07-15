#!/usr/bin/env python3
"""Thin adapter for the E8 alpha=1 logarithmic high-c boundary scan.

The training objective and runtime are inherited unchanged from the preceding
continuous-EXP pilots. This module changes only experiment identity, the frozen
parameter points, development seeds, selection gates, and protected-source
identity.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

from drpo import countdown_e8_alpha1_highc_scan_common as _base

EXPERIMENT_ID = (
    "EXT-C-E8-ORACLE-OFFLINE-V2-ALPHA1-LOGC-BOUNDARY-SCAN-0.5B-01"
)
VERSION = "0.1.0-dev-code-first"
DEFAULT_GRID_CONFIG = (
    "configs/countdown_e8_oracle_offline_v2_alpha1_logc_boundary_scan_0p5b.yaml"
)
PARAMETER_POINTS = (
    (0.5, 1.0),
    (1.0, 8.0),
    (1.0, 12.0),
    (1.0, 16.0),
    (1.0, 24.0),
    (1.0, 32.0),
    (1.0, 64.0),
    (1.0, 128.0),
)
SEED_OFFSETS = (13000, 14000, 15000, 16000)
EXPECTED_POINTS = 8
EXPECTED_CELLS = 32

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

_PATCH_KEYS = (
    "EXPERIMENT_ID",
    "VERSION",
    "DEFAULT_GRID_CONFIG",
    "PARAMETER_POINTS",
    "SEED_OFFSETS",
    "EXPECTED_POINTS",
    "EXPECTED_CELLS",
    "_identity",
)


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
        "logc_common": Path(__file__).resolve(),
        "highc_common": Path(_base.__file__).resolve(),
        "alpha1_base_common": Path(_base._base.__file__).resolve(),
        "alpha1_base_trainer": package_dir / "countdown_e8_alpha1_c_scan_trainer.py",
        "alpha1_base_runtime": package_dir / "countdown_e8_alpha1_c_scan_runtime.py",
        "logc_runtime": package_dir
        / "countdown_e8_alpha1_logc_boundary_scan_runtime.py",
        "logc_auto_launcher": repo
        / "scripts"
        / "run_countdown_e8_oracle_offline_v2_alpha1_logc_boundary_scan_auto.py",
    }
    missing = [name for name, path in source_paths.items() if not path.is_file()]
    if missing:
        raise RuntimeError(
            "Alpha=1 log-c boundary identity is missing protected sources: "
            + ", ".join(sorted(missing))
        )
    return {
        "experiment_id": EXPERIMENT_ID,
        "version": VERSION,
        "source": _base.git_state(repo),
        "source_sha256": {
            name: _base.sha256_file(path) for name, path in sorted(source_paths.items())
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


def _apply() -> None:
    values = {
        "EXPERIMENT_ID": EXPERIMENT_ID,
        "VERSION": VERSION,
        "DEFAULT_GRID_CONFIG": DEFAULT_GRID_CONFIG,
        "PARAMETER_POINTS": PARAMETER_POINTS,
        "SEED_OFFSETS": SEED_OFFSETS,
        "EXPECTED_POINTS": EXPECTED_POINTS,
        "EXPECTED_CELLS": EXPECTED_CELLS,
        "_identity": _identity,
    }
    for name, value in values.items():
        setattr(_base, name, value)


@contextmanager
def activated() -> Iterator[None]:
    """Temporarily activate this identity without polluting other tests."""
    previous = {name: getattr(_base, name) for name in _PATCH_KEYS}
    _apply()
    try:
        with _base.activated():
            yield
    finally:
        for name, value in previous.items():
            setattr(_base, name, value)


def activate() -> None:
    """Permanently activate this identity in a dedicated runtime process."""
    _apply()
    _base.activate()


def validate_grid_config(config: Mapping[str, Any]) -> None:
    with activated():
        _base.validate_grid_config(config)
    sweep = config["sweep"]
    evaluation = config["evaluation"]
    if sweep.get("logarithmic_boundary_search") is not True:
        raise ValueError("logarithmic_boundary_search must remain true")
    if sweep.get("automatic_further_extension_forbidden") is not True:
        raise ValueError("Automatic further c extension must remain forbidden")
    if evaluation.get("paired_direction_gate_for_followup") != (
        "at_least_3_of_4_seeds"
    ):
        raise ValueError("Paired direction gate must remain at least 3 of 4 seeds")
    if evaluation.get("material_lift_gate") != (
        "qualitative_paired_lift_no_numeric_cutoff_registered"
    ):
        raise ValueError("Material lift gate must remain qualitative and paired")


def parameter_points(config: Mapping[str, Any]) -> tuple[tuple[float, float], ...]:
    validate_grid_config(config)
    points = tuple(
        (float(item["alpha"]), float(item["c"]))
        for item in config["sweep"]["parameter_points"]
    )
    if points != PARAMETER_POINTS or len(set(points)) != EXPECTED_POINTS:
        raise AssertionError("Alpha=1 log-c boundary scan must produce 8 unique points")
    return points


def build_cells(config: Mapping[str, Any]) -> tuple[Cell, ...]:
    points = parameter_points(config)
    cells = tuple(
        Cell(alpha=alpha, c=coefficient, seed_offset=seed_offset)
        for alpha, coefficient in points
        for seed_offset in SEED_OFFSETS
    )
    if (
        len(cells) != EXPECTED_CELLS
        or len({cell.name for cell in cells}) != EXPECTED_CELLS
    ):
        raise AssertionError("Alpha=1 log-c boundary scan must produce 32 unique cells")
    return cells


def identity_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)
