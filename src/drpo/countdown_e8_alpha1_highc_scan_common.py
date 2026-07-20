#!/usr/bin/env python3
"""Thin adapter for the paper-aligned E8 linear-surprisal scan.

The trainer and runtime are inherited unchanged from the alpha=1 c-scan. This
module changes only the experiment identity, explicit parameter points,
development seeds, and the single taper exponent coordinate.
"""
from __future__ import annotations

import copy
import json
import math
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch

from drpo import countdown_e8_alpha1_c_scan_common as _base

_BASE_VALIDATE_GRID_CONFIG = _base.validate_grid_config
EXPERIMENT_ID = "EXT-C-E8-ORACLE-OFFLINE-V2-PAPER-ALIGNED-LINEAR-SCAN-0.5B-01"
VERSION = "0.2.0-dev-code-first-one-line"
DEFAULT_GRID_CONFIG = (
    "configs/countdown_e8_oracle_offline_v2_alpha1_highc_scan_0p5b.yaml"
)
PARAMETER_POINTS = (
    (0.0, 0.0),
    (1.0, 0.0),
    (1.0, 0.051293294),
    (1.0, 0.105360516),
    (1.0, 0.162518929),
    (1.0, 0.223143551),
    (1.0, 0.287682072),
    (1.0, 0.430782916),
    (1.0, 0.693147181),
    (1.0, 0.916290732),
    (1.0, 1.203972804),
    (1.0, 1.386294361),
    (1.0, 1.609437912),
    (1.0, 1.897119985),
    (1.0, 2.302585093),
    (1.0, 2.995732274),
)
SEED_OFFSETS = (4000, 5000)
EXPECTED_POINTS = 16
EXPECTED_CELLS = 32

Cell = _base.Cell
sha256_file = _base.sha256_file
atomic_json = _base.atomic_json
load_yaml = _base.load_yaml
continuous_remoteness = _base.continuous_remoteness
mean_unique_negative_term = _base.mean_unique_negative_term
unique_negative_expressions = _base.unique_negative_expressions
ContinuousUniqueBankDataset = _base.ContinuousUniqueBankDataset
make_continuous_unique_bank_collator = _base.make_continuous_unique_bank_collator
weight_diagnostics = _base.weight_diagnostics
git_state = _base.git_state


def continuous_exp_weights(
    seq_lp: torch.Tensor,
    *,
    alpha: float,
    c: float,
    reference_distance: float = _base.REFERENCE_DISTANCE,
) -> torch.Tensor:
    if not math.isfinite(alpha) or alpha < 0.0:
        raise ValueError("alpha must be finite and non-negative")
    if not math.isfinite(c) or c < 0.0:
        raise ValueError("c must be finite and non-negative")
    u = continuous_remoteness(seq_lp, reference_distance=reference_distance)
    return (float(alpha) * torch.exp(-float(c) * u)).detach()


_PATCH_KEYS = (
    "EXPERIMENT_ID",
    "VERSION",
    "DEFAULT_GRID_CONFIG",
    "PARAMETER_POINTS",
    "SEED_OFFSETS",
    "EXPECTED_POINTS",
    "EXPECTED_CELLS",
    "continuous_exp_weights",
    "validate_grid_config",
    "parameter_points",
    "build_cells",
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
        "highc_common": Path(__file__).resolve(),
        "base_common": Path(_base.__file__).resolve(),
        "base_trainer": package_dir / "countdown_e8_alpha1_c_scan_trainer.py",
        "base_runtime": package_dir / "countdown_e8_alpha1_c_scan_runtime.py",
        "highc_runtime": package_dir / "countdown_e8_alpha1_highc_scan_runtime.py",
        "highc_auto_launcher": (
            repo
            / "scripts"
            / "run_countdown_e8_oracle_offline_v2_alpha1_highc_scan_auto.py"
        ),
    }
    missing = [name for name, path in source_paths.items() if not path.is_file()]
    if missing:
        raise RuntimeError(
            "Paper-aligned scan identity is missing protected sources: "
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
        "continuous_exp_weights": continuous_exp_weights,
        "validate_grid_config": validate_grid_config,
        "parameter_points": parameter_points,
        "build_cells": build_cells,
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
        yield
    finally:
        for name, value in previous.items():
            setattr(_base, name, value)


def activate() -> None:
    """Permanently activate this identity in a dedicated runtime process."""
    _apply()


def validate_grid_config(config: Mapping[str, Any]) -> None:
    predecessor_compatible = copy.deepcopy(config)
    predecessor_compatible["remoteness"]["weight"] = "alpha*exp(-c*u^2)"
    with activated():
        _BASE_VALIDATE_GRID_CONFIG(predecessor_compatible)
    if config["remoteness"].get("weight") != "alpha*exp(-c*u)":
        raise ValueError("The paper-aligned weight must be alpha*exp(-c*u)")
    points = tuple(
        (float(item["alpha"]), float(item["c"]))
        for item in config["sweep"]["parameter_points"]
    )
    if sum(alpha == 0.0 for alpha, _ in points) != 1:
        raise ValueError("Exactly one Positive-only point is required")
    if config["sweep"].get("positive_only_same_seed_control") is not True:
        raise ValueError("positive_only_same_seed_control must remain true")
    if config["execution"].get("default_gpus") != list(range(8)):
        raise ValueError("The paper-aligned scan requires GPU 0-7")
    if config["evaluation"].get("primary_selection_metric") != (
        "late_window_pass_at_8"
    ):
        raise ValueError("Primary selection metric must remain late_window_pass_at_8")
    if config["evaluation"].get("secondary_selection_metric") != (
        "terminal_pass_at_8"
    ):
        raise ValueError("Secondary selection metric must remain terminal_pass_at_8")
    if (
        config["evaluation"].get("best_checkpoint_metric_is_supplementary_only")
        is not True
    ):
        raise ValueError("Best checkpoint metric must remain supplementary only")


def parameter_points(config: Mapping[str, Any]) -> tuple[tuple[float, float], ...]:
    validate_grid_config(config)
    points = tuple(
        (float(item["alpha"]), float(item["c"]))
        for item in config["sweep"]["parameter_points"]
    )
    if points != PARAMETER_POINTS or len(set(points)) != EXPECTED_POINTS:
        raise AssertionError("Paper-aligned scan must produce 16 unique points")
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
        raise AssertionError("Paper-aligned scan must produce 32 unique cells")
    return cells


def identity_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)
