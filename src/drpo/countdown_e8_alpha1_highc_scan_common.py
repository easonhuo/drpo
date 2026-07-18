#!/usr/bin/env python3
"""Thin adapters for the paper-aligned E8 linear-surprisal scans.

The trainer and runtime are inherited unchanged from the alpha=1 c-scan. This
module changes only the experiment identity, explicit parameter points,
development seeds, and the single taper exponent coordinate. The completed
Round-1 profile remains the default; the 16-cell extension profile is selected
only from its frozen grid config.
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
ROUND1_EXPERIMENT_ID = (
    "EXT-C-E8-ORACLE-OFFLINE-V2-PAPER-ALIGNED-LINEAR-SCAN-0.5B-01"
)
C_EXTENSION_EXPERIMENT_ID = (
    "EXT-C-E8-ORACLE-OFFLINE-V2-PAPER-ALIGNED-LINEAR-C-EXTENSION-0.5B-01"
)
TAU_EXPERIMENT_ID = (
    "EXT-C-E8-ORACLE-OFFLINE-V2-PAPER-ALIGNED-TAU-CURVE-0.5B-01"
)
TAU_C_RANGE_EXTENSION_EXPERIMENT_ID = (
    "EXT-C-E8-ORACLE-OFFLINE-V2-PAPER-ALIGNED-"
    "TAU-C-RANGE-EXTENSION-0.5B-01"
)
ROUND1_PARAMETER_POINTS = (
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
C_EXTENSION_PARAMETER_POINTS = (
    (1.0, 0.01),
    (1.0, 0.025),
    (1.0, 0.04),
    (1.0, 3.506557897),
    (1.0, 4.605170186),
    (1.0, 5.298317367),
    (1.0, 6.907755279),
    (1.0, 9.210340372),
)
TAU_C_VALUES = (1.609437912, 1.897119985, 2.995732274, 4.605170186)
TAU_VALUES = (0.0, 0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.25)
TAU_PARAMETER_POINTS = tuple(
    (1.0, coefficient, tau) for coefficient in TAU_C_VALUES for tau in TAU_VALUES
)
TAU_BASE_POINTS = tuple(
    (alpha, coefficient) for alpha, coefficient, _ in TAU_PARAMETER_POINTS
)
TAU_C_RANGE_EXTENSION_C_VALUES = (
    1.386294361,
    1.609437912,
    4.605170186,
    5.298317367,
)
TAU_C_RANGE_EXTENSION_PARAMETER_POINTS = tuple(
    (1.0, coefficient, tau)
    for coefficient in TAU_C_RANGE_EXTENSION_C_VALUES
    for tau in TAU_VALUES
)
TAU_C_RANGE_EXTENSION_BASE_POINTS = tuple(
    (alpha, coefficient)
    for alpha, coefficient, _ in TAU_C_RANGE_EXTENSION_PARAMETER_POINTS
)
SEED_OFFSETS = (4000, 5000)
ROUND1_RESULT_MANIFEST_SHA256 = (
    "24635fbb634b23450cdfb560fd7b16a2dc0fe4a6d0586f10e1cf385e58bab333"
)
_PROFILES: dict[str, dict[str, Any]] = {
    ROUND1_EXPERIMENT_ID: {
        "experiment_id": ROUND1_EXPERIMENT_ID,
        "version": "0.2.0-dev-code-first-one-line",
        "default_grid_config": (
            "configs/countdown_e8_oracle_offline_v2_alpha1_highc_scan_0p5b.yaml"
        ),
        "parameter_points": ROUND1_PARAMETER_POINTS,
        "seed_offsets": SEED_OFFSETS,
        "expected_points": 16,
        "expected_cells": 32,
        "requires_positive_only": True,
    },
    C_EXTENSION_EXPERIMENT_ID: {
        "experiment_id": C_EXTENSION_EXPERIMENT_ID,
        "version": "0.3.0-dev-code-first-c-extension",
        "default_grid_config": (
            "configs/"
            "countdown_e8_oracle_offline_v2_linear_c_extension_0p5b.yaml"
        ),
        "parameter_points": C_EXTENSION_PARAMETER_POINTS,
        "seed_offsets": SEED_OFFSETS,
        "expected_points": 8,
        "expected_cells": 16,
        "requires_positive_only": False,
        "previous_scan_experiment": ROUND1_EXPERIMENT_ID,
    },
    TAU_EXPERIMENT_ID: {
        "experiment_id": TAU_EXPERIMENT_ID,
        "version": "0.4.0-dev-code-first-tau-curve",
        "default_grid_config": (
            "configs/countdown_e8_oracle_offline_v2_paper_aligned_tau_curve_0p5b.yaml"
        ),
        "parameter_points": TAU_BASE_POINTS,
        "tau_points": TAU_PARAMETER_POINTS,
        "seed_offsets": (4000,),
        "expected_points": 32,
        "expected_cells": 32,
        "requires_positive_only": False,
        "previous_scan_experiment": C_EXTENSION_EXPERIMENT_ID,
        "weight": "alpha*exp(-c*max(u-tau,0))",
    },
    TAU_C_RANGE_EXTENSION_EXPERIMENT_ID: {
        "experiment_id": TAU_C_RANGE_EXTENSION_EXPERIMENT_ID,
        "version": "0.5.0-dev-code-first-tau-c-range-extension",
        "default_grid_config": (
            "configs/countdown_e8_oracle_offline_v2_"
            "paper_aligned_tau_c_range_extension_0p5b.yaml"
        ),
        "parameter_points": TAU_C_RANGE_EXTENSION_BASE_POINTS,
        "tau_points": TAU_C_RANGE_EXTENSION_PARAMETER_POINTS,
        "seed_offsets": (4000,),
        "expected_points": 32,
        "expected_cells": 32,
        "requires_positive_only": False,
        "previous_scan_experiment": TAU_EXPERIMENT_ID,
        "weight": "alpha*exp(-c*max(u-tau,0))",
    },
}

EXPERIMENT_ID = ROUND1_EXPERIMENT_ID
VERSION = str(_PROFILES[ROUND1_EXPERIMENT_ID]["version"])
DEFAULT_GRID_CONFIG = str(
    _PROFILES[ROUND1_EXPERIMENT_ID]["default_grid_config"]
)
PARAMETER_POINTS = ROUND1_PARAMETER_POINTS
EXPECTED_POINTS = 16
EXPECTED_CELLS = 32

_ACTIVE_TAU = 0.0


class Cell(_base.Cell):
    def __init__(self, alpha: float, c: float, seed_offset: int, tau: float | None = None):
        super().__init__(alpha=alpha, c=c, seed_offset=seed_offset)
        object.__setattr__(self, "tau", _ACTIVE_TAU if tau is None else float(tau))

    @property
    def name(self) -> str:
        base = super().name
        tag = f"_tau{_base._number_tag(self.tau)}" if self.tau != 0.0 else ""
        return base.replace(f"_seed{self.seed_offset}", f"{tag}_seed{self.seed_offset}")


def set_active_tau(value: float) -> None:
    global _ACTIVE_TAU
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("tau must be finite and non-negative")
    _ACTIVE_TAU = value


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
    effective_u = (u - _ACTIVE_TAU).clamp_min(0.0)
    return (float(alpha) * torch.exp(-float(c) * effective_u)).detach()


_PATCH_KEYS = (
    "Cell",
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
_PROFILE_KEYS = (
    "EXPERIMENT_ID",
    "VERSION",
    "DEFAULT_GRID_CONFIG",
    "PARAMETER_POINTS",
    "SEED_OFFSETS",
    "EXPECTED_POINTS",
    "EXPECTED_CELLS",
)


def _profile_for_config(config: Mapping[str, Any]) -> dict[str, Any]:
    experiment_id = str(config.get("experiment_id") or "")
    try:
        return _PROFILES[experiment_id]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported paper-aligned scan experiment_id: {experiment_id}"
        ) from exc


def _set_profile(profile: Mapping[str, Any]) -> None:
    global EXPERIMENT_ID
    global VERSION
    global DEFAULT_GRID_CONFIG
    global PARAMETER_POINTS
    global SEED_OFFSETS
    global EXPECTED_POINTS
    global EXPECTED_CELLS

    EXPERIMENT_ID = str(profile["experiment_id"])
    VERSION = str(profile["version"])
    DEFAULT_GRID_CONFIG = str(profile["default_grid_config"])
    PARAMETER_POINTS = tuple(profile["parameter_points"])
    SEED_OFFSETS = tuple(int(value) for value in profile["seed_offsets"])
    EXPECTED_POINTS = int(profile["expected_points"])
    EXPECTED_CELLS = int(profile["expected_cells"])


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
    config = load_yaml(grid_config)
    profile = _profile_for_config(config)
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
        "experiment_id": str(profile["experiment_id"]),
        "version": str(profile["version"]),
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
            "tau": float(getattr(cell, "tau", 0.0)),
            "seed_offset": cell.seed_offset,
        },
        "smoke": bool(smoke),
        "test_data_used": False,
    }


def _apply() -> None:
    values = {
        "Cell": Cell,
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
def activated(profile: Mapping[str, Any] | None = None) -> Iterator[None]:
    """Temporarily activate one profile without polluting other tests."""
    previous_base = {name: getattr(_base, name) for name in _PATCH_KEYS}
    previous_profile = {name: globals()[name] for name in _PROFILE_KEYS}
    if profile is not None:
        _set_profile(profile)
    _apply()
    try:
        yield
    finally:
        for name, value in previous_profile.items():
            globals()[name] = value
        for name, value in previous_base.items():
            setattr(_base, name, value)


def activate(profile: Mapping[str, Any] | None = None) -> None:
    """Permanently activate one profile in a dedicated runtime process."""
    if profile is not None:
        _set_profile(profile)
    _apply()


def activate_for_grid_config(path: str | Path) -> None:
    """Select and activate the frozen profile declared by ``path``."""
    config = load_yaml(path)
    activate(_profile_for_config(config))
    set_active_tau(float(config["execution"]["liveness"].get("representative_tau", 0.0)))


def validate_grid_config(config: Mapping[str, Any]) -> None:
    profile = _profile_for_config(config)
    predecessor_compatible = copy.deepcopy(config)
    predecessor_compatible["remoteness"]["weight"] = "alpha*exp(-c*u^2)"
    with activated(profile):
        _BASE_VALIDATE_GRID_CONFIG(predecessor_compatible)
    expected_weight = str(profile.get("weight", "alpha*exp(-c*u)"))
    if config["remoteness"].get("weight") != expected_weight:
        raise ValueError(f"The paper-aligned weight must be {expected_weight}")

    points = tuple(
        (float(item["alpha"]), float(item["c"]))
        for item in config["sweep"]["parameter_points"]
    )
    sweep = config["sweep"]
    tau_points = profile.get("tau_points")
    if tau_points is not None and tuple(
        (float(item["alpha"]), float(item["c"]), float(item["tau"]))
        for item in sweep["parameter_points"]
    ) != tuple(tau_points):
        raise ValueError("The frozen tau response grid changed")
    if profile["requires_positive_only"]:
        if sum(alpha == 0.0 for alpha, _ in points) != 1:
            raise ValueError("Exactly one Positive-only point is required")
        if sweep.get("positive_only_same_seed_control") is not True:
            raise ValueError("positive_only_same_seed_control must remain true")
        if sweep.get("historical_positive_only_reference_only") is not False:
            raise ValueError("Round 1 must use its same-seed Positive-only cells")
    else:
        if any(alpha != 1.0 for alpha, _ in points):
            raise ValueError("The extension round must keep alpha fixed at 1")
        if any(coefficient <= 0.0 for _, coefficient in points):
            raise ValueError("The extension round contains only positive c values")
        if sweep.get("positive_only_same_seed_control") is not False:
            raise ValueError("The extension round must not rerun Positive-only")
        if sweep.get("historical_positive_only_reference_only") is not True:
            raise ValueError("The extension round must reuse the historical reference")
        if sweep.get("previous_scan_experiment") != profile["previous_scan_experiment"]:
            raise ValueError("The extension predecessor experiment changed")
        reference = config.get("historical_reference", {})
        if reference.get("source_experiment") != ROUND1_EXPERIMENT_ID:
            raise ValueError("Historical reference experiment mismatch")
        if (
            reference.get("source_run_id")
            != "E8_PAPER_ALIGNED_LINEAR_SCAN_20260716_01"
        ):
            raise ValueError("Historical reference run_id mismatch")
        if reference.get("result_manifest_sha256") != ROUND1_RESULT_MANIFEST_SHA256:
            raise ValueError("Historical result manifest mismatch")
        if reference.get("positive_only_rerun_in_this_round") is not False:
            raise ValueError("Positive-only must not be rerun in the extension")
        if tuple(reference.get("seed_offsets", ())) != SEED_OFFSETS:
            raise ValueError("Historical reference seed offsets changed")

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
    profile = _profile_for_config(config)
    points = tuple(
        (float(item["alpha"]), float(item["c"]))
        for item in config["sweep"]["parameter_points"]
    )
    expected = tuple(profile["parameter_points"])
    unique_required = profile.get("tau_points") is None
    if points != expected or unique_required and len(set(points)) != len(expected):
        raise AssertionError(
            f"Paper-aligned scan must produce {profile['expected_points']} unique points"
        )
    return points


def build_cells(config: Mapping[str, Any]) -> tuple[Cell, ...]:
    parameter_points(config)
    profile = _profile_for_config(config)
    seed_offsets = tuple(int(value) for value in profile["seed_offsets"])
    cells = tuple(
        Cell(
            alpha=float(item["alpha"]),
            c=float(item["c"]),
            seed_offset=seed_offset,
            tau=float(item.get("tau", 0.0)),
        )
        for item in config["sweep"]["parameter_points"]
        for seed_offset in seed_offsets
    )
    expected_cells = int(profile["expected_cells"])
    if (
        len(cells) != expected_cells
        or len({cell.name for cell in cells}) != expected_cells
    ):
        raise AssertionError(
            f"Paper-aligned scan must produce {expected_cells} unique cells"
        )
    return cells


def identity_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)
