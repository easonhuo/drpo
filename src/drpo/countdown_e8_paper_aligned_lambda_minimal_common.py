#!/usr/bin/env python3
"""Thin paper-aligned adapter over the canonical E8 alpha1-c scan lineage.

The predecessor trainer, runtime scheduler, evaluator, resume logic, aggregation,
and 8-GPU x 2-slot placement are inherited unchanged.  The compatibility field
``Cell.c`` carries paper ``lambda`` only inside the predecessor call signature.
"""
from __future__ import annotations

import json
import math
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import torch

from drpo import countdown_e8_alpha1_c_scan_common as _base

EXPERIMENT_ID = "EXT-C-E8-ORACLE-OFFLINE-V2-PAPER-ALIGNED-LAMBDA-MINIMAL-0.5B-01"
VERSION = "0.2.0-dev-code-first-minimal-lineage"
DEFAULT_GRID_CONFIG = "configs/countdown_e8_paper_aligned_lambda_minimal_0p5b.yaml"
CALIBRATION_ENV = "DRPO_E8_PAPER_CALIBRATION"
LAMBDA_VALUES = (
    0.0,
    0.051293294,
    0.105360516,
    0.162518929,
    0.223143551,
    0.287682072,
    0.430782916,
    0.693147181,
    0.916290732,
    1.203972804,
    1.386294361,
    1.609437912,
    1.897119985,
    2.302585093,
    2.995732274,
)
PARAMETER_POINTS = ((0.0, 0.0),) + tuple((1.0, value) for value in LAMBDA_VALUES)
SEED_OFFSETS = (4000, 5000)
EXPECTED_POINTS = 16
EXPECTED_CELLS = 32

arena = _base.arena
sha256_file = _base.sha256_file
atomic_json = _base.atomic_json
load_yaml = _base.load_yaml
mean_unique_negative_term = _base.mean_unique_negative_term
unique_negative_expressions = _base.unique_negative_expressions
ContinuousUniqueBankDataset = _base.ContinuousUniqueBankDataset
make_continuous_unique_bank_collator = _base.make_continuous_unique_bank_collator
weight_diagnostics = _base.weight_diagnostics
git_state = _base.git_state


def _number_tag(value: float) -> str:
    return f"{value:.9g}".replace("-", "m").replace(".", "p")


@dataclass(frozen=True)
class Cell:
    alpha: float
    c: float  # predecessor compatibility alias for paper lambda
    seed_offset: int

    @property
    def lambda_value(self) -> float:
        return float(self.c)

    @property
    def method(self) -> str:
        if self.alpha == 0.0:
            return "positive_only"
        if self.c == 0.0:
            return "uncontrolled_negative"
        return "paper_aligned_exp"

    @property
    def name(self) -> str:
        return (
            f"base_{self.method}_alpha{_number_tag(self.alpha)}_"
            f"lambda{_number_tag(self.c)}_seed{self.seed_offset}"
        )


def calibration_from_surprisals(
    values: list[float], *, minimum_scale: float, minimum_active_fraction: float
) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 4 or not np.isfinite(array).all():
        raise ValueError("Calibration surprisals must be a finite vector with at least four values")
    ordered = np.sort(array)
    midpoint = ordered.size // 2
    tau = float(np.median(ordered))
    lower = float(np.median(ordered[:midpoint]))
    upper = float(np.median(ordered[midpoint:]))
    scale_c = upper - lower
    active_fraction = float(np.mean(array > tau))
    if not math.isfinite(scale_c) or scale_c < minimum_scale:
        raise ValueError("Calibrated surprisal scale is degenerate")
    if active_fraction < minimum_active_fraction:
        raise ValueError("Calibrated active-tail fraction is below the registered gate")
    return {
        "tau": tau,
        "scale_c": scale_c,
        "active_fraction": active_fraction,
        "surprisal_min": float(array.min()),
        "surprisal_max": float(array.max()),
        "surprisal_mean": float(array.mean()),
    }


def _calibration() -> tuple[dict[str, Any], Path]:
    raw = os.environ.get(CALIBRATION_ENV, "").strip()
    if not raw:
        raise RuntimeError(f"{CALIBRATION_ENV} must point to the frozen calibration JSON")
    path = Path(raw).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("experiment_id") != EXPERIMENT_ID:
        raise RuntimeError("Calibration experiment identity mismatch")
    tau = float(payload["tau"])
    scale_c = float(payload["scale_c"])
    if not math.isfinite(tau) or not math.isfinite(scale_c) or scale_c <= 0.0:
        raise RuntimeError("Calibration tau/scale_c is invalid")
    return payload, path


def continuous_remoteness(
    seq_lp: torch.Tensor, *, reference_distance: float = _base.REFERENCE_DISTANCE
) -> torch.Tensor:
    del reference_distance
    payload, _ = _calibration()
    distance = (-seq_lp.detach()).clamp_min(0.0)
    return ((distance - float(payload["tau"])) / float(payload["scale_c"])).clamp_min(0.0)


def continuous_exp_weights(
    seq_lp: torch.Tensor,
    *,
    alpha: float,
    c: float,
    reference_distance: float = _base.REFERENCE_DISTANCE,
) -> torch.Tensor:
    del reference_distance
    if not math.isfinite(alpha) or alpha < 0.0:
        raise ValueError("alpha must be finite and non-negative")
    if not math.isfinite(c) or c < 0.0:
        raise ValueError("paper lambda must be finite and non-negative")
    z = continuous_remoteness(seq_lp)
    return (float(alpha) * torch.exp(-float(c) * z)).detach()


def validate_grid_config(config: Mapping[str, Any]) -> None:
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("Paper-aligned minimal experiment_id mismatch")
    if config.get("result_status") != "pilot":
        raise ValueError("Round 1 must remain a pilot")
    if config.get("registration_state") not in {
        "dev_code_first_unregistered",
        "registered_pilot",
    }:
        raise ValueError("Invalid registration_state")
    remoteness = config.get("remoteness", {})
    if remoteness.get("weight") != "alpha*exp(-lambda*relu((D-tau)/scale_c))":
        raise ValueError("Paper-aligned weight formula changed")
    if remoteness.get("extra_square_forbidden") is not True:
        raise ValueError("Extra surprisal square must remain forbidden")
    sweep = config.get("sweep", {})
    lambdas = tuple(float(value) for value in sweep.get("lambda_values", ()))
    offsets = tuple(int(value) for value in sweep.get("seed_offsets", ()))
    if lambdas != LAMBDA_VALUES:
        raise ValueError("Round-1 lambda grid changed")
    if offsets != SEED_OFFSETS:
        raise ValueError("Round-1 development seeds changed")
    if int(sweep.get("unique_parameter_points", -1)) != EXPECTED_POINTS:
        raise ValueError("Round 1 requires 16 parameter points")
    if int(sweep.get("cells", -1)) != EXPECTED_CELLS:
        raise ValueError("Round 1 requires 32 cells")
    if float(sweep.get("fixed_alpha", -1.0)) != 1.0:
        raise ValueError("Round 1 fixes alpha=1")
    if sweep.get("positive_only_included") is not True or sweep.get("global_rerun") is not False:
        raise ValueError("Baseline matrix changed")
    training = config.get("training", {})
    required_training = {
        "steps": 1200,
        "early_stop": False,
        "eval_every": 100,
        "pass64_every": 200,
        "denominator": "unique_negative_count_per_prompt",
        "normalize_by_weight_sum": False,
        "hidden_negative_scale": False,
        "gradient_budget_matching": False,
    }
    for key, expected in required_training.items():
        if training.get(key) != expected:
            raise ValueError(f"Frozen training field changed: {key}")
    execution = config.get("execution", {})
    if list(execution.get("default_gpus", ())) != list(range(8)):
        raise ValueError("The canonical E8 pool is GPU 0-7")
    if int(execution.get("parallel_cells_per_gpu", -1)) != 2:
        raise ValueError("The canonical runtime requires two cells per GPU")
    if execution.get("runtime_scope") != "GOV-RUNTIME-E8-GPU-SLOT-HOTFIX-01":
        raise ValueError("The canonical two-slot runtime scope changed")
    if execution.get("identity_checked_resume") is not True:
        raise ValueError("Identity-checked resume is required")
    liveness = execution.get("liveness", {})
    if float(liveness.get("representative_c", -1.0)) != float(
        liveness.get("representative_lambda", -2.0)
    ):
        raise ValueError("representative_c is only the predecessor lambda alias")
    evaluation = config.get("evaluation", {})
    if evaluation.get("validation_only_during_tuning") is not True:
        raise ValueError("Tuning must remain validation-only")
    if evaluation.get("test_access_forbidden") is not True:
        raise ValueError("Test access must remain forbidden")
    if evaluation.get("primary_selection_metric") != "late_window_pass_at_8":
        raise ValueError("Primary metric must remain late-window Pass@8")


def parameter_points(config: Mapping[str, Any]) -> tuple[tuple[float, float], ...]:
    validate_grid_config(config)
    return PARAMETER_POINTS


def build_cells(config: Mapping[str, Any]) -> tuple[Cell, ...]:
    points = parameter_points(config)
    cells = tuple(
        Cell(alpha=alpha, c=lambda_value, seed_offset=seed)
        for alpha, lambda_value in points
        for seed in SEED_OFFSETS
    )
    if len(cells) != EXPECTED_CELLS or len({cell.name for cell in cells}) != EXPECTED_CELLS:
        raise AssertionError("Paper-aligned Round 1 must produce 32 unique cells")
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
    calibration, calibration_path = _calibration()
    package_dir = Path(__file__).resolve().parent
    source_paths = {
        "paper_common": Path(__file__).resolve(),
        "base_common": Path(_base.__file__).resolve(),
        "base_trainer": package_dir / "countdown_e8_alpha1_c_scan_trainer.py",
        "base_runtime": package_dir / "countdown_e8_alpha1_c_scan_runtime.py",
        "paper_runtime": package_dir / "countdown_e8_paper_aligned_lambda_minimal_runtime.py",
        "paper_auto": repo / "scripts" / "run_countdown_e8_paper_aligned_lambda_minimal_auto.py",
        "paper_calibration": repo / "scripts" / "calibrate_countdown_e8_paper_aligned_lambda.py",
    }
    missing = [name for name, path in source_paths.items() if not path.is_file()]
    if missing:
        raise RuntimeError("Missing protected paper-aligned sources: " + ", ".join(sorted(missing)))
    return {
        "experiment_id": EXPERIMENT_ID,
        "version": VERSION,
        "predecessor_commit": "929142930a3e2efaa7cafc8e4afe3866600027a5",
        "source": _base.git_state(repo),
        "source_sha256": {name: _base.sha256_file(path) for name, path in sorted(source_paths.items())},
        "model_path": str(model_path),
        "bank_sha256": _base.sha256_file(bank),
        "validation_sha256": _base.sha256_file(val),
        "base_config_sha256": _base.sha256_file(base_config),
        "grid_config_sha256": _base.sha256_file(grid_config),
        "calibration_sha256": _base.sha256_file(calibration_path),
        "tau": float(calibration["tau"]),
        "scale_c": float(calibration["scale_c"]),
        "cell": {
            "name": cell.name,
            "method": cell.method,
            "alpha": cell.alpha,
            "lambda": cell.lambda_value,
            "seed_offset": cell.seed_offset,
        },
        "smoke": bool(smoke),
        "test_data_used": False,
    }


_PATCH_KEYS = (
    "EXPERIMENT_ID",
    "VERSION",
    "DEFAULT_GRID_CONFIG",
    "PARAMETER_POINTS",
    "SEED_OFFSETS",
    "EXPECTED_POINTS",
    "EXPECTED_CELLS",
    "Cell",
    "continuous_remoteness",
    "continuous_exp_weights",
    "validate_grid_config",
    "parameter_points",
    "build_cells",
    "_identity",
)


def _apply() -> None:
    for name in _PATCH_KEYS:
        setattr(_base, name, globals()[name])


@contextmanager
def activated() -> Iterator[None]:
    previous = {name: getattr(_base, name) for name in _PATCH_KEYS}
    _apply()
    try:
        yield
    finally:
        for name, value in previous.items():
            setattr(_base, name, value)


def activate() -> None:
    _apply()
