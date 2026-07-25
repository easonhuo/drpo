from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import torch

EXPERIMENT_ID = "EXT-H-E7-SQEXP-GAE-HOPPER-REPLAY-POSBASE-DRPO-CGRID-500K-01"
PROFILE_ID = "hopper_replay_posbase_drpo_cgrid_500k_v1"
EXPECTED_STEPS = 500_000
EXPECTED_BATCH = 512
EXPECTED_LR = 3.0e-4
DIAGNOSTIC_INTERVAL = 1_000
REFERENCE_DISTANCE = 2.0
LOG_STD_LOWER_BOUND = -20.0
LOG_STD_UPPER_BOUND = 5.0
ALLOWED_BASELINES = {0.0, 4.0, 5.0, 6.0}
ALLOWED_SCALES = {0.06, 0.08, 0.10, 0.125}
ALLOWED_METHODS = {"positive_only", "thresholded_exponential"}


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _optimizer_lrs(optimizer: Any) -> list[float]:
    groups = getattr(optimizer, "param_groups", None)
    if not isinstance(groups, list) or not groups:
        raise RuntimeError("canonical optimizer has no parameter groups")
    values = [float(group["lr"]) for group in groups]
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        raise RuntimeError("canonical optimizer has invalid learning rates")
    return values


def _tensor_norm(tensors: list[torch.Tensor]) -> float:
    total = 0.0
    for tensor in tensors:
        value = tensor.detach().float()
        total += float(value.square().sum().cpu())
    return math.sqrt(total)


def _snapshot_stats(
    raw: torch.Tensor,
    baseline: float,
    *,
    provider_update: int,
    include_quantiles: bool = False,
) -> dict[str, Any]:
    values = raw.detach().float().cpu().reshape(-1)
    shifted = values - baseline
    if values.numel() == 0 or shifted.numel() != values.numel():
        raise RuntimeError("advantage shift changed sample cardinality")
    positive = shifted > 0
    negative = shifted < 0
    raw_negative = values < 0
    positive_count = int(positive.sum())
    negative_count = int(negative.sum())
    raw_std = float(values.std(unbiased=False))
    payload: dict[str, Any] = {
        "provider_update": int(provider_update),
        "samples": int(values.numel()),
        "raw_mean": float(values.mean()),
        "raw_std": raw_std,
        "raw_positive_fraction": float((values > 0).float().mean()),
        "raw_negative_fraction": float(raw_negative.float().mean()),
        "raw_negative_abs_mass": float(values[raw_negative].abs().sum()),
        "shifted_mean": float(shifted.mean()),
        "shifted_std": float(shifted.std(unbiased=False)),
        "effective_positive_fraction": float(positive.float().mean()),
        "effective_negative_fraction": float(negative.float().mean()),
        "effective_positive_count": positive_count,
        "effective_negative_count": negative_count,
        "shifted_positive_mass": float(shifted[positive].sum()),
        "shifted_negative_abs_mass": float(shifted[negative].abs().sum()),
        "crossed_positive_to_nonpositive_fraction": float(
            ((values > 0) & (shifted <= 0)).float().mean()
        ),
        "baseline_over_raw_std": None if raw_std <= 0.0 else float(baseline / raw_std),
        "selected_raw_advantage_mean": (
            None if positive_count == 0 else float(values[positive].mean())
        ),
        "selected_shifted_advantage_mean": (
            None if positive_count == 0 else float(shifted[positive].mean())
        ),
    }
    if include_quantiles:
        quantiles = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], dtype=values.dtype)
        raw_q = torch.quantile(values, quantiles)
