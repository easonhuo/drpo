"""Hard near/far selection on detached remoteness coordinates."""

from __future__ import annotations

import math

import torch


def _validated(distance: torch.Tensor, threshold: float) -> torch.Tensor:
    value = distance.detach()
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError("threshold must be finite and non-negative")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("distance must be finite")
    if bool((value < 0).any()):
        raise ValueError("distance must be non-negative")
    return value


def near_mask(distance: torch.Tensor, *, threshold: float) -> torch.Tensor:
    """Return the legacy-compatible near mask ``distance <= threshold``."""

    return _validated(distance, threshold) <= float(threshold)


def far_mask(distance: torch.Tensor, *, threshold: float) -> torch.Tensor:
    """Return the exact complement of :func:`near_mask`."""

    return _validated(distance, threshold) > float(threshold)
