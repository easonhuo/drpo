"""Raw-gradient L2 accounting and deterministic budget matching."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

GradientSequence = Sequence[torch.Tensor | None]


def gradient_l2_norm(gradients: GradientSequence) -> torch.Tensor:
    """Return the L2 norm of all non-``None`` gradient tensors."""

    present = [gradient.detach() for gradient in gradients if gradient is not None]
    if not present:
        return torch.zeros((), dtype=torch.float64)
    device = present[0].device
    if any(gradient.device != device for gradient in present):
        raise ValueError("all gradients must be on the same device")
    total = torch.zeros((), dtype=torch.float64, device=device)
    for gradient in present:
        total = total + gradient.to(dtype=torch.float64).square().sum()
    return torch.sqrt(total)


def scale_to_match_norm(
    target_gradients: GradientSequence,
    source_gradients: GradientSequence,
    *,
    zero_tolerance: float = 1.0e-12,
    maximum_scale: float | None = None,
) -> float:
    """Return the scalar that matches source raw-gradient L2 to target L2."""

    if not math.isfinite(zero_tolerance) or zero_tolerance < 0.0:
        raise ValueError("zero_tolerance must be finite and non-negative")
    if maximum_scale is not None and (
        not math.isfinite(maximum_scale) or maximum_scale <= 0.0
    ):
        raise ValueError("maximum_scale must be finite and positive")
    target = float(gradient_l2_norm(target_gradients).cpu())
    source = float(gradient_l2_norm(source_gradients).cpu())
    if not math.isfinite(target) or not math.isfinite(source):
        raise ValueError("gradient norms must be finite")
    if source <= zero_tolerance:
        if target <= zero_tolerance:
            return 0.0
        raise ZeroDivisionError("cannot match a non-zero target with a zero source gradient")
    scale = target / source
    if maximum_scale is not None and scale > maximum_scale:
        raise ValueError(f"required scale {scale} exceeds maximum_scale {maximum_scale}")
    return scale
