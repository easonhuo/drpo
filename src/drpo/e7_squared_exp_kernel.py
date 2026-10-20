"""Squared-remoteness exponential kernel for the canonical E7 adapters."""

from __future__ import annotations

import contextlib
import math
from collections.abc import Iterator

import torch

from drpo import e7_canonical_injection as canonical_injection


FORMULA = "w(d)=w(0)*exp(-c*(d/2)^2)"
THRESHOLDED_FORMULA = (
    "w(D)=w(0)*exp(-taper_lambda*relu((D-remoteness_threshold)/remoteness_scale))"
)


def squared_exponential_factor(
    distance: torch.Tensor,
    *,
    coefficient: float,
    reference_distance: float,
    remoteness_threshold: float = 0.0,
) -> torch.Tensor:
    """Return ``exp(-coefficient * relu(D - tau))`` for ``D=(d/r)^2``."""

    if not math.isfinite(float(coefficient)) or coefficient < 0.0:
        raise ValueError("coefficient must be finite and non-negative")
    if not math.isfinite(float(reference_distance)) or reference_distance <= 0.0:
        raise ValueError("reference_distance must be finite and positive")
    if (
        not math.isfinite(float(remoteness_threshold))
        or remoteness_threshold < 0.0
    ):
        raise ValueError("remoteness_threshold must be finite and non-negative")
    if not bool(torch.isfinite(distance).all()):
        raise FloatingPointError("non-finite distance in squared EXP kernel")

    remoteness = (distance / float(reference_distance)).square()
    excess = torch.clamp_min(remoteness - float(remoteness_threshold), 0.0)
    exponent = torch.clamp(-float(coefficient) * excess, min=-40.0, max=0.0)
    factor = torch.exp(exponent)
    if not bool(torch.isfinite(factor).all()):
        raise FloatingPointError("non-finite squared EXP factor")
    return factor


@contextlib.contextmanager
def install_squared_exponential_kernel(
    *,
    remoteness_threshold: float = 0.0,
) -> Iterator[None]:
    """Temporarily change only the canonical exponential taper shape."""

    original = canonical_injection.taper_factor

    def patched(
        distance: torch.Tensor,
        control: canonical_injection.NegativeControl,
    ) -> torch.Tensor:
        if control.method != "exponential":
            return original(distance, control)
        return squared_exponential_factor(
            distance,
            coefficient=control.exponential_coefficient,
            reference_distance=control.reference_distance,
            remoteness_threshold=remoteness_threshold,
        )

    canonical_injection.taper_factor = patched
    try:
        yield
    finally:
        canonical_injection.taper_factor = original
