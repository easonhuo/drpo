"""Manuscript-consistent squared-remoteness exponential taper for E7.

The canonical E7 adapter deliberately exposes standardized RMS distance ``d`` so
reciprocal-linear and reciprocal-quadratic branches can choose their own powers.
For a Gaussian policy, learner-relative negative log probability is proportional
to squared standardized distance.  This module therefore changes only the E7
exponential shape from ``exp(-c * (d / r))`` to
``exp(-c * (d / r) ** 2)`` inside an explicit context manager.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import torch

from drpo import e7_canonical_injection as canonical_injection


FORMULA = "w(d)=w(0)*exp(-c*(d/2)^2)"


def squared_exponential_factor(
    distance: torch.Tensor,
    *,
    coefficient: float,
    reference_distance: float,
) -> torch.Tensor:
    """Return the detached squared-remoteness exponential shape factor.

    ``distance`` is the standardized RMS distance produced by the canonical E7
    geometry.  The exponent is clamped only for finite numerical evaluation; the
    mathematical factor remains in ``(0, 1]``.
    """

    if coefficient < 0.0:
        raise ValueError("coefficient must be non-negative")
    if reference_distance <= 0.0:
        raise ValueError("reference_distance must be positive")
    if not bool(torch.isfinite(distance).all()):
        raise FloatingPointError("non-finite distance in squared EXP kernel")
    u = distance / float(reference_distance)
    exponent = torch.clamp(
        -float(coefficient) * u.square(),
        min=-40.0,
        max=0.0,
    )
    factor = torch.exp(exponent)
    if not bool(torch.isfinite(factor).all()):
        raise FloatingPointError("non-finite squared EXP factor")
    return factor


@contextlib.contextmanager
def install_squared_exponential_kernel() -> Iterator[None]:
    """Temporarily replace only the canonical exponential taper shape.

    ``controlled_advantage`` in both the A2C and PPO adapters resolves
    ``canonical_injection.taper_factor`` at call time.  Patching this single
    function therefore preserves the canonical actor, critic, optimizer,
    advantage, and distance implementations while changing only the exponential
    remoteness power.  The original function is restored even when training
    raises.
    """

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
        )

    canonical_injection.taper_factor = patched
    try:
        yield
    finally:
        canonical_injection.taper_factor = original
