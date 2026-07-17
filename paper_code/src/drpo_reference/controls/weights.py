"""Canonical paper-facing remoteness coordinates and taper functions."""

from __future__ import annotations

import math
from enum import Enum

import torch


class TaperFamily(str, Enum):
    """Unambiguous taper families defined on a non-negative distance coordinate."""

    POSITIVE_ONLY = "positive_only"
    UNCONTROLLED = "uncontrolled"
    GLOBAL = "global"
    RECIPROCAL_LINEAR = "reciprocal_linear"
    RECIPROCAL_QUADRATIC = "reciprocal_quadratic"
    EXPONENTIAL_LINEAR = "exponential_linear"
    EXPONENTIAL_QUADRATIC = "exponential_quadratic"


_FAMILY_ALIASES = {
    "positive_only": TaperFamily.POSITIVE_ONLY,
    "uncontrolled": TaperFamily.UNCONTROLLED,
    "uncontrolled_negative": TaperFamily.UNCONTROLLED,
    "unweighted": TaperFamily.UNCONTROLLED,
    "global": TaperFamily.GLOBAL,
    "global_matched": TaperFamily.GLOBAL,
    "reciprocal_linear": TaperFamily.RECIPROCAL_LINEAR,
    "reciprocal_linear_distance": TaperFamily.RECIPROCAL_LINEAR,
    "reciprocal_quadratic": TaperFamily.RECIPROCAL_QUADRATIC,
    "reciprocal_quadratic_distance": TaperFamily.RECIPROCAL_QUADRATIC,
    "exponential": TaperFamily.EXPONENTIAL_LINEAR,
    "exponential_linear": TaperFamily.EXPONENTIAL_LINEAR,
    "squared_distance_exponential": TaperFamily.EXPONENTIAL_QUADRATIC,
    "exponential_quadratic": TaperFamily.EXPONENTIAL_QUADRATIC,
    "exponential_quadratic_distance": TaperFamily.EXPONENTIAL_QUADRATIC,
}


def _coerce_family(family: TaperFamily | str) -> TaperFamily:
    if isinstance(family, TaperFamily):
        return family
    try:
        return _FAMILY_ALIASES[str(family)]
    except KeyError as exc:
        raise ValueError(f"unknown taper family: {family}") from exc


def _validate_distance(distance: torch.Tensor) -> torch.Tensor:
    if not torch.is_floating_point(distance):
        distance = distance.to(dtype=torch.get_default_dtype())
    if not bool(torch.isfinite(distance).all()):
        raise ValueError("distance must be finite")
    if bool((distance < 0).any()):
        raise ValueError("distance must be non-negative")
    return distance


def normalized_excess_surprisal(
    log_probability: torch.Tensor,
    *,
    threshold: float,
    scale: float,
    detach: bool = True,
) -> torch.Tensor:
    """Return ``relu((-log p - threshold) / scale)``.

    The default detaches learner-relative surprisal because the paper controls
    use it as a sample weight rather than as an additional differentiable loss.
    """

    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be finite and positive")
    value = log_probability.detach() if detach else log_probability
    if not bool(torch.isfinite(value).all()):
        raise ValueError("log_probability must be finite")
    return torch.relu((-value - float(threshold)) / float(scale))


def surprisal_distance(
    log_probability: torch.Tensor,
    *,
    threshold: float,
    scale: float,
    detach: bool = True,
) -> torch.Tensor:
    """Return the distance coordinate ``sqrt(normalized excess surprisal)``."""

    normalized = normalized_excess_surprisal(
        log_probability,
        threshold=threshold,
        scale=scale,
        detach=detach,
    )
    return torch.sqrt(normalized)


def point_retention_coefficient(
    family: TaperFamily | str,
    *,
    retention: float,
    reference_distance: float = 1.0,
) -> float:
    """Solve the coefficient that gives ``w(reference_distance)=retention``."""

    resolved = _coerce_family(family)
    if not math.isfinite(retention) or not 0.0 < retention <= 1.0:
        raise ValueError("retention must lie in (0, 1]")
    if not math.isfinite(reference_distance) or reference_distance <= 0.0:
        raise ValueError("reference_distance must be finite and positive")
    if resolved is TaperFamily.GLOBAL:
        return float(retention)
    if resolved in {TaperFamily.POSITIVE_ONLY, TaperFamily.UNCONTROLLED}:
        raise ValueError(f"{resolved.value} has no tunable point-retention coefficient")
    if resolved is TaperFamily.RECIPROCAL_LINEAR:
        return (1.0 / retention - 1.0) / reference_distance
    if resolved is TaperFamily.RECIPROCAL_QUADRATIC:
        return (1.0 / retention - 1.0) / (reference_distance**2)
    if resolved is TaperFamily.EXPONENTIAL_LINEAR:
        return -math.log(retention) / reference_distance
    if resolved is TaperFamily.EXPONENTIAL_QUADRATIC:
        return -math.log(retention) / (reference_distance**2)
    raise AssertionError("unreachable")


def taper_weight(
    distance: torch.Tensor,
    *,
    family: TaperFamily | str,
    coefficient: float = 1.0,
    detach_distance: bool = True,
) -> torch.Tensor:
    """Evaluate one taper on a non-negative distance coordinate.

    Linear and quadratic names refer to the power of ``distance`` used by the
    taper. Callers must choose the scientific coordinate explicitly: C-U1 uses
    standardized action distance, while D-U1/Countdown may use
    ``sqrt(normalized excess surprisal)`` so that quadratic-distance tapers are
    linear in normalized excess surprisal.
    """

    resolved = _coerce_family(family)
    value = distance.detach() if detach_distance else distance
    value = _validate_distance(value)
    if not math.isfinite(coefficient) or coefficient < 0.0:
        raise ValueError("coefficient must be finite and non-negative")
    if resolved is TaperFamily.POSITIVE_ONLY:
        return torch.zeros_like(value)
    if resolved is TaperFamily.UNCONTROLLED:
        return torch.ones_like(value)
    if resolved is TaperFamily.GLOBAL:
        return torch.full_like(value, float(coefficient))
    if resolved is TaperFamily.RECIPROCAL_LINEAR:
        return 1.0 / (1.0 + float(coefficient) * value)
    if resolved is TaperFamily.RECIPROCAL_QUADRATIC:
        return 1.0 / (1.0 + float(coefficient) * value.square())
    if resolved is TaperFamily.EXPONENTIAL_LINEAR:
        return torch.exp(-float(coefficient) * value)
    if resolved is TaperFamily.EXPONENTIAL_QUADRATIC:
        return torch.exp(-float(coefficient) * value.square())
    raise AssertionError("unreachable")
