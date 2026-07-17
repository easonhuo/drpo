"""Shared negative-gradient control functions."""

from .budget import gradient_l2_norm, scale_to_match_norm
from .selection import far_mask, near_mask
from .weights import (
    TaperFamily,
    normalized_excess_surprisal,
    point_retention_coefficient,
    surprisal_distance,
    taper_weight,
)

__all__ = [
    "TaperFamily",
    "far_mask",
    "gradient_l2_norm",
    "near_mask",
    "normalized_excess_surprisal",
    "point_retention_coefficient",
    "scale_to_match_norm",
    "surprisal_distance",
    "taper_weight",
]
