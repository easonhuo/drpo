"""Categorical controlled-mechanism experiment components."""

from .du1_controls import (
    active_cell_loss,
    coordinate_calibration,
    negative_loss_and_diagnostics,
    normalized_excess_surprisal,
    rarity_logit_anchor_loss,
    taper_coefficients,
    taper_weight,
)
from .du1_environment import CartesianSemanticEnvironment
from .du1_metrics import evaluate, policy_geometry_audit
from .du1_policy import (
    CartesianPolicy,
    batch_indices,
    cache_reference_directions,
    cell_log_probs,
    gather_log_probs,
    trainable_parameters,
)
from .du1_protocol import (
    CELL_NAMES,
    FORMAL_METHODS,
    HISTORICAL_EXCLUDED_METHODS,
    DU1Protocol,
    MethodSpec,
    method_specs,
    smoke_protocol,
)

__all__ = [
    "CELL_NAMES",
    "FORMAL_METHODS",
    "HISTORICAL_EXCLUDED_METHODS",
    "CartesianPolicy",
    "CartesianSemanticEnvironment",
    "DU1Protocol",
    "MethodSpec",
    "active_cell_loss",
    "batch_indices",
    "cache_reference_directions",
    "cell_log_probs",
    "coordinate_calibration",
    "evaluate",
    "gather_log_probs",
    "method_specs",
    "negative_loss_and_diagnostics",
    "normalized_excess_surprisal",
    "policy_geometry_audit",
    "rarity_logit_anchor_loss",
    "smoke_protocol",
    "taper_coefficients",
    "taper_weight",
    "trainable_parameters",
]
