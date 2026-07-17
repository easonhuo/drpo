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
from .du1_public import run_du1, smoke_terminal_protocol
from .du1_reports import (
    mechanism_report,
    paired_metric_effect,
    taper_report,
)
from .du1_suite import (
    SeedBundle,
    aggregate,
    assign_task_collapse,
    build_terminal_audit,
    paired_effect,
    run_seed_bundle,
)
from .du1_training import (
    DU1TerminalProtocol,
    MethodRun,
    SharedStart,
    build_shared_start,
    legacy_run_config,
    run_method,
    terminal_classification,
)

__all__ = [
    "CELL_NAMES",
    "FORMAL_METHODS",
    "HISTORICAL_EXCLUDED_METHODS",
    "CartesianPolicy",
    "CartesianSemanticEnvironment",
    "DU1Protocol",
    "DU1TerminalProtocol",
    "MethodRun",
    "MethodSpec",
    "SeedBundle",
    "SharedStart",
    "active_cell_loss",
    "aggregate",
    "assign_task_collapse",
    "batch_indices",
    "build_shared_start",
    "build_terminal_audit",
    "cache_reference_directions",
    "cell_log_probs",
    "coordinate_calibration",
    "evaluate",
    "gather_log_probs",
    "legacy_run_config",
    "mechanism_report",
    "method_specs",
    "negative_loss_and_diagnostics",
    "normalized_excess_surprisal",
    "paired_effect",
    "paired_metric_effect",
    "policy_geometry_audit",
    "rarity_logit_anchor_loss",
    "run_du1",
    "run_method",
    "run_seed_bundle",
    "smoke_protocol",
    "smoke_terminal_protocol",
    "taper_coefficients",
    "taper_report",
    "taper_weight",
    "terminal_classification",
    "trainable_parameters",
]
