"""Public experiment entry points for the paper-facing reference package."""

from .hopper import (
    CanonicalCriticContext,
    HopperExecutionPlan,
    aggregate_seed_summaries,
    build_root_terminal_audit,
    flatten_seed_summary,
    prepare_canonical_critic_context,
    resolve_hopper_execution,
    run_hopper,
    validate_dataset_identity,
)

__all__ = [
    "CanonicalCriticContext",
    "HopperExecutionPlan",
    "aggregate_seed_summaries",
    "build_root_terminal_audit",
    "flatten_seed_summary",
    "prepare_canonical_critic_context",
    "resolve_hopper_execution",
    "run_hopper",
    "validate_dataset_identity",
]
