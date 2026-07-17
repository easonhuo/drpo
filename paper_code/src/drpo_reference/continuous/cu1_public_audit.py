"""Aggregation and terminal-audit rules for the C-U1 public runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from drpo_reference.common import atomic_json, write_csv
from drpo_reference.common.aggregate import aggregate_rows, audit_run_matrix

from .cu1_phase_taper import CONTROL_METHODS
from .cu1_public_protocol import CU1Protocols, EVENT_FIELDS, formal_seeds
from .cu1_taper import method_configs


def expected_identities(
    stage: str,
    seeds: Sequence[int],
    protocols: CU1Protocols,
) -> tuple[tuple[str, ...], list[tuple[Any, ...]], tuple[str, ...]]:
    if stage == "source":
        return (
            "seed",
        ), [(seed,) for seed in seeds], (
            "seed",
            "advantage_far_near_ratio",
            "output_score_far_near_ratio",
            "full_parameter_single_sample_far_near_ratio",
            "aggregate_far_near_ratio",
        )
    if stage == "causal":
        methods = protocols.causal.primary_methods + protocols.causal.appendix_methods
        expected = [
            (seed, branch, method)
            for seed in seeds
            for branch in ("fixed_variance", "learnable_variance")
            for method in methods
        ]
        return (
            "seed",
            "branch",
            "method",
        ), expected, (
            "seed",
            "branch",
            "method",
            "steps_completed",
            "stop_reason",
        )
    if stage == "phase":
        expected = [
            (seed, "fixed_variance", float(alpha))
            for seed in seeds
            for alpha in protocols.phase.fixed_alphas
        ] + [
            (seed, "learnable_variance", float(alpha))
            for seed in seeds
            for alpha in protocols.phase.learnable_alphas
        ]
        return (
            "seed",
            "branch",
            "alpha",
        ), expected, (
            "seed",
            "branch",
            "alpha",
            "state_class",
            "steps_completed",
            "stop_reason",
        )
    if stage == "taper":
        expected = [
            (seed, family, float(retention))
            for seed in seeds
            for family, retention in method_configs(protocols.taper)
        ]
        return (
            "seed",
            "family",
            "rho",
        ), expected, (
            "seed",
            "family",
            "rho",
            "steps_completed",
            "stop_reason",
        )
    raise ValueError(stage)


def aggregate_and_audit(
    *,
    stage: str,
    rows: list[dict[str, Any]],
    seeds: Sequence[int],
    root: Path,
    protocols: CU1Protocols,
    smoke: bool,
    control_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if stage == "source":
        group_keys: tuple[str, ...] = ()
        event_fields: tuple[str, ...] = ("environment_invalid_event",)
    elif stage == "causal":
        group_keys = ("branch", "method")
        event_fields = EVENT_FIELDS
    elif stage == "phase":
        group_keys = ("branch", "alpha")
        event_fields = EVENT_FIELDS
    elif stage == "taper":
        group_keys = ("family", "rho")
        event_fields = EVENT_FIELDS
    else:
        raise ValueError(stage)

    aggregate = aggregate_rows(
        rows,
        group_keys=group_keys,
        event_fields=event_fields,
    )
    write_csv(root / "aggregate" / f"{stage}.csv", aggregate)
    identity_fields, expected, required = expected_identities(
        stage,
        seeds,
        protocols,
    )
    matrix = audit_run_matrix(
        rows,
        identity_fields=identity_fields,
        expected_identities=expected,
        required_fields=required,
    )
    complete_formal_seed_set = tuple(seeds) == formal_seeds(stage, protocols)
    audit: dict[str, Any] = {
        "stage": stage,
        "smoke": smoke,
        "selected_seeds": list(seeds),
        "registered_formal_seeds": list(formal_seeds(stage, protocols)),
        "complete_formal_seed_set": complete_formal_seed_set,
        "matrix": matrix,
        "formal_evidence_allowed": bool(
            not smoke and complete_formal_seed_set and matrix["passed"]
        ),
    }

    if stage == "phase":
        if control_rows is None:
            raise ValueError("phase audit requires control rows")
        control_aggregate = aggregate_rows(
            control_rows,
            group_keys=("method",),
            event_fields=EVENT_FIELDS,
        )
        write_csv(
            root / "aggregate" / "phase_controls.csv",
            control_aggregate,
        )
        control_matrix = audit_run_matrix(
            control_rows,
            identity_fields=("seed", "method"),
            expected_identities=[
                (seed, method)
                for seed in seeds
                for method in CONTROL_METHODS
            ],
            required_fields=("seed", "method", "steps_completed"),
        )
        audit["control_matrix"] = control_matrix
        audit["formal_evidence_allowed"] = bool(
            audit["formal_evidence_allowed"] and control_matrix["passed"]
        )

    if stage == "taper":
        resolved_reasons = {
            "stable_plateau_2x_confirmed",
            "support_or_variance_boundary_event",
            "nan_inf_numerical_event",
        }
        unresolved = [
            {
                "seed": row.get("seed"),
                "family": row.get("family"),
                "rho": row.get("rho"),
                "stop_reason": row.get("stop_reason"),
            }
            for row in rows
            if row.get("stop_reason") not in resolved_reasons
        ]
        audit["terminal_resolution"] = {
            "resolved_reasons": sorted(resolved_reasons),
            "unresolved_runs": unresolved,
            "passed": not unresolved,
        }
        audit["formal_evidence_allowed"] = bool(
            audit["formal_evidence_allowed"] and not unresolved
        )

    atomic_json(root / "terminal_audit" / f"{stage}.json", audit)
    return audit
