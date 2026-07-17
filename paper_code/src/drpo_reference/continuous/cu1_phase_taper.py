"""Phase, control, and taper C-U1 public-stage execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch

from .cu1_artifacts import normalize_events, prepare_seed, write_run
from .cu1_control import run_far_pressure_control
from .cu1_phase import (
    analytic_positive_sigma,
    evaluation_from_geometry,
    run_phase_scan,
)
from .cu1_public_protocol import CU1Protocols
from .cu1_taper import config_name, method_configs, run_taper_method

CONTROL_METHODS = (
    "uncontrolled_all",
    "far_cap",
    "budget_matched_global",
)


def run_phase_rows(
    *,
    seeds: Sequence[int],
    root: Path,
    protocols: CU1Protocols,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scan_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    sigma = analytic_positive_sigma(protocols.core)
    positive_ceiling_reward = evaluation_from_geometry(
        protocols.core.gap_to_unseen_optimum,
        protocols.core,
    )

    for seed in seeds:
        positive = prepare_seed(seed, root, protocols, device)
        branch_specs = (
            (
                "fixed_variance",
                sigma,
                protocols.phase.fixed_alphas,
            ),
            (
                "learnable_variance",
                None,
                protocols.phase.learnable_alphas,
            ),
        )
        for branch, fixed_sigma, alphas in branch_specs:
            for alpha in alphas:
                run = run_phase_scan(
                    seed=seed,
                    initialization_state=positive.initialization_state,
                    environment=positive.environment,
                    protocol=protocols.core,
                    positive_training=protocols.positive,
                    phase=protocols.phase,
                    alpha=alpha,
                    fixed_sigma=fixed_sigma,
                    branch=branch,
                )
                summary = dict(run.summary)
                summary["steps_completed"] = max(
                    (int(row["step"]) for row in run.trajectory),
                    default=0,
                )
                summary["task_performance_collapse_event"] = bool(
                    float(summary["reward"])
                    < protocols.core.task_failure_retention
                    * positive_ceiling_reward
                )
                summary["support_or_variance_boundary_event"] = bool(
                    summary.get("support_boundary_onset") is not None
                )
                summary["nan_inf_numerical_event"] = bool(
                    summary.get("stop_reason") == "non_finite_parameter"
                )
                summary["environment_invalid_event"] = False
                scan_rows.append(summary)
                write_run(
                    root,
                    Path("phase")
                    / branch
                    / f"alpha_{float(alpha):.2f}"
                    / f"seed_{seed}",
                    summary=summary,
                    trajectory=run.trajectory,
                    state_dict=run.actor.state_dict(),
                )

        for method in CONTROL_METHODS:
            run = run_far_pressure_control(
                seed=seed,
                initialization_state=positive.initialization_state,
                environment=positive.environment,
                protocol=protocols.core,
                positive_training=protocols.positive,
                control=protocols.control,
                method=method,
            )
            summary = normalize_events(dict(run.summary))
            control_rows.append(summary)
            write_run(
                root,
                Path("phase") / "controls" / method / f"seed_{seed}",
                summary=summary,
                trajectory=run.trajectory,
                state_dict=run.actor.state_dict(),
            )
    return scan_rows, control_rows


def run_taper_rows(
    *,
    seeds: Sequence[int],
    root: Path,
    protocols: CU1Protocols,
    device: torch.device,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        positive = prepare_seed(seed, root, protocols, device)
        for family, retention in method_configs(protocols.taper):
            run = run_taper_method(
                seed=seed,
                initialization_state=positive.initialization_state,
                environment=positive.environment,
                protocol=protocols.core,
                positive_training=protocols.positive,
                taper=protocols.taper,
                family=family,
                retention=retention,
            )
            summary = normalize_events(dict(run.summary))
            rows.append(summary)
            write_run(
                root,
                Path("taper")
                / config_name(family, retention)
                / f"seed_{seed}",
                summary=summary,
                trajectory=run.trajectory,
                diagnostics=run.diagnostics,
                state_dict=run.actor.state_dict(),
            )
    return rows
