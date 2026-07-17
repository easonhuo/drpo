"""Artifact writing and event normalization for the C-U1 public runner."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch

from drpo_reference.common import atomic_json, write_csv

from .cu1 import audit_environment
from .cu1_public_protocol import CU1Protocols
from .cu1_training import PositiveRun, train_positive


def normalize_events(summary: dict[str, Any]) -> dict[str, Any]:
    """Add one stable public event schema without deleting native diagnostics."""

    summary.setdefault(
        "task_performance_collapse_event",
        bool(
            summary.get(
                "task_performance_collapse",
                summary.get("task_failure_onset") is not None,
            )
        ),
    )
    summary.setdefault(
        "support_or_variance_boundary_event",
        bool(
            summary.get(
                "support_or_probability_boundary",
                summary.get("support_boundary_onset") is not None,
            )
        ),
    )
    summary.setdefault(
        "nan_inf_numerical_event",
        bool(
            summary.get(
                "nan_inf_numerical_failure",
                not bool(summary.get("finite_parameters", True)),
            )
        ),
    )
    summary.setdefault(
        "environment_invalid_event",
        bool(summary.get("environment_invalid", False)),
    )
    return summary


def write_run(
    root: Path,
    relative: Path,
    *,
    summary: dict[str, Any],
    trajectory: Iterable[dict[str, Any]] = (),
    diagnostics: Iterable[dict[str, Any]] = (),
    state_dict: dict[str, torch.Tensor] | None = None,
) -> None:
    target = root / relative
    atomic_json(target.with_suffix(".json"), summary)
    trajectory_rows = list(trajectory)
    diagnostic_rows = list(diagnostics)
    if trajectory_rows:
        write_csv(
            target.with_name(target.name + "_trajectory.csv"),
            trajectory_rows,
        )
    if diagnostic_rows:
        write_csv(
            target.with_name(target.name + "_diagnostics.csv"),
            diagnostic_rows,
        )
    if state_dict is not None:
        checkpoint = target.with_suffix(".pt")
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state_dict, checkpoint)


def prepare_seed(
    seed: int,
    root: Path,
    protocols: CU1Protocols,
    device: torch.device,
) -> PositiveRun:
    """Create the exact positive preparation used by one downstream seed."""

    run = train_positive(
        seed=seed,
        protocol=protocols.core,
        training=protocols.positive,
        device=device,
    )
    environment_audit = audit_environment(run.environment, protocols.core)
    if not environment_audit["passed"]:
        raise RuntimeError(f"C-U1 environment audit failed for seed {seed}")

    preparation = root / "preparation"
    checkpoint_dir = preparation / "positive_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        run.initialization_state,
        checkpoint_dir
        / (
            f"seed_{seed}_adam{protocols.positive.positive_steps}"
            "_initialization.pt"
        ),
    )
    torch.save(
        run.actor.state_dict(),
        checkpoint_dir / f"seed_{seed}_final.pt",
    )
    write_csv(
        preparation / "positive" / f"seed_{seed}_trajectory.csv",
        run.trajectory,
    )
    atomic_json(
        preparation / "positive" / f"seed_{seed}.json",
        run.summary,
    )
    atomic_json(
        preparation / "environment_audits" / f"seed_{seed}.json",
        environment_audit,
    )
    return run
