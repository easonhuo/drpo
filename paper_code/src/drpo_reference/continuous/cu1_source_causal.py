"""Source and causal C-U1 public-stage execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch

from .cu1_artifacts import normalize_events, prepare_seed, write_run
from .cu1_mechanism import run_causal_intervention, source_diagnostic
from .cu1_phase import analytic_positive_sigma
from .cu1_public_protocol import CU1Protocols


def run_source_rows(
    *,
    seeds: Sequence[int],
    root: Path,
    protocols: CU1Protocols,
    device: torch.device,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        positive = prepare_seed(seed, root, protocols, device)
        row = dict(
            source_diagnostic(
                seed=seed,
                actor=positive.actor,
                environment=positive.environment,
                protocol=protocols.core,
                source=protocols.source,
            )
        )
        row["environment_invalid_event"] = False
        rows.append(row)
        write_run(root, Path("source") / f"seed_{seed}", summary=row)
    return rows


def run_causal_rows(
    *,
    seeds: Sequence[int],
    root: Path,
    protocols: CU1Protocols,
    device: torch.device,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    methods = protocols.causal.primary_methods + protocols.causal.appendix_methods
    branch_specs = (
        (
            "fixed_variance",
            analytic_positive_sigma(protocols.core),
            protocols.causal.fixed_alpha,
            protocols.causal.fixed_learning_rate,
            protocols.causal.fixed_steps,
        ),
        (
            "learnable_variance",
            None,
            protocols.causal.learnable_alpha,
            protocols.causal.learnable_learning_rate,
            protocols.causal.learnable_steps,
        ),
    )
    for seed in seeds:
        positive = prepare_seed(seed, root, protocols, device)
        for branch, sigma, alpha, learning_rate, steps in branch_specs:
            for method in methods:
                run = run_causal_intervention(
                    seed=seed,
                    initialization_state=positive.initialization_state,
                    environment=positive.environment,
                    protocol=protocols.core,
                    positive_training=protocols.positive,
                    method=method,
                    fixed_sigma=sigma,
                    alpha=alpha,
                    learning_rate=learning_rate,
                    steps=steps,
                    branch=branch,
                    causal=protocols.causal,
                )
                summary = normalize_events(dict(run.summary))
                rows.append(summary)
                write_run(
                    root,
                    Path("causal") / branch / method / f"seed_{seed}",
                    summary=summary,
                    trajectory=run.trajectory,
                    state_dict=run.actor.state_dict(),
                )
    return rows
