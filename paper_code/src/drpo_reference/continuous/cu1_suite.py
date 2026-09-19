"""Public runner for the four C-U1 paper experiments."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from drpo_reference.common import atomic_json

from .cu1 import CU1Protocol
from .cu1_control import CU1ControlProtocol, run_far_pressure_control
from .cu1_mechanism import (
    CU1CausalProtocol,
    CU1SourceProtocol,
    run_causal_intervention,
    source_diagnostic,
)
from .cu1_phase import CU1PhaseProtocol, analytic_positive_sigma, run_phase_scan
from .cu1_taper import CU1TaperProtocol, method_configs, run_taper_method
from .cu1_training import CU1PositiveProtocol, train_positive

STAGES = ("source", "causal", "phase", "taper")


@dataclass(frozen=True)
class CU1Protocols:
    core: CU1Protocol = field(default_factory=CU1Protocol)
    positive: CU1PositiveProtocol = field(default_factory=CU1PositiveProtocol)
    source: CU1SourceProtocol = field(default_factory=CU1SourceProtocol)
    causal: CU1CausalProtocol = field(default_factory=CU1CausalProtocol)
    phase: CU1PhaseProtocol = field(default_factory=CU1PhaseProtocol)
    control: CU1ControlProtocol = field(default_factory=CU1ControlProtocol)
    taper: CU1TaperProtocol = field(default_factory=CU1TaperProtocol)


CONTROL_METHODS = (
    "uncontrolled_all",
    "far_cap",
    "budget_matched_global",
)


def _positive_run(seed: int, protocols: CU1Protocols, device: torch.device):
    return train_positive(
        seed=seed,
        protocol=protocols.core,
        training=protocols.positive,
        device=device,
    )


def _source_rows(
    seeds: Sequence[int],
    protocols: CU1Protocols,
    device: torch.device,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        positive = _positive_run(seed, protocols, device)
        rows.append(
            dict(
                source_diagnostic(
                    seed=seed,
                    actor=positive.actor,
                    environment=positive.environment,
                    protocol=protocols.core,
                    source=protocols.source,
                )
            )
        )
    return rows


def _causal_rows(
    seeds: Sequence[int],
    protocols: CU1Protocols,
    device: torch.device,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    methods = protocols.causal.primary_methods + protocols.causal.appendix_methods
    branches = (
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
        positive = _positive_run(seed, protocols, device)
        for branch, sigma, alpha, learning_rate, steps in branches:
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
                rows.append(dict(run))
    return rows


def _phase_rows(
    seeds: Sequence[int],
    protocols: CU1Protocols,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []
    sigma = analytic_positive_sigma(protocols.core)
    for seed in seeds:
        positive = _positive_run(seed, protocols, device)
        for branch, fixed_sigma, alphas in (
            ("fixed_variance", sigma, protocols.phase.fixed_alphas),
            ("learnable_variance", None, protocols.phase.learnable_alphas),
        ):
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
                rows.append(dict(run))

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
            controls.append(dict(run))
    return rows, controls


def _taper_rows(
    seeds: Sequence[int],
    protocols: CU1Protocols,
    device: torch.device,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        positive = _positive_run(seed, protocols, device)
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
            rows.append(dict(run))
    return rows


def run_cu1_stage(
    *,
    stage: str,
    output_root: Path,
    seeds: Sequence[int] | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    protocols = CU1Protocols()
    selected = (
        tuple(getattr(protocols, stage).seeds)
        if seeds is None
        else tuple(int(seed) for seed in seeds)
    )
    target = torch.device(
        "cuda" if device == "auto" and torch.cuda.is_available()
        else "cpu" if device == "auto"
        else device
    )

    if stage == "source":
        result: dict[str, Any] = {
            "stage": stage,
            "rows": _source_rows(selected, protocols, target),
        }
    elif stage == "causal":
        result = {
            "stage": stage,
            "rows": _causal_rows(selected, protocols, target),
        }
    elif stage == "phase":
        rows, controls = _phase_rows(selected, protocols, target)
        result = {
            "stage": stage,
            "rows": rows,
            "controls": controls,
        }
    else:
        result = {
            "stage": stage,
            "rows": _taper_rows(selected, protocols, target),
        }

    result["seeds"] = list(selected)
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / f"{stage}.json", result)
    return result
