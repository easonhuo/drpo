"""Public runner for the four C-U1 paper experiments."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from drpo_reference.common import atomic_json

from .cu1 import CU1Protocol
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
    taper: CU1TaperProtocol = field(default_factory=CU1TaperProtocol)


CONTROL_METHODS = (
    "uncontrolled_all",
    "far_cap",
    "budget_matched_global",
)


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
        "cuda"
        if device == "auto" and torch.cuda.is_available()
        else "cpu"
        if device == "auto"
        else device
    )
    rows: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []
    sigma = analytic_positive_sigma(protocols.core)
    causal_methods = protocols.causal.primary_methods + protocols.causal.appendix_methods
    causal_branches = (
        (
            "fixed_variance",
            sigma,
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

    for seed in selected:
        positive = train_positive(
            seed=seed,
            protocol=protocols.core,
            training=protocols.positive,
            device=target,
        )
        common = {
            "seed": seed,
            "initialization_state": positive.initialization_state,
            "environment": positive.environment,
            "protocol": protocols.core,
            "positive_training": protocols.positive,
        }
        if stage == "source":
            rows.append(
                source_diagnostic(
                    seed=seed,
                    actor=positive.actor,
                    environment=positive.environment,
                    protocol=protocols.core,
                    source=protocols.source,
                )
            )
        elif stage == "causal":
            for branch, fixed_sigma, alpha, learning_rate, steps in causal_branches:
                for method in causal_methods:
                    rows.append(
                        run_causal_intervention(
                            **common,
                            causal=protocols.causal,
                            method=method,
                            fixed_sigma=fixed_sigma,
                            alpha=alpha,
                            learning_rate=learning_rate,
                            steps=steps,
                            branch=branch,
                        )
                    )
        elif stage == "phase":
            for branch, fixed_sigma, alphas in (
                ("fixed_variance", sigma, protocols.phase.fixed_alphas),
                ("learnable_variance", None, protocols.phase.learnable_alphas),
            ):
                for alpha in alphas:
                    rows.append(
                        run_phase_scan(
                            **common,
                            phase=protocols.phase,
                            alpha=alpha,
                            fixed_sigma=fixed_sigma,
                            branch=branch,
                        )
                    )
            for method in CONTROL_METHODS:
                controls.append(
                    run_causal_intervention(
                        **common,
                        causal=protocols.causal,
                        method=method,
                        fixed_sigma=sigma,
                        alpha=1.0,
                        learning_rate=protocols.phase.control_learning_rate,
                        steps=protocols.phase.control_steps,
                        branch="far_pressure_control",
                        partition="contour",
                        component_scales=(
                            protocols.phase.control_alpha_local,
                            protocols.phase.control_lambda_far,
                        ),
                        cap_ratio=protocols.phase.control_far_cap_ratio,
                        generator_offset=500009,
                    )
                )
        else:
            for family, retention in method_configs(protocols.taper):
                rows.append(
                    run_taper_method(
                        **common,
                        taper=protocols.taper,
                        family=family,
                        retention=retention,
                    )
                )

    result: dict[str, Any] = {"stage": stage, "rows": rows, "seeds": list(selected)}
    if stage == "phase":
        result["controls"] = controls
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / f"{stage}.json", result)
    return result
