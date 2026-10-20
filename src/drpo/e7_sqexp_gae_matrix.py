"""Exact branch matrix for EXT-H-E7-SQEXP-GAE-01."""
from __future__ import annotations

import math
from typing import Any, Mapping

from drpo import e7_canonical_sweep as base
from drpo import e7_squared_exp_night as night
from drpo.e7_sqexp_gae_protocol import (
    ACTOR_MODES,
    COEFFICIENTS,
    ESTIMATORS,
    EXPECTED_BRANCHES,
    EXPECTED_SEEDS,
    STEPS,
)


def _label(value: float) -> str:
    return f"{value:.8g}".replace("-", "m").replace(".", "p")


def build_branches(
    contract: base.CanonicalContract,
    run_spec: Mapping[str, Any],
    grid: Mapping[str, Any],
) -> list[base.Branch]:
    if not math.isclose(contract.expected_canonical_alpha, 0.11, abs_tol=1e-12):
        raise ValueError("canonical source alpha changed")
    datasets = [base.DatasetSpec.from_mapping(item) for item in run_spec["datasets"]]
    controls = [("positive_only", 0.0, 0.0)] + [
        ("squared_exponential", 1.0, coefficient) for coefficient in COEFFICIENTS
    ]
    branches: list[base.Branch] = []
    for estimator in ESTIMATORS:
        for actor_mode in ACTOR_MODES:
            for method, weight_at_zero, coefficient in controls:
                label = (
                    "positive_only"
                    if method == "positive_only"
                    else f"sqexp_c{_label(coefficient)}"
                )
                for dataset in datasets:
                    for seed in EXPECTED_SEEDS:
                        branches.append(
                            base.Branch(
                                branch_id=(
                                    f"{dataset.id}__seed{seed}__{estimator}__{label}__"
                                    f"{actor_mode}__steps1m"
                                ),
                                branch_kind="injected",
                                dataset=dataset,
                                seed=seed,
                                template_values={
                                    "steps": str(STEPS),
                                    "actor_update_mode": actor_mode,
                                    "advantage_estimator": estimator,
                                    "weight_method": method,
                                    "weight_at_zero": f"{weight_at_zero:.17g}",
                                    "exp_coefficient": f"{coefficient:.17g}",
                                    "reference_distance": f"{night.REFERENCE_DISTANCE:.17g}",
                                    "diagnostics_interval": "1000",
                                    "sampled_values_per_update": "16",
                                },
                                negative_control=None,
                            )
                        )
    ids = [branch.branch_id for branch in branches]
    if len(ids) != EXPECTED_BRANCHES or len(ids) != len(set(ids)):
        raise RuntimeError(f"expected {EXPECTED_BRANCHES} unique branches")
    return branches
