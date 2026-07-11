"""Scale-one coefficient grid adapter for the canonical E7 sweep runner."""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping

from drpo.e7_canonical_injection import NegativeControl
from drpo import e7_canonical_sweep as base

COEFFICIENT_GRID_METHODS = {
    "reciprocal_linear",
    "reciprocal_quadratic",
    "exponential",
}


def _label(value: float) -> str:
    return f"{value:.8g}".replace("-", "m").replace(".", "p")


def expand_scale1_controls(grid: Mapping[str, Any]) -> list[NegativeControl]:
    """Expand anchors plus scale-one coefficient variants."""

    common = {
        "canonical_alpha": float(grid["canonical_alpha"]),
        "reference_distance": float(grid["reference_distance"]),
        "reciprocal_linear_coefficient": float(
            grid["coefficients"]["reciprocal_linear"]
        ),
        "reciprocal_quadratic_coefficient": float(
            grid["coefficients"]["reciprocal_quadratic"]
        ),
        "exponential_coefficient": float(grid["coefficients"]["exponential"]),
    }
    controls: list[NegativeControl] = []
    for method, raw in grid["anchors"].items():
        controls.append(
            NegativeControl(
                method=str(method),
                negative_scale=float(raw["negative_scale"]),
                **common,
            )
        )

    coefficient_grid = grid.get("coefficient_grid", {})
    if set(coefficient_grid) != COEFFICIENT_GRID_METHODS:
        raise ValueError(
            "coefficient_grid must contain exactly reciprocal_linear, "
            "reciprocal_quadratic, and exponential"
        )
    for method, coefficients in coefficient_grid.items():
        if not coefficients:
            raise ValueError(f"coefficient_grid[{method!r}] must not be empty")
        field = f"{method}_coefficient"
        for coefficient in coefficients:
            value = float(coefficient)
            if value <= 0.0:
                raise ValueError(
                    f"coefficient_grid[{method!r}] values must be positive"
                )
            method_common = dict(common)
            method_common[field] = value
            controls.append(
                NegativeControl(
                    method=method,
                    negative_scale=1.0,
                    **method_common,
                )
            )

    identities = [dataclasses.astuple(control) for control in controls]
    if len(identities) != len(set(identities)):
        raise ValueError("scale-one coefficient grid contains duplicate branches")
    expected = int(grid["branch_count_per_dataset_seed"])
    if len(controls) != expected:
        raise ValueError(
            f"branch_count_per_dataset_seed={expected} but expanded {len(controls)}"
        )
    return controls


def build_scale1_branches(
    contract: base.CanonicalContract,
    run_spec: Mapping[str, Any],
    grid: Mapping[str, Any],
) -> list[base.Branch]:
    """Build unique dataset-seed-method-coefficient branches."""

    datasets = [base.DatasetSpec.from_mapping(item) for item in run_spec["datasets"]]
    seeds = [int(value) for value in run_spec["seeds"]]
    injected_values = {
        str(key): str(value)
        for key, value in run_spec.get("injected_template_values", {}).items()
    }
    branches: list[base.Branch] = []
    for dataset in datasets:
        for seed in seeds:
            for control in expand_scale1_controls(grid):
                if control.canonical_alpha != contract.expected_canonical_alpha:
                    raise ValueError(
                        "grid canonical_alpha does not match canonical contract"
                    )
                if control.method in COEFFICIENT_GRID_METHODS:
                    coefficient = getattr(control, f"{control.method}_coefficient")
                    suffix = f"scale1__coef{_label(coefficient)}"
                else:
                    suffix = f"scale{_label(control.negative_scale)}"
                branches.append(
                    base.Branch(
                        branch_id=(
                            f"{dataset.id}__seed{seed}__{control.method}__{suffix}"
                        ),
                        branch_kind="injected",
                        dataset=dataset,
                        seed=seed,
                        template_values=dict(injected_values),
                        negative_control=control,
                    )
                )
            for raw_variant in run_spec.get("passthrough_variants", []):
                variant_id = str(raw_variant["id"])
                values = {
                    str(key): str(value)
                    for key, value in raw_variant.get("template_values", {}).items()
                }
                branches.append(
                    base.Branch(
                        branch_id=(
                            f"{dataset.id}__seed{seed}__baseline__{variant_id}"
                        ),
                        branch_kind="passthrough",
                        dataset=dataset,
                        seed=seed,
                        template_values=values,
                        negative_control=None,
                    )
                )
    ids = [branch.branch_id for branch in branches]
    if len(ids) != len(set(ids)):
        raise ValueError("branch IDs are not unique")
    return branches


def load_scale1_grid(path: str) -> tuple[dict[str, Any], str]:
    raw, digest = base.load_grid(path)
    if raw.get("negative_scale_grid") not in ({}, None):
        raise ValueError("scale-one coefficient tuning forbids negative_scale_grid")
    if raw.get("primary_selection_metric") != "final_score":
        raise ValueError("primary_selection_metric must be final_score")
    expand_scale1_controls(raw)
    return raw, digest


def main(argv: list[str] | None = None) -> int:
    """Run the existing canonical sweep CLI with scale-one grid expansion."""

    base.load_grid = load_scale1_grid
    base.build_branches = build_scale1_branches
    return base.main(argv)
