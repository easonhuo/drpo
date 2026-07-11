"""EXP-only mixed-horizon pilot adapter for the canonical E7 sweep runner."""

from __future__ import annotations

import copy
import math
import sys
from typing import Any, Mapping

from drpo import e7_canonical_sweep as base
from drpo.e7_canonical_injection import NegativeControl

_BASE_LOAD_GRID = base.load_grid
_BASE_LOAD_RUN_SPEC = base.load_run_spec
_BASE_BUILD_BRANCHES = base.build_branches

EXPECTED_DATASETS = (
    "hopper-medium-v2",
    "hopper-medium-replay-v2",
    "hopper-medium-expert-v2",
    "walker2d-medium-v2",
    "walker2d-medium-replay-v2",
    "walker2d-medium-expert-v2",
    "halfcheetah-medium-v2",
    "halfcheetah-medium-replay-v2",
    "halfcheetah-medium-expert-v2",
)
EXPECTED_SEEDS = (200, 201, 202, 203)
EXPECTED_MAX_WORKERS = 60
LEGACY_EXP_COEFFICIENT = 0.374162511054291


def _label(value: float) -> str:
    return f"{value:.8g}".replace("-", "m").replace(".", "p")


def _steps_label(steps: int) -> str:
    if steps % 1_000_000 == 0:
        return f"{steps // 1_000_000}m"
    return str(steps)


def _control(*, scale: float, coefficient: float, alpha: float, distance: float) -> NegativeControl:
    return NegativeControl(
        method="exponential",
        negative_scale=scale,
        canonical_alpha=alpha,
        reference_distance=distance,
        exponential_coefficient=coefficient,
    )


def load_exp_horizon_grid(path: str) -> tuple[dict[str, Any], str]:
    raw, digest = _BASE_LOAD_GRID(path)
    if raw.get("scientific_status") != "exp_scale1_coefficient_horizon_joint_pilot_only":
        raise ValueError("unexpected scientific_status for EXP horizon joint pilot")
    if raw.get("primary_selection_metric") != "final_score":
        raise ValueError("primary_selection_metric must remain final_score")
    if tuple(raw.get("expected_datasets", ())) != EXPECTED_DATASETS:
        raise ValueError("expected_datasets changed")
    if tuple(int(value) for value in raw.get("tuning_seeds", ())) != EXPECTED_SEEDS:
        raise ValueError("tuning_seeds changed")
    if int(raw.get("branch_count_per_dataset_seed", -1)) != 12:
        raise ValueError("branch_count_per_dataset_seed must remain 12")
    if int(raw.get("expected_total_branches", -1)) != 432:
        raise ValueError("expected_total_branches must remain 432")

    alpha = float(raw.get("canonical_alpha"))
    distance = float(raw.get("reference_distance"))
    if not math.isclose(alpha, 0.11, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("canonical_alpha must remain 0.11")
    if not math.isclose(distance, 2.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("reference_distance must remain 2.0")

    long_coefficients = [float(value) for value in raw.get("exp_scale1_2m_coefficients", [])]
    short_coefficients = [float(value) for value in raw.get("exp_scale1_1m_coefficients", [])]
    if long_coefficients != [LEGACY_EXP_COEFFICIENT, 1.0, 1.5]:
        raise ValueError("2M EXP coefficients changed")
    if short_coefficients != [0.25, 0.5, 0.75, 1.25, 2.0, 3.0]:
        raise ValueError("1M EXP coefficients changed")
    all_coefficients = long_coefficients + short_coefficients
    if len(all_coefficients) != len(set(all_coefficients)):
        raise ValueError("EXP coefficient grid contains duplicates")
    return raw, digest


def load_exp_horizon_run_spec(path: str) -> tuple[dict[str, Any], str]:
    raw, digest = _BASE_LOAD_RUN_SPEC(path)
    run_spec = copy.deepcopy(raw)
    dataset_ids = tuple(str(item["id"]) for item in run_spec["datasets"])
    if dataset_ids != EXPECTED_DATASETS:
        raise ValueError(f"run_spec datasets changed: {dataset_ids}")
    seeds = tuple(int(value) for value in run_spec["seeds"])
    if seeds != EXPECTED_SEEDS:
        raise ValueError(f"run_spec seeds changed: {seeds}")
    passthrough = run_spec.get("passthrough_variants", [])
    if [str(item.get("id")) for item in passthrough] != ["original_exp_rank_mr"]:
        raise ValueError("run_spec passthrough baseline changed")

    argv = [str(item) for item in run_spec["trainer_argv_template"]]
    step_positions = [index for index, token in enumerate(argv) if token == "--steps"]
    if step_positions != [step_positions[0]] if step_positions else True:
        raise ValueError("trainer_argv_template must contain exactly one --steps flag")
    step_index = step_positions[0]
    if step_index + 1 >= len(argv) or argv[step_index + 1] != "1000000":
        raise ValueError("source run_spec --steps must remain the prior 1000000")
    argv[step_index + 1] = "{steps}"
    run_spec["trainer_argv_template"] = argv
    return run_spec, digest


def build_exp_horizon_branches(
    contract: base.CanonicalContract,
    run_spec: Mapping[str, Any],
    grid: Mapping[str, Any],
) -> list[base.Branch]:
    datasets = [base.DatasetSpec.from_mapping(item) for item in run_spec["datasets"]]
    seeds = [int(value) for value in run_spec["seeds"]]
    injected_values = {
        str(key): str(value)
        for key, value in run_spec.get("injected_template_values", {}).items()
    }
    alpha = float(grid["canonical_alpha"])
    distance = float(grid["reference_distance"])
    if not math.isclose(alpha, contract.expected_canonical_alpha, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("grid canonical_alpha does not match canonical contract")

    branch_specs: list[tuple[str, int, NegativeControl | None, dict[str, str]]] = []
    for coefficient in grid["exp_scale1_2m_coefficients"]:
        value = float(coefficient)
        branch_specs.append(
            (
                f"exponential__scale1__coef{_label(value)}__steps2m",
                2_000_000,
                _control(scale=1.0, coefficient=value, alpha=alpha, distance=distance),
                {},
            )
        )
    for coefficient in grid["exp_scale1_1m_coefficients"]:
        value = float(coefficient)
        branch_specs.append(
            (
                f"exponential__scale1__coef{_label(value)}__steps1m",
                1_000_000,
                _control(scale=1.0, coefficient=value, alpha=alpha, distance=distance),
                {},
            )
        )
    branch_specs.extend(
        [
            (
                "positive_only__scale0__steps1m",
                1_000_000,
                NegativeControl(
                    method="positive_only",
                    negative_scale=0.0,
                    canonical_alpha=alpha,
                    reference_distance=distance,
                    exponential_coefficient=LEGACY_EXP_COEFFICIENT,
                ),
                {},
            ),
            (
                f"exponential__scale0p1__coef{_label(LEGACY_EXP_COEFFICIENT)}__steps1m",
                1_000_000,
                _control(
                    scale=0.1,
                    coefficient=LEGACY_EXP_COEFFICIENT,
                    alpha=alpha,
                    distance=distance,
                ),
                {},
            ),
        ]
    )

    branches: list[base.Branch] = []
    for suffix, steps, control, extra_values in branch_specs:
        for dataset in datasets:
            for seed in seeds:
                values = {**injected_values, **extra_values, "steps": str(steps)}
                branches.append(
                    base.Branch(
                        branch_id=f"{dataset.id}__seed{seed}__{suffix}",
                        branch_kind="injected",
                        dataset=dataset,
                        seed=seed,
                        template_values=values,
                        negative_control=control,
                    )
                )

    baseline = run_spec["passthrough_variants"][0]
    baseline_values = {
        str(key): str(value)
        for key, value in baseline.get("template_values", {}).items()
    }
    baseline_values["steps"] = "1000000"
    for dataset in datasets:
        for seed in seeds:
            branches.append(
                base.Branch(
                    branch_id=(
                        f"{dataset.id}__seed{seed}__baseline__"
                        "original_exp_rank_mr__steps1m"
                    ),
                    branch_kind="passthrough",
                    dataset=dataset,
                    seed=seed,
                    template_values=dict(baseline_values),
                    negative_control=None,
                )
            )

    ids = [branch.branch_id for branch in branches]
    if len(ids) != len(set(ids)):
        raise ValueError("branch IDs are not unique")
    expected = int(grid["expected_total_branches"])
    if len(branches) != expected:
        raise ValueError(f"expected {expected} branches but built {len(branches)}")
    return branches


def _normalized_argv(argv: list[str] | None) -> list[str]:
    values = list(sys.argv[1:] if argv is None else argv)
    if "--max-workers" in values:
        index = values.index("--max-workers")
        if index + 1 >= len(values) or int(values[index + 1]) != EXPECTED_MAX_WORKERS:
            raise ValueError("this pilot fixes --max-workers at the previously validated value 60")
    else:
        values.extend(["--max-workers", str(EXPECTED_MAX_WORKERS)])
    return values


def main(argv: list[str] | None = None) -> int:
    """Delegate to the generic canonical runner with mixed-horizon EXP hooks."""

    previous_load_grid = base.load_grid
    previous_load_run_spec = base.load_run_spec
    previous_build_branches = base.build_branches
    base.load_grid = load_exp_horizon_grid
    base.load_run_spec = load_exp_horizon_run_spec
    base.build_branches = build_exp_horizon_branches
    try:
        return base.main(_normalized_argv(argv))
    finally:
        base.load_grid = previous_load_grid
        base.load_run_spec = previous_load_run_spec
        base.build_branches = previous_build_branches
