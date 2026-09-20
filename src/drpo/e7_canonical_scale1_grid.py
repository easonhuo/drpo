"""Scale-one coefficient grids and D4RL-9 task-wise GLQ tuning.

The legacy scale-one pilot remains supported byte-for-byte at the protocol level.
A second, fail-closed mode reuses the same canonical runner for a nine-task
pilot that tunes only Global, Reciprocal-Linear, and Reciprocal-Quadratic.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from drpo import e7_canonical_sweep as base
from drpo.e7_canonical_injection import CanonicalContract, NegativeControl

LEGACY_COEFFICIENT_GRID_METHODS = {
    "reciprocal_linear",
    "reciprocal_quadratic",
    "exponential",
}
D4RL9_TUNING_METHODS = {
    "global",
    "reciprocal_linear",
    "reciprocal_quadratic",
}
D4RL9_SCIENTIFIC_STATUS = (
    "d4rl9_taskwise_global_linear_quadratic_tuning_pilot_only"
)
D4RL9_RUNNER_VERSION = "2.0.0-d4rl9-glq-taskwise-tuning"
D4RL9_EXPECTED_DATASETS = (
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
D4RL9_SOURCE_RUN_SPEC_SEEDS = (200, 201)
D4RL9_TUNING_SEEDS = (200, 201, 202, 203)
D4RL9_HELD_OUT_SEEDS = (204, 205, 206, 207)
D4RL9_EXPECTED_MAX_WORKERS = 60
D4RL9_EXPECTED_EVALUATION_STEPS = tuple(range(50_000, 1_000_001, 50_000))
D4RL9_LATE_WINDOW_STEPS = (750_000, 800_000, 850_000, 900_000, 950_000, 1_000_000)

_BASE_LOAD_GRID = base.load_grid
_BASE_LOAD_RUN_SPEC = base.load_run_spec
_BASE_BUILD_BRANCHES = base.build_branches


def _label(value: float) -> str:
    return f"{value:.8g}".replace("-", "m").replace(".", "p")


def _is_d4rl9_grid(grid: Mapping[str, Any]) -> bool:
    return str(grid.get("scientific_status")) == D4RL9_SCIENTIFIC_STATUS


def _common_control_values(grid: Mapping[str, Any]) -> dict[str, float]:
    coefficients = grid["coefficients"]
    return {
        "canonical_alpha": float(grid["canonical_alpha"]),
        "reference_distance": float(grid["reference_distance"]),
        "reciprocal_linear_coefficient": float(coefficients["reciprocal_linear"]),
        "reciprocal_quadratic_coefficient": float(
            coefficients["reciprocal_quadratic"]
        ),
        "exponential_coefficient": float(coefficients["exponential"]),
    }


def _expand_legacy_scale1_controls(grid: Mapping[str, Any]) -> list[NegativeControl]:
    common = _common_control_values(grid)
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
    if set(coefficient_grid) != LEGACY_COEFFICIENT_GRID_METHODS:
        raise ValueError(
            "legacy coefficient_grid must contain exactly reciprocal_linear, "
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
    return controls


def _expand_d4rl9_controls(grid: Mapping[str, Any]) -> list[NegativeControl]:
    common = _common_control_values(grid)
    if grid.get("anchors") not in ({}, None):
        raise ValueError("D4RL-9 GLQ tuning forbids injected anchor branches")
    if grid.get("negative_scale_grid") not in ({}, None):
        raise ValueError(
            "D4RL-9 GLQ tuning uses global_scale_grid, not negative_scale_grid"
        )

    global_scales = [float(value) for value in grid.get("global_scale_grid", [])]
    if not global_scales or any(value <= 0.0 for value in global_scales):
        raise ValueError("global_scale_grid must contain positive values")
    if len(global_scales) != len(set(global_scales)):
        raise ValueError("global_scale_grid contains duplicates")

    coefficient_grid = grid.get("coefficient_grid", {})
    expected_coefficients = {"reciprocal_linear", "reciprocal_quadratic"}
    if set(coefficient_grid) != expected_coefficients:
        raise ValueError(
            "D4RL-9 coefficient_grid must contain exactly reciprocal_linear and "
            "reciprocal_quadratic"
        )

    controls = [
        NegativeControl(method="global", negative_scale=scale, **common)
        for scale in global_scales
    ]
    for method in ("reciprocal_linear", "reciprocal_quadratic"):
        coefficients = [float(value) for value in coefficient_grid[method]]
        if not coefficients or any(value <= 0.0 for value in coefficients):
            raise ValueError(
                f"coefficient_grid[{method!r}] must contain positive values"
            )
        if len(coefficients) != len(set(coefficients)):
            raise ValueError(f"coefficient_grid[{method!r}] contains duplicates")
        field = f"{method}_coefficient"
        for value in coefficients:
            method_common = dict(common)
            method_common[field] = value
            controls.append(
                NegativeControl(
                    method=method,
                    negative_scale=1.0,
                    **method_common,
                )
            )
    return controls


def expand_scale1_controls(grid: Mapping[str, Any]) -> list[NegativeControl]:
    """Expand the legacy scale-one grid or the D4RL-9 GLQ tuning grid."""

    if _is_d4rl9_grid(grid):
        controls = _expand_d4rl9_controls(grid)
    else:
        controls = _expand_legacy_scale1_controls(grid)

    identities = [dataclasses.astuple(control) for control in controls]
    if len(identities) != len(set(identities)):
        raise ValueError("scale-one grid contains duplicate branches")
    expected = int(grid["branch_count_per_dataset_seed"])
    if len(controls) != expected:
        raise ValueError(
            f"branch_count_per_dataset_seed={expected} but expanded {len(controls)}"
        )
    return controls


def _branch_suffix(control: NegativeControl) -> str:
    if control.method == "global":
        return f"scale{_label(control.negative_scale)}"
    if control.method in LEGACY_COEFFICIENT_GRID_METHODS:
        coefficient = getattr(control, f"{control.method}_coefficient")
        return f"scale1__coef{_label(coefficient)}"
    return f"scale{_label(control.negative_scale)}"


def build_scale1_branches(
    contract: CanonicalContract,
    run_spec: Mapping[str, Any],
    grid: Mapping[str, Any],
) -> list[base.Branch]:
    """Build unique dataset-seed-method-hyperparameter branches."""

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
                if not math.isclose(
                    control.canonical_alpha,
                    contract.expected_canonical_alpha,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        "grid canonical_alpha does not match canonical contract"
                    )
                branches.append(
                    base.Branch(
                        branch_id=(
                            f"{dataset.id}__seed{seed}__{control.method}__"
                            f"{_branch_suffix(control)}"
                        ),
                        branch_kind="injected",
                        dataset=dataset,
                        seed=seed,
                        template_values=dict(injected_values),
                        negative_control=control,
                    )
                )
            if not _is_d4rl9_grid(grid):
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
    if _is_d4rl9_grid(grid):
        expected = int(grid["expected_total_branches"])
        if len(branches) != expected:
            raise ValueError(
                f"expected {expected} D4RL-9 branches but built {len(branches)}"
            )
    return branches


def _validate_d4rl9_grid(raw: Mapping[str, Any]) -> None:
    if tuple(raw.get("expected_datasets", ())) != D4RL9_EXPECTED_DATASETS:
        raise ValueError("expected_datasets changed")
    if tuple(int(value) for value in raw.get("source_run_spec_seeds", ())) != (
        D4RL9_SOURCE_RUN_SPEC_SEEDS
    ):
        raise ValueError("source_run_spec_seeds changed")
    if tuple(int(value) for value in raw.get("tuning_seeds", ())) != D4RL9_TUNING_SEEDS:
        raise ValueError("tuning_seeds changed")
    if tuple(
        int(value) for value in raw.get("held_out_seeds", ())
    ) != D4RL9_HELD_OUT_SEEDS:
        raise ValueError("held_out_seeds changed")
    if int(raw.get("fixed_max_workers", -1)) != D4RL9_EXPECTED_MAX_WORKERS:
        raise ValueError("fixed_max_workers must remain 60")
    if raw.get("primary_selection_metric") != "late_window_mean_750k_to_1m":
        raise ValueError("primary_selection_metric changed")
    if raw.get("selection_scope") != "per_dataset_per_method":
        raise ValueError("selection_scope changed")
    if tuple(int(value) for value in raw.get("late_window_steps", ())) != (
        D4RL9_LATE_WINDOW_STEPS
    ):
        raise ValueError("late_window_steps changed")
    if not math.isclose(
        float(raw.get("canonical_alpha")), 0.11, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("canonical_alpha must remain 0.11")
    if not math.isclose(
        float(raw.get("reference_distance")), 2.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("reference_distance must remain 2.0")
    controls = expand_scale1_controls(raw)
    method_counts = {
        method: sum(control.method == method for control in controls)
        for method in D4RL9_TUNING_METHODS
    }
    if set(method_counts.values()) != {5}:
        raise ValueError(
            "D4RL-9 GLQ tuning requires exactly five candidates per method"
        )
    expected = (
        len(D4RL9_EXPECTED_DATASETS)
        * len(D4RL9_TUNING_SEEDS)
        * len(controls)
    )
    if int(raw.get("expected_total_branches", -1)) != expected:
        raise ValueError(f"expected_total_branches must remain {expected}")


def load_scale1_grid(path: str | Path) -> tuple[dict[str, Any], str]:
    raw, digest = _BASE_LOAD_GRID(path)
    if _is_d4rl9_grid(raw):
        _validate_d4rl9_grid(raw)
        return raw, digest
    if raw.get("negative_scale_grid") not in ({}, None):
        raise ValueError("scale-one coefficient tuning forbids negative_scale_grid")
    if raw.get("primary_selection_metric") != "final_score":
        raise ValueError("primary_selection_metric must be final_score")
    expand_scale1_controls(raw)
    return raw, digest


def _flag_value(argv: Sequence[str], flag: str) -> str:
    positions = [index for index, token in enumerate(argv) if token == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        raise ValueError(f"trainer_argv_template must contain exactly one {flag}")
    return str(argv[positions[0] + 1])


def load_d4rl9_run_spec(path: str | Path) -> tuple[dict[str, Any], str]:
    """Validate the server-local nine-task spec and expose four tuning seeds."""

    raw, digest = _BASE_LOAD_RUN_SPEC(path)
    run_spec = copy.deepcopy(raw)
    if run_spec.get("experiment_id") != base.EXPERIMENT_ID:
        raise ValueError("run_spec experiment_id changed")
    dataset_ids = tuple(str(item["id"]) for item in run_spec["datasets"])
    if dataset_ids != D4RL9_EXPECTED_DATASETS:
        raise ValueError(f"run_spec datasets changed: {dataset_ids}")
    source_seeds = tuple(int(value) for value in run_spec["seeds"])
    if source_seeds != D4RL9_SOURCE_RUN_SPEC_SEEDS:
        raise ValueError(
            "source run_spec seeds changed: "
            f"{source_seeds}; expected {D4RL9_SOURCE_RUN_SPEC_SEEDS}"
        )
    run_spec["seeds"] = list(D4RL9_TUNING_SEEDS)

    passthrough = run_spec.get("passthrough_variants", [])
    passthrough_ids = [str(item.get("id")) for item in passthrough]
    if passthrough_ids not in ([], ["original_exp_rank_mr"]):
        raise ValueError("unexpected passthrough baseline in source run_spec")
    run_spec["passthrough_variants"] = []

    environment = run_spec.get("environment", {})
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        if str(environment.get(name)) != "1":
            raise ValueError(f"run_spec {name} must remain 1")

    argv = [str(item) for item in run_spec["trainer_argv_template"]]
    expected_flags = {
        "--variant": "iqlv_exp_rank",
        "--alpha": "0.11",
        "--tau": "0.5",
        "--temp": "5.0",
        "--steps": "1000000",
        "--batch": "256",
        "--lr": "0.0003",
        "--eval_interval": "50000",
        "--eval_episodes": "10",
    }
    changed = {
        flag: {"expected": value, "actual": _flag_value(argv, flag)}
        for flag, value in expected_flags.items()
        if _flag_value(argv, flag) != value
    }
    if changed:
        raise ValueError(f"source run_spec trainer flags changed: {changed}")
    return run_spec, digest


def _mean(values: Sequence[float]) -> float:
    return float(statistics.fmean(values))


def _sample_std(values: Sequence[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def _population_std(values: Sequence[float]) -> float:
    return float(statistics.pstdev(values)) if len(values) > 1 else 0.0


def _least_squares_slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    x_mean = _mean(xs)
    y_mean = _mean(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if denominator == 0.0:
        return 0.0
    return float(
        sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
        / denominator
    )


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return payload


def _candidate_metadata(control: NegativeControl) -> tuple[str, str, float]:
    if control.method == "global":
        return "global", "negative_scale", float(control.negative_scale)
    if control.method == "reciprocal_linear":
        return (
            "reciprocal_linear",
            "reciprocal_linear_coefficient",
            float(control.reciprocal_linear_coefficient),
        )
    if control.method == "reciprocal_quadratic":
        return (
            "reciprocal_quadratic",
            "reciprocal_quadratic_coefficient",
            float(control.reciprocal_quadratic_coefficient),
        )
    raise RuntimeError(f"unexpected D4RL-9 tuning method: {control.method}")


def _audit_branch(work_dir: Path, branch: base.Branch) -> dict[str, Any]:
    if branch.negative_control is None:
        raise RuntimeError("D4RL-9 tuning audit received a passthrough branch")
    branch_dir = work_dir / "branches" / branch.branch_id
    completed_path = branch_dir / "COMPLETED.json"
    if not completed_path.is_file():
        raise RuntimeError(f"branch is not complete: {branch.branch_id}")
    completed = _read_json(completed_path)
    if completed.get("branch_id") != branch.branch_id:
        raise RuntimeError(f"completed manifest identity mismatch: {completed_path}")
    if int(completed.get("return_code", -1)) != 0:
        raise RuntimeError(
            f"completed branch has nonzero return code: {branch.branch_id}"
        )

    summaries = sorted((branch_dir / "trainer_output").glob("*_summary.json"))
    if len(summaries) != 1:
        raise RuntimeError(
            f"expected one trainer summary for {branch.branch_id}, "
            f"found {len(summaries)}"
        )
    summary_path = summaries[0]
    summary = _read_json(summary_path)
    expected_metadata = {
        "dataset": branch.dataset.id,
        "variant": "iqlv_exp_rank",
        "seed": branch.seed,
        "steps": 1_000_000,
        "score_type": "norm",
        "goal_conditioned": False,
    }
    mismatches = {
        key: {"expected": expected, "actual": summary.get(key)}
        for key, expected in expected_metadata.items()
        if summary.get(key) != expected
    }
    for key, expected in (("alpha", 0.11), ("tau", 0.5)):
        actual = float(summary.get(key))
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
            mismatches[key] = {"expected": expected, "actual": actual}
    if mismatches:
        raise RuntimeError(f"trainer summary metadata mismatch: {mismatches}")

    history = summary.get("history")
    if not isinstance(history, Mapping):
        raise RuntimeError(f"trainer summary has no history: {summary_path}")
    steps = [int(value) for value in history.get("steps", [])]
    metric_keys = [str(key) for key in history if key != "steps"]
    if len(metric_keys) != 1:
        raise RuntimeError(
            f"trainer history must contain one score series: {summary_path}"
        )
    scores = [float(value) for value in history[metric_keys[0]]]
    if tuple(steps) != D4RL9_EXPECTED_EVALUATION_STEPS:
        raise RuntimeError(f"evaluation cadence mismatch: {summary_path}")
    if len(scores) != len(steps) or any(not math.isfinite(value) for value in scores):
        raise RuntimeError(f"non-finite or misaligned score history: {summary_path}")
    if not math.isclose(
        float(summary.get("final_score")), scores[-1], rel_tol=0.0, abs_tol=1e-12
    ):
        raise RuntimeError(f"final score mismatch: {summary_path}")

    score_by_step = dict(zip(steps, scores))
    late_scores = [score_by_step[step] for step in D4RL9_LATE_WINDOW_STEPS]
    best_index = max(range(len(scores)), key=scores.__getitem__)
    best_score = scores[best_index]
    late_mean = _mean(late_scores)
    method, parameter_name, parameter_value = _candidate_metadata(
        branch.negative_control
    )
    return {
        "branch_id": branch.branch_id,
        "dataset_id": branch.dataset.id,
        "seed": branch.seed,
        "method": method,
        "parameter_name": parameter_name,
        "parameter_value": parameter_value,
        "late_window_mean": late_mean,
        "late_window_std": _population_std(late_scores),
        "late_window_min": min(late_scores),
        "late_window_max": max(late_scores),
        "final_score": scores[-1],
        "best_score": best_score,
        "best_step": steps[best_index],
        "best_to_final_drop": best_score - scores[-1],
        "best_to_late_mean_drop": best_score - late_mean,
        "terminal_slope_per_100k_steps": _least_squares_slope(
            [float(step) for step in D4RL9_LATE_WINDOW_STEPS], late_scores
        )
        * 100_000.0,
        "completed_manifest": str(completed_path.relative_to(work_dir)),
        "trainer_summary": str(summary_path.relative_to(work_dir)),
        "terminal_classification": "fixed_horizon_inconclusive",
    }


def _aggregate_candidates(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, float], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["dataset_id"]),
            str(row["method"]),
            str(row["parameter_name"]),
            float(row["parameter_value"]),
        )
        grouped.setdefault(key, []).append(row)

    groups: list[dict[str, Any]] = []
    for (dataset_id, method, parameter_name, parameter_value), members in sorted(
        grouped.items()
    ):
        members = sorted(members, key=lambda item: int(item["seed"]))
        seeds = [int(item["seed"]) for item in members]
        if seeds != list(D4RL9_TUNING_SEEDS):
            raise RuntimeError(
                f"candidate seed coverage mismatch for {dataset_id}/{method}/"
                f"{parameter_value}: {seeds}"
            )
        late = [float(item["late_window_mean"]) for item in members]
        groups.append(
            {
                "dataset_id": dataset_id,
                "method": method,
                "parameter_name": parameter_name,
                "parameter_value": parameter_value,
                "seeds": seeds,
                "seed_count": len(seeds),
                "late_window_mean_across_seeds": _mean(late),
                "late_window_std_across_seeds": _sample_std(late),
                "late_window_min_across_seeds": min(late),
                "late_window_max_across_seeds": max(late),
                "final_mean_across_seeds": _mean(
                    [float(item["final_score"]) for item in members]
                ),
                "best_mean_across_seeds": _mean(
                    [float(item["best_score"]) for item in members]
                ),
                "best_to_final_drop_mean": _mean(
                    [float(item["best_to_final_drop"]) for item in members]
                ),
                "best_to_late_mean_drop_mean": _mean(
                    [float(item["best_to_late_mean_drop"]) for item in members]
                ),
                "terminal_slope_per_100k_mean": _mean(
                    [float(item["terminal_slope_per_100k_steps"]) for item in members]
                ),
                "terminal_classification": "fixed_horizon_inconclusive",
            }
        )
    return groups


def _select_taskwise(groups: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_cell: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in groups:
        by_cell.setdefault((str(row["dataset_id"]), str(row["method"])), []).append(row)

    selections: list[dict[str, Any]] = []
    for dataset_id in D4RL9_EXPECTED_DATASETS:
        for method in ("global", "reciprocal_linear", "reciprocal_quadratic"):
            candidates = by_cell.get((dataset_id, method), [])
            if len(candidates) != 5:
                raise RuntimeError(
                    f"expected five candidates for {dataset_id}/{method}, "
                    f"found {len(candidates)}"
                )
            ranked = sorted(
                candidates,
                key=lambda row: (
                    -float(row["late_window_mean_across_seeds"]),
                    -float(row["late_window_min_across_seeds"]),
                    float(row["best_to_late_mean_drop_mean"]),
                    float(row["parameter_value"]),
                ),
            )
            winner = dict(ranked[0])
            winner["selection_rank"] = 1
            winner["selection_rule"] = [
                "maximize late_window_mean_across_seeds",
                "then maximize late_window_min_across_seeds",
                "then minimize best_to_late_mean_drop_mean",
                "then choose the smaller numeric hyperparameter",
            ]
            winner["candidate_count"] = len(candidates)
            selections.append(winner)
    return selections


def build_d4rl9_terminal_audit(
    *, work_dir: Path, branches: Sequence[base.Branch], grid: Mapping[str, Any]
) -> dict[str, Any]:
    """Audit all 540 tuning branches and select one setting per task/method."""

    expected = int(grid["expected_total_branches"])
    if len(branches) != expected:
        raise RuntimeError(f"audit expected {expected} branches, found {len(branches)}")
    rows = [_audit_branch(work_dir, branch) for branch in branches]
    groups = _aggregate_candidates(rows)
    selections = _select_taskwise(groups)
    payload = {
        "schema_version": 1,
        "experiment_id": base.EXPERIMENT_ID,
        "scientific_status": D4RL9_SCIENTIFIC_STATUS,
        "runner_version": D4RL9_RUNNER_VERSION,
        "status": "PASS",
        "expected_branch_count": expected,
        "audited_branch_count": len(rows),
        "datasets": list(D4RL9_EXPECTED_DATASETS),
        "tuning_seeds": list(D4RL9_TUNING_SEEDS),
        "held_out_seeds_untouched": list(D4RL9_HELD_OUT_SEEDS),
        "methods": ["global", "reciprocal_linear", "reciprocal_quadratic"],
        "primary_metric": "late_window_mean_750k_to_1m",
        "late_window_steps": list(D4RL9_LATE_WINDOW_STEPS),
        "selection_scope": "per_dataset_per_method",
        "cross_method_ranking_allowed": False,
        "formal_d4rl9_table_population_allowed": False,
        "steady_state_claim_allowed": False,
        "fixed_horizon_is_not_convergence": True,
        "positive_only_rerun_included": False,
        "exponential_rerun_included": False,
        "branches": rows,
        "candidate_groups": groups,
        "taskwise_selections": selections,
        "event_separation_summary": {
            "task_performance_collapse": "not_classified_without_registered_threshold",
            "support_or_variance_boundary_event": (
                "not_available_in_unchanged_canonical_trainer_summary"
            ),
            "nan_inf_numerical_failure": (
                "not_observed_in_zero_exit_branches_and_finite_evaluation_histories"
            ),
        },
    }
    base.atomic_write_json(work_dir / "TERMINAL_AUDIT.json", payload)
    base.atomic_write_json(
        work_dir / "TASKWISE_SELECTION.json",
        {
            "schema_version": 1,
            "experiment_id": base.EXPERIMENT_ID,
            "scientific_status": D4RL9_SCIENTIFIC_STATUS,
            "primary_metric": payload["primary_metric"],
            "tuning_seeds": list(D4RL9_TUNING_SEEDS),
            "held_out_seeds_untouched": list(D4RL9_HELD_OUT_SEEDS),
            "selection_scope": payload["selection_scope"],
            "taskwise_selections": selections,
            "confirmation_required_before_method_ranking": True,
        },
    )
    return payload


def _extract_option(argv: Sequence[str], name: str) -> str:
    positions = [index for index, token in enumerate(argv) if token == name]
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        raise ValueError(f"command must contain exactly one {name}")
    return str(argv[positions[0] + 1])


def _d4rl9_mode_from_argv(argv: Sequence[str]) -> bool:
    if "--grid" not in argv:
        return False
    path = Path(_extract_option(argv, "--grid"))
    raw = json.loads(path.read_text())
    return _is_d4rl9_grid(raw)


def _normalized_d4rl9_argv(argv: Sequence[str]) -> list[str]:
    values = list(argv)
    if "--max-workers" in values:
        index = values.index("--max-workers")
        if (
            index + 1 >= len(values)
            or int(values[index + 1]) != D4RL9_EXPECTED_MAX_WORKERS
        ):
            raise ValueError("D4RL-9 GLQ tuning fixes --max-workers at 60")
    else:
        values.extend(["--max-workers", str(D4RL9_EXPECTED_MAX_WORKERS)])
    return values


def _prepare_d4rl9_audit(
    *, contract_path: str, run_spec_path: str, grid_path: str, work_dir: str
) -> tuple[list[base.Branch], dict[str, Any], Path]:
    contract = CanonicalContract.load(Path(contract_path).expanduser().resolve())
    contract.verify_runtime()
    run_spec, _ = load_d4rl9_run_spec(run_spec_path)
    grid, _ = load_scale1_grid(grid_path)
    if not _is_d4rl9_grid(grid):
        raise ValueError("audit is only available for the D4RL-9 GLQ tuning grid")
    branches = build_scale1_branches(contract, run_spec, grid)
    for dataset in {branch.dataset for branch in branches}:
        dataset.verify()
    return branches, grid, Path(work_dir).expanduser().resolve()


def _audit_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit completed D4RL-9 GLQ task-wise tuning branches"
    )
    parser.add_argument("command", choices=("audit",))
    parser.add_argument("--contract", required=True)
    parser.add_argument("--run-spec", required=True)
    parser.add_argument("--grid", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--max-workers", type=int, default=D4RL9_EXPECTED_MAX_WORKERS)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run legacy scale-one tuning or the D4RL-9 task-wise GLQ pilot."""

    values = list(sys.argv[1:] if argv is None else argv)
    if values and values[0] == "audit":
        args = _audit_parser().parse_args(values)
        if args.max_workers != D4RL9_EXPECTED_MAX_WORKERS:
            raise ValueError("D4RL-9 GLQ tuning fixes --max-workers at 60")
        branches, grid, work_dir = _prepare_d4rl9_audit(
            contract_path=args.contract,
            run_spec_path=args.run_spec,
            grid_path=args.grid,
            work_dir=args.work_dir,
        )
        audit = build_d4rl9_terminal_audit(
            work_dir=work_dir, branches=branches, grid=grid
        )
        print(
            json.dumps(
                {
                    "status": audit["status"],
                    "terminal_audit": str(work_dir / "TERMINAL_AUDIT.json"),
                    "taskwise_selection": str(work_dir / "TASKWISE_SELECTION.json"),
                },
                sort_keys=True,
            )
        )
        return 0

    d4rl9_mode = _d4rl9_mode_from_argv(values)
    delegated = _normalized_d4rl9_argv(values) if d4rl9_mode else values
    previous_load_grid = base.load_grid
    previous_load_run_spec = base.load_run_spec
    previous_build_branches = base.build_branches
    previous_status = base.SCIENTIFIC_STATUS
    previous_version = base.RUNNER_VERSION
    base.load_grid = load_scale1_grid
    base.build_branches = build_scale1_branches
    if d4rl9_mode:
        base.load_run_spec = load_d4rl9_run_spec
        base.SCIENTIFIC_STATUS = D4RL9_SCIENTIFIC_STATUS
        base.RUNNER_VERSION = D4RL9_RUNNER_VERSION
    try:
        returncode = base.main(delegated)
        if returncode == 0 and d4rl9_mode and delegated and delegated[0] == "run":
            branches, grid, work_dir = _prepare_d4rl9_audit(
                contract_path=_extract_option(delegated, "--contract"),
                run_spec_path=_extract_option(delegated, "--run-spec"),
                grid_path=_extract_option(delegated, "--grid"),
                work_dir=_extract_option(delegated, "--work-dir"),
            )
            build_d4rl9_terminal_audit(
                work_dir=work_dir, branches=branches, grid=grid
            )
        return returncode
    finally:
        base.load_grid = previous_load_grid
        base.load_run_spec = previous_load_run_spec
        base.build_branches = previous_build_branches
        base.SCIENTIFIC_STATUS = previous_status
        base.RUNNER_VERSION = previous_version


if __name__ == "__main__":
    raise SystemExit(main())
