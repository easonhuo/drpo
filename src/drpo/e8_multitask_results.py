"""Method-agnostic result projection helpers for E8 multitask experiments."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import zipfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

import numpy as np


class CellLike(Protocol):
    @property
    def key(self) -> str: ...

    task: str
    method: str
    seed: int
    stage: str


class MethodSpecLike(Protocol):
    name: str
    scientific_kernel: str
    parameters: Callable[[CellLike], Mapping[str, Any]]
    single_aggregate_metadata: Callable[[Mapping[str, Any]], Mapping[str, Any]]
    matrix_aggregate_metadata: Callable[[Mapping[str, Any]], Mapping[str, Any]]


_RESERVED_COMMON_COLUMNS = frozenset(
    {"source", "task", "method", "seed", "stage", "cell_key"}
)


def _checked_merge(
    target: dict[str, Any], values: Mapping[str, Any], *, label: str
) -> None:
    collisions = sorted(set(target).intersection(values))
    if collisions:
        raise ValueError(
            f"{label} attempted to overwrite result columns: {collisions}"
        )
    target.update(values)


def common_result_row(
    cell: CellLike,
    *,
    source: str,
    method_columns: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    compatibility = dict(method_columns)
    reserved = sorted(_RESERVED_COMMON_COLUMNS.intersection(compatibility))
    if reserved:
        raise ValueError(
            f"Method projection uses reserved result columns: {reserved}"
        )
    public: dict[str, Any] = {
        "source": source,
        "task": cell.task,
        "method": cell.method,
    }
    _checked_merge(public, compatibility, label="Method projection")
    public.update(
        {
            "seed": int(cell.seed),
            "stage": cell.stage,
            "cell_key": cell.key,
        }
    )
    _checked_merge(public, dict(metrics), label="Metric projection")
    return public


def parameter_identity(parameters: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(parameters),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def parameter_sort_key(parameters: Mapping[str, Any]) -> tuple[Any, ...]:
    def sortable(value: Any) -> tuple[int, Any]:
        if value is None:
            return (0, "")
        if isinstance(value, bool):
            return (1, int(value))
        if isinstance(value, (int, float)):
            return (2, float(value))
        return (3, str(value))

    return tuple(
        (str(key), *sortable(value))
        for key, value in sorted(
            parameters.items(), key=lambda item: str(item[0])
        )
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    materialized = [dict(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(materialized[0])
    for row in materialized[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(materialized)


def _coldstart_result_row(
    cell: CellLike,
    value: Mapping[str, Any],
    *,
    source: str,
    method_columns: Mapping[str, Any],
    require_late_window_metrics: bool,
) -> dict[str, Any]:
    if require_late_window_metrics and (
        "validation_late_window_pass8_mean" not in value
        or "validation_late_window_greedy_mean" not in value
    ):
        raise RuntimeError(
            f"{cell.key} is missing the paper primary late-window metric"
        )
    return common_result_row(
        cell,
        source=source,
        method_columns=method_columns,
        metrics={
            "nan_inf_failure": bool(value["nan_inf_failure"]),
            "late_window_pass8_mean": value.get(
                "validation_late_window_pass8_mean",
                value["validation_best_pass8"],
            ),
            "late_window_greedy_mean": value.get(
                "validation_late_window_greedy_mean",
                value["validation_best_greedy"],
            ),
            "best_pass8": value["validation_best_pass8"],
            "terminal_pass8": value["validation_terminal_pass8"],
            "best_greedy": value["validation_best_greedy"],
            "terminal_greedy": value["validation_terminal_greedy"],
            "best_greedy_valid_rate": value[
                "validation_best_greedy_valid_rate"
            ],
            "terminal_greedy_valid_rate": value[
                "validation_terminal_greedy_valid_rate"
            ],
            "best_step": value["best_step"],
            "terminal_step": value["terminal_step"],
            "stop_reason": value["stop_reason"],
        },
    )





def coldstart_completed_task_rows(
    output_root: Path,
    *,
    task: str,
    expected_cells: Sequence[CellLike],
    experiment_id_value: str,
    config_hash: str,
    result_row_fn: Callable[
        [CellLike, Mapping[str, Any]], Mapping[str, Any]
    ],
) -> list[dict[str, Any]] | None:
    """Return task-local response rows only after every expected cell completes."""

    if not expected_cells:
        return None
    rows: list[dict[str, Any]] = []
    for cell in expected_cells:
        path = output_root / "cells" / cell.key / "cell_manifest.json"
        if not path.is_file():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        if (
            value.get("complete") is not True
            or value.get("evaluation_status") != "complete"
        ):
            return None
        if (
            value.get("experiment_id") != experiment_id_value
            or value.get("config_hash") != config_hash
        ):
            raise RuntimeError(
                f"{cell.key} per-task result identity mismatch"
            )
        rows.append(dict(result_row_fn(cell, value)))
    return rows


def materialize_completed_task_results(
    tasks: Sequence[str],
    *,
    completed_rows_fn: Callable[[str], list[dict[str, Any]] | None],
    write_task_result_fn: Callable[
        [str, list[dict[str, Any]]], dict[str, Any]
    ],
) -> dict[str, dict[str, Any]]:
    """Publish each fully complete task through one result path."""

    ready: dict[str, dict[str, Any]] = {}
    for task_value in tasks:
        task = str(task_value)
        rows = completed_rows_fn(task)
        if rows is not None:
            ready[task] = write_task_result_fn(task, rows)
    return ready


def _write_coldstart_task_result(
    config: Mapping[str, Any],
    output_root: Path,
    task: str,
    rows: list[dict[str, Any]],
    *,
    configured_cells: Sequence[CellLike],
    method_specs: Mapping[str, MethodSpecLike],
    experiment_id_value: str,
    config_hash: str,
    engineering_self_test: bool,
    write_json: Callable[[Path, Any], None],
    sha256_fn: Callable[[Path], str],
) -> dict[str, Any]:
    """Publish deterministic task-local CSVs and write TASK_COMPLETE.json last."""

    provenance_path = output_root / "source_provenance.json"
    if not provenance_path.is_file():
        raise RuntimeError("Per-task result materialization requires source_provenance.json")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    source_commit = str(provenance.get("source_commit", ""))
    if len(source_commit) != 40 or any(
        character not in "0123456789abcdef" for character in source_commit
    ):
        raise RuntimeError("Per-task result materialization requires one full source commit")
    run_id = str(provenance.get("run_id", output_root.name))
    root = output_root / "task_results" / task
    root.mkdir(parents=True, exist_ok=True)
    marker_path = root / "TASK_COMPLETE.json"
    marker_path.unlink(missing_ok=True)

    all_cells_path = root / "all_cells.csv"
    plot_path = root / "plot_curve_points.csv"
    write_csv(all_cells_path, rows)
    cells_by_key = {
        cell.key: cell for cell in configured_cells if cell.task == task
    }
    legacy_parameter_columns = (
        "delta_v", "beta", "dpo_initialization", "lambda", "rho"
    )
    extra_parameter_names = tuple(
        dict.fromkeys(
            name
            for cell in cells_by_key.values()
            for name in method_specs[cell.method].parameters(cell)
            if name not in legacy_parameter_columns
        )
    )
    plot_rows: list[dict[str, Any]] = []
    for row in rows:
        cell = cells_by_key[str(row["cell_key"])]
        parameters = dict(method_specs[cell.method].parameters(cell))
        plot_rows.append(
            {
                "experiment_id": experiment_id_value,
                "run_id": run_id,
                "source_commit": source_commit,
                "task": row["task"],
                "method": row["method"],
                "delta_v": row.get("delta_v"),
                "beta": row.get("beta"),
                "dpo_initialization": row.get("dpo_initialization"),
                "lambda": row["lambda"],
                "rho": row["rho"],
                **{name: parameters.get(name) for name in extra_parameter_names},
                "seed": row["seed"],
                "stage": row["stage"],
                **_coldstart_plot_metrics(row),
            }
        )
    write_csv(plot_path, plot_rows)
    cell_manifest_sha256 = {
        row["cell_key"]: sha256_fn(
            output_root / "cells" / str(row["cell_key"]) / "cell_manifest.json"
        )
        for row in rows
    }
    marker = {
        "schema_version": 1,
        "experiment_id": experiment_id_value,
        "config_hash": config_hash,
        "run_id": run_id,
        "source_commit": source_commit,
        "task": task,
        "expected_cells": len(rows),
        "cell_count": len(rows),
        "all_cells_csv": f"task_results/{task}/all_cells.csv",
        "all_cells_csv_sha256": sha256_fn(all_cells_path),
        "plot_curve_points_csv": f"task_results/{task}/plot_curve_points.csv",
        "plot_curve_points_csv_sha256": sha256_fn(plot_path),
        "cell_manifest_sha256": cell_manifest_sha256,
        "analysis_ready": True,
        "final_aggregate_authority": False,
        "test_partition_accessed": False,
        "method_ranking_allowed": False,
        "complete": True,
        "scientific_status": ("not_run" if engineering_self_test else "pilot"),
        "note": (
            "Deterministic early task snapshot from completed identity-matched cells; "
            "the terminal aggregate remains the final reporting authority."
        ),
    }
    write_json(marker_path, marker)
    return marker



def _aggregate_dense(
    config: Mapping[str, Any],
    output_root: Path,
    rows: list[dict[str, Any]],
    *,
    experiment_id_value: str,
    config_hash: str,
    task_lambdas_fn: Callable[[Mapping[str, Any], str], Sequence[float]],
    positive_only_method: str,
    exponential_method: str,
    write_json: Callable[[Path, Any], None],
) -> dict[str, Any]:
    parent_path = output_root / "inherited" / "parent_response.json"
    if not parent_path.is_file():
        raise FileNotFoundError("Dense aggregation requires inherited parent_response.json")
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if (
        parent.get("experiment_id") != experiment_id_value
        or parent.get("config_hash") != config_hash
        or not parent.get("complete")
    ):
        raise RuntimeError("Inherited parent response identity mismatch")
    parent_rows = list(parent.get("rows", ()))
    if len(parent_rows) != len(config["suite"]["tasks"]) * 8:
        raise RuntimeError("Inherited parent response does not contain eight anchors per task")
    combined_rows = parent_rows + rows
    write_csv(output_root / "aggregate" / "combined_response.csv", combined_rows)

    task_summaries: dict[str, Any] = {}
    selected_rows: list[dict[str, Any]] = []
    minimum_valid = float(config["selection"]["terminal_valid_rate_minimum"])
    for task_value in config["suite"]["tasks"]:
        task = str(task_value)
        task_dense = [row for row in rows if row["task"] == task]
        task_parent = [row for row in parent_rows if row["task"] == task]
        positive_rows = [row for row in task_parent if row["method"] == positive_only_method]
        parent_exp = [row for row in task_parent if row["method"] == exponential_method]
        if len(task_dense) != 16 or len(positive_rows) != 1 or len(parent_exp) != 7:
            raise RuntimeError(f"{task} dense/predecessor response geometry is incomplete")
        eligible = [
            row
            for row in task_dense
            if not row["nan_inf_failure"]
            and float(row["terminal_greedy_valid_rate"]) >= minimum_valid
        ]
        selected = (
            max(
                eligible,
                key=lambda row: (
                    float(row["late_window_pass8_mean"]),
                    float(row["terminal_pass8"]),
                    float(row["late_window_greedy_mean"]),
                    float(row["terminal_greedy"]),
                    -float(row["lambda"]),
                ),
            )
            if eligible
            else None
        )
        positive = positive_rows[0]
        best_observed = max(
            task_dense,
            key=lambda row: float(row["late_window_pass8_mean"]),
        )
        task_lambdas = task_lambdas_fn(config, task)
        bridge_lambda = float(config["sweep"]["bridge_lambda"][task])
        bridge_dense = next(
            row
            for row in task_dense
            if math.isclose(
                float(row["lambda"]),
                bridge_lambda,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        )
        bridge_parent = next(
            row
            for row in parent_exp
            if math.isclose(
                float(row["rho"]),
                math.exp(-bridge_lambda),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        )
        selected_lambda = None if selected is None else float(selected["lambda"])
        selected_on_grid_edge = bool(
            selected is not None
            and (
                math.isclose(selected_lambda, min(task_lambdas))
                or math.isclose(selected_lambda, max(task_lambdas))
            )
        )
        strong_boundary_unclosed = bool(
            selected is not None and math.isclose(selected_lambda, max(task_lambdas))
        )
        summary = {
            "task": task,
            "task_role": config["sweep"]["task_role"][task],
            "positive_only": positive,
            "eligible_dense_count": len(eligible),
            "selected_dense_exp": selected,
            "selected_on_grid_edge": selected_on_grid_edge,
            "strong_taper_boundary_unclosed": strong_boundary_unclosed,
            "all_dense_below_positive_only": float(best_observed["late_window_pass8_mean"])
            < float(positive["late_window_pass8_mean"]),
            "bridge": {
                "lambda": bridge_lambda,
                "rho": math.exp(-bridge_lambda),
                "parent": bridge_parent,
                "dense_rerun": bridge_dense,
                "late_window_pass8_delta": float(bridge_dense["late_window_pass8_mean"])
                - float(bridge_parent["late_window_pass8_mean"]),
                "terminal_pass8_delta": float(bridge_dense["terminal_pass8"])
                - float(bridge_parent["terminal_pass8"]),
                "late_window_greedy_delta": float(bridge_dense["late_window_greedy_mean"])
                - float(bridge_parent["late_window_greedy_mean"]),
                "report_only": True,
            },
            "selection_metric": config["selection"]["primary_metric"],
        }
        task_summaries[task] = summary
        selected_rows.append(
            {
                "task": task,
                "task_role": summary["task_role"],
                "selected_lambda": selected_lambda,
                "selected_rho": None if selected is None else selected["rho"],
                "selected_late_window_pass8_mean": (
                    None if selected is None else selected["late_window_pass8_mean"]
                ),
                "positive_only_late_window_pass8_mean": positive["late_window_pass8_mean"],
                "all_dense_below_positive_only": summary["all_dense_below_positive_only"],
                "selected_on_grid_edge": selected_on_grid_edge,
                "strong_taper_boundary_unclosed": strong_boundary_unclosed,
                "bridge_late_window_pass8_delta": summary["bridge"]["late_window_pass8_delta"],
            }
        )
    write_csv(output_root / "aggregate" / "selected_exp_by_task.csv", selected_rows)
    summary = {
        "schema_version": 1,
        "experiment_id": experiment_id_value,
        "cell_count": len(rows),
        "combined_response_point_count": len(combined_rows),
        "tasks": task_summaries,
        "excluded_tasks": dict(config["suite"]["excluded_tasks"]),
        "parent_run_id": config["parent"]["run_id"],
        "parent_result_commit": config["parent"]["result_commit"],
        "positive_only_source": "inherited_parent",
        "test_partition_accessed": False,
        "task_performance_reported_separately": True,
        "structure_diagnostic_reported_separately": True,
        "nan_inf_reported_separately": True,
        "single_seed_shape_discovery": True,
        "fresh_seed_confirmation_required": True,
        "fixed_horizon_is_convergence": False,
        "scientific_status": "pilot",
        "claim_boundary": (
            "Development response-shape refinement only; no significance, convergence, "
            "universal superiority, or categorical causal-identification claim."
        ),
    }
    write_json(output_root / "aggregate" / "aggregate_summary.json", summary)
    return summary



def _coldstart_plot_metrics(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "late_window_pass8_mean": row["late_window_pass8_mean"],
        "late_window_greedy_mean": row["late_window_greedy_mean"],
        "best_validation_pass8": row["best_pass8"],
        "terminal_pass8": row["terminal_pass8"],
        "best_validation_greedy": row["best_greedy"],
        "terminal_greedy": row["terminal_greedy"],
        "best_greedy_valid_rate": row["best_greedy_valid_rate"],
        "terminal_greedy_valid_rate": row["terminal_greedy_valid_rate"],
        "best_step": row["best_step"],
        "terminal_step": row["terminal_step"],
        "stop_reason": row["stop_reason"],
        "nan_inf_failure": row["nan_inf_failure"],
        "complete": True,
    }



def _coldstart_group_metrics(group: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def mean(key: str) -> float:
        return float(np.mean([float(row[key]) for row in group]))

    return {
        "seeds": sorted(int(row["seed"]) for row in group),
        "late_window_pass8_mean": mean("late_window_pass8_mean"),
        "late_window_greedy_mean": mean("late_window_greedy_mean"),
        "terminal_pass8_mean": mean("terminal_pass8"),
        "terminal_greedy_valid_rate_mean": mean("terminal_greedy_valid_rate"),
        "nan_inf_failure": any(bool(row["nan_inf_failure"]) for row in group),
    }



def _coldstart_run_provenance(output_root: Path) -> tuple[str, str]:
    path = output_root / "source_provenance.json"
    value = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    return (
        str(value.get("run_id", output_root.name)),
        str(value.get("source_commit", "unrecorded")),
    )



def _coldstart_method_grouped_curve(
    *,
    task: str,
    method: str,
    method_rows: Sequence[Mapping[str, Any]],
    cells_by_key: Mapping[str, CellLike],
    parameter_fn: Callable[[CellLike], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Group one method through its registered opaque parameter contract."""

    groups: dict[
        str,
        tuple[dict[str, Any], list[Mapping[str, Any]]],
    ] = {}
    for row in method_rows:
        cell_key = str(row["cell_key"])
        if cell_key not in cells_by_key:
            raise RuntimeError(f"Unknown cold-start cell in aggregate: {cell_key}")
        cell = cells_by_key[cell_key]
        if cell.task != task or cell.method != method:
            raise RuntimeError(
                f"Cold-start aggregate identity mismatch for {cell_key}"
            )
        parameters = dict(parameter_fn(cell))
        identity = parameter_identity(parameters)
        if identity not in groups:
            groups[identity] = (parameters, [])
        elif groups[identity][0] != parameters:
            raise RuntimeError(f"Method parameter identity changed for {cell_key}")
        groups[identity][1].append(row)

    ordered = sorted(
        groups.values(),
        key=lambda item: parameter_sort_key(item[0]),
    )
    return [
        {
            "task": task,
            "method": method,
            **parameters,
            **_coldstart_group_metrics(group),
        }
        for parameters, group in ordered
    ]



def _aggregate_coldstart_unranked(
    config: Mapping[str, Any],
    output_root: Path,
    rows: list[dict[str, Any]],
    *,
    spec: MethodSpecLike,
    configured_cells: Sequence[CellLike],
    experiment_id_value: str,
    protocol_diagnostic: Mapping[str, Any],
    engineering_self_test: bool,
    positive_only_method: str,
    global_method: str,
    write_json: Callable[[Path, Any], None],
) -> dict[str, Any]:
    """Aggregate one registered non-Exp curve without selection or ranking."""

    method = spec.name
    cells_by_key = {cell.key: cell for cell in configured_cells}
    if len(cells_by_key) != len(configured_cells):
        raise RuntimeError("Cold-start aggregate contains duplicate cell keys")

    run_id, source_commit = _coldstart_run_provenance(output_root)
    plot_rows: list[dict[str, Any]] = []
    for row in rows:
        cell_key = str(row["cell_key"])
        if cell_key not in cells_by_key:
            raise RuntimeError(f"Unknown cold-start cell in plot aggregate: {cell_key}")
        projection = dict(spec.parameters(cells_by_key[cell_key]))
        parameter_columns = {
            name: value
            for name, value in projection.items()
            if name != "dpo_initialization"
        }
        plot_rows.append(
            {
                "task": row["task"],
                "method": row["method"],
                "dpo_initialization": row["dpo_initialization"],
                "seed": row["seed"],
                "stage": row["stage"],
                "experiment_id": experiment_id_value,
                "run_id": run_id,
                "source_commit": source_commit,
                **parameter_columns,
                **_coldstart_plot_metrics(row),
            }
        )
    write_csv(output_root / "aggregate" / "plot_curve_points.csv", plot_rows)

    summaries: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    for task_value in config["suite"]["tasks"]:
        task = str(task_value)
        task_cells = [cell for cell in configured_cells if cell.task == task]
        if not task_cells:
            continue
        task_rows = [row for row in rows if row["task"] == task]
        controls = {
            control: [row for row in task_rows if row["method"] == control]
            for control in (positive_only_method, global_method)
        }
        method_rows = [row for row in task_rows if row["method"] == method]
        expected = {
            candidate: sum(cell.method == candidate for cell in task_cells)
            for candidate in (method, *controls)
        }
        if len(method_rows) != expected[method] or any(
            len(group) != expected[control]
            for control, group in controls.items()
        ):
            raise RuntimeError(
                f"{task} {method} cold-start cell geometry is incomplete"
            )

        grouped_curve = _coldstart_method_grouped_curve(
            task=task,
            method=method,
            method_rows=method_rows,
            cells_by_key=cells_by_key,
            parameter_fn=spec.parameters,
        )
        summaries[task] = {
            "task": task,
            "grouped_curve": grouped_curve,
            "positive_only": controls[positive_only_method] or None,
            "global": controls[global_method] or None,
            "parameter_selection_deferred_to_reviewed_protocol": True,
            "terminal_valid_rate_role": "diagnostic_only_not_selection_eligibility",
        }
        summary_rows.append(
            {
                "task": task,
                f"{method}_parameter_points": len(grouped_curve),
                "finite_parameter_points": sum(
                    not bool(row["nan_inf_failure"]) for row in grouped_curve
                ),
                "parameter_selection_deferred": True,
            }
        )
    write_csv(output_root / "aggregate" / "task_summary.csv", summary_rows)

    summary = {
        "schema_version": 1,
        "experiment_id": experiment_id_value,
        "run_id": run_id,
        "source_commit": source_commit,
        "cell_count": len(rows),
        "plot_curve_point_count": len(plot_rows),
        "method": method,
        "tasks": summaries,
        "excluded_tasks": dict(config["suite"]["excluded_tasks"]),
        "initialization": dict(config["initialization"]),
        "scientific_kernel": spec.scientific_kernel,
        **dict(spec.single_aggregate_metadata(config)),
        "countdown_protocol_diagnostic": protocol_diagnostic,
        "countdown_result_gate": False,
        "primary_metric": "validation_late_window_pass8_mean",
        "parameter_selection_deferred_to_reviewed_protocol": True,
        "test_partition_accessed": False,
        "method_ranking_allowed": False,
        "significance_claim_allowed": False,
        "fixed_horizon_is_convergence": False,
        "task_performance_reported_separately": True,
        "structure_diagnostic_reported_separately": True,
        "nan_inf_reported_separately": True,
        "scientific_status": (
            "not_run" if engineering_self_test else "pilot"
        ),
        "engineering_placeholder_backend": engineering_self_test,
    }
    write_json(output_root / "aggregate" / "aggregate_summary.json", summary)
    return summary



def _aggregate_coldstart_matrix_unranked(
    config: Mapping[str, Any],
    output_root: Path,
    rows: list[dict[str, Any]],
    *,
    methods: Sequence[str],
    configured_cells: Sequence[CellLike],
    method_specs: Mapping[str, MethodSpecLike],
    experiment_id_value: str,
    matrix_method: str,
    execution_class: str,
    transfer_seed_offsets: Sequence[int],
    protocol_diagnostic: Mapping[str, Any],
    engineering_self_test: bool,
    write_json: Callable[[Path, Any], None],
) -> dict[str, Any]:
    """Aggregate a multi-method transfer matrix without selecting or ranking methods."""

    run_id, source_commit = _coldstart_run_provenance(output_root)
    cells_by_key = {cell.key: cell for cell in configured_cells}
    if len(cells_by_key) != len(configured_cells):
        raise RuntimeError("Baseline matrix contains duplicate cell keys")
    legacy_parameter_columns = ("delta_v", "beta", "dpo_initialization")
    extra_parameter_names = tuple(
        dict.fromkeys(
            name
            for cell in configured_cells
            if cell.method in methods
            for name in method_specs[cell.method].parameters(cell)
            if name not in legacy_parameter_columns
        )
    )
    plot_rows: list[dict[str, Any]] = []
    for row in rows:
        cell = cells_by_key[str(row["cell_key"])]
        parameters = dict(method_specs[cell.method].parameters(cell))
        plot_rows.append(
            {
                "task": row["task"],
                "method": row["method"],
                "delta_v": row.get("delta_v"),
                "beta": row.get("beta"),
                "dpo_initialization": row.get("dpo_initialization"),
                **{name: parameters.get(name) for name in extra_parameter_names},
                "seed": row["seed"],
                "stage": row["stage"],
                "experiment_id": experiment_id_value,
                "run_id": run_id,
                "source_commit": source_commit,
                **_coldstart_plot_metrics(row),
            }
        )
    write_csv(output_root / "aggregate" / "plot_curve_points.csv", plot_rows)
    task_summaries: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    for task_value in config["suite"]["p0_tasks"]:
        task = str(task_value)
        task_rows = [row for row in rows if row["task"] == task]
        task_cells = [cell for cell in configured_cells if cell.task == task]
        method_summaries: dict[str, Any] = {}
        summary_row: dict[str, Any] = {"task": task}
        for method in methods:
            method_rows = [row for row in task_rows if row["method"] == method]
            expected = sum(cell.method == method for cell in task_cells)
            if len(method_rows) != expected:
                raise RuntimeError(
                    f"{task} {method} baseline-matrix cell geometry is incomplete"
                )
            grouped_curve = _coldstart_method_grouped_curve(
                task=task,
                method=method,
                method_rows=method_rows,
                cells_by_key=cells_by_key,
                parameter_fn=method_specs[method].parameters,
            )
            method_summaries[method] = {
                "grouped_curve": grouped_curve,
                "parameter_selection_deferred_to_reviewed_protocol": True,
                "terminal_valid_rate_role": "diagnostic_only_not_selection_eligibility",
            }
            summary_row[f"{method}_parameter_points"] = len(grouped_curve)
        task_summaries[task] = {"task": task, "methods": method_summaries}
        summary_rows.append(summary_row)
    write_csv(output_root / "aggregate" / "task_summary.csv", summary_rows)

    method_metadata = {
        method: dict(method_specs[method].matrix_aggregate_metadata(config))
        for method in methods
    }
    summary = {
        "schema_version": 1,
        "experiment_id": experiment_id_value,
        "run_id": run_id,
        "source_commit": source_commit,
        "cell_count": len(rows),
        "plot_curve_point_count": len(plot_rows),
        "method": matrix_method,
        "methods": list(methods),
        "execution_class": execution_class,
        "method_metadata": method_metadata,
        "transfer_seed_offsets": list(transfer_seed_offsets),
        "tasks": task_summaries,
        "excluded_tasks": dict(config["suite"]["excluded_tasks"]),
        "countdown_protocol_diagnostic": protocol_diagnostic,
        "countdown_result_gate": False,
        "primary_metric": "validation_late_window_pass8_mean",
        "parameter_selection_deferred_to_reviewed_protocol": True,
        "method_ranking_allowed": False,
        "significance_claim_allowed": False,
        "test_partition_accessed": False,
        "fixed_horizon_is_convergence": False,
        "task_performance_reported_separately": True,
        "structure_diagnostic_reported_separately": True,
        "nan_inf_reported_separately": True,
        "scientific_status": (
            "not_run" if engineering_self_test else "pilot"
        ),
        "engineering_placeholder_backend": engineering_self_test,
    }
    write_json(output_root / "aggregate" / "aggregate_summary.json", summary)
    return summary



def _aggregate_coldstart_exponential(
    config: Mapping[str, Any],
    output_root: Path,
    rows: list[dict[str, Any]],
    *,
    configured_cells: Sequence[CellLike],
    experiment_id_value: str,
    protocol_diagnostic_value: Mapping[str, Any],
    engineering_self_test: bool,
    positive_only_method: str,
    global_method: str,
    exponential_method: str,
    write_json: Callable[[Path, Any], None],
) -> dict[str, Any]:
    run_id, source_commit = _coldstart_run_provenance(output_root)
    plot_rows = [
        {
            "experiment_id": experiment_id_value,
            "run_id": run_id,
            "source_commit": source_commit,
            "task": row["task"],
            "method": row["method"],
            "lambda": row["lambda"],
            "rho": row["rho"],
            "seed": row["seed"],
            "stage": row["stage"],
            **_coldstart_plot_metrics(row),
        }
        for row in rows
    ]
    write_csv(output_root / "aggregate" / "plot_curve_points.csv", plot_rows)

    summaries: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    for task_value in config["suite"]["tasks"]:
        task = str(task_value)
        task_rows = [row for row in rows if row["task"] == task]
        positive_rows = [row for row in task_rows if row["method"] == positive_only_method]
        global_rows = [row for row in task_rows if row["method"] == global_method]
        exp_rows = [row for row in task_rows if row["method"] == exponential_method]
        configured_methods = [cell.method for cell in configured_cells if cell.task == task]
        if not configured_methods:
            continue
        expected_counts = tuple(
            configured_methods.count(method)
            for method in (positive_only_method, global_method, exponential_method)
        )
        if (len(positive_rows), len(global_rows), len(exp_rows)) != expected_counts:
            raise RuntimeError(f"{task} cold-start cell counts differ from {expected_counts}")

        def aggregate_group(
            group: Sequence[Mapping[str, Any]], task_name: str = task
        ) -> dict[str, Any]:
            first = group[0]
            return {
                "task": task_name,
                "method": first["method"],
                "lambda": first["lambda"],
                "rho": first["rho"],
                **_coldstart_group_metrics(group),
                "best_pass8_mean": float(
                    np.mean([float(row["best_pass8"]) for row in group])
                ),
            }

        coefficient_groups: dict[tuple[str, float | None], list[dict[str, Any]]] = {}
        for row in task_rows:
            coefficient_groups.setdefault((str(row["method"]), row["lambda"]), []).append(row)
        grouped = [aggregate_group(group) for group in coefficient_groups.values()]
        positive = next((row for row in grouped if row["method"] == positive_only_method), None)
        positive_score = None if positive is None else float(positive["late_window_pass8_mean"])
        grouped_exp = [row for row in grouped if row["method"] == exponential_method]
        selectable = [row for row in grouped_exp if not row["nan_inf_failure"]]
        selected = (
            max(
                selectable,
                key=lambda row: (
                    float(row["late_window_pass8_mean"]),
                    float(row["terminal_pass8_mean"]),
                    float(row["late_window_greedy_mean"]),
                    -float(row["lambda"]),
                ),
            )
            if selectable
            else None
        )
        min_lambda = min(float(row["lambda"]) for row in grouped_exp)
        max_lambda = max(float(row["lambda"]) for row in grouped_exp)
        selected_on_edge = bool(
            selected is not None
            and (
                math.isclose(float(selected["lambda"]), min_lambda)
                or math.isclose(float(selected["lambda"]), max_lambda)
            )
        )
        task_summary = {
            "task": task,
            "positive_only": positive,
            "global": next((row for row in grouped if row["method"] == global_method), None),
            "selectable_exp_count": len(selectable),
            "selected_exp": selected,
            "selected_on_grid_edge": selected_on_edge,
            "terminal_valid_rate_role": "diagnostic_only_not_selection_eligibility",
            "all_exp_below_positive_only": None
            if positive_score is None
            else all(float(row["late_window_pass8_mean"]) < positive_score for row in grouped_exp),
            "grouped_curve": sorted(
                grouped,
                key=lambda row: (
                    row["method"] != positive_only_method,
                    -1.0 if row["lambda"] is None else float(row["lambda"]),
                ),
            ),
        }
        summaries[task] = task_summary
        summary_rows.append(
            {
                "task": task,
                "positive_only_late_window_pass8_mean": positive_score,
                "selected_lambda": None if selected is None else selected["lambda"],
                "selected_rho": None if selected is None else selected["rho"],
                "selected_late_window_pass8_mean": (
                    None if selected is None else selected["late_window_pass8_mean"]
                ),
                "selected_on_grid_edge": selected_on_edge,
                "all_exp_below_positive_only": task_summary["all_exp_below_positive_only"],
            }
        )
    write_csv(output_root / "aggregate" / "task_summary.csv", summary_rows)

    protocol_diagnostic = protocol_diagnostic_value
    summary = {
        "schema_version": 1,
        "experiment_id": experiment_id_value,
        "run_id": run_id,
        "source_commit": source_commit,
        "cell_count": len(rows),
        "plot_curve_point_count": len(plot_rows),
        "tasks": summaries,
        "excluded_tasks": dict(config["suite"]["excluded_tasks"]),
        "initialization": dict(config["initialization"]),
        "positive_only_and_exp_share_fresh_initialization": bool(
            tuple(config["sweep"].get("transfer_positive_only_seed_offsets", ()))
        ),
        "scientific_kernel": "canonical_old_coldstart_imports",
        "canonical_source_git_blob_shas": dict(
            config["canonical_coldstart"]["expected_git_blob_shas"]
        ),
        "countdown_protocol_diagnostic": protocol_diagnostic,
        "countdown_result_gate": False,
        "primary_metric": "validation_late_window_pass8_mean",
        "terminal_valid_rate_role": "diagnostic_only_not_selection_eligibility",
        "test_partition_accessed": False,
        "transfer_exp_single_seed_response_shape_localization": True,
        "transfer_positive_only_seed_count": len(
            tuple(
                int(value)
                for value in config["sweep"].get("transfer_positive_only_seed_offsets", ())
            )
        ),
        "fresh_seed_confirmation_required_for_winner_claim": True,
        "method_ranking_allowed": False,
        "significance_claim_allowed": False,
        "fixed_horizon_is_convergence": False,
        "task_performance_reported_separately": True,
        "structure_diagnostic_reported_separately": True,
        "nan_inf_reported_separately": True,
        "scientific_status": "not_run" if engineering_self_test else "pilot",
        "engineering_placeholder_backend": engineering_self_test,
    }
    write_json(output_root / "aggregate" / "aggregate_summary.json", summary)
    return summary

# Artifact/package ownership
PACKAGE_REQUIRED_MEMBERS = {
    "RUN_COMPLETE.json",
    "run_manifest.json",
    "scientific_run_manifest.json",
    "source_provenance.json",
    "terminal_audit.json",
    "scheduler/dynamic_run.json",
    "aggregate/plot_curve_points.csv",
    "package_contents_manifest.json",
    "SHA256SUMS.txt",
}






def aggregate_coldstart_dispatch(
    rows: list[dict[str, Any]],
    *,
    method_matrix: bool,
    method: str,
    exponential_method: str,
    matrix_fn: Callable[[list[dict[str, Any]]], dict[str, Any]],
    unranked_fn: Callable[[list[dict[str, Any]]], dict[str, Any]],
    exponential_fn: Callable[[list[dict[str, Any]]], dict[str, Any]],
) -> dict[str, Any]:
    """Dispatch cold-start aggregation without owning method semantics."""

    if method_matrix:
        return matrix_fn(rows)
    if method != exponential_method:
        return unranked_fn(rows)
    return exponential_fn(rows)


def cmd_aggregate(
    config: Mapping[str, Any],
    output_root: Path,
    *,
    cells: Sequence[CellLike],
    experiment_id_value: str,
    dense_profile: bool,
    coldstart_profile: bool,
    method_columns_fn: Callable[[CellLike], Mapping[str, Any]],
    coldstart_result_row_fn: Callable[
        [CellLike, Mapping[str, Any], str], Mapping[str, Any]
    ],
    coldstart_aggregate_fn: Callable[
        [list[dict[str, Any]]], dict[str, Any]
    ],
    dense_aggregate_fn: Callable[[list[dict[str, Any]]], dict[str, Any]],
    positive_only_method: str,
    exponential_method: str,
    write_json: Callable[[Path, Any], None],
) -> dict[str, Any]:
    """Materialize result rows and dispatch to the profile-specific aggregate."""

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for cell in cells:
        path = output_root / "cells" / cell.key / "cell_manifest.json"
        if not path.is_file():
            missing.append(cell.key)
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        if not value.get("complete") or value.get("evaluation_status") != "complete":
            missing.append(cell.key)
            continue
        source = "dense" if dense_profile else "current"
        if coldstart_profile:
            rows.append(
                dict(coldstart_result_row_fn(cell, value, source))
            )
            continue
        rows.append(
            common_result_row(
                cell,
                source=source,
                method_columns=method_columns_fn(cell),
                metrics={
                    "nan_inf_failure": bool(value["nan_inf_failure"]),
                    "late_window_pass8_mean": value[
                        "validation_late_window_pass8_mean"
                    ],
                    "terminal_pass8": value["validation_terminal_pass8"],
                    "late_window_greedy_mean": value[
                        "validation_late_window_greedy_mean"
                    ],
                    "terminal_greedy": value["validation_terminal_greedy"],
                    "terminal_greedy_valid_rate": value[
                        "validation_terminal_greedy_valid_rate"
                    ],
                },
            )
        )

    if missing:
        raise RuntimeError(
            f"Cannot aggregate; missing/incomplete cells: {missing}"
        )
    write_csv(output_root / "aggregate" / "all_cells.csv", rows)
    if coldstart_profile:
        return coldstart_aggregate_fn(rows)
    if dense_profile:
        return dense_aggregate_fn(rows)

    task_summaries: dict[str, Any] = {}
    selected_rows: list[dict[str, Any]] = []
    minimum_valid = float(
        config["selection"]["terminal_valid_rate_minimum"]
    )
    boundary_rho = float(config["selection"]["boundary_rho"])
    for task_value in config["suite"]["tasks"]:
        task = str(task_value)
        task_rows = [row for row in rows if row["task"] == task]
        positive_rows = [
            row for row in task_rows
            if row["method"] == positive_only_method
        ]
        exp_rows = [
            row for row in task_rows
            if row["method"] == exponential_method
        ]
        if len(positive_rows) != 1 or len(exp_rows) != 7:
            raise RuntimeError(
                f"{task} does not contain one Positive-only and seven Exp cells"
            )
        eligible = [
            row
            for row in exp_rows
            if not row["nan_inf_failure"]
            and float(row["terminal_greedy_valid_rate"]) >= minimum_valid
        ]
        selected = (
            max(
                eligible,
                key=lambda row: (
                    float(row["late_window_pass8_mean"]),
                    float(row["terminal_pass8"]),
                    float(row["late_window_greedy_mean"]),
                    float(row["terminal_greedy"]),
                    float(row["rho"]),
                ),
            )
            if eligible
            else None
        )
        positive = positive_rows[0]
        best_observed = max(
            exp_rows,
            key=lambda row: float(row["late_window_pass8_mean"]),
        )
        summary = {
            "task": task,
            "positive_only": positive,
            "eligible_exp_count": len(eligible),
            "selected_exp": selected,
            "all_exp_below_positive_only": (
                float(best_observed["late_window_pass8_mean"])
                < float(positive["late_window_pass8_mean"])
            ),
            "strong_taper_boundary_unclosed": bool(
                selected is not None
                and math.isclose(float(selected["rho"]), boundary_rho)
            ),
            "selection_metric": config["selection"]["primary_metric"],
        }
        task_summaries[task] = summary
        selected_rows.append(
            {
                "task": task,
                "selected_rho": (
                    None if selected is None else selected["rho"]
                ),
                "selected_late_window_pass8_mean": (
                    None
                    if selected is None
                    else selected["late_window_pass8_mean"]
                ),
                "positive_only_late_window_pass8_mean": positive[
                    "late_window_pass8_mean"
                ],
                "all_exp_below_positive_only": summary[
                    "all_exp_below_positive_only"
                ],
                "strong_taper_boundary_unclosed": summary[
                    "strong_taper_boundary_unclosed"
                ],
            }
        )
    write_csv(
        output_root / "aggregate" / "selected_exp_by_task.csv",
        selected_rows,
    )
    summary = {
        "schema_version": 1,
        "experiment_id": experiment_id_value,
        "cell_count": len(rows),
        "tasks": task_summaries,
        "test_partition_accessed": False,
        "task_performance_reported_separately": True,
        "structure_diagnostic_reported_separately": True,
        "nan_inf_reported_separately": True,
        "fixed_horizon_is_convergence": False,
        "scientific_status": "pilot",
        "claim_boundary": (
            "Development hyperparameter response only; no significance, "
            "convergence, cross-task method ranking, or categorical "
            "causal-identification claim."
        ),
    }
    write_json(output_root / "aggregate" / "aggregate_summary.json", summary)
    return summary


def _write_completion_manifests(
    output_root: Path,
    audit: Mapping[str, Any],
    *,
    experiment_id_value: str,
    config_hash: str,
    expected_cells: int,
    engineering_self_test: bool,
    execution_class: str,
    write_json: Callable[[Path, Any], None],
    sha256_fn: Callable[[Path], str],
) -> None:
    provenance_path = output_root / "source_provenance.json"
    scheduler_path = output_root / "scheduler" / "dynamic_run.json"
    aggregate_path = output_root / "aggregate" / "aggregate_summary.json"
    if (
        not provenance_path.is_file()
        or not scheduler_path.is_file()
        or not aggregate_path.is_file()
    ):
        raise RuntimeError("Source provenance, scheduler result, and aggregate are required")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    scheduler = json.loads(scheduler_path.read_text(encoding="utf-8"))
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    source_commit = str(provenance.get("source_commit", ""))
    if len(source_commit) != 40 or any(
        char not in "0123456789abcdef" for char in source_commit
    ):
        raise RuntimeError("source_provenance.json must contain one full lowercase Git SHA")
    if (
        scheduler.get("experiment_id") != experiment_id_value
        or not scheduler.get("complete")
        or int(scheduler.get("expected_cells", 0)) != expected_cells
        or int(scheduler.get("completed_cells", 0)) != expected_cells
        or int(aggregate.get("cell_count", 0)) != expected_cells
    ):
        raise RuntimeError("Scheduler or aggregate is not terminal-complete")
    run_manifest = {
        "schema_version": 1,
        "experiment_id": experiment_id_value,
        "base_commit": source_commit,
        "run_id": str(provenance.get("run_id", output_root.name)),
        "source_commit": source_commit,
        "config_hash": config_hash,
        "expected_cells": expected_cells,
        "completed_cells": expected_cells,
        "scheduler": "dynamic_slot_queue",
        "scheduler_run_id": scheduler["scheduler_run_id"],
        "test_partition_accessed": False,
        "engineering_placeholder_backend": engineering_self_test,
        "execution_class": execution_class,
        "scientific_status": str(audit["scientific_status"]),
        "artifact_state": (
            "engineering_self_test_complete" if engineering_self_test else "raw_complete"
        ),
    }
    write_json(output_root / "run_manifest.json", run_manifest)
    write_json(output_root / "scientific_run_manifest.json", run_manifest)
    write_json(
        output_root / "RUN_COMPLETE.json",
        {
            **run_manifest,
            "all_training_and_evaluation_complete": bool(
                audit["all_training_and_evaluation_complete"]
            ),
            "terminal_audit_sha256": sha256_fn(output_root / "terminal_audit.json"),
            "aggregate_sha256": sha256_fn(aggregate_path),
            "complete": True,
        },
    )


def _result_payload_paths(output_root: Path, excluded_parts: set[str]) -> list[Path]:
    paths: list[Path] = []
    for path in sorted(output_root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(output_root)
        if path.is_symlink():
            raise RuntimeError(f"Result package refuses symlink payload: {relative}")
        if "packages" in relative.parts or relative.as_posix() in {
            "package_contents_manifest.json",
            "SHA256SUMS.txt",
        }:
            continue
        if any(part in excluded_parts for part in relative.parts) or path.suffix in {
            ".bin",
            ".safetensors",
        }:
            continue
        paths.append(path)
    return paths


def verify_result_package(
    package_manifest_path: Path,
    *,
    sha256_fn: Callable[[Path], str],
    zip_override: Path | None = None,
) -> dict[str, Any]:
    """Reopen a result ZIP and verify paths, inventory, hashes, and required members."""

    manifest = json.loads(package_manifest_path.read_text(encoding="utf-8"))
    zip_path = (zip_override or Path(str(manifest["full_results_zip"]))).resolve()
    if not zip_path.is_file():
        raise FileNotFoundError(f"Result ZIP is missing: {zip_path}")
    observed_zip_sha = sha256_fn(zip_path)
    if observed_zip_sha != manifest["full_results_zip_sha256"]:
        raise RuntimeError("Result ZIP SHA-256 does not match package_manifest.json")
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise RuntimeError("Result ZIP contains duplicate members")
        for name in names:
            member = PurePosixPath(name)
            if member.is_absolute() or ".." in member.parts or not member.parts:
                raise RuntimeError(f"Unsafe result ZIP member: {name}")
        missing = sorted(PACKAGE_REQUIRED_MEMBERS - set(names))
        if missing:
            raise RuntimeError(f"Result ZIP is missing required members: {missing}")
        contents = json.loads(archive.read("package_contents_manifest.json"))
        inventory = {
            str(item["path"]): str(item["sha256"])
            for item in contents.get("files", ())
        }
        expected_names = set(inventory) | {
            "package_contents_manifest.json",
            "SHA256SUMS.txt",
        }
        if set(names) != expected_names:
            raise RuntimeError(
                "Result ZIP members do not match package_contents_manifest.json"
            )
        for name, expected_sha in inventory.items():
            observed = hashlib.sha256(archive.read(name)).hexdigest()
            if observed != expected_sha:
                raise RuntimeError(f"Result ZIP payload hash mismatch: {name}")
        checksum_rows: dict[str, str] = {}
        for line in archive.read("SHA256SUMS.txt").decode("utf-8").splitlines():
            digest, separator, name = line.partition("  ")
            if not separator or name in checksum_rows:
                raise RuntimeError("Malformed or duplicate SHA256SUMS.txt entry")
            checksum_rows[name] = digest
        if checksum_rows != inventory:
            raise RuntimeError("SHA256SUMS.txt does not match the package inventory")
        if not any(name.startswith("logs/") for name in names):
            raise RuntimeError("Result ZIP contains no execution logs")
    return {
        "verified": True,
        "zip": str(zip_path),
        "zip_sha256": observed_zip_sha,
        "member_count": len(names),
        "required_members_present": True,
    }


def cmd_package(
    output_root: Path,
    *,
    experiment_id_value: str,
    execution_class: str,
    config_hash: str,
    expected_cells: int,
    engineering_self_test: bool,
    write_json: Callable[[Path, Any], None],
    sha256_fn: Callable[[Path], str],
) -> dict[str, Any]:
    """Create and independently reopen a portable text-first result ZIP."""

    audit_path = output_root / "terminal_audit.json"
    plot_path = output_root / "aggregate" / "plot_curve_points.csv"
    if not audit_path.is_file() or not plot_path.is_file():
        raise RuntimeError("Run aggregate and audit before packaging")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit.get("all_training_and_evaluation_complete"):
        raise RuntimeError("Refusing to package a non-terminal run")
    _write_completion_manifests(
        output_root,
        audit,
        experiment_id_value=experiment_id_value,
        config_hash=config_hash,
        expected_cells=expected_cells,
        engineering_self_test=engineering_self_test,
        execution_class=execution_class,
        write_json=write_json,
        sha256_fn=sha256_fn,
    )
    package_root = output_root / "packages"
    package_root.mkdir(parents=True, exist_ok=True)
    zip_path = package_root / f"{output_root.name}_full_results.zip"
    excluded_parts = {
        "best_adapter",
        "terminal_adapter",
        "last_finite_adapter",
        "supplementary_best_adapter",
        "initial_adapter",
    }
    payload_paths = _result_payload_paths(output_root, excluded_parts)
    inventory = [
        {
            "path": path.relative_to(output_root).as_posix(),
            "size": path.stat().st_size,
            "sha256": sha256_fn(path),
        }
        for path in payload_paths
    ]
    write_json(
        output_root / "package_contents_manifest.json",
        {
            "schema_version": 1,
            "experiment_id": experiment_id_value,
            "engineering_placeholder_backend": engineering_self_test,
            "files": inventory,
        },
    )
    (output_root / "SHA256SUMS.txt").write_text(
        "".join(f"{item['sha256']}  {item['path']}\n" for item in inventory),
        encoding="utf-8",
    )
    payload_paths.extend(
        [
            output_root / "package_contents_manifest.json",
            output_root / "SHA256SUMS.txt",
        ]
    )
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in payload_paths:
            archive.write(path, arcname=path.relative_to(output_root).as_posix())
    result = {
        "schema_version": 1,
        "experiment_id": experiment_id_value,
        "execution_class": execution_class,
        "scientific_status": str(audit["scientific_status"]),
        "artifact_kind": (
            "engineering_self_test"
            if engineering_self_test
            else ("formal_results" if execution_class == "formal" else "pilot_results")
        ),
        "full_results_zip": str(zip_path.resolve()),
        "full_results_zip_sha256": sha256_fn(zip_path),
        "full_results_zip_bytes": zip_path.stat().st_size,
        "plot_curve_points_csv": str(plot_path.resolve()),
        "plot_curve_points_csv_sha256": sha256_fn(plot_path),
        "included_file_count": len(payload_paths),
        "excluded_model_weights": sorted(excluded_parts),
        "complete": True,
    }
    manifest_path = package_root / "package_manifest.json"
    write_json(manifest_path, result)
    verification = verify_result_package(
        manifest_path,
        sha256_fn=sha256_fn,
    )
    result["reopen_verification"] = verification
    write_json(manifest_path, result)
    return result


def cmd_finalize(
    output_root: Path,
    *,
    experiment_id_value: str,
    execution_class: str,
    config_hash: str,
    expected_cells: int,
    engineering_self_test: bool,
    write_json: Callable[[Path, Any], None],
    sha256_fn: Callable[[Path], str],
) -> dict[str, Any]:
    """Finalize result markers while leaving archive ownership to the hardened guard."""

    audit_path = output_root / "terminal_audit.json"
    plot_path = output_root / "aggregate" / "plot_curve_points.csv"
    if not audit_path.is_file() or not plot_path.is_file():
        raise RuntimeError("Run aggregate and audit before finalizing")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit.get("all_training_and_evaluation_complete"):
        raise RuntimeError("Refusing to finalize a non-terminal run")
    _write_completion_manifests(
        output_root,
        audit,
        experiment_id_value=experiment_id_value,
        config_hash=config_hash,
        expected_cells=expected_cells,
        engineering_self_test=engineering_self_test,
        execution_class=execution_class,
        write_json=write_json,
        sha256_fn=sha256_fn,
    )
    return {
        "schema_version": 1,
        "experiment_id": experiment_id_value,
        "base_commit": audit["base_commit"],
        "artifact_state": "raw_complete",
        "execution_class": execution_class,
        "scientific_status": str(audit["scientific_status"]),
        "canonical_archive_owner": "scripts/run_experiment_guard_hardened.py",
        "plot_curve_points_csv": str(plot_path.resolve()),
        "plot_curve_points_csv_sha256": sha256_fn(plot_path),
        "complete": True,
    }

