"""Method-agnostic result projection and aggregation helpers for E8.

Concrete methods own their parameter projection.  This module only handles
common cell identity, common metric payloads, deterministic grouping, and CSV
materialization; it must not acquire branches for individual E8 algorithms.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np


class CellLike(Protocol):
    @property
    def key(self) -> str: ...

    task: str
    method: str
    seed: int
    stage: str


@dataclass(frozen=True)
class MethodResultProjection:
    """Method-owned result metadata exposed to generic aggregation."""

    parameters: Mapping[str, Any]
    compatibility_columns: Mapping[str, Any]


def common_result_row(
    cell: CellLike,
    value: Mapping[str, Any],
    *,
    source: str,
    project_method: Callable[[CellLike], MethodResultProjection],
    project_metrics: Callable[[Mapping[str, Any]], Mapping[str, Any]],
) -> dict[str, Any]:
    """Build one generic row without interpreting method-specific parameters."""

    projection = project_method(cell)
    return {
        "source": source,
        "task": cell.task,
        "method": cell.method,
        "seed": cell.seed,
        "stage": cell.stage,
        "cell_key": cell.key,
        "method_parameters": dict(projection.parameters),
        **dict(projection.compatibility_columns),
        **dict(project_metrics(value)),
    }


def parameter_identity(parameters: Mapping[str, Any]) -> str:
    """Stable JSON identity used only for generic grouping."""

    return json.dumps(dict(parameters), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def group_parameter_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, str], list[Mapping[str, Any]]]:
    """Group by task, method and opaque method-parameter identity."""

    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        parameters = row.get("method_parameters", {})
        if not isinstance(parameters, Mapping):
            raise TypeError("method_parameters must be a mapping")
        key = (str(row["task"]), str(row["method"]), parameter_identity(parameters))
        grouped.setdefault(key, []).append(row)
    return grouped


def mean_metrics(
    rows: Sequence[Mapping[str, Any]],
    metric_names: Sequence[str],
) -> dict[str, float]:
    if not rows:
        raise ValueError("Cannot aggregate an empty result group")
    return {
        f"{name}_mean": float(np.mean([float(row[name]) for row in rows]))
        for name in metric_names
    }


def grouped_curve(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric_names: Sequence[str],
) -> list[dict[str, Any]]:
    """Create deterministic method curves without knowing parameter names."""

    grouped = group_parameter_rows(rows)
    output: list[dict[str, Any]] = []
    for (task, method, identity), group in sorted(grouped.items()):
        parameters = dict(group[0].get("method_parameters", {}))
        if any(parameter_identity(row.get("method_parameters", {})) != identity for row in group):
            raise RuntimeError("Method parameter identity changed inside one aggregation group")
        output.append(
            {
                "task": task,
                "method": method,
                "method_parameters": parameters,
                "seed_count": len({int(row["seed"]) for row in group}),
                **mean_metrics(group, metric_names),
            }
        )
    return output


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write deterministic CSV while encoding nested method parameters as JSON."""

    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            materialized = dict(row)
            parameters = materialized.get("method_parameters")
            if isinstance(parameters, Mapping):
                materialized["method_parameters"] = parameter_identity(parameters)
            writer.writerow(materialized)
