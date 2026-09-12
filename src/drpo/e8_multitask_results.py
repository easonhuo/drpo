"""Method-agnostic result projection and aggregation helpers for E8.

Concrete methods own their parameter projection.  This module keeps opaque
method parameters in memory while materializing only explicitly projected
compatibility columns.  That separation prevents a refactor from silently
changing frozen CSV/JSON schemas merely to support future methods.
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
    """Method-owned opaque parameters plus backward-compatible public columns."""

    parameters: Mapping[str, Any]
    compatibility_columns: Mapping[str, Any]


@dataclass(frozen=True)
class ResultRecord:
    """One generic result record with a schema-private method payload."""

    public: Mapping[str, Any]
    method_parameters: Mapping[str, Any]

    def public_row(self) -> dict[str, Any]:
        return dict(self.public)


_RESERVED_COMMON_COLUMNS = frozenset(
    {"source", "task", "method", "seed", "stage", "cell_key"}
)


def _checked_merge(
    target: dict[str, Any],
    values: Mapping[str, Any],
    *,
    label: str,
) -> None:
    collisions = sorted(set(target).intersection(values))
    if collisions:
        raise ValueError(f"{label} attempted to overwrite result columns: {collisions}")
    target.update(values)


def common_result_record(
    cell: CellLike,
    value: Mapping[str, Any],
    *,
    source: str,
    project_method: Callable[[CellLike], MethodResultProjection],
    project_metrics: Callable[[Mapping[str, Any]], Mapping[str, Any]],
) -> ResultRecord:
    """Build one record without exposing opaque parameters in public artifacts."""

    projection = project_method(cell)
    compatibility = dict(projection.compatibility_columns)
    reserved = sorted(_RESERVED_COMMON_COLUMNS.intersection(compatibility))
    if reserved:
        raise ValueError(f"Method compatibility projection uses reserved columns: {reserved}")
    public: dict[str, Any] = {
        "source": source,
        "task": cell.task,
        "method": cell.method,
        "seed": int(cell.seed),
        "stage": cell.stage,
        "cell_key": cell.key,
    }
    _checked_merge(public, compatibility, label="Method compatibility projection")
    _checked_merge(public, dict(project_metrics(value)), label="Metric projection")
    return ResultRecord(
        public=public,
        method_parameters=dict(projection.parameters),
    )


def parameter_identity(parameters: Mapping[str, Any]) -> str:
    """Stable JSON identity used only for generic in-memory grouping."""

    return json.dumps(dict(parameters), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def group_parameter_records(
    records: Sequence[ResultRecord],
) -> dict[tuple[str, str, str], list[ResultRecord]]:
    """Group by task, method and opaque method-parameter identity."""

    grouped: dict[tuple[str, str, str], list[ResultRecord]] = {}
    for record in records:
        row = record.public
        key = (
            str(row["task"]),
            str(row["method"]),
            parameter_identity(record.method_parameters),
        )
        grouped.setdefault(key, []).append(record)
    return grouped


def mean_metrics(
    records: Sequence[ResultRecord],
    metric_names: Sequence[str],
) -> dict[str, float]:
    if not records:
        raise ValueError("Cannot aggregate an empty result group")
    return {
        f"{name}_mean": float(np.mean([float(record.public[name]) for record in records]))
        for name in metric_names
    }


def grouped_curve(
    records: Sequence[ResultRecord],
    *,
    metric_names: Sequence[str],
    group_order: Callable[[str, str, Mapping[str, Any]], Any] | None = None,
    project_group: Callable[[str, Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Create deterministic method curves without knowing parameter names.

    Opaque parameters never appear in the returned public rows.  A scientific
    method adapter may explicitly project reviewed compatibility columns through
    ``project_group``.  If no projection is requested, the curve contains only
    common identity/count/metric fields.
    """

    grouped = group_parameter_records(records)
    entries: list[tuple[Any, str, str, str, list[ResultRecord]]] = []
    for (task, method, identity), group in grouped.items():
        parameters = dict(group[0].method_parameters)
        if any(parameter_identity(record.method_parameters) != identity for record in group):
            raise RuntimeError("Method parameter identity changed inside one aggregation group")
        sort_value = (
            group_order(task, method, parameters)
            if group_order is not None
            else (task, method, identity)
        )
        entries.append((sort_value, task, method, identity, group))

    output: list[dict[str, Any]] = []
    for _, task, method, _, group in sorted(entries, key=lambda item: item[0]):
        parameters = dict(group[0].method_parameters)
        row: dict[str, Any] = {
            "task": task,
            "method": method,
            "seed_count": len({int(record.public["seed"]) for record in group}),
        }
        if project_group is not None:
            _checked_merge(
                row,
                dict(project_group(method, parameters)),
                label="Grouped method compatibility projection",
            )
        _checked_merge(row, mean_metrics(group, metric_names), label="Grouped metric projection")
        output.append(row)
    return output


def write_csv(path: Path, rows: Sequence[Mapping[str, Any] | ResultRecord]) -> None:
    """Write deterministic public CSV rows; private parameters cannot leak."""

    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    materialized = [
        row.public_row() if isinstance(row, ResultRecord) else dict(row)
        for row in rows
    ]
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
