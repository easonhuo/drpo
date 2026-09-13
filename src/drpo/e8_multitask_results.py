"""Method-agnostic result projection helpers for E8 multitask experiments."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


class CellLike(Protocol):
    @property
    def key(self) -> str: ...

    task: str
    method: str
    seed: int
    stage: str


@dataclass(frozen=True)
class MethodResultProjection:
    parameters: Mapping[str, Any]
    compatibility_columns: Mapping[str, Any]


@dataclass(frozen=True)
class ResultRecord:
    public: Mapping[str, Any]
    method_parameters: Mapping[str, Any]

    def public_row(self) -> dict[str, Any]:
        return dict(self.public)


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


def common_result_record(
    cell: CellLike,
    value: Mapping[str, Any],
    *,
    source: str,
    project_method: Callable[[CellLike], MethodResultProjection],
    project_metrics: Callable[[Mapping[str, Any]], Mapping[str, Any]],
) -> ResultRecord:
    projection = project_method(cell)
    compatibility = dict(projection.compatibility_columns)
    reserved = sorted(_RESERVED_COMMON_COLUMNS.intersection(compatibility))
    if reserved:
        raise ValueError(
            f"Method compatibility projection uses reserved columns: {reserved}"
        )
    public: dict[str, Any] = {
        "source": source,
        "task": cell.task,
        "method": cell.method,
    }
    _checked_merge(
        public,
        compatibility,
        label="Method compatibility projection",
    )
    public.update(
        {
            "seed": int(cell.seed),
            "stage": cell.stage,
            "cell_key": cell.key,
        }
    )
    _checked_merge(
        public,
        dict(project_metrics(value)),
        label="Metric projection",
    )
    return ResultRecord(
        public=public,
        method_parameters=dict(projection.parameters),
    )


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


def write_csv(
    path: Path, rows: Sequence[Mapping[str, Any] | ResultRecord]
) -> None:
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
