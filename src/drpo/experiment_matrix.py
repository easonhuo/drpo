"""Small, science-agnostic helpers for deterministic experiment matrices."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any


def _sequence(values: Any, name: str) -> Sequence[Any]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{name} must be a sequence")
    if not values:
        raise ValueError(f"{name} must not be empty")
    return values


def finite_float_values(
    values: Any,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> tuple[float, ...]:
    """Normalize a non-empty sequence of unique finite floats."""

    result: list[float] = []
    for index, value in enumerate(_sequence(values, name)):
        if isinstance(value, bool):
            raise ValueError(f"{name}[{index}] must be numeric, not boolean")
        try:
            normalized = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}[{index}] must be numeric") from exc
        if not math.isfinite(normalized):
            raise ValueError(f"{name}[{index}] must be finite")
        if minimum is not None and normalized < minimum:
            raise ValueError(f"{name}[{index}] must be >= {minimum}")
        if maximum is not None and normalized > maximum:
            raise ValueError(f"{name}[{index}] must be <= {maximum}")
        result.append(normalized)
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must be unique")
    return tuple(result)


def integer_values(
    values: Any,
    *,
    name: str,
    minimum: int | None = None,
) -> tuple[int, ...]:
    """Normalize a non-empty sequence of unique integers."""

    result: list[int] = []
    for index, value in enumerate(_sequence(values, name)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name}[{index}] must be an integer")
        if minimum is not None and value < minimum:
            raise ValueError(f"{name}[{index}] must be >= {minimum}")
        result.append(value)
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must be unique")
    return tuple(result)


def expand_matrix(dimensions: Mapping[str, Sequence[Any]]) -> tuple[dict[str, Any], ...]:
    """Expand ordered dimensions into deterministic row dictionaries."""

    if not isinstance(dimensions, Mapping) or not dimensions:
        raise ValueError("dimensions must be a non-empty mapping")
    names: list[str] = []
    values: list[tuple[Any, ...]] = []
    for name, dimension in dimensions.items():
        if not isinstance(name, str) or not name or name != name.strip():
            raise ValueError("dimension names must be non-empty stripped strings")
        names.append(name)
        values.append(tuple(_sequence(dimension, f"dimension {name}")))
    return tuple(
        dict(zip(names, combination, strict=True))
        for combination in itertools.product(*values)
    )


def require_declared_count(*, name: str, declared: Any, actual: int) -> None:
    """Fail closed when a declared matrix count differs from expansion."""

    if isinstance(declared, bool) or not isinstance(declared, int):
        raise ValueError(f"{name} must be an integer")
    if declared != actual:
        raise ValueError(f"{name} declares {declared}, but expansion produced {actual}")


def canonical_digest(value: Any) -> str:
    """Return a stable SHA-256 digest for a JSON-compatible value."""

    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"matrix payload is not canonical JSON: {exc}") from exc
    return hashlib.sha256(payload).hexdigest()
