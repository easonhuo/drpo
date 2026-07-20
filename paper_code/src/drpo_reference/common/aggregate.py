"""Deterministic aggregation helpers for reviewer-facing experiment artifacts."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    seed: int,
    samples: int = 4000,
) -> tuple[float, float, float]:
    """Return mean and percentile bootstrap 95% interval for finite values."""

    array = np.asarray([float(value) for value in values], dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan"), float("nan"), float("nan")
    if array.size == 1:
        value = float(array[0])
        return value, value, value
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, array.size, size=(samples, array.size))
    means = array[indices].mean(axis=1)
    return (
        float(array.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def _group_key(row: Mapping[str, Any], keys: Sequence[str]) -> tuple[Any, ...]:
    return tuple(row[key] for key in keys)


def aggregate_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    group_keys: Sequence[str],
    event_fields: Sequence[str] = (),
    bootstrap_seed: int = 20260624,
    bootstrap_samples: int = 4000,
) -> list[dict[str, Any]]:
    """Aggregate numeric fields while preserving explicit event counts.

    Booleans listed in ``event_fields`` are reported as counts and rates. Other
    booleans are excluded from numeric means so that event semantics remain
    visible rather than being silently folded into unrelated metrics.
    """

    materialized = [dict(row) for row in rows]
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in materialized:
        groups.setdefault(_group_key(row, group_keys), []).append(row)

    output: list[dict[str, Any]] = []
    for group_index, (key, group) in enumerate(sorted(groups.items(), key=str)):
        result = {name: value for name, value in zip(group_keys, key)}
        result["n_runs"] = len(group)
        result["n_seeds"] = len({row.get("seed") for row in group})

        numeric_fields: list[str] = []
        for row in group:
            for name, value in row.items():
                if name in group_keys or name in event_fields:
                    continue
                if isinstance(value, bool) or value is None:
                    continue
                if isinstance(value, (int, float)) and name not in numeric_fields:
                    numeric_fields.append(name)

        for field_index, name in enumerate(numeric_fields):
            values = [
                float(row[name])
                for row in group
                if isinstance(row.get(name), (int, float))
                and not isinstance(row.get(name), bool)
                and math.isfinite(float(row[name]))
            ]
            if not values:
                continue
            mean, low, high = bootstrap_mean_ci(
                values,
                seed=bootstrap_seed + 1009 * group_index + field_index,
                samples=bootstrap_samples,
            )
            result[f"{name}_mean"] = mean
            result[f"{name}_ci_low"] = low
            result[f"{name}_ci_high"] = high

        for name in event_fields:
            count = sum(bool(row.get(name, False)) for row in group)
            result[f"{name}_count"] = count
            result[f"{name}_rate"] = count / len(group)

        output.append(result)
    return output


def audit_run_matrix(
    rows: Iterable[Mapping[str, Any]],
    *,
    identity_fields: Sequence[str],
    expected_identities: Iterable[Sequence[Any]],
    required_fields: Sequence[str],
) -> dict[str, Any]:
    """Audit run presence, uniqueness, and required terminal fields."""

    materialized = [dict(row) for row in rows]
    observed = [tuple(row.get(name) for name in identity_fields) for row in materialized]
    expected = [tuple(identity) for identity in expected_identities]
    observed_set = set(observed)
    expected_set = set(expected)
    duplicate_identities = sorted(
        {identity for identity in observed if observed.count(identity) > 1},
        key=str,
    )
    missing = sorted(expected_set - observed_set, key=str)
    unexpected = sorted(observed_set - expected_set, key=str)
    incomplete = []
    for row, identity in zip(materialized, observed):
        absent = [name for name in required_fields if name not in row]
        if absent:
            incomplete.append({"identity": identity, "missing_fields": absent})

    checks = {
        "all_expected_runs_present": not missing,
        "no_unexpected_runs": not unexpected,
        "no_duplicate_run_identities": not duplicate_identities,
        "required_terminal_fields_present": not incomplete,
    }
    return {
        "identity_fields": list(identity_fields),
        "expected_runs": len(expected),
        "observed_runs": len(materialized),
        "missing_identities": missing,
        "unexpected_identities": unexpected,
        "duplicate_identities": duplicate_identities,
        "incomplete_runs": incomplete,
        "checks": checks,
        "passed": all(checks.values()),
    }
