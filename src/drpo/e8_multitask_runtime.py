"""Method-agnostic reliability primitives for E8 multitask experiments.

Recovery, terminal completeness, provenance, and package inventory should be
stable when a new scientific method is added. Concrete method invariants are
supplied as hooks and their evidence is namespaced rather than hard-coded here.
"""

from __future__ import annotations

import hashlib
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
class MethodAuditResult:
    passed: bool
    evidence: Mapping[str, Any]
    failures: tuple[str, ...] = ()


MethodAuditHook = Callable[[CellLike, Mapping[str, Any]], MethodAuditResult]


def stable_json_hash(value: Any) -> str:
    """Match the repository's existing stable-hash encoding exactly."""

    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def recovery_identity(
    cell: CellLike,
    *,
    experiment_id: str,
    config_hash: str,
    cell_identity_fields: Mapping[str, Any],
    common_identity_fields: Mapping[str, Any],
    schema_version: int = 1,
) -> dict[str, Any]:
    """Build the existing nested E8 identity shape without knowing method fields.

    The scientific layer projects the compatibility fields that belong inside
    ``cell`` (today these include legacy named parameters). Future methods may
    project one opaque ``method_parameters`` mapping instead. Common provenance
    such as bank/split/model/initialization/calibration identity is supplied
    separately. Keeping the nested layout and hash encoding here lets current
    cells preserve their frozen ``identity_hash`` byte-for-byte.
    """

    cell_identity: dict[str, Any] = {
        "task": cell.task,
        "method": cell.method,
    }
    collisions = sorted(set(cell_identity).intersection(cell_identity_fields))
    if collisions:
        raise ValueError(f"Method cell identity attempted to overwrite: {collisions}")
    cell_identity.update(cell_identity_fields)
    for reserved, expected in (("seed", int(cell.seed)), ("stage", cell.stage)):
        if reserved in cell_identity:
            raise ValueError(f"Method cell identity attempted to overwrite: ['{reserved}']")
        cell_identity[reserved] = expected

    identity: dict[str, Any] = {
        "schema_version": int(schema_version),
        "experiment_id": experiment_id,
        "config_hash": config_hash,
        "cell": cell_identity,
    }
    collisions = sorted(set(identity).intersection(common_identity_fields))
    if collisions:
        raise ValueError(f"Common recovery identity attempted to overwrite: {collisions}")
    identity.update(common_identity_fields)
    identity["identity_hash"] = stable_json_hash(identity)
    return identity


def _matches_int(value: Any, expected: int) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return int(value) == int(expected)
    except (TypeError, ValueError, OverflowError):
        return False


def validate_common_terminal_record(
    cell: CellLike,
    record: Mapping[str, Any],
    *,
    expected_terminal_step: int,
    expected_stop_reason: str,
) -> tuple[str, ...]:
    """Audit only invariants shared by every method and fail closed on bad types."""

    failures: list[str] = []
    if record.get("complete") is not True:
        failures.append("cell_not_complete")
    if record.get("evaluation_status") != "complete":
        failures.append("evaluation_not_complete")
    if not _matches_int(record.get("terminal_step"), expected_terminal_step):
        failures.append("terminal_step_mismatch")
    if str(record.get("stop_reason", "")) != expected_stop_reason:
        failures.append("stop_reason_mismatch")
    if record.get("nan_inf_failure") is not False:
        failures.append("nan_inf_failure")
    recorded_key = record.get("cell_key")
    if recorded_key is not None and str(recorded_key) != cell.key:
        failures.append("cell_key_mismatch")
    recorded_method = record.get("method")
    if recorded_method is not None and str(recorded_method) != cell.method:
        failures.append("method_mismatch")
    recorded_seed = record.get("seed")
    if recorded_seed is not None and not _matches_int(recorded_seed, int(cell.seed)):
        failures.append("seed_mismatch")
    return tuple(failures)


def audit_terminal_cell(
    cell: CellLike,
    record: Mapping[str, Any],
    *,
    expected_terminal_step: int,
    expected_stop_reason: str,
    method_audit: MethodAuditHook,
) -> dict[str, Any]:
    """Combine common runtime checks with one method-owned invariant hook."""

    common_failures = validate_common_terminal_record(
        cell,
        record,
        expected_terminal_step=expected_terminal_step,
        expected_stop_reason=expected_stop_reason,
    )
    method_result = method_audit(cell, record)
    method_failures = tuple(method_result.failures)
    if not method_result.passed and not method_failures:
        method_failures = ("method_audit_failed",)
    failures = tuple(common_failures) + method_failures
    return {
        "cell_key": cell.key,
        "task": cell.task,
        "method": cell.method,
        "seed": int(cell.seed),
        "common_failures": list(common_failures),
        "method_audit_passed": bool(method_result.passed),
        "method_audit_evidence": dict(method_result.evidence),
        "method_failures": list(method_failures),
        "passed": not failures and bool(method_result.passed),
    }


def audit_terminal_cells(
    cells: Sequence[CellLike],
    records: Mapping[str, Mapping[str, Any]],
    *,
    expected_terminal_step: int,
    expected_stop_reason: str,
    method_audit: MethodAuditHook,
) -> dict[str, Any]:
    """Audit a full plan and report missing, unexpected, and failed cells."""

    expected = {cell.key: cell for cell in cells}
    duplicate_count = len(cells) - len(expected)
    if duplicate_count:
        raise ValueError("Terminal audit received duplicate cell keys")
    missing = sorted(set(expected).difference(records))
    unexpected = sorted(set(records).difference(expected))
    audited = [
        audit_terminal_cell(
            expected[key],
            records[key],
            expected_terminal_step=expected_terminal_step,
            expected_stop_reason=expected_stop_reason,
            method_audit=method_audit,
        )
        for key in expected
        if key in records
    ]
    failed = [row["cell_key"] for row in audited if not row["passed"]]
    return {
        "expected_cell_count": len(expected),
        "observed_cell_count": len(records),
        "missing_cells": missing,
        "unexpected_cells": unexpected,
        "failed_cells": failed,
        "cells": audited,
        "passed": not missing and not unexpected and not failed,
    }


def file_inventory(
    root: Path,
    paths: Sequence[Path],
) -> list[dict[str, Any]]:
    """Return portable package inventory records independent of method type."""

    resolved_root = root.resolve()
    resolved_paths: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        try:
            resolved.relative_to(resolved_root)
        except ValueError as exc:
            raise ValueError(f"Package path escapes inventory root: {path}") from exc
        resolved_paths.append(resolved)

    inventory: list[dict[str, Any]] = []
    for path in sorted(resolved_paths, key=lambda item: item.relative_to(resolved_root).as_posix()):
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        inventory.append(
            {
                "path": path.relative_to(resolved_root).as_posix(),
                "size": path.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )
    return inventory
