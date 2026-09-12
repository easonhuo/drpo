"""Method-agnostic reliability primitives for E8 multitask experiments.

Recovery, terminal completeness, provenance, and package inventory should be
stable when a new scientific method is added.  Concrete method invariants are
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


def stable_json_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def recovery_identity(
    cell: CellLike,
    *,
    experiment_id: str,
    config_hash: str,
    method_parameters: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a generic identity whose method payload is opaque to runtime."""

    identity = {
        "experiment_id": experiment_id,
        "config_hash": config_hash,
        "cell_key": cell.key,
        "task": cell.task,
        "method": cell.method,
        "seed": int(cell.seed),
        "stage": cell.stage,
        "method_parameters": dict(method_parameters),
    }
    return {**identity, "identity_sha256": stable_json_hash(identity)}


def validate_common_terminal_record(
    cell: CellLike,
    record: Mapping[str, Any],
    *,
    expected_terminal_step: int,
    expected_stop_reason: str,
) -> tuple[str, ...]:
    """Audit only invariants shared by every method."""

    failures: list[str] = []
    if record.get("complete") is not True:
        failures.append("cell_not_complete")
    if record.get("evaluation_status") != "complete":
        failures.append("evaluation_not_complete")
    if int(record.get("terminal_step", -1)) != int(expected_terminal_step):
        failures.append("terminal_step_mismatch")
    if str(record.get("stop_reason", "")) != expected_stop_reason:
        failures.append("stop_reason_mismatch")
    if bool(record.get("nan_inf_failure", False)):
        failures.append("nan_inf_failure")
    recorded_key = record.get("cell_key")
    if recorded_key is not None and str(recorded_key) != cell.key:
        failures.append("cell_key_mismatch")
    recorded_method = record.get("method")
    if recorded_method is not None and str(recorded_method) != cell.method:
        failures.append("method_mismatch")
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
    failures = tuple(common_failures) + tuple(method_result.failures)
    return {
        "cell_key": cell.key,
        "task": cell.task,
        "method": cell.method,
        "seed": int(cell.seed),
        "common_failures": list(common_failures),
        "method_audit_passed": bool(method_result.passed),
        "method_audit_evidence": dict(method_result.evidence),
        "method_failures": list(method_result.failures),
        "passed": not failures and bool(method_result.passed),
    }


def file_inventory(
    root: Path,
    paths: Sequence[Path],
) -> list[dict[str, Any]]:
    """Return portable package inventory records independent of method type."""

    inventory: list[dict[str, Any]] = []
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        inventory.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": path.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )
    return inventory
