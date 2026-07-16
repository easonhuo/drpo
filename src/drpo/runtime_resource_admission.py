"""Pre-launch worker admission under current measured CPU and memory capacity.

The planned runtime selection remains immutable provenance. This module derives an
attempt-local admitted worker count immediately before launch. A positive safe lower
count is allowed; zero safe workers remains fail-closed unless the caller explicitly
requests a structured blocked result for foreground capacity waiting.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Callable, Mapping

from drpo.runtime_resource_autotune import (
    RuntimeResourceError,
    atomic_write_json,
    load_json,
    utc_now,
)

CAPACITY_FAILURES = frozenset({"cpu_capacity_changed", "memory_capacity_changed"})
REVALIDATION_NAME = "RUNTIME_REVALIDATION.json"
ADMISSION_NAME = "RUNTIME_ADMISSION.json"

RevalidateRuntime = Callable[..., dict[str, Any]]


def _attempt_paths(work_dir: Path) -> set[Path]:
    return set(
        work_dir.glob(
            f"_runtime_resource_attempts/attempt-*/{REVALIDATION_NAME}"
        )
    )


def _new_attempt_path(work_dir: Path, before: set[Path]) -> Path:
    created = sorted(_attempt_paths(work_dir) - before)
    if len(created) != 1:
        raise RuntimeResourceError(
            "runtime revalidation did not produce exactly one attempt-local record"
        )
    return created[0]


def _finite_nonnegative(value: object, context: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeResourceError(f"{context} is missing or non-numeric") from exc
    if not math.isfinite(number) or number < 0:
        raise RuntimeResourceError(f"{context} must be finite and non-negative")
    return number


def _positive(value: object, context: str) -> float:
    number = _finite_nonnegative(value, context)
    if number <= 0:
        raise RuntimeResourceError(f"{context} must be positive")
    return number


def _floor_capacity(*, budget: object, observed: object, per_worker: float) -> int:
    available = _finite_nonnegative(budget, "capacity budget") - _finite_nonnegative(
        observed, "observed capacity"
    )
    return max(0, math.floor(max(0.0, available) / per_worker))


def _selection_payload(
    document: Mapping[str, Any],
    *,
    proposed_workers: int,
    selection_digest: str,
) -> Mapping[str, Any]:
    if document.get("selection_digest") != selection_digest:
        raise RuntimeResourceError("runtime selection digest changed before admission")
    selection = document.get("selection")
    if not isinstance(selection, Mapping):
        raise RuntimeResourceError("runtime selection payload is missing")
    observed = int(selection.get("selected_workers", 0) or 0)
    if observed != proposed_workers:
        raise RuntimeResourceError(
            "runtime selection proposed worker count changed before admission"
        )
    return selection


def _capacity_limits(
    *,
    record: Mapping[str, Any],
    selection: Mapping[str, Any],
    proposed_workers: int,
) -> tuple[int, int, dict[str, Any]]:
    reserved_cpu = _positive(
        selection.get("per_worker_reserved_cpu_cores"),
        "per-worker reserved CPU cores",
    )
    reserved_memory = int(selection.get("per_worker_reserved_bytes", 0) or 0)
    if reserved_memory < 1:
        raise RuntimeResourceError("per-worker reserved memory must be positive")

    cpu = record.get("cpu_revalidation")
    if not isinstance(cpu, Mapping):
        raise RuntimeResourceError("CPU revalidation evidence is missing")
    cpu_limits = [
        _floor_capacity(
            budget=cpu.get("affinity_budget_cores"),
            observed=cpu.get("conservative_system_busy_cores"),
            per_worker=reserved_cpu,
        )
    ]
    quota_rows = cpu.get("quota_domains", [])
    if not isinstance(quota_rows, list):
        raise RuntimeResourceError("CPU quota-domain evidence must be a list")
    for row in quota_rows:
        if not isinstance(row, Mapping):
            raise RuntimeResourceError("CPU quota-domain evidence is malformed")
        cpu_limits.append(
            _floor_capacity(
                budget=row.get("budget_cores"),
                observed=row.get("observed_busy_cores"),
                per_worker=reserved_cpu,
            )
        )
    cpu_worker_limit = min([proposed_workers, *cpu_limits])

    memory = record.get("memory_revalidation")
    if not isinstance(memory, Mapping):
        raise RuntimeResourceError("memory revalidation evidence is missing")
    usable_memory = int(memory.get("usable_memory_bytes", 0) or 0)
    if usable_memory < 0:
        raise RuntimeResourceError("usable memory cannot be negative")
    memory_worker_limit = min(proposed_workers, usable_memory // reserved_memory)

    evidence = {
        "per_worker_reserved_cpu_cores": reserved_cpu,
        "per_worker_reserved_bytes": reserved_memory,
        "cpu_worker_limit": cpu_worker_limit,
        "memory_worker_limit": memory_worker_limit,
        "affinity_budget_cores": cpu.get("affinity_budget_cores"),
        "conservative_system_busy_cores": cpu.get(
            "conservative_system_busy_cores"
        ),
        "quota_domains": quota_rows,
        "usable_memory_bytes": usable_memory,
    }
    return cpu_worker_limit, memory_worker_limit, evidence


def revalidate_with_safe_downshift(
    *,
    revalidate_runtime: RevalidateRuntime,
    work_dir: str | Path,
    proposed_workers: int,
    selection_digest: str,
    revalidate_kwargs: Mapping[str, Any],
    allow_zero: bool = False,
) -> dict[str, Any]:
    """Revalidate and admit the largest currently safe worker count.

    Identity, checkout, binding, measurement, and non-capacity failures remain fatal.
    Only exact CPU/memory capacity failures may produce a lower pre-launch worker count.
    The immutable selection document is never rewritten. By default zero capacity raises;
    ``allow_zero=True`` returns the same structured blocked evidence so a foreground
    launcher can wait and remeasure without starting work.
    """

    if proposed_workers < 1:
        raise RuntimeResourceError("proposed_workers must be positive")
    if not selection_digest:
        raise RuntimeResourceError("selection_digest must be non-empty")

    work = Path(work_dir).resolve()
    selection_path = work / "RUNTIME_SELECTION.json"
    if not selection_path.is_file():
        raise RuntimeResourceError("missing immutable RUNTIME_SELECTION.json")
    immutable_document = load_json(selection_path)
    selection = _selection_payload(
        immutable_document,
        proposed_workers=proposed_workers,
        selection_digest=selection_digest,
    )

    before = _attempt_paths(work)
    blocked_error: RuntimeResourceError | None = None
    try:
        validated_document = revalidate_runtime(**dict(revalidate_kwargs))
        revalidation_value = validated_document.get("revalidation")
        if not isinstance(revalidation_value, Mapping):
            raise RuntimeResourceError("successful revalidation lacks evidence path")
        record_path = Path(str(revalidation_value.get("path", ""))).resolve()
        if not record_path.is_file():
            raise RuntimeResourceError("successful revalidation evidence is missing")
        record = load_json(record_path)
    except RuntimeResourceError as exc:
        blocked_error = exc
        record_path = _new_attempt_path(work, before)
        record = load_json(record_path)
        raw_failures = record.get("failures")
        if not isinstance(raw_failures, list) or not raw_failures:
            raise
        failures = {str(value) for value in raw_failures}
        if record.get("decision") != "BLOCK" or not failures <= CAPACITY_FAILURES:
            raise
        validated_document = copy.deepcopy(immutable_document)

    _selection_payload(
        validated_document,
        proposed_workers=proposed_workers,
        selection_digest=selection_digest,
    )
    raw_failures = record.get("failures", [])
    if not isinstance(raw_failures, list):
        raise RuntimeResourceError("revalidation failure list is malformed")
    failures = sorted(str(value) for value in raw_failures)
    if failures and not set(failures) <= CAPACITY_FAILURES:
        if blocked_error is not None:
            raise blocked_error
        raise RuntimeResourceError(
            "successful revalidation unexpectedly contains non-capacity failures"
        )

    cpu_limit, memory_limit, capacity = _capacity_limits(
        record=record,
        selection=selection,
        proposed_workers=proposed_workers,
    )
    admitted_workers = min(proposed_workers, cpu_limit, memory_limit)
    downshifted = admitted_workers < proposed_workers
    decision = "ALLOW" if admitted_workers >= 1 else "BLOCK"
    admission_path = record_path.with_name(ADMISSION_NAME)
    admission = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "decision": decision,
        "proposed_workers": proposed_workers,
        "admitted_workers": admitted_workers,
        "downshifted": downshifted,
        "reason": (
            "capacity_downshift_before_launch"
            if downshifted and admitted_workers >= 1
            else (
                "planned_capacity_still_safe"
                if admitted_workers >= 1
                else "no_safe_worker_capacity"
            )
        ),
        "selection_path": str(selection_path),
        "selection_digest": selection_digest,
        "revalidation_path": str(record_path),
        "original_revalidation_decision": record.get("decision"),
        "original_capacity_failures": failures,
        "capacity": capacity,
        "scientific_matrix_changed": False,
        "running_workers_resized": False,
    }
    atomic_write_json(admission_path, admission)
    admission["path"] = str(admission_path)

    result = copy.deepcopy(validated_document)
    result["runtime_admission"] = admission
    result["revalidation"] = {
        "decision": (
            "BLOCK"
            if admitted_workers < 1
            else ("ALLOW_WITH_DOWNSHIFT" if downshifted else "ALLOW")
        ),
        "attempt_id": record.get("attempt_id"),
        "path": str(record_path),
        "admission_path": str(admission_path),
        "original_decision": record.get("decision"),
        "capacity_failures": failures,
    }

    if admitted_workers < 1 and not allow_zero:
        raise RuntimeResourceError(
            "RUNTIME_CAPACITY_CHANGED_REPLAN_REQUIRED: "
            + ",".join(failures or ["no_safe_worker_capacity"])
        )
    return result
