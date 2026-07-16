"""Foreground waiting for safe runtime planning and pre-launch admission.

This module never invents capacity. It repeats only verified capacity-limited
operations while preserving immutable scientific identity. Scientific work starts only
after a positive measured worker admission.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from drpo.runtime_resource_autotune import (
    RuntimeResourceError,
    atomic_write_json,
    load_json,
    utc_now,
)

WAIT_STATE_NAME = "RUNTIME_CAPACITY_WAIT.json"
WAIT_EVENTS_NAME = "RUNTIME_CAPACITY_WAIT.jsonl"
PLAN_WAIT_STATE_NAME = "RUNTIME_PLAN_CAPACITY_WAIT.json"
PLAN_WAIT_EVENTS_NAME = "RUNTIME_PLAN_CAPACITY_WAIT.jsonl"
ADMISSION_NAME = "RUNTIME_ADMISSION.json"
PLAN_CAPACITY_MESSAGES = (
    "measured CPU capacity cannot support one worker",
    "insufficient host memory for one worker after safety headroom",
    "measured CPU/RAM capacity produced no worker slot",
)
NO_VALID_CANDIDATE_MESSAGE = "no resource-valid concurrency candidate completed"

AdmissionFunction = Callable[..., dict[str, Any]]
KwargsFactory = Callable[[], Mapping[str, Any]]
PlanFunction = Callable[[], dict[str, Any]]
Clock = Callable[[], float]
Sleeper = Callable[[float], None]
EventSink = Callable[[Mapping[str, Any]], None]


def _admission_paths(work_dir: Path) -> set[Path]:
    return set(
        work_dir.glob(
            f"_runtime_resource_attempts/attempt-*/{ADMISSION_NAME}"
        )
    )


def _new_blocked_admission_path(work_dir: Path, before: set[Path]) -> Path | None:
    created = sorted(_admission_paths(work_dir) - before)
    if len(created) == 1:
        return created[0]
    return None


def _finite(value: object, context: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeResourceError(f"{context} must be numeric") from exc
    if not math.isfinite(number):
        raise RuntimeResourceError(f"{context} must be finite")
    return number


def _validate_wait_policy(
    *,
    wait_timeout_seconds: float,
    poll_seconds: float,
) -> tuple[float, float]:
    timeout = _finite(wait_timeout_seconds, "wait_timeout_seconds")
    interval = _finite(poll_seconds, "poll_seconds")
    if interval <= 0:
        raise RuntimeResourceError("poll_seconds must be positive")
    return timeout, interval


def _validate_admission(
    admission: Mapping[str, Any],
    *,
    proposed_workers: int,
    selection_digest: str,
) -> int:
    if int(admission.get("proposed_workers", 0) or 0) != proposed_workers:
        raise RuntimeResourceError(
            "capacity-wait admission changed the proposed worker count"
        )
    if admission.get("selection_digest") != selection_digest:
        raise RuntimeResourceError(
            "capacity-wait admission changed the selection digest"
        )
    admitted_workers = int(admission.get("admitted_workers", -1) or 0)
    if admitted_workers < 0 or admitted_workers > proposed_workers:
        raise RuntimeResourceError(
            "capacity-wait admission contains an invalid worker count"
        )
    decision = admission.get("decision")
    if admitted_workers == 0 and decision != "BLOCK":
        raise RuntimeResourceError(
            "zero-worker admission must have decision BLOCK"
        )
    if admitted_workers > 0 and decision != "ALLOW":
        raise RuntimeResourceError(
            "positive-worker admission must have decision ALLOW"
        )
    return admitted_workers


def _append_event(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True) + "\n")
        handle.flush()


def plan_capacity_shortage(
    error: RuntimeResourceError,
    *,
    work_dir: str | Path,
) -> bool:
    """Return true only for evidence-backed transient planning capacity shortage."""

    message = str(error)
    if any(value in message for value in PLAN_CAPACITY_MESSAGES):
        return True
    if NO_VALID_CANDIDATE_MESSAGE not in message:
        return False

    work = Path(work_dir).resolve()
    summaries = sorted(
        work.glob(
            "_runtime_resource_probe/w0_throughput/"
            "workers-*/BENCHMARK_SUMMARY.json"
        )
    )
    if not summaries:
        return False
    for path in summaries:
        row = load_json(path)
        concurrency = int(row.get("concurrency", 0) or 0)
        completed = int(row.get("completed", 0) or 0)
        if concurrency < 1 or completed != concurrency:
            return False
        if int(row.get("failed", 0) or 0) != 0:
            return False
        if int(row.get("timed_out", 0) or 0) != 0:
            return False
        if int(row.get("controller_terminated", 0) or 0) != 0:
            return False
        if int(row.get("orphan_process_groups", 0) or 0) != 0:
            return False
        if float(row.get("measured_candidate_cpu_cores", 0.0) or 0.0) <= 0:
            return False
        if int(row.get("aggregate_peak_rss_bytes", 0) or 0) <= 0:
            return False
        if row.get("valid") is True:
            return False
        cpu_ok = row.get("cpu_capacity_ok") is True
        memory_ok = row.get("memory_capacity_ok") is True
        if cpu_ok and memory_ok:
            return False
    return True


def wait_for_runtime_plan(
    *,
    plan_once: PlanFunction,
    work_dir: str | Path,
    wait_timeout_seconds: float,
    poll_seconds: float,
    clock: Clock = time.monotonic,
    sleep: Sleeper = time.sleep,
    on_event: EventSink | None = None,
) -> dict[str, Any]:
    """Wait in the foreground until the measured runtime plan can be materialized."""

    timeout, interval = _validate_wait_policy(
        wait_timeout_seconds=wait_timeout_seconds,
        poll_seconds=poll_seconds,
    )
    work = Path(work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    state_path = work / PLAN_WAIT_STATE_NAME
    events_path = work / PLAN_WAIT_EVENTS_NAME
    started_utc = utc_now()
    started = clock()
    attempts = 0

    while True:
        attempts += 1
        try:
            result = plan_once()
        except RuntimeResourceError as exc:
            if not plan_capacity_shortage(exc, work_dir=work):
                raise
            elapsed = max(0.0, clock() - started)
            event = {
                "schema_version": 1,
                "created_utc": utc_now(),
                "attempt": attempts,
                "status": "WAITING_FOR_CAPACITY",
                "error": str(exc),
                "elapsed_wait_seconds": elapsed,
                "scientific_matrix_changed": False,
            }
            _append_event(events_path, event)
            if on_event is not None:
                on_event(event)
            state = {
                "schema_version": 1,
                "status": "WAITING_FOR_CAPACITY",
                "started_utc": started_utc,
                "updated_utc": event["created_utc"],
                "attempt_count": attempts,
                "wait_timeout_seconds": timeout,
                "poll_seconds": interval,
                "last_error": str(exc),
                "elapsed_wait_seconds": elapsed,
                "events_path": str(events_path),
                "scientific_matrix_changed": False,
            }
            deadline_reached = timeout >= 0 and elapsed >= timeout
            if deadline_reached:
                state["status"] = (
                    "BLOCKED_NO_WAIT" if timeout == 0 else "BLOCKED_WAIT_TIMEOUT"
                )
                atomic_write_json(state_path, state)
                if timeout == 0:
                    raise
                raise RuntimeResourceError(
                    "RUNTIME_PLAN_CAPACITY_WAIT_TIMEOUT: "
                    f"elapsed={elapsed:.3f}s,last_error={exc}"
                ) from exc
            atomic_write_json(state_path, state)
            remaining = None if timeout < 0 else max(0.0, timeout - elapsed)
            sleep_seconds = interval if remaining is None else min(interval, remaining)
            sleep(sleep_seconds)
            continue

        elapsed = max(0.0, clock() - started)
        event = {
            "schema_version": 1,
            "created_utc": utc_now(),
            "attempt": attempts,
            "status": "PLANNED",
            "elapsed_wait_seconds": elapsed,
            "scientific_matrix_changed": False,
        }
        _append_event(events_path, event)
        if on_event is not None:
            on_event(event)
        state = {
            "schema_version": 1,
            "status": "PLANNED",
            "started_utc": started_utc,
            "updated_utc": event["created_utc"],
            "attempt_count": attempts,
            "wait_timeout_seconds": timeout,
            "poll_seconds": interval,
            "elapsed_wait_seconds": elapsed,
            "events_path": str(events_path),
            "scientific_matrix_changed": False,
        }
        atomic_write_json(state_path, state)
        result["plan_capacity_wait"] = {
            "status": "PLANNED",
            "path": str(state_path),
            "events_path": str(events_path),
            "attempt_count": attempts,
            "elapsed_wait_seconds": elapsed,
            "scientific_matrix_changed": False,
        }
        return result


def wait_for_runtime_admission(
    *,
    admit_once: AdmissionFunction,
    work_dir: str | Path,
    proposed_workers: int,
    selection_digest: str,
    revalidate_kwargs_factory: KwargsFactory,
    wait_timeout_seconds: float,
    poll_seconds: float,
    minimum_admitted_workers: int,
    clock: Clock = time.monotonic,
    sleep: Sleeper = time.sleep,
    on_event: EventSink | None = None,
) -> dict[str, Any]:
    """Wait in the foreground until current safe capacity reaches the floor.

    ``wait_timeout_seconds`` semantics:

    - negative: wait without an automatic deadline;
    - zero: perform exactly one admission attempt;
    - positive: wait for at most that many monotonic seconds.

    Identity and non-capacity errors are never retried. Only a structurally verified
    zero-capacity admission, or a positive admission below the configured practical
    floor, enters the wait loop.
    """

    if proposed_workers < 1:
        raise RuntimeResourceError("proposed_workers must be positive")
    if not selection_digest:
        raise RuntimeResourceError("selection_digest must be non-empty")
    if minimum_admitted_workers < 1:
        raise RuntimeResourceError("minimum_admitted_workers must be positive")
    if minimum_admitted_workers > proposed_workers:
        raise RuntimeResourceError(
            "minimum_admitted_workers cannot exceed proposed_workers"
        )
    timeout, interval = _validate_wait_policy(
        wait_timeout_seconds=wait_timeout_seconds,
        poll_seconds=poll_seconds,
    )

    work = Path(work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    state_path = work / WAIT_STATE_NAME
    events_path = work / WAIT_EVENTS_NAME
    started_utc = utc_now()
    started = clock()
    attempts = 0

    while True:
        attempts += 1
        before = _admission_paths(work)
        blocked_error: RuntimeResourceError | None = None
        result: dict[str, Any] | None = None
        try:
            kwargs = dict(revalidate_kwargs_factory())
            result = admit_once(
                work_dir=work,
                proposed_workers=proposed_workers,
                selection_digest=selection_digest,
                revalidate_kwargs=kwargs,
            )
            admission_value = result.get("runtime_admission")
            if not isinstance(admission_value, Mapping):
                raise RuntimeResourceError(
                    "successful capacity-wait attempt lacks runtime admission"
                )
            admission = dict(admission_value)
        except RuntimeResourceError as exc:
            blocked_error = exc
            admission_path = _new_blocked_admission_path(work, before)
            if admission_path is None:
                raise blocked_error
            admission = load_json(admission_path)
            admission["path"] = str(admission_path)
            admitted = _validate_admission(
                admission,
                proposed_workers=proposed_workers,
                selection_digest=selection_digest,
            )
            if (
                admitted != 0
                or admission.get("reason") != "no_safe_worker_capacity"
                or admission.get("decision") != "BLOCK"
            ):
                raise blocked_error

        admitted_workers = _validate_admission(
            admission,
            proposed_workers=proposed_workers,
            selection_digest=selection_digest,
        )
        elapsed = max(0.0, clock() - started)
        admitted = admitted_workers >= minimum_admitted_workers
        event = {
            "schema_version": 1,
            "created_utc": utc_now(),
            "attempt": attempts,
            "status": "ADMITTED" if admitted else "WAITING_FOR_CAPACITY",
            "proposed_workers": proposed_workers,
            "minimum_admitted_workers": minimum_admitted_workers,
            "admitted_workers": admitted_workers,
            "selection_digest": selection_digest,
            "admission_path": admission.get("path"),
            "revalidation_path": admission.get("revalidation_path"),
            "admission_reason": admission.get("reason"),
            "capacity": admission.get("capacity"),
            "elapsed_wait_seconds": elapsed,
            "error": None if blocked_error is None else str(blocked_error),
            "scientific_matrix_changed": False,
        }
        _append_event(events_path, event)
        if on_event is not None:
            on_event(event)

        state = {
            "schema_version": 1,
            "status": event["status"],
            "started_utc": started_utc,
            "updated_utc": event["created_utc"],
            "attempt_count": attempts,
            "wait_timeout_seconds": timeout,
            "poll_seconds": interval,
            "proposed_workers": proposed_workers,
            "minimum_admitted_workers": minimum_admitted_workers,
            "selection_digest": selection_digest,
            "last_admission": admission,
            "last_error": event["error"],
            "elapsed_wait_seconds": elapsed,
            "events_path": str(events_path),
            "scientific_matrix_changed": False,
            "running_workers_resized": False,
        }

        if admitted:
            if result is None:
                raise RuntimeResourceError(
                    "admitted capacity-wait attempt lacks successful result"
                )
            state["status"] = "ADMITTED"
            state["admitted_workers"] = admitted_workers
            atomic_write_json(state_path, state)
            result["capacity_wait"] = {
                "status": "ADMITTED",
                "path": str(state_path),
                "events_path": str(events_path),
                "attempt_count": attempts,
                "elapsed_wait_seconds": elapsed,
                "minimum_admitted_workers": minimum_admitted_workers,
                "admitted_workers": admitted_workers,
                "scientific_matrix_changed": False,
            }
            return result

        deadline_reached = timeout >= 0 and elapsed >= timeout
        if deadline_reached:
            state["status"] = (
                "BLOCKED_NO_WAIT" if timeout == 0 else "BLOCKED_WAIT_TIMEOUT"
            )
            atomic_write_json(state_path, state)
            if blocked_error is not None and timeout == 0:
                raise blocked_error
            raise RuntimeResourceError(
                "RUNTIME_CAPACITY_WAIT_TIMEOUT: "
                f"admitted={admitted_workers},required={minimum_admitted_workers},"
                f"elapsed={elapsed:.3f}s"
            )

        atomic_write_json(state_path, state)
        remaining = None if timeout < 0 else max(0.0, timeout - elapsed)
        sleep_seconds = interval if remaining is None else min(interval, remaining)
        sleep(sleep_seconds)
