"""Small foreground wait loop for E7 resource planning and launch admission."""
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

PLAN_CAPACITY_MESSAGES = (
    "measured CPU capacity cannot support one worker",
    "insufficient host memory for one worker after safety headroom",
    "measured CPU/RAM capacity produced no worker slot",
)
NO_VALID_CANDIDATE_MESSAGE = "no resource-valid concurrency candidate completed"

Attempt = Callable[[], dict[str, Any]]
Ready = Callable[[Mapping[str, Any]], bool]
Describe = Callable[[Mapping[str, Any]], Mapping[str, Any]]
Retryable = Callable[[RuntimeResourceError], bool]
EventSink = Callable[[Mapping[str, Any]], None]


def _number(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeResourceError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise RuntimeResourceError(f"{name} must be finite")
    return result


def _append(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True) + "\n")
        handle.flush()


def _wait(
    *,
    attempt: Attempt,
    ready: Ready,
    describe: Describe,
    retryable: Retryable | None,
    work_dir: Path,
    state_name: str,
    events_name: str,
    metadata_key: str,
    ready_status: str,
    timeout_prefix: str,
    wait_timeout_seconds: float,
    poll_seconds: float,
    clock: Callable[[], float],
    sleep: Callable[[float], None],
    on_event: EventSink | None,
) -> dict[str, Any]:
    timeout = _number(wait_timeout_seconds, "wait_timeout_seconds")
    interval = _number(poll_seconds, "poll_seconds")
    if interval <= 0:
        raise RuntimeResourceError("poll_seconds must be positive")

    work_dir.mkdir(parents=True, exist_ok=True)
    state_path = work_dir / state_name
    events_path = work_dir / events_name
    started = clock()
    started_utc = utc_now()
    attempt_count = 0

    while True:
        attempt_count += 1
        error: RuntimeResourceError | None = None
        result: dict[str, Any] | None = None
        try:
            result = attempt()
            is_ready = ready(result)
            detail = dict(describe(result))
        except RuntimeResourceError as exc:
            if retryable is None or not retryable(exc):
                raise
            error = exc
            is_ready = False
            detail = {"error": str(exc)}

        elapsed = max(0.0, clock() - started)
        status = ready_status if is_ready else "WAITING_FOR_CAPACITY"
        event = {
            "schema_version": 1,
            "created_utc": utc_now(),
            "attempt": attempt_count,
            "status": status,
            "elapsed_wait_seconds": elapsed,
            "scientific_matrix_changed": False,
            **detail,
        }
        _append(events_path, event)
        if on_event is not None:
            on_event(event)

        state = {
            "schema_version": 1,
            "status": status,
            "started_utc": started_utc,
            "updated_utc": event["created_utc"],
            "attempt_count": attempt_count,
            "wait_timeout_seconds": timeout,
            "poll_seconds": interval,
            "elapsed_wait_seconds": elapsed,
            "events_path": str(events_path),
            "last_event": event,
            "scientific_matrix_changed": False,
            "running_workers_resized": False,
        }

        if is_ready:
            if result is None:
                raise RuntimeResourceError("ready capacity attempt has no result")
            atomic_write_json(state_path, state)
            result[metadata_key] = {
                "status": ready_status,
                "path": str(state_path),
                "events_path": str(events_path),
                "attempt_count": attempt_count,
                "elapsed_wait_seconds": elapsed,
                "scientific_matrix_changed": False,
                **detail,
            }
            return result

        if timeout >= 0 and elapsed >= timeout:
            state["status"] = (
                "BLOCKED_NO_WAIT" if timeout == 0 else "BLOCKED_WAIT_TIMEOUT"
            )
            atomic_write_json(state_path, state)
            if error is not None and timeout == 0:
                raise error
            raise RuntimeResourceError(
                f"{timeout_prefix}: elapsed={elapsed:.3f}s"
            )

        atomic_write_json(state_path, state)
        remaining = None if timeout < 0 else max(0.0, timeout - elapsed)
        sleep(interval if remaining is None else min(interval, remaining))


def plan_capacity_shortage(
    error: RuntimeResourceError,
    *,
    work_dir: str | Path,
) -> bool:
    """Retry only explicit or structured resource-capacity planning failures."""

    message = str(error)
    if any(value in message for value in PLAN_CAPACITY_MESSAGES):
        return True
    if NO_VALID_CANDIDATE_MESSAGE not in message:
        return False

    summaries = sorted(
        Path(work_dir).resolve().glob(
            "_runtime_resource_probe/w0_throughput/"
            "workers-*/BENCHMARK_SUMMARY.json"
        )
    )
    if not summaries:
        return False
    for path in summaries:
        row = load_json(path)
        concurrency = int(row.get("concurrency", 0) or 0)
        if concurrency < 1 or int(row.get("completed", 0) or 0) != concurrency:
            return False
        if any(
            int(row.get(key, 0) or 0) != 0
            for key in (
                "failed",
                "timed_out",
                "controller_terminated",
                "orphan_process_groups",
            )
        ):
            return False
        if float(row.get("measured_candidate_cpu_cores", 0.0) or 0.0) <= 0:
            return False
        if int(row.get("aggregate_peak_rss_bytes", 0) or 0) <= 0:
            return False
        if row.get("valid") is True:
            return False
        if (
            row.get("cpu_capacity_ok") is True
            and row.get("memory_capacity_ok") is True
        ):
            return False
    return True


def wait_for_runtime_plan(
    *,
    plan_once: Attempt,
    work_dir: str | Path,
    wait_timeout_seconds: float,
    poll_seconds: float,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    on_event: EventSink | None = None,
) -> dict[str, Any]:
    work = Path(work_dir).resolve()
    return _wait(
        attempt=plan_once,
        ready=lambda _result: True,
        describe=lambda _result: {},
        retryable=lambda error: plan_capacity_shortage(error, work_dir=work),
        work_dir=work,
        state_name="RUNTIME_PLAN_CAPACITY_WAIT.json",
        events_name="RUNTIME_PLAN_CAPACITY_WAIT.jsonl",
        metadata_key="plan_capacity_wait",
        ready_status="PLANNED",
        timeout_prefix="RUNTIME_PLAN_CAPACITY_WAIT_TIMEOUT",
        wait_timeout_seconds=wait_timeout_seconds,
        poll_seconds=poll_seconds,
        clock=clock,
        sleep=sleep,
        on_event=on_event,
    )


def wait_for_runtime_admission(
    *,
    admit_once: Callable[..., dict[str, Any]],
    work_dir: str | Path,
    proposed_workers: int,
    selection_digest: str,
    revalidate_kwargs_factory: Callable[[], Mapping[str, Any]],
    wait_timeout_seconds: float,
    poll_seconds: float,
    minimum_admitted_workers: int,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    on_event: EventSink | None = None,
) -> dict[str, Any]:
    if proposed_workers < 1:
        raise RuntimeResourceError("proposed_workers must be positive")
    if not selection_digest:
        raise RuntimeResourceError("selection_digest must be non-empty")
    if not 1 <= minimum_admitted_workers <= proposed_workers:
        raise RuntimeResourceError(
            "minimum_admitted_workers must be positive and cannot exceed proposed_workers"
        )

    def attempt() -> dict[str, Any]:
        return admit_once(
            work_dir=Path(work_dir).resolve(),
            proposed_workers=proposed_workers,
            selection_digest=selection_digest,
            revalidate_kwargs=dict(revalidate_kwargs_factory()),
            allow_zero=True,
        )

    def admission(result: Mapping[str, Any]) -> Mapping[str, Any]:
        value = result.get("runtime_admission")
        if not isinstance(value, Mapping):
            raise RuntimeResourceError("capacity attempt lacks runtime admission")
        if int(value.get("proposed_workers", 0) or 0) != proposed_workers:
            raise RuntimeResourceError("capacity attempt changed proposed workers")
        if value.get("selection_digest") != selection_digest:
            raise RuntimeResourceError("capacity attempt changed selection digest")
        admitted = int(value.get("admitted_workers", -1) or 0)
        if not 0 <= admitted <= proposed_workers:
            raise RuntimeResourceError("capacity attempt has invalid admitted workers")
        expected = "ALLOW" if admitted > 0 else "BLOCK"
        if value.get("decision") != expected:
            raise RuntimeResourceError("capacity attempt has inconsistent decision")
        return value

    def describe(result: Mapping[str, Any]) -> Mapping[str, Any]:
        value = admission(result)
        return {
            "proposed_workers": proposed_workers,
            "minimum_admitted_workers": minimum_admitted_workers,
            "admitted_workers": int(value.get("admitted_workers", 0) or 0),
            "selection_digest": selection_digest,
            "admission_path": value.get("path"),
            "revalidation_path": value.get("revalidation_path"),
            "admission_reason": value.get("reason"),
            "capacity": value.get("capacity"),
        }

    return _wait(
        attempt=attempt,
        ready=lambda result: int(
            admission(result).get("admitted_workers", 0) or 0
        )
        >= minimum_admitted_workers,
        describe=describe,
        retryable=None,
        work_dir=Path(work_dir).resolve(),
        state_name="RUNTIME_CAPACITY_WAIT.json",
        events_name="RUNTIME_CAPACITY_WAIT.jsonl",
        metadata_key="capacity_wait",
        ready_status="ADMITTED",
        timeout_prefix="RUNTIME_CAPACITY_WAIT_TIMEOUT",
        wait_timeout_seconds=wait_timeout_seconds,
        poll_seconds=poll_seconds,
        clock=clock,
        sleep=sleep,
        on_event=on_event,
    )
