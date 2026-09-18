"""Method-agnostic execution primitives for E8 multitask experiments.

This module owns mechanics that should not change when a scientific method is
added: cell identity validation, seed barriers, nominal batch geometry, and
bounded dynamic scheduling. Scientific method dispatch is supplied through
callbacks; concrete method names and method-specific hyperparameters do not
belong here.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar


class CellLike(Protocol):
    """Minimum cell surface required by generic orchestration."""

    @property
    def key(self) -> str: ...

    task: str
    method: str
    seed: int
    stage: str


TCell = TypeVar("TCell", bound=CellLike)


@dataclass(frozen=True)
class ExecutionGeometry:
    """Deterministic scheduling geometry independent of method semantics."""

    cell_count: int
    slot_count: int
    gpu_ids: tuple[int, ...]
    slots_per_gpu: int
    seed_barrier: bool
    seed_order: tuple[int, ...]
    expected_cells_by_seed: Mapping[int, int]
    nominal_batch_count: int


def validate_unique_cell_keys(cells: Sequence[CellLike]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for cell in cells:
        if cell.key in seen:
            duplicates.add(cell.key)
        seen.add(cell.key)
    if duplicates:
        raise ValueError(f"Duplicate E8 cell keys: {sorted(duplicates)}")


def _validated_seed_order(
    cells: Sequence[CellLike],
    seed_order: Sequence[int] | None,
) -> tuple[int, ...]:
    observed_first_seen: list[int] = []
    observed_set: set[int] = set()
    for cell in cells:
        seed = int(cell.seed)
        if seed not in observed_set:
            observed_set.add(seed)
            observed_first_seen.append(seed)
    if seed_order is None:
        return tuple(observed_first_seen)
    configured = tuple(int(seed) for seed in seed_order)
    if len(configured) != len(set(configured)):
        raise ValueError("seed_order must contain unique seeds")
    if set(configured) != observed_set:
        raise ValueError(
            "seed_order must contain exactly the seeds present in the cell plan: "
            f"configured={configured}, observed={tuple(observed_first_seen)}"
        )
    return configured


def ordered_seed_groups(
    cells: Sequence[TCell],
    *,
    seed_order: Sequence[int] | None = None,
) -> tuple[tuple[int, tuple[TCell, ...]], ...]:
    """Group cells by seed while preserving within-seed cell order.

    When ``seed_order`` is provided, it is authoritative and must contain
    exactly the seeds present in ``cells``. This lets a reviewed protocol keep
    an explicit hard barrier order even if a caller later changes plan ordering.
    """

    order = _validated_seed_order(cells, seed_order)
    groups: dict[int, list[TCell]] = {seed: [] for seed in order}
    for cell in cells:
        groups[int(cell.seed)].append(cell)
    return tuple((seed, tuple(groups[seed])) for seed in order)


def nominal_batches(
    cells: Sequence[TCell],
    *,
    slot_count: int,
    seed_barrier: bool,
    seed_order: Sequence[int] | None = None,
) -> tuple[tuple[TCell, ...], ...]:
    """Return deterministic audit batches; they are not runtime wave barriers."""

    if slot_count <= 0:
        raise ValueError("slot_count must be positive")
    validate_unique_cell_keys(cells)
    batches: list[tuple[TCell, ...]] = []
    groups: Iterable[tuple[int, tuple[TCell, ...]]]
    if seed_barrier:
        groups = ordered_seed_groups(cells, seed_order=seed_order)
    else:
        _validated_seed_order(cells, seed_order)
        groups = ((0, tuple(cells)),)
    for _, group in groups:
        for offset in range(0, len(group), slot_count):
            batches.append(tuple(group[offset : offset + slot_count]))
    return tuple(batches)


def execution_geometry(
    cells: Sequence[CellLike],
    *,
    gpu_ids: Sequence[int],
    slots_per_gpu: int,
    max_concurrent_cells: int,
    seed_barrier: bool,
    seed_order: Sequence[int] | None = None,
) -> ExecutionGeometry:
    if not gpu_ids:
        raise ValueError("gpu_ids must be non-empty")
    if len({int(value) for value in gpu_ids}) != len(gpu_ids):
        raise ValueError("gpu_ids must be unique")
    if slots_per_gpu <= 0:
        raise ValueError("slots_per_gpu must be positive")
    physical_capacity = len(gpu_ids) * slots_per_gpu
    if max_concurrent_cells <= 0 or max_concurrent_cells > physical_capacity:
        raise ValueError(
            "max_concurrent_cells must be positive and no larger than GPU slot capacity"
        )
    validate_unique_cell_keys(cells)
    groups = ordered_seed_groups(cells, seed_order=seed_order)
    expected = {seed: len(group) for seed, group in groups}
    batches = nominal_batches(
        cells,
        slot_count=max_concurrent_cells,
        seed_barrier=seed_barrier,
        seed_order=seed_order,
    )
    return ExecutionGeometry(
        cell_count=len(cells),
        slot_count=max_concurrent_cells,
        gpu_ids=tuple(int(value) for value in gpu_ids),
        slots_per_gpu=int(slots_per_gpu),
        seed_barrier=bool(seed_barrier),
        seed_order=tuple(seed for seed, _ in groups),
        expected_cells_by_seed=expected,
        nominal_batch_count=len(batches),
    )


def plan_rows(
    batches: Sequence[Sequence[TCell]],
    *,
    gpu_ids: Sequence[int],
    project_cell: Callable[[TCell], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Materialize deterministic plan rows without interpreting method parameters."""

    if not gpu_ids:
        raise ValueError("gpu_ids must be non-empty")
    rows: list[dict[str, Any]] = []
    seen: list[TCell] = []
    for batch_index, batch in enumerate(batches, start=1):
        for slot, cell in enumerate(batch):
            row: dict[str, Any] = {
                "wave": batch_index,
                "nominal_batch": batch_index,
                "slot": slot,
                "gpu_id": int(gpu_ids[slot % len(gpu_ids)]),
                "cell_key": cell.key,
                "task": cell.task,
                "method": cell.method,
            }
            projected = dict(project_cell(cell))
            collisions = sorted(set(row).intersection(projected))
            if collisions:
                raise ValueError(
                    f"Method plan projection attempted to overwrite: {collisions}"
                )
            row.update(projected)
            row.update({"seed": int(cell.seed), "stage": cell.stage})
            rows.append(row)
            seen.append(cell)
    validate_unique_cell_keys(seen)
    return rows

@dataclass(frozen=True)
class SchedulerCallbacks:
    """Hooks supplied by the E8 scientific/runtime layers.

    ``run_cell`` is the only method-dependent operation. ``after_success`` is
    part of the success transaction: a seed barrier is not released until it
    returns successfully. The scheduler otherwise treats returned mappings as
    opaque apart from ``returncode``.
    """

    run_cell: Callable[[CellLike, int, int], Mapping[str, Any]]
    record_event: Callable[[Mapping[str, Any]], None] | None = None
    after_success: Callable[[CellLike, Mapping[str, Any]], None] | None = None


def run_dynamic_queue(
    cells: Sequence[TCell],
    *,
    gpu_ids: Sequence[int],
    slots_per_gpu: int,
    max_concurrent_cells: int,
    seed_barrier: bool,
    callbacks: SchedulerCallbacks,
    seed_order: Sequence[int] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Run cells with bounded slots and an optional hard seed-batch barrier.

    The primitive does not know method names, hyperparameters, result schemas,
    recovery formats, or scientific metrics. Any exception, non-zero
    ``returncode``, or ``after_success`` failure stops release of new work;
    workers that had already started a cell are allowed to report terminal
    state. Under a seed barrier, the next seed is enqueued only after every cell
    in the active seed has completed *and* its success hook has returned.
    """

    geometry = execution_geometry(
        cells,
        gpu_ids=gpu_ids,
        slots_per_gpu=slots_per_gpu,
        max_concurrent_cells=max_concurrent_cells,
        seed_barrier=seed_barrier,
        seed_order=seed_order,
    )
    if not cells:
        return ()

    pending: queue.Queue[TCell] = queue.Queue()
    stop = threading.Event()
    lock = threading.Lock()
    event_lock = threading.Lock()
    results: list[dict[str, Any]] = []
    seed_groups = ordered_seed_groups(cells, seed_order=geometry.seed_order)
    seed_index = 0
    seed_completed = {seed: 0 for seed, _ in seed_groups}
    seed_expected = {seed: len(group) for seed, group in seed_groups}

    if seed_barrier:
        for cell in seed_groups[0][1]:
            pending.put(cell)
    else:
        for cell in cells:
            pending.put(cell)

    def emit(event: Mapping[str, Any]) -> None:
        if callbacks.record_event is not None:
            with event_lock:
                callbacks.record_event(event)

    def worker(slot: int, gpu_id: int) -> None:
        nonlocal seed_index
        while True:
            if stop.is_set():
                return
            try:
                cell = pending.get(timeout=0.05)
            except queue.Empty:
                with lock:
                    finished = len(results)
                    no_more_seed_work = (
                        not seed_barrier
                        or seed_index == len(seed_groups) - 1
                        and seed_completed[seed_groups[seed_index][0]]
                        == seed_expected[seed_groups[seed_index][0]]
                    )
                if pending.empty() and no_more_seed_work and finished >= len(cells):
                    return
                continue

            if stop.is_set():
                pending.task_done()
                return

            emit(
                {
                    "event": "start",
                    "cell_key": cell.key,
                    "task": cell.task,
                    "method": cell.method,
                    "seed": int(cell.seed),
                    "slot": slot,
                    "gpu_id": gpu_id,
                }
            )
            try:
                raw = dict(callbacks.run_cell(cell, slot, gpu_id))
                raw.setdefault("returncode", 0)
            except Exception as exc:  # noqa: BLE001  # pragma: no cover
                raw = {
                    "returncode": 1,
                    "error": f"{type(exc).__name__}: {exc}",
                }

            raw.update(
                {
                    "cell_key": cell.key,
                    "task": cell.task,
                    "method": cell.method,
                    "seed": int(cell.seed),
                    "slot": slot,
                    "gpu_id": gpu_id,
                }
            )
            succeeded = int(raw.get("returncode", 1)) == 0
            if succeeded and callbacks.after_success is not None:
                try:
                    callbacks.after_success(cell, raw)
                except Exception as exc:  # noqa: BLE001 - callback failure evidence
                    succeeded = False
                    existing_code = int(raw.get("returncode", 0))
                    raw["returncode"] = existing_code if existing_code != 0 else 1
                    raw["error"] = f"after_success {type(exc).__name__}: {exc}"

            emit({"event": "finish", **raw})
            with lock:
                results.append(raw)
                if not succeeded:
                    stop.set()
                elif seed_barrier:
                    seed = int(cell.seed)
                    seed_completed[seed] += 1
                    active_seed = seed_groups[seed_index][0]
                    if (
                        seed == active_seed
                        and seed_completed[active_seed] == seed_expected[active_seed]
                        and seed_index < len(seed_groups) - 1
                    ):
                        seed_index += 1
                        for candidate in seed_groups[seed_index][1]:
                            pending.put(candidate)
            pending.task_done()

    with ThreadPoolExecutor(max_workers=geometry.slot_count) as executor:
        futures = [
            executor.submit(worker, slot, geometry.gpu_ids[slot % len(geometry.gpu_ids)])
            for slot in range(geometry.slot_count)
        ]
        for future in as_completed(futures):
            future.result()

    by_key = {str(row["cell_key"]): row for row in results}
    return tuple(dict(by_key[cell.key]) for cell in cells if cell.key in by_key)


@dataclass(frozen=True)
class DynamicExecutionHooks:
    """Composition callbacks for one dynamic execution lifecycle."""

    execute_cell: Callable[[CellLike, int, bool], Mapping[str, Any]]
    is_reusable_complete: Callable[[CellLike], bool]
    validate_completed_cell: Callable[[CellLike], None]
    publish_task: Callable[[str], Mapping[str, Any] | None]
    reusable_cell_count: Callable[[], int]
    publish_checkpoint: Callable[[], Mapping[str, Any]] | None
    record_event: Callable[[Mapping[str, Any]], None]
    protocol_diagnostic: Callable[[], Mapping[str, Any]]


def run_dynamic_execution(
    cells: Sequence[TCell],
    *,
    gpu_ids: Sequence[int],
    slots_per_gpu: int,
    max_concurrent_cells: int,
    seed_barrier: bool,
    seed_order: Sequence[int] | None,
    nominal_batch: Mapping[str, int],
    force: bool,
    retry_incomplete: bool,
    recovery_interval: int,
    initial_reusable_count: int,
    scheduler_run_id: str,
    wave_count: int,
    wave_count_role: str,
    experiment_id_value: str,
    execution_class: str,
    queue_events_path: str,
    scientific_status: str,
    engineering_placeholder_backend: bool,
    hooks: DynamicExecutionHooks,
) -> dict[str, Any]:
    """Run the complete method-agnostic dynamic scheduling lifecycle."""

    geometry = execution_geometry(
        cells,
        gpu_ids=gpu_ids,
        slots_per_gpu=slots_per_gpu,
        max_concurrent_cells=max_concurrent_cells,
        seed_barrier=seed_barrier,
        seed_order=seed_order,
    )
    if recovery_interval <= 0:
        raise ValueError("recovery_interval must be positive")

    task_result_lock = threading.Lock()
    checkpoint_lock = threading.Lock()
    task_results: dict[str, dict[str, Any]] = {}
    last_checkpoint_count = (
        initial_reusable_count // recovery_interval
    ) * recovery_interval

    def publish_completed_task(task: str) -> None:
        with task_result_lock:
            if task in task_results:
                return
            value = hooks.publish_task(task)
            if value is not None:
                task_results[task] = dict(value)

    def run_cell(cell: TCell, slot: int, gpu_id: int) -> Mapping[str, Any]:
        del slot
        reusable_complete = hooks.is_reusable_complete(cell)
        child_force = force or (
            retry_incomplete and not reusable_complete
        )
        result = dict(hooks.execute_cell(cell, gpu_id, child_force))
        if int(result["returncode"]) == 0:
            try:
                hooks.validate_completed_cell(cell)
            except Exception as exc:  # noqa: BLE001 - child evidence becomes failure
                result["returncode"] = 75
                result["cell_completion_error"] = (
                    f"{type(exc).__name__}: {exc}"
                )
        result["nominal_batch"] = nominal_batch[cell.key]
        return result

    def after_success(cell: TCell, row: Mapping[str, Any]) -> None:
        nonlocal last_checkpoint_count
        mutable = row if isinstance(row, dict) else dict(row)
        if hooks.publish_checkpoint is not None:
            try:
                with checkpoint_lock:
                    completed_count = hooks.reusable_cell_count()
                    if (
                        completed_count
                        >= last_checkpoint_count + recovery_interval
                    ):
                        checkpoint = hooks.publish_checkpoint()
                        last_checkpoint_count = int(
                            checkpoint["completed_cells"]
                        )
                        mutable["recovery_checkpoint"] = checkpoint["package"]
                        mutable[
                            "recovery_checkpoint_completed_cells"
                        ] = last_checkpoint_count
            except Exception:
                mutable["returncode"] = 74
                raise
        publish_completed_task(cell.task)

    results = list(
        run_dynamic_queue(
            cells,
            gpu_ids=gpu_ids,
            slots_per_gpu=slots_per_gpu,
            max_concurrent_cells=geometry.slot_count,
            seed_barrier=seed_barrier,
            seed_order=seed_order,
            callbacks=SchedulerCallbacks(
                run_cell=run_cell,
                record_event=hooks.record_event,
                after_success=after_success,
            ),
        )
    )
    results.sort(key=lambda row: str(row["cell_key"]))
    failures = [
        row for row in results if int(row["returncode"]) != 0
    ]
    returned_keys = {str(row["cell_key"]) for row in results}
    completed_keys = {
        str(row["cell_key"])
        for row in results
        if int(row["returncode"]) == 0
    }
    unscheduled = [
        cell.key for cell in cells if cell.key not in returned_keys
    ]
    seed_expected = (
        dict(geometry.expected_cells_by_seed) if seed_barrier else {}
    )
    seed_completed = (
        {
            seed: sum(
                int(row["returncode"]) == 0
                and int(row["seed"]) == seed
                for row in results
            )
            for seed in geometry.seed_order
        }
        if seed_barrier
        else {}
    )
    protocol_diagnostic = (
        dict(hooks.protocol_diagnostic())
        if not failures and not unscheduled
        else {
            "status": "PENDING",
            "result_gate": False,
            "controls_task_transfer_release": False,
        }
    )
    manifest = {
        "schema_version": 1,
        "experiment_id": experiment_id_value,
        "scheduler": "dynamic_slot_queue",
        "scheduler_run_id": scheduler_run_id,
        "wave_barriers": False,
        "wave_count": wave_count,
        "wave_count_role": wave_count_role,
        "seed_batch_barriers": seed_barrier,
        "seed_batch_order": (
            list(geometry.seed_order) if seed_barrier else []
        ),
        "seed_batch_expected_cells": {
            str(key): value for key, value in seed_expected.items()
        },
        "seed_batch_completed_cells": {
            str(key): value for key, value in seed_completed.items()
        },
        "execution_class": execution_class,
        "slot_count": geometry.slot_count,
        "gpu_ids": list(gpu_ids),
        "slots_per_gpu": slots_per_gpu,
        "countdown_protocol_diagnostic": protocol_diagnostic,
        "countdown_result_controls_transfer_release": False,
        "expected_cells": len(cells),
        "completed_cells": len(completed_keys),
        "results": results,
        "failed_cells": [row["cell_key"] for row in failures],
        "unscheduled_cells": unscheduled,
        "queue_events": queue_events_path,
        "analysis_ready_tasks": sorted(task_results),
        "task_results": task_results,
        "complete": (
            not failures
            and not unscheduled
            and len(completed_keys) == len(cells)
        ),
        "scientific_status": scientific_status,
        "engineering_placeholder_backend": engineering_placeholder_backend,
    }
    if failures or unscheduled:
        raise RuntimeError(
            "Cold-start scheduling stopped fail-closed; "
            f"failed={manifest['failed_cells']} unscheduled={len(unscheduled)}"
        )
    return manifest

