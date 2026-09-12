"""Method-agnostic execution primitives for E8 multitask experiments.

This module owns mechanics that should not change when a scientific method is
added: cell identity validation, seed barriers, nominal batch geometry, and
bounded dynamic scheduling.  Scientific method dispatch is supplied through
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
TResult = TypeVar("TResult")


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
    keys = [cell.key for cell in cells]
    if len(keys) != len(set(keys)):
        duplicates = sorted({key for key in keys if keys.count(key) > 1})
        raise ValueError(f"Duplicate E8 cell keys: {duplicates}")


def ordered_seed_groups(cells: Sequence[TCell]) -> tuple[tuple[int, tuple[TCell, ...]], ...]:
    """Group cells by first-seen seed while preserving cell order."""

    groups: dict[int, list[TCell]] = {}
    order: list[int] = []
    for cell in cells:
        if cell.seed not in groups:
            groups[cell.seed] = []
            order.append(cell.seed)
        groups[cell.seed].append(cell)
    return tuple((seed, tuple(groups[seed])) for seed in order)


def nominal_batches(
    cells: Sequence[TCell],
    *,
    slot_count: int,
    seed_barrier: bool,
) -> tuple[tuple[TCell, ...], ...]:
    """Return deterministic audit batches; they are not runtime wave barriers."""

    if slot_count <= 0:
        raise ValueError("slot_count must be positive")
    validate_unique_cell_keys(cells)
    batches: list[tuple[TCell, ...]] = []
    groups: Iterable[tuple[int, tuple[TCell, ...]]]
    if seed_barrier:
        groups = ordered_seed_groups(cells)
    else:
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
) -> ExecutionGeometry:
    if not gpu_ids:
        raise ValueError("gpu_ids must be non-empty")
    if slots_per_gpu <= 0:
        raise ValueError("slots_per_gpu must be positive")
    physical_capacity = len(gpu_ids) * slots_per_gpu
    if max_concurrent_cells <= 0 or max_concurrent_cells > physical_capacity:
        raise ValueError(
            "max_concurrent_cells must be positive and no larger than GPU slot capacity"
        )
    validate_unique_cell_keys(cells)
    groups = ordered_seed_groups(cells)
    expected = {seed: len(group) for seed, group in groups}
    batches = nominal_batches(
        cells,
        slot_count=max_concurrent_cells,
        seed_barrier=seed_barrier,
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


@dataclass(frozen=True)
class SchedulerCallbacks:
    """Hooks supplied by the E8 scientific/runtime layers.

    `run_cell` is the only required method-dependent operation.  The scheduler
    itself treats the returned mapping as opaque apart from `returncode`.
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
) -> tuple[dict[str, Any], ...]:
    """Run cells with bounded slots and an optional hard seed-batch barrier.

    This primitive deliberately does not know method names, hyperparameters,
    result schemas, recovery formats, or scientific metrics.  Any exception or
    non-zero `returncode` stops release of new work; already-running workers are
    allowed to report their terminal state.
    """

    geometry = execution_geometry(
        cells,
        gpu_ids=gpu_ids,
        slots_per_gpu=slots_per_gpu,
        max_concurrent_cells=max_concurrent_cells,
        seed_barrier=seed_barrier,
    )
    if not cells:
        return ()

    pending: queue.Queue[TCell] = queue.Queue()
    stop = threading.Event()
    lock = threading.Lock()
    results: list[dict[str, Any]] = []
    seed_groups = ordered_seed_groups(cells)
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
                if stop.is_set() and pending.empty():
                    return
                continue

            emit(
                {
                    "event": "start",
                    "cell_key": cell.key,
                    "task": cell.task,
                    "method": cell.method,
                    "seed": cell.seed,
                    "slot": slot,
                    "gpu_id": gpu_id,
                }
            )
            try:
                raw = dict(callbacks.run_cell(cell, slot, gpu_id))
                raw.setdefault("returncode", 0)
            except Exception as exc:  # pragma: no cover - exercised by callers.
                raw = {
                    "returncode": 1,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            raw.update(
                {
                    "cell_key": cell.key,
                    "task": cell.task,
                    "method": cell.method,
                    "seed": cell.seed,
                    "slot": slot,
                    "gpu_id": gpu_id,
                }
            )
            emit({"event": "finish", **raw})

            succeeded = int(raw.get("returncode", 1)) == 0
            with lock:
                results.append(raw)
                if not succeeded:
                    stop.set()
                elif seed_barrier:
                    seed_completed[cell.seed] += 1
                    active_seed = seed_groups[seed_index][0]
                    if (
                        cell.seed == active_seed
                        and seed_completed[active_seed] == seed_expected[active_seed]
                        and seed_index < len(seed_groups) - 1
                    ):
                        seed_index += 1
                        for candidate in seed_groups[seed_index][1]:
                            pending.put(candidate)
            if succeeded and callbacks.after_success is not None:
                try:
                    callbacks.after_success(cell, raw)
                except Exception:
                    stop.set()
                    raise
            pending.task_done()

    with ThreadPoolExecutor(max_workers=geometry.slot_count) as executor:
        futures = [
            executor.submit(worker, slot, geometry.gpu_ids[slot % len(geometry.gpu_ids)])
            for slot in range(geometry.slot_count)
        ]
        for future in as_completed(futures):
            future.result()

    by_key = {str(row["cell_key"]): row for row in results}
    ordered = tuple(dict(by_key[cell.key]) for cell in cells if cell.key in by_key)
    return ordered
