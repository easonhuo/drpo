"""Method-agnostic execution primitives for E8 multitask experiments.

This module owns mechanics that should not change when a scientific method is
added: cell identity validation, seed barriers, nominal batch geometry, and
bounded dynamic scheduling. Scientific method dispatch is supplied through
callbacks; concrete method names and method-specific hyperparameters do not
belong here.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
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
    should_force_retry: Callable[[CellLike], bool]
    completed_cell_error: Callable[[CellLike], str | None]
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
        child_force = force or (
            retry_incomplete and hooks.should_force_retry(cell)
        )
        result = dict(hooks.execute_cell(cell, gpu_id, child_force))
        if int(result["returncode"]) == 0:
            completion_error = hooks.completed_cell_error(cell)
            if completion_error is not None:
                result["returncode"] = 75
                result["cell_completion_error"] = completion_error
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
    return manifest


@dataclass(frozen=True)
class DynamicCommandBindings:
    """Explicit E8 composition callbacks for the dynamic command."""

    build_cells: Callable[[Mapping[str, Any]], Sequence[CellLike]]
    build_waves: Callable[[Mapping[str, Any]], Sequence[Sequence[CellLike]]]
    experiment_id: Callable[[Mapping[str, Any]], str]
    execution_class: Callable[[Mapping[str, Any]], str]
    engineering_self_test: Callable[[Mapping[str, Any]], bool]
    reusable_cell_manifests: Callable[
        [Mapping[str, Any], Path],
        tuple[Mapping[str, Mapping[str, Any]], Mapping[str, str]],
    ]
    run_subprocess_cell: Callable[..., Mapping[str, Any]]
    read_json_object: Callable[[Path], Mapping[str, Any]]
    materialize_task_results: Callable[..., Mapping[str, Mapping[str, Any]]]
    publish_recovery_checkpoint: Callable[..., Mapping[str, Any]]
    protocol_diagnostic: Callable[..., Mapping[str, Any]]
    append_jsonl: Callable[[Path, Mapping[str, Any]], None]
    write_json: Callable[[Path, Any], None]


def cmd_run_dynamic(
    config: Mapping[str, Any],
    config_path: Path,
    output_root: Path,
    *,
    base_model_path: str,
    force: bool,
    retry_incomplete: bool,
    bindings: DynamicCommandBindings,
) -> dict[str, Any]:
    """Compose the E8 dynamic execution lifecycle without method semantics."""

    cells = tuple(bindings.build_cells(config))
    gpu_ids = tuple(int(value) for value in config["execution"]["gpu_ids"])
    slots_per_gpu = int(config["execution"]["slots_per_gpu"])
    seed_barrier = bool(
        config["execution"].get("seed_batch_barriers")
    )
    seed_order = (
        tuple(
            int(value)
            for value in config["execution"].get("seed_batch_order", ())
        )
        if seed_barrier
        else None
    )
    geometry = execution_geometry(
        cells,
        gpu_ids=gpu_ids,
        slots_per_gpu=slots_per_gpu,
        max_concurrent_cells=int(
            config["execution"]["max_concurrent_cells"]
        ),
        seed_barrier=seed_barrier,
        seed_order=seed_order,
    )
    if geometry.slot_count != 16:
        raise RuntimeError(
            "Declared 16-slot capacity is internally inconsistent"
        )

    waves = tuple(bindings.build_waves(config))
    nominal_batch = {
        cell.key: index
        for index, wave in enumerate(waves, 1)
        for cell in wave
    }
    event_path = output_root / "scheduler" / "queue_events.jsonl"
    event_path.parent.mkdir(parents=True, exist_ok=True)
    scheduler_run_id = f"queue-{int(time.time())}-{os.getpid()}"

    recovery_package_value = os.environ.get(
        "E8_COLDSTART_RECOVERY_PACKAGE", ""
    ).strip()
    recovery_package = (
        Path(recovery_package_value).resolve()
        if recovery_package_value
        else None
    )
    recovery_interval = int(
        os.environ.get("E8_COLDSTART_RECOVERY_INTERVAL_CELLS", "5")
    )
    if recovery_interval <= 0:
        raise ValueError(
            "E8_COLDSTART_RECOVERY_INTERVAL_CELLS must be positive"
        )
    initially_reusable, _ = bindings.reusable_cell_manifests(
        config, output_root
    )

    def record(event: Mapping[str, Any]) -> None:
        bindings.append_jsonl(
            event_path,
            {
                "scheduler_run_id": scheduler_run_id,
                **dict(event),
                "unix_time": time.time(),
            },
        )

    def should_force_retry(cell: CellLike) -> bool:
        cell_root = output_root / "cells" / cell.key
        manifest_path = cell_root / "cell_manifest.json"
        reusable_complete = False
        if manifest_path.is_file():
            try:
                reusable_complete = bool(
                    json.loads(
                        manifest_path.read_text(encoding="utf-8")
                    ).get("complete")
                )
            except (OSError, json.JSONDecodeError):
                reusable_complete = False
        return cell_root.exists() and not reusable_complete

    def execute_cell(
        cell: CellLike,
        gpu_id: int,
        child_force: bool,
    ) -> Mapping[str, Any]:
        return bindings.run_subprocess_cell(
            config_path=config_path.resolve(),
            output_root=output_root.resolve(),
            base_model_path=base_model_path,
            cell=cell,
            gpu_id=gpu_id,
            force=child_force,
        )

    def completed_cell_error(cell: CellLike) -> str | None:
        manifest_path = (
            output_root / "cells" / cell.key / "cell_manifest.json"
        )
        try:
            completed_manifest = bindings.read_json_object(manifest_path)
            if (
                completed_manifest.get("complete") is not True
                or completed_manifest.get("evaluation_status") != "complete"
                or completed_manifest.get("nan_inf_failure") is not False
            ):
                raise RuntimeError(
                    "child returned zero without a complete finite cell"
                )
        except (
            OSError,
            ValueError,
            TypeError,
            RuntimeError,
            json.JSONDecodeError,
        ) as exc:
            return f"{type(exc).__name__}: {exc}"
        return None

    def publish_task(task: str) -> Mapping[str, Any] | None:
        ready = bindings.materialize_task_results(
            config,
            output_root,
            tasks=(task,),
        )
        return ready.get(task)

    def reusable_cell_count() -> int:
        current_reusable, _ = bindings.reusable_cell_manifests(
            config,
            output_root,
        )
        return len(current_reusable)

    def publish_checkpoint() -> Mapping[str, Any]:
        if recovery_package is None:
            raise RuntimeError(
                "Recovery checkpoint callback requires a configured package"
            )
        return bindings.publish_recovery_checkpoint(
            config,
            output_root,
            package_output=recovery_package,
        )

    def protocol_diagnostic() -> Mapping[str, Any]:
        return bindings.protocol_diagnostic(
            config,
            output_root,
            destination=(
                output_root
                / "scheduler"
                / "countdown_protocol_diagnostic.json"
            ),
        )

    engineering_self_test = bindings.engineering_self_test(config)
    manifest = run_dynamic_execution(
        cells,
        gpu_ids=gpu_ids,
        slots_per_gpu=slots_per_gpu,
        max_concurrent_cells=geometry.slot_count,
        seed_barrier=seed_barrier,
        seed_order=seed_order,
        nominal_batch=nominal_batch,
        force=force,
        retry_incomplete=retry_incomplete,
        recovery_interval=recovery_interval,
        initial_reusable_count=len(initially_reusable),
        scheduler_run_id=scheduler_run_id,
        wave_count=len(waves),
        wave_count_role=(
            "seed_local_nominal_capacity_audit_only"
            if seed_barrier
            else "nominal_audit_geometry_only_not_scheduling_barrier"
        ),
        experiment_id_value=bindings.experiment_id(config),
        execution_class=bindings.execution_class(config),
        queue_events_path=str(event_path.resolve()),
        scientific_status=(
            "not_run" if engineering_self_test else "pilot"
        ),
        engineering_placeholder_backend=engineering_self_test,
        hooks=DynamicExecutionHooks(
            execute_cell=execute_cell,
            should_force_retry=should_force_retry,
            completed_cell_error=completed_cell_error,
            publish_task=publish_task,
            reusable_cell_count=reusable_cell_count,
            publish_checkpoint=(
                publish_checkpoint
                if recovery_package is not None
                else None
            ),
            record_event=record,
            protocol_diagnostic=protocol_diagnostic,
        ),
    )
    bindings.write_json(
        output_root / "scheduler" / "dynamic_run.json",
        manifest,
    )
    if manifest["failed_cells"] or manifest["unscheduled_cells"]:
        raise RuntimeError(
            "Cold-start scheduling stopped fail-closed; "
            f"failed={manifest['failed_cells']} "
            f"unscheduled={len(manifest['unscheduled_cells'])}"
        )
    return manifest

