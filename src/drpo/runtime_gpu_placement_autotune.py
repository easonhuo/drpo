"""Bounded GPU placement autotuning for independent single-GPU tasks.

This module does not select DDP, tensor parallelism, FSDP, batch size, precision,
or any scientific setting. It only determines how many already-defined one-GPU
workers may safely share each eligible GPU.
"""
from __future__ import annotations

import dataclasses
import math
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from drpo.runtime_resource_autotune import (
    MachineSnapshot,
    RuntimeResourceError,
    atomic_write_json,
    canonical_json_sha256,
    discover_gpus,
    git_state,
    load_json,
    utc_now,
)

ADAPTER_ID = "single_gpu_independent_task_placement_v1"
OOM_SIGNATURES = (
    "cuda out of memory",
    "outofmemoryerror",
    "cublas_status_alloc_failed",
    "failed to allocate",
)


@dataclasses.dataclass(frozen=True)
class GPUConcurrencyProbeResult:
    concurrency: int
    device_id: str
    started_utc: str
    finished_utc: str
    elapsed_seconds: float
    success: bool
    sample_window_completed: bool
    global_deadline_reached: bool
    oom_detected: bool
    worker_returncodes: tuple[int | None, ...]
    initial_free_vram_bytes: int
    minimum_free_vram_bytes: int
    peak_incremental_vram_bytes: int
    log_paths: tuple[str, ...]
    reason: str

    def as_dict(self) -> dict[str, Any]:
        value = dataclasses.asdict(self)
        value["worker_returncodes"] = list(self.worker_returncodes)
        value["log_paths"] = list(self.log_paths)
        return value


def _gpu_free_bytes(device_id: str, *, nvidia_smi: str) -> int:
    inventory = {gpu.index: gpu for gpu in discover_gpus(nvidia_smi=nvidia_smi)}
    gpu = inventory.get(str(device_id))
    if gpu is None:
        raise RuntimeResourceError(f"GPU {device_id} is not visible during placement probe")
    return int(gpu.memory_free_bytes)


def _terminate_process_group(
    process: subprocess.Popen[str], *, terminate_grace_seconds: float
) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + max(0.0, terminate_grace_seconds)
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.1)
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=max(1.0, terminate_grace_seconds))
    except subprocess.TimeoutExpired:
        pass


def _logs_contain_oom(paths: Sequence[Path]) -> bool:
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8", errors="replace").lower()
        except OSError:
            continue
        if any(signature in text for signature in OOM_SIGNATURES):
            return True
    return False


def probe_same_gpu_concurrency(
    *,
    device_id: str,
    concurrency: int,
    command_factory: Callable[[int, Path], Sequence[str]],
    environment: Mapping[str, str] | None,
    log_dir: str | Path,
    sample_seconds: float,
    global_deadline_monotonic: float,
    nvidia_smi: str = "nvidia-smi",
    poll_interval_seconds: float = 1.0,
    minimum_live_seconds: float = 10.0,
    required_free_floor_bytes: int = 0,
    terminate_grace_seconds: float = 15.0,
    working_directory: str | Path | None = None,
) -> GPUConcurrencyProbeResult:
    """Launch ``concurrency`` isolated workers on one GPU for a bounded interval."""

    if concurrency < 1:
        raise RuntimeResourceError("GPU probe concurrency must be positive")
    if sample_seconds <= 0:
        raise RuntimeResourceError("GPU probe sample_seconds must be positive")
    if poll_interval_seconds <= 0:
        raise RuntimeResourceError("GPU probe poll interval must be positive")
    if minimum_live_seconds < 0:
        raise RuntimeResourceError("GPU probe minimum_live_seconds must be non-negative")

    root = Path(log_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    cwd = Path(working_directory).resolve() if working_directory else Path.cwd()
    started_utc = utc_now()
    started = time.monotonic()
    initial_free = _gpu_free_bytes(device_id, nvidia_smi=nvidia_smi)
    minimum_free = initial_free
    processes: list[subprocess.Popen[str]] = []
    handles: list[Any] = []
    log_paths: list[Path] = []
    worker_roots: list[Path] = []
    launch_error: str | None = None
    sample_window_completed = False
    global_deadline_reached = False
    nonzero_seen = False
    observed_returncodes: tuple[int | None, ...] = ()
    elapsed_before_cleanup = 0.0

    try:
        for worker_index in range(concurrency):
            worker_root = root / f"worker_{worker_index:02d}"
            worker_root.mkdir(parents=True, exist_ok=True)
            worker_roots.append(worker_root)
            log_path = root / f"worker_{worker_index:02d}.log"
            log_paths.append(log_path)
            handle = log_path.open("w", encoding="utf-8")
            handles.append(handle)
            try:
                command = [
                    str(item) for item in command_factory(worker_index, worker_root)
                ]
                worker_environment = dict(os.environ)
                if environment is not None:
                    worker_environment.update(
                        {str(key): str(value) for key, value in environment.items()}
                    )
                worker_environment["CUDA_VISIBLE_DEVICES"] = str(device_id)
                worker_environment["DRPO_RUNTIME_RESOURCE_PROBE"] = "1"
                worker_environment["DRPO_RUNTIME_RESOURCE_PROBE_CONCURRENCY"] = str(
                    concurrency
                )
                handle.write(f"GPU={device_id}\nCOMMAND={' '.join(command)}\n")
                handle.flush()
                process = subprocess.Popen(
                    command,
                    cwd=cwd,
                    env=worker_environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
            except Exception as exc:  # noqa: BLE001
                handle.write(f"LAUNCH_ERROR={exc}\n")
                handle.flush()
                launch_error = str(exc)
                break
            processes.append(process)

        sample_deadline = min(started + sample_seconds, global_deadline_monotonic)
        while launch_error is None and processes:
            now = time.monotonic()
            if now >= sample_deadline:
                sample_window_completed = now >= started + sample_seconds
                global_deadline_reached = now >= global_deadline_monotonic
                break
            try:
                minimum_free = min(
                    minimum_free,
                    _gpu_free_bytes(device_id, nvidia_smi=nvidia_smi),
                )
            except RuntimeResourceError:
                launch_error = "GPU became invisible during placement probe"
                break
            returncodes = [process.poll() for process in processes]
            if any(code is not None and code != 0 for code in returncodes):
                nonzero_seen = True
                break
            if all(code == 0 for code in returncodes):
                sample_window_completed = True
                break
            time.sleep(poll_interval_seconds)

        elapsed_before_cleanup = time.monotonic() - started
        observed_returncodes = tuple(process.poll() for process in processes)
    finally:
        for process in processes:
            _terminate_process_group(
                process,
                terminate_grace_seconds=terminate_grace_seconds,
            )
        for handle in handles:
            try:
                handle.close()
            except OSError:
                pass
        for worker_root in worker_roots:
            shutil.rmtree(worker_root, ignore_errors=True)

    try:
        minimum_free = min(
            minimum_free,
            _gpu_free_bytes(device_id, nvidia_smi=nvidia_smi),
        )
    except RuntimeResourceError:
        pass
    oom_detected = _logs_contain_oom(log_paths)
    peak_incremental = max(0, initial_free - minimum_free)
    lived_long_enough = elapsed_before_cleanup >= minimum_live_seconds
    floor_ok = minimum_free >= max(0, int(required_free_floor_bytes))
    clean_exit_or_live = bool(processes) and not nonzero_seen and all(
        code in (None, 0) for code in observed_returncodes
    )
    success = (
        launch_error is None
        and len(processes) == concurrency
        and clean_exit_or_live
        and lived_long_enough
        and not oom_detected
        and floor_ok
        and not global_deadline_reached
    )
    if launch_error is not None:
        reason = f"launch_failure:{launch_error}"
    elif oom_detected:
        reason = "oom_signature_detected"
    elif nonzero_seen:
        reason = "worker_nonzero_exit"
    elif not floor_ok:
        reason = "free_vram_floor_crossed"
    elif global_deadline_reached:
        reason = "global_probe_deadline_reached"
    elif not lived_long_enough:
        reason = "insufficient_live_interval"
    elif success:
        reason = "bounded_concurrency_probe_passed"
    else:
        reason = "inconclusive_probe"

    return GPUConcurrencyProbeResult(
        concurrency=concurrency,
        device_id=str(device_id),
        started_utc=started_utc,
        finished_utc=utc_now(),
        elapsed_seconds=time.monotonic() - started,
        success=success,
        sample_window_completed=sample_window_completed,
        global_deadline_reached=global_deadline_reached,
        oom_detected=oom_detected,
        worker_returncodes=observed_returncodes,
        initial_free_vram_bytes=initial_free,
        minimum_free_vram_bytes=minimum_free,
        peak_incremental_vram_bytes=peak_incremental,
        log_paths=tuple(str(path) for path in log_paths),
        reason=reason,
    )


def derive_slots_per_gpu(
    *,
    free_vram_bytes: int,
    single_worker_peak_vram_bytes: int,
    device_count: int,
    total_tasks: int,
    host_worker_limit_total: int,
    gpu_memory_headroom_fraction: float,
    per_worker_vram_safety_factor: float,
    max_slots_per_gpu: int,
) -> dict[str, int]:
    """Derive a bounded candidate from measured single-worker capacity."""

    if free_vram_bytes <= 0 or single_worker_peak_vram_bytes <= 0:
        raise RuntimeResourceError("measured GPU memory values must be positive")
    if device_count < 1 or total_tasks < 1 or host_worker_limit_total < 1:
        raise RuntimeResourceError("device, task, and host-worker limits must be positive")
    if not 0.0 <= gpu_memory_headroom_fraction < 0.9:
        raise RuntimeResourceError("GPU memory headroom must be in [0, 0.9)")
    if per_worker_vram_safety_factor < 1.0:
        raise RuntimeResourceError("per-worker VRAM safety factor must be >= 1")
    if max_slots_per_gpu < 1:
        raise RuntimeResourceError("max_slots_per_gpu must be positive")

    usable_vram = math.floor(free_vram_bytes * (1.0 - gpu_memory_headroom_fraction))
    reserved_per_worker = math.ceil(
        single_worker_peak_vram_bytes * per_worker_vram_safety_factor
    )
    vram_limit = usable_vram // reserved_per_worker
    host_limit_per_device = host_worker_limit_total // device_count
    task_limit_per_device = math.ceil(total_tasks / device_count)
    candidate = min(
        max_slots_per_gpu,
        vram_limit,
        host_limit_per_device,
        task_limit_per_device,
    )
    return {
        "candidate": max(1, candidate),
        "usable_vram_bytes": usable_vram,
        "reserved_vram_bytes_per_worker": reserved_per_worker,
        "vram_limit_per_device": max(0, vram_limit),
        "host_limit_per_device": max(0, host_limit_per_device),
        "task_limit_per_device": max(1, task_limit_per_device),
    }


def bounded_backoff_candidates(initial: int) -> list[int]:
    if initial < 1:
        raise RuntimeResourceError("initial GPU slot candidate must be positive")
    candidates = [initial]
    if initial > 2:
        candidates.append(initial - 1)
    if initial > 1:
        candidates.append(max(1, initial // 2))
    candidates.append(1)
    result: list[int] = []
    for value in candidates:
        if value not in result:
            result.append(value)
    return result


def _archive_selection(path: Path) -> None:
    if not path.is_file():
        return
    history = path.parent / "_runtime_resources" / "selection_history"
    history.mkdir(parents=True, exist_ok=True)
    stamp = time.time_ns()
    try:
        previous = load_json(path)
    except Exception:  # noqa: BLE001
        previous = {"unreadable_previous_selection": True}
    atomic_write_json(history / f"RUNTIME_SELECTION.{stamp}.json", previous)


def _cached_selection_is_safe(
    cached: Mapping[str, Any],
    *,
    active_ids: list[str],
    machine: MachineSnapshot,
    host_worker_limit_total: int,
    gpu_memory_headroom_fraction: float,
    required_free_floor_bytes: int,
) -> bool:
    selection = cached.get("selection")
    if not isinstance(selection, Mapping):
        return False
    if selection.get("selected_device_ids") != active_ids:
        return False
    slots = selection.get("slots_per_gpu")
    if not isinstance(slots, int) or slots < 1:
        return False
    if slots * len(active_ids) > host_worker_limit_total:
        return False
    capacity = selection.get("capacity")
    reserved = capacity.get("reserved_vram_bytes_per_worker") if isinstance(capacity, Mapping) else None
    inventory = {gpu.index: gpu for gpu in machine.gpus}
    for device_id in active_ids:
        gpu = inventory.get(device_id)
        if gpu is None or gpu.memory_free_bytes < required_free_floor_bytes:
            return False
        if isinstance(reserved, int) and reserved > 0:
            usable = math.floor(
                gpu.memory_free_bytes * (1.0 - gpu_memory_headroom_fraction)
            )
            if slots * reserved > usable:
                return False
    return True


def autotune_single_gpu_task_placement(
    *,
    machine: MachineSnapshot,
    repo_root: str | Path,
    work_dir: str | Path,
    selected_device_ids: Sequence[str],
    total_tasks: int,
    workload_fingerprint: Mapping[str, Any],
    command_factory: Callable[[int, Path], Sequence[str]],
    base_environment: Mapping[str, str] | None,
    required_host_memory_bytes_per_worker: int,
    host_memory_headroom_fraction: float,
    gpu_memory_headroom_fraction: float,
    per_worker_vram_safety_factor: float,
    max_slots_per_gpu: int,
    single_probe_seconds: float,
    validation_probe_seconds: float,
    probe_budget_seconds: float,
    required_free_floor_bytes: int,
    nvidia_smi: str = "nvidia-smi",
    probe_runner: Callable[..., GPUConcurrencyProbeResult] = probe_same_gpu_concurrency,
) -> dict[str, Any]:
    """Select and persist a uniform per-GPU slot count within a hard deadline."""

    normalized_ids = [str(value) for value in selected_device_ids]
    if not normalized_ids or len(normalized_ids) != len(set(normalized_ids)):
        raise RuntimeResourceError("selected_device_ids must be non-empty and unique")
    if total_tasks < 1:
        raise RuntimeResourceError("total_tasks must be positive")
    if required_host_memory_bytes_per_worker <= 0:
        raise RuntimeResourceError("required host memory per worker must be positive")
    if not 0.0 <= host_memory_headroom_fraction < 0.9:
        raise RuntimeResourceError("host memory headroom must be in [0, 0.9)")
    if single_probe_seconds <= 0 or validation_probe_seconds <= 0:
        raise RuntimeResourceError("GPU probe durations must be positive")
    if probe_budget_seconds <= 0:
        raise RuntimeResourceError("GPU probe budget must be positive")

    inventory = {gpu.index: gpu for gpu in machine.gpus}
    missing = [device_id for device_id in normalized_ids if device_id not in inventory]
    if missing:
        raise RuntimeResourceError(f"selected GPUs are not visible: {missing}")
    usable_host = math.floor(
        machine.effective_memory_available_bytes * (1.0 - host_memory_headroom_fraction)
    )
    host_worker_limit_total = usable_host // required_host_memory_bytes_per_worker
    device_limit = min(len(normalized_ids), host_worker_limit_total, total_tasks)
    if device_limit < 1:
        raise RuntimeResourceError("host memory cannot support one GPU worker")
    active_ids = normalized_ids[:device_limit]
    probe_device = min(active_ids, key=lambda item: inventory[item].memory_free_bytes)
    free_vram = min(inventory[item].memory_free_bytes for item in active_ids)

    policy = {
        "required_host_memory_bytes_per_worker": int(
            required_host_memory_bytes_per_worker
        ),
        "host_memory_headroom_fraction": float(host_memory_headroom_fraction),
        "gpu_memory_headroom_fraction": float(gpu_memory_headroom_fraction),
        "per_worker_vram_safety_factor": float(per_worker_vram_safety_factor),
        "max_slots_per_gpu": int(max_slots_per_gpu),
        "single_probe_seconds": float(single_probe_seconds),
        "validation_probe_seconds": float(validation_probe_seconds),
        "probe_budget_seconds": float(probe_budget_seconds),
        "required_free_floor_bytes": int(required_free_floor_bytes),
    }
    work = Path(work_dir).resolve()
    selection_path = work / "RUNTIME_SELECTION.json"
    fingerprint_sha = canonical_json_sha256(dict(workload_fingerprint))
    policy_sha = canonical_json_sha256(policy)
    machine_sha = canonical_json_sha256(machine.static_identity())
    if selection_path.is_file():
        try:
            cached = load_json(selection_path)
        except Exception:  # noqa: BLE001
            cached = {}
        if (
            cached.get("adapter_id") == ADAPTER_ID
            and cached.get("workload_fingerprint_sha256") == fingerprint_sha
            and cached.get("selector_policy_sha256") == policy_sha
            and cached.get("machine_static_sha256") == machine_sha
            and _cached_selection_is_safe(
                cached,
                active_ids=active_ids,
                machine=machine,
                host_worker_limit_total=host_worker_limit_total,
                gpu_memory_headroom_fraction=gpu_memory_headroom_fraction,
                required_free_floor_bytes=required_free_floor_bytes,
            )
        ):
            cached["mode"] = "cached"
            cached["cache_validated_utc"] = utc_now()
            cached["cache_validated_machine_snapshot"] = machine.as_dict()
            atomic_write_json(selection_path, cached)
            return cached

    _archive_selection(selection_path)
    probe_root = work / "_runtime_resource_probe" / "e8_gpu_placement"
    probe_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    deadline = started + probe_budget_seconds
    probe_records: list[GPUConcurrencyProbeResult] = []

    probe_kwargs = {
        "device_id": probe_device,
        "command_factory": command_factory,
        "environment": base_environment,
        "global_deadline_monotonic": deadline,
        "nvidia_smi": nvidia_smi,
        "required_free_floor_bytes": required_free_floor_bytes,
        "working_directory": repo_root,
    }
    single = probe_runner(
        **probe_kwargs,
        concurrency=1,
        log_dir=probe_root / "single",
        sample_seconds=single_probe_seconds,
    )
    probe_records.append(single)
    capacity: dict[str, int] | None = None
    selected_slots = 1
    selection_reason = "single_worker_probe_failed_fallback_one"
    if single.success and single.peak_incremental_vram_bytes > 0:
        capacity = derive_slots_per_gpu(
            free_vram_bytes=min(free_vram, single.initial_free_vram_bytes),
            single_worker_peak_vram_bytes=single.peak_incremental_vram_bytes,
            device_count=len(active_ids),
            total_tasks=total_tasks,
            host_worker_limit_total=host_worker_limit_total,
            gpu_memory_headroom_fraction=gpu_memory_headroom_fraction,
            per_worker_vram_safety_factor=per_worker_vram_safety_factor,
            max_slots_per_gpu=max_slots_per_gpu,
        )
        for candidate in bounded_backoff_candidates(capacity["candidate"]):
            if candidate == 1:
                selected_slots = 1
                selection_reason = "single_worker_probe_validated_only"
                break
            if time.monotonic() >= deadline:
                selection_reason = "probe_budget_exhausted_fallback_one"
                break
            validation = probe_runner(
                **probe_kwargs,
                concurrency=candidate,
                log_dir=probe_root / f"validate_{candidate}",
                sample_seconds=validation_probe_seconds,
            )
            probe_records.append(validation)
            if validation.success:
                selected_slots = candidate
                selection_reason = "highest_bounded_candidate_validated"
                break

    slot_device_ids = [
        device_id for device_id in active_ids for _ in range(selected_slots)
    ][:total_tasks]
    document = {
        "schema_version": 1,
        "adapter_id": ADAPTER_ID,
        "created_utc": utc_now(),
        "mode": "auto",
        "source": git_state(repo_root),
        "machine_snapshot": machine.as_dict(),
        "machine_static_sha256": machine_sha,
        "workload_fingerprint": dict(workload_fingerprint),
        "workload_fingerprint_sha256": fingerprint_sha,
        "selector_policy": policy,
        "selector_policy_sha256": policy_sha,
        "selection": {
            "selected_device_ids": active_ids,
            "probe_device_id": probe_device,
            "slots_per_gpu": selected_slots,
            "total_runtime_slots": len(slot_device_ids),
            "slot_device_ids": slot_device_ids,
            "host_worker_limit_total": host_worker_limit_total,
            "device_limit": device_limit,
            "capacity": capacity,
            "reason": selection_reason,
        },
        "probe": {
            "elapsed_seconds": time.monotonic() - started,
            "budget_seconds": probe_budget_seconds,
            "records": [record.as_dict() for record in probe_records],
        },
        "scientific_matrix_changed": False,
        "limitations": [
            "single_gpu_independent_tasks_only",
            "does_not_select_tp_ddp_fsdp",
            "capacity_guard_not_global_throughput_optimum",
            "no_dynamic_scaling_after_launch",
        ],
    }
    atomic_write_json(selection_path, document)
    return document
