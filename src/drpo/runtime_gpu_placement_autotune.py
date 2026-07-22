"""Bounded phase-aware GPU placement autotuning for independent single-GPU tasks."""
from __future__ import annotations

import dataclasses
import json
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
    process_tree_rss,
    utc_now,
)

ADAPTER_ID = "single_gpu_independent_task_placement_v1"
PROBE_CONTRACT_VERSION = 2
PROBE_STATE_FILENAME = "PROBE_PHASES.json"
DEFAULT_REQUIRED_PHASES = (
    "model_loaded",
    "training_peak_completed",
    "evaluation_peak_completed",
    "probe_complete",
)
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
    peak_host_rss_bytes: int
    required_phases: tuple[str, ...]
    completed_phases_by_worker: tuple[tuple[str, ...], ...]
    phase_contract_satisfied: bool
    log_paths: tuple[str, ...]
    phase_evidence_paths: tuple[str, ...]
    reason: str

    def as_dict(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload["worker_returncodes"] = list(self.worker_returncodes)
        payload["required_phases"] = list(self.required_phases)
        payload["completed_phases_by_worker"] = [
            list(phases) for phases in self.completed_phases_by_worker
        ]
        payload["log_paths"] = list(self.log_paths)
        payload["phase_evidence_paths"] = list(self.phase_evidence_paths)
        return payload


def _gpu_free_bytes(device_id: str, *, nvidia_smi: str) -> int:
    inventory = {gpu.index: gpu for gpu in discover_gpus(nvidia_smi=nvidia_smi)}
    gpu = inventory.get(str(device_id))
    if gpu is None:
        raise RuntimeResourceError(f"GPU {device_id} disappeared during placement probe")
    return int(gpu.memory_free_bytes)


def _stop_processes(
    processes: Sequence[subprocess.Popen[str]], *, grace_seconds: float
) -> None:
    live = [process for process in processes if process.poll() is None]
    for process in live:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + max(0.0, grace_seconds)
    while any(process.poll() is None for process in live) and time.monotonic() < deadline:
        time.sleep(0.1)
    for process in live:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    for process in live:
        try:
            process.wait(timeout=1.0)
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


def _phase_snapshot(
    worker_roots: Sequence[Path], *, required_phases: Sequence[str]
) -> tuple[tuple[tuple[str, ...], ...], bool]:
    required = tuple(str(value) for value in required_phases)
    completed: list[tuple[str, ...]] = []
    all_satisfied = bool(worker_roots)
    for worker_root in worker_roots:
        path = worker_root / PROBE_STATE_FILENAME
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            completed.append(())
            all_satisfied = False
            continue
        phases = tuple(str(value) for value in payload.get("completed_phases", []))
        completed.append(phases)
        contract_ok = payload.get("contract_version") == PROBE_CONTRACT_VERSION
        declared = tuple(str(value) for value in payload.get("required_phases", []))
        phases_ok = declared == required and all(phase in phases for phase in required)
        complete_ok = payload.get("complete") is True
        all_satisfied = all_satisfied and contract_ok and phases_ok and complete_ok
    return tuple(completed), all_satisfied


def _archive_phase_evidence(root: Path, worker_roots: Sequence[Path]) -> tuple[str, ...]:
    paths: list[str] = []
    for index, worker_root in enumerate(worker_roots):
        source = worker_root / PROBE_STATE_FILENAME
        if not source.is_file():
            continue
        target = root / f"worker_{index:02d}.phases.json"
        shutil.copy2(source, target)
        paths.append(str(target))
    return tuple(paths)


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
    required_phases: Sequence[str] = DEFAULT_REQUIRED_PHASES,
) -> GPUConcurrencyProbeResult:
    """Run a bounded same-GPU probe that must complete the declared phase contract."""

    required = tuple(str(value) for value in required_phases)
    if concurrency < 1 or sample_seconds <= 0 or poll_interval_seconds <= 0:
        raise RuntimeResourceError("GPU probe concurrency and durations must be positive")
    if not required or len(required) != len(set(required)):
        raise RuntimeResourceError("required probe phases must be non-empty and unique")

    root = Path(log_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    cwd = Path(working_directory).resolve() if working_directory else Path.cwd()
    started_utc = utc_now()
    started = time.monotonic()
    initial_free = _gpu_free_bytes(device_id, nvidia_smi=nvidia_smi)
    minimum_free = initial_free
    peak_host_rss = 0
    processes: list[subprocess.Popen[str]] = []
    handles: list[Any] = []
    log_paths: list[Path] = []
    worker_roots: list[Path] = []
    launch_error: str | None = None
    nonzero_seen = False
    sample_completed = False
    global_deadline_reached = False
    observed_returncodes: tuple[int | None, ...] = ()
    elapsed_before_cleanup = 0.0
    completed_phases: tuple[tuple[str, ...], ...] = ()
    phase_contract_satisfied = False
    phase_evidence_paths: tuple[str, ...] = ()

    try:
        for worker_index in range(concurrency):
            if time.monotonic() >= global_deadline_monotonic:
                global_deadline_reached = True
                break
            worker_root = root / f"worker_{worker_index:02d}"
            worker_root.mkdir(parents=True, exist_ok=True)
            worker_roots.append(worker_root)
            log_path = root / f"worker_{worker_index:02d}.log"
            log_paths.append(log_path)
            handle = log_path.open("w", encoding="utf-8")
            handles.append(handle)
            try:
                command = [str(item) for item in command_factory(worker_index, worker_root)]
                worker_environment = dict(os.environ)
                if environment:
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
        while launch_error is None and len(processes) == concurrency:
            minimum_free = min(
                minimum_free,
                _gpu_free_bytes(device_id, nvidia_smi=nvidia_smi),
            )
            peak_host_rss = max(
                peak_host_rss,
                sum(process_tree_rss(process.pid) for process in processes),
            )
            completed_phases, phase_contract_satisfied = _phase_snapshot(
                worker_roots,
                required_phases=required,
            )
            if phase_contract_satisfied:
                sample_completed = True
                break

            returncodes = [process.poll() for process in processes]
            if any(code is not None and code != 0 for code in returncodes):
                nonzero_seen = True
                break
            if all(code == 0 for code in returncodes):
                break

            now = time.monotonic()
            if now >= sample_deadline:
                global_deadline_reached = now >= global_deadline_monotonic
                break
            time.sleep(poll_interval_seconds)

        elapsed_before_cleanup = time.monotonic() - started
        observed_returncodes = tuple(process.poll() for process in processes)
        completed_phases, phase_contract_satisfied = _phase_snapshot(
            worker_roots,
            required_phases=required,
        )
        phase_evidence_paths = _archive_phase_evidence(root, worker_roots)
    finally:
        _stop_processes(processes, grace_seconds=terminate_grace_seconds)
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
    lived_long_enough = elapsed_before_cleanup >= minimum_live_seconds
    floor_ok = minimum_free >= max(0, int(required_free_floor_bytes))
    complete_launch = len(processes) == concurrency
    clean_exit_or_live = complete_launch and not nonzero_seen and all(
        code in (None, 0) for code in observed_returncodes
    )
    success = (
        launch_error is None
        and clean_exit_or_live
        and lived_long_enough
        and peak_host_rss > 0
        and not oom_detected
        and floor_ok
        and not global_deadline_reached
        and phase_contract_satisfied
    )
    if launch_error:
        reason = f"launch_failure:{launch_error}"
    elif not complete_launch:
        reason = "global_probe_deadline_reached_before_full_launch"
    elif oom_detected:
        reason = "oom_signature_detected"
    elif nonzero_seen:
        reason = "worker_nonzero_exit"
    elif not floor_ok:
        reason = "free_vram_floor_crossed"
    elif global_deadline_reached:
        reason = "global_probe_deadline_reached"
    elif not phase_contract_satisfied:
        reason = "required_probe_phases_incomplete"
    elif not lived_long_enough:
        reason = "insufficient_live_interval"
    elif peak_host_rss <= 0:
        reason = "host_rss_not_observed"
    elif success:
        reason = "phase_complete_concurrency_probe_passed"
    else:
        reason = "inconclusive_probe"

    return GPUConcurrencyProbeResult(
        concurrency=concurrency,
        device_id=str(device_id),
        started_utc=started_utc,
        finished_utc=utc_now(),
        elapsed_seconds=time.monotonic() - started,
        success=success,
        sample_window_completed=sample_completed,
        global_deadline_reached=global_deadline_reached,
        oom_detected=oom_detected,
        worker_returncodes=observed_returncodes,
        initial_free_vram_bytes=initial_free,
        minimum_free_vram_bytes=minimum_free,
        peak_incremental_vram_bytes=max(0, initial_free - minimum_free),
        peak_host_rss_bytes=peak_host_rss,
        required_phases=required,
        completed_phases_by_worker=completed_phases,
        phase_contract_satisfied=phase_contract_satisfied,
        log_paths=tuple(str(path) for path in log_paths),
        phase_evidence_paths=phase_evidence_paths,
        reason=reason,
    )


def derive_slots_per_gpu(
    *,
    free_vram_bytes: int,
    single_worker_peak_vram_bytes: int,
    device_count: int,
    total_tasks: int,
    host_worker_limit_total: int,
    cpu_worker_limit_total: int,
    gpu_memory_headroom_fraction: float,
    per_worker_vram_safety_factor: float,
    max_slots_per_gpu: int,
) -> dict[str, int]:
    if min(
        free_vram_bytes,
        single_worker_peak_vram_bytes,
        device_count,
        total_tasks,
        host_worker_limit_total,
        cpu_worker_limit_total,
        max_slots_per_gpu,
    ) < 1:
        raise RuntimeResourceError("GPU placement capacity inputs must be positive")
    if not 0.0 <= gpu_memory_headroom_fraction < 0.9:
        raise RuntimeResourceError("GPU memory headroom must be in [0, 0.9)")
    if per_worker_vram_safety_factor < 1.0:
        raise RuntimeResourceError("per-worker VRAM safety factor must be >= 1")
    usable_vram = math.floor(free_vram_bytes * (1.0 - gpu_memory_headroom_fraction))
    reserved_vram = math.ceil(
        single_worker_peak_vram_bytes * per_worker_vram_safety_factor
    )
    limits = {
        "vram_limit_per_device": usable_vram // reserved_vram,
        "host_limit_per_device": host_worker_limit_total // device_count,
        "cpu_limit_per_device": cpu_worker_limit_total // device_count,
        "task_limit_per_device": math.ceil(total_tasks / device_count),
        "configured_limit_per_device": max_slots_per_gpu,
    }
    candidate = max(1, min(limits.values()))
    return {
        "candidate": candidate,
        "usable_vram_bytes": usable_vram,
        "reserved_vram_bytes_per_worker": reserved_vram,
        **{key: max(0, value) for key, value in limits.items()},
    }


def bounded_backoff_candidates(initial: int) -> list[int]:
    if initial < 1:
        raise RuntimeResourceError("initial GPU slot candidate must be positive")
    values = [initial, initial - 1, math.ceil(initial / 2), 1]
    return list(dict.fromkeys(value for value in values if value >= 1))


def _archive_selection(path: Path) -> None:
    if not path.is_file():
        return
    history = path.parent / "_runtime_resources" / "selection_history"
    history.mkdir(parents=True, exist_ok=True)
    try:
        previous = load_json(path)
    except Exception:  # noqa: BLE001
        previous = {"unreadable_previous_selection": True}
    atomic_write_json(
        history / f"RUNTIME_SELECTION.{time.time_ns()}.json",
        previous,
    )


def _record_for_selected_slots(cached: Mapping[str, Any], slots: int) -> Mapping[str, Any] | None:
    probe = cached.get("probe")
    if not isinstance(probe, Mapping):
        return None
    records = probe.get("records")
    if not isinstance(records, list):
        return None
    for record in records:
        if isinstance(record, Mapping) and record.get("concurrency") == slots:
            return record
    return None


def _cache_is_safe(
    cached: Mapping[str, Any],
    *,
    active_ids: list[str],
    machine: MachineSnapshot,
    usable_host_bytes: int,
    cpu_worker_limit_total: int,
    gpu_memory_headroom_fraction: float,
    required_free_floor_bytes: int,
    required_phases: Sequence[str],
) -> bool:
    if cached.get("probe_contract_version") != PROBE_CONTRACT_VERSION:
        return False
    probe = cached.get("probe")
    if not isinstance(probe, Mapping):
        return False
    if tuple(probe.get("required_phases", [])) != tuple(required_phases):
        return False

    selection = cached.get("selection")
    if not isinstance(selection, Mapping):
        return False
    slots = selection.get("slots_per_gpu")
    capacity = selection.get("capacity")
    if (
        selection.get("selected_device_ids") != active_ids
        or not isinstance(slots, int)
        or slots < 1
        or not isinstance(capacity, Mapping)
    ):
        return False
    selected_record = _record_for_selected_slots(cached, slots)
    if not isinstance(selected_record, Mapping) or selected_record.get(
        "phase_contract_satisfied"
    ) is not True:
        return False

    total_workers = slots * len(active_ids)
    reserved_host = capacity.get("reserved_host_bytes_per_worker")
    reserved_vram = capacity.get("reserved_vram_bytes_per_worker")
    if (
        not isinstance(reserved_host, int)
        or reserved_host < 1
        or total_workers * reserved_host > usable_host_bytes
        or total_workers > cpu_worker_limit_total
    ):
        return False
    inventory = {gpu.index: gpu for gpu in machine.gpus}
    for device_id in active_ids:
        gpu = inventory.get(device_id)
        if gpu is None or gpu.memory_free_bytes < required_free_floor_bytes:
            return False
        if not isinstance(reserved_vram, int) or reserved_vram < 1:
            return False
        usable_vram = math.floor(
            gpu.memory_free_bytes * (1.0 - gpu_memory_headroom_fraction)
        )
        if slots * reserved_vram > usable_vram:
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
    per_worker_host_memory_safety_factor: float,
    cpu_fraction: float,
    gpu_memory_headroom_fraction: float,
    per_worker_vram_safety_factor: float,
    max_slots_per_gpu: int,
    single_probe_seconds: float,
    validation_probe_seconds: float,
    probe_budget_seconds: float,
    required_free_floor_bytes: int,
    nvidia_smi: str = "nvidia-smi",
    required_probe_phases: Sequence[str] = DEFAULT_REQUIRED_PHASES,
    probe_runner: Callable[..., GPUConcurrencyProbeResult] = probe_same_gpu_concurrency,
) -> dict[str, Any]:
    """Select and persist a uniform per-GPU slot count within a hard deadline."""

    required_phases = tuple(str(value) for value in required_probe_phases)
    device_ids = [str(value) for value in selected_device_ids]
    if not device_ids or len(device_ids) != len(set(device_ids)) or total_tasks < 1:
        raise RuntimeResourceError("GPU ids must be non-empty and unique; tasks positive")
    if not required_phases or len(required_phases) != len(set(required_phases)):
        raise RuntimeResourceError("required probe phases must be non-empty and unique")
    if required_host_memory_bytes_per_worker < 1:
        raise RuntimeResourceError("host-memory floor per worker must be positive")
    if per_worker_host_memory_safety_factor < 1.0:
        raise RuntimeResourceError("host-memory safety factor must be >= 1")
    if not 0.05 <= cpu_fraction <= 1.0:
        raise RuntimeResourceError("cpu_fraction must be in [0.05, 1.0]")
    if min(single_probe_seconds, validation_probe_seconds, probe_budget_seconds) <= 0:
        raise RuntimeResourceError("GPU probe durations must be positive")

    inventory = {gpu.index: gpu for gpu in machine.gpus}
    if any(device_id not in inventory for device_id in device_ids):
        raise RuntimeResourceError("a selected GPU is no longer visible")
    profiles = {
        (inventory[device_id].name, inventory[device_id].memory_total_bytes)
        for device_id in device_ids
    }
    if len(profiles) != 1:
        raise RuntimeResourceError("V1 requires a homogeneous selected GPU pool")

    usable_host = math.floor(
        machine.effective_memory_available_bytes * (1.0 - host_memory_headroom_fraction)
    )
    initial_host_limit = usable_host // required_host_memory_bytes_per_worker
    cpu_limit = max(
        1,
        math.floor(machine.logical_cpu_count * cpu_fraction - machine.load_average_1m),
    )
    initial_device_limit = min(len(device_ids), total_tasks, initial_host_limit, cpu_limit)
    if initial_device_limit < 1:
        raise RuntimeResourceError("machine cannot support one GPU worker")
    initial_ids = device_ids[:initial_device_limit]
    probe_device = min(
        initial_ids,
        key=lambda device_id: inventory[device_id].memory_free_bytes,
    )

    policy = {
        "probe_contract_version": PROBE_CONTRACT_VERSION,
        "required_probe_phases": list(required_phases),
        "required_host_memory_bytes_per_worker": required_host_memory_bytes_per_worker,
        "host_memory_headroom_fraction": host_memory_headroom_fraction,
        "per_worker_host_memory_safety_factor": per_worker_host_memory_safety_factor,
        "cpu_fraction": cpu_fraction,
        "gpu_memory_headroom_fraction": gpu_memory_headroom_fraction,
        "per_worker_vram_safety_factor": per_worker_vram_safety_factor,
        "max_slots_per_gpu": max_slots_per_gpu,
        "single_probe_seconds": single_probe_seconds,
        "validation_probe_seconds": validation_probe_seconds,
        "probe_budget_seconds": probe_budget_seconds,
        "required_free_floor_bytes": required_free_floor_bytes,
    }
    work = Path(work_dir).resolve()
    selection_path = work / "RUNTIME_SELECTION.json"
    workload_sha = canonical_json_sha256(dict(workload_fingerprint))
    policy_sha = canonical_json_sha256(policy)
    machine_sha = canonical_json_sha256(machine.static_identity())
    if selection_path.is_file():
        try:
            cached = load_json(selection_path)
        except Exception:  # noqa: BLE001
            cached = {}
        cached_ids = cached.get("selection", {}).get("selected_device_ids")
        if (
            cached.get("adapter_id") == ADAPTER_ID
            and cached.get("workload_fingerprint_sha256") == workload_sha
            and cached.get("selector_policy_sha256") == policy_sha
            and cached.get("machine_static_sha256") == machine_sha
            and isinstance(cached_ids, list)
            and _cache_is_safe(
                cached,
                active_ids=[str(value) for value in cached_ids],
                machine=machine,
                usable_host_bytes=usable_host,
                cpu_worker_limit_total=cpu_limit,
                gpu_memory_headroom_fraction=gpu_memory_headroom_fraction,
                required_free_floor_bytes=required_free_floor_bytes,
                required_phases=required_phases,
            )
        ):
            cached["mode"] = "cached"
            cached["cache_validated_utc"] = utc_now()
            cached["cache_validated_machine_snapshot"] = machine.as_dict()
            atomic_write_json(selection_path, cached)
            return cached

    _archive_selection(selection_path)
    probe_root = work / "_runtime_resource_probe" / "e8_gpu_placement"
    started = time.monotonic()
    deadline = started + probe_budget_seconds
    common_probe = {
        "device_id": probe_device,
        "command_factory": command_factory,
        "environment": base_environment,
        "global_deadline_monotonic": deadline,
        "nvidia_smi": nvidia_smi,
        "required_free_floor_bytes": required_free_floor_bytes,
        "working_directory": repo_root,
        "required_phases": required_phases,
    }
    records: list[GPUConcurrencyProbeResult] = []
    checks: list[dict[str, Any]] = []
    single = probe_runner(
        **common_probe,
        concurrency=1,
        log_dir=probe_root / "single",
        sample_seconds=single_probe_seconds,
    )
    records.append(single)
    if (
        not single.success
        or not single.phase_contract_satisfied
        or single.peak_incremental_vram_bytes <= 0
        or single.peak_host_rss_bytes <= 0
    ):
        document = {
            "schema_version": 2,
            "adapter_id": ADAPTER_ID,
            "probe_contract_version": PROBE_CONTRACT_VERSION,
            "created_utc": utc_now(),
            "mode": "failed",
            "source": git_state(repo_root),
            "machine_snapshot": machine.as_dict(),
            "machine_static_sha256": machine_sha,
            "workload_fingerprint": dict(workload_fingerprint),
            "workload_fingerprint_sha256": workload_sha,
            "selector_policy": policy,
            "selector_policy_sha256": policy_sha,
            "selection": None,
            "probe": {
                "elapsed_seconds": time.monotonic() - started,
                "budget_seconds": probe_budget_seconds,
                "required_phases": list(required_phases),
                "records": [single.as_dict()],
                "candidate_checks": [],
            },
            "scientific_matrix_changed": False,
            "failure_reason": "single_worker_resource_envelope_incomplete",
        }
        atomic_write_json(selection_path, document)
        raise RuntimeResourceError(
            "single-worker GPU resource envelope did not complete all required phases"
        )

    reserved_host = max(
        required_host_memory_bytes_per_worker,
        math.ceil(single.peak_host_rss_bytes * per_worker_host_memory_safety_factor),
    )
    host_limit = usable_host // reserved_host
    device_limit = min(len(device_ids), total_tasks, host_limit, cpu_limit)
    if device_limit < 1:
        raise RuntimeResourceError("measured worker cannot fit after host headroom")
    active_ids = device_ids[:device_limit]
    free_vram = min(inventory[item].memory_free_bytes for item in active_ids)
    capacity = derive_slots_per_gpu(
        free_vram_bytes=min(free_vram, single.initial_free_vram_bytes),
        single_worker_peak_vram_bytes=single.peak_incremental_vram_bytes,
        device_count=len(active_ids),
        total_tasks=total_tasks,
        host_worker_limit_total=host_limit,
        cpu_worker_limit_total=cpu_limit,
        gpu_memory_headroom_fraction=gpu_memory_headroom_fraction,
        per_worker_vram_safety_factor=per_worker_vram_safety_factor,
        max_slots_per_gpu=max_slots_per_gpu,
    )
    capacity["single_worker_peak_host_rss_bytes"] = single.peak_host_rss_bytes
    capacity["reserved_host_bytes_per_worker"] = reserved_host
    capacity["host_worker_limit_total"] = host_limit
    capacity["cpu_worker_limit_total"] = cpu_limit

    selected_slots = 1
    reason = "single_worker_resource_envelope_validated_only"
    for candidate in bounded_backoff_candidates(capacity["candidate"]):
        if candidate == 1:
            break
        if time.monotonic() >= deadline:
            reason = "probe_budget_exhausted_fallback_one"
            break
        validation = probe_runner(
            **common_probe,
            concurrency=candidate,
            log_dir=probe_root / f"validate_{candidate}",
            sample_seconds=validation_probe_seconds,
        )
        records.append(validation)
        projected_host = validation.peak_host_rss_bytes * len(active_ids)
        host_ok = projected_host <= usable_host
        cpu_ok = candidate * len(active_ids) <= cpu_limit
        phase_ok = validation.phase_contract_satisfied
        accepted = validation.success and phase_ok and host_ok and cpu_ok
        checks.append(
            {
                "candidate": candidate,
                "probe_success": validation.success,
                "phase_contract_satisfied": phase_ok,
                "completed_phases_by_worker": [
                    list(phases) for phases in validation.completed_phases_by_worker
                ],
                "projected_host_rss_bytes": projected_host,
                "host_capacity_ok": host_ok,
                "cpu_capacity_ok": cpu_ok,
                "accepted": accepted,
            }
        )
        if accepted:
            selected_slots = candidate
            reason = "highest_phase_complete_candidate_validated"
            break

    slot_ids = [
        device_id for device_id in active_ids for _ in range(selected_slots)
    ][:total_tasks]
    document = {
        "schema_version": 2,
        "adapter_id": ADAPTER_ID,
        "probe_contract_version": PROBE_CONTRACT_VERSION,
        "created_utc": utc_now(),
        "mode": "auto",
        "source": git_state(repo_root),
        "machine_snapshot": machine.as_dict(),
        "machine_static_sha256": machine_sha,
        "workload_fingerprint": dict(workload_fingerprint),
        "workload_fingerprint_sha256": workload_sha,
        "selector_policy": policy,
        "selector_policy_sha256": policy_sha,
        "selection": {
            "selected_device_ids": active_ids,
            "probe_device_id": probe_device,
            "slots_per_gpu": selected_slots,
            "total_runtime_slots": len(slot_ids),
            "slot_device_ids": slot_ids,
            "capacity": capacity,
            "reason": reason,
        },
        "probe": {
            "elapsed_seconds": time.monotonic() - started,
            "budget_seconds": probe_budget_seconds,
            "required_phases": list(required_phases),
            "records": [record.as_dict() for record in records],
            "candidate_checks": checks,
        },
        "scientific_matrix_changed": False,
        "limitations": [
            "single_gpu_independent_tasks_only",
            "homogeneous_gpu_pool_only",
            "does_not_select_tp_ddp_fsdp",
            "capacity_guard_not_global_throughput_optimum",
            "no_dynamic_scaling_after_launch",
        ],
    }
    atomic_write_json(selection_path, document)
    return document
