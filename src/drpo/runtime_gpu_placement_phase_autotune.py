"""Phase-complete GPU placement probing layered on the V1 capacity selector."""
from __future__ import annotations

import dataclasses
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from drpo import runtime_gpu_placement_autotune as legacy
from drpo.runtime_resource_autotune import (
    MachineSnapshot,
    RuntimeResourceError,
    atomic_write_json,
    git_state,
    process_tree_rss,
    utc_now,
)

ADAPTER_ID = legacy.ADAPTER_ID
PROBE_CONTRACT_VERSION = 2
DEFAULT_REQUIRED_PHASES = (
    "model_loaded",
    "training_peak_completed",
    "evaluation_peak_completed",
    "probe_complete",
)
DEFAULT_EVENT_FILENAME = "resource_probe_events.jsonl"


@dataclasses.dataclass(frozen=True)
class GPUPhaseProbeContract:
    version: int = PROBE_CONTRACT_VERSION
    required_phases: tuple[str, ...] = DEFAULT_REQUIRED_PHASES
    event_filename: str = DEFAULT_EVENT_FILENAME

    def __post_init__(self) -> None:
        if self.version < 1:
            raise ValueError("probe contract version must be positive")
        if not self.required_phases or len(set(self.required_phases)) != len(
            self.required_phases
        ):
            raise ValueError("required probe phases must be non-empty and unique")
        if not self.event_filename or Path(self.event_filename).name != self.event_filename:
            raise ValueError("probe event filename must be a plain filename")

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "required_phases": list(self.required_phases),
            "event_filename": self.event_filename,
        }


@dataclasses.dataclass(frozen=True)
class GPUPhaseProbeResult(legacy.GPUConcurrencyProbeResult):
    required_phases: tuple[str, ...] = ()
    completed_phases_by_worker: tuple[tuple[str, ...], ...] = ()
    phase_contract_satisfied: bool = False
    maximum_reported_worker_vram_bytes: int = 0
    event_paths: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        payload = super().as_dict()
        payload.update(
            {
                "required_phases": list(self.required_phases),
                "completed_phases_by_worker": [
                    list(phases) for phases in self.completed_phases_by_worker
                ],
                "phase_contract_satisfied": self.phase_contract_satisfied,
                "maximum_reported_worker_vram_bytes": (
                    self.maximum_reported_worker_vram_bytes
                ),
                "event_paths": list(self.event_paths),
            }
        )
        return payload


def _read_phase_events(
    worker_roots: Sequence[Path], contract: GPUPhaseProbeContract
) -> tuple[tuple[tuple[str, ...], ...], tuple[int, ...]]:
    completed: list[tuple[str, ...]] = []
    peaks: list[int] = []
    for worker_root in worker_roots:
        path = worker_root / contract.event_filename
        phases: list[str] = []
        peak = 0
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            lines = []
        for line in lines:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            phase = event.get("phase")
            if isinstance(phase, str) and phase not in phases:
                phases.append(phase)
            for key in (
                "peak_vram_reserved_bytes",
                "peak_vram_allocated_bytes",
                "peak_vram_bytes",
            ):
                value = event.get(key)
                if isinstance(value, int) and value >= 0:
                    peak = max(peak, value)
        completed.append(tuple(phases))
        peaks.append(peak)
    return tuple(completed), tuple(peaks)


def _contract_satisfied(
    completed: Sequence[Sequence[str]], contract: GPUPhaseProbeContract
) -> bool:
    required = set(contract.required_phases)
    return bool(completed) and all(required.issubset(set(phases)) for phases in completed)


def probe_phase_complete_gpu_concurrency(
    *,
    device_id: str,
    concurrency: int,
    command_factory: Callable[[int, Path], Sequence[str]],
    environment: Mapping[str, str] | None,
    log_dir: str | Path,
    sample_seconds: float,
    global_deadline_monotonic: float,
    probe_contract: GPUPhaseProbeContract,
    nvidia_smi: str = "nvidia-smi",
    poll_interval_seconds: float = 0.5,
    required_free_floor_bytes: int = 0,
    terminate_grace_seconds: float = 15.0,
    working_directory: str | Path | None = None,
) -> GPUPhaseProbeResult:
    """Accept a concurrency only after every worker completes every phase."""

    if concurrency < 1 or sample_seconds <= 0 or poll_interval_seconds <= 0:
        raise RuntimeResourceError("GPU probe concurrency and durations must be positive")
    root = Path(log_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    cwd = Path(working_directory).resolve() if working_directory else Path.cwd()
    started_utc = utc_now()
    started = time.monotonic()
    initial_free = legacy._gpu_free_bytes(device_id, nvidia_smi=nvidia_smi)
    minimum_free = initial_free
    peak_host_rss = 0
    processes: list[subprocess.Popen[str]] = []
    handles: list[Any] = []
    logs: list[Path] = []
    worker_roots: list[Path] = []
    launch_error: str | None = None
    nonzero_seen = False
    global_deadline_reached = False
    returncodes: tuple[int | None, ...] = ()
    completed: tuple[tuple[str, ...], ...] = ()
    worker_peaks: tuple[int, ...] = ()
    phase_ok = False

    try:
        for worker_index in range(concurrency):
            if time.monotonic() >= global_deadline_monotonic:
                global_deadline_reached = True
                break
            worker_root = root / f"worker_{worker_index:02d}"
            worker_root.mkdir(parents=True, exist_ok=True)
            worker_roots.append(worker_root)
            log_path = root / f"worker_{worker_index:02d}.log"
            logs.append(log_path)
            handle = log_path.open("w", encoding="utf-8")
            handles.append(handle)
            try:
                command = [
                    str(value) for value in command_factory(worker_index, worker_root)
                ]
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
                worker_environment["DRPO_RUNTIME_RESOURCE_PROBE_EVENT_PATH"] = str(
                    worker_root / probe_contract.event_filename
                )
                handle.write(f"GPU={device_id}\nCOMMAND={' '.join(command)}\n")
                handle.flush()
                processes.append(
                    subprocess.Popen(
                        command,
                        cwd=cwd,
                        env=worker_environment,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        text=True,
                        start_new_session=True,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                handle.write(f"LAUNCH_ERROR={exc}\n")
                handle.flush()
                launch_error = str(exc)
                break

        sample_deadline = min(started + sample_seconds, global_deadline_monotonic)
        while launch_error is None and len(processes) == concurrency:
            try:
                minimum_free = min(
                    minimum_free,
                    legacy._gpu_free_bytes(device_id, nvidia_smi=nvidia_smi),
                )
            except RuntimeResourceError:
                launch_error = "GPU became invisible during placement probe"
                break
            peak_host_rss = max(
                peak_host_rss,
                sum(process_tree_rss(process.pid) for process in processes),
            )
            completed, worker_peaks = _read_phase_events(worker_roots, probe_contract)
            phase_ok = _contract_satisfied(completed, probe_contract)
            observed = [process.poll() for process in processes]
            if any(code is not None and code != 0 for code in observed):
                nonzero_seen = True
                break
            if phase_ok or all(code == 0 for code in observed):
                break
            now = time.monotonic()
            if now >= sample_deadline:
                global_deadline_reached = now >= global_deadline_monotonic
                break
            time.sleep(poll_interval_seconds)
        returncodes = tuple(process.poll() for process in processes)
        completed, worker_peaks = _read_phase_events(worker_roots, probe_contract)
        phase_ok = _contract_satisfied(completed, probe_contract)
    finally:
        for index, worker_root in enumerate(worker_roots):
            source = worker_root / probe_contract.event_filename
            destination = root / f"worker_{index:02d}.events.jsonl"
            if source.is_file():
                try:
                    shutil.copyfile(source, destination)
                except OSError:
                    pass
        legacy._stop_processes(processes, grace_seconds=terminate_grace_seconds)
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
            legacy._gpu_free_bytes(device_id, nvidia_smi=nvidia_smi),
        )
    except RuntimeResourceError:
        pass
    oom = legacy._logs_contain_oom(logs)
    floor_ok = minimum_free >= max(0, int(required_free_floor_bytes))
    complete_launch = len(processes) == concurrency
    clean_exit_or_live = complete_launch and not nonzero_seen and all(
        code in (None, 0) for code in returncodes
    )
    success = (
        launch_error is None
        and clean_exit_or_live
        and phase_ok
        and peak_host_rss > 0
        and not oom
        and floor_ok
        and not global_deadline_reached
    )
    if launch_error:
        reason = f"launch_failure:{launch_error}"
    elif not complete_launch:
        reason = "global_probe_deadline_reached_before_full_launch"
    elif oom:
        reason = "oom_signature_detected"
    elif nonzero_seen:
        reason = "worker_nonzero_exit"
    elif not floor_ok:
        reason = "free_vram_floor_crossed"
    elif not phase_ok and global_deadline_reached:
        reason = "required_probe_phases_incomplete_at_global_deadline"
    elif not phase_ok:
        reason = "required_probe_phases_incomplete"
    elif peak_host_rss <= 0:
        reason = "host_rss_not_observed"
    elif success:
        reason = "phase_complete_concurrency_probe_passed"
    else:
        reason = "inconclusive_probe"

    events = tuple(
        str(root / f"worker_{index:02d}.events.jsonl")
        for index in range(len(worker_roots))
        if (root / f"worker_{index:02d}.events.jsonl").is_file()
    )
    return GPUPhaseProbeResult(
        concurrency=concurrency,
        device_id=str(device_id),
        started_utc=started_utc,
        finished_utc=utc_now(),
        elapsed_seconds=time.monotonic() - started,
        success=success,
        sample_window_completed=phase_ok,
        global_deadline_reached=global_deadline_reached,
        oom_detected=oom,
        worker_returncodes=returncodes,
        initial_free_vram_bytes=initial_free,
        minimum_free_vram_bytes=minimum_free,
        peak_incremental_vram_bytes=max(
            max(0, initial_free - minimum_free), max(worker_peaks, default=0)
        ),
        peak_host_rss_bytes=peak_host_rss,
        log_paths=tuple(str(path) for path in logs),
        reason=reason,
        required_phases=probe_contract.required_phases,
        completed_phases_by_worker=completed,
        phase_contract_satisfied=phase_ok,
        maximum_reported_worker_vram_bytes=max(worker_peaks, default=0),
        event_paths=events,
    )


def autotune_phase_complete_single_gpu_task_placement(
    *,
    probe_contract: GPUPhaseProbeContract,
    probe_runner: Callable[..., GPUPhaseProbeResult] = (
        probe_phase_complete_gpu_concurrency
    ),
    **kwargs: Any,
) -> dict[str, Any]:
    """Use the existing capacity model, then enforce phase-complete evidence."""

    work = Path(kwargs["work_dir"]).resolve()
    selection_path = work / "RUNTIME_SELECTION.json"
    failure_path = work / "RUNTIME_PROBE_FAILURE.json"
    fingerprint = dict(kwargs["workload_fingerprint"])
    fingerprint["probe_contract"] = probe_contract.as_dict()
    kwargs["workload_fingerprint"] = fingerprint

    def strict_runner(**runner_kwargs: Any) -> GPUPhaseProbeResult:
        return probe_runner(
            **runner_kwargs,
            probe_contract=probe_contract,
        )

    document = legacy.autotune_single_gpu_task_placement(
        **kwargs,
        probe_runner=strict_runner,
    )
    records = document.get("probe", {}).get("records", [])
    single = records[0] if records else {}
    if (
        single.get("success") is not True
        or single.get("phase_contract_satisfied") is not True
    ):
        selection_path.unlink(missing_ok=True)
        failure = {
            "schema_version": 2,
            "adapter_id": ADAPTER_ID,
            "created_utc": utc_now(),
            "source": git_state(kwargs["repo_root"]),
            "probe_contract": probe_contract.as_dict(),
            "failure": "phase_complete_single_worker_probe_failed",
            "record": single,
            "scientific_matrix_changed": False,
        }
        atomic_write_json(failure_path, failure)
        raise RuntimeResourceError(
            "phase-complete single-worker GPU probe failed: "
            + str(single.get("reason", "missing_phase_evidence"))
        )

    selected_slots = int(document["selection"]["slots_per_gpu"])
    selected_record = next(
        (
            record
            for record in records
            if int(record.get("concurrency", -1)) == selected_slots
            and record.get("success") is True
            and record.get("phase_contract_satisfied") is True
        ),
        None,
    )
    if selected_record is None:
        selection_path.unlink(missing_ok=True)
        raise RuntimeResourceError(
            "selected GPU concurrency has no phase-complete probe record"
        )

    capacity = document["selection"].get("capacity")
    if isinstance(capacity, dict):
        reported_peak = int(single.get("maximum_reported_worker_vram_bytes", 0))
        parent_peak = int(single.get("peak_incremental_vram_bytes", 0))
        capacity["single_worker_reported_peak_vram_bytes"] = reported_peak
        capacity["single_worker_parent_peak_incremental_vram_bytes"] = parent_peak
        capacity["effective_single_worker_peak_vram_bytes"] = max(
            reported_peak, parent_peak
        )
    document["schema_version"] = 2
    document["probe_contract"] = probe_contract.as_dict()
    document["selection"]["phase_contract_satisfied"] = True
    document["selection"]["accepted_probe_concurrency"] = selected_slots
    document["limitations"] = list(document.get("limitations", [])) + [
        "phase_complete_resource_envelope_required"
    ]
    failure_path.unlink(missing_ok=True)
    atomic_write_json(selection_path, document)
    return document
