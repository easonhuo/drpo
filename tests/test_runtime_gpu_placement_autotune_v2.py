from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

from drpo import runtime_gpu_placement_autotune_v2 as placement
from drpo.runtime_resource_autotune import (
    GIB,
    GPUDevice,
    MachineSnapshot,
    RuntimeResourceError,
)


def snapshot(*, load_average_1m: float = 0.0) -> MachineSnapshot:
    host_total = 512 * GIB
    host_available = 400 * GIB
    return MachineSnapshot(
        logical_cpu_count=384,
        memory_total_bytes=host_total,
        memory_available_bytes=host_available,
        effective_memory_limit_bytes=host_total,
        effective_memory_current_bytes=host_total - host_available,
        effective_memory_available_bytes=host_available,
        swap_total_bytes=0,
        swap_free_bytes=0,
        cgroup_version="v2",
        load_average_1m=load_average_1m,
        gpus=tuple(
            GPUDevice(
                index=str(index),
                name="H20",
                memory_total_bytes=96 * GIB,
                memory_free_bytes=90 * GIB,
                utilization_percent=0.0,
            )
            for index in range(8)
        ),
    )


def probe_result(
    concurrency: int,
    *,
    success: bool = True,
    peak_gib: float = 12.0,
    host_gib: float | None = None,
    average_cpu_cores: float | None = None,
    baseline_busy_cores: float = 20.0,
    clean_exit: bool = True,
) -> placement.GPUConcurrencyProbeResult:
    cpu_cores = (
        float(2 * concurrency)
        if average_cpu_cores is None
        else float(average_cpu_cores)
    )
    phases = (
        tuple(placement.DEFAULT_REQUIRED_PHASES)
        if success
        else ("model_loaded",)
    )
    initial = 90 * GIB
    peak = int(peak_gib * GIB)
    host_peak = int(
        (3 * concurrency if host_gib is None else host_gib) * GIB
    )
    return placement.GPUConcurrencyProbeResult(
        concurrency=concurrency,
        device_id="0",
        started_utc="a",
        finished_utc="b",
        elapsed_seconds=20.0,
        success=success,
        sample_window_completed=success,
        global_deadline_reached=False,
        oom_detected=False,
        worker_returncodes=tuple(
            0 if clean_exit else None for _ in range(concurrency)
        ),
        workers_exited_cleanly=clean_exit,
        controller_terminated_workers=not clean_exit,
        initial_free_vram_bytes=initial,
        minimum_free_vram_bytes=initial - peak,
        peak_incremental_vram_bytes=peak,
        peak_host_rss_bytes=host_peak,
        aggregate_cpu_seconds=cpu_cores * 20.0,
        average_cpu_cores=cpu_cores,
        system_average_busy_cores=cpu_cores + baseline_busy_cores,
        baseline_system_busy_cores=baseline_busy_cores,
        required_phases=tuple(placement.DEFAULT_REQUIRED_PHASES),
        completed_phases_by_worker=tuple(
            phases for _ in range(concurrency)
        ),
        phase_contract_satisfied=success,
        log_paths=(),
        phase_evidence_paths=(),
        reason="test",
    )


def autotune_kwargs(tmp_path: Path) -> dict:
    return {
        "machine": snapshot(load_average_1m=387.5),
        "repo_root": tmp_path,
        "work_dir": tmp_path / "work",
        "selected_device_ids": [str(index) for index in range(8)],
        "total_tasks": 62,
        "workload_fingerprint": {"workload": "countdown"},
        "command_factory": lambda index, root: [
            "unused",
            str(index),
            str(root),
        ],
        "base_environment": None,
        "required_host_memory_bytes_per_worker": 4 * GIB,
        "host_memory_headroom_fraction": 0.15,
        "per_worker_host_memory_safety_factor": 1.25,
        "cpu_fraction": 0.85,
        "per_worker_cpu_safety_factor": 1.5,
        "minimum_cpu_cores_per_worker": 1.0,
        "gpu_memory_headroom_fraction": 0.10,
        "per_worker_vram_safety_factor": 1.25,
        "max_slots_per_gpu": 8,
        "single_probe_seconds": 240,
        "validation_probe_seconds": 300,
        "probe_budget_seconds": 600,
        "required_free_floor_bytes": 4 * GIB,
        "required_probe_phases": placement.DEFAULT_REQUIRED_PHASES,
        "system_busy_cores_sampler": lambda: 20.0,
    }


def test_high_load_average_does_not_collapse_idle_gpu_pool(
    tmp_path: Path,
) -> None:
    calls: list[int] = []

    def fake_probe(**kwargs):
        concurrency = int(kwargs["concurrency"])
        calls.append(concurrency)
        if concurrency == 1:
            return probe_result(1)
        if concurrency == 5:
            return probe_result(5, peak_gib=60)
        raise AssertionError(concurrency)

    document = placement.autotune_single_gpu_task_placement(
        **autotune_kwargs(tmp_path),
        probe_runner=fake_probe,
    )

    assert calls == [1, 5]
    assert document["selection"]["selected_device_ids"] == [
        str(index) for index in range(8)
    ]
    assert document["selection"]["slots_per_gpu"] == 5
    capacity = document["selection"]["capacity"]
    assert capacity["machine_load_average_1m_diagnostic"] == 387.5
    assert capacity["load_average_used_as_hard_capacity"] is False


def test_candidate_cpu_projection_can_force_bounded_backoff(
    tmp_path: Path,
) -> None:
    calls: list[int] = []

    def fake_probe(**kwargs):
        concurrency = int(kwargs["concurrency"])
        calls.append(concurrency)
        if concurrency == 1:
            return probe_result(1)
        if concurrency == 5:
            return probe_result(
                5,
                peak_gib=60,
                average_cpu_cores=40.0,
            )
        if concurrency == 4:
            return probe_result(
                4,
                peak_gib=48,
                average_cpu_cores=8.0,
            )
        raise AssertionError(concurrency)

    document = placement.autotune_single_gpu_task_placement(
        **autotune_kwargs(tmp_path),
        probe_runner=fake_probe,
    )

    assert calls == [1, 5, 4]
    assert document["selection"]["slots_per_gpu"] == 4
    first = document["probe"]["candidate_checks"][0]
    assert first["cpu_runtime_capacity_ok"] is False
    assert first["accepted"] is False


def test_measured_external_cpu_occupancy_can_fail_closed(
    tmp_path: Path,
) -> None:
    def fake_probe(**kwargs):
        assert int(kwargs["concurrency"]) == 1
        return probe_result(1, baseline_busy_cores=325.5)

    with pytest.raises(RuntimeResourceError, match="no safe capacity"):
        placement.autotune_single_gpu_task_placement(
            **autotune_kwargs(tmp_path),
            probe_runner=fake_probe,
        )


def _phase_writer_code(
    path: Path,
    *,
    sleep_seconds: float,
) -> str:
    payload = {
        "contract_version": placement.PROBE_CONTRACT_VERSION,
        "required_phases": list(placement.DEFAULT_REQUIRED_PHASES),
        "completed_phases": list(placement.DEFAULT_REQUIRED_PHASES),
        "complete": True,
    }
    return (
        "import json,pathlib,time; "
        f"pathlib.Path({str(path)!r}).write_text(json.dumps({payload!r})); "
        f"time.sleep({sleep_seconds})"
    )


def _patch_probe_measurements(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        placement,
        "_gpu_free_bytes",
        lambda *_args, **_kwargs: 20 * GIB,
    )
    monkeypatch.setattr(
        placement,
        "process_tree_rss",
        lambda _pid: 1024,
    )
    monkeypatch.setattr(
        placement,
        "_process_tree_cpu_seconds",
        lambda _pid: 0.5,
    )
    monkeypatch.setattr(
        placement,
        "_system_cpu_ticks",
        lambda **_kwargs: (0, 100),
    )


def test_phase_complete_worker_must_exit_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_probe_measurements(monkeypatch)

    def command_factory(_index: int, root: Path) -> list[str]:
        return [
            sys.executable,
            "-c",
            _phase_writer_code(
                root / placement.PROBE_STATE_FILENAME,
                sleep_seconds=0.0,
            ),
        ]

    result = placement.probe_same_gpu_concurrency(
        device_id="0",
        concurrency=1,
        command_factory=command_factory,
        environment=None,
        log_dir=tmp_path / "clean",
        sample_seconds=3.0,
        global_deadline_monotonic=time.monotonic() + 4.0,
        poll_interval_seconds=0.05,
        minimum_live_seconds=0.0,
        phase_exit_grace_seconds=0.5,
        terminate_grace_seconds=0.1,
    )

    assert result.success is True
    assert result.worker_returncodes == (0,)
    assert result.workers_exited_cleanly is True
    assert result.controller_terminated_workers is False


def test_phase_complete_but_hanging_worker_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_probe_measurements(monkeypatch)

    def command_factory(_index: int, root: Path) -> list[str]:
        return [
            sys.executable,
            "-c",
            _phase_writer_code(
                root / placement.PROBE_STATE_FILENAME,
                sleep_seconds=5.0,
            ),
        ]

    result = placement.probe_same_gpu_concurrency(
        device_id="0",
        concurrency=1,
        command_factory=command_factory,
        environment=None,
        log_dir=tmp_path / "hang",
        sample_seconds=2.0,
        global_deadline_monotonic=time.monotonic() + 3.0,
        poll_interval_seconds=0.05,
        minimum_live_seconds=0.0,
        phase_exit_grace_seconds=0.2,
        terminate_grace_seconds=0.1,
    )

    assert result.success is False
    assert result.controller_terminated_workers is True
    assert result.reason == "worker_exit_after_phase_complete_timeout"


def test_proc_stat_busy_measurement_excludes_iowait(tmp_path: Path) -> None:
    stat = tmp_path / "stat"
    stat.write_text(
        "cpu  0 0 0 0 0 0 0 0 0 0\n"
        "cpu0 10 0 10 30 40 0 0 0 0 0\n",
        encoding="utf-8",
    )
    busy, total = placement._system_cpu_ticks(  # noqa: SLF001
        proc_stat_path=stat,
        cpu_ids=[0],
    )
    assert busy == 20
    assert total == 90
