from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from drpo import countdown_e8_oracle_offline_v2_taper_slot_runtime as slot_runtime
from drpo import runtime_gpu_placement_autotune as placement
from drpo.runtime_resource_autotune import (
    GIB,
    GPUDevice,
    MachineSnapshot,
    RuntimeResourceError,
    load_json,
)


def snapshot(
    *,
    available_host_gib: float = 256,
    gpu_free_gib: float = 90,
    gpu_count: int = 8,
) -> MachineSnapshot:
    host_total = 512 * GIB
    host_available = int(available_host_gib * GIB)
    return MachineSnapshot(
        logical_cpu_count=128,
        memory_total_bytes=host_total,
        memory_available_bytes=host_available,
        effective_memory_limit_bytes=host_total,
        effective_memory_current_bytes=host_total - host_available,
        effective_memory_available_bytes=host_available,
        swap_total_bytes=0,
        swap_free_bytes=0,
        cgroup_version="v2",
        load_average_1m=0.0,
        gpus=tuple(
            GPUDevice(
                str(index),
                "H20",
                96 * GIB,
                int(gpu_free_gib * GIB),
                0.0,
            )
            for index in range(gpu_count)
        ),
    )


def probe_result(
    concurrency: int,
    *,
    success: bool,
    peak_gib: float = 12,
    reason: str = "test",
    deadline: bool = False,
) -> placement.GPUConcurrencyProbeResult:
    initial = 90 * GIB
    peak = int(peak_gib * GIB)
    return placement.GPUConcurrencyProbeResult(
        concurrency=concurrency,
        device_id="0",
        started_utc="a",
        finished_utc="b",
        elapsed_seconds=1.0,
        success=success,
        sample_window_completed=success,
        global_deadline_reached=deadline,
        oom_detected=not success and reason == "oom",
        worker_returncodes=tuple(None for _ in range(concurrency)),
        initial_free_vram_bytes=initial,
        minimum_free_vram_bytes=initial - peak,
        peak_incremental_vram_bytes=peak,
        log_paths=(),
        reason=reason,
    )


def test_derive_slots_uses_measured_capacity_not_a_fixed_constant() -> None:
    derived = placement.derive_slots_per_gpu(
        free_vram_bytes=90 * GIB,
        single_worker_peak_vram_bytes=12 * GIB,
        device_count=8,
        total_tasks=100,
        host_worker_limit_total=64,
        gpu_memory_headroom_fraction=0.10,
        per_worker_vram_safety_factor=1.25,
        max_slots_per_gpu=8,
    )
    # usable=81 GiB, reserved=15 GiB -> five workers per GPU.
    assert derived["candidate"] == 5
    assert derived["vram_limit_per_device"] == 5


def test_derive_slots_honors_host_memory_limit() -> None:
    derived = placement.derive_slots_per_gpu(
        free_vram_bytes=90 * GIB,
        single_worker_peak_vram_bytes=8 * GIB,
        device_count=8,
        total_tasks=100,
        host_worker_limit_total=16,
        gpu_memory_headroom_fraction=0.10,
        per_worker_vram_safety_factor=1.0,
        max_slots_per_gpu=8,
    )
    assert derived["host_limit_per_device"] == 2
    assert derived["candidate"] == 2


def test_bounded_backoff_does_not_enumerate_every_slot_count() -> None:
    assert placement.bounded_backoff_candidates(8) == [8, 7, 4, 1]
    assert placement.bounded_backoff_candidates(3) == [3, 2, 1]
    assert placement.bounded_backoff_candidates(1) == [1]


def test_slot_runtime_accepts_exact_automatic_placement() -> None:
    document = {
        "adapter_id": placement.ADAPTER_ID,
        "scientific_matrix_changed": False,
        "selection": {
            "selected_device_ids": ["0", "1"],
            "slots_per_gpu": 3,
            "total_runtime_slots": 6,
            "slot_device_ids": ["0", "0", "0", "1", "1", "1"],
        },
    }
    slots, expanded = slot_runtime._validated_placement(  # noqa: SLF001
        document,
        gpu_pool=["0", "1"],
        total_tasks=8,
    )
    assert slots == 3
    assert expanded == ["0", "0", "0", "1", "1", "1"]


def test_slot_runtime_rejects_inconsistent_or_scientific_placement() -> None:
    document = {
        "adapter_id": placement.ADAPTER_ID,
        "scientific_matrix_changed": False,
        "selection": {
            "selected_device_ids": ["0", "1"],
            "slots_per_gpu": 2,
            "total_runtime_slots": 4,
            "slot_device_ids": ["0", "0", "1", "0"],
        },
    }
    with pytest.raises(RuntimeResourceError, match="slot expansion"):
        slot_runtime._validated_placement(  # noqa: SLF001
            document,
            gpu_pool=["0", "1"],
            total_tasks=8,
        )
    document["scientific_matrix_changed"] = True
    with pytest.raises(RuntimeResourceError, match="scientific matrix"):
        slot_runtime._validated_placement(  # noqa: SLF001
            document,
            gpu_pool=["0", "1"],
            total_tasks=8,
        )


def test_autotune_backs_off_to_highest_validated_candidate(tmp_path: Path) -> None:
    calls: list[int] = []

    def fake_probe(**kwargs):
        concurrency = int(kwargs["concurrency"])
        calls.append(concurrency)
        if concurrency == 1:
            return probe_result(1, success=True, peak_gib=12)
        if concurrency == 5:
            return probe_result(5, success=False, peak_gib=70, reason="oom")
        if concurrency == 4:
            return probe_result(4, success=True, peak_gib=48)
        raise AssertionError(concurrency)

    document = placement.autotune_single_gpu_task_placement(
        machine=snapshot(),
        repo_root=tmp_path,
        work_dir=tmp_path / "work",
        selected_device_ids=[str(index) for index in range(8)],
        total_tasks=62,
        workload_fingerprint={"workload": "countdown"},
        command_factory=lambda index, root: ["unused", str(index), str(root)],
        base_environment=None,
        required_host_memory_bytes_per_worker=4 * GIB,
        host_memory_headroom_fraction=0.10,
        gpu_memory_headroom_fraction=0.10,
        per_worker_vram_safety_factor=1.25,
        max_slots_per_gpu=8,
        single_probe_seconds=60,
        validation_probe_seconds=60,
        probe_budget_seconds=600,
        required_free_floor_bytes=4 * GIB,
        probe_runner=fake_probe,
    )
    assert calls == [1, 5, 4]
    assert document["selection"]["slots_per_gpu"] == 4
    assert document["selection"]["total_runtime_slots"] == 32
    assert document["selection"]["slot_device_ids"].count("0") == 4
    assert document["scientific_matrix_changed"] is False


def test_autotune_falls_back_to_one_when_no_higher_candidate_validates(
    tmp_path: Path,
) -> None:
    calls: list[int] = []

    def fake_probe(**kwargs):
        concurrency = int(kwargs["concurrency"])
        calls.append(concurrency)
        if concurrency == 1:
            return probe_result(1, success=True, peak_gib=12)
        return probe_result(concurrency, success=False, reason="nonzero")

    document = placement.autotune_single_gpu_task_placement(
        machine=snapshot(),
        repo_root=tmp_path,
        work_dir=tmp_path / "work",
        selected_device_ids=[str(index) for index in range(8)],
        total_tasks=62,
        workload_fingerprint={"workload": "countdown"},
        command_factory=lambda index, root: ["unused", str(index), str(root)],
        base_environment=None,
        required_host_memory_bytes_per_worker=4 * GIB,
        host_memory_headroom_fraction=0.10,
        gpu_memory_headroom_fraction=0.10,
        per_worker_vram_safety_factor=1.25,
        max_slots_per_gpu=8,
        single_probe_seconds=60,
        validation_probe_seconds=60,
        probe_budget_seconds=600,
        required_free_floor_bytes=4 * GIB,
        probe_runner=fake_probe,
    )
    assert calls == [1, 5, 4, 2]
    assert document["selection"]["slots_per_gpu"] == 1
    assert document["selection"]["reason"] == "single_worker_probe_validated_only"


def test_exact_cached_selection_skips_probe_after_dynamic_revalidation(
    tmp_path: Path,
) -> None:
    def first_probe(**kwargs):
        concurrency = int(kwargs["concurrency"])
        return probe_result(concurrency, success=True, peak_gib=12 * concurrency)

    common = dict(
        machine=snapshot(),
        repo_root=tmp_path,
        work_dir=tmp_path / "work",
        selected_device_ids=[str(index) for index in range(8)],
        total_tasks=62,
        workload_fingerprint={"workload": "countdown"},
        command_factory=lambda index, root: ["unused", str(index), str(root)],
        base_environment=None,
        required_host_memory_bytes_per_worker=4 * GIB,
        host_memory_headroom_fraction=0.10,
        gpu_memory_headroom_fraction=0.10,
        per_worker_vram_safety_factor=1.25,
        max_slots_per_gpu=8,
        single_probe_seconds=60,
        validation_probe_seconds=60,
        probe_budget_seconds=600,
        required_free_floor_bytes=4 * GIB,
    )
    first = placement.autotune_single_gpu_task_placement(
        **common,
        probe_runner=first_probe,
    )

    def should_not_run(**_kwargs):
        raise AssertionError("exact safe cache should skip the probe")

    second = placement.autotune_single_gpu_task_placement(
        **common,
        probe_runner=should_not_run,
    )
    assert first["selection"]["slots_per_gpu"] == second["selection"]["slots_per_gpu"]
    assert second["mode"] == "cached"
    assert load_json(tmp_path / "work" / "RUNTIME_SELECTION.json")["mode"] == "cached"


def test_real_probe_terminates_workers_and_removes_probe_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(placement, "_gpu_free_bytes", lambda *_args, **_kwargs: 20 * GIB)
    result = placement.probe_same_gpu_concurrency(
        device_id="0",
        concurrency=2,
        command_factory=lambda _index, _root: [
            sys.executable,
            "-c",
            "import time; time.sleep(5)",
        ],
        environment=None,
        log_dir=tmp_path / "probe",
        sample_seconds=0.3,
        global_deadline_monotonic=time.monotonic() + 2.0,
        poll_interval_seconds=0.05,
        minimum_live_seconds=0.1,
        terminate_grace_seconds=0.1,
    )
    assert result.success is True
    probe_root = tmp_path / "probe"
    assert list(probe_root.glob("worker_*.log"))
    assert not any(
        path.is_dir() and path.name.startswith("worker_")
        for path in probe_root.iterdir()
    )


def test_real_probe_rejects_oom_signature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(placement, "_gpu_free_bytes", lambda *_args, **_kwargs: 20 * GIB)
    result = placement.probe_same_gpu_concurrency(
        device_id="0",
        concurrency=1,
        command_factory=lambda _index, _root: [
            sys.executable,
            "-c",
            "print('CUDA out of memory'); raise SystemExit(1)",
        ],
        environment=None,
        log_dir=tmp_path / "probe",
        sample_seconds=1.0,
        global_deadline_monotonic=time.monotonic() + 2.0,
        poll_interval_seconds=0.05,
        minimum_live_seconds=0.0,
        terminate_grace_seconds=0.1,
    )
    assert result.success is False
    assert result.oom_detected is True
    assert result.reason == "oom_signature_detected"
