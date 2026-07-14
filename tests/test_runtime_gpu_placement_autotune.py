from __future__ import annotations

import json
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
    heterogeneous: bool = False,
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
                "H100" if heterogeneous and index == gpu_count - 1 else "H20",
                (80 if heterogeneous and index == gpu_count - 1 else 96) * GIB,
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
    host_gib: float | None = None,
    reason: str = "test",
    deadline: bool = False,
    phase_complete: bool | None = None,
) -> placement.GPUConcurrencyProbeResult:
    initial = 90 * GIB
    peak = int(peak_gib * GIB)
    host_peak = int((3 * concurrency if host_gib is None else host_gib) * GIB)
    phase_ok = success if phase_complete is None else phase_complete
    completed = (
        tuple(placement.DEFAULT_REQUIRED_PHASES) if phase_ok else ("model_loaded",)
    )
    return placement.GPUConcurrencyProbeResult(
        concurrency=concurrency,
        device_id="0",
        started_utc="a",
        finished_utc="b",
        elapsed_seconds=1.0,
        success=success,
        sample_window_completed=phase_ok,
        global_deadline_reached=deadline,
        oom_detected=not success and reason == "oom",
        worker_returncodes=tuple(None for _ in range(concurrency)),
        initial_free_vram_bytes=initial,
        minimum_free_vram_bytes=initial - peak,
        peak_incremental_vram_bytes=peak,
        peak_host_rss_bytes=host_peak,
        required_phases=tuple(placement.DEFAULT_REQUIRED_PHASES),
        completed_phases_by_worker=tuple(completed for _ in range(concurrency)),
        phase_contract_satisfied=phase_ok,
        log_paths=(),
        phase_evidence_paths=(),
        reason=reason,
    )


def autotune_kwargs(tmp_path: Path) -> dict:
    return {
        "machine": snapshot(),
        "repo_root": tmp_path,
        "work_dir": tmp_path / "work",
        "selected_device_ids": [str(index) for index in range(8)],
        "total_tasks": 62,
        "workload_fingerprint": {"workload": "countdown"},
        "command_factory": lambda index, root: ["unused", str(index), str(root)],
        "base_environment": None,
        "required_host_memory_bytes_per_worker": 4 * GIB,
        "host_memory_headroom_fraction": 0.10,
        "per_worker_host_memory_safety_factor": 1.25,
        "cpu_fraction": 0.85,
        "gpu_memory_headroom_fraction": 0.10,
        "per_worker_vram_safety_factor": 1.25,
        "max_slots_per_gpu": 8,
        "single_probe_seconds": 60,
        "validation_probe_seconds": 60,
        "probe_budget_seconds": 600,
        "required_free_floor_bytes": 4 * GIB,
        "required_probe_phases": placement.DEFAULT_REQUIRED_PHASES,
    }


def test_derive_slots_uses_measured_capacity_not_a_fixed_constant() -> None:
    derived = placement.derive_slots_per_gpu(
        free_vram_bytes=90 * GIB,
        single_worker_peak_vram_bytes=12 * GIB,
        device_count=8,
        total_tasks=100,
        host_worker_limit_total=64,
        cpu_worker_limit_total=80,
        gpu_memory_headroom_fraction=0.10,
        per_worker_vram_safety_factor=1.25,
        max_slots_per_gpu=8,
    )
    assert derived["candidate"] == 5
    assert derived["vram_limit_per_device"] == 5


def test_derive_slots_honors_host_and_cpu_limits() -> None:
    host_limited = placement.derive_slots_per_gpu(
        free_vram_bytes=90 * GIB,
        single_worker_peak_vram_bytes=8 * GIB,
        device_count=8,
        total_tasks=100,
        host_worker_limit_total=16,
        cpu_worker_limit_total=80,
        gpu_memory_headroom_fraction=0.10,
        per_worker_vram_safety_factor=1.0,
        max_slots_per_gpu=8,
    )
    cpu_limited = placement.derive_slots_per_gpu(
        free_vram_bytes=90 * GIB,
        single_worker_peak_vram_bytes=8 * GIB,
        device_count=8,
        total_tasks=100,
        host_worker_limit_total=80,
        cpu_worker_limit_total=24,
        gpu_memory_headroom_fraction=0.10,
        per_worker_vram_safety_factor=1.0,
        max_slots_per_gpu=8,
    )
    assert host_limited["candidate"] == 2
    assert cpu_limited["candidate"] == 3


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


def test_autotune_rejects_heterogeneous_gpu_pool(tmp_path: Path) -> None:
    kwargs = autotune_kwargs(tmp_path)
    kwargs["machine"] = snapshot(heterogeneous=True)
    with pytest.raises(RuntimeResourceError, match="homogeneous"):
        placement.autotune_single_gpu_task_placement(
            **kwargs,
            probe_runner=lambda **_kwargs: probe_result(1, success=True),
        )


def test_autotune_backs_off_to_highest_phase_complete_candidate(tmp_path: Path) -> None:
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
        **autotune_kwargs(tmp_path),
        probe_runner=fake_probe,
    )
    assert calls == [1, 5, 4]
    assert document["selection"]["slots_per_gpu"] == 4
    assert document["selection"]["total_runtime_slots"] == 32
    assert document["selection"]["slot_device_ids"].count("0") == 4
    assert document["probe_contract_version"] == placement.PROBE_CONTRACT_VERSION
    assert document["scientific_matrix_changed"] is False


def test_autotune_rejects_live_candidate_without_required_phases(tmp_path: Path) -> None:
    calls: list[int] = []

    def fake_probe(**kwargs):
        concurrency = int(kwargs["concurrency"])
        calls.append(concurrency)
        if concurrency == 1:
            return probe_result(1, success=True, peak_gib=12)
        if concurrency == 5:
            return probe_result(
                5,
                success=False,
                peak_gib=30,
                reason="required_probe_phases_incomplete",
                phase_complete=False,
            )
        if concurrency == 4:
            return probe_result(4, success=True, peak_gib=48)
        raise AssertionError(concurrency)

    document = placement.autotune_single_gpu_task_placement(
        **autotune_kwargs(tmp_path),
        probe_runner=fake_probe,
    )
    assert calls == [1, 5, 4]
    assert document["selection"]["slots_per_gpu"] == 4
    assert document["probe"]["candidate_checks"][0][
        "phase_contract_satisfied"
    ] is False


def test_autotune_rejects_candidate_that_exceeds_projected_host_memory(
    tmp_path: Path,
) -> None:
    calls: list[int] = []

    def fake_probe(**kwargs):
        concurrency = int(kwargs["concurrency"])
        calls.append(concurrency)
        if concurrency == 1:
            return probe_result(1, success=True, peak_gib=12, host_gib=3)
        if concurrency == 5:
            return probe_result(5, success=True, peak_gib=60, host_gib=40)
        if concurrency == 4:
            return probe_result(4, success=True, peak_gib=48, host_gib=20)
        raise AssertionError(concurrency)

    document = placement.autotune_single_gpu_task_placement(
        **autotune_kwargs(tmp_path),
        probe_runner=fake_probe,
    )
    assert calls == [1, 5, 4]
    assert document["selection"]["slots_per_gpu"] == 4
    first_check = document["probe"]["candidate_checks"][0]
    assert first_check["probe_success"] is True
    assert first_check["host_capacity_ok"] is False


def test_incomplete_single_worker_envelope_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(RuntimeResourceError, match="did not complete"):
        placement.autotune_single_gpu_task_placement(
            **autotune_kwargs(tmp_path),
            probe_runner=lambda **_kwargs: probe_result(
                1,
                success=False,
                phase_complete=False,
                reason="required_probe_phases_incomplete",
            ),
        )
    failure = load_json(tmp_path / "work" / "RUNTIME_SELECTION.json")
    assert failure["mode"] == "failed"
    assert failure["selection"] is None


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
        **autotune_kwargs(tmp_path),
        probe_runner=fake_probe,
    )
    assert calls == [1, 5, 4, 3]
    assert document["selection"]["slots_per_gpu"] == 1
    assert document["selection"]["reason"] == (
        "single_worker_resource_envelope_validated_only"
    )


def test_exact_cached_selection_skips_probe_after_dynamic_revalidation(
    tmp_path: Path,
) -> None:
    def first_probe(**kwargs):
        concurrency = int(kwargs["concurrency"])
        return probe_result(
            concurrency,
            success=True,
            peak_gib=12 * concurrency,
        )

    common = autotune_kwargs(tmp_path)
    first = placement.autotune_single_gpu_task_placement(
        **common,
        probe_runner=first_probe,
    )

    def should_not_run(**_kwargs):
        raise AssertionError("exact phase-complete cache should skip the probe")

    second = placement.autotune_single_gpu_task_placement(
        **common,
        probe_runner=should_not_run,
    )
    assert first["selection"]["slots_per_gpu"] == second["selection"]["slots_per_gpu"]
    assert second["mode"] == "cached"


def test_v1_cache_without_phase_contract_is_invalidated(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir(parents=True)
    (work / "RUNTIME_SELECTION.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "adapter_id": placement.ADAPTER_ID,
                "selection": {"selected_device_ids": [str(i) for i in range(8)]},
            }
        ),
        encoding="utf-8",
    )
    calls: list[int] = []

    def probe(**kwargs):
        concurrency = int(kwargs["concurrency"])
        calls.append(concurrency)
        return probe_result(concurrency, success=True, peak_gib=12 * concurrency)

    document = placement.autotune_single_gpu_task_placement(
        **autotune_kwargs(tmp_path),
        probe_runner=probe,
    )
    assert calls
    assert document["mode"] == "auto"
    assert document["probe_contract_version"] == placement.PROBE_CONTRACT_VERSION


def _phase_writer_code(path: Path, phases: list[str], *, sleep_seconds: float) -> str:
    payload = {
        "contract_version": placement.PROBE_CONTRACT_VERSION,
        "required_phases": list(placement.DEFAULT_REQUIRED_PHASES),
        "completed_phases": phases,
        "complete": phases == list(placement.DEFAULT_REQUIRED_PHASES),
    }
    return (
        "import json,time,pathlib; "
        f"pathlib.Path({str(path)!r}).write_text(json.dumps({payload!r})); "
        f"time.sleep({sleep_seconds})"
    )


def test_real_probe_requires_phase_contract_and_cleans_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(placement, "_gpu_free_bytes", lambda *_args, **_kwargs: 20 * GIB)

    def command_factory(_index: int, root: Path) -> list[str]:
        return [
            sys.executable,
            "-c",
            _phase_writer_code(
                root / placement.PROBE_STATE_FILENAME,
                list(placement.DEFAULT_REQUIRED_PHASES),
                sleep_seconds=5,
            ),
        ]

    result = placement.probe_same_gpu_concurrency(
        device_id="0",
        concurrency=2,
        command_factory=command_factory,
        environment=None,
        log_dir=tmp_path / "probe",
        sample_seconds=1.0,
        global_deadline_monotonic=time.monotonic() + 2.0,
        poll_interval_seconds=0.05,
        minimum_live_seconds=0.0,
        terminate_grace_seconds=0.1,
    )
    assert result.success is True
    assert result.phase_contract_satisfied is True
    assert len(result.phase_evidence_paths) == 2
    assert not any(
        path.is_dir() and path.name.startswith("worker_")
        for path in (tmp_path / "probe").iterdir()
    )


def test_real_probe_rejects_process_that_only_stays_alive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(placement, "_gpu_free_bytes", lambda *_args, **_kwargs: 20 * GIB)
    result = placement.probe_same_gpu_concurrency(
        device_id="0",
        concurrency=1,
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
        minimum_live_seconds=0.0,
        terminate_grace_seconds=0.1,
    )
    assert result.success is False
    assert result.phase_contract_satisfied is False
    assert result.reason == "required_probe_phases_incomplete"


def test_real_probe_rejects_partial_phase_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(placement, "_gpu_free_bytes", lambda *_args, **_kwargs: 20 * GIB)

    def command_factory(_index: int, root: Path) -> list[str]:
        return [
            sys.executable,
            "-c",
            _phase_writer_code(
                root / placement.PROBE_STATE_FILENAME,
                ["model_loaded", "training_peak_completed"],
                sleep_seconds=5,
            ),
        ]

    result = placement.probe_same_gpu_concurrency(
        device_id="0",
        concurrency=1,
        command_factory=command_factory,
        environment=None,
        log_dir=tmp_path / "probe",
        sample_seconds=0.3,
        global_deadline_monotonic=time.monotonic() + 2.0,
        poll_interval_seconds=0.05,
        minimum_live_seconds=0.0,
        terminate_grace_seconds=0.1,
    )
    assert result.success is False
    assert result.reason == "required_probe_phases_incomplete"


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
