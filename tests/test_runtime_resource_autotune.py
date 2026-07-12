from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import pytest

from drpo.runtime_resource_autotune import (
    GIB,
    GPUDevice,
    MachineSnapshot,
    RuntimeResourceError,
    canonical_json_sha256,
    discover_machine,
    measure_command_peak_memory,
    select_cpu_workers,
    select_gpu_devices,
    selection_document,
)


def snapshot(
    *,
    cpus: int = 32,
    available_gib: float = 64,
    limit_gib: float = 128,
    gpus: tuple[GPUDevice, ...] = (),
) -> MachineSnapshot:
    available = int(available_gib * GIB)
    limit = int(limit_gib * GIB)
    return MachineSnapshot(
        logical_cpu_count=cpus,
        memory_total_bytes=limit,
        memory_available_bytes=available,
        effective_memory_limit_bytes=limit,
        effective_memory_current_bytes=max(0, limit - available),
        effective_memory_available_bytes=available,
        swap_total_bytes=8 * GIB,
        swap_free_bytes=8 * GIB,
        cgroup_version="v2",
        load_average_1m=0.0,
        gpus=gpus,
    )


def test_canonical_json_hash_is_order_independent() -> None:
    assert canonical_json_sha256({"b": 2, "a": 1}) == canonical_json_sha256(
        {"a": 1, "b": 2}
    )


def test_discover_machine_honors_cgroup_v2_memory_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        "MemTotal: 131072000 kB\n"
        "MemAvailable: 98304000 kB\n"
        "SwapTotal: 8388608 kB\n"
        "SwapFree: 8388608 kB\n",
        encoding="utf-8",
    )
    loadavg = tmp_path / "loadavg"
    loadavg.write_text("3.50 2.00 1.00 1/10 123\n", encoding="utf-8")
    cgroup = tmp_path / "cgroup"
    cgroup.mkdir()
    (cgroup / "cgroup.controllers").write_text("memory cpu\n", encoding="utf-8")
    (cgroup / "memory.max").write_text(str(32 * GIB), encoding="utf-8")
    (cgroup / "memory.current").write_text(str(8 * GIB), encoding="utf-8")
    monkeypatch.setattr("os.sched_getaffinity", lambda _pid: set(range(24)))

    machine = discover_machine(
        meminfo_path=meminfo,
        loadavg_path=loadavg,
        cgroup_root=cgroup,
        nvidia_smi=str(tmp_path / "missing-nvidia-smi"),
    )
    assert machine.logical_cpu_count == 24
    assert machine.cgroup_version == "v2"
    assert machine.load_average_1m == pytest.approx(3.5)
    assert machine.effective_memory_limit_bytes == 32 * GIB
    assert machine.effective_memory_current_bytes == 8 * GIB
    assert machine.effective_memory_available_bytes == 24 * GIB
    assert machine.gpus == ()


def test_cpu_selection_is_cpu_bound_when_memory_is_abundant() -> None:
    selected = select_cpu_workers(
        snapshot(cpus=100, available_gib=512, limit_gib=512),
        total_tasks=432,
        fallback_workers=60,
        per_worker_peak_bytes=1 * GIB,
        cpu_fraction=0.80,
        memory_headroom_fraction=0.10,
        per_worker_safety_factor=1.0,
        max_growth_factor=3.0,
    )
    assert selected.selected_workers == 80
    assert selected.cpu_limit == 80
    assert selected.memory_limit > 80


def test_cpu_selection_is_memory_bound_even_when_cpu_is_available() -> None:
    selected = select_cpu_workers(
        snapshot(cpus=384, available_gib=40, limit_gib=128),
        total_tasks=432,
        fallback_workers=60,
        per_worker_peak_bytes=2 * GIB,
        cpu_fraction=0.90,
        memory_headroom_fraction=0.20,
        per_worker_safety_factor=1.25,
        max_growth_factor=4.0,
    )
    assert selected.memory_limit == 12
    assert selected.selected_workers == 12
    assert selected.cpu_limit > selected.selected_workers


def test_cpu_selection_reserves_capacity_for_existing_load() -> None:
    machine = dataclasses.replace(snapshot(cpus=32, available_gib=128), load_average_1m=10.0)
    selected = select_cpu_workers(
        machine,
        total_tasks=100,
        fallback_workers=16,
        per_worker_peak_bytes=1 * GIB,
        cpu_fraction=0.75,
        max_growth_factor=4.0,
    )
    assert selected.cpu_limit == 14
    assert selected.selected_workers == 14


def test_cpu_selection_without_probe_uses_conservative_fallback() -> None:
    selected = select_cpu_workers(
        snapshot(cpus=384, available_gib=512, limit_gib=512),
        total_tasks=432,
        fallback_workers=60,
        per_worker_peak_bytes=None,
    )
    assert selected.selected_workers == 60
    assert selected.reason == "no_worker_memory_probe_use_conservative_fallback"


def test_cpu_selection_fails_when_one_worker_cannot_fit() -> None:
    with pytest.raises(RuntimeResourceError, match="insufficient host memory"):
        select_cpu_workers(
            snapshot(cpus=64, available_gib=1, limit_gib=64),
            total_tasks=10,
            fallback_workers=4,
            per_worker_peak_bytes=2 * GIB,
            memory_headroom_fraction=0.20,
        )


def test_gpu_selection_applies_visibility_busy_and_vram_gates() -> None:
    machine = snapshot(
        gpus=(
            GPUDevice("0", "A", 24 * GIB, 20 * GIB, 0.0),
            GPUDevice("1", "A", 24 * GIB, 20 * GIB, 75.0),
            GPUDevice("2", "A", 24 * GIB, 6 * GIB, 0.0),
        )
    )
    selected = select_gpu_devices(
        machine,
        candidate_device_ids=["0", "1", "2", "9"],
        total_tasks=72,
        required_free_bytes_per_device=8 * GIB,
        headroom_fraction=0.10,
        maximum_utilization_percent=20.0,
    )
    assert selected.selected_device_ids == ("0",)
    reasons = {item["device_id"]: item["reason"] for item in selected.rejected_devices}
    assert reasons == {
        "1": "device_busy",
        "2": "insufficient_free_memory",
        "9": "not_visible",
    }


def test_gpu_selection_rejects_duplicate_candidates() -> None:
    with pytest.raises(RuntimeResourceError, match="unique"):
        select_gpu_devices(
            snapshot(gpus=(GPUDevice("0", "A", 24 * GIB, 24 * GIB, 0.0),)),
            candidate_device_ids=["0", "0"],
            total_tasks=2,
            required_free_bytes_per_device=4 * GIB,
        )


def test_measure_command_peak_memory_captures_process_tree(tmp_path: Path) -> None:
    script = tmp_path / "allocate.py"
    script.write_text(
        "import time\n"
        "payload = bytearray(24 * 1024 * 1024)\n"
        "payload[0] = 1\n"
        "time.sleep(10)\n",
        encoding="utf-8",
    )
    result = measure_command_peak_memory(
        [sys.executable, str(script)],
        cwd=tmp_path,
        environment=None,
        log_path=tmp_path / "probe.log",
        sample_seconds=0.8,
        sample_interval_seconds=0.05,
        terminate_grace_seconds=0.2,
        accept_timeout=True,
    )
    assert result.timed_out is True
    assert result.peak_rss_bytes >= 16 * 1024 * 1024
    assert Path(result.log_path).is_file()


def test_selection_document_separates_runtime_from_science(tmp_path: Path) -> None:
    machine = snapshot()
    document = selection_document(
        adapter_id="demo",
        resource_fingerprint={"workload": "x"},
        machine=machine,
        mode="auto",
        selection={"workers": 8},
        probe={"peak_rss_bytes": 123},
        fallback={"workers": 4},
        repo_root=tmp_path,
        limitations=["demo_only"],
    )
    assert document["scientific_matrix_changed"] is False
    assert document["selection"] == {"workers": 8}
    assert document["resource_fingerprint_sha256"] == canonical_json_sha256(
        {"workload": "x"}
    )
    assert json.loads(json.dumps(document))["limitations"] == ["demo_only"]
