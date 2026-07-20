from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from drpo import runtime_cpu_capacity as cpu
from drpo import runtime_resource_adapters as adapters
from drpo.runtime_resource_autotune import (
    GIB,
    GPUDevice,
    MachineSnapshot,
    canonical_json_sha256,
    load_json,
)


def snapshot(
    *,
    cpus: int = 64,
    available_gib: float = 128,
    gpus: tuple[GPUDevice, ...] = (),
) -> MachineSnapshot:
    total = 256 * GIB
    available = int(available_gib * GIB)
    return MachineSnapshot(
        logical_cpu_count=cpus,
        memory_total_bytes=total,
        memory_available_bytes=available,
        effective_memory_limit_bytes=total,
        effective_memory_current_bytes=total - available,
        effective_memory_available_bytes=available,
        swap_total_bytes=0,
        swap_free_bytes=0,
        cgroup_version="v2",
        load_average_1m=999.0,
        gpus=gpus,
    )


def write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    contract = tmp_path / "contract.json"
    run_spec = tmp_path / "run_spec.json"
    grid = tmp_path / "grid.json"
    contract.write_text('{"contract": 1}\n', encoding="utf-8")
    run_spec.write_text('{"run_spec": 1}\n', encoding="utf-8")
    grid.write_text('{"grid": 1}\n', encoding="utf-8")
    return contract, run_spec, grid


def fingerprint_kwargs(tmp_path: Path) -> dict:
    contract, run_spec, grid = write_inputs(tmp_path)
    return {
        "repo_root": Path.cwd(),
        "contract_path": contract,
        "run_spec_path": run_spec,
        "grid_path": grid,
        "probe_steps": 100,
        "probe_seed": 999,
        "probe_seconds": 1.0,
        "throughput_retention_fraction": 1.0,
        "fallback_workers": 60,
        "cpu_fraction": 0.85,
        "memory_headroom_fraction": 0.15,
        "per_worker_safety_factor": 1.2,
        "per_worker_cpu_safety_factor": 1.25,
        "minimum_cpu_cores_per_worker": 1.0,
        "max_workers": None,
        "max_growth_factor": 3.0,
        "revalidation_samples": 3,
        "revalidation_sample_seconds": 1.0,
    }


def test_e7_fingerprint_changes_when_runtime_policy_changes(tmp_path: Path) -> None:
    common = fingerprint_kwargs(tmp_path)
    first = adapters.e7_resource_fingerprint(**common)
    second = adapters.e7_resource_fingerprint(**{**common, "cpu_fraction": 0.75})
    assert canonical_json_sha256(first) != canonical_json_sha256(second)
    assert first["scientific_matrix_changed"] is False
    assert first["selection_policy"]["load_average_role"] == "diagnostic_only"


def test_e7_selects_from_measured_cpu_and_refuses_silent_replan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract, run_spec, grid = write_inputs(tmp_path)
    probe_dir = tmp_path / "probe"
    (probe_dir / "trainer_output").mkdir(parents=True)
    (probe_dir / "trainer_output" / "checkpoint.bin").write_bytes(b"large-placeholder")
    binding = cpu.CPUBinding(
        affinity_cpu_ids=tuple(range(64)),
        affinity_source="sched_getaffinity",
        cgroup_version=None,
        current_cgroup_path=None,
        quota_domains=(),
    )
    monkeypatch.setattr(
        adapters,
        "build_e7_probe_command",
        lambda **_kwargs: (["demo"], probe_dir, {}, 120, tmp_path),
    )
    monkeypatch.setattr(adapters.cpu, "discover_cpu_binding", lambda **_kwargs: binding)
    monkeypatch.setattr(
        adapters.measured,
        "_measure_representative_resources",
        lambda **_kwargs: {
            "command": ["demo"],
            "started_utc": "a",
            "finished_utc": "b",
            "elapsed_seconds": 1.0,
            "peak_rss_bytes": 1 * GIB,
            "process_tree_cpu_seconds": 0.5,
            "measured_cpu_cores": 0.5,
            "cpu_interval": cpu.CPUIntervalMeasurement(
                elapsed_seconds=1.0,
                affinity_cpu_ids=binding.affinity_cpu_ids,
                system_busy_tick_delta=1,
                system_total_tick_delta=100,
                system_busy_cores=0.5,
                quota_domain_usage_cores=(),
                started_monotonic_seconds=0.0,
                finished_monotonic_seconds=1.0,
            ).as_dict(),
            "returncode": -15,
            "timed_out": True,
            "controller_terminated": True,
            "process_group_alive_after_cleanup": False,
            "log_path": str(probe_dir / "log"),
        },
    )
    kwargs = dict(
        machine=snapshot(cpus=64, available_gib=100),
        repo_root=Path.cwd(),
        contract_path=contract,
        run_spec_path=run_spec,
        grid_path=grid,
        work_dir=tmp_path / "work",
        fallback_workers=16,
        probe_steps=100,
        probe_seed=999,
        probe_seconds=1.0,
        cpu_fraction=0.75,
        memory_headroom_fraction=0.20,
        per_worker_safety_factor=1.25,
        per_worker_cpu_safety_factor=1.25,
        minimum_cpu_cores_per_worker=1.0,
        max_workers=48,
        max_growth_factor=3.0,
        minimum_branches_for_probe=8,
        revalidation_samples=3,
        revalidation_sample_seconds=1.0,
    )
    first = adapters.select_e7_runtime(**kwargs)
    assert first["mode"] == "auto"
    assert first["selection"]["selected_workers"] == 48
    assert first["selector_policy_version"] == 2
    assert first["load_average_is_diagnostic_only"] is True
    assert not (probe_dir / "trainer_output").exists()

    with pytest.raises(adapters.RuntimeResourceError, match="already exists"):
        adapters.select_e7_runtime(**kwargs)


def test_legacy_e7_cache_is_always_incompatible() -> None:
    assert adapters.cached_cpu_selection(Path("legacy.json")) is None


def frozen_e8_config() -> dict:
    return {
        "experiment_id": "x",
        "sweep": {"methods": ["a"]},
        "execution": {
            "required_gpu_count": 8,
            "default_gpus": list(range(8)),
            "parallel_cells_per_gpu": 1,
        },
        "scientific": {"untouched": [1, 2, 3]},
    }


def test_e8_parent_runtime_view_changes_only_runtime_gpu_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(adapters, "validate_e8_scientific_config", lambda _config: None)
    original = frozen_e8_config()
    runtime_view = adapters.e8_parent_runtime_config(original, selected_gpu_count=3)
    assert original["execution"]["required_gpu_count"] == 8
    assert runtime_view["execution"]["required_gpu_count"] == 3
    assert runtime_view["scientific"] == original["scientific"]
    assert runtime_view["execution"]["runtime_resource_override"][
        "scientific_matrix_changed"
    ] is False

    observed: dict = {}

    def original_validator(value: dict) -> None:
        observed.update(copy.deepcopy(value))

    adapters.validate_e8_parent_runtime_config(
        runtime_view,
        original_validator=original_validator,
    )
    assert observed["execution"]["required_gpu_count"] == 8
    assert "runtime_resource_override" not in observed["execution"]
    assert observed["scientific"] == original["scientific"]


def test_e8_selection_is_limited_by_host_ram_and_vram(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(adapters, "validate_e8_scientific_config", lambda _config: None)
    sweep = tmp_path / "sweep.yaml"
    base = tmp_path / "base.yaml"
    sweep.write_text(yaml.safe_dump(frozen_e8_config()), encoding="utf-8")
    base.write_text("model: demo\n", encoding="utf-8")
    machine_value = snapshot(
        cpus=64,
        available_gib=13,
        gpus=tuple(
            GPUDevice(str(i), "A", 24 * GIB, 20 * GIB, 0.0) for i in range(4)
        ),
    )
    document = adapters.select_e8_runtime(
        machine=machine_value,
        repo_root=tmp_path,
        sweep_config_path=sweep,
        base_config_path=base,
        work_dir=tmp_path / "work",
        candidate_device_ids=["0", "1", "2", "3"],
        total_tasks=72,
        required_free_gpu_memory_gib=8.0,
        required_host_memory_gib_per_device=4.0,
        gpu_memory_headroom_fraction=0.1,
        host_memory_headroom_fraction=0.1,
        maximum_gpu_utilization_percent=20.0,
        max_devices=None,
    )
    assert document["selection"]["selected_device_ids"] == ["0", "1"]
    assert document["selection"]["host_memory_slot_limit"] == 2
    written = load_json(tmp_path / "work" / "RUNTIME_SELECTION.json")
    assert written["scientific_matrix_changed"] is False
    assert "configured_vram_floor_not_dynamic_training_peak_probe" in written["limitations"]

    adapters.select_e8_runtime(
        machine=machine_value,
        repo_root=tmp_path,
        sweep_config_path=sweep,
        base_config_path=base,
        work_dir=tmp_path / "work",
        candidate_device_ids=["0", "1", "2", "3"],
        total_tasks=72,
        required_free_gpu_memory_gib=8.0,
        required_host_memory_gib_per_device=4.0,
        gpu_memory_headroom_fraction=0.1,
        host_memory_headroom_fraction=0.1,
        maximum_gpu_utilization_percent=20.0,
        max_devices=1,
    )
    history = list(
        (tmp_path / "work" / "_runtime_resources" / "selection_history").glob("*.json")
    )
    assert len(history) == 1
    assert load_json(history[0])["selection"]["selected_device_ids"] == ["0", "1"]
