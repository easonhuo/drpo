from __future__ import annotations

import os
import sys
from pathlib import Path

from drpo.runtime_resource_acceptance_commands import (
    ENV_EXECUTABLE,
    gpu_selection_command,
    pool_command,
)


def _profile() -> dict[str, object]:
    return {
        "e8": {
            "model_path": "/model",
            "bank": "/bank.jsonl",
            "val": "/val.jsonl",
            "global_calibration": "/calibration.json",
            "base_config": "/base.yaml",
            "sweep_config": "/sweep.yaml",
            "required_free_gpu_memory_gib": 8.0,
            "required_host_memory_gib_per_worker": 4.0,
            "gpu_memory_headroom_fraction": 0.12,
            "host_memory_headroom_fraction": 0.15,
            "per_worker_host_memory_safety_factor": 1.25,
            "per_worker_vram_safety_factor": 1.25,
            "cpu_fraction": 0.85,
            "per_worker_cpu_safety_factor": 1.5,
            "minimum_cpu_cores_per_worker": 1.0,
            "maximum_gpu_utilization_percent": 20.0,
            "max_devices": 8,
            "max_slots_per_gpu": 2,
            "single_probe_seconds": 120.0,
            "validation_probe_seconds": 120.0,
            "probe_budget_seconds": 600.0,
            "probe_free_floor_gib": 4.0,
        }
    }


def test_nested_pool_and_gpu_commands_pin_separate_pythonpaths(
    tmp_path: Path, monkeypatch
) -> None:
    harness_repo = tmp_path / "harness"
    gpu_repo = tmp_path / "gpu"
    monkeypatch.setenv("PYTHONPATH", "/inherited/pythonpath")

    inner = gpu_selection_command(
        gpu_repo,
        _profile(),
        work_dir=tmp_path / "gpu-work",
        gpu_ids=("0", "1"),
    )
    outer = pool_command(
        harness_repo,
        cpu_pool="0-3",
        identity=tmp_path / "RESOURCE_POOL.json",
        command=inner,
        gpu_ids=("0", "1"),
    )

    assert outer[0] == ENV_EXECUTABLE
    assert outer[1] == (
        f"PYTHONPATH={(harness_repo / 'src').resolve()}"
        f"{os.pathsep}/inherited/pythonpath"
    )
    assert outer[2] == sys.executable
    assert outer[3] == str(harness_repo / "scripts/run_with_resource_pool.py")

    delegated_index = outer.index("--") + 1
    assert outer[delegated_index] == ENV_EXECUTABLE
    assert outer[delegated_index + 1] == (
        f"PYTHONPATH={(gpu_repo / 'src').resolve()}"
        f"{os.pathsep}/inherited/pythonpath"
    )
    assert outer[delegated_index + 2] == sys.executable
    assert outer[delegated_index + 3] == str(
        gpu_repo / "scripts/run_countdown_e8_oracle_offline_v2_taper_auto.py"
    )
    gpu_flag = outer.index("--gpus", delegated_index)
    assert outer[gpu_flag + 1] == "0,1"


def test_pool_dry_run_keeps_environment_boundary(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("PYTHONPATH", raising=False)
    repo = tmp_path / "repo"
    command = pool_command(
        repo,
        cpu_pool="0-1",
        identity=tmp_path / "RESOURCE_POOL.json",
        command=[sys.executable, "-c", "print('ok')"],
        dry_run=True,
    )

    assert command[:2] == [
        ENV_EXECUTABLE,
        f"PYTHONPATH={(repo / 'src').resolve()}",
    ]
    assert "--dry-run" in command
