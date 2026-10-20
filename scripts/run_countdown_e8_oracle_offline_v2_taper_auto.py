#!/usr/bin/env python3
"""Opt-in E8 taper runner with measured-CPU phase-aware GPU placement."""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from drpo import countdown_e8_oracle_offline_v2_taper_resource_probe as resource_probe
from drpo import countdown_e8_oracle_offline_v2_taper_runtime as legacy_runtime
from drpo import countdown_e8_oracle_offline_v2_taper_slot_runtime as slot_runtime
from drpo import runtime_gpu_placement_autotune_v2 as placement
from drpo.runtime_gpu_placement_autotune_v2 import (
    ADAPTER_ID,
    DEFAULT_REQUIRED_PHASES,
    PROBE_CONTRACT_VERSION,
    SELECTOR_POLICY_VERSION,
    GPUConcurrencyProbeResult,
    autotune_single_gpu_task_placement,
    probe_same_gpu_concurrency,
)
from drpo.runtime_resource_autotune import (
    GIB,
    RuntimeResourceError,
    discover_machine,
    select_gpu_devices,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--global_calibration", required=True)
    parser.add_argument("--base_config", required=True)
    parser.add_argument("--sweep_config", required=True)
    parser.add_argument("--gpus", help="comma-separated candidate GPU ids")
    parser.add_argument("--required-free-gpu-memory-gib", type=float, default=8.0)
    parser.add_argument(
        "--required-host-memory-gib-per-worker",
        "--required-host-memory-gib-per-gpu",
        dest="required_host_memory_gib_per_worker",
        type=float,
        default=4.0,
    )
    parser.add_argument("--gpu-memory-headroom-fraction", type=float, default=0.12)
    parser.add_argument("--host-memory-headroom-fraction", type=float, default=0.15)
    parser.add_argument(
        "--per-worker-host-memory-safety-factor", type=float, default=1.25
    )
    parser.add_argument("--per-worker-vram-safety-factor", type=float, default=1.25)
    parser.add_argument("--cpu-fraction", type=float, default=0.85)
    parser.add_argument("--per-worker-cpu-safety-factor", type=float, default=1.5)
    parser.add_argument("--minimum-cpu-cores-per-worker", type=float, default=1.0)
    parser.add_argument("--maximum-gpu-utilization-percent", type=float, default=20.0)
    parser.add_argument("--max-devices", type=int)
    parser.add_argument("--max-slots-per-gpu", type=int, default=8)
    parser.add_argument("--single-probe-seconds", type=float, default=240.0)
    parser.add_argument("--validation-probe-seconds", type=float, default=300.0)
    parser.add_argument("--probe-budget-seconds", type=float, default=600.0)
    parser.add_argument("--probe-free-floor-gib", type=float, default=4.0)
    parser.add_argument("--meminfo", default="/proc/meminfo")
    parser.add_argument("--loadavg", default="/proc/loadavg")
    parser.add_argument("--cgroup-root", default="/sys/fs/cgroup")
    parser.add_argument("--nvidia-smi", default="nvidia-smi")
    return parser


def _load_original_config(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeResourceError("E8 sweep config root must be a mapping")
    legacy_runtime.core.validate_sweep_config(value)
    return value


def _reject_legacy_work_dir(work_dir: Path) -> None:
    selection_path = work_dir / "RUNTIME_SELECTION.json"
    legacy_evidence = (
        work_dir / "SWEEP_PLAN.json",
        work_dir / "SWEEP_COMPLETE.json",
        work_dir / "RUN_FAILED.json",
        work_dir / "run_manifest.json",
    )
    if any(path.exists() for path in legacy_evidence) and not selection_path.is_file():
        raise RuntimeResourceError(
            "refusing to attach the GPU-placement auto runner to a pre-existing fixed "
            "E8 work directory; use the registered E8 resume controller"
        )


def _ensure_calibration(args: argparse.Namespace, *, gpu_id: str) -> Path:
    repo = Path(__file__).resolve().parents[1]
    work_dir = Path(args.work_dir).resolve()
    calibration_path = work_dir / "calibration" / "taper_budget_calibration.json"
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    expected = legacy_runtime.calibration_identity(
        repo=repo,
        model_path=Path(args.model_path).resolve(),
        bank_path=Path(args.bank).resolve(),
        global_calibration_path=Path(args.global_calibration).resolve(),
        base_config_path=Path(args.base_config).resolve(),
        sweep_config_path=Path(args.sweep_config).resolve(),
    )
    if calibration_path.is_file():
        current = json.loads(calibration_path.read_text(encoding="utf-8"))
        if not legacy_runtime.calibration_matches(current, expected):
            raise RuntimeResourceError(
                "existing taper calibration is stale; use a new work_dir"
            )
        return calibration_path

    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    command = [
        sys.executable,
        str(Path(legacy_runtime.__file__).resolve()),
        "calibrate",
        "--model_path",
        str(Path(args.model_path).resolve()),
        "--bank",
        str(Path(args.bank).resolve()),
        "--global_calibration",
        str(Path(args.global_calibration).resolve()),
        "--base_config",
        str(Path(args.base_config).resolve()),
        "--sweep_config",
        str(Path(args.sweep_config).resolve()),
        "--output",
        str(calibration_path),
    ]
    completed = subprocess.run(command, cwd=repo, env=environment, check=False)
    if completed.returncode != 0:
        raise RuntimeResourceError(
            f"E8 calibration failed before GPU placement probe: {completed.returncode}"
        )
    return calibration_path


def _workload_fingerprint(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": 3,
        "experiment_id": legacy_runtime.EXPERIMENT_ID,
        "model_path": str(Path(args.model_path).resolve()),
        "bank_sha256": legacy_runtime.sha256_file(args.bank),
        "validation_sha256": legacy_runtime.sha256_file(args.val),
        "test_sha256": legacy_runtime.sha256_file(args.test),
        "global_calibration_sha256": legacy_runtime.sha256_file(
            args.global_calibration
        ),
        "base_config_sha256": legacy_runtime.sha256_file(args.base_config),
        "sweep_config_sha256": legacy_runtime.sha256_file(args.sweep_config),
        "worker_runtime_sha256": legacy_runtime.sha256_file(legacy_runtime.__file__),
        "resource_probe_sha256": legacy_runtime.sha256_file(resource_probe.__file__),
        "placement_selector_sha256": legacy_runtime.sha256_file(placement.__file__),
        "selector_policy_version": SELECTOR_POLICY_VERSION,
        "probe_contract_version": PROBE_CONTRACT_VERSION,
        "required_probe_phases": list(DEFAULT_REQUIRED_PHASES),
        "placement_topology": "one_gpu_per_independent_task",
        "scientific_matrix_changed": False,
    }


def _phase_peak_probe_runner(**kwargs: Any) -> GPUConcurrencyProbeResult:
    result = probe_same_gpu_concurrency(**kwargs)
    reported_peak = max(
        (
            resource_probe.reported_peak_from_state(path)
            for path in result.phase_evidence_paths
        ),
        default=0,
    )
    if reported_peak <= result.peak_incremental_vram_bytes:
        return result
    return replace(result, peak_incremental_vram_bytes=reported_peak)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    work_dir = Path(args.work_dir).resolve()
    _reject_legacy_work_dir(work_dir)
    original_config = _load_original_config(args.sweep_config)
    if args.gpus:
        candidate_ids = [item.strip() for item in args.gpus.split(",") if item.strip()]
    else:
        candidate_ids = [
            str(value) for value in original_config["execution"]["default_gpus"]
        ]
    cells = legacy_runtime.core.build_cells(original_config)
    machine = discover_machine(
        meminfo_path=args.meminfo,
        loadavg_path=args.loadavg,
        cgroup_root=args.cgroup_root,
        nvidia_smi=args.nvidia_smi,
    )
    static_selection = select_gpu_devices(
        machine,
        candidate_device_ids=candidate_ids,
        total_tasks=len(cells),
        required_free_bytes_per_device=int(args.required_free_gpu_memory_gib * GIB),
        headroom_fraction=args.gpu_memory_headroom_fraction,
        maximum_utilization_percent=args.maximum_gpu_utilization_percent,
        max_devices=args.max_devices,
    )
    static_ids = [str(value) for value in static_selection.selected_device_ids]
    calibration_path = _ensure_calibration(args, gpu_id=static_ids[0])
    representative = cells[0]
    runtime_args = copy.copy(args)
    runtime_args.gpus = ",".join(static_ids)
    runtime_args.calibration_gpu = static_ids[0]

    def command_factory(_worker_index: int, worker_root: Path) -> list[str]:
        return resource_probe.resource_probe_command(
            args=runtime_args,
            cell=representative,
            output_dir=worker_root,
            calibration=calibration_path,
        )

    document = autotune_single_gpu_task_placement(
        machine=machine,
        repo_root=repo,
        work_dir=work_dir,
        selected_device_ids=static_ids,
        total_tasks=len(cells),
        workload_fingerprint=_workload_fingerprint(args),
        command_factory=command_factory,
        base_environment=None,
        required_host_memory_bytes_per_worker=int(
            args.required_host_memory_gib_per_worker * GIB
        ),
        host_memory_headroom_fraction=args.host_memory_headroom_fraction,
        per_worker_host_memory_safety_factor=(
            args.per_worker_host_memory_safety_factor
        ),
        cpu_fraction=args.cpu_fraction,
        per_worker_cpu_safety_factor=args.per_worker_cpu_safety_factor,
        minimum_cpu_cores_per_worker=args.minimum_cpu_cores_per_worker,
        gpu_memory_headroom_fraction=args.gpu_memory_headroom_fraction,
        per_worker_vram_safety_factor=args.per_worker_vram_safety_factor,
        max_slots_per_gpu=args.max_slots_per_gpu,
        single_probe_seconds=args.single_probe_seconds,
        validation_probe_seconds=args.validation_probe_seconds,
        probe_budget_seconds=args.probe_budget_seconds,
        required_free_floor_bytes=int(args.probe_free_floor_gib * GIB),
        nvidia_smi=args.nvidia_smi,
        required_probe_phases=DEFAULT_REQUIRED_PHASES,
        probe_runner=_phase_peak_probe_runner,
    )
    selected_ids = [
        str(value) for value in document["selection"]["selected_device_ids"]
    ]
    runtime_args.gpus = ",".join(selected_ids)
    runtime_args.calibration_gpu = selected_ids[0]
    print(
        json.dumps(
            {
                "adapter_id": ADAPTER_ID,
                "selector_policy_version": SELECTOR_POLICY_VERSION,
                "probe_contract_version": PROBE_CONTRACT_VERSION,
                "runtime_selection": str(work_dir / "RUNTIME_SELECTION.json"),
                "selected_device_ids": selected_ids,
                "slots_per_gpu": document["selection"]["slots_per_gpu"],
                "total_runtime_slots": document["selection"]["total_runtime_slots"],
                "scientific_matrix_changed": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return slot_runtime.run(
        runtime_args,
        placement_path=work_dir / "RUNTIME_SELECTION.json",
    )


if __name__ == "__main__":
    raise SystemExit(main())
