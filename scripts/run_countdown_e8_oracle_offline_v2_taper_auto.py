#!/usr/bin/env python3
"""Opt-in E8 taper runner with host-RAM and GPU-slot safety selection."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import yaml

from drpo import countdown_e8_oracle_offline_v2_taper_runtime as runtime
from drpo.runtime_resource_adapters import (
    E8_ADAPTER_ID,
    e8_parent_runtime_config,
    select_e8_runtime,
    validate_e8_parent_runtime_config,
)
from drpo.runtime_resource_autotune import RuntimeResourceError, discover_machine


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
    parser.add_argument("--required-host-memory-gib-per-gpu", type=float, default=4.0)
    parser.add_argument("--gpu-memory-headroom-fraction", type=float, default=0.12)
    parser.add_argument("--host-memory-headroom-fraction", type=float, default=0.15)
    parser.add_argument("--maximum-gpu-utilization-percent", type=float, default=20.0)
    parser.add_argument("--max-devices", type=int)
    parser.add_argument("--meminfo", default="/proc/meminfo")
    parser.add_argument("--loadavg", default="/proc/loadavg")
    parser.add_argument("--cgroup-root", default="/sys/fs/cgroup")
    parser.add_argument("--nvidia-smi", default="nvidia-smi")
    return parser


def _load_original_config(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeResourceError("E8 sweep config root must be a mapping")
    runtime.core.validate_sweep_config(value)
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
            "refusing to attach the auto runner to a pre-existing fixed E8 work directory; "
            "use the registered E8 resume controller for an interrupted fixed run"
        )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    work_dir = Path(args.work_dir).resolve()
    _reject_legacy_work_dir(work_dir)
    original_config = _load_original_config(args.sweep_config)
    if args.gpus:
        candidate_ids = [item.strip() for item in args.gpus.split(",") if item.strip()]
    else:
        candidate_ids = [str(value) for value in original_config["execution"]["default_gpus"]]
    cells = runtime.core.build_cells(original_config)
    machine = discover_machine(
        meminfo_path=args.meminfo,
        loadavg_path=args.loadavg,
        cgroup_root=args.cgroup_root,
        nvidia_smi=args.nvidia_smi,
    )
    document = select_e8_runtime(
        machine=machine,
        repo_root=repo,
        sweep_config_path=args.sweep_config,
        base_config_path=args.base_config,
        work_dir=args.work_dir,
        candidate_device_ids=candidate_ids,
        total_tasks=len(cells),
        required_free_gpu_memory_gib=args.required_free_gpu_memory_gib,
        required_host_memory_gib_per_device=args.required_host_memory_gib_per_gpu,
        gpu_memory_headroom_fraction=args.gpu_memory_headroom_fraction,
        host_memory_headroom_fraction=args.host_memory_headroom_fraction,
        maximum_gpu_utilization_percent=args.maximum_gpu_utilization_percent,
        max_devices=args.max_devices,
    )
    selected_ids = [str(value) for value in document["selection"]["selected_device_ids"]]
    selected_count = len(selected_ids)
    print(
        json.dumps(
            {
                "adapter_id": E8_ADAPTER_ID,
                "runtime_selection": str(
                    Path(args.work_dir).resolve() / "RUNTIME_SELECTION.json"
                ),
                "selected_device_ids": selected_ids,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    original_load_yaml = runtime.core.load_yaml
    original_validate = runtime.core.validate_sweep_config
    sweep_path = Path(args.sweep_config).resolve()

    def parent_load_yaml(path: str | Path) -> dict[str, Any]:
        value = original_load_yaml(path)
        if Path(path).resolve() == sweep_path:
            return e8_parent_runtime_config(value, selected_gpu_count=selected_count)
        return value

    def parent_validate(config: dict[str, Any]) -> None:
        execution = config.get("execution", {})
        marker = execution.get("runtime_resource_override") if isinstance(execution, dict) else None
        if isinstance(marker, dict) and marker.get("adapter_id") == E8_ADAPTER_ID:
            validate_e8_parent_runtime_config(
                config,
                original_validator=original_validate,
            )
        else:
            original_validate(config)

    runtime.core.load_yaml = parent_load_yaml
    runtime.core.validate_sweep_config = parent_validate
    runtime_args = copy.copy(args)
    runtime_args.gpus = ",".join(selected_ids)
    runtime_args.calibration_gpu = selected_ids[0]
    try:
        return runtime.run(runtime_args)
    finally:
        runtime.core.load_yaml = original_load_yaml
        runtime.core.validate_sweep_config = original_validate


if __name__ == "__main__":
    raise SystemExit(main())
