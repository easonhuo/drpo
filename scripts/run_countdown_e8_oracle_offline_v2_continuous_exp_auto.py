#!/usr/bin/env python3
"""Autotuned code-first launcher for the E8 continuous EXP alpha-by-c pilot."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from drpo import countdown_e8_continuous_exp_common as common
from drpo.runtime_resource_autotune import (
    GIB,
    RuntimeResourceError,
    atomic_write_json,
    discover_machine,
    select_gpu_devices,
    selection_document,
)

ADAPTER_ID = "e8_continuous_exp_grid_cuda_dev_v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "smoke", "run"))
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--base_config", required=True)
    parser.add_argument("--grid_config", required=True)
    parser.add_argument("--gpus", help="comma-separated candidate GPU ids")
    parser.add_argument("--required-free-gpu-memory-gib", type=float)
    parser.add_argument("--required-host-memory-gib-per-gpu", type=float)
    parser.add_argument("--gpu-memory-headroom-fraction", type=float)
    parser.add_argument("--host-memory-headroom-fraction", type=float)
    parser.add_argument("--maximum-gpu-utilization-percent", type=float)
    parser.add_argument("--max-devices", type=int)
    parser.add_argument("--meminfo", default="/proc/meminfo")
    parser.add_argument("--loadavg", default="/proc/loadavg")
    parser.add_argument("--cgroup-root", default="/sys/fs/cgroup")
    parser.add_argument("--nvidia-smi", default="nvidia-smi")
    parser.add_argument(
        "--allow-dev-unregistered",
        action="store_true",
        help="required acknowledgement that this is an unregistered dev pilot",
    )
    return parser


def _reject_existing_nonauto_work_dir(work_dir: Path) -> None:
    selection = work_dir / "RUNTIME_SELECTION.json"
    evidence = (
        work_dir / "SWEEP_PLAN.json",
        work_dir / "SWEEP_COMPLETE.json",
        work_dir / "SMOKE_GATE.json",
    )
    if any(path.exists() for path in evidence) and not selection.is_file():
        raise RuntimeResourceError(
            "refusing to attach the autotuned launcher to an existing non-autotuned work_dir"
        )


def _effective(value: float | None, config_value: Any) -> float:
    return float(config_value if value is None else value)


def _selection(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    execution = config["execution"]
    policy = execution["autotune"]
    candidate_ids = (
        [item.strip() for item in args.gpus.split(",") if item.strip()]
        if args.gpus
        else [str(value) for value in execution["default_gpus"]]
    )
    if not candidate_ids or len(set(candidate_ids)) != len(candidate_ids):
        raise RuntimeResourceError("candidate GPU ids must be non-empty and unique")
    required_free = _effective(
        args.required_free_gpu_memory_gib,
        policy["required_free_gpu_memory_gib"],
    )
    required_host = _effective(
        args.required_host_memory_gib_per_gpu,
        policy["required_host_memory_gib_per_gpu"],
    )
    gpu_headroom = _effective(
        args.gpu_memory_headroom_fraction,
        policy["gpu_memory_headroom_fraction"],
    )
    host_headroom = _effective(
        args.host_memory_headroom_fraction,
        policy["host_memory_headroom_fraction"],
    )
    max_utilization = _effective(
        args.maximum_gpu_utilization_percent,
        policy["maximum_gpu_utilization_percent"],
    )
    if required_free <= 0.0 or required_host <= 0.0:
        raise RuntimeResourceError("GPU and host-memory requirements must be positive")
    if not 0.0 <= host_headroom < 0.9:
        raise RuntimeResourceError("host-memory headroom must be in [0, 0.9)")

    machine = discover_machine(
        meminfo_path=args.meminfo,
        loadavg_path=args.loadavg,
        cgroup_root=args.cgroup_root,
        nvidia_smi=args.nvidia_smi,
    )
    required_host_bytes = int(required_host * GIB)
    usable_host_bytes = int(
        machine.effective_memory_available_bytes * (1.0 - host_headroom)
    )
    host_slot_limit = usable_host_bytes // required_host_bytes
    if host_slot_limit < 1:
        raise RuntimeResourceError("insufficient host memory for one E8 worker")
    effective_max = host_slot_limit
    if args.max_devices is not None:
        effective_max = min(effective_max, args.max_devices)
    cells = common.build_cells(config)
    selection = select_gpu_devices(
        machine,
        candidate_device_ids=candidate_ids,
        total_tasks=len(cells),
        required_free_bytes_per_device=int(required_free * GIB),
        headroom_fraction=gpu_headroom,
        maximum_utilization_percent=max_utilization,
        max_devices=effective_max,
    )
    payload = selection.as_dict()
    payload.update(
        {
            "host_memory_slot_limit": host_slot_limit,
            "required_host_memory_bytes_per_device": required_host_bytes,
            "host_memory_headroom_fraction": host_headroom,
        }
    )
    fingerprint = {
        "schema_version": 1,
        "adapter_id": ADAPTER_ID,
        "grid_config_sha256": common.sha256_file(args.grid_config),
        "base_config_sha256": common.sha256_file(args.base_config),
        "candidate_device_ids": candidate_ids,
        "required_free_gpu_memory_gib": required_free,
        "required_host_memory_gib_per_gpu": required_host,
        "gpu_memory_headroom_fraction": gpu_headroom,
        "host_memory_headroom_fraction": host_headroom,
        "maximum_gpu_utilization_percent": max_utilization,
        "max_devices": args.max_devices,
        "tuned_runtime_field": "active_gpu_device_slots",
        "max_processes_per_device": 1,
        "scientific_matrix_changed": False,
    }
    document = selection_document(
        adapter_id=ADAPTER_ID,
        resource_fingerprint=fingerprint,
        machine=machine,
        mode="auto",
        selection=payload,
        probe={
            "kind": "nvidia_smi_visibility_utilization_free_memory_plus_real_liveness",
            "dynamic_training_probe": "SMOKE_GATE.json",
            "scientific_evidence": False,
        },
        fallback={"device_ids": candidate_ids, "reason": "configured_candidate_pool"},
        repo_root=Path(__file__).resolve().parents[1],
        limitations=[
            "one_process_per_gpu_only",
            "configured_vram_floor_precedes_actual_two_step_liveness",
            "dev_branch_pilot_not_formal_execution",
        ],
    )
    atomic_write_json(Path(args.work_dir).resolve() / "RUNTIME_SELECTION.json", document)
    return document


def _core_command(
    args: argparse.Namespace, command: str, *, selected_ids: list[str]
) -> list[str]:
    repo = Path(__file__).resolve().parents[1]
    result = [
        sys.executable,
        str(repo / "src" / "drpo" / "countdown_e8_continuous_exp_runtime.py"),
        command,
        "--model_path",
        str(Path(args.model_path).resolve()),
        "--work_dir",
        str(Path(args.work_dir).resolve()),
        "--bank",
        str(Path(args.bank).resolve()),
        "--val",
        str(Path(args.val).resolve()),
        "--base_config",
        str(Path(args.base_config).resolve()),
        "--grid_config",
        str(Path(args.grid_config).resolve()),
    ]
    if command == "run":
        result.extend(["--gpus", ",".join(selected_ids)])
    return result


def _run_core(args: argparse.Namespace, command: str, selected_ids: list[str]) -> int:
    repo = Path(__file__).resolve().parents[1]
    environment = None
    if command == "smoke":
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = selected_ids[0]
    completed = subprocess.run(
        _core_command(args, command, selected_ids=selected_ids),
        cwd=repo,
        env=environment,
        check=False,
    )
    return int(completed.returncode)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.allow_dev_unregistered:
        raise RuntimeResourceError(
            "--allow-dev-unregistered is required: this dev branch is not an authoritative RunSpec"
        )
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    _reject_existing_nonauto_work_dir(work_dir)
    config = common.load_yaml(args.grid_config)
    common.validate_grid_config(config)
    document = _selection(args, config)
    selected_ids = [str(value) for value in document["selection"]["selected_device_ids"]]
    print(
        json.dumps(
            {
                "adapter_id": ADAPTER_ID,
                "selected_device_ids": selected_ids,
                "cell_count": len(common.build_cells(config)),
                "runtime_selection": str(work_dir / "RUNTIME_SELECTION.json"),
                "registration_state": "dev_code_first_unregistered",
            },
            sort_keys=True,
        ),
        flush=True,
    )

    if args.command == "plan":
        return _run_core(args, "plan", selected_ids)
    if args.command == "smoke":
        return _run_core(args, "smoke", selected_ids)

    plan_code = _run_core(args, "plan", selected_ids)
    if plan_code != 0:
        return plan_code
    smoke_code = _run_core(args, "smoke", selected_ids)
    if smoke_code != 0:
        return smoke_code
    gate = json.loads((work_dir / "SMOKE_GATE.json").read_text())
    if gate.get("status") != "PASS":
        raise RuntimeResourceError("representative liveness did not pass")
    return _run_core(args, "run", selected_ids)


if __name__ == "__main__":
    raise SystemExit(main())
