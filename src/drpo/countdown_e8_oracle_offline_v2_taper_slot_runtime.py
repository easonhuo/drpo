#!/usr/bin/env python3
"""Multi-slot parent scheduler for the frozen Countdown E8 taper worker.

The scientific worker, calibration code, configs, cells, and evaluation protocol stay
in ``countdown_e8_oracle_offline_v2_taper_runtime``. This module changes only parent
runtime placement by allowing multiple independent workers to share one selected GPU.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import LifoQueue
from typing import Any, Mapping

import torch

from drpo import countdown_e8_oracle_offline_v2_taper_runtime as legacy
from drpo.runtime_gpu_placement_autotune import ADAPTER_ID
from drpo.runtime_resource_autotune import RuntimeResourceError, load_json

VERSION = "0.1.0-gpu-slots"


def _validated_placement(
    document: Mapping[str, Any], *, gpu_pool: list[str], total_tasks: int
) -> tuple[int, list[str]]:
    if document.get("adapter_id") != ADAPTER_ID:
        raise RuntimeResourceError("GPU placement document has the wrong adapter id")
    if document.get("scientific_matrix_changed") is not False:
        raise RuntimeResourceError("GPU placement document changed the scientific matrix")
    selection = document.get("selection")
    if not isinstance(selection, Mapping):
        raise RuntimeResourceError("GPU placement selection is missing")
    selected_ids = [str(value) for value in selection.get("selected_device_ids", [])]
    if selected_ids != gpu_pool:
        raise RuntimeResourceError(
            "GPU placement selected-device ids do not match the runtime GPU pool"
        )
    slots_per_gpu = selection.get("slots_per_gpu")
    if not isinstance(slots_per_gpu, int) or slots_per_gpu < 1:
        raise RuntimeResourceError("GPU placement slots_per_gpu must be positive")
    expected_slots = [
        device_id for device_id in gpu_pool for _ in range(slots_per_gpu)
    ][:total_tasks]
    slot_device_ids = [str(value) for value in selection.get("slot_device_ids", [])]
    if slot_device_ids != expected_slots:
        raise RuntimeResourceError("GPU placement slot expansion is inconsistent")
    if int(selection.get("total_runtime_slots", -1)) != len(slot_device_ids):
        raise RuntimeResourceError("GPU placement total_runtime_slots is inconsistent")
    if not slot_device_ids:
        raise RuntimeResourceError("GPU placement produced no runtime slots")
    return slots_per_gpu, slot_device_ids


def run(args: Any, *, placement_path: str | Path) -> int:
    repo = Path(__file__).resolve().parents[2]
    model = Path(args.model_path).resolve()
    bank = Path(args.bank).resolve()
    val = Path(args.val).resolve()
    test = Path(args.test).resolve()
    global_calibration = Path(args.global_calibration).resolve()
    base_config = Path(args.base_config).resolve()
    sweep_config = Path(args.sweep_config).resolve()
    work_dir = Path(args.work_dir).resolve()
    calibration_path = work_dir / "calibration" / "taper_budget_calibration.json"
    methods_dir = work_dir / "methods"
    logs_dir = work_dir / "logs"
    for required_path in (
        model,
        bank,
        val,
        test,
        global_calibration,
        base_config,
        sweep_config,
        Path(placement_path).resolve(),
    ):
        if not required_path.exists():
            raise SystemExit(f"Missing required input: {required_path}")

    config = legacy.core.load_yaml(sweep_config)
    legacy.core.validate_sweep_config(config)
    cells = legacy.core.build_cells(config)
    gpu_pool = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpu_pool or len(gpu_pool) != len(set(gpu_pool)):
        raise SystemExit(f"Expected unique selected GPU ids, received {gpu_pool}")
    visible_count = torch.cuda.device_count()
    if visible_count < len(gpu_pool):
        raise SystemExit(
            f"Selected E8 placement requires {len(gpu_pool)} visible CUDA devices; "
            f"torch reports {visible_count}"
        )
    placement = load_json(placement_path)
    slots_per_gpu, slot_device_ids = _validated_placement(
        placement,
        gpu_pool=gpu_pool,
        total_tasks=len(cells),
    )

    work_dir.mkdir(parents=True, exist_ok=True)
    methods_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    expected_calibration_identity = legacy.calibration_identity(
        repo=repo,
        model_path=model,
        bank_path=bank,
        global_calibration_path=global_calibration,
        base_config_path=base_config,
        sweep_config_path=sweep_config,
    )
    if calibration_path.exists():
        current = json.loads(calibration_path.read_text(encoding="utf-8"))
        if not legacy.calibration_matches(current, expected_calibration_identity):
            raise RuntimeError("Existing taper calibration is stale; use a new work_dir")
    else:
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = args.calibration_gpu
        command = [
            sys.executable,
            str(Path(legacy.__file__).resolve()),
            "calibrate",
            "--model_path",
            str(model),
            "--bank",
            str(bank),
            "--global_calibration",
            str(global_calibration),
            "--base_config",
            str(base_config),
            "--sweep_config",
            str(sweep_config),
            "--output",
            str(calibration_path),
        ]
        completed = subprocess.run(command, cwd=repo, env=environment, check=False)
        if completed.returncode != 0:
            return int(completed.returncode)

    plan = {
        "schema_version": 1,
        "experiment_id": legacy.EXPERIMENT_ID,
        "runtime_version": legacy.VERSION,
        "slot_runtime_version": VERSION,
        "source": legacy.source_identity(repo),
        "gpu_pool": gpu_pool,
        "slots_per_gpu": slots_per_gpu,
        "slot_device_ids": slot_device_ids,
        "total_runtime_slots": len(slot_device_ids),
        "cell_count": len(cells),
        "methods": list(legacy.core.METHODS),
        "rho_values": [float(value) for value in config["sweep"]["rho_values"]],
        "seed_offsets": [int(value) for value in config["sweep"]["seed_offsets"]],
        "global_rerun": False,
        "scientific_matrix_changed": False,
        "cells": [cell.__dict__ | {"name": cell.name} for cell in cells],
    }
    legacy.atomic_json(work_dir / "SWEEP_PLAN.json", plan)
    legacy.atomic_json(
        work_dir / "RUNTIME_SLOTS.json",
        {
            "schema_version": 1,
            "adapter_id": ADAPTER_ID,
            "slot_runtime_version": VERSION,
            "selected_device_ids": gpu_pool,
            "slots_per_gpu": slots_per_gpu,
            "total_runtime_slots": len(slot_device_ids),
            "slot_device_ids": slot_device_ids,
            "per_gpu_assignment": {
                device_id: slot_device_ids.count(device_id) for device_id in gpu_pool
            },
            "scientific_matrix_changed": False,
            "placement_path": str(Path(placement_path).resolve()),
        },
    )

    pending: list[Any] = []
    for cell in cells:
        summary_path = methods_dir / cell.name / "summary.json"
        if not summary_path.exists():
            pending.append(cell)
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("runtime_source_sha256") != legacy.sha256_file(legacy.__file__):
            raise RuntimeError(f"Stale runtime identity for completed cell {cell.name}")
        if summary.get("best_evaluation", {}).get("deferred_posthoc_evaluation"):
            pending.append(cell)

    results: dict[str, Any] = {}
    results_lock = threading.Lock()
    available: LifoQueue[str] = LifoQueue()
    for device_id in slot_device_ids:
        available.put(device_id)

    def run_one(cell: Any) -> tuple[str, int, str, str]:
        gpu = available.get()
        log_path = logs_dir / f"{cell.name}.log"
        try:
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            command = legacy.worker_command(
                args=args,
                cell=cell,
                output_dir=methods_dir / cell.name,
                calibration=calibration_path,
            )
            with log_path.open("w", encoding="utf-8") as handle:
                handle.write(f"GPU={gpu}\nCOMMAND={' '.join(command)}\n")
                handle.flush()
                process = subprocess.Popen(
                    command,
                    cwd=repo,
                    env=environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                returncode = int(process.wait())
            return cell.name, returncode, str(log_path), gpu
        finally:
            available.put(gpu)

    with ThreadPoolExecutor(max_workers=len(slot_device_ids)) as executor:
        futures = {executor.submit(run_one, cell): cell for cell in pending}
        for future in as_completed(futures):
            name, returncode, log_path, gpu = future.result()
            with results_lock:
                results[name] = {
                    "returncode": returncode,
                    "status": "OK" if returncode == 0 else "FAIL",
                    "log": log_path,
                    "gpu": gpu,
                }
                legacy.atomic_json(
                    work_dir / "SWEEP_STATUS.json",
                    {
                        "experiment_id": legacy.EXPERIMENT_ID,
                        "expected_cells": len(cells),
                        "completed_this_invocation": len(results),
                        "remaining_after_invocation": len(pending) - len(results),
                        "slots_per_gpu": slots_per_gpu,
                        "total_runtime_slots": len(slot_device_ids),
                        "results": dict(sorted(results.items())),
                    },
                )
            print(f"[{name}] returncode={returncode} gpu={gpu} log={log_path}", flush=True)

    summaries = list(methods_dir.glob("*/summary.json"))
    failed = sorted(
        name for name, result in results.items() if result["returncode"] != 0
    )
    complete = len(summaries) == len(cells) and not failed
    legacy.atomic_json(
        work_dir / "SWEEP_COMPLETE.json",
        {
            "experiment_id": legacy.EXPERIMENT_ID,
            "runtime_version": legacy.VERSION,
            "slot_runtime_version": VERSION,
            "result_status": config["result_status"],
            "expected_cells": len(cells),
            "summary_count": len(summaries),
            "failed_cells": failed,
            "all_expected_cells_present": complete,
            "source": legacy.source_identity(repo),
            "calibration_sha256": legacy.sha256_file(calibration_path),
            "slots_per_gpu": slots_per_gpu,
            "total_runtime_slots": len(slot_device_ids),
            "scientific_matrix_changed": False,
        },
    )
    return 0 if complete else 1
