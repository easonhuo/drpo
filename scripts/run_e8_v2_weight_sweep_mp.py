#!/usr/bin/env python3
"""E8 v2 negative-weight fine-grid sweep - PARALLEL with process pool.

24 cells (6 multipliers × 4 seeds) run in PARALLEL across 6 GPUs.
Each process calls train_cell() directly (no worker subprocess).
"""
from __future__ import annotations

import json
import os
import sys
from multiprocessing import Process, Queue
from pathlib import Path

import yaml

DRPO_SRC = Path("/root/drpo/src/drpo")
sys.path.insert(0, str(DRPO_SRC))

import countdown_e8_oracle_offline_v2_matrix as mx  # noqa: E402

CONFIG_PATH = Path("/root/drpo/configs/countdown_e8_base_rl_replay_0p5b.yaml")
MODEL = Path("/root/models/Qwen2.5-0.5B-Instruct")
V2_BANK = Path("/root/experiment_output/e8_oracle_bank_v2/data/offline_bank_v2.jsonl")
V2_VAL = Path("/root/experiment_output/e8_oracle_bank_v2/data/val.jsonl")
V2_TEST = Path("/root/experiment_output/e8_oracle_bank_v2/data/test.jsonl")
WORK_DIR = Path("/root/experiment_output/e8_v2_weight_sweep")
LOGS_DIR = Path("/root/countdown/logs")
CALIB_JSON = Path("/root/experiment_output/e8_v2_matrix/calibration/base/calibration.json")

MULTIPLIERS = [0.0, 1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4]
SEED_OFFSETS = [0, 1000, 2000, 3000]
GPUS = ["0", "1", "2", "3", "4", "5"]


def build_cells() -> list[mx.Cell]:
    cells = []
    for mult in MULTIPLIERS:
        for seed_off in SEED_OFFSETS:
            method = "positive_only" if mult == 0.0 else "bank_global_matched"
            mult_str = f"x{mult:.4f}".replace(".", "p")
            name = (
                f"base_{method}_{mult_str}_seed{seed_off}"
                if mult > 0
                else f"base_positive_only_seed{seed_off}"
            )
            cell = mx.Cell(
                name=name,
                init="base",
                method=method,
                calibration="base",
                negative_scale_multiplier=mult if mult > 0 else 1.0,
                seed_offset=seed_off,
                kind="train",
            )
            cells.append(cell)
    return cells


def run_cell_process(cell: mx.Cell, gpu: str, result_queue: Queue) -> None:
    """Run one cell in a subprocess with CUDA_VISIBLE_DEVICES=gpu."""
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    config = yaml.safe_load(CONFIG_PATH.read_text())
    data_paths = {
        "train": V2_BANK,
        "validation": V2_VAL,
        "test": V2_TEST,
        "split_manifest": V2_BANK,
    }
    identity = {"experiment": "e8_v2_weight_sweep", "calibration": "base"}

    try:
        summary = mx.train_cell(
            cell,
            MODEL,
            WORK_DIR,
            V2_BANK,
            data_paths,
            config,
            CALIB_JSON,
            sft_dir=None,
            identity=identity,
        )
        result_queue.put((cell.name, 0, summary))
    except Exception as exc:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        result_queue.put((cell.name, 1, str(exc)))


def main() -> int:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    cells = build_cells()
    print(
        f"[sweep] {len(cells)} cells, running in PARALLEL on GPUs {GPUS}",
        flush=True,
    )

    gpu_assignments = {
        cell.name: GPUS[index % len(GPUS)] for index, cell in enumerate(cells)
    }
    results = {}
    result_queue = Queue()
    processes: list[tuple[str, Process]] = []

    for cell in cells:
        gpu = gpu_assignments[cell.name]
        process = Process(target=run_cell_process, args=(cell, gpu, result_queue))
        process.start()
        processes.append((cell.name, process))
        print(f"[sweep] started {cell.name} on GPU {gpu}", flush=True)

        if len(processes) >= len(GPUS):
            name, returncode, payload = result_queue.get()
            results[name] = {
                "returncode": returncode,
                "status": "OK" if returncode == 0 else "FAIL",
                "payload": payload,
            }
            print(f"[sweep] finished {name} -> rc={returncode}", flush=True)
            for process_name, running_process in list(processes):
                if process_name == name:
                    running_process.join()
                    processes.remove((process_name, running_process))
                    break

    while processes:
        name, returncode, payload = result_queue.get()
        results[name] = {
            "returncode": returncode,
            "status": "OK" if returncode == 0 else "FAIL",
            "payload": payload,
        }
        print(f"[sweep] finished {name} -> rc={returncode}", flush=True)
        for process_name, running_process in list(processes):
            if process_name == name:
                running_process.join()
                processes.remove((process_name, running_process))
                break

    manifest = {
        "experiment": "e8_v2_weight_sweep",
        "cells": {
            name: {
                "returncode": result["returncode"],
                "status": result["status"],
            }
            for name, result in results.items()
        },
        "all_ok": all(result["returncode"] == 0 for result in results.values()),
    }
    (WORK_DIR / "sweep_status.json").write_text(json.dumps(manifest, indent=2))
    print(f"[sweep] done. all_ok={manifest['all_ok']}", flush=True)
    return 0 if manifest["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
