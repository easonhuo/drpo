#!/usr/bin/env python3
"""Runtime orchestration for the E8 alpha=1 c-only scan development pilot."""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import LifoQueue
from typing import Any, Sequence

from drpo.countdown_e8_alpha1_c_scan_common import (
    EXPECTED_POINTS,
    EXPERIMENT_ID,
    SEED_OFFSETS,
    VERSION,
    Cell,
    _identity,
    _identity_equal,
    atomic_json,
    build_cells,
    git_state,
    load_yaml,
    parameter_points,
    sha256_file,
    validate_grid_config,
)
from drpo.countdown_e8_alpha1_c_scan_trainer import train_cell


def _worker_command(args: argparse.Namespace, cell: Cell, output_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "worker",
        "--model_path",
        args.model_path,
        "--bank",
        args.bank,
        "--val",
        args.val,
        "--base_config",
        args.base_config,
        "--grid_config",
        args.grid_config,
        "--output_dir",
        str(output_dir),
        "--alpha",
        str(cell.alpha),
        "--c",
        str(cell.c),
        "--seed_offset",
        str(cell.seed_offset),
    ]


def _required_inputs(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path]:
    model = Path(args.model_path).resolve()
    bank = Path(args.bank).resolve()
    val = Path(args.val).resolve()
    base_config = Path(args.base_config).resolve()
    grid_config = Path(args.grid_config).resolve()
    for path in (model, bank, val, base_config, grid_config):
        if not path.exists():
            raise SystemExit(f"Missing required input: {path}")
    return model, bank, val, base_config, grid_config


def plan(args: argparse.Namespace) -> int:
    repo = Path(__file__).resolve().parents[2]
    model, bank, val, base_config, grid_config = _required_inputs(args)
    config = load_yaml(grid_config)
    cells = build_cells(config)
    payload = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "version": VERSION,
        "registration_state": str(config["registration_state"]),
        "source": git_state(repo),
        "model_path": str(model),
        "bank_sha256": sha256_file(bank),
        "validation_sha256": sha256_file(val),
        "base_config_sha256": sha256_file(base_config),
        "grid_config_sha256": sha256_file(grid_config),
        "parameter_points": EXPECTED_POINTS,
        "cell_count": len(cells),
        "test_data_used": False,
        "cells": [
            {
                "name": cell.name,
                "method": cell.method,
                "alpha": cell.alpha,
                "c": cell.c,
                "seed_offset": cell.seed_offset,
            }
            for cell in cells
        ],
    }
    output = Path(args.work_dir).resolve() / "SWEEP_PLAN.json"
    atomic_json(output, payload)
    print(json.dumps({"plan": str(output), "cells": len(cells)}), flush=True)
    return 0


def smoke(args: argparse.Namespace) -> int:
    repo = Path(__file__).resolve().parents[2]
    model, bank, val, base_config, grid_config = _required_inputs(args)
    config = load_yaml(grid_config)
    validate_grid_config(config)
    liveness = config["execution"]["liveness"]
    cell = Cell(
        alpha=float(liveness["representative_alpha"]),
        c=float(liveness["representative_c"]),
        seed_offset=SEED_OFFSETS[0],
    )
    output_dir = Path(args.work_dir).resolve() / "_liveness" / cell.name
    if output_dir.exists() and not (output_dir / "summary.json").exists():
        shutil.rmtree(output_dir)
    try:
        summary = train_cell(
            cell=cell,
            model_path=model,
            bank=bank,
            val=val,
            base_config_path=base_config,
            grid_config_path=grid_config,
            output_dir=output_dir,
            repo=repo,
            smoke=True,
        )
        passed = summary.get("numerical_failure") is None and int(
            summary.get("terminal_step", -1)
        ) == int(liveness["steps"])
        gate = {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "registration_state": str(config["registration_state"]),
            "status": "PASS" if passed else "FAIL",
            "scientific_evidence": False,
            "cell": cell.name,
            "summary": str(output_dir / "summary.json"),
            "run_identity": summary.get("run_identity"),
            "test_data_used": False,
        }
    except BaseException as error:
        gate = {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "registration_state": str(config["registration_state"]),
            "status": "FAIL",
            "scientific_evidence": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "test_data_used": False,
        }
        atomic_json(Path(args.work_dir).resolve() / "SMOKE_GATE.json", gate)
        raise
    atomic_json(Path(args.work_dir).resolve() / "SMOKE_GATE.json", gate)
    return 0 if gate["status"] == "PASS" else 1


def _aggregate(
    work_dir: Path, cells: Sequence[Cell], *, registration_state: str
) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    for cell in cells:
        path = work_dir / "methods" / cell.name / "summary.json"
        if path.is_file():
            summaries.append(json.loads(path.read_text()))
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        terminal = summary.get("terminal_metrics", {})
        bests = summary.get("metric_bests", {})
        rows.append(
            {
                "cell": summary["cell"],
                "method": summary["method"],
                "alpha": summary["alpha"],
                "c": summary["c"],
                "seed_offset": summary["seed_offset"],
                "best_greedy": (bests.get("val_greedy") or {}).get("value"),
                "best_pass_at_8": (bests.get("val_pass_at_8") or {}).get("value"),
                "best_pass_at_64": (bests.get("val_pass_at_64") or {}).get("value"),
                "best_valid_rate": (bests.get("val_valid_rate") or {}).get("value"),
                "terminal_greedy": terminal.get("val_greedy"),
                "terminal_pass_at_8": terminal.get("val_pass_at_8"),
                "terminal_pass_at_64": terminal.get("val_pass_at_64"),
                "terminal_valid_rate": terminal.get("val_valid_rate"),
                "terminal_step": summary.get("terminal_step"),
                "numerical_failure": summary.get("numerical_failure"),
                "test_data_used": False,
            }
        )
    aggregate_dir = work_dir / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        with (aggregate_dir / "per_cell_summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    numerical_failures = sum(row["numerical_failure"] is not None for row in rows)
    complete = len(rows) == len(cells)
    audit = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "registration_state": registration_state,
        "expected_cells": len(cells),
        "summary_count": len(rows),
        "all_expected_cells_present": complete,
        "numerical_failures": numerical_failures,
        "task_performance_status": "reported_not_adjudicated",
        "support_or_structure_boundary_status": (
            "not_formally_instrumented_valid_rate_only"
        ),
        "nan_inf_status": "observed" if numerical_failures else "not_observed",
        "test_data_used": False,
        "fixed_1200_steps_is_convergence": False,
        "method_ranking_claim_allowed": False,
        "status": (
            "PASS" if complete and numerical_failures == 0 else "INCOMPLETE_OR_FAILURE"
        ),
    }
    atomic_json(aggregate_dir / "terminal_audit.json", audit)
    return audit


def run(args: argparse.Namespace) -> int:
    repo = Path(__file__).resolve().parents[2]
    model, bank, val, base_config, grid_config = _required_inputs(args)
    config = load_yaml(grid_config)
    cells = build_cells(config)
    work_dir = Path(args.work_dir).resolve()
    smoke_gate = work_dir / "SMOKE_GATE.json"
    if not smoke_gate.is_file() or json.loads(smoke_gate.read_text()).get(
        "status"
    ) != "PASS":
        raise RuntimeError("A passing representative SMOKE_GATE.json is required")
    gpu_pool = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpu_pool or len(set(gpu_pool)) != len(gpu_pool):
        raise ValueError("At least one unique GPU id is required")
    methods_dir = work_dir / "methods"
    logs_dir = work_dir / "logs"
    methods_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    plan(args)

    pending: list[Cell] = []
    for cell in cells:
        summary_path = methods_dir / cell.name / "summary.json"
        if not summary_path.exists():
            pending.append(cell)
            continue
        summary = json.loads(summary_path.read_text())
        expected_identity = _identity(
            repo=repo,
            model_path=model,
            bank=bank,
            val=val,
            base_config=base_config,
            grid_config=grid_config,
            cell=cell,
            smoke=False,
        )
        if not _identity_equal(summary.get("run_identity", {}), expected_identity):
            raise RuntimeError(f"Stale completed-cell identity: {cell.name}")

    runtime_slots_per_gpu = int(args.runtime_slots_per_gpu)
    configured_slots = int(config["execution"]["parallel_cells_per_gpu"])
    if runtime_slots_per_gpu != configured_slots:
        raise ValueError(
            "--runtime-slots-per-gpu must match execution.parallel_cells_per_gpu"
        )
    if not 1 <= runtime_slots_per_gpu <= 4:
        raise ValueError("--runtime-slots-per-gpu must be in [1, 4]")
    slot_queue: LifoQueue[str] = LifoQueue()
    per_gpu_slot_assignment: dict[str, int] = {}
    for gpu in gpu_pool:
        per_gpu_slot_assignment[gpu] = runtime_slots_per_gpu
        for _ in range(runtime_slots_per_gpu):
            slot_queue.put(gpu)
    total_runtime_slots = len(gpu_pool) * runtime_slots_per_gpu
    atomic_json(
        work_dir / "RUNTIME_SLOTS.json",
        {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "registration_state": str(config["registration_state"]),
            "scope": str(config["execution"]["runtime_scope"]),
            "runtime_slots_per_gpu": runtime_slots_per_gpu,
            "gpu_count": len(gpu_pool),
            "gpu_pool": gpu_pool,
            "total_runtime_slots": total_runtime_slots,
            "per_gpu_slot_assignment": per_gpu_slot_assignment,
            "scientific_matrix_changed": False,
        },
    )
    results_lock = threading.Lock()
    results: dict[str, Any] = {}

    def take_gpu() -> str:
        return slot_queue.get()

    def return_gpu(gpu: str) -> None:
        slot_queue.put(gpu)

    def run_one(cell: Cell) -> tuple[str, int, str]:
        gpu = take_gpu()
        output_dir = methods_dir / cell.name
        if output_dir.exists() and not (output_dir / "summary.json").exists():
            shutil.rmtree(output_dir)
        log_path = logs_dir / f"{cell.name}.log"
        try:
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            environment["LOCAL_RANK"] = "0"
            command = _worker_command(args, cell, output_dir)
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
            return cell.name, returncode, str(log_path)
        finally:
            return_gpu(gpu)

    with ThreadPoolExecutor(max_workers=total_runtime_slots) as executor:
        futures = {executor.submit(run_one, cell): cell for cell in pending}
        for future in as_completed(futures):
            name, returncode, log_path = future.result()
            with results_lock:
                results[name] = {
                    "returncode": returncode,
                    "status": "OK" if returncode == 0 else "FAIL",
                    "log": log_path,
                }
                atomic_json(
                    work_dir / "SWEEP_STATUS.json",
                    {
                        "experiment_id": EXPERIMENT_ID,
                        "expected_cells": len(cells),
                        "completed_this_invocation": len(results),
                        "remaining_after_invocation": len(pending) - len(results),
                        "results": dict(sorted(results.items())),
                    },
                )
            print(f"[{name}] returncode={returncode} log={log_path}", flush=True)

    audit = _aggregate(
        work_dir,
        cells,
        registration_state=str(config["registration_state"]),
    )
    failed = [name for name, result in results.items() if result["returncode"] != 0]
    complete = audit["all_expected_cells_present"] and not failed
    atomic_json(
        work_dir / "SWEEP_COMPLETE.json",
        {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "version": VERSION,
            "registration_state": str(config["registration_state"]),
            "expected_cells": len(cells),
            "summary_count": audit["summary_count"],
            "failed_cells": sorted(failed),
            "all_expected_cells_present": complete,
            "source": git_state(repo),
            "test_data_used": False,
            "terminal_audit": str(work_dir / "aggregate" / "terminal_audit.json"),
        },
    )
    return 0 if complete else 1


def worker(args: argparse.Namespace) -> int:
    cell = Cell(
        alpha=float(args.alpha), c=float(args.c), seed_offset=int(args.seed_offset)
    )
    config = load_yaml(args.grid_config)
    if (cell.alpha, cell.c) not in parameter_points(config):
        raise ValueError(f"Worker cell is outside the frozen grid: {cell}")
    summary = train_cell(
        cell=cell,
        model_path=Path(args.model_path).resolve(),
        bank=Path(args.bank).resolve(),
        val=Path(args.val).resolve(),
        base_config_path=Path(args.base_config).resolve(),
        grid_config_path=Path(args.grid_config).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        repo=Path(__file__).resolve().parents[2],
        smoke=False,
    )
    return 0 if summary.get("numerical_failure") is None else 1


def parser() -> argparse.ArgumentParser:
    command_parser = argparse.ArgumentParser(description=__doc__)
    command_parser.add_argument("--version", action="version", version=VERSION)
    subparsers = command_parser.add_subparsers(dest="command", required=True)

    def common(
        subparser: argparse.ArgumentParser, *, include_work_dir: bool = True
    ) -> None:
        subparser.add_argument("--model_path", required=True)
        subparser.add_argument("--bank", required=True)
        subparser.add_argument("--val", required=True)
        subparser.add_argument("--base_config", required=True)
        subparser.add_argument("--grid_config", required=True)
        if include_work_dir:
            subparser.add_argument("--work_dir", required=True)

    plan_parser = subparsers.add_parser("plan")
    common(plan_parser)

    smoke_parser = subparsers.add_parser("smoke")
    common(smoke_parser)

    run_parser = subparsers.add_parser("run")
    common(run_parser)
    run_parser.add_argument("--gpus", required=True)
    run_parser.add_argument(
        "--runtime-slots-per-gpu",
        type=int,
        default=2,
        help="parent-controller runtime scheduling slots per GPU; scientific matrix unchanged",
    )

    worker_parser = subparsers.add_parser("worker")
    common(worker_parser, include_work_dir=False)
    worker_parser.add_argument("--output_dir", required=True)
    worker_parser.add_argument("--alpha", type=float, required=True)
    worker_parser.add_argument("--c", type=float, required=True)
    worker_parser.add_argument("--seed_offset", type=int, required=True)
    return command_parser


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "plan":
        return plan(args)
    if args.command == "smoke":
        return smoke(args)
    if args.command == "run":
        return run(args)
    if args.command == "worker":
        return worker(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
