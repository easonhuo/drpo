#!/usr/bin/env python3
"""Runtime orchestration for paper-aligned Countdown E8 lambda round 1."""

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
from typing import Any, Sequence

import numpy as np
import torch

from drpo.countdown_e8_paper_aligned_lambda_common import (
    EXPECTED_POINTS,
    EXPERIMENT_ID,
    FIXED_ALPHA,
    LAMBDA_VALUES,
    SEED_OFFSETS,
    VERSION,
    Cell,
    ContinuousUniqueBankDataset,
    _identity,
    _identity_equal,
    arena,
    atomic_json,
    build_cells,
    calibration_from_surprisals,
    git_state,
    load_yaml,
    make_continuous_unique_bank_collator,
    parameter_points,
    sha256_file,
    validate_grid_config,
)
from drpo.countdown_e8_paper_aligned_lambda_trainer import train_cell


def _calibration_path(work_dir: Path) -> Path:
    return work_dir / "TAPER_CALIBRATION.json"


def _worker_command(
    args: argparse.Namespace, cell: Cell, output_dir: Path, calibration: Path
) -> list[str]:
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
        "--calibration",
        str(calibration),
        "--output_dir",
        str(output_dir),
        "--alpha",
        str(cell.alpha),
        "--lambda_value",
        str(cell.lambda_value),
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
        "registration_state": "registered_pilot",
        "source": git_state(repo),
        "model_path": str(model),
        "bank_sha256": sha256_file(bank),
        "validation_sha256": sha256_file(val),
        "base_config_sha256": sha256_file(base_config),
        "grid_config_sha256": sha256_file(grid_config),
        "formula": "alpha*exp(-lambda*relu((D-tau)/scale_c))",
        "D_definition": "negative_mean_completion_token_log_probability_with_eos",
        "fixed_alpha": FIXED_ALPHA,
        "lambda_values": list(LAMBDA_VALUES),
        "parameter_points": EXPECTED_POINTS,
        "cell_count": len(cells),
        "positive_only_cells": sum(cell.method == "positive_only" for cell in cells),
        "global_rerun": False,
        "test_data_used": False,
        "cells": [
            {
                "name": cell.name,
                "method": cell.method,
                "alpha": cell.alpha,
                "lambda": cell.lambda_value,
                "seed_offset": cell.seed_offset,
            }
            for cell in cells
        ],
    }
    output = Path(args.work_dir).resolve() / "SWEEP_PLAN.json"
    atomic_json(output, payload)
    print(json.dumps({"plan": str(output), "cells": len(cells)}), flush=True)
    return 0


def calibrate(args: argparse.Namespace) -> int:
    repo = Path(__file__).resolve().parents[2]
    model_path, bank_path, _, base_config_path, grid_config_path = _required_inputs(args)
    config = load_yaml(grid_config_path)
    validate_grid_config(config)
    base_config = load_yaml(base_config_path)
    calibration_cfg = config["calibration"]
    work_dir = Path(args.work_dir).resolve()
    output = _calibration_path(work_dir)
    expected_identity = {
        "experiment_id": EXPERIMENT_ID,
        "version": VERSION,
        "source": git_state(repo),
        "model_path": str(model_path),
        "bank_sha256": sha256_file(bank_path),
        "base_config_sha256": sha256_file(base_config_path),
        "grid_config_sha256": sha256_file(grid_config_path),
        "seed": int(calibration_cfg["seed"]),
    }
    if output.is_file():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing.get("identity") == expected_identity:
            print(json.dumps({"calibration": str(output), "reused": True}), flush=True)
            return 0
        raise RuntimeError("Stale calibration identity; use a new work_dir")

    arena.seed_all(int(calibration_cfg["seed"]))
    tokenizer = arena.load_tokenizer(str(model_path))
    model = arena.load_model(
        str(model_path),
        adapter_path=None,
        trainable_adapter=True,
        load_in_4bit=bool(base_config["model"].get("load_in_4bit", False)),
        dtype=str(base_config["model"].get("dtype", "auto")),
        gradient_checkpointing=False,
        parameterization="lora",
    )
    model.eval()
    device = next(model.parameters()).device
    rows = arena.read_jsonl(bank_path)
    selected = list(arena.balanced_diagnostic_rows(rows, int(calibration_cfg["prompt_rows"])))
    if len(selected) != int(calibration_cfg["prompt_rows"]):
        raise RuntimeError("Calibration row count is incomplete")
    dataset = ContinuousUniqueBankDataset(
        selected, tokenizer, int(base_config["model"]["max_length"])
    )
    collate = make_continuous_unique_bank_collator(tokenizer.pad_token_id)
    surprisals: list[float] = []
    with torch.no_grad():
        for index in range(len(dataset)):
            packed = collate([dataset[index]])
            stats = arena.completion_stats(model, arena.move_to_device(packed["bank"], device))
            surprisals.extend(float(value) for value in (-stats["seq_lp"]).float().cpu().tolist())
    summary = calibration_from_surprisals(
        surprisals,
        minimum_scale=float(calibration_cfg["minimum_surprisal_scale"]),
        minimum_active_fraction=float(calibration_cfg["minimum_active_fraction"]),
    )
    payload = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "version": VERSION,
        "identity": expected_identity,
        "formula": "alpha*exp(-lambda*relu((D-tau)/scale_c))",
        "D_definition": "negative_mean_completion_token_log_probability_with_eos",
        "tau_rule": calibration_cfg["tau_rule"],
        "scale_rule": calibration_cfg["scale_rule"],
        "calibration_prompt_rows": len(selected),
        "calibration_row_ids": [str(row.get("id", index)) for index, row in enumerate(selected)],
        "training_bank_reuse": True,
        "task_metrics_used": False,
        "test_data_used": False,
        **summary,
    }
    atomic_json(output, payload)
    print(
        json.dumps(
            {
                "calibration": str(output),
                "tau": payload["tau"],
                "scale_c": payload["scale_c"],
                "active_fraction": payload["active_fraction"],
            }
        ),
        flush=True,
    )
    return 0


def smoke(args: argparse.Namespace) -> int:
    repo = Path(__file__).resolve().parents[2]
    model, bank, val, base_config, grid_config = _required_inputs(args)
    config = load_yaml(grid_config)
    validate_grid_config(config)
    work_dir = Path(args.work_dir).resolve()
    calibration = _calibration_path(work_dir)
    if not calibration.is_file():
        raise RuntimeError("TAPER_CALIBRATION.json is required before liveness")
    liveness = config["execution"]["liveness"]
    cell = Cell(
        alpha=float(liveness["representative_alpha"]),
        lambda_value=float(liveness["representative_lambda"]),
        seed_offset=SEED_OFFSETS[0],
    )
    output_dir = work_dir / "_liveness" / cell.name
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
            calibration_path=calibration,
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
            "status": "PASS" if passed else "FAIL",
            "scientific_evidence": False,
            "cell": cell.name,
            "summary": str(output_dir / "summary.json"),
            "calibration_sha256": sha256_file(calibration),
            "run_identity": summary.get("run_identity"),
            "test_data_used": False,
        }
    except BaseException as error:
        gate = {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "status": "FAIL",
            "scientific_evidence": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "test_data_used": False,
        }
        atomic_json(work_dir / "SMOKE_GATE.json", gate)
        raise
    atomic_json(work_dir / "SMOKE_GATE.json", gate)
    return 0 if gate["status"] == "PASS" else 1


def _mean(values: list[float | None]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    return float(np.mean(numeric)) if numeric else None


def _aggregate(work_dir: Path, cells: Sequence[Cell], calibration: Path) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    for cell in cells:
        path = work_dir / "methods" / cell.name / "summary.json"
        if path.is_file():
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        terminal = summary.get("terminal_metrics", {})
        late = summary.get("late_window_metrics", {})
        bests = summary.get("metric_bests", {})
        rows.append(
            {
                "cell": summary["cell"],
                "method": summary["method"],
                "alpha": summary["alpha"],
                "lambda": summary["lambda"],
                "effective_beta": summary["effective_beta"],
                "seed_offset": summary["seed_offset"],
                "best_greedy": (bests.get("val_greedy") or {}).get("value"),
                "best_pass_at_8": (bests.get("val_pass_at_8") or {}).get("value"),
                "best_pass_at_64": (bests.get("val_pass_at_64") or {}).get("value"),
                "best_valid_rate": (bests.get("val_valid_rate") or {}).get("value"),
                "late_greedy": late.get("val_greedy"),
                "late_pass_at_8": late.get("val_pass_at_8"),
                "late_pass_at_64": late.get("val_pass_at_64"),
                "late_valid_rate": late.get("val_valid_rate"),
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

    grouped: list[dict[str, Any]] = []
    for alpha, lambda_value in parameter_points(load_yaml(work_dir / "FROZEN_CONFIG.yaml")):
        subset = [
            row
            for row in rows
            if float(row["alpha"]) == alpha and float(row["lambda"]) == lambda_value
        ]
        if not subset:
            continue
        grouped.append(
            {
                "method": "positive_only" if alpha == 0.0 else "paper_aligned_exp",
                "alpha": alpha,
                "lambda": lambda_value,
                "n": len(subset),
                "best_pass_at_8_mean": _mean([row["best_pass_at_8"] for row in subset]),
                "late_pass_at_8_mean": _mean([row["late_pass_at_8"] for row in subset]),
                "terminal_pass_at_8_mean": _mean([row["terminal_pass_at_8"] for row in subset]),
                "best_pass_at_64_mean": _mean([row["best_pass_at_64"] for row in subset]),
                "terminal_valid_rate_mean": _mean([row["terminal_valid_rate"] for row in subset]),
            }
        )
    if grouped:
        with (aggregate_dir / "lambda_summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(grouped[0].keys()))
            writer.writeheader()
            writer.writerows(grouped)

    taper_rows = [row for row in grouped if row["method"] == "paper_aligned_exp"]
    positive = next((row for row in grouped if row["method"] == "positive_only"), None)
    decision: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "automatic_scientific_conclusion_forbidden": True,
        "historical_trend_invalidated": False,
        "test_data_used": False,
    }
    if len(taper_rows) == len(LAMBDA_VALUES):
        best_index = max(
            range(len(taper_rows)),
            key=lambda index: float(taper_rows[index]["best_pass_at_8_mean"]),
        )
        if best_index == 0:
            next_action = "extend_lambda_left_before_interpretation"
        elif best_index == len(taper_rows) - 1:
            next_action = "extend_lambda_right_before_interpretation"
        else:
            next_action = "locally_refine_around_internal_peak"
        decision.update(
            {
                "status": "ROUND1_COMPLETE",
                "best_grid_lambda": taper_rows[best_index]["lambda"],
                "best_grid_pass_at_8_mean": taper_rows[best_index]["best_pass_at_8_mean"],
                "positive_only_best_pass_at_8_mean": (
                    positive["best_pass_at_8_mean"] if positive else None
                ),
                "positive_only_late_pass_at_8_mean": (
                    positive["late_pass_at_8_mean"] if positive else None
                ),
                "positive_only_terminal_pass_at_8_mean": (
                    positive["terminal_pass_at_8_mean"] if positive else None
                ),
                "next_action": next_action,
                "terminal_reversal_check_required": True,
                "large_lambda_endpoint": "near_field_only_not_positive_only",
                "fresh_seed_confirmation_required": True,
            }
        )
    else:
        decision.update({"status": "INCOMPLETE", "next_action": "finish_or_repair_round1"})
    atomic_json(aggregate_dir / "ROUND1_DECISION.json", decision)

    numerical_failures = sum(row["numerical_failure"] is not None for row in rows)
    complete = len(rows) == len(cells)
    audit = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "registration_state": "registered_pilot",
        "expected_cells": len(cells),
        "summary_count": len(rows),
        "all_expected_cells_present": complete,
        "calibration_sha256": sha256_file(calibration),
        "numerical_failures": numerical_failures,
        "task_performance_status": "reported_not_adjudicated",
        "support_or_structure_boundary_status": "valid_rate_only_not_formal_boundary",
        "nan_inf_status": "observed" if numerical_failures else "not_observed",
        "test_data_used": False,
        "fixed_1200_steps_is_convergence": False,
        "method_ranking_claim_allowed": False,
        "status": "PASS" if complete and numerical_failures == 0 else "INCOMPLETE_OR_FAILURE",
    }
    atomic_json(aggregate_dir / "terminal_audit.json", audit)
    return audit


def run(args: argparse.Namespace) -> int:
    repo = Path(__file__).resolve().parents[2]
    model, bank, val, base_config, grid_config = _required_inputs(args)
    config = load_yaml(grid_config)
    cells = build_cells(config)
    work_dir = Path(args.work_dir).resolve()
    calibration = _calibration_path(work_dir)
    if not calibration.is_file():
        raise RuntimeError("TAPER_CALIBRATION.json is required")
    smoke_gate = work_dir / "SMOKE_GATE.json"
    if not smoke_gate.is_file() or json.loads(smoke_gate.read_text()).get("status") != "PASS":
        raise RuntimeError("A passing representative SMOKE_GATE.json is required")
    gpu_pool = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpu_pool or len(set(gpu_pool)) != len(gpu_pool):
        raise ValueError("At least one unique GPU id is required")
    methods_dir = work_dir / "methods"
    logs_dir = work_dir / "logs"
    methods_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(grid_config, work_dir / "FROZEN_CONFIG.yaml")
    plan(args)

    pending: list[Cell] = []
    for cell in cells:
        summary_path = methods_dir / cell.name / "summary.json"
        if not summary_path.exists():
            pending.append(cell)
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        expected_identity = _identity(
            repo=repo,
            model_path=model,
            bank=bank,
            val=val,
            base_config=base_config,
            grid_config=grid_config,
            calibration=calibration,
            cell=cell,
            smoke=False,
        )
        if not _identity_equal(summary.get("run_identity", {}), expected_identity):
            raise RuntimeError(f"Stale completed-cell identity: {cell.name}")

    available = list(gpu_pool)
    available_lock = threading.Lock()
    results_lock = threading.Lock()
    results: dict[str, Any] = {}

    def take_gpu() -> str:
        with available_lock:
            return available.pop(0)

    def return_gpu(gpu: str) -> None:
        with available_lock:
            available.append(gpu)

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
            command = _worker_command(args, cell, output_dir, calibration)
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

    with ThreadPoolExecutor(max_workers=len(gpu_pool)) as executor:
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

    audit = _aggregate(work_dir, cells, calibration)
    failed = [name for name, result in results.items() if result["returncode"] != 0]
    complete = audit["all_expected_cells_present"] and not failed
    atomic_json(
        work_dir / "SWEEP_COMPLETE.json",
        {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "version": VERSION,
            "registration_state": "registered_pilot",
            "expected_cells": len(cells),
            "summary_count": audit["summary_count"],
            "failed_cells": sorted(failed),
            "all_expected_cells_present": complete,
            "source": git_state(repo),
            "test_data_used": False,
            "terminal_audit": str(work_dir / "aggregate" / "terminal_audit.json"),
            "round1_decision": str(work_dir / "aggregate" / "ROUND1_DECISION.json"),
        },
    )
    return 0 if complete else 1


def worker(args: argparse.Namespace) -> int:
    cell = Cell(
        alpha=float(args.alpha),
        lambda_value=float(args.lambda_value),
        seed_offset=int(args.seed_offset),
    )
    config = load_yaml(args.grid_config)
    if (cell.alpha, cell.lambda_value) not in parameter_points(config):
        raise ValueError(f"Worker cell is outside the frozen grid: {cell}")
    summary = train_cell(
        cell=cell,
        model_path=Path(args.model_path).resolve(),
        bank=Path(args.bank).resolve(),
        val=Path(args.val).resolve(),
        base_config_path=Path(args.base_config).resolve(),
        grid_config_path=Path(args.grid_config).resolve(),
        calibration_path=Path(args.calibration).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        repo=Path(__file__).resolve().parents[2],
        smoke=False,
    )
    return 0 if summary.get("numerical_failure") is None else 1


def parser() -> argparse.ArgumentParser:
    command_parser = argparse.ArgumentParser(description=__doc__)
    command_parser.add_argument("--version", action="version", version=VERSION)
    subparsers = command_parser.add_subparsers(dest="command", required=True)

    def common(subparser: argparse.ArgumentParser, *, include_work_dir: bool = True) -> None:
        subparser.add_argument("--model_path", required=True)
        subparser.add_argument("--bank", required=True)
        subparser.add_argument("--val", required=True)
        subparser.add_argument("--base_config", required=True)
        subparser.add_argument("--grid_config", required=True)
        if include_work_dir:
            subparser.add_argument("--work_dir", required=True)

    for command in ("plan", "calibrate", "smoke"):
        common(subparsers.add_parser(command))
    run_parser = subparsers.add_parser("run")
    common(run_parser)
    run_parser.add_argument("--gpus", required=True)

    worker_parser = subparsers.add_parser("worker")
    common(worker_parser, include_work_dir=False)
    worker_parser.add_argument("--calibration", required=True)
    worker_parser.add_argument("--output_dir", required=True)
    worker_parser.add_argument("--alpha", type=float, required=True)
    worker_parser.add_argument("--lambda_value", type=float, required=True)
    worker_parser.add_argument("--seed_offset", type=int, required=True)
    return command_parser


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "plan":
        return plan(args)
    if args.command == "calibrate":
        return calibrate(args)
    if args.command == "smoke":
        return smoke(args)
    if args.command == "run":
        return run(args)
    if args.command == "worker":
        return worker(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
