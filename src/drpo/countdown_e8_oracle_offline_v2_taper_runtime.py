#!/usr/bin/env python3
"""Hardened runtime for the E8 V2 Linear/Quadratic/Exp sweep.

The companion ``countdown_e8_oracle_offline_v2_taper_sweep`` module owns the
registered formulas and training core.  This runtime adds the production
orchestration invariants required for the eight-GPU pilot:

* near-median/far-median remoteness anchoring (u=0 / u=1);
* calibration identity tied to both source files and all frozen inputs;
* initial negative-gradient budget matching to Global x1/32;
* clean restart of partial cells and identity-checked completed-cell resume;
* deferred post-hoc evaluation after the training model has left scope, avoiding
  two Qwen models resident on one GPU;
* exactly eight visible GPUs, one worker per GPU, and structured sweep status.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

try:
    from drpo import countdown_e8_oracle_offline_v2_taper_sweep as core
except ImportError:  # pragma: no cover - direct execution from src/drpo
    import countdown_e8_oracle_offline_v2_taper_sweep as core  # type: ignore

EXPERIMENT_ID = core.EXPERIMENT_ID
VERSION = "0.2.0-runtime"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(target)


def source_identity(repo: Path) -> dict[str, Any]:
    state = core._git_state(repo)
    return {
        "commit": state.get("commit"),
        "dirty": bool(state.get("dirty")),
        "runtime_source_sha256": sha256_file(__file__),
        "core_source_sha256": sha256_file(core.__file__),
    }


def remoteness_anchors(
    near_surprisals: Sequence[float], far_surprisals: Sequence[float]
) -> tuple[float, float, float, float]:
    if not near_surprisals or not far_surprisals:
        raise ValueError("Near and far surprisal samples must be non-empty")
    values = [float(value) for value in (*near_surprisals, *far_surprisals)]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Remoteness calibration surprisals must be finite")
    near_median = float(np.median(np.asarray(near_surprisals, dtype=float)))
    far_median = float(np.median(np.asarray(far_surprisals, dtype=float)))
    scale = far_median - near_median
    return near_median, scale, near_median, far_median


def calibration_identity(
    *,
    repo: Path,
    model_path: Path,
    bank_path: Path,
    global_calibration_path: Path,
    base_config_path: Path,
    sweep_config_path: Path,
) -> dict[str, Any]:
    return {
        "source": source_identity(repo),
        "model_path": str(model_path.resolve()),
        "bank_sha256": sha256_file(bank_path),
        "global_calibration_sha256": sha256_file(global_calibration_path),
        "base_config_sha256": sha256_file(base_config_path),
        "sweep_config_sha256": sha256_file(sweep_config_path),
    }


def calibration_matches(current: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    return (
        current.get("experiment_id") == EXPERIMENT_ID
        and current.get("runtime_version") == VERSION
        and current.get("identity") == expected
    )


def calibrate(
    *,
    model_path: Path,
    bank_path: Path,
    global_calibration_path: Path,
    base_config_path: Path,
    sweep_config_path: Path,
    output_path: Path,
    repo: Path,
) -> dict[str, Any]:
    sweep_config = core.load_yaml(sweep_config_path)
    core.validate_sweep_config(sweep_config)
    base_config = core.load_yaml(base_config_path)
    calibration_cfg = sweep_config["calibration"]
    core.arena.seed_all(int(calibration_cfg["seed"]))

    global_calibration = json.loads(global_calibration_path.read_text())
    required = (
        "positive_rms_gradient_norm",
        "bank_uncontrolled_rms_gradient_norm",
        "bank_negative_scale",
        "bank_global_gamma",
    )
    missing = [key for key in required if key not in global_calibration]
    if missing:
        raise RuntimeError(f"Global calibration is missing keys: {missing}")
    reference_multiplier = float(calibration_cfg["reference_global_multiplier"])
    target_positive = (
        float(global_calibration["positive_rms_gradient_norm"])
        * float(global_calibration["bank_global_gamma"])
        * reference_multiplier
    )
    target_uncontrolled = (
        float(global_calibration["bank_uncontrolled_rms_gradient_norm"])
        * float(global_calibration["bank_negative_scale"])
        * float(global_calibration["bank_global_gamma"])
        * reference_multiplier
    )
    if not math.isclose(target_positive, target_uncontrolled, rel_tol=1e-5):
        raise RuntimeError("Global x1/32 target has inconsistent calibration identities")
    target_rms = target_positive

    tokenizer = core.arena.load_tokenizer(str(model_path))
    model = core.arena.load_model(
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
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    rows = core.arena.read_jsonl(bank_path)
    remoteness_rows = core._balanced_rows(
        rows, int(calibration_cfg["remoteness_prompt_rows"])
    )

    near_surprisals: list[float] = []
    far_surprisals: list[float] = []
    with torch.no_grad():
        for row in remoteness_rows:
            packed, bank_size = core._single_row_bank_batches(
                row, tokenizer, int(base_config["model"]["max_length"])
            )
            stats = core.arena.completion_stats(
                model, core.arena.move_to_device(packed["bank"], device)
            )
            near, far, _, _ = core.arena.select_current_bank_extremes(
                stats, 1, bank_size
            )
            near_surprisals.append(float(-near["seq_lp"].item()))
            far_surprisals.append(float(-far["seq_lp"].item()))

    tau, scale, near_median, far_median = remoteness_anchors(
        near_surprisals, far_surprisals
    )
    minimum_scale = float(calibration_cfg["minimum_surprisal_scale"])
    if not math.isfinite(scale) or scale < minimum_scale:
        raise RuntimeError(
            f"Degenerate V2 near/far remoteness scale: {scale} < {minimum_scale}"
        )

    geometry_rows = remoteness_rows[: int(calibration_cfg["gradient_prompt_rows"])]
    geometries: list[dict[str, float]] = []
    for row in geometry_rows:
        packed, bank_size = core._single_row_bank_batches(
            row, tokenizer, int(base_config["model"]["max_length"])
        )
        bank_batch = core.arena.move_to_device(packed["bank"], device)
        near_batch, far_batch, _, _ = core.arena.current_bank_training_batches(
            model, bank_batch, 1, bank_size
        )
        model.zero_grad(set_to_none=True)
        near_stats = core.arena.completion_stats(model, near_batch)
        near_lp = near_stats["seq_lp"].mean()
        near_grads = torch.autograd.grad(near_lp, trainable, allow_unused=True)
        near_surprisal = float(-near_lp.detach())

        model.zero_grad(set_to_none=True)
        far_stats = core.arena.completion_stats(model, far_batch)
        far_lp = far_stats["seq_lp"].mean()
        far_grads = torch.autograd.grad(far_lp, trainable, allow_unused=True)
        far_surprisal = float(-far_lp.detach())
        near_sq, far_sq, dot = core._gradient_geometry(near_grads, far_grads)
        geometries.append(
            {
                "near_surprisal": near_surprisal,
                "far_surprisal": far_surprisal,
                "near_sq": near_sq,
                "far_sq": far_sq,
                "dot": dot,
            }
        )
        model.zero_grad(set_to_none=True)

    methods: dict[str, Any] = {}
    for method in core.METHODS:
        for rho_value in sweep_config["sweep"]["rho_values"]:
            rho = float(rho_value)
            coefficient = core.coefficient_from_rho(method, rho)
            per_row_norms: list[float] = []
            near_weights: list[float] = []
            far_weights: list[float] = []
            for row in geometries:
                near_seq_lp = torch.tensor([-row["near_surprisal"]])
                far_seq_lp = torch.tensor([-row["far_surprisal"]])
                near_distance = core.normalized_distance(
                    near_seq_lp, tau=tau, surprisal_scale=scale
                )
                far_distance = core.normalized_distance(
                    far_seq_lp, tau=tau, surprisal_scale=scale
                )
                near_weight = float(
                    core.taper_weight(method, near_distance, coefficient).item()
                )
                far_weight = float(
                    core.taper_weight(method, far_distance, coefficient).item()
                )
                near_factor = 0.5 * near_weight
                far_factor = 0.5 * far_weight
                norm_sq = (
                    near_factor * near_factor * row["near_sq"]
                    + far_factor * far_factor * row["far_sq"]
                    + 2.0 * near_factor * far_factor * row["dot"]
                )
                per_row_norms.append(math.sqrt(max(norm_sq, 0.0)))
                near_weights.append(near_weight)
                far_weights.append(far_weight)
            unscaled_rms = float(np.sqrt(np.mean(np.square(per_row_norms))))
            if not math.isfinite(unscaled_rms) or unscaled_rms <= 0:
                raise RuntimeError(
                    f"Invalid weighted negative RMS for {method}, rho={rho}"
                )
            negative_scale = target_rms / unscaled_rms
            if not math.isfinite(negative_scale) or negative_scale <= 0:
                raise RuntimeError(
                    f"Invalid negative scale for {method}, rho={rho}"
                )
            methods[core._method_key(method, rho)] = {
                "method": method,
                "rho": rho,
                "coefficient": coefficient,
                "unscaled_weighted_negative_rms": unscaled_rms,
                "target_negative_rms": target_rms,
                "negative_scale": negative_scale,
                "mean_near_weight": float(np.mean(near_weights)),
                "mean_far_weight": float(np.mean(far_weights)),
                "mean_far_over_near_weight": float(
                    np.mean(
                        np.asarray(far_weights)
                        / np.maximum(np.asarray(near_weights), 1e-30)
                    )
                ),
            }

    result = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "runtime_version": VERSION,
        "identity": calibration_identity(
            repo=repo,
            model_path=model_path,
            bank_path=bank_path,
            global_calibration_path=global_calibration_path,
            base_config_path=base_config_path,
            sweep_config_path=sweep_config_path,
        ),
        "reference_global": {
            "multiplier": reference_multiplier,
            "target_negative_rms": target_rms,
            "rerun_in_sweep": False,
        },
        "remoteness": {
            "definition": "u=sqrt(relu(sequence_surprisal-tau)/scale)",
            "anchor": "near_median_u0_far_median_u1",
            "tau": tau,
            "scale": scale,
            "near_median_surprisal": near_median,
            "far_median_surprisal": far_median,
            "rows": len(remoteness_rows),
        },
        "gradient_rows": len(geometries),
        "methods": methods,
        "task_metrics_used": False,
        "test_data_used": False,
        "frozen_before_training": True,
    }
    atomic_json(output_path, result)
    return result


def real_posthoc_evaluation(
    *,
    model_path: Path,
    checkpoint: Path,
    bank: Path,
    val: Path,
    test: Path,
    base_config: Mapping[str, Any],
    seed_offset: int,
) -> dict[str, Any]:
    return core.base_runner.evaluate_adapter_checkpoint(
        model_path,
        checkpoint,
        {
            "train": bank,
            "validation": val,
            "test": test,
            "split_manifest": bank,
        },
        base_config,
        seed_offset=seed_offset,
    )


def worker(args: argparse.Namespace) -> int:
    repo = Path(__file__).resolve().parents[2]
    cell = core.Cell(args.method, float(args.rho), int(args.seed_offset))
    output_dir = Path(args.output_dir).resolve()
    summary_path = output_dir / "summary.json"
    if output_dir.exists() and not summary_path.exists():
        shutil.rmtree(output_dir)

    original_evaluator = core.base_runner.evaluate_adapter_checkpoint

    def deferred_evaluator(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {"deferred_posthoc_evaluation": True}

    core.base_runner.evaluate_adapter_checkpoint = deferred_evaluator
    try:
        summary = core.train_cell(
            cell=cell,
            model_path=Path(args.model_path).resolve(),
            bank=Path(args.bank).resolve(),
            val=Path(args.val).resolve(),
            test=Path(args.test).resolve(),
            base_config_path=Path(args.base_config).resolve(),
            sweep_config_path=Path(args.sweep_config).resolve(),
            calibration_path=Path(args.calibration).resolve(),
            output_dir=output_dir,
            repo=repo,
        )
    finally:
        core.base_runner.evaluate_adapter_checkpoint = original_evaluator

    if not summary.get("best_evaluation", {}).get("deferred_posthoc_evaluation"):
        return 0
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    base_config = core.load_yaml(args.base_config)
    terminal_kind = summary.get("terminal_checkpoint_kind")
    terminal_checkpoint = output_dir / (
        "last_finite_adapter" if terminal_kind == "last_finite" else "terminal_adapter"
    )
    best_evaluation = real_posthoc_evaluation(
        model_path=Path(args.model_path).resolve(),
        checkpoint=output_dir / "best_adapter",
        bank=Path(args.bank).resolve(),
        val=Path(args.val).resolve(),
        test=Path(args.test).resolve(),
        base_config=base_config,
        seed_offset=cell.seed_offset,
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    terminal_evaluation = real_posthoc_evaluation(
        model_path=Path(args.model_path).resolve(),
        checkpoint=terminal_checkpoint,
        bank=Path(args.bank).resolve(),
        val=Path(args.val).resolve(),
        test=Path(args.test).resolve(),
        base_config=base_config,
        seed_offset=cell.seed_offset,
    )
    summary["best_evaluation"] = best_evaluation
    summary["terminal_evaluation"] = terminal_evaluation
    summary["runtime_version"] = VERSION
    summary["runtime_source_sha256"] = sha256_file(__file__)
    summary["best_terminal_same_generation_seed"] = True
    atomic_json(summary_path, summary)
    return 0


def worker_command(
    *, args: argparse.Namespace, cell: core.Cell, output_dir: Path, calibration: Path
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
        "--test",
        args.test,
        "--base_config",
        args.base_config,
        "--sweep_config",
        args.sweep_config,
        "--calibration",
        str(calibration),
        "--output_dir",
        str(output_dir),
        "--method",
        cell.method,
        "--rho",
        str(cell.rho),
        "--seed_offset",
        str(cell.seed_offset),
    ]


def run(args: argparse.Namespace) -> int:
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
    ):
        if not required_path.exists():
            raise SystemExit(f"Missing required input: {required_path}")
    config = core.load_yaml(sweep_config)
    core.validate_sweep_config(config)
    gpu_pool = [item.strip() for item in args.gpus.split(",") if item.strip()]
    required_gpus = int(config["execution"]["required_gpu_count"])
    if len(gpu_pool) != required_gpus or len(set(gpu_pool)) != required_gpus:
        raise SystemExit(
            f"Expected exactly {required_gpus} unique GPU ids, received {gpu_pool}"
        )
    visible_count = torch.cuda.device_count()
    if visible_count < required_gpus:
        raise SystemExit(
            f"Eight-GPU sweep requires {required_gpus} visible CUDA devices; "
            f"torch reports {visible_count}"
        )
    work_dir.mkdir(parents=True, exist_ok=True)
    methods_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    expected_calibration_identity = calibration_identity(
        repo=repo,
        model_path=model,
        bank_path=bank,
        global_calibration_path=global_calibration,
        base_config_path=base_config,
        sweep_config_path=sweep_config,
    )
    if calibration_path.exists():
        current = json.loads(calibration_path.read_text())
        if not calibration_matches(current, expected_calibration_identity):
            raise RuntimeError("Existing taper calibration is stale; use a new work_dir")
    else:
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = args.calibration_gpu
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
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

    cells = core.build_cells(config)
    plan = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "runtime_version": VERSION,
        "source": source_identity(repo),
        "gpu_pool": gpu_pool,
        "cell_count": len(cells),
        "methods": list(core.METHODS),
        "rho_values": [float(value) for value in config["sweep"]["rho_values"]],
        "seed_offsets": [int(value) for value in config["sweep"]["seed_offsets"]],
        "global_rerun": False,
        "cells": [cell.__dict__ | {"name": cell.name} for cell in cells],
    }
    atomic_json(work_dir / "SWEEP_PLAN.json", plan)

    pending: list[core.Cell] = []
    for cell in cells:
        summary_path = methods_dir / cell.name / "summary.json"
        if not summary_path.exists():
            pending.append(cell)
            continue
        summary = json.loads(summary_path.read_text())
        if summary.get("runtime_source_sha256") != sha256_file(__file__):
            raise RuntimeError(f"Stale runtime identity for completed cell {cell.name}")
        if summary.get("best_evaluation", {}).get("deferred_posthoc_evaluation"):
            pending.append(cell)

    results: dict[str, Any] = {}
    available = list(gpu_pool)
    available_lock = threading.Lock()
    results_lock = threading.Lock()

    def take_gpu() -> str:
        with available_lock:
            return available.pop(0)

    def give_gpu(gpu: str) -> None:
        with available_lock:
            available.append(gpu)

    def run_one(cell: core.Cell) -> tuple[str, int, str]:
        gpu = take_gpu()
        log_path = logs_dir / f"{cell.name}.log"
        try:
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            command = worker_command(
                args=args,
                cell=cell,
                output_dir=methods_dir / cell.name,
                calibration=calibration_path,
            )
            with log_path.open("w") as handle:
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
            give_gpu(gpu)

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

    summaries = list(methods_dir.glob("*/summary.json"))
    failed = sorted(
        name for name, result in results.items() if result["returncode"] != 0
    )
    complete = len(summaries) == len(cells) and not failed
    atomic_json(
        work_dir / "SWEEP_COMPLETE.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "runtime_version": VERSION,
            "result_status": config["result_status"],
            "expected_cells": len(cells),
            "summary_count": len(summaries),
            "failed_cells": failed,
            "all_expected_cells_present": complete,
            "source": source_identity(repo),
            "calibration_sha256": sha256_file(calibration_path),
        },
    )
    return 0 if complete else 1


def parser() -> argparse.ArgumentParser:
    command_parser = argparse.ArgumentParser(description="Hardened E8 V2 taper runtime")
    command_parser.add_argument("--version", action="version", version=VERSION)
    subparsers = command_parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--model_path", required=True)
    run_parser.add_argument("--work_dir", required=True)
    run_parser.add_argument("--bank", required=True)
    run_parser.add_argument("--val", required=True)
    run_parser.add_argument("--test", required=True)
    run_parser.add_argument("--global_calibration", required=True)
    run_parser.add_argument("--base_config", required=True)
    run_parser.add_argument("--sweep_config", required=True)
    run_parser.add_argument("--calibration_gpu", default="0")
    run_parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")

    calibrate_parser = subparsers.add_parser("calibrate")
    calibrate_parser.add_argument("--model_path", required=True)
    calibrate_parser.add_argument("--bank", required=True)
    calibrate_parser.add_argument("--global_calibration", required=True)
    calibrate_parser.add_argument("--base_config", required=True)
    calibrate_parser.add_argument("--sweep_config", required=True)
    calibrate_parser.add_argument("--output", required=True)

    worker_parser = subparsers.add_parser("worker")
    worker_parser.add_argument("--model_path", required=True)
    worker_parser.add_argument("--bank", required=True)
    worker_parser.add_argument("--val", required=True)
    worker_parser.add_argument("--test", required=True)
    worker_parser.add_argument("--base_config", required=True)
    worker_parser.add_argument("--sweep_config", required=True)
    worker_parser.add_argument("--calibration", required=True)
    worker_parser.add_argument("--output_dir", required=True)
    worker_parser.add_argument("--method", choices=core.METHODS, required=True)
    worker_parser.add_argument("--rho", type=float, required=True)
    worker_parser.add_argument("--seed_offset", type=int, required=True)
    return command_parser


def main() -> int:
    args = parser().parse_args()
    repo = Path(__file__).resolve().parents[2]
    if args.command == "run":
        return run(args)
    if args.command == "calibrate":
        calibrate(
            model_path=Path(args.model_path).resolve(),
            bank_path=Path(args.bank).resolve(),
            global_calibration_path=Path(args.global_calibration).resolve(),
            base_config_path=Path(args.base_config).resolve(),
            sweep_config_path=Path(args.sweep_config).resolve(),
            output_path=Path(args.output).resolve(),
            repo=repo,
        )
        return 0
    if args.command == "worker":
        return worker(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
