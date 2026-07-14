#!/usr/bin/env python3
"""Resource-equivalent phase probe for the frozen Countdown E8 taper workload.

This module is engineering-only. It preserves the real model, train micro-batch,
negative-bank shape, sequence limits, generation batch size, and maximum pass@k,
while reducing outer repetition to one train micro-batch and one evaluation batch.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from drpo import countdown_e8_oracle_offline_v2_taper_sweep as core
from drpo.runtime_gpu_placement_autotune import (
    DEFAULT_REQUIRED_PHASES,
    PROBE_CONTRACT_VERSION,
    PROBE_STATE_FILENAME,
)

VERSION = "0.1.0-phase-envelope"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _write_state(
    output_dir: Path,
    completed_phases: list[str],
    *,
    phase_details: dict[str, Any],
    complete: bool,
    error: str | None = None,
) -> None:
    _atomic_json(
        output_dir / PROBE_STATE_FILENAME,
        {
            "schema_version": 1,
            "contract_version": PROBE_CONTRACT_VERSION,
            "probe_version": VERSION,
            "required_phases": list(DEFAULT_REQUIRED_PHASES),
            "completed_phases": list(completed_phases),
            "phase_details": phase_details,
            "complete": bool(complete),
            "error": error,
            "scientific_matrix_changed": False,
        },
    )


def _mark_phase(
    output_dir: Path,
    completed_phases: list[str],
    phase: str,
    *,
    phase_details: dict[str, Any],
) -> None:
    if phase not in completed_phases:
        completed_phases.append(phase)
    _write_state(
        output_dir,
        completed_phases,
        phase_details=phase_details,
        complete=phase == "probe_complete",
    )
    print(json.dumps({"resource_probe_phase": phase}, sort_keys=True), flush=True)


def resource_probe_command(
    *,
    args: argparse.Namespace,
    cell: core.Cell,
    output_dir: Path,
    calibration: Path,
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--model_path",
        str(args.model_path),
        "--bank",
        str(args.bank),
        "--val",
        str(args.val),
        "--base_config",
        str(args.base_config),
        "--sweep_config",
        str(args.sweep_config),
        "--calibration",
        str(calibration),
        "--output_dir",
        str(output_dir),
        "--method",
        str(cell.method),
        "--rho",
        str(cell.rho),
        "--seed_offset",
        str(cell.seed_offset),
    ]


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    completed_phases: list[str] = []
    phase_details: dict[str, Any] = {}
    _write_state(
        output_dir,
        completed_phases,
        phase_details=phase_details,
        complete=False,
    )

    try:
        base_config = core.load_yaml(args.base_config)
        sweep_config = core.load_yaml(args.sweep_config)
        core.validate_sweep_config(sweep_config)
        calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
        cell = core.Cell(args.method, float(args.rho), int(args.seed_offset))
        calibration_row = calibration["methods"][core._method_key(cell.method, cell.rho)]
        tau = float(calibration["remoteness"]["tau"])
        scale = float(calibration["remoteness"]["scale"])
        negative_scale = float(calibration_row["negative_scale"])
        coefficient = float(calibration_row["coefficient"])
        train_cfg = base_config["offline_training"]
        model_cfg = base_config["model"]
        eval_cfg = base_config["evaluation"]
        seed = int(train_cfg["seed"]) + cell.seed_offset
        core.arena.seed_all(seed)

        tokenizer = core.arena.load_tokenizer(str(Path(args.model_path).resolve()))
        train_rows = core.arena.read_jsonl(args.bank)
        val_rows = core.arena.read_jsonl(args.val)
        dataset = core.arena.OfflineDataset(
            train_rows,
            tokenizer,
            int(model_cfg["max_length"]),
        )
        generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(
            dataset,
            batch_size=int(train_cfg["micro_batch"]),
            shuffle=True,
            generator=generator,
            collate_fn=core.arena.make_offline_collator(tokenizer.pad_token_id),
            num_workers=int(train_cfg["num_workers"]),
        )
        packed = next(iter(loader))

        model = core.arena.load_model(
            str(Path(args.model_path).resolve()),
            adapter_path=None,
            trainable_adapter=True,
            load_in_4bit=bool(model_cfg.get("load_in_4bit", False)),
            dtype=str(model_cfg.get("dtype", "auto")),
            gradient_checkpointing=True,
            parameterization="lora",
        )
        device = next(model.parameters()).device
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable,
            lr=float(train_cfg["learning_rate"]),
            weight_decay=0.01,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        phase_details["model"] = {
            "max_length": int(model_cfg["max_length"]),
            "max_new_tokens": int(model_cfg["max_new_tokens"]),
            "dtype": str(model_cfg.get("dtype", "auto")),
            "load_in_4bit": bool(model_cfg.get("load_in_4bit", False)),
            "gradient_checkpointing": True,
        }
        _mark_phase(
            output_dir,
            completed_phases,
            "model_loaded",
            phase_details=phase_details,
        )

        model.train()
        optimizer.zero_grad(set_to_none=True)
        positive = core.arena.completion_stats(
            model,
            core.arena.move_to_device(packed["positive"], device),
        )
        positive_lp = positive["seq_lp"].mean()
        bank_size = int(packed.get("bank_size", 0))
        if bank_size < 2 or "bank" not in packed:
            raise RuntimeError("resource probe requires the populated frozen negative bank")
        near_batch, far_batch, _, _ = core.arena.current_bank_training_batches(
            model,
            core.arena.move_to_device(packed["bank"], device),
            packed["positive"]["input_ids"].shape[0],
            bank_size,
        )
        near = core.arena.completion_stats(model, near_batch)
        far = core.arena.completion_stats(model, far_batch)
        near_distance = core.normalized_distance(
            near["seq_lp"],
            tau=tau,
            surprisal_scale=scale,
        )
        far_distance = core.normalized_distance(
            far["seq_lp"],
            tau=tau,
            surprisal_scale=scale,
        )
        near_weights = core.taper_weight(
            cell.method,
            near_distance,
            coefficient,
        ).detach()
        far_weights = core.taper_weight(
            cell.method,
            far_distance,
            coefficient,
        ).detach()
        negative_lp = 0.5 * (near_weights * near["seq_lp"]).mean() + 0.5 * (
            far_weights * far["seq_lp"]
        ).mean()
        raw_loss = -(positive_lp - negative_scale * negative_lp)
        if not bool(torch.isfinite(raw_loss)):
            raise RuntimeError("resource probe produced a non-finite training loss")
        raw_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable,
            float(train_cfg["maximum_gradient_norm"]),
        )
        if not bool(torch.isfinite(grad_norm)):
            raise RuntimeError("resource probe produced a non-finite gradient")
        optimizer.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        phase_details["training"] = {
            "micro_batch": int(train_cfg["micro_batch"]),
            "negative_bank_size": bank_size,
            "gradient_accumulation_repetitions": 1,
            "production_gradient_accumulation": int(train_cfg["gradient_accumulation"]),
            "loss_finite": True,
            "gradient_finite": True,
        }
        _mark_phase(
            output_dir,
            completed_phases,
            "training_peak_completed",
            phase_details=phase_details,
        )

        eval_batch_size = int(eval_cfg["batch_size"])
        pass_k = max(int(value) for value in eval_cfg["pass_ks"])
        if len(val_rows) < eval_batch_size:
            raise RuntimeError(
                "validation set is smaller than the production evaluation batch size"
            )
        probe_rows = val_rows[:eval_batch_size]
        known_structures = {
            row.get("oracle_structure") or core.arena.expression_structure(row["oracle"])
            for row in train_rows
        }
        model.eval()
        core.arena.evaluate_rows(
            model,
            tokenizer,
            probe_rows,
            eval_batch_size,
            int(model_cfg["max_new_tokens"]),
            pass_k,
            int(eval_cfg["seed"]),
            known_structures,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        phase_details["evaluation"] = {
            "outer_batches": 1,
            "examples": eval_batch_size,
            "batch_size": eval_batch_size,
            "pass_k": pass_k,
            "max_new_tokens": int(model_cfg["max_new_tokens"]),
            "preserves_production_inner_shape": True,
        }
        _mark_phase(
            output_dir,
            completed_phases,
            "evaluation_peak_completed",
            phase_details=phase_details,
        )
        _mark_phase(
            output_dir,
            completed_phases,
            "probe_complete",
            phase_details=phase_details,
        )
        return 0
    except Exception as exc:
        _write_state(
            output_dir,
            completed_phases,
            phase_details=phase_details,
            complete=False,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model_path", required=True)
    result.add_argument("--bank", required=True)
    result.add_argument("--val", required=True)
    result.add_argument("--base_config", required=True)
    result.add_argument("--sweep_config", required=True)
    result.add_argument("--calibration", required=True)
    result.add_argument("--output_dir", required=True)
    result.add_argument("--method", choices=core.METHODS, required=True)
    result.add_argument("--rho", type=float, required=True)
    result.add_argument("--seed_offset", type=int, required=True)
    return result


def main(argv: list[str] | None = None) -> int:
    return run(parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
