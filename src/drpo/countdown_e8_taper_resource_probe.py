#!/usr/bin/env python3
"""Resource-equivalent phase probe for the Countdown E8 taper workload.

This probe is engineering-only. It preserves the registered model, training
micro-batch, gradient-accumulation count, negative-bank shape, sequence limits,
evaluation batch size, maximum pass@k, and generation length. It reduces only
outer repetition and never writes a scientific result.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader

from drpo import countdown_e8_oracle_offline_v2_taper_sweep as core

PROBE_CONTRACT_VERSION = 2
REQUIRED_PHASES = (
    "model_loaded",
    "training_peak_completed",
    "evaluation_peak_completed",
    "probe_complete",
)
EVENT_FILENAME = "resource_probe_events.jsonl"


def _event_path(output_dir: Path) -> Path:
    configured = os.environ.get("DRPO_RUNTIME_RESOURCE_PROBE_EVENT_PATH")
    return Path(configured).resolve() if configured else output_dir / EVENT_FILENAME


def _cuda_memory() -> dict[str, int]:
    if not torch.cuda.is_available():
        return {
            "peak_vram_allocated_bytes": 0,
            "peak_vram_reserved_bytes": 0,
            "current_vram_allocated_bytes": 0,
            "current_vram_reserved_bytes": 0,
        }
    torch.cuda.synchronize()
    return {
        "peak_vram_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_vram_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "current_vram_allocated_bytes": int(torch.cuda.memory_allocated()),
        "current_vram_reserved_bytes": int(torch.cuda.memory_reserved()),
    }


def _reset_cuda_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def emit_phase(output_dir: Path, phase: str, **payload: Any) -> None:
    if phase not in (*REQUIRED_PHASES, "probe_failed"):
        raise ValueError(f"unsupported resource-probe phase: {phase}")
    target = _event_path(output_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "schema_version": 1,
        "probe_contract_version": PROBE_CONTRACT_VERSION,
        "phase": phase,
        "time_unix": time.time(),
        **_cuda_memory(),
        **payload,
    }
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def evaluation_envelope_rows(
    validation_rows: list[dict[str, Any]], base_config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    batch_size = int(base_config["evaluation"]["batch_size"])
    if batch_size < 1:
        raise ValueError("evaluation batch size must be positive")
    if len(validation_rows) < batch_size:
        raise RuntimeError(
            "validation split is too small for the registered evaluation batch shape"
        )
    return list(validation_rows[:batch_size])


def maximum_pass_k(base_config: Mapping[str, Any]) -> int:
    values = [int(value) for value in base_config["evaluation"]["pass_ks"]]
    if not values or min(values) < 1:
        raise ValueError("evaluation pass_ks must be non-empty and positive")
    return max(values)


def _load_inputs(args: argparse.Namespace) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    core.Cell,
]:
    base_config = core.load_yaml(args.base_config)
    sweep_config = core.load_yaml(args.sweep_config)
    core.validate_sweep_config(sweep_config)
    calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    cell = core.Cell(args.method, float(args.rho), int(args.seed_offset))
    calibration_row = calibration["methods"][core._method_key(cell.method, cell.rho)]
    return base_config, sweep_config, calibration_row, cell


def run_probe(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    event_path = _event_path(output_dir)
    event_path.unlink(missing_ok=True)

    base_config, _sweep_config, calibration_row, cell = _load_inputs(args)
    calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    model_cfg = base_config["model"]
    train_cfg = base_config["offline_training"]
    eval_cfg = base_config["evaluation"]
    seed = int(train_cfg["seed"]) + cell.seed_offset
    core.arena.seed_all(seed)

    tokenizer: Any | None = None
    model: Any | None = None
    optimizer: Any | None = None
    loader: Any | None = None
    iterator: Any | None = None
    adapter_dir = output_dir / "probe_adapter"

    try:
        tokenizer = core.arena.load_tokenizer(str(Path(args.model_path).resolve()))
        train_rows = core.arena.read_jsonl(args.bank)
        validation_rows = core.arena.read_jsonl(args.val)
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
        iterator = iter(loader)

        _reset_cuda_peak()
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
        trainable = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        optimizer = torch.optim.AdamW(
            trainable,
            lr=float(train_cfg["learning_rate"]),
            weight_decay=0.01,
        )
        emit_phase(
            output_dir,
            "model_loaded",
            model_path=str(Path(args.model_path).resolve()),
            parameterization="lora",
        )

        _reset_cuda_peak()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        tau = float(calibration["remoteness"]["tau"])
        scale = float(calibration["remoteness"]["scale"])
        negative_scale = float(calibration_row["negative_scale"])
        coefficient = float(calibration_row["coefficient"])
        accumulation = int(train_cfg["gradient_accumulation"])
        for _ in range(accumulation):
            try:
                packed = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                packed = next(iterator)
            positive = core.arena.completion_stats(
                model,
                core.arena.move_to_device(packed["positive"], device),
            )
            positive_lp = positive["seq_lp"].mean()
            bank_size = int(packed.get("bank_size", 0))
            if bank_size < 2 or "bank" not in packed:
                raise RuntimeError(
                    "resource probe requires the registered populated negative bank"
                )
            near_batch, far_batch, _, _ = core.arena.current_bank_training_batches(
                model,
                core.arena.move_to_device(packed["bank"], device),
                packed["positive"]["input_ids"].shape[0],
                bank_size,
            )
            near = core.arena.completion_stats(model, near_batch)
            far = core.arena.completion_stats(model, far_batch)
            near_distance = core.normalized_distance(
                near["seq_lp"], tau=tau, surprisal_scale=scale
            )
            far_distance = core.normalized_distance(
                far["seq_lp"], tau=tau, surprisal_scale=scale
            )
            near_weights = core.taper_weight(
                cell.method, near_distance, coefficient
            ).detach()
            far_weights = core.taper_weight(
                cell.method, far_distance, coefficient
            ).detach()
            negative_lp = 0.5 * (near_weights * near["seq_lp"]).mean() + 0.5 * (
                far_weights * far["seq_lp"]
            ).mean()
            raw_loss = -(positive_lp - negative_scale * negative_lp)
            if not bool(torch.isfinite(raw_loss)):
                raise RuntimeError("resource probe observed a non-finite training loss")
            (raw_loss / accumulation).backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable,
            float(train_cfg["maximum_gradient_norm"]),
        )
        if not bool(torch.isfinite(grad_norm)):
            raise RuntimeError("resource probe observed a non-finite gradient")
        if not core.arena.optimizer_step_with_last_finite_guard(
            optimizer, trainable
        ):
            raise RuntimeError("resource probe observed non-finite parameters")
        emit_phase(
            output_dir,
            "training_peak_completed",
            micro_batch=int(train_cfg["micro_batch"]),
            gradient_accumulation=accumulation,
            negative_bank_size=bank_size,
            max_length=int(model_cfg["max_length"]),
        )

        if adapter_dir.exists():
            import shutil

            shutil.rmtree(adapter_dir)
        model.save_pretrained(adapter_dir)
        del optimizer, trainable, model, iterator, loader, dataset
        optimizer = model = iterator = loader = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        _reset_cuda_peak()
        evaluation_model = core.arena.load_model(
            str(Path(args.model_path).resolve()),
            adapter_path=str(adapter_dir),
            trainable_adapter=False,
            load_in_4bit=bool(model_cfg.get("load_in_4bit", False)),
            dtype=str(model_cfg.get("dtype", "auto")),
            gradient_checkpointing=False,
            parameterization="lora",
        )
        envelope_rows = evaluation_envelope_rows(validation_rows, base_config)
        known_structures = {
            row.get("oracle_structure")
            or core.arena.expression_structure(row["oracle"])
            for row in train_rows
        }
        max_pass = maximum_pass_k(base_config)
        with core.base_runner.onp._temporary_generation_context(evaluation_model):
            metrics = core.arena.evaluate_rows(
                evaluation_model,
                tokenizer,
                envelope_rows,
                int(eval_cfg["batch_size"]),
                int(model_cfg["max_new_tokens"]),
                max_pass,
                int(eval_cfg["seed"]) + max_pass + cell.seed_offset,
                known_structures,
            )
        emit_phase(
            output_dir,
            "evaluation_peak_completed",
            evaluation_rows=len(envelope_rows),
            evaluation_batch_size=int(eval_cfg["batch_size"]),
            max_new_tokens=int(model_cfg["max_new_tokens"]),
            pass_k=max_pass,
            n_eval=int(metrics["n_eval"]),
        )
        emit_phase(
            output_dir,
            "probe_complete",
            scientific_evidence=False,
            scientific_matrix_changed=False,
        )
        return 0
    except Exception as exc:  # noqa: BLE001
        try:
            emit_phase(
                output_dir,
                "probe_failed",
                error_type=type(exc).__name__,
                error=str(exc),
            )
        except Exception:  # noqa: BLE001
            pass
        raise
    finally:
        del optimizer, model, iterator, loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--base_config", required=True)
    parser.add_argument("--sweep_config", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--rho", required=True, type=float)
    parser.add_argument("--seed_offset", required=True, type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    return run_probe(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
