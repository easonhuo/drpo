#!/usr/bin/env python3
"""Training core for the E8 continuous EXP code-first pilot."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import DataLoader

from drpo.countdown_e8_continuous_exp_common import (
    EXPERIMENT_ID,
    VERSION,
    Cell,
    ContinuousUniqueBankDataset,
    _identity,
    arena,
    atomic_json,
    continuous_exp_weights,
    load_yaml,
    make_continuous_unique_bank_collator,
    mean_unique_negative_term,
    unique_negative_expressions,
    validate_grid_config,
    weight_diagnostics,
)


def _identity_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def _best_row(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any] | None:
    candidates = [row for row in rows if row.get(key) is not None]
    if not candidates:
        return None
    winner = max(candidates, key=lambda row: float(row[key]))
    return {"step": int(winner["step"]), "value": float(winner[key])}


def _evaluate_validation(
    *,
    model: Any,
    tokenizer: Any,
    val_rows: list[dict[str, Any]],
    known_structures: set[str],
    model_cfg: Mapping[str, Any],
    eval_cfg: Mapping[str, Any],
    step: int,
    pass64_every: int,
    pass64_enabled: bool,
    examples_override: int | None,
) -> dict[str, Any]:
    examples = int(eval_cfg["examples"]) if examples_override is None else examples_override
    selected = val_rows[:examples]
    pass8 = arena.evaluate_rows(
        model,
        tokenizer,
        selected,
        int(eval_cfg["batch_size"]),
        int(model_cfg["max_new_tokens"]),
        8,
        int(eval_cfg["seed"]),
        known_structures,
    )
    row: dict[str, Any] = {
        "step": int(step),
        "val_greedy": float(pass8["greedy_success"]),
        "val_pass_at_8": float(pass8["pass_at_k"]),
        "val_valid_rate": float(pass8["valid_rate"]),
        "val_pass_at_64": None,
        "validation_examples": len(selected),
    }
    if pass64_enabled and step % pass64_every == 0:
        pass64 = arena.evaluate_rows(
            model,
            tokenizer,
            selected,
            int(eval_cfg["batch_size"]),
            int(model_cfg["max_new_tokens"]),
            64,
            int(eval_cfg["seed"]),
            known_structures,
        )
        row["val_pass_at_64"] = float(pass64["pass_at_k"])
    return row


def _parameter_update_norm(
    before: Sequence[torch.Tensor], parameters: Sequence[torch.nn.Parameter]
) -> float:
    total = torch.zeros((), dtype=torch.float64)
    for saved, parameter in zip(before, parameters, strict=True):
        delta = parameter.detach().float().cpu() - saved
        total += delta.double().square().sum()
    return float(torch.sqrt(total).item())


def train_cell(
    *,
    cell: Cell,
    model_path: Path,
    bank: Path,
    val: Path,
    base_config_path: Path,
    grid_config_path: Path,
    output_dir: Path,
    repo: Path,
    smoke: bool = False,
) -> dict[str, Any]:
    base_config = load_yaml(base_config_path)
    grid_config = load_yaml(grid_config_path)
    validate_grid_config(grid_config)
    train_cfg = base_config["offline_training"]
    model_cfg = base_config["model"]
    eval_cfg = base_config["evaluation"]
    fixed_training = grid_config["training"]
    liveness = grid_config["execution"]["liveness"]
    steps = int(liveness["steps"]) if smoke else int(fixed_training["steps"])
    eval_every = steps if smoke else int(fixed_training["eval_every"])
    pass64_every = 10**9 if smoke else int(fixed_training["pass64_every"])
    examples_override = int(liveness["validation_examples"]) if smoke else None
    seed = int(train_cfg["seed"]) + int(cell.seed_offset)
    arena.seed_all(seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    identity = _identity(
        repo=repo,
        model_path=model_path,
        bank=bank,
        val=val,
        base_config=base_config_path,
        grid_config=grid_config_path,
        cell=cell,
        smoke=smoke,
    )
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        existing = json.loads(summary_path.read_text())
        if _identity_equal(existing.get("run_identity", {}), identity):
            return existing
        raise RuntimeError(f"Stale result identity at {summary_path}; use a new work_dir")

    tokenizer = arena.load_tokenizer(str(model_path))
    train_rows = arena.read_jsonl(bank)
    val_rows = arena.read_jsonl(val)
    unique_counts = [len(unique_negative_expressions(row)) for row in train_rows]
    raw_counts = [len(row.get("negative_bank", [])) for row in train_rows]
    if min(unique_counts) < 1:
        raise RuntimeError("At least one training row has no unique negative")
    dataset = ContinuousUniqueBankDataset(
        train_rows, tokenizer, int(model_cfg["max_length"])
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=int(train_cfg["micro_batch"]),
        shuffle=True,
        generator=generator,
        collate_fn=make_continuous_unique_bank_collator(tokenizer.pad_token_id),
        num_workers=int(train_cfg["num_workers"]),
    )
    iterator = iter(loader)
    model = arena.load_model(
        str(model_path),
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
        trainable, lr=float(train_cfg["learning_rate"]), weight_decay=0.01
    )
    scheduler = arena.get_cosine_schedule_with_warmup(
        optimizer,
        max(1, int(steps * float(train_cfg["warmup_ratio"]))),
        steps,
    )

    best_dir = output_dir / "best_pass8_adapter"
    terminal_dir = output_dir / "terminal_adapter"
    last_finite_dir = output_dir / "last_finite_adapter"
    validation_diagnostics_path = output_dir / "validation_diagnostics.jsonl"
    training_diagnostics_path = output_dir / "training_diagnostics.jsonl"
    metric_rows: list[dict[str, Any]] = []
    checkpoint_records: list[dict[str, Any]] = []
    best_pass8 = -float("inf")
    best_step = 0
    numerical_failure: str | None = None
    terminal_step = 0
    last_finite_step = 0
    stop_reason = "smoke_complete" if smoke else "max_steps"
    known_structures = {
        row.get("oracle_structure") or arena.expression_structure(row["oracle"])
        for row in train_rows
    }

    def evaluate(step: int) -> dict[str, Any]:
        row = _evaluate_validation(
            model=model,
            tokenizer=tokenizer,
            val_rows=val_rows,
            known_structures=known_structures,
            model_cfg=model_cfg,
            eval_cfg=eval_cfg,
            step=step,
            pass64_every=pass64_every,
            pass64_enabled=not smoke,
            examples_override=examples_override,
        )
        row.update(
            {
                "method": cell.method,
                "alpha": cell.alpha,
                "c": cell.c,
                "test_data_used": False,
            }
        )
        metric_rows.append(row)
        with validation_diagnostics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    initial = evaluate(0)
    best_pass8 = float(initial["val_pass_at_8"])
    checkpoint_records.append(
        arena.save_local_model_checkpoint(model, tokenizer, best_dir, "best_pass8", 0)
    )
    model.train()

    for update_step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        diagnostic_keys = (
            "negative_surprisal_mean",
            "u_mean",
            "u_p10",
            "u_p50",
            "u_p90",
            "weight_mean",
            "weight_p10",
            "weight_p50",
            "weight_p90",
            "unique_negative_count_mean",
            "raw_bank_count_mean",
            "duplicates_removed_mean",
        )
        accum: dict[str, float | None] = {
            "loss": 0.0,
            "positive_lp": 0.0,
            "weighted_negative_lp": 0.0,
            **{key: 0.0 for key in diagnostic_keys},
        }
        diagnostic_observations = {key: 0 for key in diagnostic_keys}
        abort_update = False
        for _ in range(int(train_cfg["gradient_accumulation"])):
            try:
                packed = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                packed = next(iterator)
            positive_stats = arena.completion_stats(
                model, arena.move_to_device(packed["positive"], device)
            )
            positive_lp = positive_stats["seq_lp"].mean()
            if cell.alpha == 0.0:
                weighted_negative_lp = torch.zeros(
                    (), device=device, dtype=positive_lp.dtype
                )
                diagnostics = {
                    "negative_surprisal_mean": None,
                    "u_mean": None,
                    "u_p10": None,
                    "u_p50": None,
                    "u_p90": None,
                    "weight_mean": 0.0,
                    "weight_p10": 0.0,
                    "weight_p50": 0.0,
                    "weight_p90": 0.0,
                    "unique_negative_count_mean": float(
                        packed["unique_counts"].float().mean()
                    ),
                    "raw_bank_count_mean": float(
                        packed["raw_bank_counts"].float().mean()
                    ),
                    "duplicates_removed_mean": float(
                        (packed["raw_bank_counts"] - packed["unique_counts"])
                        .float()
                        .mean()
                    ),
                }
            else:
                bank_stats = arena.completion_stats(
                    model, arena.move_to_device(packed["bank"], device)
                )
                weights = continuous_exp_weights(
                    bank_stats["seq_lp"], alpha=cell.alpha, c=cell.c
                )
                weighted_negative_lp = mean_unique_negative_term(
                    bank_stats["seq_lp"],
                    weights,
                    packed["bank_row_index"],
                    packed["unique_counts"],
                )
                diagnostics = weight_diagnostics(
                    bank_stats["seq_lp"],
                    weights,
                    packed["unique_counts"],
                    packed["raw_bank_counts"],
                )
            raw_loss = -(positive_lp - weighted_negative_lp)
            if not bool(torch.isfinite(raw_loss)):
                numerical_failure = f"nonfinite_loss_at_step_{update_step}"
                stop_reason = numerical_failure
                abort_update = True
                break
            (raw_loss / int(train_cfg["gradient_accumulation"])).backward()
            divisor = float(train_cfg["gradient_accumulation"])
            accum["loss"] = float(accum["loss"] or 0.0) + float(
                raw_loss.detach()
            ) / divisor
            accum["positive_lp"] = float(accum["positive_lp"] or 0.0) + float(
                positive_lp.detach()
            ) / divisor
            accum["weighted_negative_lp"] = float(
                accum["weighted_negative_lp"] or 0.0
            ) + float(weighted_negative_lp.detach()) / divisor
            for key in diagnostic_keys:
                value = diagnostics.get(key)
                if value is None:
                    continue
                accum[key] = float(accum[key] or 0.0) + float(value) / divisor
                diagnostic_observations[key] += 1
        if abort_update:
            break

        raw_grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable, float(train_cfg["maximum_gradient_norm"])
        )
        if not bool(torch.isfinite(raw_grad_norm)):
            numerical_failure = f"nonfinite_gradient_at_step_{update_step}"
            stop_reason = numerical_failure
            break
        sample_update_norm = (
            update_step % int(train_cfg["log_every"]) == 0 or update_step == steps
        )
        before = (
            [parameter.detach().float().cpu().clone() for parameter in trainable]
            if sample_update_norm
            else []
        )
        if not arena.optimizer_step_with_last_finite_guard(optimizer, trainable):
            numerical_failure = f"nonfinite_parameters_at_step_{update_step}"
            stop_reason = numerical_failure
            break
        update_norm = (
            _parameter_update_norm(before, trainable) if sample_update_norm else None
        )
        scheduler.step()
        terminal_step = update_step
        last_finite_step = update_step
        for key in diagnostic_keys:
            if diagnostic_observations[key] == 0:
                accum[key] = None
        accum["raw_gradient_norm"] = float(raw_grad_norm)
        accum["optimizer_update_norm"] = update_norm
        if update_step % int(train_cfg["log_every"]) == 0 or update_step == steps:
            training_record = {
                "cell": cell.name,
                "step": update_step,
                "method": cell.method,
                "alpha": cell.alpha,
                "c": cell.c,
                "test_data_used": False,
                **accum,
            }
            with training_diagnostics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(training_record, ensure_ascii=False) + "\n")
            print(json.dumps(training_record), flush=True)
        if update_step % eval_every == 0 or update_step == steps:
            row = evaluate(update_step)
            value = float(row["val_pass_at_8"])
            if value > best_pass8:
                best_pass8 = value
                best_step = update_step
                checkpoint_records = [
                    record
                    for record in checkpoint_records
                    if record["kind"] != "best_pass8"
                ]
                checkpoint_records.append(
                    arena.save_local_model_checkpoint(
                        model, tokenizer, best_dir, "best_pass8", update_step
                    )
                )
            model.train()

    if numerical_failure:
        checkpoint_records.append(
            arena.save_local_model_checkpoint(
                model, tokenizer, last_finite_dir, "last_finite", last_finite_step
            )
        )
        terminal_kind = "last_finite"
    else:
        checkpoint_records.append(
            arena.save_local_model_checkpoint(
                model, tokenizer, terminal_dir, "terminal", terminal_step
            )
        )
        terminal_kind = "terminal"

    fieldnames = sorted({key for row in metric_rows for key in row})
    with (output_dir / "metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(arena.csv_safe_row(row) for row in metric_rows)

    metric_bests = {
        key: _best_row(metric_rows, key)
        for key in ("val_greedy", "val_pass_at_8", "val_pass_at_64", "val_valid_rate")
    }
    manifest = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "version": VERSION,
        "result_status": "smoke_only" if smoke else "pilot",
        "registration_state": "dev_code_first_unregistered",
        "cell": cell.name,
        "method": cell.method,
        "alpha": cell.alpha,
        "c": cell.c,
        "weight_formula": "alpha*exp(-c*u^2)",
        "u_definition": "current_sequence_surprisal/2",
        "unique_negative_denominator": True,
        "weight_sum_normalization": False,
        "extreme_selection_used": False,
        "hidden_scale_used": False,
        "seed": seed,
        "seed_offset": cell.seed_offset,
        "steps_requested": steps,
        "terminal_step": terminal_step,
        "last_finite_step": last_finite_step,
        "best_pass8_step": best_step,
        "best_pass8_value": best_pass8,
        "stop_reason": stop_reason,
        "numerical_failure": numerical_failure,
        "terminal_checkpoint_kind": terminal_kind,
        "checkpoint_policy": "server-local LoRA only",
        "checkpoints": checkpoint_records,
        "run_identity": identity,
        "bank_audit": {
            "rows": len(train_rows),
            "raw_bank_count_min": min(raw_counts),
            "raw_bank_count_max": max(raw_counts),
            "unique_negative_count_min": min(unique_counts),
            "unique_negative_count_max": max(unique_counts),
            "rows_with_duplicates_removed": sum(
                raw > unique
                for raw, unique in zip(raw_counts, unique_counts, strict=True)
            ),
        },
        "metric_bests": metric_bests,
        "terminal_metrics": dict(metric_rows[-1]),
        "diagnostic_files": {
            "training": str(training_diagnostics_path),
            "validation": str(validation_diagnostics_path),
        },
        "test_data_used": False,
        "reporting_separation": {
            "task_performance": "validation trajectories and metric-specific best values",
            "support_or_structure_boundary": (
                "valid_rate diagnostic only; no formal boundary threshold registered"
            ),
            "nan_inf_numerical_failure": numerical_failure,
        },
        "fixed_horizon_is_convergence": False,
        "method_ranking_claim_allowed": False,
    }
    atomic_json(output_dir / "manifest.json", manifest)
    atomic_json(summary_path, manifest)
    return manifest
