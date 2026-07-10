#!/usr/bin/env python3
"""Countdown E8 base-start RL/replay diagnostic pilot.

Registered experiment: ``EXT-C-E8-BASE-RL-REPLAY-0.5B-01``.

This runner removes Countdown SFT as the default warm start.  It starts from the
Qwen pretrained base and compares:

* base_eval;
* oracle-offline positive-only training;
* oracle-offline positive+negative with base-specific calibration;
* online on-policy positive-only RFT;
* online replay positive-only;
* online replay positive+negative.

The goal is to distinguish oracle/offline corpus coverage, online self-sampled
positive support, dynamic replay reuse, and base-calibrated negative updates. It
is a pilot diagnostic, not a formal method ranking or a replacement for the
controlled C-U1/D-U1 mechanism experiments.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml

try:
    from drpo import countdown_e8_onpolicy as onp
    from drpo import countdown_qwen_arena_onefile as arena
except ImportError:  # pragma: no cover
    import countdown_e8_onpolicy as onp  # type: ignore
    import countdown_qwen_arena_onefile as arena  # type: ignore

EXPERIMENT_ID = "EXT-C-E8-BASE-RL-REPLAY-0.5B-01"
VERSION = "0.1.0-base-rl-replay"
DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "countdown_e8_base_rl_replay_0p5b.yaml"
)
METHODS = (
    "base_eval",
    "base_oracle_offline_positive_only",
    "base_oracle_offline_pos_neg_recalibrated",
    "base_onpolicy_positive_only",
    "base_online_replay_positive_only",
    "base_online_replay_pos_neg",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    tmp.replace(path)


def _jsonl_write(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _csv_write(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, ensure_ascii=False, sort_keys=True)
                        if isinstance(value, (dict, list, tuple))
                        else value
                    )
                    for key, value in row.items()
                }
            )


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("base RL/replay config must be a YAML mapping")
    if value.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("experiment_id mismatch")
    if tuple(value.get("methods") or []) != METHODS:
        raise ValueError("method set/order is frozen for this pilot")
    if value.get("result_status") != "pilot":
        raise ValueError("base RL/replay diagnostic must remain pilot")
    data = value["data"]
    if (int(data["train_rows"]), int(data["validation_rows"]), int(data["test_rows"])) != (6000, 500, 1000):
        raise ValueError("Countdown row counts are frozen at 6000/500/1000")
    if data.get("split_protocol") != "structural_family_holdout":
        raise ValueError("must reuse structural family-holdout split")
    guards = value["scope_guards"]
    required_true = (
        "no_countdown_sft_warmstart",
        "starts_from_qwen_pretrained_base",
        "random_initialization_forbidden",
        "sft_reference_is_baseline_only",
        "methods_do_not_use_taper_family",
        "model_weights_not_in_update_package",
    )
    for key in required_true:
        if guards.get(key) is not True:
            raise ValueError(f"scope guard {key} must be true")
    if not value["negative_calibration"].get("negative_scale_multipliers"):
        raise ValueError("negative_scale_multipliers must be non-empty")
    return value


def source_provenance() -> dict[str, Any]:
    runner = Path(__file__).resolve()
    return {
        "experiment_id": EXPERIMENT_ID,
        "implementation_version": VERSION,
        "runner_source_file": str(runner),
        "runner_source_sha256": _sha256_file(runner),
        "onpolicy_helper_source_file": str(Path(onp.__file__).resolve()),
        "onpolicy_helper_source_sha256": _sha256_file(Path(onp.__file__).resolve()),
        "shared_arena_source_file": str(Path(arena.__file__).resolve()),
        "shared_arena_source_sha256": _sha256_file(Path(arena.__file__).resolve()),
        "git_commit": arena.source_provenance().get("git_commit"),
        "git_branch": arena.source_provenance().get("git_branch"),
        "git_dirty": arena.source_provenance().get("git_dirty"),
    }


def _as_data_config(config: Mapping[str, Any]) -> dict[str, Any]:
    data = config["data"]
    return {
        "data": {
            "train_rows": int(data["train_rows"]),
            "validation_rows": int(data["validation_rows"]),
            "test_rows": int(data["test_rows"]),
            "generation_seed": int(data["generation_seed"]),
            "numbers_per_problem": int(data["numbers_per_problem"]),
            "split_protocol": str(data["split_protocol"]),
        }
    }


def generate_or_load_data(work_dir: Path, config: Mapping[str, Any]) -> dict[str, Path]:
    return onp.generate_or_load_data(work_dir, _as_data_config(config))


def _model_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    model_cfg = config["model"]
    return {
        "load_in_4bit": bool(model_cfg.get("load_in_4bit", False)),
        "dtype": str(model_cfg.get("dtype", "auto")),
    }


def load_base_model(
    model_path: Path,
    config: Mapping[str, Any],
    *,
    trainable: bool,
    gradient_checkpointing: bool,
) -> Any:
    kwargs = _model_kwargs(config)
    return arena.load_model(
        str(model_path),
        None,
        trainable_adapter=trainable,
        load_in_4bit=bool(kwargs["load_in_4bit"]),
        dtype=str(kwargs["dtype"]),
        gradient_checkpointing=gradient_checkpointing,
        parameterization="lora",
    )


def evaluate_model(
    model: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    seed: int,
    prefix: str,
) -> dict[str, float]:
    eval_cfg = config["evaluation"]
    model_cfg = config["model"]
    batch_size = int(eval_cfg["batch_size"])
    max_new = int(model_cfg["max_new_tokens"])
    result: dict[str, float] = {}
    with onp._temporary_generation_context(model):
        for pass_k in eval_cfg["pass_ks"]:
            metrics = arena.evaluate_rows(
                model,
                tokenizer,
                list(rows)[: int(eval_cfg.get("test_examples" if prefix == "test" else "examples", len(rows)))],
                batch_size,
                max_new,
                int(pass_k),
                seed + int(pass_k),
            )
            if pass_k == eval_cfg["pass_ks"][0]:
                result[f"{prefix}_greedy_success"] = float(metrics["greedy_success"])
                result[f"{prefix}_valid_rate"] = float(metrics["valid_rate"])
                result[f"{prefix}_n_eval"] = float(metrics["n_eval"])
            result[f"{prefix}_pass_at_{int(pass_k)}"] = float(metrics["pass_at_k"])
    return result


def evaluate_base(model_path: Path, work_dir: Path, data_paths: Mapping[str, Path], config: Mapping[str, Any]) -> dict[str, Any]:
    method_dir = work_dir / "methods" / "base_eval"
    if (method_dir / "summary.json").exists():
        return json.loads((method_dir / "summary.json").read_text())
    method_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = arena.load_tokenizer(str(model_path))
    model = load_base_model(model_path, config, trainable=False, gradient_checkpointing=False)
    val_rows = arena.read_jsonl(data_paths["validation"])
    test_rows = arena.read_jsonl(data_paths["test"])
    summary: dict[str, Any] = {"method": "base_eval", "status": "eval_complete"}
    summary.update(evaluate_model(model, tokenizer, val_rows, config, seed=int(config["evaluation"]["seed"]), prefix="validation"))
    summary.update(evaluate_model(model, tokenizer, test_rows, config, seed=int(config["evaluation"]["test_seed"]), prefix="test"))
    _atomic_json(method_dir / "summary.json", summary)
    return summary


def build_offline_bank(model_path: Path, work_dir: Path, data_paths: Mapping[str, Path], config: Mapping[str, Any], *, gpu: str) -> Path:
    out = work_dir / "offline_bank" / "offline_6000.jsonl"
    if out.exists():
        return out
    if gpu != "auto":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu).split(",")[0]
    cfg = config["data"]
    model_cfg = config["model"]
    args = argparse.Namespace(
        model_path=str(model_path),
        reference_adapter=None,
        sft_adapter=None,
        input_data=str(data_paths["train"]),
        split_manifest=str(data_paths["split_manifest"]),
        output_data=str(out),
        manifest_json=str(work_dir / "offline_bank" / "manifest.json"),
        nested_output_dir=None,
        rollouts=int(cfg["offline_rollouts"]),
        batch_size=4,
        pair_resample_rounds=int(cfg["offline_pair_resample_rounds"]),
        min_negative_candidates=int(cfg["offline_bank_size"]),
        negative_bank_size=int(cfg["offline_bank_size"]),
        synthetic_rescue_candidates=64,
        score_batch_size=16,
        max_examples=int(cfg["offline_bank_rows"]),
        balance_by_oracle_pattern=True,
        nested_sizes="",
        temperature=0.8,
        top_p=0.95,
        max_new_tokens=int(model_cfg["max_new_tokens"]),
        max_length=int(model_cfg["max_length"]),
        min_surprisal_gap=0.5,
        max_token_length_diff=2,
        max_tree_depth_diff=1,
        max_value_error_ratio=4.0,
        seed=int(cfg["generation_seed"]),
        load_in_4bit=bool(model_cfg.get("load_in_4bit", False)),
        dtype=str(model_cfg.get("dtype", "auto")),
    )
    arena.cmd_build_offline(args)
    return out


def calibrate_base_negative(model_path: Path, work_dir: Path, offline_data: Path, config: Mapping[str, Any]) -> Path:
    out = work_dir / "negative_calibration" / "base_negative_budget_calibration.json"
    if out.exists():
        return out
    neg = config["negative_calibration"]
    model_cfg = config["model"]
    args = argparse.Namespace(
        model_path=str(model_path),
        reference_adapter=None,
        offline_data=str(offline_data),
        output_json=str(out),
        batch_size=int(neg["batch_size"]),
        calibration_batches=int(neg["batches"]),
        max_length=int(model_cfg["max_length"]),
        near_mix=float(neg["near_mix"]),
        far_mix=float(neg["far_mix"]),
        exp_lambda=float(neg["exp_lambda"]),
        surprisal_threshold=float(neg["surprisal_threshold"]),
        seed=int(neg["seed"]),
        load_in_4bit=bool(model_cfg.get("load_in_4bit", False)),
        dtype=str(model_cfg.get("dtype", "auto")),
    )
    arena.cmd_calibrate_global(args)
    return out



def evaluate_adapter_checkpoint(
    model_path: Path,
    adapter_path: Path,
    data_paths: Mapping[str, Path],
    config: Mapping[str, Any],
    *,
    seed_offset: int = 0,
) -> dict[str, Any]:
    tokenizer = arena.load_tokenizer(str(model_path))
    kwargs = _model_kwargs(config)
    model = arena.load_model(
        str(model_path),
        str(adapter_path),
        trainable_adapter=False,
        load_in_4bit=bool(kwargs["load_in_4bit"]),
        dtype=str(kwargs["dtype"]),
        gradient_checkpointing=False,
        parameterization="lora",
    )
    val_rows = arena.read_jsonl(data_paths["validation"])
    test_rows = arena.read_jsonl(data_paths["test"])
    result: dict[str, Any] = {}
    result.update(evaluate_model(model, tokenizer, val_rows, config, seed=int(config["evaluation"]["seed"]) + seed_offset, prefix="validation"))
    result.update(evaluate_model(model, tokenizer, test_rows, config, seed=int(config["evaluation"]["test_seed"]) + seed_offset, prefix="test"))
    return result

def train_offline_method(
    model_path: Path,
    work_dir: Path,
    data_paths: Mapping[str, Path],
    offline_data: Path,
    config: Mapping[str, Any],
    *,
    method: str,
    output_name: str,
    seed: int,
    calibration_json: Path | None = None,
    negative_scale_multiplier: float = 1.0,
) -> dict[str, Any]:
    out_dir = work_dir / "methods" / output_name
    if (out_dir / "summary.json").exists():
        return json.loads((out_dir / "summary.json").read_text())
    train_cfg = config["offline_training"]
    model_cfg = config["model"]
    args = argparse.Namespace(
        model_path=str(model_path),
        reference_adapter=None,
        sft_adapter=None,
        offline_data=str(offline_data),
        val_data=str(data_paths["validation"]),
        structure_reference_data=str(data_paths["train"]),
        output_dir=str(out_dir),
        method=method,
        steps=int(train_cfg["steps"]),
        min_steps=int(train_cfg["min_steps"]),
        early_stop_patience=int(train_cfg["early_stop_patience"]),
        early_stop_delta=float(train_cfg["early_stop_delta"]),
        selection_metric=str(train_cfg["selection_metric"]),
        micro_batch=int(train_cfg["micro_batch"]),
        grad_accum=int(train_cfg["gradient_accumulation"]),
        lr=float(train_cfg["learning_rate"]),
        warmup_ratio=float(train_cfg["warmup_ratio"]),
        max_grad_norm=float(train_cfg["maximum_gradient_norm"]),
        max_length=int(model_cfg["max_length"]),
        max_new_tokens=int(model_cfg["max_new_tokens"]),
        eval_examples=int(config["evaluation"]["examples"]),
        eval_batch=int(config["evaluation"]["batch_size"]),
        pass_k=int(config["evaluation"]["pass_ks"][0]),
        negative_scale=None,
        negative_scale_multiplier=float(negative_scale_multiplier),
        near_mix=float(config["negative_calibration"]["near_mix"]),
        far_mix=float(config["negative_calibration"]["far_mix"]),
        global_gamma=0.55,
        negative_calibration_json=str(calibration_json) if calibration_json else None,
        exp_lambda=float(config["negative_calibration"]["exp_lambda"]),
        surprisal_threshold=float(config["negative_calibration"]["surprisal_threshold"]),
        entropy_coef=0.02,
        target_entropy=1.8,
        target_entropy_coef=0.05,
        sbrc_kappa=0.92,
        entropy_floor=1.0,
        eval_every=int(train_cfg["eval_every"]),
        eval_seed=int(config["evaluation"]["seed"]),
        diagnostic_examples=int(train_cfg["diagnostic_examples"]),
        diagnostic_gradient_examples=int(train_cfg["diagnostic_gradient_examples"]),
        diagnostic_batch=int(train_cfg["diagnostic_batch"]),
        log_every=int(train_cfg["log_every"]),
        num_workers=int(train_cfg["num_workers"]),
        seed=int(seed),
        result_status=str(config["result_status"]),
        load_in_4bit=bool(model_cfg.get("load_in_4bit", False)),
        dtype=str(model_cfg.get("dtype", "auto")),
    )
    arena.cmd_train_method(args)
    manifest = json.loads((out_dir / "manifest.json").read_text())
    best_eval = evaluate_adapter_checkpoint(
        model_path, out_dir / "best_adapter", data_paths, config, seed_offset=seed
    )
    terminal_eval = None
    if (out_dir / "terminal_adapter" / "adapter_config.json").exists():
        terminal_eval = evaluate_adapter_checkpoint(
            model_path, out_dir / "terminal_adapter", data_paths, config, seed_offset=seed + 17
        )
    summary = {
        "method": output_name,
        "arena_method": method,
        "seed": seed,
        "best_step": manifest.get("best_step"),
        "best_value": manifest.get("best_value"),
        "terminal_step": manifest.get("terminal_step"),
        "stop_reason": manifest.get("stop_reason"),
        "negative_scale_multiplier": negative_scale_multiplier,
        "negative_calibration_json": str(calibration_json) if calibration_json else None,
        "best_evaluation": best_eval,
        "terminal_evaluation": terminal_eval,
        "summary_path": str(out_dir / "manifest.json"),
    }
    _atomic_json(out_dir / "summary.json", summary)
    return summary


def _collate_examples(tokenizer: Any, rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    encoded = [
        arena.encode_prompt_completion(
            tokenizer,
            str(row["prompt"]),
            str(row["completion"]),
            int(config["model"]["max_length"]),
        )
        for row in rows
    ]
    return arena.pad_encoded(encoded, tokenizer.pad_token_id)


def _sample_batch(buffer: Sequence[Mapping[str, Any]], batch_size: int, rng: random.Random) -> list[Mapping[str, Any]]:
    if not buffer:
        return []
    if len(buffer) <= batch_size:
        return [buffer[rng.randrange(len(buffer))] for _ in range(batch_size)]
    return rng.sample(list(buffer), batch_size)


def verify_rollouts(
    rows: Sequence[Mapping[str, Any]],
    outputs: Sequence[Sequence[str]],
    *,
    max_correct_per_prompt: int,
    max_negative_per_prompt: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, float]]:
    positives: list[dict[str, Any]] = []
    negatives: list[dict[str, Any]] = []
    prompts_with_positive = 0
    prompts_with_negative = 0
    total = 0
    for row, completions in zip(rows, outputs):
        seen_pos: set[str] = set()
        seen_neg: set[str] = set()
        row_pos = 0
        row_neg = 0
        for text in completions:
            total += 1
            check = arena.verify_expression(text, row["numbers"], int(row["target"]))
            expression = str(check.get("expression") or "")
            key = "".join(expression.split())
            if not expression:
                continue
            item = {
                "prompt_id": row.get("id"),
                "prompt": row["prompt"],
                "completion": expression,
                "numbers": row["numbers"],
                "target": row["target"],
            }
            if bool(check.get("correct")):
                if key not in seen_pos and row_pos < max_correct_per_prompt:
                    seen_pos.add(key)
                    positives.append(item)
                    row_pos += 1
            elif key not in seen_neg and row_neg < max_negative_per_prompt:
                seen_neg.add(key)
                negatives.append(item)
                row_neg += 1
        prompts_with_positive += int(row_pos > 0)
        prompts_with_negative += int(row_neg > 0)
    stats = {
        "sampled_completions": float(total),
        "correct_selected": float(len(positives)),
        "negative_selected": float(len(negatives)),
        "usable_positive_prompt_fraction": prompts_with_positive / max(len(rows), 1),
        "usable_negative_prompt_fraction": prompts_with_negative / max(len(rows), 1),
    }
    return positives, negatives, stats


def _finite_trainable(model: Any) -> bool:
    params = [p for p in model.parameters() if p.requires_grad]
    return arena._trainable_parameters_finite(params)


def _optimizer_step(
    model: Any,
    tokenizer: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    config: Mapping[str, Any],
    positives: Sequence[Mapping[str, Any]],
    negatives: Sequence[Mapping[str, Any]],
    *,
    negative_scale: float,
    max_grad_norm: float,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer.zero_grad(set_to_none=True)
    pos_batch = arena.move_to_device(_collate_examples(tokenizer, positives, config), device)
    pos_loss = model(**pos_batch, use_cache=False).loss
    if not bool(torch.isfinite(pos_loss)):
        return {"applied": False, "failure": "nonfinite_positive_loss"}
    loss = pos_loss
    neg_loss_value = None
    if negatives and negative_scale > 0:
        neg_batch = arena.move_to_device(_collate_examples(tokenizer, negatives, config), device)
        neg_loss = model(**neg_batch, use_cache=False).loss
        if not bool(torch.isfinite(neg_loss)):
            return {"applied": False, "failure": "nonfinite_negative_loss"}
        loss = pos_loss - float(negative_scale) * neg_loss
        neg_loss_value = float(neg_loss.detach().cpu())
    if not bool(torch.isfinite(loss)):
        return {"applied": False, "failure": "nonfinite_loss"}
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(trainable, float(max_grad_norm))
    if not bool(torch.isfinite(grad_norm)):
        return {"applied": False, "failure": "nonfinite_gradient"}
    applied = arena.optimizer_step_with_last_finite_guard(optimizer, trainable)
    if not applied or not _finite_trainable(model):
        return {"applied": False, "failure": "nonfinite_parameters"}
    scheduler.step()
    return {
        "applied": True,
        "loss": float(loss.detach().cpu()),
        "positive_loss": float(pos_loss.detach().cpu()),
        "negative_loss": neg_loss_value,
        "grad_norm_before_clip": float(grad_norm.detach().cpu()),
    }


def prompt_plan(num_rows: int, *, seed: int, steps: int, prompts_per_step: int) -> list[list[int]]:
    return onp.prompt_attempt_plan(num_rows, seed=seed, attempts=steps, prompts_per_attempt=prompts_per_step)


def train_online_or_replay(
    model_path: Path,
    work_dir: Path,
    data_paths: Mapping[str, Path],
    config: Mapping[str, Any],
    *,
    method: str,
    seed: int,
    replay: bool,
    use_negatives: bool,
) -> dict[str, Any]:
    method_dir = work_dir / "methods" / method / f"seed_{seed}"
    if (method_dir / "summary.json").exists():
        return json.loads((method_dir / "summary.json").read_text())
    method_dir.mkdir(parents=True, exist_ok=True)
    arena.seed_all(seed)
    rng = random.Random(seed)
    tokenizer = arena.load_tokenizer(str(model_path))
    model = load_base_model(model_path, config, trainable=True, gradient_checkpointing=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError(f"{method} has no trainable LoRA parameters")
    train_rows = arena.read_jsonl(data_paths["train"])
    val_rows = arena.read_jsonl(data_paths["validation"])
    test_rows = arena.read_jsonl(data_paths["test"])
    train_cfg = config["replay_training"] if replay else config["onpolicy_training"]
    total_units = int(train_cfg["cycles"] if replay else train_cfg["sampling_attempts"])
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    scheduler = onp._cosine_warmup_scheduler(
        optimizer,
        int(total_units * float(train_cfg["warmup_ratio"])),
        max(total_units, 1) * max(1, int(train_cfg.get("train_steps_per_cycle", 1))),
    )
    pool = train_rows[: int(train_cfg.get("prompt_pool_rows", len(train_rows)))]
    plan = prompt_plan(
        len(pool),
        seed=seed,
        steps=total_units,
        prompts_per_step=int(train_cfg["prompts_per_cycle"] if replay else train_cfg["prompts_per_attempt"]),
    )
    metrics_rows: list[dict[str, Any]] = []
    train_log: list[dict[str, Any]] = []
    positive_buffer: list[dict[str, Any]] = []
    negative_buffer: list[dict[str, Any]] = []
    optimizer_steps = 0
    skipped_units = 0
    numerical_failure: str | None = None
    best_value = -float("inf")
    best_unit = 0
    best_dir = method_dir / "best_adapter"
    terminal_dir = method_dir / "terminal_adapter"
    initial = evaluate_model(model, tokenizer, val_rows, config, seed=int(config["evaluation"]["seed"]) + seed, prefix="validation")
    metrics_rows.append({"unit": 0, "optimizer_step": 0, **initial})
    best_value = float(initial[f"validation_{config['evaluation']['selection_metric']}"])
    onp._save_adapter(model, tokenizer, best_dir)
    model.train()
    for unit, indices in enumerate(plan, start=1):
        rows = [pool[i] for i in indices]
        with onp._temporary_generation_context(model):
            outputs = arena.generate_outputs(
                model,
                tokenizer,
                [row["prompt"] for row in rows],
                int(config["model"]["max_new_tokens"]),
                True,
                float(train_cfg["temperature"]),
                float(train_cfg["top_p"]),
                int(train_cfg["rollouts_per_prompt"]),
            )
        positives, negatives, sample_stats = verify_rollouts(
            rows,
            outputs,
            max_correct_per_prompt=int(train_cfg["max_correct_per_prompt"]),
            max_negative_per_prompt=int(train_cfg.get("max_negative_per_prompt", 0)),
        )
        if replay:
            positive_buffer.extend(positives)
            negative_buffer.extend(negatives)
            positive_buffer = positive_buffer[-int(train_cfg["max_positive_buffer"]):]
            negative_buffer = negative_buffer[-int(train_cfg["max_negative_buffer"]):]
            step_source_pos = positive_buffer
            step_source_neg = negative_buffer if use_negatives else []
            train_steps_this_unit = int(train_cfg["train_steps_per_cycle"])
        else:
            step_source_pos = positives
            step_source_neg = []
            train_steps_this_unit = 1
        applied_this_unit = 0
        loss_rows: list[dict[str, Any]] = []
        for _ in range(train_steps_this_unit):
            pos_batch = _sample_batch(
                step_source_pos,
                int(train_cfg.get("replay_batch_positives", len(step_source_pos))),
                rng,
            )
            if not pos_batch:
                break
            neg_batch = _sample_batch(
                step_source_neg,
                int(train_cfg.get("replay_batch_negatives", 0)),
                rng,
            ) if use_negatives else []
            result = _optimizer_step(
                model,
                tokenizer,
                optimizer,
                scheduler,
                config,
                pos_batch,
                neg_batch,
                negative_scale=float(train_cfg.get("negative_scale", 0.0)) if use_negatives else 0.0,
                max_grad_norm=float(train_cfg["maximum_gradient_norm"]),
            )
            if not result["applied"]:
                numerical_failure = f"{result['failure']}_at_unit_{unit}"
                loss_rows.append(result)
                break
            optimizer_steps += 1
            applied_this_unit += 1
            loss_rows.append(result)
        if applied_this_unit == 0:
            skipped_units += 1
        log_row = {
            "unit": unit,
            "optimizer_step_after": optimizer_steps,
            "applied_train_steps": applied_this_unit,
            "positive_buffer_size": len(positive_buffer),
            "negative_buffer_size": len(negative_buffer),
            "skipped": applied_this_unit == 0,
            "numerical_failure": numerical_failure,
            **sample_stats,
        }
        if loss_rows:
            log_row.update({f"last_{k}": v for k, v in loss_rows[-1].items() if k != "applied"})
        train_log.append(log_row)
        if unit % int(train_cfg["log_every_cycles"] if replay else train_cfg["log_every_attempts"]) == 0:
            print(json.dumps({"method": method, "seed": seed, **log_row}), flush=True)
        if numerical_failure:
            break
        eval_every = int(train_cfg["eval_every_cycles"] if replay else train_cfg["eval_every_attempts"])
        if unit % eval_every == 0:
            metrics = evaluate_model(model, tokenizer, val_rows, config, seed=int(config["evaluation"]["seed"]) + seed, prefix="validation")
            row = {"unit": unit, "optimizer_step": optimizer_steps, **metrics}
            metrics_rows.append(row)
            value = float(metrics[f"validation_{config['evaluation']['selection_metric']}"])
            if value > best_value + float(config["evaluation"]["selection_delta"]):
                best_value = value
                best_unit = unit
                onp._save_adapter(model, tokenizer, best_dir)
            model.train()
    status = "finite_budget_complete" if numerical_failure is None else "numerical_failure"
    terminal_validation = None
    terminal_test = None
    if numerical_failure is None:
        terminal_validation = evaluate_model(model, tokenizer, val_rows, config, seed=int(config["evaluation"]["seed"]) + seed, prefix="validation")
        terminal_test = evaluate_model(model, tokenizer, test_rows, config, seed=int(config["evaluation"]["test_seed"]) + seed, prefix="test")
        onp._save_adapter(model, tokenizer, terminal_dir)
    _csv_write(method_dir / "training_log.csv", train_log)
    _csv_write(method_dir / "metrics.csv", metrics_rows)
    _jsonl_write(method_dir / "positive_replay_buffer_tail.jsonl", positive_buffer[-200:])
    if use_negatives:
        _jsonl_write(method_dir / "negative_replay_buffer_tail.jsonl", negative_buffer[-200:])
    summary = {
        "method": method,
        "seed": seed,
        "status": status,
        "replay": replay,
        "uses_negatives": use_negatives,
        "units": total_units,
        "optimizer_steps": optimizer_steps,
        "skipped_units": skipped_units,
        "positive_buffer_size": len(positive_buffer),
        "negative_buffer_size": len(negative_buffer),
        "best_unit": best_unit,
        "best_validation_value": best_value,
        "terminal_validation": terminal_validation,
        "terminal_test": terminal_test,
        "numerical_failure": numerical_failure,
    }
    _atomic_json(method_dir / "summary.json", summary)
    return summary


def run_offline_grid(
    model_path: Path,
    work_dir: Path,
    data_paths: Mapping[str, Path],
    offline_data: Path,
    calibration_json: Path,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seed = int(config["offline_training"]["seed"])
    rows.append(
        train_offline_method(
            model_path,
            work_dir,
            data_paths,
            offline_data,
            config,
            method="positive_only",
            output_name="base_oracle_offline_positive_only",
            seed=seed,
        )
    )
    for multiplier in config["negative_calibration"]["negative_scale_multipliers"]:
        rows.append(
            train_offline_method(
                model_path,
                work_dir,
                data_paths,
                offline_data,
                config,
                method=str(config["negative_calibration"]["method"]),
                output_name=f"base_oracle_offline_pos_neg_recalibrated_x{str(multiplier).replace('.', 'p')}",
                seed=seed + int(float(multiplier) * 1000),
                calibration_json=calibration_json,
                negative_scale_multiplier=float(multiplier),
            )
        )
    return rows


def terminal_audit(summary_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failures = [row for row in summary_rows if row.get("numerical_failure")]
    return {
        "experiment_id": EXPERIMENT_ID,
        "status": "pilot_raw_complete" if not failures else "pilot_incomplete_numerical_failure",
        "result_interpretation_limit": "finite-budget single-seed pilot; not formal method ranking",
        "separates_task_performance_from_numerical_failure": True,
        "separates_online_signal_sparsity_from_task_failure": True,
        "numerical_failure_count": len(failures),
        "failed_methods": [row.get("method") for row in failures],
        "notes": [
            "All RL methods start from the Qwen pretrained base without Countdown SFT warmstart.",
            "Oracle-offline corpus uses fixed oracle positives and generated negative bank.",
            "Online replay methods keep and reuse historical self-sampled positives/negatives; they are not fixed offline corpora.",
            "base_oracle_offline_pos_neg_recalibrated scans only pre-registered negative-scale multipliers after base-specific calibration.",
        ],
    }


def cmd_run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    model_path = Path(args.model_path).resolve()
    work_dir = Path(args.work_dir).resolve()
    if not model_path.is_dir():
        raise SystemExit(f"Model directory does not exist: {model_path}")
    if (work_dir / "RUN_COMPLETE.json").exists():
        raise RuntimeError("work_dir already contains a completed E8 base RL/replay run")
    work_dir.mkdir(parents=True, exist_ok=True)
    if args.gpu != "auto":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu).split(",")[0]
    _atomic_json(
        work_dir / "run_config.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "implementation_version": VERSION,
            "model_path": str(model_path),
            "work_dir": str(work_dir),
            "config": config,
            "source_provenance": source_provenance(),
        },
    )
    data_paths = generate_or_load_data(work_dir, config)
    summary_rows: list[dict[str, Any]] = []
    if args.run_base_eval:
        summary_rows.append(evaluate_base(model_path, work_dir, data_paths, config))
    offline_data = build_offline_bank(model_path, work_dir, data_paths, config, gpu=args.gpu)
    calibration_json = calibrate_base_negative(model_path, work_dir, offline_data, config)
    if args.run_offline:
        summary_rows.extend(run_offline_grid(model_path, work_dir, data_paths, offline_data, calibration_json, config))
    if args.run_online:
        for seed in config["onpolicy_training"]["seeds"]:
            summary_rows.append(
                train_online_or_replay(
                    model_path,
                    work_dir,
                    data_paths,
                    config,
                    method="base_onpolicy_positive_only",
                    seed=int(seed),
                    replay=False,
                    use_negatives=False,
                )
            )
    if args.run_replay:
        for seed in config["replay_training"]["seeds"]:
            summary_rows.append(
                train_online_or_replay(
                    model_path,
                    work_dir,
                    data_paths,
                    config,
                    method="base_online_replay_positive_only",
                    seed=int(seed),
                    replay=True,
                    use_negatives=False,
                )
            )
            summary_rows.append(
                train_online_or_replay(
                    model_path,
                    work_dir,
                    data_paths,
                    config,
                    method="base_online_replay_pos_neg",
                    seed=int(seed) + 100,
                    replay=True,
                    use_negatives=True,
                )
            )
    _csv_write(work_dir / "base_rl_replay_summary.csv", summary_rows)
    audit = terminal_audit(summary_rows)
    _atomic_json(work_dir / "terminal_audit.json", audit)
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "implementation_version": VERSION,
        "status": "raw_complete" if audit["numerical_failure_count"] == 0 else "incomplete_numerical_failure",
        "result_status": "pilot",
        "methods": list(METHODS),
        "data_files": {key: str(path) for key, path in data_paths.items()},
        "data_sha256": {key: _sha256_file(path) for key, path in data_paths.items()},
        "offline_data": str(offline_data),
        "negative_calibration_json": str(calibration_json),
        "summary_rows": summary_rows,
        "source_provenance": source_provenance(),
    }
    _atomic_json(work_dir / "scientific_run_manifest.json", manifest)
    _atomic_json(work_dir / "RUN_COMPLETE.json", {"status": "complete", **audit})
    return 0


def cmd_selftest(_: argparse.Namespace) -> None:
    config = load_config(DEFAULT_CONFIG)
    assert tuple(config["methods"]) == METHODS
    plan = prompt_plan(5, seed=1, steps=4, prompts_per_step=3)
    assert len(plan) == 4
    row = {"id": "p0", "prompt": "Numbers: 1, 2, 3, 4\nTarget: 10", "numbers": [1, 2, 3, 4], "target": 10}
    positives, negatives, stats = verify_rollouts(
        [row],
        [["1+2+3+4", "1+2+3-4", "not an expression"]],
        max_correct_per_prompt=1,
        max_negative_per_prompt=2,
    )
    assert len(positives) == 1
    assert len(negatives) >= 1
    assert stats["sampled_completions"] == 3.0
    print("BASE_RL_REPLAY_SELFTEST_OK")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Countdown E8 base-start RL/replay diagnostic")
    parser.add_argument("--version", action="version", version=VERSION)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--model_path", required=True)
    run.add_argument("--work_dir", required=True)
    run.add_argument("--config", default=str(DEFAULT_CONFIG))
    run.add_argument("--gpu", default="0")
    run.add_argument("--skip_base_eval", dest="run_base_eval", action="store_false")
    run.add_argument("--skip_offline", dest="run_offline", action="store_false")
    run.add_argument("--skip_online", dest="run_online", action="store_false")
    run.add_argument("--skip_replay", dest="run_replay", action="store_false")
    run.set_defaults(func=cmd_run, run_base_eval=True, run_offline=True, run_online=True, run_replay=True)
    selftest = sub.add_parser("selftest")
    selftest.set_defaults(func=cmd_selftest)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = args.func(args)
    if isinstance(result, int):
        raise SystemExit(result)


if __name__ == "__main__":
    main()
