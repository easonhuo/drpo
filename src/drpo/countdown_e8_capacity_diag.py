#!/usr/bin/env python3
"""Countdown E8 on-policy capacity diagnostic pilot.

Registered experiment: ``EXT-C-E8-ONPOLICY-CAPACITY-DIAG-0.5B-01``.
This runner is a second-layer diagnostic for the single-seed same-LoRA RFT
failure mode observed in the unpolished pilot.  It compares only capacity and
continuation variants:

* lora_sft_only: LoRA SFT reference, no RFT;
* same_lora_rft: continue the SFT LoRA adapter directly;
* fresh_lora_rft: merge SFT LoRA, attach a fresh LoRA adapter, then RFT;
* full_param_rft: merge SFT LoRA, then full-parameter RFT;
* full_param_sft_only: full-parameter SFT reference, no RFT.

All RFT branches remain verifier-correct positive-only.  This runner does not
introduce taper methods, signed negative updates, or frozen off-policy replay.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
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

EXPERIMENT_ID = "EXT-C-E8-ONPOLICY-CAPACITY-DIAG-0.5B-01"
VERSION = "0.1.0-capacity-diag"
DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "countdown_e8_onpolicy_capacity_diag_0p5b.yaml"
)
RFT_BRANCHES = ("same_lora_rft", "fresh_lora_rft", "full_param_rft")
METHODS = ("lora_sft_only", *RFT_BRANCHES, "full_param_sft_only")


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
            writer.writerow({k: json.dumps(v, ensure_ascii=False, sort_keys=True) if isinstance(v, (dict, list, tuple)) else v for k, v in row.items()})


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text())
    if not isinstance(config, dict):
        raise ValueError("capacity diagnostic config must be a YAML mapping")
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("experiment_id mismatch")
    if tuple(config.get("methods") or []) != METHODS:
        raise ValueError("method set/order is frozen for the capacity diagnostic")
    if config.get("result_status") != "pilot":
        raise ValueError("capacity diagnostic must remain pilot")
    data = config["data"]
    if data.get("split_protocol") != "structural_family_holdout":
        raise ValueError("capacity diagnostic must reuse structural family-holdout split")
    if (int(data["train_rows"]), int(data["validation_rows"]), int(data["test_rows"])) != (6000, 500, 1000):
        raise ValueError("Countdown row counts are frozen at 6000/500/1000")
    guards = config["scope_guards"]
    for key in ("uses_negative_updates", "uses_taper_methods", "uses_frozen_offpolicy_replay"):
        if guards.get(key) is not False:
            raise ValueError(f"scope guard {key} must remain false")
    if guards.get("full_param_branches_are_capacity_diagnostics_only") is not True:
        raise ValueError("full-param branches must remain diagnostic-only")
    return config


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
        "git_dirty": arena.source_provenance().get("git_dirty"),
    }


def _model_plan(model_path: Path, config: Mapping[str, Any], gpu: str | int) -> dict[str, Any]:
    model_cfg = config["model"]
    return arena.resolve_execution_plan(str(model_path), "auto", str(model_cfg["memory_mode"]), 0, str(gpu))


def _sft_args(
    *,
    model_path: Path,
    output_dir: Path,
    data_paths: Mapping[str, Path],
    config: Mapping[str, Any],
    parameterization: str,
    gpu: str | int,
) -> argparse.Namespace:
    ref = config["reference"]
    model_cfg = config["model"]
    plan = _model_plan(model_path, config, gpu)
    prefix = "lora_sft" if parameterization == "lora" else "full_sft"
    return argparse.Namespace(
        model_path=str(model_path),
        train_data=str(data_paths["train"]),
        val_data=str(data_paths["validation"]),
        output_dir=str(output_dir),
        seed=int(ref[f"{prefix}_seed"]),
        max_length=int(model_cfg["max_length"]),
        max_new_tokens=int(model_cfg["max_new_tokens"]),
        epochs=int(ref[f"{prefix}_epochs"]),
        min_epochs=int(ref[f"{prefix}_min_epochs"]),
        early_stop_patience=int(ref[f"{prefix}_early_stop_patience"]),
        parameterization=parameterization,
        micro_batch=int(ref.get(f"{prefix}_micro_batch", plan["micro_batch"])),
        grad_accum=int(ref[f"{prefix}_gradient_accumulation"]),
        lr=float(ref[f"{prefix}_learning_rate"]),
        warmup_ratio=float(ref[f"{prefix}_warmup_ratio"]),
        max_grad_norm=float(ref[f"{prefix}_max_gradient_norm"]),
        num_workers=0,
        eval_examples=int(ref["validation_examples"]),
        eval_batch=int(ref.get("evaluation_batch_size", plan["eval_batch"])),
        pass_k=int(ref["pass_at_k"]),
        eval_seed=int(ref["evaluation_seed"]),
        selection_metric=str(ref["selection_metric"]),
        selection_delta=float(ref["selection_delta"]),
        log_every=int(ref["log_every_updates"]),
        load_in_4bit=False if parameterization == "full" else bool(model_cfg.get("load_in_4bit", False)),
        dtype=str(model_cfg.get("dtype", plan["dtype"])),
        result_status=str(config["result_status"]),
    )


def prepare_data_and_references(
    model_path: Path,
    work_dir: Path,
    config: Mapping[str, Any],
    *,
    gpu: str,
    sft_adapter_path: str | Path | None,
) -> dict[str, Any]:
    if gpu != "auto":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu).split(",")[0]
    data_paths = onp.generate_or_load_data(work_dir, config)
    lora_adapter = onp.train_sft_reference(model_path, work_dir, data_paths, _as_unpolished_config(config), sft_adapter_path)
    lora_eval = evaluate_checkpoint(
        model_path=model_path,
        adapter_path=lora_adapter,
        output_dir=work_dir / "methods" / "lora_sft_only",
        data_paths=data_paths,
        config=config,
        parameterization="lora",
        base_model_for_full=None,
    )
    merged_model = merge_lora_to_full_model(
        model_path,
        lora_adapter,
        work_dir / "merged_lora_sft_model",
        config,
    )
    return {
        "data_paths": {k: str(v) for k, v in data_paths.items()},
        "lora_adapter": str(lora_adapter),
        "merged_model": str(merged_model),
        "lora_sft_only": lora_eval,
    }


def _as_unpolished_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Shape the shared SFT/data helpers expect without changing semantics."""
    ref = config["reference"]
    return {
        "experiment_id": onp.EXPERIMENT_ID,
        "result_status": config["result_status"],
        "methods": list(onp.METHODS),
        "model": {"parameterization": "lora", **dict(config["model"])},
        "policy_update": {
            "uses_negative_updates": False,
            "uses_taper_methods": False,
            "uses_frozen_offpolicy_replay": False,
        },
        "data": dict(config["data"]),
        "confirmation": {"paired_training_seeds": [2026070701, 2026070702, 2026070703]},
        "onpolicy_training": dict(config["onpolicy_training"]),
        "reference": {
            "external_adapter_reuse_allowed": ref["external_lora_sft_adapter_reuse_allowed"],
            "sft_seed": ref["lora_sft_seed"],
            "sft_epochs": ref["lora_sft_epochs"],
            "sft_min_epochs": ref["lora_sft_min_epochs"],
            "sft_early_stop_patience": ref["lora_sft_early_stop_patience"],
            "sft_learning_rate": ref["lora_sft_learning_rate"],
            "sft_gradient_accumulation": ref["lora_sft_gradient_accumulation"],
            "sft_micro_batch": ref["lora_sft_micro_batch"],
            "sft_warmup_ratio": ref["lora_sft_warmup_ratio"],
            "sft_max_gradient_norm": ref["lora_sft_max_gradient_norm"],
            "validation_examples": ref["validation_examples"],
            "evaluation_batch_size": ref["evaluation_batch_size"],
            "pass_at_k": ref["pass_at_k"],
            "evaluation_seed": ref["evaluation_seed"],
            "selection_metric": ref["selection_metric"],
            "selection_delta": ref["selection_delta"],
            "log_every_updates": ref["log_every_updates"],
        },
    }


def merge_lora_to_full_model(model_path: Path, adapter_path: Path, output_dir: Path, config: Mapping[str, Any]) -> Path:
    if (output_dir / "config.json").is_file():
        return output_dir
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Refusing to reuse incomplete merged model dir: {output_dir}")
    tokenizer = arena.load_tokenizer(str(model_path))
    model = arena.load_model(
        str(model_path),
        str(adapter_path),
        trainable_adapter=False,
        load_in_4bit=False,
        dtype=str(config["model"].get("dtype", "bf16")),
        gradient_checkpointing=False,
        parameterization="lora",
    )
    if not hasattr(model, "merge_and_unload"):
        raise RuntimeError("Loaded PEFT model does not support merge_and_unload")
    merged = model.merge_and_unload()
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    _atomic_json(output_dir / "merge_manifest.json", {"source_model": str(model_path), "source_adapter": str(adapter_path)})
    del model, merged
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output_dir


def evaluate_checkpoint(
    *,
    model_path: Path,
    adapter_path: Path | None,
    output_dir: Path,
    data_paths: Mapping[str, Path],
    config: Mapping[str, Any],
    parameterization: str,
    base_model_for_full: Path | None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    load_path = base_model_for_full or model_path
    tokenizer = arena.load_tokenizer(str(load_path))
    model = arena.load_model(
        str(load_path),
        str(adapter_path) if adapter_path else None,
        trainable_adapter=False,
        load_in_4bit=False,
        dtype=str(config["model"].get("dtype", "bf16")),
        gradient_checkpointing=False,
        parameterization=parameterization,
    )
    val_rows = arena.read_jsonl(data_paths["validation"])[: int(config["evaluation"]["validation_examples"])]
    test_rows = arena.read_jsonl(data_paths["test"])[: int(config["evaluation"]["test_examples"])]
    result = {
        "validation": evaluate_pass_ks(model, tokenizer, val_rows, config, int(config["evaluation"]["seed"])),
        "test": evaluate_pass_ks(model, tokenizer, test_rows, config, int(config["evaluation"]["test_seed"])),
    }
    _atomic_json(output_dir / "evaluation.json", result)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def evaluate_pass_ks(model: Any, tokenizer: Any, rows: list[dict[str, Any]], config: Mapping[str, Any], seed: int) -> dict[str, float]:
    metrics: dict[str, float] = {}
    with onp._temporary_generation_context(model):
        for pass_k in config["evaluation"]["pass_ks"]:
            current = arena.evaluate_rows(
                model,
                tokenizer,
                rows,
                int(config["evaluation"]["batch_size"]),
                int(config["model"]["max_new_tokens"]),
                int(pass_k),
                seed + int(pass_k),
            )
            metrics[f"pass_at_{pass_k}"] = float(current["pass_at_k"])
            if int(pass_k) == int(config["evaluation"]["pass_ks"][0]):
                metrics["greedy_success"] = float(current["greedy_success"])
                metrics["valid_rate"] = float(current["valid_rate"])
                metrics["n_eval"] = float(current["n_eval"])
    return metrics


def _load_branch_model(branch: str, model_path: Path, lora_adapter: Path, merged_model: Path, config: Mapping[str, Any]) -> tuple[Any, Any, str, Path, str | None]:
    if branch == "same_lora_rft":
        tokenizer = arena.load_tokenizer(str(model_path))
        model = arena.load_model(str(model_path), str(lora_adapter), True, False, str(config["model"].get("dtype", "bf16")), True, "lora")
        return model, tokenizer, "lora", model_path, str(lora_adapter)
    if branch == "fresh_lora_rft":
        tokenizer = arena.load_tokenizer(str(merged_model))
        model = arena.load_model(str(merged_model), None, True, False, str(config["model"].get("dtype", "bf16")), True, "lora")
        return model, tokenizer, "lora", merged_model, None
    if branch == "full_param_rft":
        tokenizer = arena.load_tokenizer(str(merged_model))
        model = arena.load_model(str(merged_model), None, True, False, str(config["model"].get("dtype", "bf16")), True, "full")
        return model, tokenizer, "full", merged_model, None
    raise ValueError(f"Unknown branch: {branch}")


def _save_checkpoint(model: Any, tokenizer: Any, path: Path) -> None:
    if path.exists():
        import shutil
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


def train_rft_branch(
    *,
    branch: str,
    seed: int,
    model_path: Path,
    lora_adapter: Path,
    merged_model: Path,
    work_dir: Path,
    data_paths: Mapping[str, Path],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    method_dir = work_dir / "methods" / branch / f"seed_{seed}"
    method_dir.mkdir(parents=True, exist_ok=True)
    train_rows = arena.read_jsonl(data_paths["train"])
    val_rows = arena.read_jsonl(data_paths["validation"])[: int(config["evaluation"]["validation_examples"])]
    test_rows = arena.read_jsonl(data_paths["test"])[: int(config["evaluation"]["test_examples"])]
    model, tokenizer, parameterization, load_path, adapter_for_eval = _load_branch_model(branch, model_path, lora_adapter, merged_model, config)
    device = next(model.parameters()).device
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError(f"{branch} has no trainable parameters")
    lr_cfg = config["onpolicy_training"]["learning_rate"]
    optimizer = torch.optim.AdamW(trainable, lr=float(lr_cfg[branch]), weight_decay=0.01)
    plan = onp.prompt_attempt_plan(
        len(train_rows),
        seed=seed,
        attempts=int(config["onpolicy_training"]["sampling_attempts"]),
        prompts_per_attempt=int(config["onpolicy_training"]["prompts_per_attempt"]),
    )
    best_dir = method_dir / ("best_adapter" if parameterization == "lora" else "best_model")
    terminal_dir = method_dir / ("terminal_adapter" if parameterization == "lora" else "terminal_model")
    selected_rows: list[dict[str, Any]] = []
    training_log: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []
    optimizer_steps = skipped = total_sampled = 0
    best_value = -float("inf")
    best_attempt = 0
    eval_seed = int(config["evaluation"]["seed"]) + seed
    initial_metrics = evaluate_pass_ks(model, tokenizer, val_rows, config, eval_seed)
    metrics_rows.append({"attempt": 0, "optimizer_step": 0, **initial_metrics})
    best_value = float(initial_metrics[str(config["evaluation"]["selection_metric"])])
    _save_checkpoint(model, tokenizer, best_dir)
    numerical_failure: str | None = None
    for attempt, indices in enumerate(plan, start=1):
        rows = [train_rows[i] for i in indices]
        prompts = [row["prompt"] for row in rows]
        with onp._temporary_generation_context(model):
            grouped = arena.generate_outputs(
                model, tokenizer, prompts, int(config["model"]["max_new_tokens"]), True,
                float(config["onpolicy_training"]["sampling_temperature"]),
                float(config["onpolicy_training"]["sampling_top_p"]),
                int(config["onpolicy_training"]["rollouts_per_prompt"]),
            )
        total_sampled += sum(len(group) for group in grouped)
        selected: list[dict[str, Any]] = []
        for row, completions in zip(rows, grouped):
            selected.extend(onp.select_correct_completions(row, completions, max_per_prompt=int(config["onpolicy_training"]["max_correct_per_prompt"])))
        if selected:
            batch = onp._collate_positive_examples(tokenizer, selected, max_length=int(config["model"]["max_length"]))
            batch = arena.move_to_device(batch, device)
            model.train()
            loss = model(**batch, use_cache=False).loss
            if not bool(torch.isfinite(loss)):
                numerical_failure = f"nonfinite_loss_at_attempt_{attempt}"
                break
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, float(config["onpolicy_training"]["max_gradient_norm"]))
            if not bool(torch.isfinite(grad_norm)):
                numerical_failure = f"nonfinite_gradient_at_attempt_{attempt}"
                break
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            if not onp._trainable_parameters_finite(trainable):
                numerical_failure = f"nonfinite_parameters_at_attempt_{attempt}"
                break
            selected_rows.extend({"attempt": attempt, **item} for item in selected)
            log_row = {"attempt": attempt, "selected": len(selected), "loss": float(loss.detach()), "grad_norm": float(grad_norm), "optimizer_step": optimizer_steps}
        else:
            skipped += 1
            log_row = {"attempt": attempt, "selected": 0, "loss": None, "grad_norm": None, "optimizer_step": optimizer_steps}
        training_log.append(log_row)
        if attempt % int(config["onpolicy_training"]["log_every_attempts"]) == 0:
            print(json.dumps({"event": "branch_progress", "branch": branch, "seed": seed, **log_row}), flush=True)
        if attempt % int(config["onpolicy_training"]["eval_every_attempts"]) == 0:
            metrics = evaluate_pass_ks(model, tokenizer, val_rows, config, eval_seed + attempt)
            metrics_rows.append({"attempt": attempt, "optimizer_step": optimizer_steps, **metrics})
            value = float(metrics[str(config["evaluation"]["selection_metric"])])
            if value > best_value + float(config["evaluation"]["selection_delta"]):
                best_value = value
                best_attempt = attempt
                _save_checkpoint(model, tokenizer, best_dir)
    if numerical_failure is None:
        _save_checkpoint(model, tokenizer, terminal_dir)
    selected_path = method_dir / "selected_positives.jsonl"
    arena.write_jsonl(selected_path, selected_rows)
    _csv_write(method_dir / "training_log.csv", training_log)
    _csv_write(method_dir / "metrics.csv", metrics_rows)
    terminal_eval = None if numerical_failure else {
        "validation": evaluate_pass_ks(model, tokenizer, val_rows, config, eval_seed + 9000),
        "test": evaluate_pass_ks(model, tokenizer, test_rows, config, int(config["evaluation"]["test_seed"]) + seed),
    }
    summary = {
        "method": branch,
        "seed": seed,
        "parameterization": parameterization,
        "sampling_attempts": len(plan),
        "optimizer_steps": optimizer_steps,
        "skipped_attempts": skipped,
        "sampled_completions": total_sampled,
        "correct_selected": len(selected_rows),
        "exploration": exploration_summary(selected_rows, total_sampled),
        "best_attempt": best_attempt,
        "best_validation_value": best_value,
        "terminal_evaluation": terminal_eval,
        "numerical_failure": numerical_failure,
        "status": "finite_budget_complete" if numerical_failure is None else "numerical_failure",
        "load_path": str(load_path),
        "initial_adapter_path": adapter_for_eval,
    }
    _atomic_json(method_dir / "summary.json", summary)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def exploration_summary(selected_rows: Sequence[Mapping[str, Any]], sampled_completions: int) -> dict[str, Any]:
    expressions = ["".join(str(row["completion"]).split()) for row in selected_rows]
    by_prompt: dict[str, set[str]] = {}
    for row, expression in zip(selected_rows, expressions):
        by_prompt.setdefault(str(row.get("prompt_id")), set()).add(expression)
    per_prompt_counts = [len(values) for values in by_prompt.values()]
    return {
        "selected_correct_total": len(expressions),
        "sampled_completions": sampled_completions,
        "selected_correct_rate": len(expressions) / max(sampled_completions, 1),
        "unique_correct_expressions": len(set(expressions)),
        "unique_correct_ratio": len(set(expressions)) / max(len(expressions), 1),
        "prompts_with_selected_correct": len(by_prompt),
        "mean_unique_correct_per_prompt": sum(per_prompt_counts) / max(len(per_prompt_counts), 1),
    }


def train_full_sft_only(model_path: Path, work_dir: Path, data_paths: Mapping[str, Path], config: Mapping[str, Any], gpu: str) -> dict[str, Any]:
    if gpu != "auto":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    out = work_dir / "methods" / "full_param_sft_only" / "sft"
    args = _sft_args(model_path=model_path, output_dir=out, data_paths=data_paths, config=config, parameterization="full", gpu=gpu)
    if not (out / "best_model" / "config.json").is_file():
        arena.cmd_sft(args)
    best = out / "best_model"
    eval_result = evaluate_checkpoint(model_path=best, adapter_path=None, output_dir=work_dir / "methods" / "full_param_sft_only", data_paths=data_paths, config=config, parameterization="full", base_model_for_full=best)
    summary = {"method": "full_param_sft_only", "checkpoint": str(best), "evaluation": eval_result, "status": "complete"}
    _atomic_json(work_dir / "methods" / "full_param_sft_only" / "summary.json", summary)
    return summary


def branch_jobs(config: Mapping[str, Any], gpu_ids: Sequence[str]) -> list[dict[str, Any]]:
    seeds = [int(x) for x in config["onpolicy_training"]["first_round_seeds"]]
    branches = [*RFT_BRANCHES, "full_param_sft_only"]
    jobs = []
    for index, branch in enumerate(branches):
        for seed in (seeds if branch in RFT_BRANCHES else [0]):
            jobs.append({"branch": branch, "seed": seed, "gpu": str(gpu_ids[index % len(gpu_ids)])})
    return jobs


def _run_parallel_jobs(args: argparse.Namespace, prep: Mapping[str, Any], config: Mapping[str, Any]) -> list[dict[str, Any]]:
    gpu_ids = [x.strip() for x in str(args.gpu_ids).split(",") if x.strip()]
    if not gpu_ids:
        gpu_ids = [str(x) for x in config["parallel"]["default_gpu_ids"]]
    jobs = branch_jobs(config, gpu_ids)
    max_workers = min(int(config["parallel"]["max_workers"]), len(jobs), len(gpu_ids))
    pending = list(jobs)
    running: list[tuple[subprocess.Popen[Any], dict[str, Any]]] = []
    completed: list[dict[str, Any]] = []
    while pending or running:
        while pending and len(running) < max_workers:
            job = pending.pop(0)
            cmd = [
                sys.executable, str(Path(__file__).resolve()), "worker",
                "--branch", job["branch"], "--seed", str(job["seed"]),
                "--model_path", str(Path(args.model_path).resolve()),
                "--work_dir", str(Path(args.work_dir).resolve()),
                "--config", str(Path(args.config).resolve()),
                "--lora_adapter", str(prep["lora_adapter"]),
                "--merged_model", str(prep["merged_model"]),
                "--gpu", job["gpu"],
            ]
            log = Path(args.work_dir).resolve() / "logs" / f"worker_{job['branch']}_{job['seed']}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            handle = log.open("w")
            proc = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT)
            running.append((proc, {**job, "log": str(log), "handle": handle}))
        still = []
        for proc, job in running:
            code = proc.poll()
            if code is None:
                still.append((proc, job))
                continue
            job["handle"].close()
            completed.append({k: v for k, v in job.items() if k != "handle"} | {"returncode": code})
            if code != 0:
                raise RuntimeError(f"Worker failed: {job}; see {job['log']}")
        running = still
        if running:
            import time
            time.sleep(5)
    return completed


def cmd_worker(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    model_path = Path(args.model_path).resolve()
    work_dir = Path(args.work_dir).resolve()
    data_paths = {k: Path(v) for k, v in onp.generate_or_load_data(work_dir, config).items()}
    if args.gpu != "auto":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if args.branch == "full_param_sft_only":
        summary = train_full_sft_only(model_path, work_dir, data_paths, config, args.gpu)
    else:
        summary = train_rft_branch(
            branch=args.branch,
            seed=int(args.seed),
            model_path=model_path,
            lora_adapter=Path(args.lora_adapter).resolve(),
            merged_model=Path(args.merged_model).resolve(),
            work_dir=work_dir,
            data_paths=data_paths,
            config=config,
        )
    print(json.dumps({"event": "worker_complete", "summary": summary}), flush=True)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    model_path = Path(args.model_path).resolve()
    work_dir = Path(args.work_dir).resolve()
    if not model_path.is_dir():
        raise SystemExit(f"Model directory does not exist: {model_path}")
    if (work_dir / "RUN_COMPLETE.json").exists():
        raise RuntimeError("work_dir already contains RUN_COMPLETE.json")
    work_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(work_dir / "run_config.json", {"experiment_id": EXPERIMENT_ID, "config": config, "source_provenance": source_provenance()})
    prep = prepare_data_and_references(model_path, work_dir, config, gpu=str(args.gpu_ids).split(",")[0], sft_adapter_path=args.sft_adapter_path)
    _atomic_json(work_dir / "preparation_manifest.json", prep)
    completed = _run_parallel_jobs(args, prep, config) if bool(config["parallel"]["enabled"]) and not args.serial else []
    if args.serial:
        completed = []
        for job in branch_jobs(config, [str(args.gpu_ids).split(",")[0]]):
            worker_args = argparse.Namespace(**vars(args), branch=job["branch"], seed=job["seed"], lora_adapter=prep["lora_adapter"], merged_model=prep["merged_model"], gpu=job["gpu"])
            cmd_worker(worker_args)
            completed.append({**job, "returncode": 0})
    summaries = [prep["lora_sft_only"]]
    for summary_file in (work_dir / "methods").glob("*/*/summary.json"):
        summaries.append(json.loads(summary_file.read_text()))
    fps = work_dir / "methods" / "full_param_sft_only" / "summary.json"
    if fps.exists():
        summaries.append(json.loads(fps.read_text()))
    _csv_write(work_dir / "capacity_diag_summary.csv", [{"method": s.get("method"), "status": s.get("status"), "summary_path": str(i)} for i, s in enumerate(summaries)])
    audit = {
        "experiment_id": EXPERIMENT_ID,
        "status": "pilot_raw_complete",
        "result_interpretation_limit": "single-seed first-round capacity diagnostic unless confirmation seeds are explicitly run",
        "parallel_workers": completed,
        "methods": list(METHODS),
        "negative_updates_used": False,
        "taper_methods_used": False,
        "frozen_offpolicy_replay_used": False,
        "summaries": summaries,
    }
    _atomic_json(work_dir / "terminal_audit.json", audit)
    _atomic_json(work_dir / "scientific_run_manifest.json", {"experiment_id": EXPERIMENT_ID, "status": "raw_complete", "source_provenance": source_provenance(), "preparation": prep, "workers": completed})
    _atomic_json(work_dir / "RUN_COMPLETE.json", {"status": "complete", **audit})
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Countdown E8 on-policy capacity diagnostic")
    parser.add_argument("--version", action="version", version=VERSION)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--model_path", required=True)
    run.add_argument("--work_dir", required=True)
    run.add_argument("--config", default=str(DEFAULT_CONFIG))
    run.add_argument("--gpu_ids", default="0,1,2,3")
    run.add_argument("--serial", action="store_true")
    run.add_argument("--sft_adapter_path", default=None)
    run.set_defaults(func=cmd_run)
    worker = sub.add_parser("worker")
    worker.add_argument("--branch", required=True, choices=[*RFT_BRANCHES, "full_param_sft_only"])
    worker.add_argument("--seed", type=int, default=0)
    worker.add_argument("--model_path", required=True)
    worker.add_argument("--work_dir", required=True)
    worker.add_argument("--config", default=str(DEFAULT_CONFIG))
    worker.add_argument("--lora_adapter", required=True)
    worker.add_argument("--merged_model", required=True)
    worker.add_argument("--gpu", default="0")
    worker.set_defaults(func=cmd_worker)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
