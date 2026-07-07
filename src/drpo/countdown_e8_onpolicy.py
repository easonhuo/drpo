#!/usr/bin/env python3
"""Countdown E8 on-policy RFT unpolished pilot.

This runner implements the registered ``EXT-C-E8-ONPOLICY-UNPOLISHED-0.5B-01``
first-step diagnostic. It intentionally keeps the method set small:

* ``sft_only``: evaluate the frozen SFT/reference adapter without continuation;
* ``onpolicy_rft_positive_only``: continue the same SFT LoRA adapter by sampling
  from the current policy, verifying Countdown completions, and applying a
  supervised loss only to correct sampled expressions.

The experiment is a capacity / trainability diagnostic for Countdown 0.5B. It is
not a DRPO/taper method-ranking experiment, does not use frozen off-policy replay,
and does not replace C-U1 or D-U1 controlled mechanism identification. Fixed
sampling attempts are reported as finite-budget pilot evidence, not convergence.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from contextlib import contextmanager
import math
import os
import random
import shutil
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch
import yaml

try:  # direct script execution and package import are both supported
    from drpo import countdown_qwen_arena_onefile as arena
except ImportError:  # pragma: no cover - direct execution from src/drpo
    import countdown_qwen_arena_onefile as arena  # type: ignore


EXPERIMENT_ID = "EXT-C-E8-ONPOLICY-UNPOLISHED-0.5B-01"
VERSION = "0.2.0-unpolished-lora-rft"
DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "countdown_e8_onpolicy_0p5b_unpolished.yaml"
)
METHODS = ("sft_only", "onpolicy_rft_positive_only")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    tmp.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def source_provenance() -> dict[str, Any]:
    """Record both this runner and its shared Countdown dependency."""
    runner = Path(__file__).resolve()
    shared = Path(arena.__file__).resolve()
    inherited = arena.source_provenance()
    return {
        "experiment_id": EXPERIMENT_ID,
        "implementation_version": VERSION,
        "runner_source_file": str(runner),
        "runner_source_sha256": _sha256_file(runner),
        "shared_arena_source_file": str(shared),
        "shared_arena_source_sha256": _sha256_file(shared),
        "git_commit": inherited.get("git_commit"),
        "git_branch": inherited.get("git_branch"),
        "git_dirty": inherited.get("git_dirty"),
        "repository_root": inherited.get("repository_root"),
    }


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = Path(path).resolve()
    value = yaml.safe_load(config_path.read_text())
    if not isinstance(value, dict):
        raise ValueError("Countdown E8 on-policy config must be a YAML mapping")
    if value.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("Config experiment_id does not match the registered experiment")
    methods = tuple(value.get("methods") or [])
    if methods != METHODS:
        raise ValueError(f"Frozen method order mismatch: {methods!r}")
    if value.get("result_status") != "pilot":
        raise ValueError("Unpolished on-policy run must remain a pilot")
    if value["model"].get("parameterization") != "lora":
        raise ValueError("The unpolished diagnostic keeps LoRA for offline comparability")
    if value["policy_update"].get("uses_negative_updates") is not False:
        raise ValueError("First unpolished on-policy run must be positive-only RFT")
    if value["policy_update"].get("uses_taper_methods") is not False:
        raise ValueError("First unpolished on-policy run must not include taper methods")
    if value["policy_update"].get("uses_frozen_offpolicy_replay") is not False:
        raise ValueError("First unpolished on-policy run must not use frozen replay")
    if int(value["data"]["train_rows"]) != 6000:
        raise ValueError("Frozen Countdown oracle train rows must remain 6000")
    if int(value["data"]["validation_rows"]) != 500:
        raise ValueError("Frozen Countdown validation rows must remain 500")
    if int(value["data"]["test_rows"]) != 1000:
        raise ValueError("Frozen Countdown test rows must remain 1000")
    if value["data"].get("split_protocol") != "structural_family_holdout":
        raise ValueError("On-policy diagnostic must reuse the current structural split protocol")
    seeds = tuple(value["confirmation"]["paired_training_seeds"])
    if seeds != (2026070701, 2026070702, 2026070703):
        raise ValueError(f"Frozen on-policy seed mismatch: {seeds!r}")
    training = value["onpolicy_training"]
    if int(training["sampling_attempts"]) <= 0:
        raise ValueError("onpolicy_training.sampling_attempts must be positive")
    if int(training["rollouts_per_prompt"]) <= 0:
        raise ValueError("onpolicy_training.rollouts_per_prompt must be positive")
    if int(training["prompts_per_attempt"]) <= 0:
        raise ValueError("onpolicy_training.prompts_per_attempt must be positive")
    if float(training["learning_rate"]) <= 0:
        raise ValueError("onpolicy_training.learning_rate must be positive")
    return value


def generate_or_load_data(work_dir: Path, config: Mapping[str, Any]) -> dict[str, Path]:
    """Create the same structural family-holdout Countdown split used by E8.

    Existing partially generated data are rejected rather than silently mixing
    split protocols.  The split manifest is part of the scientific provenance.
    """
    data_dir = work_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": data_dir / "train.jsonl",
        "validation": data_dir / "validation.jsonl",
        "test": data_dir / "test.jsonl",
        "split_manifest": data_dir / "split_manifest.json",
    }
    data_files = [paths["train"], paths["validation"], paths["test"]]
    existing = [path for path in [*data_files, paths["split_manifest"]] if path.exists()]
    if existing:
        if all(path.is_file() for path in data_files) and paths["split_manifest"].is_file():
            return paths
        raise RuntimeError(
            "Countdown data directory is partially populated or lacks split_manifest.json; "
            "use a new work_dir or provide a complete structural split."
        )
    train_rows, val_rows, test_rows, manifest = arena.generate_structural_splits(
        int(config["data"]["train_rows"]),
        int(config["data"]["validation_rows"]),
        int(config["data"]["test_rows"]),
        int(config["data"]["generation_seed"]),
        int(config["data"]["numbers_per_problem"]),
    )
    arena.write_jsonl(paths["train"], train_rows)
    arena.write_jsonl(paths["validation"], val_rows)
    arena.write_jsonl(paths["test"], test_rows)
    paths["split_manifest"].write_text(json.dumps(manifest, indent=2))
    return paths


def resolve_adapter_path(path: str | Path) -> Path:
    """Resolve either an adapter directory or a parent containing best_adapter."""
    root = Path(path).expanduser().resolve()
    if (root / "adapter_config.json").is_file():
        return root
    best = root / "best_adapter"
    if (best / "adapter_config.json").is_file():
        return best
    raise FileNotFoundError(
        f"Expected a LoRA adapter directory or parent with best_adapter: {root}"
    )


def adapter_manifest_summary(adapter_path: Path) -> dict[str, Any]:
    parent_manifest = adapter_path.parent / "sft_manifest.json"
    checkpoint_manifest = adapter_path.parent / "checkpoint_manifest.json"
    summary: dict[str, Any] = {
        "adapter_path": str(adapter_path),
        "adapter_config_sha256": _sha256_file(adapter_path / "adapter_config.json"),
        "parent_sft_manifest": str(parent_manifest) if parent_manifest.exists() else None,
        "parent_checkpoint_manifest": str(checkpoint_manifest) if checkpoint_manifest.exists() else None,
    }
    if parent_manifest.exists():
        try:
            manifest = json.loads(parent_manifest.read_text())
            summary["sft_manifest_subset"] = {
                key: manifest.get(key)
                for key in (
                    "model_path",
                    "parameterization",
                    "result_status",
                    "best_epoch",
                    "best_value",
                    "stop_reason",
                    "numerical_failure",
                )
            }
        except Exception as exc:  # pragma: no cover - malformed external provenance
            summary["sft_manifest_parse_error"] = str(exc)
    return summary


def train_sft_reference(
    model_path: Path,
    work_dir: Path,
    data_paths: Mapping[str, Path],
    config: Mapping[str, Any],
    external_adapter_path: str | Path | None = None,
) -> Path:
    if external_adapter_path is not None:
        if config["reference"].get("external_adapter_reuse_allowed") is not True:
            raise RuntimeError("External SFT adapter reuse is disabled by config")
        adapter = resolve_adapter_path(external_adapter_path)
        _atomic_json(
            work_dir / "reference_adapter_reuse_manifest.json",
            {
                "reuse_mode": "explicit_external_sft_lora_adapter",
                "adapter": adapter_manifest_summary(adapter),
                "model_path": str(model_path),
                "data_files": {key: str(value) for key, value in data_paths.items()},
                "data_sha256": {key: _sha256_file(value) for key, value in data_paths.items()},
                "safety_note": (
                    "The runner verifies that the supplied path is a loadable LoRA adapter "
                    "directory. Scientific comparability still requires the operator to use "
                    "an adapter trained on the same model identity and structural split."
                ),
            },
        )
        return adapter

    reference_dir = work_dir / "reference_adapter"
    existing_best = reference_dir / "best_adapter"
    if (existing_best / "adapter_config.json").is_file():
        return existing_best
    if reference_dir.exists() and any(reference_dir.iterdir()):
        raise RuntimeError(
            "reference_adapter exists but does not contain best_adapter/adapter_config.json; "
            "use a new work_dir or remove the incomplete reference output."
        )
    model_cfg = config["model"]
    reference_cfg = config["reference"]
    plan = arena.resolve_execution_plan(
        str(model_path),
        "auto",
        str(model_cfg["memory_mode"]),
        0,
        os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
    )
    args = argparse.Namespace(
        model_path=str(model_path),
        train_data=str(data_paths["train"]),
        val_data=str(data_paths["validation"]),
        output_dir=str(reference_dir),
        seed=int(reference_cfg["sft_seed"]),
        max_length=int(model_cfg["max_length"]),
        max_new_tokens=int(model_cfg["max_new_tokens"]),
        epochs=int(reference_cfg["sft_epochs"]),
        min_epochs=int(reference_cfg["sft_min_epochs"]),
        early_stop_patience=int(reference_cfg["sft_early_stop_patience"]),
        parameterization="lora",
        micro_batch=int(reference_cfg.get("sft_micro_batch", plan["micro_batch"])),
        grad_accum=int(reference_cfg["sft_gradient_accumulation"]),
        lr=float(reference_cfg["sft_learning_rate"]),
        warmup_ratio=float(reference_cfg["sft_warmup_ratio"]),
        max_grad_norm=float(reference_cfg["sft_max_gradient_norm"]),
        num_workers=int(reference_cfg.get("num_workers", 0)),
        eval_examples=int(reference_cfg["validation_examples"]),
        eval_batch=int(reference_cfg.get("evaluation_batch_size", plan["eval_batch"])),
        pass_k=int(reference_cfg["pass_at_k"]),
        eval_seed=int(reference_cfg["evaluation_seed"]),
        selection_metric=str(reference_cfg["selection_metric"]),
        selection_delta=float(reference_cfg["selection_delta"]),
        log_every=int(reference_cfg["log_every_updates"]),
        load_in_4bit=bool(model_cfg.get("load_in_4bit", plan["load_in_4bit"])),
        dtype=str(model_cfg.get("dtype", plan["dtype"])),
        result_status=str(config["result_status"]),
    )
    arena.cmd_sft(args)
    return existing_best


def prompt_attempt_plan(
    num_rows: int,
    *,
    seed: int,
    attempts: int,
    prompts_per_attempt: int,
) -> list[list[int]]:
    if num_rows <= 0:
        raise ValueError("num_rows must be positive")
    if attempts <= 0:
        raise ValueError("attempts must be positive")
    if prompts_per_attempt <= 0:
        raise ValueError("prompts_per_attempt must be positive")
    rng = random.Random(seed)
    indices = list(range(num_rows))
    plan: list[list[int]] = []
    cursor = 0
    for _ in range(attempts):
        chunk: list[int] = []
        while len(chunk) < prompts_per_attempt:
            if cursor == 0:
                rng.shuffle(indices)
            take = min(prompts_per_attempt - len(chunk), len(indices) - cursor)
            chunk.extend(indices[cursor : cursor + take])
            cursor = (cursor + take) % len(indices)
        plan.append(chunk)
    return plan


def select_correct_completions(
    row: Mapping[str, Any],
    completions: Sequence[str],
    *,
    max_per_prompt: int,
) -> list[dict[str, Any]]:
    """Return unique verifier-correct expressions sampled for one prompt."""
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for text in completions:
        check = arena.verify_expression(text, row["numbers"], int(row["target"]))
        expression = str(check.get("expression") or "")
        duplicate_key = "".join(expression.split())
        if not expression or duplicate_key in seen:
            continue
        seen.add(duplicate_key)
        if not bool(check.get("correct")):
            continue
        selected.append(
            {
                "prompt_id": row.get("id"),
                "prompt": row["prompt"],
                "completion": expression,
                "numbers": row["numbers"],
                "target": row["target"],
            }
        )
        if len(selected) >= max_per_prompt:
            break
    return selected


def _collate_positive_examples(
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    max_length: int,
) -> dict[str, torch.Tensor]:
    encoded = [
        arena.encode_prompt_completion(
            tokenizer, str(row["prompt"]), str(row["completion"]), max_length
        )
        for row in rows
    ]
    return arena.pad_encoded(encoded, tokenizer.pad_token_id)


def _set_model_use_cache(model: Any, value: bool | None) -> bool | None:
    config = getattr(model, "config", None)
    if config is None or not hasattr(config, "use_cache"):
        return None
    previous = bool(getattr(config, "use_cache"))
    if value is not None:
        setattr(config, "use_cache", bool(value))
    return previous


def _checkpointing_enabled(model: Any) -> bool:
    for candidate in (model, getattr(model, "base_model", None), getattr(model, "model", None)):
        value = getattr(candidate, "is_gradient_checkpointing", None)
        if isinstance(value, bool):
            return value
        if callable(value):
            return bool(value())
    return False


def _call_checkpointing(model: Any, method_name: str) -> bool:
    for candidate in (model, getattr(model, "base_model", None), getattr(model, "model", None)):
        method = getattr(candidate, method_name, None)
        if callable(method):
            method()
            return True
    return False


@contextmanager
def _temporary_generation_context(model: Any) -> Iterator[None]:
    """Use fast generation settings without leaking them into training."""
    was_training = bool(getattr(model, "training", False))
    previous_use_cache = _set_model_use_cache(model, True)
    disabled_checkpointing = False
    if _checkpointing_enabled(model):
        disabled_checkpointing = _call_checkpointing(model, "gradient_checkpointing_disable")
    try:
        model.eval()
        yield
    finally:
        if previous_use_cache is not None:
            _set_model_use_cache(model, previous_use_cache)
        if disabled_checkpointing:
            _call_checkpointing(model, "gradient_checkpointing_enable")
            if was_training and hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
        if was_training:
            model.train()
        else:
            model.eval()


def _evaluate_model(
    model: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    seed: int,
) -> dict[str, float]:
    eval_cfg = config["evaluation"]
    model_cfg = config["model"]
    eval_rows = list(rows)[: int(eval_cfg["examples"])]
    batch_size = int(eval_cfg["batch_size"])
    total_batches = math.ceil(len(eval_rows) / max(batch_size, 1))
    weighted = {"greedy_success": 0.0, "pass_at_k": 0.0, "valid_rate": 0.0}
    print(
        json.dumps(
            {
                "stage": "evaluation",
                "event": "start",
                "n_eval": len(eval_rows),
                "batch_size": batch_size,
                "pass_k": int(eval_cfg["pass_at_k"]),
                "total_batches": total_batches,
            }
        ),
        flush=True,
    )
    with _temporary_generation_context(model):
        for batch_index, start in enumerate(range(0, len(eval_rows), batch_size), start=1):
            chunk = eval_rows[start : start + batch_size]
            metrics = arena.evaluate_rows(
                model,
                tokenizer,
                chunk,
                batch_size,
                int(model_cfg["max_new_tokens"]),
                int(eval_cfg["pass_at_k"]),
                seed + batch_index,
            )
            n_eval = float(metrics["n_eval"])
            for key in weighted:
                weighted[key] += float(metrics[key]) * n_eval
            processed = start + len(chunk)
            if batch_index % 10 == 0 or batch_index == total_batches:
                print(
                    json.dumps(
                        {
                            "stage": "evaluation",
                            "event": "progress",
                            "batch": batch_index,
                            "total_batches": total_batches,
                            "processed": processed,
                        }
                    ),
                    flush=True,
                )
    denom = float(max(len(eval_rows), 1))
    result = {key: value / denom for key, value in weighted.items()}
    result["n_eval"] = float(len(eval_rows))
    print(json.dumps({"stage": "evaluation", "event": "complete", **result}), flush=True)
    return result


def _cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    def factor(step: int) -> float:
        if total_steps <= 0:
            return 1.0
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _load_policy(
    model_path: Path,
    adapter_path: Path,
    config: Mapping[str, Any],
    *,
    trainable_adapter: bool,
    gradient_checkpointing: bool,
) -> Any:
    model_cfg = config["model"]
    return arena.load_model(
        str(model_path),
        str(adapter_path),
        trainable_adapter=trainable_adapter,
        load_in_4bit=bool(model_cfg["load_in_4bit"]),
        dtype=str(model_cfg["dtype"]),
        gradient_checkpointing=gradient_checkpointing,
        parameterization="lora",
    )


def evaluate_sft_only(
    model_path: Path,
    adapter_path: Path,
    work_dir: Path,
    data_paths: Mapping[str, Path],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    method_dir = work_dir / "methods" / "sft_only"
    method_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = arena.load_tokenizer(str(model_path))
    model = _load_policy(
        model_path,
        adapter_path,
        config,
        trainable_adapter=False,
        gradient_checkpointing=False,
    )
    val_rows = arena.read_jsonl(data_paths["validation"])
    test_rows = arena.read_jsonl(data_paths["test"])
    val_metrics = _evaluate_model(
        model, tokenizer, val_rows, config, int(config["evaluation"]["seed"])
    )
    test_metrics = _evaluate_model(
        model, tokenizer, test_rows, config, int(config["evaluation"]["test_seed"])
    )
    summary = {
        "method": "sft_only",
        "stage": "no_continuation",
        "validation": val_metrics,
        "test": test_metrics,
        "checkpoint_role": "reference_adapter",
        "adapter_path": str(adapter_path),
    }
    _atomic_json(method_dir / "evaluation.json", summary)
    _csv_write(
        method_dir / "metrics.csv",
        [
            {"split": "validation", **val_metrics},
            {"split": "test", **test_metrics},
        ],
    )
    return summary


def _save_adapter(model: Any, tokenizer: Any, output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def _evaluate_saved_adapter(
    model_path: Path,
    adapter_path: Path,
    data_paths: Mapping[str, Path],
    config: Mapping[str, Any],
    seed: int,
    test_seed: int,
) -> dict[str, dict[str, float]]:
    tokenizer = arena.load_tokenizer(str(model_path))
    model = _load_policy(
        model_path,
        adapter_path,
        config,
        trainable_adapter=False,
        gradient_checkpointing=False,
    )
    val_rows = arena.read_jsonl(data_paths["validation"])
    test_rows = arena.read_jsonl(data_paths["test"])
    return {
        "validation": _evaluate_model(model, tokenizer, val_rows, config, seed),
        "test": _evaluate_model(model, tokenizer, test_rows, config, test_seed),
    }


def train_onpolicy_positive_only(
    model_path: Path,
    adapter_path: Path,
    work_dir: Path,
    data_paths: Mapping[str, Path],
    config: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    method = "onpolicy_rft_positive_only"
    method_dir = work_dir / "methods" / method / f"seed_{seed}"
    method_dir.mkdir(parents=True, exist_ok=True)
    arena.seed_all(seed)
    tokenizer = arena.load_tokenizer(str(model_path))
    model_cfg = config["model"]
    train_cfg = config["onpolicy_training"]
    eval_cfg = config["evaluation"]
    model = _load_policy(
        model_path,
        adapter_path,
        config,
        trainable_adapter=True,
        gradient_checkpointing=True,
    )
    device = next(model.parameters()).device
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("On-policy RFT has no trainable LoRA parameters")
    optimizer = torch.optim.AdamW(
        trainable, lr=float(train_cfg["learning_rate"]), weight_decay=float(train_cfg["weight_decay"])
    )
    scheduler = _cosine_warmup_scheduler(
        optimizer,
        int(int(train_cfg["sampling_attempts"]) * float(train_cfg["warmup_ratio"])),
        int(train_cfg["sampling_attempts"]),
    )
    train_rows_all = arena.read_jsonl(data_paths["train"])
    prompt_pool = train_rows_all[: int(train_cfg["prompt_pool_rows"])]
    val_rows = arena.read_jsonl(data_paths["validation"])
    test_rows = arena.read_jsonl(data_paths["test"])
    plan = prompt_attempt_plan(
        len(prompt_pool),
        seed=seed,
        attempts=int(train_cfg["sampling_attempts"]),
        prompts_per_attempt=int(train_cfg["prompts_per_attempt"]),
    )
    metrics_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    best_value = -float("inf")
    best_attempt = 0
    best_dir = method_dir / "best_adapter"
    terminal_dir = method_dir / "terminal_adapter"
    last_finite_dir = method_dir / "last_finite_adapter"
    if best_dir.exists():
        shutil.rmtree(best_dir)
    initial_metrics = _evaluate_model(
        model, tokenizer, val_rows, config, int(eval_cfg["seed"]) + seed
    )
    metrics_rows.append({"attempt": 0, "optimizer_step": 0, **initial_metrics})
    best_value = float(initial_metrics[str(eval_cfg["selection_metric"])])
    best_attempt = 0
    _save_adapter(model, tokenizer, best_dir)
    model.train()
    optimizer_steps = 0
    total_correct = 0
    total_sampled = 0
    skipped_attempts = 0
    numerical_failure: str | None = None
    failure_attempt: int | None = None

    for attempt, prompt_indices in enumerate(plan, start=1):
        rows = [prompt_pool[index] for index in prompt_indices]
        with _temporary_generation_context(model):
            outputs = arena.generate_outputs(
                model,
                tokenizer,
                [row["prompt"] for row in rows],
                int(model_cfg["max_new_tokens"]),
                True,
                float(train_cfg["temperature"]),
                float(train_cfg["top_p"]),
                int(train_cfg["rollouts_per_prompt"]),
            )
        positives: list[dict[str, Any]] = []
        prompts_with_positive = 0
        for row, completions in zip(rows, outputs):
            selected = select_correct_completions(
                row, completions, max_per_prompt=int(train_cfg["max_correct_per_prompt"])
            )
            if selected:
                prompts_with_positive += 1
            positives.extend(selected)
            total_correct += len(selected)
            total_sampled += len(completions)
        log_row: dict[str, Any] = {
            "attempt": attempt,
            "optimizer_step_before": optimizer_steps,
            "sampled_completions": sum(len(item) for item in outputs),
            "correct_selected": len(positives),
            "usable_prompt_fraction": prompts_with_positive / max(len(outputs), 1),
            "learning_rate": scheduler.get_last_lr()[0],
        }
        if positives:
            batch = _collate_positive_examples(
                tokenizer, positives, max_length=int(model_cfg["max_length"])
            )
            batch = arena.move_to_device(batch, device)
            out = model(**batch, use_cache=False)
            loss = out.loss
            if not bool(torch.isfinite(loss)):
                numerical_failure = f"nonfinite_loss_at_attempt_{attempt}"
                failure_attempt = attempt
                log_row.update({"skipped": False, "numerical_failure": numerical_failure})
                training_rows.append(log_row)
                break
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable, float(train_cfg["maximum_gradient_norm"])
            )
            if not bool(torch.isfinite(grad_norm)):
                numerical_failure = f"nonfinite_gradient_at_attempt_{attempt}"
                failure_attempt = attempt
                log_row.update({"skipped": False, "numerical_failure": numerical_failure})
                training_rows.append(log_row)
                break
            applied = arena.optimizer_step_with_last_finite_guard(optimizer, trainable)
            if not applied:
                numerical_failure = f"nonfinite_parameters_at_attempt_{attempt}"
                failure_attempt = attempt
                log_row.update({"skipped": False, "numerical_failure": numerical_failure})
                training_rows.append(log_row)
                break
            scheduler.step()
            optimizer_steps += 1
            log_row.update(
                {
                    "optimizer_step_after": optimizer_steps,
                    "loss": float(loss.detach().cpu()),
                    "grad_norm_before_clip": float(grad_norm.detach().cpu()),
                    "skipped": False,
                    "numerical_failure": None,
                }
            )
        else:
            skipped_attempts += 1
            log_row.update(
                {
                    "optimizer_step_after": optimizer_steps,
                    "loss": None,
                    "grad_norm_before_clip": None,
                    "skipped": True,
                    "numerical_failure": None,
                }
            )
        training_rows.append(log_row)
        if attempt % int(train_cfg["log_every_attempts"]) == 0:
            print(json.dumps({"method": method, "seed": seed, **log_row}), flush=True)
        if attempt % int(eval_cfg["every_attempts"]) == 0:
            model.eval()
            metrics = _evaluate_model(
                model, tokenizer, val_rows, config, int(eval_cfg["seed"]) + seed
            )
            row = {"attempt": attempt, "optimizer_step": optimizer_steps, **metrics}
            metrics_rows.append(row)
            value = float(metrics[str(eval_cfg["selection_metric"])])
            if value > best_value + float(eval_cfg["selection_delta"]):
                best_value = value
                best_attempt = attempt
                _save_adapter(model, tokenizer, best_dir)
            model.train()

    model.eval()
    if numerical_failure:
        _save_adapter(model, tokenizer, last_finite_dir)
        terminal_val = None
        terminal_test = None
        status = "numerical_failure"
    else:
        terminal_val = _evaluate_model(model, tokenizer, val_rows, config, int(eval_cfg["seed"]) + seed)
        terminal_test = _evaluate_model(
            model, tokenizer, test_rows, config, int(eval_cfg["test_seed"]) + seed
        )
        _save_adapter(model, tokenizer, terminal_dir)
        status = "finite_budget_complete"
    selected_best = _evaluate_saved_adapter(
        model_path,
        best_dir,
        data_paths,
        config,
        int(eval_cfg["seed"]) + seed,
        int(eval_cfg["test_seed"]) + seed,
    )
    _atomic_json(method_dir / "selected_best_evaluation.json", selected_best)
    _csv_write(method_dir / "training_log.csv", training_rows)
    _csv_write(method_dir / "metrics.csv", metrics_rows)
    summary = {
        "method": method,
        "seed": seed,
        "sampling_attempts": int(train_cfg["sampling_attempts"]),
        "optimizer_steps": optimizer_steps,
        "skipped_attempts": skipped_attempts,
        "sampled_completions": total_sampled,
        "correct_selected": total_correct,
        "sample_correct_selection_rate": total_correct / max(total_sampled, 1),
        "best_attempt": best_attempt,
        "best_validation_value": best_value,
        "selected_best": selected_best,
        "terminal_validation": terminal_val,
        "terminal_test": terminal_test,
        "numerical_failure": numerical_failure,
        "failure_attempt": failure_attempt,
        "status": status,
    }
    _atomic_json(method_dir / "summary.json", summary)
    return summary


def terminal_audit(summary_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    onpolicy_rows = [row for row in summary_rows if row.get("method") == "onpolicy_rft_positive_only"]
    skipped = [float(row.get("skipped_attempts", 0)) for row in onpolicy_rows]
    attempts = [float(row.get("sampling_attempts", 0)) for row in onpolicy_rows]
    skip_fraction = [s / max(a, 1.0) for s, a in zip(skipped, attempts)]
    failures = [row for row in onpolicy_rows if row.get("numerical_failure")]
    return {
        "experiment_id": EXPERIMENT_ID,
        "status": "pilot_raw_complete" if not failures else "pilot_incomplete_numerical_failure",
        "result_interpretation_limit": "finite-budget unpolished diagnostic; not formal method ranking",
        "task_performance_collapse_checked_separately": True,
        "support_or_structure_boundary_checked_separately": True,
        "nan_inf_numerical_failure_checked_separately": True,
        "onpolicy_seed_count": len(onpolicy_rows),
        "max_skip_fraction": max(skip_fraction) if skip_fraction else None,
        "numerical_failure_count": len(failures),
        "numerical_failures": [
            {"seed": row.get("seed"), "failure": row.get("numerical_failure")}
            for row in failures
        ],
        "notes": [
            "sft_only is a no-continuation reference",
            "onpolicy_rft_positive_only uses only verifier-correct sampled completions",
            "skipped attempts indicate insufficient on-policy positives, not numerical failure",
            "nonfinite loss, gradient norm, and LoRA parameters are checked separately",
        ],
    }


def cmd_run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    model_path = Path(args.model_path).resolve()
    work_dir = Path(args.work_dir).resolve()
    external_adapter_path = Path(args.sft_adapter_path).resolve() if args.sft_adapter_path else None
    if not model_path.is_dir():
        raise SystemExit(f"Model directory does not exist: {model_path}")
    if (work_dir / "RUN_COMPLETE.json").exists():
        raise RuntimeError("work_dir already contains a completed E8 on-policy run")
    work_dir.mkdir(parents=True, exist_ok=True)
    if args.gpu != "auto":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    _atomic_json(
        work_dir / "run_config.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "implementation_version": VERSION,
            "model_path": str(model_path),
            "work_dir": str(work_dir),
            "config": config,
            "sft_adapter_path": str(external_adapter_path) if external_adapter_path else None,
            "source_provenance": source_provenance(),
        },
    )
    data_paths = generate_or_load_data(work_dir, config)
    adapter_path = train_sft_reference(model_path, work_dir, data_paths, config, external_adapter_path)
    summary_rows: list[dict[str, Any]] = []
    sft_summary = evaluate_sft_only(model_path, adapter_path, work_dir, data_paths, config)
    summary_rows.append({"method": "sft_only", **sft_summary})
    for seed in config["confirmation"]["paired_training_seeds"]:
        summary = train_onpolicy_positive_only(
            model_path, adapter_path, work_dir, data_paths, config, int(seed)
        )
        summary_rows.append(summary)
    _csv_write(work_dir / "onpolicy_summary.csv", summary_rows)
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
        "sft_adapter_path": str(adapter_path),
        "summary_rows": summary_rows,
        "source_provenance": source_provenance(),
    }
    _atomic_json(work_dir / "scientific_run_manifest.json", manifest)
    _atomic_json(work_dir / "RUN_COMPLETE.json", {"status": "complete", **audit})
    return 0


def cmd_selftest(_: argparse.Namespace) -> None:
    config = load_config(DEFAULT_CONFIG)
    assert tuple(config["methods"]) == METHODS
    row = {"id": "p0", "prompt": "Numbers: 1, 2, 3, 4\nTarget: 10", "numbers": [1, 2, 3, 4], "target": 10}
    selected = select_correct_completions(row, ["1+2+3+4", "1+2+3+4", "1+2+3-4"], max_per_prompt=2)
    assert len(selected) == 1
    plan = prompt_attempt_plan(5, seed=1, attempts=7, prompts_per_attempt=2)
    assert len(plan) == 7
    assert all(len(chunk) == 2 for chunk in plan)
    print("ONPOLICY_SELFTEST_OK")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Countdown E8 on-policy unpolished runner")
    parser.add_argument("--version", action="version", version=VERSION)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--model_path", required=True)
    run.add_argument("--work_dir", required=True)
    run.add_argument("--config", default=str(DEFAULT_CONFIG))
    run.add_argument("--gpu", default="0")
    run.add_argument(
        "--sft_adapter_path",
        default=None,
        help="Optional existing LoRA SFT adapter directory or parent containing best_adapter.",
    )
    run.set_defaults(func=cmd_run)
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
