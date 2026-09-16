"""Historical warm-start/rho/dense E8 training implementation.

This module owns the native historical multitask path only. Formal cold-start
execution is rejected here and remains the responsibility of the frozen
canonical Countdown/paper implementations wired by ``e8_multitask_exp_tuning``.

The composition root supplies the few shared orchestration callbacks needed by
the native trainer explicitly through :class:`WarmstartTrainingBindings`; this
module never imports the composition root back.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from drpo import e8_experiment_config as experiment_config
from drpo import e8_multitask_inputs as e8_inputs
from drpo.e8_multitask_inputs import _ordered_by_prompt_hash
from drpo.e8_multitask_p0 import (
    append_jsonl,
    atomic_json,
    collate_encoded_completions,
    encode_prompt_completion,
    format_chat_prompt,
    model_identity,
    read_jsonl,
    resolve_torch_dtype,
    stable_config_hash,
)
from drpo.e8_multitask_tasks import TaskInstance, stable_hash

TaskInputs = e8_inputs.TaskInputs
_load_task_adapter_and_instances = e8_inputs._load_task_adapter_and_instances

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except ImportError:  # Planning/import validation does not require Torch.
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    Dataset = object  # type: ignore[assignment,misc]


@dataclass(frozen=True)
class WarmstartTrainingBindings:
    """Shared composition callbacks required by the historical native trainer."""

    cell_identity: Callable[..., dict[str, Any]]
    prepare_cell_output: Callable[..., tuple[Path, Path, dict[str, Any] | None]]
    summarize_evaluations: Callable[
        [Sequence[Mapping[str, Any]], Mapping[str, Any]], dict[str, Any]
    ]
    method_exponential: str


class RowDataset(Dataset):
    def __init__(self, rows: Sequence[Mapping[str, Any]]) -> None:
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        return self.rows[index]


def _is_coldstart(config: Mapping[str, Any]) -> bool:
    return (
        experiment_config.sweep_profile(config)
        == experiment_config.SWEEP_PROFILE_COLDSTART
    )


def _seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed % (2**32))
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def _task_rhos(config: Mapping[str, Any], task: str) -> tuple[float, ...]:
    profile = experiment_config.sweep_profile(config)
    if profile in {
        experiment_config.SWEEP_PROFILE_DENSE,
        experiment_config.SWEEP_PROFILE_COLDSTART,
    }:
        return tuple(
            math.exp(-float(value))
            for value in experiment_config.task_lambdas(config, task)
        )
    return tuple(float(value) for value in config["sweep"]["all_rho"])


def _coefficient_from_rho(rho: float) -> float:
    if not math.isfinite(rho) or not 0.0 < rho < 1.0:
        raise ValueError("rho must be finite and strictly between zero and one")
    return -math.log(rho)


def normalized_distance(
    sequence_log_probability: Any,
    *,
    tau: float,
    scale: float,
) -> Any:
    if torch is None:
        raise RuntimeError("Torch is required")
    if not math.isfinite(tau) or tau < 0.0:
        raise ValueError("tau must be finite and non-negative")
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be finite and positive")
    return torch.sqrt(torch.relu(-sequence_log_probability.detach() - tau) / scale)


def taper_weight(distance: Any, rho: float) -> Any:
    if torch is None:
        raise RuntimeError("Torch is required")
    return torch.exp(-_coefficient_from_rho(rho) * distance)


def _move_batch(batch: Mapping[str, Any], device: Any) -> dict[str, Any]:
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


def _stack_encoded(
    tokenizer: Any,
    prompts: Sequence[str],
    completions: Sequence[str],
    max_length: int,
) -> dict[str, Any]:
    encoded = [
        encode_prompt_completion(tokenizer, prompt, completion, max_length)
        for prompt, completion in zip(prompts, completions, strict=True)
    ]
    return collate_encoded_completions(
        encoded, pad_token_id=int(tokenizer.pad_token_id)
    )


def completion_stats_batch(model: Any, batch: Mapping[str, Any]) -> dict[str, Any]:
    if torch is None or F is None:
        raise RuntimeError("Torch is required")
    output = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
    )
    logits = output.logits[:, :-1, :].float()
    shifted_labels = batch["labels"][:, 1:]
    mask = shifted_labels.ne(-100)
    safe_labels = shifted_labels.masked_fill(~mask, 0)
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    lengths = mask.sum(-1).clamp_min(1)
    sequence_log_probability = (token_log_probs * mask).sum(-1) / lengths
    return {"seq_lp": sequence_log_probability, "lengths": lengths}


def _select_current_extremes(
    model: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    max_length: int,
) -> tuple[dict[str, Any], dict[str, Any], list[float], list[float]]:
    if torch is None:
        raise RuntimeError("Torch is required")
    device = next(model.parameters()).device
    prompts: list[str] = []
    all_encoded = []
    row_sizes: list[int] = []
    for row in rows:
        prompt = str(row["prompt"])
        negatives = list(row["negatives"])
        if len(negatives) != 16:
            raise RuntimeError(
                f"{row['prompt_id']} must expose 16 negatives during dynamic selection"
            )
        prompts.append(prompt)
        row_sizes.append(len(negatives))
        all_encoded.extend(
            encode_prompt_completion(
                tokenizer,
                prompt,
                str(item["completion"]),
                max_length,
            )
            for item in negatives
        )
    packed = collate_encoded_completions(
        all_encoded,
        pad_token_id=int(tokenizer.pad_token_id),
    )
    was_training = model.training
    model.eval()
    with torch.no_grad():
        all_surprisals = -completion_stats_batch(
            model,
            _move_batch(packed, device),
        )["seq_lp"]
    if was_training:
        model.train()

    near_completions: list[str] = []
    far_completions: list[str] = []
    near_values: list[float] = []
    far_values: list[float] = []
    cursor = 0
    for row, size in zip(rows, row_sizes, strict=True):
        row_surprisal = all_surprisals[cursor : cursor + size]
        near_index = int(torch.argmin(row_surprisal).item())
        far_index = int(torch.argmax(row_surprisal).item())
        negatives = list(row["negatives"])
        near_completions.append(str(negatives[near_index]["completion"]))
        far_completions.append(str(negatives[far_index]["completion"]))
        near_values.append(float(row_surprisal[near_index].cpu()))
        far_values.append(float(row_surprisal[far_index].cpu()))
        cursor += size
    return (
        _stack_encoded(tokenizer, prompts, near_completions, max_length),
        _stack_encoded(tokenizer, prompts, far_completions, max_length),
        near_values,
        far_values,
    )


def _load_reference_model(
    base_model_path: str,
    reference_adapter: Path | None,
    config: Mapping[str, Any],
    *,
    train_mode: bool,
) -> tuple[Any, Any, Any]:
    if _is_coldstart(config):
        raise RuntimeError(
            "Cold-start may not use the multitask model loader; the old canonical "
            "arena.load_model(adapter_path=None) owns initialization"
        )
    if reference_adapter is None:
        raise RuntimeError("Warm-start profile requires a reference adapter")
    if torch is None:
        raise RuntimeError("Training requires Torch")
    try:
        from peft import PeftModel
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            get_cosine_schedule_with_warmup,
        )
    except ImportError as exc:
        raise RuntimeError("Training requires transformers and peft") from exc
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": resolve_torch_dtype(str(config["model"]["dtype"])),
    }
    if torch.cuda.is_available():
        kwargs["device_map"] = {"": int(os.environ.get("LOCAL_RANK", "0"))}
    base = AutoModelForCausalLM.from_pretrained(base_model_path, **kwargs)
    if not torch.cuda.is_available():
        base.to(torch.device("cpu"))
    model = PeftModel.from_pretrained(
        base,
        str(reference_adapter),
        is_trainable=True,
    )
    if train_mode and bool(config["model"].get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    model.config.use_cache = False
    return model, tokenizer, get_cosine_schedule_with_warmup


def _trainable_state_sha256(model: Any) -> str:
    """Hash fresh/trainable state without serializing an adapter to disk."""

    if torch is None:
        raise RuntimeError("Torch is required")
    digest = hashlib.sha256()
    found = False
    for name, parameter in sorted(
        model.named_parameters(), key=lambda item: item[0]
    ):
        if not parameter.requires_grad:
            continue
        found = True
        value = parameter.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    if not found:
        raise RuntimeError("No trainable parameters to hash")
    return digest.hexdigest()


def _raw_gradient_norm(grads: Sequence[Any | None]) -> float:
    if torch is None:
        raise RuntimeError("Torch is required")
    total = torch.zeros((), dtype=torch.float64)
    for gradient in grads:
        if gradient is not None:
            total += gradient.detach().double().cpu().square().sum()
    return float(torch.sqrt(total))


def _calibration_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    task: str,
    count: int,
    seed: int,
    role: str,
) -> list[dict[str, Any]]:
    selected = _ordered_by_prompt_hash(
        rows, task=task, seed=seed, role=role
    )[:count]
    if len(selected) != count:
        raise RuntimeError(f"{task} calibration requires {count} prompts")
    return selected


def _calibration_identity(
    task: str,
    *,
    inputs: TaskInputs,
    split_manifest: Mapping[str, Any],
    base_model_path: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    identity = {
        "schema_version": 1,
        "experiment_id": experiment_config.experiment_id(config),
        "config_hash": stable_config_hash(config),
        "task": task,
        "bank_sha256": split_manifest["tasks"][task]["bank_sha256"],
        "train_prompt_hash": split_manifest["tasks"][task]["prompt_id_hashes"][
            "train"
        ],
        "base_model_identity": model_identity(base_model_path, None)["model"],
        "initialization": (
            dict(config["initialization"])
            if _is_coldstart(config)
            else {
                "source": "reference_adapter",
                "reference_adapter_identity": model_identity(
                    base_model_path,
                    str(inputs.reference_adapter),
                )["adapter"],
            }
        ),
    }
    identity["identity_hash"] = stable_hash(identity)
    return identity


def calibrate_task(
    task: str,
    *,
    inputs: TaskInputs,
    split_manifest: Mapping[str, Any],
    base_model_path: str,
    config: Mapping[str, Any],
    output_root: Path,
    force: bool,
) -> dict[str, Any]:
    if _is_coldstart(config):
        raise RuntimeError(
            "Cold-start calibration must call the canonical base/taper modules"
        )
    path = output_root / "calibration" / f"{task}.json"
    identity = _calibration_identity(
        task,
        inputs=inputs,
        split_manifest=split_manifest,
        base_model_path=base_model_path,
        config=config,
    )
    if path.is_file() and not force:
        existing = json.loads(path.read_text(encoding="utf-8"))
        if (
            existing.get("identity_hash") == identity["identity_hash"]
            and existing.get("complete")
        ):
            return existing
        raise RuntimeError(f"Existing calibration identity mismatch for {task}")

    initialization_seed = (
        int(config["initialization"]["seed"])
        if _is_coldstart(config)
        else int(config["remoteness_calibration"]["seed"])
    )
    _seed_everything(initialization_seed)
    model, tokenizer, _ = _load_reference_model(
        base_model_path,
        inputs.reference_adapter,
        config,
        train_mode=False,
    )
    initialization_state_sha256 = _trainable_state_sha256(model)
    model.eval()
    train_rows = read_jsonl(
        Path(split_manifest["tasks"][task]["paths"]["train"])
    )
    calibration = config["remoteness_calibration"]
    seed = int(calibration["seed"])
    remoteness_rows = _calibration_rows(
        train_rows,
        task=task,
        count=int(calibration["prompt_rows"]),
        seed=seed,
        role="remoteness",
    )
    device = next(model.parameters()).device
    max_length = int(config["model"]["max_length"])
    near_values: list[float] = []
    far_values: list[float] = []
    for start_index in range(0, len(remoteness_rows), 8):
        _, _, current_near, current_far = _select_current_extremes(
            model,
            tokenizer,
            remoteness_rows[start_index : start_index + 8],
            max_length=max_length,
        )
        near_values.extend(current_near)
        far_values.extend(current_far)
    tau = float(np.median(np.asarray(near_values, dtype=float)))
    scale = float(np.median(np.asarray(far_values, dtype=float)) - tau)
    minimum_scale = float(calibration["minimum_surprisal_scale"])
    if not math.isfinite(tau) or not math.isfinite(scale) or scale < minimum_scale:
        raise RuntimeError(
            f"{task} degenerate remoteness calibration: tau={tau}, scale={scale}"
        )

    gradient_rows = _calibration_rows(
        train_rows,
        task=task,
        count=int(calibration["gradient_prompt_rows"]),
        seed=seed,
        role="gradient_budget",
    )
    prompts = [str(row["prompt"]) for row in gradient_rows]
    positives = [str(row["oracle_completion"]) for row in gradient_rows]
    positive_batch = _move_batch(
        _stack_encoded(tokenizer, prompts, positives, max_length),
        device,
    )
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    positive_lp = completion_stats_batch(model, positive_batch)["seq_lp"]
    positive_grads = torch.autograd.grad(
        -positive_lp.mean(), trainable, allow_unused=True
    )
    positive_norm = _raw_gradient_norm(positive_grads)
    if not math.isfinite(positive_norm) or positive_norm <= 0.0:
        raise RuntimeError(
            f"{task} positive calibration gradient is not finite and positive"
        )

    near_batch, far_batch, _, _ = _select_current_extremes(
        model,
        tokenizer,
        gradient_rows,
        max_length=max_length,
    )
    near_lp = completion_stats_batch(
        model, _move_batch(near_batch, device)
    )["seq_lp"]
    far_lp = completion_stats_batch(
        model, _move_batch(far_batch, device)
    )["seq_lp"]
    near_distance = normalized_distance(near_lp, tau=tau, scale=scale)
    far_distance = normalized_distance(far_lp, tau=tau, scale=scale)
    rhos = _task_rhos(config, task)
    raw_negative_norms: dict[str, float] = {}
    for index, rho in enumerate(rhos):
        scalar = 0.5 * (
            taper_weight(near_distance, rho).detach() * near_lp
        ).mean()
        scalar = scalar + 0.5 * (
            taper_weight(far_distance, rho).detach() * far_lp
        ).mean()
        grads = torch.autograd.grad(
            scalar,
            trainable,
            allow_unused=True,
            retain_graph=index < len(rhos) - 1,
        )
        raw_negative_norms[f"{rho:.12g}"] = _raw_gradient_norm(grads)
    target_ratio = float(
        calibration["target_negative_to_positive_gradient_ratio"]
    )
    target_negative_norm = positive_norm * target_ratio
    negative_scales: dict[str, float] = {}
    for key, raw_norm in raw_negative_norms.items():
        if not math.isfinite(raw_norm) or raw_norm <= 0.0:
            raise RuntimeError(
                f"{task} invalid raw negative gradient for rho={key}: {raw_norm}"
            )
        negative_scales[key] = target_negative_norm / raw_norm

    result = {
        **identity,
        "tau": tau,
        "scale": scale,
        "near_median": tau,
        "far_median": tau + scale,
        "positive_gradient_norm": positive_norm,
        "target_negative_to_positive_gradient_ratio": target_ratio,
        "target_negative_gradient_norm": target_negative_norm,
        "raw_negative_gradient_norms": raw_negative_norms,
        "negative_scales": negative_scales,
        "prompt_rows": int(calibration["prompt_rows"]),
        "gradient_prompt_rows": int(calibration["gradient_prompt_rows"]),
        "initialization_state_sha256": initialization_state_sha256,
        "complete": True,
        "scientific_status": "not_run",
        "note": (
            "Training-only initialization calibration; not scientific outcome evidence."
        ),
    }
    atomic_json(path, result)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _generate_completions(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    num_return_sequences: int,
    do_sample: bool,
    seed: int,
    config: Mapping[str, Any],
) -> list[list[str]]:
    evaluation = config["evaluation"]
    device = next(model.parameters()).device
    batch_size = int(evaluation["batch_size"])
    previous_padding = tokenizer.padding_side
    tokenizer.padding_side = "left"
    outputs: list[list[str]] = []
    _seed_everything(seed)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            current = list(prompts[start : start + batch_size])
            formatted = [
                format_chat_prompt(tokenizer, prompt) for prompt in current
            ]
            tokenized = tokenizer(
                formatted,
                add_special_tokens=False,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=int(config["model"]["max_length"]),
            )
            tokenized = _move_batch(tokenized, device)
            input_width = int(tokenized["input_ids"].shape[1])
            generation_kwargs: dict[str, Any] = {
                "max_new_tokens": int(evaluation["max_new_tokens"]),
                "do_sample": do_sample,
                "num_return_sequences": num_return_sequences,
                "pad_token_id": int(tokenizer.pad_token_id),
                "eos_token_id": int(tokenizer.eos_token_id),
                "use_cache": True,
            }
            if do_sample:
                generation_kwargs.update(
                    {
                        "temperature": float(
                            evaluation["sampling_temperature"]
                        ),
                        "top_p": float(evaluation["top_p"]),
                    }
                )
            generated = model.generate(**tokenized, **generation_kwargs)
            suffixes = generated[:, input_width:]
            decoded = tokenizer.batch_decode(
                suffixes, skip_special_tokens=True
            )
            for row_index in range(len(current)):
                left = row_index * num_return_sequences
                outputs.append(
                    decoded[left : left + num_return_sequences]
                )
    tokenizer.padding_side = previous_padding
    return outputs


def evaluate_model(
    model: Any,
    tokenizer: Any,
    *,
    task: str,
    validation_rows: Sequence[Mapping[str, Any]],
    adapter: Any,
    instances: Mapping[str, TaskInstance],
    update: int,
    cell_seed: int,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    evaluation = config["evaluation"]
    seed = int(evaluation["generation_seed"]) + update * 1009 + cell_seed
    ordered = _ordered_by_prompt_hash(
        validation_rows,
        task=task,
        seed=int(evaluation["generation_seed"]),
        role="validation_evaluation",
    )
    greedy_rows = ordered[: int(evaluation["greedy_prompt_rows"])]
    passk_rows = ordered[: int(evaluation["passk_prompt_rows"])]
    greedy_outputs = _generate_completions(
        model,
        tokenizer,
        [str(row["prompt"]) for row in greedy_rows],
        num_return_sequences=1,
        do_sample=False,
        seed=seed,
        config=config,
    )
    greedy_results = [
        adapter.verify(instances[str(row["prompt_id"])], completions[0])
        for row, completions in zip(
            greedy_rows, greedy_outputs, strict=True
        )
    ]
    k = int(evaluation["pass_k"])
    sampled_outputs = _generate_completions(
        model,
        tokenizer,
        [str(row["prompt"]) for row in passk_rows],
        num_return_sequences=k,
        do_sample=True,
        seed=seed + 1,
        config=config,
    )
    pass_successes: list[bool] = []
    sampled_valid: list[bool] = []
    sampled_lengths: list[int] = []
    for row, completions in zip(passk_rows, sampled_outputs, strict=True):
        results = [
            adapter.verify(instances[str(row["prompt_id"])], completion)
            for completion in completions
        ]
        pass_successes.append(any(result.correct for result in results))
        sampled_valid.extend(result.format_valid for result in results)
        sampled_lengths.extend(len(completion) for completion in completions)
    greedy_lengths = [
        len(completions[0]) for completions in greedy_outputs
    ]
    metrics = {
        "update": update,
        "greedy_prompt_rows": len(greedy_rows),
        "passk_prompt_rows": len(passk_rows),
        "pass_k": k,
        "greedy_success": float(
            np.mean([result.correct for result in greedy_results])
        ),
        "greedy_valid_rate": float(
            np.mean([result.format_valid for result in greedy_results])
        ),
        "greedy_mean_response_characters": float(np.mean(greedy_lengths)),
        "pass8": float(np.mean(pass_successes)),
        "sampled_valid_rate": float(np.mean(sampled_valid)),
        "sampled_mean_response_characters": float(np.mean(sampled_lengths)),
        "generation_seed": seed,
    }
    integer_fields = {
        "update",
        "greedy_prompt_rows",
        "passk_prompt_rows",
        "pass_k",
        "generation_seed",
    }
    if not all(
        math.isfinite(float(value))
        for key, value in metrics.items()
        if key not in integer_fields
    ):
        raise RuntimeError(
            f"{task} produced non-finite validation metrics at update {update}"
        )
    return metrics


def _load_cell_splits(
    split_manifest: Mapping[str, Any],
    task: str,
    *,
    engineering_liveness: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    paths = split_manifest["tasks"][task]["paths"]
    train_rows = read_jsonl(Path(paths["train"]))
    validation_rows = (
        []
        if engineering_liveness
        else read_jsonl(Path(paths["validation"]))
    )
    return train_rows, validation_rows


def train_cell_impl(
    cell: Any,
    *,
    inputs: TaskInputs,
    split_manifest: Mapping[str, Any],
    base_model_path: str,
    config: Mapping[str, Any],
    output_root: Path,
    force: bool,
    bindings: WarmstartTrainingBindings,
    updates_override: int | None = None,
    engineering_liveness: bool = False,
) -> dict[str, Any]:
    if _is_coldstart(config):
        raise RuntimeError(
            "The custom multitask training implementation is forbidden for cold-start; "
            "dispatch through the canonical old code"
        )
    if torch is None or DataLoader is None:
        raise RuntimeError("Training requires Torch")
    calibration_path = output_root / "calibration" / f"{cell.task}.json"
    if not calibration_path.is_file():
        raise RuntimeError(f"Run calibration before training {cell.task}")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if not calibration.get("complete"):
        raise RuntimeError(f"Incomplete calibration for {cell.task}")

    root_name = "liveness" if engineering_liveness else "cells"
    identity = bindings.cell_identity(
        cell,
        inputs=inputs,
        split_manifest=split_manifest,
        base_model_path=base_model_path,
        config=config,
        calibration=calibration,
    )
    identity["engineering_liveness"] = engineering_liveness
    identity["updates_override"] = updates_override
    identity["identity_hash"] = stable_hash(identity)
    cell_root, manifest_path, reusable = bindings.prepare_cell_output(
        output_root,
        root_name=root_name,
        cell=cell,
        identity=identity,
        force=force,
    )
    if reusable is not None:
        return reusable

    initialization_seed = (
        int(config["initialization"]["seed"])
        if _is_coldstart(config)
        else cell.seed
    )
    _seed_everything(initialization_seed)
    model, tokenizer, scheduler_factory = _load_reference_model(
        base_model_path,
        inputs.reference_adapter,
        config,
        train_mode=True,
    )
    initialization_state_sha256 = _trainable_state_sha256(model)
    if _is_coldstart(config) and initialization_state_sha256 != calibration.get(
        "initialization_state_sha256"
    ):
        raise RuntimeError(
            f"Fresh LoRA initialization identity mismatch for {cell.task}; "
            "refusing a non-comparable cell"
        )
    _seed_everything(cell.seed)
    training = config["training"]
    train_rows, validation_rows = _load_cell_splits(
        split_manifest,
        cell.task,
        engineering_liveness=engineering_liveness,
    )
    adapter = None
    validation_instances: dict[str, TaskInstance] = {}
    if not engineering_liveness:
        adapter, validation_instances = _load_task_adapter_and_instances(
            cell.task,
            inputs=inputs,
            validation_rows=validation_rows,
        )
    dataset = RowDataset(train_rows)
    generator = torch.Generator()
    generator.manual_seed(cell.seed)
    loader = DataLoader(
        dataset,
        batch_size=int(training["micro_batch"]),
        shuffle=True,
        generator=generator,
        num_workers=0,
        collate_fn=lambda items: list(items),
        drop_last=True,
    )
    iterator = iter(loader)
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable:
        raise RuntimeError("No trainable LoRA parameters")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    updates = int(updates_override or training["optimizer_updates"])
    accumulation = int(training["gradient_accumulation"])
    scheduler = scheduler_factory(
        optimizer,
        num_warmup_steps=max(
            1, int(updates * float(training["warmup_ratio"]))
        ),
        num_training_steps=updates,
    )
    device = next(model.parameters()).device
    max_length = int(config["model"]["max_length"])
    evaluation_path = cell_root / "evaluation_metrics.jsonl"
    training_path = cell_root / "training_metrics.jsonl"
    best_pass8 = -math.inf
    optimizer.zero_grad(set_to_none=True)
    model.train()
    for update in range(1, updates + 1):
        positive_loss_total = 0.0
        negative_scalar_total = 0.0
        near_weight_total = 0.0
        far_weight_total = 0.0
        for _ in range(accumulation):
            try:
                rows = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                rows = next(iterator)
            prompts = [str(row["prompt"]) for row in rows]
            positives = [str(row["oracle_completion"]) for row in rows]
            positive_batch = _move_batch(
                _stack_encoded(tokenizer, prompts, positives, max_length),
                device,
            )
            positive_lp = completion_stats_batch(model, positive_batch)["seq_lp"]
            positive_loss = -positive_lp.mean()
            loss = positive_loss
            positive_loss_total += float(positive_loss.detach().cpu())
            if cell.method == bindings.method_exponential:
                if cell.rho is None:
                    raise AssertionError("Exponential cell has no rho")
                near_batch, far_batch, _, _ = _select_current_extremes(
                    model,
                    tokenizer,
                    rows,
                    max_length=max_length,
                )
                near_lp = completion_stats_batch(
                    model, _move_batch(near_batch, device)
                )["seq_lp"]
                far_lp = completion_stats_batch(
                    model, _move_batch(far_batch, device)
                )["seq_lp"]
                near_distance = normalized_distance(
                    near_lp,
                    tau=float(calibration["tau"]),
                    scale=float(calibration["scale"]),
                )
                far_distance = normalized_distance(
                    far_lp,
                    tau=float(calibration["tau"]),
                    scale=float(calibration["scale"]),
                )
                near_weight = taper_weight(near_distance, cell.rho).detach()
                far_weight = taper_weight(far_distance, cell.rho).detach()
                negative_scale = float(
                    calibration["negative_scales"][f"{cell.rho:.12g}"]
                )
                negative_scalar = negative_scale * (
                    0.5 * (near_weight * near_lp).mean()
                    + 0.5 * (far_weight * far_lp).mean()
                )
                loss = loss + negative_scalar
                negative_scalar_total += float(negative_scalar.detach().cpu())
                near_weight_total += float(near_weight.mean().cpu())
                far_weight_total += float(far_weight.mean().cpu())
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(
                    f"{cell.key} non-finite loss at update {update}"
                )
            (loss / accumulation).backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            trainable,
            float(training["max_grad_norm"]),
        )
        if not bool(torch.isfinite(gradient_norm)):
            raise RuntimeError(
                f"{cell.key} non-finite gradient at update {update}"
            )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        if not all(
            bool(torch.isfinite(parameter).all()) for parameter in trainable
        ):
            raise RuntimeError(
                f"{cell.key} non-finite trainable parameter at update {update}"
            )
        if update % 10 == 0 or update == updates:
            append_jsonl(
                training_path,
                {
                    "update": update,
                    "positive_loss": positive_loss_total / accumulation,
                    "negative_scalar": negative_scalar_total / accumulation,
                    "mean_near_weight": near_weight_total / accumulation,
                    "mean_far_weight": far_weight_total / accumulation,
                    "raw_gradient_norm_before_clip": float(
                        gradient_norm.detach().cpu()
                    ),
                    "learning_rate": float(scheduler.get_last_lr()[0]),
                },
            )
        should_evaluate = not engineering_liveness and (
            update % int(training["evaluation_every_updates"]) == 0
            or update == updates
        )
        if should_evaluate:
            if adapter is None:
                raise AssertionError(
                    "Validation adapter is unavailable outside liveness"
                )
            metrics = evaluate_model(
                model,
                tokenizer,
                task=cell.task,
                validation_rows=validation_rows,
                adapter=adapter,
                instances=validation_instances,
                update=update,
                cell_seed=cell.seed,
                config=config,
            )
            append_jsonl(evaluation_path, metrics)
            if float(metrics["pass8"]) > best_pass8:
                best_pass8 = float(metrics["pass8"])
                best_dir = cell_root / "supplementary_best_adapter"
                if best_dir.exists():
                    shutil.rmtree(best_dir)
                model.save_pretrained(best_dir, safe_serialization=True)
                tokenizer.save_pretrained(best_dir)
            model.train()

    terminal_state_sha256 = _trainable_state_sha256(model)
    terminal_adapter = cell_root / "terminal_adapter"
    model.save_pretrained(terminal_adapter, safe_serialization=True)
    tokenizer.save_pretrained(terminal_adapter)
    if engineering_liveness:
        summary: dict[str, Any] = {
            "engineering_liveness": True,
            "optimizer_updates": updates,
            "finite_parameters": True,
            "reload_gate_pending": True,
        }
        scientific_status = "not_run"
        evaluation_status = "not_applicable"
    else:
        evaluations = read_jsonl(evaluation_path)
        summary = bindings.summarize_evaluations(evaluations, config)
        scientific_status = "pilot"
        evaluation_status = "complete"
    result = {
        **identity,
        **summary,
        "optimizer_updates": updates,
        "effective_prompt_batch": int(training["micro_batch"])
        * int(training["gradient_accumulation"]),
        "terminal_adapter": str(terminal_adapter.resolve()),
        "terminal_adapter_identity": model_identity(
            base_model_path, str(terminal_adapter)
        )["adapter"],
        "initialization_state_sha256": initialization_state_sha256,
        "terminal_trainable_state_sha256": terminal_state_sha256,
        "training_metrics": str(training_path.resolve()),
        "evaluation_metrics": (
            str(evaluation_path.resolve())
            if evaluation_path.is_file()
            else None
        ),
        "nan_inf_failure": False,
        "complete": True,
        "evaluation_status": evaluation_status,
        "scientific_status": scientific_status,
    }
    atomic_json(manifest_path, result)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result
