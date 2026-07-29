"""Nine-task exponential taper tuning on frozen P0 banks and reference adapters.

This module intentionally starts from the P0 task-specific positive warm starts and
keeps P0's occurrence diagnostic separate from task-performance tuning.  The pilot
builds one deterministic 72-cell plan: seven exponential retentions and one
Positive-only baseline for each of nine tasks.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import shutil
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except ImportError:  # Plan and split validation do not require Torch.
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    Dataset = object  # type: ignore[assignment,misc]

from drpo.e8_multitask_p0 import (
    EncodedCompletion,
    atomic_json,
    atomic_jsonl,
    bank_path,
    collate_encoded_completions,
    completion_stats,
    encode_prompt_completion,
    format_chat_prompt,
    load_positive_warmstart_model,
    model_identity,
    read_jsonl,
    resolve_torch_dtype,
    sha256_file,
    stable_config_hash,
    validate_work_dir,
)
from drpo.e8_multitask_tasks import TASK_NAMES, build_adapters, stable_hash

EXPERIMENT_ID = "EXT-C-E8-MULTITASK-EXP-TUNING-01"
PARENT_EXPERIMENT_ID = "EXT-C-E8-MULTITASK-P0-01"
DEFAULT_CONFIG = Path("configs/e8_multitask_exp_tuning.yaml")
METHOD_POSITIVE_ONLY = "positive_only"
METHOD_EXPONENTIAL = "exponential"


@dataclass(frozen=True)
class Cell:
    task: str
    method: str
    rho: float | None
    seed: int
    stage: str

    @property
    def key(self) -> str:
        if self.method == METHOD_POSITIVE_ONLY:
            return f"{self.task}__positive_only__seed{self.seed}"
        assert self.rho is not None
        tag = f"{self.rho:.6f}".rstrip("0").rstrip(".").replace(".", "p")
        return f"{self.task}__exp_rho{tag}__seed{self.seed}"


@dataclass(frozen=True)
class TaskInputs:
    task: str
    bank: Path
    adapter: Path
    sources_root: Path


class TrainingDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[Mapping[str, Any]],
        tokenizer: Any,
        max_length: int,
    ) -> None:
        self.rows = list(rows)
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        return self.rows[index]


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    temporary.replace(path)


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    path = Path(path)
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("Configuration root must be a mapping")
    validate_config(value)
    return value


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("Expected schema_version: 1")
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError(f"Expected experiment_id: {EXPERIMENT_ID}")
    parent = config.get("parent", {})
    if parent.get("experiment_id") != PARENT_EXPERIMENT_ID:
        raise ValueError("Unexpected parent experiment")
    tasks = tuple(config.get("suite", {}).get("tasks", ()))
    if len(tasks) != 9 or len(set(tasks)) != 9:
        raise ValueError("The tuning suite must contain exactly nine unique tasks")
    if set(tasks) != set(TASK_NAMES):
        raise ValueError("The tuning suite must be Countdown plus the eight P0 tasks")
    p0_tasks = tuple(config["suite"].get("p0_tasks", ()))
    if set(p0_tasks) != set(TASK_NAMES) - {"countdown"}:
        raise ValueError("suite.p0_tasks must be the exact eight P0 tasks")
    if tuple(config["suite"].get("external_tasks", ())) != ("countdown",):
        raise ValueError("Countdown must be the only external task")

    split = config["split"]
    sizes = [int(split[key]) for key in ("train_rows", "validation_rows", "test_rows")]
    if sizes != [5000, 500, 500]:
        raise ValueError("The frozen split must be 5000/500/500")
    if bool(split.get("test_access_allowed", True)):
        raise ValueError("Tuning must forbid test access")

    training = config["training"]
    if int(training["optimizer_updates"]) != 1200:
        raise ValueError("The candidate tuning horizon is frozen to 1200 updates")
    if int(training["evaluation_every_updates"]) != 100:
        raise ValueError("Evaluation cadence must be 100 updates")
    if bool(training.get("early_stopping", True)):
        raise ValueError("Early stopping is forbidden")
    late = tuple(int(value) for value in training["late_window_updates"])
    if late != (800, 900, 1000, 1100, 1200):
        raise ValueError("Unexpected late-window updates")

    sweep = config["sweep"]
    coarse = tuple(float(value) for value in sweep["coarse_rho"])
    refinement = tuple(float(value) for value in sweep["refinement_rho"])
    all_rho = tuple(float(value) for value in sweep["all_rho"])
    if coarse != (0.9, 0.6, 0.35, 0.125):
        raise ValueError("Unexpected coarse rho grid")
    if refinement != (0.75, 0.5, 0.25):
        raise ValueError("Unexpected refinement rho grid")
    if all_rho != (0.9, 0.75, 0.6, 0.5, 0.35, 0.25, 0.125):
        raise ValueError("Unexpected full rho grid")
    if int(sweep["positive_only_per_task"]) != 1:
        raise ValueError("Exactly one Positive-only cell per task is required")
    if int(sweep["expected_cells"]) != 72:
        raise ValueError("The frozen tuning matrix must contain 72 cells")
    execution = config["execution"]
    if int(execution["max_concurrent_cells"]) != 16:
        raise ValueError("The scheduler must expose exactly 16 slots")
    if int(execution["expected_waves"]) != 5:
        raise ValueError("The frozen schedule must contain five waves")


def coefficient_from_rho(rho: float) -> float:
    if not math.isfinite(rho) or not 0.0 < rho < 1.0:
        raise ValueError("rho must be finite and strictly between zero and one")
    return -math.log(rho)


def taper_weight(distance: "torch.Tensor", rho: float) -> "torch.Tensor":
    return torch.exp(-coefficient_from_rho(rho) * distance)


def normalized_distance(
    sequence_log_probability: "torch.Tensor",
    *,
    tau: float,
    scale: float,
) -> "torch.Tensor":
    if not math.isfinite(tau) or tau < 0.0:
        raise ValueError("tau must be finite and non-negative")
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be finite and positive")
    excess = torch.relu(-sequence_log_probability.detach() - tau)
    return torch.sqrt(excess / scale)


def build_cells(config: Mapping[str, Any]) -> tuple[Cell, ...]:
    validate_config(config)
    seed = int(config["sweep"]["tuning_seed"])
    tasks = tuple(str(task) for task in config["suite"]["tasks"])
    coarse = tuple(float(value) for value in config["sweep"]["coarse_rho"])
    refinement = tuple(float(value) for value in config["sweep"]["refinement_rho"])
    cells: list[Cell] = []
    for task in tasks:
        cells.append(Cell(task, METHOD_POSITIVE_ONLY, None, seed, "coarse"))
        cells.extend(Cell(task, METHOD_EXPONENTIAL, rho, seed, "coarse") for rho in coarse)
    for task in tasks:
        cells.extend(
            Cell(task, METHOD_EXPONENTIAL, rho, seed, "refinement") for rho in refinement
        )
    if len(cells) != int(config["sweep"]["expected_cells"]):
        raise AssertionError("Internal cell-count mismatch")
    if len({cell.key for cell in cells}) != len(cells):
        raise AssertionError("Cell keys are not unique")
    return tuple(cells)


def build_waves(config: Mapping[str, Any]) -> tuple[tuple[Cell, ...], ...]:
    cells = build_cells(config)
    capacity = int(config["execution"]["max_concurrent_cells"])
    waves = tuple(tuple(cells[index : index + capacity]) for index in range(0, len(cells), capacity))
    if len(waves) != int(config["execution"]["expected_waves"]):
        raise AssertionError("Internal wave-count mismatch")
    if any(len(wave) > capacity for wave in waves):
        raise AssertionError("Wave exceeds configured capacity")
    return waves


def write_plan(config: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    waves = build_waves(config)
    rows: list[dict[str, Any]] = []
    for wave_index, wave in enumerate(waves, start=1):
        for slot, cell in enumerate(wave):
            rows.append(
                {
                    "wave": wave_index,
                    "slot": slot,
                    "cell_key": cell.key,
                    "task": cell.task,
                    "method": cell.method,
                    "rho": cell.rho,
                    "lambda": None if cell.rho is None else coefficient_from_rho(cell.rho),
                    "seed": cell.seed,
                    "stage": cell.stage,
                }
            )
    plan = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "config_hash": stable_config_hash(config),
        "cell_count": len(rows),
        "wave_count": len(waves),
        "max_concurrent_cells": int(config["execution"]["max_concurrent_cells"]),
        "rows": rows,
        "scientific_status": "not_run",
    }
    atomic_json(output_root / "plan.json", plan)
    atomic_jsonl(output_root / "plan.jsonl", rows)
    return plan


def split_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    task: str,
    config: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    split = config["split"]
    required = int(split["train_rows"]) + int(split["validation_rows"]) + int(
        split["test_rows"]
    )
    if len(rows) != required:
        raise RuntimeError(f"{task} bank must contain exactly {required} rows, found {len(rows)}")
    seed = int(split["hash_seed"])
    ordered = sorted(
        (dict(row) for row in rows),
        key=lambda row: stable_hash(
            {
                "task": task,
                "prompt_id": str(row["prompt_id"]),
                "seed": seed,
                "role": "tuning_split",
            }
        ),
    )
    train_end = int(split["train_rows"])
    validation_end = train_end + int(split["validation_rows"])
    partitions = {
        "train": ordered[:train_end],
        "validation": ordered[train_end:validation_end],
        "test": ordered[validation_end:],
    }
    prompt_sets = {
        name: {str(row["prompt_id"]) for row in values} for name, values in partitions.items()
    }
    if any(
        prompt_sets[left] & prompt_sets[right]
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    ):
        raise RuntimeError(f"{task} split prompt IDs overlap")
    return partitions


def write_split_manifest(
    task_inputs: Mapping[str, TaskInputs],
    config: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    tasks: dict[str, Any] = {}
    for task in config["suite"]["tasks"]:
        inputs = task_inputs[str(task)]
        rows = read_jsonl(inputs.bank)
        partitions = split_rows(rows, task=str(task), config=config)
        task_root = output_root / "splits" / str(task)
        task_root.mkdir(parents=True, exist_ok=True)
        for name, values in partitions.items():
            atomic_jsonl(task_root / f"{name}.jsonl", values)
        tasks[str(task)] = {
            "bank": str(inputs.bank.resolve()),
            "bank_sha256": sha256_file(inputs.bank),
            "reference_adapter": str(inputs.adapter.resolve()),
            "reference_adapter_identity": model_identity("unresolved_backbone", str(inputs.adapter))[
                "adapter"
            ],
            "sources_root": str(inputs.sources_root.resolve()),
            "counts": {name: len(values) for name, values in partitions.items()},
            "prompt_id_hashes": {
                name: stable_hash(sorted(str(row["prompt_id"]) for row in values))
                for name, values in partitions.items()
            },
            "paths": {name: str(task_root / f"{name}.jsonl") for name in partitions},
        }
    manifest = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "config_hash": stable_config_hash(config),
        "test_access_allowed": False,
        "tasks": tasks,
        "complete": len(tasks) == 9,
        "scientific_status": "not_run",
    }
    atomic_json(output_root / "split_manifest.json", manifest)
    return manifest


def resolve_task_inputs(
    config: Mapping[str, Any],
    *,
    p0_work_dir: Path,
    countdown_bank: Path,
    countdown_adapter: Path,
    countdown_sources_root: Path,
) -> dict[str, TaskInputs]:
    manifest_path = p0_work_dir / "warmstart" / "warmstart_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing P0 warm-start manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("experiment_id") != PARENT_EXPERIMENT_ID or not manifest.get("complete"):
        raise RuntimeError("P0 warm-start manifest identity or completeness mismatch")
    if manifest.get("checkpoint_kind") != config["parent"]["required_checkpoint_kind"]:
        raise RuntimeError("P0 checkpoint kind mismatch")
    result: dict[str, TaskInputs] = {}
    for task in config["suite"]["p0_tasks"]:
        task = str(task)
        task_manifest = manifest.get("tasks", {}).get(task)
        if not task_manifest or not task_manifest.get("complete"):
            raise RuntimeError(f"Missing complete P0 warm start for {task}")
        bank = bank_path(p0_work_dir, task)
        adapter = Path(str(task_manifest["adapter_path"]))
        sources = p0_work_dir / "sources"
        result[task] = TaskInputs(task, bank, adapter, sources)
    external = TaskInputs(
        "countdown",
        countdown_bank.resolve(),
        countdown_adapter.resolve(),
        countdown_sources_root.resolve(),
    )
    result["countdown"] = external
    for task, inputs in result.items():
        if not inputs.bank.is_file():
            raise FileNotFoundError(f"Missing bank for {task}: {inputs.bank}")
        if not (inputs.adapter / "adapter_config.json").is_file():
            raise FileNotFoundError(f"Missing adapter for {task}: {inputs.adapter}")
        if not inputs.sources_root.is_dir():
            raise FileNotFoundError(f"Missing sources root for {task}: {inputs.sources_root}")
    if set(result) != set(config["suite"]["tasks"]):
        raise AssertionError("Resolved task inputs do not match suite")
    return result


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def _move_batch(batch: Mapping[str, Any], device: "torch.device") -> dict[str, Any]:
    return {
        key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()
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
    return collate_encoded_completions(encoded, pad_token_id=int(tokenizer.pad_token_id))


def _select_current_extremes(
    model: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    max_length: int,
) -> tuple[dict[str, Any], dict[str, Any], list[int], list[int]]:
    device = next(model.parameters()).device
    near_indices: list[int] = []
    far_indices: list[int] = []
    near_completions: list[str] = []
    far_completions: list[str] = []
    prompts: list[str] = []
    with torch.no_grad():
        for row in rows:
            prompt = str(row["prompt"])
            negatives = list(row["negatives"])
            if not negatives:
                raise RuntimeError("Training row has no negatives")
            encoded = [
                encode_prompt_completion(
                    tokenizer,
                    prompt,
                    str(item["completion"]),
                    max_length,
                )
                for item in negatives
            ]
            batch = collate_encoded_completions(
                encoded,
                pad_token_id=int(tokenizer.pad_token_id),
            )
            batch = _move_batch(batch, device)
            stats = completion_stats_batch(model, batch)
            surprisals = -stats["seq_lp"]
            near_index = int(torch.argmin(surprisals).item())
            far_index = int(torch.argmax(surprisals).item())
            near_indices.append(near_index)
            far_indices.append(far_index)
            near_completions.append(str(negatives[near_index]["completion"]))
            far_completions.append(str(negatives[far_index]["completion"]))
            prompts.append(prompt)
    near = _stack_encoded(tokenizer, prompts, near_completions, max_length)
    far = _stack_encoded(tokenizer, prompts, far_completions, max_length)
    return near, far, near_indices, far_indices


def completion_stats_batch(model: Any, batch: Mapping[str, Any]) -> dict[str, Any]:
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    attention_mask = batch["attention_mask"]
    output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = output.logits[:, :-1, :].float()
    shifted_labels = labels[:, 1:]
    mask = shifted_labels.ne(-100)
    safe_labels = shifted_labels.masked_fill(~mask, 0)
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    lengths = mask.sum(-1).clamp_min(1)
    sequence_log_prob = (token_log_probs * mask).sum(-1) / lengths
    return {"seq_lp": sequence_log_prob, "lengths": lengths}


def calibrate_task(
    model: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    calibration = config["remoteness_calibration"]
    count = int(calibration["prompt_rows"])
    selected = list(rows[:count])
    if len(selected) != count:
        raise RuntimeError("Insufficient calibration prompts")
    near_values: list[float] = []
    far_values: list[float] = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for row in selected:
            negatives = list(row["negatives"])
            batch = _stack_encoded(
                tokenizer,
                [str(row["prompt"])] * len(negatives),
                [str(item["completion"]) for item in negatives],
                int(config["model"]["max_length"]),
            )
            batch = _move_batch(batch, device)
            surprisal = -completion_stats_batch(model, batch)["seq_lp"]
            near_values.append(float(torch.min(surprisal).cpu()))
            far_values.append(float(torch.max(surprisal).cpu()))
    tau = float(np.median(np.asarray(near_values, dtype=float)))
    scale = float(np.median(np.asarray(far_values, dtype=float)) - tau)
    minimum = float(calibration["minimum_surprisal_scale"])
    if not math.isfinite(tau) or not math.isfinite(scale) or scale < minimum:
        raise RuntimeError(f"Degenerate remoteness calibration: tau={tau}, scale={scale}")
    return {
        "tau": tau,
        "scale": scale,
        "near_median": tau,
        "far_median": tau + scale,
        "prompt_rows": count,
    }


def _load_child_model(
    base_model_path: str,
    reference_adapter: Path,
    config: Mapping[str, Any],
) -> tuple[Any, Any, Any]:
    if torch is None:
        raise RuntimeError("Training requires Torch")
    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
    except ImportError as exc:
        raise RuntimeError("Training requires transformers and peft") from exc
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
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
    model = PeftModel.from_pretrained(base, str(reference_adapter), is_trainable=True)
    if bool(config["model"].get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    model.config.use_cache = False
    return model, tokenizer, get_cosine_schedule_with_warmup


def _raw_gradient_norm(grads: Sequence["torch.Tensor | None"]) -> float:
    total = torch.zeros((), dtype=torch.float64)
    for grad in grads:
        if grad is not None:
            total += grad.detach().double().cpu().square().sum()
    return float(torch.sqrt(total))


def calibrate_negative_scales(
    model: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    remoteness: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, float]:
    count = int(config["remoteness_calibration"]["gradient_prompt_rows"])
    selected = list(rows[:count])
    if len(selected) != count:
        raise RuntimeError("Insufficient gradient calibration prompts")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    device = next(model.parameters()).device
    max_length = int(config["model"]["max_length"])
    near, far, _, _ = _select_current_extremes(
        model,
        tokenizer,
        selected,
        max_length=max_length,
    )
    near = _move_batch(near, device)
    far = _move_batch(far, device)
    near_lp = completion_stats_batch(model, near)["seq_lp"]
    far_lp = completion_stats_batch(model, far)["seq_lp"]
    raw_norms: dict[str, float] = {}
    for rho in config["sweep"]["all_rho"]:
        rho = float(rho)
        near_weight = taper_weight(
            normalized_distance(near_lp, tau=float(remoteness["tau"]), scale=float(remoteness["scale"])),
            rho,
        )
        far_weight = taper_weight(
            normalized_distance(far_lp, tau=float(remoteness["tau"]), scale=float(remoteness["scale"])),
            rho,
        )
        scalar = (near_weight * near_lp + far_weight * far_lp).mean()
        grads = torch.autograd.grad(scalar, trainable, allow_unused=True, retain_graph=True)
        raw_norms[f"{rho:.12g}"] = _raw_gradient_norm(grads)
    anchor_key = f"{float(config['sweep']['all_rho'][0]):.12g}"
    target = raw_norms[anchor_key]
    if not math.isfinite(target) or target <= 0.0:
        raise RuntimeError("Reference negative gradient RMS is not finite and positive")
    scales: dict[str, float] = {}
    for key, value in raw_norms.items():
        if not math.isfinite(value) or value <= 0.0:
            raise RuntimeError(f"Invalid raw negative gradient norm for rho={key}: {value}")
        scales[key] = target / value
    return scales


def _collate_rows(items: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return list(items)


def train_cell(
    cell: Cell,
    *,
    inputs: TaskInputs,
    split_manifest: Mapping[str, Any],
    base_model_path: str,
    config: Mapping[str, Any],
    output_root: Path,
    force: bool,
) -> dict[str, Any]:
    if torch is None or DataLoader is None:
        raise RuntimeError("Training requires Torch")
    cell_root = output_root / "cells" / cell.key
    manifest_path = cell_root / "cell_manifest.json"
    identity = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "config_hash": stable_config_hash(config),
        "cell": {
            "task": cell.task,
            "method": cell.method,
            "rho": cell.rho,
            "seed": cell.seed,
            "stage": cell.stage,
        },
        "bank_sha256": split_manifest["tasks"][cell.task]["bank_sha256"],
        "split_prompt_hashes": split_manifest["tasks"][cell.task]["prompt_id_hashes"],
        "base_model_identity": model_identity(base_model_path, None)["model"],
        "reference_adapter_identity": model_identity(base_model_path, str(inputs.adapter))["adapter"],
    }
    identity["identity_hash"] = stable_hash(identity)
    if manifest_path.is_file() and not force:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("identity_hash") == identity["identity_hash"] and existing.get("complete"):
            return existing
        raise RuntimeError(f"Existing cell identity mismatch: {cell.key}")
    if cell_root.exists():
        if not force:
            raise RuntimeError(f"Cell output exists without reusable manifest: {cell_root}")
        if (output_root / "cells").resolve() not in cell_root.resolve().parents:
            raise RuntimeError(f"Refusing unsafe cell removal: {cell_root}")
        shutil.rmtree(cell_root)
    cell_root.mkdir(parents=True, exist_ok=False)

    _seed_everything(cell.seed)
    model, tokenizer, scheduler_factory = _load_child_model(
        base_model_path,
        inputs.adapter,
        config,
    )
    train_path = Path(split_manifest["tasks"][cell.task]["paths"]["train"])
    train_rows = read_jsonl(train_path)
    adapter = build_adapters({"tasks": {"names": [cell.task]}}, inputs.sources_root)[cell.task]
    training = config["training"]
    max_length = int(config["model"]["max_length"])
    remoteness = calibrate_task(model, tokenizer, train_rows, config=config)
    negative_scales = (
        calibrate_negative_scales(
            model,
            tokenizer,
            train_rows,
            remoteness=remoteness,
            config=config,
        )
        if cell.method == METHOD_EXPONENTIAL
        else {}
    )
    dataset = TrainingDataset(train_rows, tokenizer, max_length)
    generator = torch.Generator()
    generator.manual_seed(cell.seed)
    loader = DataLoader(
        dataset,
        batch_size=int(training["micro_batch"]),
        shuffle=True,
        generator=generator,
        num_workers=0,
        collate_fn=_collate_rows,
        drop_last=True,
    )
    iterator = iter(loader)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    updates = int(training["optimizer_updates"])
    accumulation = int(training["gradient_accumulation"])
    scheduler = scheduler_factory(
        optimizer,
        num_warmup_steps=max(1, int(updates * float(training["warmup_ratio"]))),
        num_training_steps=updates,
    )
    device = next(model.parameters()).device
    metrics: list[dict[str, Any]] = []
    optimizer.zero_grad(set_to_none=True)
    model.train()
    for update in range(1, updates + 1):
        positive_loss_sum = 0.0
        negative_loss_sum = 0.0
        near_weight_sum = 0.0
        far_weight_sum = 0.0
        for _ in range(accumulation):
            try:
                rows = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                rows = next(iterator)
            prompts = [str(row["prompt"]) for row in rows]
            positives = [str(row["oracle_completion"]) for row in rows]
            positive = _move_batch(_stack_encoded(tokenizer, prompts, positives, max_length), device)
            positive_lp = completion_stats_batch(model, positive)["seq_lp"]
            loss = -positive_lp.mean()
            positive_loss_sum += float((-positive_lp.mean()).detach().cpu())
            if cell.method == METHOD_EXPONENTIAL:
                assert cell.rho is not None
                near, far, _, _ = _select_current_extremes(
                    model,
                    tokenizer,
                    rows,
                    max_length=max_length,
                )
                near = _move_batch(near, device)
                far = _move_batch(far, device)
                near_lp = completion_stats_batch(model, near)["seq_lp"]
                far_lp = completion_stats_batch(model, far)["seq_lp"]
                near_distance = normalized_distance(
                    near_lp,
                    tau=float(remoteness["tau"]),
                    scale=float(remoteness["scale"]),
                )
                far_distance = normalized_distance(
                    far_lp,
                    tau=float(remoteness["tau"]),
                    scale=float(remoteness["scale"]),
                )
                near_weight = taper_weight(near_distance, cell.rho).detach()
                far_weight = taper_weight(far_distance, cell.rho).detach()
                scale = float(negative_scales[f"{cell.rho:.12g}"])
                negative_scalar = scale * (
                    (near_weight * near_lp).mean() + (far_weight * far_lp).mean()
                )
                loss = loss + negative_scalar
                negative_loss_sum += float(negative_scalar.detach().cpu())
                near_weight_sum += float(near_weight.mean().cpu())
                far_weight_sum += float(far_weight.mean().cpu())
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"{cell.key} non-finite loss at update {update}")
            (loss / accumulation).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable,
            float(training["max_grad_norm"]),
        )
        if not bool(torch.isfinite(grad_norm)):
            raise RuntimeError(f"{cell.key} non-finite gradient at update {update}")
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        if not all(bool(torch.isfinite(parameter).all()) for parameter in trainable):
            raise RuntimeError(f"{cell.key} non-finite trainable parameter at update {update}")
        if update % int(training["evaluation_every_updates"]) == 0 or update == updates:
            metrics.append(
                {
                    "update": update,
                    "positive_loss": positive_loss_sum / accumulation,
                    "negative_scalar": negative_loss_sum / accumulation,
                    "mean_near_weight": near_weight_sum / accumulation,
                    "mean_far_weight": far_weight_sum / accumulation,
                    "gradient_norm": float(grad_norm.detach().cpu()),
                    "learning_rate": float(scheduler.get_last_lr()[0]),
                }
            )
    terminal_adapter = cell_root / "terminal_adapter"
    model.save_pretrained(terminal_adapter, safe_serialization=True)
    tokenizer.save_pretrained(terminal_adapter)
    atomic_jsonl(cell_root / "training_metrics.jsonl", metrics)
    result = {
        **identity,
        "remoteness_calibration": remoteness,
        "negative_scales": negative_scales,
        "optimizer_updates": updates,
        "terminal_adapter": str(terminal_adapter.resolve()),
        "training_metrics": str(cell_root / "training_metrics.jsonl"),
        "nan_inf_failure": False,
        "complete": True,
        "scientific_status": "not_run",
        "evaluation_status": "pending",
    }
    atomic_json(manifest_path, result)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def cmd_prepare(
    config: Mapping[str, Any],
    output_root: Path,
    *,
    p0_work_dir: Path,
    countdown_bank: Path,
    countdown_adapter: Path,
    countdown_sources_root: Path,
) -> dict[str, Any]:
    task_inputs = resolve_task_inputs(
        config,
        p0_work_dir=p0_work_dir,
        countdown_bank=countdown_bank,
        countdown_adapter=countdown_adapter,
        countdown_sources_root=countdown_sources_root,
    )
    plan = write_plan(config, output_root)
    splits = write_split_manifest(task_inputs, config, output_root)
    inputs = {
        task: {
            "bank": str(value.bank.resolve()),
            "adapter": str(value.adapter.resolve()),
            "sources_root": str(value.sources_root.resolve()),
        }
        for task, value in task_inputs.items()
    }
    manifest = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "config_hash": stable_config_hash(config),
        "plan": str(output_root / "plan.json"),
        "split_manifest": str(output_root / "split_manifest.json"),
        "inputs": inputs,
        "complete": plan["cell_count"] == 72 and splits["complete"],
        "scientific_status": "not_run",
    }
    atomic_json(output_root / "prepare_manifest.json", manifest)
    return manifest


def _load_prepared_inputs(output_root: Path) -> tuple[dict[str, Any], dict[str, TaskInputs]]:
    prepare_path = output_root / "prepare_manifest.json"
    split_path = output_root / "split_manifest.json"
    if not prepare_path.is_file() or not split_path.is_file():
        raise RuntimeError("Run prepare before training")
    prepare = json.loads(prepare_path.read_text(encoding="utf-8"))
    splits = json.loads(split_path.read_text(encoding="utf-8"))
    if not prepare.get("complete") or not splits.get("complete"):
        raise RuntimeError("Prepared inputs are incomplete")
    inputs = {
        task: TaskInputs(
            task,
            Path(value["bank"]),
            Path(value["adapter"]),
            Path(value["sources_root"]),
        )
        for task, value in prepare["inputs"].items()
    }
    return splits, inputs


def cmd_train_cell(
    config: Mapping[str, Any],
    output_root: Path,
    *,
    cell_key: str,
    base_model_path: str,
    force: bool,
) -> dict[str, Any]:
    cells = {cell.key: cell for cell in build_cells(config)}
    if cell_key not in cells:
        raise ValueError(f"Unknown cell key: {cell_key}")
    splits, inputs = _load_prepared_inputs(output_root)
    return train_cell(
        cells[cell_key],
        inputs=inputs[cells[cell_key].task],
        split_manifest=splits,
        base_model_path=base_model_path,
        config=config,
        output_root=output_root,
        force=force,
    )


def cmd_audit(config: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    cells = build_cells(config)
    completed: dict[str, Any] = {}
    missing: list[str] = []
    failures: list[str] = []
    for cell in cells:
        path = output_root / "cells" / cell.key / "cell_manifest.json"
        if not path.is_file():
            missing.append(cell.key)
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        completed[cell.key] = value
        if not value.get("complete") or value.get("nan_inf_failure"):
            failures.append(cell.key)
    audit = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "expected_cells": len(cells),
        "completed_cells": len(completed),
        "missing_cells": missing,
        "failed_or_nonfinite_cells": failures,
        "all_training_complete": not missing and not failures,
        "evaluation_complete": all(
            value.get("evaluation_status") == "complete" for value in completed.values()
        )
        if completed
        else False,
        "test_partition_accessed": False,
        "scientific_status": "not_run",
        "claim_boundary": (
            "Development tuning only; fixed horizon is not convergence and D-U1 remains the "
            "categorical causal-identification environment."
        ),
    }
    atomic_json(output_root / "terminal_audit.json", audit)
    return audit


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-root", required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--p0-work-dir", required=True)
    prepare.add_argument("--countdown-bank", required=True)
    prepare.add_argument("--countdown-adapter", required=True)
    prepare.add_argument("--countdown-sources-root", required=True)

    train = subparsers.add_parser("train-cell")
    train.add_argument("--cell-key", required=True)
    train.add_argument("--base-model-path", required=True)
    train.add_argument("--force", action="store_true")

    subparsers.add_parser("audit")
    subparsers.add_parser("plan")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = make_parser().parse_args(argv)
    config = load_config(args.config)
    output_root = validate_work_dir(args.output_root)
    if args.command == "prepare":
        result = cmd_prepare(
            config,
            output_root,
            p0_work_dir=Path(args.p0_work_dir).resolve(),
            countdown_bank=Path(args.countdown_bank).resolve(),
            countdown_adapter=Path(args.countdown_adapter).resolve(),
            countdown_sources_root=Path(args.countdown_sources_root).resolve(),
        )
    elif args.command == "train-cell":
        result = cmd_train_cell(
            config,
            output_root,
            cell_key=args.cell_key,
            base_model_path=args.base_model_path,
            force=bool(args.force),
        )
    elif args.command == "audit":
        result = cmd_audit(config, output_root)
    elif args.command == "plan":
        result = write_plan(config, output_root)
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
