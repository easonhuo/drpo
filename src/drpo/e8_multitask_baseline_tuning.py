"""Eight-task fitted-reference beta-TOPR and AsymRE tuning.

This module is the single method-specific sibling of the Exp tuning executor. It
reuses the exact Exp split/reference/evaluation contract but keeps an independent
experiment identity, 80-cell plan, output tree, and terminal audit.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import math
import os
import shutil
import subprocess
import sys
import traceback
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

try:
    import torch
    from torch.utils.data import DataLoader
except ImportError:  # Planning does not require Torch.
    torch = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]

from drpo import e8_multitask_exp_tuning as shared
from drpo.e8_multitask_p0 import (
    append_jsonl,
    atomic_json,
    atomic_jsonl,
    model_identity,
    read_jsonl,
    stable_config_hash,
    validate_work_dir,
)

EXPERIMENT_ID = "EXT-C-E8-MULTITASK-TOPR-ASYMRE-TUNING-01"
PARENT_EXPERIMENT_ID = "EXT-C-E8-MULTITASK-EXP-TUNING-01"
DEFAULT_CONFIG = Path("configs/e8_multitask_topr_asymre_tuning.yaml")
DEFAULT_EXP_CONFIG = Path("configs/e8_multitask_exp_tuning.yaml")
METHOD_TOPR = "joint_fitted_reference_beta_topr"
METHOD_ASYMRE = "asymre"
POLICY_ADAPTER = "default"
REFERENCE_ADAPTER = "reference"


@dataclass(frozen=True)
class Cell:
    task: str
    method: str
    parameter: float
    seed: int

    @property
    def key(self) -> str:
        label = "beta" if self.method == METHOD_TOPR else "delta"
        value = f"{self.parameter:.6f}".rstrip("0").rstrip(".")
        value = value.replace("-", "m").replace(".", "p")
        method = "topr" if self.method == METHOD_TOPR else "asymre"
        return f"{self.task}__{method}_{label}{value}__seed{self.seed}"


def _as_floats(values: Sequence[Any]) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def load_raw_yaml(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Configuration root must be a mapping: {path}")
    return value


def load_config(
    path: str | Path = DEFAULT_CONFIG,
    exp_config_path: str | Path = DEFAULT_EXP_CONFIG,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_raw_yaml(path)
    exp_config = load_raw_yaml(exp_config_path)
    validate_config(config, exp_config)
    return config, exp_config


def validate_config(config: Mapping[str, Any], exp_config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("Expected schema_version: 1")
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError(f"Expected experiment_id: {EXPERIMENT_ID}")
    if config.get("parent", {}).get("experiment_id") != PARENT_EXPERIMENT_ID:
        raise ValueError("Unexpected parent experiment")
    if exp_config.get("experiment_id") != PARENT_EXPERIMENT_ID:
        raise ValueError("Exp configuration identity mismatch")

    tasks = tuple(str(value) for value in config["suite"]["tasks"])
    expected = tuple(str(value) for value in exp_config["suite"]["p0_tasks"])
    if tasks != expected or len(tasks) != 8 or len(set(tasks)) != 8:
        raise ValueError("Baseline suite must be the exact ordered eight P0 tasks")
    if "countdown" in tasks:
        raise ValueError("Countdown must not be rerun in this tuning experiment")

    if _as_floats(config["sweep"]["topr_beta"]) != (0.0, 0.04, 0.08, 0.25, 0.5):
        raise ValueError("Unexpected TOPR beta grid")
    if _as_floats(config["sweep"]["asymre_delta_v"]) != (-1.0, -0.9, -0.7, -0.5, 0.0):
        raise ValueError("Unexpected AsymRE delta_v grid")
    if int(config["sweep"]["expected_cells"]) != 80:
        raise ValueError("Expected exactly 80 cells")

    training = exp_config["training"]
    if int(training["optimizer_updates"]) != 1200:
        raise ValueError("Inherited horizon must remain 1200 updates")
    if int(training["evaluation_every_updates"]) != 100:
        raise ValueError("Inherited evaluation cadence must remain 100 updates")
    if bool(training.get("early_stopping", True)):
        raise ValueError("Early stopping is forbidden")
    if tuple(int(value) for value in training["late_window_updates"]) != (
        800,
        900,
        1000,
        1100,
        1200,
    ):
        raise ValueError("Unexpected inherited late-window contract")

    execution = config["execution"]
    if int(execution["max_concurrent_cells"]) != 16:
        raise ValueError("Expected 16 concurrent cells")
    if tuple(int(value) for value in execution["gpu_ids"]) != tuple(range(8)):
        raise ValueError("Default GPU pool must remain 0--7")
    if int(execution["slots_per_gpu"]) != 2:
        raise ValueError("Expected two slots per GPU")
    if int(execution["expected_waves"]) != 5:
        raise ValueError("Expected exactly five waves")

    reference = config["shared_reference"]
    if reference["checkpoint_kind"] != "train_only_task_positive_warmstart_100":
        raise ValueError("Only leakage-safe train-only references are accepted")
    if int(reference["validation_rows_seen"]) != 0 or int(reference["test_rows_seen"]) != 0:
        raise ValueError("Shared references must not see validation or test rows")


def build_cells(config: Mapping[str, Any], exp_config: Mapping[str, Any]) -> tuple[Cell, ...]:
    validate_config(config, exp_config)
    seed = int(config["sweep"]["tuning_seed"])
    cells: list[Cell] = []
    for task in config["suite"]["tasks"]:
        cells.extend(
            Cell(str(task), METHOD_TOPR, beta, seed)
            for beta in _as_floats(config["sweep"]["topr_beta"])
        )
        cells.extend(
            Cell(str(task), METHOD_ASYMRE, delta, seed)
            for delta in _as_floats(config["sweep"]["asymre_delta_v"])
        )
    if len(cells) != 80 or len({cell.key for cell in cells}) != 80:
        raise AssertionError("Internal 80-cell identity failure")
    return tuple(cells)


def build_waves(
    config: Mapping[str, Any],
    exp_config: Mapping[str, Any],
) -> tuple[tuple[Cell, ...], ...]:
    cells = build_cells(config, exp_config)
    capacity = int(config["execution"]["max_concurrent_cells"])
    waves = tuple(
        tuple(cells[index : index + capacity]) for index in range(0, len(cells), capacity)
    )
    if tuple(len(wave) for wave in waves) != (16, 16, 16, 16, 16):
        raise AssertionError("Frozen wave geometry must be 16/16/16/16/16")
    return waves


def write_plan(
    config: Mapping[str, Any],
    exp_config: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    gpu_ids = tuple(int(value) for value in config["execution"]["gpu_ids"])
    waves = build_waves(config, exp_config)
    for wave_index, wave in enumerate(waves, start=1):
        for slot, cell in enumerate(wave):
            rows.append(
                {
                    "wave": wave_index,
                    "slot": slot,
                    "gpu_id": gpu_ids[slot % len(gpu_ids)],
                    "cell_key": cell.key,
                    "task": cell.task,
                    "method": cell.method,
                    "beta": cell.parameter if cell.method == METHOD_TOPR else None,
                    "delta_v": cell.parameter if cell.method == METHOD_ASYMRE else None,
                    "seed": cell.seed,
                }
            )
    plan = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "config_hash": stable_config_hash(config),
        "inherited_exp_config_hash": stable_config_hash(exp_config),
        "cell_count": len(rows),
        "wave_count": len(waves),
        "wave_sizes": [len(wave) for wave in waves],
        "rows": rows,
        "scientific_status": "not_run",
    }
    atomic_json(output_root / "plan.json", plan)
    atomic_jsonl(output_root / "plan.jsonl", rows)
    return plan


def _load_shared_contract(
    shared_root: Path,
    config: Mapping[str, Any],
    exp_config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, shared.TaskInputs]]:
    split_path = shared_root / "split_manifest.json"
    prepare_path = shared_root / "prepare_manifest.json"
    reference_path = shared_root / "references" / "reference_manifest.json"
    for path in (split_path, prepare_path, reference_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing shared Exp artifact: {path}")
    splits = json.loads(split_path.read_text(encoding="utf-8"))
    prepare = json.loads(prepare_path.read_text(encoding="utf-8"))
    references = json.loads(reference_path.read_text(encoding="utf-8"))
    exp_hash = stable_config_hash(exp_config)
    if splits.get("experiment_id") != PARENT_EXPERIMENT_ID:
        raise RuntimeError("Shared split experiment identity mismatch")
    if prepare.get("experiment_id") != PARENT_EXPERIMENT_ID:
        raise RuntimeError("Shared prepare experiment identity mismatch")
    if splits.get("config_hash") != exp_hash or prepare.get("config_hash") != exp_hash:
        raise RuntimeError("Shared Exp configuration hash mismatch")
    if bool(splits.get("test_access_allowed", True)):
        raise RuntimeError("Shared split manifest permits test access")
    if references.get("experiment_id") != PARENT_EXPERIMENT_ID:
        raise RuntimeError("Shared reference experiment identity mismatch")
    if references.get("checkpoint_kind") != config["shared_reference"]["checkpoint_kind"]:
        raise RuntimeError("Shared reference checkpoint kind mismatch")
    if int(references.get("validation_rows_seen", -1)) != 0:
        raise RuntimeError("Shared reference manifest reports validation exposure")
    if int(references.get("test_rows_seen", -1)) != 0:
        raise RuntimeError("Shared reference manifest reports test exposure")

    tasks: dict[str, shared.TaskInputs] = {}
    expected_tasks = tuple(str(task) for task in config["suite"]["tasks"])
    for task in expected_tasks:
        split_record = splits.get("tasks", {}).get(task)
        reference_record = references.get("tasks", {}).get(task)
        prepare_record = prepare.get("inputs", {}).get(task)
        if not split_record or not reference_record or not prepare_record:
            raise RuntimeError(f"Incomplete shared contract for {task}")
        counts = split_record.get("counts", {})
        split_counts = (
            int(counts.get("train", -1)),
            int(counts.get("validation", -1)),
            int(counts.get("test", -1)),
        )
        if split_counts != (5000, 500, 500):
            raise RuntimeError(f"Unexpected shared split counts for {task}: {counts}")
        if reference_record.get("train_prompt_hash") != split_record["prompt_id_hashes"]["train"]:
            raise RuntimeError(f"Reference/train split hash mismatch for {task}")
        adapter = Path(str(reference_record["adapter_path"])).resolve()
        if not (adapter / "adapter_config.json").is_file():
            raise FileNotFoundError(f"Missing shared reference adapter for {task}: {adapter}")
        tasks[task] = shared.TaskInputs(
            task=task,
            bank=Path(str(prepare_record["bank"])).resolve(),
            reference_adapter=adapter,
            sources_root=Path(str(prepare_record["sources_root"])).resolve(),
            p0_config=Path(str(prepare_record["p0_config"])).resolve(),
            countdown_validation=None,
        )
    if set(tasks) != set(expected_tasks):
        raise AssertionError("Shared task contract mismatch")
    return splits, tasks


def cmd_prepare(
    config: Mapping[str, Any],
    exp_config: Mapping[str, Any],
    output_root: Path,
    *,
    shared_root: Path,
) -> dict[str, Any]:
    splits, task_inputs = _load_shared_contract(shared_root, config, exp_config)
    plan = write_plan(config, exp_config, output_root)
    serialized = {
        task: {
            "bank": str(value.bank),
            "reference_adapter": str(value.reference_adapter),
            "sources_root": str(value.sources_root),
            "p0_config": str(value.p0_config),
            "train_path": splits["tasks"][task]["paths"]["train"],
            "validation_path": splits["tasks"][task]["paths"]["validation"],
            "train_prompt_hash": splits["tasks"][task]["prompt_id_hashes"]["train"],
            "validation_prompt_hash": splits["tasks"][task]["prompt_id_hashes"]["validation"],
        }
        for task, value in task_inputs.items()
    }
    manifest = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "config_hash": stable_config_hash(config),
        "inherited_exp_config_hash": stable_config_hash(exp_config),
        "shared_root": str(shared_root.resolve()),
        "inputs": serialized,
        "test_partition_opened": False,
        "complete": plan["cell_count"] == 80 and len(serialized) == 8,
        "scientific_status": "not_run",
    }
    atomic_json(output_root / "prepare_manifest.json", manifest)
    return manifest


def _load_prepared(
    output_root: Path,
    config: Mapping[str, Any],
    exp_config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, shared.TaskInputs]]:
    path = output_root / "prepare_manifest.json"
    if not path.is_file():
        raise RuntimeError("Run prepare before training")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("experiment_id") != EXPERIMENT_ID or not value.get("complete"):
        raise RuntimeError("Prepared baseline identity mismatch")
    if value.get("config_hash") != stable_config_hash(config):
        raise RuntimeError("Prepared baseline configuration mismatch")
    if value.get("inherited_exp_config_hash") != stable_config_hash(exp_config):
        raise RuntimeError("Prepared inherited Exp configuration mismatch")
    if value.get("test_partition_opened"):
        raise RuntimeError("Prepared manifest reports test access")
    inputs = {
        task: shared.TaskInputs(
            task=task,
            bank=Path(record["bank"]),
            reference_adapter=Path(record["reference_adapter"]),
            sources_root=Path(record["sources_root"]),
            p0_config=Path(record["p0_config"]),
            countdown_validation=None,
        )
        for task, record in value["inputs"].items()
    }
    return value, inputs


def _adapter_parameters(model: Any, adapter_name: str) -> list[Any]:
    token = f".{adapter_name}."
    result = [parameter for name, parameter in model.named_parameters() if token in name]
    if not result:
        raise RuntimeError(f"No parameters found for adapter {adapter_name!r}")
    return result


def _copy_adapter_parameters(model: Any, source: str, destination: str) -> None:
    source_token = f".{source}."
    destination_token = f".{destination}."
    source_parameters = {
        name.replace(source_token, ".<adapter>."): parameter
        for name, parameter in model.named_parameters()
        if source_token in name
    }
    destination_parameters = {
        name.replace(destination_token, ".<adapter>."): parameter
        for name, parameter in model.named_parameters()
        if destination_token in name
    }
    if source_parameters.keys() != destination_parameters.keys():
        raise RuntimeError("Policy/reference adapter structures differ")
    with torch.no_grad():
        for key in sorted(source_parameters):
            destination_parameters[key].copy_(source_parameters[key])


def _capture_rng_state() -> tuple[Any, list[Any] | None]:
    cpu_state = torch.get_rng_state().clone()
    cuda_states = (
        [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else None
    )
    return cpu_state, cuda_states


def _restore_rng_state(state: tuple[Any, list[Any] | None]) -> None:
    cpu_state, cuda_states = state
    torch.set_rng_state(cpu_state)
    if cuda_states is not None:
        torch.cuda.set_rng_state_all(cuda_states)


def _joint_optimizer_step_with_finite_guard(
    policy_optimizer: Any,
    reference_optimizer: Any | None,
    policy_parameters: Sequence[Any],
    reference_parameters: Sequence[Any],
) -> bool:
    all_parameters = [*policy_parameters, *reference_parameters]
    snapshots = [parameter.detach().clone() for parameter in all_parameters]
    if reference_optimizer is not None:
        reference_optimizer.step()
    policy_optimizer.step()
    if all(bool(torch.isfinite(parameter).all()) for parameter in all_parameters):
        return True
    with torch.no_grad():
        for parameter, snapshot in zip(all_parameters, snapshots, strict=True):
            parameter.copy_(snapshot)
    return False


def _flatten_negative_batch(
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    max_length: int,
) -> tuple[dict[str, Any], int]:
    prompts: list[str] = []
    completions: list[str] = []
    expected_count: int | None = None
    for row in rows:
        negatives = list(row["negatives"])
        if expected_count is None:
            expected_count = len(negatives)
        if len(negatives) != expected_count or len(negatives) != 16:
            raise RuntimeError("Every training prompt must expose exactly 16 negatives")
        prompts.extend([str(row["prompt"])] * len(negatives))
        completions.extend(str(item["completion"]) for item in negatives)
    if expected_count is None:
        raise RuntimeError("Empty microbatch")
    return shared._stack_encoded(tokenizer, prompts, completions, max_length), expected_count


def _negative_stats(
    model: Any,
    batch: Mapping[str, Any],
    *,
    prompt_count: int,
    negatives_per_prompt: int,
) -> dict[str, Any]:
    stats = shared.completion_stats_batch(model, batch)
    mean_lp = stats["seq_lp"].reshape(prompt_count, negatives_per_prompt)
    lengths = stats["lengths"].reshape(prompt_count, negatives_per_prompt)
    return {
        "mean_lp": mean_lp,
        "sum_lp": mean_lp * lengths,
        "lengths": lengths,
    }


def _save_adapter(model: Any, tokenizer: Any, path: Path, adapter_name: str) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=False)
    try:
        model.save_pretrained(path, safe_serialization=True, selected_adapters=[adapter_name])
    except TypeError:
        model.set_adapter(adapter_name)
        model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)


def _parameter_update_norm(before: Sequence[Any], parameters: Sequence[Any]) -> float:
    total = torch.zeros((), dtype=torch.float64)
    for saved, parameter in zip(before, parameters, strict=True):
        total += (parameter.detach().float().cpu() - saved).double().square().sum()
    return float(torch.sqrt(total))


def _cell_identity(
    cell: Cell,
    *,
    config: Mapping[str, Any],
    exp_config: Mapping[str, Any],
    prepared: Mapping[str, Any],
    base_model_path: str,
) -> dict[str, Any]:
    record = prepared["inputs"][cell.task]
    identity = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "config_hash": stable_config_hash(config),
        "inherited_exp_config_hash": stable_config_hash(exp_config),
        "cell": {
            "task": cell.task,
            "method": cell.method,
            "parameter": cell.parameter,
            "seed": cell.seed,
        },
        "train_prompt_hash": record["train_prompt_hash"],
        "validation_prompt_hash": record["validation_prompt_hash"],
        "base_model_identity": model_identity(base_model_path, None)["model"],
        "reference_adapter_identity": model_identity(base_model_path, record["reference_adapter"])[
            "adapter"
        ],
    }
    identity["identity_hash"] = stable_config_hash(identity)
    return identity


def _train_cell_impl(
    cell: Cell,
    *,
    config: Mapping[str, Any],
    exp_config: Mapping[str, Any],
    prepared: Mapping[str, Any],
    inputs: shared.TaskInputs,
    output_root: Path,
    base_model_path: str,
    force: bool,
    updates_override: int | None = None,
    liveness: bool = False,
) -> dict[str, Any]:
    if torch is None or DataLoader is None:
        raise RuntimeError("Training requires Torch")
    root_name = "liveness" if liveness else "cells"
    cell_root = output_root / root_name / cell.key
    manifest_path = cell_root / "cell_manifest.json"
    identity = _cell_identity(
        cell,
        config=config,
        exp_config=exp_config,
        prepared=prepared,
        base_model_path=base_model_path,
    )
    identity["updates_override"] = updates_override
    identity["liveness"] = liveness
    identity["identity_hash"] = stable_config_hash(identity)
    if manifest_path.is_file() and not force:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("identity_hash") == identity["identity_hash"] and existing.get("complete"):
            return existing
        raise RuntimeError(f"Existing cell identity mismatch: {cell.key}")
    if cell_root.exists():
        if not force:
            raise RuntimeError(f"Cell output exists without reusable manifest: {cell_root}")
        shutil.rmtree(cell_root)
    cell_root.mkdir(parents=True, exist_ok=False)

    shared._seed_everything(cell.seed)
    model, tokenizer, scheduler_factory = shared._load_reference_model(
        base_model_path,
        inputs.reference_adapter,
        exp_config,
        train_mode=True,
    )
    is_topr = cell.method == METHOD_TOPR
    if is_topr:
        if not hasattr(model, "add_adapter") or not hasattr(model, "set_adapter"):
            raise RuntimeError("Joint fitted-reference TOPR requires PEFT multi-adapter support")
        if REFERENCE_ADAPTER in model.peft_config:
            raise RuntimeError("Reference adapter already exists before initialization")
        model.add_adapter(REFERENCE_ADAPTER, copy.deepcopy(model.peft_config[POLICY_ADAPTER]))
        _copy_adapter_parameters(model, POLICY_ADAPTER, REFERENCE_ADAPTER)
        model.set_adapter(POLICY_ADAPTER)
    policy_parameters = _adapter_parameters(model, POLICY_ADAPTER)
    reference_parameters = _adapter_parameters(model, REFERENCE_ADAPTER) if is_topr else []

    training = exp_config["training"]
    updates = int(updates_override or training["optimizer_updates"])
    accumulation = int(training["gradient_accumulation"])
    micro_batch = int(training["micro_batch"])
    train_rows = read_jsonl(Path(prepared["inputs"][cell.task]["train_path"]))
    validation_rows = read_jsonl(Path(prepared["inputs"][cell.task]["validation_path"]))
    dataset = shared.RowDataset(train_rows)
    generator = torch.Generator().manual_seed(cell.seed)
    loader = DataLoader(
        dataset,
        batch_size=micro_batch,
        shuffle=True,
        generator=generator,
        num_workers=0,
        collate_fn=lambda values: list(values),
        drop_last=True,
    )
    iterator = iter(loader)
    policy_optimizer = torch.optim.AdamW(
        policy_parameters,
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    reference_optimizer = None
    if is_topr:
        reference_optimizer = torch.optim.AdamW(
            reference_parameters,
            lr=float(training["learning_rate"])
            * float(config["topr"]["reference_learning_rate_multiplier"]),
            weight_decay=float(training["weight_decay"]),
        )
    warmup = max(1, int(updates * float(training["warmup_ratio"])))
    policy_scheduler = scheduler_factory(policy_optimizer, warmup, updates)
    reference_scheduler = (
        scheduler_factory(reference_optimizer, warmup, updates)
        if reference_optimizer is not None
        else None
    )
    device = next(model.parameters()).device
    max_length = int(exp_config["model"]["max_length"])
    adapter, instances = shared._load_task_adapter_and_instances(
        cell.task,
        inputs=inputs,
        validation_rows=validation_rows,
    )
    training_path = cell_root / "training_metrics.jsonl"
    evaluation_path = cell_root / "evaluation_metrics.jsonl"
    best_pass8 = -math.inf
    initial_ratio_max_abs: float | None = None
    first_policy_update_norm: float | None = None
    first_reference_update_norm: float | None = None

    model.train()
    for update in range(1, updates + 1):
        policy_optimizer.zero_grad(set_to_none=True)
        if reference_optimizer is not None:
            reference_optimizer.zero_grad(set_to_none=True)
        accum_metrics = {
            "policy_loss": 0.0,
            "reference_loss": 0.0,
            "positive_lp": 0.0,
            "negative_lp": 0.0,
            "weight_mean": 0.0,
            "log_ratio_mean": 0.0,
        }
        for accumulation_index in range(accumulation):
            try:
                rows = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                rows = next(iterator)
            prompts = [str(row["prompt"]) for row in rows]
            positives = [str(row["oracle_completion"]) for row in rows]
            positive_batch = shared._move_batch(
                shared._stack_encoded(tokenizer, prompts, positives, max_length),
                device,
            )
            negative_batch_cpu, negatives_per_prompt = _flatten_negative_batch(
                tokenizer, rows, max_length
            )
            negative_batch = shared._move_batch(negative_batch_cpu, device)
            prompt_count = len(rows)

            reference_loss = torch.zeros((), device=device)
            reference_negative_stats = None
            rng_state = None
            if is_topr:
                rng_state = _capture_rng_state()
                model.set_adapter(REFERENCE_ADAPTER)
                reference_positive_lp = shared.completion_stats_batch(model, positive_batch)[
                    "seq_lp"
                ].mean()
                reference_negative_stats = _negative_stats(
                    model,
                    negative_batch,
                    prompt_count=prompt_count,
                    negatives_per_prompt=negatives_per_prompt,
                )
                reference_negative_lp = reference_negative_stats["mean_lp"].mean()
                reference_loss = -0.5 * (reference_positive_lp + reference_negative_lp)
                if not bool(torch.isfinite(reference_loss)):
                    raise RuntimeError(f"{cell.key} non-finite reference loss at update {update}")
                (reference_loss / accumulation).backward()
                model.set_adapter(POLICY_ADAPTER)
                assert rng_state is not None
                _restore_rng_state(rng_state)

            positive_lp = shared.completion_stats_batch(model, positive_batch)["seq_lp"].mean()
            negative_stats = _negative_stats(
                model,
                negative_batch,
                prompt_count=prompt_count,
                negatives_per_prompt=negatives_per_prompt,
            )
            if is_topr:
                assert reference_negative_stats is not None
                log_ratio = negative_stats["sum_lp"] - reference_negative_stats["sum_lp"].detach()
                weights = torch.exp(
                    float(cell.parameter)
                    * torch.minimum(log_ratio.detach(), torch.zeros_like(log_ratio))
                )
                if update == 1 and accumulation_index == 0:
                    initial_ratio_max_abs = float(log_ratio.abs().max().detach().cpu())
                    tolerance = float(config["topr"]["initial_ratio_max_abs_tolerance"])
                    if initial_ratio_max_abs > tolerance:
                        raise RuntimeError(
                            "Initial TOPR policy/reference ratio mismatch: "
                            f"{initial_ratio_max_abs} > {tolerance}"
                        )
                weighted_negative_lp = (weights * negative_stats["mean_lp"]).mean()
                policy_loss = -(positive_lp - weighted_negative_lp)
                weight_mean = float(weights.mean().detach().cpu())
                log_ratio_mean = float(log_ratio.mean().detach().cpu())
            else:
                delta_v = float(cell.parameter)
                positive_coefficient = 1.0 - delta_v
                negative_coefficient = 1.0 + delta_v
                negative_lp = negative_stats["mean_lp"].mean()
                policy_loss = -(
                    positive_coefficient * positive_lp - negative_coefficient * negative_lp
                )
                weighted_negative_lp = negative_coefficient * negative_lp
                weight_mean = negative_coefficient
                log_ratio_mean = 0.0
            if not bool(torch.isfinite(policy_loss)):
                raise RuntimeError(f"{cell.key} non-finite policy loss at update {update}")
            (policy_loss / accumulation).backward()
            divisor = float(accumulation)
            accum_metrics["policy_loss"] += float(policy_loss.detach().cpu()) / divisor
            accum_metrics["reference_loss"] += float(reference_loss.detach().cpu()) / divisor
            accum_metrics["positive_lp"] += float(positive_lp.detach().cpu()) / divisor
            accum_metrics["negative_lp"] += float(weighted_negative_lp.detach().cpu()) / divisor
            accum_metrics["weight_mean"] += weight_mean / divisor
            accum_metrics["log_ratio_mean"] += log_ratio_mean / divisor

        model.set_adapter(POLICY_ADAPTER)
        policy_before = [
            parameter.detach().float().cpu().clone() for parameter in policy_parameters
        ]
        policy_grad_norm = torch.nn.utils.clip_grad_norm_(
            policy_parameters, float(training["max_grad_norm"])
        )
        if not bool(torch.isfinite(policy_grad_norm)):
            raise RuntimeError(f"{cell.key} non-finite policy gradient at update {update}")
        reference_grad_norm = torch.zeros(())
        reference_before: list[Any] = []
        if is_topr:
            model.set_adapter(REFERENCE_ADAPTER)
            reference_before = [
                parameter.detach().float().cpu().clone() for parameter in reference_parameters
            ]
            reference_grad_norm = torch.nn.utils.clip_grad_norm_(
                reference_parameters, float(training["max_grad_norm"])
            )
            if not bool(torch.isfinite(reference_grad_norm)):
                raise RuntimeError(f"{cell.key} non-finite reference gradient at update {update}")
            model.set_adapter(POLICY_ADAPTER)

        if not _joint_optimizer_step_with_finite_guard(
            policy_optimizer,
            reference_optimizer,
            policy_parameters,
            reference_parameters,
        ):
            raise RuntimeError(
                f"{cell.key} non-finite adapter parameter after optimizer step; "
                "both adapters were restored"
            )
        if reference_scheduler is not None:
            reference_scheduler.step()
        policy_scheduler.step()
        policy_optimizer.zero_grad(set_to_none=True)
        if reference_optimizer is not None:
            reference_optimizer.zero_grad(set_to_none=True)
        all_parameters = [*policy_parameters, *reference_parameters]
        if not all(bool(torch.isfinite(parameter).all()) for parameter in all_parameters):
            raise RuntimeError(f"{cell.key} non-finite adapter parameter at update {update}")
        policy_update_norm = _parameter_update_norm(policy_before, policy_parameters)
        reference_update_norm = (
            _parameter_update_norm(reference_before, reference_parameters) if is_topr else 0.0
        )
        if update == 1:
            first_policy_update_norm = policy_update_norm
            first_reference_update_norm = reference_update_norm
            if policy_update_norm <= 0.0:
                raise RuntimeError(f"{cell.key} policy adapter did not change")
            if is_topr and reference_update_norm <= 0.0:
                raise RuntimeError(f"{cell.key} reference adapter did not change")

        if update % 10 == 0 or update == updates:
            append_jsonl(
                training_path,
                {
                    "update": update,
                    **accum_metrics,
                    "policy_raw_gradient_norm": float(policy_grad_norm.detach().cpu()),
                    "reference_raw_gradient_norm": float(reference_grad_norm.detach().cpu()),
                    "policy_parameter_update_norm": policy_update_norm,
                    "reference_parameter_update_norm": reference_update_norm,
                    "learning_rate": float(policy_scheduler.get_last_lr()[0]),
                },
            )

        should_evaluate = not liveness and (
            update % int(training["evaluation_every_updates"]) == 0 or update == updates
        )
        if should_evaluate:
            model.set_adapter(POLICY_ADAPTER)
            metrics = shared.evaluate_model(
                model,
                tokenizer,
                task=cell.task,
                validation_rows=validation_rows,
                adapter=adapter,
                instances=instances,
                update=update,
                cell_seed=cell.seed,
                config=exp_config,
            )
            append_jsonl(evaluation_path, metrics)
            if float(metrics["pass8"]) > best_pass8:
                best_pass8 = float(metrics["pass8"])
                _save_adapter(
                    model,
                    tokenizer,
                    cell_root / "supplementary_best_policy_adapter",
                    POLICY_ADAPTER,
                )
            model.train()

    model.set_adapter(POLICY_ADAPTER)
    terminal_policy = cell_root / "terminal_policy_adapter"
    _save_adapter(model, tokenizer, terminal_policy, POLICY_ADAPTER)
    terminal_reference = None
    if is_topr:
        terminal_reference = cell_root / "terminal_reference_adapter"
        _save_adapter(model, tokenizer, terminal_reference, REFERENCE_ADAPTER)
    if liveness:
        summary: dict[str, Any] = {
            "evaluation_status": "not_applicable",
            "first_policy_update_norm": first_policy_update_norm,
            "first_reference_update_norm": first_reference_update_norm,
            "reload_gate_pending": True,
        }
        scientific_status = "not_run"
    else:
        summary = shared._summarize_evaluations(read_jsonl(evaluation_path), exp_config)
        summary["evaluation_status"] = "complete"
        scientific_status = "pilot"
    result = {
        **identity,
        **summary,
        "method": cell.method,
        "beta": cell.parameter if is_topr else None,
        "delta_v": cell.parameter if not is_topr else None,
        "positive_coefficient": 1.0 if is_topr else 1.0 - cell.parameter,
        "negative_repulsion_coefficient": None if is_topr else 1.0 + cell.parameter,
        "initial_ratio_max_abs": initial_ratio_max_abs,
        "optimizer_updates": updates,
        "terminal_policy_adapter": str(terminal_policy.resolve()),
        "terminal_reference_adapter": (
            str(terminal_reference.resolve()) if terminal_reference is not None else None
        ),
        "training_metrics": str(training_path.resolve()),
        "evaluation_metrics": str(evaluation_path.resolve()) if evaluation_path.is_file() else None,
        "nan_inf_failure": False,
        "complete": True,
        "test_partition_opened": False,
        "scientific_status": scientific_status,
    }
    atomic_json(manifest_path, result)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def train_cell(
    cell: Cell,
    **kwargs: Any,
) -> dict[str, Any]:
    output_root = Path(kwargs["output_root"])
    liveness = bool(kwargs.get("liveness", False))
    root_name = "liveness" if liveness else "cells"
    failure_root = output_root / root_name / cell.key
    try:
        return _train_cell_impl(cell, **kwargs)
    except Exception as exc:
        failure_root.mkdir(parents=True, exist_ok=True)
        atomic_json(
            failure_root / "failure.json",
            {
                "schema_version": 1,
                "experiment_id": EXPERIMENT_ID,
                "cell_key": cell.key,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "nan_inf_failure": "non-finite" in str(exc).lower(),
                "complete": False,
                "scientific_status": "not_run" if liveness else "pilot",
            },
        )
        raise


def cmd_train_cell(
    config: Mapping[str, Any],
    exp_config: Mapping[str, Any],
    output_root: Path,
    *,
    cell_key: str,
    base_model_path: str,
    force: bool,
) -> dict[str, Any]:
    prepared, inputs = _load_prepared(output_root, config, exp_config)
    cells = {cell.key: cell for cell in build_cells(config, exp_config)}
    if cell_key not in cells:
        raise ValueError(f"Unknown cell key: {cell_key}")
    cell = cells[cell_key]
    return train_cell(
        cell,
        config=config,
        exp_config=exp_config,
        prepared=prepared,
        inputs=inputs[cell.task],
        output_root=output_root,
        base_model_path=base_model_path,
        force=force,
    )


def cmd_liveness(
    config: Mapping[str, Any],
    exp_config: Mapping[str, Any],
    output_root: Path,
    *,
    task: str,
    base_model_path: str,
    force: bool,
) -> dict[str, Any]:
    if task not in config["suite"]["tasks"]:
        raise ValueError(f"Unknown liveness task: {task}")
    prepared, inputs = _load_prepared(output_root, config, exp_config)
    seed = int(config["sweep"]["tuning_seed"])
    representatives = (
        Cell(task, METHOD_TOPR, 0.25, seed),
        Cell(task, METHOD_ASYMRE, -0.5, seed),
    )
    results = []
    for cell in representatives:
        result = train_cell(
            cell,
            config=config,
            exp_config=exp_config,
            prepared=prepared,
            inputs=inputs[cell.task],
            output_root=output_root,
            base_model_path=base_model_path,
            force=force,
            updates_override=2,
            liveness=True,
        )
        try:
            from peft import PeftModel
            from transformers import AutoModelForCausalLM
        except ImportError as exc:
            raise RuntimeError("Liveness reload requires transformers and peft") from exc
        kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "torch_dtype": shared.resolve_torch_dtype(str(exp_config["model"]["dtype"])),
        }
        if torch.cuda.is_available():
            kwargs["device_map"] = {"": 0}
        base = AutoModelForCausalLM.from_pretrained(base_model_path, **kwargs)
        reloaded = PeftModel.from_pretrained(
            base, result["terminal_policy_adapter"], is_trainable=False
        )
        finite = all(
            bool(torch.isfinite(parameter).all())
            for parameter in reloaded.parameters()
            if parameter.is_floating_point()
        )
        if not finite:
            raise RuntimeError(f"Reloaded policy adapter is non-finite: {cell.key}")
        result["reload_gate_pending"] = False
        result["reload_gate_passed"] = True
        atomic_json(output_root / "liveness" / cell.key / "cell_manifest.json", result)
        results.append(result)
        del base, reloaded
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    manifest = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "task": task,
        "methods": [METHOD_TOPR, METHOD_ASYMRE],
        "results": results,
        "complete": all(result.get("reload_gate_passed") for result in results),
        "scientific_status": "not_run",
    }
    atomic_json(output_root / "liveness" / "liveness_manifest.json", manifest)
    return manifest


def _run_subprocess_cell(
    *,
    config_path: Path,
    exp_config_path: Path,
    output_root: Path,
    base_model_path: str,
    cell: Cell,
    gpu_id: int,
    force: bool,
) -> dict[str, Any]:
    log_path = output_root / "logs" / f"{cell.key}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "drpo.e8_multitask_baseline_tuning",
        "--config",
        str(config_path),
        "--exp-config",
        str(exp_config_path),
        "--output-root",
        str(output_root),
        "train-cell",
        "--cell-key",
        cell.key,
        "--base-model-path",
        base_model_path,
    ]
    if force:
        command.append("--force")
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu_id),
            "LOCAL_RANK": "0",
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": environment.get("OMP_NUM_THREADS", "4"),
        }
    )
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=environment,
            text=True,
            check=False,
        )
    return {
        "cell_key": cell.key,
        "gpu_id": gpu_id,
        "returncode": completed.returncode,
        "log": str(log_path.resolve()),
    }


def cmd_run_wave(
    config: Mapping[str, Any],
    exp_config: Mapping[str, Any],
    config_path: Path,
    exp_config_path: Path,
    output_root: Path,
    *,
    wave_index: int,
    base_model_path: str,
    force: bool,
) -> dict[str, Any]:
    liveness_path = output_root / "liveness" / "liveness_manifest.json"
    if not liveness_path.is_file():
        raise RuntimeError("Run both method liveness gates before a full wave")
    liveness = json.loads(liveness_path.read_text(encoding="utf-8"))
    if not liveness.get("complete"):
        raise RuntimeError("Method liveness gates are incomplete")
    waves = build_waves(config, exp_config)
    if not 1 <= wave_index <= len(waves):
        raise ValueError(f"wave must be in [1,{len(waves)}]")
    wave = waves[wave_index - 1]
    gpu_ids = tuple(int(value) for value in config["execution"]["gpu_ids"])
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(wave)) as executor:
        futures = {
            executor.submit(
                _run_subprocess_cell,
                config_path=config_path.resolve(),
                exp_config_path=exp_config_path.resolve(),
                output_root=output_root.resolve(),
                base_model_path=base_model_path,
                cell=cell,
                gpu_id=gpu_ids[index % len(gpu_ids)],
                force=force,
            ): cell
            for index, cell in enumerate(wave)
        }
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda row: str(row["cell_key"]))
    failures = [row for row in results if int(row["returncode"]) != 0]
    manifest = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "wave": wave_index,
        "expected_cells": len(wave),
        "results": results,
        "failed_cells": [row["cell_key"] for row in failures],
        "complete": not failures and len(results) == len(wave),
        "scientific_status": "pilot",
    }
    atomic_json(output_root / "waves" / f"wave_{wave_index:02d}.json", manifest)
    if failures:
        raise RuntimeError(f"Wave {wave_index} failed cells: {manifest['failed_cells']}")
    return manifest


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def cmd_aggregate(
    config: Mapping[str, Any],
    exp_config: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for cell in build_cells(config, exp_config):
        path = output_root / "cells" / cell.key / "cell_manifest.json"
        if not path.is_file():
            missing.append(cell.key)
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        if not value.get("complete") or value.get("evaluation_status") != "complete":
            missing.append(cell.key)
            continue
        rows.append(
            {
                "task": cell.task,
                "method": cell.method,
                "parameter": cell.parameter,
                "cell_key": cell.key,
                "late_window_pass8_mean": value["validation_late_window_pass8_mean"],
                "terminal_pass8": value["validation_terminal_pass8"],
                "late_window_greedy_mean": value["validation_late_window_greedy_mean"],
                "terminal_greedy": value["validation_terminal_greedy"],
                "terminal_greedy_valid_rate": value["validation_terminal_greedy_valid_rate"],
                "nan_inf_failure": bool(value["nan_inf_failure"]),
            }
        )
    if missing:
        raise RuntimeError(f"Cannot aggregate; missing/incomplete cells: {missing}")
    _write_csv(output_root / "aggregate" / "all_cells.csv", rows)

    minimum_valid = float(config["selection"]["terminal_valid_rate_minimum"])
    task_summaries: dict[str, Any] = {}
    selected_rows: list[dict[str, Any]] = []
    for task in config["suite"]["tasks"]:
        methods: dict[str, Any] = {}
        for method in (METHOD_TOPR, METHOD_ASYMRE):
            method_rows = [row for row in rows if row["task"] == task and row["method"] == method]
            if len(method_rows) != 5:
                raise RuntimeError(f"{task}/{method} does not contain five cells")
            boundary_parameter = 0.0 if method == METHOD_TOPR else -1.0
            boundary = next(
                row
                for row in method_rows
                if math.isclose(float(row["parameter"]), boundary_parameter)
            )
            active = [
                row
                for row in method_rows
                if not math.isclose(float(row["parameter"]), boundary_parameter)
                and not row["nan_inf_failure"]
                and float(row["terminal_greedy_valid_rate"]) >= minimum_valid
            ]
            selected = (
                max(
                    active,
                    key=lambda row: (
                        float(row["late_window_pass8_mean"]),
                        float(row["terminal_pass8"]),
                        float(row["late_window_greedy_mean"]),
                        float(row["terminal_greedy"]),
                        -abs(float(row["parameter"])),
                    ),
                )
                if active
                else None
            )
            all_active_below_boundary = bool(
                active
                and max(float(row["late_window_pass8_mean"]) for row in active)
                < float(boundary["late_window_pass8_mean"])
            )
            parameter_boundary_unclosed = bool(
                selected is not None
                and (
                    (method == METHOD_TOPR and math.isclose(float(selected["parameter"]), 0.5))
                    or (
                        method == METHOD_ASYMRE and math.isclose(float(selected["parameter"]), -0.9)
                    )
                )
            )
            methods[method] = {
                "boundary": boundary,
                "eligible_active_count": len(active),
                "selected_active": selected,
                "all_active_below_boundary": all_active_below_boundary,
                "parameter_boundary_unclosed": parameter_boundary_unclosed,
            }
            selected_rows.append(
                {
                    "task": task,
                    "method": method,
                    "selected_parameter": None if selected is None else selected["parameter"],
                    "selected_late_window_pass8_mean": (
                        None if selected is None else selected["late_window_pass8_mean"]
                    ),
                    "boundary_parameter": boundary_parameter,
                    "boundary_late_window_pass8_mean": boundary["late_window_pass8_mean"],
                    "all_active_below_boundary": all_active_below_boundary,
                    "parameter_boundary_unclosed": parameter_boundary_unclosed,
                }
            )
        task_summaries[str(task)] = methods
    _write_csv(output_root / "aggregate" / "selected_parameters.csv", selected_rows)
    summary = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "cell_count": len(rows),
        "tasks": task_summaries,
        "countdown_rerun": False,
        "positive_only_rerun": False,
        "test_partition_opened": False,
        "fixed_horizon_is_convergence": False,
        "scientific_status": "pilot",
        "claim_boundary": (
            "Eight-task development response curves only; no convergence, significance, "
            "cross-method ranking, or categorical causal-identification claim."
        ),
    }
    atomic_json(output_root / "aggregate" / "aggregate_summary.json", summary)
    return summary


def cmd_audit(
    config: Mapping[str, Any],
    exp_config: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    missing: list[str] = []
    incomplete: list[str] = []
    nan_inf: list[str] = []
    for cell in build_cells(config, exp_config):
        root = output_root / "cells" / cell.key
        manifest_path = root / "cell_manifest.json"
        if not manifest_path.is_file():
            failure_path = root / "failure.json"
            if failure_path.is_file():
                failure = json.loads(failure_path.read_text(encoding="utf-8"))
                incomplete.append(cell.key)
                if failure.get("nan_inf_failure"):
                    nan_inf.append(cell.key)
            else:
                missing.append(cell.key)
            continue
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not value.get("complete") or value.get("evaluation_status") != "complete":
            incomplete.append(cell.key)
        if value.get("nan_inf_failure"):
            nan_inf.append(cell.key)
        if value.get("test_partition_opened"):
            raise RuntimeError(f"Test partition access reported by {cell.key}")
    complete = not missing and not incomplete and not nan_inf
    audit = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "expected_cells": 80,
        "missing_cells": sorted(set(missing)),
        "incomplete_cells": sorted(set(incomplete)),
        "nan_inf_cells": sorted(set(nan_inf)),
        "all_training_and_evaluation_complete": complete,
        "task_performance_event": "validation_response_reported_no_posthoc_collapse_threshold",
        "structure_event": "valid_rate_diagnostic_only",
        "nan_inf_event_count": len(set(nan_inf)),
        "countdown_rerun": False,
        "positive_only_rerun": False,
        "test_partition_opened": False,
        "fixed_horizon_is_convergence": False,
        "scientific_status": "pilot" if complete else "not_run",
    }
    atomic_json(output_root / "terminal_audit.json", audit)
    return audit


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--exp-config", default=str(DEFAULT_EXP_CONFIG))
    parser.add_argument("--output-root", required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--shared-root", required=True)

    liveness = subparsers.add_parser("liveness")
    liveness.add_argument("--task", default="word_sorting")
    liveness.add_argument("--base-model-path", required=True)
    liveness.add_argument("--force", action="store_true")

    train = subparsers.add_parser("train-cell")
    train.add_argument("--cell-key", required=True)
    train.add_argument("--base-model-path", required=True)
    train.add_argument("--force", action="store_true")

    wave = subparsers.add_parser("run-wave")
    wave.add_argument("--wave", type=int, required=True)
    wave.add_argument("--base-model-path", required=True)
    wave.add_argument("--force", action="store_true")

    subparsers.add_parser("plan")
    subparsers.add_parser("aggregate")
    subparsers.add_parser("audit")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = make_parser().parse_args(argv)
    config_path = Path(args.config).resolve()
    exp_config_path = Path(args.exp_config).resolve()
    config, exp_config = load_config(config_path, exp_config_path)
    output_root = validate_work_dir(args.output_root)
    if args.command == "prepare":
        result = cmd_prepare(
            config,
            exp_config,
            output_root,
            shared_root=Path(args.shared_root).resolve(),
        )
    elif args.command == "liveness":
        result = cmd_liveness(
            config,
            exp_config,
            output_root,
            task=args.task,
            base_model_path=args.base_model_path,
            force=bool(args.force),
        )
    elif args.command == "train-cell":
        result = cmd_train_cell(
            config,
            exp_config,
            output_root,
            cell_key=args.cell_key,
            base_model_path=args.base_model_path,
            force=bool(args.force),
        )
    elif args.command == "run-wave":
        result = cmd_run_wave(
            config,
            exp_config,
            config_path,
            exp_config_path,
            output_root,
            wave_index=int(args.wave),
            base_model_path=args.base_model_path,
            force=bool(args.force),
        )
    elif args.command == "plan":
        result = write_plan(config, exp_config, output_root)
    elif args.command == "aggregate":
        result = cmd_aggregate(config, exp_config, output_root)
    elif args.command == "audit":
        result = cmd_audit(config, exp_config, output_root)
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
