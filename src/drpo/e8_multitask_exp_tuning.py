"""Exponential-taper response tuning on frozen multitask banks and references.

The module keeps the P0 occurrence/gradient diagnostic separate from downstream
method tuning.  It supports the original nine-task 72-cell rho sweep and its
seven-task, 112-cell task-local lambda refinement successor without changing the
training, validation, or no-test-access contracts.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import math
import os
import random
import shutil
import subprocess
import sys
import traceback
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except ImportError:  # Planning and split validation do not require Torch.
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    Dataset = object  # type: ignore[assignment,misc]

from drpo.e8_multitask_p0 import (
    append_jsonl,
    atomic_json,
    atomic_jsonl,
    bank_path,
    collate_encoded_completions,
    encode_prompt_completion,
    format_chat_prompt,
    model_identity,
    read_jsonl,
    resolve_torch_dtype,
    sha256_file,
    stable_config_hash,
    train_task_positive_warmstart,
    validate_work_dir,
    with_smoke_overrides,
)
from drpo.e8_multitask_tasks import (
    TASK_NAMES,
    TaskInstance,
    build_adapters,
    stable_hash,
)

EXPERIMENT_ID = "EXT-C-E8-MULTITASK-EXP-TUNING-01"
DENSE_EXPERIMENT_ID = "EXT-C-E8-MULTITASK-EXP-LAMBDA-DENSE-01"
SUPPORTED_EXPERIMENT_IDS = (EXPERIMENT_ID, DENSE_EXPERIMENT_ID)
P0_EXPERIMENT_ID = "EXT-C-E8-MULTITASK-P0-01"
# Backward-compatible name used by predecessor tests and downstream callers.
PARENT_EXPERIMENT_ID = P0_EXPERIMENT_ID
DEFAULT_CONFIG = Path("configs/e8_multitask_exp_tuning.yaml")
DEFAULT_P0_CONFIG = Path("configs/e8_multitask_p0.yaml")
METHOD_POSITIVE_ONLY = "positive_only"
METHOD_EXPONENTIAL = "exponential"
SWEEP_PROFILE_RHO = "nine_task_rho_v1"
SWEEP_PROFILE_DENSE = "task_lambda_dense_v1"


@dataclass(frozen=True)
class Cell:
    task: str
    method: str
    rho: float | None
    seed: int
    stage: str
    lambda_value: float | None = None

    @property
    def key(self) -> str:
        if self.method == METHOD_POSITIVE_ONLY:
            return f"{self.task}__positive_only__seed{self.seed}"
        if self.rho is None:
            raise AssertionError("Exponential cell requires rho")
        if self.lambda_value is not None:
            tag = f"{self.lambda_value:.12g}".replace(".", "p")
            return f"{self.task}__exp_lambda{tag}__seed{self.seed}"
        tag = f"{self.rho:.6f}".rstrip("0").rstrip(".").replace(".", "p")
        return f"{self.task}__exp_rho{tag}__seed{self.seed}"


@dataclass(frozen=True)
class TaskInputs:
    task: str
    bank: Path
    reference_adapter: Path | None
    sources_root: Path
    p0_config: Path
    countdown_validation: Path | None = None


class RowDataset(Dataset):
    def __init__(self, rows: Sequence[Mapping[str, Any]]) -> None:
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        return self.rows[index]


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("Configuration root must be a mapping")
    validate_config(value)
    return value


def _tuple_floats(values: Sequence[Any]) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def experiment_id(config: Mapping[str, Any]) -> str:
    value = str(config.get("experiment_id", ""))
    if value not in SUPPORTED_EXPERIMENT_IDS:
        raise ValueError(f"Unsupported experiment_id: {value}")
    return value


def sweep_profile(config: Mapping[str, Any]) -> str:
    return str(config.get("sweep", {}).get("profile", SWEEP_PROFILE_RHO))


def _is_dense(config: Mapping[str, Any]) -> bool:
    return sweep_profile(config) == SWEEP_PROFILE_DENSE


def _dense_tasks() -> set[str]:
    return set(TASK_NAMES) - {"countdown", "spiral_matrix"}


def _task_lambdas(config: Mapping[str, Any], task: str) -> tuple[float, ...]:
    if not _is_dense(config):
        raise ValueError("Task-local lambdas are only defined for the dense profile")
    values = _tuple_floats(config["sweep"]["task_lambda"][task])
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError(f"{task} lambda values must be finite and positive")
    return values


def _task_rhos(config: Mapping[str, Any], task: str) -> tuple[float, ...]:
    if _is_dense(config):
        return tuple(math.exp(-value) for value in _task_lambdas(config, task))
    return _tuple_floats(config["sweep"]["all_rho"])


def _reference_seed(
    config: Mapping[str, Any],
    warmstart_config: Mapping[str, Any],
    task: str,
) -> int:
    configured = config.get("reference", {}).get("task_seeds")
    if isinstance(configured, Mapping):
        if task not in configured:
            raise ValueError(f"reference.task_seeds is missing {task}")
        return int(configured[task])
    p0_tasks = tuple(str(value) for value in config["suite"]["p0_tasks"])
    return int(warmstart_config["seed"]) + p0_tasks.index(task) * 100_003


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("Expected schema_version: 1")
    current_experiment = experiment_id(config)
    profile = sweep_profile(config)
    if profile not in (SWEEP_PROFILE_RHO, SWEEP_PROFILE_DENSE):
        raise ValueError(f"Unsupported sweep profile: {profile}")
    expected_parent = EXPERIMENT_ID if profile == SWEEP_PROFILE_DENSE else P0_EXPERIMENT_ID
    if config.get("parent", {}).get("experiment_id") != expected_parent:
        raise ValueError("Unexpected parent experiment")

    tasks = tuple(config.get("suite", {}).get("tasks", ()))
    if profile == SWEEP_PROFILE_RHO:
        if len(tasks) != 9 or len(set(tasks)) != 9 or set(tasks) != set(TASK_NAMES):
            raise ValueError("The rho suite must be Countdown plus the exact eight P0 tasks")
        if set(config["suite"].get("p0_tasks", ())) != set(TASK_NAMES) - {"countdown"}:
            raise ValueError("suite.p0_tasks must be the exact eight P0 tasks")
        if tuple(config["suite"].get("external_tasks", ())) != ("countdown",):
            raise ValueError("Countdown must be the only external task")
    else:
        if (
            current_experiment != DENSE_EXPERIMENT_ID
            or len(tasks) != 7
            or len(set(tasks)) != 7
            or set(tasks) != _dense_tasks()
        ):
            raise ValueError(
                "The dense suite must be the exact seven non-Countdown, non-Spiral tasks"
            )
        if tuple(config["suite"].get("p0_tasks", ())) != tasks:
            raise ValueError("Dense suite.p0_tasks must preserve the exact task order")
        if tuple(config["suite"].get("external_tasks", ())) != ():
            raise ValueError("Dense refinement has no external Countdown task")
    reference = config["reference"]
    if reference["checkpoint_kind"] != "train_only_task_positive_warmstart_100":
        raise ValueError("P0 task references must be train-only 100-update warm starts")
    if int(reference["optimizer_updates"]) != 100:
        raise ValueError("Train-only P0 references must remain fixed at 100 updates")
    if int(reference["validation_rows_seen"]) != 0 or int(reference["test_rows_seen"]) != 0:
        raise ValueError("Train-only reference preparation must not see validation or test rows")

    split = config["split"]
    expected_split = {
        "p0_train_rows": 5000,
        "p0_validation_rows": 500,
        "p0_test_rows": 500,
    }
    if profile == SWEEP_PROFILE_RHO:
        expected_split.update(
            {
                "countdown_train_rows": 5000,
                "countdown_validation_rows": 500,
            }
        )
    for key, expected in expected_split.items():
        if int(split[key]) != expected:
            raise ValueError(f"{key} must remain {expected}")
    if bool(split.get("test_access_allowed", True)):
        raise ValueError("Tuning must forbid test access")

    training = config["training"]
    if int(training["optimizer_updates"]) != 1200:
        raise ValueError("The tuning horizon must remain 1200 updates")
    if int(training["micro_batch"]) != 1 or int(training["gradient_accumulation"]) != 8:
        raise ValueError("The method-training effective prompt batch must remain 8")
    if not math.isclose(
        float(training["learning_rate"]),
        5.0e-5,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("The method-training learning rate must remain 5e-5")
    if not math.isclose(
        float(training["warmup_ratio"]),
        0.03,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("The method-training warmup ratio must remain 0.03")
    if int(training["evaluation_every_updates"]) != 100:
        raise ValueError("Evaluation cadence must remain 100 updates")
    if bool(training.get("early_stopping", True)):
        raise ValueError("Early stopping is forbidden")
    if tuple(int(value) for value in training["late_window_updates"]) != (
        800,
        900,
        1000,
        1100,
        1200,
    ):
        raise ValueError("Unexpected late-window updates")

    evaluation = config["evaluation"]
    if int(evaluation["greedy_prompt_rows"]) != 500:
        raise ValueError("Greedy validation must use 500 prompts")
    if int(evaluation["passk_prompt_rows"]) != 128 or int(evaluation["pass_k"]) != 8:
        raise ValueError("Pass@8 validation must use the frozen 128-prompt subset")

    negative = config["negative_sampling"]
    if int(negative["negatives_per_prompt"]) != 16:
        raise ValueError("Every training prompt must retain exactly 16 negatives")
    if _tuple_floats(negative["near_far_mix"]) != (0.5, 0.5):
        raise ValueError("Near/far branch mass must remain 0.5/0.5")
    if not bool(negative["selection_stop_gradient"]):
        raise ValueError("Current near/far selection must be stop-gradient")
    if bool(negative["weight_sum_normalization"]):
        raise ValueError("Weight-sum normalization is forbidden")

    calibration = config["remoteness_calibration"]
    if not math.isclose(
        float(calibration["target_negative_to_positive_gradient_ratio"]),
        1.0 / 32.0,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("Initial negative-gradient target must remain 1/32 of positive")

    sweep = config["sweep"]
    if profile == SWEEP_PROFILE_RHO:
        if _tuple_floats(sweep["coarse_rho"]) != (0.9, 0.6, 0.35, 0.125):
            raise ValueError("Unexpected coarse rho grid")
        if _tuple_floats(sweep["refinement_rho"]) != (0.75, 0.5, 0.25):
            raise ValueError("Unexpected refinement rho grid")
        if _tuple_floats(sweep["all_rho"]) != (
            0.9,
            0.75,
            0.6,
            0.5,
            0.35,
            0.25,
            0.125,
        ):
            raise ValueError("Unexpected full rho grid")
        if int(sweep["positive_only_per_task"]) != 1 or int(sweep["expected_cells"]) != 72:
            raise ValueError("The rho matrix must be 7 Exp plus 1 Positive-only per task")
    else:
        task_lambda = sweep.get("task_lambda")
        bridges = sweep.get("bridge_lambda")
        if not isinstance(task_lambda, Mapping) or set(task_lambda) != set(tasks):
            raise ValueError("Dense task_lambda must contain the exact seven tasks")
        if not isinstance(bridges, Mapping) or set(bridges) != set(tasks):
            raise ValueError("Dense bridge_lambda must contain the exact seven tasks")
        for task in tasks:
            values = _task_lambdas(config, task)
            if len(values) != 16 or len(set(values)) != 16:
                raise ValueError(f"{task} must contain 16 unique lambda values")
            bridge = float(bridges[task])
            if bridge not in values:
                raise ValueError(f"{task} bridge lambda must be one of its 16 cells")
        if int(sweep["positive_only_per_task"]) != 0 or int(sweep["expected_cells"]) != 112:
            raise ValueError("The dense matrix must be 16 Exp cells for each of seven tasks")
        if int(sweep["tuning_seed"]) != 2026072904:
            raise ValueError("Dense shape discovery must preserve the predecessor tuning seed")

    execution = config["execution"]
    if int(execution["max_concurrent_cells"]) != 16:
        raise ValueError("The scheduler must expose exactly 16 slots")
    if tuple(int(value) for value in execution["gpu_ids"]) != tuple(range(8)):
        raise ValueError("The default GPU pool must remain 0--7")
    expected_waves = 7 if profile == SWEEP_PROFILE_DENSE else 5
    if int(execution["slots_per_gpu"]) != 2 or int(execution["expected_waves"]) != expected_waves:
        raise ValueError(f"The frozen topology is two slots per GPU and {expected_waves} waves")


def coefficient_from_rho(rho: float) -> float:
    if not math.isfinite(rho) or not 0.0 < rho < 1.0:
        raise ValueError("rho must be finite and strictly between zero and one")
    return -math.log(rho)


def normalized_distance(
    sequence_log_probability: torch.Tensor,
    *,
    tau: float,
    scale: float,
) -> torch.Tensor:
    if not math.isfinite(tau) or tau < 0.0:
        raise ValueError("tau must be finite and non-negative")
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be finite and positive")
    return torch.sqrt(torch.relu(-sequence_log_probability.detach() - tau) / scale)


def taper_weight(distance: torch.Tensor, rho: float) -> torch.Tensor:
    return torch.exp(-coefficient_from_rho(rho) * distance)


def build_cells(config: Mapping[str, Any]) -> tuple[Cell, ...]:
    validate_config(config)
    tasks = tuple(str(task) for task in config["suite"]["tasks"])
    seed = int(config["sweep"]["tuning_seed"])
    if _is_dense(config):
        cells = tuple(
            Cell(
                task,
                METHOD_EXPONENTIAL,
                math.exp(-lambda_value),
                seed,
                "dense",
                lambda_value,
            )
            for task in tasks
            for lambda_value in _task_lambdas(config, task)
        )
        if len(cells) != 112 or len({cell.key for cell in cells}) != 112:
            raise AssertionError("Internal 112-cell identity failure")
        return cells
    coarse = _tuple_floats(config["sweep"]["coarse_rho"])
    refinement = _tuple_floats(config["sweep"]["refinement_rho"])
    cells: list[Cell] = []
    for task in tasks:
        cells.append(Cell(task, METHOD_POSITIVE_ONLY, None, seed, "coarse"))
        cells.extend(Cell(task, METHOD_EXPONENTIAL, rho, seed, "coarse") for rho in coarse)
    for task in tasks:
        cells.extend(Cell(task, METHOD_EXPONENTIAL, rho, seed, "refinement") for rho in refinement)
    if len(cells) != 72 or len({cell.key for cell in cells}) != 72:
        raise AssertionError("Internal 72-cell identity failure")
    return tuple(cells)


def build_waves(config: Mapping[str, Any]) -> tuple[tuple[Cell, ...], ...]:
    cells = build_cells(config)
    capacity = int(config["execution"]["max_concurrent_cells"])
    if _is_dense(config):
        waves = tuple(
            tuple(cell for cell in cells if cell.task == str(task))
            for task in config["suite"]["tasks"]
        )
        if len(waves) != 7 or any(len(wave) != capacity for wave in waves):
            raise AssertionError("Dense wave geometry must be seven task-local 16-cell waves")
        return waves
    coarse = tuple(cell for cell in cells if cell.stage == "coarse")
    refinement = tuple(cell for cell in cells if cell.stage == "refinement")

    def chunk(values: tuple[Cell, ...]) -> tuple[tuple[Cell, ...], ...]:
        return tuple(
            tuple(values[index : index + capacity]) for index in range(0, len(values), capacity)
        )

    waves = chunk(coarse) + chunk(refinement)
    if tuple(len(wave) for wave in waves) != (16, 16, 13, 16, 11):
        raise AssertionError("Frozen wave geometry must be 16/16/13/16/11")
    if any(cell.stage != "coarse" for wave in waves[:3] for cell in wave):
        raise AssertionError("Refinement leaked into the first three waves")
    if any(cell.stage != "refinement" for wave in waves[3:] for cell in wave):
        raise AssertionError("Coarse cells leaked into the last two waves")
    return waves


def write_plan(config: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    waves = build_waves(config)
    gpu_ids = tuple(int(value) for value in config["execution"]["gpu_ids"])
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
                    "rho": cell.rho,
                    "lambda": (
                        cell.lambda_value
                        if cell.lambda_value is not None
                        else (None if cell.rho is None else coefficient_from_rho(cell.rho))
                    ),
                    "seed": cell.seed,
                    "stage": cell.stage,
                }
            )
    plan = {
        "schema_version": 1,
        "experiment_id": experiment_id(config),
        "config_hash": stable_config_hash(config),
        "cell_count": len(rows),
        "wave_count": len(waves),
        "wave_sizes": [len(wave) for wave in waves],
        "max_concurrent_cells": int(config["execution"]["max_concurrent_cells"]),
        "rows": rows,
        "scientific_status": "not_run",
    }
    atomic_json(output_root / "plan.json", plan)
    atomic_jsonl(output_root / "plan.jsonl", rows)
    return plan


def _ordered_by_prompt_hash(
    rows: Sequence[Mapping[str, Any]],
    *,
    task: str,
    seed: int,
    role: str,
) -> list[dict[str, Any]]:
    return sorted(
        (dict(row) for row in rows),
        key=lambda row: stable_hash(
            {
                "task": task,
                "prompt_id": str(row["prompt_id"]),
                "seed": seed,
                "role": role,
            }
        ),
    )


def _normalize_p0_row(row: Mapping[str, Any]) -> dict[str, Any]:
    negatives = [dict(item) for item in row["negatives"]]
    normalized = dict(row)
    normalized["prompt_id"] = str(row["prompt_id"])
    normalized["oracle_completion"] = str(row["oracle_completion"])
    normalized["negatives"] = [
        {
            **item,
            "negative_id": str(item["negative_id"]),
            "completion": str(item["completion"]),
        }
        for item in negatives
    ]
    return normalized


def _normalize_countdown_train_row(row: Mapping[str, Any]) -> dict[str, Any]:
    negatives = []
    for index, item_value in enumerate(row["negatives"]):
        item = dict(item_value)
        negatives.append(
            {
                **item,
                "negative_id": f"{row['row_id']}_neg_{index:03d}",
                "completion": str(item["expression"]),
                "format_valid": bool(item.get("valid_format", True)),
                "binary_correct": bool(item.get("correct", False)),
                "error_class": str(item.get("negative_bin", item.get("source", "wrong_answer"))),
            }
        )
    return {
        "schema_version": 1,
        "task": "countdown",
        "prompt_id": str(row["row_id"]),
        "source_prompt_id": str(row.get("source_prompt_id", row["row_id"])),
        "prompt": str(row["prompt"]),
        "oracle_completion": str(row["oracle_positive"]),
        "metadata": {
            "numbers": [int(value) for value in row["numbers"]],
            "target": int(row["target"]),
        },
        "negatives": negatives,
        "source_schema": "countdown_oracle_offline_bank_v2",
    }


def _normalize_countdown_validation_row(row: Mapping[str, Any]) -> dict[str, Any]:
    prompt_id = str(row.get("id", row.get("row_id", row.get("source_prompt_id", ""))))
    if not prompt_id:
        raise RuntimeError("Countdown validation row has no stable ID")
    oracle = row.get("oracle", row.get("oracle_positive"))
    if oracle is None:
        raise RuntimeError(f"Countdown validation row {prompt_id} has no oracle")
    return {
        "schema_version": 1,
        "task": "countdown",
        "prompt_id": prompt_id,
        "prompt": str(row["prompt"]),
        "oracle_completion": str(oracle),
        "metadata": {
            "numbers": [int(value) for value in row["numbers"]],
            "target": int(row["target"]),
        },
        "source_schema": "countdown_structural_validation",
    }


def _audit_training_rows(task: str, rows: Sequence[Mapping[str, Any]], expected: int) -> None:
    if len(rows) != expected:
        raise RuntimeError(f"{task} expected {expected} training rows, found {len(rows)}")
    prompt_ids = [str(row["prompt_id"]) for row in rows]
    if len(set(prompt_ids)) != len(prompt_ids):
        raise RuntimeError(f"{task} has duplicate prompt IDs")
    for row in rows:
        negatives = list(row.get("negatives", ()))
        if len(negatives) != 16:
            raise RuntimeError(
                f"{task}/{row['prompt_id']} must have exactly 16 negatives, found {len(negatives)}"
            )
        completions = [str(item["completion"]) for item in negatives]
        if task != "countdown" and len(set(completions)) != 16:
            raise RuntimeError(f"{task}/{row['prompt_id']} has duplicate negative completions")
        if any(bool(item.get("binary_correct", item.get("correct", False))) for item in negatives):
            raise RuntimeError(f"{task}/{row['prompt_id']} contains a verifier-correct negative")


def _audit_partition_prompt_ids(
    task: str,
    partitions: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    seen: dict[str, str] = {}
    for partition, rows in partitions.items():
        prompt_ids = [str(row["prompt_id"]) for row in rows]
        if len(set(prompt_ids)) != len(prompt_ids):
            raise RuntimeError(f"{task} has duplicate prompt IDs within {partition}")
        for prompt_id in prompt_ids:
            previous = seen.get(prompt_id)
            if previous is not None:
                raise RuntimeError(
                    f"{task} prompt ID {prompt_id} overlaps {previous} and {partition}"
                )
            seen[prompt_id] = partition


def split_p0_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    task: str,
    config: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    split = config["split"]
    required = (
        int(split["p0_train_rows"]) + int(split["p0_validation_rows"]) + int(split["p0_test_rows"])
    )
    if len(rows) != required:
        raise RuntimeError(f"{task} P0 bank must contain exactly {required} rows")
    ordered = _ordered_by_prompt_hash(
        [_normalize_p0_row(row) for row in rows],
        task=task,
        seed=int(split["hash_seed"]),
        role="p0_tuning_split",
    )
    train_end = int(split["p0_train_rows"])
    validation_end = train_end + int(split["p0_validation_rows"])
    partitions = {
        "train": ordered[:train_end],
        "validation": ordered[train_end:validation_end],
        "test": ordered[validation_end:],
    }
    _audit_training_rows(task, partitions["train"], int(split["p0_train_rows"]))
    _audit_partition_prompt_ids(task, partitions)
    return partitions


def split_countdown_rows(
    train_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    split = config["split"]
    normalized_train = [_normalize_countdown_train_row(row) for row in train_rows]
    normalized_validation = [_normalize_countdown_validation_row(row) for row in validation_rows]
    train = _ordered_by_prompt_hash(
        normalized_train,
        task="countdown",
        seed=int(split["hash_seed"]),
        role="countdown_train_select",
    )[: int(split["countdown_train_rows"])]
    validation = _ordered_by_prompt_hash(
        normalized_validation,
        task="countdown",
        seed=int(split["hash_seed"]),
        role="countdown_validation_select",
    )[: int(split["countdown_validation_rows"])]
    _audit_training_rows("countdown", train, int(split["countdown_train_rows"]))
    if len(validation) != int(split["countdown_validation_rows"]):
        raise RuntimeError("Countdown validation file does not contain 500 rows")
    partitions = {"train": train, "validation": validation}
    _audit_partition_prompt_ids("countdown", partitions)
    return partitions


def resolve_task_inputs(
    config: Mapping[str, Any],
    *,
    p0_work_dir: Path,
    p0_config: Path,
    countdown_bank: Path,
    countdown_validation: Path,
    countdown_adapter: Path,
) -> dict[str, TaskInputs]:
    p0_config_value = yaml.safe_load(p0_config.read_text(encoding="utf-8"))
    if (
        not isinstance(p0_config_value, dict)
        or p0_config_value.get("experiment_id") != P0_EXPERIMENT_ID
    ):
        raise RuntimeError("P0 config identity mismatch")
    qualification_path = p0_work_dir / "qualification_audit.json"
    if not qualification_path.is_file():
        raise FileNotFoundError(f"Missing P0 qualification audit: {qualification_path}")
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    if (
        qualification.get("experiment_id") != P0_EXPERIMENT_ID
        or qualification.get("config_hash")
        != stable_config_hash(
            with_smoke_overrides(
                p0_config_value,
                rows=None,
                negatives=None,
            )
        )
        or not qualification.get("passed")
    ):
        raise RuntimeError("P0 bank qualification identity or pass status mismatch")

    result: dict[str, TaskInputs] = {}
    for task_value in config["suite"]["p0_tasks"]:
        task = str(task_value)
        if not qualification.get("tasks", {}).get(task, {}).get("passed", False):
            raise RuntimeError(f"P0 bank is not qualified for {task}")
        result[task] = TaskInputs(
            task=task,
            bank=bank_path(p0_work_dir, task).resolve(),
            reference_adapter=None,
            sources_root=(p0_work_dir / "sources").resolve(),
            p0_config=p0_config.resolve(),
        )
    result["countdown"] = TaskInputs(
        task="countdown",
        bank=countdown_bank.resolve(),
        reference_adapter=countdown_adapter.resolve(),
        sources_root=(p0_work_dir / "sources").resolve(),
        p0_config=p0_config.resolve(),
        countdown_validation=countdown_validation.resolve(),
    )
    for task, inputs in result.items():
        if not inputs.bank.is_file():
            raise FileNotFoundError(f"Missing bank for {task}: {inputs.bank}")
        if task == "countdown":
            if (
                inputs.reference_adapter is None
                or not (inputs.reference_adapter / "adapter_config.json").is_file()
            ):
                raise FileNotFoundError("Missing supplied Countdown reference adapter")
            if inputs.countdown_validation is None or not inputs.countdown_validation.is_file():
                raise FileNotFoundError("Missing Countdown validation file")
        if not inputs.sources_root.is_dir():
            raise FileNotFoundError(f"Missing sources root for {task}: {inputs.sources_root}")
    if set(result) != set(config["suite"]["tasks"]):
        raise AssertionError("Resolved inputs do not match the configured suite")
    return result


def write_split_manifest(
    task_inputs: Mapping[str, TaskInputs],
    config: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    task_records: dict[str, Any] = {}
    for task_value in config["suite"]["tasks"]:
        task = str(task_value)
        inputs = task_inputs[task]
        if task == "countdown":
            if inputs.countdown_validation is None:
                raise AssertionError("Countdown validation path missing")
            partitions = split_countdown_rows(
                read_jsonl(inputs.bank),
                read_jsonl(inputs.countdown_validation),
                config=config,
            )
        else:
            partitions = split_p0_rows(read_jsonl(inputs.bank), task=task, config=config)
        task_root = output_root / "splits" / task
        task_root.mkdir(parents=True, exist_ok=True)
        paths: dict[str, str] = {}
        prompt_hashes: dict[str, str] = {}
        counts: dict[str, int] = {}
        for name, values in partitions.items():
            path = task_root / f"{name}.jsonl"
            atomic_jsonl(path, values)
            paths[name] = str(path.resolve())
            counts[name] = len(values)
            prompt_hashes[name] = stable_hash(sorted(str(row["prompt_id"]) for row in values))
        task_records[task] = {
            "bank": str(inputs.bank),
            "bank_sha256": sha256_file(inputs.bank),
            "reference_adapter": (
                str(inputs.reference_adapter) if inputs.reference_adapter is not None else None
            ),
            "reference_adapter_identity": (
                model_identity("unresolved_backbone", str(inputs.reference_adapter))["adapter"]
                if inputs.reference_adapter is not None
                else None
            ),
            "sources_root": str(inputs.sources_root),
            "p0_config": str(inputs.p0_config),
            "p0_config_sha256": sha256_file(inputs.p0_config),
            "countdown_validation_source": (
                str(inputs.countdown_validation) if inputs.countdown_validation else None
            ),
            "countdown_validation_sha256": (
                sha256_file(inputs.countdown_validation) if inputs.countdown_validation else None
            ),
            "counts": counts,
            "prompt_id_hashes": prompt_hashes,
            "paths": paths,
        }
    manifest = {
        "schema_version": 1,
        "experiment_id": experiment_id(config),
        "config_hash": stable_config_hash(config),
        "test_access_allowed": False,
        "tasks": task_records,
        "complete": len(task_records) == len(config["suite"]["tasks"]),
        "scientific_status": "not_run",
    }
    atomic_json(output_root / "split_manifest.json", manifest)
    return manifest


def cmd_prepare(
    config: Mapping[str, Any],
    output_root: Path,
    *,
    p0_work_dir: Path,
    p0_config: Path,
    countdown_bank: Path,
    countdown_validation: Path,
    countdown_adapter: Path,
) -> dict[str, Any]:
    if _is_dense(config):
        raise RuntimeError("Dense refinement must use inherit, not prepare")
    inputs = resolve_task_inputs(
        config,
        p0_work_dir=p0_work_dir,
        p0_config=p0_config,
        countdown_bank=countdown_bank,
        countdown_validation=countdown_validation,
        countdown_adapter=countdown_adapter,
    )
    plan = write_plan(config, output_root)
    splits = write_split_manifest(inputs, config, output_root)
    serialized_inputs = {
        task: {
            "bank": str(value.bank),
            "reference_adapter": (
                str(value.reference_adapter) if value.reference_adapter is not None else None
            ),
            "sources_root": str(value.sources_root),
            "p0_config": str(value.p0_config),
            "countdown_validation": (
                str(value.countdown_validation) if value.countdown_validation else None
            ),
        }
        for task, value in inputs.items()
    }
    manifest = {
        "schema_version": 1,
        "experiment_id": experiment_id(config),
        "config_hash": stable_config_hash(config),
        "plan": str((output_root / "plan.json").resolve()),
        "split_manifest": str((output_root / "split_manifest.json").resolve()),
        "inputs": serialized_inputs,
        "complete": plan["cell_count"] == int(config["sweep"]["expected_cells"])
        and splits["complete"],
        "scientific_status": "not_run",
    }
    atomic_json(output_root / "prepare_manifest.json", manifest)
    return manifest


def _parent_response_rows(
    parent_config: Mapping[str, Any],
    parent_output_root: Path,
    tasks: set[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    expected_hash = stable_config_hash(parent_config)
    for cell in build_cells(parent_config):
        if cell.task not in tasks:
            continue
        path = parent_output_root / "cells" / cell.key / "cell_manifest.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing predecessor cell manifest: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
        if (
            value.get("experiment_id") != experiment_id(parent_config)
            or value.get("config_hash") != expected_hash
            or not value.get("complete")
            or value.get("evaluation_status") != "complete"
            or value.get("nan_inf_failure") is not False
        ):
            raise RuntimeError(f"Predecessor cell is not reusable: {cell.key}")
        rows.append(
            {
                "source": "predecessor",
                "task": cell.task,
                "method": cell.method,
                "rho": cell.rho,
                "lambda": None if cell.rho is None else coefficient_from_rho(cell.rho),
                "seed": cell.seed,
                "cell_key": cell.key,
                "late_window_pass8_mean": value["validation_late_window_pass8_mean"],
                "terminal_pass8": value["validation_terminal_pass8"],
                "late_window_greedy_mean": value["validation_late_window_greedy_mean"],
                "terminal_greedy": value["validation_terminal_greedy"],
                "terminal_greedy_valid_rate": value["validation_terminal_greedy_valid_rate"],
                "nan_inf_failure": False,
            }
        )
    expected = len(tasks) * 8
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} predecessor response rows, found {len(rows)}")
    return rows


def cmd_inherit(
    config: Mapping[str, Any],
    output_root: Path,
    *,
    parent_output_root: Path,
    parent_config_path: Path,
    base_model_path: str,
) -> dict[str, Any]:
    if not _is_dense(config):
        raise RuntimeError("inherit is only valid for the dense refinement profile")
    parent_config = load_config(parent_config_path)
    parent_contract = config["parent"]
    if (
        experiment_id(parent_config) != EXPERIMENT_ID
        or stable_config_hash(parent_config) != parent_contract["config_hash"]
        or int(parent_config["sweep"]["expected_cells"]) != int(parent_contract["expected_cells"])
    ):
        raise RuntimeError("Predecessor config identity mismatch")

    parent_artifacts = {
        "plan": parent_output_root / "plan.json",
        "split_manifest": parent_output_root / "split_manifest.json",
        "reference_manifest": reference_manifest_path(parent_output_root),
        "aggregate_summary": parent_output_root / "aggregate" / "aggregate_summary.json",
    }
    for name, path in parent_artifacts.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing predecessor {name}: {path}")
        expected_sha = str(parent_contract["artifact_sha256"][name])
        if sha256_file(path) != expected_sha:
            raise RuntimeError(f"Predecessor {name} does not match the delivered result")

    parent_plan = json.loads(parent_artifacts["plan"].read_text(encoding="utf-8"))
    parent_aggregate = json.loads(parent_artifacts["aggregate_summary"].read_text(encoding="utf-8"))
    if (
        parent_plan.get("experiment_id") != EXPERIMENT_ID
        or parent_plan.get("config_hash") != parent_contract["config_hash"]
        or int(parent_plan.get("cell_count", 0)) != int(parent_contract["expected_cells"])
        or parent_aggregate.get("experiment_id") != EXPERIMENT_ID
        or parent_aggregate.get("test_partition_accessed") is not False
        or int(parent_aggregate.get("cell_count", 0)) != int(parent_contract["expected_cells"])
    ):
        raise RuntimeError("Predecessor plan or aggregate contract mismatch")

    parent_splits, parent_inputs = _load_ready_inputs(
        parent_output_root,
        parent_config,
        base_model_path=base_model_path,
    )
    tasks = tuple(str(task) for task in config["suite"]["tasks"])
    child_config_hash = stable_config_hash(config)
    child_split_tasks = {task: copy.deepcopy(parent_splits["tasks"][task]) for task in tasks}
    child_splits = {
        "schema_version": 1,
        "experiment_id": experiment_id(config),
        "config_hash": child_config_hash,
        "test_access_allowed": False,
        "tasks": child_split_tasks,
        "complete": True,
        "scientific_status": "not_run",
        "inherited_from": {
            "experiment_id": EXPERIMENT_ID,
            "run_id": parent_contract["run_id"],
            "result_commit": parent_contract["result_commit"],
            "split_manifest_sha256": parent_contract["artifact_sha256"]["split_manifest"],
        },
    }
    atomic_json(output_root / "split_manifest.json", child_splits)

    plan = write_plan(config, output_root)
    serialized_inputs = {
        task: {
            "bank": str(parent_inputs[task].bank),
            "reference_adapter": None,
            "sources_root": str(parent_inputs[task].sources_root),
            "p0_config": str(parent_inputs[task].p0_config),
            "countdown_validation": None,
        }
        for task in tasks
    }
    prepare = {
        "schema_version": 1,
        "experiment_id": experiment_id(config),
        "config_hash": child_config_hash,
        "plan": str((output_root / "plan.json").resolve()),
        "split_manifest": str((output_root / "split_manifest.json").resolve()),
        "inputs": serialized_inputs,
        "complete": plan["cell_count"] == int(config["sweep"]["expected_cells"]),
        "scientific_status": "not_run",
        "inherited_from": {
            "experiment_id": EXPERIMENT_ID,
            "run_id": parent_contract["run_id"],
            "result_commit": parent_contract["result_commit"],
        },
    }
    atomic_json(output_root / "prepare_manifest.json", prepare)

    base_identity = model_identity(base_model_path, None)["model"]
    warmstart = _reference_warmstart_config(config, parent_inputs[tasks[0]].p0_config)
    parent_reference = json.loads(
        parent_artifacts["reference_manifest"].read_text(encoding="utf-8")
    )
    inherited_reference_tasks: dict[str, Any] = {}
    for task in tasks:
        parent_task = copy.deepcopy(parent_reference["tasks"][task])
        expected_identity = _reference_identity(
            task=task,
            config=config,
            split_manifest=child_splits,
            warmstart_config=warmstart,
            base_model_identity=base_identity,
            seed=_reference_seed(config, warmstart, task),
        )
        parent_task.update(expected_identity)
        parent_task["inherited_from"] = {
            "experiment_id": EXPERIMENT_ID,
            "run_id": parent_contract["run_id"],
            "parent_identity_hash": parent_reference["tasks"][task]["identity_hash"],
            "parent_task_manifest_sha256": sha256_file(
                parent_output_root / "references" / task / "task_manifest.json"
            ),
        }
        inherited_reference_tasks[task] = parent_task
        atomic_json(output_root / "references" / task / "task_manifest.json", parent_task)
    child_reference = _reference_manifest_payload(
        config=config,
        base_model_identity=base_identity,
        tasks=inherited_reference_tasks,
    )
    child_reference["inherited_from"] = {
        "experiment_id": EXPERIMENT_ID,
        "run_id": parent_contract["run_id"],
        "result_commit": parent_contract["result_commit"],
        "reference_manifest_sha256": parent_contract["artifact_sha256"]["reference_manifest"],
    }
    atomic_json(reference_manifest_path(output_root), child_reference)

    response_rows = _parent_response_rows(parent_config, parent_output_root, set(tasks))
    parent_response = {
        "schema_version": 1,
        "experiment_id": experiment_id(config),
        "config_hash": child_config_hash,
        "parent_experiment_id": EXPERIMENT_ID,
        "parent_run_id": parent_contract["run_id"],
        "parent_result_commit": parent_contract["result_commit"],
        "rows": response_rows,
        "complete": True,
    }
    atomic_json(output_root / "inherited" / "parent_response.json", parent_response)
    snapshot = {
        "schema_version": 1,
        "experiment_id": experiment_id(config),
        "config_hash": child_config_hash,
        "parent_run_id": parent_contract["run_id"],
        "parent_result_repository": parent_contract["result_repository"],
        "parent_result_commit": parent_contract["result_commit"],
        "parent_source_commit": parent_contract["source_commit"],
        "parent_config_hash": parent_contract["config_hash"],
        "parent_artifact_sha256": dict(parent_contract["artifact_sha256"]),
        "tasks": list(tasks),
        "excluded_tasks": dict(config["suite"]["excluded_tasks"]),
        "inherited_split": True,
        "inherited_train_only_references": True,
        "inherited_positive_only_anchor": True,
        "calibration_must_be_rerun": True,
        "test_partition_accessed": False,
        "complete": True,
        "scientific_status": "not_run",
    }
    atomic_json(output_root / "inherited" / "parent_snapshot.json", snapshot)
    _load_ready_inputs(output_root, config, base_model_path=base_model_path)
    return snapshot


def reference_manifest_path(output_root: Path) -> Path:
    return output_root / "references" / "reference_manifest.json"


def _reference_warmstart_config(
    config: Mapping[str, Any],
    p0_config_path: Path,
) -> dict[str, Any]:
    p0_config = yaml.safe_load(p0_config_path.read_text(encoding="utf-8"))
    if not isinstance(p0_config, dict) or p0_config.get("experiment_id") != P0_EXPERIMENT_ID:
        raise RuntimeError("P0 config identity mismatch during reference preparation")
    inherited = copy.deepcopy(p0_config.get("positive_warmstart"))
    if not isinstance(inherited, dict):
        raise TypeError("P0 positive-warm-start contract is missing")
    if inherited.get("checkpoint_kind") != "task_positive_warmstart_100":
        raise RuntimeError("Unexpected inherited P0 checkpoint kind")
    if str(inherited.get("parameterization")) != "lora":
        raise RuntimeError("P0 reference preparation must inherit LoRA parameterization")
    if int(inherited.get("optimizer_updates", 0)) != int(config["reference"]["optimizer_updates"]):
        raise RuntimeError("Inherited P0 reference optimizer-update contract mismatch")
    if (
        int(inherited.get("micro_batch", 0)) != 2
        or int(inherited.get("gradient_accumulation", 0)) != 32
    ):
        raise RuntimeError("Inherited reference warm start must remain 2 x 32")

    model_contract = {
        "lora_rank": int(config["model"]["lora_rank"]),
        "lora_alpha": int(config["model"]["lora_alpha"]),
        "lora_dropout": float(config["model"]["lora_dropout"]),
        "max_length": int(config["model"]["max_length"]),
        "gradient_checkpointing": bool(config["model"]["gradient_checkpointing"]),
        "dtype": str(config["model"]["dtype"]),
    }
    inherited_contract = {
        "lora_rank": int(inherited["lora_rank"]),
        "lora_alpha": int(inherited["lora_alpha"]),
        "lora_dropout": float(inherited["lora_dropout"]),
        "max_length": int(inherited["max_length"]),
        "gradient_checkpointing": bool(inherited["gradient_checkpointing"]),
        "dtype": str(inherited["dtype"]),
    }
    if inherited_contract != model_contract:
        raise RuntimeError("Inherited P0 LoRA/model contract does not match tuning config")

    inherited["checkpoint_kind"] = str(config["reference"]["checkpoint_kind"])
    return inherited


def _reference_identity(
    *,
    task: str,
    config: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
    warmstart_config: Mapping[str, Any],
    base_model_identity: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    train_path = Path(split_manifest["tasks"][task]["paths"]["train"])
    identity = {
        "schema_version": 1,
        "experiment_id": experiment_id(config),
        "task": task,
        "config_hash": stable_config_hash(config),
        "p0_config_sha256": split_manifest["tasks"][task]["p0_config_sha256"],
        "train_prompt_hash": split_manifest["tasks"][task]["prompt_id_hashes"]["train"],
        "train_rows_sha256": sha256_file(train_path),
        "base_model_identity": base_model_identity,
        "reference_config": dict(warmstart_config),
        "seed": seed,
    }
    identity["identity_hash"] = stable_hash(identity)
    return identity


def _reference_manifest_payload(
    *,
    config: Mapping[str, Any],
    base_model_identity: Mapping[str, Any],
    tasks: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    expected_tasks = tuple(str(task) for task in config["suite"]["p0_tasks"])
    complete = set(tasks) == set(expected_tasks) and all(
        bool(tasks[task].get("complete"))
        and bool(tasks[task].get("train_only_reference"))
        and int(tasks[task].get("validation_rows_seen", -1)) == 0
        and int(tasks[task].get("test_rows_seen", -1)) == 0
        for task in expected_tasks
    )
    return {
        "schema_version": 1,
        "experiment_id": experiment_id(config),
        "config_hash": stable_config_hash(config),
        "base_model_identity": base_model_identity,
        "checkpoint_kind": str(config["reference"]["checkpoint_kind"]),
        "tasks": dict(tasks),
        "complete": complete,
        "validation_rows_seen": 0,
        "test_rows_seen": 0,
        "scientific_status": "not_run",
    }


def _validate_reference_manifest_header(
    manifest: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    base_model_identity: Mapping[str, Any],
) -> None:
    if (
        manifest.get("experiment_id") != experiment_id(config)
        or manifest.get("config_hash") != stable_config_hash(config)
        or manifest.get("checkpoint_kind") != config["reference"]["checkpoint_kind"]
        or manifest.get("base_model_identity") != base_model_identity
        or int(manifest.get("validation_rows_seen", -1)) != 0
        or int(manifest.get("test_rows_seen", -1)) != 0
    ):
        raise RuntimeError("Train-only reference manifest identity or leakage audit mismatch")
    recorded_tasks = manifest.get("tasks")
    if not isinstance(recorded_tasks, dict):
        raise TypeError("Train-only reference manifest tasks are malformed")
    unknown = set(recorded_tasks) - set(config["suite"]["p0_tasks"])
    if unknown:
        raise RuntimeError(f"Train-only reference manifest has unknown tasks: {sorted(unknown)}")


def _validate_reference_task_manifest(
    task_manifest: Mapping[str, Any],
    *,
    expected_identity: Mapping[str, Any],
    expected_train_rows: int,
) -> Path:
    if (
        task_manifest.get("identity_hash") != expected_identity["identity_hash"]
        or task_manifest.get("checkpoint_kind")
        != expected_identity["reference_config"]["checkpoint_kind"]
        or not task_manifest.get("complete")
        or not task_manifest.get("train_only_reference")
        or int(task_manifest.get("train_rows_seen", -1)) != expected_train_rows
        or int(task_manifest.get("validation_rows_seen", -1)) != 0
        or int(task_manifest.get("test_rows_seen", -1)) != 0
    ):
        raise RuntimeError(
            f"Train-only reference identity or leakage audit mismatch for "
            f"{expected_identity['task']}"
        )
    adapter = Path(str(task_manifest.get("adapter_path", "")))
    if not (adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(
            f"Missing train-only adapter for {expected_identity['task']}: {adapter}"
        )
    return adapter


def cmd_reference(
    config: Mapping[str, Any],
    output_root: Path,
    *,
    base_model_path: str,
    tasks: Sequence[str] | None,
    force: bool,
) -> dict[str, Any]:
    splits, inputs = _load_prepared(output_root, config)
    p0_tasks = tuple(str(task) for task in config["suite"]["p0_tasks"])
    requested = tuple(str(task) for task in (tasks or p0_tasks))
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("Reference tasks must be a non-empty unique list")
    unknown = sorted(set(requested) - set(p0_tasks))
    if unknown:
        raise ValueError(f"Only configured P0 tasks require train-only references: {unknown}")

    p0_config_path = inputs[requested[0]].p0_config
    if any(inputs[task].p0_config != p0_config_path for task in requested):
        raise RuntimeError("P0 tasks do not share one frozen config path")
    warmstart_config = _reference_warmstart_config(config, p0_config_path)
    base_identity = model_identity(base_model_path, None)["model"]
    task_seeds = {task: _reference_seed(config, warmstart_config, task) for task in p0_tasks}

    root = output_root / "references"
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = reference_manifest_path(output_root)
    completed: dict[str, Any] = {}
    if manifest_path.is_file():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _validate_reference_manifest_header(
            existing_manifest,
            config=config,
            base_model_identity=base_identity,
        )
        for task, recorded in existing_manifest["tasks"].items():
            task_manifest_path = root / task / "task_manifest.json"
            if not task_manifest_path.is_file():
                raise FileNotFoundError(
                    f"Reference manifest points to a missing task manifest: {task}"
                )
            task_manifest = json.loads(task_manifest_path.read_text(encoding="utf-8"))
            if task_manifest != recorded:
                raise RuntimeError(f"Reference task/top-level manifest mismatch for {task}")
            expected = _reference_identity(
                task=task,
                config=config,
                split_manifest=splits,
                warmstart_config=warmstart_config,
                base_model_identity=base_identity,
                seed=task_seeds[task],
            )
            try:
                _validate_reference_task_manifest(
                    task_manifest,
                    expected_identity=expected,
                    expected_train_rows=int(config["split"]["p0_train_rows"]),
                )
            except (FileNotFoundError, RuntimeError):
                if not force or task not in requested:
                    raise
                continue
            completed[task] = task_manifest

    for task in requested:
        task_root = root / task
        train_path = Path(splits["tasks"][task]["paths"]["train"])
        identity = _reference_identity(
            task=task,
            config=config,
            split_manifest=splits,
            warmstart_config=warmstart_config,
            base_model_identity=base_identity,
            seed=task_seeds[task],
        )
        if task in completed and not force:
            continue
        if task_root.exists():
            if not force:
                raise RuntimeError(f"Reference directory exists without reusable identity: {task}")
            if root.resolve() not in task_root.resolve().parents:
                raise RuntimeError(f"Refusing unsafe reference removal: {task_root}")
            shutil.rmtree(task_root)
        task_root.mkdir(parents=True, exist_ok=False)
        train_rows = read_jsonl(train_path)
        result = train_task_positive_warmstart(
            task=task,
            rows=train_rows,
            model_path=base_model_path,
            output_dir=task_root,
            warmstart_config=warmstart_config,
            seed=task_seeds[task],
        )
        result.update(identity)
        result.update(
            {
                "train_only_reference": True,
                "train_rows_seen": len(train_rows),
                "validation_rows_seen": 0,
                "test_rows_seen": 0,
            }
        )
        atomic_json(task_root / "task_manifest.json", result)
        completed[task] = result
        atomic_json(
            manifest_path,
            _reference_manifest_payload(
                config=config,
                base_model_identity=base_identity,
                tasks=completed,
            ),
        )

    manifest = _reference_manifest_payload(
        config=config,
        base_model_identity=base_identity,
        tasks=completed,
    )
    atomic_json(manifest_path, manifest)
    return manifest


def _load_prepared(
    output_root: Path,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, TaskInputs]]:
    prepare_path = output_root / "prepare_manifest.json"
    split_path = output_root / "split_manifest.json"
    if not prepare_path.is_file() or not split_path.is_file():
        raise RuntimeError("Run prepare before calibration or training")
    prepare = json.loads(prepare_path.read_text(encoding="utf-8"))
    splits = json.loads(split_path.read_text(encoding="utf-8"))
    expected_hash = stable_config_hash(config)
    if (
        prepare.get("experiment_id") != experiment_id(config)
        or splits.get("experiment_id") != experiment_id(config)
        or prepare.get("config_hash") != expected_hash
        or splits.get("config_hash") != expected_hash
        or not prepare.get("complete")
        or not splits.get("complete")
    ):
        raise RuntimeError("Prepared input identity mismatch")
    expected_tasks = set(config["suite"]["tasks"])
    if set(prepare.get("inputs", {})) != expected_tasks or set(splits.get("tasks", {})) != (
        expected_tasks
    ):
        raise RuntimeError("Prepared input task set mismatch")
    inputs = {
        task: TaskInputs(
            task=task,
            bank=Path(value["bank"]),
            reference_adapter=(
                Path(value["reference_adapter"]) if value.get("reference_adapter") else None
            ),
            sources_root=Path(value["sources_root"]),
            p0_config=Path(value["p0_config"]),
            countdown_validation=(
                Path(value["countdown_validation"]) if value.get("countdown_validation") else None
            ),
        )
        for task, value in prepare["inputs"].items()
    }
    for task, inputs_for_task in inputs.items():
        split_record = splits["tasks"][task]
        if not inputs_for_task.bank.is_file() or sha256_file(inputs_for_task.bank) != (
            split_record.get("bank_sha256")
        ):
            raise RuntimeError(f"Prepared bank identity mismatch for {task}")
        if not inputs_for_task.p0_config.is_file() or sha256_file(
            inputs_for_task.p0_config
        ) != split_record.get("p0_config_sha256"):
            raise RuntimeError(f"Prepared P0 config identity mismatch for {task}")
        if not inputs_for_task.sources_root.is_dir():
            raise FileNotFoundError(f"Prepared sources root is missing for {task}")
        if inputs_for_task.countdown_validation is not None and (
            not inputs_for_task.countdown_validation.is_file()
            or sha256_file(inputs_for_task.countdown_validation)
            != split_record.get("countdown_validation_sha256")
        ):
            raise RuntimeError("Prepared Countdown validation identity mismatch")
        if inputs_for_task.reference_adapter is not None:
            if not (inputs_for_task.reference_adapter / "adapter_config.json").is_file():
                raise FileNotFoundError(f"Prepared reference adapter is missing for {task}")
            current_identity = model_identity(
                "unresolved_backbone",
                str(inputs_for_task.reference_adapter),
            )["adapter"]
            if current_identity != split_record.get("reference_adapter_identity"):
                raise RuntimeError(f"Prepared reference adapter identity mismatch for {task}")
    return splits, inputs


def _attach_references(
    output_root: Path,
    config: Mapping[str, Any],
    splits: Mapping[str, Any],
    inputs: Mapping[str, TaskInputs],
    *,
    base_model_path: str,
) -> dict[str, TaskInputs]:
    path = reference_manifest_path(output_root)
    if not path.is_file():
        raise RuntimeError("Run train-only reference preparation before calibration")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    base_identity = model_identity(base_model_path, None)["model"]
    _validate_reference_manifest_header(
        manifest,
        config=config,
        base_model_identity=base_identity,
    )
    p0_tasks = tuple(str(task) for task in config["suite"]["p0_tasks"])
    if not manifest.get("complete") or set(manifest["tasks"]) != set(p0_tasks):
        raise RuntimeError("Train-only reference manifest is incomplete")

    p0_config_path = inputs[p0_tasks[0]].p0_config
    warmstart_config = _reference_warmstart_config(config, p0_config_path)
    attached: dict[str, TaskInputs] = {}
    for task, value in inputs.items():
        if task == "countdown":
            if value.reference_adapter is None:
                raise RuntimeError("Countdown supplied reference adapter is missing")
            attached[task] = value
            continue

        task_manifest_path = output_root / "references" / task / "task_manifest.json"
        if not task_manifest_path.is_file():
            raise FileNotFoundError(f"Missing train-only task manifest for {task}")
        task_manifest = json.loads(task_manifest_path.read_text(encoding="utf-8"))
        if task_manifest != manifest["tasks"].get(task):
            raise RuntimeError(f"Reference task/top-level manifest mismatch for {task}")
        expected_identity = _reference_identity(
            task=task,
            config=config,
            split_manifest=splits,
            warmstart_config=warmstart_config,
            base_model_identity=base_identity,
            seed=_reference_seed(config, warmstart_config, task),
        )
        adapter = _validate_reference_task_manifest(
            task_manifest,
            expected_identity=expected_identity,
            expected_train_rows=int(config["split"]["p0_train_rows"]),
        )
        if (
            task_manifest.get("adapter_identity")
            != model_identity(
                base_model_path,
                str(adapter),
            )["adapter"]
        ):
            raise RuntimeError(f"Train-only adapter content identity mismatch for {task}")
        attached[task] = TaskInputs(
            task=value.task,
            bank=value.bank,
            reference_adapter=adapter,
            sources_root=value.sources_root,
            p0_config=value.p0_config,
            countdown_validation=value.countdown_validation,
        )
    return attached


def _load_ready_inputs(
    output_root: Path,
    config: Mapping[str, Any],
    *,
    base_model_path: str,
) -> tuple[dict[str, Any], dict[str, TaskInputs]]:
    splits, inputs = _load_prepared(output_root, config)
    return splits, _attach_references(
        output_root,
        config,
        splits,
        inputs,
        base_model_path=base_model_path,
    )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
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
    reference_adapter: Path,
    config: Mapping[str, Any],
    *,
    train_mode: bool,
) -> tuple[Any, Any, Any]:
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


def _raw_gradient_norm(grads: Sequence[torch.Tensor | None]) -> float:
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
    selected = _ordered_by_prompt_hash(rows, task=task, seed=seed, role=role)[:count]
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
    if inputs.reference_adapter is None:
        raise RuntimeError(f"Reference adapter is missing for {task}")
    identity = {
        "schema_version": 1,
        "experiment_id": experiment_id(config),
        "config_hash": stable_config_hash(config),
        "task": task,
        "bank_sha256": split_manifest["tasks"][task]["bank_sha256"],
        "train_prompt_hash": split_manifest["tasks"][task]["prompt_id_hashes"]["train"],
        "base_model_identity": model_identity(base_model_path, None)["model"],
        "reference_adapter_identity": model_identity(
            base_model_path,
            str(inputs.reference_adapter),
        )["adapter"],
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
        if existing.get("identity_hash") == identity["identity_hash"] and existing.get("complete"):
            return existing
        raise RuntimeError(f"Existing calibration identity mismatch for {task}")

    if inputs.reference_adapter is None:
        raise RuntimeError(f"Reference adapter is missing for {task}")
    model, tokenizer, _ = _load_reference_model(
        base_model_path,
        inputs.reference_adapter,
        config,
        train_mode=False,
    )
    model.eval()
    train_rows = read_jsonl(Path(split_manifest["tasks"][task]["paths"]["train"]))
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
        raise RuntimeError(f"{task} degenerate remoteness calibration: tau={tau}, scale={scale}")

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
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    positive_lp = completion_stats_batch(model, positive_batch)["seq_lp"]
    positive_grads = torch.autograd.grad(-positive_lp.mean(), trainable, allow_unused=True)
    positive_norm = _raw_gradient_norm(positive_grads)
    if not math.isfinite(positive_norm) or positive_norm <= 0.0:
        raise RuntimeError(f"{task} positive calibration gradient is not finite and positive")

    near_batch, far_batch, _, _ = _select_current_extremes(
        model,
        tokenizer,
        gradient_rows,
        max_length=max_length,
    )
    near_lp = completion_stats_batch(model, _move_batch(near_batch, device))["seq_lp"]
    far_lp = completion_stats_batch(model, _move_batch(far_batch, device))["seq_lp"]
    near_distance = normalized_distance(near_lp, tau=tau, scale=scale)
    far_distance = normalized_distance(far_lp, tau=tau, scale=scale)
    rhos = _task_rhos(config, task)
    raw_negative_norms: dict[str, float] = {}
    for index, rho in enumerate(rhos):
        scalar = 0.5 * (taper_weight(near_distance, rho).detach() * near_lp).mean()
        scalar = scalar + 0.5 * (taper_weight(far_distance, rho).detach() * far_lp).mean()
        grads = torch.autograd.grad(
            scalar,
            trainable,
            allow_unused=True,
            retain_graph=index < len(rhos) - 1,
        )
        raw_negative_norms[f"{rho:.12g}"] = _raw_gradient_norm(grads)
    target_ratio = float(calibration["target_negative_to_positive_gradient_ratio"])
    target_negative_norm = positive_norm * target_ratio
    negative_scales: dict[str, float] = {}
    for key, raw_norm in raw_negative_norms.items():
        if not math.isfinite(raw_norm) or raw_norm <= 0.0:
            raise RuntimeError(f"{task} invalid raw negative gradient for rho={key}: {raw_norm}")
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
        "complete": True,
        "scientific_status": "not_run",
        "note": "Training-only initialization calibration; not scientific outcome evidence.",
    }
    atomic_json(path, result)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def cmd_calibrate(
    config: Mapping[str, Any],
    output_root: Path,
    *,
    base_model_path: str,
    tasks: Sequence[str] | None,
    force: bool,
) -> dict[str, Any]:
    splits, inputs = _load_ready_inputs(
        output_root,
        config,
        base_model_path=base_model_path,
    )
    requested = list(tasks or config["suite"]["tasks"])
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("Calibration tasks must be a non-empty unique list")
    unknown = sorted(set(requested) - set(config["suite"]["tasks"]))
    if unknown:
        raise ValueError(f"Unknown calibration tasks: {unknown}")
    requested_results = {
        task: calibrate_task(
            task,
            inputs=inputs[task],
            split_manifest=splits,
            base_model_path=base_model_path,
            config=config,
            output_root=output_root,
            force=force,
        )
        for task in requested
    }
    results: dict[str, Any] = {}
    for task in config["suite"]["tasks"]:
        path = output_root / "calibration" / f"{task}.json"
        if not path.is_file():
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        expected_identity = _calibration_identity(
            task,
            inputs=inputs[task],
            split_manifest=splits,
            base_model_path=base_model_path,
            config=config,
        )
        if result.get("identity_hash") != expected_identity["identity_hash"] or not result.get(
            "complete"
        ):
            if task in requested_results:
                raise RuntimeError(f"Calibration identity mismatch after writing {task}")
            continue
        results[task] = result
    expected_tasks = set(config["suite"]["tasks"])
    manifest = {
        "schema_version": 1,
        "experiment_id": experiment_id(config),
        "config_hash": stable_config_hash(config),
        "requested_tasks": requested,
        "tasks": results,
        "complete": set(results) == expected_tasks
        and all(result.get("complete") for result in results.values()),
        "scientific_status": "not_run",
    }
    atomic_json(output_root / "calibration" / "calibration_manifest.json", manifest)
    return manifest


def _load_task_adapter_and_instances(
    task: str,
    *,
    inputs: TaskInputs,
    validation_rows: Sequence[Mapping[str, Any]],
) -> tuple[Any, dict[str, TaskInstance]]:
    p0_config = yaml.safe_load(inputs.p0_config.read_text(encoding="utf-8"))
    if not isinstance(p0_config, dict):
        raise TypeError("P0 config root is not a mapping")
    adapter_config = copy.deepcopy(p0_config)
    adapter_config["tasks"]["names"] = [task]
    adapter = build_adapters(adapter_config, inputs.sources_root)[task]
    if task == "countdown":
        instances = {
            str(row["prompt_id"]): TaskInstance(
                task="countdown",
                prompt_id=str(row["prompt_id"]),
                prompt=str(row["prompt"]),
                oracle_completion=str(row["oracle_completion"]),
                metadata=dict(row["metadata"]),
                source_entry={},
            )
            for row in validation_rows
        }
        return adapter, instances

    seeds = {int(row["generation_seed"]) for row in validation_rows}
    if len(seeds) != 1:
        raise RuntimeError(f"{task} validation rows do not share one generation seed")
    required_ids = {str(row["prompt_id"]) for row in validation_rows}
    candidate_count = int(p0_config["bank"]["candidate_rows_per_task"])
    instances: dict[str, TaskInstance] = {}
    for instance in adapter.generate_instances(candidate_count, seeds.pop()):
        if instance.prompt_id in required_ids:
            instances[instance.prompt_id] = instance
            if len(instances) == len(required_ids):
                break
    missing = sorted(required_ids - set(instances))
    if missing:
        raise RuntimeError(f"Could not reconstruct {task} validation instances: {missing[:5]}")
    return adapter, instances


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
            formatted = [format_chat_prompt(tokenizer, prompt) for prompt in current]
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
                        "temperature": float(evaluation["sampling_temperature"]),
                        "top_p": float(evaluation["top_p"]),
                    }
                )
            generated = model.generate(**tokenized, **generation_kwargs)
            suffixes = generated[:, input_width:]
            decoded = tokenizer.batch_decode(suffixes, skip_special_tokens=True)
            for row_index in range(len(current)):
                left = row_index * num_return_sequences
                outputs.append(decoded[left : left + num_return_sequences])
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
        for row, completions in zip(greedy_rows, greedy_outputs, strict=True)
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
    greedy_lengths = [len(completions[0]) for completions in greedy_outputs]
    metrics = {
        "update": update,
        "greedy_prompt_rows": len(greedy_rows),
        "passk_prompt_rows": len(passk_rows),
        "pass_k": k,
        "greedy_success": float(np.mean([result.correct for result in greedy_results])),
        "greedy_valid_rate": float(np.mean([result.format_valid for result in greedy_results])),
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
        math.isfinite(float(value)) for key, value in metrics.items() if key not in integer_fields
    ):
        raise RuntimeError(f"{task} produced non-finite validation metrics at update {update}")
    return metrics


def _cell_identity(
    cell: Cell,
    *,
    inputs: TaskInputs,
    split_manifest: Mapping[str, Any],
    base_model_path: str,
    config: Mapping[str, Any],
    calibration: Mapping[str, Any],
) -> dict[str, Any]:
    if inputs.reference_adapter is None:
        raise RuntimeError(f"Reference adapter is missing for {cell.task}")
    value = {
        "schema_version": 1,
        "experiment_id": experiment_id(config),
        "config_hash": stable_config_hash(config),
        "cell": {
            "task": cell.task,
            "method": cell.method,
            "rho": cell.rho,
            "lambda": (
                cell.lambda_value
                if cell.lambda_value is not None
                else (None if cell.rho is None else coefficient_from_rho(cell.rho))
            ),
            "seed": cell.seed,
            "stage": cell.stage,
        },
        "bank_sha256": split_manifest["tasks"][cell.task]["bank_sha256"],
        "split_prompt_hashes": split_manifest["tasks"][cell.task]["prompt_id_hashes"],
        "base_model_identity": model_identity(base_model_path, None)["model"],
        "reference_adapter_identity": model_identity(
            base_model_path, str(inputs.reference_adapter)
        )["adapter"],
        "calibration_identity_hash": calibration["identity_hash"],
    }
    value["identity_hash"] = stable_hash(value)
    return value


def _summarize_evaluations(
    evaluations: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    by_update = {int(row["update"]): row for row in evaluations}
    late_updates = tuple(int(value) for value in config["training"]["late_window_updates"])
    missing = [update for update in late_updates if update not in by_update]
    if missing:
        raise RuntimeError(f"Missing late-window evaluations: {missing}")
    terminal_update = int(config["training"]["optimizer_updates"])
    if terminal_update not in by_update:
        raise RuntimeError("Missing terminal evaluation")
    late_rows = [by_update[update] for update in late_updates]
    terminal = by_update[terminal_update]
    best = max(
        evaluations,
        key=lambda row: (
            float(row["pass8"]),
            float(row["greedy_success"]),
            int(row["update"]),
        ),
    )
    return {
        "late_window_updates": list(late_updates),
        "validation_late_window_pass8_mean": float(
            np.mean([float(row["pass8"]) for row in late_rows])
        ),
        "validation_late_window_greedy_mean": float(
            np.mean([float(row["greedy_success"]) for row in late_rows])
        ),
        "validation_late_window_valid_mean": float(
            np.mean([float(row["greedy_valid_rate"]) for row in late_rows])
        ),
        "validation_terminal_pass8": float(terminal["pass8"]),
        "validation_terminal_greedy": float(terminal["greedy_success"]),
        "validation_terminal_greedy_valid_rate": float(terminal["greedy_valid_rate"]),
        "validation_terminal_sampled_valid_rate": float(terminal["sampled_valid_rate"]),
        "supplementary_best_step": int(best["update"]),
        "supplementary_best_pass8": float(best["pass8"]),
        "supplementary_best_greedy": float(best["greedy_success"]),
    }


def _load_cell_splits(
    split_manifest: Mapping[str, Any],
    task: str,
    *,
    engineering_liveness: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    paths = split_manifest["tasks"][task]["paths"]
    train_rows = read_jsonl(Path(paths["train"]))
    validation_rows = [] if engineering_liveness else read_jsonl(Path(paths["validation"]))
    return train_rows, validation_rows


def _train_cell_impl(
    cell: Cell,
    *,
    inputs: TaskInputs,
    split_manifest: Mapping[str, Any],
    base_model_path: str,
    config: Mapping[str, Any],
    output_root: Path,
    force: bool,
    updates_override: int | None = None,
    engineering_liveness: bool = False,
) -> dict[str, Any]:
    if torch is None or DataLoader is None:
        raise RuntimeError("Training requires Torch")
    calibration_path = output_root / "calibration" / f"{cell.task}.json"
    if not calibration_path.is_file():
        raise RuntimeError(f"Run calibration before training {cell.task}")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if not calibration.get("complete"):
        raise RuntimeError(f"Incomplete calibration for {cell.task}")

    root_name = "liveness" if engineering_liveness else "cells"
    cell_root = output_root / root_name / cell.key
    manifest_path = cell_root / "cell_manifest.json"
    identity = _cell_identity(
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
    if manifest_path.is_file() and not force:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("identity_hash") == identity["identity_hash"] and existing.get("complete"):
            return existing
        raise RuntimeError(f"Existing cell identity mismatch: {cell.key}")
    if cell_root.exists():
        if not force:
            raise RuntimeError(f"Cell output exists without reusable manifest: {cell_root}")
        expected_parent = (output_root / root_name).resolve()
        if expected_parent not in cell_root.resolve().parents:
            raise RuntimeError(f"Refusing unsafe cell removal: {cell_root}")
        shutil.rmtree(cell_root)
    cell_root.mkdir(parents=True, exist_ok=False)

    _seed_everything(cell.seed)
    if inputs.reference_adapter is None:
        raise RuntimeError(f"Reference adapter is missing for {cell.task}")
    model, tokenizer, scheduler_factory = _load_reference_model(
        base_model_path,
        inputs.reference_adapter,
        config,
        train_mode=True,
    )
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
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
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
        num_warmup_steps=max(1, int(updates * float(training["warmup_ratio"]))),
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
            if cell.method == METHOD_EXPONENTIAL:
                if cell.rho is None:
                    raise AssertionError("Exponential cell has no rho")
                near_batch, far_batch, _, _ = _select_current_extremes(
                    model,
                    tokenizer,
                    rows,
                    max_length=max_length,
                )
                near_lp = completion_stats_batch(model, _move_batch(near_batch, device))["seq_lp"]
                far_lp = completion_stats_batch(model, _move_batch(far_batch, device))["seq_lp"]
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
                negative_scale = float(calibration["negative_scales"][f"{cell.rho:.12g}"])
                negative_scalar = negative_scale * (
                    0.5 * (near_weight * near_lp).mean() + 0.5 * (far_weight * far_lp).mean()
                )
                loss = loss + negative_scalar
                negative_scalar_total += float(negative_scalar.detach().cpu())
                near_weight_total += float(near_weight.mean().cpu())
                far_weight_total += float(far_weight.mean().cpu())
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"{cell.key} non-finite loss at update {update}")
            (loss / accumulation).backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            trainable,
            float(training["max_grad_norm"]),
        )
        if not bool(torch.isfinite(gradient_norm)):
            raise RuntimeError(f"{cell.key} non-finite gradient at update {update}")
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        if not all(bool(torch.isfinite(parameter).all()) for parameter in trainable):
            raise RuntimeError(f"{cell.key} non-finite trainable parameter at update {update}")
        if update % 10 == 0 or update == updates:
            append_jsonl(
                training_path,
                {
                    "update": update,
                    "positive_loss": positive_loss_total / accumulation,
                    "negative_scalar": negative_scalar_total / accumulation,
                    "mean_near_weight": near_weight_total / accumulation,
                    "mean_far_weight": far_weight_total / accumulation,
                    "raw_gradient_norm_before_clip": float(gradient_norm.detach().cpu()),
                    "learning_rate": float(scheduler.get_last_lr()[0]),
                },
            )
        should_evaluate = not engineering_liveness and (
            update % int(training["evaluation_every_updates"]) == 0 or update == updates
        )
        if should_evaluate:
            if adapter is None:
                raise AssertionError("Validation adapter is unavailable outside liveness")
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
        summary = _summarize_evaluations(evaluations, config)
        scientific_status = "pilot"
        evaluation_status = "complete"
    result = {
        **identity,
        **summary,
        "optimizer_updates": updates,
        "effective_prompt_batch": int(training["micro_batch"])
        * int(training["gradient_accumulation"]),
        "terminal_adapter": str(terminal_adapter.resolve()),
        "terminal_adapter_identity": model_identity(base_model_path, str(terminal_adapter))[
            "adapter"
        ],
        "training_metrics": str(training_path.resolve()),
        "evaluation_metrics": (
            str(evaluation_path.resolve()) if evaluation_path.is_file() else None
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


def train_cell(
    cell: Cell,
    *,
    inputs: TaskInputs,
    split_manifest: Mapping[str, Any],
    base_model_path: str,
    config: Mapping[str, Any],
    output_root: Path,
    force: bool,
    updates_override: int | None = None,
    engineering_liveness: bool = False,
) -> dict[str, Any]:
    root_name = "liveness" if engineering_liveness else "cells"
    failure_root = output_root / root_name / cell.key
    try:
        return _train_cell_impl(
            cell,
            inputs=inputs,
            split_manifest=split_manifest,
            base_model_path=base_model_path,
            config=config,
            output_root=output_root,
            force=force,
            updates_override=updates_override,
            engineering_liveness=engineering_liveness,
        )
    except Exception as exc:
        failure_root.mkdir(parents=True, exist_ok=True)
        atomic_json(
            failure_root / "failure.json",
            {
                "schema_version": 1,
                "experiment_id": experiment_id(config),
                "cell_key": cell.key,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "scientific_status": "not_run" if engineering_liveness else "pilot",
                "nan_inf_failure": (
                    "non-finite" in str(exc).lower() or "nan/inf" in str(exc).lower()
                ),
                "complete": False,
            },
        )
        raise


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
    splits, inputs = _load_ready_inputs(
        output_root,
        config,
        base_model_path=base_model_path,
    )
    cell = cells[cell_key]
    return train_cell(
        cell,
        inputs=inputs[cell.task],
        split_manifest=splits,
        base_model_path=base_model_path,
        config=config,
        output_root=output_root,
        force=force,
    )


def cmd_reload_adapter(
    config: Mapping[str, Any],
    *,
    base_model_path: str,
    adapter_path: Path,
) -> dict[str, Any]:
    if torch is None:
        raise RuntimeError("Reload verification requires Torch")
    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError("Reload verification requires transformers and peft") from exc
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": resolve_torch_dtype(str(config["model"]["dtype"])),
    }
    if torch.cuda.is_available():
        kwargs["device_map"] = {"": int(os.environ.get("LOCAL_RANK", "0"))}
    base = AutoModelForCausalLM.from_pretrained(base_model_path, **kwargs)
    reloaded = PeftModel.from_pretrained(base, str(adapter_path), is_trainable=False)
    finite = all(
        bool(torch.isfinite(parameter).all())
        for parameter in reloaded.parameters()
        if parameter.is_floating_point()
    )
    if not finite:
        raise RuntimeError("Reloaded adapter contains non-finite parameters")
    return {
        "schema_version": 1,
        "experiment_id": experiment_id(config),
        "process_id": os.getpid(),
        "base_model_identity": model_identity(base_model_path, None)["model"],
        "adapter_identity": model_identity(base_model_path, str(adapter_path))["adapter"],
        "finite": True,
        "complete": True,
    }


def cmd_liveness(
    config: Mapping[str, Any],
    config_path: Path,
    output_root: Path,
    *,
    task: str,
    rho: float,
    base_model_path: str,
    force: bool,
) -> dict[str, Any]:
    if task not in config["suite"]["tasks"]:
        raise ValueError(f"Unknown liveness task: {task}")
    if rho not in _task_rhos(config, task):
        raise ValueError("Liveness rho must be one frozen grid point")
    splits, inputs = _load_ready_inputs(
        output_root,
        config,
        base_model_path=base_model_path,
    )
    lambda_value = None
    if _is_dense(config):
        lambda_value = next(
            value
            for value in _task_lambdas(config, task)
            if math.isclose(math.exp(-value), rho, rel_tol=0.0, abs_tol=1.0e-15)
        )
    cell = Cell(
        task,
        METHOD_EXPONENTIAL,
        rho,
        int(config["sweep"]["tuning_seed"]),
        "liveness",
        lambda_value,
    )
    result = train_cell(
        cell,
        inputs=inputs[task],
        split_manifest=splits,
        base_model_path=base_model_path,
        config=config,
        output_root=output_root,
        force=force,
        updates_override=2,
        engineering_liveness=True,
    )
    reload_command = [
        sys.executable,
        "-m",
        "drpo.e8_multitask_exp_tuning",
        "--config",
        str(config_path.resolve()),
        "--output-root",
        str(output_root.resolve()),
        "reload-adapter",
        "--base-model-path",
        base_model_path,
        "--adapter-path",
        str(result["terminal_adapter"]),
    ]
    reload_process = subprocess.run(
        reload_command,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
        check=False,
    )
    if reload_process.returncode != 0:
        raise RuntimeError(f"Fresh-process adapter reload failed: {reload_process.stderr[-2000:]}")
    try:
        reload_result = json.loads(reload_process.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Fresh-process adapter reload returned invalid JSON") from exc
    if (
        not reload_result.get("complete")
        or not reload_result.get("finite")
        or int(reload_result.get("process_id", os.getpid())) == os.getpid()
        or reload_result.get("base_model_identity")
        != model_identity(base_model_path, None)["model"]
        or reload_result.get("adapter_identity") != result["terminal_adapter_identity"]
    ):
        raise RuntimeError("Fresh-process adapter reload identity or finiteness mismatch")
    training_rows = read_jsonl(Path(result["training_metrics"]))
    if not training_rows:
        raise RuntimeError("Liveness training metrics are missing")
    terminal_metrics = training_rows[-1]
    positive_loss = float(terminal_metrics["positive_loss"])
    negative_scalar = float(terminal_metrics["negative_scalar"])
    raw_gradient_norm = float(terminal_metrics["raw_gradient_norm_before_clip"])
    if not math.isfinite(positive_loss) or positive_loss <= 0.0:
        raise RuntimeError("Liveness positive loss is not finite and positive")
    if not math.isfinite(negative_scalar) or negative_scalar == 0.0:
        raise RuntimeError("Liveness repulsive scalar is not finite and nonzero")
    if not math.isfinite(raw_gradient_norm) or raw_gradient_norm <= 0.0:
        raise RuntimeError("Liveness raw gradient norm is not finite and positive")

    def adapter_weight_file(adapter_root: Path) -> Path:
        for name in ("adapter_model.safetensors", "adapter_model.bin"):
            candidate = adapter_root / name
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"Adapter weight file is missing: {adapter_root}")

    reference_adapter = inputs[task].reference_adapter
    if reference_adapter is None:
        raise RuntimeError("Liveness reference adapter is missing")
    reference_hash = sha256_file(adapter_weight_file(reference_adapter))
    terminal_hash = sha256_file(adapter_weight_file(Path(result["terminal_adapter"])))
    if reference_hash == terminal_hash:
        raise RuntimeError("Liveness adapter weights did not change after two updates")
    result.update(
        {
            "reload_gate_pending": False,
            "reload_gate_passed": True,
            "positive_loss_finite_nonzero": True,
            "repulsive_scalar_finite_nonzero": True,
            "raw_gradient_finite_nonzero": True,
            "adapter_weight_changed": True,
            "fresh_process_reload_passed": True,
            "liveness_parent_process_id": os.getpid(),
            "reload_process_id": int(reload_result["process_id"]),
            "reference_adapter_weight_sha256": reference_hash,
            "terminal_adapter_weight_sha256": terminal_hash,
        }
    )
    atomic_json(
        output_root / "liveness" / cell.key / "cell_manifest.json",
        result,
    )
    return result


def _run_subprocess_cell(
    *,
    config_path: Path,
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
        "drpo.e8_multitask_exp_tuning",
        "--config",
        str(config_path),
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
            text=True,
            env=environment,
            check=False,
        )
    return {
        "cell_key": cell.key,
        "gpu_id": gpu_id,
        "returncode": completed.returncode,
        "log": str(log_path.resolve()),
    }


def _require_calibration_gate(
    config: Mapping[str, Any],
    output_root: Path,
    *,
    base_model_path: str,
) -> None:
    path = output_root / "calibration" / "calibration_manifest.json"
    if not path.is_file():
        raise RuntimeError("Run all configured calibrations before launching a wave")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected_tasks = set(config["suite"]["tasks"])
    if (
        manifest.get("experiment_id") != experiment_id(config)
        or manifest.get("config_hash") != stable_config_hash(config)
        or not manifest.get("complete")
        or set(manifest.get("tasks", {})) != expected_tasks
    ):
        raise RuntimeError("Calibration manifest is incomplete or has the wrong identity")

    splits, inputs = _load_ready_inputs(
        output_root,
        config,
        base_model_path=base_model_path,
    )
    for task in config["suite"]["tasks"]:
        result = manifest["tasks"][task]
        expected_identity = _calibration_identity(
            task,
            inputs=inputs[task],
            split_manifest=splits,
            base_model_path=base_model_path,
            config=config,
        )
        if result.get("identity_hash") != expected_identity["identity_hash"] or not result.get(
            "complete"
        ):
            raise RuntimeError(f"Calibration gate identity mismatch for {task}")


def _require_liveness_gate(
    config: Mapping[str, Any],
    output_root: Path,
    *,
    base_model_path: str,
) -> None:
    root = output_root / "liveness"
    if not root.is_dir():
        raise RuntimeError("Run and pass the two-update liveness gate before launching a wave")
    base_identity = model_identity(base_model_path, None)["model"]
    passed: list[str] = []
    for path in sorted(root.glob("*/cell_manifest.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        if (
            result.get("experiment_id") == experiment_id(config)
            and result.get("config_hash") == stable_config_hash(config)
            and result.get("base_model_identity") == base_identity
            and result.get("engineering_liveness") is True
            and int(result.get("optimizer_updates", 0)) == 2
            and result.get("complete") is True
            and result.get("reload_gate_passed") is True
            and result.get("positive_loss_finite_nonzero") is True
            and result.get("repulsive_scalar_finite_nonzero") is True
            and result.get("raw_gradient_finite_nonzero") is True
            and result.get("adapter_weight_changed") is True
            and result.get("fresh_process_reload_passed") is True
            and result.get("reload_process_id") != result.get("liveness_parent_process_id")
            and result.get("nan_inf_failure") is False
            and result.get("reference_adapter_weight_sha256")
            != result.get("terminal_adapter_weight_sha256")
        ):
            passed.append(str(result.get("cell", {}).get("task", path.parent.name)))
    if not passed:
        raise RuntimeError("No identity-matched liveness result passes every engineering gate")


def cmd_run_wave(
    config: Mapping[str, Any],
    config_path: Path,
    output_root: Path,
    *,
    wave_index: int,
    base_model_path: str,
    force: bool,
) -> dict[str, Any]:
    _require_calibration_gate(
        config,
        output_root,
        base_model_path=base_model_path,
    )
    _require_liveness_gate(
        config,
        output_root,
        base_model_path=base_model_path,
    )
    waves = build_waves(config)
    if not 1 <= wave_index <= len(waves):
        raise ValueError(f"wave must be in [1,{len(waves)}]")
    wave = waves[wave_index - 1]
    gpu_ids = tuple(int(value) for value in config["execution"]["gpu_ids"])
    slots_per_gpu = int(config["execution"]["slots_per_gpu"])
    if len(wave) > len(gpu_ids) * slots_per_gpu:
        raise RuntimeError("Wave exceeds declared GPU slot capacity")
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(wave)) as executor:
        futures = {
            executor.submit(
                _run_subprocess_cell,
                config_path=config_path.resolve(),
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
        "experiment_id": experiment_id(config),
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


def cmd_run_all(
    config: Mapping[str, Any],
    config_path: Path,
    output_root: Path,
    *,
    base_model_path: str,
    force: bool,
) -> dict[str, Any]:
    results = []
    for wave_index in range(1, int(config["execution"]["expected_waves"]) + 1):
        results.append(
            cmd_run_wave(
                config,
                config_path,
                output_root,
                wave_index=wave_index,
                base_model_path=base_model_path,
                force=force,
            )
        )
    manifest = {
        "schema_version": 1,
        "experiment_id": experiment_id(config),
        "waves": results,
        "complete": all(result["complete"] for result in results),
        "scientific_status": "pilot",
    }
    atomic_json(output_root / "waves" / "all_waves.json", manifest)
    return manifest


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_dense(
    config: Mapping[str, Any],
    output_root: Path,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    parent_path = output_root / "inherited" / "parent_response.json"
    if not parent_path.is_file():
        raise FileNotFoundError("Dense aggregation requires inherited parent_response.json")
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if (
        parent.get("experiment_id") != experiment_id(config)
        or parent.get("config_hash") != stable_config_hash(config)
        or not parent.get("complete")
    ):
        raise RuntimeError("Inherited parent response identity mismatch")
    parent_rows = list(parent.get("rows", ()))
    if len(parent_rows) != len(config["suite"]["tasks"]) * 8:
        raise RuntimeError("Inherited parent response does not contain eight anchors per task")
    combined_rows = parent_rows + rows
    _write_csv(output_root / "aggregate" / "combined_response.csv", combined_rows)

    task_summaries: dict[str, Any] = {}
    selected_rows: list[dict[str, Any]] = []
    minimum_valid = float(config["selection"]["terminal_valid_rate_minimum"])
    for task_value in config["suite"]["tasks"]:
        task = str(task_value)
        task_dense = [row for row in rows if row["task"] == task]
        task_parent = [row for row in parent_rows if row["task"] == task]
        positive_rows = [row for row in task_parent if row["method"] == METHOD_POSITIVE_ONLY]
        parent_exp = [row for row in task_parent if row["method"] == METHOD_EXPONENTIAL]
        if len(task_dense) != 16 or len(positive_rows) != 1 or len(parent_exp) != 7:
            raise RuntimeError(f"{task} dense/predecessor response geometry is incomplete")
        eligible = [
            row
            for row in task_dense
            if not row["nan_inf_failure"]
            and float(row["terminal_greedy_valid_rate"]) >= minimum_valid
        ]
        selected = (
            max(
                eligible,
                key=lambda row: (
                    float(row["late_window_pass8_mean"]),
                    float(row["terminal_pass8"]),
                    float(row["late_window_greedy_mean"]),
                    float(row["terminal_greedy"]),
                    -float(row["lambda"]),
                ),
            )
            if eligible
            else None
        )
        positive = positive_rows[0]
        best_observed = max(
            task_dense,
            key=lambda row: float(row["late_window_pass8_mean"]),
        )
        task_lambdas = _task_lambdas(config, task)
        bridge_lambda = float(config["sweep"]["bridge_lambda"][task])
        bridge_dense = next(
            row
            for row in task_dense
            if math.isclose(
                float(row["lambda"]),
                bridge_lambda,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        )
        bridge_parent = next(
            row
            for row in parent_exp
            if math.isclose(
                float(row["rho"]),
                math.exp(-bridge_lambda),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        )
        selected_lambda = None if selected is None else float(selected["lambda"])
        selected_on_grid_edge = bool(
            selected is not None
            and (
                math.isclose(selected_lambda, min(task_lambdas))
                or math.isclose(selected_lambda, max(task_lambdas))
            )
        )
        strong_boundary_unclosed = bool(
            selected is not None and math.isclose(selected_lambda, max(task_lambdas))
        )
        summary = {
            "task": task,
            "task_role": config["sweep"]["task_role"][task],
            "positive_only": positive,
            "eligible_dense_count": len(eligible),
            "selected_dense_exp": selected,
            "selected_on_grid_edge": selected_on_grid_edge,
            "strong_taper_boundary_unclosed": strong_boundary_unclosed,
            "all_dense_below_positive_only": float(best_observed["late_window_pass8_mean"])
            < float(positive["late_window_pass8_mean"]),
            "bridge": {
                "lambda": bridge_lambda,
                "rho": math.exp(-bridge_lambda),
                "parent": bridge_parent,
                "dense_rerun": bridge_dense,
                "late_window_pass8_delta": float(bridge_dense["late_window_pass8_mean"])
                - float(bridge_parent["late_window_pass8_mean"]),
                "terminal_pass8_delta": float(bridge_dense["terminal_pass8"])
                - float(bridge_parent["terminal_pass8"]),
                "late_window_greedy_delta": float(bridge_dense["late_window_greedy_mean"])
                - float(bridge_parent["late_window_greedy_mean"]),
                "report_only": True,
            },
            "selection_metric": config["selection"]["primary_metric"],
        }
        task_summaries[task] = summary
        selected_rows.append(
            {
                "task": task,
                "task_role": summary["task_role"],
                "selected_lambda": selected_lambda,
                "selected_rho": None if selected is None else selected["rho"],
                "selected_late_window_pass8_mean": (
                    None if selected is None else selected["late_window_pass8_mean"]
                ),
                "positive_only_late_window_pass8_mean": positive["late_window_pass8_mean"],
                "all_dense_below_positive_only": summary["all_dense_below_positive_only"],
                "selected_on_grid_edge": selected_on_grid_edge,
                "strong_taper_boundary_unclosed": strong_boundary_unclosed,
                "bridge_late_window_pass8_delta": summary["bridge"]["late_window_pass8_delta"],
            }
        )
    _write_csv(output_root / "aggregate" / "selected_exp_by_task.csv", selected_rows)
    summary = {
        "schema_version": 1,
        "experiment_id": experiment_id(config),
        "cell_count": len(rows),
        "combined_response_point_count": len(combined_rows),
        "tasks": task_summaries,
        "excluded_tasks": dict(config["suite"]["excluded_tasks"]),
        "parent_run_id": config["parent"]["run_id"],
        "parent_result_commit": config["parent"]["result_commit"],
        "positive_only_source": "inherited_parent",
        "test_partition_accessed": False,
        "task_performance_reported_separately": True,
        "structure_diagnostic_reported_separately": True,
        "nan_inf_reported_separately": True,
        "single_seed_shape_discovery": True,
        "fresh_seed_confirmation_required": True,
        "fixed_horizon_is_convergence": False,
        "scientific_status": "pilot",
        "claim_boundary": (
            "Development response-shape refinement only; no significance, convergence, "
            "universal superiority, or categorical causal-identification claim."
        ),
    }
    atomic_json(output_root / "aggregate" / "aggregate_summary.json", summary)
    return summary


def cmd_aggregate(config: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    cells = build_cells(config)
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for cell in cells:
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
                "source": "dense" if _is_dense(config) else "current",
                "task": cell.task,
                "method": cell.method,
                "rho": cell.rho,
                "lambda": (
                    cell.lambda_value
                    if cell.lambda_value is not None
                    else (None if cell.rho is None else coefficient_from_rho(cell.rho))
                ),
                "seed": cell.seed,
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
    if _is_dense(config):
        return _aggregate_dense(config, output_root, rows)

    task_summaries: dict[str, Any] = {}
    selected_rows: list[dict[str, Any]] = []
    minimum_valid = float(config["selection"]["terminal_valid_rate_minimum"])
    boundary_rho = float(config["selection"]["boundary_rho"])
    for task_value in config["suite"]["tasks"]:
        task = str(task_value)
        task_rows = [row for row in rows if row["task"] == task]
        positive_rows = [row for row in task_rows if row["method"] == METHOD_POSITIVE_ONLY]
        exp_rows = [row for row in task_rows if row["method"] == METHOD_EXPONENTIAL]
        if len(positive_rows) != 1 or len(exp_rows) != 7:
            raise RuntimeError(f"{task} does not contain one Positive-only and seven Exp cells")
        eligible = [
            row
            for row in exp_rows
            if not row["nan_inf_failure"]
            and float(row["terminal_greedy_valid_rate"]) >= minimum_valid
        ]
        selected = (
            max(
                eligible,
                key=lambda row: (
                    float(row["late_window_pass8_mean"]),
                    float(row["terminal_pass8"]),
                    float(row["late_window_greedy_mean"]),
                    float(row["terminal_greedy"]),
                    float(row["rho"]),
                ),
            )
            if eligible
            else None
        )
        positive = positive_rows[0]
        best_observed = max(exp_rows, key=lambda row: float(row["late_window_pass8_mean"]))
        summary = {
            "task": task,
            "positive_only": positive,
            "eligible_exp_count": len(eligible),
            "selected_exp": selected,
            "all_exp_below_positive_only": float(best_observed["late_window_pass8_mean"])
            < float(positive["late_window_pass8_mean"]),
            "strong_taper_boundary_unclosed": bool(
                selected is not None and math.isclose(float(selected["rho"]), boundary_rho)
            ),
            "selection_metric": config["selection"]["primary_metric"],
        }
        task_summaries[task] = summary
        selected_rows.append(
            {
                "task": task,
                "selected_rho": None if selected is None else selected["rho"],
                "selected_late_window_pass8_mean": (
                    None if selected is None else selected["late_window_pass8_mean"]
                ),
                "positive_only_late_window_pass8_mean": positive["late_window_pass8_mean"],
                "all_exp_below_positive_only": summary["all_exp_below_positive_only"],
                "strong_taper_boundary_unclosed": summary["strong_taper_boundary_unclosed"],
            }
        )
    _write_csv(output_root / "aggregate" / "selected_exp_by_task.csv", selected_rows)
    summary = {
        "schema_version": 1,
        "experiment_id": experiment_id(config),
        "cell_count": len(rows),
        "tasks": task_summaries,
        "test_partition_accessed": False,
        "task_performance_reported_separately": True,
        "structure_diagnostic_reported_separately": True,
        "nan_inf_reported_separately": True,
        "fixed_horizon_is_convergence": False,
        "scientific_status": "pilot",
        "claim_boundary": (
            "Development hyperparameter response only; no significance, convergence, "
            "cross-task method ranking, or categorical causal-identification claim."
        ),
    }
    atomic_json(output_root / "aggregate" / "aggregate_summary.json", summary)
    return summary


def cmd_audit(config: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    cells = build_cells(config)
    missing: list[str] = []
    incomplete: list[str] = []
    nan_inf: list[str] = []
    for cell in cells:
        cell_root = output_root / "cells" / cell.key
        path = cell_root / "cell_manifest.json"
        if not path.is_file():
            failure_path = cell_root / "failure.json"
            if failure_path.is_file():
                failure = json.loads(failure_path.read_text(encoding="utf-8"))
                incomplete.append(cell.key)
                if failure.get("nan_inf_failure"):
                    nan_inf.append(cell.key)
            else:
                missing.append(cell.key)
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        if not value.get("complete") or value.get("evaluation_status") != "complete":
            incomplete.append(cell.key)
        if value.get("nan_inf_failure"):
            nan_inf.append(cell.key)
    inherited_complete = True
    aggregate_complete = True
    if _is_dense(config):
        snapshot_path = output_root / "inherited" / "parent_snapshot.json"
        aggregate_path = output_root / "aggregate" / "aggregate_summary.json"
        inherited_complete = snapshot_path.is_file() and bool(
            json.loads(snapshot_path.read_text(encoding="utf-8")).get("complete")
        )
        aggregate_complete = aggregate_path.is_file() and int(
            json.loads(aggregate_path.read_text(encoding="utf-8")).get("cell_count", 0)
        ) == len(cells)
    all_complete = (
        not missing and not incomplete and not nan_inf and inherited_complete and aggregate_complete
    )
    audit = {
        "schema_version": 1,
        "experiment_id": experiment_id(config),
        "expected_cells": len(cells),
        "missing_cells": sorted(set(missing)),
        "incomplete_cells": sorted(set(incomplete)),
        "nan_inf_cells": sorted(set(nan_inf)),
        "all_training_and_evaluation_complete": all_complete,
        "test_partition_accessed": False,
        "task_performance_event": "not_adjudicated_without_registered_collapse_threshold",
        "structure_event": "greedy_and_sampled_valid_rate_diagnostic_only",
        "nan_inf_event_count": len(set(nan_inf)),
        "inherited_parent_inputs_complete": inherited_complete,
        "aggregate_complete": aggregate_complete,
        "excluded_tasks": (dict(config["suite"]["excluded_tasks"]) if _is_dense(config) else {}),
        "single_seed_shape_discovery": _is_dense(config),
        "fresh_seed_confirmation_required": _is_dense(config),
        "fixed_horizon_is_convergence": False,
        "scientific_status": "pilot" if all_complete else "not_run",
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
    prepare.add_argument("--p0-config", default=str(DEFAULT_P0_CONFIG))
    prepare.add_argument("--countdown-bank", required=True)
    prepare.add_argument("--countdown-validation", required=True)
    prepare.add_argument("--countdown-adapter", required=True)

    inherit = subparsers.add_parser("inherit")
    inherit.add_argument("--parent-output-root", required=True)
    inherit.add_argument("--parent-config", default=str(DEFAULT_CONFIG))
    inherit.add_argument("--base-model-path", required=True)

    reference = subparsers.add_parser("reference")
    reference.add_argument("--base-model-path", required=True)
    reference.add_argument("--tasks", nargs="+")
    reference.add_argument("--force", action="store_true")

    reload_adapter = subparsers.add_parser("reload-adapter", help=argparse.SUPPRESS)
    reload_adapter.add_argument("--base-model-path", required=True)
    reload_adapter.add_argument("--adapter-path", required=True)

    calibrate = subparsers.add_parser("calibrate")
    calibrate.add_argument("--base-model-path", required=True)
    calibrate.add_argument("--tasks", nargs="+")
    calibrate.add_argument("--force", action="store_true")

    liveness = subparsers.add_parser("liveness")
    liveness.add_argument("--task", required=True)
    liveness_values = liveness.add_mutually_exclusive_group()
    liveness_values.add_argument("--rho", type=float)
    liveness_values.add_argument("--lambda", dest="lambda_value", type=float)
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

    run_all = subparsers.add_parser("run-all")
    run_all.add_argument("--base-model-path", required=True)
    run_all.add_argument("--force", action="store_true")

    subparsers.add_parser("aggregate")
    subparsers.add_parser("audit")
    subparsers.add_parser("plan")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = make_parser().parse_args(argv)
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    output_root = validate_work_dir(args.output_root)
    if args.command == "prepare":
        result = cmd_prepare(
            config,
            output_root,
            p0_work_dir=Path(args.p0_work_dir).resolve(),
            p0_config=Path(args.p0_config).resolve(),
            countdown_bank=Path(args.countdown_bank).resolve(),
            countdown_validation=Path(args.countdown_validation).resolve(),
            countdown_adapter=Path(args.countdown_adapter).resolve(),
        )
    elif args.command == "inherit":
        result = cmd_inherit(
            config,
            output_root,
            parent_output_root=Path(args.parent_output_root).resolve(),
            parent_config_path=Path(args.parent_config).resolve(),
            base_model_path=args.base_model_path,
        )
    elif args.command == "reference":
        result = cmd_reference(
            config,
            output_root,
            base_model_path=args.base_model_path,
            tasks=args.tasks,
            force=bool(args.force),
        )
    elif args.command == "reload-adapter":
        result = cmd_reload_adapter(
            config,
            base_model_path=args.base_model_path,
            adapter_path=Path(args.adapter_path).resolve(),
        )
    elif args.command == "calibrate":
        result = cmd_calibrate(
            config,
            output_root,
            base_model_path=args.base_model_path,
            tasks=args.tasks,
            force=bool(args.force),
        )
    elif args.command == "liveness":
        rho = args.rho
        if args.lambda_value is not None:
            rho = math.exp(-float(args.lambda_value))
        if rho is None:
            rho = _task_rhos(config, str(args.task))[0]
        result = cmd_liveness(
            config,
            config_path,
            output_root,
            task=args.task,
            rho=float(rho),
            base_model_path=args.base_model_path,
            force=bool(args.force),
        )
    elif args.command == "train-cell":
        result = cmd_train_cell(
            config,
            output_root,
            cell_key=args.cell_key,
            base_model_path=args.base_model_path,
            force=bool(args.force),
        )
    elif args.command == "run-wave":
        result = cmd_run_wave(
            config,
            config_path,
            output_root,
            wave_index=int(args.wave),
            base_model_path=args.base_model_path,
            force=bool(args.force),
        )
    elif args.command == "run-all":
        result = cmd_run_all(
            config,
            config_path,
            output_root,
            base_model_path=args.base_model_path,
            force=bool(args.force),
        )
    elif args.command == "aggregate":
        result = cmd_aggregate(config, output_root)
    elif args.command == "audit":
        result = cmd_audit(config, output_root)
    elif args.command == "plan":
        result = write_plan(config, output_root)
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
