"""Paper-aligned exponential-taper response tuning on frozen multitask banks.

The module keeps the P0 occurrence/gradient diagnostic separate from downstream
method tuning.  The cold-start profile dispatches every scientific update to
the byte-locked paper Countdown trainer.  Task adapters may change only data
schema, verifier, and explicitly whitelisted length/evaluation-batch fields.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import importlib
import json
import math
import os
import queue
import random
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import traceback
import zipfile
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import yaml

from drpo import e8_experiment_config as experiment_config

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

# Backward-compatible aliases; e8_experiment_config is the single authority for
# experiment IDs and sweep-profile names.
EXPERIMENT_ID = experiment_config.RHO_EXPERIMENT_ID
DENSE_EXPERIMENT_ID = experiment_config.DENSE_EXPERIMENT_ID
COLDSTART_EXPERIMENT_ID = experiment_config.COLDSTART_EXPERIMENT_ID
LAMBDA_COMPLETION_EXPERIMENT_ID = experiment_config.LAMBDA_COMPLETION_EXPERIMENT_ID
LAMBDA_CURVE_COMPLETION_EXPERIMENT_ID = (
    experiment_config.LAMBDA_CURVE_COMPLETION_EXPERIMENT_ID
)
P0_EXPERIMENT_ID = experiment_config.P0_EXPERIMENT_ID
# Backward-compatible name used by predecessor tests and downstream callers.
PARENT_EXPERIMENT_ID = P0_EXPERIMENT_ID
DEFAULT_CONFIG = Path("configs/e8_multitask_exp_tuning.yaml")
DEFAULT_P0_CONFIG = Path("configs/e8_multitask_p0.yaml")
METHOD_POSITIVE_ONLY = "positive_only"
METHOD_EXPONENTIAL = "exponential"
METHOD_GLOBAL = "global"
TRANSFER_SYSTEM_PROMPT = "Answer with only the requested final output and no explanation."
SWEEP_PROFILE_RHO = experiment_config.SWEEP_PROFILE_RHO
SWEEP_PROFILE_DENSE = experiment_config.SWEEP_PROFILE_DENSE
SWEEP_PROFILE_COLDSTART = experiment_config.SWEEP_PROFILE_COLDSTART

RECOVERY_SNAPSHOT_SCHEMA_VERSION = 1
RECOVERY_TRANSIENT_TOP_LEVEL = {
    "aggregate",
    "packages",
    "recovery",
    "scheduler",
    "task_results",
}
RECOVERY_TRANSIENT_FILES = {
    "ENGINEERING_SELF_TEST_REPORT.json",
    "RUN_COMPLETE.json",
    "SHA256SUMS.txt",
    "package_contents_manifest.json",
    "run_manifest.json",
    "scientific_run_manifest.json",
    "terminal_audit.json",
}

CANONICAL_COLD_MODULES = {
    "arena": "drpo.countdown_qwen_arena_onefile",
    # Import paper_runtime before the base runtime/trainer so its activation
    # patches the base symbols before the trainer binds them.
    "paper_common": "drpo.countdown_e8_alpha1_highc_scan_common",
    "paper_runtime": "drpo.countdown_e8_alpha1_highc_scan_runtime",
    "scan_common": "drpo.countdown_e8_alpha1_c_scan_common",
    "scan_runtime": "drpo.countdown_e8_alpha1_c_scan_runtime",
    "scan_trainer": "drpo.countdown_e8_alpha1_c_scan_trainer",
}

PAPER_ROUND1_COEFFICIENTS = (
    0.051293294,
    0.105360516,
    0.162518929,
    0.223143551,
    0.287682072,
    0.430782916,
    0.693147181,
    0.916290732,
    1.203972804,
    1.386294361,
    1.609437912,
    1.897119985,
    2.302585093,
    2.995732274,
)
PAPER_EXTENSION_COEFFICIENTS = (
    0.01,
    0.025,
    0.04,
    3.506557897,
    4.605170186,
    5.298317367,
    6.907755279,
    9.210340372,
)
TASK_TRANSFER_COEFFICIENTS = PAPER_ROUND1_COEFFICIENTS + PAPER_EXTENSION_COEFFICIENTS[3:]
PAPER_SEED_OFFSETS = (4000, 5000)


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
        if self.method == METHOD_GLOBAL:
            return f"{self.task}__global__seed{self.seed}"
        if self.lambda_value is not None:
            tag = f"{self.lambda_value:.12g}".replace(".", "p")
            return f"{self.task}__exp_lambda{tag}__seed{self.seed}"
        if self.rho is None:
            raise AssertionError("Exponential cell requires rho or lambda")
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
    source = Path(path)
    repo_root = Path(__file__).resolve().parents[2]
    if not source.is_absolute():
        source = repo_root / source
    value = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("Configuration root must be a mapping")
    validate_config(value)
    experiment_config.validate_historical_config_identity(source, value, repo_root=repo_root)
    return value


def _tuple_floats(values: Sequence[Any]) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def experiment_id(config: Mapping[str, Any]) -> str:
    return experiment_config.experiment_id(config)


def sweep_profile(config: Mapping[str, Any]) -> str:
    return experiment_config.sweep_profile(config)


def _is_dense(config: Mapping[str, Any]) -> bool:
    return sweep_profile(config) == SWEEP_PROFILE_DENSE


def _is_coldstart(config: Mapping[str, Any]) -> bool:
    return sweep_profile(config) == SWEEP_PROFILE_COLDSTART


def _is_engineering_self_test(config: Mapping[str, Any]) -> bool:
    value = config.get("engineering_self_test")
    return isinstance(value, Mapping) and value.get("placeholder_backend") is True


def _uses_task_lambdas(config: Mapping[str, Any]) -> bool:
    return _is_dense(config) or _is_coldstart(config)


def _dense_tasks() -> set[str]:
    return set(TASK_NAMES) - {"countdown", "spiral_matrix"}


def _task_lambdas(config: Mapping[str, Any], task: str) -> tuple[float, ...]:
    if not _uses_task_lambdas(config):
        raise ValueError("Task-local lambdas are not defined for this profile")
    return experiment_config.task_lambdas(config, task)


def _task_rhos(config: Mapping[str, Any], task: str) -> tuple[float, ...]:
    if _uses_task_lambdas(config):
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
    profile = sweep_profile(config)
    experiment_config.validate_profile_experiment_id(config)
    if profile == SWEEP_PROFILE_COLDSTART:
        return

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
        if len(tasks) != 7 or len(set(tasks)) != 7 or set(tasks) != _dense_tasks():
            raise ValueError(
                "The dense suite must be the exact seven non-Countdown, non-Spiral tasks"
            )
        if tuple(config["suite"].get("p0_tasks", ())) != tasks:
            raise ValueError("Dense suite.p0_tasks must preserve the exact task order")
        if tuple(config["suite"].get("external_tasks", ())) != ():
            raise ValueError("Dense refinement has no external Countdown task")

    reference = config["reference"]
    if reference["checkpoint_kind"] != "train_only_task_positive_warmstart_100":
        raise ValueError("reference.checkpoint_kind must be train_only_task_positive_warmstart_100")
    if int(reference["optimizer_updates"]) != 100:
        raise ValueError("Reference initialization must use 100 updates")
    if int(reference["validation_rows_seen"]) != 0 or int(reference["test_rows_seen"]) != 0:
        raise ValueError("Train-only reference preparation must not see validation or test rows")

    split = config["split"]
    if _is_engineering_self_test(config):
        expected_split = {
            "p0_train_rows": 2,
            "p0_validation_rows": 1,
            "p0_test_rows": 1,
            "countdown_train_rows": 2,
            "countdown_validation_rows": 1,
        }
    else:
        expected_split = {
            "p0_train_rows": 5000,
            "p0_validation_rows": 500,
            "p0_test_rows": 500,
        }
        if profile == SWEEP_PROFILE_RHO:
            expected_split.update({"countdown_train_rows": 5000, "countdown_validation_rows": 500})
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
    if not math.isclose(float(training["learning_rate"]), 5.0e-5, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("The method-training learning rate must remain 5e-5")
    if not math.isclose(float(training["warmup_ratio"]), 0.03, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("The method-training warmup ratio must remain 0.03")
    if int(training["evaluation_every_updates"]) != 100:
        raise ValueError("Evaluation cadence must remain 100 updates")
    if not math.isclose(float(training["weight_decay"]), 0.01) or not math.isclose(
        float(training["max_grad_norm"]), 1.0
    ):
        raise ValueError("The old optimizer weight-decay/gradient-clip contract changed")
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
    expected_capacity = 16
    if int(execution["max_concurrent_cells"]) != expected_capacity:
        raise ValueError(f"The scheduler must expose exactly {expected_capacity} slots")
    if tuple(int(value) for value in execution["gpu_ids"]) != tuple(range(8)):
        raise ValueError("The default GPU pool must remain 0--7")
    expected_waves = 7 if profile == SWEEP_PROFILE_DENSE else 5
    if int(execution["slots_per_gpu"]) != 2 or int(execution["expected_waves"]) != expected_waves:
        raise ValueError(f"The frozen topology is two slots per GPU and {expected_waves} waves")


def coefficient_from_rho(rho: float) -> float:
    if not math.isfinite(rho) or not 0.0 < rho < 1.0:
        raise ValueError("rho must be finite and strictly between zero and one")
    return -math.log(rho)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _canonical_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    root = _repo_root()
    return {
        name: (root / str(relative)).resolve()
        for name, relative in config["canonical_coldstart"]["paths"].items()
    }


def _git_blob_sha(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "hash-object", str(path)],
            cwd=_repo_root(),
            text=True,
            stderr=subprocess.PIPE,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"Cannot compute Git blob identity for canonical source: {path}"
        ) from exc


def audit_canonical_coldstart_sources(config: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed unless every imported old cold-start source is byte-identical."""

    if not _is_coldstart(config):
        raise RuntimeError("Canonical cold-start source audit is cold-profile only")
    paths = _canonical_paths(config)
    expected = config["canonical_coldstart"]["expected_git_blob_shas"]
    observed: dict[str, str] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing canonical cold-start source: {path}")
        observed[name] = _git_blob_sha(path)
        if observed[name] != str(expected[name]):
            raise RuntimeError(
                f"Canonical cold-start source drift for {name}: "
                f"expected {expected[name]}, found {observed[name]}"
            )
    return {
        "paths": {name: str(path) for name, path in paths.items()},
        "git_blob_shas": observed,
        "verified": True,
    }


def _canonical_cold_modules(config: Mapping[str, Any]) -> dict[str, Any]:
    audit_canonical_coldstart_sources(config)
    modules = {
        name: importlib.import_module(module_name)
        for name, module_name in CANONICAL_COLD_MODULES.items()
    }
    scan_common = modules["scan_common"]
    scan_runtime = modules["scan_runtime"]
    scan_trainer = modules["scan_trainer"]
    paper_common = modules["paper_common"]
    paper_runtime = modules["paper_runtime"]
    arena = modules["arena"]
    if (
        scan_common.arena is not arena
        or scan_trainer.arena is not arena
        or scan_trainer.continuous_exp_weights is not scan_common.continuous_exp_weights
        or paper_common._base is not scan_common
        or paper_runtime.highc is not paper_common
        or paper_runtime._base_runtime is not scan_runtime
    ):
        raise RuntimeError("Paper cold-start modules do not share one locked implementation graph")
    return modules


def _activate_paper_grid_modules(modules: dict[str, Any], grid_path: Path) -> dict[str, Any]:
    """Bind base trainer imports to the selected paper profile in this cell process."""

    paper_common = modules["paper_common"]
    paper_common.activate_for_grid_config(grid_path)
    modules["scan_trainer"] = importlib.reload(modules["scan_trainer"])
    modules["scan_runtime"] = importlib.reload(modules["scan_runtime"])
    # Reloading the paper adapter after the base modules recreates its wrappers
    # around the just-reloaded, profile-correct trainer.
    modules["paper_runtime"] = importlib.reload(modules["paper_runtime"])
    if (
        modules["scan_trainer"].continuous_exp_weights
        is not modules["scan_common"].continuous_exp_weights
        or modules["paper_runtime"]._base_runtime is not modules["scan_runtime"]
    ):
        raise RuntimeError("Paper grid activation did not bind the selected trainer profile")
    return modules


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
    if _is_coldstart(config):
        cells: list[Cell] = []
        lambda_only = config["sweep"]["parameterization"] == "paper_lambda_c1"
        countdown_coefficients = _task_lambdas(config, "countdown")
        countdown_include_positive_only = bool(
            config["sweep"].get("countdown_include_positive_only", True)
        )
        include_global_endpoint = bool(config["sweep"].get("include_global_endpoint", False))
        for seed_offset in tuple(int(value) for value in config["sweep"]["countdown_seed_offsets"]):
            if countdown_include_positive_only:
                cells.append(
                    Cell(
                        "countdown",
                        METHOD_POSITIVE_ONLY,
                        None,
                        seed_offset,
                        "countdown_sentinel",
                    )
                )
            cells.append(
                Cell("countdown", METHOD_GLOBAL, 1.0, seed_offset, "countdown_sentinel", 0.0)
            )
            cells.extend(
                Cell(
                    "countdown",
                    METHOD_EXPONENTIAL,
                    None if lambda_only else math.exp(-coefficient),
                    seed_offset,
                    "countdown_sentinel",
                    coefficient,
                )
                for coefficient in countdown_coefficients
            )
        positive_seeds = tuple(
            int(value) for value in config["sweep"]["transfer_positive_only_seed_offsets"]
        )
        exp_seed = int(config["sweep"]["task_transfer_seed_offset"])
        for task in tasks:
            if task == "countdown":
                continue
            coefficients = _task_lambdas(config, task)
            if not coefficients:
                continue
            cells.extend(
                Cell(task, METHOD_POSITIVE_ONLY, None, seed_offset, "task_transfer")
                for seed_offset in positive_seeds
            )
            if include_global_endpoint:
                cells.append(Cell(task, METHOD_GLOBAL, 1.0, exp_seed, "task_transfer", 0.0))
            cells.extend(
                Cell(
                    task,
                    METHOD_EXPONENTIAL,
                    None if lambda_only else math.exp(-coefficient),
                    exp_seed,
                    "task_transfer",
                    coefficient,
                )
                for coefficient in coefficients
            )
        result = tuple(cells)
        if len(result) != int(config["sweep"]["expected_cells"]) or len(
            {cell.key for cell in result}
        ) != len(result):
            raise AssertionError("Internal cold-start cell identity failure")
        return result
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
    if _is_coldstart(config):
        waves = tuple(
            tuple(cells[index : index + capacity]) for index in range(0, len(cells), capacity)
        )
        if (
            not waves
            or any(len(wave) != capacity for wave in waves[:-1])
            or not 0 < len(waves[-1]) <= capacity
        ):
            raise AssertionError(
                "Cold-start nominal batches must fill capacity except possibly the final batch"
            )
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
                    "nominal_batch": wave_index,
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
        "scheduler": str(config["execution"].get("scheduler", "wave_barrier")),
        "wave_is_scheduling_barrier": bool(config["execution"].get("wave_barriers", False)),
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
    source_negatives = row["negative_bank"] if "negative_bank" in row else row["negatives"]
    for index, item_value in enumerate(source_negatives):
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
    if _is_coldstart(config):
        if not bool(split.get("countdown_subsampling_forbidden", False)):
            raise RuntimeError("Paper Countdown forbids wrapper-level train subsampling")
        train = normalized_train
        validation = normalized_validation
    else:
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
        raise RuntimeError("Countdown validation file does not contain the exact frozen rows")
    partitions = {"train": train, "validation": validation}
    _audit_partition_prompt_ids("countdown", partitions)
    return partitions


def _canonical_train_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Translate task schema while preserving the paper all-unique-negative loss."""

    task = str(row["task"])
    prompt_id = str(row["prompt_id"])
    negatives = list(row["negatives"])
    if len(negatives) != 16:
        raise RuntimeError(f"{task}/{prompt_id} canonical conversion requires 16 negatives")
    bank = [
        {
            **dict(item),
            "expression": str(item["completion"]),
        }
        for item in negatives
    ]
    oracle = str(row["oracle_completion"])
    if any(item["expression"] == oracle for item in bank):
        raise RuntimeError(f"{task}/{prompt_id} negative completion matches the positive")
    return {
        **dict(row),
        "id": prompt_id,
        "oracle": oracle,
        "positive": oracle,
        "negative_bank": bank,
        "negative_bank_size": 16,
        "pair_matched": True,
        # The old core uses this only for balanced diagnostics.  Task correctness
        # is supplied by the environment verifier, not Countdown expression parsing.
        "oracle_structure": f"{task}:task_verifier",
        "canonical_training_core": "countdown_e8_alpha1_c_scan.ContinuousUniqueBankDataset",
        "canonical_negative_consumer": "all_unique_negatives_per_prompt",
    }


def _canonical_validation_row(row: Mapping[str, Any]) -> dict[str, Any]:
    task = str(row["task"])
    prompt_id = str(row["prompt_id"])
    oracle = str(row["oracle_completion"])
    return {
        **dict(row),
        "id": prompt_id,
        "oracle": oracle,
        "oracle_structure": f"{task}:task_verifier",
    }


def _paper_grid_name(coefficient: float) -> str:
    if coefficient == 0.0 or coefficient in PAPER_ROUND1_COEFFICIENTS:
        return "round1_grid"
    if coefficient in PAPER_EXTENSION_COEFFICIENTS:
        return "extension_grid"
    raise ValueError(f"Coefficient {coefficient} is outside the locked paper grids")


def _leaf_values(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {prefix: value}
    result: dict[str, Any] = {}
    for key, item in value.items():
        child = f"{prefix}.{key}" if prefix else str(key)
        result.update(_leaf_values(item, child))
    return result


def _changed_leaf_paths(original: Mapping[str, Any], derived: Mapping[str, Any]) -> list[str]:
    left = _leaf_values(original)
    right = _leaf_values(derived)
    return sorted(key for key in set(left) | set(right) if left.get(key) != right.get(key))


def _atomic_yaml(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(dict(value), sort_keys=False), encoding="utf-8")
    temporary.replace(path)


def _task_base_config(
    config: Mapping[str, Any],
    *,
    task: str,
    canonical_paths: Mapping[str, Path],
    task_root: Path,
) -> tuple[Path, list[str]]:
    """Materialize effective base runtime without editing the canonical source."""

    base_path = canonical_paths["base_config"]
    original = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    if not isinstance(original, dict):
        raise TypeError("Paper base config root must be a mapping")
    historical = experiment_config.is_historical_coldstart_config(config)
    if historical and task == "countdown":
        return base_path, []

    derived = copy.deepcopy(original)
    effective = experiment_config.effective_coldstart_runtime(config, task)
    runtime = config["task_runtime"][task]
    if historical:
        # Preserve the exact wrapper behavior of the three closed historical IDs.
        derived["model"]["max_length"] = int(runtime["max_length"])
        derived["model"]["max_new_tokens"] = int(runtime["max_new_tokens"])
        derived["evaluation"]["batch_size"] = int(runtime["evaluation_batch_size"])
        derived["evaluation"]["pass_ks"] = [8] + [
            int(value) for value in runtime["auxiliary_pass_ks"]
        ]
    else:
        model = effective["model"]
        training = effective["training"]
        evaluation = effective["evaluation"]
        derived["model"].update(
            {
                "max_length": int(model["max_length"]),
                "max_new_tokens": int(model["max_new_tokens"]),
                "dtype": str(model["dtype"]),
                "lora_rank": int(model["lora_rank"]),
                "lora_alpha": int(model["lora_alpha"]),
                "lora_dropout": float(model["lora_dropout"]),
                "gradient_checkpointing": bool(model["gradient_checkpointing"]),
            }
        )
        derived["offline_training"].update(
            {
                "seed": int(effective["initialization_seed"]),
                "steps": int(training["optimizer_updates"]),
                "micro_batch": int(training["micro_batch"]),
                "gradient_accumulation": int(training["gradient_accumulation"]),
                "learning_rate": float(training["learning_rate"]),
                "weight_decay": float(training["weight_decay"]),
                "warmup_ratio": float(training["warmup_ratio"]),
                "maximum_gradient_norm": float(training["max_grad_norm"]),
                "eval_every": int(training["evaluation_every_updates"]),
            }
        )
        derived["evaluation"].update(
            {
                "examples": int(evaluation["examples"]),
                "batch_size": int(evaluation["batch_size"]),
                "pass_ks": [int(value) for value in evaluation["pass_ks"]],
                "seed": int(evaluation["generation_seed"]),
                "sampling_temperature": float(evaluation["sampling_temperature"]),
                "top_p": float(evaluation["top_p"]),
                "greedy_prompt_rows": int(evaluation["greedy_prompt_rows"]),
                "passk_prompt_rows": int(evaluation["passk_prompt_rows"]),
            }
        )
    path = task_root / "paper_base_task_interface.yaml"
    _atomic_yaml(path, derived)
    return path, _changed_leaf_paths(original, derived)


def _task_grid_configs(
    config: Mapping[str, Any],
    *,
    canonical_paths: Mapping[str, Path],
    task_root: Path,
) -> dict[str, dict[str, Any]]:
    """Return historical grids unchanged or generic derived runtime-grid copies."""

    result: dict[str, dict[str, Any]] = {}
    historical = experiment_config.is_historical_coldstart_config(config)
    training = config["training"]
    for name in ("round1_grid", "extension_grid"):
        source = canonical_paths[name]
        original = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(original, dict):
            raise TypeError(f"Paper grid root must be a mapping: {source}")
        if historical:
            runtime_path = source
            changed: list[str] = []
        else:
            derived = copy.deepcopy(original)
            derived["training"]["steps"] = int(training["optimizer_updates"])
            derived["training"]["eval_every"] = int(training["evaluation_every_updates"])
            runtime_path = task_root / f"paper_{name}_runtime.yaml"
            _atomic_yaml(runtime_path, derived)
            changed = _changed_leaf_paths(original, derived)
        result[name] = {
            "path": runtime_path,
            "source": source,
            "changed_fields": changed,
        }
    return result


def _evenly_spaced_rank_indices(candidate_count: int, selected_count: int = 16) -> tuple[int, ...]:
    if selected_count < 2:
        raise ValueError("Reference-remoteness selection requires at least two selected ranks")
    if candidate_count < selected_count:
        raise ValueError(
            f"Reference-remoteness selection requires >= {selected_count} candidates; "
            f"found {candidate_count}"
        )
    indices = tuple(
        (index * (candidate_count - 1)) // (selected_count - 1) for index in range(selected_count)
    )
    if len(set(indices)) != selected_count or indices[0] != 0 or indices[-1] != candidate_count - 1:
        raise AssertionError("Even rank selection must be unique and include both extremes")
    return indices


def _coverage_first_reference_rank_indices(
    scored: Sequence[Mapping[str, Any]],
    source_negatives: Sequence[Mapping[str, Any]],
    selected_count: int = 16,
) -> tuple[int, ...]:
    if len(scored) < selected_count:
        raise RuntimeError(f"Coverage-first selection needs >= {selected_count} candidates")
    buckets: dict[str, list[int]] = {}
    for rank, item in enumerate(scored):
        buckets.setdefault(str(item["error_class"]), []).append(rank)
    class_order = [str(item["error_class"]) for item in source_negatives]
    queues = {}
    for name in sorted(set(class_order)):
        quota, ranks = class_order.count(name), buckets[name]
        if quota == 1:
            original = next(item for item in source_negatives if str(item["error_class"]) == name)
            canonical = str(original.get("canonical_completion", original["completion"]))
            local = tuple(
                index
                for index, rank in enumerate(ranks)
                if str(scored[rank]["canonical_completion"]) == canonical
            )
            if len(local) != 1:
                raise RuntimeError(f"Source P0 singleton not uniquely reconstructed for {name}")
        else:
            local = _evenly_spaced_rank_indices(len(ranks), quota)
        queues[name] = iter([ranks[index] for index in local])
    selected = tuple(next(queues[name]) for name in class_order)
    if len(set(selected)) != selected_count:
        raise RuntimeError("Coverage-first selector produced duplicate negatives")
    return selected


def _reference_surprisal_summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray([float(value) for value in values], dtype=float)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise RuntimeError("Reference-surprisal audit requires finite non-empty values")
    q25, median, q75 = np.quantile(array, [0.25, 0.5, 0.75])
    return {
        "min": float(array.min()),
        "q25": float(q25),
        "median": float(median),
        "q75": float(q75),
        "max": float(array.max()),
        "range": float(array.max() - array.min()),
        "iqr": float(q75 - q25),
    }


def _reference_error_class_audit(
    scored: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
    source_negatives: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    source_classes = [str(item["error_class"]) for item in source_negatives]
    selected_classes = [str(item["error_class"]) for item in selected]
    if selected_classes != source_classes:
        raise RuntimeError("Coverage-first selector changed the July-29 P0 error-class sequence")
    candidate_buckets: dict[str, list[tuple[int, Mapping[str, Any]]]] = {}
    selected_buckets: dict[str, list[Mapping[str, Any]]] = {}
    for rank, item in enumerate(scored):
        candidate_buckets.setdefault(str(item["error_class"]), []).append((rank, item))
    for item in selected:
        selected_buckets.setdefault(str(item["error_class"]), []).append(item)
    class_audit: dict[str, Any] = {}
    endpoint_total = 0
    endpoint_covered = 0
    for error_class, bucket in sorted(candidate_buckets.items()):
        chosen = selected_buckets.get(error_class, [])
        candidate_ranks = [rank for rank, _ in bucket]
        selected_ranks = [int(item["reference_rank"]) for item in chosen]
        endpoint_ok: bool | None = None
        if len(chosen) >= 2:
            endpoint_total += 1
            endpoint_ok = (
                candidate_ranks[0] in selected_ranks and candidate_ranks[-1] in selected_ranks
            )
            if not endpoint_ok:
                raise RuntimeError(
                    f"Coverage-first selector missed a class-local endpoint: {error_class}"
                )
            endpoint_covered += 1
        class_audit[error_class] = {
            "candidate_count": len(bucket),
            "source_p0_count": source_classes.count(error_class),
            "selected_count": len(chosen),
            "candidate_global_rank_min": candidate_ranks[0],
            "candidate_global_rank_max": candidate_ranks[-1],
            "selected_global_ranks": selected_ranks,
            "candidate_reference_surprisal": _reference_surprisal_summary(
                [float(item["reference_surprisal"]) for _, item in bucket]
            ),
            "selected_reference_surprisal": (
                _reference_surprisal_summary(
                    [float(item["reference_surprisal"]) for item in chosen]
                )
                if chosen
                else None
            ),
            "near_far_endpoint_coverage": endpoint_ok,
        }
    selected_class_count = len(selected_buckets)
    candidate_class_count = len(candidate_buckets)
    selected_ranks = [int(item["reference_rank"]) for item in selected]
    return {
        "source_p0_error_class_sequence": source_classes,
        "selected_error_class_sequence": selected_classes,
        "coverage_sequence_matches_source_p0": True,
        "candidate_error_class_counts": {
            name: len(bucket) for name, bucket in sorted(candidate_buckets.items())
        },
        "source_p0_error_class_counts": {
            name: source_classes.count(name) for name in sorted(set(source_classes))
        },
        "selected_error_class_counts": {
            name: len(bucket) for name, bucket in sorted(selected_buckets.items())
        },
        "candidate_distinct_error_class_count": candidate_class_count,
        "selected_distinct_error_class_count": selected_class_count,
        "error_class_coverage_fraction": selected_class_count / candidate_class_count,
        "singleton_selected_error_class_count": sum(
            len(bucket) == 1 for bucket in selected_buckets.values()
        ),
        "multi_slot_selected_error_class_count": endpoint_total,
        "multi_slot_endpoint_coverage_count": endpoint_covered,
        "global_reference_rank_span_fraction": (
            (max(selected_ranks) - min(selected_ranks)) / (len(scored) - 1)
            if len(scored) > 1
            else 0.0
        ),
        "error_class_reference_surprisal": class_audit,
    }


def _verified_wrong_candidates(
    adapter: Any,
    instance: TaskInstance,
    source_row: Mapping[str, Any],
) -> list[dict[str, Any]]:
    generation_seed = int(source_row["generation_seed"])
    rng = random.Random(
        int(
            stable_hash(
                {
                    "task": str(source_row["task"]),
                    "prompt_id": str(source_row["prompt_id"]),
                    "seed": generation_seed,
                }
            )[:16],
            16,
        )
    )
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for mutation in adapter.mutation_candidates(instance, rng):
        result = adapter.verify(
            instance,
            mutation.completion,
            mutation_class=mutation.mutation_class,
        )
        canonical = str(result.canonical_completion)
        if canonical in seen or not adapter.accept_negative(result):
            continue
        seen.add(canonical)
        candidates.append(
            {
                "completion": str(mutation.completion),
                "canonical_completion": canonical,
                "verifier_score": float(result.score),
                "binary_correct": bool(result.correct),
                "format_valid": bool(result.format_valid),
                "error_class": str(result.error_class),
                "verification_details": dict(result.details),
            }
        )
    original = {
        str(item.get("canonical_completion", item["completion"]))
        for item in source_row["negatives"]
    }
    missing = sorted(original - seen)
    if missing:
        raise RuntimeError(
            f"{source_row['task']}/{source_row['prompt_id']} cannot reconstruct the original "
            f"P0 negative universe: {missing[:3]}"
        )
    if len(candidates) < 16:
        raise RuntimeError(
            f"{source_row['task']}/{source_row['prompt_id']} has only {len(candidates)} "
            "deterministic verified wrong candidates"
        )
    return candidates


def _score_reference_candidates(
    *,
    arena: Any,
    model: Any,
    tokenizer: Any,
    prompt: str,
    candidates: Sequence[Mapping[str, Any]],
    max_length: int,
    batch_size: int,
) -> list[float]:
    device = next(model.parameters()).device
    scores: list[float] = []
    for start in range(0, len(candidates), batch_size):
        chunk = candidates[start : start + batch_size]
        encoded = [
            arena.encode_prompt_completion(
                tokenizer,
                prompt,
                str(item["completion"]),
                max_length,
            )
            for item in chunk
        ]
        packed = arena.pad_encoded(encoded, int(tokenizer.pad_token_id))
        packed = arena.move_to_device(packed, device)
        with torch.no_grad():
            surprisal = arena.sequence_surprisal_only(model, packed)
        scores.extend(float(value) for value in surprisal.detach().cpu())
    if len(scores) != len(candidates) or not all(math.isfinite(value) for value in scores):
        raise RuntimeError("Reference-policy candidate scoring is incomplete or non-finite")
    return scores


def _derive_reference_remoteness_banks(
    config: Mapping[str, Any],
    output_root: Path,
    *,
    base_model_path: str,
) -> dict[str, Any]:
    """Create the fixed training bank without modifying the model-independent P0 bank."""

    if not _is_coldstart(config) or _is_engineering_self_test(config):
        raise RuntimeError("Reference-remoteness bank derivation is formal cold-start only")
    if torch is None:
        raise RuntimeError("Reference-remoteness bank derivation requires Torch")
    splits, inputs = _load_prepared(output_root, config)
    selector = dict(config["negative_sampling"]["reference_remoteness_bank"])
    base_identity = model_identity(base_model_path, None)["model"]
    pending: list[str] = []
    identities: dict[str, str] = {}
    for task_value in config["suite"]["p0_tasks"]:
        task = str(task_value)
        record = splits["tasks"][task]
        identity = stable_hash(
            {
                "schema_version": 1,
                "experiment_id": experiment_id(config),
                "config_hash": stable_config_hash(config),
                "task": task,
                "source_train_sha256": sha256_file(Path(record["paths"]["train"])),
                "source_bank_sha256": record["bank_sha256"],
                "p0_config_sha256": record["p0_config_sha256"],
                "base_model_identity": base_identity,
                "task_runtime": dict(config["task_runtime"][task]),
                "model_facing_text": "raw_completion_generic_prompt_v1",
                "selector": selector,
                "selector_implementation": "source_p0_error_class_sequence_then_within_class_reference_rank_spread_v1",
            }
        )
        identities[task] = identity
        existing = record.get("reference_remoteness_bank")
        if isinstance(existing, Mapping):
            path = Path(str(existing.get("path", "")))
            if (
                existing.get("identity_hash") == identity
                and path.is_file()
                and sha256_file(path) == existing.get("sha256")
                and int(existing.get("rows", -1)) == int(config["split"]["p0_train_rows"])
            ):
                continue
        pending.append(task)

    if pending:
        modules = _canonical_cold_modules(config)
        arena = modules["arena"]
        _seed_everything(int(config["initialization"]["seed"]))
        tokenizer = arena.load_tokenizer(str(Path(base_model_path).resolve()))
        base_config = yaml.safe_load(
            _canonical_paths(config)["base_config"].read_text(encoding="utf-8")
        )
        if not isinstance(base_config, Mapping):
            raise TypeError("Canonical base config is not a mapping")
        reference_effective = experiment_config.effective_coldstart_runtime(config, pending[0])
        with _legacy_arena_runtime_bridge(arena, reference_effective):
            model = arena.load_model(
                str(Path(base_model_path).resolve()),
                adapter_path=None,
                trainable_adapter=True,
                load_in_4bit=bool(base_config["model"].get("load_in_4bit", False)),
                dtype=str(base_config["model"].get("dtype", "auto")),
                gradient_checkpointing=False,
                parameterization="lora",
            )
        model.eval()
        original_clean_expression = arena.clean_expression
        original_system_prompt = arena.SYSTEM_PROMPT
        try:
            arena.clean_expression = lambda value: str(value)
            arena.SYSTEM_PROMPT = TRANSFER_SYSTEM_PROMPT
            for task in pending:
                record = splits["tasks"][task]
                train_rows = read_jsonl(Path(record["paths"]["train"]))
                adapter, instances = _load_task_adapter_and_instances(
                    task,
                    inputs=inputs[task],
                    validation_rows=train_rows,
                )
                derived_rows: list[dict[str, Any]] = []
                audit_rows: list[dict[str, Any]] = []
                runtime = config["task_runtime"][task]
                for source_row in train_rows:
                    prompt_id = str(source_row["prompt_id"])
                    candidates = _verified_wrong_candidates(
                        adapter,
                        instances[prompt_id],
                        source_row,
                    )
                    scores = _score_reference_candidates(
                        arena=arena,
                        model=model,
                        tokenizer=tokenizer,
                        prompt=str(source_row["prompt"]),
                        candidates=candidates,
                        max_length=int(runtime["max_length"]),
                        batch_size=int(runtime["evaluation_batch_size"]),
                    )
                    scored = []
                    for candidate, score in zip(candidates, scores, strict=True):
                        scored.append({**candidate, "reference_surprisal": float(score)})
                    scored.sort(
                        key=lambda item: (
                            float(item["reference_surprisal"]),
                            stable_hash(
                                {
                                    "task": task,
                                    "prompt_id": prompt_id,
                                    "canonical_completion": item["canonical_completion"],
                                }
                            ),
                        )
                    )
                    selected_indices = _coverage_first_reference_rank_indices(
                        scored, source_row["negatives"], 16
                    )
                    selected: list[dict[str, Any]] = []
                    for slot, rank in enumerate(selected_indices):
                        item = dict(scored[rank])
                        item.update(
                            {
                                "negative_id": f"{prompt_id}_refrem_{slot:03d}",
                                "reference_rank": int(rank),
                                "reference_candidate_count": len(scored),
                                "reference_rank_role": "provenance_and_diagnostic_only",
                            }
                        )
                        selected.append(item)
                    coverage_audit = _reference_error_class_audit(
                        scored, selected, list(source_row["negatives"])
                    )
                    derived = dict(source_row)
                    derived["negatives"] = selected
                    derived["reference_remoteness_selection"] = {
                        "identity_hash": identities[task],
                        "reference_policy": "zero_update_base_plus_fresh_lora",
                        "coordinate": "mean_completion_token_surprisal",
                        "candidate_count": len(scored),
                        "selected_ranks": list(selected_indices),
                        "training_weight_uses_reference_rank": False,
                        "current_policy_surprisal_recomputed_each_update": True,
                    }
                    derived_rows.append(derived)
                    audit_rows.append(
                        {
                            "task": task,
                            "prompt_id": prompt_id,
                            "candidate_count": len(scored),
                            "selected_ranks": list(selected_indices),
                            "candidate_reference_surprisal": _reference_surprisal_summary(
                                [float(item["reference_surprisal"]) for item in scored]
                            ),
                            "selected_reference_surprisal": _reference_surprisal_summary(
                                [float(item["reference_surprisal"]) for item in selected]
                            ),
                            **coverage_audit,
                            "coverage_threshold": None,
                            "coverage_gate": False,
                        }
                    )
                root = output_root / "reference_remoteness" / task
                bank_path_value = root / "train.jsonl"
                audit_path = root / "prompt_audit.jsonl"
                atomic_jsonl(bank_path_value, derived_rows)
                atomic_jsonl(audit_path, audit_rows)
                ranges = [float(row["selected_reference_surprisal"]["range"]) for row in audit_rows]
                class_rows = [
                    value
                    for row in audit_rows
                    for value in row["error_class_reference_surprisal"].values()
                ]
                selected_class_rows = [
                    value for value in class_rows if int(value["selected_count"]) > 0
                ]
                endpoint_total = sum(
                    int(row["multi_slot_selected_error_class_count"]) for row in audit_rows
                )
                endpoint_covered = sum(
                    int(row["multi_slot_endpoint_coverage_count"]) for row in audit_rows
                )
                summary = {
                    "schema_version": 1,
                    "experiment_id": experiment_id(config),
                    "config_hash": stable_config_hash(config),
                    "task": task,
                    "identity_hash": identities[task],
                    "source_train": record["paths"]["train"],
                    "source_train_sha256": sha256_file(Path(record["paths"]["train"])),
                    "source_p0_bank_preserved": True,
                    "path": str(bank_path_value.resolve()),
                    "sha256": sha256_file(bank_path_value),
                    "rows": len(derived_rows),
                    "selected_negatives_per_prompt": 16,
                    "selection": "source_p0_error_class_sequence_then_within_class_reference_rank_spread",
                    "candidate_pool": "all_deterministic_verified_wrong_mutations",
                    "reference_rank_enters_training_weight": False,
                    "current_policy_surprisal_recomputed_each_update": True,
                    "coverage_threshold": None,
                    "coverage_sequence_matches_all_prompts": all(
                        bool(row["coverage_sequence_matches_source_p0"]) for row in audit_rows
                    ),
                    "error_class_coverage_fraction": _reference_surprisal_summary(
                        [float(row["error_class_coverage_fraction"]) for row in audit_rows]
                    ),
                    "singleton_selected_error_class_instances": sum(
                        int(row["singleton_selected_error_class_count"]) for row in audit_rows
                    ),
                    "multi_slot_selected_error_class_instances": endpoint_total,
                    "multi_slot_endpoint_coverage_count": endpoint_covered,
                    "multi_slot_endpoint_coverage_fraction": (
                        endpoint_covered / endpoint_total if endpoint_total else None
                    ),
                    "global_reference_rank_span_fraction": _reference_surprisal_summary(
                        [float(row["global_reference_rank_span_fraction"]) for row in audit_rows]
                    ),
                    "candidate_within_class_range": _reference_surprisal_summary(
                        [float(row["candidate_reference_surprisal"]["range"]) for row in class_rows]
                    ),
                    "candidate_within_class_iqr": _reference_surprisal_summary(
                        [float(row["candidate_reference_surprisal"]["iqr"]) for row in class_rows]
                    ),
                    "selected_within_class_range": _reference_surprisal_summary(
                        [
                            float(row["selected_reference_surprisal"]["range"])
                            for row in selected_class_rows
                        ]
                    ),
                    "selected_within_class_iqr": _reference_surprisal_summary(
                        [
                            float(row["selected_reference_surprisal"]["iqr"])
                            for row in selected_class_rows
                        ]
                    ),
                    "selected_range_median": float(np.median(np.asarray(ranges, dtype=float))),
                    "prompt_audit": str(audit_path.resolve()),
                    "prompt_audit_sha256": sha256_file(audit_path),
                    "complete": len(derived_rows) == int(config["split"]["p0_train_rows"]),
                    "scientific_status": "not_run",
                }
                if not summary["complete"]:
                    raise RuntimeError(f"Reference-remoteness bank is incomplete for {task}")
                atomic_json(root / "summary.json", summary)
                record["reference_remoteness_bank"] = summary
                atomic_json(output_root / "split_manifest.json", splits)
        finally:
            arena.clean_expression = original_clean_expression
            arena.SYSTEM_PROMPT = original_system_prompt
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    task_reference_summaries = {
        str(task): dict(splits["tasks"][str(task)]["reference_remoteness_bank"])
        for task in config["suite"]["p0_tasks"]
    }
    reference_summary_path = output_root / "reference_remoteness" / "summary.json"
    atomic_json(
        reference_summary_path,
        {
            "schema_version": 1,
            "experiment_id": experiment_id(config),
            "config_hash": stable_config_hash(config),
            "task_count": len(task_reference_summaries),
            "coverage_sequence_matches_all_prompts": all(
                bool(value["coverage_sequence_matches_all_prompts"])
                for value in task_reference_summaries.values()
            ),
            "tasks": {
                task: {
                    key: value[key]
                    for key in (
                        "rows",
                        "selected_negatives_per_prompt",
                        "coverage_sequence_matches_all_prompts",
                        "error_class_coverage_fraction",
                        "singleton_selected_error_class_instances",
                        "multi_slot_selected_error_class_instances",
                        "multi_slot_endpoint_coverage_count",
                        "multi_slot_endpoint_coverage_fraction",
                        "global_reference_rank_span_fraction",
                        "candidate_within_class_range",
                        "candidate_within_class_iqr",
                        "selected_within_class_range",
                        "selected_within_class_iqr",
                        "selected_range_median",
                    )
                }
                for task, value in sorted(task_reference_summaries.items())
            },
            "scientific_status": "not_run",
        },
    )
    splits["reference_remoteness_audit"] = {
        "path": str(reference_summary_path.resolve()),
        "sha256": sha256_file(reference_summary_path),
        "complete": True,
    }
    atomic_json(output_root / "split_manifest.json", splits)
    manifest = write_canonical_cold_inputs(config, output_root, splits)
    if not all(
        bool(manifest["tasks"][str(task)].get("reference_remoteness_bank_applied"))
        for task in config["suite"]["p0_tasks"]
    ):
        raise RuntimeError("Canonical transfer inputs did not bind every derived reference bank")
    return manifest


def write_canonical_cold_inputs(
    config: Mapping[str, Any],
    output_root: Path,
    split_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Write schema adapters and valid old-core configs; never implement training math."""

    if not _is_coldstart(config):
        raise RuntimeError("Canonical input conversion is cold-profile only")
    source_audit = audit_canonical_coldstart_sources(config)
    canonical_paths = _canonical_paths(config)
    records: dict[str, Any] = {}
    for task_value in config["suite"]["tasks"]:
        task = str(task_value)
        record = split_manifest["tasks"][task]
        source_paths = record["paths"]
        task_root = output_root / "canonical_inputs" / task
        task_root.mkdir(parents=True, exist_ok=True)
        if task == "countdown":
            train_path = Path(str(record["bank"])).resolve()
            validation_path = Path(str(record["countdown_validation_source"])).resolve()
            train_rows = read_jsonl(train_path)
            validation_rows = read_jsonl(validation_path)
            exact_countdown_sources = True
            reference_selection_applied = False
            reference_selection_identity = None
        else:
            reference_record = record.get("reference_remoteness_bank")
            if isinstance(reference_record, Mapping):
                reference_path = Path(str(reference_record.get("path", "")))
                if (
                    not reference_path.is_file()
                    or sha256_file(reference_path) != reference_record.get("sha256")
                    or not reference_record.get("complete")
                ):
                    raise RuntimeError(
                        f"Derived reference-remoteness bank identity failed for {task}"
                    )
                train_source = reference_path
                reference_selection_applied = True
                reference_selection_identity = str(reference_record["identity_hash"])
            else:
                # Initial prepare deliberately preserves P0 semantics. Formal calibration
                # replaces this with the derived training-only bank before any cell can run.
                train_source = Path(source_paths["train"])
                reference_selection_applied = False
                reference_selection_identity = None
            train_rows = [_canonical_train_row(row) for row in read_jsonl(train_source)]
            validation_rows = [
                _canonical_validation_row(row)
                for row in read_jsonl(Path(source_paths["validation"]))
            ]
            train_path = task_root / "train.jsonl"
            validation_path = task_root / "validation.jsonl"
            atomic_jsonl(train_path, train_rows)
            atomic_jsonl(validation_path, validation_rows)
            exact_countdown_sources = False
        sealed_test_path = task_root / "SEALED_TEST_NOT_ACCESSED.jsonl"
        sealed_test_path.parent.mkdir(parents=True, exist_ok=True)
        sealed_test_path.write_text("", encoding="utf-8")
        task_base_config, changed_fields = _task_base_config(
            config,
            task=task,
            canonical_paths=canonical_paths,
            task_root=task_root,
        )
        runtime_grids = _task_grid_configs(
            config,
            canonical_paths=canonical_paths,
            task_root=task_root,
        )
        canonical_record = {
            "train": str(train_path.resolve()),
            "validation": str(validation_path.resolve()),
            "sealed_test": str(sealed_test_path.resolve()),
            "base_config": str(task_base_config.resolve()),
            "base_config_sha256": sha256_file(task_base_config),
            "round1_grid": str(runtime_grids["round1_grid"]["path"].resolve()),
            "round1_grid_sha256": sha256_file(runtime_grids["round1_grid"]["path"]),
            "extension_grid": str(runtime_grids["extension_grid"]["path"].resolve()),
            "extension_grid_sha256": sha256_file(runtime_grids["extension_grid"]["path"]),
            "task_interface_changed_fields": changed_fields,
            "countdown_exact_source_files": exact_countdown_sources,
            "reference_remoteness_bank_applied": reference_selection_applied,
            "reference_remoteness_bank_identity_hash": reference_selection_identity,
            "negative_consumer": "all_unique_negatives_per_prompt",
            "calibration": "forbidden",
            "train_sha256": sha256_file(train_path),
            "validation_sha256": sha256_file(validation_path),
            "sealed_test_sha256": sha256_file(sealed_test_path),
            "train_rows": len(train_rows),
            "validation_rows": len(validation_rows),
            "test_rows": 0,
        }
        if not experiment_config.is_historical_coldstart_config(config):
            canonical_record["effective_runtime"] = experiment_config.effective_coldstart_runtime(
                config, task
            )
            canonical_record["runtime_grid_sources"] = {
                name: {
                    "source": str(runtime_grids[name]["source"].resolve()),
                    "source_sha256": sha256_file(runtime_grids[name]["source"]),
                    "changed_fields": list(runtime_grids[name]["changed_fields"]),
                }
                for name in ("round1_grid", "extension_grid")
            }
        record["canonical_coldstart"] = canonical_record
        records[task] = canonical_record

    split_manifest["canonical_source_audit"] = source_audit
    split_manifest["canonical_coldstart_complete"] = True
    atomic_json(output_root / "split_manifest.json", split_manifest)
    manifest = {
        "schema_version": 1,
        "experiment_id": experiment_id(config),
        "config_hash": stable_config_hash(config),
        "source_audit": source_audit,
        "tasks": records,
        "test_partition_accessed": False,
        "complete": set(records) == set(config["suite"]["tasks"]),
    }
    atomic_json(output_root / "canonical_inputs" / "manifest.json", manifest)
    return manifest


def resolve_task_inputs(
    config: Mapping[str, Any],
    *,
    p0_work_dir: Path,
    p0_config: Path,
    countdown_bank: Path,
    countdown_validation: Path,
    countdown_adapter: Path | None,
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
        reference_adapter=(countdown_adapter.resolve() if countdown_adapter is not None else None),
        sources_root=(p0_work_dir / "sources").resolve(),
        p0_config=p0_config.resolve(),
        countdown_validation=countdown_validation.resolve(),
    )
    for task, inputs in result.items():
        if not inputs.bank.is_file():
            raise FileNotFoundError(f"Missing bank for {task}: {inputs.bank}")
        if task == "countdown":
            if inputs.countdown_validation is None or not inputs.countdown_validation.is_file():
                raise FileNotFoundError("Missing Countdown validation file")
            if not _is_coldstart(config) and (
                inputs.reference_adapter is None
                or not (inputs.reference_adapter / "adapter_config.json").is_file()
            ):
                raise FileNotFoundError("Missing supplied Countdown reference adapter")
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
    countdown_adapter: Path | None,
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
    canonical_inputs = (
        write_canonical_cold_inputs(config, output_root, splits) if _is_coldstart(config) else None
    )
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
        "canonical_inputs": (
            str((output_root / "canonical_inputs" / "manifest.json").resolve())
            if canonical_inputs is not None
            else None
        ),
        "inputs": serialized_inputs,
        "complete": plan["cell_count"] == int(config["sweep"]["expected_cells"])
        and splits["complete"]
        and (canonical_inputs is None or bool(canonical_inputs["complete"])),
        "scientific_status": "not_run",
    }
    atomic_json(output_root / "prepare_manifest.json", manifest)
    atomic_json(output_root / "frozen_config.json", dict(config))
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
        if _is_coldstart(config):
            canonical = split_record.get("canonical_coldstart")
            if not isinstance(canonical, Mapping):
                raise RuntimeError(f"Missing canonical cold-start inputs for {task}")
            identity_fields = (
                ("train", "train_sha256"),
                ("validation", "validation_sha256"),
                ("sealed_test", "sealed_test_sha256"),
            )
            for path_key, sha_key in identity_fields:
                path = Path(str(canonical[path_key]))
                if not path.is_file() or sha256_file(path) != canonical[sha_key]:
                    raise RuntimeError(
                        f"Canonical cold-start {path_key} identity mismatch for {task}"
                    )
            if int(canonical.get("test_rows", -1)) != 0:
                raise RuntimeError("Canonical tuning input must keep the test partition sealed")
            for grid_key in ("round1_grid", "extension_grid", "base_config"):
                grid_path = Path(str(canonical[grid_key]))
                if (
                    not grid_path.is_file()
                    or sha256_file(grid_path) != canonical[f"{grid_key}_sha256"]
                ):
                    raise RuntimeError(f"Canonical paper input {grid_key} is missing for {task}")
            if canonical.get("negative_consumer") != "all_unique_negatives_per_prompt":
                raise RuntimeError(f"Canonical negative consumer drifted for {task}")
            if task == "countdown" and canonical.get("countdown_exact_source_files") is not True:
                raise RuntimeError(
                    "Countdown must dispatch the exact generated bank/validation files"
                )
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
    if _is_coldstart(config):
        if any(value.reference_adapter is not None for value in inputs.values()):
            raise RuntimeError("Cold-start prepared inputs must not contain external adapters")
        return splits, inputs
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


def _trainable_state_sha256(model: Any) -> str:
    """Hash fresh/trainable state without serializing an adapter to disk."""

    if torch is None:
        raise RuntimeError("Torch is required")
    digest = hashlib.sha256()
    found = False
    for name, parameter in sorted(model.named_parameters(), key=lambda item: item[0]):
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
    identity = {
        "schema_version": 1,
        "experiment_id": experiment_id(config),
        "config_hash": stable_config_hash(config),
        "task": task,
        "bank_sha256": split_manifest["tasks"][task]["bank_sha256"],
        "train_prompt_hash": split_manifest["tasks"][task]["prompt_id_hashes"]["train"],
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


def _canonical_task_record(split_manifest: Mapping[str, Any], task: str) -> Mapping[str, Any]:
    value = split_manifest["tasks"][task].get("canonical_coldstart")
    if not isinstance(value, Mapping):
        raise TypeError(f"Missing canonical cold-start task record for {task}")
    return value


def _canonical_calibration_identity(
    task: str,
    *,
    split_manifest: Mapping[str, Any],
    base_model_path: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    record = _canonical_task_record(split_manifest, task)
    value = {
        "schema_version": 1,
        "experiment_id": experiment_id(config),
        "config_hash": stable_config_hash(config),
        "task": task,
        "base_model_identity": model_identity(base_model_path, None)["model"],
        "canonical_train_sha256": record["train_sha256"],
        "reference_remoteness_bank_identity_hash": record.get(
            "reference_remoteness_bank_identity_hash"
        ),
        "canonical_base_config_sha256": sha256_file(Path(str(record["base_config"]))),
        "canonical_base_config_git_blob_sha": config["canonical_coldstart"][
            "expected_git_blob_shas"
        ]["base_config"],
        "canonical_source_git_blob_shas": dict(
            config["canonical_coldstart"]["expected_git_blob_shas"]
        ),
    }
    value["identity_hash"] = stable_hash(value)
    return value


def _paper_grid_for_cell(record: Mapping[str, Any], cell: Cell) -> Path:
    if cell.task != "countdown":
        # Transfer c values are passed directly to the locked trainer. The round-1
        # grid supplies only the frozen training/runtime profile.
        return Path(str(record["round1_grid"]))
    coefficient = 0.0 if cell.lambda_value is None else float(cell.lambda_value)
    return Path(str(record[_paper_grid_name(coefficient)]))


def calibrate_canonical_cold_task(
    task: str,
    *,
    split_manifest: Mapping[str, Any],
    base_model_path: str,
    config: Mapping[str, Any],
    output_root: Path,
    force: bool,
) -> dict[str, Any]:
    record = _canonical_task_record(split_manifest, task)
    identity = _canonical_calibration_identity(
        task,
        split_manifest=split_manifest,
        base_model_path=base_model_path,
        config=config,
    )
    result_path = output_root / "calibration" / f"{task}.json"
    if result_path.is_file() and not force:
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("identity_hash") == identity["identity_hash"] and existing.get("complete"):
            return existing
        raise RuntimeError(f"Existing canonical calibration identity mismatch for {task}")
    result = {
        **identity,
        "enabled": False,
        "mode": "paper_linear_surprisal_no_calibration",
        "coordinate": "current_sequence_surprisal/2",
        "detached": True,
        "extra_square": False,
        "gradient_rms_matching": False,
        "canonical_train_sha256": record["train_sha256"],
        "task_metrics_used": False,
        "test_data_used": False,
        "complete": True,
        "scientific_status": "not_run",
    }
    atomic_json(result_path, result)
    return result


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
        raise RuntimeError("Cold-start calibration must call the canonical base/taper modules")
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
        "initialization_state_sha256": initialization_state_sha256,
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
    if _is_coldstart(config) and not _is_engineering_self_test(config):
        _derive_reference_remoteness_banks(
            config,
            output_root,
            base_model_path=base_model_path,
        )
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
    if _is_coldstart(config):
        requested_results = {
            task: calibrate_canonical_cold_task(
                task,
                split_manifest=splits,
                base_model_path=base_model_path,
                config=config,
                output_root=output_root,
                force=force,
            )
            for task in requested
        }
    else:
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
        expected_identity = (
            _canonical_calibration_identity(
                task,
                split_manifest=splits,
                base_model_path=base_model_path,
                config=config,
            )
            if _is_coldstart(config)
            else _calibration_identity(
                task,
                inputs=inputs[task],
                split_manifest=splits,
                base_model_path=base_model_path,
                config=config,
            )
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


def cmd_calibrate_task(
    config: Mapping[str, Any],
    output_root: Path,
    *,
    base_model_path: str,
    task: str,
    force: bool,
) -> dict[str, Any]:
    """Calibrate one cold task without racing the shared top-level manifest."""

    if not _is_coldstart(config):
        raise RuntimeError("calibrate-task is available only for canonical cold-start")
    if task not in config["suite"]["tasks"]:
        raise ValueError(f"Unknown calibration task: {task}")
    if not _is_engineering_self_test(config):
        _derive_reference_remoteness_banks(
            config,
            output_root,
            base_model_path=base_model_path,
        )
    splits, _ = _load_ready_inputs(
        output_root,
        config,
        base_model_path=base_model_path,
    )
    return calibrate_canonical_cold_task(
        task,
        split_manifest=splits,
        base_model_path=base_model_path,
        config=config,
        output_root=output_root,
        force=force,
    )


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


def _canonical_environment_evaluator(
    *,
    arena: Any,
    task_adapter: Any,
    instances: Mapping[str, TaskInstance],
    greedy_prompt_rows: int,
    passk_prompt_rows: int,
) -> Any:
    """Return the old evaluator signature backed only by the selected task verifier."""

    def evaluate_rows(
        model: Any,
        tokenizer: Any,
        rows: Sequence[Mapping[str, Any]],
        batch_size: int,
        max_new_tokens: int,
        pass_k: int,
        seed: int,
        known_structures: set[str] | None = None,
    ) -> dict[str, Any]:
        del known_structures
        if not rows:
            raise RuntimeError("Task validation rows must be non-empty")
        if len(rows) < int(greedy_prompt_rows) or len(rows) < int(passk_prompt_rows):
            raise RuntimeError(
                "Task validation input is smaller than the frozen Greedy/Pass@k budgets"
            )
        arena.seed_all(seed)
        was_training = bool(model.training)
        greedy_correct: list[float] = []
        greedy_valid: list[float] = []
        pass_success: list[float] = []
        sampled_valid: list[float] = []
        greedy_rows = list(rows[: int(greedy_prompt_rows)])
        sampled_rows = list(rows[: int(passk_prompt_rows)])
        for start in range(0, len(greedy_rows), int(batch_size)):
            chunk = greedy_rows[start : start + int(batch_size)]
            prompts = [str(row["prompt"]) for row in chunk]
            greedy = arena.generate_outputs(
                model,
                tokenizer,
                prompts,
                int(max_new_tokens),
                False,
                1.0,
                1.0,
                1,
            )
            for row, greedy_outputs in zip(chunk, greedy, strict=True):
                instance = instances[str(row["prompt_id"])]
                greedy_result = task_adapter.verify(instance, greedy_outputs[0])
                greedy_correct.append(float(greedy_result.correct))
                greedy_valid.append(float(greedy_result.format_valid))
        for start in range(0, len(sampled_rows), int(batch_size)):
            chunk = sampled_rows[start : start + int(batch_size)]
            prompts = [str(row["prompt"]) for row in chunk]
            sampled = arena.generate_outputs(
                model,
                tokenizer,
                prompts,
                int(max_new_tokens),
                int(pass_k) > 1,
                0.8 if int(pass_k) > 1 else 1.0,
                0.95 if int(pass_k) > 1 else 1.0,
                int(pass_k),
            )
            for row, sampled_outputs in zip(chunk, sampled, strict=True):
                instance = instances[str(row["prompt_id"])]
                sample_results = [
                    task_adapter.verify(instance, completion) for completion in sampled_outputs
                ]
                pass_success.append(float(any(result.correct for result in sample_results)))
                sampled_valid.extend(float(result.format_valid) for result in sample_results)
        metrics = {
            "greedy_success": float(np.mean(greedy_correct)),
            "pass_at_k": float(np.mean(pass_success)),
            "valid_rate": float(np.mean(greedy_valid)),
            "sampled_valid_rate": float(np.mean(sampled_valid)),
            "n_eval": float(len(greedy_rows)),
            "greedy_prompt_rows": float(len(greedy_rows)),
            "passk_prompt_rows": float(len(sampled_rows)),
            "task_verifier_interface": True,
        }
        numeric = [value for value in metrics.values() if isinstance(value, (int, float))]
        if not all(math.isfinite(float(value)) for value in numeric):
            raise RuntimeError("Task verifier evaluation produced a non-finite metric")
        if int(pass_k) == 8:
            setattr(
                evaluate_rows,
                "_last_primary_sampled_valid_rate",
                float(metrics["sampled_valid_rate"]),
            )
        if was_training:
            model.train()
        return metrics

    return evaluate_rows


def _canonical_generic_posthoc(
    *,
    arena: Any,
    evaluator: Any,
    model_path: Path,
    checkpoint: Path,
    validation_path: Path,
    base_config: Mapping[str, Any],
    seed_offset: int,
) -> dict[str, Any]:
    tokenizer = arena.load_tokenizer(str(model_path))
    model_cfg = base_config["model"]
    model = arena.load_model(
        str(model_path),
        str(checkpoint),
        trainable_adapter=False,
        load_in_4bit=bool(model_cfg.get("load_in_4bit", False)),
        dtype=str(model_cfg.get("dtype", "auto")),
        gradient_checkpointing=False,
        parameterization="lora",
    )
    rows = arena.read_jsonl(validation_path)
    eval_cfg = base_config["evaluation"]
    result: dict[str, Any] = {}
    for pass_k_value in eval_cfg["pass_ks"]:
        pass_k = int(pass_k_value)
        metrics = evaluator(
            model,
            tokenizer,
            rows[: int(eval_cfg["examples"])],
            int(eval_cfg["batch_size"]),
            int(model_cfg["max_new_tokens"]),
            pass_k,
            int(eval_cfg["seed"]) + int(seed_offset) + pass_k,
        )
        if pass_k == int(eval_cfg["pass_ks"][0]):
            result["validation_greedy_success"] = float(metrics["greedy_success"])
            result["validation_valid_rate"] = float(metrics["valid_rate"])
            result["validation_sampled_valid_rate"] = float(
                metrics.get("sampled_valid_rate", metrics["valid_rate"])
            )
            result["validation_n_eval"] = float(metrics["n_eval"])
        result[f"validation_pass_at_{pass_k}"] = float(metrics["pass_at_k"])
    del model
    gc.collect()
    if torch is not None and torch.cuda.is_available():
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
        "initialization": (
            dict(config["initialization"])
            if _is_coldstart(config)
            else {
                "source": "reference_adapter",
                "reference_adapter_identity": model_identity(
                    base_model_path, str(inputs.reference_adapter)
                )["adapter"],
            }
        ),
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
        "validation_terminal_sampled_valid_rate": (
            None
            if terminal.get("sampled_valid_rate") is None
            else float(terminal["sampled_valid_rate"])
        ),
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


def _runtime_bridge_contract(effective: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "fresh_lora": {
            "rank": int(effective["model"]["lora_rank"]),
            "alpha": int(effective["model"]["lora_alpha"]),
            "dropout": float(effective["model"]["lora_dropout"]),
        },
        "gradient_checkpointing": bool(effective["model"]["gradient_checkpointing"]),
        "optimizer_weight_decay": float(effective["training"]["weight_decay"]),
        "sampling_temperature": float(effective["evaluation"]["sampling_temperature"]),
        "top_p": float(effective["evaluation"]["top_p"]),
    }


@contextmanager
def _legacy_arena_runtime_bridge(arena: Any, effective: Mapping[str, Any]) -> Any:
    """Temporarily parameterize legacy arena interface literals; never touch loss math."""

    original_lora_config = arena.LoraConfig
    original_load_model = arena.load_model
    original_generate_outputs = arena.generate_outputs
    original_scheduler = getattr(arena, "get_cosine_schedule_with_warmup", None)
    contract = _runtime_bridge_contract(effective)

    def configured_lora_config(*args: Any, **kwargs: Any) -> Any:
        kwargs["r"] = int(contract["fresh_lora"]["rank"])
        kwargs["lora_alpha"] = int(contract["fresh_lora"]["alpha"])
        kwargs["lora_dropout"] = float(contract["fresh_lora"]["dropout"])
        return original_lora_config(*args, **kwargs)

    def configured_load_model(*args: Any, **kwargs: Any) -> Any:
        values = list(args)
        if len(values) > 5:
            values[5] = bool(contract["gradient_checkpointing"])
        else:
            kwargs["gradient_checkpointing"] = bool(contract["gradient_checkpointing"])
        return original_load_model(*values, **kwargs)

    def configured_generate_outputs(
        model: Any,
        tokenizer: Any,
        prompts: list[str],
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
        num_return_sequences: int = 1,
    ) -> Any:
        if do_sample:
            temperature = float(contract["sampling_temperature"])
            top_p = float(contract["top_p"])
        return original_generate_outputs(
            model,
            tokenizer,
            prompts,
            max_new_tokens,
            do_sample,
            temperature,
            top_p,
            num_return_sequences,
        )

    def configured_scheduler(
        optimizer: Any,
        num_warmup_steps: int,
        num_training_steps: int,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if original_scheduler is None:
            raise RuntimeError("Canonical arena scheduler is unavailable")
        if float(effective["training"]["warmup_ratio"]) == 0.0:
            num_warmup_steps = 0
        return original_scheduler(optimizer, num_warmup_steps, num_training_steps, *args, **kwargs)

    arena.LoraConfig = configured_lora_config
    arena.load_model = configured_load_model
    arena.generate_outputs = configured_generate_outputs
    if original_scheduler is not None:
        arena.get_cosine_schedule_with_warmup = configured_scheduler
    try:
        yield contract
    finally:
        arena.LoraConfig = original_lora_config
        arena.load_model = original_load_model
        arena.generate_outputs = original_generate_outputs
        if original_scheduler is not None:
            arena.get_cosine_schedule_with_warmup = original_scheduler


def _validated_runtime_grid(
    candidate: Mapping[str, Any],
    *,
    canonical_grid: Mapping[str, Any],
    effective: Mapping[str, Any],
    strict_validator: Any,
) -> None:
    allowed = {"training.steps", "training.eval_every"}
    changed = set(_changed_leaf_paths(canonical_grid, candidate))
    forbidden = sorted(changed - allowed)
    if forbidden:
        raise ValueError(f"Derived runtime grid changed non-runtime fields: {forbidden}")
    training = candidate.get("training", {})
    if int(training.get("steps", -1)) != int(effective["training"]["optimizer_updates"]):
        raise ValueError("Derived runtime grid steps do not match effective runtime")
    if int(training.get("eval_every", -1)) != int(
        effective["training"]["evaluation_every_updates"]
    ):
        raise ValueError("Derived runtime grid eval_every does not match effective runtime")
    strict = copy.deepcopy(dict(candidate))
    strict["training"]["steps"] = canonical_grid["training"]["steps"]
    strict["training"]["eval_every"] = canonical_grid["training"]["eval_every"]
    strict_validator(strict)


@contextmanager
def _legacy_paper_runtime_bridge(
    modules: Mapping[str, Any],
    effective: Mapping[str, Any],
    *,
    grid_path: Path,
    grid_source_path: Path,
) -> Any:
    """Bridge configured runtime scalars into byte-locked paper interfaces."""

    scan_trainer = modules["scan_trainer"]
    paper_common = modules["paper_common"]
    canonical_grid = yaml.safe_load(grid_source_path.read_text(encoding="utf-8"))
    if not isinstance(canonical_grid, dict):
        raise TypeError("Canonical paper grid root must be a mapping")
    candidate_grid = yaml.safe_load(grid_path.read_text(encoding="utf-8"))
    if not isinstance(candidate_grid, dict):
        raise TypeError("Derived paper grid root must be a mapping")

    validator_targets: list[tuple[Any, str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for module_name in ("paper_common", "scan_common", "scan_trainer", "scan_runtime"):
        module = modules[module_name]
        if not hasattr(module, "validate_grid_config"):
            continue
        key = (id(module), "validate_grid_config")
        if key in seen:
            continue
        seen.add(key)
        validator_targets.append((module, "validate_grid_config", module.validate_grid_config))
    strict_validator = paper_common.validate_grid_config

    def configured_validator(value: Mapping[str, Any]) -> None:
        _validated_runtime_grid(
            value,
            canonical_grid=canonical_grid,
            effective=effective,
            strict_validator=strict_validator,
        )

    optimizer_holder = scan_trainer.torch.optim
    original_adamw = optimizer_holder.AdamW

    def configured_adamw(*args: Any, **kwargs: Any) -> Any:
        kwargs["weight_decay"] = float(effective["training"]["weight_decay"])
        return original_adamw(*args, **kwargs)

    with _legacy_arena_runtime_bridge(modules["arena"], effective) as contract:
        optimizer_holder.AdamW = configured_adamw
        for module, name, _ in validator_targets:
            setattr(module, name, configured_validator)
        try:
            configured_validator(candidate_grid)
            yield contract
        finally:
            optimizer_holder.AdamW = original_adamw
            for module, name, original in validator_targets:
                setattr(module, name, original)


def _train_canonical_cold_cell(
    cell: Cell,
    *,
    inputs: TaskInputs,
    split_manifest: Mapping[str, Any],
    base_model_path: str,
    config: Mapping[str, Any],
    output_root: Path,
    force: bool,
    root_name: str = "cells",
    base_config_override: Path | None = None,
) -> dict[str, Any]:
    """Dispatch one cell to the exact paper runtime; this owns no loss math."""

    if not _is_coldstart(config):
        raise RuntimeError("Canonical cold dispatch is cold-profile only")
    modules = _canonical_cold_modules(config)
    arena = modules["arena"]
    record = _canonical_task_record(split_manifest, cell.task)
    calibration_path = output_root / "calibration" / f"{cell.task}.json"
    if not calibration_path.is_file():
        raise RuntimeError(f"Run the no-calibration identity gate before {cell.task}")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    expected_calibration = _canonical_calibration_identity(
        cell.task,
        split_manifest=split_manifest,
        base_model_path=base_model_path,
        config=config,
    )
    if (
        calibration.get("identity_hash") != expected_calibration["identity_hash"]
        or calibration.get("enabled") is not False
        or not calibration.get("complete")
    ):
        raise RuntimeError(f"Paper no-calibration identity mismatch for {cell.task}")

    bank = Path(str(record["train"]))
    validation = Path(str(record["validation"]))
    base_config_path = base_config_override or Path(str(record["base_config"]))
    grid_path = _paper_grid_for_cell(record, cell)
    grid_source_name = (
        "round1_grid"
        if cell.task != "countdown"
        else _paper_grid_name(0.0 if cell.lambda_value is None else float(cell.lambda_value))
    )
    grid_source_path = _canonical_paths(config)[grid_source_name]
    modules = _activate_paper_grid_modules(modules, grid_source_path)
    arena = modules["arena"]
    runtime = modules["paper_runtime"]
    effective_runtime = experiment_config.effective_coldstart_runtime(config, cell.task)
    base_config = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    if not isinstance(base_config, dict):
        raise TypeError("Paper base config root must be a mapping")
    identity = _cell_identity(
        cell,
        inputs=inputs,
        split_manifest=split_manifest,
        base_model_path=base_model_path,
        config=config,
        calibration=calibration,
    )
    identity.update(
        {
            "canonical_source_git_blob_shas": dict(
                config["canonical_coldstart"]["expected_git_blob_shas"]
            ),
            "canonical_dispatch": (
                "countdown_e8_alpha1_highc_scan_runtime.worker"
                if cell.task == "countdown"
                else "countdown_e8_alpha1_c_scan_trainer.train_cell"
            ),
            "paper_formula": "alpha*exp(-c*(current_sequence_surprisal/2))",
            "paper_grid_config": str(grid_path.resolve()),
            "paper_grid_config_sha256": sha256_file(grid_path),
            "paper_grid_source": str(grid_source_path.resolve()),
            "paper_grid_source_sha256": sha256_file(grid_source_path),
            "paper_base_config": str(base_config_path.resolve()),
            "paper_base_config_sha256": sha256_file(base_config_path),
            "all_unique_negatives": True,
            "near_far_selection": False,
            "gradient_rms_matching": False,
            "task_runtime_contract": dict(config["task_runtime"][cell.task]),
        }
    )
    if not experiment_config.is_historical_coldstart_config(config):
        identity["effective_runtime"] = effective_runtime
        identity["legacy_runtime_bridge"] = _runtime_bridge_contract(effective_runtime)
    identity["identity_hash"] = stable_hash(identity)

    cell_root = output_root / root_name / cell.key
    manifest_path = cell_root / "cell_manifest.json"
    if manifest_path.is_file() and not force:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("identity_hash") == identity["identity_hash"] and existing.get("complete"):
            return existing
        raise RuntimeError(f"Existing paper cell identity mismatch: {cell.key}")
    if cell_root.exists():
        if not force:
            raise RuntimeError(f"Cell output exists without a reusable manifest: {cell_root}")
        expected_parent = (output_root / root_name).resolve()
        if expected_parent not in cell_root.resolve().parents:
            raise RuntimeError(f"Refusing unsafe paper cell removal: {cell_root}")
        shutil.rmtree(cell_root)
    cell_root.mkdir(parents=True, exist_ok=False)
    canonical_output = cell_root / "canonical"

    evaluator = None
    if cell.task != "countdown":
        validation_rows = read_jsonl(validation)
        task_adapter, instances = _load_task_adapter_and_instances(
            cell.task,
            inputs=inputs,
            validation_rows=validation_rows,
        )
        evaluator = _canonical_environment_evaluator(
            arena=arena,
            task_adapter=task_adapter,
            instances=instances,
            greedy_prompt_rows=int(config["task_runtime"][cell.task]["greedy_prompt_rows"]),
            passk_prompt_rows=int(config["task_runtime"][cell.task]["passk_prompt_rows"]),
        )

    scan_trainer = modules["scan_trainer"]
    paper_common = modules["paper_common"]

    @contextmanager
    def task_interface() -> Any:
        original_evaluate_rows = arena.evaluate_rows
        original_clean_expression = arena.clean_expression
        original_system_prompt = arena.SYSTEM_PROMPT
        original_completion_stats = arena.completion_stats
        original_trainer_evaluate = scan_trainer._evaluate_validation
        try:
            if evaluator is None:

                def configured_evaluate(**kwargs: Any) -> dict[str, Any]:
                    kwargs["pass64_enabled"] = 64 in set(
                        int(value) for value in effective_runtime["evaluation"]["auxiliary_pass_ks"]
                    )
                    return original_trainer_evaluate(**kwargs)

                scan_trainer._evaluate_validation = configured_evaluate
            if evaluator is not None:
                arena.evaluate_rows = evaluator
                arena.SYSTEM_PROMPT = TRANSFER_SYSTEM_PROMPT
                arena.completion_stats = lambda model, batch: {
                    "seq_lp": -arena.sequence_surprisal_only(model, batch)
                }
                # Non-arithmetic task outputs are already canonicalized by the P0
                # verifier. Arithmetic-only cleanup would corrupt structured outputs.
                arena.clean_expression = lambda value: str(value)

                def configured_evaluate(**kwargs: Any) -> dict[str, Any]:
                    kwargs["pass64_enabled"] = 64 in set(
                        int(value) for value in effective_runtime["evaluation"]["auxiliary_pass_ks"]
                    )
                    row = original_trainer_evaluate(**kwargs)
                    sampled_valid_rate = getattr(
                        evaluator, "_last_primary_sampled_valid_rate", None
                    )
                    if sampled_valid_rate is None:
                        raise RuntimeError(
                            "Transfer evaluator did not report primary sampled validity"
                        )
                    row["val_sampled_valid_rate"] = float(sampled_valid_rate)
                    return row

                scan_trainer._evaluate_validation = configured_evaluate
            yield
        finally:
            arena.evaluate_rows = original_evaluate_rows
            arena.clean_expression = original_clean_expression
            arena.SYSTEM_PROMPT = original_system_prompt
            arena.completion_stats = original_completion_stats
            scan_trainer._evaluate_validation = original_trainer_evaluate

    alpha = 0.0 if cell.method == METHOD_POSITIVE_ONLY else 1.0
    coefficient = (
        0.0 if cell.method in {METHOD_POSITIVE_ONLY, METHOD_GLOBAL} else float(cell.lambda_value)
    )
    with (
        _legacy_paper_runtime_bridge(
            modules,
            effective_runtime,
            grid_path=grid_path,
            grid_source_path=grid_source_path,
        ),
        task_interface(),
    ):
        if cell.task == "countdown":
            returncode = runtime.worker(
                argparse.Namespace(
                    family="exponential",
                    alpha=alpha,
                    c=coefficient,
                    seed_offset=int(cell.seed),
                    output_dir=str(canonical_output),
                    model_path=str(Path(base_model_path).resolve()),
                    bank=str(bank),
                    val=str(validation),
                    base_config=str(base_config_path),
                    grid_config=str(grid_path),
                )
            )
            if int(returncode) != 0:
                raise RuntimeError(f"Paper runtime failed for {cell.key}")
        else:
            paper_cell = paper_common.Cell(
                alpha=alpha,
                coefficient=coefficient,
                seed_offset=int(cell.seed),
                family="exponential",
            )
            scan_trainer.train_cell(
                cell=paper_cell,
                model_path=Path(base_model_path).resolve(),
                bank=bank,
                val=validation,
                base_config_path=base_config_path,
                grid_config_path=grid_path,
                output_dir=canonical_output,
                repo=_repo_root(),
                smoke=False,
            )

    canonical_summary_path = canonical_output / "summary.json"
    canonical_summary = json.loads(canonical_summary_path.read_text(encoding="utf-8"))
    metrics_path = canonical_output / "metrics.csv"
    with metrics_path.open(encoding="utf-8", newline="") as handle:
        metric_rows = list(csv.DictReader(handle))
    evaluations = [
        {
            "update": int(row["step"]),
            "pass8": float(row["val_pass_at_8"]),
            "greedy_success": float(row["val_greedy"]),
            "greedy_valid_rate": float(row["val_valid_rate"]),
            "sampled_valid_rate": (
                None
                if row.get("val_sampled_valid_rate") in (None, "")
                else float(row["val_sampled_valid_rate"])
            ),
        }
        for row in metric_rows
    ]
    metrics_summary = _summarize_evaluations(evaluations, config)
    numerical_failure = canonical_summary.get("numerical_failure")
    best_adapter = canonical_output / "best_pass8_adapter"
    terminal_adapter = canonical_output / (
        "last_finite_adapter" if numerical_failure else "terminal_adapter"
    )
    result = {
        **identity,
        **metrics_summary,
        "canonical_summary": str(canonical_summary_path.resolve()),
        "canonical_summary_sha256": sha256_file(canonical_summary_path),
        "canonical_output": str(canonical_output.resolve()),
        "canonical_training_metrics": str(metrics_path.resolve()),
        "training_metrics": str(metrics_path.resolve()),
        "best_adapter": str(best_adapter.resolve()),
        "terminal_adapter": str(terminal_adapter.resolve()),
        "terminal_adapter_identity": model_identity(base_model_path, str(terminal_adapter))[
            "adapter"
        ],
        "canonical_dispatch_verified": True,
        "countdown_protocol_exact": cell.task == "countdown",
        "task_interface_adapter_only": cell.task != "countdown",
        "reference_remoteness_bank_identity_hash": record.get(
            "reference_remoteness_bank_identity_hash"
        ),
        "static_reference_rank_enters_training_weight": False,
        "current_policy_surprisal_recomputed_each_update": True,
        "adapter_path_argument": None,
        "sft_adapter_path_argument": None,
        "initialization_optimizer_updates": 0,
        "best_step": metrics_summary["supplementary_best_step"],
        "terminal_step": canonical_summary.get("terminal_step"),
        "stop_reason": canonical_summary.get("stop_reason"),
        "validation_best_pass8": metrics_summary["supplementary_best_pass8"],
        "validation_terminal_pass8": metrics_summary["validation_terminal_pass8"],
        "validation_best_greedy": metrics_summary["supplementary_best_greedy"],
        "validation_terminal_greedy": metrics_summary["validation_terminal_greedy"],
        "validation_best_greedy_valid_rate": max(
            float(row["greedy_valid_rate"]) for row in evaluations
        ),
        "validation_terminal_greedy_valid_rate": metrics_summary[
            "validation_terminal_greedy_valid_rate"
        ],
        "validation_terminal_sampled_valid_rate": metrics_summary[
            "validation_terminal_sampled_valid_rate"
        ],
        "optimizer_updates": int(canonical_summary.get("terminal_step", 0)),
        "numerical_failure": numerical_failure,
        "nan_inf_failure": numerical_failure is not None,
        "evaluation_status": "complete" if evaluations else "incomplete",
        "test_partition_accessed": False,
        "complete": bool(evaluations and numerical_failure is None),
        "scientific_status": "pilot",
    }
    atomic_json(manifest_path, result)
    return result


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

    initialization_seed = (
        int(config["initialization"]["seed"]) if _is_coldstart(config) else cell.seed
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
        "initialization_state_sha256": initialization_state_sha256,
        "terminal_trainable_state_sha256": terminal_state_sha256,
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
        if _is_coldstart(config):
            if updates_override is not None or engineering_liveness:
                raise RuntimeError(
                    "Canonical cold liveness requires its derived old-core config path"
                )
            return _train_canonical_cold_cell(
                cell,
                inputs=inputs,
                split_manifest=split_manifest,
                base_model_path=base_model_path,
                config=config,
                output_root=output_root,
                force=force,
            )
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


def _adapter_weight_file(adapter_root: Path) -> Path:
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        candidate = adapter_root / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Adapter weight file is missing: {adapter_root}")


def _canonical_liveness_base_config(config: Mapping[str, Any], output_root: Path) -> Path:
    base_path = _canonical_paths(config)["base_config"]
    value = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("Canonical base config root must be a mapping")
    value["offline_training"].update(
        {
            "steps": 2,
            "min_steps": 2,
            "early_stop_patience": 10,
            # Preserve the step-0 adapter so it can be compared bytewise with
            # the terminal adapter after the canonical optimizer updates.
            "early_stop_delta": 1.0e9,
            "eval_every": 2,
            "log_every": 1,
            "diagnostic_examples": 2,
            "diagnostic_gradient_examples": 1,
            "diagnostic_batch": 1,
            "num_workers": 0,
        }
    )
    value["evaluation"].update(
        {
            "examples": 2,
            "test_examples": 0,
            "batch_size": 1,
            "pass_ks": [8],
        }
    )
    path = output_root / "liveness" / "canonical_liveness_base_config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".yaml.tmp")
    temporary.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    temporary.replace(path)
    return path


def _canonical_cold_liveness_cell(grid_path: Path) -> Cell:
    """Derive the wrapper identity from the exact grid consumed by canonical smoke."""

    grid = yaml.safe_load(grid_path.read_text(encoding="utf-8"))
    if not isinstance(grid, dict):
        raise TypeError("Canonical liveness grid root must be a mapping")
    liveness = grid["execution"]["liveness"]
    seed_offsets = grid["sweep"]["seed_offsets"]
    coefficient = float(liveness["representative_c"])
    return Cell(
        "countdown",
        METHOD_EXPONENTIAL,
        math.exp(-coefficient),
        int(seed_offsets[0]),
        "liveness",
        coefficient,
    )


def _cmd_canonical_cold_liveness(
    config: Mapping[str, Any],
    config_path: Path,
    output_root: Path,
    *,
    inputs: TaskInputs,
    splits: Mapping[str, Any],
    base_model_path: str,
    force: bool,
) -> dict[str, Any]:
    modules = _canonical_cold_modules(config)
    runtime = modules["paper_runtime"]
    record = _canonical_task_record(splits, "countdown")
    grid_path = Path(str(record["round1_grid"]))
    cell = _canonical_cold_liveness_cell(grid_path)
    smoke_root = output_root / "liveness" / "paper_runtime_smoke"
    if force and smoke_root.exists():
        shutil.rmtree(smoke_root)
    returncode = runtime.smoke(
        argparse.Namespace(
            model_path=str(Path(base_model_path).resolve()),
            bank=str(Path(str(record["train"])).resolve()),
            val=str(Path(str(record["validation"])).resolve()),
            base_config=str(Path(str(record["base_config"])).resolve()),
            grid_config=str(grid_path.resolve()),
            work_dir=str(smoke_root.resolve()),
        )
    )
    gate = json.loads((smoke_root / "SMOKE_GATE.json").read_text(encoding="utf-8"))
    if int(returncode) != 0 or gate.get("status") != "PASS":
        raise RuntimeError("Paper-runtime two-update smoke gate failed")
    summary_path = Path(str(gate["summary"]))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    diagnostics_path = Path(str(summary["diagnostic_files"]["training"]))
    diagnostics = read_jsonl(diagnostics_path)
    optimizer_update_norms = [
        float(row["optimizer_update_norm"])
        for row in diagnostics
        if row.get("optimizer_update_norm") is not None
    ]
    if (
        int(summary.get("terminal_step") or -1) != 2
        or summary.get("numerical_failure") is not None
        or not optimizer_update_norms
        or not all(math.isfinite(value) and value > 0.0 for value in optimizer_update_norms)
    ):
        raise RuntimeError("Paper-runtime liveness did not perform two finite optimizer updates")
    canonical_output = summary_path.parent
    terminal_adapter = canonical_output / "terminal_adapter"
    terminal_hash = sha256_file(_adapter_weight_file(terminal_adapter))

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
        str(terminal_adapter),
    ]
    reload_process = subprocess.run(
        reload_command,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
        check=False,
    )
    if reload_process.returncode != 0:
        raise RuntimeError(
            f"Fresh-process canonical adapter reload failed: {reload_process.stderr[-2000:]}"
        )
    try:
        reload_result = json.loads(reload_process.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Fresh-process canonical reload returned invalid JSON") from exc
    if (
        not reload_result.get("complete")
        or not reload_result.get("finite")
        or int(reload_result.get("process_id", os.getpid())) == os.getpid()
        or reload_result.get("adapter_identity")
        != model_identity(base_model_path, str(terminal_adapter))["adapter"]
    ):
        raise RuntimeError("Fresh-process canonical adapter reload gate failed")
    calibration = json.loads(
        (output_root / "calibration" / "countdown.json").read_text(encoding="utf-8")
    )
    result = {
        **_cell_identity(
            cell,
            inputs=inputs,
            split_manifest=splits,
            base_model_path=base_model_path,
            config=config,
            calibration=calibration,
        ),
        "canonical_dispatch": "countdown_e8_alpha1_highc_scan_runtime.smoke",
        "canonical_dispatch_verified": True,
        "canonical_summary": str(summary_path.resolve()),
        "canonical_summary_sha256": sha256_file(summary_path),
        "terminal_adapter": str(terminal_adapter.resolve()),
        "terminal_adapter_identity": model_identity(base_model_path, str(terminal_adapter))[
            "adapter"
        ],
        "engineering_liveness": True,
        "optimizer_updates": 2,
        "optimizer_update_norm": min(optimizer_update_norms),
        "terminal_adapter_weight_sha256": terminal_hash,
        "adapter_weight_changed": True,
        "finite_old_core_updates": True,
        "reload_gate_passed": True,
        "fresh_process_reload_passed": True,
        "liveness_parent_process_id": os.getpid(),
        "reload_process_id": int(reload_result["process_id"]),
        "nan_inf_failure": False,
        "evaluation_status": "complete",
        "complete": True,
        "scientific_status": "not_run",
    }
    result["identity_hash"] = stable_hash(result)
    atomic_json(
        output_root / "liveness" / cell.key / "cell_manifest.json",
        result,
    )
    return result


def cmd_liveness(
    config: Mapping[str, Any],
    config_path: Path,
    output_root: Path,
    *,
    task: str,
    rho: float | None,
    base_model_path: str,
    force: bool,
) -> dict[str, Any]:
    if task not in config["suite"]["tasks"]:
        raise ValueError(f"Unknown liveness task: {task}")
    if _is_coldstart(config):
        if task != "countdown":
            raise RuntimeError("The paper-runtime liveness anchor must be Countdown")
        splits, inputs = _load_ready_inputs(
            output_root,
            config,
            base_model_path=base_model_path,
        )
        return _cmd_canonical_cold_liveness(
            config,
            config_path,
            output_root,
            inputs=inputs["countdown"],
            splits=splits,
            base_model_path=base_model_path,
            force=force,
        )
    if rho is None:
        raise ValueError("Liveness rho is required outside cold-start")
    if rho not in _task_rhos(config, task):
        raise ValueError("Liveness rho must be one frozen grid point")
    splits, inputs = _load_ready_inputs(
        output_root,
        config,
        base_model_path=base_model_path,
    )
    lambda_value = None
    if _uses_task_lambdas(config):
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

    terminal_hash = sha256_file(adapter_weight_file(Path(result["terminal_adapter"])))
    if _is_coldstart(config):
        reference_hash = str(result["initialization_state_sha256"])
        changed = reference_hash != str(result["terminal_trainable_state_sha256"])
    else:
        reference_adapter = inputs[task].reference_adapter
        if reference_adapter is None:
            raise RuntimeError("Liveness reference adapter is missing")
        reference_hash = sha256_file(adapter_weight_file(reference_adapter))
        changed = reference_hash != terminal_hash
    if not changed:
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
            "initialization_trainable_state_sha256": result.get("initialization_state_sha256"),
            "terminal_trainable_state_sha256": result.get("terminal_trainable_state_sha256"),
        }
    )
    atomic_json(
        output_root / "liveness" / cell.key / "cell_manifest.json",
        result,
    )
    return result


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected one JSON object: {path}")
    return value


def _successful_attempt_matches_current_identity(
    config: Mapping[str, Any],
    workload_root: Path,
    *,
    source_commit: str,
    artifact_path: Path | None = None,
) -> bool:
    """Return whether live and packaged completed evidence matches this invocation."""

    expected_id = experiment_id(config)
    expected_hash = stable_config_hash(config)
    try:
        provenance = _read_json_object(workload_root / "source_provenance.json")
        prepare = _read_json_object(workload_root / "prepare_manifest.json")
        if not (
            provenance.get("source_commit") == source_commit
            and prepare.get("experiment_id") == expected_id
            and prepare.get("config_hash") == expected_hash
        ):
            return False
        if artifact_path is None:
            return True
        prefix = f"results/{expected_id}"
        with zipfile.ZipFile(artifact_path) as archive:
            artifact_manifest = json.loads(archive.read("ARTIFACT_MANIFEST.json"))
            base_commit = archive.read("BASE_COMMIT.txt").decode("utf-8").strip()
            run_manifest = json.loads(archive.read(f"{prefix}/run_manifest.json"))
            packaged_provenance = json.loads(
                archive.read(f"{prefix}/workload/source_provenance.json")
            )
            packaged_prepare = json.loads(
                archive.read(f"{prefix}/workload/prepare_manifest.json")
            )
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        UnicodeDecodeError,
        zipfile.BadZipFile,
    ):
        return False
    if not all(
        isinstance(value, dict)
        for value in (
            artifact_manifest,
            run_manifest,
            packaged_provenance,
            packaged_prepare,
        )
    ):
        return False
    return (
        artifact_manifest.get("package_kind") == "experiment-raw-complete"
        and artifact_manifest.get("experiment_id") == expected_id
        and artifact_manifest.get("base_commit") == source_commit
        and base_commit == source_commit
        and run_manifest.get("experiment_id") == expected_id
        and run_manifest.get("base_commit") == source_commit
        and packaged_provenance.get("source_commit") == source_commit
        and packaged_prepare.get("experiment_id") == expected_id
        and packaged_prepare.get("config_hash") == expected_hash
    )


def _effective_recovery_config(
    config: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    engineering_config = output_root / "engineering_self_test_config.yaml"
    if engineering_config.is_file():
        recovered = load_config(engineering_config)
        if not _is_engineering_self_test(recovered):
            raise RuntimeError("Recovery engineering config is not a placeholder config")
        return recovered
    return dict(config)


def _reusable_cell_manifests(
    config: Mapping[str, Any],
    output_root: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    reusable: dict[str, dict[str, Any]] = {}
    rejected: dict[str, str] = {}
    expected_hash = stable_config_hash(config)
    expected_id = experiment_id(config)
    for cell in build_cells(config):
        manifest_path = output_root / "cells" / cell.key / "cell_manifest.json"
        if not manifest_path.is_file():
            rejected[cell.key] = "missing_cell_manifest"
            continue
        try:
            value = _read_json_object(manifest_path)
            if (
                value.get("experiment_id") != expected_id
                or value.get("config_hash") != expected_hash
                or value.get("complete") is not True
                or value.get("evaluation_status") != "complete"
                or value.get("nan_inf_failure") is not False
            ):
                raise RuntimeError("identity_or_completion_fields_mismatch")
            if _is_engineering_self_test(config):
                if value.get("engineering_placeholder_backend") is not True:
                    raise RuntimeError("placeholder_backend_marker_missing")
            else:
                if (
                    not isinstance(value.get("identity_hash"), str)
                    or len(str(value["identity_hash"])) != 64
                    or value.get("canonical_dispatch_verified") is not True
                ):
                    raise RuntimeError("canonical_identity_or_dispatch_marker_missing")
                summary = Path(str(value.get("canonical_summary", "")))
                expected_summary_hash = str(value.get("canonical_summary_sha256", ""))
                if (
                    not summary.is_file()
                    or len(expected_summary_hash) != 64
                    or sha256_file(summary) != expected_summary_hash
                ):
                    raise RuntimeError("canonical_summary_missing_or_corrupt")
                for field in ("best_adapter", "terminal_adapter"):
                    adapter = Path(str(value.get(field, "")))
                    if not (adapter / "adapter_config.json").is_file() or not any(
                        (adapter / name).is_file()
                        for name in ("adapter_model.safetensors", "adapter_model.bin")
                    ):
                        raise RuntimeError(f"{field}_missing_or_incomplete")
            reusable[cell.key] = value
        except (OSError, ValueError, TypeError, RuntimeError, json.JSONDecodeError) as exc:
            rejected[cell.key] = f"{type(exc).__name__}: {exc}"
    return reusable, rejected


def _recovery_stage_plan(
    config: Mapping[str, Any],
    output_root: Path,
    *,
    base_model_path: str,
) -> dict[str, Any]:
    config = _effective_recovery_config(config, output_root)
    prepare_error: str | None = None
    calibration_error: str | None = None
    liveness_error: str | None = None
    try:
        _load_prepared(output_root, config)
        prepare_complete = True
    except Exception as exc:  # The plan records the exact fail-closed reason.
        prepare_complete = False
        prepare_error = f"{type(exc).__name__}: {exc}"
    if prepare_complete:
        try:
            _require_calibration_gate(config, output_root, base_model_path=base_model_path)
            calibration_complete = True
        except Exception as exc:
            calibration_complete = False
            calibration_error = f"{type(exc).__name__}: {exc}"
    else:
        calibration_complete = False
        calibration_error = "prepare_incomplete"
    if calibration_complete:
        try:
            _require_liveness_gate(config, output_root, base_model_path=base_model_path)
            liveness_complete = True
        except Exception as exc:
            liveness_complete = False
            liveness_error = f"{type(exc).__name__}: {exc}"
    else:
        liveness_complete = False
        liveness_error = "calibration_incomplete"
    reusable, rejected = _reusable_cell_manifests(config, output_root)
    expected_cells = len(build_cells(config))
    cells_complete = len(reusable) == expected_cells
    aggregate_path = output_root / "aggregate" / "aggregate_summary.json"
    aggregate_complete = False
    if aggregate_path.is_file():
        try:
            aggregate_complete = int(_read_json_object(aggregate_path).get("cell_count", 0)) == (
                expected_cells
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            aggregate_complete = False
    audit_path = output_root / "terminal_audit.json"
    audit_complete = False
    if audit_path.is_file():
        try:
            audit_complete = bool(
                _read_json_object(audit_path).get("all_training_and_evaluation_complete")
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            audit_complete = False
    finalized = False
    complete_path = output_root / "RUN_COMPLETE.json"
    if complete_path.is_file():
        try:
            finalized = bool(_read_json_object(complete_path).get("complete")) and audit_complete
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            finalized = False
    if not prepare_complete:
        next_stage = "prepare"
    elif not calibration_complete:
        next_stage = "calibrate"
    elif not liveness_complete:
        next_stage = "liveness"
    elif not cells_complete:
        next_stage = "run_queue"
    elif not aggregate_complete:
        next_stage = "aggregate"
    elif not audit_complete:
        next_stage = "audit"
    elif not finalized:
        next_stage = "finalize"
    else:
        next_stage = "delivery_preflight"
    return {
        "schema_version": RECOVERY_SNAPSHOT_SCHEMA_VERSION,
        "experiment_id": experiment_id(config),
        "config_hash": stable_config_hash(config),
        "output_root": str(output_root.resolve()),
        "prepare_complete": prepare_complete,
        "prepare_error": prepare_error,
        "calibration_complete": calibration_complete,
        "calibration_error": calibration_error,
        "liveness_complete": liveness_complete,
        "liveness_error": liveness_error,
        "expected_cells": expected_cells,
        "reusable_completed_cells": len(reusable),
        "reusable_cell_keys": sorted(reusable),
        "rejected_cells": rejected,
        "cells_complete": cells_complete,
        "aggregate_complete": aggregate_complete,
        "audit_complete": audit_complete,
        "finalized": finalized,
        "next_stage": next_stage,
        "intra_cell_resume_supported": False,
        "intra_cell_resume_reason": (
            "locked canonical kernels do not persist complete optimizer, scheduler, RNG, "
            "and dataloader state"
        ),
    }


def cmd_recovery_plan(
    config: Mapping[str, Any],
    output_root: Path,
    *,
    base_model_path: str,
) -> dict[str, Any]:
    plan = _recovery_stage_plan(config, output_root, base_model_path=base_model_path)
    atomic_json(output_root / "recovery" / "RECOVERY_PLAN.json", plan)
    return plan


def _hardlink_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError as exc:
        raise RuntimeError(
            "Recovery requires source and destination on one hard-link-capable persistent "
            f"filesystem; could not link {source} -> {destination}: {exc}"
        ) from exc


def _replace_path_prefix(value: Any, source: str, destination: str) -> Any:
    if isinstance(value, str) and (value == source or value.startswith(source + os.sep)):
        return destination + value[len(source) :]
    if isinstance(value, list):
        return [_replace_path_prefix(item, source, destination) for item in value]
    if isinstance(value, dict):
        return {key: _replace_path_prefix(item, source, destination) for key, item in value.items()}
    return value


def cmd_import_recovery(
    config: Mapping[str, Any],
    output_root: Path,
    *,
    source_output_root: Path,
    base_model_path: str,
    source_commit: str,
) -> dict[str, Any]:
    source_output_root = source_output_root.resolve()
    output_root = output_root.resolve()
    if source_output_root == output_root:
        raise ValueError("Recovery source and destination must differ")
    if not source_output_root.is_dir():
        raise FileNotFoundError(f"Recovery source does not exist: {source_output_root}")
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError("Recovery destination must be new and empty")
    if len(source_commit) != 40 or any(
        character not in "0123456789abcdef" for character in source_commit
    ):
        raise ValueError("Recovery import requires one full lowercase source commit")
    provenance = _read_json_object(source_output_root / "source_provenance.json")
    if provenance.get("source_commit") != source_commit:
        raise RuntimeError("Recovery source commit does not match the reviewed execution commit")
    output_root.mkdir(parents=True, exist_ok=True)
    effective = _effective_recovery_config(config, source_output_root)
    source_plan = _recovery_stage_plan(
        effective,
        source_output_root,
        base_model_path=base_model_path,
    )
    reusable = set(source_plan["reusable_cell_keys"]) if source_plan["prepare_complete"] else set()
    source_text = str(source_output_root)
    destination_text = str(output_root)
    linked_files = 0
    linked_bytes = 0
    source_cell_hashes = {
        key: sha256_file(source_output_root / "cells" / key / "cell_manifest.json")
        for key in reusable
    }
    if source_plan["prepare_complete"]:
        recovery_label = source_output_root.parent.name

        def mapped_relative(relative: Path) -> Path | None:
            if relative.parts[0] in RECOVERY_TRANSIENT_TOP_LEVEL:
                return None
            if len(relative.parts) == 1 and relative.name in RECOVERY_TRANSIENT_FILES:
                return None
            if relative.parts[0] == "cells" and (
                len(relative.parts) < 2 or relative.parts[1] not in reusable
            ):
                return None
            if relative.parts[0] == "liveness" and not source_plan["liveness_complete"]:
                return None
            if relative.parts[0] == "logs":
                return Path("logs") / f"recovered_{recovery_label}" / Path(*relative.parts[1:])
            return relative

        for source in sorted(path for path in source_output_root.rglob("*") if path.is_dir()):
            if source.is_symlink():
                raise RuntimeError(f"Recovery refuses symbolic links: {source}")
            relative = mapped_relative(source.relative_to(source_output_root))
            if relative is not None:
                (output_root / relative).mkdir(parents=True, exist_ok=True)
        for source in sorted(source_output_root.rglob("*")):
            if source.is_symlink():
                raise RuntimeError(f"Recovery refuses symbolic links: {source}")
            if not source.is_file():
                continue
            relative = mapped_relative(source.relative_to(source_output_root))
            if relative is None:
                continue
            destination = output_root / relative
            _hardlink_file(source, destination)
            linked_files += 1
            linked_bytes += source.stat().st_size
        path_manifest_targets = (
            output_root / "prepare_manifest.json",
            output_root / "split_manifest.json",
            output_root / "source_provenance.json",
        )
        for path in path_manifest_targets:
            if not path.is_file():
                continue
            try:
                value = _read_json_object(path)
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
            updated = _replace_path_prefix(value, source_text, destination_text)
            if updated != value:
                atomic_json(path, updated)
        for key, source_hash in sorted(source_cell_hashes.items()):
            manifest_path = output_root / "cells" / key / "cell_manifest.json"
            value = _read_json_object(manifest_path)
            value = _replace_path_prefix(value, source_text, destination_text)
            value["recovery_provenance"] = {
                "source_output_root": source_text,
                "source_manifest_sha256": source_hash,
                "import_mode": "identity_checked_hardlink",
                "scientific_variables_changed": False,
            }
            atomic_json(manifest_path, value)
    imported_cell_manifests = [
        {
            "cell_key": key,
            "source_manifest_sha256": source_cell_hashes[key],
            "imported_manifest_sha256": sha256_file(
                output_root / "cells" / key / "cell_manifest.json"
            ),
        }
        for key in sorted(reusable)
    ]
    import_manifest = {
        "schema_version": RECOVERY_SNAPSHOT_SCHEMA_VERSION,
        "experiment_id": experiment_id(effective),
        "source_commit": source_commit,
        "source_output_root": source_text,
        "destination_output_root": destination_text,
        "source_plan": source_plan,
        "imported_reusable_cells": sorted(reusable),
        "imported_cell_manifests": imported_cell_manifests,
        "linked_files": linked_files,
        "linked_bytes": linked_bytes,
        "copy_mode": "hardlink_read_only_then_copy_on_atomic_json_rewrite",
        "incomplete_cells_imported": False,
        "scientific_variables_changed": False,
        "complete": True,
    }
    atomic_json(output_root / "recovery" / "IMPORT_MANIFEST.json", import_manifest)
    return import_manifest


def _recovery_checkpoint_snapshot(
    config: Mapping[str, Any],
    output_root: Path,
    snapshot_root: Path,
    *,
    source_commit: str,
) -> dict[str, Any]:
    reusable, rejected = _reusable_cell_manifests(config, output_root)
    temporary = snapshot_root.with_name(f".{snapshot_root.name}.tmp-{os.getpid()}-{time.time_ns()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    (temporary / "logs").mkdir(parents=True)
    (temporary / "cell_manifests").mkdir(parents=True)
    cells: list[dict[str, Any]] = []
    for key, manifest in sorted(reusable.items()):
        source = output_root / "cells" / key / "cell_manifest.json"
        destination = temporary / "cell_manifests" / f"{key}.json"
        shutil.copy2(source, destination)
        cells.append(
            {
                "cell_key": key,
                "manifest_sha256": sha256_file(source),
                "manifest_path": str(source.resolve()),
                "canonical_output": manifest.get("canonical_output"),
                "terminal_adapter": manifest.get("terminal_adapter"),
            }
        )
    payload = {
        "schema_version": RECOVERY_SNAPSHOT_SCHEMA_VERSION,
        "experiment_id": experiment_id(config),
        "base_commit": source_commit,
        "config_hash": stable_config_hash(config),
        "output_root": str(output_root.resolve()),
        "expected_cells": len(build_cells(config)),
        "completed_cells": len(cells),
        "cells": cells,
        "rejected_cells": rejected,
        "recovery_semantics": "reuse complete identity-checked cells; rerun incomplete cells",
        "intra_cell_resume_supported": False,
        "scientific_status": "not_run" if _is_engineering_self_test(config) else "pilot",
    }
    atomic_json(temporary / "RECOVERY_SNAPSHOT.json", payload)
    atomic_json(
        temporary / "run_manifest.json",
        {
            "schema_version": 1,
            "experiment_id": experiment_id(config),
            "base_commit": source_commit,
            "run_id": output_root.parent.name,
            "execution_state": "checkpoint",
            "artifact_state": "checkpoint",
            "completed_cells": len(cells),
            "expected_cells": len(build_cells(config)),
            "scientific_status": payload["scientific_status"],
        },
    )
    (temporary / "logs" / "recovery_checkpoint.log").write_text(
        f"completed_cells={len(cells)} expected_cells={len(build_cells(config))}\n",
        encoding="utf-8",
    )
    if snapshot_root.exists():
        shutil.rmtree(snapshot_root)
    os.replace(temporary, snapshot_root)
    return payload


def _publish_recovery_checkpoint(
    config: Mapping[str, Any],
    output_root: Path,
    *,
    package_output: Path,
) -> dict[str, Any]:
    provenance = _read_json_object(output_root / "source_provenance.json")
    source_commit = str(provenance.get("source_commit", ""))
    if len(source_commit) != 40 or any(char not in "0123456789abcdef" for char in source_commit):
        raise RuntimeError("Recovery checkpoint requires a full source commit")
    snapshot_root = package_output.parent / "snapshot"
    payload = _recovery_checkpoint_snapshot(
        config,
        output_root,
        snapshot_root,
        source_commit=source_commit,
    )
    command = [
        sys.executable,
        str(_repo_root() / "scripts" / "package_experiment_hardened.py"),
        "--repo-root",
        str(_repo_root()),
        "--experiment-id",
        experiment_id(config),
        "--package-kind",
        "experiment-checkpoint",
        "--result-dir",
        str(snapshot_root),
        "--output",
        str(package_output),
        "--base-commit",
        source_commit,
        "--no-repository-changes",
        "--large-file-persistence",
        "persistent_local",
        "--source-file",
        "scripts/run_e8_multitask_exp_coldstart.sh",
        "--source-file",
        "src/drpo/e8_multitask_exp_tuning.py",
    ]
    if os.environ.get("E8_COLDSTART_RECOVERY_REQUIRE_ORIGIN_MAIN") == "1":
        command.append("--require-origin-main-match")
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            "Recovery checkpoint packaging failed: "
            + (completed.stderr.strip() or completed.stdout.strip())
        )
    mirror_value = os.environ.get("E8_COLDSTART_RECOVERY_MIRROR", "").strip()
    mirror_path: Path | None = None
    if mirror_value:
        mirror_root = Path(mirror_value).resolve()
        mirror_root.mkdir(parents=True, exist_ok=True)
        mirror_path = mirror_root / package_output.name
        temporary = mirror_path.with_name(f".{mirror_path.name}.tmp-{os.getpid()}")
        shutil.copy2(package_output, temporary)
        if sha256_file(temporary) != sha256_file(package_output):
            temporary.unlink(missing_ok=True)
            raise RuntimeError("Recovery mirror copy failed checksum verification")
        os.replace(temporary, mirror_path)
    status = {
        **payload,
        "package": str(package_output.resolve()),
        "package_sha256": sha256_file(package_output),
        "mirror": str(mirror_path) if mirror_path else None,
        "mirror_configured": mirror_path is not None,
        "complete": True,
    }
    atomic_json(package_output.parent / "RECOVERY_CHECKPOINT_STATUS.json", status)
    return status


def cmd_compact_logs(config: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    logs_root = output_root / "logs"
    archive_root = output_root / "persistent_raw_archives"
    archive = archive_root / "cell_and_stage_logs.tar.gz"
    index_path = logs_root / "LOG_ARCHIVE_INDEX.json"
    prepared_path = logs_root / "LOG_ARCHIVE_PREPARED.json"
    if index_path.is_file() and archive.is_file():
        value = _read_json_object(index_path)
        if value.get("archive_sha256") == sha256_file(archive) and value.get("complete"):
            return value
    if prepared_path.is_file():
        prepared = _read_json_object(prepared_path)
        if (
            not archive.is_file()
            or prepared.get("archive_sha256") != sha256_file(archive)
            or prepared.get("prepared") is not True
        ):
            raise RuntimeError("Prepared log archive transaction is missing or corrupt")
        rows = list(prepared.get("members", ()))
        if not rows or not all(isinstance(row, dict) for row in rows):
            raise RuntimeError("Prepared log archive inventory is empty or invalid")
    else:
        log_files = [
            path
            for path in sorted(logs_root.rglob("*"))
            if path.is_file()
            and path not in {index_path, prepared_path}
            and "tails" not in path.relative_to(logs_root).parts
            and not path.is_symlink()
        ]
        if not log_files:
            raise RuntimeError("No logs are available for transactional compaction")
        archive_root.mkdir(parents=True, exist_ok=True)
        temporary = archive.with_name(f".{archive.name}.tmp-{os.getpid()}")
        rows = []
        with tarfile.open(temporary, "w:gz") as handle:
            for path in log_files:
                relative = path.relative_to(logs_root)
                rows.append(
                    {
                        "path": relative.as_posix(),
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
                handle.add(path, arcname=relative.as_posix(), recursive=False)
        os.replace(temporary, archive)
        with tarfile.open(archive, "r:gz") as handle:
            members = {member.name: member for member in handle.getmembers()}
            for row in rows:
                name = str(row["path"])
                member = members.get(name)
                extracted = handle.extractfile(member) if member is not None else None
                if member is None or not member.isfile() or extracted is None:
                    raise RuntimeError(f"Log archive member is missing or invalid: {name}")
                digest = hashlib.sha256()
                size = 0
                while chunk := extracted.read(1024 * 1024):
                    digest.update(chunk)
                    size += len(chunk)
                if size != row["size_bytes"] or digest.hexdigest() != row["sha256"]:
                    raise RuntimeError(f"Log archive member verification failed: {name}")
        tails_root = logs_root / "tails"
        for path in log_files:
            relative = path.relative_to(logs_root)
            tail = tails_root / relative
            tail.parent.mkdir(parents=True, exist_ok=True)
            with path.open("rb") as source:
                size = path.stat().st_size
                if size > 65536:
                    source.seek(-65536, os.SEEK_END)
                tail.write_bytes(source.read())
        prepared = {
            "schema_version": 1,
            "experiment_id": experiment_id(config),
            "archive": str(archive.resolve()),
            "archive_sha256": sha256_file(archive),
            "members": rows,
            "prepared": True,
        }
        atomic_json(prepared_path, prepared)
    for row in rows:
        relative = Path(str(row.get("path", "")))
        if (
            not relative.parts
            or relative.is_absolute()
            or ".." in relative.parts
            or "tails" in relative.parts
        ):
            raise RuntimeError(f"Unsafe prepared log path: {relative}")
        path = logs_root / relative
        tail = logs_root / "tails" / relative
        if not tail.is_file():
            raise RuntimeError(f"Prepared log tail is missing: {tail}")
        if path.is_file():
            if path.stat().st_size != int(row.get("size_bytes", -1)) or sha256_file(
                path
            ) != row.get("sha256"):
                raise RuntimeError(f"Log changed during compaction transaction: {path}")
            path.unlink()
    value = {
        "schema_version": 1,
        "experiment_id": experiment_id(config),
        "archive": str(archive.resolve()),
        "archive_size_bytes": archive.stat().st_size,
        "archive_sha256": sha256_file(archive),
        "members": rows,
        "tail_bytes_per_log": 65536,
        "raw_logs_persist_locally": True,
        "transactionally_resumable": True,
        "complete": True,
    }
    atomic_json(index_path, value)
    prepared_path.unlink(missing_ok=True)
    return value


def _run_subprocess_cell(
    *,
    config_path: Path,
    output_root: Path,
    base_model_path: str,
    cell: Cell,
    gpu_id: int,
    force: bool,
) -> dict[str, Any]:
    started_at = time.time()
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
        "started_unix": started_at,
        "finished_unix": time.time(),
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
        expected_identity = (
            _canonical_calibration_identity(
                task,
                split_manifest=splits,
                base_model_path=base_model_path,
                config=config,
            )
            if _is_coldstart(config)
            else _calibration_identity(
                task,
                inputs=inputs[task],
                split_manifest=splits,
                base_model_path=base_model_path,
                config=config,
            )
        )
        if result.get("identity_hash") != expected_identity["identity_hash"] or not result.get(
            "complete"
        ):
            raise RuntimeError(f"Calibration gate identity mismatch for {task}")
        if _is_coldstart(config) and (
            result.get("enabled") is not False
            or result.get("mode") != "paper_linear_surprisal_no_calibration"
        ):
            raise RuntimeError(f"Paper calibration must remain disabled for {task}")


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
        common = (
            result.get("experiment_id") == experiment_id(config)
            and result.get("config_hash") == stable_config_hash(config)
            and result.get("base_model_identity") == base_identity
            and result.get("engineering_liveness") is True
            and int(result.get("optimizer_updates", 0)) == 2
            and result.get("complete") is True
            and result.get("reload_gate_passed") is True
            and result.get("adapter_weight_changed") is True
            and result.get("fresh_process_reload_passed") is True
            and result.get("reload_process_id") != result.get("liveness_parent_process_id")
            and result.get("nan_inf_failure") is False
        )
        canonical_cold = (
            _is_coldstart(config)
            and result.get("canonical_dispatch_verified") is True
            and result.get("finite_old_core_updates") is True
            and math.isfinite(float(result.get("optimizer_update_norm", 0.0)))
            and float(result.get("optimizer_update_norm", 0.0)) > 0.0
        )
        legacy = (
            not _is_coldstart(config)
            and result.get("positive_loss_finite_nonzero") is True
            and result.get("repulsive_scalar_finite_nonzero") is True
            and result.get("raw_gradient_finite_nonzero") is True
            and result.get("reference_adapter_weight_sha256")
            != result.get("terminal_adapter_weight_sha256")
        )
        if common and (canonical_cold or legacy):
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
    if _is_coldstart(config):
        raise RuntimeError("Cold-start has no wave barriers; use run-all dynamic scheduling")
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


def _countdown_protocol_diagnostic(
    config: Mapping[str, Any],
    output_root: Path,
    *,
    destination: Path | None = None,
) -> dict[str, Any]:
    """Audit Countdown implementation identity without gating any scientific outcome."""

    countdown_cells = [cell for cell in build_cells(config) if cell.task == "countdown"]
    if _is_engineering_self_test(config):
        diagnostic = {
            "schema_version": 1,
            "experiment_id": experiment_id(config),
            "status": "NOT_RUN_ENGINEERING",
            "countdown_cells": len(countdown_cells),
            "result_gate": False,
            "controls_task_transfer_release": False,
            "scientific_evidence": False,
        }
    else:
        expected_sources = dict(config["canonical_coldstart"]["expected_git_blob_shas"])
        identity_failures: list[str] = []
        for cell in countdown_cells:
            path = output_root / "cells" / cell.key / "cell_manifest.json"
            if not path.is_file():
                identity_failures.append(f"{cell.key}:missing")
                continue
            value = json.loads(path.read_text(encoding="utf-8"))
            if (
                value.get("complete") is not True
                or value.get("evaluation_status") != "complete"
                or value.get("nan_inf_failure") is not False
                or value.get("countdown_protocol_exact") is not True
                or value.get("canonical_dispatch")
                != "countdown_e8_alpha1_highc_scan_runtime.worker"
                or value.get("canonical_source_git_blob_shas") != expected_sources
                or int(value.get("terminal_step", -1)) != 1200
                or value.get("stop_reason") != "max_steps"
            ):
                identity_failures.append(f"{cell.key}:protocol_identity")
        countdown_not_run = not countdown_cells
        diagnostic = {
            "schema_version": 1,
            "experiment_id": experiment_id(config),
            "status": "NOT_RUN"
            if countdown_not_run
            else ("PASS" if not identity_failures else "FAIL"),
            "countdown_cells": len(countdown_cells),
            "identity_failures": identity_failures,
            "result_gate": False,
            "controls_task_transfer_release": False,
            "scientific_evidence": not countdown_not_run,
        }
    if destination is not None:
        atomic_json(destination, diagnostic)
    return diagnostic


def cmd_run_dynamic(
    config: Mapping[str, Any],
    config_path: Path,
    output_root: Path,
    *,
    base_model_path: str,
    force: bool,
    retry_incomplete: bool,
) -> dict[str, Any]:
    """Run one shared recovery-aware queue on 16 fixed GPU slots, without batch barriers."""

    if not _is_coldstart(config):
        raise RuntimeError("Dynamic scheduling is frozen for the cold-start profile only")
    _require_calibration_gate(config, output_root, base_model_path=base_model_path)
    _require_liveness_gate(config, output_root, base_model_path=base_model_path)
    cells = build_cells(config)
    gpu_ids = tuple(int(value) for value in config["execution"]["gpu_ids"])
    slots_per_gpu = int(config["execution"]["slots_per_gpu"])
    slot_count = len(gpu_ids) * slots_per_gpu
    if slot_count != int(config["execution"]["max_concurrent_cells"]) or slot_count != 16:
        raise RuntimeError("Declared 16-slot capacity is internally inconsistent")

    pending: queue.Queue[Cell] = queue.Queue()
    for cell in cells:
        pending.put(cell)
    stop = threading.Event()
    lock = threading.Lock()
    checkpoint_lock = threading.Lock()
    task_result_lock = threading.Lock()
    results: list[dict[str, Any]] = []
    task_results: dict[str, dict[str, Any]] = {}
    event_path = output_root / "scheduler" / "queue_events.jsonl"
    event_path.parent.mkdir(parents=True, exist_ok=True)
    scheduler_run_id = f"queue-{int(time.time())}-{os.getpid()}"
    recovery_package_value = os.environ.get("E8_COLDSTART_RECOVERY_PACKAGE", "").strip()
    recovery_package = Path(recovery_package_value).resolve() if recovery_package_value else None
    recovery_interval = int(os.environ.get("E8_COLDSTART_RECOVERY_INTERVAL_CELLS", "5"))
    if recovery_interval <= 0:
        raise ValueError("E8_COLDSTART_RECOVERY_INTERVAL_CELLS must be positive")
    initially_reusable, _ = _reusable_cell_manifests(config, output_root)
    last_checkpoint_count = (len(initially_reusable) // recovery_interval) * recovery_interval

    def record(event: Mapping[str, Any]) -> None:
        with lock:
            append_jsonl(event_path, {"scheduler_run_id": scheduler_run_id, **dict(event)})

    def publish_completed_task(task: str) -> None:
        with task_result_lock:
            if task in task_results:
                return
            rows = _coldstart_completed_task_rows(config, output_root, task)
            if rows is not None:
                task_results[task] = _write_coldstart_task_result(
                    config,
                    output_root,
                    task,
                    rows,
                )

    def worker(slot: int, gpu_id: int) -> list[dict[str, Any]]:
        nonlocal last_checkpoint_count
        local: list[dict[str, Any]] = []
        while not stop.is_set():
            try:
                cell = pending.get_nowait()
            except queue.Empty:
                break
            cell_root = output_root / "cells" / cell.key
            manifest_path = cell_root / "cell_manifest.json"
            reusable_complete = False
            if manifest_path.is_file():
                try:
                    reusable_complete = bool(
                        json.loads(manifest_path.read_text(encoding="utf-8")).get("complete")
                    )
                except (OSError, json.JSONDecodeError):
                    reusable_complete = False
            child_force = force or (
                retry_incomplete and cell_root.exists() and not reusable_complete
            )
            record(
                {
                    "event": "start",
                    "cell_key": cell.key,
                    "slot": slot,
                    "gpu_id": gpu_id,
                    "unix_time": time.time(),
                    "retry_incomplete": child_force and not force,
                }
            )
            result = _run_subprocess_cell(
                config_path=config_path.resolve(),
                output_root=output_root.resolve(),
                base_model_path=base_model_path,
                cell=cell,
                gpu_id=gpu_id,
                force=child_force,
            )
            if int(result["returncode"]) == 0:
                try:
                    completed_manifest = _read_json_object(manifest_path)
                    if (
                        completed_manifest.get("complete") is not True
                        or completed_manifest.get("evaluation_status") != "complete"
                        or completed_manifest.get("nan_inf_failure") is not False
                    ):
                        raise RuntimeError("child returned zero without a complete finite cell")
                except (
                    OSError,
                    ValueError,
                    TypeError,
                    RuntimeError,
                    json.JSONDecodeError,
                ) as exc:
                    result["returncode"] = 75
                    result["cell_completion_error"] = f"{type(exc).__name__}: {exc}"
            if int(result["returncode"]) == 0 and recovery_package is not None:
                try:
                    with checkpoint_lock:
                        current_reusable, _ = _reusable_cell_manifests(config, output_root)
                        completed_count = len(current_reusable)
                        if completed_count >= last_checkpoint_count + recovery_interval:
                            checkpoint = _publish_recovery_checkpoint(
                                config,
                                output_root,
                                package_output=recovery_package,
                            )
                            last_checkpoint_count = int(checkpoint["completed_cells"])
                            result["recovery_checkpoint"] = checkpoint["package"]
                            result["recovery_checkpoint_completed_cells"] = last_checkpoint_count
                except Exception as exc:
                    result["returncode"] = 74
                    result["recovery_checkpoint_error"] = f"{type(exc).__name__}: {exc}"
            result.update({"slot": slot, "nominal_batch": cells.index(cell) // slot_count + 1})
            local.append(result)
            record({"event": "finish", **result, "unix_time": time.time()})
            pending.task_done()
            if int(result["returncode"]) != 0:
                stop.set()
            else:
                try:
                    publish_completed_task(cell.task)
                except Exception:
                    stop.set()
                    raise
        return local

    with ThreadPoolExecutor(max_workers=slot_count) as executor:
        futures = [
            executor.submit(worker, slot, gpu_ids[slot % len(gpu_ids)])
            for slot in range(slot_count)
        ]
        for future in as_completed(futures):
            results.extend(future.result())

    results.sort(key=lambda row: str(row["cell_key"]))
    failures = [row for row in results if int(row["returncode"]) != 0]
    returned_keys = {str(row["cell_key"]) for row in results}
    completed_keys = {str(row["cell_key"]) for row in results if int(row["returncode"]) == 0}
    unscheduled = [cell.key for cell in cells if cell.key not in returned_keys]
    protocol_diagnostic = (
        _countdown_protocol_diagnostic(
            config,
            output_root,
            destination=output_root / "scheduler" / "countdown_protocol_diagnostic.json",
        )
        if not failures and not unscheduled
        else {
            "status": "PENDING",
            "result_gate": False,
            "controls_task_transfer_release": False,
        }
    )
    manifest = {
        "schema_version": 1,
        "experiment_id": experiment_id(config),
        "scheduler": "dynamic_slot_queue",
        "scheduler_run_id": scheduler_run_id,
        "wave_barriers": False,
        "wave_count": len(build_waves(config)),
        "wave_count_role": "nominal_audit_geometry_only_not_scheduling_barrier",
        "slot_count": slot_count,
        "gpu_ids": list(gpu_ids),
        "slots_per_gpu": slots_per_gpu,
        "countdown_protocol_diagnostic": protocol_diagnostic,
        "countdown_result_controls_transfer_release": False,
        "expected_cells": len(cells),
        "completed_cells": len(completed_keys),
        "results": results,
        "failed_cells": [row["cell_key"] for row in failures],
        "unscheduled_cells": unscheduled,
        "queue_events": str(event_path.resolve()),
        "analysis_ready_tasks": sorted(task_results),
        "task_results": task_results,
        "complete": not failures and not unscheduled and len(completed_keys) == len(cells),
        "scientific_status": "not_run" if _is_engineering_self_test(config) else "pilot",
        "engineering_placeholder_backend": _is_engineering_self_test(config),
    }
    atomic_json(output_root / "scheduler" / "dynamic_run.json", manifest)
    if failures or unscheduled:
        raise RuntimeError(
            "Cold-start scheduling stopped fail-closed; "
            f"failed={manifest['failed_cells']} unscheduled={len(unscheduled)}"
        )
    return manifest


def cmd_run_all(
    config: Mapping[str, Any],
    config_path: Path,
    output_root: Path,
    *,
    base_model_path: str,
    force: bool,
    retry_incomplete: bool = False,
) -> dict[str, Any]:
    if _is_coldstart(config):
        return cmd_run_dynamic(
            config,
            config_path,
            output_root,
            base_model_path=base_model_path,
            force=force,
            retry_incomplete=retry_incomplete,
        )
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
        fieldnames = list(rows[0])
        for row in rows[1:]:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _coldstart_completed_task_rows(
    config: Mapping[str, Any],
    output_root: Path,
    task: str,
) -> list[dict[str, Any]] | None:
    """Return task-local response rows once every frozen cell for the task is complete."""

    if not _is_coldstart(config):
        raise RuntimeError("Per-task early result materialization is cold-start only")
    expected = [cell for cell in build_cells(config) if cell.task == task]
    if not expected:
        return None
    expected_hash = stable_config_hash(config)
    rows: list[dict[str, Any]] = []
    for cell in expected:
        path = output_root / "cells" / cell.key / "cell_manifest.json"
        if not path.is_file():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("complete") is not True or value.get("evaluation_status") != "complete":
            return None
        if (
            value.get("experiment_id") != experiment_id(config)
            or value.get("config_hash") != expected_hash
        ):
            raise RuntimeError(f"{cell.key} per-task result identity mismatch")
        if not _is_engineering_self_test(config) and (
            "validation_late_window_pass8_mean" not in value
            or "validation_late_window_greedy_mean" not in value
        ):
            raise RuntimeError(f"{cell.key} is missing the paper primary late-window metric")
        rows.append(
            {
                "source": "current",
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
                "cell_key": cell.key,
                "nan_inf_failure": bool(value["nan_inf_failure"]),
                "late_window_pass8_mean": value.get(
                    "validation_late_window_pass8_mean",
                    value["validation_best_pass8"],
                ),
                "late_window_greedy_mean": value.get(
                    "validation_late_window_greedy_mean",
                    value["validation_best_greedy"],
                ),
                "best_pass8": value["validation_best_pass8"],
                "terminal_pass8": value["validation_terminal_pass8"],
                "best_greedy": value["validation_best_greedy"],
                "terminal_greedy": value["validation_terminal_greedy"],
                "best_greedy_valid_rate": value["validation_best_greedy_valid_rate"],
                "terminal_greedy_valid_rate": value["validation_terminal_greedy_valid_rate"],
                "best_step": value["best_step"],
                "terminal_step": value["terminal_step"],
                "stop_reason": value["stop_reason"],
            }
        )
    return rows


def _write_coldstart_task_result(
    config: Mapping[str, Any],
    output_root: Path,
    task: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Publish deterministic task-local CSVs and write TASK_COMPLETE.json last."""

    provenance_path = output_root / "source_provenance.json"
    if not provenance_path.is_file():
        raise RuntimeError("Per-task result materialization requires source_provenance.json")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    source_commit = str(provenance.get("source_commit", ""))
    if len(source_commit) != 40 or any(
        character not in "0123456789abcdef" for character in source_commit
    ):
        raise RuntimeError("Per-task result materialization requires one full source commit")
    run_id = str(provenance.get("run_id", output_root.name))
    root = output_root / "task_results" / task
    root.mkdir(parents=True, exist_ok=True)
    marker_path = root / "TASK_COMPLETE.json"
    marker_path.unlink(missing_ok=True)

    all_cells_path = root / "all_cells.csv"
    plot_path = root / "plot_curve_points.csv"
    _write_csv(all_cells_path, rows)
    plot_rows = [
        {
            "experiment_id": experiment_id(config),
            "run_id": run_id,
            "source_commit": source_commit,
            "task": row["task"],
            "method": row["method"],
            "lambda": row["lambda"],
            "rho": row["rho"],
            "seed": row["seed"],
            "stage": row["stage"],
            "late_window_pass8_mean": row["late_window_pass8_mean"],
            "late_window_greedy_mean": row["late_window_greedy_mean"],
            "best_validation_pass8": row["best_pass8"],
            "terminal_pass8": row["terminal_pass8"],
            "best_validation_greedy": row["best_greedy"],
            "terminal_greedy": row["terminal_greedy"],
            "best_greedy_valid_rate": row["best_greedy_valid_rate"],
            "terminal_greedy_valid_rate": row["terminal_greedy_valid_rate"],
            "best_step": row["best_step"],
            "terminal_step": row["terminal_step"],
            "stop_reason": row["stop_reason"],
            "nan_inf_failure": row["nan_inf_failure"],
            "complete": True,
        }
        for row in rows
    ]
    _write_csv(plot_path, plot_rows)
    cell_manifest_sha256 = {
        row["cell_key"]: sha256_file(
            output_root / "cells" / str(row["cell_key"]) / "cell_manifest.json"
        )
        for row in rows
    }
    marker = {
        "schema_version": 1,
        "experiment_id": experiment_id(config),
        "config_hash": stable_config_hash(config),
        "run_id": run_id,
        "source_commit": source_commit,
        "task": task,
        "expected_cells": len(rows),
        "cell_count": len(rows),
        "all_cells_csv": f"task_results/{task}/all_cells.csv",
        "all_cells_csv_sha256": sha256_file(all_cells_path),
        "plot_curve_points_csv": f"task_results/{task}/plot_curve_points.csv",
        "plot_curve_points_csv_sha256": sha256_file(plot_path),
        "cell_manifest_sha256": cell_manifest_sha256,
        "analysis_ready": True,
        "final_aggregate_authority": False,
        "test_partition_accessed": False,
        "method_ranking_allowed": False,
        "complete": True,
        "scientific_status": ("not_run" if _is_engineering_self_test(config) else "pilot"),
        "note": (
            "Deterministic early task snapshot from completed identity-matched cells; "
            "the terminal aggregate remains the final reporting authority."
        ),
    }
    atomic_json(marker_path, marker)
    return marker


def _materialize_completed_coldstart_task_results(
    config: Mapping[str, Any],
    output_root: Path,
) -> dict[str, dict[str, Any]]:
    """Publish every fully complete task independently of nominal-batch boundaries."""

    ready: dict[str, dict[str, Any]] = {}
    for task_value in config["suite"]["tasks"]:
        task = str(task_value)
        rows = _coldstart_completed_task_rows(config, output_root, task)
        if rows is None:
            continue
        ready[task] = _write_coldstart_task_result(config, output_root, task, rows)
    return ready


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


def _aggregate_coldstart(
    config: Mapping[str, Any],
    output_root: Path,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    provenance_path = output_root / "source_provenance.json"
    provenance = (
        json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.is_file() else {}
    )
    run_id = str(provenance.get("run_id", output_root.name))
    source_commit = str(provenance.get("source_commit", "unrecorded"))
    plot_rows: list[dict[str, Any]] = []
    for row in rows:
        plot_rows.append(
            {
                "experiment_id": experiment_id(config),
                "run_id": run_id,
                "source_commit": source_commit,
                "task": row["task"],
                "method": row["method"],
                "lambda": row["lambda"],
                "rho": row["rho"],
                "seed": row["seed"],
                "stage": row["stage"],
                "late_window_pass8_mean": row["late_window_pass8_mean"],
                "late_window_greedy_mean": row["late_window_greedy_mean"],
                "best_validation_pass8": row["best_pass8"],
                "terminal_pass8": row["terminal_pass8"],
                "best_validation_greedy": row["best_greedy"],
                "terminal_greedy": row["terminal_greedy"],
                "best_greedy_valid_rate": row["best_greedy_valid_rate"],
                "terminal_greedy_valid_rate": row["terminal_greedy_valid_rate"],
                "best_step": row["best_step"],
                "terminal_step": row["terminal_step"],
                "stop_reason": row["stop_reason"],
                "nan_inf_failure": row["nan_inf_failure"],
                "complete": True,
            }
        )
    _write_csv(output_root / "aggregate" / "plot_curve_points.csv", plot_rows)

    summaries: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    for task_value in config["suite"]["tasks"]:
        task = str(task_value)
        task_rows = [row for row in rows if row["task"] == task]
        positive_rows = [row for row in task_rows if row["method"] == METHOD_POSITIVE_ONLY]
        global_rows = [row for row in task_rows if row["method"] == METHOD_GLOBAL]
        exp_rows = [row for row in task_rows if row["method"] == METHOD_EXPONENTIAL]
        configured_methods = [cell.method for cell in build_cells(config) if cell.task == task]
        if not configured_methods:
            continue
        expected_counts = tuple(
            configured_methods.count(method)
            for method in (METHOD_POSITIVE_ONLY, METHOD_GLOBAL, METHOD_EXPONENTIAL)
        )
        if (len(positive_rows), len(global_rows), len(exp_rows)) != expected_counts:
            raise RuntimeError(f"{task} cold-start cell counts differ from {expected_counts}")

        def aggregate_group(group: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
            first = group[0]
            return {
                "task": task,
                "method": first["method"],
                "lambda": first["lambda"],
                "rho": first["rho"],
                "seeds": sorted(int(row["seed"]) for row in group),
                "late_window_pass8_mean": float(
                    np.mean([float(row["late_window_pass8_mean"]) for row in group])
                ),
                "late_window_greedy_mean": float(
                    np.mean([float(row["late_window_greedy_mean"]) for row in group])
                ),
                "terminal_pass8_mean": float(
                    np.mean([float(row["terminal_pass8"]) for row in group])
                ),
                "terminal_greedy_valid_rate_mean": float(
                    np.mean([float(row["terminal_greedy_valid_rate"]) for row in group])
                ),
                "best_pass8_mean": float(np.mean([float(row["best_pass8"]) for row in group])),
                "nan_inf_failure": any(bool(row["nan_inf_failure"]) for row in group),
            }

        coefficient_groups: dict[tuple[str, float | None], list[dict[str, Any]]] = {}
        for row in task_rows:
            coefficient_groups.setdefault((str(row["method"]), row["lambda"]), []).append(row)
        grouped = [aggregate_group(group) for group in coefficient_groups.values()]
        positive = next((row for row in grouped if row["method"] == METHOD_POSITIVE_ONLY), None)
        positive_score = None if positive is None else float(positive["late_window_pass8_mean"])
        grouped_exp = [row for row in grouped if row["method"] == METHOD_EXPONENTIAL]
        selectable = [row for row in grouped_exp if not row["nan_inf_failure"]]
        selected = (
            max(
                selectable,
                key=lambda row: (
                    float(row["late_window_pass8_mean"]),
                    float(row["terminal_pass8_mean"]),
                    float(row["late_window_greedy_mean"]),
                    -float(row["lambda"]),
                ),
            )
            if selectable
            else None
        )
        min_lambda = min(float(row["lambda"]) for row in grouped_exp)
        max_lambda = max(float(row["lambda"]) for row in grouped_exp)
        selected_on_edge = bool(
            selected is not None
            and (
                math.isclose(float(selected["lambda"]), min_lambda)
                or math.isclose(float(selected["lambda"]), max_lambda)
            )
        )
        task_summary = {
            "task": task,
            "positive_only": positive,
            "global": next((row for row in grouped if row["method"] == METHOD_GLOBAL), None),
            "selectable_exp_count": len(selectable),
            "selected_exp": selected,
            "selected_on_grid_edge": selected_on_edge,
            "terminal_valid_rate_role": "diagnostic_only_not_selection_eligibility",
            "all_exp_below_positive_only": None
            if positive_score is None
            else all(float(row["late_window_pass8_mean"]) < positive_score for row in grouped_exp),
            "grouped_curve": sorted(
                grouped,
                key=lambda row: (
                    row["method"] != METHOD_POSITIVE_ONLY,
                    -1.0 if row["lambda"] is None else float(row["lambda"]),
                ),
            ),
        }
        summaries[task] = task_summary
        summary_rows.append(
            {
                "task": task,
                "positive_only_late_window_pass8_mean": positive_score,
                "selected_lambda": None if selected is None else selected["lambda"],
                "selected_rho": None if selected is None else selected["rho"],
                "selected_late_window_pass8_mean": (
                    None if selected is None else selected["late_window_pass8_mean"]
                ),
                "selected_on_grid_edge": selected_on_edge,
                "all_exp_below_positive_only": task_summary["all_exp_below_positive_only"],
            }
        )
    _write_csv(output_root / "aggregate" / "task_summary.csv", summary_rows)

    protocol_diagnostic = _countdown_protocol_diagnostic(
        config,
        output_root,
        destination=output_root / "aggregate" / "countdown_protocol_diagnostic.json",
    )
    summary = {
        "schema_version": 1,
        "experiment_id": experiment_id(config),
        "run_id": run_id,
        "source_commit": source_commit,
        "cell_count": len(rows),
        "plot_curve_point_count": len(plot_rows),
        "tasks": summaries,
        "excluded_tasks": dict(config["suite"]["excluded_tasks"]),
        "initialization": dict(config["initialization"]),
        "positive_only_and_exp_share_fresh_initialization": bool(
            tuple(config["sweep"].get("transfer_positive_only_seed_offsets", ()))
        ),
        "scientific_kernel": "canonical_old_coldstart_imports",
        "canonical_source_git_blob_shas": dict(
            config["canonical_coldstart"]["expected_git_blob_shas"]
        ),
        "countdown_protocol_diagnostic": protocol_diagnostic,
        "countdown_result_gate": False,
        "primary_metric": "validation_late_window_pass8_mean",
        "terminal_valid_rate_role": "diagnostic_only_not_selection_eligibility",
        "test_partition_accessed": False,
        "transfer_exp_single_seed_response_shape_localization": True,
        "transfer_positive_only_seed_count": len(
            tuple(
                int(value)
                for value in config["sweep"].get("transfer_positive_only_seed_offsets", ())
            )
        ),
        "fresh_seed_confirmation_required_for_winner_claim": True,
        "method_ranking_allowed": False,
        "significance_claim_allowed": False,
        "fixed_horizon_is_convergence": False,
        "task_performance_reported_separately": True,
        "structure_diagnostic_reported_separately": True,
        "nan_inf_reported_separately": True,
        "scientific_status": "not_run" if _is_engineering_self_test(config) else "pilot",
        "engineering_placeholder_backend": _is_engineering_self_test(config),
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
        common = {
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
            "stage": cell.stage,
            "cell_key": cell.key,
            "nan_inf_failure": bool(value["nan_inf_failure"]),
        }
        if _is_coldstart(config):
            if not _is_engineering_self_test(config) and (
                "validation_late_window_pass8_mean" not in value
                or "validation_late_window_greedy_mean" not in value
            ):
                raise RuntimeError(f"{cell.key} is missing the paper primary late-window metric")
            common.update(
                {
                    "late_window_pass8_mean": value.get(
                        "validation_late_window_pass8_mean",
                        value["validation_best_pass8"],
                    ),
                    "late_window_greedy_mean": value.get(
                        "validation_late_window_greedy_mean",
                        value["validation_best_greedy"],
                    ),
                    "best_pass8": value["validation_best_pass8"],
                    "terminal_pass8": value["validation_terminal_pass8"],
                    "best_greedy": value["validation_best_greedy"],
                    "terminal_greedy": value["validation_terminal_greedy"],
                    "best_greedy_valid_rate": value["validation_best_greedy_valid_rate"],
                    "terminal_greedy_valid_rate": value["validation_terminal_greedy_valid_rate"],
                    "best_step": value["best_step"],
                    "terminal_step": value["terminal_step"],
                    "stop_reason": value["stop_reason"],
                }
            )
        else:
            common.update(
                {
                    "late_window_pass8_mean": value["validation_late_window_pass8_mean"],
                    "terminal_pass8": value["validation_terminal_pass8"],
                    "late_window_greedy_mean": value["validation_late_window_greedy_mean"],
                    "terminal_greedy": value["validation_terminal_greedy"],
                    "terminal_greedy_valid_rate": value["validation_terminal_greedy_valid_rate"],
                }
            )
        rows.append(common)
    if missing:
        raise RuntimeError(f"Cannot aggregate; missing/incomplete cells: {missing}")
    _write_csv(output_root / "aggregate" / "all_cells.csv", rows)
    if _is_coldstart(config):
        return _aggregate_coldstart(config, output_root, rows)
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
    provenance_path = output_root / "source_provenance.json"
    if not provenance_path.is_file():
        raise RuntimeError("source_provenance.json is required before terminal audit")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    base_commit = str(provenance.get("source_commit", ""))
    if len(base_commit) != 40 or any(char not in "0123456789abcdef" for char in base_commit):
        raise RuntimeError("source_provenance.json must contain one full lowercase Git SHA")
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
    reproduction_gate_status: str | None = None
    if _is_dense(config):
        snapshot_path = output_root / "inherited" / "parent_snapshot.json"
        aggregate_path = output_root / "aggregate" / "aggregate_summary.json"
        inherited_complete = snapshot_path.is_file() and bool(
            json.loads(snapshot_path.read_text(encoding="utf-8")).get("complete")
        )
        aggregate_complete = aggregate_path.is_file() and int(
            json.loads(aggregate_path.read_text(encoding="utf-8")).get("cell_count", 0)
        ) == len(cells)
    elif _is_coldstart(config):
        aggregate_path = output_root / "aggregate" / "aggregate_summary.json"
        protocol_path = output_root / "aggregate" / "countdown_protocol_diagnostic.json"
        if protocol_path.is_file():
            reproduction_gate_status = str(
                json.loads(protocol_path.read_text(encoding="utf-8")).get("status")
            )
        expected_protocol_status = (
            "NOT_RUN_ENGINEERING"
            if _is_engineering_self_test(config)
            else ("NOT_RUN" if not any(cell.task == "countdown" for cell in cells) else "PASS")
        )
        aggregate_complete = (
            aggregate_path.is_file()
            and int(json.loads(aggregate_path.read_text(encoding="utf-8")).get("cell_count", 0))
            == len(cells)
            and reproduction_gate_status == expected_protocol_status
        )
    all_complete = (
        not missing and not incomplete and not nan_inf and inherited_complete and aggregate_complete
    )
    audit = {
        "schema_version": 1,
        "experiment_id": experiment_id(config),
        "base_commit": base_commit,
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
        "countdown_protocol_diagnostic_status": reproduction_gate_status,
        "countdown_result_gate": False if _is_coldstart(config) else None,
        "transfer_exp_single_seed_response_shape_localization": _is_coldstart(config),
        "excluded_tasks": (
            dict(config["suite"]["excluded_tasks"])
            if (_is_dense(config) or _is_coldstart(config))
            else {}
        ),
        "single_seed_shape_discovery": _is_dense(config) or _is_coldstart(config),
        "fresh_seed_confirmation_required": _is_dense(config) or _is_coldstart(config),
        "fixed_horizon_is_convergence": False,
        "scientific_status": (
            "not_run" if _is_engineering_self_test(config) or not all_complete else "pilot"
        ),
        "engineering_placeholder_backend": _is_engineering_self_test(config),
    }
    atomic_json(output_root / "terminal_audit.json", audit)
    return audit


PACKAGE_REQUIRED_MEMBERS = {
    "RUN_COMPLETE.json",
    "run_manifest.json",
    "scientific_run_manifest.json",
    "source_provenance.json",
    "terminal_audit.json",
    "scheduler/dynamic_run.json",
    "aggregate/plot_curve_points.csv",
    "package_contents_manifest.json",
    "SHA256SUMS.txt",
}


def _write_completion_manifests(
    config: Mapping[str, Any],
    output_root: Path,
    audit: Mapping[str, Any],
) -> None:
    provenance_path = output_root / "source_provenance.json"
    scheduler_path = output_root / "scheduler" / "dynamic_run.json"
    aggregate_path = output_root / "aggregate" / "aggregate_summary.json"
    if (
        not provenance_path.is_file()
        or not scheduler_path.is_file()
        or not aggregate_path.is_file()
    ):
        raise RuntimeError("Source provenance, scheduler result, and aggregate are required")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    scheduler = json.loads(scheduler_path.read_text(encoding="utf-8"))
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    source_commit = str(provenance.get("source_commit", ""))
    if len(source_commit) != 40 or any(char not in "0123456789abcdef" for char in source_commit):
        raise RuntimeError("source_provenance.json must contain one full lowercase Git SHA")
    expected_cells = len(build_cells(config))
    if (
        scheduler.get("experiment_id") != experiment_id(config)
        or not scheduler.get("complete")
        or int(scheduler.get("expected_cells", 0)) != expected_cells
        or int(scheduler.get("completed_cells", 0)) != expected_cells
        or int(aggregate.get("cell_count", 0)) != expected_cells
    ):
        raise RuntimeError("Scheduler or aggregate is not terminal-complete")
    self_test = _is_engineering_self_test(config)
    run_manifest = {
        "schema_version": 1,
        "experiment_id": experiment_id(config),
        "base_commit": source_commit,
        "run_id": str(provenance.get("run_id", output_root.name)),
        "source_commit": source_commit,
        "config_hash": stable_config_hash(config),
        "expected_cells": expected_cells,
        "completed_cells": expected_cells,
        "scheduler": "dynamic_slot_queue",
        "scheduler_run_id": scheduler["scheduler_run_id"],
        "test_partition_accessed": False,
        "engineering_placeholder_backend": self_test,
        "scientific_status": "not_run" if self_test else "pilot",
        "artifact_state": "engineering_self_test_complete" if self_test else "raw_complete",
    }
    atomic_json(output_root / "run_manifest.json", run_manifest)
    atomic_json(output_root / "scientific_run_manifest.json", run_manifest)
    atomic_json(
        output_root / "RUN_COMPLETE.json",
        {
            **run_manifest,
            "all_training_and_evaluation_complete": bool(
                audit["all_training_and_evaluation_complete"]
            ),
            "terminal_audit_sha256": sha256_file(output_root / "terminal_audit.json"),
            "aggregate_sha256": sha256_file(aggregate_path),
            "complete": True,
        },
    )


def _result_payload_paths(output_root: Path, excluded_parts: set[str]) -> list[Path]:
    paths: list[Path] = []
    for path in sorted(output_root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(output_root)
        if path.is_symlink():
            raise RuntimeError(f"Result package refuses symlink payload: {relative}")
        if "packages" in relative.parts or relative.as_posix() in {
            "package_contents_manifest.json",
            "SHA256SUMS.txt",
        }:
            continue
        if any(part in excluded_parts for part in relative.parts) or path.suffix in {
            ".bin",
            ".safetensors",
        }:
            continue
        paths.append(path)
    return paths


def verify_result_package(
    package_manifest_path: Path,
    *,
    zip_override: Path | None = None,
) -> dict[str, Any]:
    """Reopen a result ZIP and verify paths, inventory, hashes, and required members."""

    manifest = json.loads(package_manifest_path.read_text(encoding="utf-8"))
    zip_path = (zip_override or Path(str(manifest["full_results_zip"]))).resolve()
    if not zip_path.is_file():
        raise FileNotFoundError(f"Result ZIP is missing: {zip_path}")
    observed_zip_sha = sha256_file(zip_path)
    if observed_zip_sha != manifest["full_results_zip_sha256"]:
        raise RuntimeError("Result ZIP SHA-256 does not match package_manifest.json")
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise RuntimeError("Result ZIP contains duplicate members")
        for name in names:
            member = PurePosixPath(name)
            if member.is_absolute() or ".." in member.parts or not member.parts:
                raise RuntimeError(f"Unsafe result ZIP member: {name}")
        missing = sorted(PACKAGE_REQUIRED_MEMBERS - set(names))
        if missing:
            raise RuntimeError(f"Result ZIP is missing required members: {missing}")
        contents = json.loads(archive.read("package_contents_manifest.json"))
        inventory = {str(item["path"]): str(item["sha256"]) for item in contents.get("files", ())}
        expected_names = set(inventory) | {"package_contents_manifest.json", "SHA256SUMS.txt"}
        if set(names) != expected_names:
            raise RuntimeError("Result ZIP members do not match package_contents_manifest.json")
        for name, expected_sha in inventory.items():
            observed = hashlib.sha256(archive.read(name)).hexdigest()
            if observed != expected_sha:
                raise RuntimeError(f"Result ZIP payload hash mismatch: {name}")
        checksum_rows: dict[str, str] = {}
        for line in archive.read("SHA256SUMS.txt").decode("utf-8").splitlines():
            digest, separator, name = line.partition("  ")
            if not separator or name in checksum_rows:
                raise RuntimeError("Malformed or duplicate SHA256SUMS.txt entry")
            checksum_rows[name] = digest
        if checksum_rows != inventory:
            raise RuntimeError("SHA256SUMS.txt does not match the package inventory")
        if not any(name.startswith("logs/") for name in names):
            raise RuntimeError("Result ZIP contains no execution logs")
    return {
        "verified": True,
        "zip": str(zip_path),
        "zip_sha256": observed_zip_sha,
        "member_count": len(names),
        "required_members_present": True,
    }


def cmd_package(config: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    """Create and independently reopen a portable text-first result ZIP."""

    audit_path = output_root / "terminal_audit.json"
    plot_path = output_root / "aggregate" / "plot_curve_points.csv"
    if not audit_path.is_file() or not plot_path.is_file():
        raise RuntimeError("Run aggregate and audit before packaging")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit.get("all_training_and_evaluation_complete"):
        raise RuntimeError("Refusing to package a non-terminal run")
    _write_completion_manifests(config, output_root, audit)
    package_root = output_root / "packages"
    package_root.mkdir(parents=True, exist_ok=True)
    zip_path = package_root / f"{output_root.name}_full_results.zip"
    excluded_parts = {
        "best_adapter",
        "terminal_adapter",
        "last_finite_adapter",
        "supplementary_best_adapter",
        "initial_adapter",
    }
    payload_paths = _result_payload_paths(output_root, excluded_parts)
    inventory = [
        {
            "path": path.relative_to(output_root).as_posix(),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in payload_paths
    ]
    atomic_json(
        output_root / "package_contents_manifest.json",
        {
            "schema_version": 1,
            "experiment_id": experiment_id(config),
            "engineering_placeholder_backend": _is_engineering_self_test(config),
            "files": inventory,
        },
    )
    (output_root / "SHA256SUMS.txt").write_text(
        "".join(f"{item['sha256']}  {item['path']}\n" for item in inventory),
        encoding="utf-8",
    )
    payload_paths.extend(
        [output_root / "package_contents_manifest.json", output_root / "SHA256SUMS.txt"]
    )
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in payload_paths:
            archive.write(path, arcname=path.relative_to(output_root).as_posix())
    result = {
        "schema_version": 1,
        "experiment_id": experiment_id(config),
        "artifact_kind": (
            "engineering_self_test" if _is_engineering_self_test(config) else "pilot_results"
        ),
        "full_results_zip": str(zip_path.resolve()),
        "full_results_zip_sha256": sha256_file(zip_path),
        "full_results_zip_bytes": zip_path.stat().st_size,
        "plot_curve_points_csv": str(plot_path.resolve()),
        "plot_curve_points_csv_sha256": sha256_file(plot_path),
        "included_file_count": len(payload_paths),
        "excluded_model_weights": sorted(excluded_parts),
        "complete": True,
    }
    manifest_path = package_root / "package_manifest.json"
    atomic_json(manifest_path, result)
    verification = verify_result_package(manifest_path)
    result["reopen_verification"] = verification
    atomic_json(manifest_path, result)
    return result


def cmd_finalize(config: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    """Finalize result markers while leaving archive ownership to the hardened guard."""

    audit_path = output_root / "terminal_audit.json"
    plot_path = output_root / "aggregate" / "plot_curve_points.csv"
    if not audit_path.is_file() or not plot_path.is_file():
        raise RuntimeError("Run aggregate and audit before finalizing")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit.get("all_training_and_evaluation_complete"):
        raise RuntimeError("Refusing to finalize a non-terminal run")
    _write_completion_manifests(config, output_root, audit)
    return {
        "schema_version": 1,
        "experiment_id": experiment_id(config),
        "base_commit": audit["base_commit"],
        "artifact_state": "raw_complete",
        "canonical_archive_owner": "scripts/run_experiment_guard_hardened.py",
        "plot_curve_points_csv": str(plot_path.resolve()),
        "plot_curve_points_csv_sha256": sha256_file(plot_path),
        "complete": True,
    }


def _engineering_self_test_config(config: Mapping[str, Any]) -> dict[str, Any]:
    updated = copy.deepcopy(dict(config))
    updated["split"].update(
        {
            "p0_train_rows": 2,
            "p0_validation_rows": 1,
            "p0_test_rows": 1,
            "countdown_train_rows": 2,
            "countdown_validation_rows": 1,
        }
    )
    updated["engineering_self_test"] = {
        "placeholder_backend": True,
        "scientific_evidence_allowed": False,
        "purpose": "non_gpu_end_to_end_delivery_acceptance",
    }
    validate_config(updated)
    return updated


def _write_engineering_input_fixtures(
    config: Mapping[str, Any],
    output_root: Path,
) -> tuple[Path, Path, Path, Path]:
    fixture_root = output_root / "engineering_fixtures"
    p0_work_dir = fixture_root / "p0"
    sources_root = p0_work_dir / "sources"
    sources_root.mkdir(parents=True, exist_ok=True)
    p0_config_path = _repo_root() / "configs" / "e8_multitask_p0.yaml"
    p0_config = yaml.safe_load(p0_config_path.read_text(encoding="utf-8"))
    if not isinstance(p0_config, dict):
        raise TypeError("P0 configuration root must be a mapping")
    row_count = (
        int(config["split"]["p0_train_rows"])
        + int(config["split"]["p0_validation_rows"])
        + int(config["split"]["p0_test_rows"])
    )
    task_qualification: dict[str, Any] = {}
    for task_value in config["suite"]["p0_tasks"]:
        task = str(task_value)
        rows: list[dict[str, Any]] = []
        for row_index in range(row_count):
            rows.append(
                {
                    "schema_version": 1,
                    "task": task,
                    "prompt_id": f"{task}-placeholder-{row_index:03d}",
                    "prompt": f"[{task}] engineering prompt {row_index}",
                    "oracle_completion": f"{task}-oracle-{row_index}",
                    "metadata": {"engineering_placeholder": True},
                    "generation_seed": 2026080901,
                    "negatives": [
                        {
                            "negative_id": f"{task}-{row_index:03d}-neg-{negative:02d}",
                            "completion": f"{task}-wrong-{row_index}-{negative}",
                            "format_valid": True,
                            "binary_correct": False,
                            "error_class": "engineering_placeholder_wrong",
                        }
                        for negative in range(16)
                    ],
                }
            )
        atomic_jsonl(bank_path(p0_work_dir, task), rows)
        task_qualification[task] = {"passed": True, "engineering_placeholder": True}
    qualification = {
        "schema_version": 1,
        "experiment_id": P0_EXPERIMENT_ID,
        "config_hash": stable_config_hash(
            with_smoke_overrides(p0_config, rows=None, negatives=None)
        ),
        "tasks": task_qualification,
        "passed": True,
        "scientific_status": "not_run",
        "engineering_placeholder_backend": True,
    }
    atomic_json(p0_work_dir / "qualification_audit.json", qualification)

    countdown_bank = fixture_root / "countdown" / "offline_bank_v2.jsonl"
    countdown_rows = []
    for row_index in range(int(config["split"]["countdown_train_rows"])):
        countdown_rows.append(
            {
                "row_id": f"countdown-placeholder-train-{row_index:03d}",
                "source_prompt_id": f"countdown-placeholder-source-{row_index:03d}",
                "prompt": f"Use 1 and 2 to make {3 + row_index}",
                "oracle_positive": "1 + 2",
                "numbers": [1, 2],
                "target": 3 + row_index,
                "negative_bank": [
                    {
                        "expression": f"1 - 2 + {negative}",
                        "valid_format": True,
                        "correct": False,
                        "source": "engineering_placeholder_wrong",
                    }
                    for negative in range(16)
                ],
            }
        )
    atomic_jsonl(countdown_bank, countdown_rows)
    countdown_validation = fixture_root / "countdown" / "val.jsonl"
    atomic_jsonl(
        countdown_validation,
        [
            {
                "id": "countdown-placeholder-validation-000",
                "prompt": "Use 2 and 2 to make 4",
                "oracle": "2 + 2",
                "numbers": [2, 2],
                "target": 4,
            }
        ],
    )
    return p0_work_dir, p0_config_path, countdown_bank, countdown_validation


def _write_engineering_gates(
    config: Mapping[str, Any],
    output_root: Path,
    *,
    base_model_path: str,
) -> None:
    splits, _ = _load_ready_inputs(output_root, config, base_model_path=base_model_path)
    calibration_tasks: dict[str, Any] = {}
    for task_value in config["suite"]["tasks"]:
        task = str(task_value)
        identity = _canonical_calibration_identity(
            task,
            split_manifest=splits,
            base_model_path=base_model_path,
            config=config,
        )
        result = {
            **identity,
            "enabled": False,
            "mode": "paper_linear_surprisal_no_calibration",
            "complete": True,
            "scientific_status": "not_run",
            "engineering_placeholder_backend": True,
        }
        atomic_json(output_root / "calibration" / f"{task}.json", result)
        calibration_tasks[task] = result
        log_path = output_root / "logs" / "calibration" / f"{task}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("engineering placeholder calibration complete\n", encoding="utf-8")
    atomic_json(
        output_root / "calibration" / "calibration_manifest.json",
        {
            "schema_version": 1,
            "experiment_id": experiment_id(config),
            "config_hash": stable_config_hash(config),
            "requested_tasks": list(config["suite"]["tasks"]),
            "tasks": calibration_tasks,
            "complete": True,
            "scientific_status": "not_run",
            "engineering_placeholder_backend": True,
        },
    )
    base_identity = model_identity(base_model_path, None)["model"]
    liveness_key = "countdown__engineering_placeholder_liveness"
    liveness = {
        "schema_version": 1,
        "experiment_id": experiment_id(config),
        "config_hash": stable_config_hash(config),
        "base_model_identity": base_identity,
        "engineering_liveness": True,
        "engineering_placeholder_backend": True,
        "optimizer_updates": 2,
        "complete": True,
        "reload_gate_passed": True,
        "adapter_weight_changed": True,
        "fresh_process_reload_passed": True,
        "liveness_parent_process_id": 1001,
        "reload_process_id": 1002,
        "nan_inf_failure": False,
        "canonical_dispatch_verified": True,
        "finite_old_core_updates": True,
        "optimizer_update_norm": 1.0,
        "initial_adapter_weight_sha256": "0" * 64,
        "terminal_adapter_weight_sha256": "1" * 64,
        "cell": {"task": "countdown"},
        "scientific_status": "not_run",
    }
    atomic_json(output_root / "liveness" / liveness_key / "cell_manifest.json", liveness)
    (output_root / "logs" / "liveness.log").write_text(
        "engineering placeholder liveness complete\n",
        encoding="utf-8",
    )


def _audit_engineering_queue(
    config: Mapping[str, Any],
    output_root: Path,
    scheduler: Mapping[str, Any],
) -> dict[str, Any]:
    cells = build_cells(config)
    events = [
        json.loads(line)
        for line in (output_root / "scheduler" / "queue_events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    events = [row for row in events if row["scheduler_run_id"] == scheduler["scheduler_run_id"]]
    active_by_gpu = {int(gpu): 0 for gpu in config["execution"]["gpu_ids"]}
    maximum_by_gpu = dict(active_by_gpu)
    starts: dict[str, float] = {}
    finishes: dict[str, float] = {}
    for event in events:
        gpu_id = int(event["gpu_id"])
        cell_key = str(event["cell_key"])
        if event["event"] == "start":
            active_by_gpu[gpu_id] += 1
            maximum_by_gpu[gpu_id] = max(maximum_by_gpu[gpu_id], active_by_gpu[gpu_id])
            starts[cell_key] = float(event["unix_time"])
        elif event["event"] == "finish":
            active_by_gpu[gpu_id] -= 1
            finishes[cell_key] = float(event["unix_time"])
    expected_keys = {cell.key for cell in cells}
    if set(starts) != expected_keys or set(finishes) != expected_keys:
        raise RuntimeError(f"Engineering queue did not observe all {len(cells)} starts/finishes")
    slots_per_gpu = int(config["execution"]["slots_per_gpu"])
    if (
        any(value != 0 for value in active_by_gpu.values())
        or max(maximum_by_gpu.values()) > slots_per_gpu
    ):
        raise RuntimeError("Engineering queue exceeded the declared per-GPU capacity")
    slot_count = int(config["execution"]["max_concurrent_cells"])
    initial_keys = {cell.key for cell in cells[:slot_count]}
    replacement_keys = {cell.key for cell in cells[slot_count:]}
    if not replacement_keys:
        raise RuntimeError("Engineering queue requires replacement cells to audit dynamic refill")
    if min(starts[key] for key in replacement_keys) >= max(finishes[key] for key in initial_keys):
        raise RuntimeError("Engineering queue did not refill before the initial 16 cells finished")
    return {
        "all_cells_observed": True,
        "maximum_active_by_gpu": maximum_by_gpu,
        "dynamic_refill_observed": True,
        "nominal_batch_barrier_absent": True,
        "nominal_batch_count": len(build_waves(config)),
        "slots_per_gpu": slots_per_gpu,
    }


def cmd_engineering_self_test(
    config: Mapping[str, Any],
    output_root: Path,
    *,
    source_commit: str,
) -> dict[str, Any]:
    """Exercise the delivery pipeline with an isolated, non-scientific backend."""

    if not _is_coldstart(config):
        raise RuntimeError("Engineering self-test is available only for the cold-start profile")
    if len(source_commit) != 40 or any(char not in "0123456789abcdef" for char in source_commit):
        raise ValueError("Engineering self-test requires one full lowercase source commit")
    output_root = validate_work_dir(output_root)
    fresh_run = not (output_root / "prepare_manifest.json").is_file()
    self_test_config = _engineering_self_test_config(config)
    config_path = output_root / "engineering_self_test_config.yaml"
    base_model = output_root / "engineering_fixtures" / "placeholder_model"
    if fresh_run:
        config_path.write_text(yaml.safe_dump(self_test_config, sort_keys=False), encoding="utf-8")
        p0_work_dir, p0_config_path, countdown_bank, countdown_validation = (
            _write_engineering_input_fixtures(self_test_config, output_root)
        )
        cmd_prepare(
            self_test_config,
            output_root,
            p0_work_dir=p0_work_dir,
            p0_config=p0_config_path,
            countdown_bank=countdown_bank,
            countdown_validation=countdown_validation,
            countdown_adapter=None,
        )
        base_model.mkdir(parents=True, exist_ok=True)
        atomic_json(base_model / "config.json", {"engineering_placeholder_backend": True})
        atomic_json(
            output_root / "source_provenance.json",
            {
                "schema_version": 1,
                "run_id": output_root.name,
                "source_commit": source_commit,
                "model_repo": "engineering-placeholder-no-model-loaded",
                "model_revision": "not_applicable",
                "model_path": str(base_model.resolve()),
                "test_partition_accessed": False,
                "engineering_placeholder_backend": True,
            },
        )
        _write_engineering_gates(
            self_test_config,
            output_root,
            base_model_path=str(base_model),
        )
    else:
        recovered_config = load_config(config_path)
        if recovered_config != self_test_config:
            raise RuntimeError("Engineering recovery config differs from the reviewed config")
        provenance = _read_json_object(output_root / "source_provenance.json")
        if provenance.get("source_commit") != source_commit:
            raise RuntimeError("Engineering recovery source commit mismatch")
        _load_prepared(output_root, self_test_config)
        _write_engineering_gates(
            self_test_config,
            output_root,
            base_model_path=str(base_model),
        )
        _require_calibration_gate(
            self_test_config,
            output_root,
            base_model_path=str(base_model),
        )
        _require_liveness_gate(
            self_test_config,
            output_root,
            base_model_path=str(base_model),
        )

    cells = build_cells(self_test_config)
    cell_index = {cell.key: index for index, cell in enumerate(cells)}
    failed_once = False
    failure_lock = threading.Lock()

    def placeholder_cell_runner(
        *,
        config_path: Path,
        output_root: Path,
        base_model_path: str,
        cell: Cell,
        gpu_id: int,
        force: bool,
    ) -> dict[str, Any]:
        del config_path, base_model_path, force
        nonlocal failed_once
        started_at = time.time()
        cell_root = output_root / "cells" / cell.key
        manifest_path = cell_root / "cell_manifest.json"
        log_path = output_root / "logs" / f"{cell.key}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if manifest_path.is_file():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                existing.get("config_hash") != stable_config_hash(self_test_config)
                or existing.get("engineering_placeholder_backend") is not True
                or existing.get("complete") is not True
            ):
                raise RuntimeError(f"Placeholder resume identity mismatch: {cell.key}")
            if cell.key == cells[0].key:
                time.sleep(0.05)
            return {
                "cell_key": cell.key,
                "gpu_id": gpu_id,
                "returncode": 0,
                "log": str(log_path.resolve()),
                "started_unix": started_at,
                "finished_unix": time.time(),
                "reused_complete": True,
            }
        with failure_lock:
            should_fail = fresh_run and cell.key == cells[0].key and not failed_once
            if should_fail:
                failed_once = True
        if should_fail:
            cell_root.mkdir(parents=True, exist_ok=True)
            log_path.write_text("intentional engineering failure\n", encoding="utf-8")
            atomic_json(
                cell_root / "failure.json",
                {
                    "experiment_id": experiment_id(self_test_config),
                    "cell_key": cell.key,
                    "engineering_placeholder_backend": True,
                    "complete": False,
                },
            )
            return {
                "cell_key": cell.key,
                "gpu_id": gpu_id,
                "returncode": 73,
                "log": str(log_path.resolve()),
                "started_unix": started_at,
                "finished_unix": time.time(),
            }
        if cell.key == cells[0].key:
            time.sleep(0.05)
        else:
            time.sleep(0.001)
        index = cell_index[cell.key]
        score = round(0.1 + (index % 20) * 0.01, 6)
        manifest = {
            "schema_version": 1,
            "experiment_id": experiment_id(self_test_config),
            "config_hash": stable_config_hash(self_test_config),
            "source_commit": source_commit,
            "cell_key": cell.key,
            "validation_best_pass8": score,
            "validation_terminal_pass8": max(0.0, score - 0.005),
            "validation_best_greedy": max(0.0, score - 0.02),
            "validation_terminal_greedy": max(0.0, score - 0.025),
            "validation_best_greedy_valid_rate": 1.0,
            "validation_terminal_greedy_valid_rate": 1.0,
            "best_step": 2,
            "terminal_step": 2,
            "stop_reason": "engineering_placeholder_complete",
            "nan_inf_failure": False,
            "evaluation_status": "complete",
            "complete": True,
            "scientific_status": "not_run",
            "engineering_placeholder_backend": True,
        }
        atomic_json(manifest_path, manifest)
        log_path.write_text("engineering placeholder cell complete\n", encoding="utf-8")
        return {
            "cell_key": cell.key,
            "gpu_id": gpu_id,
            "returncode": 0,
            "log": str(log_path.resolve()),
            "started_unix": started_at,
            "finished_unix": time.time(),
            "reused_complete": False,
        }

    original_runner = globals()["_run_subprocess_cell"]
    globals()["_run_subprocess_cell"] = placeholder_cell_runner
    try:
        if fresh_run:
            first_failure: dict[str, Any]
            try:
                cmd_run_dynamic(
                    self_test_config,
                    config_path,
                    output_root,
                    base_model_path=str(base_model),
                    force=False,
                    retry_incomplete=True,
                )
            except RuntimeError:
                first_failure = json.loads(
                    (output_root / "scheduler" / "dynamic_run.json").read_text(encoding="utf-8")
                )
            else:
                raise RuntimeError("Engineering failure injection did not fail closed")
            if not first_failure["failed_cells"] or not first_failure["unscheduled_cells"]:
                raise RuntimeError(
                    "Engineering failure did not preserve failed and unscheduled work"
                )
            resumed = cmd_run_dynamic(
                self_test_config,
                config_path,
                output_root,
                base_model_path=str(base_model),
                force=False,
                retry_incomplete=True,
            )
        else:
            resumed = cmd_run_dynamic(
                self_test_config,
                config_path,
                output_root,
                base_model_path=str(base_model),
                force=False,
                retry_incomplete=True,
            )
        queue_audit = _audit_engineering_queue(self_test_config, output_root, resumed)
        before = {
            cell.key: sha256_file(output_root / "cells" / cell.key / "cell_manifest.json")
            for cell in cells
        }
        repeated = cmd_run_dynamic(
            self_test_config,
            config_path,
            output_root,
            base_model_path=str(base_model),
            force=False,
            retry_incomplete=True,
        )
        after = {
            cell.key: sha256_file(output_root / "cells" / cell.key / "cell_manifest.json")
            for cell in cells
        }
        if before != after or not repeated["complete"]:
            raise RuntimeError("Engineering repeat run changed a completed cell")
    finally:
        globals()["_run_subprocess_cell"] = original_runner

    failure_stage = os.environ.get("E8_COLDSTART_ENGINEERING_FAIL_STAGE", "").strip()
    if failure_stage == "after_queue":
        raise RuntimeError("Intentional engineering failure after all cells completed")
    aggregate = cmd_aggregate(self_test_config, output_root)
    if failure_stage == "after_aggregate":
        raise RuntimeError("Intentional engineering failure after aggregate")
    audit = cmd_audit(self_test_config, output_root)
    if failure_stage == "after_audit":
        raise RuntimeError("Intentional engineering failure after audit")
    finalized = cmd_finalize(self_test_config, output_root)
    if failure_stage == "after_finalize":
        raise RuntimeError("Intentional engineering failure after finalize")
    preliminary_package = cmd_package(self_test_config, output_root)
    package_manifest_path = output_root / "packages" / "package_manifest.json"
    tampered = output_root / "packages" / "tampered_self_test.zip"
    shutil.copyfile(preliminary_package["full_results_zip"], tampered)
    with tampered.open("ab") as handle:
        handle.write(b"tamper")
    tamper_rejected = False
    try:
        verify_result_package(package_manifest_path, zip_override=tampered)
    except RuntimeError:
        tamper_rejected = True
    finally:
        tampered.unlink(missing_ok=True)
    if not tamper_rejected:
        raise RuntimeError("Result-package verification accepted a tampered ZIP")
    report = {
        "schema_version": 1,
        "experiment_id": experiment_id(self_test_config),
        "source_commit": source_commit,
        "scientific_status": "not_run",
        "engineering_placeholder_backend": True,
        "prepare_complete": True,
        "canonical_source_lock_passed": True,
        "intentional_failure_returncode": 73,
        "failure_preserved_unscheduled_work": True,
        "recovered_from_previous_attempt": not fresh_run,
        "resume_completed_cells": resumed["completed_cells"],
        "repeat_run_preserved_cell_hashes": True,
        "analysis_ready_tasks": sorted(repeated.get("analysis_ready_tasks", ())),
        "task_result_count": len(repeated.get("task_results", {})),
        "queue_audit": queue_audit,
        "aggregate_cell_count": aggregate["cell_count"],
        "terminal_audit_complete": audit["all_training_and_evaluation_complete"],
        "canonical_archive_owner": finalized["canonical_archive_owner"],
        "package_reopen_verification_passed": True,
        "tampered_package_rejected": True,
        "complete": True,
        "note": "No model, GPU, optimizer, or scientific metric was executed.",
    }
    atomic_json(output_root / "ENGINEERING_SELF_TEST_REPORT.json", report)
    final_package = cmd_package(self_test_config, output_root)
    return {
        **report,
        "output_root": str(output_root.resolve()),
        "full_results_zip": final_package["full_results_zip"],
        "full_results_zip_sha256": final_package["full_results_zip_sha256"],
        "plot_curve_points_csv": final_package["plot_curve_points_csv"],
        "plot_curve_points_csv_sha256": final_package["plot_curve_points_csv_sha256"],
    }


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
    prepare.add_argument("--countdown-adapter")

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

    calibrate_task = subparsers.add_parser("calibrate-task", help=argparse.SUPPRESS)
    calibrate_task.add_argument("--base-model-path", required=True)
    calibrate_task.add_argument("--task", required=True)
    calibrate_task.add_argument("--force", action="store_true")

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
    run_all.add_argument("--retry-incomplete", action="store_true")

    subparsers.add_parser("aggregate")
    subparsers.add_parser("audit")
    subparsers.add_parser("finalize")
    subparsers.add_parser("package")
    subparsers.add_parser("plan")
    recovery_plan = subparsers.add_parser("recovery-plan")
    recovery_plan.add_argument("--base-model-path", required=True)
    import_recovery = subparsers.add_parser("import-recovery")
    import_recovery.add_argument("--source-output-root", required=True)
    import_recovery.add_argument("--base-model-path", required=True)
    import_recovery.add_argument("--source-commit", required=True)
    subparsers.add_parser("compact-logs")
    engineering_self_test = subparsers.add_parser("engineering-self-test")
    engineering_self_test.add_argument("--source-commit", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = make_parser().parse_args(argv)
    config_path, _, _ = experiment_config.require_tracked_config(args.config, _repo_root())
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
            countdown_adapter=(
                Path(args.countdown_adapter).resolve() if args.countdown_adapter else None
            ),
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
    elif args.command == "calibrate-task":
        result = cmd_calibrate_task(
            config,
            output_root,
            base_model_path=args.base_model_path,
            task=args.task,
            force=bool(args.force),
        )
    elif args.command == "liveness":
        rho = args.rho
        if args.lambda_value is not None:
            rho = math.exp(-float(args.lambda_value))
        if rho is None and not _is_coldstart(config):
            rho = _task_rhos(config, str(args.task))[0]
        result = cmd_liveness(
            config,
            config_path,
            output_root,
            task=args.task,
            rho=None if rho is None else float(rho),
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
            retry_incomplete=bool(args.retry_incomplete),
        )
    elif args.command == "aggregate":
        result = cmd_aggregate(config, output_root)
    elif args.command == "audit":
        result = cmd_audit(config, output_root)
    elif args.command == "finalize":
        result = cmd_finalize(config, output_root)
    elif args.command == "package":
        result = cmd_package(config, output_root)
    elif args.command == "plan":
        result = write_plan(config, output_root)
    elif args.command == "recovery-plan":
        result = cmd_recovery_plan(
            config,
            output_root,
            base_model_path=args.base_model_path,
        )
    elif args.command == "import-recovery":
        result = cmd_import_recovery(
            config,
            output_root,
            source_output_root=Path(args.source_output_root),
            base_model_path=args.base_model_path,
            source_commit=args.source_commit,
        )
    elif args.command == "compact-logs":
        result = cmd_compact_logs(config, output_root)
    elif args.command == "engineering-self-test":
        result = cmd_engineering_self_test(
            config,
            output_root,
            source_commit=str(args.source_commit),
        )
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
