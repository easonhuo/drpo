#!/usr/bin/env python3
"""Frozen Qwen2.5-3B DRPO c/tau scale-transfer profile and runtime adapter.

This approved module is intentionally narrow. It reuses the reviewed E8 V2
trainer, runtime scheduler, resource probe, bank, held-out evaluation semantics,
and result packaging. It changes only the model scale and the four explicitly
frozen ``(c, tau)`` points selected from the historical 0.5B response surface.
It does not implement TOPR, AsymRE, Reciprocal control, new bank construction,
or new hyperparameter search.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import shutil
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch

from drpo import countdown_e8_alpha1_c_scan_common as _base
from drpo import countdown_e8_alpha1_highc_scan_common as _highc

EXPERIMENT_ID = "EXT-C-E8-DRPO-CTAU-SCALE-TRANSFER-3B-01"
DRPO_CTAU_SCALE_TRANSFER_3B_EXPERIMENT_ID = EXPERIMENT_ID
VERSION = "0.1.0-dev-code-first-drpo-ctau-scale-transfer-3b"
DEFAULT_GRID_CONFIG = "configs/countdown_e8_drpo_ctau_scale_transfer_3b.yaml"
DEFAULT_BASE_CONFIG = "configs/countdown_e8_base_rl_replay_3b.yaml"
MODEL_IDENTITY = "Qwen2.5-3B-Instruct"

# Exact four-point transfer set approved for this experiment. These are the
# historical implementation-facing coefficient c and threshold tau in
# w = exp(-c * relu(u - tau)), u = current mean-token surprisal / 2.
PARAMETER_POINTS = (
    ("A", 1.609437912, 0.125),
    ("B", 1.897119985, 0.25),
    ("C", 2.995732274, 0.125),
    ("D", 4.605170186, 0.75),
)
SEED_OFFSETS = (4000, 5000)
EXPECTED_POINTS = 4
EXPECTED_CELLS = 8
_ACTIVE_TAU = 0.0
_REPO_ROOT = Path(__file__).resolve().parents[2]

EVALUATION_SEMANTICS: dict[str, Any] = {
    "evaluation_split_file": "val.jsonl",
    "evaluation_split_role": "structurally_disjoint_held_out_evaluation",
    "evaluation_enters_training_loss": False,
    "training_structure_overlap": "none",
    "training_problem_key_overlap": "none",
    "paper_facing_checkpoint_policy": "late_window_and_terminal",
    "paper_facing_summary": ["late_window_pass_at_8", "terminal_pass_at_8"],
    "best_checkpoint_role": "supplementary_only",
    "separate_test_jsonl_used": False,
    "separate_test_jsonl_required_for_existing_curve_validity": False,
}


@dataclass(frozen=True)
class Cell:
    coefficient: float
    tau: float
    seed_offset: int
    label: str
    alpha: float = 1.0
    family: str = "exponential"

    @property
    def c(self) -> float:
        return float(self.coefficient)

    @property
    def method(self) -> str:
        return "continuous_exp"

    @property
    def delta_v(self) -> float:
        return 0.0

    @property
    def name(self) -> str:
        def tag(value: float) -> str:
            return f"{value:.8g}".replace("-", "m").replace(".", "p")

        return (
            f"base_drpo_ctau_{self.label.lower()}_"
            f"c{tag(self.coefficient)}_tau{tag(self.tau)}_"
            f"seed{self.seed_offset}"
        )


sha256_file = _base.sha256_file
atomic_json = _base.atomic_json
load_yaml = _base.load_yaml
continuous_remoteness = _base.continuous_remoteness
mean_unique_negative_term = _base.mean_unique_negative_term
unique_negative_expressions = _base.unique_negative_expressions
ContinuousUniqueBankDataset = _base.ContinuousUniqueBankDataset
make_continuous_unique_bank_collator = _base.make_continuous_unique_bank_collator
weight_diagnostics = _base.weight_diagnostics
git_state = _base.git_state


def set_active_tau(value: float) -> None:
    global _ACTIVE_TAU
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("tau must be finite and non-negative")
    _ACTIVE_TAU = value


def continuous_exp_weights(
    seq_lp: torch.Tensor,
    *,
    alpha: float,
    c: float,
    reference_distance: float = _base.REFERENCE_DISTANCE,
) -> torch.Tensor:
    """Return detached DRPO weights for the active frozen tau cell."""
    if not math.isfinite(alpha) or not math.isclose(alpha, 1.0, abs_tol=1.0e-12):
        raise ValueError("3B scale transfer keeps alpha fixed at 1")
    coefficient = float(c)
    if not math.isfinite(coefficient) or coefficient <= 0.0:
        raise ValueError("c must be finite and strictly positive")
    u = continuous_remoteness(seq_lp, reference_distance=reference_distance)
    coordinate = torch.relu(u - float(_ACTIVE_TAU))
    return torch.exp(-coefficient * coordinate).detach()


def _configured_points(config: Mapping[str, Any]) -> tuple[tuple[str, float, float], ...]:
    return tuple(
        (str(item["label"]), float(item["c"]), float(item["tau"]))
        for item in config.get("sweep", {}).get("parameter_points", ())
    )


def validate_grid_config(config: Mapping[str, Any]) -> None:
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("3B DRPO c/tau transfer experiment_id mismatch")
    if config.get("result_status") != "pilot":
        raise ValueError("3B DRPO c/tau transfer must remain a pilot")
    if config.get("registration_state") != "dev_code_first_unregistered":
        raise ValueError("3B DRPO c/tau transfer must remain code-first unregistered")

    model = config.get("model", {})
    if model.get("identity") != MODEL_IDENTITY:
        raise ValueError("3B scale-transfer model identity changed")
    if model.get("initialization") != "pretrained_base_plus_fresh_lora":
        raise ValueError("3B scale transfer must start from pretrained base plus fresh LoRA")
    if model.get("parameterization") != "lora":
        raise ValueError("3B scale transfer must use LoRA")
    if model.get("transfer_source_model") != "Qwen2.5-0.5B-Instruct":
        raise ValueError("3B scale-transfer source model changed")

    bank = config.get("bank", {})
    if bank.get("source_experiment") != "EXT-C-E8-ORACLE-OFFLINE-BANK-V2-0.5B-01":
        raise ValueError("Frozen E8 V2 bank source changed")
    if bank.get("model_independent") is not True:
        raise ValueError("The transferred bank must remain model-independent")
    if bank.get("use_all_unique_negatives") is not True:
        raise ValueError("Every unique negative must participate")
    if bank.get("explicit_near_far_training_classes") is not False:
        raise ValueError("Explicit near/far training classes are forbidden")
    if bank.get("extreme_selection_forbidden") is not True:
        raise ValueError("Extreme negative selection is forbidden")

    remoteness = config.get("remoteness", {})
    if remoteness.get("coordinate") != "u=current_sequence_surprisal/2":
        raise ValueError("The transferred remoteness coordinate changed")
    if remoteness.get("weight") != "alpha*exp(-c*max(u-tau,0))":
        raise ValueError("The transferred DRPO weight formula changed")
    if float(remoteness.get("alpha", float("nan"))) != 1.0:
        raise ValueError("The transferred global alpha must remain 1")
    if remoteness.get("detached") is not True:
        raise ValueError("DRPO transfer weights must be detached")
    if remoteness.get("retune_on_3b") is not False:
        raise ValueError("3B c/tau retuning is forbidden")

    sweep = config.get("sweep", {})
    if _configured_points(config) != PARAMETER_POINTS:
        raise ValueError("Frozen 3B c/tau transfer points changed")
    if tuple(int(value) for value in sweep.get("seed_offsets", ())) != SEED_OFFSETS:
        raise ValueError("3B transfer seed offsets changed")
    if int(sweep.get("unique_parameter_points", -1)) != EXPECTED_POINTS:
        raise ValueError("3B scale transfer requires four parameter points")
    if int(sweep.get("cells", -1)) != EXPECTED_CELLS:
        raise ValueError("3B scale transfer requires eight cells")
    if sweep.get("cartesian_product") is not False:
        raise ValueError("3B scale transfer must remain an explicit four-point list")

    training = config.get("training", {})
    if int(training.get("steps", -1)) != 1200:
        raise ValueError("3B scale transfer requires fixed 1200 steps")
    if training.get("early_stop") is not False:
        raise ValueError("Early stopping is forbidden")
    if int(training.get("eval_every", -1)) != 100:
        raise ValueError("Greedy and Pass@8 cadence must remain 100")
    if int(training.get("pass64_every", -1)) != 200:
        raise ValueError("Pass@64 cadence must remain 200")
    if training.get("denominator") != "unique_negative_count_per_prompt":
        raise ValueError("The loss denominator changed")
    if training.get("normalize_by_weight_sum") is not False:
        raise ValueError("Weight-sum normalization is forbidden")
    if training.get("hidden_negative_scale") is not False:
        raise ValueError("Hidden negative scaling is forbidden")
    if training.get("gradient_budget_matching") is not False:
        raise ValueError("Gradient-budget matching is forbidden")

    execution = config.get("execution", {})
    if execution.get("default_gpus") != [0, 1, 2, 3]:
        raise ValueError("3B scale transfer requires GPU 0-3")
    if int(execution.get("parallel_cells_per_gpu", -1)) != 1:
        raise ValueError("3B scale transfer requires one slot per GPU")
    if int(execution.get("expected_full_waves", -1)) != 2:
        raise ValueError("3B scale transfer requires exactly two full waves")
    if execution.get("identity_checked_resume") is not True:
        raise ValueError("Identity-checked resume is required")
    liveness = execution.get("liveness", {})
    if float(liveness.get("representative_c", float("nan"))) != PARAMETER_POINTS[0][1]:
        raise ValueError("3B liveness c must use frozen point A")
    if float(liveness.get("representative_tau", float("nan"))) != PARAMETER_POINTS[0][2]:
        raise ValueError("3B liveness tau must use frozen point A")

    evaluation = config.get("evaluation", {})
    if evaluation.get("split_role") != "structurally_disjoint_held_out_evaluation":
        raise ValueError("3B evaluation split role changed")
    if evaluation.get("enters_training_loss") is not False:
        raise ValueError("Held-out evaluation must not enter training loss")
    if evaluation.get("separate_test_split_access") is not False:
        raise ValueError("Separate test split access must remain false")
    if evaluation.get("primary_reporting_metric") != "late_window_pass_at_8":
        raise ValueError("Primary reporting metric must remain late_window_pass_at_8")
    if evaluation.get("secondary_reporting_metric") != "terminal_pass_at_8":
        raise ValueError("Secondary reporting metric must remain terminal_pass_at_8")
    if evaluation.get("paper_facing_checkpoint_policy") != "late_window_and_terminal":
        raise ValueError("Paper-facing checkpoint policy changed")
    if evaluation.get("best_checkpoint_metric_is_supplementary_only") is not True:
        raise ValueError("Best-checkpoint metrics must remain supplementary only")


def parameter_points(config: Mapping[str, Any]) -> tuple[tuple[str, float, float], ...]:
    validate_grid_config(config)
    return PARAMETER_POINTS


def build_cells(config: Mapping[str, Any]) -> tuple[Cell, ...]:
    points = parameter_points(config)
    cells = tuple(
        Cell(
            label=label,
            coefficient=coefficient,
            tau=tau,
            seed_offset=seed_offset,
        )
        for label, coefficient, tau in points
        for seed_offset in SEED_OFFSETS
    )
    if len(cells) != EXPECTED_CELLS or len({cell.name for cell in cells}) != EXPECTED_CELLS:
        raise AssertionError("3B DRPO c/tau transfer must produce eight unique cells")
    return cells


def _validate_model_identity(model_path: Path) -> dict[str, Any]:
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise RuntimeError(f"3B model identity requires {config_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if payload.get("model_type") != "qwen2":
        raise RuntimeError("3B model config model_type must be qwen2")
    architectures = tuple(str(value) for value in payload.get("architectures", ()))
    if "Qwen2ForCausalLM" not in architectures:
        raise RuntimeError("3B model config must declare Qwen2ForCausalLM")
    identity_fields = (
        model_path.name,
        str(payload.get("_name_or_path", "")),
        str(payload.get("name_or_path", "")),
    )
    if not any(MODEL_IDENTITY.lower() in value.lower() for value in identity_fields):
        raise RuntimeError(
            "3B model identity is not Qwen2.5-3B-Instruct: "
            + ", ".join(value for value in identity_fields if value)
        )
    return {
        "expected": MODEL_IDENTITY,
        "model_type": payload["model_type"],
        "architectures": list(architectures),
        "identity_fields": list(identity_fields),
        "config_sha256": sha256_file(config_path),
    }


def _identity(
    *,
    repo: Path,
    model_path: Path,
    bank: Path,
    val: Path,
    base_config: Path,
    grid_config: Path,
    cell: Cell,
    smoke: bool,
) -> dict[str, Any]:
    config = load_yaml(grid_config)
    validate_grid_config(config)
    package_dir = Path(__file__).resolve().parent
    source_paths = {
        "transfer_profile_and_runtime": Path(__file__).resolve(),
        "base_common": Path(_base.__file__).resolve(),
        "base_trainer": package_dir / "countdown_e8_alpha1_c_scan_trainer.py",
        "base_runtime": package_dir / "countdown_e8_alpha1_c_scan_runtime.py",
        "highc_common": Path(_highc.__file__).resolve(),
        "resource_launcher": repo
        / "scripts"
        / "run_countdown_e8_oracle_offline_v2_alpha1_c_scan_auto.py",
    }
    missing = [name for name, path in source_paths.items() if not path.is_file()]
    if missing:
        raise RuntimeError("3B transfer identity is missing protected sources: " + ", ".join(sorted(missing)))
    return {
        "experiment_id": EXPERIMENT_ID,
        "version": VERSION,
        "source": _base.git_state(repo),
        "source_sha256": {
            name: _base.sha256_file(path) for name, path in sorted(source_paths.items())
        },
        "model_path": str(model_path),
        "model_identity": _validate_model_identity(model_path),
        "bank_sha256": _base.sha256_file(bank),
        "validation_sha256": _base.sha256_file(val),
        "base_config_sha256": _base.sha256_file(base_config),
        "grid_config_sha256": _base.sha256_file(grid_config),
        "cell": {
            "name": cell.name,
            "label": cell.label,
            "method": cell.method,
            "family": cell.family,
            "alpha": cell.alpha,
            "c": cell.c,
            "tau": cell.tau,
            "seed_offset": cell.seed_offset,
        },
        "smoke": bool(smoke),
        "test_data_used": False,
    }


_PATCH_VALUES = {
    "EXPERIMENT_ID": EXPERIMENT_ID,
    "DRPO_CTAU_SCALE_TRANSFER_3B_EXPERIMENT_ID": EXPERIMENT_ID,
    "VERSION": VERSION,
    "DEFAULT_GRID_CONFIG": DEFAULT_GRID_CONFIG,
    "PARAMETER_POINTS": PARAMETER_POINTS,
    "SEED_OFFSETS": SEED_OFFSETS,
    "EXPECTED_POINTS": EXPECTED_POINTS,
    "EXPECTED_CELLS": EXPECTED_CELLS,
    "Cell": Cell,
    "continuous_exp_weights": continuous_exp_weights,
    "validate_grid_config": validate_grid_config,
    "parameter_points": parameter_points,
    "build_cells": build_cells,
    "_identity": _identity,
    "set_active_tau": set_active_tau,
    "MODEL_IDENTITY": MODEL_IDENTITY,
}


def _apply() -> None:
    for target in (_base, _highc):
        for name, value in _PATCH_VALUES.items():
            setattr(target, name, value)


def activate() -> None:
    set_active_tau(0.0)
    _apply()


@contextmanager
def activated() -> Iterator[None]:
    previous: list[tuple[Any, dict[str, Any], set[str]]] = []
    for target in (_base, _highc):
        values = {name: getattr(target, name, None) for name in _PATCH_VALUES}
        missing = {name for name in _PATCH_VALUES if not hasattr(target, name)}
        previous.append((target, values, missing))
    previous_tau = _ACTIVE_TAU
    activate()
    try:
        yield
    finally:
        set_active_tau(previous_tau)
        for target, values, missing in previous:
            for name, value in values.items():
                if name in missing:
                    delattr(target, name)
                else:
                    setattr(target, name, value)


def _semantic_payload(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    payload = dict(EVALUATION_SEMANTICS)
    if config is not None:
        evaluation = config.get("evaluation", {})
        payload.update(
            {
                "evaluation_split_file": str(
                    evaluation.get("split_file", payload["evaluation_split_file"])
                ),
                "evaluation_split_role": str(
                    evaluation.get("split_role", payload["evaluation_split_role"])
                ),
                "evaluation_enters_training_loss": bool(
                    evaluation.get(
                        "enters_training_loss",
                        payload["evaluation_enters_training_loss"],
                    )
                ),
                "paper_facing_checkpoint_policy": str(
                    evaluation.get(
                        "paper_facing_checkpoint_policy",
                        payload["paper_facing_checkpoint_policy"],
                    )
                ),
                "best_checkpoint_role": str(
                    evaluation.get("best_checkpoint_role", payload["best_checkpoint_role"])
                ),
            }
        )
    return payload


def _load_runtime():
    activate()
    from drpo import countdown_e8_alpha1_c_scan_runtime as runtime

    return runtime


def _core_worker_command(args: argparse.Namespace, cell: Cell, output_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "worker",
        "--model_path",
        args.model_path,
        "--bank",
        args.bank,
        "--val",
        args.val,
        "--base_config",
        args.base_config,
        "--grid_config",
        args.grid_config,
        "--output_dir",
        str(output_dir),
        "--label",
        cell.label,
        "--c",
        str(cell.c),
        "--tau",
        str(cell.tau),
        "--seed_offset",
        str(cell.seed_offset),
    ]


def _augment_plan(work_dir: Path, config: Mapping[str, Any]) -> None:
    path = work_dir / "SWEEP_PLAN.json"
    if not path.is_file():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(_semantic_payload(config))
    payload["model_identity"] = MODEL_IDENTITY
    payload["resource_contract"] = {
        "gpu_count": 4,
        "runtime_slots_per_gpu": 1,
        "total_runtime_slots": 4,
        "expected_full_waves": 2,
    }
    by_name = {cell.name: cell for cell in build_cells(config)}
    for row in payload.get("cells", []):
        cell = by_name[row["name"]]
        row.update({"label": cell.label, "tau": cell.tau})
    atomic_json(path, payload)


def _augment_csv_constants(path: Path, constants: Mapping[str, Any]) -> None:
    if not path.is_file():
        return
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return
    for row in rows:
        row.update({key: str(value) for key, value in constants.items()})
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _augment_jsonl_constants(path: Path, constants: Mapping[str, Any]) -> None:
    if not path.is_file():
        return
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        row.update(constants)
        rows.append(row)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _augment_cell_summary(output_dir: Path, cell: Cell, summary: dict[str, Any]) -> dict[str, Any]:
    constants = {"ctau_label": cell.label, "tau": cell.tau}
    summary.update(constants)
    summary.update(_semantic_payload(load_yaml(DEFAULT_GRID_CONFIG)))
    terminal = dict(summary.get("terminal_metrics", {}))
    terminal.update(constants)
    summary["terminal_metrics"] = terminal
    reporting = dict(summary.get("reporting_separation", {}))
    reporting["task_performance"] = (
        "predeclared late-window and terminal held-out metrics; "
        "metric-specific best values are supplementary only"
    )
    summary["reporting_separation"] = reporting
    summary["best_checkpoint_saved"] = True
    atomic_json(output_dir / "manifest.json", summary)
    atomic_json(output_dir / "summary.json", summary)
    _augment_csv_constants(output_dir / "metrics.csv", constants)
    _augment_jsonl_constants(output_dir / "validation_diagnostics.jsonl", constants)
    _augment_jsonl_constants(output_dir / "training_diagnostics.jsonl", constants)
    return summary


def _augment_aggregate(work_dir: Path, config: Mapping[str, Any]) -> None:
    aggregate_dir = work_dir / "aggregate"
    csv_path = aggregate_dir / "per_cell_summary.csv"
    if csv_path.is_file():
        with csv_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        by_name = {cell.name: cell for cell in build_cells(config)}
        for row in rows:
            cell = by_name[row["cell"]]
            row.update(
                {
                    "ctau_label": cell.label,
                    "tau": str(cell.tau),
                    "evaluation_split_role": EVALUATION_SEMANTICS[
                        "evaluation_split_role"
                    ],
                    "evaluation_enters_training_loss": "false",
                    "paper_facing_checkpoint_policy": EVALUATION_SEMANTICS[
                        "paper_facing_checkpoint_policy"
                    ],
                    "best_checkpoint_role": EVALUATION_SEMANTICS[
                        "best_checkpoint_role"
                    ],
                    "separate_test_jsonl_used": "false",
                }
            )
        if rows:
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    audit_path = aggregate_dir / "terminal_audit.json"
    if audit_path.is_file():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        audit.update(_semantic_payload(config))
        audit["task_performance_status"] = (
            "late_window_and_terminal_reported_not_adjudicated"
        )
        audit["resource_contract"] = {
            "gpu_count": 4,
            "runtime_slots_per_gpu": 1,
            "total_runtime_slots": 4,
            "expected_full_waves": 2,
        }
        atomic_json(audit_path, audit)


def core_plan(args: argparse.Namespace) -> int:
    runtime = _load_runtime()
    config = load_yaml(args.grid_config)
    validate_grid_config(config)
    result = runtime.plan(args)
    _augment_plan(Path(args.work_dir).resolve(), config)
    return result


def core_smoke(args: argparse.Namespace) -> int:
    runtime = _load_runtime()
    repo = _REPO_ROOT
    model, bank, val, base_config, grid_config = runtime._required_inputs(args)
    config = load_yaml(grid_config)
    validate_grid_config(config)
    liveness = config["execution"]["liveness"]
    cell = Cell(
        label=str(liveness["representative_label"]),
        coefficient=float(liveness["representative_c"]),
        tau=float(liveness["representative_tau"]),
        seed_offset=SEED_OFFSETS[0],
    )
    set_active_tau(cell.tau)
    output_dir = Path(args.work_dir).resolve() / "_liveness" / cell.name
    if output_dir.exists() and not (output_dir / "summary.json").exists():
        shutil.rmtree(output_dir)
    try:
        summary = runtime.train_cell(
            cell=cell,
            model_path=model,
            bank=bank,
            val=val,
            base_config_path=base_config,
            grid_config_path=grid_config,
            output_dir=output_dir,
            repo=repo,
            smoke=True,
        )
        summary = _augment_cell_summary(output_dir, cell, summary)
        passed = summary.get("numerical_failure") is None and int(
            summary.get("terminal_step", -1)
        ) == int(liveness["steps"])
        gate = {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "registration_state": str(config["registration_state"]),
            "status": "PASS" if passed else "FAIL",
            "scientific_evidence": False,
            "cell": cell.name,
            "summary": str(output_dir / "summary.json"),
            "run_identity": summary.get("run_identity"),
            "resource_contract": {
                "gpu_count": 4,
                "runtime_slots_per_gpu": 1,
                "expected_full_waves": 2,
            },
            "test_data_used": False,
            **_semantic_payload(config),
        }
    except BaseException as error:
        gate = {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "registration_state": str(config["registration_state"]),
            "status": "FAIL",
            "scientific_evidence": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "test_data_used": False,
            **_semantic_payload(config),
        }
        atomic_json(Path(args.work_dir).resolve() / "SMOKE_GATE.json", gate)
        raise
    atomic_json(Path(args.work_dir).resolve() / "SMOKE_GATE.json", gate)
    return 0 if gate["status"] == "PASS" else 1


def core_run(args: argparse.Namespace) -> int:
    runtime = _load_runtime()
    config = load_yaml(args.grid_config)
    validate_grid_config(config)
    if int(args.runtime_slots_per_gpu) != 1:
        raise ValueError("3B scale transfer requires --runtime-slots-per-gpu=1")
    gpu_pool = [value for value in args.gpus.split(",") if value]
    if len(gpu_pool) != 4 or len(set(gpu_pool)) != 4:
        raise ValueError("3B scale transfer requires exactly four unique GPUs")
    runtime._worker_command = _core_worker_command
    result = runtime.run(args)
    work_dir = Path(args.work_dir).resolve()
    _augment_plan(work_dir, config)
    _augment_aggregate(work_dir, config)
    complete_path = work_dir / "SWEEP_COMPLETE.json"
    if complete_path.is_file():
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        complete.update(_semantic_payload(config))
        complete["model_identity"] = MODEL_IDENTITY
        complete["resource_contract"] = {
            "gpu_count": 4,
            "runtime_slots_per_gpu": 1,
            "total_runtime_slots": 4,
            "expected_full_waves": 2,
        }
        atomic_json(complete_path, complete)
    return result


def worker(args: argparse.Namespace) -> int:
    runtime = _load_runtime()
    cell = Cell(
        label=str(args.label),
        coefficient=float(args.c),
        tau=float(args.tau),
        seed_offset=int(args.seed_offset),
    )
    config = load_yaml(args.grid_config)
    if cell not in build_cells(config):
        raise ValueError(f"Worker cell is outside the frozen grid: {cell}")
    set_active_tau(cell.tau)
    output_dir = Path(args.output_dir).resolve()
    summary = runtime.train_cell(
        cell=cell,
        model_path=Path(args.model_path).resolve(),
        bank=Path(args.bank).resolve(),
        val=Path(args.val).resolve(),
        base_config_path=Path(args.base_config).resolve(),
        grid_config_path=Path(args.grid_config).resolve(),
        output_dir=output_dir,
        repo=_REPO_ROOT,
        smoke=False,
    )
    summary = _augment_cell_summary(output_dir, cell, summary)
    return 0 if summary.get("numerical_failure") is None else 1


def _load_auto_launcher():
    activate()
    path = _REPO_ROOT / "scripts" / "run_countdown_e8_oracle_offline_v2_alpha1_c_scan_auto.py"
    spec = importlib.util.spec_from_file_location("_e8_3b_transfer_auto_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load reviewed resource launcher: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _auto_main(tokens: list[str]) -> int:
    auto = _load_auto_launcher()
    original_selection = auto._selection

    def selection(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
        document = original_selection(args, config)
        document["limitations"] = [
            "fixed_one_process_per_gpu_for_qwen2p5_3b_bf16_lora",
            "configured_vram_floor_precedes_actual_two_step_liveness",
            "smoke_probes_one_worker_before_two_wave_full_run",
            "countdown_external_validity_pilot_not_method_ranking",
        ]
        atomic_json(Path(args.work_dir).resolve() / "RUNTIME_SELECTION.json", document)
        return document

    def core_command(
        args: argparse.Namespace,
        command: str,
        *,
        selected_ids: list[str],
    ) -> list[str]:
        if len(selected_ids) != 4:
            raise RuntimeError("3B DRPO c/tau scale transfer requires exactly four GPUs")
        result = [
            sys.executable,
            str(Path(__file__).resolve()),
            f"_core-{command}",
            "--model_path",
            str(Path(args.model_path).resolve()),
            "--work_dir",
            str(Path(args.work_dir).resolve()),
            "--bank",
            str(Path(args.bank).resolve()),
            "--val",
            str(Path(args.val).resolve()),
            "--base_config",
            str(Path(args.base_config).resolve()),
            "--grid_config",
            str(Path(args.grid_config).resolve()),
        ]
        if command == "run":
            result.extend(
                [
                    "--gpus",
                    ",".join(selected_ids),
                    "--runtime-slots-per-gpu",
                    "1",
                ]
            )
        return result

    auto.ADAPTER_ID = "e8_drpo_ctau_scale_transfer_3b_cuda_dev_v1"
    auto._selection = selection
    auto._core_command = core_command
    return int(auto.main(tokens))


def _core_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(subparser: argparse.ArgumentParser, *, include_work_dir: bool = True) -> None:
        subparser.add_argument("--model_path", required=True)
        subparser.add_argument("--bank", required=True)
        subparser.add_argument("--val", required=True)
        subparser.add_argument("--base_config", required=True)
        subparser.add_argument("--grid_config", required=True)
        if include_work_dir:
            subparser.add_argument("--work_dir", required=True)

    common(subparsers.add_parser("_core-plan"))
    common(subparsers.add_parser("_core-smoke"))
    run_parser = subparsers.add_parser("_core-run")
    common(run_parser)
    run_parser.add_argument("--gpus", required=True)
    run_parser.add_argument("--runtime-slots-per-gpu", type=int, required=True)
    worker_parser = subparsers.add_parser("worker")
    common(worker_parser, include_work_dir=False)
    worker_parser.add_argument("--output_dir", required=True)
    worker_parser.add_argument("--label", required=True)
    worker_parser.add_argument("--c", type=float, required=True)
    worker_parser.add_argument("--tau", type=float, required=True)
    worker_parser.add_argument("--seed_offset", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    tokens = list(sys.argv[1:] if argv is None else argv)
    if not tokens:
        raise SystemExit("A command is required")
    if tokens[0] in {"plan", "smoke", "run"}:
        return _auto_main(tokens)
    args = _core_parser().parse_args(tokens)
    if args.command == "_core-plan":
        return core_plan(args)
    if args.command == "_core-smoke":
        return core_smoke(args)
    if args.command == "_core-run":
        return core_run(args)
    if args.command == "worker":
        return worker(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
