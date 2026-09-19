"""Paper-aligned multitask cold-start baseline orchestration on frozen banks.

The module keeps the P0 occurrence/gradient diagnostic separate from downstream
method tuning.  EXP, AsymRE, and TOPR dispatch to the byte-locked paper trainers;
DPO ports the audited PR #268 semantics because no canonical DPO runtime is merged
on main.  Task adapters may change only the explicitly governed task-interface
fields.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from drpo import e8_experiment_config as experiment_config
from drpo import e8_multitask_canonical_bridge as e8_canonical_bridge
from drpo import e8_multitask_inputs as e8_inputs
from drpo import e8_multitask_orchestration as e8_orchestration
from drpo import e8_multitask_results as e8_results
from drpo import e8_multitask_runtime as e8_runtime
from drpo import e8_multitask_selftest as e8_selftest
from drpo import e8_multitask_warmstart_training as e8_warmstart
from drpo.e8_multitask_inputs import TaskInputs


def _canonical_bridge() -> e8_canonical_bridge.CanonicalBridge:
    return e8_canonical_bridge.build_bridge(
        e8_canonical_bridge.CanonicalBridgeBindings(host=sys.modules[__name__])
    )


def _selftest_bindings() -> e8_selftest.SelfTestBindings:
    return e8_selftest.SelfTestBindings(host=sys.modules[__name__])

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
except ImportError:  # Planning and split validation do not require Torch.
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]

from drpo.e8_multitask_p0 import (
    append_jsonl,
    atomic_json,
    atomic_jsonl,
    model_identity,
    read_jsonl,
    resolve_torch_dtype,
    sha256_file,
    stable_config_hash,
    train_task_positive_warmstart,
    validate_work_dir,
)
from drpo.e8_multitask_tasks import (
    TASK_NAMES,
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
CANONICAL_ASYMRE_GRID = Path(
    "configs/countdown_e8_oracle_offline_v2_asymre_deltav_scan_0p5b.yaml"
)
CANONICAL_TOPR_GRID = Path(
    "configs/countdown_e8_oracle_offline_v2_joint_fitted_reference_beta_topr_dense_0p5b.yaml"
)
CANONICAL_RECIPROCAL_GRID = Path(
    "configs/countdown_e8_oracle_offline_v2_reciprocal_shape_screen_0p5b.yaml"
)
METHOD_POSITIVE_ONLY = "positive_only"
METHOD_EXPONENTIAL = experiment_config.COLDSTART_METHOD_EXPONENTIAL
METHOD_ASYMRE = experiment_config.COLDSTART_METHOD_ASYMRE
METHOD_TOPR = experiment_config.COLDSTART_METHOD_TOPR
METHOD_DPO = experiment_config.COLDSTART_METHOD_DPO
METHOD_RECIPROCAL_LINEAR = experiment_config.COLDSTART_METHOD_RECIPROCAL_LINEAR
METHOD_RECIPROCAL_QUADRATIC = experiment_config.COLDSTART_METHOD_RECIPROCAL_QUADRATIC
METHOD_BASELINE_MATRIX = experiment_config.COLDSTART_METHOD_BASELINE_MATRIX
METHOD_RECIPROCAL_MATRIX = experiment_config.COLDSTART_METHOD_RECIPROCAL_MATRIX
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
    delta_v: float | None = None
    beta: float | None = None
    dpo_initialization: str | None = None
    method_parameters: Mapping[str, Any] | None = None

    @property
    def key(self) -> str:
        return _method_spec(self.method).cell_key(self)


def _default_method_audit(
    cell: Cell, record: Mapping[str, Any]
) -> e8_runtime.MethodAuditResult:
    del cell, record
    return e8_runtime.MethodAuditResult(True)


def _empty_metadata(config: Mapping[str, Any]) -> Mapping[str, Any]:
    del config
    return {}


@dataclass(frozen=True)
class PaperRuntimeSpec:
    """Optional capability for methods using the canonical paper runtime."""

    liveness_grid: Callable[[Mapping[str, Any], Mapping[str, Any]], Path]
    liveness_parameter: str
    grid_paths: Callable[
        [Mapping[str, Any], Mapping[str, Any], Cell], tuple[Path, Path]
    ]
    cell_parameters: Callable[[Cell], tuple[str, float, float]]
    formula: str


@dataclass(frozen=True)
class MethodSpec:
    """One scientific-method adapter; generic infrastructure never branches on names."""

    name: str
    build_cell: Callable[..., Cell]
    cell_key: Callable[[Cell], str]
    parameters: Callable[[Cell], Mapping[str, Any]]
    cell_initialization: Callable[[Mapping[str, Any]], str | None]
    initialization_identity: Callable[[Mapping[str, Any]], Mapping[str, Any]]
    train_cold: Callable[..., dict[str, Any]]
    liveness_task: Callable[[Mapping[str, Any]], str]
    liveness_runner: Callable[..., dict[str, Any]]
    scientific_kernel: str
    paper_runtime: PaperRuntimeSpec | None = None
    audit_record: Callable[
        [Cell, Mapping[str, Any]], e8_runtime.MethodAuditResult
    ] = _default_method_audit
    single_aggregate_metadata: Callable[
        [Mapping[str, Any]], Mapping[str, Any]
    ] = _empty_metadata
    matrix_aggregate_metadata: Callable[
        [Mapping[str, Any]], Mapping[str, Any]
    ] = _empty_metadata



_METHOD_SPECS: dict[str, MethodSpec] = {}


def _register_method_spec(spec: MethodSpec, *, replace: bool = False) -> None:
    if spec.name in _METHOD_SPECS and not replace:
        raise ValueError(f"Method spec already registered: {spec.name}")
    _METHOD_SPECS[spec.name] = spec


def _method_spec(method: str) -> MethodSpec:
    try:
        return _METHOD_SPECS[method]
    except KeyError as exc:
        raise ValueError(f"No E8 method spec registered for {method}") from exc


def _cell_lambda(cell: Cell) -> float | None:
    if cell.lambda_value is not None:
        return float(cell.lambda_value)
    return None if cell.rho is None else coefficient_from_rho(float(cell.rho))


def _legacy_compatibility_columns(cell: Cell) -> Mapping[str, Any]:
    return {
        "delta_v": cell.delta_v,
        "beta": cell.beta,
        "dpo_initialization": cell.dpo_initialization,
        "rho": cell.rho,
        "lambda": _cell_lambda(cell),
    }

def _method_output_columns(cell: Cell) -> dict[str, Any]:
    """Merge frozen legacy columns with the method's single parameter source."""

    columns = dict(_legacy_compatibility_columns(cell))
    for name, value in _method_spec(cell.method).parameters(cell).items():
        if name in columns and columns[name] != value:
            raise ValueError(
                f"Method parameter {name!r} conflicts with its legacy cell value"
            )
        columns[name] = value
    return columns



def _positive_key(cell: Cell) -> str:
    return f"{cell.task}__positive_only__seed{cell.seed}"


def _global_key(cell: Cell) -> str:
    return f"{cell.task}__global__seed{cell.seed}"


def _exp_key(cell: Cell) -> str:
    if cell.lambda_value is not None:
        tag = f"{cell.lambda_value:.12g}".replace(".", "p")
        return f"{cell.task}__exp_lambda{tag}__seed{cell.seed}"
    if cell.rho is None:
        raise AssertionError("Exponential cell requires rho or lambda")
    tag = f"{cell.rho:.6f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"{cell.task}__exp_rho{tag}__seed{cell.seed}"


def _reciprocal_key(cell: Cell) -> str:
    if cell.lambda_value is None:
        raise AssertionError("Reciprocal cell requires lambda")
    tag = f"{cell.lambda_value:.12g}".replace(".", "p")
    return f"{cell.task}__{cell.method}_lambda{tag}__seed{cell.seed}"


def _asymre_key(cell: Cell) -> str:
    if cell.delta_v is None:
        raise AssertionError("AsymRE cell requires delta_v")
    tag = f"{cell.delta_v:.12g}".replace("-", "m").replace(".", "p")
    return f"{cell.task}__asymre_delta_v{tag}__seed{cell.seed}"


def _topr_key(cell: Cell) -> str:
    if cell.beta is None:
        raise AssertionError("Joint Fitted-Reference TOPR cell requires beta")
    tag = f"{cell.beta:.12g}".replace("-", "m").replace(".", "p")
    return f"{cell.task}__joint_fitted_reference_topr_beta{tag}__seed{cell.seed}"


def _dpo_key(cell: Cell) -> str:
    if cell.beta is None:
        raise AssertionError("Canonical DPO cell requires beta")
    if cell.dpo_initialization is None:
        raise AssertionError("Canonical DPO cell requires initialization identity")
    tag = f"{cell.beta:.12g}".replace("-", "m").replace(".", "p")
    return (
        f"{cell.task}__canonical_dpo_beta{tag}__"
        f"init_{cell.dpo_initialization}__seed{cell.seed}"
    )


def _build_control_cell(
    *, task: str, method: str, seed: int, stage: str, value: float,
    lambda_only: bool, dpo_initialization: str | None,
) -> Cell:
    del lambda_only, dpo_initialization
    if method == METHOD_POSITIVE_ONLY:
        return Cell(task, method, None, seed, stage)
    if method == METHOD_GLOBAL:
        return Cell(task, method, 1.0, seed, stage, 0.0)
    raise ValueError(f"Unsupported control method: {method}")


def _build_exponential_cell(
    *, task: str, method: str, seed: int, stage: str, value: float,
    lambda_only: bool, dpo_initialization: str | None,
) -> Cell:
    del dpo_initialization
    rho = None if lambda_only else math.exp(-value)
    return Cell(task, method, rho, seed, stage, value)


def _build_reciprocal_cell(
    *, task: str, method: str, seed: int, stage: str, value: float,
    lambda_only: bool, dpo_initialization: str | None,
) -> Cell:
    del lambda_only, dpo_initialization
    return Cell(task, method, None, seed, stage, value)


def _build_asymre_cell(
    *, task: str, method: str, seed: int, stage: str, value: float,
    lambda_only: bool, dpo_initialization: str | None,
) -> Cell:
    del lambda_only, dpo_initialization
    return Cell(task, method, None, seed, stage, delta_v=value)


def _build_topr_cell(
    *, task: str, method: str, seed: int, stage: str, value: float,
    lambda_only: bool, dpo_initialization: str | None,
) -> Cell:
    del lambda_only, dpo_initialization
    return Cell(task, method, None, seed, stage, beta=value)


def _build_dpo_cell(
    *, task: str, method: str, seed: int, stage: str, value: float,
    lambda_only: bool, dpo_initialization: str | None,
) -> Cell:
    del lambda_only
    if dpo_initialization is None:
        raise ValueError("DPO cell construction requires initialization identity")
    return Cell(
        task, method, None, seed, stage, beta=value,
        dpo_initialization=dpo_initialization,
    )


def _default_initialization_identity(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return dict(config["initialization"])


def _dpo_initialization_identity(config: Mapping[str, Any]) -> Mapping[str, Any]:
    if _is_baseline_matrix(config):
        return {
            "source": str(config["dpo"]["initialization_mode"]),
            "shared_sft_adapter_env": config["dpo"].get("shared_sft_adapter_env"),
        }
    return dict(config["initialization"])


def _cold_train_paper(cell: Cell, **kwargs: Any) -> dict[str, Any]:
    updates_override = kwargs.pop("updates_override", None)
    engineering_liveness = bool(kwargs.pop("engineering_liveness", False))
    if updates_override is not None or engineering_liveness:
        raise RuntimeError(
            "Canonical cold liveness requires its derived old-core config path"
        )
    return _train_canonical_cold_cell(cell, **kwargs)


def _cold_train_dpo(cell: Cell, **kwargs: Any) -> dict[str, Any]:
    return _train_canonical_dpo_transfer_cell(cell, **kwargs)




def _run_canonical_method_liveness(
    method: str,
    *,
    config: Mapping[str, Any],
    config_path: Path,
    output_root: Path,
    inputs: Mapping[str, TaskInputs],
    splits: Mapping[str, Any],
    base_model_path: str,
    task: str,
    force: bool,
) -> dict[str, Any]:
    if task != "countdown":
        raise RuntimeError(f"{method} liveness anchor must be Countdown")
    return _cmd_canonical_cold_liveness(
        config,
        config_path,
        output_root,
        inputs=inputs["countdown"],
        splits=splits,
        base_model_path=base_model_path,
        force=force,
        method=method,
    )


def _run_dpo_method_liveness(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    output_root: Path,
    inputs: Mapping[str, TaskInputs],
    splits: Mapping[str, Any],
    base_model_path: str,
    task: str,
    force: bool,
) -> dict[str, Any]:
    return _cmd_dpo_liveness(
        config,
        config_path,
        output_root,
        inputs=inputs,
        splits=splits,
        base_model_path=base_model_path,
        task=task,
        force=force,
    )


def _dpo_method_audit(
    cell: Cell, record: Mapping[str, Any]
) -> e8_runtime.MethodAuditResult:
    del cell
    failures: list[str] = []
    if record.get("reference_initial_state_sha256") != record.get(
        "reference_terminal_state_sha256"
    ):
        failures.append("reference_state_changed")
    if record.get("reference_trainable") is not False:
        failures.append("reference_trainable")
    return e8_runtime.MethodAuditResult(
        passed=not failures,
        failure_bucket="dpo_reference_identity_failures",
        failures=tuple(failures),
    )



def _asymre_single_metadata(config: Mapping[str, Any]) -> Mapping[str, Any]:
    del config
    identity = _canonical_baseline_grid_identity(METHOD_ASYMRE)
    return {
        "canonical_asymre_grid": identity["canonical_grid"],
        "canonical_asymre_grid_sha256": identity["canonical_grid_sha256"],
        "canonical_grid_identity_policy": identity["identity_policy"],
        "canonical_grid_expected_git_blob_gate": identity[
            "expected_git_blob_gate"
        ],
    }


def _topr_single_metadata(config: Mapping[str, Any]) -> Mapping[str, Any]:
    del config
    identity = _canonical_baseline_grid_identity(METHOD_TOPR)
    return {
        "canonical_topr_grid": identity["canonical_grid"],
        "canonical_topr_grid_sha256": identity["canonical_grid_sha256"],
        "canonical_grid_identity_policy": identity["identity_policy"],
        "canonical_grid_expected_git_blob_gate": identity[
            "expected_git_blob_gate"
        ],
    }


def _dpo_single_metadata(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return {"dpo_initialization_mode": str(config["dpo"]["initialization_mode"])}


def _dpo_matrix_metadata(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "initialization_mode": str(config["dpo"]["initialization_mode"]),
        "shared_sft_adapter_contract_hash": (
            stable_hash(config["dpo"]["shared_sft_adapter_contract"])
            if config["dpo"]["initialization_mode"] == "shared_sft_adapter"
            else None
        ),
        "semantics_source": (
            "historical_PR_268_protected_implementation_"
            "cc0ead2be00c89a3c35296b7adc1ddeae8d14759"
        ),
    }




def _register_builtin_method_specs() -> None:
    exponential_paper_runtime = PaperRuntimeSpec(
        liveness_grid=lambda config, record: Path(str(record["round1_grid"])),
        liveness_parameter="representative_c",
        grid_paths=lambda *args, **kwargs: (
            _canonical_bridge()._paper_grid_paths_exponential(*args, **kwargs)
        ),
        cell_parameters=lambda *args, **kwargs: (
            _canonical_bridge()._paper_params_exponential(*args, **kwargs)
        ),
        formula="alpha*exp(-c*(current_sequence_surprisal/2))",
    )
    for method, build_cell, cell_key, parameters in (
        (METHOD_POSITIVE_ONLY, _build_control_cell, _positive_key, lambda cell: {}),
        (
            METHOD_GLOBAL,
            _build_control_cell,
            _global_key,
            lambda cell: {"lambda": 0.0},
        ),
        (
            METHOD_EXPONENTIAL,
            _build_exponential_cell,
            _exp_key,
            lambda cell: {"lambda": _cell_lambda(cell), "rho": cell.rho},
        ),
    ):
        _register_method_spec(
            MethodSpec(
                name=method,
                build_cell=build_cell,
                cell_key=cell_key,
                parameters=parameters,
                cell_initialization=lambda config: None,
                initialization_identity=_default_initialization_identity,
                train_cold=_cold_train_paper,
                liveness_task=lambda config: "countdown",
                liveness_runner=lambda _method=method, **kwargs: (
                    _run_canonical_method_liveness(_method, **kwargs)
                ),
                scientific_kernel="canonical_old_coldstart_imports",
                paper_runtime=exponential_paper_runtime,
            )
        )

    reciprocal_formulas = {
        METHOD_RECIPROCAL_LINEAR: (
            "1/(1+lambda*sqrt(relu(current_sequence_surprisal/2-0.125)))"
        ),
        METHOD_RECIPROCAL_QUADRATIC: (
            "1/(1+lambda*relu(current_sequence_surprisal/2-0.125))"
        ),
    }
    for method, formula in reciprocal_formulas.items():
        _register_method_spec(
            MethodSpec(
                name=method,
                build_cell=_build_reciprocal_cell,
                cell_key=_reciprocal_key,
                parameters=lambda cell: {"lambda": _cell_lambda(cell)},
                cell_initialization=lambda config: None,
                initialization_identity=_default_initialization_identity,
                train_cold=_cold_train_paper,
                liveness_task=lambda config: "countdown",
                liveness_runner=lambda _method=method, **kwargs: (
                    _run_canonical_method_liveness(_method, **kwargs)
                ),
                scientific_kernel="canonical_old_coldstart_imports",
                paper_runtime=PaperRuntimeSpec(
                    liveness_grid=lambda config, record: _canonical_reciprocal_grid_path(),
                    liveness_parameter="representative_c",
                    grid_paths=lambda config, record, cell: (_canonical_reciprocal_grid_path(),) * 2,
                    cell_parameters=lambda *args, **kwargs: (
                _canonical_bridge()._paper_params_reciprocal(*args, **kwargs)
            ),
                    formula=formula,
                ),
            )
        )

    _register_method_spec(
        MethodSpec(
            name=METHOD_ASYMRE,
            build_cell=_build_asymre_cell,
            cell_key=_asymre_key,
            parameters=lambda cell: {"delta_v": cell.delta_v},
            cell_initialization=lambda config: None,
            initialization_identity=_default_initialization_identity,
            train_cold=_cold_train_paper,
            liveness_task=lambda config: "countdown",
            liveness_runner=lambda **kwargs: _run_canonical_method_liveness(
                METHOD_ASYMRE, **kwargs
            ),
            scientific_kernel="canonical_old_coldstart_imports",
            paper_runtime=PaperRuntimeSpec(
                liveness_grid=lambda config, record: _canonical_asymre_grid_path(),
                liveness_parameter="representative_delta_v",
                grid_paths=lambda config, record, cell: (
                    _canonical_asymre_grid_path(),
                    _canonical_asymre_grid_path(),
                ),
                cell_parameters=lambda *args, **kwargs: (
                _canonical_bridge()._paper_params_asymre(*args, **kwargs)
            ),
                formula="delegated_to_existing_canonical_asymre",
            ),
            single_aggregate_metadata=_asymre_single_metadata,
            matrix_aggregate_metadata=lambda config: (
                _canonical_baseline_grid_identity(METHOD_ASYMRE)
            ),
        )
    )
    _register_method_spec(
        MethodSpec(
            name=METHOD_TOPR,
            build_cell=_build_topr_cell,
            cell_key=_topr_key,
            parameters=lambda cell: {"beta": cell.beta},
            cell_initialization=lambda config: None,
            initialization_identity=_default_initialization_identity,
            train_cold=_cold_train_paper,
            liveness_task=lambda config: "countdown",
            liveness_runner=lambda **kwargs: _run_canonical_method_liveness(
                METHOD_TOPR, **kwargs
            ),
            scientific_kernel="canonical_old_coldstart_imports",
            paper_runtime=PaperRuntimeSpec(
                liveness_grid=lambda config, record: _canonical_topr_grid_path(),
                liveness_parameter="representative_c",
                grid_paths=lambda config, record, cell: (
                    _canonical_topr_grid_path(),
                    _canonical_topr_grid_path(),
                ),
                cell_parameters=lambda *args, **kwargs: (
                _canonical_bridge()._paper_params_topr(*args, **kwargs)
            ),
                formula="delegated_to_existing_joint_fitted_reference_beta_topr",
            ),
            single_aggregate_metadata=_topr_single_metadata,
            matrix_aggregate_metadata=lambda config: (
                _canonical_baseline_grid_identity(METHOD_TOPR)
            ),
        )
    )
    _register_method_spec(
        MethodSpec(
            name=METHOD_DPO,
            build_cell=_build_dpo_cell,
            cell_key=_dpo_key,
            parameters=lambda cell: {"beta": cell.beta},
            cell_initialization=lambda config: str(
                config["dpo"]["initialization_mode"]
            ),
            initialization_identity=_dpo_initialization_identity,
            train_cold=_cold_train_dpo,
            liveness_task=lambda config: str(config["dpo"]["liveness_task"]),
            liveness_runner=_run_dpo_method_liveness,
            scientific_kernel=(
                "historical_pr268_semantics_port_in_existing_multitask_runner"
            ),
            audit_record=_dpo_method_audit,
            single_aggregate_metadata=_dpo_single_metadata,
            matrix_aggregate_metadata=_dpo_matrix_metadata,
        )
    )



_register_builtin_method_specs()





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


def _execution_class(config: Mapping[str, Any]) -> str:
    value = str(config.get("execution_class", "pilot"))
    if value not in {"pilot", "formal"}:
        raise ValueError("execution_class must be pilot or formal")
    return value


def _audited_scientific_status(config: Mapping[str, Any], complete: bool) -> str:
    if _is_engineering_self_test(config):
        return "not_run"
    return "finite_step_validated" if _execution_class(config) == "formal" and complete else "pilot"


def _uses_task_lambdas(config: Mapping[str, Any]) -> bool:
    return _is_dense(config) or _is_coldstart(config)


def _dense_tasks() -> set[str]:
    return set(TASK_NAMES) - {"countdown", "spiral_matrix"}


def _task_lambdas(config: Mapping[str, Any], task: str) -> tuple[float, ...]:
    if not _uses_task_lambdas(config):
        raise ValueError("Task-local lambdas are not defined for this profile")
    return experiment_config.task_lambdas(config, task)


def _coldstart_method(config: Mapping[str, Any]) -> str:
    if not _is_coldstart(config):
        raise ValueError("Cold-start method is defined only for the cold-start profile")
    return experiment_config.coldstart_method(config)


def _coldstart_methods(config: Mapping[str, Any]) -> tuple[str, ...]:
    if not _is_coldstart(config):
        raise ValueError("Cold-start methods are defined only for the cold-start profile")
    return experiment_config.coldstart_methods(config)


def _is_baseline_matrix(config: Mapping[str, Any]) -> bool:
    return _is_coldstart(config) and _coldstart_method(config) == METHOD_BASELINE_MATRIX


def _is_method_matrix(config: Mapping[str, Any]) -> bool:
    return _is_coldstart(config) and _coldstart_method(config) in {
        METHOD_BASELINE_MATRIX, METHOD_RECIPROCAL_MATRIX
    }


def _task_rhos(config: Mapping[str, Any], task: str) -> tuple[float, ...]:
    if _uses_task_lambdas(config):
        return tuple(math.exp(-value) for value in _task_lambdas(config, task))
    return _tuple_floats(config["sweep"]["all_rho"])


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


def _canonical_cold_modules(*args: Any, **kwargs: Any) -> Any:
    return _canonical_bridge()._canonical_cold_modules(*args, **kwargs)


def _activate_paper_grid_modules(*args: Any, **kwargs: Any) -> Any:
    return _canonical_bridge()._activate_paper_grid_modules(*args, **kwargs)


normalized_distance = e8_warmstart.normalized_distance
taper_weight = e8_warmstart.taper_weight


def _coldstart_method_cell(
    task: str,
    method: str,
    seed: int,
    stage: str,
    value: float,
    *,
    lambda_only: bool,
    dpo_initialization: str | None = None,
) -> Cell:
    return _method_spec(method).build_cell(
        task=task,
        method=method,
        seed=seed,
        stage=stage,
        value=value,
        lambda_only=lambda_only,
        dpo_initialization=dpo_initialization,
    )


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
        methods = _coldstart_methods(config)
        method_seeds = experiment_config.task_transfer_seeds(config)
        if _is_method_matrix(config):
            for method_seed in method_seeds:
                for task in tasks:
                    if task == "countdown":
                        continue
                    for method in methods:
                        values = experiment_config.task_method_values(config, task, method=method)
                        cells.extend(
                            _coldstart_method_cell(
                                task, method, method_seed, "task_transfer", value,
                                lambda_only=False,
                                dpo_initialization=_method_spec(method).cell_initialization(config),
                            )
                            for value in values
                        )
        else:
            method = methods[0]
            lambda_only = (
                method == METHOD_EXPONENTIAL
                and config["sweep"]["parameterization"] == "paper_lambda_c1"
            )
            dpo_initialization = _method_spec(method).cell_initialization(config)
            countdown_values = experiment_config.task_method_values(
                config, "countdown", method=method
            )
            countdown_include_positive_only = bool(
                config["sweep"].get("countdown_include_positive_only", True)
            )
            include_global_endpoint = bool(
                config["sweep"].get("include_global_endpoint", False)
            )
            for seed_offset in tuple(
                int(value) for value in config["sweep"]["countdown_seed_offsets"]
            ):
                if countdown_include_positive_only:
                    cells.append(
                        Cell(
                            "countdown", METHOD_POSITIVE_ONLY, None, seed_offset,
                            "countdown_sentinel"
                        )
                    )
                cells.append(
                    Cell(
                        "countdown", METHOD_GLOBAL, 1.0, seed_offset,
                        "countdown_sentinel", 0.0
                    )
                )
                if method == METHOD_DPO:
                    raise AssertionError(
                        "Countdown DPO cells are disabled by config validation"
                    )
                cells.extend(
                    _coldstart_method_cell(
                        "countdown",
                        method,
                        seed_offset,
                        "countdown_sentinel",
                        value,
                        lambda_only=lambda_only,
                        dpo_initialization=dpo_initialization,
                    )
                    for value in countdown_values
                )
            positive_seeds = tuple(
                int(value)
                for value in config["sweep"]["transfer_positive_only_seed_offsets"]
            )
            for task in tasks:
                if task == "countdown":
                    continue
                values = experiment_config.task_method_values(
                    config, task, method=method
                )
                if not values:
                    continue
                cells.extend(
                    Cell(task, METHOD_POSITIVE_ONLY, None, seed_offset, "task_transfer")
                    for seed_offset in positive_seeds
                )
                for method_seed in method_seeds:
                    if include_global_endpoint:
                        cells.append(
                            Cell(
                                task, METHOD_GLOBAL, 1.0, method_seed,
                                "task_transfer", 0.0
                            )
                        )
                    cells.extend(
                        _coldstart_method_cell(
                            task,
                            method,
                            method_seed,
                            "task_transfer",
                            value,
                            lambda_only=lambda_only,
                            dpo_initialization=dpo_initialization,
                        )
                        for value in values
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
        seed_barrier = _is_method_matrix(config) and bool(
            config["execution"].get("seed_batch_barriers")
        )
        seed_order = (
            tuple(int(value) for value in config["execution"].get("seed_batch_order", ()))
            if seed_barrier
            else None
        )
        waves = e8_orchestration.nominal_batches(
            cells,
            slot_count=capacity,
            seed_barrier=seed_barrier,
            seed_order=seed_order,
        )
        if _is_baseline_matrix(config) and seed_barrier and tuple(
            len(wave) for wave in waves
        ) != (16, 16, 16, 16, 16, 8) * 2:
            raise AssertionError("Baseline matrix must form two 88-cell seed-local batches")
        if not waves or any(len(wave) != capacity for wave in waves[:-1]) and not seed_barrier:
            raise AssertionError(
                "Cold-start nominal batches must fill capacity except possibly the final batch"
            )
        return waves
    coarse = tuple(cell for cell in cells if cell.stage == "coarse")
    refinement = tuple(cell for cell in cells if cell.stage == "refinement")

    def chunk(values: tuple[Cell, ...]) -> tuple[tuple[Cell, ...], ...]:
        return tuple(
            tuple(values[index : index + capacity])
            for index in range(0, len(values), capacity)
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
    waves = build_waves(config)
    gpu_ids = tuple(int(value) for value in config["execution"]["gpu_ids"])
    rows = e8_orchestration.plan_rows(
        waves,
        gpu_ids=gpu_ids,
        project_cell=_method_output_columns,
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
        "wave_is_scheduling_barrier": bool(
            config["execution"].get("wave_barriers", False)
        ),
        "seed_batch_barriers": bool(
            config["execution"].get("seed_batch_barriers", False)
        ),
        "seed_batch_order": list(config["execution"].get("seed_batch_order", ())),
        "execution_class": _execution_class(config),
        "rows": rows,
        "scientific_status": "not_run",
    }
    atomic_json(output_root / "plan.json", plan)
    atomic_jsonl(output_root / "plan.jsonl", rows)
    return plan


def _paper_grid_name(coefficient: float) -> str:
    if coefficient == 0.0 or coefficient in PAPER_ROUND1_COEFFICIENTS:
        return "round1_grid"
    if coefficient in PAPER_EXTENSION_COEFFICIENTS:
        return "extension_grid"
    raise ValueError(f"Coefficient {coefficient} is outside the locked paper grids")


def _canonical_asymre_grid_path() -> Path:
    path = (_repo_root() / CANONICAL_ASYMRE_GRID).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Canonical AsymRE grid is missing: {path}")
    return path


def _canonical_topr_grid_path() -> Path:
    path = (_repo_root() / CANONICAL_TOPR_GRID).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Canonical TOPR grid is missing: {path}")
    return path


def _canonical_reciprocal_grid_path() -> Path:
    path = (_repo_root() / CANONICAL_RECIPROCAL_GRID).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Canonical reciprocal grid is missing: {path}")
    return path


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
                "selector_implementation": (
                    "source_p0_error_class_sequence_then_"
                    "within_class_reference_rank_spread_v1"
                ),
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
                and int(existing.get("rows", -1))
                == int(config["split"]["p0_train_rows"])
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
        reference_effective = experiment_config.effective_coldstart_runtime(
            config, pending[0]
        )
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

            def score_candidates(
                prompt: str,
                candidates: Sequence[Mapping[str, Any]],
                max_length: int,
                batch_size: int,
            ) -> Sequence[float]:
                return _score_reference_candidates(
                    arena=arena,
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    candidates=candidates,
                    max_length=max_length,
                    batch_size=batch_size,
                )

            for task in pending:
                record = splits["tasks"][task]
                summary = e8_inputs.materialize_reference_remoteness_task(
                    config,
                    output_root,
                    task=task,
                    record=record,
                    inputs=inputs[task],
                    identity_hash=identities[task],
                    score_candidates=score_candidates,
                )
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
        raise RuntimeError(
            "Canonical transfer inputs did not bind every derived reference bank"
        )
    return manifest


def write_canonical_cold_inputs(
    config: Mapping[str, Any],
    output_root: Path,
    split_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Compatibility entry point for canonical input materialization."""

    return e8_inputs.write_canonical_cold_inputs(
        config,
        output_root,
        split_manifest,
        audit_canonical_sources=audit_canonical_coldstart_sources,
        canonical_paths_for_config=_canonical_paths,
    )


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
    inputs = e8_inputs.resolve_task_inputs(
        config,
        p0_work_dir=p0_work_dir,
        p0_config=p0_config,
        countdown_bank=countdown_bank,
        countdown_validation=countdown_validation,
        countdown_adapter=countdown_adapter,
    )
    plan = write_plan(config, output_root)
    splits = e8_inputs.write_split_manifest(inputs, config, output_root)
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


def cmd_inherit(
    config: Mapping[str, Any],
    output_root: Path,
    *,
    parent_output_root: Path,
    parent_config_path: Path,
    base_model_path: str,
) -> dict[str, Any]:
    return e8_warmstart.inherit_dense_run(
        config,
        output_root,
        parent_output_root=parent_output_root,
        parent_config_path=parent_config_path,
        base_model_path=base_model_path,
        parent_experiment_id=EXPERIMENT_ID,
        is_dense_fn=_is_dense,
        load_config_fn=load_config,
        load_ready_inputs_fn=_load_ready_inputs,
        write_plan_fn=write_plan,
        build_cells_fn=build_cells,
        experiment_id_fn=experiment_id,
        coefficient_from_rho_fn=coefficient_from_rho,
        model_identity_fn=model_identity,
    )


def cmd_reference(
    config: Mapping[str, Any],
    output_root: Path,
    *,
    base_model_path: str,
    tasks: Sequence[str] | None,
    force: bool,
) -> dict[str, Any]:
    return e8_warmstart.prepare_references(
        config,
        output_root,
        base_model_path=base_model_path,
        tasks=tasks,
        force=force,
        load_prepared_fn=_load_prepared,
        model_identity_fn=model_identity,
        train_warmstart_fn=train_task_positive_warmstart,
    )


def _load_prepared(
    output_root: Path,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, TaskInputs]]:
    return e8_inputs.load_prepared_inputs(output_root, config)


def _load_ready_inputs(
    output_root: Path,
    config: Mapping[str, Any],
    *,
    base_model_path: str,
) -> tuple[dict[str, Any], dict[str, TaskInputs]]:
    return e8_warmstart.load_ready_inputs(
        output_root,
        config,
        base_model_path=base_model_path,
        model_identity_fn=model_identity,
    )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


completion_stats_batch = e8_warmstart.completion_stats_batch
_select_current_extremes = e8_warmstart._select_current_extremes
_load_reference_model = e8_warmstart._load_reference_model
_calibration_identity = e8_warmstart._calibration_identity


def _canonical_calibration_identity(
    task: str,
    *,
    split_manifest: Mapping[str, Any],
    base_model_path: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    record = _canonical_bridge()._canonical_task_record(split_manifest, task)
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


def _paper_grid_for_cell(*args: Any, **kwargs: Any) -> Any:
    return _canonical_bridge()._paper_grid_for_cell(*args, **kwargs)


def calibrate_canonical_cold_task(
    task: str,
    *,
    split_manifest: Mapping[str, Any],
    base_model_path: str,
    config: Mapping[str, Any],
    output_root: Path,
    force: bool,
) -> dict[str, Any]:
    record = _canonical_bridge()._canonical_task_record(split_manifest, task)
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
        **experiment_config.coldstart_remoteness_metadata(config),
        "canonical_train_sha256": record["train_sha256"],
        "task_metrics_used": False,
        "test_data_used": False,
        "complete": True,
        "scientific_status": "not_run",
    }
    atomic_json(result_path, result)
    return result


calibrate_task = e8_warmstart.calibrate_task


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



def _canonical_environment_evaluator(*args: Any, **kwargs: Any) -> Any:
    return _canonical_bridge()._canonical_environment_evaluator(*args, **kwargs)


def _cell_identity(
    cell: Cell,
    *,
    inputs: TaskInputs,
    split_manifest: Mapping[str, Any],
    base_model_path: str,
    config: Mapping[str, Any],
    calibration: Mapping[str, Any],
) -> dict[str, Any]:
    spec = _method_spec(cell.method)
    initialization = (
        spec.initialization_identity(config)
        if _is_coldstart(config)
        else {
            "source": "reference_adapter",
            "reference_adapter_identity": model_identity(
                base_model_path, str(inputs.reference_adapter)
            )["adapter"],
        }
    )
    return e8_runtime.recovery_identity(
        cell,
        experiment_id=experiment_id(config),
        config_hash=stable_config_hash(config),
        cell_identity_fields=_method_output_columns(cell),
        common_identity_fields={
            "bank_sha256": split_manifest["tasks"][cell.task]["bank_sha256"],
            "split_prompt_hashes": split_manifest["tasks"][cell.task][
                "prompt_id_hashes"
            ],
            "base_model_identity": model_identity(base_model_path, None)["model"],
            "initialization": initialization,
            "calibration_identity_hash": calibration["identity_hash"],
        },
    )


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


_load_cell_splits = e8_warmstart._load_cell_splits


def _legacy_arena_runtime_bridge(*args: Any, **kwargs: Any) -> Any:
    return _canonical_bridge()._legacy_arena_runtime_bridge(*args, **kwargs)


def _legacy_paper_runtime_bridge(*args: Any, **kwargs: Any) -> Any:
    return _canonical_bridge()._legacy_paper_runtime_bridge(*args, **kwargs)




def _canonical_baseline_grid_identity(*args: Any, **kwargs: Any) -> Any:
    return _canonical_bridge()._canonical_baseline_grid_identity(*args, **kwargs)


def _dpo_shared_sft_adapter_identity(*args: Any, **kwargs: Any) -> Any:
    return _canonical_bridge()._dpo_shared_sft_adapter_identity(*args, **kwargs)


def _dpo_shared_sft_adapter(*args: Any, **kwargs: Any) -> Any:
    return _canonical_bridge()._dpo_shared_sft_adapter(*args, **kwargs)


def _dpo_prompt_balanced_mean(*args: Any, **kwargs: Any) -> Any:
    return _canonical_bridge()._dpo_prompt_balanced_mean(*args, **kwargs)


def _is_nan_inf_numerical_failure(*args: Any, **kwargs: Any) -> Any:
    return _canonical_bridge()._is_nan_inf_numerical_failure(*args, **kwargs)


def _prepare_cell_output(
    output_root: Path,
    *,
    root_name: str,
    cell: Cell,
    identity: Mapping[str, Any],
    force: bool,
    mismatch_prefix: str = "Existing cell identity mismatch",
    existing_prefix: str = "Cell output exists without reusable manifest",
    unsafe_prefix: str = "Refusing unsafe cell removal",
) -> tuple[Path, Path, dict[str, Any] | None]:
    cell_root = output_root / root_name / cell.key
    manifest_path = cell_root / "cell_manifest.json"
    if manifest_path.is_file() and not force:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("identity_hash") == identity["identity_hash"] and existing.get("complete"):
            return cell_root, manifest_path, existing
        raise RuntimeError(f"{mismatch_prefix}: {cell.key}")
    if cell_root.exists():
        if not force:
            raise RuntimeError(f"{existing_prefix}: {cell_root}")
        expected_parent = (output_root / root_name).resolve()
        if expected_parent not in cell_root.resolve().parents:
            raise RuntimeError(f"{unsafe_prefix}: {cell_root}")
        shutil.rmtree(cell_root)
    cell_root.mkdir(parents=True, exist_ok=False)
    return cell_root, manifest_path, None


def _verify_fresh_process_adapter_reload(
    config_path: Path,
    output_root: Path,
    *,
    base_model_path: str,
    adapter_path: Path,
    expected_adapter_identity: Mapping[str, Any],
    require_base_identity: bool,
    label: str,
) -> dict[str, Any]:
    command = [
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
        str(adapter_path),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
        check=False,
    )
    descriptor = f" {label}" if label else ""
    if completed.returncode != 0:
        raise RuntimeError(
            f"Fresh-process{descriptor} adapter reload failed: {completed.stderr[-2000:]}"
        )
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Fresh-process{descriptor} adapter reload returned invalid JSON"
        ) from exc
    identity_ok = result.get("adapter_identity") == expected_adapter_identity
    if require_base_identity:
        identity_ok = identity_ok and (
            result.get("base_model_identity") == model_identity(base_model_path, None)["model"]
        )
    if (
        not result.get("complete")
        or not result.get("finite")
        or int(result.get("process_id", os.getpid())) == os.getpid()
        or not identity_ok
    ):
        raise RuntimeError(
            f"Fresh-process{descriptor} adapter reload identity or finiteness mismatch"
        )
    return result


def _train_canonical_dpo_transfer_cell(*args: Any, **kwargs: Any) -> Any:
    return _canonical_bridge()._train_canonical_dpo_transfer_cell(*args, **kwargs)


def _cmd_dpo_liveness(*args: Any, **kwargs: Any) -> Any:
    return _canonical_bridge()._cmd_dpo_liveness(*args, **kwargs)

def _train_canonical_cold_cell(*args: Any, **kwargs: Any) -> Any:
    return _canonical_bridge()._train_canonical_cold_cell(*args, **kwargs)


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
    return e8_warmstart.train_cell_impl(
        cell,
        inputs=inputs,
        split_manifest=split_manifest,
        base_model_path=base_model_path,
        config=config,
        output_root=output_root,
        force=force,
        bindings=e8_warmstart.WarmstartTrainingBindings(
            cell_identity=_cell_identity,
            prepare_cell_output=_prepare_cell_output,
            summarize_evaluations=_summarize_evaluations,
            method_exponential=METHOD_EXPONENTIAL,
        ),
        updates_override=updates_override,
        engineering_liveness=engineering_liveness,
    )


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
            return _method_spec(cell.method).train_cold(
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
                    "non-finite" in str(exc).lower()
                    or "nan/inf" in str(exc).lower()
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


def _method_liveness_grid(
    grid_path: Path, method: str, output_root: Path
) -> Path:
    if method not in {METHOD_RECIPROCAL_LINEAR, METHOD_RECIPROCAL_QUADRATIC}:
        return grid_path
    value = yaml.safe_load(grid_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("Canonical reciprocal grid root must be a mapping")
    value["execution"]["liveness"]["representative_family"] = method
    path = output_root / "liveness" / f"canonical_liveness_grid_{method}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".yaml.tmp")
    temporary.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    temporary.replace(path)
    return path


def _canonical_cold_liveness_cell(grid_path: Path) -> Cell:
    """Derive one paper-runtime liveness cell through the method contract."""

    grid = yaml.safe_load(grid_path.read_text(encoding="utf-8"))
    if not isinstance(grid, dict):
        raise TypeError("Canonical liveness grid root must be a mapping")
    liveness = grid["execution"]["liveness"]
    seed_offsets = grid["sweep"]["seed_offsets"]
    method = str(liveness.get("representative_family", METHOD_EXPONENTIAL))
    spec = _method_spec(method)
    if spec.paper_runtime is None:
        raise RuntimeError(f"{method} does not use canonical paper-runtime liveness")
    parameter_key = spec.paper_runtime.liveness_parameter
    if parameter_key not in liveness:
        raise RuntimeError(
            f"Canonical liveness grid for {method} lacks {parameter_key}"
        )
    return spec.build_cell(
        task="countdown",
        method=method,
        seed=int(seed_offsets[0]),
        stage="liveness",
        value=float(liveness[parameter_key]),
        lambda_only=False,
        dpo_initialization=None,
    )


def _cmd_canonical_cold_liveness(*args: Any, **kwargs: Any) -> Any:
    return _canonical_bridge()._cmd_canonical_cold_liveness(*args, **kwargs)


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
        splits, inputs = _load_ready_inputs(
            output_root,
            config,
            base_model_path=base_model_path,
        )
        if _is_method_matrix(config):
            if task != "countdown":
                raise RuntimeError(
                    "Baseline-matrix liveness is launched once from the Countdown anchor"
                )
            results: dict[str, Any] = {}
            for method in _coldstart_methods(config):
                spec = _method_spec(method)
                method_task = spec.liveness_task(config)
                results[method] = spec.liveness_runner(
                    config=config,
                    config_path=config_path,
                    output_root=output_root,
                    inputs=inputs,
                    splits=splits,
                    base_model_path=base_model_path,
                    task=method_task,
                    force=force,
                )
            return {
                "schema_version": 1,
                "experiment_id": experiment_id(config),
                "methods": results,
                "complete": all(
                    bool(value.get("complete")) for value in results.values()
                ),
                "scientific_status": "not_run",
            }
        method = _coldstart_method(config)
        spec = _method_spec(method)
        expected_task = spec.liveness_task(config)
        if task != expected_task:
            raise RuntimeError(
                f"{method} liveness anchor must be {expected_task}"
            )
        return spec.liveness_runner(
            config=config,
            config_path=config_path,
            output_root=output_root,
            inputs=inputs,
            splits=splits,
            base_model_path=base_model_path,
            task=task,
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
    reload_result = _verify_fresh_process_adapter_reload(
        config_path,
        output_root,
        base_model_path=base_model_path,
        adapter_path=Path(result["terminal_adapter"]),
        expected_adapter_identity=result["terminal_adapter_identity"],
        require_base_identity=True,
        label="",
    )
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

    terminal_hash = sha256_file(_adapter_weight_file(Path(result["terminal_adapter"])))
    if _is_coldstart(config):
        reference_hash = str(result["initialization_state_sha256"])
        changed = reference_hash != str(result["terminal_trainable_state_sha256"])
    else:
        reference_adapter = inputs[task].reference_adapter
        if reference_adapter is None:
            raise RuntimeError("Liveness reference adapter is missing")
        reference_hash = sha256_file(_adapter_weight_file(reference_adapter))
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
    return e8_runtime.read_json_object(path)


def _successful_attempt_matches_current_identity(
    config: Mapping[str, Any],
    workload_root: Path,
    *,
    source_commit: str,
    artifact_path: Path | None = None,
) -> bool:
    return e8_runtime.successful_attempt_matches_current_identity(
        workload_root,
        source_commit=source_commit,
        experiment_id_value=experiment_id(config),
        config_hash=stable_config_hash(config),
        artifact_path=artifact_path,
    )


def _effective_recovery_config(
    config: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    return e8_runtime.effective_recovery_config(
        config,
        output_root,
        load_config_fn=load_config,
        is_engineering_self_test_fn=_is_engineering_self_test,
    )


def _reusable_cell_manifests(
    config: Mapping[str, Any],
    output_root: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    return e8_runtime.reusable_cell_manifests(
        output_root,
        cells=build_cells(config),
        experiment_id_value=experiment_id(config),
        config_hash=stable_config_hash(config),
        engineering_self_test=_is_engineering_self_test(config),
        sha256_fn=sha256_file,
    )


def _recovery_stage_plan(
    config: Mapping[str, Any],
    output_root: Path,
    *,
    base_model_path: str,
) -> dict[str, Any]:
    effective = _effective_recovery_config(config, output_root)
    reusable, rejected = _reusable_cell_manifests(effective, output_root)
    return e8_runtime.recovery_stage_plan(
        effective,
        output_root,
        base_model_path=base_model_path,
        schema_version=RECOVERY_SNAPSHOT_SCHEMA_VERSION,
        experiment_id_value=experiment_id(effective),
        config_hash=stable_config_hash(effective),
        expected_cells=len(build_cells(effective)),
        reusable=reusable,
        rejected=rejected,
        load_prepared_fn=_load_prepared,
        require_calibration_fn=_require_calibration_gate,
        require_liveness_fn=_require_liveness_gate,
    )


def cmd_recovery_plan(
    config: Mapping[str, Any],
    output_root: Path,
    *,
    base_model_path: str,
) -> dict[str, Any]:
    plan = _recovery_stage_plan(config, output_root, base_model_path=base_model_path)
    atomic_json(output_root / "recovery" / "RECOVERY_PLAN.json", plan)
    return plan



def cmd_import_recovery(
    config: Mapping[str, Any],
    output_root: Path,
    *,
    source_output_root: Path,
    base_model_path: str,
    source_commit: str,
) -> dict[str, Any]:
    return e8_runtime.import_recovery(
        config,
        output_root,
        source_output_root=source_output_root,
        base_model_path=base_model_path,
        source_commit=source_commit,
        schema_version=RECOVERY_SNAPSHOT_SCHEMA_VERSION,
        transient_top_level=RECOVERY_TRANSIENT_TOP_LEVEL,
        transient_files=RECOVERY_TRANSIENT_FILES,
        effective_config_fn=_effective_recovery_config,
        recovery_stage_plan_fn=_recovery_stage_plan,
        experiment_id_fn=experiment_id,
        sha256_fn=sha256_file,
        write_json=atomic_json,
    )


def _recovery_checkpoint_snapshot(
    config: Mapping[str, Any],
    output_root: Path,
    snapshot_root: Path,
    *,
    source_commit: str,
) -> dict[str, Any]:
    reusable, rejected = _reusable_cell_manifests(config, output_root)
    return e8_runtime.recovery_checkpoint_snapshot(
        output_root,
        snapshot_root,
        source_commit=source_commit,
        schema_version=RECOVERY_SNAPSHOT_SCHEMA_VERSION,
        reusable=reusable,
        rejected=rejected,
        experiment_id_value=experiment_id(config),
        config_hash=stable_config_hash(config),
        expected_cells=len(build_cells(config)),
        scientific_status=("not_run" if _is_engineering_self_test(config) else "pilot"),
        sha256_fn=sha256_file,
        write_json=atomic_json,
    )



def _publish_recovery_checkpoint(
    config: Mapping[str, Any],
    output_root: Path,
    *,
    package_output: Path,
) -> dict[str, Any]:
    provenance = _read_json_object(output_root / "source_provenance.json")
    source_commit = str(provenance.get("source_commit", ""))
    if len(source_commit) != 40 or any(
        char not in "0123456789abcdef" for char in source_commit
    ):
        raise RuntimeError("Recovery checkpoint requires a full source commit")
    snapshot_root = package_output.parent / "snapshot"
    payload = _recovery_checkpoint_snapshot(
        config,
        output_root,
        snapshot_root,
        source_commit=source_commit,
    )
    return e8_runtime.publish_recovery_checkpoint(
        payload=payload,
        snapshot_root=snapshot_root,
        package_output=package_output,
        repo_root=_repo_root(),
        experiment_id_value=experiment_id(config),
        source_commit=source_commit,
        require_origin_main_match=(
            os.environ.get("E8_COLDSTART_RECOVERY_REQUIRE_ORIGIN_MAIN") == "1"
        ),
        mirror_value=os.environ.get("E8_COLDSTART_RECOVERY_MIRROR", ""),
        sha256_fn=sha256_file,
        write_json=atomic_json,
    )


def cmd_compact_logs(config: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    return e8_runtime.compact_logs(
        output_root,
        experiment_id_value=experiment_id(config),
        sha256_fn=sha256_file,
        write_json=atomic_json,
    )


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
        if _is_coldstart(config):
            expected_remoteness = experiment_config.coldstart_remoteness_metadata(config)
            if (
                result.get("enabled") is not False
                or result.get("mode") != expected_remoteness["mode"]
            ):
                raise RuntimeError(f"Paper calibration identity mismatch for {task}")


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
    passed_methods: set[str] = set()
    required_methods = set(_coldstart_methods(config)) if _is_coldstart(config) else set()
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
            and result.get("cell", {}).get("method") in required_methods
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
            if canonical_cold:
                passed_methods.add(str(result.get("cell", {}).get("method")))
    if not passed:
        raise RuntimeError("No identity-matched liveness result passes every engineering gate")
    if _is_coldstart(config) and passed_methods != required_methods:
        missing = sorted(required_methods - passed_methods)
        raise RuntimeError(f"Missing method-specific cold-start liveness gates: {missing}")



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
        raise RuntimeError(
            "Cold-start has no wave barriers; use run-all dynamic scheduling"
        )
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
    gpu_ids = tuple(
        int(value) for value in config["execution"]["gpu_ids"]
    )
    return e8_orchestration.run_wave(
        waves[wave_index - 1],
        wave_index=wave_index,
        gpu_ids=gpu_ids,
        slots_per_gpu=int(config["execution"]["slots_per_gpu"]),
        experiment_id_value=experiment_id(config),
        run_cell=lambda cell, gpu_id: _run_subprocess_cell(
            config_path=config_path.resolve(),
            output_root=output_root.resolve(),
            base_model_path=base_model_path,
            cell=cell,
            gpu_id=gpu_id,
            force=force,
        ),
        write_json=atomic_json,
        manifest_path=output_root / "waves" / f"wave_{wave_index:02d}.json",
    )


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
    """Run the cold-start plan through method-agnostic slot scheduling."""

    if not _is_coldstart(config):
        raise RuntimeError(
            "Dynamic scheduling is frozen for the cold-start profile only"
        )
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
    return e8_orchestration.cmd_run_dynamic(
        config,
        config_path,
        output_root,
        base_model_path=base_model_path,
        force=force,
        retry_incomplete=retry_incomplete,
        bindings=e8_orchestration.DynamicCommandBindings(
            build_cells=build_cells,
            build_waves=build_waves,
            experiment_id=experiment_id,
            execution_class=_execution_class,
            engineering_self_test=_is_engineering_self_test,
            method_matrix=_is_method_matrix,
            reusable_cell_manifests=_reusable_cell_manifests,
            run_subprocess_cell=_run_subprocess_cell,
            read_json_object=_read_json_object,
            materialize_task_results=(
                _materialize_completed_coldstart_task_results
            ),
            publish_recovery_checkpoint=_publish_recovery_checkpoint,
            protocol_diagnostic=_countdown_protocol_diagnostic,
            append_jsonl=append_jsonl,
            write_json=atomic_json,
        ),
    )




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
    return e8_orchestration.run_all_waves(
        wave_count=int(config["execution"]["expected_waves"]),
        experiment_id_value=experiment_id(config),
        run_wave_fn=lambda wave_index: cmd_run_wave(
            config,
            config_path,
            output_root,
            wave_index=wave_index,
            base_model_path=base_model_path,
            force=force,
        ),
        write_json=atomic_json,
        manifest_path=output_root / "waves" / "all_waves.json",
    )


def _coldstart_completed_task_rows(
    config: Mapping[str, Any],
    output_root: Path,
    task: str,
) -> list[dict[str, Any]] | None:
    if not _is_coldstart(config):
        raise RuntimeError(
            "Per-task early result materialization is cold-start only"
        )
    expected = [
        cell for cell in build_cells(config) if cell.task == task
    ]
    return e8_results.coldstart_completed_task_rows(
        output_root,
        expected_cells=expected,
        experiment_id_value=experiment_id(config),
        config_hash=stable_config_hash(config),
        result_row_fn=lambda cell, value: e8_results._coldstart_result_row(
            cell,
            value,
            source="current",
            method_columns=_method_output_columns(cell),
            require_late_window_metrics=not _is_engineering_self_test(config),
        ),
    )


def _materialize_completed_coldstart_task_results(
    config: Mapping[str, Any],
    output_root: Path,
    *,
    tasks: Sequence[str] | None = None,
) -> dict[str, dict[str, Any]]:
    task_values = (
        tuple(str(value) for value in config["suite"]["tasks"])
        if tasks is None
        else tuple(str(value) for value in tasks)
    )
    return e8_results.materialize_completed_task_results(
        task_values,
        completed_rows_fn=lambda task: _coldstart_completed_task_rows(
            config, output_root, task
        ),
        write_task_result_fn=lambda task, rows: e8_results._write_coldstart_task_result(
            config,
            output_root,
            task,
            rows,
            configured_cells=build_cells(config),
            method_specs={
                cell.method: _method_spec(cell.method)
                for cell in build_cells(config)
            },
            experiment_id_value=experiment_id(config),
            config_hash=stable_config_hash(config),
            engineering_self_test=_is_engineering_self_test(config),
            write_json=atomic_json,
            sha256_fn=sha256_file,
        ),
    )


def _coldstart_method_grouped_curve(
    *,
    task: str,
    method: str,
    method_rows: Sequence[Mapping[str, Any]],
    cells_by_key: Mapping[str, Cell],
) -> list[dict[str, Any]]:
    return e8_results._coldstart_method_grouped_curve(
        task=task,
        method=method,
        method_rows=method_rows,
        cells_by_key=cells_by_key,
        parameter_fn=_method_spec(method).parameters,
    )



def _aggregate_coldstart_matrix_unranked(
    config: Mapping[str, Any],
    output_root: Path,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    methods = _coldstart_methods(config)
    protocol_diagnostic = _countdown_protocol_diagnostic(
        config,
        output_root,
        destination=output_root / "aggregate" / "countdown_protocol_diagnostic.json",
    )
    return e8_results._aggregate_coldstart_matrix_unranked(
        config,
        output_root,
        rows,
        methods=methods,
        configured_cells=build_cells(config),
        method_specs={method: _method_spec(method) for method in methods},
        experiment_id_value=experiment_id(config),
        matrix_method=_coldstart_method(config),
        execution_class=_execution_class(config),
        transfer_seed_offsets=experiment_config.task_transfer_seeds(config),
        protocol_diagnostic=protocol_diagnostic,
        engineering_self_test=_is_engineering_self_test(config),
        write_json=atomic_json,
    )




def _aggregate_coldstart(
    config: Mapping[str, Any],
    output_root: Path,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    method = _coldstart_method(config)
    if _is_method_matrix(config):
        return _aggregate_coldstart_matrix_unranked(
            config, output_root, rows
        )
    protocol_diagnostic = _countdown_protocol_diagnostic(
        config,
        output_root,
        destination=(
            output_root
            / "aggregate"
            / "countdown_protocol_diagnostic.json"
        ),
    )
    if method != METHOD_EXPONENTIAL:
        return e8_results._aggregate_coldstart_unranked(
            config,
            output_root,
            rows,
            spec=_method_spec(method),
            configured_cells=build_cells(config),
            experiment_id_value=experiment_id(config),
            protocol_diagnostic=protocol_diagnostic,
            engineering_self_test=_is_engineering_self_test(config),
            positive_only_method=METHOD_POSITIVE_ONLY,
            global_method=METHOD_GLOBAL,
            write_json=atomic_json,
        )
    return e8_results._aggregate_coldstart_exponential(
        config,
        output_root,
        rows,
        configured_cells=build_cells(config),
        experiment_id_value=experiment_id(config),
        protocol_diagnostic_value=protocol_diagnostic,
        engineering_self_test=_is_engineering_self_test(config),
        positive_only_method=METHOD_POSITIVE_ONLY,
        global_method=METHOD_GLOBAL,
        exponential_method=METHOD_EXPONENTIAL,
        write_json=atomic_json,
    )


def cmd_aggregate(config: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    cells = build_cells(config)
    return e8_results.cmd_aggregate(
        config,
        output_root,
        cells=cells,
        experiment_id_value=experiment_id(config),
        dense_profile=_is_dense(config),
        coldstart_profile=_is_coldstart(config),
        method_columns_fn=_method_output_columns,
        coldstart_result_row_fn=lambda cell, value, source: (
            e8_results._coldstart_result_row(
                cell,
                value,
                source=source,
                method_columns=_method_output_columns(cell),
                require_late_window_metrics=not _is_engineering_self_test(config),
            )
        ),
        coldstart_aggregate_fn=lambda rows: _aggregate_coldstart(
            config, output_root, rows
        ),
        dense_aggregate_fn=lambda rows: e8_results._aggregate_dense(
            config,
            output_root,
            rows,
            experiment_id_value=experiment_id(config),
            config_hash=stable_config_hash(config),
            task_lambdas_fn=_task_lambdas,
            positive_only_method=METHOD_POSITIVE_ONLY,
            exponential_method=METHOD_EXPONENTIAL,
            write_json=atomic_json,
        ),
        positive_only_method=METHOD_POSITIVE_ONLY,
        exponential_method=METHOD_EXPONENTIAL,
        write_json=atomic_json,
    )


def cmd_audit(config: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    cells = build_cells(config)

    def method_audit(
        cell: Cell,
        value: Mapping[str, Any],
    ) -> e8_runtime.MethodAuditResult:
        return _method_spec(cell.method).audit_record(cell, value)

    coldstart_single_seed_shape_discovery = (
        _is_coldstart(config)
        and len(experiment_config.task_transfer_seeds(config)) == 1
    )
    transfer_exp_single_seed_response_shape_localization = (
        coldstart_single_seed_shape_discovery
        and _coldstart_method(config) == METHOD_EXPONENTIAL
    )
    return e8_runtime.terminal_audit(
        output_root,
        cells=cells,
        experiment_id_value=experiment_id(config),
        expected_terminal_step=int(config["training"]["optimizer_updates"]),
        engineering_self_test=_is_engineering_self_test(config),
        dense_profile=_is_dense(config),
        coldstart_profile=_is_coldstart(config),
        method_matrix=_is_method_matrix(config),
        execution_class=_execution_class(config),
        excluded_tasks=(
            dict(config["suite"]["excluded_tasks"])
            if (_is_dense(config) or _is_coldstart(config))
            else {}
        ),
        seed_batch_order=(
            tuple(int(v) for v in config["execution"]["seed_batch_order"])
            if _is_method_matrix(config)
            else ()
        ),
        transfer_exp_single_seed_response_shape_localization=(
            transfer_exp_single_seed_response_shape_localization
        ),
        coldstart_single_seed_shape_discovery=coldstart_single_seed_shape_discovery,
        compatibility_failure_buckets=("dpo_reference_identity_failures",),
        method_audit_fn=method_audit,
        audited_status_fn=lambda all_complete: _audited_scientific_status(
            config, all_complete
        ),
        write_json=atomic_json,
    )


def verify_result_package(
    package_manifest_path: Path,
    *,
    zip_override: Path | None = None,
) -> dict[str, Any]:
    return e8_results.verify_result_package(
        package_manifest_path,
        zip_override=zip_override,
        sha256_fn=sha256_file,
    )



def cmd_package(config: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    return e8_results.cmd_package(
        output_root,
        experiment_id_value=experiment_id(config),
        execution_class=_execution_class(config),
        config_hash=stable_config_hash(config),
        expected_cells=len(build_cells(config)),
        engineering_self_test=_is_engineering_self_test(config),
        write_json=atomic_json,
        sha256_fn=sha256_file,
    )



def cmd_finalize(config: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    return e8_results.cmd_finalize(
        output_root,
        experiment_id_value=experiment_id(config),
        execution_class=_execution_class(config),
        config_hash=stable_config_hash(config),
        expected_cells=len(build_cells(config)),
        engineering_self_test=_is_engineering_self_test(config),
        write_json=atomic_json,
        sha256_fn=sha256_file,
    )



def _engineering_self_test_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return e8_selftest._engineering_self_test_config(
        config,
        bindings=_selftest_bindings(),
    )


def cmd_engineering_self_test(
    config: Mapping[str, Any],
    output_root: Path,
    *,
    source_commit: str,
) -> dict[str, Any]:
    return e8_selftest.cmd_engineering_self_test(
        config,
        output_root,
        source_commit=source_commit,
        bindings=_selftest_bindings(),
    )




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
