#!/usr/bin/env python3
"""Parallel pilot/scaffold runner for EXT-H-E7-BENCH-01.

The parameter-sweep pilot is a registered substage of the existing benchmark ID. It
runs two Hopper offline datasets and two development seeds through three parallel
stages while sweeping method-specific parameters:

1. one canonical frozen-advantage critic per dataset (2 workers);
2. one shared Positive-only warm-start per ``(dataset, seed)`` pair (4 workers);
3. one equal-horizon continuation per ``(dataset, seed, method_variant)`` tuple
   (88 workers for the registered 22-variant sweep, including Positive-only).

Both seeds and all method-variant continuations are parallel at the coordinator
level. Every continuation verifies and loads the same 100k Positive-only
warm-start for its dataset/seed pair, preserving paired initialization without
serializing methods or giving Positive-only a shorter total actor horizon.
The formal nine-cell benchmark registers the same ``task_seed_method`` topology,
but remains fail-closed until its exact versions, seeds, base algorithm, optimizer,
and full budgets are frozen.

Pilot sweep results are pilot evidence only. They may be used to choose a
separate follow-up candidate configuration, but they may not populate the formal
nine-task table, retune per D4RL task, or become a formal method ranking without a
separately registered freeze and terminal audit.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import dataclasses
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np
import torch
import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drpo import e7_hopper_q2 as q2

EXPERIMENT_ID = "EXT-H-E7-BENCH-01"
RUNNER_VERSION = "0.2.3-recovered-network-profile-param-sweep"
PILOT_STATUS = "pilot"
WARMSTART_METHOD = "positive_only_warmstart"
PILOT_METHOD_FAMILIES = (
    "positive_only",
    "signed",
    "global_alpha",
    "reciprocal_linear",
    "reciprocal_quadratic",
    "exponential",
)
# Backward-compatible alias: these are method *families*, not sweep variants.
PILOT_METHODS = PILOT_METHOD_FAMILIES
TAPER_METHODS = (
    "reciprocal_linear",
    "reciprocal_quadratic",
    "exponential",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n")
    os.replace(temp, path)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_csv(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@dataclass(frozen=True)
class DatasetSpec:
    id: str
    relative_path: str
    sha256: str
    format: str
    env_id: str
    dataset_family: str
    score_protocol: str
    reference_min_score: float | None
    reference_max_score: float | None
    formal_cell_eligible: bool
    provenance_note: str


@dataclass(frozen=True)
class PilotBudget:
    seeds: tuple[int, ...]
    canonical_critic_seed: int
    max_transitions: int | None
    critic_steps: int
    critic_eval_interval: int
    shared_positive_warmstart_steps: int
    method_continuation_steps: int
    total_actor_steps_per_method: int
    actor_eval_interval: int
    rollout_eval_interval: int
    rollout_episodes: int
    final_rollout_episodes: int
    audit_sample_size: int


@dataclass(frozen=True)
class ParallelConfig:
    critic_workers: int
    warmstart_workers: int
    branch_workers: int
    critic_cpus_per_worker: int
    warmstart_cpus_per_worker: int
    branch_cpus_per_worker: int
    device_pool: tuple[str, ...]
    serial_seed_loop_forbidden: bool
    parallel_unit: str


@dataclass(frozen=True)
class NetworkProfile:
    source: str
    hidden_sizes: tuple[int, ...]
    activation: str
    init_scheme: str
    init_gain: float
    log_std_mode: str
    log_std_min: float
    log_std_max: float


@dataclass(frozen=True)
class MethodVariant:
    id: str
    family: str
    global_alpha: float | None = None
    coefficient: float | None = None


@dataclass(frozen=True)
class MethodConfig:
    global_alpha: float
    reference_distance: float
    near_region_boundary: float
    coefficients: dict[str, float]
    coefficient_source: str
    d4rl_retuning_allowed: bool
    pilot_parameter_search_enabled: bool
    per_task_retuning_allowed: bool
    variants: tuple[MethodVariant, ...]

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(variant.id for variant in self.variants)


@dataclass(frozen=True)
class BenchConfig:
    experiment_id: str
    protocol_version: str
    base_config_path: str
    datasets: tuple[DatasetSpec, ...]
    budget: PilotBudget
    parallel: ParallelConfig
    network_profile: NetworkProfile
    methods: MethodConfig
    formal_protocol_locked: bool
    formal_parallel_unit: str
    formal_serial_seed_loop_forbidden: bool
    formal_serial_method_loop_forbidden: bool
    formal_task_count: int
    config_sha256: str
    base_config_sha256: str


def load_bench_config(path: str | Path) -> BenchConfig:
    source_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(source_path.read_text())
    if raw.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError(f"experiment_id must be {EXPERIMENT_ID}")
    datasets = tuple(
        DatasetSpec(
            id=str(item["id"]),
            relative_path=str(item["relative_path"]),
            sha256=str(item["sha256"]),
            format=str(item["format"]),
            env_id=str(item["env_id"]),
            dataset_family=str(item["dataset_family"]),
            score_protocol=str(item["score_protocol"]),
            reference_min_score=(
                None
                if item.get("reference_min_score") is None
                else float(item["reference_min_score"])
            ),
            reference_max_score=(
                None
                if item.get("reference_max_score") is None
                else float(item["reference_max_score"])
            ),
            formal_cell_eligible=bool(item["formal_cell_eligible"]),
            provenance_note=str(item["provenance_note"]),
        )
        for item in raw["pilot"]["datasets"]
    )
    budget_raw = raw["pilot"]["budget"]
    budget = PilotBudget(
        seeds=tuple(int(x) for x in raw["pilot"]["seeds"]),
        canonical_critic_seed=int(raw["pilot"]["canonical_critic_seed"]),
        max_transitions=(
            None
            if budget_raw.get("max_transitions") in (None, 0)
            else int(budget_raw["max_transitions"])
        ),
        critic_steps=int(budget_raw["critic_steps"]),
        critic_eval_interval=int(budget_raw["critic_eval_interval"]),
        shared_positive_warmstart_steps=int(
            budget_raw["shared_positive_warmstart_steps"]
        ),
        method_continuation_steps=int(budget_raw["method_continuation_steps"]),
        total_actor_steps_per_method=int(budget_raw["total_actor_steps_per_method"]),
        actor_eval_interval=int(budget_raw["actor_eval_interval"]),
        rollout_eval_interval=int(budget_raw["rollout_eval_interval"]),
        rollout_episodes=int(budget_raw["rollout_episodes"]),
        final_rollout_episodes=int(budget_raw["final_rollout_episodes"]),
        audit_sample_size=int(budget_raw["audit_sample_size"]),
    )
    parallel_raw = raw["execution"]["pilot_parallel"]
    parallel = ParallelConfig(
        critic_workers=int(parallel_raw["critic_workers"]),
        warmstart_workers=int(parallel_raw["warmstart_workers"]),
        branch_workers=int(parallel_raw["branch_workers"]),
        critic_cpus_per_worker=int(parallel_raw["critic_cpus_per_worker"]),
        warmstart_cpus_per_worker=int(parallel_raw["warmstart_cpus_per_worker"]),
        branch_cpus_per_worker=int(parallel_raw["branch_cpus_per_worker"]),
        device_pool=tuple(str(x) for x in parallel_raw["device_pool"]),
        serial_seed_loop_forbidden=bool(parallel_raw["serial_seed_loop_forbidden"]),
        parallel_unit=str(parallel_raw["parallel_unit"]),
    )
    profile_raw = raw["model_profile"]
    network_profile = NetworkProfile(
        source=str(profile_raw["source"]),
        hidden_sizes=tuple(int(x) for x in profile_raw["hidden_sizes"]),
        activation=str(profile_raw["activation"]),
        init_scheme=str(profile_raw["init"]),
        init_gain=float(profile_raw["init_gain"]),
        log_std_mode=str(profile_raw["log_std_mode"]),
        log_std_min=float(profile_raw["log_std_min"]),
        log_std_max=float(profile_raw["log_std_max"]),
    )
    method_raw = raw["methods"]
    coefficients = {str(k): float(v) for k, v in method_raw["coefficients"].items()}
    search_raw = method_raw.get("pilot_parameter_search", {}) or {}
    variants_raw = search_raw.get("variants") or []
    variants = tuple(
        MethodVariant(
            id=str(item["id"]),
            family=str(item["family"]),
            global_alpha=(
                None if item.get("global_alpha") is None else float(item["global_alpha"])
            ),
            coefficient=(
                None if item.get("coefficient") is None else float(item["coefficient"])
            ),
        )
        for item in variants_raw
    )
    if not variants:
        variants = (
            MethodVariant(id="positive_only", family="positive_only"),
            MethodVariant(id="signed", family="signed"),
            MethodVariant(
                id="global_alpha",
                family="global_alpha",
                global_alpha=float(method_raw["global_alpha"]),
            ),
            *(
                MethodVariant(id=name, family=name, coefficient=value)
                for name, value in coefficients.items()
            ),
        )
    methods = MethodConfig(
        global_alpha=float(method_raw["global_alpha"]),
        reference_distance=float(method_raw["reference_distance"]),
        near_region_boundary=float(method_raw["near_region_boundary"]),
        coefficients=coefficients,
        coefficient_source=str(method_raw["coefficient_source"]),
        d4rl_retuning_allowed=bool(method_raw["d4rl_retuning_allowed"]),
        pilot_parameter_search_enabled=bool(search_raw.get("enabled", False)),
        per_task_retuning_allowed=bool(search_raw.get("per_task_retuning_allowed", False)),
        variants=variants,
    )
    formal = raw["formal_parallel_contract"]
    base_path = Path(str(raw["pilot"]["base_config_path"]))
    if not base_path.is_absolute():
        repo_root = Path(__file__).resolve().parents[2]
        base_path = (repo_root / base_path).resolve()
    if not base_path.is_file():
        raise FileNotFoundError(f"base config does not exist: {base_path}")
    config = BenchConfig(
        experiment_id=str(raw["experiment_id"]),
        protocol_version=str(raw["protocol_version"]),
        base_config_path=str(raw["pilot"]["base_config_path"]),
        datasets=datasets,
        budget=budget,
        parallel=parallel,
        network_profile=network_profile,
        methods=methods,
        formal_protocol_locked=bool(formal["protocol_locked"]),
        formal_parallel_unit=str(formal["parallel_unit"]),
        formal_serial_seed_loop_forbidden=bool(formal["serial_seed_loop_forbidden"]),
        formal_serial_method_loop_forbidden=bool(formal["serial_method_loop_forbidden"]),
        formal_task_count=int(formal["task_count"]),
        config_sha256=sha256_file(source_path),
        base_config_sha256=sha256_file(base_path),
    )
    validate_bench_config(config)
    return config


def validate_bench_config(config: BenchConfig) -> None:
    if len(config.datasets) != 2:
        raise ValueError("pilot must contain exactly two registered dataset cells")
    if len(config.budget.seeds) != 2 or len(set(config.budget.seeds)) != 2:
        raise ValueError("parameter-sweep pilot must contain exactly two unique development seeds")
    if config.parallel.parallel_unit != "dataset_seed_method_variant":
        raise ValueError("parameter-sweep pilot parallel_unit must be dataset_seed_method_variant")
    if not config.parallel.serial_seed_loop_forbidden:
        raise ValueError("pilot must forbid top-level serial seed or method loops")
    task_seed_jobs = len(config.datasets) * len(config.budget.seeds)
    branch_jobs = task_seed_jobs * len(config.methods.variants)
    if config.parallel.warmstart_workers < task_seed_jobs:
        raise ValueError(
            f"pilot warmstart_workers must cover all {task_seed_jobs} task-seed jobs"
        )
    if config.parallel.branch_workers < branch_jobs:
        raise ValueError(
            f"pilot branch_workers must cover all {branch_jobs} task-seed-method jobs"
        )
    for value in (
        config.parallel.critic_cpus_per_worker,
        config.parallel.warmstart_cpus_per_worker,
        config.parallel.branch_cpus_per_worker,
    ):
        if value < 1:
            raise ValueError("all per-stage CPU thread allocations must be positive")
    if config.formal_parallel_unit != "task_seed_method":
        raise ValueError("formal parallel_unit must be task_seed_method")
    if not config.formal_serial_seed_loop_forbidden:
        raise ValueError("formal benchmark must forbid serial seed execution")
    if not config.formal_serial_method_loop_forbidden:
        raise ValueError("formal benchmark must forbid serial method execution")
    if config.formal_task_count != 9:
        raise ValueError("formal benchmark must retain exactly nine task cells")
    if config.network_profile.hidden_sizes != (256, 256):
        raise ValueError("recovered E7 network profile must retain 2x256 hidden sizes")
    if config.network_profile.activation.strip().lower() != "relu":
        raise ValueError("recovered E7 network profile must use ReLU activations")
    if config.network_profile.init_scheme.strip().lower() != "orthogonal":
        raise ValueError("recovered E7 network profile must use orthogonal initialization")
    if config.network_profile.log_std_mode != "independent_global_diagonal":
        raise ValueError("recovered E7 network profile must retain independent global diagonal log_std")
    if config.network_profile.log_std_min != -5.0 or config.network_profile.log_std_max != 2.0:
        raise ValueError("recovered E7 network profile must retain log_std clamp [-5, 2]")
    if config.methods.d4rl_retuning_allowed:
        raise ValueError("formal D4RL retuning must remain disabled")
    if config.methods.per_task_retuning_allowed:
        raise ValueError("per-task D4RL retuning is forbidden for this pilot sweep")
    if set(config.methods.coefficients) != set(TAPER_METHODS):
        raise ValueError("pilot taper coefficients must retain the frozen three-family baseline")
    if not (0.0 < config.methods.global_alpha <= 1.0):
        raise ValueError("global_alpha baseline must be in (0, 1]")
    variant_ids = config.methods.ids
    if len(set(variant_ids)) != len(variant_ids):
        raise ValueError("method variant ids must be unique")
    if len(variant_ids) < len(PILOT_METHOD_FAMILIES):
        raise ValueError("method parameter sweep must include more than the six family baselines")
    families = {variant.family for variant in config.methods.variants}
    if families != set(PILOT_METHOD_FAMILIES):
        raise ValueError("method parameter sweep must cover exactly the registered families")
    if not config.methods.pilot_parameter_search_enabled:
        raise ValueError("pilot parameter search must be explicitly enabled")
    for variant in config.methods.variants:
        if variant.family in {"positive_only", "signed"}:
            if variant.global_alpha is not None or variant.coefficient is not None:
                raise ValueError(f"{variant.family} variant must not carry a tunable scalar")
        elif variant.family == "global_alpha":
            if variant.global_alpha is None or not (0.0 < variant.global_alpha <= 1.0):
                raise ValueError("global_alpha variants must define a scalar in (0, 1]")
            if variant.coefficient is not None:
                raise ValueError("global_alpha variants must not define a taper coefficient")
        elif variant.family in TAPER_METHODS:
            if variant.coefficient is None or variant.coefficient <= 0.0:
                raise ValueError("taper variants must define a positive coefficient")
            if variant.global_alpha is not None:
                raise ValueError("taper variants must not define global_alpha")
        else:
            raise ValueError(f"unknown method family: {variant.family}")
    if config.budget.critic_steps != 100000:
        raise ValueError("pilot critic_steps must remain frozen at 100000")
    if config.budget.shared_positive_warmstart_steps != 100000:
        raise ValueError("pilot shared Positive-only warm-start must remain 100000 steps")
    if config.budget.method_continuation_steps != 200000:
        raise ValueError("pilot method continuation must remain 200000 steps")
    expected_total = (
        config.budget.shared_positive_warmstart_steps
        + config.budget.method_continuation_steps
    )
    if config.budget.total_actor_steps_per_method != expected_total:
        raise ValueError(
            "total_actor_steps_per_method must equal warm-start plus continuation"
        )


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run_identity_payload(config: BenchConfig) -> dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "runner_version": RUNNER_VERSION,
        "protocol_version": config.protocol_version,
        "config_sha256": config.config_sha256,
        "base_config_sha256": config.base_config_sha256,
        "datasets": [
            {
                "id": spec.id,
                "sha256": spec.sha256,
                "format": spec.format,
                "env_id": spec.env_id,
                "score_protocol": spec.score_protocol,
            }
            for spec in config.datasets
        ],
        "seeds": list(config.budget.seeds),
        "canonical_critic_seed": config.budget.canonical_critic_seed,
        "budget": dataclasses.asdict(config.budget),
        "network_profile": dataclasses.asdict(config.network_profile),
        "methods": {
            "family_ids": list(PILOT_METHOD_FAMILIES),
            **dataclasses.asdict(config.methods),
        },
        "parallel": dataclasses.asdict(config.parallel),
    }


def run_identity_sha256(config: BenchConfig) -> str:
    return canonical_json_sha256(run_identity_payload(config))


def worker_identity_payload(
    config: BenchConfig,
    spec: DatasetSpec,
    *,
    worker: str,
    seed: int | None = None,
    method: str | None = None,
) -> dict[str, Any]:
    if worker not in {"critic", "warmstart", "branch"}:
        raise ValueError(f"unknown worker identity type: {worker}")
    if worker == "critic":
        stage_budget = {
            "optimizer_steps": config.budget.critic_steps,
            "eval_interval": config.budget.critic_eval_interval,
        }
    elif worker == "warmstart":
        stage_budget = {
            "optimizer_steps": config.budget.shared_positive_warmstart_steps,
            "actor_eval_interval": config.budget.actor_eval_interval,
            "rollout_eval_interval": config.budget.rollout_eval_interval,
            "rollout_episodes": config.budget.rollout_episodes,
            "final_rollout_episodes": config.budget.final_rollout_episodes,
        }
    else:
        stage_budget = {
            "optimizer_steps": config.budget.method_continuation_steps,
            "actor_eval_interval": config.budget.actor_eval_interval,
            "rollout_eval_interval": config.budget.rollout_eval_interval,
            "rollout_episodes": config.budget.rollout_episodes,
            "final_rollout_episodes": config.budget.final_rollout_episodes,
        }
    return {
        "run_identity_sha256": run_identity_sha256(config),
        "worker": worker,
        "dataset_id": spec.id,
        "dataset_sha256": spec.sha256,
        "seed": seed,
        "method": method,
        "stage_budget": stage_budget,
        "network_profile": dataclasses.asdict(config.network_profile),
        "method_parameters": dataclasses.asdict(config.methods),
    }


def worker_identity_sha256(
    config: BenchConfig,
    spec: DatasetSpec,
    *,
    worker: str,
    seed: int | None = None,
    method: str | None = None,
) -> str:
    return canonical_json_sha256(
        worker_identity_payload(
            config,
            spec,
            worker=worker,
            seed=seed,
            method=method,
        )
    )


def ensure_run_identity(work_dir: Path, config: BenchConfig, *, resume: bool) -> str:
    identity = run_identity_sha256(config)
    path = work_dir / "RUN_IDENTITY.json"
    payload = {
        **run_identity_payload(config),
        "run_identity_sha256": identity,
        "created_utc": utc_now(),
        "resume_contract": (
            "Exact identity match is required. Budget, config, base config, dataset, "
            "runner, or taper changes require a new work directory."
        ),
    }
    if path.is_file():
        existing = json.loads(path.read_text())
        if existing.get("run_identity_sha256") != identity:
            raise RuntimeError(
                "work directory belongs to a different E7-BENCH protocol identity; "
                "use a new work directory instead of resuming stale short-budget outputs"
            )
        if not resume:
            raise RuntimeError(
                "work directory already contains this run identity; pass --resume or use a new directory"
            )
        return identity
    existing = [child for child in work_dir.iterdir() if child.name != path.name]
    if existing:
        raise RuntimeError(
            "non-empty work directory has no RUN_IDENTITY.json; refusing to mix or overwrite artifacts"
        )
    atomic_write_json(path, payload)
    return identity


def verify_worker_identity(
    args: argparse.Namespace,
    config: BenchConfig,
    spec: DatasetSpec,
    *,
    worker: str,
    seed: int | None = None,
    method: str | None = None,
) -> tuple[str, dict[str, Any]]:
    payload = worker_identity_payload(
        config,
        spec,
        worker=worker,
        seed=seed,
        method=method,
    )
    identity = canonical_json_sha256(payload)
    expected = str(args.expected_worker_identity_sha256)
    if identity != expected:
        raise RuntimeError(
            f"{worker} worker protocol identity changed after scheduling: expected {expected}, got {identity}"
        )
    return identity, payload


def build_execution_plan(config: BenchConfig, mode: str) -> dict[str, Any]:
    if mode == "pilot":
        critics = [dataset.id for dataset in config.datasets]
        positives = [
            {"dataset_id": dataset.id, "seed": seed}
            for dataset in config.datasets
            for seed in config.budget.seeds
        ]
        branches = [
            {"dataset_id": dataset.id, "seed": seed, "method": method}
            for dataset in config.datasets
            for seed in config.budget.seeds
            for method in config.methods.ids
        ]
        return {
            "mode": mode,
            "scientific_status": PILOT_STATUS,
            "critic_parallel_stage": critics,
            "warmstart_parallel_stage": positives,
            "branch_parallel_stage": branches,
            "critic_workers": config.parallel.critic_workers,
            "warmstart_workers": config.parallel.warmstart_workers,
            "branch_workers": config.parallel.branch_workers,
            "critic_cpus_per_worker": config.parallel.critic_cpus_per_worker,
            "warmstart_cpus_per_worker": config.parallel.warmstart_cpus_per_worker,
            "branch_cpus_per_worker": config.parallel.branch_cpus_per_worker,
            "shared_positive_warmstart_steps": (
                config.budget.shared_positive_warmstart_steps
            ),
            "method_continuation_steps": config.budget.method_continuation_steps,
            "total_actor_steps_per_method": config.budget.total_actor_steps_per_method,
            "method_continuation_count": len(branches),
            "parallel_unit": config.parallel.parallel_unit,
            "top_level_serial_seed_loop": False,
            "top_level_serial_method_loop": False,
        }
    if mode == "formal":
        return {
            "mode": mode,
            "protocol_locked": config.formal_protocol_locked,
            "task_count": config.formal_task_count,
            "parallel_unit": config.formal_parallel_unit,
            "serial_seed_loop_forbidden": config.formal_serial_seed_loop_forbidden,
            "serial_method_loop_forbidden": config.formal_serial_method_loop_forbidden,
            "launch_allowed": False,
            "blocking_reason": (
                "formal seeds, exact dataset versions, base algorithm, optimizer, and "
                "full-budget protocol remain frozen-pending in the registry"
            ),
        }
    raise ValueError(f"unknown mode: {mode}")

def resolve_dataset_path(dataset_root: Path, spec: DatasetSpec) -> Path:
    path = (dataset_root / spec.relative_path).resolve()
    root = dataset_root.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"unsafe dataset path outside dataset root: {path}") from exc
    return path


def validate_dataset_file(dataset_root: Path, spec: DatasetSpec) -> dict[str, Any]:
    path = resolve_dataset_path(dataset_root, spec)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing registered dataset {spec.id}: {path}. Extract the registered bundle "
            "without renaming its internal paths."
        )
    actual = sha256_file(path)
    if actual != spec.sha256:
        raise ValueError(
            f"Dataset hash mismatch for {spec.id}: expected {spec.sha256}, got {actual}"
        )
    stat = path.stat()
    return {
        "id": spec.id,
        "path": str(path),
        "sha256": actual,
        "size_bytes": stat.st_size,
        "mtime_ns_at_validation": stat.st_mtime_ns,
        "format": spec.format,
        "env_id": spec.env_id,
        "dataset_family": spec.dataset_family,
        "score_protocol": spec.score_protocol,
        "formal_cell_eligible": spec.formal_cell_eligible,
        "provenance_note": spec.provenance_note,
    }


def worker_dataset_manifest(
    dataset_root: Path,
    spec: DatasetSpec,
    validated_manifest_path: str | None,
) -> dict[str, Any]:
    """Reuse the coordinator's one-time SHA check without 48 concurrent re-hashes."""
    if not validated_manifest_path:
        return validate_dataset_file(dataset_root, spec)
    manifest_path = Path(validated_manifest_path).expanduser().resolve()
    rows = json.loads(manifest_path.read_text())
    row = next((item for item in rows if item.get("id") == spec.id), None)
    if row is None:
        raise ValueError(f"validated dataset manifest has no entry for {spec.id}")
    expected_path = resolve_dataset_path(dataset_root, spec)
    if Path(row.get("path", "")).resolve() != expected_path:
        raise ValueError(f"validated path mismatch for {spec.id}")
    if row.get("sha256") != spec.sha256:
        raise ValueError(f"validated SHA mismatch for {spec.id}")
    stat = expected_path.stat()
    if int(row.get("size_bytes", -1)) != stat.st_size:
        raise ValueError(f"dataset size changed after coordinator validation: {spec.id}")
    if int(row.get("mtime_ns_at_validation", -1)) != stat.st_mtime_ns:
        raise ValueError(f"dataset mtime changed after coordinator validation: {spec.id}")
    return row


def preflight_pilot_runtime(
    config: BenchConfig,
    manifests: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Fail before long critic training if CPU capacity or MuJoCo rollout is unavailable."""
    detected_cpus = os.cpu_count()
    stage_threads = {
        "parallel_canonical_critics": (
            config.parallel.critic_workers * config.parallel.critic_cpus_per_worker
        ),
        "parallel_shared_positive_warmstarts": (
            config.parallel.warmstart_workers * config.parallel.warmstart_cpus_per_worker
        ),
        "parallel_equal_horizon_method_continuations": (
            config.parallel.branch_workers * config.parallel.branch_cpus_per_worker
        ),
    }
    peak_threads = max(stage_threads.values())
    if detected_cpus is not None and detected_cpus < peak_threads:
        raise RuntimeError(
            f"registered pilot needs {peak_threads} CPU threads at peak, but os.cpu_count() "
            f"reports {detected_cpus}; refusing silent oversubscription"
        )
    manifest_by_id = {row["id"]: row for row in manifests}
    environment_checks: list[dict[str, Any]] = []
    for spec in config.datasets:
        row = manifest_by_id[spec.id]
        data = load_dataset(Path(row["path"]), spec, max_transitions=8)
        env = None
        try:
            env, metadata = q2._open_gymnasium_mujoco_env(spec.env_id)
            observation, reset_metadata = q2._reset_env(env, seed=0)
            action_shape = tuple(int(value) for value in env.action_space.shape)
            if observation.shape != data.observations[0].shape:
                raise RuntimeError(
                    f"observation shape mismatch for {spec.id}: dataset "
                    f"{data.observations[0].shape}, env {observation.shape}"
                )
            if action_shape != data.actions[0].shape:
                raise RuntimeError(
                    f"action shape mismatch for {spec.id}: dataset "
                    f"{data.actions[0].shape}, env {action_shape}"
                )
            environment_checks.append(
                {
                    "dataset_id": spec.id,
                    "env_id": spec.env_id,
                    "observation_shape": list(observation.shape),
                    "action_shape": list(action_shape),
                    "rollout_backend": metadata,
                    "reset_metadata": reset_metadata,
                }
            )
        finally:
            if env is not None:
                env.close()
    return {
        "detected_cpu_count": detected_cpus,
        "registered_stage_threads": stage_threads,
        "registered_peak_threads": peak_threads,
        "environment_checks": environment_checks,
        "completed_utc": utc_now(),
    }


def _episode_sort_key(name: str) -> tuple[int, str]:
    suffix = name.rsplit("_", 1)[-1]
    return (int(suffix), name) if suffix.isdigit() else (sys.maxsize, name)


def load_minari_episode_hdf5(
    path: str | Path, max_transitions: int | None
) -> q2.OfflineData:
    observations: list[np.ndarray] = []
    next_observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[np.ndarray] = []
    terminals: list[np.ndarray] = []
    timeouts: list[np.ndarray] = []
    episode_ids: list[np.ndarray] = []
    remaining = None if max_transitions is None else int(max_transitions)
    with h5py.File(path, "r") as handle:
        episode_names = sorted(
            (name for name in handle.keys() if name.startswith("episode_")),
            key=_episode_sort_key,
        )
        if not episode_names:
            raise ValueError("Minari episodic HDF5 has no episode_* groups")
        for episode_index, name in enumerate(episode_names):
            if remaining is not None and remaining <= 0:
                break
            group = handle[name]
            required = ("observations", "actions", "rewards", "terminations", "truncations")
            missing = [key for key in required if key not in group]
            if missing:
                raise ValueError(f"{name} is missing Minari arrays: {missing}")
            action = np.asarray(group["actions"], dtype=np.float32)
            obs = np.asarray(group["observations"], dtype=np.float32)
            reward = np.asarray(group["rewards"], dtype=np.float32).reshape(-1)
            terminal = np.asarray(group["terminations"], dtype=np.bool_).reshape(-1)
            timeout = np.asarray(group["truncations"], dtype=np.bool_).reshape(-1)
            count = len(action)
            if len(obs) != count + 1:
                raise ValueError(f"{name} observations must contain T+1 rows")
            take = count if remaining is None else min(count, remaining)
            if take <= 0:
                break
            observations.append(obs[:take])
            next_observations.append(obs[1 : take + 1])
            actions.append(action[:take])
            rewards.append(reward[:take])
            terminals.append(terminal[:take])
            timeouts.append(timeout[:take])
            episode_ids.append(np.full(take, episode_index, dtype=np.int64))
            if remaining is not None:
                remaining -= take
    if not actions:
        raise ValueError("Minari dataset yielded no transitions")
    return q2.OfflineData(
        observations=np.concatenate(observations),
        actions=np.concatenate(actions),
        rewards=np.concatenate(rewards),
        next_observations=np.concatenate(next_observations),
        terminals=np.concatenate(terminals),
        timeouts=np.concatenate(timeouts),
        episode_ids=np.concatenate(episode_ids),
    )


def load_dataset(path: Path, spec: DatasetSpec, max_transitions: int | None) -> q2.OfflineData:
    if spec.format == "legacy_d4rl_hdf5":
        return q2.load_hopper_hdf5(path, max_transitions)
    if spec.format == "minari_episode_hdf5":
        return load_minari_episode_hdf5(path, max_transitions)
    raise ValueError(f"unsupported dataset format: {spec.format}")


def make_q2_config(
    config: BenchConfig,
    spec: DatasetSpec,
    dataset_path: Path,
) -> tuple[q2.E7Config, q2.ModeConfig]:
    base_path = Path(config.base_config_path)
    if not base_path.is_absolute():
        repo_root = Path(__file__).resolve().parents[2]
        base_path = (repo_root / base_path).resolve()
    base = q2.load_config(base_path)
    budget = config.budget
    mode = q2.ModeConfig(
        max_transitions=budget.max_transitions,
        seeds=budget.seeds,
        canonical_critic_seed=budget.canonical_critic_seed,
        critic_max_steps=budget.critic_steps,
        critic_min_steps=max(1, budget.critic_steps // 4),
        critic_eval_interval=budget.critic_eval_interval,
        positive_max_steps=budget.shared_positive_warmstart_steps,
        positive_min_steps=max(1, budget.shared_positive_warmstart_steps // 4),
        actor_eval_interval=budget.actor_eval_interval,
        branch_max_steps=budget.method_continuation_steps,
        branch_min_steps=max(1, budget.method_continuation_steps // 4),
        matched_pairs=0,
        audit_sample_size=budget.audit_sample_size,
        rollout_episodes=budget.rollout_episodes,
        final_rollout_episodes=budget.final_rollout_episodes,
        rollout_eval_interval=budget.rollout_eval_interval,
    )
    reference_min = spec.reference_min_score if spec.reference_min_score is not None else 0.0
    reference_max = spec.reference_max_score if spec.reference_max_score is not None else 1.0
    adapted = dataclasses.replace(
        base,
        experiment_id=base.experiment_id,
        dataset_basename=dataset_path.name,
        dataset_sha256=spec.sha256,
        rollout_dataset_id=spec.id,
        env_id=spec.env_id,
        normalized_score_percent=(spec.score_protocol == "d4rl_v2_percent"),
        normalized_score_reference_min=reference_min,
        normalized_score_reference_max=reference_max,
        pilot=mode,
        hidden_sizes=config.network_profile.hidden_sizes,
        activation=config.network_profile.activation,
        init_scheme=config.network_profile.init_scheme,
        init_gain=config.network_profile.init_gain,
        log_std_min=config.network_profile.log_std_min,
        log_std_max=config.network_profile.log_std_max,
    )
    return adapted, mode


def configure_worker_threads(cpus_per_worker: int) -> None:
    value = str(max(1, int(cpus_per_worker)))
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = value
    torch.set_num_threads(int(value))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def get_method_variant(method: str, methods: MethodConfig) -> MethodVariant:
    for variant in methods.variants:
        if variant.id == method:
            return variant
    raise ValueError(f"unknown benchmark method variant: {method}")


def taper_weight(
    distance: torch.Tensor,
    method: str,
    methods: MethodConfig,
) -> torch.Tensor:
    variant = get_method_variant(method, methods)
    family = variant.family
    if family == "signed":
        return torch.ones_like(distance)
    if family == "global_alpha":
        if variant.global_alpha is None:
            raise ValueError(f"global_alpha variant lacks scalar: {method}")
        return torch.full_like(distance, float(variant.global_alpha))
    if family == "positive_only":
        return torch.zeros_like(distance)
    coefficient = variant.coefficient
    if coefficient is None:
        raise ValueError(f"taper variant lacks coefficient: {method}")
    u = distance / methods.reference_distance
    if family == "reciprocal_linear":
        return 1.0 / (1.0 + coefficient * u)
    if family == "reciprocal_quadratic":
        return 1.0 / (1.0 + coefficient * u.square())
    if family == "exponential":
        return torch.exp(-coefficient * u)
    raise ValueError(f"unknown taper family: {family}")


def benchmark_actor_loss(
    policy: q2.SquashedGaussianPolicy,
    obs_t: torch.Tensor,
    actions_t: torch.Tensor,
    advantages_t: torch.Tensor,
    method: str,
    methods: MethodConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    log_prob = policy.log_prob(obs_t, actions_t)
    negative = advantages_t < 0
    distance = policy.standardized_distance(obs_t, actions_t).detach()
    variant = get_method_variant(method, methods)
    factor = torch.ones_like(advantages_t)
    if variant.family == "positive_only":
        factor = torch.where(negative, torch.zeros_like(factor), factor)
    else:
        factor = torch.where(negative, taper_weight(distance, method, methods), factor)
    weighted_advantage = advantages_t * factor
    active = weighted_advantage.ne(0)
    if not bool(active.any()):
        raise RuntimeError(f"method {method} produced an empty active batch")
    loss = -(weighted_advantage[active] * log_prob[active]).mean()
    near = negative & (distance <= methods.near_region_boundary)
    far = negative & ~near
    return loss, {
        "active_fraction": float(active.float().mean().detach().cpu()),
        "negative_fraction": float(negative.float().mean().detach().cpu()),
        "negative_weight_mean": (
            float(factor[negative].mean().detach().cpu()) if bool(negative.any()) else float("nan")
        ),
        "near_negative_weight_mean": (
            float(factor[near].mean().detach().cpu()) if bool(near.any()) else float("nan")
        ),
        "far_negative_weight_mean": (
            float(factor[far].mean().detach().cpu()) if bool(far.any()) else float("nan")
        ),
        "near_negative_fraction": float(near.float().mean().detach().cpu()),
        "far_negative_fraction": float(far.float().mean().detach().cpu()),
        "reference_distance": methods.reference_distance,
        "method_family": variant.family,
        "method_global_alpha": (
            float(variant.global_alpha) if variant.global_alpha is not None else float("nan")
        ),
        "method_coefficient": (
            float(variant.coefficient) if variant.coefficient is not None else float("nan")
        ),
    }


def evaluate_rollouts(
    *,
    policy: q2.SquashedGaussianPolicy,
    obs_norm: q2.Normalizer,
    spec: DatasetSpec,
    episodes: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    if episodes <= 0:
        return {
            "rollout_status": "disabled",
            "rollout_return_mean": float("nan"),
            "rollout_return_std": float("nan"),
            "normalized_return": float("nan"),
            "normalized_return_available": False,
            "rollout_episodes": 0,
        }
    env = None
    try:
        env, metadata = q2._open_gymnasium_mujoco_env(spec.env_id)
        returns: list[float] = []
        lengths: list[int] = []
        for episode in range(episodes):
            observation, _ = q2._reset_env(env, seed + episode)
            total = 0.0
            done = False
            steps = 0
            limit = q2._max_episode_steps(env, 10000)
            while not done and steps < limit:
                normalized = obs_norm.transform(observation.reshape(1, -1))
                with torch.no_grad():
                    action = policy.action_mean(q2.tensor(normalized, device))[0].cpu().numpy()
                action = q2._clip_action_to_space(env, action)
                observation, reward, done, _ = q2._step_env(env, action)
                total += reward
                steps += 1
            if not done and steps >= limit:
                raise RuntimeError(f"rollout exceeded safety limit {limit}")
            returns.append(total)
            lengths.append(steps)
        raw = float(np.mean(returns))
        if spec.score_protocol == "d4rl_v2_percent":
            if spec.reference_min_score is None or spec.reference_max_score is None:
                raise RuntimeError("D4RL score protocol requires frozen reference scores")
            score = q2.normalize_d4rl_reference_score(
                raw,
                spec.reference_min_score,
                spec.reference_max_score,
                percent=True,
            )
            score_name = "d4rl_v2_percent"
            normalized_available = True
        elif spec.score_protocol == "raw_return_only":
            # Keep the common terminal-audit scalar field, but mark it explicitly as
            # a pilot-only raw-return identity score rather than D4RL normalization.
            score = raw
            score_name = "raw_return_identity_pilot_only"
            normalized_available = False
        else:
            raise ValueError(f"unknown score protocol: {spec.score_protocol}")
        return {
            "rollout_status": "available",
            "rollout_return_mean": raw,
            "rollout_return_std": float(np.std(returns)),
            "normalized_return": score,
            "normalized_return_available": normalized_available,
            "rollout_episodes": episodes,
            "rollout_episode_steps_mean": float(np.mean(lengths)),
            "score_protocol": score_name,
            "evaluation_env_id": spec.env_id,
            "rollout_open_metadata": metadata,
        }
    finally:
        if env is not None:
            env.close()


def _policy_eval_row(
    *,
    policy: q2.SquashedGaussianPolicy,
    method: str,
    obs: np.ndarray,
    actions: np.ndarray,
    advantages: np.ndarray,
    audit_indices: np.ndarray,
    negative_indices: np.ndarray,
    q2_config: q2.E7Config,
    method_config: MethodConfig,
    device: torch.device,
    step: int,
    update_stats: dict[str, float],
    rollout: dict[str, Any] | None,
) -> dict[str, Any]:
    obs_t = q2.tensor(obs[audit_indices], device)
    action_t = q2.tensor(actions[audit_indices], device)
    advantage_t = q2.tensor(advantages[audit_indices], device)
    loss, diagnostics = benchmark_actor_loss(
        policy, obs_t, action_t, advantage_t, method, method_config
    )
    gradient = q2.full_gradient_statistics(loss, policy.parameters())
    row = q2.actor_eval_metrics(
        policy=policy,
        obs=obs,
        actions=actions,
        advantages=advantages,
        audit_indices=audit_indices,
        fixed_negative_indices=negative_indices,
        device=device,
        loss_value=float(loss.detach().cpu()),
        gradient_norm=gradient["raw"],
        gradient_rms=gradient["rms"],
        relative_gradient_norm=gradient["relative_to_parameter_norm"],
        update_norm=update_stats["raw_per_step"],
        relative_update_norm=update_stats["relative_per_step"],
        step=step,
        boundary_threshold=q2_config.support_boundary_threshold,
        rollout_metrics=rollout,
    )
    row.update({f"audit_{key}": value for key, value in diagnostics.items()})
    return row


def _apply_training_failure_to_terminal(
    terminal: dict[str, Any], failure_reason: str | None
) -> dict[str, Any]:
    if failure_reason is None:
        return terminal
    terminal.update(
        {
            "state": "nan_inf_numerical_collapse",
            "numerical_nonfinite": True,
            "fixed_budget_completed": False,
            "explicit_terminal_classification": True,
        }
    )
    return terminal


def train_policy_method(
    *,
    policy: q2.SquashedGaussianPolicy,
    method: str,
    obs: np.ndarray,
    actions: np.ndarray,
    advantages: np.ndarray,
    train_indices: np.ndarray,
    audit_indices: np.ndarray,
    negative_indices: np.ndarray,
    q2_config: q2.E7Config,
    method_config: MethodConfig,
    max_steps: int,
    eval_interval: int,
    rollout_eval_interval: int,
    rollout_episodes: int,
    final_rollout_episodes: int,
    seed: int,
    device: torch.device,
    output_dir: Path,
    spec: DatasetSpec,
    obs_norm: q2.Normalizer,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=q2_config.actor_lr, weight_decay=q2_config.weight_decay
    )
    rng = np.random.default_rng(seed + 2000)
    rows: list[dict[str, Any]] = []
    eval_snapshot = [parameter.detach().clone() for parameter in policy.parameters()]
    last_eval_step = 0
    candidate_step: int | None = None
    extension_target: int | None = None
    failure_reason: str | None = None

    def rollout_for(step: int) -> dict[str, Any] | None:
        if step == 0 or step == max_steps or (
            rollout_eval_interval > 0 and step % rollout_eval_interval == 0
        ):
            episodes = final_rollout_episodes if step == max_steps else rollout_episodes
            return evaluate_rollouts(
                policy=policy,
                obs_norm=obs_norm,
                spec=spec,
                episodes=episodes,
                seed=seed * 100000 + step,
                device=device,
            )
        return None

    rows.append(
        _policy_eval_row(
            policy=policy,
            method=method,
            obs=obs,
            actions=actions,
            advantages=advantages,
            audit_indices=audit_indices,
            negative_indices=negative_indices,
            q2_config=q2_config,
            method_config=method_config,
            device=device,
            step=0,
            update_stats={"raw_per_step": 0.0, "rms_per_step": 0.0, "relative_per_step": 0.0},
            rollout=rollout_for(0),
        )
    )
    for step in range(1, max_steps + 1):
        batch = q2.sample_indices(rng, train_indices, q2_config.actor_batch_size)
        loss, diagnostics = benchmark_actor_loss(
            policy,
            q2.tensor(obs[batch], device),
            q2.tensor(actions[batch], device),
            q2.tensor(advantages[batch], device),
            method,
            method_config,
        )
        optimizer.zero_grad(set_to_none=True)
        loss_value = float(loss.detach().cpu())
        if not math.isfinite(loss_value):
            failure_reason = "nonfinite_train_loss"
            break
        loss.backward()
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(policy.parameters(), q2_config.max_gradient_norm)
            .detach()
            .cpu()
        )
        if not math.isfinite(grad_norm):
            failure_reason = "nonfinite_train_gradient"
            optimizer.zero_grad(set_to_none=True)
            break
        optimizer.step()
        if step % eval_interval == 0 or step == max_steps:
            update_stats = q2.parameter_update_statistics(
                eval_snapshot, policy.parameters(), step - last_eval_step
            )
            eval_snapshot = [parameter.detach().clone() for parameter in policy.parameters()]
            last_eval_step = step
            row = _policy_eval_row(
                policy=policy,
                method=method,
                obs=obs,
                actions=actions,
                advantages=advantages,
                audit_indices=audit_indices,
                negative_indices=negative_indices,
                q2_config=q2_config,
                method_config=method_config,
                device=device,
                step=step,
                update_stats=update_stats,
                rollout=rollout_for(step),
            )
            row.update({f"train_{key}": value for key, value in diagnostics.items()})
            rows.append(row)
            state_drifts = [
                q2.normalized_window_drift(rows, key, q2_config.audit_windows)
                for key in ("mean_abs", "sigma_mean", "phantom_distance_mean")
            ]
            if (
                candidate_step is None
                and step >= max(1, max_steps // 4)
                and 2 * step <= max_steps
                and all(value <= q2_config.actor_state_drift_tolerance for value in state_drifts)
                and float(row["relative_update_norm"]) <= q2_config.actor_update_tolerance
            ):
                candidate_step = step
                extension_target = 2 * step
    final_step = int(rows[-1]["step"])
    fixed_budget_completed = failure_reason is None and final_step == max_steps
    terminal = q2.classify_actor_terminal(
        rows,
        q2_config,
        candidate_step,
        bool(extension_target is not None and final_step >= extension_target),
        fixed_budget_completed=fixed_budget_completed,
    )
    # The shared E7 terminal classifier inspects the latest recorded audit row.
    # A non-finite loss/gradient can occur between audit rows, so explicitly carry
    # the training-loop failure into the terminal classification instead of
    # misreporting the last finite audit as a non-numerical stop.
    terminal = _apply_training_failure_to_terminal(terminal, failure_reason)
    terminal.update(
        {
            "method": method,
            "failure_reason": failure_reason,
            "scientific_status": PILOT_STATUS,
            "method_ranking_claim_allowed": False,
            "formal_evidence_allowed": False,
        }
    )
    torch.save(
        {
            "model": policy.state_dict(),
            "method": method,
            "step": final_step,
            "fixed_budget_completed": fixed_budget_completed,
            "failure_reason": failure_reason,
        },
        output_dir / "terminal_actor.pt",
    )
    write_csv(output_dir / "metrics.csv", rows)
    atomic_write_json(output_dir / "terminal_audit.json", terminal)
    summary = {
        "method": method,
        "step": final_step,
        "fixed_budget_completed": fixed_budget_completed,
        "terminal_state": terminal["state"],
        "task_performance_collapse": terminal["task_performance_collapse"],
        "support_boundary_event": terminal["support_boundary_event"],
        "numerical_nonfinite": terminal["numerical_nonfinite"],
        "failure_reason": failure_reason,
        "rollout_return_mean": rows[-1].get("rollout_return_mean"),
        "normalized_return": rows[-1].get("normalized_return"),
        "normalized_return_available": rows[-1].get("normalized_return_available"),
        "score_protocol": rows[-1].get("score_protocol"),
        "mean_boundary_fraction": rows[-1].get("mean_boundary_fraction"),
        "sigma_mean": rows[-1].get("sigma_mean"),
        "negative_weight_mean": rows[-1].get("audit_negative_weight_mean"),
        "near_negative_weight_mean": rows[-1].get("audit_near_negative_weight_mean"),
        "far_negative_weight_mean": rows[-1].get("audit_far_negative_weight_mean"),
    }
    atomic_write_json(output_dir / "summary.json", summary)
    return summary


def critic_worker(args: argparse.Namespace) -> int:
    config = load_bench_config(args.config)
    configure_worker_threads(args.cpus_per_worker)
    spec = next(item for item in config.datasets if item.id == args.dataset_id)
    worker_identity, identity_payload = verify_worker_identity(
        args,
        config,
        spec,
        worker="critic",
    )
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    dataset_manifest = worker_dataset_manifest(
        dataset_root, spec, getattr(args, "validated_datasets_manifest", None)
    )
    dataset_path = Path(dataset_manifest["path"])
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_dataset(dataset_path, spec, config.budget.max_transitions)
    q2_config, mode = make_q2_config(config, spec, dataset_path)
    device = q2.choose_device(args.device)
    q2.seed_everything(mode.canonical_critic_seed)
    split = q2.split_episode_indices(
        data.episode_ids,
        mode.canonical_critic_seed,
        q2_config.train_fraction,
        q2_config.validation_fraction,
    )
    obs_norm = q2.Normalizer.fit(data.observations[split["train"]])
    returns = q2.discounted_returns(
        data.rewards, data.terminals, data.timeouts, q2_config.gamma
    )
    critic, target_norm, critic_audit = q2.train_critic(
        data=data,
        split=split,
        obs_norm=obs_norm,
        returns=returns,
        config=q2_config,
        mode=mode,
        seed=mode.canonical_critic_seed,
        device=device,
        output_dir=output_dir / "critic",
    )
    advantage, advantage_manifest = q2.freeze_advantages(
        critic=critic,
        data=data,
        obs_norm=obs_norm,
        target_norm=target_norm,
        gamma=q2_config.gamma,
        standardize=q2_config.advantage_standardize,
        standardization_indices=split["train"],
        device=device,
        output_dir=output_dir / "frozen_advantage",
    )
    np.savez_compressed(output_dir / "episode_split.npz", **split)
    np.savez_compressed(
        output_dir / "normalizers.npz",
        observation_mean=obs_norm.mean,
        observation_std=obs_norm.std,
        target_mean=target_norm.mean,
        target_std=target_norm.std,
    )
    complete = {
        "worker": "critic",
        "experiment_id": EXPERIMENT_ID,
        "runner_version": RUNNER_VERSION,
        "protocol_version": config.protocol_version,
        "run_identity_sha256": run_identity_sha256(config),
        "worker_identity_sha256": worker_identity,
        "worker_identity_payload": identity_payload,
        "scientific_status": PILOT_STATUS,
        "dataset": dataset_manifest,
        "transitions": data.size,
        "episodes": int(np.max(data.episode_ids) + 1),
        "critic_fixed_budget_completed": critic_audit["fixed_budget_completed"],
        "advantage_manifest": advantage_manifest,
        "negative_fraction": float(np.mean(advantage < 0)),
        "completed_utc": utc_now(),
    }
    if not bool(critic_audit["fixed_budget_completed"]):
        atomic_write_json(
            output_dir / "WORKER_FAILED.json",
            {
                **complete,
                "worker_status": "failed_before_downstream_branching",
                "failure_reason": "canonical_critic_fixed_budget_incomplete",
            },
        )
        raise RuntimeError(
            f"canonical critic did not complete the frozen {config.budget.critic_steps}-step budget"
        )
    atomic_write_json(output_dir / "WORKER_COMPLETE.json", complete)
    return 0


def _load_critic_context(path: Path) -> tuple[dict[str, np.ndarray], q2.Normalizer, np.ndarray]:
    split_npz = np.load(path / "episode_split.npz")
    split = {key: np.asarray(split_npz[key], dtype=np.int64) for key in split_npz.files}
    norms = np.load(path / "normalizers.npz")
    obs_norm = q2.Normalizer(
        mean=np.asarray(norms["observation_mean"], dtype=np.float32),
        std=np.asarray(norms["observation_std"], dtype=np.float32),
    )
    advantages = np.load(path / "frozen_advantage" / "frozen_advantages.npz")
    return split, obs_norm, np.asarray(advantages["advantage"], dtype=np.float32)


def _actor_worker_context(
    args: argparse.Namespace,
) -> tuple[
    BenchConfig,
    DatasetSpec,
    dict[str, Any],
    q2.OfflineData,
    q2.E7Config,
    dict[str, np.ndarray],
    q2.Normalizer,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    torch.device,
]:
    config = load_bench_config(args.config)
    configure_worker_threads(args.cpus_per_worker)
    spec = next(item for item in config.datasets if item.id == args.dataset_id)
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    dataset_manifest = worker_dataset_manifest(
        dataset_root, spec, getattr(args, "validated_datasets_manifest", None)
    )
    dataset_path = Path(dataset_manifest["path"])
    data = load_dataset(dataset_path, spec, config.budget.max_transitions)
    q2_config, _ = make_q2_config(config, spec, dataset_path)
    split, obs_norm, advantages = _load_critic_context(
        Path(args.critic_dir).expanduser().resolve()
    )
    seed = int(args.seed)
    device = q2.choose_device(args.device)
    q2.seed_everything(seed)
    obs = obs_norm.transform(data.observations)
    rng = np.random.default_rng(seed + 5000)
    audit_indices = rng.choice(
        split["validation"],
        size=min(config.budget.audit_sample_size, len(split["validation"])),
        replace=False,
    )
    negative_pool = np.flatnonzero(advantages < 0)
    if len(negative_pool) == 0:
        raise RuntimeError("frozen critic produced no negative advantages")
    negative_indices = rng.choice(
        negative_pool,
        size=min(config.budget.audit_sample_size, len(negative_pool)),
        replace=False,
    )
    return (
        config,
        spec,
        dataset_manifest,
        data,
        q2_config,
        split,
        obs_norm,
        advantages,
        obs,
        audit_indices,
        negative_indices,
        device,
    )


def warmstart_worker(args: argparse.Namespace) -> int:
    (
        config,
        spec,
        dataset_manifest,
        data,
        q2_config,
        split,
        obs_norm,
        advantages,
        obs,
        audit_indices,
        negative_indices,
        device,
    ) = _actor_worker_context(args)
    if args.method not in config.methods.ids:
        raise ValueError(f"invalid branch method variant: {args.method}")
    seed = int(args.seed)
    worker_identity, identity_payload = verify_worker_identity(
        args,
        config,
        spec,
        worker="warmstart",
        seed=seed,
        method=WARMSTART_METHOD,
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    policy = q2.SquashedGaussianPolicy(
        obs.shape[1],
        data.actions.shape[1],
        q2_config.hidden_sizes,
        q2_config.log_std_min,
        q2_config.log_std_max,
        q2_config.action_clip_epsilon,
        q2_config.activation,
        q2_config.init_scheme,
        q2_config.init_gain,
    ).to(device)
    summary = train_policy_method(
        policy=policy,
        method="positive_only",
        obs=obs,
        actions=data.actions,
        advantages=advantages,
        train_indices=split["train"],
        audit_indices=audit_indices,
        negative_indices=negative_indices,
        q2_config=q2_config,
        method_config=config.methods,
        max_steps=config.budget.shared_positive_warmstart_steps,
        eval_interval=config.budget.actor_eval_interval,
        rollout_eval_interval=config.budget.rollout_eval_interval,
        rollout_episodes=config.budget.rollout_episodes,
        final_rollout_episodes=config.budget.final_rollout_episodes,
        seed=seed,
        device=device,
        output_dir=output_dir / "method",
        spec=spec,
        obs_norm=obs_norm,
    )
    summary.update(
        {
            "dataset_id": spec.id,
            "seed": seed,
            "stage_role": "shared_positive_only_warmstart",
            "total_actor_steps_at_stage_end": (
                config.budget.shared_positive_warmstart_steps
            ),
        }
    )
    atomic_write_json(output_dir / "summary.json", summary)
    checkpoint = output_dir / "method" / "terminal_actor.pt"
    complete = {
        "worker": "warmstart",
        "experiment_id": EXPERIMENT_ID,
        "runner_version": RUNNER_VERSION,
        "protocol_version": config.protocol_version,
        "run_identity_sha256": run_identity_sha256(config),
        "worker_identity_sha256": worker_identity,
        "worker_identity_payload": identity_payload,
        "scientific_status": PILOT_STATUS,
        "formal_evidence_allowed": False,
        "dataset": dataset_manifest,
        "seed": seed,
        "method": WARMSTART_METHOD,
        "training_objective": "positive_only",
        "summary": summary,
        "branch_checkpoint": {
            "path": str(checkpoint),
            "sha256": sha256_file(checkpoint),
        },
        "completed_utc": utc_now(),
    }
    if not bool(summary["fixed_budget_completed"]):
        atomic_write_json(
            output_dir / "WORKER_FAILED.json",
            {
                **complete,
                "worker_status": "failed_before_method_branching",
                "failure_reason": summary.get("failure_reason")
                or "shared_warmstart_fixed_budget_incomplete",
            },
        )
        raise RuntimeError(
            "shared Positive-only warm-start did not complete its frozen fixed budget"
        )
    atomic_write_json(output_dir / "WORKER_COMPLETE.json", complete)
    return 0


def branch_worker(args: argparse.Namespace) -> int:
    (
        config,
        spec,
        dataset_manifest,
        data,
        q2_config,
        split,
        obs_norm,
        advantages,
        obs,
        audit_indices,
        negative_indices,
        device,
    ) = _actor_worker_context(args)
    if args.method not in config.methods.ids:
        raise ValueError(f"invalid branch method variant: {args.method}")
    seed = int(args.seed)
    worker_identity, identity_payload = verify_worker_identity(
        args,
        config,
        spec,
        worker="branch",
        seed=seed,
        method=args.method,
    )
    warmstart_dir = Path(args.warmstart_dir).expanduser().resolve()
    marker = warmstart_dir / "WORKER_COMPLETE.json"
    if not marker.is_file():
        raise FileNotFoundError(f"shared warm-start worker is incomplete: {marker}")
    warmstart_payload = json.loads(marker.read_text())
    expected_warmstart_identity = worker_identity_sha256(
        config,
        spec,
        worker="warmstart",
        seed=seed,
        method=WARMSTART_METHOD,
    )
    if warmstart_payload.get("worker_identity_sha256") != expected_warmstart_identity:
        raise RuntimeError("shared warm-start identity does not match this branch protocol")
    warmstart_summary = warmstart_payload.get("summary", {})
    if not bool(warmstart_summary.get("fixed_budget_completed")):
        raise RuntimeError("shared Positive-only warm-start did not complete its fixed budget")
    if int(warmstart_summary.get("step", -1)) != config.budget.shared_positive_warmstart_steps:
        raise RuntimeError("shared Positive-only warm-start step count does not match protocol")
    checkpoint = Path(warmstart_payload["branch_checkpoint"]["path"])
    if sha256_file(checkpoint) != warmstart_payload["branch_checkpoint"]["sha256"]:
        raise RuntimeError("shared warm-start checkpoint SHA-256 mismatch")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    policy = q2.SquashedGaussianPolicy(
        obs.shape[1],
        data.actions.shape[1],
        q2_config.hidden_sizes,
        q2_config.log_std_min,
        q2_config.log_std_max,
        q2_config.action_clip_epsilon,
        q2_config.activation,
        q2_config.init_scheme,
        q2_config.init_gain,
    ).to(device)
    policy.load_state_dict(payload["model"])
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = train_policy_method(
        policy=policy,
        method=args.method,
        obs=obs,
        actions=data.actions,
        advantages=advantages,
        train_indices=split["train"],
        audit_indices=audit_indices,
        negative_indices=negative_indices,
        q2_config=q2_config,
        method_config=config.methods,
        max_steps=config.budget.method_continuation_steps,
        eval_interval=config.budget.actor_eval_interval,
        rollout_eval_interval=config.budget.rollout_eval_interval,
        rollout_episodes=config.budget.rollout_episodes,
        final_rollout_episodes=config.budget.final_rollout_episodes,
        seed=seed,
        device=device,
        output_dir=output_dir / "method",
        spec=spec,
        obs_norm=obs_norm,
    )
    summary.update(
        {
            "dataset_id": spec.id,
            "seed": seed,
            "scheduled_continuation_steps": config.budget.method_continuation_steps,
            "executed_continuation_steps": int(summary["step"]),
            "shared_warmstart_steps": config.budget.shared_positive_warmstart_steps,
            "scheduled_total_actor_steps": config.budget.total_actor_steps_per_method,
            "executed_total_actor_steps": (
                config.budget.shared_positive_warmstart_steps + int(summary["step"])
            ),
        }
    )
    atomic_write_json(output_dir / "summary.json", summary)
    complete = {
        "worker": "branch",
        "experiment_id": EXPERIMENT_ID,
        "runner_version": RUNNER_VERSION,
        "protocol_version": config.protocol_version,
        "run_identity_sha256": run_identity_sha256(config),
        "worker_identity_sha256": worker_identity,
        "worker_identity_payload": identity_payload,
        "scientific_status": PILOT_STATUS,
        "formal_evidence_allowed": False,
        "method_ranking_claim_allowed": False,
        "dataset": dataset_manifest,
        "seed": seed,
        "method": args.method,
        "summary": summary,
        "source_warmstart_checkpoint_sha256": warmstart_payload["branch_checkpoint"]["sha256"],
        "source_warmstart_worker_identity_sha256": expected_warmstart_identity,
        "completed_utc": utc_now(),
    }
    atomic_write_json(output_dir / "WORKER_COMPLETE.json", complete)
    return 0


def _worker_complete(
    path: Path,
    *,
    dataset_id: str,
    seed: int | None = None,
    method: str | None = None,
    worker: str | None = None,
    expected_worker_identity_sha256: str | None = None,
) -> bool:
    marker = path / "WORKER_COMPLETE.json"
    if not marker.is_file():
        return False
    try:
        payload = json.loads(marker.read_text())
    except Exception:
        return False
    if payload.get("experiment_id") != EXPERIMENT_ID:
        return False
    if payload.get("dataset", {}).get("id") != dataset_id:
        return False
    if seed is not None and int(payload.get("seed", -1)) != int(seed):
        return False
    if method is not None and payload.get("method") != method:
        return False
    if worker is not None and payload.get("worker") != worker:
        return False
    if (
        expected_worker_identity_sha256 is not None
        and payload.get("worker_identity_sha256") != expected_worker_identity_sha256
    ):
        return False
    return True


def prepare_worker_output(
    path: Path,
    *,
    resume: bool,
    dataset_id: str,
    expected_worker_identity_sha256: str,
    seed: int | None = None,
    method: str | None = None,
    worker: str,
) -> bool:
    if _worker_complete(
        path,
        dataset_id=dataset_id,
        seed=seed,
        method=method,
        worker=worker,
        expected_worker_identity_sha256=expected_worker_identity_sha256,
    ):
        if not resume:
            raise RuntimeError(
                f"completed worker output already exists at {path}; pass --resume or use a new work directory"
            )
        return True
    if path.exists() and any(path.iterdir()):
        if not resume:
            raise RuntimeError(
                f"incomplete or stale worker output exists at {path}; pass --resume or use a new work directory"
            )
        archive_root = path.parent / "_stale_worker_outputs"
        archive_root.mkdir(parents=True, exist_ok=True)
        archive = archive_root / (
            f"{worker}_{dataset_id}_seed_{seed}_method_{method}_{time.time_ns()}"
        )
        shutil.move(str(path), str(archive))
    return False

def _device_for_job(pool: tuple[str, ...], index: int) -> str:
    if not pool:
        return "cpu"
    return pool[index % len(pool)]


def _terminate_subprocess(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGTERM)
        else:
            proc.terminate()
        proc.wait(timeout=5)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if proc.poll() is None:
            try:
                if os.name == "posix":
                    os.killpg(proc.pid, signal.SIGKILL)
                else:
                    proc.kill()
            except ProcessLookupError:
                pass
            proc.wait(timeout=5)


def _run_subprocess_job(
    *,
    command: list[str],
    log_path: Path,
    cpus_per_worker: int,
    stop_event: threading.Event,
) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    threads = str(cpus_per_worker)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = threads
    started = time.time()
    if stop_event.is_set():
        return {
            "command": command,
            "returncode": 143,
            "elapsed_seconds": 0.0,
            "log": str(log_path),
            "cancelled_by_peer_failure": True,
        }
    with log_path.open("w") as log:
        proc = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            start_new_session=(os.name == "posix"),
        )
        cancelled = False
        while proc.poll() is None:
            if stop_event.wait(timeout=0.25):
                cancelled = True
                _terminate_subprocess(proc)
                break
        returncode = int(proc.wait())
    return {
        "command": command,
        "returncode": returncode,
        "elapsed_seconds": time.time() - started,
        "log": str(log_path),
        "cancelled_by_peer_failure": cancelled,
    }


def run_parallel_stage(
    jobs: Sequence[dict[str, Any]],
    *,
    max_workers: int,
    cpus_per_worker: int,
    stage: str,
    heartbeat_path: Path,
) -> list[dict[str, Any]]:
    if len(jobs) > 1 and max_workers <= 1:
        raise RuntimeError(
            f"{stage} has {len(jobs)} jobs but max_workers={max_workers}; "
            "top-level serial execution is forbidden by protocol"
        )
    max_workers = min(max_workers, len(jobs))
    results: list[dict[str, Any]] = []
    stop_event = threading.Event()
    first_failure: tuple[dict[str, Any], dict[str, Any]] | None = None
    atomic_write_json(
        heartbeat_path,
        {"stage": stage, "state": "running", "jobs": len(jobs), "completed": 0, "utc": utc_now()},
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _run_subprocess_job,
                command=job["command"],
                log_path=job["log_path"],
                cpus_per_worker=cpus_per_worker,
                stop_event=stop_event,
            ): job
            for job in jobs
        }
        for future in concurrent.futures.as_completed(futures):
            job = futures[future]
            if future.cancelled():
                result = {
                    "command": job["command"],
                    "returncode": 143,
                    "elapsed_seconds": 0.0,
                    "log": str(job["log_path"]),
                    "cancelled_by_peer_failure": True,
                }
            else:
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "command": job["command"],
                        "returncode": 1,
                        "elapsed_seconds": 0.0,
                        "log": str(job["log_path"]),
                        "cancelled_by_peer_failure": False,
                        "coordinator_exception": repr(exc),
                    }
            result["job_id"] = job["job_id"]
            results.append(result)
            atomic_write_json(
                heartbeat_path,
                {
                    "stage": stage,
                    "state": "running",
                    "jobs": len(jobs),
                    "completed": len(results),
                    "completed_job_ids": sorted(item["job_id"] for item in results),
                    "utc": utc_now(),
                },
            )
            if result["returncode"] != 0 and not result.get("cancelled_by_peer_failure"):
                if first_failure is None:
                    first_failure = (job, result)
                    stop_event.set()
                    for pending in futures:
                        pending.cancel()
    if first_failure is not None:
        job, result = first_failure
        atomic_write_json(
            heartbeat_path,
            {
                "stage": stage,
                "state": "failed",
                "jobs": len(jobs),
                "completed": len(results),
                "failed_job_id": job["job_id"],
                "returncode": result["returncode"],
                "log": result["log"],
                "utc": utc_now(),
            },
        )
        raise RuntimeError(
            f"parallel worker failed: {job['job_id']} returncode={result['returncode']} "
            f"log={result['log']}; peer workers were terminated"
        )
    atomic_write_json(
        heartbeat_path,
        {"stage": stage, "state": "complete", "jobs": len(jobs), "completed": len(results), "utc": utc_now()},
    )
    return sorted(results, key=lambda item: item["job_id"])


def aggregate_pilot(work_dir: Path, config: BenchConfig) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for spec in config.datasets:
        for seed in config.budget.seeds:
            for method in config.methods.ids:
                branch_dir = (
                    work_dir
                    / "branches"
                    / spec.id
                    / f"seed_{seed}"
                    / method
                )
                expected_identity = worker_identity_sha256(
                    config,
                    spec,
                    worker="branch",
                    seed=seed,
                    method=method,
                )
                if not _worker_complete(
                    branch_dir,
                    dataset_id=spec.id,
                    seed=seed,
                    method=method,
                    worker="branch",
                    expected_worker_identity_sha256=expected_identity,
                ):
                    raise RuntimeError(
                        "cannot aggregate incomplete or stale branch output: "
                        f"{branch_dir}"
                    )
                branch_path = branch_dir / "summary.json"
                if not branch_path.is_file():
                    raise FileNotFoundError(f"missing branch summary: {branch_path}")
                row = json.loads(branch_path.read_text())
                if int(row.get("scheduled_total_actor_steps", -1)) != (
                    config.budget.total_actor_steps_per_method
                ):
                    raise RuntimeError(
                        "branch summary does not report the frozen equal scheduled actor horizon: "
                        f"{branch_path}"
                    )
                executed = int(row.get("executed_total_actor_steps", -1))
                if bool(row.get("fixed_budget_completed")):
                    if executed != config.budget.total_actor_steps_per_method:
                        raise RuntimeError(
                            "fixed-budget branch has an inconsistent executed actor step count: "
                            f"{branch_path}"
                        )
                elif not bool(row.get("numerical_nonfinite")):
                    raise RuntimeError(
                        "branch stopped before the fixed budget without a registered NaN/Inf exception: "
                        f"{branch_path}"
                    )
                rows.append(row)
    write_csv(work_dir / "pilot_method_seed_summary.csv", rows)
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "runner_version": RUNNER_VERSION,
        "scientific_status": PILOT_STATUS,
        "formal_evidence_allowed": False,
        "method_ranking_claim_allowed": False,
        "datasets": [spec.id for spec in config.datasets],
        "seeds": list(config.budget.seeds),
        "method_variants": list(config.methods.ids),
        "method_families": list(PILOT_METHOD_FAMILIES),
        "task_seed_jobs": len(config.datasets) * len(config.budget.seeds),
        "task_seed_method_branch_jobs": (
            len(config.datasets) * len(config.budget.seeds) * len(config.methods.ids)
        ),
        "shared_positive_warmstart_steps": (
            config.budget.shared_positive_warmstart_steps
        ),
        "method_continuation_steps": config.budget.method_continuation_steps,
        "total_actor_steps_per_method": config.budget.total_actor_steps_per_method,
        "equal_scheduled_actor_horizon_across_methods": True,
        "early_termination_exception": "nan_inf_numerical_failure_only",
        "run_identity_sha256": run_identity_sha256(config),
        "parallel_contract": build_execution_plan(config, "pilot"),
        "reporting_separation": [
            "task_performance_collapse",
            "support_or_variance_boundary_event",
            "nan_inf_numerical_failure",
        ],
        "interpretation_boundary": (
            "Pilot checks implementation, runtime, and preliminary paired direction only. "
            "It may not select a new method family, retune per task, or populate the formal nine-task table."
        ),
        "completed_utc": utc_now(),
    }
    atomic_write_json(work_dir / "PILOT_COMPLETE.json", payload)
    return payload


def run_pilot(args: argparse.Namespace) -> int:
    config = load_bench_config(args.config)
    plan = build_execution_plan(config, "pilot")
    work_dir = Path(args.work_dir).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    run_identity = ensure_run_identity(work_dir, config, resume=args.resume)
    resolved_config = work_dir / "resolved_config.yaml"
    shutil.copy2(args.config, resolved_config)

    manifests = [validate_dataset_file(dataset_root, spec) for spec in config.datasets]
    atomic_write_json(work_dir / "DATASETS.json", manifests)
    atomic_write_json(work_dir / "EXECUTION_PLAN.json", plan)
    atomic_write_json(
        work_dir / "PREFLIGHT.json",
        preflight_pilot_runtime(config, manifests),
    )
    python = sys.executable
    runner_path = str(Path(__file__).resolve())

    critic_jobs: list[dict[str, Any]] = []
    for index, spec in enumerate(config.datasets):
        output = work_dir / "critics" / spec.id
        expected_identity = worker_identity_sha256(config, spec, worker="critic")
        if prepare_worker_output(
            output,
            resume=args.resume,
            dataset_id=spec.id,
            worker="critic",
            expected_worker_identity_sha256=expected_identity,
        ):
            continue
        critic_jobs.append(
            {
                "job_id": f"critic:{spec.id}",
                "log_path": work_dir / "logs" / f"critic_{spec.id}.log",
                "command": [
                    python,
                    runner_path,
                    "critic-worker",
                    "--config",
                    str(resolved_config),
                    "--dataset-root",
                    str(dataset_root),
                    "--validated-datasets-manifest",
                    str(work_dir / "DATASETS.json"),
                    "--dataset-id",
                    spec.id,
                    "--output-dir",
                    str(output),
                    "--device",
                    _device_for_job(config.parallel.device_pool, index),
                    "--cpus-per-worker",
                    str(config.parallel.critic_cpus_per_worker),
                    "--expected-worker-identity-sha256",
                    expected_identity,
                ],
            }
        )
    if critic_jobs:
        run_parallel_stage(
            critic_jobs,
            max_workers=config.parallel.critic_workers,
            cpus_per_worker=config.parallel.critic_cpus_per_worker,
            stage="parallel_canonical_critics",
            heartbeat_path=work_dir / "critic_stage_heartbeat.json",
        )

    warmstart_jobs: list[dict[str, Any]] = []
    job_index = 0
    for spec in config.datasets:
        critic_dir = work_dir / "critics" / spec.id
        critic_identity = worker_identity_sha256(config, spec, worker="critic")
        if not _worker_complete(
            critic_dir,
            dataset_id=spec.id,
            worker="critic",
            expected_worker_identity_sha256=critic_identity,
        ):
            raise RuntimeError(f"canonical critic is incomplete for {spec.id}")
        for seed in config.budget.seeds:
            output = work_dir / "warmstarts" / spec.id / f"seed_{seed}"
            expected_identity = worker_identity_sha256(
                config,
                spec,
                worker="warmstart",
                seed=seed,
                method=WARMSTART_METHOD,
            )
            if prepare_worker_output(
                output,
                resume=args.resume,
                dataset_id=spec.id,
                seed=seed,
                method=WARMSTART_METHOD,
                worker="warmstart",
                expected_worker_identity_sha256=expected_identity,
            ):
                job_index += 1
                continue
            warmstart_jobs.append(
                {
                    "job_id": f"warmstart:{spec.id}:seed_{seed}",
                    "log_path": work_dir / "logs" / f"warmstart_{spec.id}_seed_{seed}.log",
                    "command": [
                        python,
                        runner_path,
                        "warmstart-worker",
                        "--config",
                        str(resolved_config),
                        "--dataset-root",
                        str(dataset_root),
                        "--validated-datasets-manifest",
                        str(work_dir / "DATASETS.json"),
                        "--dataset-id",
                        spec.id,
                        "--seed",
                        str(seed),
                        "--critic-dir",
                        str(critic_dir),
                        "--output-dir",
                        str(output),
                        "--device",
                        _device_for_job(config.parallel.device_pool, job_index),
                        "--cpus-per-worker",
                        str(config.parallel.warmstart_cpus_per_worker),
                        "--expected-worker-identity-sha256",
                        expected_identity,
                    ],
                }
            )
            job_index += 1
    if warmstart_jobs:
        run_parallel_stage(
            warmstart_jobs,
            max_workers=config.parallel.warmstart_workers,
            cpus_per_worker=config.parallel.warmstart_cpus_per_worker,
            stage="parallel_shared_positive_warmstarts",
            heartbeat_path=work_dir / "warmstart_stage_heartbeat.json",
        )

    branch_jobs: list[dict[str, Any]] = []
    job_index = 0
    for spec in config.datasets:
        critic_dir = work_dir / "critics" / spec.id
        for seed in config.budget.seeds:
            warmstart_dir = work_dir / "warmstarts" / spec.id / f"seed_{seed}"
            warmstart_identity = worker_identity_sha256(
                config,
                spec,
                worker="warmstart",
                seed=seed,
                method=WARMSTART_METHOD,
            )
            if not _worker_complete(
                warmstart_dir,
                dataset_id=spec.id,
                seed=seed,
                method=WARMSTART_METHOD,
                worker="warmstart",
                expected_worker_identity_sha256=warmstart_identity,
            ):
                raise RuntimeError(
                    f"shared Positive-only warm-start is incomplete for {spec.id} seed {seed}"
                )
            for method in config.methods.ids:
                output = work_dir / "branches" / spec.id / f"seed_{seed}" / method
                expected_identity = worker_identity_sha256(
                    config,
                    spec,
                    worker="branch",
                    seed=seed,
                    method=method,
                )
                if prepare_worker_output(
                    output,
                    resume=args.resume,
                    dataset_id=spec.id,
                    seed=seed,
                    method=method,
                    worker="branch",
                    expected_worker_identity_sha256=expected_identity,
                ):
                    job_index += 1
                    continue
                branch_jobs.append(
                    {
                        "job_id": f"branch:{spec.id}:seed_{seed}:{method}",
                        "log_path": (
                            work_dir
                            / "logs"
                            / f"branch_{spec.id}_seed_{seed}_{method}.log"
                        ),
                        "command": [
                            python,
                            runner_path,
                            "branch-worker",
                            "--config",
                            str(resolved_config),
                            "--dataset-root",
                            str(dataset_root),
                            "--validated-datasets-manifest",
                            str(work_dir / "DATASETS.json"),
                            "--dataset-id",
                            spec.id,
                            "--seed",
                            str(seed),
                            "--method",
                            method,
                            "--critic-dir",
                            str(critic_dir),
                            "--warmstart-dir",
                            str(warmstart_dir),
                            "--output-dir",
                            str(output),
                            "--device",
                            _device_for_job(config.parallel.device_pool, job_index),
                            "--cpus-per-worker",
                            str(config.parallel.branch_cpus_per_worker),
                            "--expected-worker-identity-sha256",
                            expected_identity,
                        ],
                    }
                )
                job_index += 1
    if branch_jobs:
        run_parallel_stage(
            branch_jobs,
            max_workers=config.parallel.branch_workers,
            cpus_per_worker=config.parallel.branch_cpus_per_worker,
            stage="parallel_equal_horizon_method_continuations",
            heartbeat_path=work_dir / "branch_stage_heartbeat.json",
        )
    if run_identity != run_identity_sha256(config):
        raise RuntimeError("run identity changed during coordinator execution")
    aggregate_pilot(work_dir, config)
    return 0


def inspect_command(args: argparse.Namespace) -> int:
    config = load_bench_config(args.config)
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    rows = []
    for spec in config.datasets:
        manifest = validate_dataset_file(dataset_root, spec)
        data = load_dataset(Path(manifest["path"]), spec, args.max_transitions)
        manifest.update(
            {
                "transitions_loaded": data.size,
                "episodes_loaded": int(np.max(data.episode_ids) + 1),
                "observation_dim": int(data.observations.shape[1]),
                "action_dim": int(data.actions.shape[1]),
            }
        )
        rows.append(manifest)
    print(json.dumps({"datasets": rows, "plan": build_execution_plan(config, "pilot")}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run the registered pilot or show formal block")
    run.add_argument("--mode", choices=("pilot", "formal"), required=True)
    run.add_argument("--config", default="configs/e7_bench_pilot.yaml")
    run.add_argument("--dataset-root", required=True)
    run.add_argument("--work-dir", required=True)
    run.add_argument("--resume", action="store_true")
    plan = sub.add_parser("plan", help="print pilot or formal execution plan")
    plan.add_argument("--mode", choices=("pilot", "formal"), required=True)
    plan.add_argument("--config", default="configs/e7_bench_pilot.yaml")
    inspect = sub.add_parser("inspect", help="verify both pilot dataset files and loaders")
    inspect.add_argument("--config", default="configs/e7_bench_pilot.yaml")
    inspect.add_argument("--dataset-root", required=True)
    inspect.add_argument("--max-transitions", type=int, default=2048)
    critic = sub.add_parser("critic-worker", help=argparse.SUPPRESS)
    critic.add_argument("--config", required=True)
    critic.add_argument("--dataset-root", required=True)
    critic.add_argument("--validated-datasets-manifest")
    critic.add_argument("--dataset-id", required=True)
    critic.add_argument("--output-dir", required=True)
    critic.add_argument("--device", required=True)
    critic.add_argument("--cpus-per-worker", type=int, required=True)
    critic.add_argument("--expected-worker-identity-sha256", required=True)
    warmstart = sub.add_parser("warmstart-worker", help=argparse.SUPPRESS)
    warmstart.add_argument("--config", required=True)
    warmstart.add_argument("--dataset-root", required=True)
    warmstart.add_argument("--validated-datasets-manifest")
    warmstart.add_argument("--dataset-id", required=True)
    warmstart.add_argument("--seed", type=int, required=True)
    warmstart.add_argument("--critic-dir", required=True)
    warmstart.add_argument("--output-dir", required=True)
    warmstart.add_argument("--device", required=True)
    warmstart.add_argument("--cpus-per-worker", type=int, required=True)
    warmstart.add_argument("--expected-worker-identity-sha256", required=True)
    branch = sub.add_parser("branch-worker", help=argparse.SUPPRESS)
    branch.add_argument("--config", required=True)
    branch.add_argument("--dataset-root", required=True)
    branch.add_argument("--validated-datasets-manifest")
    branch.add_argument("--dataset-id", required=True)
    branch.add_argument("--seed", type=int, required=True)
    branch.add_argument("--method", required=True)
    branch.add_argument("--critic-dir", required=True)
    branch.add_argument("--warmstart-dir", required=True)
    branch.add_argument("--output-dir", required=True)
    branch.add_argument("--device", required=True)
    branch.add_argument("--cpus-per-worker", type=int, required=True)
    branch.add_argument("--expected-worker-identity-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "plan":
        print(json.dumps(build_execution_plan(load_bench_config(args.config), args.mode), indent=2))
        return 0
    if args.command == "inspect":
        return inspect_command(args)
    if args.command == "critic-worker":
        return critic_worker(args)
    if args.command == "warmstart-worker":
        return warmstart_worker(args)
    if args.command == "branch-worker":
        return branch_worker(args)
    if args.mode == "formal":
        config = load_bench_config(args.config)
        print(json.dumps(build_execution_plan(config, "formal"), indent=2))
        raise SystemExit(
            "Formal E7-BENCH launch remains blocked until the full protocol lock is registered."
        )
    return run_pilot(args)


if __name__ == "__main__":
    raise SystemExit(main())
