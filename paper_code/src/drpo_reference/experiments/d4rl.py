"""D4RL-9 canonical performance adapter and fail-closed planning.

The repository's vendored ``SNA2C_IQLV_ExpRankAgent`` remains the single
performance implementation.  This module exposes that source through the
paper-facing task catalog and runner without copying a second trainer.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from drpo_reference.common.io import atomic_json
from drpo_reference.external.d4rl_tasks import (
    D4RL9_TASKS,
    D4RLTaskSpec,
    validate_d4rl9_matrix,
    validate_dataset_path,
)

D4RL9_EXPERIMENT_ID = "EXT-H-E7-BENCH-01"
D4RL9_RUNNER_VERSION = "0.3.0-canonical-exprank-adapter"


@dataclass(frozen=True)
class D4RLPerformanceBackendSpec:
    backend_id: str
    algorithm_family: str
    source_paths: tuple[str, ...]
    implementation_selected: bool
    implementation_migrated: bool
    protocol_status: str
    protocol_frozen: bool
    formal_task_matrix_eligible: bool
    mechanism_runner_reusable: bool
    shared_contracts: tuple[str, ...]
    distinct_contracts: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.backend_id or not self.algorithm_family:
            raise ValueError("D4RL backend identity must be non-empty")
        if not self.source_paths:
            raise ValueError("D4RL backend source provenance is required")
        if self.implementation_migrated and not self.implementation_selected:
            raise ValueError("a migrated backend must first be selected")
        if self.mechanism_runner_reusable:
            raise ValueError(
                "Hopper mechanism runner is not the performance backend"
            )
        if set(self.shared_contracts) & set(self.distinct_contracts):
            raise ValueError("D4RL shared and distinct contracts overlap")


CANONICAL_EXPRANK_BACKEND = D4RLPerformanceBackendSpec(
    backend_id="canonical_sna2c_iqlv_exprank",
    algorithm_family="SNA2C_IQLV_ExpRank",
    source_paths=(
        "src/drpo/e7_canonical_vendor/d4rl/agents.py",
        "src/drpo/e7_canonical_vendor/d4rl/train_sna2c_variant.py",
        "src/drpo/e7_canonical_vendor/d4rl/d4rl_common/train_loop.py",
        "src/drpo/e7_canonical_vendor/d4rl/d4rl_common/normalize.py",
    ),
    implementation_selected=True,
    implementation_migrated=True,
    protocol_status="selected_backend_adapter_migrated_protocol_unfrozen",
    protocol_frozen=False,
    formal_task_matrix_eligible=False,
    mechanism_runner_reusable=False,
    shared_contracts=(
        "d4rl_v2_dataset_identity",
        "locomotion_task_catalog",
        "gymnasium_mujoco_rollout_boundary",
        "d4rl_reference_score_normalization",
        "event_taxonomy",
    ),
    distinct_contracts=(
        "actor_likelihood_contract",
        "critic_update_lifecycle",
        "advantage_lifecycle",
        "optimizer_schedule",
        "method_matrix",
        "terminal_audit_protocol",
    ),
)
LEGACY_CANONICAL_BACKEND_CANDIDATE = CANONICAL_EXPRANK_BACKEND


@dataclass(frozen=True)
class CanonicalExpRankRunConfig:
    """Legacy trainer arguments; none are the formal protocol here."""

    steps: int
    batch_size: int
    learning_rate: float
    alpha: float
    tau: float
    temperature: float
    eval_interval: int
    eval_episodes: int
    checkpoint_interval: int
    checkpoint_last_fraction: float
    omp_threads: int = 2

    def __post_init__(self) -> None:
        integer_fields = (
            self.steps,
            self.batch_size,
            self.eval_interval,
            self.eval_episodes,
            self.checkpoint_interval,
            self.omp_threads,
        )
        if any(value <= 0 for value in integer_fields):
            raise ValueError(
                "canonical ExpRank integer controls must be positive"
            )
        if self.learning_rate <= 0.0 or self.alpha < 0.0:
            raise ValueError("canonical ExpRank lr/alpha are invalid")
        if not 0.5 <= self.tau < 1.0:
            raise ValueError("canonical ExpRank tau must be in [0.5, 1)")
        if self.temperature < 0.0:
            raise ValueError(
                "canonical ExpRank temperature must be non-negative"
            )
        if not 0.0 < self.checkpoint_last_fraction <= 1.0:
            raise ValueError("checkpoint_last_fraction must be in (0, 1]")


def repository_root() -> Path:
    return Path(__file__).resolve().parents[4]


def canonical_source_root(
    repo_root: str | Path | None = None,
) -> Path:
    root = repository_root() if repo_root is None else Path(repo_root).resolve()
    source = root / "src" / "drpo" / "e7_canonical_vendor" / "d4rl"
    if not source.is_dir():
        raise FileNotFoundError(
            f"canonical D4RL source root is missing: {source}"
        )
    return source


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_backend_provenance(
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    root = repository_root() if repo_root is None else Path(repo_root).resolve()
    files: dict[str, dict[str, Any]] = {}
    for relative in CANONICAL_EXPRANK_BACKEND.source_paths:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(
                f"canonical backend source is missing: {path}"
            )
        files[relative] = {
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    return {
        "backend": asdict(CANONICAL_EXPRANK_BACKEND),
        "source_files": files,
    }


def load_canonical_exprank_module(
    repo_root: str | Path | None = None,
) -> ModuleType:
    source = canonical_source_root(repo_root) / "agents.py"
    name = "_drpo_reference_canonical_d4rl_agents"
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"cannot load canonical D4RL agents: {source}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "SNA2C_IQLV_ExpRankAgent"):
        raise AttributeError(
            "canonical source lacks SNA2C_IQLV_ExpRankAgent"
        )
    return module


def load_canonical_exprank_agent_class(
    repo_root: str | Path | None = None,
) -> type[Any]:
    return load_canonical_exprank_module(
        repo_root
    ).SNA2C_IQLV_ExpRankAgent


def build_canonical_exprank_command(
    *,
    task: D4RLTaskSpec,
    dataset_path: str | Path,
    output_root: str | Path,
    seed: int,
    config: CanonicalExpRankRunConfig,
    repo_root: str | Path | None = None,
) -> list[str]:
    source = canonical_source_root(repo_root)
    output = Path(output_root).resolve()
    return [
        sys.executable,
        str(source / "train_sna2c_variant.py"),
        "--dataset",
        task.dataset_id,
        "--hdf5",
        str(Path(dataset_path).resolve()),
        "--variant",
        "iqlv_exp_rank",
        "--alpha",
        repr(config.alpha),
        "--tau",
        repr(config.tau),
        "--temp",
        repr(config.temperature),
        "--steps",
        str(config.steps),
        "--batch",
        str(config.batch_size),
        "--lr",
        repr(config.learning_rate),
        "--eval_interval",
        str(config.eval_interval),
        "--eval_episodes",
        str(config.eval_episodes),
        "--seed",
        str(int(seed)),
        "--out_dir",
        str(output),
        "--ckpt_dir",
        str(output / "ckpts"),
        "--ckpt_interval",
        str(config.checkpoint_interval),
        "--last_pct",
        repr(config.checkpoint_last_fraction),
    ]


def run_canonical_exprank_task(
    *,
    task: D4RLTaskSpec,
    backend: D4RLPerformanceBackendSpec,
    dataset_path: str | Path,
    output_root: str | Path,
    seeds: Sequence[int],
    config: CanonicalExpRankRunConfig,
    execute: bool = False,
    repo_root: str | Path | None = None,
    formal_evidence_eligible: bool = False,
    method_ranking_claim_allowed: bool = False,
) -> dict[str, Any]:
    """Plan or execute one task through the canonical trainer."""

    if backend != CANONICAL_EXPRANK_BACKEND:
        raise ValueError("unsupported D4RL performance backend")
    if formal_evidence_eligible or method_ranking_claim_allowed:
        raise RuntimeError("formal D4RL execution is not authorized")
    identity = validate_dataset_path(
        dataset_path,
        task,
        require_verified_sha=False,
    )
    root = Path(output_root).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(
            f"D4RL task output must be new or empty: {root}"
        )
    root.mkdir(parents=True, exist_ok=True)
    provenance = canonical_backend_provenance(repo_root)
    runs: dict[str, Any] = {}
    for seed in tuple(int(value) for value in seeds):
        run_root = root / f"seed_{seed}"
        command = build_canonical_exprank_command(
            task=task,
            dataset_path=dataset_path,
            output_root=run_root,
            seed=seed,
            config=config,
            repo_root=repo_root,
        )
        record: dict[str, Any] = {
            "seed": seed,
            "command": command,
            "execution_kind": "non_formal" if execute else "plan_only",
            "formal_result_claim": False,
            "method_ranking_claim_allowed": False,
        }
        if execute:
            environment = os.environ.copy()
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
            ):
                environment[name] = str(config.omp_threads)
            completed = subprocess.run(
                command,
                cwd=str(canonical_source_root(repo_root)),
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            record.update(
                {
                    "returncode": completed.returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                    "status": (
                        "completed"
                        if completed.returncode == 0
                        else "failed"
                    ),
                }
            )
        atomic_json(root / f"seed_{seed}_RUN.json", record)
        runs[str(seed)] = record
    result = {
        "task": asdict(task),
        "dataset_identity": identity,
        "provenance": provenance,
        "runs": runs,
        "formal_result_claim": False,
        "method_ranking_claim_allowed": False,
    }
    atomic_json(root / "TASK_RESULT.json", result)
    return result


@dataclass(frozen=True)
class D4RL9ExecutionPlan:
    tasks: tuple[D4RLTaskSpec, ...]
    dataset_paths: dict[str, Path]
    seeds: tuple[int, ...]
    backend: D4RLPerformanceBackendSpec
    execution_kind: str
    dataset_identity_complete: bool
    performance_protocol_frozen: bool
    backend_protocol_complete: bool
    formal_evidence_eligible: bool
    method_ranking_claim_allowed: bool
    blocked_reasons: tuple[str, ...]

    def as_manifest(self) -> dict[str, Any]:
        return {
            "experiment_id": D4RL9_EXPERIMENT_ID,
            "runner_version": D4RL9_RUNNER_VERSION,
            "execution_kind": self.execution_kind,
            "tasks": [asdict(task) for task in self.tasks],
            "dataset_paths": {
                task_id: str(path)
                for task_id, path in self.dataset_paths.items()
            },
            "seeds": list(self.seeds),
            "backend": asdict(self.backend),
            "dataset_identity_complete": self.dataset_identity_complete,
            "performance_protocol_frozen": (
                self.performance_protocol_frozen
            ),
            "backend_protocol_complete": self.backend_protocol_complete,
            "formal_evidence_eligible": self.formal_evidence_eligible,
            "method_ranking_claim_allowed": (
                self.method_ranking_claim_allowed
            ),
            "blocked_reasons": list(self.blocked_reasons),
            "shared_task_data_rollout_boundary": True,
            "single_canonical_trainer_across_d4rl9_tasks": True,
            "shared_training_engine_with_hopper_mechanism": False,
            "separate_per_task_trainers_allowed": False,
        }


def resolve_d4rl9_execution(
    *,
    dataset_paths: Mapping[str, str | Path],
    seeds: Sequence[int],
    tasks: Sequence[D4RLTaskSpec] = D4RL9_TASKS,
    backend: D4RLPerformanceBackendSpec = CANONICAL_EXPRANK_BACKEND,
    performance_protocol_frozen: bool = False,
    smoke: bool = False,
) -> D4RL9ExecutionPlan:
    resolved_tasks = validate_d4rl9_matrix(tasks)
    resolved_seeds = tuple(int(seed) for seed in seeds)
    if not resolved_seeds:
        raise ValueError("at least one D4RL-9 seed is required")
    if len(set(resolved_seeds)) != len(resolved_seeds):
        raise ValueError("D4RL-9 seed list contains duplicates")
    expected_ids = {task.task_id for task in resolved_tasks}
    actual_ids = set(dataset_paths)
    missing = sorted(expected_ids - actual_ids)
    extra = sorted(actual_ids - expected_ids)
    if missing or extra:
        raise ValueError(
            "D4RL-9 dataset mapping mismatch; "
            f"missing={missing}, extra={extra}"
        )
    resolved_paths = {
        task.task_id: Path(
            dataset_paths[task.task_id]
        ).resolve()
        for task in resolved_tasks
    }
    unresolved = tuple(
        task.task_id
        for task in resolved_tasks
        if not task.dataset_identity_verified
    )
    blocked: list[str] = []
    if unresolved:
        blocked.append(
            "unresolved_dataset_sha256:" + ",".join(unresolved)
        )
    if not backend.implementation_selected:
        blocked.append("d4rl9_performance_backend_not_selected")
    if not backend.implementation_migrated:
        blocked.append("d4rl9_performance_backend_not_migrated")
    if not backend.protocol_frozen:
        blocked.append(
            "d4rl9_performance_backend_protocol_not_frozen"
        )
    if not backend.formal_task_matrix_eligible:
        blocked.append("d4rl9_backend_not_formal_matrix_eligible")
    if not performance_protocol_frozen:
        blocked.append("d4rl9_performance_protocol_not_frozen")
    if len(resolved_seeds) != 10:
        blocked.append("manuscript_ten_run_coordinate_not_complete")
    if smoke:
        blocked.append("smoke_is_not_scientific_evidence")
    dataset_complete = not unresolved
    backend_complete = bool(
        backend.implementation_selected
        and backend.implementation_migrated
        and backend.protocol_frozen
        and backend.formal_task_matrix_eligible
    )
    formal_eligible = bool(
        not smoke
        and dataset_complete
        and backend_complete
        and performance_protocol_frozen
        and len(resolved_seeds) == 10
    )
    return D4RL9ExecutionPlan(
        tasks=resolved_tasks,
        dataset_paths=resolved_paths,
        seeds=resolved_seeds,
        backend=backend,
        execution_kind=(
            "formal_candidate"
            if formal_eligible
            else "blocked_or_non_evidence"
        ),
        dataset_identity_complete=dataset_complete,
        performance_protocol_frozen=performance_protocol_frozen,
        backend_protocol_complete=backend_complete,
        formal_evidence_eligible=formal_eligible,
        method_ranking_claim_allowed=False,
        blocked_reasons=tuple(blocked),
    )


def validate_d4rl9_datasets(
    plan: D4RL9ExecutionPlan,
    *,
    require_formal_identity: bool,
) -> dict[str, dict[str, object]]:
    return {
        task.task_id: validate_dataset_path(
            plan.dataset_paths[task.task_id],
            task,
            require_verified_sha=require_formal_identity,
        )
        for task in plan.tasks
    }


def dispatch_d4rl9(
    *,
    plan: D4RL9ExecutionPlan,
    output_root: str | Path,
    task_runner: Any,
    allow_non_evidence: bool = False,
) -> dict[str, Any]:
    if plan.blocked_reasons and not allow_non_evidence:
        raise RuntimeError(
            "D4RL-9 dispatch is blocked: "
            + "; ".join(plan.blocked_reasons)
        )
    output = Path(output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"D4RL-9 output must be new or empty: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    results = {
        task.task_id: task_runner(
            task=task,
            backend=plan.backend,
            dataset_path=plan.dataset_paths[task.task_id],
            output_root=output / task.task_id,
            seeds=plan.seeds,
            formal_evidence_eligible=(
                plan.formal_evidence_eligible
            ),
            method_ranking_claim_allowed=False,
        )
        for task in plan.tasks
    }
    return {
        "plan": plan.as_manifest(),
        "tasks": results,
        "formal_result_claim": False,
        "method_ranking_claim_allowed": False,
        "shared_task_data_rollout_boundary": True,
        "single_canonical_trainer_across_d4rl9_tasks": True,
        "shared_training_engine_with_hopper_mechanism": False,
    }
