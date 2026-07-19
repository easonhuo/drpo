"""D4RL-9 performance planning with an explicit backend boundary.

Hopper E7-Q2 and the D4RL-9 benchmark share locomotion task, dataset,
rollout, and score-normalization contracts where those contracts are actually
equivalent. Their training backends are not assumed to be interchangeable.
This module keeps that distinction explicit and fails closed while the final
D4RL-9 backend and protocol remain unfrozen.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from drpo_reference.external.d4rl_tasks import (
    D4RL9_TASKS,
    D4RLTaskSpec,
    validate_d4rl9_matrix,
    validate_dataset_path,
)


D4RL9_EXPERIMENT_ID = "D4RL-9-PERFORMANCE"
D4RL9_RUNNER_VERSION = "0.2.0-backend-audit"
TaskRunner = Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class D4RLPerformanceBackendSpec:
    """Audited candidate backend for the D4RL-9 performance profile."""

    backend_id: str
    algorithm_family: str
    source_paths: tuple[str, ...]
    protocol_status: str
    protocol_frozen: bool
    formal_task_matrix_eligible: bool
    mechanism_runner_reusable: bool
    shared_contracts: tuple[str, ...]
    distinct_contracts: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.backend_id.strip():
            raise ValueError("D4RL performance backend_id must be non-empty")
        if not self.algorithm_family.strip():
            raise ValueError("D4RL performance algorithm_family must be non-empty")
        if not self.source_paths:
            raise ValueError("D4RL performance backend needs source provenance")
        if self.protocol_frozen and not self.formal_task_matrix_eligible:
            raise ValueError(
                "a frozen D4RL backend must declare formal matrix eligibility"
            )
        if self.mechanism_runner_reusable:
            raise ValueError(
                "the audited Hopper mechanism runner is not a D4RL-9 "
                "performance backend"
            )
        overlap = set(self.shared_contracts) & set(self.distinct_contracts)
        if overlap:
            raise ValueError(
                "D4RL backend contracts cannot be both shared and distinct: "
                f"{sorted(overlap)}"
            )


LEGACY_CANONICAL_BACKEND_CANDIDATE = D4RLPerformanceBackendSpec(
    backend_id="legacy_canonical_sna2c_iqlv_candidate",
    algorithm_family="SNA2C_IQLV_ExpRank",
    source_paths=(
        "src/drpo/e7_canonical_vendor/d4rl/agents.py",
        "src/drpo/e7_canonical_vendor/d4rl/train_sna2c_variant.py",
        "src/drpo/e7_canonical_shortlist_protocol.py",
    ),
    protocol_status="pilot_only_unfrozen",
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


@dataclass(frozen=True)
class D4RL9ExecutionPlan:
    """Resolved matrix without pretending unresolved inputs are formal."""

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
            "shared_full_training_engine": False,
            "mechanism_runner_reusable_for_performance": (
                self.backend.mechanism_runner_reusable
            ),
            "separate_per_task_trainers_allowed": False,
        }


def resolve_d4rl9_execution(
    *,
    dataset_paths: Mapping[str, str | Path],
    seeds: Sequence[int],
    tasks: Sequence[D4RLTaskSpec] = D4RL9_TASKS,
    backend: D4RLPerformanceBackendSpec = (
        LEGACY_CANONICAL_BACKEND_CANDIDATE
    ),
    performance_protocol_frozen: bool = False,
    smoke: bool = False,
) -> D4RL9ExecutionPlan:
    """Resolve the matrix and expose every formal blocker explicitly."""

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
        ).expanduser().resolve()
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
    if not backend.protocol_frozen:
        blocked.append("d4rl9_performance_backend_not_frozen")
    if not backend.formal_task_matrix_eligible:
        blocked.append("d4rl9_backend_not_formal_matrix_eligible")
    if not performance_protocol_frozen:
        blocked.append("d4rl9_performance_protocol_not_frozen")
    if len(resolved_seeds) != 10:
        blocked.append("manuscript_ten_run_coordinate_not_complete")
    if smoke:
        blocked.append("smoke_is_not_scientific_evidence")

    dataset_identity_complete = not unresolved
    backend_complete = bool(
        backend.protocol_frozen
        and backend.formal_task_matrix_eligible
        and not backend.mechanism_runner_reusable
    )
    formal_eligible = bool(
        not smoke
        and dataset_identity_complete
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
        dataset_identity_complete=dataset_identity_complete,
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
    """Validate all paths using the task catalog's exact identity state."""

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
    task_runner: TaskRunner,
    allow_non_evidence: bool = False,
) -> dict[str, Any]:
    """Dispatch all tasks through one backend-specific task runner.

    This function prevents per-task trainer copies. It does not claim that the
    Hopper mechanism trainer and the final D4RL-9 performance trainer are the
    same scientific backend.
    """

    if plan.blocked_reasons and not allow_non_evidence:
        raise RuntimeError(
            "D4RL-9 dispatch is blocked: "
            + "; ".join(plan.blocked_reasons)
        )
    output = Path(output_root).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"D4RL-9 output root must be new or empty: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {}
    for task in plan.tasks:
        results[task.task_id] = task_runner(
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
    return {
        "plan": plan.as_manifest(),
        "tasks": results,
        "formal_result_claim": False,
        "method_ranking_claim_allowed": False,
        "shared_task_data_rollout_boundary": True,
        "shared_full_training_engine": False,
    }
