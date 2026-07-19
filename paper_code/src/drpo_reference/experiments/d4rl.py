"""D4RL-9 performance-profile planning over the shared locomotion engine.

The Hopper mechanism runner and this performance profile share data, model,
critic, actor, and rollout primitives. This module owns only the nine-task
matrix and fail-closed dispatch. It does not authorize a formal run or a method
ranking while dataset provenance and the final performance protocol are open.
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
D4RL9_RUNNER_VERSION = "0.1.0"
TaskRunner = Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class D4RL9ExecutionPlan:
    """Resolved D4RL-9 matrix without pretending unresolved inputs are formal."""

    tasks: tuple[D4RLTaskSpec, ...]
    dataset_paths: dict[str, Path]
    seeds: tuple[int, ...]
    execution_kind: str
    dataset_identity_complete: bool
    performance_protocol_frozen: bool
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
            "dataset_identity_complete": self.dataset_identity_complete,
            "performance_protocol_frozen": self.performance_protocol_frozen,
            "formal_evidence_eligible": self.formal_evidence_eligible,
            "method_ranking_claim_allowed": self.method_ranking_claim_allowed,
            "blocked_reasons": list(self.blocked_reasons),
            "shared_locomotion_engine": True,
            "separate_d4rl_trainer_implemented": False,
        }


def resolve_d4rl9_execution(
    *,
    dataset_paths: Mapping[str, str | Path],
    seeds: Sequence[int],
    tasks: Sequence[D4RLTaskSpec] = D4RL9_TASKS,
    performance_protocol_frozen: bool = False,
    smoke: bool = False,
) -> D4RL9ExecutionPlan:
    """Resolve the complete matrix and expose every formal blocker explicitly."""

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
            f"D4RL-9 dataset mapping mismatch; missing={missing}, extra={extra}"
        )
    resolved_paths = {
        task.task_id: Path(dataset_paths[task.task_id]).expanduser().resolve()
        for task in resolved_tasks
    }

    unresolved = tuple(
        task.task_id for task in resolved_tasks if not task.dataset_identity_verified
    )
    blocked: list[str] = []
    if unresolved:
        blocked.append(
            "unresolved_dataset_sha256:" + ",".join(unresolved)
        )
    if not performance_protocol_frozen:
        blocked.append("d4rl9_performance_protocol_not_frozen")
    if len(resolved_seeds) != 10:
        blocked.append("manuscript_ten_run_coordinate_not_complete")
    if smoke:
        blocked.append("smoke_is_not_scientific_evidence")

    dataset_identity_complete = not unresolved
    formal_eligible = bool(
        not smoke
        and dataset_identity_complete
        and performance_protocol_frozen
        and len(resolved_seeds) == 10
    )
    return D4RL9ExecutionPlan(
        tasks=resolved_tasks,
        dataset_paths=resolved_paths,
        seeds=resolved_seeds,
        execution_kind=(
            "formal_candidate" if formal_eligible else "blocked_or_non_evidence"
        ),
        dataset_identity_complete=dataset_identity_complete,
        performance_protocol_frozen=performance_protocol_frozen,
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
    """Dispatch every task through one injected shared-engine runner.

    The caller supplies the single-task runner. This module deliberately does
    not implement an alternative actor/critic/training stack. Formal dispatch
    fails closed until every registered blocker is resolved.
    """

    if plan.blocked_reasons and not allow_non_evidence:
        raise RuntimeError(
            "D4RL-9 dispatch is blocked: " + "; ".join(plan.blocked_reasons)
        )
    output = Path(output_root).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"D4RL-9 output root must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {}
    for task in plan.tasks:
        results[task.task_id] = task_runner(
            task=task,
            dataset_path=plan.dataset_paths[task.task_id],
            output_root=output / task.task_id,
            seeds=plan.seeds,
            formal_evidence_eligible=plan.formal_evidence_eligible,
            method_ranking_claim_allowed=False,
        )
    return {
        "plan": plan.as_manifest(),
        "tasks": results,
        "formal_result_claim": False,
        "method_ranking_claim_allowed": False,
        "shared_locomotion_engine": True,
    }
