"""Public experiment entry points for the paper-facing reference package."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from drpo_reference.common.io import atomic_json
from drpo_reference.external.d4rl_tasks import (
    D4RL9_TASKS,
    D4RLTaskSpec,
    resolve_d4rl_task,
    validate_dataset_path,
)
from drpo_reference.external.hopper_data import load_hopper_hdf5

from .d4rl import (
    D4RL9_EXPERIMENT_ID,
    D4RL9_RUNNER_VERSION,
    CanonicalExpRankTrainingConfig,
    prepare_canonical_locomotion_dataset,
    train_canonical_exprank,
)
from .hopper import (
    CanonicalCriticContext,
    HopperExecutionPlan,
    aggregate_seed_summaries,
    build_root_terminal_audit,
    flatten_seed_summary,
    prepare_canonical_critic_context,
    resolve_hopper_execution,
    run_hopper,
    validate_dataset_identity,
)


def _resolve_public_d4rl_tasks(
    task_ids: Sequence[str] | None,
) -> tuple[D4RLTaskSpec, ...]:
    if task_ids is None:
        return D4RL9_TASKS
    resolved_ids = tuple(str(task_id) for task_id in task_ids)
    if not resolved_ids:
        raise ValueError("at least one D4RL task is required")
    if len(set(resolved_ids)) != len(resolved_ids):
        raise ValueError("D4RL task list contains duplicates")
    return tuple(resolve_d4rl_task(task_id) for task_id in resolved_ids)


def _resolve_public_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def run_d4rl(
    *,
    dataset_root: str | Path,
    output_root: str | Path,
    seeds: Sequence[int],
    steps: int | None,
    batch_size: int = 256,
    task_ids: Sequence[str] | None = None,
    device: str = "auto",
    smoke: bool = False,
) -> dict[str, Any]:
    """Run the reviewer-facing ExpRank trainer on selected D4RL-9 tasks.

    This entry point deliberately provides a readable train-and-checkpoint path,
    not the repository's internal formal-evidence governance. It always records
    a non-formal lightweight completion state. Real rollout evaluation is a
    separate reviewer-facing slice and is not silently replaced by training
    metrics.
    """

    resolved_seeds = tuple(int(seed) for seed in seeds)
    if not resolved_seeds:
        raise ValueError("at least one D4RL seed is required")
    if len(set(resolved_seeds)) != len(resolved_seeds):
        raise ValueError("D4RL seed list contains duplicates")
    if batch_size <= 0:
        raise ValueError("D4RL batch_size must be positive")
    if not smoke and steps is None:
        raise ValueError("--steps is required unless --smoke is used")
    if steps is not None and steps <= 0:
        raise ValueError("D4RL steps must be positive")

    tasks = _resolve_public_d4rl_tasks(task_ids)
    data_root = Path(dataset_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"D4RL output must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    effective_steps = 3 if smoke else int(steps)
    effective_batch_size = min(int(batch_size), 8) if smoke else int(batch_size)
    max_transitions = 64 if smoke else None
    resolved_device = _resolve_public_device(device)
    training_config = CanonicalExpRankTrainingConfig(
        steps=effective_steps,
        batch_size=effective_batch_size,
        eval_interval=effective_steps,
        checkpoint_interval=effective_steps,
        checkpoint_last_fraction=1.0,
    )
    manifest = {
        "experiment_id": D4RL9_EXPERIMENT_ID,
        "runner_version": D4RL9_RUNNER_VERSION,
        "scope": "reviewer_facing_algorithm_training",
        "tasks": [task.task_id for task in tasks],
        "seeds": list(resolved_seeds),
        "dataset_root": str(data_root),
        "output_root": str(output),
        "device": resolved_device,
        "smoke": bool(smoke),
        "training_config": {
            "steps": training_config.steps,
            "batch_size": training_config.batch_size,
            "learning_rate": training_config.learning_rate,
            "gamma": training_config.gamma,
            "alpha": training_config.alpha,
            "tau": training_config.tau,
            "temperature": training_config.temperature,
        },
        "rollout_evaluation_configured": False,
        "formal_result_claim": False,
        "method_ranking_claim_allowed": False,
        "internal_formal_audit_included": False,
        "paper_artifact_binding_included": False,
    }
    atomic_json(output / "RUN_MANIFEST.json", manifest)

    task_results: dict[str, Any] = {}
    completed_runs = 0
    try:
        for task in tasks:
            dataset_path = data_root / task.dataset_basename
            identity = validate_dataset_path(
                dataset_path,
                task,
                require_verified_sha=False,
            )
            offline_data = load_hopper_hdf5(
                dataset_path,
                max_transitions=max_transitions,
            )
            dataset = prepare_canonical_locomotion_dataset(offline_data)
            runs: list[dict[str, Any]] = []
            for seed in resolved_seeds:
                run_root = output / task.task_id / f"seed_{seed}"
                result = train_canonical_exprank(
                    dataset=dataset,
                    seed=seed,
                    config=training_config,
                    device=resolved_device,
                    output_root=run_root,
                )
                losses = [
                    float(record["loss"])
                    for record in result["loss_records"]
                ]
                finite = all(math.isfinite(loss) for loss in losses)
                if not finite:
                    raise FloatingPointError(
                        f"non-finite D4RL loss for {task.task_id}, seed={seed}"
                    )
                checkpoints = tuple(str(path) for path in result["checkpoints"])
                if not checkpoints:
                    raise RuntimeError(
                        f"D4RL run produced no checkpoint: "
                        f"{task.task_id}, seed={seed}"
                    )
                runs.append(
                    {
                        "seed": seed,
                        "output_root": str(run_root),
                        "final_step": effective_steps,
                        "final_checkpoint": checkpoints[-1],
                        "finite": finite,
                        "training_completed": True,
                        "evaluation_completed": False,
                        "formal_result_claim": False,
                    }
                )
                completed_runs += 1
            task_results[task.task_id] = {
                "task": task.task_id,
                "dataset_identity": identity,
                "transition_count": dataset.size,
                "runs": runs,
            }

        expected_runs = len(tasks) * len(resolved_seeds)
        summary = {
            "manifest": manifest,
            "tasks": task_results,
            "expected_runs": expected_runs,
            "completed_runs": completed_runs,
            "training_completed": completed_runs == expected_runs,
            "evaluation_completed": False,
            "formal_result_claim": False,
            "method_ranking_claim_allowed": False,
        }
        atomic_json(output / "SUMMARY.json", summary)
        atomic_json(
            output / "COMPLETED.json",
            {
                "status": "training_completed_non_formal",
                "expected_runs": expected_runs,
                "completed_runs": completed_runs,
                "training_completed": completed_runs == expected_runs,
                "evaluation_completed": False,
                "formal_result_claim": False,
                "method_ranking_claim_allowed": False,
            },
        )
        return summary
    except Exception as exc:
        atomic_json(
            output / "FAILED.json",
            {
                "status": "failed",
                "completed_runs": completed_runs,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "formal_result_claim": False,
                "method_ranking_claim_allowed": False,
            },
        )
        raise


__all__ = [
    "CanonicalCriticContext",
    "HopperExecutionPlan",
    "aggregate_seed_summaries",
    "build_root_terminal_audit",
    "flatten_seed_summary",
    "prepare_canonical_critic_context",
    "resolve_hopper_execution",
    "run_d4rl",
    "run_hopper",
    "validate_dataset_identity",
]
