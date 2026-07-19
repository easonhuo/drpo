"""Public experiment entry points for the paper-facing reference package."""

from __future__ import annotations

import importlib
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
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
    SNA2CIQLVExpRankAgent,
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


def _reset_rollout_env(env: Any, seed: int) -> np.ndarray:
    result = env.reset(seed=int(seed))
    observation = result[0] if isinstance(result, tuple) else result
    array = np.asarray(observation, dtype=np.float32).reshape(-1)
    if not np.all(np.isfinite(array)):
        raise RuntimeError("D4RL rollout reset returned non-finite observation")
    return array


def _step_rollout_env(
    env: Any,
    action: np.ndarray,
) -> tuple[np.ndarray, float, bool]:
    result = env.step(action)
    if not isinstance(result, tuple):
        raise RuntimeError("D4RL env.step must return a tuple")
    if len(result) == 5:
        observation, reward, terminated, truncated, _info = result
        done = bool(terminated or truncated)
    elif len(result) == 4:
        observation, reward, done, _info = result
        done = bool(done)
    else:
        raise RuntimeError(
            f"D4RL env.step returned {len(result)} values; expected 4 or 5"
        )
    array = np.asarray(observation, dtype=np.float32).reshape(-1)
    scalar_reward = float(reward)
    if not np.all(np.isfinite(array)) or not math.isfinite(scalar_reward):
        raise RuntimeError("D4RL rollout produced non-finite transition")
    return array, scalar_reward, done


def _normalized_d4rl_score(
    raw_return: float,
    task: D4RLTaskSpec,
) -> float:
    minimum = task.normalized_score_reference_min
    maximum = task.normalized_score_reference_max
    return 100.0 * (float(raw_return) - minimum) / (maximum - minimum)


def evaluate_d4rl_agent(
    *,
    agent: SNA2CIQLVExpRankAgent,
    task: D4RLTaskSpec,
    observation_dim: int,
    action_dim: int,
    episodes: int,
    seed: int,
    max_steps: int = 1000,
) -> dict[str, Any]:
    """Evaluate one trained ExpRank actor in its Gymnasium MuJoCo task."""

    if episodes <= 0 or max_steps <= 0:
        raise ValueError("D4RL rollout episodes and max_steps must be positive")
    try:
        gymnasium = importlib.import_module("gymnasium")
        env = gymnasium.make(task.env_id)
    except Exception as exc:
        raise RuntimeError(
            f"Gymnasium/MuJoCo rollout unavailable for {task.env_id}: {exc}"
        ) from exc

    raw_returns: list[float] = []
    normalized_scores: list[float] = []
    episode_lengths: list[int] = []
    try:
        action_space = getattr(env, "action_space", None)
        low = getattr(action_space, "low", None)
        high = getattr(action_space, "high", None)
        if low is None or high is None:
            raise RuntimeError("D4RL environment action bounds are unavailable")
        low_array = np.asarray(low, dtype=np.float32).reshape(-1)
        high_array = np.asarray(high, dtype=np.float32).reshape(-1)
        if low_array.size != action_dim or high_array.size != action_dim:
            raise RuntimeError(
                "D4RL environment action dimension does not match dataset"
            )
        env_limit = getattr(getattr(env, "spec", None), "max_episode_steps", None)
        step_limit = (
            min(int(env_limit), int(max_steps))
            if isinstance(env_limit, int) and env_limit > 0
            else int(max_steps)
        )
        for episode in range(episodes):
            observation = _reset_rollout_env(env, seed + episode)
            if observation.size != observation_dim:
                raise RuntimeError(
                    "D4RL environment observation dimension does not match dataset"
                )
            total = 0.0
            length = 0
            done = False
            while not done and length < step_limit:
                action, _log_probability = agent.get_action(observation)
                action = np.asarray(action, dtype=np.float32).reshape(-1)
                if action.size != action_dim or not np.all(np.isfinite(action)):
                    raise RuntimeError("D4RL actor returned invalid action")
                clipped = np.clip(action, low_array, high_array)
                observation, reward, done = _step_rollout_env(env, clipped)
                if observation.size != observation_dim:
                    raise RuntimeError(
                        "D4RL rollout observation dimension changed during episode"
                    )
                total += reward
                length += 1
            if length <= 0 or not math.isfinite(total):
                raise RuntimeError("D4RL rollout produced an invalid episode")
            raw_returns.append(total)
            normalized_scores.append(_normalized_d4rl_score(total, task))
            episode_lengths.append(length)
    finally:
        try:
            env.close()
        except Exception:
            pass

    raw = np.asarray(raw_returns, dtype=np.float64)
    normalized = np.asarray(normalized_scores, dtype=np.float64)
    return {
        "status": "completed",
        "backend": task.rollout_backend,
        "dataset_id": task.dataset_id,
        "env_id": task.env_id,
        "episodes": int(episodes),
        "seed": int(seed),
        "max_steps": int(max_steps),
        "episode_lengths": episode_lengths,
        "raw_returns": raw_returns,
        "normalized_scores": normalized_scores,
        "raw_return_mean": float(raw.mean()),
        "raw_return_std": float(raw.std(ddof=0)),
        "normalized_score_mean": float(normalized.mean()),
        "normalized_score_std": float(normalized.std(ddof=0)),
    }


def _aggregate_task_evaluations(
    runs: Sequence[dict[str, Any]],
) -> dict[str, Any] | None:
    evaluations = [run.get("evaluation") for run in runs]
    if not evaluations or any(evaluation is None for evaluation in evaluations):
        return None
    raw_means = np.asarray(
        [float(evaluation["raw_return_mean"]) for evaluation in evaluations],
        dtype=np.float64,
    )
    normalized_means = np.asarray(
        [
            float(evaluation["normalized_score_mean"])
            for evaluation in evaluations
        ],
        dtype=np.float64,
    )
    return {
        "seed_count": len(evaluations),
        "raw_return_mean_across_seeds": float(raw_means.mean()),
        "raw_return_std_across_seeds": float(raw_means.std(ddof=0)),
        "normalized_score_mean_across_seeds": float(normalized_means.mean()),
        "normalized_score_std_across_seeds": float(normalized_means.std(ddof=0)),
    }


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
    eval_episodes: int = 0,
    eval_max_steps: int = 1000,
) -> dict[str, Any]:
    """Run reviewer-facing ExpRank training and optional real rollouts."""

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
    if eval_episodes < 0 or eval_max_steps <= 0:
        raise ValueError("D4RL evaluation controls are invalid")

    tasks = _resolve_public_d4rl_tasks(task_ids)
    data_root = Path(dataset_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    if output.exists() and not output.is_dir():
        raise FileExistsError(f"D4RL output is not a directory: {output}")
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
        "scope": "reviewer_facing_algorithm_training_and_evaluation",
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
        "evaluation": {
            "episodes": int(eval_episodes),
            "max_steps": int(eval_max_steps),
            "configured": bool(eval_episodes > 0),
        },
        "formal_result_claim": False,
        "method_ranking_claim_allowed": False,
        "internal_formal_audit_included": False,
        "paper_artifact_binding_included": False,
    }
    atomic_json(output / "RUN_MANIFEST.json", manifest)

    task_results: dict[str, Any] = {}
    completed_runs = 0
    completed_evaluations = 0
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
                completed_runs += 1
                evaluation = None
                if eval_episodes > 0:
                    try:
                        evaluation = evaluate_d4rl_agent(
                            agent=result["agent"],
                            task=task,
                            observation_dim=dataset.observation_dim,
                            action_dim=dataset.action_dim,
                            episodes=eval_episodes,
                            seed=seed,
                            max_steps=eval_max_steps,
                        )
                        atomic_json(run_root / "EVALUATION.json", evaluation)
                        completed_evaluations += 1
                    except Exception as exc:
                        atomic_json(
                            run_root / "EVALUATION_FAILED.json",
                            {
                                "status": "failed",
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            },
                        )
                        raise
                runs.append(
                    {
                        "seed": seed,
                        "output_root": str(run_root),
                        "final_step": effective_steps,
                        "final_checkpoint": checkpoints[-1],
                        "finite": finite,
                        "training_completed": True,
                        "evaluation_completed": evaluation is not None,
                        "evaluation": evaluation,
                        "formal_result_claim": False,
                    }
                )
            task_results[task.task_id] = {
                "task": task.task_id,
                "dataset_identity": identity,
                "transition_count": dataset.size,
                "runs": runs,
                "evaluation_summary": _aggregate_task_evaluations(runs),
            }

        expected_runs = len(tasks) * len(resolved_seeds)
        evaluation_configured = eval_episodes > 0
        evaluation_completed = bool(
            evaluation_configured and completed_evaluations == expected_runs
        )
        summary = {
            "manifest": manifest,
            "tasks": task_results,
            "expected_runs": expected_runs,
            "completed_runs": completed_runs,
            "completed_evaluations": completed_evaluations,
            "training_completed": completed_runs == expected_runs,
            "evaluation_configured": evaluation_configured,
            "evaluation_completed": evaluation_completed,
            "formal_result_claim": False,
            "method_ranking_claim_allowed": False,
        }
        atomic_json(output / "SUMMARY.json", summary)
        atomic_json(
            output / "COMPLETED.json",
            {
                "status": (
                    "training_and_evaluation_completed_non_formal"
                    if evaluation_completed
                    else "training_completed_non_formal"
                ),
                "expected_runs": expected_runs,
                "completed_runs": completed_runs,
                "completed_evaluations": completed_evaluations,
                "training_completed": completed_runs == expected_runs,
                "evaluation_configured": evaluation_configured,
                "evaluation_completed": evaluation_completed,
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
                "completed_evaluations": completed_evaluations,
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
    "evaluate_d4rl_agent",
    "flatten_seed_summary",
    "prepare_canonical_critic_context",
    "resolve_hopper_execution",
    "run_d4rl",
    "run_hopper",
    "validate_dataset_identity",
]
