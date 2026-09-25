"""Public experiment entry points for the paper-facing reference package."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from drpo_reference.common.io import atomic_json

from .d4rl import (
    D4RL9_TASKS,
    D4RLAgent,
    D4RLTaskSpec,
    D4RLTrainingConfig,
    load_d4rl_hdf5,
    prepare_canonical_locomotion_dataset,
    resolve_d4rl_task,
    train_drpo,
)


def _resolve_public_d4rl_tasks(
    task_ids: Sequence[str] | None,
) -> tuple[D4RLTaskSpec, ...]:
    if task_ids is None:
        return D4RL9_TASKS
    return tuple(resolve_d4rl_task(str(task_id)) for task_id in task_ids)


def _resolve_public_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _normalized_d4rl_score(raw_return: float, task: D4RLTaskSpec) -> float:
    minimum = task.normalized_score_reference_min
    maximum = task.normalized_score_reference_max
    return 100.0 * (float(raw_return) - minimum) / (maximum - minimum)


def evaluate_d4rl_agent(
    *,
    agent: D4RLAgent,
    task: D4RLTaskSpec,
    episodes: int,
    seed: int,
    max_steps: int = 1000,
) -> dict[str, Any]:
    """Evaluate the deterministic actor mean in the corresponding MuJoCo task."""

    gymnasium = importlib.import_module("gymnasium")
    env = gymnasium.make(task.env_id)
    low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
    env_limit = getattr(getattr(env, "spec", None), "max_episode_steps", None)
    step_limit = min(int(env_limit), max_steps) if env_limit else max_steps

    raw_returns: list[float] = []
    try:
        for episode in range(episodes):
            reset = env.reset(seed=int(seed) + episode)
            observation = np.asarray(
                reset[0] if isinstance(reset, tuple) else reset,
                dtype=np.float32,
            ).reshape(-1)
            total = 0.0
            length = 0
            done = False
            while not done and length < step_limit:
                action = agent.get_action(observation)
                step = env.step(np.clip(action, low, high))
                if len(step) == 5:
                    observation, reward, terminated, truncated, _ = step
                    done = bool(terminated or truncated)
                else:
                    observation, reward, done, _ = step
                observation = np.asarray(observation, dtype=np.float32).reshape(-1)
                total += float(reward)
                length += 1
            raw_returns.append(total)
    finally:
        env.close()

    normalized_scores = [_normalized_d4rl_score(value, task) for value in raw_returns]
    raw = np.asarray(raw_returns, dtype=np.float64)
    normalized = np.asarray(normalized_scores, dtype=np.float64)
    return {
        "episodes": int(episodes),
        "seed": int(seed),
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
    evaluations = [run["evaluation"] for run in runs if run["evaluation"] is not None]
    if len(evaluations) != len(runs):
        return None
    raw = np.asarray(
        [float(evaluation["raw_return_mean"]) for evaluation in evaluations],
        dtype=np.float64,
    )
    normalized = np.asarray(
        [float(evaluation["normalized_score_mean"]) for evaluation in evaluations],
        dtype=np.float64,
    )
    return {
        "seed_count": len(evaluations),
        "raw_return_mean_across_seeds": float(raw.mean()),
        "raw_return_std_across_seeds": float(raw.std(ddof=0)),
        "normalized_score_mean_across_seeds": float(normalized.mean()),
        "normalized_score_std_across_seeds": float(normalized.std(ddof=0)),
    }


def run_d4rl(
    *,
    dataset_root: str | Path,
    output_root: str | Path,
    seeds: Sequence[int],
    steps: int,
    batch_size: int = 256,
    task_ids: Sequence[str] | None = None,
    device: str = "auto",
    eval_episodes: int = 0,
    eval_max_steps: int = 1000,
) -> dict[str, Any]:
    """Train DRPO and optionally evaluate it in MuJoCo."""

    resolved_seeds = tuple(int(seed) for seed in seeds)
    tasks = _resolve_public_d4rl_tasks(task_ids)
    data_root = Path(dataset_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    effective_steps = int(steps)
    effective_batch_size = int(batch_size)
    resolved_device = _resolve_public_device(device)
    config = D4RLTrainingConfig(
        steps=effective_steps,
        batch_size=effective_batch_size,
    )

    task_results: dict[str, Any] = {}
    for task in tasks:
        dataset = prepare_canonical_locomotion_dataset(
            load_d4rl_hdf5(data_root / task.dataset_basename)
        )
        runs: list[dict[str, Any]] = []
        for seed in resolved_seeds:
            agent = train_drpo(
                dataset=dataset,
                seed=seed,
                config=config,
                device=resolved_device,
            )
            evaluation = None
            if eval_episodes > 0:
                evaluation = evaluate_d4rl_agent(
                    agent=agent,
                    task=task,
                    episodes=eval_episodes,
                    seed=seed,
                    max_steps=eval_max_steps,
                )
            runs.append({"seed": seed, "evaluation": evaluation})
        task_results[task.task_id] = {
            "transition_count": dataset.size,
            "methods": {
                "drpo": {
                    "runs": runs,
                    "evaluation_summary": _aggregate_task_evaluations(runs),
                }
            },
        }

    result = {
        "tasks": task_results,
        "seeds": list(resolved_seeds),
        "methods": ["drpo"],
        "steps": effective_steps,
        "batch_size": effective_batch_size,
        "device": resolved_device,
    }
    atomic_json(output / "results.json", result)
    return result


__all__ = [
    "evaluate_d4rl_agent",
    "run_d4rl",
]
