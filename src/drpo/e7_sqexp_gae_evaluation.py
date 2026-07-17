"""Canonical external-validity rollout evaluation."""
from __future__ import annotations

from typing import Any
import numpy as np
import torch
from drpo.e7_sqexp_gae_contract import DatasetSpec
from drpo.e7_sqexp_gae_models import CanonicalActor, normalize_observations


def rollout(
    actor: CanonicalActor,
    dataset: DatasetSpec,
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
    episodes: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    from drpo import e7_hopper_q2 as q2

    env = None
    try:
        env, metadata = q2._open_gymnasium_mujoco_env(dataset.env_id)  # noqa: SLF001
        returns: list[float] = []
        for episode in range(episodes):
            observation, _ = q2._reset_env(env, seed + episode)  # noqa: SLF001
            done = False
            total = 0.0
            steps = 0
            limit = q2._max_episode_steps(env, 10000)  # noqa: SLF001
            while not done and steps < limit:
                normalized = normalize_observations(observation.reshape(1, -1), obs_mean, obs_std)
                with torch.no_grad():
                    mean, _ = actor(torch.as_tensor(normalized, device=device))
                action = q2._clip_action_to_space(env, mean[0].cpu().numpy())  # noqa: SLF001
                observation, reward, done, _ = q2._step_env(env, action)  # noqa: SLF001
                total += reward
                steps += 1
            returns.append(total)
        raw = float(np.mean(returns))
        normalized_return = float("nan")
        normalized_available = False
        if dataset.score_protocol == "d4rl_v2_percent":
            if dataset.reference_min_score is None or dataset.reference_max_score is None:
                raise ValueError("D4RL score references are required")
            normalized_return = q2.normalize_d4rl_reference_score(
                raw,
                dataset.reference_min_score,
                dataset.reference_max_score,
                percent=True,
            )
            normalized_available = True
        return {
            "rollout_status": "available",
            "rollout_return_mean": raw,
            "rollout_return_std": float(np.std(returns)),
            "normalized_return": normalized_return,
            "normalized_return_available": normalized_available,
            "rollout_episodes": episodes,
            "rollout_backend": metadata,
        }
    finally:
        if env is not None:
            env.close()


