"""Canonical Hopper E7-Q2 offline-data and episode primitives."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


@dataclass
class OfflineData:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    terminals: np.ndarray
    timeouts: np.ndarray
    episode_ids: np.ndarray

    @property
    def size(self) -> int:
        return int(self.observations.shape[0])


def load_hopper_hdf5(
    path: str | Path,
    max_transitions: int | None = None,
) -> OfflineData:
    """Load the frozen legacy D4RL HDF5 contract."""

    path = Path(path)
    with h5py.File(path, "r") as handle:
        required = (
            "observations",
            "actions",
            "rewards",
            "terminals",
        )
        missing = [
            key for key in required if key not in handle
        ]
        if missing:
            raise ValueError(
                f"Missing legacy D4RL arrays: {missing}"
            )
        total = int(handle["observations"].shape[0])
        limit = (
            total
            if max_transitions is None
            else min(total, int(max_transitions))
        )
        observations = np.asarray(
            handle["observations"][:limit],
            dtype=np.float32,
        )
        actions = np.asarray(
            handle["actions"][:limit],
            dtype=np.float32,
        )
        rewards = np.asarray(
            handle["rewards"][:limit],
            dtype=np.float32,
        ).reshape(-1)
        terminals = np.asarray(
            handle["terminals"][:limit],
            dtype=np.bool_,
        ).reshape(-1)
        if "timeouts" in handle:
            timeouts = np.asarray(
                handle["timeouts"][:limit],
                dtype=np.bool_,
            ).reshape(-1)
        else:
            timeouts = np.zeros(limit, dtype=np.bool_)
        if "next_observations" in handle:
            next_observations = np.asarray(
                handle["next_observations"][:limit],
                dtype=np.float32,
            )
        else:
            next_observations = np.concatenate(
                [observations[1:], observations[-1:]],
                axis=0,
            )
    if observations.ndim != 2 or actions.ndim != 2:
        raise ValueError(
            "observations and actions must be rank-2 arrays"
        )
    if len(observations) < 2:
        raise ValueError(
            "dataset must contain at least two transitions"
        )
    return OfflineData(
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        terminals=terminals,
        timeouts=timeouts,
        episode_ids=build_episode_ids(
            terminals,
            timeouts,
        ),
    )


def build_episode_ids(
    terminals: np.ndarray,
    timeouts: np.ndarray,
) -> np.ndarray:
    terminals = np.asarray(
        terminals,
        dtype=np.bool_,
    ).reshape(-1)
    timeouts = np.asarray(
        timeouts,
        dtype=np.bool_,
    ).reshape(-1)
    if terminals.shape != timeouts.shape:
        raise ValueError(
            "terminals and timeouts must have the same shape"
        )
    output = np.empty(len(terminals), dtype=np.int64)
    episode = 0
    for index in range(len(terminals)):
        output[index] = episode
        if terminals[index] or timeouts[index]:
            episode += 1
    return output


def discounted_returns(
    rewards: np.ndarray,
    terminals: np.ndarray,
    timeouts: np.ndarray,
    gamma: float,
) -> np.ndarray:
    rewards = np.asarray(
        rewards,
        dtype=np.float32,
    ).reshape(-1)
    terminals = np.asarray(
        terminals,
        dtype=np.bool_,
    ).reshape(-1)
    timeouts = np.asarray(
        timeouts,
        dtype=np.bool_,
    ).reshape(-1)
    returns = np.empty_like(rewards, dtype=np.float32)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        if (
            index == len(rewards) - 1
            or terminals[index]
            or timeouts[index]
        ):
            running = float(rewards[index])
        else:
            running = (
                float(rewards[index])
                + float(gamma) * running
            )
        returns[index] = running
    return returns


def split_episode_indices(
    episode_ids: np.ndarray,
    seed: int,
    train_fraction: float,
    validation_fraction: float,
) -> dict[str, np.ndarray]:
    episodes = np.unique(episode_ids)
    if len(episodes) < 3:
        raise ValueError(
            "At least three episodes are required for "
            "train/validation/test split"
        )
    generator = np.random.default_rng(seed)
    shuffled = episodes.copy()
    generator.shuffle(shuffled)
    n_train = max(
        1,
        int(round(len(shuffled) * train_fraction)),
    )
    n_validation = max(
        1,
        int(
            round(
                len(shuffled) * validation_fraction
            )
        ),
    )
    if n_train + n_validation >= len(shuffled):
        n_train = max(1, len(shuffled) - 2)
        n_validation = 1
    groups = {
        "train": shuffled[:n_train],
        "validation": shuffled[
            n_train : n_train + n_validation
        ],
        "test": shuffled[n_train + n_validation :],
    }
    return {
        name: np.flatnonzero(
            np.isin(episode_ids, group)
        ).astype(np.int64)
        for name, group in groups.items()
    }


@dataclass(frozen=True)
class Normalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, array: np.ndarray) -> "Normalizer":
        mean = np.mean(
            array,
            axis=0,
            dtype=np.float64,
        ).astype(np.float32)
        std = np.std(
            array,
            axis=0,
            dtype=np.float64,
        ).astype(np.float32)
        return cls(
            mean=mean,
            std=np.maximum(std, 1.0e-6),
        )

    def transform(
        self,
        array: np.ndarray,
    ) -> np.ndarray:
        return (
            (array - self.mean) / self.std
        ).astype(np.float32)
