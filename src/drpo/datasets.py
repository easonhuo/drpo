from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


@dataclass(frozen=True)
class TransitionBatch:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray | None
    terminals: np.ndarray
    timeouts: np.ndarray | None = None

    @property
    def size(self) -> int:
        return int(self.observations.shape[0])


def load_d4rl_hdf5(path: str | Path, max_transitions: int | None = None) -> TransitionBatch:
    """Load legacy D4RL HDF5 transition arrays."""
    with h5py.File(path, "r") as dataset:
        limit = max_transitions or dataset["observations"].shape[0]
        next_observations = dataset["next_observations"][:limit] if "next_observations" in dataset else None
        timeouts = dataset["timeouts"][:limit] if "timeouts" in dataset else None
        return TransitionBatch(
            observations=dataset["observations"][:limit],
            actions=dataset["actions"][:limit],
            rewards=dataset["rewards"][:limit],
            next_observations=next_observations,
            terminals=dataset["terminals"][:limit],
            timeouts=timeouts,
        )
