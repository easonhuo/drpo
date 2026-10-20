"""Deterministic seeding helpers shared by paper experiments."""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_all(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch exactly once for one experiment run."""

    if not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cpu_generator(seed: int) -> torch.Generator:
    """Return a CPU generator with an explicit seed."""

    if not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    return torch.Generator(device="cpu").manual_seed(seed)
