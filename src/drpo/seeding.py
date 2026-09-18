from __future__ import annotations

import random

import numpy as np

try:
    import torch
except ImportError:  # Planning-only imports must not require Torch.
    torch = None  # type: ignore[assignment]


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch when available."""

    random.seed(seed)
    np.random.seed(seed % (2**32))
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
