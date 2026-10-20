"""Trajectory-boundary GAE accumulation for prepared offline TD residuals."""
from __future__ import annotations

import numpy as np


def compute_gae_from_td(
    td: np.ndarray,
    terminals: np.ndarray,
    timeouts: np.ndarray,
    *,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> np.ndarray:
    """Accumulate TD residuals without crossing terminal, timeout, or dataset tail."""
    td64 = np.asarray(td, dtype=np.float64).reshape(-1)
    terminals = np.asarray(terminals, dtype=np.bool_).reshape(-1)
    timeouts = np.asarray(timeouts, dtype=np.bool_).reshape(-1)
    if not (td64.shape == terminals.shape == timeouts.shape) or not td64.size:
        raise ValueError("TD, terminal, and timeout vectors must be non-empty and aligned")
    if bool((terminals & timeouts).any()):
        raise ValueError("terminal and timeout flags overlap")
    if (
        not np.isfinite(td64).all()
        or not 0.0 <= gamma <= 1.0
        or not 0.0 <= gae_lambda <= 1.0
    ):
        raise ValueError("invalid GAE input")
    result = np.empty_like(td64)
    running = 0.0
    continuation = ~(terminals | timeouts)
    for index in range(td64.size - 1, -1, -1):
        running = td64[index] + gamma * gae_lambda * continuation[index] * running
        result[index] = running
    return result.astype(np.float32)
