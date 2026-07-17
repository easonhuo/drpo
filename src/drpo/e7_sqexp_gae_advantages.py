"""Ordered-trajectory TD and GAE boundary implementation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from drpo.e7_sqexp_gae_protocol import GAMMA, GAE_LAMBDA

def trajectory_end_mask(
    episode_ids: np.ndarray,
    terminals: np.ndarray,
    timeouts: np.ndarray,
) -> np.ndarray:
    episodes = np.asarray(episode_ids, dtype=np.int64).reshape(-1)
    terminal = np.asarray(terminals, dtype=np.bool_).reshape(-1)
    timeout = np.asarray(timeouts, dtype=np.bool_).reshape(-1)
    if not (episodes.shape == terminal.shape == timeout.shape):
        raise ValueError("episode_ids, terminals, and timeouts must have the same shape")
    if len(episodes) == 0:
        raise ValueError("ordered trajectory cannot be empty")
    if np.any(terminal & timeout):
        raise ValueError("terminal and timeout flags must not overlap")
    if np.any(np.diff(episodes) < 0):
        raise ValueError("episode_ids must be nondecreasing in stored trajectory order")
    if np.any(np.diff(episodes) > 1):
        raise ValueError("episode_ids must be contiguous without skipped episode identities")
    changed = np.zeros(len(episodes), dtype=np.bool_)
    changed[:-1] = episodes[:-1] != episodes[1:]
    changed[-1] = True
    flagged = terminal | timeout
    if np.any(flagged & ~changed):
        bad = int(np.flatnonzero(flagged & ~changed)[0])
        raise ValueError(f"terminal/timeout at row {bad} is not the final stored row of its episode")
    return changed


@dataclass(frozen=True)
class AdvantageArrays:
    td_float64: np.ndarray
    gae_float64: np.ndarray
    td_float32: np.ndarray
    gae_float32: np.ndarray
    delta_float64: np.ndarray
    trajectory_end: np.ndarray
    bootstrap_mask: np.ndarray
    carry_mask: np.ndarray
    storage_audit: dict[str, float]


def compute_td_and_gae(
    *,
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    terminals: np.ndarray,
    timeouts: np.ndarray,
    episode_ids: np.ndarray,
    gamma: float = GAMMA,
    gae_lambda: float = GAE_LAMBDA,
) -> AdvantageArrays:
    if not (0.0 <= gamma <= 1.0):
        raise ValueError("gamma must be in [0,1]")
    if not (0.0 <= gae_lambda <= 1.0):
        raise ValueError("gae_lambda must be in [0,1]")
    reward = np.asarray(rewards, dtype=np.float64).reshape(-1)
    value = np.asarray(values, dtype=np.float64).reshape(-1)
    next_value = np.asarray(next_values, dtype=np.float64).reshape(-1)
    terminal = np.asarray(terminals, dtype=np.bool_).reshape(-1)
    timeout = np.asarray(timeouts, dtype=np.bool_).reshape(-1)
    episodes = np.asarray(episode_ids, dtype=np.int64).reshape(-1)
    shapes = {x.shape for x in (reward, value, next_value, terminal, timeout, episodes)}
    if len(shapes) != 1:
        raise ValueError("all advantage inputs must have the same one-dimensional shape")
    if not all(np.all(np.isfinite(x)) for x in (reward, value, next_value)):
        raise FloatingPointError("non-finite reward or critic value")
    ends = trajectory_end_mask(episodes, terminal, timeout)
    bootstrap = ~terminal
    carry = ~(terminal | timeout | ends)
    delta = reward + float(gamma) * bootstrap.astype(np.float64) * next_value - value
    gae = np.empty_like(delta, dtype=np.float64)
    running = 0.0
    for index in range(len(delta) - 1, -1, -1):
        if carry[index]:
            running = float(delta[index]) + float(gamma) * float(gae_lambda) * running
        else:
            running = float(delta[index])
        gae[index] = running
    td32 = delta.astype(np.float32)
    gae32 = gae.astype(np.float32)
    audit = {
        "td_storage_max_abs_error": float(np.max(np.abs(td32.astype(np.float64) - delta))),
        "gae_storage_max_abs_error": float(np.max(np.abs(gae32.astype(np.float64) - gae))),
        "terminal_count": float(np.sum(terminal)),
        "timeout_count": float(np.sum(timeout)),
        "nonterminal_tail_count": float(np.sum(ends & ~terminal & ~timeout)),
        "carry_count": float(np.sum(carry)),
    }
    return AdvantageArrays(
        td_float64=delta.copy(),
        gae_float64=gae,
        td_float32=td32,
        gae_float32=gae32,
        delta_float64=delta,
        trajectory_end=ends,
        bootstrap_mask=bootstrap,
        carry_mask=carry,
        storage_audit=audit,
    )


def reference_gae_by_episode(
    delta: np.ndarray,
    episode_ids: np.ndarray,
    terminals: np.ndarray,
    timeouts: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> np.ndarray:
    """Independent episode-slice implementation used by the numerical audit."""
    values = np.asarray(delta, dtype=np.float64).reshape(-1)
    episodes = np.asarray(episode_ids, dtype=np.int64).reshape(-1)
    terminal = np.asarray(terminals, dtype=np.bool_).reshape(-1)
    timeout = np.asarray(timeouts, dtype=np.bool_).reshape(-1)
    ends = trajectory_end_mask(episodes, terminal, timeout)
    out = np.zeros_like(values)
    start = 0
    for stop in np.flatnonzero(ends):
        running = 0.0
        for index in range(int(stop), start - 1, -1):
            if index == int(stop) or terminal[index] or timeout[index]:
                running = float(values[index])
            else:
                running = float(values[index]) + gamma * gae_lambda * running
            out[index] = running
        start = int(stop) + 1
    return out


def validate_advantage_numerics(
    arrays: AdvantageArrays,
    *,
    episode_ids: np.ndarray,
    terminals: np.ndarray,
    timeouts: np.ndarray,
    gamma: float,
    gae_lambda: float,
    disagreement_tolerance: float = 1e-10,
) -> dict[str, Any]:
    independent = reference_gae_by_episode(
        arrays.delta_float64,
        episode_ids,
        terminals,
        timeouts,
        gamma,
        gae_lambda,
    )
    implementation_error = float(np.max(np.abs(independent - arrays.gae_float64)))
    lambda_zero = compute_td_and_gae(
        rewards=arrays.delta_float64,
        values=np.zeros_like(arrays.delta_float64),
        next_values=np.zeros_like(arrays.delta_float64),
        terminals=terminals,
        timeouts=timeouts,
        episode_ids=episode_ids,
        gamma=gamma,
        gae_lambda=0.0,
    )
    lambda_zero_error = float(np.max(np.abs(lambda_zero.gae_float64 - arrays.delta_float64)))
    if implementation_error > disagreement_tolerance:
        raise ArithmeticError(
            "independent GAE implementation disagreement exceeds tolerance: "
            f"{implementation_error} > {disagreement_tolerance}"
        )
    if lambda_zero_error != 0.0:
        raise ArithmeticError(f"lambda=0 did not equal one-step TD exactly: {lambda_zero_error}")
    return {
        "independent_implementation_max_abs_disagreement": implementation_error,
        "independent_implementation_tolerance": disagreement_tolerance,
        "lambda_zero_td_max_abs_disagreement": lambda_zero_error,
        "storage_quantization": dict(arrays.storage_audit),
        "storage_quantization_is_not_implementation_disagreement": True,
    }


