"""Trajectory-contract and offline GAE utilities for E7.

This module treats D4RL rows as ordered behavior-policy trajectories.  It keeps
environment termination, time-limit truncation, and dataset-end truncation
separate:

* true terminal: no value bootstrap and no recursive carry;
* timeout: bootstrap from ``next_observation`` but stop the recursive carry;
* dataset end without a boundary flag: bootstrap from the recorded
  ``next_observation`` but stop because no later transition exists.

The implementation is intentionally independent of the actor update so that the
same prepared advantages can be shared by A2C and PPO branches.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


@dataclasses.dataclass(frozen=True)
class TrajectoryAudit:
    transition_count: int
    episode_count: int
    terminal_count: int
    timeout_count: int
    dataset_end_truncation: bool
    open_tail_length: int
    non_boundary_link_count: int
    continuity_failure_count: int
    continuity_max_abs_error: float
    terminal_timeout_overlap_count: int
    status: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, target)


def _as_bool_array(name: str, value: Any, length: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.bool_).reshape(-1)
    if array.shape != (length,):
        raise ValueError(f"{name} shape must be ({length},), got {array.shape}")
    return array


def reconstruct_episode_ids(
    terminals: Any,
    timeouts: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(episode_ids, episode_end_mask)`` for ordered transitions."""

    terminals_array = np.asarray(terminals, dtype=np.bool_).reshape(-1)
    n = int(terminals_array.shape[0])
    timeouts_array = _as_bool_array("timeouts", timeouts, n)
    end_mask = terminals_array | timeouts_array
    starts = np.zeros(n, dtype=np.bool_)
    if n:
        starts[0] = True
        starts[1:] = end_mask[:-1]
    episode_ids = np.cumsum(starts, dtype=np.int64) - 1
    return episode_ids, end_mask


def audit_ordered_trajectories(
    observations: Any,
    next_observations: Any,
    terminals: Any,
    timeouts: Any,
    *,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> TrajectoryAudit:
    """Fail closed if the arrays cannot support trajectory-wise GAE."""

    observations_array = np.asarray(observations)
    next_observations_array = np.asarray(next_observations)
    if observations_array.ndim < 2:
        raise ValueError("observations must have a transition dimension and features")
    if next_observations_array.shape != observations_array.shape:
        raise ValueError(
            "next_observations shape must match observations: "
            f"{next_observations_array.shape} != {observations_array.shape}"
        )
    if not np.isfinite(observations_array).all():
        raise ValueError("observations contain NaN/Inf")
    if not np.isfinite(next_observations_array).all():
        raise ValueError("next_observations contain NaN/Inf")

    n = int(observations_array.shape[0])
    if n == 0:
        raise ValueError("trajectory dataset must contain at least one transition")
    terminals_array = _as_bool_array("terminals", terminals, n)
    timeouts_array = _as_bool_array("timeouts", timeouts, n)
    overlap = terminals_array & timeouts_array
    overlap_count = int(overlap.sum())

    episode_ids, end_mask = reconstruct_episode_ids(terminals_array, timeouts_array)
    dataset_end_truncation = bool(n and not end_mask[-1])
    if n == 0:
        open_tail_length = 0
    elif dataset_end_truncation:
        previous_boundaries = np.flatnonzero(end_mask[:-1])
        tail_start = int(previous_boundaries[-1] + 1) if previous_boundaries.size else 0
        open_tail_length = n - tail_start
    else:
        open_tail_length = 0

    link_mask = ~end_mask[:-1] if n > 1 else np.zeros(0, dtype=np.bool_)
    linked_next = next_observations_array[:-1][link_mask]
    linked_observations = observations_array[1:][link_mask]
    if linked_next.size:
        abs_error = np.abs(linked_next - linked_observations)
        per_link_max = abs_error.reshape(abs_error.shape[0], -1).max(axis=1)
        allowed = atol + rtol * np.abs(linked_observations).reshape(
            linked_observations.shape[0], -1
        ).max(axis=1)
        failure_count = int((per_link_max > allowed).sum())
        max_abs_error = float(per_link_max.max())
    else:
        failure_count = 0
        max_abs_error = 0.0

    episode_count = int(episode_ids[-1] + 1) if n else 0
    status = "PASS" if overlap_count == 0 and failure_count == 0 else "FAIL"
    audit = TrajectoryAudit(
        transition_count=n,
        episode_count=episode_count,
        terminal_count=int(terminals_array.sum()),
        timeout_count=int(timeouts_array.sum()),
        dataset_end_truncation=dataset_end_truncation,
        open_tail_length=open_tail_length,
        non_boundary_link_count=int(link_mask.sum()),
        continuity_failure_count=failure_count,
        continuity_max_abs_error=max_abs_error,
        terminal_timeout_overlap_count=overlap_count,
        status=status,
    )
    if audit.status != "PASS":
        raise ValueError(f"trajectory audit failed: {audit.to_dict()}")
    return audit


def _validate_advantage_inputs(
    rewards: Any,
    values: Any,
    next_values: Any,
    terminals: Any,
    timeouts: Any,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be finite and in [0, 1]")
    if not math.isfinite(gae_lambda) or not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gae_lambda must be finite and in [0, 1]")
    rewards_array = np.asarray(rewards, dtype=np.float64).reshape(-1)
    n = int(rewards_array.shape[0])
    values_array = np.asarray(values, dtype=np.float64).reshape(-1)
    next_values_array = np.asarray(next_values, dtype=np.float64).reshape(-1)
    if values_array.shape != (n,) or next_values_array.shape != (n,):
        raise ValueError("values and next_values must match rewards")
    terminals_array = _as_bool_array("terminals", terminals, n)
    timeouts_array = _as_bool_array("timeouts", timeouts, n)
    if bool((terminals_array & timeouts_array).any()):
        raise ValueError("terminal and timeout flags must not overlap")
    for name, array in (
        ("rewards", rewards_array),
        ("values", values_array),
        ("next_values", next_values_array),
    ):
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains NaN/Inf")
    return (
        rewards_array,
        values_array,
        next_values_array,
        terminals_array,
        timeouts_array,
    )


def compute_gae_numpy(
    rewards: Any,
    values: Any,
    next_values: Any,
    terminals: Any,
    timeouts: Any,
    *,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> np.ndarray:
    """Compute behavior-trajectory GAE with explicit boundary semantics."""

    (
        rewards_array,
        values_array,
        next_values_array,
        terminals_array,
        timeouts_array,
    ) = _validate_advantage_inputs(
        rewards,
        values,
        next_values,
        terminals,
        timeouts,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )
    bootstrap_mask = (~terminals_array).astype(np.float64)
    continuation_mask = (~(terminals_array | timeouts_array)).astype(np.float64)
    deltas = (
        rewards_array
        + gamma * bootstrap_mask * next_values_array
        - values_array
    )
    advantages = np.empty_like(deltas)
    running = 0.0
    for index in range(deltas.shape[0] - 1, -1, -1):
        running = deltas[index] + (
            gamma * gae_lambda * continuation_mask[index] * running
        )
        advantages[index] = running
    if not np.isfinite(advantages).all():
        raise FloatingPointError("computed GAE contains NaN/Inf")
    return advantages.astype(np.float32)


def compute_gae_torch(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    terminals: torch.Tensor,
    timeouts: torch.Tensor,
    *,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> torch.Tensor:
    """Torch reference implementation used for cross-implementation checks."""

    rewards_flat = rewards.reshape(-1)
    values_flat = values.reshape(-1)
    next_values_flat = next_values.reshape(-1)
    terminals_flat = terminals.to(dtype=torch.bool).reshape(-1)
    timeouts_flat = timeouts.to(dtype=torch.bool).reshape(-1)
    n = rewards_flat.numel()
    if values_flat.numel() != n or next_values_flat.numel() != n:
        raise ValueError("values and next_values must match rewards")
    if terminals_flat.numel() != n or timeouts_flat.numel() != n:
        raise ValueError("boundary arrays must match rewards")
    if bool((terminals_flat & timeouts_flat).any()):
        raise ValueError("terminal and timeout flags must not overlap")
    if not 0.0 <= gamma <= 1.0 or not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gamma and gae_lambda must be in [0, 1]")
    bootstrap = (~terminals_flat).to(dtype=rewards_flat.dtype)
    continuation = (~(terminals_flat | timeouts_flat)).to(
        dtype=rewards_flat.dtype
    )
    deltas = rewards_flat + gamma * bootstrap * next_values_flat - values_flat
    advantages = torch.empty_like(deltas)
    running = torch.zeros((), dtype=deltas.dtype, device=deltas.device)
    for index in range(n - 1, -1, -1):
        running = deltas[index] + (
            gamma * gae_lambda * continuation[index] * running
        )
        advantages[index] = running
    if not bool(torch.isfinite(advantages).all()):
        raise FloatingPointError("computed torch GAE contains NaN/Inf")
    return advantages


def compute_td_advantage(
    rewards: Any,
    values: Any,
    next_values: Any,
    terminals: Any,
    *,
    gamma: float = 0.99,
) -> np.ndarray:
    rewards_array = np.asarray(rewards, dtype=np.float64).reshape(-1)
    values_array = np.asarray(values, dtype=np.float64).reshape(-1)
    next_values_array = np.asarray(next_values, dtype=np.float64).reshape(-1)
    terminals_array = np.asarray(terminals, dtype=np.bool_).reshape(-1)
    if not (
        rewards_array.shape
        == values_array.shape
        == next_values_array.shape
        == terminals_array.shape
    ):
        raise ValueError("TD arrays must have the same shape")
    advantages = (
        rewards_array
        + gamma * (~terminals_array).astype(np.float64) * next_values_array
        - values_array
    )
    if not np.isfinite(advantages).all():
        raise FloatingPointError("computed TD advantage contains NaN/Inf")
    return advantages.astype(np.float32)


def advantage_diagnostics(
    td_advantages: Any,
    gae_advantages: Any,
) -> dict[str, Any]:
    td = np.asarray(td_advantages, dtype=np.float64).reshape(-1)
    gae = np.asarray(gae_advantages, dtype=np.float64).reshape(-1)
    if td.shape != gae.shape or td.size == 0:
        raise ValueError("advantage arrays must be non-empty and shape-matched")
    if not np.isfinite(td).all() or not np.isfinite(gae).all():
        raise ValueError("advantage arrays contain NaN/Inf")
    td_std = float(td.std())
    gae_std = float(gae.std())
    pearson = (
        float(np.corrcoef(td, gae)[0, 1])
        if td_std > 0.0 and gae_std > 0.0
        else None
    )
    td_order = np.argsort(np.argsort(td, kind="stable"), kind="stable")
    gae_order = np.argsort(np.argsort(gae, kind="stable"), kind="stable")
    spearman = (
        float(np.corrcoef(td_order, gae_order)[0, 1])
        if td.size > 1
        else None
    )
    sign_flip = np.signbit(td) != np.signbit(gae)
    quantiles = (0.01, 0.1, 0.5, 0.9, 0.99)
    return {
        "transition_count": int(td.size),
        "pearson": pearson,
        "spearman": spearman,
        "sign_flip_fraction": float(sign_flip.mean()),
        "td_negative_fraction": float((td < 0).mean()),
        "gae_negative_fraction": float((gae < 0).mean()),
        "td_mean": float(td.mean()),
        "td_std": td_std,
        "gae_mean": float(gae.mean()),
        "gae_std": gae_std,
        "td_quantiles": {str(q): float(np.quantile(td, q)) for q in quantiles},
        "gae_quantiles": {str(q): float(np.quantile(gae, q)) for q in quantiles},
    }
