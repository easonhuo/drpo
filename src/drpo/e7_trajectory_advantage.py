"""Ordered-trajectory audits and frozen-critic advantage construction for E7.

The functions in this module are deliberately independent of the canonical D4RL
trainer.  They operate on already-loaded arrays and enforce the trajectory
contract needed by offline GAE:

* rows are in behavior-trajectory order;
* ``terminal`` ends an episode and suppresses value bootstrap;
* ``timeout`` ends the recursive trace but keeps value bootstrap;
* a non-terminal/non-timeout dataset tail bootstraps from ``next_value`` while
  stopping the recursive trace because no following transition is available.

No function in this module mutates a policy, critic, dataset, or optimizer.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
from typing import Any, Mapping

import numpy as np


@dataclasses.dataclass(frozen=True)
class TrajectoryAuditConfig:
    """Numerical tolerances for ordered-transition continuity checks."""

    absolute_tolerance: float = 1e-5
    relative_tolerance: float = 1e-5

    def validate(self) -> None:
        for name, value in dataclasses.asdict(self).items():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclasses.dataclass(frozen=True)
class AdvantageConfig:
    """Frozen settings for one-step TD and behavior-trajectory GAE."""

    gamma: float = 0.99
    gae_lambda: float = 0.95

    def validate(self) -> None:
        if not math.isfinite(self.gamma) or not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be finite and in [0, 1]")
        if not math.isfinite(self.gae_lambda) or not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be finite and in [0, 1]")


def _as_float64(name: str, value: Any, *, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, got {array.ndim}")
    if not np.isfinite(array).all():
        raise FloatingPointError(f"{name} contains NaN/Inf")
    return np.ascontiguousarray(array)


def _as_bool1d(name: str, value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.bool_)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return np.ascontiguousarray(array)


def _require_same_length(reference_name: str, reference: np.ndarray, **arrays: np.ndarray) -> None:
    expected = len(reference)
    if expected <= 0:
        raise ValueError(f"{reference_name} must be non-empty")
    for name, array in arrays.items():
        if len(array) != expected:
            raise ValueError(
                f"{name} length {len(array)} does not match {reference_name} length {expected}"
            )


def array_sha256(array: Any) -> str:
    """Hash dtype, shape, and contiguous bytes of one NumPy-compatible array."""

    value = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(repr(tuple(value.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def ordered_trajectory_identity(
    *,
    observations: Any,
    actions: Any,
    rewards: Any,
    next_observations: Any,
    terminals: Any,
    timeouts: Any,
) -> str:
    """Return a deterministic identity for the exact ordered transition table."""

    fields = {
        "observations": np.asarray(observations),
        "actions": np.asarray(actions),
        "rewards": np.asarray(rewards),
        "next_observations": np.asarray(next_observations),
        "terminals": np.asarray(terminals, dtype=np.bool_),
        "timeouts": np.asarray(timeouts, dtype=np.bool_),
    }
    digest = hashlib.sha256()
    for name, value in fields.items():
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(array_sha256(value)))
        digest.update(b"\0")
    return digest.hexdigest()


def audit_ordered_trajectory(
    *,
    observations: Any,
    next_observations: Any,
    terminals: Any,
    timeouts: Any,
    config: TrajectoryAuditConfig | None = None,
) -> dict[str, Any]:
    """Fail closed unless rows satisfy the behavior-trajectory ordering contract.

    For every row except the final row, ``next_observations[t]`` must equal
    ``observations[t+1]`` whenever row ``t`` is neither terminal nor timeout.
    Boundaries intentionally do not require continuity because the next row is a
    reset state from another episode.
    """

    cfg = config or TrajectoryAuditConfig()
    cfg.validate()
    obs = _as_float64("observations", observations, ndim=2)
    next_obs = _as_float64("next_observations", next_observations, ndim=2)
    terms = _as_bool1d("terminals", terminals)
    touts = _as_bool1d("timeouts", timeouts)
    _require_same_length(
        "observations",
        obs,
        next_observations=next_obs,
        terminals=terms,
        timeouts=touts,
    )
    if obs.shape != next_obs.shape:
        raise ValueError(
            f"observation shape {obs.shape} does not match next_observation shape {next_obs.shape}"
        )
    overlap = terms & touts
    if bool(overlap.any()):
        indices = np.flatnonzero(overlap)[:10].tolist()
        raise ValueError(f"terminal and timeout overlap at rows {indices}")

    n = len(obs)
    boundary = terms | touts
    candidate = np.flatnonzero(~boundary[:-1]) if n > 1 else np.empty(0, dtype=np.int64)
    if candidate.size:
        lhs = next_obs[candidate]
        rhs = obs[candidate + 1]
        close = np.isclose(
            lhs,
            rhs,
            rtol=cfg.relative_tolerance,
            atol=cfg.absolute_tolerance,
            equal_nan=False,
        ).all(axis=1)
        bad = candidate[~close]
        absolute = np.abs(lhs - rhs)
        max_abs = float(absolute.max())
        scale = np.maximum(np.maximum(np.abs(lhs), np.abs(rhs)), 1.0)
        max_scaled = float((absolute / scale).max())
    else:
        bad = np.empty(0, dtype=np.int64)
        max_abs = 0.0
        max_scaled = 0.0
    if bad.size:
        preview = bad[:10].tolist()
        raise ValueError(
            "ordered trajectory continuity failed at non-boundary rows "
            f"{preview}; total_failures={int(bad.size)}"
        )

    episode_end = boundary.copy()
    episode_end[-1] = True
    return {
        "schema_version": 1,
        "status": "PASS",
        "transition_count": n,
        "observation_dim": int(obs.shape[1]),
        "terminal_count": int(terms.sum()),
        "timeout_count": int(touts.sum()),
        "explicit_boundary_count": int(boundary.sum()),
        "episode_segment_count": int(episode_end.sum()),
        "continuity_rows_checked": int(candidate.size),
        "continuity_failure_count": 0,
        "continuity_max_abs_error": max_abs,
        "continuity_max_scaled_error": max_scaled,
        "absolute_tolerance": cfg.absolute_tolerance,
        "relative_tolerance": cfg.relative_tolerance,
        "tail_is_terminal": bool(terms[-1]),
        "tail_is_timeout": bool(touts[-1]),
        "tail_bootstrap_and_stop_recursion": bool(not terms[-1]),
        "terminal_bootstrap": False,
        "timeout_bootstrap": True,
        "terminal_stops_recursion": True,
        "timeout_stops_recursion": True,
    }


def compute_one_step_td(
    *,
    rewards: Any,
    values: Any,
    next_values: Any,
    terminals: Any,
    gamma: float,
) -> np.ndarray:
    """Compute ``r + gamma * V(s') * (1-terminal) - V(s)``.

    Timeouts intentionally do not appear in this equation: they bootstrap just
    like ordinary transitions.  They are handled only by the GAE recursion stop.
    """

    cfg = AdvantageConfig(gamma=float(gamma), gae_lambda=0.0)
    cfg.validate()
    rews = _as_float64("rewards", rewards, ndim=1)
    vals = _as_float64("values", values, ndim=1)
    next_vals = _as_float64("next_values", next_values, ndim=1)
    terms = _as_bool1d("terminals", terminals)
    _require_same_length(
        "rewards", rews, values=vals, next_values=next_vals, terminals=terms
    )
    bootstrap = (~terms).astype(np.float64)
    delta = rews + cfg.gamma * next_vals * bootstrap - vals
    if not np.isfinite(delta).all():
        raise FloatingPointError("one-step TD advantage contains NaN/Inf")
    return np.ascontiguousarray(delta.astype(np.float32))


def compute_behavior_trajectory_gae(
    *,
    rewards: Any,
    values: Any,
    next_values: Any,
    terminals: Any,
    timeouts: Any,
    gamma: float,
    gae_lambda: float,
) -> np.ndarray:
    """Compute offline GAE on the stored behavior-trajectory order.

    Recurrence:

    ``A_t = delta_t + gamma * lambda * continue_t * A_{t+1}``

    where ``continue_t`` is false at terminals, timeouts, and the final stored
    row.  The final-row delta still bootstraps when it is not terminal, so an
    incomplete dataset tail is treated as truncation: bootstrap, then stop.
    """

    cfg = AdvantageConfig(gamma=float(gamma), gae_lambda=float(gae_lambda))
    cfg.validate()
    rews = _as_float64("rewards", rewards, ndim=1)
    vals = _as_float64("values", values, ndim=1)
    next_vals = _as_float64("next_values", next_values, ndim=1)
    terms = _as_bool1d("terminals", terminals)
    touts = _as_bool1d("timeouts", timeouts)
    _require_same_length(
        "rewards",
        rews,
        values=vals,
        next_values=next_vals,
        terminals=terms,
        timeouts=touts,
    )
    if bool((terms & touts).any()):
        raise ValueError("terminal and timeout cannot both be true")

    delta = compute_one_step_td(
        rewards=rews,
        values=vals,
        next_values=next_vals,
        terminals=terms,
        gamma=cfg.gamma,
    ).astype(np.float64)
    continuation = ~(terms | touts)
    continuation[-1] = False
    advantages = np.empty_like(delta, dtype=np.float64)
    running = 0.0
    multiplier = cfg.gamma * cfg.gae_lambda
    for index in range(len(delta) - 1, -1, -1):
        running = float(delta[index]) + multiplier * float(continuation[index]) * running
        advantages[index] = running
    if not np.isfinite(advantages).all():
        raise FloatingPointError("GAE advantage contains NaN/Inf")
    return np.ascontiguousarray(advantages.astype(np.float32))


def build_frozen_advantage_arrays(
    *,
    rewards: Any,
    values: Any,
    next_values: Any,
    terminals: Any,
    timeouts: Any,
    config: AdvantageConfig | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build matched one-step and GAE arrays plus regression diagnostics."""

    cfg = config or AdvantageConfig()
    cfg.validate()
    one_step = compute_one_step_td(
        rewards=rewards,
        values=values,
        next_values=next_values,
        terminals=terminals,
        gamma=cfg.gamma,
    )
    gae = compute_behavior_trajectory_gae(
        rewards=rewards,
        values=values,
        next_values=next_values,
        terminals=terminals,
        timeouts=timeouts,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
    )
    lambda_zero = compute_behavior_trajectory_gae(
        rewards=rewards,
        values=values,
        next_values=next_values,
        terminals=terminals,
        timeouts=timeouts,
        gamma=cfg.gamma,
        gae_lambda=0.0,
    )
    lambda_zero_error = float(np.max(np.abs(lambda_zero.astype(np.float64) - one_step)))
    if lambda_zero_error != 0.0:
        raise AssertionError(
            f"lambda=0 GAE must be bit-identical to one-step TD; max_error={lambda_zero_error}"
        )
    arrays = {
        "one_step_td": one_step,
        "gae_lambda_0p95": gae,
        "values": np.ascontiguousarray(np.asarray(values, dtype=np.float32)),
        "next_values": np.ascontiguousarray(np.asarray(next_values, dtype=np.float32)),
    }
    diagnostics: dict[str, Any] = {
        "schema_version": 1,
        "gamma": cfg.gamma,
        "gae_lambda": cfg.gae_lambda,
        "transition_count": int(len(one_step)),
        "lambda_zero_regression_max_abs_error": lambda_zero_error,
        "one_step_td_sha256": array_sha256(one_step),
        "gae_lambda_0p95_sha256": array_sha256(gae),
        "value_sha256": array_sha256(arrays["values"]),
        "next_value_sha256": array_sha256(arrays["next_values"]),
        "one_step_mean": float(one_step.mean(dtype=np.float64)),
        "one_step_std": float(one_step.std(dtype=np.float64)),
        "gae_mean": float(gae.mean(dtype=np.float64)),
        "gae_std": float(gae.std(dtype=np.float64)),
    }
    return arrays, diagnostics


def verify_array_manifest(arrays: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    """Verify array lengths and SHA-256 fields from a prepared manifest."""

    expected = int(manifest["transition_count"])
    fields = {
        "one_step_td": "one_step_td_sha256",
        "gae_lambda_0p95": "gae_lambda_0p95_sha256",
        "values": "value_sha256",
        "next_values": "next_value_sha256",
    }
    for key, digest_key in fields.items():
        if key not in arrays:
            raise ValueError(f"prepared advantage archive is missing {key}")
        value = np.asarray(arrays[key])
        if value.ndim != 1 or len(value) != expected:
            raise ValueError(
                f"prepared array {key} shape {value.shape} does not match ({expected},)"
            )
        actual = array_sha256(value)
        if actual != str(manifest[digest_key]):
            raise ValueError(
                f"prepared array {key} SHA-256 mismatch: expected {manifest[digest_key]}, got {actual}"
            )
        if not np.isfinite(value).all():
            raise FloatingPointError(f"prepared array {key} contains NaN/Inf")
