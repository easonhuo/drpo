"""Hopper E7-Q2 correlation and terminal-classification helpers."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np

EPS = 1.0e-6


class ActorAuditProtocol(Protocol):
    audit_windows: int
    actor_state_drift_tolerance: float
    actor_update_tolerance: float
    support_boundary_fraction: float
    task_return_drop_threshold: float


def r2_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    denominator = float(
        np.sum(
            (
                y_true - np.mean(y_true)
            )
            ** 2
        )
    )
    if denominator <= EPS:
        return float("nan")
    return (
        1.0
        - float(
            np.sum(
                (y_true - y_pred) ** 2
            )
        )
        / denominator
    )


def pearson(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    if (
        len(y_true) < 2
        or np.std(y_true) <= EPS
        or np.std(y_pred) <= EPS
    ):
        return float("nan")
    return float(
        np.corrcoef(y_true, y_pred)[0, 1]
    )


def relative_slope(
    rows: Sequence[dict[str, Any]],
    key: str,
    windows: int,
) -> float:
    if len(rows) < windows:
        return float("inf")
    tail = rows[-windows:]
    steps = np.asarray(
        [float(row["step"]) for row in tail],
        dtype=np.float64,
    )
    values = np.asarray(
        [float(row[key]) for row in tail],
        dtype=np.float64,
    )
    if (
        not np.all(np.isfinite(values))
        or steps[-1] == steps[0]
    ):
        return float("inf")
    slope = float(np.polyfit(steps, values, 1)[0])
    scale = max(
        float(np.mean(np.abs(values))),
        1.0,
    )
    return abs(slope) / scale


def normalized_window_drift(
    rows: Sequence[dict[str, Any]],
    key: str,
    windows: int,
) -> float:
    if len(rows) < windows:
        return float("inf")
    tail = rows[-windows:]
    steps = np.asarray(
        [float(row["step"]) for row in tail],
        dtype=np.float64,
    )
    values = np.asarray(
        [float(row[key]) for row in tail],
        dtype=np.float64,
    )
    if (
        not np.all(np.isfinite(steps))
        or not np.all(np.isfinite(values))
        or steps[-1] <= steps[0]
    ):
        return float("inf")
    slope = float(
        np.polyfit(
            steps - steps[0],
            values,
            1,
        )[0]
    )
    span = float(steps[-1] - steps[0])
    scale = max(
        float(np.median(np.abs(values))),
        1.0e-3,
    )
    return abs(slope) * span / scale


def classify_actor_terminal(
    rows: Sequence[dict[str, Any]],
    protocol: ActorAuditProtocol,
    candidate_step: int | None,
    extension_complete: bool,
    fixed_budget_completed: bool = False,
) -> dict[str, Any]:
    last = rows[-1]
    relative_update_norm = float(
        last.get(
            "relative_update_norm",
            last.get(
                "update_norm",
                float("inf"),
            ),
        )
    )
    nonfinite = any(
        not math.isfinite(float(last[key]))
        for key in (
            "loss",
            "positive_nll",
            "gradient_norm",
            "update_norm",
            "sigma_mean",
        )
    ) or not math.isfinite(relative_update_norm)
    support_event = bool(
        float(last["mean_boundary_fraction"])
        >= protocol.support_boundary_fraction
        or float(last["log_std_min_fraction"]) > 0.0
        or float(last["log_std_max_fraction"]) > 0.0
    )
    slopes = {
        key: relative_slope(
            rows,
            key,
            protocol.audit_windows,
        )
        for key in (
            "positive_nll",
            "mean_abs",
            "sigma_mean",
            "phantom_distance_mean",
        )
    }
    state_drifts = {
        key: normalized_window_drift(
            rows,
            key,
            protocol.audit_windows,
        )
        for key in (
            "mean_abs",
            "sigma_mean",
            "phantom_distance_mean",
        )
    }
    stable = bool(
        candidate_step is not None
        and extension_complete
        and all(
            value
            <= protocol.actor_state_drift_tolerance
            for value in state_drifts.values()
        )
        and relative_update_norm
        <= protocol.actor_update_tolerance
        and not nonfinite
    )
    rollout_values = [
        float(
            row.get(
                "normalized_return",
                float("nan"),
            )
        )
        for row in rows
    ]
    finite_rollouts = [
        value
        for value in rollout_values
        if math.isfinite(value)
    ]
    rollout_statuses = {
        str(
            row.get(
                "rollout_status",
                "not_evaluated",
            )
        )
        for row in rows
    }
    initial_return = (
        finite_rollouts[0]
        if finite_rollouts
        else float("nan")
    )
    final_return = (
        finite_rollouts[-1]
        if finite_rollouts
        else float("nan")
    )
    if finite_rollouts:
        task_status = "available"
        task_collapse: bool | None = bool(
            initial_return - final_return
            >= protocol.task_return_drop_threshold
        )
    elif "unavailable" in rollout_statuses:
        task_status = "unavailable"
        task_collapse = None
    elif rollout_statuses == {"disabled"}:
        task_status = "disabled"
        task_collapse = None
    else:
        task_status = "not_evaluated"
        task_collapse = None

    if nonfinite:
        state = "nan_inf_numerical_collapse"
    elif stable and support_event:
        state = (
            "finite_terminal_with_"
            "support_boundary_event"
        )
    elif stable:
        state = "finite_terminal"
    elif support_event:
        state = (
            "support_or_variance_boundary_event_"
            "without_terminal_convergence"
        )
    elif (
        len(rows) >= protocol.audit_windows
        and any(
            value
            > protocol.actor_state_drift_tolerance
            for value in state_drifts.values()
        )
    ):
        state = "persistent_or_slow_drift"
    elif fixed_budget_completed:
        state = "fixed_horizon_inconclusive"
    else:
        state = (
            "training_incomplete_without_"
            "terminal_classification"
        )
    explicit = (
        state
        != "training_incomplete_without_terminal_classification"
    )
    return {
        "state": state,
        "candidate_step": candidate_step,
        "extension_complete": extension_complete,
        "fixed_budget_completed": fixed_budget_completed,
        "terminal_audit_controls_stopping": False,
        "slopes": slopes,
        "state_drifts": state_drifts,
        "state_drift_tolerance": (
            protocol.actor_state_drift_tolerance
        ),
        "relative_update_norm": relative_update_norm,
        "support_boundary_event": support_event,
        "numerical_nonfinite": nonfinite,
        "task_performance_status": task_status,
        "task_performance_collapse": task_collapse,
        "normalized_return_available": (
            task_status == "available"
        ),
        "initial_normalized_return": initial_return,
        "final_normalized_return": final_return,
        "task_return_drop_threshold": (
            protocol.task_return_drop_threshold
        ),
        "explicit_terminal_classification": explicit,
        "reporting_separation": [
            "task_performance_status_and_collapse",
            "support_or_variance_boundary_event",
            "nan_inf_numerical_collapse",
        ],
    }
