from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from drpo import e7_hopper_q2 as legacy
from drpo_reference.external.hopper_metrics import (
    classify_actor_terminal,
    normalized_window_drift,
    pearson,
    r2_score,
    relative_slope,
)


@dataclass(frozen=True)
class AuditConfig:
    audit_windows: int = 4
    actor_state_drift_tolerance: float = 0.03
    actor_update_tolerance: float = 0.002
    support_boundary_fraction: float = 0.10
    task_return_drop_threshold: float = 20.0


def _assert_nested(actual: Any, expected: Any) -> None:
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert set(actual) == set(expected)
        for key in expected:
            _assert_nested(actual[key], expected[key])
    elif isinstance(expected, list):
        assert actual == expected
    elif isinstance(expected, float):
        if math.isnan(expected):
            assert math.isnan(actual)
        else:
            assert actual == pytest.approx(expected, rel=1.0e-12, abs=1.0e-12)
    else:
        assert actual == expected


def _rows(
    *,
    support: bool = False,
    rollout_status: str = "available",
) -> list[dict[str, Any]]:
    output = []
    for index, step in enumerate((0, 100, 200, 300, 400, 500)):
        output.append(
            {
                "step": step,
                "loss": 1.0 - 0.01 * index,
                "positive_nll": 2.0 + 0.001 * index,
                "gradient_norm": 0.3,
                "update_norm": 0.001,
                "relative_update_norm": 0.001,
                "sigma_mean": 0.5 + 0.0001 * index,
                "mean_abs": 0.2 + 0.0001 * index,
                "phantom_distance_mean": 4.0 + 0.0001 * index,
                "mean_boundary_fraction": 0.2 if support else 0.0,
                "log_std_min_fraction": 0.0,
                "log_std_max_fraction": 0.0,
                "rollout_status": rollout_status,
                "normalized_return": (
                    40.0 - 5.0 * index
                    if rollout_status == "available"
                    else float("nan")
                ),
            }
        )
    return output


def test_scalar_metrics_match() -> None:
    truth = np.asarray([1.0, 2.0, 3.0, 5.0], dtype=np.float64)
    predicted = np.asarray([1.1, 1.9, 2.7, 5.2], dtype=np.float64)
    assert r2_score(truth, predicted) == legacy.r2_score(truth, predicted)
    assert pearson(truth, predicted) == legacy.pearson(truth, predicted)
    rows = _rows()
    for key in ("positive_nll", "mean_abs", "sigma_mean"):
        assert relative_slope(rows, key, 4) == legacy.relative_slope(rows, key, 4)
        assert normalized_window_drift(
            rows, key, 4
        ) == legacy.normalized_window_drift(rows, key, 4)


@pytest.mark.parametrize(
    ("support", "rollout_status", "candidate_step", "extension_complete", "fixed"),
    (
        (False, "available", 300, True, True),
        (True, "available", 300, True, True),
        (False, "unavailable", None, False, True),
        (False, "disabled", None, False, False),
    ),
)
def test_terminal_classification_matches(
    support: bool,
    rollout_status: str,
    candidate_step: int | None,
    extension_complete: bool,
    fixed: bool,
) -> None:
    rows = _rows(support=support, rollout_status=rollout_status)
    config = AuditConfig()
    expected = legacy.classify_actor_terminal(
        rows,
        config,
        candidate_step,
        extension_complete,
        fixed_budget_completed=fixed,
    )
    actual = classify_actor_terminal(
        rows,
        config,
        candidate_step,
        extension_complete,
        fixed_budget_completed=fixed,
    )
    _assert_nested(actual, expected)
