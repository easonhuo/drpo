from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from drpo import e7_hopper_q2 as legacy
from drpo_reference.external.hopper_metrics import (
    aggregate_negative_gradient_norm,
    analytic_output_autograd_relative_error,
    classify_actor_terminal,
    create_gradient_probe,
    loglog_slope,
    match_near_far_indices,
    normalized_window_drift,
    pearson,
    per_sample_gradient_norm,
    r2_score,
    relative_slope,
    resolve_global_scale,
)
from drpo_reference.external.hopper_models import SquashedGaussianPolicy


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
            assert actual == pytest.approx(
                expected,
                rel=1.0e-12,
                abs=1.0e-12,
            )
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
                    40.0 - 5.0 * index if rollout_status == "available" else float("nan")
                ),
            }
        )
    return output


def _policy_pair() -> tuple[
    legacy.SquashedGaussianPolicy,
    SquashedGaussianPolicy,
]:
    torch.manual_seed(19)
    expected = legacy.SquashedGaussianPolicy(
        4,
        2,
        (8,),
        -5.0,
        2.0,
        1.0e-6,
        "tanh",
        "default",
        1.0,
    )
    torch.manual_seed(19)
    actual = SquashedGaussianPolicy(
        4,
        2,
        (8,),
        -5.0,
        2.0,
        1.0e-6,
        "tanh",
        "default",
        1.0,
    )
    return expected, actual


def _mechanism_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    generator = np.random.default_rng(23)
    observations = generator.normal(size=(48, 4)).astype(np.float32)
    actions = np.tanh(generator.normal(size=(48, 2))).astype(np.float32)
    magnitudes = np.repeat(
        np.linspace(0.25, 2.0, 12, dtype=np.float32),
        4,
    )
    advantages = -magnitudes
    return observations, actions, advantages


def test_scalar_metrics_match() -> None:
    truth = np.asarray([1.0, 2.0, 3.0, 5.0], dtype=np.float64)
    predicted = np.asarray([1.1, 1.9, 2.7, 5.2], dtype=np.float64)
    assert r2_score(truth, predicted) == legacy.r2_score(truth, predicted)
    assert pearson(truth, predicted) == legacy.pearson(truth, predicted)
    rows = _rows()
    for key in ("positive_nll", "mean_abs", "sigma_mean"):
        assert relative_slope(
            rows,
            key,
            4,
        ) == legacy.relative_slope(rows, key, 4)
        assert normalized_window_drift(
            rows,
            key,
            4,
        ) == legacy.normalized_window_drift(rows, key, 4)


@pytest.mark.parametrize(
    (
        "support",
        "rollout_status",
        "candidate_step",
        "extension_complete",
        "fixed",
    ),
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


def test_advantage_matched_near_far_indices_match() -> None:
    advantages = -np.repeat(
        np.linspace(0.2, 1.4, 12, dtype=np.float64),
        4,
    )
    distances = np.tile(
        np.asarray([0.3, 0.6, 2.5, 3.5], dtype=np.float64),
        12,
    )
    negative_indices = np.arange(len(advantages), dtype=np.int64)
    expected = legacy.match_near_far_indices(
        advantages,
        distances,
        negative_indices,
        0.25,
        0.75,
        6,
        20,
        0.01,
        31,
    )
    actual = match_near_far_indices(
        advantages,
        distances,
        negative_indices,
        0.25,
        0.75,
        6,
        20,
        0.01,
        31,
    )
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    _assert_nested(actual[2], expected[2])


def test_gradient_and_formula_diagnostics_match() -> None:
    expected_policy, actual_policy = _policy_pair()
    observations, actions, advantages = _mechanism_arrays()
    indices = np.asarray([0, 3, 8, 11, 19, 27], dtype=np.int64)
    expected_norms = legacy.per_sample_gradient_norm(
        expected_policy,
        observations,
        actions,
        advantages,
        indices,
        torch.device("cpu"),
    )
    actual_norms = per_sample_gradient_norm(
        actual_policy,
        observations,
        actions,
        advantages,
        indices,
        "cpu",
    )
    np.testing.assert_allclose(
        actual_norms,
        expected_norms,
        rtol=1.0e-7,
        atol=1.0e-8,
    )
    assert aggregate_negative_gradient_norm(
        actual_policy,
        observations,
        actions,
        advantages,
        indices,
        "cpu",
    ) == pytest.approx(
        legacy.aggregate_negative_gradient_norm(
            expected_policy,
            observations,
            actions,
            advantages,
            indices,
            torch.device("cpu"),
        ),
        rel=1.0e-7,
        abs=1.0e-8,
    )
    expected_error = legacy.analytic_output_autograd_relative_error(
        expected_policy,
        observations,
        actions,
        indices,
        torch.device("cpu"),
    )
    actual_error = analytic_output_autograd_relative_error(
        actual_policy,
        observations,
        actions,
        indices,
        "cpu",
    )
    assert actual_error == pytest.approx(
        expected_error,
        rel=1.0e-7,
        abs=1.0e-8,
    )
    x_values = np.asarray([1.0, 2.0, 4.0, 8.0])
    y_values = x_values**2
    assert loglog_slope(x_values, y_values) == legacy.loglog_slope(
        x_values,
        y_values,
    )


def test_global_budget_matching_matches() -> None:
    expected_policy, actual_policy = _policy_pair()
    observations, actions, advantages = _mechanism_arrays()
    negative_indices = np.arange(len(advantages), dtype=np.int64)
    expected = legacy.resolve_global_scale(
        policy=expected_policy,
        obs=observations,
        actions=actions,
        advantages=advantages,
        negative_indices=negative_indices,
        far_threshold=1.5,
        far_cap_score=2.0,
        audit_size=24,
        seed=37,
        device=torch.device("cpu"),
    )
    actual = resolve_global_scale(
        policy=actual_policy,
        observations=observations,
        actions=actions,
        advantages=advantages,
        negative_indices=negative_indices,
        far_threshold=1.5,
        far_cap_score=2.0,
        audit_size=24,
        seed=37,
        device="cpu",
    )
    _assert_nested(actual, expected)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_gradient_probe_summary_and_artifacts_match(
    tmp_path: Path,
) -> None:
    expected_policy, actual_policy = _policy_pair()
    observations, actions, advantages = _mechanism_arrays()
    with torch.no_grad():
        distances = (
            expected_policy.standardized_distance(
                torch.as_tensor(observations),
                torch.as_tensor(actions),
            )
            .cpu()
            .numpy()
        )
    negative_indices = np.arange(len(advantages), dtype=np.int64)
    near_indices, far_indices, _ = legacy.match_near_far_indices(
        advantages,
        distances,
        negative_indices,
        0.25,
        0.75,
        6,
        12,
        0.25,
        41,
    )
    population = negative_indices[:32]
    expected_dir = tmp_path / "legacy"
    actual_dir = tmp_path / "reference"
    expected = legacy.create_gradient_probe(
        policy=expected_policy,
        obs=observations,
        actions=actions,
        advantages=advantages,
        near_indices=near_indices,
        far_indices=far_indices,
        population_indices=population,
        max_gradient_pairs=6,
        distance_bins=4,
        device=torch.device("cpu"),
        output_dir=expected_dir,
    )
    actual = create_gradient_probe(
        policy=actual_policy,
        observations=observations,
        actions=actions,
        advantages=advantages,
        near_indices=near_indices,
        far_indices=far_indices,
        population_indices=population,
        max_gradient_pairs=6,
        distance_bins=4,
        device="cpu",
        output_dir=actual_dir,
    )
    _assert_nested(actual, expected)
    assert _read_csv(actual_dir / "matched_near_far_components.csv") == _read_csv(
        expected_dir / "matched_near_far_components.csv"
    )
    assert _read_csv(actual_dir / "component_distance_bins.csv") == _read_csv(
        expected_dir / "component_distance_bins.csv"
    )
    _assert_nested(
        json.loads((actual_dir / "gradient_probe_summary.json").read_text()),
        json.loads((expected_dir / "gradient_probe_summary.json").read_text()),
    )
