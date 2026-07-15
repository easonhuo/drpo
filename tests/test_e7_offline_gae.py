from __future__ import annotations

import numpy as np
import pytest
import torch

from drpo.e7_offline_gae import (
    audit_ordered_trajectories,
    compute_gae_numpy,
    compute_gae_torch,
    compute_td_advantage,
)


def test_lambda_zero_is_exact_one_step_td() -> None:
    rewards = np.array([1.0, -2.0, 0.5, 3.0], dtype=np.float32)
    values = np.array([0.1, 0.2, -0.3, 0.4], dtype=np.float32)
    next_values = np.array([0.2, -0.3, 0.4, 0.5], dtype=np.float32)
    terminals = np.array([False, True, False, False])
    timeouts = np.array([False, False, True, False])

    td = compute_td_advantage(
        rewards, values, next_values, terminals, gamma=0.99
    )
    gae_zero = compute_gae_numpy(
        rewards,
        values,
        next_values,
        terminals,
        timeouts,
        gamma=0.99,
        gae_lambda=0.0,
    )

    np.testing.assert_array_equal(gae_zero, td)


def test_terminal_has_no_bootstrap_and_stops_trace() -> None:
    advantage = compute_gae_numpy(
        rewards=np.array([1.0, 7.0]),
        values=np.array([0.25, 0.0]),
        next_values=np.array([100.0, 0.0]),
        terminals=np.array([True, False]),
        timeouts=np.array([False, False]),
        gamma=0.9,
        gae_lambda=0.95,
    )

    assert advantage[0] == pytest.approx(0.75)
    assert advantage[1] == pytest.approx(7.0)


def test_timeout_bootstraps_but_stops_trace() -> None:
    advantage = compute_gae_numpy(
        rewards=np.array([1.0, 7.0]),
        values=np.array([0.25, 0.0]),
        next_values=np.array([2.0, 0.0]),
        terminals=np.array([False, False]),
        timeouts=np.array([True, False]),
        gamma=0.9,
        gae_lambda=0.95,
    )

    assert advantage[0] == pytest.approx(1.0 + 0.9 * 2.0 - 0.25)
    assert advantage[1] == pytest.approx(7.0)


def test_open_dataset_tail_bootstraps_and_has_no_future_carry() -> None:
    advantage = compute_gae_numpy(
        rewards=np.array([1.0]),
        values=np.array([0.25]),
        next_values=np.array([2.0]),
        terminals=np.array([False]),
        timeouts=np.array([False]),
        gamma=0.9,
        gae_lambda=0.95,
    )

    assert advantage[0] == pytest.approx(1.0 + 0.9 * 2.0 - 0.25)


def test_trajectory_audit_rejects_nonboundary_discontinuity() -> None:
    observations = np.array([[0.0], [1.0]], dtype=np.float32)
    next_observations = np.array([[2.0], [3.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="trajectory audit failed"):
        audit_ordered_trajectories(
            observations,
            next_observations,
            terminals=np.array([False, False]),
            timeouts=np.array([False, False]),
        )


def test_trajectory_audit_rejects_terminal_timeout_overlap() -> None:
    with pytest.raises(ValueError, match="trajectory audit failed"):
        audit_ordered_trajectories(
            observations=np.array([[0.0]], dtype=np.float32),
            next_observations=np.array([[1.0]], dtype=np.float32),
            terminals=np.array([True]),
            timeouts=np.array([True]),
        )


def test_numpy_and_torch_implementations_match() -> None:
    rewards = np.array([1.0, 2.0, -1.0, 0.5], dtype=np.float64)
    values = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    next_values = np.array([0.2, 0.3, 0.4, 0.5], dtype=np.float64)
    terminals = np.array([False, True, False, False])
    timeouts = np.array([False, False, True, False])

    numpy_result = compute_gae_numpy(
        rewards,
        values,
        next_values,
        terminals,
        timeouts,
        gamma=0.99,
        gae_lambda=0.95,
    )
    torch_result = compute_gae_torch(
        torch.from_numpy(rewards),
        torch.from_numpy(values),
        torch.from_numpy(next_values),
        torch.from_numpy(terminals),
        torch.from_numpy(timeouts),
        gamma=0.99,
        gae_lambda=0.95,
    ).numpy()

    np.testing.assert_allclose(numpy_result, torch_result, rtol=0.0, atol=1e-6)
