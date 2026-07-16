from __future__ import annotations

import numpy as np
import pytest

from drpo.e7_trajectory_gae import compute_gae_from_td


def test_terminal_timeout_and_tail_boundaries() -> None:
    td = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    terminal = np.array([False, True, False, False])
    timeout = np.array([False, False, True, False])
    result = compute_gae_from_td(td, terminal, timeout, gamma=0.9, gae_lambda=0.8)
    expected = np.array([1.0 + 0.9 * 0.8 * 2.0, 2.0, 3.0, 4.0], dtype=np.float32)
    np.testing.assert_allclose(result, expected)


def test_lambda_zero_equals_td() -> None:
    rng = np.random.default_rng(7)
    td = rng.normal(size=32).astype(np.float32)
    terminal = np.zeros(32, dtype=bool)
    timeout = np.zeros(32, dtype=bool)
    terminal[[5, 17]] = True
    timeout[[9, 23]] = True
    np.testing.assert_array_equal(
        compute_gae_from_td(td, terminal, timeout, gae_lambda=0.0), td
    )


def test_overlap_and_nonfinite_fail_closed() -> None:
    with pytest.raises(ValueError, match="overlap"):
        compute_gae_from_td(
            np.ones(2), np.array([True, False]), np.array([True, False])
        )
    with pytest.raises(ValueError, match="invalid"):
        compute_gae_from_td(
            np.array([np.nan]), np.array([False]), np.array([False])
        )
