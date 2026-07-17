from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from drpo import e7_hopper_q2 as legacy
from drpo_reference.external.hopper_data import (
    Normalizer,
    build_episode_ids,
    discounted_returns,
    load_hopper_hdf5,
    split_episode_indices,
)


def _write_dataset(path: Path, *, include_optional: bool) -> None:
    observations = np.arange(60, dtype=np.float32).reshape(12, 5) / 10.0
    actions = np.linspace(-0.9, 0.9, 36, dtype=np.float32).reshape(12, 3)
    rewards = np.asarray(
        [1.0, 2.0, -1.0, 0.5, 3.0, 1.0, -2.0, 4.0, 0.0, 1.5, 2.5, -0.5],
        dtype=np.float32,
    )
    terminals = np.asarray(
        [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
        dtype=np.bool_,
    )
    timeouts = np.asarray(
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
        dtype=np.bool_,
    )
    with h5py.File(path, "w") as handle:
        handle["observations"] = observations
        handle["actions"] = actions
        handle["rewards"] = rewards
        handle["terminals"] = terminals
        if include_optional:
            handle["timeouts"] = timeouts
            handle["next_observations"] = observations + 0.25


def _assert_data_equal(actual: object, expected: object) -> None:
    for field in (
        "observations",
        "actions",
        "rewards",
        "next_observations",
        "terminals",
        "timeouts",
        "episode_ids",
    ):
        np.testing.assert_array_equal(
            getattr(actual, field),
            getattr(expected, field),
        )
    assert actual.size == expected.size


def test_hdf5_loader_matches_with_optional_arrays(tmp_path: Path) -> None:
    path = tmp_path / "hopper.hdf5"
    _write_dataset(path, include_optional=True)
    expected = legacy.load_hopper_hdf5(path, 10)
    actual = load_hopper_hdf5(path, 10)
    _assert_data_equal(actual, expected)


def test_hdf5_loader_matches_fallback_arrays(tmp_path: Path) -> None:
    path = tmp_path / "hopper_fallback.hdf5"
    _write_dataset(path, include_optional=False)
    expected = legacy.load_hopper_hdf5(path, None)
    actual = load_hopper_hdf5(path)
    _assert_data_equal(actual, expected)


def test_episode_returns_split_and_normalizer_match() -> None:
    terminals = np.asarray(
        [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0],
        dtype=np.bool_,
    )
    timeouts = np.asarray(
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        dtype=np.bool_,
    )
    rewards = np.linspace(-1.0, 2.0, len(terminals), dtype=np.float32)
    expected_ids = legacy.build_episode_ids(terminals, timeouts)
    actual_ids = build_episode_ids(terminals, timeouts)
    np.testing.assert_array_equal(actual_ids, expected_ids)
    np.testing.assert_array_equal(
        discounted_returns(rewards, terminals, timeouts, 0.99),
        legacy.discounted_returns(rewards, terminals, timeouts, 0.99),
    )
    expected_split = legacy.split_episode_indices(expected_ids, 17, 0.6, 0.2)
    actual_split = split_episode_indices(actual_ids, 17, 0.6, 0.2)
    assert set(actual_split) == set(expected_split)
    for name in expected_split:
        np.testing.assert_array_equal(actual_split[name], expected_split[name])

    array = np.arange(35, dtype=np.float32).reshape(7, 5) / 7.0
    expected_normalizer = legacy.Normalizer.fit(array)
    actual_normalizer = Normalizer.fit(array)
    np.testing.assert_array_equal(actual_normalizer.mean, expected_normalizer.mean)
    np.testing.assert_array_equal(actual_normalizer.std, expected_normalizer.std)
    np.testing.assert_array_equal(
        actual_normalizer.transform(array),
        expected_normalizer.transform(array),
    )
