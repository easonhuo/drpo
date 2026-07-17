from __future__ import annotations

import math

import pytest
import torch

from drpo import du1_e6_cartesian_taper_v4 as legacy
from drpo_reference.categorical.du1_protocol import method_specs
from drpo_reference.categorical.du1_training import (
    DU1TerminalProtocol,
    build_shared_start,
    legacy_run_config,
    run_method,
)

from du1_helpers import small_protocol


def _assert_value(actual: object, expected: object) -> None:
    if isinstance(expected, bool) or expected is None:
        assert actual == expected
    elif isinstance(expected, (int, float)):
        expected_float = float(expected)
        if math.isnan(expected_float):
            assert math.isnan(float(actual))
        else:
            assert float(actual) == pytest.approx(
                expected_float,
                rel=1.0e-6,
                abs=1.0e-7,
            )
    elif isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert set(actual) == set(expected)
        for key, value in expected.items():
            _assert_value(actual[key], value)
    elif isinstance(expected, list):
        assert isinstance(actual, list)
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_value(actual_item, expected_item)
    else:
        assert actual == expected


@pytest.mark.parametrize(
    "method",
    ("positive_only", "global_matched"),
)
def test_short_method_trajectory_matches_revision_4(
    method: str,
) -> None:
    protocol = small_protocol()
    terminal = DU1TerminalProtocol(
        window_1_steps=(0, 2),
        window_2_steps=(2, 4),
    )
    config = legacy_run_config(protocol, terminal)
    seed = 204

    legacy.seed_all(seed)
    old_environment = legacy.CartesianSemanticEnvironment(
        config,
        seed,
    )
    old_model = legacy.CartesianPolicy(
        config,
        old_environment,
    ).to("cpu")
    old_environment.action_embeddings = (
        old_environment.action_embeddings.to("cpu")
    )
    for split in (old_environment.train, old_environment.test):
        for key, value in list(split.items()):
            if isinstance(value, torch.Tensor):
                split[key] = value.to("cpu")
    legacy.cache_reference_directions(
        old_model,
        old_environment,
    )
    (
        old_state,
        old_optimizer_state,
        _,
    ) = legacy.build_positive_warm_start(
        config,
        seed,
        old_model,
        old_environment,
        torch.device("cpu"),
    )
    old_calibration = legacy.coordinate_calibration(
        old_model,
        old_environment,
        config,
    )
    old_spec = legacy.method_specs([method])[0]
    expected_trajectory, expected_summary = legacy.run_one(
        config,
        seed,
        old_spec,
        old_state,
        old_optimizer_state,
        old_calibration,
        torch.device("cpu"),
    )

    shared = build_shared_start(
        protocol,
        seed,
        torch.device("cpu"),
    )
    actual = run_method(
        protocol=protocol,
        terminal=terminal,
        seed=seed,
        spec=method_specs((method,))[0],
        base_state=shared.state_dict,
        base_optimizer_state=shared.optimizer_state,
        calibration=shared.calibration,
        device=torch.device("cpu"),
    )

    assert len(actual.trajectory) == len(expected_trajectory)
    for actual_row, expected_row in zip(
        actual.trajectory,
        expected_trajectory,
    ):
        assert set(actual_row) == set(expected_row)
        for key, expected_value in expected_row.items():
            _assert_value(actual_row[key], expected_value)
    assert set(actual.summary) == set(expected_summary)
    for key, expected_value in expected_summary.items():
        _assert_value(actual.summary[key], expected_value)
