from __future__ import annotations

import torch

from drpo import e7_hopper_q2 as legacy
from drpo_reference.external.hopper_models import (
    SquashedGaussianPolicy,
    ValueNetwork,
)


def _assert_state_equal(actual: torch.nn.Module, expected: torch.nn.Module) -> None:
    assert set(actual.state_dict()) == set(expected.state_dict())
    for name, expected_value in expected.state_dict().items():
        torch.testing.assert_close(
            actual.state_dict()[name],
            expected_value,
            rtol=0.0,
            atol=0.0,
        )


def test_value_network_initialization_and_output_match() -> None:
    torch.manual_seed(13)
    expected = legacy.ValueNetwork(5, (12, 8), "tanh", "orthogonal", 0.7)
    torch.manual_seed(13)
    actual = ValueNetwork(5, (12, 8), "tanh", "orthogonal", 0.7)
    _assert_state_equal(actual, expected)
    observations = torch.randn(9, 5, generator=torch.Generator().manual_seed(41))
    torch.testing.assert_close(
        actual(observations),
        expected(observations),
        rtol=0.0,
        atol=0.0,
    )


def test_policy_initialization_log_prob_and_scores_match() -> None:
    kwargs = {
        "obs_dim": 5,
        "action_dim": 3,
        "hidden_sizes": (11, 7),
        "log_std_min": -5.0,
        "log_std_max": 2.0,
        "action_clip_epsilon": 1.0e-6,
        "activation": "relu",
        "init_scheme": "orthogonal",
        "init_gain": 0.8,
    }
    torch.manual_seed(23)
    expected = legacy.SquashedGaussianPolicy(**kwargs)
    torch.manual_seed(23)
    actual = SquashedGaussianPolicy(**kwargs)
    _assert_state_equal(actual, expected)
    generator = torch.Generator().manual_seed(57)
    observations = torch.randn(10, 5, generator=generator)
    actions = torch.tanh(torch.randn(10, 3, generator=generator))

    torch.testing.assert_close(
        actual.log_prob(observations, actions),
        expected.log_prob(observations, actions),
        rtol=1.0e-7,
        atol=1.0e-8,
    )
    expected_components = expected.score_components(observations, actions)
    actual_components = actual.score_components(observations, actions)
    assert set(actual_components) == set(expected_components)
    for name in expected_components:
        torch.testing.assert_close(
            actual_components[name],
            expected_components[name],
            rtol=1.0e-7,
            atol=1.0e-8,
        )
    torch.testing.assert_close(
        actual.standardized_distance(observations, actions),
        expected.standardized_distance(observations, actions),
        rtol=1.0e-7,
        atol=1.0e-8,
    )
    torch.testing.assert_close(
        actual.output_score_norm(observations, actions),
        expected.output_score_norm(observations, actions),
        rtol=1.0e-7,
        atol=1.0e-8,
    )
