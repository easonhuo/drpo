from __future__ import annotations

from dataclasses import replace

import torch

from drpo import du1_e6_cartesian_taper_v4 as legacy
from drpo_reference.categorical.du1_environment import (
    CartesianSemanticEnvironment,
)
from drpo_reference.categorical.du1_policy import (
    CartesianPolicy,
    cache_reference_directions,
)
from drpo_reference.categorical.du1_protocol import DU1Protocol


def small_protocol() -> DU1Protocol:
    return replace(
        DU1Protocol(),
        train_states=48,
        test_states=40,
        hidden_dim=16,
        batch_size=12,
        audit_states=24,
        maximum_steps=4,
        evaluation_interval_steps=2,
        formal_seeds=(200,),
    )


def assert_split_equal(
    actual: dict[str, torch.Tensor],
    expected: dict[str, torch.Tensor],
) -> None:
    assert set(actual) == set(expected)
    for key in expected:
        torch.testing.assert_close(
            actual[key],
            expected[key],
            rtol=0.0,
            atol=0.0,
        )


def matching_models(
    protocol: DU1Protocol,
    seed: int,
) -> tuple[
    legacy.CartesianSemanticEnvironment,
    CartesianSemanticEnvironment,
    legacy.CartesianPolicy,
    CartesianPolicy,
]:
    config = protocol.legacy_config()
    old_environment = legacy.CartesianSemanticEnvironment(
        config,
        seed,
    )
    new_environment = CartesianSemanticEnvironment(
        protocol,
        seed,
    )
    torch.manual_seed(seed + 17)
    old_model = legacy.CartesianPolicy(
        config,
        old_environment,
    )
    torch.manual_seed(seed + 17)
    new_model = CartesianPolicy(
        protocol,
        new_environment,
    )
    assert set(old_model.state_dict()) == set(
        new_model.state_dict()
    )
    for name, old_value in old_model.state_dict().items():
        torch.testing.assert_close(
            new_model.state_dict()[name],
            old_value,
            rtol=0.0,
            atol=0.0,
        )
    legacy.cache_reference_directions(
        old_model,
        old_environment,
    )
    cache_reference_directions(
        new_model,
        new_environment,
    )
    return (
        old_environment,
        new_environment,
        old_model,
        new_model,
    )
