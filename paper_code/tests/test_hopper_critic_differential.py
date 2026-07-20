from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from drpo import e7_hopper_q2 as legacy
from drpo_reference.external.hopper_advantages import (
    critic_advantage_arrays,
)
from drpo_reference.external.hopper_critic import train_critic
from drpo_reference.external.hopper_optim import (
    full_gradient_statistics,
    parameter_update_statistics,
    rankdata,
    sample_indices,
    spearman,
)
from drpo_reference.external.hopper_data import Normalizer, OfflineData
from drpo_reference.external.hopper_models import ValueNetwork
from drpo_reference.external.hopper_protocol import HopperProtocol


def _assert_nested_close(actual: object, expected: object) -> None:
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert set(actual) == set(expected)
        for key, value in expected.items():
            _assert_nested_close(actual[key], value)
        return
    if isinstance(expected, list):
        assert isinstance(actual, list)
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_nested_close(actual_item, expected_item)
        return
    if isinstance(expected, float):
        if np.isnan(expected):
            assert np.isnan(float(actual))
        else:
            assert float(actual) == pytest.approx(
                expected,
                rel=1.0e-6,
                abs=1.0e-7,
            )
        return
    assert actual == expected


def _data() -> OfflineData:
    generator = np.random.default_rng(9)
    observations = generator.normal(size=(24, 5)).astype(np.float32)
    actions = np.tanh(generator.normal(size=(24, 3))).astype(np.float32)
    rewards = generator.normal(size=24).astype(np.float32)
    terminals = np.zeros(24, dtype=np.bool_)
    terminals[[5, 11, 17, 23]] = True
    timeouts = np.zeros(24, dtype=np.bool_)
    episode_ids = np.repeat(np.arange(4), 6).astype(np.int64)
    return OfflineData(
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=np.roll(observations, -1, axis=0),
        terminals=terminals,
        timeouts=timeouts,
        episode_ids=episode_ids,
    )


def test_sampling_rank_and_gradient_utilities_match() -> None:
    pool = np.arange(7, dtype=np.int64)
    old_rng = np.random.default_rng(17)
    new_rng = np.random.default_rng(17)
    np.testing.assert_array_equal(
        sample_indices(new_rng, pool, 11),
        legacy.sample_indices(old_rng, pool, 11),
    )
    values = np.asarray([3.0, 1.0, 1.0, 5.0, 2.0])
    np.testing.assert_array_equal(rankdata(values), legacy._rankdata(values))
    assert spearman(values, values[::-1]) == legacy.spearman(
        values,
        values[::-1],
    )

    torch.manual_seed(3)
    old_model = legacy.ValueNetwork(5, (8,), "tanh", "default", 1.0)
    torch.manual_seed(3)
    new_model = ValueNetwork(5, (8,), "tanh", "default", 1.0)
    features = torch.randn(6, 5, generator=torch.Generator().manual_seed(4))
    old_loss = old_model(features).square().mean()
    new_loss = new_model(features).square().mean()
    expected = legacy.full_gradient_statistics(
        old_loss,
        old_model.parameters(),
    )
    actual = full_gradient_statistics(
        new_loss,
        new_model.parameters(),
    )
    assert actual == pytest.approx(expected, rel=1.0e-7, abs=1.0e-8)

    old_snapshot = [parameter.detach().clone() for parameter in old_model.parameters()]
    new_snapshot = [parameter.detach().clone() for parameter in new_model.parameters()]
    with torch.no_grad():
        for old_parameter, new_parameter in zip(
            old_model.parameters(),
            new_model.parameters(),
        ):
            old_parameter.add_(0.01)
            new_parameter.add_(0.01)
    assert parameter_update_statistics(
        new_snapshot,
        new_model.parameters(),
        2,
    ) == pytest.approx(
        legacy.parameter_update_statistics(
            old_snapshot,
            old_model.parameters(),
            2,
        ),
        rel=1.0e-7,
        abs=1.0e-8,
    )


def test_critic_advantage_arrays_match() -> None:
    data = _data()
    observation_normalizer = Normalizer.fit(data.observations)
    targets = np.linspace(-2.0, 3.0, data.size, dtype=np.float32)
    target_normalizer = Normalizer.fit(targets.reshape(-1, 1))
    torch.manual_seed(7)
    old_critic = legacy.ValueNetwork(5, (8,), "tanh", "default", 1.0)
    torch.manual_seed(7)
    new_critic = ValueNetwork(5, (8,), "tanh", "default", 1.0)
    indices = np.arange(18, dtype=np.int64)
    expected = legacy.critic_advantage_arrays(
        critic=old_critic,
        data=data,
        obs_norm=observation_normalizer,
        target_norm=target_normalizer,
        gamma=0.99,
        standardize=True,
        standardization_indices=indices,
        device=torch.device("cpu"),
    )
    actual = critic_advantage_arrays(
        critic=new_critic,
        data=data,
        observation_normalizer=observation_normalizer,
        target_normalizer=target_normalizer,
        gamma=0.99,
        standardize=True,
        standardization_indices=indices,
        device=torch.device("cpu"),
    )
    assert set(actual) == set(expected)
    for key in ("advantage", "raw_advantage", "value", "next_value"):
        np.testing.assert_allclose(
            actual[key],
            expected[key],
            rtol=0.0,
            atol=0.0,
        )
    assert actual["center"] == expected["center"]
    assert actual["scale"] == expected["scale"]


def test_protocol_contains_frozen_critic_contract() -> None:
    protocol = HopperProtocol()
    assert protocol.critic_steps == 100_000
    assert protocol.critic_min_steps == 50_000
    assert protocol.canonical_critic_seed == 100
    smoke = replace(
        protocol,
        execution_profile="smoke",
        formal_seeds=(1,),
        critic_steps=4,
        critic_min_steps=2,
    )
    assert smoke.critic_steps == 4


def test_short_fixed_budget_critic_matches_authoritative_runner(
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    from drpo_reference.external.hopper_data import (
        discounted_returns,
        split_episode_indices,
    )

    root = Path(tmp_path)
    data = _data()
    split = split_episode_indices(data.episode_ids, 29, 0.6, 0.2)
    observation_normalizer = Normalizer.fit(data.observations[split["train"]])
    returns = discounted_returns(
        data.rewards,
        data.terminals,
        data.timeouts,
        0.99,
    )
    protocol = replace(
        HopperProtocol(),
        hidden_sizes=(8,),
        critic_batch_size=8,
        audit_sample_size=12,
        audit_windows=2,
        critic_min_steps=2,
        critic_steps=4,
        critic_eval_interval=2,
        critic_validation_r2_min=-100.0,
        critic_validation_pearson_min=-1.0,
        critic_max_final_to_best_validation_mse_ratio=1.0e9,
        critic_advantage_sign_agreement_min=0.0,
        critic_advantage_pearson_min=-1.0,
        critic_advantage_spearman_min=-1.0,
        critic_negative_set_jaccard_min=0.0,
    )
    old_config = SimpleNamespace(
        hidden_sizes=protocol.hidden_sizes,
        activation=protocol.activation,
        init_scheme=protocol.init_scheme,
        init_gain=protocol.init_gain,
        critic_lr=protocol.critic_learning_rate,
        weight_decay=protocol.weight_decay,
        critic_batch_size=protocol.critic_batch_size,
        audit_windows=protocol.audit_windows,
        critic_relative_slope_tolerance=(protocol.critic_relative_slope_tolerance),
        critic_update_tolerance=protocol.critic_update_tolerance,
        gamma=protocol.gamma,
        advantage_standardize=protocol.advantage_standardize_once,
        critic_validation_r2_min=protocol.critic_validation_r2_min,
        critic_validation_pearson_min=(protocol.critic_validation_pearson_min),
        critic_max_final_to_best_validation_mse_ratio=(
            protocol.critic_max_final_to_best_validation_mse_ratio
        ),
        critic_advantage_sign_agreement_min=(protocol.critic_advantage_sign_agreement_min),
        critic_advantage_pearson_min=(protocol.critic_advantage_pearson_min),
        critic_advantage_spearman_min=(protocol.critic_advantage_spearman_min),
        critic_negative_set_jaccard_min=(protocol.critic_negative_set_jaccard_min),
    )
    old_mode = SimpleNamespace(
        audit_sample_size=protocol.audit_sample_size,
        critic_max_steps=protocol.critic_steps,
        critic_eval_interval=protocol.critic_eval_interval,
        critic_min_steps=protocol.critic_min_steps,
    )

    torch.manual_seed(31)
    expected_model, expected_target, expected_audit = legacy.train_critic(
        data=data,
        split=split,
        obs_norm=observation_normalizer,
        returns=returns,
        config=old_config,
        mode=old_mode,
        seed=29,
        device=torch.device("cpu"),
        output_dir=root / "legacy",
    )
    expected_advantages, _ = legacy.freeze_advantages(
        critic=expected_model,
        data=data,
        obs_norm=observation_normalizer,
        target_norm=expected_target,
        gamma=protocol.gamma,
        standardize=protocol.advantage_standardize_once,
        standardization_indices=split["train"],
        device=torch.device("cpu"),
        output_dir=root / "legacy" / "advantages",
    )
    torch.manual_seed(31)
    actual = train_critic(
        data=data,
        split=split,
        observation_normalizer=observation_normalizer,
        returns=returns,
        protocol=protocol,
        seed=29,
        device="cpu",
        output_dir=root / "reference",
    )
    np.testing.assert_array_equal(
        actual.target_normalizer.mean,
        expected_target.mean,
    )
    np.testing.assert_array_equal(
        actual.target_normalizer.std,
        expected_target.std,
    )
    for name, expected_value in expected_model.state_dict().items():
        torch.testing.assert_close(
            actual.critic.state_dict()[name],
            expected_value,
            rtol=1.0e-7,
            atol=1.0e-8,
        )
    for key in (
        "best_step",
        "best_validation_mse",
        "fixed_budget_steps",
        "fixed_budget_completed",
        "early_stop_reason",
        "candidate_step",
        "extension_target",
        "extension_complete",
        "final_stationarity_reconfirmed",
        "validation_mse_relative_slope",
        "train_audit_loss_relative_slope",
        "optimization_terminal",
        "critic_accepted_for_frozen_advantage",
        "operational_acceptance_checks",
        "critic_quality_audit_passed",
        "quality_audit_checks",
        "acceptance_metrics",
        "selected_checkpoint_role",
        "selected_checkpoint_step",
        "terminal_checkpoint_eligible",
        "selected_checkpoint_metrics",
        "final_training_metrics",
    ):
        _assert_nested_close(
            actual.audit[key],
            expected_audit[key],
        )
    for key in (
        "advantage",
        "raw_advantage",
        "value",
        "next_value",
    ):
        if key == "advantage":
            expected_array = expected_advantages
        else:
            expected_array = np.load(root / "legacy" / "advantages" / "frozen_advantages.npz")[key]
        np.testing.assert_allclose(
            actual.advantages[key],
            expected_array,
            rtol=1.0e-7,
            atol=1.0e-8,
        )
