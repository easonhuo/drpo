from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from drpo import e7_hopper_q2 as legacy
from drpo_reference.external.hopper_actor import (
    actor_batch_loss,
    actor_eval_metrics,
    train_actor_stage,
)
from drpo_reference.external.hopper_models import SquashedGaussianPolicy
from drpo_reference.external.hopper_optim import full_gradient_statistics
from drpo_reference.external.hopper_protocol import METHODS, HopperProtocol


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


def _arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    generator = np.random.default_rng(41)
    observations = generator.normal(size=(32, 5)).astype(np.float32)
    actions = np.tanh(generator.normal(size=(32, 3))).astype(np.float32)
    advantages = np.linspace(-2.0, 2.0, 32).astype(np.float32)
    return observations, actions, advantages


def _old_policy(seed: int = 13) -> legacy.SquashedGaussianPolicy:
    torch.manual_seed(seed)
    return legacy.SquashedGaussianPolicy(
        5,
        3,
        (8,),
        -5.0,
        2.0,
        1.0e-6,
        "tanh",
        "default",
        1.0,
    )


def _new_policy(seed: int = 13) -> SquashedGaussianPolicy:
    torch.manual_seed(seed)
    return SquashedGaussianPolicy(
        5,
        3,
        (8,),
        -5.0,
        2.0,
        1.0e-6,
        "tanh",
        "default",
        1.0,
    )


def _thresholds(
    policy: SquashedGaussianPolicy,
    observations: np.ndarray,
    actions: np.ndarray,
) -> tuple[float, float]:
    with torch.no_grad():
        components = policy.score_components(
            torch.as_tensor(observations),
            torch.as_tensor(actions),
        )
    return (
        float(torch.quantile(components["radius"], 0.5)),
        float(
            torch.quantile(
                components["joint_output_score_norm"],
                0.6,
            )
        ),
    )


@pytest.mark.parametrize("method", METHODS)
def test_actor_loss_gradient_and_first_adamw_update_match(
    method: str,
) -> None:
    observations, actions, advantages = _arrays()
    old_policy = _old_policy()
    new_policy = _new_policy()
    far_threshold, far_cap_score = _thresholds(
        new_policy,
        observations,
        actions,
    )
    observation_t = torch.as_tensor(observations)
    action_t = torch.as_tensor(actions)
    advantage_t = torch.as_tensor(advantages)

    expected_loss, expected_diagnostics = legacy.actor_batch_loss(
        old_policy,
        observation_t,
        action_t,
        advantage_t,
        method,
        far_threshold,
        0.5,
        far_cap_score,
    )
    actual_loss, actual_diagnostics = actor_batch_loss(
        new_policy,
        observation_t,
        action_t,
        advantage_t,
        method,
        far_threshold,
        0.5,
        far_cap_score,
    )
    torch.testing.assert_close(
        actual_loss,
        expected_loss,
        rtol=1.0e-7,
        atol=1.0e-8,
    )
    assert actual_diagnostics == pytest.approx(
        expected_diagnostics,
        rel=1.0e-7,
        abs=1.0e-8,
    )

    expected_gradient = torch.autograd.grad(
        expected_loss,
        tuple(old_policy.parameters()),
        retain_graph=True,
    )
    actual_gradient = torch.autograd.grad(
        actual_loss,
        tuple(new_policy.parameters()),
        retain_graph=True,
    )
    for actual, expected in zip(actual_gradient, expected_gradient):
        torch.testing.assert_close(
            actual,
            expected,
            rtol=1.0e-7,
            atol=1.0e-8,
        )

    old_optimizer = torch.optim.AdamW(
        old_policy.parameters(),
        lr=3.0e-4,
        weight_decay=1.0e-4,
    )
    new_optimizer = torch.optim.AdamW(
        new_policy.parameters(),
        lr=3.0e-4,
        weight_decay=1.0e-4,
    )
    old_optimizer.zero_grad(set_to_none=True)
    new_optimizer.zero_grad(set_to_none=True)
    expected_loss.backward()
    actual_loss.backward()
    torch.nn.utils.clip_grad_norm_(old_policy.parameters(), 100.0)
    torch.nn.utils.clip_grad_norm_(new_policy.parameters(), 100.0)
    old_optimizer.step()
    new_optimizer.step()
    for name, expected in old_policy.state_dict().items():
        torch.testing.assert_close(
            new_policy.state_dict()[name],
            expected,
            rtol=1.0e-7,
            atol=1.0e-8,
        )


def test_actor_eval_metrics_match_authoritative_runner() -> None:
    observations, actions, advantages = _arrays()
    old_policy = _old_policy()
    new_policy = _new_policy()
    audit_indices = np.arange(24, dtype=np.int64)
    negative_indices = np.flatnonzero(advantages < 0)[:8]
    old_loss = legacy.actor_batch_loss(
        old_policy,
        torch.as_tensor(observations[audit_indices]),
        torch.as_tensor(actions[audit_indices]),
        torch.as_tensor(advantages[audit_indices]),
        "signed",
        2.0,
        1.0,
        3.0,
    )[0]
    new_loss = actor_batch_loss(
        new_policy,
        torch.as_tensor(observations[audit_indices]),
        torch.as_tensor(actions[audit_indices]),
        torch.as_tensor(advantages[audit_indices]),
        "signed",
        2.0,
        1.0,
        3.0,
    )[0]
    old_gradient = legacy.full_gradient_statistics(
        old_loss,
        old_policy.parameters(),
    )
    new_gradient = full_gradient_statistics(
        new_loss,
        new_policy.parameters(),
    )
    expected = legacy.actor_eval_metrics(
        policy=old_policy,
        obs=observations,
        actions=actions,
        advantages=advantages,
        audit_indices=audit_indices,
        fixed_negative_indices=negative_indices,
        device=torch.device("cpu"),
        loss_value=float(old_loss.detach()),
        gradient_norm=old_gradient["raw"],
        gradient_rms=old_gradient["rms"],
        relative_gradient_norm=(old_gradient["relative_to_parameter_norm"]),
        update_norm=0.25,
        relative_update_norm=0.05,
        step=7,
        boundary_threshold=0.99,
    )
    actual = actor_eval_metrics(
        policy=new_policy,
        observations=observations,
        actions=actions,
        advantages=advantages,
        audit_indices=audit_indices,
        fixed_negative_indices=negative_indices,
        device="cpu",
        loss_value=float(new_loss.detach()),
        gradient_norm=new_gradient["raw"],
        gradient_rms=new_gradient["rms"],
        relative_gradient_norm=(new_gradient["relative_to_parameter_norm"]),
        update_norm=0.25,
        relative_update_norm=0.05,
        step=7,
        boundary_threshold=0.99,
    )
    _assert_nested_close(actual, expected)


def _legacy_config(protocol: HopperProtocol) -> SimpleNamespace:
    return SimpleNamespace(
        actor_lr=protocol.actor_learning_rate,
        weight_decay=protocol.weight_decay,
        actor_batch_size=protocol.actor_batch_size,
        max_gradient_norm=protocol.max_gradient_norm,
        support_boundary_threshold=(protocol.support_boundary_threshold),
        audit_windows=protocol.audit_windows,
        actor_state_drift_tolerance=(protocol.actor_state_drift_tolerance),
        actor_update_tolerance=protocol.actor_update_tolerance,
        support_boundary_fraction=(protocol.support_boundary_fraction),
        task_return_drop_threshold=(protocol.task_return_drop_threshold),
    )


@pytest.mark.parametrize(
    "method",
    ["positive_only", "dynamic_budget_matched_global"],
)
def test_fixed_seed_short_actor_trajectory_matches(
    tmp_path: Path,
    method: str,
) -> None:
    observations, actions, advantages = _arrays()
    protocol = replace(
        HopperProtocol(),
        execution_profile="smoke",
        formal_seeds=(17,),
        hidden_sizes=(8,),
        actor_batch_size=8,
        audit_windows=2,
        actor_state_drift_tolerance=1.0e9,
        actor_update_tolerance=1.0e9,
    )
    old_policy = _old_policy(seed=23)
    new_policy = _new_policy(seed=23)
    far_threshold, far_cap_score = _thresholds(
        new_policy,
        observations,
        actions,
    )
    train_indices = np.arange(24, dtype=np.int64)
    audit_indices = np.arange(16, dtype=np.int64)
    negative_indices = np.flatnonzero(advantages < 0)[:8]

    expected_policy, expected_audit = legacy.train_actor_stage(
        policy=old_policy,
        method=method,
        obs=observations,
        actions=actions,
        advantages=advantages,
        train_indices=train_indices,
        audit_indices=audit_indices,
        fixed_negative_indices=negative_indices,
        config=_legacy_config(protocol),
        min_steps=2,
        max_steps=4,
        eval_interval=2,
        seed=17,
        device=torch.device("cpu"),
        output_dir=tmp_path / "legacy",
        far_threshold=far_threshold,
        global_scale=0.5,
        far_cap_score=far_cap_score,
    )
    actual_policy, actual_audit = train_actor_stage(
        policy=new_policy,
        method=method,
        observations=observations,
        actions=actions,
        advantages=advantages,
        train_indices=train_indices,
        audit_indices=audit_indices,
        fixed_negative_indices=negative_indices,
        protocol=protocol,
        min_steps=2,
        max_steps=4,
        eval_interval=2,
        seed=17,
        device="cpu",
        output_dir=tmp_path / "reference",
        far_threshold=far_threshold,
        global_scale=0.5,
        far_cap_score=far_cap_score,
    )
    for name, expected in expected_policy.state_dict().items():
        torch.testing.assert_close(
            actual_policy.state_dict()[name],
            expected,
            rtol=1.0e-7,
            atol=1.0e-8,
        )
    assert set(actual_audit) == set(expected_audit)
    for key, value in expected_audit.items():
        if key != "checkpoint":
            _assert_nested_close(actual_audit[key], value)

    with (tmp_path / "legacy" / "curves.csv").open() as handle:
        expected_rows = list(csv.DictReader(handle))
    with (tmp_path / "reference" / "curves.csv").open() as handle:
        actual_rows = list(csv.DictReader(handle))
    assert actual_rows == expected_rows


def test_nonfinite_loss_does_not_apply_optimizer_update(
    tmp_path: Path,
) -> None:
    observations, actions, advantages = _arrays()
    advantages[0] = np.nan
    policy = _new_policy(seed=31)
    initial = {name: value.detach().clone() for name, value in policy.state_dict().items()}
    protocol = replace(
        HopperProtocol(),
        execution_profile="smoke",
        formal_seeds=(1,),
        hidden_sizes=(8,),
        actor_batch_size=1,
        audit_windows=2,
    )
    _, audit = train_actor_stage(
        policy=policy,
        method="signed",
        observations=observations,
        actions=actions,
        advantages=advantages,
        train_indices=np.asarray([0], dtype=np.int64),
        audit_indices=np.asarray([0, 1], dtype=np.int64),
        fixed_negative_indices=np.asarray([1], dtype=np.int64),
        protocol=protocol,
        min_steps=1,
        max_steps=1,
        eval_interval=1,
        seed=1,
        device="cpu",
        output_dir=tmp_path / "nonfinite",
    )
    for name, expected in initial.items():
        torch.testing.assert_close(
            policy.state_dict()[name],
            expected,
            rtol=0.0,
            atol=0.0,
        )
    assert audit["fixed_budget_completed"] is False
    assert audit["early_stop_reason"] == "nonfinite_train_loss"
    assert audit["numerical_nonfinite"] is True
    assert audit["terminal_audit_complete"] is True
