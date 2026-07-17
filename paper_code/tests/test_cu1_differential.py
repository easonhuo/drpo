from __future__ import annotations

import torch

from drpo import cu1_core
from drpo import drpo_cu1_e1_e4_oneclick as legacy
from drpo_reference.continuous import cu1 as reference
from drpo_reference.continuous.gaussian import (
    gaussian_output_components as reference_output_components,
)


ATOL = 0.0
RTOL = 0.0


def assert_tensor_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)


def reference_protocol(**overrides: object) -> reference.CU1Protocol:
    values = {
        "state_dim": legacy.P.state_dim,
        "action_dim": legacy.P.action_dim,
        "n_train_states": legacy.P.n_train_states,
        "n_test_states": legacy.P.n_test_states,
        "positive_samples_per_state": legacy.P.positive_samples_per_state,
        "negative_samples_per_state": legacy.P.negative_samples_per_state,
        "gap_to_unseen_optimum": legacy.P.gap_to_unseen_optimum,
        "negative_offset_from_positive": legacy.P.negative_offset_from_positive,
        "positive_contour_radius": legacy.P.positive_contour_radius,
        "negative_contour_radius": legacy.P.negative_contour_radius,
        "reward_width": legacy.P.reward_width,
        "baseline": legacy.P.baseline,
        "positive_angle_1": legacy.P.positive_angle_1,
        "hidden_dim": legacy.P.hidden_dim,
        "hidden_layers": legacy.P.hidden_layers,
        "initial_sigma": legacy.P.initial_sigma,
        "near_far_standardized_threshold": (
            legacy.P.near_far_standardized_threshold
        ),
        "task_failure_retention": legacy.P.task_failure_retention,
        "task_failure_consecutive_evals": (
            legacy.P.task_failure_consecutive_evals
        ),
        "log_sigma_event_boundary": legacy.P.log_sigma_event_boundary,
    }
    values.update(overrides)
    return reference.CU1Protocol(**values)


def small_protocol() -> reference.CU1Protocol:
    return reference_protocol(
        n_train_states=96,
        n_test_states=80,
        hidden_dim=16,
    )


def legacy_small_protocol() -> legacy.Protocol:
    return legacy.Protocol(
        n_train_states=96,
        n_test_states=80,
        hidden_dim=16,
    )


def make_legacy_environment(seed: int) -> cu1_core.Environment:
    return cu1_core.make_environment(
        seed,
        legacy_small_protocol(),
        torch.device("cpu"),
        torch.float32,
    )


def make_actor_pair(
    protocol: reference.CU1Protocol,
) -> tuple[cu1_core.GaussianActor, torch.nn.Module]:
    torch.manual_seed(8123)
    old = cu1_core.GaussianActor(
        state_dim=protocol.state_dim,
        action_dim=protocol.action_dim,
        hidden_dim=protocol.hidden_dim,
        initial_sigma=protocol.initial_sigma,
    )
    torch.manual_seed(8123)
    new = reference.make_actor(protocol)
    for key, value in old.state_dict().items():
        assert_tensor_equal(new.state_dict()[key], value)
    return old, new


def test_environment_identity_all_tensors_and_audit() -> None:
    protocol = small_protocol()
    old_protocol = legacy_small_protocol()
    old = cu1_core.make_environment(
        37,
        old_protocol,
        torch.device("cpu"),
        torch.float32,
    )
    new = reference.make_environment(37, protocol, "cpu", torch.float32)
    for split_name in ("train", "test"):
        old_split = getattr(old, split_name)
        new_split = getattr(new, split_name)
        for field_name in old_split.__dataclass_fields__:
            assert_tensor_equal(
                getattr(new_split, field_name),
                getattr(old_split, field_name),
            )
    assert reference.audit_environment(
        new,
        protocol,
    ) == cu1_core.audit_environment(old, old_protocol)


def test_gaussian_primitives_identity() -> None:
    protocol = small_protocol()
    old, new = make_actor_pair(protocol)
    environment = reference.make_environment(41, protocol)
    states = environment.train.s[:11]
    actions = environment.train.negative_actions[:11]
    old_mu, old_log_std = old(states)
    new_mu, new_log_std = new(states)
    assert_tensor_equal(new_mu, old_mu)
    assert_tensor_equal(new_log_std, old_log_std)
    assert_tensor_equal(
        reference.gaussian_log_prob(
            new_mu,
            new_log_std,
            actions,
            protocol.action_dim,
        ),
        cu1_core.gaussian_log_prob(
            old_mu,
            old_log_std,
            actions,
            protocol.action_dim,
        ),
    )
    assert_tensor_equal(
        reference.standardized_distance(new_mu, new_log_std, actions),
        cu1_core.standardized_distance(old_mu, old_log_std, actions),
    )
    old_components = cu1_core.gaussian_output_components(
        old_mu,
        old_log_std,
        actions,
        protocol.action_dim,
    )
    new_components = reference_output_components(
        new_mu,
        new_log_std,
        actions,
        protocol.action_dim,
    )
    assert old_components.keys() == new_components.keys()
    for key in old_components:
        assert_tensor_equal(new_components[key], old_components[key])


def old_loss(
    kind: str,
    actor: torch.nn.Module,
    split: cu1_core.Split,
    ids: torch.Tensor,
    fixed_sigma: float | None,
) -> torch.Tensor:
    old_protocol = legacy.P
    legacy.P = legacy_small_protocol()
    try:
        return getattr(legacy, kind)(actor, split, ids, fixed_sigma)
    finally:
        legacy.P = old_protocol


def new_loss(
    kind: str,
    actor: torch.nn.Module,
    split: reference.Split,
    protocol: reference.CU1Protocol,
    ids: torch.Tensor,
    fixed_sigma: float | None,
) -> torch.Tensor:
    return getattr(reference, kind)(
        actor,
        split,
        protocol,
        ids,
        fixed_sigma,
    )


def test_losses_and_raw_gradients_identity() -> None:
    protocol = small_protocol()
    old_environment = make_legacy_environment(53)
    new_environment = reference.make_environment(53, protocol)
    ids = torch.tensor([0, 2, 7, 11, 31, 45, 80], dtype=torch.long)
    for fixed_sigma in (None, 0.6):
        for kind in (
            "positive_loss",
            "local_negative_loss",
            "all_negative_loss",
        ):
            old_actor, new_actor = make_actor_pair(protocol)
            legacy_loss = old_loss(
                kind,
                old_actor,
                old_environment.train,
                ids,
                fixed_sigma,
            )
            migrated_loss = new_loss(
                kind,
                new_actor,
                new_environment.train,
                protocol,
                ids,
                fixed_sigma,
            )
            assert_tensor_equal(migrated_loss, legacy_loss)
            old_parameters = (
                old_actor.mean_parameters()
                if fixed_sigma is not None
                else old_actor.all_parameters()
            )
            new_parameters = (
                new_actor.mean_parameters()
                if fixed_sigma is not None
                else new_actor.all_parameters()
            )
            old_gradients = torch.autograd.grad(
                legacy_loss,
                old_parameters,
                allow_unused=True,
            )
            new_gradients = torch.autograd.grad(
                migrated_loss,
                new_parameters,
                allow_unused=True,
            )
            assert len(old_gradients) == len(new_gradients)
            for actual, expected in zip(new_gradients, old_gradients):
                if expected is None:
                    assert actual is None
                else:
                    assert actual is not None
                    assert_tensor_equal(actual, expected)


def test_near_far_losses_and_diagnostics_identity() -> None:
    protocol = small_protocol()
    old_environment = make_legacy_environment(61)
    new_environment = reference.make_environment(61, protocol)
    ids = torch.arange(0, 64, 3)
    old_actor, new_actor = make_actor_pair(protocol)
    old_protocol = legacy.P
    legacy.P = legacy_small_protocol()
    try:
        old_near, old_far, old_diagnostics = legacy.near_far_losses(
            old_actor,
            old_environment.train,
            ids,
            None,
        )
    finally:
        legacy.P = old_protocol
    new_near, new_far, new_diagnostics = reference.near_far_losses(
        new_actor,
        new_environment.train,
        protocol,
        ids,
        None,
    )
    assert_tensor_equal(new_near, old_near)
    assert_tensor_equal(new_far, old_far)
    assert new_diagnostics == old_diagnostics


def test_first_adam_update_identity() -> None:
    protocol = small_protocol()
    old_environment = make_legacy_environment(71)
    new_environment = reference.make_environment(71, protocol)
    ids = torch.tensor([1, 4, 9, 16, 25, 36, 49, 64], dtype=torch.long)
    old_actor, new_actor = make_actor_pair(protocol)
    optimizer_kwargs = {
        "lr": legacy.P.positive_adam_lr,
        "betas": (legacy.P.adam_beta1, legacy.P.adam_beta2),
        "eps": legacy.P.adam_eps,
    }
    old_optimizer = torch.optim.Adam(old_actor.parameters(), **optimizer_kwargs)
    new_optimizer = torch.optim.Adam(new_actor.parameters(), **optimizer_kwargs)
    legacy_loss = old_loss(
        "positive_loss",
        old_actor,
        old_environment.train,
        ids,
        None,
    )
    migrated_loss = reference.positive_loss(
        new_actor,
        new_environment.train,
        protocol,
        ids,
    )
    old_optimizer.zero_grad(set_to_none=True)
    new_optimizer.zero_grad(set_to_none=True)
    legacy_loss.backward()
    migrated_loss.backward()
    old_optimizer.step()
    new_optimizer.step()
    for key, value in old_actor.state_dict().items():
        assert_tensor_equal(new_actor.state_dict()[key], value)


def test_fixed_seed_short_positive_trajectory_identity() -> None:
    protocol = small_protocol()
    old_environment = make_legacy_environment(83)
    new_environment = reference.make_environment(83, protocol)
    old_actor, new_actor = make_actor_pair(protocol)
    optimizer_kwargs = {
        "lr": 1e-3,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
    }
    old_optimizer = torch.optim.Adam(old_actor.parameters(), **optimizer_kwargs)
    new_optimizer = torch.optim.Adam(new_actor.parameters(), **optimizer_kwargs)
    old_generator = torch.Generator(device="cpu").manual_seed(90083)
    new_generator = torch.Generator(device="cpu").manual_seed(90083)
    old_losses: list[float] = []
    new_losses: list[float] = []
    for _ in range(12):
        old_ids = torch.randint(
            0,
            protocol.n_train_states,
            (32,),
            generator=old_generator,
        )
        new_ids = torch.randint(
            0,
            protocol.n_train_states,
            (32,),
            generator=new_generator,
        )
        assert_tensor_equal(new_ids, old_ids)
        legacy_loss = old_loss(
            "positive_loss",
            old_actor,
            old_environment.train,
            old_ids,
            None,
        )
        migrated_loss = reference.positive_loss(
            new_actor,
            new_environment.train,
            protocol,
            new_ids,
        )
        old_optimizer.zero_grad(set_to_none=True)
        new_optimizer.zero_grad(set_to_none=True)
        legacy_loss.backward()
        migrated_loss.backward()
        old_optimizer.step()
        new_optimizer.step()
        old_losses.append(float(legacy_loss.detach()))
        new_losses.append(float(migrated_loss.detach()))
    assert new_losses == old_losses
    for key, value in old_actor.state_dict().items():
        assert_tensor_equal(new_actor.state_dict()[key], value)

    old_protocol = legacy.P
    legacy.P = legacy_small_protocol()
    try:
        old_evaluation = legacy.evaluation(old_actor, old_environment.test)
    finally:
        legacy.P = old_protocol
    new_evaluation = reference.evaluation(
        new_actor,
        new_environment.test,
        protocol,
    )
    assert new_evaluation == old_evaluation


def test_event_flags_do_not_collapse_distinct_failure_classes() -> None:
    support = {
        "log_sigma_output_finite_all_states": False,
        "sigma_output_finite_all_states": False,
        "support_contraction_boundary": True,
        "unexpected_support_expansion_boundary": False,
    }
    flags = reference.event_flags(
        task_performance_collapse=True,
        support=support,
        finite_parameters=False,
    )
    assert flags.task_performance_collapse
    assert flags.support_or_probability_boundary
    assert flags.nan_inf_numerical_failure
    assert not flags.environment_invalid
