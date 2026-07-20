from __future__ import annotations

import copy
import csv
from pathlib import Path

import pytest
import torch

from drpo import drpo_cu1_e1_e4_oneclick as legacy
from drpo_reference.continuous.cu1 import CU1Protocol, make_actor, make_environment
from drpo_reference.continuous.cu1_mechanism import (
    CU1CausalProtocol,
    CU1SourceProtocol,
    intervention_gradients,
    run_causal_intervention,
    source_diagnostic,
)
from drpo_reference.continuous.cu1_training import CU1PositiveProtocol


def _configure_legacy(
    tmp_path: Path,
    *,
    train_states: int = 32,
    test_states: int = 24,
    hidden_dim: int = 16,
    batch_states: int = 8,
    eval_every: int = 2,
    probe_states: int = 6,
) -> legacy.Protocol:
    protocol = legacy.Protocol(
        n_train_states=train_states,
        n_test_states=test_states,
        hidden_dim=hidden_dim,
        positive_batch_states=batch_states,
        eval_every=eval_every,
        probe_states=probe_states,
        positive_steps=3,
        positive_continuation_steps=2,
        lbfgs_max_iter=1,
        positive_polish_min_steps=1,
        positive_polish_max_steps=1,
        positive_polish_check_every=1,
        e3_fixed_steps=4,
        e3_learn_steps=4,
        e4_local_warm_steps=2,
        e4_local_continuation_steps=1,
        e4_runaway_steps=3,
        e4_control_steps=3,
        e1_e2_seeds=(10,),
        e3_seeds=(30,),
        e4_seeds=(50,),
        variance_robustness_seeds=(5,),
    )
    legacy.P = protocol
    legacy.DEVICE = torch.device("cpu")
    legacy.DTYPE = torch.float32
    legacy.ROOT = tmp_path
    return protocol


def _reference_protocol(protocol: legacy.Protocol) -> CU1Protocol:
    return CU1Protocol(
        state_dim=protocol.state_dim,
        action_dim=protocol.action_dim,
        n_train_states=protocol.n_train_states,
        n_test_states=protocol.n_test_states,
        positive_samples_per_state=protocol.positive_samples_per_state,
        negative_samples_per_state=protocol.negative_samples_per_state,
        gap_to_unseen_optimum=protocol.gap_to_unseen_optimum,
        negative_offset_from_positive=protocol.negative_offset_from_positive,
        positive_contour_radius=protocol.positive_contour_radius,
        negative_contour_radius=protocol.negative_contour_radius,
        reward_width=protocol.reward_width,
        baseline=protocol.baseline,
        positive_angle_1=protocol.positive_angle_1,
        hidden_dim=protocol.hidden_dim,
        hidden_layers=protocol.hidden_layers,
        initial_sigma=protocol.initial_sigma,
        near_far_standardized_threshold=(protocol.near_far_standardized_threshold),
        task_failure_retention=protocol.task_failure_retention,
        task_failure_consecutive_evals=protocol.task_failure_consecutive_evals,
        log_sigma_event_boundary=protocol.log_sigma_event_boundary,
    )


def _matching_actors(protocol: CU1Protocol, seed: int):
    torch.manual_seed(seed)
    old_actor = legacy.GaussianActor().to("cpu")
    torch.manual_seed(seed)
    new_actor = make_actor(protocol).to("cpu")
    for name, old_value in old_actor.state_dict().items():
        torch.testing.assert_close(new_actor.state_dict()[name], old_value, rtol=0.0, atol=0.0)
    return old_actor, new_actor


def _assert_mapping_close(
    actual: dict[str, object],
    expected: dict[str, object],
    *,
    keys: tuple[str, ...] | None = None,
) -> None:
    selected = tuple(expected) if keys is None else keys
    for key in selected:
        expected_value = expected[key]
        actual_value = actual[key]
        if isinstance(expected_value, bool) or expected_value is None:
            assert actual_value == expected_value
        elif isinstance(expected_value, (int, float)):
            assert float(actual_value) == pytest.approx(float(expected_value), rel=1e-6, abs=1e-7)
        else:
            assert actual_value == expected_value


def test_source_diagnostic_matches_authoritative_e1(tmp_path: Path) -> None:
    old_protocol = _configure_legacy(tmp_path)
    protocol = _reference_protocol(old_protocol)
    old_environment = legacy.make_environment(17)
    new_environment = make_environment(17, protocol)
    old_actor, new_actor = _matching_actors(protocol, seed=91)

    expected = legacy.run_e1_seed(17, old_actor, old_environment)
    actual = source_diagnostic(
        seed=17,
        actor=new_actor,
        environment=new_environment,
        protocol=protocol,
        source=CU1SourceProtocol(probe_states=old_protocol.probe_states),
    )
    _assert_mapping_close(actual, expected)


def test_intervention_gradients_match_all_legacy_methods(tmp_path: Path) -> None:
    old_protocol = _configure_legacy(tmp_path)
    protocol = _reference_protocol(old_protocol)
    old_environment = legacy.make_environment(30)
    new_environment = make_environment(30, protocol)
    ids = torch.tensor([0, 3, 7, 11, 15, 19, 23, 27])
    methods = (
        "baseline",
        "near_zero",
        "far_zero",
        "far_cap",
        "global_scale",
        "far_to_near",
    )

    for fixed_sigma in (legacy.analytic_positive_sigma(), None):
        for method in methods:
            old_actor, new_actor = _matching_actors(protocol, seed=3101)
            expected_gradients, expected_diagnostics = legacy.intervention_gradients(
                old_actor,
                old_environment.train,
                ids,
                fixed_sigma,
                0.37,
                method,
                old_protocol.e3_cap_ratio,
            )
            actual_gradients, actual_diagnostics = intervention_gradients(
                new_actor,
                new_environment.train,
                protocol,
                ids,
                fixed_sigma=fixed_sigma,
                alpha=0.37,
                method=method,
                cap_ratio=old_protocol.e3_cap_ratio,
            )
            assert len(actual_gradients) == len(expected_gradients)
            for actual, expected in zip(actual_gradients, expected_gradients):
                if expected is None:
                    assert actual is None
                else:
                    assert actual is not None
                    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
            _assert_mapping_close(actual_diagnostics, expected_diagnostics)


def _read_legacy_trajectory(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_fixed_variance_short_trajectory_matches_legacy(tmp_path: Path) -> None:
    old_protocol = _configure_legacy(tmp_path)
    protocol = _reference_protocol(old_protocol)
    old_environment = legacy.make_environment(30)
    new_environment = make_environment(30, protocol)
    old_actor, _ = _matching_actors(protocol, seed=778)
    initialization = copy.deepcopy(old_actor.state_dict())
    fixed_sigma = legacy.analytic_positive_sigma()

    expected_summary = legacy.run_intervention(
        30,
        initialization,
        old_environment,
        "far_cap",
        fixed_sigma,
        old_protocol.e3_fixed_alpha,
        old_protocol.e3_fixed_lr,
        old_protocol.e3_fixed_steps,
        "fixed_variance",
    )
    actual_run = run_causal_intervention(
        seed=30,
        initialization_state=initialization,
        environment=new_environment,
        protocol=protocol,
        positive_training=CU1PositiveProtocol(
            positive_batch_states=old_protocol.positive_batch_states,
            eval_every=old_protocol.eval_every,
            formal_seeds=(10,),
        ),
        method="far_cap",
        fixed_sigma=fixed_sigma,
        alpha=old_protocol.e3_fixed_alpha,
        learning_rate=old_protocol.e3_fixed_lr,
        steps=old_protocol.e3_fixed_steps,
        branch="fixed_variance",
        causal=CU1CausalProtocol(
            fixed_steps=old_protocol.e3_fixed_steps,
            learnable_steps=old_protocol.e3_learn_steps,
            far_cap_ratio=old_protocol.e3_cap_ratio,
            evaluation_interval=old_protocol.eval_every,
            formal_seeds=(30,),
        ),
    )

    legacy_rows = _read_legacy_trajectory(
        tmp_path / "e3" / "fixed_variance" / "far_cap" / "seed_30_trajectory.csv"
    )
    assert len(actual_run.trajectory) == len(legacy_rows)
    legacy_keys = tuple(legacy_rows[0])
    for actual_row, expected_row in zip(actual_run.trajectory, legacy_rows):
        for key in legacy_keys:
            expected_value = expected_row[key]
            if expected_value == "":
                assert actual_row[key] is None
            elif expected_value in {"True", "False"}:
                assert actual_row[key] is (expected_value == "True")
            elif key in {"method", "optimizer", "support_event_type"}:
                assert str(actual_row[key]) == expected_value
            else:
                assert float(actual_row[key]) == pytest.approx(
                    float(expected_value), rel=1e-6, abs=1e-7
                )
    _assert_mapping_close(
        actual_run.summary,
        expected_summary,
        keys=tuple(expected_summary),
    )
