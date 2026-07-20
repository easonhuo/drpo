from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from drpo import e7_hopper_q2 as legacy
from drpo_reference.external.hopper_actor import train_actor_stage
from drpo_reference.external.hopper_models import SquashedGaussianPolicy
from drpo_reference.external.hopper_protocol import METHODS, HopperProtocol
from drpo_reference.external.hopper_suite import (
    clone_policy,
    make_policy,
    run_hopper_six_branch_suite,
)


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


def _arrays() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    generator = np.random.default_rng(81)
    observations = generator.normal(size=(96, 5)).astype(np.float32)
    actions = np.tanh(generator.normal(size=(96, 3))).astype(np.float32)
    advantages = np.empty(96, dtype=np.float32)
    negative_magnitudes = np.tile(
        np.linspace(0.5, 2.0, 16, dtype=np.float32),
        2,
    )
    advantages[:32] = -negative_magnitudes
    advantages[32:64] = np.linspace(
        0.5,
        2.0,
        32,
        dtype=np.float32,
    )
    advantages[64:80] = -np.linspace(
        0.5,
        2.0,
        16,
        dtype=np.float32,
    )
    advantages[80:] = np.linspace(
        0.5,
        2.0,
        16,
        dtype=np.float32,
    )
    return (
        observations,
        actions,
        advantages,
        np.arange(64, dtype=np.int64),
        np.arange(64, 96, dtype=np.int64),
    )


def _protocol() -> HopperProtocol:
    return replace(
        HopperProtocol(),
        execution_profile="smoke",
        formal_seeds=(17,),
        hidden_sizes=(8,),
        actor_batch_size=8,
        positive_min_steps=2,
        positive_steps=4,
        actor_eval_interval=2,
        branch_min_steps=2,
        branch_steps=4,
        matched_pairs=4,
        audit_sample_size=16,
        gradient_probe_pairs=2,
        distance_bins=2,
        advantage_bins=4,
        advantage_match_relative_tolerance=1.0,
        global_budget_audit_size=8,
        audit_windows=2,
        actor_state_drift_tolerance=1.0e9,
        actor_update_tolerance=1.0e9,
    )


def _legacy_config(protocol: HopperProtocol) -> SimpleNamespace:
    return SimpleNamespace(
        hidden_sizes=protocol.hidden_sizes,
        log_std_min=protocol.log_std_min,
        log_std_max=protocol.log_std_max,
        action_clip_epsilon=protocol.action_clip_epsilon,
        activation=protocol.activation,
        init_scheme=protocol.init_scheme,
        init_gain=protocol.init_gain,
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
        near_quantile=protocol.near_quantile,
        far_quantile=protocol.far_quantile,
        advantage_bins=protocol.advantage_bins,
        advantage_match_relative_tolerance=(protocol.advantage_match_relative_tolerance),
        gradient_probe_pairs=protocol.gradient_probe_pairs,
        distance_bins=protocol.distance_bins,
        far_cap_reference_quantile=(protocol.far_cap_reference_quantile),
        global_budget_audit_size=(protocol.global_budget_audit_size),
    )


def _legacy_suite(
    *,
    observations: np.ndarray,
    actions: np.ndarray,
    advantages: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    protocol: HopperProtocol,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    device = torch.device("cpu")
    config = _legacy_config(protocol)
    legacy.seed_everything(seed)
    positive_train = train_indices[advantages[train_indices] > 0]
    negative_train = train_indices[advantages[train_indices] < 0]
    validation_positive = validation_indices[advantages[validation_indices] > 0]
    validation_negative = validation_indices[advantages[validation_indices] < 0]
    generator = np.random.default_rng(seed + 321)
    half = max(1, protocol.audit_sample_size // 2)
    audit_positive = generator.choice(
        validation_positive,
        size=min(half, len(validation_positive)),
        replace=False,
    )
    audit_negative = generator.choice(
        validation_negative,
        size=min(half, len(validation_negative)),
        replace=False,
    )
    audit_indices = np.concatenate([audit_positive, audit_negative]).astype(np.int64)
    generator.shuffle(audit_indices)
    fixed_negative_indices = generator.choice(
        negative_train,
        size=min(protocol.audit_sample_size, len(negative_train)),
        replace=False,
    )

    policy = legacy.SquashedGaussianPolicy(
        observations.shape[1],
        actions.shape[1],
        protocol.hidden_sizes,
        protocol.log_std_min,
        protocol.log_std_max,
        protocol.action_clip_epsilon,
        protocol.activation,
        protocol.init_scheme,
        protocol.init_gain,
    ).to(device)
    policy, positive_audit = legacy.train_actor_stage(
        policy=policy,
        method="positive_only",
        obs=observations,
        actions=actions,
        advantages=advantages,
        train_indices=positive_train,
        audit_indices=audit_indices,
        fixed_negative_indices=fixed_negative_indices,
        config=config,
        min_steps=protocol.positive_min_steps,
        max_steps=protocol.positive_steps,
        eval_interval=protocol.actor_eval_interval,
        seed=seed + 500_000,
        device=device,
        output_dir=output_dir / "positive_only_initialization",
    )

    distances = np.full(len(advantages), np.nan, dtype=np.float32)
    with torch.no_grad():
        distances[negative_train] = (
            policy.standardized_distance(
                torch.as_tensor(observations[negative_train]),
                torch.as_tensor(actions[negative_train]),
            )
            .cpu()
            .numpy()
        )
    near_indices, far_indices, matching = legacy.match_near_far_indices(
        advantages,
        distances,
        negative_train,
        protocol.near_quantile,
        protocol.far_quantile,
        protocol.advantage_bins,
        protocol.matched_pairs,
        protocol.advantage_match_relative_tolerance,
        seed,
    )
    gradient_probe = legacy.create_gradient_probe(
        policy=policy,
        obs=observations,
        actions=actions,
        advantages=advantages,
        near_indices=near_indices,
        far_indices=far_indices,
        population_indices=fixed_negative_indices,
        max_gradient_pairs=min(
            protocol.gradient_probe_pairs,
            len(near_indices),
        ),
        distance_bins=protocol.distance_bins,
        device=device,
        output_dir=output_dir / "probes",
    )
    far_threshold = float((matching["near_cut"] + matching["far_cut"]) / 2.0)
    near_pool = negative_train[distances[negative_train] <= matching["near_cut"]]
    with torch.no_grad():
        near_scores = (
            policy.output_score_norm(
                torch.as_tensor(observations[near_pool]),
                torch.as_tensor(actions[near_pool]),
            )
            .cpu()
            .numpy()
        )
    far_cap_score = float(
        np.quantile(
            near_scores,
            protocol.far_cap_reference_quantile,
        )
    )
    budget = legacy.resolve_global_scale(
        policy=policy,
        obs=observations,
        actions=actions,
        advantages=advantages,
        negative_indices=negative_train,
        far_threshold=far_threshold,
        far_cap_score=far_cap_score,
        audit_size=protocol.global_budget_audit_size,
        seed=seed,
        device=device,
    )
    branch_audits: dict[str, Any] = {}
    branch_states: dict[str, dict[str, torch.Tensor]] = {}
    for method in METHODS:
        branch = legacy.copy_policy(
            policy,
            config,
            observations.shape[1],
            actions.shape[1],
            device,
        )
        branch, audit = legacy.train_actor_stage(
            policy=branch,
            method=method,
            obs=observations,
            actions=actions,
            advantages=advantages,
            train_indices=train_indices,
            audit_indices=audit_indices,
            fixed_negative_indices=fixed_negative_indices,
            config=config,
            min_steps=protocol.branch_min_steps,
            max_steps=protocol.branch_steps,
            eval_interval=protocol.actor_eval_interval,
            seed=seed,
            device=device,
            output_dir=output_dir / "methods" / method,
            far_threshold=far_threshold,
            global_scale=float(budget["global_scale"]),
            far_cap_score=far_cap_score,
        )
        branch_audits[method] = audit
        branch_states[method] = {
            name: value.detach().clone() for name, value in branch.state_dict().items()
        }
    return {
        "positive_audit": positive_audit,
        "audit_indices": audit_indices,
        "fixed_negative_indices": fixed_negative_indices,
        "near_indices": near_indices,
        "far_indices": far_indices,
        "matching": matching,
        "gradient_probe": gradient_probe,
        "far_threshold": far_threshold,
        "far_cap_score": far_cap_score,
        "budget": budget,
        "branch_audits": branch_audits,
        "branch_states": branch_states,
    }


def _checkpoint_model(path: Path) -> dict[str, torch.Tensor]:
    return torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )["model"]


def test_protocol_carries_registered_actor_minimum_steps() -> None:
    formal = HopperProtocol()
    assert formal.positive_min_steps == 10_000
    assert formal.branch_min_steps == 10_000
    smoke = replace(
        formal,
        execution_profile="smoke",
        formal_seeds=(1,),
        positive_min_steps=2,
        positive_steps=4,
        branch_min_steps=2,
        branch_steps=4,
    )
    assert smoke.positive_min_steps <= smoke.positive_steps
    assert smoke.branch_min_steps <= smoke.branch_steps


def test_preparation_and_six_branches_match_legacy(
    tmp_path: Path,
) -> None:
    (
        observations,
        actions,
        advantages,
        train_indices,
        validation_indices,
    ) = _arrays()
    protocol = _protocol()
    expected = _legacy_suite(
        observations=observations,
        actions=actions,
        advantages=advantages,
        train_indices=train_indices,
        validation_indices=validation_indices,
        protocol=protocol,
        seed=17,
        output_dir=tmp_path / "legacy",
    )
    actual = run_hopper_six_branch_suite(
        observations=observations,
        actions=actions,
        advantages=advantages,
        train_indices=train_indices,
        validation_indices=validation_indices,
        protocol=protocol,
        seed=17,
        device="cpu",
        output_dir=tmp_path / "reference",
    )

    assert actual["method_order"] == list(METHODS)
    assert actual["positive_training_seed"] == 500_017
    assert actual["branch_training_seed"] == 17
    assert actual["all_methods_completed"] is True
    assert actual["all_branch_initial_states_identical"] is True
    assert actual["prepared_checkpoint"]["reload_identity"] is True
    _assert_nested_close(
        actual["matching"],
        expected["matching"],
    )
    _assert_nested_close(
        actual["gradient_probe"],
        expected["gradient_probe"],
    )
    _assert_nested_close(
        actual["global_budget"],
        expected["budget"],
    )
    assert actual["far_threshold"] == pytest.approx(expected["far_threshold"])
    assert actual["far_cap_score"] == pytest.approx(expected["far_cap_score"])

    manifest = json.loads((tmp_path / "reference" / "prepared_actor_manifest.json").read_text())
    assert manifest["audit_indices"] == expected["audit_indices"].tolist()
    assert manifest["fixed_negative_indices"] == expected["fixed_negative_indices"].tolist()
    assert manifest["near_indices"] == expected["near_indices"].tolist()
    assert manifest["far_indices"] == expected["far_indices"].tolist()

    for method in METHODS:
        for key, value in expected["branch_audits"][method].items():
            if key != "checkpoint":
                _assert_nested_close(actual["methods"][method][key], value)
        reference_state = _checkpoint_model(
            tmp_path / "reference" / "methods" / method / "terminal_actor.pt"
        )
        for name, value in expected["branch_states"][method].items():
            torch.testing.assert_close(
                reference_state[name],
                value,
                rtol=1.0e-7,
                atol=1.0e-8,
            )


def test_clone_policy_has_equal_values_without_shared_storage() -> None:
    protocol = _protocol()
    torch.manual_seed(7)
    policy = make_policy(protocol, 5, 3, "cpu")
    clone = clone_policy(policy, protocol, 5, 3, "cpu")
    for original, copied in zip(policy.parameters(), clone.parameters()):
        torch.testing.assert_close(original, copied)
        assert original.data_ptr() != copied.data_ptr()


def test_branch_failure_is_isolated_and_other_methods_complete(
    tmp_path: Path,
) -> None:
    (
        observations,
        actions,
        advantages,
        train_indices,
        validation_indices,
    ) = _arrays()
    protocol = _protocol()

    def fail_one_branch(
        **kwargs: Any,
    ) -> tuple[
        SquashedGaussianPolicy,
        dict[str, Any],
    ]:
        if kwargs["method"] == "near_zero" and kwargs["max_steps"] == protocol.branch_steps:
            raise RuntimeError("injected branch failure")
        return train_actor_stage(**kwargs)

    summary = run_hopper_six_branch_suite(
        observations=observations,
        actions=actions,
        advantages=advantages,
        train_indices=train_indices,
        validation_indices=validation_indices,
        protocol=protocol,
        seed=17,
        device="cpu",
        output_dir=tmp_path / "isolated",
        stage_runner=fail_one_branch,
    )
    assert summary["suite_status"] == "partial_failure"
    assert summary["failed_methods"] == ["near_zero"]
    assert set(summary["completed_methods"]) == set(METHODS) - {"near_zero"}
    assert set(summary["methods"]) == set(METHODS) - {"near_zero"}
    assert summary["branch_failures"]["near_zero"]["failure_isolated"] is True
    assert (tmp_path / "isolated" / "methods" / "near_zero" / "branch_failure.json").is_file()
    for method in set(METHODS) - {"near_zero"}:
        assert (tmp_path / "isolated" / "methods" / method / "terminal_audit.json").is_file()


def test_suite_rejects_a_mixed_output_root(tmp_path: Path) -> None:
    output_dir = tmp_path / "mixed"
    output_dir.mkdir()
    (output_dir / "stale.txt").write_text("stale")
    (
        observations,
        actions,
        advantages,
        train_indices,
        validation_indices,
    ) = _arrays()
    with pytest.raises(FileExistsError, match="new or empty"):
        run_hopper_six_branch_suite(
            observations=observations,
            actions=actions,
            advantages=advantages,
            train_indices=train_indices,
            validation_indices=validation_indices,
            protocol=_protocol(),
            seed=17,
            device="cpu",
            output_dir=output_dir,
        )
