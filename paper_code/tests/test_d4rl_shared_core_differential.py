from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from drpo_reference.experiments.d4rl import (
    CANONICAL_EXPRANK_BACKEND,
    D4RL9_EXPERIMENT_ID,
    CanonicalD4RLDataset,
    CanonicalExpRankTrainingConfig,
    SNA2CIQLVExpRankAgent,
    canonical_exprank_negative_weights,
    dispatch_d4rl9,
    prepare_canonical_locomotion_dataset,
    resolve_d4rl9_execution,
    reward_norm_locomotion,
    train_canonical_exprank,
)
from drpo_reference.external.d4rl_tasks import (
    D4RL9_TASKS,
    DATASET_TIERS,
    ENVIRONMENTS,
    resolve_d4rl_task,
    validate_d4rl9_matrix,
    validate_dataset_path,
)
from drpo_reference.external.hopper_data import OfflineData
from drpo_reference.external.hopper_protocol import HopperProtocol


def _paths(root: Path) -> dict[str, Path]:
    return {
        task.task_id: root / task.dataset_basename
        for task in D4RL9_TASKS
    }


def _config(*, steps: int = 3) -> CanonicalExpRankTrainingConfig:
    return CanonicalExpRankTrainingConfig(
        steps=steps,
        batch_size=8,
        learning_rate=3.0e-4,
        alpha=0.11,
        tau=0.5,
        temperature=5.0,
        eval_interval=10,
        checkpoint_interval=2,
        checkpoint_last_fraction=1.0,
    )


def _fixed_batch() -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(19)
    observations = torch.randn(16, 5, generator=generator)
    actions = torch.tanh(torch.randn(16, 2, generator=generator))
    rewards = torch.randn(16, generator=generator)
    next_observations = torch.randn(16, 5, generator=generator)
    dones = torch.tensor([False, True] * 8)
    returns = torch.zeros(16)
    return (
        observations,
        actions,
        rewards,
        next_observations,
        dones,
        returns,
    )


def _synthetic_dataset() -> CanonicalD4RLDataset:
    generator = np.random.default_rng(31)
    observations = generator.normal(size=(32, 5)).astype(np.float32)
    actions = np.tanh(
        generator.normal(size=(32, 2))
    ).astype(np.float32)
    rewards = generator.normal(size=32).astype(np.float32)
    next_observations = generator.normal(size=(32, 5)).astype(np.float32)
    terminals = np.zeros(32, dtype=np.bool_)
    terminals[[7, 15, 23, 31]] = True
    timeouts = np.zeros(32, dtype=np.bool_)
    mc_returns = _mc_returns(rewards, terminals, timeouts)
    return CanonicalD4RLDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        terminals=terminals,
        mc_returns=mc_returns,
    )


def _mc_returns(
    rewards: np.ndarray,
    terminals: np.ndarray,
    timeouts: np.ndarray,
) -> np.ndarray:
    output = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        if terminals[index] or timeouts[index]:
            running = 0.0
        running = float(rewards[index]) + 0.99 * running
        output[index] = running
    return output


def _assert_state_close(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
) -> None:
    assert left.keys() == right.keys()
    for name in left:
        torch.testing.assert_close(
            left[name],
            right[name],
            rtol=1.0e-6,
            atol=1.0e-7,
            msg=lambda message, name=name: f"{name}: {message}",
        )


def _load_legacy_module() -> Any:
    import importlib.util
    import sys

    source = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "drpo"
        / "e7_canonical_vendor"
        / "d4rl"
        / "agents.py"
    )
    name = "_d4rl_exprank_differential_oracle"
    spec = importlib.util.spec_from_file_location(name, source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    module.DEVICE = torch.device("cpu")
    module.set_network_preset("default")
    return module


def _legacy_agent(seed: int) -> Any:
    module = _load_legacy_module()
    torch.manual_seed(seed)
    return module.SNA2C_IQLV_ExpRankAgent(
        5,
        2,
        lr=3.0e-4,
        gamma=0.99,
        alpha=0.11,
        tau=0.5,
        T=5.0,
    )


def _migrated_agent(seed: int) -> SNA2CIQLVExpRankAgent:
    torch.manual_seed(seed)
    return SNA2CIQLVExpRankAgent(
        5,
        2,
        learning_rate=3.0e-4,
        gamma=0.99,
        alpha=0.11,
        tau=0.5,
        temperature=5.0,
        device="cpu",
    )


def test_d4rl9_task_matrix_matches_manuscript_order() -> None:
    assert len(D4RL9_TASKS) == 9
    assert tuple(task.task_id for task in D4RL9_TASKS) == tuple(
        f"{environment}-{tier}-v2"
        for tier in DATASET_TIERS
        for environment in ENVIRONMENTS
    )
    assert validate_d4rl9_matrix(D4RL9_TASKS) == D4RL9_TASKS
    with pytest.raises(ValueError, match="exact manuscript matrix"):
        validate_d4rl9_matrix(D4RL9_TASKS[:-1])


def test_reference_scores_and_hopper_identity_match_frozen_protocol() -> None:
    expected = {
        "halfcheetah": (-280.178953, 12135.0),
        "hopper": (-20.272305, 3234.3),
        "walker2d": (1.629008, 4592.3),
    }
    for task in D4RL9_TASKS:
        assert (
            task.normalized_score_reference_min,
            task.normalized_score_reference_max,
        ) == expected[task.environment]

    task = resolve_d4rl_task("hopper-medium-replay-v2")
    protocol = HopperProtocol()
    assert task.dataset_basename == protocol.dataset_basename
    assert task.dataset_sha256 == protocol.dataset_sha256
    assert task.dataset_id == protocol.rollout_dataset_id
    assert task.env_id == protocol.env_id


def test_task_specs_own_backend_independent_rollout_identity() -> None:
    for task in D4RL9_TASKS:
        expected = {
            "backend": "gymnasium_mujoco",
            "dataset_id": task.dataset_id,
            "env_id": task.env_id,
        }
        assert task.rollout_identity() == expected
        assert task.validate_rollout_identity(**expected) == expected


def test_backend_is_selected_and_code_migrated() -> None:
    backend = CANONICAL_EXPRANK_BACKEND
    assert D4RL9_EXPERIMENT_ID == "EXT-H-E7-BENCH-01"
    assert backend.backend_id == "canonical_sna2c_iqlv_exprank"
    assert backend.algorithm_family == "SNA2C_IQLV_ExpRank"
    assert backend.implementation_selected is True
    assert backend.implementation_migrated is True
    assert backend.protocol_status == (
        "selected_backend_code_migrated_protocol_unfrozen"
    )
    assert backend.protocol_frozen is False
    assert backend.formal_task_matrix_eligible is False
    assert backend.mechanism_runner_reusable is False


def test_migrated_network_initialization_and_forward_match_legacy() -> None:
    legacy = _legacy_agent(17)
    migrated = _migrated_agent(17)
    _assert_state_close(
        legacy.actor.state_dict(),
        migrated.actor.state_dict(),
    )
    _assert_state_close(
        legacy.critic.state_dict(),
        migrated.critic.state_dict(),
    )
    observations = _fixed_batch()[0]
    with torch.no_grad():
        legacy_mean, legacy_log_std = legacy.actor(observations)
        migrated_mean, migrated_log_std = migrated.actor(observations)
        legacy_value = legacy.critic(observations)
        migrated_value = migrated.critic(observations)
    torch.testing.assert_close(legacy_mean, migrated_mean)
    torch.testing.assert_close(legacy_log_std, migrated_log_std)
    torch.testing.assert_close(legacy_value, migrated_value)


def test_rank_weights_match_canonical_formula() -> None:
    advantages = torch.tensor([-4.0, -1.0, -3.0, -2.0])
    actual = canonical_exprank_negative_weights(
        advantages,
        alpha=0.11,
        temperature=5.0,
    )
    order = advantages.argsort()
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(4)
    score = 1.0 - ranks.float() / 3.0
    expected = 0.11 * torch.exp(torch.clamp(-5.0 * score, min=-20.0))
    torch.testing.assert_close(actual, expected)


def test_first_adam_update_matches_legacy() -> None:
    legacy = _legacy_agent(23)
    migrated = _migrated_agent(23)
    batch = _fixed_batch()
    legacy_loss = legacy.update(*batch)
    migrated_loss = migrated.update(*batch)
    assert migrated_loss == pytest.approx(
        legacy_loss,
        rel=1.0e-6,
        abs=1.0e-7,
    )
    _assert_state_close(
        legacy.actor.state_dict(),
        migrated.actor.state_dict(),
    )
    _assert_state_close(
        legacy.critic.state_dict(),
        migrated.critic.state_dict(),
    )


def test_short_training_trajectory_matches_legacy(tmp_path: Path) -> None:
    dataset = _synthetic_dataset()
    seed = 29
    legacy = _legacy_agent(seed)
    tensors = {
        "observations": torch.from_numpy(dataset.observations),
        "actions": torch.from_numpy(dataset.actions),
        "rewards": torch.from_numpy(dataset.rewards),
        "next_observations": torch.from_numpy(dataset.next_observations),
        "terminals": torch.from_numpy(dataset.terminals),
        "mc_returns": torch.from_numpy(dataset.mc_returns),
    }
    generator = torch.Generator().manual_seed(seed)
    for _ in range(3):
        indices = torch.randint(
            0,
            dataset.size,
            (8,),
            generator=generator,
        )
        legacy.update(
            tensors["observations"].index_select(0, indices),
            tensors["actions"].index_select(0, indices),
            tensors["rewards"].index_select(0, indices),
            tensors["next_observations"].index_select(0, indices),
            tensors["terminals"].index_select(0, indices),
            tensors["mc_returns"].index_select(0, indices),
        )
    result = train_canonical_exprank(
        dataset=dataset,
        seed=seed,
        config=_config(steps=3),
        output_root=tmp_path / "run",
    )
    migrated = result["agent"]
    _assert_state_close(
        legacy.actor.state_dict(),
        migrated.actor.state_dict(),
    )
    _assert_state_close(
        legacy.critic.state_dict(),
        migrated.critic.state_dict(),
    )
    assert (tmp_path / "run" / "ckpts" / "step_0000002.pt").is_file()
    assert (tmp_path / "run" / "COMPLETED.json").is_file()


def test_dataset_preparation_matches_canonical_transformations() -> None:
    observations = np.arange(30, dtype=np.float32).reshape(10, 3)
    actions = np.linspace(-2.0, 2.0, 20, dtype=np.float32).reshape(10, 2)
    rewards = np.arange(1, 11, dtype=np.float32)
    terminals = np.zeros(10, dtype=np.bool_)
    terminals[[3, 9]] = True
    timeouts = np.zeros(10, dtype=np.bool_)
    timeouts[6] = True
    next_observations = np.concatenate(
        [observations[1:], observations[-1:]],
        axis=0,
    )
    data = OfflineData(
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        terminals=terminals,
        timeouts=timeouts,
        episode_ids=np.arange(10, dtype=np.int64),
    )
    prepared = prepare_canonical_locomotion_dataset(data)
    expected_rewards = reward_norm_locomotion(
        rewards,
        terminals,
        timeouts,
    ).astype(np.float32)
    np.testing.assert_allclose(prepared.rewards, expected_rewards)
    assert np.max(np.abs(prepared.actions)) <= 1.0 - 1.0e-5
    np.testing.assert_array_equal(
        prepared.next_observations,
        next_observations,
    )


def test_unverified_dataset_hashes_fail_closed(tmp_path: Path) -> None:
    task = resolve_d4rl_task("halfcheetah-medium-v2")
    path = tmp_path / task.dataset_basename
    path.write_bytes(b"not-a-registered-dataset")
    nonformal = validate_dataset_path(
        path,
        task,
        require_verified_sha=False,
    )
    assert nonformal["identity_verified"] is False
    with pytest.raises(RuntimeError, match="formal execution is blocked"):
        validate_dataset_path(
            path,
            task,
            require_verified_sha=True,
        )


def test_execution_plan_exposes_remaining_formal_blockers(
    tmp_path: Path,
) -> None:
    plan = resolve_d4rl9_execution(
        dataset_paths=_paths(tmp_path),
        seeds=tuple(range(10)),
    )
    assert plan.formal_evidence_eligible is False
    assert plan.backend_protocol_complete is False
    assert (
        "d4rl9_performance_backend_protocol_not_frozen"
        in plan.blocked_reasons
    )
    assert "d4rl9_performance_backend_not_migrated" not in plan.blocked_reasons
    manifest = plan.as_manifest()
    assert manifest["single_migrated_trainer_across_d4rl9_tasks"] is True
    assert manifest["shared_training_engine_with_hopper_mechanism"] is False
    assert manifest["separate_per_task_trainers_allowed"] is False


def test_dispatch_uses_one_backend_runner_for_every_task(tmp_path: Path) -> None:
    plan = resolve_d4rl9_execution(
        dataset_paths=_paths(tmp_path / "data"),
        seeds=(1,),
        smoke=True,
    )
    calls: list[dict[str, Any]] = []

    def backend_runner(**kwargs: Any) -> dict[str, object]:
        calls.append(kwargs)
        return {
            "task_id": kwargs["task"].task_id,
            "formal_result_claim": False,
        }

    result = dispatch_d4rl9(
        plan=plan,
        output_root=tmp_path / "output",
        task_runner=backend_runner,
        allow_non_evidence=True,
    )
    assert len(calls) == 9
    assert all(
        call["backend"] is CANONICAL_EXPRANK_BACKEND
        for call in calls
    )
    assert result["single_migrated_trainer_across_d4rl9_tasks"] is True
    assert result["shared_training_engine_with_hopper_mechanism"] is False
    with pytest.raises(RuntimeError, match="dispatch is blocked"):
        dispatch_d4rl9(
            plan=plan,
            output_root=tmp_path / "blocked",
            task_runner=backend_runner,
        )
