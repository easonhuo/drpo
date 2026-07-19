from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest
import torch

from drpo_reference.experiments.d4rl import (
    CANONICAL_EXPRANK_BACKEND,
    D4RL9_EXPERIMENT_ID,
    CanonicalExpRankRunConfig,
    build_canonical_exprank_command,
    canonical_backend_provenance,
    dispatch_d4rl9,
    load_canonical_exprank_module,
    resolve_d4rl9_execution,
    run_canonical_exprank_task,
)
from drpo_reference.external.d4rl_tasks import (
    D4RL9_TASKS,
    DATASET_TIERS,
    ENVIRONMENTS,
    resolve_d4rl_task,
    validate_d4rl9_matrix,
    validate_dataset_path,
)
from drpo_reference.external.hopper_protocol import HopperProtocol


def _paths(root: Path) -> dict[str, Path]:
    return {
        task.task_id: root / task.dataset_basename
        for task in D4RL9_TASKS
    }


def _config() -> CanonicalExpRankRunConfig:
    return CanonicalExpRankRunConfig(
        steps=1_000_000,
        batch_size=256,
        learning_rate=3.0e-4,
        alpha=0.11,
        tau=0.5,
        temperature=5.0,
        eval_interval=50_000,
        eval_episodes=10,
        checkpoint_interval=50_000,
        checkpoint_last_fraction=0.1,
    )


def _flag_map(command: list[str]) -> dict[str, str]:
    arguments = command[2:]
    assert len(arguments) % 2 == 0
    return {
        arguments[index]: arguments[index + 1]
        for index in range(0, len(arguments), 2)
    }


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
    assert (
        task.normalized_score_reference_min
        == protocol.normalized_score_reference_min
    )
    assert (
        task.normalized_score_reference_max
        == protocol.normalized_score_reference_max
    )


def test_task_specs_own_backend_independent_rollout_identity() -> None:
    for task in D4RL9_TASKS:
        expected = {
            "backend": "gymnasium_mujoco",
            "dataset_id": task.dataset_id,
            "env_id": task.env_id,
        }
        assert task.rollout_identity() == expected
        assert task.validate_rollout_identity(**expected) == expected

    task = resolve_d4rl_task("walker2d-medium-replay-v2")
    with pytest.raises(ValueError, match="rollout identity mismatch"):
        task.validate_rollout_identity(
            backend="gymnasium_mujoco",
            dataset_id=task.dataset_id,
            env_id="Hopper-v4",
        )


def test_backend_is_selected_and_attached_without_a_second_trainer() -> None:
    backend = CANONICAL_EXPRANK_BACKEND
    assert D4RL9_EXPERIMENT_ID == "EXT-H-E7-BENCH-01"
    assert backend.backend_id == "canonical_sna2c_iqlv_exprank"
    assert backend.algorithm_family == "SNA2C_IQLV_ExpRank"
    assert backend.implementation_selected is True
    assert backend.implementation_migrated is True
    assert backend.protocol_status == (
        "selected_backend_adapter_migrated_protocol_unfrozen"
    )
    assert backend.protocol_frozen is False
    assert backend.formal_task_matrix_eligible is False
    assert backend.mechanism_runner_reusable is False
    assert "actor_likelihood_contract" in backend.distinct_contracts
    assert "advantage_lifecycle" in backend.distinct_contracts


def test_canonical_source_loader_exposes_exact_exprank_agent() -> None:
    module = load_canonical_exprank_module()
    module.DEVICE = torch.device("cpu")
    module.set_network_preset("default")
    agent_class = module.SNA2C_IQLV_ExpRankAgent
    assert agent_class.__name__ == "SNA2C_IQLV_ExpRankAgent"
    torch.manual_seed(17)
    agent = agent_class(
        5,
        2,
        lr=3.0e-4,
        gamma=0.99,
        alpha=0.11,
        tau=0.5,
        T=5.0,
    )
    generator = torch.Generator().manual_seed(19)
    observations = torch.randn(8, 5, generator=generator)
    actions = torch.tanh(torch.randn(8, 2, generator=generator))
    rewards = torch.randn(8, generator=generator)
    next_observations = torch.randn(8, 5, generator=generator)
    dones = torch.tensor([False, True] * 4)
    returns = torch.zeros(8)
    loss = agent.update(
        observations,
        actions,
        rewards,
        next_observations,
        dones,
        returns,
    )
    assert math.isfinite(loss)


def test_backend_provenance_fingerprints_all_authoritative_sources() -> None:
    provenance = canonical_backend_provenance()
    assert provenance["backend"]["backend_id"] == (
        "canonical_sna2c_iqlv_exprank"
    )
    files = provenance["source_files"]
    assert set(files) == set(CANONICAL_EXPRANK_BACKEND.source_paths)
    assert all(len(item["sha256"]) == 64 for item in files.values())
    assert all(item["size_bytes"] > 0 for item in files.values())


def test_command_matches_canonical_exprank_trainer_contract(tmp_path: Path) -> None:
    task = resolve_d4rl_task("halfcheetah-medium-v2")
    dataset = tmp_path / task.dataset_basename
    command = build_canonical_exprank_command(
        task=task,
        dataset_path=dataset,
        output_root=tmp_path / "output",
        seed=200,
        config=_config(),
    )
    assert command[1].endswith("train_sna2c_variant.py")
    flags = _flag_map(command)
    assert flags == {
        "--dataset": task.dataset_id,
        "--hdf5": str(dataset.resolve()),
        "--variant": "iqlv_exp_rank",
        "--alpha": "0.11",
        "--tau": "0.5",
        "--temp": "5.0",
        "--steps": "1000000",
        "--batch": "256",
        "--lr": "0.0003",
        "--eval_interval": "50000",
        "--eval_episodes": "10",
        "--seed": "200",
        "--out_dir": str((tmp_path / "output").resolve()),
        "--ckpt_dir": str((tmp_path / "output" / "ckpts").resolve()),
        "--ckpt_interval": "50000",
        "--last_pct": "0.1",
    }


def test_every_task_uses_the_same_canonical_trainer(tmp_path: Path) -> None:
    trainer_paths = {
        build_canonical_exprank_command(
            task=task,
            dataset_path=tmp_path / task.dataset_basename,
            output_root=tmp_path / task.task_id,
            seed=200,
            config=_config(),
        )[1]
        for task in D4RL9_TASKS
    }
    assert len(trainer_paths) == 1


def test_plan_only_task_adapter_writes_nonformal_records(tmp_path: Path) -> None:
    task = resolve_d4rl_task("halfcheetah-medium-v2")
    dataset = tmp_path / task.dataset_basename
    dataset.write_bytes(b"non-formal identity fixture")
    result = run_canonical_exprank_task(
        task=task,
        backend=CANONICAL_EXPRANK_BACKEND,
        dataset_path=dataset,
        output_root=tmp_path / "task",
        seeds=(200, 201),
        config=_config(),
        execute=False,
    )
    assert set(result["runs"]) == {"200", "201"}
    assert all(
        run["execution_kind"] == "plan_only"
        for run in result["runs"].values()
    )
    assert result["formal_result_claim"] is False
    assert result["method_ranking_claim_allowed"] is False
    assert (tmp_path / "task" / "TASK_RESULT.json").is_file()


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
    assert nonformal["registered_sha256"] is None
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
    assert plan.method_ranking_claim_allowed is False
    assert plan.dataset_identity_complete is False
    assert plan.backend_protocol_complete is False
    assert any(
        reason.startswith("unresolved_dataset_sha256:")
        for reason in plan.blocked_reasons
    )
    assert (
        "d4rl9_performance_backend_protocol_not_frozen"
        in plan.blocked_reasons
    )
    assert "d4rl9_performance_backend_not_migrated" not in plan.blocked_reasons
    assert "d4rl9_performance_protocol_not_frozen" in plan.blocked_reasons
    manifest = plan.as_manifest()
    assert manifest["single_canonical_trainer_across_d4rl9_tasks"] is True
    assert manifest["shared_training_engine_with_hopper_mechanism"] is False
    assert manifest["separate_per_task_trainers_allowed"] is False

    with pytest.raises(ValueError, match="duplicates"):
        resolve_d4rl9_execution(
            dataset_paths=_paths(tmp_path),
            seeds=(1, 1),
        )


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
    assert result["single_canonical_trainer_across_d4rl9_tasks"] is True
    assert result["shared_training_engine_with_hopper_mechanism"] is False
    assert result["method_ranking_claim_allowed"] is False

    with pytest.raises(RuntimeError, match="dispatch is blocked"):
        dispatch_d4rl9(
            plan=plan,
            output_root=tmp_path / "blocked",
            task_runner=backend_runner,
        )
