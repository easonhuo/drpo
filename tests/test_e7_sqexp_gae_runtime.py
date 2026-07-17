from __future__ import annotations

import dataclasses
import json
import subprocess
from types import SimpleNamespace
import sys
from pathlib import Path

import numpy as np
import pytest

from drpo import e7_sqexp_gae as gae

CONFIG = Path(__file__).parents[1] / "configs" / "e7_sqexp_gae_v1.yaml"

def protocol() -> gae.FrozenProtocol:
    return gae.load_protocol(CONFIG)

def test_critic_and_prepared_advantage_identity_binding() -> None:
    value = protocol()
    dataset = gae.DatasetSpec(
        id=gae.EXPECTED_DATASETS[0],
        path="/tmp/fake.hdf5",
        sha256="a" * 64,
        format="legacy_d4rl_hdf5",
        env_id="Hopper-v4",
        score_protocol="d4rl_v2_percent",
        reference_min_score=-1.0,
        reference_max_score=1.0,
    )
    first = gae.critic_identity(
        dataset=dataset,
        seed=200,
        protocol=value,
        source_run_spec_sha256="b" * 64,
    )
    second = gae.critic_identity(
        dataset=dataset,
        seed=201,
        protocol=value,
        source_run_spec_sha256="b" * 64,
    )
    assert first != second
    array = np.array([1.0, 2.0], dtype=np.float32)
    td = gae.prepared_advantage_identity(
        critic_identity_sha256=first,
        critic_checkpoint_sha256="c" * 64,
        estimator="one_step_td",
        protocol=value,
        arrays_sha256=gae.array_sha256(array),
    )
    behavior = gae.prepared_advantage_identity(
        critic_identity_sha256=first,
        critic_checkpoint_sha256="c" * 64,
        estimator="behavior_gae",
        protocol=value,
        arrays_sha256=gae.array_sha256(array),
    )
    assert td != behavior


def test_aggregation_pairs_without_failed_imputation(tmp_path: Path) -> None:
    value = protocol()
    branches = gae.build_actor_branches(value)
    excluded_pair = branches[0].pair_key
    for branch in branches:
        directory = (
            tmp_path
            / "branches"
            / branch.dataset_id
            / f"seed_{branch.seed}"
            / branch.id
        )
        directory.mkdir(parents=True)
        complete = branch.pair_key != excluded_pair or branch.estimator != "behavior_gae"
        payload = {
            "experiment_id": gae.EXPERIMENT_ID,
            "branch_id": branch.id,
            "fixed_budget_completed": complete,
            "terminal_state": (
                "finite_fixed_horizon_terminal" if complete else "nan_inf_numerical_failure"
            ),
            "late_window_normalized_return_mean": 2.0 if branch.estimator == "behavior_gae" else 1.0,
            "task_performance_collapse": "not_adjudicated_no_frozen_task_collapse_threshold",
            "support_or_variance_boundary_event": False,
            "nan_inf_numerical_failure": not complete,
            "critic_immutability_verified": True,
        }
        (directory / "summary.json").write_text(json.dumps(payload))
    audit = gae.aggregate_results(tmp_path, value)
    assert audit["paired_cells_included"] == 95
    assert audit["paired_cells_excluded_without_imputation"] == 1
    assert audit["failed_cell_imputation_used"] is False
    assert audit["fixed_budget_completed_branches"] == 191
    assert audit["nan_inf_numerical_failure_branches"] == 1
    assert audit["critic_immutability_failures"] == 0
    paired = (tmp_path / "aggregate" / "paired_gae_minus_td.csv").read_text()
    assert ",1.0\n" in paired


def test_plan_cli_reports_192_branches() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "drpo.e7_sqexp_gae",
            "plan",
            "--config",
            str(CONFIG),
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["critic_job_count"] == 12
    assert payload["actor_branch_count"] == 192
    assert payload["held_out_seeds_scheduled"] is False
    assert payload["formal_run_allowed"] is False


def test_coordinator_file_is_a_runnable_worker_entrypoint() -> None:
    coordinator = (
        Path(__file__).parents[1]
        / "src"
        / "drpo"
        / "e7_sqexp_gae_coordinator.py"
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(coordinator),
            "plan",
            "--config",
            str(CONFIG),
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["critic_job_count"] == 12
    assert payload["actor_branch_count"] == 192



def test_tiny_end_to_end_critic_prepare_and_actor_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from drpo import e7_sqexp_gae_actor_train as actor_train
    from drpo import e7_sqexp_gae_evaluation as evaluation
    from drpo import e7_sqexp_gae_preparation as preparation

    rng = np.random.default_rng(17)
    episode_length = 4
    episodes = 4
    size = episode_length * episodes
    observations = rng.normal(size=(size, 3)).astype(np.float32)
    next_observations = rng.normal(size=(size, 3)).astype(np.float32)
    actions = np.tanh(rng.normal(size=(size, 2))).astype(np.float32)
    rewards = rng.normal(size=size).astype(np.float32)
    terminals = np.zeros(size, dtype=np.bool_)
    timeouts = np.zeros(size, dtype=np.bool_)
    terminals[episode_length - 1] = True
    timeouts[2 * episode_length - 1] = True
    terminals[3 * episode_length - 1] = True
    # The final stored trajectory is deliberately nonterminal to exercise its
    # bootstrap-and-stop-carry contract in the executable path.
    episode_ids = np.repeat(np.arange(episodes), episode_length)
    data = SimpleNamespace(
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        terminals=terminals,
        timeouts=timeouts,
        episode_ids=episode_ids,
        size=size,
    )
    monkeypatch.setattr(preparation, "_load_data", lambda dataset: data)
    split = {
        "train": np.arange(0, 8, dtype=np.int64),
        "validation": np.arange(8, 12, dtype=np.int64),
        "test": np.arange(12, 16, dtype=np.int64),
    }
    monkeypatch.setattr(preparation, "_split_episodes", lambda episode_ids, seed: split)
    monkeypatch.setattr(actor_train, "_load_data", lambda dataset: data)
    monkeypatch.setattr(
        actor_train,
        "rollout",
        lambda *args, **kwargs: {
            "rollout_status": "synthetic_test_only",
            "rollout_return_mean": 1.0,
            "rollout_return_std": 0.0,
            "normalized_return": 1.0,
            "normalized_return_available": True,
            "rollout_episodes": 1,
            "rollout_backend": {"test_only": True},
        },
    )
    tiny = dataclasses.replace(
        protocol(),
        critic_steps=4,
        actor_steps=4,
        batch_size=4,
        evaluation_interval=2,
        evaluation_episodes=1,
        late_window_start=2,
    )
    dataset = gae.DatasetSpec(
        id=gae.EXPECTED_DATASETS[0],
        path=str(tmp_path / "synthetic.hdf5"),
        sha256="d" * 64,
        format="legacy_d4rl_hdf5",
        env_id="Hopper-v4",
        score_protocol="d4rl_v2_percent",
        reference_min_score=-1.0,
        reference_max_score=1.0,
    )
    critic_dir = tmp_path / "critic"
    manifest = preparation.train_frozen_critic_and_prepare(
        dataset=dataset,
        seed=200,
        protocol=tiny,
        source_run_spec_sha256="e" * 64,
        output_dir=critic_dir,
        device_name="cpu",
    )
    assert manifest["fixed_budget_completed"] is True
    assert manifest["numerical_audit"]["lambda_zero_td_max_abs_disagreement"] == 0.0
    branch = gae.ActorBranch(
        dataset_id=dataset.id,
        seed=200,
        estimator="behavior_gae",
        actor_mode="ppo_clip_k4",
        control_id="sqexp_c64",
        coefficient=64.0,
    )
    result = actor_train.train_actor_branch(
        branch=branch,
        dataset=dataset,
        protocol=tiny,
        critic_dir=critic_dir,
        output_dir=tmp_path / "actor",
        device_name="cpu",
        source_run_spec_sha256="e" * 64,
    )
    assert result["fixed_budget_completed"] is True
    assert result["critic_immutability_verified"] is True
    assert result["actor_and_critic_parameters_disjoint"] is True
    assert result["ppo_reference_refresh_count"] == 1
    assert result["late_window_rows"] == 2
