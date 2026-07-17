from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from drpo.e7_canonical_gae_injection import (
    OrderedReplay,
    SnapshotEstimatorConfig,
    build_joint_snapshot_agent_class,
    compute_snapshot_tables,
    transition_id_channel,
)
from drpo.e7_canonical_injection import NegativeControl
from drpo.e7_sqexp_gae import (
    EXPECTED_BRANCHES,
    EXPECTED_DATASETS,
    EXPECTED_SEEDS,
    _build_branches,
    _validate_trainer_args,
)


class FixtureActor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mean = torch.nn.Linear(2, 1, bias=False)
        self.log_std = torch.nn.Parameter(torch.zeros(1, 1))

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.mean(states)
        return mean, self.log_std.expand_as(mean)


class FixtureCritic(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.value = torch.nn.Linear(2, 1, bias=False)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.value(states)


class FixtureAgent:
    def __init__(self) -> None:
        self.gamma = 0.9
        self.tau = 0.7
        self.alpha = 0.11
        self.actor = FixtureActor()
        self.critic = FixtureCritic()
        self.a_opt = torch.optim.SGD(self.actor.parameters(), lr=1e-2)
        self.c_opt = torch.optim.SGD(self.critic.parameters(), lr=1e-2)


def _replay() -> OrderedReplay:
    observations = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]],
        dtype=np.float32,
    )
    return OrderedReplay(
        observations=observations,
        actions=np.zeros((4, 1), dtype=np.float32),
        rewards=np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        next_observations=observations * 0.5,
        terminals=np.asarray([False, True, False, False]),
        timeouts=np.asarray([False, False, True, False]),
    )


def test_snapshot_tables_respect_terminal_timeout_and_tail() -> None:
    replay = _replay()
    td, gae = compute_snapshot_tables(
        replay.rewards,
        np.zeros(replay.size),
        np.full(replay.size, 10.0),
        replay.terminals,
        replay.timeouts,
        gamma=0.9,
        gae_lambda=0.5,
    )
    np.testing.assert_allclose(td, [10.0, 2.0, 12.0, 13.0])
    np.testing.assert_allclose(gae, [10.9, 2.0, 12.0, 13.0])


def test_lambda_zero_reduces_exactly_to_td() -> None:
    replay = _replay()
    td, gae = compute_snapshot_tables(
        replay.rewards,
        np.zeros(replay.size),
        np.ones(replay.size),
        replay.terminals,
        replay.timeouts,
        gamma=0.99,
        gae_lambda=0.0,
    )
    np.testing.assert_array_equal(td, gae)
    np.testing.assert_array_equal(transition_id_channel(4), np.arange(4, dtype=np.float32))


def test_joint_snapshot_agent_updates_critic_and_refreshes() -> None:
    replay = _replay()
    replay.validate()
    instances: list[FixtureAgent] = []
    injected = build_joint_snapshot_agent_class(
        FixtureAgent,
        replay=replay,
        negative_control=NegativeControl(
            method="canonical_signed",
            negative_scale=1.0,
            canonical_alpha=0.11,
        ),
        estimator=SnapshotEstimatorConfig(
            estimator="gae",
            gae_lambda=0.95,
            canonical_batch_size=2,
        ),
        return_mode="metrics_dict",
        instance_sink=instances,
    )
    agent = injected()
    before = [parameter.detach().clone() for parameter in agent.critic.parameters()]
    ids = torch.tensor([0.0, 2.0])
    batch = (
        torch.from_numpy(replay.observations[[0, 2]]),
        torch.from_numpy(replay.actions[[0, 2]]),
        torch.from_numpy(replay.rewards[[0, 2]]),
        torch.from_numpy(replay.next_observations[[0, 2]]),
        torch.from_numpy(replay.terminals[[0, 2]]),
        ids,
    )
    agent.update(*batch)
    assert agent._drpo_snapshot_count == 1
    assert any(
        not torch.equal(left, right)
        for left, right in zip(before, agent.critic.parameters(), strict=True)
    )
    agent.update(*batch)
    assert agent._drpo_snapshot_count == 1
    agent.update(*batch)
    summary = agent._drpo_snapshot_summary()
    assert summary["snapshot_count"] == 2
    assert summary["critic_evolution_observed"] is True
    assert instances == [agent]


def test_terminal_timeout_overlap_fails_closed() -> None:
    replay = _replay()
    with pytest.raises(ValueError, match="must not overlap"):
        OrderedReplay(
            observations=replay.observations,
            actions=replay.actions,
            rewards=replay.rewards,
            next_observations=replay.next_observations,
            terminals=np.asarray([True, False, False, False]),
            timeouts=np.asarray([True, False, False, False]),
        ).validate()


def test_joint_gae_matrix_has_exactly_96_unique_branches() -> None:
    run_spec = {
        "datasets": [
            {"id": dataset, "path": f"/{dataset}.hdf5", "sha256": "0" * 64}
            for dataset in EXPECTED_DATASETS
        ]
    }
    branches = _build_branches(
        SimpleNamespace(expected_canonical_alpha=0.11),
        run_spec,
        {},
    )
    assert len(branches) == EXPECTED_BRANCHES
    assert len({branch.branch_id for branch in branches}) == EXPECTED_BRANCHES
    assert {branch.seed for branch in branches} == set(EXPECTED_SEEDS)
    assert {branch.template_values["actor_update_mode"] for branch in branches} == {"a2c"}


def test_transition_id_channel_rejects_return_weighted_sampling() -> None:
    base_args = [
        "--variant",
        "iqlv_exp_rank",
        "--batch",
        "256",
        "--steps",
        "1000000",
    ]
    _validate_trainer_args(base_args, runtime_probe=False)
    with pytest.raises(ValueError, match="ret_weight_mode=none"):
        _validate_trainer_args(
            [*base_args, "--ret_weight_mode", "rank_pow"],
            runtime_probe=False,
        )
