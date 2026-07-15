from __future__ import annotations

import json
from pathlib import Path

import torch

from drpo.e7_canonical_injection import NegativeControl
from drpo.e7_canonical_ppo_injection import PPOActorControl
from drpo.e7_ppo_kl_refresh import (
    PPOKLEarlyRefreshControl,
    build_ppo_kl_refresh_agent_class,
    diagonal_gaussian_kl_old_to_new,
)


def test_diagonal_gaussian_kl_zero_for_identical_distributions() -> None:
    mean = torch.tensor([[0.0, 1.0], [2.0, -1.0]])
    log_std = torch.zeros_like(mean)
    actual = diagonal_gaussian_kl_old_to_new(mean, log_std, mean, log_std)
    assert torch.allclose(actual, torch.zeros(2))


def test_diagonal_gaussian_kl_known_mean_shift() -> None:
    old_mean = torch.zeros(3, 1)
    new_mean = torch.ones(3, 1)
    log_std = torch.zeros(3, 1)
    actual = diagonal_gaussian_kl_old_to_new(
        old_mean,
        log_std,
        new_mean,
        log_std,
    )
    assert torch.allclose(actual, torch.full((3,), 0.5))


class _TinyActor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mean = torch.nn.Linear(1, 1, bias=False)
        torch.nn.init.zeros_(self.mean.weight)
        self.log_std = torch.nn.Parameter(torch.zeros(1))

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.mean(states)
        return mean, self.log_std.expand_as(mean)


class _TinyCritic(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.value = torch.nn.Linear(1, 1, bias=False)
        torch.nn.init.zeros_(self.value.weight)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.value(states)


class _TinyAgent:
    def __init__(self) -> None:
        self.actor = _TinyActor()
        self.critic = _TinyCritic()
        self.a_opt = torch.optim.SGD(self.actor.parameters(), lr=0.1)
        self.c_opt = torch.optim.SGD(self.critic.parameters(), lr=0.01)
        self.gamma = 0.99
        self.tau = 0.5
        self.alpha = 0.11


def test_kl_early_refresh_triggers_and_writes_complete_diagnostics(
    tmp_path: Path,
) -> None:
    negative = NegativeControl(
        method="positive_only",
        negative_scale=0.0,
        canonical_alpha=0.11,
    )
    ppo = PPOActorControl(
        clip_epsilon=0.2,
        updates_per_old_policy=16,
        diagnostics_interval=1,
        total_steps=2,
    )
    kl = PPOKLEarlyRefreshControl(target_kl=1e-12, diagnostics_interval=1)
    agent_class = build_ppo_kl_refresh_agent_class(
        _TinyAgent,
        negative_control=negative,
        ppo_control=ppo,
        kl_control=kl,
        return_mode="metrics_dict",
        ppo_diagnostics_jsonl=tmp_path / "ppo.jsonl",
        ppo_diagnostics_latest=tmp_path / "ppo_latest.json",
        kl_diagnostics_jsonl=tmp_path / "kl.jsonl",
        kl_diagnostics_latest=tmp_path / "kl_latest.json",
    )
    agent = agent_class()
    states = torch.ones(8, 1)
    actions = torch.ones(8, 1)
    rewards = torch.ones(8)
    next_states = torch.zeros(8, 1)
    dones = torch.zeros(8, dtype=torch.bool)

    first = agent.update(states, actions, rewards, next_states, dones)
    second = agent.update(states, actions, rewards, next_states, dones)

    assert first["actor_update_mode"] == "ppo_clip"
    assert second["actor_update_mode"] == "ppo_clip"
    assert agent._drpo_kl_triggered_refresh_count >= 1
    latest = json.loads((tmp_path / "kl_latest.json").read_text())
    assert latest["status"] == "complete"
    assert latest["update"] == 2
    assert latest["target_kl"] == 1e-12
    assert latest["interval_updates"] >= 1
    assert latest["kl_triggered_refresh_count"] >= 1


def test_kl_control_rejects_invalid_threshold() -> None:
    try:
        PPOKLEarlyRefreshControl(target_kl=0.0).validate()
    except ValueError as exc:
        assert "target_kl" in str(exc)
    else:
        raise AssertionError("zero target KL was accepted")
