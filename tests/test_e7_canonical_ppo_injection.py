from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from drpo.e7_canonical_injection import NegativeControl
from drpo.e7_canonical_ppo_injection import (
    PPOActorControl,
    build_ppo_injected_agent_class,
)


class TinyActor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
        self.log_std = torch.nn.Parameter(torch.zeros(1, 1))
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.linear(states)
        return mean, self.log_std.expand_as(mean)


class TinyCritic(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.linear(states)


class TinyCanonicalAgent:
    def __init__(self, actor_lr: float = 0.05, critic_lr: float = 0.01) -> None:
        self.gamma = 0.0
        self.tau = 0.5
        self.alpha = 0.11
        self.actor = TinyActor()
        self.critic = TinyCritic()
        self.a_opt = torch.optim.SGD(self.actor.parameters(), lr=actor_lr)
        self.c_opt = torch.optim.SGD(self.critic.parameters(), lr=critic_lr)


def signed_control() -> NegativeControl:
    return NegativeControl(
        method="global",
        negative_scale=1.0,
        canonical_alpha=0.11,
        reference_distance=2.0,
        exponential_coefficient=1.0,
    )


def batch() -> tuple[torch.Tensor, ...]:
    states = torch.zeros(4, 1)
    actions = torch.tensor([[1.0], [1.0], [-1.0], [-1.0]])
    rewards = torch.tensor([1.0, 1.0, -1.0, -1.0])
    next_states = torch.zeros_like(states)
    dones = torch.ones(4, dtype=torch.bool)
    return states, actions, rewards, next_states, dones


def state_dict_clone(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().clone()
        for name, value in module.state_dict().items()
    }


def states_equal(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
) -> bool:
    return all(torch.equal(left[key], right[key]) for key in left)


def test_ppo_actor_control_validation() -> None:
    PPOActorControl().validate()
    with pytest.raises(ValueError, match="clip_epsilon"):
        PPOActorControl(clip_epsilon=0.0).validate()
    with pytest.raises(ValueError, match="updates_per_old_policy"):
        PPOActorControl(updates_per_old_policy=1).validate()
    with pytest.raises(ValueError, match="diagnostics_interval"):
        PPOActorControl(diagnostics_interval=0).validate()


def test_first_update_ratio_is_one_and_unclipped(tmp_path: Path) -> None:
    latest = tmp_path / "latest.json"
    injected = build_ppo_injected_agent_class(
        TinyCanonicalAgent,
        negative_control=signed_control(),
        ppo_control=PPOActorControl(
            diagnostics_interval=1,
            total_steps=1,
        ),
        return_mode="metrics_dict",
        diagnostics_jsonl=tmp_path / "diag.jsonl",
        diagnostics_latest=latest,
    )
    agent = injected(actor_lr=0.0)
    agent.update(*batch())
    payload = json.loads(latest.read_text())
    sampled = payload["sampled_pre_update"]
    assert sampled["ratio_p01"] == pytest.approx(1.0)
    assert sampled["ratio_p99"] == pytest.approx(1.0)
    assert payload["pre_update"]["objective_clip_fraction"] == pytest.approx(0.0)
    assert payload["status"] == "complete"


def test_sign_aware_clipping_activates_for_positive_and_negative(
    tmp_path: Path,
) -> None:
    latest = tmp_path / "latest.json"
    injected = build_ppo_injected_agent_class(
        TinyCanonicalAgent,
        negative_control=signed_control(),
        ppo_control=PPOActorControl(
            diagnostics_interval=1,
            total_steps=1,
        ),
        return_mode="metrics_dict",
        diagnostics_jsonl=tmp_path / "diag.jsonl",
        diagnostics_latest=latest,
    )
    agent = injected(actor_lr=0.0)
    with torch.no_grad():
        agent.actor.linear.bias.fill_(1.0)
    agent.update(*batch())
    payload = json.loads(latest.read_text())
    pre = payload["pre_update"]
    assert pre["ratio_outside_fraction"] == pytest.approx(1.0)
    assert pre["objective_clip_fraction"] == pytest.approx(1.0)
    assert pre["positive_objective_clip_fraction"] == pytest.approx(1.0)
    assert pre["negative_objective_clip_fraction"] == pytest.approx(1.0)
    assert payload["sampled_pre_update"]["ratio_p01"] < 0.8
    assert payload["sampled_pre_update"]["ratio_p99"] > 1.2


def test_old_policy_is_frozen_for_four_updates_then_refreshed(
    tmp_path: Path,
) -> None:
    injected = build_ppo_injected_agent_class(
        TinyCanonicalAgent,
        negative_control=signed_control(),
        ppo_control=PPOActorControl(
            updates_per_old_policy=4,
            diagnostics_interval=5,
            total_steps=5,
        ),
        return_mode="metrics_dict",
        diagnostics_jsonl=tmp_path / "diag.jsonl",
        diagnostics_latest=tmp_path / "latest.json",
    )
    agent = injected(actor_lr=0.05)
    old_initial = state_dict_clone(agent._drpo_old_actor)
    for _ in range(4):
        agent.update(*batch())
        assert states_equal(old_initial, state_dict_clone(agent._drpo_old_actor))
    actor_before_fifth = state_dict_clone(agent.actor)
    agent.update(*batch())
    assert states_equal(actor_before_fifth, state_dict_clone(agent._drpo_old_actor))
    assert agent._drpo_old_policy_refresh_count == 1
    assert not states_equal(
        state_dict_clone(agent.actor),
        state_dict_clone(agent._drpo_old_actor),
    )


def test_positive_only_has_no_negative_objective_clip_denominator(
    tmp_path: Path,
) -> None:
    control = NegativeControl(
        method="positive_only",
        negative_scale=0.0,
        canonical_alpha=0.11,
        reference_distance=2.0,
        exponential_coefficient=1.0,
    )
    injected = build_ppo_injected_agent_class(
        TinyCanonicalAgent,
        negative_control=control,
        ppo_control=PPOActorControl(diagnostics_interval=1, total_steps=1),
        return_mode="metrics_dict",
        diagnostics_jsonl=tmp_path / "diag.jsonl",
        diagnostics_latest=tmp_path / "latest.json",
    )
    agent = injected(actor_lr=0.0)
    with torch.no_grad():
        agent.actor.linear.bias.fill_(1.0)
    agent.update(*batch())
    payload = json.loads((tmp_path / "latest.json").read_text())
    assert payload["pre_update"]["negative_samples"] == 0
    assert payload["pre_update"]["negative_objective_clip_fraction"] is None
