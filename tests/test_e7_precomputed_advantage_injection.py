from __future__ import annotations

import pytest
import torch

from drpo.e7_canonical_injection import NegativeControl
from drpo.e7_canonical_ppo_injection import PPOActorControl
from drpo.e7_precomputed_advantage_injection import (
    build_precomputed_a2c_agent_class,
    build_precomputed_ppo_agent_class,
)


class _Actor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mean = torch.nn.Linear(2, 1)
        self.log_std = torch.nn.Parameter(torch.zeros(1, 1))

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.mean(states)
        return mean, self.log_std.expand_as(mean)


class _TinyAgent:
    def __init__(self) -> None:
        self.alpha = 0.11
        self.gamma = 0.99
        self.tau = 0.5
        self.actor = _Actor()
        self.critic = torch.nn.Linear(2, 1)
        self.a_opt = torch.optim.Adam(self.actor.parameters(), lr=1e-2)
        self.c_opt = torch.optim.Adam(self.critic.parameters(), lr=1e-2)


def _control() -> NegativeControl:
    control = NegativeControl(
        method="global",
        negative_scale=1.0,
        canonical_alpha=0.11,
        reference_distance=2.0,
    )
    control.validate()
    return control


def _batch() -> tuple[torch.Tensor, ...]:
    states = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    actions = torch.tensor([[0.5], [-0.5]])
    rewards = torch.zeros(2)
    next_states = states.flip(0)
    terminals = torch.tensor([False, True])
    prepared_advantage = torch.tensor([1.0, -0.25])
    return states, actions, rewards, next_states, terminals, prepared_advantage


def _state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().clone()
        for name, value in module.state_dict().items()
    }


def _changed(
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
) -> bool:
    return any(not torch.equal(before[name], after[name]) for name in before)


def test_precomputed_a2c_updates_actor_and_keeps_critic_frozen() -> None:
    agent_class = build_precomputed_a2c_agent_class(
        _TinyAgent,
        negative_control=_control(),
        return_mode="metrics_dict",
    )
    agent = agent_class()
    actor_before = _state(agent.actor)
    critic_before = _state(agent.critic)

    metrics = agent.update(*_batch())

    assert _changed(actor_before, _state(agent.actor))
    assert not _changed(critic_before, _state(agent.critic))
    assert all(not parameter.requires_grad for parameter in agent.critic.parameters())
    assert metrics["critic_frozen"] is True
    assert metrics["critic_loss"] is None


def test_precomputed_ppo_updates_actor_and_keeps_critic_frozen() -> None:
    agent_class = build_precomputed_ppo_agent_class(
        _TinyAgent,
        negative_control=_control(),
        ppo_control=PPOActorControl(
            clip_epsilon=0.2,
            updates_per_old_policy=2,
            diagnostics_interval=1,
            total_steps=2,
        ),
        return_mode="metrics_dict",
    )
    agent = agent_class()
    actor_before = _state(agent.actor)
    critic_before = _state(agent.critic)

    first = agent.update(*_batch())
    second = agent.update(*_batch())

    assert _changed(actor_before, _state(agent.actor))
    assert not _changed(critic_before, _state(agent.critic))
    assert all(not parameter.requires_grad for parameter in agent.critic.parameters())
    assert first["critic_frozen"] is True
    assert second["ppo_update_index"] == 2


def test_precomputed_update_requires_explicit_advantage() -> None:
    agent_class = build_precomputed_a2c_agent_class(
        _TinyAgent,
        negative_control=_control(),
        return_mode="zero_float",
    )
    agent = agent_class()
    states, actions, rewards, next_states, terminals, _ = _batch()

    with pytest.raises(ValueError, match="prepared advantage is required"):
        agent.update(states, actions, rewards, next_states, terminals, None)
