from __future__ import annotations

import copy

import torch

from drpo.e7_sqexp_gae_artifacts import state_digest
from drpo.e7_sqexp_gae_minimal import _agent


class _Optimizer:
    def __init__(self, parameters: list[torch.nn.Parameter]) -> None:
        self.inner = torch.optim.Adam(parameters, lr=0.05)

    def zero_grad(self, *args, **kwargs) -> None:
        self.inner.zero_grad(*args, **kwargs)

    def step(self, *args, **kwargs) -> None:
        self.inner.step(*args, **kwargs)


class _CanonicalLikeAgent:
    def __init__(self) -> None:
        self.actor = torch.nn.Linear(2, 1, bias=False)
        self.critic = torch.nn.Linear(2, 1, bias=False)
        self.a_opt = _Optimizer(list(self.actor.parameters()))
        self.c_opt = _Optimizer(list(self.critic.parameters()))
        self.gamma = 0.9
        self.last_advantage: torch.Tensor | None = None

    def update(self, s, a, r, ns, d, ep_ret=None):
        del a, ep_ret
        values = self.critic(s).squeeze(-1)
        with torch.no_grad():
            next_values = self.critic(ns).squeeze(-1)
            target = r + self.gamma * next_values * (~d).float()
        advantage = target - values.detach()
        self.last_advantage = advantage.clone()
        actor_loss = -(self.actor(s).squeeze(-1) * advantage).mean()
        self.a_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.a_opt.step()
        critic_loss = ((target - values) ** 2).mean()
        self.c_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.c_opt.step()
        return float(actor_loss.detach())


def test_adapter_preserves_advantage_and_freezes_critic() -> None:
    torch.manual_seed(3)
    reference = _CanonicalLikeAgent()
    state = copy.deepcopy(reference.critic.state_dict())
    instances = []
    wrapped = _agent(_CanonicalLikeAgent, state, instances)
    agent = wrapped()
    actor_before = copy.deepcopy(agent.actor.state_dict())
    critic_before = state_digest(agent.critic.state_dict())
    states = torch.randn(8, 2)
    next_states = torch.randn(8, 2)
    actions = torch.zeros(8, 1)
    dones = torch.tensor([False, False, True, False, False, True, False, False])
    prepared = torch.linspace(-2.0, 2.0, 8)
    agent.update(states, actions, torch.zeros(8), next_states, dones, prepared)
    assert agent.last_advantage is not None
    torch.testing.assert_close(agent.last_advantage, prepared)
    assert state_digest(agent.critic.state_dict()) == critic_before
    assert any(
        not torch.equal(actor_before[name], agent.actor.state_dict()[name])
        for name in actor_before
    )
