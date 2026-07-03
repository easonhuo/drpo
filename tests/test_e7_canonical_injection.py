from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from drpo.e7_canonical_injection import (
    CanonicalContract,
    CanonicalContractError,
    NegativeControl,
    build_injected_agent_class,
    controlled_advantage,
    load_verified_canonical_module,
    patch_canonical_module,
    write_fingerprint_contract,
)


class FixtureActor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mean = torch.nn.Linear(2, 1, bias=False)
        torch.nn.init.constant_(self.mean.weight, 0.2)
        self.log_std = torch.nn.Parameter(torch.tensor([[-0.3]]))

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.mean(states)
        return mean, self.log_std.expand_as(mean)


class FixtureCritic(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.value = torch.nn.Linear(2, 1, bias=False)
        torch.nn.init.constant_(self.value.weight, 0.1)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.value(states)


class FixtureSignedAgent:
    def __init__(self) -> None:
        self.gamma = 0.99
        self.tau = 0.7
        self.alpha = 0.11
        self.actor = FixtureActor()
        self.critic = FixtureCritic()
        self.a_opt = torch.optim.SGD(self.actor.parameters(), lr=1e-2)
        self.c_opt = torch.optim.SGD(self.critic.parameters(), lr=1e-2)

    def update(self, s, a, r, ns, d, ep_ret=None):
        del ep_ret
        s = torch.as_tensor(s, dtype=torch.float32)
        a = torch.as_tensor(a, dtype=torch.float32)
        r = torch.as_tensor(r, dtype=torch.float32).reshape(-1)
        ns = torch.as_tensor(ns, dtype=torch.float32)
        d = torch.as_tensor(d, dtype=torch.bool).reshape(-1)
        v = self.critic(s).squeeze(-1)
        with torch.no_grad():
            target = r + self.gamma * self.critic(ns).squeeze(-1) * (~d).float()
        advantage = target - v.detach()
        mean, log_std = self.actor(s)
        log_prob = torch.distributions.Normal(mean, log_std.exp()).log_prob(a).sum(-1)
        adjusted = torch.where(advantage < 0, advantage * self.alpha, advantage)
        actor_loss = -(log_prob * adjusted).mean()
        self.a_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.a_opt.step()
        error = target - v
        weight = torch.where(error > 0, self.tau, 1 - self.tau)
        critic_loss = (weight * error.square()).mean()
        self.c_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.c_opt.step()
        return 0.0


def batch():
    states = torch.tensor([[1.0, -1.0], [0.5, 0.5], [-1.0, 1.0]])
    actions = torch.tensor([[0.2], [1.5], [-1.0]])
    rewards = torch.tensor([1.0, -1.0, 0.2])
    next_states = states * 0.5
    dones = torch.tensor([False, False, True])
    return states, actions, rewards, next_states, dones


def parameters(module: torch.nn.Module) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in module.parameters()]


def test_canonical_signed_is_one_step_equivalent() -> None:
    original = FixtureSignedAgent()
    injected_base = copy.deepcopy(original)
    injected_type = build_injected_agent_class(
        FixtureSignedAgent,
        control=NegativeControl(
            method="canonical_signed",
            negative_scale=1.0,
            canonical_alpha=0.11,
        ),
        return_mode="zero_float",
    )
    injected_base.__class__ = injected_type
    original.update(*batch())
    injected_base.update(*batch())
    for left, right in zip(parameters(original.actor), parameters(injected_base.actor)):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    for left, right in zip(parameters(original.critic), parameters(injected_base.critic)):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_positive_only_preserves_positive_terms_and_full_batch_denominator() -> None:
    advantage = torch.tensor([2.0, -4.0, 1.0, -1.0])
    distance = torch.ones_like(advantage)
    weighted, factor = controlled_advantage(
        advantage,
        distance,
        NegativeControl(
            method="positive_only",
            negative_scale=0.0,
            canonical_alpha=0.11,
        ),
    )
    torch.testing.assert_close(weighted, torch.tensor([2.0, 0.0, 1.0, 0.0]))
    torch.testing.assert_close(factor, torch.tensor([1.0, 0.0, 1.0, 0.0]))
    assert weighted.mean().item() == pytest.approx(0.75)


def test_zero_decay_taper_reduces_to_global_shape() -> None:
    advantage = torch.tensor([-1.0, -2.0, 3.0])
    distance = torch.tensor([0.1, 100.0, 10.0])
    exp_control = NegativeControl(
        method="exponential",
        negative_scale=0.03,
        canonical_alpha=0.11,
        exponential_coefficient=0.0,
    )
    global_control = NegativeControl(
        method="global",
        negative_scale=0.03,
        canonical_alpha=0.11,
    )
    exp_weighted, _ = controlled_advantage(advantage, distance, exp_control)
    global_weighted, _ = controlled_advantage(advantage, distance, global_control)
    torch.testing.assert_close(exp_weighted, global_weighted, rtol=0, atol=0)


def _write_fixture_source(root: Path) -> None:
    (root / "agents.py").write_text(
        "class SNA2C_IQLV_DistAgent:\n"
        "    pass\n"
    )
    (root / "trainer.py").write_text("import agents\n")


def test_fingerprint_and_fail_closed_hash(tmp_path: Path) -> None:
    source = tmp_path / "canonical"
    source.mkdir()
    _write_fixture_source(source)
    contract_path = tmp_path / "contract.json"
    contract = write_fingerprint_contract(
        canonical_root=source,
        agents_relpath="agents.py",
        trainer_relpath="trainer.py",
        module_name="fixture_agents_hash",
        target_class="SNA2C_IQLV_DistAgent",
        expected_canonical_alpha=0.11,
        output=contract_path,
    )
    assert CanonicalContract.load(contract_path) == contract
    contract.verify_runtime()
    (source / "trainer.py").write_text("import agents\n# changed\n")
    with pytest.raises(CanonicalContractError, match="trainer SHA-256 mismatch"):
        contract.verify_runtime()


def test_module_patch_uses_exact_file(tmp_path: Path) -> None:
    source = tmp_path / "canonical"
    source.mkdir()
    _write_fixture_source(source)
    contract_path = tmp_path / "contract.json"
    contract = write_fingerprint_contract(
        canonical_root=source,
        agents_relpath="agents.py",
        trainer_relpath="trainer.py",
        module_name="fixture_agents_patch",
        target_class="SNA2C_IQLV_DistAgent",
        expected_canonical_alpha=0.11,
        output=contract_path,
    )
    module, checks = load_verified_canonical_module(contract)
    original = module.SNA2C_IQLV_DistAgent
    injected = patch_canonical_module(
        module,
        contract,
        NegativeControl(
            method="global",
            negative_scale=0.01,
            canonical_alpha=0.11,
        ),
    )
    assert module.SNA2C_IQLV_DistAgent is injected
    assert issubclass(injected, original)
    assert checks["agents_path"] == str(source / "agents.py")


def test_reciprocal_quadratic_is_quadratic_in_distance_not_quartic() -> None:
    class ZeroActor(torch.nn.Module):
        def forward(self, states: torch.Tensor):
            mean = torch.zeros((states.shape[0], 1))
            log_std = torch.zeros_like(mean)
            return mean, log_std

    from drpo.e7_canonical_injection import (
        detached_standardized_distance,
        taper_factor,
    )

    states = torch.zeros((1, 1))
    actions = torch.tensor([[3.0]])
    _, _, distance = detached_standardized_distance(ZeroActor(), states, actions)
    assert distance.item() == pytest.approx(3.0)
    factor = taper_factor(
        distance,
        NegativeControl(
            method="reciprocal_quadratic",
            negative_scale=1.0,
            canonical_alpha=0.11,
            reference_distance=1.0,
            reciprocal_quadratic_coefficient=1.0,
        ),
    )
    assert factor.item() == pytest.approx(1.0 / 10.0)
