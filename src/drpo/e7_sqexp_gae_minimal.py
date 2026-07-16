"""Canonical squared-EXP A2C/PPO with verified GAE and a frozen critic."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from drpo import e7_squared_exp_night_bootstrap as canonical
from drpo.e7_canonical_injection import _agent_device, _as_tensor
from drpo.e7_sqexp_gae_artifacts import load_prepared, state_digest

EXPERIMENT_ID = "EXT-H-E7-SQEXP-GAE-01"


def _agent(parent: type, state_dict: dict[str, Any], instances: list[Any]) -> type:
    class PreparedAgent(parent):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.critic.load_state_dict(state_dict, strict=True)
            self._drpo_critic_initial_sha256 = state_digest(self.critic.state_dict())
            self.c_opt.step = lambda *args, **kwargs: None
            instances.append(self)

        def update(
            self,
            s: Any,
            a: Any,
            r: Any,
            ns: Any,
            d: Any,
            ep_ret: Any = None,
        ) -> Any:
            del r
            device = _agent_device(self)
            states = _as_tensor(s, device=device)
            next_states = _as_tensor(ns, device=device)
            dones = _as_tensor(d, device=device, dtype=torch.bool).reshape(-1)
            advantage = _as_tensor(ep_ret, device=device).reshape(-1).detach()
            with torch.no_grad():
                value = self.critic(states).squeeze(-1)
                next_value = self.critic(next_states).squeeze(-1)
                reward = advantage + value - float(self.gamma) * next_value * (~dones)
                recovered = reward + float(self.gamma) * next_value * (~dones) - value
                if not torch.allclose(recovered, advantage, atol=1e-6, rtol=1e-6):
                    raise RuntimeError("adapter changed the prepared advantage")
            return super().update(s, a, reward, ns, d, ep_ret)

    return PreparedAgent


def main(argv: list[str] | None = None) -> int:
    args = canonical.build_parser().parse_args(argv)
    branch = json.loads(Path(args.branch_config).expanduser().read_text())
    if branch["template_values"]["actor_update_mode"] not in {
        "a2c",
        "ppo_clip_k4",
    }:
        raise ValueError("GAE pilot supports only canonical A2C and PPO-K4")
    advantage, state_dict, estimator, provenance = load_prepared(branch)
    instances: list[Any] = []
    old_id, old_a2c = canonical.EXPERIMENT_ID, canonical.patch_canonical_module
    old_ppo = canonical.patch_canonical_module_ppo
    old_atomic = canonical._atomic_json
    old_returns: Any = None

    def install(module: Any, target: str) -> None:
        nonlocal old_returns
        setattr(module, target, _agent(getattr(module, target), state_dict, instances))
        import d4rl_common.train_loop as loop

        old_returns = loop.compute_mc_returns

        def prepared_returns(rewards: np.ndarray, *_: Any) -> np.ndarray:
            if len(rewards) != len(advantage):
                raise RuntimeError("prepared advantage length mismatch")
            return advantage.copy()

        loop.compute_mc_returns = prepared_returns

    def patch_a2c(module: Any, contract: Any, control: Any) -> type:
        result = old_a2c(module, contract, control)
        install(module, contract.target_class)
        return result

    def patch_ppo(module: Any, target: str, **kwargs: Any) -> type:
        result = old_ppo(module, target, **kwargs)
        install(module, target)
        return result

    canonical.EXPERIMENT_ID = EXPERIMENT_ID
    canonical.patch_canonical_module = patch_a2c
    canonical.patch_canonical_module_ppo = patch_ppo
    try:
        result = canonical.main(argv)
        if len(instances) != 1:
            raise RuntimeError(f"expected one canonical agent, found {len(instances)}")
        initial_sha = instances[0]._drpo_critic_initial_sha256
        final_sha = state_digest(instances[0].critic.state_dict())
        manifest_path = Path(args.branch_manifest).expanduser().resolve()
        payload = json.loads(manifest_path.read_text())
        payload.update(
            {
                "advantage_estimator": estimator,
                "gae_used": estimator == "gae",
                "advantage_provenance": provenance,
                "critic_initial_state_sha256": initial_sha,
                "critic_final_state_sha256": final_sha,
                "critic_immutability_verified": initial_sha == final_sha,
            }
        )
        old_atomic(manifest_path, payload)
        if initial_sha != final_sha:
            raise RuntimeError("frozen critic changed during actor training")
        return result
    finally:
        canonical.EXPERIMENT_ID = old_id
        canonical.patch_canonical_module = old_a2c
        canonical.patch_canonical_module_ppo = old_ppo
        if old_returns is not None:
            import d4rl_common.train_loop as loop

            loop.compute_mc_returns = old_returns


if __name__ == "__main__":
    raise SystemExit(main())
