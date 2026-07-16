"""Canonical squared-EXP A2C/PPO with a shared frozen critic and prepared TD/GAE."""
import json
from pathlib import Path
from typing import Any
import numpy as np
import torch
from drpo import e7_squared_exp_night_bootstrap as canonical
from drpo.e7_canonical_injection import _agent_device, _as_tensor, sha256_file

EXPERIMENT_ID = "EXT-H-E7-SQEXP-GAE-01"

def _load(branch: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any], str]:
    manifest = json.loads(Path(branch["advantage_manifest"]).expanduser().read_text())
    estimator = str(branch["template_values"]["advantage_estimator"])
    if manifest.get("status") != "complete":
        raise RuntimeError("prepared manifest is incomplete")
    if (manifest.get("dataset_id"), int(manifest.get("seed", -1))) != (
        branch["dataset_id"], int(branch["seed"])
    ):
        raise RuntimeError("prepared identity mismatch")
    arrays = Path(manifest["advantages"]["path"]).expanduser().resolve()
    critic = Path(manifest["critic"]["path"]).expanduser().resolve()
    if sha256_file(arrays) != manifest["advantages"]["sha256"]:
        raise RuntimeError("prepared advantage hash mismatch")
    if sha256_file(critic) != manifest["critic"]["sha256"]:
        raise RuntimeError("prepared critic hash mismatch")
    with np.load(arrays, allow_pickle=False) as payload:
        advantage = payload[estimator].astype(np.float32, copy=True)
    if advantage.ndim != 1 or not np.isfinite(advantage).all():
        raise RuntimeError("prepared advantage must be one finite vector")
    checkpoint = torch.load(critic, map_location="cpu", weights_only=True)
    return advantage, checkpoint["state_dict"], estimator

def _agent(parent: type, state_dict: dict[str, Any]) -> type:
    class PreparedAgent(parent):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.critic.load_state_dict(state_dict)
            self.c_opt.step = lambda *args, **kwargs: None
        def update(self, s: Any, a: Any, r: Any, ns: Any, d: Any, ep_ret: Any = None) -> Any:
            device = _agent_device(self)
            states, next_states = _as_tensor(s, device=device), _as_tensor(ns, device=device)
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
    advantage, state_dict, estimator = _load(branch)
    old_id, old_a2c = canonical.EXPERIMENT_ID, canonical.patch_canonical_module
    old_ppo, old_atomic = canonical.patch_canonical_module_ppo, canonical._atomic_json
    old_returns: Any = None
    def install(module: Any, target: str) -> None:
        nonlocal old_returns
        setattr(module, target, _agent(getattr(module, target), state_dict))
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
    def atomic(path: Path, payload: Any) -> None:
        if isinstance(payload, dict) and payload.get("experiment_id") == EXPERIMENT_ID:
            payload = {**payload, "gae_used": estimator == "gae",
                       "advantage_estimator": estimator}
        old_atomic(path, payload)
    canonical.EXPERIMENT_ID = EXPERIMENT_ID
    canonical.patch_canonical_module, canonical.patch_canonical_module_ppo = patch_a2c, patch_ppo
    canonical._atomic_json = atomic
    try:
        return canonical.main(argv)
    finally:
        canonical.EXPERIMENT_ID, canonical.patch_canonical_module = old_id, old_a2c
        canonical.patch_canonical_module_ppo, canonical._atomic_json = old_ppo, old_atomic
        if old_returns is not None:
            import d4rl_common.train_loop as loop
            loop.compute_mc_returns = old_returns

if __name__ == "__main__":
    raise SystemExit(main())
