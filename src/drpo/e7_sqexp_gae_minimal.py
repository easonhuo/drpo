"""Canonical squared-EXP A2C/PPO with verified trajectory GAE and a frozen critic."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from drpo import e7_squared_exp_night_bootstrap as canonical
from drpo.e7_canonical_injection import _agent_device, _as_tensor, sha256_file

EXPERIMENT_ID = "EXT-H-E7-SQEXP-GAE-01"
_ALLOWED_ESTIMATORS = {"td", "gae"}


def compute_gae_from_td(
    td: np.ndarray,
    terminals: np.ndarray,
    timeouts: np.ndarray,
    *,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> np.ndarray:
    """Accumulate one-step TD residuals without crossing terminal/timeout/tail boundaries."""
    td64 = np.asarray(td, dtype=np.float64).reshape(-1)
    terminals = np.asarray(terminals, dtype=np.bool_).reshape(-1)
    timeouts = np.asarray(timeouts, dtype=np.bool_).reshape(-1)
    if not (td64.shape == terminals.shape == timeouts.shape) or not td64.size:
        raise ValueError("TD, terminal, and timeout vectors must be non-empty and aligned")
    if bool((terminals & timeouts).any()):
        raise ValueError("terminal and timeout flags overlap")
    if not np.isfinite(td64).all() or not 0.0 <= gamma <= 1.0 or not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("invalid GAE input")
    result = np.empty_like(td64)
    running = 0.0
    continuation = ~(terminals | timeouts)
    for index in range(td64.size - 1, -1, -1):
        running = td64[index] + gamma * gae_lambda * continuation[index] * running
        result[index] = running
    return result.astype(np.float32)


def _state_digest(state: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _load(branch: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any], str, dict[str, Any]]:
    manifest_path = Path(branch["advantage_manifest"]).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    estimator = str(branch["template_values"]["advantage_estimator"])
    if estimator not in _ALLOWED_ESTIMATORS or manifest.get("status") != "complete":
        raise RuntimeError("unsupported estimator or incomplete prepared manifest")
    identity = manifest.get("dataset_id"), int(manifest.get("seed", -1)), manifest.get("dataset_sha256")
    expected = branch["dataset_id"], int(branch["seed"]), branch["dataset_sha256"]
    if identity != expected:
        raise RuntimeError(f"prepared identity mismatch: {identity} != {expected}")
    arrays = Path(manifest["advantages"]["path"]).expanduser().resolve()
    critic = Path(manifest["critic"]["path"]).expanduser().resolve()
    if sha256_file(arrays) != manifest["advantages"]["sha256"]:
        raise RuntimeError("prepared advantage hash mismatch")
    if sha256_file(critic) != manifest["critic"]["sha256"]:
        raise RuntimeError("prepared critic hash mismatch")
    with np.load(arrays, allow_pickle=False) as payload:
        td = payload["td"].astype(np.float32, copy=True)
        stored_gae = payload["gae"].astype(np.float32, copy=True)
    dataset = Path(branch.get("dataset_path", manifest["dataset_path"])).expanduser().resolve()
    if sha256_file(dataset) != branch["dataset_sha256"]:
        raise RuntimeError("ordered dataset hash mismatch")
    with h5py.File(dataset, "r") as source:
        terminals = source["terminals"][:].astype(np.bool_)
        timeouts = source["timeouts"][:].astype(np.bool_)
    gae = compute_gae_from_td(
        td,
        terminals,
        timeouts,
        gamma=float(manifest.get("gamma", 0.99)),
        gae_lambda=float(manifest.get("gae_lambda", 0.95)),
    )
    if not np.allclose(gae, stored_gae, atol=1e-6, rtol=1e-6):
        raise RuntimeError("current GAE implementation disagrees with prepared artifact")
    advantage = td if estimator == "td" else gae
    if advantage.ndim != 1 or not np.isfinite(advantage).all():
        raise RuntimeError("prepared advantage must be one finite vector")
    try:
        checkpoint = torch.load(critic, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(critic, map_location="cpu")
    provenance = {
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "gae_recomputed_from_td_and_boundaries": True,
        "gae_matches_prepared_artifact": True,
    }
    return advantage, checkpoint["state_dict"], estimator, provenance


def _agent(parent: type, state_dict: dict[str, Any], instances: list[Any]) -> type:
    class PreparedAgent(parent):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.critic.load_state_dict(state_dict, strict=True)
            self._drpo_critic_initial_sha256 = _state_digest(self.critic.state_dict())
            self.c_opt.step = lambda *args, **kwargs: None
            instances.append(self)

        def update(self, s: Any, a: Any, r: Any, ns: Any, d: Any, ep_ret: Any = None) -> Any:
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
    if branch["template_values"]["actor_update_mode"] not in {"a2c", "ppo_clip_k4"}:
        raise ValueError("GAE pilot supports only canonical A2C and PPO-K4")
    advantage, state_dict, estimator, provenance = _load(branch)
    instances: list[Any] = []
    old_id, old_a2c = canonical.EXPERIMENT_ID, canonical.patch_canonical_module
    old_ppo, old_atomic = canonical.patch_canonical_module_ppo, canonical._atomic_json
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

    def atomic(path: Path, payload: Any) -> None:
        if isinstance(payload, dict) and payload.get("experiment_id") == EXPERIMENT_ID:
            payload = {**payload, "advantage_estimator": estimator, "gae_used": estimator == "gae"}
        old_atomic(path, payload)

    canonical.EXPERIMENT_ID = EXPERIMENT_ID
    canonical.patch_canonical_module, canonical.patch_canonical_module_ppo = patch_a2c, patch_ppo
    canonical._atomic_json = atomic
    try:
        result = canonical.main(argv)
        if len(instances) != 1:
            raise RuntimeError(f"expected one canonical agent instance, found {len(instances)}")
        final_sha = _state_digest(instances[0].critic.state_dict())
        initial_sha = instances[0]._drpo_critic_initial_sha256
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
        canonical.EXPERIMENT_ID, canonical.patch_canonical_module = old_id, old_a2c
        canonical.patch_canonical_module_ppo, canonical._atomic_json = old_ppo, old_atomic
        if old_returns is not None:
            import d4rl_common.train_loop as loop

            loop.compute_mc_returns = old_returns


if __name__ == "__main__":
    raise SystemExit(main())
