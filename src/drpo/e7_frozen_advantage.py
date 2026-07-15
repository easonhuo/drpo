"""Feed precomputed trajectory advantages into existing canonical E7 updates.

The wrapper is applied *after* the existing A2C or PPO injection.  It does not
reimplement either actor objective.  It synthesizes rewards such that the parent
class reconstructs the exact supplied advantage under a shared frozen critic,
while replacing critic optimizer steps with no-ops.  Periodic diagnostics prove
that the critic is unchanged and keep support/variance-boundary events separate
from task performance and NaN/Inf failures.
"""

from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import torch

from drpo.e7_canonical_injection import (
    CanonicalContractError,
    _agent_device,
    _as_tensor,
)


class FrozenOptimizer:
    """Preserve zero_grad/backward diagnostics while making optimizer steps no-op."""

    def __init__(self, optimizer: Any) -> None:
        self._optimizer = optimizer
        self.step_calls = 0

    def zero_grad(self, *args: Any, **kwargs: Any) -> Any:
        return self._optimizer.zero_grad(*args, **kwargs)

    def step(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self.step_calls += 1
        return None

    def state_dict(self) -> Mapping[str, Any]:
        return self._optimizer.state_dict()

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> Any:
        return self._optimizer.load_state_dict(state_dict)

    @property
    def param_groups(self) -> Any:
        return self._optimizer.param_groups

    @property
    def defaults(self) -> Any:
        return self._optimizer.defaults

    def __getattr__(self, name: str) -> Any:
        return getattr(self._optimizer, name)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True) + "\n")


def _load_checkpoint(path: str | Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"frozen critic checkpoint is missing: {checkpoint_path}")
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("frozen critic checkpoint must be a mapping")
    state = payload.get("critic")
    metadata = payload.get("metadata", {})
    if not isinstance(state, Mapping) or not state:
        raise ValueError("frozen critic checkpoint has no critic state_dict")
    if not isinstance(metadata, Mapping):
        raise ValueError("frozen critic checkpoint metadata must be a mapping")
    tensors: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if not torch.is_tensor(value):
            raise ValueError(f"critic state entry {key!r} is not a tensor")
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"critic state entry {key!r} contains NaN/Inf")
        tensors[str(key)] = value.detach().cpu()
    return tensors, dict(metadata)


def _parameter_snapshot(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in module.named_parameters()
    }


def _max_parameter_change(
    module: torch.nn.Module,
    reference: Mapping[str, torch.Tensor],
) -> float:
    maximum = 0.0
    current = dict(module.named_parameters())
    if set(current) != set(reference):
        raise CanonicalContractError("critic parameter names changed after checkpoint load")
    for name, parameter in current.items():
        delta = parameter.detach().cpu() - reference[name]
        maximum = max(maximum, float(delta.abs().max()))
    return maximum


def build_external_advantage_agent_class(
    base_class: type,
    *,
    critic_checkpoint: str | Path,
    advantage_metadata: Mapping[str, Any],
    return_mode: str,
    diagnostics_interval: int = 1000,
    total_steps: int = 1_000_000,
    support_diagnostics_jsonl: str | Path | None = None,
    support_diagnostics_latest: str | Path | None = None,
) -> type:
    """Wrap an already-injected canonical agent with frozen external advantages."""

    if return_mode not in {"zero_float", "metrics_dict"}:
        raise ValueError(f"unsupported return_mode={return_mode!r}")
    if diagnostics_interval <= 0 or total_steps <= 0:
        raise ValueError("diagnostics_interval and total_steps must be positive")
    source = str(advantage_metadata.get("advantage_source"))
    if source not in {"one_step_td", "gae_lambda_0p95"}:
        raise ValueError(f"unsupported advantage_source={source!r}")
    gamma = float(advantage_metadata.get("gamma"))
    gae_lambda = float(advantage_metadata.get("gae_lambda"))
    if not math.isfinite(gamma) or not math.isfinite(gae_lambda):
        raise ValueError("advantage gamma/lambda must be finite")
    critic_state, checkpoint_metadata = _load_checkpoint(critic_checkpoint)
    metadata = {
        **dict(advantage_metadata),
        "critic_checkpoint_metadata": checkpoint_metadata,
    }
    support_jsonl = (
        None
        if support_diagnostics_jsonl is None
        else Path(support_diagnostics_jsonl).resolve()
    )
    support_latest = (
        None
        if support_diagnostics_latest is None
        else Path(support_diagnostics_latest).resolve()
    )

    class ExternalAdvantageAgent(base_class):  # type: ignore[misc, valid-type]
        _drpo_advantage_metadata = metadata

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            if not hasattr(self, "critic") or not hasattr(self, "c_opt"):
                raise CanonicalContractError(
                    "external-advantage wrapper requires critic and c_opt"
                )
            incompatible = self.critic.load_state_dict(critic_state, strict=True)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise CanonicalContractError(
                    "frozen critic state mismatch: "
                    f"missing={incompatible.missing_keys}, "
                    f"unexpected={incompatible.unexpected_keys}"
                )
            self.critic.eval()
            actor_ids = {id(parameter) for parameter in self.actor.parameters()}
            critic_ids = {id(parameter) for parameter in self.critic.parameters()}
            if actor_ids & critic_ids:
                raise CanonicalContractError(
                    "actor and critic share parameters; frozen-critic isolation failed"
                )
            self._drpo_frozen_critic_reference = _parameter_snapshot(self.critic)
            self._drpo_original_critic_optimizer = self.c_opt
            self.c_opt = FrozenOptimizer(self.c_opt)
            self._drpo_external_advantage_updates = 0
            self._drpo_last_external_advantage_metrics: dict[str, Any] | None = None
            if support_jsonl is not None:
                support_jsonl.unlink(missing_ok=True)
            if support_latest is not None:
                support_latest.unlink(missing_ok=True)

        def _drpo_diagnostic(
            self,
            states: torch.Tensor,
            reconstruction_error: float,
        ) -> dict[str, Any]:
            critic_change = _max_parameter_change(
                self.critic,
                self._drpo_frozen_critic_reference,
            )
            if critic_change != 0.0:
                raise AssertionError(
                    f"frozen critic changed by max_abs={critic_change}"
                )
            with torch.no_grad():
                actor_output = self.actor(states)
                if not isinstance(actor_output, (tuple, list)) or len(actor_output) != 2:
                    raise CanonicalContractError(
                        "support diagnostics require actor(state) -> (mean, log_std)"
                    )
                _, log_std = actor_output
                if not bool(torch.isfinite(log_std).all()):
                    raise FloatingPointError("actor support diagnostics contain NaN/Inf")
                sigma = log_std.exp()
                lower_fraction = float((log_std <= -5.0 + 1e-6).float().mean().cpu())
                upper_fraction = float((log_std >= 2.0 - 1e-6).float().mean().cpu())
                payload = {
                    "schema_version": 1,
                    "status": (
                        "complete"
                        if self._drpo_external_advantage_updates == total_steps
                        else "running"
                    ),
                    "update": self._drpo_external_advantage_updates,
                    "total_steps": total_steps,
                    "advantage_source": source,
                    "critic_frozen": True,
                    "critic_optimizer_step_calls": int(self.c_opt.step_calls),
                    "critic_max_abs_parameter_change": critic_change,
                    "advantage_reconstruction_max_abs_error": reconstruction_error,
                    "actor_log_std_min": float(log_std.min().cpu()),
                    "actor_log_std_max": float(log_std.max().cpu()),
                    "actor_sigma_min": float(sigma.min().cpu()),
                    "actor_sigma_max": float(sigma.max().cpu()),
                    "lower_log_std_boundary_fraction": lower_fraction,
                    "upper_log_std_boundary_fraction": upper_fraction,
                    "support_or_variance_boundary_event": bool(
                        lower_fraction > 0.0 or upper_fraction > 0.0
                    ),
                    "nan_inf_numerical_failure": False,
                }
            if support_jsonl is not None:
                _append_jsonl(support_jsonl, payload)
            if support_latest is not None:
                _atomic_json(support_latest, payload)
            return payload

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
            if ep_ret is None:
                raise CanonicalContractError(
                    "external trajectory advantage is missing from trainer ep_ret slot"
                )
            device = _agent_device(self)
            states = _as_tensor(s, device=device)
            next_states = _as_tensor(ns, device=device)
            dones = _as_tensor(d, device=device, dtype=torch.bool).reshape(-1)
            external = _as_tensor(ep_ret, device=device).reshape(-1)
            if external.shape[0] != states.shape[0]:
                raise CanonicalContractError(
                    "external advantage batch length does not match state batch"
                )
            if not bool(torch.isfinite(external).all()):
                raise FloatingPointError("external advantage contains NaN/Inf")
            self._drpo_external_advantage_updates += 1
            if self._drpo_external_advantage_updates > total_steps:
                raise RuntimeError("external advantage wrapper exceeded registered total_steps")
            with torch.no_grad():
                values = self.critic(states).squeeze(-1)
                next_values = self.critic(next_states).squeeze(-1)
                synthetic_rewards = (
                    external
                    + values
                    - float(self.gamma) * next_values * (~dones).float()
                )
                reconstructed = (
                    synthetic_rewards
                    + float(self.gamma) * next_values * (~dones).float()
                    - values
                )
                reconstruction_error = float(
                    (reconstructed - external).abs().max().cpu()
                )
            if reconstruction_error > 1e-6:
                raise AssertionError(
                    "external advantage reconstruction exceeded tolerance: "
                    f"{reconstruction_error}"
                )

            result = super().update(
                states,
                a,
                synthetic_rewards,
                next_states,
                dones,
                None,
            )
            diagnostic = None
            if (
                self._drpo_external_advantage_updates % diagnostics_interval == 0
                or self._drpo_external_advantage_updates == total_steps
            ):
                diagnostic = self._drpo_diagnostic(states, reconstruction_error)
            metrics = {
                "advantage_source": source,
                "gamma": gamma,
                "gae_lambda": gae_lambda,
                "critic_frozen": True,
                "critic_optimizer_step_mode": "noop",
                "critic_optimizer_step_calls": int(self.c_opt.step_calls),
                "external_advantage_update_index": self._drpo_external_advantage_updates,
                "external_advantage_mean": float(external.mean().detach().cpu()),
                "external_advantage_std": float(
                    external.std(unbiased=False).detach().cpu()
                ),
                "external_advantage_positive_fraction": float(
                    (external > 0).float().mean().cpu()
                ),
                "external_advantage_negative_fraction": float(
                    (external < 0).float().mean().cpu()
                ),
                "external_advantage_zero_fraction": float(
                    (external == 0).float().mean().cpu()
                ),
                "external_advantage_reconstruction_max_abs_error": reconstruction_error,
                "frozen_critic_max_abs_parameter_change": (
                    None
                    if diagnostic is None
                    else diagnostic["critic_max_abs_parameter_change"]
                ),
            }
            if diagnostic is not None:
                metrics.update(diagnostic)
            self._drpo_last_external_advantage_metrics = metrics
            for attribute in (
                "_drpo_last_negative_control_metrics",
                "_drpo_last_ppo_metrics",
            ):
                existing = getattr(self, attribute, None)
                if isinstance(existing, dict):
                    existing.update(metrics)
            if return_mode == "metrics_dict":
                if isinstance(result, dict):
                    return {**result, **metrics}
                return copy.deepcopy(metrics)
            return result

    ExternalAdvantageAgent.__name__ = base_class.__name__
    ExternalAdvantageAgent.__qualname__ = base_class.__qualname__
    ExternalAdvantageAgent.__module__ = base_class.__module__
    ExternalAdvantageAgent.__doc__ = (
        f"Runtime external-advantage wrapper for {base_class.__name__}; "
        f"source={source}, frozen critic."
    )
    return ExternalAdvantageAgent
