"""Analytic Gaussian-KL early refresh for the E7 offline PPO-style actor.

This is not behavior-policy importance correction. It retains the existing
clipped surrogate as an ablation and uses a reference actor only as a proximal
update anchor. The old actor is refreshed after an update whose analytic
``KL(pi_old || pi_new)`` on the current offline states exceeds the registered
threshold. The normal fixed refresh cadence remains a hard upper bound.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import torch

from drpo.e7_canonical_injection import NegativeControl, _agent_device, _as_tensor
from drpo.e7_canonical_ppo_injection import (
    PPOActorControl,
    build_ppo_injected_agent_class,
)


@dataclasses.dataclass(frozen=True)
class PPOKLEarlyRefreshControl:
    """Frozen analytic-KL reference lifecycle settings."""

    target_kl: float = 0.01
    diagnostics_interval: int = 1000

    def validate(self) -> None:
        if not math.isfinite(self.target_kl) or self.target_kl <= 0.0:
            raise ValueError("target_kl must be finite and positive")
        if self.diagnostics_interval <= 0:
            raise ValueError("diagnostics_interval must be positive")


def diagonal_gaussian_kl_old_to_new(
    old_mean: torch.Tensor,
    old_log_std: torch.Tensor,
    new_mean: torch.Tensor,
    new_log_std: torch.Tensor,
) -> torch.Tensor:
    """Return per-state ``KL(N_old || N_new)`` summed over action dimensions."""

    if not (
        old_mean.shape
        == old_log_std.shape
        == new_mean.shape
        == new_log_std.shape
    ):
        raise ValueError("Gaussian KL tensors must have identical shapes")
    tensors = (old_mean, old_log_std, new_mean, new_log_std)
    if not all(bool(torch.isfinite(value).all()) for value in tensors):
        raise FloatingPointError("non-finite Gaussian parameter in KL diagnostic")
    old_log_std = old_log_std.clamp(min=-20.0, max=5.0)
    new_log_std = new_log_std.clamp(min=-20.0, max=5.0)
    old_var = (2.0 * old_log_std).exp()
    new_var = (2.0 * new_log_std).exp().clamp_min(1e-16)
    per_dimension = (
        new_log_std
        - old_log_std
        + (old_var + (old_mean - new_mean).square()) / (2.0 * new_var)
        - 0.5
    )
    value = per_dimension.sum(dim=-1)
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError("non-finite analytic Gaussian KL")
    return value.clamp_min(0.0)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True) + "\n")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def build_ppo_kl_refresh_agent_class(
    base_class: type,
    *,
    negative_control: NegativeControl,
    ppo_control: PPOActorControl,
    kl_control: PPOKLEarlyRefreshControl,
    return_mode: str,
    ppo_diagnostics_jsonl: str | Path | None = None,
    ppo_diagnostics_latest: str | Path | None = None,
    kl_diagnostics_jsonl: str | Path | None = None,
    kl_diagnostics_latest: str | Path | None = None,
) -> type:
    """Build the existing PPO actor plus analytic-KL-triggered early refresh."""

    ppo_control.validate()
    kl_control.validate()
    injected = build_ppo_injected_agent_class(
        base_class,
        negative_control=negative_control,
        ppo_control=ppo_control,
        return_mode=return_mode,
        diagnostics_jsonl=ppo_diagnostics_jsonl,
        diagnostics_latest=ppo_diagnostics_latest,
    )
    kl_jsonl = (
        None
        if kl_diagnostics_jsonl is None
        else Path(kl_diagnostics_jsonl).resolve()
    )
    kl_latest = (
        None
        if kl_diagnostics_latest is None
        else Path(kl_diagnostics_latest).resolve()
    )

    class CanonicalPPOKLEarlyRefreshAgent(injected):  # type: ignore[misc, valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._drpo_kl_updates_since_refresh = 0
            self._drpo_kl_triggered_refresh_count = 0
            self._drpo_kl_sum = 0.0
            self._drpo_kl_max = 0.0
            self._drpo_kl_interval_updates = 0
            self._drpo_last_analytic_kl = 0.0
            if kl_jsonl is not None:
                kl_jsonl.unlink(missing_ok=True)
            if kl_latest is not None:
                kl_latest.unlink(missing_ok=True)

        def _drpo_write_kl_diagnostics(
            self,
            *,
            update_index: int,
            mean_kl: float,
            triggered: bool,
            scheduled_refresh_before_update: bool,
        ) -> None:
            interval_updates = self._drpo_kl_interval_updates
            if interval_updates <= 0:
                raise RuntimeError("cannot write an empty KL diagnostics interval")
            payload = {
                "schema_version": 1,
                "status": (
                    "complete"
                    if update_index == ppo_control.total_steps
                    else "running"
                ),
                "update": update_index,
                "total_steps": ppo_control.total_steps,
                "target_kl": kl_control.target_kl,
                "kl_direction": "old_to_new",
                "analytic_kl_current_batch": mean_kl,
                "interval_updates": interval_updates,
                "interval_analytic_kl_mean": self._drpo_kl_sum / interval_updates,
                "interval_analytic_kl_max": self._drpo_kl_max,
                "kl_triggered_refresh": triggered,
                "kl_triggered_refresh_count": self._drpo_kl_triggered_refresh_count,
                "scheduled_refresh_before_update": scheduled_refresh_before_update,
                "old_policy_refresh_count": self._drpo_old_policy_refresh_count,
                "updates_since_old_policy_refresh": (
                    self._drpo_kl_updates_since_refresh
                ),
                "max_updates_per_old_policy": ppo_control.updates_per_old_policy,
                "clip_epsilon": ppo_control.clip_epsilon,
            }
            if kl_jsonl is not None:
                _append_jsonl(kl_jsonl, payload)
            if kl_latest is not None:
                _atomic_json(kl_latest, payload)
            self._drpo_kl_sum = 0.0
            self._drpo_kl_max = 0.0
            self._drpo_kl_interval_updates = 0

        def update(
            self,
            s: Any,
            a: Any,
            r: Any,
            ns: Any,
            d: Any,
            ep_ret: Any = None,
        ) -> Any:
            refresh_count_before = self._drpo_old_policy_refresh_count
            result = super().update(s, a, r, ns, d, ep_ret)
            update_index = self._drpo_ppo_update_count
            scheduled_refresh = (
                self._drpo_old_policy_refresh_count > refresh_count_before
            )
            if scheduled_refresh:
                # The scheduled copy occurs before this optimizer update.
                self._drpo_kl_updates_since_refresh = 1
            else:
                self._drpo_kl_updates_since_refresh += 1

            states = _as_tensor(s, device=_agent_device(self))
            with torch.no_grad():
                old_mean, old_log_std = self._drpo_old_actor(states)
                new_mean, new_log_std = self.actor(states)
                mean_kl = float(
                    diagonal_gaussian_kl_old_to_new(
                        old_mean,
                        old_log_std,
                        new_mean,
                        new_log_std,
                    )
                    .mean()
                    .cpu()
                )
            if not math.isfinite(mean_kl):
                raise FloatingPointError("non-finite mean analytic Gaussian KL")
            self._drpo_last_analytic_kl = mean_kl
            self._drpo_kl_sum += mean_kl
            self._drpo_kl_max = max(self._drpo_kl_max, mean_kl)
            self._drpo_kl_interval_updates += 1

            triggered = mean_kl > kl_control.target_kl
            if triggered:
                self._drpo_refresh_old_actor()
                self._drpo_kl_triggered_refresh_count += 1
                self._drpo_kl_updates_since_refresh = 0

            should_write = (
                triggered
                or update_index % kl_control.diagnostics_interval == 0
                or update_index == ppo_control.total_steps
            )
            if should_write:
                self._drpo_write_kl_diagnostics(
                    update_index=update_index,
                    mean_kl=mean_kl,
                    triggered=triggered,
                    scheduled_refresh_before_update=scheduled_refresh,
                )
            if isinstance(self._drpo_last_ppo_metrics, dict):
                self._drpo_last_ppo_metrics.update(
                    {
                        "analytic_kl_old_to_new": mean_kl,
                        "kl_target": kl_control.target_kl,
                        "kl_triggered_refresh": triggered,
                        "kl_triggered_refresh_count": (
                            self._drpo_kl_triggered_refresh_count
                        ),
                    }
                )
            return result

    CanonicalPPOKLEarlyRefreshAgent.__name__ = base_class.__name__
    CanonicalPPOKLEarlyRefreshAgent.__qualname__ = base_class.__qualname__
    CanonicalPPOKLEarlyRefreshAgent.__module__ = base_class.__module__
    return CanonicalPPOKLEarlyRefreshAgent


def patch_canonical_module_ppo_kl_refresh(
    module: Any,
    target_class: str,
    *,
    negative_control: NegativeControl,
    ppo_control: PPOActorControl,
    kl_control: PPOKLEarlyRefreshControl,
    return_mode: str,
    ppo_diagnostics_jsonl: str | Path | None = None,
    ppo_diagnostics_latest: str | Path | None = None,
    kl_diagnostics_jsonl: str | Path | None = None,
    kl_diagnostics_latest: str | Path | None = None,
) -> type:
    """Replace the configured canonical class with the KL-refresh PPO subclass."""

    if not hasattr(module, target_class):
        raise AttributeError(f"canonical module has no target class {target_class!r}")
    original = getattr(module, target_class)
    if not isinstance(original, type):
        raise TypeError(f"canonical target {target_class!r} is not a class")
    injected = build_ppo_kl_refresh_agent_class(
        original,
        negative_control=negative_control,
        ppo_control=ppo_control,
        kl_control=kl_control,
        return_mode=return_mode,
        ppo_diagnostics_jsonl=ppo_diagnostics_jsonl,
        ppo_diagnostics_latest=ppo_diagnostics_latest,
        kl_diagnostics_jsonl=kl_diagnostics_jsonl,
        kl_diagnostics_latest=kl_diagnostics_latest,
    )
    setattr(module, target_class, injected)
    return injected
