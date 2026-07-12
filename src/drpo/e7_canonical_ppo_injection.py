"""PPO-clipped actor injection for the canonical E7 D4RL framework.

The module deliberately reuses the existing canonical E7 source contract,
policy-relative distance, negative-control transform, critic target, and
expectile loss. Only the actor surrogate and old-policy snapshot lifecycle are
changed.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import torch

from drpo.e7_canonical_injection import (
    CanonicalContractError,
    NegativeControl,
    _agent_device,
    _as_tensor,
    controlled_advantage,
    detached_standardized_distance,
    validate_agent_instance,
)


@dataclasses.dataclass(frozen=True)
class PPOActorControl:
    """Frozen actor-update settings for the E7 PPO stability pilot."""

    clip_epsilon: float = 0.2
    updates_per_old_policy: int = 4
    diagnostics_interval: int = 1000
    total_steps: int = 1_000_000

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "PPOActorControl":
        control = cls(
            clip_epsilon=float(raw.get("clip_epsilon", 0.2)),
            updates_per_old_policy=int(raw.get("updates_per_old_policy", 4)),
            diagnostics_interval=int(raw.get("diagnostics_interval", 1000)),
            total_steps=int(raw.get("total_steps", 1_000_000)),
        )
        control.validate()
        return control

    def validate(self) -> None:
        if not math.isfinite(self.clip_epsilon):
            raise ValueError("clip_epsilon must be finite")
        if not (0.0 < self.clip_epsilon < 1.0):
            raise ValueError("clip_epsilon must be in (0, 1)")
        if self.updates_per_old_policy < 2:
            raise ValueError("updates_per_old_policy must be at least 2")
        if self.diagnostics_interval <= 0:
            raise ValueError("diagnostics_interval must be positive")
        if self.total_steps <= 0:
            raise ValueError("total_steps must be positive")


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _gradient_l2(parameters: list[torch.nn.Parameter]) -> float:
    total = torch.zeros((), device=parameters[0].device)
    for parameter in parameters:
        if parameter.grad is not None:
            total = total + parameter.grad.detach().square().sum()
    return float(total.sqrt().cpu())


def _parameter_l2(parameters: list[torch.nn.Parameter]) -> float:
    total = torch.zeros((), device=parameters[0].device)
    for parameter in parameters:
        total = total + parameter.detach().square().sum()
    return float(total.sqrt().cpu())


def _parameter_update_l2(
    parameters: list[torch.nn.Parameter],
    before: list[torch.Tensor],
) -> float:
    total = torch.zeros((), device=parameters[0].device)
    for parameter, prior in zip(parameters, before, strict=True):
        total = total + (parameter.detach() - prior).square().sum()
    return float(total.sqrt().cpu())


def _quantile(tensor: torch.Tensor, q: float) -> float:
    flat = tensor.detach().reshape(-1)
    if flat.numel() == 0:
        return float("nan")
    return float(torch.quantile(flat, q).cpu())


def _finite_tensor(name: str, tensor: torch.Tensor) -> None:
    if not bool(torch.isfinite(tensor).all()):
        raise FloatingPointError(f"non-finite PPO tensor: {name}")


def _new_bucket() -> dict[str, Any]:
    return {
        "updates": 0,
        "samples": 0,
        "ratio_sum": 0.0,
        "ratio_min": float("inf"),
        "ratio_max": float("-inf"),
        "abs_log_ratio_sum": 0.0,
        "abs_log_ratio_max": 0.0,
        "ratio_outside_count": 0,
        "objective_clip_count": 0,
        "positive_count": 0,
        "positive_clip_count": 0,
        "negative_count": 0,
        "negative_clip_count": 0,
    }


def _update_bucket(
    bucket: dict[str, Any],
    *,
    ratio: torch.Tensor,
    log_ratio: torch.Tensor,
    weighted_advantage: torch.Tensor,
    clip_epsilon: float,
) -> None:
    detached_ratio = ratio.detach().reshape(-1)
    detached_log_ratio = log_ratio.detach().reshape(-1)
    detached_advantage = weighted_advantage.detach().reshape(-1)
    sample_count = int(detached_ratio.numel())
    positive = detached_advantage > 0
    negative = detached_advantage < 0
    outside = (detached_ratio < 1.0 - clip_epsilon) | (
        detached_ratio > 1.0 + clip_epsilon
    )
    objective_clip = (positive & (detached_ratio > 1.0 + clip_epsilon)) | (
        negative & (detached_ratio < 1.0 - clip_epsilon)
    )

    bucket["updates"] += 1
    bucket["samples"] += sample_count
    bucket["ratio_sum"] += float(detached_ratio.sum().cpu())
    bucket["ratio_min"] = min(
        bucket["ratio_min"], float(detached_ratio.min().cpu())
    )
    bucket["ratio_max"] = max(
        bucket["ratio_max"], float(detached_ratio.max().cpu())
    )
    abs_log_ratio = detached_log_ratio.abs()
    bucket["abs_log_ratio_sum"] += float(abs_log_ratio.sum().cpu())
    bucket["abs_log_ratio_max"] = max(
        bucket["abs_log_ratio_max"], float(abs_log_ratio.max().cpu())
    )
    bucket["ratio_outside_count"] += int(outside.sum().cpu())
    bucket["objective_clip_count"] += int(objective_clip.sum().cpu())
    bucket["positive_count"] += int(positive.sum().cpu())
    bucket["positive_clip_count"] += int((objective_clip & positive).sum().cpu())
    bucket["negative_count"] += int(negative.sum().cpu())
    bucket["negative_clip_count"] += int((objective_clip & negative).sum().cpu())


def _finalize_bucket(bucket: Mapping[str, Any]) -> dict[str, Any]:
    samples = int(bucket["samples"])
    positive_count = int(bucket["positive_count"])
    negative_count = int(bucket["negative_count"])
    if samples <= 0:
        raise RuntimeError("cannot finalize an empty PPO diagnostics bucket")
    return {
        "updates": int(bucket["updates"]),
        "samples": samples,
        "ratio_mean": float(bucket["ratio_sum"]) / samples,
        "ratio_min": float(bucket["ratio_min"]),
        "ratio_max": float(bucket["ratio_max"]),
        "abs_log_ratio_mean": float(bucket["abs_log_ratio_sum"]) / samples,
        "abs_log_ratio_max": float(bucket["abs_log_ratio_max"]),
        "ratio_outside_fraction": int(bucket["ratio_outside_count"]) / samples,
        "objective_clip_fraction": int(bucket["objective_clip_count"]) / samples,
        "positive_samples": positive_count,
        "positive_objective_clip_fraction": (
            int(bucket["positive_clip_count"]) / positive_count
            if positive_count
            else None
        ),
        "negative_samples": negative_count,
        "negative_objective_clip_fraction": (
            int(bucket["negative_clip_count"]) / negative_count
            if negative_count
            else None
        ),
    }


class _IntervalDiagnostics:
    def __init__(self, updates_per_old_policy: int) -> None:
        self.overall = _new_bucket()
        self.by_block_position = {
            position: _new_bucket()
            for position in range(1, updates_per_old_policy + 1)
        }
        self.start_update: int | None = None

    def add(
        self,
        *,
        update: int,
        block_position: int,
        ratio: torch.Tensor,
        log_ratio: torch.Tensor,
        weighted_advantage: torch.Tensor,
        clip_epsilon: float,
    ) -> None:
        if self.start_update is None:
            self.start_update = update
        _update_bucket(
            self.overall,
            ratio=ratio,
            log_ratio=log_ratio,
            weighted_advantage=weighted_advantage,
            clip_epsilon=clip_epsilon,
        )
        _update_bucket(
            self.by_block_position[block_position],
            ratio=ratio,
            log_ratio=log_ratio,
            weighted_advantage=weighted_advantage,
            clip_epsilon=clip_epsilon,
        )

    def payload(self, end_update: int) -> dict[str, Any]:
        if self.start_update is None:
            raise RuntimeError("PPO diagnostics interval has no updates")
        return {
            "interval_start_update": self.start_update,
            "interval_end_update": end_update,
            "pre_update": _finalize_bucket(self.overall),
            "pre_update_by_block_position": {
                str(position): _finalize_bucket(bucket)
                for position, bucket in self.by_block_position.items()
                if bucket["samples"] > 0
            },
        }


def build_ppo_injected_agent_class(
    base_class: type,
    *,
    negative_control: NegativeControl,
    ppo_control: PPOActorControl,
    return_mode: str,
    diagnostics_jsonl: str | Path | None = None,
    diagnostics_latest: str | Path | None = None,
) -> type:
    """Build a canonical agent subclass that changes only the actor surrogate."""

    ppo_control.validate()
    if return_mode not in {"zero_float", "metrics_dict"}:
        raise ValueError(f"unsupported return_mode={return_mode!r}")
    diagnostics_jsonl_path = (
        None if diagnostics_jsonl is None else Path(diagnostics_jsonl).resolve()
    )
    diagnostics_latest_path = (
        None if diagnostics_latest is None else Path(diagnostics_latest).resolve()
    )

    class CanonicalPPOControlAgent(base_class):  # type: ignore[misc, valid-type]
        _drpo_negative_control = negative_control
        _drpo_ppo_control = ppo_control

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            validate_agent_instance(
                self,
                expected_alpha=negative_control.canonical_alpha,
            )
            self._drpo_old_actor = copy.deepcopy(self.actor)
            self._drpo_old_actor.eval()
            for parameter in self._drpo_old_actor.parameters():
                parameter.requires_grad_(False)
            self._drpo_ppo_update_count = 0
            self._drpo_old_policy_refresh_count = 0
            self._drpo_ppo_interval = _IntervalDiagnostics(
                ppo_control.updates_per_old_policy
            )
            self._drpo_last_ppo_metrics: dict[str, Any] | None = None

        def _drpo_refresh_old_actor(self) -> None:
            self._drpo_old_actor.load_state_dict(self.actor.state_dict())
            self._drpo_old_actor.eval()
            self._drpo_old_policy_refresh_count += 1

        def _drpo_write_diagnostics(self, payload: Mapping[str, Any]) -> None:
            if diagnostics_jsonl_path is not None:
                _append_jsonl(diagnostics_jsonl_path, payload)
            if diagnostics_latest_path is not None:
                _atomic_write_json(diagnostics_latest_path, payload)

        def update(
            self,
            s: Any,
            a: Any,
            r: Any,
            ns: Any,
            d: Any,
            ep_ret: Any = None,
        ) -> Any:
            del ep_ret
            validate_agent_instance(
                self,
                expected_alpha=negative_control.canonical_alpha,
            )
            self._drpo_ppo_update_count += 1
            update_index = self._drpo_ppo_update_count
            if update_index > ppo_control.total_steps:
                raise RuntimeError(
                    "PPO actor received more updates than the registered total_steps"
                )
            if (
                update_index > 1
                and (update_index - 1) % ppo_control.updates_per_old_policy == 0
            ):
                self._drpo_refresh_old_actor()
            block_position = (
                (update_index - 1) % ppo_control.updates_per_old_policy
            ) + 1

            device = _agent_device(self)
            states = _as_tensor(s, device=device)
            actions = _as_tensor(a, device=device)
            rewards = _as_tensor(r, device=device).reshape(-1)
            next_states = _as_tensor(ns, device=device)
            dones = _as_tensor(d, device=device, dtype=torch.bool).reshape(-1)

            values = self.critic(states).squeeze(-1)
            with torch.no_grad():
                next_values = self.critic(next_states).squeeze(-1)
                targets = rewards + float(self.gamma) * next_values * (~dones).float()
            advantages = targets - values.detach()

            mean, log_std, distance = detached_standardized_distance(
                self.actor,
                states,
                actions,
            )
            current_distribution = torch.distributions.Normal(mean, log_std.exp())
            current_log_prob = current_distribution.log_prob(actions).sum(dim=-1)
            with torch.no_grad():
                old_mean, old_log_std = self._drpo_old_actor(states)
                old_distribution = torch.distributions.Normal(
                    old_mean,
                    old_log_std.exp(),
                )
                old_log_prob = old_distribution.log_prob(actions).sum(dim=-1)

            weighted_advantage, factor = controlled_advantage(
                advantages,
                distance,
                negative_control,
            )
            weighted_advantage = weighted_advantage.detach()
            log_ratio = current_log_prob - old_log_prob
            _finite_tensor("log_ratio", log_ratio)
            ratio = log_ratio.exp()
            _finite_tensor("ratio", ratio)
            clipped_ratio = ratio.clamp(
                1.0 - ppo_control.clip_epsilon,
                1.0 + ppo_control.clip_epsilon,
            )
            unclipped_surrogate = ratio * weighted_advantage
            clipped_surrogate = clipped_ratio * weighted_advantage
            actor_loss = -torch.minimum(
                unclipped_surrogate,
                clipped_surrogate,
            ).mean()
            _finite_tensor("actor_loss", actor_loss)

            diagnostic_step = (
                update_index % ppo_control.diagnostics_interval == 0
                or update_index == ppo_control.total_steps
            )
            actor_parameters = list(self.actor.parameters())
            before_parameters = (
                [parameter.detach().clone() for parameter in actor_parameters]
                if diagnostic_step
                else None
            )
            parameter_norm_before = (
                _parameter_l2(actor_parameters) if diagnostic_step else None
            )

            self.a_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_gradient_norm = (
                _gradient_l2(actor_parameters) if diagnostic_step else None
            )
            self.a_opt.step()

            actor_update_norm = None
            actor_relative_update_norm = None
            post_diagnostics: dict[str, Any] | None = None
            if diagnostic_step:
                assert before_parameters is not None
                assert parameter_norm_before is not None
                actor_update_norm = _parameter_update_l2(
                    actor_parameters,
                    before_parameters,
                )
                actor_relative_update_norm = actor_update_norm / max(
                    parameter_norm_before,
                    1e-12,
                )
                with torch.no_grad():
                    post_mean, post_log_std = self.actor(states)
                    post_distribution = torch.distributions.Normal(
                        post_mean,
                        post_log_std.exp(),
                    )
                    post_log_prob = post_distribution.log_prob(actions).sum(dim=-1)
                    post_ratio_to_old = (post_log_prob - old_log_prob).exp()
                    single_step_ratio = (post_log_prob - current_log_prob.detach()).exp()
                    _finite_tensor("post_ratio_to_old", post_ratio_to_old)
                    _finite_tensor("single_step_ratio", single_step_ratio)
                    post_diagnostics = {
                        "ratio_to_old_p01": _quantile(post_ratio_to_old, 0.01),
                        "ratio_to_old_p10": _quantile(post_ratio_to_old, 0.10),
                        "ratio_to_old_p50": _quantile(post_ratio_to_old, 0.50),
                        "ratio_to_old_p90": _quantile(post_ratio_to_old, 0.90),
                        "ratio_to_old_p99": _quantile(post_ratio_to_old, 0.99),
                        "ratio_to_old_min": float(post_ratio_to_old.min().cpu()),
                        "ratio_to_old_max": float(post_ratio_to_old.max().cpu()),
                        "ratio_to_old_outside_fraction": float(
                            (
                                (post_ratio_to_old < 1.0 - ppo_control.clip_epsilon)
                                | (
                                    post_ratio_to_old
                                    > 1.0 + ppo_control.clip_epsilon
                                )
                            )
                            .float()
                            .mean()
                            .cpu()
                        ),
                        "single_step_ratio_p01": _quantile(single_step_ratio, 0.01),
                        "single_step_ratio_p10": _quantile(single_step_ratio, 0.10),
                        "single_step_ratio_p50": _quantile(single_step_ratio, 0.50),
                        "single_step_ratio_p90": _quantile(single_step_ratio, 0.90),
                        "single_step_ratio_p99": _quantile(single_step_ratio, 0.99),
                        "single_step_ratio_min": float(single_step_ratio.min().cpu()),
                        "single_step_ratio_max": float(single_step_ratio.max().cpu()),
                    }

            value_error = targets - values
            expectile = torch.where(
                value_error > 0,
                torch.full_like(value_error, float(self.tau)),
                torch.full_like(value_error, 1.0 - float(self.tau)),
            )
            critic_loss = (expectile * value_error.square()).mean()
            _finite_tensor("critic_loss", critic_loss)
            self.c_opt.zero_grad(set_to_none=True)
            critic_loss.backward()
            self.c_opt.step()

            self._drpo_ppo_interval.add(
                update=update_index,
                block_position=block_position,
                ratio=ratio,
                log_ratio=log_ratio,
                weighted_advantage=weighted_advantage,
                clip_epsilon=ppo_control.clip_epsilon,
            )

            negative = advantages < 0
            metrics: dict[str, Any] = {
                "actor_loss": float(actor_loss.detach().cpu()),
                "critic_loss": float(critic_loss.detach().cpu()),
                "positive_fraction": float((~negative).float().mean().cpu()),
                "negative_fraction": float(negative.float().mean().cpu()),
                "negative_factor_mean": (
                    float(factor[negative].mean().detach().cpu())
                    if bool(negative.any())
                    else float("nan")
                ),
                "negative_distance_mean": (
                    float(distance[negative].mean().detach().cpu())
                    if bool(negative.any())
                    else float("nan")
                ),
                "canonical_alpha": negative_control.canonical_alpha,
                "negative_scale": negative_control.negative_scale,
                "effective_alpha": negative_control.effective_alpha,
                "method": negative_control.method,
                "actor_update_mode": "ppo_clip",
                "clip_epsilon": ppo_control.clip_epsilon,
                "updates_per_old_policy": ppo_control.updates_per_old_policy,
                "ppo_update_index": update_index,
                "ppo_block_position": block_position,
                "old_policy_refresh_count": self._drpo_old_policy_refresh_count,
            }
            self._drpo_last_negative_control_metrics = metrics
            self._drpo_last_ppo_metrics = metrics

            if diagnostic_step:
                interval_payload = self._drpo_ppo_interval.payload(update_index)
                sampled_pre = {
                    "ratio_p01": _quantile(ratio, 0.01),
                    "ratio_p10": _quantile(ratio, 0.10),
                    "ratio_p50": _quantile(ratio, 0.50),
                    "ratio_p90": _quantile(ratio, 0.90),
                    "ratio_p99": _quantile(ratio, 0.99),
                    "abs_log_ratio_p50": _quantile(log_ratio.abs(), 0.50),
                    "abs_log_ratio_p90": _quantile(log_ratio.abs(), 0.90),
                    "abs_log_ratio_p99": _quantile(log_ratio.abs(), 0.99),
                }
                payload = {
                    "schema_version": 1,
                    "status": (
                        "complete"
                        if update_index == ppo_control.total_steps
                        else "running"
                    ),
                    "update": update_index,
                    "total_steps": ppo_control.total_steps,
                    "block_position": block_position,
                    "old_policy_refresh_count": self._drpo_old_policy_refresh_count,
                    "ppo_control": dataclasses.asdict(ppo_control),
                    "negative_control": dataclasses.asdict(negative_control),
                    **interval_payload,
                    "sampled_pre_update": sampled_pre,
                    "sampled_post_update": post_diagnostics,
                    "actor_gradient_norm": actor_gradient_norm,
                    "actor_parameter_update_norm": actor_update_norm,
                    "actor_relative_parameter_update_norm": actor_relative_update_norm,
                }
                self._drpo_write_diagnostics(payload)
                self._drpo_ppo_interval = _IntervalDiagnostics(
                    ppo_control.updates_per_old_policy
                )

            if return_mode == "metrics_dict":
                return metrics
            return 0.0

    CanonicalPPOControlAgent.__name__ = base_class.__name__
    CanonicalPPOControlAgent.__qualname__ = base_class.__qualname__
    CanonicalPPOControlAgent.__module__ = base_class.__module__
    CanonicalPPOControlAgent.__doc__ = (
        f"Runtime-injected PPO {base_class.__name__} for "
        f"method={negative_control.method}, "
        f"negative_scale={negative_control.negative_scale}."
    )
    return CanonicalPPOControlAgent


def patch_canonical_module_ppo(
    module: Any,
    target_class: str,
    *,
    negative_control: NegativeControl,
    ppo_control: PPOActorControl,
    return_mode: str,
    diagnostics_jsonl: str | Path | None = None,
    diagnostics_latest: str | Path | None = None,
) -> type:
    """Replace only the configured canonical class with the PPO subclass."""

    if not hasattr(module, target_class):
        raise CanonicalContractError(
            f"canonical module has no target class {target_class!r}"
        )
    original = getattr(module, target_class)
    if not isinstance(original, type):
        raise CanonicalContractError(
            f"canonical target {target_class!r} is not a class"
        )
    injected = build_ppo_injected_agent_class(
        original,
        negative_control=negative_control,
        ppo_control=ppo_control,
        return_mode=return_mode,
        diagnostics_jsonl=diagnostics_jsonl,
        diagnostics_latest=diagnostics_latest,
    )
    setattr(module, target_class, injected)
    return injected
