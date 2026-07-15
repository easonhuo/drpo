"""Actor injections that consume prepared TD/GAE advantages.

The sixth historical ``update`` argument is repurposed only inside this
experiment as a prepared per-transition advantage.  The shared critic is loaded
for provenance and evaluation diagnostics but is frozen: all branch differences
are actor-side differences.
"""

from __future__ import annotations

import copy
import dataclasses
from pathlib import Path
from typing import Any, Mapping

import torch

from drpo import e7_canonical_injection as canonical_injection
from drpo.e7_canonical_injection import (
    NegativeControl,
    _agent_device,
    _as_tensor,
    detached_standardized_distance,
    validate_agent_instance,
)
from drpo.e7_canonical_ppo_injection import (
    PPOActorControl,
    _IntervalDiagnostics,
    _append_jsonl,
    _atomic_write_json,
    _finite_tensor,
    _gradient_l2,
    _parameter_l2,
    _parameter_update_l2,
    _quantile,
)


def _prepared_advantage(value: Any, *, device: torch.device) -> torch.Tensor:
    if value is None:
        raise ValueError("prepared advantage is required in the sixth update argument")
    advantage = _as_tensor(value, device=device).reshape(-1).detach()
    if not bool(torch.isfinite(advantage).all()):
        raise FloatingPointError("prepared advantage contains NaN/Inf")
    return advantage


def _freeze_critic(agent: Any) -> None:
    agent.critic.eval()
    for parameter in agent.critic.parameters():
        parameter.requires_grad_(False)


def _common_metrics(
    *,
    actor_loss: torch.Tensor,
    advantages: torch.Tensor,
    factor: torch.Tensor,
    distance: torch.Tensor,
    control: NegativeControl,
    actor_update_mode: str,
) -> dict[str, Any]:
    negative = advantages < 0
    return {
        "actor_loss": float(actor_loss.detach().cpu()),
        "critic_loss": None,
        "critic_frozen": True,
        "advantage_source": "prepared_shared_frozen_critic",
        "positive_fraction": float((advantages > 0).float().mean().cpu()),
        "zero_fraction": float((advantages == 0).float().mean().cpu()),
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
        "canonical_alpha": control.canonical_alpha,
        "negative_scale": control.negative_scale,
        "effective_alpha": control.effective_alpha,
        "method": control.method,
        "actor_update_mode": actor_update_mode,
    }


def build_precomputed_a2c_agent_class(
    base_class: type,
    *,
    negative_control: NegativeControl,
    return_mode: str,
) -> type:
    if return_mode not in {"zero_float", "metrics_dict"}:
        raise ValueError(f"unsupported return_mode={return_mode!r}")

    class PrecomputedAdvantageA2CAgent(base_class):  # type: ignore[misc, valid-type]
        _drpo_negative_control = negative_control

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            validate_agent_instance(
                self, expected_alpha=negative_control.canonical_alpha
            )
            _freeze_critic(self)

        def update(
            self,
            s: Any,
            a: Any,
            r: Any,
            ns: Any,
            d: Any,
            ep_ret: Any = None,
        ) -> Any:
            del r, ns, d
            validate_agent_instance(
                self, expected_alpha=negative_control.canonical_alpha
            )
            device = _agent_device(self)
            states = _as_tensor(s, device=device)
            actions = _as_tensor(a, device=device)
            advantages = _prepared_advantage(ep_ret, device=device)
            if advantages.shape[0] != states.shape[0]:
                raise ValueError("prepared advantage batch does not match states")
            mean, log_std, distance = detached_standardized_distance(
                self.actor, states, actions
            )
            distribution = torch.distributions.Normal(mean, log_std.exp())
            log_prob = distribution.log_prob(actions).sum(dim=-1)
            weighted_advantage, factor = canonical_injection.controlled_advantage(
                advantages, distance, negative_control
            )
            weighted_advantage = weighted_advantage.detach()
            actor_loss = -(log_prob * weighted_advantage).mean()
            _finite_tensor("actor_loss", actor_loss)
            self.a_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.a_opt.step()
            metrics = _common_metrics(
                actor_loss=actor_loss,
                advantages=advantages,
                factor=factor,
                distance=distance,
                control=negative_control,
                actor_update_mode="a2c",
            )
            self._drpo_last_negative_control_metrics = metrics
            if return_mode == "metrics_dict":
                return metrics
            return 0.0

    PrecomputedAdvantageA2CAgent.__name__ = base_class.__name__
    PrecomputedAdvantageA2CAgent.__qualname__ = base_class.__qualname__
    PrecomputedAdvantageA2CAgent.__module__ = base_class.__module__
    return PrecomputedAdvantageA2CAgent


def build_precomputed_ppo_agent_class(
    base_class: type,
    *,
    negative_control: NegativeControl,
    ppo_control: PPOActorControl,
    return_mode: str,
    diagnostics_jsonl: str | Path | None = None,
    diagnostics_latest: str | Path | None = None,
) -> type:
    ppo_control.validate()
    if return_mode not in {"zero_float", "metrics_dict"}:
        raise ValueError(f"unsupported return_mode={return_mode!r}")
    diagnostics_jsonl_path = (
        None if diagnostics_jsonl is None else Path(diagnostics_jsonl).resolve()
    )
    diagnostics_latest_path = (
        None if diagnostics_latest is None else Path(diagnostics_latest).resolve()
    )

    class PrecomputedAdvantagePPOAgent(base_class):  # type: ignore[misc, valid-type]
        _drpo_negative_control = negative_control
        _drpo_ppo_control = ppo_control

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            validate_agent_instance(
                self, expected_alpha=negative_control.canonical_alpha
            )
            _freeze_critic(self)
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
            del r, ns, d
            validate_agent_instance(
                self, expected_alpha=negative_control.canonical_alpha
            )
            self._drpo_ppo_update_count += 1
            update_index = self._drpo_ppo_update_count
            if update_index > ppo_control.total_steps:
                raise RuntimeError(
                    "PPO actor received more updates than registered total_steps"
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
            advantages = _prepared_advantage(ep_ret, device=device)
            if advantages.shape[0] != states.shape[0]:
                raise ValueError("prepared advantage batch does not match states")

            mean, log_std, distance = detached_standardized_distance(
                self.actor, states, actions
            )
            current_distribution = torch.distributions.Normal(mean, log_std.exp())
            current_log_prob = current_distribution.log_prob(actions).sum(dim=-1)
            with torch.no_grad():
                old_mean, old_log_std = self._drpo_old_actor(states)
                old_distribution = torch.distributions.Normal(
                    old_mean, old_log_std.exp()
                )
                old_log_prob = old_distribution.log_prob(actions).sum(dim=-1)

            weighted_advantage, factor = canonical_injection.controlled_advantage(
                advantages, distance, negative_control
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
            actor_loss = -torch.minimum(
                ratio * weighted_advantage,
                clipped_ratio * weighted_advantage,
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
                    actor_parameters, before_parameters
                )
                actor_relative_update_norm = actor_update_norm / max(
                    parameter_norm_before, 1e-12
                )
                with torch.no_grad():
                    post_mean, post_log_std = self.actor(states)
                    post_distribution = torch.distributions.Normal(
                        post_mean, post_log_std.exp()
                    )
                    post_log_prob = post_distribution.log_prob(actions).sum(dim=-1)
                    post_ratio_to_old = (post_log_prob - old_log_prob).exp()
                    single_step_ratio = (
                        post_log_prob - current_log_prob.detach()
                    ).exp()
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

            self._drpo_ppo_interval.add(
                update=update_index,
                block_position=block_position,
                ratio=ratio,
                log_ratio=log_ratio,
                weighted_advantage=weighted_advantage,
                clip_epsilon=ppo_control.clip_epsilon,
            )
            metrics = _common_metrics(
                actor_loss=actor_loss,
                advantages=advantages,
                factor=factor,
                distance=distance,
                control=negative_control,
                actor_update_mode="ppo_clip",
            )
            metrics.update(
                {
                    "clip_epsilon": ppo_control.clip_epsilon,
                    "updates_per_old_policy": ppo_control.updates_per_old_policy,
                    "ppo_update_index": update_index,
                    "ppo_block_position": block_position,
                    "old_policy_refresh_count": self._drpo_old_policy_refresh_count,
                }
            )
            self._drpo_last_negative_control_metrics = metrics
            self._drpo_last_ppo_metrics = metrics

            if diagnostic_step:
                interval_payload = self._drpo_ppo_interval.payload(update_index)
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
                    "critic_frozen": True,
                    "advantage_source": "prepared_shared_frozen_critic",
                    **interval_payload,
                    "sampled_pre_update": {
                        "ratio_p01": _quantile(ratio, 0.01),
                        "ratio_p10": _quantile(ratio, 0.10),
                        "ratio_p50": _quantile(ratio, 0.50),
                        "ratio_p90": _quantile(ratio, 0.90),
                        "ratio_p99": _quantile(ratio, 0.99),
                        "abs_log_ratio_p50": _quantile(log_ratio.abs(), 0.50),
                        "abs_log_ratio_p90": _quantile(log_ratio.abs(), 0.90),
                        "abs_log_ratio_p99": _quantile(log_ratio.abs(), 0.99),
                    },
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

    PrecomputedAdvantagePPOAgent.__name__ = base_class.__name__
    PrecomputedAdvantagePPOAgent.__qualname__ = base_class.__qualname__
    PrecomputedAdvantagePPOAgent.__module__ = base_class.__module__
    return PrecomputedAdvantagePPOAgent


def patch_canonical_module_precomputed_a2c(
    module: Any,
    target_class: str,
    *,
    negative_control: NegativeControl,
    return_mode: str,
) -> type:
    original = getattr(module, target_class)
    injected = build_precomputed_a2c_agent_class(
        original,
        negative_control=negative_control,
        return_mode=return_mode,
    )
    setattr(module, target_class, injected)
    return injected


def patch_canonical_module_precomputed_ppo(
    module: Any,
    target_class: str,
    *,
    negative_control: NegativeControl,
    ppo_control: PPOActorControl,
    return_mode: str,
    diagnostics_jsonl: str | Path | None = None,
    diagnostics_latest: str | Path | None = None,
) -> type:
    original = getattr(module, target_class)
    injected = build_precomputed_ppo_agent_class(
        original,
        negative_control=negative_control,
        ppo_control=ppo_control,
        return_mode=return_mode,
        diagnostics_jsonl=diagnostics_jsonl,
        diagnostics_latest=diagnostics_latest,
    )
    setattr(module, target_class, injected)
    return injected
