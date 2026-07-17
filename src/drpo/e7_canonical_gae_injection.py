"""Trajectory-snapshot GAE adapter for the canonical E7 joint actor--critic path."""

from __future__ import annotations

import dataclasses
import hashlib
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from drpo import e7_canonical_injection as canonical


ESTIMATORS = {"td", "gae"}
_FLOAT32_EXACT_INTEGER_LIMIT = 2**24


@dataclasses.dataclass(frozen=True)
class OrderedReplay:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    terminals: np.ndarray
    timeouts: np.ndarray

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "OrderedReplay":
        replay = cls(
            observations=np.asarray(raw["obs"], dtype=np.float32),
            actions=np.asarray(raw["acts"], dtype=np.float32),
            rewards=np.asarray(raw["rews"], dtype=np.float32).reshape(-1),
            next_observations=np.asarray(raw["next_obs"], dtype=np.float32),
            terminals=np.asarray(raw["terms"], dtype=np.bool_).reshape(-1),
            timeouts=np.asarray(raw["touts"], dtype=np.bool_).reshape(-1),
        )
        replay.validate()
        return replay

    @property
    def size(self) -> int:
        return int(self.rewards.size)

    def validate(self) -> None:
        lengths = {
            len(self.observations),
            len(self.actions),
            len(self.rewards),
            len(self.next_observations),
            len(self.terminals),
            len(self.timeouts),
        }
        if lengths == {0} or len(lengths) != 1:
            raise ValueError("ordered replay arrays must be non-empty and aligned")
        if self.observations.ndim != 2 or self.next_observations.shape != self.observations.shape:
            raise ValueError("ordered replay observations must be aligned rank-2 arrays")
        if self.actions.ndim != 2:
            raise ValueError("ordered replay actions must be rank-2")
        if bool((self.terminals & self.timeouts).any()):
            raise ValueError("terminal and timeout flags must not overlap")
        for name, value in (
            ("observations", self.observations),
            ("actions", self.actions),
            ("rewards", self.rewards),
            ("next_observations", self.next_observations),
        ):
            if not np.isfinite(value).all():
                raise ValueError(f"ordered replay contains non-finite {name}")
        if self.size > _FLOAT32_EXACT_INTEGER_LIMIT:
            raise ValueError("transition IDs cannot be represented exactly through float32 ep_ret")


@dataclasses.dataclass(frozen=True)
class SnapshotEstimatorConfig:
    estimator: str
    gae_lambda: float = 0.95
    canonical_batch_size: int = 256

    def validate(self) -> None:
        if self.estimator not in ESTIMATORS:
            raise ValueError(f"unsupported estimator={self.estimator!r}")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be in [0, 1]")
        if self.canonical_batch_size <= 0:
            raise ValueError("canonical_batch_size must be positive")

    def refresh_interval(self, transition_count: int) -> int:
        self.validate()
        if transition_count <= 0:
            raise ValueError("transition_count must be positive")
        return int(math.ceil(transition_count / self.canonical_batch_size))


def transition_id_channel(transition_count: int) -> np.ndarray:
    """Return exact float32 transition IDs for the trainer's existing ``ep_ret`` slot."""

    if transition_count <= 0 or transition_count > _FLOAT32_EXACT_INTEGER_LIMIT:
        raise ValueError("transition_count is outside the exact float32 ID range")
    return np.arange(transition_count, dtype=np.float32)


def compute_snapshot_tables(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    terminals: np.ndarray,
    timeouts: np.ndarray,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute matched one-step TD and GAE tables from one critic snapshot."""

    reward = np.asarray(rewards, dtype=np.float64).reshape(-1)
    value = np.asarray(values, dtype=np.float64).reshape(-1)
    next_value = np.asarray(next_values, dtype=np.float64).reshape(-1)
    terminal = np.asarray(terminals, dtype=np.bool_).reshape(-1)
    timeout = np.asarray(timeouts, dtype=np.bool_).reshape(-1)
    if not reward.size or not (
        reward.shape == value.shape == next_value.shape == terminal.shape == timeout.shape
    ):
        raise ValueError("snapshot arrays must be non-empty and aligned")
    if bool((terminal & timeout).any()):
        raise ValueError("terminal and timeout flags must not overlap")
    if not all(np.isfinite(item).all() for item in (reward, value, next_value)):
        raise ValueError("snapshot values must be finite")
    if not 0.0 <= gamma <= 1.0 or not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gamma and gae_lambda must be in [0, 1]")

    td = reward + gamma * next_value * (~terminal) - value
    continuation = ~(terminal | timeout)
    continuation[-1] = False
    gae = np.empty_like(td)
    running = 0.0
    for index in range(td.size - 1, -1, -1):
        running = td[index] + gamma * gae_lambda * float(continuation[index]) * running
        gae[index] = running
    return td.astype(np.float32), gae.astype(np.float32)


def state_dict_sha256(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def load_ordered_replay(
    *,
    canonical_root: str | Path,
    dataset_path: str | Path,
    dataset_id: str,
) -> OrderedReplay:
    root = str(Path(canonical_root).expanduser().resolve())
    inserted = root not in sys.path
    if inserted:
        sys.path.insert(0, root)
    try:
        from d4rl_common.train_loop import load_hdf5

        return OrderedReplay.from_mapping(
            load_hdf5(Path(dataset_path).expanduser().resolve(), dataset_name=dataset_id)
        )
    finally:
        if inserted:
            sys.path.remove(root)


def _snapshot_values(
    critic: torch.nn.Module,
    observations: np.ndarray,
    *,
    device: torch.device,
    chunk_size: int,
) -> np.ndarray:
    output = np.empty(len(observations), dtype=np.float32)
    was_training = critic.training
    critic.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(observations), chunk_size):
                end = min(len(observations), start + chunk_size)
                states = torch.as_tensor(
                    observations[start:end],
                    dtype=torch.float32,
                    device=device,
                )
                output[start:end] = (
                    critic(states).squeeze(-1).detach().cpu().to(torch.float32).numpy()
                )
    finally:
        critic.train(was_training)
    if not np.isfinite(output).all():
        raise FloatingPointError("critic snapshot produced non-finite values")
    return output


def build_joint_snapshot_agent_class(
    base_class: type,
    *,
    replay: OrderedReplay,
    negative_control: canonical.NegativeControl,
    estimator: SnapshotEstimatorConfig,
    return_mode: str,
    instance_sink: list[Any] | None = None,
) -> type:
    """Build a joint-training agent that swaps only the actor advantage estimator."""

    replay.validate()
    estimator.validate()
    if return_mode not in {"zero_float", "metrics_dict"}:
        raise ValueError(f"unsupported return_mode={return_mode!r}")
    refresh_interval = estimator.refresh_interval(replay.size)
    snapshot_chunk_size = max(4096, estimator.canonical_batch_size * 32)

    class CanonicalJointSnapshotAgent(base_class):  # type: ignore[misc, valid-type]
        _drpo_negative_control = negative_control
        _drpo_snapshot_estimator = estimator

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            canonical.validate_agent_instance(
                self,
                expected_alpha=negative_control.canonical_alpha,
            )
            self._drpo_update_count = 0
            self._drpo_advantage_table: torch.Tensor | None = None
            self._drpo_snapshot_hashes: list[str] = []
            self._drpo_snapshot_refresh_interval = refresh_interval
            self._drpo_snapshot_count = 0
            self._drpo_last_snapshot_update = 0
            if instance_sink is not None:
                instance_sink.append(self)

        def _drpo_refresh_snapshot(self) -> None:
            device = canonical._agent_device(self)  # noqa: SLF001
            values = _snapshot_values(
                self.critic,
                replay.observations,
                device=device,
                chunk_size=snapshot_chunk_size,
            )
            next_values = _snapshot_values(
                self.critic,
                replay.next_observations,
                device=device,
                chunk_size=snapshot_chunk_size,
            )
            td, gae = compute_snapshot_tables(
                replay.rewards,
                values,
                next_values,
                replay.terminals,
                replay.timeouts,
                gamma=float(self.gamma),
                gae_lambda=estimator.gae_lambda,
            )
            selected = td if estimator.estimator == "td" else gae
            self._drpo_advantage_table = torch.from_numpy(selected.copy())
            self._drpo_snapshot_hashes.append(state_dict_sha256(self.critic.state_dict()))
            self._drpo_snapshot_count += 1
            self._drpo_last_snapshot_update = self._drpo_update_count

        def _drpo_snapshot_summary(self) -> dict[str, Any]:
            final_hash = state_dict_sha256(self.critic.state_dict())
            first_hash = self._drpo_snapshot_hashes[0] if self._drpo_snapshot_hashes else None
            latest_hash = self._drpo_snapshot_hashes[-1] if self._drpo_snapshot_hashes else None
            return {
                "estimator": estimator.estimator,
                "gae_lambda": estimator.gae_lambda,
                "snapshot_count": self._drpo_snapshot_count,
                "snapshot_refresh_interval": refresh_interval,
                "snapshot_hashes": list(self._drpo_snapshot_hashes),
                "first_snapshot_critic_sha256": first_hash,
                "latest_snapshot_critic_sha256": latest_hash,
                "final_critic_sha256": final_hash,
                "critic_evolution_observed": bool(first_hash and final_hash != first_hash),
                "last_snapshot_update": self._drpo_last_snapshot_update,
            }

        def update(
            self,
            s: Any,
            a: Any,
            r: Any,
            ns: Any,
            d: Any,
            ep_ret: Any = None,
        ) -> Any:
            canonical.validate_agent_instance(
                self,
                expected_alpha=negative_control.canonical_alpha,
            )
            self._drpo_update_count += 1
            if (
                self._drpo_advantage_table is None
                or (self._drpo_update_count - 1) % refresh_interval == 0
            ):
                self._drpo_refresh_snapshot()

            device = canonical._agent_device(self)  # noqa: SLF001
            states = canonical._as_tensor(s, device=device)  # noqa: SLF001
            actions = canonical._as_tensor(a, device=device)  # noqa: SLF001
            rewards = canonical._as_tensor(r, device=device).reshape(-1)  # noqa: SLF001
            next_states = canonical._as_tensor(ns, device=device)  # noqa: SLF001
            dones = canonical._as_tensor(  # noqa: SLF001
                d,
                device=device,
                dtype=torch.bool,
            ).reshape(-1)
            raw_ids = canonical._as_tensor(ep_ret, device=device).reshape(-1)  # noqa: SLF001
            rounded_ids = raw_ids.round()
            if not bool(torch.isfinite(raw_ids).all()) or not torch.equal(raw_ids, rounded_ids):
                raise ValueError("transition IDs must be finite exact integers")
            transition_ids = rounded_ids.to(dtype=torch.long).cpu()
            if transition_ids.numel() != states.shape[0]:
                raise ValueError("transition ID batch is not aligned with states")
            if int(transition_ids.min()) < 0 or int(transition_ids.max()) >= replay.size:
                raise ValueError("transition ID is outside the ordered replay")
            assert self._drpo_advantage_table is not None
            actor_advantage = self._drpo_advantage_table.index_select(
                0,
                transition_ids,
            ).to(device=device)

            values = self.critic(states).squeeze(-1)
            with torch.no_grad():
                next_values = self.critic(next_states).squeeze(-1)
                targets = rewards + float(self.gamma) * next_values * (~dones).float()

            mean, log_std, distance = canonical.detached_standardized_distance(
                self.actor,
                states,
                actions,
            )
            distribution = torch.distributions.Normal(mean, log_std.exp())
            log_prob = distribution.log_prob(actions).sum(dim=-1)
            weighted_advantage, factor = canonical.controlled_advantage(
                actor_advantage,
                distance,
                negative_control,
            )
            actor_loss = -(log_prob * weighted_advantage.detach()).mean()
            self.a_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.a_opt.step()

            value_error = targets - values
            expectile = torch.where(
                value_error > 0,
                torch.full_like(value_error, float(self.tau)),
                torch.full_like(value_error, 1.0 - float(self.tau)),
            )
            critic_loss = (expectile * value_error.square()).mean()
            self.c_opt.zero_grad(set_to_none=True)
            critic_loss.backward()
            self.c_opt.step()

            negative = actor_advantage < 0
            metrics = {
                "actor_loss": float(actor_loss.detach().cpu()),
                "critic_loss": float(critic_loss.detach().cpu()),
                "advantage_estimator": estimator.estimator,
                "snapshot_count": self._drpo_snapshot_count,
                "snapshot_refresh_interval": refresh_interval,
                "positive_fraction": float((~negative).float().mean().cpu()),
                "negative_fraction": float(negative.float().mean().cpu()),
                "negative_factor_mean": (
                    float(factor[negative].mean().detach().cpu())
                    if bool(negative.any())
                    else float("nan")
                ),
                "method": negative_control.method,
            }
            self._drpo_last_negative_control_metrics = metrics
            self._drpo_last_snapshot_metrics = metrics
            if return_mode == "metrics_dict":
                return metrics
            return 0.0

    CanonicalJointSnapshotAgent.__name__ = base_class.__name__
    CanonicalJointSnapshotAgent.__qualname__ = base_class.__qualname__
    CanonicalJointSnapshotAgent.__module__ = base_class.__module__
    return CanonicalJointSnapshotAgent
