"""Side-effect-only diagnostics for direct-w(0) negative geometry.

The observer is installed around the existing canonical A2C or PPO injection.  It
receives the exact ``advantage``, detached standardized ``distance`` and applied
``factor`` tensors already used by the actor update, so it adds no extra actor or
critic forward pass and does not alter the scientific objective.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch

from drpo import e7_canonical_injection as canonical_injection
from drpo import e7_canonical_ppo_injection as ppo_injection

QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90, 0.99)
WEIGHT_THRESHOLDS = (0.5, 0.1, 0.05, 0.01)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True) + "\n")


def _quantile_payload(prefix: str, values: list[torch.Tensor]) -> dict[str, float | None]:
    if not values:
        return {f"{prefix}_p{int(q * 100):02d}": None for q in QUANTILES}
    flat = torch.cat(values).to(dtype=torch.float64)
    return {
        f"{prefix}_p{int(q * 100):02d}": float(torch.quantile(flat, q).item())
        for q in QUANTILES
    }


class GeometryDiagnostics:
    """Accumulate exact interval totals plus bounded deterministic quantile samples."""

    def __init__(
        self,
        *,
        public_control: Mapping[str, Any],
        actor_update_mode: str,
        interval: int,
        total_steps: int,
        sampled_values_per_update: int,
        jsonl_path: str | Path,
        latest_path: str | Path,
    ) -> None:
        if actor_update_mode not in {"a2c", "ppo_clip"}:
            raise ValueError("actor_update_mode must be a2c or ppo_clip")
        if interval <= 0 or total_steps <= 0:
            raise ValueError("diagnostic interval and total_steps must be positive")
        if sampled_values_per_update <= 0:
            raise ValueError("sampled_values_per_update must be positive")
        forbidden = {"negative_scale", "canonical_alpha", "effective_alpha"}
        if forbidden & set(public_control):
            raise ValueError("public geometry control contains a legacy scale/alpha field")
        self.public_control = dict(public_control)
        self.actor_update_mode = actor_update_mode
        self.interval = int(interval)
        self.total_steps = int(total_steps)
        self.sampled_values_per_update = int(sampled_values_per_update)
        self.jsonl_path = Path(jsonl_path).resolve()
        self.latest_path = Path(latest_path).resolve()
        self.update_count = 0
        self._reset_interval()
        self.jsonl_path.unlink(missing_ok=True)
        self.latest_path.unlink(missing_ok=True)

    def _reset_interval(self) -> None:
        self.interval_start_update = self.update_count + 1
        self.negative_samples = 0
        self.distance_sum = 0.0
        self.factor_sum = 0.0
        self.abs_advantage_sum = 0.0
        self.weighted_abs_advantage_sum = 0.0
        self.threshold_counts = {threshold: 0 for threshold in WEIGHT_THRESHOLDS}
        self.distance_samples: list[torch.Tensor] = []
        self.factor_samples: list[torch.Tensor] = []

    @staticmethod
    def _bounded_sample(values: torch.Tensor, limit: int) -> torch.Tensor:
        flat = values.detach().reshape(-1).to(device="cpu", dtype=torch.float64)
        if flat.numel() <= limit:
            return flat
        indices = torch.linspace(0, flat.numel() - 1, steps=limit, dtype=torch.float64)
        return flat[indices.round().to(dtype=torch.long)]

    def observe(
        self,
        advantage: torch.Tensor,
        distance: torch.Tensor,
        factor: torch.Tensor,
    ) -> None:
        self.update_count += 1
        if self.update_count > self.total_steps:
            raise RuntimeError("geometry observer received more updates than total_steps")
        with torch.no_grad():
            negative = advantage.detach().reshape(-1) < 0
            negative_distance = distance.detach().reshape(-1)[negative]
            negative_factor = factor.detach().reshape(-1)[negative]
            negative_abs_advantage = advantage.detach().reshape(-1)[negative].abs()
            if negative_distance.numel():
                if not bool(torch.isfinite(negative_distance).all()):
                    raise FloatingPointError("non-finite negative distance diagnostics")
                if not bool(torch.isfinite(negative_factor).all()):
                    raise FloatingPointError("non-finite negative factor diagnostics")
                if not bool(torch.isfinite(negative_abs_advantage).all()):
                    raise FloatingPointError("non-finite negative advantage diagnostics")
                count = int(negative_distance.numel())
                self.negative_samples += count
                self.distance_sum += float(negative_distance.sum().cpu())
                self.factor_sum += float(negative_factor.sum().cpu())
                self.abs_advantage_sum += float(negative_abs_advantage.sum().cpu())
                self.weighted_abs_advantage_sum += float(
                    (negative_abs_advantage * negative_factor).sum().cpu()
                )
                for threshold in WEIGHT_THRESHOLDS:
                    self.threshold_counts[threshold] += int(
                        (negative_factor > threshold).sum().cpu()
                    )
                self.distance_samples.append(
                    self._bounded_sample(
                        negative_distance,
                        self.sampled_values_per_update,
                    )
                )
                self.factor_samples.append(
                    self._bounded_sample(
                        negative_factor,
                        self.sampled_values_per_update,
                    )
                )
        if self.update_count % self.interval == 0 or self.update_count == self.total_steps:
            self._flush()

    def _flush(self) -> None:
        count = self.negative_samples
        payload: dict[str, Any] = {
            "schema_version": 1,
            "status": "complete" if self.update_count == self.total_steps else "running",
            "update": self.update_count,
            "total_steps": self.total_steps,
            "interval_start_update": self.interval_start_update,
            "interval_end_update": self.update_count,
            "actor_update_mode": self.actor_update_mode,
            "weight_control": self.public_control,
            "negative_samples": count,
            "negative_distance_mean": self.distance_sum / count if count else None,
            "negative_weight_mean": self.factor_sum / count if count else None,
            "negative_abs_advantage_sum": self.abs_advantage_sum,
            "weighted_negative_abs_advantage_sum": self.weighted_abs_advantage_sum,
            "effective_negative_mass_fraction": (
                self.weighted_abs_advantage_sum / self.abs_advantage_sum
                if self.abs_advantage_sum > 0.0
                else None
            ),
            "sampled_negative_values": sum(
                int(value.numel()) for value in self.distance_samples
            ),
        }
        for threshold in WEIGHT_THRESHOLDS:
            label = str(threshold).replace(".", "p")
            payload[f"negative_weight_gt_{label}_fraction"] = (
                self.threshold_counts[threshold] / count if count else None
            )
        payload.update(_quantile_payload("negative_distance", self.distance_samples))
        payload.update(_quantile_payload("negative_weight", self.factor_samples))
        for key, value in payload.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise FloatingPointError(f"non-finite geometry diagnostic field: {key}")
        _append_jsonl(self.jsonl_path, payload)
        _atomic_json(self.latest_path, payload)
        self._reset_interval()

    def validate_complete(self) -> dict[str, Any]:
        if self.update_count != self.total_steps:
            raise RuntimeError(
                f"geometry diagnostics stopped at {self.update_count}, expected {self.total_steps}"
            )
        if not self.latest_path.is_file():
            raise RuntimeError("geometry diagnostics latest file is missing")
        payload = json.loads(self.latest_path.read_text())
        if payload.get("status") != "complete":
            raise RuntimeError("geometry diagnostics did not reach complete status")
        if int(payload.get("update", -1)) != self.total_steps:
            raise RuntimeError("geometry diagnostics final update mismatch")
        return payload


@contextlib.contextmanager
def install_controlled_advantage_observer(
    observer: GeometryDiagnostics,
) -> Iterator[None]:
    """Observe the exact controlled-advantage tensors without changing their values."""

    canonical_original = canonical_injection.controlled_advantage
    ppo_original = ppo_injection.controlled_advantage

    def observed(
        advantage: torch.Tensor,
        distance: torch.Tensor,
        control: canonical_injection.NegativeControl,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weighted, factor = canonical_original(advantage, distance, control)
        observer.observe(advantage, distance, factor)
        return weighted, factor

    canonical_injection.controlled_advantage = observed
    ppo_injection.controlled_advantage = observed
    try:
        yield
    finally:
        canonical_injection.controlled_advantage = canonical_original
        ppo_injection.controlled_advantage = ppo_original
