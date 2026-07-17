"""Frozen Hopper E7-Q2 critic-advantage construction."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .hopper_data import Normalizer, OfflineData
from .hopper_models import ValueNetwork
from .hopper_optim import tensor


def critic_advantage_arrays(
    *,
    critic: ValueNetwork,
    data: OfflineData,
    observation_normalizer: Normalizer,
    target_normalizer: Normalizer,
    gamma: float,
    standardize: bool,
    standardization_indices: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    observations = observation_normalizer.transform(data.observations)
    next_observations = observation_normalizer.transform(
        data.next_observations
    )
    values: list[np.ndarray] = []
    next_values: list[np.ndarray] = []
    critic.eval()
    with torch.no_grad():
        for offset in range(0, data.size, 65_536):
            stop = min(data.size, offset + 65_536)
            values.append(
                critic(
                    tensor(observations[offset:stop], device)
                )
                .cpu()
                .numpy()
            )
            next_values.append(
                critic(
                    tensor(next_observations[offset:stop], device)
                )
                .cpu()
                .numpy()
            )
    value_normalized = np.concatenate(values).astype(np.float32)
    next_value_normalized = np.concatenate(next_values).astype(
        np.float32
    )
    target_scale = float(target_normalizer.std[0])
    target_center = float(target_normalizer.mean[0])
    value = value_normalized * target_scale + target_center
    next_value = next_value_normalized * target_scale + target_center
    bootstrap_mask = (~(data.terminals | data.timeouts)).astype(
        np.float32
    )
    raw = data.rewards + gamma * bootstrap_mask * next_value - value
    center = float(np.mean(raw[standardization_indices]))
    scale = float(np.std(raw[standardization_indices]))
    if standardize:
        advantage = (
            (raw - center) / max(scale, 1.0e-8)
        ).astype(np.float32)
    else:
        advantage = raw.astype(np.float32)
        center, scale = 0.0, 1.0
    return {
        "advantage": advantage,
        "raw_advantage": raw.astype(np.float32),
        "value": value.astype(np.float32),
        "next_value": next_value.astype(np.float32),
        "center": center,
        "scale": scale,
    }
