"""Optimizer and rank utilities shared by Hopper E7-Q2 training stages."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn

from .hopper_metrics import pearson

EPS = 1.0e-6


def sample_indices(
    rng: np.random.Generator,
    pool: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    return rng.choice(
        pool,
        size=batch_size,
        replace=len(pool) < batch_size,
    )


def tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.float32, device=device)


def parameter_norm(parameters: Iterable[nn.Parameter]) -> float:
    total = 0.0
    for parameter in parameters:
        total += float(parameter.detach().square().sum().cpu())
    return math.sqrt(total)


def full_gradient_statistics(
    loss: torch.Tensor,
    parameters: Iterable[nn.Parameter],
) -> dict[str, float]:
    parameter_list = list(parameters)
    gradients = torch.autograd.grad(
        loss,
        parameter_list,
        retain_graph=False,
        allow_unused=True,
    )
    total_square = 0.0
    elements = 0
    for gradient in gradients:
        if gradient is not None:
            total_square += float(gradient.detach().square().sum().cpu())
            elements += int(gradient.numel())
    raw = math.sqrt(total_square)
    return {
        "raw": raw,
        "rms": raw / math.sqrt(max(elements, 1)),
        "relative_to_parameter_norm": (raw / max(parameter_norm(parameter_list), EPS)),
        "elements": float(elements),
    }


def full_gradient_norm(
    loss: torch.Tensor,
    parameters: Iterable[nn.Parameter],
) -> float:
    return full_gradient_statistics(loss, parameters)["raw"]


def parameter_update_statistics(
    previous: Sequence[torch.Tensor],
    parameters: Iterable[nn.Parameter],
    elapsed_steps: int,
) -> dict[str, float]:
    parameter_list = list(parameters)
    delta_square = 0.0
    elements = 0
    for old, current in zip(previous, parameter_list):
        delta_square += float((current.detach() - old).square().sum().cpu())
        elements += int(current.numel())
    elapsed = max(int(elapsed_steps), 1)
    delta = math.sqrt(delta_square)
    return {
        "raw_per_step": delta / elapsed,
        "rms_per_step": (delta / math.sqrt(max(elements, 1)) / elapsed),
        "relative_per_step": (delta / max(parameter_norm(parameter_list), EPS) / elapsed),
        "elements": float(elements),
    }


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    position = 0
    while position < len(values):
        stop = position + 1
        while stop < len(values) and values[order[stop]] == values[order[position]]:
            stop += 1
        average_rank = 0.5 * (position + stop - 1) + 1.0
        ranks[order[position:stop]] = average_rank
        position = stop
    return ranks


def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return pearson(rankdata(y_true), rankdata(y_pred))
