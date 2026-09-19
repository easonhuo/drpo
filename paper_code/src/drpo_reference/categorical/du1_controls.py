"""D-U1 learner-relative surprisal coordinates and negative controls."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch
from torch import nn

from drpo_reference.controls import TaperFamily
from drpo_reference.controls import (
    normalized_excess_surprisal as shared_normalized_excess,
)
from drpo_reference.controls import taper_weight as shared_taper_weight

from .du1_environment import CartesianSemanticEnvironment
from .du1_policy import (
    CartesianPolicy,
    cell_log_probs,
    trainable_parameters,
)
from .du1_protocol import CELL_NAMES, DU1Protocol, MethodSpec

EPS = 1.0e-12


def taper_coefficients(retention: float) -> dict[str, float]:
    """Calibrate every active taper to the same rare reference retention."""

    return {
        "reciprocal_linear_distance": 1.0 / retention - 1.0,
        "reciprocal_quadratic_distance": 1.0 / retention - 1.0,
        "exponential_quadratic_distance": -math.log(retention),
    }


def normalized_excess_surprisal(
    log_probability: torch.Tensor,
    calibration: Mapping[str, float],
) -> torch.Tensor:
    return shared_normalized_excess(
        log_probability,
        threshold=float(calibration["threshold"]),
        scale=float(calibration["scale"]),
        detach=True,
    )


def taper_weight(
    normalized_excess: torch.Tensor,
    family: str,
    coefficient: float,
) -> torch.Tensor:
    """Evaluate the revision-4 taper on normalized excess surprisal.

    The paper distance coordinate is ``sqrt(u)``. Linear-distance attenuation is
    therefore linear in ``sqrt(u)``, while quadratic-distance attenuation is
    linear in ``u``.
    """

    distance = torch.sqrt(torch.clamp(normalized_excess.detach(), min=0.0))
    mapping = {
        "reciprocal_linear_distance": TaperFamily.RECIPROCAL_LINEAR,
        "reciprocal_quadratic_distance": TaperFamily.RECIPROCAL_QUADRATIC,
        "exponential_quadratic_distance": TaperFamily.EXPONENTIAL_QUADRATIC,
    }
    resolved = mapping[family]
    return shared_taper_weight(
        distance,
        family=resolved,
        coefficient=coefficient,
        detach_distance=True,
    )


def coordinate_calibration(
    model: CartesianPolicy,
    environment: CartesianSemanticEnvironment,
    protocol: DU1Protocol,
) -> dict[str, float]:
    count = min(protocol.calibration_states, environment.train_count)
    index = torch.arange(count)
    with torch.no_grad():
        _, cells, _ = cell_log_probs(
            model,
            environment,
            environment.train,
            index,
        )
    common = torch.cat(
        [
            -cells["useful_common"],
            -cells["unhelpful_common"],
        ]
    )
    rare = torch.cat(
        [
            -cells["useful_rare"],
            -cells["unhelpful_rare"],
        ]
    )
    threshold = float(common.median())
    return {
        "threshold": threshold,
        "scale": float(rare.median()) - threshold,
    }


def rarity_logit_anchor_loss(
    model: CartesianPolicy,
    states: torch.Tensor,
) -> torch.Tensor:
    residual = model.rarity_coordinate(states) - model.initial_rarity_half_gap
    return 0.5 * residual.square().mean()


def active_cell_loss(
    cells: Mapping[str, torch.Tensor],
    spec: MethodSpec,
    calibration: Mapping[str, float],
    coefficients: Mapping[str, float],
    global_scale: float,
) -> torch.Tensor:
    if not spec.active_cells:
        return next(iter(cells.values())).sum() * 0.0

    pieces: list[torch.Tensor] = []
    for cell in spec.active_cells:
        log_probability = cells[cell]
        if spec.taper_family is None:
            weight = torch.ones_like(log_probability)
        elif spec.taper_family == "global":
            weight = torch.full_like(log_probability, float(global_scale))
        else:
            coordinate = normalized_excess_surprisal(log_probability, calibration)
            weight = taper_weight(
                coordinate,
                spec.taper_family,
                float(coefficients[spec.taper_family]),
            )
        pieces.append((weight * log_probability).mean())
    return torch.stack(pieces).sum() / float(len(CELL_NAMES))


def flat_grad_norm(
    loss: torch.Tensor,
    parameters: Sequence[nn.Parameter],
    *,
    retain_graph: bool = True,
) -> float:
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    total = torch.zeros((), dtype=torch.float64)
    for gradient in gradients:
        if gradient is not None:
            total += gradient.detach().double().square().sum().cpu()
    return float(torch.sqrt(total))


def negative_loss(
    *,
    cells: Mapping[str, torch.Tensor],
    spec: MethodSpec,
    calibration: Mapping[str, float],
    protocol: DU1Protocol,
    model: CartesianPolicy,
) -> torch.Tensor:
    coefficients = taper_coefficients(protocol.reference_rare_retention)
    if spec.taper_family == "global":
        raw_spec = MethodSpec("all_negative", CELL_NAMES)
        target_spec = MethodSpec(
            "exponential_quadratic_distance",
            CELL_NAMES,
            "exponential_quadratic_distance",
        )
        parameters = trainable_parameters(model)
        raw_loss = active_cell_loss(
            cells,
            raw_spec,
            calibration,
            coefficients,
            1.0,
        )
        target_loss = active_cell_loss(
            cells,
            target_spec,
            calibration,
            coefficients,
            1.0,
        )
        raw_norm = flat_grad_norm(raw_loss, parameters, retain_graph=True)
        target_norm = flat_grad_norm(target_loss, parameters, retain_graph=True)
        scale = target_norm / max(raw_norm, EPS)
        return active_cell_loss(
            cells,
            spec,
            calibration,
            coefficients,
            scale,
        )

    return active_cell_loss(
        cells,
        spec,
        calibration,
        coefficients,
        1.0,
    )
