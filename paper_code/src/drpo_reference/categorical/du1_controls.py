"""D-U1 learner-relative surprisal coordinates and negative controls."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn

from drpo_reference.controls import (
    TaperFamily,
    normalized_excess_surprisal as shared_normalized_excess,
    taper_weight as shared_taper_weight,
)

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

    if not 0.0 < retention < 1.0:
        raise ValueError("reference retention must lie in (0, 1)")
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
    try:
        resolved = mapping[family]
    except KeyError as exc:
        raise ValueError(f"unknown active taper family: {family}") from exc
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
    count = min(protocol.audit_states, environment.train_count)
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
    rare_median = float(rare.median())
    scale = rare_median - threshold
    if scale <= protocol.minimum_calibration_gap:
        raise RuntimeError(f"initial common/rare surprisal gap too small: {scale}")
    return {
        "threshold": threshold,
        "scale": scale,
        "common_surprisal_median": threshold,
        "rare_surprisal_median": rare_median,
        "rare_minus_common_median": scale,
        "initial_cartesian_exact": True,
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
) -> tuple[torch.Tensor, dict[str, float]]:
    if not spec.active_cells:
        zero = next(iter(cells.values())).sum() * 0.0
        return zero, {f"weight_{cell}": 0.0 for cell in CELL_NAMES}

    pieces: list[torch.Tensor] = []
    diagnostics: dict[str, float] = {}
    for cell in spec.active_cells:
        log_probability = cells[cell]
        if spec.taper_family is None:
            weight = torch.ones_like(log_probability)
        elif spec.taper_family == "global":
            weight = torch.full_like(
                log_probability,
                float(global_scale),
            )
        else:
            coordinate = normalized_excess_surprisal(
                log_probability,
                calibration,
            )
            weight = taper_weight(
                coordinate,
                spec.taper_family,
                float(coefficients[spec.taper_family]),
            )
        pieces.append((weight * log_probability).mean())
        diagnostics[f"weight_{cell}"] = float(weight.detach().mean())
    for cell in CELL_NAMES:
        diagnostics.setdefault(f"weight_{cell}", 0.0)
    return (
        torch.stack(pieces).sum() / float(len(CELL_NAMES)),
        diagnostics,
    )


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


def negative_loss_and_diagnostics(
    *,
    cells: Mapping[str, torch.Tensor],
    spec: MethodSpec,
    calibration: Mapping[str, float],
    protocol: DU1Protocol,
    model: CartesianPolicy,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return the registered negative loss and stepwise budget audit."""

    coefficients = taper_coefficients(protocol.reference_rare_retention)
    if spec.taper_family == "global":
        raw_spec = MethodSpec(
            "all_negative",
            CELL_NAMES,
        )
        target_spec = MethodSpec(
            "exponential_quadratic_distance",
            CELL_NAMES,
            "exponential_quadratic_distance",
        )
        parameters = trainable_parameters(model)
        raw_loss, _ = active_cell_loss(
            cells,
            raw_spec,
            calibration,
            coefficients,
            1.0,
        )
        target_loss, _ = active_cell_loss(
            cells,
            target_spec,
            calibration,
            coefficients,
            1.0,
        )
        raw_norm = flat_grad_norm(
            raw_loss,
            parameters,
            retain_graph=True,
        )
        target_norm = flat_grad_norm(
            target_loss,
            parameters,
            retain_graph=True,
        )
        scale = target_norm / max(raw_norm, EPS)
        negative, weights = active_cell_loss(
            cells,
            spec,
            calibration,
            coefficients,
            scale,
        )
        return negative, {
            **weights,
            "negative_raw_gradient_norm": raw_norm,
            "negative_target_gradient_norm": target_norm,
            "negative_applied_gradient_norm": scale * raw_norm,
            "stepwise_budget_match_error": abs(scale * raw_norm - target_norm),
            "stepwise_global_scale": scale,
        }

    negative, weights = active_cell_loss(
        cells,
        spec,
        calibration,
        coefficients,
        1.0,
    )
    return negative, {
        **weights,
        "negative_raw_gradient_norm": 0.0,
        "negative_target_gradient_norm": 0.0,
        "negative_applied_gradient_norm": 0.0,
        "stepwise_budget_match_error": 0.0,
        "stepwise_global_scale": (1.0 if spec.active_cells else 0.0),
    }
