from __future__ import annotations

import math

import pytest
import torch

from drpo.countdown_e8_oracle_offline_v2_taper_runtime import remoteness_anchors
from drpo.countdown_e8_oracle_offline_v2_taper_sweep import (
    EXPERIMENT_ID,
    METHODS,
    build_cells,
    coefficient_from_rho,
    normalized_distance,
    taper_weight,
    validate_sweep_config,
)


def config() -> dict:
    return {
        "experiment_id": EXPERIMENT_ID,
        "result_status": "pilot",
        "sweep": {
            "methods": list(METHODS),
            "rho_values": [0.9, 0.75, 0.6, 0.5, 0.35, 0.25, 0.125, 0.03125],
            "seed_offsets": [0, 1000, 2000],
        },
        "execution": {"required_gpu_count": 8},
        "calibration": {
            "reference_global_multiplier": 1 / 32,
            "remoteness_prompt_rows": 256,
            "gradient_prompt_rows": 16,
        },
    }


def test_plan_is_exactly_72_active_paper_family_cells() -> None:
    cells = build_cells(config())
    assert len(cells) == 72
    assert {cell.method for cell in cells} == set(METHODS)
    assert {cell.seed_offset for cell in cells} == {0, 1000, 2000}
    names = {cell.name for cell in cells}
    assert len(names) == 72
    assert not any("global" in name for name in names)
    assert not any("sbrc" in name for name in names)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("rho", [0.9, 0.5, 0.125, 0.03125])
def test_anchor_weight_equals_rho(method: str, rho: float) -> None:
    coefficient = coefficient_from_rho(method, rho)
    weight = taper_weight(method, torch.tensor([1.0]), coefficient)
    assert float(weight.item()) == pytest.approx(rho, rel=1e-6, abs=1e-7)


@pytest.mark.parametrize("method", METHODS)
def test_near_origin_retention_is_one(method: str) -> None:
    coefficient = coefficient_from_rho(method, 0.25)
    weight = taper_weight(method, torch.tensor([0.0]), coefficient)
    assert float(weight.item()) == pytest.approx(1.0)


def test_quadratic_is_reciprocal_not_squared_distance_exponential() -> None:
    coefficient = coefficient_from_rho("reciprocal_quadratic", 0.5)
    distance = torch.tensor([2.0])
    actual = taper_weight("reciprocal_quadratic", distance, coefficient)
    assert float(actual.item()) == pytest.approx(1.0 / 5.0)
    assert not math.isclose(float(actual.item()), math.exp(-4.0))


def test_normalized_distance_uses_excess_surprisal() -> None:
    seq_lp = torch.tensor([-1.0, -3.0, -6.0])
    distance = normalized_distance(seq_lp, tau=2.0, surprisal_scale=4.0)
    assert distance.tolist() == pytest.approx([0.0, 0.5, 1.0])


def test_remoteness_anchors_near_median_at_zero_and_far_median_at_one() -> None:
    tau, scale, near_median, far_median = remoteness_anchors(
        [1.0, 2.0, 3.0], [5.0, 6.0, 7.0]
    )
    assert tau == pytest.approx(near_median)
    assert scale == pytest.approx(far_median - near_median)
    seq_lp = torch.tensor([-near_median, -far_median])
    distance = normalized_distance(seq_lp, tau=tau, surprisal_scale=scale)
    assert distance.tolist() == pytest.approx([0.0, 1.0])


def test_config_rejects_global_or_sbrc() -> None:
    bad = config()
    bad["sweep"]["methods"] = ["reciprocal_linear", "global_matched", "sbrc"]
    with pytest.raises(ValueError):
        validate_sweep_config(bad)


def test_config_requires_eight_gpus() -> None:
    bad = config()
    bad["execution"]["required_gpu_count"] = 6
    with pytest.raises(ValueError):
        validate_sweep_config(bad)
