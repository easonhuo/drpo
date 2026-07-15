from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from drpo import e7_canonical_injection as canonical_injection
from drpo import e7_canonical_ppo_injection as ppo_injection
from drpo.e7_canonical_injection import NegativeControl
from drpo.e7_w0_geometry_diagnostics import (
    GeometryDiagnostics,
    install_controlled_advantage_observer,
)


def _public_control() -> dict:
    return {
        "method": "exponential",
        "weight_at_zero": 1.0,
        "exp_coefficient": 4.0,
        "reference_distance": 2.0,
        "formula": "w(d)=w(0)*exp(-c*(d/2))",
    }


def test_geometry_observer_writes_exact_mass_and_thresholds(tmp_path: Path) -> None:
    observer = GeometryDiagnostics(
        public_control=_public_control(),
        actor_update_mode="a2c",
        interval=2,
        total_steps=2,
        sampled_values_per_update=16,
        jsonl_path=tmp_path / "geometry.jsonl",
        latest_path=tmp_path / "latest.json",
    )
    advantage = torch.tensor([-2.0, -1.0, 3.0])
    distance = torch.tensor([0.5, 2.0, 1.0])
    factor = torch.tensor([0.5, 0.1, 1.0])
    observer.observe(advantage, distance, factor)
    observer.observe(advantage, distance, factor)
    final = observer.validate_complete()
    assert final["negative_samples"] == 4
    assert final["negative_distance_mean"] == pytest.approx(1.25)
    assert final["negative_weight_mean"] == pytest.approx(0.3)
    assert final["effective_negative_mass_fraction"] == pytest.approx(1.1 / 3.0)
    assert final["negative_weight_gt_0p5_fraction"] == 0.0
    assert final["negative_weight_gt_0p1_fraction"] == pytest.approx(0.5)
    assert final["negative_weight_gt_0p05_fraction"] == 1.0
    assert final["negative_weight_gt_0p01_fraction"] == 1.0
    assert final["status"] == "complete"
    assert "negative_control" not in json.dumps(final)


def test_installed_observer_preserves_controlled_advantage_values(tmp_path: Path) -> None:
    observer = GeometryDiagnostics(
        public_control=_public_control(),
        actor_update_mode="ppo_clip",
        interval=1,
        total_steps=1,
        sampled_values_per_update=8,
        jsonl_path=tmp_path / "geometry.jsonl",
        latest_path=tmp_path / "latest.json",
    )
    control = NegativeControl(
        method="exponential",
        negative_scale=1.0 / 0.11,
        canonical_alpha=0.11,
        reference_distance=2.0,
        exponential_coefficient=4.0,
    )
    advantage = torch.tensor([-2.0, 1.0])
    distance = torch.tensor([1.0, 2.0])
    expected = canonical_injection.controlled_advantage(advantage, distance, control)
    canonical_original = canonical_injection.controlled_advantage
    ppo_original = ppo_injection.controlled_advantage
    with install_controlled_advantage_observer(observer):
        actual = ppo_injection.controlled_advantage(advantage, distance, control)
        assert torch.equal(actual[0], expected[0])
        assert torch.equal(actual[1], expected[1])
    assert canonical_injection.controlled_advantage is canonical_original
    assert ppo_injection.controlled_advantage is ppo_original
    assert observer.validate_complete()["actor_update_mode"] == "ppo_clip"
