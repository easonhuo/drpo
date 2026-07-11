from __future__ import annotations

import json
from pathlib import Path

import torch

from drpo.e7_canonical_injection import controlled_advantage
from drpo.e7_canonical_scale1_grid import (
    build_scale1_branches,
    expand_scale1_controls,
)
from tests.test_e7_canonical_sweep import contract


GRID_PATH = Path("configs/e7_canonical_scale1_coefficient_grid_v1.json")


def _grid() -> dict:
    return json.loads(GRID_PATH.read_text())


def test_repository_scale1_grid_has_expected_branches() -> None:
    raw = _grid()
    controls = expand_scale1_controls(raw)
    assert len(controls) == raw["branch_count_per_dataset_seed"] == 17

    taper_controls = [
        control
        for control in controls
        if control.method in {
            "reciprocal_linear",
            "reciprocal_quadratic",
            "exponential",
        }
    ]
    assert len(taper_controls) == 15
    assert {control.negative_scale for control in taper_controls} == {1.0}


def test_scale1_preserves_full_near_field_negative_alpha() -> None:
    advantage = torch.tensor([-2.0, 3.0])
    zero_distance = torch.zeros(2)
    for control in expand_scale1_controls(_grid()):
        if control.method not in {
            "reciprocal_linear",
            "reciprocal_quadratic",
            "exponential",
        }:
            continue
        adjusted, factor = controlled_advantage(
            advantage, zero_distance, control
        )
        assert torch.isclose(factor[0], torch.tensor(0.11))
        assert torch.isclose(adjusted[0], torch.tensor(-0.22))
        assert torch.isclose(factor[1], torch.tensor(1.0))
        assert torch.isclose(adjusted[1], torch.tensor(3.0))


def test_coefficient_grid_branch_ids_are_unique(tmp_path: Path) -> None:
    dataset_file = tmp_path / "dataset.hdf5"
    dataset_file.write_bytes(b"fixture")
    import hashlib

    digest = hashlib.sha256(b"fixture").hexdigest()
    run_spec = {
        "run_kind": "pilot",
        "datasets": [
            {"id": "hopper", "path": str(dataset_file), "sha256": digest}
        ],
        "seeds": [200],
        "trainer_argv_template": [],
    }
    branches = build_scale1_branches(contract(tmp_path), run_spec, _grid())
    ids = [branch.branch_id for branch in branches]
    assert len(ids) == 17
    assert len(set(ids)) == 17
    taper_ids = [
        branch_id
        for branch_id in ids
        if "reciprocal_" in branch_id or "exponential" in branch_id
    ]
    assert taper_ids
    assert all("__scale1__coef" in branch_id for branch_id in taper_ids)
