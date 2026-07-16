from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from drpo.e7_sqexp_gae import EXPECTED_BRANCHES, build_branches, load_grid


def test_frozen_grid_expands_exact_192_branch_matrix(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    grid, _ = load_grid(repo / "configs/e7_sqexp_gae_v1.json")
    run_spec = {
        "datasets": [
            {
                "id": dataset,
                "path": str(tmp_path / f"{dataset}.hdf5"),
                "sha256": "0" * 64,
            }
            for dataset in grid["datasets"]
        ],
        "seeds": grid["development_seeds"],
    }
    branches = build_branches(
        SimpleNamespace(expected_canonical_alpha=0.11), run_spec, grid
    )
    assert len(branches) == EXPECTED_BRANCHES
    assert len({branch.branch_id for branch in branches}) == EXPECTED_BRANCHES
    assert {branch.seed for branch in branches} == {200, 201, 202, 203}
    assert {branch.template_values["advantage_estimator"] for branch in branches} == {
        "td",
        "gae",
    }
    assert {branch.template_values["actor_update_mode"] for branch in branches} == {
        "a2c",
        "ppo_clip_k4",
    }
    assert json.loads((repo / "configs/e7_sqexp_gae_v1.json").read_text())[
        "held_out_seeds"
    ] == [204, 205, 206, 207]
