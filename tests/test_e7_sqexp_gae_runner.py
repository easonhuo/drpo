from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from drpo.e7_sqexp_gae import build_branches, load_grid
from drpo.e7_sqexp_gae_protocol import EXPECTED_BRANCHES


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


def test_terminal_audit_checks_critic_and_estimator_matrix(tmp_path: Path) -> None:
    from drpo.e7_sqexp_gae_audit import terminal_audit

    results = []
    for index in range(EXPECTED_BRANCHES):
        branch_id = f"branch_{index:03d}"
        estimator = "td" if index < EXPECTED_BRANCHES // 2 else "gae"
        actor_mode = "a2c" if index % 2 == 0 else "ppo_clip_k4"
        seed = 200 + index % 4
        branch_dir = tmp_path / "branches" / branch_id
        branch_dir.mkdir(parents=True)
        (branch_dir / "branch_manifest.json").write_text(
            json.dumps(
                {
                    "branch": {
                        "seed": seed,
                        "template_values": {
                            "advantage_estimator": estimator,
                            "actor_update_mode": actor_mode,
                        },
                    },
                    "advantage_estimator": estimator,
                    "gae_used": estimator == "gae",
                    "advantage_provenance": {
                        "gae_recomputed_from_td_and_boundaries": True,
                        "gae_matches_prepared_artifact": True,
                    },
                    "critic_initial_state_sha256": "same",
                    "critic_final_state_sha256": "same",
                    "critic_immutability_verified": True,
                }
            )
        )
        results.append({"branch_id": branch_id, "status": "completed"})
    (tmp_path / "RUN_SUMMARY.json").write_text(
        json.dumps(
            {
                "branch_count": EXPECTED_BRANCHES,
                "completed": EXPECTED_BRANCHES,
                "failed": 0,
                "results": results,
            }
        )
    )
    aggregate_dir = tmp_path / "aggregate"
    aggregate_dir.mkdir()
    (aggregate_dir / "gae_vs_td_summary.json").write_text(
        json.dumps({"status": "PASS", "paired_cells": EXPECTED_BRANCHES // 2})
    )
    audit = terminal_audit(tmp_path)
    assert audit["status"] == "PASS"
    assert audit["held_out_seeds_touched"] is False
    assert audit["critic_immutability_failures"] == 0
