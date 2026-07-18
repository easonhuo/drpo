from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from drpo import e7_sqexp_gae as gae
from drpo import e7_squared_exp_night as night


GRID = Path("configs/e7_squared_exp_night_v1.json")
GAE_GRID = Path("configs/e7_sqexp_gae_v1.json")


def _run_spec() -> dict[str, object]:
    digest = "0" * 64
    return {
        "datasets": [
            {
                "id": dataset,
                "path": f"/tmp/{dataset}.hdf5",
                "sha256": digest,
            }
            for dataset in night.EXPECTED_DATASETS
        ],
        "seeds": list(night.EXPECTED_SEEDS),
    }


def test_grid_freezes_squared_kernel_and_blocks_gae() -> None:
    grid, digest = night.load_grid(GRID)
    assert len(digest) == 64
    assert grid["steps"] == 1_000_000
    assert grid["weight_control"]["formula"] == "w(d)=w(0)*exp(-c*(d/2)^2)"
    assert grid["weight_control"]["exp_coefficients"] == [
        0.25,
        0.5,
        1.0,
        2.0,
        4.0,
        8.0,
    ]
    stage_c = {stage["id"]: stage for stage in grid["stages"]}["stage_c_gae"]
    assert stage_c["enabled"] is False
    assert stage_c["gae_lambda"] == 0.95
    assert stage_c["status"] == "blocked_pending_verified_trajectory_contract"


def test_build_branches_creates_exact_126_branch_matrix() -> None:
    grid, _ = night.load_grid(GRID)
    contract = SimpleNamespace(expected_canonical_alpha=0.11)
    branches = night.build_branches(contract, _run_spec(), grid)
    assert len(branches) == 126
    assert len({branch.branch_id for branch in branches}) == 126
    assert {branch.dataset.id for branch in branches} == set(night.EXPECTED_DATASETS)
    assert {branch.seed for branch in branches} == set(night.EXPECTED_SEEDS)
    assert {
        branch.template_values["actor_update_mode"] for branch in branches
    } == set(night.EXPECTED_ACTOR_MODES)
    assert {
        branch.template_values["stage"] for branch in branches
    } == {"stage_a", "stage_b"}
    assert all(branch.template_values["steps"] == "1000000" for branch in branches)
    assert all(seed not in night.HELD_OUT_SEEDS for seed in {b.seed for b in branches})


def test_each_actor_mode_has_positive_only_plus_six_coefficients() -> None:
    grid, _ = night.load_grid(GRID)
    contract = SimpleNamespace(expected_canonical_alpha=0.11)
    branches = night.build_branches(contract, _run_spec(), grid)
    for dataset in night.EXPECTED_DATASETS:
        for seed in night.EXPECTED_SEEDS:
            for actor_mode in night.EXPECTED_ACTOR_MODES:
                selected = [
                    branch
                    for branch in branches
                    if branch.dataset.id == dataset
                    and branch.seed == seed
                    and branch.template_values["actor_update_mode"] == actor_mode
                ]
                assert len(selected) == 7
                points = {
                    (
                        branch.template_values["weight_method"],
                        float(branch.template_values["exp_coefficient"]),
                    )
                    for branch in selected
                }
                assert ("positive_only", 0.0) in points
                assert {
                    coefficient
                    for method, coefficient in points
                    if method == "squared_exponential"
                } == set(night.EXPECTED_COEFFICIENTS)


def test_branch_command_exposes_no_legacy_scale(tmp_path: Path) -> None:
    grid, _ = night.load_grid(GRID)
    contract = SimpleNamespace(
        expected_canonical_alpha=0.11,
        source_root=tmp_path,
    )
    branch = night.build_branches(contract, _run_spec(), grid)[0]
    trainer_template = [
        "--dataset",
        "{dataset_path}",
        "--seed",
        "{seed}",
        "--steps",
        "{steps}",
        "--output_dir",
        "{output_dir}",
    ]
    command, config = night.branch_command(
        contract_path=tmp_path / "contract.json",
        contract=contract,
        branch=branch,
        branch_dir=tmp_path / "branch",
        trainer_argv_template=trainer_template,
    )
    assert "drpo.e7_squared_exp_night_bootstrap" in command
    assert config["weight_control"]["formula"] == "w(d)=w(0)*exp(-c*(d/2)^2)"
    assert "negative_control" not in config
    assert "negative_scale" not in str(config)
    assert "canonical_alpha" not in str(config)


def test_gae_boundaries_and_lambda_zero() -> None:
    td = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    terminal = np.array([False, True, False, False])
    timeout = np.array([False, False, True, False])
    result = gae.compute_gae_from_td(td, terminal, timeout, gamma=0.9, gae_lambda=0.8)
    np.testing.assert_allclose(
        result,
        np.array([1.0 + 0.9 * 0.8 * 2.0, 2.0, 3.0, 4.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        gae.compute_gae_from_td(td, terminal, timeout, gae_lambda=0.0),
        td,
    )


def test_gae_grid_builds_exact_192_branch_matrix() -> None:
    grid, digest = gae._load_grid(GAE_GRID)
    assert len(digest) == 64
    run_spec = _run_spec()
    run_spec["seeds"] = list(gae.EXPECTED_SEEDS)
    contract = SimpleNamespace(expected_canonical_alpha=0.11)
    branches = gae._build_branches(contract, run_spec, grid)
    assert len(branches) == 192
    assert len({branch.branch_id for branch in branches}) == 192
    assert {branch.template_values["advantage_estimator"] for branch in branches} == {
        "td",
        "gae",
    }
    assert {branch.template_values["actor_update_mode"] for branch in branches} == {
        "a2c",
        "ppo_clip_k4",
    }
    assert {branch.seed for branch in branches} == set(gae.EXPECTED_SEEDS)
    assert not ({branch.seed for branch in branches} & set(gae.HELD_OUT_SEEDS))
