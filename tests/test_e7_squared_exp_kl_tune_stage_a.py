from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from drpo import e7_squared_exp_kl_tune_stage_a as stage_a


GRID = Path("configs/e7_squared_exp_kl_tune_stage_a_v1.json")


def _run_spec() -> dict[str, object]:
    digest = "0" * 64
    return {
        "datasets": [
            {
                "id": dataset,
                "path": f"/tmp/{dataset}.hdf5",
                "sha256": digest,
            }
            for dataset in stage_a.EXPECTED_DATASETS
        ],
        "seeds": list(stage_a.EXPECTED_SEEDS),
    }


def _contract(tmp_path: Path | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        expected_canonical_alpha=0.11,
        source_root=Path("/tmp") if tmp_path is None else tmp_path,
    )


def test_grid_freezes_stage_a_matrix() -> None:
    grid, digest = stage_a.load_grid(GRID)
    assert len(digest) == 64
    assert grid["steps"] == 1_000_000
    assert grid["weight_control"]["formula"] == (
        "w(d)=w(0)*exp(-c*(d/2)^2)"
    )
    assert grid["weight_control"]["exp_coefficients"] == [4.0, 8.0, 16.0, 32.0]
    assert [item["id"] for item in grid["reference_lifecycles"]] == list(
        stage_a.EXPECTED_LIFECYCLES
    )
    assert grid["expected_runnable_branches"] == 150
    assert grid["registration_blocks_launch"] is False
    assert grid["formal_evidence_allowed"] is False


def test_build_branches_creates_exact_150_branch_matrix() -> None:
    grid, _ = stage_a.load_grid(GRID)
    branches = stage_a.build_branches(_contract(), _run_spec(), grid)
    assert len(branches) == 150
    assert len({branch.branch_id for branch in branches}) == 150
    assert {branch.dataset.id for branch in branches} == set(
        stage_a.EXPECTED_DATASETS
    )
    assert {branch.seed for branch in branches} == set(stage_a.EXPECTED_SEEDS)
    assert {
        branch.template_values["actor_update_mode"] for branch in branches
    } == set(stage_a.EXPECTED_LIFECYCLES)
    assert all(branch.template_values["stage"] == "stage_a" for branch in branches)
    assert all(branch.template_values["steps"] == "1000000" for branch in branches)
    assert all(branch.seed not in stage_a.HELD_OUT_SEEDS for branch in branches)


def test_each_lifecycle_has_positive_only_plus_four_coefficients() -> None:
    grid, _ = stage_a.load_grid(GRID)
    branches = stage_a.build_branches(_contract(), _run_spec(), grid)
    for dataset in stage_a.EXPECTED_DATASETS:
        for seed in stage_a.EXPECTED_SEEDS:
            for lifecycle_id in stage_a.EXPECTED_LIFECYCLES:
                selected = [
                    branch
                    for branch in branches
                    if branch.dataset.id == dataset
                    and branch.seed == seed
                    and branch.template_values["actor_update_mode"] == lifecycle_id
                ]
                assert len(selected) == 5
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
                } == set(stage_a.EXPECTED_COEFFICIENTS)


def test_lifecycle_contract_separates_fixed_and_adaptive_paths() -> None:
    assert stage_a.lifecycle_spec("ppo_clip_k4") == {
        "clip_epsilon": 0.2,
        "updates_per_old_policy": 4,
        "analytic_kl_early_refresh": False,
        "target_kl": None,
    }
    assert stage_a.lifecycle_spec("ppo_clip_k16")["target_kl"] is None
    assert stage_a.lifecycle_spec("ppo_clip_k16")["updates_per_old_policy"] == 16
    assert stage_a.lifecycle_spec("ppo_clip_kl_k16_t0p003")["target_kl"] == 0.003
    assert stage_a.lifecycle_spec("ppo_clip_kl_k16_t0p01")["target_kl"] == 0.01
    assert stage_a.lifecycle_spec("ppo_clip_kl_k16_t0p03")["target_kl"] == 0.03


def test_branch_command_reaches_supported_variant_and_bootstrap(
    tmp_path: Path,
) -> None:
    grid, _ = stage_a.load_grid(GRID)
    branches = stage_a.build_branches(_contract(tmp_path), _run_spec(), grid)
    branch = next(
        item
        for item in branches
        if item.template_values["actor_update_mode"]
        == "ppo_clip_kl_k16_t0p003"
        and float(item.template_values["exp_coefficient"]) == 16.0
    )
    trainer_template = [
        "--variant",
        "{variant}",
        "--dataset",
        "{dataset_path}",
        "--seed",
        "{seed}",
        "--steps",
        "{steps}",
        "--output_dir",
        "{output_dir}",
    ]
    command, config = stage_a.branch_command(
        contract_path=tmp_path / "contract.json",
        contract=_contract(tmp_path),
        branch=branch,
        branch_dir=tmp_path / "branch",
        trainer_argv_template=trainer_template,
    )
    assert "drpo.e7_squared_exp_kl_tune_stage_a_bootstrap" in command
    assert "iqlv_exp_rank" in command
    assert "iqlv_squared_exp_night" not in command
    assert config["reference_lifecycle"] == {
        "id": "ppo_clip_kl_k16_t0p003",
        "clip_epsilon": 0.2,
        "max_updates_per_old_policy": 16,
        "analytic_kl_early_refresh": True,
        "target_kl": 0.003,
    }
    assert config["weight_control"]["formula"] == (
        "w(d)=w(0)*exp(-c*(d/2)^2)"
    )
    assert "negative_scale" not in str(config)
    assert "canonical_alpha" not in str(config)
