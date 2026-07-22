from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from drpo import e7_sqexp_actor_decision as decision


GRID = Path("configs/e7_sqexp_actor_decision_v1.json")


def _run_spec() -> dict[str, object]:
    digest = "0" * 64
    return {
        "datasets": [
            {"id": dataset, "path": f"/tmp/{dataset}.hdf5", "sha256": digest}
            for dataset in decision.EXPECTED_DATASETS
        ],
        "seeds": list(decision.EXPECTED_SEEDS),
    }


def _contract(tmp_path: Path | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        expected_canonical_alpha=0.11,
        source_root=Path("/tmp") if tmp_path is None else tmp_path,
    )


def test_grid_freezes_192_branch_decision_matrix() -> None:
    grid, digest = decision.load_grid(GRID)
    assert len(digest) == 64
    assert grid["development_seeds"] == [200, 201, 202, 203]
    assert grid["held_out_seeds"] == [204, 205, 206, 207]
    assert [item["id"] for item in grid["controls"]] == list(
        decision.EXPECTED_CONTROL_IDS
    )
    assert [item["id"] for item in grid["actor_modes"]] == list(
        decision.EXPECTED_ACTOR_MODES
    )
    assert grid["diagnostics"]["interval"] == 10_000
    assert grid["diagnostics"]["kl_event_jsonl"] is False
    assert grid["expected_runnable_branches"] == 192
    assert grid["registration_blocks_launch"] is False
    assert grid["gae_included"] is False


def test_build_branches_creates_exact_four_seed_matrix() -> None:
    grid, _ = decision.load_grid(GRID)
    branches = decision.build_branches(_contract(), _run_spec(), grid)
    assert len(branches) == 192
    assert len({branch.branch_id for branch in branches}) == 192
    assert {branch.seed for branch in branches} == {200, 201, 202, 203}
    assert all(branch.seed not in decision.HELD_OUT_SEEDS for branch in branches)
    assert {
        branch.template_values["actor_update_mode"] for branch in branches
    } == set(decision.EXPECTED_ACTOR_MODES)
    assert {
        branch.template_values["control_id"] for branch in branches
    } == set(decision.EXPECTED_CONTROL_IDS)


def test_every_dataset_actor_control_has_four_paired_seeds() -> None:
    grid, _ = decision.load_grid(GRID)
    branches = decision.build_branches(_contract(), _run_spec(), grid)
    for dataset in decision.EXPECTED_DATASETS:
        for actor in decision.EXPECTED_ACTOR_MODES:
            for control in decision.EXPECTED_CONTROL_IDS:
                seeds = sorted(
                    branch.seed
                    for branch in branches
                    if branch.dataset.id == dataset
                    and branch.template_values["actor_update_mode"] == actor
                    and branch.template_values["control_id"] == control
                )
                assert seeds == [200, 201, 202, 203]


def test_controls_preserve_linear_anchor_and_squared_high_c() -> None:
    grid, _ = decision.load_grid(GRID)
    controls = {item["id"]: item for item in decision.controls(grid)}
    assert controls["positive_only"]["formula"] == "w(d)=0"
    assert controls["linear_c12"]["exp_coefficient"] == 12.0
    assert controls["linear_c12"]["formula"] == "w(d)=w(0)*exp(-c*(d/2))"
    assert controls["squared_c64"]["exp_coefficient"] == 64.0
    assert controls["squared_c128"]["exp_coefficient"] == 128.0
    assert controls["squared_c128"]["formula"] == (
        "w(d)=w(0)*exp(-c*(d/2)^2)"
    )


def test_command_contract_uses_supported_variant_and_k4_kl(tmp_path: Path) -> None:
    grid, _ = decision.load_grid(GRID)
    branches = decision.build_branches(_contract(tmp_path), _run_spec(), grid)
    branch = next(
        item
        for item in branches
        if item.template_values["actor_update_mode"] == "ppo_clip_kl_k4"
        and item.template_values["control_id"] == "squared_c64"
        and item.seed == 202
    )
    command, config = decision.branch_command(
        contract_path=tmp_path / "contract.json",
        contract=_contract(tmp_path),
        branch=branch,
        branch_dir=tmp_path / "branch",
        trainer_argv_template=[
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
        ],
    )
    assert "drpo.e7_sqexp_actor_decision_bootstrap" in command
    assert "iqlv_exp_rank" in command
    assert "iqlv_squared_exp_night" not in command
    assert config["actor_update"] == {
        "id": "ppo_clip_kl_k4",
        "clip_epsilon": 0.2,
        "max_updates_per_old_policy": 4,
        "analytic_kl_early_refresh": True,
        "target_kl": 0.01,
        "kl_penalty": False,
    }
    assert config["weight_control"]["id"] == "squared_c64"
    assert "negative_scale" not in str(config)
    assert "canonical_alpha" not in str(config)
