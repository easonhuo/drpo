from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from drpo import e7_sqexp_highc_boundary as boundary
from drpo import e7_sqexp_highc_boundary_aggregate as aggregate
from drpo import e7_sqexp_highc_boundary_bootstrap as bootstrap


GRID = Path("configs/e7_sqexp_highc_boundary_v1.json")


def _run_spec() -> dict[str, object]:
    digest = "0" * 64
    return {
        "datasets": [
            {"id": dataset, "path": f"/tmp/{dataset}.hdf5", "sha256": digest}
            for dataset in boundary.EXPECTED_DATASETS
        ],
        "seeds": list(boundary.EXPECTED_SEEDS),
    }


def _contract(tmp_path: Path | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        expected_canonical_alpha=0.11,
        source_root=Path("/tmp") if tmp_path is None else tmp_path,
    )


def test_grid_freezes_48_branch_boundary_extension() -> None:
    grid, digest = boundary.load_grid(GRID)
    assert len(digest) == 64
    assert grid["experiment_id"] == boundary.EXPERIMENT_ID
    assert grid["predecessor"]["implementation_commit"] == (
        boundary.PREDECESSOR_IMPLEMENTATION_COMMIT
    )
    assert grid["development_seeds"] == [200, 201, 202, 203]
    assert grid["held_out_seeds"] == [204, 205, 206, 207]
    assert [item["id"] for item in grid["controls"]] == [
        "squared_c256",
        "squared_c512",
    ]
    assert [item["id"] for item in grid["actor_modes"]] == [
        "a2c",
        "ppo_clip_kl_k4",
    ]
    assert grid["expected_runnable_branches"] == 48
    assert grid["gae_included"] is False
    assert grid["analysis_contract"]["c1024_extension_not_authorized"] is True


def test_build_branches_creates_exact_paired_matrix() -> None:
    grid, _ = boundary.load_grid(GRID)
    branches = boundary.build_branches(_contract(), _run_spec(), grid)
    assert len(branches) == 48
    assert len({branch.branch_id for branch in branches}) == 48
    assert {branch.seed for branch in branches} == {200, 201, 202, 203}
    assert all(branch.seed not in boundary.HELD_OUT_SEEDS for branch in branches)
    for dataset in boundary.EXPECTED_DATASETS:
        for actor in boundary.EXPECTED_ACTOR_MODES:
            for control in boundary.EXPECTED_CONTROL_IDS:
                seeds = sorted(
                    branch.seed
                    for branch in branches
                    if branch.dataset.id == dataset
                    and branch.template_values["actor_update_mode"] == actor
                    and branch.template_values["control_id"] == control
                )
                assert seeds == [200, 201, 202, 203]


def test_command_contract_uses_supported_variant_and_thin_bootstrap(
    tmp_path: Path,
) -> None:
    grid, _ = boundary.load_grid(GRID)
    branches = boundary.build_branches(_contract(tmp_path), _run_spec(), grid)
    branch = next(
        item
        for item in branches
        if item.template_values["actor_update_mode"] == "ppo_clip_kl_k4"
        and item.template_values["control_id"] == "squared_c512"
        and item.seed == 203
    )
    command, config = boundary.branch_command(
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
    assert "drpo.e7_sqexp_highc_boundary_bootstrap" in command
    assert "iqlv_exp_rank" in command
    assert "iqlv_squared_exp_night" not in command
    assert config["experiment_id"] == boundary.EXPERIMENT_ID
    assert config["predecessor_implementation_commit"] == (
        boundary.PREDECESSOR_IMPLEMENTATION_COMMIT
    )
    assert config["actor_update"] == {
        "id": "ppo_clip_kl_k4",
        "clip_epsilon": 0.2,
        "max_updates_per_old_policy": 4,
        "analytic_kl_early_refresh": True,
        "target_kl": 0.01,
        "kl_penalty": False,
    }
    assert config["weight_control"]["id"] == "squared_c512"
    assert config["weight_control"]["exp_coefficient"] == 512.0
    assert "negative_scale" not in str(config)
    assert "canonical_alpha" not in str(config)


def test_bootstrap_accepts_only_authorized_high_c_controls() -> None:
    accepted = bootstrap._validate_weight_control(  # noqa: SLF001
        {
            "id": "squared_c256",
            "family": "squared_exponential",
            "weight_at_zero": 1.0,
            "exp_coefficient": 256.0,
            "reference_distance": 2.0,
            "formula": boundary.SQUARED_FORMULA,
        }
    )
    assert accepted["exp_coefficient"] == 256.0
    with pytest.raises(ValueError, match="unsupported high-c boundary control"):
        bootstrap._validate_weight_control(  # noqa: SLF001
            {
                "id": "squared_c1024",
                "family": "squared_exponential",
                "weight_at_zero": 1.0,
                "exp_coefficient": 1024.0,
                "reference_distance": 2.0,
                "formula": boundary.SQUARED_FORMULA,
            }
        )


def test_pair_summary_reports_every_boundary_signal() -> None:
    rows = [
        {
            "c512_minus_c256_late": 3.0,
            "c512_minus_c256_final": 1.0,
            "c512_minus_c256_best_to_final_drop": -2.0,
            "c512_minus_c256_absolute_late_slope": -1.0,
            "c256_effective_negative_mass_fraction": 0.04,
            "c512_effective_negative_mass_fraction": 0.02,
        },
        {
            "c512_minus_c256_late": -1.0,
            "c512_minus_c256_final": 2.0,
            "c512_minus_c256_best_to_final_drop": 1.0,
            "c512_minus_c256_absolute_late_slope": 0.5,
            "c256_effective_negative_mass_fraction": 0.06,
            "c512_effective_negative_mass_fraction": 0.03,
        },
    ]
    summary = aggregate._pair_summary(rows)  # noqa: SLF001
    assert summary["pair_count"] == 2
    assert summary["c512_minus_c256_late_mean"] == 1.0
    assert summary["c512_late_wins"] == 1
    assert summary["c512_late_losses"] == 1
    assert summary["c512_over_c256_effective_negative_mass_ratio"] == 0.5
