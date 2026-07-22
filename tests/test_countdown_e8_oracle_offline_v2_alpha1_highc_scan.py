from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

from drpo import countdown_e8_alpha1_c_scan_common as predecessor
from drpo import countdown_e8_alpha1_highc_scan_common as scan


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs"
    / "countdown_e8_oracle_offline_v2_alpha1_highc_scan_0p5b.yaml"
)
TRAINER = ROOT / "src" / "drpo" / "countdown_e8_alpha1_c_scan_trainer.py"
RUNTIME = ROOT / "src" / "drpo" / "countdown_e8_alpha1_highc_scan_runtime.py"
AUTO = (
    ROOT
    / "scripts"
    / "run_countdown_e8_oracle_offline_v2_alpha1_highc_scan_auto.py"
)
ONE_CLICK = (
    ROOT
    / "scripts"
    / "run_countdown_e8_oracle_offline_v2_alpha1_highc_scan_auto_one_click.sh"
)


def _config() -> dict:
    value = yaml.safe_load(CONFIG.read_text())
    assert isinstance(value, dict)
    return value


def test_linear_scan_has_16_points_and_32_cells() -> None:
    config = _config()
    points = scan.parameter_points(config)
    cells = scan.build_cells(config)
    assert points == (
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 0.051293294),
        (1.0, 0.105360516),
        (1.0, 0.162518929),
        (1.0, 0.223143551),
        (1.0, 0.287682072),
        (1.0, 0.430782916),
        (1.0, 0.693147181),
        (1.0, 0.916290732),
        (1.0, 1.203972804),
        (1.0, 1.386294361),
        (1.0, 1.609437912),
        (1.0, 1.897119985),
        (1.0, 2.302585093),
        (1.0, 2.995732274),
    )
    assert len(cells) == 32
    assert len({cell.name for cell in cells}) == 32
    assert {cell.seed_offset for cell in cells} == {4000, 5000}
    assert sum(cell.method == "positive_only" for cell in cells) == 2
    assert sum(cell.method == "global" for cell in cells) == 2
    assert sum(cell.method == "continuous_exp" for cell in cells) == 28
    assert config["execution"]["default_gpus"] == list(range(8))
    assert config["execution"]["parallel_cells_per_gpu"] == 2


def test_predecessor_module_is_restored_after_scoped_validation() -> None:
    original = (
        predecessor.EXPERIMENT_ID,
        predecessor.PARAMETER_POINTS,
        predecessor.SEED_OFFSETS,
        predecessor.continuous_exp_weights,
    )
    scan.validate_grid_config(_config())
    assert (
        predecessor.EXPERIMENT_ID,
        predecessor.PARAMETER_POINTS,
        predecessor.SEED_OFFSETS,
        predecessor.continuous_exp_weights,
    ) == original


def test_activation_exposes_new_identity_only_inside_context() -> None:
    original_id = predecessor.EXPERIMENT_ID
    with scan.activated():
        assert predecessor.EXPERIMENT_ID == scan.EXPERIMENT_ID
        assert predecessor.PARAMETER_POINTS == scan.PARAMETER_POINTS
        assert predecessor.SEED_OFFSETS == scan.SEED_OFFSETS
        assert predecessor.continuous_exp_weights is scan.continuous_exp_weights
        assert predecessor.validate_grid_config is scan.validate_grid_config
        assert predecessor.build_cells is scan.build_cells
    assert predecessor.EXPERIMENT_ID == original_id


def test_weight_formula_removes_only_the_extra_square() -> None:
    seq_lp = torch.tensor([-2.0, -4.0])
    actual = scan.continuous_exp_weights(seq_lp, alpha=1.0, c=1.0)
    u = torch.tensor([1.0, 2.0])
    assert torch.allclose(actual, torch.exp(-u))
    assert not torch.allclose(actual, torch.exp(-u.square()))
    assert actual.requires_grad is False


def test_real_runtime_plan_uses_linear_validation(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    bank = tmp_path / "bank.jsonl"
    val = tmp_path / "val.jsonl"
    bank.write_text("{}\n", encoding="utf-8")
    val.write_text("{}\n", encoding="utf-8")
    work = tmp_path / "work"
    subprocess.run(
        [
            sys.executable,
            str(RUNTIME),
            "plan",
            "--model_path",
            str(model),
            "--bank",
            str(bank),
            "--val",
            str(val),
            "--base_config",
            str(ROOT / "configs/countdown_e8_base_rl_replay_0p5b.yaml"),
            "--grid_config",
            str(CONFIG),
            "--work_dir",
            str(work),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads((work / "SWEEP_PLAN.json").read_text(encoding="utf-8"))
    assert plan["experiment_id"] == scan.EXPERIMENT_ID
    assert plan["parameter_points"] == 16
    assert plan["cell_count"] == 32


def test_config_rejects_positive_only_seed_gpu_or_metric_drift() -> None:
    config = _config()
    config["sweep"]["parameter_points"][0] = {
        "alpha": 1.0,
        "c": 0.0,
        "role": "unauthorized",
    }
    with pytest.raises(ValueError, match="Exactly one Positive-only"):
        scan.validate_grid_config(config)

    config = _config()
    config["sweep"]["seed_offsets"][0] = 3000
    with pytest.raises(ValueError, match="development seed offsets changed"):
        scan.validate_grid_config(config)

    config = _config()
    config["execution"]["default_gpus"] = list(range(7))
    with pytest.raises(ValueError, match="requires GPU 0-7"):
        scan.validate_grid_config(config)

    config = _config()
    config["evaluation"]["primary_selection_metric"] = "terminal_pass_at_8"
    with pytest.raises(ValueError, match="Primary selection metric"):
        scan.validate_grid_config(config)


def test_runtime_auto_and_manifest_provenance_are_current() -> None:
    runtime_source = RUNTIME.read_text()
    auto_source = AUTO.read_text()
    trainer_source = TRAINER.read_text()
    assert "highc.activate()" in runtime_source
    assert "countdown_e8_alpha1_c_scan_runtime" in runtime_source
    assert "countdown_e8_alpha1_highc_scan_runtime.py" in auto_source
    assert "e8_alpha1_highc_scan_cuda_dev_v1" in auto_source
    assert "configured GPUs" in auto_source
    assert 'grid_config["remoteness"]["weight"]' in trainer_source


def test_one_click_uses_highc_config_and_unregistered_acknowledgement() -> None:
    source = ONE_CLICK.read_text()
    assert "countdown_e8_oracle_offline_v2_alpha1_highc_scan_0p5b.yaml" in source
    assert "--allow-dev-unregistered" in source
    assert "E8_ALPHA1_HIGHC_MAX_DEVICES:-8" in source


EXTENSION_CONFIG = (
    ROOT
    / "configs"
    / "countdown_e8_oracle_offline_v2_linear_c_extension_0p5b.yaml"
)
EXTENSION_RUNSPEC = (
    ROOT
    / "runspecs"
    / "ready"
    / "E8_PAPER_ALIGNED_LINEAR_C_EXTENSION_20260717_01.yaml"
)


def _load_path(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_extension_has_eight_new_c_points_and_sixteen_cells() -> None:
    config = _load_path(EXTENSION_CONFIG)
    points = scan.parameter_points(config)
    cells = scan.build_cells(config)
    assert points == scan.C_EXTENSION_PARAMETER_POINTS
    assert len(cells) == 16
    assert len({cell.name for cell in cells}) == 16
    assert {cell.seed_offset for cell in cells} == {4000, 5000}
    assert {cell.alpha for cell in cells} == {1.0}
    assert {cell.method for cell in cells} == {"continuous_exp"}
    assert not any(cell.c in {0.0, 0.051293294, 2.995732274} for cell in cells)
    assert config["sweep"]["positive_only_same_seed_control"] is False
    assert config["historical_reference"]["positive_only_rerun_in_this_round"] is False


def test_round1_profile_remains_unchanged_after_extension_validation() -> None:
    original = (
        scan.EXPERIMENT_ID,
        scan.PARAMETER_POINTS,
        scan.SEED_OFFSETS,
        scan.EXPECTED_POINTS,
        scan.EXPECTED_CELLS,
    )
    scan.validate_grid_config(_load_path(EXTENSION_CONFIG))
    assert (
        scan.EXPERIMENT_ID,
        scan.PARAMETER_POINTS,
        scan.SEED_OFFSETS,
        scan.EXPECTED_POINTS,
        scan.EXPECTED_CELLS,
    ) == original
    assert scan.parameter_points(_config()) == scan.ROUND1_PARAMETER_POINTS


def test_extension_activation_updates_base_identity_only_inside_context() -> None:
    config = _load_path(EXTENSION_CONFIG)
    profile = scan._profile_for_config(config)
    original_id = predecessor.EXPERIMENT_ID
    with scan.activated(profile):
        assert predecessor.EXPERIMENT_ID == scan.C_EXTENSION_EXPERIMENT_ID
        assert predecessor.PARAMETER_POINTS == scan.C_EXTENSION_PARAMETER_POINTS
        assert predecessor.SEED_OFFSETS == (4000, 5000)
        assert predecessor.EXPECTED_POINTS == 8
        assert predecessor.EXPECTED_CELLS == 16
    assert predecessor.EXPERIMENT_ID == original_id


def test_extension_rejects_alpha_positive_only_and_reference_drift() -> None:
    config = _load_path(EXTENSION_CONFIG)
    config["sweep"]["parameter_points"][0]["alpha"] = 0.5
    with pytest.raises(ValueError, match="alpha fixed at 1"):
        scan.validate_grid_config(config)

    config = _load_path(EXTENSION_CONFIG)
    config["sweep"]["positive_only_same_seed_control"] = True
    with pytest.raises(ValueError, match="must not rerun Positive-only"):
        scan.validate_grid_config(config)

    config = _load_path(EXTENSION_CONFIG)
    config["historical_reference"]["result_manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Historical result manifest"):
        scan.validate_grid_config(config)


def test_runtime_plan_selects_extension_profile(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    bank = tmp_path / "bank.jsonl"
    val = tmp_path / "val.jsonl"
    bank.write_text("{}\n", encoding="utf-8")
    val.write_text("{}\n", encoding="utf-8")
    work = tmp_path / "work"
    subprocess.run(
        [
            sys.executable,
            str(RUNTIME),
            "plan",
            "--model_path",
            str(model),
            "--bank",
            str(bank),
            "--val",
            str(val),
            "--base_config",
            str(ROOT / "configs/countdown_e8_base_rl_replay_0p5b.yaml"),
            "--grid_config",
            str(EXTENSION_CONFIG),
            "--work_dir",
            str(work),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads((work / "SWEEP_PLAN.json").read_text(encoding="utf-8"))
    assert plan["experiment_id"] == scan.C_EXTENSION_EXPERIMENT_ID
    assert plan["parameter_points"] == 8
    assert plan["cell_count"] == 16
    assert {row["method"] for row in plan["cells"]} == {"continuous_exp"}


def test_extension_runspec_enables_default_e8_results_delivery() -> None:
    spec = _load_path(EXTENSION_RUNSPEC)
    assert spec["registration"] == {"mode": "deferred", "closure_required": True}
    assert spec["delivery"] == {
        "enabled": True,
        "auto": True,
        "mode": "results_repo",
        "repository": "easonhuo/drpo-results",
        "branch": "ingest/e8",
        "export_profile": "manifest_text_v1",
        "max_total_size_mb": 30,
        "max_file_size_mb": 10,
    }
    assert "E8_ALPHA1_HIGHC_GRID_CONFIG=" in spec["entrypoint"]["command"]
    assert "16 parameter points" not in "\n".join(spec["success_criteria"])
    assert any(
        "8 parameter points and 16 cells" in item
        for item in spec["success_criteria"]
    )


def test_extension_reuses_existing_launcher_without_new_training_stack() -> None:
    source = AUTO.read_text(encoding="utf-8")
    assert "activate_for_grid_config" in source
    assert "countdown_e8_alpha1_highc_scan_runtime.py" in source
    assert "e8_linear_c_extension_cuda_dev_v1" in source
    assert "run_countdown_e8_oracle_offline_v2_alpha1_c_scan_auto.py" in source


RECIPROCAL_CONFIG = (
    ROOT
    / "configs"
    / "countdown_e8_oracle_offline_v2_reciprocal_shape_screen_0p5b.yaml"
)


def test_reciprocal_screen_has_eight_points_and_sixteen_paired_cells() -> None:
    config = _load_path(RECIPROCAL_CONFIG)
    points = scan.parameter_points(config)
    cells = scan.build_cells(config)
    assert points == scan.RECIPROCAL_SCREEN_POINTS
    assert len(points) == 8
    assert len(cells) == 16
    assert len({cell.name for cell in cells}) == 16
    assert {cell.seed_offset for cell in cells} == {4000, 5000}
    assert sum(cell.method == "reciprocal_linear" for cell in cells) == 8
    assert sum(cell.method == "reciprocal_quadratic" for cell in cells) == 8
    assert {cell.alpha for cell in cells} == {1.0}
    assert {cell.coefficient for cell in cells} == {1.0, 3.0, 7.0, 19.0}
    assert not any(
        cell.method in {"positive_only", "global", "continuous_exp"}
        for cell in cells
    )
    assert config["historical_controls"]["rerun_in_this_round"] is False
    assert config["historical_controls"]["methods"] == [
        "positive_only",
        "global",
        "exponential",
    ]


def test_reciprocal_weight_formulas_use_sqrt_x_and_x() -> None:
    config = _load_path(RECIPROCAL_CONFIG)
    profile = scan._profile_for_config(config)
    seq_lp = torch.tensor([-0.25, -2.25, -8.25])
    with scan.activated(profile):
        linear = scan.continuous_exp_weights(
            seq_lp,
            alpha=1.0,
            c=scan.FamilyCoefficient(3.0, "reciprocal_linear"),
        )
        quadratic = scan.continuous_exp_weights(
            seq_lp,
            alpha=1.0,
            c=scan.FamilyCoefficient(3.0, "reciprocal_quadratic"),
        )
    expected_linear = torch.tensor([1.0, 0.25, 1.0 / 7.0])
    expected_quadratic = torch.tensor([1.0, 0.25, 1.0 / 13.0])
    assert torch.allclose(linear, expected_linear)
    assert torch.allclose(quadratic, expected_quadratic)
    assert not torch.allclose(linear, quadratic)
    assert linear.requires_grad is False
    assert quadratic.requires_grad is False


def test_reciprocal_config_rejects_seed_or_historical_control_drift() -> None:
    config = _load_path(RECIPROCAL_CONFIG)
    config["sweep"]["seed_offsets"] = [4000]
    with pytest.raises(ValueError, match="seed offsets changed"):
        scan.validate_grid_config(config)

    config = _load_path(RECIPROCAL_CONFIG)
    config["historical_controls"]["rerun_in_this_round"] = True
    with pytest.raises(ValueError, match="must not be rerun"):
        scan.validate_grid_config(config)


def test_runtime_plan_selects_reciprocal_profile(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    bank = tmp_path / "bank.jsonl"
    val = tmp_path / "val.jsonl"
    bank.write_text("{}\n", encoding="utf-8")
    val.write_text("{}\n", encoding="utf-8")
    work = tmp_path / "work"
    subprocess.run(
        [
            sys.executable,
            str(RUNTIME),
            "plan",
            "--model_path",
            str(model),
            "--bank",
            str(bank),
            "--val",
            str(val),
            "--base_config",
            str(ROOT / "configs/countdown_e8_base_rl_replay_0p5b.yaml"),
            "--grid_config",
            str(RECIPROCAL_CONFIG),
            "--work_dir",
            str(work),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads((work / "SWEEP_PLAN.json").read_text(encoding="utf-8"))
    assert plan["experiment_id"] == scan.RECIPROCAL_SCREEN_EXPERIMENT_ID
    assert plan["parameter_points"] == 8
    assert plan["cell_count"] == 16
    assert {row["method"] for row in plan["cells"]} == {
        "reciprocal_linear",
        "reciprocal_quadratic",
    }


RECIPROCAL_Q_DENSE_CONFIG = (
    ROOT
    / "configs"
    / "countdown_e8_oracle_offline_v2_reciprocal_quadratic_dense_lambda_curve_0p5b.yaml"
)


def test_reciprocal_q_dense_curve_has_sixteen_points_and_thirty_two_cells() -> None:
    config = _load_path(RECIPROCAL_Q_DENSE_CONFIG)
    points = scan.parameter_points(config)
    cells = scan.build_cells(config)
    assert points == scan.RECIPROCAL_Q_DENSE_POINTS
    assert tuple(point[2] for point in points) == scan.RECIPROCAL_Q_DENSE_LAMBDAS
    assert len(points) == 16
    assert len(cells) == 32
    assert len({cell.name for cell in cells}) == 32
    assert {cell.seed_offset for cell in cells} == {4000, 5000}
    assert {cell.method for cell in cells} == {"reciprocal_quadratic"}
    assert {cell.alpha for cell in cells} == {1.0}
    completed = {1.0, 3.0, 7.0, 19.0, 39.0, 79.0, 159.0, 319.0}
    assert not completed.intersection({cell.coefficient for cell in cells})
    assert config["execution"]["parallel_cells_per_gpu"] == 2
    assert config["execution"]["expected_full_waves"] == 2
    assert config["historical_controls"]["rerun_in_this_round"] is False
    assert config["reporting"]["trend_resolution_claim_only"] is True


ASYMRE_CONFIG = (
    ROOT
    / "configs"
    / "countdown_e8_oracle_offline_v2_asymre_deltav_scan_0p5b.yaml"
)


def test_asymre_scan_has_eight_delta_v_points_and_sixteen_cells() -> None:
    config = _load_path(ASYMRE_CONFIG)
    points = scan.parameter_points(config)
    cells = scan.build_cells(config)
    assert points == scan.ASYMRE_DELTA_VS
    assert len(cells) == 16
    assert len({cell.name for cell in cells}) == 16
    assert {cell.seed_offset for cell in cells} == {4000, 5000}
    assert {cell.method for cell in cells} == {"asymre"}
    assert {cell.family for cell in cells} == {"asymre"}
    assert {cell.coefficient for cell in cells} == {0.0}
    assert {cell.delta_v for cell in cells} == set(scan.ASYMRE_DELTA_VS)
    assert {cell.alpha for cell in cells} == {
        0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0, 1.1,
    }
    assert config["execution"]["default_gpus"] == list(range(8))
    assert config["execution"]["parallel_cells_per_gpu"] == 2


def test_asymre_uses_constant_negative_weights_and_no_distance_control() -> None:
    config = _load_path(ASYMRE_CONFIG)
    profile = scan._profile_for_config(config)
    seq_lp = torch.tensor([-0.25, -2.25, -8.25])
    with scan.activated(profile):
        weights = scan.continuous_exp_weights(
            seq_lp,
            alpha=0.9,
            c=scan.FamilyCoefficient(0.0, "asymre"),
        )
    assert torch.allclose(weights, torch.full_like(seq_lp, 0.9))
    assert weights.requires_grad is False
    assert config["remoteness"] == {
        "enabled": False,
        "weight": "constant_1_no_distance_control",
        "detached": True,
    }


def test_asymre_trainer_changes_only_the_signed_objective_coefficients() -> None:
    source = TRAINER.read_text(encoding="utf-8")
    assert 'cell.method == "asymre"' in source
    assert "(1.0 - cell.delta_v)" in source
    assert "weighted_negative_lp" in source
    assert "value_network" not in source

    positive_lp = torch.tensor(2.0)
    negative_lp = torch.tensor(3.0)
    for delta_v, expected in (
        (0.0, 1.0),
        (-1.0, -4.0),
        (0.1, 1.5),
    ):
        actual = -((1.0 - delta_v) * positive_lp - (1.0 + delta_v) * negative_lp)
        assert float(actual) == pytest.approx(expected)


def test_asymre_config_rejects_delta_v_seed_or_value_network_drift() -> None:
    config = _load_path(ASYMRE_CONFIG)
    config["sweep"]["parameter_points"][0]["delta_v"] = -1.1
    with pytest.raises(ValueError, match="parameter points changed"):
        scan.validate_grid_config(config)

    config = _load_path(ASYMRE_CONFIG)
    config["sweep"]["seed_offsets"] = [4000]
    with pytest.raises(ValueError, match="seed offsets changed"):
        scan.validate_grid_config(config)

    config = _load_path(ASYMRE_CONFIG)
    config["objective"]["value_network"] = True
    with pytest.raises(ValueError, match="must not add a value network"):
        scan.validate_grid_config(config)


ASYMRE_BOUNDARY_DENSE_CONFIG = (
    ROOT
    / "configs"
    / "countdown_e8_oracle_offline_v2_asymre_deltav_boundary_dense_0p5b.yaml"
)


def test_asymre_boundary_dense_has_eight_points_and_sixteen_paired_cells() -> None:
    config = _load_path(ASYMRE_BOUNDARY_DENSE_CONFIG)
    points = scan.parameter_points(config)
    cells = scan.build_cells(config)
    assert points == scan.ASYMRE_BOUNDARY_DENSE_DELTA_VS
    assert points == (-1.0, -0.95, -0.9, -0.85, -0.8, -0.7, -0.6, -0.5)
    assert len(cells) == 16
    assert len({cell.name for cell in cells}) == 16
    assert {cell.seed_offset for cell in cells} == {4000, 5000}
    assert {cell.method for cell in cells} == {"asymre"}
    assert {cell.family for cell in cells} == {"asymre"}
    assert {cell.coefficient for cell in cells} == {0.0}
    assert {cell.delta_v for cell in cells} == set(points)
    assert {cell.alpha for cell in cells} == {
        0.0,
        0.05,
        0.1,
        0.15,
        0.2,
        0.3,
        0.4,
        0.5,
    }
    assert config["sweep"]["cells"] == 16
    assert config["execution"]["expected_full_waves"] == 1
    assert config["predecessor"]["rerun_boundary_points_as_internal_anchors"] is True


def test_asymre_boundary_dense_rejects_parameter_or_seed_drift() -> None:
    config = _load_path(ASYMRE_BOUNDARY_DENSE_CONFIG)
    config["sweep"]["parameter_points"][1]["delta_v"] = -0.94
    with pytest.raises(ValueError, match="parameter points changed"):
        scan.validate_grid_config(config)

    config = _load_path(ASYMRE_BOUNDARY_DENSE_CONFIG)
    config["sweep"]["seed_offsets"] = [4000]
    with pytest.raises(ValueError, match="seed offsets changed"):
        scan.validate_grid_config(config)


def test_runtime_plan_selects_asymre_boundary_dense_profile(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    bank = tmp_path / "bank.jsonl"
    val = tmp_path / "val.jsonl"
    bank.write_text("{}\n", encoding="utf-8")
    val.write_text("{}\n", encoding="utf-8")
    work = tmp_path / "work"
    subprocess.run(
        [
            sys.executable,
            str(RUNTIME),
            "plan",
            "--model_path",
            str(model),
            "--bank",
            str(bank),
            "--val",
            str(val),
            "--base_config",
            str(ROOT / "configs/countdown_e8_base_rl_replay_0p5b.yaml"),
            "--grid_config",
            str(ASYMRE_BOUNDARY_DENSE_CONFIG),
            "--work_dir",
            str(work),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads((work / "SWEEP_PLAN.json").read_text(encoding="utf-8"))
    assert plan["experiment_id"] == scan.ASYMRE_DELTAV_BOUNDARY_DENSE_EXPERIMENT_ID
    assert plan["parameter_points"] == 8
    assert plan["cell_count"] == 16
    assert {row["method"] for row in plan["cells"]} == {"asymre"}
    assert {row["seed_offset"] for row in plan["cells"]} == {4000, 5000}


TOPR_CONFIG = (
    ROOT
    / "configs"
    / "countdown_e8_oracle_offline_v2_joint_fitted_reference_topr_0p5b.yaml"
)
TOPR_RUNSPEC = (
    ROOT
    / "runspecs"
    / "ready"
    / "E8_JOINT_FITTED_REFERENCE_TOPR_20260722_01.yaml"
)


def test_joint_topr_has_eight_beta_points_and_sixteen_paired_cells() -> None:
    config = _load_path(TOPR_CONFIG)
    points = scan.parameter_points(config)
    cells = scan.build_cells(config)
    assert points == scan.JOINT_FITTED_REFERENCE_TOPR_POINTS
    assert tuple(point[2] for point in points) == (
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
        1.5,
        2.0,
        4.0,
    )
    assert len(points) == 8
    assert len(cells) == 16
    assert len({cell.name for cell in cells}) == 16
    assert {cell.seed_offset for cell in cells} == {4000, 5000}
    assert {cell.method for cell in cells} == {"joint_fitted_reference_topr"}
    assert {cell.family for cell in cells} == {"joint_fitted_reference_topr"}
    assert {cell.coefficient for cell in cells} == set(
        scan.JOINT_FITTED_REFERENCE_TOPR_BETAS
    )
    assert config["method_identity"] == (
        "joint_fitted_reference_beta_topr_scan_not_canonical_topr"
    )
    assert config["execution"]["default_gpus"] == [0, 1]
    assert config["execution"]["parallel_cells_per_gpu"] == 1
    assert config["execution"]["expected_full_waves"] == 8


def test_joint_topr_beta_controls_summed_sequence_ratio_and_detached_weights() -> None:
    policy_stats = {
        "seq_lp": torch.tensor([-2.0, -3.0], requires_grad=True),
        "lengths": torch.tensor([2.0, 4.0]),
    }
    reference_stats = {
        "seq_lp": torch.tensor([-1.5, -4.0], requires_grad=True),
        "lengths": torch.tensor([2.0, 4.0]),
    }
    for beta, expected_first in (
        (0.0, 1.0),
        (0.5, float(torch.exp(torch.tensor(-0.5)))),
        (1.0, float(torch.exp(torch.tensor(-1.0)))),
        (2.0, float(torch.exp(torch.tensor(-2.0)))),
    ):
        weights, log_ratio = scan.joint_topr_negative_weights(
            policy_stats,
            reference_stats,
            beta=beta,
        )
        assert torch.allclose(log_ratio, torch.tensor([-1.0, 4.0]))
        assert torch.allclose(weights, torch.tensor([expected_first, 1.0]))
        assert weights.requires_grad is False
        assert log_ratio.requires_grad is False

    with pytest.raises(ValueError, match="finite and non-negative"):
        scan.joint_topr_negative_weights(
            policy_stats,
            reference_stats,
            beta=-0.1,
        )
    with pytest.raises(ValueError, match="finite and non-negative"):
        scan.joint_topr_negative_weights(
            policy_stats,
            reference_stats,
            beta=float("nan"),
        )


def test_joint_topr_branch_balanced_reference_loss_uses_prompt_denominator() -> None:
    positive_lp = torch.tensor(-2.0)
    negative_lp = torch.tensor([-1.0, -3.0, -5.0])
    bank_row_index = torch.tensor([0, 0, 1])
    unique_counts = torch.tensor([2, 1])
    loss = scan.branch_balanced_reference_loss(
        positive_lp,
        negative_lp,
        bank_row_index,
        unique_counts,
    )
    expected_negative = ((-1.0 - 3.0) / 2.0 + -5.0) / 2.0
    assert float(loss) == pytest.approx(-(0.5 * -2.0 + 0.5 * expected_negative))


def test_joint_topr_config_rejects_ratio_beta_reference_or_seed_drift() -> None:
    config = _load_path(TOPR_CONFIG)
    config["objective"]["ratio_coordinate"] = "mean_token_log_ratio"
    with pytest.raises(ValueError, match="summed sequence"):
        scan.validate_grid_config(config)

    config = _load_path(TOPR_CONFIG)
    config["objective"]["negative_weight"] = "exp(min(sum_logpi-sum_logmu,0))"
    with pytest.raises(ValueError, match="beta ratio formula"):
        scan.validate_grid_config(config)

    config = _load_path(TOPR_CONFIG)
    config["sweep"]["parameter_points"][1]["coefficient"] = 0.3
    with pytest.raises(ValueError, match="parameter points changed"):
        scan.validate_grid_config(config)

    config = _load_path(TOPR_CONFIG)
    config["reference_policy"]["positive_branch_mass"] = 0.6
    with pytest.raises(ValueError, match="positive branch mass"):
        scan.validate_grid_config(config)

    config = _load_path(TOPR_CONFIG)
    config["sweep"]["seed_offsets"] = [4000]
    with pytest.raises(ValueError, match="seed offsets changed"):
        scan.validate_grid_config(config)


def test_joint_topr_runspec_freezes_beta_curve_and_delivery() -> None:
    spec = _load_path(TOPR_RUNSPEC)
    criteria = "\n".join(spec["success_criteria"])
    assert spec["registration"] == {"mode": "deferred", "closure_required": True}
    assert "8 beta points and 16 cells" in criteria
    assert "0, 0.25, 0.5, 0.75, 1, 1.5, 2, and 4" in criteria
    assert spec["delivery"] == {
        "enabled": True,
        "auto": True,
        "mode": "results_repo",
        "repository": "easonhuo/drpo-results",
        "branch": "ingest/e8",
        "export_profile": "manifest_text_v1",
        "max_total_size_mb": 30,
        "max_file_size_mb": 10,
    }


def test_joint_topr_reuses_existing_trainer_and_launcher() -> None:
    trainer_source = TRAINER.read_text(encoding="utf-8")
    auto_source = AUTO.read_text(encoding="utf-8")
    assert 'cell.method == "joint_fitted_reference_topr"' in trainer_source
    assert "model.add_adapter" in trainer_source
    assert "joint_topr_negative_weights" in trainer_source
    assert "beta=topr_beta" in trainer_source
    assert '"topr_beta": topr_beta' in trainer_source
    assert "branch_balanced_reference_loss" in trainer_source
    assert "full_completion_summed_log_probability" in trainer_source
    assert "e8_joint_fitted_reference_topr_cuda_dev_v1" in auto_source
    assert "required_devices" in auto_source
