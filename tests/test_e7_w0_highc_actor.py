from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from drpo import e7_w0_highc_actor as pilot
from drpo import e7_w0_highc_actor_bootstrap as bootstrap


def _grid() -> dict:
    return {
        "experiment_id": pilot.EXPERIMENT_ID,
        "run_kind": "pilot",
        "scientific_status": pilot.SCIENTIFIC_STATUS,
        "datasets": list(pilot.EXPECTED_DATASETS),
        "development_seeds": list(pilot.EXPECTED_SEEDS),
        "held_out_seeds": list(pilot.HELD_OUT_SEEDS),
        "weight_at_zero": 1.0,
        "positive_only_anchor": True,
        "exp_coefficients": list(pilot.EXPECTED_COEFFICIENTS),
        "actor_update_modes": list(pilot.EXPECTED_ACTOR_UPDATE_MODES),
        "ppo": {
            "clip_epsilon": 0.2,
            "updates_per_old_policy": 4,
            "diagnostics_interval": 1000,
            "kl_penalty": False,
            "target_kl": False,
            "entropy_bonus": False,
            "actor_gradient_clip": False,
            "value_clip": False,
        },
        "geometry_diagnostics": {
            "interval": 1000,
            "sampled_values_per_update": 16,
            "weight_thresholds": [0.5, 0.1, 0.05, 0.01],
        },
        "steps": 500_000,
        "evaluation_interval": 50_000,
        "evaluation_episodes": 10,
        "expected_controls_per_actor_mode": 7,
        "expected_total_branches": 84,
        "formal_evidence_allowed": False,
    }


def _run_spec() -> dict:
    return {
        "datasets": [
            {"id": name, "path": f"/tmp/{name}.hdf5", "sha256": "0" * 64}
            for name in pilot.EXPECTED_DATASETS
        ],
        "seeds": list(pilot.EXPECTED_SEEDS),
        "trainer_argv_template": [],
        "environment": {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        },
    }


def test_control_grid_contains_positive_only_and_six_high_c_points() -> None:
    points = pilot.control_points(_grid())
    assert points == [
        (0.0, None),
        (1.0, 2.0),
        (1.0, 3.0),
        (1.0, 4.0),
        (1.0, 6.0),
        (1.0, 8.0),
        (1.0, 12.0),
    ]


def test_builds_full_paired_84_branch_matrix() -> None:
    contract = SimpleNamespace(expected_canonical_alpha=0.11)
    branches = pilot.build_w0_branches(contract, _run_spec(), _grid())
    assert len(branches) == 84
    assert len({branch.branch_id for branch in branches}) == 84
    assert {branch.seed for branch in branches} == {200, 201}
    assert {branch.dataset.id for branch in branches} == set(pilot.EXPECTED_DATASETS)
    assert {branch.template_values["actor_update_mode"] for branch in branches} == {
        "a2c",
        "ppo_clip",
    }
    assert all(branch.negative_control is None for branch in branches)
    assert sum("positive_only__w0_0" in branch.branch_id for branch in branches) == 12
    assert sum("__a2c__" in branch.branch_id for branch in branches) == 42
    assert sum("__ppo_clip_eps0p2__k4__" in branch.branch_id for branch in branches) == 42


def test_direct_w0_one_private_conversion_is_exact() -> None:
    public = bootstrap._validate_weight_control(
        {
            "method": "exponential",
            "weight_at_zero": 1.0,
            "exp_coefficient": 6.0,
            "reference_distance": 2.0,
        }
    )
    internal = bootstrap._internal_control(public, canonical_alpha=0.11)
    assert internal.negative_scale == pytest.approx(1.0 / 0.11)
    assert internal.effective_alpha == pytest.approx(1.0)


def test_grid_rejects_legacy_fields_and_changed_coefficients(tmp_path: Path) -> None:
    raw = _grid()
    raw["negative_scale"] = 1.0
    path = tmp_path / "grid.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="forbids legacy"):
        pilot.load_w0_grid(path)

    raw = _grid()
    raw["exp_coefficients"] = [2.0, 4.0, 8.0]
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="coefficient set changed"):
        pilot.load_w0_grid(path)


def test_branch_command_persists_actor_mode_and_public_w0_only(tmp_path: Path) -> None:
    contract = SimpleNamespace(source_root=tmp_path, expected_canonical_alpha=0.11)
    branches = pilot.build_w0_branches(contract, _run_spec(), _grid())
    branch = next(
        value
        for value in branches
        if value.template_values["actor_update_mode"] == "a2c"
        and value.template_values["exp_coefficient"] == "6"
    )
    command, branch_config = pilot.w0_branch_command(
        contract_path=tmp_path / "contract.json",
        contract=contract,
        branch=branch,
        branch_dir=tmp_path / "branch",
        trainer_argv_template=["--variant", "{variant}"],
    )
    assert "drpo.e7_w0_highc_actor_bootstrap" in command
    assert branch_config["template_values"]["actor_update_mode"] == "a2c"
    assert branch_config["weight_control"]["weight_at_zero"] == 1.0
    assert branch_config["weight_control"]["exp_coefficient"] == 6.0
    assert "negative_control" not in branch_config
    serialized = json.dumps(branch_config)
    assert "negative_scale" not in serialized
    assert "canonical_alpha" not in serialized
