from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from drpo import e7_ppo_w0_bootstrap as bootstrap
from drpo import e7_ppo_w0_grid_pilot as pilot
from drpo.e7_canonical_injection import NegativeControl, taper_factor


def _grid() -> dict:
    return {
        "experiment_id": pilot.EXPERIMENT_ID,
        "run_kind": "pilot",
        "scientific_status": pilot.SCIENTIFIC_STATUS,
        "datasets": list(pilot.EXPECTED_DATASETS),
        "development_seeds": list(pilot.EXPECTED_SEEDS),
        "held_out_seeds": list(pilot.HELD_OUT_SEEDS),
        "weight_at_zero_grid": list(pilot.EXPECTED_W0),
        "exp_coefficients": list(pilot.EXPECTED_COEFFICIENTS),
        "actor_update_mode": "ppo_clip",
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
        "steps": 500_000,
        "evaluation_interval": 50_000,
        "evaluation_episodes": 10,
        "expected_unique_parameter_points": 31,
        "expected_total_branches": 186,
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


def test_parameter_grid_contains_31_unique_points_and_one_positive_only() -> None:
    points = pilot.parameter_points(_grid())
    assert len(points) == 31
    assert len(set(points)) == 31
    assert points.count((0.0, None)) == 1
    assert all(w0 > 0.0 for w0, _ in points[1:])


def test_builds_full_186_branch_ppo_only_matrix() -> None:
    contract = SimpleNamespace(expected_canonical_alpha=0.11)
    branches = pilot.build_w0_branches(contract, _run_spec(), _grid())
    assert len(branches) == 186
    assert len({branch.branch_id for branch in branches}) == 186
    assert {branch.seed for branch in branches} == {200, 201}
    assert {branch.dataset.id for branch in branches} == set(pilot.EXPECTED_DATASETS)
    assert {branch.template_values["actor_update_mode"] for branch in branches} == {
        "ppo_clip"
    }
    assert all(branch.negative_control is None for branch in branches)
    assert sum("positive_only__w0_0" in branch.branch_id for branch in branches) == 6


def test_direct_w0_011_matches_historical_alpha_011_scale_1() -> None:
    distance = torch.tensor([0.0, 1.0, 2.0, 4.0])
    old = NegativeControl(
        method="exponential",
        negative_scale=1.0,
        canonical_alpha=0.11,
        reference_distance=2.0,
        exponential_coefficient=1.0,
    )
    public = bootstrap._validate_weight_control(
        {
            "method": "exponential",
            "weight_at_zero": 0.11,
            "exp_coefficient": 1.0,
            "reference_distance": 2.0,
        }
    )
    new_internal = bootstrap._internal_control(public, canonical_alpha=0.11)
    assert new_internal.effective_alpha == pytest.approx(0.11)
    assert torch.allclose(
        old.effective_alpha * taper_factor(distance, old),
        new_internal.effective_alpha * taper_factor(distance, new_internal),
    )


def test_grid_rejects_legacy_scale_and_alpha_fields(tmp_path: Path) -> None:
    raw = _grid()
    raw["negative_scale"] = 1.0
    raw["canonical_alpha"] = 0.11
    path = tmp_path / "grid.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="forbids legacy"):
        pilot.load_w0_grid(path)


def test_branch_command_persists_only_public_w0_contract(tmp_path: Path) -> None:
    contract = SimpleNamespace(source_root=tmp_path, expected_canonical_alpha=0.11)
    branch = pilot.build_w0_branches(contract, _run_spec(), _grid())[6]
    command, branch_config = pilot.w0_branch_command(
        contract_path=tmp_path / "contract.json",
        contract=contract,
        branch=branch,
        branch_dir=tmp_path / "branch",
        trainer_argv_template=["--variant", "{variant}"],
    )
    assert "drpo.e7_ppo_w0_bootstrap" in command
    assert command[-2:] == ["--variant", "iqlv_exp_rank"]
    assert "weight_control" in branch_config
    assert "negative_control" not in branch_config
    serialized = json.dumps(branch_config)
    assert "negative_scale" not in serialized
    assert "canonical_alpha" not in serialized


def test_bootstrap_sanitizes_legacy_diagnostics(tmp_path: Path) -> None:
    jsonl = tmp_path / "ppo_diagnostics.jsonl"
    latest = tmp_path / "PPO_DIAGNOSTICS_LATEST.json"
    record = {
        "status": "complete",
        "update": 500_000,
        "negative_control": {
            "negative_scale": 1.0,
            "canonical_alpha": 0.11,
        },
    }
    jsonl.write_text(json.dumps(record) + "\n")
    latest.write_text(json.dumps(record))
    public = {
        "method": "exponential",
        "weight_at_zero": 0.11,
        "exp_coefficient": 1.0,
        "reference_distance": 2.0,
        "formula": "w(d)=w(0)*exp(-c*(d/2))",
    }
    bootstrap._sanitize_diagnostics(jsonl, latest, public)
    assert "negative_scale" not in jsonl.read_text()
    assert "canonical_alpha" not in latest.read_text()
    assert json.loads(latest.read_text())["weight_control"] == public
