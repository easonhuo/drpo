from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from drpo import e7_canonical_ppo_stability as pilot


def valid_grid() -> dict:
    return {
        "experiment_id": pilot.EXPERIMENT_ID,
        "run_kind": "pilot",
        "scientific_status": pilot.SCIENTIFIC_STATUS,
        "datasets": list(pilot.EXPECTED_DATASETS),
        "development_seeds": list(pilot.EXPECTED_SEEDS),
        "held_out_seeds": [204, 205, 206, 207],
        "canonical_alpha": 0.11,
        "reference_distance": 2.0,
        "exp_coefficients": list(pilot.EXPECTED_COEFFICIENTS),
        "actor_update_modes": list(pilot.EXPECTED_ACTOR_UPDATES),
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
        "steps": 1_000_000,
        "expected_total_branches": 96,
    }


def run_spec() -> dict:
    datasets = [
        {"id": name, "path": f"/{name}.hdf5", "sha256": "0" * 64}
        for name in pilot.EXPECTED_DATASETS
    ]
    return {
        "datasets": datasets,
        "seeds": list(pilot.EXPECTED_SEEDS),
        "trainer_argv_template": [],
    }


def test_grid_validation_rejects_auxiliary_ppo_tricks(tmp_path: Path) -> None:
    grid = valid_grid()
    source = tmp_path / "grid.json"
    source.write_text(json.dumps(grid))
    pilot.load_ppo_grid(source)
    grid["ppo"]["target_kl"] = 0.01
    source.write_text(json.dumps(grid))
    with pytest.raises(ValueError, match="forbids auxiliary PPO tricks"):
        pilot.load_ppo_grid(source)


def test_builds_exact_frozen_96_branch_matrix() -> None:
    contract = SimpleNamespace(expected_canonical_alpha=0.11)
    branches = pilot.build_ppo_branches(contract, run_spec(), valid_grid())
    assert len(branches) == 96
    assert len({branch.branch_id for branch in branches}) == 96
    assert {branch.dataset.id for branch in branches} == set(
        pilot.EXPECTED_DATASETS
    )
    assert {branch.seed for branch in branches} == set(pilot.EXPECTED_SEEDS)
    assert not ({204, 205, 206, 207} & {branch.seed for branch in branches})
    assert {
        branch.template_values["actor_update_mode"]
        for branch in branches
    } == {"a2c", "ppo_clip"}
    assert {branch.template_values["steps"] for branch in branches} == {
        "1000000"
    }
    coefficients = {
        branch.negative_control.exponential_coefficient
        for branch in branches
        if branch.negative_control.method == "exponential"
    }
    assert coefficients == {0.5, 1.0, 1.5}


def test_branch_command_uses_ppo_bootstrap(tmp_path: Path) -> None:
    contract = SimpleNamespace(source_root=tmp_path)
    branches = pilot.build_ppo_branches(
        SimpleNamespace(expected_canonical_alpha=0.11),
        run_spec(),
        valid_grid(),
    )
    command, branch_config = pilot.ppo_branch_command(
        contract_path=tmp_path / "contract.json",
        contract=contract,
        branch=branches[0],
        branch_dir=tmp_path / "branch",
        trainer_argv_template=[],
    )
    assert "drpo.e7_canonical_ppo_bootstrap" in command
    assert branch_config["template_values"]["actor_update_mode"] in {
        "a2c",
        "ppo_clip",
    }
