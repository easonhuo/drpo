from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from drpo import e7_sqexp_gae as runner
from drpo import e7_sqexp_gae_contract as protocol


GRID = Path("configs/e7_sqexp_gae_v1.json")


def _run_spec() -> dict[str, object]:
    digest = "0" * 64
    return {
        "datasets": [
            {
                "id": dataset,
                "path": f"/tmp/{dataset}.hdf5",
                "sha256": digest,
            }
            for dataset in protocol.EXPECTED_DATASETS
        ],
        "seeds": list(runner.EXPECTED_SEEDS),
    }


def test_frozen_matrix_has_192_unique_development_branches() -> None:
    runner._activate_protocol()  # noqa: SLF001
    grid, _ = protocol.load_grid(GRID)
    contract = SimpleNamespace(expected_canonical_alpha=0.11)

    branches = protocol.build_branches(contract, _run_spec(), grid)

    assert len(branches) == 192
    assert len({branch.branch_id for branch in branches}) == 192
    assert {branch.seed for branch in branches} == set(runner.EXPECTED_SEEDS)
    assert not ({branch.seed for branch in branches} & set(runner.HELD_OUT_SEEDS))
    assert {
        branch.template_values["advantage_estimator"] for branch in branches
    } == {"td", "gae"}
    assert {
        branch.template_values["actor_update_mode"] for branch in branches
    } == {"a2c", "ppo_clip_k4"}


def test_matrix_contains_positive_only_and_three_registered_coefficients() -> None:
    runner._activate_protocol()  # noqa: SLF001
    grid, _ = protocol.load_grid(GRID)
    contract = SimpleNamespace(expected_canonical_alpha=0.11)

    branches = protocol.build_branches(contract, _run_spec(), grid)
    controls = [
        (
            branch.template_values["weight_method"],
            float(branch.template_values["exp_coefficient"]),
        )
        for branch in branches
    ]

    assert controls.count(("positive_only", 0.0)) == 48
    assert {
        coefficient
        for method, coefficient in controls
        if method == "squared_exponential"
    } == {64.0, 128.0, 256.0}


def test_grid_rejects_coefficient_shortlist_drift(tmp_path: Path) -> None:
    runner._activate_protocol()  # noqa: SLF001
    raw = json.loads(GRID.read_text())
    changed = copy.deepcopy(raw)
    changed["weight_control"]["exp_coefficients"] = [128.0]
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(changed))

    with pytest.raises(ValueError, match="coefficient shortlist changed"):
        protocol.load_grid(path)


def test_plan_gate_requires_prepared_artifact(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="prepare TD/GAE artifacts"):
        runner._verify_prepared(  # noqa: SLF001
            tmp_path / "missing",
            dataset_id="hopper-medium-expert-v2",
            dataset_sha256="0" * 64,
            seed=200,
        )
