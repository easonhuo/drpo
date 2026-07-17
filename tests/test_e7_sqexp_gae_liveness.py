from __future__ import annotations

import pytest

from drpo import e7_canonical_sweep as base
from scripts.run_e7_sqexp_gae_liveness import (
    REPRESENTATIVE_COEFFICIENT,
    _probe_branch,
    _probe_template,
    _representative,
    _validate_matched_critic,
)


def _branch(estimator: str) -> base.Branch:
    return base.Branch(
        branch_id=f"hopper-medium-expert-v2__seed200__{estimator}__sqexp_c128__a2c__steps1m",
        branch_kind="injected",
        dataset=base.DatasetSpec(
            id="hopper-medium-expert-v2",
            path="unused.hdf5",
            sha256="0" * 64,
        ),
        seed=200,
        template_values={
            "advantage_estimator": estimator,
            "weight_method": "squared_exponential",
            "exp_coefficient": str(REPRESENTATIVE_COEFFICIENT),
            "steps": "1000000",
        },
        negative_control=None,
    )


def test_probe_template_changes_only_liveness_evaluation() -> None:
    template = [
        "--steps",
        "{steps}",
        "--eval_interval",
        "50000",
        "--eval_episodes",
        "10",
    ]
    result = _probe_template({"trainer_argv_template": template}, 4001)
    assert result == [
        "--steps",
        "{steps}",
        "--eval_interval",
        "4001",
        "--eval_episodes",
        "1",
    ]
    assert template[-3:] == ["50000", "--eval_episodes", "10"]


def test_representative_pair_and_probe_identity() -> None:
    branches = [_branch("td"), _branch("gae")]
    selected = _representative(branches, "gae")
    probe = _probe_branch(selected, 4001)
    assert selected.template_values["steps"] == "1000000"
    assert probe.template_values["steps"] == "4001"
    assert probe.branch_id.endswith("liveness_steps4001")


def test_liveness_requires_identical_td_gae_critic_snapshots() -> None:
    records = [
        {"estimator": "td", "snapshot_hashes": ["a", "b"]},
        {"estimator": "gae", "snapshot_hashes": ["a", "b"]},
    ]
    _validate_matched_critic(records)
    records[1]["snapshot_hashes"] = ["a", "c"]
    with pytest.raises(RuntimeError, match="trajectories diverged"):
        _validate_matched_critic(records)
