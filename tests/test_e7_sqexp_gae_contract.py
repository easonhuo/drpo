from __future__ import annotations

import dataclasses
import json
import math
import subprocess
from types import SimpleNamespace
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from drpo import e7_sqexp_gae as gae


CONFIG = Path(__file__).parents[1] / "configs" / "e7_sqexp_gae_v1.yaml"


def protocol() -> gae.FrozenProtocol:
    return gae.load_protocol(CONFIG)


def test_frozen_protocol_and_exact_matrix() -> None:
    value = protocol()
    critics = gae.build_critic_jobs(value)
    branches = gae.build_actor_branches(value)
    assert len(critics) == 12
    assert len(branches) == 192
    assert len({job.id for job in critics}) == 12
    assert len({branch.id for branch in branches}) == 192
    assert {branch.seed for branch in branches} == {200, 201, 202, 203}
    assert not ({204, 205, 206, 207} & {branch.seed for branch in branches})
    assert {branch.estimator for branch in branches} == set(gae.ESTIMATORS)
    assert {branch.actor_mode for branch in branches} == set(gae.ACTOR_MODES)
    assert {branch.control_id for branch in branches} == set(gae.CONTROL_IDS)
    for critic in critics:
        dependent = [
            branch
            for branch in branches
            if branch.dataset_id == critic.dataset_id and branch.seed == critic.seed
        ]
        assert len(dependent) == 16


def test_protocol_rejects_frozen_change(tmp_path: Path) -> None:
    text = CONFIG.read_text().replace("gae_lambda: 0.95", "gae_lambda: 0.90")
    changed = tmp_path / "changed.yaml"
    changed.write_text(text)
    with pytest.raises(ValueError, match="gae_lambda"):
        gae.load_protocol(changed)


def test_source_runspec_accepts_existing_schema_and_resolves_paths(tmp_path: Path) -> None:
    records = []
    for index, dataset_id in enumerate(gae.EXPECTED_DATASETS):
        records.append(
            {
                "id": dataset_id,
                "path": f"data/{index}.hdf5",
                "sha256": f"{index + 1:064x}",
                "format": "legacy_d4rl_hdf5",
                "env_id": "Hopper-v4" if index == 0 else "Walker2d-v4",
                "score_protocol": "d4rl_v2_percent",
                "reference_min_score": -1.0,
                "reference_max_score": 1.0,
            }
        )
    path = tmp_path / "run_spec.json"
    path.write_text(json.dumps({"datasets": records}))
    datasets, digest = gae.load_source_run_spec(path)
    assert tuple(item.id for item in datasets) == gae.EXPECTED_DATASETS
    assert all(Path(item.path).is_absolute() for item in datasets)
    assert datasets[0].path == str((tmp_path / "data/0.hdf5").resolve())
    assert len(digest) == 64


def _boundary_fixture() -> dict[str, np.ndarray]:
    return {
        "rewards": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64),
        "values": np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float64),
        "next_values": np.array([20.0, 999.0, 40.0, 60.0, 70.0], dtype=np.float64),
        "terminals": np.array([False, True, False, False, False]),
        "timeouts": np.array([False, False, True, False, False]),
        "episode_ids": np.array([0, 0, 1, 2, 2]),
    }


def test_gae_boundary_semantics_and_independent_reference() -> None:
    fixture = _boundary_fixture()
    arrays = gae.compute_td_and_gae(**fixture)
    # True terminal: no bootstrap and no carry.
    assert arrays.td_float64[1] == pytest.approx(2.0 - 20.0)
    # Timeout: bootstrap from V(next) but no carry.
    assert arrays.td_float64[2] == pytest.approx(3.0 + 0.99 * 40.0 - 30.0)
    assert arrays.gae_float64[2] == pytest.approx(arrays.td_float64[2])
    # Final stored nonterminal: bootstrap and stop carry.
    assert arrays.td_float64[4] == pytest.approx(5.0 + 0.99 * 70.0 - 50.0)
    assert arrays.gae_float64[4] == pytest.approx(arrays.td_float64[4])
    # Interior rows carry only within the same behavior trajectory.
    assert arrays.gae_float64[0] == pytest.approx(
        arrays.td_float64[0] + 0.99 * 0.95 * arrays.td_float64[1]
    )
    assert arrays.gae_float64[3] == pytest.approx(
        arrays.td_float64[3] + 0.99 * 0.95 * arrays.td_float64[4]
    )
    audit = gae.validate_advantage_numerics(
        arrays,
        episode_ids=fixture["episode_ids"],
        terminals=fixture["terminals"],
        timeouts=fixture["timeouts"],
        gamma=0.99,
        gae_lambda=0.95,
    )
    assert audit["independent_implementation_max_abs_disagreement"] == 0.0
    assert audit["lambda_zero_td_max_abs_disagreement"] == 0.0
    assert audit["storage_quantization_is_not_implementation_disagreement"] is True
    assert arrays.td_float32.dtype == np.float32
    assert arrays.gae_float32.dtype == np.float32


def test_lambda_zero_exactly_equals_td() -> None:
    fixture = _boundary_fixture()
    arrays = gae.compute_td_and_gae(**fixture, gae_lambda=0.0)
    np.testing.assert_array_equal(arrays.gae_float64, arrays.td_float64)
    np.testing.assert_array_equal(arrays.gae_float32, arrays.td_float32)


def test_boundary_validation_rejects_overlap_and_bad_order() -> None:
    fixture = _boundary_fixture()
    overlap = dict(fixture)
    overlap["timeouts"] = fixture["timeouts"].copy()
    overlap["timeouts"][1] = True
    with pytest.raises(ValueError, match="must not overlap"):
        gae.compute_td_and_gae(**overlap)
    bad_order = dict(fixture)
    bad_order["episode_ids"] = np.array([0, 1, 0, 2, 2])
    with pytest.raises(ValueError, match="nondecreasing"):
        gae.compute_td_and_gae(**bad_order)
    premature = dict(fixture)
    premature["terminals"] = fixture["terminals"].copy()
    premature["terminals"][0] = True
    with pytest.raises(ValueError, match="not the final stored row"):
        gae.compute_td_and_gae(**premature)


def test_squared_exp_controls_leave_positive_untouched() -> None:
    advantage = torch.tensor([2.0, -2.0, -1.0])
    distance = torch.tensor([100.0, 0.0, 2.0])
    weighted, factor = gae.controlled_advantage(advantage, distance, "sqexp_c64")
    assert weighted[0].item() == 2.0
    assert factor[0].item() == 1.0
    assert factor[1].item() == 1.0
    assert factor[2].item() == pytest.approx(math.exp(-64.0), rel=1e-5)
    positive_only, po_factor = gae.controlled_advantage(
        advantage, distance, "positive_only"
    )
    assert positive_only.tolist() == [2.0, 0.0, 0.0]
    assert po_factor.tolist() == [1.0, 0.0, 0.0]


def test_old_policy_refresh_is_exact_k4() -> None:
    cadence = gae.OldPolicyCadence(4)
    assert [step for step in range(1, 14) if cadence.should_refresh_before(step)] == [1, 5, 9, 13]
    actor = torch.nn.Linear(2, 2)
    old = torch.nn.Linear(2, 2)
    cadence.refresh(old, actor, 1)
    assert cadence.refresh_count == 1
    assert cadence.first_refresh_step == 1
    assert cadence.last_refresh_step == 1
    assert all(not parameter.requires_grad for parameter in old.parameters())
    with pytest.raises(ValueError, match="not a scheduled"):
        cadence.refresh(old, actor, 2)


def test_a2c_and_ppo_objectives_are_finite_and_ppo_clips() -> None:
    torch.manual_seed(4)
    actor = gae.CanonicalActor(3, 2)
    old_actor = gae.CanonicalActor(3, 2)
    old_actor.load_state_dict(actor.state_dict())
    observations = torch.randn(8, 3)
    actions = torch.tanh(torch.randn(8, 2))
    advantages = torch.tensor([1.0, -1.0, 0.5, -0.5, 1.5, -2.0, 0.25, -0.25])
    a2c, a2c_diag = gae.actor_objective(
        actor=actor,
        old_actor=None,
        observations=observations,
        actions=actions,
        advantages=advantages,
        actor_mode="a2c",
        control_id="sqexp_c64",
    )
    ppo, ppo_diag = gae.actor_objective(
        actor=actor,
        old_actor=old_actor,
        observations=observations,
        actions=actions,
        advantages=advantages,
        actor_mode="ppo_clip_k4",
        control_id="sqexp_c64",
    )
    assert torch.isfinite(a2c)
    assert torch.isfinite(ppo)
    assert a2c_diag["ratio_mean"] == 1.0
    assert ppo_diag["ratio_mean"] == pytest.approx(1.0)
    # Move current policy far enough from old to exercise clipping.
    with torch.no_grad():
        actor.mean.bias.add_(3.0)
    _, shifted = gae.actor_objective(
        actor=actor,
        old_actor=old_actor,
        observations=observations,
        actions=actions,
        advantages=advantages,
        actor_mode="ppo_clip_k4",
        control_id="positive_only",
    )
    assert shifted["ratio_outside_fraction"] > 0.0
    assert shifted["objective_clip_fraction"] >= 0.0


