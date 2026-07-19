from __future__ import annotations

from pathlib import Path

import pytest

from drpo_reference.experiments.d4rl import (
    dispatch_d4rl9,
    resolve_d4rl9_execution,
)
from drpo_reference.external.d4rl_tasks import (
    D4RL9_TASKS,
    DATASET_TIERS,
    ENVIRONMENTS,
    resolve_d4rl_task,
    validate_d4rl9_matrix,
    validate_dataset_path,
)
from drpo_reference.external.hopper_protocol import HopperProtocol


def _paths(root: Path) -> dict[str, Path]:
    return {
        task.task_id: root / task.dataset_basename
        for task in D4RL9_TASKS
    }


def test_d4rl9_task_matrix_matches_manuscript_order() -> None:
    assert len(D4RL9_TASKS) == 9
    assert tuple(task.task_id for task in D4RL9_TASKS) == tuple(
        f"{environment}-{tier}-v2"
        for tier in DATASET_TIERS
        for environment in ENVIRONMENTS
    )
    assert validate_d4rl9_matrix(D4RL9_TASKS) == D4RL9_TASKS
    with pytest.raises(ValueError, match="exact manuscript matrix"):
        validate_d4rl9_matrix(D4RL9_TASKS[:-1])


def test_reference_scores_and_hopper_identity_match_frozen_protocol() -> None:
    expected = {
        "halfcheetah": (-280.178953, 12135.0),
        "hopper": (-20.272305, 3234.3),
        "walker2d": (1.629008, 4592.3),
    }
    for task in D4RL9_TASKS:
        assert (
            task.normalized_score_reference_min,
            task.normalized_score_reference_max,
        ) == expected[task.environment]

    task = resolve_d4rl_task("hopper-medium-replay-v2")
    protocol = HopperProtocol()
    assert task.dataset_basename == protocol.dataset_basename
    assert task.dataset_sha256 == protocol.dataset_sha256
    assert task.dataset_id == protocol.rollout_dataset_id
    assert task.env_id == protocol.env_id
    assert (
        task.normalized_score_reference_min
        == protocol.normalized_score_reference_min
    )
    assert (
        task.normalized_score_reference_max
        == protocol.normalized_score_reference_max
    )


def test_unverified_dataset_hashes_fail_closed(tmp_path: Path) -> None:
    task = resolve_d4rl_task("halfcheetah-medium-v2")
    path = tmp_path / task.dataset_basename
    path.write_bytes(b"not-a-registered-dataset")

    nonformal = validate_dataset_path(
        path,
        task,
        require_verified_sha=False,
    )
    assert nonformal["identity_verified"] is False
    assert nonformal["registered_sha256"] is None

    with pytest.raises(RuntimeError, match="formal execution is blocked"):
        validate_dataset_path(
            path,
            task,
            require_verified_sha=True,
        )


def test_execution_plan_exposes_provenance_and_protocol_blockers(
    tmp_path: Path,
) -> None:
    plan = resolve_d4rl9_execution(
        dataset_paths=_paths(tmp_path),
        seeds=tuple(range(10)),
    )
    assert plan.formal_evidence_eligible is False
    assert plan.method_ranking_claim_allowed is False
    assert plan.dataset_identity_complete is False
    assert any(
        reason.startswith("unresolved_dataset_sha256:")
        for reason in plan.blocked_reasons
    )
    assert "d4rl9_performance_protocol_not_frozen" in plan.blocked_reasons
    manifest = plan.as_manifest()
    assert manifest["shared_locomotion_engine"] is True
    assert manifest["separate_d4rl_trainer_implemented"] is False

    with pytest.raises(ValueError, match="duplicates"):
        resolve_d4rl9_execution(
            dataset_paths=_paths(tmp_path),
            seeds=(1, 1),
        )
    incomplete = _paths(tmp_path)
    incomplete.pop("walker2d-medium-expert-v2")
    with pytest.raises(ValueError, match="dataset mapping mismatch"):
        resolve_d4rl9_execution(
            dataset_paths=incomplete,
            seeds=(1,),
        )


def test_dispatch_uses_one_injected_runner_for_every_task(tmp_path: Path) -> None:
    plan = resolve_d4rl9_execution(
        dataset_paths=_paths(tmp_path / "data"),
        seeds=(1,),
        smoke=True,
    )
    calls: list[dict[str, object]] = []

    def shared_runner(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {
            "task_id": kwargs["task"].task_id,
            "formal_result_claim": False,
        }

    result = dispatch_d4rl9(
        plan=plan,
        output_root=tmp_path / "output",
        task_runner=shared_runner,
        allow_non_evidence=True,
    )
    assert len(calls) == 9
    assert [call["task"].task_id for call in calls] == [
        task.task_id for task in D4RL9_TASKS
    ]
    assert all(
        call["method_ranking_claim_allowed"] is False
        for call in calls
    )
    assert result["shared_locomotion_engine"] is True
    assert result["method_ranking_claim_allowed"] is False

    blocked = resolve_d4rl9_execution(
        dataset_paths=_paths(tmp_path / "data2"),
        seeds=(1,),
    )
    with pytest.raises(RuntimeError, match="dispatch is blocked"):
        dispatch_d4rl9(
            plan=blocked,
            output_root=tmp_path / "blocked",
            task_runner=shared_runner,
        )
