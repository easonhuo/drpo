from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from drpo.runtime_capacity_wait import (
    plan_capacity_shortage,
    wait_for_runtime_admission,
    wait_for_runtime_plan,
)
from drpo.runtime_resource_autotune import RuntimeResourceError


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.value += seconds


def _blocked_admission(
    work: Path,
    *,
    attempt: int,
    proposed_workers: int,
    selection_digest: str,
) -> None:
    directory = work / "_runtime_resource_attempts" / f"attempt-{attempt}"
    directory.mkdir(parents=True, exist_ok=False)
    revalidation_path = directory / "RUNTIME_REVALIDATION.json"
    revalidation_path.write_text(
        json.dumps(
            {
                "attempt_id": f"attempt-{attempt}",
                "decision": "BLOCK",
                "failures": ["cpu_capacity_changed"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (directory / "RUNTIME_ADMISSION.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "decision": "BLOCK",
                "proposed_workers": proposed_workers,
                "admitted_workers": 0,
                "downshifted": True,
                "reason": "no_safe_worker_capacity",
                "selection_digest": selection_digest,
                "revalidation_path": str(revalidation_path),
                "capacity": {
                    "cpu_worker_limit": 0,
                    "memory_worker_limit": proposed_workers,
                },
                "scientific_matrix_changed": False,
                "running_workers_resized": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _allowed_result(
    *,
    proposed_workers: int,
    admitted_workers: int,
    selection_digest: str,
) -> dict[str, Any]:
    return {
        "mode": "auto",
        "runtime_admission": {
            "schema_version": 1,
            "decision": "ALLOW",
            "proposed_workers": proposed_workers,
            "admitted_workers": admitted_workers,
            "downshifted": admitted_workers < proposed_workers,
            "reason": "capacity_downshift_before_launch",
            "selection_digest": selection_digest,
            "capacity": {
                "cpu_worker_limit": admitted_workers,
                "memory_worker_limit": proposed_workers,
            },
            "scientific_matrix_changed": False,
            "running_workers_resized": False,
        },
    }


def test_waits_through_zero_and_subfloor_capacity_then_admits(
    tmp_path: Path,
) -> None:
    work = tmp_path / "work"
    clock = FakeClock()
    sequence = [0, 2, 4]
    attempts = 0
    kwargs_calls = 0
    events: list[dict[str, Any]] = []

    def kwargs_factory() -> dict[str, Any]:
        nonlocal kwargs_calls
        kwargs_calls += 1
        return {"machine_snapshot": kwargs_calls}

    def admit_once(**kwargs: Any) -> dict[str, Any]:
        nonlocal attempts
        attempts += 1
        assert kwargs["revalidate_kwargs"] == {"machine_snapshot": attempts}
        admitted = sequence[attempts - 1]
        if admitted == 0:
            _blocked_admission(
                work,
                attempt=attempts,
                proposed_workers=8,
                selection_digest="digest",
            )
            raise RuntimeResourceError(
                "RUNTIME_CAPACITY_CHANGED_REPLAN_REQUIRED: cpu_capacity_changed"
            )
        return _allowed_result(
            proposed_workers=8,
            admitted_workers=admitted,
            selection_digest="digest",
        )

    result = wait_for_runtime_admission(
        admit_once=admit_once,
        work_dir=work,
        proposed_workers=8,
        selection_digest="digest",
        revalidate_kwargs_factory=kwargs_factory,
        wait_timeout_seconds=-1,
        poll_seconds=5,
        minimum_admitted_workers=3,
        clock=clock,
        sleep=clock.sleep,
        on_event=lambda value: events.append(dict(value)),
    )

    assert attempts == 3
    assert kwargs_calls == 3
    assert clock.sleeps == [5, 5]
    assert [row["admitted_workers"] for row in events] == [0, 2, 4]
    assert [row["status"] for row in events] == [
        "WAITING_FOR_CAPACITY",
        "WAITING_FOR_CAPACITY",
        "ADMITTED",
    ]
    assert result["runtime_admission"]["admitted_workers"] == 4
    assert result["capacity_wait"]["attempt_count"] == 3
    assert result["capacity_wait"]["minimum_admitted_workers"] == 3

    state = json.loads((work / "RUNTIME_CAPACITY_WAIT.json").read_text())
    assert state["status"] == "ADMITTED"
    assert state["admitted_workers"] == 4
    assert state["scientific_matrix_changed"] is False
    assert state["running_workers_resized"] is False

    rows = [
        json.loads(line)
        for line in (work / "RUNTIME_CAPACITY_WAIT.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 3
    assert rows[-1]["status"] == "ADMITTED"


def test_non_capacity_error_is_immediately_fatal(tmp_path: Path) -> None:
    clock = FakeClock()
    attempts = 0

    def admit_once(**_: Any) -> dict[str, Any]:
        nonlocal attempts
        attempts += 1
        raise RuntimeResourceError("source_commit_mismatch")

    with pytest.raises(RuntimeResourceError, match="source_commit_mismatch"):
        wait_for_runtime_admission(
            admit_once=admit_once,
            work_dir=tmp_path / "work",
            proposed_workers=8,
            selection_digest="digest",
            revalidate_kwargs_factory=dict,
            wait_timeout_seconds=-1,
            poll_seconds=5,
            minimum_admitted_workers=3,
            clock=clock,
            sleep=clock.sleep,
        )

    assert attempts == 1
    assert clock.sleeps == []


def test_finite_timeout_preserves_blocked_evidence(tmp_path: Path) -> None:
    work = tmp_path / "work"
    clock = FakeClock()
    attempts = 0

    def admit_once(**_: Any) -> dict[str, Any]:
        nonlocal attempts
        attempts += 1
        _blocked_admission(
            work,
            attempt=attempts,
            proposed_workers=8,
            selection_digest="digest",
        )
        raise RuntimeResourceError(
            "RUNTIME_CAPACITY_CHANGED_REPLAN_REQUIRED: cpu_capacity_changed"
        )

    with pytest.raises(RuntimeResourceError, match="RUNTIME_CAPACITY_WAIT_TIMEOUT"):
        wait_for_runtime_admission(
            admit_once=admit_once,
            work_dir=work,
            proposed_workers=8,
            selection_digest="digest",
            revalidate_kwargs_factory=dict,
            wait_timeout_seconds=5,
            poll_seconds=2,
            minimum_admitted_workers=3,
            clock=clock,
            sleep=clock.sleep,
        )

    assert clock.value == 5
    assert clock.sleeps == [2, 2, 1]
    state = json.loads((work / "RUNTIME_CAPACITY_WAIT.json").read_text())
    assert state["status"] == "BLOCKED_WAIT_TIMEOUT"
    assert state["last_admission"]["admitted_workers"] == 0
    assert state["attempt_count"] == 4


def test_zero_timeout_preserves_one_shot_behavior(tmp_path: Path) -> None:
    work = tmp_path / "work"
    clock = FakeClock()

    def admit_once(**_: Any) -> dict[str, Any]:
        _blocked_admission(
            work,
            attempt=1,
            proposed_workers=8,
            selection_digest="digest",
        )
        raise RuntimeResourceError(
            "RUNTIME_CAPACITY_CHANGED_REPLAN_REQUIRED: cpu_capacity_changed"
        )

    with pytest.raises(RuntimeResourceError, match="RUNTIME_CAPACITY_CHANGED"):
        wait_for_runtime_admission(
            admit_once=admit_once,
            work_dir=work,
            proposed_workers=8,
            selection_digest="digest",
            revalidate_kwargs_factory=dict,
            wait_timeout_seconds=0,
            poll_seconds=2,
            minimum_admitted_workers=3,
            clock=clock,
            sleep=clock.sleep,
        )

    assert clock.sleeps == []
    state = json.loads((work / "RUNTIME_CAPACITY_WAIT.json").read_text())
    assert state["status"] == "BLOCKED_NO_WAIT"


def test_rejects_invalid_wait_policy_before_measurement(tmp_path: Path) -> None:
    called = False

    def admit_once(**_: Any) -> dict[str, Any]:
        nonlocal called
        called = True
        return {}

    with pytest.raises(RuntimeResourceError, match="cannot exceed proposed"):
        wait_for_runtime_admission(
            admit_once=admit_once,
            work_dir=tmp_path / "work",
            proposed_workers=4,
            selection_digest="digest",
            revalidate_kwargs_factory=dict,
            wait_timeout_seconds=-1,
            poll_seconds=2,
            minimum_admitted_workers=5,
        )
    assert called is False


def test_plan_wait_retries_only_capacity_shortage_then_materializes(
    tmp_path: Path,
) -> None:
    work = tmp_path / "work"
    clock = FakeClock()
    attempts = 0
    events: list[dict[str, Any]] = []

    def plan_once() -> dict[str, Any]:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RuntimeResourceError(
                "measured CPU capacity cannot support one worker"
            )
        return {"mode": "auto", "selection": {"selected_workers": 7}}

    result = wait_for_runtime_plan(
        plan_once=plan_once,
        work_dir=work,
        wait_timeout_seconds=-1,
        poll_seconds=4,
        clock=clock,
        sleep=clock.sleep,
        on_event=lambda value: events.append(dict(value)),
    )

    assert attempts == 3
    assert clock.sleeps == [4, 4]
    assert [row["status"] for row in events] == [
        "WAITING_FOR_CAPACITY",
        "WAITING_FOR_CAPACITY",
        "PLANNED",
    ]
    assert result["selection"]["selected_workers"] == 7
    assert result["plan_capacity_wait"]["attempt_count"] == 3
    state = json.loads((work / "RUNTIME_PLAN_CAPACITY_WAIT.json").read_text())
    assert state["status"] == "PLANNED"


def test_plan_wait_never_retries_non_capacity_error(tmp_path: Path) -> None:
    clock = FakeClock()
    attempts = 0

    def plan_once() -> dict[str, Any]:
        nonlocal attempts
        attempts += 1
        raise RuntimeResourceError("canonical contract hash mismatch")

    with pytest.raises(RuntimeResourceError, match="canonical contract"):
        wait_for_runtime_plan(
            plan_once=plan_once,
            work_dir=tmp_path / "work",
            wait_timeout_seconds=-1,
            poll_seconds=4,
            clock=clock,
            sleep=clock.sleep,
        )

    assert attempts == 1
    assert clock.sleeps == []


def test_plan_capacity_classification_requires_clean_resource_only_probe(
    tmp_path: Path,
) -> None:
    work = tmp_path / "work"
    summary_path = (
        work
        / "_runtime_resource_probe"
        / "w0_throughput"
        / "workers-004"
        / "BENCHMARK_SUMMARY.json"
    )
    summary_path.parent.mkdir(parents=True)
    summary = {
        "concurrency": 4,
        "completed": 4,
        "failed": 0,
        "timed_out": 0,
        "controller_terminated": 0,
        "orphan_process_groups": 0,
        "measured_candidate_cpu_cores": 3.8,
        "aggregate_peak_rss_bytes": 1000,
        "cpu_capacity_ok": False,
        "memory_capacity_ok": True,
        "valid": False,
    }
    summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")
    error = RuntimeResourceError(
        "no resource-valid concurrency candidate completed"
    )
    assert plan_capacity_shortage(error, work_dir=work) is True

    summary["failed"] = 1
    summary["completed"] = 3
    summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")
    assert plan_capacity_shortage(error, work_dir=work) is False


def test_plan_zero_timeout_preserves_one_shot_error(tmp_path: Path) -> None:
    clock = FakeClock()

    def plan_once() -> dict[str, Any]:
        raise RuntimeResourceError(
            "insufficient host memory for one worker after safety headroom"
        )

    with pytest.raises(RuntimeResourceError, match="insufficient host memory"):
        wait_for_runtime_plan(
            plan_once=plan_once,
            work_dir=tmp_path / "work",
            wait_timeout_seconds=0,
            poll_seconds=4,
            clock=clock,
            sleep=clock.sleep,
        )

    assert clock.sleeps == []
    state = json.loads(
        (tmp_path / "work" / "RUNTIME_PLAN_CAPACITY_WAIT.json").read_text()
    )
    assert state["status"] == "BLOCKED_NO_WAIT"
