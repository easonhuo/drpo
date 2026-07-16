from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drpo.workflow_replay.execute import (  # noqa: E402
    CommandSpec,
    ExecutionError,
    build_paired_plans,
    build_plan,
    run_fixture_plan,
)
from drpo.workflow_replay.model import load_case_manifest  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "workflow_replay" / "valid_code_only.yaml"


def manifest():
    return load_case_manifest(FIXTURE)


def commands(prefix: str = "step") -> tuple[CommandSpec, ...]:
    return (
        CommandSpec(f"{prefix}-1", ("python3", "-m", "pytest", "tests/test_example.py", "-q")),
        CommandSpec(
            f"{prefix}-2",
            ("python3", "scripts/handoff_authority.py", "verify", "--repo-root", "."),
        ),
    )


def read_events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_paired_plans_are_deterministic_and_share_frozen_inputs() -> None:
    first_a, first_b = build_paired_plans(manifest(), commands("a"), commands("b"))
    second_a, second_b = build_paired_plans(manifest(), commands("a"), commands("b"))
    assert first_a == second_a
    assert first_b == second_b
    assert first_a.input_sha256 == first_b.input_sha256
    assert first_a.plan_sha256 != first_b.plan_sha256


@pytest.mark.parametrize(
    "items, message",
    [
        ((CommandSpec("same", ("echo", "1")), CommandSpec("same", ("echo", "2"))), "names"),
        ((CommandSpec("one", ("echo", "1")), CommandSpec("two", ("echo", "1"))), "duplicate"),
        ((CommandSpec("bad name", ("echo",)),), "name"),
        ((CommandSpec("ok", ()),), "argv"),
        ((CommandSpec("ok", ["echo"]),), "tuple argv"),
        ((), "at least one"),
    ],
)
def test_invalid_command_plans_fail_closed(items, message: str) -> None:
    with pytest.raises(ExecutionError, match=message):
        build_plan(manifest(), "A", items)


def test_successful_fixture_run_records_append_only_events_and_timing(tmp_path: Path) -> None:
    plan = build_plan(manifest(), "A", commands())
    event_path = tmp_path / "events.jsonl"
    summary = run_fixture_plan(plan, event_path, "run-001", lambda _: 0)
    events = read_events(event_path)
    assert summary["terminal_state"] == "READY"
    assert summary["command_count"] == 2
    assert summary["total_ns"] >= summary["child_ns"]
    assert summary["self_overhead_ns"] == summary["total_ns"] - summary["child_ns"]
    assert [event["sequence"] for event in events] == list(range(len(events)))
    assert [event["event"] for event in events] == [
        "run_started",
        "command_started",
        "command_finished",
        "command_started",
        "command_finished",
        "run_finished",
    ]
    with pytest.raises(FileExistsError):
        run_fixture_plan(plan, event_path, "run-002", lambda _: 0)


def test_nonzero_child_stops_without_claiming_success(tmp_path: Path) -> None:
    plan = build_plan(manifest(), "B", commands())
    summary = run_fixture_plan(plan, tmp_path / "blocked.jsonl", "run-002", lambda _: 7)
    events = read_events(tmp_path / "blocked.jsonl")
    assert summary["terminal_state"] == "BLOCKED"
    assert summary["command_count"] == 1
    assert events[-1]["event"] == "run_blocked"
    assert all(event["event"] != "run_finished" for event in events)


def test_interruption_is_diagnosable_and_never_claims_success(tmp_path: Path) -> None:
    plan = build_plan(manifest(), "A", commands())

    def interrupt(_: CommandSpec) -> int:
        raise KeyboardInterrupt

    path = tmp_path / "interrupted.jsonl"
    with pytest.raises(KeyboardInterrupt):
        run_fixture_plan(plan, path, "run-003", interrupt)
    events = read_events(path)
    assert events[-1]["event"] == "run_interrupted"
    assert events[-1]["payload"]["exception_type"] == "KeyboardInterrupt"
    assert all(event["event"] != "run_finished" for event in events)


def test_timing_separates_child_work_from_engine_overhead(tmp_path: Path) -> None:
    values = iter((0, 10, 20, 30, 130, 140, 150, 160))

    def clock() -> int:
        return next(values)

    plan = build_plan(manifest(), "A", (commands()[0],))
    summary = run_fixture_plan(plan, tmp_path / "timing.jsonl", "run-004", lambda _: 0, clock)
    assert summary == {
        "terminal_state": "READY",
        "command_count": 1,
        "total_ns": 150,
        "child_ns": 100,
        "self_overhead_ns": 50,
    }


def test_event_parent_symlink_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    plan = build_plan(manifest(), "A", (commands()[0],))
    with pytest.raises(ExecutionError, match="symlink"):
        run_fixture_plan(plan, link / "events.jsonl", "run-005", lambda _: 0)


def test_planning_runtime_guardrail() -> None:
    samples = []
    case = manifest()
    for _ in range(1000):
        start = time.perf_counter_ns()
        build_paired_plans(case, commands("a"), commands("b"))
        samples.append((time.perf_counter_ns() - start) / 1_000_000_000)
    ordered = sorted(samples)
    assert statistics.median(ordered) <= 0.250
    assert ordered[int(0.95 * (len(ordered) - 1))] <= 1.000
