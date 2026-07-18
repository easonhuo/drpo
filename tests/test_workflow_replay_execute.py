from __future__ import annotations

import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from drpo.workflow_replay.execute import (  # noqa: E402
    CommandSpec,
    ExecutionError,
    build_paired_plans,
    build_plan,
    run_fixture_plan,
)
from dev_integration_write_path import sha256, write_json  # noqa: E402
from drpo.workflow_replay.evidence import (  # noqa: E402
    RunIdentity,
    load_run_artifact,
    validate_r1_case_contract,
)
from drpo.workflow_replay.model import load_case_manifest  # noqa: E402
from drpo.workflow_replay.orchestrate import (  # noqa: E402
    CandidateOutcome,
    OrchestrationError,
    ProcessResult,
)
import run_workflow_replay as replay  # noqa: E402
from run_workflow_replay import Journal, _clone, _commit_workspace, _workspace, build_parser  # noqa: E402

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
    assert summary["total_ns"] == (
        events[-1]["monotonic_ns"] - events[0]["payload"]["origin_ns"]
    )
    assert events[0]["payload"]["input_sha256"] == plan.input_sha256
    assert events[0]["payload"]["environment_id"] == "linux-py311-fixture-v1"
    assert events[0]["payload"]["cache_policy"] == "cold"
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


def _run_git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, text=True, capture_output=True
    ).stdout.strip()


def test_real_pair_parser_preserves_candidate_and_adds_local_backend() -> None:
    parser = build_parser()
    candidate = parser.parse_args(
        ["candidate", "--spec", "s", "--preparation-root", "p", "--transaction-root", "t"]
    )
    pair = parser.parse_args(
        [
            "real-pair",
            "--contract",
            "c",
            "--case-packet",
            "p",
            "--source-repo",
            "s",
            "--output-root",
            "o",
        ]
    )
    assert candidate.command == "candidate"
    assert pair.backend_id == "local-git-v1"


def test_local_ref_reconstruction_and_workspace_identity(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _run_git(source, "init", "-q")
    _run_git(source, "config", "user.email", "replay@example.invalid")
    _run_git(source, "config", "user.name", "Replay Test")
    (source / "value.txt").write_text("base\n", encoding="utf-8")
    _run_git(source, "add", "value.txt")
    _run_git(source, "commit", "-qm", "base")
    base = _run_git(source, "rev-parse", "HEAD")
    (source / "value.txt").write_text("dev\n", encoding="utf-8")
    _run_git(source, "commit", "-qam", "dev")
    dev = _run_git(source, "rev-parse", "HEAD")
    (source / "tool.txt").write_text("tool\n", encoding="utf-8")
    _run_git(source, "add", "tool.txt")
    _run_git(source, "commit", "-qm", "tool")
    toolchain = _run_git(source, "rev-parse", "HEAD")
    contract = SimpleNamespace(
        base=SimpleNamespace(
            historical_task={"base_sha": base, "frozen_implementation_sha": dev},
            benchmark={"toolchain_sha": toolchain},
        )
    )
    run_root = tmp_path / "run"
    run_root.mkdir()
    workspace = _clone(
        source,
        run_root,
        {"main_ref": "refs/heads/main", "dev_branch": "case-dev", "expected_dev_sha": dev},
        contract,
    )
    assert _run_git(workspace, "rev-parse", "HEAD") == toolchain
    assert _run_git(run_root / "source.git", "rev-parse", "refs/heads/main") == base
    assert _run_git(run_root / "source.git", "rev-parse", "refs/heads/case-dev") == dev
    clean = _workspace(workspace)
    (workspace / "untracked.txt").write_text("new\n", encoding="utf-8")
    assert _workspace(workspace) != clean
    assert _commit_workspace(run_root / "source.git", base) == _commit_workspace(
        run_root / "source.git", base
    )


def test_real_journal_binds_commands_placements_and_operator_actions(tmp_path: Path) -> None:
    identity = SimpleNamespace(
        run_id="a" * 64,
        arm="A",
        case_id="CASE-001",
        pair_id="pair-0",
        order_position=0,
    )
    journal = Journal(tmp_path / "events.jsonl", identity, "b" * 64, "c" * 64, tmp_path)
    result = journal.invoke(CommandSpec("child", (sys.executable, "-c", "pass")))
    journal.place(("repository:file.txt",))
    timing = journal.finish("READY", "d" * 64, 2)
    events = read_events(tmp_path / "events.jsonl")
    assert result.returncode == 0
    assert events[0]["event"] == "run_started"
    assert events[-1]["event"] == "run_finished"
    assert events[-1]["payload"]["child_command_count"] == 1
    assert events[-1]["payload"]["placement_path_count"] == 1
    assert events[-1]["payload"]["operator_action_count"] == 2
    assert timing["total_ns"] >= timing["child_ns"]


def _r1_contract(packet: Path, *, ready: bool, artifact_sha: str | None = None):
    case_id = "C01-ADAPTER-READY" if ready else "C06-ADAPTER-STALE"
    path = "docs/example.txt"
    base, toolchain = "1" * 40, "2" * 40
    gates = ["artifact_digest"] if ready else ["source_lock"]
    return validate_r1_case_contract(
        {
            "schema_version": 2,
            "case_id": case_id,
            "task_class": "code_only" if ready else "stale_recovery",
            "historical_task": {
                "base_sha": base,
                "frozen_implementation_sha": None,
                "source_prs": [1],
                "source_commits": [base],
                "historical_real_time_evidence": [],
            },
            "benchmark": {
                "toolchain_sha": toolchain,
                "input_spec_sha256": sha256(packet),
                "expected_terminal_state": "READY" if ready else "BLOCKED",
                "expected_safety_boundary": None if ready else "source_lock",
                "expected_changed_paths": [path] if ready else [],
                "expected_final_tree_or_semantic_hashes": (
                    {"artifact_sha256": artifact_sha} if ready else {}
                ),
                "required_gates": gates,
                "environment_id": "local-adapter-test-v1",
                "cache_policy": "cold",
                "replayability": "complete",
                "predeclared_exclusions": [],
            },
            "r1": {
                "comparison_mode": "exact_artifact" if ready else "failure_boundary",
                "expected_file_modes": {path: "100644"} if ready else {},
                "expected_authority_result": "PASS" if ready else "NOT_RUN",
                "expected_gate_results": {gates[0]: "PASS" if ready else "NOT_RUN"},
                "expected_diagnostic_codes": [] if ready else ["SOURCE_DRIFT"],
                "expected_recovery_class": (
                    None if ready else "refresh_main_and_regenerate_packet"
                ),
                "workspace_rule": "changed_as_expected" if ready else "unchanged",
                "evaluator_sha256": "3" * 64,
                "evidence_schema_sha256": "4" * 64,
                "order_policy": "two_opposite_pairs",
            },
        }
    )


def test_adapter_ready_artifact_is_accepted_by_unchanged_r1_loader(
    tmp_path: Path, monkeypatch
) -> None:
    packet = tmp_path / "packet.yaml"
    packet.write_text("source: {}\n", encoding="utf-8")
    result = {
        "case_id": "C01-ADAPTER-READY",
        "base_sha": "1" * 40,
        "tree_sha": "5" * 40,
        "changed_paths": ("docs/example.txt",),
        "file_modes": {"docs/example.txt": "100644"},
    }
    expected_result = tmp_path / "expected-result.json"
    write_json(expected_result, result)
    contract = _r1_contract(packet, ready=True, artifact_sha=sha256(expected_result))
    transaction = tmp_path / "transaction"
    (transaction / "integration-repo").mkdir(parents=True)
    write_json(
        transaction / "READY_COMMIT.json",
        {
            "ready_commit_sha": "6" * 40,
            "tree_sha": "5" * 40,
            "changed_paths": ["docs/example.txt"],
            "authority_verify": {"status": "PASS"},
        },
    )
    write_json(
        transaction / "GATE_REPORT.json",
        {"outcomes": [{"label": "artifact_digest", "passed": True}]},
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = tmp_path / "source"
    source.mkdir()
    monkeypatch.setattr(replay, "_clone", lambda *args: workspace)
    monkeypatch.setattr(replay, "_commit_workspace", lambda *args: "7" * 64)
    monkeypatch.setattr(replay, "_workspace", lambda *args: "8" * 64)
    monkeypatch.setattr(
        replay,
        "_modes",
        lambda *args: {"docs/example.txt": "100644"},
    )

    def explicit(repo, spec, preparations, transactions, python, journal):
        journal.invoke(CommandSpec("fake-stage", (sys.executable, "-c", "pass")))
        journal.place(("repository:docs/example.txt",))
        return CandidateOutcome(
            "PREP-001",
            str(tmp_path / "preparation"),
            str(transaction),
            "6" * 40,
            tuple(journal.commands),
            tuple(journal.placements),
        )

    monkeypatch.setattr(replay, "_explicit", explicit)
    output = tmp_path / "evidence"
    output.mkdir()
    identity = RunIdentity.build(contract.base.case_id, "A", "pair-0", 0, 0, "local-git-v1")
    artifact, _ = replay._write_run(
        SimpleNamespace(case_packet=str(packet), source_repo=str(source)),
        contract,
        identity,
        output,
        {},
    )
    normalized = load_run_artifact(artifact, output, contract)
    assert normalized.execution_valid
    assert normalized.execution_terminal == "READY"
    assert normalized.outcome is not None


def test_adapter_stale_boundary_is_accepted_without_target_mutation(
    tmp_path: Path, monkeypatch
) -> None:
    packet = tmp_path / "packet.yaml"
    packet.write_text("source: {}\n", encoding="utf-8")
    contract = _r1_contract(packet, ready=False)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = tmp_path / "source"
    source.mkdir()
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    monkeypatch.setattr(replay, "_clone", lambda *args: workspace)
    monkeypatch.setattr(replay, "_commit_workspace", lambda *args: "9" * 64)

    def stale(repo, spec, preparations, transactions, python, journal):
        journal.last_result = ProcessResult(
            2,
            json.dumps(
                {
                    "status": "FAIL",
                    "state": "BLOCKED",
                    "error_code": "SOURCE_DRIFT",
                    "phase": "source_lock",
                    "attempt_dir": str(attempt),
                }
            ),
        )
        raise OrchestrationError("v1-plan", "stale main")

    monkeypatch.setattr(replay, "_explicit", stale)
    output = tmp_path / "evidence"
    output.mkdir()
    identity = RunIdentity.build(contract.base.case_id, "A", "pair-0", 0, 0, "local-git-v1")
    artifact, _ = replay._write_run(
        SimpleNamespace(case_packet=str(packet), source_repo=str(source)),
        contract,
        identity,
        output,
        {},
    )
    normalized = load_run_artifact(artifact, output, contract)
    assert normalized.execution_valid
    assert normalized.execution_terminal == "BLOCKED"
    assert normalized.outcome is not None
    assert normalized.outcome.diagnostic_codes == ("SOURCE_DRIFT",)
