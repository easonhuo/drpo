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
    normalize_fixture_attempt,
    run_fixture_plan,
)
from dev_integration_write_path import sha256, write_json  # noqa: E402
from drpo.workflow_replay.evidence import (  # noqa: E402
    EvidenceLocator,
    RunIdentity,
    canonical_sha256,
    load_run_artifact,
    validate_r1_case_contract,
)
from drpo.workflow_replay.model import load_case_manifest  # noqa: E402
from drpo.workflow_replay.trajectory import (  # noqa: E402
    summarize_trajectory,
    validate_r3_run_artifact,
)
from drpo.workflow_replay.orchestrate import (  # noqa: E402
    CandidateOutcome,
    OrchestrationError,
    ProcessResult,
)
import run_workflow_replay as replay  # noqa: E402
from run_workflow_replay import (  # noqa: E402
    CONTROL_PLANE_BOOTSTRAP,
    Journal,
    ReplayCheckout,
    _assert_historical_result,
    _clone,
    _commit_workspace,
    _controlled_command,
    _workspace,
    build_parser,
)

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
    checkout = _clone(
        source,
        run_root,
        {"main_ref": "refs/heads/main", "dev_branch": "case-dev", "expected_dev_sha": dev},
        contract,
    )
    workspace = checkout.workspace
    assert _run_git(workspace, "rev-parse", "HEAD") == toolchain
    assert _run_git(checkout.control_plane, "rev-parse", "HEAD") == toolchain
    assert checkout.control_plane != workspace
    assert _run_git(checkout.control_plane, "status", "--porcelain=v1") == ""
    assert _run_git(run_root / "source.git", "rev-parse", "refs/heads/main") == base
    assert _run_git(run_root / "source.git", "rev-parse", "refs/heads/case-dev") == dev
    clean = _workspace(workspace)
    (workspace / "untracked.txt").write_text("new\n", encoding="utf-8")
    assert _workspace(workspace) != clean
    assert _run_git(checkout.control_plane, "status", "--porcelain=v1") == ""
    assert _commit_workspace(run_root / "source.git", base) == _commit_workspace(
        run_root / "source.git", base
    )


def test_controlled_finalizer_command_keeps_treatment_outside_v1(
    tmp_path: Path,
) -> None:
    original = CommandSpec(
        "v1-normalize",
        (
            sys.executable,
            "scripts/dev_integration_finalize.py",
            "normalize",
            "--transaction-dir",
            "/tmp/tx",
            "--json",
        ),
    )
    controlled = _controlled_command(
        original,
        tmp_path / "control-plane",
        "1" * 40,
        "2" * 40,
    )
    assert controlled.name == original.name
    assert controlled.argv[:3] == (sys.executable, "-c", CONTROL_PLANE_BOOTSTRAP)
    assert controlled.argv[3:6] == (
        str(tmp_path / "control-plane"),
        "2" * 40,
        "1" * 40,
    )
    assert controlled.argv[6:] == original.argv[2:]

    ordinary = CommandSpec("v1-plan", (sys.executable, "scripts/integrate_dev_branch.py"))
    assert _controlled_command(ordinary, tmp_path, "1" * 40, "2" * 40) == ordinary


def test_control_plane_drift_blocks_and_is_recorded(
    tmp_path: Path,
) -> None:
    control = tmp_path / "control"
    control.mkdir()
    _run_git(control, "init", "-q")
    _run_git(control, "config", "user.email", "replay@example.invalid")
    _run_git(control, "config", "user.name", "Replay Test")
    (control / "tool.txt").write_text("tool\n", encoding="utf-8")
    _run_git(control, "add", "tool.txt")
    _run_git(control, "commit", "-qm", "tool")
    toolchain = _run_git(control, "rev-parse", "HEAD")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    identity = SimpleNamespace(
        run_id="e" * 64,
        arm="A",
        case_id="CASE-CONTROL",
        pair_id="pair-0",
        order_position=0,
    )
    journal = Journal(
        tmp_path / "control-events.jsonl",
        identity,
        "f" * 64,
        "0" * 64,
        workspace,
        control_plane=control,
        historical_base_sha="1" * 40,
        toolchain_sha=toolchain,
    )
    first = journal.invoke(CommandSpec("ordinary", (sys.executable, "-c", "pass")))
    assert first.returncode == 0
    (control / "unexpected.txt").write_text("dirty\n", encoding="utf-8")
    second = journal.invoke(CommandSpec("ordinary-2", (sys.executable, "-c", "pass")))
    payload = json.loads(second.stdout)
    assert second.returncode == 2
    assert payload["error_code"] == "CONTROL_PLANE_DRIFT"
    assert payload["phase"] == "replay_control_plane"
    journal.finish("BLOCKED", "0" * 64, 2)
    assert read_events(tmp_path / "control-events.jsonl")[-1]["event"] == "run_blocked"


def test_historical_result_requires_exact_parent_and_paths(tmp_path: Path) -> None:
    repo = tmp_path / "result-repo"
    repo.mkdir()
    _run_git(repo, "init", "-q")
    _run_git(repo, "config", "user.email", "replay@example.invalid")
    _run_git(repo, "config", "user.name", "Replay Test")
    (repo / "base.txt").write_text("base\n", encoding="utf-8")
    _run_git(repo, "add", "base.txt")
    _run_git(repo, "commit", "-qm", "base")
    base = _run_git(repo, "rev-parse", "HEAD")
    (repo / "docs").mkdir()
    (repo / "docs" / "example.txt").write_text("ready\n", encoding="utf-8")
    _run_git(repo, "add", "docs/example.txt")
    _run_git(repo, "commit", "-qm", "ready")
    ready = _run_git(repo, "rev-parse", "HEAD")
    _assert_historical_result(
        repo,
        ready,
        base,
        ("docs/example.txt",),
        ("docs/example.txt",),
    )
    with pytest.raises(OrchestrationError, match="paths drifted"):
        _assert_historical_result(
            repo,
            ready,
            base,
            ("docs/example.txt",),
            ("scripts/run_workflow_replay.py",),
        )
    (repo / "second.txt").write_text("second\n", encoding="utf-8")
    _run_git(repo, "add", "second.txt")
    _run_git(repo, "commit", "-qm", "second")
    grandchild = _run_git(repo, "rev-parse", "HEAD")
    with pytest.raises(OrchestrationError, match="not based directly"):
        _assert_historical_result(
            repo,
            grandchild,
            base,
            ("docs/example.txt", "second.txt"),
            ("docs/example.txt", "second.txt"),
        )


def test_real_journal_binds_commands_placements_and_operator_actions(tmp_path: Path) -> None:
    identity = RunIdentity.build("CASE-001", "A", "pair-0", 0, 0, "fixture-v1")
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
    attempt = normalize_fixture_attempt(
        identity, 0, _r3_locator(tmp_path, "events.jsonl", "real-jsonl"), tmp_path,
        "real-binding.json", timing=timing,
        output_artifact_locator=_r3_candidate(tmp_path, identity, 0),
    )
    assert attempt["observed_resources"]["command_count"] == 1
    assert attempt["observed_resources"]["active_ns"] == timing["child_ns"]


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
    control = tmp_path / "control"
    control.mkdir()
    monkeypatch.setattr(
        replay,
        "_clone",
        lambda *args: ReplayCheckout(workspace, control),
    )
    monkeypatch.setattr(replay, "_validate_control_plane", lambda path, sha: path)
    monkeypatch.setattr(replay, "_assert_historical_result", lambda *args: None)
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
    control = tmp_path / "control"
    control.mkdir()
    monkeypatch.setattr(
        replay,
        "_clone",
        lambda *args: ReplayCheckout(workspace, control),
    )
    monkeypatch.setattr(replay, "_validate_control_plane", lambda path, sha: path)
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

R3_CAPABILITIES = {
    "command_count": "OBSERVED", "active_ns": "OBSERVED", "retained_bytes": "OBSERVED",
    "tool_operation_count": "UNAVAILABLE", "token_count": "UNAVAILABLE",
    "message_count": "UNAVAILABLE", "monetary_microunits": "UNAVAILABLE",
}


def _r3_locator(root: Path, path: str, kind: str) -> EvidenceLocator:
    raw = (root / path).read_bytes()
    return EvidenceLocator(kind, path, sha256(root / path), len(raw))


def _r3_json(root: Path, path: str, kind: str, payload: dict) -> EvidenceLocator:
    write_json(root / path, payload)
    return _r3_locator(root, path, kind)


def _r3_candidate(root: Path, identity: RunIdentity, ordinal: int) -> EvidenceLocator:
    attempt_id = canonical_sha256({"run_id": identity.run_id, "ordinal": ordinal})
    return _r3_json(root, f"candidate-{ordinal}.json", f"candidate-{attempt_id}", {})


def _r3_feedback(root: Path, identity: RunIdentity, ordinal: int) -> EvidenceLocator:
    parent = canonical_sha256({"run_id": identity.run_id, "ordinal": ordinal - 1})
    attempt = canonical_sha256({"run_id": identity.run_id, "ordinal": ordinal})
    digest = canonical_sha256({"parent_attempt_id": parent, "repair_attempt_id": attempt})
    return _r3_json(root, f"feedback-{ordinal}.json", f"feedback-{digest}", {})


def _r3_run_payload(
    root: Path, identity: RunIdentity, attempts: list[dict], acceptance: str
) -> dict:
    final_id = attempts[-1]["attempt_id"]
    outcome = accepted = None
    if acceptance != "NOT_AVAILABLE":
        binding = {
            "schema_version": 1, "case_id": identity.case_id, "arm": identity.arm,
            "run_id": identity.run_id, "final_attempt_id": final_id,
        }
        outcome = _r3_json(root, "final-outcome.json", "final-outcome", binding)
        accepted = _r3_json(
            root, "acceptance.json", "acceptance",
            {**binding, "final_acceptance": acceptance},
        )
    aggregate = {
        name: sum(item["observed_resources"][name] for item in attempts)
        for name, state in R3_CAPABILITIES.items() if state == "OBSERVED"
    }
    payload = {
        "schema_version": 1,
        "run_identity": {
            "case_id": identity.case_id, "arm": identity.arm, "pair_id": identity.pair_id,
            "repetition": identity.repetition, "order_position": identity.order_position,
            "backend_id": identity.backend_id, "run_id": identity.run_id,
        },
        "base_sha": "1" * 40, "toolchain_sha": "2" * 40,
        "environment_id": "linux-py311-fixture-v1", "cache_policy": "cold",
        "backend_id": identity.backend_id, "resource_capabilities": R3_CAPABILITIES,
        "attempts": attempts, "first_attempt_id": attempts[0]["attempt_id"],
        "final_attempt_id": final_id,
        "final_outcome_locator": None if outcome is None else vars(outcome),
        "final_acceptance": acceptance,
        "acceptance_evidence_locator": None if accepted is None else vars(accepted),
        "aggregate_observed_resources": aggregate, "run_artifact_sha256": "",
    }
    unsigned = {key: value for key, value in payload.items() if key != "run_artifact_sha256"}
    payload["run_artifact_sha256"] = canonical_sha256(unsigned)
    return payload


def _r3_fixture_attempt(
    root: Path,
    identity: RunIdentity,
    ordinal: int,
    status: int | BaseException,
    *,
    disposition: str | None = None,
    feedback: bool = False,
) -> dict:
    raw = root / f"raw-{ordinal}.jsonl"
    plan = build_plan(manifest(), identity.arm, (commands(f"r3-{ordinal}")[0],))
    runner = (
        (lambda _: (_ for _ in ()).throw(status))
        if isinstance(status, BaseException) else (lambda _: status)
    )
    if isinstance(status, BaseException):
        with pytest.raises(type(status)):
            run_fixture_plan(plan, raw, identity.run_id, runner)
    else:
        run_fixture_plan(plan, raw, identity.run_id, runner)
    needs_output = not isinstance(status, BaseException) and disposition != "ENVIRONMENT"
    feedback_locator = _r3_feedback(root, identity, ordinal) if feedback else None
    return normalize_fixture_attempt(
        identity, ordinal, _r3_locator(root, raw.name, f"fixture-jsonl-{ordinal}"),
        root, f"binding-{ordinal}.json", disposition=disposition,
        output_artifact_locator=_r3_candidate(root, identity, ordinal) if needs_output else None,
        feedback_class="EVALUATOR" if feedback else "NONE",
        feedback_locator=feedback_locator,
    )


@pytest.mark.parametrize(
    ("status", "acceptance", "terminal", "candidate_failures", "interruptions"),
    [(0, "PASS", "SUCCEEDED", 0, 0), (7, "REJECTED", "FAILED", 1, 0),
     (KeyboardInterrupt(), "NOT_AVAILABLE", "INTERRUPTED", 0, 1)],
)
def test_r3_fixture_normalization_single_attempts(
    tmp_path: Path,
    status: int | BaseException,
    acceptance: str,
    terminal: str,
    candidate_failures: int,
    interruptions: int,
) -> None:
    identity = RunIdentity.build(manifest().case_id, "A", "r3-single", 0, 0, "fixture-v1")
    attempt = _r3_fixture_attempt(tmp_path, identity, 0, status)
    summary = summarize_trajectory(
        validate_r3_run_artifact(
            _r3_run_payload(tmp_path, identity, [attempt], acceptance), tmp_path
        )
    )
    assert summary.final_attempt_terminal == terminal
    assert summary.candidate_failure_count == candidate_failures
    assert summary.interruption_count == interruptions


def test_r3_fixture_repair_success_retains_failed_initial_attempt(tmp_path: Path) -> None:
    identity = RunIdentity.build(manifest().case_id, "B", "r3-repair", 0, 1, "fixture-v1")
    attempts = [
        _r3_fixture_attempt(tmp_path, identity, 0, 5),
        _r3_fixture_attempt(tmp_path, identity, 1, 0, feedback=True),
    ]
    summary = summarize_trajectory(
        validate_r3_run_artifact(_r3_run_payload(tmp_path, identity, attempts, "PASS"), tmp_path)
    )
    assert (summary.initial_terminal, summary.repair_count) == ("FAILED", 1)
    assert (summary.candidate_failure_count, summary.final_attempt_terminal) == (1, "SUCCEEDED")


def test_r3_fixture_environment_block_is_an_invalidation(tmp_path: Path) -> None:
    identity = RunIdentity.build(manifest().case_id, "A", "r3-invalid", 0, 0, "fixture-v1")
    attempt = _r3_fixture_attempt(tmp_path, identity, 0, 7, disposition="ENVIRONMENT")
    summary = summarize_trajectory(
        validate_r3_run_artifact(
            _r3_run_payload(tmp_path, identity, [attempt], "NOT_AVAILABLE"), tmp_path
        )
    )
    assert summary.invalidation_count == 1
    assert summary.candidate_failure_count == 0


def test_r3_fixture_and_historical_payload_share_run_schema(tmp_path: Path) -> None:
    identity = RunIdentity.build(manifest().case_id, "A", "r3-history", 0, 0, "fixture-v1")
    attempt = _r3_fixture_attempt(tmp_path, identity, 0, 0)
    fixture = _r3_run_payload(tmp_path, identity, [attempt], "PASS")
    historical = json.loads(json.dumps(fixture))
    assert validate_r3_run_artifact(fixture, tmp_path) == validate_r3_run_artifact(
        historical, tmp_path
    )
