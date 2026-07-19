#!/usr/bin/env python3
"""Run the DRPO A/B Replay Engine candidate composition path."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dev_integration_write_path import WritePathError, git, load_json, load_yaml, sha256, write_json  # noqa: E402
from drpo.workflow_replay.evidence import EvidenceError, build_opposite_order_schedule, canonical_sha256, compare_normalized_runs, load_r1_case_contract, load_run_artifact, release_bound_efficiency  # noqa: E402
from drpo.workflow_replay.orchestrate import CandidateOutcome, OrchestrationError, ProcessResult, _copy_exact_tree, _existing_dir, _payload, run_candidate  # noqa: E402

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    candidate = commands.add_parser("candidate")
    candidate.add_argument("--repo-root", default=".")
    candidate.add_argument("--spec", required=True)
    candidate.add_argument("--preparation-root", required=True)
    candidate.add_argument("--transaction-root", required=True)
    candidate.add_argument("--python", default=sys.executable)
    candidate.add_argument("--json", action="store_true")
    pair = commands.add_parser("real-pair")
    for name in ("contract", "case-packet", "source-repo", "output-root"):
        pair.add_argument(f"--{name}", required=True)
    pair.add_argument("--backend-id", default="local-git-v1")
    pair.add_argument("--json", action="store_true")
    return parser



def _git(args: Sequence[str], cwd: Path | None = None, *, binary: bool = False, timeout: int = 180):
    return git(args, cwd=cwd, phase="replay_adapter", code="REPLAY_ERROR", binary=binary, timeout=timeout)

def _locator(path: Path, root: Path, kind: str) -> dict[str, object]:
    return {"kind": kind, "relative_path": path.relative_to(root).as_posix(), "sha256": sha256(path), "byte_size": path.stat().st_size}


def _clone(source: Path, root: Path, source_spec: dict[str, object], contract) -> Path:
    mirror, target = root / "source.git", root / "workspace"
    main = contract.base.historical_task["base_sha"]
    dev = contract.base.historical_task["frozen_implementation_sha"] or source_spec["expected_dev_sha"]
    for commit in {main, dev, contract.base.benchmark["toolchain_sha"]}:
        _git(["cat-file", "-e", f"{commit}^{{commit}}"], source)
    _git(["clone", "--bare", "--no-hardlinks", str(source), str(mirror)], timeout=300)
    _git(["update-ref", source_spec["main_ref"], main], mirror)
    _git(["update-ref", f"refs/heads/{source_spec['dev_branch']}", dev], mirror)
    _git(["clone", "--no-hardlinks", "--shared", str(mirror), str(target)], timeout=300)
    _git(["checkout", "--detach", contract.base.benchmark["toolchain_sha"]], target)
    return target.resolve()


def _commit_workspace(repo: Path, commit: str) -> str:
    return canonical_sha256({"head": commit, "index_tree": str(_git(["rev-parse", f"{commit}^{{tree}}"], repo)).strip(), "status_hex": "", "untracked": []})


def _workspace(repo: Path) -> str:
    status = _git(["status", "--porcelain=v1", "-z", "--untracked-files=all"], repo, binary=True)
    untracked = _git(["ls-files", "--others", "--exclude-standard", "-z"], repo, binary=True)
    entries = []
    for raw in filter(None, untracked.split(b"\0")):
        relative = raw.decode("utf-8")
        target = repo / relative
        if target.is_symlink() or not target.is_file():
            raise ValueError(f"unsafe untracked entry: {relative}")
        entries.append((relative, f"{target.stat().st_mode & 0o777777:06o}", sha256(target)))
    payload = {"head": str(_git(["rev-parse", "HEAD"], repo)).strip(), "index_tree": str(_git(["write-tree"], repo)).strip(),
               "status_hex": status.hex(), "untracked": entries}
    return canonical_sha256(payload)


def _modes(repo: Path, commit: str, paths: Sequence[str]) -> dict[str, str]:
    raw = str(_git(["ls-tree", "-z", commit, "--", *paths], repo))
    records = [record.split("\t", 1) for record in raw.split("\0") if record]
    result = {path: metadata.split()[0] for metadata, path in records if metadata.split()[1] == "blob"}
    if len(result) != len(records) or set(result) != set(paths):
        raise ValueError("final tree modes do not cover unique blob paths")
    return result


class Journal:
    def __init__(self, path: Path, identity, contract_sha: str, before: str, workspace: Path):
        self.path, self.identity, self.workspace = path, identity, workspace
        self.started, self.child_ns, self.commands, self.placements = time.monotonic_ns(), 0, [], []
        self.sequence, self.last_result = 0, None
        self.handle = path.open("x", encoding="utf-8")
        self.record("run_started", origin_ns=self.started, arm=identity.arm, case_id=identity.case_id,
                    pair_id=identity.pair_id, order_position=identity.order_position,
                    case_contract_sha256=contract_sha, workspace_before_sha256=before)

    def record(self, event: str, **payload: object) -> int:
        stamp = time.monotonic_ns()
        row = {"run_id": self.identity.run_id, "sequence": self.sequence, "event": event, "monotonic_ns": stamp, "payload": payload}
        self.handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        self.handle.flush()
        self.sequence += 1
        return stamp

    def invoke(self, command) -> ProcessResult:
        self.record("command_started", name=command.name, argv=command.argv)
        started = time.monotonic_ns()
        try:
            process = subprocess.run(command.argv, cwd=self.workspace, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True, check=False)
            result = ProcessResult(process.returncode, process.stdout, process.stderr)
        except OSError as exc:
            result = ProcessResult(127, "", str(exc))
        elapsed = time.monotonic_ns() - started
        self.child_ns += elapsed
        self.commands.append(command)
        self.last_result = result
        self.record("command_finished", name=command.name, exit_status=result.returncode,
                    child_elapsed_ns=elapsed)
        return result

    def place(self, values: Sequence[str]) -> None:
        for value in values:
            self.placements.append(value)
            self.record("placement", path=value)

    def finish(self, terminal: str, after: str, operator_actions: int) -> dict[str, int]:
        event = {"READY": "run_finished", "BLOCKED": "run_blocked", "STALE": "run_stale", "INTERRUPTED": "run_interrupted", "INVALIDATED": "run_invalidated"}[terminal]
        ended = self.record(event, terminal_state=terminal, workspace_after_sha256=after,
                            child_command_count=len(self.commands), placement_path_count=len(self.placements),
                            operator_action_count=operator_actions)
        self.handle.close()
        total = ended - self.started
        return {"total_ns": total, "child_ns": self.child_ns, "self_overhead_ns": total - self.child_ns}


def _explicit(repo: Path, spec: Path, preparations: Path, transactions: Path,
              python: str, journal: Journal) -> CandidateOutcome:
    def call(name: str, argv: tuple[str, ...], state: str) -> dict[str, object]:
        command = type("Command", (), {"name": name, "argv": argv})()
        return _payload(command, journal.invoke(command), state)

    prepared = call("prepare-inputs", (python, "scripts/prepare_dev_pilot_registration.py", "--repo-root",
                    str(repo), "--spec", str(spec), "--output-root", str(preparations), "--json"),
                    "PREPARED_INPUTS")
    preparation_id = prepared.get("preparation_id")
    if not isinstance(preparation_id, str):
        raise OrchestrationError("prepare-inputs", "prepared identity is invalid")
    preparation_dir = _existing_dir(prepared.get("preparation_dir"), preparations, "preparation_dir")
    journal.place(_copy_exact_tree(preparation_dir / "repository_overlay", repo, "repository"))
    request = repo / "docs" / "integrations" / preparation_id / "INTEGRATION_REQUEST.yaml"
    reviewed = call("v1-plan", (python, "scripts/integrate_dev_branch.py", "plan", "--repo-root", str(repo),
                    "--request", str(request), "--transaction-root", str(transactions), "--json"), "REVIEWED")
    transaction_dir = _existing_dir(reviewed.get("attempt_dir"), transactions, "transaction_dir")
    call("v1-prepare", (python, "scripts/dev_integration_write_path.py", "--transaction-dir",
         str(transaction_dir), "--json"), "PREPARED")
    if (preparation_dir / "transaction_inputs").exists():
        journal.place(_copy_exact_tree(preparation_dir / "transaction_inputs", transaction_dir, "transaction"))
    final = {}
    for name, state in (("normalize", "NORMALIZED"), ("gate", "REQUIRED_GATES_PASSED"), ("finalize", "READY")):
        final = call(f"v1-{name}", (python, "scripts/dev_integration_finalize.py", name,
                     "--transaction-dir", str(transaction_dir), "--json"), state)
    ready = final.get("ready_commit_sha")
    if not isinstance(ready, str) or len(ready) != 40:
        raise OrchestrationError("v1-finalize", "READY output lacks a full commit SHA")
    return CandidateOutcome(preparation_id, str(preparation_dir), str(transaction_dir), ready, tuple(journal.commands), tuple(journal.placements))


def _failure(journal: Journal) -> dict[str, object]:
    if journal.last_result is None:
        return {}
    for text in (journal.last_result.stdout, journal.last_result.stderr):
        try:
            payload = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(payload, dict):
            diagnostic = payload.get("diagnostic")
            return load_json(Path(diagnostic), "replay diagnostic") if isinstance(diagnostic, str) and Path(diagnostic).is_file() else payload
    return {}


def _write_run(args, contract, identity, output: Path, source_spec) -> tuple[Path, Path]:
    run_dir = output / identity.run_id
    run_dir.mkdir()
    subject = run_dir / "case-packet.yaml"
    shutil.copyfile(args.case_packet, subject)
    workspace = _clone(Path(args.source_repo).resolve(), run_dir, source_spec, contract)
    before = _commit_workspace(run_dir / "source.git", contract.base.historical_task["base_sha"])
    journal = Journal(run_dir / "events.jsonl", identity, contract.sha256, before, workspace)
    result_path = outcome_path = transaction = None
    terminal, outcome = "INTERRUPTED", None
    provenance = {"benchmark_toolchain_sha": contract.base.benchmark["toolchain_sha"],
                  "cache_policy": contract.base.benchmark["cache_policy"],
                  "environment_id": contract.base.benchmark["environment_id"],
                  "historical_base_sha": contract.base.historical_task["base_sha"],
                  "input_spec_sha256": contract.base.benchmark["input_spec_sha256"]}
    if contract.base.historical_task["frozen_implementation_sha"] is not None:
        provenance["frozen_implementation_sha"] = contract.base.historical_task["frozen_implementation_sha"]
    preparations, transactions = run_dir / "preparations", run_dir / "transactions"
    try:
        if identity.arm == "A":
            completed = _explicit(workspace, subject, preparations, transactions, sys.executable, journal)
            operator_actions = len(journal.commands) + len({item.split(":", 1)[0] for item in journal.placements})
        else:
            completed = run_candidate(repo_root=workspace, spec_path=subject, preparation_root=preparations,
                                      transaction_root=transactions, python_executable=sys.executable,
                                      invoke=journal.invoke)
            journal.place(completed.placements)
            operator_actions = 1
        transaction = Path(completed.transaction_dir)
        ready = load_json(transaction / "READY_COMMIT.json", "ready commit")
        gate_report = load_json(transaction / "GATE_REPORT.json", "gate report")
        if ready.get("ready_commit_sha") != completed.ready_commit_sha:
            raise ValueError("READY commit identity mismatch")
        paths = tuple(sorted(ready.get("changed_paths", [])))
        modes = _modes(transaction / "integration-repo", completed.ready_commit_sha, paths)
        result_path = run_dir / "result.json"
        result = {"case_id": contract.base.case_id, "base_sha": contract.base.historical_task["base_sha"],
                  "tree_sha": ready.get("tree_sha"), "changed_paths": paths, "file_modes": modes}
        write_json(result_path, result)
        hashes = {key: ready.get("tree_sha") if key.endswith("tree_sha") else sha256(result_path)
                  for key in contract.base.benchmark["expected_final_tree_or_semantic_hashes"]}
        actual = {item.get("label"): "PASS" if item.get("passed") else "FAIL"
                  for item in gate_report.get("outcomes", []) if isinstance(item, dict)}
        gates = {name: actual.get(name, "NOT_RUN") for name in contract.base.benchmark["required_gates"]}
        terminal = "READY"
        protected_after = _workspace(transaction / "integration-repo")
        outcome = {"case_id": contract.base.case_id, "terminal_state": terminal, "safety_boundary": None,
                   "changed_paths": paths, "file_modes": modes, "output_hashes": hashes,
                   "authority_result": "PASS" if ready.get("authority_verify", {}).get("status") == "PASS" else "FAIL",
                   "gate_results": gates, "provenance": provenance,
                   "diagnostic_codes": [], "recovery_class": None}
    except OrchestrationError as error:
        payload = _failure(journal)
        attempt = payload.get("attempt_dir")
        transaction = Path(attempt) if isinstance(attempt, str) else None
        if transaction is None:
            for command in reversed(journal.commands):
                if "--transaction-dir" in command.argv:
                    transaction = Path(command.argv[command.argv.index("--transaction-dir") + 1])
                    break
        if transaction is not None and (transaction / "DIAGNOSTIC.json").is_file():
            payload = load_json(transaction / "DIAGNOSTIC.json", "transaction diagnostic")
        terminal = payload.get("state") if payload.get("state") in {"BLOCKED", "STALE"} else "BLOCKED"
        code = str(payload.get("error_code") or "ORCHESTRATION_ERROR")
        authority, gates = "NOT_RUN", {name: "NOT_RUN" for name in contract.base.benchmark["required_gates"]}
        if transaction is not None and (transaction / "NORMALIZATION_REPORT.json").is_file():
            normal = load_json(transaction / "NORMALIZATION_REPORT.json", "normalization report")
            authority = "PASS" if normal.get("authority_verify", {}).get("status") == "PASS" else "FAIL"
        outcome = {"case_id": contract.base.case_id, "terminal_state": terminal,
                   "safety_boundary": str(payload.get("phase") or error.step), "changed_paths": [],
                   "file_modes": {}, "output_hashes": {}, "authority_result": authority,
                   "gate_results": gates, "provenance": provenance,
                   "diagnostic_codes": [code], "recovery_class":
                       "refresh_main_and_regenerate_packet" if code in {"SOURCE_DRIFT", "STALE_MAIN"}
                       else str(payload.get("recovery_class") or "inspect_and_regenerate")}
        operator_actions = 1 if identity.arm == "B" else len(journal.commands) + len({item.split(":", 1)[0] for item in journal.placements})
        protected = None if transaction is None else transaction / "integration-repo"
        protected_after = _workspace(protected) if protected is not None and protected.is_dir() else before
    except BaseException:
        operator_actions = 1 if identity.arm == "B" else len(journal.commands)
        protected_after = before
    after = protected_after
    timing = journal.finish(terminal, after, operator_actions)
    if outcome is not None:
        outcome_path = run_dir / "outcome.json"
        write_json(outcome_path, outcome)
    artifact = {"schema_version": 1, "case_contract_sha256": contract.sha256,
                "run_identity": asdict(identity), "identities": {
                    "base_sha": contract.base.historical_task["base_sha"],
                    "input_sha256": contract.base.benchmark["input_spec_sha256"],
                    "toolchain_sha": contract.base.benchmark["toolchain_sha"],
                    "evaluator_sha256": contract.r1["evaluator_sha256"],
                    "evidence_schema_sha256": contract.r1["evidence_schema_sha256"],
                    "environment_id": contract.base.benchmark["environment_id"],
                    "cache_policy": contract.base.benchmark["cache_policy"],
                    "backend_id": identity.backend_id,
                    "plan_sha256": canonical_sha256({"case_id": identity.case_id, "arm": identity.arm,
                                                       "commands": [(item.name, item.argv) for item in journal.commands]})},
                "evidence": {"event_log": _locator(journal.path, output, "event_log"),
                    "outcome": None if outcome_path is None else _locator(outcome_path, output, "outcome"),
                    "result": None if result_path is None else _locator(result_path, output, "result"),
                    "subject": _locator(subject, output, "subject")},
                "workspace_before_sha256": before, "workspace_after_sha256": after,
                "execution_terminal": terminal, "timing": timing, "producer_id": "candidate01-c1-real-pair-v1"}
    artifact_path = run_dir / "run-artifact.json"
    write_json(artifact_path, artifact)
    return artifact_path, journal.path


def run_real_pair(args) -> tuple[int, dict[str, object]]:
    contract = load_r1_case_contract(args.contract)
    packet = Path(args.case_packet).expanduser().absolute()
    source = Path(args.source_repo).expanduser().absolute()
    output = Path(args.output_root).expanduser().absolute()
    if packet.is_symlink() or any(parent.is_symlink() for parent in packet.parents) or not packet.is_file() or sha256(packet) != contract.base.benchmark["input_spec_sha256"]:
        raise ValueError("case packet does not match the frozen input SHA-256")
    source_spec = load_yaml(packet, "case packet")["source"]
    if source.is_symlink() or any(parent.is_symlink() for parent in source.parents) or not source.is_dir():
        raise ValueError("source repository is unavailable or unsafe")
    if output.exists() or output.is_symlink() or any(parent.is_symlink() for parent in output.parents):
        raise ValueError("output root must be new")
    output.mkdir(parents=True)
    if str(_git(["rev-parse", "HEAD"], ROOT)).strip() != contract.base.benchmark["toolchain_sha"]:
        raise ValueError("running toolchain HEAD does not match the contract")
    schedule = build_opposite_order_schedule(contract, args.backend_id)
    artifacts, journals = {}, {}
    for identity in schedule:
        artifacts[identity.run_id], journals[identity.run_id] = _write_run(
            args, contract, identity, output, source_spec
        )
    pairs = []
    for pair_id in ("pair-0", "pair-1"):
        by_arm = {item.arm: item for item in schedule if item.pair_id == pair_id}
        arm_a = load_run_artifact(artifacts[by_arm["A"].run_id], output, contract)
        arm_b = load_run_artifact(artifacts[by_arm["B"].run_id], output, contract)
        report = compare_normalized_runs(contract, arm_a, arm_b)
        record = asdict(report)
        if report.equivalent and contract.r1["comparison_mode"] == "exact_artifact":
            release_bound_efficiency(report, {"run_ids": report.run_ids,
                                     "evidence_sha256": report.evidence_sha256, "timing": report.timing})
            record["operation_metrics"] = {arm: {key: json.loads(journals[item.run_id].read_text(encoding="utf-8").splitlines()[-1])["payload"][key]
                                                          for key in ("child_command_count", "placement_path_count", "operator_action_count")}
                                           for arm, item in by_arm.items()}
        pairs.append(record)
    passed = all(item["equivalent"] for item in pairs)
    payload = {"status": "PASS" if passed else "FAIL", "state": "C1_LIVENESS_READY" if passed else "MISMATCH", "case_id": contract.base.case_id, "run_count": 4, "pair_count": 2, "pairs": pairs}
    write_json(output / "PAIR_REPORT.json", payload)
    return (0 if passed else 3), payload


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "real-pair":
        try:
            code, payload = run_real_pair(args)
        except (EvidenceError, OrchestrationError, WritePathError, KeyError, OSError, TypeError, ValueError) as error:
            code, payload = 2, {"status": "FAIL", "state": "INVALIDATED", "message": str(error)}
        print(
            json.dumps(payload, sort_keys=True)
            if args.json
            else (
                f"PASS {payload.get('case_id', '')}"
                if code == 0
                else f"FAIL {payload.get('message', payload.get('state'))}"
            ),
            file=sys.stdout if args.json or code == 0 else sys.stderr,
        )
        return code
    repository = Path(args.repo_root).expanduser().resolve()

    def invoke(command) -> ProcessResult:
        try:
            process = subprocess.run(
                command.argv,
                cwd=repository,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
        except OSError as exc:
            return ProcessResult(127, "", str(exc))
        return ProcessResult(process.returncode, process.stdout, process.stderr)

    try:
        outcome = run_candidate(
            repo_root=repository,
            spec_path=args.spec,
            preparation_root=args.preparation_root,
            transaction_root=args.transaction_root,
            python_executable=args.python,
            invoke=invoke,
        )
        payload = asdict(outcome)
        payload["status"] = "PASS"
        payload["state"] = "READY"
        payload["command_count"] = len(outcome.commands)
        payload["placement_count"] = len(outcome.placements)
        print(
            json.dumps(payload, sort_keys=True)
            if args.json
            else f"PASS {outcome.preparation_id}: {outcome.ready_commit_sha}"
        )
        return 0
    except OrchestrationError as error:
        payload = {
            "status": "FAIL",
            "state": "BLOCKED",
            "step": error.step,
            "message": error.message,
        }
        print(
            json.dumps(payload, sort_keys=True) if args.json else f"FAIL {error}",
            file=sys.stdout if args.json else sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
