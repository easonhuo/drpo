from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import integrate_dev_branch as cli  # noqa: E402
from validate_dev_integration import IntegrationError, validate_request  # noqa: E402


def git(cwd: Path, *args: str, check: bool = True) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and proc.returncode != 0:
        raise AssertionError(f"git {' '.join(args)} failed: {proc.stderr or proc.stdout}")
    return proc.stdout.strip()


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def commit_all(repo: Path, message: str) -> str:
    git(repo, "add", "-A")
    git(repo, "commit", "-m", message)
    return git(repo, "rev-parse", "HEAD")


def blob_sha(repo: Path, ref: str, path: str) -> str:
    return git(repo, "rev-parse", f"{ref}:{path}")


def make_remote_fixture(
    tmp_path: Path, *, changed_path: str = "src/example.py", symlink: bool = False
):
    work = tmp_path / "work"
    remote = tmp_path / "remote.git"
    work.mkdir()
    git(work, "init", "-b", "main")
    git(work, "config", "user.name", "Test User")
    git(work, "config", "user.email", "test@example.com")
    write(work / "README.md", "base\n")
    main_sha = commit_all(work, "base")
    git(tmp_path, "init", "--bare", str(remote))
    git(work, "remote", "add", "origin", str(remote))
    git(work, "push", "-u", "origin", "main")

    git(work, "checkout", "-b", "dev/example")
    target = work / changed_path
    target.parent.mkdir(parents=True, exist_ok=True)
    if symlink:
        os.symlink("../README.md", target)
    else:
        target.write_text("print('dev')\n", encoding="utf-8")
    dev_sha = commit_all(work, "dev change")
    git(work, "push", "-u", "origin", "dev/example")
    return work, remote, main_sha, dev_sha


def decision_payload(
    integration_id: str,
    *,
    evidence: str = "not_applicable",
    status: str = "not_applicable",
):
    return {
        "schema_version": 1,
        "integration_id": integration_id,
        "decision": {
            "approved": True,
            "code_integration_eligible": True,
            "evidence_level": evidence,
            "result_status": status,
            "claim_support_level": "none",
            "terminal_audit": "not_required" if status == "not_applicable" else "complete",
            "task_performance_collapse": "not_assessed",
            "support_boundary": "not_assessed",
            "numerical_failure": "not_assessed",
        },
        "reviewer": {"id": "reviewer-test", "decision_token": "approved-test-token"},
        "limitations": [],
        "unresolved": [],
    }


def request_payload(
    integration_id: str,
    *,
    main_sha: str,
    dev_sha: str,
    changed_path: str,
    expected_blob: str,
    result_commit_sha: str | None = None,
    result_git_dirty: bool = False,
):
    return {
        "schema_version": 1,
        "integration_id": integration_id,
        "source": {
            "repository": "easonhuo/drpo",
            "remote": "origin",
            "main_ref": "refs/heads/main",
            "expected_main_sha": main_sha,
            "dev_branch": "dev/example",
            "expected_dev_sha": dev_sha,
            "result_commit_sha": result_commit_sha,
            "result_git_dirty": result_git_dirty,
        },
        "subject": {"experiment_ids": [], "governance_claims": ["GOV-TEST-01"]},
        "files": {
            "operations": [
                {
                    "op": "add",
                    "source_path": changed_path,
                    "destination_path": changed_path,
                    "expected_blob_sha": expected_blob,
                    "expected_mode": "100644",
                }
            ]
        },
        "review": {"decision_file": "review.yaml"},
        "checks": {"requested_tier": "auto"},
    }


def save_inputs(repo: Path, request: dict, decision: dict) -> Path:
    request_path = repo / "request.yaml"
    write(request_path, yaml.safe_dump(request, sort_keys=False))
    write(repo / "review.yaml", yaml.safe_dump(decision, sort_keys=False))
    return request_path


def run_plan(repo: Path, request_path: Path, transactions: Path):
    args = cli.build_parser().parse_args(
        [
            "plan",
            "--repo-root",
            str(repo),
            "--request",
            str(request_path),
            "--transaction-root",
            str(transactions),
            "--json",
        ]
    )
    return cli.plan(args)


def only_attempt(transactions: Path, integration_id: str) -> Path:
    attempts = sorted((transactions / integration_id).glob("attempt-*"))
    assert len(attempts) == 1
    return attempts[0]


def test_plan_locks_sources_and_audits_exact_add(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    work, _, main_sha, dev_sha = make_remote_fixture(tmp_path)
    integration_id = "GOV-TEST-INTEGRATION-01"
    request = request_payload(
        integration_id,
        main_sha=main_sha,
        dev_sha=dev_sha,
        changed_path="src/example.py",
        expected_blob=blob_sha(work, dev_sha, "src/example.py"),
    )
    request_path = save_inputs(work, request, decision_payload(integration_id))
    transactions = tmp_path / "transactions"

    assert run_plan(work, request_path, transactions) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "PASS"
    assert output["state"] == "REVIEWED"
    assert output["changed_file_count"] == 1

    attempt = only_attempt(transactions, integration_id)
    source_lock = json.loads((attempt / "SOURCE_LOCK.json").read_text())
    scope = json.loads((attempt / "SCOPE_AUDIT.json").read_text())
    transaction = json.loads((attempt / "TRANSACTION.json").read_text())
    assert source_lock["main_sha"] == main_sha
    assert source_lock["dev_sha"] == dev_sha
    assert scope["status"] == "PASS"
    assert scope["operation_audits"][0]["dev_entry"]["mode"] == "100644"
    assert transaction["state"] == "REVIEWED"
    assert not (attempt / "audit.git").exists()


def test_remote_dev_drift_fails_closed_and_preserves_diagnostic(tmp_path: Path):
    work, _, main_sha, dev_sha = make_remote_fixture(tmp_path)
    integration_id = "GOV-TEST-DRIFT-01"
    request = request_payload(
        integration_id,
        main_sha=main_sha,
        dev_sha=dev_sha,
        changed_path="src/example.py",
        expected_blob=blob_sha(work, dev_sha, "src/example.py"),
    )
    request_path = save_inputs(work, request, decision_payload(integration_id))

    write(work / "src/example.py", "print('moved')\n")
    commit_all(work, "move dev")
    git(work, "push", "origin", "dev/example")

    transactions = tmp_path / "transactions"
    assert run_plan(work, request_path, transactions) == 2
    attempt = only_attempt(transactions, integration_id)
    diagnostic = json.loads((attempt / "DIAGNOSTIC.json").read_text())
    assert diagnostic["error_code"] == "SOURCE_DRIFT"
    assert diagnostic["state"] == "BLOCKED"


def test_system_forbidden_handoff_path_rejected_even_when_allowlisted(tmp_path: Path):
    work, _, main_sha, dev_sha = make_remote_fixture(tmp_path, changed_path="docs/handoff.md")
    integration_id = "GOV-TEST-FORBIDDEN-01"
    request = request_payload(
        integration_id,
        main_sha=main_sha,
        dev_sha=dev_sha,
        changed_path="docs/handoff.md",
        expected_blob=blob_sha(work, dev_sha, "docs/handoff.md"),
    )
    request_path = save_inputs(work, request, decision_payload(integration_id))
    transactions = tmp_path / "transactions"

    assert run_plan(work, request_path, transactions) == 2
    diagnostic = json.loads(
        (only_attempt(transactions, integration_id) / "DIAGNOSTIC.json").read_text()
    )
    assert diagnostic["error_code"] == "SCOPE_VIOLATION"
    assert "system-forbidden" in diagnostic["message"]


def test_symlink_mode_is_rejected_before_import(tmp_path: Path):
    work, _, main_sha, dev_sha = make_remote_fixture(
        tmp_path, changed_path="src/link.py", symlink=True
    )
    integration_id = "GOV-TEST-SYMLINK-01"
    request = request_payload(
        integration_id,
        main_sha=main_sha,
        dev_sha=dev_sha,
        changed_path="src/link.py",
        expected_blob=blob_sha(work, dev_sha, "src/link.py"),
    )
    request_path = save_inputs(work, request, decision_payload(integration_id))
    transactions = tmp_path / "transactions"

    assert run_plan(work, request_path, transactions) == 2
    diagnostic = json.loads(
        (only_attempt(transactions, integration_id) / "DIAGNOSTIC.json").read_text()
    )
    assert diagnostic["error_code"] == "UNSAFE_PATH"
    assert "unsafe mode 120000" in diagnostic["message"]


def test_dirty_formal_evidence_is_rejected_before_remote_fetch(tmp_path: Path):
    work, _, main_sha, dev_sha = make_remote_fixture(tmp_path)
    integration_id = "GOV-TEST-DIRTY-FORMAL-01"
    request = request_payload(
        integration_id,
        main_sha=main_sha,
        dev_sha=dev_sha,
        changed_path="src/example.py",
        expected_blob=blob_sha(work, dev_sha, "src/example.py"),
        result_commit_sha=dev_sha,
        result_git_dirty=True,
    )
    decision = decision_payload(integration_id, evidence="formal", status="long_run_validated")
    request_path = save_inputs(work, request, decision)
    transactions = tmp_path / "transactions"

    assert run_plan(work, request_path, transactions) == 2
    diagnostic = json.loads(
        (only_attempt(transactions, integration_id) / "DIAGNOSTIC.json").read_text()
    )
    assert diagnostic["error_code"] == "PROVENANCE_INCOMPLETE"
    assert "dirty-worktree" in diagnostic["message"]


def test_result_commit_relation_is_recorded(tmp_path: Path):
    work, _, main_sha, dev_sha = make_remote_fixture(tmp_path)
    integration_id = "GOV-TEST-RESULT-RELATION-01"
    request = request_payload(
        integration_id,
        main_sha=main_sha,
        dev_sha=dev_sha,
        changed_path="src/example.py",
        expected_blob=blob_sha(work, dev_sha, "src/example.py"),
        result_commit_sha=main_sha,
    )
    request_path = save_inputs(work, request, decision_payload(integration_id))
    transactions = tmp_path / "transactions"

    assert run_plan(work, request_path, transactions) == 0
    scope = json.loads(
        (only_attempt(transactions, integration_id) / "SCOPE_AUDIT.json").read_text()
    )
    assert scope["result_to_dev_relation"] == "ancestor_or_equal"


def test_each_plan_creates_new_attempt_without_rewriting_old_source_lock(tmp_path: Path):
    work, _, main_sha, dev_sha = make_remote_fixture(tmp_path)
    integration_id = "GOV-TEST-ATTEMPT-01"
    request = request_payload(
        integration_id,
        main_sha=main_sha,
        dev_sha=dev_sha,
        changed_path="src/example.py",
        expected_blob=blob_sha(work, dev_sha, "src/example.py"),
    )
    request_path = save_inputs(work, request, decision_payload(integration_id))
    transactions = tmp_path / "transactions"

    assert run_plan(work, request_path, transactions) == 0
    first = transactions / integration_id / "attempt-0001" / "SOURCE_LOCK.json"
    first_hash = hashlib.sha256(first.read_bytes()).hexdigest()
    assert run_plan(work, request_path, transactions) == 0
    second = transactions / integration_id / "attempt-0002" / "SOURCE_LOCK.json"
    assert second.exists()
    assert hashlib.sha256(first.read_bytes()).hexdigest() == first_hash


def test_status_json_reads_transaction(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    transaction_dir = tmp_path / "transaction"
    transaction_dir.mkdir()
    write(
        transaction_dir / "TRANSACTION.json",
        json.dumps({"status": "PASS", "state": "REVIEWED", "integration_id": "GOV-X"}),
    )
    args = cli.build_parser().parse_args(
        ["status", "--transaction-dir", str(transaction_dir), "--json"]
    )
    assert cli.status(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["state"] == "REVIEWED"


def test_request_rejects_casefold_target_collision():
    payload = {
        "schema_version": 1,
        "integration_id": "GOV-CASEFOLD-01",
        "source": {
            "repository": "easonhuo/drpo",
            "remote": "origin",
            "main_ref": "refs/heads/main",
            "expected_main_sha": "1" * 40,
            "dev_branch": "dev/x",
            "expected_dev_sha": "2" * 40,
            "result_commit_sha": None,
            "result_git_dirty": False,
        },
        "subject": {"experiment_ids": [], "governance_claims": ["GOV-X"]},
        "files": {
            "operations": [
                {
                    "op": "add",
                    "source_path": "src/A.py",
                    "destination_path": "src/A.py",
                    "expected_blob_sha": "3" * 40,
                    "expected_mode": "100644",
                },
                {
                    "op": "add",
                    "source_path": "src/a.py",
                    "destination_path": "src/a.py",
                    "expected_blob_sha": "4" * 40,
                    "expected_mode": "100644",
                },
            ]
        },
        "review": {"decision_file": "review.yaml"},
        "checks": {"requested_tier": "auto"},
    }
    with pytest.raises(IntegrationError) as exc:
        validate_request(payload)
    assert exc.value.error_code == "SCOPE_VIOLATION"


def test_request_rejects_unknown_top_level_key():
    payload = {
        "schema_version": 1,
        "integration_id": "GOV-UNKNOWN-KEY-01",
        "source": {
            "repository": "easonhuo/drpo",
            "remote": "origin",
            "main_ref": "refs/heads/main",
            "expected_main_sha": "1" * 40,
            "dev_branch": "dev/x",
            "expected_dev_sha": "2" * 40,
            "result_commit_sha": None,
            "result_git_dirty": False,
        },
        "subject": {"experiment_ids": [], "governance_claims": ["GOV-X"]},
        "files": {
            "operations": [
                {
                    "op": "add",
                    "source_path": "src/x.py",
                    "destination_path": "src/x.py",
                    "expected_blob_sha": "3" * 40,
                    "expected_mode": "100644",
                }
            ]
        },
        "review": {"decision_file": "review.yaml"},
        "checks": {"requested_tier": "auto"},
        "typo_field": True,
    }
    with pytest.raises(IntegrationError) as exc:
        validate_request(payload)
    assert exc.value.error_code == "REQUEST_INVALID"
    assert "unknown keys" in exc.value.message


def test_formal_result_commit_must_reach_reviewed_dev(tmp_path: Path):
    work, _, _, dev_sha = make_remote_fixture(tmp_path)
    git(work, "checkout", "main")
    write(work / "MAIN_ONLY.md", "main moved\n")
    main_sha = commit_all(work, "advance main")
    git(work, "push", "origin", "main")

    integration_id = "GOV-TEST-FORMAL-RELATION-01"
    request = request_payload(
        integration_id,
        main_sha=main_sha,
        dev_sha=dev_sha,
        changed_path="src/example.py",
        expected_blob=blob_sha(work, dev_sha, "src/example.py"),
        result_commit_sha=main_sha,
    )
    decision = decision_payload(
        integration_id, evidence="formal", status="finite_step_validated"
    )
    request_path = save_inputs(work, request, decision)
    transactions = tmp_path / "transactions"

    assert run_plan(work, request_path, transactions) == 2
    diagnostic = json.loads(
        (only_attempt(transactions, integration_id) / "DIAGNOSTIC.json").read_text()
    )
    assert diagnostic["error_code"] == "PROVENANCE_INCOMPLETE"
    assert "ancestor" in diagnostic["message"]


def test_validator_cli_reports_pass(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    work, _, main_sha, dev_sha = make_remote_fixture(tmp_path)
    integration_id = "GOV-TEST-VALIDATOR-CLI-01"
    request = request_payload(
        integration_id,
        main_sha=main_sha,
        dev_sha=dev_sha,
        changed_path="src/example.py",
        expected_blob=blob_sha(work, dev_sha, "src/example.py"),
    )
    request_path = save_inputs(work, request, decision_payload(integration_id))

    import validate_dev_integration as validator

    assert validator.main(
        ["--repo-root", str(work), "--request", str(request_path), "--json"]
    ) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "PASS"
    assert output["integration_id"] == integration_id
