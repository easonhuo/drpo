from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import dev_integration_write_path as write_path  # noqa: E402


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


def write(path: Path, text: str, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    os.chmod(path, mode)


def commit_all(repo: Path, message: str) -> str:
    git(repo, "add", "-A")
    git(repo, "commit", "-m", message)
    return git(repo, "rev-parse", "HEAD")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_entry(repo: Path, ref: str, path: str) -> dict[str, str] | None:
    raw = subprocess.run(
        ["git", "ls-tree", "-z", ref, "--", path],
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    ).stdout
    if not raw:
        return None
    metadata, encoded = raw.rstrip(b"\0").split(b"\t", 1)
    mode, object_type, object_sha = metadata.decode().split()
    return {"mode": mode, "type": object_type, "sha": object_sha, "path": encoded.decode()}


def make_fixture(tmp_path: Path) -> dict[str, object]:
    work = tmp_path / "work"
    remote = tmp_path / "remote.git"
    work.mkdir()
    git(work, "init", "-b", "main")
    git(work, "config", "user.name", "Test User")
    git(work, "config", "user.email", "test@example.com")
    write(work / "README.md", "base\n")
    write(work / "modify.txt", "old\n")
    write(work / "delete.txt", "delete me\n")
    write(work / "rename_old.txt", "rename payload\n")
    main_sha = commit_all(work, "base")

    git(tmp_path, "init", "--bare", str(remote))
    git(work, "remote", "add", "origin", str(remote))
    git(work, "push", "-u", "origin", "main")

    git(work, "checkout", "-b", "dev/example")
    write(work / "new.txt", "new\n")
    write(work / "modify.txt", "modified\n")
    (work / "delete.txt").unlink()
    os.rename(work / "rename_old.txt", work / "rename_new.txt")
    write(work / "scripts/run.sh", "#!/bin/sh\necho ok\n", mode=0o755)
    dev_sha = commit_all(work, "dev changes")
    git(work, "push", "-u", "origin", "dev/example")

    operations = [
        {
            "op": "add",
            "source_path": "new.txt",
            "destination_path": "new.txt",
            "expected_blob_sha": tree_entry(work, dev_sha, "new.txt")["sha"],
            "expected_old_blob_sha": None,
            "expected_mode": "100644",
        },
        {
            "op": "modify",
            "source_path": "modify.txt",
            "destination_path": "modify.txt",
            "expected_blob_sha": tree_entry(work, dev_sha, "modify.txt")["sha"],
            "expected_old_blob_sha": tree_entry(work, main_sha, "modify.txt")["sha"],
            "expected_mode": "100644",
        },
        {
            "op": "delete",
            "source_path": "delete.txt",
            "destination_path": None,
            "expected_blob_sha": None,
            "expected_old_blob_sha": tree_entry(work, main_sha, "delete.txt")["sha"],
            "expected_mode": None,
        },
        {
            "op": "rename",
            "source_path": "rename_old.txt",
            "destination_path": "rename_new.txt",
            "expected_blob_sha": tree_entry(work, dev_sha, "rename_new.txt")["sha"],
            "expected_old_blob_sha": tree_entry(work, main_sha, "rename_old.txt")["sha"],
            "expected_mode": "100644",
        },
        {
            "op": "add",
            "source_path": "scripts/run.sh",
            "destination_path": "scripts/run.sh",
            "expected_blob_sha": tree_entry(work, dev_sha, "scripts/run.sh")["sha"],
            "expected_old_blob_sha": None,
            "expected_mode": "100755",
        },
    ]
    return {
        "work": work,
        "remote": remote,
        "main_sha": main_sha,
        "dev_sha": dev_sha,
        "operations": operations,
    }


def make_transaction(
    tmp_path: Path, fixture: dict[str, object], *, transaction_root: Path | None = None
) -> Path:
    work = fixture["work"]
    remote = fixture["remote"]
    main_sha = fixture["main_sha"]
    dev_sha = fixture["dev_sha"]
    operations = fixture["operations"]
    integration_id = "GOV-TEST-BATCH2A-01"
    request = work / "request.yaml"
    review = work / "review.yaml"
    root = transaction_root or (tmp_path / "transactions")
    transaction_dir = root / integration_id / "attempt-0001"
    transaction_dir.mkdir(parents=True)

    review_payload = {
        "schema_version": 1,
        "integration_id": integration_id,
        "decision": {
            "approved": True,
            "code_integration_eligible": True,
            "evidence_level": "not_applicable",
            "result_status": "not_applicable",
            "claim_support_level": "none",
            "terminal_audit": "not_required",
            "task_performance_collapse": "not_assessed",
            "support_boundary": "not_assessed",
            "numerical_failure": "not_assessed",
        },
        "reviewer": {
            "id": "reviewer-test",
            "decision_token": "approved-batch2a-test",
        },
        "limitations": [],
        "unresolved": [],
    }
    request_payload = {
        "schema_version": 1,
        "integration_id": integration_id,
        "source": {
            "repository": "easonhuo/drpo",
            "remote": "origin",
            "main_ref": "refs/heads/main",
            "expected_main_sha": main_sha,
            "dev_branch": "dev/example",
            "expected_dev_sha": dev_sha,
            "result_commit_sha": None,
            "result_git_dirty": False,
        },
        "subject": {
            "experiment_ids": [],
            "governance_claims": ["GOV-TEST-BATCH2A-01"],
        },
        "files": {"operations": operations},
        "review": {"decision_file": "review.yaml"},
        "checks": {"requested_tier": "auto"},
    }
    write(request, yaml.safe_dump(request_payload, sort_keys=False))
    write(review, yaml.safe_dump(review_payload, sort_keys=False))

    audits = []
    changed_paths = []
    for operation in operations:
        op = operation["op"]
        source = operation["source_path"]
        destination = operation["destination_path"]
        audits.append(
            {
                "operation": operation,
                "main_entry": None if op == "add" else tree_entry(work, main_sha, source),
                "dev_entry": None if op == "delete" else tree_entry(work, dev_sha, destination),
                "status": "PASS",
            }
        )
        changed_paths.append(
            {
                "status": {"add": "A", "modify": "M", "delete": "D", "rename": "R100"}[op],
                "kind": op,
                "source_path": source,
                "destination_path": destination,
            }
        )

    source_lock = {
        "schema_version": 1,
        "tool_version": "0.1.0-batch1",
        "integration_id": integration_id,
        "repository": "easonhuo/drpo",
        "remote_name": "origin",
        "remote_location": str(remote),
        "main_ref": "refs/heads/main",
        "main_sha": main_sha,
        "dev_ref": "refs/heads/dev/example",
        "dev_sha": dev_sha,
        "result_commit_sha": None,
        "result_git_dirty": False,
        "request_sha256": sha256(request),
        "request_semantic_sha256": "0" * 64,
        "review_decision_path": "review.yaml",
        "review_decision_sha256": sha256(review),
        "review_decision_semantic_sha256": "1" * 64,
        "locked_at": "2026-07-11T00:00:00Z",
    }
    scope_audit = {
        "schema_version": 1,
        "tool_version": "0.1.0-batch1",
        "integration_id": integration_id,
        "status": "PASS",
        "main_sha": main_sha,
        "dev_sha": dev_sha,
        "result_commit_sha": None,
        "result_to_dev_relation": "not_provided",
        "result_git_dirty": False,
        "changed_paths": changed_paths,
        "operation_audits": audits,
        "review_decision": review_payload,
        "audited_at": "2026-07-11T00:00:00Z",
    }
    transaction = {
        "schema_version": 1,
        "tool_version": "0.1.0-batch1",
        "integration_id": integration_id,
        "state": "REVIEWED",
        "status": "PASS",
        "completed_states": ["RECEIVED", "SOURCE_LOCKED", "REVIEWED"],
        "attempt_dir": str(transaction_dir),
        "repo_root": str(work),
        "request_path": str(request),
        "main_sha": main_sha,
        "dev_sha": dev_sha,
        "result_commit_sha": None,
        "requested_gate_tier": "auto",
        "created_at": "2026-07-11T00:00:00Z",
        "updated_at": "2026-07-11T00:00:00Z",
        "next_action": "prepare",
    }
    for name, payload in {
        "SOURCE_LOCK.json": source_lock,
        "SCOPE_AUDIT.json": scope_audit,
        "TRANSACTION.json": transaction,
    }.items():
        (transaction_dir / name).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return transaction_dir


def test_prepare_builds_exact_source_commit_and_is_idempotent(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    transaction_dir = make_transaction(tmp_path, fixture)

    result = write_path.prepare_transaction(transaction_dir)
    assert result["state"] == "PREPARED"
    assert result["operation_count"] == 5
    repo = Path(result["integration_repo"])
    assert git(repo, "rev-parse", "HEAD^") == fixture["main_sha"]
    assert (repo / "new.txt").read_text() == "new\n"
    assert (repo / "modify.txt").read_text() == "modified\n"
    assert not (repo / "delete.txt").exists()
    assert not (repo / "rename_old.txt").exists()
    assert (repo / "rename_new.txt").read_text() == "rename payload\n"
    assert (repo / "scripts/run.sh").stat().st_mode & stat.S_IXUSR
    assert git(repo, "status", "--porcelain=v1") == ""
    source_commit = git(repo, "rev-parse", "HEAD")
    report = json.loads((transaction_dir / "PREPARE_REPORT.json").read_text())
    assert report["source_commit_sha"] == source_commit
    transaction = json.loads((transaction_dir / "TRANSACTION.json").read_text())
    assert transaction["state"] == "PREPARED"
    assert transaction["source_commit_sha"] == source_commit

    second = write_path.prepare_transaction(transaction_dir)
    assert second["idempotent"] is True
    assert second["source_commit_sha"] == source_commit
    assert git(repo, "rev-parse", "HEAD") == source_commit


def test_prepare_marks_attempt_stale_when_main_moves(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    transaction_dir = make_transaction(tmp_path, fixture)
    work = fixture["work"]
    git(work, "checkout", "main")
    write(work / "main-drift.txt", "drift\n")
    commit_all(work, "main drift")
    git(work, "push", "origin", "main")

    with pytest.raises(write_path.WritePathError) as caught:
        write_path.prepare_transaction(transaction_dir)
    assert caught.value.error_code == "SOURCE_DRIFT"
    transaction = json.loads((transaction_dir / "TRANSACTION.json").read_text())
    assert transaction["state"] == "STALE"
    diagnostic = json.loads((transaction_dir / "DIAGNOSTIC.json").read_text())
    assert diagnostic["error_code"] == "SOURCE_DRIFT"
    assert not (transaction_dir / "integration-repo").exists()


def test_prepare_rejects_request_mutation_after_plan(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    transaction_dir = make_transaction(tmp_path, fixture)
    transaction = json.loads((transaction_dir / "TRANSACTION.json").read_text())
    Path(transaction["request_path"]).write_text("mutated\n", encoding="utf-8")

    with pytest.raises(write_path.WritePathError) as caught:
        write_path.prepare_transaction(transaction_dir)
    assert caught.value.error_code == "IMMUTABILITY_ERROR"
    assert json.loads((transaction_dir / "TRANSACTION.json").read_text())["state"] == "BLOCKED"


def test_prepare_reaudits_scope_against_fetched_git_trees(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    transaction_dir = make_transaction(tmp_path, fixture)
    audit_path = transaction_dir / "SCOPE_AUDIT.json"
    audit = json.loads(audit_path.read_text())
    audit["operation_audits"][0]["dev_entry"]["path"] = "new.txt"
    audit["operation_audits"][0]["dev_entry"]["sha"] = fixture["operations"][1][
        "expected_blob_sha"
    ]
    audit["operation_audits"][0]["operation"]["expected_blob_sha"] = fixture["operations"][1][
        "expected_blob_sha"
    ]
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")

    with pytest.raises(write_path.WritePathError) as caught:
        write_path.prepare_transaction(transaction_dir)
    assert caught.value.error_code in {"BLOB_OR_MODE_MISMATCH", "IMMUTABILITY_ERROR"}
    assert not (transaction_dir / "integration-repo").exists()


def test_symlink_git_mode_is_rejected_before_tree_construction() -> None:
    operation = {
        "op": "add",
        "source_path": "link",
        "destination_path": "link",
        "expected_blob_sha": "0" * 40,
        "expected_old_blob_sha": None,
        "expected_mode": "120000",
    }
    with pytest.raises(write_path.WritePathError) as caught:
        write_path.operation(operation, "operation")
    assert caught.value.error_code == "UNSAFE_PATH"


def test_prepared_worktree_drift_breaks_idempotence(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    transaction_dir = make_transaction(tmp_path, fixture)
    result = write_path.prepare_transaction(transaction_dir)
    repo = Path(result["integration_repo"])
    write(repo / "unexpected.txt", "dirty\n")

    with pytest.raises(write_path.WritePathError) as caught:
        write_path.prepare_transaction(transaction_dir)
    assert caught.value.error_code == "WORKTREE_DIRTY"
    transaction = json.loads((transaction_dir / "TRANSACTION.json").read_text())
    assert transaction["state"] == "BLOCKED"


def test_prepare_recovers_after_report_before_transaction_update(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    transaction_dir = make_transaction(tmp_path, fixture)
    first = write_path.prepare_transaction(transaction_dir)
    transaction_path = transaction_dir / "TRANSACTION.json"
    transaction = json.loads(transaction_path.read_text())
    transaction.update(
        {
            "tool_version": "0.1.0-batch1",
            "state": "REVIEWED",
            "status": "PASS",
            "completed_states": ["RECEIVED", "SOURCE_LOCKED", "REVIEWED"],
            "next_action": "prepare",
        }
    )
    for key in (
        "integration_repo",
        "source_commit_sha",
        "source_commit_parent_sha",
        "source_commit_tree_sha",
    ):
        transaction.pop(key, None)
    transaction_path.write_text(json.dumps(transaction, indent=2) + "\n", encoding="utf-8")

    recovered = write_path.prepare_transaction(transaction_dir)
    assert recovered["idempotent"] is True
    assert recovered["recovered"] is True
    assert recovered["source_commit_sha"] == first["source_commit_sha"]
    final_transaction = json.loads(transaction_path.read_text())
    assert final_transaction["state"] == "PREPARED"
    assert final_transaction["source_commit_sha"] == first["source_commit_sha"]


def test_prepare_rejects_forged_forbidden_path_even_with_rehashed_inputs(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    transaction_dir = make_transaction(tmp_path, fixture)
    transaction = json.loads((transaction_dir / "TRANSACTION.json").read_text())
    request_path = Path(transaction["request_path"])
    request = yaml.safe_load(request_path.read_text())
    request["files"]["operations"][0]["source_path"] = "docs/handoff.md"
    request["files"]["operations"][0]["destination_path"] = "docs/handoff.md"
    request_path.write_text(yaml.safe_dump(request, sort_keys=False), encoding="utf-8")

    audit_path = transaction_dir / "SCOPE_AUDIT.json"
    audit = json.loads(audit_path.read_text())
    audit["operation_audits"][0]["operation"]["source_path"] = "docs/handoff.md"
    audit["operation_audits"][0]["operation"]["destination_path"] = "docs/handoff.md"
    audit["changed_paths"][0]["source_path"] = "docs/handoff.md"
    audit["changed_paths"][0]["destination_path"] = "docs/handoff.md"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")

    lock_path = transaction_dir / "SOURCE_LOCK.json"
    lock = json.loads(lock_path.read_text())
    lock["request_sha256"] = sha256(request_path)
    lock_path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(write_path.WritePathError) as caught:
        write_path.prepare_transaction(transaction_dir)
    assert caught.value.error_code == "SCOPE_VIOLATION"
    assert not (transaction_dir / "integration-repo").exists()


def test_prepare_rejects_git_lfs_pointer_blob(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    work = fixture["work"]
    pointer = (
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:" + "a" * 64 + "\n"
        "size 12345\n"
    )
    write(work / "weights.bin", pointer)
    dev_sha = commit_all(work, "add lfs pointer")
    git(work, "push", "origin", "dev/example")
    fixture["dev_sha"] = dev_sha
    fixture["operations"].append(
        {
            "op": "add",
            "source_path": "weights.bin",
            "destination_path": "weights.bin",
            "expected_blob_sha": tree_entry(work, dev_sha, "weights.bin")["sha"],
            "expected_old_blob_sha": None,
            "expected_mode": "100644",
        }
    )
    transaction_dir = make_transaction(tmp_path, fixture)

    with pytest.raises(write_path.WritePathError) as caught:
        write_path.prepare_transaction(transaction_dir)
    assert caught.value.error_code == "SCOPE_VIOLATION"
    assert "Git LFS pointer" in caught.value.message
    assert not (transaction_dir / "integration-repo").exists()


def test_prepare_rejects_transaction_root_inside_source_repo(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    work = fixture["work"]
    transaction_dir = make_transaction(
        tmp_path, fixture, transaction_root=work / ".integration-transactions"
    )

    with pytest.raises(write_path.WritePathError) as caught:
        write_path.prepare_transaction(transaction_dir)
    assert caught.value.error_code == "UNSAFE_PATH"
    assert not (transaction_dir / "integration-repo").exists()


def test_prepare_report_tampering_breaks_idempotence(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    transaction_dir = make_transaction(tmp_path, fixture)
    write_path.prepare_transaction(transaction_dir)
    report_path = transaction_dir / "PREPARE_REPORT.json"
    report = json.loads(report_path.read_text())
    report["operation_count"] += 1
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(write_path.WritePathError) as caught:
        write_path.prepare_transaction(transaction_dir)
    assert caught.value.error_code == "IMMUTABILITY_ERROR"
    assert json.loads((transaction_dir / "TRANSACTION.json").read_text())["state"] == "BLOCKED"
