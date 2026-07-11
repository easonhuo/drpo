#!/usr/bin/env python3
"""Read-only planner for reviewed DRPO dev-branch integration transactions."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from validate_dev_integration import (
    MODES,
    SCHEMA,
    IntegrationError,
    enforce_provenance,
    file_hash,
    is_forbidden_path,
    json_hash,
    load_yaml,
    repo_path,
    validate_request,
    validate_reviewer_decision,
    write_json,
)

VERSION = "0.1.0-batch1"


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def run(
    command: Sequence[str],
    phase: str,
    code: str,
    *,
    binary: bool = False,
    timeout: int = 120,
) -> str | bytes:
    try:
        proc = subprocess.run(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=not binary,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise IntegrationError(code, phase, f"command failed to start or timed out: {exc}") from exc
    if proc.returncode:
        error = proc.stderr or proc.stdout
        detail = error.decode(errors="replace") if binary else error
        raise IntegrationError(
            code,
            phase,
            f"command failed: {' '.join(command)}: {str(detail).strip()[-2000:]}",
        )
    return proc.stdout


def next_attempt(root: Path, integration_id: str) -> Path:
    parent = root / integration_id
    parent.mkdir(parents=True, exist_ok=True)
    numbers = []
    for child in parent.iterdir():
        match = re.fullmatch(r"attempt-(\d{4})", child.name)
        if match and child.is_dir():
            numbers.append(int(match.group(1)))
    attempt = parent / f"attempt-{max(numbers, default=0) + 1:04d}"
    attempt.mkdir(mode=0o700)
    return attempt


def invalid_attempt(root: Path) -> Path:
    path = root / "_invalid" / datetime.now(timezone.utc).strftime("attempt-%Y%m%dT%H%M%S%fZ")
    path.mkdir(parents=True, mode=0o700)
    return path


def resolve_ref(remote: str, ref: str) -> str:
    output = str(run(
        ["git", "ls-remote", "--refs", remote, ref],
        "source_lock",
        "SOURCE_UNRESOLVED",
    ))
    matches = [
        line.split("\t", 1)[0]
        for line in output.splitlines()
        if line.endswith(f"\t{ref}")
    ]
    if len(matches) != 1:
        raise IntegrationError(
            "SOURCE_UNRESOLVED", "source_lock", f"could not resolve exactly one {ref}"
        )
    return matches[0]


def create_audit_repo(attempt: Path, remote: str, main_ref: str, dev_ref: str) -> Path:
    bare = attempt / "audit.git"
    run(["git", "init", "--bare", str(bare)], "source_lock", "SOURCE_UNRESOLVED")
    run(
        [
            "git", "--git-dir", str(bare), "fetch", "--no-tags", remote,
            f"{main_ref}:refs/audit/main", f"{dev_ref}:refs/audit/dev",
        ],
        "source_lock",
        "SOURCE_UNRESOLVED",
        timeout=300,
    )
    return bare


def tree_entry(bare: Path, commit: str, path: str) -> dict[str, str] | None:
    raw = bytes(run(
        ["git", "--git-dir", str(bare), "ls-tree", "-z", commit, "--", path],
        "scope_audit",
        "BLOB_OR_MODE_MISMATCH",
        binary=True,
    ))
    if not raw:
        return None
    entries = [item for item in raw.split(b"\0") if item]
    if len(entries) != 1:
        raise IntegrationError(
            "BLOB_OR_MODE_MISMATCH", "scope_audit", f"ambiguous tree entry: {path}"
        )
    metadata, encoded = entries[0].split(b"\t", 1)
    mode, object_type, sha = metadata.decode().split()
    actual = encoded.decode()
    if actual != path:
        raise IntegrationError(
            "BLOB_OR_MODE_MISMATCH", "scope_audit", f"tree path mismatch: {path}"
        )
    return {"mode": mode, "type": object_type, "sha": sha, "path": actual}


def diff_changes(bare: Path, main_sha: str, dev_sha: str) -> list[dict[str, Any]]:
    raw = bytes(run(
        [
            "git", "--git-dir", str(bare), "diff", "--name-status", "-z",
            "--find-renames=50%", f"{main_sha}...{dev_sha}",
        ],
        "scope_audit",
        "SCOPE_VIOLATION",
        binary=True,
    ))
    fields = [item for item in raw.split(b"\0") if item]
    result: list[dict[str, Any]] = []
    index = 0
    while index < len(fields):
        status = fields[index].decode()
        index += 1
        code = status[0]
        if code in {"R", "C"}:
            source = fields[index].decode()
            destination = fields[index + 1].decode()
            index += 2
            kind = "rename" if code == "R" else "copy"
        else:
            source = fields[index].decode()
            index += 1
            destination = None if code == "D" else source
            kind = {"A": "add", "M": "modify", "D": "delete"}.get(code, "unsupported")
        result.append({
            "status": status,
            "kind": kind,
            "source_path": repo_path(source, "diff path"),
            "destination_path": None if destination is None else repo_path(destination, "diff path"),
        })
    unsupported = [item for item in result if item["kind"] in {"copy", "unsupported"}]
    if unsupported:
        raise IntegrationError(
            "SCOPE_VIOLATION", "scope_audit", f"unsupported diff status: {unsupported}"
        )
    seen: dict[str, str] = {}
    for item in result:
        target = item["destination_path"] or item["source_path"]
        previous = seen.get(target.casefold())
        if previous is not None and previous != target:
            raise IntegrationError(
                "SCOPE_VIOLATION", "scope_audit", f"case-fold collision: {previous}, {target}"
            )
        seen[target.casefold()] = target
    return result


def operation_key(item: dict[str, Any]) -> tuple[str, str, str | None]:
    return (
        item["kind"] if "kind" in item else item["op"],
        item["source_path"],
        item["destination_path"],
    )


def audit_operations(
    bare: Path,
    main_sha: str,
    dev_sha: str,
    actual: list[dict[str, Any]],
    approved: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    actual_keys = {operation_key(item) for item in actual}
    approved_keys = {operation_key(item) for item in approved}
    if actual_keys != approved_keys:
        raise IntegrationError(
            "SCOPE_VIOLATION",
            "scope_audit",
            "diff/allowlist mismatch; "
            f"unapproved={sorted(actual_keys - approved_keys)}, "
            f"absent={sorted(approved_keys - actual_keys)}",
        )
    reports = []
    for operation in sorted(approved, key=operation_key):
        source = operation["source_path"]
        destination = operation["destination_path"]
        paths = [source] + ([] if destination is None else [destination])
        blocked = sorted(path for path in paths if is_forbidden_path(path))
        if blocked:
            raise IntegrationError(
                "SCOPE_VIOLATION", "scope_audit", f"system-forbidden paths: {blocked}"
            )
        old = tree_entry(bare, main_sha, source)
        new = None if destination is None else tree_entry(bare, dev_sha, destination)
        op = operation["op"]
        if op == "add" and old is not None:
            raise IntegrationError(
                "BLOB_OR_MODE_MISMATCH", "scope_audit", f"add already exists: {source}"
            )
        if op in {"modify", "delete", "rename"} and old is None:
            raise IntegrationError(
                "BLOB_OR_MODE_MISMATCH", "scope_audit", f"source missing: {source}"
            )
        if op != "delete" and new is None:
            raise IntegrationError(
                "BLOB_OR_MODE_MISMATCH", "scope_audit", f"destination missing: {destination}"
            )
        for entry in (old, new):
            if entry is not None and (entry["type"] != "blob" or entry["mode"] not in MODES):
                raise IntegrationError(
                    "UNSAFE_PATH",
                    "scope_audit",
                    f"unsafe mode {entry['mode']} or type {entry['type']} for {entry['path']}",
                )
        if new is not None and (
            new["sha"] != operation["expected_blob_sha"]
            or new["mode"] != operation["expected_mode"]
        ):
            raise IntegrationError(
                "BLOB_OR_MODE_MISMATCH", "scope_audit", f"new blob or mode mismatch: {destination}"
            )
        expected_old = operation["expected_old_blob_sha"]
        if expected_old is not None and (old is None or old["sha"] != expected_old):
            raise IntegrationError(
                "BLOB_OR_MODE_MISMATCH", "scope_audit", f"old blob mismatch: {source}"
            )
        reports.append({
            "operation": operation,
            "main_entry": old,
            "dev_entry": new,
            "status": "PASS",
        })
    return reports


def commit_relation(bare: Path, ancestor: str | None, descendant: str) -> str:
    if ancestor is None:
        return "not_provided"
    run(
        ["git", "--git-dir", str(bare), "cat-file", "-e", f"{ancestor}^{{commit}}"],
        "provenance",
        "PROVENANCE_INCOMPLETE",
    )
    proc = subprocess.run(
        ["git", "--git-dir", str(bare), "merge-base", "--is-ancestor", ancestor, descendant],
        check=False,
    )
    if proc.returncode == 0:
        return "ancestor_or_equal"
    if proc.returncode == 1:
        return "not_ancestor"
    raise IntegrationError(
        "PROVENANCE_INCOMPLETE", "provenance", "could not determine result/dev relation"
    )


def write_failure(
    attempt: Path,
    error: IntegrationError,
    request_path: Path,
    integration_id: str | None,
    main_sha: str | None,
    dev_sha: str | None,
) -> None:
    created = now()
    write_json(attempt / "DIAGNOSTIC.json", {
        "schema_version": SCHEMA,
        "tool_version": VERSION,
        "status": "FAIL",
        "state": "BLOCKED",
        "error_code": error.error_code,
        "phase": error.phase,
        "message": error.message,
        "recovery": list(error.recovery),
        "integration_id": integration_id,
        "request_path": str(request_path),
        "attempt_dir": str(attempt),
        "main_sha": main_sha,
        "dev_sha": dev_sha,
        "created_at": created,
    }, False)
    write_json(attempt / "TRANSACTION.json", {
        "schema_version": SCHEMA,
        "tool_version": VERSION,
        "integration_id": integration_id,
        "state": "BLOCKED",
        "status": "FAIL",
        "error_code": error.error_code,
        "phase": error.phase,
        "created_at": created,
        "updated_at": created,
    })


def plan(args: argparse.Namespace) -> int:
    request_path = Path(args.request).expanduser().resolve()
    transaction_root = Path(args.transaction_root).expanduser().resolve()
    attempt: Path | None = None
    integration_id = main_sha = dev_sha = None
    try:
        repo_root = Path(args.repo_root).expanduser().resolve()
        if not repo_root.is_dir():
            raise IntegrationError("REQUEST_INVALID", "preflight", f"invalid repo root: {repo_root}")
        run(
            ["git", "-C", str(repo_root), "rev-parse", "--git-dir"],
            "preflight",
            "SOURCE_UNRESOLVED",
        )
        request = validate_request(load_yaml(request_path, "REQUEST_INVALID"))
        integration_id = request["integration_id"]
        attempt = next_attempt(transaction_root, integration_id)
        decision_path = (repo_root / request["review"]["decision_file"]).resolve()
        try:
            decision_path.relative_to(repo_root)
        except ValueError as exc:
            raise IntegrationError(
                "UNSAFE_PATH", "scientific_review", "review decision escapes repo root"
            ) from exc
        review = validate_reviewer_decision(
            load_yaml(decision_path, "REVIEW_DECISION_INVALID"), integration_id
        )
        enforce_provenance(request, review)
        source = request["source"]
        run(["git", "check-ref-format", source["main_ref"]], "preflight", "REQUEST_INVALID")
        run(
            ["git", "check-ref-format", "--branch", source["dev_branch"]],
            "preflight",
            "REQUEST_INVALID",
        )
        remote = str(run(
            ["git", "-C", str(repo_root), "remote", "get-url", source["remote"]],
            "source_lock",
            "SOURCE_UNRESOLVED",
        )).strip()
        dev_ref = f"refs/heads/{source['dev_branch']}"
        main_sha = resolve_ref(remote, source["main_ref"])
        dev_sha = resolve_ref(remote, dev_ref)
        if main_sha != source["expected_main_sha"] or dev_sha != source["expected_dev_sha"]:
            raise IntegrationError(
                "SOURCE_DRIFT",
                "source_lock",
                "remote SHA differs from reviewed request",
                ("create a new immutable request and review",),
            )
        locked_at = now()
        write_json(attempt / "SOURCE_LOCK.json", {
            "schema_version": SCHEMA,
            "tool_version": VERSION,
            "integration_id": integration_id,
            "repository": source["repository"],
            "remote_name": source["remote"],
            "remote_location": remote,
            "main_ref": source["main_ref"],
            "main_sha": main_sha,
            "dev_ref": dev_ref,
            "dev_sha": dev_sha,
            "result_commit_sha": source["result_commit_sha"],
            "result_git_dirty": source["result_git_dirty"],
            "request_sha256": file_hash(request_path),
            "request_semantic_sha256": json_hash(request),
            "review_decision_path": request["review"]["decision_file"],
            "review_decision_sha256": file_hash(decision_path),
            "review_decision_semantic_sha256": json_hash(review),
            "locked_at": locked_at,
        }, False)
        bare = create_audit_repo(attempt, remote, source["main_ref"], dev_ref)
        relation = commit_relation(bare, source["result_commit_sha"], dev_sha)
        if review["decision"]["evidence_level"] in {"formal", "closure"} and relation != "ancestor_or_equal":
            raise IntegrationError(
                "PROVENANCE_INCOMPLETE",
                "provenance",
                "formal result commit must be an ancestor of the reviewed dev SHA",
            )
        actual = diff_changes(bare, main_sha, dev_sha)
        reports = audit_operations(
            bare, main_sha, dev_sha, actual, request["files"]["operations"]
        )
        write_json(attempt / "SCOPE_AUDIT.json", {
            "schema_version": SCHEMA,
            "tool_version": VERSION,
            "integration_id": integration_id,
            "status": "PASS",
            "main_sha": main_sha,
            "dev_sha": dev_sha,
            "result_commit_sha": source["result_commit_sha"],
            "result_to_dev_relation": relation,
            "result_git_dirty": source["result_git_dirty"],
            "changed_paths": actual,
            "operation_audits": reports,
            "review_decision": review,
            "audited_at": now(),
        }, False)
        write_json(attempt / "TRANSACTION.json", {
            "schema_version": SCHEMA,
            "tool_version": VERSION,
            "integration_id": integration_id,
            "state": "REVIEWED",
            "status": "PASS",
            "completed_states": ["RECEIVED", "SOURCE_LOCKED", "REVIEWED"],
            "attempt_dir": str(attempt),
            "repo_root": str(repo_root),
            "request_path": str(request_path),
            "main_sha": main_sha,
            "dev_sha": dev_sha,
            "result_commit_sha": source["result_commit_sha"],
            "requested_gate_tier": request["checks"]["requested_tier"],
            "created_at": locked_at,
            "updated_at": now(),
            "next_action": "Batch 2 prepare is not implemented",
        })
        if not args.keep_audit_repo:
            shutil.rmtree(bare)
        output = {
            "status": "PASS",
            "state": "REVIEWED",
            "integration_id": integration_id,
            "attempt_dir": str(attempt),
            "main_sha": main_sha,
            "dev_sha": dev_sha,
            "result_relation": relation,
            "changed_file_count": len(actual),
        }
        print(json.dumps(output, sort_keys=True) if args.json else f"PASS {integration_id}: {attempt}")
        return 0
    except IntegrationError as error:
        attempt = attempt or invalid_attempt(transaction_root)
        write_failure(attempt, error, request_path, integration_id, main_sha, dev_sha)
        output = {
            "status": "FAIL",
            "state": "BLOCKED",
            "error_code": error.error_code,
            "phase": error.phase,
            "message": error.message,
            "attempt_dir": str(attempt),
        }
        print(
            json.dumps(output, sort_keys=True) if args.json else f"FAIL {error}",
            file=sys.stdout if args.json else sys.stderr,
        )
        return 2
    except Exception as exc:
        attempt = attempt or invalid_attempt(transaction_root)
        error = IntegrationError("INTERNAL_ERROR", "internal", f"unexpected error: {exc}")
        write_failure(attempt, error, request_path, integration_id, main_sha, dev_sha)
        output = {
            "status": "FAIL", "state": "BLOCKED", "error_code": error.error_code,
            "message": error.message, "attempt_dir": str(attempt),
        }
        print(
            json.dumps(output, sort_keys=True) if args.json else f"FAIL {error}",
            file=sys.stdout if args.json else sys.stderr,
        )
        return 3


def status(args: argparse.Namespace) -> int:
    path = Path(args.transaction_dir).expanduser().resolve() / "TRANSACTION.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        payload = {
            "status": "FAIL",
            "error_code": "REQUEST_INVALID",
            "message": f"cannot read {path}: {exc}",
        }
        print(json.dumps(payload, sort_keys=True) if args.json else f"FAIL {payload['message']}")
        return 2
    print(
        json.dumps(payload, sort_keys=True)
        if args.json
        else f"{payload.get('status')} {payload.get('state')} {payload.get('integration_id')}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan_parser = commands.add_parser("plan")
    plan_parser.add_argument("--repo-root", default=".")
    plan_parser.add_argument("--request", required=True)
    plan_parser.add_argument("--transaction-root", required=True)
    plan_parser.add_argument("--keep-audit-repo", action="store_true")
    plan_parser.add_argument("--json", action="store_true")
    plan_parser.set_defaults(func=plan)
    status_parser = commands.add_parser("status")
    status_parser.add_argument("--transaction-dir", required=True)
    status_parser.add_argument("--json", action="store_true")
    status_parser.set_defaults(func=status)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
