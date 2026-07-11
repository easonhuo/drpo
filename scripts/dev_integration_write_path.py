#!/usr/bin/env python3
"""Create a local source commit from a reviewed DRPO integration transaction.

Batch 2A consumes Batch 1 records, rechecks locked refs and blobs, builds an
exact tree in an isolated Git index, and checks out a clean one-parent source
commit. It stops before registry/handoff changes, normalization, gates,
finalization, push, pull request, or merge.
"""

from __future__ import annotations

import argparse
import fcntl
import fnmatch
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

import yaml

SCHEMA = 1
VERSION = "0.2.0-batch2a"
MODES = {"100644", "100755"}
OPS = {"add", "modify", "delete", "rename"}
SHA_RE = re.compile(r"[0-9a-f]{40}")
ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{2,127}")
FORBIDDEN = (
    "docs/handoff.md",
    "experiments/registry.yaml",
    "docs/handoff_deltas/**",
    "docs/handoff_shadow/**/generated/**",
    ".git/**",
)
LFS_PREFIX = b"version https://git-lfs.github.com/spec/v1\n"
MAX_BLOB_BYTES = 10 * 1024 * 1024


@dataclass
class WritePathError(Exception):
    error_code: str
    phase: str
    message: str
    recovery: tuple[str, ...] = ()

    def __str__(self) -> str:
        return f"{self.error_code}: {self.message}"


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def fail(code: str, phase: str, message: str, *recovery: str) -> None:
    raise WritePathError(code, phase, message, tuple(recovery))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def locked(transaction_dir: Path):
    with (transaction_dir / ".prepare.lock").open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({"pid": os.getpid(), "locked_at": now()}) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        fail("REQUEST_INVALID", "prepare_preflight", f"cannot read {label}: {exc}")
    if not isinstance(value, dict):
        fail("REQUEST_INVALID", "prepare_preflight", f"{label} must be a JSON object")
    return value


def load_yaml(path: Path, label: str) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        fail("REQUEST_INVALID", "prepare_preflight", f"cannot read {label}: {exc}")
    if not isinstance(value, dict):
        fail("REQUEST_INVALID", "prepare_preflight", f"{label} must be a YAML mapping")
    return value


def text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        fail("REQUEST_INVALID", "prepare_preflight", f"{label} must be non-empty text")
    return value.strip()


def full_sha(value: Any, label: str) -> str:
    value = text(value, label)
    if not SHA_RE.fullmatch(value):
        fail("REQUEST_INVALID", "prepare_preflight", f"{label} must be a full Git SHA")
    return value


def path_value(value: Any, label: str) -> str:
    raw = text(value, label).replace("\\", "/")
    if any(ord(char) < 32 or ord(char) == 127 for char in raw):
        fail("UNSAFE_PATH", "prepare_preflight", f"{label} contains control bytes")
    path = PurePosixPath(raw)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        fail("UNSAFE_PATH", "prepare_preflight", f"unsafe path: {raw!r}")
    normalized = path.as_posix()
    if normalized == ".git" or normalized.startswith(".git/"):
        fail("UNSAFE_PATH", "prepare_preflight", f"unsafe path: {raw!r}")
    return normalized


def forbidden(path: str) -> bool:
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in FORBIDDEN)


def git(
    args: Sequence[str],
    *,
    phase: str,
    code: str,
    cwd: Path | None = None,
    timeout: int = 180,
    binary: bool = False,
    env: dict[str, str] | None = None,
) -> str | bytes:
    command = ["git", *args]
    process_env = os.environ.copy()
    process_env.setdefault("GIT_TERMINAL_PROMPT", "0")
    process_env.setdefault("GIT_LFS_SKIP_SMUDGE", "1")
    if env:
        process_env.update(env)
    try:
        proc = subprocess.run(
            command,
            cwd=None if cwd is None else str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=not binary,
            check=False,
            timeout=timeout,
            env=process_env,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        fail(code, phase, f"command failed to start or timed out: {' '.join(command)}: {exc}")
    if proc.returncode:
        detail = proc.stderr or proc.stdout
        if binary:
            detail = detail.decode(errors="replace")
        fail(code, phase, f"command failed ({proc.returncode}): {' '.join(command)}: {str(detail)[-3000:]}")
    return proc.stdout


def operation(raw: Any, label: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        fail("REQUEST_INVALID", "prepare_preflight", f"{label} must be a mapping")
    op = text(raw.get("op"), f"{label}.op")
    if op not in OPS:
        fail("SCOPE_VIOLATION", "prepare_preflight", f"unsupported operation: {op}")
    source = path_value(raw.get("source_path"), f"{label}.source_path")
    destination = None if op == "delete" else path_value(
        raw.get("destination_path"), f"{label}.destination_path"
    )
    if op in {"add", "modify"} and source != destination:
        fail("SCOPE_VIOLATION", "prepare_preflight", f"{op} must keep its path")
    if op == "rename" and source == destination:
        fail("SCOPE_VIOLATION", "prepare_preflight", "rename must change path")
    paths = [source] + ([] if destination is None else [destination])
    blocked = sorted(item for item in paths if forbidden(item))
    if blocked:
        fail("SCOPE_VIOLATION", "prepare_preflight", f"system-forbidden paths: {blocked}")
    blob = mode = None
    if op != "delete":
        blob = full_sha(raw.get("expected_blob_sha"), f"{label}.expected_blob_sha")
        mode = text(raw.get("expected_mode"), f"{label}.expected_mode")
        if mode not in MODES:
            fail("UNSAFE_PATH", "prepare_preflight", f"unsupported Git mode: {mode}")
    old_blob = raw.get("expected_old_blob_sha")
    if old_blob is not None:
        old_blob = full_sha(old_blob, f"{label}.expected_old_blob_sha")
    return {
        "op": op,
        "source_path": source,
        "destination_path": destination,
        "expected_blob_sha": blob,
        "expected_old_blob_sha": old_blob,
        "expected_mode": mode,
    }


def op_key(item: dict[str, Any]) -> tuple[str, str, str | None]:
    return item["op"], item["source_path"], item["destination_path"]


def stored_entry(raw: Any, expected_path: str, label: str) -> dict[str, str]:
    if not isinstance(raw, dict):
        fail("BLOB_OR_MODE_MISMATCH", "prepare_preflight", f"missing {label}: {expected_path}")
    entry = {
        "mode": text(raw.get("mode"), f"{label}.mode"),
        "type": text(raw.get("type"), f"{label}.type"),
        "sha": full_sha(raw.get("sha"), f"{label}.sha"),
        "path": path_value(raw.get("path"), f"{label}.path"),
    }
    if entry["path"] != expected_path or entry["type"] != "blob" or entry["mode"] not in MODES:
        fail("UNSAFE_PATH", "prepare_preflight", f"unsafe or mismatched {label}: {expected_path}")
    return entry


def audited_operation(raw: Any, index: int) -> dict[str, Any]:
    if not isinstance(raw, dict) or raw.get("status") != "PASS":
        fail("SCOPE_VIOLATION", "prepare_preflight", f"operation audit {index} is not PASS")
    item = operation(raw.get("operation"), f"operation_audits[{index}].operation")
    op, source, destination = item["op"], item["source_path"], item["destination_path"]
    main_entry = dev_entry = None
    if op == "add":
        if raw.get("main_entry") is not None:
            fail("BLOB_OR_MODE_MISMATCH", "prepare_preflight", f"add exists in main: {source}")
    else:
        main_entry = stored_entry(raw.get("main_entry"), source, "main_entry")
        if item["expected_old_blob_sha"] and main_entry["sha"] != item["expected_old_blob_sha"]:
            fail("BLOB_OR_MODE_MISMATCH", "prepare_preflight", f"old blob mismatch: {source}")
    if op == "delete":
        if raw.get("dev_entry") is not None:
            fail("BLOB_OR_MODE_MISMATCH", "prepare_preflight", f"delete exists in dev: {source}")
    else:
        assert destination is not None
        dev_entry = stored_entry(raw.get("dev_entry"), destination, "dev_entry")
        if dev_entry["sha"] != item["expected_blob_sha"] or dev_entry["mode"] != item["expected_mode"]:
            fail("BLOB_OR_MODE_MISMATCH", "prepare_preflight", f"new blob mismatch: {destination}")
    return {**item, "main_entry": main_entry, "dev_entry": dev_entry}


def normalize_review(raw: dict[str, Any], integration_id: str) -> dict[str, Any]:
    if raw.get("schema_version") != SCHEMA or raw.get("integration_id") != integration_id:
        fail("REVIEW_DECISION_INVALID", "prepare_preflight", "review identity mismatch")
    decision, reviewer = raw.get("decision"), raw.get("reviewer")
    if not isinstance(decision, dict) or not isinstance(reviewer, dict):
        fail("REVIEW_DECISION_INVALID", "prepare_preflight", "review mappings are missing")
    if decision.get("approved") is not True or decision.get("code_integration_eligible") is not True:
        fail("SCIENTIFIC_REVIEW_MISSING", "prepare_preflight", "review does not approve integration")
    decision_keys = (
        "approved",
        "code_integration_eligible",
        "evidence_level",
        "result_status",
        "claim_support_level",
        "terminal_audit",
        "task_performance_collapse",
        "support_boundary",
        "numerical_failure",
    )
    normalized_decision = {key: decision.get(key) for key in decision_keys}
    for key in decision_keys[2:]:
        normalized_decision[key] = text(normalized_decision[key], f"decision.{key}")
    limitations = raw.get("limitations", [])
    unresolved = raw.get("unresolved", [])
    if not isinstance(limitations, list) or not isinstance(unresolved, list):
        fail("REVIEW_DECISION_INVALID", "prepare_preflight", "review lists are invalid")
    return {
        "schema_version": SCHEMA,
        "integration_id": integration_id,
        "decision": normalized_decision,
        "reviewer": {
            "id": text(reviewer.get("id"), "reviewer.id"),
            "decision_token": text(reviewer.get("decision_token"), "reviewer.decision_token"),
        },
        "limitations": [text(item, "limitations item") for item in limitations],
        "unresolved": [text(item, "unresolved item") for item in unresolved],
    }


def load_context(transaction_dir: Path) -> dict[str, Any]:
    transaction_path = transaction_dir / "TRANSACTION.json"
    source_path = transaction_dir / "SOURCE_LOCK.json"
    audit_path = transaction_dir / "SCOPE_AUDIT.json"
    tx = load_json(transaction_path, "transaction")
    source = load_json(source_path, "source lock")
    audit = load_json(audit_path, "scope audit")
    if any(item.get("schema_version") != SCHEMA for item in (tx, source, audit)):
        fail("REQUEST_INVALID", "prepare_preflight", "artifact schema mismatch")
    if Path(text(tx.get("attempt_dir"), "attempt_dir")).expanduser().resolve() != transaction_dir:
        fail("IMMUTABILITY_ERROR", "prepare_preflight", "transaction attempt_dir mismatch")
    if tx.get("state") not in {"REVIEWED", "PREPARED"}:
        fail("REQUEST_INVALID", "prepare_preflight", f"invalid transaction state: {tx.get('state')!r}")
    integration_id = text(tx.get("integration_id"), "integration_id")
    if not ID_RE.fullmatch(integration_id):
        fail("REQUEST_INVALID", "prepare_preflight", "invalid integration_id")
    if source.get("integration_id") != integration_id or audit.get("integration_id") != integration_id:
        fail("IMMUTABILITY_ERROR", "prepare_preflight", "integration ID mismatch")
    main_sha = full_sha(source.get("main_sha"), "main_sha")
    dev_sha = full_sha(source.get("dev_sha"), "dev_sha")
    if any(item.get("main_sha") != main_sha or item.get("dev_sha") != dev_sha for item in (tx, audit)):
        fail("IMMUTABILITY_ERROR", "prepare_preflight", "source SHA mismatch")
    if audit.get("status") != "PASS":
        fail("SCOPE_VIOLATION", "prepare_preflight", "scope audit is not PASS")

    repo_root = Path(text(tx.get("repo_root"), "repo_root")).expanduser().resolve()
    request_path = Path(text(tx.get("request_path"), "request_path")).expanduser().resolve()
    if sha256(request_path) != source.get("request_sha256"):
        fail("IMMUTABILITY_ERROR", "prepare_preflight", "request changed after plan")
    review_rel = path_value(source.get("review_decision_path"), "review decision path")
    review_path = (repo_root / review_rel).resolve()
    try:
        review_path.relative_to(repo_root)
    except ValueError:
        fail("UNSAFE_PATH", "prepare_preflight", "review decision escapes repository")
    if sha256(review_path) != source.get("review_decision_sha256"):
        fail("IMMUTABILITY_ERROR", "prepare_preflight", "review changed after plan")
    try:
        transaction_dir.relative_to(repo_root)
    except ValueError:
        pass
    else:
        fail("UNSAFE_PATH", "prepare_preflight", "transaction directory is inside source repository")

    request = load_yaml(request_path, "locked request")
    review = normalize_review(load_yaml(review_path, "locked review"), integration_id)
    if request.get("schema_version") != SCHEMA or request.get("integration_id") != integration_id:
        fail("IMMUTABILITY_ERROR", "prepare_preflight", "request identity mismatch")
    request_source = request.get("source")
    request_review = request.get("review")
    if not isinstance(request_source, dict) or not isinstance(request_review, dict):
        fail("REQUEST_INVALID", "prepare_preflight", "request source/review is missing")
    main_ref = text(source.get("main_ref"), "main_ref")
    dev_ref = text(source.get("dev_ref"), "dev_ref")
    if (
        request_source.get("repository") != source.get("repository")
        or request_source.get("remote", "origin") != source.get("remote_name")
        or request_source.get("main_ref", "refs/heads/main") != main_ref
        or request_source.get("expected_main_sha") != main_sha
        or request_source.get("expected_dev_sha") != dev_sha
        or f"refs/heads/{request_source.get('dev_branch')}" != dev_ref
        or request_source.get("result_commit_sha") != source.get("result_commit_sha")
        or request_source.get("result_git_dirty", False) != source.get("result_git_dirty")
        or path_value(request_review.get("decision_file"), "decision_file") != review_rel
    ):
        fail("IMMUTABILITY_ERROR", "prepare_preflight", "request differs from source lock")
    if review != audit.get("review_decision"):
        fail("IMMUTABILITY_ERROR", "prepare_preflight", "review differs from scope audit")

    request_files = request.get("files")
    raw_request_ops = request_files.get("operations") if isinstance(request_files, dict) else None
    raw_audits = audit.get("operation_audits")
    if not isinstance(raw_request_ops, list) or not raw_request_ops:
        fail("REQUEST_INVALID", "prepare_preflight", "request operations are missing")
    if not isinstance(raw_audits, list) or not raw_audits:
        fail("SCOPE_VIOLATION", "prepare_preflight", "operation audits are missing")
    request_ops = [operation(raw, f"files.operations[{i}]") for i, raw in enumerate(raw_request_ops)]
    operations = [audited_operation(raw, i) for i, raw in enumerate(raw_audits)]
    request_map = {op_key(item): item for item in request_ops}
    audit_map = {
        op_key(item): {key: item[key] for key in (
            "op", "source_path", "destination_path", "expected_blob_sha",
            "expected_old_blob_sha", "expected_mode",
        )}
        for item in operations
    }
    if len(request_map) != len(request_ops) or len(audit_map) != len(operations):
        fail("SCOPE_VIOLATION", "prepare_preflight", "duplicate operations")
    if request_map != audit_map:
        fail("IMMUTABILITY_ERROR", "prepare_preflight", "audit operations differ from request")
    targets: dict[str, str] = {}
    for item in operations:
        target = item["destination_path"] or item["source_path"]
        previous = targets.get(target.casefold())
        if previous is not None:
            fail("SCOPE_VIOLATION", "prepare_preflight", f"target collision: {previous}, {target}")
        targets[target.casefold()] = target
    changed = audit.get("changed_paths")
    if not isinstance(changed, list):
        fail("SCOPE_VIOLATION", "prepare_preflight", "changed_paths are missing")
    changed_keys = set()
    for i, raw in enumerate(changed):
        if not isinstance(raw, dict):
            fail("SCOPE_VIOLATION", "prepare_preflight", f"changed_paths[{i}] is invalid")
        destination = raw.get("destination_path")
        changed_keys.add((
            text(raw.get("kind"), f"changed_paths[{i}].kind"),
            path_value(raw.get("source_path"), f"changed_paths[{i}].source"),
            None if destination is None else path_value(destination, f"changed_paths[{i}].destination"),
        ))
    if len(changed_keys) != len(changed) or changed_keys != set(audit_map):
        fail("SCOPE_VIOLATION", "prepare_preflight", "changed-path coverage mismatch")

    return {
        "transaction": tx,
        "transaction_path": transaction_path,
        "source_path": source_path,
        "audit_path": audit_path,
        "request_path": request_path,
        "review_path": review_path,
        "integration_id": integration_id,
        "main_sha": main_sha,
        "dev_sha": dev_sha,
        "remote": text(source.get("remote_location"), "remote_location"),
        "main_ref": main_ref,
        "dev_ref": dev_ref,
        "operations": operations,
    }


def remote_refs(remote: str, refs: Sequence[str]) -> dict[str, str]:
    output = str(git(["ls-remote", "--refs", remote, *refs], phase="prepare_freshness", code="SOURCE_UNRESOLVED"))
    result = {}
    for line in output.splitlines():
        if "\t" in line:
            object_sha, ref = line.split("\t", 1)
            if ref in refs:
                if ref in result:
                    fail("SOURCE_UNRESOLVED", "prepare_freshness", f"duplicate ref: {ref}")
                result[ref] = object_sha
    missing = sorted(set(refs) - set(result))
    if missing:
        fail("SOURCE_UNRESOLVED", "prepare_freshness", f"missing refs: {missing}")
    return result


def tree_entry(repo: Path, commit: str, path: str, phase: str) -> dict[str, str] | None:
    raw = bytes(git(["ls-tree", "-z", commit, "--", path], cwd=repo, phase=phase, code="BLOB_OR_MODE_MISMATCH", binary=True))
    entries = [item for item in raw.split(b"\0") if item]
    if not entries:
        return None
    if len(entries) != 1:
        fail("BLOB_OR_MODE_MISMATCH", phase, f"ambiguous tree entry: {path}")
    metadata, encoded = entries[0].split(b"\t", 1)
    mode, object_type, object_sha = metadata.decode().split()
    actual = encoded.decode()
    if actual != path:
        fail("BLOB_OR_MODE_MISMATCH", phase, f"tree path mismatch: {path}")
    return {"mode": mode, "type": object_type, "sha": object_sha, "path": actual}


def validate_blob(repo: Path, object_sha: str) -> None:
    object_type = str(git(["cat-file", "-t", object_sha], cwd=repo, phase="prepare_source_reaudit", code="BLOB_OR_MODE_MISMATCH")).strip()
    if object_type != "blob":
        fail("BLOB_OR_MODE_MISMATCH", "prepare_source_reaudit", f"not a blob: {object_sha}")
    size = int(str(git(["cat-file", "-s", object_sha], cwd=repo, phase="prepare_source_reaudit", code="BLOB_OR_MODE_MISMATCH")).strip())
    if size > MAX_BLOB_BYTES:
        fail("SCOPE_VIOLATION", "prepare_source_reaudit", f"blob exceeds {MAX_BLOB_BYTES} bytes: {object_sha}")
    payload = bytes(git(["cat-file", "blob", object_sha], cwd=repo, phase="prepare_source_reaudit", code="BLOB_OR_MODE_MISMATCH", binary=True))
    if payload.startswith(LFS_PREFIX):
        fail("SCOPE_VIOLATION", "prepare_source_reaudit", "Git LFS pointer import is not supported")


def init_repo(repo: Path, context: dict[str, Any]) -> None:
    git(["init", str(repo)], phase="prepare_worktree", code="INTERNAL_ERROR")
    (repo / ".git" / "no-hooks").mkdir(parents=True)
    git(["config", "core.hooksPath", ".git/no-hooks"], cwd=repo, phase="prepare_worktree", code="INTERNAL_ERROR")
    git(["remote", "add", "source", context["remote"]], cwd=repo, phase="prepare_worktree", code="INTERNAL_ERROR")
    git([
        "fetch", "--no-tags", "source",
        f"+{context['main_ref']}:refs/integration/source-main",
        f"+{context['dev_ref']}:refs/integration/source-dev",
    ], cwd=repo, phase="prepare_worktree", code="SOURCE_UNRESOLVED", timeout=300)
    fetched = {
        context["main_ref"]: str(git(["rev-parse", "refs/integration/source-main"], cwd=repo, phase="prepare_worktree", code="SOURCE_UNRESOLVED")).strip(),
        context["dev_ref"]: str(git(["rev-parse", "refs/integration/source-dev"], cwd=repo, phase="prepare_worktree", code="SOURCE_UNRESOLVED")).strip(),
    }
    if fetched[context["main_ref"]] != context["main_sha"] or fetched[context["dev_ref"]] != context["dev_sha"]:
        fail("SOURCE_DRIFT", "prepare_worktree", "refs moved during fetch", "create a new plan attempt")


def reaudit(repo: Path, context: dict[str, Any]) -> None:
    for item in context["operations"]:
        source, destination = item["source_path"], item["destination_path"]
        main_entry = tree_entry(repo, context["main_sha"], source, "prepare_source_reaudit")
        dev_entry = None if destination is None else tree_entry(repo, context["dev_sha"], destination, "prepare_source_reaudit")
        if main_entry != item["main_entry"] or dev_entry != item["dev_entry"]:
            fail("IMMUTABILITY_ERROR", "prepare_source_reaudit", f"tree audit drift: {source}")
        if item["expected_blob_sha"]:
            validate_blob(repo, item["expected_blob_sha"])


def parse_diff(raw: bytes) -> list[dict[str, str | None]]:
    fields = [item for item in raw.split(b"\0") if item]
    result, index = [], 0
    while index < len(fields):
        status = fields[index].decode()
        index += 1
        code = status[0]
        if code in {"R", "C"}:
            source, destination = fields[index].decode(), fields[index + 1].decode()
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
            "source_path": path_value(source, "diff source"),
            "destination_path": None if destination is None else path_value(destination, "diff destination"),
        })
    if any(item["kind"] in {"copy", "unsupported"} for item in result):
        fail("SCOPE_VIOLATION", "prepare_verify", f"unsupported diff: {result}")
    return result


def verify_commit(repo: Path, context: dict[str, Any], commit_sha: str) -> dict[str, Any]:
    parents = str(git(["rev-list", "--parents", "-n", "1", commit_sha], cwd=repo, phase="prepare_verify", code="HEAD_DRIFT")).strip().split()
    if parents != [commit_sha, context["main_sha"]]:
        fail("HEAD_DRIFT", "prepare_verify", "source commit parent mismatch")
    raw = bytes(git(["diff-tree", "--no-commit-id", "--name-status", "-r", "-z", "-M", commit_sha], cwd=repo, phase="prepare_verify", code="SCOPE_VIOLATION", binary=True))
    changes = parse_diff(raw)
    expected = {op_key(item) for item in context["operations"]}
    actual = {(item["kind"], item["source_path"], item["destination_path"]) for item in changes}
    if actual != expected:
        fail("SCOPE_VIOLATION", "prepare_verify", f"commit scope mismatch: {actual ^ expected}")
    for item in context["operations"]:
        if item["op"] in {"delete", "rename"} and tree_entry(repo, commit_sha, item["source_path"], "prepare_verify"):
            fail("BLOB_OR_MODE_MISMATCH", "prepare_verify", f"source remains: {item['source_path']}")
        if item["op"] != "delete":
            entry = tree_entry(repo, commit_sha, item["destination_path"], "prepare_verify")
            if not entry or entry["type"] != "blob" or entry["sha"] != item["expected_blob_sha"] or entry["mode"] != item["expected_mode"]:
                fail("BLOB_OR_MODE_MISMATCH", "prepare_verify", f"destination mismatch: {item['destination_path']}")
    tree_sha = str(git(["rev-parse", f"{commit_sha}^{{tree}}"], cwd=repo, phase="prepare_verify", code="HEAD_DRIFT")).strip()
    return {"tree_sha": tree_sha, "changes": changes}


def build_commit(repo: Path, context: dict[str, Any]) -> dict[str, Any]:
    index = repo / ".git" / "drpo-integration.index"
    index_env = {"GIT_INDEX_FILE": str(index)}
    try:
        git(["read-tree", context["main_sha"]], cwd=repo, phase="prepare_commit", code="INTERNAL_ERROR", env=index_env)
        for item in context["operations"]:
            if item["op"] in {"delete", "rename"}:
                git(["update-index", "--force-remove", "--", item["source_path"]], cwd=repo, phase="prepare_commit", code="BLOB_OR_MODE_MISMATCH", env=index_env)
            if item["op"] != "delete":
                git(["update-index", "--add", "--cacheinfo", item["expected_mode"], item["expected_blob_sha"], item["destination_path"]], cwd=repo, phase="prepare_commit", code="BLOB_OR_MODE_MISMATCH", env=index_env)
        git(["diff", "--cached", "--check", context["main_sha"]], cwd=repo, phase="prepare_commit", code="SCOPE_VIOLATION", env=index_env)
        tree_sha = str(git(["write-tree"], cwd=repo, phase="prepare_commit", code="INTERNAL_ERROR", env=index_env)).strip()
    finally:
        index.unlink(missing_ok=True)
    message = f"Integrate reviewed dev snapshot {context['integration_id']}"
    identity = {
        "GIT_AUTHOR_NAME": "DRPO Integration Tool",
        "GIT_AUTHOR_EMAIL": "drpo-integration@local.invalid",
        "GIT_COMMITTER_NAME": "DRPO Integration Tool",
        "GIT_COMMITTER_EMAIL": "drpo-integration@local.invalid",
    }
    commit_sha = str(git(["commit-tree", tree_sha, "-p", context["main_sha"], "-m", message], cwd=repo, phase="prepare_commit", code="INTERNAL_ERROR", env=identity)).strip()
    fragment = re.sub(r"[^a-z0-9._-]+", "-", context["integration_id"].lower()).strip("-")
    branch = f"integration/{fragment}/{context['transaction_dir'].name}"
    git(["update-ref", f"refs/heads/{branch}", commit_sha], cwd=repo, phase="prepare_commit", code="INTERNAL_ERROR")
    git(["checkout", "-f", branch], cwd=repo, phase="prepare_checkout", code="WORKTREE_DIRTY", timeout=300)
    if str(git(["status", "--porcelain=v1"], cwd=repo, phase="prepare_checkout", code="WORKTREE_DIRTY")):
        fail("WORKTREE_DIRTY", "prepare_checkout", "source checkout is dirty")
    verified = verify_commit(repo, context, commit_sha)
    if verified["tree_sha"] != tree_sha:
        fail("HEAD_DRIFT", "prepare_verify", "written tree changed")
    return {"source_commit_sha": commit_sha, "tree_sha": tree_sha, "branch_name": branch, "commit_message": message, "changes": verified["changes"]}


def verify_prepared(transaction_dir: Path, context: dict[str, Any]) -> dict[str, Any]:
    report_path, repo = transaction_dir / "PREPARE_REPORT.json", transaction_dir / "integration-repo"
    if not report_path.is_file() or not repo.is_dir():
        fail("IMMUTABILITY_ERROR", "prepare_idempotence", "prepared artifacts are incomplete")
    report = load_json(report_path, "prepare report")
    commit_sha = full_sha(report.get("source_commit_sha"), "source_commit_sha")
    expected = {
        "schema_version": SCHEMA,
        "integration_id": context["integration_id"],
        "main_sha": context["main_sha"],
        "dev_sha": context["dev_sha"],
        "integration_repo": str(repo),
        "parent_sha": context["main_sha"],
        "operation_count": len(context["operations"]),
        "source_lock_sha256": sha256(context["source_path"]),
        "scope_audit_sha256": sha256(context["audit_path"]),
        "request_sha256": sha256(context["request_path"]),
        "review_decision_sha256": sha256(context["review_path"]),
    }
    for key, value in expected.items():
        if report.get(key) != value:
            fail("IMMUTABILITY_ERROR", "prepare_idempotence", f"report mismatch: {key}")
    if str(git(["rev-parse", "HEAD"], cwd=repo, phase="prepare_idempotence", code="HEAD_DRIFT")).strip() != commit_sha:
        fail("HEAD_DRIFT", "prepare_idempotence", "prepared HEAD drifted")
    if str(git(["status", "--porcelain=v1"], cwd=repo, phase="prepare_idempotence", code="WORKTREE_DIRTY")):
        fail("WORKTREE_DIRTY", "prepare_idempotence", "prepared repo is dirty")
    verified = verify_commit(repo, context, commit_sha)
    if report.get("tree_sha") != verified["tree_sha"]:
        fail("HEAD_DRIFT", "prepare_idempotence", "prepared tree drifted")
    branch = str(git(["symbolic-ref", "--short", "HEAD"], cwd=repo, phase="prepare_idempotence", code="HEAD_DRIFT")).strip()
    message = str(git(["log", "-1", "--format=%B", "HEAD"], cwd=repo, phase="prepare_idempotence", code="HEAD_DRIFT")).strip()
    if report.get("branch_name") != branch or report.get("commit_message") != message:
        fail("IMMUTABILITY_ERROR", "prepare_idempotence", "prepared branch or commit message drifted")
    if report.get("committed_changes") != verified["changes"]:
        fail("IMMUTABILITY_ERROR", "prepare_idempotence", "reported committed changes drifted")
    tx, recovered = context["transaction"], context["transaction"].get("state") == "REVIEWED"
    if not recovered:
        for key, value in {
            "source_commit_sha": commit_sha,
            "source_commit_parent_sha": context["main_sha"],
            "source_commit_tree_sha": verified["tree_sha"],
            "integration_repo": str(repo),
        }.items():
            if tx.get(key) != value:
                fail("IMMUTABILITY_ERROR", "prepare_idempotence", f"transaction mismatch: {key}")
    else:
        completed = list(tx.get("completed_states", []))
        if "PREPARED" not in completed:
            completed.append("PREPARED")
        tx.update({
            "tool_version": VERSION, "state": "PREPARED", "status": "PASS",
            "completed_states": completed, "integration_repo": str(repo),
            "source_commit_sha": commit_sha, "source_commit_parent_sha": context["main_sha"],
            "source_commit_tree_sha": verified["tree_sha"], "updated_at": now(),
            "next_action": "Batch 2B registry/normalization/gates/finalize is not implemented",
        })
        write_json(context["transaction_path"], tx)
    return {"status": "PASS", "state": "PREPARED", "integration_id": context["integration_id"], "source_commit_sha": commit_sha, "integration_repo": str(repo), "operation_count": len(context["operations"]), "idempotent": True, "recovered": recovered}


def diagnostic(transaction_dir: Path, context: dict[str, Any] | None, error: WritePathError) -> None:
    transaction_path = transaction_dir / "TRANSACTION.json"
    try:
        tx = load_json(transaction_path, "transaction")
    except WritePathError:
        tx = {"schema_version": SCHEMA}
    state, timestamp = ("STALE" if error.error_code == "SOURCE_DRIFT" else "BLOCKED"), now()
    write_json(transaction_dir / "DIAGNOSTIC.json", {
        "schema_version": SCHEMA, "tool_version": VERSION, "status": "FAIL", "state": state,
        "error_code": error.error_code, "phase": error.phase, "message": error.message,
        "recovery": list(error.recovery), "integration_id": None if context is None else context.get("integration_id"),
        "main_sha": None if context is None else context.get("main_sha"), "dev_sha": None if context is None else context.get("dev_sha"),
        "transaction_dir": str(transaction_dir), "created_at": timestamp,
    })
    tx.update({
        "schema_version": SCHEMA, "tool_version": VERSION, "state": state, "status": "FAIL",
        "error_code": error.error_code, "phase": error.phase, "updated_at": timestamp,
        "next_action": "create a new plan attempt" if state == "STALE" else "resolve diagnostic and create a new attempt",
    })
    write_json(transaction_path, tx)


def _prepare(transaction_dir: Path) -> dict[str, Any]:
    context = temporary = None
    try:
        context = load_context(transaction_dir)
        context["transaction_dir"] = transaction_dir
        final_repo, report = transaction_dir / "integration-repo", transaction_dir / "PREPARE_REPORT.json"
        if context["transaction"]["state"] == "PREPARED" or final_repo.exists() or report.exists():
            if final_repo.exists() != report.exists():
                fail("IMMUTABILITY_ERROR", "prepare_preflight", "partial prepared artifacts", "preserve and inspect this attempt")
            return verify_prepared(transaction_dir, context)
        refs = remote_refs(context["remote"], [context["main_ref"], context["dev_ref"]])
        if refs[context["main_ref"]] != context["main_sha"] or refs[context["dev_ref"]] != context["dev_sha"]:
            fail("SOURCE_DRIFT", "prepare_freshness", "main or dev moved after plan", "create a new plan attempt")
        temporary = transaction_dir / f".integration-repo.tmp-{os.getpid()}"
        if temporary.exists():
            fail("IMMUTABILITY_ERROR", "prepare_preflight", f"temporary repo exists: {temporary}")
        init_repo(temporary, context)
        reaudit(temporary, context)
        built = build_commit(temporary, context)
        os.replace(temporary, final_repo)
        temporary = None
        timestamp = now()
        if report.exists():
            fail("IMMUTABILITY_ERROR", "prepare_commit", "prepare report already exists")
        report_payload = {
            "schema_version": SCHEMA, "tool_version": VERSION, "status": "PASS", "state": "PREPARED",
            "integration_id": context["integration_id"], "main_sha": context["main_sha"], "dev_sha": context["dev_sha"],
            "branch_name": built["branch_name"], "integration_repo": str(final_repo), "source_commit_sha": built["source_commit_sha"],
            "parent_sha": context["main_sha"], "tree_sha": built["tree_sha"], "commit_message": built["commit_message"],
            "committed_changes": built["changes"], "operation_count": len(context["operations"]),
            "source_lock_sha256": sha256(context["source_path"]), "scope_audit_sha256": sha256(context["audit_path"]),
            "request_sha256": sha256(context["request_path"]), "review_decision_sha256": sha256(context["review_path"]),
            "prepared_at": timestamp, "next_action": "Batch 2B registry/normalization/gates/finalize is not implemented",
        }
        write_json(report, report_payload)
        tx = context["transaction"]
        completed = list(tx.get("completed_states", []))
        if "PREPARED" not in completed:
            completed.append("PREPARED")
        tx.update({
            "tool_version": VERSION, "state": "PREPARED", "status": "PASS", "completed_states": completed,
            "integration_repo": str(final_repo), "source_commit_sha": built["source_commit_sha"],
            "source_commit_parent_sha": context["main_sha"], "source_commit_tree_sha": built["tree_sha"],
            "updated_at": timestamp, "next_action": "Batch 2B registry/normalization/gates/finalize is not implemented",
        })
        write_json(context["transaction_path"], tx)
        return {"status": "PASS", "state": "PREPARED", "integration_id": context["integration_id"], "source_commit_sha": built["source_commit_sha"], "integration_repo": str(final_repo), "operation_count": len(context["operations"]), "idempotent": False, "recovered": False}
    except WritePathError as error:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)
        diagnostic(transaction_dir, context, error)
        raise
    except Exception as exc:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)
        error = WritePathError("INTERNAL_ERROR", "prepare_internal", f"unexpected error: {exc}")
        diagnostic(transaction_dir, context, error)
        raise error from exc


def prepare_transaction(transaction_dir: Path) -> dict[str, Any]:
    transaction_dir = transaction_dir.expanduser().resolve()
    if not transaction_dir.is_dir():
        fail("REQUEST_INVALID", "prepare_preflight", f"transaction directory is missing: {transaction_dir}")
    with locked(transaction_dir):
        return _prepare(transaction_dir)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transaction-dir", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = prepare_transaction(Path(args.transaction_dir))
        print(json.dumps(result, sort_keys=True) if args.json else f"PASS {result['integration_id']}: {result['source_commit_sha']}")
        return 0
    except WritePathError as error:
        payload = {"status": "FAIL", "error_code": error.error_code, "phase": error.phase, "message": error.message}
        print(json.dumps(payload, sort_keys=True) if args.json else f"FAIL {error}", file=sys.stdout if args.json else sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
