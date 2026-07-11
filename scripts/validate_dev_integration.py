#!/usr/bin/env python3
"""Validate DRPO dev-integration requests before Git or worktree mutation."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

import yaml

SCHEMA = 1
SHA_RE = re.compile(r"[0-9a-f]{40}")
ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{2,127}")
REPO_RE = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
REMOTE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
BRANCH_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,254}")
OPERATIONS = {"add", "modify", "delete", "rename"}
MODES = {"100644", "100755"}
EVIDENCE = {"not_applicable", "smoke", "pilot", "finite_step", "formal", "closure"}
RESULTS = {
    "not_applicable", "not_run", "pilot", "finite_step_validated",
    "long_run_validated", "analytically_proven", "rejected", "superseded",
}
CLAIM_SUPPORT = {"none", "diagnostic", "partial", "supports_claim"}
TERMINAL = {"not_required", "missing", "partial", "complete"}
EVENTS = {"not_assessed", "none", "observed", "inconclusive"}
GATE_TIERS = {"auto", "fast", "full"}
FORBIDDEN = (
    "docs/handoff.md",
    "experiments/registry.yaml",
    "docs/handoff_deltas/**",
    "docs/handoff_shadow/**/generated/**",
    ".git/**",
)
REQUEST_KEYS = {"schema_version", "integration_id", "source", "subject", "files", "review", "checks"}
SOURCE_KEYS = {
    "repository", "remote", "main_ref", "expected_main_sha", "dev_branch",
    "expected_dev_sha", "result_commit_sha", "result_git_dirty",
}
OP_KEYS = {
    "op", "source_path", "destination_path", "expected_blob_sha",
    "expected_old_blob_sha", "expected_mode",
}
DECISION_KEYS = {
    "approved", "code_integration_eligible", "evidence_level", "result_status",
    "claim_support_level", "terminal_audit", "task_performance_collapse",
    "support_boundary", "numerical_failure",
}


@dataclass(frozen=True)
class IntegrationError(Exception):
    error_code: str
    phase: str
    message: str
    recovery: tuple[str, ...] = ()

    def __str__(self) -> str:
        return f"{self.error_code}: {self.message}"


def mapping(value: Any, label: str, code: str = "REQUEST_INVALID") -> dict[str, Any]:
    if not isinstance(value, dict):
        raise IntegrationError(code, "validation", f"{label} must be a mapping")
    return value


def string(value: Any, label: str, code: str = "REQUEST_INVALID") -> str:
    if not isinstance(value, str) or not value.strip():
        raise IntegrationError(code, "validation", f"{label} must be a non-empty string")
    return value.strip()


def string_list(value: Any, label: str, code: str = "REQUEST_INVALID") -> list[str]:
    if not isinstance(value, list):
        raise IntegrationError(code, "validation", f"{label} must be a list")
    result = [string(item, f"{label}[{i}]", code) for i, item in enumerate(value)]
    if len(set(result)) != len(result):
        raise IntegrationError(code, "validation", f"{label} contains duplicates")
    return result


def exact_keys(data: dict[str, Any], allowed: set[str], label: str, code: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise IntegrationError(code, "validation", f"{label} contains unknown keys: {unknown}")


def full_sha(value: Any, label: str, allow_none: bool = False) -> str | None:
    if value is None and allow_none:
        return None
    result = string(value, label)
    if not SHA_RE.fullmatch(result):
        raise IntegrationError("REQUEST_INVALID", "validation", f"{label} must be a full lowercase Git SHA")
    return result


def repo_path(value: Any, label: str) -> str:
    raw = string(value, label).replace("\\", "/")
    if any(ord(char) < 32 or ord(char) == 127 for char in raw):
        raise IntegrationError("UNSAFE_PATH", "scope_audit", f"{label} contains control bytes")
    path = PurePosixPath(raw)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise IntegrationError("UNSAFE_PATH", "scope_audit", f"{label} is unsafe: {value!r}")
    normalized = path.as_posix()
    if normalized == ".git" or normalized.startswith(".git/"):
        raise IntegrationError("UNSAFE_PATH", "scope_audit", f"{label} is unsafe: {value!r}")
    return normalized


def is_forbidden_path(path: str) -> bool:
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in FORBIDDEN)


def load_yaml(path: Path, code: str) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise IntegrationError(code, "validation", f"cannot read YAML {path}: {exc}") from exc
    return mapping(payload, str(path), code)


def validate_operation(raw: Any, index: int) -> dict[str, Any]:
    label = f"files.operations[{index}]"
    item = mapping(raw, label)
    exact_keys(item, OP_KEYS, label, "REQUEST_INVALID")
    op = string(item.get("op"), f"{label}.op")
    if op not in OPERATIONS:
        raise IntegrationError("REQUEST_INVALID", "validation", f"invalid operation {op!r}")
    source = repo_path(item.get("source_path"), f"{label}.source_path")
    destination = None if op == "delete" else repo_path(
        item.get("destination_path"), f"{label}.destination_path"
    )
    if op == "delete" and item.get("destination_path") not in (None, ""):
        raise IntegrationError("REQUEST_INVALID", "validation", "delete may not define destination_path")
    if op in {"add", "modify"} and source != destination:
        raise IntegrationError("REQUEST_INVALID", "validation", f"{op} must keep the same path")
    if op == "rename" and source == destination:
        raise IntegrationError("REQUEST_INVALID", "validation", "rename must change the path")
    blob = full_sha(item.get("expected_blob_sha"), f"{label}.expected_blob_sha", op == "delete")
    old_blob = full_sha(item.get("expected_old_blob_sha"), f"{label}.expected_old_blob_sha", True)
    if op == "delete":
        if item.get("expected_mode") not in (None, ""):
            raise IntegrationError("REQUEST_INVALID", "validation", "delete may not define expected_mode")
        mode = None
    else:
        mode = string(item.get("expected_mode"), f"{label}.expected_mode")
        if mode not in MODES:
            raise IntegrationError("UNSAFE_PATH", "scope_audit", f"unsafe Git mode {mode}")
    return {
        "op": op,
        "source_path": source,
        "destination_path": destination,
        "expected_blob_sha": blob,
        "expected_old_blob_sha": old_blob,
        "expected_mode": mode,
    }


def validate_request(payload: dict[str, Any]) -> dict[str, Any]:
    exact_keys(payload, REQUEST_KEYS, "request", "REQUEST_INVALID")
    if payload.get("schema_version") != SCHEMA:
        raise IntegrationError("REQUEST_INVALID", "validation", f"schema_version must be {SCHEMA}")
    integration_id = string(payload.get("integration_id"), "integration_id")
    if not ID_RE.fullmatch(integration_id):
        raise IntegrationError("REQUEST_INVALID", "validation", "invalid integration_id")

    source = mapping(payload.get("source"), "source")
    exact_keys(source, SOURCE_KEYS, "source", "REQUEST_INVALID")
    repository = string(source.get("repository"), "source.repository")
    remote = string(source.get("remote", "origin"), "source.remote")
    main_ref = string(source.get("main_ref", "refs/heads/main"), "source.main_ref")
    dev_branch = string(source.get("dev_branch"), "source.dev_branch")
    if not REPO_RE.fullmatch(repository) or not REMOTE_RE.fullmatch(remote):
        raise IntegrationError("REQUEST_INVALID", "validation", "invalid repository or remote")
    if not main_ref.startswith("refs/heads/") or not BRANCH_RE.fullmatch(dev_branch):
        raise IntegrationError("REQUEST_INVALID", "validation", "invalid main ref or dev branch")
    normalized_source = {
        "repository": repository,
        "remote": remote,
        "main_ref": main_ref,
        "expected_main_sha": full_sha(source.get("expected_main_sha"), "source.expected_main_sha"),
        "dev_branch": dev_branch,
        "expected_dev_sha": full_sha(source.get("expected_dev_sha"), "source.expected_dev_sha"),
        "result_commit_sha": full_sha(source.get("result_commit_sha"), "source.result_commit_sha", True),
        "result_git_dirty": source.get("result_git_dirty", False),
    }
    if not isinstance(normalized_source["result_git_dirty"], bool):
        raise IntegrationError("REQUEST_INVALID", "validation", "source.result_git_dirty must be boolean")

    subject = mapping(payload.get("subject", {}), "subject")
    exact_keys(subject, {"experiment_ids", "governance_claims"}, "subject", "REQUEST_INVALID")
    experiments = string_list(subject.get("experiment_ids", []), "subject.experiment_ids")
    claims = string_list(subject.get("governance_claims", []), "subject.governance_claims")
    if not experiments and not claims:
        raise IntegrationError("REQUEST_INVALID", "validation", "subject must identify an experiment or claim")

    files = mapping(payload.get("files"), "files")
    exact_keys(files, {"operations"}, "files", "REQUEST_INVALID")
    raw_operations = files.get("operations")
    if not isinstance(raw_operations, list) or not raw_operations:
        raise IntegrationError("REQUEST_INVALID", "validation", "files.operations must be non-empty")
    operations = [validate_operation(item, i) for i, item in enumerate(raw_operations)]
    keys: set[tuple[str, str, str | None]] = set()
    targets: dict[str, str] = {}
    for operation in operations:
        key = (operation["op"], operation["source_path"], operation["destination_path"])
        if key in keys:
            raise IntegrationError("REQUEST_INVALID", "validation", f"duplicate operation: {key}")
        keys.add(key)
        target = operation["destination_path"] or operation["source_path"]
        previous = targets.get(target.casefold())
        if previous is not None:
            raise IntegrationError("SCOPE_VIOLATION", "scope_audit", f"target collision: {previous!r}, {target!r}")
        targets[target.casefold()] = target

    review = mapping(payload.get("review"), "review")
    exact_keys(review, {"decision_file"}, "review", "REQUEST_INVALID")
    checks = mapping(payload.get("checks", {}), "checks")
    exact_keys(checks, {"requested_tier"}, "checks", "REQUEST_INVALID")
    tier = string(checks.get("requested_tier", "auto"), "checks.requested_tier")
    if tier not in GATE_TIERS:
        raise IntegrationError("REQUEST_INVALID", "validation", f"invalid gate tier {tier!r}")
    return {
        "schema_version": SCHEMA,
        "integration_id": integration_id,
        "source": normalized_source,
        "subject": {"experiment_ids": experiments, "governance_claims": claims},
        "files": {"operations": operations},
        "review": {"decision_file": repo_path(review.get("decision_file"), "review.decision_file")},
        "checks": {"requested_tier": tier},
    }


def validate_reviewer_decision(payload: dict[str, Any], integration_id: str) -> dict[str, Any]:
    code = "REVIEW_DECISION_INVALID"
    exact_keys(
        payload,
        {"schema_version", "integration_id", "decision", "reviewer", "limitations", "unresolved"},
        "review decision",
        code,
    )
    if payload.get("schema_version") != SCHEMA or payload.get("integration_id") != integration_id:
        raise IntegrationError(code, "scientific_review", "review schema or integration_id mismatch")
    decision = mapping(payload.get("decision"), "decision", code)
    exact_keys(decision, DECISION_KEYS, "decision", code)
    approved = decision.get("approved")
    eligible = decision.get("code_integration_eligible")
    if not isinstance(approved, bool) or not isinstance(eligible, bool):
        raise IntegrationError(code, "scientific_review", "approval fields must be boolean")
    normalized = {
        "approved": approved,
        "code_integration_eligible": eligible,
        "evidence_level": string(decision.get("evidence_level"), "decision.evidence_level", code),
        "result_status": string(decision.get("result_status"), "decision.result_status", code),
        "claim_support_level": string(decision.get("claim_support_level"), "decision.claim_support_level", code),
        "terminal_audit": string(decision.get("terminal_audit"), "decision.terminal_audit", code),
        "task_performance_collapse": string(decision.get("task_performance_collapse"), "decision.task_performance_collapse", code),
        "support_boundary": string(decision.get("support_boundary"), "decision.support_boundary", code),
        "numerical_failure": string(decision.get("numerical_failure"), "decision.numerical_failure", code),
    }
    choices = {
        "evidence_level": EVIDENCE,
        "result_status": RESULTS,
        "claim_support_level": CLAIM_SUPPORT,
        "terminal_audit": TERMINAL,
        "task_performance_collapse": EVENTS,
        "support_boundary": EVENTS,
        "numerical_failure": EVENTS,
    }
    for key, allowed in choices.items():
        if normalized[key] not in allowed:
            raise IntegrationError(code, "scientific_review", f"invalid {key}: {normalized[key]!r}")
    reviewer = mapping(payload.get("reviewer"), "reviewer", code)
    exact_keys(reviewer, {"id", "decision_token"}, "reviewer", code)
    result = {
        "schema_version": SCHEMA,
        "integration_id": integration_id,
        "decision": normalized,
        "reviewer": {
            "id": string(reviewer.get("id"), "reviewer.id", code),
            "decision_token": string(reviewer.get("decision_token"), "reviewer.decision_token", code),
        },
        "limitations": string_list(payload.get("limitations", []), "limitations", code),
        "unresolved": string_list(payload.get("unresolved", []), "unresolved", code),
    }
    if not approved or not eligible:
        raise IntegrationError("SCIENTIFIC_REVIEW_MISSING", "scientific_review", "review does not approve integration")
    return result


def enforce_provenance(request: dict[str, Any], review: dict[str, Any]) -> None:
    source, decision = request["source"], review["decision"]
    formal = decision["evidence_level"] in {"formal", "closure"}
    if formal and source["result_commit_sha"] is None:
        raise IntegrationError("PROVENANCE_INCOMPLETE", "provenance", "formal evidence requires result_commit_sha")
    if formal and source["result_git_dirty"]:
        raise IntegrationError("PROVENANCE_INCOMPLETE", "provenance", "dirty-worktree evidence cannot be formal or closure")
    if decision["result_status"] in {"finite_step_validated", "long_run_validated"} and decision["terminal_audit"] != "complete":
        raise IntegrationError("PROVENANCE_INCOMPLETE", "provenance", "validated status requires terminal_audit=complete")


def validate_inputs(repo_root: Path, request_path: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    request = validate_request(load_yaml(request_path, "REQUEST_INVALID"))
    decision_path = (repo_root / request["review"]["decision_file"]).resolve()
    try:
        decision_path.relative_to(repo_root)
    except ValueError as exc:
        raise IntegrationError("UNSAFE_PATH", "scientific_review", "review decision escapes repo root") from exc
    review = validate_reviewer_decision(
        load_yaml(decision_path, "REVIEW_DECISION_INVALID"), request["integration_id"]
    )
    enforce_provenance(request, review)
    return request, review, decision_path


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


def write_json(path: Path, payload: Any, overwrite: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise IntegrationError("INTERNAL_ERROR", "transaction", f"refusing to overwrite {path}")
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--request", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root).expanduser().resolve()
    request_path = Path(args.request).expanduser().resolve()
    try:
        request, _, decision_path = validate_inputs(repo_root, request_path)
        output = {
            "status": "PASS",
            "integration_id": request["integration_id"],
            "request_sha256": file_hash(request_path),
            "review_decision_sha256": file_hash(decision_path),
        }
        print(json.dumps(output, sort_keys=True) if args.json else f"PASS {request['integration_id']}")
        return 0
    except IntegrationError as error:
        output = {
            "status": "FAIL", "error_code": error.error_code,
            "phase": error.phase, "message": error.message,
        }
        print(
            json.dumps(output, sort_keys=True) if args.json else f"FAIL {error}",
            file=sys.stdout if args.json else sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
