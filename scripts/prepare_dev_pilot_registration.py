#!/usr/bin/env python3
"""Compile one reviewed pilot spec into existing DRPO V1 integration inputs.

This is a local preparation adapter only. It performs no Git network operation,
repository mutation, authority normalization, CI action, push, PR, or merge.
The accepted V1 transaction remains the only path to local READY.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from validate_dev_integration import (
    IntegrationError,
    enforce_provenance,
    is_forbidden_path,
    json_hash,
    repo_path,
    validate_request,
    validate_reviewer_decision,
)

SCHEMA_VERSION = 1
TOOL_VERSION = "0.1.1-pr-a"
ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{2,127}")
SHA_RE = re.compile(r"[0-9a-f]{40}")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
REPOSITORY_RE = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
REMOTE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
BRANCH_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,254}")
REGISTRATION_MODES = {"none", "add_experiment", "replace_experiment"}

TOP_KEYS = {
    "schema_version",
    "preparation_id",
    "source",
    "subject",
    "implementation",
    "review",
    "registration",
}
SOURCE_KEYS = {
    "repository",
    "remote",
    "main_ref",
    "expected_main_sha",
    "dev_branch",
    "expected_dev_sha",
    "result_commit_sha",
    "result_git_dirty",
}
SUBJECT_KEYS = {"experiment_id", "governance_claims"}
IMPLEMENTATION_KEYS = {"operations"}
REVIEW_KEYS = {
    "reviewer_id",
    "decision_token",
    "decision",
    "limitations",
    "unresolved",
}
REGISTRATION_KEYS = {
    "mode",
    "update_id",
    "expected_before_semantic_sha256",
    "experiment",
    "handoff_operations",
    "registry_changes",
}


@dataclass(frozen=True)
class PreparationError(Exception):
    error_code: str
    phase: str
    message: str

    def __str__(self) -> str:
        return f"{self.error_code}: {self.message}"


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def fail(code: str, phase: str, message: str) -> None:
    raise PreparationError(code, phase, message)


def mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        fail("SPEC_INVALID", "validation", f"{label} must be a mapping")
    return value


def list_value(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        fail("SPEC_INVALID", "validation", f"{label} must be a list")
    return value


def string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        fail("SPEC_INVALID", "validation", f"{label} must be a non-empty string")
    return value.strip()


def exact_keys(value: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(value) - allowed)
    missing = sorted(allowed - set(value))
    if unknown:
        fail("SPEC_INVALID", "validation", f"{label} contains unknown keys: {unknown}")
    if missing:
        fail("SPEC_INVALID", "validation", f"{label} is missing keys: {missing}")


def full_sha(value: Any, label: str, *, allow_none: bool = False) -> str | None:
    if value is None and allow_none:
        return None
    result = string(value, label)
    if not SHA_RE.fullmatch(result):
        fail("SPEC_INVALID", "validation", f"{label} must be a full lowercase Git SHA")
    return result


def sha256_value(value: Any, label: str, *, allow_none: bool = False) -> str | None:
    if value is None and allow_none:
        return None
    result = string(value, label)
    if not SHA256_RE.fullmatch(result):
        fail("SPEC_INVALID", "validation", f"{label} must be a lowercase SHA-256")
    return result


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_yaml_load(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink():
        fail("UNSAFE_PATH", "validation", f"{label} may not be a symlink")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        fail("SPEC_INVALID", "validation", f"cannot read {label}: {exc}")
    return mapping(payload, label)


def yaml_bytes(payload: Any) -> bytes:
    rendered = yaml.safe_dump(
        payload,
        sort_keys=False,
        allow_unicode=True,
        width=1000,
    )
    if not rendered.endswith("\n"):
        rendered += "\n"
    return rendered.encode("utf-8")


def json_bytes(payload: Any) -> bytes:
    rendered = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False)
    return (rendered + "\n").encode("utf-8")


def normalize_string_list(value: Any, label: str) -> list[str]:
    items = list_value(value, label)
    result = [string(item, f"{label}[{index}]") for index, item in enumerate(items)]
    if len(result) != len(set(result)):
        fail("SPEC_INVALID", "validation", f"{label} contains duplicates")
    return result


def normalize_source(raw: Any) -> dict[str, Any]:
    source = mapping(raw, "source")
    exact_keys(source, SOURCE_KEYS, "source")
    repository = string(source["repository"], "source.repository")
    remote = string(source["remote"], "source.remote")
    main_ref = string(source["main_ref"], "source.main_ref")
    dev_branch = string(source["dev_branch"], "source.dev_branch")
    dirty = source["result_git_dirty"]
    if not REPOSITORY_RE.fullmatch(repository):
        fail("SPEC_INVALID", "validation", "source.repository must be owner/name")
    if not REMOTE_RE.fullmatch(remote):
        fail("SPEC_INVALID", "validation", "source.remote is invalid")
    if not main_ref.startswith("refs/heads/"):
        fail("SPEC_INVALID", "validation", "source.main_ref must be refs/heads/<name>")
    if not BRANCH_RE.fullmatch(dev_branch):
        fail("SPEC_INVALID", "validation", "source.dev_branch is invalid")
    if not isinstance(dirty, bool):
        fail("SPEC_INVALID", "validation", "source.result_git_dirty must be boolean")
    return {
        "repository": repository,
        "remote": remote,
        "main_ref": main_ref,
        "expected_main_sha": full_sha(source["expected_main_sha"], "source.expected_main_sha"),
        "dev_branch": dev_branch,
        "expected_dev_sha": full_sha(source["expected_dev_sha"], "source.expected_dev_sha"),
        "result_commit_sha": full_sha(
            source["result_commit_sha"],
            "source.result_commit_sha",
            allow_none=True,
        ),
        "result_git_dirty": dirty,
    }


def normalize_subject(raw: Any) -> dict[str, Any]:
    subject = mapping(raw, "subject")
    exact_keys(subject, SUBJECT_KEYS, "subject")
    experiment_id = string(subject["experiment_id"], "subject.experiment_id")
    if not ID_RE.fullmatch(experiment_id):
        fail("SPEC_INVALID", "validation", "subject.experiment_id is invalid")
    claims = normalize_string_list(subject["governance_claims"], "subject.governance_claims")
    for claim in claims:
        if not ID_RE.fullmatch(claim):
            fail("SPEC_INVALID", "validation", f"invalid governance claim: {claim}")
    return {"experiment_id": experiment_id, "governance_claims": claims}


def normalize_implementation(raw: Any) -> list[dict[str, Any]]:
    implementation = mapping(raw, "implementation")
    exact_keys(implementation, IMPLEMENTATION_KEYS, "implementation")
    operations = list_value(implementation["operations"], "implementation.operations")
    if not operations:
        fail("SPEC_INVALID", "validation", "implementation.operations must be non-empty")
    return [
        dict(mapping(operation, f"implementation.operations[{index}]"))
        for index, operation in enumerate(operations)
    ]


def normalize_review(raw: Any, preparation_id: str) -> dict[str, Any]:
    review = mapping(raw, "review")
    exact_keys(review, REVIEW_KEYS, "review")
    payload = {
        "schema_version": 1,
        "integration_id": preparation_id,
        "decision": mapping(review["decision"], "review.decision"),
        "reviewer": {
            "id": string(review["reviewer_id"], "review.reviewer_id"),
            "decision_token": string(review["decision_token"], "review.decision_token"),
        },
        "limitations": normalize_string_list(review["limitations"], "review.limitations"),
        "unresolved": normalize_string_list(review["unresolved"], "review.unresolved"),
    }
    try:
        validate_reviewer_decision(payload, preparation_id)
    except IntegrationError as exc:
        fail(exc.error_code, exc.phase, exc.message)
    return payload


def load_registry(repo_root: Path) -> list[dict[str, Any]]:
    path = repo_root / "experiments" / "registry.yaml"
    if not path.is_file() or path.is_symlink():
        fail("REGISTRY_STRUCTURE_ERROR", "registration", f"registry is unavailable: {path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        fail("REGISTRY_STRUCTURE_ERROR", "registration", f"cannot read registry: {exc}")
    root = mapping(payload, "registry")
    if root.get("schema_version") != 2:
        fail("REGISTRY_STRUCTURE_ERROR", "registration", "registry schema_version must equal 2")
    experiments = root.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        fail("REGISTRY_STRUCTURE_ERROR", "registration", "registry experiments must be non-empty")
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for index, experiment in enumerate(experiments):
        item = mapping(experiment, f"registry.experiments[{index}]")
        experiment_id = string(item.get("id"), f"registry.experiments[{index}].id")
        if experiment_id in seen:
            fail("REGISTRY_STRUCTURE_ERROR", "registration", "registry contains duplicate IDs")
        seen.add(experiment_id)
        normalized.append(item)
    return normalized


def normalize_registration(
    raw: Any,
    *,
    subject: dict[str, Any],
    review: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any] | None:
    registration = mapping(raw, "registration")
    exact_keys(registration, REGISTRATION_KEYS, "registration")
    mode = string(registration["mode"], "registration.mode")
    if mode not in REGISTRATION_MODES:
        fail("SPEC_INVALID", "registration", f"unsupported registration mode: {mode}")
    if mode == "none":
        if registration["update_id"] is not None:
            fail("SPEC_INVALID", "registration", "mode=none requires update_id=null")
        if registration["expected_before_semantic_sha256"] is not None:
            fail(
                "SPEC_INVALID",
                "registration",
                "mode=none requires expected_before_semantic_sha256=null",
            )
        if registration["experiment"] is not None:
            fail("SPEC_INVALID", "registration", "mode=none requires experiment=null")
        if registration["handoff_operations"] != [] or registration["registry_changes"] != []:
            fail("SPEC_INVALID", "registration", "mode=none requires empty operation lists")
        return None

    decision = review["decision"]
    if not decision["approved"] or not decision["code_integration_eligible"]:
        fail(
            "SCIENTIFIC_REVIEW_MISSING",
            "registration",
            "registration preparation requires an explicitly approved reviewer decision",
        )
    update_id = string(registration["update_id"], "registration.update_id")
    if not ID_RE.fullmatch(update_id):
        fail("SPEC_INVALID", "registration", "registration.update_id is invalid")
    before_hash = sha256_value(
        registration["expected_before_semantic_sha256"],
        "registration.expected_before_semantic_sha256",
        allow_none=True,
    )
    experiment = mapping(registration["experiment"], "registration.experiment")
    experiment_id = subject["experiment_id"]
    if experiment.get("id") != experiment_id:
        fail("REGISTRY_STRUCTURE_ERROR", "registration", "registration experiment ID mismatch")
    handoff_operations = list_value(
        registration["handoff_operations"],
        "registration.handoff_operations",
    )
    registry_changes = list_value(
        registration["registry_changes"],
        "registration.registry_changes",
    )
    if not handoff_operations:
        fail("SPEC_INVALID", "registration", "handoff_operations must be non-empty")
    if not registry_changes or not all(isinstance(item, dict) for item in registry_changes):
        fail("SPEC_INVALID", "registration", "registry_changes must contain mappings")
    for index, change in enumerate(registry_changes):
        if change.get("entity_id") != experiment_id:
            fail(
                "SCOPE_VIOLATION",
                "registration",
                f"registry_changes[{index}] targets a different experiment",
            )

    existing = next(
        (item for item in load_registry(repo_root) if item.get("id") == experiment_id),
        None,
    )
    if mode == "add_experiment":
        if before_hash is not None:
            fail(
                "REGISTRY_STRUCTURE_ERROR",
                "registration",
                "add_experiment requires expected_before_semantic_sha256=null",
            )
        if existing is not None:
            fail("REGISTRY_STRUCTURE_ERROR", "registration", "add target already exists")
    else:
        if before_hash is None:
            fail(
                "REGISTRY_STRUCTURE_ERROR",
                "registration",
                "replace_experiment requires expected_before_semantic_sha256",
            )
        if existing is None:
            fail("REGISTRY_STRUCTURE_ERROR", "registration", "replace target is missing")
        if json_hash(existing) != before_hash:
            fail(
                "IMMUTABILITY_ERROR",
                "registration",
                "replacement before-image semantic hash mismatch",
            )
    return {
        "mode": mode,
        "update_id": update_id,
        "expected_before_semantic_sha256": before_hash,
        "experiment": experiment,
        "handoff_operations": handoff_operations,
        "registry_changes": registry_changes,
    }


def compile_inputs(spec: dict[str, Any], repo_root: Path) -> dict[str, bytes]:
    exact_keys(spec, TOP_KEYS, "spec")
    if spec["schema_version"] != SCHEMA_VERSION:
        fail("SPEC_INVALID", "validation", f"schema_version must equal {SCHEMA_VERSION}")
    preparation_id = string(spec["preparation_id"], "preparation_id")
    if not ID_RE.fullmatch(preparation_id):
        fail("SPEC_INVALID", "validation", "preparation_id is invalid")

    source = normalize_source(spec["source"])
    subject = normalize_subject(spec["subject"])
    operations = normalize_implementation(spec["implementation"])
    decision_path = f"docs/integrations/{preparation_id}/REVIEW_DECISION.yaml"
    request_payload = {
        "schema_version": 1,
        "integration_id": preparation_id,
        "source": source,
        "subject": {
            "experiment_ids": [subject["experiment_id"]],
            "governance_claims": subject["governance_claims"],
        },
        "files": {"operations": operations},
        "review": {"decision_file": decision_path},
        "checks": {"requested_tier": "auto"},
    }
    try:
        normalized_request = validate_request(request_payload)
    except IntegrationError as exc:
        fail(exc.error_code, exc.phase, exc.message)
    for operation in normalized_request["files"]["operations"]:
        paths = [operation["source_path"]]
        if operation["destination_path"] is not None:
            paths.append(operation["destination_path"])
        blocked = sorted(path for path in paths if is_forbidden_path(path))
        if blocked:
            fail("SCOPE_VIOLATION", "validation", f"system-forbidden paths: {blocked}")

    review_payload = normalize_review(spec["review"], preparation_id)
    try:
        enforce_provenance(normalized_request, review_payload)
    except IntegrationError as exc:
        fail(exc.error_code, exc.phase, exc.message)
    registration = normalize_registration(
        spec["registration"],
        subject=subject,
        review=review_payload,
        repo_root=repo_root,
    )

    overlay = Path("repository_overlay") / "docs" / "integrations" / preparation_id
    request_name = (overlay / "INTEGRATION_REQUEST.yaml").as_posix()
    review_name = (overlay / "REVIEW_DECISION.yaml").as_posix()
    result: dict[str, bytes] = {
        request_name: yaml_bytes(request_payload),
        review_name: yaml_bytes(review_payload),
    }
    if registration is not None:
        intent = {
            "schema_version": 1,
            "integration_id": preparation_id,
            "mode": "authoritative_delta",
            "update_id": registration["update_id"],
            "registry_mutation": {
                "kind": registration["mode"],
                "experiment_id": subject["experiment_id"],
                "expected_before_semantic_sha256": registration[
                    "expected_before_semantic_sha256"
                ],
                "experiment": registration["experiment"],
            },
            "handoff_operations": registration["handoff_operations"],
            "registry_changes": registration["registry_changes"],
        }
        intent_data = yaml_bytes(intent)
        approval = {
            "schema_version": 1,
            "integration_id": preparation_id,
            "intent_sha256": sha256_bytes(intent_data),
            "request_sha256": sha256_bytes(result[request_name]),
            "review_decision_sha256": sha256_bytes(result[review_name]),
            "reviewer": review_payload["reviewer"],
        }
        result["transaction_inputs/REGISTRATION_INTENT.yaml"] = intent_data
        result["transaction_inputs/REGISTRATION_APPROVAL.yaml"] = yaml_bytes(approval)
    return result


def write_file(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def verify_manifest(root: Path, manifest: dict[str, Any]) -> None:
    files = mapping(manifest.get("files"), "manifest.files")
    for relative, expected in sorted(files.items()):
        path = root / repo_path(relative, "manifest path")
        if not path.is_file() or path.is_symlink():
            fail("OUTPUT_CONFLICT", "idempotency", f"prepared file missing or unsafe: {relative}")
        if sha256_file(path) != expected:
            fail("OUTPUT_CONFLICT", "idempotency", f"prepared file hash mismatch: {relative}")


def existing_idempotent(final_dir: Path, expected_manifest: dict[str, Any]) -> bool:
    manifest_path = final_dir / "PREPARATION_MANIFEST.json"
    report_path = final_dir / "PREPARATION_REPORT.json"
    if not manifest_path.is_file() or not report_path.is_file():
        return False
    try:
        actual = json.loads(manifest_path.read_text(encoding="utf-8"))
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    if actual != expected_manifest:
        return False
    if report.get("status") != "PASS" or report.get("state") != "PREPARED_INPUTS":
        return False
    if report.get("manifest_sha256") != sha256_file(manifest_path):
        return False
    verify_manifest(final_dir, actual)
    return True


def outside_repo(repo_root: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(repo_root)
    except ValueError:
        return True
    return False


def acquire_lock(path: Path) -> int:
    try:
        return os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        fail("OUTPUT_LOCKED", "atomic_publish", f"preparation lock already exists: {path}")


def validate_written_overlay(temporary: Path, preparation_id: str) -> None:
    overlay = temporary / "repository_overlay"
    request_path = (
        overlay
        / "docs"
        / "integrations"
        / preparation_id
        / "INTEGRATION_REQUEST.yaml"
    )
    request = safe_yaml_load(request_path, "compiled integration request")
    review_path = overlay / request["review"]["decision_file"]
    review = safe_yaml_load(review_path, "compiled review decision")
    try:
        normalized_request = validate_request(request)
        normalized_review = validate_reviewer_decision(review, preparation_id)
        enforce_provenance(normalized_request, normalized_review)
    except IntegrationError as exc:
        fail(exc.error_code, exc.phase, exc.message)


def prepare(repo_root: Path, spec_path: Path, output_root: Path) -> dict[str, Any]:
    started_clock = time.monotonic()
    started_at = now()
    repo_root = repo_root.expanduser().resolve()
    spec_path = spec_path.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if not repo_root.is_dir():
        fail("SPEC_INVALID", "preflight", f"repo root is not a directory: {repo_root}")
    if not outside_repo(repo_root, output_root):
        fail(
            "UNSAFE_OUTPUT_ROOT",
            "preflight",
            "output root must be outside the repository worktree",
        )

    spec = safe_yaml_load(spec_path, "pilot registration spec")
    files = compile_inputs(spec, repo_root)
    preparation_id = string(spec["preparation_id"], "preparation_id")
    spec_sha = sha256_file(spec_path)
    manifest = {
        "schema_version": 1,
        "tool_version": TOOL_VERSION,
        "preparation_id": preparation_id,
        "spec_sha256": spec_sha,
        "files": {name: sha256_bytes(data) for name, data in sorted(files.items())},
        "network_used": False,
        "repository_modified": False,
    }

    output_root.mkdir(parents=True, exist_ok=True)
    final_dir = output_root / preparation_id
    if final_dir.exists():
        if (
            final_dir.is_dir()
            and not final_dir.is_symlink()
            and existing_idempotent(final_dir, manifest)
        ):
            return {
                "status": "PASS",
                "state": "PREPARED_INPUTS",
                "preparation_id": preparation_id,
                "preparation_dir": str(final_dir),
                "idempotent_reuse": True,
                "manifest_sha256": sha256_file(final_dir / "PREPARATION_MANIFEST.json"),
                "network_used": False,
                "repository_modified": False,
            }
        fail("OUTPUT_CONFLICT", "idempotency", f"output already exists and differs: {final_dir}")

    lock_path = output_root / f".{preparation_id}.lock"
    lock_fd = acquire_lock(lock_path)
    temporary: Path | None = None
    try:
        os.write(lock_fd, f"pid={os.getpid()}\n".encode())
        os.fsync(lock_fd)
        if final_dir.exists():
            if final_dir.is_dir() and existing_idempotent(final_dir, manifest):
                return {
                    "status": "PASS",
                    "state": "PREPARED_INPUTS",
                    "preparation_id": preparation_id,
                    "preparation_dir": str(final_dir),
                    "idempotent_reuse": True,
                    "manifest_sha256": sha256_file(
                        final_dir / "PREPARATION_MANIFEST.json"
                    ),
                    "network_used": False,
                    "repository_modified": False,
                }
            fail("OUTPUT_CONFLICT", "idempotency", f"output appeared and differs: {final_dir}")

        temporary = Path(tempfile.mkdtemp(prefix=f".{preparation_id}.", dir=output_root))
        injected_after = os.environ.get("DRPO_PREPARATION_INJECT_FAILURE_AFTER_FILES")
        count = 0
        for relative, data in sorted(files.items()):
            write_file(temporary / relative, data)
            count += 1
            if injected_after is not None and count >= int(injected_after):
                fail("INJECTED_FAILURE", "atomic_publish", "injected write failure")

        manifest_data = json_bytes(manifest)
        write_file(temporary / "PREPARATION_MANIFEST.json", manifest_data)
        validate_written_overlay(temporary, preparation_id)
        verify_manifest(temporary, manifest)
        report = {
            "schema_version": 1,
            "tool_version": TOOL_VERSION,
            "status": "PASS",
            "state": "PREPARED_INPUTS",
            "preparation_id": preparation_id,
            "spec_path": str(spec_path),
            "spec_sha256": spec_sha,
            "started_at": started_at,
            "ended_at": now(),
            "elapsed_ms": round((time.monotonic() - started_clock) * 1000, 3),
            "mode": spec["registration"]["mode"],
            "operation_count": len(spec["implementation"]["operations"]),
            "generated_file_count": len(files),
            "manifest_sha256": sha256_bytes(manifest_data),
            "repository_overlay": f"repository_overlay/docs/integrations/{preparation_id}",
            "transaction_inputs": (
                "transaction_inputs" if spec["registration"]["mode"] != "none" else None
            ),
            "network_used": False,
            "repository_modified": False,
            "idempotent_reuse": False,
        }
        write_file(temporary / "PREPARATION_REPORT.json", json_bytes(report))
        os.rename(temporary, final_dir)
        temporary = None
        return {
            "status": "PASS",
            "state": "PREPARED_INPUTS",
            "preparation_id": preparation_id,
            "preparation_dir": str(final_dir),
            "idempotent_reuse": False,
            "manifest_sha256": sha256_file(final_dir / "PREPARATION_MANIFEST.json"),
            "network_used": False,
            "repository_modified": False,
        }
    finally:
        os.close(lock_fd)
        lock_path.unlink(missing_ok=True)
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)


def write_diagnostic(
    repo_root: Path,
    output_root: Path,
    *,
    spec_path: Path,
    error: PreparationError,
) -> Path | None:
    try:
        repo_root = repo_root.expanduser().resolve()
        output_root = output_root.expanduser().resolve()
        if not outside_repo(repo_root, output_root):
            return None
        directory = output_root / "_diagnostics"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (
            "PREPARATION_FAILURE_"
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            + ".json"
        )
        payload = {
            "schema_version": 1,
            "tool_version": TOOL_VERSION,
            "status": "FAIL",
            "state": "BLOCKED",
            "error_code": error.error_code,
            "phase": error.phase,
            "message": error.message,
            "spec_path": str(spec_path),
            "network_used": False,
            "repository_modified": False,
            "created_at": now(),
        }
        write_file(path, json_bytes(payload))
        return path
    except (OSError, ValueError):
        return None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).expanduser().resolve()
    spec_path = Path(args.spec).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    try:
        result = prepare(repo_root, spec_path, output_root)
        print(
            json.dumps(result, sort_keys=True)
            if args.json
            else f"PASS {result['preparation_id']} {result['preparation_dir']}"
        )
        return 0
    except PreparationError as error:
        diagnostic = write_diagnostic(
            repo_root,
            output_root,
            spec_path=spec_path,
            error=error,
        )
        result = {
            "status": "FAIL",
            "state": "BLOCKED",
            "error_code": error.error_code,
            "phase": error.phase,
            "message": error.message,
            "diagnostic": None if diagnostic is None else str(diagnostic),
            "network_used": False,
            "repository_modified": False,
        }
        print(
            json.dumps(result, sort_keys=True) if args.json else f"FAIL {error}",
            file=sys.stdout if args.json else sys.stderr,
        )
        return 2
    except (OSError, ValueError, yaml.YAMLError) as exc:
        error = PreparationError("INTERNAL_ERROR", "internal", str(exc))
        diagnostic = write_diagnostic(
            repo_root,
            output_root,
            spec_path=spec_path,
            error=error,
        )
        result = {
            "status": "FAIL",
            "state": "BLOCKED",
            "error_code": error.error_code,
            "phase": error.phase,
            "message": error.message,
            "diagnostic": None if diagnostic is None else str(diagnostic),
            "network_used": False,
            "repository_modified": False,
        }
        print(
            json.dumps(result, sort_keys=True) if args.json else f"FAIL {error}",
            file=sys.stdout if args.json else sys.stderr,
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
