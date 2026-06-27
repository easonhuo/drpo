#!/usr/bin/env python3
"""Validate the canonical DRPO governance-pipeline stage closure ledger.

Stage 1 and the current Stage 2 are closed in maintenance-only mode.  Their
owned core files are content-addressed and every accepted after-image must be
bound to an explicit authorization record.  This validator is intentionally
independent of Git history so it can run in update-package worktrees, CI, and
local checkouts before a commit is created.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml


SHA256_RE = re.compile(r"[0-9a-f]{64}")
GIT_SHA_RE = re.compile(r"[0-9a-f]{40}")
AUTH_ID_RE = re.compile(r"GOV-[A-Z0-9][A-Z0-9-]{2,127}")
STAGE_IDS = tuple(f"stage_{index}" for index in range(6))
ALLOWED_STAGE_STATUSES = {
    "closed",
    "closed_maintenance_only",
    "ready_not_started",
    "blocked_by_predecessor",
    "reopened",
    "active",
    "shadow_active",
    "shadow_validated",
}
ALLOWED_AUTH_KINDS = {
    "closure",
    "maintenance",
    "reopen",
    "stage_transition",
    "emergency_security",
}
ALLOWED_CLOSED_CHANGE_CLASSES = {
    "bugfix",
    "security_fix",
    "compatibility_fix",
    "documentation_clarification",
}
PROHIBITED_WITHOUT_REOPEN = {
    "new_feature",
    "architecture_expansion",
    "responsibility_change",
    "default_policy_change",
}
REOPEN_REQUIREMENTS = {
    "explicit_user_approval",
    "registered_governance_claim",
    "authorization_record",
    "rollback_plan",
}
EXPECTED_HISTORICAL_RENUMBERING = {
    "historical_stage_2_handoff_delta": "stage_3",
    "historical_stage_3_handoff_history_split": "stage_4",
    "historical_stage_4_controlled_switch": "stage_5",
}


class StageStatusError(ValueError):
    """Raised when the stage ledger or its authorization chain is invalid."""


def load_mapping(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise StageStatusError(f"Could not read {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise StageStatusError(f"{label} must contain one YAML mapping: {path}")
    return payload


def reject_symlink_components(repo_root: Path, relative: Path, label: str) -> None:
    current = repo_root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise StageStatusError(f"{label} may not contain a symlink: {relative.as_posix()}")


def safe_repo_path(repo_root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise StageStatusError(f"{label} must be a non-empty repository-relative path")
    relative = Path(value)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise StageStatusError(f"{label} is unsafe: {value!r}")
    reject_symlink_components(repo_root, relative, label)
    resolved = (repo_root / relative).resolve()
    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        raise StageStatusError(f"{label} escapes the repository: {value!r}") from exc
    if not resolved.exists() or not resolved.is_file():
        raise StageStatusError(f"{label} does not name a regular file: {value}")
    return resolved


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StageStatusError(f"{label} must be a non-empty string")
    return value.strip()


def require_string_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise StageStatusError(f"{label} must be a non-empty list")
    if not all(isinstance(item, str) and item for item in value):
        raise StageStatusError(f"{label} must contain only non-empty strings")
    if len(set(value)) != len(value):
        raise StageStatusError(f"{label} contains duplicates")
    return value


def load_authorizations(repo_root: Path, relative_dir: Any) -> dict[str, dict[str, Any]]:
    directory_value = require_string(relative_dir, "authorization_directory")
    relative = Path(directory_value)
    if relative.is_absolute() or ".." in relative.parts:
        raise StageStatusError("authorization_directory must be repository-relative")
    reject_symlink_components(repo_root, relative, "authorization_directory")
    directory = (repo_root / relative).resolve()
    try:
        directory.relative_to(repo_root)
    except ValueError as exc:
        raise StageStatusError("authorization_directory escapes the repository") from exc
    if not directory.is_dir():
        raise StageStatusError(f"authorization_directory is missing or unsafe: {directory_value}")

    authorizations: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.glob("*.yaml")):
        if path.is_symlink():
            raise StageStatusError(f"Authorization record may not be a symlink: {path}")
        item = load_mapping(path, "authorization record")
        if item.get("schema_version") != 1:
            raise StageStatusError(f"{path.name}: schema_version must be 1")
        auth_id = require_string(item.get("authorization_id"), f"{path.name} authorization_id")
        if not AUTH_ID_RE.fullmatch(auth_id):
            raise StageStatusError(f"{path.name}: invalid authorization_id {auth_id!r}")
        if auth_id in authorizations:
            raise StageStatusError(f"Duplicate authorization_id: {auth_id}")
        if path.stem != auth_id:
            raise StageStatusError(
                f"Authorization filename must equal authorization_id: {path.name} != {auth_id}.yaml"
            )
        kind = item.get("kind")
        if kind not in ALLOWED_AUTH_KINDS:
            raise StageStatusError(f"{auth_id}: invalid kind {kind!r}")
        change_class = require_string(item.get("change_class"), f"{auth_id} change_class")
        if kind == "maintenance" and change_class not in ALLOWED_CLOSED_CHANGE_CLASSES:
            raise StageStatusError(
                f"{auth_id}: maintenance change_class must be one of "
                f"{sorted(ALLOWED_CLOSED_CHANGE_CLASSES)}"
            )
        if kind == "reopen" and change_class != "reopen":
            raise StageStatusError(f"{auth_id}: reopen authorization must use change_class=reopen")
        if kind == "closure" and change_class != "closure":
            raise StageStatusError(
                f"{auth_id}: closure authorization must use change_class=closure"
            )
        if kind == "stage_transition" and change_class != "stage_transition":
            raise StageStatusError(
                f"{auth_id}: stage_transition authorization must use change_class=stage_transition"
            )
        base_commit = require_string(item.get("base_commit"), f"{auth_id} base_commit")
        if not GIT_SHA_RE.fullmatch(base_commit):
            raise StageStatusError(f"{auth_id}: base_commit must be a full lowercase Git SHA")
        require_string(item.get("claim_id"), f"{auth_id} claim_id")
        require_string(item.get("approval_record"), f"{auth_id} approval_record")
        stage_ids = require_string_list(item.get("stage_ids"), f"{auth_id} stage_ids")
        unknown = sorted(set(stage_ids) - set(STAGE_IDS))
        if unknown:
            raise StageStatusError(f"{auth_id}: unknown stage_ids {unknown}")
        statuses = item.get("authorized_stage_statuses")
        if not isinstance(statuses, dict) or not statuses:
            raise StageStatusError(f"{auth_id}: authorized_stage_statuses must be non-empty")
        for stage_id, status in statuses.items():
            if stage_id not in stage_ids:
                raise StageStatusError(
                    f"{auth_id}: status authorization for undeclared stage {stage_id}"
                )
            if status not in ALLOWED_STAGE_STATUSES:
                raise StageStatusError(f"{auth_id}: invalid authorized status {status!r}")
        hashes = item.get("authorized_file_hashes", {})
        if not isinstance(hashes, dict):
            raise StageStatusError(f"{auth_id}: authorized_file_hashes must be a mapping")
        for file_path, digest in hashes.items():
            safe_repo_path(repo_root, file_path, f"{auth_id} authorized file")
            if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
                raise StageStatusError(f"{auth_id}: invalid SHA-256 for {file_path}")
        authorizations[auth_id] = item
    if not authorizations:
        raise StageStatusError("At least one governance-stage authorization record is required")
    return authorizations


def validate(repo_root: Path, ledger_path: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    ledger = load_mapping(ledger_path, "stage ledger")
    if ledger.get("schema_version") != 1:
        raise StageStatusError("stage ledger schema_version must be 1")
    if ledger.get("policy_id") != "GOV-PIPELINE-STAGE-CLOSURE-01":
        raise StageStatusError("stage ledger policy_id must be GOV-PIPELINE-STAGE-CLOSURE-01")
    if ledger.get("authority") != "governance_status_ledger_not_research_master":
        raise StageStatusError("stage ledger authority boundary is missing")

    renumbering = ledger.get("historical_renumbering")
    if renumbering != EXPECTED_HISTORICAL_RENUMBERING:
        raise StageStatusError("historical_renumbering must preserve the canonical Stage 3/4/5 map")

    change_control = ledger.get("change_control")
    if not isinstance(change_control, dict):
        raise StageStatusError("change_control must be a mapping")
    if (
        set(change_control.get("closed_stage_allowed_change_classes", []))
        != ALLOWED_CLOSED_CHANGE_CLASSES
    ):
        raise StageStatusError("closed-stage allowed change classes changed")
    if set(change_control.get("prohibited_without_reopen", [])) != PROHIBITED_WITHOUT_REOPEN:
        raise StageStatusError("prohibited_without_reopen changed")
    if set(change_control.get("reopen_requires", [])) != REOPEN_REQUIREMENTS:
        raise StageStatusError("reopen requirements changed")
    authorizations = load_authorizations(repo_root, change_control.get("authorization_directory"))

    stages = ledger.get("stages")
    if not isinstance(stages, dict) or set(stages) != set(STAGE_IDS):
        raise StageStatusError(f"stages must exactly contain {list(STAGE_IDS)}")

    protected_paths: dict[str, str] = {}
    stage_reports: dict[str, Any] = {}
    for stage_id in STAGE_IDS:
        stage = stages[stage_id]
        if not isinstance(stage, dict):
            raise StageStatusError(f"{stage_id} must be a mapping")
        if stage.get("number") != int(stage_id.split("_")[1]):
            raise StageStatusError(f"{stage_id}: number field is inconsistent")
        status = stage.get("status")
        if status not in ALLOWED_STAGE_STATUSES:
            raise StageStatusError(f"{stage_id}: invalid status {status!r}")
        require_string(stage.get("name"), f"{stage_id} name")
        require_string(stage.get("responsibility"), f"{stage_id} responsibility")
        status_auth_id = require_string(
            stage.get("status_authorization"), f"{stage_id} status_authorization"
        )
        status_auth = authorizations.get(status_auth_id)
        if status_auth is None:
            raise StageStatusError(f"{stage_id}: unknown status authorization {status_auth_id}")
        if stage_id not in status_auth["stage_ids"]:
            raise StageStatusError(f"{stage_id}: authorization {status_auth_id} omits this stage")
        if status_auth["authorized_stage_statuses"].get(stage_id) != status:
            raise StageStatusError(
                f"{stage_id}: authorization {status_auth_id} does not authorize status {status}"
            )
        if status == "reopened" and status_auth["kind"] != "reopen":
            raise StageStatusError(f"{stage_id}: reopened status requires a reopen authorization")

        protected = stage.get("protected_files", [])
        if stage_id in {"stage_1", "stage_2"}:
            if status not in {"closed_maintenance_only", "reopened"}:
                raise StageStatusError(
                    f"{stage_id}: pipeline core status must be closed_maintenance_only or reopened"
                )
            if status == "reopened" and status_auth["kind"] != "reopen":
                raise StageStatusError(
                    f"{stage_id}: reopened status requires a user-approved reopen authorization"
                )
            if set(stage.get("allowed_change_classes", [])) != ALLOWED_CLOSED_CHANGE_CLASSES:
                raise StageStatusError(f"{stage_id}: allowed_change_classes changed")
            if set(stage.get("prohibited_without_reopen", [])) != PROHIBITED_WITHOUT_REOPEN:
                raise StageStatusError(f"{stage_id}: prohibited_without_reopen changed")
            if set(stage.get("reopen_requires", [])) != REOPEN_REQUIREMENTS:
                raise StageStatusError(f"{stage_id}: reopen requirements changed")
            if not isinstance(protected, list) or not protected:
                raise StageStatusError(f"{stage_id}: protected_files must be non-empty")
        elif protected:
            raise StageStatusError(f"{stage_id}: only Stage 1/2 may own protected_files")

        file_reports: list[dict[str, str]] = []
        for entry in protected:
            if not isinstance(entry, dict):
                raise StageStatusError(f"{stage_id}: protected file entry must be a mapping")
            path_value = require_string(entry.get("path"), f"{stage_id} protected path")
            if path_value in protected_paths:
                raise StageStatusError(
                    f"Protected path {path_value} is owned by both {protected_paths[path_value]} and {stage_id}"
                )
            expected_hash = require_string(entry.get("sha256"), f"{path_value} sha256")
            if not SHA256_RE.fullmatch(expected_hash):
                raise StageStatusError(f"{path_value}: sha256 must be lowercase hex")
            auth_id = require_string(entry.get("authorized_by"), f"{path_value} authorized_by")
            auth = authorizations.get(auth_id)
            if auth is None:
                raise StageStatusError(f"{path_value}: unknown authorization {auth_id}")
            if stage_id not in auth["stage_ids"]:
                raise StageStatusError(f"{path_value}: {auth_id} does not cover {stage_id}")
            if auth["authorized_file_hashes"].get(path_value) != expected_hash:
                raise StageStatusError(
                    f"{path_value}: ledger hash is not bound by authorization {auth_id}"
                )
            path = safe_repo_path(repo_root, path_value, f"{stage_id} protected file")
            actual_hash = sha256(path)
            if actual_hash != expected_hash:
                raise StageStatusError(
                    f"{path_value}: protected file hash changed without authorization; "
                    f"expected {expected_hash}, got {actual_hash}"
                )
            protected_paths[path_value] = stage_id
            file_reports.append(
                {"path": path_value, "sha256": actual_hash, "authorization": auth_id}
            )

        stage_reports[stage_id] = {
            "status": status,
            "status_authorization": status_auth_id,
            "protected_files": file_reports,
        }

    return {
        "matched": True,
        "policy_id": ledger["policy_id"],
        "stages": stage_reports,
        "authorization_count": len(authorizations),
        "protected_file_count": len(protected_paths),
        "stage_1_closed": stages["stage_1"]["status"] == "closed_maintenance_only",
        "stage_2_closed": stages["stage_2"]["status"] == "closed_maintenance_only",
        "stage_3_ready_not_started": stages["stage_3"]["status"] == "ready_not_started",
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    ledger = (args.ledger or repo_root / "docs" / "governance_pipeline_stage_status.yaml").resolve()
    try:
        report = validate(repo_root, ledger)
    except StageStatusError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        current = report["stages"]
        print(
            "Governance pipeline stage status: PASS "
            f"(Stage 1={current['stage_1']['status']}, "
            f"Stage 2={current['stage_2']['status']}, "
            f"Stage 3={current['stage_3']['status']}, "
            f"protected_files={report['protected_file_count']}, "
            f"authorizations={report['authorization_count']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
