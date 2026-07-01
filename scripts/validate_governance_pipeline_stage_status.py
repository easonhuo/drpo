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
ALLOWED_STAGE3_ACCEPTANCE_STATES = {
    "bootstrap_validation_pending",
    "bootstrap_full_passed_shadow_observation_pending",
    "real_shadow_observation_active",
    "critical_mismatch_open",
    "shadow_validation_complete",
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


def validate_stage4a_after_image(repo_root: Path, path_value: str) -> dict[str, Any]:
    path = safe_repo_path(repo_root, path_value, "stage_4 acceptance after-image")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StageStatusError(f"invalid Stage 4A after-image: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise StageStatusError("Stage 4A after-image schema_version must be 1")
    if payload.get("policy_id") != "GOV-HANDOFF-INDEX-01":
        raise StageStatusError("Stage 4A after-image policy_id mismatch")
    if payload.get("authority") != "shadow_only":
        raise StageStatusError("Stage 4A after-image must remain shadow_only")
    if payload.get("base_commit") != "9674cb167080dfdeecb353c9f328ad86b74f87c5":
        raise StageStatusError("Stage 4A after-image base_commit mismatch")
    entries = payload.get("files")
    if not isinstance(entries, list) or not entries:
        raise StageStatusError("Stage 4A after-image files must be non-empty")
    seen: set[str] = set()
    canonical: list[dict[str, str]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise StageStatusError(f"Stage 4A after-image files[{index}] must be a mapping")
        relative = require_string(entry.get("path"), f"Stage 4A after-image files[{index}].path")
        digest = require_string(entry.get("sha256"), f"Stage 4A after-image {relative} sha256")
        if not SHA256_RE.fullmatch(digest):
            raise StageStatusError(f"Stage 4A after-image invalid SHA-256 for {relative}")
        if relative in seen:
            raise StageStatusError(f"Stage 4A after-image duplicate path: {relative}")
        seen.add(relative)
        actual = sha256(safe_repo_path(repo_root, relative, "Stage 4A after-image file"))
        if actual != digest:
            raise StageStatusError(
                f"Stage 4A after-image drift for {relative}; expected {digest}, got {actual}"
            )
        canonical.append({"path": relative, "sha256": digest})
    expected_tree = hashlib.sha256(
        (
            json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
    ).hexdigest()
    if payload.get("tree_hash") != expected_tree:
        raise StageStatusError("Stage 4A after-image tree_hash mismatch")
    if payload.get("file_count") != len(entries):
        raise StageStatusError("Stage 4A after-image file_count mismatch")
    return {"file_count": len(entries), "tree_hash": expected_tree}


def validate_stage4b_after_image(repo_root: Path, path_value: str) -> dict[str, Any]:
    path = safe_repo_path(repo_root, path_value, "stage_4b acceptance after-image")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StageStatusError(f"invalid Stage 4B after-image: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise StageStatusError("Stage 4B after-image schema_version must be 1")
    if (
        payload.get("policy_id") != "GOV-HANDOFF-INDEX-01"
        or payload.get("authority") != "shadow_only"
    ):
        raise StageStatusError("Stage 4B after-image authority/policy mismatch")
    if payload.get("base_commit") != "cf775893b9885ba893278437556abb4d1d5dd1a8":
        raise StageStatusError("Stage 4B after-image base_commit mismatch")
    entries = payload.get("files")
    if not isinstance(entries, list) or not entries:
        raise StageStatusError("Stage 4B after-image files must be non-empty")
    seen = set()
    canonical = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise StageStatusError(f"Stage 4B after-image files[{index}] must be a mapping")
        relative = require_string(entry.get("path"), f"Stage 4B after-image files[{index}].path")
        digest = require_string(entry.get("sha256"), f"Stage 4B after-image {relative} sha256")
        if not SHA256_RE.fullmatch(digest):
            raise StageStatusError(f"Stage 4B after-image invalid SHA-256 for {relative}")
        if relative in seen:
            raise StageStatusError(f"Stage 4B after-image duplicate path: {relative}")
        seen.add(relative)
        actual = sha256(safe_repo_path(repo_root, relative, "Stage 4B after-image file"))
        if actual != digest:
            raise StageStatusError(
                f"Stage 4B after-image drift for {relative}; expected {digest}, got {actual}"
            )
        canonical.append({"path": relative, "sha256": digest})
    expected = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if payload.get("tree_hash") != expected or payload.get("file_count") != len(entries):
        raise StageStatusError("Stage 4B after-image tree/file_count mismatch")
    return {"file_count": len(entries), "tree_hash": expected}


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

        if stage_id == "stage_3" and status == "shadow_active":
            required_stage3_files = {
                "shadow_policy": "docs/handoff_delta_policy.yaml",
                "schema_protocol": "docs/handoff_delta_protocol.md",
                "state_machines": "docs/handoff_delta_state_machines.yaml",
                "renderer": "scripts/handoff_delta_shadow.py",
                "acceptance_entrypoint": "scripts/run_handoff_delta_acceptance.py",
                "bootstrap_delta": (
                    "docs/handoff_deltas/GOV-STAGE3-SHADOW-BOOTSTRAP-2026-06-27/HANDOFF_DELTA.yaml"
                ),
            }
            if stage.get("implementation_state") != "implemented":
                raise StageStatusError(
                    "stage_3 shadow_active requires implementation_state=implemented"
                )
            if stage.get("feature_state") != "feature_frozen_bugfix_only":
                raise StageStatusError("stage_3 feature_state must be feature_frozen_bugfix_only")
            freeze_auth_id = require_string(
                stage.get("freeze_authorization"), "stage_3 freeze_authorization"
            )
            freeze_auth = authorizations.get(freeze_auth_id)
            if freeze_auth is None or freeze_auth.get("kind") != "stage_transition":
                raise StageStatusError(
                    "stage_3 feature freeze requires a stage_transition authorization"
                )
            if freeze_auth["authorized_stage_statuses"].get("stage_3") != "shadow_active":
                raise StageStatusError(
                    "stage_3 feature freeze authorization must preserve shadow_active"
                )
            checkpoint = require_string(
                stage.get("freeze_checkpoint_full_acceptance_report"),
                "stage_3 freeze_checkpoint_full_acceptance_report",
            )
            expected_checkpoint = (
                "docs/handoff_deltas/GOV-STAGE3-FREEZE-STAGE4-DESIGN-2026-06-28/"
                "FULL_ACCEPTANCE_REPORT.json"
            )
            if checkpoint != expected_checkpoint:
                raise StageStatusError(
                    "stage_3 freeze checkpoint must use the registered Full Acceptance report"
                )
            safe_repo_path(repo_root, checkpoint, "stage_3 freeze Full Acceptance report")
            if stage.get("manual_handoff_remains_authoritative") is not True:
                raise StageStatusError(
                    "stage_3 shadow mode must keep the manual handoff authoritative"
                )
            if stage.get("authority_cutover_allowed") is not False:
                raise StageStatusError("stage_3 shadow mode must forbid authority cutover")
            acceptance_state = require_string(
                stage.get("acceptance_state"), "stage_3 acceptance_state"
            )
            if acceptance_state not in ALLOWED_STAGE3_ACCEPTANCE_STATES:
                raise StageStatusError(
                    f"stage_3 acceptance_state is not recognized: {acceptance_state!r}"
                )
            if stage.get("observation_ledger_mode") != "derived_from_delta_reports_and_git_history":
                raise StageStatusError(
                    "stage_3 observation_ledger_mode must be derived_from_delta_reports_and_git_history"
                )
            if stage.get("report_persistence") != "required_and_revalidated":
                raise StageStatusError(
                    "stage_3 report_persistence must be required_and_revalidated"
                )
            if stage.get("full_trigger_enforcement") != "automatic_on_fast_gate":
                raise StageStatusError(
                    "stage_3 full_trigger_enforcement must be automatic_on_fast_gate"
                )
            if stage.get("current_delta_schema_version") != 2:
                raise StageStatusError("stage_3 current_delta_schema_version must be 2")
            if stage.get("candidate_authoritative_delta_schema_version") != 3:
                raise StageStatusError(
                    "stage_3 candidate_authoritative_delta_schema_version must be 3"
                )
            if (
                stage.get("candidate_authority_config")
                != "docs/handoff_versions/AUTHORITY.yaml"
                or stage.get("candidate_authority_engine") != "scripts/handoff_authority.py"
                or stage.get("candidate_implementation_authorization")
                != "GOV-STAGE5-CANDIDATE-IMPLEMENTATION-2026-07-01"
            ):
                raise StageStatusError("stage_3 Stage 5 candidate linkage is invalid")
            safe_repo_path(
                repo_root,
                stage["candidate_authority_config"],
                "stage_3 candidate authority config",
            )
            safe_repo_path(
                repo_root,
                stage["candidate_authority_engine"],
                "stage_3 candidate authority engine",
            )
            audit_entrypoint = require_string(
                stage.get("observation_audit_entrypoint"),
                "stage_3 observation_audit_entrypoint",
            )
            if audit_entrypoint != "scripts/handoff_delta_shadow.py":
                raise StageStatusError(
                    "stage_3 observation_audit_entrypoint must be scripts/handoff_delta_shadow.py"
                )
            safe_repo_path(repo_root, audit_entrypoint, "stage_3 observation_audit_entrypoint")
            for key, expected_path in required_stage3_files.items():
                value = require_string(stage.get(key), f"stage_3 {key}")
                if value != expected_path:
                    raise StageStatusError(
                        f"stage_3 {key} must be {expected_path!r}, got {value!r}"
                    )
                safe_repo_path(repo_root, value, f"stage_3 {key}")

        if stage_id == "stage_4":
            if status != "active":
                raise StageStatusError("stage_4 must remain active after Stage 4B acceptance")
            if stage.get("design_state") != "specification_authorized":
                raise StageStatusError("stage_4 design_state must be specification_authorized")
            design_auth_id = require_string(
                stage.get("design_authorization"), "stage_4 design authorization"
            )
            if design_auth_id != "GOV-STAGE3-FREEZE-STAGE4-DESIGN-2026-06-28":
                raise StageStatusError("stage_4 design_authorization changed")
            expected = "GOV-STAGE4B-FINAL-ACCEPTANCE-2026-07-01"
            if (
                require_string(
                    stage.get("implementation_authorization"),
                    "stage_4 implementation authorization",
                )
                != expected
                or require_string(
                    stage.get("acceptance_authorization"), "stage_4 acceptance authorization"
                )
                != expected
                or status_auth_id != expected
            ):
                raise StageStatusError(
                    "stage_4 status, implementation, and acceptance must use the Stage 4B authorization"
                )
            auth = authorizations.get(expected)
            if (
                auth is None
                or auth.get("kind") != "stage_transition"
                or auth.get("base_commit") != "cf775893b9885ba893278437556abb4d1d5dd1a8"
            ):
                raise StageStatusError("Stage 4B authorization missing or invalid")
            if (
                stage.get("stage4a_implementation_authorization")
                != "GOV-STAGE4A-PARALLEL-IMPLEMENTATION-2026-06-28"
                or stage.get("stage4a_acceptance_authorization")
                != "GOV-STAGE4A-FINAL-ACCEPTANCE-2026-06-30"
            ):
                raise StageStatusError("Stage 4A authorization history changed")
            paths = {
                "semantic_context_spec": "docs/governance_stage4_semantic_context_spec.md",
                "stage4a_acceptance_spec": "docs/governance_stage4a_acceptance_spec.md",
                "stage4a_acceptance_entrypoint": "scripts/run_stage4a_acceptance.py",
                "stage4a_acceptance_report": "docs/governance_stage4a_acceptance/ACCEPTANCE_REPORT.json",
                "stage4a_acceptance_after_image": "docs/governance_stage4a_acceptance/AFTER_IMAGE.json",
                "stage4b_spec": "docs/governance_stage4b_lossless_source_promotion_spec.md",
                "stage4b_config": "docs/handoff_shadow/stage4/candidate/STAGE4B_CONFIG.yaml",
                "stage4b_builder": "scripts/build_stage4b_candidate.py",
                "stage4b_validator": "scripts/validate_stage4b_candidate.py",
                "acceptance_spec": "docs/governance_stage4b_lossless_source_promotion_spec.md",
                "acceptance_entrypoint": "scripts/run_stage4b_acceptance.py",
                "acceptance_report": "docs/governance_stage4b_acceptance/ACCEPTANCE_REPORT.json",
                "acceptance_after_image": "docs/governance_stage4b_acceptance/AFTER_IMAGE.json",
            }
            for key, expected_path in paths.items():
                value = require_string(stage.get(key), f"stage_4 {key}")
                if value != expected_path:
                    raise StageStatusError(f"stage_4 {key} must be {expected_path}")
                safe_repo_path(repo_root, value, f"stage_4 {key}")
            if (
                stage.get("implementation_state") != "stage_4b_accepted"
                or stage.get("implementation_allowed") is not False
            ):
                raise StageStatusError(
                    "stage_4 must record accepted Stage 4B and pause implementation"
                )
            if stage.get("active_phase") != "stage_4b_accepted_waiting_stage_4c_authorization":
                raise StageStatusError("stage_4 must wait for separate Stage 4C authorization")
            if (
                stage.get("feature_state")
                != "stage_4b_feature_frozen_bugfix_compatibility_clarification_only"
            ):
                raise StageStatusError("stage_4b feature state changed")
            if stage.get("depends_on") != ["stage_3_feature_frozen"]:
                raise StageStatusError("stage_4 must depend on Stage 3 feature freeze")
            if (
                stage.get("shadow_candidate_only") is not True
                or stage.get("manual_handoff_remains_authoritative") is not True
                or stage.get("authority_cutover_allowed") is not False
            ):
                raise StageStatusError("stage_4 shadow authority boundary changed")
            if stage.get("phase_plan") != [
                "stage_4a_schema_inventory",
                "stage_4b_lossless_candidate",
                "stage_4c_context_assembly_shadow_validation",
            ]:
                raise StageStatusError(
                    "stage_4 phase_plan must preserve the registered 4A/4B/4C sequence"
                )
            if stage.get("phase_states") != {
                "stage_4a_schema_inventory": "accepted",
                "stage_4b_lossless_candidate": "accepted",
                "stage_4c_context_assembly_shadow_validation": "ready_for_authorization",
            }:
                raise StageStatusError(
                    "stage_4 phase states must accept 4A/4B and only make 4C ready"
                )
            p4a = safe_repo_path(
                repo_root, stage["stage4a_acceptance_report"], "stage_4a acceptance report"
            )
            r4a = json.loads(p4a.read_text())
            if (
                r4a.get("status") != "PASS"
                or r4a.get("evaluated_base_commit") != "9674cb167080dfdeecb353c9f328ad86b74f87c5"
            ):
                raise StageStatusError("Stage 4A acceptance report changed")
            a4a = validate_stage4a_after_image(repo_root, stage["stage4a_acceptance_after_image"])
            if r4a.get("after_image_tree_hash") != a4a["tree_hash"]:
                raise StageStatusError("Stage 4A acceptance report after-image hash mismatch")
            rp = safe_repo_path(repo_root, stage["acceptance_report"], "stage_4b acceptance report")
            report = json.loads(rp.read_text())
            if (
                report.get("status") != "PASS"
                or report.get("policy_id") != "GOV-HANDOFF-INDEX-01"
                or report.get("authority") != "non_authoritative_stage4b_shadow_candidate"
            ):
                raise StageStatusError("Stage 4B acceptance report invalid")
            if (
                report.get("authority_cutover_allowed") is not False
                or report.get("manual_handoff_remains_authoritative") is not True
                or report.get("evaluated_base_commit") != "cf775893b9885ba893278437556abb4d1d5dd1a8"
                or report.get("hard_blockers") != []
                or report.get("stage_4c_state") != "ready_for_authorization"
            ):
                raise StageStatusError("Stage 4B acceptance boundary/state invalid")
            fault = report.get("fault_injection", {})
            if fault.get("status") != "PASS" or fault.get("passed") != fault.get("total"):
                raise StageStatusError("Stage 4B fault injection incomplete")
            if report.get("coverage") != {
                "unmapped_count": 0,
                "multi_owner_conflict_count": 0,
                "unresolved_overlap_count": 0,
                "missing_history_count": 0,
            }:
                raise StageStatusError("Stage 4B coverage blockers are not zero")
            after = validate_stage4b_after_image(repo_root, stage["acceptance_after_image"])
            if report.get("after_image_tree_hash") != after["tree_hash"]:
                raise StageStatusError("Stage 4B acceptance report after-image hash mismatch")

        if stage_id == "stage_5":
            expected_stage5_auth_id = "GOV-STAGE5-CANDIDATE-IMPLEMENTATION-2026-07-01"
            if status_auth_id != expected_stage5_auth_id:
                raise StageStatusError(
                    f"stage_5 status_authorization must be {expected_stage5_auth_id}"
                )
            auth = authorizations.get(expected_stage5_auth_id)
            if (
                auth is None
                or auth.get("kind") != "reopen"
                or auth.get("base_commit")
                != "4ad8b09ca80bc4b98aebffc6540f9be29440ba28"
                or auth.get("claim_id") != "GOV-HANDOFF-AUTHORITY-CUTOVER-01"
            ):
                raise StageStatusError("Stage 5 candidate authorization missing or invalid")
            if status != "active":
                raise StageStatusError("stage_5 candidate implementation must be active")
            if (
                stage.get("implementation_state")
                != "candidate_implemented_pending_pre_cutover_acceptance"
                or stage.get("implementation_allowed") is not False
                or stage.get("candidate_only") is not True
            ):
                raise StageStatusError("stage_5 candidate implementation boundary is invalid")
            expected_paths = {
                "candidate_spec": "docs/governance_stage5_versioned_handoff_spec.md",
                "candidate_authority_config": "docs/handoff_versions/AUTHORITY.yaml",
                "candidate_engine": "scripts/handoff_authority.py",
            }
            for key, expected_path in expected_paths.items():
                value = require_string(stage.get(key), f"stage_5 {key}")
                if value != expected_path:
                    raise StageStatusError(f"stage_5 {key} must be {expected_path}")
                safe_repo_path(repo_root, value, f"stage_5 {key}")
            if stage.get("candidate_authorization") != expected_stage5_auth_id:
                raise StageStatusError("stage_5 candidate_authorization mismatch")
            checkpoint_report_value = require_string(
                stage.get("stage3_pre_stage5_full_acceptance_report"),
                "stage_5 Stage 3 pre-Stage 5 Full Acceptance report",
            )
            expected_checkpoint_report = (
                "docs/handoff_deltas/"
                "GOV-STAGE3-PRE-STAGE5-FULL-CHECKPOINT-2026-07-01/"
                "FULL_ACCEPTANCE_REPORT.json"
            )
            if checkpoint_report_value != expected_checkpoint_report:
                raise StageStatusError("stage_5 pre-Stage 5 Full Acceptance report changed")
            checkpoint_report_path = safe_repo_path(
                repo_root,
                checkpoint_report_value,
                "stage_5 pre-Stage 5 Full Acceptance report",
            )
            checkpoint_report = json.loads(checkpoint_report_path.read_text(encoding="utf-8"))
            coverage = checkpoint_report.get("coverage", {})
            if (
                checkpoint_report.get("status") != "PASS"
                or checkpoint_report.get("tier") != "full"
                or coverage.get("successful_real_observation_count") != 17
                or len(coverage.get("covered_update_ids", [])) != 17
            ):
                raise StageStatusError("stage_5 requires the current 17-observation Full Acceptance")
            if stage.get("current_write_authority") != "manual_handoff":
                raise StageStatusError("stage_5 candidate must preserve manual handoff write authority")
            if stage.get("authority_cutover_allowed") is not False:
                raise StageStatusError("stage_5 candidate must forbid authority cutover")
            if stage.get("cutover_requires_independent_user_authorization") is not True:
                raise StageStatusError(
                    "stage_5 cutover must require independent user authorization"
                )
            expected_stage5_dependencies = [
                "stage_3_shadow_validation",
                "stage_4_lossless_validation",
            ]
            if stage.get("depends_on") != expected_stage5_dependencies:
                raise StageStatusError("stage_5 must preserve Stage 3/4 validation dependencies")
            authority_config = load_mapping(
                repo_root / "docs/handoff_versions/AUTHORITY.yaml",
                "Stage 5 authority config",
            )
            if (
                authority_config.get("schema_version") != 1
                or authority_config.get("policy_id")
                != "GOV-HANDOFF-AUTHORITY-CUTOVER-01"
                or authority_config.get("mode") != "manual"
                or authority_config.get("read_master") != "docs/handoff.md"
                or authority_config.get("registry_write_authority")
                != "experiments/registry.yaml"
            ):
                raise StageStatusError("Stage 5 authority config must remain manual candidate mode")
            delta_authority = authority_config.get("delta_authority", {})
            safety = authority_config.get("safety", {})
            generated = authority_config.get("generated_views", {})
            if (
                delta_authority.get("checkpoint_manifest") is not None
                or delta_authority.get("activation_parent_commit") is not None
                or delta_authority.get("current_schema_version") != 3
                or delta_authority.get("maximum_deltas_per_update") != 1
                or generated.get("stage4a_minimal_refresh") is not False
                or safety.get("direct_handoff_edit_forbidden") is not False
                or safety.get("authority_transition_requires_explicit_flag") is not True
            ):
                raise StageStatusError("Stage 5 candidate config illegally activates cutover behavior")

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
        "stage_3_shadow_active": stages["stage_3"]["status"] == "shadow_active",
        "stage_4_active": stages["stage_4"]["status"] == "active",
        "stage_5_blocked": False,
        "stage_5_candidate_active": stages["stage_5"]["status"] == "active",
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
            f"Stage 4={current['stage_4']['status']}, "
            f"Stage 5={current['stage_5']['status']}, "
            f"protected_files={report['protected_file_count']}, "
            f"authorizations={report['authorization_count']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
