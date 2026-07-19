#!/usr/bin/env python3
"""Trusted Stage 5 versioned-handoff materializer and validator.

The current repository remains in manual mode.  This tool implements the future
schema-v3 delta authority behind an explicit authority file and is invoked by
``drpo-update`` from the pre-integration current-main checkout.  It deliberately
reuses the Stage 3 renderer core rather than creating a second document engine.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import yaml

import handoff_delta_shadow as shadow

AUTHORITY_PATH = Path("docs/handoff_versions/AUTHORITY.yaml")
STAGE_LEDGER_PATH = Path("docs/governance_pipeline_stage_status.yaml")
HANDOFF_PATH = Path("docs/handoff.md")
REGISTRY_PATH = Path("experiments/registry.yaml")
DELTA_ROOT = Path("docs/handoff_deltas")
DELTA_FILENAME = "HANDOFF_DELTA.yaml"
REPORT_FILENAME = "MATERIALIZATION_REPORT.json"
CHECKPOINT_ROOT = Path("docs/handoff_versions/checkpoints")
ROLLBACK_ROOT = Path("docs/handoff_versions/rollbacks")
CHECKPOINT_MANIFEST_FILENAME = "CHECKPOINT_MANIFEST.json"
CHECKPOINT_AUDIT_FILENAME = "STAGE4B_CUTOVER_AUDIT.json"
ROLLBACK_REPORT_FILENAME = "ROLLBACK_REPORT.json"
CHECKPOINT_MANIFEST_SCHEMA_VERSION = 2
CHECKPOINT_AUDIT_SCHEMA_VERSION = 1
ROLLBACK_REPORT_SCHEMA_VERSION = 1
DEFAULT_STAGE3_FULL_ACCEPTANCE_REPORT = Path(
    "docs/handoff_deltas/GOV-STAGE5-PRE-CUTOVER-ACCEPTANCE-CLOSURE-2026-07-02/"
    "FULL_ACCEPTANCE_REPORT.json"
)
DEFAULT_STAGE4B_ACCEPTANCE_REPORT = Path(
    "docs/governance_stage4b_acceptance/ACCEPTANCE_REPORT.json"
)
DEFAULT_STAGE4B_AFTER_IMAGE = Path(
    "docs/governance_stage4b_acceptance/AFTER_IMAGE.json"
)
SOURCE_AUTHORED_REPORT_NAMES = {
    REPORT_FILENAME,
    "SHADOW_REPORT.json",
    "FULL_ACCEPTANCE_REPORT.json",
}
SCHEMA_VERSION = 3
MARKER_TOKEN_RE = re.compile(r"<!--\s*/?HANDOFF-DELTA-BLOCK\b")
MARKER_ID_RE = re.compile(
    r"<!--\s*HANDOFF-DELTA-BLOCK\s+location=[^\s]+\s+id=([^\s]+)\s*-->"
)

# One historical schema-v3 delta was committed without a materialization report
# and declared an unchanged candidate hash even though replaying its operation
# would change the handoff.  The repository never materialized that operation:
# the introducing commit changed only the delta file and preserved handoff bytes.
# Keep the compatibility boundary content-addressed and fail closed for every
# other missing report or any mutation of this exact historical anomaly.
LEGACY_INERT_V3_DELTAS = {
    (
        "docs/handoff_deltas/"
        "EXT-H-E7-SQEXP-GAE-FROZEN-DIAGNOSTIC-2026-07-18/"
        "HANDOFF_DELTA.yaml"
    ): {
        "update_id": "EXT-H-E7-SQEXP-GAE-FROZEN-DIAGNOSTIC-2026-07-18",
        "delta_sha256": "cf9da4d13d9d6521a578165e37c099803b5498b24e9b2235820fb130078fd4ae",
        "first_add_commit": "11992ca5de7f2c4a3837cf32aa4e23696ec18ef3",
        "first_add_parent": "d07964c95a8faa8d2b53c36ec85de84e8c6f2385",
        "integration_commit": "cd770f47b89f8971923945c19caec49720c0e139",
        "integration_parent": "bb637503e1289f24f7a28e587f50665afb20e0de",
        "source_base_commit": "bb637503e1289f24f7a28e587f50665afb20e0de",
        "historical_handoff_sha256": (
            "f8ff67ab71c0f53b21fc96967a13aa3e5b8500e42d25464e378df23e1f62c4e8"
        ),
        "declared_candidate_sha256": (
            "f8ff67ab71c0f53b21fc96967a13aa3e5b8500e42d25464e378df23e1f62c4e8"
        ),
        "rendered_candidate_sha256": (
            "0ee60d4003e33e53685a0ead28695938c26b62c371c9cf06554ff36da6d7ed7a"
        ),
    }
}

CONTROL_PLANE_EXACT = {
    "AGENTS.md",
    "docs/governance_stage5_versioned_handoff_spec.md",
    "docs/governance_pipeline_stage_status.yaml",
    "docs/handoff_delta_policy.yaml",
    "docs/handoff_delta_protocol.md",
    "docs/handoff_delta_state_machines.yaml",
    AUTHORITY_PATH.as_posix(),
    "scripts/handoff_delta_shadow.py",
    "scripts/handoff_authority.py",
    "scripts/run_handoff_delta_acceptance.py",
    "scripts/validate_governance_pipeline_stage_status.py",
    "scripts/build_stage4_context.py",
    "scripts/package_update.py",
    "scripts/verify_update_package.py",
    "scripts/create_update_git_bundle.py",
    "scripts/verify_update_git_bundle.py",
    "scripts/select_update_tests.py",
    "docs/handoff_shadow/stage4/minimal/MODULES.yaml",
    "docs/handoff_shadow/stage4/minimal/DEPENDENCIES.yaml",
}
CONTROL_PLANE_PREFIXES = (
    "docs/handoff_versions/",
    "docs/governance_stage_authorizations/",
    "docs/governance_stage4",
    "docs/handoff_shadow/stage4/minimal/",
    "docs/handoff_shadow/stage4/schema/",
    "scripts/build_stage4",
    "scripts/validate_stage4",
    "scripts/validate_stage4a",
    "tools/drpo-update/",
)
DYNAMIC_STAGE4A_PREFIX = "docs/handoff_shadow/stage4/minimal/generated/"


class HandoffAuthorityError(ValueError):
    """Raised when authority validation or normalization fails closed."""


@dataclass(frozen=True)
class ExactIntent:
    delta: dict[str, Any]
    candidate: str
    registry_report: dict[str, Any]


@dataclass(frozen=True)
class DiscoveredDelta:
    integration_commit: str
    path: Path
    delta: dict[str, Any]
    report: dict[str, Any]
    legacy_inert: bool = False


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise HandoffAuthorityError(f"{label} must be a mapping")
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise HandoffAuthorityError(f"{label} must be a list")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HandoffAuthorityError(f"{label} must be a non-empty string")
    return value.strip()


def _reject_unknown(mapping: dict[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise HandoffAuthorityError(f"{label} contains unknown keys: {unknown}")


def _load_yaml(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink():
        raise HandoffAuthorityError(f"{label} may not be a symlink: {path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise HandoffAuthorityError(f"Could not read {label} {path}: {exc}") from exc
    return _mapping(payload, label)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink():
        raise HandoffAuthorityError(f"{label} may not be a symlink: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HandoffAuthorityError(f"Could not read {label} {path}: {exc}") from exc
    return _mapping(payload, label)


def _utc_timestamp(value: str | None = None) -> str:
    if value is None:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise HandoffAuthorityError("timestamp must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise HandoffAuthorityError("timestamp must include a timezone")
    return parsed.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def _require_clean_worktree(repo_root: Path, label: str) -> None:
    status = _git_text(repo_root, "status", "--porcelain")
    if status:
        raise HandoffAuthorityError(f"{label} requires a clean Git worktree")


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    temp.replace(path)


def _repo_relative(repo_root: Path, path: Path, label: str) -> str:
    root = repo_root.resolve()
    resolved = path.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise HandoffAuthorityError(f"{label} escapes repository: {path}") from exc


def _safe_path(repo_root: Path, relative: str, label: str, *, must_exist: bool = True) -> Path:
    rel = Path(relative)
    if rel.is_absolute() or not rel.parts or ".." in rel.parts:
        raise HandoffAuthorityError(f"{label} is not repository-relative: {relative!r}")
    current = repo_root
    for part in rel.parts:
        current = current / part
        if current.is_symlink():
            raise HandoffAuthorityError(f"{label} may not contain a symlink: {relative!r}")
    path = (repo_root / rel).resolve()
    try:
        path.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise HandoffAuthorityError(f"{label} escapes repository: {relative!r}") from exc
    if must_exist and not path.is_file():
        raise HandoffAuthorityError(f"{label} does not exist: {relative}")
    return path


def _git(repo_root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and proc.returncode != 0:
        raise HandoffAuthorityError(
            f"git {' '.join(args)} failed: {(proc.stderr or proc.stdout).strip()}"
        )
    return proc


def _git_text(repo_root: Path, *args: str) -> str:
    return _git(repo_root, *args).stdout.strip()


def _git_show(repo_root: Path, commit: str, relative: Path) -> str:
    proc = _git(repo_root, "show", f"{commit}:{relative.as_posix()}")
    return proc.stdout


def _worktree_changed_paths(repo_root: Path) -> set[str]:
    """Return tracked, staged, and untracked paths changed from ``HEAD``."""

    changed: set[str] = set()
    for args in (("diff", "--name-only"), ("diff", "--cached", "--name-only")):
        changed.update(line for line in _git_text(repo_root, *args).splitlines() if line.strip())
    changed.update(
        line
        for line in _git_text(repo_root, "ls-files", "--others", "--exclude-standard").splitlines()
        if line.strip()
    )
    return changed


def _verify_stage4a_current_source(repo_root: Path) -> None:
    builder = repo_root / "scripts/build_stage4_context.py"
    proc = subprocess.run(
        [sys.executable, str(builder), "--repo-root", str(repo_root), "--json", "check"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise HandoffAuthorityError(
            "Stage 4A minimal views are stale: " + (proc.stderr or proc.stdout).strip()
        )


def load_authority(repo_root: Path) -> dict[str, Any]:
    path = repo_root / AUTHORITY_PATH
    payload = _load_yaml(path, "handoff authority")
    _reject_unknown(
        payload,
        {
            "schema_version",
            "policy_id",
            "mode",
            "read_master",
            "registry_write_authority",
            "delta_authority",
            "generated_views",
            "safety",
        },
        "authority",
    )
    if payload.get("schema_version") != 1:
        raise HandoffAuthorityError("authority.schema_version must be 1")
    if payload.get("policy_id") != "GOV-HANDOFF-AUTHORITY-CUTOVER-01":
        raise HandoffAuthorityError("authority.policy_id mismatch")
    mode = payload.get("mode")
    if mode not in {"manual", "delta"}:
        raise HandoffAuthorityError("authority.mode must be manual or delta")
    if payload.get("read_master") != HANDOFF_PATH.as_posix():
        raise HandoffAuthorityError("authority.read_master must remain docs/handoff.md")
    if payload.get("registry_write_authority") != REGISTRY_PATH.as_posix():
        raise HandoffAuthorityError("registry authority may not change in Stage 5 v1")

    delta = _mapping(payload.get("delta_authority"), "authority.delta_authority")
    _reject_unknown(
        delta,
        {
            "checkpoint_manifest",
            "activation_parent_commit",
            "current_schema_version",
            "report_filename",
            "maximum_deltas_per_update",
        },
        "authority.delta_authority",
    )
    if delta.get("current_schema_version") != SCHEMA_VERSION:
        raise HandoffAuthorityError("authority current schema version must be 3")
    if delta.get("report_filename") != REPORT_FILENAME:
        raise HandoffAuthorityError("authority report filename mismatch")
    if delta.get("maximum_deltas_per_update") != 1:
        raise HandoffAuthorityError("Stage 5 v1 permits one authoritative delta per update")

    generated = _mapping(payload.get("generated_views"), "authority.generated_views")
    _reject_unknown(
        generated,
        {"stage4a_minimal_refresh", "stage4a_output_root"},
        "authority.generated_views",
    )
    if generated.get("stage4a_output_root") != DYNAMIC_STAGE4A_PREFIX.rstrip("/"):
        raise HandoffAuthorityError("authority Stage 4A output root mismatch")

    safety = _mapping(payload.get("safety"), "authority.safety")
    _reject_unknown(
        safety,
        {
            "direct_handoff_edit_forbidden",
            "immutable_checkpoint",
            "immutable_accepted_deltas",
            "trusted_normalizer_from_current_main",
            "authority_transition_requires_explicit_flag",
        },
        "authority.safety",
    )
    required_true = {
        "immutable_checkpoint",
        "immutable_accepted_deltas",
        "trusted_normalizer_from_current_main",
        "authority_transition_requires_explicit_flag",
    }
    for key in required_true:
        if safety.get(key) is not True:
            raise HandoffAuthorityError(f"authority.safety.{key} must be true")

    if mode == "manual":
        if delta.get("checkpoint_manifest") is not None or delta.get("activation_parent_commit") is not None:
            raise HandoffAuthorityError("manual mode must not bind an active checkpoint")
        if generated.get("stage4a_minimal_refresh") is not False:
            raise HandoffAuthorityError("manual mode must not enable Stage 4A normalization")
        if safety.get("direct_handoff_edit_forbidden") is not False:
            raise HandoffAuthorityError("manual mode must permit the current direct handoff workflow")
    else:
        checkpoint = _string(delta.get("checkpoint_manifest"), "checkpoint_manifest")
        activation = _string(delta.get("activation_parent_commit"), "activation_parent_commit")
        if not shadow.GIT_SHA_RE.fullmatch(activation):
            raise HandoffAuthorityError("activation_parent_commit must be a full lowercase Git SHA")
        _safe_path(repo_root, checkpoint, "checkpoint manifest")
        if generated.get("stage4a_minimal_refresh") is not True:
            raise HandoffAuthorityError("delta mode must refresh Stage 4A minimal views")
        if safety.get("direct_handoff_edit_forbidden") is not True:
            raise HandoffAuthorityError("delta mode must forbid direct handoff edits")
    return payload


def _validate_operation(raw: Any, index: int) -> dict[str, Any]:
    op = _mapping(raw, f"operations[{index}]")
    op_type = _string(op.get("op"), f"operations[{index}].op")
    if op_type not in shadow.SUPPORTED_OPERATIONS:
        raise HandoffAuthorityError(f"Unsupported operation: {op_type}")
    operation_id = _string(op.get("operation_id"), f"operations[{index}].operation_id")
    if not shadow.OP_ID_RE.fullmatch(operation_id):
        raise HandoffAuthorityError(f"Invalid operation_id: {operation_id!r}")
    path = _list(op.get("heading_path"), f"operations[{index}].heading_path")
    if not path or not all(isinstance(item, str) and item for item in path):
        raise HandoffAuthorityError("heading_path must contain non-empty strings")
    common = {"operation_id", "op", "heading_path"}
    if op_type == "replace_heading":
        _reject_unknown(op, common | {"new_heading", "reason"}, f"operations[{index}]")
        _string(op.get("new_heading"), f"operations[{index}].new_heading")
        _string(op.get("reason"), f"operations[{index}].reason")
    else:
        _reject_unknown(op, common | {"block_id", "content"}, f"operations[{index}]")
        block_id = _string(op.get("block_id"), f"operations[{index}].block_id")
        if not shadow.BLOCK_ID_RE.fullmatch(block_id):
            raise HandoffAuthorityError(f"Invalid block_id: {block_id!r}")
        content = _string(op.get("content"), f"operations[{index}].content")
        if MARKER_TOKEN_RE.search(content):
            raise HandoffAuthorityError("delta content may not forge HANDOFF-DELTA-BLOCK markers")
    return op


def validate_v3_delta(delta: dict[str, Any], delta_path: Path) -> None:
    _reject_unknown(
        delta,
        {
            "schema_version",
            "update_id",
            "mode",
            "base",
            "renderer_version",
            "operations",
            "registry",
            "expected",
        },
        "delta",
    )
    if delta.get("schema_version") != SCHEMA_VERSION:
        raise HandoffAuthorityError("authoritative delta.schema_version must be 3")
    update_id = _string(delta.get("update_id"), "delta.update_id")
    if not shadow.UPDATE_ID_RE.fullmatch(update_id):
        raise HandoffAuthorityError(f"Invalid update_id: {update_id!r}")
    if delta_path.name != DELTA_FILENAME or delta_path.parent.name != update_id:
        raise HandoffAuthorityError(
            f"Delta path must be docs/handoff_deltas/{update_id}/{DELTA_FILENAME}"
        )
    if delta.get("mode") != "authoritative":
        raise HandoffAuthorityError("schema-v3 delta.mode must be authoritative")
    if delta.get("renderer_version") != 1:
        raise HandoffAuthorityError("schema-v3 renderer_version must remain 1")

    base = _mapping(delta.get("base"), "delta.base")
    _reject_unknown(base, {"commit", "handoff_sha256", "registry_sha256"}, "delta.base")
    commit = _string(base.get("commit"), "delta.base.commit")
    if not shadow.GIT_SHA_RE.fullmatch(commit):
        raise HandoffAuthorityError("delta.base.commit must be a full lowercase Git SHA")
    for key in ("handoff_sha256", "registry_sha256"):
        value = _string(base.get(key), f"delta.base.{key}")
        if not shadow.SHA256_RE.fullmatch(value):
            raise HandoffAuthorityError(f"delta.base.{key} must be lowercase SHA-256")

    operations = _list(delta.get("operations"), "delta.operations")
    operation_ids: set[str] = set()
    block_ids: set[str] = set()
    for index, raw in enumerate(operations):
        op = _validate_operation(raw, index)
        if op["operation_id"] in operation_ids:
            raise HandoffAuthorityError(f"Duplicate operation_id: {op['operation_id']}")
        operation_ids.add(op["operation_id"])
        if "block_id" in op:
            if op["block_id"] in block_ids:
                raise HandoffAuthorityError(f"Duplicate block_id: {op['block_id']}")
            block_ids.add(op["block_id"])

    registry = _mapping(delta.get("registry"), "delta.registry")
    _reject_unknown(
        registry,
        {"mode", "exact_base_after_sha256", "changes"},
        "delta.registry",
    )
    registry_mode = registry.get("mode")
    if registry_mode not in {"unchanged", "expected_after"}:
        raise HandoffAuthorityError("delta.registry.mode must be unchanged or expected_after")
    changes = _list(registry.get("changes", []), "delta.registry.changes")
    exact_after = registry.get("exact_base_after_sha256")
    if registry_mode == "unchanged":
        if exact_after is not None or changes:
            raise HandoffAuthorityError("unchanged registry mode forbids after hash and changes")
    else:
        exact_after_text = _string(exact_after, "delta.registry.exact_base_after_sha256")
        if not shadow.SHA256_RE.fullmatch(exact_after_text):
            raise HandoffAuthorityError("registry exact-base after hash must be lowercase SHA-256")
        if not changes:
            raise HandoffAuthorityError("expected_after registry mode requires changes")
        seen: set[str] = set()
        for index, raw in enumerate(changes):
            item = _mapping(raw, f"registry.changes[{index}]")
            try:
                shadow.validate_registry_change_shape(item, index)
            except shadow.HandoffDeltaError as exc:
                raise HandoffAuthorityError(str(exc)) from exc
            if item["change_id"] in seen:
                raise HandoffAuthorityError(f"Duplicate registry change_id: {item['change_id']}")
            seen.add(item["change_id"])

    if not operations and not changes:
        raise HandoffAuthorityError("schema-v3 delta must change handoff or registry")

    expected = _mapping(delta.get("expected"), "delta.expected")
    _reject_unknown(expected, {"exact_base_candidate_sha256"}, "delta.expected")
    candidate_hash = _string(
        expected.get("exact_base_candidate_sha256"),
        "delta.expected.exact_base_candidate_sha256",
    )
    if not shadow.SHA256_RE.fullmatch(candidate_hash):
        raise HandoffAuthorityError("exact-base candidate hash must be lowercase SHA-256")


def _heading_path_counts(text: str) -> dict[tuple[str, ...], int]:
    counts: dict[tuple[str, ...], int] = {}
    for heading in shadow.parse_headings(text):
        counts[heading.path] = counts.get(heading.path, 0) + 1
    return counts


def _new_duplicate_heading_paths(before: str, after: str) -> list[tuple[str, ...]]:
    before_counts = _heading_path_counts(before)
    after_counts = _heading_path_counts(after)
    return sorted(
        path
        for path, count in after_counts.items()
        if count > 1 and count > before_counts.get(path, 0)
    )


def _existing_block_ids(text: str) -> set[str]:
    return set(MARKER_ID_RE.findall(text))


def _proxy_delta_for_registry(
    delta: dict[str, Any], *, expected_after_sha256: str
) -> dict[str, Any]:
    registry = delta["registry"]
    return {
        "schema_version": 2,
        "base": delta["base"],
        "registry": {
            "mode": registry["mode"],
            "expected_after_sha256": (
                expected_after_sha256 if registry["mode"] == "expected_after" else None
            ),
            "changes": registry.get("changes", []),
        },
    }


def validate_exact_base_intent(
    repo_root: Path,
    delta_path: Path,
    *,
    source_patch_commit: str | None = None,
) -> ExactIntent:
    delta = _load_yaml(delta_path, "schema-v3 delta")
    validate_v3_delta(delta, delta_path)
    base = delta["base"]["commit"]
    _git(repo_root, "cat-file", "-e", f"{base}^{{commit}}")
    base_handoff = _git_show(repo_root, base, HANDOFF_PATH)
    base_registry = _git_show(repo_root, base, REGISTRY_PATH)
    if shadow.sha256_text(base_handoff) != delta["base"]["handoff_sha256"]:
        raise HandoffAuthorityError("Base handoff SHA-256 does not match schema-v3 delta")
    if shadow.sha256_text(base_registry) != delta["base"]["registry_sha256"]:
        raise HandoffAuthorityError("Base registry SHA-256 does not match schema-v3 delta")

    try:
        rendered = shadow.render(base_handoff, delta["operations"])
        replayed = shadow.render(rendered.text, delta["operations"])
        shadow.verify_history_preservation(base_handoff, rendered.text, delta["operations"])
    except shadow.HandoffDeltaError as exc:
        raise HandoffAuthorityError(str(exc)) from exc
    if replayed.text != rendered.text:
        raise HandoffAuthorityError("schema-v3 exact-base replay is not idempotent")
    if _new_duplicate_heading_paths(base_handoff, rendered.text):
        raise HandoffAuthorityError("schema-v3 delta creates duplicate full heading paths")
    candidate_hash = shadow.sha256_text(rendered.text)
    if candidate_hash != delta["expected"]["exact_base_candidate_sha256"]:
        raise HandoffAuthorityError("schema-v3 exact-base candidate SHA-256 mismatch")

    if source_patch_commit:
        source_registry = _git_show(repo_root, source_patch_commit, REGISTRY_PATH)
    else:
        source_registry = (repo_root / REGISTRY_PATH).read_text(encoding="utf-8")
    registry = delta["registry"]
    if registry["mode"] == "expected_after":
        if shadow.sha256_text(source_registry) != registry["exact_base_after_sha256"]:
            raise HandoffAuthorityError("schema-v3 exact-base registry after SHA-256 mismatch")
    proxy = _proxy_delta_for_registry(
        delta,
        expected_after_sha256=shadow.sha256_text(source_registry),
    )
    try:
        registry_report = shadow.validate_registry(
            repo_root, proxy, base_registry, source_registry
        )
    except shadow.HandoffDeltaError as exc:
        raise HandoffAuthorityError(str(exc)) from exc
    return ExactIntent(delta=delta, candidate=rendered.text, registry_report=registry_report)


def _source_changed_paths(
    repo_root: Path,
    *,
    source_base: str,
    source_patch_commit: str,
) -> dict[str, str]:
    proc = _git(
        repo_root,
        "diff",
        "--name-status",
        "--find-renames",
        f"{source_base}..{source_patch_commit}",
    )
    result: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        status = parts[0]
        path = parts[-1]
        result[path] = status
    return result


def _is_control_plane(path: str) -> bool:
    return path in CONTROL_PLANE_EXACT or any(path.startswith(prefix) for prefix in CONTROL_PLANE_PREFIXES)


def _find_new_v3_delta(repo_root: Path, changed: dict[str, str]) -> Path:
    candidates = []
    for path, status in changed.items():
        if not path.startswith(f"{DELTA_ROOT.as_posix()}/") or not path.endswith(
            f"/{DELTA_FILENAME}"
        ):
            continue
        if not status.startswith("A"):
            raise HandoffAuthorityError("accepted authoritative deltas are immutable")
        candidate = repo_root / path
        payload = _load_yaml(candidate, "new handoff delta")
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise HandoffAuthorityError(
                "delta authority accepts only newly added schema-v3 deltas"
            )
        candidates.append(candidate)
    if len(candidates) != 1:
        raise HandoffAuthorityError(
            f"delta mode requires exactly one newly added schema-v3 delta; found {len(candidates)}"
        )
    return candidates[0]


def _hash_tree(root: Path) -> dict[str, str]:
    if not root.exists():
        return {}
    return {
        path.relative_to(root).as_posix(): shadow.sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _run_stage4a_refresh(target_repo: Path, trusted_repo: Path) -> tuple[list[str], list[str]]:
    output_root = target_repo / DYNAMIC_STAGE4A_PREFIX.rstrip("/")
    before = _hash_tree(output_root)
    builder = trusted_repo / "scripts/build_stage4_context.py"
    command = [
        sys.executable,
        str(builder),
        "--repo-root",
        str(target_repo),
        "--json",
        "build",
    ]
    proc = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise HandoffAuthorityError(
            "Stage 4A minimal refresh failed: " + (proc.stderr or proc.stdout).strip()
        )
    check = subprocess.run(
        [
            sys.executable,
            str(builder),
            "--repo-root",
            str(target_repo),
            "--json",
            "check",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check.returncode != 0:
        raise HandoffAuthorityError(
            "Stage 4A minimal post-refresh check failed: "
            + (check.stderr or check.stdout).strip()
        )
    after = _hash_tree(output_root)
    refreshed = sorted(
        key for key in set(before) | set(after) if before.get(key) != after.get(key)
    )
    reused = sorted(key for key in set(before) & set(after) if before[key] == after[key])
    return refreshed, reused


def _validate_stage3_full_acceptance_report(
    repo_root: Path, relative: str
) -> tuple[Path, dict[str, Any]]:
    path = _safe_path(repo_root, relative, "Stage 3 Full Acceptance report")
    report = _load_json(path, "Stage 3 Full Acceptance report")
    coverage = _mapping(report.get("coverage"), "Stage 3 Full Acceptance coverage")
    covered = _list(
        coverage.get("covered_update_ids"),
        "Stage 3 Full Acceptance covered_update_ids",
    )
    count = coverage.get("successful_real_observation_count")
    covered_are_strings = all(isinstance(item, str) and item for item in covered)
    if (
        report.get("status") != "PASS"
        or report.get("tier") != "full"
        or report.get("policy_id") != "GOV-HANDOFF-INDEX-01"
        or not isinstance(count, int)
        or count <= 0
        or not covered_are_strings
        or len(covered) != count
        or len(set(covered)) != len(covered)
    ):
        raise HandoffAuthorityError("Stage 3 Full Acceptance report is not a passing full report")
    return path, report


def _source_parent_real_update_ids(repo_root: Path, source_parent: str) -> list[str]:
    listing = _git(
        repo_root,
        "ls-tree",
        "-r",
        "--name-only",
        source_parent,
        "--",
        DELTA_ROOT.as_posix(),
    ).stdout.splitlines()
    update_ids: list[str] = []
    for relative in sorted(listing):
        if not relative.endswith(f"/{DELTA_FILENAME}"):
            continue
        try:
            payload = yaml.safe_load(_git_show(repo_root, source_parent, Path(relative)))
        except yaml.YAMLError as exc:
            raise HandoffAuthorityError(
                f"source-parent handoff delta is invalid YAML: {relative}"
            ) from exc
        mapping = _mapping(payload, f"source-parent handoff delta {relative}")
        update_id = _string(mapping.get("update_id"), f"source-parent update_id {relative}")
        if shadow.classify_observation(update_id) == "real":
            update_ids.append(update_id)
    if len(update_ids) != len(set(update_ids)):
        raise HandoffAuthorityError("source parent contains duplicate real handoff update IDs")
    return sorted(update_ids)


def _validate_stage3_full_acceptance_covers_source_parent(
    repo_root: Path,
    *,
    source_parent: str,
    report: dict[str, Any],
) -> list[str]:
    covered = _list(
        _mapping(report.get("coverage"), "Stage 3 Full Acceptance coverage").get(
            "covered_update_ids"
        ),
        "Stage 3 Full Acceptance covered_update_ids",
    )
    covered_ids = [_string(value, "Stage 3 covered update ID") for value in covered]
    expected_ids = _source_parent_real_update_ids(repo_root, source_parent)
    if covered_ids != expected_ids:
        missing = sorted(set(expected_ids) - set(covered_ids))
        extra = sorted(set(covered_ids) - set(expected_ids))
        raise HandoffAuthorityError(
            "Stage 3 Full Acceptance report does not cover all real observations "
            f"at the cutover source parent (missing={missing}, extra={extra})"
        )
    return expected_ids


def _validate_stage4b_acceptance_report(
    repo_root: Path, relative: str
) -> tuple[Path, dict[str, Any], Path, dict[str, Any]]:
    report_path = _safe_path(repo_root, relative, "Stage 4B acceptance report")
    report = _load_json(report_path, "Stage 4B acceptance report")
    if (
        report.get("status") != "PASS"
        or report.get("policy_id") != "GOV-HANDOFF-INDEX-01"
        or report.get("authority") != "non_authoritative_stage4b_shadow_candidate"
        or report.get("manual_handoff_remains_authoritative") is not True
        or report.get("authority_cutover_allowed") is not False
        or report.get("hard_blockers") != []
    ):
        raise HandoffAuthorityError("Stage 4B acceptance report is not PASS at its frozen boundary")
    after_path = _safe_path(
        repo_root,
        DEFAULT_STAGE4B_AFTER_IMAGE.as_posix(),
        "Stage 4B acceptance after-image",
    )
    after = _load_json(after_path, "Stage 4B acceptance after-image")
    if (
        after.get("schema_version") != 1
        or after.get("policy_id") != "GOV-HANDOFF-INDEX-01"
        or after.get("authority") != "shadow_only"
        or report.get("after_image_tree_hash") != after.get("tree_hash")
    ):
        raise HandoffAuthorityError("Stage 4B acceptance after-image binding is invalid")
    return report_path, report, after_path, after


def _source_parent_file_hash(repo_root: Path, source_parent: str, relative: Path) -> str:
    return shadow.sha256_text(_git_show(repo_root, source_parent, relative))


def _load_stage5_ledger(repo_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    ledger = _load_yaml(repo_root / STAGE_LEDGER_PATH, "governance stage ledger")
    stages = _mapping(ledger.get("stages"), "governance stage ledger stages")
    stage5 = _mapping(stages.get("stage_5"), "governance stage ledger stage_5")
    return ledger, stage5


def _stage5_cutover_ledger_payload(
    repo_root: Path,
    *,
    authorization_id: str,
    checkpoint_manifest: str,
) -> dict[str, Any]:
    ledger, stage5 = _load_stage5_ledger(repo_root)
    if (
        stage5.get("current_write_authority") != "manual_handoff"
        or stage5.get("authority_cutover_allowed") is not False
        or stage5.get("production_cutover_executed") is not False
        or stage5.get("implementation_state")
        != "candidate_hardened_pre_cutover_accepted"
        or stage5.get("pre_cutover_acceptance_state")
        != "independently_accepted"
        or stage5.get("repository_pre_cutover_closure") != "complete"
        or not isinstance(stage5.get("pre_cutover_acceptance_report"), str)
    ):
        raise HandoffAuthorityError(
            "stage ledger is not at the independently accepted manual pre-cutover boundary"
        )
    stage5["status_authorization"] = authorization_id
    stage5["implementation_state"] = "production_delta_authority_active"
    stage5["candidate_only"] = False
    stage5["pre_cutover_acceptance_state"] = "accepted_by_cutover_authorization"
    stage5["current_write_authority"] = "schema_v3_delta"
    stage5["authority_cutover_allowed"] = True
    stage5["production_cutover_executed"] = True
    stage5["cutover_authorization"] = authorization_id
    stage5["cutover_checkpoint_manifest"] = checkpoint_manifest
    stage5["last_rollback_report"] = None
    return ledger


def _stage5_rollback_ledger_payload(
    repo_root: Path,
    *,
    rollback_report: str,
    previous_cutover_authorization: str,
) -> dict[str, Any]:
    ledger, stage5 = _load_stage5_ledger(repo_root)
    if (
        stage5.get("current_write_authority") != "schema_v3_delta"
        or stage5.get("production_cutover_executed") is not True
        or stage5.get("cutover_authorization") != previous_cutover_authorization
    ):
        raise HandoffAuthorityError("stage ledger is not at an active delta cutover state")
    stage5["status_authorization"] = _string(
        stage5.get("pre_cutover_acceptance_authorization"),
        "Stage 5 pre-cutover acceptance authorization",
    )
    stage5["implementation_state"] = "candidate_hardened_pre_cutover_accepted"
    stage5["candidate_only"] = True
    stage5["pre_cutover_acceptance_state"] = "independently_accepted"
    stage5["current_write_authority"] = "manual_handoff"
    stage5["authority_cutover_allowed"] = False
    stage5["production_cutover_executed"] = False
    stage5["last_cutover_authorization"] = previous_cutover_authorization
    stage5["cutover_authorization"] = None
    stage5["cutover_checkpoint_manifest"] = None
    stage5["last_rollback_report"] = rollback_report
    return ledger


def _validate_stage5_pre_cutover_acceptance(
    repo_root: Path, *, source_parent: str
) -> tuple[Path, dict[str, Any]]:
    """Require committed independent acceptance before preparing a cutover.

    ``evaluated_commit`` identifies the tested candidate, while
    ``accepted_files`` binds the snapshot committed with the acceptance report.
    Neither permanently freezes later, separately authorized maintenance bytes;
    those remain governed by the current stage ledger and test gates.
    """

    _ledger, stage5 = _load_stage5_ledger(repo_root)
    relative = _string(
        stage5.get("pre_cutover_acceptance_report"),
        "Stage 5 pre-cutover acceptance report",
    )
    path = _safe_path(repo_root, relative, "Stage 5 pre-cutover acceptance report")
    report = _load_json(path, "Stage 5 pre-cutover acceptance report")
    evaluated_commit = _string(
        report.get("evaluated_commit"), "Stage 5 acceptance evaluated_commit"
    )
    if (
        report.get("schema_version") != 1
        or report.get("claim_id") != "GOV-HANDOFF-AUTHORITY-CUTOVER-01"
        or report.get("status") != "PASS"
        or report.get("authority_mode") != "manual"
        or report.get("candidate_implementation_acceptance") != "PASS"
        or report.get("isolated_cutover_rollback_rehearsal") != "PASS"
        or report.get("repository_pre_cutover_closure") != "PASS"
        or report.get("production_cutover_authorized") is not False
        or report.get("production_cutover_executed") is not False
        or stage5.get("accepted_candidate_commit") != evaluated_commit
    ):
        raise HandoffAuthorityError("Stage 5 independent pre-cutover acceptance report is invalid")
    if not shadow.GIT_SHA_RE.fullmatch(evaluated_commit):
        raise HandoffAuthorityError("Stage 5 acceptance evaluated_commit must be a full Git SHA")
    _git(repo_root, "merge-base", "--is-ancestor", evaluated_commit, source_parent)

    accepted_files_commit_value = report.get("accepted_files_commit")
    if accepted_files_commit_value is None:
        introduction_commits = _git_text(
            repo_root,
            "log",
            "--format=%H",
            "--diff-filter=A",
            "--reverse",
            source_parent,
            "--",
            relative,
        ).splitlines()
        if len(introduction_commits) != 1:
            raise HandoffAuthorityError(
                "Stage 5 acceptance report introduction commit is ambiguous"
            )
        accepted_files_commit = introduction_commits[0]
    else:
        accepted_files_commit = _string(
            accepted_files_commit_value, "Stage 5 acceptance accepted_files_commit"
        )
    if not shadow.GIT_SHA_RE.fullmatch(accepted_files_commit):
        raise HandoffAuthorityError(
            "Stage 5 acceptance accepted_files_commit must be a full Git SHA"
        )
    _git(
        repo_root,
        "merge-base",
        "--is-ancestor",
        accepted_files_commit,
        source_parent,
    )

    accepted_files = _mapping(report.get("accepted_files"), "Stage 5 acceptance accepted_files")
    if not accepted_files:
        raise HandoffAuthorityError("Stage 5 acceptance report must bind accepted candidate files")
    for relative_file, expected_hash in accepted_files.items():
        relative_file = _string(relative_file, "Stage 5 acceptance accepted file path")
        expected_hash = _string(
            expected_hash, f"Stage 5 acceptance hash for {relative_file}"
        )
        if not shadow.SHA256_RE.fullmatch(expected_hash):
            raise HandoffAuthorityError(
                f"Stage 5 acceptance hash is invalid for {relative_file}"
            )
        accepted_relative = Path(relative_file)
        if accepted_relative.is_absolute() or ".." in accepted_relative.parts:
            raise HandoffAuthorityError(
                f"Stage 5 acceptance path is unsafe: {relative_file}"
            )
        try:
            accepted_text = _git_show(
                repo_root, accepted_files_commit, accepted_relative
            )
        except HandoffAuthorityError as exc:
            raise HandoffAuthorityError(
                f"Stage 5 accepted candidate file is absent at "
                f"{accepted_files_commit}: {relative_file}"
            ) from exc
        if shadow.sha256_text(accepted_text) != expected_hash:
            raise HandoffAuthorityError(
                f"Stage 5 historical acceptance hash mismatch: {relative_file}"
            )

    relative_path = Path(_repo_relative(repo_root, path, "Stage 5 pre-cutover acceptance report"))
    if _source_parent_file_hash(repo_root, source_parent, relative_path) != shadow.sha256_file(
        path
    ):
        raise HandoffAuthorityError(
            "Stage 5 pre-cutover acceptance report bytes do not match source parent"
        )
    return path, report


def _validate_cutover_authorization(
    repo_root: Path,
    relative: str,
    *,
    checkpoint_id: str,
    source_parent: str,
) -> tuple[Path, dict[str, Any]]:
    path = _safe_path(repo_root, relative, "cutover authorization record")
    payload = _load_yaml(path, "cutover authorization record")
    required_scope = {
        "activate_delta_handoff_authority",
        "create_cutover_checkpoint",
        "enable_manual_to_delta_cutover_transaction",
    }
    scope_items = _list(payload.get("scope"), "cutover authorization scope")
    rollback = _list(payload.get("rollback_plan"), "cutover authorization rollback_plan")
    stage_items = _list(payload.get("stage_ids"), "cutover authorization stage_ids")
    excluded_items = _list(
        payload.get("excluded_scope", []), "cutover authorization excluded_scope"
    )
    if not all(isinstance(item, str) and item for item in scope_items + rollback + stage_items + excluded_items):
        raise HandoffAuthorityError("cutover authorization lists must contain strings")
    scope = set(scope_items)
    stage_ids = set(stage_items)
    excluded_scope = set(excluded_items)
    base_commit = _string(payload.get("base_commit"), "cutover authorization base_commit")
    authorization_id = _string(
        payload.get("authorization_id"), "cutover authorization_id"
    )
    statuses = _mapping(
        payload.get("authorized_stage_statuses"),
        "cutover authorization authorized_stage_statuses",
    )
    if (
        payload.get("schema_version") != 1
        or payload.get("kind") != "stage_transition"
        or payload.get("change_class") != "stage_transition"
        or payload.get("claim_id") != "GOV-HANDOFF-AUTHORITY-CUTOVER-01"
        or payload.get("cutover_checkpoint_id") != checkpoint_id
        or "stage_5" not in stage_ids
        or statuses.get("stage_5") != "active"
        or not required_scope.issubset(scope)
        or bool(required_scope & excluded_scope)
        or not rollback
        or path.stem != authorization_id
        or not shadow.GIT_SHA_RE.fullmatch(base_commit)
    ):
        raise HandoffAuthorityError(
            "cutover requires a separate passing Stage 5 stage-transition authorization"
        )
    _string(payload.get("approval_record"), "cutover authorization approval_record")
    _git(repo_root, "merge-base", "--is-ancestor", base_commit, source_parent)
    relative_path = Path(_repo_relative(repo_root, path, "cutover authorization record"))
    if _source_parent_file_hash(repo_root, source_parent, relative_path) != shadow.sha256_file(path):
        raise HandoffAuthorityError("cutover authorization bytes do not match source parent")
    return path, payload


def prepare_cutover(
    repo_root: Path,
    *,
    checkpoint_id: str,
    authorization_record: str,
    stage3_report: str = DEFAULT_STAGE3_FULL_ACCEPTANCE_REPORT.as_posix(),
    stage4b_report: str = DEFAULT_STAGE4B_ACCEPTANCE_REPORT.as_posix(),
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    """Prepare, but do not commit, one authority cutover transaction."""

    repo_root = repo_root.resolve()
    _require_clean_worktree(repo_root, "cutover preparation")
    authority = load_authority(repo_root)
    if authority["mode"] != "manual":
        raise HandoffAuthorityError("cutover preparation requires manual authority mode")
    if not shadow.UPDATE_ID_RE.fullmatch(checkpoint_id):
        raise HandoffAuthorityError(f"invalid checkpoint_id: {checkpoint_id!r}")

    source_parent = _git_text(repo_root, "rev-parse", "HEAD")
    _validate_stage5_pre_cutover_acceptance(
        repo_root, source_parent=source_parent
    )
    cutover_auth_path, cutover_auth = _validate_cutover_authorization(
        repo_root,
        authorization_record,
        checkpoint_id=checkpoint_id,
        source_parent=source_parent,
    )
    checkpoint_dir = repo_root / CHECKPOINT_ROOT / checkpoint_id
    if checkpoint_dir.exists():
        raise HandoffAuthorityError(f"checkpoint already exists: {checkpoint_id}")
    manifest_relative = (checkpoint_dir / CHECKPOINT_MANIFEST_FILENAME).relative_to(
        repo_root
    ).as_posix()
    cutover_ledger = _stage5_cutover_ledger_payload(
        repo_root,
        authorization_id=cutover_auth["authorization_id"],
        checkpoint_manifest=manifest_relative,
    )

    stage3_path, stage3_payload = _validate_stage3_full_acceptance_report(
        repo_root, stage3_report
    )
    stage3_covered_update_ids = _validate_stage3_full_acceptance_covers_source_parent(
        repo_root, source_parent=source_parent, report=stage3_payload
    )
    stage4b_path, stage4b_payload, after_path, after_payload = (
        _validate_stage4b_acceptance_report(repo_root, stage4b_report)
    )
    for path, label in (
        (stage3_path, "Stage 3 Full Acceptance report"),
        (stage4b_path, "Stage 4B acceptance report"),
        (after_path, "Stage 4B acceptance after-image"),
    ):
        relative = Path(_repo_relative(repo_root, path, label))
        if _source_parent_file_hash(repo_root, source_parent, relative) != shadow.sha256_file(path):
            raise HandoffAuthorityError(f"{label} bytes do not match source parent")

    handoff_text = (repo_root / HANDOFF_PATH).read_text(encoding="utf-8")
    registry_text = (repo_root / REGISTRY_PATH).read_text(encoding="utf-8")
    handoff_hash = shadow.sha256_text(handoff_text)
    registry_hash = shadow.sha256_text(registry_text)
    if _source_parent_file_hash(repo_root, source_parent, HANDOFF_PATH) != handoff_hash:
        raise HandoffAuthorityError("current handoff bytes do not match source parent")
    if _source_parent_file_hash(repo_root, source_parent, REGISTRY_PATH) != registry_hash:
        raise HandoffAuthorityError("current registry bytes do not match source parent")

    checkpoint_dir.mkdir(parents=True)
    checkpoint_handoff = checkpoint_dir / "handoff.md"
    checkpoint_handoff.write_text(handoff_text, encoding="utf-8")
    timestamp = _utc_timestamp(created_at_utc)
    audit_path = checkpoint_dir / CHECKPOINT_AUDIT_FILENAME
    audit = {
        "schema_version": CHECKPOINT_AUDIT_SCHEMA_VERSION,
        "policy_id": "GOV-HANDOFF-AUTHORITY-CUTOVER-01",
        "audit_type": "stage4b_checkpoint_binding",
        "status": "PASS",
        "checkpoint_id": checkpoint_id,
        "source_parent_commit": source_parent,
        "cutover_authorization_id": cutover_auth["authorization_id"],
        "cutover_authorization_record": _repo_relative(
            repo_root, cutover_auth_path, "cutover authorization record"
        ),
        "cutover_authorization_record_sha256": shadow.sha256_file(cutover_auth_path),
        "checkpoint_handoff_sha256": handoff_hash,
        "checkpoint_registry_sha256": registry_hash,
        "stage3_full_acceptance_report": _repo_relative(
            repo_root, stage3_path, "Stage 3 Full Acceptance report"
        ),
        "stage3_full_acceptance_report_sha256": shadow.sha256_file(stage3_path),
        "stage3_successful_real_observation_count": stage3_payload["coverage"][
            "successful_real_observation_count"
        ],
        "stage3_covered_update_ids_fingerprint": shadow.observation_fingerprint(
            stage3_covered_update_ids
        ),
        "stage3_uncovered_real_observation_count": 0,
        "stage3_current_at_source_parent": True,
        "stage4b_acceptance_report": _repo_relative(
            repo_root, stage4b_path, "Stage 4B acceptance report"
        ),
        "stage4b_acceptance_report_sha256": shadow.sha256_file(stage4b_path),
        "stage4b_after_image": _repo_relative(
            repo_root, after_path, "Stage 4B acceptance after-image"
        ),
        "stage4b_after_image_sha256": shadow.sha256_file(after_path),
        "stage4b_after_image_tree_hash": after_payload["tree_hash"],
        "stage4b_acceptance_evaluated_base_commit": stage4b_payload[
            "evaluated_base_commit"
        ],
        "created_at_utc": timestamp,
    }
    shadow.write_json(audit_path, audit)

    manifest_path = checkpoint_dir / CHECKPOINT_MANIFEST_FILENAME
    manifest = {
        "schema_version": CHECKPOINT_MANIFEST_SCHEMA_VERSION,
        "policy_id": "GOV-HANDOFF-AUTHORITY-CUTOVER-01",
        "checkpoint_id": checkpoint_id,
        "source_parent_commit": source_parent,
        "cutover_authorization_id": cutover_auth["authorization_id"],
        "cutover_authorization_record": audit["cutover_authorization_record"],
        "cutover_authorization_record_sha256": audit[
            "cutover_authorization_record_sha256"
        ],
        "handoff_path": _repo_relative(repo_root, checkpoint_handoff, "checkpoint handoff"),
        "handoff_sha256": handoff_hash,
        "registry_sha256_for_provenance": registry_hash,
        "stage3_full_acceptance_report": audit["stage3_full_acceptance_report"],
        "stage3_full_acceptance_report_sha256": audit[
            "stage3_full_acceptance_report_sha256"
        ],
        "stage4b_cutover_audit_report": _repo_relative(
            repo_root, audit_path, "Stage 4B cutover audit report"
        ),
        "stage4b_cutover_audit_report_sha256": shadow.sha256_file(audit_path),
        "created_at_utc": timestamp,
    }
    shadow.write_json(manifest_path, manifest)

    authority["mode"] = "delta"
    authority["delta_authority"]["checkpoint_manifest"] = _repo_relative(
        repo_root, manifest_path, "checkpoint manifest"
    )
    authority["delta_authority"]["activation_parent_commit"] = source_parent
    authority["generated_views"]["stage4a_minimal_refresh"] = True
    authority["safety"]["direct_handoff_edit_forbidden"] = True
    _write_yaml(repo_root / AUTHORITY_PATH, authority)
    _write_yaml(repo_root / STAGE_LEDGER_PATH, cutover_ledger)
    return {
        "status": "PASS",
        "mode": "cutover_prepared",
        "checkpoint_id": checkpoint_id,
        "source_parent_commit": source_parent,
        "cutover_authorization_id": cutover_auth["authorization_id"],
        "checkpoint_manifest": manifest_path.relative_to(repo_root).as_posix(),
        "checkpoint_handoff_sha256": handoff_hash,
        "checkpoint_registry_sha256": registry_hash,
        "requires_commit": True,
        "production_delta_in_cutover_commit_forbidden": True,
    }


def prepare_rollback(
    repo_root: Path,
    *,
    rollback_id: str,
    reason: str,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    """Prepare a delta-to-manual rollback while preserving current handoff bytes."""

    repo_root = repo_root.resolve()
    _require_clean_worktree(repo_root, "rollback preparation")
    authority = load_authority(repo_root)
    if authority["mode"] != "delta":
        raise HandoffAuthorityError("rollback preparation requires delta authority mode")
    if not shadow.UPDATE_ID_RE.fullmatch(rollback_id):
        raise HandoffAuthorityError(f"invalid rollback_id: {rollback_id!r}")
    reason = _string(reason, "rollback reason")
    verified = verify_current_state(repo_root)
    rollback_dir = repo_root / ROLLBACK_ROOT / rollback_id
    if rollback_dir.exists():
        raise HandoffAuthorityError(f"rollback record already exists: {rollback_id}")
    handoff_hash = shadow.sha256_file(repo_root / HANDOFF_PATH)
    registry_hash = shadow.sha256_file(repo_root / REGISTRY_PATH)
    previous_manifest = authority["delta_authority"]["checkpoint_manifest"]
    previous_activation_parent = authority["delta_authority"]["activation_parent_commit"]
    _, stage5 = _load_stage5_ledger(repo_root)
    previous_cutover_authorization = _string(
        stage5.get("cutover_authorization"), "active cutover authorization"
    )
    report_path = rollback_dir / ROLLBACK_REPORT_FILENAME
    rollback_ledger = _stage5_rollback_ledger_payload(
        repo_root,
        rollback_report=report_path.relative_to(repo_root).as_posix(),
        previous_cutover_authorization=previous_cutover_authorization,
    )
    rollback_dir.mkdir(parents=True)
    report = {
        "schema_version": ROLLBACK_REPORT_SCHEMA_VERSION,
        "policy_id": "GOV-HANDOFF-AUTHORITY-CUTOVER-01",
        "status": "PASS",
        "rollback_id": rollback_id,
        "reason": reason,
        "delta_head_commit": _git_text(repo_root, "rev-parse", "HEAD"),
        "previous_checkpoint_manifest": previous_manifest,
        "previous_activation_parent_commit": previous_activation_parent,
        "previous_cutover_authorization": previous_cutover_authorization,
        "accepted_authoritative_update_ids": verified["authoritative_update_ids"],
        "preserved_handoff_sha256": handoff_hash,
        "preserved_registry_sha256": registry_hash,
        "target_authority_mode": "manual",
        "created_at_utc": _utc_timestamp(created_at_utc),
    }
    shadow.write_json(report_path, report)

    authority["mode"] = "manual"
    authority["delta_authority"]["checkpoint_manifest"] = None
    authority["delta_authority"]["activation_parent_commit"] = None
    authority["generated_views"]["stage4a_minimal_refresh"] = False
    authority["safety"]["direct_handoff_edit_forbidden"] = False
    _write_yaml(repo_root / AUTHORITY_PATH, authority)
    _write_yaml(repo_root / STAGE_LEDGER_PATH, rollback_ledger)
    return {
        "status": "PASS",
        "mode": "rollback_prepared",
        "rollback_id": rollback_id,
        "rollback_report": report_path.relative_to(repo_root).as_posix(),
        "preserved_handoff_sha256": handoff_hash,
        "preserved_registry_sha256": registry_hash,
        "requires_commit": True,
    }


def _validate_checkpoint_audit(
    repo_root: Path,
    *,
    audit_path: Path,
    manifest: dict[str, Any],
    source_parent: str,
    checkpoint_handoff_hash: str,
    checkpoint_registry_hash: str,
) -> dict[str, Any]:
    audit = _load_json(audit_path, "Stage 4B cutover audit report")
    allowed = {
        "schema_version",
        "policy_id",
        "audit_type",
        "status",
        "checkpoint_id",
        "source_parent_commit",
        "cutover_authorization_id",
        "cutover_authorization_record",
        "cutover_authorization_record_sha256",
        "checkpoint_handoff_sha256",
        "checkpoint_registry_sha256",
        "stage3_full_acceptance_report",
        "stage3_full_acceptance_report_sha256",
        "stage3_successful_real_observation_count",
        "stage3_covered_update_ids_fingerprint",
        "stage3_uncovered_real_observation_count",
        "stage3_current_at_source_parent",
        "stage4b_acceptance_report",
        "stage4b_acceptance_report_sha256",
        "stage4b_after_image",
        "stage4b_after_image_sha256",
        "stage4b_after_image_tree_hash",
        "stage4b_acceptance_evaluated_base_commit",
        "created_at_utc",
    }
    _reject_unknown(audit, allowed, "Stage 4B cutover audit report")
    if (
        audit.get("schema_version") != CHECKPOINT_AUDIT_SCHEMA_VERSION
        or audit.get("policy_id") != "GOV-HANDOFF-AUTHORITY-CUTOVER-01"
        or audit.get("audit_type") != "stage4b_checkpoint_binding"
        or audit.get("status") != "PASS"
        or audit.get("checkpoint_id") != manifest["checkpoint_id"]
        or audit.get("source_parent_commit") != source_parent
        or audit.get("cutover_authorization_id") != manifest["cutover_authorization_id"]
        or audit.get("cutover_authorization_record")
        != manifest["cutover_authorization_record"]
        or audit.get("cutover_authorization_record_sha256")
        != manifest["cutover_authorization_record_sha256"]
        or audit.get("checkpoint_handoff_sha256") != checkpoint_handoff_hash
        or audit.get("checkpoint_registry_sha256") != checkpoint_registry_hash
    ):
        raise HandoffAuthorityError("Stage 4B cutover audit does not bind checkpoint bytes")
    _utc_timestamp(_string(audit.get("created_at_utc"), "cutover audit created_at_utc"))

    stage3_relative = _string(
        audit.get("stage3_full_acceptance_report"),
        "cutover audit Stage 3 Full Acceptance report",
    )
    if stage3_relative != manifest["stage3_full_acceptance_report"]:
        raise HandoffAuthorityError("checkpoint manifest and cutover audit disagree on Stage 3 report")
    stage3_path, stage3 = _validate_stage3_full_acceptance_report(repo_root, stage3_relative)
    stage3_covered_update_ids = _validate_stage3_full_acceptance_covers_source_parent(
        repo_root, source_parent=source_parent, report=stage3
    )
    stage3_hash = shadow.sha256_file(stage3_path)
    if (
        stage3_hash != audit.get("stage3_full_acceptance_report_sha256")
        or stage3_hash != manifest["stage3_full_acceptance_report_sha256"]
        or _source_parent_file_hash(repo_root, source_parent, Path(stage3_relative)) != stage3_hash
        or audit.get("stage3_successful_real_observation_count")
        != stage3["coverage"]["successful_real_observation_count"]
        or audit.get("stage3_covered_update_ids_fingerprint")
        != shadow.observation_fingerprint(stage3_covered_update_ids)
        or audit.get("stage3_uncovered_real_observation_count") != 0
        or audit.get("stage3_current_at_source_parent") is not True
    ):
        raise HandoffAuthorityError("Stage 3 Full Acceptance report provenance mismatch")

    stage4b_relative = _string(
        audit.get("stage4b_acceptance_report"),
        "cutover audit Stage 4B acceptance report",
    )
    stage4b_path, stage4b, after_path, after = _validate_stage4b_acceptance_report(
        repo_root, stage4b_relative
    )
    after_relative = _repo_relative(repo_root, after_path, "Stage 4B acceptance after-image")
    stage4b_hash = shadow.sha256_file(stage4b_path)
    after_hash = shadow.sha256_file(after_path)
    if (
        stage4b_hash != audit.get("stage4b_acceptance_report_sha256")
        or _source_parent_file_hash(repo_root, source_parent, Path(stage4b_relative))
        != stage4b_hash
        or after_relative != audit.get("stage4b_after_image")
        or after_hash != audit.get("stage4b_after_image_sha256")
        or _source_parent_file_hash(repo_root, source_parent, Path(after_relative)) != after_hash
        or after.get("tree_hash") != audit.get("stage4b_after_image_tree_hash")
        or stage4b.get("after_image_tree_hash") != after.get("tree_hash")
        or stage4b.get("evaluated_base_commit")
        != audit.get("stage4b_acceptance_evaluated_base_commit")
    ):
        raise HandoffAuthorityError("Stage 4B cutover audit provenance mismatch")
    return audit


def _verify_committed_checkpoint_history(
    repo_root: Path,
    *,
    source_parent: str,
    cutover_auth_relative: str,
    manifest_path: Path,
    handoff_file: Path,
    audit_path: Path,
) -> None:
    head = _git_text(repo_root, "rev-parse", "HEAD")
    activation_commits = _git_text(
        repo_root,
        "rev-list",
        "--first-parent",
        "--reverse",
        f"{source_parent}..{head}",
    ).splitlines()
    if not activation_commits:
        raise HandoffAuthorityError("delta mode lacks an authority activation commit")
    activation_commit = activation_commits[0]
    activation_paths = _git_text(
        repo_root,
        "diff-tree",
        "--no-commit-id",
        "--name-only",
        "-r",
        activation_commit,
    ).splitlines()
    if HANDOFF_PATH.as_posix() in activation_paths or REGISTRY_PATH.as_posix() in activation_paths:
        raise HandoffAuthorityError("cutover commit may not modify handoff or registry bytes")
    delta_paths = [
        path
        for path in activation_paths
        if path.startswith(f"{DELTA_ROOT.as_posix()}/") and path.endswith(f"/{DELTA_FILENAME}")
    ]
    if delta_paths:
        raise HandoffAuthorityError(
            "cutover commit may not include the first production schema-v3 delta"
        )

    cutover_auth_touches = _git_text(
        repo_root,
        "log",
        "--format=%H",
        f"{source_parent}..{head}",
        "--",
        cutover_auth_relative,
    ).splitlines()
    if cutover_auth_touches:
        raise HandoffAuthorityError("cutover authorization record is not immutable")

    protected_assets = (manifest_path, handoff_file, audit_path, repo_root / AUTHORITY_PATH)
    for protected in protected_assets:
        relative = _repo_relative(repo_root, protected, "protected authority asset")
        touches = _git_text(
            repo_root,
            "log",
            "--format=%H",
            f"{source_parent}..{head}",
            "--",
            relative,
        ).splitlines()
        if touches != [activation_commit]:
            raise HandoffAuthorityError(
                f"cutover checkpoint/authority asset is not immutable: {relative}"
            )


def _load_checkpoint(
    repo_root: Path,
    authority: dict[str, Any],
    *,
    require_activation_commit: bool = True,
) -> tuple[dict[str, Any], str]:
    manifest_path = _safe_path(
        repo_root,
        authority["delta_authority"]["checkpoint_manifest"],
        "checkpoint manifest",
    )
    manifest = _load_json(manifest_path, "checkpoint manifest")
    if manifest.get("schema_version") != CHECKPOINT_MANIFEST_SCHEMA_VERSION:
        raise HandoffAuthorityError(
            f"checkpoint manifest schema_version must be {CHECKPOINT_MANIFEST_SCHEMA_VERSION}"
        )
    allowed = {
        "schema_version",
        "policy_id",
        "checkpoint_id",
        "source_parent_commit",
        "cutover_authorization_id",
        "cutover_authorization_record",
        "cutover_authorization_record_sha256",
        "handoff_path",
        "handoff_sha256",
        "registry_sha256_for_provenance",
        "stage3_full_acceptance_report",
        "stage3_full_acceptance_report_sha256",
        "stage4b_cutover_audit_report",
        "stage4b_cutover_audit_report_sha256",
        "created_at_utc",
    }
    _reject_unknown(manifest, allowed, "checkpoint manifest")
    if manifest.get("policy_id") != "GOV-HANDOFF-AUTHORITY-CUTOVER-01":
        raise HandoffAuthorityError("checkpoint manifest policy_id mismatch")
    checkpoint_id = _string(manifest.get("checkpoint_id"), "checkpoint_id")
    if not shadow.UPDATE_ID_RE.fullmatch(checkpoint_id):
        raise HandoffAuthorityError("checkpoint_id is invalid")
    expected_checkpoint_dir = CHECKPOINT_ROOT / checkpoint_id
    manifest_checkpoint_dir = Path(
        _repo_relative(repo_root, manifest_path.parent, "checkpoint manifest directory")
    )
    if manifest_checkpoint_dir != expected_checkpoint_dir:
        raise HandoffAuthorityError("checkpoint manifest is outside its checkpoint directory")

    source_parent = _string(manifest.get("source_parent_commit"), "source_parent_commit")
    if not shadow.GIT_SHA_RE.fullmatch(source_parent):
        raise HandoffAuthorityError("checkpoint source_parent_commit must be a full Git SHA")
    if source_parent != authority["delta_authority"]["activation_parent_commit"]:
        raise HandoffAuthorityError("checkpoint source parent must equal activation parent")

    cutover_auth_id = _string(
        manifest.get("cutover_authorization_id"), "cutover_authorization_id"
    )
    cutover_auth_relative = _string(
        manifest.get("cutover_authorization_record"), "cutover_authorization_record"
    )
    cutover_auth_hash = _string(
        manifest.get("cutover_authorization_record_sha256"),
        "cutover_authorization_record_sha256",
    )
    if not shadow.SHA256_RE.fullmatch(cutover_auth_hash):
        raise HandoffAuthorityError("cutover authorization hash must be SHA-256")
    cutover_auth_path, cutover_auth = _validate_cutover_authorization(
        repo_root,
        cutover_auth_relative,
        checkpoint_id=checkpoint_id,
        source_parent=source_parent,
    )
    if (
        cutover_auth["authorization_id"] != cutover_auth_id
        or shadow.sha256_file(cutover_auth_path) != cutover_auth_hash
    ):
        raise HandoffAuthorityError("cutover authorization provenance mismatch")

    handoff_file = _safe_path(
        repo_root,
        _string(manifest.get("handoff_path"), "handoff_path"),
        "checkpoint handoff",
    )
    if handoff_file.parent != manifest_path.parent or handoff_file.name != "handoff.md":
        raise HandoffAuthorityError("checkpoint handoff must live beside its manifest")
    text = handoff_file.read_text(encoding="utf-8")
    handoff_hash = shadow.sha256_text(text)
    if handoff_hash != manifest.get("handoff_sha256"):
        raise HandoffAuthorityError("checkpoint handoff SHA-256 mismatch")
    if _git_show(repo_root, source_parent, HANDOFF_PATH) != text:
        raise HandoffAuthorityError("checkpoint bytes do not match source parent handoff")

    registry_hash = _string(
        manifest.get("registry_sha256_for_provenance"),
        "registry_sha256_for_provenance",
    )
    if not shadow.SHA256_RE.fullmatch(registry_hash):
        raise HandoffAuthorityError("checkpoint registry provenance hash must be SHA-256")
    source_registry_hash = shadow.sha256_text(
        _git_show(repo_root, source_parent, REGISTRY_PATH)
    )
    if registry_hash != source_registry_hash:
        raise HandoffAuthorityError(
            "checkpoint registry provenance hash does not match source-parent registry"
        )

    stage3_relative = _string(
        manifest.get("stage3_full_acceptance_report"),
        "stage3_full_acceptance_report",
    )
    stage3_hash = _string(
        manifest.get("stage3_full_acceptance_report_sha256"),
        "stage3_full_acceptance_report_sha256",
    )
    if not shadow.SHA256_RE.fullmatch(stage3_hash):
        raise HandoffAuthorityError("Stage 3 Full Acceptance report hash must be SHA-256")

    audit_path = _safe_path(
        repo_root,
        _string(manifest.get("stage4b_cutover_audit_report"), "stage4b_cutover_audit_report"),
        "Stage 4B cutover audit report",
    )
    if audit_path.parent != manifest_path.parent or audit_path.name != CHECKPOINT_AUDIT_FILENAME:
        raise HandoffAuthorityError("Stage 4B cutover audit must live beside checkpoint manifest")
    audit_hash = _string(
        manifest.get("stage4b_cutover_audit_report_sha256"),
        "stage4b_cutover_audit_report_sha256",
    )
    if not shadow.SHA256_RE.fullmatch(audit_hash) or shadow.sha256_file(audit_path) != audit_hash:
        raise HandoffAuthorityError("Stage 4B cutover audit SHA-256 mismatch")
    _validate_checkpoint_audit(
        repo_root,
        audit_path=audit_path,
        manifest=manifest,
        source_parent=source_parent,
        checkpoint_handoff_hash=handoff_hash,
        checkpoint_registry_hash=registry_hash,
    )
    _utc_timestamp(_string(manifest.get("created_at_utc"), "created_at_utc"))

    stage3_path = _safe_path(repo_root, stage3_relative, "Stage 3 Full Acceptance report")
    if shadow.sha256_file(stage3_path) != stage3_hash:
        raise HandoffAuthorityError("Stage 3 Full Acceptance report SHA-256 mismatch")
    if require_activation_commit:
        _verify_committed_checkpoint_history(
            repo_root,
            source_parent=source_parent,
            cutover_auth_relative=cutover_auth_relative,
            manifest_path=manifest_path,
            handoff_file=handoff_file,
            audit_path=audit_path,
        )
    return manifest, text


def _first_parent_positions(repo_root: Path, activation: str, head: str) -> dict[str, int]:
    commits = _git_text(
        repo_root, "rev-list", "--first-parent", "--reverse", f"{activation}..{head}"
    ).splitlines()
    return {commit: index for index, commit in enumerate(commits)}


def _path_exists_at_commit(repo_root: Path, commit: str, relative: str) -> bool:
    return _git(
        repo_root,
        "cat-file",
        "-e",
        f"{commit}:{relative}",
        check=False,
    ).returncode == 0


def _first_parent_integration_commit(
    repo_root: Path,
    *,
    positions: dict[str, int],
    first_add: str,
    relative: str,
) -> str:
    """Map a side-branch delta addition to its first-parent merge integration."""

    ordered = sorted(positions, key=positions.__getitem__)
    for commit in ordered:
        if _git(
            repo_root,
            "merge-base",
            "--is-ancestor",
            first_add,
            commit,
            check=False,
        ).returncode != 0:
            continue
        if not _path_exists_at_commit(repo_root, commit, relative):
            continue
        first_parent = _git_text(repo_root, "rev-parse", f"{commit}^1")
        if _path_exists_at_commit(repo_root, first_parent, relative):
            continue
        return commit
    raise HandoffAuthorityError(
        "authoritative delta was not integrated on the first-parent history: "
        + relative
    )


def _legacy_inert_materialization_report(
    repo_root: Path,
    path: Path,
    delta: dict[str, Any],
    *,
    first_add: str,
    integration_commit: str,
) -> dict[str, Any] | None:
    relative = _repo_relative(repo_root, path, "authoritative delta")
    expected = LEGACY_INERT_V3_DELTAS.get(relative)
    if expected is None:
        return None

    parent = _git_text(repo_root, "rev-parse", f"{first_add}^1")
    integration_parent = _git_text(repo_root, "rev-parse", f"{integration_commit}^1")
    changed_paths = _git_text(
        repo_root,
        "diff-tree",
        "--no-commit-id",
        "--name-only",
        "-r",
        first_add,
    ).splitlines()
    base_handoff = _git_show(repo_root, delta["base"]["commit"], HANDOFF_PATH)
    before_handoff = _git_show(repo_root, parent, HANDOFF_PATH)
    after_handoff = _git_show(repo_root, first_add, HANDOFF_PATH)
    integration_before_handoff = _git_show(repo_root, integration_parent, HANDOFF_PATH)
    integration_after_handoff = _git_show(repo_root, integration_commit, HANDOFF_PATH)
    try:
        rendered = shadow.render(base_handoff, delta["operations"])
        shadow.verify_history_preservation(base_handoff, rendered.text, delta["operations"])
    except shadow.HandoffDeltaError as exc:
        raise HandoffAuthorityError(
            f"legacy inert delta replay failed unexpectedly: {relative}: {exc}"
        ) from exc

    actual = {
        "update_id": delta.get("update_id"),
        "delta_sha256": shadow.sha256_file(path),
        "first_add_commit": first_add,
        "first_add_parent": parent,
        "integration_commit": integration_commit,
        "integration_parent": integration_parent,
        "source_base_commit": delta.get("base", {}).get("commit"),
        "historical_handoff_sha256": shadow.sha256_text(base_handoff),
        "declared_candidate_sha256": delta.get("expected", {}).get(
            "exact_base_candidate_sha256"
        ),
        "rendered_candidate_sha256": shadow.sha256_text(rendered.text),
    }
    if actual != expected:
        raise HandoffAuthorityError(
            f"legacy inert delta provenance mismatch: {relative}"
        )
    if changed_paths != [relative]:
        raise HandoffAuthorityError(
            f"legacy inert delta introduction boundary changed: {relative}"
        )
    historical_hash = expected["historical_handoff_sha256"]
    if (
        shadow.sha256_text(before_handoff) != historical_hash
        or shadow.sha256_text(after_handoff) != historical_hash
        or shadow.sha256_text(integration_before_handoff) != historical_hash
        or shadow.sha256_text(integration_after_handoff) != historical_hash
        or delta["base"].get("handoff_sha256") != historical_hash
    ):
        raise HandoffAuthorityError(
            f"legacy inert delta handoff history mismatch: {relative}"
        )
    return {
        "current_handoff_before_sha256": historical_hash,
        "materialized_handoff_after_sha256": historical_hash,
        "update_id": delta["update_id"],
    }


def _discover_v3_deltas(
    repo_root: Path, authority: dict[str, Any]
) -> list[DiscoveredDelta]:
    activation = authority["delta_authority"]["activation_parent_commit"]
    head = _git_text(repo_root, "rev-parse", "HEAD")
    positions = _first_parent_positions(repo_root, activation, head)
    records: list[DiscoveredDelta] = []
    for path in sorted((repo_root / DELTA_ROOT).glob(f"*/{DELTA_FILENAME}")):
        payload = _load_yaml(path, "handoff delta")
        if payload.get("schema_version") != SCHEMA_VERSION:
            continue
        validate_v3_delta(payload, path)
        relative = _repo_relative(repo_root, path, "authoritative delta")
        touches = _git_text(repo_root, "log", "--format=%H", "--", relative).splitlines()
        adds = _git_text(
            repo_root, "log", "--diff-filter=A", "--format=%H", "--", relative
        ).splitlines()
        if len(adds) != 1 or len(touches) != 1:
            raise HandoffAuthorityError(f"authoritative delta is not immutable: {relative}")
        first_add = adds[0]
        if first_add in positions:
            integration_commit = first_add
        else:
            integration_commit = _first_parent_integration_commit(
                repo_root,
                positions=positions,
                first_add=first_add,
                relative=relative,
            )
        report = path.parent / REPORT_FILENAME
        if not report.is_file():
            legacy_report = _legacy_inert_materialization_report(
                repo_root,
                path,
                payload,
                first_add=first_add,
                integration_commit=integration_commit,
            )
            if legacy_report is None:
                raise HandoffAuthorityError(
                    f"authoritative delta lacks materialization report: {relative}"
                )
            records.append(
                DiscoveredDelta(
                    integration_commit,
                    path,
                    payload,
                    legacy_report,
                    legacy_inert=True,
                )
            )
            continue
        report_touches = _git_text(
            repo_root,
            "log",
            "--format=%H",
            "--",
            _repo_relative(repo_root, report, "materialization report"),
        ).splitlines()
        if report_touches != [first_add]:
            raise HandoffAuthorityError(f"materialization report is not immutable: {report}")
        records.append(
            DiscoveredDelta(
                integration_commit,
                path,
                payload,
                _load_materialization_report(report, payload),
            )
        )
    records.sort(key=lambda item: positions[item.integration_commit])
    return records


def _load_materialization_report(path: Path, delta: dict[str, Any]) -> dict[str, Any]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HandoffAuthorityError(f"Invalid materialization report {path}: {exc}") from exc
    if not isinstance(report, dict):
        raise HandoffAuthorityError("materialization report must be a mapping")
    if (
        report.get("report_schema_version") != 1
        or report.get("status") != "PASS"
        or report.get("mode") != "authoritative"
        or report.get("update_id") != delta["update_id"]
        or report.get("delta_sha256") != shadow.sha256_file(path.parent / DELTA_FILENAME)
        or report.get("exact_base_candidate_sha256")
        != delta["expected"]["exact_base_candidate_sha256"]
        or report.get("source_base_commit") != delta["base"]["commit"]
        or report.get("idempotence_passed") is not True
        or report.get("history_preservation_passed") is not True
        or report.get("structural_lint_passed") is not True
        or report.get("network_used") is not False
    ):
        raise HandoffAuthorityError(f"materialization report contract mismatch: {path}")
    return report


def materialize_all(
    repo_root: Path, authority: dict[str, Any]
) -> tuple[str, list[str], list[str]]:
    _, text = _load_checkpoint(repo_root, authority)
    existing_ids = _existing_block_ids(text)
    update_ids: list[str] = []
    legacy_inert_ids: list[str] = []
    for record in _discover_v3_deltas(repo_root, authority):
        path = record.path
        delta = record.delta
        report = record.report
        if report.get("current_handoff_before_sha256") != shadow.sha256_text(text):
            raise HandoffAuthorityError(
                f"materialization report before-hash breaks replay chain: {path}"
            )
        if record.legacy_inert:
            if report.get("materialized_handoff_after_sha256") != shadow.sha256_text(text):
                raise HandoffAuthorityError(
                    f"legacy inert delta breaks unchanged replay chain: {path}"
                )
            legacy_inert_ids.append(delta["update_id"])
            continue
        for op in delta["operations"]:
            block_id = op.get("block_id")
            if block_id and block_id in existing_ids:
                raise HandoffAuthorityError(f"authoritative block ID reused: {block_id}")
        try:
            rendered = shadow.render(text, delta["operations"])
            shadow.verify_history_preservation(text, rendered.text, delta["operations"])
        except shadow.HandoffDeltaError as exc:
            raise HandoffAuthorityError(str(exc)) from exc
        previous_text = text
        text = rendered.text
        existing_ids = _existing_block_ids(text)
        if _new_duplicate_heading_paths(previous_text, text):
            raise HandoffAuthorityError("authoritative replay created duplicate heading paths")
        if report.get("materialized_handoff_after_sha256") != shadow.sha256_text(text):
            raise HandoffAuthorityError(
                f"materialization report after-hash breaks replay chain: {path}"
            )
        update_ids.append(delta["update_id"])
    return text, update_ids, legacy_inert_ids


def verify_prepared_cutover(repo_root: Path, *, check_stage4a: bool = True) -> dict[str, Any]:
    """Verify an uncommitted cutover transaction without requiring Git history.

    This phase validates the exact worktree boundary plus all checkpoint and
    provenance bindings.  The normal committed verifier remains responsible
    for proving activation-commit history and post-activation immutability.
    """

    repo_root = repo_root.resolve()
    authority = load_authority(repo_root)
    if authority["mode"] != "delta":
        raise HandoffAuthorityError(
            "prepared cutover verification requires delta authority in the worktree"
        )
    manifest, checkpoint_text = _load_checkpoint(
        repo_root, authority, require_activation_commit=False
    )
    source_parent = _string(manifest.get("source_parent_commit"), "source_parent_commit")
    head = _git_text(repo_root, "rev-parse", "HEAD")
    if head != source_parent:
        raise HandoffAuthorityError("prepared cutover must remain uncommitted at its source parent")

    manifest_path = Path(authority["delta_authority"]["checkpoint_manifest"])
    checkpoint_dir = repo_root / manifest_path.parent
    expected_paths = {
        AUTHORITY_PATH.as_posix(),
        STAGE_LEDGER_PATH.as_posix(),
        manifest_path.as_posix(),
        _string(manifest.get("handoff_path"), "handoff_path"),
        _string(
            manifest.get("stage4b_cutover_audit_report"),
            "stage4b_cutover_audit_report",
        ),
    }
    changed_paths = _worktree_changed_paths(repo_root)
    if changed_paths != expected_paths:
        missing = sorted(expected_paths - changed_paths)
        unexpected = sorted(changed_paths - expected_paths)
        raise HandoffAuthorityError(
            "prepared cutover worktree boundary mismatch "
            f"(missing={missing}, unexpected={unexpected})"
        )
    checkpoint_files = {
        path.relative_to(repo_root).as_posix()
        for path in checkpoint_dir.iterdir()
        if path.is_file()
    }
    expected_checkpoint_files = expected_paths - {
        AUTHORITY_PATH.as_posix(),
        STAGE_LEDGER_PATH.as_posix(),
    }
    if checkpoint_files != expected_checkpoint_files:
        raise HandoffAuthorityError(
            "prepared cutover checkpoint directory contains unexpected files"
        )

    current_handoff = (repo_root / HANDOFF_PATH).read_text(encoding="utf-8")
    current_registry = (repo_root / REGISTRY_PATH).read_text(encoding="utf-8")
    if _git_show(repo_root, head, HANDOFF_PATH) != current_handoff:
        raise HandoffAuthorityError("prepared cutover may not modify docs/handoff.md")
    if _git_show(repo_root, head, REGISTRY_PATH) != current_registry:
        raise HandoffAuthorityError("prepared cutover may not modify experiments/registry.yaml")
    if current_handoff != checkpoint_text:
        raise HandoffAuthorityError(
            "prepared cutover checkpoint does not preserve current handoff bytes"
        )

    _ledger, stage5 = _load_stage5_ledger(repo_root)
    cutover_auth = _string(manifest.get("cutover_authorization_id"), "cutover_authorization_id")
    if (
        stage5.get("status_authorization") != cutover_auth
        or stage5.get("implementation_state") != "production_delta_authority_active"
        or stage5.get("candidate_only") is not False
        or stage5.get("pre_cutover_acceptance_state") != "accepted_by_cutover_authorization"
        or stage5.get("current_write_authority") != "schema_v3_delta"
        or stage5.get("authority_cutover_allowed") is not True
        or stage5.get("production_cutover_executed") is not True
        or stage5.get("cutover_authorization") != cutover_auth
        or stage5.get("cutover_checkpoint_manifest") != manifest_path.as_posix()
    ):
        raise HandoffAuthorityError(
            "prepared cutover ledger does not match the authority transaction"
        )

    if check_stage4a:
        _verify_stage4a_current_source(repo_root)
    return {
        "status": "PASS",
        "mode": "cutover_prepared",
        "source_parent_commit": source_parent,
        "checkpoint_id": manifest["checkpoint_id"],
        "cutover_authorization_id": cutover_auth,
        "changed_paths": sorted(changed_paths),
        "handoff_sha256": shadow.sha256_text(current_handoff),
        "registry_sha256": shadow.sha256_text(current_registry),
        "stage4a_checked": check_stage4a,
        "requires_commit": True,
    }


def verify_current_state(repo_root: Path, *, check_stage4a: bool = True) -> dict[str, Any]:
    authority = load_authority(repo_root)
    if authority["mode"] == "manual":
        return {
            "status": "PASS",
            "mode": "manual",
            "manual_handoff_authoritative": True,
            "authority_cutover_allowed": False,
        }
    expected, update_ids, legacy_inert_ids = materialize_all(repo_root, authority)
    current = (repo_root / HANDOFF_PATH).read_text(encoding="utf-8")
    if current != expected:
        raise HandoffAuthorityError("tracked handoff is stale or was directly edited")
    if check_stage4a:
        _verify_stage4a_current_source(repo_root)
    return {
        "status": "PASS",
        "mode": "delta",
        "manual_handoff_authoritative": False,
        "authoritative_delta_count": len(update_ids),
        "authoritative_update_ids": update_ids,
        "legacy_inert_delta_count": len(legacy_inert_ids),
        "legacy_inert_update_ids": legacy_inert_ids,
        "handoff_sha256": shadow.sha256_text(current),
        "stage4a_checked": check_stage4a,
    }


def normalize_update(
    target_repo: Path,
    trusted_repo: Path,
    *,
    current_before: str,
    source_base: str,
    source_patch_commit: str,
    allow_authority_transition: bool = False,
) -> dict[str, Any]:
    target_repo = target_repo.resolve()
    trusted_repo = trusted_repo.resolve()
    trusted_authority_path = trusted_repo / AUTHORITY_PATH
    target_authority_path = target_repo / AUTHORITY_PATH
    if trusted_authority_path.read_bytes() != target_authority_path.read_bytes():
        if not allow_authority_transition:
            raise HandoffAuthorityError(
                "authority transition is forbidden in an ordinary content package"
            )
        raise HandoffAuthorityError(
            "authority transition support is intentionally deferred to the independently authorized cutover"
        )
    authority = load_authority(trusted_repo)
    if authority["mode"] == "manual":
        return {
            "status": "PASS",
            "mode": "manual",
            "normalization": "not_applicable",
            "authority_transitioned": False,
        }

    verify_current_state(trusted_repo)
    _git(target_repo, "merge-base", "--is-ancestor", source_base, current_before)
    changed = _source_changed_paths(
        target_repo, source_base=source_base, source_patch_commit=source_patch_commit
    )
    if HANDOFF_PATH.as_posix() in changed:
        raise HandoffAuthorityError("source package may not directly modify docs/handoff.md")
    if any(path.startswith(DYNAMIC_STAGE4A_PREFIX) for path in changed):
        raise HandoffAuthorityError("source package may not directly modify Stage 4A generated views")
    reports_in_source = [
        path
        for path in changed
        if Path(path).name in SOURCE_AUTHORED_REPORT_NAMES
        and path.startswith(f"{DELTA_ROOT.as_posix()}/")
    ]
    if reports_in_source:
        raise HandoffAuthorityError(
            "source package may not self-author handoff acceptance/materialization reports"
        )
    control_changes = sorted(path for path in changed if _is_control_plane(path))
    if control_changes:
        raise HandoffAuthorityError(
            "content package modifies trusted control-plane paths: " + ", ".join(control_changes)
        )

    changed_delta_files = sorted(
        path
        for path in changed
        if path.startswith(f"{DELTA_ROOT.as_posix()}/")
        and path.endswith(f"/{DELTA_FILENAME}")
    )
    if not changed_delta_files:
        if REGISTRY_PATH.as_posix() in changed:
            raise HandoffAuthorityError(
                "registry changes in delta mode require exactly one newly added schema-v3 delta"
            )
        verified_target = verify_current_state(target_repo)
        return {
            "status": "PASS",
            "mode": "delta",
            "normalization": "no_op",
            "authority_transitioned": False,
            "handoff_sha256": verified_target["handoff_sha256"],
            "authoritative_delta_count": verified_target["authoritative_delta_count"],
            "reason": "source package does not change handoff or registry authority state",
        }

    delta_path = _find_new_v3_delta(target_repo, changed)
    intent = validate_exact_base_intent(
        target_repo, delta_path, source_patch_commit=source_patch_commit
    )
    delta = intent.delta
    if delta["base"]["commit"] != source_base:
        raise HandoffAuthorityError("schema-v3 delta base commit differs from package base")

    current_handoff = _git_show(target_repo, current_before, HANDOFF_PATH)
    current_registry = _git_show(target_repo, current_before, REGISTRY_PATH)
    target_registry = (target_repo / REGISTRY_PATH).read_text(encoding="utf-8")
    existing = _existing_block_ids(current_handoff)
    for op in delta["operations"]:
        block_id = op.get("block_id")
        if block_id and block_id in existing:
            raise HandoffAuthorityError(f"authoritative block ID already exists: {block_id}")
    try:
        rendered = shadow.render(current_handoff, delta["operations"])
        shadow.verify_history_preservation(current_handoff, rendered.text, delta["operations"])
    except shadow.HandoffDeltaError as exc:
        raise HandoffAuthorityError(str(exc)) from exc
    if _new_duplicate_heading_paths(current_handoff, rendered.text):
        raise HandoffAuthorityError("current-state application creates duplicate heading paths")

    current_proxy = _proxy_delta_for_registry(
        delta, expected_after_sha256=shadow.sha256_text(target_registry)
    )
    current_proxy["base"] = {
        **delta["base"],
        "registry_sha256": shadow.sha256_text(current_registry),
    }
    try:
        current_registry_report = shadow.validate_registry(
            target_repo, current_proxy, current_registry, target_registry
        )
    except shadow.HandoffDeltaError as exc:
        raise HandoffAuthorityError(str(exc)) from exc

    handoff_file = target_repo / HANDOFF_PATH
    handoff_before_hash = shadow.sha256_text(current_handoff)
    handoff_file.write_text(rendered.text, encoding="utf-8")
    refreshed, reused = _run_stage4a_refresh(target_repo, trusted_repo)

    report_path = delta_path.parent / REPORT_FILENAME
    report = {
        "report_schema_version": 1,
        "policy_id": "GOV-HANDOFF-AUTHORITY-CUTOVER-01",
        "mode": "authoritative",
        "status": "PASS",
        "update_id": delta["update_id"],
        "source_base_commit": source_base,
        "current_before_commit": current_before,
        "source_patch_commit": source_patch_commit,
        "delta_sha256": shadow.sha256_file(delta_path),
        "exact_base_candidate_sha256": delta["expected"]["exact_base_candidate_sha256"],
        "current_handoff_before_sha256": handoff_before_hash,
        "materialized_handoff_after_sha256": shadow.sha256_text(rendered.text),
        "registry_before_sha256": shadow.sha256_text(current_registry),
        "registry_after_sha256": shadow.sha256_text(target_registry),
        "trusted_engine_sha256": shadow.sha256_file(trusted_repo / "scripts/handoff_authority.py"),
        "trusted_renderer_sha256": shadow.sha256_file(trusted_repo / "scripts/handoff_delta_shadow.py"),
        "trusted_policy_sha256": shadow.sha256_file(trusted_repo / AUTHORITY_PATH),
        "exact_base_registry_validation": intent.registry_report,
        "current_registry_validation": current_registry_report,
        "stage4a_refresh": {
            "refreshed": [f"{DYNAMIC_STAGE4A_PREFIX}{path}" for path in refreshed],
            "reused": [f"{DYNAMIC_STAGE4A_PREFIX}{path}" for path in reused],
        },
        "idempotence_passed": True,
        "history_preservation_passed": True,
        "structural_lint_passed": True,
        "network_used": False,
    }
    shadow.write_json(report_path, report)
    return {
        "status": "PASS",
        "mode": "delta",
        "normalization": "materialized",
        "update_id": delta["update_id"],
        "delta_path": delta_path.relative_to(target_repo).as_posix(),
        "report_path": report_path.relative_to(target_repo).as_posix(),
        "handoff_before_sha256": handoff_before_hash,
        "handoff_after_sha256": shadow.sha256_text(rendered.text),
        "stage4a_refreshed": report["stage4a_refresh"]["refreshed"],
        "stage4a_reused_count": len(reused),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    verify = sub.add_parser("verify")
    verify.add_argument("--repo-root", type=Path, default=Path.cwd())
    verify.add_argument(
        "--prepared",
        action="store_true",
        help="verify an uncommitted cutover transaction before creating its activation commit",
    )
    verify.add_argument("--skip-stage4a", action="store_true")
    verify.add_argument("--json", action="store_true")

    validate = sub.add_parser("validate-delta")
    validate.add_argument("--repo-root", type=Path, default=Path.cwd())
    validate.add_argument("--delta", type=Path, required=True)
    validate.add_argument("--source-patch-commit")
    validate.add_argument("--json", action="store_true")

    normalize = sub.add_parser("normalize")
    normalize.add_argument("--repo-root", type=Path, required=True)
    normalize.add_argument("--trusted-repo-root", type=Path, required=True)
    normalize.add_argument("--current-before", required=True)
    normalize.add_argument("--source-base", required=True)
    normalize.add_argument("--source-patch-commit", required=True)
    normalize.add_argument("--allow-authority-transition", action="store_true")
    normalize.add_argument("--json", action="store_true")

    cutover = sub.add_parser(
        "cutover",
        help="prepare checkpoint plus manual-to-delta authority transition without committing",
    )
    cutover.add_argument("--repo-root", type=Path, default=Path.cwd())
    cutover.add_argument("--checkpoint-id", required=True)
    cutover.add_argument("--authorization-record", required=True)
    cutover.add_argument(
        "--stage3-full-acceptance-report",
        default=DEFAULT_STAGE3_FULL_ACCEPTANCE_REPORT.as_posix(),
    )
    cutover.add_argument(
        "--stage4b-acceptance-report",
        default=DEFAULT_STAGE4B_ACCEPTANCE_REPORT.as_posix(),
    )
    cutover.add_argument("--created-at-utc")
    cutover.add_argument("--json", action="store_true")

    rollback = sub.add_parser(
        "rollback",
        help="prepare delta-to-manual authority rollback while preserving current materialized bytes",
    )
    rollback.add_argument("--repo-root", type=Path, default=Path.cwd())
    rollback.add_argument("--rollback-id", required=True)
    rollback.add_argument("--reason", required=True)
    rollback.add_argument("--created-at-utc")
    rollback.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "verify":
            verifier = verify_prepared_cutover if args.prepared else verify_current_state
            payload = verifier(
                args.repo_root.resolve(), check_stage4a=not args.skip_stage4a
            )
        elif args.command == "validate-delta":
            repo = args.repo_root.resolve()
            delta = args.delta if args.delta.is_absolute() else repo / args.delta
            intent = validate_exact_base_intent(
                repo, delta.resolve(), source_patch_commit=args.source_patch_commit
            )
            payload = {
                "status": "PASS",
                "mode": "authoritative",
                "update_id": intent.delta["update_id"],
                "exact_base_candidate_sha256": shadow.sha256_text(intent.candidate),
                "registry_validation": intent.registry_report,
            }
        elif args.command == "normalize":
            payload = normalize_update(
                args.repo_root,
                args.trusted_repo_root,
                current_before=args.current_before,
                source_base=args.source_base,
                source_patch_commit=args.source_patch_commit,
                allow_authority_transition=args.allow_authority_transition,
            )
        elif args.command == "cutover":
            payload = prepare_cutover(
                args.repo_root,
                checkpoint_id=args.checkpoint_id,
                authorization_record=args.authorization_record,
                stage3_report=args.stage3_full_acceptance_report,
                stage4b_report=args.stage4b_acceptance_report,
                created_at_utc=args.created_at_utc,
            )
        elif args.command == "rollback":
            payload = prepare_rollback(
                args.repo_root,
                rollback_id=args.rollback_id,
                reason=args.reason,
                created_at_utc=args.created_at_utc,
            )
        else:  # pragma: no cover
            raise AssertionError(args.command)
    except (HandoffAuthorityError, shadow.HandoffDeltaError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            "Stage 5 handoff authority: PASS "
            f"(command={args.command}, mode={payload.get('mode')})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
