#!/usr/bin/env python3
"""Trusted Stage 5 versioned-handoff materializer and validator.

The current repository remains in manual mode.  This tool implements the future
schema-v3 delta authority behind an explicit authority file and is invoked by
``drpo-update`` from the pre-integration current-main checkout.  It deliberately
reuses the Stage 3 renderer core rather than creating a second document engine.
"""
from __future__ import annotations

import argparse
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
HANDOFF_PATH = Path("docs/handoff.md")
REGISTRY_PATH = Path("experiments/registry.yaml")
DELTA_ROOT = Path("docs/handoff_deltas")
DELTA_FILENAME = "HANDOFF_DELTA.yaml"
REPORT_FILENAME = "MATERIALIZATION_REPORT.json"
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


def _load_checkpoint(repo_root: Path, authority: dict[str, Any]) -> tuple[dict[str, Any], str]:
    manifest_path = _safe_path(
        repo_root,
        authority["delta_authority"]["checkpoint_manifest"],
        "checkpoint manifest",
    )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HandoffAuthorityError(f"Invalid checkpoint manifest: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise HandoffAuthorityError("checkpoint manifest schema_version must be 1")
    allowed = {
        "schema_version",
        "checkpoint_id",
        "source_parent_commit",
        "handoff_path",
        "handoff_sha256",
        "registry_sha256_for_provenance",
        "stage3_full_acceptance_report",
        "stage4b_cutover_audit_report",
        "created_at_utc",
    }
    _reject_unknown(manifest, allowed, "checkpoint manifest")
    source_parent = _string(manifest.get("source_parent_commit"), "source_parent_commit")
    if not shadow.GIT_SHA_RE.fullmatch(source_parent):
        raise HandoffAuthorityError("checkpoint source_parent_commit must be a full Git SHA")
    if source_parent != authority["delta_authority"]["activation_parent_commit"]:
        raise HandoffAuthorityError("checkpoint source parent must equal activation parent")
    handoff_file = _safe_path(
        repo_root,
        _string(manifest.get("handoff_path"), "handoff_path"),
        "checkpoint handoff",
    )
    text = handoff_file.read_text(encoding="utf-8")
    if shadow.sha256_text(text) != manifest.get("handoff_sha256"):
        raise HandoffAuthorityError("checkpoint handoff SHA-256 mismatch")
    registry_hash = _string(
        manifest.get("registry_sha256_for_provenance"),
        "registry_sha256_for_provenance",
    )
    if not shadow.SHA256_RE.fullmatch(registry_hash):
        raise HandoffAuthorityError("checkpoint registry provenance hash must be SHA-256")
    _safe_path(
        repo_root,
        _string(manifest.get("stage3_full_acceptance_report"), "stage3_full_acceptance_report"),
        "Stage 3 Full Acceptance report",
    )
    _safe_path(
        repo_root,
        _string(manifest.get("stage4b_cutover_audit_report"), "stage4b_cutover_audit_report"),
        "Stage 4B cutover audit report",
    )
    _string(manifest.get("created_at_utc"), "created_at_utc")
    if _git_show(repo_root, source_parent, HANDOFF_PATH) != text:
        raise HandoffAuthorityError("checkpoint bytes do not match source parent handoff")

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
    for protected in (manifest_path, handoff_file, repo_root / AUTHORITY_PATH):
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
    return manifest, text


def _first_parent_positions(repo_root: Path, activation: str, head: str) -> dict[str, int]:
    commits = _git_text(
        repo_root, "rev-list", "--first-parent", "--reverse", f"{activation}..{head}"
    ).splitlines()
    return {commit: index for index, commit in enumerate(commits)}


def _discover_v3_deltas(repo_root: Path, authority: dict[str, Any]) -> list[tuple[str, Path, dict[str, Any]]]:
    activation = authority["delta_authority"]["activation_parent_commit"]
    head = _git_text(repo_root, "rev-parse", "HEAD")
    positions = _first_parent_positions(repo_root, activation, head)
    records = []
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
        if first_add not in positions:
            raise HandoffAuthorityError(f"authoritative delta was not added after activation: {relative}")
        report = path.parent / REPORT_FILENAME
        if not report.is_file():
            raise HandoffAuthorityError(f"authoritative delta lacks materialization report: {relative}")
        report_touches = _git_text(
            repo_root,
            "log",
            "--format=%H",
            "--",
            _repo_relative(repo_root, report, "materialization report"),
        ).splitlines()
        if report_touches != [first_add]:
            raise HandoffAuthorityError(f"materialization report is not immutable: {report}")
        records.append((first_add, path, payload))
    records.sort(key=lambda item: positions[item[0]])
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


def materialize_all(repo_root: Path, authority: dict[str, Any]) -> tuple[str, list[str]]:
    _, text = _load_checkpoint(repo_root, authority)
    existing_ids = _existing_block_ids(text)
    update_ids: list[str] = []
    for _, path, delta in _discover_v3_deltas(repo_root, authority):
        report = _load_materialization_report(path.parent / REPORT_FILENAME, delta)
        if report.get("current_handoff_before_sha256") != shadow.sha256_text(text):
            raise HandoffAuthorityError(
                f"materialization report before-hash breaks replay chain: {path}"
            )
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
    return text, update_ids


def verify_current_state(repo_root: Path, *, check_stage4a: bool = True) -> dict[str, Any]:
    authority = load_authority(repo_root)
    if authority["mode"] == "manual":
        return {
            "status": "PASS",
            "mode": "manual",
            "manual_handoff_authoritative": True,
            "authority_cutover_allowed": False,
        }
    expected, update_ids = materialize_all(repo_root, authority)
    current = (repo_root / HANDOFF_PATH).read_text(encoding="utf-8")
    if current != expected:
        raise HandoffAuthorityError("tracked handoff is stale or was directly edited")
    if check_stage4a:
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
    return {
        "status": "PASS",
        "mode": "delta",
        "manual_handoff_authoritative": False,
        "authoritative_delta_count": len(update_ids),
        "authoritative_update_ids": update_ids,
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
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "verify":
            payload = verify_current_state(
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
