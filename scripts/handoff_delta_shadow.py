#!/usr/bin/env python3
"""Deterministic Stage 3 HANDOFF_DELTA shadow renderer and validator.

The tool replays one append-oriented delta against its exact Git base, renders a
candidate handoff, and compares it with the still-authoritative manual handoff.
It is deliberately local, deterministic, and free of network or LLM calls.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml

GIT_SHA_RE = re.compile(r"[0-9a-f]{40}")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
UPDATE_ID_RE = re.compile(r"[A-Z0-9][A-Z0-9._-]{2,127}")
OP_ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]{2,127}")
BLOCK_ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]{2,127}")
HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*$", re.MULTILINE)
IDENTIFIER_RE = re.compile(
    r"\b(?:[A-Z][A-Z0-9]*-[A-Z0-9][A-Z0-9.-]*|GOV-[A-Z0-9][A-Z0-9-]*)\b"
)
SUPPORTED_OPERATIONS = {"replace_heading", "insert_after_heading", "append_to_section"}
DELTA_FILENAME = "HANDOFF_DELTA.yaml"
REPORT_FILENAME = "SHADOW_REPORT.json"
FULL_REPORT_FILENAME = "FULL_ACCEPTANCE_REPORT.json"
CURRENT_DELTA_SCHEMA_VERSION = 2
LEGACY_DELTA_SCHEMA_VERSION = 1
CURRENT_REPORT_SCHEMA_VERSION = 2


class HandoffDeltaError(ValueError):
    """Raised when a delta, replay, comparison, or safety invariant fails."""


@dataclass(frozen=True)
class Heading:
    level: int
    title: str
    path: tuple[str, ...]
    start: int
    line_end: int
    section_end: int


@dataclass(frozen=True)
class RenderResult:
    text: str
    affected_selectors: tuple[str, ...]
    operation_count: int


@dataclass(frozen=True)
class CheckResult:
    report: dict[str, Any]
    candidate: str


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise HandoffDeltaError(f"{label} must be a mapping")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise HandoffDeltaError(f"{label} must be a list")
    return value


def require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HandoffDeltaError(f"{label} must be a non-empty string")
    return value.strip()


def reject_unknown_keys(mapping: dict[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise HandoffDeltaError(f"{label} contains unknown keys: {unknown}")


def load_yaml(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise HandoffDeltaError(f"Could not read {label} {path}: {exc}") from exc
    return require_mapping(payload, label)


def git(repo_root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and result.returncode != 0:
        raise HandoffDeltaError(
            f"git {' '.join(args)} failed ({result.returncode}): {result.stderr.strip()}"
        )
    return result


def git_text(repo_root: Path, *args: str) -> str:
    return git(repo_root, *args).stdout.strip()


def git_show_text(repo_root: Path, commit: str, relative_path: str) -> str:
    result = git(repo_root, "show", f"{commit}:{relative_path}")
    return result.stdout


def safe_repo_file(repo_root: Path, relative_value: str, label: str) -> Path:
    relative = Path(relative_value)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise HandoffDeltaError(f"{label} must be a safe repository-relative path")
    current = repo_root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise HandoffDeltaError(f"{label} may not traverse a symlink: {relative_value}")
    resolved = (repo_root / relative).resolve()
    try:
        resolved.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise HandoffDeltaError(f"{label} escapes repository: {relative_value}") from exc
    if not resolved.is_file():
        raise HandoffDeltaError(f"{label} is missing or not a regular file: {relative_value}")
    return resolved


def parse_headings(text: str) -> list[Heading]:
    raw: list[tuple[int, str, tuple[str, ...], int, int]] = []
    stack: list[str] = []
    for match in HEADING_RE.finditer(text):
        level = len(match.group(1))
        title = match.group(2)
        while len(stack) >= level:
            stack.pop()
        while len(stack) < level - 1:
            stack.append("")
        stack.append(title)
        line_end = text.find("\n", match.end())
        if line_end == -1:
            line_end = len(text)
        else:
            line_end += 1
        raw.append((level, title, tuple(stack), match.start(), line_end))
    headings: list[Heading] = []
    for index, item in enumerate(raw):
        level, title, path, start, line_end = item
        section_end = len(text)
        for following in raw[index + 1 :]:
            if following[0] <= level:
                section_end = following[3]
                break
        headings.append(
            Heading(
                level=level,
                title=title,
                path=path,
                start=start,
                line_end=line_end,
                section_end=section_end,
            )
        )
    return headings


def require_heading(text: str, path_value: Any, label: str) -> Heading:
    path_list = require_list(path_value, f"{label}.heading_path")
    if not path_list or not all(isinstance(item, str) and item for item in path_list):
        raise HandoffDeltaError(f"{label}.heading_path must contain non-empty strings")
    path = tuple(path_list)
    matches = [heading for heading in parse_headings(text) if heading.path == path]
    if len(matches) != 1:
        raise HandoffDeltaError(
            f"{label}.heading_path must resolve uniquely; path={path!r}, matches={len(matches)}"
        )
    return matches[0]


def marker_pair(location: str, block_id: str) -> tuple[str, str]:
    if location not in {"after_heading", "section_end"}:
        raise AssertionError(location)
    return (
        f"<!-- HANDOFF-DELTA-BLOCK:{location}:{block_id}:START -->",
        f"<!-- HANDOFF-DELTA-BLOCK:{location}:{block_id}:END -->",
    )


def normalize_block_content(content: str) -> str:
    return content.strip("\n") + "\n"


def block_text(location: str, block_id: str, content: str) -> str:
    start, end = marker_pair(location, block_id)
    return f"{start}\n{normalize_block_content(content)}{end}\n"


def find_existing_block(
    text: str, location: str, block_id: str
) -> tuple[int, int, str] | None:
    start_marker, end_marker = marker_pair(location, block_id)
    start = text.find(start_marker)
    if start == -1:
        return None
    second = text.find(start_marker, start + 1)
    if second != -1:
        raise HandoffDeltaError(f"Duplicate block marker found for {block_id}")
    end_start = text.find(end_marker, start)
    if end_start == -1:
        raise HandoffDeltaError(f"Unclosed block marker found for {block_id}")
    end = end_start + len(end_marker)
    if end < len(text) and text[end] == "\n":
        end += 1
    return start, end, text[start:end]


def sorted_block_cluster(location: str, blocks: dict[str, str]) -> str:
    return (
        "\n".join(
            block_text(location, block_id, blocks[block_id]).rstrip("\n")
            for block_id in sorted(blocks)
        )
        + "\n"
    )


def collect_cluster(
    text: str, start: int, end: int, location: str
) -> tuple[int, int, dict[str, str]]:
    region = text[start:end]
    pattern = re.compile(
        rf"(?:\n*)<!-- HANDOFF-DELTA-BLOCK:{location}:([a-z0-9][a-z0-9._-]{{2,127}}):START -->\n"
        r"(.*?)"
        rf"<!-- HANDOFF-DELTA-BLOCK:{location}:\1:END -->\n?",
        re.DOTALL,
    )
    blocks: dict[str, str] = {}
    first: int | None = None
    last: int | None = None
    for match in pattern.finditer(region):
        if first is None:
            first = match.start()
        last = match.end()
        block_id = match.group(1)
        content = match.group(2).rstrip("\n")
        if block_id in blocks:
            raise HandoffDeltaError(f"Duplicate block ID in cluster: {block_id}")
        blocks[block_id] = content
    if first is None or last is None:
        return start, start, {}
    return start + first, start + last, blocks


def insert_canonical_block(
    text: str,
    *,
    heading: Heading,
    location: str,
    block_id: str,
    content: str,
) -> str:
    existing = find_existing_block(text, location, block_id)
    expected = block_text(location, block_id, content)
    if existing is not None:
        if existing[2].strip() != expected.strip():
            raise HandoffDeltaError(f"Block {block_id} already exists with different content")
        return text

    if location == "after_heading":
        cluster_start = heading.line_end
        cluster_limit = heading.section_end
    elif location == "section_end":
        cluster_start = heading.line_end
        cluster_limit = heading.section_end
    else:
        raise AssertionError(location)

    cluster_first, cluster_last, blocks = collect_cluster(
        text, cluster_start, cluster_limit, location
    )
    blocks[block_id] = content
    rendered = sorted_block_cluster(location, blocks)
    if location == "after_heading":
        if blocks and cluster_first == cluster_start and cluster_last > cluster_first:
            replacement_start, replacement_end = cluster_first, cluster_last
        else:
            replacement_start = replacement_end = heading.line_end
        prefix = "\n" if replacement_start == replacement_end else ""
        return text[:replacement_start] + prefix + rendered + text[replacement_end:]

    # Append-only blocks are kept at the end of the section. Existing marked blocks
    # anywhere in the section are normalized into one deterministic terminal cluster.
    if cluster_last > cluster_first:
        without = text[:cluster_first] + text[cluster_last:]
        reparsed = require_heading(without, list(heading.path), "append_to_section")
        insertion = reparsed.section_end
        separator = "" if without[:insertion].endswith("\n\n") else "\n"
        terminal_rendered = rendered.rstrip("\n") + "\n\n"
        return without[:insertion] + separator + terminal_rendered + without[insertion:]
    insertion = heading.section_end
    separator = "" if text[:insertion].endswith("\n\n") else "\n"
    terminal_rendered = rendered.rstrip("\n") + "\n\n"
    return text[:insertion] + separator + terminal_rendered + text[insertion:]


def validate_evidence_paths_shape(value: Any, label: str) -> list[str]:
    evidence = require_list(value, label)
    if not evidence or not all(isinstance(path, str) and path for path in evidence):
        raise HandoffDeltaError(f"{label} must contain non-empty repository-relative paths")
    return evidence


def validate_registry_change_shape(item: dict[str, Any], index: int) -> None:
    label = f"registry.changes[{index}]"
    kind = require_string(item.get("kind"), f"{label}.kind")
    common = {"change_id", "kind", "entity_id", "evidence"}
    change_id = require_string(item.get("change_id"), f"{label}.change_id")
    if not OP_ID_RE.fullmatch(change_id):
        raise HandoffDeltaError(f"Invalid registry change_id: {change_id!r}")
    require_string(item.get("entity_id"), f"{label}.entity_id")
    validate_evidence_paths_shape(item.get("evidence"), f"{label}.evidence")
    if kind == "transition":
        allowed = common | {"field_path", "machine", "from", "to"}
        require_string(item.get("machine"), f"{label}.machine")
        require_string(item.get("from"), f"{label}.from")
        require_string(item.get("to"), f"{label}.to")
        field_path = require_list(item.get("field_path"), f"{label}.field_path")
        if not field_path or not all(isinstance(part, str) and part for part in field_path):
            raise HandoffDeltaError(f"{label}.field_path must contain non-empty strings")
    elif kind == "add_entity":
        allowed = common
    elif kind == "update_field":
        allowed = common | {"field_path", "from", "to", "reason"}
        field_path = require_list(item.get("field_path"), f"{label}.field_path")
        if not field_path or not all(isinstance(part, str) and part for part in field_path):
            raise HandoffDeltaError(f"{label}.field_path must contain non-empty strings")
        if "from" not in item or "to" not in item:
            raise HandoffDeltaError(f"{label} must declare exact from/to values")
        require_string(item.get("reason"), f"{label}.reason")
    else:
        raise HandoffDeltaError(f"Unsupported registry change kind: {kind}")
    reject_unknown_keys(item, allowed, label)


def validate_delta_shape(
    delta: dict[str, Any],
    delta_path: Path,
    policy: dict[str, Any] | None = None,
) -> None:
    reject_unknown_keys(
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
    schema_version = delta.get("schema_version")
    if schema_version not in {LEGACY_DELTA_SCHEMA_VERSION, CURRENT_DELTA_SCHEMA_VERSION}:
        raise HandoffDeltaError(
            f"delta.schema_version must be {LEGACY_DELTA_SCHEMA_VERSION} or "
            f"{CURRENT_DELTA_SCHEMA_VERSION}"
        )
    update_id = require_string(delta.get("update_id"), "delta.update_id")
    if not UPDATE_ID_RE.fullmatch(update_id):
        raise HandoffDeltaError(f"Invalid update_id: {update_id!r}")
    if delta_path.parent.name != update_id or delta_path.name != DELTA_FILENAME:
        raise HandoffDeltaError(
            f"Delta path must be docs/handoff_deltas/{update_id}/{DELTA_FILENAME}"
        )
    if delta.get("mode") != "shadow":
        raise HandoffDeltaError("delta.mode must remain shadow during Stage 3")
    if delta.get("renderer_version") != 1:
        raise HandoffDeltaError("renderer_version must be 1")
    if policy is not None:
        schema_policy = require_mapping(policy.get("delta_schema"), "policy.delta_schema")
        accepted = require_list(schema_policy.get("accepted_versions"), "policy.delta_schema.accepted_versions")
        if schema_version not in accepted:
            raise HandoffDeltaError(f"Delta schema version {schema_version} is not accepted")
        current = schema_policy.get("current_version")
        legacy_ids = set(
            require_list(
                schema_policy.get("legacy_schema1_update_ids", []),
                "policy.delta_schema.legacy_schema1_update_ids",
            )
        )
        if schema_version != current and update_id not in legacy_ids:
            raise HandoffDeltaError(
                f"New delta {update_id} must use current schema_version={current}; "
                f"legacy schema is restricted to registered historical update IDs"
            )

    base = require_mapping(delta.get("base"), "delta.base")
    reject_unknown_keys(base, {"commit", "handoff_sha256", "registry_sha256"}, "delta.base")
    commit = require_string(base.get("commit"), "delta.base.commit")
    if not GIT_SHA_RE.fullmatch(commit):
        raise HandoffDeltaError("delta.base.commit must be a full lowercase Git SHA")
    for key in ("handoff_sha256", "registry_sha256"):
        value = require_string(base.get(key), f"delta.base.{key}")
        if not SHA256_RE.fullmatch(value):
            raise HandoffDeltaError(f"delta.base.{key} must be 64 lowercase hex characters")

    operations = require_list(delta.get("operations"), "delta.operations")
    if not operations:
        raise HandoffDeltaError("delta.operations must not be empty")
    operation_ids: set[str] = set()
    block_ids: set[str] = set()
    for index, raw in enumerate(operations):
        op = require_mapping(raw, f"delta.operations[{index}]")
        operation_id = require_string(op.get("operation_id"), f"operation[{index}].operation_id")
        if not OP_ID_RE.fullmatch(operation_id):
            raise HandoffDeltaError(f"Invalid operation_id: {operation_id!r}")
        if operation_id in operation_ids:
            raise HandoffDeltaError(f"Duplicate operation_id: {operation_id}")
        operation_ids.add(operation_id)
        op_type = require_string(op.get("op"), f"operation[{index}].op")
        if op_type not in SUPPORTED_OPERATIONS:
            raise HandoffDeltaError(f"Unsupported operation: {op_type}")
        common = {"operation_id", "op", "heading_path"}
        if op_type == "replace_heading":
            allowed = common | {"new_heading", "reason"}
            require_string(op.get("new_heading"), f"operation[{index}].new_heading")
            require_string(op.get("reason"), f"operation[{index}].reason")
        else:
            allowed = common | {"block_id", "content"}
            block_id = require_string(op.get("block_id"), f"operation[{index}].block_id")
            if not BLOCK_ID_RE.fullmatch(block_id):
                raise HandoffDeltaError(f"Invalid block_id: {block_id!r}")
            if block_id in block_ids:
                raise HandoffDeltaError(f"Duplicate block_id: {block_id}")
            block_ids.add(block_id)
            require_string(op.get("content"), f"operation[{index}].content")
        reject_unknown_keys(op, allowed, f"delta.operations[{index}]")
        path = require_list(op.get("heading_path"), f"operation[{index}].heading_path")
        if not path or not all(isinstance(item, str) and item for item in path):
            raise HandoffDeltaError(f"operation[{index}].heading_path must be non-empty strings")

    registry = require_mapping(delta.get("registry"), "delta.registry")
    mode = registry.get("mode")
    if mode not in {"unchanged", "expected_after"}:
        raise HandoffDeltaError("delta.registry.mode must be unchanged or expected_after")
    expected_after = registry.get("expected_after_sha256")
    if schema_version == LEGACY_DELTA_SCHEMA_VERSION:
        reject_unknown_keys(
            registry, {"mode", "expected_after_sha256", "transitions"}, "delta.registry"
        )
        transitions = require_list(registry.get("transitions", []), "delta.registry.transitions")
        if mode == "unchanged":
            if expected_after is not None or transitions:
                raise HandoffDeltaError("unchanged registry mode forbids after hash and transitions")
        else:
            value = require_string(expected_after, "delta.registry.expected_after_sha256")
            if not SHA256_RE.fullmatch(value):
                raise HandoffDeltaError("registry expected_after_sha256 must be lowercase SHA-256")
            if not transitions:
                raise HandoffDeltaError("expected_after registry mode requires transitions")
            for index, transition in enumerate(transitions):
                item = require_mapping(transition, f"registry.transitions[{index}]")
                reject_unknown_keys(
                    item,
                    {
                        "assertion_id",
                        "entity_id",
                        "field_path",
                        "machine",
                        "from",
                        "to",
                        "evidence",
                    },
                    f"registry.transitions[{index}]",
                )
                for key in ("assertion_id", "entity_id", "machine", "from", "to"):
                    require_string(item.get(key), f"registry.transitions[{index}].{key}")
                field_path = require_list(
                    item.get("field_path"), f"registry.transitions[{index}].field_path"
                )
                if not field_path or not all(
                    isinstance(part, str) and part for part in field_path
                ):
                    raise HandoffDeltaError(
                        "registry transition field_path must be non-empty strings"
                    )
                validate_evidence_paths_shape(
                    item.get("evidence"), f"registry.transitions[{index}].evidence"
                )
    else:
        reject_unknown_keys(
            registry, {"mode", "expected_after_sha256", "changes"}, "delta.registry"
        )
        changes = require_list(registry.get("changes", []), "delta.registry.changes")
        if mode == "unchanged":
            if expected_after is not None or changes:
                raise HandoffDeltaError("unchanged registry mode forbids after hash and changes")
        else:
            value = require_string(expected_after, "delta.registry.expected_after_sha256")
            if not SHA256_RE.fullmatch(value):
                raise HandoffDeltaError("registry expected_after_sha256 must be lowercase SHA-256")
            if not changes:
                raise HandoffDeltaError("schema v2 expected_after registry mode requires changes")
            seen_change_ids: set[str] = set()
            for index, raw in enumerate(changes):
                item = require_mapping(raw, f"registry.changes[{index}]")
                validate_registry_change_shape(item, index)
                change_id = item["change_id"]
                if change_id in seen_change_ids:
                    raise HandoffDeltaError(f"Duplicate registry change_id: {change_id}")
                seen_change_ids.add(change_id)

    expected = require_mapping(delta.get("expected"), "delta.expected")
    reject_unknown_keys(expected, {"candidate_sha256", "manual_sha256"}, "delta.expected")
    for key in ("candidate_sha256", "manual_sha256"):
        value = require_string(expected.get(key), f"delta.expected.{key}")
        if not SHA256_RE.fullmatch(value):
            raise HandoffDeltaError(f"delta.expected.{key} must be lowercase SHA-256")

def load_policy(repo_root: Path) -> dict[str, Any]:
    policy_path = repo_root / "docs" / "handoff_delta_policy.yaml"
    policy = load_yaml(policy_path, "handoff delta policy")
    if policy.get("schema_version") != 1 or policy.get("policy_id") != "GOV-HANDOFF-INDEX-01":
        raise HandoffDeltaError("Unsupported handoff delta policy")
    if policy.get("mode") != "shadow":
        raise HandoffDeltaError("handoff delta policy must remain in shadow mode")
    safety = require_mapping(policy.get("safety"), "policy.safety")
    if safety.get("allow_network") is not False or safety.get("allow_llm_blocking_gate") is not False:
        raise HandoffDeltaError("Fast Gate must not use network or an LLM blocking oracle")
    if safety.get("maximum_deltas_per_update") != 1:
        raise HandoffDeltaError("Version 1 requires exactly one delta per relevant update")
    schema_policy = require_mapping(policy.get("delta_schema"), "policy.delta_schema")
    accepted = require_list(
        schema_policy.get("accepted_versions"), "policy.delta_schema.accepted_versions"
    )
    if accepted != [LEGACY_DELTA_SCHEMA_VERSION, CURRENT_DELTA_SCHEMA_VERSION]:
        raise HandoffDeltaError(
            "policy.delta_schema.accepted_versions must preserve legacy v1 and current v2"
        )
    if schema_policy.get("current_version") != CURRENT_DELTA_SCHEMA_VERSION:
        raise HandoffDeltaError(
            f"policy.delta_schema.current_version must be {CURRENT_DELTA_SCHEMA_VERSION}"
        )
    legacy_ids = require_list(
        schema_policy.get("legacy_schema1_update_ids"),
        "policy.delta_schema.legacy_schema1_update_ids",
    )
    if len(set(legacy_ids)) != len(legacy_ids):
        raise HandoffDeltaError("policy.delta_schema.legacy_schema1_update_ids contains duplicates")
    supported = set(require_list(policy.get("supported_operations"), "policy.supported_operations"))
    if supported != SUPPORTED_OPERATIONS:
        raise HandoffDeltaError("policy supported_operations drifted from renderer implementation")
    return policy


def render(base_text: str, operations: Sequence[dict[str, Any]]) -> RenderResult:
    text = base_text
    affected: list[str] = []
    for index, op in enumerate(operations):
        op_type = op["op"]
        label = f"operation[{index}] {op['operation_id']}"
        if op_type == "replace_heading":
            old_path = tuple(op["heading_path"])
            new_title = op["new_heading"]
            old_matches = [heading for heading in parse_headings(text) if heading.path == old_path]
            new_path = old_path[:-1] + (new_title,)
            new_matches = [heading for heading in parse_headings(text) if heading.path == new_path]
            if len(old_matches) == 1 and not new_matches:
                heading = old_matches[0]
                affected.append(" / ".join(heading.path))
                old_line = "#" * heading.level + " " + heading.title
                new_line = "#" * heading.level + " " + new_title
                text = text[: heading.start] + new_line + text[heading.start + len(old_line) :]
                continue
            if not old_matches and len(new_matches) == 1:
                affected.append(" / ".join(new_path))
                continue
            raise HandoffDeltaError(
                f"{label} heading rename is ambiguous or conflicted; "
                f"old_matches={len(old_matches)}, new_matches={len(new_matches)}"
            )
        heading = require_heading(text, op["heading_path"], label)
        affected.append(" / ".join(heading.path))
        if op_type == "insert_after_heading":
            text = insert_canonical_block(
                text,
                heading=heading,
                location="after_heading",
                block_id=op["block_id"],
                content=op["content"],
            )
        elif op_type == "append_to_section":
            text = insert_canonical_block(
                text,
                heading=heading,
                location="section_end",
                block_id=op["block_id"],
                content=op["content"],
            )
        else:  # guarded by shape validation
            raise AssertionError(op_type)
    return RenderResult(text=text, affected_selectors=tuple(affected), operation_count=len(operations))


def verify_history_preservation(base_text: str, candidate_text: str, operations: Sequence[dict[str, Any]]) -> None:
    base_ids = set(IDENTIFIER_RE.findall(base_text))
    candidate_ids = set(IDENTIFIER_RE.findall(candidate_text))
    missing_ids = sorted(base_ids - candidate_ids)
    if missing_ids:
        raise HandoffDeltaError(f"Historical identifiers were removed: {missing_ids[:20]}")

    allowed_renames = {
        tuple(op["heading_path"]): op["new_heading"]
        for op in operations
        if op["op"] == "replace_heading"
    }
    candidate_titles = [heading.title for heading in parse_headings(candidate_text)]
    candidate_counts: dict[str, int] = {}
    for title in candidate_titles:
        candidate_counts[title] = candidate_counts.get(title, 0) + 1
    for heading in parse_headings(base_text):
        replacement = allowed_renames.get(heading.path)
        expected_title = replacement or heading.title
        if candidate_counts.get(expected_title, 0) <= 0:
            raise HandoffDeltaError(f"Historical heading disappeared: {heading.path!r}")
        candidate_counts[expected_title] -= 1


def find_experiment(registry: dict[str, Any], entity_id: str) -> dict[str, Any]:
    rows = registry.get("experiments")
    if not isinstance(rows, list):
        raise HandoffDeltaError("registry.experiments must be a list")
    matches = [row for row in rows if isinstance(row, dict) and row.get("id") == entity_id]
    if len(matches) != 1:
        raise HandoffDeltaError(f"Experiment {entity_id!r} must resolve uniquely in registry")
    return matches[0]


def nested_get(mapping: dict[str, Any], field_path: Sequence[str], label: str) -> Any:
    current: Any = mapping
    for part in field_path:
        if not isinstance(current, dict) or part not in current:
            raise HandoffDeltaError(f"{label} missing field path component {part!r}")
        current = current[part]
    return current


def registry_experiments(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = registry.get("experiments")
    if not isinstance(rows, list):
        raise HandoffDeltaError("registry.experiments must be a list")
    result: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise HandoffDeltaError(f"registry.experiments[{index}] must be a mapping")
        entity_id = require_string(row.get("id"), f"registry.experiments[{index}].id")
        if entity_id in result:
            raise HandoffDeltaError(f"Duplicate registry experiment ID: {entity_id}")
        result[entity_id] = row
    return result


def diff_values(before: Any, after: Any, path: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    if before == after:
        return []
    if isinstance(before, dict) and isinstance(after, dict):
        rows: list[dict[str, Any]] = []
        for key in sorted(set(before) | set(after)):
            child_path = path + (str(key),)
            if key not in before:
                rows.append(
                    {
                        "field_path": list(child_path),
                        "from": None,
                        "to": after[key],
                        "change_type": "added_key",
                    }
                )
            elif key not in after:
                rows.append(
                    {
                        "field_path": list(child_path),
                        "from": before[key],
                        "to": None,
                        "change_type": "removed_key",
                    }
                )
            else:
                rows.extend(diff_values(before[key], after[key], child_path))
        return rows
    return [
        {
            "field_path": list(path),
            "from": before,
            "to": after,
            "change_type": "value_change",
        }
    ]


def compute_registry_diff(
    base_registry: dict[str, Any], current_registry: dict[str, Any]
) -> dict[str, Any]:
    base_entities = registry_experiments(base_registry)
    current_entities = registry_experiments(current_registry)
    added = sorted(set(current_entities) - set(base_entities))
    removed = sorted(set(base_entities) - set(current_entities))
    changed: list[dict[str, Any]] = []
    for entity_id in sorted(set(base_entities) & set(current_entities)):
        for row in diff_values(base_entities[entity_id], current_entities[entity_id]):
            if row["field_path"] == ["id"]:
                continue
            changed.append({"entity_id": entity_id, **row})

    base_top = {key: value for key, value in base_registry.items() if key != "experiments"}
    current_top = {key: value for key, value in current_registry.items() if key != "experiments"}
    for row in diff_values(base_top, current_top):
        changed.append({"entity_id": "__registry__", **row})
    changed.sort(key=lambda item: (item["entity_id"], tuple(item["field_path"])))
    return {"added_entities": added, "removed_entities": removed, "changed_fields": changed}


def evidence_reports(repo_root: Path, values: Sequence[str], label: str) -> list[dict[str, str]]:
    reports: list[dict[str, str]] = []
    for relative in values:
        path = safe_repo_file(repo_root, relative, label)
        reports.append({"path": relative, "sha256": sha256_file(path)})
    return reports


def validate_state_transition(
    machines: dict[str, Any], machine_name: str, before: Any, after: Any, entity_id: str
) -> None:
    machine = require_mapping(machines.get(machine_name), f"state machine {machine_name}")
    states = require_list(machine.get("states"), f"state machine {machine_name}.states")
    transitions = require_mapping(
        machine.get("transitions"), f"state machine {machine_name}.transitions"
    )
    if before not in states or after not in states:
        raise HandoffDeltaError(
            f"Registry transition uses a state outside machine {machine_name}: {before!r}->{after!r}"
        )
    allowed = transitions.get(before)
    if not isinstance(allowed, list) or after not in allowed:
        raise HandoffDeltaError(
            f"Illegal {machine_name} transition for {entity_id}: {before!r}->{after!r}"
        )


def validate_registry(
    repo_root: Path,
    delta: dict[str, Any],
    base_registry_text: str,
    current_registry_text: str,
) -> dict[str, Any]:
    spec = delta["registry"]
    schema_version = delta["schema_version"]
    base_hash = sha256_text(base_registry_text)
    current_hash = sha256_text(current_registry_text)
    if base_hash != delta["base"]["registry_sha256"]:
        raise HandoffDeltaError("Base registry SHA-256 does not match delta")
    if spec["mode"] == "unchanged":
        if current_hash != base_hash:
            raise HandoffDeltaError("Registry changed but delta.registry.mode is unchanged")
        return {
            "assertions": [],
            "legacy_transition_assertions": [],
            "coverage": {
                "mode": "exact_unchanged",
                "added_entities": [],
                "removed_entities": [],
                "changed_field_count": 0,
                "fully_declared": True,
            },
        }
    if current_hash != spec["expected_after_sha256"]:
        raise HandoffDeltaError("Current registry SHA-256 does not match expected after-image")

    state_payload = load_yaml(
        repo_root / "docs" / "handoff_delta_state_machines.yaml", "state machines"
    )
    machines = require_mapping(state_payload.get("machines"), "state_machines.machines")
    base_registry = require_mapping(yaml.safe_load(base_registry_text), "base registry")
    current_registry = require_mapping(yaml.safe_load(current_registry_text), "current registry")
    actual = compute_registry_diff(base_registry, current_registry)
    if actual["removed_entities"]:
        raise HandoffDeltaError(
            "Schema v2 forbids destructive registry entity removal: "
            f"{actual['removed_entities']}"
        )
    removed_fields = [
        item
        for item in actual["changed_fields"]
        if item.get("change_type") == "removed_key"
    ]
    if removed_fields:
        preview = [
            f"{item['entity_id']}:{item['field_path']}" for item in removed_fields[:20]
        ]
        raise HandoffDeltaError(
            "Schema v2 forbids destructive registry field removal: " + ", ".join(preview)
        )

    assertions: list[dict[str, Any]] = []
    if schema_version == LEGACY_DELTA_SCHEMA_VERSION:
        for transition in spec["transitions"]:
            entity_id = transition["entity_id"]
            base_entity = find_experiment(base_registry, entity_id)
            current_entity = find_experiment(current_registry, entity_id)
            field_path = transition["field_path"]
            before = nested_get(base_entity, field_path, f"base experiment {entity_id}")
            after = nested_get(current_entity, field_path, f"current experiment {entity_id}")
            if before != transition["from"] or after != transition["to"]:
                raise HandoffDeltaError(
                    f"Registry transition {transition['assertion_id']} does not match actual values: "
                    f"{before!r}->{after!r}"
                )
            validate_state_transition(machines, transition["machine"], before, after, entity_id)
            assertions.append(
                {
                    "change_id": transition["assertion_id"],
                    "kind": "transition",
                    "entity_id": entity_id,
                    "field_path": field_path,
                    "machine": transition["machine"],
                    "from": before,
                    "to": after,
                    "evidence": evidence_reports(
                        repo_root,
                        transition["evidence"],
                        f"transition {transition['assertion_id']} evidence",
                    ),
                }
            )
        return {
            "assertions": assertions,
            "legacy_transition_assertions": [
                {
                    "assertion_id": item["change_id"],
                    "entity_id": item["entity_id"],
                    "field_path": item["field_path"],
                    "machine": item["machine"],
                    "from": item["from"],
                    "to": item["to"],
                    "evidence": item["evidence"],
                }
                for item in assertions
            ],
            "coverage": {
                "mode": "legacy_schema1_partial_assertions",
                "added_entities": actual["added_entities"],
                "removed_entities": actual["removed_entities"],
                "changed_field_count": len(actual["changed_fields"]),
                "declared_assertion_count": len(assertions),
                "fully_declared": False,
            },
        }

    actual_added = set(actual["added_entities"])
    actual_fields = {
        (item["entity_id"], tuple(item["field_path"])): item
        for item in actual["changed_fields"]
    }
    declared_added: set[str] = set()
    declared_fields: set[tuple[str, tuple[str, ...]]] = set()
    for change in spec["changes"]:
        kind = change["kind"]
        entity_id = change["entity_id"]
        report: dict[str, Any] = {
            "change_id": change["change_id"],
            "kind": kind,
            "entity_id": entity_id,
            "evidence": evidence_reports(
                repo_root,
                change["evidence"],
                f"registry change {change['change_id']} evidence",
            ),
        }
        if kind == "add_entity":
            if entity_id not in actual_added:
                raise HandoffDeltaError(
                    f"Registry add_entity {change['change_id']} does not match an actual addition: "
                    f"{entity_id}"
                )
            if entity_id in declared_added:
                raise HandoffDeltaError(f"Registry addition declared twice: {entity_id}")
            declared_added.add(entity_id)
        else:
            field_path = tuple(change["field_path"])
            key = (entity_id, field_path)
            actual_row = actual_fields.get(key)
            if actual_row is None:
                raise HandoffDeltaError(
                    f"Registry change {change['change_id']} does not match an actual changed field: "
                    f"{entity_id}:{list(field_path)}"
                )
            if key in declared_fields:
                raise HandoffDeltaError(
                    f"Registry field change declared twice: {entity_id}:{list(field_path)}"
                )
            if actual_row["from"] != change["from"] or actual_row["to"] != change["to"]:
                raise HandoffDeltaError(
                    f"Registry change {change['change_id']} from/to does not match actual values: "
                    f"{actual_row['from']!r}->{actual_row['to']!r}"
                )
            declared_fields.add(key)
            report.update(
                {
                    "field_path": list(field_path),
                    "from": actual_row["from"],
                    "to": actual_row["to"],
                }
            )
            if kind == "transition":
                validate_state_transition(
                    machines, change["machine"], actual_row["from"], actual_row["to"], entity_id
                )
                report["machine"] = change["machine"]
            else:
                report["reason"] = change["reason"]
        assertions.append(report)

    missing_added = sorted(actual_added - declared_added)
    extra_added = sorted(declared_added - actual_added)
    missing_fields = sorted(
        (entity_id, list(path)) for entity_id, path in set(actual_fields) - declared_fields
    )
    extra_fields = sorted(
        (entity_id, list(path)) for entity_id, path in declared_fields - set(actual_fields)
    )
    if missing_added or extra_added or missing_fields or extra_fields:
        raise HandoffDeltaError(
            "Schema v2 registry change coverage is incomplete; "
            f"missing_added={missing_added} extra_added={extra_added} "
            f"missing_fields={missing_fields[:20]} extra_fields={extra_fields[:20]}"
        )
    return {
        "assertions": assertions,
        "legacy_transition_assertions": [
            {
                "assertion_id": item["change_id"],
                "entity_id": item["entity_id"],
                "field_path": item["field_path"],
                "machine": item["machine"],
                "from": item["from"],
                "to": item["to"],
                "evidence": item["evidence"],
            }
            for item in assertions
            if item["kind"] == "transition"
        ],
        "coverage": {
            "mode": "schema2_complete",
            "added_entities": sorted(actual_added),
            "removed_entities": [],
            "changed_field_count": len(actual_fields),
            "declared_assertion_count": len(assertions),
            "fully_declared": True,
        },
    }

def classify_observation(update_id: str) -> str:
    if update_id.startswith("GOV-STAGE3-SHADOW-BOOTSTRAP-"):
        return "bootstrap"
    return "real"


def check_delta(
    repo_root: Path,
    delta_path: Path,
    *,
    enforce_performance: bool = True,
    target_commit: str | None = None,
) -> CheckResult:
    started = time.perf_counter()
    repo_root = repo_root.resolve()
    delta_path = delta_path.resolve()
    try:
        delta_path.relative_to(repo_root)
    except ValueError as exc:
        raise HandoffDeltaError("Delta path must be inside repository") from exc

    policy_start = time.perf_counter()
    policy = load_policy(repo_root)
    delta = load_yaml(delta_path, "handoff delta")
    validate_delta_shape(delta, delta_path, policy)
    policy_ms = (time.perf_counter() - policy_start) * 1000

    base_start = time.perf_counter()
    base_commit = delta["base"]["commit"]
    git(repo_root, "cat-file", "-e", f"{base_commit}^{{commit}}")
    comparison_commit = target_commit or git_text(repo_root, "rev-parse", "HEAD")
    git(repo_root, "cat-file", "-e", f"{comparison_commit}^{{commit}}")
    git(repo_root, "merge-base", "--is-ancestor", base_commit, comparison_commit)
    handoff_path = require_string(policy.get("research_master"), "policy.research_master")
    registry_path = require_string(policy.get("registry"), "policy.registry")
    base_handoff = git_show_text(repo_root, base_commit, handoff_path)
    base_registry = git_show_text(repo_root, base_commit, registry_path)
    if sha256_text(base_handoff) != delta["base"]["handoff_sha256"]:
        raise HandoffDeltaError("Base handoff SHA-256 does not match delta")
    if sha256_text(base_registry) != delta["base"]["registry_sha256"]:
        raise HandoffDeltaError("Base registry SHA-256 does not match delta")
    base_ms = (time.perf_counter() - base_start) * 1000

    render_start = time.perf_counter()
    rendered = render(base_handoff, delta["operations"])
    replayed = render(rendered.text, delta["operations"])
    if replayed.text != rendered.text:
        raise HandoffDeltaError("Delta replay is not idempotent")
    verify_history_preservation(base_handoff, rendered.text, delta["operations"])
    render_ms = (time.perf_counter() - render_start) * 1000

    compare_start = time.perf_counter()
    if target_commit is None:
        manual_path = safe_repo_file(repo_root, handoff_path, "manual handoff")
        registry_file = safe_repo_file(repo_root, registry_path, "current registry")
        manual = manual_path.read_text(encoding="utf-8")
        current_registry = registry_file.read_text(encoding="utf-8")
        comparison_target = "working_tree"
        repository_commit = None
    else:
        manual = git_show_text(repo_root, target_commit, handoff_path)
        current_registry = git_show_text(repo_root, target_commit, registry_path)
        comparison_target = "repository_commit"
        repository_commit = target_commit
    candidate_hash = sha256_text(rendered.text)
    manual_hash = sha256_text(manual)
    if candidate_hash != delta["expected"]["candidate_sha256"]:
        raise HandoffDeltaError("Rendered candidate SHA-256 differs from delta.expected")
    if manual_hash != delta["expected"]["manual_sha256"]:
        raise HandoffDeltaError("Manual handoff SHA-256 differs from delta.expected")
    if rendered.text != manual:
        raise HandoffDeltaError("Shadow candidate does not exactly match the authoritative manual handoff")
    registry_result = validate_registry(repo_root, delta, base_registry, current_registry)
    compare_ms = (time.perf_counter() - compare_start) * 1000

    total_ms = (time.perf_counter() - started) * 1000
    fast_gate = require_mapping(policy.get("fast_gate"), "policy.fast_gate")
    hard_limit_ms = float(fast_gate.get("hard_timeout_seconds", 15)) * 1000
    target_ms = float(fast_gate.get("target_p95_seconds", 5)) * 1000
    if enforce_performance and total_ms > hard_limit_ms:
        raise HandoffDeltaError(
            f"Fast Gate exceeded hard performance limit: {total_ms:.1f}ms > {hard_limit_ms:.1f}ms"
        )

    report = {
        "schema_version": 1,
        "report_schema_version": CURRENT_REPORT_SCHEMA_VERSION,
        "policy_id": policy["policy_id"],
        "update_id": delta["update_id"],
        "observation_kind": classify_observation(delta["update_id"]),
        "mode": "shadow",
        "status": "PASS",
        "base_commit": base_commit,
        "validation_worktree_head": git_text(repo_root, "rev-parse", "HEAD"),
        "comparison_target": comparison_target,
        "repository_commit": repository_commit,
        "manual_handoff_authoritative": True,
        "candidate_replaced_authority": False,
        "candidate_sha256": candidate_hash,
        "manual_sha256": manual_hash,
        "base_handoff_sha256": sha256_text(base_handoff),
        "base_registry_sha256": sha256_text(base_registry),
        "current_registry_sha256": sha256_text(current_registry),
        "exact_manual_candidate_match": True,
        "idempotence_passed": True,
        "history_preservation_passed": True,
        "operation_count": rendered.operation_count,
        "affected_selectors": list(rendered.affected_selectors),
        "registry_change_assertions": registry_result["assertions"],
        "registry_change_coverage": registry_result["coverage"],
        "registry_transition_assertions": registry_result["legacy_transition_assertions"],
        "network_used": False,
        "llm_blocking_gate_used": False,
        "full_candidate_saved": False,
        "performance": {
            "policy_and_schema_ms": round(policy_ms, 3),
            "base_validation_ms": round(base_ms, 3),
            "render_and_idempotence_ms": round(render_ms, 3),
            "comparison_and_registry_ms": round(compare_ms, 3),
            "total_ms": round(total_ms, 3),
            "target_p95_ms": target_ms,
            "hard_limit_ms": hard_limit_ms,
            "within_target": total_ms <= target_ms,
            "within_hard_limit": total_ms <= hard_limit_ms,
        },
    }
    return CheckResult(report=report, candidate=rendered.text)


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HandoffDeltaError(f"Could not read {label} {path}: {exc}") from exc
    return require_mapping(payload, label)


def evidence_path_projection(value: Any, label: str) -> list[str]:
    rows = require_list(value, label)
    paths: list[str] = []
    for index, raw in enumerate(rows):
        item = require_mapping(raw, f"{label}[{index}]")
        reject_unknown_keys(item, {"path", "sha256"}, f"{label}[{index}]")
        path = require_string(item.get("path"), f"{label}[{index}].path")
        digest = require_string(item.get("sha256"), f"{label}[{index}].sha256")
        if not SHA256_RE.fullmatch(digest):
            raise HandoffDeltaError(f"{label}[{index}].sha256 must be lowercase SHA-256")
        paths.append(path)
    return paths


def registry_assertion_projection(value: Any, label: str) -> list[dict[str, Any]]:
    rows = require_list(value, label)
    projected: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        item = require_mapping(raw, f"{label}[{index}]")
        normalized = {key: item[key] for key in item if key != "evidence"}
        normalized["evidence"] = evidence_path_projection(
            item.get("evidence"), f"{label}[{index}].evidence"
        )
        projected.append(normalized)
    return projected


def validate_report_runtime_fields(report: dict[str, Any], label: str) -> float:
    validation_head = report.get("validation_worktree_head") or report.get("head_commit")
    if not isinstance(validation_head, str) or not GIT_SHA_RE.fullmatch(validation_head):
        raise HandoffDeltaError(
            f"{label} must record validation_worktree_head "
            "(or legacy head_commit) as a full Git SHA"
        )

    comparison_target = report.get("comparison_target")
    repository_commit = report.get("repository_commit")
    if comparison_target is not None and comparison_target not in {
        "working_tree",
        "repository_commit",
    }:
        raise HandoffDeltaError(f"{label}.comparison_target is invalid")
    if repository_commit is not None and (
        not isinstance(repository_commit, str) or not GIT_SHA_RE.fullmatch(repository_commit)
    ):
        raise HandoffDeltaError(f"{label}.repository_commit must be null or a full Git SHA")

    performance = require_mapping(report.get("performance"), f"{label}.performance")
    values: dict[str, float] = {}
    for key in ("total_ms", "target_p95_ms", "hard_limit_ms"):
        raw = performance.get(key)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise HandoffDeltaError(f"{label}.performance.{key} must be numeric")
        values[key] = float(raw)
        if not math.isfinite(values[key]):
            raise HandoffDeltaError(f"{label}.performance.{key} must be finite")
    if values["total_ms"] < 0 or values["target_p95_ms"] <= 0:
        raise HandoffDeltaError(f"{label}.performance contains invalid values")
    if values["hard_limit_ms"] < values["target_p95_ms"]:
        raise HandoffDeltaError(f"{label}.performance hard limit is below target")

    within_target = performance.get("within_target")
    within_hard_limit = performance.get("within_hard_limit")
    if not isinstance(within_target, bool) or not isinstance(within_hard_limit, bool):
        raise HandoffDeltaError(f"{label}.performance limit flags must be booleans")
    if within_target != (values["total_ms"] <= values["target_p95_ms"]):
        raise HandoffDeltaError(f"{label}.performance.within_target is inconsistent")
    if within_hard_limit != (values["total_ms"] <= values["hard_limit_ms"]):
        raise HandoffDeltaError(f"{label}.performance.within_hard_limit is inconsistent")
    if not within_hard_limit:
        raise HandoffDeltaError(f"{label} exceeded its recorded hard performance limit")
    return values["total_ms"]


def report_projection(
    report: dict[str, Any], *, compatibility_version: int | None = None
) -> dict[str, Any]:
    """Return deterministic semantic fields while excluding runtime-only provenance."""

    keys = (
        "policy_id",
        "update_id",
        "observation_kind",
        "mode",
        "status",
        "base_commit",
        "manual_handoff_authoritative",
        "candidate_replaced_authority",
        "candidate_sha256",
        "manual_sha256",
        "base_handoff_sha256",
        "base_registry_sha256",
        "current_registry_sha256",
        "exact_manual_candidate_match",
        "idempotence_passed",
        "history_preservation_passed",
        "operation_count",
        "affected_selectors",
        "network_used",
        "llm_blocking_gate_used",
    )
    projection = {key: report.get(key) for key in keys}
    report_version = (
        compatibility_version
        if compatibility_version is not None
        else report.get("report_schema_version", 1)
    )
    if report_version >= CURRENT_REPORT_SCHEMA_VERSION:
        projection["registry_change_assertions"] = registry_assertion_projection(
            report.get("registry_change_assertions", []),
            "report.registry_change_assertions",
        )
        projection["registry_change_coverage"] = report.get("registry_change_coverage")
        projection["registry_transition_assertions"] = registry_assertion_projection(
            report.get("registry_transition_assertions", []),
            "report.registry_transition_assertions",
        )
    else:
        projection.pop("observation_kind", None)
        projection["registry_transition_assertions"] = registry_assertion_projection(
            report.get("registry_transition_assertions", []),
            "report.registry_transition_assertions",
        )
    return projection


def stored_report_metadata(repo_root: Path, delta_path: Path) -> dict[str, Any]:
    report_path = delta_path.parent / REPORT_FILENAME
    if not report_path.is_file() or report_path.is_symlink():
        raise HandoffDeltaError(
            f"Delta {delta_path.parent.name} is missing required {REPORT_FILENAME}"
        )
    stored = load_json(report_path, "stored shadow report")
    report_version = stored.get("report_schema_version", 1)
    if report_version not in {1, CURRENT_REPORT_SCHEMA_VERSION}:
        raise HandoffDeltaError(f"Unsupported stored report schema version: {report_version}")
    provenance_head = stored.get("validation_worktree_head") or stored.get("head_commit")
    if not isinstance(provenance_head, str) or not GIT_SHA_RE.fullmatch(provenance_head):
        raise HandoffDeltaError(
            "Stored shadow report must record validation_worktree_head "
            "(or legacy head_commit) as a full Git SHA"
        )
    if stored.get("status") != "PASS" or stored.get("mode") != "shadow":
        raise HandoffDeltaError("Stored shadow report must be a successful shadow report")
    performance_total_ms = validate_report_runtime_fields(stored, "stored shadow report")
    return {
        "stored": stored,
        "path": report_path.relative_to(repo_root).as_posix(),
        "sha256": sha256_file(report_path),
        "report_schema_version": report_version,
        "validation_worktree_head": provenance_head,
        "legacy_head_commit_field": "head_commit" in stored,
        "performance_total_ms": performance_total_ms,
    }


def validate_stored_report(
    repo_root: Path,
    delta_path: Path,
    fresh_report: dict[str, Any],
) -> dict[str, Any]:
    metadata = stored_report_metadata(repo_root, delta_path)
    stored = metadata.pop("stored")
    if stored.get("update_id") != fresh_report.get("update_id"):
        raise HandoffDeltaError("Stored shadow report update_id does not match delta")
    report_version = metadata["report_schema_version"]
    validate_report_runtime_fields(fresh_report, "fresh replay report")
    if report_projection(
        stored, compatibility_version=report_version
    ) != report_projection(fresh_report, compatibility_version=report_version):
        raise HandoffDeltaError(
            f"Stored {REPORT_FILENAME} is stale or does not match deterministic replay"
        )
    return metadata

def repository_commit_for_added_path(repo_root: Path, path: Path) -> str | None:
    relative = path.resolve().relative_to(repo_root.resolve()).as_posix()
    result = git(
        repo_root,
        "log",
        "--diff-filter=A",
        "--format=%H",
        "--",
        relative,
        check=False,
    )
    commits = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return commits[-1] if commits else None


def latest_commit_for_path(repo_root: Path, path: Path) -> str | None:
    relative = path.resolve().relative_to(repo_root.resolve()).as_posix()
    result = git(repo_root, "log", "-1", "--format=%H", "--", relative, check=False)
    value = result.stdout.strip()
    return value if GIT_SHA_RE.fullmatch(value) else None


def commit_datetime(repo_root: Path, commit: str) -> datetime:
    value = git_text(repo_root, "show", "-s", "--format=%cI", commit)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise HandoffDeltaError(f"Invalid commit timestamp for {commit}: {value}") from exc
    return parsed.astimezone(timezone.utc)


def percentile(values: Sequence[float], percentile_value: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * percentile_value))))
    return ordered[index]


def observation_records(
    repo_root: Path,
    *,
    replay: bool = True,
) -> list[dict[str, Any]]:
    """Return immutable observation metadata.

    Full corpus and acceptance runs use ``replay=True`` to reconstruct every
    historical candidate. The ordinary Fast Gate uses ``replay=False`` and only
    reads already-validated report metadata; any report touched by the current
    update is separately replayed by ``auto_check``.
    """

    repo_root = repo_root.resolve()
    policy = load_policy(repo_root)
    root = repo_root / "docs" / "handoff_deltas"
    records: list[dict[str, Any]] = []
    for delta_path in sorted(root.glob(f"*/{DELTA_FILENAME}")):
        delta = load_yaml(delta_path, "handoff delta")
        validate_delta_shape(delta, delta_path, policy)
        repository_commit = repository_commit_for_added_path(repo_root, delta_path)
        if repository_commit is None:
            continue
        if replay:
            result = check_delta(
                repo_root,
                delta_path,
                enforce_performance=False,
                target_commit=repository_commit,
            )
            report_meta = validate_stored_report(repo_root, delta_path, result.report)
        else:
            report_meta = stored_report_metadata(repo_root, delta_path)
            stored = report_meta.pop("stored")
            if stored.get("update_id") != delta["update_id"]:
                raise HandoffDeltaError("Stored shadow report update_id does not match delta")
            if stored.get("base_commit") != delta["base"]["commit"]:
                raise HandoffDeltaError("Stored shadow report base_commit does not match delta")
        records.append(
            {
                "update_id": delta["update_id"],
                "kind": classify_observation(delta["update_id"]),
                "delta_schema_version": delta["schema_version"],
                "delta_path": delta_path.relative_to(repo_root).as_posix(),
                "report_path": report_meta["path"],
                "report_sha256": report_meta["sha256"],
                "repository_commit": repository_commit,
                "repository_commit_time": commit_datetime(
                    repo_root, repository_commit
                ).isoformat(),
                "validation_worktree_head": report_meta["validation_worktree_head"],
                "legacy_head_commit_field": report_meta["legacy_head_commit_field"],
                "performance_total_ms": report_meta["performance_total_ms"],
            }
        )
    records.sort(key=lambda item: (item["repository_commit_time"], item["update_id"]))
    return records


def observation_fingerprint(update_ids: Sequence[str]) -> str:
    return sha256_text("\n".join(sorted(update_ids)) + "\n")


def validate_full_acceptance_report(
    repo_root: Path,
    path: Path,
    observations: Sequence[dict[str, Any]],
) -> dict[str, Any] | None:
    payload = load_json(path, "full acceptance report")
    if payload.get("tier") != "full" or payload.get("status") != "PASS":
        return None
    repository_commit = latest_commit_for_path(repo_root, path)
    if repository_commit is None:
        return None
    report_version = payload.get("report_schema_version", 1)
    coverage = payload.get("coverage")
    covered_ids: list[str] = []
    if report_version >= CURRENT_REPORT_SCHEMA_VERSION:
        if not isinstance(coverage, dict):
            raise HandoffDeltaError(f"Full report {path} is missing coverage")
        raw_ids = coverage.get("covered_update_ids")
        if not isinstance(raw_ids, list) or not all(isinstance(value, str) for value in raw_ids):
            raise HandoffDeltaError(f"Full report {path} has invalid covered_update_ids")
        covered_ids = list(raw_ids)
        if covered_ids != sorted(set(covered_ids)):
            raise HandoffDeltaError(
                f"Full report {path} covered_update_ids must be unique and sorted"
            )
        real_ids = {
            row["update_id"]
            for row in observations
            if row["kind"] == "real"
            and git(
                repo_root,
                "merge-base",
                "--is-ancestor",
                row["repository_commit"],
                repository_commit,
                check=False,
            ).returncode
            == 0
        }
        unknown = sorted(set(covered_ids) - real_ids)
        if unknown:
            raise HandoffDeltaError(
                f"Full report {path} covers unknown or future observations: {unknown}"
            )
        if coverage.get("successful_real_observation_count") != len(covered_ids):
            raise HandoffDeltaError(f"Full report {path} coverage count is inconsistent")
        if coverage.get("observation_fingerprint") != observation_fingerprint(covered_ids):
            raise HandoffDeltaError(f"Full report {path} observation fingerprint is invalid")
        outcomes = payload.get("outcomes")
        if not isinstance(outcomes, list) or not outcomes:
            raise HandoffDeltaError(f"Full report {path} must retain command outcomes")
        if any(
            not isinstance(item, dict)
            or item.get("returncode") != 0
            or item.get("timed_out") is not False
            for item in outcomes
        ):
            raise HandoffDeltaError(f"Full report {path} contains a failed outcome")
        corpus = payload.get("corpus_audit")
        if not isinstance(corpus, dict) or corpus.get("all_stored_reports_revalidated") is not True:
            raise HandoffDeltaError(f"Full report {path} lacks a successful corpus audit")
    elif isinstance(coverage, dict) and isinstance(coverage.get("covered_update_ids"), list):
        covered_ids = [str(value) for value in coverage["covered_update_ids"]]
    return {
        "path": path.relative_to(repo_root).as_posix(),
        "sha256": sha256_file(path),
        "repository_commit": repository_commit,
        "repository_commit_time": commit_datetime(repo_root, repository_commit).isoformat(),
        "covered_update_ids": covered_ids,
        "reasons": payload.get("reasons", []),
        "report_schema_version": report_version,
    }


def full_acceptance_records(
    repo_root: Path,
    observations: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    repo_root = repo_root.resolve()
    root = repo_root / "docs" / "handoff_deltas"
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob(f"*/{FULL_REPORT_FILENAME}")):
        row = validate_full_acceptance_report(repo_root, path, observations)
        if row is not None:
            rows.append(row)
    rows.sort(key=lambda item: (item["repository_commit_time"], item["path"]))
    return rows

def parse_as_of(value: str | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise HandoffDeltaError(f"Invalid --as-of timestamp: {value}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def acceptance_status(repo_root: Path, *, as_of: str | None = None) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    policy = load_policy(repo_root)
    observations = observation_records(repo_root, replay=False)
    real = [row for row in observations if row["kind"] == "real"]
    bootstrap = [row for row in observations if row["kind"] == "bootstrap"]
    full_reports = full_acceptance_records(repo_root, observations)
    latest_full = full_reports[-1] if full_reports else None
    covered = set(latest_full["covered_update_ids"] if latest_full else [])
    uncovered = [row for row in real if row["update_id"] not in covered]

    config = require_mapping(policy.get("full_acceptance"), "policy.full_acceptance")
    count_interval = int(config.get("successful_update_interval", 20))
    day_interval = int(config.get("max_days_with_relevant_update", 7))
    now = parse_as_of(as_of)
    elapsed_days: float | None = None
    if latest_full is not None:
        full_time = datetime.fromisoformat(latest_full["repository_commit_time"])
        elapsed_days = max(0.0, (now - full_time).total_seconds() / 86400.0)
    reasons: list[str] = []
    if latest_full is None:
        reasons.append("no_successful_full_acceptance_report")
    if len(uncovered) >= count_interval:
        reasons.append("successful_relevant_update_interval_reached")
    if uncovered and elapsed_days is not None and elapsed_days >= day_interval:
        reasons.append("calendar_interval_with_relevant_update_reached")

    performance_values = [
        float(row["performance_total_ms"])
        for row in real
        if isinstance(row.get("performance_total_ms"), (int, float))
    ]
    return {
        "schema_version": 1,
        "policy_id": policy["policy_id"],
        "status": "FULL_DUE" if reasons else "CURRENT",
        "as_of": now.isoformat(),
        "bootstrap_observation_count": len(bootstrap),
        "successful_real_observation_count": len(real),
        "successful_real_observations": real,
        "latest_real_observation": real[-1] if real else None,
        "full_acceptance_reports": full_reports,
        "latest_full_acceptance": latest_full,
        "uncovered_real_observation_count": len(uncovered),
        "uncovered_update_ids": [row["update_id"] for row in uncovered],
        "full_acceptance_due": bool(reasons),
        "full_acceptance_due_reasons": reasons,
        "remaining_until_count_trigger": max(0, count_interval - len(uncovered)),
        "days_since_latest_full_acceptance": (
            round(elapsed_days, 3) if elapsed_days is not None else None
        ),
        "performance_ms": {
            "sample_count": len(performance_values),
            "mean": round(statistics.fmean(performance_values), 3)
            if performance_values
            else None,
            "p95": round(percentile(performance_values, 0.95), 3)
            if performance_values
            else None,
            "max": round(max(performance_values), 3) if performance_values else None,
        },
    }


def corpus_check(repo_root: Path) -> dict[str, Any]:
    observations = observation_records(repo_root)
    return {
        "status": "PASS",
        "mode": "shadow",
        "observation_count": len(observations),
        "bootstrap_count": sum(item["kind"] == "bootstrap" for item in observations),
        "real_count": sum(item["kind"] == "real" for item in observations),
        "observations": observations,
    }

def changed_paths(repo_root: Path) -> set[str]:
    working_paths: set[str] = set()
    for args in (("diff", "--name-only"), ("diff", "--cached", "--name-only")):
        result = git(repo_root, *args)
        working_paths.update(line for line in result.stdout.splitlines() if line)
    result = git(repo_root, "ls-files", "--others", "--exclude-standard")
    working_paths.update(line for line in result.stdout.splitlines() if line)
    if working_paths:
        # During authoring or isolated integration, the current update is the
        # worktree/index delta. Including HEAD's already-committed parent diff
        # would double-count the previous observation.
        return working_paths

    head_parent = git(repo_root, "rev-parse", "HEAD^", check=False)
    if head_parent.returncode != 0:
        return set()
    result = git(repo_root, "diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD")
    return {line for line in result.stdout.splitlines() if line}


def discover_changed_deltas(repo_root: Path, paths: Iterable[str]) -> list[Path]:
    """Map any touched observation artifact back to its sibling delta."""

    found: set[Path] = set()
    prefix = Path("docs/handoff_deltas")
    for value in sorted(set(paths)):
        relative = Path(value)
        if len(relative.parts) < 4 or Path(*relative.parts[:2]) != prefix:
            continue
        delta = (repo_root / Path(*relative.parts[:3]) / DELTA_FILENAME).resolve()
        if delta.is_file():
            found.add(delta)
    return sorted(found)


def discover_changed_full_reports(repo_root: Path, paths: Iterable[str]) -> list[Path]:
    return sorted(
        (repo_root / value).resolve()
        for value in set(paths)
        if value.startswith("docs/handoff_deltas/")
        and value.endswith(f"/{FULL_REPORT_FILENAME}")
    )


def auto_check(repo_root: Path, *, allow_full_due: bool = False) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    policy = load_policy(repo_root)
    paths = changed_paths(repo_root)
    relevant_authority_change = bool({"docs/handoff.md", "experiments/registry.yaml"} & paths)
    deltas = discover_changed_deltas(repo_root, paths)
    changed_delta_files = {
        (repo_root / value).resolve()
        for value in paths
        if value.startswith("docs/handoff_deltas/")
        and value.endswith(f"/{DELTA_FILENAME}")
    }
    maximum = policy["safety"]["maximum_deltas_per_update"]
    if relevant_authority_change and not changed_delta_files:
        raise HandoffDeltaError("handoff/registry changed without a HANDOFF_DELTA.yaml")
    if len(changed_delta_files) > maximum:
        raise HandoffDeltaError(
            f"Version 1 allows at most {maximum} new or modified delta per update; "
            f"found {len(changed_delta_files)}"
        )
    reports = []
    stored_reports = []
    for path in deltas:
        target_commit = None
        if path.resolve() not in changed_delta_files:
            target_commit = repository_commit_for_added_path(repo_root, path)
            if target_commit is None:
                raise HandoffDeltaError(
                    f"Historical observation artifact changed before its delta entered Git history: {path}"
                )
        result = check_delta(repo_root, path, target_commit=target_commit)
        reports.append(result.report)
        stored_reports.append(validate_stored_report(repo_root, path, result.report))
    lightweight_observations = observation_records(repo_root, replay=False)
    changed_full_reports = discover_changed_full_reports(repo_root, paths)
    validated_full_reports = [
        validate_full_acceptance_report(repo_root, path, lightweight_observations)
        for path in changed_full_reports
    ]
    acceptance = acceptance_status(repo_root)
    if acceptance["full_acceptance_due"] and not allow_full_due:
        raise HandoffDeltaError(
            "Full Acceptance is due before this update can pass: "
            + ", ".join(acceptance["full_acceptance_due_reasons"])
        )
    return {
        "status": "PASS",
        "mode": "shadow",
        "changed_paths": sorted(paths),
        "relevant_authority_change": relevant_authority_change,
        "delta_count": len(deltas),
        "changed_delta_file_count": len(changed_delta_files),
        "reports": reports,
        "stored_reports": stored_reports,
        "validated_full_reports": [row for row in validated_full_reports if row is not None],
        "acceptance_status": acceptance,
    }


def operation_footprint(op: dict[str, Any]) -> tuple[str, tuple[str, ...], str | None]:
    return op["op"], tuple(op["heading_path"]), op.get("block_id")


def registry_change_footprints(delta: dict[str, Any]) -> list[tuple[tuple[str, tuple[str, ...]], dict[str, Any]]]:
    registry = delta["registry"]
    if registry["mode"] == "unchanged":
        return []
    rows = registry.get("changes")
    if rows is None:
        rows = [
            {
                "change_id": item["assertion_id"],
                "kind": "transition",
                "entity_id": item["entity_id"],
                "field_path": item["field_path"],
                "machine": item["machine"],
                "from": item["from"],
                "to": item["to"],
            }
            for item in registry.get("transitions", [])
        ]
    result = []
    for item in rows:
        path = tuple(item.get("field_path", ("<entity>",)))
        result.append(((item["entity_id"], path), item))
    return result


def pair_check(repo_root: Path, delta_a_path: Path, delta_b_path: Path) -> dict[str, Any]:
    policy = load_policy(repo_root)
    delta_a = load_yaml(delta_a_path, "delta A")
    delta_b = load_yaml(delta_b_path, "delta B")
    validate_delta_shape(delta_a, delta_a_path, policy)
    validate_delta_shape(delta_b, delta_b_path, policy)
    if delta_a["base"] != delta_b["base"]:
        raise HandoffDeltaError("Pair check requires identical base metadata")

    footprints: dict[tuple[str, tuple[str, ...], str | None], dict[str, Any]] = {}
    for label, delta in (("A", delta_a), ("B", delta_b)):
        for op in delta["operations"]:
            footprint = operation_footprint(op)
            previous = footprints.get(footprint)
            if previous is not None and previous != op:
                raise HandoffDeltaError(f"Semantic operation conflict at {footprint} between deltas")
            footprints[footprint] = op

    registry_footprints: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
    for delta in (delta_a, delta_b):
        for footprint, change in registry_change_footprints(delta):
            previous = registry_footprints.get(footprint)
            comparable = {
                key: value
                for key, value in change.items()
                if key not in {"change_id", "assertion_id", "evidence", "reason"}
            }
            if previous is not None and previous != comparable:
                raise HandoffDeltaError(
                    f"Semantic registry conflict at {footprint} between deltas"
                )
            registry_footprints[footprint] = comparable

    base_text = git_show_text(repo_root, delta_a["base"]["commit"], "docs/handoff.md")
    ab = render(render(base_text, delta_a["operations"]).text, delta_b["operations"]).text
    ba = render(render(base_text, delta_b["operations"]).text, delta_a["operations"]).text
    if ab != ba:
        raise HandoffDeltaError("Independent delta application is not order-independent")
    return {
        "status": "PASS",
        "delta_a": delta_a["update_id"],
        "delta_b": delta_b["update_id"],
        "commutative": True,
        "registry_conflicts_checked": True,
        "combined_sha256": sha256_text(ab),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser("check")
    check_parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    check_parser.add_argument("--delta", type=Path, required=True)
    check_parser.add_argument("--report", type=Path)
    check_parser.add_argument("--candidate-on-failure", type=Path)
    check_parser.add_argument("--target-commit")
    check_parser.add_argument("--json", action="store_true")
    check_parser.add_argument("--no-performance-enforcement", action="store_true")

    record_parser = subparsers.add_parser("record-report")
    record_parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    record_parser.add_argument("--delta", type=Path, required=True)
    record_parser.add_argument("--force", action="store_true")
    record_parser.add_argument("--json", action="store_true")

    auto_parser = subparsers.add_parser("auto-check")
    auto_parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    auto_parser.add_argument("--allow-full-due", action="store_true")
    auto_parser.add_argument("--json", action="store_true")

    corpus_parser = subparsers.add_parser("corpus-check")
    corpus_parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    corpus_parser.add_argument("--json", action="store_true")

    status_parser = subparsers.add_parser("acceptance-status")
    status_parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    status_parser.add_argument("--as-of")
    status_parser.add_argument("--require-current", action="store_true")
    status_parser.add_argument("--json", action="store_true")

    render_parser = subparsers.add_parser("render")
    render_parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    render_parser.add_argument("--delta", type=Path, required=True)
    render_parser.add_argument("--output", type=Path, required=True)

    pair_parser = subparsers.add_parser("pair-check")
    pair_parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    pair_parser.add_argument("--delta-a", type=Path, required=True)
    pair_parser.add_argument("--delta-b", type=Path, required=True)
    pair_parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "check":
            result = check_delta(
                args.repo_root,
                args.delta,
                enforce_performance=not args.no_performance_enforcement,
                target_commit=args.target_commit,
            )
            if args.report:
                write_json(args.report, result.report)
            payload = result.report
        elif args.command == "record-report":
            result = check_delta(args.repo_root, args.delta)
            report_path = args.delta.resolve().parent / REPORT_FILENAME
            if report_path.exists() and not args.force:
                raise HandoffDeltaError(
                    f"{report_path} already exists; use --force only while authoring this update"
                )
            write_json(report_path, result.report)
            payload = {
                "status": "PASS",
                "mode": "shadow",
                "report": report_path.relative_to(args.repo_root.resolve()).as_posix(),
                **result.report,
            }
        elif args.command == "auto-check":
            payload = auto_check(args.repo_root, allow_full_due=args.allow_full_due)
        elif args.command == "corpus-check":
            payload = corpus_check(args.repo_root)
        elif args.command == "acceptance-status":
            payload = acceptance_status(args.repo_root, as_of=args.as_of)
            if args.require_current and payload["full_acceptance_due"]:
                raise HandoffDeltaError(
                    "Full Acceptance is due: "
                    + ", ".join(payload["full_acceptance_due_reasons"])
                )
        elif args.command == "render":
            result = check_delta(args.repo_root, args.delta, enforce_performance=False)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(result.candidate, encoding="utf-8")
            payload = {"status": "PASS", "output": str(args.output), **result.report}
        elif args.command == "pair-check":
            payload = pair_check(args.repo_root.resolve(), args.delta_a.resolve(), args.delta_b.resolve())
        else:
            raise AssertionError(args.command)
    except Exception as exc:
        if args.command == "check" and getattr(args, "candidate_on_failure", None):
            try:
                delta = load_yaml(args.delta.resolve(), "handoff delta")
                base_text = git_show_text(
                    args.repo_root.resolve(), delta["base"]["commit"], "docs/handoff.md"
                )
                rendered = render(base_text, delta["operations"]).text
                args.candidate_on_failure.parent.mkdir(parents=True, exist_ok=True)
                args.candidate_on_failure.write_text(rendered, encoding="utf-8")
            except Exception:
                pass
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            f"HANDOFF_DELTA shadow gate: PASS "
            f"(command={args.command}, mode={payload.get('mode', 'shadow')})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
