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
import re
import subprocess
import sys
import time
from dataclasses import dataclass
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


def validate_delta_shape(delta: dict[str, Any], delta_path: Path) -> None:
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
    if delta.get("schema_version") != 1:
        raise HandoffDeltaError("delta.schema_version must be 1")
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
    reject_unknown_keys(registry, {"mode", "expected_after_sha256", "transitions"}, "delta.registry")
    mode = registry.get("mode")
    if mode not in {"unchanged", "expected_after"}:
        raise HandoffDeltaError("delta.registry.mode must be unchanged or expected_after")
    transitions = require_list(registry.get("transitions", []), "delta.registry.transitions")
    expected_after = registry.get("expected_after_sha256")
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
            field_path = require_list(item.get("field_path"), f"registry.transitions[{index}].field_path")
            if not field_path or not all(isinstance(part, str) and part for part in field_path):
                raise HandoffDeltaError("registry transition field_path must be non-empty strings")
            evidence = require_list(item.get("evidence"), f"registry.transitions[{index}].evidence")
            if not evidence or not all(isinstance(path, str) and path for path in evidence):
                raise HandoffDeltaError("registry transition evidence must be non-empty paths")

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


def validate_registry(
    repo_root: Path,
    delta: dict[str, Any],
    base_registry_text: str,
    current_registry_text: str,
) -> list[dict[str, Any]]:
    spec = delta["registry"]
    base_hash = sha256_text(base_registry_text)
    current_hash = sha256_text(current_registry_text)
    if base_hash != delta["base"]["registry_sha256"]:
        raise HandoffDeltaError("Base registry SHA-256 does not match delta")
    if spec["mode"] == "unchanged":
        if current_hash != base_hash:
            raise HandoffDeltaError("Registry changed but delta.registry.mode is unchanged")
        return []
    if current_hash != spec["expected_after_sha256"]:
        raise HandoffDeltaError("Current registry SHA-256 does not match expected after-image")

    state_payload = load_yaml(
        repo_root / "docs" / "handoff_delta_state_machines.yaml", "state machines"
    )
    machines = require_mapping(state_payload.get("machines"), "state_machines.machines")
    base_registry = require_mapping(yaml.safe_load(base_registry_text), "base registry")
    current_registry = require_mapping(yaml.safe_load(current_registry_text), "current registry")
    reports: list[dict[str, Any]] = []
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
        machine_name = transition["machine"]
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
        evidence_reports = []
        for relative in transition["evidence"]:
            path = safe_repo_file(repo_root, relative, f"transition {transition['assertion_id']} evidence")
            evidence_reports.append({"path": relative, "sha256": sha256_file(path)})
        reports.append(
            {
                "assertion_id": transition["assertion_id"],
                "entity_id": entity_id,
                "field_path": field_path,
                "machine": machine_name,
                "from": before,
                "to": after,
                "evidence": evidence_reports,
            }
        )
    return reports


def check_delta(repo_root: Path, delta_path: Path, *, enforce_performance: bool = True) -> CheckResult:
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
    validate_delta_shape(delta, delta_path)
    policy_ms = (time.perf_counter() - policy_start) * 1000

    base_start = time.perf_counter()
    base_commit = delta["base"]["commit"]
    git(repo_root, "cat-file", "-e", f"{base_commit}^{{commit}}")
    git(repo_root, "merge-base", "--is-ancestor", base_commit, "HEAD")
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
    manual_path = safe_repo_file(repo_root, handoff_path, "manual handoff")
    registry_file = safe_repo_file(repo_root, registry_path, "current registry")
    manual = manual_path.read_text(encoding="utf-8")
    current_registry = registry_file.read_text(encoding="utf-8")
    candidate_hash = sha256_text(rendered.text)
    manual_hash = sha256_text(manual)
    if candidate_hash != delta["expected"]["candidate_sha256"]:
        raise HandoffDeltaError("Rendered candidate SHA-256 differs from delta.expected")
    if manual_hash != delta["expected"]["manual_sha256"]:
        raise HandoffDeltaError("Manual handoff SHA-256 differs from delta.expected")
    if rendered.text != manual:
        raise HandoffDeltaError("Shadow candidate does not exactly match the authoritative manual handoff")
    registry_transitions = validate_registry(repo_root, delta, base_registry, current_registry)
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
        "policy_id": policy["policy_id"],
        "update_id": delta["update_id"],
        "mode": "shadow",
        "status": "PASS",
        "base_commit": base_commit,
        "head_commit": git_text(repo_root, "rev-parse", "HEAD"),
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
        "registry_transition_assertions": registry_transitions,
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


def changed_paths(repo_root: Path) -> set[str]:
    paths: set[str] = set()
    head_parent = git(repo_root, "rev-parse", "HEAD^", check=False)
    if head_parent.returncode == 0:
        result = git(repo_root, "diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD")
        paths.update(line for line in result.stdout.splitlines() if line)
    for args in (("diff", "--name-only"), ("diff", "--cached", "--name-only")):
        result = git(repo_root, *args)
        paths.update(line for line in result.stdout.splitlines() if line)
    result = git(repo_root, "ls-files", "--others", "--exclude-standard")
    paths.update(line for line in result.stdout.splitlines() if line)
    return paths


def discover_changed_deltas(repo_root: Path, paths: Iterable[str]) -> list[Path]:
    found = []
    for value in sorted(set(paths)):
        if value.startswith("docs/handoff_deltas/") and value.endswith(f"/{DELTA_FILENAME}"):
            found.append((repo_root / value).resolve())
    return found


def auto_check(repo_root: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    policy = load_policy(repo_root)
    paths = changed_paths(repo_root)
    relevant_authority_change = bool({"docs/handoff.md", "experiments/registry.yaml"} & paths)
    deltas = discover_changed_deltas(repo_root, paths)
    maximum = policy["safety"]["maximum_deltas_per_update"]
    if relevant_authority_change and not deltas:
        raise HandoffDeltaError("handoff/registry changed without a HANDOFF_DELTA.yaml")
    if len(deltas) > maximum:
        raise HandoffDeltaError(
            f"Version 1 allows at most {maximum} delta per update; found {len(deltas)}"
        )
    reports = [check_delta(repo_root, path).report for path in deltas]
    return {
        "status": "PASS",
        "mode": "shadow",
        "changed_paths": sorted(paths),
        "relevant_authority_change": relevant_authority_change,
        "delta_count": len(deltas),
        "reports": reports,
    }


def operation_footprint(op: dict[str, Any]) -> tuple[str, tuple[str, ...], str | None]:
    return op["op"], tuple(op["heading_path"]), op.get("block_id")


def pair_check(repo_root: Path, delta_a_path: Path, delta_b_path: Path) -> dict[str, Any]:
    delta_a = load_yaml(delta_a_path, "delta A")
    delta_b = load_yaml(delta_b_path, "delta B")
    validate_delta_shape(delta_a, delta_a_path)
    validate_delta_shape(delta_b, delta_b_path)
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
    check_parser.add_argument("--json", action="store_true")
    check_parser.add_argument("--no-performance-enforcement", action="store_true")

    auto_parser = subparsers.add_parser("auto-check")
    auto_parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    auto_parser.add_argument("--json", action="store_true")

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
            )
            if args.report:
                write_json(args.report, result.report)
            payload = result.report
        elif args.command == "auto-check":
            payload = auto_check(args.repo_root)
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
