#!/usr/bin/env python3
"""Build deterministic Stage 4 minimal modules, dependency views, and context packs.

This tool intentionally does not infer or mutate module structure.  The module
configuration and the single ``depends_on`` graph are human-reviewed inputs.
``docs/handoff.md`` and ``experiments/registry.yaml`` remain authoritative;
everything under the minimal ``generated`` directory is disposable shadow output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml


DEFAULT_MODULES = Path("docs/handoff_shadow/stage4/minimal/MODULES.yaml")
DEFAULT_DEPENDENCIES = Path("docs/handoff_shadow/stage4/minimal/DEPENDENCIES.yaml")
DEFAULT_OUTPUT = Path("docs/handoff_shadow/stage4/minimal/generated")
CORE_GENERATED_FILES = (
    "MODULE_INDEX.json",
    "DEPENDENCY_GRAPH.dot",
    "DEPENDENCY_GRAPH.md",
    "STRUCTURE_SUGGESTIONS.md",
)


class ContextBuildError(ValueError):
    """Raised when minimal-context inputs or generated outputs are invalid."""


@dataclass(frozen=True)
class SourceChunk:
    label: str
    descriptor: dict[str, Any]
    text: str


@dataclass(frozen=True)
class ModuleSnapshot:
    module_id: str
    title: str
    responsibility: str
    source_hash: str
    snapshot_hash: str
    source_chars: int
    bytes_payload: bytes
    source_labels: tuple[str, ...]


@dataclass(frozen=True)
class BuildPlan:
    outputs: dict[Path, bytes]
    snapshots: dict[str, ModuleSnapshot]
    dependencies: dict[str, tuple[str, ...]]
    module_order: tuple[str, ...]
    graph_hash: str
    suggestions: tuple[dict[str, Any], ...]
    acceptance_results: tuple[dict[str, Any], ...]


def load_yaml(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ContextBuildError(f"missing {label}: {path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ContextBuildError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ContextBuildError(f"{label} must be a YAML mapping: {path}")
    return payload


def safe_repo_path(repo_root: Path, value: str, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ContextBuildError(f"{label} must be a non-empty repository-relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ContextBuildError(f"unsafe {label}: {value!r}")
    resolved = (repo_root / relative).resolve()
    root = repo_root.resolve()
    if resolved != root and root not in resolved.parents:
        raise ContextBuildError(f"{label} escapes repository root: {value!r}")
    return resolved


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_hash(payload: Any) -> str:
    data = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256_bytes(data)


def json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def yaml_text(payload: Any) -> str:
    return yaml.safe_dump(
        payload,
        sort_keys=False,
        allow_unicode=True,
        width=120,
    )


def require_exact_line(text: str, literal: str, label: str) -> int:
    if not isinstance(literal, str) or not literal:
        raise ContextBuildError(f"{label} must be a non-empty exact line")
    lines = text.splitlines(keepends=True)
    matches = [index for index, line in enumerate(lines) if line.rstrip("\r\n") == literal]
    if len(matches) != 1:
        raise ContextBuildError(
            f"{label} must match exactly one line, found {len(matches)}: {literal!r}"
        )
    return matches[0]


def extract_markdown_range(repo_root: Path, source: dict[str, Any]) -> SourceChunk:
    path_value = source.get("path")
    path = safe_repo_path(repo_root, path_value, "markdown source path")
    if not path.is_file():
        raise ContextBuildError(f"missing markdown source: {path}")
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    start_literal = source.get("start")
    end_literal = source.get("end")
    start_index = require_exact_line(text, start_literal, f"range start in {path_value}")
    if end_literal is None:
        end_index = len(lines)
    else:
        end_index = require_exact_line(text, end_literal, f"range end in {path_value}")
        if end_index <= start_index:
            raise ContextBuildError(
                f"range end must follow start in {path_value}: {start_literal!r} -> "
                f"{end_literal!r}"
            )
    extracted = "".join(lines[start_index:end_index]).rstrip() + "\n"
    descriptor = {
        "kind": "markdown_range",
        "path": path_value,
        "start": start_literal,
        "end": end_literal,
    }
    end_label = end_literal if end_literal is not None else "<EOF>"
    label = f"{path_value}: {start_literal} -> {end_label}"
    return SourceChunk(label=label, descriptor=descriptor, text=extracted)


def extract_marker_block(repo_root: Path, source: dict[str, Any]) -> SourceChunk:
    path_value = source.get("path")
    block_id = source.get("block_id")
    if not isinstance(block_id, str) or not block_id:
        raise ContextBuildError("marker_block.block_id must be a non-empty string")
    path = safe_repo_path(repo_root, path_value, "marker source path")
    if not path.is_file():
        raise ContextBuildError(f"missing marker source: {path}")
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    start_literal = f"<!-- HANDOFF-DELTA-BLOCK:{block_id}:START -->"
    end_literal = f"<!-- HANDOFF-DELTA-BLOCK:{block_id}:END -->"
    start_index = require_exact_line(text, start_literal, f"marker start in {path_value}")
    end_index = require_exact_line(text, end_literal, f"marker end in {path_value}")
    if end_index <= start_index:
        raise ContextBuildError(f"marker block is reversed in {path_value}: {block_id}")
    extracted = "".join(lines[start_index + 1 : end_index]).strip() + "\n"
    descriptor = {
        "kind": "marker_block",
        "path": path_value,
        "block_id": block_id,
    }
    return SourceChunk(
        label=f"{path_value}: HANDOFF-DELTA-BLOCK {block_id}",
        descriptor=descriptor,
        text=extracted,
    )



def extract_matching_marker_blocks(repo_root: Path, source: dict[str, Any]) -> SourceChunk:
    path_value = source.get("path")
    tokens = source.get("match_any")
    if not isinstance(tokens, list) or not tokens or not all(
        isinstance(token, str) and token for token in tokens
    ):
        raise ContextBuildError(
            "marker_blocks_matching.match_any must be a non-empty string list"
        )
    if len(tokens) != len(set(tokens)):
        raise ContextBuildError("marker_blocks_matching.match_any contains duplicates")
    path = safe_repo_path(repo_root, path_value, "marker source path")
    if not path.is_file():
        raise ContextBuildError(f"missing marker source: {path}")
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    start_prefix = "<!-- HANDOFF-DELTA-BLOCK:"
    start_suffix = ":START -->"
    seen_ids: set[str] = set()
    matched: list[tuple[str, str]] = []
    index = 0
    while index < len(lines):
        stripped = lines[index].rstrip("\r\n")
        if not (stripped.startswith(start_prefix) and stripped.endswith(start_suffix)):
            index += 1
            continue
        block_id = stripped[len(start_prefix) : -len(start_suffix)]
        if not block_id or block_id in seen_ids:
            raise ContextBuildError(
                f"duplicate or empty HANDOFF-DELTA-BLOCK ID in {path_value}: {block_id!r}"
            )
        seen_ids.add(block_id)
        expected_end = f"<!-- HANDOFF-DELTA-BLOCK:{block_id}:END -->"
        end_index = index + 1
        while end_index < len(lines):
            candidate = lines[end_index].rstrip("\r\n")
            if candidate == expected_end:
                break
            if candidate.startswith(start_prefix) and candidate.endswith(start_suffix):
                raise ContextBuildError(
                    f"nested or unterminated HANDOFF-DELTA-BLOCK in {path_value}: "
                    f"{block_id}"
                )
            end_index += 1
        if end_index >= len(lines):
            raise ContextBuildError(
                f"unterminated HANDOFF-DELTA-BLOCK in {path_value}: {block_id}"
            )
        body = "".join(lines[index + 1 : end_index]).strip()
        searchable = block_id + "\n" + body
        if any(token in searchable for token in tokens):
            matched.append((block_id, body))
        index = end_index + 1
    if not matched:
        raise ContextBuildError(
            f"marker_blocks_matching found no blocks in {path_value} for tokens {tokens}"
        )
    rendered_parts: list[str] = []
    for block_id, body in matched:
        rendered_parts.extend([f"### Delta block `{block_id}`", "", body, ""])
    descriptor = {
        "kind": "marker_blocks_matching",
        "path": path_value,
        "match_any": tokens,
        "matched_block_ids": [block_id for block_id, _ in matched],
    }
    return SourceChunk(
        label=(
            f"{path_value}: HANDOFF-DELTA-BLOCKs matching "
            + ", ".join(repr(token) for token in tokens)
        ),
        descriptor=descriptor,
        text="\n".join(rendered_parts).rstrip() + "\n",
    )

def registry_entry_id(entry: dict[str, Any]) -> str | None:
    for key in ("experiment_id", "id", "claim_id"):
        value = entry.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def load_registry_collection(
    repo_root: Path,
    path_value: str,
    collection: str,
) -> list[dict[str, Any]]:
    path = safe_repo_path(repo_root, path_value, "registry source path")
    payload = load_yaml(path, "experiment registry")
    entries = payload.get(collection)
    if not isinstance(entries, list):
        raise ContextBuildError(
            f"registry collection {collection!r} must be a list in {path_value}"
        )
    result: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ContextBuildError(
                f"registry {collection}[{index}] must be a mapping in {path_value}"
            )
        result.append(entry)
    return result


def extract_registry_entries(repo_root: Path, source: dict[str, Any]) -> SourceChunk:
    path_value = source.get("path")
    collection = source.get("collection", "experiments")
    ids = source.get("experiment_ids")
    if not isinstance(collection, str) or not collection:
        raise ContextBuildError("registry_entries.collection must be a non-empty string")
    if not isinstance(ids, list) or not ids or not all(isinstance(x, str) and x for x in ids):
        raise ContextBuildError("registry_entries.experiment_ids must be a non-empty string list")
    if len(ids) != len(set(ids)):
        raise ContextBuildError(
            f"registry_entries contains duplicate IDs in {collection}: {ids}"
        )
    entries = load_registry_collection(repo_root, path_value, collection)
    index: dict[str, dict[str, Any]] = {}
    for entry in entries:
        item_id = registry_entry_id(entry)
        if item_id is None:
            raise ContextBuildError(
                f"registry entry without experiment_id/id/claim_id in {path_value}:{collection}"
            )
        if item_id in index:
            raise ContextBuildError(
                f"duplicate registry ID {item_id!r} in {path_value}:{collection}"
            )
        index[item_id] = entry
    missing = [item_id for item_id in ids if item_id not in index]
    if missing:
        raise ContextBuildError(
            f"missing registry IDs in {path_value}:{collection}: {', '.join(missing)}"
        )
    selected = [index[item_id] for item_id in ids]
    rendered = yaml_text({"collection": collection, "entries": selected})
    descriptor = {
        "kind": "registry_entries",
        "path": path_value,
        "collection": collection,
        "experiment_ids": ids,
    }
    return SourceChunk(
        label=f"{path_value}: {collection}[{', '.join(ids)}]",
        descriptor=descriptor,
        text=rendered,
    )


def extract_source(repo_root: Path, source: dict[str, Any]) -> SourceChunk:
    kind = source.get("kind")
    if kind == "markdown_range":
        return extract_markdown_range(repo_root, source)
    if kind == "marker_block":
        return extract_marker_block(repo_root, source)
    if kind == "marker_blocks_matching":
        return extract_matching_marker_blocks(repo_root, source)
    if kind == "registry_entries":
        return extract_registry_entries(repo_root, source)
    raise ContextBuildError(f"unknown module source kind: {kind!r}")


def normalize_modules(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    if payload.get("schema_version") != 1:
        raise ContextBuildError("MODULES.yaml schema_version must be 1")
    if payload.get("research_master") != "docs/handoff.md":
        raise ContextBuildError("MODULES.yaml must keep docs/handoff.md as research_master")
    if payload.get("authority") != "non_authoritative_stage4_minimal_context_shadow":
        raise ContextBuildError("MODULES.yaml authority must remain the non-authoritative shadow")
    if payload.get("structure_change_policy") != "suggestion_only_human_approval_required":
        raise ContextBuildError("module structure changes must remain suggestion-only")
    default_limit = payload.get("default_split_suggestion_chars", 50000)
    if not isinstance(default_limit, int) or default_limit <= 0:
        raise ContextBuildError("default_split_suggestion_chars must be a positive integer")
    modules = payload.get("modules")
    if not isinstance(modules, list) or not modules:
        raise ContextBuildError("MODULES.yaml modules must be a non-empty list")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, module in enumerate(modules):
        if not isinstance(module, dict):
            raise ContextBuildError(f"modules[{index}] must be a mapping")
        module_id = module.get("module_id")
        title = module.get("title")
        responsibility = module.get("responsibility")
        sources = module.get("sources")
        if not isinstance(module_id, str) or not module_id:
            raise ContextBuildError(f"modules[{index}].module_id must be non-empty")
        if module_id in seen:
            raise ContextBuildError(f"duplicate module_id: {module_id}")
        seen.add(module_id)
        if not isinstance(title, str) or not title:
            raise ContextBuildError(f"module {module_id} title must be non-empty")
        if not isinstance(responsibility, str) or not responsibility:
            raise ContextBuildError(f"module {module_id} responsibility must be non-empty")
        if not isinstance(sources, list) or not sources:
            raise ContextBuildError(f"module {module_id} sources must be a non-empty list")
        if not all(isinstance(source, dict) for source in sources):
            raise ContextBuildError(f"module {module_id} sources must contain mappings")
        limit = module.get("split_suggestion_chars", default_limit)
        if not isinstance(limit, int) or limit <= 0:
            raise ContextBuildError(
                f"module {module_id} split_suggestion_chars must be positive"
            )
        normalized.append(
            {
                "module_id": module_id,
                "title": title,
                "responsibility": responsibility,
                "sources": sources,
                "split_suggestion_chars": limit,
            }
        )
    return normalized, default_limit


def normalize_dependencies(
    payload: dict[str, Any],
    module_ids: Iterable[str],
) -> tuple[dict[str, tuple[str, ...]], list[dict[str, Any]]]:
    if payload.get("schema_version") != 1:
        raise ContextBuildError("DEPENDENCIES.yaml schema_version must be 1")
    if payload.get("relation") != "depends_on":
        raise ContextBuildError("minimal context core supports only relation: depends_on")
    raw = payload.get("depends_on")
    if not isinstance(raw, dict):
        raise ContextBuildError("DEPENDENCIES.yaml depends_on must be a mapping")
    ids = list(module_ids)
    id_set = set(ids)
    if set(raw) != id_set:
        missing = sorted(id_set - set(raw))
        unknown = sorted(set(raw) - id_set)
        raise ContextBuildError(
            f"dependency keys must match modules; missing={missing}, unknown={unknown}"
        )
    dependencies: dict[str, tuple[str, ...]] = {}
    for module_id in ids:
        values = raw[module_id]
        if not isinstance(values, list) or not all(
            isinstance(value, str) and value for value in values
        ):
            raise ContextBuildError(f"depends_on[{module_id}] must be a string list")
        if len(values) != len(set(values)):
            raise ContextBuildError(f"depends_on[{module_id}] contains duplicates")
        unknown = sorted(set(values) - id_set)
        if unknown:
            raise ContextBuildError(
                f"depends_on[{module_id}] references unknown modules: {unknown}"
            )
        if module_id in values:
            raise ContextBuildError(f"module {module_id} cannot depend on itself")
        dependencies[module_id] = tuple(values)
    validate_acyclic(dependencies)
    acceptance = payload.get("acceptance_targets", [])
    if not isinstance(acceptance, list) or not all(isinstance(x, dict) for x in acceptance):
        raise ContextBuildError("acceptance_targets must be a list of mappings")
    return dependencies, acceptance


def validate_acyclic(dependencies: dict[str, tuple[str, ...]]) -> None:
    state: dict[str, int] = {module_id: 0 for module_id in dependencies}
    stack: list[str] = []

    def visit(module_id: str) -> None:
        if state[module_id] == 2:
            return
        if state[module_id] == 1:
            start = stack.index(module_id)
            cycle = stack[start:] + [module_id]
            raise ContextBuildError("dependency cycle: " + " -> ".join(cycle))
        state[module_id] = 1
        stack.append(module_id)
        for dependency in dependencies[module_id]:
            visit(dependency)
        stack.pop()
        state[module_id] = 2

    for module_id in dependencies:
        visit(module_id)


def dependency_closure(
    target: str,
    dependencies: dict[str, tuple[str, ...]],
    module_order: Iterable[str],
) -> tuple[str, ...]:
    if target not in dependencies:
        raise ContextBuildError(f"unknown target module: {target}")
    selected: set[str] = set()

    def collect(module_id: str) -> None:
        if module_id in selected:
            return
        selected.add(module_id)
        for dependency in dependencies[module_id]:
            collect(dependency)

    collect(target)
    ordered: list[str] = []
    visited: set[str] = set()

    def emit(module_id: str) -> None:
        if module_id in visited or module_id not in selected:
            return
        for dependency in dependencies[module_id]:
            emit(dependency)
        visited.add(module_id)
        ordered.append(module_id)

    for module_id in module_order:
        emit(module_id)
    if target not in ordered:
        raise ContextBuildError(f"failed to order context closure for {target}")
    return tuple(ordered)


def render_module(module: dict[str, Any], chunks: list[SourceChunk]) -> ModuleSnapshot:
    module_id = module["module_id"]
    source_payload = {
        "module_id": module_id,
        "title": module["title"],
        "responsibility": module["responsibility"],
        "sources": [
            {"descriptor": chunk.descriptor, "text": chunk.text} for chunk in chunks
        ],
    }
    source_hash = canonical_hash(source_payload)
    parts = [
        f"# {module['title']}",
        "",
        "> Generated Stage 4 minimal-context shadow module. Do not edit manually.",
        "> `docs/handoff.md` and `experiments/registry.yaml` remain authoritative.",
        "",
        f"- Module ID: `{module_id}`",
        f"- Responsibility: {module['responsibility']}",
        f"- Source hash: `{source_hash}`",
        "",
    ]
    for index, chunk in enumerate(chunks, start=1):
        parts.extend(
            [
                f"## Source {index}: {chunk.label}",
                "",
                chunk.text.rstrip(),
                "",
            ]
        )
    payload = ("\n".join(parts).rstrip() + "\n").encode("utf-8")
    return ModuleSnapshot(
        module_id=module_id,
        title=module["title"],
        responsibility=module["responsibility"],
        source_hash=source_hash,
        snapshot_hash=sha256_bytes(payload),
        source_chars=sum(len(chunk.text) for chunk in chunks),
        bytes_payload=payload,
        source_labels=tuple(chunk.label for chunk in chunks),
    )


def render_graph_dot(
    modules: list[dict[str, Any]], dependencies: dict[str, tuple[str, ...]]
) -> bytes:
    lines = [
        "digraph stage4_minimal_context {",
        '  graph [rankdir="LR", label="Stage 4 Minimal Context Dependencies", labelloc="t"];',
        '  node [shape="box", style="rounded"];',
    ]
    for module in modules:
        module_id = module["module_id"]
        label = module["title"].replace("\\", "\\\\").replace('"', '\\"')
        lines.append(f'  "{module_id}" [label="{label}"];')
    for module in modules:
        target = module["module_id"]
        for dependency in dependencies[target]:
            lines.append(f'  "{target}" -> "{dependency}" [label="depends_on"];')
    lines.append("}")
    return ("\n".join(lines) + "\n").encode("utf-8")


def render_graph_markdown(
    modules: list[dict[str, Any]], dependencies: dict[str, tuple[str, ...]]
) -> bytes:
    lines = [
        "# Stage 4 Minimal Context Dependency Graph",
        "",
        "> Generated shadow view. `docs/handoff.md` remains authoritative.",
        "",
        "```mermaid",
        "flowchart LR",
    ]
    for module in modules:
        module_id = module["module_id"]
        title = module["title"].replace('"', "'")
        lines.append(f'  {module_id}["{title}"]')
    for module in modules:
        target = module["module_id"]
        for dependency in dependencies[target]:
            lines.append(f"  {target} -->|depends_on| {dependency}")
    lines.extend(["```", ""])
    return "\n".join(lines).encode("utf-8")


def all_registry_ids(repo_root: Path, path_value: str, collection: str) -> list[str]:
    entries = load_registry_collection(repo_root, path_value, collection)
    result: list[str] = []
    for entry in entries:
        item_id = registry_entry_id(entry)
        if item_id is None:
            raise ContextBuildError(
                f"registry entry without ID in {path_value}:{collection}"
            )
        result.append(item_id)
    return result


def generate_suggestions(
    repo_root: Path,
    modules: list[dict[str, Any]],
    snapshots: dict[str, ModuleSnapshot],
    dependencies: dict[str, tuple[str, ...]],
) -> tuple[dict[str, Any], ...]:
    mapped_canonical: set[str] = set()
    for module in modules:
        for source in module["sources"]:
            if (
                source.get("kind") == "registry_entries"
                and source.get("path") == "experiments/registry.yaml"
                and source.get("collection", "experiments") == "experiments"
            ):
                mapped_canonical.update(source.get("experiment_ids", []))
    canonical_ids = set(
        all_registry_ids(repo_root, "experiments/registry.yaml", "experiments")
    )
    suggestions: list[dict[str, Any]] = []
    for experiment_id in sorted(canonical_ids - mapped_canonical):
        suggestions.append(
            {
                "kind": "candidate_add_or_map_module",
                "object_id": experiment_id,
                "reason": "canonical registry experiment is not mapped to any minimal module",
                "automatic_action": False,
            }
        )
    inbound = {module["module_id"]: 0 for module in modules}
    for values in dependencies.values():
        for dependency in values:
            inbound[dependency] += 1
    for module in modules:
        module_id = module["module_id"]
        snapshot = snapshots[module_id]
        if snapshot.source_chars > module["split_suggestion_chars"]:
            suggestions.append(
                {
                    "kind": "candidate_split_module",
                    "object_id": module_id,
                    "reason": (
                        f"source content has {snapshot.source_chars} characters, above the "
                        f"configured {module['split_suggestion_chars']} threshold"
                    ),
                    "automatic_action": False,
                }
            )
        if (
            module_id != "global_core_governance"
            and not dependencies[module_id]
            and inbound[module_id] == 0
        ):
            suggestions.append(
                {
                    "kind": "candidate_review_orphan_module",
                    "object_id": module_id,
                    "reason": "module has no incoming or outgoing dependency",
                    "automatic_action": False,
                }
            )
    return tuple(suggestions)


def render_suggestions(suggestions: tuple[dict[str, Any], ...]) -> bytes:
    lines = [
        "# Stage 4 Minimal Context Structure Suggestions",
        "",
        "> Generated advisory report. No suggestion changes module structure automatically.",
        "> Adding, removing, splitting, merging, or rewiring modules requires human approval.",
        "",
    ]
    if not suggestions:
        lines.append("No structural suggestions were produced.")
    else:
        for suggestion in suggestions:
            lines.extend(
                [
                    f"## {suggestion['kind']}: `{suggestion['object_id']}`",
                    "",
                    f"- Reason: {suggestion['reason']}",
                    "- Automatic action: `false`",
                    "",
                ]
            )
    return ("\n".join(lines).rstrip() + "\n").encode("utf-8")


def validate_acceptance_targets(
    acceptance: list[dict[str, Any]],
    dependencies: dict[str, tuple[str, ...]],
    module_order: tuple[str, ...],
) -> tuple[dict[str, Any], ...]:
    results: list[dict[str, Any]] = []
    for index, item in enumerate(acceptance):
        target = item.get("target")
        must_include = item.get("must_include", [])
        must_exclude = item.get("must_exclude", [])
        if not isinstance(target, str) or target not in dependencies:
            raise ContextBuildError(f"acceptance_targets[{index}] has unknown target")
        if not isinstance(must_include, list) or not all(
            isinstance(value, str) for value in must_include
        ):
            raise ContextBuildError(
                f"acceptance target {target} must_include must be a string list"
            )
        if not isinstance(must_exclude, list) or not all(
            isinstance(value, str) for value in must_exclude
        ):
            raise ContextBuildError(
                f"acceptance target {target} must_exclude must be a string list"
            )
        closure = dependency_closure(target, dependencies, module_order)
        closure_set = set(closure)
        missing = sorted(set(must_include) - closure_set)
        leaked = sorted(set(must_exclude) & closure_set)
        if missing or leaked:
            raise ContextBuildError(
                f"acceptance target {target} failed: missing={missing}, leaked={leaked}"
            )
        results.append(
            {
                "target": target,
                "closure": list(closure),
                "must_include": must_include,
                "must_exclude": must_exclude,
                "status": "pass",
            }
        )
    return tuple(results)


def build_plan(
    repo_root: Path,
    modules_path: Path = DEFAULT_MODULES,
    dependencies_path: Path = DEFAULT_DEPENDENCIES,
    output_path: Path = DEFAULT_OUTPUT,
) -> BuildPlan:
    repo_root = repo_root.resolve()
    modules_file = safe_repo_path(repo_root, str(modules_path), "modules config")
    dependencies_file = safe_repo_path(
        repo_root, str(dependencies_path), "dependencies config"
    )
    output_root = safe_repo_path(repo_root, str(output_path), "generated output")
    modules_payload = load_yaml(modules_file, "minimal module configuration")
    dependencies_payload = load_yaml(
        dependencies_file, "minimal dependency configuration"
    )
    modules, _ = normalize_modules(modules_payload)
    module_order = tuple(module["module_id"] for module in modules)
    dependencies, acceptance = normalize_dependencies(
        dependencies_payload, module_order
    )
    snapshots: dict[str, ModuleSnapshot] = {}
    outputs: dict[Path, bytes] = {}
    for module in modules:
        chunks = [extract_source(repo_root, source) for source in module["sources"]]
        snapshot = render_module(module, chunks)
        snapshots[snapshot.module_id] = snapshot
        outputs[Path("modules") / f"{snapshot.module_id}.md"] = snapshot.bytes_payload
    suggestions = generate_suggestions(repo_root, modules, snapshots, dependencies)
    acceptance_results = validate_acceptance_targets(
        acceptance, dependencies, module_order
    )
    graph_payload = {
        "modules": [
            {
                "module_id": module["module_id"],
                "title": module["title"],
                "responsibility": module["responsibility"],
                "depends_on": list(dependencies[module["module_id"]]),
            }
            for module in modules
        ]
    }
    graph_hash = canonical_hash(graph_payload)
    outputs[Path("DEPENDENCY_GRAPH.dot")] = render_graph_dot(modules, dependencies)
    outputs[Path("DEPENDENCY_GRAPH.md")] = render_graph_markdown(modules, dependencies)
    outputs[Path("STRUCTURE_SUGGESTIONS.md")] = render_suggestions(suggestions)
    input_files = {
        str(modules_path): sha256_bytes(modules_file.read_bytes()),
        str(dependencies_path): sha256_bytes(dependencies_file.read_bytes()),
        "docs/handoff.md": sha256_bytes((repo_root / "docs/handoff.md").read_bytes()),
        "experiments/registry.yaml": sha256_bytes(
            (repo_root / "experiments/registry.yaml").read_bytes()
        ),
    }
    index = {
        "schema_version": 1,
        "policy_id": modules_payload.get("policy_id"),
        "authority": "non_authoritative_stage4_minimal_context_shadow",
        "research_master": "docs/handoff.md",
        "structure_change_policy": "suggestion_only_human_approval_required",
        "input_hashes": input_files,
        "graph_hash": graph_hash,
        "module_order": list(module_order),
        "modules": [
            {
                "module_id": module_id,
                "source_hash": snapshots[module_id].source_hash,
                "snapshot_hash": snapshots[module_id].snapshot_hash,
                "source_chars": snapshots[module_id].source_chars,
                "source_labels": list(snapshots[module_id].source_labels),
                "depends_on": list(dependencies[module_id]),
            }
            for module_id in module_order
        ],
        "suggestion_count": len(suggestions),
        "acceptance_targets": list(acceptance_results),
        "generated_files": sorted(
            [str(path) for path in outputs] + ["MODULE_INDEX.json"]
        ),
    }
    outputs[Path("MODULE_INDEX.json")] = json_bytes(index)
    # Validate the output root only after all source reads. It need not exist yet.
    _ = output_root
    return BuildPlan(
        outputs=outputs,
        snapshots=snapshots,
        dependencies=dependencies,
        module_order=module_order,
        graph_hash=graph_hash,
        suggestions=suggestions,
        acceptance_results=acceptance_results,
    )


def existing_generated_files(output_root: Path) -> set[Path]:
    if not output_root.exists():
        return set()
    if not output_root.is_dir():
        raise ContextBuildError(f"generated output is not a directory: {output_root}")
    return {
        path.relative_to(output_root)
        for path in output_root.rglob("*")
        if path.is_file()
    }


def check_generated(output_root: Path, plan: BuildPlan) -> list[str]:
    problems: list[str] = []
    expected = set(plan.outputs)
    actual = existing_generated_files(output_root)
    for relative in sorted(expected - actual):
        problems.append(f"missing generated file: {relative}")
    for relative in sorted(actual - expected):
        problems.append(f"unexpected generated file: {relative}")
    for relative in sorted(expected & actual):
        current = (output_root / relative).read_bytes()
        if current != plan.outputs[relative]:
            problems.append(f"stale generated file: {relative}")
    return problems


def write_generated(output_root: Path, plan: BuildPlan) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    expected = set(plan.outputs)
    actual = existing_generated_files(output_root)
    removed: list[str] = []
    for relative in sorted(actual - expected):
        (output_root / relative).unlink()
        removed.append(str(relative))
    refreshed_modules: list[str] = []
    reused_modules: list[str] = []
    refreshed_supporting: list[str] = []
    for relative in sorted(expected):
        path = output_root / relative
        payload = plan.outputs[relative]
        if path.is_file() and path.read_bytes() == payload:
            if relative.parts[0] == "modules":
                reused_modules.append(relative.stem)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_bytes(payload)
        temporary.replace(path)
        if relative.parts[0] == "modules":
            refreshed_modules.append(relative.stem)
        else:
            refreshed_supporting.append(str(relative))
    for directory in sorted(
        [path for path in output_root.rglob("*") if path.is_dir()],
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        if not any(directory.iterdir()):
            directory.rmdir()
    return {
        "status": "PASS",
        "refreshed_modules": sorted(refreshed_modules),
        "reused_modules": sorted(reused_modules),
        "refreshed_supporting_files": sorted(refreshed_supporting),
        "removed_stale_files": removed,
        "module_count": len(plan.module_order),
        "edge_count": sum(len(values) for values in plan.dependencies.values()),
        "graph_hash": plan.graph_hash,
        "suggestion_count": len(plan.suggestions),
    }


def context_pack_bytes(plan: BuildPlan, target: str) -> bytes:
    closure = dependency_closure(target, plan.dependencies, plan.module_order)
    total_chars = sum(plan.snapshots[module_id].source_chars for module_id in closure)
    lines = [
        "# Stage 4 Minimal Context Pack",
        "",
        "> Generated non-authoritative shadow context. Do not edit or treat as the",
        "> research master.",
        "> `docs/handoff.md` and `experiments/registry.yaml` remain authoritative.",
        "",
        f"- Target module: `{target}`",
        f"- Dependency order: {', '.join(f'`{item}`' for item in closure)}",
        f"- Included module count: {len(closure)}",
        f"- Mapped source characters before wrapper text: {total_chars}",
        f"- Dependency graph hash: `{plan.graph_hash}`",
        "",
    ]
    for module_id in closure:
        snapshot = plan.snapshots[module_id]
        lines.extend(
            [
                "---",
                "",
                f"## Module `{module_id}`",
                "",
                snapshot.bytes_payload.decode("utf-8").rstrip(),
                "",
            ]
        )
    return ("\n".join(lines).rstrip() + "\n").encode("utf-8")


def report_for_plan(plan: BuildPlan, output_root: Path) -> dict[str, Any]:
    return {
        "status": "PASS",
        "module_count": len(plan.module_order),
        "edge_count": sum(len(values) for values in plan.dependencies.values()),
        "graph_hash": plan.graph_hash,
        "suggestion_count": len(plan.suggestions),
        "generated_output": str(output_root),
        "acceptance_targets": list(plan.acceptance_results),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--modules", type=Path, default=DEFAULT_MODULES)
    parser.add_argument("--dependencies", type=Path, default=DEFAULT_DEPENDENCIES)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("build", help="refresh dirty module snapshots and all small views")
    subparsers.add_parser("check", help="verify generated output without writing")
    context = subparsers.add_parser("context", help="write one dependency-closed context pack")
    context.add_argument("--target", required=True)
    context.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    output_root = safe_repo_path(repo_root, str(args.output_root), "generated output")
    try:
        plan = build_plan(
            repo_root,
            modules_path=args.modules,
            dependencies_path=args.dependencies,
            output_path=args.output_root,
        )
        if args.command == "build":
            report = write_generated(output_root, plan)
        elif args.command == "check":
            problems = check_generated(output_root, plan)
            report = report_for_plan(plan, output_root)
            report["problems"] = problems
            report["status"] = "PASS" if not problems else "FAIL"
            if problems:
                if args.json:
                    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
                else:
                    print("Stage 4 minimal context check: FAIL", file=sys.stderr)
                    for problem in problems:
                        print(f"- {problem}", file=sys.stderr)
                return 1
        elif args.command == "context":
            problems = check_generated(output_root, plan)
            if problems:
                raise ContextBuildError(
                    "generated minimal modules are stale; run the build command first: "
                    + "; ".join(problems)
                )
            payload = context_pack_bytes(plan, args.target)
            destination = args.output
            if not destination.is_absolute():
                destination = repo_root / destination
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
            closure = dependency_closure(
                args.target, plan.dependencies, plan.module_order
            )
            report = {
                "status": "PASS",
                "target": args.target,
                "output": str(destination),
                "included_modules": list(closure),
                "context_sha256": sha256_bytes(payload),
                "context_bytes": len(payload),
                "full_handoff_bytes": (repo_root / "docs/handoff.md").stat().st_size,
            }
        else:  # pragma: no cover - argparse enforces the command set.
            raise ContextBuildError(f"unknown command: {args.command}")
    except (ContextBuildError, OSError) as exc:
        if args.json:
            print(
                json.dumps(
                    {"status": "FAIL", "error": str(exc)},
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            print(f"Stage 4 minimal context: FAIL: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print("Stage 4 minimal context: PASS")
        for key, value in report.items():
            if key != "status":
                print(f"- {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
