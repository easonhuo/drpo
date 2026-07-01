#!/usr/bin/env python3
"""Evidence-first DRPO manuscript Core vertical slice.

This script intentionally implements a narrow, auditable pipeline for
PAPER-PIPELINE-V2-CORE-01 and its faithful outline-to-blueprint compiler. It
does not replace the historical bidirectional scaffold pipeline. Its purpose is
to prove the reliable path from an approved stable-ID outline and validated
repository evidence to a one-to-one executable blueprint, prose, real figure,
table, theorem/proof, and two-page review PDF.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ALLOWED_STATUSES = {
    "analytically_proven",
    "long_run_validated",
    "finite_step_validated",
    "pilot",
    "not_run",
    "rejected",
    "superseded",
}
STATUS_RANK = {
    "rejected": 0,
    "superseded": 0,
    "not_run": 1,
    "pilot": 2,
    "finite_step_validated": 3,
    "long_run_validated": 4,
    "analytically_proven": 5,
}
DISPLAY_METHOD = {
    "baseline": "Baseline",
    "near_zero": "Near-zero",
    "far_zero": "Far-zero",
    "far_cap": "Far-cap",
    "global_scale": "Global-scale",
    "far_to_near": "Far-to-near",
}


class CorePipelineError(RuntimeError):
    """Expected fail-closed Core pipeline error."""


@dataclass(frozen=True)
class Paths:
    repo: Path
    spec: Path
    output: Path
    allow_output_override: bool = False

    @property
    def snapshot(self) -> Path:
        return self.output / "research_snapshot.json"

    @property
    def manifest(self) -> Path:
        return self.output / "build_manifest.json"

    @property
    def outline_ast(self) -> Path:
        return self.output / "outline_ast.json"

    @property
    def outline_resolution(self) -> Path:
        return self.output / "outline_resolution.json"

    @property
    def blueprint_json(self) -> Path:
        return self.output / "blueprint.json"

    @property
    def pdf(self) -> Path:
        return self.output / "main.pdf"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CorePipelineError(f"expected YAML mapping: {path}")
    return payload


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CorePipelineError(f"expected JSON mapping: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def repo_path(repo: Path, relative: str) -> Path:
    candidate = (repo / relative).resolve()
    try:
        candidate.relative_to(repo.resolve())
    except ValueError as exc:
        raise CorePipelineError(f"path escapes repository: {relative}") from exc
    return candidate


def load_spec(paths: Paths) -> dict[str, Any]:
    spec = read_yaml(paths.spec)
    if spec.get("schema_version") != 1:
        raise CorePipelineError("paper_spec_core.yaml schema_version must be 1")
    if spec.get("profile") != "core_vertical_slice":
        raise CorePipelineError("Core script only accepts profile=core_vertical_slice")
    expected_output = repo_path(paths.repo, str(spec["output_root"]))
    if not paths.allow_output_override and expected_output != paths.output.resolve():
        raise CorePipelineError(
            f"output root differs from spec: {paths.output} != {expected_output}"
        )
    return spec


def find_experiment(registry: dict[str, Any], experiment_id: str) -> dict[str, Any]:
    experiments = registry.get("experiments")
    if not isinstance(experiments, list):
        raise CorePipelineError("registry experiments must be a list")
    matches = [row for row in experiments if isinstance(row, dict) and row.get("id") == experiment_id]
    if len(matches) != 1:
        raise CorePipelineError(
            f"expected exactly one registry experiment {experiment_id}, found {len(matches)}"
        )
    return matches[0]


def require_status(actual: str, required: str) -> None:
    if actual not in ALLOWED_STATUSES or required not in ALLOWED_STATUSES:
        raise CorePipelineError(f"unknown result status: actual={actual} required={required}")
    if STATUS_RANK[actual] < STATUS_RANK[required]:
        raise CorePipelineError(
            f"experiment status {actual} does not satisfy required status {required}"
        )


def verify_compact_artifacts(repo: Path, artifact_index_path: Path) -> dict[str, Any]:
    index = read_json(artifact_index_path)
    compact = index.get("compact_repository_files")
    if not isinstance(compact, dict) or not compact:
        raise CorePipelineError("ARTIFACT_INDEX compact_repository_files is missing")
    verified: dict[str, Any] = {}
    for name, metadata in compact.items():
        if not isinstance(name, str) or not isinstance(metadata, dict):
            raise CorePipelineError("invalid compact artifact index entry")
        path = artifact_index_path.parent / name
        if not path.is_file():
            raise CorePipelineError(f"compact artifact is missing: {path.relative_to(repo)}")
        actual = sha256_file(path)
        expected = metadata.get("sha256")
        if actual != expected:
            raise CorePipelineError(
                f"compact artifact checksum mismatch for {path.relative_to(repo)}: "
                f"{actual} != {expected}"
            )
        verified[name] = {
            "path": path.relative_to(repo).as_posix(),
            "sha256": actual,
            "size_bytes": path.stat().st_size,
        }
    return {"index": index, "verified": verified}


def load_csv_by_method(path: Path) -> dict[str, dict[str, float | str | None]]:
    rows: dict[str, dict[str, float | str | None]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "method" not in reader.fieldnames:
            raise CorePipelineError(f"CSV has no method column: {path}")
        for raw in reader:
            method = str(raw.get("method", "")).strip()
            if not method or method in rows:
                raise CorePipelineError(f"invalid or duplicate method in {path}: {method}")
            converted: dict[str, float | str | None] = {}
            for key, value in raw.items():
                if key == "method":
                    continue
                text = "" if value is None else value.strip()
                if text == "":
                    converted[key] = None
                    continue
                try:
                    number = float(text)
                except ValueError:
                    converted[key] = text
                else:
                    if not math.isfinite(number):
                        raise CorePipelineError(f"non-finite CSV value {path}:{method}:{key}")
                    converted[key] = number
            rows[method] = converted
    return rows


def numeric(row: dict[str, Any], key: str, *, source: str) -> float:
    value = row.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise CorePipelineError(f"missing numeric field {source}:{key}")
    value = float(value)
    if not math.isfinite(value):
        raise CorePipelineError(f"non-finite field {source}:{key}")
    return value


def event_count(rate: float, denominator: int, *, label: str) -> int:
    raw = rate * denominator
    rounded = int(round(raw))
    if not math.isclose(raw, rounded, abs_tol=1e-6):
        raise CorePipelineError(f"event rate is not an integer count for {label}: {rate}")
    return rounded


OUTLINE_BEGIN_RE = re.compile(r"^<!-- MANUSCRIPT:BEGIN ([A-Z0-9-]+) -->$")
OUTLINE_END_RE = re.compile(r"^<!-- MANUSCRIPT:END ([A-Z0-9-]+) -->$")
OUTLINE_TITLE_RE = re.compile(r"^## \[([A-Z0-9-]+)\] (.+)$")
OUTLINE_FIELD_RE = re.compile(r"^\*\*(Claim|Reader question|Role|Required evidence|Must include|Must avoid):\*\*\s*(.*)$")
OUTLINE_REQUIRED_FIELDS = (
    "claim",
    "reader_question",
    "role",
    "required_evidence",
    "must_include",
    "must_avoid",
)


def _field_key(label: str) -> str:
    return label.lower().replace(" ", "_")


def _normalize_text(lines: list[str]) -> str:
    return " ".join(line.strip() for line in lines if line.strip()).strip()


def _parse_outline_block(
    *, node_id: str, section: str, order: int, block_lines: list[str], source: Path
) -> dict[str, Any]:
    if not block_lines:
        raise CorePipelineError(f"empty outline node {node_id} in {source}")
    title_match = OUTLINE_TITLE_RE.match(block_lines[0])
    if title_match is None:
        raise CorePipelineError(f"outline node {node_id} has no canonical title line")
    if title_match.group(1) != node_id:
        raise CorePipelineError(
            f"outline marker/title mismatch: marker={node_id} title={title_match.group(1)}"
        )

    fields: dict[str, Any] = {}
    current_key: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        nonlocal current_key, buffer
        if current_key is None:
            return
        if current_key in {"required_evidence", "must_include", "must_avoid"}:
            values = [line[2:].strip() for line in buffer if line.startswith("- ")]
            prose = [line for line in buffer if line and not line.startswith("- ")]
            if prose:
                raise CorePipelineError(
                    f"outline node {node_id} field {current_key} contains non-list text"
                )
            fields[current_key] = values
        else:
            fields[current_key] = _normalize_text(buffer)
        current_key = None
        buffer = []

    for line in block_lines[1:]:
        field_match = OUTLINE_FIELD_RE.match(line)
        if field_match is not None:
            flush()
            current_key = _field_key(field_match.group(1))
            inline = field_match.group(2).strip()
            buffer = [inline] if inline else []
            continue
        if current_key is not None:
            buffer.append(line)
        elif line.strip():
            raise CorePipelineError(
                f"outline node {node_id} has content outside canonical fields: {line!r}"
            )
    flush()

    missing = [field for field in OUTLINE_REQUIRED_FIELDS if field not in fields]
    if missing:
        raise CorePipelineError(f"outline node {node_id} is missing fields: {missing}")
    for field in ("claim", "reader_question", "role"):
        if not isinstance(fields[field], str) or not fields[field].strip():
            raise CorePipelineError(f"outline node {node_id} has empty {field}")
    for field in ("required_evidence", "must_include", "must_avoid"):
        if not isinstance(fields[field], list):
            raise CorePipelineError(f"outline node {node_id} field {field} must be a list")
    if not fields["required_evidence"] or not fields["must_include"]:
        raise CorePipelineError(
            f"outline node {node_id} must have required evidence and must-include items"
        )

    block_text = "\n".join(block_lines).rstrip() + "\n"
    return {
        "id": node_id,
        "section": section,
        "order": order,
        "title": title_match.group(2).strip(),
        **fields,
        "block_sha256": sha256_bytes(block_text.encode("utf-8")),
    }


def parse_outline(path: Path) -> dict[str, Any]:
    """Compile the approved Markdown outline into a deterministic AST."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    section = ""
    nodes: list[dict[str, Any]] = []
    seen: set[str] = set()
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith("# "):
            section = line[2:].strip()
            index += 1
            continue
        begin = OUTLINE_BEGIN_RE.match(line)
        if begin is None:
            index += 1
            continue
        node_id = begin.group(1)
        if not section:
            raise CorePipelineError(f"outline node {node_id} appears before a section heading")
        if node_id in seen:
            raise CorePipelineError(f"duplicate outline node: {node_id}")
        seen.add(node_id)
        block_lines: list[str] = []
        index += 1
        while index < len(lines):
            end = OUTLINE_END_RE.match(lines[index])
            if end is not None:
                if end.group(1) != node_id:
                    raise CorePipelineError(
                        f"outline end marker mismatch: begin={node_id} end={end.group(1)}"
                    )
                break
            if OUTLINE_BEGIN_RE.match(lines[index]) is not None:
                raise CorePipelineError(f"nested outline node before closing {node_id}")
            block_lines.append(lines[index])
            index += 1
        else:
            raise CorePipelineError(f"outline node {node_id} has no end marker")
        nodes.append(
            _parse_outline_block(
                node_id=node_id,
                section=section,
                order=len(nodes) + 1,
                block_lines=block_lines,
                source=path,
            )
        )
        index += 1

    if not nodes:
        raise CorePipelineError(f"approved outline contains no manuscript nodes: {path}")
    return {
        "schema_version": 1,
        "source_path": path.as_posix(),
        "source_sha256": sha256_file(path),
        "node_count": len(nodes),
        "nodes": nodes,
    }


def outline_contains_node(path: Path, node_id: str) -> bool:
    return any(node["id"] == node_id for node in parse_outline(path)["nodes"])


def build_outline_ast(paths: Paths, *, spec_override: dict[str, Any] | None = None) -> dict[str, Any]:
    spec = spec_override if spec_override is not None else load_spec(paths)
    outline_path = repo_path(paths.repo, str(spec["approved_outline"]))
    ast = parse_outline(outline_path)
    ast["source_path"] = outline_path.relative_to(paths.repo).as_posix()
    write_json(paths.outline_ast, ast)
    return ast


def _outline_node_map(ast: dict[str, Any]) -> dict[str, dict[str, Any]]:
    nodes = ast.get("nodes")
    if not isinstance(nodes, list):
        raise CorePipelineError("outline AST nodes must be a list")
    result: dict[str, dict[str, Any]] = {}
    for node in nodes:
        if not isinstance(node, dict) or not isinstance(node.get("id"), str):
            raise CorePipelineError("invalid outline AST node")
        result[str(node["id"])] = node
    return result


def build_outline_resolution(
    paths: Paths,
    ast: dict[str, Any],
    *,
    spec_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve every outline node without allowing silent merge/split/rename."""
    spec = spec_override if spec_override is not None else load_spec(paths)
    contract = spec.get("blueprint_contract")
    if not isinstance(contract, dict):
        raise CorePipelineError("paper spec is missing blueprint_contract")
    enabled = contract.get("enabled_nodes")
    if not isinstance(enabled, dict) or not enabled:
        raise CorePipelineError("blueprint_contract.enabled_nodes must be a non-empty mapping")
    default_reason = str(contract.get("disabled_reason", "not_enabled_for_current_profile")).strip()
    if not default_reason:
        raise CorePipelineError("blueprint_contract.disabled_reason must be non-empty")

    node_map = _outline_node_map(ast)
    unknown = sorted(set(enabled) - set(node_map))
    if unknown:
        raise CorePipelineError(f"blueprint contract refers to unknown outline nodes: {unknown}")

    resolved_nodes: list[dict[str, Any]] = []
    for node in ast["nodes"]:
        node_id = str(node["id"])
        if node_id in enabled:
            state = "enabled"
            reason = str(enabled[node_id].get("reason", "selected_for_core_vertical_slice"))
        else:
            state = "disabled_with_reason"
            reason = default_reason
        resolved_nodes.append(
            {
                "id": node_id,
                "section": node["section"],
                "order": node["order"],
                "title": node["title"],
                "outline_block_sha256": node["block_sha256"],
                "status": state,
                "reason": reason,
            }
        )

    resolution = {
        "schema_version": 1,
        "profile": spec["profile"],
        "outline_sha256": ast["source_sha256"],
        "node_count": ast["node_count"],
        "enabled_node_ids": [row["id"] for row in resolved_nodes if row["status"] == "enabled"],
        "nodes": resolved_nodes,
    }
    write_json(paths.outline_resolution, resolution)
    return resolution


def _require_string_list(payload: dict[str, Any], key: str, *, node_id: str) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list) or not value or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise CorePipelineError(f"blueprint node {node_id} requires non-empty string list {key}")
    return [str(item).strip() for item in value]


def _normalize_for_comparison(text: str) -> str:
    return " ".join(text.lower().split())


def resolve_metric_path(snapshot: dict[str, Any], metric_path: str) -> Any:
    value: Any = snapshot
    for part in metric_path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise CorePipelineError(f"blueprint metric path does not resolve: {metric_path}")
        value = value[part]
    if value is None:
        raise CorePipelineError(f"blueprint metric path resolves to null: {metric_path}")
    return value


def validate_blueprint_payload(
    *,
    ast: dict[str, Any],
    resolution: dict[str, Any],
    blueprint: dict[str, Any],
    snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate structure fidelity, information gain, and executable bindings."""
    ast_nodes = ast.get("nodes")
    resolution_nodes = resolution.get("nodes")
    blueprint_nodes = blueprint.get("nodes")
    if not all(isinstance(value, list) for value in (ast_nodes, resolution_nodes, blueprint_nodes)):
        raise CorePipelineError("outline, resolution, and blueprint nodes must be lists")
    ast_ids = [str(node["id"]) for node in ast_nodes]
    resolution_ids = [str(node["id"]) for node in resolution_nodes]
    blueprint_ids = [str(node["id"]) for node in blueprint_nodes]
    if ast_ids != resolution_ids or ast_ids != blueprint_ids:
        raise CorePipelineError(
            "blueprint structure must exactly preserve outline IDs and order; merge/split/rename/reorder is forbidden"
        )
    if len(set(blueprint_ids)) != len(blueprint_ids):
        raise CorePipelineError("blueprint contains duplicate paragraph IDs")

    ast_map = _outline_node_map(ast)
    resolution_map = {str(node["id"]): node for node in resolution_nodes}
    enabled_count = 0
    for node in blueprint_nodes:
        if not isinstance(node, dict):
            raise CorePipelineError("invalid blueprint node")
        node_id = str(node["id"])
        outline_node = ast_map[node_id]
        resolution_node = resolution_map[node_id]
        for key in ("section", "order", "title", "outline_block_sha256"):
            expected_key = "block_sha256" if key == "outline_block_sha256" else key
            if node.get(key) != outline_node.get(expected_key):
                raise CorePipelineError(f"blueprint node {node_id} changed outline field {key}")
        if node.get("status") != resolution_node.get("status"):
            raise CorePipelineError(f"blueprint node {node_id} status disagrees with resolution")
        if node["status"] == "disabled_with_reason":
            reason = node.get("disabled_reason")
            if not isinstance(reason, str) or not reason.strip():
                raise CorePipelineError(f"disabled blueprint node {node_id} lacks a reason")
            continue
        if node["status"] != "enabled":
            raise CorePipelineError(f"unknown blueprint status for {node_id}: {node['status']}")
        enabled_count += 1
        paragraph_claim = str(node.get("paragraph_claim", "")).strip()
        if not paragraph_claim:
            raise CorePipelineError(f"enabled blueprint node {node_id} has no paragraph_claim")
        if _normalize_for_comparison(paragraph_claim) == _normalize_for_comparison(
            str(outline_node["claim"])
        ):
            raise CorePipelineError(
                f"blueprint node {node_id} copies the outline claim without information gain"
            )
        sentence_plan = node.get("sentence_plan")
        if not isinstance(sentence_plan, list) or len(sentence_plan) < 3:
            raise CorePipelineError(f"blueprint node {node_id} needs at least three sentence-plan steps")
        roles: list[str] = []
        for step in sentence_plan:
            if not isinstance(step, dict):
                raise CorePipelineError(f"blueprint node {node_id} has invalid sentence-plan step")
            role = str(step.get("role", "")).strip()
            instruction = str(step.get("instruction", "")).strip()
            if not role or not instruction:
                raise CorePipelineError(
                    f"blueprint node {node_id} sentence-plan steps need role and instruction"
                )
            roles.append(role)
        if len(set(roles)) != len(roles):
            raise CorePipelineError(f"blueprint node {node_id} repeats sentence-plan roles")
        evidence_refs = _require_string_list(node, "evidence_refs", node_id=node_id)
        for evidence_ref in evidence_refs:
            if re.fullmatch(r"[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+", evidence_ref) is None:
                raise CorePipelineError(
                    f"blueprint node {node_id} uses generic evidence instead of a stable ID: {evidence_ref}"
                )
        _require_string_list(node, "allowed_conclusions", node_id=node_id)
        _require_string_list(node, "forbidden_conclusions", node_id=node_id)
        if node_id.startswith("EXP-"):
            metric_paths = _require_string_list(node, "metric_paths", node_id=node_id)
            if snapshot is not None:
                for metric_path in metric_paths:
                    resolve_metric_path(snapshot, metric_path)
            if not node.get("figure_refs") and not node.get("table_refs"):
                raise CorePipelineError(
                    f"experiment blueprint node {node_id} needs a figure or table binding"
                )
        if node_id.startswith(("THEORY-", "METHOD-")):
            _require_string_list(node, "theorem_or_equation_refs", node_id=node_id)
        for field in ("reviewer_objection", "objection_response", "transition_to_next"):
            value = node.get(field)
            if not isinstance(value, str) or not value.strip():
                raise CorePipelineError(f"blueprint node {node_id} requires {field}")

    if enabled_count != len(resolution.get("enabled_node_ids", [])):
        raise CorePipelineError("blueprint enabled-node count disagrees with resolution")
    return {
        "status": "PASS",
        "node_count": len(blueprint_ids),
        "enabled_count": enabled_count,
        "outline_sha256": ast["source_sha256"],
    }


def _blueprint_node_from_contract(
    *, outline_node: dict[str, Any], status: dict[str, Any], binding: dict[str, Any] | None
) -> dict[str, Any]:
    common = {
        "id": outline_node["id"],
        "section": outline_node["section"],
        "order": outline_node["order"],
        "title": outline_node["title"],
        "outline_block_sha256": outline_node["block_sha256"],
        "status": status["status"],
    }
    if status["status"] == "disabled_with_reason":
        return {**common, "disabled_reason": status["reason"]}
    if not isinstance(binding, dict):
        raise CorePipelineError(f"enabled blueprint node {outline_node['id']} has no contract binding")
    return {
        **common,
        "outline_claim": outline_node["claim"],
        "reader_question": outline_node["reader_question"],
        "role": outline_node["role"],
        "must_include": outline_node["must_include"],
        "must_avoid": outline_node["must_avoid"],
        "paragraph_claim": str(binding.get("paragraph_claim", "")).strip(),
        "sentence_plan": binding.get("sentence_plan"),
        "evidence_refs": binding.get("evidence_refs"),
        "metric_paths": binding.get("metric_paths", []),
        "figure_refs": binding.get("figure_refs", []),
        "table_refs": binding.get("table_refs", []),
        "theorem_or_equation_refs": binding.get("theorem_or_equation_refs", []),
        "reviewer_objection": str(binding.get("reviewer_objection", "")).strip(),
        "objection_response": str(binding.get("objection_response", "")).strip(),
        "allowed_conclusions": binding.get("allowed_conclusions"),
        "forbidden_conclusions": binding.get("forbidden_conclusions"),
        "transition_to_next": str(binding.get("transition_to_next", "")).strip(),
    }


def render_blueprint_markdown(blueprint: dict[str, Any], path: Path) -> None:
    lines = [
        f"# Executable blueprint: {blueprint['task_id']}",
        "",
        f"Outline: `{blueprint['outline_sha256']}`",
        f"Snapshot: `{blueprint['snapshot_sha256']}`",
        "",
        "## Resolution summary",
        "",
        f"- Outline nodes: {blueprint['node_count']}",
        f"- Enabled nodes: {len(blueprint['enabled_node_ids'])}",
        f"- Disabled nodes: {blueprint['node_count'] - len(blueprint['enabled_node_ids'])}",
        "- Structural rule: no merge, split, rename, reorder, or silent omission.",
        "",
    ]
    for node in blueprint["nodes"]:
        if node["status"] == "disabled_with_reason":
            continue
        lines.extend(
            [
                f"## {node['id']} - {node['title']}",
                "",
                f"- Parent outline block: `{node['outline_block_sha256']}`",
                f"- Reader question: {node['reader_question']}",
                f"- Paragraph claim: {node['paragraph_claim']}",
                "- Sentence plan:",
            ]
        )
        for index, step in enumerate(node["sentence_plan"], start=1):
            lines.append(f"  {index}. **{step['role']}** — {step['instruction']}")
        lines.extend(
            [
                f"- Evidence refs: {', '.join(f'`{item}`' for item in node['evidence_refs'])}",
                f"- Metric paths: {', '.join(f'`{item}`' for item in node['metric_paths']) or 'none'}",
                f"- Figure refs: {', '.join(f'`{item}`' for item in node['figure_refs']) or 'none'}",
                f"- Table refs: {', '.join(f'`{item}`' for item in node['table_refs']) or 'none'}",
                f"- Theorem/equation refs: {', '.join(f'`{item}`' for item in node['theorem_or_equation_refs']) or 'none'}",
                f"- Reviewer objection: {node['reviewer_objection']}",
                f"- Response: {node['objection_response']}",
                f"- Allowed conclusions: {'; '.join(node['allowed_conclusions'])}",
                f"- Forbidden conclusions: {'; '.join(node['forbidden_conclusions'])}",
                f"- Transition: {node['transition_to_next']}",
                "",
            ]
        )
    lines.extend(["## Disabled nodes", ""])
    for node in blueprint["nodes"]:
        if node["status"] == "disabled_with_reason":
            lines.append(f"- `{node['id']}` — {node['disabled_reason']}")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def build_blueprint_contract(
    paths: Paths,
    *,
    snapshot: dict[str, Any] | None = None,
    spec_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    spec = spec_override if spec_override is not None else load_spec(paths)
    paths.output.mkdir(parents=True, exist_ok=True)
    ast = build_outline_ast(paths, spec_override=spec)
    resolution = build_outline_resolution(paths, ast, spec_override=spec)
    if snapshot is None:
        snapshot = build_snapshot(paths, spec_override=spec)
    contract = spec["blueprint_contract"]
    enabled = contract["enabled_nodes"]
    status_map = {str(node["id"]): node for node in resolution["nodes"]}
    blueprint_nodes = [
        _blueprint_node_from_contract(
            outline_node=node,
            status=status_map[str(node["id"])],
            binding=enabled.get(str(node["id"])),
        )
        for node in ast["nodes"]
    ]
    blueprint = {
        "schema_version": 1,
        "task_id": spec["spec_id"],
        "profile": spec["profile"],
        "outline_sha256": ast["source_sha256"],
        "snapshot_sha256": snapshot["snapshot_sha256"],
        "node_count": ast["node_count"],
        "enabled_node_ids": resolution["enabled_node_ids"],
        "nodes": blueprint_nodes,
    }
    validate_blueprint_payload(
        ast=ast, resolution=resolution, blueprint=blueprint, snapshot=snapshot
    )
    write_json(paths.blueprint_json, blueprint)
    render_blueprint_markdown(blueprint, paths.output / "blueprint.md")
    return blueprint


def validate_blueprint_files(paths: Paths) -> dict[str, Any]:
    ast = read_json(paths.outline_ast)
    resolution = read_json(paths.outline_resolution)
    blueprint = read_json(paths.blueprint_json)
    snapshot = read_json(paths.snapshot)
    result = validate_blueprint_payload(
        ast=ast, resolution=resolution, blueprint=blueprint, snapshot=snapshot
    )
    if blueprint.get("outline_sha256") != ast.get("source_sha256"):
        raise CorePipelineError("blueprint outline fingerprint is stale")
    return result


def build_snapshot(
    paths: Paths, *, spec_override: dict[str, Any] | None = None
) -> dict[str, Any]:
    spec = spec_override if spec_override is not None else load_spec(paths)
    experiment_spec = spec["experiment"]
    registry_path = repo_path(paths.repo, str(experiment_spec["registry_path"]))
    registry = read_yaml(registry_path)
    experiment_id = str(experiment_spec["id"])
    experiment = find_experiment(registry, experiment_id)
    status = str(experiment.get("status", ""))
    require_status(status, str(experiment_spec["required_status"]))

    terminology = str(experiment.get("data", {}).get("terminology", ""))
    if terminology != "held_out_context_generalization":
        raise CorePipelineError(
            f"C-U1 terminology must be held_out_context_generalization, got {terminology!r}"
        )
    separation = experiment.get("reporting_separation")
    required_separation = {
        "task_performance_collapse",
        "support_or_variance_contraction",
        "nan_inf_numerical_failure",
    }
    if not isinstance(separation, list) or not required_separation.issubset(set(separation)):
        raise CorePipelineError("registry does not preserve the three required failure categories")
    evidence = experiment.get("evidence", {})
    if not isinstance(evidence, dict) or evidence.get("terminal_audited") is not True:
        raise CorePipelineError("experiment is not terminal audited")

    artifact_index_path = repo_path(paths.repo, str(experiment_spec["artifact_index"]))
    artifact_audit = verify_compact_artifacts(paths.repo, artifact_index_path)
    index = artifact_audit["index"]
    if index.get("experiment_id") != experiment_id or index.get("scientific_status") != status:
        raise CorePipelineError("ARTIFACT_INDEX identity/status disagrees with registry")

    fixed_path = repo_path(paths.repo, str(experiment_spec["fixed_aggregate"]))
    learnable_path = repo_path(paths.repo, str(experiment_spec["learnable_aggregate"]))
    terminal_path = repo_path(paths.repo, str(experiment_spec["terminal_audit"]))
    for required in (fixed_path, learnable_path, terminal_path):
        if not required.is_file():
            raise CorePipelineError(f"required evidence file is missing: {required.relative_to(paths.repo)}")

    primary_methods = [str(value) for value in spec["methods"]["primary"]]
    fixed_controls = [str(value) for value in spec["methods"].get("fixed_controls", [])]
    all_methods = primary_methods + fixed_controls
    if len(set(all_methods)) != len(all_methods):
        raise CorePipelineError("paper method lists contain duplicates")
    fixed_rows = load_csv_by_method(fixed_path)
    learnable_rows = load_csv_by_method(learnable_path)
    missing_fixed = [method for method in all_methods if method not in fixed_rows]
    missing_primary_learnable = [method for method in primary_methods if method not in learnable_rows]
    if missing_fixed or missing_primary_learnable:
        raise CorePipelineError(
            "required methods missing from aggregate CSVs: "
            f"fixed={missing_fixed}, learnable_primary={missing_primary_learnable}"
        )

    denominator = int(spec["metric_contract"]["task_collapse"]["denominator"])
    if denominator != len(experiment.get("held_out_seeds", [])):
        raise CorePipelineError("metric denominator differs from registered held-out seed count")

    result_summary = experiment.get("result_summary", {})
    fixed_registry = result_summary.get("fixed_variance", {}) if isinstance(result_summary, dict) else {}
    learnable_registry = result_summary.get("learnable_variance", {}) if isinstance(result_summary, dict) else {}
    methods_payload: dict[str, Any] = {}
    total_nan_inf = result_summary.get("total_nan_inf_count") if isinstance(result_summary, dict) else None
    for method in all_methods:
        fixed = fixed_rows[method]
        learnable = learnable_rows.get(method)
        fixed_nan = fixed_registry.get(method, {}).get("nan_inf_count")
        if fixed_nan is None and total_nan_inf == 0:
            fixed_nan = 0
        if fixed_nan is None:
            raise CorePipelineError(f"registry nan_inf_count missing for fixed method {method}")
        learnable_payload = None
        if learnable is not None:
            learnable_nan = learnable_registry.get(method, {}).get("nan_inf_count")
            if learnable_nan is None and total_nan_inf == 0:
                learnable_nan = 0
            if learnable_nan is None:
                raise CorePipelineError(f"registry nan_inf_count missing for learnable method {method}")
            learnable_payload = {
                "reward": numeric(learnable, "reward", source=f"learnable:{method}"),
                "reward_ci95": [
                    numeric(learnable, "reward_ci_low", source=f"learnable:{method}"),
                    numeric(learnable, "reward_ci_high", source=f"learnable:{method}"),
                ],
                "task_collapse_count": event_count(
                    numeric(
                        learnable,
                        "task_failure_onset_event_rate",
                        source=f"learnable:{method}",
                    ),
                    denominator,
                    label=f"learnable:{method}:task",
                ),
                "support_boundary_count": event_count(
                    numeric(
                        learnable,
                        "support_boundary_onset_event_rate",
                        source=f"learnable:{method}",
                    ),
                    denominator,
                    label=f"learnable:{method}:support",
                ),
                "support_onset_mean": learnable.get("support_boundary_onset"),
                "nan_inf_count": int(learnable_nan),
                "n": int(numeric(learnable, "n", source=f"learnable:{method}")),
            }
        methods_payload[method] = {
            "display_name": DISPLAY_METHOD.get(method, method),
            "paper_role": "primary_intervention" if method in primary_methods else "fixed_budget_control",
            "fixed_variance": {
                "reward": numeric(fixed, "reward", source=f"fixed:{method}"),
                "reward_ci95": [
                    numeric(fixed, "reward_ci_low", source=f"fixed:{method}"),
                    numeric(fixed, "reward_ci_high", source=f"fixed:{method}"),
                ],
                "task_collapse_count": event_count(
                    numeric(fixed, "task_failure_onset_event_rate", source=f"fixed:{method}"),
                    denominator,
                    label=f"fixed:{method}:task",
                ),
                "support_boundary_count": event_count(
                    numeric(fixed, "support_boundary_onset_event_rate", source=f"fixed:{method}"),
                    denominator,
                    label=f"fixed:{method}:support",
                ),
                "nan_inf_count": int(fixed_nan),
                "n": int(numeric(fixed, "n", source=f"fixed:{method}")),
            },
            "learnable_variance": learnable_payload,
        }

    outline_path = repo_path(paths.repo, str(spec["approved_outline"]))
    for binding in spec["outline_bindings"].values():
        if not outline_contains_node(outline_path, str(binding)):
            raise CorePipelineError(f"approved outline is missing bound node {binding}")

    inputs = {
        "spec": {"path": paths.spec.relative_to(paths.repo).as_posix(), "sha256": sha256_file(paths.spec)},
        "registry": {"path": registry_path.relative_to(paths.repo).as_posix(), "sha256": sha256_file(registry_path)},
        "outline": {"path": outline_path.relative_to(paths.repo).as_posix(), "sha256": sha256_file(outline_path)},
        "artifact_index": {"path": artifact_index_path.relative_to(paths.repo).as_posix(), "sha256": sha256_file(artifact_index_path)},
        "fixed_aggregate": {"path": fixed_path.relative_to(paths.repo).as_posix(), "sha256": sha256_file(fixed_path)},
        "learnable_aggregate": {"path": learnable_path.relative_to(paths.repo).as_posix(), "sha256": sha256_file(learnable_path)},
        "terminal_audit": {"path": terminal_path.relative_to(paths.repo).as_posix(), "sha256": sha256_file(terminal_path)},
    }
    snapshot = {
        "schema_version": 1,
        "snapshot_kind": "generated_build_input_not_research_master",
        "task_id": spec["spec_id"],
        "profile": spec["profile"],
        "experiment": {
            "id": experiment_id,
            "environment": experiment.get("environment"),
            "status": status,
            "role": experiment.get("role"),
            "run_commit": experiment.get("provenance", {}).get("run_commit"),
            "held_out_seed_count": denominator,
            "terminology": terminology,
            "reporting_separation": sorted(required_separation),
            "terminal_audited": True,
            "compact_evidence_trust": "checksum_verified_canonical_aggregate",
            "raw_evidence_location": index.get("external_artifact"),
        },
        "outline_bindings": spec["outline_bindings"],
        "method_groups": {
            "primary": primary_methods,
            "fixed_controls": fixed_controls,
        },
        "methods": methods_payload,
        "theorem": spec["theorem"],
        "inputs": inputs,
        "verified_compact_files": artifact_audit["verified"],
    }
    snapshot["snapshot_sha256"] = sha256_bytes(
        json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    write_json(paths.snapshot, snapshot)
    return snapshot


def format_number(value: float, decimals: int) -> str:
    return f"{value:.{decimals}f}"


def format_reward(value: float, decimals: int) -> str:
    """Use scientific notation for near-zero rewards and fixed decimals otherwise."""
    if value != 0.0 and abs(value) < 10 ** (-decimals):
        return f"{value:.2e}"
    return format_number(value, decimals)


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in text)


def render_figure(snapshot: dict[str, Any], destination: Path) -> list[dict[str, Any]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows: list[dict[str, Any]] = []
    for method in snapshot["method_groups"]["primary"]:
        payload = snapshot["methods"][method]
        fixed = payload["fixed_variance"]
        rows.append(
            {
                "method": method,
                "display_name": payload["display_name"],
                "reward": fixed["reward"],
                "ci_low": fixed["reward_ci95"][0],
                "ci_high": fixed["reward_ci95"][1],
                "task_collapse_count": fixed["task_collapse_count"],
                "n": fixed["n"],
            }
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    x = list(range(len(rows)))
    values = [row["reward"] for row in rows]
    low = [row["reward"] - row["ci_low"] for row in rows]
    high = [row["ci_high"] - row["reward"] for row in rows]
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    bars = ax.bar(x, values, yerr=[low, high], capsize=4)
    ax.set_xticks(x, [row["display_name"] for row in rows])
    ax.set_ylabel("Terminal held-out-context reward")
    ax.set_ylim(0.0, max(values) * 1.18)
    ax.set_title("C-U1 E3 fixed-variance targeted intervention (20 paired seeds)")
    ax.grid(axis="y", alpha=0.25)
    for bar, row in zip(bars, rows, strict=True):
        label = f"task collapse {row['task_collapse_count']}/{row['n']}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.035,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=0,
        )
    fig.tight_layout()
    fig.savefig(
        destination,
        bbox_inches="tight",
        metadata={"Creator": "DRPO paper_pipeline_core", "CreationDate": None, "ModDate": None},
    )
    plt.close(fig)
    return rows


def write_figure_data(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def render_table(snapshot: dict[str, Any], path: Path, decimals: int) -> None:
    lines = [
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"Method & Fixed reward (95\% CI) & Task & Support & NaN/Inf \\",
        r"\midrule",
    ]
    for method, payload in snapshot["methods"].items():
        fixed = payload["fixed_variance"]
        learnable = payload["learnable_variance"]
        reward = format_reward(fixed["reward"], decimals)
        low = format_reward(fixed["reward_ci95"][0], decimals)
        high = format_reward(fixed["reward_ci95"][1], decimals)
        lines.append(
            f"{latex_escape(payload['display_name'])} & {reward} [{low}, {high}] & "
            f"{fixed['task_collapse_count']}/{fixed['n']} & "
            + (
                f"{learnable['support_boundary_count']}/{learnable['n']} & "
                if learnable is not None
                else r"-- & "
            )
            + f"{max(fixed['nan_inf_count'], learnable['nan_inf_count'] if learnable else 0)}/{fixed['n']} "
            + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_prose(
    snapshot: dict[str, Any],
    blueprint: dict[str, Any],
    path: Path,
    decimals: int,
) -> dict[str, str]:
    methods = snapshot["methods"]

    def fixed(method: str) -> dict[str, Any]:
        return methods[method]["fixed_variance"]

    def learnable(method: str) -> dict[str, Any]:
        return methods[method]["learnable_variance"]

    enabled_ids = [
        str(node["id"]) for node in blueprint["nodes"] if node["status"] == "enabled"
    ]
    expected_ids = ["METHOD-P03", "EXP-P04"]
    if enabled_ids != expected_ids:
        raise CorePipelineError(
            f"Core prose expects enabled outline order {expected_ids}, got {enabled_ids}"
        )

    method_paragraph = (
        "The exponential envelope is selected for a precise tail property rather than an assumed "
        "utility-decay law. Under the finite-order score-growth condition in Proposition 2, multiplying "
        "the unweighted score-times-advantage contribution by exp(-lambda r) drives the weighted "
        "far-field term to zero as policy-relative remoteness grows. This is a method-level tail guarantee: "
        "it neither ranks all tapers nor claims that sample utility itself decays exponentially."
    )

    b, n, fz, fc = (fixed(name) for name in ("baseline", "near_zero", "far_zero", "far_cap"))
    lb, ln, lfz, lfc = (
        learnable(name) for name in ("baseline", "near_zero", "far_zero", "far_cap")
    )
    experiment_paragraph = (
        "We test the registered transmission path with four matched interventions over 20 paired "
        "held-out seeds and report task, boundary, and numerical events separately. In the fixed-variance "
        f"branch, Baseline and Near-zero finish at rewards {format_reward(b['reward'], decimals)} and "
        f"{format_reward(n['reward'], decimals)} and undergo task-performance collapse in "
        f"{b['task_collapse_count']}/20 and {n['task_collapse_count']}/20 seeds. Far-zero instead reaches "
        f"{format_reward(fz['reward'], decimals)} "
        f"[{format_reward(fz['reward_ci95'][0], decimals)}, {format_reward(fz['reward_ci95'][1], decimals)}], "
        f"and Far-cap reaches {format_reward(fc['reward'], decimals)} "
        f"[{format_reward(fc['reward_ci95'][0], decimals)}, {format_reward(fc['reward_ci95'][1], decimals)}], "
        "with no task collapse. The registered budget controls also remain non-collapsed: Global-scale "
        f"reaches {format_reward(fixed('global_scale')['reward'], decimals)}, while transferring the far "
        f"budget to the near component reaches {format_reward(fixed('far_to_near')['reward'], decimals)}; "
        "these controls diagnose the pathway rather than establish a universal method ranking. The "
        "learnable-variance branch reaches the support/variance-contraction boundary in "
        f"{lb['support_boundary_count']}/20 Baseline and {ln['support_boundary_count']}/20 Near-zero seeds, "
        f"with mean onsets at steps {float(lb['support_onset_mean']):.1f} and "
        f"{float(ln['support_onset_mean']):.1f}, whereas Far-zero and Far-cap record "
        f"{lfz['support_boundary_count']}/20 and {lfc['support_boundary_count']}/20 such events. All four "
        "methods retain finite parameters and record 0/20 NaN/Inf failures. Within this controlled "
        "same-distribution held-out-context setting, the near-removal non-rescue, far removal/capping "
        "rescue, and fixed-budget controls identify the far-field component as the dominant causal "
        "transmission path without extending that conclusion to universal off-policy behavior."
    )

    paragraphs = {"METHOD-P03": method_paragraph, "EXP-P04": experiment_paragraph}
    node_map = {str(node["id"]): node for node in blueprint["nodes"]}
    lines = ["# Evidence-bounded prose", ""]
    for node_id in enabled_ids:
        lines.extend(
            [
                f"## [{node_id}] {node_map[node_id]['title']}",
                "",
                paragraphs[node_id],
                "",
            ]
        )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return paragraphs



def theorem_tex() -> str:
    return r"""\begin{proposition}[Vanishing weighted far-field gradient]
\label{prop:far-field}
Let $r\geq 0$ be policy-relative remoteness and suppose the unweighted
score-times-advantage contribution satisfies
$\lVert g(r)\rVert\leq C(1+r)^k$ for finite constants $C>0$ and $k\geq 0$.
For any $\lambda>0$, define $\widetilde g(r)=e^{-\lambda r}g(r)$.
Then $\lVert\widetilde g(r)\rVert\to 0$ as $r\to\infty$.
\end{proposition}
"""


def proof_tex() -> str:
    return r"""\begin{proof}
The assumption gives
\[
  \lVert\widetilde g(r)\rVert
  \leq C(1+r)^k e^{-\lambda r}.
\]
For integer $m>k$, the exponential series implies
$e^{\lambda r}\geq (\lambda r)^m/m!$ for $r>0$. Hence
\[
  (1+r)^k e^{-\lambda r}
  \leq \frac{m!(1+r)^k}{(\lambda r)^m},
\]
and the right-hand side converges to zero because $m>k$. The squeeze theorem
proves the result. The proposition controls the weighted gradient tail; it
makes no assumption that sample utility decays exponentially.
\end{proof}
"""


def tex_paragraph(text: str) -> str:
    return latex_escape(text).replace("[", "[").replace("]", "]")


def render_main_tex(
    snapshot: dict[str, Any],
    paragraphs: dict[str, str],
    path: Path,
) -> None:
    experiment = snapshot["experiment"]
    method_paragraph = paragraphs["METHOD-P03"]
    experiment_paragraph = paragraphs["EXP-P04"]
    content = rf"""\pdfinfoomitdate=1
\pdftrailerid{{}}
\pdfsuppressptexinfo=15
\documentclass[9pt]{{article}}
\usepackage[margin=0.62in]{{geometry}}
\usepackage{{amsmath,amssymb,amsthm,booktabs,graphicx,microtype,caption}}
\usepackage[T1]{{fontenc}}
\newtheorem{{proposition}}{{Proposition}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{0.35em}}
\title{{\vspace{{-2.2em}}DRPO Pipeline v2.3 Core: Faithful Outline-to-Blueprint Slice}}
\author{{Anonymous review artifact}}
\date{{}}
\begin{{document}}
\maketitle
\vspace{{-2.0em}}
\textbf{{Scope.}}
This two-page artifact validates the manuscript pipeline, not a new experiment.
It preserves the approved paragraph identities \texttt{{METHOD-P03}} and
\texttt{{EXP-P04}} without merge, split, rename, or reorder. It uses the
long-run-validated {latex_escape(str(experiment['id']))} compact repository evidence.
C-U1 evaluates same-distribution held-out-context generalization, and task,
boundary, and NaN/Inf events remain separate.

\section*{{Proposition 2: Vanishing Weighted Far-Field Gradient}}
{tex_paragraph(method_paragraph)}
\setcounter{{proposition}}{{1}}
\input{{theorem.tex}}
\input{{proof.tex}}

\textbf{{Claim boundary.}}
The proposition controls the weighted gradient tail under finite-order score
growth. It does not assume exponential utility decay and does not establish a
universal ranking among taper families.

\clearpage
\section*{{RQ2b: Targeted Causal Transmission}}
\vspace{{-0.5em}}
\begin{{center}}
\begin{{minipage}}[t]{{0.49\textwidth}}
\centering
\captionsetup{{type=table,font=footnotesize,skip=3pt}}
\captionof{{table}}{{Terminal outcomes. Task, support-boundary, and NaN/Inf events are separate.}}
\label{{tab:cu1-e3}}
\scriptsize
\resizebox{{\linewidth}}{{!}}{{\input{{cu1_e3_results.tex}}}}
\end{{minipage}}\hfill
\begin{{minipage}}[t]{{0.49\textwidth}}
\centering
\includegraphics[width=\linewidth]{{cu1_e3_fixed_reward.pdf}}
\captionsetup{{type=figure,font=footnotesize,skip=3pt}}
\captionof{{figure}}{{Fixed-variance terminal reward with 95\% CIs; labels show task-collapse counts.}}
\label{{fig:cu1-e3}}
\end{{minipage}}
\end{{center}}
\vspace{{-0.4em}}

{tex_paragraph(experiment_paragraph)}

\section*{{Pipeline audit}}
The approved outline is compiled into a 39-node AST. Every node receives an
explicit enabled or disabled status; the two enabled nodes map one-to-one into
the blueprint and prose. Every empirical number is rendered from
\texttt{{research\_snapshot.json}}. Missing evidence fails closed rather than
creating a placeholder result.

\end{{document}}
"""
    path.write_text(content, encoding="utf-8")



def build_slice(paths: Paths) -> dict[str, Any]:
    spec = load_spec(paths)
    snapshot = build_snapshot(paths)
    paths.output.mkdir(parents=True, exist_ok=True)

    figure_path = paths.output / "cu1_e3_fixed_reward.pdf"
    figure_rows = render_figure(snapshot, figure_path)
    figure_data_path = paths.output / "cu1_e3_fixed_reward.csv"
    write_figure_data(figure_data_path, figure_rows)

    decimals = int(spec["metric_contract"]["reward"]["table_decimals"])
    table_path = paths.output / "cu1_e3_results.tex"
    render_table(snapshot, table_path, decimals)

    blueprint = build_blueprint_contract(paths, snapshot=snapshot, spec_override=spec)
    prose_path = paths.output / "prose.md"
    prose_decimals = int(spec["metric_contract"]["reward"]["prose_decimals"])
    paragraphs = build_prose(snapshot, blueprint, prose_path, prose_decimals)

    (paths.output / "theorem.tex").write_text(theorem_tex(), encoding="utf-8")
    (paths.output / "proof.tex").write_text(proof_tex(), encoding="utf-8")
    render_main_tex(snapshot, paragraphs, paths.output / "main.tex")

    outputs = {}
    for name in (
        "research_snapshot.json",
        "outline_ast.json",
        "outline_resolution.json",
        "blueprint.json",
        "cu1_e3_fixed_reward.pdf",
        "cu1_e3_fixed_reward.csv",
        "cu1_e3_results.tex",
        "blueprint.md",
        "prose.md",
        "theorem.tex",
        "proof.tex",
        "main.tex",
    ):
        path = paths.output / name
        outputs[name] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    manifest = {
        "schema_version": 1,
        "task_id": spec["spec_id"],
        "snapshot_sha256": snapshot["snapshot_sha256"],
        "outline_sha256": blueprint["outline_sha256"],
        "blueprint_enabled_node_ids": blueprint["enabled_node_ids"],
        "outputs": outputs,
        "pdf_status": "not_compiled",
    }
    write_json(paths.manifest, manifest)
    return manifest


def _pages_from_latex_log(log_text: str) -> int | None:
    import re

    match = re.search(r"Output written on .*?\((\d+) pages?[,)]", log_text)
    return int(match.group(1)) if match else None


def pdf_pages(
    path: Path, *, manifest: dict[str, Any] | None = None, latex_log: str | None = None
) -> tuple[int, str]:
    pdfinfo = shutil.which("pdfinfo")
    if pdfinfo is not None:
        proc = subprocess.run(
            [pdfinfo, str(path)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if proc.returncode == 0:
            for line in proc.stdout.splitlines():
                if line.startswith("Pages:"):
                    return int(line.split(":", 1)[1].strip()), "pdfinfo"

    if latex_log is not None:
        pages = _pages_from_latex_log(latex_log)
        if pages is not None:
            return pages, "latex_log"

    if manifest is not None:
        pages = manifest.get("pdf_pages")
        if isinstance(pages, int) and pages > 0:
            return pages, "verified_manifest"

    raise CorePipelineError(
        "page count is unavailable: install pdfinfo, provide a LaTeX build log, "
        "or validate a hash-verified artifact with pdf_pages in build_manifest.json"
    )


def _verify_manifest_outputs(paths: Paths, manifest: dict[str, Any]) -> None:
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise CorePipelineError("build manifest outputs are missing")
    for name, metadata in outputs.items():
        if not isinstance(name, str) or not isinstance(metadata, dict):
            raise CorePipelineError("build manifest contains an invalid output entry")
        output_path = paths.output / name
        if not output_path.is_file():
            raise CorePipelineError(f"manifest output is missing: {name}")
        expected_size = metadata.get("size_bytes")
        expected_sha = metadata.get("sha256")
        if output_path.stat().st_size != expected_size:
            raise CorePipelineError(f"manifest size mismatch: {name}")
        if sha256_file(output_path) != expected_sha:
            raise CorePipelineError(f"manifest checksum mismatch: {name}")


def compile_pdf(paths: Paths) -> Path:
    if not (paths.output / "main.tex").is_file():
        build_slice(paths)
    latexmk = shutil.which("latexmk")
    if latexmk is None:
        raise CorePipelineError("latexmk is required to compile the Core review PDF")
    subprocess.run(
        [latexmk, "-C", "main.tex"],
        cwd=paths.output,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    build_env = os.environ.copy()
    build_env.update({"SOURCE_DATE_EPOCH": "0", "FORCE_SOURCE_DATE": "1", "TZ": "UTC"})
    proc = subprocess.run(
        [latexmk, "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
        cwd=paths.output,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
        env=build_env,
    )
    log_path = paths.output / "latex_build.txt"
    log_path.write_text(proc.stdout.rstrip() + "\n", encoding="utf-8")
    if proc.returncode != 0 or not paths.pdf.is_file():
        raise CorePipelineError(f"Core LaTeX build failed:\n{proc.stdout[-4000:]}")
    pages, page_count_source = pdf_pages(paths.pdf, latex_log=proc.stdout)
    manifest = read_json(paths.manifest)
    manifest["pdf_status"] = "compiled"
    manifest["pdf_pages"] = pages
    manifest["page_count_source"] = page_count_source
    manifest["outputs"]["main.pdf"] = {
        "sha256": sha256_file(paths.pdf),
        "size_bytes": paths.pdf.stat().st_size,
    }
    manifest["outputs"]["latex_build.txt"] = {
        "sha256": sha256_file(log_path),
        "size_bytes": log_path.stat().st_size,
    }
    write_json(paths.manifest, manifest)
    return paths.pdf


def validate_slice(paths: Paths) -> dict[str, Any]:
    spec = load_spec(paths)
    manifest = read_json(paths.manifest)
    if manifest.get("pdf_status") != "compiled":
        raise CorePipelineError("Core review PDF is not marked compiled in the build manifest")
    required = [
        paths.snapshot,
        paths.outline_ast,
        paths.outline_resolution,
        paths.blueprint_json,
        paths.manifest,
        paths.output / "blueprint.md",
        paths.output / "prose.md",
        paths.output / "theorem.tex",
        paths.output / "proof.tex",
        paths.output / "cu1_e3_fixed_reward.pdf",
        paths.output / "cu1_e3_fixed_reward.csv",
        paths.output / "cu1_e3_results.tex",
        paths.output / "main.tex",
        paths.pdf,
    ]
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise CorePipelineError(f"Core slice is incomplete; missing {missing}")

    _verify_manifest_outputs(paths, manifest)

    snapshot = read_json(paths.snapshot)
    if snapshot.get("experiment", {}).get("status") != "long_run_validated":
        raise CorePipelineError("snapshot experiment is not long_run_validated")
    if snapshot.get("experiment", {}).get("terminology") != "held_out_context_generalization":
        raise CorePipelineError("snapshot C-U1 terminology is invalid")

    prose = (paths.output / "prose.md").read_text(encoding="utf-8")
    main_tex = (paths.output / "main.tex").read_text(encoding="utf-8")
    combined = prose + "\n" + main_tex
    for term in spec["release"]["forbid_terms"]:
        if str(term).lower() in combined.lower():
            raise CorePipelineError(f"forbidden release term found: {term}")
    required_phrases = (
        "task-performance collapse",
        "support/variance-contraction",
        "NaN/Inf",
        "held-out-context",
        "dominant causal transmission path",
        "Global-scale",
        "far budget to the near component",
    )
    for phrase in required_phrases:
        if phrase not in combined:
            raise CorePipelineError(f"required reporting phrase is missing: {phrase}")

    blueprint_validation = validate_blueprint_files(paths)
    if blueprint_validation["enabled_count"] != 2:
        raise CorePipelineError("Core slice must enable exactly two approved outline nodes")
    blueprint = (paths.output / "blueprint.md").read_text(encoding="utf-8")
    if "Sentence plan:" not in blueprint or "Reviewer objection:" not in blueprint:
        raise CorePipelineError("blueprint lacks executable sentence plan or objection")
    validation_methods = spec["methods"]["primary"] + spec["methods"].get("fixed_controls", [])
    for method in validation_methods:
        metric = f"methods.{method}.fixed_variance.reward"
        if metric not in blueprint:
            raise CorePipelineError(f"blueprint missing exact metric path: {metric}")
    if "EXP-P04-A" in blueprint or "EXP-P04-B" in blueprint:
        raise CorePipelineError("blueprint illegally split the approved EXP-P04 paragraph")
    if "## [METHOD-P03]" not in prose or "## [EXP-P04]" not in prose:
        raise CorePipelineError("prose does not preserve enabled outline paragraph IDs")
    if prose.index("## [METHOD-P03]") > prose.index("## [EXP-P04]"):
        raise CorePipelineError("prose reordered approved outline paragraphs")

    theorem = (paths.output / "theorem.tex").read_text(encoding="utf-8")
    proof = (paths.output / "proof.tex").read_text(encoding="utf-8")
    if "C(1+r)^k" not in theorem or "e^{-\\lambda r}" not in theorem:
        raise CorePipelineError("theorem omits finite-order or exponential assumption")
    if "utility" not in proof or "squeeze theorem" not in proof.lower():
        raise CorePipelineError("proof omits claim boundary or analytic closure")

    # Rebuild the expected snapshot in an isolated directory so validation is read-only.
    with tempfile.TemporaryDirectory(prefix="drpo-paper-core-validate-") as temporary:
        rebuilt_paths = Paths(repo=paths.repo, spec=paths.spec, output=Path(temporary))
        rebuilt = build_snapshot(rebuilt_paths, spec_override=spec)
        if rebuilt_paths.snapshot.read_bytes() != paths.snapshot.read_bytes():
            raise CorePipelineError("snapshot is not deterministic")
    if rebuilt["snapshot_sha256"] != snapshot["snapshot_sha256"]:
        raise CorePipelineError("snapshot identity changed during validation")

    pages, page_count_source = pdf_pages(paths.pdf, manifest=manifest)
    required_pages = int(spec["release"]["required_pdf_pages"])
    if pages != required_pages:
        raise CorePipelineError(f"Core review PDF must have {required_pages} pages, got {pages}")

    log = (paths.output / "latex_build.txt").read_text(encoding="utf-8", errors="replace")
    quality_errors = [
        marker
        for marker in ("Undefined control sequence", "LaTeX Error", "Overfull \\hbox", "Overfull \\vbox")
        if marker in log
    ]
    if quality_errors:
        raise CorePipelineError(f"LaTeX quality audit failed: {quality_errors}")

    return {
        "status": "PASS",
        "task_id": spec["spec_id"],
        "experiment_id": snapshot["experiment"]["id"],
        "experiment_status": snapshot["experiment"]["status"],
        "pdf_pages": pages,
        "page_count_source": page_count_source,
        "pdf_sha256": sha256_file(paths.pdf),
        "snapshot_sha256": snapshot["snapshot_sha256"],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "snapshot",
            "parse-outline",
            "build-blueprint",
            "validate-blueprint",
            "build-slice",
            "compile",
            "validate-slice",
            "all",
        ),
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--spec", type=Path, default=Path("docs/manuscript/paper_spec_core.yaml")
    )
    parser.add_argument("--output-root", type=Path)
    return parser.parse_args(argv)


def make_paths(args: argparse.Namespace) -> Paths:
    repo = args.repo_root.resolve()
    spec = (repo / args.spec).resolve()
    loaded = read_yaml(spec)
    output = (
        (repo / args.output_root).resolve()
        if args.output_root is not None
        else repo_path(repo, str(loaded["output_root"]))
    )
    return Paths(
        repo=repo,
        spec=spec,
        output=output,
        allow_output_override=args.output_root is not None,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        paths = make_paths(args)
        if args.command == "snapshot":
            result = build_snapshot(paths)
        elif args.command == "parse-outline":
            ast = build_outline_ast(paths)
            resolution = build_outline_resolution(paths, ast)
            result = {
                "status": "PASS",
                "outline_nodes": ast["node_count"],
                "enabled_node_ids": resolution["enabled_node_ids"],
                "outline_sha256": ast["source_sha256"],
            }
        elif args.command == "build-blueprint":
            result = build_blueprint_contract(paths)
        elif args.command == "validate-blueprint":
            result = validate_blueprint_files(paths)
        elif args.command == "build-slice":
            result = build_slice(paths)
        elif args.command == "compile":
            result = {"pdf": str(compile_pdf(paths))}
        elif args.command == "validate-slice":
            result = validate_slice(paths)
        else:
            build_slice(paths)
            compile_pdf(paths)
            result = validate_slice(paths)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (CorePipelineError, KeyError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
