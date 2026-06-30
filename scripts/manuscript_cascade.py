#!/usr/bin/env python3
"""Validate top-down manuscript changes: outline -> blueprint -> prose.

The outline is the canonical structural contract.  A reported problem must be
triaged from the outline downward.  The first failing layer determines the
minimum change scope, and every configured downstream layer must be regenerated
or revalidated against its parent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

LAYER_ORDER = ("outline", "blueprint", "prose")
BEGIN_RE = re.compile(r"^<!--\s*MANUSCRIPT:BEGIN\s+([A-Z][A-Z0-9-]*)\s*-->\s*$")
END_RE = re.compile(r"^<!--\s*MANUSCRIPT:END\s+([A-Z][A-Z0-9-]*)\s*-->\s*$")
HEADING_RE = re.compile(r"^##\s+\[([A-Z][A-Z0-9-]*)\]\s+(.+?)\s*$")
PARENT_RE = {
    "blueprint": re.compile(
        r"^Parent-Outline-SHA256:\s*`?([0-9a-f]{64})`?\s*$", re.IGNORECASE
    ),
    "prose": re.compile(
        r"^Parent-Blueprint-SHA256:\s*`?([0-9a-f]{64})`?\s*$", re.IGNORECASE
    ),
}


class ManuscriptCascadeError(RuntimeError):
    """Expected manuscript hierarchy or change-record validation failure."""


@dataclass(frozen=True)
class ParagraphBlock:
    paragraph_id: str
    title: str
    payload: str
    parent_hash: str | None

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SectionConfig:
    section_id: str
    layers: dict[str, str | None]


@dataclass(frozen=True)
class HierarchyConfig:
    manuscript_id: str
    sections: dict[str, SectionConfig]


def _safe_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ManuscriptCascadeError(f"unsafe repository-relative path: {value}")
    return path.as_posix()


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ManuscriptCascadeError(f"missing YAML file: {path}") from exc
    except yaml.YAMLError as exc:
        raise ManuscriptCascadeError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ManuscriptCascadeError(f"YAML root must be a mapping: {path}")
    return data


def load_config(path: Path) -> HierarchyConfig:
    data = _load_yaml(path)
    if data.get("schema_version") != 1:
        raise ManuscriptCascadeError("hierarchy schema_version must be 1")
    manuscript_id = data.get("manuscript_id")
    if not isinstance(manuscript_id, str) or not manuscript_id.strip():
        raise ManuscriptCascadeError("manuscript_id must be a non-empty string")
    raw_sections = data.get("sections")
    if not isinstance(raw_sections, list) or not raw_sections:
        raise ManuscriptCascadeError("sections must be a non-empty list")

    sections: dict[str, SectionConfig] = {}
    for raw in raw_sections:
        if not isinstance(raw, dict):
            raise ManuscriptCascadeError("each section entry must be a mapping")
        section_id = raw.get("id")
        if not isinstance(section_id, str) or not section_id.strip():
            raise ManuscriptCascadeError("section id must be a non-empty string")
        if section_id in sections:
            raise ManuscriptCascadeError(f"duplicate section id: {section_id}")
        raw_layers = raw.get("layers")
        if not isinstance(raw_layers, dict):
            raise ManuscriptCascadeError(f"section {section_id} must define layers")
        unknown = sorted(set(raw_layers) - set(LAYER_ORDER))
        if unknown:
            raise ManuscriptCascadeError(
                f"section {section_id} has unknown layers: {', '.join(unknown)}"
            )
        layers: dict[str, str | None] = {}
        for layer in LAYER_ORDER:
            value = raw_layers.get(layer)
            if value is None:
                layers[layer] = None
            elif isinstance(value, str) and value.strip():
                layers[layer] = _safe_relative_path(value.strip())
            else:
                raise ManuscriptCascadeError(
                    f"section {section_id} layer {layer} must be a path or null"
                )
        if layers["outline"] is None:
            raise ManuscriptCascadeError(f"section {section_id} requires an outline")
        if layers["blueprint"] is None and layers["prose"] is not None:
            raise ManuscriptCascadeError(
                f"section {section_id} cannot configure prose without a blueprint"
            )
        sections[section_id] = SectionConfig(section_id=section_id, layers=layers)
    return HierarchyConfig(manuscript_id=manuscript_id.strip(), sections=sections)


def _normalize_payload(lines: list[str]) -> str:
    normalized = [line.rstrip() for line in lines]
    while normalized and not normalized[0]:
        normalized.pop(0)
    while normalized and not normalized[-1]:
        normalized.pop()
    return "\n".join(normalized) + "\n"


def parse_markdown_blocks(text: str, *, layer: str, source: str) -> list[ParagraphBlock]:
    if layer not in LAYER_ORDER:
        raise ManuscriptCascadeError(f"unknown layer: {layer}")
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    blocks: list[ParagraphBlock] = []
    seen: set[str] = set()
    index = 0
    while index < len(lines):
        begin = BEGIN_RE.match(lines[index])
        if begin is None:
            index += 1
            continue
        paragraph_id = begin.group(1)
        if paragraph_id in seen:
            raise ManuscriptCascadeError(f"duplicate paragraph id {paragraph_id} in {source}")
        start = index + 1
        end = start
        while end < len(lines) and END_RE.match(lines[end]) is None:
            nested = BEGIN_RE.match(lines[end])
            if nested is not None:
                raise ManuscriptCascadeError(
                    f"nested paragraph block {nested.group(1)} inside {paragraph_id} in {source}"
                )
            end += 1
        if end >= len(lines):
            raise ManuscriptCascadeError(f"missing end marker for {paragraph_id} in {source}")
        end_match = END_RE.match(lines[end])
        assert end_match is not None
        if end_match.group(1) != paragraph_id:
            raise ManuscriptCascadeError(
                f"end marker {end_match.group(1)} does not match {paragraph_id} in {source}"
            )
        payload_lines = lines[start:end]
        heading_index = next((i for i, line in enumerate(payload_lines) if line.strip()), None)
        if heading_index is None:
            raise ManuscriptCascadeError(f"empty paragraph block {paragraph_id} in {source}")
        heading = HEADING_RE.match(payload_lines[heading_index])
        if heading is None or heading.group(1) != paragraph_id:
            raise ManuscriptCascadeError(
                f"block {paragraph_id} must begin with '## [{paragraph_id}] <title>' in {source}"
            )
        title = heading.group(2).strip()
        parent_hash: str | None = None
        if layer in PARENT_RE:
            nonempty_after_heading = [
                line.strip() for line in payload_lines[heading_index + 1 :] if line.strip()
            ]
            if not nonempty_after_heading:
                raise ManuscriptCascadeError(
                    f"block {paragraph_id} in {source} is missing its parent hash"
                )
            parent = PARENT_RE[layer].match(nonempty_after_heading[0])
            if parent is None:
                expected = (
                    "Parent-Outline-SHA256" if layer == "blueprint" else "Parent-Blueprint-SHA256"
                )
                raise ManuscriptCascadeError(
                    f"block {paragraph_id} in {source} must place {expected} "
                    "immediately after its heading"
                )
            parent_hash = parent.group(1).lower()
        payload = _normalize_payload(payload_lines)
        blocks.append(
            ParagraphBlock(
                paragraph_id=paragraph_id,
                title=title,
                payload=payload,
                parent_hash=parent_hash,
            )
        )
        seen.add(paragraph_id)
        index = end + 1
    if not blocks:
        raise ManuscriptCascadeError(f"no MANUSCRIPT paragraph blocks found in {source}")
    return blocks


def parse_markdown_file(path: Path, *, layer: str) -> list[ParagraphBlock]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ManuscriptCascadeError(f"missing {layer} artifact: {path}") from exc
    return parse_markdown_blocks(text, layer=layer, source=str(path))


def _blocks_by_id(blocks: list[ParagraphBlock]) -> dict[str, ParagraphBlock]:
    return {block.paragraph_id: block for block in blocks}


def validate_artifacts(config: HierarchyConfig, *, repo_root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"manuscript_id": config.manuscript_id, "sections": {}}
    for section_id, section in config.sections.items():
        parsed: dict[str, list[ParagraphBlock]] = {}
        for layer in LAYER_ORDER:
            relative = section.layers[layer]
            if relative is None:
                continue
            parsed[layer] = parse_markdown_file(repo_root / relative, layer=layer)

        outline = parsed["outline"]
        outline_ids = [block.paragraph_id for block in outline]
        outline_titles = [block.title for block in outline]
        for child_layer, parent_layer in (("blueprint", "outline"), ("prose", "blueprint")):
            if child_layer not in parsed:
                continue
            child = parsed[child_layer]
            parent = parsed[parent_layer]
            child_ids = [block.paragraph_id for block in child]
            child_titles = [block.title for block in child]
            parent_ids = [block.paragraph_id for block in parent]
            parent_titles = [block.title for block in parent]
            if child_ids != parent_ids:
                raise ManuscriptCascadeError(
                    f"section {section_id}: {child_layer} paragraph ids/order do not "
                    f"match {parent_layer}; "
                    f"{parent_layer}={parent_ids}, {child_layer}={child_ids}"
                )
            if child_titles != parent_titles:
                raise ManuscriptCascadeError(
                    f"section {section_id}: {child_layer} titles do not match {parent_layer}; "
                    f"{parent_layer}={parent_titles}, {child_layer}={child_titles}"
                )
            parent_map = _blocks_by_id(parent)
            for block in child:
                expected = parent_map[block.paragraph_id].sha256
                if block.parent_hash != expected:
                    raise ManuscriptCascadeError(
                        f"section {section_id} paragraph {block.paragraph_id}: stale "
                        f"{child_layer} parent hash; "
                        f"expected {expected}, found {block.parent_hash}"
                    )
        result["sections"][section_id] = {
            "paragraph_ids": outline_ids,
            "titles": outline_titles,
            "configured_layers": [layer for layer in LAYER_ORDER if layer in parsed],
            "status": "pass",
        }
    return result


def _load_issue(path: Path) -> dict[str, Any]:
    data = _load_yaml(path)
    if data.get("schema_version") not in {1, 2}:
        raise ManuscriptCascadeError("issue schema_version must be 1 or 2")
    for key in ("issue_id", "section_id", "problem", "paragraph_ids", "checks", "resolution"):
        if key not in data:
            raise ManuscriptCascadeError(f"issue is missing required field: {key}")
    if not isinstance(data["issue_id"], str) or not data["issue_id"].strip():
        raise ManuscriptCascadeError("issue_id must be a non-empty string")
    if not isinstance(data["section_id"], str) or not data["section_id"].strip():
        raise ManuscriptCascadeError("section_id must be a non-empty string")
    if not isinstance(data["problem"], str) or not data["problem"].strip():
        raise ManuscriptCascadeError("problem must be a non-empty string")
    paragraph_ids = data["paragraph_ids"]
    if (
        not isinstance(paragraph_ids, list)
        or not paragraph_ids
        or any(not isinstance(item, str) or not item.strip() for item in paragraph_ids)
    ):
        raise ManuscriptCascadeError("paragraph_ids must be a non-empty list of strings")
    if len(set(paragraph_ids)) != len(paragraph_ids):
        raise ManuscriptCascadeError("paragraph_ids must be unique")
    return data


def validate_issue(
    issue: dict[str, Any], config: HierarchyConfig
) -> tuple[str, list[str], dict[str, Any]]:
    if issue.get("schema_version") != 2:
        raise ManuscriptCascadeError(
            "legacy schema_version 1 issues are historical only; active triage requires schema_version 2"
        )
    change_control = issue.get("change_control")
    if not isinstance(change_control, dict):
        raise ManuscriptCascadeError("schema_version 2 issues require change_control")
    kind = change_control.get("kind")
    if kind not in {
        "alignment_repair",
        "content_revision",
        "structural_revision",
        "infrastructure_migration",
    }:
        raise ManuscriptCascadeError(
            "change_control.kind must be alignment_repair, content_revision, "
            "structural_revision, or infrastructure_migration"
        )
    reported_layer = change_control.get("reported_layer")
    if reported_layer not in LAYER_ORDER:
        raise ManuscriptCascadeError("change_control.reported_layer must be a known layer")
    outline_change_authorized = change_control.get("outline_change_authorized")
    if not isinstance(outline_change_authorized, bool):
        raise ManuscriptCascadeError(
            "change_control.outline_change_authorized must be boolean"
        )
    authorization_evidence = change_control.get("authorization_evidence")
    if not isinstance(authorization_evidence, str) or not authorization_evidence.strip():
        raise ManuscriptCascadeError(
            "change_control.authorization_evidence must be non-empty"
        )

    section_id = issue["section_id"]
    if section_id not in config.sections:
        raise ManuscriptCascadeError(f"issue references unknown section: {section_id}")
    section = config.sections[section_id]
    checks = issue["checks"]
    if not isinstance(checks, list) or len(checks) != len(LAYER_ORDER):
        raise ManuscriptCascadeError(
            "checks must contain outline, blueprint, and prose exactly once"
        )
    check_layers = [row.get("layer") if isinstance(row, dict) else None for row in checks]
    if tuple(check_layers) != LAYER_ORDER:
        raise ManuscriptCascadeError(
            f"checks must be top-down in order {list(LAYER_ORDER)}; found {check_layers}"
        )
    statuses: dict[str, str] = {}
    for row in checks:
        if not isinstance(row, dict):
            raise ManuscriptCascadeError("each check must be a mapping")
        layer = row["layer"]
        status = row.get("status")
        evidence = row.get("evidence")
        if status not in {"pass", "fail", "blocked", "not_present"}:
            raise ManuscriptCascadeError(
                f"invalid status for {layer}: {status}; expected pass/fail/blocked/not_present"
            )
        if not isinstance(evidence, str) or not evidence.strip():
            raise ManuscriptCascadeError(f"check {layer} requires non-empty evidence")
        configured = section.layers[layer] is not None
        if not configured and status != "not_present":
            raise ManuscriptCascadeError(
                f"section {section_id} has no {layer} artifact; status must be not_present"
            )
        if configured and status == "not_present":
            raise ManuscriptCascadeError(
                f"section {section_id} configures {layer}; status cannot be not_present"
            )
        statuses[layer] = status

    root: str | None = None
    blocked_expected = False
    for layer in LAYER_ORDER:
        status = statuses[layer]
        if section.layers[layer] is None:
            continue
        if blocked_expected:
            if status != "blocked":
                raise ManuscriptCascadeError(
                    f"{layer} must be blocked after an upstream failure; found {status}"
                )
            continue
        if status == "blocked":
            raise ManuscriptCascadeError(f"{layer} cannot be blocked before an upstream failure")
        if status == "fail":
            root = layer
            blocked_expected = True
    if root is None:
        raise ManuscriptCascadeError("issue checks contain no failing layer")

    if kind == "alignment_repair":
        if reported_layer == "outline":
            raise ManuscriptCascadeError(
                "alignment_repair must report a downstream layer, not the outline"
            )
        if statuses["outline"] != "pass" or root == "outline":
            raise ManuscriptCascadeError(
                "a downstream alignment mismatch is not evidence that the outline is wrong; "
                "alignment_repair requires outline=pass"
            )
        if outline_change_authorized:
            raise ManuscriptCascadeError(
                "alignment_repair must not authorize an outline change"
            )
    elif root == "outline":
        if not outline_change_authorized:
            raise ManuscriptCascadeError(
                "changing a user-approved outline requires explicit outline-change authorization"
            )
    elif outline_change_authorized:
        raise ManuscriptCascadeError(
            "outline_change_authorized must be false when the outline is verified correct"
        )

    root_index = LAYER_ORDER.index(root)
    required_layers = [
        layer
        for layer in LAYER_ORDER[root_index:]
        if section.layers[layer] is not None
    ]
    resolution = issue["resolution"]
    if not isinstance(resolution, dict):
        raise ManuscriptCascadeError("resolution must be a mapping")
    state = resolution.get("state")
    if state not in {"planned", "completed"}:
        raise ManuscriptCascadeError("resolution.state must be planned or completed")
    declared_required = resolution.get("required_layers")
    if declared_required != required_layers:
        raise ManuscriptCascadeError(
            f"resolution.required_layers must be {required_layers}; found {declared_required}"
        )
    changed_layers = resolution.get("changed_layers")
    if not isinstance(changed_layers, list) or any(
        layer not in LAYER_ORDER for layer in changed_layers
    ):
        raise ManuscriptCascadeError("resolution.changed_layers must be a list of known layers")
    if state == "planned" and changed_layers:
        raise ManuscriptCascadeError("planned issues must keep changed_layers empty")
    if state == "completed" and changed_layers != required_layers:
        raise ManuscriptCascadeError(
            f"completed issues must change exactly the required cascade {required_layers}; "
            f"found {changed_layers}"
        )
    summary = {
        "issue_id": issue["issue_id"],
        "section_id": section_id,
        "root_cause_layer": root,
        "required_layers": required_layers,
        "state": state,
        "paragraph_ids": issue["paragraph_ids"],
        "change_kind": kind,
        "reported_layer": reported_layer,
        "outline_change_authorized": outline_change_authorized,
    }
    return root, required_layers, summary


def _git_changed_paths(repo_root: Path, base: str, head: str) -> set[str]:
    proc = subprocess.run(
        ["git", "-C", str(repo_root), "diff", "--name-only", base, head, "--"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise ManuscriptCascadeError(
            f"git diff failed: {(proc.stderr or proc.stdout).strip()}"
        )
    return {line.strip() for line in proc.stdout.splitlines() if line.strip()}


def _git_file_text(repo_root: Path, revision: str, path: str) -> str | None:
    proc = subprocess.run(
        ["git", "-C", str(repo_root), "show", f"{revision}:{path}"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode == 0:
        return proc.stdout
    missing_markers = ("does not exist", "exists on disk, but not in", "Path '")
    if any(marker in proc.stderr for marker in missing_markers):
        return None
    raise ManuscriptCascadeError(
        f"cannot read {path} at {revision}: {(proc.stderr or proc.stdout).strip()}"
    )


def _changed_paragraph_ids(
    *, repo_root: Path, base: str, head: str, path: str, layer: str
) -> set[str]:
    before_text = _git_file_text(repo_root, base, path)
    after_text = _git_file_text(repo_root, head, path)
    before = (
        parse_markdown_blocks(before_text, layer=layer, source=f"{base}:{path}")
        if before_text is not None
        else []
    )
    after = (
        parse_markdown_blocks(after_text, layer=layer, source=f"{head}:{path}")
        if after_text is not None
        else []
    )
    before_map = _blocks_by_id(before)
    after_map = _blocks_by_id(after)
    changed = {
        paragraph_id
        for paragraph_id in set(before_map) | set(after_map)
        if paragraph_id not in before_map
        or paragraph_id not in after_map
        or before_map[paragraph_id].payload != after_map[paragraph_id].payload
    }
    before_order = [block.paragraph_id for block in before]
    after_order = [block.paragraph_id for block in after]
    if before_order != after_order:
        changed.update(before_order)
        changed.update(after_order)
    return changed


def validate_git_cascade(
    *,
    issue: dict[str, Any],
    config: HierarchyConfig,
    required_layers: list[str],
    repo_root: Path,
    base: str,
    head: str,
) -> dict[str, Any]:
    if issue["resolution"]["state"] != "completed":
        raise ManuscriptCascadeError("git cascade validation requires a completed issue")
    section = config.sections[issue["section_id"]]
    changed = _git_changed_paths(repo_root, base, head)
    required_paths = {section.layers[layer] for layer in required_layers}
    assert None not in required_paths
    missing = sorted(path for path in required_paths if path not in changed)
    if missing:
        raise ManuscriptCascadeError(
            "downstream cascade is incomplete; required artifact paths were not changed: "
            + ", ".join(missing)
        )
    root_index = LAYER_ORDER.index(required_layers[0])
    forbidden_upstream = {
        section.layers[layer]
        for layer in LAYER_ORDER[:root_index]
        if section.layers[layer] is not None
    }
    unexpected = sorted(path for path in forbidden_upstream if path in changed)
    if unexpected:
        raise ManuscriptCascadeError(
            "upstream artifacts marked correct must not be rewritten: " + ", ".join(unexpected)
        )

    requested_ids = set(issue["paragraph_ids"])
    allowed_additional = issue.get("allowed_additional_paragraph_ids", [])
    if not isinstance(allowed_additional, list) or any(
        not isinstance(item, str) or not item.strip() for item in allowed_additional
    ):
        raise ManuscriptCascadeError(
            "allowed_additional_paragraph_ids must be a list of paragraph IDs"
        )
    allowed_ids = requested_ids | set(allowed_additional)
    changed_by_layer: dict[str, list[str]] = {}
    for layer in required_layers:
        path = section.layers[layer]
        assert path is not None
        changed_ids = _changed_paragraph_ids(
            repo_root=repo_root,
            base=base,
            head=head,
            path=path,
            layer=layer,
        )
        missing_ids = sorted(requested_ids - changed_ids)
        if missing_ids:
            raise ManuscriptCascadeError(
                f"layer {layer} did not propagate the reported paragraph changes: "
                + ", ".join(missing_ids)
            )
        undeclared = sorted(changed_ids - allowed_ids)
        if undeclared:
            raise ManuscriptCascadeError(
                f"layer {layer} changed undeclared paragraphs: " + ", ".join(undeclared)
            )
        changed_by_layer[layer] = sorted(changed_ids)
    return {
        "base": base,
        "head": head,
        "required_paths": sorted(required_paths),
        "changed_paths": sorted(changed),
        "changed_paragraph_ids": changed_by_layer,
        "status": "pass",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    artifacts = sub.add_parser("validate-artifacts", help="validate outline/blueprint/prose")
    artifacts.add_argument("--repo-root", type=Path, default=Path("."))
    artifacts.add_argument("--config", type=Path, required=True)
    artifacts.add_argument("--json", action="store_true")

    issue = sub.add_parser("validate-issue", help="validate top-down issue triage")
    issue.add_argument("--config", type=Path, required=True)
    issue.add_argument("--issue", type=Path, required=True)
    issue.add_argument("--json", action="store_true")

    cascade = sub.add_parser(
        "validate-change", help="validate issue, artifact hierarchy, and Git cascade"
    )
    cascade.add_argument("--repo-root", type=Path, default=Path("."))
    cascade.add_argument("--config", type=Path, required=True)
    cascade.add_argument("--issue", type=Path, required=True)
    cascade.add_argument("--base", required=True)
    cascade.add_argument("--head", required=True)
    cascade.add_argument("--json", action="store_true")
    return parser


def _print_result(result: dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print("PASS")
        for key, value in result.items():
            print(f"{key}: {value}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "validate-artifacts":
            result = validate_artifacts(config, repo_root=args.repo_root.resolve())
        elif args.command == "validate-issue":
            issue = _load_issue(args.issue)
            _, _, result = validate_issue(issue, config)
        else:
            issue = _load_issue(args.issue)
            _, required_layers, issue_result = validate_issue(issue, config)
            artifact_result = validate_artifacts(config, repo_root=args.repo_root.resolve())
            git_result = validate_git_cascade(
                issue=issue,
                config=config,
                required_layers=required_layers,
                repo_root=args.repo_root.resolve(),
                base=args.base,
                head=args.head,
            )
            result = {
                "issue": issue_result,
                "artifacts": artifact_result,
                "git_cascade": git_result,
                "status": "pass",
            }
        _print_result(result, as_json=args.json)
        return 0
    except ManuscriptCascadeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
