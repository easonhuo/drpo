#!/usr/bin/env python3
"""Validate the DRPO governance rule migration inventory.

The validator is intentionally read-only.  Stage 0 adds a safety net without
moving or weakening any existing governance rule.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml


RULE_ID_RE = re.compile(r"GOV-[A-Z0-9][A-Z0-9-]{2,127}")
SHA_RE = re.compile(r"[0-9a-f]{40}")
ITEM_RE = re.compile(r"^(?P<indent>\s*)(?P<marker>[*-]|\d+\.)\s+(?P<text>.+?)\s*$")
HEADING_RE = re.compile(r"^(?P<marks>#{1,6})\s+(?P<title>.+?)\s*$")
ALLOWED_MIGRATION_STATUSES = {"unchanged", "shadowed", "migrated", "retired"}


class InventoryError(ValueError):
    """Raised when the governance inventory does not match the repository."""


@dataclass(frozen=True)
class MarkdownItem:
    index: int
    text: str
    normalized_sha256: str


def normalize_text(value: str) -> str:
    return " ".join(value.split())


def text_sha256(value: str) -> str:
    return hashlib.sha256(normalize_text(value).encode("utf-8")).hexdigest()


def section_item_digest(items: Iterable[MarkdownItem]) -> str:
    payload = "\n".join(item.normalized_sha256 for item in items)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def extract_top_level_list_items(markdown: str, heading: str) -> list[MarkdownItem]:
    """Extract top-level list items from one exact Markdown section.

    Continuation lines indented beneath a list item are included in that item.
    Nested lists are treated as continuation text rather than independent rules.
    """

    lines = markdown.splitlines()
    start: int | None = None
    level: int | None = None
    for index, line in enumerate(lines):
        match = HEADING_RE.match(line)
        if match and match.group("title") == heading:
            start = index + 1
            level = len(match.group("marks"))
            break
    if start is None or level is None:
        raise InventoryError(f"Missing Markdown heading: {heading!r}")

    section_lines: list[str] = []
    for line in lines[start:]:
        match = HEADING_RE.match(line)
        if match and len(match.group("marks")) <= level:
            break
        section_lines.append(line)

    raw_items: list[list[str]] = []
    current: list[str] | None = None
    current_indent = 0
    for line in section_lines:
        match = ITEM_RE.match(line)
        if match and len(match.group("indent")) == 0:
            if current is not None:
                raw_items.append(current)
            current = [match.group("text")]
            current_indent = 0
            continue
        if current is None:
            continue
        stripped = line.strip()
        if not stripped:
            continue
        nested = ITEM_RE.match(line)
        if nested and len(nested.group("indent")) > current_indent:
            current.append(nested.group("text"))
        elif line.startswith(" ") or line.startswith("\t"):
            current.append(stripped)
        else:
            # A non-indented paragraph terminates the active list item.
            raw_items.append(current)
            current = None
    if current is not None:
        raw_items.append(current)

    items: list[MarkdownItem] = []
    for index, parts in enumerate(raw_items, start=1):
        text = normalize_text(" ".join(parts))
        items.append(MarkdownItem(index=index, text=text, normalized_sha256=text_sha256(text)))
    return items


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise InventoryError(f"Could not read {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise InventoryError(f"{path} must contain one YAML mapping")
    return payload


def require_repo_path(repo_root: Path, value: str, label: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise InventoryError(f"{label} must be a safe repository-relative path: {value!r}")
    resolved = repo_root / path
    if not resolved.exists():
        raise InventoryError(f"{label} does not exist: {value}")
    return resolved


def _validate_rule_common(rule: dict[str, Any], repo_root: Path) -> None:
    rule_id = rule.get("rule_id")
    if not isinstance(rule_id, str) or not RULE_ID_RE.fullmatch(rule_id):
        raise InventoryError(f"Invalid rule_id: {rule_id!r}")

    source = rule.get("source")
    if not isinstance(source, dict):
        raise InventoryError(f"{rule_id}: source must be a mapping")
    excerpt = source.get("excerpt")
    digest = source.get("normalized_text_sha256")
    if not isinstance(excerpt, str) or not excerpt.strip():
        raise InventoryError(f"{rule_id}: source excerpt is required")
    if digest != text_sha256(excerpt):
        raise InventoryError(f"{rule_id}: source excerpt hash mismatch")

    locations = rule.get("current_locations")
    if not isinstance(locations, list) or not locations:
        raise InventoryError(f"{rule_id}: current_locations must be non-empty")
    for location in locations:
        if not isinstance(location, dict) or not isinstance(location.get("path"), str):
            raise InventoryError(f"{rule_id}: malformed current location")
        require_repo_path(repo_root, location["path"], f"{rule_id} current location")

    migration = rule.get("migration")
    if not isinstance(migration, dict):
        raise InventoryError(f"{rule_id}: migration must be a mapping")
    status = migration.get("status")
    if status not in ALLOWED_MIGRATION_STATUSES:
        raise InventoryError(f"{rule_id}: invalid migration status {status!r}")
    if status in {"migrated", "retired"}:
        destinations = migration.get("destination_locations")
        evidence = migration.get("evidence")
        if not isinstance(destinations, list) or not destinations:
            raise InventoryError(f"{rule_id}: migrated/retired rule needs destinations")
        if not isinstance(evidence, list) or not evidence:
            raise InventoryError(f"{rule_id}: migrated/retired rule needs evidence")
        for destination in destinations:
            if not isinstance(destination, dict) or not isinstance(destination.get("path"), str):
                raise InventoryError(f"{rule_id}: malformed destination")
            require_repo_path(repo_root, destination["path"], f"{rule_id} destination")

    machine = rule.get("machine_enforcement", {})
    if not isinstance(machine, dict):
        raise InventoryError(f"{rule_id}: machine_enforcement must be a mapping")
    if machine.get("required") is True:
        implementations = machine.get("implementations")
        tests = machine.get("tests")
        if not isinstance(implementations, list) or not implementations:
            raise InventoryError(f"{rule_id}: machine-enforceable rule lacks implementation")
        if not isinstance(tests, list) or not tests:
            raise InventoryError(f"{rule_id}: machine-enforceable rule lacks tests")
        for path in implementations:
            if not isinstance(path, str):
                raise InventoryError(f"{rule_id}: implementation paths must be strings")
            require_repo_path(repo_root, path, f"{rule_id} implementation")
        for path in tests:
            if not isinstance(path, str):
                raise InventoryError(f"{rule_id}: test paths must be strings")
            require_repo_path(repo_root, path, f"{rule_id} test")


def validate_inventory(repo_root: Path, inventory_path: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    inventory = load_yaml_mapping(inventory_path)
    if inventory.get("schema_version") != 1:
        raise InventoryError("schema_version must be 1")
    base_commit = inventory.get("inventory_base_commit")
    if not isinstance(base_commit, str) or not SHA_RE.fullmatch(base_commit):
        raise InventoryError("inventory_base_commit must be a full lowercase Git SHA")

    tracked_sections = inventory.get("tracked_sections")
    rules = inventory.get("rules")
    if not isinstance(tracked_sections, list) or not tracked_sections:
        raise InventoryError("tracked_sections must be non-empty")
    if not isinstance(rules, list) or not rules:
        raise InventoryError("rules must be non-empty")

    rule_ids: set[str] = set()
    source_keys: set[tuple[str, int]] = set()
    rules_by_source: dict[str, dict[int, dict[str, Any]]] = {}
    for rule in rules:
        if not isinstance(rule, dict):
            raise InventoryError("Every rule must be a mapping")
        _validate_rule_common(rule, repo_root)
        rule_id = rule["rule_id"]
        if rule_id in rule_ids:
            raise InventoryError(f"Duplicate rule_id: {rule_id}")
        rule_ids.add(rule_id)
        source = rule["source"]
        source_id = source.get("source_id")
        item_index = source.get("item_index")
        if not isinstance(source_id, str) or not isinstance(item_index, int) or item_index <= 0:
            raise InventoryError(f"{rule_id}: invalid source_id/item_index")
        key = (source_id, item_index)
        if key in source_keys:
            raise InventoryError(f"Duplicate source coverage: {source_id} item {item_index}")
        source_keys.add(key)
        rules_by_source.setdefault(source_id, {})[item_index] = rule

    section_ids: set[str] = set()
    covered_rules = 0
    for section in tracked_sections:
        if not isinstance(section, dict):
            raise InventoryError("Every tracked section must be a mapping")
        source_id = section.get("source_id")
        path_value = section.get("path")
        heading = section.get("heading")
        if not all(isinstance(value, str) and value for value in (source_id, path_value, heading)):
            raise InventoryError("tracked section requires source_id, path, and heading")
        if source_id in section_ids:
            raise InventoryError(f"Duplicate source_id: {source_id}")
        section_ids.add(source_id)
        path = require_repo_path(repo_root, path_value, f"{source_id} source")
        items = extract_top_level_list_items(path.read_text(encoding="utf-8"), heading)
        if section.get("expected_item_count") != len(items):
            raise InventoryError(
                f"{source_id}: expected {section.get('expected_item_count')} items, "
                f"found {len(items)}"
            )
        if section.get("section_item_digest") != section_item_digest(items):
            raise InventoryError(f"{source_id}: section item digest changed")
        mapped = rules_by_source.get(source_id, {})
        if set(mapped) != set(range(1, len(items) + 1)):
            missing = sorted(set(range(1, len(items) + 1)) - set(mapped))
            extra = sorted(set(mapped) - set(range(1, len(items) + 1)))
            raise InventoryError(
                f"{source_id}: incomplete coverage; missing={missing}, extra={extra}"
            )
        for item in items:
            rule = mapped[item.index]
            source = rule["source"]
            if source["normalized_text_sha256"] != item.normalized_sha256:
                raise InventoryError(
                    f"{rule['rule_id']}: tracked source text changed at "
                    f"{source_id} item {item.index}"
                )
            if normalize_text(source["excerpt"]) != item.text:
                raise InventoryError(f"{rule['rule_id']}: excerpt does not match Markdown source")
            covered_rules += 1

    unknown_sources = sorted(set(rules_by_source) - section_ids)
    if unknown_sources:
        raise InventoryError(f"Rules reference untracked source IDs: {unknown_sources}")

    return {
        "matched": True,
        "schema_version": 1,
        "tracked_sections": len(tracked_sections),
        "covered_rules": covered_rules,
        "machine_enforced_rules": sum(
            1 for rule in rules if rule.get("machine_enforcement", {}).get("required") is True
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--inventory",
        type=Path,
        default=Path("docs/governance_rule_inventory.yaml"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    inventory = args.inventory
    if not inventory.is_absolute():
        inventory = args.repo_root / inventory
    try:
        report = validate_inventory(args.repo_root, inventory)
    except InventoryError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
