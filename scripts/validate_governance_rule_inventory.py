#!/usr/bin/env python3
"""Validate the DRPO governance rule inventory and its assurance evidence.

Stage 0 protects rule presence, source fingerprints, and migration bookkeeping.
Stage 0.1 adds one explicit assurance mode per tracked rule.  Machine-enforced
rules must name concrete implementation paths and exact pytest nodes; review and
structural rules must declare the evidence required when they change.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
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
ALLOWED_ASSURANCE_TYPES = {"machine", "review", "structural"}
ALLOWED_COVERAGE_LEVELS = {"direct", "grouped"}
PYTEST_NODE_RE = re.compile(
    r"^(?P<path>[^:]+\.py)::(?P<selectors>[A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)?)(?:\[[^\]]+\])?$"
)


class InventoryError(ValueError):
    """Raised when the governance inventory does not match the repository."""


@dataclass(frozen=True)
class MarkdownItem:
    index: int
    text: str
    normalized_sha256: str


@dataclass(frozen=True)
class PytestEvidence:
    node: str
    path: str
    selectors: tuple[str, ...]


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


def _inventory_rule_map(inventory: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rules = inventory.get("rules")
    if not isinstance(rules, list):
        raise InventoryError("inventory rules must be a list")
    return {rule["rule_id"]: rule for rule in rules if isinstance(rule, dict) and "rule_id" in rule}


def _parse_pytest_node(repo_root: Path, node: str) -> PytestEvidence:
    if not isinstance(node, str):
        raise InventoryError(f"pytest node must be a string: {node!r}")
    match = PYTEST_NODE_RE.fullmatch(node)
    if not match:
        raise InventoryError(f"Invalid exact pytest node: {node!r}")
    path_value = match.group("path")
    path = require_repo_path(repo_root, path_value, f"pytest node {node}")
    if not path.is_file():
        raise InventoryError(f"pytest node path is not a file: {path_value}")
    selectors = tuple(match.group("selectors").split("::"))
    return PytestEvidence(node=node, path=path_value, selectors=selectors)


def _python_node_exists(path: Path, selectors: tuple[str, ...]) -> bool:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as exc:
        raise InventoryError(f"Could not parse pytest source {path}: {exc}") from exc

    if len(selectors) == 1:
        target = selectors[0]
        return any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == target
            for node in tree.body
        )
    class_name, method_name = selectors
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return any(
                isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                and child.name == method_name
                for child in node.body
            )
    return False


def validate_assurance(
    repo_root: Path,
    inventory_path: Path,
    assurance_path: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    inventory = load_yaml_mapping(inventory_path)
    assurance = load_yaml_mapping(assurance_path)
    if assurance.get("schema_version") != 1:
        raise InventoryError("assurance schema_version must be 1")
    if assurance.get("inventory_path") != "docs/governance_rule_inventory.yaml":
        raise InventoryError("assurance inventory_path must point to the canonical inventory")

    inventory_rules = _inventory_rule_map(inventory)
    entries = assurance.get("rules")
    if not isinstance(entries, dict) or not entries:
        raise InventoryError("assurance rules must be a non-empty mapping")
    assurance_ids = set(entries)
    inventory_ids = set(inventory_rules)
    missing = sorted(inventory_ids - assurance_ids)
    extra = sorted(assurance_ids - inventory_ids)
    if missing or extra:
        raise InventoryError(f"assurance coverage mismatch; missing={missing}, extra={extra}")

    unique_nodes: list[str] = []
    seen_nodes: set[str] = set()
    rule_reports: list[dict[str, Any]] = []
    counts = {kind: 0 for kind in ALLOWED_ASSURANCE_TYPES}
    direct_machine_rules = 0
    grouped_machine_rules = 0

    for rule_id in sorted(inventory_rules):
        rule = inventory_rules[rule_id]
        entry = entries[rule_id]
        if not isinstance(entry, dict):
            raise InventoryError(f"{rule_id}: assurance entry must be a mapping")
        assurance_type = entry.get("type")
        if assurance_type not in ALLOWED_ASSURANCE_TYPES:
            raise InventoryError(f"{rule_id}: invalid assurance type {assurance_type!r}")
        counts[assurance_type] += 1
        machine_required = rule.get("machine_enforcement", {}).get("required") is True
        if machine_required != (assurance_type == "machine"):
            raise InventoryError(
                f"{rule_id}: machine_enforcement.required and assurance type disagree"
            )

        report: dict[str, Any] = {"rule_id": rule_id, "type": assurance_type}
        if assurance_type == "machine":
            implementation_paths = entry.get("implementation_paths")
            nodes = entry.get("pytest_nodes")
            coverage_level = entry.get("coverage_level")
            if not isinstance(implementation_paths, list) or not implementation_paths:
                raise InventoryError(f"{rule_id}: machine assurance needs implementation_paths")
            if not isinstance(nodes, list) or not nodes:
                raise InventoryError(f"{rule_id}: machine assurance needs exact pytest_nodes")
            if coverage_level not in ALLOWED_COVERAGE_LEVELS:
                raise InventoryError(f"{rule_id}: invalid coverage_level {coverage_level!r}")
            if coverage_level == "grouped":
                note = entry.get("coverage_note")
                if not isinstance(note, str) or not note.strip():
                    raise InventoryError(f"{rule_id}: grouped coverage needs coverage_note")
                grouped_machine_rules += 1
            else:
                direct_machine_rules += 1

            declared_implementations = set(
                rule.get("machine_enforcement", {}).get("implementations", [])
            )
            for path_value in implementation_paths:
                if not isinstance(path_value, str):
                    raise InventoryError(f"{rule_id}: implementation_paths must be strings")
                require_repo_path(repo_root, path_value, f"{rule_id} assured implementation")
                if path_value not in declared_implementations:
                    raise InventoryError(
                        f"{rule_id}: assured implementation is absent from inventory: {path_value}"
                    )

            declared_test_paths = set(rule.get("machine_enforcement", {}).get("tests", []))
            checked_nodes: list[str] = []
            for node_value in nodes:
                evidence = _parse_pytest_node(repo_root, node_value)
                if evidence.path not in declared_test_paths:
                    raise InventoryError(
                        f"{rule_id}: pytest node file is absent from inventory tests: {evidence.path}"
                    )
                source_path = repo_root / evidence.path
                if not _python_node_exists(source_path, evidence.selectors):
                    raise InventoryError(f"{rule_id}: pytest node does not exist: {node_value}")
                checked_nodes.append(node_value)
                if node_value not in seen_nodes:
                    seen_nodes.add(node_value)
                    unique_nodes.append(node_value)
            report.update(
                {
                    "coverage_level": coverage_level,
                    "implementation_paths": implementation_paths,
                    "pytest_nodes": checked_nodes,
                    "static_node_check": "passed",
                }
            )
        elif assurance_type == "review":
            triggers = entry.get("triggers")
            evidence = entry.get("required_evidence")
            if not isinstance(triggers, list) or not triggers:
                raise InventoryError(f"{rule_id}: review assurance needs triggers")
            if not isinstance(evidence, list) or not evidence:
                raise InventoryError(f"{rule_id}: review assurance needs required_evidence")
            report.update({"triggers": triggers, "required_evidence": evidence})
        else:
            checks = entry.get("checks")
            if not isinstance(checks, list) or not checks:
                raise InventoryError(f"{rule_id}: structural assurance needs checks")
            report["checks"] = checks
        rule_reports.append(report)

    return {
        "matched": True,
        "schema_version": 1,
        "assurance_rules": len(entries),
        "assurance_type_counts": counts,
        "direct_machine_rules": direct_machine_rules,
        "grouped_machine_rules": grouped_machine_rules,
        "unique_pytest_nodes": unique_nodes,
        "rule_reports": rule_reports,
    }


def run_pytest_nodes(
    repo_root: Path,
    nodes: list[str],
    *,
    collect_only: bool,
) -> dict[str, Any]:
    if not nodes:
        raise InventoryError("No pytest nodes were supplied")
    command = [sys.executable, "-m", "pytest"]
    if collect_only:
        command.extend(["--collect-only", "-q"])
    else:
        command.append("-q")
    command.extend(nodes)
    completed = subprocess.run(
        command,
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        mode = "collection" if collect_only else "execution"
        detail = (completed.stdout + "\n" + completed.stderr).strip()
        raise InventoryError(f"pytest {mode} failed with code {completed.returncode}: {detail}")
    return {
        "mode": "collect_only" if collect_only else "run",
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def validate_governance(
    repo_root: Path,
    inventory_path: Path,
    assurance_path: Path,
    *,
    collect_pytest: bool = False,
    run_machine_tests: bool = False,
) -> dict[str, Any]:
    inventory_report = validate_inventory(repo_root, inventory_path)
    assurance_report = validate_assurance(repo_root, inventory_path, assurance_path)
    pytest_report: dict[str, Any] = {"status": "not_requested"}
    nodes = assurance_report["unique_pytest_nodes"]
    if collect_pytest or run_machine_tests:
        collected = run_pytest_nodes(repo_root, nodes, collect_only=True)
        pytest_report = {"status": "collected", "collection": collected}
    if run_machine_tests:
        executed = run_pytest_nodes(repo_root, nodes, collect_only=False)
        pytest_report["status"] = "passed"
        pytest_report["execution"] = executed
    return {
        "matched": True,
        "inventory": inventory_report,
        "assurance": assurance_report,
        "pytest": pytest_report,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--inventory",
        type=Path,
        default=Path("docs/governance_rule_inventory.yaml"),
    )
    parser.add_argument(
        "--assurance",
        type=Path,
        default=Path("docs/governance_rule_assurance.yaml"),
    )
    parser.add_argument(
        "--collect-pytest",
        action="store_true",
        help="Run pytest --collect-only for every unique machine-assurance node.",
    )
    parser.add_argument(
        "--run-machine-tests",
        action="store_true",
        help="Collect and execute every unique machine-assurance node.",
    )
    parser.add_argument("--report-out", type=Path)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print the complete per-rule JSON report instead of the compact summary.",
    )
    return parser.parse_args(argv)


def compact_summary(report: dict[str, Any], report_out: Path | None = None) -> str:
    inventory = report["inventory"]
    assurance = report["assurance"]
    counts = assurance["assurance_type_counts"]
    pytest_report = report["pytest"]
    lines = [
        "Governance validation: PASS",
        (
            f"Rules: {assurance['assurance_rules']} "
            f"(machine={counts['machine']}, review={counts['review']}, "
            f"structural={counts['structural']})"
        ),
        (
            f"Coverage: direct={assurance['direct_machine_rules']}, "
            f"grouped={assurance['grouped_machine_rules']}"
        ),
        (
            f"Pytest nodes: {len(assurance['unique_pytest_nodes'])}; "
            f"status={pytest_report['status']}"
        ),
        (
            f"Inventory: sections={inventory['tracked_sections']}, "
            f"covered_rules={inventory['covered_rules']}"
        ),
    ]
    if report_out is not None:
        lines.append(f"Full report: {report_out}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    inventory = args.inventory
    assurance = args.assurance
    if not inventory.is_absolute():
        inventory = args.repo_root / inventory
    if not assurance.is_absolute():
        assurance = args.repo_root / assurance
    try:
        report = validate_governance(
            args.repo_root,
            inventory,
            assurance,
            collect_pytest=args.collect_pytest,
            run_machine_tests=args.run_machine_tests,
        )
    except InventoryError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    rendered = json.dumps(report, indent=2, sort_keys=True)
    report_out: Path | None = None
    if args.report_out:
        report_out = args.report_out
        if not report_out.is_absolute():
            report_out = args.repo_root / report_out
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(rendered + "\n", encoding="utf-8")
    if args.verbose:
        print(rendered)
    else:
        print(compact_summary(report, report_out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
