#!/usr/bin/env python3
"""Validate the non-authoritative Stage 4A schema and static inventories.

The validator is deliberately read-only.  It binds the Stage 4A inventory to
exact source objects in ``docs/handoff.md`` and ``experiments/registry.yaml``
without generating a Stage 4B split candidate or a Stage 4C context pack.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml


POLICY_ID = "GOV-HANDOFF-INDEX-01"
AUTHORITY = "non_authoritative_stage4a_shadow_inventory"
ROOT_REL = Path("docs/handoff_shadow/stage4")
SCHEMA_REL = ROOT_REL / "schema/STAGE4A_SCHEMA.yaml"
MODULES_REL = ROOT_REL / "inventory/MODULES.yaml"
HEADINGS_REL = ROOT_REL / "inventory/HEADINGS.yaml"
CLAIMS_REL = ROOT_REL / "inventory/CLAIMS.yaml"
EXPERIMENTS_REL = ROOT_REL / "inventory/EXPERIMENTS.yaml"
HANDOFF_REL = Path("docs/handoff.md")
REGISTRY_REL = Path("experiments/registry.yaml")

NODE_TYPES = {
    "question",
    "hypothesis",
    "claim",
    "assumption",
    "experiment",
    "evidence",
    "method",
    "limitation",
    "alternative",
    "open_issue",
}
RELATION_TYPES = {
    "depends_on",
    "tests",
    "supports",
    "contradicts",
    "supersedes",
    "external_validates",
    "does_not_replace",
    "motivates",
}
TEMPORAL_ROLES = {"current", "historical", "mixed"}
CLAIM_STATUSES = {
    "current_locked",
    "current_supported",
    "current_limited",
    "open",
    "historical_superseded",
}
MODULE_ID_RE = re.compile(r"[a-z][a-z0-9_]{2,63}")
CLAIM_ID_RE = re.compile(r"(?:GOV|SCI|HIST)-[A-Z0-9][A-Z0-9-]{2,95}")
HEADING_ID_RE = re.compile(r"H[0-9]{4}")


class Stage4AError(ValueError):
    """Raised when Stage 4A content is incomplete, ambiguous, or stale."""


@dataclass(frozen=True)
class Heading:
    heading_id: str
    ordinal: int
    level: int
    line: int
    title: str
    slug: str
    occurrence: int


@dataclass(frozen=True)
class Section:
    heading: Heading
    start_line: int
    end_line: int
    text: str


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def load_mapping(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise Stage4AError(f"Could not read {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise Stage4AError(f"{label} must contain one YAML mapping: {path}")
    return payload


def reject_symlink_components(repo_root: Path, relative: Path, label: str) -> None:
    current = repo_root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise Stage4AError(f"{label} may not contain a symlink: {relative.as_posix()}")


def safe_file(repo_root: Path, relative: Path, label: str) -> Path:
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise Stage4AError(f"{label} must be a safe repository-relative path")
    reject_symlink_components(repo_root, relative, label)
    path = (repo_root / relative).resolve()
    try:
        path.relative_to(repo_root)
    except ValueError as exc:
        raise Stage4AError(f"{label} escapes the repository: {relative}") from exc
    if not path.is_file():
        raise Stage4AError(f"{label} is missing: {relative}")
    return path


def require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Stage4AError(f"{label} must be a non-empty string")
    return value.strip()


def require_list(value: Any, label: str, *, non_empty: bool = True) -> list[Any]:
    if not isinstance(value, list) or (non_empty and not value):
        qualifier = "non-empty " if non_empty else ""
        raise Stage4AError(f"{label} must be a {qualifier}list")
    return value


def require_unique_strings(value: Any, label: str, *, non_empty: bool = True) -> list[str]:
    items = require_list(value, label, non_empty=non_empty)
    if not all(isinstance(item, str) and item for item in items):
        raise Stage4AError(f"{label} must contain only non-empty strings")
    if len(set(items)) != len(items):
        raise Stage4AError(f"{label} contains duplicates")
    return list(items)


def check_common_header(payload: dict[str, Any], label: str) -> None:
    if payload.get("schema_version") != 1:
        raise Stage4AError(f"{label} schema_version must be 1")
    if payload.get("policy_id") != POLICY_ID:
        raise Stage4AError(f"{label} policy_id must be {POLICY_ID}")
    if payload.get("authority") != AUTHORITY:
        raise Stage4AError(f"{label} authority boundary is missing")


def slugify(title: str) -> str:
    normalized = unicodedata.normalize("NFKC", title).strip().lower()
    output: list[str] = []
    pending_dash = False
    for char in normalized:
        if char.isalnum():
            if pending_dash and output:
                output.append("-")
            output.append(char)
            pending_dash = False
        else:
            pending_dash = True
    return "".join(output).strip("-") or "heading"


def parse_headings(path: Path) -> tuple[list[Heading], list[str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    headings: list[Heading] = []
    occurrences: dict[tuple[int, str], int] = {}
    fence: str | None = None
    for line_number, line in enumerate(lines, 1):
        stripped = line.lstrip()
        fence_match = re.match(r"(`{3,}|~{3,})", stripped)
        if fence_match:
            marker = fence_match.group(1)[0]
            if fence is None:
                fence = marker
            elif fence == marker:
                fence = None
            continue
        if fence is not None:
            continue
        match = re.match(r"^(#{1,6})[ \t]+(.+?)[ \t]*$", line)
        if not match:
            continue
        level = len(match.group(1))
        title = match.group(2).strip()
        key = (level, title)
        occurrence = occurrences.get(key, 0) + 1
        occurrences[key] = occurrence
        ordinal = len(headings) + 1
        headings.append(
            Heading(
                heading_id=f"H{ordinal:04d}",
                ordinal=ordinal,
                level=level,
                line=line_number,
                title=title,
                slug=slugify(title),
                occurrence=occurrence,
            )
        )
    return headings, lines


def build_sections(headings: list[Heading], lines: list[str]) -> dict[str, Section]:
    sections: dict[str, Section] = {}
    for index, heading in enumerate(headings):
        end_line = len(lines)
        for later in headings[index + 1 :]:
            if later.level <= heading.level:
                end_line = later.line - 1
                break
        text = "\n".join(lines[heading.line - 1 : end_line])
        sections[heading.heading_id] = Section(
            heading=heading,
            start_line=heading.line,
            end_line=end_line,
            text=text,
        )
    return sections


def validate_schema(payload: dict[str, Any]) -> None:
    check_common_header(payload, "Stage 4A schema")
    if payload.get("phase") != "stage_4a_schema_inventory":
        raise Stage4AError("Stage 4A schema phase must be stage_4a_schema_inventory")
    if payload.get("manual_handoff_remains_authoritative") is not True:
        raise Stage4AError("Stage 4A schema must keep the manual handoff authoritative")
    if payload.get("authority_cutover_allowed") is not False:
        raise Stage4AError("Stage 4A schema must forbid authority cutover")
    if set(require_unique_strings(payload.get("node_types"), "node_types")) != NODE_TYPES:
        raise Stage4AError("node_types must equal the frozen Stage 4 minimum set")
    if set(require_unique_strings(payload.get("relation_types"), "relation_types")) != (
        RELATION_TYPES
    ):
        raise Stage4AError("relation_types must equal the frozen Stage 4 minimum set")

    closure = payload.get("dependency_closure")
    if not isinstance(closure, dict):
        raise Stage4AError("dependency_closure must be a mapping")
    expected_closure = {
        "traversal_relation": "depends_on",
        "traversal": "transitive",
        "include_target": True,
        "include_direct_limitations_lineage_and_provenance": True,
        "missing_dependency_policy": "fail_closed",
        "uncertain_dependency_policy": "conservative_overinclude_with_warning",
        "cycle_policy": "reject",
    }
    if closure != expected_closure:
        raise Stage4AError("dependency_closure does not match the frozen conservative rules")

    ambiguity = payload.get("ambiguity_policy")
    expected_ambiguity = {
        "automatic_guessing_allowed": False,
        "unresolved_classification": "reject",
        "resolved_multi_module_requires_rationale": True,
        "unknown_node_or_relation_type": "reject",
        "dangling_reference": "reject",
        "duplicate_identifier": "reject",
    }
    if ambiguity != expected_ambiguity:
        raise Stage4AError("ambiguity_policy must fail closed")

    paths = payload.get("paths")
    if not isinstance(paths, dict):
        raise Stage4AError("paths must be a mapping")
    expected_paths = {
        "root": ROOT_REL.as_posix(),
        "schema": SCHEMA_REL.as_posix(),
        "modules": MODULES_REL.as_posix(),
        "headings": HEADINGS_REL.as_posix(),
        "claims": CLAIMS_REL.as_posix(),
        "experiments": EXPERIMENTS_REL.as_posix(),
    }
    if paths != expected_paths:
        raise Stage4AError("Stage 4A paths do not match the frozen shadow layout")

    forbidden = require_unique_strings(
        payload.get("forbidden_stage4a_outputs"), "forbidden_stage4a_outputs"
    )
    required_forbidden = {
        "CURRENT_CANDIDATE.md",
        "modules/",
        "history/",
        "graph/NODES.yaml",
        "graph/EDGES.yaml",
        "graph/CLAIM_LINEAGE.yaml",
        "context_packs/",
    }
    if set(forbidden) != required_forbidden:
        raise Stage4AError("forbidden_stage4a_outputs must preserve the 4A/4B/4C boundary")


def validate_modules(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    check_common_header(payload, "module inventory")
    modules = require_list(payload.get("modules"), "modules")
    result: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(modules, 1):
        if not isinstance(item, dict):
            raise Stage4AError(f"module {index} must be a mapping")
        module_id = require_string(item.get("module_id"), f"module {index} module_id")
        if not MODULE_ID_RE.fullmatch(module_id):
            raise Stage4AError(f"invalid module_id: {module_id}")
        if module_id in result:
            raise Stage4AError(f"duplicate module_id: {module_id}")
        require_string(item.get("name"), f"{module_id} name")
        require_string(item.get("purpose"), f"{module_id} purpose")
        dependencies = require_unique_strings(
            item.get("default_dependencies", []),
            f"{module_id} default_dependencies",
            non_empty=False,
        )
        result[module_id] = item
        item["_dependencies"] = dependencies
    for module_id, item in result.items():
        unknown = sorted(set(item["_dependencies"]) - set(result))
        if unknown:
            raise Stage4AError(f"{module_id} has unknown module dependencies: {unknown}")
        if module_id in item["_dependencies"]:
            raise Stage4AError(f"{module_id} may not depend on itself")
    _assert_acyclic(
        {module_id: list(item["_dependencies"]) for module_id, item in result.items()},
        "module dependency graph",
    )
    return result


def validate_source_hash(
    payload: dict[str, Any], path: Path, expected_relative: Path, label: str
) -> None:
    source = payload.get("source")
    if not isinstance(source, dict):
        raise Stage4AError(f"{label} source must be a mapping")
    if source.get("path") != expected_relative.as_posix():
        raise Stage4AError(
            f"{label} source path must be {expected_relative.as_posix()}"
        )
    actual = sha256_path(path)
    if source.get("sha256") != actual:
        raise Stage4AError(f"{label} source SHA-256 is stale")


def validate_headings(
    payload: dict[str, Any],
    handoff_path: Path,
    modules: dict[str, dict[str, Any]],
) -> tuple[dict[str, Heading], dict[str, Section]]:
    check_common_header(payload, "heading inventory")
    validate_source_hash(payload, handoff_path, HANDOFF_REL, "heading inventory")
    actual, lines = parse_headings(handoff_path)
    records = require_list(payload.get("headings"), "headings")
    if len(records) != len(actual):
        raise Stage4AError(
            f"heading inventory count mismatch: inventory={len(records)} source={len(actual)}"
        )
    by_id: dict[str, Heading] = {}
    for expected, record in zip(actual, records):
        if not isinstance(record, dict):
            raise Stage4AError(f"{expected.heading_id} record must be a mapping")
        for field in ("heading_id", "ordinal", "level", "line", "title", "slug", "occurrence"):
            if record.get(field) != getattr(expected, field):
                raise Stage4AError(
                    f"{expected.heading_id} {field} mismatch: "
                    f"inventory={record.get(field)!r} source={getattr(expected, field)!r}"
                )
        if not HEADING_ID_RE.fullmatch(expected.heading_id):
            raise Stage4AError(f"invalid heading_id: {expected.heading_id}")
        module_ids = require_unique_strings(
            record.get("module_ids"), f"{expected.heading_id} module_ids"
        )
        unknown = sorted(set(module_ids) - set(modules))
        if unknown:
            raise Stage4AError(f"{expected.heading_id} has unknown modules: {unknown}")
        classification = record.get("classification")
        if classification not in {"resolved_single", "resolved_multi"}:
            raise Stage4AError(
                f"{expected.heading_id} classification is unresolved or invalid: {classification!r}"
            )
        if len(module_ids) == 1 and classification != "resolved_single":
            raise Stage4AError(f"{expected.heading_id} single-module classification is inconsistent")
        if len(module_ids) > 1:
            if classification != "resolved_multi":
                raise Stage4AError(f"{expected.heading_id} multi-module classification is inconsistent")
            require_string(record.get("classification_rationale"), "multi-module rationale")
        temporal_role = record.get("temporal_role")
        if temporal_role not in TEMPORAL_ROLES:
            raise Stage4AError(f"{expected.heading_id} has invalid temporal_role")
        by_id[expected.heading_id] = expected
    return by_id, build_sections(actual, lines)


def _assert_acyclic(edges: dict[str, list[str]], label: str) -> None:
    active: set[str] = set()
    complete: set[str] = set()

    def visit(node: str) -> None:
        if node in active:
            raise Stage4AError(f"{label} contains a cycle at {node}")
        if node in complete:
            return
        active.add(node)
        for child in edges.get(node, []):
            visit(child)
        active.remove(node)
        complete.add(node)

    for node in edges:
        visit(node)


def validate_claims(
    payload: dict[str, Any],
    handoff_path: Path,
    modules: dict[str, dict[str, Any]],
    headings: dict[str, Heading],
    sections: dict[str, Section],
) -> dict[str, dict[str, Any]]:
    check_common_header(payload, "claim inventory")
    validate_source_hash(payload, handoff_path, HANDOFF_REL, "claim inventory")
    claims = require_list(payload.get("claims"), "claims")
    by_id: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(claims, 1):
        if not isinstance(item, dict):
            raise Stage4AError(f"claim {index} must be a mapping")
        claim_id = require_string(item.get("claim_id"), f"claim {index} claim_id")
        if not CLAIM_ID_RE.fullmatch(claim_id):
            raise Stage4AError(f"invalid claim_id: {claim_id}")
        if claim_id in by_id:
            raise Stage4AError(f"duplicate claim_id: {claim_id}")
        if item.get("node_type") not in NODE_TYPES:
            raise Stage4AError(f"{claim_id} has unknown node_type")
        if item.get("node_type") not in {"claim", "hypothesis", "assumption", "limitation", "open_issue"}:
            raise Stage4AError(f"{claim_id} uses an invalid claim-inventory node_type")
        status = item.get("status")
        if status not in CLAIM_STATUSES:
            raise Stage4AError(f"{claim_id} has invalid status: {status!r}")
        require_string(item.get("statement_summary"), f"{claim_id} statement_summary")
        module_ids = require_unique_strings(item.get("module_ids"), f"{claim_id} module_ids")
        unknown_modules = sorted(set(module_ids) - set(modules))
        if unknown_modules:
            raise Stage4AError(f"{claim_id} has unknown modules: {unknown_modules}")
        source = item.get("source_anchor")
        if not isinstance(source, dict):
            raise Stage4AError(f"{claim_id} source_anchor must be a mapping")
        if source.get("file") != HANDOFF_REL.as_posix():
            raise Stage4AError(f"{claim_id} source_anchor must point to docs/handoff.md")
        heading_id = require_string(source.get("heading_id"), f"{claim_id} heading_id")
        if heading_id not in headings:
            raise Stage4AError(f"{claim_id} references unknown heading {heading_id}")
        anchor = require_string(source.get("text"), f"{claim_id} anchor text")
        if source.get("sha256") != sha256_bytes(anchor.encode("utf-8")):
            raise Stage4AError(f"{claim_id} anchor SHA-256 mismatch")
        section = sections[heading_id]
        count = section.text.count(anchor)
        if count != 1:
            raise Stage4AError(
                f"{claim_id} anchor must occur exactly once in {heading_id}; found {count}"
            )
        if source.get("occurrence_in_section") != 1:
            raise Stage4AError(f"{claim_id} occurrence_in_section must be 1")
        lineage = item.get("lineage")
        if not isinstance(lineage, dict):
            raise Stage4AError(f"{claim_id} lineage must be a mapping")
        supersedes = require_unique_strings(
            lineage.get("supersedes", []), f"{claim_id} supersedes", non_empty=False
        )
        superseded_by = require_unique_strings(
            lineage.get("superseded_by", []), f"{claim_id} superseded_by", non_empty=False
        )
        if status == "historical_superseded":
            if len(superseded_by) != 1:
                raise Stage4AError(f"{claim_id} historical claim needs one superseded_by target")
            require_string(item.get("archive_pointer"), f"{claim_id} archive_pointer")
            require_unique_strings(item.get("reopen_conditions"), f"{claim_id} reopen_conditions")
        elif superseded_by:
            raise Stage4AError(f"{claim_id} current/open claim may not be superseded_by")
        by_id[claim_id] = item
        item["_supersedes"] = supersedes
        item["_superseded_by"] = superseded_by

    for claim_id, item in by_id.items():
        for relation in ("_supersedes", "_superseded_by"):
            unknown = sorted(set(item[relation]) - set(by_id))
            if unknown:
                raise Stage4AError(f"{claim_id} has dangling lineage references: {unknown}")
        for old_id in item["_supersedes"]:
            if claim_id not in by_id[old_id]["_superseded_by"]:
                raise Stage4AError(
                    f"lineage is not reciprocal: {claim_id} supersedes {old_id}"
                )
    _assert_acyclic(
        {claim_id: list(item["_supersedes"]) for claim_id, item in by_id.items()},
        "claim lineage",
    )
    return by_id


def registry_experiments(registry: dict[str, Any]) -> list[dict[str, Any]]:
    experiments = registry.get("experiments")
    if not isinstance(experiments, list):
        raise Stage4AError("registry experiments must be a list")
    result: list[dict[str, Any]] = []
    for index, item in enumerate(experiments, 1):
        if not isinstance(item, dict):
            raise Stage4AError(f"registry experiment {index} must be a mapping")
        result.append(item)
    return result


def infer_expected_module(experiment_id: str) -> str:
    if experiment_id.startswith("C-U1-"):
        return "continuous_e1_e4"
    if experiment_id.startswith("D-U1-"):
        return "categorical_e5_e6"
    if experiment_id.startswith("EXT-H-"):
        return "hopper_e7"
    if experiment_id.startswith("EXT-C-"):
        return "countdown_e8"
    raise Stage4AError(f"experiment ID has no frozen semantic-module mapping: {experiment_id}")


def validate_experiments(
    payload: dict[str, Any],
    registry_path: Path,
    modules: dict[str, dict[str, Any]],
    headings: dict[str, Heading],
    claims: dict[str, dict[str, Any]],
) -> int:
    check_common_header(payload, "experiment inventory")
    validate_source_hash(payload, registry_path, REGISTRY_REL, "experiment inventory")
    registry = load_mapping(registry_path, "experiment registry")
    actual = registry_experiments(registry)
    records = require_list(payload.get("experiments"), "experiment inventory experiments")
    if len(records) != len(actual):
        raise Stage4AError(
            f"experiment inventory count mismatch: inventory={len(records)} source={len(actual)}"
        )
    seen: set[str] = set()
    for index, (source, record) in enumerate(zip(actual, records), 1):
        if not isinstance(record, dict):
            raise Stage4AError(f"experiment inventory record {index} must be a mapping")
        experiment_id = require_string(source.get("id"), f"registry experiment {index} id")
        if experiment_id in seen:
            raise Stage4AError(f"registry contains duplicate experiment ID: {experiment_id}")
        seen.add(experiment_id)
        expected_fields = {
            "experiment_id": experiment_id,
            "registry_index": index,
            "environment": source.get("environment"),
            "registry_status": source.get("status"),
            "role": source.get("role"),
        }
        for field, expected in expected_fields.items():
            if record.get(field) != expected:
                raise Stage4AError(
                    f"{experiment_id} {field} mismatch: "
                    f"inventory={record.get(field)!r} registry={expected!r}"
                )
        module_ids = require_unique_strings(
            record.get("module_ids"), f"{experiment_id} module_ids"
        )
        unknown_modules = sorted(set(module_ids) - set(modules))
        if unknown_modules:
            raise Stage4AError(f"{experiment_id} has unknown modules: {unknown_modules}")
        expected_module = infer_expected_module(experiment_id)
        if expected_module not in module_ids:
            raise Stage4AError(
                f"{experiment_id} must be classified under {expected_module}"
            )
        claim_ids = require_unique_strings(record.get("claim_ids"), f"{experiment_id} claim_ids")
        dangling_claims = sorted(set(claim_ids) - set(claims))
        if dangling_claims:
            raise Stage4AError(f"{experiment_id} has dangling claim references: {dangling_claims}")
        heading_ids = require_unique_strings(
            record.get("handoff_heading_ids"), f"{experiment_id} handoff_heading_ids"
        )
        dangling_headings = sorted(set(heading_ids) - set(headings))
        if dangling_headings:
            raise Stage4AError(
                f"{experiment_id} has dangling heading references: {dangling_headings}"
            )
        classification = record.get("classification")
        if classification not in {"resolved_single", "resolved_multi"}:
            raise Stage4AError(f"{experiment_id} classification must be resolved")
        if len(module_ids) > 1 and classification != "resolved_multi":
            raise Stage4AError(f"{experiment_id} multi-module classification is inconsistent")
        if len(module_ids) == 1 and classification != "resolved_single":
            raise Stage4AError(f"{experiment_id} single-module classification is inconsistent")
        if len(module_ids) > 1:
            require_string(
                record.get("classification_rationale"),
                f"{experiment_id} classification_rationale",
            )
    return len(records)


def reject_stage4b_or_stage4c_outputs(repo_root: Path, schema: dict[str, Any]) -> None:
    root = repo_root / ROOT_REL
    for value in schema["forbidden_stage4a_outputs"]:
        path = root / value.rstrip("/")
        if path.exists() or path.is_symlink():
            raise Stage4AError(f"Stage 4A contains forbidden 4B/4C output: {value}")


def validate(repo_root: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    schema_path = safe_file(repo_root, SCHEMA_REL, "Stage 4A schema")
    modules_path = safe_file(repo_root, MODULES_REL, "module inventory")
    headings_path = safe_file(repo_root, HEADINGS_REL, "heading inventory")
    claims_path = safe_file(repo_root, CLAIMS_REL, "claim inventory")
    experiments_path = safe_file(repo_root, EXPERIMENTS_REL, "experiment inventory")
    handoff_path = safe_file(repo_root, HANDOFF_REL, "manual handoff")
    registry_path = safe_file(repo_root, REGISTRY_REL, "experiment registry")

    schema = load_mapping(schema_path, "Stage 4A schema")
    validate_schema(schema)
    reject_stage4b_or_stage4c_outputs(repo_root, schema)
    modules = validate_modules(load_mapping(modules_path, "module inventory"))
    headings, sections = validate_headings(
        load_mapping(headings_path, "heading inventory"), handoff_path, modules
    )
    claims = validate_claims(
        load_mapping(claims_path, "claim inventory"),
        handoff_path,
        modules,
        headings,
        sections,
    )
    experiment_count = validate_experiments(
        load_mapping(experiments_path, "experiment inventory"),
        registry_path,
        modules,
        headings,
        claims,
    )
    return {
        "status": "PASS",
        "policy_id": POLICY_ID,
        "phase": "stage_4a_schema_inventory",
        "authority": AUTHORITY,
        "manual_handoff_remains_authoritative": True,
        "authority_cutover_allowed": False,
        "module_count": len(modules),
        "heading_count": len(headings),
        "claim_count": len(claims),
        "experiment_count": experiment_count,
        "handoff_sha256": sha256_path(handoff_path),
        "registry_sha256": sha256_path(registry_path),
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = validate(args.repo_root)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print("Stage 4A schema/inventory: PASS")
        print(
            "Objects: "
            f"modules={report['module_count']} headings={report['heading_count']} "
            f"claims={report['claim_count']} experiments={report['experiment_count']}"
        )
        print("Manual docs/handoff.md remains authoritative; cutover is forbidden.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
