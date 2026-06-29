#!/usr/bin/env python3
"""Build the deterministic Stage 4A dynamic semantic graph.

The builder is intentionally offline and project-agnostic.  Project-specific
semantics live in a versioned profile and small explicit overrides.  Unresolved
semantics are emitted to a review queue and never leak into accepted graph
edges.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml


DEFAULT_PROFILE = Path(
    "docs/handoff_shadow/stage4/dynamic/profiles/DRPO_PROFILE.yaml"
)
DEFAULT_OVERRIDES = Path(
    "docs/handoff_shadow/stage4/dynamic/overrides/SEMANTIC_OVERRIDES.yaml"
)
DEFAULT_OUTPUT = Path("docs/handoff_shadow/stage4/dynamic/generated")
GENERATED_FILES = (
    "NODES.yaml",
    "EDGES.yaml",
    "REVIEW_QUEUE.yaml",
    "graph/OVERVIEW.md",
    "graph/CLAIM_EXPERIMENT.md",
    "graph/LINEAGE.md",
    "graph/FULL_GRAPH.dot",
    "GRAPH_MANIFEST.json",
)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
SEMANTIC_FINGERPRINT_VERSION = 1


class SemanticGraphError(ValueError):
    """Raised when graph inputs or generated outputs violate the contract."""


class StableDumper(yaml.SafeDumper):
    def ignore_aliases(self, data: Any) -> bool:  # noqa: ANN401
        return True


def load_yaml(path: Path, label: str, *, required: bool = True) -> dict[str, Any]:
    if not path.exists():
        if required:
            raise SemanticGraphError(f"missing {label}: {path}")
        return {}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise SemanticGraphError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SemanticGraphError(f"{label} must be a YAML mapping: {path}")
    return payload


def dump_yaml(payload: dict[str, Any]) -> bytes:
    return yaml.dump(
        payload,
        Dumper=StableDumper,
        sort_keys=False,
        allow_unicode=True,
        width=120,
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_hash(payload: Any) -> str:
    data = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256_bytes(data)


def stable_token(*parts: str, length: int = 16) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:length]


def review_id_for(
    kind: str,
    object_id: str,
    reason: str,
    candidates: list[str] | None = None,
) -> str:
    key = (kind, object_id, reason, tuple(candidates or []))
    return "review:" + stable_token(*(str(x) for x in key), length=20)


def semantic_fingerprint(payload: dict[str, Any], *ignored_keys: str) -> str:
    ignored = set(ignored_keys)
    return canonical_hash({key: value for key, value in payload.items() if key not in ignored})


def safe_rel_path(repo_root: Path, value: str, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise SemanticGraphError(f"{label} must be a non-empty path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise SemanticGraphError(f"unsafe {label}: {value}")
    path = (repo_root / relative).resolve()
    try:
        path.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise SemanticGraphError(f"{label} escapes repository: {value}") from exc
    return path


def require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise SemanticGraphError(f"{label} must be a list")
    return value


def discover_headings(path: Path) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        match = HEADING_RE.match(line)
        if not match:
            continue
        title = match.group(2).strip()
        counts[title] += 1
        occurrence = counts[title]
        identity = f"{path.as_posix()}|{title}|{occurrence}"
        rows.append(
            {
                "line": line_no,
                "level": len(match.group(1)),
                "title": title,
                "occurrence": occurrence,
                "identity": identity,
            }
        )
    return rows


def as_mapping_by(items: Iterable[dict[str, Any]], key: str, label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get(key), str):
            raise SemanticGraphError(f"{label} item missing string {key}")
        value = item[key]
        if value in result:
            raise SemanticGraphError(f"duplicate {label} {key}: {value}")
        result[value] = item
    return result


def normalize_node_id(kind: str, raw: str) -> str:
    return f"{kind}:{raw}"


def edge_id(source: str, relation: str, target: str) -> str:
    return f"edge:{stable_token(source, relation, target, length=20)}"


def mermaid_id(node_id: str) -> str:
    return "n_" + stable_token(node_id, length=12)


def mermaid_label(text: str) -> str:
    return text.replace('"', "'").replace("\n", " ")[:100]


def dot_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


@dataclass(frozen=True)
class BuildConfig:
    repo_root: Path
    profile_path: Path
    overrides_path: Path
    output_dir: Path


class GraphBuilder:
    def __init__(self, config: BuildConfig):
        self.config = config
        self.repo_root = config.repo_root.resolve()
        self.profile = load_yaml(config.profile_path, "project profile")
        self.overrides = load_yaml(config.overrides_path, "semantic overrides")
        kernel_value = self.profile.get("kernel")
        if not isinstance(kernel_value, str):
            raise SemanticGraphError("profile.kernel must be a repository-relative path")
        self.kernel_path = safe_rel_path(self.repo_root, kernel_value, "kernel path")
        self.kernel = load_yaml(self.kernel_path, "research semantic kernel")
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: dict[tuple[str, str, str], dict[str, Any]] = {}
        self.review: list[dict[str, Any]] = []
        self.sources: dict[str, dict[str, str]] = {}
        self.module_ids: set[str] = set()
        self.override_assignments: dict[tuple[str, str], list[str]] = {}
        self.explicit_edge_keys: set[tuple[str, str, str]] = set()
        self.rejected_decisions: dict[str, dict[str, Any]] = {}
        self.rejected_current: list[dict[str, Any]] = []
        self.rejected_matched: set[str] = set()
        self.module_lifecycle_changes: list[dict[str, Any]] = []
        self.effective_modules: list[dict[str, Any]] = []
        self.override_version = 0

    def add_node(self, node: dict[str, Any]) -> None:
        node_id = node.get("node_id")
        if not isinstance(node_id, str) or not node_id:
            raise SemanticGraphError("node_id must be a non-empty string")
        if node_id in self.nodes and self.nodes[node_id] != node:
            raise SemanticGraphError(f"duplicate node_id with different payload: {node_id}")
        self.nodes[node_id] = node

    def add_edge(
        self,
        source: str,
        relation: str,
        target: str,
        *,
        provenance: str,
        rationale: str | None = None,
    ) -> None:
        key = (source, relation, target)
        payload: dict[str, Any] = {
            "edge_id": edge_id(*key),
            "source": source,
            "relation": relation,
            "target": target,
            "lifecycle_status": "accepted",
            "review_state": "not_required",
            "provenance": provenance,
        }
        if rationale:
            payload["rationale"] = rationale
        existing = self.edges.get(key)
        if existing and existing != payload:
            raise SemanticGraphError(f"conflicting accepted edge: {key}")
        self.edges[key] = payload

    def add_review(self, kind: str, object_id: str, reason: str, candidates: list[str] | None = None) -> None:
        normalized_candidates = candidates or []
        review_id = review_id_for(kind, object_id, reason, normalized_candidates)
        decision = self.rejected_decisions.get(review_id)
        if decision is not None:
            if review_id not in self.rejected_matched:
                self.rejected_matched.add(review_id)
                self.rejected_current.append(
                    {
                        "review_id": review_id,
                        "kind": kind,
                        "object_id": object_id,
                        "state": "rejected",
                        "reason": reason,
                        "candidates": normalized_candidates,
                        "rationale": decision["rationale"],
                        "decision_version": decision.get("decision_version"),
                        "match_state": "matched_current_candidate",
                    }
                )
            return
        self.review.append(
            {
                "review_id": review_id,
                "kind": kind,
                "object_id": object_id,
                "state": "pending",
                "reason": reason,
                "candidates": normalized_candidates,
            }
        )

    def source_path(self, key: str, *, required: bool = True) -> Path | None:
        sources = self.profile.get("sources")
        if not isinstance(sources, dict):
            raise SemanticGraphError("profile.sources must be a mapping")
        value = sources.get(key)
        if value in (None, ""):
            if required:
                raise SemanticGraphError(f"profile.sources.{key} is required")
            return None
        if not isinstance(value, str):
            raise SemanticGraphError(f"profile.sources.{key} must be a path")
        path = safe_rel_path(self.repo_root, value, f"source {key}")
        if required and not path.is_file():
            raise SemanticGraphError(f"missing source {key}: {value}")
        if path.is_file():
            self.sources[key] = {"path": value, "sha256": sha256_file(path)}
        return path

    def prepare_overrides(self) -> None:
        if self.overrides.get("schema_version") != 1:
            raise SemanticGraphError("semantic overrides schema_version must be 1")
        if self.overrides.get("profile_id") != self.profile.get("profile_id"):
            raise SemanticGraphError("overrides profile_id does not match project profile")
        if self.overrides.get("profile_version") != self.profile.get("profile_version"):
            raise SemanticGraphError("overrides profile_version does not match project profile")
        override_version = self.overrides.get("override_version")
        if not isinstance(override_version, int) or override_version < 1:
            raise SemanticGraphError("overrides override_version must be a positive integer")
        self.override_version = override_version

        for item in require_list(self.overrides.get("module_assignments", []), "module_assignments"):
            if not isinstance(item, dict):
                raise SemanticGraphError("module_assignments entries must be mappings")
            kind = item.get("object_type")
            object_id = item.get("object_id")
            module_ids = item.get("module_ids")
            if not isinstance(kind, str) or not isinstance(object_id, str):
                raise SemanticGraphError("module assignment requires object_type and object_id")
            if not isinstance(module_ids, list) or not module_ids or not all(
                isinstance(x, str) and x for x in module_ids
            ):
                raise SemanticGraphError("module assignment module_ids must be non-empty strings")
            key = (kind, object_id)
            if key in self.override_assignments:
                raise SemanticGraphError(f"duplicate module assignment override: {key}")
            self.override_assignments[key] = sorted(set(module_ids))

        for item in require_list(
            self.overrides.get("rejected_candidates", []), "rejected_candidates"
        ):
            if not isinstance(item, dict):
                raise SemanticGraphError("rejected_candidates entries must be mappings")
            required = ("review_id", "kind", "object_id", "reason", "rationale")
            if not all(isinstance(item.get(key), str) and item.get(key) for key in required):
                raise SemanticGraphError(
                    "rejected candidate requires review_id, kind, object_id, reason, and rationale"
                )
            candidates = item.get("candidates", [])
            if not isinstance(candidates, list) or not all(isinstance(x, str) for x in candidates):
                raise SemanticGraphError("rejected candidate candidates must be strings")
            expected_review_id = review_id_for(
                item["kind"], item["object_id"], item["reason"], candidates
            )
            if item["review_id"] != expected_review_id:
                raise SemanticGraphError(
                    f"rejected candidate review_id does not match semantic signature: {item['review_id']}"
                )
            if item["review_id"] in self.rejected_decisions:
                raise SemanticGraphError(f"duplicate rejected candidate: {item['review_id']}")
            decision_version = item.get("decision_version", override_version)
            if not isinstance(decision_version, int) or decision_version < 1:
                raise SemanticGraphError("rejected candidate decision_version must be positive")
            if decision_version > override_version:
                raise SemanticGraphError(
                    "rejected candidate decision_version cannot exceed override_version"
                )
            normalized = dict(item)
            normalized["candidates"] = candidates
            normalized["decision_version"] = decision_version
            self.rejected_decisions[item["review_id"]] = normalized

        lifecycle_changes = require_list(
            self.overrides.get("module_lifecycle_changes", []),
            "module_lifecycle_changes",
        )
        change_ids: set[str] = set()
        for item in lifecycle_changes:
            if not isinstance(item, dict):
                raise SemanticGraphError("module_lifecycle_changes entries must be mappings")
            change_id = item.get("change_id")
            operation = item.get("operation")
            rationale = item.get("rationale")
            if not isinstance(change_id, str) or not change_id:
                raise SemanticGraphError("module lifecycle change requires change_id")
            if change_id in change_ids:
                raise SemanticGraphError(f"duplicate module lifecycle change_id: {change_id}")
            change_ids.add(change_id)
            if operation not in {"rename", "supersede", "split", "merge"}:
                raise SemanticGraphError(f"unsupported module lifecycle operation: {operation}")
            if not isinstance(rationale, str) or not rationale.strip():
                raise SemanticGraphError("module lifecycle change requires rationale")
            sources = item.get("source_module_ids")
            targets = item.get("target_module_ids")
            if not isinstance(sources, list) or not sources or not all(
                isinstance(x, str) and x for x in sources
            ):
                raise SemanticGraphError("module lifecycle source_module_ids must be non-empty")
            if not isinstance(targets, list) or not targets or not all(
                isinstance(x, str) and x for x in targets
            ):
                raise SemanticGraphError("module lifecycle target_module_ids must be non-empty")
            if len(set(sources)) != len(sources) or len(set(targets)) != len(targets):
                raise SemanticGraphError("module lifecycle source/target IDs must be unique")
            if operation == "rename" and not (len(sources) == len(targets) == 1 and sources == targets):
                raise SemanticGraphError("rename requires the same single source and target module")
            if operation == "rename" and not isinstance(item.get("new_name"), str):
                raise SemanticGraphError("rename requires new_name")
            if operation == "split" and not (len(sources) == 1 and len(targets) >= 2):
                raise SemanticGraphError("split requires one source and at least two targets")
            if operation == "merge" and not (len(sources) >= 2 and len(targets) == 1):
                raise SemanticGraphError("merge requires at least two sources and one target")
            if operation != "rename" and set(sources) & set(targets):
                raise SemanticGraphError("supersede/split/merge sources and targets must be disjoint")
            from_versions = item.get("from_versions")
            to_versions = item.get("to_versions")
            if not isinstance(from_versions, dict) or not isinstance(to_versions, dict):
                raise SemanticGraphError("module lifecycle change requires from_versions and to_versions")
            touched = set(sources) | set(targets)
            if set(from_versions) != touched or set(to_versions) != touched:
                raise SemanticGraphError(
                    "module lifecycle version maps must cover every source and target exactly"
                )
            if not all(isinstance(value, int) and value >= 1 for value in from_versions.values()):
                raise SemanticGraphError("module lifecycle from_versions must be positive integers")
            if not all(isinstance(value, int) and value >= 1 for value in to_versions.values()):
                raise SemanticGraphError("module lifecycle to_versions must be positive integers")
            if any(to_versions[key] <= from_versions[key] for key in touched):
                raise SemanticGraphError("module lifecycle changes must increment every touched module version")
            self.module_lifecycle_changes.append(dict(item))

        combined_edges = list(self.profile.get("explicit_edges", [])) + list(
            self.overrides.get("accepted_edges", [])
        )
        for item in combined_edges:
            if isinstance(item, dict) and all(
                isinstance(item.get(key), str) and item.get(key)
                for key in ("source", "relation", "target")
            ):
                self.explicit_edge_keys.add(
                    (item["source"], item["relation"], item["target"])
                )

    def effective_module_records(self) -> list[dict[str, Any]]:
        modules = require_list(self.profile.get("modules"), "profile.modules")
        mapping: dict[str, dict[str, Any]] = {}
        for item in modules:
            if not isinstance(item, dict):
                raise SemanticGraphError("module entries must be mappings")
            raw_id = item.get("module_id")
            version = item.get("version")
            if not isinstance(raw_id, str) or not raw_id:
                raise SemanticGraphError("module_id must be a string")
            if raw_id in mapping:
                raise SemanticGraphError(f"duplicate module_id: {raw_id}")
            if not isinstance(version, int) or version < 1:
                raise SemanticGraphError(f"module {raw_id} version must be a positive integer")
            mapping[raw_id] = dict(item)

        policies = self.profile.get("policies", {})
        require_increment = bool(
            isinstance(policies, dict)
            and policies.get("module_change_requires_version_increment", False)
        )
        require_supersedes = bool(
            isinstance(policies, dict)
            and policies.get("module_split_merge_requires_supersedes_record", False)
        )
        for change in self.module_lifecycle_changes:
            sources = list(change["source_module_ids"])
            targets = list(change["target_module_ids"])
            touched = sources + [x for x in targets if x not in sources]
            missing = sorted(set(touched) - set(mapping))
            if missing:
                raise SemanticGraphError(
                    f"module lifecycle change {change['change_id']} references unknown modules {missing}"
                )
            for module_id in touched:
                expected = change["from_versions"][module_id]
                current = mapping[module_id].get("version")
                if current != expected:
                    raise SemanticGraphError(
                        f"module lifecycle change {change['change_id']} expected {module_id} "
                        f"version {expected}, found {current}"
                    )
            if require_increment and any(
                change["to_versions"][module_id] <= change["from_versions"][module_id]
                for module_id in touched
            ):
                raise SemanticGraphError("module lifecycle version increment policy violated")

            operation = change["operation"]
            if operation == "rename":
                module_id = sources[0]
                mapping[module_id]["name"] = change["new_name"]
            else:
                for source in sources:
                    mapping[source]["lifecycle_status"] = "superseded"
                    superseded_by = set(mapping[source].get("superseded_by", []))
                    superseded_by.update(targets)
                    mapping[source]["superseded_by"] = sorted(superseded_by)
                for target in targets:
                    mapping[target]["lifecycle_status"] = "active"
                    supersedes = set(mapping[target].get("supersedes", []))
                    supersedes.update(sources)
                    mapping[target]["supersedes"] = sorted(supersedes)
                if require_supersedes and not all(
                    set(sources).issubset(set(mapping[target].get("supersedes", [])))
                    for target in targets
                ):
                    raise SemanticGraphError("split/merge must preserve supersedes records")
            for module_id in touched:
                mapping[module_id]["version"] = change["to_versions"][module_id]
                history = list(mapping[module_id].get("lifecycle_changes", []))
                history.append(
                    {
                        "change_id": change["change_id"],
                        "operation": operation,
                        "rationale": change["rationale"],
                    }
                )
                mapping[module_id]["lifecycle_changes"] = history
        return [mapping[key] for key in sorted(mapping)]

    def previous_manifest(self) -> dict[str, Any] | None:
        path = self.config.output_dir / "GRAPH_MANIFEST.json"
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SemanticGraphError(f"cannot read existing graph manifest: {exc}") from exc
        if not isinstance(payload, dict):
            raise SemanticGraphError("existing graph manifest must be a mapping")
        return payload

    def current_version_state(self) -> dict[str, Any]:
        module_state: dict[str, dict[str, Any]] = {}
        for item in self.effective_modules:
            raw_id = str(item["module_id"])
            module_state[raw_id] = {
                "version": item["version"],
                "semantic_fingerprint": semantic_fingerprint(item, "version"),
                "lifecycle_status": item.get("lifecycle_status", "active"),
                "title": str(item.get("name", raw_id)),
            }
        return {
            "profile_semantic_fingerprint": semantic_fingerprint(
                self.profile, "profile_version"
            ),
            "override_semantic_fingerprint": semantic_fingerprint(
                self.overrides,
                "schema_version",
                "profile_id",
                "profile_version",
                "authority",
                "override_version",
                "notes",
            ),
            "override_version": self.override_version,
            "module_state": module_state,
        }

    def enforce_version_history(self) -> None:
        previous = self.previous_manifest()
        if not previous:
            return
        current = self.current_version_state()
        previous_profile_fp = previous.get("profile_semantic_fingerprint")
        previous_override_fp = previous.get("override_semantic_fingerprint")
        previous_module_state = previous.get("module_state")
        # Old manifests and manifests produced before fingerprint-algorithm versioning
        # are migrated once; subsequent builds are strictly versioned.
        if (
            previous.get("semantic_fingerprint_version") != SEMANTIC_FINGERPRINT_VERSION
            or not isinstance(previous_profile_fp, str)
            or not isinstance(previous_module_state, dict)
        ):
            return
        previous_profile_version = previous.get("profile_version")
        current_profile_version = self.profile.get("profile_version")
        if (
            current["profile_semantic_fingerprint"] != previous_profile_fp
            and (
                not isinstance(previous_profile_version, int)
                or not isinstance(current_profile_version, int)
                or current_profile_version <= previous_profile_version
            )
        ):
            raise SemanticGraphError("project profile semantics changed without profile_version increment")
        if isinstance(previous_override_fp, str):
            previous_override_version = previous.get("override_version")
            if (
                current["override_semantic_fingerprint"] != previous_override_fp
                and (
                    not isinstance(previous_override_version, int)
                    or self.override_version <= previous_override_version
                )
            ):
                raise SemanticGraphError("semantic overrides changed without override_version increment")

        current_modules = current["module_state"]
        removed = sorted(set(previous_module_state) - set(current_modules))
        if removed:
            raise SemanticGraphError(
                f"modules may not be destructively removed; preserve and supersede them: {removed}"
            )
        policies = self.profile.get("policies", {})
        require_increment = bool(
            isinstance(policies, dict)
            and policies.get("module_change_requires_version_increment", False)
        )
        for module_id in sorted(set(previous_module_state) & set(current_modules)):
            before = previous_module_state[module_id]
            after = current_modules[module_id]
            if not isinstance(before, dict):
                continue
            if before.get("semantic_fingerprint") != after.get("semantic_fingerprint"):
                if require_increment and (
                    not isinstance(before.get("version"), int)
                    or after.get("version") <= before.get("version")
                ):
                    raise SemanticGraphError(
                        f"module {module_id} semantics changed without version increment"
                    )

    def build_modules(self) -> None:
        project = self.profile.get("project")
        if not isinstance(project, dict):
            raise SemanticGraphError("profile.project must be a mapping")
        project_id = project.get("node_id")
        title = project.get("title")
        if not isinstance(project_id, str) or not isinstance(title, str):
            raise SemanticGraphError("project requires node_id and title")
        self.add_node(
            {
                "node_id": project_id,
                "node_type": "project",
                "lifecycle_status": "active",
                "title": title,
                "module_ids": [],
                "provenance": {"kind": "project_profile"},
            }
        )
        self.effective_modules = self.effective_module_records()
        self.enforce_version_history()
        for item in self.effective_modules:
            raw_id = item["module_id"]
            self.module_ids.add(raw_id)
            node_id = normalize_node_id("module", raw_id)
            node = {
                "node_id": node_id,
                "node_type": "module",
                "lifecycle_status": item.get("lifecycle_status", "active"),
                "title": str(item.get("name", raw_id)),
                "module_ids": [],
                "provenance": {
                    "kind": "project_profile_with_approved_overrides",
                    "profile_version": self.profile.get("profile_version"),
                    "override_version": self.override_version,
                },
                "attributes": {
                    "module_id": raw_id,
                    "version": item.get("version"),
                    "purpose": item.get("purpose", ""),
                    "supersedes": sorted(item.get("supersedes", [])),
                    "superseded_by": sorted(item.get("superseded_by", [])),
                    "lifecycle_changes": item.get("lifecycle_changes", []),
                },
            }
            self.add_node(node)
            self.add_edge(node_id, "member_of", project_id, provenance="project_profile")
        for item in self.effective_modules:
            source = normalize_node_id("module", item["module_id"])
            for dependency in sorted(item.get("default_dependencies", [])):
                if dependency not in self.module_ids:
                    raise SemanticGraphError(
                        f"module {item['module_id']} has unknown dependency {dependency}"
                    )
                self.add_edge(
                    source,
                    "depends_on",
                    normalize_node_id("module", dependency),
                    provenance="project_profile",
                )
            for old in sorted(item.get("supersedes", [])):
                if old not in self.module_ids:
                    raise SemanticGraphError(f"module {item['module_id']} supersedes unknown {old}")
                self.add_edge(
                    source,
                    "supersedes",
                    normalize_node_id("module", old),
                    provenance="project_profile_or_lifecycle_override",
                )

    def module_assignment_from_rules(self, record: dict[str, Any]) -> list[str]:
        result: set[str] = set()
        for rule in require_list(
            self.profile.get("module_inference_rules", []), "module_inference_rules"
        ):
            if not isinstance(rule, dict):
                raise SemanticGraphError("module inference rules must be mappings")
            field = rule.get("field")
            module_id = rule.get("module_id")
            if not isinstance(field, str) or module_id not in self.module_ids:
                raise SemanticGraphError("invalid module inference rule")
            value = record.get(field)
            matches = "equals" in rule and value == rule.get("equals")
            if "regex" in rule:
                matches = bool(re.fullmatch(str(rule["regex"]), str(value or "")))
            if matches:
                result.add(module_id)
        return sorted(result)

    def build_headings(self) -> None:
        handoff = self.source_path("handoff")
        assert handoff is not None
        bootstrap_path = self.source_path("bootstrap_headings", required=False)
        bootstrap: dict[tuple[str, int], dict[str, Any]] = {}
        if bootstrap_path and bootstrap_path.is_file():
            payload = load_yaml(bootstrap_path, "bootstrap heading inventory")
            for row in require_list(payload.get("headings", []), "bootstrap headings"):
                if isinstance(row, dict):
                    bootstrap[(str(row.get("title")), int(row.get("occurrence", 0)))] = row
        for heading in discover_headings(handoff):
            seed = bootstrap.get((heading["title"], heading["occurrence"]))
            if seed:
                raw_id = str(seed["heading_id"])
                modules = sorted(seed.get("module_ids", []))
            else:
                raw_id = "AUTO-" + stable_token(heading["identity"], length=16)
                modules = self.override_assignments.get(("source_section", raw_id), [])
            unknown = sorted(set(modules) - self.module_ids)
            if unknown:
                raise SemanticGraphError(f"heading {raw_id} uses unknown modules {unknown}")
            node_id = normalize_node_id("heading", raw_id)
            lifecycle = "accepted" if modules else "proposed"
            self.add_node(
                {
                    "node_id": node_id,
                    "node_type": "source_section",
                    "lifecycle_status": lifecycle,
                    "title": heading["title"],
                    "module_ids": modules,
                    "provenance": {
                        "kind": "mechanical_markdown_discovery",
                        "source": self.sources["handoff"],
                        "line": heading["line"],
                        "level": heading["level"],
                        "occurrence": heading["occurrence"],
                    },
                }
            )
            if not modules:
                self.add_review(
                    "module_assignment",
                    node_id,
                    "new heading has no accepted module assignment",
                    [],
                )
            for module_id in modules:
                self.add_edge(
                    node_id,
                    "member_of",
                    normalize_node_id("module", module_id),
                    provenance="bootstrap_heading_inventory",
                )

    def build_claims(self) -> None:
        path = self.source_path("bootstrap_claims", required=False)
        if not path or not path.is_file():
            return
        payload = load_yaml(path, "bootstrap claim inventory")
        for item in require_list(payload.get("claims", []), "bootstrap claims"):
            if not isinstance(item, dict):
                raise SemanticGraphError("bootstrap claim must be a mapping")
            raw_id = item.get("claim_id")
            if not isinstance(raw_id, str):
                raise SemanticGraphError("claim_id must be a string")
            modules = self.override_assignments.get(
                ("claim", raw_id), sorted(item.get("module_ids", []))
            )
            unknown = sorted(set(modules) - self.module_ids)
            if unknown:
                raise SemanticGraphError(f"claim {raw_id} uses unknown modules {unknown}")
            domain_status = str(item.get("status", "unknown"))
            lifecycle = "superseded" if "superseded" in domain_status else "accepted"
            node_id = normalize_node_id("claim", raw_id)
            self.add_node(
                {
                    "node_id": node_id,
                    "node_type": str(item.get("node_type", "claim")),
                    "lifecycle_status": lifecycle,
                    "title": str(item.get("statement_summary", raw_id)),
                    "module_ids": modules,
                    "provenance": {
                        "kind": "bootstrap_claim_inventory",
                        "source_anchor": item.get("source_anchor", {}),
                    },
                    "attributes": {"claim_id": raw_id, "domain_status": domain_status},
                }
            )
            for module_id in modules:
                self.add_edge(
                    node_id,
                    "member_of",
                    normalize_node_id("module", module_id),
                    provenance="bootstrap_claim_inventory",
                )
            lineage = item.get("lineage", {})
            if isinstance(lineage, dict):
                for old in sorted(lineage.get("supersedes", [])):
                    self.add_edge(
                        node_id,
                        "supersedes",
                        normalize_node_id("claim", str(old)),
                        provenance="bootstrap_claim_inventory",
                    )

    def build_experiments(self) -> None:
        registry_path = self.source_path("registry")
        assert registry_path is not None
        registry = load_yaml(registry_path, "experiment registry")
        rows = require_list(registry.get("experiments", []), "registry experiments")
        bootstrap_path = self.source_path("bootstrap_experiments", required=False)
        bootstrap: dict[str, dict[str, Any]] = {}
        if bootstrap_path and bootstrap_path.is_file():
            payload = load_yaml(bootstrap_path, "bootstrap experiment inventory")
            bootstrap = as_mapping_by(
                require_list(payload.get("experiments", []), "bootstrap experiments"),
                "experiment_id",
                "experiment",
            )
        for index, record in enumerate(rows, 1):
            if not isinstance(record, dict) or not isinstance(record.get("id"), str):
                raise SemanticGraphError(f"registry experiment {index} lacks string id")
            raw_id = record["id"]
            seed = bootstrap.get(raw_id)
            modules = self.override_assignments.get(
                ("experiment", raw_id),
                sorted(seed.get("module_ids", [])) if seed else self.module_assignment_from_rules(record),
            )
            unknown = sorted(set(modules) - self.module_ids)
            if unknown:
                raise SemanticGraphError(f"experiment {raw_id} uses unknown modules {unknown}")
            claim_ids = sorted(seed.get("claim_ids", [])) if seed else []
            role = str(seed.get("role", record.get("role", "unknown"))) if seed else str(
                record.get("role", "unknown")
            )
            domain_status = str(seed.get("registry_status", record.get("status", "unknown"))) if seed else str(
                record.get("status", "unknown")
            )
            node_id = normalize_node_id("experiment", raw_id)
            lifecycle = "superseded" if domain_status == "superseded" else (
                "blocked" if "blocked" in str(record.get("execution_gate", "")) else "active"
            )
            self.add_node(
                {
                    "node_id": node_id,
                    "node_type": "experiment",
                    "lifecycle_status": lifecycle,
                    "title": str(record.get("name") or raw_id),
                    "module_ids": modules,
                    "provenance": {
                        "kind": "mechanical_registry_discovery",
                        "source": self.sources["registry"],
                        "registry_index": index,
                    },
                    "attributes": {
                        "experiment_id": raw_id,
                        "environment": record.get("environment"),
                        "role": role,
                        "domain_status": domain_status,
                    },
                }
            )
            if not modules:
                self.add_review(
                    "module_assignment",
                    node_id,
                    "experiment did not match any accepted project module rule",
                    [],
                )
            for module_id in modules:
                self.add_edge(
                    node_id,
                    "member_of",
                    normalize_node_id("module", module_id),
                    provenance="bootstrap_or_profile_inference",
                )
            relation = "external_validates" if role.startswith("external") else "tests"
            explicit_claim_relation = any(
                source == node_id
                and accepted_relation in {"tests", "external_validates", "supports"}
                and target.startswith("claim:")
                for source, accepted_relation, target in self.explicit_edge_keys
            )
            if not claim_ids and not explicit_claim_relation:
                self.add_review(
                    "claim_relation",
                    node_id,
                    "experiment has no accepted claim relation",
                    [],
                )
            for claim_id in claim_ids:
                self.add_edge(
                    node_id,
                    relation,
                    normalize_node_id("claim", str(claim_id)),
                    provenance="bootstrap_experiment_inventory",
                )

    def build_explicit_edges(self) -> None:
        combined = list(self.profile.get("explicit_edges", [])) + list(
            self.overrides.get("accepted_edges", [])
        )
        for item in combined:
            if not isinstance(item, dict):
                raise SemanticGraphError("explicit edge entries must be mappings")
            source = item.get("source")
            relation = item.get("relation")
            target = item.get("target")
            if not all(isinstance(x, str) and x for x in (source, relation, target)):
                raise SemanticGraphError("explicit edge requires source, relation, target")
            self.add_edge(
                source,
                relation,
                target,
                provenance="project_profile_or_override",
                rationale=str(item.get("rationale", "")) or None,
            )

    def finalize_rejected_decisions(self) -> list[dict[str, Any]]:
        decisions = list(self.rejected_current)
        for review_id, item in sorted(self.rejected_decisions.items()):
            if review_id in self.rejected_matched:
                continue
            decisions.append(
                {
                    "review_id": review_id,
                    "kind": item["kind"],
                    "object_id": item["object_id"],
                    "state": "rejected",
                    "reason": item["reason"],
                    "candidates": item.get("candidates", []),
                    "rationale": item["rationale"],
                    "decision_version": item.get("decision_version"),
                    "match_state": "historical_not_current",
                }
            )
        return sorted(decisions, key=lambda x: x["review_id"])

    def validate_graph(self) -> None:
        node_types = set(self.kernel.get("node_types", []))
        relations = set(self.kernel.get("relation_types", []))
        node_statuses = set(self.kernel.get("node_lifecycle_statuses", []))
        for node in self.nodes.values():
            if node["node_type"] not in node_types:
                raise SemanticGraphError(
                    f"node {node['node_id']} has unknown type {node['node_type']}"
                )
            if node["lifecycle_status"] not in node_statuses:
                raise SemanticGraphError(
                    f"node {node['node_id']} has unknown lifecycle {node['lifecycle_status']}"
                )
        for edge in self.edges.values():
            if edge["relation"] not in relations:
                raise SemanticGraphError(
                    f"edge {edge['edge_id']} has unknown relation {edge['relation']}"
                )
            if edge["source"] not in self.nodes or edge["target"] not in self.nodes:
                raise SemanticGraphError(f"dangling accepted edge {edge['edge_id']}")
            if self.nodes[edge["source"]]["lifecycle_status"] == "rejected" or self.nodes[
                edge["target"]
            ]["lifecycle_status"] == "rejected":
                raise SemanticGraphError(f"accepted edge references rejected node {edge['edge_id']}")
        # Cycle detection for supersedes only.
        graph: dict[str, list[str]] = {}
        for edge in self.edges.values():
            if edge["relation"] == "supersedes":
                graph.setdefault(edge["source"], []).append(edge["target"])
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node: str) -> None:
            if node in visiting:
                raise SemanticGraphError("supersedes graph contains a cycle")
            if node in visited:
                return
            visiting.add(node)
            for target in graph.get(node, []):
                visit(target)
            visiting.remove(node)
            visited.add(node)

        for node in sorted(graph):
            visit(node)

    def build(self) -> dict[str, bytes]:
        if self.profile.get("schema_version") != 1 or self.kernel.get("schema_version") != 1:
            raise SemanticGraphError("kernel and profile schema_version must be 1")
        self.prepare_overrides()
        self.build_modules()
        self.build_headings()
        self.build_claims()
        self.build_experiments()
        self.build_explicit_edges()
        self.validate_graph()

        nodes = [self.nodes[key] for key in sorted(self.nodes)]
        edges = [self.edges[key] for key in sorted(self.edges)]
        review = sorted(self.review, key=lambda x: x["review_id"])
        rejected = self.finalize_rejected_decisions()
        graph_hash = canonical_hash(
            {
                "nodes": nodes,
                "edges": edges,
                "review_queue": review,
                "rejected_candidates": rejected,
            }
        )
        version_state = self.current_version_state()
        common = {
            "schema_version": 1,
            "authority": "non_authoritative_stage4a_dynamic_shadow_graph",
            "kernel_id": self.kernel.get("kernel_id"),
            "kernel_version": self.kernel.get("kernel_version"),
            "profile_id": self.profile.get("profile_id"),
            "profile_version": self.profile.get("profile_version"),
            "override_version": self.override_version,
            "graph_hash": graph_hash,
            "sources": dict(sorted(self.sources.items())),
        }
        files: dict[str, bytes] = {
            "NODES.yaml": dump_yaml({**common, "nodes": nodes}),
            "EDGES.yaml": dump_yaml({**common, "edges": edges}),
            "REVIEW_QUEUE.yaml": dump_yaml(
                {
                    **common,
                    "review_queue": review,
                    "rejected_candidates": rejected,
                }
            ),
        }
        files["graph/OVERVIEW.md"] = self.render_overview(graph_hash).encode("utf-8")
        files["graph/CLAIM_EXPERIMENT.md"] = self.render_claim_experiment(graph_hash).encode(
            "utf-8"
        )
        files["graph/LINEAGE.md"] = self.render_lineage(graph_hash).encode("utf-8")
        files["graph/FULL_GRAPH.dot"] = self.render_dot(graph_hash).encode("utf-8")
        manifest_files = {
            name: sha256_bytes(data) for name, data in sorted(files.items())
        }
        manifest = {
            **common,
            **version_state,
            "semantic_fingerprint_version": SEMANTIC_FINGERPRINT_VERSION,
            "generator": "scripts/build_stage4_semantic_graph.py",
            "generated_files": manifest_files,
            "counts": {
                "nodes": len(nodes),
                "edges": len(edges),
                "review_queue": len(review),
                "rejected_candidates": len(rejected),
                "modules": sum(node["node_type"] == "module" for node in nodes),
                "headings": sum(node["node_type"] == "source_section" for node in nodes),
                "claims": sum(node["node_type"] in {"claim", "hypothesis"} for node in nodes),
                "experiments": sum(node["node_type"] == "experiment" for node in nodes),
            },
        }
        files["GRAPH_MANIFEST.json"] = (
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        return files

    def render_overview(self, graph_hash: str) -> str:
        modules = [n for n in self.nodes.values() if n["node_type"] == "module"]
        lines = [
            "# Generated Stage 4 Semantic Module Overview",
            "",
            f"Graph hash: `{graph_hash}`",
            "",
            "> Generated from the canonical semantic graph. Do not edit manually.",
            "",
            "```mermaid",
            "flowchart LR",
        ]
        for node in sorted(modules, key=lambda x: x["node_id"]):
            lines.append(f'  {mermaid_id(node["node_id"])}["{mermaid_label(node["title"])}"]')
        for edge in sorted(self.edges.values(), key=lambda x: x["edge_id"]):
            if edge["relation"] in {"depends_on", "does_not_replace", "supersedes"} and edge[
                "source"
            ].startswith("module:") and edge["target"].startswith("module:"):
                lines.append(
                    f'  {mermaid_id(edge["source"])} -->|{edge["relation"]}| {mermaid_id(edge["target"])}'
                )
        lines.extend(["```", ""])
        return "\n".join(lines)

    def render_claim_experiment(self, graph_hash: str) -> str:
        relevant = {
            node_id: node
            for node_id, node in self.nodes.items()
            if node["node_type"] in {"claim", "hypothesis", "experiment"}
        }
        logic_edges = [
            edge
            for edge in self.edges.values()
            if edge["relation"] in {"tests", "external_validates", "supports", "contradicts"}
            and edge["source"] in relevant
            and edge["target"] in relevant
        ]
        used = {x for edge in logic_edges for x in (edge["source"], edge["target"])}
        lines = [
            "# Generated Claim--Experiment View",
            "",
            f"Graph hash: `{graph_hash}`",
            "",
            "> Generated from accepted edges only. Pending semantic suggestions remain in REVIEW_QUEUE.yaml.",
            "",
            "```mermaid",
            "flowchart LR",
        ]
        for node_id in sorted(used):
            node = relevant[node_id]
            lines.append(f'  {mermaid_id(node_id)}["{mermaid_label(node["title"])}"]')
        for edge in sorted(logic_edges, key=lambda x: x["edge_id"]):
            lines.append(
                f'  {mermaid_id(edge["source"])} -->|{edge["relation"]}| {mermaid_id(edge["target"])}'
            )
        lines.extend(["```", ""])
        return "\n".join(lines)

    def render_lineage(self, graph_hash: str) -> str:
        lineage = [e for e in self.edges.values() if e["relation"] == "supersedes"]
        used = {x for edge in lineage for x in (edge["source"], edge["target"])}
        lines = [
            "# Generated Claim and Module Lineage View",
            "",
            f"Graph hash: `{graph_hash}`",
            "",
            "> Generated from accepted supersedes edges. Do not edit manually.",
            "",
            "```mermaid",
            "flowchart LR",
        ]
        for node_id in sorted(used):
            node = self.nodes[node_id]
            lines.append(f'  {mermaid_id(node_id)}["{mermaid_label(node["title"])}"]')
        for edge in sorted(lineage, key=lambda x: x["edge_id"]):
            lines.append(
                f'  {mermaid_id(edge["source"])} -->|supersedes| {mermaid_id(edge["target"])}'
            )
        lines.extend(["```", ""])
        return "\n".join(lines)

    def render_dot(self, graph_hash: str) -> str:
        lines = [
            "digraph stage4_semantic_graph {",
            f'  graph [label="graph_hash={graph_hash}", labelloc="t"];',
            "  rankdir=LR;",
            "  node [shape=box];",
        ]
        for node in sorted(self.nodes.values(), key=lambda x: x["node_id"]):
            label = dot_escape(f"{node['node_type']}: {node['title']}")
            lines.append(f'  "{dot_escape(node["node_id"])}" [label="{label}"];')
        for edge in sorted(self.edges.values(), key=lambda x: x["edge_id"]):
            lines.append(
                f'  "{dot_escape(edge["source"])}" -> "{dot_escape(edge["target"])}" '
                f'[label="{dot_escape(edge["relation"])}"];'
            )
        lines.extend(["}", ""])
        return "\n".join(lines)


def resolve_config(args: argparse.Namespace) -> BuildConfig:
    repo_root = args.repo_root.resolve()
    profile = args.profile or repo_root / DEFAULT_PROFILE
    overrides = args.overrides or repo_root / DEFAULT_OVERRIDES
    output = args.output_dir or repo_root / DEFAULT_OUTPUT
    return BuildConfig(repo_root, profile.resolve(), overrides.resolve(), output.resolve())


def write_files(output_dir: Path, files: dict[str, bytes]) -> None:
    for relative, data in files.items():
        path = output_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def check_files(output_dir: Path, files: dict[str, bytes]) -> list[str]:
    problems: list[str] = []
    expected = set(files)
    for relative, data in files.items():
        path = output_dir / relative
        if not path.is_file():
            problems.append(f"missing generated file: {relative}")
        elif path.read_bytes() != data:
            problems.append(f"stale generated file: {relative}")
    for relative in GENERATED_FILES:
        path = output_dir / relative
        if path.exists() and relative not in expected:
            problems.append(f"unexpected generated file: {relative}")
    return problems


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--overrides", type=Path)
    parser.add_argument("--output-dir", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = resolve_config(args)
    try:
        files = GraphBuilder(config).build()
        if args.write:
            write_files(config.output_dir, files)
            status = "WRITTEN"
            problems: list[str] = []
        else:
            problems = check_files(config.output_dir, files)
            status = "PASS" if not problems else "FAIL"
    except SemanticGraphError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    report = {
        "status": status,
        "output_dir": str(config.output_dir),
        "file_count": len(files),
        "problems": problems,
    }
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Stage 4 dynamic semantic graph: {status} (files={len(files)})")
        for problem in problems:
            print(f"ERROR: {problem}", file=sys.stderr)
    return 0 if not problems else 2


if __name__ == "__main__":
    raise SystemExit(main())
