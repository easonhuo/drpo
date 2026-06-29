#!/usr/bin/env python3
"""Fail-closed validation for the generated Stage 4 semantic graph."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = ROOT / "scripts" / "build_stage4_semantic_graph.py"
DEFAULT_OUTPUT = Path("docs/handoff_shadow/stage4/dynamic/generated")


class SemanticGraphValidationError(ValueError):
    """Raised when generated semantic graph state is invalid or stale."""


def load_builder():
    spec = importlib.util.spec_from_file_location("stage4_semantic_builder", BUILDER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {BUILDER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BUILDER = load_builder()


def load_yaml(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise SemanticGraphValidationError(f"cannot read {label}: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SemanticGraphValidationError(f"{label} must be a mapping")
    return payload


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def detect_cycle(edges: list[dict[str, Any]], relation: str) -> None:
    graph: dict[str, list[str]] = {}
    for edge in edges:
        if edge.get("relation") == relation:
            graph.setdefault(str(edge.get("source")), []).append(str(edge.get("target")))
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            raise SemanticGraphValidationError(f"{relation} graph contains a cycle")
        if node in visited:
            return
        visiting.add(node)
        for target in graph.get(node, []):
            visit(target)
        visiting.remove(node)
        visited.add(node)

    for node in sorted(graph):
        visit(node)


def validate(
    repo_root: Path,
    profile_path: Path | None = None,
    overrides_path: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    profile = (profile_path or repo_root / BUILDER.DEFAULT_PROFILE).resolve()
    overrides = (overrides_path or repo_root / BUILDER.DEFAULT_OVERRIDES).resolve()
    output = (output_dir or repo_root / DEFAULT_OUTPUT).resolve()
    config = BUILDER.BuildConfig(repo_root, profile, overrides, output)

    try:
        expected = BUILDER.GraphBuilder(config).build()
    except BUILDER.SemanticGraphError as exc:
        raise SemanticGraphValidationError(str(exc)) from exc
    stale = BUILDER.check_files(output, expected)
    if stale:
        raise SemanticGraphValidationError("; ".join(stale))

    manifest_path = output / "GRAPH_MANIFEST.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SemanticGraphValidationError(f"invalid graph manifest: {exc}") from exc
    nodes_payload = load_yaml(output / "NODES.yaml", "generated nodes")
    edges_payload = load_yaml(output / "EDGES.yaml", "generated edges")
    review_payload = load_yaml(output / "REVIEW_QUEUE.yaml", "generated review queue")
    nodes = nodes_payload.get("nodes")
    edges = edges_payload.get("edges")
    review = review_payload.get("review_queue")
    if not isinstance(nodes, list) or not isinstance(edges, list) or not isinstance(review, list):
        raise SemanticGraphValidationError("generated node, edge, and review payloads must be lists")

    kernel_path = BUILDER.safe_rel_path(
        repo_root,
        str(BUILDER.load_yaml(profile, "project profile").get("kernel")),
        "kernel",
    )
    kernel = BUILDER.load_yaml(kernel_path, "research semantic kernel")
    node_types = set(kernel.get("node_types", []))
    relation_types = set(kernel.get("relation_types", []))
    node_statuses = set(kernel.get("node_lifecycle_statuses", []))

    node_ids: set[str] = set()
    for node in nodes:
        if not isinstance(node, dict):
            raise SemanticGraphValidationError("node entries must be mappings")
        node_id = node.get("node_id")
        if not isinstance(node_id, str) or not node_id:
            raise SemanticGraphValidationError("node_id must be a non-empty string")
        if node_id in node_ids:
            raise SemanticGraphValidationError(f"duplicate node_id: {node_id}")
        node_ids.add(node_id)
        if node.get("node_type") not in node_types:
            raise SemanticGraphValidationError(f"unknown node type for {node_id}")
        if node.get("lifecycle_status") not in node_statuses:
            raise SemanticGraphValidationError(f"unknown node lifecycle for {node_id}")

    edge_ids: set[str] = set()
    triples: set[tuple[str, str, str]] = set()
    for edge in edges:
        if not isinstance(edge, dict):
            raise SemanticGraphValidationError("edge entries must be mappings")
        edge_id = edge.get("edge_id")
        if not isinstance(edge_id, str) or not edge_id:
            raise SemanticGraphValidationError("edge_id must be a non-empty string")
        if edge_id in edge_ids:
            raise SemanticGraphValidationError(f"duplicate edge_id: {edge_id}")
        edge_ids.add(edge_id)
        source = edge.get("source")
        target = edge.get("target")
        relation = edge.get("relation")
        triple = (str(source), str(relation), str(target))
        if triple in triples:
            raise SemanticGraphValidationError(f"duplicate accepted edge: {triple}")
        triples.add(triple)
        if source not in node_ids or target not in node_ids:
            raise SemanticGraphValidationError(f"dangling edge: {edge_id}")
        if relation not in relation_types:
            raise SemanticGraphValidationError(f"unknown relation type: {relation}")
        if edge.get("lifecycle_status") != "accepted":
            raise SemanticGraphValidationError("generated accepted edge file contains non-accepted edge")
        if edge.get("review_state") != "not_required":
            raise SemanticGraphValidationError("pending semantic edge leaked into accepted graph")
    detect_cycle(edges, "supersedes")

    review_ids: set[str] = set()
    for item in review:
        if not isinstance(item, dict) or item.get("state") != "pending":
            raise SemanticGraphValidationError("review queue entries must be pending mappings")
        review_id = item.get("review_id")
        if not isinstance(review_id, str) or review_id in review_ids:
            raise SemanticGraphValidationError("review queue contains missing or duplicate review_id")
        review_ids.add(review_id)

    graph_hashes = {
        nodes_payload.get("graph_hash"),
        edges_payload.get("graph_hash"),
        review_payload.get("graph_hash"),
        manifest.get("graph_hash"),
    }
    if len(graph_hashes) != 1 or None in graph_hashes:
        raise SemanticGraphValidationError("generated files disagree on graph_hash")
    graph_hash = next(iter(graph_hashes))
    for relative, expected_hash in manifest.get("generated_files", {}).items():
        path = output / relative
        if not path.is_file() or sha256(path) != expected_hash:
            raise SemanticGraphValidationError(f"manifest hash mismatch: {relative}")
        if relative.startswith("graph/") and graph_hash not in path.read_text(encoding="utf-8"):
            raise SemanticGraphValidationError(
                f"generated visualization does not carry current graph_hash: {relative}"
            )

    counts = manifest.get("counts", {})
    if counts.get("nodes") != len(nodes) or counts.get("edges") != len(edges):
        raise SemanticGraphValidationError("manifest object counts are stale")
    if counts.get("review_queue") != len(review):
        raise SemanticGraphValidationError("manifest review count is stale")

    return {
        "status": "PASS",
        "graph_hash": graph_hash,
        "nodes": len(nodes),
        "edges": len(edges),
        "review_queue": len(review),
        "manual_handoff_remains_authoritative": True,
        "authority_cutover_allowed": False,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--overrides", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = validate(args.repo_root, args.profile, args.overrides, args.output_dir)
    except SemanticGraphValidationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            "Stage 4 dynamic semantic graph: PASS "
            f"(nodes={report['nodes']}, edges={report['edges']}, "
            f"review={report['review_queue']}, graph={report['graph_hash'][:12]})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
