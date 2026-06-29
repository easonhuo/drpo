from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = ROOT / "scripts" / "build_stage4_semantic_graph.py"
VALIDATOR_PATH = ROOT / "scripts" / "validate_stage4_semantic_graph.py"
DYNAMIC_ROOT = ROOT / "docs" / "handoff_shadow" / "stage4" / "dynamic"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BUILDER = load_module("stage4_semantic_builder_test", BUILDER_PATH)
VALIDATOR = load_module("stage4_semantic_validator_test", VALIDATOR_PATH)


def dump_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=120),
        encoding="utf-8",
    )


def make_synthetic_project(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    repo = tmp_path / "research_project"
    (repo / "docs").mkdir(parents=True)
    (repo / "experiments").mkdir(parents=True)
    (repo / "semantic").mkdir(parents=True)
    shutil.copy2(
        DYNAMIC_ROOT / "kernel" / "RESEARCH_SEMANTIC_KERNEL.yaml",
        repo / "semantic" / "kernel.yaml",
    )
    (repo / "docs" / "handoff.md").write_text(
        "# Minimal research project\n\n## Q1\n\nDoes the intervention improve the outcome?\n",
        encoding="utf-8",
    )
    dump_yaml(
        repo / "experiments" / "registry.yaml",
        {
            "schema_version": 1,
            "experiments": [
                {
                    "id": "E1",
                    "name": "Initial experiment",
                    "environment": "toy",
                    "role": "controlled_test",
                    "status": "not_run",
                }
            ],
        },
    )
    dump_yaml(
        repo / "semantic" / "headings.yaml",
        {
            "headings": [
                {
                    "heading_id": "H0001",
                    "title": "Minimal research project",
                    "occurrence": 1,
                    "module_ids": ["core"],
                },
                {
                    "heading_id": "H0002",
                    "title": "Q1",
                    "occurrence": 1,
                    "module_ids": ["core"],
                },
            ]
        },
    )
    dump_yaml(
        repo / "semantic" / "claims.yaml",
        {
            "claims": [
                {
                    "claim_id": "H1",
                    "node_type": "hypothesis",
                    "status": "current_supported",
                    "statement_summary": "The intervention may improve the outcome.",
                    "module_ids": ["core"],
                    "source_anchor": {"file": "docs/handoff.md", "heading_id": "H0002"},
                    "lineage": {"supersedes": [], "superseded_by": []},
                }
            ]
        },
    )
    dump_yaml(
        repo / "semantic" / "experiments.yaml",
        {
            "experiments": [
                {
                    "experiment_id": "E1",
                    "role": "controlled_test",
                    "registry_status": "not_run",
                    "module_ids": ["experiments"],
                    "claim_ids": ["H1"],
                }
            ]
        },
    )
    profile = {
        "schema_version": 1,
        "profile_id": "minimal-project",
        "profile_version": 1,
        "authority": "non_authoritative_test_profile",
        "kernel": "semantic/kernel.yaml",
        "sources": {
            "handoff": "docs/handoff.md",
            "registry": "experiments/registry.yaml",
            "bootstrap_headings": "semantic/headings.yaml",
            "bootstrap_claims": "semantic/claims.yaml",
            "bootstrap_experiments": "semantic/experiments.yaml",
        },
        "project": {"node_id": "project:minimal", "title": "Minimal project"},
        "modules": [
            {
                "module_id": "core",
                "version": 1,
                "lifecycle_status": "active",
                "name": "Core",
                "purpose": "Core question and hypothesis",
                "default_dependencies": [],
            },
            {
                "module_id": "experiments",
                "version": 1,
                "lifecycle_status": "active",
                "name": "Experiments",
                "purpose": "Project experiments",
                "default_dependencies": ["core"],
            },
        ],
        "module_inference_rules": [
            {
                "rule_id": "all-experiments",
                "field": "id",
                "regex": ".+",
                "module_id": "experiments",
            }
        ],
        "explicit_edges": [],
        "policies": {
            "unknown_experiment_module_assignment": "review_queue",
            "missing_experiment_claim_relation": "review_queue",
        },
    }
    profile_path = repo / "semantic" / "profile.yaml"
    dump_yaml(profile_path, profile)
    overrides_path = repo / "semantic" / "overrides.yaml"
    dump_yaml(
        overrides_path,
        {
            "schema_version": 1,
            "profile_id": "minimal-project",
            "profile_version": 1,
            "authority": "test_overrides",
            "module_assignments": [],
            "accepted_edges": [],
            "rejected_candidates": [],
            "module_lifecycle_changes": [],
        },
    )
    return repo, profile_path, overrides_path, repo / "generated"


def build(repo: Path, profile: Path, overrides: Path, output: Path) -> dict[str, bytes]:
    config = BUILDER.BuildConfig(repo, profile, overrides, output)
    return BUILDER.GraphBuilder(config).build()


def write(repo: Path, profile: Path, overrides: Path, output: Path) -> dict[str, bytes]:
    files = build(repo, profile, overrides, output)
    BUILDER.write_files(output, files)
    return files


def read_manifest(output: Path) -> dict:
    return json.loads((output / "GRAPH_MANIFEST.json").read_text(encoding="utf-8"))


def test_current_dynamic_graph_is_valid_and_has_no_pending_review() -> None:
    report = VALIDATOR.validate(ROOT)
    assert report["status"] == "PASS"
    assert report["nodes"] > 400
    assert report["edges"] > 1000
    assert report["review_queue"] == 0
    assert report["manual_handoff_remains_authoritative"] is True
    assert report["authority_cutover_allowed"] is False


def test_current_build_is_byte_deterministic() -> None:
    config = BUILDER.BuildConfig(
        ROOT,
        ROOT / BUILDER.DEFAULT_PROFILE,
        ROOT / BUILDER.DEFAULT_OVERRIDES,
        ROOT / BUILDER.DEFAULT_OUTPUT,
    )
    first = BUILDER.GraphBuilder(config).build()
    second = BUILDER.GraphBuilder(config).build()
    assert first == second
    assert BUILDER.check_files(config.output_dir, first) == []


def test_new_project_starts_with_only_q1_h1_e1(tmp_path: Path) -> None:
    repo, profile, overrides, output = make_synthetic_project(tmp_path)
    write(repo, profile, overrides, output)
    report = VALIDATOR.validate(repo, profile, overrides, output)
    manifest = read_manifest(output)
    assert report["status"] == "PASS"
    assert manifest["counts"]["experiments"] == 1
    assert manifest["counts"]["review_queue"] == 0
    nodes = yaml.safe_load((output / "NODES.yaml").read_text(encoding="utf-8"))["nodes"]
    assert {node["node_id"] for node in nodes} >= {
        "project:minimal",
        "claim:H1",
        "experiment:E1",
    }


def test_new_experiment_is_discovered_without_python_change_and_enters_review(
    tmp_path: Path,
) -> None:
    repo, profile, overrides, output = make_synthetic_project(tmp_path)
    registry_path = repo / "experiments" / "registry.yaml"
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    registry["experiments"].append(
        {
            "id": "E2",
            "name": "Follow-up experiment",
            "environment": "toy",
            "role": "controlled_test",
            "status": "not_run",
        }
    )
    dump_yaml(registry_path, registry)
    write(repo, profile, overrides, output)
    nodes = yaml.safe_load((output / "NODES.yaml").read_text(encoding="utf-8"))["nodes"]
    review = yaml.safe_load((output / "REVIEW_QUEUE.yaml").read_text(encoding="utf-8"))[
        "review_queue"
    ]
    assert "experiment:E2" in {node["node_id"] for node in nodes}
    assert any(item["object_id"] == "experiment:E2" for item in review)
    assert all(item["state"] == "pending" for item in review)


def test_human_override_resolves_new_experiment_claim_relation(tmp_path: Path) -> None:
    repo, profile, overrides, output = make_synthetic_project(tmp_path)
    registry_path = repo / "experiments" / "registry.yaml"
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    registry["experiments"].append(
        {
            "id": "E2",
            "name": "Follow-up experiment",
            "environment": "toy",
            "role": "controlled_test",
            "status": "not_run",
        }
    )
    dump_yaml(registry_path, registry)
    override_payload = yaml.safe_load(overrides.read_text(encoding="utf-8"))
    override_payload["accepted_edges"].append(
        {
            "source": "experiment:E2",
            "relation": "tests",
            "target": "claim:H1",
            "rationale": "Human approved E2 as a direct test of H1.",
        }
    )
    dump_yaml(overrides, override_payload)
    write(repo, profile, overrides, output)
    report = VALIDATOR.validate(repo, profile, overrides, output)
    assert report["review_queue"] == 0
    edges = yaml.safe_load((output / "EDGES.yaml").read_text(encoding="utf-8"))["edges"]
    assert any(
        edge["source"] == "experiment:E2"
        and edge["relation"] == "tests"
        and edge["target"] == "claim:H1"
        for edge in edges
    )


def test_module_supersedes_lifecycle_is_preserved(tmp_path: Path) -> None:
    repo, profile_path, overrides, output = make_synthetic_project(tmp_path)
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    profile["modules"].extend(
        [
            {
                "module_id": "old_topic",
                "version": 1,
                "lifecycle_status": "superseded",
                "name": "Old topic",
                "purpose": "Historical module",
                "default_dependencies": ["core"],
            },
            {
                "module_id": "new_topic",
                "version": 1,
                "lifecycle_status": "active",
                "name": "New topic",
                "purpose": "Replacement module",
                "default_dependencies": ["core"],
                "supersedes": ["old_topic"],
            },
        ]
    )
    dump_yaml(profile_path, profile)
    write(repo, profile_path, overrides, output)
    nodes = {
        node["node_id"]: node
        for node in yaml.safe_load((output / "NODES.yaml").read_text(encoding="utf-8"))[
            "nodes"
        ]
    }
    edges = yaml.safe_load((output / "EDGES.yaml").read_text(encoding="utf-8"))["edges"]
    assert nodes["module:old_topic"]["lifecycle_status"] == "superseded"
    assert any(
        edge["source"] == "module:new_topic"
        and edge["relation"] == "supersedes"
        and edge["target"] == "module:old_topic"
        for edge in edges
    )


def test_accepted_edge_change_updates_graph_hash_and_visualization(tmp_path: Path) -> None:
    repo, profile, overrides, output = make_synthetic_project(tmp_path)
    write(repo, profile, overrides, output)
    first_hash = read_manifest(output)["graph_hash"]
    first_view = (output / "graph" / "CLAIM_EXPERIMENT.md").read_text(encoding="utf-8")
    override_payload = yaml.safe_load(overrides.read_text(encoding="utf-8"))
    override_payload["accepted_edges"].append(
        {
            "source": "experiment:E1",
            "relation": "supports",
            "target": "claim:H1",
            "rationale": "Additional approved support relation.",
        }
    )
    dump_yaml(overrides, override_payload)
    write(repo, profile, overrides, output)
    second_hash = read_manifest(output)["graph_hash"]
    second_view = (output / "graph" / "CLAIM_EXPERIMENT.md").read_text(encoding="utf-8")
    assert second_hash != first_hash
    assert second_view != first_view
    assert "supports" in second_view
    assert second_hash in second_view


def test_manual_visualization_edit_is_rejected(tmp_path: Path) -> None:
    repo, profile, overrides, output = make_synthetic_project(tmp_path)
    write(repo, profile, overrides, output)
    view = output / "graph" / "OVERVIEW.md"
    view.write_text(view.read_text(encoding="utf-8") + "manual edit\n", encoding="utf-8")
    with pytest.raises(VALIDATOR.SemanticGraphValidationError, match="stale generated file"):
        VALIDATOR.validate(repo, profile, overrides, output)


def test_source_drift_is_rejected_until_regenerated(tmp_path: Path) -> None:
    repo, profile, overrides, output = make_synthetic_project(tmp_path)
    write(repo, profile, overrides, output)
    handoff = repo / "docs" / "handoff.md"
    handoff.write_text(handoff.read_text(encoding="utf-8") + "\n## New section\n", encoding="utf-8")
    with pytest.raises(VALIDATOR.SemanticGraphValidationError, match="stale generated file"):
        VALIDATOR.validate(repo, profile, overrides, output)


def test_dangling_and_unknown_relations_fail_closed(tmp_path: Path) -> None:
    repo, profile, overrides, output = make_synthetic_project(tmp_path)
    payload = yaml.safe_load(overrides.read_text(encoding="utf-8"))
    payload["accepted_edges"] = [
        {
            "source": "experiment:E1",
            "relation": "related_to",
            "target": "claim:missing",
            "rationale": "invalid mutation",
        }
    ]
    dump_yaml(overrides, payload)
    with pytest.raises(BUILDER.SemanticGraphError, match="unknown relation"):
        build(repo, profile, overrides, output)


def test_supersedes_cycle_fails_closed(tmp_path: Path) -> None:
    repo, profile_path, overrides, output = make_synthetic_project(tmp_path)
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    profile["modules"].extend(
        [
            {
                "module_id": "a",
                "version": 1,
                "lifecycle_status": "active",
                "name": "A",
                "purpose": "A",
                "default_dependencies": [],
                "supersedes": ["b"],
            },
            {
                "module_id": "b",
                "version": 1,
                "lifecycle_status": "active",
                "name": "B",
                "purpose": "B",
                "default_dependencies": [],
                "supersedes": ["a"],
            },
        ]
    )
    dump_yaml(profile_path, profile)
    with pytest.raises(BUILDER.SemanticGraphError, match="supersedes graph contains a cycle"):
        build(repo, profile_path, overrides, output)


def test_duplicate_module_id_fails_closed(tmp_path: Path) -> None:
    repo, profile_path, overrides, output = make_synthetic_project(tmp_path)
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    profile["modules"].append(dict(profile["modules"][0]))
    dump_yaml(profile_path, profile)
    with pytest.raises(BUILDER.SemanticGraphError, match="duplicate module_id"):
        build(repo, profile_path, overrides, output)


def test_engine_contains_no_drpo_experiment_or_module_hardcoding() -> None:
    source = BUILDER_PATH.read_text(encoding="utf-8")
    for forbidden in ("C-U1", "D-U1", "Hopper", "Countdown", "E18"):
        assert forbidden not in source
