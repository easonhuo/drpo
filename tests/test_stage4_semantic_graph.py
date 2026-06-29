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
            "module_change_requires_version_increment": True,
            "module_split_merge_requires_supersedes_record": True,
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
            "override_version": 1,
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


def bump_override_version(payload: dict) -> None:
    payload["override_version"] = int(payload.get("override_version", 0)) + 1


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
    bump_override_version(override_payload)
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



def test_rejected_candidate_is_suppressed_and_audited(tmp_path: Path) -> None:
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
    queue = yaml.safe_load((output / "REVIEW_QUEUE.yaml").read_text(encoding="utf-8"))
    candidate = next(
        item
        for item in queue["review_queue"]
        if item["object_id"] == "experiment:E2" and item["kind"] == "claim_relation"
    )

    payload = yaml.safe_load(overrides.read_text(encoding="utf-8"))
    bump_override_version(payload)
    payload["rejected_candidates"].append(
        {
            "review_id": candidate["review_id"],
            "kind": candidate["kind"],
            "object_id": candidate["object_id"],
            "reason": candidate["reason"],
            "candidates": candidate["candidates"],
            "rationale": "Human rejected a claim relation until stronger evidence exists.",
            "decision_version": payload["override_version"],
        }
    )
    dump_yaml(overrides, payload)
    first = write(repo, profile, overrides, output)
    second = build(repo, profile, overrides, output)
    assert first == second
    report = VALIDATOR.validate(repo, profile, overrides, output)
    assert report["review_queue"] == 0
    assert report["rejected_candidates"] == 1
    queue = yaml.safe_load((output / "REVIEW_QUEUE.yaml").read_text(encoding="utf-8"))
    assert queue["rejected_candidates"][0]["review_id"] == candidate["review_id"]
    assert queue["rejected_candidates"][0]["match_state"] == "matched_current_candidate"


def test_rejected_candidate_signature_must_match_review_id(tmp_path: Path) -> None:
    repo, profile, overrides, output = make_synthetic_project(tmp_path)
    payload = yaml.safe_load(overrides.read_text(encoding="utf-8"))
    payload["rejected_candidates"].append(
        {
            "review_id": "review:not-the-right-id",
            "kind": "claim_relation",
            "object_id": "experiment:E1",
            "reason": "experiment has no accepted claim relation",
            "candidates": [],
            "rationale": "Invalid mutation.",
        }
    )
    dump_yaml(overrides, payload)
    with pytest.raises(BUILDER.SemanticGraphError, match="review_id does not match"):
        build(repo, profile, overrides, output)


def test_module_rename_override_updates_version_hash_and_view(tmp_path: Path) -> None:
    repo, profile, overrides, output = make_synthetic_project(tmp_path)
    write(repo, profile, overrides, output)
    first_manifest = read_manifest(output)
    first_view = (output / "graph" / "OVERVIEW.md").read_text(encoding="utf-8")
    payload = yaml.safe_load(overrides.read_text(encoding="utf-8"))
    bump_override_version(payload)
    payload["module_lifecycle_changes"].append(
        {
            "change_id": "rename-core-v2",
            "operation": "rename",
            "source_module_ids": ["core"],
            "target_module_ids": ["core"],
            "from_versions": {"core": 1},
            "to_versions": {"core": 2},
            "new_name": "Research core",
            "rationale": "Clarify the module scope without changing its stable ID.",
        }
    )
    dump_yaml(overrides, payload)
    write(repo, profile, overrides, output)
    report = VALIDATOR.validate(repo, profile, overrides, output)
    nodes = {
        node["node_id"]: node
        for node in yaml.safe_load((output / "NODES.yaml").read_text(encoding="utf-8"))["nodes"]
    }
    assert nodes["module:core"]["title"] == "Research core"
    assert nodes["module:core"]["attributes"]["version"] == 2
    assert nodes["module:core"]["attributes"]["lifecycle_changes"][0]["change_id"] == "rename-core-v2"
    second_manifest = read_manifest(output)
    second_view = (output / "graph" / "OVERVIEW.md").read_text(encoding="utf-8")
    assert second_manifest["graph_hash"] != first_manifest["graph_hash"]
    assert second_view != first_view
    assert report["status"] == "PASS"


def test_module_split_override_preserves_old_identity_and_supersedes_edges(tmp_path: Path) -> None:
    repo, profile_path, overrides, output = make_synthetic_project(tmp_path)
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    profile["modules"].extend(
        [
            {
                "module_id": "old_topic",
                "version": 1,
                "lifecycle_status": "active",
                "name": "Old topic",
                "purpose": "Broad historical topic",
                "default_dependencies": ["core"],
            },
            {
                "module_id": "topic_a",
                "version": 1,
                "lifecycle_status": "proposed",
                "name": "Topic A",
                "purpose": "First narrower topic",
                "default_dependencies": ["core"],
            },
            {
                "module_id": "topic_b",
                "version": 1,
                "lifecycle_status": "proposed",
                "name": "Topic B",
                "purpose": "Second narrower topic",
                "default_dependencies": ["core"],
            },
        ]
    )
    dump_yaml(profile_path, profile)
    write(repo, profile_path, overrides, output)
    payload = yaml.safe_load(overrides.read_text(encoding="utf-8"))
    bump_override_version(payload)
    payload["module_lifecycle_changes"].append(
        {
            "change_id": "split-old-topic-v2",
            "operation": "split",
            "source_module_ids": ["old_topic"],
            "target_module_ids": ["topic_a", "topic_b"],
            "from_versions": {"old_topic": 1, "topic_a": 1, "topic_b": 1},
            "to_versions": {"old_topic": 2, "topic_a": 2, "topic_b": 2},
            "rationale": "The broad topic accumulated two independently useful substructures.",
        }
    )
    dump_yaml(overrides, payload)
    write(repo, profile_path, overrides, output)
    VALIDATOR.validate(repo, profile_path, overrides, output)
    nodes = {
        node["node_id"]: node
        for node in yaml.safe_load((output / "NODES.yaml").read_text(encoding="utf-8"))["nodes"]
    }
    edges = yaml.safe_load((output / "EDGES.yaml").read_text(encoding="utf-8"))["edges"]
    assert nodes["module:old_topic"]["lifecycle_status"] == "superseded"
    assert nodes["module:old_topic"]["attributes"]["superseded_by"] == ["topic_a", "topic_b"]
    for target in ("topic_a", "topic_b"):
        assert nodes[f"module:{target}"]["lifecycle_status"] == "active"
        assert nodes[f"module:{target}"]["attributes"]["version"] == 2
        assert any(
            edge["source"] == f"module:{target}"
            and edge["relation"] == "supersedes"
            and edge["target"] == "module:old_topic"
            for edge in edges
        )


def test_module_merge_override_preserves_all_source_identities(tmp_path: Path) -> None:
    repo, profile_path, overrides, output = make_synthetic_project(tmp_path)
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    profile["modules"].extend(
        [
            {
                "module_id": "topic_a",
                "version": 1,
                "lifecycle_status": "active",
                "name": "Topic A",
                "purpose": "First source",
                "default_dependencies": ["core"],
            },
            {
                "module_id": "topic_b",
                "version": 1,
                "lifecycle_status": "active",
                "name": "Topic B",
                "purpose": "Second source",
                "default_dependencies": ["core"],
            },
            {
                "module_id": "combined_topic",
                "version": 1,
                "lifecycle_status": "proposed",
                "name": "Combined topic",
                "purpose": "Merged target",
                "default_dependencies": ["core"],
            },
        ]
    )
    dump_yaml(profile_path, profile)
    write(repo, profile_path, overrides, output)
    payload = yaml.safe_load(overrides.read_text(encoding="utf-8"))
    bump_override_version(payload)
    payload["module_lifecycle_changes"].append(
        {
            "change_id": "merge-topics-v2",
            "operation": "merge",
            "source_module_ids": ["topic_a", "topic_b"],
            "target_module_ids": ["combined_topic"],
            "from_versions": {"topic_a": 1, "topic_b": 1, "combined_topic": 1},
            "to_versions": {"topic_a": 2, "topic_b": 2, "combined_topic": 2},
            "rationale": "The two topics no longer have independent semantic roles.",
        }
    )
    dump_yaml(overrides, payload)
    write(repo, profile_path, overrides, output)
    nodes = {
        node["node_id"]: node
        for node in yaml.safe_load((output / "NODES.yaml").read_text(encoding="utf-8"))["nodes"]
    }
    edges = yaml.safe_load((output / "EDGES.yaml").read_text(encoding="utf-8"))["edges"]
    assert nodes["module:topic_a"]["lifecycle_status"] == "superseded"
    assert nodes["module:topic_b"]["lifecycle_status"] == "superseded"
    assert nodes["module:combined_topic"]["lifecycle_status"] == "active"
    assert {
        edge["target"]
        for edge in edges
        if edge["source"] == "module:combined_topic" and edge["relation"] == "supersedes"
    } >= {"module:topic_a", "module:topic_b"}


def test_override_semantics_change_requires_override_version_increment(tmp_path: Path) -> None:
    repo, profile, overrides, output = make_synthetic_project(tmp_path)
    write(repo, profile, overrides, output)
    payload = yaml.safe_load(overrides.read_text(encoding="utf-8"))
    payload["accepted_edges"].append(
        {
            "source": "experiment:E1",
            "relation": "supports",
            "target": "claim:H1",
            "rationale": "Version policy mutation.",
        }
    )
    dump_yaml(overrides, payload)
    with pytest.raises(BUILDER.SemanticGraphError, match="override_version increment"):
        build(repo, profile, overrides, output)


def test_profile_and_module_semantics_require_version_increments(tmp_path: Path) -> None:
    repo, profile_path, overrides, output = make_synthetic_project(tmp_path)
    write(repo, profile_path, overrides, output)
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    profile["modules"][0]["name"] = "Changed without version"
    dump_yaml(profile_path, profile)
    with pytest.raises(BUILDER.SemanticGraphError, match="profile_version increment"):
        build(repo, profile_path, overrides, output)

    profile["profile_version"] = 2
    dump_yaml(profile_path, profile)
    override_payload = yaml.safe_load(overrides.read_text(encoding="utf-8"))
    override_payload["profile_version"] = 2
    dump_yaml(overrides, override_payload)
    with pytest.raises(BUILDER.SemanticGraphError, match="module core semantics changed"):
        build(repo, profile_path, overrides, output)

    profile["modules"][0]["version"] = 2
    dump_yaml(profile_path, profile)
    write(repo, profile_path, overrides, output)
    assert VALIDATOR.validate(repo, profile_path, overrides, output)["status"] == "PASS"


def test_module_removal_is_rejected_even_with_profile_version_increment(tmp_path: Path) -> None:
    repo, profile_path, overrides, output = make_synthetic_project(tmp_path)
    write(repo, profile_path, overrides, output)
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    profile["profile_version"] = 2
    profile["modules"] = [m for m in profile["modules"] if m["module_id"] != "core"]
    dump_yaml(profile_path, profile)
    payload = yaml.safe_load(overrides.read_text(encoding="utf-8"))
    payload["profile_version"] = 2
    dump_yaml(overrides, payload)
    with pytest.raises(BUILDER.SemanticGraphError, match="may not be destructively removed"):
        build(repo, profile_path, overrides, output)


def test_split_without_version_increment_fails_closed(tmp_path: Path) -> None:
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
            },
            {
                "module_id": "b",
                "version": 1,
                "lifecycle_status": "proposed",
                "name": "B",
                "purpose": "B",
                "default_dependencies": [],
            },
            {
                "module_id": "c",
                "version": 1,
                "lifecycle_status": "proposed",
                "name": "C",
                "purpose": "C",
                "default_dependencies": [],
            },
        ]
    )
    dump_yaml(profile_path, profile)
    payload = yaml.safe_load(overrides.read_text(encoding="utf-8"))
    payload["module_lifecycle_changes"].append(
        {
            "change_id": "invalid-split",
            "operation": "split",
            "source_module_ids": ["a"],
            "target_module_ids": ["b", "c"],
            "from_versions": {"a": 1, "b": 1, "c": 1},
            "to_versions": {"a": 1, "b": 2, "c": 2},
            "rationale": "Invalid version mutation.",
        }
    )
    dump_yaml(overrides, payload)
    with pytest.raises(BUILDER.SemanticGraphError, match="increment every touched module"):
        build(repo, profile_path, overrides, output)


def test_engine_contains_no_drpo_experiment_or_module_hardcoding() -> None:
    source = BUILDER_PATH.read_text(encoding="utf-8")
    for forbidden in ("C-U1", "D-U1", "Hopper", "Countdown", "E18"):
        assert forbidden not in source


def test_validator_rejects_stale_semantic_fingerprint_version(tmp_path: Path) -> None:
    repo, profile, overrides, output = make_synthetic_project(tmp_path)
    write(repo, profile, overrides, output)
    manifest = read_manifest(output)
    manifest["semantic_fingerprint_version"] = 0
    (output / "GRAPH_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        VALIDATOR.SemanticGraphValidationError,
        match="stale generated file: GRAPH_MANIFEST.json",
    ):
        VALIDATOR.validate(repo, profile, overrides, output)
