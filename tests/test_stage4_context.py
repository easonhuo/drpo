from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = ROOT / "scripts" / "build_stage4_context.py"
VALIDATOR_PATH = ROOT / "scripts" / "validate_stage4_context.py"
MINIMAL_ROOT = ROOT / "docs" / "handoff_shadow" / "stage4" / "minimal"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BUILDER = load_module("stage4_context_builder_test", BUILDER_PATH)
VALIDATOR = load_module("stage4_context_validator_test", VALIDATOR_PATH)


def dump_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=120),
        encoding="utf-8",
    )


def copy_current_project(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "docs" / "handoff_shadow" / "stage4").mkdir(parents=True)
    (repo / "experiments").mkdir(parents=True)
    shutil.copy2(ROOT / "docs" / "handoff.md", repo / "docs" / "handoff.md")
    shutil.copy2(
        ROOT / "experiments" / "registry.yaml",
        repo / "experiments" / "registry.yaml",
    )
    shutil.copytree(MINIMAL_ROOT, repo / "docs" / "handoff_shadow" / "stage4" / "minimal")
    return repo


def make_synthetic_project(tmp_path: Path) -> Path:
    repo = tmp_path / "synthetic"
    (repo / "docs").mkdir(parents=True)
    (repo / "experiments").mkdir(parents=True)
    minimal = repo / "docs" / "handoff_shadow" / "stage4" / "minimal"
    minimal.mkdir(parents=True)
    (repo / "docs" / "handoff.md").write_text(
        "# Synthetic\n\n# Core\ncore-v1\n\n# A\na-v1\n\n# B\nb-v1\n\n# End\n"
        "<!-- HANDOFF-DELTA-BLOCK:after_heading:e1-initial:START -->\n"
        "E1 initial result.\n"
        "<!-- HANDOFF-DELTA-BLOCK:after_heading:e1-initial:END -->\n",
        encoding="utf-8",
    )
    dump_yaml(
        repo / "experiments" / "registry.yaml",
        {
            "schema_version": 1,
            "experiments": [
                {"experiment_id": "E1", "name": "A experiment", "status": "not_run"},
                {"experiment_id": "E2", "name": "B experiment", "status": "not_run"},
            ],
            "development_experiment_registrations": [],
        },
    )
    dump_yaml(
        minimal / "MODULES.yaml",
        {
            "schema_version": 1,
            "policy_id": "GOV-HANDOFF-INDEX-01",
            "authority": "non_authoritative_stage4_minimal_context_shadow",
            "research_master": "docs/handoff.md",
            "registry": "experiments/registry.yaml",
            "module_granularity": "independent_research_responsibility",
            "structure_change_policy": "suggestion_only_human_approval_required",
            "default_split_suggestion_chars": 1000,
            "modules": [
                {
                    "module_id": "core",
                    "title": "Core",
                    "responsibility": "Shared rules.",
                    "sources": [
                        {
                            "kind": "markdown_range",
                            "path": "docs/handoff.md",
                            "start": "# Core",
                            "end": "# A",
                        }
                    ],
                },
                {
                    "module_id": "a",
                    "title": "A",
                    "responsibility": "A responsibility.",
                    "sources": [
                        {
                            "kind": "markdown_range",
                            "path": "docs/handoff.md",
                            "start": "# A",
                            "end": "# B",
                        },
                        {
                            "kind": "marker_blocks_matching",
                            "path": "docs/handoff.md",
                            "match_any": ["E1"],
                        },
                        {
                            "kind": "registry_entries",
                            "path": "experiments/registry.yaml",
                            "collection": "experiments",
                            "experiment_ids": ["E1"],
                        },
                    ],
                },
                {
                    "module_id": "b",
                    "title": "B",
                    "responsibility": "B responsibility.",
                    "sources": [
                        {
                            "kind": "markdown_range",
                            "path": "docs/handoff.md",
                            "start": "# B",
                            "end": "# End",
                        },
                        {
                            "kind": "registry_entries",
                            "path": "experiments/registry.yaml",
                            "collection": "experiments",
                            "experiment_ids": ["E2"],
                        },
                    ],
                },
            ],
        },
    )
    dump_yaml(
        minimal / "DEPENDENCIES.yaml",
        {
            "schema_version": 1,
            "policy_id": "GOV-HANDOFF-INDEX-01",
            "relation": "depends_on",
            "depends_on": {"core": [], "a": ["core"], "b": ["core"]},
            "acceptance_targets": [
                {
                    "target": "a",
                    "must_include": ["core", "a"],
                    "must_exclude": ["b"],
                }
            ],
        },
    )
    return repo


def plan(repo: Path):
    return BUILDER.build_plan(repo)


def output_root(repo: Path) -> Path:
    return repo / BUILDER.DEFAULT_OUTPUT


def test_current_repository_generated_outputs_are_exact() -> None:
    current = plan(ROOT)
    assert BUILDER.check_generated(ROOT / BUILDER.DEFAULT_OUTPUT, current) == []
    report = VALIDATOR.validate(ROOT)
    assert report["status"] == "PASS"
    assert report["manual_handoff_remains_authoritative"] is True
    assert report["authority_cutover_allowed"] is False


@pytest.mark.parametrize(
    ("target", "required", "excluded"),
    [
        (
            "continuous_e4_taper",
            {"continuous_e4_taper", "continuous_mechanism_e1_e3", "terminal_audit"},
            {"hopper_e7", "countdown_e8"},
        ),
        (
            "hopper_e7",
            {"hopper_e7", "continuous_mechanism_e1_e3", "execution_status_gates"},
            {"countdown_e8", "categorical_e6_generalization"},
        ),
        (
            "countdown_e8",
            {"countdown_e8", "categorical_e5_mechanism", "categorical_e6_generalization"},
            {"hopper_e7", "continuous_e4_taper"},
        ),
    ],
)
def test_real_target_context_closures(
    target: str, required: set[str], excluded: set[str]
) -> None:
    current = plan(ROOT)
    closure = set(
        BUILDER.dependency_closure(target, current.dependencies, current.module_order)
    )
    assert required <= closure
    assert not (excluded & closure)
    payload = BUILDER.context_pack_bytes(current, target).decode("utf-8")
    assert f"Target module: `{target}`" in payload
    for module_id in required:
        assert f"## Module `{module_id}`" in payload
    for module_id in excluded:
        assert f"## Module `{module_id}`" not in payload


def test_repeated_build_reuses_all_unchanged_modules(tmp_path: Path) -> None:
    repo = make_synthetic_project(tmp_path)
    first_plan = plan(repo)
    first = BUILDER.write_generated(output_root(repo), first_plan)
    before = {
        path.relative_to(output_root(repo)): path.stat().st_mtime_ns
        for path in output_root(repo).rglob("*")
        if path.is_file()
    }
    second_plan = plan(repo)
    second = BUILDER.write_generated(output_root(repo), second_plan)
    after = {
        path.relative_to(output_root(repo)): path.stat().st_mtime_ns
        for path in output_root(repo).rglob("*")
        if path.is_file()
    }
    assert sorted(first["refreshed_modules"]) == ["a", "b", "core"]
    assert second["refreshed_modules"] == []
    assert sorted(second["reused_modules"]) == ["a", "b", "core"]
    assert before == after


def test_only_dirty_module_snapshot_is_rewritten(tmp_path: Path) -> None:
    repo = make_synthetic_project(tmp_path)
    BUILDER.write_generated(output_root(repo), plan(repo))
    before = {
        module_id: (output_root(repo) / "modules" / f"{module_id}.md").read_bytes()
        for module_id in ("core", "a", "b")
    }
    handoff = repo / "docs" / "handoff.md"
    handoff.write_text(
        handoff.read_text(encoding="utf-8").replace("a-v1", "a-v2"),
        encoding="utf-8",
    )
    report = BUILDER.write_generated(output_root(repo), plan(repo))
    after = {
        module_id: (output_root(repo) / "modules" / f"{module_id}.md").read_bytes()
        for module_id in ("core", "a", "b")
    }
    assert report["refreshed_modules"] == ["a"]
    assert sorted(report["reused_modules"]) == ["b", "core"]
    assert before["a"] != after["a"]
    assert before["core"] == after["core"]
    assert before["b"] == after["b"]



def test_new_matching_delta_block_refreshes_existing_module_only(tmp_path: Path) -> None:
    repo = make_synthetic_project(tmp_path)
    BUILDER.write_generated(output_root(repo), plan(repo))
    handoff = repo / "docs" / "handoff.md"
    handoff.write_text(
        handoff.read_text(encoding="utf-8")
        + "\n<!-- HANDOFF-DELTA-BLOCK:section_end:e1-new-result:START -->\n"
        + "E1 added hyperparameter result.\n"
        + "<!-- HANDOFF-DELTA-BLOCK:section_end:e1-new-result:END -->\n",
        encoding="utf-8",
    )
    report = BUILDER.write_generated(output_root(repo), plan(repo))
    assert report["refreshed_modules"] == ["a"]
    assert sorted(report["reused_modules"]) == ["b", "core"]
    module_a = (output_root(repo) / "modules" / "a.md").read_text(encoding="utf-8")
    assert "E1 added hyperparameter result" in module_a
    assert "e1-new-result" in module_a


def test_registry_current_status_refreshes_only_owning_module(tmp_path: Path) -> None:
    repo = make_synthetic_project(tmp_path)
    BUILDER.write_generated(output_root(repo), plan(repo))
    registry_path = repo / "experiments" / "registry.yaml"
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    registry["experiments"][0]["status"] = "rejected"
    dump_yaml(registry_path, registry)
    report = BUILDER.write_generated(output_root(repo), plan(repo))
    assert report["refreshed_modules"] == ["a"]
    assert "status: rejected" in (
        output_root(repo) / "modules" / "a.md"
    ).read_text(encoding="utf-8")
    assert "status: rejected" not in (
        output_root(repo) / "modules" / "b.md"
    ).read_text(encoding="utf-8")


def test_new_module_requires_only_yaml_changes(tmp_path: Path) -> None:
    repo = make_synthetic_project(tmp_path)
    modules_path = repo / BUILDER.DEFAULT_MODULES
    dependencies_path = repo / BUILDER.DEFAULT_DEPENDENCIES
    modules = yaml.safe_load(modules_path.read_text(encoding="utf-8"))
    modules["modules"].append(
        {
            "module_id": "new_module",
            "title": "New module",
            "responsibility": "New independent responsibility.",
            "sources": [
                {
                    "kind": "markdown_range",
                    "path": "docs/handoff.md",
                    "start": "# Synthetic",
                    "end": "# Core",
                }
            ],
        }
    )
    dump_yaml(modules_path, modules)
    dependencies = yaml.safe_load(dependencies_path.read_text(encoding="utf-8"))
    dependencies["depends_on"]["new_module"] = ["core"]
    dump_yaml(dependencies_path, dependencies)
    updated = plan(repo)
    assert "new_module" in updated.module_order
    assert BUILDER.dependency_closure(
        "new_module", updated.dependencies, updated.module_order
    ) == ("core", "new_module")


def test_dependency_change_updates_graph_and_context(tmp_path: Path) -> None:
    repo = make_synthetic_project(tmp_path)
    initial = plan(repo)
    initial_graph = initial.outputs[Path("DEPENDENCY_GRAPH.dot")]
    initial_context = BUILDER.context_pack_bytes(initial, "b")
    dependencies_path = repo / BUILDER.DEFAULT_DEPENDENCIES
    dependencies = yaml.safe_load(dependencies_path.read_text(encoding="utf-8"))
    dependencies["depends_on"]["b"] = ["core", "a"]
    dump_yaml(dependencies_path, dependencies)
    updated = plan(repo)
    assert updated.outputs[Path("DEPENDENCY_GRAPH.dot")] != initial_graph
    assert BUILDER.context_pack_bytes(updated, "b") != initial_context
    assert BUILDER.dependency_closure(
        "b", updated.dependencies, updated.module_order
    ) == ("core", "a", "b")


def test_dangling_dependency_fails_closed(tmp_path: Path) -> None:
    repo = make_synthetic_project(tmp_path)
    path = repo / BUILDER.DEFAULT_DEPENDENCIES
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["depends_on"]["a"] = ["missing"]
    dump_yaml(path, payload)
    with pytest.raises(BUILDER.ContextBuildError, match="unknown modules"):
        plan(repo)


def test_dependency_cycle_fails_closed(tmp_path: Path) -> None:
    repo = make_synthetic_project(tmp_path)
    path = repo / BUILDER.DEFAULT_DEPENDENCIES
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["depends_on"]["core"] = ["a"]
    dump_yaml(path, payload)
    with pytest.raises(BUILDER.ContextBuildError, match="dependency cycle"):
        plan(repo)


def test_duplicate_module_id_fails_closed(tmp_path: Path) -> None:
    repo = make_synthetic_project(tmp_path)
    path = repo / BUILDER.DEFAULT_MODULES
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["modules"].append(dict(payload["modules"][0]))
    dump_yaml(path, payload)
    with pytest.raises(BUILDER.ContextBuildError, match="duplicate module_id"):
        plan(repo)


def test_missing_or_ambiguous_range_anchor_fails_closed(tmp_path: Path) -> None:
    repo = make_synthetic_project(tmp_path)
    handoff = repo / "docs" / "handoff.md"
    handoff.write_text(
        handoff.read_text(encoding="utf-8") + "\n# A\nduplicate\n",
        encoding="utf-8",
    )
    with pytest.raises(BUILDER.ContextBuildError, match="found 2"):
        plan(repo)
    handoff.write_text(
        handoff.read_text(encoding="utf-8").replace("# Core", "# Renamed Core", 1),
        encoding="utf-8",
    )
    with pytest.raises(BUILDER.ContextBuildError, match="found 0"):
        plan(repo)


def test_unknown_registry_id_fails_closed(tmp_path: Path) -> None:
    repo = make_synthetic_project(tmp_path)
    path = repo / BUILDER.DEFAULT_MODULES
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    registry_source = next(
        source
        for source in payload["modules"][1]["sources"]
        if source["kind"] == "registry_entries"
    )
    registry_source["experiment_ids"] = ["E404"]
    dump_yaml(path, payload)
    with pytest.raises(BUILDER.ContextBuildError, match="missing registry IDs"):
        plan(repo)


def test_unmapped_registry_experiment_is_suggestion_only(tmp_path: Path) -> None:
    repo = make_synthetic_project(tmp_path)
    modules_before = (repo / BUILDER.DEFAULT_MODULES).read_bytes()
    dependencies_before = (repo / BUILDER.DEFAULT_DEPENDENCIES).read_bytes()
    registry_path = repo / "experiments" / "registry.yaml"
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    registry["experiments"].append(
        {"experiment_id": "E3", "name": "Unmapped", "status": "not_run"}
    )
    dump_yaml(registry_path, registry)
    updated = plan(repo)
    assert any(
        item["kind"] == "candidate_add_or_map_module" and item["object_id"] == "E3"
        for item in updated.suggestions
    )
    BUILDER.write_generated(output_root(repo), updated)
    assert modules_before == (repo / BUILDER.DEFAULT_MODULES).read_bytes()
    assert dependencies_before == (repo / BUILDER.DEFAULT_DEPENDENCIES).read_bytes()
    assert "E3" in (
        output_root(repo) / "STRUCTURE_SUGGESTIONS.md"
    ).read_text(encoding="utf-8")


def test_tampered_and_extra_generated_files_fail_check(tmp_path: Path) -> None:
    repo = make_synthetic_project(tmp_path)
    current = plan(repo)
    BUILDER.write_generated(output_root(repo), current)
    graph = output_root(repo) / "DEPENDENCY_GRAPH.md"
    graph.write_text(graph.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")
    (output_root(repo) / "OLD_VIEW.md").write_text("stale\n", encoding="utf-8")
    problems = BUILDER.check_generated(output_root(repo), current)
    assert "stale generated file: DEPENDENCY_GRAPH.md" in problems
    assert "unexpected generated file: OLD_VIEW.md" in problems


def test_build_removes_stale_generated_file(tmp_path: Path) -> None:
    repo = make_synthetic_project(tmp_path)
    BUILDER.write_generated(output_root(repo), plan(repo))
    stale = output_root(repo) / "OLD_VIEW.md"
    stale.write_text("stale\n", encoding="utf-8")
    report = BUILDER.write_generated(output_root(repo), plan(repo))
    assert report["removed_stale_files"] == ["OLD_VIEW.md"]
    assert not stale.exists()


def test_deleted_index_is_disposable_and_rebuilt(tmp_path: Path) -> None:
    repo = make_synthetic_project(tmp_path)
    current = plan(repo)
    BUILDER.write_generated(output_root(repo), current)
    expected_modules = {
        path.name: path.read_bytes()
        for path in (output_root(repo) / "modules").glob("*.md")
    }
    (output_root(repo) / "MODULE_INDEX.json").unlink()
    report = BUILDER.write_generated(output_root(repo), plan(repo))
    assert report["refreshed_modules"] == []
    assert report["refreshed_supporting_files"] == ["MODULE_INDEX.json"]
    assert expected_modules == {
        path.name: path.read_bytes()
        for path in (output_root(repo) / "modules").glob("*.md")
    }


def test_build_does_not_modify_authoritative_inputs(tmp_path: Path) -> None:
    repo = make_synthetic_project(tmp_path)
    handoff = repo / "docs" / "handoff.md"
    registry = repo / "experiments" / "registry.yaml"
    before = (handoff.read_bytes(), registry.read_bytes())
    BUILDER.write_generated(output_root(repo), plan(repo))
    assert before == (handoff.read_bytes(), registry.read_bytes())


def test_context_pack_is_byte_deterministic(tmp_path: Path) -> None:
    repo = make_synthetic_project(tmp_path)
    current = plan(repo)
    first = BUILDER.context_pack_bytes(current, "a")
    second = BUILDER.context_pack_bytes(plan(repo), "a")
    assert first == second
    assert first.count(b"## Module `core`") == 1
    assert first.count(b"## Module `a`") == 1
    assert b"## Module `b`" not in first


def test_acceptance_target_leak_fails_closed(tmp_path: Path) -> None:
    repo = make_synthetic_project(tmp_path)
    path = repo / BUILDER.DEFAULT_DEPENDENCIES
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["acceptance_targets"][0]["must_exclude"] = ["core"]
    dump_yaml(path, payload)
    with pytest.raises(BUILDER.ContextBuildError, match="acceptance target a failed"):
        plan(repo)


def test_module_index_records_current_authoritative_hashes(tmp_path: Path) -> None:
    repo = make_synthetic_project(tmp_path)
    current = plan(repo)
    BUILDER.write_generated(output_root(repo), current)
    index = json.loads((output_root(repo) / "MODULE_INDEX.json").read_text())
    assert index["authority"] == "non_authoritative_stage4_minimal_context_shadow"
    assert index["research_master"] == "docs/handoff.md"
    assert index["graph_hash"] == current.graph_hash
    assert index["input_hashes"]["docs/handoff.md"] == BUILDER.sha256_bytes(
        (repo / "docs/handoff.md").read_bytes()
    )
