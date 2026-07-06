from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "paper_pipeline.py"

PAPER_PIPELINE_TEST_ENV = "DRPO_RUN_PAPER_PIPELINE_TESTS"
pytestmark = [
    pytest.mark.paper_pipeline,
    pytest.mark.skipif(
        os.environ.get(PAPER_PIPELINE_TEST_ENV) != "1",
        reason=(
            "paper pipeline tests are opt-in; set "
            f"{PAPER_PIPELINE_TEST_ENV}=1 to run them explicitly"
        ),
    ),
]


def load_module():
    spec = importlib.util.spec_from_file_location("drpo_paper_pipeline_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def prepare_root(tmp_path: Path) -> tuple[object, dict, Path]:
    module = load_module()
    graph_src = ROOT / "docs/manuscript/paper_graph.yaml"
    graph_dst = tmp_path / "docs/manuscript/paper_graph.yaml"
    graph_dst.parent.mkdir(parents=True)
    shutil.copy2(graph_src, graph_dst)
    profile_src = ROOT / "docs/manuscript/projects/drpo_profile.yaml"
    profile_dst = tmp_path / "docs/manuscript/projects/drpo_profile.yaml"
    profile_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(profile_src, profile_dst)
    (tmp_path / "paper/overleaf/sections").mkdir(parents=True)
    (tmp_path / "paper/overleaf/appendix").mkdir(parents=True)
    graph = module.read_yaml(graph_dst)
    graph["graph_path"] = "docs/manuscript/paper_graph.yaml"
    module.render(graph, tmp_path)
    return module, graph, tmp_path


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert text.count(old) == 1
    path.write_text(text.replace(old, new), encoding="utf-8")


def test_repository_graph_renders_and_validates() -> None:
    module = load_module()
    graph = module.read_yaml(ROOT / "docs/manuscript/paper_graph.yaml")
    graph["graph_path"] = "docs/manuscript/paper_graph.yaml"
    result = module.validate(graph, ROOT)
    assert result == {"status": "PASS", "nodes": 39, "sections": 17}


def test_outline_edit_propagates_to_blueprint_prose_and_tex(tmp_path: Path) -> None:
    module, graph, root = prepare_root(tmp_path)
    outline = root / graph["artifacts"]["outline"]
    old = "Off-policy policy optimization needs both attraction toward successful behavior and suppression of known failures, but the latter must remain dynamically controlled."
    new = "Approved test claim propagated from the outline."
    replace_once(outline, old, new)
    imported = module.sync(graph, root, generator_cmd=None, prefer=None)
    assert "INTRO-P01:outline" in imported
    persisted = yaml.safe_load((root / "docs/manuscript/paper_graph.yaml").read_text())
    node = next(row for row in persisted["nodes"] if row["id"] == "INTRO-P01")
    assert node["claim"] == new
    assert new in (root / persisted["artifacts"]["blueprint"]).read_text()
    assert new in (root / persisted["artifacts"]["prose"]).read_text()
    assert new in (root / "paper/overleaf/sections/01_introduction.tex").read_text()


def test_structured_prose_edit_propagates_upstream(tmp_path: Path) -> None:
    module, graph, root = prepare_root(tmp_path)
    prose = root / graph["artifacts"]["prose"]
    old = "Off-policy policy optimization needs both attraction toward successful behavior and suppression of known failures, but the latter must remain dynamically controlled."
    new = "Approved semantic change authored in the prose layer."
    replace_once(prose, old, new)
    imported = module.sync(graph, root, generator_cmd=None, prefer=None)
    assert "INTRO-P01:prose" in imported
    persisted = yaml.safe_load((root / "docs/manuscript/paper_graph.yaml").read_text())
    node = next(row for row in persisted["nodes"] if row["id"] == "INTRO-P01")
    assert node["claim"] == new
    assert new in (root / persisted["artifacts"]["outline"]).read_text()
    assert new in (root / persisted["artifacts"]["blueprint"]).read_text()


def test_conflicting_same_node_edits_fail_closed(tmp_path: Path) -> None:
    module, graph, root = prepare_root(tmp_path)
    outline = root / graph["artifacts"]["outline"]
    prose = root / graph["artifacts"]["prose"]
    phrase = "Off-policy policy optimization needs both attraction toward successful behavior and suppression of known failures, but the latter must remain dynamically controlled."
    replace_once(outline, phrase, "Outline conflict claim.")
    replace_once(prose, phrase, "Prose conflict claim.")
    with pytest.raises(module.PipelineError, match="conflicting edits"):
        module.sync(graph, root, generator_cmd=None, prefer=None)


def test_structured_delta_updates_all_layers(tmp_path: Path) -> None:
    module, graph, root = prepare_root(tmp_path)
    delta = root / "delta.yaml"
    delta.write_text(
        yaml.safe_dump(
            {
                "changes": [
                    {
                        "id": "INTRO-P03",
                        "claim": "Delta claim combining existing controls and matched isolation.",
                        "must_include": ["existing controls", "matched badness-distance isolation"],
                        "prose": "Delta prose body.",
                    }
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    applied = module.apply_delta(graph, root, delta, generator_cmd=None)
    assert applied == ["INTRO-P03"]
    persisted = yaml.safe_load((root / "docs/manuscript/paper_graph.yaml").read_text())
    assert "Delta claim" in (root / persisted["artifacts"]["outline"]).read_text()
    assert "Delta prose body" in (root / persisted["artifacts"]["prose"]).read_text()


def test_overleaf_package_excludes_legacy_and_build_files(tmp_path: Path) -> None:
    module, graph, root = prepare_root(tmp_path)
    overleaf = root / graph["artifacts"]["overleaf_root"]
    (overleaf / "legacy_source").mkdir()
    (overleaf / "legacy_source/old.tex").write_text("old")
    (overleaf / "build").mkdir()
    (overleaf / "build/temp.log").write_text("log")
    (overleaf / "main.pdf").write_bytes(b"%PDF-test")
    output = root / "release.zip"
    module.package_overleaf(graph, root, output)
    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
    assert "main.tex" in names
    assert "main.pdf" in names
    assert not any(name.startswith("legacy_source/") for name in names)
    assert not any(name.startswith("build/") for name in names)


def test_overleaf_package_is_byte_reproducible(tmp_path: Path) -> None:
    module, graph, root = prepare_root(tmp_path)
    overleaf = root / graph["artifacts"]["overleaf_root"]
    (overleaf / "main.pdf").write_bytes(b"%PDF-test")
    first = root / "release-first.zip"
    second = root / "release-second.zip"
    module.package_overleaf(graph, root, first)
    (overleaf / "main.tex").touch()
    module.package_overleaf(graph, root, second)
    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(first) as archive:
        assert {info.date_time for info in archive.infolist()} == {module.ZIP_EPOCH}


def test_rich_blueprint_projection_preserves_sentence_and_proof_contracts(tmp_path: Path) -> None:
    module, graph, root = prepare_root(tmp_path)
    blueprint = (root / graph["artifacts"]["blueprint"]).read_text()
    assert "**Sentence plan:**" in blueprint
    assert '"role": "aggregate_setup"' in blueprint
    assert "**Theorem or equation refs:**" in blueprint
    assert "thm:equilibrium" in blueprint
    assert "**Appendix bindings:**" in blueprint
    assert "app:proof-theorem-equilibrium" in blueprint


def test_tracked_overleaf_text_has_no_diff_check_whitespace() -> None:
    suffixes = {".tex", ".sty", ".bst", ".bib"}
    for path in sorted((ROOT / "paper/overleaf").rglob("*")):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        text = path.read_text(encoding="utf-8")
        assert text.endswith("\n") and not text.endswith("\n\n"), path
        assert all(line == line.rstrip(" \t") for line in text.splitlines()), path


def test_same_manuscript_and_p03_hard_constraints() -> None:
    graph = yaml.safe_load((ROOT / "docs/manuscript/paper_graph.yaml").read_text())
    text = json.dumps(graph, ensure_ascii=False).lower()
    intro = {node["id"]: node for node in graph["nodes"]}
    p03 = intro["INTRO-P03"]["prose"].lower()
    assert "existing methods" in p03
    assert "positive-only" in p03 and "clipping" in p03 and "quality filtering" in p03
    assert "isolate policy remoteness" in p03
    assert "negative-advantage magnitude" in p03 and "sample count" in p03
    assert "old paper + new sequel" not in text
    assert "original formulation and revised formulation" not in text
    assert graph["merge_policy"]["content_baseline"] == "user-approved v0.9-review"
