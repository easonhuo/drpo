from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/manuscript_publication_pipeline.py"


def load_module():
    spec = importlib.util.spec_from_file_location("publication_quality_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def paths_for(module, root: Path):
    return module.Paths(
        root=root,
        graph=root / "docs/manuscript/paper_graph.yaml",
        contract=root / "docs/manuscript/publication_quality_contract.yaml",
        output=root / "paper/publication_quality_v1",
    )


def prepare_root(tmp_path: Path) -> Path:
    for rel in (
        "docs/manuscript/paper_graph.yaml",
        "docs/manuscript/publication_quality_contract.yaml",
        "paper/overleaf/references.bib",
    ):
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, dst)
    for source in (ROOT / "paper/overleaf").rglob("*.tex"):
        dst = tmp_path / source.relative_to(ROOT)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dst)
    return tmp_path


def test_repository_front_four_quality_gate_passes() -> None:
    module = load_module()
    result = module.validate(paths_for(module, ROOT))
    assert result["status"] == "PASS"
    assert result["node_count"] == 16
    assert all(node["trace"] for node in result["nodes"])


def test_build_emits_auditable_generation_packets(tmp_path: Path) -> None:
    module = load_module()
    root = prepare_root(tmp_path)
    result = module.build(paths_for(module, root))
    assert result == {"status": "BUILT", "node_count": 16, "generated_count": 0}
    packet_file = root / "paper/publication_quality_v1/prose_packets.json"
    packets = json.loads(packet_file.read_text())["packets"]
    theory = next(row for row in packets if row["id"] == "THEORY-P03")
    assert len(theory["sentence_units"]) == 9
    assert "thm:equilibrium" in theory["theorem_or_equation_refs"]
    assert theory["appendix_bindings"] == ["app:proof-theorem-equilibrium"]


def test_gate_fails_when_required_sentence_role_is_removed(tmp_path: Path) -> None:
    module = load_module()
    root = prepare_root(tmp_path)
    graph_path = root / "docs/manuscript/paper_graph.yaml"
    graph = yaml.safe_load(graph_path.read_text())
    node = next(row for row in graph["nodes"] if row["id"] == "THEORY-P01")
    node["blueprint"]["sentence_plan"] = node["blueprint"]["sentence_plan"][:-1]
    graph_path.write_text(yaml.safe_dump(graph, sort_keys=False, width=1000))
    with pytest.raises(module.PublicationQualityError, match="missing required sentence roles"):
        module.validate(paths_for(module, root))


def test_gate_fails_for_unregistered_citation(tmp_path: Path) -> None:
    module = load_module()
    root = prepare_root(tmp_path)
    graph_path = root / "docs/manuscript/paper_graph.yaml"
    graph = yaml.safe_load(graph_path.read_text())
    node = next(row for row in graph["nodes"] if row["id"] == "RELATED-P01")
    node["blueprint"]["citation_refs"].append("missing_reference_key")
    graph_path.write_text(yaml.safe_dump(graph, sort_keys=False, width=1000))
    with pytest.raises(module.PublicationQualityError, match="missing bibliography keys"):
        module.validate(paths_for(module, root))


def test_gate_fails_for_missing_appendix_proof_label(tmp_path: Path) -> None:
    module = load_module()
    root = prepare_root(tmp_path)
    graph_path = root / "docs/manuscript/paper_graph.yaml"
    graph = yaml.safe_load(graph_path.read_text())
    node = next(row for row in graph["nodes"] if row["id"] == "THEORY-P02")
    node["blueprint"]["appendix_bindings"] = ["app:not-a-real-proof"]
    graph_path.write_text(yaml.safe_dump(graph, sort_keys=False, width=1000))
    with pytest.raises(module.PublicationQualityError, match="appendix binding has no label"):
        module.validate(paths_for(module, root))
