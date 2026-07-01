from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "paper_pipeline_core.py"


def load_module():
    spec = importlib.util.spec_from_file_location("drpo_paper_pipeline_core_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def prepare_minimal_repo(tmp_path: Path):
    module = load_module()
    relative_files = [
        "docs/manuscript/paper_spec_core.yaml",
        "docs/paper_rewrite_outline_v0_9_2.md",
        "experiments/registry.yaml",
        "outputs/cu1_e3_adam/ARTIFACT_INDEX.json",
        "outputs/cu1_e3_adam/RESULT_SUMMARY.md",
        "outputs/cu1_e3_adam/TERMINAL_AUDIT.md",
        "outputs/cu1_e3_adam/fixed_variance_aggregate.csv",
        "outputs/cu1_e3_adam/learnable_variance_aggregate.csv",
    ]
    for relative in relative_files:
        source = ROOT / relative
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    spec_path = tmp_path / "docs/manuscript/paper_spec_core.yaml"
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    output = tmp_path / spec["output_root"]
    return module, module.Paths(repo=tmp_path, spec=spec_path, output=output)


def test_repository_vertical_slice_validates_without_local_tex(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_module()
    original_which = module.shutil.which
    monkeypatch.setattr(
        module.shutil,
        "which",
        lambda name: None if name == "pdfinfo" else original_which(name),
    )
    paths = module.Paths(
        repo=ROOT,
        spec=ROOT / "docs/manuscript/paper_spec_core.yaml",
        output=ROOT / "paper/core_review_v2_core",
    )
    result = module.validate_slice(paths)
    assert result["status"] == "PASS"
    assert result["experiment_status"] == "long_run_validated"
    assert result["pdf_pages"] == 2
    assert result["page_count_source"] == "verified_manifest"


def test_vertical_slice_build_is_deterministic_without_latex(tmp_path: Path) -> None:
    module, paths = prepare_minimal_repo(tmp_path)
    first = module.build_slice(paths)
    first_bytes = {path.name: path.read_bytes() for path in sorted(paths.output.iterdir())}
    second = module.build_slice(paths)
    second_bytes = {path.name: path.read_bytes() for path in sorted(paths.output.iterdir())}
    assert first == second
    assert first_bytes == second_bytes


@pytest.mark.skipif(shutil.which("latexmk") is None, reason="latexmk is not installed")
def test_vertical_slice_compiles_in_isolated_output(tmp_path: Path) -> None:
    module, paths = prepare_minimal_repo(tmp_path)
    module.build_slice(paths)
    module.compile_pdf(paths)
    result = module.validate_slice(paths)
    assert result["status"] == "PASS"
    assert result["pdf_pages"] == 2


def test_snapshot_uses_registered_evidence_and_budget_controls(tmp_path: Path) -> None:
    module, paths = prepare_minimal_repo(tmp_path)
    snapshot = module.build_snapshot(paths)
    assert snapshot["experiment"]["terminology"] == "held_out_context_generalization"
    assert snapshot["methods"]["baseline"]["fixed_variance"]["task_collapse_count"] == 20
    assert snapshot["methods"]["far_zero"]["fixed_variance"]["task_collapse_count"] == 0
    assert snapshot["methods"]["baseline"]["learnable_variance"]["support_boundary_count"] == 20
    assert snapshot["methods"]["far_cap"]["learnable_variance"]["support_boundary_count"] == 0
    assert snapshot["methods"]["global_scale"]["paper_role"] == "fixed_budget_control"
    assert snapshot["methods"]["far_to_near"]["fixed_variance"]["reward"] == pytest.approx(
        0.8753230214118958
    )
    assert snapshot["methods"]["far_to_near"]["learnable_variance"] is None


def test_compact_artifact_checksum_mismatch_fails_closed(tmp_path: Path) -> None:
    module, paths = prepare_minimal_repo(tmp_path)
    aggregate = tmp_path / "outputs/cu1_e3_adam/fixed_variance_aggregate.csv"
    aggregate.write_text(aggregate.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(module.CorePipelineError, match="checksum mismatch"):
        module.build_snapshot(paths)


def test_downgraded_experiment_status_fails_closed(tmp_path: Path) -> None:
    module, paths = prepare_minimal_repo(tmp_path)
    registry_path = tmp_path / "experiments/registry.yaml"
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    experiment = next(row for row in registry["experiments"] if row["id"] == "C-U1-E3-ADAM-RERUN")
    experiment["status"] = "pilot"
    registry_path.write_text(yaml.safe_dump(registry, sort_keys=False), encoding="utf-8")
    with pytest.raises(module.CorePipelineError, match="does not satisfy required status"):
        module.build_snapshot(paths)


def test_generated_prose_preserves_claim_boundaries() -> None:
    output = ROOT / "paper/core_review_v2_core"
    text = (output / "prose.md").read_text(encoding="utf-8")
    assert "same-distribution held-out-context" in text
    assert "dominant causal transmission path" in text
    assert "Global-scale" in text
    assert "far budget to the near component" in text
    assert "universal method ranking" in text
    assert "OOD generalization" not in text
    assert "TBD" not in text
    assert "task-performance collapse" in text
    assert "support/variance-contraction" in text
    assert "NaN/Inf" in text


def test_outline_compiler_preserves_all_stable_ids_and_fields(tmp_path: Path) -> None:
    module, paths = prepare_minimal_repo(tmp_path)
    ast = module.build_outline_ast(paths)
    ids = [node["id"] for node in ast["nodes"]]
    assert ast["node_count"] == 39
    assert len(ids) == len(set(ids))
    assert ids[0] == "ABSTRACT-P01"
    assert ids[-1] == "APP-CORR-P01"
    exp = next(node for node in ast["nodes"] if node["id"] == "EXP-P04")
    assert exp["section"] == "Experiments"
    assert exp["title"] == "RQ2b: Targeted Causal Transmission"
    assert exp["reader_question"] == "Do large far-field updates actually cause the observed instability?"
    assert "global equal-budget control" in exp["must_include"]


def test_outline_resolution_is_explicit_and_one_to_one(tmp_path: Path) -> None:
    module, paths = prepare_minimal_repo(tmp_path)
    ast = module.build_outline_ast(paths)
    resolution = module.build_outline_resolution(paths, ast)
    assert [node["id"] for node in resolution["nodes"]] == [
        node["id"] for node in ast["nodes"]
    ]
    assert resolution["enabled_node_ids"] == ["METHOD-P03", "EXP-P04"]
    disabled = [node for node in resolution["nodes"] if node["status"] == "disabled_with_reason"]
    assert len(disabled) == 37
    assert all(node["reason"] == "not_selected_for_core_vertical_slice" for node in disabled)


def test_blueprint_contract_rejects_merge_split_or_reorder(tmp_path: Path) -> None:
    module, paths = prepare_minimal_repo(tmp_path)
    snapshot = module.build_snapshot(paths)
    blueprint = module.build_blueprint_contract(paths, snapshot=snapshot)
    ast = module.read_json(paths.outline_ast)
    resolution = module.read_json(paths.outline_resolution)

    broken = dict(blueprint)
    broken["nodes"] = [dict(node) for node in blueprint["nodes"]]
    exp_index = next(index for index, node in enumerate(broken["nodes"]) if node["id"] == "EXP-P04")
    broken["nodes"][exp_index]["id"] = "EXP-P04-A"
    with pytest.raises(module.CorePipelineError, match="exactly preserve outline IDs and order"):
        module.validate_blueprint_payload(ast=ast, resolution=resolution, blueprint=broken)


def test_blueprint_contract_rejects_outline_claim_copy(tmp_path: Path) -> None:
    module, paths = prepare_minimal_repo(tmp_path)
    snapshot = module.build_snapshot(paths)
    blueprint = module.build_blueprint_contract(paths, snapshot=snapshot)
    ast = module.read_json(paths.outline_ast)
    resolution = module.read_json(paths.outline_resolution)
    ast_map = {node["id"]: node for node in ast["nodes"]}

    broken = dict(blueprint)
    broken["nodes"] = [dict(node) for node in blueprint["nodes"]]
    exp = next(node for node in broken["nodes"] if node["id"] == "EXP-P04")
    exp["paragraph_claim"] = ast_map["EXP-P04"]["claim"]
    with pytest.raises(module.CorePipelineError, match="without information gain"):
        module.validate_blueprint_payload(ast=ast, resolution=resolution, blueprint=broken)


def test_blueprint_contract_requires_exact_experiment_metrics(tmp_path: Path) -> None:
    module, paths = prepare_minimal_repo(tmp_path)
    snapshot = module.build_snapshot(paths)
    blueprint = module.build_blueprint_contract(paths, snapshot=snapshot)
    ast = module.read_json(paths.outline_ast)
    resolution = module.read_json(paths.outline_resolution)

    broken = dict(blueprint)
    broken["nodes"] = [dict(node) for node in blueprint["nodes"]]
    exp = next(node for node in broken["nodes"] if node["id"] == "EXP-P04")
    exp["metric_paths"] = []
    with pytest.raises(module.CorePipelineError, match="metric_paths"):
        module.validate_blueprint_payload(ast=ast, resolution=resolution, blueprint=broken)


def test_generated_blueprint_and_prose_preserve_outline_identity() -> None:
    output = ROOT / "paper/core_review_v2_core"
    ast = yaml.safe_load((ROOT / "docs/manuscript/paper_spec_core.yaml").read_text(encoding="utf-8"))
    assert ast["blueprint_contract"]["enabled_nodes"].keys() == {"METHOD-P03", "EXP-P04"}
    blueprint = (output / "blueprint.md").read_text(encoding="utf-8")
    prose = (output / "prose.md").read_text(encoding="utf-8")
    assert "EXP-P04-A" not in blueprint
    assert "EXP-P04-B" not in blueprint
    assert "EXP-P04-A" not in prose
    assert "EXP-P04-B" not in prose
    assert blueprint.index("## METHOD-P03") < blueprint.index("## EXP-P04")
    assert prose.index("## [METHOD-P03]") < prose.index("## [EXP-P04]")


def test_blueprint_contract_rejects_generic_evidence_labels(tmp_path: Path) -> None:
    module, paths = prepare_minimal_repo(tmp_path)
    snapshot = module.build_snapshot(paths)
    blueprint = module.build_blueprint_contract(paths, snapshot=snapshot)
    ast = module.read_json(paths.outline_ast)
    resolution = module.read_json(paths.outline_resolution)

    broken = dict(blueprint)
    broken["nodes"] = [dict(node) for node in blueprint["nodes"]]
    exp = next(node for node in broken["nodes"] if node["id"] == "EXP-P04")
    exp["evidence_refs"] = ["C-U1 E3"]
    with pytest.raises(module.CorePipelineError, match="generic evidence"):
        module.validate_blueprint_payload(
            ast=ast, resolution=resolution, blueprint=broken, snapshot=snapshot
        )


def test_blueprint_contract_rejects_unresolvable_metric_path(tmp_path: Path) -> None:
    module, paths = prepare_minimal_repo(tmp_path)
    spec = yaml.safe_load(paths.spec.read_text(encoding="utf-8"))
    spec["blueprint_contract"]["enabled_nodes"]["EXP-P04"]["metric_paths"][0] = (
        "methods.baseline.fixed_variance.not_a_metric"
    )
    paths.spec.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    with pytest.raises(module.CorePipelineError, match="does not resolve"):
        module.build_blueprint_contract(paths)
