from pathlib import Path
import importlib.util
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "scripts/compile_full_manuscript.py"
    spec = importlib.util.spec_from_file_location("compile_full_manuscript", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_template_contract_is_explicitly_two_column_icml():
    cfg = yaml.safe_load((ROOT / "docs/manuscript/full_paper_assets.yaml").read_text())
    assert cfg["paper_template"]["family"] == "icml2026"
    assert cfg["paper_template"]["columns"] == 2
    release = load_module()._release_module()
    template = cfg["paper_template"]
    active_source = release.resolve_active_template_source(
        ROOT / template["main_tex"],
        ROOT / template["root"],
    )
    assert "\\twocolumn" in active_source
    assert "icml2026" in active_source


def test_generated_assets_are_bound_to_stable_nodes():
    cfg = yaml.safe_load((ROOT / "docs/manuscript/full_paper_assets.yaml").read_text())
    tex = "\n".join(p.read_text() for p in (ROOT / "paper/overleaf").rglob("*.tex"))
    for assets in cfg["assets"].values():
        for asset in assets:
            assert (ROOT / "paper/overleaf" / asset).exists()
            assert f"\\input{{{asset}}}" in tex


def test_proof_and_citation_validation_passes():
    module = load_module()
    cfg = yaml.safe_load((ROOT / "docs/manuscript/full_paper_assets.yaml").read_text())
    module.validate(ROOT, cfg)


def test_release_pdf_is_committed_and_nontrivial():
    pdf = ROOT / "paper/releases/DRPO_FULL_REVIEW_V1.pdf"
    assert pdf.exists()
    assert pdf.stat().st_size > 100_000


def test_release_engine_uses_portable_tex_tool_discovery():
    source = (ROOT / "scripts/manuscript_release_pipeline.py").read_text()
    assert 'shutil.which("latexmk")' in source
    assert 'shutil.which("bibtex")' in source
    assert 'bibtex = shutil.which("bibtex")' in source
    assert "--skip-compile" in source


def test_compiler_runs_publication_quality_gate_through_generic_release_manifest():
    wrapper = (ROOT / "scripts/compile_full_manuscript.py").read_text()
    assert "manuscript_release_pipeline.py" in wrapper
    cfg = yaml.safe_load((ROOT / "docs/manuscript/full_paper_assets.yaml").read_text())
    assert cfg["quality_gate"]["script"] == "scripts/manuscript_publication_pipeline.py"
    assert cfg["quality_gate"]["contract"] == "docs/manuscript/publication_quality_contract.yaml"
    assert cfg["asset_build_commands"][0][1] == "scripts/projects/drpo/build_manuscript_assets.py"
    obligations = {row["statement_label"]: row["proof_label"] for row in cfg["proof_obligations"]}
    assert obligations["prop:score-remoteness"] == "app:proof-score-remoteness"
    assert obligations["thm:reuse"] == "app:proof-reuse"
    assert obligations["thm:family-runaway"] == "app:proof-family-runaway"
