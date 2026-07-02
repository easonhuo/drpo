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
    main = (ROOT / "paper/overleaf/main.tex").read_text()
    assert "\\twocolumn" in main
    assert "icml2026" in main


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


def test_compiler_uses_portable_tex_tool_discovery():
    source = (ROOT / "scripts/compile_full_manuscript.py").read_text()
    assert 'shutil.which("latexmk")' in source
    assert 'shutil.which("bibtex")' in source
    assert 'bibtex = shutil.which("bibtex")' in source
    assert "--skip-compile" in source


def test_compiler_runs_publication_quality_gate():
    source = (ROOT / "scripts/compile_full_manuscript.py").read_text()
    assert "scripts/manuscript_publication_pipeline.py" in source
    cfg = yaml.safe_load((ROOT / "docs/manuscript/full_paper_assets.yaml").read_text())
    assert (
        cfg["publication_quality_contract"] == "docs/manuscript/publication_quality_contract.yaml"
    )
    obligations = {row["statement_label"]: row["proof_label"] for row in cfg["proof_obligations"]}
    assert obligations["prop:score-remoteness"] == "app:proof-score-remoteness"
    assert obligations["thm:reuse"] == "app:proof-reuse"
    assert obligations["thm:family-runaway"] == "app:proof-family-runaway"
