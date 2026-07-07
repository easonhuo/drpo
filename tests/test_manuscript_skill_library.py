from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SKILL_SCRIPT = ROOT / "scripts/manuscript_skill_library.py"
PUBLICATION_SCRIPT = ROOT / "scripts/manuscript_publication_pipeline.py"
SKILLS_ROOT = ROOT / "docs/manuscript/skills"
PROJECT_PROFILE = ROOT / "docs/manuscript/projects/drpo_profile.yaml"


def load_skill_module():
    spec = importlib.util.spec_from_file_location(
        "drpo_manuscript_skill_library_test", SKILL_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_publication_module():
    spec = importlib.util.spec_from_file_location(
        "drpo_publication_with_skills_test", PUBLICATION_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_skill_library_schema_router_and_leakage_checks_pass() -> None:
    module = load_skill_module()
    result = module.validate_library(SKILLS_ROOT, PROJECT_PROFILE)
    assert result["status"] == "PASS"
    assert result["core_skill_count"] >= 10
    assert result["domain_skill_count"] >= 6
    assert "prose_generation" in result["tasks"]
    assert "claim_evidence_audit" in result["tasks"]


def test_task_router_resolves_minimal_obligations() -> None:
    module = load_skill_module()
    report = module.build_obligations_report(
        SKILLS_ROOT, PROJECT_PROFILE, ["prose_generation", "experiment_section"]
    )
    assert report["status"] == "PASS"
    assert report["mode"] == "report_only"
    prose = report["obligations_by_task"]["prose_generation"]
    assert any(skill["skill_id"] == "core.section_contract" for skill in prose["core"])
    assert any(skill["skill_id"] == "rl.mechanism_vs_performance" for skill in prose["domain"])
    assert "project.claim_boundary" in prose["project_obligations"]


def test_core_skill_project_term_leakage_fails_closed(tmp_path: Path) -> None:
    module = load_skill_module()
    copied = tmp_path / "skills"
    shutil.copytree(SKILLS_ROOT, copied)
    core_path = copied / "core_paper_writing_skills.yaml"
    payload = yaml.safe_load(core_path.read_text(encoding="utf-8"))
    payload["skills"][0]["rule"]["summary"] += " DRPO-specific wording must fail here."
    core_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(module.SkillLibraryError, match="project-specific terms"):
        module.validate_library(copied, PROJECT_PROFILE)


def test_publication_pipeline_emits_report_only_skill_obligations(tmp_path: Path) -> None:
    module = load_publication_module()
    paths = module.Paths(
        root=ROOT,
        graph=ROOT / "docs/manuscript/paper_graph.yaml",
        contract=ROOT / "docs/manuscript/publication_quality_contract.yaml",
        output=tmp_path / "publication_quality",
        skills_root=SKILLS_ROOT,
        project_profile=PROJECT_PROFILE,
    )
    result = module.build(paths)
    assert result["status"] == "BUILT"
    assert result["skill_library_status"] == "PASS"
    packets = module.json.loads((paths.output / "prose_packets.json").read_text(encoding="utf-8"))
    assert packets["skill_library"]["mode"] == "report_only"
    first = packets["packets"][0]
    obligations = first["quality_requirements"]["skill_obligations"]
    assert any(skill["skill_id"] == "core.claim_evidence_binding" for skill in obligations["core"])


def test_publication_pipeline_disable_path_removes_skill_obligations(tmp_path: Path) -> None:
    module = load_publication_module()
    paths = module.Paths(
        root=ROOT,
        graph=ROOT / "docs/manuscript/paper_graph.yaml",
        contract=ROOT / "docs/manuscript/publication_quality_contract.yaml",
        output=tmp_path / "publication_quality_disabled",
        skills_root=SKILLS_ROOT,
        project_profile=PROJECT_PROFILE,
        disable_skill_library=True,
    )
    result = module.build(paths)
    assert result["skill_library_status"] == "DISABLED"
    packets = module.json.loads((paths.output / "prose_packets.json").read_text(encoding="utf-8"))
    assert packets["skill_library"]["mode"] == "disabled"
    assert packets["packets"][0]["quality_requirements"]["skill_obligations"] == {}
