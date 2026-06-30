from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "manuscript_cascade.py"


def load_module():
    spec = importlib.util.spec_from_file_location("drpo_manuscript_cascade_live", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_live_introduction_hierarchy_is_aligned_to_v07() -> None:
    module = load_module()
    config = module.load_config(ROOT / "docs/manuscript/hierarchy.yaml")
    result = module.validate_artifacts(config, repo_root=ROOT)
    section = result["sections"]["introduction"]
    assert section["paragraph_ids"] == [f"INTRO-P{i:02d}" for i in range(1, 8)]
    assert section["titles"] == [
        "背景与重要性",
        "正负 advantage 的不同作用",
        "为什么 fixed/stale off-policy data 特别危险",
        "已有解决方法",
        "共同缺口",
        "本文理论与方法",
        "实验版图与贡献",
    ]
    assert section["configured_layers"] == ["outline", "blueprint"]
    assert section["status"] == "pass"


def test_reverse_alignment_correction_keeps_outline_as_pass() -> None:
    module = load_module()
    config = module.load_config(ROOT / "docs/manuscript/hierarchy.yaml")
    issue = module._load_issue(
        ROOT
        / "docs/manuscript/issues/PAPER-INTRO-REVERSE-ALIGNMENT-CORRECTION-02.yaml"
    )
    root, required, summary = module.validate_issue(issue, config)
    assert root == "blueprint"
    assert required == ["blueprint"]
    assert summary["change_kind"] == "alignment_repair"
    assert summary["outline_change_authorized"] is False


def test_v07_metadata_migration_is_explicitly_authorized() -> None:
    module = load_module()
    config = module.load_config(ROOT / "docs/manuscript/hierarchy.yaml")
    issue = module._load_issue(
        ROOT / "docs/manuscript/issues/PAPER-INTRO-V07-LIVE-METADATA-01.yaml"
    )
    root, required, summary = module.validate_issue(issue, config)
    assert root == "outline"
    assert required == ["outline", "blueprint"]
    assert summary["change_kind"] == "infrastructure_migration"
    assert summary["outline_change_authorized"] is True


def test_superseded_reverse_alignment_artifacts_are_not_active() -> None:
    config = (ROOT / "docs/manuscript/hierarchy.yaml").read_text(encoding="utf-8")
    assert "paper_rewrite_outline_v0_7.md" in config
    assert "paper_rewrite_intro_blueprint_v0_3.md" in config
    assert "paper_rewrite_outline_v0_8.md" not in config
    assert "paper_rewrite_intro_blueprint_v0_2.md" not in config
