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


def test_full_paper_hierarchy_is_aligned_to_v092() -> None:
    module = load_module()
    config = module.load_config(ROOT / "docs/manuscript/hierarchy.yaml")
    result = module.validate_artifacts(config, repo_root=ROOT)
    section = result["sections"]["full-paper"]
    assert section["configured_layers"] == ["outline", "blueprint", "prose"]
    assert section["status"] == "pass"
    assert len(section["paragraph_ids"]) == 39
    assert section["paragraph_ids"][0] == "ABSTRACT-P01"
    assert "INTRO-P03" in section["paragraph_ids"]
    assert "APP-DRO-P01" in section["paragraph_ids"]


def test_v092_merge_and_pipeline_issue_is_authorized() -> None:
    module = load_module()
    config = module.load_config(ROOT / "docs/manuscript/hierarchy.yaml")
    issue = module._load_issue(ROOT / "docs/manuscript/issues/PAPER-V092-MERGE-PIPELINE-01.yaml")
    root, required, summary = module.validate_issue(issue, config)
    assert root == "outline"
    assert required == ["outline", "blueprint", "prose"]
    assert summary["change_kind"] == "infrastructure_migration"
    assert summary["outline_change_authorized"] is True


def test_active_hierarchy_uses_v092_and_preserves_history() -> None:
    config = (ROOT / "docs/manuscript/hierarchy.yaml").read_text(encoding="utf-8")
    readme = (ROOT / "docs/manuscript/README.md").read_text(encoding="utf-8")
    assert "paper_rewrite_outline_v0_9_2.md" in config
    assert "paper_rewrite_blueprint_v0_6.md" in config
    assert "paper_rewrite_prose_v0_1.md" in config
    assert "Historical artifacts remain" in readme
    assert "must not be destructively deleted" in readme
    assert "V092_MERGE_LEDGER.md" in readme
    assert "RL_PAPER_WRITING_PLAYBOOK.md" in readme
