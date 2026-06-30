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


def test_live_introduction_hierarchy_is_aligned() -> None:
    module = load_module()
    config = module.load_config(ROOT / "docs/manuscript/hierarchy.yaml")
    result = module.validate_artifacts(config, repo_root=ROOT)
    section = result["sections"]["introduction"]
    assert section["paragraph_ids"] == [f"INTRO-P{i:02d}" for i in range(1, 9)]
    assert section["titles"] == [
        "Background and Motivation",
        "Far-Field Negative-Gradient Mechanism",
        "Persistence under Off-Policy Data Reuse",
        "Existing Controls and the Remaining Gap",
        "Why Negative Updates Cannot Simply Be Removed",
        "Equilibrium and Divergence of Repulsive Policy Updates",
        "DRPO",
        "Evidence Chain and Contributions",
    ]
    assert section["configured_layers"] == ["outline", "blueprint"]
    assert section["status"] == "pass"


def test_migration_issue_records_outline_root_and_blueprint_cascade() -> None:
    module = load_module()
    config = module.load_config(ROOT / "docs/manuscript/hierarchy.yaml")
    issue = module._load_issue(
        ROOT / "docs/manuscript/issues/PAPER-INTRO-CASCADE-MIGRATION-01.yaml"
    )
    root, required, summary = module.validate_issue(issue, config)
    assert root == "outline"
    assert required == ["outline", "blueprint"]
    assert summary["state"] == "completed"
