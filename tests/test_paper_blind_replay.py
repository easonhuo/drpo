from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "paper_blind_replay.py"


def load_module():
    spec = importlib.util.spec_from_file_location("drpo_paper_blind_replay_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_outline(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """# Introduction

<!-- MANUSCRIPT:BEGIN INTRO-P01 -->
## [INTRO-P01] Problem setup

**Claim:** Off-policy policy optimization needs a controlled replay boundary.

**Reader question:** What is being tested without seeing the label paper?

**Role:** Establish the blind replay protocol.

**Required evidence:**
- BLIND-MANIFEST

**Must include:**
- frozen outline
- manifest-bound inputs

**Must avoid:**
- label leakage
- old prose reuse
<!-- MANUSCRIPT:END INTRO-P01 -->
""",
        encoding="utf-8",
    )


def make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    write_outline(repo / "docs/paper_rewrite_outline_v0_9_2.md")
    (repo / "docs").mkdir(exist_ok=True)
    (repo / "docs/handoff.md").write_text("handoff\n", encoding="utf-8")
    (repo / "experiments").mkdir(parents=True)
    (repo / "experiments/registry.yaml").write_text("experiments: []\n", encoding="utf-8")
    (repo / "paper/releases").mkdir(parents=True)
    (repo / "paper/releases/DRPO_REWRITTEN_DRAFT_20260705.pdf").write_bytes(
        b"%PDF-label sentinel LABEL_ONLY_SENTINEL"
    )
    (repo / "docs/paper_rewrite_prose_v0_1.md").write_text(
        "old prose LABEL_ONLY_SENTINEL\n", encoding="utf-8"
    )
    return repo


def test_blind_all_creates_manifest_and_scaffold_without_label_copy(tmp_path: Path) -> None:
    module = load_module()
    repo = make_repo(tmp_path)
    workspace = tmp_path / "workspace"
    rc = module.main(
        [
            "all",
            "--repo-root",
            str(repo),
            "--workspace",
            str(workspace),
            "--optimized-outline",
            "docs/paper_rewrite_outline_v0_9_2.md",
            "--label-source",
            "paper/releases/DRPO_REWRITTEN_DRAFT_20260705.pdf",
            "--allow-input",
            "docs/handoff.md",
            "--allow-input",
            "experiments/registry.yaml",
            "--sentinel",
            "LABEL_ONLY_SENTINEL",
        ]
    )
    assert rc == 0
    manifest = json.loads((workspace / "BLIND_INPUT_MANIFEST.json").read_text())
    copied = {item["repo_path"] for item in manifest["allowed_inputs"]}
    assert copied == {
        "docs/paper_rewrite_outline_v0_9_2.md",
        "docs/handoff.md",
        "experiments/registry.yaml",
    }
    assert manifest["label_source"]["copied_into_generation_workspace"] is False
    assert not (workspace / "inputs/paper/releases/DRPO_REWRITTEN_DRAFT_20260705.pdf").exists()
    assert (workspace / "generated/replay_blueprint.md").is_file()
    assert (workspace / "BLIND_REPLAY_AUDIT.json").is_file()


def test_forbidden_label_pdf_cannot_be_allowed_generation_input(tmp_path: Path) -> None:
    module = load_module()
    repo = make_repo(tmp_path)
    rc = module.main(
        [
            "init",
            "--repo-root",
            str(repo),
            "--workspace",
            str(tmp_path / "workspace"),
            "--optimized-outline",
            "docs/paper_rewrite_outline_v0_9_2.md",
            "--allow-input",
            "paper/releases/DRPO_REWRITTEN_DRAFT_20260705.pdf",
        ]
    )
    assert rc == 2


def test_forbidden_old_prose_cannot_be_allowed_generation_input(tmp_path: Path) -> None:
    module = load_module()
    repo = make_repo(tmp_path)
    rc = module.main(
        [
            "init",
            "--repo-root",
            str(repo),
            "--workspace",
            str(tmp_path / "workspace"),
            "--optimized-outline",
            "docs/paper_rewrite_outline_v0_9_2.md",
            "--allow-input",
            "docs/paper_rewrite_prose_v0_1.md",
        ]
    )
    assert rc == 2


def test_audit_fails_on_sentinel_leakage(tmp_path: Path) -> None:
    module = load_module()
    repo = make_repo(tmp_path)
    workspace = tmp_path / "workspace"
    assert module.main(
        [
            "init",
            "--repo-root",
            str(repo),
            "--workspace",
            str(workspace),
            "--optimized-outline",
            "docs/paper_rewrite_outline_v0_9_2.md",
            "--sentinel",
            "LABEL_ONLY_SENTINEL",
        ]
    ) == 0
    generated = workspace / "generated"
    generated.mkdir()
    (generated / "replay_prose.md").write_text(
        "This copied LABEL_ONLY_SENTINEL from the label.\n", encoding="utf-8"
    )
    rc = module.main(["audit", "--workspace", str(workspace)])
    assert rc == 2
    report = json.loads((workspace / "BLIND_REPLAY_AUDIT.json").read_text())
    assert report["status"] == "FAIL"
    assert report["leakages"][0]["kind"] == "sentinel"


def test_audit_fails_on_forbidden_path_token(tmp_path: Path) -> None:
    module = load_module()
    repo = make_repo(tmp_path)
    workspace = tmp_path / "workspace"
    assert module.main(
        [
            "init",
            "--repo-root",
            str(repo),
            "--workspace",
            str(workspace),
            "--optimized-outline",
            "docs/paper_rewrite_outline_v0_9_2.md",
        ]
    ) == 0
    generated = workspace / "generated"
    generated.mkdir()
    (generated / "replay_prose.md").write_text(
        "This cites paper/releases/DRPO_REWRITTEN_DRAFT_20260705.pdf.\n",
        encoding="utf-8",
    )
    rc = module.main(["audit", "--workspace", str(workspace)])
    assert rc == 2
    report = json.loads((workspace / "BLIND_REPLAY_AUDIT.json").read_text())
    assert report["status"] == "FAIL"
    assert report["leakages"][0]["kind"] == "forbidden_path_token"


def test_scaffold_uses_manifest_inputs_after_init(tmp_path: Path) -> None:
    module = load_module()
    repo = make_repo(tmp_path)
    workspace = tmp_path / "workspace"
    assert module.main(
        [
            "init",
            "--repo-root",
            str(repo),
            "--workspace",
            str(workspace),
            "--optimized-outline",
            "docs/paper_rewrite_outline_v0_9_2.md",
        ]
    ) == 0
    assert module.main(["scaffold", "--workspace", str(workspace)]) == 0
    prose = (workspace / "generated/replay_prose.md").read_text(encoding="utf-8")
    assert "Off-policy policy optimization needs a controlled replay boundary" in prose
    assert "LABEL_ONLY_SENTINEL" not in prose
    assert module.main(["audit", "--workspace", str(workspace)]) == 0
