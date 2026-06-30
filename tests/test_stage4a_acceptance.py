from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "scripts" / "run_stage4a_acceptance.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("stage4a_acceptance_runner_test", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = load_runner()


def copy_repository(tmp_path: Path) -> Path:
    destination = tmp_path / "repo"
    shutil.copytree(
        REPO_ROOT,
        destination,
        ignore=shutil.ignore_patterns(
            ".git", "__pycache__", ".pytest_cache", ".ruff_cache", "*.pyc", "outputs"
        ),
    )
    return destination


def test_current_stage4a_core_acceptance_passes() -> None:
    report = RUNNER.run_core_acceptance(REPO_ROOT, check_determinism=False)
    assert report["minimal_context"]["status"] == "PASS"
    assert report["semantic_graph"]["review_queue"] == 0
    assert report["mapping"]["unmapped_objects"] == []
    assert set(report["semantic_contracts"]) == set(RUNNER.EXPECTED_CONTRACTS)
    assert {item["target"] for item in report["acceptance_targets"]} == RUNNER.EXPECTED_TARGETS


def test_missing_terminal_topic_is_a_hard_failure(tmp_path: Path) -> None:
    repo = copy_repository(tmp_path)
    path = repo / "docs/handoff_shadow/stage4/minimal/MODULES.yaml"
    payload = yaml.safe_load(path.read_text())
    terminal = next(item for item in payload["modules"] if item["module_id"] == "terminal_audit")
    terminal["content_contract"]["required_topics"].pop()
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))

    with pytest.raises(RUNNER.AcceptanceError, match="semantic contract terminal_audit topic mismatch"):
        RUNNER._targeted_context_check(repo)


def test_checked_in_acceptance_evidence_is_complete() -> None:
    root = REPO_ROOT / "docs/governance_stage4a_acceptance"
    if not (root / "ACCEPTANCE_REPORT.json").exists():
        pytest.skip("acceptance evidence is generated after the pre-evidence gates")
    report = json.loads((root / "ACCEPTANCE_REPORT.json").read_text())
    faults = json.loads((root / "FAULT_INJECTION_REPORT.json").read_text())
    after = json.loads((root / "AFTER_IMAGE.json").read_text())
    assert report["status"] == "PASS"
    assert report["hard_blockers"] == []
    assert report["authority"] == "shadow_only"
    assert report["fault_injection"]["passed"] == report["fault_injection"]["total"]
    assert faults["passed"] == faults["total"]
    assert after["tree_hash"] == report["after_image_tree_hash"]
    assert after["file_count"] == len(after["files"])
