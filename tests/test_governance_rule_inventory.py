from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "validate_governance_rule_inventory.py"
SPEC = importlib.util.spec_from_file_location("governance_inventory", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _load_inventory() -> dict:
    return yaml.safe_load((ROOT / "docs" / "governance_rule_inventory.yaml").read_text())


def _write_inventory(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))


def test_repository_inventory_validates() -> None:
    report = MODULE.validate_inventory(
        ROOT,
        ROOT / "docs" / "governance_rule_inventory.yaml",
    )
    assert report["matched"] is True
    assert report["tracked_sections"] == 3
    assert report["covered_rules"] > 30
    assert report["machine_enforced_rules"] >= 3


def test_untracked_new_bullet_is_rejected(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    for rel in [
        "AGENTS.md",
        "docs/governance_rule_inventory.yaml",
        "docs/formal_experiment_artifact_protocol.md",
        "scripts/artifact_protocol_hardened.py",
        "scripts/run_experiment_guard_hardened.py",
        "scripts/package_experiment_hardened.py",
        "scripts/verify_experiment_package_hardened.py",
        "tests/test_experiment_artifact_protocol.py",
        "tests/test_experiment_artifact_hardening.py",
    ]:
        source = ROOT / rel
        target = repo / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    agents = (repo / "AGENTS.md").read_text()
    marker = "## Method-comparison discipline"
    agents = agents.replace(marker, "* A newly added unregistered hard rule.\n\n" + marker)
    (repo / "AGENTS.md").write_text(agents)
    with pytest.raises(
        MODULE.InventoryError,
        match="expected .* items|section item digest changed",
    ):
        MODULE.validate_inventory(repo, repo / "docs/governance_rule_inventory.yaml")


def test_machine_enforceable_rule_requires_implementation_and_test(tmp_path: Path) -> None:
    payload = _load_inventory()
    candidate = copy.deepcopy(payload)
    rule = next(
        item
        for item in candidate["rules"]
        if item["machine_enforcement"].get("required") is True
    )
    rule["machine_enforcement"]["implementations"] = []
    path = tmp_path / "inventory.yaml"
    _write_inventory(path, candidate)
    with pytest.raises(MODULE.InventoryError, match="lacks implementation"):
        MODULE.validate_inventory(ROOT, path)


def test_migrated_rule_requires_destination_and_evidence(tmp_path: Path) -> None:
    payload = _load_inventory()
    candidate = copy.deepcopy(payload)
    candidate["rules"][0]["migration"] = {
        "status": "migrated",
        "destination_locations": [],
        "evidence": [],
    }
    path = tmp_path / "inventory.yaml"
    _write_inventory(path, candidate)
    with pytest.raises(MODULE.InventoryError, match="needs destinations"):
        MODULE.validate_inventory(ROOT, path)
