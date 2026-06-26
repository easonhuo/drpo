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


def _load_assurance() -> dict:
    return yaml.safe_load((ROOT / "docs" / "governance_rule_assurance.yaml").read_text())


def _write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))


def test_repository_inventory_and_assurance_validate() -> None:
    report = MODULE.validate_governance(
        ROOT,
        ROOT / "docs" / "governance_rule_inventory.yaml",
        ROOT / "docs" / "governance_rule_assurance.yaml",
    )
    assert report["matched"] is True
    assert report["inventory"]["tracked_sections"] == 3
    assert report["inventory"]["covered_rules"] == 49
    assert report["inventory"]["machine_enforced_rules"] == 11
    assert report["assurance"]["assurance_rules"] == 49
    assert report["assurance"]["assurance_type_counts"]["machine"] == 11
    assert report["assurance"]["assurance_type_counts"]["review"] == 38
    assert report["assurance"]["grouped_machine_rules"] == 3
    assert report["assurance"]["unique_pytest_nodes"]


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
    _write_yaml(path, candidate)
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
    _write_yaml(path, candidate)
    with pytest.raises(MODULE.InventoryError, match="needs destinations"):
        MODULE.validate_inventory(ROOT, path)


def test_every_inventory_rule_requires_one_assurance_entry(tmp_path: Path) -> None:
    candidate = copy.deepcopy(_load_assurance())
    candidate["rules"].pop(next(iter(candidate["rules"])))
    path = tmp_path / "assurance.yaml"
    _write_yaml(path, candidate)
    with pytest.raises(MODULE.InventoryError, match="assurance coverage mismatch"):
        MODULE.validate_assurance(
            ROOT,
            ROOT / "docs" / "governance_rule_inventory.yaml",
            path,
        )


def test_machine_assurance_requires_exact_pytest_nodes(tmp_path: Path) -> None:
    candidate = copy.deepcopy(_load_assurance())
    rule_id = next(
        rule_id for rule_id, entry in candidate["rules"].items() if entry["type"] == "machine"
    )
    candidate["rules"][rule_id]["pytest_nodes"] = []
    path = tmp_path / "assurance.yaml"
    _write_yaml(path, candidate)
    with pytest.raises(MODULE.InventoryError, match="needs exact pytest_nodes"):
        MODULE.validate_assurance(
            ROOT,
            ROOT / "docs" / "governance_rule_inventory.yaml",
            path,
        )


def test_nonexistent_pytest_function_is_rejected(tmp_path: Path) -> None:
    candidate = copy.deepcopy(_load_assurance())
    rule_id = next(
        rule_id for rule_id, entry in candidate["rules"].items() if entry["type"] == "machine"
    )
    test_path = candidate["rules"][rule_id]["pytest_nodes"][0].split("::", 1)[0]
    candidate["rules"][rule_id]["pytest_nodes"] = [
        f"{test_path}::test_function_that_does_not_exist"
    ]
    path = tmp_path / "assurance.yaml"
    _write_yaml(path, candidate)
    with pytest.raises(MODULE.InventoryError, match="pytest node does not exist"):
        MODULE.validate_assurance(
            ROOT,
            ROOT / "docs" / "governance_rule_inventory.yaml",
            path,
        )


def test_machine_assurance_must_match_machine_enforcement_flag(tmp_path: Path) -> None:
    candidate = copy.deepcopy(_load_assurance())
    review_rule_id = next(
        rule_id for rule_id, entry in candidate["rules"].items() if entry["type"] == "review"
    )
    candidate["rules"][review_rule_id] = {
        "type": "machine",
        "implementation_paths": ["scripts/artifact_protocol_hardened.py"],
        "pytest_nodes": [
            "tests/test_experiment_artifact_protocol.py::test_build_and_verify_final_package"
        ],
        "coverage_level": "direct",
    }
    path = tmp_path / "assurance.yaml"
    _write_yaml(path, candidate)
    with pytest.raises(MODULE.InventoryError, match="assurance type disagree"):
        MODULE.validate_assurance(
            ROOT,
            ROOT / "docs" / "governance_rule_inventory.yaml",
            path,
        )


def test_grouped_machine_coverage_requires_visible_note(tmp_path: Path) -> None:
    candidate = copy.deepcopy(_load_assurance())
    rule_id = next(
        rule_id
        for rule_id, entry in candidate["rules"].items()
        if entry.get("coverage_level") == "grouped"
    )
    candidate["rules"][rule_id].pop("coverage_note")
    path = tmp_path / "assurance.yaml"
    _write_yaml(path, candidate)
    with pytest.raises(MODULE.InventoryError, match="grouped coverage needs coverage_note"):
        MODULE.validate_assurance(
            ROOT,
            ROOT / "docs" / "governance_rule_inventory.yaml",
            path,
        )


def test_review_assurance_requires_triggers_and_evidence(tmp_path: Path) -> None:
    candidate = copy.deepcopy(_load_assurance())
    rule_id = next(
        rule_id for rule_id, entry in candidate["rules"].items() if entry["type"] == "review"
    )
    candidate["rules"][rule_id]["required_evidence"] = []
    path = tmp_path / "assurance.yaml"
    _write_yaml(path, candidate)
    with pytest.raises(MODULE.InventoryError, match="review assurance needs required_evidence"):
        MODULE.validate_assurance(
            ROOT,
            ROOT / "docs" / "governance_rule_inventory.yaml",
            path,
        )


def test_pytest_node_collection_and_execution_use_exact_node(tmp_path: Path) -> None:
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_sample.py").write_text(
        "def test_selected():\n    assert True\n\n"
        "def test_not_selected():\n    assert False\n"
    )
    node = "tests/test_sample.py::test_selected"
    collected = MODULE.run_pytest_nodes(tmp_path, [node], collect_only=True)
    assert collected["returncode"] == 0
    executed = MODULE.run_pytest_nodes(tmp_path, [node], collect_only=False)
    assert executed["returncode"] == 0
