from __future__ import annotations

import hashlib
import importlib.util
import shutil
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "scripts" / "validate_stage4a_inventory.py"
STAGE4_ROOT = Path("docs/handoff_shadow/stage4")


def load_validator():
    spec = importlib.util.spec_from_file_location("stage4a_validator", VALIDATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {VALIDATOR}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = load_validator()


def copy_repository(tmp_path: Path) -> Path:
    destination = tmp_path / "repo"
    (destination / "scripts").mkdir(parents=True)
    (destination / "docs").mkdir(parents=True)
    (destination / "experiments").mkdir(parents=True)
    shutil.copy2(VALIDATOR, destination / "scripts" / VALIDATOR.name)
    shutil.copy2(ROOT / "docs" / "handoff.md", destination / "docs" / "handoff.md")
    shutil.copy2(
        ROOT / "experiments" / "registry.yaml",
        destination / "experiments" / "registry.yaml",
    )
    shutil.copytree(ROOT / STAGE4_ROOT, destination / STAGE4_ROOT)
    return destination


def dump_yaml(path: Path, payload: dict) -> None:
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=140),
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_invalid(repo: Path, message: str) -> None:
    with pytest.raises(MODULE.Stage4AError, match=message):
        MODULE.validate(repo)


def test_current_stage4a_inventory_is_valid_and_deterministic() -> None:
    first = MODULE.validate(ROOT)
    second = MODULE.validate(ROOT)
    assert first == second
    assert first["status"] == "PASS"
    assert first["phase"] == "stage_4a_schema_inventory"
    assert first["manual_handoff_remains_authoritative"] is True
    assert first["authority_cutover_allowed"] is False
    assert first["module_count"] == 9
    assert first["heading_count"] > 300
    assert first["claim_count"] >= 20
    assert first["experiment_count"] == 25



def test_module_dependency_cycle_is_rejected(tmp_path: Path) -> None:
    repo = copy_repository(tmp_path)
    path = repo / STAGE4_ROOT / "inventory" / "MODULES.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    modules = {item["module_id"]: item for item in payload["modules"]}
    modules["global_core_governance"]["default_dependencies"] = [
        "execution_status_gates"
    ]
    dump_yaml(path, payload)
    assert_invalid(repo, "module dependency graph contains a cycle")

def test_handoff_source_drift_is_rejected(tmp_path: Path) -> None:
    repo = copy_repository(tmp_path)
    handoff = repo / "docs" / "handoff.md"
    handoff.write_text(handoff.read_text(encoding="utf-8") + "\nsource drift\n")
    assert_invalid(repo, "source SHA-256 is stale")


def test_missing_heading_inventory_entry_is_rejected(tmp_path: Path) -> None:
    repo = copy_repository(tmp_path)
    path = repo / STAGE4_ROOT / "inventory" / "HEADINGS.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["headings"].pop()
    dump_yaml(path, payload)
    assert_invalid(repo, "heading inventory count mismatch")


def test_unresolved_heading_classification_is_rejected(tmp_path: Path) -> None:
    repo = copy_repository(tmp_path)
    path = repo / STAGE4_ROOT / "inventory" / "HEADINGS.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["headings"][0]["classification"] = "ambiguous"
    dump_yaml(path, payload)
    assert_invalid(repo, "classification is unresolved or invalid")


def test_multi_module_classification_requires_rationale(tmp_path: Path) -> None:
    repo = copy_repository(tmp_path)
    path = repo / STAGE4_ROOT / "inventory" / "HEADINGS.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["headings"][0].pop("classification_rationale")
    dump_yaml(path, payload)
    assert_invalid(repo, "multi-module rationale")


def test_duplicate_claim_id_is_rejected(tmp_path: Path) -> None:
    repo = copy_repository(tmp_path)
    path = repo / STAGE4_ROOT / "inventory" / "CLAIMS.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["claims"][1]["claim_id"] = payload["claims"][0]["claim_id"]
    dump_yaml(path, payload)
    assert_invalid(repo, "duplicate claim_id")


def test_claim_anchor_mutation_is_rejected(tmp_path: Path) -> None:
    repo = copy_repository(tmp_path)
    path = repo / STAGE4_ROOT / "inventory" / "CLAIMS.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    claim = payload["claims"][0]
    claim["source_anchor"]["text"] = "invented claim text"
    claim["source_anchor"]["sha256"] = hashlib.sha256(
        b"invented claim text"
    ).hexdigest()
    dump_yaml(path, payload)
    assert_invalid(repo, "anchor must occur exactly once")


def test_dangling_experiment_claim_reference_is_rejected(tmp_path: Path) -> None:
    repo = copy_repository(tmp_path)
    path = repo / STAGE4_ROOT / "inventory" / "EXPERIMENTS.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["experiments"][0]["claim_ids"] = ["SCI-NOT-REGISTERED-01"]
    dump_yaml(path, payload)
    assert_invalid(repo, "dangling claim references")


def test_registry_experiment_without_inventory_entry_is_rejected(tmp_path: Path) -> None:
    repo = copy_repository(tmp_path)
    registry_path = repo / "experiments" / "registry.yaml"
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    registry["experiments"].append(
        {
            "id": "EXT-H-E7-MUTATION-01",
            "environment": "EXT-H",
            "name": "mutation-only",
            "status": "not_run",
            "role": "external_validation",
        }
    )
    dump_yaml(registry_path, registry)

    inventory_path = repo / STAGE4_ROOT / "inventory" / "EXPERIMENTS.yaml"
    inventory = yaml.safe_load(inventory_path.read_text(encoding="utf-8"))
    inventory["source"]["sha256"] = sha256(registry_path)
    dump_yaml(inventory_path, inventory)
    assert_invalid(repo, "experiment inventory count mismatch")


def test_stage4b_candidate_is_rejected_before_acceptance(tmp_path: Path) -> None:
    repo = copy_repository(tmp_path)
    candidate = repo / STAGE4_ROOT / "CURRENT_CANDIDATE.md"
    candidate.write_text("not authorized\n", encoding="utf-8")
    assert_invalid(repo, "forbidden 4B/4C output")


def test_unknown_relation_type_is_rejected(tmp_path: Path) -> None:
    repo = copy_repository(tmp_path)
    path = repo / STAGE4_ROOT / "schema" / "STAGE4A_SCHEMA.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["relation_types"].append("related_to")
    dump_yaml(path, payload)
    assert_invalid(repo, "relation_types must equal")


def test_unknown_claim_node_type_is_rejected(tmp_path: Path) -> None:
    repo = copy_repository(tmp_path)
    path = repo / STAGE4_ROOT / "inventory" / "CLAIMS.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["claims"][0]["node_type"] = "observation"
    dump_yaml(path, payload)
    assert_invalid(repo, "has unknown node_type")


def test_claim_lineage_cycle_is_rejected(tmp_path: Path) -> None:
    repo = copy_repository(tmp_path)
    path = repo / STAGE4_ROOT / "inventory" / "CLAIMS.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    by_id = {item["claim_id"]: item for item in payload["claims"]}
    first = by_id["SCI-GAUSSIAN-VARIANCE-CORRECTION-01"]
    second = by_id["HIST-MU-SIGMA-EXPAND-01"]
    first["status"] = "historical_superseded"
    first["archive_pointer"] = "docs/handoff.md#mutation-only"
    first["reopen_conditions"] = ["mutation-only"]
    first["lineage"] = {
        "supersedes": [second["claim_id"]],
        "superseded_by": [second["claim_id"]],
    }
    second["lineage"] = {
        "supersedes": [first["claim_id"]],
        "superseded_by": [first["claim_id"]],
    }
    dump_yaml(path, payload)
    assert_invalid(repo, "claim lineage contains a cycle")
