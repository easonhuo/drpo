from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
import yaml

from drpo.du1_e6_semantic import SemanticEnvironment, execute, run_specs
from drpo.du1_e6_semantic_gap import (
    ALPHA_GRID,
    EXPERIMENT_ID,
    HELD_OUT_SEEDS,
    smoke_config,
    validate_formal_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "du1_e6_semantic_gap_longrun.yaml"


def load_config() -> dict:
    value = yaml.safe_load(CONFIG_PATH.read_text())
    assert isinstance(value, dict)
    return value


def test_formal_config_is_exact_minimum_change_successor() -> None:
    config = load_config()
    validate_formal_config(config)
    assert config["experiment_id"] == EXPERIMENT_ID
    assert config["predecessor"] == "D-U1-E6-SEMANTIC-LONGRUN-01"
    assert config["data"]["action_count"] == 64
    assert config["geometry"]["reward_scale"] == 0.5
    assert config["protocol_a"]["local_alpha_grid"] == ALPHA_GRID
    assert max(ALPHA_GRID) == 1.0
    assert config["seeds"]["held_out_formal"] == HELD_OUT_SEEDS
    assert config["optimization"]["maximum_steps"] == 32000


def test_formal_config_rejects_unapproved_parameter_drift() -> None:
    config = load_config()
    changed = copy.deepcopy(config)
    changed["protocol_a"]["local_alpha_grid"] = [0.0, 0.5, 1.0]
    with pytest.raises(ValueError, match="protocol_a"):
        validate_formal_config(changed)
    changed = copy.deepcopy(config)
    changed["data"]["action_count"] = 256
    with pytest.raises(ValueError, match="data"):
        validate_formal_config(changed)
    changed = copy.deepcopy(config)
    changed["conditional_coverage"]["withheld_action_fraction"] = 0.5
    with pytest.raises(ValueError, match="conditional_coverage"):
        validate_formal_config(changed)


def test_structured_gap_is_only_logged_role_intervention() -> None:
    config = load_config()
    environment = SemanticEnvironment(config, seed=150, embedding_mode="aligned")
    audit = environment.audit()
    assert audit["passed"] is True
    for split_name in ("train", "test"):
        coverage = audit["splits"][split_name]["conditional_coverage"]
        assert coverage["mode"] == "structured_semantic_neighbourhood_gap"
        assert coverage["gap_state_count"] == 1024
        assert coverage["withheld_action_count_min"] == 16
        assert coverage["withheld_action_count_max"] == 16
        assert coverage["logged_role_gap_violations"] == 0
        assert coverage["hidden_not_withheld_violations"] == 0
        assert coverage["globally_unobserved_action_count"] == 0


def test_original_dense_e6_environment_remains_unchanged() -> None:
    dense = yaml.safe_load(
        (REPO_ROOT / "configs" / "du1_e6_semantic_longrun.yaml").read_text()
    )
    environment = SemanticEnvironment(dense, seed=10, embedding_mode="aligned")
    audit = environment.audit()
    assert audit["passed"] is True
    for split_name in ("train", "test"):
        coverage = audit["splits"][split_name]["conditional_coverage"]
        assert coverage["mode"] == "dense"
        assert coverage["gap_state_count"] == 0
        assert coverage["withheld_action_count_min"] == 0
        assert coverage["logged_role_gap_violations"] == 0


def test_run_matrix_has_only_legal_alpha_domain() -> None:
    specs = run_specs(load_config())
    assert len(specs) == 5
    assert [spec.alpha for spec in specs] == ALPHA_GRID
    assert all(0.0 <= spec.alpha <= 1.0 for spec in specs)
    assert {spec.method for spec in specs} == {"positive_only", "local_only"}
    assert all(spec.far_lambda == 0.0 for spec in specs)


def test_engineering_smoke_is_nonformal_and_writes_horizon_free_outputs(
    tmp_path: Path,
) -> None:
    config = smoke_config(load_config())
    output = tmp_path / "smoke"
    execute(config, "all", output, torch.device("cpu"))
    complete = json.loads((output / "RUN_COMPLETE.json").read_text())
    audits = json.loads((output / "environment_audits.json").read_text())
    assert complete["completed"] is True
    assert complete["formal_result"] is False
    assert complete["scientific_status"] == "pilot"
    assert complete["actual_runs"] == 3
    assert all(item["passed"] for item in audits)
    assert not (output / "formal_protocol_freeze.json").exists()


def test_registry_and_handoff_register_ready_formal_successor() -> None:
    registry = yaml.safe_load((REPO_ROOT / "experiments" / "registry.yaml").read_text())
    experiments = {item["id"]: item for item in registry["experiments"]}
    item = experiments[EXPERIMENT_ID]
    assert item["status"] == "not_run"
    assert item["execution_gate"]["state"] == "ready"
    assert item["formal_execution"]["activation_state"] == "active"
    assert item["protocol"]["action_count"] == 64
    assert item["protocol"]["alpha_grid"] == ALPHA_GRID
    assert item["held_out_seeds"] == HELD_OUT_SEEDS
    handoff = (REPO_ROOT / "docs" / "handoff.md").read_text()
    assert "v51（D-U1 E6 条件缺口闭环与最小改动正式协议版）" in handoff
    assert EXPERIMENT_ID in handoff
