from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import pytest
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drpo.du1_e6_cartesian_taper_v4 import (  # noqa: E402
    CELL_NAMES,
    FORMAL_METHODS,
    HISTORICAL_EXCLUDED_METHODS,
    PROTOCOL_REVISION,
    CartesianPolicy,
    CartesianSemanticEnvironment,
    cache_reference_directions,
    cell_log_probs,
    coordinate_calibration,
    evaluate,
    load_config,
    main,
    method_specs,
    policy_geometry_audit,
    run_seed_bundle,
    taper_coefficients,
    taper_weight,
    validate_config,
)

CONFIG = REPO_ROOT / "configs" / "du1_e6_cartesian_taper_v4.yaml"


def tiny_config() -> dict:
    config = load_config(CONFIG)
    config = copy.deepcopy(config)
    config["formal_parameter_freeze"] = False
    config["data"]["train_states"] = 64
    config["data"]["test_states"] = 64
    config["optimization"]["maximum_steps"] = 4
    config["optimization"]["evaluation_interval_steps"] = 2
    config["optimization"]["audit_states"] = 32
    config["terminal_audit"]["window_1_steps"] = [0, 2]
    config["terminal_audit"]["window_2_steps"] = [2, 4]
    config["seeds"]["held_out_formal"] = [0]
    return config


def test_formal_protocol_is_revision_4_and_quartic_is_not_active() -> None:
    config = load_config(CONFIG)
    validate_config(config, "formal")
    assert PROTOCOL_REVISION == 4
    assert tuple(config["methods"]) == FORMAL_METHODS
    assert HISTORICAL_EXCLUDED_METHODS == ("reciprocal_quartic_distance",)
    assert "reciprocal_quartic_distance" not in config["methods"]
    assert config["optimization"]["negative_alpha"] == pytest.approx(0.5)
    assert config["optimization"]["rarity_logit_anchor_coefficient"] == pytest.approx(0.25)
    assert config["taper"]["reference_rare_retention"] == pytest.approx(0.25)
    assert config["seeds"]["held_out_formal"] == list(range(200, 220))


def test_formal_config_tampering_fails_closed() -> None:
    config = load_config(CONFIG)
    config["optimization"]["negative_alpha"] = 0.25
    with pytest.raises(RuntimeError, match="negative_alpha"):
        validate_config(config, "formal")
    config = load_config(CONFIG)
    config["methods"].append("reciprocal_quartic_distance")
    with pytest.raises(ValueError, match="six-method"):
        validate_config(config, "formal")


def test_environment_has_exact_observed_cartesian_cells_and_hidden_rare_channel() -> None:
    config = tiny_config()
    environment = CartesianSemanticEnvironment(config, seed=0)
    audit = environment.audit()
    assert audit["passed"] is True
    assert environment.observed_action_count == 64
    assert environment.hidden_action_count == 16
    assert environment.action_count == 80
    assert audit["hidden_actions_share_rare_coordinate"] is True
    for split_name in ("train", "test"):
        split = getattr(environment, split_name)
        assert torch.equal(split["useful_common"] // 2, split["useful_rare"] // 2)
        assert torch.equal(split["unhelpful_common"] // 2, split["unhelpful_rare"] // 2)
        assert torch.all(split["hidden_optimal_actions"] >= environment.observed_action_count)
        advantages = torch.stack([split[f"{cell}_advantage"] for cell in CELL_NAMES], dim=1)
        assert torch.equal(advantages, torch.full_like(advantages, -1.0))


def test_geometry_audit_proves_positive_neutrality_utility_signs_and_support_cost() -> None:
    config = tiny_config()
    environment = CartesianSemanticEnvironment(config, seed=1)
    torch.manual_seed(1)
    model = CartesianPolicy(config, environment)
    cache_reference_directions(model, environment)
    audit = policy_geometry_audit(model, environment, config)
    assert audit["passed"] is True
    assert audit["positive_rarity_gradient_norm"] <= 1.0e-6
    assert audit["utility_oracle_sign_valid_fraction"] >= 0.995
    assert audit["rarity_shift_reward_drop"] >= 0.005
    assert audit["rarity_shift_hidden_probability_drop"] >= 0.001
    assert audit["useful_rare_to_common_shared_rarity_gradient_ratio"] >= 5.0
    assert audit["unhelpful_rare_to_common_shared_rarity_gradient_ratio"] >= 5.0


def test_positive_objective_is_exactly_rarity_neutral() -> None:
    config = tiny_config()
    environment = CartesianSemanticEnvironment(config, seed=2)
    torch.manual_seed(2)
    model = CartesianPolicy(config, environment)
    cache_reference_directions(model, environment)
    index = torch.arange(16)
    positive, _, _ = cell_log_probs(model, environment, environment.train, index)
    params = tuple(model.rarity_residual_head.parameters())
    grads = torch.autograd.grad(-positive.mean(), params, allow_unused=True)
    norm = math.sqrt(
        sum(float(grad.detach().double().square().sum()) for grad in grads if grad is not None)
    )
    assert norm <= 1.0e-6


def test_calibration_and_tapers_share_reference_points() -> None:
    config = tiny_config()
    environment = CartesianSemanticEnvironment(config, seed=3)
    torch.manual_seed(3)
    model = CartesianPolicy(config, environment)
    cache_reference_directions(model, environment)
    calibration = coordinate_calibration(model, environment, config)
    assert calibration["rare_minus_common_median"] == pytest.approx(4.0, abs=1.0e-5)
    coefficients = taper_coefficients(0.25)
    assert "reciprocal_quartic_distance" not in coefficients
    zero = torch.tensor([0.0])
    one = torch.tensor([1.0])
    for family, coefficient in coefficients.items():
        assert taper_weight(zero, family, coefficient).item() == pytest.approx(1.0)
        assert taper_weight(one, family, coefficient).item() == pytest.approx(0.25)


def test_tiny_run_preserves_environment_validity_and_shared_initial_state() -> None:
    config = tiny_config()
    bundle = run_seed_bundle(config, 0, method_specs(config["methods"]), "cpu")
    assert bundle["audit"]["passed"] is True
    assert bundle["audit"]["positive_warm_start"]["steps"] == 0
    assert bundle["audit"]["positive_warm_start"]["parameter_delta_norm"] == pytest.approx(0.0)
    assert len(bundle["summaries"]) == 6
    assert {row["method"] for row in bundle["summaries"]} == set(FORMAL_METHODS)
    assert all(row["environment_validity_failure"] is False for row in bundle["summaries"])
    assert all(row["minimum_utility_oracle_sign_valid_fraction"] >= 0.995 for row in bundle["summaries"])


def test_counterfactual_common_shift_harms_hidden_probability_and_task_reward() -> None:
    config = tiny_config()
    environment = CartesianSemanticEnvironment(config, seed=4)
    torch.manual_seed(4)
    model = CartesianPolicy(config, environment)
    cache_reference_directions(model, environment)
    calibration = coordinate_calibration(model, environment, config)
    metrics = evaluate(model, environment, environment.test, calibration)
    assert metrics["counterfactual_common_shift_reward_delta"] < 0.0
    assert metrics["counterfactual_common_shift_hidden_probability_delta"] < 0.0


def test_cpu_smoke_writes_six_method_outputs(tmp_path: Path) -> None:
    output = tmp_path / "smoke"
    rc = main(
        [
            "--config",
            str(CONFIG),
            "--output-root",
            str(output),
            "--stage",
            "smoke",
            "--device",
            "cpu",
        ]
    )
    assert rc == 0
    complete = json.loads((output / "RUN_COMPLETE.json").read_text())
    manifest = json.loads((output / "run_manifest.json").read_text())
    freeze = json.loads((output / "formal_protocol_freeze.json").read_text())
    mechanism = json.loads((output / "mechanism_summary.json").read_text())
    taper = json.loads((output / "taper_summary.json").read_text())
    assert complete["formal_result"] is False
    assert complete["actual_runs"] == 6
    assert manifest["quartic_excluded_from_active_matrix"] is True
    assert manifest["methods"] == list(FORMAL_METHODS)
    assert freeze["negative_alpha"] == pytest.approx(0.5)
    assert freeze["selection_not_conditioned_on_exponential_winning"] is True
    assert freeze["historical_excluded_methods"] == ["reciprocal_quartic_distance"]
    assert mechanism["all_environment_audits_passed"] is True
    assert taper["quartic_active"] is False
    assert (output / "checkpoints" / "seed_0" / "CHECKPOINT_COMPLETE.json").is_file()


def test_registry_and_materialized_handoff_register_revision_4() -> None:
    registry = yaml.safe_load((REPO_ROOT / "experiments" / "registry.yaml").read_text())
    entry = next(item for item in registry["experiments"] if item["id"] == "D-U1-E6-CARTESIAN-TAPER-01")
    assert entry["protocol_revision"] == 4
    assert entry["formal_parameter_freeze"] is True
    assert entry["execution_gate"]["state"] == "ready"
    assert entry["formal_execution"]["activation_state"] == "active"
    assert entry["protocol"]["methods"] == list(FORMAL_METHODS)
    assert entry["evidence"]["development_calibration_complete"] is True
    assert entry["evidence"]["formal_protocol_frozen"] is True
    handoff = (REPO_ROOT / "docs" / "handoff.md").read_text()
    delta = (
        REPO_ROOT
        / "docs"
        / "handoff_deltas"
        / "DU1-E6-REV4-FORMAL-FREEZE-2026-07-03"
        / "HANDOFF_DELTA.yaml"
    ).read_text()
    # Source packages carry the authoritative delta; trusted normalization
    # materializes the same content into handoff.md during application.
    authority_text = handoff if "v74-du1-e6-rev4-formal-freeze" in handoff else delta
    assert "v74（D-U1 E6 revision-4 正式协议冻结版）" in authority_text
    assert "v74-du1-e6-rev4-formal-freeze" in authority_text
