from __future__ import annotations

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

from drpo.du1_e6_cartesian_taper import (  # noqa: E402
    ALL_METHODS,
    CELL_NAMES,
    CartesianPolicy,
    CartesianSemanticEnvironment,
    MethodSpec,
    active_cell_loss,
    aggregate,
    assign_task_collapse,
    cell_log_probs,
    coordinate_calibration,
    load_config,
    main,
    mechanism_report,
    method_specs,
    run_seed_bundle,
    taper_report,
    taper_coefficients,
    taper_weight,
    validate_config,
)

CONFIG = REPO_ROOT / "configs" / "du1_e6_cartesian_taper.yaml"


def tiny_config() -> dict:
    config = load_config(CONFIG)
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


def test_cartesian_environment_exactly_decouples_utility_and_rarity() -> None:
    config = tiny_config()
    environment = CartesianSemanticEnvironment(config, seed=0)
    audit = environment.audit()
    assert audit["passed"] is True
    for split_name in ("train", "test"):
        split = getattr(environment, split_name)
        assert torch.equal(split["useful_common"] // 2, split["useful_rare"] // 2)
        assert torch.equal(split["unhelpful_common"] // 2, split["unhelpful_rare"] // 2)
        assert torch.all(split["useful_common"] % 2 == 0)
        assert torch.all(split["useful_rare"] % 2 == 1)
        assert torch.all(split["unhelpful_common"] % 2 == 0)
        assert torch.all(split["unhelpful_rare"] % 2 == 1)
        advantages = torch.stack([split[f"{cell}_advantage"] for cell in CELL_NAMES], dim=1)
        assert torch.equal(advantages, torch.full_like(advantages, -1.0))
        assert torch.all(split["useful_utility"] > 0)
        assert torch.all(split["unhelpful_utility"] < 0)
        useful_reward = split["reward_matrix"][torch.arange(len(split["states"])), split["useful_common"]]
        useful_rare_reward = split["reward_matrix"][torch.arange(len(split["states"])), split["useful_rare"]]
        unhelpful_reward = split["reward_matrix"][torch.arange(len(split["states"])), split["unhelpful_common"]]
        unhelpful_rare_reward = split["reward_matrix"][torch.arange(len(split["states"])), split["unhelpful_rare"]]
        torch.testing.assert_close(useful_reward, useful_rare_reward)
        torch.testing.assert_close(unhelpful_reward, unhelpful_rare_reward)


def test_initial_surprisal_gap_is_exactly_the_frozen_logit_gap() -> None:
    config = tiny_config()
    environment = CartesianSemanticEnvironment(config, seed=1)
    torch.manual_seed(1)
    model = CartesianPolicy(config, environment)
    calibration = coordinate_calibration(model, environment, config)
    assert calibration["rare_minus_common_median"] == pytest.approx(
        config["policy"]["initial_rarity_logit_gap"], abs=1.0e-5
    )
    assert calibration["rare_surprisal_median"] > calibration["common_surprisal_median"]
    assert calibration["initial_cartesian_exact"] is True
    assert calibration["initial_common_surprisal_utility_axis_max_error"] <= 1.0e-6
    assert calibration["initial_rare_surprisal_utility_axis_max_error"] <= 1.0e-6


def test_rarity_roles_are_reassigned_from_current_logits() -> None:
    config = tiny_config()
    environment = CartesianSemanticEnvironment(config, seed=2)
    torch.manual_seed(2)
    model = CartesianPolicy(config, environment)
    index = torch.tensor([0])
    _, before, _ = cell_log_probs(model, environment, environment.train, index)
    useful_pair = environment.train["useful_pair"][0]
    initial_common = int(useful_pair[0])
    initial_rare = int(useful_pair[1])
    assert before["useful_common"].item() > before["useful_rare"].item()
    with torch.no_grad():
        model.action_bias[initial_common] = -10.0
        model.action_bias[initial_rare] = 10.0
    _, after, _ = cell_log_probs(model, environment, environment.train, index)
    logits, _ = model(environment.train["states"][index], environment.action_embeddings)
    log_probs = torch.log_softmax(logits, dim=-1)
    assert after["useful_common"].item() == pytest.approx(
        log_probs[0, initial_rare].item()
    )
    assert after["useful_rare"].item() == pytest.approx(
        log_probs[0, initial_common].item()
    )


def test_subset_interventions_zero_cells_without_renormalizing() -> None:
    cells = {name: torch.ones(4, requires_grad=True) for name in CELL_NAMES}
    calibration = {"threshold": 0.0, "scale": 1.0}
    coefficients = taper_coefficients(0.25)
    single, _ = active_cell_loss(
        cells,
        MethodSpec("useful_common_only", ("useful_common",)),
        calibration,
        coefficients,
        1.0,
    )
    all_negative, _ = active_cell_loss(
        cells,
        MethodSpec("all_negative", CELL_NAMES),
        calibration,
        coefficients,
        1.0,
    )
    assert single.item() == pytest.approx(0.25)
    assert all_negative.item() == pytest.approx(1.0)


def test_taper_families_match_common_and_reference_rare_retention() -> None:
    rho = 0.25
    coefficients = taper_coefficients(rho)
    zero = torch.tensor([0.0])
    one = torch.tensor([1.0])
    for family, coefficient in coefficients.items():
        assert taper_weight(zero, family, coefficient).item() == pytest.approx(1.0)
        assert taper_weight(one, family, coefficient).item() == pytest.approx(rho)
    assert coefficients["reciprocal_linear"] == pytest.approx(3.0)
    assert coefficients["reciprocal_quadratic"] == pytest.approx(3.0)
    assert coefficients["exponential"] == pytest.approx(math.log(4.0))


def test_formal_freeze_rejects_silent_protocol_changes() -> None:
    config = load_config(CONFIG)
    validate_config(config, "formal")
    tampered = load_config(CONFIG)
    tampered["taper"]["dynamic_rarity_role_assignment"] = False
    with pytest.raises(RuntimeError, match="dynamic_rarity_role_assignment"):
        validate_config(tampered, "formal")


def test_full_joint_matrix_builds_factorial_and_taper_contrasts() -> None:
    config = tiny_config()
    bundle = run_seed_bundle(config, 0, method_specs(), "cpu")
    summaries = list(bundle["summaries"])
    assign_task_collapse(summaries, config)
    aggregated = aggregate(summaries)
    mechanism = mechanism_report(summaries, aggregated)
    taper = taper_report(summaries, aggregated)
    assert set(mechanism["methods"]) == set(ALL_METHODS[:10])
    assert mechanism["paired_contrasts"]["rarity_effect_at_useful"][
        "final_expected_semantic_reward"
    ]["seeds"] == [0]
    assert taper["paired_contrasts"]["exponential_minus_global_matched"][
        "final_expected_semantic_reward"
    ]["seeds"] == [0]


def test_cpu_smoke_writes_joint_mechanism_and_taper_outputs(tmp_path: Path) -> None:
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
    aggregate = json.loads((output / "aggregate_summary.json").read_text())
    audit = json.loads((output / "environment_audits.json").read_text())
    freeze = json.loads((output / "formal_protocol_freeze.json").read_text())
    assert complete["completed"] is True
    assert complete["formal_result"] is False
    assert complete["actual_runs"] == 8
    assert audit[0]["passed"] is True
    assert set(aggregate) == {
        "positive_only",
        "useful_common_only",
        "useful_rare_only",
        "all_negative",
        "global_matched",
        "reciprocal_linear",
        "reciprocal_quadratic",
        "exponential",
    }
    assert freeze["four_cells"] == list(CELL_NAMES)
    mechanism = json.loads((output / "mechanism_summary.json").read_text())
    taper = json.loads((output / "taper_summary.json").read_text())
    assert mechanism["block"] == "E6_CARTESIAN_MECHANISM"
    assert "utility_x_rarity_interaction" in mechanism
    assert taper["block"] == "E6_TAPER_METHOD_COMPARISON"
    assert taper["no_method_winner_assumed"] is True
    assert (output / "checkpoints" / "seed_0" / "CHECKPOINT_COMPLETE.json").exists()
    assert (output / "checkpoints" / "seed_0" / "trajectories.jsonl").exists()
    assert (output / "checkpoints" / "seed_0" / "per_run_summary.json").exists()


def test_registry_and_handoff_register_joint_successor() -> None:
    registry = yaml.safe_load((REPO_ROOT / "experiments" / "registry.yaml").read_text())
    canonical = {item["id"]: item for item in registry["experiments"]}
    development = {
        item["id"]: item for item in registry["development_experiment_registrations"]
    }
    entry = canonical["D-U1-E6-CARTESIAN-TAPER-01"]
    assert entry["status"] == "not_run"
    assert entry["implementation_state"] == "implemented"
    assert entry["execution_gate"]["state"] == "ready"
    assert entry["formal_execution"]["activation_state"] == "active"
    assert entry["protocol"]["cartesian_cells"] == list(CELL_NAMES)
    assert entry["protocol"]["methods"] == list(ALL_METHODS)
    old = development["D-U1-E6-TAPER-01"]
    assert old["status"] == "not_run"
    assert old["implementation_state"] == "not_implemented"
    assert entry["supersedes_preregistration"] == "D-U1-E6-TAPER-01"
    handoff = (REPO_ROOT / "docs" / "handoff.md").read_text()
    assert "HANDOFF-DELTA-BLOCK:after_heading:v70-du1-e6-cartesian-taper:START" in handoff
    assert "D-U1-E6-CARTESIAN-TAPER-01" in handoff
    assert "v70-du1-e6-cartesian-taper-current-gate" in handoff
    assert "v70-du1-e6-cartesian-taper-execution-order" in handoff
