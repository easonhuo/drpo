from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

from drpo.du1_e6_semantic import (
    EXPERIMENT_ID,
    SemanticEnvironment,
    SemanticPolicy,
    controlled_gradient,
    gradient_branches,
    main,
    smoke_config,
    validate_config,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "du1_e6_semantic_pilot.yaml"


def load_config() -> dict:
    value = yaml.safe_load(CONFIG_PATH.read_text())
    assert isinstance(value, dict)
    return value


def tiny_config() -> dict:
    return smoke_config(load_config())


def test_environment_invariants_and_equal_negative_advantage() -> None:
    config = tiny_config()
    environment = SemanticEnvironment(config, seed=0, embedding_mode="aligned")
    audit = environment.audit()
    assert audit["passed"] is True
    for split in (environment.train, environment.test):
        assert not torch.any(split["positive"] == split["hidden"][:, None])
        assert not torch.any(split["far"] == split["hidden"][:, None])
        assert not torch.any(split["local"] == split["hidden"])
        all_negative = torch.cat([split["local_advantage"][:, None], split["far_advantage"]], dim=1)
        assert torch.equal(all_negative, torch.full_like(all_negative, -1.0))


def test_shuffled_policy_semantics_preserve_catalogue_but_break_alignment() -> None:
    config = tiny_config()
    aligned = SemanticEnvironment(config, seed=2, embedding_mode="aligned")
    shuffled = SemanticEnvironment(config, seed=2, embedding_mode="shuffled")
    assert torch.equal(aligned.reward_embeddings, shuffled.reward_embeddings)
    assert torch.equal(aligned.train["hidden"], shuffled.train["hidden"])
    assert torch.allclose(shuffled.policy_embeddings.norm(dim=1), torch.ones(16))
    assert not torch.equal(shuffled.policy_embeddings, shuffled.reward_embeddings)
    assert shuffled.audit()["policy_mapping_changed_when_shuffled"] is True


def test_far_cap_and_budget_matched_global_use_raw_gradient_budget() -> None:
    config = tiny_config()
    environment = SemanticEnvironment(config, seed=0, embedding_mode="aligned")
    torch.manual_seed(0)
    model = SemanticPolicy(config, concentration_mode="learnable")
    split = environment.train
    index = torch.arange(16)
    positive_grad, local_grad, far_grad, _ = gradient_branches(
        model,
        split["states"][index],
        split["positive"][index],
        split["local"][index],
        split["far"][index],
        environment.policy_embeddings,
    )
    _, capped = controlled_gradient(
        "far_cap",
        positive_grad,
        local_grad,
        far_grad,
        alpha=0.5,
        far_lambda=1.0,
        far_cap_ratio=1.0,
    )
    _, matched = controlled_gradient(
        "budget_matched_global",
        positive_grad,
        local_grad,
        far_grad,
        alpha=0.5,
        far_lambda=1.0,
        far_cap_ratio=1.0,
    )
    assert capped["far_cap_scale"] <= 1.0
    assert (
        capped["far_cap_scale"] * capped["weighted_far_gradient_norm"]
        <= capped["far_cap_target_norm"] + 1.0e-6
    )
    assert matched["global_budget_match_error"] <= 1.0e-6
    assert matched["controlled_negative_gradient_norm"] == pytest.approx(
        capped["controlled_negative_gradient_norm"], abs=1.0e-6
    )


def test_formal_gate_fails_closed() -> None:
    config = load_config()
    assert config["experiment_id"] == EXPERIMENT_ID
    with pytest.raises(RuntimeError, match="blocked"):
        validate_config(config, "formal")


def test_cpu_smoke_writes_complete_nonformal_outputs(tmp_path: Path) -> None:
    output = tmp_path / "smoke"
    rc = main(
        [
            "--config",
            str(CONFIG_PATH),
            "--stage",
            "smoke",
            "--output-root",
            str(output),
            "--device",
            "cpu",
        ]
    )
    assert rc == 0
    complete = json.loads((output / "RUN_COMPLETE.json").read_text())
    terminal = json.loads((output / "terminal_audit.json").read_text())
    summaries = json.loads((output / "per_run_summary.json").read_text())
    assert complete["completed"] is True
    assert complete["formal_result"] is False
    assert complete["scientific_status"] == "pilot"
    assert terminal["formal_scientific_acceptance"] is False
    assert complete["actual_runs"] == complete["expected_runs"]
    assert len(summaries) == complete["expected_runs"]
    assert all(item["nan_inf_numerical_failure"] is False for item in summaries)
    assert (output / "pilot_freeze_recommendation.json").exists()


def test_registry_preregistration_preserves_canonical_channel_and_history() -> None:
    registry = yaml.safe_load((REPO_ROOT / "experiments" / "registry.yaml").read_text())
    canonical_ids = [item["id"] for item in registry["experiments"]]
    development = {item["id"]: item for item in registry["development_experiment_registrations"]}
    assert "D-U1-E6-SEMANTIC-PILOT-01" not in canonical_ids
    assert "D-U1-E6-SEMANTIC-LONGRUN-01" not in canonical_ids
    assert development["D-U1-E6-SEMANTIC-PILOT-01"]["status"] == "pilot"
    longrun = development["D-U1-E6-SEMANTIC-LONGRUN-01"]
    assert longrun["status"] == "not_run"
    assert longrun["formal_execution"]["activation_state"] == "blocked"
    assert longrun["formal_execution"]["entrypoint_status"] == "planned"
    assert longrun["formal_execution"]["launch_mode"] == "canonical_guard"
    handoff = (REPO_ROOT / "docs" / "handoff.md").read_text()
    assert "v37（D-U1 E5 长程复核闭环版）" in handoff


def test_focused_config_supports_explicit_pressure_settings_and_two_x_audit() -> None:
    from drpo.du1_e6_semantic import FOCUSED_EXPERIMENT_ID, run_specs

    path = REPO_ROOT / "configs" / "du1_e6_semantic_focused_dev.yaml"
    config = yaml.safe_load(path.read_text())
    assert config["experiment_id"] == FOCUSED_EXPERIMENT_ID
    validate_config(config, "pilot")
    specs = run_specs(config)
    assert len(specs) == 11
    assert sum(spec.protocol == "E6-A" for spec in specs) == 4
    assert sum(spec.protocol == "E6-B" for spec in specs) == 7
    assert {spec.alpha for spec in specs if spec.protocol == "E6-B"} == {
        0.0,
        0.005,
        0.01,
        0.02,
        0.05,
        0.1,
        0.2,
    }
    assert config["terminal_audit"]["mode"] == "focused_two_x_windows"
    assert config["seeds"]["held_out_formal"] == []


def test_focused_terminal_audit_accepts_stable_nonzero_gradient() -> None:
    from drpo.du1_e6_semantic import terminal_classification

    config = yaml.safe_load(
        (REPO_ROOT / "configs" / "du1_e6_semantic_focused_dev.yaml").read_text()
    )
    trajectory = []
    for step in range(0, 4001, 50):
        trajectory.append(
            {
                "step": step,
                "nan_inf_numerical_failure": False,
                "support_or_temperature_boundary": False,
                "test_expected_semantic_reward": 0.88 + 1.0e-6 * step,
                "test_hidden_optimal_probability": 0.20 + 1.0e-6 * step,
                "test_normalized_semantic_extrapolation": 0.90 + 2.0e-6 * step,
                "test_entropy_mean": 1.80 - 1.0e-6 * step,
                "audit_raw_total_gradient_norm": 0.50,
                "adam_parameter_update_norm": 0.01,
            }
        )
    result = terminal_classification(trajectory, config)
    assert result["class"] == "focused_terminal_plateau"
    assert result["raw_total_gradient_medians"] == pytest.approx([0.5, 0.5])
    assert result["formal_acceptance"] is False


def test_focused_phase2_config_follows_preregistered_selection() -> None:
    from drpo.du1_e6_semantic import run_specs

    config = yaml.safe_load(
        (REPO_ROOT / "configs" / "du1_e6_semantic_focused_dev_phase2.yaml").read_text()
    )
    validate_config(config, "pilot")
    evidence = config["focused_extension"]["phase1_selection_evidence"]
    assert evidence["selected_local_alpha"] == 0.1
    assert evidence["alpha_0_1_support_events"] == 0
    specs = run_specs(config)
    assert len(specs) == 22
    far_values = {spec.far_lambda for spec in specs if spec.far_lambda > 0}
    assert far_values == {0.01, 0.02, 0.05, 0.1, 0.2}
    assert all(spec.alpha in {0.0, 0.1} for spec in specs)


def test_focused_development_result_closure_and_formal_gate() -> None:
    registry = yaml.safe_load((REPO_ROOT / "experiments" / "registry.yaml").read_text())
    development = {item["id"]: item for item in registry["development_experiment_registrations"]}
    focused = development["D-U1-E6-SEMANTIC-FOCUSED-DEV-01"]
    assert focused["status"] == "pilot"
    assert focused["execution"]["state"] == "delivered"
    assert focused["result"]["total_actual_runs"] == 165
    assert focused["result"]["nan_inf_numerical_failure_count"] == 0
    assert focused["result"]["selected_learnable_local_alpha"] == 0.1
    assert focused["result"]["support_transition_far_lambda"] == 0.02
    assert focused["formal_freeze_recommendation"]["automatic_freeze_allowed"] is False
    assert focused["formal_freeze_recommendation"]["maximum_steps"] == 8000
    longrun = development["D-U1-E6-SEMANTIC-LONGRUN-01"]
    assert longrun["formal_execution"]["activation_state"] == "blocked"
    assert longrun["formal_parameter_freeze"] is False
    assert "explicit_user_approval_of_D-U1-E6-SEMANTIC-FOCUSED-DEV-01_freeze" in longrun["blocked_by"]
    summary = REPO_ROOT / "outputs" / "du1_e6_semantic_focused_dev" / "FOCUSED_DEV_SUMMARY.md"
    assert summary.exists()
    handoff = (REPO_ROOT / "docs" / "handoff.md").read_text()
    assert "v43（D-U1 E6 聚焦开发扩展结果审计版）" in handoff
    assert "v37（D-U1 E5 长程复核闭环版）" in handoff
