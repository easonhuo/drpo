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
