from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
import yaml

from drpo.du1_e6_conditional_gap import (
    FORMAL_EXPERIMENT_ID,
    ConditionalGapEnvironment,
    RunSpec,
    gradient_branches,
    main,
    smoke_config,
    training_gradient,
    validate_config,
)
from drpo.du1_e6_semantic import SemanticPolicy, controlled_gradient

REPO_ROOT = Path(__file__).resolve().parents[1]
DEV_CONFIG = REPO_ROOT / "configs" / "du1_e6_conditional_gap_dev.yaml"
FORMAL_CONFIG = REPO_ROOT / "configs" / "du1_e6_conditional_gap_longrun.yaml"


def load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text())
    assert isinstance(value, dict)
    return value


def test_large_structured_gap_and_paired_control_invariants() -> None:
    config = load(DEV_CONFIG)
    for mode in ("optimal_group_covered", "structured_gap"):
        environment = ConditionalGapEnvironment(config, seed=0, coverage_mode=mode)
        audit = environment.audit()
        assert audit["passed"] is True
        assert audit["splits"]["train"]["correct_group_reward_min"] == pytest.approx(0.85)
        assert audit["splits"]["train"]["correct_group_reward_max"] == pytest.approx(1.0)
        for split in (environment.train, environment.test):
            gap = split["gap_mask"]
            target = split["target_group"]
            observed = split["observed_group_mask"].gather(1, target[:, None]).squeeze(1)
            assert gap.float().mean().item() == pytest.approx(0.5)
            assert (~split["observed_group_mask"]).float().mean().item() == pytest.approx(0.625)
            assert torch.all(split["observed_group_mask"].sum(dim=1) == 3)
            assert not torch.any(split["positive"] == split["hidden"][:, None])
            assert torch.all(observed[~gap])
            if mode == "structured_gap":
                assert not torch.any(observed[gap])
            else:
                assert torch.all(observed[gap])


def test_pairing_preserves_target_and_nuisance_coordinates() -> None:
    environment = ConditionalGapEnvironment(
        load(DEV_CONFIG), seed=1, coverage_mode="structured_gap"
    )
    split = environment.train
    half = split["states"].shape[0] // 2
    assert torch.equal(split["target_group"][:half], split["target_group"][half:])
    assert torch.allclose(
        split["states"][:half, [0, 1, 2, 4, 5]],
        split["states"][half:, [0, 1, 2, 4, 5]],
    )
    assert torch.all(split["states"][:half, 3] < 0)
    assert torch.all(split["states"][half:, 3] > 0)


def test_fast_training_gradient_matches_explicit_branch_sum() -> None:
    config = smoke_config(load(DEV_CONFIG))
    environment = ConditionalGapEnvironment(config, 0, "structured_gap")
    index = torch.arange(16)
    for method in (
        "positive_only",
        "local_only",
        "near_zero",
        "uncontrolled",
        "far_cap",
        "budget_matched_global",
    ):
        torch.manual_seed(7)
        model = SemanticPolicy(config, "fixed")
        arguments = (
            model,
            environment.train["states"][index],
            environment.train["positive"][index],
            environment.train["local"][index],
            environment.train["far"][index],
            environment.policy_embeddings,
        )
        positive, local, far, _ = gradient_branches(*arguments)
        expected, _ = controlled_gradient(method, positive, local, far, 0.5, 4.0, 1.0)
        actual = training_gradient(
            *arguments,
            RunSpec("structured_gap", method, 0.5, 4.0),
            1.0,
        )
        for got, want in zip(actual, expected):
            if got is None or want is None:
                assert got is want
            else:
                assert torch.allclose(got, want, atol=1.0e-5, rtol=1.0e-5)


def test_formal_config_is_frozen_and_mutations_fail_closed() -> None:
    config = load(FORMAL_CONFIG)
    validate_config(config, "formal")
    assert config["experiment_id"] == FORMAL_EXPERIMENT_ID
    assert config["seeds"]["held_out_formal"] == list(range(130, 150))
    assert config["terminal_audit"]["formal_extension_factor"] == 8.0
    mutated = copy.deepcopy(config)
    mutated["data"]["gap_state_fraction"] = 0.75
    with pytest.raises((ValueError, RuntimeError), match="gap_state_fraction"):
        validate_config(mutated, "formal")


def test_smoke_is_nonformal_and_complete(tmp_path: Path) -> None:
    output = tmp_path / "smoke"
    assert (
        main(
            [
                "--config",
                str(DEV_CONFIG),
                "--stage",
                "smoke",
                "--output-root",
                str(output),
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    complete = json.loads((output / "RUN_COMPLETE.json").read_text())
    terminal = json.loads((output / "terminal_audit.json").read_text())
    assert complete["formal_result"] is False
    assert complete["actual_runs"] == complete["expected_runs"] == 2
    assert terminal["formal_scientific_acceptance"] is False


def test_registry_and_handoff_activate_only_the_new_formal_protocol() -> None:
    registry = yaml.safe_load((REPO_ROOT / "experiments" / "registry.yaml").read_text())
    canonical = {item["id"]: item for item in registry["experiments"]}
    development = {item["id"]: item for item in registry["development_experiment_registrations"]}
    formal = canonical[FORMAL_EXPERIMENT_ID]
    assert formal["status"] == "not_run"
    assert formal["execution_gate"]["state"] == "ready"
    assert formal["formal_execution"]["activation_state"] == "active"
    assert formal["held_out_seeds"] == list(range(130, 150))
    assert development["D-U1-E6-CONDITIONAL-GAP-DEV-01"]["status"] == "pilot"
    assert canonical["D-U1-E6-SEMANTIC-LONGRUN-01"]["status"] == "long_run_validated"
    taper = development["D-U1-E6-TAPER-01"]
    assert taper["predecessor"] == "D-U1-E6-SEMANTIC-LONGRUN-01"
    assert taper["predecessor_delivery_satisfied"] is True
    assert taper["additional_predecessor"] == FORMAL_EXPERIMENT_ID
    assert "D-U1-E6-CONDITIONAL-GAP-01_delivery" in taper["blocked_by"]
    handoff = (REPO_ROOT / "docs" / "handoff.md").read_text()
    assert "v48（D-U1 E6 大规模条件支持缺口协议与正式执行准备版）" in handoff
    assert "禁止称为 OOD generalization" in handoff


def test_compact_pilot_closure_remains_nonformal() -> None:
    root = REPO_ROOT / "outputs" / "du1_e6_conditional_gap_dev"
    terminal = json.loads((root / "terminal_audit.json").read_text())
    index = json.loads((root / "ARTIFACT_INDEX.json").read_text())
    assert terminal["actual_runs"] == terminal["expected_runs"] == 20
    assert terminal["task_performance_collapse_count"] == 4
    assert terminal["nan_inf_numerical_failure_count"] == 0
    assert terminal["formal_scientific_acceptance"] is False
    assert index["formal_result"] is False
