from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
FOLLOWUPS = [
    "C-U1-E4-TAPER-NEAR-RETENTION-01",
    "C-U1-E4-TAPER-BUDGET-MATCH-01",
    "C-U1-E4-TAPER-CONV-01",
    "C-U1-E4-TAPER-CONFIRM-01",
]


def _experiments() -> dict[str, dict]:
    registry = yaml.safe_load((ROOT / "experiments" / "registry.yaml").read_text())
    return {row["id"]: row for row in registry["experiments"]}


def test_taper_utility_extension_separates_bounded_and_vanishing_influence() -> None:
    taper = _experiments()["C-U1-E4-TAPER-01"]
    utility = taper["theory"]["utility_extension"]
    assert utility["status"] == "conditional_empirical_hypothesis"
    assert utility["universal_law_claimed"] is False
    assert utility["exact_decay_rate_claimed"] is False
    assert "may approach zero or become negative" in utility["hypothesis"]

    levels = taper["theory"]["influence_control_levels"]
    assert levels["quadratic_weight_itself_vanishes"] is True
    assert levels["quadratic_weighted_influence_limit"] == (
        "bounded_nonzero_constant_in_general"
    )
    assert levels["bounded_influence_condition"] == "w(d)=O(d^-2)"
    assert levels["vanishing_influence_condition"] == "w(d)=o(d^-2)"
    assert levels["exponential_role"] == (
        "smooth_vanishing_tail_candidate_not_unique_solution"
    )
    assert levels["current_E4_exponential_formula_unchanged"] is True
    assert levels["no_universal_method_winner_assumed"] is True


def test_four_followups_are_registered_in_order_but_not_runnable() -> None:
    experiments = _experiments()
    taper = experiments["C-U1-E4-TAPER-01"]
    followup = taper["followup_evidence_requirements"]
    assert followup["authorization_state"] == "user_approved_registered_not_runnable"
    assert followup["registered_followup_experiments"] == FOLLOWUPS
    assert followup["local_execution_order"] == (
        "near_retention_then_budget_match_then_convergence_then_confirmation"
    )
    assert followup["geometry_robustness_priority"] == (
        "low_optional_not_a_current_gate"
    )
    assert followup["no_automatic_execution"] is True

    for experiment_id in FOLLOWUPS:
        row = experiments[experiment_id]
        assert row["status"] == "not_run"
        assert row["scientific_status"] == "not_run"
        assert row["implementation_state"] == "not_implemented"
        assert row["execution_gate"]["state"] == "blocked"
        assert row["execution_gate"]["blocked_by"]
        assert row["formal_execution"]["activation_state"] == "blocked"
        assert row["formal_execution"]["entrypoint_status"] == "planned"
        assert row["evidence"]["run_started"] is False
        assert row["evidence"]["raw_complete"] is False
        assert row["no_method_winner_assumed"] is True


def test_followup_dependency_chain_and_seed_firewall_are_explicit() -> None:
    experiments = _experiments()
    near = experiments[FOLLOWUPS[0]]
    budget = experiments[FOLLOWUPS[1]]
    conv = experiments[FOLLOWUPS[2]]
    confirm = experiments[FOLLOWUPS[3]]

    assert near["seed_policy"]["seeds_70_89_reuse_as_confirmation"] == "forbidden"
    assert budget["predecessor"] == FOLLOWUPS[0]
    assert conv["predecessors"] == FOLLOWUPS[:2]
    assert confirm["predecessor"] == FOLLOWUPS[2]
    assert confirm["confirmation_contract"]["seeds_70_89"] == (
        "development_evidence_not_confirmatory"
    )
    assert confirm["confirmation_contract"][
        "hyperparameter_retuning_after_confirmation_start"
    ] == "forbidden"


def test_handoff_records_theory_upgrade_and_experiment_sequence() -> None:
    handoff = (ROOT / "docs" / "handoff.md").read_text()
    assert "v60（E4-TAPER 效用假设与公平实验登记版）" in handoff
    assert "Quadratic bounded influence 与 Exponential vanishing influence" in handoff
    assert "Exponential 的价值在于" in handoff
    assert "不假设效用按指数速度下降" in handoff
    assert "Quadratic 权重本身趋零" in handoff
    assert "一般非零的有界常数" in handoff
    for experiment_id in FOLLOWUPS:
        assert f"`{experiment_id}`" in handoff
    assert "几何 robustness extension 保持低优先级" in handoff
