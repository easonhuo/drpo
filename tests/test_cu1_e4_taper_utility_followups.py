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


def test_near_result_is_deposited_and_budget_match_is_the_only_active_successor() -> None:
    experiments = _experiments()
    taper = experiments["C-U1-E4-TAPER-01"]
    followup = taper["followup_evidence_requirements"]
    assert followup["registered_followup_experiments"] == FOLLOWUPS
    assert followup["local_execution_order"] == (
        "near_retention_then_budget_match_then_convergence_then_confirmation"
    )

    near = experiments[FOLLOWUPS[0]]
    assert near["status"] == "finite_step_validated"
    assert near["scientific_status"] == "finite_step_validated"
    assert near["evidence"]["completed_runs"] == 280
    assert near["evidence"]["unresolved_at_maximum_steps"] == 260
    assert near["paper_use"]["suitable_for_terminally_stable_method_ranking"] is False

    budget = experiments[FOLLOWUPS[1]]
    assert budget["status"] == "not_run"
    assert budget["implementation_state"] == "implemented"
    assert budget["execution_gate"]["state"] == "ready"
    assert budget["formal_execution"]["activation_state"] == "active"
    assert budget["formal_execution"]["entrypoint_status"] == "implemented"
    assert budget["formal_execution"]["entrypoint"] == (
        "src/drpo/cu1_taper_budget_match_formal.py"
    )
    assert budget["budget_contract"]["primary_mode"] == (
        "stepwise_raw_negative_gradient_l2_before_Adam"
    )
    assert budget["budget_contract"]["Adam_parameter_update_norm_matched"] is False
    assert budget["protocol"]["formal_paired_seeds"] == list(range(110, 130))
    assert budget["evidence"]["formal_run_started"] is False

    for experiment_id in FOLLOWUPS[2:]:
        row = experiments[experiment_id]
        assert row["status"] == "not_run"
        assert row["scientific_status"] == "not_run"
        assert row["implementation_state"] == "not_implemented"
        assert row["execution_gate"]["state"] == "blocked"
        assert row["execution_gate"]["blocked_by"]
        assert row["formal_execution"]["activation_state"] == "blocked"
        assert row["formal_execution"]["entrypoint_status"] == "planned"
        assert row["evidence"]["run_started"] is False
        assert row["no_method_winner_assumed"] is True

def test_followup_dependency_chain_and_seed_firewall_are_explicit() -> None:
    experiments = _experiments()
    near = experiments[FOLLOWUPS[0]]
    budget = experiments[FOLLOWUPS[1]]
    conv = experiments[FOLLOWUPS[2]]
    confirm = experiments[FOLLOWUPS[3]]

    assert near["protocol"]["formal_paired_seeds"] == list(range(90, 110))
    assert budget["predecessor"] == FOLLOWUPS[0]
    assert budget["protocol"]["formal_paired_seeds"] == list(range(110, 130))
    assert budget["protocol"]["confirmation_seeds_130_149_access"] == "forbidden"
    assert conv["predecessors"] == FOLLOWUPS[:2]
    assert conv["terminal_contract"]["continuation_seeds"] == list(range(110, 130))
    assert conv["terminal_contract"]["maximum_total_steps"] == 32000
    assert conv["terminal_contract"]["Adam_optimizer_state_continuity"] == "required"
    assert confirm["predecessor"] == FOLLOWUPS[2]
    assert confirm["confirmation_contract"]["exact_untouched_seeds"] == list(range(130, 150))
    assert confirm["confirmation_contract"]["seed_access_before_frozen_confirmation_config"] == "forbidden"
    assert confirm["confirmation_contract"][
        "hyperparameter_retuning_after_confirmation_start"
    ] == "forbidden"

def test_handoff_preserves_history_and_records_v63_closure_route() -> None:
    handoff = (ROOT / "docs" / "handoff.md").read_text()
    assert "v63（E4-TAPER Near-Retention 结果沉淀与闭环协议版）" in handoff
    assert "v60" in handoff
    assert "Quadratic bounded influence 与 Exponential vanishing influence" in handoff
    assert "Exponential 的价值在于" in handoff
    assert "不假设效用按指数速度下降" in handoff
    assert "Quadratic 权重本身趋零" in handoff
    assert "一般非零的有界常数" in handoff
    for experiment_id in FOLLOWUPS:
        assert f"`{experiment_id}`" in handoff
    assert "几何 robustness extension 保持低优先级" in handoff
    assert "每一步、Adam 之前的 raw negative-gradient L2 norm" in handoff
    assert "seeds `130--149`" in handoff
    assert "260/280" in handoff
