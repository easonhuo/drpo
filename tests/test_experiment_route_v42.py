from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _registry() -> dict:
    value = yaml.safe_load((ROOT / "experiments" / "registry.yaml").read_text())
    assert isinstance(value, dict)
    return value


def _experiments() -> dict[str, dict]:
    return {row["id"]: row for row in _registry()["experiments"]}


def _development() -> dict[str, dict]:
    return {row["id"]: row for row in _registry()["development_experiment_registrations"]}


def test_e4_taper_records_delivered_finite_step_result() -> None:
    taper = _experiments()["C-U1-E4-TAPER-01"]
    assert taper["status"] == "finite_step_validated"
    assert taper["execution_gate"]["state"] == "ready"
    assert taper["formal_execution"]["activation_state"] == "active"
    assert taper["formal_execution"]["entrypoint_status"] == "implemented"
    assert "--experiment-id C-U1-E4-TAPER-01" in taper["formal_launch_template"]
    assert "scripts/run_experiment_guard_hardened.py" in taper["formal_launch_template"]
    assert taper["execution"]["state"] == "delivered"
    assert taper["execution"]["process_exit_code"] == 0
    assert taper["evidence"]["raw_complete"] is True
    assert taper["evidence"]["terminal_audit_all_checks_passed"] is False
    assert taper["evidence"]["actual_runs"] == 220
    assert taper["evidence"]["scientific_status"] == "finite_step_validated"


def test_e6_longrun_is_delivered_and_taper_remains_review_blocked() -> None:
    experiments = _experiments()
    development = _development()
    pilot = development["D-U1-E6-SEMANTIC-PILOT-01"]
    longrun = experiments["D-U1-E6-SEMANTIC-LONGRUN-01"]
    taper = development["D-U1-E6-TAPER-01"]
    assert pilot["status"] == "pilot"
    assert pilot["protocol"]["development_seeds"] == [0, 1, 2, 3, 4]
    assert pilot["execution"]["state"] == "delivered"
    assert pilot["pilot_result"]["actual_runs"] == 105
    assert pilot["evidence"]["repository_applied"] is True
    assert pilot["evidence"]["applied_commit"] == ("2e04f6dba6d4e87f61920bedb1c464656906bf2b")
    assert longrun["status"] == "long_run_validated"
    assert longrun["parent_claim_closure"]["state"] == "closed"
    assert longrun["parent_claim_closure"]["required_for_parent_closure"] is False
    assert longrun["parent_claim_closure"]["required_before_EXT_H_E7_Q2"] is False
    assert longrun["next_gate"]["experiment_id"] == "EXT-H-E7-Q2"
    assert longrun["next_gate"]["state"] == "ready"
    assert longrun["implementation_state"] == "implemented"
    assert longrun["execution_gate"]["state"] == "blocked"
    assert longrun["formal_execution"]["activation_state"] == "blocked"
    assert longrun["execution"]["state"] == "delivered"
    assert longrun["evidence"]["actual_runs"] == 360
    assert longrun["formal_execution"]["entrypoint_status"] == "implemented"
    assert longrun["formal_execution"]["entrypoint"] == ("src/drpo/du1_e6_semantic_longrun.py")
    assert (ROOT / longrun["formal_execution"]["entrypoint"]).is_file()
    assert taper["status"] == "not_run"
    assert taper["implementation_state"] == "not_implemented"
    assert taper["formal_execution"]["activation_state"] == "blocked"
    assert taper["formal_execution"]["entrypoint_status"] == "planned"
    assert taper["predecessor_delivery_satisfied"] is True
    assert "delivered_D-U1-E6-SEMANTIC-LONGRUN-01" not in taper["blocked_by"]
    assert taper["blocked_by"]


def test_e7_mechanism_is_delivered_and_closed_to_unregistered_reruns() -> None:
    entry = _experiments()["EXT-H-E7-Q2"]
    assert entry["status"] == "long_run_validated"
    assert entry["scientific_status"] == "long_run_validated"
    assert entry["implementation_state"] == "implemented"
    assert entry["implementation_commit"] == ("f64452a7452274a183b03c87c39b847039230c00")
    assert entry["execution_gate"]["state"] == "blocked"
    assert entry["execution_gate"]["blocked_by"] == [
        "completed_formal_execution_no_rerun_without_new_registration"
    ]
    assert entry["formal_execution"]["activation_state"] == "blocked"
    assert entry["formal_execution"]["entrypoint_status"] == "implemented"
    assert entry["formal_execution"]["entrypoint"] == "src/drpo/e7_hopper_q2.py"
    assert (ROOT / entry["formal_execution"]["entrypoint"]).is_file()


def test_external_scale_entry_remains_planned_and_e7_bench_pilot_is_implemented() -> None:
    experiments = _experiments()
    scale = experiments["EXT-C-E8-SCALE-01"]
    assert scale["status"] == "not_run"
    assert scale["implementation_state"] == "not_implemented"
    assert scale["execution_gate"]["state"] == "blocked"
    assert scale["formal_execution"]["activation_state"] == "blocked"
    assert scale["formal_execution"]["entrypoint_status"] == "planned"
    assert scale["formal_execution"]["entrypoint"] is None

    bench = experiments["EXT-H-E7-BENCH-01"]
    assert bench["status"] == "not_run"
    assert bench["implementation_state"] == "pilot_implemented_formal_parallel_scaffold"
    assert bench["execution_gate"]["state"] == "blocked"
    assert bench["formal_execution"]["activation_state"] == "blocked"
    assert bench["formal_execution"]["entrypoint_status"] == "implemented"
    assert bench["formal_execution"]["entrypoint"] == "src/drpo/e7_bench.py"
    assert (ROOT / bench["formal_execution"]["entrypoint"]).is_file()
    assert bench["pilot_execution"]["execution_gate"]["state"] == "ready"


def test_e7_benchmark_scope_is_exactly_nine_locomotion_tasks() -> None:
    bench = _experiments()["EXT-H-E7-BENCH-01"]
    suite = bench["suite"]
    assert suite["name"] == "D4RL_MuJoCo_locomotion"
    assert suite["environments"] == ["hopper", "walker2d", "halfcheetah"]
    assert suite["dataset_qualities"] == [
        "medium",
        "medium_replay",
        "medium_expert",
    ]
    assert suite["task_count"] == 9
    assert bench["execution_gate"]["blocked_by"] == ["formal_protocol_lock"]
    assert bench["prerequisite_status"]["EXT-H-E7-Q2"] == (
        "satisfied_long_run_validated"
    )
    assert bench["prerequisite_status"]["controlled_method_shortlist_freeze"] == (
        "satisfied_without_d4rl_retuning"
    )
    assert bench["pilot_execution"]["development_seeds"] == [200, 201, 202, 203]
    parallel = bench["pilot_execution"]["parallel_execution"]
    assert parallel["parallel_unit"] == "dataset_seed_method"
    assert parallel["critic_workers"] == 2
    # Backward-compatible aliases remain present for Stage 3 schema stability.
    assert parallel["positive_workers"] == 8
    assert parallel["warmstart_workers"] == 8
    assert parallel["positive_cpus_per_worker"] == 32
    assert parallel["warmstart_cpus_per_worker"] == 32
    assert parallel["branch_workers"] == 48
    assert parallel["branch_cpus_per_worker"] == 7
    assert parallel["peak_registered_cpu_threads"] == 336
    budget = bench["pilot_execution"]["fixed_budget"]
    assert budget["critic_optimizer_steps"] == 100_000
    assert budget["shared_positive_only_warmstart_steps"] == 100_000
    assert budget["method_continuation_steps_each"] == 200_000
    assert budget["total_actor_optimizer_steps_each"] == 300_000
    assert budget["positive_only_receives_equal_continuation"] is True
    assert parallel["serial_seed_loop_forbidden"] is True
    assert parallel["serial_method_loop_forbidden"] is True
    formal_parallel = bench["formal_parallel_contract"]
    assert formal_parallel["parallel_unit"] == "task_seed_method"
    assert formal_parallel["serial_seed_loop_forbidden"] is True
    assert formal_parallel["serial_method_loop_forbidden"] is True
    assert bench["shortlist_rule"] == (
        "freeze_after_E4_E6_core_closure_and_E7_mechanism_without_D4RL_retuning"
    )


def test_e8_offline_bank_precedes_scale_and_keeps_roles_separate() -> None:
    experiments = _experiments()
    prior = experiments["EXT-C-E8-V4.2"]
    mechanism = experiments["EXT-C-E8-V4.3"]
    bank = experiments["EXT-C-E8-V4.4-OFFLINE-BANK"]
    tuning = experiments["EXT-C-E8-V4.5-OFFLINE-BANK-TUNING"]
    online = experiments["EXT-C-E8-V4.6-ONLINE-OFFPOLICY-REPLAY"]
    taper = experiments["EXT-C-E8-TAPER-0.5B-01"]
    scale = experiments["EXT-C-E8-SCALE-01"]
    assert prior["execution_class"] == "superseded"
    assert prior["status"] == "superseded"
    assert prior["superseded_by"] == "EXT-C-E8-V4.3"
    assert mechanism["execution_class"] == "pilot"
    assert mechanism["status"] == "not_run"
    assert mechanism["methods"] == [
        "positive_only",
        "controlled_negative",
        "dynamic_controlled_negative",
        "uncontrolled_negative",
    ]
    assert bank["execution_class"] == "pilot"
    assert bank["status"] == "not_run"
    assert bank["predecessor"] == "EXT-C-E8-V4.3"
    assert bank["data"]["negative_bank_size_per_prompt"] == 16
    assert bank["data"]["online_rollout_during_method_training"] is False
    assert bank["methods"] == [
        "positive_only",
        "dynamic_controlled_negative",
        "bank_dynamic_controlled_negative",
        "bank_global_matched",
        "bank_uncontrolled_negative",
    ]
    assert tuning["execution_class"] == "pilot"
    assert tuning["status"] == "not_run"
    assert tuning["predecessor"] == "EXT-C-E8-V4.4-OFFLINE-BANK"
    assert tuning["parameterization"]["online_rollout_during_training"] is False
    assert tuning["tuning_protocol"]["stage_a_global_negative_strength"]["values"] == [
        0.5, 1.0, 1.5, 2.0,
    ]
    assert tuning["tuning_protocol"]["stage_b_exponential_taper"]["values"] == [
        0.3, 0.7, 1.2,
    ]
    assert tuning["confirmation_protocol"]["untouched_training_seeds"] == [
        3234, 4234, 5234,
    ]
    assert online["execution_class"] == "pilot"
    assert online["status"] == "not_run"
    assert online["predecessor"] == "EXT-C-E8-V4.5-OFFLINE-BANK-TUNING"
    assert online["design"]["cells"] == [
        "frozen_positive_only",
        "frozen_dynamic",
        "online_positive_only",
        "online_dynamic",
    ]
    assert online["online_replay_protocol"]["post_warmup_stale_fraction"] == 0.5
    assert online["confirmation_protocol"]["training_seeds"] == [6234, 7234, 8234]
    assert taper["execution_class"] == "pilot"
    assert taper["status"] == "not_run"
    assert taper["implementation_state"] == "implemented"
    assert taper["execution_gate"]["state"] == "ready"
    assert taper["code_entrypoint"] == "src/drpo/countdown_e8_taper.py"
    assert taper["operator_entrypoint"] == "scripts/run_countdown_e8_taper.py"
    remoteness = taper["learner_relative_remoteness"]
    assert remoteness["normalized_excess_formula"] == (
        "S_theta=max(0,(-log_pi_theta(x_given_s)-tau)/c_cal)"
    )
    assert remoteness["distance_formula"] == "d_theta=sqrt(S_theta)"
    assert taper["scope_decision"]["countdown_0_5b_mechanism_exploration"] == (
        "closed_for_current_scope"
    )
    assert taper["scope_decision"]["closure_is_result_upgrade"] is False
    assert taper["calibration_protocol"]["development_seed"] == 9134
    assert taper["confirmation_protocol"]["paired_training_seeds"] == [
        9234,
        10234,
        11234,
    ]
    assert [method["id"] for method in taper["methods"]] == [
        "positive_only",
        "uncontrolled_negative",
        "global_matched",
        "reciprocal_linear",
        "exponential",
        "squared_distance_exponential",
    ]
    assert scale["execution_gate"]["blocked_by"] == [
        "EXT-C-E8-TAPER-0.5B-01",
        "EXT-H-E7-BENCH-01",
    ]
    assert scale["scaling_plan"]["mechanism_owner"] == (
        "EXT-C-E8-V4.6-ONLINE-OFFPOLICY-REPLAY"
    )
    assert scale["scaling_plan"]["method_shortlist_owner"] == (
        "EXT-C-E8-TAPER-0.5B-01"
    )
    assert scale["scaling_plan"]["primary_model"] == "Qwen_Instruct_3B"
    assert scale["scaling_plan"]["frozen_confirmation_model"] == "Qwen_Instruct_7B"
    assert scale["scaling_plan"]["retune_method_family_on_scale_tasks"] is False


def test_handoff_preserves_v42_route_and_records_v45_taper_result() -> None:
    handoff = (ROOT / "docs" / "handoff.md").read_text()
    assert "v42 增量登记：状态机一致性、E7 已实现门禁与 E4--E8 路线锁定" in handoff
    assert "v45（E4-TAPER 结果闭环、环境识别与公平性边界版）" in handoff
    assert "ready gate + implemented entrypoint 必须 active" in handoff
    assert "E4-TAPER -> E6 -> E6-TAPER -> E7-MECH -> E7-BENCH -> E8-MECH -> E8-SCALE" in handoff
    assert "D4RL MuJoCo locomotion 9-task suite" in handoff
    assert "implemented + not_run + blocked" in handoff
    assert "v56 增量登记：E6 父 claim 关闭与 E7-MECH 路线解锁" in handoff
    assert "v57 增量登记：Countdown `EXT-C-E8-V4.4-OFFLINE-BANK`" in handoff
    assert "v59 增量登记：Countdown `EXT-C-E8-V4.5-OFFLINE-BANK-TUNING`" in handoff
    assert "EXT-C-E8-V4.5-OFFLINE-BANK-TUNING" in handoff
    assert "v62 增量登记：Countdown `EXT-C-E8-V4.6-ONLINE-OFFPOLICY-REPLAY`" in handoff
    assert "EXT-C-E8-V4.6-ONLINE-OFFPOLICY-REPLAY" in handoff
    assert "v67 增量登记：关闭 Countdown 0.5B 机制探索职责" in handoff
    assert "EXT-C-E8-TAPER-0.5B-01" in handoff
    assert "`EXT-H-E7-Q2` 仍是下一正式 route item" in handoff
    assert "它不再是 E6 父 claim 关闭或 E7-MECH 启动的前置条件" in handoff
    assert "不是原 DRPO 分布鲁棒章节中的 linear weighting" in handoff
