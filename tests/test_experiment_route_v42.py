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


def test_e7_mechanism_is_implemented_but_formal_launch_is_blocked() -> None:
    entry = _experiments()["EXT-H-E7-Q2"]
    assert entry["status"] == "not_run"
    assert entry["scientific_status"] == "not_run"
    assert entry["implementation_state"] == "implemented"
    assert entry["implementation_commit"] == ("f64452a7452274a183b03c87c39b847039230c00")
    assert entry["execution_gate"]["state"] == "blocked"
    assert entry["execution_gate"]["blocked_by"] == ["D-U1-E6-TAPER-01"]
    assert entry["formal_execution"]["activation_state"] == "blocked"
    assert entry["formal_execution"]["entrypoint_status"] == "implemented"
    assert entry["formal_execution"]["entrypoint"] == "src/drpo/e7_hopper_q2.py"
    assert (ROOT / entry["formal_execution"]["entrypoint"]).is_file()


def test_unimplemented_external_scale_entries_are_planned_and_fail_closed() -> None:
    experiments = _experiments()
    for experiment_id in ["EXT-H-E7-BENCH-01", "EXT-C-E8-SCALE-01"]:
        entry = experiments[experiment_id]
        assert entry["status"] == "not_run"
        assert entry["implementation_state"] == "not_implemented"
        assert entry["execution_gate"]["state"] == "blocked"
        assert entry["execution_gate"].get("blocked_by")
        assert entry["execution_gate"].get("blocking_reason")
        assert entry["formal_execution"]["activation_state"] == "blocked"
        assert entry["formal_execution"]["entrypoint_status"] == "planned"
        assert entry["formal_execution"]["entrypoint"] is None


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
    assert bench["shortlist_rule"] == ("freeze_after_E4_and_E6_taper_without_D4RL_retuning")


def test_e8_scale_keeps_mechanism_and_scale_roles_separate() -> None:
    experiments = _experiments()
    mechanism = experiments["EXT-C-E8-V4.2"]
    scale = experiments["EXT-C-E8-SCALE-01"]
    assert mechanism["execution_class"] == "pilot"
    assert mechanism["status"] == "not_run"
    assert scale["scaling_plan"]["mechanism_owner"] == "EXT-C-E8-V4.2"
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
    assert "不是原 DRPO 分布鲁棒章节中的 linear weighting" in handoff
