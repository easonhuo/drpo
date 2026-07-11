#!/usr/bin/env python3
"""Prepare registry and schema-v3 handoff delta for the E8 V2 milestone/sweep."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

GLOBAL_ID = "EXT-C-E8-ORACLE-OFFLINE-V2-GLOBAL-LOW-SCALE-SWEEP-0.5B-01"
TAPER_ID = "EXT-C-E8-ORACLE-OFFLINE-V2-TAPER-SWEEP-0.5B-01"
UPDATE_ID = "EXT-C-E8-V2-GLOBAL-MILESTONE-AND-TAPER-SWEEP-2026-07-11"


def dedent(block: str) -> str:
    return "\n".join(
        line[8:] if line.startswith("        ") else line for line in block.splitlines()
    ) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()
    root = Path(args.repo_root).resolve()
    base = args.base
    registry = root / "experiments/registry.yaml"
    before_registry = subprocess.check_output(
        ["git", "-C", str(root), "show", f"{base}:experiments/registry.yaml"],
        text=True,
    )
    text = registry.read_text()
    for experiment_id in (GLOBAL_ID, TAPER_ID):
        if f"- id: {experiment_id}\n" in text:
            raise SystemExit(f"Experiment already registered: {experiment_id}")

    global_entry = dedent(
        f'''        - id: {GLOBAL_ID}
          environment: EXT-C
          name: countdown_oracle_offline_v2_global_low_scale_sweep_0p5b
          status: pilot
          result_status: pilot_complete
          claim: >-
            Test whether a substantially reduced bank_global_matched negative
            budget can expose useful negative signal on the frozen V2 bank and
            whether a fixed uniform budget preserves that gain to terminal checkpoints.
          role: external_validity_global_low_scale_milestone_diagnostic
          execution_class: pilot
          code_entrypoint: src/drpo/countdown_e8_oracle_offline_v2_matrix.py
          operator_entrypoint: scripts/run_e8_v2_weight_sweep_mp.py
          config_entrypoint: configs/countdown_e8_base_rl_replay_0p5b.yaml
          primary_model: Qwen2.5-0.5B-Instruct
          initialization: pretrained_base_plus_fresh_lora
          method: bank_global_matched
          design:
            multipliers: [0.0, 0.015625, 0.03125, 0.0625, 0.125, 0.25]
            paired_seed_offsets: [0, 1000, 2000, 3000]
            cells: 24
            current_policy_near_far_reselection: true
            global_distance_taper: false
          result:
            scientific_status: milestone_diagnostic_pilot
            best_candidate_multiplier: 0.03125
            positive_only_best_test_pass_at_8_mean: 0.12975
            global_x1_32_best_test_pass_at_8_mean: 0.17375
            best_test_pass_at_8_delta: 0.044
            positive_only_best_test_pass_at_64_mean: 0.25325
            global_x1_32_best_test_pass_at_64_mean: 0.374
            best_test_pass_at_64_delta: 0.12075
            positive_only_terminal_test_pass_at_8_mean: 0.14825
            global_x1_32_terminal_test_pass_at_8_mean: 0.141
            terminal_test_pass_at_8_delta: -0.00725
            numerical_failure_count: 0
            interpretation: >-
              Very small Global pressure yields a clear paired early-checkpoint
              benefit, while sustained uniform pressure does not preserve the
              benefit at terminal checkpoints.
          evidence:
            compact_result_path: experiments/results/e8_oracle_offline_v2_global_low_scale_pilot
            package_filename: 8c44fbd1-ce55-4a21-b6fb-68a64b8ab698.gz
            package_sha256: 2cfed67a417fe1f3ce657341285e8f8188610f03d48efbd235167cbb09d6cbaa
            package_size_bytes: 10021
            run_commit: 64a2fa2d031b0cde2cb22482ce7a1842e72172b5
            git_dirty: true
            runner_integrated_commit: e4aa36bf5ce03794c0d935b4570e276a0703d93e
          reporting_separation:
            task_performance: early_benefit_and_terminal_degradation_observed
            support_or_structure_boundary: not_formally_audited_valid_rate_only
            nan_inf_numerical_failure: not_observed
          limitations:
          - pilot
          - dirty_worktree
          - tuning_seeds_not_confirmatory
          - original_best_and_terminal_posthoc_evaluation_used_different_seed_offsets
          - no_formal_convergence_or_method_ranking
          safeguards:
          - do_not_call_milestone_pilot_formal_result
          - do_not_claim_global_terminal_superiority
          - do_not_call_valid_rate_a_formal_support_boundary_audit
          - countdown_does_not_replace_cu1_or_du1_controlled_identification
        '''
    )
    taper_entry = dedent(
        f'''        - id: {TAPER_ID}
          environment: EXT-C
          name: countdown_oracle_offline_v2_taper_sweep_0p5b
          status: pilot
          result_status: not_run
          claim: >-
            Tune the active paper taper families on the frozen V2 bank under an
            initialization aggregate negative-gradient budget matched to the
            successful Global x1/32 pilot, testing whether remoteness-aware
            weighting can retain useful negative signal and improve terminal stability.
          role: external_validity_v2_active_taper_family_tuning_pilot
          execution_class: pilot
          code_entrypoint: src/drpo/countdown_e8_oracle_offline_v2_taper_runtime.py
          core_entrypoint: src/drpo/countdown_e8_oracle_offline_v2_taper_sweep.py
          operator_entrypoint: scripts/run_countdown_e8_oracle_offline_v2_taper_sweep.py
          config_entrypoint: configs/countdown_e8_oracle_offline_v2_taper_sweep_0p5b.yaml
          protocol_document: docs/experiments/E8_V2_TAPER_SWEEP_PROTOCOL.md
          primary_model: Qwen2.5-0.5B-Instruct
          initialization: pretrained_base_plus_fresh_lora
          data:
            corpus_experiment: EXT-C-E8-ORACLE-OFFLINE-BANK-V2-0.5B-01
            frozen_v2_bank: true
            current_policy_near_far_reselection: true
            near_mix: 0.5
            far_mix: 0.5
          methods: [reciprocal_linear, reciprocal_quadratic, exponential]
          excluded_methods: [global_retuning, sbrc, hybrid, squared_distance_exponential]
          grid:
            rho_values: [0.9, 0.75, 0.6, 0.5, 0.35, 0.25, 0.125, 0.03125]
            paired_tuning_seed_offsets: [0, 1000, 2000]
            cells: 72
            required_gpus: 8
            maximum_waves: 9
          remoteness:
            definition: u=sqrt(relu(sequence_surprisal-tau)/scale)
            tau: base_current_near_median_surprisal
            scale: base_current_far_median_minus_current_near_median
            anchor: current_near_median_u0_current_far_median_u1
            rho_semantics: initialization_far_median_retention_w_at_u1
          calibration:
            reference_global_multiplier: 0.03125
            target: initialization_aggregate_negative_gradient_rms
            task_metrics_used: false
            test_data_used: false
            frozen_before_training: true
          frozen_training_protocol:
            steps: 1200
            minimum_steps: 400
            early_stop_patience: 6
            eval_every_steps: 100
            base_config: configs/countdown_e8_base_rl_replay_0p5b.yaml
          execution_gate:
            current_status: implemented_ready_not_run
            one_click_command: python scripts/run_countdown_e8_oracle_offline_v2_taper_sweep.py
            identity_checked_resume: true
            partial_cell_restart: true
            best_terminal_same_generation_seed: true
            posthoc_evaluation_after_training_model_release: true
          terminal_audit:
            required: true
            separates:
            - task_performance_degradation
            - valid_or_structure_proxy
            - nan_inf_numerical_failure
            - best_checkpoint_from_terminal_checkpoint
          interpretation_limits:
          - tuning_seeds_are_not_confirmatory_seeds
          - no_method_ranking_before_fresh_seed_confirmation
          - fixed_budget_or_early_stop_is_not_convergence
          - countdown_is_external_validity_only
          safeguards:
          - do_not_rerun_global_in_this_sweep
          - do_not_add_sbrc_or_hybrid
          - do_not_predeclare_linear_quadratic_or_exp_winner
          - do_not_merge_task_performance_with_nan_inf_failure
        '''
    )
    boundary = text.find("\ndevelopment_experiment_registrations:")
    if boundary < 0:
        raise SystemExit("Registry experiments boundary missing")
    text = text[: boundary + 1] + global_entry + "\n" + taper_entry + text[boundary + 1 :]
    registry.write_text(text)

    sys.path.insert(0, str(root / "scripts"))
    import handoff_delta_shadow as shadow

    handoff = subprocess.check_output(
        ["git", "-C", str(root), "show", f"{base}:docs/handoff.md"], text=True
    )
    heading = [
        "0. 研究与执行原则（每次新会话首先阅读）",
        "0.1 当前执行门禁",
    ]
    operations = [
        {
            "operation_id": "append-e8-v2-global-low-scale-milestone",
            "op": "append_to_section",
            "heading_path": heading,
            "block_id": "e8-v2-global-low-scale-milestone-pilot",
            "content": (
                f"- **Countdown E8 V2 Global low-scale milestone pilot:** 注册 `{GLOBAL_ID}`。"
                "四个 paired seeds 的 Global `x1/32` 在 validation-selected best checkpoint 上，"
                "test pass@8 相对 Positive-only 提高 4.4 个百分点，pass@64 提高 12.075 个百分点；"
                "terminal pass@8 则低 0.725 个百分点。该结果支持足够小的负优势可被利用，"
                "同时显示持续、无距离区分的 Global 压力不能保持收益。24 cells 无 NaN/Inf；"
                "support/structure boundary 未正式审计。本证据为 dirty-worktree milestone diagnostic pilot，"
                "不构成正式排名、收敛或 Global 终态优越性结论。"
            ),
        },
        {
            "operation_id": "append-e8-v2-active-taper-sweep-ready",
            "op": "append_to_section",
            "heading_path": heading,
            "block_id": "e8-v2-active-taper-sweep-ready",
            "content": (
                f"- **Countdown E8 V2 active taper tuning:** 注册 `{TAPER_ID}`，状态为 `implemented_ready_not_run`。"
                "本轮停止继续调 Global，只比较 Linear、Quadratic、Exp；8 个 `rho` × 3 个 paired tuning seeds，"
                "共 72 cells，使用 GPU 0--7。初始化 aggregate negative-gradient RMS 均匹配 Global `x1/32` 预算；"
                "current-near 中位点锚定 `u=0`，current-far 中位点锚定 `u=1`。"
                "SBRC、Hybrid、Global retuning、SFT init、on-policy 和 replay 均排除。"
                "本轮仅为调参 pilot；必须报告 best 与 terminal，并在冻结超参后使用新 seeds 才能形成方法排名。"
            ),
        },
    ]
    candidate = shadow.render(handoff, operations).text
    evidence_common = [
        "experiments/registry.yaml",
        f"docs/handoff_deltas/{UPDATE_ID}/HANDOFF_DELTA.yaml",
    ]
    delta = {
        "schema_version": 3,
        "update_id": UPDATE_ID,
        "mode": "authoritative",
        "base": {
            "commit": base,
            "handoff_sha256": shadow.sha256_text(handoff),
            "registry_sha256": shadow.sha256_text(before_registry),
        },
        "renderer_version": 1,
        "operations": operations,
        "registry": {
            "mode": "expected_after",
            "exact_base_after_sha256": shadow.sha256_text(text),
            "changes": [
                {
                    "change_id": "add-e8-v2-global-low-scale-milestone",
                    "kind": "add_entity",
                    "entity_id": GLOBAL_ID,
                    "evidence": evidence_common
                    + [
                        "experiments/results/e8_oracle_offline_v2_global_low_scale_pilot/RESULT_SUMMARY.json",
                        "scripts/run_e8_v2_weight_sweep_mp.py",
                    ],
                },
                {
                    "change_id": "add-e8-v2-active-taper-sweep",
                    "kind": "add_entity",
                    "entity_id": TAPER_ID,
                    "evidence": evidence_common
                    + [
                        "configs/countdown_e8_oracle_offline_v2_taper_sweep_0p5b.yaml",
                        "docs/experiments/E8_V2_TAPER_SWEEP_PROTOCOL.md",
                        "scripts/run_countdown_e8_oracle_offline_v2_taper_sweep.py",
                        "src/drpo/countdown_e8_oracle_offline_v2_taper_runtime.py",
                        "src/drpo/countdown_e8_oracle_offline_v2_taper_sweep.py",
                        "tests/test_countdown_e8_oracle_offline_v2_taper_sweep.py",
                    ],
                },
            ],
        },
        "expected": {"exact_base_candidate_sha256": shadow.sha256_text(candidate)},
    }
    delta_dir = root / "docs/handoff_deltas" / UPDATE_ID
    delta_dir.mkdir(parents=True, exist_ok=False)
    (delta_dir / "HANDOFF_DELTA.yaml").write_text(
        yaml.safe_dump(delta, sort_keys=False, allow_unicode=True, width=110)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
