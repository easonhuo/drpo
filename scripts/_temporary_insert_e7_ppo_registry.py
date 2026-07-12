#!/usr/bin/env python3
from pathlib import Path

path = Path("experiments/registry.yaml")
text = path.read_text()
marker = "development_experiment_registrations:\n"
entity_id = "EXT-H-E7-PPO-STABILITY-01"
if f"- id: {entity_id}\n" not in text:
    entry = '''- id: EXT-H-E7-PPO-STABILITY-01
  environment: E7-D4RL
  name: e7_canonical_ppo_actor_stability_pilot
  status: not_run
  scientific_status: pilot
  role: external_validity_actor_update_stability
  execution_class: pilot
  claim: >-
    Test whether replacing only the canonical E7 actor surrogate with a PPO
    clipped surrogate reduces seed sensitivity and branch-wise BEST-to-FINAL
    degradation while preserving the existing network, critic, advantage,
    EXP remoteness taper, normalization, dataset, optimizer, learning rate,
    batch size, evaluation protocol, and one-million-step horizon.
  scientific_boundary:
  - Hopper/D4RL is external-validity evidence only
  - does_not_replace_C-U1_or_D-U1_controlled_mechanism_identification
  - fixed_1M_endpoint_is_not_convergence
  - no_universal_PPO_superiority_or_steady_state_ranking
  implementation:
    state: implementation_ready_real_data_smoke_pending
    source_commit: 76874d6cc40cca83dcf9917fbf779761c222a1be
    smoke_entrypoint: scripts/run_e7_ppo_stability_smoke_one_click.sh
    pilot_entrypoint: scripts/run_e7_ppo_stability_pilot_auto_one_click.sh
    grid: configs/e7_canonical_ppo_stability_v1.json
    execution_protocol: docs/experiments/EXT-H-E7-PPO-STABILITY-01_EXECUTION.md
  preserved_components:
  - canonical_actor_and_critic_architecture
  - critic_target_and_expectile_update
  - advantage_definition_and_normalization
  - positive_only_and_EXP_remoteness_transform
  - D4RL_data_optimizer_learning_rate_batch_and_evaluation_protocol
  ppo_delta:
    clip_epsilon: 0.2
    updates_per_old_policy: 4
    old_policy_source: frozen_snapshot_of_same_actor
    forbidden_additions:
    - KL_penalty
    - target_KL_early_stop
    - entropy_bonus
    - actor_gradient_clipping
    - value_clipping
  smoke_gate:
    status: pending
    scientific_aggregation_allowed: false
    dataset: walker2d-medium-v2
    seed: 200
    exp_coefficient: 1.5
    actor_update_modes: [a2c, ppo_clip]
    steps: 20000
    diagnostics_interval: 1000
    required_checks:
    - matched_A2C_PPO_completion
    - old_policy_refresh_count
    - ratio_position_1_equals_one
    - ratio_positions_2_to_4_leave_one
    - sign_aware_clip_fractions_finite
    - actor_gradient_and_parameter_update_norms_finite
    - no_NaN_or_Inf
  runtime_capacity:
    status: pending_after_smoke
    adapter: e7_canonical_ppo_stability_cpu_v1
    safe_ceiling_inputs:
    - available_logical_CPUs_and_current_load
    - representative_branch_peak_RSS
    - host_memory_headroom
    - configured_growth_and_task_limits
    empirical_candidate_grid:
    - approximately_50_percent_of_safe_ceiling
    - verified_fallback_when_within_ceiling
    - approximately_75_percent_of_safe_ceiling
    - safe_ceiling
    probe_steps_per_branch: 5000
    selection_metric: aggregate_completed_updates_per_second
    selection_rule: smallest_successful_candidate_reaching_97_percent_of_measured_peak
    limitations:
    - candidate_grid_not_continuous_global_optimization
    - short_probe_excludes_50k_evaluation_bursts
    - single_representative_PPO_workload_family
    scientific_matrix_changed: false
  full_pilot:
    state: blocked
    blocked_by:
    - real_data_smoke_gate_pass
    - reviewer_smoke_acceptance
    runspec_state: template_not_ready
    datasets:
    - hopper-medium-expert-v2
    - walker2d-medium-v2
    - walker2d-medium-replay-v2
    development_seeds: [200, 201, 202, 203]
    held_out_seeds_reserved_untouched: [204, 205, 206, 207]
    controls:
    - positive_only
    - exp_scale1_c0.5
    - exp_scale1_c1.0
    - exp_scale1_c1.5
    actor_update_modes: [a2c, ppo_clip]
    steps: 1000000
    expected_branches: 96
    two_million_step_continuation: false
  primary_metrics:
  - branch_wise_BEST
  - FINAL_at_1M
  - BEST_to_FINAL_drop
  - FINAL_across_seed_standard_deviation
  - paired_PPO_minus_A2C
  - paired_EXP_minus_Positive_only
  diagnostics:
  - ratio_distribution_and_ratio_outside_fraction
  - positive_and_negative_objective_clip_fraction
  - old_policy_block_position
  - actor_raw_gradient_norm
  - actor_parameter_update_norm
  reporting_separation:
  - task_performance_degradation
  - support_or_variance_boundary_events
  - NaN_Inf_numerical_failure
  execution:
    state: smoke_ready_full_pilot_blocked
    smoke_run_id: E7_PPO_STABILITY_SMOKE_20260712_01
    smoke_runspec: runspecs/ready/E7_PPO_STABILITY_SMOKE_20260712_01.yaml
    pilot_run_id: E7_PPO_STABILITY_PILOT_20260712_01
    pilot_runspec_template: runspecs/templates/E7_PPO_STABILITY_PILOT_20260712_01.yaml
    pilot_promotion_requires_new_reviewed_commit: true
  terminal_audit: required
  evidence:
    raw_complete: false
    terminal_audited: false
    package_created: false
    held_out_seeds_touched: false
  preserved_history: true
'''
    if marker not in text:
        raise SystemExit("registry insertion marker is missing")
    text = text.replace(marker, entry + marker, 1)
    path.write_text(text)
