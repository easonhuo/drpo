from __future__ import annotations

import hashlib
from pathlib import Path

BASE_REGISTRY_SHA256 = "f300089b4309fe98f9fd887f7d25411866c208df41c4373a1cadcdda9c85cdd5"
AFTER_REGISTRY_SHA256 = "5bd36d9e797f923eae52deb5ae5cc193ccd536490e2c7d8895fa744321b048f1"
EXPECTED_FILES = {
    "docs/experiments/EXT-H-E7-SQEXP-GAE-01.md": "ad767e2d03f2279dc8be6026abe38b484d013a42636b08b993ca6cbd153e64cb",
    "docs/results/E7_BENCH_JOINT_GAE_P1_RESULT_2026-07-21.md": "08aecb2d04f9185471617efce1f5ae1eb36d8a01dc286f507e2d45a490d64428",
    "docs/handoff_deltas/EXT-H-E7-SQEXP-GAE-P1-RESULT-P2-REGISTRATION-2026-07-21/HANDOFF_DELTA.yaml": "3fb6d5927014ffa4c38797600ce8045a5cd5f3fc2e1e9b583b80ee3e8f59ecf5",
}
MARKER = "development_experiment_registrations:\n"
ENTITY = """- id: EXT-H-E7-SQEXP-GAE-01
  environment: EXT-H
  name: d4rl9_joint_critic_gae_common_c_screening
  status: pilot
  scientific_status: pilot
  result_status: pilot
  parent_experiment: EXT-H-E7-BENCH-01
  predecessor: EXT-H-E7-SQUARED-EXP-NIGHT-01
  registration_base_commit: b18aea9186d7e3ccc5d43b456719cafc23761e03
  claim: >-
    On the canonical D4RL joint actor--critic path with trajectory-snapshot GAE,
    characterize the finite-horizon nine-task response to a single common
    squared-remoteness exponential scale and determine whether a stronger
    left-boundary taper can retain useful negative signal without the broad
    degradation observed under weaker control.
  role: external_validity_joint_critic_gae_common_scale_screening
  execution_class: pilot
  code_entrypoint: src/drpo/e7_squared_exp_night.py
  bootstrap_entrypoint: src/drpo/e7_squared_exp_night_bootstrap.py
  aggregation_entrypoint: src/drpo/e7_squared_exp_night_aggregate.py
  kernel_entrypoint: src/drpo/e7_squared_exp_kernel.py
  runtime_selection_entrypoint: src/drpo/e7_squared_exp_night_runtime_autotune.py
  operator_entrypoint: scripts/run_e7_squared_exp_night_auto.py
  one_click_entrypoint: scripts/run_e7_squared_exp_night_one_click.sh
  protocol_document: docs/experiments/EXT-H-E7-SQEXP-GAE-01.md
  p1_result_document: docs/results/E7_BENCH_JOINT_GAE_P1_RESULT_2026-07-21.md
  shared_protocol:
    datasets:
    - hopper-medium-v2
    - hopper-medium-replay-v2
    - hopper-medium-expert-v2
    - walker2d-medium-v2
    - walker2d-medium-replay-v2
    - walker2d-medium-expert-v2
    - halfcheetah-medium-v2
    - halfcheetah-medium-replay-v2
    - halfcheetah-medium-expert-v2
    development_seeds: [200, 201]
    held_out_seeds: [204, 205, 206, 207]
    actor_update_mode: a2c
    advantage_estimator: trajectory_snapshot_gae
    gae_lambda: 0.95
    critic_updated_every_step: true
    prepared_advantage_artifact: false
    optimizer_steps_per_branch: 1000000
    evaluation_interval_steps: 50000
    evaluation_episodes: 10
    fixed_horizon_is_not_convergence: true
  parameterization:
    coordinate: normalized_squared_standardized_action_distance
    reference_distance: 2.0
    formula: w(D)=w(0)*exp(-lambda_taper*relu((D-tau)/c))
    tau: 0.0
    taper_lambda: 1.0
    positive_only_anchor: true
  profiles:
    p1_broad_curve:
      profile_id: d4rl9_common_c_p1
      execution_state: terminal_audited_delivered
      run_id: E7_BENCH_JOINT_GAE_P1_FULL_20260719_02
      c_values: [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
      uncontrolled_anchor: true
      expected_branches: 198
      completed_branches: 198
      failed_branches: 0
      terminal_audit_status: PASS
      selected_control: null
      selection_status: response_curve_only_pending_protocol_freeze
      equal_task_weighted_late_mean:
        positive_only: 65.3053876161197
        c_0p25: 47.67854193001842
        c_0p5: 37.53881062604026
        c_1: 28.385295302010217
        c_2: 24.418075456461025
        c_4: 17.07861629827022
        c_8: 16.77621207749219
        c_16: 14.937090572288387
        c_32: 12.751391658152787
        c_64: 12.004379177930343
        uncontrolled: 0.4611400998992587
      failure_audit:
        nan_inf_numerical_failure_count: 0
        rollout_failure_count: 0
        task_performance_collapse: not_adjudicated_no_registered_threshold
        support_or_variance_boundary: not_instrumented_in_this_pilot
      evidence_locator:
        results_repository: easonhuo/drpo-results
        results_branch: ingest/e7
        result_path: runs/e7/E7_BENCH_JOINT_GAE_P1_FULL_20260719_02
        ready_manifest_sha256: 9f1ea69f0759bcd3bd79a91c7ccdb5e5d1a22f49ea67e114aaa1e7d4f5f6dbc1
        export_profile: manifest_text_v1
      provenance:
        recorded_source_commit: d0ba443154d847065965b18a43ffe897f19530fa
        source_commit_resolved: false
        authoritative_registration_complete_at_launch: false
        formal_evidence_allowed: false
    p2_left_extension:
      profile_id: d4rl9_common_c_p2_left
      execution_state: implemented_template_not_ready_not_run
      c_values: [0.2, 0.16, 0.125, 0.1, 0.08, 0.0625, 0.04, 0.025, 0.015625]
      positive_only_rerun: true
      uncontrolled_anchor: false
      p1_c_0p25_rerun: false
      p1_c_0p25_role: cross_run_reference_only_not_paired
      expected_branches: 180
      scientific_implementation_commit: 909249875c190a75301ceb2dc2c2062ca0efcb16
      development_branch: dev/ext-h-e7-bench-joint-gae-p2-left-01
      draft_pull_request: 223
      runspec_template: runspecs/templates/E7_BENCH_JOINT_GAE_P2_LEFT_FULL_20260721_01.yaml
      run_id: E7_BENCH_JOINT_GAE_P2_LEFT_FULL_20260721_01
      ready_promoted: false
      branches_started: 0
      launch_requires_separate_approval: true
  terminal_audit:
    required: true
    separates:
    - task_performance_collapse
    - support_or_variance_boundary_event
    - rollout_failure
    - nan_inf_numerical_failure
  interpretation_limits:
  - external_validity_only_not_controlled_causal_identification
  - development_seeds_are_not_confirmatory_seeds
  - fixed_1m_endpoint_is_not_convergence_or_steady_state
  - no_formal_nine_task_method_ranking
  - no_universal_common_c_or_drpo_superiority_claim
  - no_per_task_retuning
  - p1_c_0p25_is_not_a_p2_paired_observation
  - unresolved_p1_source_commit_blocks_authoritative_formal_evidence
  formal_evidence_allowed: false

"""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


for name, expected in EXPECTED_FILES.items():
    actual = sha256(Path(name))
    if actual != expected:
        raise RuntimeError(f"source document identity mismatch for {name}: {actual}")

registry = Path("experiments/registry.yaml")
if sha256(registry) != BASE_REGISTRY_SHA256:
    raise RuntimeError("registry base identity mismatch")
text = registry.read_text(encoding="utf-8")
if text.count(MARKER) != 1 or "- id: EXT-H-E7-SQEXP-GAE-01\n" in text:
    raise RuntimeError("registry insertion boundary is not unique and clean")
registry.write_text(text.replace(MARKER, ENTITY + MARKER), encoding="utf-8")
actual_after = sha256(registry)
if actual_after != AFTER_REGISTRY_SHA256:
    raise RuntimeError(f"registry after-image mismatch: {actual_after}")
