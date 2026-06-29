# Categorical D-U1 E5 repulsion and support boundary

> Generated Stage 4 minimal-context shadow module. Do not edit manually.
> `docs/handoff.md` and `experiments/registry.yaml` remain authoritative.

- Module ID: `categorical_e5_mechanism`
- Responsibility: Cover direct-softmax bounded-score analysis and repeated-update transmission to probability or support boundaries.
- Source hash: `7218da3336412e3d8a328c24861d12383c4d1e0f952cc3244930f171ad75b6a8`

## Source 1: docs/handoff.md: ## 3.7.1 D-U1 / D-Diag E5 长程复核 `D-U1-E5-LONGRUN-RERUN` -> ## 3.7.3 E6 共享语义 pilot `D-U1-E6-SEMANTIC-PILOT-01`

## 3.7.1 D-U1 / D-Diag E5 长程复核 `D-U1-E5-LONGRUN-RERUN`

本实验是 E5 历史结果的正式 provenance 重建与长期复核，不是新的方法排名，也不替代 E6。历史代码和 raw artifact 未进入当前 Git 历史，因此本轮只继承已锁定的科学职责、解析初值、方法角色和 qualitative 参照；所有重建参数均在本节和 registry 中一次性冻结。

1. **D-Diag direct-softmax：** 三动作 full-softmax，目标动作固定负优势 `A=-1`，plain Euler/SGD learning rate `1e-3`，20000 steps。两个精确初态分别匹配旧 handoff 的 `(p0,H0)=(0.8991,0.386)` 与 `(0.0038,0.292)`；保存 target probability、surprisal、entropy、direct-logit score 和 logit gap。
2. **D-U1 causal reconstruction：** 6D contexts 只生成受控 contextual provenance；26 个 action ID 由 semantic offset `[-3,3]` 的随机 permutation 得到，禁止把 ID 顺序解释成语义顺序。positive/near/far 的 offset-spread 分别为 `(0,1.2)`、`(-0.5,0.2)`、`(-2.5,0.2)`，advantage magnitude 固定相等。
3. **优化器与数据：** 每 seed 4096 contexts；positive 4096、near 2048、far 2048 empirical samples；Adam `lr=0.003,betas=(0.9,0.999),eps=1e-8`；正式 seeds 10--29。
4. **方法质量：** `positive_only=(0,0)`、`baseline=(0.25,0.45)`、`near_zero=(0,0.45)`、`far_zero=(0.25,0)`、`far_cap=(0.25,0.03)`、`global_scale=(0.10,0.18)`，元组顺序为 `(near_mass,far_mass)`。不得根据正式结果修改。
5. **终态：** 最大 20000 steps、每 100 steps 评估；W1=`10000--15000`，W2=`15000--20000`。稳定门禁为 W2 `|Δbeta|<=0.02`、`|Δtau|<=0.02`、`|Δreward|<=0.01`、raw-gradient median `<=1e-4`。`tau<=0.05` 或 effective support `<=1.5` 记为 support/temperature boundary；没有内部稳态但 surprisal 继续增长则记 persistent suppression；其他为 inconclusive。
6. **任务阈值：** 每 seed 以同一 seed 的 positive-only terminal reward 为参考，终态 reward 不高于其 20% 记 task-performance collapse。该事件与 support boundary、NaN/Inf 分报。
7. **历史比较：** 旧 20/20 qualitative pattern 是预注册 comparison。正式验收首先要求 120/120 method-seed runs 完整、direct-softmax 数值重建通过、所有终态可审计；是否与历史完全一致必须如实报告，不能作为结果后调参门禁。
8. **执行与 artifact：** canonical hardened guard 负责监督和打包；runner 只写普通 CSV/JSON/PNG/Markdown 和每 5 seeds checkpoint marker，不写 archive。正式运行完成后先交 raw-complete 包，再做 terminal audit 和仓库闭环更新。


## 3.7.2 E5 长程复核结果与论文口径

- **运行身份：** `D-U1-E5-LONGRUN-RERUN`，run commit `22c5823d66169eb90c256de342e27c5391e464c3`，formal seeds 10--29，六方法各 20000 steps，120/120 完整。
- **Direct-softmax：** 两个初态均满足 score bound；高概率负动作的 entropy 为 rise-then-fall，低概率负动作 entropy 非增；两者尾段 surprisal/logit-gap slope 均约 `2e-3` per step。该分支证明的是 persistent support suppression，而不是欧氏 logit-gradient amplitude explosion。
- **因果分类：** Baseline/Near-zero 为 task+support 双失败；Far-zero/Far-cap 为两类均救援；Global-scale 保住 task reward 但未保住 support；Positive-only 两类均不失败。每一方法均为 20/20 与历史 qualitative class 一致。
- **事件分离：** task-performance collapse、support/temperature boundary 与 NaN/Inf 继续分开报告。本次三者计数分别依方法变化、支持边界总计 60/120、NaN/Inf 总计 0/120。
- **允许论文表述：** “在该受控 categorical reconstruction 中，bounded direct-logit scores under repeated negative updates still induce monotone surprisal/logit-gap growth and simplex-boundary suppression; selective far-negative removal/capping, but not near-negative removal, breaks the harmful path.”
- **禁止升级：** 不写成旧 runner 逐字节复现、离散欧氏梯度无界、support boundary 等同数值崩溃、E5 已证明未见动作泛化、或 Far-cap/Global-scale 的普遍方法排名。

## Source 2: experiments/registry.yaml: experiments[D-U1-E5-LONGRUN-RERUN]

collection: experiments
entries:
- id: D-U1-E5-LONGRUN-RERUN
  environment: D-U1_and_D-Diag
  name: categorical_repulsion_support_boundary_longrun_reconstruction
  status: long_run_validated
  parent_experiment: E5
  registration_base_commit: d9424f1b9ab4e5ed25bc1ac00f97d84317f67cdc
  claim: 'Reconstruct and long-run audit the locked E5 categorical mechanism: repeated fixed negative updates keep the direct-logit
    score bounded while driving target surprisal and logit gaps toward the simplex boundary, and selective rare/far-negative
    interventions reproduce the historical separation between task-performance collapse, support/temperature boundary events,
    and NaN/Inf numerical failure.'
  role: controlled_categorical_mechanism_reconstruction_and_provenance_repair
  execution_class: formal
  formal_execution:
    channel_ref: hardened-v1
    activation_state: active
    entrypoint_status: implemented
    entrypoint: src/drpo/du1_e5_longrun_rerun.py
    launch_mode: canonical_guard
    artifact_owner: canonical_channel
    guard_entrypoint: scripts/run_experiment_guard_hardened.py
    package_entrypoint: scripts/package_experiment_hardened.py
    verify_entrypoint: scripts/verify_experiment_package_hardened.py
    hardened_core: scripts/artifact_protocol_hardened.py
    artifact_protocol: docs/formal_experiment_artifact_protocol.md
    inherit_default_artifact_budget: true
    runner_archive_policy:
      mode: forbid
  code_entrypoint: src/drpo/du1_e5_longrun_rerun.py
  formal_launch_template: python3 scripts/run_experiment_guard_hardened.py --experiment-id D-U1-E5-LONGRUN-RERUN --repo-root
    . --output-root experiments/results/D-U1-E5-LONGRUN-RERUN/run_001 --artifact-output artifacts/D-U1-E5-LONGRUN-RERUN_RAW_COMPLETE.zip
    --heartbeat-seconds 60 --stale-seconds 900 --fail-on-stale --progress-glob causal/per_seed_summary.csv --required-output
    RUN_COMPLETE.json --required-output terminal_audit.json --required-output REPORT.md --required-output causal/per_seed_summary.csv
    --required-output causal/aggregate_summary.json --required-output historical_reference_comparison.json --source-file src/drpo/du1_e5_longrun_rerun.py
    --run-class formal --expected-commit "$(git rev-parse HEAD)" --require-origin-main-match -- python3 src/drpo/du1_e5_longrun_rerun.py
    --mode formal --output-root experiments/results/D-U1-E5-LONGRUN-RERUN/run_001
  historical_provenance:
    historical_runner_path: run_categorical.py
    historical_result_path: unified_repulsive_dynamics/results/categorical_paper_run/
    historical_source_committed: false
    historical_raw_artifact_committed: false
    reconstruction_status: reconstructed_from_locked_handoff
    exact_legacy_code_reproduction_claimed: false
    historical_values_are_reference_not_tuning_targets: true
  independence_and_parallelism:
    independent_of: C-U1-E4-TAPER-01
    user_parallel_execution_approved: true
    approval_date: '2026-06-26'
    shared_files_requiring_serialized_repository_integration:
    - docs/handoff.md
    - experiments/registry.yaml
  protocol:
    fixed_advantage: true
    critic_or_value_network: false
    importance_sampling: false
    direct_softmax:
      action_count: 3
      fixed_negative_advantage: -1.0
      learning_rate: 0.001
      maximum_steps: 20000
      evaluation_interval_steps: 100
      high_probability_initial: 0.8991
      low_probability_initial: 0.0038
      score_upper_bound: sqrt_2
    du1_causal:
      state_dim: 6
      train_states: 4096
      action_count: 26
      action_ids: random_permutation_of_semantic_offsets
      semantic_offset_range:
      - -3.0
      - 3.0
      positive_samples: 4096
      near_negative_samples: 2048
      far_negative_samples: 2048
      initial_beta: 0.0
      initial_tau: 1.2
      positive_offset_and_spread:
      - 0.0
      - 1.2
      near_offset_and_spread:
      - -0.5
      - 0.2
      far_offset_and_spread:
      - -2.5
      - 0.2
      reward_optimum_offset: 0.7
      reward_width: 0.4
      optimizer: Adam
      learning_rate: 0.003
      betas:
      - 0.9
      - 0.999
      eps: 1.0e-08
      formal_held_out_seeds:
      - 10
      - 11
      - 12
      - 13
      - 14
      - 15
      - 16
      - 17
      - 18
      - 19
      - 20
      - 21
      - 22
      - 23
      - 24
      - 25
      - 26
      - 27
      - 28
      - 29
      maximum_steps: 20000
      evaluation_interval_steps: 100
      audit_window_1:
      - 10000
      - 15000
      audit_window_2:
      - 15000
      - 20000
      support_tau_threshold: 0.05
      effective_support_threshold: 1.5
      task_collapse_ratio_to_positive_only: 0.2
      stable_beta_change: 0.02
      stable_tau_change: 0.02
      stable_reward_change: 0.01
      stable_raw_gradient_median: 0.0001
      checkpoint_marker_every_seeds: 5
  methods:
    positive_only:
      near_mass: 0.0
      far_mass: 0.0
    baseline:
      near_mass: 0.25
      far_mass: 0.45
    near_zero:
      near_mass: 0.0
      far_mass: 0.45
    far_zero:
      near_mass: 0.25
      far_mass: 0.0
    far_cap:
      near_mass: 0.25
      far_mass: 0.03
    global_scale:
      near_mass: 0.1
      far_mass: 0.18
  primary_metrics:
  - target_action_probability_and_surprisal
  - direct_logit_score_norm
  - target_logit_gap
  - entropy_and_effective_support
  - task_reward_relative_to_positive_only
  - beta_and_temperature
  - raw_gradient_norm
  - adam_parameter_update_norm
  reporting_separation:
  - task_performance_collapse
  - support_or_temperature_boundary_event
  - nan_inf_numerical_failure
  terminal_classes:
  - stable_bounded
  - support_boundary
  - persistent_suppression
  - terminal_inconclusive
  - nan_inf_numerical_failure
  historical_reference_pattern:
    baseline:
      task_collapse: true
      support_collapse: true
    near_zero:
      task_collapse: true
      support_collapse: true
    far_zero:
      task_collapse: false
      support_collapse: false
    far_cap:
      task_collapse: false
      support_collapse: false
    global_scale:
      task_collapse: false
      support_collapse: true
    positive_only:
      task_collapse: false
      support_collapse: false
  acceptance_and_reporting:
    run_completeness_required: 120_method_seed_runs
    direct_reference_reproduction_required: true
    historical_causal_pattern_is_comparison_not_tuning_gate: true
    no_post_result_threshold_changes: true
    terminal_audit_required: true
    exact_counts_must_be_reported_even_if_history_does_not_reproduce: true
    no_method_ranking_claim: true
  does_not_replace:
  - E6
  - EXT-C-E8-V4.1
  prohibited_claims:
  - categorical_direct_logit_gradient_is_unbounded
  - support_boundary_equals_nan_inf_failure
  - E5_proves_unseen_action_semantic_generalization
  - exact_legacy_runner_reproduction
  - universal_method_ranking
  execution:
    state: delivered
    run_id: du1_e5_longrun_22c_run003
    start_utc: '2026-06-26T08:49:27.910888+00:00'
    end_utc: '2026-06-26T08:50:30.203169+00:00'
    process_exit_code: 0
    completed_method_seed_runs: 120
    failed_external_timeout_attempts_preserved: 2
  evidence:
    raw_complete: true
    terminal_audited: true
    package_created: true
    package_filename: DRPO_DU1_E5_LONGRUN_22C_RUN003_RAW_COMPLETE.zip
    package_sha256: a4b15c0862b23c78bec15a8b001ea1f1d192f86dc05699cd8bc83b41f0c1348d
    failed_attempt_packages:
    - filename: DRPO_DU1_E5_LONGRUN_22C_RUN001_RAW_COMPLETE.zip
      sha256: bb28f290c110b9410ab46a6a1d59df3599eaefc1068ec9d59d09d2d90a39ca66
      completed_runs_before_external_timeout: 66
    - filename: DRPO_DU1_E5_LONGRUN_22C_RUN002_RAW_COMPLETE.zip
      sha256: f1109c0aba9c46e3e95c25a34da1edd9c1ef13f158b5529ab97550ca3a91b855
      completed_runs_before_external_timeout: 66
    delivered_to_user: true
    applied_commit: null
    scientific_status: long_run_validated
    compact_result_path: outputs/du1_e5_longrun
  terminal_audit:
    raw_runs_complete: true
    expected_method_seed_runs: 120
    actual_method_seed_runs: 120
    all_runs_classified: true
    direct_reference_checks_passed: true
    historical_joint_class_matches: 120
    historical_joint_class_total: 120
    all_historical_classes_match: true
    task_support_nan_inf_reported_separately: true
    total_nan_inf_count: 0
    method_counts:
      positive_only:
        task_collapse: 0
        support_boundary: 0
        stable_bounded: 20
      baseline:
        task_collapse: 20
        support_boundary: 20
        stable_bounded: 0
      near_zero:
        task_collapse: 20
        support_boundary: 20
        stable_bounded: 0
      far_zero:
        task_collapse: 0
        support_boundary: 0
        stable_bounded: 20
      far_cap:
        task_collapse: 0
        support_boundary: 0
        stable_bounded: 20
      global_scale:
        task_collapse: 0
        support_boundary: 20
        stable_bounded: 0
  result_summary:
    direct_softmax:
      high_probability_negative_terminal_probability: 3.7043613598643434e-12
      low_probability_negative_terminal_probability: 1.9172584004125106e-20
      high_probability_negative_max_score: 1.4142132719131175
      low_probability_negative_max_score: 1.4142135622312735
      high_entropy_pattern: rise_then_fall
      low_entropy_pattern: nonincreasing_or_flat
    causal:
      baseline_reward_mean: 0.0006978558570929635
      near_zero_reward_mean: 0.005860351494973978
      far_zero_reward_mean: 0.26738098736568067
      far_cap_reward_mean: 0.29714767390090524
      global_scale_reward_mean: 0.9567831669200537
      positive_only_reward_mean: 0.27525529281893174
  paper_use:
    allowed:
    - bounded_direct_logit_score_with_persistent_surprisal_and_logit_gap_growth
    - simplex_or_support_boundary_under_repeated_negative_updates
    - baseline_and_near_zero_20_of_20_task_and_support_failure
    - far_zero_and_far_cap_20_of_20_task_and_support_rescue
    - global_scale_task_preservation_with_20_of_20_support_boundary
    - exact_separation_of_task_support_and_nan_inf_events
    prohibited:
    - categorical_direct_logit_gradient_is_unbounded
    - support_boundary_equals_nan_inf_failure
    - E5_proves_unseen_action_semantic_generalization
    - exact_legacy_runner_reproduction
    - universal_method_ranking
  provenance:
    run_commit: 22c5823d66169eb90c256de342e27c5391e464c3
    parent_commit: d9424f1b9ab4e5ed25bc1ac00f97d84317f67cdc
    exact_commit_object_sha_verified: true
    git_worktree_clean_at_launch: true
    git_worktree_clean_at_exit: true
    provenance_compromised: false
    historical_runner_available: false
    reconstruction_scope: locked_handoff_scientific_roles_and_reference_values
    exact_legacy_code_reproduction_claimed: false
  artifact_budget:
    main_package_hard_limit_mib: 25
    checkpoint_policy: canonical_channel_markers_every_5_seeds
    large_file_storage: persistent_local_index
