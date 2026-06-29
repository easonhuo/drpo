# Continuous C-U1 E4 stable extrapolation and phase transition

> Generated Stage 4 minimal-context shadow module. Do not edit manually.
> `docs/handoff.md` and `experiments/registry.yaml` remain authoritative.

- Module ID: `continuous_e4_extrapolation`
- Responsibility: Cover the positive-only ceiling, bounded extrapolation, phase transition, fixed or learnable variance outcomes, and E4 convergence closure before taper-family follow-ups.
- Content contract topics: none
- Deduplicated overlapping source chunks: 0
- Source hash: `b62a1253e7a54aa4fb9537955428de923360eaaf557b7a77e6d11a609068cd5f`

## Source 1: docs/handoff.md: ### 3.6.2 E4：稳定外推—相变—远场控制 -> ## 3.7 D-U1 / E6 开发配置登记（E4 已完成；用户已批准与 E4-TAPER 并行）

### 3.6.2 E4：稳定外推—相变—远场控制

1. **正式 seeds：** 开发 seeds 5–9 只用于确定扫描区间、学习率和 far-pressure 强度；正式 held-out seeds 50–69，所有方法配对。
2. **共同初始化：** 从同一 positive-only 饱和策略开始；固定方差主分支使用解析 `sigma=0.190394`，可学习方差分支保留 state-conditioned log-std。
3. **有益局部负信号：** 仅使用每状态第 0 个负动作 `a_minus=a_plus-0.50u`，其排斥方向与真实 improvement direction `a_star-a_plus` 对齐。局部目标为 `L_pos + alpha_local L_minus`。
4. **固定方差强度扫描：** 扫描 `alpha_local` 从 0 到超过解析临界值 `alpha_c=A_pos/|A_neg|≈1.693`；报告解析 signed target、经验归一化外推位移、test reward、终态类别和 2× horizon 审计。最低目标是复现 positive-only ceiling、越过 `a_plus`、在 `归一化外推位移≈1` 附近达到未见最优、随后过度外推和临界漂移。
5. **可学习方差扫描：** 在同一局部目标上扫描更细的低 alpha 区间，检验二阶矩可行性边界是否早于固定方差均值边界；方差越界与任务 reward 失效分别报告。
6. **远场压力：** 将其余 7 个等 advantage 轮廓动作定义为额外 far-pressure，目标写成 `L_pos + alpha_local L_minus + lambda_far L_far`；`alpha_local` 固定在固定方差近最优区间，`lambda_far` 由开发 seeds 预注册为能稳定触发性能反转但不依赖 NaN 的最小值。
7. **控制方法：** 比较 `positive_only`、`local_only`、`uncontrolled_all`、`far_zero/local_oracle`、`far_cap` 与 `budget_matched_global`。Far-cap 只缩放 far 分量；budget-matched global 将全部负梯度统一缩放到与 Far-cap 相同的 post-control norm，以排除“仅仅总梯度更小”。
8. **方向与影响诊断：** 逐负动作报告其梯度与真实 improvement update 的 cosine、score norm、全参数 influence；检验局部有益方向与远场低/反向 utility 是否同时伴随更大 influence。
9. **正式验收：** （a）20/20 或统计显著多数策略越过 `a_plus`；（b）held-out `a_star` reward 高于 positive-only；（c）reward 对负推力呈倒 U 型或存在明确相变；（d）Far-cap 在远场压力下恢复有益外推且不崩溃；（e）相对等预算 global 的差异用 paired bootstrap CI 报告，不预设 Distance 必然胜出。

### 3.6.3 E4 数值配置冻结与一次执行流程纠正

开发 seeds 5–9 得到以下预注册配置：

- **固定方差局部强度网格：** `alpha_local ∈ {0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75}`。其中解析均值临界值为 `alpha_c≈1.693`；1.50 用于观察有限但严重过度外推的稳态，1.75 用于观察固定点消失后的持续漂移。
- **可学习方差局部强度网格：** `alpha_local ∈ {0, 0.10, 0.20, 0.30, 0.35, 0.38, 0.40, 0.50}`。解析二阶矩可行边界约为 `alpha_sigma≈0.381`，因此 0.38/0.40 跨越该边界。
- **优化与终态审计：** 有有限解析内部解的配置先运行 200-step minibatch SGD，随后执行全数据 LBFGS stationary audit，再进行等长 200-step continuation，最后对同一目标重新 stationary audit。无内部解的配置运行 2000+2000-step 长程 SGD，不使用 LBFGS。
- **残差判据细化：** signed objective 的正负分量可各自很大并在固定点相消，因此正式使用 `||g_total||/(||g_pos||+||g_neg||)<2e-3` 作为净动力场归一化残差；`alpha=0` 单独要求 absolute norm `<1e-3`。这是对第 3.5(10) 绝对阈值的必要尺度化细化，原阈值不删除。
- **远场压力与控制：** `alpha_local=1.0`、`lambda_far=1.0`，Far-cap 约束 far weighted-gradient norm 不超过 local weighted-gradient norm 的 `0.05`。开发 seed 上该配置使 uncontrolled_all 发生有限数值下的任务崩溃，而 Far-cap 保留正向外推。Budget-matched global 的 post-control negative norm 与 Far-cap 精确匹配。
- **方向诊断：** 在 positive-only 初始化处，第 0 个负动作与真实 improvement update 的 cosine 为 1；最远第 4 个动作 cosine 为 -1，且其全参数 update norm 约为近场的 3.8 倍。正式结果使用 20 seeds 汇总，不把单 seed 数值当作结论。

**执行流程纠正：** 在本小节写入前曾误启动固定方差正式 driver，产生 12 个未完成结果。发现“精确网格尚未先回写文档”后立即停止；这些文件未删除，整体移动到 `e4_pre_freeze_fixed_pilot_091632/`，只作 provenance，不进入正式统计。正式 E4 必须在本小节冻结后从空目录重新运行。

### 3.6.4 E4 控制分支的精确长程配置

- `positive_only` 与 `local_only(alpha=1.0)` 直接复用同 seeds 的正式局部扫描结果，不重复训练。
- 新增长程方法只有 `uncontrolled_all`、`far_cap`、`budget_matched_global`；共同使用 `alpha_local=1.0`、`lambda_far=1.0`、Far-cap ratio `0.05`、固定 `sigma=0.190394`、SGD `lr=5e-4`。
- 训练 4000 steps，每 100 steps 评估；2000 steps 是候选 horizon，4000 steps 是 2× extension。报告 reward、归一化外推位移、净更新残差、任务崩溃 onset、数值有限性及方法排序是否在后半程反转。
- `budget_matched_global` 在每一步将原始全部负梯度统一缩放，使其 post-control norm 与 Far-cap 完全相同；允许缩放系数大于 1，因为原始 local/far 分量可能方向抵消。该对照匹配的是实际净负梯度预算，而不是预设“只能缩小”。
- 正式方向诊断在 positive-only 初始化处对 8 个等 advantage 负动作分别计算全参数 update norm、标准化距离及与真实 improvement update 的 cosine；20 seeds 配对汇总。

### 3.6.5 v29 统一 Adam 执行覆盖（当前有效协议）

本节覆盖 3.6.3、3.6.4 和 11.4 中的 SGD/LBFGS 执行细节；旧内容保留作 provenance。

1. E3 fixed、E3 learnable、E4 fixed、E4 learnable 与 E4 control 的训练优化器统一为 Adam，`betas=(0.9,0.999)`、`eps=1e-8`；沿用已冻结的各分支 learning rate、alpha、seeds、数据、步数上限和任务阈值，不借优化器迁移反向调参。
2. E3/E4 初始化固定为同 seed 2000-step positive-only Adam checkpoint。E2 的 LBFGS、2× continuation 和 adaptive polish 仅做 E2 终态审计。
3. E4 有有限解析内部解的配置先做 200-step Adam、全数据 residual audit、等长 200-step Adam continuation、第二次 residual audit；audit 只测量同一目标的净动力场，不再用 LBFGS 改写参数。无内部解配置按原上限做 Adam 长程并报告持续漂移或首次支持收缩。
4. Learnable-variance 每一步在完整 4096 train states 上做首次事件审计。`support_contraction`、task-performance collapse、parameter/log-sigma/sigma-output NaN/Inf 分开；任何 `unexpected_support_expansion` 都是失败诊断，不进入方法排名。
5. E3/E4 输出必须同时包含 raw total/negative gradient norm 与 Adam parameter-update norm。Raw-gradient matched control 仍用于机制对照，但论文不得称其为 actual-update matched，除非另行登记并实现 Adam update-level calibration。
6. 主文只保留最短因果链和倒 U 型相变；Global、Far-to-near、budget-matched controls 进入附录，不把优化器细节拆成多条主叙事。
7. 正式命令必须按 stage 分开执行；`--stage all` 只允许 smoke。

### 3.6.6 `C-U1-E4-CONV-01` 长程终态确认（v33 当前有效协议）

1. **实验职责：** 仅确认原 E4 固定方差 `alpha=0.75/1.00/1.25` 的长期状态是否反转。它不重跑可学习方差、控制方法、`alpha=1.50/1.75`，也不新增方法排名。
2. **Positive-only 边界：** 不追加运行 `alpha=0`。E2 承担 positive-only 完整动力学；原 E4 的 `alpha=0` 只保留为相变扫描左端 control。
3. **冻结执行：** seeds 50--69；从同 seed 的 2000-step positive-only Adam checkpoint 重新开始；固定方差、Adam、学习率、batch、advantage、数据和 RNG 与 `C-U1-E4-ADAM-RERUN` 完全一致。
4. **训练与审计：** 每个 alpha 运行 4000 steps；full-state audits 为 `400/800/1600/2400/3200/4000`；终态窗口为 `2000--3000` 和 `3000--4000`。
5. **稳定判据：** W2 位移变化绝对值 `<=0.02`，W2 reward 变化绝对值 `<=0.01`，raw full-data gradient 与 Adam update 的 W2/W1 中位比均 `<=1.25`，且长期科学角色不反转。
6. **Runaway 判据：** 两个窗口的位移均增加，W2 位移增量 `>0.05`，且 raw gradient 或 Adam update 的 W2/W1 中位比 `>1.25`。其余登记 `terminal_state_inconclusive`。
7. **残差口径：** 继续记录 full-data normalized residual，但 `2e-3` 不再是硬 gate，不为通过门禁而改学习率、optimizer、batch、threshold 或训练长度。
8. **目标状态与汇总：** `0.75/1.00 -> stable_beneficial_extrapolation`；`1.25 -> stable_over_extrapolation`。每个 alpha 至少 18/20 达标，余下只允许 inconclusive。
9. **持久化：** 每 5 seeds 生成 checkpoint 包；正式结束后必须独立报告任务性能、support/variance boundary 和 NaN/Inf，并完成终态审计与 durable delivery。

---

## Source 2: experiments/registry.yaml: experiments[C-U1-E4-ADAM-RERUN, C-U1-E4-CONV-01]

collection: experiments
entries:
- id: C-U1-E4-ADAM-RERUN
  environment: C-U1
  name: cu1_stable_extrapolation_phase_transition_unified_adam
  status: finite_step_validated
  claim: Test whether controlled local negative gradients improve beyond the positive-only ceiling and whether excessive negative
    pressure causes support contraction, continuing drift, or task-performance collapse under one Adam training pipeline.
  role: controlled_generalization_and_phase_transition
  execution_class: historical_formal
  historical_formal_execution:
    channel_status: grandfathered_completed_run
    future_rerun_requires_channel: hardened-v1
  depends_on_delivered_experiment: C-U1-E3-ADAM-RERUN
  code_entrypoint: src/drpo/drpo_cu1_e1_e4_oneclick.py
  command:
  - python
  - src/drpo/drpo_cu1_e1_e4_oneclick.py
  - --stage
  - e4
  - --output-root
  - outputs/cu1_e4_adam
  initialization:
    source: positive_only_adam_2000_step_checkpoint
    e2_terminal_audit_checkpoint_used: false
    shared_across_methods: true
  optimizer:
    name: Adam
    betas:
    - 0.9
    - 0.999
    eps: 1.0e-08
    lr: 0.0005
    lbfgs_parameter_updates_in_e4: false
  data:
    train_states: 4096
    test_states: 4096
    positive_actions_per_state: 4
    negative_actions_per_state: 8
    terminology: held_out_context_generalization
    fixed_advantage: true
  development_seeds:
  - 5
  - 6
  - 7
  - 8
  - 9
  held_out_seeds:
  - 50
  - 51
  - 52
  - 53
  - 54
  - 55
  - 56
  - 57
  - 58
  - 59
  - 60
  - 61
  - 62
  - 63
  - 64
  - 65
  - 66
  - 67
  - 68
  - 69
  fixed_variance_alpha_grid:
  - 0.0
  - 0.25
  - 0.5
  - 0.75
  - 1.0
  - 1.25
  - 1.5
  - 1.75
  learnable_variance_alpha_grid:
  - 0.0
  - 0.1
  - 0.2
  - 0.3
  - 0.35
  - 0.38
  - 0.4
  - 0.5
  main_story:
  - positive_only_ceiling
  - beneficial_controlled_negative_region
  - excessive_negative_pressure_support_contraction_or_task_collapse
  appendix_controls:
  - uncontrolled_all
  - far_cap
  - budget_matched_global
  terminal_audit:
    finite_internal_solution:
    - 200_step_adam
    - full_data_residual_audit_1
    - 200_step_adam_continuation
    - full_data_residual_audit_2
    no_internal_solution:
    - long_run_adam
    - drift_or_first_boundary_audit
    integrity_checks_all_passed: true
    scientific_terminal_acceptance_passed: false
    failure_reason: The finite-horizon beneficial branch is reproducible, but no beneficial alpha passes both frozen full-data
      residual audits in 20/20 seeds. Fixed-variance alpha=1.00 passes both audits in only 3/20 seeds.
  metrics:
  - held_out_context_reward
  - normalized_extrapolation_displacement
  - distance_to_a_plus_and_a_star
  - normalized_field_residual
  - support_contraction_onset
  - raw_total_gradient_norm
  - adam_parameter_update_norm
  - parameter_log_sigma_sigma_output_finiteness
  method_ranking_pre_registered: false
  controls_note: Raw-gradient matched controls are not called Adam-update matched.
  formal_run_status: delivered
  execution:
    state: delivered
    run_id: cu1_e4_adam_d699bb6_run001
    completion_mode: hardened_foreground_supervisor
    start_utc: '2026-06-26T03:55:06.139980+00:00'
    end_utc: '2026-06-26T04:25:23.841907+00:00'
    last_heartbeat_utc: '2026-06-26T04:25:21.671266+00:00'
    process_exit_code: 0
    elapsed_seconds: 1817.702
  evidence:
    raw_complete: true
    terminal_audited: true
    terminal_audit_integrity_all_checks_passed: true
    terminal_scientific_acceptance_passed: false
    expected_fixed_rows: 160
    actual_fixed_rows: 160
    expected_learnable_rows: 160
    actual_learnable_rows: 160
    expected_control_rows: 60
    actual_control_rows: 60
    expected_variance_robustness_rows: 45
    actual_variance_robustness_rows: 45
    missing_required_files: 0
    checkpoint_packages_created: 4
    package_created: true
    package_filename: DRPO_CU1_E4_ADAM_D699_FINAL_EVIDENCE.zip
    package_sha256: c2fbc594891b594652338b8937d02d4b283e75caa7cd475572ca7307f6f08673
    raw_complete_package_filename: DRPO_CU1_E4_ADAM_RAW_COMPLETE_D699_RUN001.zip
    raw_complete_package_sha256: daf7d133692335db477a5c5b42706b96d245e13696b6ea181f0e2895ee2387e8
    delivered_to_user: true
    applied_commit: null
    compact_result_path: outputs/cu1_e4_adam
  provenance:
    run_commit: d699bb6b1d0093d8a9b935fd6c67f049fc3c3df0
    repository_closure_base_commit: d699bb6b1d0093d8a9b935fd6c67f049fc3c3df0
    source_mode: exact_git_bundle_checkout
    source_bundle_filename: DRPO_MAIN_d699bb6b1d00.bundle
    source_bundle_sha256: d9c5dfa5b914d4e17e224849ebed6b400c6cadc26cc41ad60d6fcc510b0bc5bb
    git_bundle_verify_passed: true
    clean_worktree_at_launch: true
    clean_worktree_at_exit: true
    provenance_compromised: false
    cuda_available: false
    device: cpu
  result_summary:
    positive_only_ceiling:
      fixed_alpha_0_reward_mean: 0.646987646818161
    finite_horizon_beneficial_region:
      fixed_alpha_0_25_reward_mean: 0.7144076138734817
      fixed_alpha_0_50_reward_mean: 0.8040281385183334
      fixed_alpha_0_75_reward_mean: 0.9153103202581405
      fixed_alpha_1_00_reward_mean: 0.9917025655508042
      fixed_alpha_1_00_reward_ci95:
      - 0.99136316254735
      - 0.9920345157384872
      fixed_alpha_1_00_normalized_displacement_mean: 1.007807719707489
      each_alpha_0_25_through_1_00_beats_alpha_0_paired_seeds: 20
      stable_fixed_point_claim_terminally_validated: false
      fixed_alpha_1_00_both_residual_audits_passed: 3
    excessive_fixed_pressure:
      fixed_alpha_1_50_task_collapse_count: 20
      fixed_alpha_1_75_task_collapse_count: 20
      fixed_alpha_1_75_continuing_runaway_count: 20
      nan_inf_count: 0
    learnable_variance:
      alpha_0_40_support_contraction_count: 18
      alpha_0_40_support_onset_median: 434.5
      alpha_0_50_support_contraction_count: 20
      alpha_0_50_support_onset_median: 83.0
      unexpected_support_expansion_count: 0
      nan_inf_count: 0
    variance_robustness:
      alpha_0_38_cross_minus_8_count: 0
      alpha_0_40_cross_minus_8_count: 15
      alpha_0_40_cross_minus_12_count: 11
      alpha_0_50_cross_minus_14_count: 15
      total_rows: 45
    controls_4000_steps:
      uncontrolled_all_reward_mean: 0.0
      uncontrolled_all_task_failure_count: 20
      far_cap_reward_mean: 0.9952240824699402
      far_cap_task_failure_count: 0
      budget_matched_global_reward_mean: 0.502925130724907
      budget_matched_global_task_failure_count: 0
      nonfinite_count: 0
  paper_use:
    suitable_for_finite_horizon_phase_figure: true
    suitable_for_stable_fixed_point_claim: false
    allowed:
    - positive_only_ceiling_and_finite_horizon_benefit
    - excessive_fixed_pressure_task_collapse
    - learnable_variance_support_contraction
    - long_run_controls_as_appendix_evidence
    prohibited_claims:
    - terminally_stable_beneficial_fixed_point
    - OOD_generalization
    - universal_method_ranking
    - NaN_Inf_collapse
  next_gate:
    convergence_experiment: C-U1-E4-CONV-01
    convergence_status: long_run_validated_user_confirmed_closure
    taper_status: ready
    reason: The original 18/20 consensus did not pass, but the user explicitly accepted the preserved 15/20, 16/20, and 15/20
      expected-state counts together with 0/60 explicit opposite states and 60/60 non-reversed scientific roles as sufficient
      long-horizon E4 closure. TAPER may start only after this closure update is applied and committed.
  scientific_status: finite_step_validated
  preserved_history: true
- id: C-U1-E4-CONV-01
  environment: C-U1
  name: cu1_e4_fixed_variance_long_horizon_terminal_confirmation
  status: long_run_validated
  scientific_status: long_run_validated
  claim: Confirm that fixed-variance alpha=0.75 and alpha=1.00 remain bounded beneficial extrapolation rather than transient
    optimum crossings, and that alpha=1.25 is a stable over-extrapolated state rather than slow runaway.
  role: controlled_long_horizon_terminal_confirmation
  execution_class: formal
  formal_execution:
    channel_ref: hardened-v1
    activation_state: active
    entrypoint_status: implemented
    entrypoint: src/drpo/drpo_cu1_e1_e4_oneclick.py
    launch_mode: canonical_guard
    artifact_owner: canonical_channel
    guard_entrypoint: scripts/run_experiment_guard_hardened.py
    package_entrypoint: scripts/package_experiment_hardened.py
    verify_entrypoint: scripts/verify_experiment_package_hardened.py
    hardened_core: scripts/artifact_protocol_hardened.py
    artifact_protocol: docs/formal_experiment_artifact_protocol.md
    inherit_default_artifact_budget: true
    runner_archive_policy:
      mode: legacy_exception
      exception_id: CU1-RECOVERY-CHECKPOINT-LEGACY-01
  depends_on_delivered_experiment: C-U1-E4-ADAM-RERUN
  blocks_experiment: C-U1-E4-TAPER-01
  code_entrypoint: src/drpo/drpo_cu1_e1_e4_oneclick.py
  stage: e4_convergence
  command:
  - python3
  - src/drpo/drpo_cu1_e1_e4_oneclick.py
  - --stage
  - e4_convergence
  - --output-root
  - experiments/results/C-U1-E4-CONV-01/run_001
  guard_entrypoint: scripts/run_experiment_guard_hardened.py
  formal_launch_template: python3 scripts/run_experiment_guard_hardened.py --run-class formal --expected-commit "$(git rev-parse
    HEAD)" --experiment-id C-U1-E4-CONV-01 --repo-root . --output-root experiments/results/C-U1-E4-CONV-01/run_001 --artifact-output
    artifacts/C-U1-E4-CONV-01_RAW_COMPLETE.zip -- python3 src/drpo/drpo_cu1_e1_e4_oneclick.py --stage e4_convergence --output-root
    experiments/results/C-U1-E4-CONV-01/run_001
  scope:
    fixed_variance_only: true
    alphas:
    - 0.75
    - 1.0
    - 1.25
    positive_only_additional_run: false
    positive_only_terminal_evidence_owner: C-U1-E2
    rerun_full_e4: false
    rerun_learnable_variance: false
    rerun_controls: false
    rerun_alpha_1_50_or_1_75: false
  initialization:
    source: positive_only_adam_2000_step_checkpoint
    e2_terminal_audit_checkpoint_used: false
    restart_from_same_e2_initialization: true
    continue_from_old_400_step_checkpoint: false
    shared_across_alphas: true
  optimizer:
    name: Adam
    betas:
    - 0.9
    - 0.999
    eps: 1.0e-08
    lr: 0.0005
    fixed_sigma: 0.1903943276465978
    batch_or_rng_changes_allowed: false
  data:
    train_states: 4096
    test_states: 4096
    positive_actions_per_state: 4
    negative_actions_per_state: 8
    terminology: held_out_context_generalization
    fixed_advantage: true
  held_out_seeds:
  - 50
  - 51
  - 52
  - 53
  - 54
  - 55
  - 56
  - 57
  - 58
  - 59
  - 60
  - 61
  - 62
  - 63
  - 64
  - 65
  - 66
  - 67
  - 68
  - 69
  training:
    max_steps: 4000
    full_state_audit_steps:
    - 400
    - 800
    - 1600
    - 2400
    - 3200
    - 4000
    terminal_window_1:
    - 2000
    - 3000
    terminal_window_2:
    - 3000
    - 4000
    checkpoint_every_formal_seeds: 5
    silent_horizon_extension_allowed: false
  terminal_classification:
    residual_threshold_2e_3_is_hard_gate: false
    normalized_field_residual_retained_as_diagnostic: true
    stable_platform:
      window_2_absolute_displacement_change_max: 0.02
      window_2_absolute_reward_change_max: 0.01
      window_2_over_window_1_raw_gradient_median_ratio_max: 1.25
      window_2_over_window_1_adam_update_median_ratio_max: 1.25
      scientific_role_step_2000_to_4000_must_not_reverse: true
    continuing_runaway:
      window_1_displacement_increase_required: true
      window_2_displacement_change_min: 0.05
      raw_gradient_or_adam_update_ratio_min_exclusive: 1.25
    otherwise: terminal_state_inconclusive
  expected_terminal_states:
    alpha_0_75: stable_beneficial_extrapolation
    alpha_1_00: stable_beneficial_extrapolation
    alpha_1_25: stable_over_extrapolation
  aggregate_acceptance:
    minimum_expected_state_seeds_per_alpha: 18
    remaining_seeds_allowed_state: terminal_state_inconclusive
    explicit_opposite_terminal_state_allowed: false
  metrics:
  - held_out_context_reward
  - normalized_extrapolation_displacement
  - distance_to_a_plus_and_a_star
  - full_data_raw_total_gradient_norm
  - minibatch_raw_total_gradient_norm
  - adam_parameter_update_norm
  - normalized_field_residual_diagnostic
  - task_performance_collapse
  - support_or_variance_boundary_event
  - nan_inf_numerical_failure
  reporting_separation:
  - task_performance_collapse
  - support_or_variance_boundary
  - nan_inf_numerical_failure
  method_ranking_pre_registered: false
  formal_run_status: delivered
  execution:
    state: delivered
    run_id: cu1_e4_conv_c869df8_run002
    completion_mode: scientific_child_completed_wrapper_packaging_recovered
    start_utc: '2026-06-26T06:37:47.424130+00:00'
    end_utc: '2026-06-26T06:48:25.998924+00:00'
    process_exit_code: 0
    elapsed_seconds: 635.0746870040894
    first_attempt_failure_preserved: true
    wrapper_required_output_mismatch_preserved: true
  evidence:
    raw_complete: true
    terminal_audited: true
    package_created: true
    package_filename: DRPO_CU1_E4_CONV_C869_RUN002_RAW_COMPLETE.zip
    package_sha256: 98214c2f09f7cd6ba75472bfc489771cb2ac439031e9f3636a8472a6c2a06b13
    failed_attempt_package_filename: DRPO_CU1_E4_CONV_RUN001_FAILED_C869.zip
    failed_attempt_package_sha256: 765c40e3b2df1e980ef786cdf5f6dddd912d1d149d111207d822b098cf2a99ff
    guard_wrapper_failure_package_filename: DRPO_CU1_E4_CONV_C869_RUN002_GUARD_FAILED.zip
    guard_wrapper_failure_package_sha256: bef2e57cf0646b3aa3556156fcb710a611cebb397847af3ea033f59316e2a802
    expected_rows: 60
    actual_rows: 60
    checkpoint_packages_created: 4
    delivered_to_user: true
    applied_commit: null
    compact_result_path: outputs/cu1_e4_convergence
  paper_use:
    allowed:
    - long_horizon_beneficial_extrapolation_for_alpha_0_75_and_1_00_with_exact_counts
    - long_horizon_stable_over_extrapolation_for_alpha_1_25_with_exact_counts
    - no_explicit_opposite_terminal_state_in_60_runs
    - all_60_scientific_roles_not_reversed_from_step_2000_to_4000
    - aggregate_displacement_and_reward_remain_near_stationary
    prohibited:
    - registered_18_of_20_terminal_gate_passed
    - all_20_seeds_fixed_point_certified
    - every_seed_strictly_stationary
    - OOD_generalization
    - universal_method_ranking
  user_confirmed_closure:
    confirmed: true
    decision_date: '2026-06-26'
    decision_type: post_run_explicit_user_evidence_review
    original_pre_registered_consensus_gate_passed: false
    per_alpha_expected_state_counts:
      alpha_0_75: 15
      alpha_1_00: 16
      alpha_1_25: 15
    total_explicit_opposite_terminal_states: 0
    total_scientific_roles_not_reversed: 60
    accepted_scientific_scope: 'The 4000-step evidence closes the E4 long-horizon phase claim: alpha=0.75 and 1.00 retain
      bounded beneficial extrapolation without role reversal, while alpha=1.25 is stable over-extrapolation rather than slow
      runaway.'
    excluded_scope: This decision does not certify 20/20 fixed points, does not claim that the original 18/20 gate passed,
      and does not relabel inconclusive seed-alpha rows.
  preserved_history: true
  terminal_audit:
    integrity_checks_all_passed: true
    scientific_terminal_acceptance_passed: false
    pre_registered_18_of_20_gate_passed: false
    user_confirmed_scoped_scientific_closure_passed: true
    consensus_min: 18
    alpha_0_75_expected_state_count: 15
    alpha_0_75_inconclusive_count: 5
    alpha_1_00_expected_state_count: 16
    alpha_1_00_inconclusive_count: 4
    alpha_1_25_expected_state_count: 15
    alpha_1_25_inconclusive_count: 5
    explicit_opposite_terminal_state_count: 0
    scientific_role_not_reversed_count: 60
    task_performance_collapse_count: 0
    support_or_variance_boundary_count: 0
    nan_inf_count: 0
    failure_reason: No alpha reached the frozen 18/20 expected-state consensus. All remaining runs were inconclusive and no
      explicit opposite terminal state occurred.
    post_hoc_diagnostic_only: Fourteen seed-alpha rows were inconclusive because a raw-gradient or Adam-update W2/W1 ratio
      exceeded 1.25; displacement and reward windows remained within frozen stability bounds. This does not override the original
      gate.
    closure_note: The user explicitly accepted the preserved majority-state evidence and absence of opposite states as sufficient
      for the scoped long-horizon E4 phase claim; the original gate remains recorded as failed.
  provenance:
    run_commit: c869df8b203f13eb8389d1d300b33f1928502871
    source_mode: exact_reconstructed_git_commit_object
    parent_commit: 5b4671780c09d434d6482a71eada9422f885f10f
    reconstructed_commit_sha_matched_github_full_sha: true
    git_worktree_clean_at_launch: true
    git_worktree_clean_at_exit: true
    provenance_compromised: false
    origin_main_ls_remote_authoritative: false
    origin_main_resolution_error: container DNS could not resolve github.com
    device: cpu
    cuda_available: false
  result_summary:
    alpha_0_75_reward_mean: 0.9206409364938736
    alpha_0_75_displacement_mean: 0.566690868139267
    alpha_0_75_expected_state_count: 15
    alpha_1_00_reward_mean: 0.9982822149991989
    alpha_1_00_displacement_mean: 1.02889803647995
    alpha_1_00_expected_state_count: 16
    alpha_1_25_reward_mean: 0.6388135522603988
    alpha_1_25_displacement_mean: 2.012682521343231
    alpha_1_25_expected_state_count: 15
    explicit_opposite_count: 0
    scientific_acceptance_all_passed: false
  next_gate:
    state: ready_after_repository_closure
    unblocked_experiment: C-U1-E4-TAPER-01
    reason: The user explicitly closed the scoped E4 long-horizon claim after reviewing the unchanged 4000-step evidence.
      The original 18/20 gate remains failed and no threshold, horizon, optimizer, seed, or classification was changed.
