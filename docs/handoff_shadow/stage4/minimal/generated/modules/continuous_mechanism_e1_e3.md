# Continuous C-U1 source and causal mechanism E1-E3

> Generated Stage 4 minimal-context shadow module. Do not edit manually.
> `docs/handoff.md` and `experiments/registry.yaml` remain authoritative.

- Module ID: `continuous_mechanism_e1_e3`
- Responsibility: Cover continuous far-field gradient-source identification and causal transmission into drift, task collapse, and variance or support contraction.
- Source hash: `67398f2fb2b5a8ebc7bd4e0f1e27193b23914dd404acab1dcfe0eee46bc562aa`

## Source 1: docs/handoff.md: # 3. 连续统一环境 C-U1 的详细设计 -> ### 3.6.2 E4：稳定外推—相变—远场控制

# 3. 连续统一环境 C-U1 的详细设计

## 3.1 状态与动作

- 状态：`s in R^6`；训练集和测试集分别采样，使用同一生成函数。
- 动作：`a in R^2`；策略为 state-conditioned Gaussian，均值与方差共同学习。
- 每个状态产生 state-dependent 的 `a_plus(s)`、`a_star(s)`、任务方向和正交方向。

### 3.1.1 “context/state”在小网络中的具体含义

这里的 context 不是自然语言上下文，而是输入给 MLP 的 6 维数值向量。每一个状态 `s` 代表一个不同的一步决策条件；环境通过固定生成函数把 `s` 映射为该条件下的 `a_plus(s)`、`a_star(s)` 和奖励地形。小网络学习的是函数 `s -> (mu(s), sigma(s))`，而不是记忆一个全局动作。

- **训练状态**：其 state-action-reward 样本参与参数更新。
- **测试状态**：由同一状态生成分布独立采样，但完全不参与训练，用于检查 MLP 是否学到状态到动作的映射，而不是仅记住训练状态。
- **一个状态不等于一个样本**：同一状态下可构造多个正动作、多个负动作和额外梯度探针，因此 transition 数等于“状态数 × 每状态动作数”。
- 当前环境原型使用 1024 个基础状态做不变量检查；上一轮提出的 4096 train / 4096 test、每状态 4 正 / 8 负只是**正式配置提案**，尚未获得用户确认，也尚未用于正式训练。

训练/测试状态拆分的唯一目的，是验证 state-conditioned 网络对未见数值输入的函数泛化。E1 的距离—梯度来源识别主要按状态聚合，不把同一状态的多个复制动作当作独立样本。

## 3.2 数据与奖励

Ground-truth reward 由动作到 `a_star(s)` 的二维距离决定，因此 `a_star` 是唯一最优动作。正样本分布位于 `a_plus` 周围；负样本位于经过 `a_minus` 的等奖励轮廓。等奖励轮廓上的所有负样本 reward/advantage 精确相同，但相对当前策略的距离不同。

## 3.3 同一环境如何支持四个实验

- E1 直接读取同一状态下等 advantage 的轮廓负样本，比较距离与梯度；
- E2 仅应用正样本梯度，轮廓负样本只作为 phantom gradient 监测对象；
- E3 应用正负梯度，并按当前策略距离动态划分 near/far 进行干预；
- E4 重点使用 `a_minus` 及其邻近负样本提供指向 `a_star` 的有益排斥，再加入远场轮廓样本观察从外推到失稳的转折。

## 3.4 需要预先讨论而不能擅自决定的设计项

1. 负轮廓角度数量与距离范围；
2. positive residual spread 是否固定以及是否加入 state-dependent 噪声；
3. advantage 使用固定真实 reward-baseline，还是增加 learned-critic 附录；
4. E4 中使用全部负轮廓还是只使用方向一致的近场子集作为有益负信号；
5. 训练步数、停止标准与正式 seeds。

在这些项目冻结前只做 invariant/smoke test，不宣称正式结果。


## 3.5 v13 冻结后的 C-U1 正式配置

用户已授权冻结以下配置并开始正式执行：

1. **状态与数据量：** `s ~ N(0,I_6)`；4096 个训练状态与 4096 个独立测试状态。每个状态构造 4 个正动作和 8 个负动作。训练按 state minibatch 取样，并同时读取该状态对应动作，避免把同状态复制动作当作独立 context。
2. **任务几何：** `a_star(s)=a_plus(s)+0.70 u(s)`；有益近场负动作 `a_minus(s)=a_plus(s)-0.50 u(s)`。8 个负动作位于以 `a_star` 为圆心、半径 1.20 的等奖励圆上，包含 `a_minus`，因此其 reward/advantage 在每个状态内严格相等，但相对策略距离不同。
3. **正样本条件残差：** 4 个正动作位于 `a_plus ±0.18u` 与 `a_plus ±0.18v`；该非零 residual spread 允许 positive-only 的 Gaussian 方差存在内部有限目标，避免把确定性 MLE 的方差坍缩误当作远场机制。
4. **奖励和 advantage：** `R(s,a)=exp(-||a-a_star(s)||^2/(2*0.75^2))`；固定 baseline 为 0.40。所有 advantage 在训练前计算并冻结。负动作 advantage 跨轮廓数值误差须低于 `1e-6`，所有正动作 advantage >0，所有负动作 advantage <0。
5. **策略：** 共享两层 MLP，state-conditioned 2D Gaussian mean 与标量 log-standard-deviation head；不使用人为方差 clamp。`log_sigma<-12` 是 support/variance contraction 边界事件；参数、log-sigma 或 sigma 输出的 NaN/Inf 单独记为数值失败；`log_sigma>12` 只能记为 unexpected positive-boundary event，不构成理论中的方差扩张分支。
6. **目标归一化：** 正、负部分分别按组取均值，更新写为 `g = g_pos + alpha*g_neg`，使 alpha 表示负向总质量相对强度，不由 4/8 样本数量机械决定。
7. **Near/Far：** 依据当前策略下标准化动作距离动态划分，正式阈值 `d=5.0`；阈值稳健性在开发集检查 `4.0/6.0`。Near/Far mask 只用于干预，不回传距离权重梯度。
8. **E4 有益负信号：** 只使用轮廓中位于 `a_minus` 方向的近场动作作为方向可靠负信号；其余轮廓动作作为远场压力源。这样 E4 检验的是“有益局部排斥 + 额外远场压力”的转折，不把方向相反的负动作混入有益外推定义。
9. **seeds：** 0–4 仅用于回归、阈值和 alpha 相变定位。E1/E2 使用 held-out 10–29；由于 E3 smoke 曾意外查看 seed 10，E3 为保持严格盲测改用 held-out 30–49。所有方法在各实验内部配对相同 seeds。
10. **收敛与终态：** 每 100 steps 评估；E3/E4 论文主训练统一使用 Adam，并分别记录 raw gradient norm 与 Adam parameter-update norm。稳定候选需通过全数据净动力场残差和 2× continuation 审计，且状态分类不反转；持续漂移则报告斜率、reward 失效时间和数值状态。最大步数按各 protocol 配置记录，不用固定步数冒充稳态。

该配置替代第 3.4 节中的“待讨论”状态；第 3.4 节保留作为决策 provenance，不删除。

## 3.6 v13 执行期勘误与 E4 正式协议冻结

### 3.6.1 正样本几何勘误（不破坏性覆盖）

- **原登记：** 第 3.5(3) 写成“四个正动作位于 `a_plus ±0.18u` 与 `a_plus ±0.18v`”。
- **问题：** 该写法与第 3.5(4) 的等 reward 设定、已经运行的代码和 E1/E2 结果不一致；这些四点相对 `a_star` 的距离并不严格相等。
- **正式实现与修正：** 四个正动作位于以 `a_star` 为圆心、半径 0.75 的等 reward 圆上，角度为 `pi±theta_1` 与 `pi±theta_2`，其中 `theta_1=0.20`，`theta_2` 由质心精确等于 `a_plus` 的方程确定。其条件残差总二阶矩为 `0.75^2-0.70^2=0.0725`，二维共享标准差的 positive-only 解析目标为 `sqrt(0.0725/2)=0.190394`。
- **证据：** C-U1 invariant、E1 与 E2 均使用该等 reward 实现；E2 的 20-seed 最终平均 `sigma=0.190419`，与解析值一致。
- **处理：** 第 3.5(3) 作为错误 provenance 保留，本节为正式替代记录；后续实验不改动已运行的数据生成器。

## Source 2: experiments/registry.yaml: experiments[C-U1-E3-ADAM-RERUN, C-U1-E1-COMP-01]

collection: experiments
entries:
- id: C-U1-E3-ADAM-RERUN
  environment: C-U1
  name: cu1_far_field_causal_intervention_unified_adam
  status: long_run_validated
  claim: Test whether anomalous far-field negative gradients causally transmit into mean drift, task-performance collapse,
    and support or variance contraction under the same Adam optimizer used by the paper-facing C-U1 pipeline.
  role: controlled_mechanism_identification
  execution_class: historical_formal
  historical_formal_execution:
    channel_status: grandfathered_completed_run
    future_rerun_requires_channel: hardened-v1
  code_entrypoint: src/drpo/drpo_cu1_e1_e4_oneclick.py
  command:
  - python
  - src/drpo/drpo_cu1_e1_e4_oneclick.py
  - --stage
  - e3
  - --output-root
  - outputs/cu1_e3_adam
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
    fixed_variance_lr: 0.0001
    learnable_variance_lr: 0.0005
  data:
    train_states: 4096
    test_states: 4096
    positive_actions_per_state: 4
    negative_actions_per_state: 8
    terminology: held_out_context_generalization
    fixed_advantage: true
  held_out_seeds:
  - 30
  - 31
  - 32
  - 33
  - 34
  - 35
  - 36
  - 37
  - 38
  - 39
  - 40
  - 41
  - 42
  - 43
  - 44
  - 45
  - 46
  - 47
  - 48
  - 49
  primary_methods:
  - baseline
  - near_zero
  - far_zero
  - far_cap
  appendix_controls:
  - global_scale
  - far_to_near
  fixed_variance:
    sigma: 0.1903943276465978
    alpha: 1.4
    max_steps: 2000
  learnable_variance:
    alpha: 0.15
    max_steps: 2000
    variance_clamp: false
    full_state_boundary_audit: true
    first_scientific_boundary_event: support_contraction
    positive_boundary_event_name: unexpected_support_expansion
  metrics:
  - held_out_context_reward
  - task_failure_onset
  - mean_drift
  - log_sigma_min_max_all_states
  - support_contraction_onset
  - raw_positive_near_far_gradient_norms
  - raw_total_gradient_norm
  - adam_parameter_update_norm
  - parameter_log_sigma_sigma_output_finiteness
  reporting_separation:
  - task_performance_collapse
  - support_or_variance_contraction
  - nan_inf_numerical_failure
  - unexpected_positive_boundary_event
  controls_note: Raw-gradient budget matching does not imply Adam parameter-update matching.
  terminal_audit: required
  formal_run_status: delivered
  execution:
    state: delivered
    run_id: cu1_e3_adam_ac286a4_run1
    completion_mode: sharded_recovery_with_terminal_aggregation
    last_heartbeat_utc: null
    process_exit_code: null
    note: No single supervisor exit code represents the sharded recovered run; all expected scientific outputs and terminal
      gates were audited.
  evidence:
    raw_complete: true
    terminal_audited: true
    terminal_audit_all_checks_passed: true
    expected_fixed_rows: 120
    actual_fixed_rows: 120
    expected_learnable_rows: 100
    actual_learnable_rows: 100
    missing_required_files: 0
    package_created: true
    package_filename: DRPO_CU1_E3_ADAM_AC286A4_FINAL.zip
    package_sha256: 2b8bfdbe6f33ed1db9dc1e59f6e9fbdb6c224c7b31b1326a7f2fbaeeaaaf522b
    delivered_to_user: true
    applied_commit: null
    compact_result_path: outputs/cu1_e3_adam
  provenance:
    run_commit: ac286a46b8ffad898dfad0e7e9188b1d2e81052a
    repository_closure_base_commit: 04a5d342b4102f863f6b5a2aae1fb750295349b5
    runner_git_blob: 9b509ccd224f9cf7b4f9d79b58e04d76bad62e69
    runner_sha256: 502c345289d2b5b7c34832246478b64c33a1789e80ddcab7f6194cb09b0eac6f
    source_mode: exact_committed_runner_blob_plus_committed_handoff_registry_snapshots
    local_git_object_available_at_launch: false
    local_git_object_limitation: Shell DNS could not reach GitHub in the launch environment.
    aggregation_workaround: JSON tuple/list representation normalization only; no scientific inputs or outputs changed.
  result_summary:
    fixed_variance:
      baseline:
        reward_mean: 2.2246353978516707e-06
        task_collapse_count: 20
        support_event_count: 0
        nan_inf_count: 0
      near_zero:
        reward_mean: 2.17606851151686e-06
        task_collapse_count: 20
        support_event_count: 0
        nan_inf_count: 0
      far_zero:
        reward_mean: 0.7393616884946823
        reward_ci95:
        - 0.7388725359139248
        - 0.7398508410754399
        task_collapse_count: 0
        support_event_count: 0
        nan_inf_count: 0
      far_cap:
        reward_mean: 0.733071705698967
        reward_ci95:
        - 0.7325694714536328
        - 0.7335739399443011
        task_collapse_count: 0
        support_event_count: 0
        nan_inf_count: 0
      global_scale:
        reward_mean: 0.5990572392940521
        task_collapse_count: 0
      far_to_near:
        reward_mean: 0.8753230214118958
        task_collapse_count: 0
    learnable_variance:
      baseline:
        support_contraction_count: 20
        support_onset_median: 73
        unexpected_expansion_count: 0
        nan_inf_count: 0
      near_zero:
        support_contraction_count: 20
        support_onset_median: 73
        unexpected_expansion_count: 0
        nan_inf_count: 0
      far_zero:
        support_event_count: 0
        nan_inf_count: 0
      far_cap:
        support_event_count: 0
        nan_inf_count: 0
      global_scale:
        support_event_count: 0
        nan_inf_count: 0
    total_method_seed_runs: 220
    total_nan_inf_count: 0
    learnable_method_seed_runs: 100
    unexpected_support_expansion_count: 0
  paper_use:
    suitable: true
    main_text:
    - fixed_variance_baseline
    - fixed_variance_near_zero
    - fixed_variance_far_zero
    - fixed_variance_far_cap
    complementary_panel_or_appendix:
    - learnable_variance_support_contraction
    appendix_controls:
    - global_scale
    - far_to_near
    prohibited_claims:
    - OOD_generalization
    - universal_method_ranking
    - variance_explosion
  scientific_status: long_run_validated
  supersedes_for_paper_facing_result: C-U1 E3 SGD reconstruction and recovered transient runs
  preserved_history: true
- id: C-U1-E1-COMP-01
  environment: C-U1
  name: gaussian_output_score_componentwise_growth_law
  status: pilot
  claim: Under the frozen C-U1 E1/E2 protocol and equal negative advantages, the Gaussian mean output-score branch is linear
    in raw policy-sample distance after variance normalization, while the corrected log-scale branch is quadratic; the existing
    joint E1 score ratio is reconstructed by these two components.
  role: controlled_mechanism_supplement
  execution_class: pilot
  does_not_replace:
  - C-U1-E1
  - C-U1-E2
  scope:
    output_space_gaussian_only: true
    neural_network_pullback: excluded
    method_ranking: not_tested
  code_entrypoint: src/drpo/cu1_e1_componentwise_rerun.py
  base_runner: src/drpo/drpo_cu1_e1_e4_oneclick.py
  frozen_protocol:
    train_states: 4096
    test_states: 4096
    state_distribution: Normal(0,I_6)
    positive_actions_per_state: 4
    equal_advantage_negative_actions_per_state: 8
    seeds:
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
    terminal_positive_gradient_norm_max: 0.001
  metrics:
  - normalized_mean_distance_loglog_slope
  - normalized_corrected_log_scale_distance_loglog_slope
  - mean_score_far_near_ratio
  - log_scale_score_abs_far_near_ratio
  - corrected_log_scale_far_near_ratio
  - joint_output_score_far_near_ratio
  - analytic_autograd_relative_error
  - existing_e1_joint_ratio_reconstruction_error
  terminal_audit: required
  result_boundary: This experiment can support a nonlinear Gaussian far-field mechanism motivation but cannot establish that
    exponential tapering outperforms linear, global, SBRC, hybrid, or positive-only controls.
  independent_external_validation_boundary:
    hopper_condition: E7 uses real D4RL Hopper data, a learned critic, and an independently trained actor, and observes that
      the log-scale branch enters a squared-distance-dominant far-field regime with measurable contribution to full-parameter
      gradients or dynamics.
    counts_as: external_replication_of_mechanism_applicability_and_quadratic_dominance
    does_not_count_as: independent_proof_of_the_gaussian_score_identity
  pilot_results_2026_06_25:
    seeds_completed: 20
    scientific_checks_passed: true
    terminal_audit_passed: true
    held_out_context_reward_mean: 0.646788
    held_out_context_reward_ci95:
    - 0.646657
    - 0.64692
    learned_sigma_mean: 0.190726
    final_positive_gradient_norm_mean: 0.0006436523
    final_positive_gradient_norm_max: 0.0009232561
    raw_distance_far_near_ratio_mean: 3.797862
    mean_score_far_near_ratio_mean: 3.797862
    raw_log_scale_abs_far_near_ratio_mean: 19.970219
    corrected_log_scale_far_near_ratio_mean: 14.435378
    joint_output_score_far_near_ratio_mean: 7.563755
    joint_output_score_far_near_ratio_ci95:
    - 7.554059
    - 7.574104
    normalized_mean_distance_loglog_slope: 1.0
    normalized_corrected_log_scale_distance_loglog_slope: 2.0000000017
    max_output_autograd_relative_error: 2.47651e-07
    max_existing_joint_ratio_reconstruction_error: 9.53674e-07
    interpretation: The Gaussian output-space mean branch is exactly linear after variance normalization and the corrected
      log-scale branch is exactly quadratic. The uncorrected log-scale score includes the additive minus-action-dim term and
      is therefore described as asymptotically quadratic.
  provenance:
    pilot_run_source_commit: 1962442aea7037fac6b57e4e9232850c69e5c1b9
    update_package_base_commit: a9e0d860a6f03d1be12280885002c24ba2f1b66a
    base_is_direct_successor_of_pilot_source: true
    c_u1_runner_changed_between_commits: false
    current_scientific_classification: reconstructed_source_pilot
    consistency_check: matches_v17_formal_e1_e2_ranges
    formal_upgrade_requires: applied_and_committed_clean_rerun
  execution:
    state: delivered
    run_id: cu1-e1-comp-20260625-cpu-pilot
    last_heartbeat_utc: 2026-06-25 07:31:37+00:00
    process_exit_code: 0
    runtime: cpu
  evidence:
    raw_complete: true
    terminal_audited: true
    package_created: true
    package_filename: DRPO_CU1_E1_COMPONENTWISE_A9E0D86_UPDATE.zip
    package_sha256: null
    package_checksum_note: ZIP self-hash is reported at delivery; internal files are covered by SHA256SUMS.txt
    delivered_to_user: true
    applied_commit: null
    result_path: external_artifact/C-U1-E1-COMP-01-pilot
    scientific_status: pilot
  artifact_budget:
    main_package_hard_limit_mib: 25
    retain_terminal_checkpoints_for_all_seeds: true
    expected_checkpoint_total_mib_max: 5
