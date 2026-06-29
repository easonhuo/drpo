# Categorical D-U1 E6 shared-semantic generalization

> Generated Stage 4 minimal-context shadow module. Do not edit manually.
> `docs/handoff.md` and `experiments/registry.yaml` remain authoritative.

- Module ID: `categorical_e6_generalization`
- Responsibility: Cover positive-only ceiling, controlled local-negative benefit, support-boundary separation, semantic alignment, and structured support-gap successors.
- Source hash: `058db1516006e5dac6f55e6138ea754bea419db1e0da13851cd3453a1f26c236`

## Source 1: docs/handoff.md: ## 3.7.3 E6 共享语义 pilot `D-U1-E6-SEMANTIC-PILOT-01` -> ## 3.8 C-U1 共享实现与二次阶方法实验 `C-U1-E4-TAPER-01`

## 3.7.3 E6 共享语义 pilot `D-U1-E6-SEMANTIC-PILOT-01`

1. **实验职责：** E6 不重复 E5 的 direct-softmax/support-boundary 结论。它检验共享 semantic representation 下，受控 local negative 是否能把策略从 positive demonstrations 推向训练中未展示的 hidden optimal action，并检验 far pressure 是否导致 task failure 或 support/temperature boundary。
2. **状态与术语：** train/test contexts 独立采样自相同 `N(0,I_6)`，因此只报告同分布 held-out-context / unseen-state generalization；本实验没有显式 state distribution shift，不使用 OOD generalization。
3. **开发身份与当前状态：** experiment ID 为 `D-U1-E6-SEMANTIC-PILOT-01`，seeds 固定为 0--4，科学状态为 `pilot`。105/105 development runs 已完成、审计并交付；这不升级为 formal long-run。独立 formal ID `D-U1-E6-SEMANTIC-LONGRUN-01` 已在 untouched seeds `10--29` 上完成 360/360 runs、终态审计和交付，科学状态为 `long_run_validated`。
4. **实现入口：** `src/drpo/du1_e6_semantic.py`；开发配置 `configs/du1_e6_semantic_pilot.yaml`；实现说明 `src/drpo/README_DU1_E6_SEMANTIC.md`。runner 只写普通 JSON/JSONL/CSV/YAML，不自行打包。
5. **E6-A：** fixed concentration 下比较 positive-only 与 local-negative alpha scan，观察 hidden-optimal probability、positive-support probability、expected semantic reward 与 normalized semantic extrapolation。alpha 网格是开发值，不是正式冻结值。
6. **E6-B：** learnable concentration 下比较 `positive_only / local_only / uncontrolled / near_zero / far_zero / far_cap / budget_matched_global`。`local_only` 与 `far_zero` 在数学更新上同义但保留不同协议语义；Far-cap 与 Global 只匹配 raw controlled-negative norm。
7. **E6-C：** 对同一 reward-side catalogue、hidden optimum、demonstrations、negative sets 与 fixed advantages，只独立置换 policy-side action embeddings。若 suppression 仍存在但 hidden-optimum 改善消失，才支持 shared semantic alignment 是外推收益的必要通道；pilot 不构成正式结论。
8. **配对与审计结果：** 同一 seed 内共享网络初始化与 minibatch index stream；105/105 runs 完整。任务性能崩溃为 0/105，support/temperature boundary 为 56/105，NaN/Inf 为 0/105；三类事件继续分报。
9. **终态边界：** fixed-concentration 的 30/30 runs 均未通过两尾窗 provisional plateau，可学习 concentration 的负压力分支普遍触发 support boundary；正式 2x 延长未执行。因此本 pilot 不能升级为 long-run validated、稳定 fixed point 或正式方法排名。
10. **下一门禁：** focused-development freeze 已由用户批准，formal runner/config 已实现并激活。应用本版后直接启动 E6 long-run；运行完成后必须先做 terminal audit、durable packaging 和 delivery，随后才可进入 E6-TAPER。E4-TAPER 与 E6 的科学职责和输出仍相互独立。

---

## Source 2: docs/handoff.md: HANDOFF-DELTA-BLOCKs matching 'D-U1-E6-'

### Delta block `after_heading:v50-stage3-shadow-bootstrap`

> **v50 增量登记：治理 Pipeline Stage 3 `HANDOFF_DELTA.yaml` shadow mode 启动（不删除 v49 及更早内容）**
>
> - Stage 1 与当前 Stage 2 的 `closed_maintenance_only` 状态保持不变；本版只启动当前 Stage 3，不改变任何科学实验变量、seeds、阈值、结果或执行顺序。`D-U1-E6-CONDITIONAL-GAP-01` 继续保持 **not_run + implemented + ready + active**。
> - Stage 3 状态由 `ready_not_started` 迁移为 `shadow_active`。`docs/handoff.md` 在整个 shadow 期继续是唯一权威研究 Master；结构化 delta 只生成 candidate 并与人工 handoff 比较，不得替换正式 handoff。
> - 新增 `docs/handoff_delta_protocol.md`、机器策略 `docs/handoff_delta_policy.yaml`、状态机 `docs/handoff_delta_state_machines.yaml`、确定性入口 `scripts/handoff_delta_shadow.py` 与三级验收入口 `scripts/run_handoff_delta_acceptance.py`。版本 1 只允许 heading rename、heading 后插入和 section 末尾 append，不允许任意文本替换或破坏性删除。
> - Fast Gate 对每个 handoff / registry / delta 相关更新执行本地确定性 replay、base/hash、幂等、历史 ID、registry transition 与 candidate/manual exact-match 检查；禁止网络和 LLM 作为阻塞式 oracle。目标 p95 不超过 5 秒，硬上限 15 秒。
> - Standard Regression 在 schema、renderer、状态机、冲突规则、parser/index 或 operation 变化时运行，目标 60 秒。Full Acceptance 在 shadow 激活前、authority cutover 前、schema 主版本或架构变化、累计 20 次相关更新、7 天内发生相关更新的兜底周期、critical mismatch 修复后运行，目标 15 分钟。
> - 本更新使用 `GOV-STAGE3-SHADOW-BOOTSTRAP-2026-06-27/HANDOFF_DELTA.yaml` 自举 replay；candidate 与人工 v50 handoff 必须字节级一致。该自举通过只证明实现与门禁可运行，不等于 Stage 3 已验收或可以切换权威路径。

### Delta block `after_heading:v51-du1-e6-semantic-gap-formal`

> **v51 增量登记：`D-U1-E6-CONDITIONAL-GAP-01` 结果闭环与 `D-U1-E6-SEMANTIC-GAP-LONGRUN-01` 最小改动 formal freeze（不删除 v50 及更早内容）**
>
> - `D-U1-E6-CONDITIONAL-GAP-01` 已在 clean scientific run commit `7a70278f3d6061379c81f33e82d93ead86484908` 上完成 frozen matrix `200/200` runs、terminal audit 与 raw-complete artifact。三类事件严格分报：task-performance collapse `77/200`、support/temperature boundary `0/200`、NaN/Inf numerical failure `0/200`。全实验只有 `49/200` terminal plateau，`151/200` 为 persistent-drift-or-inconclusive，因此科学状态只登记为 **有限训练步数验证（finite-step validated）**，禁止升级为 long-run validated，并禁止稳态排名。
> - 该 group-based conditional-gap 实验保留为大缺口与强压力的 stress diagnostic。其 structured-gap local `alpha=0.5` 虽提高 withheld-block reward，却降低 overall reward；`alpha=1.5` 与 far-pressure stress 不属于后续正式方法域。该结果不得推翻旧 `D-U1-E6-SEMANTIC-LONGRUN-01` 已锁定的“适度负优势可在 overall reward 上超过 Positive-only”结论。
> - 新实验 `D-U1-E6-SEMANTIC-GAP-LONGRUN-01` 是旧 64-action shared-semantic E6 的正式最小改动 successor。继续使用 6D state、64 个 4D semantic actions、旧 `t_plus/t_star/t_minus` reward 几何、4 positive / 1 local negative / 4 far negatives、固定相等 advantage、共享 `SemanticPolicy`、fixed concentration `8.0`、Adam `lr=1e-3`、batch 128。唯一核心环境干预是在 exactly 50% contexts 上，将按该状态 reward similarity 排名前 25% 的 16 个动作从 positive/local/far 三类日志角色中移除；完整 reward oracle 与 64-action evaluation space 保持不变，并要求每个 action 在全局日志中仍然出现。
> - train/test contexts 继续独立采样自同一 `N(0,I_6)`，因此只称 same-distribution held-out-context generalization / structured state-action support gap，不称 state-distribution OOD generalization。
> - 临时 sandbox 未写入 registry、未使用正式 seeds，也不构成正式结果。它只用于 candidate qualification：64-action 最小改动环境复现了 `Positive-only ceiling -> intermediate-alpha benefit -> stronger-alpha reversal`；长 horizon 诊断显示 `alpha=1` 相对 Positive-only 的 overall reward 差距会随训练延长扩大。
> - 正式 alpha 域冻结为 `[0.0,0.25,0.5,0.75,1.0]`。`alpha=0` 是 Positive-only，`alpha=1` 是不抑制原始负梯度；`alpha>1` 不属于方法域，也不进入正式实验。唯一主指标为 overall expected semantic reward，并登记 paired difference vs Positive-only 与 4k/8k/16k/24k/32k trajectory；hidden-optimal probability、support 与 gradient 只作诊断。
> - 正式协议使用 untouched seeds `150--169`，禁止 sandbox seeds `900--909` 进入正式聚合；5 个 alpha × 20 seeds，共 `100` method-seed runs。最大 `32000` steps、每 `200` steps evaluation，每 5 seeds 写 persistent-local checkpoint；terminal windows 为 `16000--24000` 与 `24000--32000`。
> - task-performance collapse、support/temperature boundary 与 NaN/Inf numerical failure 继续分报。完整运行和登记窗口审计完成可形成有限步正式证据；只有全部登记 runs 达到 formal terminal plateau 时，才允许声称稳态方法排名。若仍持续漂移，必须报告 trajectory 与 finite-step status，禁止预设最佳 alpha。
> - 代码复用 `src/drpo/du1_e6_semantic.py`，新增最小差异 validator `src/drpo/du1_e6_semantic_gap.py`、formal entrypoint `src/drpo/du1_e6_semantic_gap_longrun.py`、冻结配置 `configs/du1_e6_semantic_gap_longrun.yaml` 和 hardened wrapper `scripts/run_du1_e6_semantic_gap_longrun.py`。应用并提交本版后，实验状态为 **implemented + ready + active + not_run**；正式训练不得在 dirty worktree 或未匹配 `origin/main` 时启动。
> - 本更新重基于当前 `main` commit `1fa7f04d4830e4d562ab147dbb11dfa8cecc9b5d`，并保留治理 Pipeline Stage 3 shadow mode 的全部新内容。`D-U1-E6-TAPER-01` 在本 successor terminal-audited、packaged、delivered 之前继续 blocked。

### Delta block `after_heading:v55-du1-e6-semantic-gap-result-closure`

> **v55 增量登记：`D-U1-E6-SEMANTIC-GAP-LONGRUN-01` 正式结果、终态审计与仓库闭环（不删除 v54 及更早内容）**
>
> - 冻结协议在科学运行 commit `0907c3c0e76fc836c2bf2b752abf554c17f79f22` 上完成 `100/100` method-seed runs；正式 seeds 为 `150--169`，sandbox seeds `900--909` 未进入正式聚合。raw-complete 包 SHA-256 为 `65630159ef85c665a3a0ac0eba5cbf751ecb77a929f267423f7a6d9a8e5c4fbf`。
> - 所有 required outputs 与 terminal audits 均存在且被接受，并完成预注册的 2× horizon 扩展到 32000 steps；但只有 `45/100` runs 达到 formal terminal plateau，`55/100` 为 persistent-drift-or-inconclusive。因此科学状态只升级为 **有限训练步数验证（finite-step validated）**，不得形成全 alpha 稳态方法排名。
> - 三类事件严格分报且均为 0：task-performance collapse `0/100`、support/temperature boundary `0/100`、NaN/Inf numerical failure `0/100`。这不等于全部方法已收敛。
> - 32k 时 Positive-only reward 为 `0.741309`；`alpha=0.25` 与 `alpha=0.50` 分别为 `0.766269` 和 `0.765975`，相对 Positive-only 的 paired mean gains 为 `+0.024960` 与 `+0.024666`，均为 `20/20` seeds 胜出。
> - `alpha=1.0` 相对 Positive-only 的差距从 4k 的 `+0.003943` 转为 8k/16k/24k/32k 的 `-0.013741/-0.039167/-0.053227/-0.061085`，自 8k 起均为 `0/20` 胜出；`alpha=0.75` 到 32k 为 `-0.001978`、9 胜 11 负且 0/20 plateau，属于有限 horizon 反转与持续漂移证据，不是稳态排名。
> - 该结果支持：Positive-only 存在有限 horizon overall-reward ceiling、适度保留负梯度可改善同分布 held-out-context reward、完全不抑制的原始负梯度会产生随 horizon 扩大的任务退化。它不支持 universal alpha optimum、跨任务方法优越性或 categorical policy 的 Gaussian 二次远场律。
> - 训练/测试 contexts 仍独立采样自同一分布；只能称 **same-distribution held-out-context generalization / structured state-action support gap**，不得称 state-distribution OOD generalization。
> - 用户上传的 raw-complete ZIP 是不可变实验/恢复证据，不是 repository update；`drpo-update` 在 `package_extract` 阶段拒绝它是预期行为。仓库只纳入 compact summaries、terminal audit、provenance 与 artifact index，33.6 MB trajectory 保持 persistent-local 索引。
> - `D-U1-E6-TAPER-01` 的 semantic-gap successor delivery blocker 已满足并移除，但 semantic remoteness coordinate、paired protocol、新 untouched seeds 与独立 formal runner 仍未冻结/实现；其状态继续是 **not_run + not_implemented + review-required/blocked**，不得自动启动。
> - 本仓库闭环更新基于 `main` commit `fa225510e3e3e4616f36d8f586611aa6af79bf6e`；未重跑正式实验，也未修改冻结变量、seeds、阈值或训练协议。

### Delta block `after_heading:v56-e6-parent-closure-route-release`

> **v56 增量登记：E6 父 claim 关闭与 E7-MECH 路线解锁（不删除 v55 及更早内容）**
>
> - 用户在确认 `main` commit `e70f0d84256cdeb6ebbf80b0495a043582787bf6` 已提交后，批准对 **E6 父实验/父 claim** 做范围受限关闭。关闭依据是：`D-U1-E6-SEMANTIC-LONGRUN-01` 的 `360/360` long-run validated 主结果、`D-U1-E6-SEMANTIC-GAP-LONGRUN-01` 的 `100/100` finite-step robustness successor，以及 `D-U1-E6-CONDITIONAL-GAP-01` 的 `200/200` finite-step stress diagnostic。
> - 本次关闭锁定五项论文可用结论：Positive-only 在共享语义 categorical 环境存在 imitation ceiling；适度受控负信号可改善同分布 held-out-context / unseen-action 表现；过强或不抑制负压力会出现性能反转或随 horizon 扩大的退化；semantic alignment 是观察到的未见动作迁移的重要排他性条件；任务性能崩溃、support/temperature boundary 与 NaN/Inf 必须继续分报。
> - 本次关闭不把两个 gap 子实验升级为 long-run validated，也不声称全 alpha 稳态排名、universal alpha optimum、state-distribution OOD generalization、categorical policy 的 Gaussian 二次远场律或跨任务方法优越性。`45/100` semantic-gap plateau 与 `49/200` conditional-gap plateau 的终态边界原样保留。
> - `D-U1-E6-TAPER-01` 降为**可选、独立、非门禁**的方法形状比较：它仍是 `not_run + not_implemented + blocked`，若未来执行，必须另行冻结 semantic remoteness coordinate、paired protocol、全新 untouched seeds 与独立 runner；但它不再是 E6 父 claim 关闭或 E7-MECH 启动的前置条件。
> - `EXT-H-E7-Q2` 由 `blocked/blocked` 迁移为 **ready/active**，科学状态仍为 `not_run`。该迁移只开放已经冻结和实现的 Hopper mechanism formal protocol，不代表 E7 已运行或已有结果。`EXT-H-E7-BENCH-01` 继续 blocked，但依赖收缩为 E7-Q2 交付和随后冻结 shortlist，不再依赖可选 E6-TAPER。
> - 本更新只修改研究治理、路线和相应测试/操作说明；未重跑实验，未更改任何冻结变量、数据规模、seeds、阈值、收敛标准或方法职责。

### Delta block `section_end:v55-du1-e6-semantic-gap-current-gate`

- **D-U1 v55 覆盖：** `D-U1-E6-SEMANTIC-GAP-LONGRUN-01` 已完成 `100/100` 正式 runs、2× horizon 与终态审计，科学状态为 **有限训练步数验证**；45/100 plateau、55/100 persistent-drift-or-inconclusive，禁止稳态方法排名或无新登记重跑。`D-U1-E6-TAPER-01` 的 successor-delivery 条件已满足，但其四项协议/实现门禁仍未完成，继续 review-required + blocked。

### Delta block `section_end:v56-e6-parent-closure-current-gate`

- **v56 E6 父 claim 关闭覆盖：** E6 的论文核心 claim 现已范围受限关闭；主 long-run 与两个 gap 子实验的原科学状态分别保持 `long_run_validated / finite_step_validated / finite_step_validated`。`D-U1-E6-TAPER-01` 保留为可选非门禁未来工作。当前下一正式 route item 为 `EXT-H-E7-Q2`，registry 状态为 **implemented + ready + active + not_run**；启动后仍须走 canonical hardened guard，且在 raw-complete、终态审计、打包和交付前不得声称 E7 完成。

### Delta block `section_end:v55-du1-e6-semantic-gap-completion-status`

**v55 E6 Semantic-Gap 结果补充：** `D-U1-E6-SEMANTIC-GAP-LONGRUN-01` 已完成 100/100 runs。32k 时 `alpha=0.25/0.50` 均 20/20 胜过 Positive-only；`alpha=1.0` 相对差距随 8k→32k 由 `-0.013741` 扩大至 `-0.061085`，20/20 失败。由于仅 45/100 terminal plateau，论文可用状态限定为有限 horizon trajectory 与 paired finite-step claim，不允许全方法稳态排名。三类失效事件分别为 0/100、0/100、0/100。

### Delta block `section_end:v56-e6-parent-closure-execution-order`

13. **v56 执行覆盖：** E6 父 claim 已关闭，`D-U1-E6-TAPER-01` 改为可选非门禁 future study；当前直接进入已实现且 registry 为 ready/active 的 `EXT-H-E7-Q2`（E7-MECH）。E7-Q2 仍为 not_run，必须先完成正式运行、终态审计、打包与交付；其后才允许冻结并实施 `EXT-H-E7-BENCH-01`。E8-MECH/V4.3 与 E8-SCALE 的相对顺序不变。

## Source 3: experiments/registry.yaml: experiments[D-U1-E6-SEMANTIC-LONGRUN-01, D-U1-E6-SEMANTIC-GAP-LONGRUN-01, D-U1-E6-CONDITIONAL-GAP-01]

collection: experiments
entries:
- id: D-U1-E6-SEMANTIC-LONGRUN-01
  environment: D-U1
  name: shared_semantic_categorical_extrapolation_longrun
  status: long_run_validated
  parent_experiment: E6
  predecessor: D-U1-E6-SEMANTIC-FOCUSED-DEV-01
  registration_base_commit: eb6a90d55127cead4d95bd0a85a78f32c47ff56a
  claim: On untouched held-out seeds, formally test the positive-only ceiling, controlled local-negative shared-semantic extrapolation,
    far-pressure task or support failure, and the policy-side semantic-alignment exclusion control.
  role: formal_controlled_shared_semantic_categorical_experiment
  execution_class: formal
  implementation_state: implemented
  registry_scope: development_registration_with_canonical_formal_execution
  execution_gate:
    state: blocked
    blocked_by:
    - completed_formal_execution_no_rerun_without_new_registration
    approval_record: user_approved_2026-06-27_exact_focused_dev_freeze
    blocker_resolution_experiment: D-U1-E6-SEMANTIC-FOCUSED-DEV-01
    blocking_reason: The frozen formal run is complete, terminal-audited, packaged, and delivered. Re-running the held-out
      seeds requires a separately registered protocol.
  formal_execution:
    channel_ref: hardened-v1
    activation_state: blocked
    entrypoint_status: implemented
    entrypoint: src/drpo/du1_e6_semantic_longrun.py
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
  code_entrypoint: src/drpo/du1_e6_semantic_longrun.py
  one_click_entrypoint: scripts/run_du1_e6_semantic_longrun.py
  config_path: configs/du1_e6_semantic_longrun.yaml
  formal_launch_template: python3 scripts/run_experiment_guard_hardened.py --experiment-id D-U1-E6-SEMANTIC-LONGRUN-01 --repo-root
    . --output-root experiments/results/D-U1-E6-SEMANTIC-LONGRUN-01/run_001 --artifact-output artifacts/D-U1-E6-SEMANTIC-LONGRUN-01_RAW_COMPLETE.zip
    --run-class formal --expected-commit "$(git rev-parse HEAD)" --require-origin-main-match --required-output RUN_COMPLETE.json
    --required-output terminal_audit.json --required-output aggregate_summary.json --required-output per_run_summary.csv --required-output
    formal_protocol_freeze.json --source-file src/drpo/du1_e6_semantic_longrun.py --source-file src/drpo/du1_e6_semantic.py
    --source-file configs/du1_e6_semantic_longrun.yaml --progress-glob checkpoints/*/CHECKPOINT_COMPLETE.json -- python3 src/drpo/du1_e6_semantic_longrun.py
    --config configs/du1_e6_semantic_longrun.yaml --output-root experiments/results/D-U1-E6-SEMANTIC-LONGRUN-01/run_001 --device
    cpu
  formal_parameter_freeze: true
  freeze_approval:
    approved_by_user: true
    approval_date: '2026-06-27'
    source: focused_development_recommendation
    automatic_retuning_allowed: false
  development_seeds_forbidden_in_formal_aggregation:
  - 0
  - 1
  - 2
  - 3
  - 4
  held_out_seeds:
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
  protocol:
    fixed_advantage: true
    critic_or_value_network: false
    importance_sampling: false
    state_dim: 6
    semantic_dim: 4
    action_count: 64
    train_states: 2048
    test_states: 2048
    state_distribution: standard_normal
    train_test_relation: independent_same_distribution
    terminology: held_out_context_generalization
    positive_actions_per_state: 4
    local_negative_actions_per_state: 1
    far_negative_actions_per_state: 4
    hidden_optimum_excluded_from_positive_demonstrations: true
    optimizer: Adam
    learning_rate: 0.001
    maximum_steps: 8000
    evaluation_interval_steps: 50
    fixed_concentration: 8.0
    fixed_alpha_grid:
    - 0.0
    - 0.25
    - 0.5
    - 0.75
    initial_learnable_concentration: 8.0
    learnable_concentration_upper_clamp: false
    learnable_local_alpha: 0.1
    far_pressure_stress_lambda: 0.05
    formal_method_matrix:
    - positive_only
    - far_zero
    - uncontrolled
    - near_zero
    - far_cap
    - budget_matched_global
    semantic_alignment_control_methods:
    - positive_only
    - far_zero
    - uncontrolled
    - far_cap
    semantic_alignment_modes:
    - aligned
    - shuffled
    far_cap_ratio_to_weighted_local_gradient: 1.0
  events:
    task_collapse_ratio_to_paired_positive_only: 0.2
    effective_support_boundary: 1.5
    concentration_warning: 80.0
  checkpointing:
    seed_block_size: 5
    seed_blocks:
    - - 10
      - 11
      - 12
      - 13
      - 14
    - - 15
      - 16
      - 17
      - 18
      - 19
    - - 20
      - 21
      - 22
      - 23
      - 24
    - - 25
      - 26
      - 27
      - 28
      - 29
    persistence: persistent_local
    write_compact_manifest_after_each_block: true
  terminal_audit:
    mode: formal_two_x_windows
    development_reference_horizon_steps: 4000
    formal_horizon_steps: 8000
    formal_extension_factor: 2.0
    window_1_steps:
    - 4000
    - 6000
    window_2_steps:
    - 6000
    - 8000
    metric_window_mean_abs_tolerances:
      test_expected_semantic_reward: 0.01
      test_hidden_optimal_probability: 0.02
      test_normalized_semantic_extrapolation: 0.08
      test_entropy_mean: 0.08
    raw_total_gradient_median_ratio_max: 1.25
    adam_update_median_ratio_max: 1.25
    scientific_failure_outcomes_are_results: true
  reporting_separation:
  - task_performance_collapse
  - support_or_temperature_boundary
  - nan_inf_numerical_failure
  required_outputs:
  - RUN_COMPLETE.json
  - scientific_run_manifest.json
  - terminal_audit.json
  - aggregate_summary.json
  - per_run_summary.csv
  - formal_protocol_freeze.json
  - run_manifest.json
  non_claims:
  - OOD_generalization
  - Transformer_external_validity
  - cross_task_method_superiority
  - Gaussian_quadratic_gradient_law_for_categorical_policy
  execution:
    state: delivered
    run_id: du1_e6_semantic_longrun_eb5e126_run001
    start_utc: '2026-06-26T22:45:21.924397+00:00'
    end_utc: '2026-06-26T23:16:11.850511+00:00'
    elapsed_seconds: 1849.926
    process_exit_code: 0
    runtime: cpu
  evidence:
    implementation_tests_passed: true
    run_started: true
    raw_complete: true
    terminal_audited: true
    terminal_audit_all_checks_passed: true
    formal_two_x_extension_performed: true
    expected_runs: 360
    actual_runs: 360
    task_performance_collapse_events: 0
    support_or_temperature_boundary_events: 120
    nan_inf_numerical_events: 0
    package_created: true
    raw_complete_package_filename: D-U1-E6-SEMANTIC-LONGRUN-01_RAW_COMPLETE.zip
    raw_complete_package_sha256: e098d4dd0483a661468db0cb1c4b67e4e563e2426a6aa078fe7b808f7ac691fa
    final_repository_closure_package_filename: DRPO_DU1_E6_LONGRUN_CLOSURE_A1672D9_UPDATE.zip
    final_repository_closure_package_sha256: null
    final_repository_closure_package_sha256_status: not_recorded_in_repository_evidence
    delivered_to_user: true
    repository_applied: true
    applied_commit: ff2afe443167154eae5de7871cda83f3aba9a89e
    repository_application_evidence: current_main_contains_e6_closure_and_compact_results
    compact_result_path: outputs/du1_e6_semantic_longrun
    scientific_status: long_run_validated
  provenance:
    run_commit: eb5e12626026854f44f4698dbc8ed8829e74e0b0
    repository_closure_base_commit: a1672d95653139964debdd5c1baf00173722c071
    origin_main_authoritative_match_at_launch: true
    git_worktree_clean_at_launch: true
    git_worktree_clean_at_exit: true
    provenance_compromised: false
    raw_artifact_package_kind: experiment-raw-complete
    raw_artifact_is_drpo_update_input: false
  result_summary:
    e6_a:
      positive_only_reward_mean: 0.851448580622673
      local_alpha_0_25_reward_mean: 0.8729863047599793
      local_alpha_0_25_minus_positive_only_reward_mean: 0.021537724137306213
      local_alpha_0_25_minus_positive_only_reward_ci95:
      - 0.02057325214147568
      - 0.02244885817170143
      local_alpha_0_25_reward_wins: 20
      local_alpha_0_50_reward_mean: 0.8674359440803527
      local_alpha_0_50_minus_positive_only_reward_mean: 0.01598736345767975
      local_alpha_0_50_minus_positive_only_reward_ci95:
      - 0.013682029843330383
      - 0.01812527000904083
      local_alpha_0_50_reward_wins: 20
      local_alpha_0_75_reward_mean: 0.8204200863838196
      local_alpha_0_75_minus_positive_only_reward_mean: -0.031028494238853455
      local_alpha_0_75_minus_positive_only_reward_ci95:
      - -0.03421750903129577
      - -0.02788556635379792
      local_alpha_0_75_reward_losses: 20
    e6_b:
      positive_only_reward_mean: 0.8643916040658951
      far_zero_reward_mean: 0.8853177160024643
      far_zero_support_boundary_events: 0
      far_zero_terminal_plateaus: 5
      far_zero_persistent_drift_or_inconclusive: 15
      uncontrolled_support_boundary_events: 20
      near_zero_support_boundary_events: 20
      far_cap_support_boundary_events: 20
      budget_matched_global_support_boundary_events: 20
      far_cap_minus_uncontrolled_reward_mean: 0.00013844966888429955
      far_cap_minus_uncontrolled_reward_ci95:
      - -0.00015761151909824173
      - 0.0004596554487943544
      budget_global_minus_far_cap_reward_mean: 7.615983486175537e-05
      budget_global_minus_far_cap_reward_ci95:
      - -0.0003324817121028858
      - 0.00045173332095150014
    e6_c:
      all_registered_methods_aligned_reward_wins_over_shuffled: 20
      positive_only_aligned_minus_shuffled_reward_mean: 0.3362449496984482
      far_zero_aligned_minus_shuffled_reward_mean: 0.3541200339794159
      uncontrolled_aligned_minus_shuffled_reward_mean: 0.37251958847045896
      far_cap_aligned_minus_shuffled_reward_mean: 0.3726573079824448
  paper_use:
    suitable_for_positive_only_ceiling_claim: true
    suitable_for_controlled_local_negative_benefit_claim: true
    suitable_for_excessive_pressure_reversal_claim: true
    suitable_for_semantic_alignment_exclusion_claim: true
    suitable_for_universal_method_ranking: false
    allowed:
    - same_distribution_held_out_context_generalization
    - unseen_action_semantic_extrapolation_in_aligned_D_U1
    - separate_task_support_and_numerical_event_reporting
    - high_reward_can_coexist_with_support_boundary
    prohibited_claims:
    - OOD_generalization
    - Transformer_external_validity
    - cross_task_method_superiority
    - far_cap_or_global_alpha_universal_winner
    - far_field_pressure_as_the_sole_cause_of_all_failures
  parent_claim_closure:
    parent_id: E6
    state: closed
    decision_date: '2026-06-28'
    approval_record: user_approved_close_after_commit_e70f0d84256cdeb6ebbf80b0495a043582787bf6
    basis_experiments:
    - D-U1-E6-SEMANTIC-LONGRUN-01
    - D-U1-E6-CONDITIONAL-GAP-01
    - D-U1-E6-SEMANTIC-GAP-LONGRUN-01
    closed_claims:
    - positive_only_has_a_shared_semantic_imitation_ceiling
    - moderate_controlled_negative_signal_can_improve_same_distribution_held_out_context_and_unseen_action_performance
    - excessive_or_unsuppressed_negative_pressure_can_reverse_or_progressively_degrade_task_performance
    - semantic_alignment_is_required_for_the_observed_unseen_action_transfer
    - task_performance_support_boundary_and_nan_inf_events_must_be_reported_separately
    preserved_limitations:
    - semantic_gap_and_conditional_gap_children_remain_finite_step_validated
    - no_steady_state_ranking_for_nonplateau_gap_runs
    - no_universal_alpha_optimum
    - no_state_distribution_OOD_generalization_claim
    - no_Gaussian_quadratic_law_claim_for_categorical_policy
    - no_cross_task_method_superiority_claim
    optional_non_gating_follow_up: D-U1-E6-TAPER-01
    optional_follow_up_status: not_run_not_implemented_blocked_until_separately_frozen
    required_for_parent_closure: false
    required_before_EXT_H_E7_Q2: false
  next_gate:
    experiment_id: EXT-H-E7-Q2
    state: ready
    automatic_activation_forbidden: false
    remaining_requirements: []
    optional_non_gating_follow_up: D-U1-E6-TAPER-01
- id: D-U1-E6-SEMANTIC-GAP-LONGRUN-01
  environment: D-U1
  name: minimum_change_shared_semantic_conditional_gap_longrun
  status: finite_step_validated
  scientific_status: finite_step_validated
  parent_experiment: E6
  predecessor: D-U1-E6-SEMANTIC-LONGRUN-01
  related_stress_diagnostic: D-U1-E6-CONDITIONAL-GAP-01
  registration_base_commit: 1fa7f04d4830e4d562ab147dbb11dfa8cecc9b5d
  claim: In the original 64-action shared-semantic E6 environment, changing only the logged state-action coverage by withholding
    the reward-optimal semantic neighbourhood on half of same-distribution contexts, test whether overall reward exhibits
    a Positive-only ceiling, an intermediate-alpha benefit, and progressively larger degradation when the original negative
    gradient is left unsuppressed at alpha=1 over a 32000-step horizon.
  role: formal_minimum_change_controlled_categorical_successor
  execution_class: formal
  implementation_state: implemented
  registry_scope: canonical_formal_confirmation_after_sandbox_qualification
  execution_gate:
    state: blocked
    blocked_by:
    - completed_formal_execution_no_rerun_without_new_registration
    approval_record: user_authorized_formal_package_generation_after_sandbox_review_2026-06-27
    blocking_reason: The frozen 100-run formal execution is complete, terminal-audited, packaged, and delivered. Re-running
      seeds 150--169 requires a separately registered protocol.
  formal_execution:
    channel_ref: hardened-v1
    activation_state: blocked
    entrypoint_status: implemented
    entrypoint: src/drpo/du1_e6_semantic_gap_longrun.py
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
  code_entrypoint: src/drpo/du1_e6_semantic_gap_longrun.py
  shared_scientific_entrypoint: src/drpo/du1_e6_semantic.py
  protocol_validator: src/drpo/du1_e6_semantic_gap.py
  one_click_entrypoint: scripts/run_du1_e6_semantic_gap_longrun.py
  config_path: configs/du1_e6_semantic_gap_longrun.yaml
  documentation_path: src/drpo/README_DU1_E6_SEMANTIC_GAP.md
  formal_launch_template: python3 scripts/run_experiment_guard_hardened.py --experiment-id D-U1-E6-SEMANTIC-GAP-LONGRUN-01
    --repo-root . --output-root experiments/results/D-U1-E6-SEMANTIC-GAP-LONGRUN-01/run_001 --artifact-output artifacts/D-U1-E6-SEMANTIC-GAP-LONGRUN-01_RAW_COMPLETE.zip
    --run-class formal --expected-commit "$(git rev-parse HEAD)" --require-origin-main-match --required-output RUN_COMPLETE.json
    --required-output scientific_run_manifest.json --required-output terminal_audit.json --required-output aggregate_summary.json
    --required-output horizon_summary.json --required-output horizon_summary.csv --required-output per_run_summary.csv --required-output
    formal_protocol_freeze.json --source-file scripts/run_du1_e6_semantic_gap_longrun.py --source-file src/drpo/du1_e6_semantic_gap_longrun.py
    --source-file src/drpo/du1_e6_semantic_gap.py --source-file src/drpo/du1_e6_semantic.py --source-file configs/du1_e6_semantic_gap_longrun.yaml
    --progress-glob checkpoints/*/CHECKPOINT_COMPLETE.json -- python3 src/drpo/du1_e6_semantic_gap_longrun.py --config configs/du1_e6_semantic_gap_longrun.yaml
    --output-root experiments/results/D-U1-E6-SEMANTIC-GAP-LONGRUN-01/run_001 --device cpu
  formal_parameter_freeze: true
  freeze_approval:
    approved_by_user: true
    approval_date: '2026-06-27'
    source: sandbox_review_and_user_request_for_formal_package
    automatic_retuning_allowed: false
  development_promotion_record:
    evidence_class: sandbox_and_candidate_only_not_formal
    repository_registration_during_exploration: false
    development_seeds:
    - 900
    - 901
    - 902
    - 903
    - 904
    - 905
    - 906
    - 907
    - 908
    - 909
    primary_metric_used_for_selection: overall_expected_semantic_reward
    inherited_alpha_anchors:
    - 0.0
    - 0.25
    - 0.5
    - 0.75
    added_unpenalized_anchor: 1.0
    alpha_greater_than_one_excluded_from_formal_domain: true
    qualitative_findings:
    - minimum_change_64_action_gap_reproduced_positive_only_ceiling_and_nonmonotonic_alpha_curve
    - longer_horizon_diagnostic_showed_alpha_1_degradation_growing_with_training
    - sandbox_results_are_not_formal_and_are_forbidden_from_formal_aggregation
    exact_steady_state_optimum_known: false
  development_seeds_forbidden_in_formal_aggregation:
  - 900
  - 901
  - 902
  - 903
  - 904
  - 905
  - 906
  - 907
  - 908
  - 909
  held_out_seeds:
  - 150
  - 151
  - 152
  - 153
  - 154
  - 155
  - 156
  - 157
  - 158
  - 159
  - 160
  - 161
  - 162
  - 163
  - 164
  - 165
  - 166
  - 167
  - 168
  - 169
  protocol:
    fixed_advantage: true
    critic_or_value_network: false
    importance_sampling: false
    state_dim: 6
    semantic_dim: 4
    action_count: 64
    train_states: 2048
    test_states: 2048
    state_distribution: standard_normal
    train_test_relation: independent_same_distribution
    terminology: held_out_context_generalization
    positive_actions_per_state: 4
    local_negative_actions_per_state: 1
    far_negative_actions_per_state: 4
    hidden_optimum_excluded_from_positive_demonstrations: true
    conditional_coverage_mode: structured_semantic_neighbourhood_gap
    gap_state_fraction: 0.5
    withheld_action_fraction_per_gap_state: 0.25
    withheld_actions_per_gap_state: 16
    withheld_from_logged_roles:
    - positive
    - local_negative
    - far_negative
    evaluation_oracle_remains_complete: true
    require_global_action_coverage: true
    reward_geometry_inherited_from_predecessor: true
    policy_and_optimizer_inherited_from_predecessor: true
    optimizer: Adam
    learning_rate: 0.001
    batch_size: 128
    fixed_concentration: 8.0
    alpha_grid:
    - 0.0
    - 0.25
    - 0.5
    - 0.75
    - 1.0
    alpha_semantics: negative_gradient_retention_weight
    alpha_zero_semantics: positive_only
    alpha_one_semantics: original_negative_gradient_without_suppression
    alpha_greater_than_one_allowed: false
    far_pressure_lambda: 0.0
    maximum_steps: 32000
    evaluation_interval_steps: 200
    registered_horizon_checkpoints:
    - 4000
    - 8000
    - 16000
    - 24000
    - 32000
    total_method_seed_runs: 100
  primary_metrics:
  - overall_expected_semantic_reward
  - paired_reward_difference_vs_positive_only
  - reward_trajectory_across_registered_horizons
  diagnostic_metrics:
  - hidden_optimal_probability
  - positive_support_probability
  - normalized_semantic_extrapolation
  - entropy_and_effective_support
  - raw_positive_local_far_gradient_norms
  - Adam_parameter_update_norm
  events:
    task_collapse_ratio_to_paired_positive_only: 0.2
    effective_support_boundary: 1.5
    concentration_warning: 80.0
  checkpointing:
    seed_block_size: 5
    seed_blocks:
    - - 150
      - 151
      - 152
      - 153
      - 154
    - - 155
      - 156
      - 157
      - 158
      - 159
    - - 160
      - 161
      - 162
      - 163
      - 164
    - - 165
      - 166
      - 167
      - 168
      - 169
    persistence: persistent_local
    write_compact_manifest_after_each_block: true
  terminal_audit:
    mode: formal_two_x_windows
    development_reference_horizon_steps: 16000
    formal_horizon_steps: 32000
    formal_extension_factor: 2.0
    window_1_steps:
    - 16000
    - 24000
    window_2_steps:
    - 24000
    - 32000
    stable_method_ranking_requires_all_registered_runs_plateau: true
    scientific_failure_outcomes_are_results: true
  reporting_separation:
  - task_performance_collapse
  - support_or_temperature_boundary
  - nan_inf_numerical_failure
  required_outputs:
  - RUN_COMPLETE.json
  - scientific_run_manifest.json
  - environment_audits.json
  - terminal_audit.json
  - aggregate_summary.json
  - horizon_summary.json
  - horizon_summary.csv
  - per_run_summary.csv
  - formal_protocol_freeze.json
  non_claims:
  - state_distribution_OOD_generalization
  - alpha_greater_than_one_method_claim
  - steady_state_method_ranking_without_terminal_plateau
  - Transformer_external_validity
  - cross_task_method_superiority
  evidence:
    implementation_tests_passed: true
    formal_run_started: true
    raw_complete: true
    terminal_audited: true
    all_terminal_audits_accepted: true
    formal_two_x_extension_performed: true
    expected_runs: 100
    actual_runs: 100
    terminal_plateau_runs: 45
    persistent_drift_or_inconclusive_runs: 55
    task_performance_collapse_events: 0
    support_or_temperature_boundary_events: 0
    nan_inf_numerical_events: 0
    package_created: true
    raw_complete_package_filename: DRPO_DU1_E6_SEMANTIC_GAP_0907C3C_RAW_COMPLETE.zip
    raw_complete_package_sha256: 65630159ef85c665a3a0ac0eba5cbf751ecb77a929f267423f7a6d9a8e5c4fbf
    final_repository_closure_package_filename: DRPO_DU1_E6_SEMANTIC_GAP_CLOSURE_FA225_UPDATE.zip
    final_repository_closure_package_sha256: null
    final_repository_closure_package_sha256_status: not_recorded_in_repository_evidence
    delivered_to_user: true
    repository_applied: false
    applied_commit: null
    compact_result_path: outputs/du1_e6_semantic_gap_longrun
    scientific_status: finite_step_validated
  provenance:
    exact_base_commit_source: complete_user_uploaded_git_bundle
    live_github_dns_available_in_generation_environment: false
    authoritative_remote_tip_recheck_required_by_formal_wrapper: true
    run_commit: 0907c3c0e76fc836c2bf2b752abf554c17f79f22
    repository_closure_base_commit: fa225510e3e3e4616f36d8f586611aa6af79bf6e
    origin_main_authoritative_match_at_launch: true
    git_worktree_clean_at_launch: true
    git_worktree_clean_at_exit: true
    provenance_compromised: false
    raw_artifact_package_kind: experiment-raw-complete
    raw_artifact_is_drpo_update_input: false
    diagnostic_rejection_phase: package_extract
    diagnostic_rejection_expected: true
  execution:
    state: delivered
    run_id: du1_e6_semantic_gap_longrun_0907c3c_run001
    start_utc: '2026-06-27T14:24:17.868186+00:00'
    end_utc: '2026-06-27T14:57:17.858160+00:00'
    elapsed_seconds: 1979.99
    process_exit_code: 0
    runtime: cpu
  result_summary:
    terminal:
      all_formal_terminal_plateau: false
      terminal_plateau_runs: 45
      persistent_drift_or_inconclusive_runs: 55
      formal_method_ranking_allowed: false
    final_32000_step:
      positive_only_reward_mean: 0.741308581829071
      alpha_0_25_reward_mean: 0.7662686556577682
      alpha_0_25_minus_positive_only_mean: 0.024960073828697204
      alpha_0_25_minus_positive_only_ci95:
      - 0.02358505330979824
      - 0.026427947282791138
      alpha_0_25_wins: 20
      alpha_0_50_reward_mean: 0.765975046157837
      alpha_0_50_minus_positive_only_mean: 0.024666464328765868
      alpha_0_50_minus_positive_only_ci95:
      - 0.022564217671751978
      - 0.026787295341491702
      alpha_0_50_wins: 20
      alpha_0_75_reward_mean: 0.7393304020166397
      alpha_0_75_minus_positive_only_mean: -0.0019781798124313354
      alpha_0_75_minus_positive_only_ci95:
      - -0.005438663214445113
      - 0.0013230986893177034
      alpha_0_75_wins: 9
      alpha_0_75_losses: 11
      alpha_1_0_reward_mean: 0.6802235990762711
      alpha_1_0_minus_positive_only_mean: -0.061084982752799985
      alpha_1_0_minus_positive_only_ci95:
      - -0.06415405839681626
      - -0.058077503591775895
      alpha_1_0_losses: 20
    alpha_1_progressive_gap_vs_positive_only:
      step_4000: 0.0039426833391189575
      step_8000: -0.01374114751815796
      step_16000: -0.03916723430156708
      step_24000: -0.05322670340538025
      step_32000: -0.061084982752799985
    terminal_class_by_alpha:
      alpha_0_0:
        plateau: 20
        drift_or_inconclusive: 0
      alpha_0_25:
        plateau: 20
        drift_or_inconclusive: 0
      alpha_0_50:
        plateau: 5
        drift_or_inconclusive: 15
      alpha_0_75:
        plateau: 0
        drift_or_inconclusive: 20
      alpha_1_0:
        plateau: 0
        drift_or_inconclusive: 20
  paper_use:
    suitable_for_finite_horizon_positive_only_ceiling_claim: true
    suitable_for_intermediate_alpha_benefit_claim: true
    suitable_for_progressive_unsuppressed_negative_degradation_claim: true
    suitable_for_steady_state_method_ranking: false
    allowed:
    - same_distribution_held_out_context_generalization
    - structured_state_action_support_gap
    - finite_step_reward_trajectory_comparison
    - separate_task_support_and_numerical_event_reporting
    prohibited_claims:
    - state_distribution_OOD_generalization
    - universal_alpha_optimum
    - steady_state_method_ranking_across_all_alpha
    - Gaussian_quadratic_far_field_law_for_categorical_policy
    - cross_task_method_superiority
  next_gate:
    experiment_id: D-U1-E6-TAPER-01
    predecessor_delivery_satisfied: true
    state: review_required_not_runnable
    automatic_retuning_forbidden: true
    D_U1_E6_TAPER_01_remains_blocked_until_delivery: false
    automatic_activation_forbidden: true
    remaining_requirements:
    - freeze_semantic_remoteness_coordinate
    - freeze_paired_method_protocol
    - freeze_new_untouched_held_out_seeds
    - implement_separate_formal_runner
- id: D-U1-E6-CONDITIONAL-GAP-01
  environment: D-U1
  name: structured_conditional_support_gap_categorical_longrun
  status: finite_step_validated
  scientific_status: finite_step_validated
  parent_experiment: E6
  predecessor: D-U1-E6-SEMANTIC-LONGRUN-01
  registration_base_commit: ff2afe443167154eae5de7871cda83f3aba9a89e
  claim: Under independent same-distribution held-out contexts, test whether a large structured state-region by action-group
    support gap permits wrong conditional generalization and task-performance collapse, whether controlled local repulsion
    improves the withheld optimal group, and whether excessive local or far negative pressure reverses that benefit.
  role: formal_controlled_categorical_conditional_gap_experiment
  execution_class: formal
  implementation_state: implemented
  execution_gate:
    state: blocked
    blocked_by:
    - completed_formal_execution_no_rerun_without_new_registration
    approval_record: user_approved_2026-06-27_large_structured_gap
    blocking_reason: The frozen formal run is complete, terminal-audited, packaged, and delivered. Re-running held-out seeds
      130--149 requires a separately registered protocol.
  formal_execution:
    channel_ref: hardened-v1
    activation_state: blocked
    entrypoint_status: implemented
    entrypoint: src/drpo/du1_e6_conditional_gap_longrun.py
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
  code_entrypoint: src/drpo/du1_e6_conditional_gap_longrun.py
  shared_scientific_entrypoint: src/drpo/du1_e6_conditional_gap.py
  one_click_entrypoint: scripts/run_du1_e6_conditional_gap_longrun.py
  config_path: configs/du1_e6_conditional_gap_longrun.yaml
  documentation_path: src/drpo/README_DU1_E6_CONDITIONAL_GAP.md
  formal_launch_template: python3 scripts/run_experiment_guard_hardened.py --experiment-id D-U1-E6-CONDITIONAL-GAP-01 --repo-root
    . --output-root experiments/results/D-U1-E6-CONDITIONAL-GAP-01/run_001 --artifact-output artifacts/D-U1-E6-CONDITIONAL-GAP-01_RAW_COMPLETE.zip
    --run-class formal --expected-commit "$(git rev-parse HEAD)" --require-origin-main-match --required-output RUN_COMPLETE.json
    --required-output scientific_run_manifest.json --required-output terminal_audit.json --required-output aggregate_summary.json
    --required-output per_run_summary.csv --required-output formal_protocol_freeze.json --source-file src/drpo/du1_e6_conditional_gap_longrun.py
    --source-file src/drpo/du1_e6_conditional_gap.py --source-file configs/du1_e6_conditional_gap_longrun.yaml --progress-glob
    checkpoints/*/CHECKPOINT_COMPLETE.json -- python3 src/drpo/du1_e6_conditional_gap_longrun.py --config configs/du1_e6_conditional_gap_longrun.yaml
    --output-root experiments/results/D-U1-E6-CONDITIONAL-GAP-01/run_001 --device cpu
  approval:
    approved_by_user: true
    approval_date: '2026-06-27'
    scope: large_structured_conditional_support_gap_rerun
    automatic_retuning_allowed: false
  protocol:
    fixed_advantage: true
    critic_or_value_network: false
    importance_sampling: false
    state_dim: 6
    semantic_dim: 4
    action_groups: 8
    actions_per_group: 32
    action_count: 256
    train_states: 4096
    test_states: 4096
    state_distribution: paired_standard_normal_marginal
    train_test_relation: independent_same_distribution
    terminology: same_distribution_structured_state_action_support_gap
    gap_state_fraction: 0.5
    observed_action_groups_per_state: 3
    conditional_block_gap_fraction: 0.625
    optimal_action_group_absent_on_gap_states: true
    action_ids_randomly_permuted: true
    positive_actions_per_state: 4
    local_negative_actions_per_state: 1
    far_negative_actions_per_state: 4
    hidden_optimum_excluded_from_positive_demonstrations: true
    correct_group_reward: 1.0
    proxy_group_reward: 0.65
    other_group_reward: 0.0
    fixed_concentration: 8.0
    optimizer: Adam
    learning_rate: 0.001
    maximum_steps: 8000
    evaluation_interval_steps: 50
    methods:
    - positive_only
    - local_only
    - uncontrolled
    - near_zero
    - far_cap
    - budget_matched_global
    local_alpha_beneficial_candidate: 0.5
    local_alpha_excessive_candidate: 1.5
    far_pressure_lambda: 4.0
    far_cap_ratio_to_weighted_local_gradient: 1.0
  held_out_seeds:
  - 130
  - 131
  - 132
  - 133
  - 134
  - 135
  - 136
  - 137
  - 138
  - 139
  - 140
  - 141
  - 142
  - 143
  - 144
  - 145
  - 146
  - 147
  - 148
  - 149
  development_seeds_forbidden_in_formal_aggregation:
  - 0
  - 1
  events:
    task_collapse_rule: normalized_margin_between_random_and_paired_positive_only_le_0.2
    below_random_is_collapse: true
    effective_support_boundary: 1.5
    concentration_warning: 80.0
  terminal_audit:
    mode: formal_extension_windows
    development_reference_horizon_steps: 1000
    formal_horizon_steps: 8000
    formal_extension_factor: 8.0
    window_1_steps:
    - 4000
    - 6000
    window_2_steps:
    - 6000
    - 8000
    raw_total_gradient_median_ratio_max: 1.25
    adam_update_median_ratio_max: 1.25
    scientific_failure_outcomes_are_results: true
  checkpointing:
    seed_block_size: 5
    seed_blocks:
    - - 130
      - 131
      - 132
      - 133
      - 134
    - - 135
      - 136
      - 137
      - 138
      - 139
    - - 140
      - 141
      - 142
      - 143
      - 144
    - - 145
      - 146
      - 147
      - 148
      - 149
    persistence: persistent_local
  reporting_separation:
  - task_performance_collapse
  - support_or_temperature_boundary
  - nan_inf_numerical_failure
  required_outputs:
  - RUN_COMPLETE.json
  - scientific_run_manifest.json
  - terminal_audit.json
  - aggregate_summary.json
  - per_run_summary.csv
  - formal_protocol_freeze.json
  non_claims:
  - state_distribution_OOD_generalization
  - Transformer_external_validity
  - cross_task_method_superiority
  - universal_far_cap_or_global_alpha_winner
  execution:
    state: delivered
    run_id: du1_e6_conditional_gap_7a70278_run001
    start_utc: '2026-06-27T08:07:04.244576+00:00'
    end_utc: '2026-06-27T08:23:30.677071+00:00'
    elapsed_seconds: 986.433
    process_exit_code: 0
    runtime: cpu
  evidence:
    development_pilot_completed: true
    environment_invariants_passed: true
    implementation_tests_passed: true
    formal_run_started: true
    raw_complete: true
    terminal_audited: true
    terminal_audit_all_checks_passed: true
    formal_extension_windows_performed: true
    expected_runs: 200
    actual_runs: 200
    terminal_plateau_runs: 49
    persistent_drift_or_inconclusive_runs: 151
    task_performance_collapse_events: 77
    support_or_temperature_boundary_events: 0
    nan_inf_numerical_events: 0
    package_created: true
    raw_complete_package_filename: D-U1-E6-CONDITIONAL-GAP-01_RAW_COMPLETE.zip
    raw_complete_package_sha256: 8c64f197e90e945f3a6bf8326c63abd6c4b3118e1c6c8bd614c73af6d1e5be93
    delivered_to_user: true
    repository_applied: false
    applied_commit: null
    compact_result_path: outputs/du1_e6_conditional_gap_longrun
    scientific_status: finite_step_validated
  provenance:
    run_commit: 7a70278f3d6061379c81f33e82d93ead86484908
    repository_closure_base_commit: 1fa7f04d4830e4d562ab147dbb11dfa8cecc9b5d
    source_checkout: complete_user_uploaded_git_bundle
    live_github_dns_available: false
    origin_main_authoritative_match_at_launch: true
    git_worktree_clean_at_launch: true
    git_worktree_clean_at_exit: true
    provenance_compromised: false
    raw_artifact_package_kind: experiment-raw-complete
    raw_artifact_is_drpo_update_input: false
  result_summary:
    positive_only_gap_effect:
      covered_gap_reward_mean: 0.7312927275896073
      structured_gap_reward_mean: 0.5801327347755432
      structured_minus_covered_mean: -0.15115999281406403
      bootstrap_ci95:
      - -0.1558610486239195
      - -0.14663811750710012
      structured_lower_seeds: 20
    controlled_local_alpha_0_5:
      positive_only_gap_reward_mean: 0.5801327347755432
      local_gap_reward_mean: 0.7639744341373443
      local_minus_positive_mean: 0.18384169936180114
      bootstrap_ci95:
      - 0.18014306128025054
      - 0.18749583370983602
      gap_reward_wins: 20
      overall_reward_difference: -0.05098388195037842
      overall_reward_losses: 20
    excessive_local_alpha_1_5:
      structured_gap_reward_mean: 0.07551877722144126
      structured_task_collapse_events: 20
      covered_task_collapse_events: 20
      structured_trap_group_probability_mean: 0.7325085878372193
    far_pressure:
      local_only_gap_reward_mean: 0.7639744341373443
      uncontrolled_gap_reward_mean: 0.17814922630786895
      uncontrolled_task_collapse_events: 20
      near_zero_task_collapse_events: 16
      far_cap_task_collapse_events: 1
      budget_matched_global_task_collapse_events: 0
      far_cap_minus_uncontrolled_gap_reward_mean: 0.1223801597952843
      global_minus_uncontrolled_gap_reward_mean: 0.13157710283994678
      global_minus_far_cap_gap_reward_mean: 0.009196943044662481
      global_minus_far_cap_ci95:
      - 8.500430732964994e-05
      - 0.01788240306079387
      global_wins_over_far_cap: 14
    terminal_boundary:
      scientific_status: finite_step_validated
      terminal_plateau_runs: 49
      persistent_drift_or_inconclusive_runs: 151
      stable_method_ranking_allowed: false

## Source 4: experiments/registry.yaml: development_experiment_registrations[D-U1-E6-CONDITIONAL-GAP-DEV-01, D-U1-E6-SEMANTIC-PILOT-01, D-U1-E6-SEMANTIC-FOCUSED-DEV-01, D-U1-E6-TAPER-01]

collection: development_experiment_registrations
entries:
- id: D-U1-E6-CONDITIONAL-GAP-DEV-01
  environment: D-U1
  name: structured_conditional_support_gap_development_pilot
  status: pilot
  parent_experiment: E6
  predecessor: D-U1-E6-SEMANTIC-LONGRUN-01
  registration_base_commit: ff2afe443167154eae5de7871cda83f3aba9a89e
  claim: On development seeds, verify a balanced large structured conditional state-action support gap and identify fixed
    local/far pressure candidates for a separately frozen formal long-run without consuming formal seeds.
  role: controlled_categorical_conditional_gap_development
  execution_class: pilot
  registry_scope: development_preregistration_not_formal_evidence
  code_entrypoint: src/drpo/du1_e6_conditional_gap.py
  config_path: configs/du1_e6_conditional_gap_dev.yaml
  documentation_path: src/drpo/README_DU1_E6_CONDITIONAL_GAP.md
  protocol:
    development_seeds:
    - 0
    - 1
    maximum_steps: 1000
    train_states: 4096
    test_states: 4096
    action_count: 256
    gap_state_fraction: 0.5
    conditional_block_gap_fraction: 0.625
    state_distribution_shift: false
    formal_seeds_consumed: false
  result:
    scientific_status: pilot
    actual_runs: 20
    environment_invariants_passed: true
    pilot_integrity_passed: true
    task_performance_collapse_count: 4
    support_or_temperature_boundary_count: 0
    nan_inf_numerical_failure_count: 0
    structured_gap_positive_only_gap_reward_mean: 0.515236884355545
    structured_gap_local_alpha_0_5_gap_reward_mean: 0.6362665593624115
    structured_gap_local_alpha_1_5_gap_reward_mean: 0.18586624413728714
    structured_gap_uncontrolled_far_4_gap_reward_mean: 0.2991461008787155
    structured_gap_far_cap_gap_reward_mean: 0.4508051127195358
    structured_gap_budget_matched_global_gap_reward_mean: 0.49083396792411804
    random_policy_gap_reward_mean: 0.19078125059604645
    compact_output_path: outputs/du1_e6_conditional_gap_dev
  formal_freeze_recommendation:
    automatic_freeze_allowed: false
    user_approved: true
    approval_date: '2026-06-27'
    activated_experiment: D-U1-E6-CONDITIONAL-GAP-01
    held_out_seeds:
    - 130
    - 131
    - 132
    - 133
    - 134
    - 135
    - 136
    - 137
    - 138
    - 139
    - 140
    - 141
    - 142
    - 143
    - 144
    - 145
    - 146
    - 147
    - 148
    - 149
    formal_horizon_steps: 8000
    formal_extension_factor_over_pilot: 8.0
  evidence:
    static_checks: true
    invariant_test: true
    smoke_test: true
    development_pilot: true
    terminal_audited: true
    formal_result: false
- id: D-U1-E6-SEMANTIC-PILOT-01
  environment: D-U1
  name: shared_semantic_categorical_extrapolation_development_pilot
  status: pilot
  parent_experiment: E6
  registration_base_commit: c5c9ebf67ed7aa08d0313e604f836907890f0964
  claim: Test whether controlled local fixed-negative gradients in a shared semantic categorical policy can move probability
    mass beyond positive demonstrations toward a hidden optimal action on independently sampled held-out contexts, while uncontrolled
    far-negative pressure can cause task-performance or support/temperature failure.
  role: controlled_shared_semantic_categorical_development
  execution_class: pilot
  registry_scope: development_preregistration_not_in_canonical_experiments
  code_entrypoint: src/drpo/du1_e6_semantic.py
  config_path: configs/du1_e6_semantic_pilot.yaml
  documentation_path: src/drpo/README_DU1_E6_SEMANTIC.md
  command:
  - python
  - src/drpo/du1_e6_semantic.py
  - --config
  - configs/du1_e6_semantic_pilot.yaml
  - --stage
  - pilot
  - --output-root
  - experiments/results/D-U1-E6-SEMANTIC-PILOT-01/run_001
  - --device
  - auto
  parallelism:
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
    state_dim: 6
    semantic_dim: 4
    action_count: 64
    development_train_states: 2048
    development_test_states: 2048
    state_distribution: standard_normal
    train_test_relation: independent_same_distribution
    terminology: held_out_context_generalization
    positive_actions_per_state: 4
    local_negative_actions_per_state: 1
    far_negative_actions_per_state: 4
    hidden_optimum_excluded_from_positive_demonstrations: true
    target_offset: 0.45
    positive_advantage: 1.0
    negative_advantage: -1.0
    negative_advantages_exactly_equal: true
    random_action_id_permutation: true
    development_seeds:
    - 0
    - 1
    - 2
    - 3
    - 4
    held_out_formal_seeds: []
    optimizer: Adam
    development_learning_rate: 0.001
    development_maximum_steps: 2000
    evaluation_interval_steps: 50
    fixed_concentration: 8.0
    initial_learnable_concentration: 8.0
    learnable_concentration_upper_clamp: false
    protocol_a_local_alpha_grid:
    - 0.0
    - 0.1
    - 0.25
    - 0.5
    - 0.75
    - 1.0
    protocol_b_methods:
    - positive_only
    - local_only
    - uncontrolled
    - near_zero
    - far_zero
    - far_cap
    - budget_matched_global
    protocol_c_policy_embedding_modes:
    - aligned
    - shuffled
  primary_metrics:
  - hidden_optimal_probability
  - positive_support_probability
  - expected_semantic_reward
  - normalized_semantic_extrapolation
  - entropy
  - effective_support
  - concentration
  - raw_positive_local_far_gradient_norms
  - raw_controlled_negative_gradient_norm
  - raw_total_gradient_norm
  - adam_parameter_update_norm
  reporting_separation:
  - task_performance_collapse
  - support_or_temperature_boundary
  - nan_inf_numerical_failure
  controls:
    paired_network_initialization: true
    paired_minibatch_index_stream: true
    policy_side_semantic_shuffle_only: true
    reward_semantics_unchanged_under_shuffle: true
    raw_gradient_budget_matching_only: true
    adam_update_budget_matching_claimed: false
  terminal_audit:
    pilot_two_trailing_windows: true
    formal_two_x_extension_performed: false
    formal_acceptance_allowed: false
  formal_parameter_freeze:
    frozen: false
    required_before_longrun:
    - local_alpha_or_grid
    - concentration_settings
    - learning_rate_and_optimizer
    - maximum_steps_and_evaluation_interval
    - event_thresholds
    - stopping_and_two_x_extension_criteria
    - untouched_held_out_seeds
    - formal_method_matrix
  pilot_result:
    expected_github_commit: e8b62dde518f593ff8325c7da94c41406311ca45
    execution_snapshot_commit: 653aa6f73b18fed17609e6096cb1c50de0a8cd66
    exact_git_object_checkout_available: false
    provenance_scope: verified_E6_source_snapshot_for_pilot_only
    completed_at_utc: '2026-06-26T13:02:30Z'
    device: cpu
    expected_runs: 105
    actual_runs: 105
    maximum_steps_per_run: 2000
    environment_invariants_passed: true
    nan_inf_numerical_failure_count: 0
    task_performance_collapse_count: 0
    support_or_temperature_boundary_count: 56
    formal_two_x_extension_performed: false
    protocol_a_development_observation:
      alpha_0_5_expected_semantic_reward_mean: 0.8882276892662049
      alpha_0_5_hidden_optimal_probability_mean: 0.21612542271614074
      alpha_0_5_hidden_probability_delta_vs_positive_only: 0.07203658819198608
      all_fixed_concentration_runs_provisional_plateau: false
      interpretation: development_peak_but_terminal_horizon_unresolved
    protocol_b_development_observation:
      positive_only_support_events: 0
      negative_pressure_aligned_support_events: 30
      negative_pressure_aligned_total_runs: 30
      interpretation: current_learnable_concentration_negative_pressure_not_freezable
    protocol_c_development_observation:
      semantic_alignment_control_supported: true
      formal_claim_allowed: false
    formal_gate_decision: remains_blocked_pending_user_review_and_focused_development_extension
    compact_output_path: outputs/du1_e6_semantic_pilot
  non_claims:
  - formal_method_ranking
  - long_run_validation
  - OOD_generalization
  - Transformer_external_validity
  - E5_replacement
  execution:
    state: delivered
    run_id: cpu-pilot-20260626T125354Z
    last_heartbeat_utc: '2026-06-26T13:02:30Z'
    process_exit_code: 0
  evidence:
    static_checks: true
    unit_tests: true
    smoke_test: true
    pilot_raw_complete: true
    terminal_audited: true
    package_created: true
    delivered_to_user: true
    pilot_integrity_passed: true
    raw_artifact_sha256: 6aee3262bcd6936f53d15889ab0c38d825b7ba8f110fa40ab40032f90d079bfb
    repository_applied: true
    applied_commit: 2e04f6dba6d4e87f61920bedb1c464656906bf2b
- id: D-U1-E6-SEMANTIC-FOCUSED-DEV-01
  environment: D-U1
  name: shared_semantic_categorical_focused_development_extension
  status: pilot
  parent_experiment: E6
  predecessor: D-U1-E6-SEMANTIC-PILOT-01
  registration_base_commit: 2e04f6dba6d4e87f61920bedb1c464656906bf2b
  claim: Resolve the fixed-concentration terminal-horizon and learnable-concentration support-boundary blockers using only
    the already registered alpha, far-pressure, concentration, and training-horizon variables.
  role: controlled_shared_semantic_categorical_blocker_resolution
  execution_class: pilot
  registry_scope: development_preregistration_not_in_canonical_experiments
  code_entrypoint: src/drpo/du1_e6_semantic.py
  config_path: configs/du1_e6_semantic_focused_dev.yaml
  phase2_config_path: configs/du1_e6_semantic_focused_dev_phase2.yaml
  command:
  - python
  - src/drpo/du1_e6_semantic.py
  - --config
  - configs/du1_e6_semantic_focused_dev.yaml
  - --stage
  - pilot
  - --output-root
  - experiments/results/D-U1-E6-SEMANTIC-FOCUSED-DEV-01/run_001
  - --device
  - auto
  protocol:
    development_seeds:
    - 0
    - 1
    - 2
    - 3
    - 4
    held_out_formal_seeds: []
    optimizer: Adam
    learning_rate: 0.001
    maximum_steps: 4000
    evaluation_interval_steps: 50
    fixed_concentration: 8.0
    initial_learnable_concentration: 8.0
    learnable_concentration_upper_clamp: false
    fixed_concentration_alpha_grid:
    - 0.0
    - 0.25
    - 0.5
    - 0.75
    learnable_local_alpha_grid:
    - 0.005
    - 0.01
    - 0.02
    - 0.05
    - 0.1
    - 0.2
    phase2_far_pressure_lambda_grid:
    - 0.01
    - 0.02
    - 0.05
    - 0.1
    - 0.2
    phase2_methods:
    - uncontrolled
    - far_cap
    - budget_matched_global
    - near_zero
    phase2_requires_preregistered_phase1_selection: true
  local_alpha_selection_rule:
    descending_candidate_order:
    - 0.2
    - 0.1
    - 0.05
    - 0.02
    - 0.01
    - 0.005
    support_boundary_count_required: 0
    numerical_failure_count_required: 0
    paired_reward_wins_required: 4
    paired_hidden_probability_wins_required: 4
    focused_terminal_plateau_seeds_required: 4
    no_candidate_action: stop_without_phase2
  terminal_audit:
    mode: focused_two_x_windows
    window_1_steps:
    - 2000
    - 3000
    window_2_steps:
    - 3000
    - 4000
    metric_window_mean_abs_tolerances:
      expected_semantic_reward: 0.01
      hidden_optimal_probability: 0.02
      normalized_semantic_extrapolation: 0.08
      entropy_mean: 0.08
    raw_total_gradient_median_ratio_max: 1.25
    adam_update_median_ratio_max: 1.25
    formal_acceptance_allowed: false
  reporting_separation:
  - task_performance_collapse
  - support_or_temperature_boundary
  - nan_inf_numerical_failure
  non_claims:
  - formal_method_ranking
  - long_run_validation
  - OOD_generalization
  - automatic_formal_parameter_freeze
  execution:
    state: delivered
    run_id: focused-dev-two-phase-20260626
    last_heartbeat_utc: '2026-06-26T14:58:00Z'
    process_exit_code: 0
  result:
    scientific_status: pilot
    base_github_commit: 2e04f6dba6d4e87f61920bedb1c464656906bf2b
    phase1_actual_runs: 55
    phase2_actual_runs: 110
    total_actual_runs: 165
    maximum_steps_per_run: 4000
    development_two_x_horizon_performed: true
    formal_held_out_seeds_consumed: false
    nan_inf_numerical_failure_count: 0
    task_performance_collapse_count: 0
    support_or_temperature_boundary_count: 78
    fixed_concentration_plateau_counts:
      alpha_0_0: 5
      alpha_0_25: 5
      alpha_0_5: 5
      alpha_0_75: 5
    selected_learnable_local_alpha: 0.1
    selected_local_alpha_support_events: 0
    selected_local_alpha_plateau_seeds: 5
    rejected_alpha_0_2_support_events: 3
    safe_far_lambda: 0.01
    support_transition_far_lambda: 0.02
    far_only_boundary_lambda: 0.05
    ratio_1_far_cap_rescued_transition: false
    budget_matched_global_rescued_transition: false
    compact_output_path: outputs/du1_e6_semantic_focused_dev
    raw_artifact_sha256: bee5b62e7715bda63ec166849f431ab5c4c90954720e672945e25e62b320e0d6
    formal_gate_decision: user_approved_and_frozen_for_D-U1-E6-SEMANTIC-LONGRUN-01
  formal_freeze_recommendation:
    automatic_freeze_allowed: false
    user_review_required: true
    user_approved: true
    approval_date: '2026-06-27'
    activated_experiment: D-U1-E6-SEMANTIC-LONGRUN-01
    held_out_seeds:
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
    optimizer: Adam
    learning_rate: 0.001
    maximum_steps: 8000
    evaluation_interval_steps: 50
    terminal_windows:
    - - 4000
      - 6000
    - - 6000
      - 8000
    fixed_concentration: 8.0
    fixed_alpha_grid:
    - 0.0
    - 0.25
    - 0.5
    - 0.75
    initial_learnable_concentration: 8.0
    learnable_concentration_upper_clamp: false
    learnable_local_alpha: 0.1
    learnable_far_pressure_stress_lambda: 0.05
    formal_method_matrix:
    - positive_only
    - far_zero
    - uncontrolled
    - near_zero
    - far_cap
    - budget_matched_global
    far_cap_ratio_to_weighted_local_gradient: 1.0
  evidence:
    static_checks: true
    unit_tests: true
    smoke_test: true
    phase1_raw_complete: true
    phase2_raw_complete: true
    terminal_audited: true
    package_created: true
    delivered_to_user: true
    repository_applied: true
    applied_commit: eb6a90d55127cead4d95bd0a85a78f32c47ff56a
- id: D-U1-E6-TAPER-01
  environment: D-U1
  name: categorical_semantic_taper_order_comparison
  status: not_run
  parent_experiment: E6-TAPER
  predecessor: D-U1-E6-SEMANTIC-LONGRUN-01
  additional_predecessor: D-U1-E6-CONDITIONAL-GAP-01
  additional_predecessors:
  - D-U1-E6-CONDITIONAL-GAP-01
  - D-U1-E6-SEMANTIC-GAP-LONGRUN-01
  registration_base_commit: f64452a7452274a183b03c87c39b847039230c00
  claim: On one frozen D-U1 semantic-remoteness coordinate and paired training stream, compare reciprocal-linear, reciprocal-quadratic,
    and exponential negative tapering for preservation of beneficial local negative signal, suppression of harmful far pressure,
    held-out unseen-action performance, and support stability. This experiment does not claim a Gaussian quadratic gradient
    law for categorical policies.
  role: controlled_cross_policy_taper_validation
  execution_class: formal
  implementation_state: not_implemented
  registry_scope: development_preregistration_not_in_canonical_experiments
  formal_execution:
    channel_ref: hardened-v1
    activation_state: blocked
    entrypoint_status: planned
    entrypoint: null
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
  predecessor_delivery_satisfied: true
  semantic_gap_successor_delivery_satisfied: true
  blocked_by:
  - frozen_semantic_remoteness_coordinate
  - frozen_paired_method_protocol
  - frozen_untouched_held_out_seeds
  - separately_implemented_formal_runner
  distance_rule:
    same_coordinate_across_methods: required
    gaussian_standardized_distance_reused: false
    exact_semantic_coordinate: pending_separate_E6_TAPER_protocol_freeze
  candidate_families:
  - reciprocal_linear
  - reciprocal_quadratic
  - exponential
  controls:
  - positive_only
  - uncontrolled_negative
  - global_alpha
  no_method_winner_assumed: true
  reporting_separation:
  - task_performance_collapse
  - support_or_temperature_boundary
  - nan_inf_numerical_failure
  evidence:
    raw_complete: false
    terminal_audited: false
    package_created: false
    delivered_to_user: false
    scientific_status: not_run
    semantic_gap_successor_raw_complete: true
    semantic_gap_successor_terminal_audited: true
    semantic_gap_successor_delivered_to_user: true
    semantic_gap_successor_scientific_status: finite_step_validated
