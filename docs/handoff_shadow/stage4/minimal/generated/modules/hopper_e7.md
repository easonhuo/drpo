# Hopper learned-critic external validation E7

> Generated Stage 4 minimal-context shadow module. Do not edit manually.
> `docs/handoff.md` and `experiments/registry.yaml` remain authoritative.

- Module ID: `hopper_e7`
- Responsibility: Cover learned-critic far-field mechanism validation and D4RL method-effect evidence while preserving the external-validity boundary.
- Source hash: `5ba68556fdc9f1a1f7ce65f57204fe02083d2834f39632509b9085750527a04f`

## Source 1: docs/handoff.md: # 15. Learned-Critic External Mechanism Validation on D4RL -> # Part V. Bandit 稳定外推子实验的收敛审计（完整保留）

# 15. Learned-Critic External Mechanism Validation on D4RL



这次实验已经补上此前 D4RL 分析最重要的缺口：advantage 不再由人工轨迹标签直接指定，而是由真实训练出的 value critic 产生；actor 使用 detached TD residual 进行重复的 signed off-policy 更新。

正式配置先在开发 seed 42 上冻结，随后使用未参与选择的 seeds 100--109。Critic 在 held-out episode 上的平均 R² 为 **0.428**，Pearson 相关为 **0.656**。它并不完美，反而说明结论能承受现实的 critic noise。

当前可形成的严谨结论是：

> 在 Hopper medium-replay 的自然数据中，匹配负 advantage 幅度后，far negative 仍具有更大的 policy score 与全参数梯度；重复 signed actor 更新会造成均值向 tanh 边界饱和并收缩策略支持。删除 near negatives 不能消除该失稳，而只删除 current far negatives 可以稳定救援。

这是一项**外部机制验证**，不是 D4RL normalized-return 方法表。

## 1. learned advantage 下的来源隔离

在 Positive-only actor 拟合过程中：

| 阶段 | |A| far/near | 标准化距离 far/near | Gaussian score far/near | 全参数梯度 far/near | 聚合梯度 far/near |
|---|---:|---:|---:|---:|---:|
| Step 0 | 1.000 | 3.659 | 1.908 | 2.210 | 3.174 |
| Step 600 | 1.001 | 7.363 | 3.629 | 2.107 | 2.615 |

因此，大梯度不是因为 far 样本具有更大的 negative advantage。Positive-only 拟合让固定坏动作相对当前策略进一步远场化，标准化距离约翻倍，score 放大随之增强。

## 2. 方差方向与修正理论一致

在所有正式 seeds 和所有记录 checkpoint 中，匹配 far negatives 对 `log sigma` 的 signed ascent direction 都为负，即：

\[
A<0,\quad \|z\|>1 \Longrightarrow \Delta\log\sigma<0.
\]

所以 far negative 的实际作用是**均值排斥 + 方差/支持收缩**，而不是旧稿中的 “mu 与 sigma 同时扩张”。Near negatives 在初期通常推动 sigma 扩张；随着策略移动，一部分 near negatives 跨过标准化距离边界后也转为收缩。

## 3. 10-seed 定点干预

| 方法 | 最终均值饱和率 | 正样本 NLL | 平均 sigma | far-negative surprisal |
|---|---:|---:|---:|---:|
| Positive-only | 0.006 | 1.975 | 0.501 | 6.589 |
| Signed baseline | 0.693 | 8.154 | 0.413 | 23.287 |
| Near-zero | 0.604 | 6.546 | 0.417 | 22.004 |
| Far-zero | 0.007 | 2.255 | 0.506 | 8.005 |
| Global scale | 0.041 | 2.571 | 0.460 | 10.949 |
| Far-cap | 0.545 | 5.956 | 0.422 | 18.970 |
| Exp taper | 0.460 | 5.039 | 0.439 | 16.534 |

核心对比均在 10/10 配对 seeds 上方向一致，Wilcoxon `p=0.001953`：

- Near-zero 仍保留严重失稳，说明删除近场不是关键救援。
- Far-zero 将饱和率从约 0.693 降至 0.007，正样本 NLL 恢复到接近 Positive-only。
- Global scaling 同样大幅救援，说明异常 repulsive magnitude 是直接中介；far-field 是其主要自然来源，但并不是唯一可行控制方式。
- 当前固定参数的 Far-cap 和 Exp 只有部分救援，说明 toy 环境超参不能直接迁移；这反而支持使用稳定边界自适应确定 taper 强度。

## 4. 相变而非 sign-only 法则

开发 seed 的 alpha 扫描显示：负梯度系数从 0.5 增至 1.0 时，均值饱和和正样本 NLL 出现明显恶化，之后继续走向边界。因此失稳取决于正负 signed field 的净平衡，而不是“只要 A<0 就必然联合发散”。这直接支持 v9 的 signed-field / moment-domain 理论，否定旧版 sign-only SPD 论证。

## 5. Advantage estimator 稳健性

额外 3 seeds 使用 `return-to-go - V(s)` 而非 TD residual，仍得到同一排序：

| 方法 | 均值饱和率 | 正样本 NLL | sigma |
|---|---:|---:|---:|
| positive_only | 0.007 | 2.180 | 0.530 |
| signed | 0.507 | 9.538 | 0.435 |
| near_zero | 0.146 | 5.721 | 0.434 |
| far_zero | 0.004 | 2.388 | 0.623 |
| global | 0.011 | 2.262 | 0.518 |

因此，主结论并非某一种 advantage estimator 的偶然产物。

## 6. 论文中的正确使用方式

可以写入机制与外部有效性章节：

1. learned critic 产生的 signed advantages 下，far-field gradient amplification 仍存在；
2. far negative 的方差方向是 support contraction；
3. near-zero/far-zero 定点干预复现主要传导路径；
4. global scale 与 far-zero 都能救援，说明风险由净 signed field 决定。

不能用这组实验声称：

- Exp/SBRC 已经取得更高 Hopper normalized return；
- 所有 offline RL 崩溃仅由该机制造成；
- hard filtering 或 distance control 是唯一最优方法。

完整方法效果仍由后续标准 IQL/AWR backbone + Hopper rollout 完成。


---

## Source 2: docs/handoff.md: HANDOFF-DELTA-BLOCKs matching 'EXT-H-E7-'

### Delta block `after_heading:v54-e7-canonical-critic-rollout-audit`

> **v54 增量登记：`EXT-H-E7-Q2` canonical critic、rollout preflight 与 audit 语义修复（不删除 v53 及更早内容）**
>
> - 用户上传的首轮 E7-Q2 单 seed、100-step 运行只保留为工程 pilot：它验证了数据、梯度 probe、干预分支和结果打包链路，但 critic、Positive-only 与方法分支均未达到正式终态，且 normalized-return rollout 不可用，因此不得进入论文正式结果或升级科学状态。
> - 修复 critic 隔离：旧实现会在每个 actor seed 内重新训练 critic，跨 seed 波动仍混入 critic 差异。v54 改为每个 run 只训练或严格复用一个 canonical critic artifact；episode split、observation/return normalizer、terminal critic checkpoint 与 frozen advantage 对全部 actor seeds 和方法完全共享。Formal 只接受通过优化终态与 2× continuation 的 terminal extension checkpoint；best-validation checkpoint 仅作诊断，不再用于 actor advantage。
> - 修复 rollout 可观测性与一键门禁：训练前必须完成 D4RL 注册、`gym.make`、reset、真实 step、随机完整 episode 和 `get_normalized_score` 检查；pilot 与 formal 均 fail closed。失败时先落盘 package versions、兼容 shim、失败阶段、exception 与完整 traceback，再由 hardened guard 打包，避免再次只得到 `rollout_unavailable=1`。
> - 修复任务性能语义：normalized return 未观测时，`task_performance_status` 必须为 `unavailable/not_evaluated/disabled`，`task_performance_collapse=null`；不得把“没有观测”写成 `false`。任务性能崩溃、支持/方差边界和 NaN/Inf 继续分开报告。
> - 修复总门禁命名：根审计分开输出 `engineering_pipeline_complete`、`mechanism_subchecks_passed_for_completed_seeds`、`paired_seed_evidence_complete`、`formal_evidence_prerequisites_complete` 与 `formal_scientific_gate_passed`。Pilot 即使工程与子检查通过，formal gate 也必须为 false；历史 `independent_validation_gate_all_seeds` 仅保留兼容别名且 pilot 固定为 false。
> - `EXT-H-E7-Q2` 的 formal 科学门禁、方法、正式 seeds、阈值与执行顺序不变，仍保持 **not_run + implemented + blocked**；本更新只修复实现隔离、环境交互可观测性和审计语义，不构成正式实验启动或结果升级。

### Delta block `after_heading:v56-e6-parent-closure-route-release`

> **v56 增量登记：E6 父 claim 关闭与 E7-MECH 路线解锁（不删除 v55 及更早内容）**
>
> - 用户在确认 `main` commit `e70f0d84256cdeb6ebbf80b0495a043582787bf6` 已提交后，批准对 **E6 父实验/父 claim** 做范围受限关闭。关闭依据是：`D-U1-E6-SEMANTIC-LONGRUN-01` 的 `360/360` long-run validated 主结果、`D-U1-E6-SEMANTIC-GAP-LONGRUN-01` 的 `100/100` finite-step robustness successor，以及 `D-U1-E6-CONDITIONAL-GAP-01` 的 `200/200` finite-step stress diagnostic。
> - 本次关闭锁定五项论文可用结论：Positive-only 在共享语义 categorical 环境存在 imitation ceiling；适度受控负信号可改善同分布 held-out-context / unseen-action 表现；过强或不抑制负压力会出现性能反转或随 horizon 扩大的退化；semantic alignment 是观察到的未见动作迁移的重要排他性条件；任务性能崩溃、support/temperature boundary 与 NaN/Inf 必须继续分报。
> - 本次关闭不把两个 gap 子实验升级为 long-run validated，也不声称全 alpha 稳态排名、universal alpha optimum、state-distribution OOD generalization、categorical policy 的 Gaussian 二次远场律或跨任务方法优越性。`45/100` semantic-gap plateau 与 `49/200` conditional-gap plateau 的终态边界原样保留。
> - `D-U1-E6-TAPER-01` 降为**可选、独立、非门禁**的方法形状比较：它仍是 `not_run + not_implemented + blocked`，若未来执行，必须另行冻结 semantic remoteness coordinate、paired protocol、全新 untouched seeds 与独立 runner；但它不再是 E6 父 claim 关闭或 E7-MECH 启动的前置条件。
> - `EXT-H-E7-Q2` 由 `blocked/blocked` 迁移为 **ready/active**，科学状态仍为 `not_run`。该迁移只开放已经冻结和实现的 Hopper mechanism formal protocol，不代表 E7 已运行或已有结果。`EXT-H-E7-BENCH-01` 继续 blocked，但依赖收缩为 E7-Q2 交付和随后冻结 shortlist，不再依赖可选 E6-TAPER。
> - 本更新只修改研究治理、路线和相应测试/操作说明；未重跑实验，未更改任何冻结变量、数据规模、seeds、阈值、收敛标准或方法职责。

### Delta block `after_heading:v57-ext-c-e8-v44-offline-negative-bank`

> **v57 增量登记：Countdown `EXT-C-E8-V4.4-OFFLINE-BANK` 固定离线 negative-bank pilot（不删除 v56 及更早内容）**
>
> - 用户确认采用两阶段路线：先完成纯离线固定 negative-bank 实验，再依据离线结果另行讨论并登记 online off-policy successor。本版禁止在方法训练期间重新 rollout、追加 replay 数据或把在线刷新与负样本密度同时改变。
> - V4.3 的 matched near/far pair 与动态 remoteness 修复继续保留。新实验只检验一个更窄的问题：每个 prompt 只有一对固定负样本时，两者可能很快同时远场化；将固定离线负样本覆盖扩大后，是否能持续提供当前局部负信号并超过 Positive-only。该动机不 retroactively 升级 V4.3 的 repository result status。
> - 新实验 ID 为 `EXT-C-E8-V4.4-OFFLINE-BANK`，状态为 **尚未运行（not_run）**，执行类别为 single-seed focused pilot。每个训练 prompt 在方法训练前冻结 `16` 个互不重复、格式合法且 verifier 判错的表达式；原 matched near/far pair 继续保存，只承担瞬时机制与 provenance 对照。
> - 方法训练期间 bank 内容不变。每个 optimizer step 使用当前 learner 对同一固定 bank 重新计分，选择最低 sequence surprisal 为 current near、最高 sequence surprisal 为 current far；选择过程 stop-gradient，随后只对选出的两条 completion 做 train-mode forward/backward，避免建立 16-candidate activation graph。
> - 冻结比较为 `positive_only`、V4.3 `dynamic_controlled_negative`、`bank_dynamic_controlled_negative`、`bank_global_matched` 与 `bank_uncontrolled_negative`。Bank dynamic 对当前 near/far 使用同一 detached token-surprisal taper；bank global 使用与 bank dynamic 初始实际负梯度 RMS 匹配的统一系数；不得预设任何方法胜出。
> - 数据规模、SFT/base gate、seed、学习率、训练 horizon、near/far mix、taper lambda、surprisal threshold、LoRA 配置和 best/terminal 审计沿用 V4.3。主要结果仍是 greedy verifier success、pass@k、valid rate 与 held-out canonical pattern-family coverage/precision；bank 槽位轮换、surprisal 和权重仅是实现诊断。
> - 0.5B 单 seed 只承担 focused pilot。reference greedy `<0.15` 时禁止正式方法排名、online-successor 成功结论或 3B scale-up 结论。任务性能退化、support/structure boundary 与 NaN/Inf 数值失败继续分开报告；不得称 state-distribution OOD generalization。
> - 本 pilot 的用户批准不改变 v56 锁定的正式路线：`EXT-H-E7-Q2` 仍是下一 formal route item。V4.4 可作为非正式外部诊断执行，但不得越过 E7/E7-BENCH 门禁解锁 `EXT-C-E8-SCALE-01`。
> - 本更新基于 `main` commit `c2ad7d5f6fe957d6a6297e6987d878cf72dbf7c8`，只完成文档、实现与测试注册；未运行真实 Qwen/CUDA/BF16-LoRA pilot。

### Delta block `after_heading:v58-e7-gymnasium-v4-rollout`

> **v58 增量登记：`EXT-H-E7-Q2` Gymnasium `Hopper-v4` rollout 兼容修复（不删除 v57 及更早内容）**
>
> - 离线训练数据仍是 `hopper-medium-replay-v2` 的 HDF5 文件，critic、frozen advantage、actor、方法组、正式 seeds、训练 horizon、收敛阈值和 E7 科学职责全部不变；本版只修复真实环境交互的执行后端与 provenance。
> - rollout 评估固定使用服务器本地 Gymnasium `Hopper-v4` 与新版 `mujoco` binding。数据集身份和模拟器环境版本明确分离：不得把 `Hopper-v4` 称为 v4 数据，也不得把该分数表述为逐位复现 legacy `mujoco-py` 环境。
> - normalized return 不再依赖 D4RL 环境对象的 `get_normalized_score()`，而按冻结的 D4RL-v2 Hopper medium-replay reference `min=-20.272305`、`max=3234.3` 手动计算百分制分数；结果必须同时保存 raw return、reference 常量、离线 dataset ID 与 evaluation env ID。
> - legacy D4RL/mujoco-py fallback 明确禁止。主 runner 不导入 `d4rl` 或 `mujoco_py`；环境 preflight 在独立子进程中执行 reset、真实 step、随机 episode 与 reference normalization。若底层 native 进程收到 SIGSEGV、超时或 Python exception，父进程必须落盘退出码、signal、stdout/stderr 与错误报告并在 critic 训练前 fail closed。
> - 正式报告中的准确口径为“offline training on D4RL Hopper medium-replay-v2, evaluated in the Gymnasium Hopper-v4 compatibility environment with D4RL-v2 reference normalization”。该兼容评估可用于 E7 内部 paired mechanism comparison，但不得冒充 exact legacy D4RL leaderboard reproduction。
> - `EXT-H-E7-Q2` 科学状态继续保持 **not_run**。静态检查、单元测试和本地无 MuJoCo 的 mock preflight 只证明实现，不构成 Hopper pilot 或正式结果；下一步仍须在服务器由一键 runner 先通过真实 Gymnasium/MuJoCo preflight。

### Delta block `after_heading:v59-ext-c-e8-v45-offline-bank-tuning`

> **v59 增量登记：Countdown `EXT-C-E8-V4.5-OFFLINE-BANK-TUNING` validation-only α×λ 调参 pilot（不删除 v58 及更早内容）**
>
> - V4.4 的固定 16-negative offline bank、current-policy near/far reselection 与结果边界全部保留。本版只检验一个后续问题：V4.4 从 uncontrolled 到 initialization-matched global 再到 dynamic 的改善趋势停在 Positive-only 附近，是否由整体负梯度强度或指数 taper 速度未落在最佳区间造成。
> - 新实验复用 V4.4 已冻结并完成终态审计的 reference adapter、train/validation/test split、6000-row offline bank 与初始化梯度 calibration；输入在 V4.5 运行前后均做 SHA-256 校验。训练期间不生成新 rollout、不追加 replay、不改变 bank 内容，也不修改 threshold、数据规模、LoRA、学习率或 horizon。
> - Stage A 只扫描 calibrated bank negative scale 的全局 multiplier `alpha in {0.5,1.0,1.5,2.0}`，固定 `lambda=0.7`；Stage B 在 Stage A 选出的 alpha 上扫描 `lambda in {0.3,0.7,1.2}`。两个阶段只用 validation，test 在唯一 alpha/lambda 组合冻结后才允许访问。
> - 调参 seeds 固定为 `1234,2234`。候选选择顺序冻结为 mean best validation greedy success、mean best pass@8、mean terminal validation greedy success、mean best valid rate，再使用保守 tie-break；valid rate `<0.95` 或任何 NaN/Inf 直接使候选失格。
> - 最终只在 untouched training seeds `3234,4234,5234` 上比较 validation-selected bank dynamic 与 Positive-only，并同时报告 best/terminal checkpoint。0.5B reference 若仍低于 greedy `0.15`，本实验即使多 seed 也只能形成 pilot 证据，不得自动升级为正式方法排名或显著性 claim。
> - 该调参只回答“力度和 taper 是否未调到位”，不能证明 negative directional utility 已解决。若所有候选仍与 Positive-only 持平或更差，应转向已讨论但尚未登记的 online off-policy successor，而不是继续扩大网格。
> - 当前 formal route 不变：`EXT-H-E7-Q2` 仍是下一正式实验。V4.5 是外部 focused pilot，不解锁 `EXT-C-E8-SCALE-01`，也不替代 D-U1/E6 的受控因果识别。
> - 本更新基于用户上传 Git bundle 的 `main` commit `58342ae7809354ef8af0e90a1d9938aa51f3a97d`，只完成协议、runner 支持与测试；未运行真实 Qwen/CUDA/BF16-LoRA 调参。

### Delta block `section_end:v56-e6-parent-closure-current-gate`

- **v56 E6 父 claim 关闭覆盖：** E6 的论文核心 claim 现已范围受限关闭；主 long-run 与两个 gap 子实验的原科学状态分别保持 `long_run_validated / finite_step_validated / finite_step_validated`。`D-U1-E6-TAPER-01` 保留为可选非门禁未来工作。当前下一正式 route item 为 `EXT-H-E7-Q2`，registry 状态为 **implemented + ready + active + not_run**；启动后仍须走 canonical hardened guard，且在 raw-complete、终态审计、打包和交付前不得声称 E7 完成。

### Delta block `section_end:v57-countdown-offline-bank-current-gate`

- **Countdown v57 覆盖：** `EXT-C-E8-V4.4-OFFLINE-BANK` 是用户批准的当前离线 focused pilot；V4.3 保留为 fixed-pair predecessor。V4.4 只改变固定负样本覆盖与 current-policy near/far reselection，不引入在线数据刷新。`EXT-H-E7-Q2` 仍是下一正式 route item，`EXT-C-E8-SCALE-01` 继续 blocked。

### Delta block `section_end:v59-countdown-offline-bank-tuning-current-gate`

- **Countdown v59 覆盖：** `EXT-C-E8-V4.5-OFFLINE-BANK-TUNING` 是当前用户批准的离线 focused successor；V4.4 作为 frozen-bank predecessor 保留。V4.5 只调 calibrated global negative multiplier 与 exponential taper lambda，禁止在线刷新、方向筛选或模型规模同时变化。`EXT-H-E7-Q2` 仍是下一 formal route item，`EXT-C-E8-SCALE-01` 继续 blocked。

### Delta block `section_end:v56-e6-parent-closure-execution-order`

13. **v56 执行覆盖：** E6 父 claim 已关闭，`D-U1-E6-TAPER-01` 改为可选非门禁 future study；当前直接进入已实现且 registry 为 ready/active 的 `EXT-H-E7-Q2`（E7-MECH）。E7-Q2 仍为 not_run，必须先完成正式运行、终态审计、打包与交付；其后才允许冻结并实施 `EXT-H-E7-BENCH-01`。E8-MECH/V4.3 与 E8-SCALE 的相对顺序不变。

### Delta block `section_end:v57-e8-offline-bank-execution-order`

14. **v57 执行覆盖：** v56 的 formal 顺序不变，`EXT-H-E7-Q2` 仍是下一正式实验。用户批准的 V4.4 作为 single-seed focused pilot 可独立执行，但必须先完成自身 best/terminal audit 与结果交付，才允许讨论 online off-policy successor；不得一次性同时改变 negative-bank 密度和数据在线刷新机制。

## Source 3: experiments/registry.yaml: experiments[EXT-H-E7-Q2, EXT-H-E7-BENCH-01]

collection: experiments
entries:
- id: EXT-H-E7-Q2
  execution_gate:
    state: ready
    blocked_by: []
    blocking_reason: E6 parent claims are closed; the implemented Hopper mechanism experiment is the next registered formal
      route item. No E7 scientific run has started.
  environment: EXT-H
  name: hopper_gaussian_log_scale_quadratic_external_validation
  status: not_run
  scientific_status: not_run
  parent_experiment: E7-MECH
  registration_base_commit: c7fd41ac663380de71bcd839b76ab4d1e52ae8d0
  implementation_base_commit: 2e04f6dba6d4e87f61920bedb1c464656906bf2b
  implementation_commit: f64452a7452274a183b03c87c39b847039230c00
  claim: In real D4RL Hopper data with a learned critic and an independently trained state-conditioned Gaussian actor, naturally
    far negative samples enter a regime where the corrected log-scale score grows with squared standardized distance, contributes
    measurably to full-parameter gradients or long-run support/task dynamics, and is selectively mitigated by far-field controls.
  role: external_mechanism_validation
  execution_class: formal
  formal_execution:
    channel_ref: hardened-v1
    activation_state: active
    entrypoint_status: implemented
    entrypoint: src/drpo/e7_hopper_q2.py
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
  implementation_state: implemented
  code_entrypoint: src/drpo/e7_hopper_q2.py
  operator_entrypoint: scripts/run_e7_hopper_q2.py
  config_path: configs/e7_hopper_q2_medium_replay_v2.yaml
  operator_guide: docs/README_E7_HOPPER_Q2_ONECLICK.md
  formal_launch_template: python3 scripts/run_experiment_guard_hardened.py --run-class formal --expected-commit "$(git rev-parse
    HEAD)" --require-origin-main-match --experiment-id EXT-H-E7-Q2 --repo-root . --output-root experiments/results/EXT-H-E7-Q2/run_001
    --artifact-output artifacts/EXT-H-E7-Q2_RAW_COMPLETE.zip --required-output RUN_COMPLETE.json --required-output terminal_audit.json
    --required-output aggregate_summary.json --required-output per_seed_summary.csv --source-file src/drpo/e7_hopper_q2.py
    --source-file scripts/run_e7_hopper_q2.py --source-file configs/e7_hopper_q2_medium_replay_v2.yaml -- python3 src/drpo/e7_hopper_q2.py
    run --mode formal --dataset-path /ABSOLUTE/PERSISTENT/PATH/hopper_medium_replay-v2.hdf5 --work-dir experiments/results/EXT-H-E7-Q2/run_001
    --config configs/e7_hopper_q2_medium_replay_v2.yaml --repo-root . --device cuda
  does_not_replace:
  - C-U1-E1-COMP-01
  - C-U1-E1
  - C-U1-E2
  - E7
  scope:
    controlled_identity_source: C-U1-E1-COMP-01
    external_dataset: D4RL_Hopper
    learned_critic: required
    independently_trained_actor: required
    neural_network_pullback_order: excluded
    method_ranking: not_tested_by_this_subclaim
  dataset:
    primary_id: hopper-medium-replay-v2
    basename: hopper_medium_replay-v2.hdf5
    format: legacy_d4rl_hdf5
    sha256: e121c5f7c9857a307baa9edc6a2c3b48e85fedb9ac316ecddd0f48ca7ef4e39b
    role: natural_near_far_external_mechanism_validation
    conservative_replication: hopper-medium
    method_effect_dataset: hopper-medium-expert
  rollout_evaluation:
    offline_dataset_id: hopper-medium-replay-v2
    evaluation_env_id: Hopper-v4
    backend: gymnasium_mujoco
    simulator_binding: mujoco_3_x
    exact_legacy_environment_equivalence: false
    compatibility_environment_disclosure_required: true
    process_isolated_preflight: true
    legacy_d4rl_fallback: forbidden
    normalization:
      protocol: d4rl_v2_reference
      reference_min_score: -20.272305
      reference_max_score: 3234.3
      output_scale: percent
    interpretation: Offline training still uses the D4RL Hopper medium-replay-v2 HDF5 dataset. Policy interaction is evaluated
      locally in Gymnasium Hopper-v4 and normalized with the frozen D4RL-v2 reference scores; this is a compatibility evaluation,
      not an exact legacy mujoco-py leaderboard reproduction.
  critic_protocol:
    target: discounted_monte_carlo_return
    episode_split:
    - 0.8
    - 0.1
    - 0.1
    gamma: 0.99
    frozen_before_actor: true
    terminal_audit: required
  frozen_advantage_protocol:
    definition: r_plus_gamma_v_next_minus_v
    materialize_once: true
    standardize_once_on_critic_train_split: true
    minibatch_renormalization: false
    critic_updates_during_actor_training: false
  actor_protocol:
    family: tanh_squashed_diagonal_gaussian_mlp
    hidden_sizes:
    - 256
    - 256
    log_scale_parameterization: global_diagonal
    positive_only_terminal_before_branching: true
    identical_branch_checkpoint: true
    identical_minibatch_stream_across_methods: true
  coordinate_protocol:
    squashed_actor: tanh_diagonal_gaussian
    inverse_squash: u=atanh(clip(a,-1+epsilon,1-epsilon))
    standardized_residual: z=(u-mu)/sigma
    primary_radius: r=norm(z,2)
    report_also:
    - raw_action_distance
    - pre_squash_distance
  analytic_components:
    mean_score: g_mu_j=(u_j-mu_j)/sigma_j^2
    log_scale_score: g_xi_j=z_j^2-1
    corrected_component: g_xi_j+1=z_j^2
    aggregate_corrected_component: Q_xi=sum_j(g_xi_j+1)=norm(z,2)^2
  preregistered_protocol:
    advantage_source: frozen_learned_critic
    negative_only_comparison: true
    match_or_stratify_by_absolute_advantage: true
    distance_binning: standardized_residual_radius
    analytic_autograd_crosscheck: required
    methods:
    - positive_only
    - signed
    - near_zero
    - far_zero
    - far_cap
    - budget_matched_global
    paired_seeds: required
    terminal_audit: required
    old_600_step_probe_is_formal_evidence: false
  seeds:
    pilot:
    - 42
    formal:
    - 100
    - 101
    - 102
    - 103
    - 104
    - 105
    - 106
    - 107
    - 108
    - 109
  metrics:
  - mean_output_score_norm_by_distance_bin
  - raw_log_scale_output_score_norm_by_distance_bin
  - corrected_log_scale_Q_xi_by_distance_bin
  - joint_output_score_norm_by_distance_bin
  - full_parameter_gradient_norm_by_distance_bin
  - log_scale_to_mean_contribution_ratio
  - analytic_autograd_relative_error
  - policy_support_or_sigma_trajectory
  - mean_saturation_trajectory
  - actor_loss_trajectory
  - normalized_return_trajectory
  - task_performance_collapse_events
  - support_or_variance_boundary_events
  - persistent_or_slow_drift_events
  - nan_inf_numerical_events
  independent_validation_gate:
    natural_far_field_present: required
    corrected_quadratic_branch_empirically_active: required
    measurable_full_parameter_or_long_run_contribution: required
    targeted_far_control_mitigates_dynamics: required
    paired_seed_evidence: required
    terminal_state_audit: required
    identity_only_autograd_check_counts_as_independent_validation: false
  output_contract:
    result_archive_owner: canonical_hardened_channel
    scientific_runner_archive_writes: forbidden
    required_root_files:
    - RUN_COMPLETE.json
    - terminal_audit.json
    - aggregate_summary.json
    - per_seed_summary.csv
    - scientific_run_manifest.json
    upload_instruction: upload_guard_generated_zip_unchanged
    large_checkpoint_policy: persistent_local_index
  reporting_boundary: Passing the gate supports independent external validation that real Hopper training enters and is affected
    by the Gaussian quadratic log-scale far-field regime. It does not independently prove the Gaussian identity, establish
    a quadratic law for neural-network parameter gradients, replace C-U1 causal identification, or rank Exp, Linear, Global
    alpha, SBRC, Hybrid, or Positive-only methods.
  execution:
    state: registered
    run_id: null
    last_heartbeat_utc: null
    process_exit_code: null
  evidence:
    code_committed: true
    implementation_tests_passed: true
    run_started: false
    raw_complete: false
    terminal_audited: false
    package_created: false
    package_filename: null
    package_sha256: null
    delivered_to_user: false
    applied_commit: null
    scientific_status: not_run
- id: EXT-H-E7-BENCH-01
  execution_gate:
    state: blocked
    blocked_by:
    - EXT-H-E7-Q2
    blocking_reason: The large public benchmark remains blocked until EXT-H-E7-Q2 is terminal-audited, packaged, delivered,
      and the controlled-method shortlist is frozen without D4RL retuning. Optional D-U1-E6-TAPER-01 is not a prerequisite.
  environment: EXT-H
  name: d4rl_mujoco_locomotion_method_benchmark
  status: not_run
  parent_experiment: E7-BENCH
  registration_base_commit: f64452a7452274a183b03c87c39b847039230c00
  claim: Test whether the controller shortlist selected in controlled bandits improves multi-seed normalized return and stability
    on the public D4RL MuJoCo locomotion suite, without per-task method-family retuning.
  role: external_large_scale_continuous_benchmark
  execution_class: formal
  implementation_state: not_implemented
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
  suite:
    name: D4RL_MuJoCo_locomotion
    environments:
    - hopper
    - walker2d
    - halfcheetah
    dataset_qualities:
    - medium
    - medium_replay
    - medium_expert
    task_count: 9
    excludes_from_primary_suite:
    - antmaze
    - kitchen
    - adroit
  mandatory_baselines:
  - offline_rl_base
  - positive_only
  - uncontrolled_negative
  - global_alpha
  candidate_controllers:
  - reciprocal_linear
  - reciprocal_quadratic
  - exponential
  shortlist_rule: freeze_after_E4_E6_core_closure_and_E7_mechanism_without_D4RL_retuning
  primary_metrics:
  - normalized_return
  - paired_multiseed_confidence_interval
  - mean_rank_across_nine_tasks
  - worst_seed_return
  - task_performance_collapse_events
  - support_or_variance_boundary_events
  - nan_inf_numerical_events
  protocol_lock_status: seeds_optimizer_base_algorithm_and_exact_versions_pending_before_implementation
  evidence:
    code_committed: false
    run_started: false
    raw_complete: false
    terminal_audited: false
    package_created: false
    scientific_status: not_run
