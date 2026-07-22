# Hopper learned-critic external validation E7

> Generated Stage 4 minimal-context shadow module. Do not edit manually.
> `docs/handoff.md` and `experiments/registry.yaml` remain authoritative.

- Module ID: `hopper_e7`
- Responsibility: Cover learned-critic far-field mechanism validation and D4RL method-effect evidence while preserving the external-validity boundary.
- Content contract topics: none
- Deduplicated overlapping source chunks: 0
- Source hash: `12b391ac8e4c5fefb1795f740513c725b88c411f09bd252b647f2d2195a79451`

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

## Source 2: docs/handoff.md: HANDOFF-DELTA-BLOCK after_heading:v54-e7-canonical-critic-rollout-audit

### Delta block `after_heading:v54-e7-canonical-critic-rollout-audit`

> **v54 增量登记：`EXT-H-E7-Q2` canonical critic、rollout preflight 与 audit 语义修复（不删除 v53 及更早内容）**
>
> - 用户上传的首轮 E7-Q2 单 seed、100-step 运行只保留为工程 pilot：它验证了数据、梯度 probe、干预分支和结果打包链路，但 critic、Positive-only 与方法分支均未达到正式终态，且 normalized-return rollout 不可用，因此不得进入论文正式结果或升级科学状态。
> - 修复 critic 隔离：旧实现会在每个 actor seed 内重新训练 critic，跨 seed 波动仍混入 critic 差异。v54 改为每个 run 只训练或严格复用一个 canonical critic artifact；episode split、observation/return normalizer、terminal critic checkpoint 与 frozen advantage 对全部 actor seeds 和方法完全共享。Formal 只接受通过优化终态与 2× continuation 的 terminal extension checkpoint；best-validation checkpoint 仅作诊断，不再用于 actor advantage。
> - 修复 rollout 可观测性与一键门禁：训练前必须完成 D4RL 注册、`gym.make`、reset、真实 step、随机完整 episode 和 `get_normalized_score` 检查；pilot 与 formal 均 fail closed。失败时先落盘 package versions、兼容 shim、失败阶段、exception 与完整 traceback，再由 hardened guard 打包，避免再次只得到 `rollout_unavailable=1`。
> - 修复任务性能语义：normalized return 未观测时，`task_performance_status` 必须为 `unavailable/not_evaluated/disabled`，`task_performance_collapse=null`；不得把“没有观测”写成 `false`。任务性能崩溃、支持/方差边界和 NaN/Inf 继续分开报告。
> - 修复总门禁命名：根审计分开输出 `engineering_pipeline_complete`、`mechanism_subchecks_passed_for_completed_seeds`、`paired_seed_evidence_complete`、`formal_evidence_prerequisites_complete` 与 `formal_scientific_gate_passed`。Pilot 即使工程与子检查通过，formal gate 也必须为 false；历史 `independent_validation_gate_all_seeds` 仅保留兼容别名且 pilot 固定为 false。
> - `EXT-H-E7-Q2` 的 formal 科学门禁、方法、正式 seeds、阈值与执行顺序不变，仍保持 **not_run + implemented + blocked**；本更新只修复实现隔离、环境交互可观测性和审计语义，不构成正式实验启动或结果升级。

## Source 3: docs/handoff.md: HANDOFF-DELTA-BLOCK after_heading:v56-e6-parent-closure-route-release

### Delta block `after_heading:v56-e6-parent-closure-route-release`

> **v56 增量登记：E6 父 claim 关闭与 E7-MECH 路线解锁（不删除 v55 及更早内容）**
>
> - 用户在确认 `main` commit `e70f0d84256cdeb6ebbf80b0495a043582787bf6` 已提交后，批准对 **E6 父实验/父 claim** 做范围受限关闭。关闭依据是：`D-U1-E6-SEMANTIC-LONGRUN-01` 的 `360/360` long-run validated 主结果、`D-U1-E6-SEMANTIC-GAP-LONGRUN-01` 的 `100/100` finite-step robustness successor，以及 `D-U1-E6-CONDITIONAL-GAP-01` 的 `200/200` finite-step stress diagnostic。
> - 本次关闭锁定五项论文可用结论：Positive-only 在共享语义 categorical 环境存在 imitation ceiling；适度受控负信号可改善同分布 held-out-context / unseen-action 表现；过强或不抑制负压力会出现性能反转或随 horizon 扩大的退化；semantic alignment 是观察到的未见动作迁移的重要排他性条件；任务性能崩溃、support/temperature boundary 与 NaN/Inf 必须继续分报。
> - 本次关闭不把两个 gap 子实验升级为 long-run validated，也不声称全 alpha 稳态排名、universal alpha optimum、state-distribution OOD generalization、categorical policy 的 Gaussian 二次远场律或跨任务方法优越性。`45/100` semantic-gap plateau 与 `49/200` conditional-gap plateau 的终态边界原样保留。
> - `D-U1-E6-TAPER-01` 降为**可选、独立、非门禁**的方法形状比较：它仍是 `not_run + not_implemented + blocked`，若未来执行，必须另行冻结 semantic remoteness coordinate、paired protocol、全新 untouched seeds 与独立 runner；但它不再是 E6 父 claim 关闭或 E7-MECH 启动的前置条件。
> - `EXT-H-E7-Q2` 由 `blocked/blocked` 迁移为 **ready/active**，科学状态仍为 `not_run`。该迁移只开放已经冻结和实现的 Hopper mechanism formal protocol，不代表 E7 已运行或已有结果。`EXT-H-E7-BENCH-01` 继续 blocked，但依赖收缩为 E7-Q2 交付和随后冻结 shortlist，不再依赖可选 E6-TAPER。
> - 本更新只修改研究治理、路线和相应测试/操作说明；未重跑实验，未更改任何冻结变量、数据规模、seeds、阈值、收敛标准或方法职责。

## Source 4: docs/handoff.md: HANDOFF-DELTA-BLOCK after_heading:v57-ext-c-e8-v44-offline-negative-bank

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

## Source 5: docs/handoff.md: HANDOFF-DELTA-BLOCK after_heading:v58-e7-gymnasium-v4-rollout

### Delta block `after_heading:v58-e7-gymnasium-v4-rollout`

> **v58 增量登记：`EXT-H-E7-Q2` Gymnasium `Hopper-v4` rollout 兼容修复（不删除 v57 及更早内容）**
>
> - 离线训练数据仍是 `hopper-medium-replay-v2` 的 HDF5 文件，critic、frozen advantage、actor、方法组、正式 seeds、训练 horizon、收敛阈值和 E7 科学职责全部不变；本版只修复真实环境交互的执行后端与 provenance。
> - rollout 评估固定使用服务器本地 Gymnasium `Hopper-v4` 与新版 `mujoco` binding。数据集身份和模拟器环境版本明确分离：不得把 `Hopper-v4` 称为 v4 数据，也不得把该分数表述为逐位复现 legacy `mujoco-py` 环境。
> - normalized return 不再依赖 D4RL 环境对象的 `get_normalized_score()`，而按冻结的 D4RL-v2 Hopper medium-replay reference `min=-20.272305`、`max=3234.3` 手动计算百分制分数；结果必须同时保存 raw return、reference 常量、离线 dataset ID 与 evaluation env ID。
> - legacy D4RL/mujoco-py fallback 明确禁止。主 runner 不导入 `d4rl` 或 `mujoco_py`；环境 preflight 在独立子进程中执行 reset、真实 step、随机 episode 与 reference normalization。若底层 native 进程收到 SIGSEGV、超时或 Python exception，父进程必须落盘退出码、signal、stdout/stderr 与错误报告并在 critic 训练前 fail closed。
> - 正式报告中的准确口径为“offline training on D4RL Hopper medium-replay-v2, evaluated in the Gymnasium Hopper-v4 compatibility environment with D4RL-v2 reference normalization”。该兼容评估可用于 E7 内部 paired mechanism comparison，但不得冒充 exact legacy D4RL leaderboard reproduction。
> - `EXT-H-E7-Q2` 科学状态继续保持 **not_run**。静态检查、单元测试和本地无 MuJoCo 的 mock preflight 只证明实现，不构成 Hopper pilot 或正式结果；下一步仍须在服务器由一键 runner 先通过真实 Gymnasium/MuJoCo preflight。

## Source 6: docs/handoff.md: HANDOFF-DELTA-BLOCK after_heading:v59-ext-c-e8-v45-offline-bank-tuning

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

## Source 7: docs/handoff.md: HANDOFF-DELTA-BLOCK after_heading:v62-ext-c-e8-v46-online-offpolicy-replay

### Delta block `after_heading:v62-ext-c-e8-v46-online-offpolicy-replay`

> **v62 增量登记：Countdown `EXT-C-E8-V4.6-ONLINE-OFFPOLICY-REPLAY` 真正在线 off-policy replay 2×2 pilot（不删除 v61 及更早内容）**
>
> **v61（E4-TAPER Near-Retention 协议冻结与实现版）历史标题与全部内容继续保留。**
>
> - V4.5 的离线调参职责与结果边界保持不变。用户已批准停止继续扩大 frozen-bank alpha/lambda 网格，转向在线刷新数据；本版登记并实现新的独立 successor，不追溯修改 V4.4/V4.5。
> - 核心问题拆成 2×2：`frozen_positive_only`、`frozen_dynamic`、`online_positive_only`、`online_dynamic`。它分别识别数据刷新收益、负梯度在冻结数据上的增量、负梯度在在线 replay 上的增量，以及 refresh×negative interaction；禁止只比较 online dynamic 与历史 Positive-only 后把差异全部归因于负梯度。
> - 在线分支保持一个 learner、optimizer 与全局 scheduler 跨 4 个 collection phases 连续训练。第 0 phase 是 fresh-only warmup；此后每个 optimizer update 精确使用 4 个 fresh microbatches 与 4 个 stale microbatches，stale 数据来自最近 3 个 collector versions 中的旧版本，因此同时满足 online data acquisition 与 off-policy replay reuse。
> - 每个 phase 从当前 learner 生成新 rollout，verifier 只接收合法且使用全部数字的表达式；16-negative bank 必须全部来自当前 collector 的真实生成，禁止 synthetic negative fallback。正分支优先使用与 oracle canonical structure 相同的当前生成正确答案，缺失时才回退 frozen oracle，并单独报告 generated-positive fraction。
> - V4.5 选出的 alpha/lambda、surprisal threshold=2、near/far 0.5/0.5、BF16 LoRA、learning rate、总 optimizer-update budget 与 gradient clipping 全部冻结；不在 V4.6 再调参。新 paired training seeds 为 `6234,7234,8234`，test 只在全部四个 cells 训练结束后访问。
> - 机制审计改为直接测量实际参与训练的 bank-selected current near/far：surprisal、raw/controlled gradient norm、与 positive update 的 cosine、collector version、replay age 和 taper weight。旧 fixed-pair diagnostics 继续保留作 provenance，但不得再代替实际选中样本诊断。
> - 任务性能退化、valid/support/structure boundary 与 NaN/Inf 数值失败继续分开报告；best 与 terminal checkpoint 同时报。0.5B reference 若仍低于既有 15% greedy floor，本实验即使多 seed 也只形成 pilot，不能自动生成正式方法排名或解锁模型规模结论。
> - 当前 formal route 不变：`EXT-H-E7-Q2` 仍是下一正式 route item。V4.6 是可独立执行的外部 focused pilot，不替代 C-U1/D-U1 因果识别；`EXT-C-E8-SCALE-01` 的 Countdown blocker 更新为 V4.6 的审计与交付。
> - 本更新基于用户上传 Git bundle 的 `main` commit `7dcde2095e0f0aa4a7302a829667c1955c187738`；只实现协议、runner、实际选中样本诊断与测试，尚未运行真实 Qwen/CUDA/BF16-LoRA pilot。

## Source 8: docs/handoff.md: HANDOFF-DELTA-BLOCK after_heading:v64-e7-q2-acceptance-pipeline

### Delta block `after_heading:v64-e7-q2-acceptance-pipeline`

> **v64 增量登记：`EXT-H-E7-Q2` acceptance pipeline v4.2 与一键正式运行（不删除 v63 及更早内容）**
>
> - 本版完整继承 v63（E4-TAPER Near-Retention 结果沉淀与闭环协议版），本次只修复 Hopper E7-Q2 的 critic/actor 验收语义、控制审计与本地执行入口；不改变数据集、模型结构、学习率、正式 seeds、训练 horizon、near/far matching、far-cap 定义或 E7 的外部机制验证职责。`EXT-H-E7-Q2` 继续保持 **not_run + implemented + ready + active**；用户上传的 formal-scale pilot 只登记为工程与协议诊断，不升级为正式科学结果。
> - **旧结论与问题：**v54 将 formal critic 绑定到 `optimization_terminal`，并用 held-out validation loss 的参数梯度和未归一化全模型 update norm 判定训练稳态。formal-scale pilot 表明这些绝对阈值与 256×256 MLP 的尺度不匹配，且 validation gradient 并不等价于训练目标 stationarity；同类门禁也阻塞 Positive-only actor。不得通过硬编码 `optimization_terminal=True` 或把 update norm 伪造为 0 绕过审计。
> - **替代协议：**v4.2 将 optimizer stationarity、checkpoint selection 与 frozen-advantage acceptance 分离。stationarity 使用固定 train-audit loss、validation-MSE slope、相对参数更新以及可容纳时的精确 2× continuation；raw gradient/update 继续保存为诊断。若真实 optimizer terminal 通过且 final/best validation-MSE ratio 仍在门限内，则选择 terminal-extension checkpoint；否则选择最低 validation MSE checkpoint。formal artifact acceptance 使用 validation R² ≥ 0.50、validation Pearson ≥ 0.75、final/best validation-MSE ratio ≤ 1.05，并在 actor training split 上要求 selected-vs-final advantage sign agreement ≥ 0.95、Pearson/Spearman ≥ 0.98、negative-set Jaccard ≥ 0.90；test R²/Pearson 只作最终报告，不参与 checkpoint 选择或门禁。`optimization_terminal` 继续如实单独报告，不得被强制置真。
> - Actor 终态同样改用相对参数更新与固定 audit window 上的 scale-normalized policy-state drift；核心状态量冻结为 `mean_abs / sigma_mean / phantom_distance_mean`，阈值为窗口拟合总漂移不超过 `0.01`。`positive_nll` 可能跨零且受 minibatch 噪声影响，只保留 slope 诊断，不再阻塞终态。任务性能崩溃、support/variance boundary、NaN/Inf numerical collapse、persistent drift 与 finite terminal 继续分开输出。2× continuation 只在 `2*candidate_step <= max_steps` 时建立候选，避免旧 `min(max_steps, 2*step)` 与 `final>=2*step` 的不可满足组合。
> - 核心机制 gate 只保留 natural far field、corrected Gaussian log-scale quadratic geometry/analytic-autograd agreement 和 measurable full-parameter far/near amplification。`log-scale 是否每个 seed 都压过 mean` 降为诊断，不再错误地作为二次几何成立的必要条件。 Registry 中旧的聚合 gate 名称仅以 `superseded_by_*` provenance 标记保留，不再参与验收。
> - 控制结果拆为 diagnostic-score mitigation、support-boundary rescue、task-performance rescue 与 finite-terminal rescue，禁止继续用任一项成立的 OR 布尔量冒充长期救援。旧 initial-only `budget_matched_global` 不再进入正式方法集合；新 `dynamic_budget_matched_global` 在每个 minibatch 上以 detached `sum(|A| × joint_output_score)` proxy 动态匹配同批 Far-cap 保留预算。该 proxy matching 不等同于精确全参数梯度预算匹配，也不得据此预设 Distance/Global 方法排名。
> - Canonical critic artifact schema 升级为 v2，并继续对 mode、config hash、dataset、transition count、dimensions、canonical seed 与 runner version 做 exact identity 校验；pilot、v4.1 或其他 formal 身份的 artifact fail closed。
> - 操作入口升级为 Countdown 风格一键命令：在 clean current `main`、已设置 `DRPO_HOPPER_MEDIUM_REPLAY` 或标准数据路径时执行 `python3 scripts/run_e7_hopper_q2.py`，默认 formal、自动创建 timestamped persistent work directory，并由 hardened guard 打包结果。`--plan-only` 只解析和打印完整命令，不启动训练；pilot 仍不得冒充 formal evidence。

## Source 9: docs/handoff.md: HANDOFF-DELTA-BLOCK after_heading:e7-q2-fixed-budget-longrun-v43

### Delta block `after_heading:e7-q2-fixed-budget-longrun-v43`

> **E7-Q2 v4.3 增量登记：`EXT-H-E7-Q2` fixed-budget long-run v4.3 与重跑协议（不删除此前任何内容）**
>
> - 本版继承当前 `main` 及此前全部历史，只修订 Hopper E7-Q2 的训练停止规则、critic canonical checkpoint 选择、终态审计职责和一键执行配置。E7 仍只承担 learned-critic 外部机制验证，不替代 C-U1 受控因果识别，也不构成 D4RL 方法排名。`EXT-H-E7-Q2` 继续是 **not_run + implemented + ready + active**；本版未运行真实 Hopper formal。
> - **旧协议—问题—新证据：**v4.2 由短窗口 stationarity candidate 和 2× extension 决定 critic/actor 提前停止。用户上传的 formal-scale pilot 中，critic 在 7600 步即被判为 terminal，而上一轮 20000 步 critic 的 test R²/Pearson 仍更高；Positive-only、Far-zero、Far-cap、Dynamic Global 与 Signed/Near-zero 又停在不同 horizon，导致“最终值”不处于相同训练预算。该结果说明短窗口暂时变慢不能代替 D4RL 长程预算，也不能作为方法间公平终态比较。
> - **Pilot 机制证据边界：**上传包 SHA-256 为 `deefbe216ca5c99622c84831b4546da10203610c07736992c51cf23f679f1017`。该 pilot 中 far/near `|A|` 约为 `0.99992`、标准化距离约为 `3.596×`、全参数负梯度约为 `3.47×`；Signed 与 Near-zero 均出现 `10/10` 任务性能崩溃和 `10/10` support/variance boundary，Far-zero 将 support/variance boundary 降为 `0/10`。这些只登记为 **pilot / finite-horizon mechanism diagnostic**，不升级为 formal result、稳态结论或方法排名。NaN/Inf 与任务崩溃、support/variance boundary 继续分开报告。
> - **Critic v4.3 固定预算：**formal canonical critic 固定训练 `100000` optimizer steps，每 `2000` 步评估一次；除 loss/gradient/parameter 出现 NaN/Inf 等数值失败外禁止提前停止。跑满后始终选择最低 validation MSE checkpoint 生成 frozen advantage；final checkpoint 仅用于 selected-vs-final 稳定性对照。旧 optimization-terminal、validation R²/Pearson、final/best ratio、advantage sign/rank/Jaccard 阈值继续原样记录，但全部降为 report-only diagnostics，不再阻塞 formal actor 执行。formal operational gate 只要求固定预算完成且 selected checkpoint 指标有限。不得把固定 100k 写成“critic 已收敛”。
> - **Actor v4.3 固定预算：**Positive-only initialization 固定 `100000` optimizer steps；从同一 fixed-budget checkpoint 分叉的 `signed / near_zero / far_zero / far_cap / dynamic_budget_matched_global` 各固定 `200000` steps，所有分支 horizon 完全相同。actor 每 `5000` 步做 audit、每 `25000` 步以 `5` episodes 做中间 rollout，固定预算末端以 `20` paired episodes 做最终 rollout；只有 NaN/Inf numerical failure 允许提前停止，support boundary、任务退化或持续漂移不得触发早停。`signed` 明确定义为保留正负 advantage、且不做 near/far 控制的 full signed-advantage baseline，不是新算法。
> - **终态审计职责：**terminal candidate、relative update、state drift 与 2× continuation 只用于训练结束后的分类，不再控制停止。满足 2× confirmation 且无 boundary 才可标为 `finite_terminal`；跑满固定 horizon 但仍漂移时标为 `persistent_or_slow_drift`，无法判定时标为 `fixed_horizon_inconclusive`。固定 horizon 本身不得自动解释为 convergence。根审计分别记录 critic fixed-budget completion、Positive-only fixed-budget completion、所有 branch fixed-budget completion、任务性能崩溃、support/variance boundary、NaN/Inf 与 terminal classification。
> - Canonical critic artifact schema 升级为 `v3`；v2、pilot、不同 mode/config/dataset/transition count/seed/runner identity 的 artifact 均 fail closed。Countdown 风格入口仍为 `python3 scripts/run_e7_hopper_q2.py`；默认 formal，通过 hardened guard 持久化 heartbeat、失败证据和最终 raw-complete 包。应用本更新后必须从 clean current `main` 重新训练 critic 与全部 actor 分支，旧 v4.2 critic 不得跨 schema 复用。

## Source 10: docs/handoff.md: HANDOFF-DELTA-BLOCK after_heading:v68-ext-h-e7-q2-longrun-closure

### Delta block `after_heading:v68-ext-h-e7-q2-longrun-closure`

> **v68 增量登记：`EXT-H-E7-Q2` Hopper learned-critic 长程机制结果闭环（不删除 v67 及更早内容）**
>
> - 正式运行绑定 clean detached commit `c5c638b47c945f5a3ecb8243f679caa31a129f9e`，运行开始时权威 `origin/main` 与本地 HEAD 一致；`hopper-medium-replay-v2` 数据 SHA-256 为 `e121c5f7c9857a307baa9edc6a2c3b48e85fedb9ac316ecddd0f48ca7ef4e39b`。共享 critic 固定 100k steps、Positive-only 固定 100k steps、五个分支各固定 200k steps，seeds `100--109` 全部完成，终态记录齐全，NaN/Inf 为 `0/60`。
> - Advantage 匹配通过：far/near `|A|` 均值比为 `0.999770x`。自然 far negatives 的标准化距离、corrected `Q_xi` 与全参数负梯度 far/near 均值比分别为 `3.845x`、`14.547x` 和 `4.206x`；`Q_xi` 对 radius 的 log-log slope 为 `2.000000000019`，解析式与 autograd 最大相对误差均值为 `6.600e-08`。
> - `Signed` 与 `Near-zero` 均为 `10/10` 任务性能崩溃、sigma 触底并接近完整动作边界饱和；删除 near negatives 没有救援。`Far-zero`、`Far-cap` 与 dynamic budget-matched Global 均在 `10/10` paired seeds 中高于 Signed，平均终态 return 增益分别为 `+21.546`、`+10.484` 和 `+14.779`。这支持远场异常负梯度是该 Hopper learned-critic 设置中 support contraction 与任务性能失败的主要传导路径之一。
> - 三类事件严格分报：task-performance collapse、support/variance boundary 与 NaN/Inf numerical failure 不得互换。二值 boundary event 也不得替代严重度：Signed/Near-zero 的 mean boundary fraction 约为 `1.0`，Far-zero 为 `0.1215`，接近 Positive-only 的 `0.1123`。
> - E7-Q2 的科学状态登记为 **long_run_validated**，范围仅限 Hopper external mechanism validation。Positive-only 是删除全部负信号的稳定参考，不是本机制实验的主 baseline；主 baseline 是 Signed，Near-zero 是负向因果对照，Far-zero/Far-cap 是定点干预，Global 是幅度中介对照。
> - 本结果不授权有限稳态、通用方法排名、当前控制超过 Positive-only、远场是所有真实任务唯一失稳原因，或 exact legacy D4RL leaderboard reproduction。near/far 二分只用于机制识别；连续 taper 和方法收益由后续独立实验承担。
> - Compact closure evidence 位于 `outputs/e7_hopper_q2/`。`EXT-H-E7-BENCH-01` 的 E7-Q2 前置条件已满足，但仍因 controlled-method shortlist 未在不使用 D4RL 调参的条件下冻结而保持 blocked；本闭环不自动启动 benchmark。
> - v67 已登记的 `E8-TAPER` 路线与门禁保持不变；本次 E7-Q2 闭环不修改 Countdown 方法实验职责或执行顺序。

## Source 11: docs/handoff.md: HANDOFF-DELTA-BLOCK after_heading:v69-e7-bench-parallel-pilot

### Delta block `after_heading:v69-e7-bench-parallel-pilot`

> **v69 增量登记：`EXT-H-E7-BENCH-01` 两数据集并行 Pilot 与正式并行拓扑（不删除 v68 及更早内容）**
>
> - 本版不新增顶层实验 ID；在既有 `EXT-H-E7-BENCH-01` 下登记一个 **pilot** 子阶段。Pilot 只检查数据加载、learned-critic/actor/rollout 链路、连续 taper 实现、运行成本、artifact 体积、断点恢复及初步 paired direction，不得填入正式 9-task 主表，不得据此更换方法族、按任务调参或升级正式科学状态。
> - Pilot development seeds 冻结为 `200, 201, 202, 203`。方法冻结为 `Positive-only`、`Signed`、`Global alpha=0.75`、`Reciprocal-Linear`、`Reciprocal-Quadratic`、`Exponential`。三种 taper 沿用 `C-U1-E4-TAPER-NEAR-RETENTION-01` development seeds `0--4` 的冻结系数：`0.4362580032734791`、`0.5520268617673281`、`0.374162511054291`；标准化距离 reference/near boundary 均为 `5.0`，禁止 D4RL 后验重调。
> - 两个上传数据单元必须按真实 provenance 区分：`hopper-medium-expert-v2` 是 legacy D4RL-v2 HDF5，使用 Hopper-v4 与 D4RL-v2 normalized return；上传的 `mujoco/hopper/medium-v0` metadata 明确属于 **Minari Hopper-v5**，不是 D4RL `hopper-medium-v2`，因此只作为 pilot/plumbing cell、只报告 raw return，不能计入正式 D4RL 9-task 主表。正式 Hopper-medium cell 仍需另行冻结精确 D4RL 版本。
> - Pilot 固定预算为：每数据集一个 canonical critic `20k` optimizer steps、每 `(dataset, seed)` Positive-only `20k` steps、其余每个 method branch `40k` steps；只有 NaN/Inf 可提前终止。固定 horizon 不等于收敛，仍需分开报告任务性能崩溃、support/variance boundary、NaN/Inf 与 persistent/slow drift。
> - 为使用 384 核 CPU，执行器冻结为三阶段并行：`2` 个 dataset critic workers 并行；`8` 个 `(dataset, seed)` Positive-only workers 并行；`40` 个 `(dataset, seed, method)` branch workers 并行。线程预算分别为 `64/32/8`，峰值 `320` threads，保留系统和 I/O 余量。seed 与 method 均禁止顶层串行；每个 branch 从对应的同一 Positive-only checkpoint 分叉，输出目录隔离，resume 粒度为 `dataset_seed_method`。
> - 正式 9-task E7-BENCH 同步登记为 staged resource-pool 并行，branch scheduling unit 为 `task_seed_method`，禁止 serial seed loop 与 serial method loop；但正式 exact D4RL versions、formal seeds、offline-RL base、optimizer 和 full budget 尚未冻结，故 formal activation 继续 blocked。Pilot ready 不等于 formal ready。
> - 新入口为 `src/drpo/e7_bench.py`、`scripts/run_e7_bench.py`，配置为 `configs/e7_bench_pilot.yaml`，协议说明为 `docs/e7_bench_pilot.md`。当前仅完成实现、静态/单元、真实数据 loader 与 canonical critic 短程 smoke；当前环境缺少 `gymnasium`，因此 actor/rollout 短程 smoke 未执行。该限制不等于 Pilot 已运行，更不支持任何方法优于 Positive-only。正式启动时 runner 会在长程 critic 之前预检 384 核线程预算、Gymnasium/MuJoCo 环境及数据—环境维度一致性。

## Source 12: docs/handoff.md: HANDOFF-DELTA-BLOCK after_heading:v71-e7-bench-long-budget-parallel-pilot

### Delta block `after_heading:v71-e7-bench-long-budget-parallel-pilot`

> **v71 增量登记：`EXT-H-E7-BENCH-01` 长预算、等 actor horizon 与强 resume identity 修正（不删除 v70 及更早内容）**
>
> - **旧设计：**v69 将 Pilot 冻结为 critic `20k`、Positive-only `20k`、其余分支 `40k`。该设计只适合作为 engineering smoke，不足以承担用户要求的科学 Pilot；同时 Positive-only 总 actor horizon 只有 `20k`，其余方法为 `60k`，比较预算不公平。
> - **问题与修正依据：**已完成的 E7-Q2 formal long-run 使用 critic `100k`、Positive-only initialization `100k`、每分支 continuation `200k`。当前 Pilot 既要初步观察连续 taper 方向，又不得以并行缩短单 worker 科学预算，因此恢复到同一训练量级。v69 短预算尚未产生科学结果，不登记为被否定结果，只保留为 superseded engineering-smoke design。
> - **新冻结预算：**每数据集 canonical critic 固定 `100000` optimizer steps；每 `(dataset, seed)` 先训练共享 Positive-only warm-start `100000` steps；随后 `Positive-only / Signed / Global alpha / Reciprocal-Linear / Reciprocal-Quadratic / Exponential` 六种方法都从同一 warm-start 并行 continuation `200000` steps。每个比较方法总 actor horizon 因而统一为 `300000` steps。只有 NaN/Inf numerical failure 可提前终止，固定 horizon 仍不等于收敛。
> - **并行修正：**三阶段改为 `2` 个 critic workers、`8` 个 shared warm-start workers、`48` 个 `(dataset, seed, method)` continuation workers。线程分配为 `64 / 32 / 7`，峰值 `336` threads，在 384 核服务器上保留 `48` threads 余量。Positive-only 不再作为第二阶段终点结果，而是第三阶段中的完整等时 continuation 分支。seed 与 method 顶层串行继续被禁止。
> - **恢复身份修正：**每个 run 和 worker 必须绑定 exact Pilot config SHA-256、E7-Q2 base-config SHA-256、runner/protocol version、dataset SHA-256、stage budget、method identity 与 taper 参数。旧 `20k/20k/40k` work directory 不允许在新协议下 `--resume`；coordinator 必须 fail closed 并要求新 work directory。相同 run identity 下的 incomplete worker 先归档，再仅重跑对应 task-seed-method 单元。
> - **并行失败与预算记账修正：**runner `0.2.1` 在任一 worker 失败后主动终止仍在运行的 peer subprocesses，避免其余几十个 200k-step worker 继续空耗。canonical critic 与共享 Positive-only warm-start 必须完整达到各自冻结预算，才允许进入下游阶段；method continuation 仅允许因 NaN/Inf 提前终止，并分别记录 scheduled horizon 与 actually executed steps，不能把数值失败伪装成完成 300k actor path。
> - **Taper 公式锁定：**令 `u=d/5` 为标准化 Gaussian distance，Reciprocal-Linear 为 `1/(1+c u)`，Reciprocal-Quadratic 为 `1/(1+c u^2)`，Exponential 为 `exp(-c u)`。其中 quadratic 指 distance-squared，即 Gaussian surprisal-order proxy；不得误写为 reciprocal-squared-surprisal 对应的四次距离形式。
> - **正式 E7-BENCH 并行约束同步：**正式 9-task benchmark 继续以 `task_seed_method` 为 continuation 调度单元，Positive-only 也必须是 equal-horizon continuation branch；formal exact seeds、D4RL versions、base algorithm、optimizer 与 full budget 仍未冻结，因此 formal activation 继续 blocked。本修正不等于正式实验可以启动。
> - Pilot 仍只允许形成 `pilot` 证据：不得据此按 D4RL task 更换函数族或系数，不得填入正式 9-task 主表，不得声称有限稳态、通用方法排名或当前 taper 必然超过 Positive-only。任务性能崩溃、support/variance boundary 与 NaN/Inf numerical failure 继续分开报告。

## Source 13: docs/handoff.md: HANDOFF-DELTA-BLOCK section_end:e7-canonical-shortlist-1m-pilot-gate

### Delta block `section_end:e7-canonical-shortlist-1m-pilot-gate`

- **Hopper E7 canonical-backbone two-dataset 1M pilot（`EXT-H-E7-BENCH-01`）：**原始 `ExpRank_MR` baseline-sanity 已在 `hopper-medium-replay-v2` 与 `hopper-medium-expert-v2`、paired seeds `200--203`、每 run `1,000,000` updates、每 `50,000` steps 以 `10` episodes 评估的协议下完成 `8/8` runs、`0` failures。Replay / Expert 的 final mean 分别为 `73.017 / 80.225`，best mean 为 `98.854 / 106.145`。该结果只证明旧 canonical joint actor--critic backbone 恢复了强性能量级；评估使用 Gymnasium Hopper-v4 与 D4RL-v2 reference normalization，不是 exact legacy `mujoco-py` 复现。结果包未绑定 DRPO repository commit，只绑定 canonical Python tree、`agents.py` 与 trainer 指纹，因此登记为可审计的 lightweight **pilot result archive**，不升级为 formal leaderboard evidence。
- **训练动力学边界：**四个 seeds 通常在约 `400k--500k` 内至少一次进入高分区，但到 `1M` 仍存在明显 best-to-final 回落和跨评估点波动，禁止称为收敛或稳定平台。后续 canonical pilot 的预注册主窗口固定为 `750k, 800k, 850k, 900k, 950k, 1000k`；主指标为 late-window mean，另报 late-window std/min/max、final、best/best-step、best-to-final、best-to-late-mean 与 terminal slope。只有终态审计通过后才允许稳态或方法排名表述。
- **E7 两条职责继续分离：**旧 frozen-critic / `EXT-H-E7-Q2` 线路保留为 learned-critic 外部机制诊断；canonical actor--critic 线路只承担强 backbone 上的方法效果与后期稳定性 external-validity pilot。两者均不替代 C-U1/D-U1 受控因果识别，也不能把当前性能差距单独归因于 critic 冻结。
- **下一轮固定 shortlist：**比较 `original_exp_rank_mr`、`positive_only`、`global_neg_0p11`、`global_neg_0p011`、`reciprocal_linear_max0p011`、`reciprocal_quadratic_max0p011`、`exponential_max0p011`，共 `2 datasets x 4 seeds x 7 methods = 56` branches；所有 branch 统一 `1M` updates、相同 evaluation cadence 与 canonical trainer。`global_neg_0p011` 与三种 distance taper 的最大负系数 `0.011` 对齐，用于区分选择性远场控制与整体缩小负梯度。
- **坐标与调参边界：**本 canonical adapter 使用 detached RMS standardized action distance、reference distance `2.0`；它与 frozen-critic E7 旧坐标中的 reference `5.0` 是不同实现，数值不得混称。本轮复用已存在的 canonical two-dataset adapter 系数 `0.4362580032734791 / 0.5520268617673281 / 0.374162511054291`，不根据 300k 结果或数据集分别重调。
- **并行与启动门禁：**56 branches 以 `dataset_seed_method` 为调度单元，默认 `40` 个可恢复 subprocess workers、每 worker `2` 个 OMP threads；seed 与 method 顶层串行均禁止。完整 sweep 启动前必须依次通过 independent reviewer、短程真实 liveness、authority/governance tests，并 merge 到 clean `main` commit；dev-branch 或未审查 commit 不得作为正式启动来源。
- **报告分离：**任务性能崩溃、support/variance-boundary event 与 NaN/Inf numerical failure 继续分别统计。该阶段仍是 two-dataset pilot，`formal_evidence_allowed=false`；正式 D4RL-9 protocol lock 继续 blocked。

## Source 14: docs/handoff.md: HANDOFF-DELTA-BLOCK section_end:e7-ppo-w0-exp-grid-running-pilot

### Delta block `section_end:e7-ppo-w0-exp-grid-running-pilot`

- **Hopper/Walker E7 PPO direct-`w(0)` × EXP 网格 pilot（`EXT-H-E7-PPO-W0-EXP-GRID-01`）：**登记为正在运行的 development screening pilot，是 `EXT-H-E7-PPO-STABILITY-01` 的非破坏性 successor；旧实验协议与历史结果完整保留。实验公开参数化固定为 `u=d/2`、`w(d)=w(0)exp(-cu)`，不再暴露 `scale` 或 `canonical_alpha`；`w(0)=0.11` 与旧 `alpha=0.11 × scale=1` 仅作为实现等价锚点。网格为 `w(0)={0,0.025,0.05,0.11,0.25,0.5,1}` 与 `c={0,0.25,0.5,1,1.5}`，`w(0)=0` 去重为 Positive-only，共 31 个参数点；在 `hopper-medium-expert-v2`、`walker2d-medium-v2`、`walker2d-medium-replay-v2` 和 development seeds `200,201` 上形成 186 个 PPO-only、500k branches。held-out seeds `204--207` 保持未使用；自动资源探测只决定安全 subprocess 数，不得改变科学矩阵。用户已报告服务器启动，但当前无终态结果，禁止把运行中状态、500k endpoint、smoke/liveness 或 dev-seed 筛选升级为收敛、稳态或正式方法排名。任务性能退化/崩溃、support/variance boundary 与 NaN/Inf 必须分开审计；最终应寻找跨三任务共同稳健的 `(w(0),c)`，不得按数据集事后挑选不同 cell 冒充统一方法。Hopper/Walker 只承担 external validity，不替代 C-U1/D-U1 受控机制识别。

## Source 15: docs/handoff.md: HANDOFF-DELTA-BLOCK section_end:e7-sqexp-gae-p1-p2-result-closure-20260722

### Delta block `section_end:e7-sqexp-gae-p1-p2-result-closure-20260722`

- **D4RL-9 joint-critic GAE P1/P2 closure（`EXT-H-E7-SQEXP-GAE-01`）：**P1 完成 `198/198`，P2 完成 `180/180`；两轮 failed=`0`、terminal audit=`PASS`、seeds 仅 `200,201`、held-out `204--207` 未触碰、NaN/Inf=`0`、rollout failure=`0`。P1/P2 随 `c` 向左总体恢复，最小有限点 `c=0.015625` late mean=`64.8863`，仅比 Positive-only `65.3054` 低 `0.4191`，但 final 仍低 `6.1422`；无已测 finite common `c` 超过 Positive-only，也未识别共同内部最优。medium/replay 有局部收益而 expert 受损；transient best 不得升级为同步 checkpoint、收敛、稳态或正式排名。task collapse 无阈值故 `not_adjudicated`，support/variance 未 instrument，与 NaN/Inf 分报。P1 结果 commit `33fa687352392df985679abddedb834535b10a3d` 的 source `d0ba443154d847065965b18a43ffe897f19530fa` 不可解析；P2 结果 commit `7c1af9fc47ddd347c1bb28d43ad8a024ca95b8a8` 绑定可解析 implementation `909249875c190a75301ceb2dc2c2062ca0efcb16`，其 stale `GAE_STAGE_STATUS.json` 不覆盖真实 `180/180`。本证据仅属 Hopper/D4RL external validity，不替代 C-U1/D-U1；本闭环不合并 stacked code PR，也不登记或启动 P3。见 `docs/results/E7_BENCH_JOINT_GAE_P1_P2_RESULT_2026-07-22.md`。

## Source 16: docs/handoff.md: HANDOFF-DELTA-BLOCK section_end:e7-squared-exp-night-result-pilot

### Delta block `section_end:e7-squared-exp-night-result-pilot`

- **Hopper/Walker squared-remoteness EXP × PPO reference-lifecycle 1M screening （`EXT-H-E7-SQUARED-EXP-NIGHT-01`）：**已完成 126/126 个 1M development branches，terminal audit `PASS`，失败与 NaN/Inf 均为 0，held-out seeds `204--207` 未使用。实验使用论文一致的 `w(d)=w(0)exp[-c(d/2)^2]`，比较 A2C、固定 K4 PPO 与 `K_max=16 + analytic KL(old||new)>0.01` 触发刷新。KL-refresh 相对固定 K4 在 21 个配对 cell 中赢 16 个，800k--1M late mean 平均差 `+3.79`，在 `c={2,4,8}` 的 9 个 cell 中赢 7 个且平均差 `+5.52`，因此登记为正向的有限步均值信号；但 KL 路径平均 seed SD 为 `7.03`，高于 K4 的 `5.87`，不得宣称普遍更稳定。KL 频繁触发使实际 old-policy reference 平均仅维持约 `3.93` 次更新，因此结果支持自适应刷新时机可能有益，不支持固定长 K 必然更优。Stage C GAE 启动 0 个 branch，因 ordered trajectory 与 terminal/truncation contract 尚未验证而保持 blocked。本结果仅为 Hopper/Walker 两 development seeds 的 external-validity screening；固定 1M 不等于收敛或稳态，也不替代 C-U1/D-U1 受控机制证据。运行报告的本地 commit `fbafb44c...` 当时未存在于 GitHub；仓库入库复现其唯一已知 plumbing 修复（trainer variant 改为 `iqlv_exp_rank`），因此状态保持 `pilot`，不得升级为 formal evidence。

## Source 17: docs/handoff.md: HANDOFF-DELTA-BLOCK section_end:e7-w0-highc-actor-screening-pilot

### Delta block `section_end:e7-w0-highc-actor-screening-pilot`

- **Hopper/Walker E7 direct-`w(0)=1` high-`c` × actor-update follow-up （`EXT-H-E7-W0-HIGHC-ACTOR-01`）：**登记为 `not_run` 的 development screening pilot，是 `EXT-H-E7-PPO-W0-EXP-GRID-01` 的非破坏性 successor；前序 186-branch 结果与失败记录完整保留。本实验固定公开参数化 `u=d/2`、`w(d)=w(0)exp(-cu)`，只比较 Positive-only 与 `w(0)=1, c={2,3,4,6,8,12}`，并在 historical canonical A2C 与 PPO clip 两种 actor update 下成对执行。三项任务为 `hopper-medium-expert-v2`、`walker2d-medium-v2`、`walker2d-medium-replay-v2`，development seeds 仅 `200,201`，共 84 个 500k branches；held-out seeds `204--207` 禁止使用。网络、critic、advantage、优化器、batch、学习率、数据与评估协议保持一致；PPO 继续固定 clip `0.2` 和 old-policy cadence `4`。两种 actor update 均记录负样本 standardized distance、实际 `w(d)`、阈值占比与 advantage-weighted effective negative mass，诊断不得改变 objective。该 external-validity pilot 只回答有限步高衰减恢复与 actor-update 敏感性，不得升级为因果识别、收敛、稳态或 PPO/A2C 普遍排名；任务性能退化/崩溃、support/variance boundary 与 NaN/Inf 必须分开终态审计，也不得按数据集事后选择不同 `c` 冒充统一方法。

## Source 18: docs/handoff.md: HANDOFF-DELTA-BLOCK section_end:v56-e6-parent-closure-current-gate

### Delta block `section_end:v56-e6-parent-closure-current-gate`

- **v56 E6 父 claim 关闭覆盖：** E6 的论文核心 claim 现已范围受限关闭；主 long-run 与两个 gap 子实验的原科学状态分别保持 `long_run_validated / finite_step_validated / finite_step_validated`。`D-U1-E6-TAPER-01` 保留为可选非门禁未来工作。当前下一正式 route item 为 `EXT-H-E7-Q2`，registry 状态为 **implemented + ready + active + not_run**；启动后仍须走 canonical hardened guard，且在 raw-complete、终态审计、打包和交付前不得声称 E7 完成。

## Source 19: docs/handoff.md: HANDOFF-DELTA-BLOCK section_end:v57-countdown-offline-bank-current-gate

### Delta block `section_end:v57-countdown-offline-bank-current-gate`

- **Countdown v57 覆盖：** `EXT-C-E8-V4.4-OFFLINE-BANK` 是用户批准的当前离线 focused pilot；V4.3 保留为 fixed-pair predecessor。V4.4 只改变固定负样本覆盖与 current-policy near/far reselection，不引入在线数据刷新。`EXT-H-E7-Q2` 仍是下一正式 route item，`EXT-C-E8-SCALE-01` 继续 blocked。

## Source 20: docs/handoff.md: HANDOFF-DELTA-BLOCK section_end:v59-countdown-offline-bank-tuning-current-gate

### Delta block `section_end:v59-countdown-offline-bank-tuning-current-gate`

- **Countdown v59 覆盖：** `EXT-C-E8-V4.5-OFFLINE-BANK-TUNING` 是当前用户批准的离线 focused successor；V4.4 作为 frozen-bank predecessor 保留。V4.5 只调 calibrated global negative multiplier 与 exponential taper lambda，禁止在线刷新、方向筛选或模型规模同时变化。`EXT-H-E7-Q2` 仍是下一 formal route item，`EXT-C-E8-SCALE-01` 继续 blocked。

## Source 21: docs/handoff.md: HANDOFF-DELTA-BLOCK section_end:v62-countdown-online-offpolicy-current-gate

### Delta block `section_end:v62-countdown-online-offpolicy-current-gate`

- **Countdown v62 覆盖：** `EXT-C-E8-V4.6-ONLINE-OFFPOLICY-REPLAY` 是当前用户批准并已实现的 Countdown focused successor，状态为 **implemented + not_run**。执行前必须提供完整 V4.5 `RUN_COMPLETE.json`/`terminal_audit.json` 及其指向的 V4.4 frozen inputs；runner fail-closed 校验输入与 reference adapter。它可作为独立 pilot 启动，但不改变 `EXT-H-E7-Q2` 的 formal 优先级，也不自动解锁 `EXT-C-E8-SCALE-01`。

## Source 22: docs/handoff.md: HANDOFF-DELTA-BLOCK section_end:v56-e6-parent-closure-execution-order

### Delta block `section_end:v56-e6-parent-closure-execution-order`

13. **v56 执行覆盖：** E6 父 claim 已关闭，`D-U1-E6-TAPER-01` 改为可选非门禁 future study；当前直接进入已实现且 registry 为 ready/active 的 `EXT-H-E7-Q2`（E7-MECH）。E7-Q2 仍为 not_run，必须先完成正式运行、终态审计、打包与交付；其后才允许冻结并实施 `EXT-H-E7-BENCH-01`。E8-MECH/V4.3 与 E8-SCALE 的相对顺序不变。

## Source 23: docs/handoff.md: HANDOFF-DELTA-BLOCK section_end:v57-e8-offline-bank-execution-order

### Delta block `section_end:v57-e8-offline-bank-execution-order`

14. **v57 执行覆盖：** v56 的 formal 顺序不变，`EXT-H-E7-Q2` 仍是下一正式实验。用户批准的 V4.4 作为 single-seed focused pilot 可独立执行，但必须先完成自身 best/terminal audit 与结果交付，才允许讨论 online off-policy successor；不得一次性同时改变 negative-bank 密度和数据在线刷新机制。

## Source 24: docs/handoff.md: HANDOFF-DELTA-BLOCK section_end:v62-countdown-online-offpolicy-execution-order

### Delta block `section_end:v62-countdown-online-offpolicy-execution-order`

18. **v62 Countdown 执行覆盖：** formal 主顺序继续由 v56/v58/v61 控制；`EXT-H-E7-Q2` 优先级不变。V4.6 允许作为独立 guarded pilot 执行，顺序固定为 predecessor/input hash audit -> 四 cell paired training -> 全部训练结束后 test evaluation -> 2×2 paired effect/interaction -> terminal audit -> canonical artifact delivery。任何 online phase 都必须保留 collector manifest、round JSONL、fresh/stale mix 与实际 selected-bank diagnostics；smoke 或单 seed 不得称实验结果。

## Source 25: experiments/registry.yaml: experiments[EXT-H-E7-Q2, EXT-H-E7-BENCH-01]

collection: experiments
entries:
- id: EXT-H-E7-Q2
  execution_gate:
    state: blocked
    blocked_by:
    - completed_formal_execution_no_rerun_without_new_registration
    blocking_reason: The frozen formal v4.3.0 run is complete, scientifically reviewed, compactly archived, and delivered.
      Any rerun requires a separately registered protocol.
  environment: EXT-H
  name: hopper_gaussian_log_scale_quadratic_external_validation
  status: long_run_validated
  scientific_status: long_run_validated
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
    activation_state: blocked
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
  protocol_version: 4.3.0
  pilot_diagnostic:
    formal_scale_pilot_available: true
    scientific_status: pilot_only_not_formal_evidence
    purpose: fixed_budget_redesign_and_mechanism_diagnostic
    latest_uploaded_artifact_sha256: deefbe216ca5c99622c84831b4546da10203610c07736992c51cf23f679f1017
    observed_pipeline_failure:
      critic_gate_stopped_at_step: 7600
      actor_methods_ended_at_unequal_horizons: true
      interpretation: gate_driven_stopping_is_not_formal_longrun_evidence
    mechanism_signal_report_only:
      matched_abs_advantage_far_near_ratio_approx: 0.99992
      standardized_distance_far_near_ratio_approx: 3.596
      full_parameter_gradient_far_near_ratio_approx: 3.47
      signed_task_collapse: 10_of_10
      near_zero_task_collapse: 10_of_10
      far_zero_support_boundary: 0_of_10
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
  one_command_launch: DRPO_HOPPER_MEDIUM_REPLAY=/ABSOLUTE/PERSISTENT/PATH/hopper_medium_replay-v2.hdf5 python3 scripts/run_e7_hopper_q2.py
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
    stopping:
      rule: fixed_optimizer_steps
      formal_steps: 100000
      evaluation_interval: 2000
      early_stop: nan_inf_numerical_failure_only
      gate_driven_early_stop: forbidden
    optimization_stationarity:
      role: diagnostic_only_never_controls_stopping_or_checkpoint_selection
      train_audit_loss: report
      validation_mse_slope: report
      relative_parameter_update: report
      exact_two_times_continuation_when_feasible: report
      validation_gradient: diagnostic_only
      raw_full_parameter_update: diagnostic_only
    checkpoint_selection: best_validation_after_fixed_budget
    frozen_advantage_acceptance:
      operational_execution_gate:
        fixed_budget_completed: required
        finite_selected_metrics: required
      validation_predictive_quality: report_only_superseded_required_gate
      thresholded_validation_predictive_quality: report_only
      selected_vs_final_sign_and_rank_stability: report_only_on_actor_training_split
      test_predictive_metrics: final_report_only
      optimization_terminal_forced_true: forbidden
    superseded_v42_checkpoint_selection: accepted_terminal_extension_else_best_validation
    superseded_v42_quality_gate: thresholded_quality_and_advantage_stability_required
    canonical_artifact_schema: 3
    terminal_audit: required_post_hoc
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
    stopping:
      rule: fixed_optimizer_steps
      positive_only_formal_steps: 100000
      branch_formal_steps: 200000
      actor_evaluation_interval: 5000
      rollout_evaluation_interval: 25000
      intermediate_rollout_episodes: 5
      final_rollout_episodes: 20
      evaluation_seed_pairing_across_methods: required
      early_stop: nan_inf_numerical_failure_only
      equal_branch_horizon: required
      gate_driven_early_stop: forbidden
    positive_only_fixed_budget_before_branching: true
    positive_only_terminal_before_branching: superseded_by_positive_only_fixed_budget_before_branching
    terminal_candidate:
      relative_parameter_update: required
      state_drift_metrics:
      - mean_abs
      - sigma_mean
      - phantom_distance_mean
      normalized_window_drift_max: 0.01
      positive_nll_slope: diagnostic_only
      exact_two_times_continuation_when_feasible: required
    terminal_candidate_status: superseded_as_stopping_gate_retained_for_provenance
    terminal_audit:
      role: post_hoc_classification_only
      fixed_horizon_is_not_convergence: true
      relative_parameter_update: report
      state_drift_metrics:
      - mean_abs
      - sigma_mean
      - phantom_distance_mean
      normalized_window_drift_max: 0.01
      positive_nll_slope: diagnostic_only
      exact_two_times_continuation_when_feasible: finite_terminal_label_only
    superseded_v42_positive_only_terminal_before_branching: true
    superseded_v42_terminal_candidate_controlled_stopping: true
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
    - dynamic_budget_matched_global
    method_semantics:
      positive_only: positive_advantage_only_initialization_reference
      signed: full_signed_advantage_baseline_without_near_far_control
      near_zero: remove_near_negative_updates_keep_far_negative_updates
      far_zero: remove_far_negative_updates_keep_near_negative_updates
      far_cap: cap_far_negative_output_score_influence
      dynamic_budget_matched_global: globally_scale_negative_updates_to_match_far_cap_proxy_each_minibatch
    fixed_budget_comparability:
      positive_only_steps: 100000
      branch_steps_each: 200000
      same_horizon_across_all_branches: required
      terminal_audit_controls_stopping: false
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
    measurable_full_parameter_contribution: required
    log_scale_relative_dominance: diagnostic_only
    control_outcomes:
      diagnostic_score_mitigation: report_separately
      support_boundary_rescue: report_separately
      task_performance_rescue: report_separately
      finite_terminal_rescue: report_separately
    measurable_full_parameter_or_long_run_contribution: superseded_by_measurable_full_parameter_contribution
    targeted_far_control_mitigates_dynamics: superseded_by_separate_control_outcomes
    paired_seed_evidence: required
    terminal_state_audit: required
    identity_only_autograd_check_counts_as_independent_validation: false
  dynamic_global_control:
    method: dynamic_budget_matched_global
    rematch_frequency: every_minibatch
    detached_proxy: sum_abs_advantage_times_joint_output_score
    exact_full_parameter_budget_match_claimed: false
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
    state: delivered
    run_id: formal-20260630T105458Z
    last_heartbeat_utc: null
    process_exit_code: 0
    note: All fixed budgets completed; scientific review accepted the external mechanism claim without authorizing steady-state
      or method-ranking claims.
  evidence:
    code_committed: true
    implementation_tests_passed: true
    run_started: true
    raw_complete: true
    terminal_audited: true
    formal_seeds_expected: 10
    formal_seeds_completed: 10
    fixed_budgets_completed: true
    terminal_audit_records_complete: true
    mechanism_subchecks_passed_seeds: 10
    post_run_scientific_review_completed: true
    package_created: true
    package_filename: e7_q2_formal_v43_EXT-H-E7-Q2_formal.zip
    package_sha256: 5db9c092f2b6e68f42de364f4e85cd3c2691e4dad472b5527a2dce11987f58b5
    delivered_to_user: true
    applied_commit: null
    scientific_status: long_run_validated
    compact_result_path: outputs/e7_hopper_q2
  provenance:
    run_commit: c5c638b47c945f5a3ecb8243f679caa31a129f9e
    repository_closure_base_commit: cd6c42db337d8f261840850a58bf60a83c37e6bd
    run_id: formal-20260630T105458Z
    dataset_sha256: e121c5f7c9857a307baa9edc6a2c3b48e85fedb9ac316ecddd0f48ca7ef4e39b
    runner_version: 4.3.0-fixed-budget-longrun
    origin_main_matched_at_launch: true
    git_dirty_at_launch: false
    provenance_compromised: false
    evaluation_environment: Gymnasium Hopper-v4 compatibility environment with frozen D4RL-v2 reference normalization
  result_summary:
    mechanism:
      abs_advantage_far_near_ratio_mean: 0.9997699141502381
      standardized_distance_far_near_ratio_mean: 3.8445521116256716
      corrected_q_xi_far_near_ratio_mean: 14.546895980834961
      corrected_q_xi_loglog_slope_mean: 2.0000000000188645
      full_parameter_gradient_far_near_ratio_mean: 4.205860980379491
      analytic_autograd_relative_error_max_mean: 6.599849733390784e-08
    terminal_normalized_return_mean:
      positive_only: 32.23070328887839
      signed: 0.9867333552044592
      near_zero: 1.1254711577084202
      far_zero: 22.532331417534287
      far_cap: 11.470778599976397
      dynamic_budget_matched_global: 15.766050787826316
    task_collapse_count:
      positive_only: 0
      signed: 10
      near_zero: 10
      far_zero: 3
      far_cap: 7
      dynamic_budget_matched_global: 6
    support_or_variance_boundary_count:
      positive_only: 9
      signed: 10
      near_zero: 10
      far_zero: 10
      far_cap: 10
      dynamic_budget_matched_global: 10
    nan_inf_count:
      positive_only: 0
      signed: 0
      near_zero: 0
      far_zero: 0
      far_cap: 0
      dynamic_budget_matched_global: 0
    mean_boundary_fraction:
      positive_only: 0.112298583984375
      signed: 0.999725341796875
      near_zero: 0.99990234375
      far_zero: 0.121490478515625
      far_cap: 0.1821044921875
      dynamic_budget_matched_global: 0.189849853515625
    paired_vs_signed:
      near_zero:
        mean_return_difference: 0.13873780250396103
        wins: 8
        wilcoxon_two_sided_p: 0.130859375
      far_zero:
        mean_return_difference: 21.54559806232983
        wins: 10
        wilcoxon_two_sided_p: 0.001953125
      far_cap:
        mean_return_difference: 10.484045244771938
        wins: 10
        wilcoxon_two_sided_p: 0.001953125
      dynamic_budget_matched_global:
        mean_return_difference: 14.779317432621857
        wins: 10
        wilcoxon_two_sided_p: 0.001953125
  closure:
    accepted_claim: Far-field anomalous negative gradients are a major transmission path into support contraction and task-performance
      failure in this Hopper learned-critic setting.
    primary_baseline: signed
    positive_only_role: stable_reference_not_primary_mechanism_baseline
    method_ranking_claim_allowed: false
    finite_terminal_claim_allowed: false
    universal_causality_claim_allowed: false
    continuous_taper_benefit_requires_separate_experiment: true
- id: EXT-H-E7-BENCH-01
  execution_gate:
    state: blocked
    blocked_by:
    - formal_protocol_lock
    blocking_reason: EXT-H-E7-Q2 is scientifically closed and the controlled three-family taper shortlist is now frozen without
      D4RL retuning. The registered two-dataset pilot is ready, but the formal nine-task benchmark remains blocked until exact
      D4RL versions, formal seeds, base algorithm, optimizer, and full budgets are frozen.
  environment: EXT-H
  name: d4rl_mujoco_locomotion_method_benchmark
  status: not_run
  parent_experiment: E7-BENCH
  registration_base_commit: f64452a7452274a183b03c87c39b847039230c00
  claim: Test whether the controller shortlist selected in controlled bandits improves multi-seed normalized return and stability
    on the public D4RL MuJoCo locomotion suite, without per-task method-family retuning.
  role: external_large_scale_continuous_benchmark
  execution_class: formal
  implementation_state: pilot_implemented_formal_parallel_scaffold
  code_entrypoint: src/drpo/e7_bench.py
  one_click_entrypoint: scripts/run_e7_bench.py
  pilot_config_path: configs/e7_bench_pilot.yaml
  pilot_protocol_document: docs/e7_bench_pilot.md
  formal_execution:
    channel_ref: hardened-v1
    activation_state: blocked
    entrypoint_status: implemented
    entrypoint: src/drpo/e7_bench.py
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
  controlled_shortlist_freeze:
    state: satisfied
    source_experiment: C-U1-E4-TAPER-NEAR-RETENTION-01
    source_scope: development_seeds_0_to_4_only
    target_average_near_retention: 0.75
    standardized_distance_reference: 5.0
    coefficients:
      reciprocal_linear: 0.4362580032734791
      reciprocal_quadratic: 0.5520268617673281
      exponential: 0.374162511054291
    global_alpha: 0.75
    d4rl_retuning_allowed: false
  shortlist_rule: freeze_after_E4_E6_core_closure_and_E7_mechanism_without_D4RL_retuning
  pilot_execution:
    execution_class: pilot
    scientific_status: not_run
    execution_gate:
      state: ready
      blocked_by: []
    purpose:
    - verify_two_dataset_loading_and_rollout_compatibility
    - verify_frozen_taper_method_implementation
    - estimate_runtime_and_artifact_volume
    - verify_terminal_audit_and_parallel_resume
    formal_evidence_allowed: false
    method_family_or_coefficient_retuning_allowed: false
    development_seeds:
    - 200
    - 201
    - 202
    - 203
    methods:
    - positive_only
    - signed
    - global_alpha
    - reciprocal_linear
    - reciprocal_quadratic
    - exponential
    datasets:
    - id: hopper-medium-minari-v0
      exact_sha256: 17c1b07d01ce461bb9b2866ad0356c542678b66b886b5ed6c066733a75c52c84
      format: minari_episode_hdf5
      environment: Hopper-v5
      metric: raw_return_only
      formal_nine_task_cell_eligible: false
      reason: uploaded medium file is Minari Hopper-v5 rather than D4RL hopper-medium-v2
    - id: hopper-medium-expert-v2
      exact_sha256: 9d51ad87f8c905be3880d84c6140bcdb7fbf39a19e046a237f238ba34fec9e26
      format: legacy_d4rl_hdf5
      environment: Hopper-v4
      metric: d4rl_v2_normalized_return_percent
      formal_nine_task_cell_eligible: true
    fixed_budget:
      critic_optimizer_steps: 100000
      positive_only_optimizer_steps: 300000
      branch_optimizer_steps: 200000
      shared_positive_only_warmstart_steps: 100000
      method_continuation_steps_each: 200000
      total_actor_optimizer_steps_each: 300000
      positive_only_receives_equal_continuation: true
      legacy_budget_field_semantics:
        positive_only_optimizer_steps: total_actor_horizon_including_shared_warmstart
        branch_optimizer_steps: method_continuation_only
      early_stop_rule: nan_inf_numerical_failure_only
      fixed_horizon_is_not_convergence: true
    parallel_execution:
      scheduler: three_stage_subprocess_worker_pool
      stages:
      - parallel_canonical_critics
      - parallel_shared_positive_warmstarts
      - parallel_equal_horizon_method_continuations
      parallel_unit: dataset_seed_method
      critic_workers: 2
      positive_workers: 8
      warmstart_workers: 8
      branch_workers: 48
      critic_cpus_per_worker: 64
      positive_cpus_per_worker: 32
      warmstart_cpus_per_worker: 32
      branch_cpus_per_worker: 7
      peak_registered_cpu_threads: 336
      server_cpu_capacity: 384
      serial_seed_loop_forbidden: true
      serial_method_loop_forbidden: true
      resume_granularity: dataset_seed_method
      resume_identity_bindings:
      - exact_pilot_config_sha256
      - exact_base_config_sha256
      - runner_and_protocol_versions
      - dataset_sha256
      - stage_budget
      - method_and_taper_parameters
      mismatched_workdir_policy: fail_closed_require_new_workdir
      legacy_parallel_field_semantics:
        positive_workers: alias_of_warmstart_workers
        positive_cpus_per_worker: alias_of_warmstart_cpus_per_worker
  formal_parallel_contract:
    state: registered_required_topology
    task_count: 9
    parallel_unit: task_seed_method
    scheduler: staged_resource_pool_subprocess_workers
    serial_seed_loop_forbidden: true
    serial_method_loop_forbidden: true
    identical_positive_checkpoint_per_task_seed: true
    identical_positive_warmstart_per_task_seed: true
    positive_only_is_equal_horizon_continuation_branch: true
    isolated_output_per_task_seed_method: true
    resume_granularity: task_seed_method
    resume_identity_must_bind_protocol_budget_and_method_parameters: true
    exact_worker_counts: pending_with_formal_seed_freeze
  primary_metrics:
  - normalized_return
  - paired_multiseed_confidence_interval
  - mean_rank_across_nine_tasks
  - worst_seed_return
  - task_performance_collapse_events
  - support_or_variance_boundary_events
  - nan_inf_numerical_events
  protocol_lock_status: long_budget_pilot_frozen_and_implemented_formal_exact_versions_seeds_optimizer_base_algorithm_pending
  evidence:
    code_committed: true
    run_started: false
    pilot_run_started: false
    formal_run_started: false
    raw_complete: false
    terminal_audited: false
    package_created: false
    scientific_status: not_run
  prerequisite_status:
    EXT-H-E7-Q2: satisfied_long_run_validated
    controlled_method_shortlist_freeze: satisfied_without_d4rl_retuning
    pilot_execution: ready_not_run
    formal_protocol_lock: pending
    d4rl_retuning_allowed: false
  canonical_backbone_two_dataset_pilot:
    scientific_status: pilot
    role: external_validity_method_effect_and_late_training_stability
    relationship_to_other_e7_lines:
      frozen_critic_e7_q2_role: preserved_learned_critic_mechanism_diagnostic
      canonical_backbone_role: strong_backbone_performance_and_late_training_stability_pilot
      replaces_e7_q2_mechanism_evidence: false
      replaces_cu1_du1_controlled_identification: false
    baseline_sanity:
      execution_status: delivered_lightweight_result_archive
      formal_evidence_allowed: false
      method: original_exp_rank_mr
      target_class: SNA2C_IQLV_ExpRankAgent
      trainer_variant: iqlv_exp_rank
      alpha: 0.11
      tau: 0.5
      temperature: 5.0
      datasets:
      - hopper-medium-replay-v2
      - hopper-medium-expert-v2
      paired_seeds:
      - 200
      - 201
      - 202
      - 203
      optimizer_steps_per_run: 1000000
      evaluation_interval_steps: 50000
      evaluation_episodes: 10
      expected_runs: 8
      completed_runs: 8
      failed_runs: 0
      aggregate_results:
        hopper_medium_replay_v2:
          final_mean: 73.017
          final_min: 44.294
          final_max: 91.521
          best_mean: 98.854
          last10pct_mean: 69.344
          last10pct_max_mean: 80.62
        hopper_medium_expert_v2:
          final_mean: 80.225
          final_min: 47.445
          final_max: 109.488
          best_mean: 106.145
          last10pct_mean: 70.616
          last10pct_max_mean: 85.503
      terminal_interpretation:
        strong_score_scale_recovered: true
        high_score_reached_by_approximately_400k_to_500k: true
        stable_plateau_by_1m: false
        classification: persistent_or_slow_drift_or_fixed_horizon_inconclusive
        fixed_horizon_is_not_convergence: true
      evaluation_protocol: Gymnasium_Hopper_v4_plus_D4RL_v2_reference_normalization
      exact_legacy_mujoco_py_reproduction: false
      archive:
        filename: EXT-H-E7-BENCH-01_ExpRank_MR_baseline_sanity_1M_4seed_archive.zip
        sha256: 6a53b5e58c220d49ac73bbeb736f0f2d3a83f04566311524b118fc1400467d8d
        checkpoints_intentionally_excluded: true
        result_and_provenance_archive_only: true
      provenance:
        repository_commit_binding_available_in_archive: false
        python_tree_sha256: 83b34aedc4439f4b26cfae7702af4adb5874cd95dbe76b8940e6adab9fd829a3
        agents_sha256: c51d257803ed2636756fb4f61580b1158946c39fec533723a0c7001ef279e1bb
        trainer_sha256: dff1955c655d7786571af76e507c4681f417235d6ac8235527fb8c93343603a9
    next_fixed_shortlist:
      scientific_status: not_run
      implementation_state: dev_branch_implemented_pending_reviewer_liveness_and_main_merge
      development_branch: dev/e7-canonical-shortlist-1m
      draft_pull_request: 4
      launch_commit: pending_post_merge_main_commit
      config_path: configs/e7_canonical_two_dataset_shortlist_1m_v1.json
      protocol_document: docs/e7_canonical_two_dataset_shortlist_1m.md
      launcher: scripts/run_e7_canonical_two_dataset_shortlist_1m.sh
      datasets:
      - hopper-medium-replay-v2
      - hopper-medium-expert-v2
      paired_seeds:
      - 200
      - 201
      - 202
      - 203
      methods:
      - original_exp_rank_mr
      - positive_only
      - global_neg_0p11
      - global_neg_0p011
      - reciprocal_linear_max0p011
      - reciprocal_quadratic_max0p011
      - exponential_max0p011
      expected_branches: 56
      optimizer_steps_per_branch: 1000000
      evaluation_interval_steps: 50000
      evaluation_episodes: 10
      primary_late_window_steps:
      - 750000
      - 800000
      - 850000
      - 900000
      - 950000
      - 1000000
      parallel_execution:
        scheduler: resumable_branch_subprocess_worker_pool
        parallel_unit: dataset_seed_method
        default_max_workers: 40
        omp_threads_per_worker: 2
        serial_seed_loop_forbidden: true
        serial_method_loop_forbidden: true
        resume_granularity: dataset_seed_method
      taper_coordinate:
        distance: detached_RMS_standardized_action_distance
        reference_distance: 2.0
        distinct_from_frozen_critic_e7_reference_distance_5: true
        coefficient_source: existing_canonical_two_dataset_adapter_configuration
        new_d4rl_specific_retuning: false
        reciprocal_linear_coefficient: 0.4362580032734791
        reciprocal_quadratic_coefficient: 0.5520268617673281
        exponential_coefficient: 0.374162511054291
      comparison_rules:
        primary_metric: late_window_mean_750k_to_1m
        best_score_is_diagnostic_only: true
        report_best_to_final_drop: true
        report_best_to_late_mean_drop: true
        report_terminal_slope: true
        terminal_audit_required_for_ranking_or_steady_state_claim: true
      reporting_separation:
      - task_performance_collapse
      - support_or_variance_boundary_event
      - nan_inf_numerical_failure
      formal_nine_task_benchmark_unlocked: false
      formal_evidence_allowed: false
      execution_gate:
        state: blocked
        blocked_by:
        - independent_reviewer_gate
        - short_liveness_gate
        - merge_to_clean_main_commit
