# Hopper learned-critic external validation E7

> Stage 4B lossless source-promotion shadow candidate.
> Generated from `docs/handoff.md`; do not edit while the manual handoff remains authoritative.

- Owner type: `canonical_module`
- Owner ID: `hopper_e7`
- Responsibility: Cover learned-critic far-field mechanism validation and D4RL method-effect evidence while preserving the external-validity boundary.
- Dependencies: `global_core_governance`, `execution_status_gates`, `theory_methods_related_work`, `terminal_audit`, `continuous_mechanism_e1_e3`
- Content-contract topics: none
- Owned source blocks: 13
- Registry references are pointers only; `experiments/registry.yaml` remains the sole editable registry source.

## Registry references

- `experiments`: `EXT-H-E7-Q2`, `EXT-H-E7-BENCH-01`
- `development_experiment_registrations`: none

## Owned source blocks

<!-- STAGE4B-SOURCE-BLOCK:B000006:START -->
<!-- HANDOFF-DELTA-BLOCK:after_heading:v54-e7-canonical-critic-rollout-audit:START -->
> **v54 增量登记：`EXT-H-E7-Q2` canonical critic、rollout preflight 与 audit 语义修复（不删除 v53 及更早内容）**
>
> - 用户上传的首轮 E7-Q2 单 seed、100-step 运行只保留为工程 pilot：它验证了数据、梯度 probe、干预分支和结果打包链路，但 critic、Positive-only 与方法分支均未达到正式终态，且 normalized-return rollout 不可用，因此不得进入论文正式结果或升级科学状态。
> - 修复 critic 隔离：旧实现会在每个 actor seed 内重新训练 critic，跨 seed 波动仍混入 critic 差异。v54 改为每个 run 只训练或严格复用一个 canonical critic artifact；episode split、observation/return normalizer、terminal critic checkpoint 与 frozen advantage 对全部 actor seeds 和方法完全共享。Formal 只接受通过优化终态与 2× continuation 的 terminal extension checkpoint；best-validation checkpoint 仅作诊断，不再用于 actor advantage。
> - 修复 rollout 可观测性与一键门禁：训练前必须完成 D4RL 注册、`gym.make`、reset、真实 step、随机完整 episode 和 `get_normalized_score` 检查；pilot 与 formal 均 fail closed。失败时先落盘 package versions、兼容 shim、失败阶段、exception 与完整 traceback，再由 hardened guard 打包，避免再次只得到 `rollout_unavailable=1`。
> - 修复任务性能语义：normalized return 未观测时，`task_performance_status` 必须为 `unavailable/not_evaluated/disabled`，`task_performance_collapse=null`；不得把“没有观测”写成 `false`。任务性能崩溃、支持/方差边界和 NaN/Inf 继续分开报告。
> - 修复总门禁命名：根审计分开输出 `engineering_pipeline_complete`、`mechanism_subchecks_passed_for_completed_seeds`、`paired_seed_evidence_complete`、`formal_evidence_prerequisites_complete` 与 `formal_scientific_gate_passed`。Pilot 即使工程与子检查通过，formal gate 也必须为 false；历史 `independent_validation_gate_all_seeds` 仅保留兼容别名且 pilot 固定为 false。
> - `EXT-H-E7-Q2` 的 formal 科学门禁、方法、正式 seeds、阈值与执行顺序不变，仍保持 **not_run + implemented + blocked**；本更新只修复实现隔离、环境交互可观测性和审计语义，不构成正式实验启动或结果升级。
<!-- HANDOFF-DELTA-BLOCK:after_heading:v54-e7-canonical-critic-rollout-audit:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000006:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000009:START -->
<!-- HANDOFF-DELTA-BLOCK:after_heading:v57-ext-c-e8-v44-offline-negative-bank:START -->
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
<!-- HANDOFF-DELTA-BLOCK:after_heading:v57-ext-c-e8-v44-offline-negative-bank:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000009:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000010:START -->
<!-- HANDOFF-DELTA-BLOCK:after_heading:v58-e7-gymnasium-v4-rollout:START -->
> **v58 增量登记：`EXT-H-E7-Q2` Gymnasium `Hopper-v4` rollout 兼容修复（不删除 v57 及更早内容）**
>
> - 离线训练数据仍是 `hopper-medium-replay-v2` 的 HDF5 文件，critic、frozen advantage、actor、方法组、正式 seeds、训练 horizon、收敛阈值和 E7 科学职责全部不变；本版只修复真实环境交互的执行后端与 provenance。
> - rollout 评估固定使用服务器本地 Gymnasium `Hopper-v4` 与新版 `mujoco` binding。数据集身份和模拟器环境版本明确分离：不得把 `Hopper-v4` 称为 v4 数据，也不得把该分数表述为逐位复现 legacy `mujoco-py` 环境。
> - normalized return 不再依赖 D4RL 环境对象的 `get_normalized_score()`，而按冻结的 D4RL-v2 Hopper medium-replay reference `min=-20.272305`、`max=3234.3` 手动计算百分制分数；结果必须同时保存 raw return、reference 常量、离线 dataset ID 与 evaluation env ID。
> - legacy D4RL/mujoco-py fallback 明确禁止。主 runner 不导入 `d4rl` 或 `mujoco_py`；环境 preflight 在独立子进程中执行 reset、真实 step、随机 episode 与 reference normalization。若底层 native 进程收到 SIGSEGV、超时或 Python exception，父进程必须落盘退出码、signal、stdout/stderr 与错误报告并在 critic 训练前 fail closed。
> - 正式报告中的准确口径为“offline training on D4RL Hopper medium-replay-v2, evaluated in the Gymnasium Hopper-v4 compatibility environment with D4RL-v2 reference normalization”。该兼容评估可用于 E7 内部 paired mechanism comparison，但不得冒充 exact legacy D4RL leaderboard reproduction。
> - `EXT-H-E7-Q2` 科学状态继续保持 **not_run**。静态检查、单元测试和本地无 MuJoCo 的 mock preflight 只证明实现，不构成 Hopper pilot 或正式结果；下一步仍须在服务器由一键 runner 先通过真实 Gymnasium/MuJoCo preflight。
<!-- HANDOFF-DELTA-BLOCK:after_heading:v58-e7-gymnasium-v4-rollout:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000010:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000011:START -->
<!-- HANDOFF-DELTA-BLOCK:after_heading:v59-ext-c-e8-v45-offline-bank-tuning:START -->
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
<!-- HANDOFF-DELTA-BLOCK:after_heading:v59-ext-c-e8-v45-offline-bank-tuning:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000011:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000014:START -->
<!-- HANDOFF-DELTA-BLOCK:after_heading:v62-ext-c-e8-v46-online-offpolicy-replay:START -->
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
<!-- HANDOFF-DELTA-BLOCK:after_heading:v62-ext-c-e8-v46-online-offpolicy-replay:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000014:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000015:START -->
<!-- HANDOFF-DELTA-BLOCK:after_heading:v64-e7-q2-acceptance-pipeline:START -->
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
<!-- HANDOFF-DELTA-BLOCK:after_heading:v64-e7-q2-acceptance-pipeline:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000015:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000287:START -->
# 15. Learned-Critic External Mechanism Validation on D4RL



这次实验已经补上此前 D4RL 分析最重要的缺口：advantage 不再由人工轨迹标签直接指定，而是由真实训练出的 value critic 产生；actor 使用 detached TD residual 进行重复的 signed off-policy 更新。

正式配置先在开发 seed 42 上冻结，随后使用未参与选择的 seeds 100--109。Critic 在 held-out episode 上的平均 R² 为 **0.428**，Pearson 相关为 **0.656**。它并不完美，反而说明结论能承受现实的 critic noise。

当前可形成的严谨结论是：

> 在 Hopper medium-replay 的自然数据中，匹配负 advantage 幅度后，far negative 仍具有更大的 policy score 与全参数梯度；重复 signed actor 更新会造成均值向 tanh 边界饱和并收缩策略支持。删除 near negatives 不能消除该失稳，而只删除 current far negatives 可以稳定救援。

这是一项**外部机制验证**，不是 D4RL normalized-return 方法表。

<!-- STAGE4B-SOURCE-BLOCK:B000287:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000288:START -->
## 1. learned advantage 下的来源隔离

在 Positive-only actor 拟合过程中：

| 阶段 | |A| far/near | 标准化距离 far/near | Gaussian score far/near | 全参数梯度 far/near | 聚合梯度 far/near |
|---|---:|---:|---:|---:|---:|
| Step 0 | 1.000 | 3.659 | 1.908 | 2.210 | 3.174 |
| Step 600 | 1.001 | 7.363 | 3.629 | 2.107 | 2.615 |

因此，大梯度不是因为 far 样本具有更大的 negative advantage。Positive-only 拟合让固定坏动作相对当前策略进一步远场化，标准化距离约翻倍，score 放大随之增强。

<!-- STAGE4B-SOURCE-BLOCK:B000288:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000289:START -->
## 2. 方差方向与修正理论一致

在所有正式 seeds 和所有记录 checkpoint 中，匹配 far negatives 对 `log sigma` 的 signed ascent direction 都为负，即：

\[
A<0,\quad \|z\|>1 \Longrightarrow \Delta\log\sigma<0.
\]

所以 far negative 的实际作用是**均值排斥 + 方差/支持收缩**，而不是旧稿中的 “mu 与 sigma 同时扩张”。Near negatives 在初期通常推动 sigma 扩张；随着策略移动，一部分 near negatives 跨过标准化距离边界后也转为收缩。

<!-- STAGE4B-SOURCE-BLOCK:B000289:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000290:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000290:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000291:START -->
## 4. 相变而非 sign-only 法则

开发 seed 的 alpha 扫描显示：负梯度系数从 0.5 增至 1.0 时，均值饱和和正样本 NLL 出现明显恶化，之后继续走向边界。因此失稳取决于正负 signed field 的净平衡，而不是“只要 A<0 就必然联合发散”。这直接支持 v9 的 signed-field / moment-domain 理论，否定旧版 sign-only SPD 论证。

<!-- STAGE4B-SOURCE-BLOCK:B000291:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000292:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000292:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000293:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000293:END -->
