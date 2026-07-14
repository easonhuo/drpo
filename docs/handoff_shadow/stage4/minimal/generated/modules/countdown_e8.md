# Countdown Transformer external validation E8

> Generated Stage 4 minimal-context shadow module. Do not edit manually.
> `docs/handoff.md` and `experiments/registry.yaml` remain authoritative.

- Module ID: `countdown_e8`
- Responsibility: Cover token-level near or far mechanism probes and fixed-offline-bank method pilots without replacing D-U1 controlled identification.
- Content contract topics: none
- Deduplicated overlapping source chunks: 0
- Source hash: `04857296433e3c7878bbad7c6decb5753a6c31d5c1ced41b8ba0c78435200aff`

## Source 1: docs/handoff.md: HANDOFF-DELTA-BLOCK after_heading:v52-ext-c-e8-v43-dynamic-control

### Delta block `after_heading:v52-ext-c-e8-v43-dynamic-control`

> **v52 增量登记：Countdown `EXT-C-E8-V4.3` 动态负样本 remoteness 控制（不删除 v51 及更早内容）**
>
> **v51（D-U1 E6 条件缺口闭环与最小改动正式协议版）历史标题与全部内容继续保留。**
>
> - 用户确认：matched near/far pair 继续用于瞬时机制研究，但 near/far 是相对当前策略变化的状态，不能把数据构造时的身份永久用于长期训练控制。V4.2 的静态 `controlled_negative` 只 taper 初始 far 分支；初始 near 在训练中进入 far 区后仍保持未衰减，因而没有真正测试“控制所有当前远场负样本”。
> - 新执行 ID 为 `EXT-C-E8-V4.3`，状态为 **尚未运行（not_run）**，科学职责为外部有效性的 focused method diagnostic。V4.2 完整保留为 provenance 并登记 superseded；它的 matched-pair mechanism probe 设计不被否定。
> - V4.3 新增 `dynamic_controlled_negative`：对初始 near 与初始 far 两个负分支都按当前模型 token surprisal 使用同一 detached exponential taper。旧 `controlled_negative` 原样保留为 static-label ablation；同时比较 `positive_only` 与 `uncontrolled_negative`。本轮不扩充 negative bank，不改数据规模、seed、学习率、训练 horizon、taper lambda、threshold、LoRA 配置或共同负梯度尺度。
> - 0.5B focused pilot 在 reference greedy `>=0.08` 且 valid `>=0.95` 时允许执行四方法，以检验实现修复和初步效果；`15%/95%` 继续作为任何正式方法排名的 floor-effect 门禁。低于 15% 时必须明确标注 single-seed pilot，不得形成方法排名、scale-up 结论或论文方法胜负。
> - 主要结果仍是 verifier success、pass@k、valid rate、held-out canonical pattern-family coverage/precision 以及 best/terminal 终态。当前 near/far 权重轨迹只是实现证据，用于确认实际运行的是动态控制，不是主要科学结果；即使权重按预期变化但任务指标不改善，方法仍判定无效。
> - 任务性能退化、support/structure boundary 与 NaN/Inf 数值失败继续分开报告。Countdown 仍不替代 D-U1/D-Diag 的受控因果识别，也不得称 state-distribution OOD generalization。
> - 本更新重基于 `main` commit `0907c3c0e76fc836c2bf2b752abf554c17f79f22`，保留 v51 的 D-U1 E6 条件缺口正式协议与治理 Stage 3 shadow mode；未运行真实 Qwen/CUDA 实验。

## Source 2: docs/handoff.md: HANDOFF-DELTA-BLOCK after_heading:v53-stage3-observation-automation

### Delta block `after_heading:v53-stage3-observation-automation`

> **v53 增量登记：治理 Pipeline Stage 3 真实观察记账、报告持久化与 Full Acceptance 自动触发（不删除 v52 及更早内容）**
>
> - 本版只优化 `GOV-HANDOFF-INDEX-01` 的 Stage 3 shadow 工程，不改变任何 C-U1、D-U1、Hopper 或 Countdown 的科学变量、seeds、阈值、实验职责、结果状态和执行顺序。人工 `docs/handoff.md` 继续是唯一权威 Master，authority cutover 仍禁止。
> - `DU1-E6-SEMANTIC-GAP-FORMAL-2026-06-27` 与 `EXT-C-E8-V4.3-DYNAMIC-CONTROL-2026-06-27` 分别登记为第一、第二个真实 shadow observation；其最终 repository commit 由 Git 历史派生。旧报告中的 `head_commit` 只解释为临时 validation-worktree head，不再被误称为最终 repository commit。
> - observation ledger 不采用容易漂移的人工计数文件，而由不可变 `HANDOFF_DELTA.yaml`、sibling `SHADOW_REPORT.json` 与 Git 历史动态推导。`corpus-check` 会在各 observation 的历史 repository after-image 上重放 delta、复核 stored report，并输出 bootstrap/real 数量、最终 repository commit 与性能统计。
> - 每个新 delta 必须保存 sibling `SHADOW_REPORT.json`；`auto-check` 会重新执行确定性 replay，并对 stored report 的科学/治理语义字段逐项比对。运行耗时和 validation-worktree commit 不参与语义等价判断；缺失或 stale report 直接 fail closed。V4.3 的历史 schema-v1 delta 在本版补写机器派生报告并加入只读兼容 allowlist。
> - 新 delta schema 升级到 v2；bootstrap、E6 observation 与 V4.3 observation 继续通过显式 legacy allowlist 使用 v1。v2 对 registry 采用完整 change coverage：新增 experiment、状态机 transition 和其他字段变化均须逐项登记并绑定 evidence；未声明变更、虚假断言和 experiment 删除均拒绝。
> - 累计 20 次成功相关更新与“距上次 Full 已满 7 天且期间存在未覆盖更新”两项触发条件改为机器计算。普通 Fast Gate 在 Full overdue 时阻塞；Full 报告必须持久化覆盖的 update IDs，完成后才重置计数/时间窗口。无相关更新时不会因日历经过而空跑。
> - Fast Gate 只完整重放本次被修改的 delta/report，历史 observation 只扫描不可变元数据；全量历史 replay 保留给 `corpus-check` 与 Full Acceptance，避免日常提交成本随历史长度线性增长。v2 Full 报告还会校验 covered IDs、计数、fingerprint、命令返回码/超时与 corpus audit，防止伪造 coverage 跳过重验收。
> - 本次属于 schema/renderer/acceptance architecture 变化，必须执行 Full Acceptance，并保存 corpus replay、mutation/conflict/idempotence 测试和 coverage report。本更新应用后构成第三个真实 shadow observation，但不等于 Stage 3 已完成验收；后续仍需真实多 session 独立/冲突更新和更多 operation coverage。

## Source 3: docs/handoff.md: HANDOFF-DELTA-BLOCK after_heading:v57-ext-c-e8-v44-offline-negative-bank

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

## Source 4: docs/handoff.md: HANDOFF-DELTA-BLOCK after_heading:v59-ext-c-e8-v45-offline-bank-tuning

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

## Source 5: docs/handoff.md: HANDOFF-DELTA-BLOCK after_heading:v62-ext-c-e8-v46-online-offpolicy-replay

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

## Source 6: docs/handoff.md: HANDOFF-DELTA-BLOCK after_heading:v67-countdown-0p5b-mechanism-close-e8-taper

### Delta block `after_heading:v67-countdown-0p5b-mechanism-close-e8-taper`

> **v67 增量登记：关闭 Countdown 0.5B 机制探索职责并登记 `EXT-C-E8-TAPER-0.5B-01`（不删除 v66 及更早内容）**
>
> **历史标题保留：v66（E4-TAPER Budget-Match 正式结果闭环版）。**
>
> - **职责关闭而非结果升级。** Countdown/Qwen 0.5B 的机制探索阶段在当前范围内关闭；这一决定只表示现有外部证据已经足以承担 Transformer 共享参数下的机制迁移说明，不把任何 smoke、单 seed、有限步 pilot 或正在运行的 V4.6 自动升级为正式多 seed 结果。C-U1/D-U1 继续承担受控因果识别与 ground truth，Countdown 不替代内部机制实验。
> - **关闭范围。** 现有证据支持以下范围受限观察：learner-relative surprisal 较高的错误 completion 往往具有更大的 raw negative influence 与共享参数 collateral effect；固定 near/far completion 会随 learner 更新而 stale；current-policy remoteness control 可以显著削弱当前远端影响；uncontrolled negative 可能造成任务性能和 valid/support 退化而不必伴随 NaN/Inf。Near/Far 仅是端点诊断工具，不是最终算法对负样本的天然二分类。
> - **禁止过度解释。** 本关闭决定不声明某个连续函数普适最优，不声明 Online Dynamic 已稳定超过 Positive-only，不把 0.5B 结果外推为 3B/7B 规模结论，也不把 task-performance degradation、support/structure boundary 与 NaN/Inf numerical failure 混为一类。
> - **V4.6 保持独立。** `EXT-C-E8-V4.6-ONLINE-OFFPOLICY-REPLAY` 继续作为 online refresh × negative update 的独立 2×2 效果 pilot，保留其既有状态和 provenance；其结果无论是否超过 Positive-only，都不得反向改写已经完成的内部机制结论，也不作为本次机制职责关闭的必要门禁。
> - **新增实验。** 登记 `EXT-C-E8-TAPER-0.5B-01`，只比较同一公共 negative replay、同一 0.5B reference adapter、同一 optimizer/update budget 下，不同连续 surprisal taper 如何分配负梯度预算，以及它们是否带来任务收益或稳定性差异。该实验是外部方法 pilot，不是新的基础机制实验；当前状态为 **not_run + not_implemented + blocked**，在 runner、公共 replay、校准输出和终态审计全部实现并冻结前禁止启动。
> - **公共 replay 与连续距离。** 固定 reference collector 对相同 Countdown prompts 生成候选，保留全部格式合法、数字使用正确、verifier 判错且 prompt 内唯一的 completion，形成 sample-level replay pool；不再要求每个 prompt 恰好具有 2、4 或 16 个负样本。训练先均匀采样 prompt，再从该 prompt 的候选池采样 completion，避免候选丰富 prompt 获得更大权重。每次更新由当前 learner 重新计算 `d_theta=max(0,-log pi_theta(x|s)-tau)`，不得沿用永久 near/far 标签。
> - **冻结方法组。** 主比较固定为 Positive-only、Uncontrolled negative、Global matched、Reciprocal-linear、Exponential 和 Squared-distance exponential。不得预设 Exp、Squared-Exp 或 Global 必然更优；Reciprocal-quadratic 若后续需要，只能作为单独登记的附录候选，不能在看到 confirmation/test 后追加。
> - **公平校准。** 在独立 calibration split 与 development seed `9134` 上，将 Global 与各 taper 的初始化 raw negative-gradient L2 匹配到同一冻结目标预算；校准完成后冻结全部系数。paired confirmation seeds 固定为 `9234/10234/11234`，test split 只在训练和方法选择全部完成后访问，禁止依据 confirmation/test 继续调函数或系数。
> - **指标与终态。** 同时报告 greedy verifier success、pass@8、valid rate、best 与 terminal checkpoint；按当前 surprisal 分位桶报告 raw/weighted negative-gradient norm、实际权重、positive-negative cosine、correct-completion collateral effect 及各桶预算贡献。任务效果退化、valid/support/entropy boundary 和 NaN/Inf 必须分开审计；不得只用 best checkpoint 宣称胜出。
> - **解释规则。** 若选择性 taper 超过 Positive-only 且 terminal 不反转、valid/support 不恶化，则支持 0.5B 上的额外负信号价值；若只优于 Uncontrolled/Global 而不超过 Positive-only，则只支持远场伤害控制；若方法接近，则按简单性与理论尾部性质冻结 3B 候选；若全部负梯度方法更差，则关闭 0.5B 方法收益路线，不继续无界 HPO。
> - **规模路线。** `EXT-C-E8-SCALE-01` 的方法 shortlist 由本实验冻结；3B 主模型与 7B frozen confirmation 仍是独立规模验证，当前不因 0.5B 机制职责关闭而自动解锁。

## Source 7: docs/handoff.md: HANDOFF-DELTA-BLOCK after_heading:v79-e8-active-tail-repair

### Delta block `after_heading:v79-e8-active-tail-repair`

> **v79 增量登记：Countdown E8-TAPER active-tail calibration 与诊断显存修复（不删除 v77 及更早内容）**
>
> - **旧问题：**0.5B 自然 replay 的独立 calibration split 在旧 `tau=2.0` 下几乎全部 `distance=0`，导致 inherited exponential target 与 uncontrolled negative aggregate L2 相同；`global_matched` 被校准为 1，`reciprocal_linear` 与 `squared_distance_exponential` 被校准为 0，从而与 uncontrolled 逐点重合。
> - **协议修复：**`EXT-C-E8-TAPER-0.5B-01` 保留同一 experiment ID、方法集合、训练 seeds、900/16 自然 replay、1200 update budget 与 synthetic-negative policy，但 calibration tau 改为由独立 calibration split 的 common-half median surprisal 解析，并登记 active-distance fraction、target/uncontrolled ratio 与 nondegenerate fail-closed guard。该修复不使用 validation/test 或确认 seeds 选参。
> - **实现修复：**正式收回 `_collate_pairs()` 误传 `batch_size` 的一行 hotfix；`surprisal_bin_diagnostics` 改为按小 batch 串流 full-vocab completion stats 与梯度诊断；诊断 OOM 时保留 metrics、training log 与 checkpoints，并在 manifest/terminal audit 中标记 `incomplete_oom`，不得混同为 NaN/Inf 数值崩溃。

> - **状态边界：**本次是实现与协议修复，不是科学结果；已有 dirty/local sanity 与 failed pilot 只作为 diagnostic evidence，不能入 formal result。修复应用后必须先重新执行短预算 sanity，确认方法不再 byte-identical，再运行登记的 0.5B pilot。

## Source 8: docs/handoff.md: HANDOFF-DELTA-BLOCK after_heading:v73-e8-taper-corrected

### Delta block `after_heading:v73-e8-taper-corrected`

> **v73 增量登记：Countdown E8-TAPER 距离坐标与执行链修正（不删除 v72 及更早内容）**
>
> - **旧定义—问题—修正：**v67 将 `max(0, sequence_surprisal - tau)` 直接记作距离 `d`。E6-TAPER 的方法命名审计已经确认 surprisal 对应距离平方量级；沿用旧定义会把 reciprocal-linear 实际做成 quadratic-distance，把 squared-distance exponential 实际做成 quartic-distance。E8 现统一定义 `S=max(0,(sequence_surprisal-tau)/c_cal)`、`d=sqrt(S)`，并只在负样本分支使用 detached 权重。
> - **冻结尺度：**保留已登记 `tau=2.0`。`c_cal` 只由独立 calibration replay 和 seed `9134` 在 reference initialization 计算：将校准 surprisal 排序为 lower/upper rarity halves，取两半中位数之差；该尺度在 confirmation 前冻结，若非有限或小于 `1e-6` 则 fail closed。不得用 validation/test 或确认 seeds 重调。
> - **方法公式：**`reciprocal_linear=1/(1+lambda d)`；`exponential=exp(-lambda d)`；`squared_distance_exponential=exp(-lambda d^2)=exp(-lambda S)`。`global_matched` 仍是不区分距离的常数控制；`uncontrolled_negative` 表示在所有方法共享的 frozen negative base scale 下不施加 taper，不等于原始系数恒为 1 的跨协议比较。
> - **确定性与身份修复：**calibration gradient measurement、learner-relative coordinate 和 teacher-forced audit 均关闭 dropout。训练使用 eval/no-grad 的 deterministic coordinate pass 与 train-mode gradient pass 分离。config、reference adapter、replay、sampler seed/plan hash 和 experiment ID 全部执行 fail-closed 身份校验。
> - **梯度预算口径修复：**共享负尺度的实际定义是 positive aggregate gradient L2 除以 uncontrolled-negative aggregate gradient L2，不再误称 per-sample RMS。Global 与 taper 的 initialization matching 继续比较 aggregate raw negative-gradient L2；Adam update 不宣称匹配。
> - **状态边界：**本次只完成实现和门禁修复，未运行 Qwen/CUDA pilot，未产生任何方法排名。`EXT-C-E8-TAPER-0.5B-01` 状态为 `not_run + implemented + ready`；任务性能退化、valid/support/entropy boundary 和 NaN/Inf 仍必须分开报告，fixed 1200-update horizon 不自动称为收敛。

## Source 9: docs/handoff.md: HANDOFF-DELTA-BLOCK section_end:e8-base-rl-replay-0p5b-gate

### Delta block `section_end:e8-base-rl-replay-0p5b-gate`

- **Countdown E8 base-start RL/replay 0.5B pilot：**登记 `EXT-C-E8-BASE-RL-REPLAY-0.5B-01`，作为移除 Countdown SFT warmstart 后的基模起点诊断。该实验只回答：Qwen pretrained base 是否能通过 oracle-offline fixed positive corpus 学起；base-specific calibrated offline negatives 是否能超过 positive-only；online on-policy self-sampled positives 是否能冷启动；dynamic replay buffer 累积历史自采 positives/negatives 是否优于 immediate on-policy 更新。所有 RL 分支从 Qwen pretrained base + fresh LoRA 开始，禁止 Countdown SFT warmstart、随机初始化主实验、taper 方法族和正式方法排名声明。固定预算 pilot 只报告有限步 evidence；结果必须分开报告 task performance、online signal sparsity/replay support、valid structure boundary 和 NaN/Inf numerical failure。

## Source 10: docs/handoff.md: HANDOFF-DELTA-BLOCK section_end:e8-lowsft-rft-dirty-pilot-record-20260708

### Delta block `section_end:e8-lowsft-rft-dirty-pilot-record-20260708`

- **Countdown E8 low-SFT / capacity diagnostic dirty pilots：**本线记录 `EXT-C-E8-ONPOLICY-CAPACITY-DIAG-0.5B-01` 与一次性 `EXT-C-E8-LOWSFT-RFT-0.5B-01` 试错结果，只能作为 single-seed pilot evidence，不得升级为正式多 seed 结论或方法排名。capacity diagnostic 的 single seed `2026070701` 显示 `same_lora_rft`、`fresh_lora_rft`、`full_param_rft` 的 `best_attempt` 均为 0；terminal 端总体表现为 greedy 持平或小升、pass@8/pass@64 下降，说明 naive verifier-correct positive-only on-policy RFT 没有超过 LoRA SFT 起点。low-SFT 试错从按 validation greedy≈0.08 选出的 epoch-3 LoRA SFT checkpoint 起跑；该 checkpoint 的 pass@8 实际已接近 full-SFT（不是 pass@8≈0.08 起点），RFT 后 `best_attempt=0`，terminal test greedy 0.100→0.113、pass@8 0.174→0.133、pass@64 0.265→0.149。解释必须保留以下限制：运行源码为 dirty pilot / one-off orchestration；不是 convergence；没有证明 3B 或更强模型无效；尚需 no-update、parameter-delta、probe-loss、Qwen pretrained-base no-SFT、ultra-low pass@8 checkpoint 与 offline fixed-corpus controls。工程上允许把 `cmd_sft --save_every_epoch` 作为 opt-in 本地 checkpoint 功能合入，以便后续选择更细粒度 ultra-low SFT 起点；模型权重与结果包不得进入 Git 更新包。

## Source 11: docs/handoff.md: HANDOFF-DELTA-BLOCK section_end:e8-onpolicy-capacity-diag-0p5b-gate

### Delta block `section_end:e8-onpolicy-capacity-diag-0p5b-gate`

- **Countdown E8 on-policy capacity diagnostic 0.5B pilot：**登记 `EXT-C-E8-ONPOLICY-CAPACITY-DIAG-0.5B-01`，作为 `EXT-C-E8-ONPOLICY-UNPOLISHED-0.5B-01` 之后的第二层诊断，只回答 same-LoRA continuation 退化是否来自 same-adapter drift、LoRA RFT 容量、LoRA SFT 容量或 on-policy 探索/样本多样性不足。第一轮只跑单 seed 的 `same_lora_rft / fresh_lora_rft / full_param_rft / full_param_sft_only` 分支并行诊断；单 seed 内部 on-policy attempts 仍保持串行。所有 RFT 分支仍为 verifier-correct positive-only，不包含 signed negative、taper 方法族或 frozen off-policy replay。固定 sampling attempts 只能报告 finite-budget pilot evidence，不得宣称收敛或方法排名；full-param 分支只作 capacity diagnostic，不替代 E8-TAPER 方法实验。

## Source 12: docs/handoff.md: HANDOFF-DELTA-BLOCK section_end:e8-onpolicy-unpolished-0p5b-gate

### Delta block `section_end:e8-onpolicy-unpolished-0p5b-gate`

- **Countdown E8 on-policy unpolished 0.5B pilot：**登记 `EXT-C-E8-ONPOLICY-UNPOLISHED-0.5B-01`，仅作为 0.5B + same-LoRA continuation 是否能从当前 policy 自采样 verifier-correct completion 继续学习的排除项诊断。第一版只允许 `sft_only` 与 `onpolicy_rft_positive_only`，不包含 full-param、fresh-LoRA、signed negative、taper 方法族或 frozen off-policy replay；数据 split 使用当前 Countdown structural family-holdout 协议。SFT 可通过显式路径复用已训练 LoRA adapter，但必须记录 provenance，且不改变 same-LoRA continuation 口径；固定 sampling attempts 只能报告 finite-budget pilot evidence，不得宣称收敛或方法排名。

## Source 13: docs/handoff.md: HANDOFF-DELTA-BLOCK section_end:e8-oracle-offline-bank-v2-0p5b-gate

### Delta block `section_end:e8-oracle-offline-bank-v2-0p5b-gate`

- **Countdown E8 oracle-offline bank v2 0.5B protocol:** 登记
  `EXT-C-E8-ORACLE-OFFLINE-BANK-V2-0.5B-01`，只定义和实现模型无关的 canonical
  oracle-offline corpus，不运行训练、不产生方法排名。该 corpus 固定 structural split，positive
  由 oracle solution 锚定，negative 按 detail / near-value / mid-value / far-value wrong
  分层构造，并按 easy / medium / hard 难度输出覆盖审计。样本 correctness/quality 与
  value distance、structure distance、后续 policy surprisal 分开记录；不得把远样本自动解释为差样本，或把近样本自动解释为好样本。base、low-SFT、full-SFT
  只能在同一个 corpus 上做 downstream scoring / calibration，不得为不同初始化重建 corpus。

## Source 14: docs/handoff.md: HANDOFF-DELTA-BLOCK section_end:e8-oracle-offline-v2-init-matrix-pilot-result

### Delta block `section_end:e8-oracle-offline-v2-init-matrix-pilot-result`

- **Countdown E8 oracle-offline v2 init-matrix pilot result:** 注册并归档 `EXT-C-E8-ORACLE-OFFLINE-V2-INIT-MATRIX-0.5B-01`，执行绑定 dev commit `fe214f010bd5fec1e0e6a83f8297132a9ae8882b` 且 `git_dirty=true`；结果包 SHA-256 为 `b0a05d54e531661bb15bd0dcc3f8f06354554513056c2cf7adeea71a919f59f6`。固定 tensor width 为 16；4943/6000 行有 16 个 unique negatives，1057/6000 行有 9--15 个 unique negatives 并循环精确重复表达式补齐。该 padding 不改变每行可达到的 current-policy argmin/argmax surprisal，但改变 tie/slot multiplicity。Base positive-only 可以学习；在已测 0.25/0.5/1/2 范围内，负压力增大总体伴随 pass@8 与终态 valid-rate 恶化，x1/x2 有严重任务/输出有效率退化但无 NaN/Inf；low-SFT x1 未超过 positive-only。本证据仅为 dirty、single-seed、不同 seed offset、不同 early-stop horizon 的 pilot，不构成方法排名或稳态结论。下一步需另行冻结靠近 0 的 paired-seed 扫描。

## Source 15: docs/handoff.md: HANDOFF-DELTA-BLOCK section_end:e8-v2-active-taper-sweep-ready

### Delta block `section_end:e8-v2-active-taper-sweep-ready`

- **Countdown E8 V2 active taper tuning:** 注册 `EXT-C-E8-ORACLE-OFFLINE-V2-TAPER-SWEEP-0.5B-01`，状态为 `implemented_ready_not_run`。本轮停止继续调 Global，只比较 Linear、Quadratic、Exp；8 个 `rho` × 3 个 paired tuning seeds，共 72 cells，使用 GPU 0--7。初始化 aggregate negative-gradient RMS 均匹配 Global `x1/32` 预算；current-near 中位点锚定 `u=0`，current-far 中位点锚定 `u=1`。SBRC、Hybrid、Global retuning、SFT init、on-policy 和 replay 均排除。本轮仅为调参 pilot；必须报告 best 与 terminal，并在冻结超参后使用新 seeds 才能形成方法排名。

## Source 16: docs/handoff.md: HANDOFF-DELTA-BLOCK section_end:e8-v2-alpha1-c-scan-running-pilot

### Delta block `section_end:e8-v2-alpha1-c-scan-running-pilot`

- **Countdown E8 alpha=1 high-`c` one-parameter scan （`EXT-C-E8-ORACLE-OFFLINE-V2-ALPHA1-C-SCAN-0.5B-01`）：**登记为已启动但尚无终态结果的 development pilot，是前序 62-cell alpha-by-c 网格的非破坏性 successor。实验在同一 frozen E8 V2 bank、Qwen2.5-0.5B、fresh LoRA 和 validation-only 协议下，固定 `d=-stopgrad(sequence_mean_logprob)`、`u=d/2`、`w=alpha*exp(-c*u^2)`，比较 Positive-only、前序最佳 `alpha=0.5,c=1.0` 与 `alpha=1,c={1.5,2,2.25,2.5,3,4}`。四个新 development seed offsets 为 `5000,6000,7000,8000`，共 32 cells；8 张 GPU 每卡 2 个运行槽，最多 16 cells 并发、预期两波。服务器从 clean launch commit `a54dc74b849561c15f6195336fca446ed36f0640` 在登记前启动；本登记不修改其 config、trainer、runtime 或 launcher，避免破坏正在运行 workdir 的 identity-checked resume。固定 1200 steps、每 100 steps Greedy/Pass@8、每 200 steps Pass@64，test split 禁止访问。该 Countdown 实验只承担external-validity tuning；不得预设 alpha 可以删除，也不得把四个 development seeds 升级为正式方法排名、收敛、稳态、受控因果识别或 OOD 结论。终态必须分别报告任务性能、valid-rate 结构代理和 NaN/Inf 数值失败，并同时审计 terminal 与 800--1200 late window。

## Source 17: docs/handoff.md: HANDOFF-DELTA-BLOCK section_end:e8-v2-fixed-alpha-continuous-exp-grid-ready

### Delta block `section_end:e8-v2-fixed-alpha-continuous-exp-grid-ready`

- **Countdown E8 V2 fixed-alpha continuous EXP pilot:** 登记 `EXT-C-E8-ORACLE-OFFLINE-V2-CONTINUOUS-EXP-GRID-0.5B-01`，状态为 `implemented_ready_not_run`。训练对每个 prompt 的全部去重负样本使用固定 `alpha * exp(-c * u^2)` 权重，其中 `u=-stopgrad(sequence mean log-prob)/2`；不再使用 current-near/current-far 极值选择、0.5/0.5 二元混合、`negative_scale`、初始化梯度预算匹配或按权重和归一化。本轮联合扫描 31 个 `(alpha,c)` 点和 2 个新 development seeds，共 62 cells，固定 1200 步且只用 validation 调参；当前 runtime 不接受 test split。GPU autotune 只选择活动设备槽，并在完整 sweep 前执行 2-step 真实 liveness。此前 72-cell budget-matched taper sweep 作为历史 pilot 保留，不得用于回答本轮固定 alpha 问题。本登记仅是 Countdown 外部有效性调参 pilot，不构成机制识别、正式方法排名、收敛、稳态或 OOD 泛化结论。

## Source 18: docs/handoff.md: HANDOFF-DELTA-BLOCK section_end:e8-v2-global-low-scale-milestone-pilot

### Delta block `section_end:e8-v2-global-low-scale-milestone-pilot`

- **Countdown E8 V2 Global low-scale milestone pilot:** 注册 `EXT-C-E8-ORACLE-OFFLINE-V2-GLOBAL-LOW-SCALE-SWEEP-0.5B-01`。四个 paired seeds 的 Global `x1/32` 在 validation-selected best checkpoint 上，test pass@8 相对 Positive-only 提高 4.4 个百分点，pass@64 提高 12.075 个百分点；terminal pass@8 则低 0.725 个百分点。该结果支持足够小的负优势可被利用，同时显示持续、无距离区分的 Global 压力不能保持收益。24 cells 无 NaN/Inf；support/structure boundary 未正式审计。本证据为 dirty-worktree milestone diagnostic pilot，不构成正式排名、收敛或 Global 终态优越性结论。

## Source 19: docs/handoff.md: HANDOFF-DELTA-BLOCK section_end:v52-countdown-current-gate-override

### Delta block `section_end:v52-countdown-current-gate-override`

- **Countdown v52 覆盖：** `EXT-C-E8-V4.3` 取代 V4.2 成为当前 E8-MECH/focused pilot；V4.2 只保留 matched-pair mechanism provenance。`EXT-C-E8-SCALE-01` 继续等待 V4.3 与 E7-BENCH，不因本次实现自动解锁。

## Source 20: docs/handoff.md: HANDOFF-DELTA-BLOCK section_end:v57-countdown-offline-bank-current-gate

### Delta block `section_end:v57-countdown-offline-bank-current-gate`

- **Countdown v57 覆盖：** `EXT-C-E8-V4.4-OFFLINE-BANK` 是用户批准的当前离线 focused pilot；V4.3 保留为 fixed-pair predecessor。V4.4 只改变固定负样本覆盖与 current-policy near/far reselection，不引入在线数据刷新。`EXT-H-E7-Q2` 仍是下一正式 route item，`EXT-C-E8-SCALE-01` 继续 blocked。

## Source 21: docs/handoff.md: HANDOFF-DELTA-BLOCK section_end:v59-countdown-offline-bank-tuning-current-gate

### Delta block `section_end:v59-countdown-offline-bank-tuning-current-gate`

- **Countdown v59 覆盖：** `EXT-C-E8-V4.5-OFFLINE-BANK-TUNING` 是当前用户批准的离线 focused successor；V4.4 作为 frozen-bank predecessor 保留。V4.5 只调 calibrated global negative multiplier 与 exponential taper lambda，禁止在线刷新、方向筛选或模型规模同时变化。`EXT-H-E7-Q2` 仍是下一 formal route item，`EXT-C-E8-SCALE-01` 继续 blocked。

## Source 22: docs/handoff.md: HANDOFF-DELTA-BLOCK section_end:v62-countdown-online-offpolicy-current-gate

### Delta block `section_end:v62-countdown-online-offpolicy-current-gate`

- **Countdown v62 覆盖：** `EXT-C-E8-V4.6-ONLINE-OFFPOLICY-REPLAY` 是当前用户批准并已实现的 Countdown focused successor，状态为 **implemented + not_run**。执行前必须提供完整 V4.5 `RUN_COMPLETE.json`/`terminal_audit.json` 及其指向的 V4.4 frozen inputs；runner fail-closed 校验输入与 reference adapter。它可作为独立 pilot 启动，但不改变 `EXT-H-E7-Q2` 的 formal 优先级，也不自动解锁 `EXT-C-E8-SCALE-01`。

## Source 23: docs/handoff.md: HANDOFF-DELTA-BLOCK section_end:v73-e8-taper-current-gate

### Delta block `section_end:v73-e8-taper-current-gate`

- **Countdown E8-TAPER v73 覆盖：**`EXT-C-E8-TAPER-0.5B-01` 已实现 corrected `S -> d=sqrt(S)` 坐标、独立冻结尺度、deterministic detached weighting、paired sampler 身份校验和终态审计，当前为 **implemented + ready + not_run**。只允许先运行登记的 0.5B pilot；不得将 smoke/static test 写成科学结果，也不得预设 Exp、Global 或任何 taper 获胜。

## Source 24: docs/handoff.md: HANDOFF-DELTA-BLOCK section_end:v75-e8-taper-diagnostic-bugfix

### Delta block `section_end:v75-e8-taper-diagnostic-bugfix`

- **Countdown E8-TAPER 0.5B diagnostic bugfix:** `EXT-C-E8-TAPER-0.5B-01`
  keeps the same experiment ID, methods, paired seeds, taper formulas and synthetic-negative policy.
  The frozen natural replay target is reduced from 1500 to 900 train prompts because the 0.5B
  frozen reference produced 913 eligible natural-negative prompts with `synthetic_negative_fallback=false`.
  Teacher-forced diagnostics are streamed with batch size 1, and same-graph raw/weighted gradient
  diagnostics retain the graph only inside the diagnostic audit. This update is an implementation/config
  repair, not a scientific result; real Qwen/CUDA pilot remains not run.

## Source 25: docs/handoff.md: HANDOFF-DELTA-BLOCK section_end:v79-e8-active-tail-current-gate

### Delta block `section_end:v79-e8-active-tail-current-gate`

- **Countdown E8-TAPER v79 覆盖：**`EXT-C-E8-TAPER-0.5B-01` 仍为 implemented + ready + not_run pilot，但当前有效协议使用 independent-calibration common-half median tau、nondegenerate calibration fail-closed guard 与 streamed surprisal-bin diagnostics。应用后必须先跑短预算 sanity 验证各方法未退化为 uncontrolled clone；smoke/sanity/pilot 不得写成正式结果或方法排名。

## Source 26: docs/handoff.md: HANDOFF-DELTA-BLOCK section_end:v52-e8-route-override

### Delta block `section_end:v52-e8-route-override`

7. **v52 路线覆盖：** 上述第 5 项的当前 E8-MECH owner 更新为 `EXT-C-E8-V4.3`。V4.3 只修复长期训练中的动态 remoteness 控制并保留 V4.2 静态方法作消融；E8-SCALE 的 3B/7B 规模结论仍需后续独立执行。

## Source 27: docs/handoff.md: HANDOFF-DELTA-BLOCK section_end:v57-e8-offline-online-route

### Delta block `section_end:v57-e8-offline-online-route`

8. **v57 E8 内部路线覆盖：** 在进入 E8 外部诊断时，先执行 `EXT-C-E8-V4.4-OFFLINE-BANK`，只改变 fixed-bank 密度与每步动态选择；online off-policy 必须作为独立 successor 重新冻结 rollout actor、同步滞后、replay age、seeds 与预算匹配，不能与 V4.4 共用结论。

## Source 28: docs/handoff.md: HANDOFF-DELTA-BLOCK section_end:v75-countdown-status-note

### Delta block `section_end:v75-countdown-status-note`

- **v75 Countdown 逐样本机制诊断补记：**`EXT-C` 已完成一个 single-seed full-bank `arithmetic_wrong` response diagnostic：`6000` puzzles × near/far = `12000` rows，固定 `negative_coefficient_abs=1.0`，观察到 surprisal 与 trainable-parameter gradient norm 的正相关、near/far 配对增益和 decile 平台化趋势。该补记只把 Countdown 机制观察从 10-puzzle smoke 升级为 full-bank pilot；不升级 `EXT-C-E8-TAPER-0.5B-01` 或 `EXT-C-E8-SCALE-01` 的 formal 状态，也不改变 Countdown 不能替代 D-U1/C-U1 因果识别的边界。

## Source 29: docs/handoff.md: HANDOFF-DELTA-BLOCK section_end:v52-execution-order-override

### Delta block `section_end:v52-execution-order-override`

11. **v52 执行覆盖：** 当锁定路线进入 E8-MECH 时，执行 `EXT-C-E8-V4.3` 而不是 V4.2；当前只完成注册和代码实现，真实 Qwen/CUDA pilot 仍为 not_run。

## Source 30: experiments/registry.yaml: experiments[EXT-C-E8-V4, EXT-C-E8-V4.1, EXT-C-E8-V4.2, EXT-C-E8-V4.3, EXT-C-E8-V4.4-OFFLINE-BANK, EXT-C-E8-V4.5-OFFLINE-BANK-TUNING, EXT-C-E8-V4.6-ONLINE-OFFPOLICY-REPLAY, EXT-C-E8-TAPER-0.5B-01, EXT-C-E8-SCALE-01]

collection: experiments
entries:
- id: EXT-C-E8-V4
  environment: EXT-C
  name: countdown_base_first_transformer_external_validation
  status: superseded
  claim: Test whether the fixed-negative-advantage near/far remoteness mechanism identified in D-U1 transfers to a shared-parameter
    Transformer, and whether controlled negative advantages improve held-out structural generalization relative to positive-only
    and uncontrolled negative updates.
  role: external_validity
  execution_class: superseded
  does_not_replace:
  - D-U1
  - D-Diag
  code_entrypoint: src/drpo/countdown_qwen_arena_onefile.py
  primary_model: Qwen Instruct 0.5B
  initialization: &id001
    protocol: base_first
    base_gate:
      greedy_verifier_success_min: 0.15
      valid_rate_min: 0.8
    fallback: minimal_sft_only_if_base_gate_fails
    sft_gate:
      greedy_verifier_success_min: 0.15
  data:
    task: four-number Countdown arithmetic
    train_validation_test_split: disjoint_canonical_operator_tree_signatures
    positive_support: training_structures_only
    negative_reward: 0
    fixed_negative_advantage: -1.0
    near_far_matching:
      surprisal_gap_min: 0.5
      token_length_difference_max: 2
      tree_depth_difference_max: 1
      value_error_ratio_max: 4.0
  mechanism_probe: &id002
    matched_pairs_min: 16
    default_pairs: 32
    metrics:
    - token_surprisal
    - direct_logit_score_norm
    - trainable_adapter_gradient_norm
    - target_surprisal_delta_after_equal_negative_updates
    - correct_answer_collateral_surprisal_delta
  methods:
  - positive_only
  - controlled_negative
  - uncontrolled_negative
  pairing:
    shared_initial_adapter: true
    shared_offline_data: true
    shared_training_seed: true
    shared_evaluation_seed: true
  primary_metrics:
  - greedy_verifier_success
  - pass_at_k
  - valid_rate
  - greedy_unseen_structure_success
  - pass_at_k_unseen_structure
  reporting_separation: &id003
  - task_performance
  - support_or_structure_coverage
  - nan_inf_numerical_failure
  model_scaling:
    qwen_3b: conditional_replication
    qwen_7b: optional
  artifact_budget:
    main_package_hard_limit_mib: 25
    single_file_main_limit_mib: 10
    max_retained_checkpoints_per_method: 2
    checkpoint_roles:
    - best
    - latest
    save_foundation_model_weights: false
    save_optimizer_state_by_default: false
    large_checkpoint_delivery: persistent_local_index
    sidecar_default: false
    sidecar_only_if_pre_registered: true
    sidecar_selection: explicit_files_only
    sidecar_purpose_required: true
    sidecar_filename_policy: new_versioned_path
  formal_run_status: not_run
  execution:
    state: registered
    run_id: null
    last_heartbeat_utc: null
    process_exit_code: null
  evidence:
    raw_complete: false
    terminal_audited: false
    package_created: false
    package_filename: null
    package_sha256: null
    delivered_to_user: false
    applied_commit: null
  superseded_by: EXT-C-E8-V4.1
  preserved_for_provenance: true
  replacement_reason: V4 allowed unmatched pairs in method training, retained only best adapters, used a weaker structural
    split, and lacked a calibrated equal-budget global control.
- id: EXT-C-E8-V4.1
  environment: EXT-C
  name: countdown_audited_base_first_transformer_external_validation
  status: not_run
  claim: Test whether the fixed-negative-advantage near/far remoteness mechanism identified in D-U1 transfers to a shared-parameter
    Transformer, and whether selective far-negative control improves held-out canonical pattern-family generalization beyond
    positive-only, uncontrolled negative updates, and an equal-budget global control.
  role: external_validity
  execution_class: pilot
  pilot_execution:
    channel_ref: hardened-v1
    launch_mode: guarded_orchestrator
    operator_entrypoint: scripts/run_countdown_pilot.py
    guard_required: true
  does_not_replace:
  - D-U1
  - D-Diag
  code_entrypoint: src/drpo/countdown_qwen_arena_onefile.py
  one_click_entrypoint: scripts/run_countdown_pilot.py
  one_click_contract:
    required_operator_inputs:
    - model_path
    - work_dir
    automatic_decisions:
    - visible_gpu_selection_up_to_eight
    - base_gate_and_sft_fallback
    - safe_stage_gpu_assignment
    - best_terminal_or_last_finite_evaluation_inventory
    - terminal_audit
    - guarded_success_or_failure_artifact_packaging
    build_offline_parallelism: disabled_to_preserve_registered_rng_stream
    hardened_guard_required: true
    required_success_outputs:
    - RUN_COMPLETE.json
    - terminal_audit.json
    - arena_summary.csv
  primary_model: Qwen Instruct 0.5B
  initialization: *id001
  parameterization:
    runner_version: 4.2.0-one-click-audited-orchestrator
    pilot: bf16_lora
    shared_across_methods: true
    qlora: engineering_smoke_only
    lora_configuration:
      rank: 32
      alpha: 64
      dropout: 0.05
      bias: none
      target_modules:
      - q_proj
      - k_proj
      - v_proj
      - o_proj
      - gate_proj
      - up_proj
      - down_proj
    full_finetune_confirmation:
      model: Qwen Instruct 0.5B
      gate: only_after_reproducible_lora_pilot_signal
      implemented_in_current_runner: false
      requires_separate_registration_before_implementation: true
      mixed_with_lora_main_comparison: false
      status: not_run
  data:
    task: four-number Countdown arithmetic
    terminology: held_out_canonical_pattern_family_generalization
    not_claimed:
    - state_distribution_ood
    - latent_subtree_skill_causality
    generation: canonical_pattern_first_capacity_audited_balanced_core
    canonical_pattern_catalog_size_for_four_numbers: 96
    validation_test_holdout: disjoint_three_number_pattern_families_and_four_number_derivatives
    exclude_heldout_families_from:
    - training_positive_completions
    - training_negative_completions
    negative_reward: 0
    fixed_negative_advantage: -1.0
    near_far_matching:
      surprisal_gap_min: 0.5
      token_length_difference_max: 2
      tree_depth_difference_max: 1
      value_error_ratio_max: 4.0
      resample_rounds_default: 3
      unresolved_pair_action: drop_example
      unmatched_pairs_allowed_in_training: false
  mechanism_probe: *id002
  methods:
  - positive_only
  - controlled_negative
  - uncontrolled_negative
  - global_matched
  negative_scale_calibration:
    symbol: beta
    replaces_unjustified_legacy_default_alpha: 0.7
    mechanism_probe_advantage_unchanged: -1.0
    calibration_split: fixed_training_calibration_subset
    calibration_time: common_initial_adapter_before_any_method_training
    positive_budget: rms_gradient_norm_of_positive_branch
    negative_budget: rms_gradient_norm_of_unscaled_uncontrolled_negative_branch
    formula: beta = positive_rms_gradient_norm / uncontrolled_negative_rms_gradient_norm
    shared_by_methods:
    - controlled_negative
    - uncontrolled_negative
    - global_matched
    frozen_after_calibration: true
    task_metrics_used_for_selection: false
    validation_data_used_for_calibration: false
    test_data_used_for_calibration: false
    failure_action: stop_without_fallback_if_nonfinite_or_nonpositive
  global_matched:
    calibration_split: fixed_training_calibration_subset
    matching_target: rms_negative_gradient_norm_of_controlled_negative
    gamma_frozen_before_method_training: true
    reads_near_far_identity: false
    test_data_used_for_calibration: false
  pilot_configuration:
    development_seeds:
    - 1234
    formal_held_out_seeds: pending_separate_registration_after_pilot
    formal_multiseed_gate: blocked_until_held_out_seeds_are_registered
    data_sizes:
      train: 6000
      validation: 500
      test: 1000
      offline_matched_rows: 1500
      rollouts_per_prompt: 12
    pair_resample_rounds: 3
    calibration_batches: 16
    optimization:
      maximum_steps: 1200
      minimum_steps_before_early_stop: 400
      evaluation_interval_steps: 100
      early_stop_patience_evaluations: 6
      early_stop_delta: 0.002
      selection_metric: greedy_success
      learning_rate: 5.0e-05
      warmup_ratio: 0.03
      maximum_gradient_norm: 1.0
    evaluation:
      pass_at_k: 8
      evaluation_examples: 500
    negative_branches:
      near_mix: 0.5
      far_mix: 0.5
      far_taper_lambda: 0.7
      surprisal_threshold: 2.0
  pairing:
    shared_initial_adapter: true
    shared_offline_data: true
    shared_training_seed: true
    shared_evaluation_seed: true
    shared_bf16_lora_configuration: true
  checkpoint_policy:
    binary_storage: server_local_or_external_persistent_storage_only
    commit_model_binaries_to_git: false
    normal_completion:
    - best_adapter
    - terminal_adapter
    nonfinite_failure:
    - best_adapter
    - last_finite_adapter
    last_finite_definition: exact_trainable_adapter_state_before_the_first_nonfinite_optimizer_update
    last_finite_implementation: cpu_snapshot_of_trainable_lora_parameters_before_each_optimizer_step
    failure_manifest_fields:
    - failure_detected_at_step
    - last_finite_step
    maximum_retained_per_method: 2
    repository_records:
    - checkpoint_manifest
    - path
    - step
    - size_bytes
    - sha256
  primary_metrics:
  - greedy_verifier_success
  - pass_at_k
  - valid_rate
  - greedy_unseen_structure_presence
  - greedy_unseen_structure_success
  - pass_at_k_unseen_structure_presence
  - pass_at_k_unseen_structure_success
  - heldout_pattern_family_coverage
  - heldout_pattern_family_precision_micro
  - heldout_pattern_family_precision_macro
  - per_pattern_attempts_correct_precision
  - correct_heldout_patterns
  structure_metric_semantics:
    presence: generated_a_valid_completion_with_a_pattern_outside_training_structure_support
    success: verifier_correct_and_pattern_outside_training_structure_support
    per_pattern_precision: correct_generations_divided_by_valid_generations_of_that_pattern
    zero_attempt_precision: null
    greedy_and_sampled_denominators: separate
  result_status_propagation:
    top_level_run_controls_children: true
    allowed:
    - pilot
    - engineering_smoke
    - standalone_unclassified
    direct_subcommand_default: standalone_unclassified
  reporting_separation: *id003
  model_scaling:
    qwen_3b: conditional_replication_after_v4_1
    qwen_7b: optional
  artifact_budget:
    main_package_hard_limit_mib: 25
    single_file_main_limit_mib: 10
    max_retained_checkpoints_per_method: 2
    checkpoint_roles:
    - best_adapter
    - terminal_adapter_on_normal_completion
    - last_finite_adapter_on_nonfinite_failure
    mutually_exclusive_latest_roles: true
    save_foundation_model_weights: false
    save_optimizer_state_by_default: false
    large_checkpoint_delivery: persistent_local_index
    sidecar_default: false
    sidecar_only_if_pre_registered: true
    sidecar_selection: explicit_files_only
    sidecar_purpose_required: true
    sidecar_filename_policy: new_versioned_path
  orchestration:
    gpu_selection_default: auto_all_visible_up_to_8
    safe_parallel_stages:
    - mechanism_probe_and_negative_budget_calibration
    - four_method_training_one_process_per_gpu
    - raw_base_and_reference_test_on_spare_gpus
    - all_best_terminal_last_finite_test_jobs
    single_gpu_stages:
    - preflight
    - pattern_first_data_generation
    - base_and_reference_gate_evaluation
    - conditional_sft
    - build_offline
    reason_build_offline_not_sharded: preserve_frozen_rng_stream_and_dataset_protocol
    automatic_decision_log: automatic_decisions.json
    failure_marker: RUN_FAILED.json
    terminal_audit: terminal_audit.json
  formal_run_status: not_run
  execution:
    state: registered
    run_id: null
    last_heartbeat_utc: null
    process_exit_code: null
  evidence:
    raw_complete: false
    terminal_audited: false
    package_created: false
    package_filename: null
    package_sha256: null
    delivered_to_user: false
    applied_commit: null
  superseded_by: EXT-C-E8-V4.2
  preserved_for_provenance: true
  replacement_reason: V4.2 registers the post-pilot capability gates, balanced 6000-row offline corpus, effective-epoch training,
    dynamic far-field diagnostics, and isolated full-FT reference diagnostic.
- id: EXT-C-E8-V4.2
  environment: EXT-C
  name: countdown_balanced_offline_dynamic_far_field_external_validation
  status: superseded
  claim: Test whether fixed-advantage far negatives exert larger and more damaging policy-relative influence than matched
    near negatives in a shared-parameter Transformer, and whether selective far-negative control preserves held-out canonical
    pattern-family performance better than positive-only, uncontrolled negative updates, and an equal-initial-budget global
    control. Near negatives are not assumed to be universally beneficial.
  role: external_validity
  execution_class: superseded
  registration_base_commit: e8b62dde518f593ff8325c7da94c41406311ca45
  superseded_by: EXT-C-E8-V4.3
  preserved_for_provenance: true
  pilot_execution:
    channel_ref: hardened-v1
    launch_mode: guarded_orchestrator
    operator_entrypoint: scripts/run_countdown_pilot.py
    guard_required: true
  does_not_replace:
  - D-U1
  - D-Diag
  supersedes: EXT-C-E8-V4.1
  preserved_prior_run_note: The dirty-worktree single-seed V4.1 server run is exploratory mechanism evidence only and does
    not satisfy this registered protocol.
  code_entrypoint: src/drpo/countdown_qwen_arena_onefile.py
  one_click_entrypoint: scripts/run_countdown_pilot.py
  one_click_contract:
    required_operator_inputs:
    - model_path
    - work_dir
    automatic_decisions:
    - visible_gpu_selection_up_to_eight
    - base_gate_and_extended_lora_sft_fallback
    - mechanism_only_versus_method_effect_capability_gate
    - isolated_full_ft_reference_diagnostic
    - safe_stage_gpu_assignment
    - balanced_offline_construction_and_nested_subsets
    - best_terminal_or_last_finite_evaluation_inventory
    - terminal_audit
    - guarded_success_or_failure_artifact_packaging
    build_offline_parallelism: disabled_to_preserve_one_rng_stream_and_pattern_quotas
    hardened_guard_required: true
    required_success_outputs:
    - RUN_COMPLETE.json
    - terminal_audit.json
    - arena_summary.csv
  primary_model: Qwen2.5-0.5B-Instruct
  initialization: *id001
  parameterization:
    runner_version: 4.3.0-balanced-offline-diagnostics
    main_comparison: bf16_lora
    shared_across_methods: true
    qlora: engineering_smoke_only
    lora_configuration:
      rank: 32
      alpha: 64
      dropout: 0.05
      bias: none
      target_modules:
      - q_proj
      - k_proj
      - v_proj
      - o_proj
      - gate_proj
      - up_proj
      - down_proj
    full_finetune_reference_diagnostic:
      model: Qwen2.5-0.5B-Instruct
      implemented: true
      parameterization: bf16_full_parameter_sft
      learning_rate: 2.0e-05
      same_sft_train_validation_split_as_lora: true
      maximum_epochs: 6
      minimum_epochs: 3
      early_stop_patience_epochs: 2
      role: isolated_parameterization_capacity_diagnostic
      eligible_to_initialize_four_method_comparison: false
      mixed_with_lora_main_comparison: false
      checkpoint_storage: server_local_index_only
      status: not_run
  capability_gates:
    base_first_sft_trigger:
      greedy_success_min: 0.15
      valid_rate_min: 0.8
    mechanism_pilot:
      greedy_success_min: 0.08
      valid_rate_min: 0.95
      failure_action: stop_before_offline_mechanism_interpretation
    method_effect_comparison:
      greedy_success_min: 0.15
      valid_rate_min: 0.95
      failure_action: complete_mechanism_and_full_ft_diagnostics_but_skip_four_method_ranking
    threshold_note: Fifteen percent is an operational floor-effect guard rather than a theoretical constant; it may only change
      in a new registered protocol, not during a run.
  sft:
    main_parameterization: lora
    maximum_epochs: 6
    minimum_epochs: 3
    early_stop_patience_epochs: 2
    selection_metric: greedy_success
    learning_rate: 0.0002
    warmup_ratio: 0.03
    maximum_gradient_norm: 1.0
    evaluate_each_epoch: true
    scheduler: cosine_over_the_full_registered_schedule
  data:
    task: four-number Countdown arithmetic
    terminology: held_out_canonical_pattern_family_generalization
    not_claimed:
    - state_distribution_ood
    - universal_benefit_of_near_negatives
    - latent_subtree_skill_causality
    generation: canonical_pattern_first_capacity_audited_balanced_core
    canonical_pattern_catalog_size_for_four_numbers: 96
    validation_test_holdout: disjoint_three_number_pattern_families_and_four_number_derivatives
    generated_sizes:
      train: 6000
      validation: 500
      test: 1000
    exclude_heldout_families_from:
    - training_positive_completions
    - training_negative_completions
    negative_reward: 0
    fixed_negative_advantage: -1.0
    offline_matched_dataset:
      rows: 6000
      construction_scope: one_matched_row_for_each_registered_training_prompt
      balance_key: oracle_structure
      expected_train_patterns: 48
      expected_rows_per_pattern: 125
      nested_balanced_subsets:
      - 1500
      - 3000
      - 6000
      reference_policy_bound: true
      reusable_across_methods_training_seeds_and_registered_hyperparameter_checks: true
      must_rebuild_after_reference_model_or_parameterization_change: true
      build_parallelism: single_gpu_single_rng_stream
      unresolved_quota_action: fail_closed_and_preserve_partial_diagnostics
    near_far_matching:
      surprisal_gap_min: 0.5
      token_length_difference_max: 2
      tree_depth_difference_max: 1
      value_error_ratio_max: 4.0
      rollouts_per_prompt: 12
      minimum_negative_candidates: 8
      resample_rounds_default: 8
      deterministic_synthetic_rescue_candidates_max: 64
      rescue_does_not_relax_matching_thresholds: true
      unmatched_pairs_allowed_in_training: false
  mechanism_probe: *id002
  methods:
  - positive_only
  - controlled_negative
  - uncontrolled_negative
  - global_matched
  core_interpretation:
    primary: Far negatives may be more harmful because their fixed-advantage updates have larger policy-relative influence.
    near_branch: Near negatives may be beneficial, neutral, or harmful; universal benefit is not required by the claim.
    prohibited_inference: Do not interpret one held-out pattern redistribution as aggregate method superiority.
  dynamic_diagnostics:
    cadence: step_zero_and_each_validation_checkpoint
    fixed_balanced_diagnostic_rows: 32
    gradient_diagnostic_rows: 8
    fields:
    - positive_near_far_surprisal_mean_median_p90
    - near_fraction_crossing_the_frozen_far_threshold
    - controlled_far_token_weight
    - positive_near_far_raw_and_scaled_gradient_norm
    - far_to_near_gradient_norm_ratio
    - positive_near_and_positive_far_gradient_cosine
    - effective_epoch
    output: methods_each_dynamic_diagnostics_jsonl
  negative_scale_calibration:
    symbol: beta
    mechanism_probe_advantage_unchanged: -1.0
    calibration_split: fixed_training_calibration_subset
    calibration_time: common_initial_lora_adapter_before_any_method_training
    positive_budget: rms_gradient_norm_of_positive_branch
    negative_budget: rms_gradient_norm_of_unscaled_uncontrolled_negative_branch
    formula: beta = positive_rms_gradient_norm / uncontrolled_negative_rms_gradient_norm
    shared_by_methods:
    - controlled_negative
    - uncontrolled_negative
    - global_matched
    frozen_after_calibration: true
    task_metrics_used_for_selection: false
    validation_data_used_for_calibration: false
    test_data_used_for_calibration: false
    failure_action: stop_without_fallback_if_nonfinite_or_nonpositive
  global_matched:
    calibration_split: fixed_training_calibration_subset
    matching_target: rms_negative_gradient_norm_of_controlled_negative
    gamma_frozen_before_method_training: true
    reads_near_far_identity: false
    test_data_used_for_calibration: false
  pilot_configuration:
    development_seeds:
    - 1234
    formal_held_out_seeds: pending_separate_registration_after_pilot
    formal_multiseed_gate: blocked_until_held_out_seeds_are_registered
    calibration_batches: 16
    optimization:
      duration_unit: effective_offline_epochs
      maximum_effective_epochs: 6
      minimum_effective_epochs_before_early_stop: 2
      evaluation_interval_effective_epochs: 1
      early_stop_patience_evaluations: 2
      early_stop_delta: 0.002
      selection_metric: greedy_success
      learning_rate: 5.0e-05
      warmup_ratio: 0.03
      maximum_gradient_norm: 1.0
      h20_reference_effective_batch_size: 32
      h20_reference_updates_per_epoch: 188
      h20_reference_maximum_steps: 1128
      h20_reference_minimum_steps: 376
    evaluation:
      pass_at_k: 8
      evaluation_examples: 500
    negative_branches:
      near_mix: 0.5
      far_mix: 0.5
      far_taper_lambda: 0.7
      surprisal_threshold: 2.0
  pairing:
    shared_initial_lora_adapter: true
    shared_offline_data: true
    shared_training_seed: true
    shared_evaluation_seed: true
    shared_bf16_lora_configuration: true
    full_ft_diagnostic_excluded_from_pairing_and_ranking: true
  checkpoint_policy:
    binary_storage: server_local_or_external_persistent_storage_only
    commit_model_binaries_to_git: false
    lora_normal_completion:
    - best_adapter
    - terminal_adapter
    lora_nonfinite_failure:
    - best_adapter
    - last_finite_adapter
    lora_last_finite_definition: exact_trainable_adapter_state_before_the_first_nonfinite_optimizer_update
    full_ft_diagnostic_normal_completion:
    - best_model
    - terminal_model
    full_ft_nonfinite_action: fail_closed_with_epoch_checkpoint_semantics
    maximum_retained_per_main_method: 2
    lora_tokenizer_storage: reference_registered_model_path_without_per_checkpoint_vocab_duplication
    full_model_tokenizer_storage: self_describing_checkpoint_metadata
    repository_records:
    - checkpoint_manifest
    - path
    - step
    - size_bytes
    - sha256
  primary_metrics:
  - greedy_verifier_success
  - pass_at_k
  - valid_rate
  - greedy_unseen_structure_presence
  - greedy_unseen_structure_success
  - pass_at_k_unseen_structure_presence
  - pass_at_k_unseen_structure_success
  - heldout_pattern_family_coverage
  - heldout_pattern_family_precision_micro
  - heldout_pattern_family_precision_macro
  - per_pattern_attempts_correct_precision
  - correct_heldout_patterns
  reporting_separation: *id003
  model_scaling:
    qwen_3b: conditional_after_v4_2_if_reference_capacity_or_cross_seed_confirmation_requires_it
    qwen_7b: optional
  orchestration:
    gpu_selection_default: auto_all_visible_up_to_8
    safe_parallel_stages:
    - balanced_offline_build_and_isolated_full_ft_reference_diagnostic_on_different_gpus
    - mechanism_probe_and_negative_budget_calibration
    - four_method_training_one_process_per_gpu
    - raw_base_lora_reference_and_full_ft_reference_test_on_spare_gpus
    - all_best_terminal_last_finite_test_jobs
    single_rng_stages:
    - pattern_first_data_generation
    - balanced_build_offline
    reason_build_offline_not_sharded: preserve_one_deterministic_rng_stream_and_pattern_quota_protocol
    automatic_decision_log: automatic_decisions.json
    failure_marker: RUN_FAILED.json
    terminal_audit: terminal_audit.json
  formal_run_status: not_run
  execution:
    state: registered
    run_id: null
    last_heartbeat_utc: null
    process_exit_code: null
  evidence:
    raw_complete: false
    terminal_audited: false
    package_created: false
    package_filename: null
    package_sha256: null
    delivered_to_user: false
    applied_commit: null
- id: EXT-C-E8-V4.3
  environment: EXT-C
  name: countdown_policy_relative_dynamic_negative_control_pilot
  status: not_run
  claim: Test whether the V4.2 static construction-time near/far labels caused the controlled method to miss negative completions
    that became far under the moving policy, and whether applying the same current-policy token-surprisal taper to every negative
    branch improves verifier success and valid rate relative to static control, positive-only, and uncontrolled negatives.
  role: external_validity_focused_method_diagnostic
  execution_class: pilot
  registration_base_commit: 0907c3c0e76fc836c2bf2b752abf554c17f79f22
  pilot_execution:
    channel_ref: hardened-v1
    launch_mode: guarded_orchestrator
    operator_entrypoint: scripts/run_countdown_pilot.py
    guard_required: true
  does_not_replace:
  - D-U1
  - D-Diag
  supersedes: EXT-C-E8-V4.2
  preserved_v4_2_role: Matched near/far pairs remain valid for the instantaneous fixed-advantage mechanism probe. Only their
    use as permanent long-run control identities is replaced.
  code_entrypoint: src/drpo/countdown_qwen_arena_onefile.py
  one_click_entrypoint: scripts/run_countdown_pilot.py
  primary_model: Qwen2.5-0.5B-Instruct
  parameterization:
    runner_version: 4.4.0-dynamic-negative-control
    main_comparison: bf16_lora
    shared_across_methods: true
    qlora: engineering_smoke_only
    full_finetune_role: isolated_reference_capacity_diagnostic_only
  capability_gates:
    mechanism_and_focused_pilot:
      greedy_success_min: 0.08
      valid_rate_min: 0.95
      failure_action: stop_before_interpreting_negative_control
    formal_method_ranking:
      greedy_success_min: 0.15
      valid_rate_min: 0.95
      failure_action: Run and report the focused single-seed pilot, but prohibit formal method ranking or scale-up claims.
    threshold_note: The 0.08 gate permits the approved 0.5B focused diagnostic; the 0.15 gate remains a floor-effect guard
      for any later ranking claim. Neither threshold may change after results are observed.
  data:
    inherited_from: EXT-C-E8-V4.2
    task: four-number Countdown arithmetic
    terminology: held_out_canonical_pattern_family_generalization
    generated_sizes:
      train: 6000
      validation: 500
      test: 1000
    offline_matched_rows: 6000
    fixed_negative_advantage: -1.0
    static_pair_role: instantaneous_mechanism_matching_and_branch_provenance
    long_run_remoteness_role: recomputed_from_current_policy_each_optimizer_step
    negative_bank_expansion: not_in_this_update
  methods:
  - positive_only
  - controlled_negative
  - dynamic_controlled_negative
  - uncontrolled_negative
  method_definitions:
    controlled_negative: 'V4.2 static ablation: initial near branch remains untapered and only the initial far branch receives
      detached token-surprisal tapering.'
    dynamic_controlled_negative: Apply the same detached current-policy token-surprisal taper to both initially-near and initially-far
      negative branches.
    uncontrolled_negative: Shared calibrated negative scale without surprisal taper.
  frozen_controls:
    shared_initial_lora_adapter: true
    shared_offline_data: true
    shared_training_seed: true
    shared_evaluation_seed: true
    near_mix: 0.5
    far_mix: 0.5
    taper_lambda: 0.7
    surprisal_threshold: 2.0
    maximum_effective_epochs: 6
    minimum_effective_epochs_before_early_stop: 2
    evaluation_interval_effective_epochs: 1
    early_stop_patience_evaluations: 2
    early_stop_delta: 0.002
    selection_metric: greedy_success
    learning_rate: 5.0e-05
    warmup_ratio: 0.03
    maximum_gradient_norm: 1.0
    development_seed: 1234
  negative_scale_calibration:
    inherited_formula: beta_equals_positive_rms_over_uncontrolled_negative_rms
    calibration_split: fixed_training_subset_before_method_training
    shared_by:
    - controlled_negative
    - dynamic_controlled_negative
    - uncontrolled_negative
    task_metrics_used_for_selection: false
    validation_or_test_used: false
  primary_metrics:
  - greedy_verifier_success
  - pass_at_k
  - valid_rate
  - heldout_pattern_family_coverage
  - heldout_pattern_family_precision_micro
  - heldout_pattern_family_precision_macro
  - best_and_terminal_checkpoint_results
  diagnostics:
  - initial_near_fraction_currently_above_threshold
  - current_near_and_far_surprisal
  - current_near_and_far_taper_weight
  - raw_and_tapered_near_and_far_gradient_norm
  - positive_negative_gradient_cosine
  reporting_separation:
  - task_performance_degradation
  - support_or_structure_boundary_event
  - nan_inf_numerical_failure
  interpretation_limits:
  - single_seed_is_pilot_only
  - weight_trajectory_is_implementation_evidence_not_primary_outcome
  - no_universal_method_ranking
  - no_state_distribution_ood_claim
  - countdown_does_not_replace_controlled_causal_identification
  expected_outputs:
  - RUN_COMPLETE.json
  - terminal_audit.json
  - arena_summary.csv
  - methods_each_dynamic_diagnostics_jsonl
  orchestration:
    gpu_selection_default: auto_all_visible_up_to_8
    method_training: one_method_per_gpu_fifo_queue
    reason_build_offline_not_sharded: preserve_one_deterministic_rng_stream_and_pattern_quota_protocol
    terminal_audit_required: true
  formal_run_status: not_run
  execution:
    state: registered
    run_id: null
    last_heartbeat_utc: null
    process_exit_code: null
  evidence:
    raw_complete: false
    terminal_audited: false
    package_created: false
    package_filename: null
    package_sha256: null
    delivered_to_user: false
    applied_commit: null
- id: EXT-C-E8-V4.4-OFFLINE-BANK
  environment: EXT-C
  name: countdown_fixed_offline_negative_bank_dynamic_selection_pilot
  status: not_run
  claim: Test whether one fixed near/far pair per prompt becomes stale too quickly to preserve useful local negative signal,
    and whether a frozen 16-negative offline bank with current-policy near/far reselection can improve verifier outcomes without
    the confound of online data refresh.
  role: external_validity_offline_negative_density_and_dynamic_selection_diagnostic
  execution_class: pilot
  registration_base_commit: c2ad7d5f6fe957d6a6297e6987d878cf72dbf7c8
  predecessor: EXT-C-E8-V4.3
  pilot_execution:
    channel_ref: hardened-v1
    launch_mode: guarded_orchestrator
    operator_entrypoint: scripts/run_countdown_pilot.py
    guard_required: true
  does_not_replace:
  - D-U1
  - D-Diag
  code_entrypoint: src/drpo/countdown_qwen_arena_onefile.py
  one_click_entrypoint: scripts/run_countdown_pilot.py
  primary_model: Qwen2.5-0.5B-Instruct
  parameterization:
    runner_version: 4.5.0-offline-negative-bank
    main_comparison: bf16_lora
    shared_across_methods: true
    qlora: engineering_smoke_only
    full_finetune_role: isolated_reference_capacity_diagnostic_only
  capability_gates:
    mechanism_and_focused_pilot:
      greedy_success_min: 0.08
      valid_rate_min: 0.95
      failure_action: stop_before_interpreting_negative_bank_control
    formal_method_ranking:
      greedy_success_min: 0.15
      valid_rate_min: 0.95
      failure_action: Run and report the focused single-seed pilot, but prohibit formal method ranking, online-successor claims,
        or scale-up claims.
    threshold_note: The 0.08 gate permits the approved 0.5B focused diagnostic; the 0.15 gate remains a floor-effect guard
      for any ranking claim.
  data:
    inherited_from: EXT-C-E8-V4.3
    task: four-number Countdown arithmetic
    terminology: held_out_canonical_pattern_family_generalization
    generated_sizes:
      train: 6000
      validation: 500
      test: 1000
    offline_rows: 6000
    fixed_negative_advantage: -1.0
    negative_bank_size_per_prompt: 16
    minimum_unique_negative_candidates: 16
    bank_generation_time: before_method_training
    bank_mutability_during_training: frozen
    online_rollout_during_method_training: false
    static_pair_role: instantaneous_mechanism_matching_and_branch_provenance
    bank_selection_rule: recompute current-policy sequence surprisal each optimizer step; select minimum as current near and
      maximum as current far
  methods:
  - positive_only
  - dynamic_controlled_negative
  - bank_dynamic_controlled_negative
  - bank_global_matched
  - bank_uncontrolled_negative
  method_definitions:
    dynamic_controlled_negative: V4.3 two-fixed-branch comparator; both original pair branches receive detached current-policy
      token-surprisal tapering.
    bank_dynamic_controlled_negative: Rescore the fixed 16-negative bank every optimizer step, select current near/far, and
      apply the same detached token-surprisal taper to both selected branches.
    bank_global_matched: Use the same current near/far bank selection with one fixed global coefficient calibrated to the
      initial actual negative-gradient RMS of bank_dynamic_controlled_negative.
    bank_uncontrolled_negative: Use the same current near/far bank selection and shared bank negative scale without selective
      taper.
  frozen_controls:
    shared_initial_lora_adapter: true
    shared_offline_rows_and_negative_banks: true
    shared_training_seed: true
    shared_evaluation_seed: true
    near_mix: 0.5
    far_mix: 0.5
    taper_lambda: 0.7
    surprisal_threshold: 2.0
    maximum_effective_epochs: 6
    minimum_effective_epochs_before_early_stop: 2
    evaluation_interval_effective_epochs: 1
    early_stop_patience_evaluations: 2
    early_stop_delta: 0.002
    selection_metric: greedy_success
    learning_rate: 5.0e-05
    warmup_ratio: 0.03
    maximum_gradient_norm: 1.0
    development_seed: 1234
  negative_scale_calibration:
    pair_formula: positive_rms_over_pair_uncontrolled_negative_rms
    bank_formula: positive_rms_over_bank_uncontrolled_negative_rms
    bank_global_match: bank_dynamic_controlled_initial_actual_negative_gradient_rms
    calibration_split: fixed_training_subset_before_method_training
    task_metrics_used_for_selection: false
    validation_or_test_used: false
  primary_metrics:
  - greedy_verifier_success
  - pass_at_k
  - valid_rate
  - heldout_pattern_family_coverage
  - heldout_pattern_family_precision_micro
  - heldout_pattern_family_precision_macro
  - best_and_terminal_checkpoint_results
  diagnostics:
  - selected_bank_near_and_far_slot
  - selected_bank_near_and_far_surprisal
  - selected_bank_near_and_far_taper_weight
  - bank_slot_turnover_and_unique_selection_fraction
  - raw_and_tapered_near_and_far_gradient_norm
  - positive_negative_gradient_cosine
  reporting_separation:
  - task_performance_degradation
  - support_or_structure_boundary_event
  - nan_inf_numerical_failure
  interpretation_limits:
  - single_seed_is_pilot_only
  - bank_selection_and_weight_trajectory_are_implementation_evidence_not_primary_outcomes
  - no_universal_method_ranking
  - no_state_distribution_ood_claim
  - no_online_off_policy_claim
  - countdown_does_not_replace_controlled_causal_identification
  expected_outputs:
  - RUN_COMPLETE.json
  - terminal_audit.json
  - arena_summary.csv
  - negative_budget_calibration.json
  - offline_bank_manifest
  - methods_each_dynamic_diagnostics_jsonl
  orchestration:
    gpu_selection_default: auto_all_visible_up_to_8
    method_training: one_method_per_gpu_fifo_queue
    reason_build_offline_not_sharded: preserve_one_deterministic_rng_stream_pattern_quota_and_fixed_bank_protocol
    terminal_audit_required: true
  online_successor:
    status: not_registered
    may_start_only_after: V4.4_offline_pilot_audited_and_delivered
    must_freeze_separately:
    - rollout_actor_sync_lag
    - replay_age_and_retention
    - online_seeds
    - fresh_vs_stale_negative_budget_matching
  formal_run_status: not_run
  execution:
    state: registered
    run_id: null
    last_heartbeat_utc: null
    process_exit_code: null
  evidence:
    raw_complete: false
    terminal_audited: false
    package_created: false
    package_filename: null
    package_sha256: null
    delivered_to_user: false
    applied_commit: null
- id: EXT-C-E8-V4.5-OFFLINE-BANK-TUNING
  environment: EXT-C
  name: countdown_offline_bank_alpha_lambda_validation_tuning_pilot
  status: not_run
  claim: Test whether the V4.4 ordering from uncontrolled to initialization-matched global scaling to dynamic remoteness control
    stopped near Positive-only because the calibrated overall negative strength or exponential taper rate was suboptimal,
    and whether a validation-selected configuration improves held-out verifier outcomes without valid-rate or numerical degradation.
  role: external_validity_offline_bank_control_hyperparameter_diagnostic
  execution_class: pilot
  registration_base_commit: 58342ae7809354ef8af0e90a1d9938aa51f3a97d
  predecessor: EXT-C-E8-V4.4-OFFLINE-BANK
  pilot_execution:
    channel_ref: hardened-v1
    launch_mode: guarded_orchestrator
    operator_entrypoint: scripts/run_countdown_v45_tuning.py
    guard_required: true
  does_not_replace:
  - D-U1
  - D-Diag
  code_entrypoint: src/drpo/countdown_qwen_arena_onefile.py
  one_click_entrypoint: scripts/run_countdown_v45_tuning.py
  primary_model: Qwen2.5-0.5B-Instruct
  parameterization:
    runner_version: 4.6.0-offline-negative-bank-tuning-support
    main_comparison: bf16_lora
    predecessor_reference_adapter_reused: true
    predecessor_split_and_bank_reused: true
    online_rollout_during_training: false
  predecessor_requirements:
    experiment_id: EXT-C-E8-V4.4-OFFLINE-BANK
    terminal_audit_required: true
    frozen_inputs:
    - reference_adapter
    - train_validation_test_split
    - offline_6000_rows
    - fixed_16_negative_bank_per_prompt
    - initial_gradient_calibration
    input_hashes_checked_before_and_after_run: true
  capability_gates:
    inherited_mechanism_gate:
      greedy_success_min: 0.08
      valid_rate_min: 0.95
    formal_method_ranking:
      greedy_success_min: 0.15
      valid_rate_min: 0.95
      failure_action: Complete and report the registered tuning pilot, but prohibit a formal method ranking or significance
        claim when the inherited reference remains below gate.
  tuning_protocol:
    test_split_access: only_after_all_validation_selection_is_frozen
    stage_a_global_negative_strength:
      parameter: negative_scale_multiplier
      interpretation: calibrated_bank_negative_scale_times_multiplier
      values:
      - 0.5
      - 1.0
      - 1.5
      - 2.0
      fixed_taper_lambda: 0.7
      tuning_seeds:
      - 1234
      - 2234
    stage_b_exponential_taper:
      parameter: taper_lambda
      values:
      - 0.3
      - 0.7
      - 1.2
      global_negative_strength: selected_from_stage_a
      tuning_seeds:
      - 1234
      - 2234
    selection_order:
    - mean_best_validation_greedy_success
    - mean_best_validation_pass_at_k
    - mean_terminal_validation_greedy_success
    - mean_best_validation_valid_rate
    - conservative_tie_break
    candidate_valid_rate_floor: 0.95
    numerical_failure_disqualifies_candidate: true
    threshold_frozen: 2.0
  confirmation_protocol:
    untouched_training_seeds:
    - 3234
    - 4234
    - 5234
    methods:
    - positive_only
    - validation_selected_bank_dynamic
    checkpoints:
    - best
    - terminal
    paired_evaluation_seed_per_training_seed: true
    test_metrics:
    - greedy_verifier_success
    - pass_at_k
    - valid_rate
    - heldout_pattern_family_coverage
    - heldout_pattern_family_precision_micro
  frozen_controls:
    negative_bank_size_per_prompt: 16
    current_policy_min_max_bank_selection: true
    near_mix: 0.5
    far_mix: 0.5
    maximum_effective_epochs: 6
    minimum_effective_epochs_before_early_stop: 2
    evaluation_interval_effective_epochs: 1
    early_stop_patience_evaluations: 2
    early_stop_delta: 0.002
    learning_rate: 5.0e-05
    warmup_ratio: 0.03
    maximum_gradient_norm: 1.0
  reporting_separation:
  - task_performance_degradation
  - support_or_structure_boundary_event
  - nan_inf_numerical_failure
  interpretation_limits:
  - hyperparameters_selected_on_validation_only
  - test_not_used_for_selection
  - multi_seed_0_5b_pilot_is_not_automatic_formal_ranking
  - no_state_distribution_ood_claim
  - no_online_off_policy_claim
  - no_claim_that_tuning_repairs_directional_utility
  - countdown_does_not_replace_controlled_causal_identification
  expected_outputs:
  - RUN_COMPLETE.json
  - terminal_audit.json
  - arena_summary.csv
  - alpha_selection.json
  - lambda_selection.json
  - run_config.json
  formal_run_status: not_run
  execution:
    state: registered
    run_id: null
    last_heartbeat_utc: null
    process_exit_code: null
  evidence:
    raw_complete: false
    terminal_audited: false
    package_created: false
    package_filename: null
    package_sha256: null
    delivered_to_user: false
    applied_commit: null
- id: EXT-C-E8-V4.6-ONLINE-OFFPOLICY-REPLAY
  environment: EXT-C
  name: countdown_online_offpolicy_replay_two_by_two_pilot
  status: not_run
  claim: Test whether policy-refreshed replay changes Countdown outcomes relative to the frozen V4.4 bank, and whether the
    V4.5-selected controlled negative update adds value beyond online Positive-only when the learner continuously mixes fresh
    collector data with stale replay from older policy versions.
  role: external_validity_online_offpolicy_replay_and_negative_utility_diagnostic
  execution_class: pilot
  registration_base_commit: 7dcde2095e0f0aa4a7302a829667c1955c187738
  predecessor: EXT-C-E8-V4.5-OFFLINE-BANK-TUNING
  frozen_source: EXT-C-E8-V4.4-OFFLINE-BANK
  pilot_execution:
    channel_ref: hardened-v1
    launch_mode: guarded_orchestrator
    operator_entrypoint: scripts/run_countdown_v46_online_replay.py
    guard_required: true
  does_not_replace:
  - D-U1
  - D-Diag
  code_entrypoint: scripts/run_countdown_v46_online_replay.py
  one_click_entrypoint: scripts/run_countdown_v46_online_replay.py
  primary_model: Qwen2.5-0.5B-Instruct
  parameterization:
    runner_version: v4.6-online-offpolicy-replay
    main_comparison: bf16_lora
    reference_adapter_reused: true
    train_validation_test_split_reused: true
    selected_alpha_lambda_inherited_from_v45: true
    optimizer_state_continuous_across_collection_phases: true
    scheduler_state_continuous_across_collection_phases: true
  predecessor_requirements:
    experiment_id: EXT-C-E8-V4.5-OFFLINE-BANK-TUNING
    terminal_audit_required: true
    selected_alpha_lambda_required: true
    v44_source_inputs_recovered_from_v45_run_config: true
    input_hashes_checked_before_and_after_run: true
  design:
    factorial: 2x2_data_refresh_by_negative_update
    cells:
    - frozen_positive_only
    - frozen_dynamic
    - online_positive_only
    - online_dynamic
    frozen_dynamic_controller: v45_selected_alpha_lambda
    online_dynamic_controller: same_v45_selected_alpha_lambda
    optimizer_update_budget_matched_across_cells: true
    test_split_access: only_after_all_training_cells_finish
  online_replay_protocol:
    collection_phases: 4
    refresh_rows_per_phase: 1000
    rollouts_per_prompt_per_attempt: 12
    resample_rounds: 4
    generated_negative_bank_only: true
    synthetic_negative_fallback: false
    negative_bank_size_per_prompt: 16
    positive_branch: generated_correct_same_oracle_structure_else_oracle_fallback
    replay_window_collector_versions: 3
    warmup_phase_stale_fraction: 0.0
    post_warmup_fresh_fraction: 0.5
    post_warmup_stale_fraction: 0.5
    exact_microbatch_mix_after_warmup:
      fresh: 4
      stale: 4
    collector_policy_digest_recorded_each_phase: true
    replay_age_and_collector_round_recorded_per_row: true
    current_policy_bank_min_max_reselection_during_training: true
  confirmation_protocol:
    training_seeds:
    - 6234
    - 7234
    - 8234
    checkpoints:
    - best
    - terminal
    paired_evaluation_seed_per_training_seed: true
    paired_effects:
    - online_refresh_effect_with_positive_only
    - online_refresh_effect_with_dynamic_negative
    - negative_effect_with_frozen_data
    - negative_effect_with_online_replay
    - refresh_by_negative_interaction
    test_metrics:
    - greedy_verifier_success
    - pass_at_k
    - valid_rate
    - heldout_pattern_family_coverage
    - heldout_pattern_family_precision_micro
  frozen_controls:
    selected_alpha: inherited_from_v45_RUN_COMPLETE
    selected_lambda: inherited_from_v45_RUN_COMPLETE
    surprisal_threshold: 2.0
    near_mix: 0.5
    far_mix: 0.5
    maximum_optimizer_updates: matched_to_v45_six_effective_epochs
    learning_rate: 5.0e-05
    warmup_ratio: 0.03
    maximum_gradient_norm: 1.0
    gradient_accumulation_microbatches: 8
  mechanism_diagnostics:
  - actual_bank_selected_near_far_surprisal
  - actual_bank_selected_near_far_gradient_norm
  - actual_bank_selected_positive_negative_cosine
  - collector_policy_version_and_replay_age
  - generated_positive_fraction
  - dynamic_near_far_taper_weights
  reporting_separation:
  - task_performance_degradation
  - support_or_structure_boundary_event
  - nan_inf_numerical_failure
  interpretation_limits:
  - online_data_refresh_and_negative_update_are_separated_by_factorial_cells
  - method_specific_collectors_make_online_cells_closed_loop_end_to_end_effects
  - multi_seed_0_5b_pilot_is_not_automatic_formal_ranking
  - no_state_distribution_ood_claim
  - countdown_does_not_replace_controlled_causal_identification
  - no_claim_that_online_refresh_must_make_negative_gradients_useful
  expected_outputs:
  - RUN_COMPLETE.json
  - terminal_audit.json
  - arena_summary.csv
  - run_config.json
  - training/online_positive_only/seed_*/replay/round_*.jsonl
  - training/online_dynamic/seed_*/replay/round_*.jsonl
  - training/*/seed_*/dynamic_diagnostics.jsonl
  formal_run_status: not_run
  execution:
    state: registered
    run_id: null
    last_heartbeat_utc: null
    process_exit_code: null
  evidence:
    raw_complete: false
    terminal_audited: false
    package_created: false
    package_filename: null
    package_sha256: null
    delivered_to_user: false
    applied_commit: null
- id: EXT-C-E8-TAPER-0.5B-01
  environment: EXT-C
  name: countdown_continuous_surprisal_taper_common_replay_pilot
  status: not_run
  claim: Under a common fixed sample-level negative replay pool, matched model and optimizer budget, and initialization-matched
    raw negative-gradient L2, test whether continuous learner-relative surprisal taper functions allocate negative influence
    differently and whether that selective allocation improves Countdown task performance or stability relative to Positive-only,
    uncontrolled negative updates, and a non-selective global control.
  role: external_validity_continuous_taper_method_pilot
  execution_class: pilot
  registration_base_commit: 5ee79b4245a52af3f8caea6b1dd27e3efcd6920a
  context_predecessor: EXT-C-E8-V4.6-ONLINE-OFFPOLICY-REPLAY
  does_not_replace:
  - C-U1
  - D-U1
  - D-Diag
  implementation_state: implemented
  execution_gate:
    state: ready
    blocked_by: []
    blocking_reason: null
    readiness_basis: Corrected normalized-distance implementation, active-tail nondegenerate calibration guard, deterministic
      detached remoteness, common replay, independent calibration, paired sampler, fixed-budget training, streamed current-surprisal
      diagnostics, hardened artifact identity checks, one-click launch, and terminal audit are implemented and tested.
  code_entrypoint: src/drpo/countdown_e8_taper.py
  operator_entrypoint: scripts/run_countdown_e8_taper.py
  primary_model: Qwen2.5-0.5B-Instruct
  scope_decision:
    countdown_0_5b_mechanism_exploration: closed_for_current_scope
    closure_is_result_upgrade: false
    mechanism_owner_remains_controlled_environments:
    - C-U1
    - D-U1
    v46_remains_separate_effect_pilot: true
  common_replay_protocol:
    collector_policy: frozen_reference_adapter
    prompts: same_frozen_countdown_prompt_source_across_methods
    generated_candidates: fixed_generation_budget_shared_across_methods
    retain_candidate_when:
    - valid_expression_format
    - uses_all_supplied_numbers_exactly_once
    - verifier_incorrect
    - unique_within_prompt
    storage: sample_level_negative_replay_pool
    fixed_negative_count_per_prompt: false
    synthetic_negative_fallback: false
    prompt_balanced_sampling:
      first_stage: uniform_over_eligible_prompts
      second_stage: sample_one_negative_completion_within_prompt
      purpose: prevent_candidate_rich_prompts_from_receiving_more_training_weight
    paired_sampler_order_across_methods: required
    replay_pool_hash_recorded: required
    final_train_prompt_rows: 900
    final_calibration_prompt_rows: 16
    collector_only_candidate_reserve:
      train: 1800
      calibration: 32
  learner_relative_remoteness:
    definition: normalized_excess_surprisal_then_square_root_distance
    normalized_excess_formula: S_theta=max(0,(-log_pi_theta(x_given_s)-tau)/c_cal)
    distance_formula: d_theta=sqrt(S_theta)
    surprisal_threshold_tau_rule: calibration_common_half_median_surprisal
    surprisal_threshold_tau: resolved_from_independent_calibration_common_half_median
    surprisal_scale_rule: calibration_upper_half_median_minus_lower_half_median
    surprisal_scale_frozen_before_confirmation: true
    recomputed_by_current_learner_each_update: true
    deterministic_eval_mode_for_weight_coordinate: true
    weight_stop_gradient: true
    permanent_near_far_labels: forbidden
    formula: d_theta=max(0,-log_pi_theta(x_given_s)-tau) [deprecated_v67_raw_excess_not_distance]
  methods:
  - id: positive_only
    weight_function: zero
    role: stable_reference_without_negative_updates
  - id: uncontrolled_negative
    weight_function: one
    role: negative_pressure_stress_reference
  - id: global_matched
    weight_function: constant_gamma
    role: non_selective_budget_matched_control
  - id: reciprocal_linear
    weight_function: 1_over_1_plus_lambda_d
    role: slow_continuous_tail_decay
  - id: exponential
    weight_function: exp_minus_lambda_d
    role: exponential_tail_decay
  - id: squared_distance_exponential
    weight_function: exp_minus_lambda_d_squared
    role: stronger_high_surprisal_tail_decay
  excluded_primary_methods:
  - reciprocal_quadratic_without_separate_registration
  - sbrc
  - hybrid
  calibration_protocol:
    split: independent_calibration_split
    development_seed: 9134
    target: initialization_raw_negative_gradient_l2
    common_target_budget: corrected_exponential_linear_distance_lambda_0.7_initialization_gradient_l2_under_active_tail_tau
    parameters_calibrated:
    - global_gamma
    - taper_lambda
    test_metric_used_for_calibration: false
    freeze_before_confirmation: required
    confirmation_or_test_retuning: forbidden
    inherited_reference_method: exponential
    inherited_reference_lambda: 0.7
    surprisal_threshold_tau_rule: calibration_common_half_median_surprisal
    surprisal_threshold_tau: resolved_from_independent_calibration_common_half_median
    surprisal_scale_rule: calibration_upper_half_median_minus_lower_half_median
    active_distance_minimum_fraction: 0.25
    nondegenerate_target_max_ratio: 0.995
    minimum_taper_lambda: 1.0e-06
    shared_negative_scale: positive_aggregate_gradient_l2_over_uncontrolled_negative_aggregate_gradient_l2
    dropout_mode: disabled_during_calibration_gradient_measurement
    degenerate_calibration_policy: fail_closed_before_method_training
  confirmation_protocol:
    paired_training_seeds:
    - 9234
    - 10234
    - 11234
    checkpoints:
    - best
    - terminal
    paired_evaluation_seed_per_training_seed: true
    test_split_access: only_after_all_training_and_selection_complete
    unbounded_hyperparameter_search: forbidden
  inherited_training_controls:
    reference_adapter: same_across_all_methods
    data_split: same_as_v46
    optimizer: AdamW
    learning_rate: 5.0e-05
    warmup_ratio: 0.03
    maximum_gradient_norm: 1.0
    gradient_accumulation_microbatches: 8
    optimizer_update_budget: 1200
    fixed_horizon_is_not_automatic_convergence: true
    test_split_access: only_after_all_method_training_and_selection_complete
  task_metrics:
  - greedy_verifier_success
  - pass_at_k
  - valid_rate
  - best_validation_checkpoint
  - terminal_checkpoint
  continuous_diagnostics:
    binning_axis: current_sequence_surprisal_quantiles
    fresh_current_learner_recomputation: required
    metrics:
    - samples_per_bin
    - raw_negative_gradient_norm
    - weighted_negative_gradient_norm
    - actual_taper_weight
    - positive_negative_gradient_cosine
    - correct_completion_collateral_effect
    - fraction_of_total_negative_gradient_budget
    deterministic_teacher_forced_audit: true
    teacher_forced_batch_size: 1
    gradient_batch_size: 1
    diagnostic_oom_policy: preserve_metrics_and_mark_incomplete_oom
  terminal_audit:
    required: true
    full_training_curves: required
    best_terminal_gap: required
    final_window_slopes: required
    last_finite_checkpoint_on_failure: required
    finite_loss_gradient_parameter_check: required
    valid_support_entropy_boundary_check: required
    fixed_horizon_is_not_automatic_convergence: true
    valid_structure_boundary_rate: 0.95
    valid_structure_boundary_threshold_source: inherited_reference_valid_rate_gate
    entropy_boundary_reporting: continuous_metric_without_separate_binary_threshold
  reporting_separation:
  - task_performance_degradation
  - valid_support_or_entropy_boundary_event
  - nan_inf_numerical_failure
  decision_rules:
    taper_beats_positive_only: Requires paired improvement, no terminal reversal, no material valid-rate degradation, and
      no additional support or numerical failure.
    taper_only_beats_uncontrolled_or_global: Supports far-tail harm control but not extra task utility beyond Positive-only.
    taper_methods_tie: Select the simplest theoretically clear tail envelope for scale validation without claiming a universal
      0.5B winner.
    all_negative_methods_lose: Close the 0.5B method-utility route and carry only one frozen simple taper into 3B confirmation
      without additional 0.5B HPO.
  interpretation_limits:
  - external_method_pilot_not_controlled_causal_identification
  - no_universal_taper_winner_claim
  - no_assumption_that_negative_updates_beat_positive_only
  - no_0_5b_to_3b_or_7b_automatic_generalization
  - countdown_does_not_replace_cu1_or_du1
  expected_outputs:
  - RUN_COMPLETE.json
  - terminal_audit.json
  - arena_summary.csv
  - taper_calibration.json
  - replay_pool_manifest.json
  - surprisal_bin_diagnostics.csv
  - run_config.json
  - scientific_run_manifest.json
  formal_run_status: not_run
  execution:
    state: registered
    run_id: null
    last_heartbeat_utc: null
    process_exit_code: null
  evidence:
    code_committed: false
    run_started: false
    raw_complete: false
    terminal_audited: false
    package_created: false
    package_filename: null
    package_sha256: null
    delivered_to_user: false
    applied_commit: null
    implementation_complete: true
    implementation_tests_passed: true
    real_qwen_cuda_run_completed: false
  config_entrypoint: configs/countdown_e8_taper_0p5b.yaml
  implementation_version: 1.2.1-minimal-active-tail-streamed-diagnostics
  artifact_identity_guards:
    config_hash_match_calibration: required
    reference_adapter_hash_match_calibration: required
    sampler_experiment_id_seed_replay_and_plan_hash_match: required
    nondegenerate_calibration_guard: required
    method_byte_identity_sanity_gate: required_for_operator_sanity_before_rerun
- id: EXT-C-E8-SCALE-01
  execution_gate:
    state: blocked
    blocked_by:
    - EXT-C-E8-TAPER-0.5B-01
    - EXT-H-E7-BENCH-01
    blocking_reason: The audited 0.5B continuous-taper pilot and the continuous external benchmark must be delivered before
      model/data scaling uses a frozen method shortlist.
  environment: EXT-C
  name: countdown_large_model_large_data_external_benchmark
  status: not_run
  parent_experiment: E8-SCALE
  registration_base_commit: f64452a7452274a183b03c87c39b847039230c00
  claim: Validate the frozen controller shortlist on a larger fixed Countdown dataset and larger Transformer models, separating
    mechanism transfer from scalable task-performance evidence.
  role: external_large_scale_discrete_benchmark
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
  scaling_plan:
    mechanism_owner: EXT-C-E8-V4.6-ONLINE-OFFPOLICY-REPLAY
    method_shortlist_owner: EXT-C-E8-TAPER-0.5B-01
    primary_model: Qwen_Instruct_3B
    frozen_confirmation_model: Qwen_Instruct_7B
    dataset: larger_fixed_offline_countdown_dataset
    exact_sizes_and_seeds: pending_before_implementation
    retune_method_family_on_scale_tasks: false
  candidate_methods: frozen_shortlist_from_controlled_and_E7_results
  reporting_separation:
  - task_performance_collapse
  - support_or_entropy_boundary
  - nan_inf_numerical_failure
  evidence:
    code_committed: false
    run_started: false
    raw_complete: false
    terminal_audited: false
    package_created: false
    scientific_status: not_run
