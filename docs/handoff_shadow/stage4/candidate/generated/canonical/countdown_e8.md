# Countdown Transformer external validation E8

> Stage 4B lossless source-promotion shadow candidate.
> Generated from `docs/handoff.md`; do not edit while the manual handoff remains authoritative.

- Owner type: `canonical_module`
- Owner ID: `countdown_e8`
- Responsibility: Cover token-level near or far mechanism probes and fixed-offline-bank method pilots without replacing D-U1 controlled identification.
- Dependencies: `global_core_governance`, `execution_status_gates`, `theory_methods_related_work`, `terminal_audit`, `categorical_e5_mechanism`, `categorical_e6_generalization`
- Content-contract topics: none
- Owned source blocks: 5
- Registry references are pointers only; `experiments/registry.yaml` remains the sole editable registry source.

## Registry references

- `experiments`: `EXT-C-E8-V4`, `EXT-C-E8-V4.1`, `EXT-C-E8-V4.2`, `EXT-C-E8-V4.3`, `EXT-C-E8-V4.4-OFFLINE-BANK`, `EXT-C-E8-V4.5-OFFLINE-BANK-TUNING`, `EXT-C-E8-V4.6-ONLINE-OFFPOLICY-REPLAY`, `EXT-C-E8-TAPER-0.5B-01`, `EXT-C-E8-SCALE-01`
- `development_experiment_registrations`: none

## Owned source blocks

<!-- STAGE4B-SOURCE-BLOCK:B000004:START -->
<!-- HANDOFF-DELTA-BLOCK:after_heading:v52-ext-c-e8-v43-dynamic-control:START -->
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
<!-- HANDOFF-DELTA-BLOCK:after_heading:v52-ext-c-e8-v43-dynamic-control:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000004:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000005:START -->
<!-- HANDOFF-DELTA-BLOCK:after_heading:v53-stage3-observation-automation:START -->
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
<!-- HANDOFF-DELTA-BLOCK:after_heading:v53-stage3-observation-automation:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000005:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000018:START -->
<!-- HANDOFF-DELTA-BLOCK:after_heading:v67-countdown-0p5b-mechanism-close-e8-taper:START -->
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
<!-- HANDOFF-DELTA-BLOCK:after_heading:v67-countdown-0p5b-mechanism-close-e8-taper:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000018:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000074:START -->
<!-- HANDOFF-DELTA-BLOCK:section_end:v52-e8-route-override:START -->
7. **v52 路线覆盖：** 上述第 5 项的当前 E8-MECH owner 更新为 `EXT-C-E8-V4.3`。V4.3 只修复长期训练中的动态 remoteness 控制并保留 V4.2 静态方法作消融；E8-SCALE 的 3B/7B 规模结论仍需后续独立执行。
<!-- HANDOFF-DELTA-BLOCK:section_end:v52-e8-route-override:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000074:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000075:START -->
<!-- HANDOFF-DELTA-BLOCK:section_end:v57-e8-offline-online-route:START -->
8. **v57 E8 内部路线覆盖：** 在进入 E8 外部诊断时，先执行 `EXT-C-E8-V4.4-OFFLINE-BANK`，只改变 fixed-bank 密度与每步动态选择；online off-policy 必须作为独立 successor 重新冻结 rollout actor、同步滞后、replay age、seeds 与预算匹配，不能与 V4.4 共用结论。
<!-- HANDOFF-DELTA-BLOCK:section_end:v57-e8-offline-online-route:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000075:END -->
