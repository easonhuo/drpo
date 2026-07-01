# Countdown Transformer external validation E8

> Stage 4B lossless source-promotion shadow candidate.
> Generated from `docs/handoff.md`; do not edit while the manual handoff remains authoritative.

- Owner type: `canonical_module`
- Owner ID: `countdown_e8`
- Responsibility: Cover token-level near or far mechanism probes and fixed-offline-bank method pilots without replacing D-U1 controlled identification.
- Dependencies: `global_core_governance`, `execution_status_gates`, `theory_methods_related_work`, `terminal_audit`, `categorical_e5_mechanism`, `categorical_e6_generalization`
- Content-contract topics: none
- Owned source blocks: 4
- Registry references are pointers only; `experiments/registry.yaml` remains the sole editable registry source.

## Registry references

- `experiments`: `EXT-C-E8-V4`, `EXT-C-E8-V4.1`, `EXT-C-E8-V4.2`, `EXT-C-E8-V4.3`, `EXT-C-E8-V4.4-OFFLINE-BANK`, `EXT-C-E8-V4.5-OFFLINE-BANK-TUNING`, `EXT-C-E8-V4.6-ONLINE-OFFPOLICY-REPLAY`, `EXT-C-E8-SCALE-01`
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
<!-- STAGE4B-SOURCE-BLOCK:B000072:START -->
<!-- HANDOFF-DELTA-BLOCK:section_end:v52-e8-route-override:START -->
7. **v52 路线覆盖：** 上述第 5 项的当前 E8-MECH owner 更新为 `EXT-C-E8-V4.3`。V4.3 只修复长期训练中的动态 remoteness 控制并保留 V4.2 静态方法作消融；E8-SCALE 的 3B/7B 规模结论仍需后续独立执行。
<!-- HANDOFF-DELTA-BLOCK:section_end:v52-e8-route-override:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000072:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000073:START -->
<!-- HANDOFF-DELTA-BLOCK:section_end:v57-e8-offline-online-route:START -->
8. **v57 E8 内部路线覆盖：** 在进入 E8 外部诊断时，先执行 `EXT-C-E8-V4.4-OFFLINE-BANK`，只改变 fixed-bank 密度与每步动态选择；online off-policy 必须作为独立 successor 重新冻结 rollout actor、同步滞后、replay age、seeds 与预算匹配，不能与 V4.4 共用结论。
<!-- HANDOFF-DELTA-BLOCK:section_end:v57-e8-offline-online-route:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000073:END -->
