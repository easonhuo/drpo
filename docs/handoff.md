# DRPO / SNA2C 远场负梯度动力学研究主文档 v80（Dev-Review 协作流程与 GLM 执行角色固化版）
<!-- HANDOFF-DELTA-BLOCK:after_heading:v50-stage3-shadow-bootstrap:START -->
> **v50 增量登记：治理 Pipeline Stage 3 `HANDOFF_DELTA.yaml` shadow mode 启动（不删除 v49 及更早内容）**
>
> - Stage 1 与当前 Stage 2 的 `closed_maintenance_only` 状态保持不变；本版只启动当前 Stage 3，不改变任何科学实验变量、seeds、阈值、结果或执行顺序。`D-U1-E6-CONDITIONAL-GAP-01` 继续保持 **not_run + implemented + ready + active**。
> - Stage 3 状态由 `ready_not_started` 迁移为 `shadow_active`。`docs/handoff.md` 在整个 shadow 期继续是唯一权威研究 Master；结构化 delta 只生成 candidate 并与人工 handoff 比较，不得替换正式 handoff。
> - 新增 `docs/handoff_delta_protocol.md`、机器策略 `docs/handoff_delta_policy.yaml`、状态机 `docs/handoff_delta_state_machines.yaml`、确定性入口 `scripts/handoff_delta_shadow.py` 与三级验收入口 `scripts/run_handoff_delta_acceptance.py`。版本 1 只允许 heading rename、heading 后插入和 section 末尾 append，不允许任意文本替换或破坏性删除。
> - Fast Gate 对每个 handoff / registry / delta 相关更新执行本地确定性 replay、base/hash、幂等、历史 ID、registry transition 与 candidate/manual exact-match 检查；禁止网络和 LLM 作为阻塞式 oracle。目标 p95 不超过 5 秒，硬上限 15 秒。
> - Standard Regression 在 schema、renderer、状态机、冲突规则、parser/index 或 operation 变化时运行，目标 60 秒。Full Acceptance 在 shadow 激活前、authority cutover 前、schema 主版本或架构变化、累计 20 次相关更新、7 天内发生相关更新的兜底周期、critical mismatch 修复后运行，目标 15 分钟。
> - 本更新使用 `GOV-STAGE3-SHADOW-BOOTSTRAP-2026-06-27/HANDOFF_DELTA.yaml` 自举 replay；candidate 与人工 v50 handoff 必须字节级一致。该自举通过只证明实现与门禁可运行，不等于 Stage 3 已验收或可以切换权威路径。
<!-- HANDOFF-DELTA-BLOCK:after_heading:v50-stage3-shadow-bootstrap:END -->
<!-- HANDOFF-DELTA-BLOCK:after_heading:v51-du1-e6-semantic-gap-formal:START -->
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
<!-- HANDOFF-DELTA-BLOCK:after_heading:v51-du1-e6-semantic-gap-formal:END -->
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
<!-- HANDOFF-DELTA-BLOCK:after_heading:v55-du1-e6-semantic-gap-result-closure:START -->
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
<!-- HANDOFF-DELTA-BLOCK:after_heading:v55-du1-e6-semantic-gap-result-closure:END -->
<!-- HANDOFF-DELTA-BLOCK:after_heading:v56-e6-parent-closure-route-release:START -->
> **v56 增量登记：E6 父 claim 关闭与 E7-MECH 路线解锁（不删除 v55 及更早内容）**
>
> - 用户在确认 `main` commit `e70f0d84256cdeb6ebbf80b0495a043582787bf6` 已提交后，批准对 **E6 父实验/父 claim** 做范围受限关闭。关闭依据是：`D-U1-E6-SEMANTIC-LONGRUN-01` 的 `360/360` long-run validated 主结果、`D-U1-E6-SEMANTIC-GAP-LONGRUN-01` 的 `100/100` finite-step robustness successor，以及 `D-U1-E6-CONDITIONAL-GAP-01` 的 `200/200` finite-step stress diagnostic。
> - 本次关闭锁定五项论文可用结论：Positive-only 在共享语义 categorical 环境存在 imitation ceiling；适度受控负信号可改善同分布 held-out-context / unseen-action 表现；过强或不抑制负压力会出现性能反转或随 horizon 扩大的退化；semantic alignment 是观察到的未见动作迁移的重要排他性条件；任务性能崩溃、support/temperature boundary 与 NaN/Inf 必须继续分报。
> - 本次关闭不把两个 gap 子实验升级为 long-run validated，也不声称全 alpha 稳态排名、universal alpha optimum、state-distribution OOD generalization、categorical policy 的 Gaussian 二次远场律或跨任务方法优越性。`45/100` semantic-gap plateau 与 `49/200` conditional-gap plateau 的终态边界原样保留。
> - `D-U1-E6-TAPER-01` 降为**可选、独立、非门禁**的方法形状比较：它仍是 `not_run + not_implemented + blocked`，若未来执行，必须另行冻结 semantic remoteness coordinate、paired protocol、全新 untouched seeds 与独立 runner；但它不再是 E6 父 claim 关闭或 E7-MECH 启动的前置条件。
> - `EXT-H-E7-Q2` 由 `blocked/blocked` 迁移为 **ready/active**，科学状态仍为 `not_run`。该迁移只开放已经冻结和实现的 Hopper mechanism formal protocol，不代表 E7 已运行或已有结果。`EXT-H-E7-BENCH-01` 继续 blocked，但依赖收缩为 E7-Q2 交付和随后冻结 shortlist，不再依赖可选 E6-TAPER。
> - 本更新只修改研究治理、路线和相应测试/操作说明；未重跑实验，未更改任何冻结变量、数据规模、seeds、阈值、收敛标准或方法职责。
<!-- HANDOFF-DELTA-BLOCK:after_heading:v56-e6-parent-closure-route-release:END -->
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
<!-- HANDOFF-DELTA-BLOCK:after_heading:v60-e4-taper-utility-registration:START -->
> **v60 增量登记：E4-TAPER 负样本净效用理论与四项公平/终态实验路线（不删除 v59 及更早内容）**
>
> - `C-U1-E4-TAPER-01` 的 220/220 正式结果和 **finite-step validated** 状态保持不变；本版不重跑、不延长原 8000-step protocol，也不把有限步排序升级为长期或普遍方法排名。
> - 正式引入负样本 alignment utility、orthogonal nuisance cost 与 net utility。条件经验假设只要求：离开局部信息区后，负样本净效用随 policy-relative distance 总体不增，并可能趋零或转负；**不假设效用按指数速度下降**，也不把该关系声明为普遍定理。
> - 澄清 Quadratic 与 Exponential 的理论职责：Quadratic 权重本身趋零，但与 learnable-log-scale 的 `Theta(d^2)` 原始影响相乘后一般只得到 bounded nonzero influence；Exponential 或任何 `o(d^-2)` 尾部进一步保证 vanishing influence。Quadratic 是最低充分有界阶，Exponential 是平滑 vanishing-tail 候选而非唯一解。
> - 当前 E4 历史公式 `exp(-lambda*u)` 不变。`exp(-beta*u^2)` 仅登记为近场一阶导数为零、远场指数趋零的候选，必须在新实验 protocol freeze 中显式批准，不能追溯性替换旧结果。
> - 用户批准登记四项后续：`C-U1-E4-TAPER-NEAR-RETENTION-01`、`C-U1-E4-TAPER-BUDGET-MATCH-01`、`C-U1-E4-TAPER-CONV-01`、`C-U1-E4-TAPER-CONFIRM-01`。四项当前均为 **not_run + not_implemented + blocked**，不得因登记而直接启动。
> - E4-TAPER 内部顺序冻结为 near-retention matching -> negative-budget matching -> long-run terminal resolution -> untouched-seed confirmation。Long-run 继续推迟到前两项冻结方法公式和超参数之后；几何 robustness extension 保持低优先级、非当前门禁。
> - 本更新只修改理论、registry、Stage 3 delta 与治理测试；没有运行新的科学实验，也不预设 Linear、Quadratic、Exponential、Global alpha 或 squared-distance exponential 的最终赢家。
<!-- HANDOFF-DELTA-BLOCK:after_heading:v60-e4-taper-utility-registration:END -->
<!-- HANDOFF-DELTA-BLOCK:after_heading:v61-e4-taper-near-retention-implementation:START -->
> **v61 增量登记：`C-U1-E4-TAPER-NEAR-RETENTION-01` 协议冻结、独立 runner 与执行解锁（不删除 v60 及更早内容）**
>
> - `C-U1-E4-TAPER-01` 的 220/220 结果、有限训练步数验证状态、anchor-normalized 结论与所有公平性边界保持不变；本版不重跑、不延长旧实验。
> - 第一项后续 `C-U1-E4-TAPER-NEAR-RETENTION-01` 已冻结：near 区域为 frozen 2000-step positive-only checkpoint 上的标准化距离 `d<=5`；匹配目标为 development seeds 0--4 上 pooled `E[w(d)|near]`；正式 paired seeds 为 90--109。
> - 保持率层级冻结为主层级 `0.75` 与敏感性层级 `0.50/0.25`。每个 family 只通过确定性单调二分求一个系数，系数在正式 seeds 和全部训练步中固定；formal/confirmatory seeds 严禁参与校准。
> - 候选函数冻结为 reciprocal-linear、reciprocal-quadratic、历史 current exponential `exp(-c u)` 与新批准的 squared-distance exponential `exp(-c u^2)`。后者只属于本新实验，不能追溯替换旧 E4-TAPER exponential。
> - 新增独立 formal runner `src/drpo/cu1_taper_near_retention_formal.py`，复用共享 C-U1 环境/actor 与原 positive checkpoint；报告 near useful retention、far harmful influence、全参数 far/near 比、distance-bin utility、同分布 held-out-context reward、sigma/support 和三类失效事件。
> - 本实验不匹配总负梯度预算，科学状态上限为 finite-step validated；长期 shortlist 与稳态排名继续由后续 `CONV-01` 负责。当前仅完成实现与 smoke，正式多 seed 尚未启动。
> - `BUDGET-MATCH-01`、`CONV-01`、`CONFIRM-01` 继续 blocked；只有 Near-Retention 正式结果完成终态审计、打包并交付后，才允许冻结下一项。
<!-- HANDOFF-DELTA-BLOCK:after_heading:v61-e4-taper-near-retention-implementation:END -->
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
<!-- HANDOFF-DELTA-BLOCK:after_heading:v65-paper-rewrite-outline-intro-blueprint:START -->
> **v65 增量登记：论文重写 v0.7、会话纠错账本与 Introduction 段落级施工图落库（不删除 v64 及更早内容）**
>
> - 本版只沉淀论文写作与理论证明计划，不运行新实验，不改变任何现有实验状态，也不修改任何数据集、模型、seeds、阈值、训练 horizon、终态门禁或既有实验职责；当前正式路线与执行顺序保持不变。
> - 完整的用户审阅通过版论文大纲、标题选择、理论/方法边界、术语替换、图表布局、纠错账本与下一阶段清单落在 `docs/paper_rewrite_outline_v0_7.md`；`docs/handoff.md` 继续是唯一研究状态 Master，该详细文档不得成为第二份实验状态来源。
> - 推荐标题继续为 **Breaking the Curse of Repulsion: Distributionally Robust Policy Optimization for Off-Policy Learning**。`curse` 仅指 fixed/stale far-field negatives 被重复优化时的失稳风险，不表示 negative advantage 本身天然有害。
> - 主理论使用 RL/optimization 常用表述 **stationary empirical actor objective**：在 fully offline actor analysis 中，dataset、advantage labels 与 base sample weights 固定，因此静态平衡分析精确适用；off-policy mismatch 的强弱与 objective stationarity 是不同维度。Replay、changing buffers、online collection 或 jointly evolving actor–critic systems 不在当前定理的全局保证范围内。
> - Fixed advantage 不作为 Preliminaries 中的通用 RL setting；C-U1/D-U1 只在实验设置中说明其用于隔离 critic error、advantage relabeling 和 policy-dependent weighting。当前论文不展开 dynamic-critic / moving-equilibrium 推导。
> - 术语继续优先复用主流 RL/optimization 表达：standardized/Mahalanobis distance、negative log-probability/surprisal、behavior–policy mismatch、policy-ratio clipping、simplex boundary、task-performance collapse 和 numerical failure。避免重新引入 `policy-relative remoteness`、utility radius、repulsive frontier、probability-boundary dynamics 或把 support boundary 自动称为 collapse。
> - 最终论文方法方向保持 exponential DRPO；SBRC 与 Hybrid 不进入最终论文候选。Exp 是压过有限阶 score growth 的 far-field gradient-control envelope，不是假设负样本 utility 指数衰减，也不假设真实安全半径或由 raw `p/q` 动态迁移推出。
> - Introduction 段落施工图 v0.1 落在 `docs/paper_rewrite_intro_blueprint_v0_1.md`，逐段冻结 reader question、topic sentence、supporting logic、citation targets、target length、transition、evidence status 与 forbidden overclaim。叙事顺序固定为背景 → 负更新双重作用 → fixed/stale data reuse → prior solutions → unresolved gap → Repulsive Dynamics/nonlinear policies → DRPO → evidence/contributions。
> - 当前主图规划不变：只有 Figure 1 默认双栏；Figures 2–5 默认单栏，Figure 3 为单栏 2×2，Figure 4 为单栏 2×1。D4RL、Countdown 与 Online 未完成结果只保留 `TBD`，禁止填预期数值。
> - 下一阶段顺序为：完成 Theory/Method/Experiments 的段落施工图，正式证明 equilibrium/boundary/divergence theorem 与离散 spectral-radius 条件，完成 Gaussian/categorical corollary、Exp proposition、逐条引用核验、主图视觉规范和真实受控数据重绘；Online formal protocol 必须先登记再执行。
<!-- HANDOFF-DELTA-BLOCK:after_heading:v65-paper-rewrite-outline-intro-blueprint:END -->
<!-- HANDOFF-DELTA-BLOCK:after_heading:v66-e4-taper-budget-match-closure:START -->
> **v66 增量登记：`C-U1-E4-TAPER-BUDGET-MATCH-01` 正式结果、收尾故障审计与闭环交付（不删除 v65 及更早内容）**
>
> - 正式运行绑定 clean `main` commit `1faea3a92f74af5d11409779d96b9ed21fe846ad`，使用冻结 paired seeds `110--129`、7 个条件、每个最多 8000 steps，完成 `140/140` method-seed runs。逐步 Adam 前 raw negative-gradient L2 budget 的最大相对误差为 `2.12e-16`，通过 `1e-6` 门槛；Adam parameter-update norm 仅记录、未匹配。
> - 以 Reciprocal-Linear 为 reference，Reciprocal-Quadratic、current Exponential、Squared-distance Exponential 的 held-out-context reward 配对均值差分别为 `+0.016011 / +0.088189 / +0.130616`，均为 `20/20` seeds 正差；harmful-far retention 配对差分别为 `-0.012528 / -0.053566 / -0.055866`，均为 `20/20` 更低。Non-selective Global stepwise scale 的 reward 差为 `-0.006883`（`0/20` 正差），harmful-far retention 差为 `+0.007659`（`0/20` 更低）。这支持“相同 raw negative-gradient 总预算下，选择性 taper 的远场分配而非仅总预算大小会改变有限步任务结果”。
> - Terminal near-useful retention 在非 Positive-only 方法上因 raw positive-projection denominator 为零而为 undefined/NaN，因此本实验不能独立声称 candidate 把更多预算保留给 useful-near；该部分仍由 Near-Retention predecessor 承担。当前 Budget-Match 的强证据是 harmful-far suppression 与 held-out-context reward 的 paired 一致性。
> - 三类事件严格分报：task-performance collapse `13/140`、support/variance boundary `20/140`、NaN/Inf `0/140`；前两类全部来自 unweighted boundary。所有 matched/controlled 方法三类事件均为 0。固定 8000-step horizon 不证明稳态，科学状态只能是 **有限训练步数验证**；禁止 steady-state winner、universal winner、OOD generalization、跨任务优越或“Adam update 已匹配”表述。
> - 计算本身 return code 为 0，coverage、budget 与 terminal audit 全部通过；hardened guard 在收尾阶段标记 failed，因为 runner 漏写已登记的 `scientific_run_manifest.json`，且默认 25 MiB 主包超限。该故障不改变数值输出或 provenance。原 failed guard tree 完整保留；闭环包加入 runner manifest 修复、compact repository deposition 与完整 raw sidecar，不重跑正式 seeds。
> - `C-U1-E4-TAPER-CONV-01` 继续 blocked。Budget-Match 交付后，下一动作必须是独立的 deterministic shortlist-freeze 更新，再实现 exact actor+Adam-state continuation runner；本版不提前生成 shortlist，不自动启动 Convergence。Seeds `130--149` 继续禁止访问。
<!-- HANDOFF-DELTA-BLOCK:after_heading:v66-e4-taper-budget-match-closure:END -->
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
<!-- HANDOFF-DELTA-BLOCK:after_heading:v74-du1-e6-rev4-formal-freeze:START -->
> **v74 增量登记：D-U1 E6 revision-3 开发校准闭环与 revision-4 正式协议冻结（不删除 v73 及更早内容）**
>
> - 实验 ID 继续为 `D-U1-E6-CARTESIAN-TAPER-01`。revision 2 的 observed common/rare 副本虽然实现概率笛卡尔积，但 rare 副本被压低时可能由同 reward 的 common 副本代偿，且固定几何 utility 标签可能在策略移动后失效；因此旧环境只保留为工程历史，不再承担正式方法比较。
> - revision 3 新增 16 个 evaluation-only hidden high-reward rare actions，使共享 rarity support 收缩真实降低 hidden-optimal probability 与 expected reward；同时在训练全过程用当前 expected-reward derivative sign 审计 useful/unhelpful 标签。环境失效、任务性能崩溃、支持边界和 NaN/Inf 必须分别报告。
> - development seeds `0--4` 完成用户批准的 120-run 校准：6 方法 × 5 seeds × `alpha∈{0.25,0.5}` × `anchor∈{0.25,0.1}` × `rho=0.25`，每 run 8000 steps。该证据身份仅为 **pilot / development calibration**，不得作为正式方法排名；formal seeds `200--219` 未访问。
> - 参数选择规则在方法排名之外预先固定为“已执行候选中负压力最强、同时通过环境有效性、support、数值与终态门禁的点”。据此冻结 `negative_alpha=0.5`、`rarity_logit_anchor_coefficient=0.25`、`reference_rare_retention=0.25`。`anchor=0.1` 在 `alpha=0.5` 下令 All-negative 5/5 触发 support boundary，因此被稳定性门禁排除。选择规则不以 Exp 是否第一为条件。
> - Quartic 的历史代码与开发结果保留，但因缺少独立实验职责，自 revision 4 起退出 active formal matrix。正式方法为 Positive-only、All-negative、Global matched、Reciprocal-linear、Reciprocal-quadratic、Exponential-quadratic；不得预设 Exp 或任何方法胜出。
> - revision 4 正式协议冻结为 20 个 held-out seeds `200--219` × 6 方法 × 8000 steps，共 120 runs；数据几何、优化器、阈值、双终态窗口、seed blocks 与方法公式不得在正式启动后调整。结果仍为 `not_run`，只有 guarded formal run、终态审计、持久打包与 commit 绑定完成后才能升级证据状态。
<!-- HANDOFF-DELTA-BLOCK:after_heading:v74-du1-e6-rev4-formal-freeze:END -->
<!-- HANDOFF-DELTA-BLOCK:after_heading:v76-rules-phase2-backlog:START -->
> **v76 增量登记：低成本常驻规则、Phase 2 backlog 与更新包交付硬约束（不删除 v75 及更早内容）**
>
> - 本版只登记工程/协作规则和 pipeline 二期规划，不运行实验，不修改任何实验状态、冻结变量、seeds、阈值、训练 horizon、终态门禁或科研结论。
> - DRPO 论文图、实验图、paper figure、plot、chart、panel、visualization、`画图`、`图片`、`生成图` 默认解释为代码绘图任务，优先使用 Python/Matplotlib/LaTeX/PGF/SVG/repository scripts；不得仅因出现图片相关词汇就调用 image generation。只有用户明确要求 AI 生图、艺术插图、照片编辑或风格迁移时，才允许使用 image generation；不确定时先问。
> - 在线轮询规则作为跨项目执行语义登记：用户说“轮询/在线轮询/盯着跑/一直等到结果/跑完再汇报/不要停/别报初步结果/等终态再说”等表达时，表示同一轮内阻塞式工具检查到成功、失败、卡死、用户停止或工具/会话限制；后台运行不等于在线轮询，用户下次问一次再查一次也不等于在线轮询，final 之后不得声称仍在持续监控。
> - DRPO 更新包交付硬规则：默认必须交付 canonical bundle-backed package；patch-only runnable package 不再作为正常交付格式。即使生成包时拿到最新 `main`，也不能假设用户应用前 `main` 不会被其他提交推进。除非用户明确要求临时 exact-base patch-only 立即应用，否则不得交付 patch-only runnable 包；无法生成并验证 bundle-backed 包时，只能交付方案、非运行 diff 草案或要求最新 Git bundle/diagnostic。
> - Pipeline Phase 2 backlog 正式记录：最大挑战是 handoff 拆分/模块化与上下文装配；需要维护 implemented、shadow-only、partial、paused、reverted、future backlog 等状态分类。`.drpoupdate` macOS 双击入口暂停并归入二期，只能在未来作为 old CLI 薄壳 wrapper 重新评估，且不得修改 `drpo-update` 核心语义、base mismatch、stale-package recovery 或 bundle-backed integration。
<!-- HANDOFF-DELTA-BLOCK:after_heading:v76-rules-phase2-backlog:END -->
<!-- HANDOFF-DELTA-BLOCK:after_heading:v77-minimal-diff-governance:START -->
> **v77 增量登记：最小改动治理规则正式进入执行上下文（不删除 v76 及更早内容）**
>
> - 本版只登记工程治理执行规则，不运行实验，不修改任何实验状态、冻结变量、seeds、阈值、训练 horizon、终态门禁或科研结论。
> - `docs/code_minimality_governance.md` 中的 Minimal Sufficient Diff、bug-intent scoping、Green/Yellow/Red/Split 分类和 first-failure classification 正式作为 DRPO 工程更新的常驻执行规则。
> - 当用户报告 bug、失败包、窄修复或小型代码更新时，执行者必须先用一句话锁定用户授权的开发对象，不得把最近一次失败现象、相邻工具改进或自行发明的 workflow 当作开发目标。
> - 受保护治理文件、handoff/registry authority、实验协议、seeds、thresholds、正式结果、package authority 和 updater internals 默认属于 Red；只有用户明确授权该 exact control-plane/protocol change 时才可进入相应修改。
> - 首次 package、CI 或 `drpo-update` 失败后必须先分类首个失败，不得把无关修复堆进下一版包；若闭环范围超过原窄修复，必须暂停并重新分类。
> - 由于普通 `drpo-update` content package 会拒绝直接修改 `AGENTS.md` 等 control-plane 文件，本次通过 Stage 5 schema-v3 handoff delta 的合法内容路径把规则写入每次新会话必须读取的 Section 0；不修改 `AGENTS.md`、`tools/drpo-update/`、handoff authority 或 governance ledger。
<!-- HANDOFF-DELTA-BLOCK:after_heading:v77-minimal-diff-governance:END -->
<!-- HANDOFF-DELTA-BLOCK:after_heading:v79-e8-active-tail-repair:START -->
> **v79 增量登记：Countdown E8-TAPER active-tail calibration 与诊断显存修复（不删除 v77 及更早内容）**
>
> - **旧问题：**0.5B 自然 replay 的独立 calibration split 在旧 `tau=2.0` 下几乎全部 `distance=0`，导致 inherited exponential target 与 uncontrolled negative aggregate L2 相同；`global_matched` 被校准为 1，`reciprocal_linear` 与 `squared_distance_exponential` 被校准为 0，从而与 uncontrolled 逐点重合。
> - **协议修复：**`EXT-C-E8-TAPER-0.5B-01` 保留同一 experiment ID、方法集合、训练 seeds、900/16 自然 replay、1200 update budget 与 synthetic-negative policy，但 calibration tau 改为由独立 calibration split 的 common-half median surprisal 解析，并登记 active-distance fraction、target/uncontrolled ratio 与 nondegenerate fail-closed guard。该修复不使用 validation/test 或确认 seeds 选参。
> - **实现修复：**正式收回 `_collate_pairs()` 误传 `batch_size` 的一行 hotfix；`surprisal_bin_diagnostics` 改为按小 batch 串流 full-vocab completion stats 与梯度诊断；诊断 OOM 时保留 metrics、training log 与 checkpoints，并在 manifest/terminal audit 中标记 `incomplete_oom`，不得混同为 NaN/Inf 数值崩溃。

> - **状态边界：**本次是实现与协议修复，不是科学结果；已有 dirty/local sanity 与 failed pilot 只作为 diagnostic evidence，不能入 formal result。修复应用后必须先重新执行短预算 sanity，确认方法不再 byte-identical，再运行登记的 0.5B pilot。
<!-- HANDOFF-DELTA-BLOCK:after_heading:v79-e8-active-tail-repair:END -->

> **v49 增量登记：治理 Pipeline Stage 1/2 冻结式关闭（不删除 v48 及更早内容）**
>
> - 当前规范编号锁定：Stage 1 是 bundle-backed 更新包与隔离集成闭环；当前 Stage 2 是正式实验 guard/package/verify 唯一通道；早期文档中名为 Stage 2 的 `HANDOFF_DELTA.yaml` shadow mode 顺延为当前 Stage 3。
> - Stage 1 与当前 Stage 2 均关闭为 `closed_maintenance_only`。关闭不冻结科研路线，也不禁止 bugfix、安全修复、兼容性修复或不改变职责的文档澄清；新功能、架构扩张、职责变化或默认策略变化必须通过用户批准的 reopen authorization 和回滚计划。
> - 新增机器状态 ledger `docs/governance_pipeline_stage_status.yaml`、授权目录 `docs/governance_stage_authorizations/` 与 fail-closed validator `scripts/validate_governance_pipeline_stage_status.py`。受保护核心文件采用 SHA-256 after-image 与授权记录绑定；直接改文件、只改 ledger 哈希或只改状态都会失败。
> - `D-U1-E6-SEMANTIC-LONGRUN-01` 作为 Stage 2 的首个完整生产验收：360/360 formal runs、2x terminal audit、canonical raw-complete artifact、durable delivery 与 repository closure 均已完成。registry 同步 `repository_applied: true`、`applied_commit: ff2afe443167154eae5de7871cda83f3aba9a89e`；无法恢复的首个 closure ZIP SHA 保持未记录，不得编造。
> - 当前 Stage 3 状态为 `ready_not_started`。本版不实现 handoff delta schema、合并器或 candidate 生成；Stage 3 必须由后续独立更新启动。


> **v48（D-U1 E6 大规模条件支持缺口协议与正式执行准备版）历史标题保留。**
>
> **v48 增量登记：`D-U1-E6-CONDITIONAL-GAP-01` 大规模条件支持缺口、开发 pilot 与 formal freeze（不删除 v47 及更早内容）**
>
> - 已完成的 `D-U1-E6-SEMANTIC-LONGRUN-01` 不被覆盖或否定，继续作为 **dense effective coverage / benign interpolation control**：其 360/360 长程结果证明高 reward 可与 support boundary 共存，但没有产生 task-performance collapse。该结果促使本版补充一个新的、职责独立的条件支持缺口实验，而不是重跑或修改历史结果。
> - 新实验只研究离散 contextual policy 的 **same-distribution structured state-action support gap**。训练和测试 context 仍为独立同分布样本，不存在显式 state-distribution shift，禁止称为 OOD generalization。实验问题是：一个 action group 在其他 contexts 中可见，但在一大片目标 contexts 中系统缺失时，共享策略是否把见过的 action 错误推广到不匹配 contexts，并出现任务性能崩溃。
> - 环境冻结为 8 个 semantic action groups、每组 32 个动作、共 256 个随机置换 action IDs。4096 个 train states 与 4096 个 test states 均使用 balanced paired standard-normal marginal；每对 state 除 gap indicator 所在坐标符号外完全相同，target group 由其余状态坐标决定。因此 covered/gap 比较不混入 target、nuisance 或 state marginal 差异。
> - exactly 50% states 为 gap states。每个 state 只记录 3/8 action groups，因此 62.5% conditional group blocks 缺失；在 gap states 上，完整 32-action optimal group 从 positive/local/far 全部日志角色中移除，但该 group 在其他 states 中继续出现。50% 是保持 exact paired covered/gap control 的最大平衡区域，且比旧 E6 的随机、密集语义插值形成显著更大的结构化条件缺口。
> - reward cliff 固定为：correct group scale `1.0`、proxy-positive group scale `0.65`、其余 groups（包括 trap group）为 `0.0`；within-group factor 下限 `0.85`。hidden optimal action 永不进入 positive demonstrations。固定 concentration `8.0`，本实验先隔离 task overgeneralization，不把 learnable-temperature support dynamics 混入主因果链。
> - 开发协议 `D-U1-E6-CONDITIONAL-GAP-DEV-01` 已使用 seeds `0--1`、1000 steps 完成 20/20 pilot runs；所有环境不变量通过，NaN/Inf `0/20`、support/temperature boundary `0/20`、task-performance collapse `4/20`。该 pilot 只选择 formal 条件，不是正式结果。random-policy gap reward 约 `0.1908`；structured-gap positive-only 为 `0.5152`；local `alpha=0.5` 为 `0.6363`；local `alpha=1.5` 为 `0.1859` 且 2/2 达到 task-collapse 规则。far-pressure `lambda=4.0` 下 uncontrolled 为 `0.2991`，Far-cap 与 raw-budget-matched Global 分别为 `0.4508`、`0.4908`；不得将两 seed pilot 升级为方法排名。
> - 正式协议 `D-U1-E6-CONDITIONAL-GAP-01` 经用户于 2026-06-27 批准，使用 untouched seeds `130--149`、Adam `lr=1e-3`、batch 128、8000 steps、每 50 steps 评估。正式条件固定为 paired covered control 与 structured gap 下的 positive-only、local `alpha=0.5/1.5`、uncontrolled、near-zero、Far-cap、budget-matched Global；far pressure 固定 `4.0`，不预设任何方法赢家。
> - 正式 horizon 是新 pilot 的 8x，终态窗口为 `4000--6000` 与 `6000--8000`；每 5 个 seeds 写 persistent-local recovery checkpoint。task-performance collapse 使用 random-policy 与 paired positive-only 之间的 normalized margin，并与 support/temperature boundary、NaN/Inf numerical failure 分开报告。科学失败事件是结果，不得丢弃或通过结果后改阈值修复。
> - 代码入口为 `src/drpo/du1_e6_conditional_gap.py`，formal entrypoint 为 `src/drpo/du1_e6_conditional_gap_longrun.py`，唯一推荐启动入口为 `scripts/run_du1_e6_conditional_gap_longrun.py`。应用并提交本版后，formal gate 为 **implemented + ready + active**；正式训练尚未启动。`D-U1-E6-TAPER-01` 新增 predecessor-delivery blocker，必须等待本实验终态审计、打包和交付，且原 semantic-remoteness / paired-protocol / untouched-seed / independent-runner blockers 继续存在。


> **v47（D-U1 E6 长程结果闭环与 raw-complete 包类型修复版）历史标题保留。**
>
> **v47 增量登记：`D-U1-E6-SEMANTIC-LONGRUN-01` 正式结果、仓库闭环与 raw-complete 更新器诊断（不删除 v45 及更早内容）**
>
> - 正式运行绑定 clean `main` commit `eb5e12626026854f44f4698dbc8ed8829e74e0b0`，使用 untouched held-out seeds `10--29`，完成冻结矩阵 `360/360` runs；8000-step 正式 horizon 是 4000-step development horizon 的 2x，所有逐 run terminal audit 均存在且被接受，科学状态登记为 **已长期验证（long-run validated）**。
> - 三类事件严格分报：task-performance collapse `0/360`、support/temperature boundary `120/360`、NaN/Inf numerical failure `0/360`。高 reward 与 support boundary 可以同时出现，禁止把 support boundary 写成任务崩溃或数值崩溃。
> - E6-A 固定 concentration 下，`alpha=0.25` 和 `0.50` 相对 positive-only 的 held-out-context reward 分别提高 `+0.021538` 与 `+0.015987`，均为 `20/20` paired seeds 胜出；`alpha=0.75` 虽继续增加语义外推，但 reward 下降 `-0.031028`，20/20 seeds 反向。该结果正式支持 positive-only imitation ceiling、受控 local negative 的未见动作收益以及过强压力的非单调反转。
> - E6-B 中 `far_zero` 保持 `0/20` support boundary 并比 positive-only 提高 reward，但只有 `5/20` 达到 frozen terminal plateau，`15/20` 为 persistent-drift-or-inconclusive。`uncontrolled / near_zero / far_cap / budget_matched_global` 均为 `20/20` support boundary；Far-cap 相对 uncontrolled、Global 相对 Far-cap 的 paired reward 区间均跨 0。因此不得把 Far-cap、Global alpha 或 uncontrolled 升级为安全方法赢家。
> - E6-C 只打乱 policy-side semantic alignment、保持 reward-side catalogue 与任务不变后，四个登记方法的 aligned reward 均在 `20/20` paired seeds 中高于 shuffled；aligned-minus-shuffled reward 均值为 `+0.336245` 至 `+0.372657`。该排他性控制支持共享语义对同分布 held-out contexts / unseen actions 的结构化外推作用，不是 OOD generalization，也不替代 Hopper 或 Countdown 外部有效性。
> - raw artifact `D-U1-E6-SEMANTIC-LONGRUN-01_RAW_COMPLETE.zip`（SHA-256 `e098d4dd0483a661468db0cb1c4b67e4e563e2426a6aa078fe7b808f7ac691fa`）是 `experiment-raw-complete` 恢复/证据包，其 `update.patch` 按协议允许为空，**不能直接交给 `drpo-update`**。此前更新器只报 `update patch is empty`，未解释包类型；本版让更新器先读取 `ARTIFACT_MANIFEST.json`，对 recovery kinds 给出明确操作提示，并修正自动生成的 `CHANGE_SUMMARY.md`，不再指示用户应用空 patch。
> - `outputs/du1_e6_semantic_longrun/` 保存 compact aggregate、per-run summary、paired comparisons、terminal audit、provenance 与 artifact index。正式结果闭环后，E6 long-run 禁止无新登记重跑；`D-U1-E6-TAPER-01` 的 predecessor-delivery blocker 已解除，但 semantic remoteness coordinate、paired protocol、新 untouched seeds 与独立 runner 尚未冻结/实现，仍是 **review-required, not runnable**。


> **v45（E4-TAPER 结果闭环、环境识别与公平性边界版）**
>
> **v45 增量登记：`C-U1-E4-TAPER-01` 正式结果、环境识别边界与方法公平性审计（不删除 v44 及更早内容）**
>
> - 正式运行绑定 commit `054c2e275cfd36e07e8883cb65d0b8df00460361`，成功 attempt `run_002` 完成 seeds `70--89`、11 个冻结条件，共 `220/220` runs；第一次 `run_001` 的 `SIGTERM` 失败证据单独保留且未进入科学汇总。
> - 主比较 `rho=0.25` 下，reciprocal-quadratic 相对 reciprocal-linear 在 `20/20` paired seeds 中获得更低终态 full-parameter far/near negative-gradient ratio，并在 `20/20` seeds 中获得更高 held-out-context reward。reward 均值差为 `+0.011372`，paired bootstrap 95% 区间 `[+0.010951,+0.011826]`；far/near ratio 均值差为 `-1.601377`，95% 区间 `[-1.617527,-1.586817]`。
> - 终态审计未通过全部科学门禁：`200/220` controlled/positive runs 在冻结 8000-step maximum 时仍未形成稳定候选，未进入 2x continuation；`20/220` unweighted runs 在 step 100 触发 support/variance boundary。因此状态登记为 **有限训练步数验证（finite-step validated）**，不是 long-run validated 或稳定固定点排名。
> - 三类事件继续分开报告：task-performance collapse `10/220`、support/variance boundary `20/220`、NaN/Inf numerical failure `0/220`。Unweighted 只作为内部负对照和 runner 回归锚点，不登记为新的失稳发现。
> - C-U1 动作空间与 reward 函数连续。每个状态的 8 个负动作是从连续等 reward 圆周上人为选择的均匀角度，用于精确匹配负 reward/advantage 并隔离 policy-relative distance；有限离线支持不等于环境不连续。
> - 在负样本内部，质量幅度与 near/far distance 严格解耦，但 directional utility 与 distance 在当前二维几何中有意相关。该设计识别 informativeness--amplification mismatch，不得升级为“所有近负样本必然有益、所有远负样本必然有害”。
> - Linear/Quadratic/Exponential 只匹配共同参考点 `w(d_ref)=rho`，未匹配参考点斜率、near-negative retention、总负梯度预算或累计 optimizer update。当前结果只支持 anchor-normalized mechanism-order claim；不支持 best-tuned family 的普遍任务排名。 不得写成 Exp、Quadratic 或其他方法的 universal winner。
> - 解析边界保持：在 bounded advantage、pre-boundary `sigma>=sigma_min>0` 与 learnable Gaussian log-scale 输出分支下，未加权远场项为 `Theta(d^2)`；reciprocal-polynomial `p=2` 是使该输出分支渐近有界的最低多项式阶。该阶数结论与有限系数无关，但不推出 Quadratic task reward 必然高于充分调参后的 Linear，也不推出 Exponential 是最终最佳方法。
> - 后续证据需求只登记为 **review-required proposals**，本版不授权新 formal experiment：连续角度/随机 phase/薄圆环几何稳健性；匹配 near-negative retention 的 family 比较；与 Global alpha 的逐步或累计负梯度预算匹配；等搜索预算调参后使用全新 confirmatory seeds；原 Adam 长程、2x continuation 与必要时独立 full-batch stationary-solution audit。
> - E4/E4-CONV 继续承担 positive-only imitation ceiling、受控负梯度外推收益和过强负压力反转；E4-TAPER 不重复这些既有职责。当前执行路线不被上述 review proposals 抢占：下一正式实验仍是已批准、implemented + ready + active 的 `D-U1-E6-SEMANTIC-LONGRUN-01`。


> **v44（D-U1 E6 formal 冻结与单次启动版）**
>
> **v44 增量登记：`D-U1-E6-SEMANTIC-LONGRUN-01` 用户批准 freeze、formal 实现与 canonical activation（不删除 v43 及更早内容）**
>
> - 用户于 2026-06-27 明确批准按 focused-development 建议原样冻结 E6 formal 协议，并要求不再拆分额外“激活包”。本版一次性完成 formal config、独立科学 runner、one-click hardened launcher、registry ready/active 状态和测试；应用并提交后即可直接启动 long-run，不再需要新的运行前仓库更新。
> - 正式 seeds 锁定为 untouched `10--29`；development seeds `0--4` 禁止进入正式聚合。Adam `lr=1e-3`、batch 128、8000 steps、每 50 steps 评估；8000 是相对 4000-step focused-development horizon 的 2x 正式延长。
> - 固定 concentration `8.0`，E6-A alpha 网格冻结为 `{0,0.25,0.5,0.75}`。Learnable concentration 初值 `8.0`、无上界 clamp，local alpha `0.1`、far-pressure stress `0.05`；E6-B 比较 `positive_only / far_zero / uncontrolled / near_zero / far_cap / budget_matched_global`，不预设胜负。
> - E6-C 保留 policy-side semantic alignment 排他性控制：`aligned/shuffled` 两种模式，方法为 `positive_only / far_zero / uncontrolled / far_cap`；reward-side catalogue、hidden optimum、数据与 advantage 不变。训练/测试 context 仍为同分布独立采样，只能称 held-out-context / unseen-state generalization。
> - 正式终态窗口锁定为 `4000--6000` 与 `6000--8000`；沿用 focused-development 的任务/支持窗口变化阈值及 raw-gradient、Adam-update 增长比 `<=1.25`。任务性能崩溃、support/temperature boundary、NaN/Inf 数值失败继续分别报告；科学失败事件本身不使 artifact 丢失。
> - 预计正式运行超过 30 分钟，因此按 held-out seeds 每 5 个一组写入 persistent-local compact checkpoint manifest（`10--14 / 15--19 / 20--24 / 25--29`）。科学 runner 不创建归档；canonical guard 在失败时把已有 partial outputs 与 checkpoint manifests 打包为 failed-run evidence，成功时生成 raw-complete artifact。
> - 正式配置为 `configs/du1_e6_semantic_longrun.yaml`，科学入口为 `src/drpo/du1_e6_semantic_longrun.py`，唯一推荐启动入口为 `scripts/run_du1_e6_semantic_longrun.py`。one-click 入口要求 clean HEAD 与 `origin/main` 一致，并通过 canonical hardened guard 前台监督、打包和验证。
> - `D-U1-E6-SEMANTIC-LONGRUN-01` 当前科学状态仍是 **尚未运行（not_run）**，但实现和执行门禁已为 **implemented + ready + active**。应用本版后的下一步就是运行 long-run；结果完成前不得启动 `D-U1-E6-TAPER-01`。
>
> **v43（D-U1 E6 聚焦开发扩展结果审计版）**
>
> **v43 增量登记：`D-U1-E6-SEMANTIC-FOCUSED-DEV-01` 两阶段完成、阻塞项解决与 formal freeze 建议（不删除 v42 及更早内容）**
>
> - 本扩展基于用户确认的 GitHub `main` commit `2e04f6dba6d4e87f61920bedb1c464656906bf2b` 的完整 tree，源码在本地 clean commits 上运行。Phase 1 完成 `55/55` runs，Phase 2 完成 `110/110` runs；均只使用 development seeds `0--4`、CPU 和 4000 steps，未消耗任何 formal held-out seed。
> - 三类事件继续分报：NaN/Inf `0/165`；任务性能崩溃 `0/165`；support/temperature boundary `78/165`。高 reward 与 support boundary 仍可并存，不能合并为单一 collapse。
> - **固定 concentration 阻塞已解决。** `alpha={0,0.25,0.5,0.75}` 均为 5/5 focused terminal plateau。`alpha=0.25/0.5` 相对 positive-only 保持 reward 与 hidden-optimal probability 改善；`alpha=0.75` 保持过度外推和性能反转。原 pilot 的“30/30 未 plateau”主要来自把 stochastic raw total gradient 错误要求低于绝对 `0.02`；该旧判据的问题是平衡状态仍可能存在非零相互抵消的样本梯度。替代判据使用预注册 W1/W2 任务/支持窗口均值变化及 raw-gradient、Adam-update 的增长比，且仍不构成 formal acceptance。
> - **learnable concentration 的安全 local pressure 已识别。** 按预注册从大到小选择规则，`alpha=0.2` 因 3/5 support boundary 且仅 1/5 plateau 被排除；`alpha=0.1` 是首个满足全部条件的候选：0/5 support boundary、0/5 NaN/Inf、reward 5/5 胜过 paired positive-only、hidden-optimal probability 5/5 胜出、5/5 focused terminal plateau。终态均值为 reward `0.888109`、hidden probability `0.193765`、concentration `12.5603`、effective-support p05 `4.0634`。
> - **far pressure 转折已识别。** 在 `local_alpha=0.1` 下，`far_lambda=0.01` 的 uncontrolled/far-cap/budget-global 均 0/5 support boundary 且 5/5 plateau；`far_lambda=0.02` 三者均 5/5 support boundary，而 far-only/near-zero 仍 0/5；到 `far_lambda=0.05`，far-only 也 5/5 触发 boundary。由此可在 formal 中冻结 `0.05` 作为 far-pressure stress。ratio-1 Far-cap 与等预算 Global 在该开发扫描中没有救援转折，必须作为非预设胜负的控制，不得宣称优越。
> - 结果支持的谨慎机制口径是：适量 local negative 在当前共享语义环境中可突破 positive-only 上限；更强的 far pressure 可独立触发 support contraction；near negative 不被提升为普遍有益，far-cap/global 也不被预设为更优。训练/测试仍是同分布 held-out contexts，不得称 OOD。
> - `outputs/du1_e6_semantic_focused_dev/` 保存 compact summary、逐条件表、终态审计和 formal freeze 建议；完整 Phase 1/2 trajectories、逐 run summaries、日志、配置、source snapshot 与 checksums 位于 raw artifact `DRPO_DU1_E6_SEMANTIC_FOCUSED_DEV_RAW.zip`，SHA-256 为 `bee5b62e7715bda63ec166849f431ab5c4c90954720e672945e25e62b320e0d6`。
> - 两个开发阻塞项已经解决，但自动 freeze 仍禁止。建议供用户审阅的 formal 配置为：held-out seeds `10--29`；Adam `lr=1e-3`；8000 steps（相对 4000-step development horizon 的 2x）；固定 concentration `8.0` 与 `alpha={0,0.25,0.5,0.75}`；learnable concentration 初值 `8.0`、无上界 clamp、`local_alpha=0.1`、far stress `lambda=0.05`；方法矩阵 `positive_only / far_zero / uncontrolled / near_zero / far_cap / budget_matched_global`。正式 runner、canonical activation 和 held-out 执行必须等待用户明确批准。


> **v42 增量登记：状态机一致性、E7 已实现门禁与 E4--E8 路线锁定（不删除 v41 及更早内容）**
>
> - 本轮基于 `main` commit `f64452a7452274a183b03c87c39b847039230c00` 合并 registry 状态一致性与跨环境执行路线。该 commit 已包含 `EXT-H-E7-Q2` Hopper runner、冻结配置、操作入口和测试；本版不重复实现 Hopper，也不把实现完成升级为实验结果。
> - 修复 `C-U1-E4-TAPER-01` 的状态机矛盾：其科学门禁已是 `ready`、entrypoint 已实现，因此 operational activation 同步为 `active`。科学状态仍为 **尚未运行（not_run）**，不产生方法排名。
> - `EXT-H-E7-Q2` 当前是 **implemented + not_run + blocked**：代码实现已完成，但按锁定路线，formal launch 必须等待 `D-U1-E6-TAPER-01` 交付。实现存在不等于执行门禁自动开放；registry 中 blocked gate 与 blocked activation 保持一致。
> - 锁定一致性规则：ready gate + implemented entrypoint 必须 active；blocked gate 不得对应 active；任何 blocked 状态必须给出依赖或 blocking reason；planned/not-implemented entrypoint 必须 fail closed。canonical experiments 与 development registrations 中的 formal 条目均由统一 validator 检查。
> - `D-U1-E6-SEMANTIC-PILOT-01` 的 105/105 development runs、终态审计和 **pilot** 科学状态完整保留；`D-U1-E6-SEMANTIC-LONGRUN-01` 继续等待用户审阅、聚焦开发扩展与 formal 参数冻结，不得自动升级。
> - 后续路线锁定为：`E4-TAPER -> E6 -> E6-TAPER -> E7-MECH -> E7-BENCH -> E8-MECH -> E8-SCALE`。E7-MECH 由已实现但尚未运行的 `EXT-H-E7-Q2` 承担；E7-BENCH 为 D4RL MuJoCo locomotion 9-task suite；E8-MECH 由 `EXT-C-E8-V4.2` 承担；E8-SCALE 使用更大固定 Countdown 数据、3B 主模型与 7B 冻结确认。
> - reciprocal-linear 是当前 taper 理论中的内部 `p=1` 同标准化距离控制，不是原 DRPO 分布鲁棒章节中的 “linear weighting”。clipped-linear、surprisal-linear 或不同距离坐标必须另行登记。
> - 本版只更新治理、registry、validator、测试和执行路线，不运行任何新实验，不预设 Distance、Exp、Global alpha、SBRC、Hybrid 或其他方法排名。

> **v41 增量登记：`EXT-H-E7-Q2` Hopper 实现、执行门禁与结果打包协议（不删除 v40 及更早内容）**
>
> - 本轮基于 `main` commit `2e04f6dba6d4e87f61920bedb1c464656906bf2b` 实现 `EXT-H-E7-Q2`。实现入口为 `src/drpo/e7_hopper_q2.py`，一键操作入口为 `scripts/run_e7_hopper_q2.py`，冻结配置为 `configs/e7_hopper_q2_medium_replay_v2.yaml`。当前科学状态仍为 **尚未运行（not_run）**；CPU 单元测试、静态检查和缩减工程 pilot 均不得升级为 Hopper 正式结果。
> - 主机制数据冻结为 legacy D4RL `hopper_medium_replay-v2.hdf5`，SHA-256 `e121c5f7c9857a307baa9edc6a2c3b48e85fedb9ac316ecddd0f48ca7ef4e39b`。该数据承担自然 near/far 外部机制验证；Hopper-medium 保留为较窄复验，Hopper-medium-expert 保留给后续方法效果或 stable-extrapolation 研究，不在本子 claim 中混用。
> - critic 先按完整 episode 划分训练/验证/测试，目标为 discounted Monte-Carlo return；通过预登记优化终态审计后冻结。冻结 critic 随后一次性物化 TD-residual advantage、训练划分统计量、正负 mask 与 provenance hash；actor 阶段禁止 minibatch advantage 重归一化或继续更新 critic，因此该阶段研究对象保持为固定 learned advantage 与动态 policy-score geometry 的重复离线更新。
> - actor 冻结为 state-conditioned tanh-squashed diagonal Gaussian MLP；理论坐标使用 `u=atanh(clip(a))`、`z=(u-mu)/sigma` 与 `r=||z||`。runner 同时保存 mean-score、raw log-scale score、校正 `Q_xi=||z||²`、joint output score、full-parameter gradient、raw action distance、pre-squash distance及 analytic/output-autograd 误差。仅解析恒等式通过仍只算实现一致性，不算独立外部验证。
> - 每个 paired seed 先将 Positive-only 训练到预登记终态候选并完成 2x continuation，再从同一个 checkpoint、同一个 minibatch 随机流分叉 `positive_only / signed / near_zero / far_zero / far_cap / budget_matched_global`。near/far 只在负 advantage 内按 `|A|` 匹配；Far-cap 使用冻结近场 joint-output-score 参考，Global control 按初始逐样本 full-parameter negative-gradient 总预算匹配。该实验不形成标准 D4RL 方法排名。
> - 正式 normalized-return 评估要求训练机可注册 `hopper-medium-replay-v2` 环境；正式 seeds 冻结为 100--109，pilot seed 为 42。任务性能崩溃、support/variance boundary、persistent/slow drift 与 NaN/Inf 数值崩溃继续分报；终态分类必须包含冻结窗口、2x continuation 与 paired-seed targeted-control gate。
> - 所有 pilot 与 formal 运行必须由 `scripts/run_experiment_guard_hardened.py` 的 canonical channel 前台监督。科学 runner 禁止创建 ZIP/TAR；guard 负责 heartbeat、失败证据、源码快照、checksums、size policy 和最终结果 ZIP。用户只需启动一次并上传 guard 生成的结果 ZIP；大型 checkpoint 与数据集保留在持久本地，仅以路径、大小、角色和 SHA-256 索引。
> - 旧 Hopper 600-step probe 继续保留为有限训练步数历史证据，不能冒充本次 E7-Q2 实现或正式结果。Hopper 仍只提供外部有效性，不能替代 C-U1 的 ground-truth 因果识别，也不预设 Exp、Linear、Global alpha、SBRC、Hybrid 或 Positive-only 的跨任务排名。

> **E6 聚焦开发前置协议记录（原并行会话暂编号 v41；集成时因主线 v41 已由 EXT-H-E7-Q2 占用，现作为 v43 的前置协议历史完整保留）**
>
> 以下内容不改变原实验变量、seeds、阈值或结果，只修正并行文档版本号冲突。
>
> **v43 前置协议登记：`D-U1-E6-SEMANTIC-FOCUSED-DEV-01` 解决 E6 formal 前两项开发门禁（不删除 v40 及更早内容）**
>
> - 当前基线为用户确认已提交的 `main` commit `2e04f6dba6d4e87f61920bedb1c464656906bf2b`。该提交已闭环 E6 pilot；本扩展仍只使用 development seeds `0--4`，不消耗 untouched formal seeds，不形成正式方法排名。
> - 本实验只解决 v40 已登记的两个阻塞项：固定 concentration 的 2000-step 终态未确认，以及 learnable concentration 下当前负压力普遍触发 support boundary。不得新增正则器、上界 clamp、critic、动态 advantage 或新的科学变量。
> - Phase 1-A 固定 concentration 保持 `8.0`、Adam `lr=1e-3` 和原数据几何不变，将 `alpha in {0,0.25,0.5,0.75}` 延长到 4000 steps。窗口固定为 `W1=2000--3000`、`W2=3000--4000`；比较 reward、hidden-optimal probability、normalized extrapolation、entropy 的窗口均值变化，并审计 raw-total-gradient 与 Adam-update 的 W2/W1 中位比。
> - Phase 1-B learnable concentration 保持初值 `8.0`、无上界 clamp，扫描 local-only `alpha in {0.005,0.01,0.02,0.05,0.1,0.2}`，并加入同 seed positive-only reference。选择规则预先锁定为：按 `0.2 -> 0.005` 从大到小，选择首个同时满足 0/5 support boundary、0/5 NaN/Inf、reward 与 hidden probability 各至少 4/5 seeds 胜过 paired positive-only、且至少 4/5 seeds 通过 focused terminal plateau 的 alpha；若无候选则停止，不得临时放宽。
> - 若 Phase 1-B 找到候选，Phase 2 才允许在该 alpha 上扫描既有 `far_pressure_lambda in {0.01,0.02,0.05,0.1,0.2}`，比较 `uncontrolled / far_cap / budget_matched_global / near_zero`，并保留 local-only 与 positive-only reference。Phase 2 的值由上述预注册选择规则触发，不是看 formal held-out 结果调参。
> - **Phase 1 checkpoint（进入 Phase 2 前登记）：** 55/55 runs 完成且 NaN/Inf 为 0。固定 concentration 的 `alpha={0,0.25,0.5,0.75}` 均为 5/5 focused terminal plateau；`alpha=0.25/0.5` 保持有益，`0.75` 保持过度外推/性能反转。learnable local-only 中 `alpha=0.2` 因 3/5 support boundary 且仅 1/5 plateau 被预注册规则排除；`alpha=0.1` 为从大到小首个满足全部门禁的候选：0/5 support boundary、0/5 NaN/Inf、reward 与 hidden probability 均 5/5 胜过 paired positive-only、5/5 focused terminal plateau。因此 Phase 2 固定使用 `local_alpha=0.1`，不得改用更优看的后验值。
> - focused terminal plateau 只是一项开发筛选：W1/W2 四项任务/支持指标窗口均值绝对变化分别不超过 `0.01/0.02/0.08/0.08`，raw gradient 与 Adam update 的 W2/W1 中位比均不超过 `1.25`，且没有 support boundary 或 NaN/Inf。非零 stochastic gradient 不再被错误要求降到任意绝对阈值；formal 仍须单独执行预注册的 2x extension。
> - 三类事件继续分报：任务性能崩溃、support/temperature boundary、NaN/Inf。训练/测试 context 仍同分布独立采样，只能称 held-out-context / unseen-state generalization，不得称 OOD。
> - `D-U1-E6-SEMANTIC-LONGRUN-01` 在本扩展完成、用户审阅并冻结 alpha、far pressure、formal method matrix、最大步数、终态/2x 规则及 untouched seeds 前继续 blocked。

> **v40（D-U1 E6 语义 pilot 结果审计版）**
>
> 以下 v40 历史标题与内容完整保留；v41 只追加 E7-Q2 实现冻结，不覆盖 D-U1 E6 pilot 结果与正式门禁判断。

> **v40 增量登记：`D-U1-E6-SEMANTIC-PILOT-01` 开发 pilot 完成、终态审计与正式门禁判断（不删除 v39 及更早内容）**
>
> - 运行依据为用户确认并由 GitHub commit 页面核对的 `main` commit `e8b62dde518f593ff8325c7da94c41406311ca45`。当前执行环境的 Git shell 无法解析 GitHub 域名，因此未取得该 commit 的完整 Git object checkout；pilot 使用与该提交 E6 runner、配置、handoff 和 registry 文件一致的本地验证快照，快照 commit 为 `653aa6f73b18fed17609e6096cb1c50de0a8cd66`。该 provenance 限制允许保留 pilot 证据，但禁止把本轮升级为 formal long-run 结果。
> - 先后完成 invariants、engineering smoke 和完整 development pilot。正式 pilot 使用 CPU、development seeds `0--4`、21 个 protocol/method 条件、每条件 5 seeds，共 `105/105` runs、每 run 2000 steps；子进程 exit code 为 0，总运行时约 513 秒。环境不变量全部通过，NaN/Inf 数值失败为 `0/105`。
> - 三类失败事件继续分报：任务性能崩溃 `0/105`；support/temperature boundary `56/105`；NaN/Inf `0/105`。高 task reward 与 support boundary 可同时出现，因此本结果再次证明不能把任务效果、支持边界和数值崩溃合并为单一“collapse”。
> - E6-A 固定 concentration 扫描中，开发集 `alpha=0.5` 的 held-out-context expected semantic reward 均值为 `0.888228`，hidden-optimal probability 为 `0.216125`，相对 positive-only 的 hidden-probability 配对均值增量为 `+0.072037`；`alpha=0.75/1.0` 出现过度外推和性能反转。但 6 个 alpha 条件的 30/30 runs 均未通过 pilot 两窗口 provisional plateau，2000 steps 不能被冻结为已收敛正式 horizon。
> - E6-B 可学习 concentration 下，positive-only 为 `0/5` support boundary；所有包含负压力的 aligned 方法均为 `5/5` support boundary，常在 step 200--400 首次触发，终态 effective support 约 `1.27--1.32`、concentration mean 约 `134--155`。该结果是开发期支持收缩证据，不构成方法排名；当前 `alpha=0.5, far_lambda=1.0` 不得原样冻结为 formal protocol。
> - E6-C 中，aligned policy-side semantic embedding 对 hidden-optimal probability 与 semantic reward 的提升显著高于 shuffled control；这在 pilot 层面支持“共享语义结构是未见动作概率转移的必要机制之一”。训练与测试 context 仍同分布独立采样，只能称 held-out-context / unseen-state generalization，不得称 OOD。
> - 自动参数冻结仍被禁止。`D-U1-E6-SEMANTIC-LONGRUN-01` 保持 blocked：必须先做用户审阅，并使用既有变量进行聚焦开发扩展——固定 concentration 围绕 `alpha in {0.25,0.5,0.75}` 延长终态检查；可学习 concentration 降低 local/far negative pressure 后重新确认支持边界。不得新增未经审批的变量或正则器。
> - compact repository outputs 位于 `outputs/du1_e6_semantic_pilot/`；完整 raw trajectories、逐 run summaries、日志、source snapshot 与 checksums 保存在交付包内嵌 raw artifact。科学状态保持 **pilot**，formal 2x extension、held-out formal seeds 与正式方法排名均未执行。

> **v38（D-U1 E6 语义 pilot 准备版）**
>
> 以下 v38 历史标题与内容完整保留；v

> **v39（Countdown V4.2 平衡离线集与动态远场诊断冻结版）**
>
> 以下 v39 历史标题与内容完整保留；v40 只追加 E6 pilot 结果和门禁判断，不覆盖 Countdown V4.2 协议。
> **v39 增量登记：Countdown `EXT-C-E8-V4.2` 平衡离线集、动态远场诊断与参数化隔离（不删除 v38 及更早内容）**
>
> - 新执行 ID 为 `EXT-C-E8-V4.2`，状态为 **尚未运行（not_run）**；它继续只承担 Transformer/Countdown 外部有效性，不替代 D-U1/D-Diag 的受控机制识别。`EXT-C-E8-V4.1` 保留 provenance 并由本版本替换。
> - 服务器上完成的单 seed V4.1 运行绑定 commit `17fdb46502cd82f0b17a2601172f32d368611507`，但运行时工作树为 dirty，且本地把 SFT greedy gate 从 15% 改为 5%；因此只登记为 **off-protocol exploratory pilot**。其 mechanism probe 可作为开发诊断，不能作为注册版方法排名或正式结果。
> - 修复昂贵阶段结束后 manifest 写入失败的问题：`argparse.set_defaults(func=...)` 注入的 callable 不得随 `vars(args)` 直接 JSON 序列化。`cmd_sft`、`cmd_build_offline` 的两个 manifest 路径和 `cmd_train_method` 统一通过 `serializable_namespace()` 过滤 callable；该修复不改变训练数学。
> - 研究主张保持：在固定 reward/advantage 和 matched nuisance 条件下，far negative 可能因 policy-relative influence 更大而比 near negative 更危险。**不要求 near negative 普遍有益**；near 可以有益、无效或有害。V4.2 主要检查 far/near influence、uncontrolled 退化以及 selective far control 是否更能保存性能，不得把单一 held-out pattern 的概率重分配升级为总体方法优势。
> - `15%` greedy 不是理论常数，而是防止 floor effect 的操作性方法门禁。冻结两级门禁：mechanism pilot 要求 reference greedy `>=0.08` 且 valid `>=0.95`；四方法效果比较要求 greedy `>=0.15` 且 valid `>=0.95`。只过前者时自动完成机制 probe 与 full-FT reference 诊断，但跳过四方法排名；不得在运行中继续降低阈值。
> - 主方法比较继续使用同一个 Qwen2.5-0.5B-Instruct BF16 LoRA reference。SFT 由固定 3 epochs 改为最多 6 epochs、至少 3 epochs、逐 epoch validation、连续 2 次无改进停止，并使用覆盖完整注册 schedule 的 cosine scheduler。LoRA 四方法仍共享初始化、离线数据、训练 seed 和评估 seed。
> - 新增隔离的 0.5B BF16 full-parameter SFT reference diagnostic：与 LoRA 使用同一 train/validation split，学习率 `2e-5`、最多 6 epochs、至少 3 epochs、patience 2。该分支只判断 LoRA 是否构成基础能力瓶颈，不得替代 LoRA reference、不得混入四方法排名；其 checkpoint 只保存在服务器本地并索引。
> - 离线 matched corpus 扩大为 **6000 rows**：继续使用 6000 个训练 prompts，并要求 48 个训练 canonical patterns 各保留 125 rows；每个 prompt 只在找到满足既有 surprisal/长度/树深/数值误差约束的 pair 后进入数据。默认 `rollouts=12`、最少 8 个合法负候选、最多 8 个 resample rounds；仍未配对时最多补充 64 个确定性 synthetic rescue candidates，但不放宽任何 matching 约束。任何 pattern quota 未填满即 fail closed，并保存 partial diagnostics，禁止静默缩小或不平衡继续。
> - 同一 frozen LoRA reference 的 6000-row corpus 自动导出 pattern-balanced、互相 nested 的 1500/3000/6000 subsets。该数据可在同一 reference 下复用于四方法、训练 seeds 与已登记的开发检查；reference checkpoint、模型大小、tokenizer 或参数化变化后必须重建。
> - 方法训练从固定 1200 steps 改为 effective-offline-epoch 口径：最多 6 epochs、至少 2 epochs、每 1 epoch validation、patience 2。H20 参考配置的 effective batch 为 32，对 6000 rows 对应约 188 updates/epoch、最大 1128 updates、最小 376 updates；实际 runner 按运行时 effective batch 自动计算，不由本地 AI 决策。
> - 每个方法在 step 0 与每次 validation 自动写 `dynamic_diagnostics.jsonl`：positive/near/far surprisal 的 mean/median/p90、原 near 动态越过 far threshold 的比例、controlled far weight、raw/scaled gradient norms、far/near gradient ratio、positive-near 与 positive-far gradient cosine、effective epoch。任务性能、结构/支持退化与 NaN/Inf 继续分开报告。
> - LoRA checkpoint 不再在每个 best/terminal 目录重复复制 tokenizer 词表，只写入 `tokenizer_reference.json` 并从注册的 foundation model path 加载 tokenizer；full-model diagnostic checkpoint 仍保存自身 tokenizer metadata。该变化只减少本地与 artifact 索引冗余，不改变模型参数或评估。
> - 本协议基于 `main` commit `e8b62dde518f593ff8325c7da94c41406311ca45`。本版本只冻结文档、registry、实现和测试，不产生新的 Qwen/H20 结果；任何 V4.2 数值结论必须等待 clean committed source 上的一键运行、终态审计与 durable artifact。

> **v38（D-U1 E6 语义 pilot 准备版）**
>
> 以下 v38 历史标题与内容完整保留；v39 只追加 Countdown V4.2 协议，不覆盖 E6 pilot 准备。

> **v38 增量登记：`D-U1-E6-SEMANTIC-PILOT-01` 共享语义 categorical pilot 准备与并行批准（不删除 v37 及更早内容）**
>
> - 用户于 2026-06-26 明确批准：`D-U1-E6-SEMANTIC-PILOT-01` 可与另一 session 正在执行的 `C-U1-E4-TAPER-01` 并行。两者科学职责、runner、结果目录和 experiment ID 完全独立；若两边都修改 `docs/handoff.md` 或 `experiments/registry.yaml`，仍须由 Stage 1 `drpo-update` 三方集成串行落库。
> - E6 本轮只进入 **pilot 准备与开发 seeds 0--4**，科学状态保持 **pilot**。正式长程 ID 预留为 `D-U1-E6-SEMANTIC-LONGRUN-01`，但在 alpha、concentration、学习率、最大步数、事件阈值、终态标准和 untouched held-out seeds 经 pilot 审阅并回写冻结前，正式门禁保持关闭。
> - 为保持 canonical formal-channel validator 的现有激活清单不被未实现 long-run 污染，两个 E6 ID 先登记在 `experiments/registry.yaml` 的 `development_experiment_registrations`；pilot 冻结完成、正式 runner 实现并通过审阅后，再把 long-run 条目提升到 canonical `experiments` 列表。该预登记不是正式激活，也不构成实验结果。
> - 新 runner `src/drpo/du1_e6_semantic.py`、开发配置 `configs/du1_e6_semantic_pilot.yaml` 与测试共同实现 E6-A/B/C：local-negative alpha scan、far-pressure 因果控制、policy-side semantic shuffle。训练与测试 context 均独立采样自同一 `N(0,I_6)`；只能称同分布 held-out-context / unseen-state generalization，不得称 OOD。
> - 环境保持 6D state、64 个随机 ID 的 4D semantic actions、隐藏最优动作不进入 positive demonstrations、4 positive / 1 local negative / 4 far negatives、负 advantage 严格相等且冻结。固定 concentration 分支隔离语义外推；可学习 concentration 分支审计 support/temperature dynamics。
> - 输出必须分别记录 hidden-optimal probability、expected semantic reward、normalized semantic extrapolation、entropy/effective support、concentration、raw positive/local/far gradients、controlled negative budget 与 Adam parameter update。task-performance collapse、support/temperature boundary 和 NaN/Inf numerical failure分开报告。
> - `Far-cap` 与 `budget-matched global` 只做 raw-negative-gradient norm 匹配；不得把它写成 Adam update matching。E6-C 只打乱 policy-side embedding 对 reward semantic 的对应，reward geometry、hidden optimum、样本标签和动作集合保持不变。
> - static/unit/smoke 只证明实现可运行；pilot 只用于冻结正式协议。任何方法排序、稳定外推、支持保护或 shared-semantic generalization 的论文结论，均须等待单独登记的 long-run held-out-seed 实验与终态 2x 审计。



> **v37（D-U1 E5 长程复核闭环版）**
>
> 以下 v37 历史标题与内容完整保留；v38 只追加 E6 pilot 准备，不覆盖 E5 闭环。


> **v37 增量登记：`D-U1-E5-LONGRUN-RERUN` 20000-step 正式复核、历史对照与长期闭环（不删除 v36 及更早内容）**
>
> - 正式运行绑定 commit `22c5823d66169eb90c256de342e27c5391e464c3`。D-Diag 两个 direct-softmax 分支均完成 20000 steps；D-U1 完成 6 methods × seeds 10--29 × 20000 steps，共 120/120 method-seed runs。所有运行均有终态分类，NaN/Inf 为 0/120。
> - Direct-softmax 精确复核通过：高概率负动作由 `p=0.8991` 降至 `3.70436e-12`，低概率负动作由 `p=0.0038` 降至 `1.91726e-20`；两者 target surprisal 与 logit gap 在尾段持续增长，direct-logit score 始终不超过 `sqrt(2)`。高概率分支 entropy 先升后降，低概率分支 entropy 从初始起非增。终值与旧 handoff 参照均在冻结的 20% 容差内。
> - D-U1 长程因果结果逐项复现历史 qualitative pattern：Baseline 与 Near-zero 均为 20/20 task-performance collapse 且 20/20 support boundary；Far-zero 与 Far-cap 均为 0/20 task collapse、0/20 support boundary；Global-scale 为 0/20 task collapse 但 20/20 support boundary；Positive-only 为两类事件均 0/20。六方法共 120/120 historical joint class match。
> - 该结果闭合 E5 的受控 categorical 机制：单步 direct-logit score 有界并不阻止 repeated negative update 持续扩大 surprisal/logit gap、将概率推向 simplex/support boundary；删除近场而保留 far negative 不能救援，删除或截断 far negative 可以同时保护任务与支持。Global-scale 的高 reward 与 support boundary 并存进一步证明任务性能与支持边界必须分报。
> - 科学状态升级为 **已长期验证（long-run validated）**。该状态仅适用于本次基于 locked handoff 重建的 D-Diag/D-U1 categorical mechanism 与历史 qualitative pattern；不声称旧未提交 runner 的 byte-identical 复现，不证明 E6 的未见动作语义泛化，不声称 categorical direct-logit gradient 无界，也不形成跨任务方法排名。
> - 第一次和第二次正式尝试均在完成 66/120 runs 后被当前工具调用的外部持续时间上限终止；两个 failed-run 证据包完整保留。第三次尝试由同一 hardened guard、同一 commit 和冻结协议完成 120/120，并生成 verified raw-complete artifact。失败尝试不进入科学聚合。
> - 仓库只提交 compact summary、逐 seed 汇总、direct trajectories、终态审计、图和 artifact index；完整逐方法轨迹保存在 raw-complete ZIP 中。E6 仍是独立未完成实验，不得由 E5 替代。


> **v36 增量登记：`D-U1-E5-LONGRUN-RERUN` 长程复核与 provenance 重建（不删除 v35 及更早内容）**
>
> - 用户于 2026-06-26 明确确认：E5 与正在其他 session 执行的 `C-U1-E4-TAPER-01` 科学职责、runner 和结果目录完全独立，允许并行推进；但两者最终都可能修改 `docs/handoff.md` 与 `experiments/registry.yaml`，仓库集成仍须通过 Stage 1 Git-bundle 三方合并串行完成。
> - 新正式实验 ID 为 **`D-U1-E5-LONGRUN-RERUN`**，当前状态为 **尚未运行（not_run）**。它只复核 E5 的 categorical 排斥与支持边界，不替代 E6 的共享语义未见动作外推，也不承担 Transformer/token 外部有效性。
> - 当前 Git 历史和 tree 中不存在旧 `run_categorical.py`、`unified_repulsive_dynamics/results/categorical_paper_run/` 或原始 categorical artifact。旧 handoff 中的 direct-softmax、20-seed near/far 干预和饱和审计结论继续保留，但本轮必须明确称为“基于已锁定 handoff 的 protocol reconstruction”，不得声称 byte-identical 复现缺失的历史 runner。
> - D-Diag 直接 softmax 分支冻结为 3 动作、固定负优势 `A=-1`、学习率 `1e-3`、20000 steps、每 100 steps 评估；高概率负动作初态 `p0=0.8991,H0=0.386`，低概率负动作初态 `p0=0.0038,H0=0.292`。验收为 target surprisal/logit gap 持续增长、direct-logit score 始终不超过 `sqrt(2)`，并与旧 handoff 的终值数量级对齐。
> - D-U1 causal reconstruction 使用 6D context、26 个随机 ID 映射的 semantic offsets、固定 advantage、无 critic/value、无 importance sampling。正式 seeds 仍为 10--29；比较 `positive_only / baseline / near_zero / far_zero / far_cap / global_scale`，并固定 near/far 负质量与所有几何、Adam、阈值和最大步数。
> - 长程上限冻结为 20000 steps，每 100 steps 评估，终态窗口为 `10000--15000` 与 `15000--20000`。内部稳态分支必须通过 beta、tau、reward 与 raw-gradient 后段门禁；无法形成内部 fixed point 的分支允许以 support/temperature boundary 或 persistent suppression 闭环。task-performance collapse、support/temperature boundary、NaN/Inf 必须分开报告。
> - 历史 qualitative pattern（Baseline/Near-zero task+support collapse；Far-zero/Far-cap 两类均不 collapse；Global-scale task 保留但 support collapse；Positive-only 两类均不 collapse）只作为**预先登记的复现参照**，不是结果后调参目标。若长程结果不一致，必须如实报告并替换旧口径，禁止修改 seed、阈值或方法质量以追求对齐。
> - 正式入口为 `src/drpo/du1_e5_longrun_rerun.py`，必须通过 `scripts/run_experiment_guard_hardened.py` 的 canonical channel 启动；runner 禁止自行写 ZIP。raw-complete、terminal audit、最终闭环包与 repository commit 继续作为分离状态管理。
> - 本协议基于 `main` commit `d9424f1b9ab4e5ed25bc1ac00f97d84317f67cdc`。在协议更新实际应用并提交前，不得启动正式 E5。


> **v35（C-U1 E4 用户确认闭环版）**
>
> **v35 增量登记：`C-U1-E4-CONV-01` 经用户审阅后的科学闭环（不删除 v34 及更早内容）**
>
> - 用户于 2026-06-26 明确审阅并确认：`alpha=0.75` 的 15/20、`alpha=1.00` 的 16/20 `stable_beneficial_extrapolation`，以及 `alpha=1.25` 的 15/20 `stable_over_extrapolation`，结合其余 14/60 仅为 inconclusive、0/60 明确相反终态、60/60 从 step 2000 到 4000 科学角色不反转，已足以闭合 E4 的长程相变结论。
> - 该决定是**结果后、用户明确确认的证据审阅闭环**，不是把原预注册 18/20 门禁改写为通过。v34 中“18/20 门禁未通过”的历史事实、逐 seed 分类、阈值、训练步数和所有原始结果全部保留，不重新标注任何 inconclusive seed。
> - `C-U1-E4-CONV-01` 的科学状态升级为 **已长期验证（long-run validated）**，但闭环范围严格限定为：`alpha=0.75/1.00` 的有益外推在 4000-step 长程中保持有界且未反转；`alpha=1.25` 表现为稳定过度外推而非慢 runaway；结合既有 `alpha=1.50` 任务性能崩溃、`alpha=1.75` 持续 raw-gradient/parameter runaway 与可学习方差 support contraction，E4 的非单调相变与失败类型链条完成论文级闭环。
> - 仍禁止声称：原 18/20 预注册门禁已经通过、20/20 seeds 均获 fixed-point 认证、所有单 seed 都严格 stationary、同分布 held-out-context 是 OOD、或任何方法跨任务必然更优。论文必须如实报告 15/20、16/20、15/20 与 0/60 明确相反终态。
> - `C-U1-E4-TAPER-01` 的 E4 前置门禁解除；本 v35 更新应用并提交后，其状态为 **尚未运行、允许按已冻结协议启动**。该解除不预设 Linear、Quadratic、Exp 或其他方法排名。
> - 仓库闭环更新基于当前 `main` commit `ba1e3710df4140ffaf54db3ecf12cd6f40ac531a`；科学运行仍绑定 run commit `c869df8b203f13eb8389d1d300b33f1928502871`，两者不得混淆。


> **v34（C-U1 E4 长程终态确认结果审计版）**
>
> **v34 增量登记：`C-U1-E4-CONV-01` 4000-step 正式结果与失败门禁审计（不删除 v33 及更早内容）**
>
> - 正式运行绑定 commit `c869df8b203f13eb8389d1d300b33f1928502871`，完成固定方差 `alpha in {0.75,1.00,1.25}`、held-out seeds 50--69、每分支 4000 updates，共 60/60 seed-alpha rows。preflight、环境不变量、网格完整性、固定方差/Adam/positive-only 排除及 reference regression 均通过。
> - 冻结的 18/20 汇总门禁未通过：`alpha=0.75` 为 15/20 `stable_beneficial_extrapolation` + 5/20 inconclusive；`alpha=1.00` 为 16/20 + 4/20 inconclusive；`alpha=1.25` 为 15/20 `stable_over_extrapolation` + 5/20 inconclusive。三组均为 0/20 明确相反终态。
> - 60/60 runs 从 step 2000 到 4000 的科学角色均未反转；没有任务性能崩溃、support/variance boundary 或 NaN/Inf。aggregate W2 displacement/reward change 接近 0，raw-gradient 与 Adam-update W2/W1 aggregate ratios 也低于 1.25。门禁失败来自 14 个 seed-alpha rows 的个体 raw-gradient 或 Adam-update ratio 超过冻结阈值，而不是漂移、reward 反转或 runaway。该诊断不得在结果后用于放宽门禁。
> - 科学状态保持 **有限训练步数验证（finite-step validated）/ convergence unresolved**，不得升级为“已长期验证”或“稳定 fixed point 已闭环”。可以报告 4000-step 长程轨迹强支持原科学角色且无相反终态，但必须同时报告预注册共识门禁失败。
> - `C-U1-E4-TAPER-01` 继续阻塞。是否另行登记阈值稳健性、解析终态判据或接受当前证据边界，必须由后续文档化决策决定；不得自动延长训练、修改 optimizer/learning rate、降低 18/20 或放宽 1.25 阈值。
> - 第一次 `run_001` 因当前工具调用在 34 秒时转发 SIGTERM，失败证据已保存；`run_002` 科学子进程 return code 为 0，但第一次 guard packaging 因启动命令误要求不存在的 `TERMINAL_AUDIT.json`（实际为 `terminal_audit.json`）而标记 wrapper failure。原始失败证据与恢复后的 verified raw-complete 包均保留；科学输出未修改。

> **v33（C-U1 E4 长程终态确认协议冻结版）**

> **v33 增量登记：`C-U1-E4-CONV-01` 长程终态确认协议（不删除 v32 及更早内容）**
>
> - 用户批准新增 `C-U1-E4-CONV-01`，只补齐 E4 有益与过度外推分支的长期终态，不重跑完整 E4，也不替代 `C-U1-E4-ADAM-RERUN` 的有限步相变、支持收缩、任务崩溃和 runaway 证据。当前状态为 **尚未运行（not_run）**。
> - 从追加运行范围中移除 `alpha=0`。Positive-only ceiling 的完整长期动力学由 E2 的 2000-step Adam、等长 2x continuation 和终态审计承担；E4 仍保留原 `alpha=0` 有限步点作为比较基线，但不再重复研究固定学习率 Adam 能否通过 `2e-3` residual。
> - 正式分支只包含固定方差 `alpha in {0.75,1.00,1.25}`、held-out seeds 50--69。`0.75/1.00` 检验高 reward 是否为稳定有益外推而非暂时经过隐藏最优点；`1.25` 检验其是否为稳定过度外推而非临界慢 runaway。
> - 数据、C-U1 几何、2000-step positive-only Adam 初始化、网络、固定 `sigma=0.1903943276465978`、Adam `betas=(0.9,0.999)`、`eps=1e-8`、学习率 `5e-4`、batch、advantage、seeds 和 minibatch RNG 均保持不变。每个 alpha 从同 seed 的 E2 初始化 checkpoint 重新完整运行，不从旧 400-step 参数或 optimizer state 续跑。
> - 每个分支运行 4000 个 E4 updates；full-state audit 固定在 steps `400,800,1600,2400,3200,4000`。终态窗口固定为 `W1=2000--3000` 与 `W2=3000--4000`，同时保存 held-out-context reward、归一化外推位移、到 `a_plus/a_star` 的距离、full-data raw total gradient norm、Adam parameter-update norm 和 residual diagnostic。
> - 稳定平台门禁：W2 位移首尾变化绝对值不超过 `0.02`，W2 reward 首尾变化绝对值不超过 `0.01`，W2/W1 raw-gradient 中位比与 Adam-update 中位比均不超过 `1.25`，且 step 2000 到 4000 的科学角色不反转。持续 runaway 要求 W1/W2 位移均增加、W2 位移增量大于 `0.05`，并且 raw gradient 或 Adam update 的 W2/W1 比超过 `1.25`。其他情况登记 `terminal_state_inconclusive`，不得临时延长或放宽。
> - 原 `normalized_field_residual < 2e-3` 继续保存为诊断量，但不再作为本实验的硬科学验收门禁。该实验不研究固定学习率 Adam 的噪声平台；验收对象是长期轨迹类别是否保持有界、是否反转以及 raw-gradient/actual-update 趋势。
> - 20-seed 汇总门禁：`alpha=0.75/1.00` 目标为 `stable_beneficial_extrapolation`，`alpha=1.25` 目标为 `stable_over_extrapolation`；每个 alpha 至少 18/20 seeds 达到目标状态，剩余 seeds 只允许 `terminal_state_inconclusive`，不得出现明确相反终态。任务性能崩溃、support/variance boundary 与 NaN/Inf 继续分报。
> - 预计运行超过 30 分钟，按每 5 个正式 seeds 生成 recovery checkpoint。`C-U1-E4-TAPER-01` 继续阻塞，直到本实验完成终态审计、打包和交付。

> **v32（C-U1 E4 统一 Adam 有限步相变证据与终态门禁审计版）**
>
> **v32 增量登记：`C-U1-E4-ADAM-RERUN` 20-seed 正式运行、终态审计与有限步证据闭环（不删除 v31 及更早内容）**
>
> - `C-U1-E4-ADAM-RERUN` 已在 exact Git bundle checkout `d699bb6b1d0093d8a9b935fd6c67f049fc3c3df0` 上完成 seeds 50--69、固定/可学习方差两个 alpha 网格、三类 4000-step 控制与 45-run 方差边界稳健性审计。hardened supervisor 退出码为 0，所有预期文件、alpha 网格、Adam 字段、环境不变量和 reference regression 均通过；运行时约 30.3 分钟。
> - 科学状态登记为 **有限训练步数验证（finite-step validated）**，不是“已长期验证”。原因是受益分支未通过冻结的终态残差门禁：固定方差 `alpha=1.00` 虽达到 held-out-context reward `0.991703 [0.991363, 0.992035]`、归一化位移约 `1.0078`，但只有 3/20 seeds 同时通过两次 full-data residual audit；不存在任何受益 alpha 达到 20/20 双审计通过。
> - 有限步相变证据本身高度一致：固定方差 `alpha in {0.25,0.50,0.75,1.00}` 相对 positive-only `alpha=0` 均在 20/20 paired seeds 中提高 reward；网格峰值为 `alpha=1.00`。过强压力发生反转：`alpha=1.50` 与 `1.75` 均为 20/20 任务性能崩溃；`alpha=1.75` 为 20/20 有限参数下持续 runaway，未出现 NaN/Inf。
> - 可学习方差分支必须按事件口径读取：`alpha=0.40` 有 18/20 support contraction，中位 onset step 434.5；`alpha=0.50` 为 20/20，中位 onset step 83。45-run robustness 中 `alpha=0.38` 对 `log_sigma<-8` 为 0/15，`alpha=0.40` 为 15/15，`alpha=0.50` 对 `-14` 仍为 15/15；unexpected support expansion 与 NaN/Inf 均为 0。
> - 4000-step 附录控制：`uncontrolled_all` reward 为 `0.000000` 且 20/20 任务失败；`far_cap` reward 为 `0.995224 [0.995023,0.995416]` 且 0/20 失败；raw-gradient budget-matched global reward 为 `0.502925 [0.501994,0.503900]` 且 0/20 失败。该比较未预注册方法排名，且 raw-gradient matching 不等于 Adam-update matching，不能升级为跨任务 Distance 必然更优。
> - 三类事件继续分报：任务性能崩溃、support/variance boundary、NaN/Inf 数值崩溃。本次 NaN/Inf 总数为 0；不能把低 reward 或有限 support contraction 写成数值崩溃，也不能把同分布 held-out-context 写成 OOD。
> - 最终科学证据包为 `DRPO_CU1_E4_ADAM_D699_FINAL_EVIDENCE.zip`，SHA-256 `c2fbc594891b594652338b8937d02d4b283e75caa7cd475572ca7307f6f08673`；4 个每五 seeds checkpoint 和 raw-complete 包均已生成并校验。仓库仅保存 compact aggregate、summary、terminal audit 与 artifact index。
> - `C-U1-E4-TAPER-01` **继续阻塞**。在另行登记并批准 E4 convergence-resolution protocol，或论文明确接受 E4 只承担有限步相变证据之前，不得启动 taper 正式方法比较。

> **v31 增量登记：`C-U1-E3-ADAM-RERUN` 20-seed 终态审计、持久交付与论文口径闭环（不删除 v30 及更早内容）**
>
> - `C-U1-E3-ADAM-RERUN` 已在 run commit `ac286a46b8ffad898dfad0e7e9188b1d2e81052a` 上完成 seeds 30--49 的统一 Adam 长程运行、全量轨迹核验、reference regression 与终态审计；科学状态升级为 **已长期验证（long_run_validated）**。
> - 持久结果包 `DRPO_CU1_E3_ADAM_AC286A4_FINAL.zip` 已交付，SHA-256 为 `2b8bfdbe6f33ed1db9dc1e59f6e9fbdb6c224c7b31b1326a7f2fbaeeaaaf522b`。仓库只保存 compact summary、aggregate CSV、terminal audit 与 artifact index，不提交 checkpoints 和完整 raw trajectories。
> - 固定方差主因果链：Baseline 与 Near-zero 均为 20/20 任务性能崩溃；Far-zero 与 Far-cap 均为 0/20。终态 held-out-context reward 分别为约 `0.000002`、`0.000002`、`0.739362`、`0.733072`；各分支均无 NaN/Inf。
> - 可学习方差补充分支：Baseline 与 Near-zero 均为 20/20 首先发生支持/方差收缩，中位 onset 都为 step 73；Far-zero、Far-cap 与 Global-scale 均为 0/20 支持边界事件。100 个 method-seed runs 中 unexpected support expansion 为 0；固定与可学习方差合计 220 个 method-seed runs 中 NaN/Inf 为 0。
> - 当前论文可用口径：E3 主文采用 Fixed-variance 的 Baseline / Near-zero / Far-zero / Far-cap 最短因果链；Learnable-variance 作为互补 panel 或附录，证明远场路径还会提前触发 support contraction；Global-scale 与 Far-to-near 只作附录控制。不得把 held-out-context 写成 OOD，也不得据此声称 Distance 或任何控制方法跨任务必然最优。
> - 结果分别闭合两条受控传导路径：固定方差下的远场负影响传导为均值漂移与任务性能崩溃；可学习方差下传导为更早的支持收缩。删除近场不能救援，删除或截断远场可以救援。任务崩溃、support/variance boundary 与 NaN/Inf 继续分报。
> - 来源边界如实保留：启动环境未持有本地 Git object，source identity 由 exact committed runner blob、runner SHA-256 与 committed handoff/registry snapshots 绑定；最终聚合仅使用 JSON tuple/list 表示归一化 workaround，不改变任何 seed、配置、优化器、梯度、阈值、轨迹或结果值。
> - E3 已满足“先审计、打包、交付”的 E4 前置门禁。应用并提交本 v31 仓库闭环更新后，下一实验为 `C-U1-E4-ADAM-RERUN`；`C-U1-E4-TAPER-01` 仍须等待 E4 Adam 也完成交付。


> **v30 增量登记：Gaussian 二次临界界、C-U1 共享实现与 `C-U1-E4-TAPER-01`（不删除 v29 及更早内容）**
>
> - 本轮不产生新实验结果。`C-U1-E4-TAPER-01` 状态为“尚未运行”；此前 seeds 0--4 的独立 taper 结果只保留为开发 pilot，不能进入正式方法排名。
> - C-U1 环境几何、数据生成、Gaussian actor、log-probability、标准化距离与 Gaussian 输出 score 统一抽入 `src/drpo/cu1_core.py`。E1--E4 runner、component-wise runner 和新 taper runner 必须调用同一实现，禁止再次复制环境或 actor。
> - 当前方法比较只使用一个距离：`d_theta(s,a)=||a-mu_theta(s)||_2/sigma_theta(s)`；所有 taper 对该距离 stop-gradient。不得以 surprisal、平方 surprisal、raw reward 或另一距离替换。
> - 已解析证明：在 advantage 幅度有界且进入 support-boundary 之前 `sigma_theta(s)>=sigma_min>0` 的 Gaussian 输出区域，learnable log-scale 的负优势输出梯度为 `Theta(d^2)`。对 `w_p(d)=[1+lambda(d/d_ref)^p]^{-1}`，加权输出梯度为 `Theta(d^{2-p})`；故 reciprocal-linear 仍无界，reciprocal-quadratic 是保证有界的最低多项式阶，严格快于二次的尾部使影响趋零。
> - 同一参考衰减 `w(d_ref)=rho` 下，reciprocal-quadratic 在 `d<d_ref` 比 reciprocal-linear 保留更多，在 `d>d_ref` 抑制更强。这是有限距离的解析排序；它不推出任务 reward 排名。
> - 全参数结论只作充分界：若 actor 输出 Jacobian 的 operator norm 有界，二次 taper 也使全参数单样本影响有界。若要声称全参数增长的必要二次阶或严格下界，还需 Jacobian 在 log-scale score 方向不退化。
> - 正式主比较冻结为 reciprocal-linear 对 reciprocal-quadratic，`d_ref=5`、`rho=0.25`、`alpha=1.0`，held-out seeds 70--89；`rho in {0.50,0.75}` 和 exponential 仅作敏感性/安全类对照。Linear 的外部 prior-work 身份在精确论文公式与引用锁定前不得声称；当前名称必须写作 **reciprocal-linear baseline**。
> - `C-U1-E4-TAPER-01` 必须等待 `C-U1-E3-ADAM-RERUN` 与 `C-U1-E4-ADAM-RERUN` 均完成终态审计、打包和交付；共享 core 重构不得改变 v29 锁定的 Adam、初始化、seeds、阈值或执行顺序。


> **v29 增量记录：C-U1 E3/E4 统一 Adam 与方差坍缩口径修正（不删除 v28 及更早内容）**
>
> - 用户确认论文主线不得拆成 SGD/Adam 两套故事。C-U1 的论文主训练统一为：E2、E3、E4 使用 Adam；E1 本身不训练，只在 E2 的 positive-only actor 上计算瞬时梯度。
> - 理论主事件只有远场负优势导致的均值排斥、支持/方差收缩与 raw score-gradient amplification。旧恢复运行中 `sigma -> Inf` 的正向越界来自 plain SGD 在巨大 raw gradient 下的一步参数过冲，降级为优化器数值诊断，不得写成“方差爆炸”或第二种科学失稳分支。
> - 新正式执行 ID 为 `C-U1-E3-ADAM-RERUN` 与 `C-U1-E4-ADAM-RERUN`，状态均为“尚未运行”。旧 E3 SGD 结果与临时恢复包保留 provenance，但不覆盖 Adam 版正式结果。E3 Adam 包完成终态审计、打包并交付前，禁止启动 E4 Adam。
> - E3/E4 使用同一 Adam 参数化与同一 2000-step positive-only Adam 初始化。E2 后续 LBFGS、2× continuation 与 adaptive polish 只承担 E2 终态审计，不得静默改变 E3/E4 起点。
> - 所有 learnable-variance 边界检查覆盖完整 4096 个训练状态，并记录更新前/后的 `log_sigma_min/max`、raw gradient norm、Adam parameter-update norm、parameter finite、log-sigma finite 与 sigma-output finite。首次负向越界记为 `support_contraction`；正向越界只能记为 `unexpected_support_expansion` 并触发实现/数值审计。
> - Adam 的 raw gradient budget 与实际 parameter-update budget 不是同一量。现有 Far-cap、Global、Far-to-near 等控制仍保留作机制/附录对照，但必须分别报告 raw negative-gradient norm 与 Adam update norm；不得把 raw 等预算自动解释成 Adam 实际等步长。
> - 主文叙事简化：E3 主图只需 Baseline、Near-zero、Far-zero、Far-cap；Global-scale 与 Far-to-near 放附录。E4 主图只讲 `positive-only ceiling -> 适度负梯度收益 -> 过强负梯度导致支持收缩或任务崩溃`，不预设 Distance、Global 或其他方法排名。
> - 本版本只冻结文档、registry、代码与测试；用户提交新 commit 后才能启动 Adam 正式运行。


> **v28 增量记录：Countdown v4.2.0 一键审计式调度（不删除 v27 及更早内容）**
>
> - `EXT-C-E8-V4.1` 继续保持“尚未运行”。本轮只增加正式的一键启动与安全多 GPU 调度层，不改变模型、数据规模、seeds、门槛、SFT fallback 规则、LoRA 配置、训练步数、early-stop、pass@k、负梯度校准、方法集合或结论职责。
> - 新增唯一推荐入口 `scripts/run_countdown_pilot.py`。操作者只需提供本地 Qwen2.5-0.5B-Instruct 路径和一个新的持久 work directory；脚本自动绑定当前完整 Git SHA、调用 hardened foreground guard、选择全部可见 GPU（最多 8 张）、执行 base gate、必要时 SFT、机制门禁、四方法、best + terminal/last-finite 评估、终态审计和 durable artifact packaging。
> - runner 版本升级为 `4.2.0-one-click-audited-orchestrator`。`--gpus auto` 为默认；旧 `--gpu` 仅作隐藏兼容参数。正式模型身份不再只相信目录名，而同时要求 Qwen2 模型元数据和 tokenizer chat template；无法确认时停止，不由本地 AI 临时猜测。
> - 安全并行范围冻结为：mechanism probe 与 calibration 可并行；四种方法按一方法一 GPU 的 FIFO queue 并行；raw-base/reference test 使用空闲 GPU；所有 best/terminal/last-finite test jobs 在全部可见 GPU 上排队。`build_offline` 仍保持单 GPU、单 RNG stream，禁止临时 shard + cat，以免改变冻结离线数据生成协议。
> - 自动决定必须写入 `automatic_decisions.json`，包括 GPU 选择、模型身份、base gate/SFT 路径、离线构造并行策略和 checkpoint 评估调度。失败写入 `RUN_FAILED.json` 并由 guard 尝试恢复包；成功必须产生 `RUN_COMPLETE.json`、`terminal_audit.json`、`arena_summary.csv` 和最终 artifact ZIP。
> - 终态审计继续分别报告任务性能、结构/支持指标与 NaN/Inf 数值事件；一键化不能把三类失效合并，也不能把单 seed pilot 升级为正式多 seed 结果。

> **v27 增量记录：Countdown v4.1.2 评估语义、逐步 last-finite 与 provenance 修正（不删除 v26 及更早内容）**
>
> - `EXT-C-E8-V4.1` 的科学状态继续保持“尚未运行”。本轮只修评估定义、非有限失败 checkpoint 语义与结果状态传播；不改变模型、数据规模、seeds、LoRA 配置、训练步数、early-stop、负梯度校准、方法集合或方法职责。
> - 修正 `greedy_unseen_structure_success`：它现在必须同时满足 verifier 正确和实际生成结构不属于训练结构支持。原先“格式合法 + 使用全部数字 + unseen pattern”只代表结构出现，不代表成功；该信息单独保留为 `greedy_unseen_structure_presence`。`pass@k` 同样分离 presence 与 correct unseen success。
> - 结构评估继续采用 Park-inspired pattern-level 口径：不要求为每道题枚举全部正确表达式，而是读取模型实际生成的 canonical pattern，并分别统计发现与可靠执行。新增 greedy/sampled 分离的 per-pattern `attempts/correct/precision`、micro precision、macro precision 和 held-out family coverage；零尝试 pattern 的 precision 记为 `null`，不得与“尝试后全错”的 0 混淆。
> - `heldout_pattern_precision` 作为兼容字段，明确指向 sampled-generation 的 micro precision，不再把 greedy 与 sampled 混在同一分母中。详细 `per_pattern_precision` 随 JSON 结果保存；CSV 中以确定性 JSON 字符串记录。该改动不增加模型生成、forward、backward 或训练预算，只增加已有输出上的 canonicalization 计数。
> - 非有限失败时不再把最近一次 validation checkpoint 冒充 `last_finite_adapter`。每次 optimizer update 前只把 trainable LoRA 参数复制到 CPU；若 step 后参数变为非有限，则恢复 step 前参数并保存精确的 `last_finite_adapter`。loss/gradient 在 step 前非有限时，当前参数本身即为最后有限状态。Manifest 新增 `failure_detected_at_step` 与 `last_finite_step`。
> - 顶层 `run` 决定的 `pilot` / `engineering_smoke` 状态现在显式传入 SFT 与各 method 子进程；直接调用子命令默认标记为 `standalone_unclassified`，不得静默冒充 pilot。SFT、method 与 checkpoint manifests 使用一致状态。
> - runner 版本更新为 `4.1.2-evaluation-terminal-audit-fix`。CPU/unit tests 只验证实现，不构成 Qwen/CUDA pilot 结果；真实实验状态不升级。


> **v26 增量记录：GOV-BASE-FRESHNESS-01（不删除 v25 及更早内容）**
>
> - 新治理 ID `GOV-BASE-FRESHNESS-01` 生效；只优化 session 的 base 新鲜度发现、源码获取与更新包交付效率，不改变任何科学 claim、实验状态、冻结变量、seeds、阈值、数据规模或执行顺序。
> - base 校验从“会话启动时一次”升级为三阶段：`session_start`、`pre_execution`、`pre_delivery`。每次记录 UTC 时间、local HEAD、选定 base、权威 remote main、解析方法和状态；同一尝试使用一个 freshness ledger。
> - 若 `main` 在任一阶段前进，当前尝试立即标记 `base_advanced`，不得继续正式运行或交付旧 base ZIP。session 应自行刷新/rebase、重新读取 `AGENTS.md`、本节与 registry，并重新执行 apply、测试和最终 ZIP 校验；不能等待用户主动告知新 commit。
> - `git ls-remote origin refs/heads/main` 仍是首选。shell DNS/网络失败时，若 session 的官方 GitHub 页面、commit API、raw/download bridge 仍可用，必须通过该权威通道解析 SHA，并把解析方法写入 ledger；用户提供 SHA 只作提示或全部自动通道失败后的兜底，不得在可自动核验时把发现责任转给用户。
> - 对代码修改和 `drpo-update` 交付，官方 GitHub **完整 SHA 固定** archive 可在记录 archive SHA-256、安全完整解压、全树 inventory、文件模式和第二次独立解压 apply/test 后升级为 verified source capsule。该规则不包括移动的 branch ZIP、用户随手压缩的目录、网页片段或人工拼装 partial tree。
> - 正式实验仍只能使用 supervisor 明确支持的来源模式：exact Git checkout/bundle，或经过 supervisor 验证的 capsule mode。不能因为 archive 可用于代码 patch 就绕过正式运行 provenance 预检。
> - 在另一个 session 所述场景中，如果它确实逐项尝试并确认 existing checkout、clone/fetch、环境 download bridge、固定 SHA archive/capsule 与项目持久来源全部不可用，请求一次完整 Git bundle 是允许的最后兜底；但仅因 shell 无法解析 `github.com` 就立即要求用户上传 bundle，不符合本版规则。
> - `scripts/resolve_main_commit.py` 新增 freshness phase、ledger、官方外部 SHA 与解析方法支持；remote 发生变化时退出码为 `3`，用于机械阻止 stale-base 执行或交付。
> - 本轮 base 为 `f6590a28fb327bb4f83a6418637187f5ab2cace0`。`C-U1-E1-COMP-01` 保持 `pilot`，`EXT-H-E7-Q2` 与 `EXT-C-E8-V4.1` 保持 `not_run`；未产生新实验结果。

> **v25 增量登记：EXT-H-E7-Q2（不删除 v24 及更早内容）**
>
> - 在 E7/Hopper 的既定 learned-critic 外部机制实验中新增子 claim `EXT-H-E7-Q2`：检验真实 D4RL Hopper 数据、learned critic、状态条件 Gaussian actor 与自然 near/far negative samples 是否实际进入 log-scale 二次分支显著放大并影响优化动力学的远场区域。
> - 该子 claim 不替代 C-U1，也不重新证明 Gaussian 解析恒等式。C-U1 负责受控识别 `mean-score ~ distance` 与校正 `log-scale-score ~ standardized-distance²`；Hopper 只回答该二次分支在真实任务中是否被实际激活、是否贡献于 full-parameter gradient、support contraction、任务性能失效或数值边界事件。
> - 对 tanh-squashed diagonal Gaussian，理论检验使用冻结 inverse-squash 坐标 `u=atanh(clip(a,-1+eps,1-eps))` 与标准化残差 `z=(u-mu)/sigma`。同时保留 raw action distance 与 pre-squash distance 作为任务直观指标，但不得用 tanh 后被压缩的动作距离替代 Gaussian base-coordinate 定律。
> - component-wise 指标冻结为：mean-score norm、raw log-scale-score norm、校正量 `Q_xi=sum_j(g_xi,j+1)=||z||²`、joint output-score、full-parameter gradient norm，以及 `log-scale/mean` contribution ratio。检验 `Q_xi` 对 `||z||` 的二次关系，并用解析公式与 output-tensor autograd 交叉核对。
> - near/far negatives 必须在负 advantage 幅度上匹配；按 standardized distance 分桶并报告时间序列。必须检查二次分支增强是否先于或伴随 support contraction、mean saturation、normalized return 下降、方差边界事件或 NaN/Inf，且将任务性能崩溃、支持/方差边界事件、数值崩溃分开报告。
> - 因果验证继续使用 E7 已有职责内的 Far-zero、Far-cap 与等预算 Global control，判断削弱远场影响是否缓解上述动力学；不新增 E9，也不把机制表与标准 offline-RL normalized-return 方法表混为一谈。
> - 只有同时观察到真实 Hopper 样本进入二次分支显著作用区、该分支对实际全参数梯度或长期动力学有可测贡献，并通过 paired seeds 与终态审计，才可称为 Gaussian 二次 log-scale 远场机制的独立外部验证。若仅复核 `g_xi+1=z²` 与 autograd 一致，只能称实现一致性检查。
> - 该结果即使成立，也不能单独证明神经网络全参数梯度对距离严格二次增长，不能证明 Exp 必然优于 Linear、Global α、SBRC 或 Hybrid；神经网络 pullback 与方法排名均不属于本子 claim。
> - 本次只完成文档与 registry 预注册，E7-Q2 实现和运行状态均为“尚未运行”；base commit 为 `c7fd41ac663380de71bcd839b76ab4d1e52ae8d0`。


> **v24 增量记录：事务式替换、失败证据、launch-commit 绑定、持久权重与正式源码预检（不删除 v23 及更早内容）**
>
> - 新治理 ID `GOV-EXP-ARTIFACT-03` 生效；只修 artifact/source pipeline，不改变任何科学 claim、实验变量、seeds、阈值、数据规模、执行顺序、优先级或结果状态。
> - 新 candidate 失败时不得删除旧的有效 final ZIP；只有 candidate 与显式 sidecar 都验证通过后才允许替换。sidecar 使用新的版本化文件名，禁止覆盖已有 sidecar；若 sidecar 发布后主包替换失败，删除新发布的孤立 sidecar并保留旧主包。
> - 实验命令在 `Popen` 阶段即失败也必须生成 `RUN_FAILED.json`、traceback log、launch commit 与轻量 failed package，不能让守护器 traceback 直接结束。signal/reader setup、monitor 与 end-provenance 异常也进入同一 failed-evidence 路径。
> - 每次受监督运行必须使用全新或空的 output root；续训 checkpoint 从单独声明的持久路径读取，新尝试仍写入新目录，禁止旧 required-output、旧日志或旧结果混入新运行。
> - recovery package 的 `BASE_COMMIT.txt`、manifest 与 source snapshot 永远绑定 launch commit；若运行期间 HEAD 改变，packaging HEAD 单独记录，禁止把结束时源码冒充启动源码。
> - `experiment-final` 强制要求可解析的 `RUN_COMPLETE.json`、`run_manifest.json`、terminal audit 和至少一份日志；所有 identity-bearing JSON 的 experiment ID/base commit 必须一致。
> - 所有真实权重、adapter、optimizer state 与 checkpoint 无论文件大小，默认仅保存在持久训练服务器；主 ZIP 只记录路径、大小、SHA-256、角色和持久状态。sidecar 默认关闭，仅对预登记、显式选择且声明 `cross_machine_transfer` / `restart` / `independent_audit` 用途的文件开启，并受文件数与总大小硬门禁。foundation-model 权重永不复制。该规则不限于 Countdown。
> - 通用大文件持久状态默认值为 `persistent_local`；只有能跨 runtime 保留且项目后续可访问的服务器路径才能使用。临时容器必须显式标记为 `ephemeral` 或 `unknown`。
> - 正式守护运行必须满足二选一：显式传入本地 Git 对象库中真实存在的完整 `--expected-commit`；或在未显式传入时由 `git ls-remote origin refs/heads/main` 实时权威核验并确认与本地 `HEAD` 一致。离线 clone / Git bundle 路径必须使用显式 full SHA，禁止把任意本地 `HEAD` 默认为正式来源。
> - 所有 `--source-file` 在子进程启动前检查其是否存在于 launch commit；路径错误或文件未提交时直接结束预检。
> - GitHub 浏览器只能用于 review，不能直接形成 shell 中的 commit-bearing checkout。正式运行前依次尝试：已有 exact checkout/bundle、shell clone/fetch、环境自带下载工具取得 pinned Git bundle 或带 Git 元数据的 verified source capsule、项目持久存储中的同 SHA bundle。只有全部自动路径失败后，才可请用户提供一个完整 Git bundle/source capsule；禁止索要零散文件，也不得把普通 Source code ZIP 冒充 formal checkout。
> - 用户上传不是优先路径，也不是正式运行的必需条件；只要任一自动路径取得包含预期完整 SHA 的 exact checkout / Git bundle / verified capsule，并通过 clean worktree 与 source-file 预检，即可在不上传源码的情况下正式运行。
> - 该来源门禁解释了为什么早期 session 可以按网页代码运行、而现在正式运行会被阻断：早期流程没有执行当前 commit-bound provenance 门禁。旧结果不因新门禁自动失效，其状态仍由 handoff 中已有证据决定。
> - 所有 result kind 的 `run_manifest.json` 与状态 marker 必须绑定同一 experiment ID/launch commit；experiment ID 路径穿越、根/父目录 symlink、未知 package kind、malformed manifest、tracked runtime path 覆盖与 scan-copy 文件变更均为硬失败。
> - 小型 `.npy`/`.npz` 原始指标仍可进入主证据包；模型、adapter、optimizer/checkpoint state 无论大小均保持 index-only。stale child 在 SIGTERM 后超过 grace period 必须升级 SIGKILL。
> - sidecar verifier 要求 experiment ID、完整 base commit、显式用途与实际 payload 精确一致；每个显式选择文件的规范路径、大小和 SHA-256 均需匹配，缺失文件、额外文件或篡改 manifest 都是硬失败。
> - 本轮新增故障测试覆盖旧包保留、命令启动失败、监督器异常、运行期 HEAD 变化、final evidence 缺失、sidecar 清单与 payload 不一致、无网络且无显式 pin 的 formal run 拒绝、离线 checkout + 显式 pin 允许，以及缺失 source snapshot 启动前拒绝。
> - heartbeat 与结果活动扫描保持现状，本轮不做无明显收益的性能重构。

> **v23 增量登记：C-U1-E1-COMP-01（不删除 v22 及更早内容）**
>
> - 在不改变 C-U1 环境、数据规模、seeds、positive-only E2 训练流程、终态门禁或 E1 原 claim 的前提下，新增 E1 Gaussian 输出空间 component-wise 诊断。实验 ID 为 `C-U1-E1-COMP-01`；正式 seeds 仍为 10--29，训练/测试状态均独立采样自 `N(0,I_6)`，术语仍为 held-out-context / 未见状态泛化。
> - 诊断只检验 Gaussian 输出 score，不研究神经网络 pullback：均值分支 `||∂ log π/∂μ||=d/σ²`；共享 log-scale 分支 `∂ log π/∂logσ=d²/σ²-D`。消去学习到的方差后，精确关系为 `||score_μ||σ²=d` 与 `(score_logσ+D)σ²=d²`。
> - 主指标为 mean 分支 log-log slope、校正 log-scale 分支 log-log slope、near/far component ratios、解析式与 output-tensor autograd 误差，以及对原 E1 joint output-score ratio 的重构误差。
> - **2026-06-25 pilot 结果（20 seeds，10--29）：**20/20 seeds 的科学门禁和终态审计通过。E2 held-out-context reward=`0.646788 [0.646657,0.646920]`，learned sigma=`0.190726 [0.190712,0.190740]`，最终 full-data positive-gradient norm 均值 `6.44e-4`、最大 `9.23e-4 < 1e-3`，20/20 均为 `stable_plateau_2x_confirmed`。
> - E1 advantage 等值误差最大约 `2.09e-7`；raw-distance far/near=`3.797862 [3.794213,3.801766]`，mean-score far/near=`3.797862 [3.794213,3.801765]`。`||score_μ||σ²` 对 `d` 的 log-log slope=`1.00000000`。
> - 原始 log-scale 绝对值 far/near=`19.970219 [19.917146,20.026194]`；去除解析常数项后的 corrected term `d²/σ²` far/near=`14.435378 [14.407362,14.465327]`，其 log-log slope=`2.00000000`。因此原始 log-scale score 应称“远场渐近二次”，corrected term 对距离平方则为精确恒等式。
> - joint output-score far/near=`7.563755 [7.554059,7.574104]`，与 v17 正式结果 `7.5638 [7.5538,7.5737]` 对齐；解析式与 output-tensor autograd 最大相对误差 `2.48e-7`，component 重构原 joint ratio 的最大误差 `9.54e-7`。
> - **Hopper 独立验证边界：**若 E7 在 D4RL Hopper 的真实离线数据、learned critic 和独立训练 actor 下，同样观察到远场 log-scale 分支进入平方距离主导区，并且该分支对实际全参数梯度或动力学具有可测贡献，则可称为该机制在外部环境中的独立验证/外部复现。它独立验证的是“真实任务确实进入并受该二次主导区影响”，而不是再次证明 Gaussian 解析恒等式；后者由策略族本身决定。
> - **方法边界：**该结果证明 Gaussian 输出空间本身具有非线性远场放大，不需要诉诸神经网络 pullback；它支持采用非线性、正值、平滑尾部控制作为机制动机，但不能单独推出 Exp 必然优于 Linear、Global α、SBRC、Hybrid 或 Positive-only。
> - **代码与 rebase provenance：**pilot 运行绑定到 `1962442aea7037fac6b57e4e9232850c69e5c1b9`。当前更新包 rebased 到其直接后继 `a9e0d860a6f03d1be12280885002c24ba2f1b66a`；该后继只修改 Countdown、handoff、registry 与 Countdown tests，未修改 C-U1 runner。结果与 v17 正式 E1/E2 数值对齐是强一致性证据，说明未观察到来源重建导致的科学偏差；但按照既有治理规则，clean committed rerun 之前科学状态仍保留为 `pilot`，不得把数值一致性等同于完整 provenance 证明。

> **v22 增量记录：Countdown v4.1.1 负优势尺度校准与执行配置锁定（不删除 v21 及更早内容）**
>
> - `EXT-C-E8-V4.1` 的科学状态保持“尚未运行”，实验职责不变。v22 只修正文档—代码口径并冻结 pilot 执行配置，不产生实验结果。
> - 旧代码中的 `alpha=0.7` 是继承的工程默认值，没有经验或理论证据支持，**不得视为已冻结科学设置**。机制 probe 继续严格使用每个 near/far 样本固定 `A=-1`；方法训练另设共同负分支尺度 `beta`。
> - `beta` 不在训练结果或 test 上调参。它在共同初始 adapter、固定 training calibration subset 上、任何方法训练开始前自动计算一次：令未缩放 Uncontrolled 负分支的 RMS 梯度范数与正分支 RMS 梯度范数相等，即 `beta = G_pos_rms / G_neg_uncontrolled_rms`。随后同一个 `beta` 对 `controlled_negative`、`uncontrolled_negative` 和 `global_matched` 全部冻结；校准非有限或非正时直接停止，不静默回退到 `0.7`。
> - `global_matched` 的 `gamma = G_neg_controlled_rms / G_neg_uncontrolled_rms` 继续在同一 calibration subset 上计算并冻结；因此 `beta` 回答共同正负预算尺度，`gamma` 回答选择性远场控制与等预算全局缩放的对照，两者不得混淆。校准不读取 validation/test task outcome。
> - 当前 0.5B BF16-LoRA pilot 配置冻结为：train/val/test=`6000/500/1000`，offline matched rows=`1500`，rollouts=`12`，pair resample rounds=`3`，calibration batches=`16`，method max/min steps=`1200/400`，eval every=`100`，early-stop patience=`6`、delta=`0.002`，selection metric=`greedy_success`，pass@k=`8`，method LR=`5e-5`，warmup ratio=`0.03`，max grad norm=`1.0`，near/far mix=`0.5/0.5`，far taper lambda=`0.7`，surprisal threshold=`2.0`。
> - LoRA 配置冻结为 rank=`32`、LoRA alpha=`64`、dropout=`0.05`，target modules=`q/k/v/o/gate/up/down_proj`；四方法共享相同初始化、离线数据、训练顺序 seed 和评测 seed。Pilot development seed 固定为 `1234`；未来正式 paired held-out seeds 尚未登记，故 formal multi-seed 运行继续被门禁阻止。
> - v12/v18 中“`/mnt/data` v3 入口、强制先 SFT、3B 主 arena、八方法比较、只评最佳 checkpoint、QLoRA 自动进入正式排名”等描述明确标记为历史 provenance，不得执行。当前唯一入口为 `src/drpo/countdown_qwen_arena_onefile.py`。
> - v4.1.1 runner 只实现 BF16-LoRA pilot。0.5B full fine-tuning confirmation 尚未实现；仅当 LoRA pilot 出现可复现信号后，才另行登记、实现并运行，且不得与 LoRA 方法混入同一主比较。
> - 对 `EXT-C-E8-V4.1`，v20 通用 artifact 预算中的 `latest` 角色具体化为：正常结束保存真实 `terminal_adapter`，非有限失败保存 `last_finite_adapter`；它们与 `best_adapter` 互斥组合，因此每方法仍最多保留两个本地 checkpoint。

> **v21 增量记录：Countdown v4.1 审计式 pilot 协议（不删除 v20 及更早内容）**
>
> - `EXT-C-E8-V4` 标记为已替换；新的执行 ID 为 `EXT-C-E8-V4.1`，科学状态仍为“尚未运行”。该实验只承担 Transformer/token 外部有效性，不替代 D-U1/D-Diag 的受控机制识别。
> - Base-first 保持不变：先零训练评测 Qwen Instruct 0.5B；仅当 Base 未过既有能力门禁时执行最小 SFT fallback。Pilot 四条训练线冻结为 `positive_only`、`controlled_negative`、`uncontrolled_negative`、`global_matched`，全部共享同一 BF16 LoRA 参数化、初始 adapter、离线数据、数据顺序与评测 seeds。
> - Near/far negative pair 必须满足预登记的 surprisal gap、token 长度、树深与数值误差匹配。候选不足时追加采样；达到冻结重采样轮数仍无合格 pair 时丢弃该题。`pair_matched=false` 不得进入 mechanism probe 或主训练。
> - `global_matched` 在固定 calibration split 上匹配 `controlled_negative` 的 RMS 负梯度预算，再冻结同一个 `gamma` 等比例缩放 near 与 far；它不读取 near/far 身份，用于区分选择性远场控制与单纯减少总负更新，不预设方法排序。
> - 结构协议采用 Park-inspired canonical pattern、pattern-first 容量审计、近似平衡生成和 held-out pattern-family 拆分。只采用其数据控制工具，不继承更强的 latent-skill 或 OOD 因果主张；当前术语为 **held-out canonical pattern-family generalization / 未见规范结构族泛化**。Held-out family 不得出现在训练 positive 或 negative completion 中。
> - Adapter/checkpoint 二进制只保存在服务器本地或外部持久存储，不提交 GitHub、不进入普通代码更新包。正常结束每方法保留 `best_adapter` 与真实 `terminal_adapter`；非有限失败时保留 `best_adapter` 与 `last_finite_adapter`，符合每方法最多两个 checkpoint 的 v20 artifact 预算。Manifest 只记录路径、step、大小、SHA-256、base model 与 adapter 配置。
> - Pilot 全部方法统一 BF16 LoRA；出现可复现信号后才在 0.5B 上执行统一 full fine-tuning 确认。不得在同一主比较中混用 LoRA 与 full FT。
> - 静态检查、CPU selftest、硬件 smoke 和单 dev seed 均不构成正式结果；正式结论仍需 paired held-out seeds、终态审计和持久 artifact。


> **v20 增量记录：提交身份、原子打包与大文件边界加固（不删除 v19 及更早内容）**
>
> - 新治理 ID `GOV-EXP-ARTIFACT-02` 生效；它只加固代码来源与 artifact 交付，不改变任何实验 claim、变量、seeds、阈值、数据规模或科学状态。
> - 当前线上 `main` 基准由用户确认并绑定为 `398a8e1dc5990fc0a2198494701d05cc27b1b73e`。以后不得用网页 commits 列表或搜索索引单独判定最新 SHA；优先使用 `git ls-remote origin refs/heads/main`，再用本地 `git rev-parse HEAD` 交叉检查。网络不可用时必须明确标为未完成远端核验。
> - 正式实验只能从 clean worktree 启动；正式重跑前须形成新的 commit 边界。dirty worktree 只允许显式 pilot，并在进程启动前保存 tracked/staged binary diff、受限 untracked 文件副本和 SHA-256。
> - 正式进程结束时再次核对 HEAD 和 worktree；运行期间 commit 或代码状态变化则标记 `provenance_compromised`，即使子进程返回 0 也只能进入 failed recovery 路径。
> - 主 ZIP 先生成 candidate，由打包器内部执行安全路径、checksums、base commit 和 `git apply --check`；全部通过后才原子重命名为最终 ZIP，验证失败不得留下看似可交付的正式包。
> - failed/checkpoint/raw-complete 包默认采用轻量证据清单；大 adapter、checkpoint、optimizer state、数据集和缓存不进入主包，而写入 `LARGE_FILE_INDEX.json`。需要续训的单个 checkpoint 作为 sidecar 独立交付并记录 SHA-256、大小和持久状态。
> - Countdown 等大模型实验必须预登记 artifact budget 和 checkpoint retention；默认每方法最多保留 best 与 latest 两个 checkpoint，不保存 foundation model，不默认保存 optimizer state。
> - 打包器绝不自动跟随软链接：指向结果目录外部的 symlink 直接拒绝；内部 symlink 只记录引用，真实目标最多打包一次。
> - 主 ZIP 默认硬上限 25 MiB，单文件主包默认上限 10 MiB；超限必须转轻量/sidecar 或打包失败，不能只在产出后警告。
> - `run_experiment_guard_hardened.py`、`package_experiment_hardened.py` 与 `verify_experiment_package_hardened.py` 必须调用同一 hardened 实现；核心模块缺失时必须 fail closed，禁止静默退回旧协议。
> - ChatGPT 更新包交付前必须在基于确认 base 的全新 clean checkout 中保留 Git executable modes，实际应用 `update.patch`，执行包内 `TEST_COMMANDS.sh`、`git diff --check` 和最终 ZIP 校验；任一步失败都不得发布可下载正式包。
> - v20 进一步锁定 exact-base 门禁：clean checkout 必须来自真实 Git clone/fetch 或固定到完整 SHA 的 GitHub source archive；禁止用网页解析片段、人工重建仓库或不相关本地 commit 生成补丁。交付前必须在第二份同 SHA checkout 中完整复现用户的 `git apply --check → apply → TEST_COMMANDS.sh` 合并路径。
> - 守护器 heartbeat 与递归输出活动扫描保持现状，本轮不做低收益性能重构。

> **v19 增量记录：正式实验守护与可持久交付协议（不删除 v18 及更早内容）**
>
> - 2026-06-25 的一次 C-U1 E3/E4 临时运行曾在容器内产生逐 seed 文件计数，但未在运行结束后立即形成可下载结果包；运行时被回收后，原始文件无法审计。聊天中的计数和“已落盘”表述不能替代正式证据。
> - 该事故不自动推翻此前已经登记且有独立材料支撑的科学结论，但该次未交付运行不得用于升级、替换或重新确认任何正式结果；E4 仍不得据此宣称完成。
> - 新治理 ID `GOV-EXP-ARTIFACT-01` 生效：正式实验在临时环境中必须由前台守护程序持续监控；计算成功或失败后都必须立即生成可持久下载包；仅写入临时目录不算持久化。
> - 正式实验采用双状态轴：科学状态继续使用既有六类标签；执行证据状态使用 `registered → running → raw_complete → terminal_audited → packaged → delivered → applied_to_repository`。
> - 只有达到 `packaged + delivered` 才能对外称“本次正式运行完成”；只有真实 commit/push 成功后才能称“已进入仓库”。`raw_complete` 必须写成“计算完成但尚未持久交付”。
> - 每个 experiment ID 都是交付边界：E3 完成后必须先打包并交付，才能启动 E4。预计超过 30 分钟的运行默认每 5 个正式 seeds 生成一个恢复 checkpoint 包，除非预注册了其他间隔。
> - 失败运行、收尾绘图错误和聚合错误同样必须保存 partial raw outputs、日志、traceback、代码 SHA 与缺失文件清单；不得因最后一步失败而丢弃已完成训练。
> - 详细操作规范见 `docs/formal_experiment_artifact_protocol.md`；机器可读规则写入 `experiments/registry.yaml`；统一守护、打包和验证入口分别为 `scripts/run_experiment_guard_hardened.py`、`scripts/package_experiment_hardened.py` 和 `scripts/verify_experiment_package_hardened.py`。


> **v18 增量记录：Countdown 0.5B base-first 最小外部验证协议（不删除 v17 及更早内容）**
>
> - 用户明确将 EXT-C / E8 收缩为“两三天内可得结论”的最小外部验证：先运行固定负优势的 near/far 机制迁移 probe，再运行 Positive-only / Controlled-negative / Uncontrolled-negative 三组配对效果实验；不再默认启动八方法大 arena。
> - 当前主模型改为 Qwen Instruct 0.5B。先对未经 Countdown 训练的原始 Base checkpoint 做零训练评测；若验证集 greedy verifier success `>=0.15` 且 valid rate `>=0.80`，直接从同一个未训练 LoRA adapter 分叉三种方法，不执行 SFT。Base 未过门槛时才允许最小 SFT fallback，SFT 后 greedy success 仍须 `>=0.15`。
> - train / validation / test 按 canonical operator-tree signature 严格分离；三组 oracle 结构集合互不重叠。当前口径为 **held-out structural generalization / 未见组合结构泛化**，不是状态分布 OOD。
> - 离线正样本只能来自训练结构支持；near/far negatives 均须语法合法、恰好使用全部数字、verifier reward 同为 0，并尽量匹配 token 长度、树深和数值误差，只以冻结参考策略 surprisal 区分近远。
> - 机制 probe 固定负优势 `A=-1`，报告参考 surprisal、direct-logit score、实际可训练 adapter 参数梯度范数、相同负更新下目标 surprisal 增量，以及对正确表达式 surprisal 的 collateral change。该 probe 只回答 D-U1 结论能否迁移到 Transformer 共享参数环境，不替代 D-U1 的受控因果识别。
> - Controlled-negative 保留 near negative，按当前 token surprisal 对 far negative 做 detached exponential taper；Uncontrolled-negative 对 near/far 不做控制；三组方法使用同一初始 adapter、同一离线数据、同一数据顺序 seed、同一验证与生成 seeds。
> - 主效果指标为 greedy verifier success、pass@k、valid rate、greedy/pass@k unseen-structure success；task-performance、support/structure coverage 与 NaN/Inf 数值失败分开记录。
> - 3B 降为条件复验，7B 为可选规模验证；0.5B 只要通过能力和数据门禁即可承担当前 E8 主实验。旧 3B-primary / 7B-confirmation 方案保留作 provenance，但由本版本覆盖。
> - 代码入口仍为 `src/drpo/countdown_qwen_arena_onefile.py`。v4 代码已完成 Python 编译、CPU selftest 与单元测试；真实 Qwen/CUDA/LoRA 运行尚未执行，结果状态仍为“尚未运行”。


> **v17 增量记录：重建代码正式重跑 E1/E2（不删除 v16 内容）**
>
> - 首次固定 100-step full-data polish 后，E2 有 7/20 seeds 未通过预注册 `<1e-3` 终态门禁；该批结果不接受。
> - 改为 adaptive polish（至少 100、每 25 检查、最多 500）后从空目录重跑，20/20 seeds 全部通过；最终全数据 positive-gradient norm 均值 `6.44e-4`，最大 `9.23e-4`。
> - E1 正式 20-seed：advantage far/near=`1.000000`；policy-output score far/near=`7.5638 [7.5538,7.5737]`。全参数倍率仅作附录诊断，不扩展为参数内部优化研究。
> - E2 正式 20-seed：held-out-context reward=`0.646788`，sigma=`0.190726`，沿隐藏最优方向的归一化外推位移约 0，phantom-gradient growth=`29.525 [28.162,30.954]`。
> - 新重跑的 held-out `distance_to_a_plus≈0.01954` 与丢失 driver 的旧报告 `≈0.00131` 不一致，已如实登记；它仍仅为 `a_plus→a_star` 间隔的约 2.8%，不改变 imitation-ceiling 结论。
> - 所有测试状态与训练状态同分布，继续统一称 held-out-context / unseen-context generalization，不称严格 OOD。


> **v16 增量记录：C-U1 正式重跑前代码审计（不删除 v15 内容）**
>
> - 正式 20-seed 重跑尚未启动；先完成重建代码的实现审计与正式规模单 seed 回归。
> - 修复 E4 continuation 后缺少第二次 stationary audit、不同 alpha 使用不同 minibatch 序列、任务失效 onset 计算、方差事件混称、audit 跳变污染时间斜率等问题。
> - Positive-only 不再把 `<1e-3` 判据私自放宽为 `<5e-3`；增加全数据 deterministic polish 后严格执行原判据。
> - 方差稳健性检查分别记录 support contraction / expansion，并新增更小学习率；有限时刻 reward 与渐近稳定性继续分开。
> - E1 主结论聚焦 policy-output score geometry；全参数 pullback/Jacobian 分解不作为本课题新增主问题。
> - 单 seed 回归保持 E1–E4 的主要定性机制，但若新正式数值与丢失 driver 的旧汇总不同，必须如实登记，禁止为追旧数值而反向调参。
> - 代码、协议、SHA256 和逐步轨迹在正式运行开始前即写入结果目录。

**文档定位：唯一研究主轴、不可破坏性删除、面向论文重写。**  
**恢复底稿：v7 全量累积文档；后续将 v8-v10 的有效新增内容作为可追踪补丁合入。**  
**当前日期：2026-06-25。**



> **v15 变更记录（2026-06-24，不删除 v14 及更早内容）**
>
> - 锁定 C-U1 的泛化术语：训练状态与测试状态均独立采样自 `N(0,I_6)`；当前结果只能称为“同分布 held-out-context generalization / 未见状态泛化”，不得称为严格 OOD generalization。
> - `a_star(s)` 未作为正样本提供，只说明“隐藏最优动作/未展示动作目标”；它不自动构成状态分布 OOD。
> - 当前主线所有 E1-E4 的 OOD 表述均由本版本正式口径替代。Part II 历史原文中的旧 OOD 说法保留作 provenance，但不得复制进新论文；若要恢复 OOD claim，必须新增显式 distribution-shift protocol。
> - 审核用户上传的 `drpo_unified_continuous_environment_v1 (1).py`：其几何骨架与 C-U1 方向一致，但正式配置不一致，不能直接复现 E1-E4。
> - 在该文件基础上重建单文件一键脚本 `drpo_cu1_e1_e4_oneclick.py`；默认命令无需参数，自动运行环境审计、E1-E4、方差边界稳健性、逐 seed 轨迹、汇总、图表和 reference regression。当前状态为“静态检查与 CPU smoke 通过，正式 20-seed 重跑尚未执行”。
> - D-U1/E6 继续暂停，直至用户审阅并确认一键脚本。

> **v14 变更记录（2026-06-24，C-U1 审计与恢复版）**
>
> - 按用户要求，当前主线不再使用已撤出的绘图归一化希腊符号；统一写作“沿隐藏最优方向的归一化外推位移”，代码字段为 `normalized_extrapolation_displacement`。旧版本/历史附录中的原符号只作为 provenance 保留，不再沿用。
> - E2 论文成熟度修正为“高（恢复复现材料后可入正文/附录）”；此前“较高”混淆了科学成熟度和当前文件工程完整性。
> - E3 可学习方差术语改为“方差/支持收缩边界事件”与“联合稳态不存在”，不再把到达 log-standard-deviation 边界直接等同 NaN/Inf 数值崩溃。
> - E4 可学习方差结果强制拆分有限时刻性能与渐近稳定性；未达到稳态的配置不得凭截面 reward 进入稳定方法排名。
> - 完成当前运行时全盘代码恢复搜索；统一 C-U1 源码未在当前挂载中找到，详细记录见 `drpo_experiments/C_U1_CODE_RECOVERY_SEARCH_LOG.md`。
> - D-U1/E6 继续暂停。

> **v13 变更记录（不删除 v12 及更早内容）**
>
> - 用户于 2026-06-24 明确确认 C-U1 方案并解除正式训练门禁；授权在无卡点且结果与预登记理论一致时自主推进。
> - 执行顺序冻结为：C-U1 单 seed 回归 → E1 → E2 → E3 → E4 → E6 categorical 长程 → Hopper；每完成一个实验立即汇报。
> - 若出现理论方向反转、协议无法同时满足、关键实现歧义、依赖/算力阻断或需要改变核心变量，则暂停并请求用户决策。
> - C-U1 正式数据规模冻结为 4096 train states / 4096 test states；每状态 4 个正动作、8 个等奖励轮廓负动作。正式 held-out seeds 为 10–29；开发 seeds 为 0–4。
> - 固定 advantage 由 ground-truth reward 减固定 baseline 得到并在训练前冻结；主实验不引入 learned critic、importance ratio 或 PPO clipping。
> - 本版本开始创建统一代码、配置、逐 seed 曲线、汇总与失败运行索引；任何实验结果仍须按状态标签回写。
> - E3 实现调试时曾使用 seed 10 做单 seed smoke，导致原定 E3 held-out 10–29 不再严格盲测。为保持实验完整性，E3 正式 held-out seeds 改为 30–49；E1/E2 的 10–29 未参与其自身调参，结论不受影响。

> **v12 变更记录（不删除 v11 内容）**
>
> - 将第 0 节改为长期研究与执行原则；v11 的恢复反思整体移动到附录。
> - 明确 6 维 numerical state/context 的含义、训练/测试状态的用途，以及“状态数不等于样本数”。
> - 将 4096/4096 状态、每状态 4 正/8 负标记为**待确认正式配置**，不是已锁定事实。
> - 在 E3 中正式区分“任务效果崩溃但数值训练仍可运行”与“任务效果和数值训练同时崩溃”；该区分是用户提醒后补入的协议细化，旧 E3 未明确登记。
> - 纠正外部验证问题：Hopper 只重复可识别的连续机制子链，不机械复制理想环境全部实验，也不替代理想环境。
> - 登记 Countdown 单文件 v3 的运行计划、硬件策略与当前验证边界。
> - 本版本仅同步计划和代码状态，**未启动 C-U1 正式训练**。

> **治理规则（锁定）**
>
> 1. 本文档中的历史结论、相关工作、实验设计与结果不得直接删除。发现错误时采用“原结论—问题—修正—证据”的替代记录。
> 2. 删除、合并或降级任何段落前，必须列出拟删除内容、理由和替代位置，并取得用户确认。
> 3. 新理论、新实验、新变量必须先写入“变更提案”，说明必要性、与原体系关系和验证计划；核心变量须经用户确认。
> 4. 所有任务只从第 6 节“论文目标—理论—实验总表”领取；对话中的临时发现必须回写本文档，不能只留在聊天记录。
> 5. 实验状态只允许使用：已解析证明、已长期验证、有限训练步数验证、pilot、尚未运行、已否定/已替换。

---

# 0. 研究与执行原则（每次新会话首先阅读）
<!-- HANDOFF-DELTA-BLOCK:after_heading:e7-q2-fixed-budget-longrun-v43:START -->
> **E7-Q2 v4.3 增量登记：`EXT-H-E7-Q2` fixed-budget long-run v4.3 与重跑协议（不删除此前任何内容）**
>
> - 本版继承当前 `main` 及此前全部历史，只修订 Hopper E7-Q2 的训练停止规则、critic canonical checkpoint 选择、终态审计职责和一键执行配置。E7 仍只承担 learned-critic 外部机制验证，不替代 C-U1 受控因果识别，也不构成 D4RL 方法排名。`EXT-H-E7-Q2` 继续是 **not_run + implemented + ready + active**；本版未运行真实 Hopper formal。
> - **旧协议—问题—新证据：**v4.2 由短窗口 stationarity candidate 和 2× extension 决定 critic/actor 提前停止。用户上传的 formal-scale pilot 中，critic 在 7600 步即被判为 terminal，而上一轮 20000 步 critic 的 test R²/Pearson 仍更高；Positive-only、Far-zero、Far-cap、Dynamic Global 与 Signed/Near-zero 又停在不同 horizon，导致“最终值”不处于相同训练预算。该结果说明短窗口暂时变慢不能代替 D4RL 长程预算，也不能作为方法间公平终态比较。
> - **Pilot 机制证据边界：**上传包 SHA-256 为 `deefbe216ca5c99622c84831b4546da10203610c07736992c51cf23f679f1017`。该 pilot 中 far/near `|A|` 约为 `0.99992`、标准化距离约为 `3.596×`、全参数负梯度约为 `3.47×`；Signed 与 Near-zero 均出现 `10/10` 任务性能崩溃和 `10/10` support/variance boundary，Far-zero 将 support/variance boundary 降为 `0/10`。这些只登记为 **pilot / finite-horizon mechanism diagnostic**，不升级为 formal result、稳态结论或方法排名。NaN/Inf 与任务崩溃、support/variance boundary 继续分开报告。
> - **Critic v4.3 固定预算：**formal canonical critic 固定训练 `100000` optimizer steps，每 `2000` 步评估一次；除 loss/gradient/parameter 出现 NaN/Inf 等数值失败外禁止提前停止。跑满后始终选择最低 validation MSE checkpoint 生成 frozen advantage；final checkpoint 仅用于 selected-vs-final 稳定性对照。旧 optimization-terminal、validation R²/Pearson、final/best ratio、advantage sign/rank/Jaccard 阈值继续原样记录，但全部降为 report-only diagnostics，不再阻塞 formal actor 执行。formal operational gate 只要求固定预算完成且 selected checkpoint 指标有限。不得把固定 100k 写成“critic 已收敛”。
> - **Actor v4.3 固定预算：**Positive-only initialization 固定 `100000` optimizer steps；从同一 fixed-budget checkpoint 分叉的 `signed / near_zero / far_zero / far_cap / dynamic_budget_matched_global` 各固定 `200000` steps，所有分支 horizon 完全相同。actor 每 `5000` 步做 audit、每 `25000` 步以 `5` episodes 做中间 rollout，固定预算末端以 `20` paired episodes 做最终 rollout；只有 NaN/Inf numerical failure 允许提前停止，support boundary、任务退化或持续漂移不得触发早停。`signed` 明确定义为保留正负 advantage、且不做 near/far 控制的 full signed-advantage baseline，不是新算法。
> - **终态审计职责：**terminal candidate、relative update、state drift 与 2× continuation 只用于训练结束后的分类，不再控制停止。满足 2× confirmation 且无 boundary 才可标为 `finite_terminal`；跑满固定 horizon 但仍漂移时标为 `persistent_or_slow_drift`，无法判定时标为 `fixed_horizon_inconclusive`。固定 horizon 本身不得自动解释为 convergence。根审计分别记录 critic fixed-budget completion、Positive-only fixed-budget completion、所有 branch fixed-budget completion、任务性能崩溃、support/variance boundary、NaN/Inf 与 terminal classification。
> - Canonical critic artifact schema 升级为 `v3`；v2、pilot、不同 mode/config/dataset/transition count/seed/runner identity 的 artifact 均 fail closed。Countdown 风格入口仍为 `python3 scripts/run_e7_hopper_q2.py`；默认 formal，通过 hardened guard 持久化 heartbeat、失败证据和最终 raw-complete 包。应用本更新后必须从 clean current `main` 重新训练 critic 与全部 actor 分支，旧 v4.2 critic 不得跨 schema 复用。
<!-- HANDOFF-DELTA-BLOCK:after_heading:e7-q2-fixed-budget-longrun-v43:END -->
<!-- HANDOFF-DELTA-BLOCK:after_heading:v68-ext-h-e7-q2-longrun-closure:START -->
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
<!-- HANDOFF-DELTA-BLOCK:after_heading:v68-ext-h-e7-q2-longrun-closure:END -->
<!-- HANDOFF-DELTA-BLOCK:after_heading:v69-e7-bench-parallel-pilot:START -->
> **v69 增量登记：`EXT-H-E7-BENCH-01` 两数据集并行 Pilot 与正式并行拓扑（不删除 v68 及更早内容）**
>
> - 本版不新增顶层实验 ID；在既有 `EXT-H-E7-BENCH-01` 下登记一个 **pilot** 子阶段。Pilot 只检查数据加载、learned-critic/actor/rollout 链路、连续 taper 实现、运行成本、artifact 体积、断点恢复及初步 paired direction，不得填入正式 9-task 主表，不得据此更换方法族、按任务调参或升级正式科学状态。
> - Pilot development seeds 冻结为 `200, 201, 202, 203`。方法冻结为 `Positive-only`、`Signed`、`Global alpha=0.75`、`Reciprocal-Linear`、`Reciprocal-Quadratic`、`Exponential`。三种 taper 沿用 `C-U1-E4-TAPER-NEAR-RETENTION-01` development seeds `0--4` 的冻结系数：`0.4362580032734791`、`0.5520268617673281`、`0.374162511054291`；标准化距离 reference/near boundary 均为 `5.0`，禁止 D4RL 后验重调。
> - 两个上传数据单元必须按真实 provenance 区分：`hopper-medium-expert-v2` 是 legacy D4RL-v2 HDF5，使用 Hopper-v4 与 D4RL-v2 normalized return；上传的 `mujoco/hopper/medium-v0` metadata 明确属于 **Minari Hopper-v5**，不是 D4RL `hopper-medium-v2`，因此只作为 pilot/plumbing cell、只报告 raw return，不能计入正式 D4RL 9-task 主表。正式 Hopper-medium cell 仍需另行冻结精确 D4RL 版本。
> - Pilot 固定预算为：每数据集一个 canonical critic `20k` optimizer steps、每 `(dataset, seed)` Positive-only `20k` steps、其余每个 method branch `40k` steps；只有 NaN/Inf 可提前终止。固定 horizon 不等于收敛，仍需分开报告任务性能崩溃、support/variance boundary、NaN/Inf 与 persistent/slow drift。
> - 为使用 384 核 CPU，执行器冻结为三阶段并行：`2` 个 dataset critic workers 并行；`8` 个 `(dataset, seed)` Positive-only workers 并行；`40` 个 `(dataset, seed, method)` branch workers 并行。线程预算分别为 `64/32/8`，峰值 `320` threads，保留系统和 I/O 余量。seed 与 method 均禁止顶层串行；每个 branch 从对应的同一 Positive-only checkpoint 分叉，输出目录隔离，resume 粒度为 `dataset_seed_method`。
> - 正式 9-task E7-BENCH 同步登记为 staged resource-pool 并行，branch scheduling unit 为 `task_seed_method`，禁止 serial seed loop 与 serial method loop；但正式 exact D4RL versions、formal seeds、offline-RL base、optimizer 和 full budget 尚未冻结，故 formal activation 继续 blocked。Pilot ready 不等于 formal ready。
> - 新入口为 `src/drpo/e7_bench.py`、`scripts/run_e7_bench.py`，配置为 `configs/e7_bench_pilot.yaml`，协议说明为 `docs/e7_bench_pilot.md`。当前仅完成实现、静态/单元、真实数据 loader 与 canonical critic 短程 smoke；当前环境缺少 `gymnasium`，因此 actor/rollout 短程 smoke 未执行。该限制不等于 Pilot 已运行，更不支持任何方法优于 Positive-only。正式启动时 runner 会在长程 critic 之前预检 384 核线程预算、Gymnasium/MuJoCo 环境及数据—环境维度一致性。
<!-- HANDOFF-DELTA-BLOCK:after_heading:v69-e7-bench-parallel-pilot:END -->
<!-- HANDOFF-DELTA-BLOCK:after_heading:v70-du1-e6-cartesian-taper:START -->
> **v70 增量登记：D-U1 E6 `utility × surprisal` 二维笛卡尔积重构与 E6-TAPER 联合正式协议（不删除 v69 及更早内容）**
>
> - 旧 `D-U1-E6-SEMANTIC-LONGRUN-01` 的长期结果完整保留，继续只支持“共享语义表示下，适度方向可靠负信号可突破 Positive-only ceiling，过强负压力会反转”的历史结论。重新审计发现：旧 E6 把 `local/far` 同时用于语义方向角色，未把 directional utility 与 learner-relative rarity/surprisal 做成严格二维笛卡尔积，因此它不能单独承担 E4 的离散对应，也不能直接作为 surprisal taper 的严格方法证据。
> - 原未实现、未运行的 `D-U1-E6-TAPER-01` development preregistration 原样保留，不改写其历史字段；新正式 successor `D-U1-E6-CARTESIAN-TAPER-01` 取代其后续执行职责，并在同一个 D-U1 环境和一次正式执行中分开报告 E6-Cartesian 机制块与 E6-TAPER 方法块。该合并避免先看正式机制结果再调 taper，同时满足“文档先于实验”和 formal artifact 串行交付门禁。
> - 新环境使用 32 个 semantic prototypes，每个 prototype 复制为 `common/rare` 两个 unordered categorical action，共 64 actions。复制对具有完全相同的 reward embedding、directional utility 与 fixed negative advantage。策略从 trainable semantic logits 中减去冻结的初始化 reference logits，使 step 0 的 useful/unhelpful 在同一 rarity level 上概率逐状态精确相等；唯一初始 rarity 差异是 trainable action-logit bias 的固定 `4.0` gap。由此形成四格：`useful_common / useful_rare / unhelpful_common / unhelpful_rare`，每个 context 每格一个负样本，数量与 advantage 严格匹配。
> - utility 轴定义为：在策略方向位于 `t_plus` 附近时，排斥该 semantic prototype 的方向在 `t_plus -> t_star` 上的投影；每个 context 分别选择最大 utility 与最小 utility prototype。rarity 轴只由当前 categorical policy 的 surprisal `-log pi_theta(a|s)` 定义：每次 forward 都在同一 semantic replica pair 内把较高概率成员动态标为 common、较低概率成员动态标为 rare，离散选择 stop-gradient；不再用固定 action ID 或 semantic distance 冒充 policy remoteness。
> - E6-Cartesian 机制块比较 `positive_only`、四个单格方法、`useful_all / unhelpful_all / common_all / rare_all / all_negative`，回答同一 utility 下 rare 是否更危险、同一 rarity 下 useful 是否更有益，以及 utility × rarity 是否存在交互。子集干预只把未选 cell 置零，保留 cell 始终维持其在四格总体中的 `1/4` 系数，禁止因删除其他 cell 而自动重归一化负梯度预算。
> - E6-TAPER 方法块在同一四格数据、同一初始化、同一 minibatch stream 与同一 seeds 上比较 `global_matched / reciprocal_linear / reciprocal_quadratic / exponential`。所有 taper 使用同一个 detached normalized excess surprisal：每 seed 在训练前以 common median 为 threshold、以 rare-minus-common median 为 scale 并冻结校准常数；surprisal 本身在每个 optimizer step 由当前 learner 重新计算。
> - 公平性冻结：common reference coordinate `u=0` 时四种选择性方法 retention 均为 `1`；reference rare coordinate `u=1` 时 reciprocal-linear、reciprocal-quadratic、exponential retention 均为 `0.25`，对应系数分别为 `3.0 / 3.0 / ln(4)`。`global_matched` 只在 step 0 用训练 audit subset 匹配 exponential 的 raw negative-gradient norm，随后冻结；不得匹配 Adam update 或 test metric。
> - 正式 seeds 冻结为 untouched `200--219`，20 seeds；6D context、4D semantic prototype、2048 train / 2048 test contexts、fixed concentration `8.0`、Adam `lr=1e-3`、negative alpha `0.25`、8000 steps 与 `4000--6000 / 6000--8000` 终态窗口全部预注册。train/test context 独立采样自同一分布，只称 same-distribution held-out-context generalization，不称 OOD。
> - 任务性能崩溃、support boundary 与 NaN/Inf numerical failure继续分开报告。categorical direct-logit score 有界；本实验不声称 Gaussian 式无界梯度、Transformer 外部有效性、跨任务方法排名或任何 taper 必然最优。
> - 新实现为 `src/drpo/du1_e6_cartesian_taper.py`，冻结配置 `configs/du1_e6_cartesian_taper.yaml`，一键入口 `scripts/run_du1_e6_cartesian_taper.py`。正式协议固定 CPU、8 个 seed workers；正式启动必须经过 hardened guard，要求 clean worktree、权威 `origin/main` 匹配、每 seed 持久化 trajectories/summary/audit/calibration 后再写 checkpoint marker、完整终态审计和 durable raw-complete artifact。应用本更新后状态为 **implemented + ready + active + not_run**；smoke/unit/static 结果不构成正式科学结果。
<!-- HANDOFF-DELTA-BLOCK:after_heading:v70-du1-e6-cartesian-taper:END -->
<!-- HANDOFF-DELTA-BLOCK:after_heading:v71-e7-bench-long-budget-parallel-pilot:START -->
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
<!-- HANDOFF-DELTA-BLOCK:after_heading:v71-e7-bench-long-budget-parallel-pilot:END -->
<!-- HANDOFF-DELTA-BLOCK:after_heading:v72-du1-e6-shared-rarity-repair:START -->
> **v72 增量登记：D-U1 E6 shared-rarity 环境修复与正式门禁回收（不删除 v71 及更早内容）**
>
> - **旧设计—问题—替代：**v70 的四格标签在 reward、advantage 和语义 utility 上确实解耦，但 common/rare 只是同一 semantic action 的两个副本，初始概率差主要由 64 维 trainable per-action bias 制造。开发 pilot 进一步表明：common/rare 的共享语义参数梯度几乎相同，Positive-only 自身会扩大副本 bias gap，action-ID support 下降又可能只是在删除无任务差异的冗余副本。因此 v70 pilot 只能保留为工程/问题发现证据，禁止用于 All-negative、Global 或 taper 方法排名。
> - **修复后的 rarity 轴：**保留 32 semantic prototypes × 2 categorical replicas，但删除 trainable per-action bias。每个 replica 在与任务 semantic space 正交的 policy-only rarity coordinate 上取 `+1/-1`；策略使用一个对所有 action/context 共享的 contextual rarity residual head，叠加冻结的初始 half-gap。common/rare 仍具有完全相同 reward、advantage 和 directional utility，但负更新现在通过共享 rarity head 改变整个 common/rare 分区，而不是只改某个动作私有 bias。
> - **Positive-only 中性化：**正样本按 semantic family 训练，目标为一对 common/rare 概率之和的 log-probability。由于所有 prototype 使用相同的正交 rarity factor，该 family likelihood 对 within-pair rarity coordinate 精确不变；rarity residual head 零初始化，Positive-only 不再自己制造或消除 rarity gap。启动前必须通过 positive rarity-gradient 近零和 family-likelihood invariance 审计。
> - **共享梯度门禁：**环境 preflight 除 v70 的 reward/utility/advantage 笛卡尔积不变量外，新增 common/rare shared-rarity gradient audit。reference gap `4.0` 下，rare negative 在共享 rarity head 上的 gradient norm 必须至少为 common 的 `5×`；否则 fail closed，不允许 pilot 或 formal。categorical direct-logit score 仍有界，本实验研究 persistent support suppression，不升级为 Gaussian 无界梯度命题。
> - **支持指标修复：**action-ID entropy/support 与 prototype-family entropy/support 分开记录；support boundary 由 prototype support 或 common/rare 总概率质量触底分别触发。不得再把冗余 replica 被压低直接写成任务语义支持坍缩。
> - **有限状态与公平 control：**所有方法加入同一 shared-rarity quadratic trust-region anchor。该 anchor 对初始 rarity gap 的残差为零，并随偏移平方增长；负 log-probability 压力只随 rarity coordinate 线性增长，因此任意正系数都给出有限的 output-level 最优点。此前 forward-KL 草案在 reference rare mass 已很小时恢复力过弱，已在实现前撤回。anchor 系数仍需 development calibration。`global_matched` 从 step-0 单次匹配改为每个 optimizer step、同一当前模型和 minibatch 上匹配 Exponential 的 Adam 前 raw negative-gradient L2 norm，并保存逐步误差。Adam update 仍只记录，不声称匹配。
> - **方法命名修正：**`reciprocal_linear_distance = 1/(1+lambda sqrt(S))`、`reciprocal_quadratic_distance = 1/(1+lambda S)`、`reciprocal_quartic_distance = 1/(1+lambda S^2)`、`exponential_quadratic_distance = exp(-lambda S)`，其中 `S` 是 normalized excess surprisal。v70 旧命名/解释由本节覆盖，但旧文件与 pilot provenance 不删除。
> - **门禁回收：**`D-U1-E6-CARTESIAN-TAPER-01` 保持 `not_run + implemented`，但从 `ready + active` 回收为 **blocked**。development seeds `0--4` 必须先完成 `negative alpha × rare retention × rarity-logit anchor` 校准并另行冻结正式 horizon、终态阈值和方法矩阵；在独立 formal-freeze 更新前禁止访问 seeds `200--219`。本次只修环境、实现审计和门禁，不产生方法排名。
> - **环境修复工程验收（非科学结果）：**development seeds `0--2`、6 个核心方法、8000 steps 的独立诊断中，Positive-only rarity gradient 与 family-likelihood shift error 均为 `0`，rare/common shared-rarity gradient ratio 最低 `54.60×`，Global 的逐步 raw-gradient budget match 最大误差 `8.88e-16`；18/18 runs 均达到登记窗口 terminal plateau，prototype-support boundary、rarity-mass boundary 与 NaN/Inf 均为 `0`。该诊断只确认旧环境缺陷已被修复，不完成超参校准，也不构成方法排名。
<!-- HANDOFF-DELTA-BLOCK:after_heading:v72-du1-e6-shared-rarity-repair:END -->
<!-- HANDOFF-DELTA-BLOCK:after_heading:v73-e8-taper-corrected:START -->
> **v73 增量登记：Countdown E8-TAPER 距离坐标与执行链修正（不删除 v72 及更早内容）**
>
> - **旧定义—问题—修正：**v67 将 `max(0, sequence_surprisal - tau)` 直接记作距离 `d`。E6-TAPER 的方法命名审计已经确认 surprisal 对应距离平方量级；沿用旧定义会把 reciprocal-linear 实际做成 quadratic-distance，把 squared-distance exponential 实际做成 quartic-distance。E8 现统一定义 `S=max(0,(sequence_surprisal-tau)/c_cal)`、`d=sqrt(S)`，并只在负样本分支使用 detached 权重。
> - **冻结尺度：**保留已登记 `tau=2.0`。`c_cal` 只由独立 calibration replay 和 seed `9134` 在 reference initialization 计算：将校准 surprisal 排序为 lower/upper rarity halves，取两半中位数之差；该尺度在 confirmation 前冻结，若非有限或小于 `1e-6` 则 fail closed。不得用 validation/test 或确认 seeds 重调。
> - **方法公式：**`reciprocal_linear=1/(1+lambda d)`；`exponential=exp(-lambda d)`；`squared_distance_exponential=exp(-lambda d^2)=exp(-lambda S)`。`global_matched` 仍是不区分距离的常数控制；`uncontrolled_negative` 表示在所有方法共享的 frozen negative base scale 下不施加 taper，不等于原始系数恒为 1 的跨协议比较。
> - **确定性与身份修复：**calibration gradient measurement、learner-relative coordinate 和 teacher-forced audit 均关闭 dropout。训练使用 eval/no-grad 的 deterministic coordinate pass 与 train-mode gradient pass 分离。config、reference adapter、replay、sampler seed/plan hash 和 experiment ID 全部执行 fail-closed 身份校验。
> - **梯度预算口径修复：**共享负尺度的实际定义是 positive aggregate gradient L2 除以 uncontrolled-negative aggregate gradient L2，不再误称 per-sample RMS。Global 与 taper 的 initialization matching 继续比较 aggregate raw negative-gradient L2；Adam update 不宣称匹配。
> - **状态边界：**本次只完成实现和门禁修复，未运行 Qwen/CUDA pilot，未产生任何方法排名。`EXT-C-E8-TAPER-0.5B-01` 状态为 `not_run + implemented + ready`；任务性能退化、valid/support/entropy boundary 和 NaN/Inf 仍必须分开报告，fixed 1200-update horizon 不自动称为收敛。
<!-- HANDOFF-DELTA-BLOCK:after_heading:v73-e8-taper-corrected:END -->
<!-- HANDOFF-DELTA-BLOCK:after_heading:v75-countdown-fullbank-gradient-pilot:START -->
> **v75 增量登记：Countdown full-bank 连续 surprisal--gradient 诊断 pilot（不删除 v74 及更早内容）**
>
> - **旧缺口与修正：**v67 已关闭 Countdown 0.5B 机制探索职责，并登记“learner-relative surprisal 较高的错误 completion 往往具有更大的 raw negative influence”的范围受限观察；但当时缺少 full-bank 逐 response 的 trainable-parameter gradient norm 统计。本次补入一个 single-seed full-bank pilot，只回答外部 Transformer 诊断问题：在 verifier outcome 与 negative coefficient 固定时，当前 SFT/reference policy 下的 completion surprisal 是否对应更大的实际可训练参数梯度。
> - **数据身份：**用户提供的 `countdown_gradient_samples_seed100_full.csv` 为 seed `100`、`6000` 个 Countdown puzzles、near/far 各 `6000` 条，共 `12000` 条逐 response 记录。所有样本均为 `verifier_category=arithmetic_wrong`，`valid_format=True`，`uses_numbers=True`，`correct=False`，`negative_coefficient_abs=1.0`。该文件是 full-bank pilot，不是 multi-seed formal result；raw CSV 尚未作为 repository artifact 入库。
> - **数据质量审计：**`mean_token_surprisal`、`direct_logit_score` 与 `trainable_parameter_gradient_norm` 全部有限；gradient norm 非负，范围 `0.650949--438.303528`；surprisal 范围 `0.008992--13.829518`。重新计算 surprisal 与 stored base surprisal 的绝对差异中位数为 `0`，`97.38%` 行完全一致，最大差异 `0.034105`，可作为小规模重算/数值差异而非数据错位信号。
> - **样本级相关：**在全部 `12000` 条 arithmetic-wrong responses 上，surprisal 与 trainable-parameter gradient norm 的 Pearson correlation 为 `0.363`，puzzle-cluster bootstrap 95% CI `[0.343,0.381]`；Spearman correlation 为 `0.445`，95% CI `[0.426,0.463]`。direct-logit score 与 gradient norm 的 Pearson/Spearman 分别为 `0.498/0.568`。
> - **near/far 配对统计：**`far_surprisal > near_surprisal` 为 `6000/6000`，`far_direct_logit_score > near_direct_logit_score` 为 `98.73%`，`far_gradient_norm > near_gradient_norm` 为 `68.55%`。near/far 平均 gradient norm 分别为 `82.294/95.185`，配对均值差 `+12.891`，cluster bootstrap 95% CI `[+11.505,+14.188]`；far/near gradient norm 中位数比为 `1.316`，95% CI `[1.294,1.340]`；Wilcoxon paired p-value 约 `1.27e-128`。
> - **连续分桶趋势：**按 surprisal 做 `10` 个等样本量 bin（每 bin `1200` 条）后，最低 bin 平均 gradient norm 为 `40.274`。中高 surprisal bins 上升到约 `99.91--102.40`，相对最低 bin 为 `2.48--2.54x`，最高 bin 为 `98.56`（`2.45x`）。bin-level Pearson/Spearman 为 `0.829/0.818`。因此图形口径应写成“随 learner-relative surprisal 系统上升并在高 surprisal 区间平台化”，不得写成逐样本严格单调或无界爆炸。
> - **控制项回归：**在 log-gradient 回归中同时控制 `token_count` 与 near/far role 后，surprisal 每增加 `1`，trainable-parameter gradient norm 的乘性因子约为 `1.0275`。这支持“固定 negative coefficient magnitude 不变时，实际参数空间中的负梯度仍随 learner-relative surprisal 增强”的外部诊断结论。
> - **结论边界：**本结果状态为 **pilot / single-seed full-bank diagnostic**。它不能提供 seed-level CI，不能称为正式 Countdown 结果，不能替代 D-U1/D-Diag 的受控因果识别，也不能说明某个 taper 方法必然改善任务性能。若要升级为正式外部证据，必须运行多个独立 SFT/offline seeds，并使用 seed-level bootstrap；E7 若要同图同口径比较，也仍需重新导出逐样本 distance/surprisal--gradient 统计。
<!-- HANDOFF-DELTA-BLOCK:after_heading:v75-countdown-fullbank-gradient-pilot:END -->

1. **唯一 Master 文档是任务轴。** 新理论、新实验、新变量、代码入口和结果状态必须先登记，再执行。
2. **文档先于实验。** 未写明 claim、环境、数据、指标、收敛条件和结果落点的实验，严格禁止启动。
3. **不得破坏性删除。** 旧内容只能移动、压缩并保留索引；结论变化写成“旧结论—问题—新证据—新结论”。
4. **核心变量先审批。** 新符号必须说明不可替代性、与原变量关系及冲突检查；未经确认不得进入主理论。
5. **实验只回答登记问题。** 不得用新术语或新叙事掩盖理论—实验不匹配；不匹配必须先报告和讨论。
6. **动力学必须做终态审计。** 涉及稳态、崩溃、方法排名时，不以任意固定训练步数代替收敛/持续漂移证据。
7. **状态标签固定。** 只允许：已解析证明、已长期验证、有限训练步数验证、pilot、尚未运行、已否定/已替换。
8. **结果必须落盘和回写。** 保存代码、配置、seeds、逐步曲线、汇总、失败运行和文件索引；聊天不能成为唯一载体。
9. **正式环境数量锁定。** 主要受控环境只有一个连续 C-U1 和一个离散 D-U1；历史小环境只作证明、回归和 provenance。
10. **外部实验不能替代理想识别。** Hopper/Countdown 回答外部有效性；C-U1/D-U1 回答可控因果与 ground truth。

## 0.1 当前执行门禁

- C-U1 E1/E2/E3：现有正式状态保留。`C-U1-E4-ADAM-RERUN` 保留“有限训练步数验证”；`C-U1-E4-CONV-01` 经用户明确审阅，在保留原 18/20 门禁失败事实的前提下，按 15/20、16/20、15/20 目标状态、0/60 明确相反终态与 60/60 长程科学角色不反转，闭合为“已长期验证”。`C-U1-E4-TAPER-01` 已完成 `220/220` 正式 runs、终态审计与交付；20/20 paired seeds 支持 Quadratic 在 anchor-normalized protocol 下比 Linear 更强抑制远场负梯度，但 200 controlled/positive runs 未形成稳定候选，故科学状态为 **有限训练步数验证**，不得称 long-run validated 或形成 universal method ranking。
- D-U1：E5 已长期闭环。E6 pilot/focused development 保持 development evidence；`D-U1-E6-SEMANTIC-LONGRUN-01` 已完成 `360/360` formal runs、2x 终态审计与 durable delivery，科学状态为 **long-run validated**。`D-U1-E6-TAPER-01` 的 predecessor delivery 已满足，但其距离坐标、paired protocol、新 untouched seeds 和独立 runner 尚未冻结/实现，仍是 review-required + blocked。
- Hopper/D4RL：`EXT-H-E7-Q2` 是 E7-MECH，runner/config 已实现但 formal launch 仍等待受控 taper 阶段交付；`EXT-H-E7-BENCH-01` 是 D4RL MuJoCo locomotion 9-task 方法效果表，等待 E7-MECH 与受控方法 shortlist 冻结。
- Countdown：`EXT-C-E8-V4.2` 是当前 E8-MECH/pilot；`EXT-C-E8-V4.1` 仅保留 provenance；`EXT-C-E8-SCALE-01` 是更大固定数据与模型规模验证，等待 E8-MECH 和 E7-BENCH。

<!-- HANDOFF-DELTA-BLOCK:section_end:e8-base-rl-replay-0p5b-gate:START -->
- **Countdown E8 base-start RL/replay 0.5B pilot：**登记 `EXT-C-E8-BASE-RL-REPLAY-0.5B-01`，作为移除 Countdown SFT warmstart 后的基模起点诊断。该实验只回答：Qwen pretrained base 是否能通过 oracle-offline fixed positive corpus 学起；base-specific calibrated offline negatives 是否能超过 positive-only；online on-policy self-sampled positives 是否能冷启动；dynamic replay buffer 累积历史自采 positives/negatives 是否优于 immediate on-policy 更新。所有 RL 分支从 Qwen pretrained base + fresh LoRA 开始，禁止 Countdown SFT warmstart、随机初始化主实验、taper 方法族和正式方法排名声明。固定预算 pilot 只报告有限步 evidence；结果必须分开报告 task performance、online signal sparsity/replay support、valid structure boundary 和 NaN/Inf numerical failure。
<!-- HANDOFF-DELTA-BLOCK:section_end:e8-base-rl-replay-0p5b-gate:END -->
<!-- HANDOFF-DELTA-BLOCK:section_end:e8-lowsft-rft-dirty-pilot-record-20260708:START -->
- **Countdown E8 low-SFT / capacity diagnostic dirty pilots：**本线记录 `EXT-C-E8-ONPOLICY-CAPACITY-DIAG-0.5B-01` 与一次性 `EXT-C-E8-LOWSFT-RFT-0.5B-01` 试错结果，只能作为 single-seed pilot evidence，不得升级为正式多 seed 结论或方法排名。capacity diagnostic 的 single seed `2026070701` 显示 `same_lora_rft`、`fresh_lora_rft`、`full_param_rft` 的 `best_attempt` 均为 0；terminal 端总体表现为 greedy 持平或小升、pass@8/pass@64 下降，说明 naive verifier-correct positive-only on-policy RFT 没有超过 LoRA SFT 起点。low-SFT 试错从按 validation greedy≈0.08 选出的 epoch-3 LoRA SFT checkpoint 起跑；该 checkpoint 的 pass@8 实际已接近 full-SFT（不是 pass@8≈0.08 起点），RFT 后 `best_attempt=0`，terminal test greedy 0.100→0.113、pass@8 0.174→0.133、pass@64 0.265→0.149。解释必须保留以下限制：运行源码为 dirty pilot / one-off orchestration；不是 convergence；没有证明 3B 或更强模型无效；尚需 no-update、parameter-delta、probe-loss、Qwen pretrained-base no-SFT、ultra-low pass@8 checkpoint 与 offline fixed-corpus controls。工程上允许把 `cmd_sft --save_every_epoch` 作为 opt-in 本地 checkpoint 功能合入，以便后续选择更细粒度 ultra-low SFT 起点；模型权重与结果包不得进入 Git 更新包。
<!-- HANDOFF-DELTA-BLOCK:section_end:e8-lowsft-rft-dirty-pilot-record-20260708:END -->
<!-- HANDOFF-DELTA-BLOCK:section_end:e8-onpolicy-capacity-diag-0p5b-gate:START -->
- **Countdown E8 on-policy capacity diagnostic 0.5B pilot：**登记 `EXT-C-E8-ONPOLICY-CAPACITY-DIAG-0.5B-01`，作为 `EXT-C-E8-ONPOLICY-UNPOLISHED-0.5B-01` 之后的第二层诊断，只回答 same-LoRA continuation 退化是否来自 same-adapter drift、LoRA RFT 容量、LoRA SFT 容量或 on-policy 探索/样本多样性不足。第一轮只跑单 seed 的 `same_lora_rft / fresh_lora_rft / full_param_rft / full_param_sft_only` 分支并行诊断；单 seed 内部 on-policy attempts 仍保持串行。所有 RFT 分支仍为 verifier-correct positive-only，不包含 signed negative、taper 方法族或 frozen off-policy replay。固定 sampling attempts 只能报告 finite-budget pilot evidence，不得宣称收敛或方法排名；full-param 分支只作 capacity diagnostic，不替代 E8-TAPER 方法实验。
<!-- HANDOFF-DELTA-BLOCK:section_end:e8-onpolicy-capacity-diag-0p5b-gate:END -->
<!-- HANDOFF-DELTA-BLOCK:section_end:e8-onpolicy-unpolished-0p5b-gate:START -->
- **Countdown E8 on-policy unpolished 0.5B pilot：**登记 `EXT-C-E8-ONPOLICY-UNPOLISHED-0.5B-01`，仅作为 0.5B + same-LoRA continuation 是否能从当前 policy 自采样 verifier-correct completion 继续学习的排除项诊断。第一版只允许 `sft_only` 与 `onpolicy_rft_positive_only`，不包含 full-param、fresh-LoRA、signed negative、taper 方法族或 frozen off-policy replay；数据 split 使用当前 Countdown structural family-holdout 协议。SFT 可通过显式路径复用已训练 LoRA adapter，但必须记录 provenance，且不改变 same-LoRA continuation 口径；固定 sampling attempts 只能报告 finite-budget pilot evidence，不得宣称收敛或方法排名。
<!-- HANDOFF-DELTA-BLOCK:section_end:e8-onpolicy-unpolished-0p5b-gate:END -->
<!-- HANDOFF-DELTA-BLOCK:section_end:v52-countdown-current-gate-override:START -->
- **Countdown v52 覆盖：** `EXT-C-E8-V4.3` 取代 V4.2 成为当前 E8-MECH/focused pilot；V4.2 只保留 matched-pair mechanism provenance。`EXT-C-E8-SCALE-01` 继续等待 V4.3 与 E7-BENCH，不因本次实现自动解锁。
<!-- HANDOFF-DELTA-BLOCK:section_end:v52-countdown-current-gate-override:END -->
<!-- HANDOFF-DELTA-BLOCK:section_end:v55-du1-e6-semantic-gap-current-gate:START -->
- **D-U1 v55 覆盖：** `D-U1-E6-SEMANTIC-GAP-LONGRUN-01` 已完成 `100/100` 正式 runs、2× horizon 与终态审计，科学状态为 **有限训练步数验证**；45/100 plateau、55/100 persistent-drift-or-inconclusive，禁止稳态方法排名或无新登记重跑。`D-U1-E6-TAPER-01` 的 successor-delivery 条件已满足，但其四项协议/实现门禁仍未完成，继续 review-required + blocked。
<!-- HANDOFF-DELTA-BLOCK:section_end:v55-du1-e6-semantic-gap-current-gate:END -->
<!-- HANDOFF-DELTA-BLOCK:section_end:v56-e6-parent-closure-current-gate:START -->
- **v56 E6 父 claim 关闭覆盖：** E6 的论文核心 claim 现已范围受限关闭；主 long-run 与两个 gap 子实验的原科学状态分别保持 `long_run_validated / finite_step_validated / finite_step_validated`。`D-U1-E6-TAPER-01` 保留为可选非门禁未来工作。当前下一正式 route item 为 `EXT-H-E7-Q2`，registry 状态为 **implemented + ready + active + not_run**；启动后仍须走 canonical hardened guard，且在 raw-complete、终态审计、打包和交付前不得声称 E7 完成。
<!-- HANDOFF-DELTA-BLOCK:section_end:v56-e6-parent-closure-current-gate:END -->
<!-- HANDOFF-DELTA-BLOCK:section_end:v57-countdown-offline-bank-current-gate:START -->
- **Countdown v57 覆盖：** `EXT-C-E8-V4.4-OFFLINE-BANK` 是用户批准的当前离线 focused pilot；V4.3 保留为 fixed-pair predecessor。V4.4 只改变固定负样本覆盖与 current-policy near/far reselection，不引入在线数据刷新。`EXT-H-E7-Q2` 仍是下一正式 route item，`EXT-C-E8-SCALE-01` 继续 blocked。
<!-- HANDOFF-DELTA-BLOCK:section_end:v57-countdown-offline-bank-current-gate:END -->
<!-- HANDOFF-DELTA-BLOCK:section_end:v59-countdown-offline-bank-tuning-current-gate:START -->
- **Countdown v59 覆盖：** `EXT-C-E8-V4.5-OFFLINE-BANK-TUNING` 是当前用户批准的离线 focused successor；V4.4 作为 frozen-bank predecessor 保留。V4.5 只调 calibrated global negative multiplier 与 exponential taper lambda，禁止在线刷新、方向筛选或模型规模同时变化。`EXT-H-E7-Q2` 仍是下一 formal route item，`EXT-C-E8-SCALE-01` 继续 blocked。
<!-- HANDOFF-DELTA-BLOCK:section_end:v59-countdown-offline-bank-tuning-current-gate:END -->
<!-- HANDOFF-DELTA-BLOCK:section_end:v60-e4-taper-current-gate:START -->
- **E4-TAPER v60 覆盖：** `C-U1-E4-TAPER-01` 仍为 finite-step validated。四个后续 ID 已获用户批准并登记，但全部保持 blocked：先冻结并实现 `NEAR-RETENTION-01`，交付后才允许冻结 `BUDGET-MATCH-01`；二者交付并冻结 shortlist 后才允许 `CONV-01`；最后才用 untouched seeds 执行 `CONFIRM-01`。原实验禁止自动延长，几何 robustness 不作为当前门禁。
<!-- HANDOFF-DELTA-BLOCK:section_end:v60-e4-taper-current-gate:END -->
<!-- HANDOFF-DELTA-BLOCK:section_end:v61-e4-taper-near-retention-current-gate:START -->
- **E4-TAPER v61 覆盖：** `C-U1-E4-TAPER-NEAR-RETENTION-01` 已完成协议冻结、独立 runner、formal-channel 登记和工程 smoke，registry 为 **implemented + ready + active + not_run**。允许下一步启动该实验的 canonical guarded formal run，但 smoke/单元测试不构成科学结果。`BUDGET-MATCH-01` 仍必须等待 Near-Retention 的 raw-complete、终态审计、打包与交付；不得提前实现为可运行状态或并行启动。
<!-- HANDOFF-DELTA-BLOCK:section_end:v61-e4-taper-near-retention-current-gate:END -->
<!-- HANDOFF-DELTA-BLOCK:section_end:v62-countdown-online-offpolicy-current-gate:START -->
- **Countdown v62 覆盖：** `EXT-C-E8-V4.6-ONLINE-OFFPOLICY-REPLAY` 是当前用户批准并已实现的 Countdown focused successor，状态为 **implemented + not_run**。执行前必须提供完整 V4.5 `RUN_COMPLETE.json`/`terminal_audit.json` 及其指向的 V4.4 frozen inputs；runner fail-closed 校验输入与 reference adapter。它可作为独立 pilot 启动，但不改变 `EXT-H-E7-Q2` 的 formal 优先级，也不自动解锁 `EXT-C-E8-SCALE-01`。
<!-- HANDOFF-DELTA-BLOCK:section_end:v62-countdown-online-offpolicy-current-gate:END -->
<!-- HANDOFF-DELTA-BLOCK:section_end:v63-e4-taper-closure-current-gate:START -->
- **E4-TAPER v63 覆盖：** `C-U1-E4-TAPER-NEAR-RETENTION-01` 已完成 `280/280` method-seed runs 与终态审计，科学状态沉淀为 **有限训练步数验证**。主保持率 `0.75` 下，Reciprocal-Quadratic、current Exponential、Squared-distance Exponential 相对 Reciprocal-Linear 的 held-out-context reward 配对均值差分别为 `+0.012002 / +0.015619 / +0.036134`，三者均为 `20/20` seeds 正差；Squared-distance Exponential 的 harmful-far retention 为 `0.010382`，低于 Reciprocal-Linear 的 `0.055886`。该结果只支持当前冻结矩阵中的有限步函数形状信号；`260/280` runs 在 8000 steps 时未获严格终态解析，禁止稳态、普遍方法排名或 OOD 表述。
- 三类事件继续严格分报：task-performance collapse `13/280`、support/variance boundary `20/280`、NaN/Inf `0/280`；前两类全部来自 unweighted control。v63 仓库只保存 compact result deposition；本次构建会话没有原始 280-run raw-complete artifact 及其哈希，禁止伪造，归档发布前必须从原交付包恢复。
- `C-U1-E4-TAPER-BUDGET-MATCH-01` 在 v63 冻结并实现为下一项 **implemented + ready + active + not_run**。唯一 primary budget coordinate 是每一步、Adam 之前的 raw negative-gradient L2 norm；paired Reciprocal-Linear actor 生成冻结目标 schedule，其他 Distance families 与 non-selective Global stepwise scale 使用 detached scalar 精确匹配该 norm。Adam 实际 parameter-update norm 只记录、不声称匹配。正式 seeds 固定为 `110--129`；seeds `130--149` 继续 untouched，专属最终 confirmation。
- `C-U1-E4-TAPER-CONV-01` 与 `C-U1-E4-TAPER-CONFIRM-01` 的 seed firewall、输入输出契约、shortlist 冻结规则、32000-step 长程上限、continuous Adam-state 要求、2× terminal audit 与确认分析计划已预登记，但二者继续 blocked。Budget-Match terminal-audited、packaged、delivered 之前不得生成 shortlist 或启动 Convergence；Convergence 交付且 confirmation config 哈希冻结前不得访问 seeds `130--149`。
<!-- HANDOFF-DELTA-BLOCK:section_end:v63-e4-taper-closure-current-gate:END -->
<!-- HANDOFF-DELTA-BLOCK:section_end:v66-e4-taper-budget-match-current-gate:START -->
- **E4-TAPER v66 覆盖：** `C-U1-E4-TAPER-BUDGET-MATCH-01` 已完成 `140/140` 正式 runs、逐步 raw-negative-gradient budget audit 与 terminal audit，科学状态为 **有限训练步数验证**。相同 Adam 前 raw negative-gradient L2 budget 下，三种 selective candidates 相对 Reciprocal-Linear 均在 `20/20` paired seeds 上提高 held-out-context reward 并降低 harmful-far retention；Global stepwise scale 则在 `0/20` seeds 上提高 reward，且保留更多 harmful-far influence。Terminal useful-near retention 因零分母不可评估，不得补写为已证明。
- 原 guard 只在计算结束后的 required-output/package 阶段失败：return code `0`、provenance 未受损、正式结果和原 failed tree 均保留。v66 修复 runner 漏写 `scientific_run_manifest.json`，并通过 compact deposition + explicit full-raw sidecar 完成交付；不得把 packaging failure 称为实验数值失败。
- `CONV-01` 仍 blocked；下一项是 Budget-Match 交付后的独立 shortlist-freeze 更新和 continuation runner 实现。不得直接延长 run_003，也不得访问 confirmation seeds `130--149`。
<!-- HANDOFF-DELTA-BLOCK:section_end:v66-e4-taper-budget-match-current-gate:END -->
<!-- HANDOFF-DELTA-BLOCK:section_end:v70-du1-e6-cartesian-taper-current-gate:START -->
- **D-U1 v70 覆盖：** 原 `D-U1-E6-TAPER-01` development preregistration 原样保留、不得启动；其执行职责由用户批准的 `D-U1-E6-CARTESIAN-TAPER-01` 取代。新 successor 已冻结 utility × surprisal 2×2 Cartesian protocol、独立 runner、正式 seeds 与终态审计，registry 状态为 **implemented + ready + active + not_run**。它在一个 formal artifact 中先报告 Cartesian 机制块、再报告预注册 taper 方法块；smoke/unit/static 结果不构成科学结果。
<!-- HANDOFF-DELTA-BLOCK:section_end:v70-du1-e6-cartesian-taper-current-gate:END -->
<!-- HANDOFF-DELTA-BLOCK:section_end:v72-du1-e6-shared-rarity-repair-current-gate:START -->
- **D-U1 v72 覆盖：** `D-U1-E6-CARTESIAN-TAPER-01` protocol revision 2 已实现 shared contextual rarity coordinate、Positive-only rarity-neutral family objective、prototype/action support 分报、quadratic rarity-logit anchor 与 stepwise raw-gradient matched Global。原 v70 formal activation 撤回；当前状态为 **implemented + blocked + not_run**，blocked by development calibration and separate formal protocol freeze。正式 seeds `200--219` 继续 untouched。
<!-- HANDOFF-DELTA-BLOCK:section_end:v72-du1-e6-shared-rarity-repair-current-gate:END -->
<!-- HANDOFF-DELTA-BLOCK:section_end:v73-e8-taper-current-gate:START -->
- **Countdown E8-TAPER v73 覆盖：**`EXT-C-E8-TAPER-0.5B-01` 已实现 corrected `S -> d=sqrt(S)` 坐标、独立冻结尺度、deterministic detached weighting、paired sampler 身份校验和终态审计，当前为 **implemented + ready + not_run**。只允许先运行登记的 0.5B pilot；不得将 smoke/static test 写成科学结果，也不得预设 Exp、Global 或任何 taper 获胜。
<!-- HANDOFF-DELTA-BLOCK:section_end:v73-e8-taper-current-gate:END -->
<!-- HANDOFF-DELTA-BLOCK:section_end:v74-du1-e6-rev4-current-gate:START -->
> **D-U1 E6 当前门禁（v74）：** `D-U1-E6-CARTESIAN-TAPER-01` 的 revision-3 development calibration 已完成，revision-4 formal protocol 已获用户批准并冻结，registry execution gate 为 `ready`、formal activation 为 `active`、科学状态仍为 `not_run`。下一步只能在 exact frozen commit 上通过 hardened guard 运行 seeds `200--219`；不得重新访问 development seeds 进行选参，也不得在看到 formal 结果后修改 `alpha=0.5 / anchor=0.25 / rho=0.25`、方法集合或终态标准。
<!-- HANDOFF-DELTA-BLOCK:section_end:v74-du1-e6-rev4-current-gate:END -->
<!-- HANDOFF-DELTA-BLOCK:section_end:v75-e8-taper-diagnostic-bugfix:START -->
- **Countdown E8-TAPER 0.5B diagnostic bugfix:** `EXT-C-E8-TAPER-0.5B-01`
  keeps the same experiment ID, methods, paired seeds, taper formulas and synthetic-negative policy.
  The frozen natural replay target is reduced from 1500 to 900 train prompts because the 0.5B
  frozen reference produced 913 eligible natural-negative prompts with `synthetic_negative_fallback=false`.
  Teacher-forced diagnostics are streamed with batch size 1, and same-graph raw/weighted gradient
  diagnostics retain the graph only inside the diagnostic audit. This update is an implementation/config
  repair, not a scientific result; real Qwen/CUDA pilot remains not run.
<!-- HANDOFF-DELTA-BLOCK:section_end:v75-e8-taper-diagnostic-bugfix:END -->
<!-- HANDOFF-DELTA-BLOCK:section_end:v76-current-gate-rules:START -->
> **v76 常驻协作规则覆盖：**DRPO figure/plot/chart/panel/画图 默认走代码绘图，不自动 image generation；在线轮询必须在当前 assistant 轮次内阻塞式检查到终态或明确无法继续，不得把后台进程或间歇查询冒充轮询；未来 DRPO 更新包默认只交付 canonical bundle-backed package，patch-only runnable 包仅在用户明确要求 immediate exact-base 临时包时允许。
<!-- HANDOFF-DELTA-BLOCK:section_end:v76-current-gate-rules:END -->
<!-- HANDOFF-DELTA-BLOCK:section_end:v77-current-gate-minimal-diff:START -->
> **v77 最小改动治理门禁：**bug、失败包、窄修复和小型代码更新默认进入 Minimal Sufficient Diff mode。执行前必须锁定用户授权的开发对象，执行 Green/Yellow/Red/Split 分类，并在首次失败后先做 first-failure classification；不得把最近失败症状、工具体验优化或自造 workflow 替代用户要求的开发目标。该规则引用 `docs/code_minimality_governance.md`，不改变科研实验状态、seeds、thresholds、registry 或结果。
<!-- HANDOFF-DELTA-BLOCK:section_end:v77-current-gate-minimal-diff:END -->
<!-- HANDOFF-DELTA-BLOCK:section_end:v79-e8-active-tail-current-gate:START -->
- **Countdown E8-TAPER v79 覆盖：**`EXT-C-E8-TAPER-0.5B-01` 仍为 implemented + ready + not_run pilot，但当前有效协议使用 independent-calibration common-half median tau、nondegenerate calibration fail-closed guard 与 streamed surprisal-bin diagnostics。应用后必须先跑短预算 sanity 验证各方法未退化为 uncontrolled clone；smoke/sanity/pilot 不得写成正式结果或方法排名。
<!-- HANDOFF-DELTA-BLOCK:section_end:v79-e8-active-tail-current-gate:END -->
<!-- HANDOFF-DELTA-BLOCK:section_end:v80-dev-review-workflow-current-gate:START -->
> **v80 常驻协作规则覆盖：**DRPO 实验代码默认采用 dev-branch implementation + independent reviewer-gate 流程。GLM/Claude Code 等执行代理只负责按已批准 scope 写代码、运行 liveness gate/实验、产出 dev 分支 artifacts；不得重新设计实验、改 claim、改锁定科学变量、解释最终方法排名或合并到 `main`。Reviewer/gatekeeper 负责 scope contract、diff、测试、liveness、result provenance、失败清单、终态审计与 merge/reject 决策。大型 sweep 必须先通过小规模 liveness gate；实验结果必须绑定产生结果的 dev branch `HEAD`；raw-complete、terminal-audited、packaged、delivered 与 applied_to_repository 继续分开。
<!-- HANDOFF-DELTA-BLOCK:section_end:v80-dev-review-workflow-current-gate:END -->

## 0.2 C-U1 泛化术语覆盖规则（v15 锁定）

1. C-U1 的训练状态与测试状态使用同一生成分布 `s ~ N(0,I_6)`，只在样本身份上独立；因此 E1-E4 报告的是 **同分布 held-out-context generalization（未见状态泛化）**。
2. 允许使用的表述：`held-out-context reward`、`unseen-context generalization`、`同分布测试状态`、`未见状态上的函数泛化`。
3. 禁止用于当前 C-U1 的表述：`OOD reward`、`OOD generalization`、`distribution-shift generalization`。
4. `a_star(s)` 是训练中未作为正样本展示的隐藏最优动作。策略接近它可称为“越过正样本支持并接近隐藏最优动作”，不能仅凭此称为 OOD。
5. “策略漂移到低 reward 区域”与“数据分布 OOD”严格区分。前者是策略相对任务最优的几何漂移，不意味着测试状态来自分布外。
6. Part II 历史记录中的 OOD 旧措辞不删除，但全部由本节覆盖。新论文若需要 OOD 结论，必须另外登记并运行显式状态分布偏移实验。

## 0.3 正式实验守护与可持久交付门禁（v19 锁定）

1. **计算结束不等于实验完成。** 正式实验必须依次经历 `registered`、`running`、`raw_complete`、`terminal_audited`、`packaged`、`delivered`；仓库闭环还需 `applied_to_repository`。科学状态标签与该执行证据状态分开维护。
2. **临时环境必须持续守护。** 正式运行不得以无人监控的后台 PID 代替当前工作。必须使用统一守护脚本或等价前台 supervisor，持续记录 heartbeat、PID、进度、日志、输出活动与退出状态。
3. **每个实验块立即持久化。** 当前 experiment ID 计算结束后，先完成审计、handoff/registry 回写、打包和交付，再启动下一个 experiment ID。C-U1 中 E3 包未交付前禁止启动 E4。
4. **阶段 checkpoint。** 预计运行超过 30 分钟时，默认每完成 5 个正式 seeds 生成恢复包。恢复包可以不是正式科学结果，但必须包含已完成 seeds、待运行 seeds、日志、源代码 SHA 和部分原始输出。
5. **失败也必须交付。** 非零退出、运行时回收、收尾绘图/聚合错误或终态审计失败时，先生成 `experiment-failed` 包，再修复或重跑；不得仅在聊天中描述错误。
6. **最终包门禁。** 最终实验包必须兼容 `drpo-update`，并包含 `update.patch`、`BASE_COMMIT.txt`、`CHANGE_SUMMARY.md`、`TEST_COMMANDS.sh`、`modified_files/`、结果原始材料、`RUN_COMPLETE.json`、终态审计、日志、manifest 和 SHA256 校验。
7. **临时路径不构成持久证据。** `/mnt/data` 或其他 ephemeral filesystem 中的文件只有在形成可下载 artifact、进入持久服务器/对象存储或提交到仓库后，才算持久化。
8. **完成表述受限。** `raw_complete` 只能表述为“计算完成、审计或交付尚未完成”；只有可下载包验证并交付后，才能说“正式运行完成”。
9. **包大小策略。** 默认最终实验包警戒线为 25 MiB；允许压缩轨迹和去除冗余 optimizer state，但不得删除逐 seed 摘要、核心轨迹、终态审计、失败索引和来源校验。
10. **详细规范唯一引用。** 具体 package kinds、命令和校验规则见 `docs/formal_experiment_artifact_protocol.md`；若其与本节冲突，以本节和 `AGENTS.md` 为准。

## 0.4 Registry 执行状态一致性（v42 锁定）

1. `execution_gate.state` 表示科学/依赖门禁，`formal_execution.activation_state` 表示 operational 启动状态；两者不得相互矛盾。
2. `execution_gate.state=ready` 且 `entrypoint_status=implemented` 时，`activation_state` 必须为 `active`。
3. `activation_state=active` 时，不得存在 `execution_gate.state=blocked`。
4. 任何 `blocked` 状态都必须登记非空依赖或 blocking reason；禁止无依据的陈旧 blocked 标记。
5. `entrypoint_status=planned`、`implementation_state=not_implemented` 的正式实验允许保持 blocked，但不得因此绕过 claim、职责和后续 protocol-freeze 登记。
6. `scripts/validate_formal_execution_channel.py` 对 canonical experiments 与 development registrations 中的 formal 条目执行 fail-closed 校验；registry 更新和 `drpo-update` 测试必须运行它。

# 1. 论文最终目标与两条主工作线

## 1.1 论文目标

以原 DRPO（arXiv:2602.10430）为起点，重写为面向一般 off-policy policy optimization 的论文，而非推荐专属论文。原推荐实验作为应用验证保留，不再承担理论合理性的唯一证据。

## 1.2 理论修改主线

1. 保留原始 repulsive dynamics 主干：正优势吸引、负优势排斥、固定离线样本随策略移动进入远场、score function 放大、正负梯度失衡与 collapse。
2. 修正 Gaussian 方差方向：远场负样本导致均值排斥并收缩方差，而不是均值与方差同时扩张。
3. 修正数学工具：固定 off-policy 样本不能用 expected Fisher 的 SPD 性质证明联合扩张；改用精确更新式与总体 signed gradient field 的 Jacobian。
4. 将 exponential-family 统一作为核心 contribution：在不抛弃原变量体系的前提下，使用指数族必要符号给出 Gaussian 与 categorical 的共同平衡/边界条件。
5. 解释负梯度的双重作用：受控负梯度可突破 positive-only 的模仿上限，远场异常负梯度则导致失稳。

## 1.3 新实验主线

1. 一个真正统一的连续 contextual-bandit 环境，完成四个连续实验块；
2. 一个独立 categorical 环境，完成两个离散实验块；
3. Hopper、Countdown、推荐作为外部验证层；
4. 所有涉及动力学终态、稳态或方法排名的实验必须达到预定义收敛标准。

---

# 2. 环境登记表（锁定）

| 环境 ID | 环境 | 状态/动作 | 角色 | 是否主文 |
|---|---|---|---|---|
| C-U1 | 统一连续 contextual bandit | 6D state, 2D continuous action | E1-E4 全部连续机制实验 | 是 |
| D-U1 | 统一 categorical contextual bandit | 6D state, finite unordered semantic actions | E5-E6 离散机制实验 | 是 |
| D-Diag | direct-softmax 解析诊断 | 单状态 logits | 验证概率衰减与 score 上界 | 附录 |
| EXT-H | Hopper/D4RL | 真实离线控制数据 | learned-critic 外部机制与方法效果 | 外部验证 |
| EXT-C | Countdown/Qwen | 序列生成 | token-level 离散外部验证 | 外部验证 |
| HIST-* | 旧 C1/C2/Product/Collapse/Extrapolation | 多个开发小环境 | 历史推导、单元测试、回归基线 | 不删除，附录保留 |

**最终允许的主要受控环境只有两个：C-U1 与 D-U1。** 旧环境不再承担最终论文主表，但必须保留其设计、结果与被替代关系。

---

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

## 3.7 D-U1 / E6 开发配置登记（E4 已完成；用户已批准与 E4-TAPER 并行）

本节把既定的“随机 action ID + semantic embedding”计划落成开发配置，不改变 E5 direct-softmax 已锁定结论。正式数值网格须经过开发 seeds 后另行冻结。

1. **状态与 catalogue：** 6D state；开发阶段 2048 train / 2048 test states；64 个动作、4D 单位 semantic embeddings。动作 ID 对 semantic embedding 做随机置换，ID 顺序不携带几何。
2. **状态语义几何：** 每个状态产生 `t_plus(s)` 和正交 improvement direction；`t_star=normalize(t_plus+0.45d)`，`t_minus=normalize(t_plus-0.45d)`。reward 只依赖动作 embedding 与 `t_star` 的相似度。
3. **数据：** 隐藏最优动作是最接近 `t_star` 的 catalogue item，禁止出现在 positive demonstrations；4 个正动作取最接近 `t_plus` 的动作；有益 local negative 取最接近 `t_minus` 的动作；4 个 far-pressure negatives 取低 utility / 近 `-t_plus` 的动作。负 advantage 标签严格相等并冻结。
4. **策略：** 共享 MLP 输出 4D direction；logit 为 concentration 乘以 state direction 与 action semantic embedding 的内积。固定 concentration 分支回答语义方向外推；可学习 state-conditioned concentration 分支回答 support/temperature collapse。
5. **主指标：** hidden-optimal probability、expected semantic reward、语义方向归一化外推位移、entropy、effective support、concentration、task collapse 和 support collapse；两类 collapse 分别报告。
6. **E6-A：** positive-only 与 local-negative alpha scan，验证未见最优动作概率是否提高、是否越过 positive support、是否存在过度外推。
7. **E6-B：** 加入 far pressure，比较 uncontrolled、Near-zero、Far-zero/local-oracle、Far-cap、budget-matched global。
8. **E6-C：** 独立打乱 policy-side embeddings 与 reward semantics 的对应关系；预期 support suppression 保留，但系统性的 hidden-optimum 外推消失。
9. **顺序门禁更新（v47）：** E6 pilot 与 focused development 分别完成 105/105 和 165/165 development runs；正式 long-run 已在 untouched seeds `10--29` 上完成 360/360 runs、2x 终态审计与交付。不得复用这些 held-out seeds 调参或无新登记重跑；下一步只能先冻结并实现独立的 `D-U1-E6-TAPER-01`。

---

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

## 3.8 C-U1 共享实现与二次阶方法实验 `C-U1-E4-TAPER-01`

### 3.8.1 代码单一来源

C-U1 的环境与 actor 不再允许嵌入新实验文件。唯一共享实现为 `src/drpo/cu1_core.py`，包含 state-to-geometry 映射、正/负轮廓、`Split/Environment`、Gaussian actor、log-probability、标准化距离和输出 score 分解。`drpo_cu1_e1_e4_oneclick.py` 只保留冻结 protocol、训练、干预、审计与报告；`cu1_e1_componentwise_rerun.py` 和 taper runner 只导入共享实现。重构必须用确定性张量、actor 初始化、log-probability、环境不变量和 smoke run 做等价回归，不能以“代码更整洁”为由改变任何冻结科学变量。

### 3.8.2 唯一距离与方法公式

对当前 isotropic Gaussian actor，定义唯一方法距离

$$
d_\theta(s,a)=\frac{\lVert a-\mu_\theta(s)\rVert_2}{\sigma_\theta(s)},\qquad u=\frac{d_\theta(s,a)}{d_{\mathrm{ref}}},\qquad d_{\mathrm{ref}}=5.
$$

距离和权重均 stop-gradient；只重权负优势项，正优势项不变。正式方法为

$$
w_{\mathrm{lin}}(u)=\frac{1}{1+\lambda u},\qquad
w_{\mathrm{quad}}(u)=\frac{1}{1+\lambda u^2},\qquad
w_{\exp}(u)=e^{-\lambda_{\exp}u}.
$$

共同参考衰减为 `w(u=1)=rho`，故 reciprocal 两族使用 `lambda=rho^{-1}-1`，指数族使用 `lambda_exp=-log rho`。这不是 gradient-budget matching；所有方法读取同一数据、固定 advantage、actor、初始化和 minibatch index stream，只改变以上函数。

### 3.8.3 正式协议

- **Experiment ID：** `C-U1-E4-TAPER-01`；补充 E4 的方法阶数 claim，不替代 E1--E4。
- **状态：** 正式 seeds 70--89 已完成 220/220 runs；终态审计未全部通过，科学状态为 **有限训练步数验证**。seeds 0--4 的旧结果继续只作开发 pilot。
- **正式 seeds：** 70--89；20 seeds 配对。
- **主比较：** reciprocal-linear 对 reciprocal-quadratic，`rho=0.25`、`alpha=1.0`。
- **次要对照：** `rho in {0.50,0.75}` 的形状敏感性；exponential 只检验更快尾部，不预设其 reward 更优；positive-only 与 unweighted-negative 为边界对照。
- **优化：** 从同 seed 的 **2000-step positive-only Adam checkpoint** 初始化，与 v29 的 E3/E4 起点完全相同；E2 后续 LBFGS、continuation 与 adaptive polish 不得进入 taper 方法初始化。Adam `lr=5e-4`，state minibatch 256。
- **终态：** 每 100 steps 评估；至少 1000 steps 后，连续 10 个窗口中 reward、归一化外推位移和 sigma 的归一化斜率均低于 `1e-4`，且 joint 方法的归一化净场残差低于 `2e-3`，才形成稳定候选；`positive-only` 因不存在负场抵消，改用全数据 absolute positive-gradient norm `<1e-3`。只有完整运行到候选步数的 2 倍且终态分类不反转，才能记为 `stable_plateau_2x_confirmed`；若候选在 4000 steps 之后才出现而 8000-step 上限容不下完整 2× continuation，必须记为未解析终态。到达 support/variance boundary 或 NaN/Inf 作为独立终态事件；固定 horizon 到期本身不构成收敛。
- **主机制指标：** 初始与终态实际全参数负梯度的 far/near ratio、far-field log-log slope、标准化距离与权重、output-space mean/log-scale 分量。
- **任务指标：** 同分布 held-out-context reward、到 `a_plus/a_star` 的距离、归一化外推位移。
- **失效拆分：** task-performance collapse、support/variance-boundary event、NaN/Inf numerical event 分开记录。
- **主统计：** 20-seed paired bootstrap，报告 quadratic-minus-linear 的 far/near ratio 与 reward 差异；理论预注册只预言前者更低，不预言后者必然更高。
- **Linear 名称边界：** `w_lin` 是本研究在同一标准化距离上的内部 `p=1` reciprocal control，不是原 DRPO 分布鲁棒章节中的 linear weighting，也不以复现任何外部方法为前置条件。clipped-linear、surprisal-linear 或不同距离上的线性族属于其他方法，必须另行登记，不能更名替换本实验。


### 3.8.4 环境连续性、质量匹配与方向效用边界（v44 澄清）

1. **连续环境与有限离线支持必须区分。** C-U1 的动作空间是 `R^2`，reward 对任意动作连续可计算；负样本集合来自以 `a_star(s)` 为圆心、半径 1.20 的连续等值圆周。正式数据每状态只取 8 个均匀角度，是有限 offline dataset 的支持设计，不是分段或不连续 reward。
2. **等 reward/advantage 是人为控制变量。** 它不是行为策略自然采样后的经验巧合。这样设计是为了排除“far 样本梯度更大只是因为 reward 更低或 `|A|` 更大”的混杂，使 near/far 差异主要来自当前 policy score geometry 与方向。
3. **质量解耦不等于方向效用解耦。** 对负样本，均值分支更新方向与 `mu-a` 同向；其相对真实 improvement direction `a_star-mu` 的 cosine 决定局部 utility。当前圆周含 `a_minus=a_plus-0.50u`，排斥该近场点朝向 hidden optimum；圆周另一侧的远点排斥方向可与 hidden optimum 相反。因此相同 advantage 可以具有不同 directional utility。
4. **允许的机制表述。** 当前环境展示一种受控且现实相关的结构：局部负样本仍可能提供 boundary shaping，随着 policy-relative remoteness 增大，方向相关性可能下降或反转，而 Gaussian score influence 仍增长。Distance taper 处理的是这种 informativeness--amplification mismatch。
5. **禁止的普遍化。** 不得写成“near negative 必然有益”“far negative 必然有害”或“distance 在任何任务中都是 oracle utility”。真实任务中的 utility--distance 关系必须由多几何稳健性和 Hopper/Countdown/推荐外部验证测量。
6. **未来透明化材料。** 论文附录至少报告：负 advantage 对 distance 的水平匹配；未加权 score/influence 随 distance 的变化；负更新与 oracle improvement direction 的 cosine；各 taper 后的有效 `utility x influence`。这属于解释与审计，不改变 v43 的冻结结果。

### 3.8.5 函数族公平性、解析阶数与后续验证（v44 澄清）

1. **当前比较匹配了什么。** 三个 family 共享 `w(d_ref)=rho`、同一距离、同一初始化、同一 advantage、同一 minibatch stream。它们没有匹配 `w'(d_ref)`、near-bin 平均权重、总负梯度 norm 或累计 optimizer update。
2. **当前结果能回答什么。** 它回答 anchor-normalized protocol 下的形状差异：在同一 `rho` 下，Quadratic 在 `d<d_ref` 保留更多、在 `d>d_ref` 抑制更强，并在正式 paired seeds 上产生更低 far/near ratio。它不回答各 family 独立充分调参后的最优 reward 排名。
3. **超参数不能改变尾部阶数。** 对 `w_p(d)=[1+lambda(d/d_ref)^p]^{-1}`，任意有限 `lambda>0` 只改变常数，不改变 `w_p(d)=Theta(d^{-p})`。在 learnable-log-scale 输出分支 `Theta(d^2)` 下，`p<2` 仍无界，`p=2` 有界，`p>2` 趋零；Exponential 支配任何有限多项式增长。该结论是渐近影响界，不是 task reward 定理。
4. **衰减并非越重越好。** 将任一 family 系数无限增大会趋近 positive-only，可能丢失 E4 已证明有价值的局部负信号。因此优化目标不是最小化全部负权重，而是在保持局部信息的条件下最小化远场风险。
5. **后续公平比较的最低要求。** 至少分别完成：
   - 匹配 `E[w(d)|near]` 或预注册 near-bin retention 后比较 far risk；
   - 与 Global alpha 做逐步或累计 negative-gradient budget matching；
   - 每个 family 使用相同 dev-search trial 数，冻结超参后在全新 confirmatory seeds 评估；
   - 报告 Pareto frontier：near retention、far influence、task reward、sigma/support 与三类失效事件；
   - 保持原 Adam 做长程状态审计，并在必要时另用预注册 full-batch polish/root finding 检查 objective stationary solution。
6. **执行门禁。** 上述项目尚无可运行 experiment ID；不得复用 seeds 70--89 作为新的 confirmatory set，也不得擅自修改 horizon、optimizer、阈值或当前 E4-TAPER 定义。任何执行必须先给出独立 ID、冻结参数和对既有路线的影响。

<!-- HANDOFF-DELTA-BLOCK:section_end:v60-e4-taper-utility-theory-and-followups:START -->
### 3.8.6 负样本 alignment utility、正交代价与净效用假设（v60）

对负优势样本定义其参数更新为

$$
g^-(s,a)=A^-(s,a)\nabla_\theta\log\pi_\theta(a\mid s),\qquad A^-<0,
$$

并以任务的 oracle improvement direction `g_star(s)` 为参照。定义条件平均 alignment utility

$$
U_{\mathrm{align}}(d)=\mathbb E[\cos(g^-,g^\star)\mid d],
$$

以及正交 nuisance cost

$$
N_\perp(d)=\mathbb E[\lVert P^\perp_{g^\star}g^-\rVert_2^2\mid d].
$$

净效用写成

$$
U_{\mathrm{net}}(d)=U_{\mathrm{align}}(d)-\kappa N_\perp(d),\qquad \kappa>0.
$$

这里正交梯度的一阶投影收益为零，但仍会占用更新预算、增加梯度方差、引入曲率路径偏移并可能推动 variance/support boundary，因此净效用可以为负。本文只采用一个**条件经验假设**：离开局部信息区后，`U_net(d)` 总体不增，并可能趋零或转负。该假设可证伪但不是普遍定理；本文不假设它具有指数衰减速度，也不要求研究其精确函数形状。

### 3.8.7 Quadratic bounded influence 与 Exponential vanishing influence（v60）

在 bounded advantage、pre-boundary `sigma>=sigma_min>0` 和 learnable Gaussian log-scale 分支下，原始 far-field influence 为 `Theta(d^2)`。若使用 reciprocal quadratic

$$
w_{\mathrm{quad}}(d)=\frac{1}{1+\lambda(d/d_{\mathrm{ref}})^2},
$$

则 `w_quad(d)->0`，但

$$
d^2w_{\mathrm{quad}}(d)\to d_{\mathrm{ref}}^2/\lambda,
$$

所以 Quadratic 的严格作用是把远场影响从无界增长压成一般非零的有界常数。它是 learnable-log-scale 二次分支的最低充分多项式有界阶，同时在近场满足 `1-w_quad(d)=O(d^2)`，比 reciprocal-linear 的一阶近场损失更平坦。

若远场净效用趋零或转负，则更强的合理目标是

$$
w(d)d^2\to0,
$$

即 `w(d)=o(d^-2)`。`p>2` reciprocal polynomial 和 exponential tail 都满足；Exponential 的价值在于对任意固定有限阶多项式增长提供平滑 vanishing influence，而不是因为本文强行假设效用按指数下降。它不是唯一解，也不由当前理论推出 universal reward winner。

历史 E4-TAPER 使用的 `exp(-lambda*u)` 公式保持不变。另一个待冻结候选 `exp(-beta*u^2)` 同时具有 `w'(0)=0` 的近场二阶平坦性和远场指数趋零；它只有在新实验显式冻结后才能加入比较，不能替换或重解释既有正式结果。

### 3.8.8 四项后续实验登记与职责拆分（v60）

1. **`C-U1-E4-TAPER-NEAR-RETENTION-01`：** 对每个 family 独立校准系数，使预注册 near 区域的平均 `E[w(d)|near]` 相同；比较 near useful retention、far harmful influence、far/near gradient ratio、held-out-context reward、sigma/support 与三类失效事件。它排除“某方法只是整体压得更重”的解释。
2. **`C-U1-E4-TAPER-BUDGET-MATCH-01`：** 在相同逐步负梯度 norm 或累计 negative optimizer update 下比较 Distance families 与 Global alpha，冻结后只允许 near/far 预算分配不同。它排除“收益只来自总负更新更小”的解释。具体 primary budget definition 必须在实施包中二选一并冻结，不能看结果后切换。
3. **`C-U1-E4-TAPER-CONV-01`：** 前两项交付并冻结 method shortlist/超参后，使用原 Adam 动力学、连续 optimizer state、预注册终态窗口和完整 2x continuation 解析长期状态；不得直接延长旧 `C-U1-E4-TAPER-01`。full-batch stationary audit 如需执行，必须另行登记且不能替代 Adam long-run。
4. **`C-U1-E4-TAPER-CONFIRM-01`：** 所有公式、超参、主要 claim、终态标准和分析计划冻结后，使用全新 untouched seeds 一次性确认；seeds 70--89 只能作为既有 development/formal evidence，不能再次充当 confirmatory set，确认开始后禁止 retune。

四项共同使用 C-U1 同分布 held-out-context terminology，并继续分报 task-performance collapse、support/variance boundary 与 NaN/Inf。任何 family winner、Exponential 优于 Quadratic、Distance 优于 Global alpha 或稳定 fixed-point 排名，都必须等待对应实验和终态审计，不能由登记本身推出。

### 3.8.9 当前阶段闭环与低优先级项目（v60）

当前 E4-TAPER 已完成机制层阶段闭环：anchor-normalized protocol 下 Quadratic 相对 Linear 的 far-field suppression order 获得正式 paired evidence，并清楚记录终态未解析和公平性限制。新四项用于升级公平方法比较和长期/确认性证据，不是修复一个已知致命漏洞。连续角度、随机 phase、轮廓分辨率、薄圆环 jitter 与 reward-bin matching 的几何 robustness extension 保持低优先级 optional study；有时间可增强附录，没有执行也不阻塞当前四项路线。
<!-- HANDOFF-DELTA-BLOCK:section_end:v60-e4-taper-utility-theory-and-followups:END -->
<!-- HANDOFF-DELTA-BLOCK:section_end:v61-e4-taper-near-retention-protocol:START -->
### 3.8.10 Near-Retention Matching 正式协议（v61）

**实验 ID 与职责。** `C-U1-E4-TAPER-NEAR-RETENTION-01` 是 E4-TAPER 四项后续中的第一项，只回答“在保留相同平均近场负信号时，函数形状如何重新分配 useful-near 与 harmful-far influence”。它不回答总负更新预算公平性，不负责长期 shortlist 稳态解析，也不构成 untouched-seed confirmation。

**Near 区域与校准防火墙。** 唯一 near 定义为 frozen 2000-step positive-only Adam checkpoint 上的

$$
d_\theta(s,a)=\frac{\lVert a-\mu_\theta(s)\rVert_2}{\sigma_\theta(s)}\le 5.
$$

校准只使用 development seeds `0--4` 的训练负样本，先 pooling 全部 near distances，再对每个 family/retention target 通过确定性单调二分求解

$$
\mathbb E_{\text{dev pooled}}[w_c(d)\mid d\le5]=r.
$$

`r` 的主层级为 `0.75`，敏感性层级为 `0.50` 和 `0.25`；绝对匹配误差必须不超过 `1e-6`。系数一经求出，在 formal seeds 与全部训练步上固定，不按 seed、minibatch、验证 reward 或终态结果重新校准。seeds `70--89` 只保留为 predecessor evidence，formal paired seeds 冻结为 `90--109`；`110+` 在后续 protocol freeze 前保持 untouched。

**函数族。** 令 `u=d/5`，冻结四个候选：

$$
w_{\mathrm{lin}}=\frac{1}{1+cu},\quad
w_{\mathrm{quad}}=\frac{1}{1+cu^2},\quad
w_{\exp}=e^{-cu},\quad
w_{\exp2}=e^{-cu^2}.
$$

`w_exp` 与历史 E4-TAPER 公式同族但采用 near-retention-derived coefficient；`w_exp2` 是本实验首次显式批准的 squared-distance exponential。二者都不得被用来重解释旧结果。Positive-only 与 unweighted-negative 每 seed 运行一次，只作为边界对照。

**Useful/Harmful 诊断。** 对负样本，输出均值分支的负更新方向为 `|A^-|(mu-a)/sigma^2`，oracle improvement direction 为 `a_star-mu`。`d<=5` 且投影为正定义为 useful-near；`d>5` 且投影为负定义为 harmful-far。主要报告：near-region mean weight、near useful positive-projection mass retention、far harmful negative-projection mass retention 与 weighted projection、全参数 contour-4/contour-0 gradient ratio，以及 `[0,2.5),[2.5,5),[5,7.5),[7.5,10),[10,inf)` 的 alignment、orthogonal fraction 和 weighted directional utility。归一化方向效用使用 `cos - (1-cos^2)`（`kappa=1`）作为无量纲诊断，不声称它等同于普遍的维度化 `U_net`。

**训练与终态。** 初始化、Adam `lr=5e-4`、negative alpha `1.0`、state minibatch `256`、8000-step 上限、每 100 steps 评估、稳定窗口与 2x candidate audit 均继承旧 TAPER protocol。任务效果崩溃、support/variance boundary 与 NaN/Inf 继续分报。由于本实验没有匹配 total negative-gradient/optimizer budget，也不承担最终 long-run shortlist，正式完成后的科学状态最高只能是 **有限训练步数验证**；即使个别运行通过 2x plateau，也不得提前关闭 `CONV-01`。

**主统计与非结论。** 主保持率 `0.75` 下，以 reciprocal-linear 为 reference，对其余三个 family 做 20-seed paired bootstrap；`0.50/0.25` 只作形状敏感性。far-risk、near-retention 与 reward 同时报；不预注册 reward winner，不预设 Exponential、Squared-Exponential、Quadratic 或 Linear 获胜，也不得由该实验声称 Distance 优于 Global alpha。
<!-- HANDOFF-DELTA-BLOCK:section_end:v61-e4-taper-near-retention-protocol:END -->
<!-- HANDOFF-DELTA-BLOCK:section_end:v63-e4-taper-near-result-and-closure-protocol:START -->
### 3.8.11 Near-Retention 结果沉淀与闭环实验协议（v63）

**Near-Retention 正式结果。** `C-U1-E4-TAPER-NEAR-RETENTION-01` 在 run commit `69c8f532570b5c4377a0cd35ff42f0bcb77afef0` 上完成 development seeds `0--4`、formal seeds `90--109`、每 seed 14 configurations，共 `280/280` runs。近场平均保留率的最大校准误差为 `1.11e-16`，通过 `1e-6` 门槛。主保留率 `r=0.75` 下，以 Reciprocal-Linear 为 reference：

| Candidate | mean held-out-context reward delta | positive paired seeds |
|---|---:|---:|
| Reciprocal-Quadratic | +0.012002 | 20/20 |
| current Exponential | +0.015619 | 20/20 |
| Squared-distance Exponential | +0.036134 | 20/20 |

Reciprocal-Linear 的 harmful-far retention 为 `0.055886`，Squared-distance Exponential 为 `0.010382`。因此可以写成：**在冻结 C-U1、相同初始近场平均保留率和 8000-step horizon 下，更快尾部衰减与更低 harmful-far influence、更高 held-out-context reward 一致相关，Squared-distance Exponential 是当前最强候选。** 不可写成 steady-state winner、universal winner、Distance 必然优于 Global alpha、跨任务优越或 OOD generalization。

**终态与失败边界。** `280/280` coverage 完整；task-performance collapse `13/280`、support/variance boundary `20/280`、NaN/Inf `0/280`，前两类全部来自 unweighted control。`260/280` runs 在 8000 steps 仍 terminally unresolved，因此科学状态只能是 **有限训练步数验证**。compact repository summary 位于 `outputs/cu1_e4_taper_near_retention/`；它记录正式汇总和 claim boundary，不替代原 raw trajectories/checkpoints。当前构建会话缺少原 raw-complete artifact 与 SHA256，归档发布前必须恢复，禁止补造。

**Budget-Match primary fairness coordinate。** `C-U1-E4-TAPER-BUDGET-MATCH-01` 唯一冻结 primary 为

$$
\left\|g^-_{m,t}\right\|_2 = \left\|g^-_{\mathrm{lin},t}\right\|_2,
$$

其中 norm 在每个 minibatch、Adam 之前、全 actor 参数空间计算。每个 paired seed 先运行 Reciprocal-Linear reference，使用与所有方法相同的初始化和 minibatch index stream，生成逐步目标 schedule。对 Candidate 方法，令 raw negative gradient 为 `g^-_m`，应用 detached scalar

$$
s_{m,t}=\frac{\lVert g^-_{\mathrm{lin},t}\rVert_2}{\lVert g^-_{m,t}\rVert_2},
$$

再与同一步 positive gradient 相加。匹配误差门槛为 `1e-6`。`global_stepwise_scale` 使用 unweighted negative-gradient direction，也按同一 schedule 缩放，因而是 non-selective global control。该 protocol 匹配 raw negative-gradient L2，不匹配 Adam preconditioned negative-only parameter update；实际 total Adam parameter-update norm 必须单独记录，不得把本实验改写成 optimizer-update matching。

**Budget-Match 方法、seeds 与 horizon。** 近场系数继续只由 development seeds `0--4`、target retention `0.75` 校准。matched methods 为 Reciprocal-Linear、Reciprocal-Quadratic、current Exponential、Squared-distance Exponential、Global stepwise scale；Positive-only 与 raw Unweighted 只作边界 controls。formal paired seeds 固定 `110--129`；8000 steps、Adam `lr=5e-4`、batch 256、每 100 steps evaluation、原三类事件阈值不变。它仍只形成 finite-horizon fairness evidence，状态上限为 **有限训练步数验证**；不承担最终终态排名。

**Convergence 冻结壳。** `C-U1-E4-TAPER-CONV-01` 只在 Budget-Match 交付后生成 `FROZEN_CONVERGENCE_SHORTLIST.json`。必含 Positive-only、Unweighted boundary、Reciprocal-Linear、Global stepwise scale；Selective 候选池是 Reciprocal-Quadratic、current Exponential、Squared-distance Exponential，最多选两个。候选必须同时满足：Near-Retention 主结果相对 Linear 至少 `18/20` reward 正差；Budget-Match 相对 Global 至少 `18/20` harmful-far retention 更低；相对 Linear 至少 `18/20` reward 非负；NaN/Inf 不多于 Linear。若超过两个，依次按 Budget-Match mean reward 降序、harmful-far retention 升序、family 名字字典序裁决，禁止人工看结果改 shortlist。

Convergence 继续使用 seeds `110--129`，从 Budget-Match 8000-step actor 与 Adam optimizer checkpoint 原位续训；Reciprocal-Linear 先继续产生 8001--32000 的 budget schedule，其余 matched methods 消费相同 schedule。最大 total steps `32000`，原 slope/residual 阈值和 2× continuation 保持；明确 persistent drift/runaway 也可作为已审计终态分类。没有 exact actor+optimizer state、shortlist hash 或 predecessor delivery 时 fail closed。

**Independent Confirmation 防火墙。** `C-U1-E4-TAPER-CONFIRM-01` 的 untouched seeds 现在冻结为 `130--149`，在 confirmation config 完整冻结前任何代码、校准、smoke 或 exploratory analysis 都不得访问。确认阶段继承最终 shortlist、系数、budget rule、32000-step 上限和终态标准，禁止 retune 或改 primary claim。机制、任务和终态分开判断：near-useful non-inferiority、far-harmful improvement、paired reward vs Linear/Global、terminal classification 与三类 failure 各自报告；最低方向一致性门槛为 `16/20`，并给 paired 95% bootstrap interval。任务 superiority 不成立不能抹除机制结果，机制成立也不能冒充 reward 或稳态 superiority。
<!-- HANDOFF-DELTA-BLOCK:section_end:v63-e4-taper-near-result-and-closure-protocol:END -->
<!-- HANDOFF-DELTA-BLOCK:section_end:v66-e4-taper-budget-match-result:START -->
### 3.8.12 Budget-Match 正式结果与证据边界（v66）

**运行与公平性。** `C-U1-E4-TAPER-BUDGET-MATCH-01` 在 commit `1faea3a92f74af5d11409779d96b9ed21fe846ad` 上完成 seeds `110--129`、7 methods，共 `140/140` runs。每个 paired seed 由 Reciprocal-Linear 先生成逐步 target schedule；Reciprocal-Quadratic、current Exponential、Squared-distance Exponential 与 non-selective Global stepwise scale 在同初始化、同 minibatch stream 下，用 detached scalar 匹配每一步 Adam 前的 raw negative-gradient L2 norm。最大相对误差为 `2.11795e-16`。Adam total parameter-update norm 只是 secondary diagnostic，不在 matched coordinate 内。

| Method | mean held-out-context reward | delta vs Linear | positive paired seeds | mean harmful-far retention | lower harmful-far seeds vs Linear |
|---|---:|---:|---:|---:|---:|
| Reciprocal-Linear | 0.631452 | 0 | — | 0.055866 | — |
| Reciprocal-Quadratic | 0.647464 | +0.016011 | 20/20 | 0.043338 | 20/20 |
| current Exponential | 0.719641 | +0.088189 | 20/20 | 0.002300 | 20/20 |
| Squared-distance Exponential | 0.762069 | +0.130616 | 20/20 | 9.28e-40 | 20/20 |
| Global stepwise scale | 0.624570 | -0.006883 | 0/20 | 0.063525 | 0/20 |
| Positive-only | 0.646858 | — | — | 0 | — |
| Unweighted boundary | 0.259398 | — | — | 1.0 | — |

因此，在当前冻结 C-U1 与 8000-step horizon 中，**总 raw negative-gradient norm 相同并不足以复现 selective taper 的结果；把预算从 harmful far field 重新分配的形状差异具有独立有限步信号。** 这不等于 Distance 必然优于任何 Global 方法，因为这里只比较一个严格登记的 non-selective stepwise control，也不等于稳态或跨任务排名。

**未闭合的 near 侧。** `near_useful_gradient_retention` 的 terminal aggregate 在非 Positive-only 方法上为 NaN，原因是 raw useful-near positive-projection denominator 为零。它是不可评估，不是 0，也不是 1。因此 Budget-Match 不能独立闭合“更多预算留给 useful-near”；该子 claim 仍由 Near-Retention 的固定初始 near-region matching 证据承担。后续 shortlist 规则只使用已预登记的 Near-Retention near 条件与本实验的 harmful-far/reward 条件，不得用 NaN 后验补门禁。

**事件与终态。** task-performance collapse 为 `13/140`、support/variance boundary 为 `20/140`、NaN/Inf 为 `0/140`，前两类只出现在 unweighted boundary；controlled methods 全部为 0。`terminal_audit.json` 通过的是 coverage、budget tolerance、reference schedule 未重心化与无 NaN/Inf，不是所有方法已收敛。科学状态固定为 **有限训练步数验证**，长期状态由 `CONV-01` 独立负责。

**收尾故障记录。** 原 guard 在子进程 return code 0 后，因缺少 `scientific_run_manifest.json` 和默认主包超过 25 MiB 而将 lifecycle 写为 failed。问题属于 runner/packaging contract，不是 task collapse、support boundary 或 numerical collapse。原 failed tree 与 failure markers 不删除；v66 只在仓库代码中补写 manifest，并把完整原树作为显式 raw sidecar 交付，compact summary 位于 `outputs/cu1_e4_taper_budget_match/`。
<!-- HANDOFF-DELTA-BLOCK:section_end:v66-e4-taper-budget-match-result:END -->

## 3.9 E6--E8 方法迁移与规模验证路线（v42 锁定）

1. **E6：** pilot 与 focused development 已完成 development seeds 0--4 的 105/105 与 165/165 runs；`D-U1-E6-SEMANTIC-LONGRUN-01` 已在 untouched seeds 10--29 上完成 360/360 formal runs并通过 2x 终态审计。结果支持 positive-only ceiling、受控 local negative 的同分布 held-out-context / unseen-action 收益、过强压力反转、任务与支持事件分离以及语义置乱排他性。
2. **`D-U1-E6-TAPER-01`（E6-TAPER）：** 在 E6 long-run 冻结的同一个 semantic remoteness coordinate 上比较 reciprocal-linear、reciprocal-quadratic 与 exponential，并包含 positive-only、uncontrolled 和 global-alpha controls。该实验验证控制思想跨策略族迁移，不声称 categorical policy 具有 Gaussian 的二次梯度临界界。
3. **`EXT-H-E7-Q2`（E7-MECH）：** Hopper learned-critic 深度机制 runner 已实现，但 formal launch 继续 blocked，直到 E6-TAPER 交付。该实验回答真实数据是否进入 Gaussian log-scale 二次主导区、是否传导到 full-parameter gradient/长期动力学；不承担大规模方法排名。
4. **`EXT-H-E7-BENCH-01`（E7-BENCH）：** 公共大规模连续控制主表固定为 D4RL MuJoCo locomotion suite：Hopper、Walker2d、HalfCheetah × medium、medium-replay、medium-expert，共 9 tasks。方法 shortlist 与超参从 E4/E6-TAPER 冻结，不得在 D4RL 上按任务重新选择方法族；主报 normalized return、多 seed 区间、跨任务平均排名、最差 seed 与三类失效事件。AntMaze/Kitchen/Adroit 不属于本主表，可另行登记 stress test。
5. **`EXT-C-E8-V4.2`（E8-MECH）：** 0.5B Countdown/Qwen 继续承担 Transformer 固定负优势 near/far probe、pipeline 与小规模方法信号，不承担最终规模结论。
6. **`EXT-C-E8-SCALE-01`（E8-SCALE）：** 在方法 shortlist 冻结后，使用更大固定 Countdown offline dataset；3B 为正式主模型，7B 只做冻结配置确认，不在规模实验重新筛选方法族。
7. **执行顺序：** `E4-TAPER -> E6 -> E6-TAPER -> E7-MECH -> E7-BENCH -> E8-MECH -> E8-SCALE`。E4-TAPER 已以 finite-step status 交付；E6 long-run 已以 long-run validated 状态交付。当前下一阶段是先审阅、冻结并实现 `D-U1-E6-TAPER-01`，不能直接运行其 planned registration；每个正式 ID 必须先完成 terminal audit、packaging 和 delivery，下一正式 ID 才可启动。

---

<!-- HANDOFF-DELTA-BLOCK:section_end:v52-e8-route-override:START -->
7. **v52 路线覆盖：** 上述第 5 项的当前 E8-MECH owner 更新为 `EXT-C-E8-V4.3`。V4.3 只修复长期训练中的动态 remoteness 控制并保留 V4.2 静态方法作消融；E8-SCALE 的 3B/7B 规模结论仍需后续独立执行。
<!-- HANDOFF-DELTA-BLOCK:section_end:v52-e8-route-override:END -->
<!-- HANDOFF-DELTA-BLOCK:section_end:v57-e8-offline-online-route:START -->
8. **v57 E8 内部路线覆盖：** 在进入 E8 外部诊断时，先执行 `EXT-C-E8-V4.4-OFFLINE-BANK`，只改变 fixed-bank 密度与每步动态选择；online off-policy 必须作为独立 successor 重新冻结 rollout actor、同步滞后、replay age、seeds 与预算匹配，不能与 V4.4 共用结论。
<!-- HANDOFF-DELTA-BLOCK:section_end:v57-e8-offline-online-route:END -->
<!-- HANDOFF-DELTA-BLOCK:section_end:v59-e8-offline-tuning-route:START -->
9. **v59 E8 内部路线覆盖：** V4.4 fixed-bank 之后先运行 V4.5 validation-only α×λ 调参，检验当前 dynamic 方法是否只是控制强度偏保守。只有调参仍不能产生稳定收益时，才进入另行登记的 online off-policy successor；不得用 test 反复挑选参数，也不得把 V4.5 变成无界 HPO。
<!-- HANDOFF-DELTA-BLOCK:section_end:v59-e8-offline-tuning-route:END -->
<!-- HANDOFF-DELTA-BLOCK:section_end:v62-e8-online-offpolicy-route:START -->
10. **v62 E8 内部路线覆盖：** V4.5 已完成其“alpha/lambda 是否未调到位”的职责后，不再扩大 frozen-bank HPO。V4.6 用全新 paired seeds 执行 frozen/online × positive/dynamic 2×2；只有 online negative 相对 online Positive-only 的 paired 增量与 refresh×negative interaction 才能支持“动态负样本有额外价值”。若 online 两个 cells 都提高但彼此持平，收益归因于数据刷新；若 online dynamic 仍不占优，不得继续用 bank staleness 解释。
<!-- HANDOFF-DELTA-BLOCK:section_end:v62-e8-online-offpolicy-route:END -->

# 4. 论文机制实验总表与验收标准

| ID | 实验 | 核心问题 | 是否要求训练饱和 | 正式验收 |
|---|---|---|---|---|
| E1 | 瞬时梯度来源隔离 | 相同 advantage 时，远样本梯度是否更大 | 否 | reward/advantage 跨距离误差接近 0；far/near score 与全参数梯度；20 seeds |
| E2 | Positive-only 完整动力学 | 正拟合是否使固定负样本远场化与梯度增长；最终是否平台 | 是 | mu、sigma、正样本 loss、负样本距离、phantom gradient 和梯度比均通过停止标准 |
| E3 | Joint + Near/Far 因果干预 | 远场异常负梯度是否是 drift/collapse 主路径 | 是 | Baseline/Near-zero/Far-zero/Far-cap/Global/Far-to-near；早期时序 + 长期结果；20 held-out seeds |

**E3 结果状态必须拆分（v12 新增协议细化）：**

1. **任务效果崩溃、数值训练仍可运行**：evaluation reward 显著失效，但 loss、梯度和参数仍为有限值，训练没有 NaN/Inf。
2. **任务效果与数值训练同时崩溃**：除 reward 失效外，还出现非有限 loss/gradient/parameter、方差触底或优化器无法继续。

该区分在早期讨论中存在概念基础，但旧 E3 表格和正式输出没有明确登记；是在用户本轮提醒后才补入实验协议。因此不能声称旧实验已经完整报告了两类崩溃。
| E4 | 稳定外推与泛化 | 受控负梯度能否突破 positive-only 上限，远场是否反转为有害 | 是 | 策略越过 a_plus、接近 a_star；训练分布内/同分布 held-out-context reward；强度扫描；控制恢复；固定/可学习方差 |
| E4-TAPER | 距离衰减阶数 | 同一标准化距离上二次 reciprocal 是否比线性 reciprocal 更强压制远场负梯度；是否改善任务效果 | 是 | 20 paired seeds；主 rho=0.25；实际全参数 far/near ratio；held-out-context reward；2× 终态审计；三类失效分报 |
| E5 | Categorical 排斥与支持边界 | 有界 logit score 下重复负更新如何把概率推向边界 | 解析 + 长期 | direct-softmax 解析、概率衰减、rare/common 干预 |
| E6 | 共享语义 categorical 外推 | 负梯度能否利用共享表示改善未见动作且避免 support collapse | 是 | unordered semantic actions；E6 pilot 只冻结协议；long-run 承担 E6-A/B/C 与语义置乱排他性 |
| E6-TAPER | categorical 方法迁移 | 同一 semantic remoteness 上 Linear/Quadratic/Exp 是否兼顾未见动作收益与支持稳定 | 是 | paired stream；distance definition 冻结；positive-only/uncontrolled/global-alpha controls；不声称 Gaussian 二次界 |
| E7-MECH | Hopper learned-critic | 真实数据是否进入并受 Gaussian 二次 log-scale 远场区影响 | 是 | 优势匹配；mean/log-scale 分解；full-parameter 传导；长期 Near/Far/Far-cap/Global；终态审计 |
| E7-BENCH | D4RL MuJoCo locomotion | bandit 中冻结的方法是否在 9 个公共连续控制任务上改善 normalized return 与稳定性 | 是 | Hopper/Walker2d/HalfCheetah × medium/replay/expert；多 seed；平均排名；三类失效分报 |
| E8-MECH | Countdown/Qwen 0.5B | Transformer 中固定负优势 near/far 机制迁移与小规模方法信号 | base-first 门禁；必要时最小 SFT fallback；best + terminal/last-finite 审计 | 0.5B BF16-LoRA pilot；固定 A=-1 probe；Positive-only/Controlled/Uncontrolled/Global-matched |
| E8-SCALE | Countdown 大模型/大数据 | 冻结方法在更大固定数据和 3B/7B 模型上是否保持效果 | 是 | 3B 主结果；7B 冻结确认；不重新筛方法族；性能、支持/熵边界、NaN/Inf 分报 |

## 4.1 动力学实验统一收敛标准

不能用固定的 500/1000/10000 步替代收敛判断。所有 E2/E3/E4/E6/E7 需要：

1. 预先定义最大训练步数；
2. 连续多个评估窗口中，核心状态量斜率低于阈值；
3. 更新向量/梯度场残差足够小，或明确持续 runaway；
4. 将训练步数延长至少 2 倍，状态分类、主要结论和方法排序不反转；
5. 检查是否由 clamp、temperature floor 或数值溢出造成假平台。

---

# 5. 当前真实完成状态

| 实验 | 旧环境结果 | 真正统一环境状态 | 论文可用状态 |
|---|---|---|---|
| E1 | Product 环境已完成，逻辑严密 | **C-U1 正式 20-seed 已完成**：positive-trained full-gradient far/near 9.093×，aggregate 10.072×；advantage 1.000× | 正式机制识别完成；数值倍率仅限本环境 |
| E2 | 零散 positive-only 曲线 | **C-U1 正式 20-seed 已完成**：20/20 通过稳态与 2× 延长审计；phantom gradient 增长 28.93× | 正式长期验证完成 |
| E3 | Product/Collapse 环境与旧 SGD C-U1 结果保留 provenance | **`C-U1-E3-ADAM-RERUN` 已完成并交付**：固定方差 Baseline/Near-zero 20/20 任务崩溃，Far-zero/Far-cap 0/20；可学习方差 Baseline/Near-zero 20/20 support contraction，远场控制 0/20；NaN/Inf 0/220 | **已长期验证，论文可用**；主文采用四方法 fixed-variance 因果链，learnable-variance 作互补 panel/附录 |
| E4 | 独立 Extrapolation 环境；部分长程审计 | **`C-U1-E4-ADAM-RERUN` 已完成并交付**：有限步 reward 相变、过强压力任务崩溃、learnable-variance support contraction 与 4000-step controls 均完成；受益分支未通过 20/20 双 residual audit | **有限训练步数验证**；可用于有限步相变图与失稳分支，暂不可写成稳定有益 fixed point |
| E4-CONV | 无历史独立环境结果 | **4000-step 正式运行已完成**：`0.75/1.00/1.25` 目标状态分别为 15/20、16/20、15/20，剩余均 inconclusive，0 个明确相反终态，60/60 科学角色未反转 | **已长期验证（用户确认闭环）**；原 18/20 门禁未通过的事实继续保留，不等同于 20/20 fixed-point 认证 |
| E4-TAPER | seeds 0--4 独立复制实现 pilot | **正式 seeds 70--89 已完成 220/220 runs**：quadratic vs linear 在主 rho=0.25 上 20/20 更强抑制远场且 20/20 reward 更高；200 controlled/positive runs 到 8000 steps 仍无稳定候选，20 unweighted runs 触发 support boundary | **有限训练步数验证**；机制阶数 claim 可用，稳定终态和 universal method ranking 不可声称 |
| E5 | 历史解析、direct-softmax 与 20-seed 因果结果保留；旧 runner/raw artifact 未入库 | **`D-U1-E5-LONGRUN-RERUN` 已完成**：direct-softmax 参照通过，120/120 长程因果 runs 全部分类且 120/120 复现历史 qualitative class，NaN/Inf 0/120 | **已长期验证**；受控 categorical 排斥、支持边界和 near/far 因果链可用于论文，E6 语义泛化仍未完成 |
| E6 | unordered semantic categorical pilot/focused runner 与 formal runner/config 均已实现 | **`D-U1-E6-SEMANTIC-LONGRUN-01` 已完成 360/360 formal runs**：E6-A 受控 local negatives 在 alpha 0.25/0.50 上 20/20 胜过 positive-only，alpha 0.75 出现 20/20 reward 反转；E6-B task collapse 0、support boundary 120；E6-C aligned 在四方法上均 20/20 胜过 shuffled | **已长期验证**；可用于 positive-only ceiling、受控负梯度非单调收益、支持边界分离与 semantic-alignment 排他性，不能称 OOD 或 universal method ranking |
| E6-TAPER | 无正式结果 | `D-U1-E6-TAPER-01` 的 predecessor delivery 已满足，但 semantic remoteness coordinate、paired protocol、新 untouched seeds 与独立 runner 尚未冻结/实现 | 未完成、review-required + blocked；不得套用 Gaussian 二次界或自动启动 |
| E7-MECH | Hopper learned-critic 600-step probe | `EXT-H-E7-Q2` runner/config/operator/test 已在 commit `f64452a7452274a183b03c87c39b847039230c00` 实现；formal launch 仍等待 E6-TAPER 交付 | 旧 probe 仅有限步；新实现科学状态仍为 not_run/blocked |
| E7-BENCH | 无 9-task 主表 | `EXT-H-E7-BENCH-01` 已登记 D4RL MuJoCo locomotion 9-task scope | 未完成、blocked；等待 E7-MECH 与 bandit shortlist |
| E8-MECH | v4.2 平衡离线集与动态诊断实现已登记；V4.1 off-protocol 单 seed 仅保留开发 provenance | V4.2 未在 clean committed source 上完成真实 Qwen/CUDA/BF16-LoRA 运行 | 尚未运行；不得把静态/CPU 测试或 V4.1 off-protocol pilot 当正式结果 |
| E8-SCALE | 无规模结果 | `EXT-C-E8-SCALE-01` 已 planned 登记 3B 主结果与 7B 冻结确认 | 未完成、blocked；精确数据规模/seeds 待运行前冻结 |

---

<!-- HANDOFF-DELTA-BLOCK:section_end:v55-du1-e6-semantic-gap-completion-status:START -->
**v55 E6 Semantic-Gap 结果补充：** `D-U1-E6-SEMANTIC-GAP-LONGRUN-01` 已完成 100/100 runs。32k 时 `alpha=0.25/0.50` 均 20/20 胜过 Positive-only；`alpha=1.0` 相对差距随 8k→32k 由 `-0.013741` 扩大至 `-0.061085`，20/20 失败。由于仅 45/100 terminal plateau，论文可用状态限定为有限 horizon trajectory 与 paired finite-step claim，不允许全方法稳态排名。三类失效事件分别为 0/100、0/100、0/100。
<!-- HANDOFF-DELTA-BLOCK:section_end:v55-du1-e6-semantic-gap-completion-status:END -->
<!-- HANDOFF-DELTA-BLOCK:section_end:v56-e6-parent-closure-completion-status:START -->
**v56 E6 父实验关闭判断：** E6 已达到当前论文所需的机制与泛化证据闭环：主语义 long-run 给出 Positive-only ceiling、适度负信号收益、过强压力反转与 semantic-alignment 排他性；semantic-gap successor 复现中等 alpha 收益和 `alpha=1` 随 horizon 扩大的退化；conditional-gap stress diagnostic 证明更强支持缺口下局部收益与 overall trade-off、强压力任务崩溃及控制救援。关闭的是上述父 claim，不是把所有子运行宣称为稳态，也不是冻结 universal 方法排名。
<!-- HANDOFF-DELTA-BLOCK:section_end:v56-e6-parent-closure-completion-status:END -->
<!-- HANDOFF-DELTA-BLOCK:section_end:v75-countdown-status-note:START -->
- **v75 Countdown 逐样本机制诊断补记：**`EXT-C` 已完成一个 single-seed full-bank `arithmetic_wrong` response diagnostic：`6000` puzzles × near/far = `12000` rows，固定 `negative_coefficient_abs=1.0`，观察到 surprisal 与 trainable-parameter gradient norm 的正相关、near/far 配对增益和 decile 平台化趋势。该补记只把 Countdown 机制观察从 10-puzzle smoke 升级为 full-bank pilot；不升级 `EXT-C-E8-TAPER-0.5B-01` 或 `EXT-C-E8-SCALE-01` 的 formal 状态，也不改变 Countdown 不能替代 D-U1/C-U1 因果识别的边界。
<!-- HANDOFF-DELTA-BLOCK:section_end:v75-countdown-status-note:END -->

# 6. 接下来唯一执行顺序

1. E1/E2/E3、E4、E4-CONV 与 E5 的既有科学状态和历史证据保持不变；原 E4 18/20 门禁失败事实继续披露。
2. **`C-U1-E4-TAPER-01` 已完成正式运行与交付。** 当前科学状态是有限训练步数验证：主 paired mechanism-order claim 得到支持，但 2× 终态门禁未通过；不得自动延长或升级为 long-run validated。
3. `D-U1-E6-SEMANTIC-PILOT-01` 已完成并交付，但只提供 development 证据，不产生论文级方法排名，也不自动冻结 E6 正式参数。
4. `D-U1-E6-SEMANTIC-LONGRUN-01` 已完成 360/360 formal runs、2x 终态审计、raw evidence 和仓库闭环；禁止复用 held-out seeds 10--29 调参或无新登记重跑。
5. 下一步先为 `D-U1-E6-TAPER-01` 单独冻结 semantic remoteness coordinate、paired method protocol、新 untouched seeds，并实现 formal runner；用户审阅前不得运行，也不得把 Gaussian 标准化距离或二次临界界直接搬到 categorical。
6. E6-TAPER 交付后，解除已实现的 `EXT-H-E7-Q2`（E7-MECH）formal gate 并启动正式运行；它只回答 Hopper learned-critic 下二次 log-scale 远场区是否真实激活并传导。
7. E7-MECH 交付后，实施 `EXT-H-E7-BENCH-01`（E7-BENCH）：D4RL MuJoCo locomotion 9 tasks，方法 shortlist/超参从受控实验冻结，不做按任务方法族重选。
8. E7-BENCH 交付后，运行 `EXT-C-E8-V4.2`（E8-MECH）真实 Qwen pilot；代码/CPU smoke 不构成实验结果。
9. E8-MECH 交付后，冻结更大固定 Countdown 数据、3B 主模型与 7B 确认协议，实施 `EXT-C-E8-SCALE-01`。
10. SBRC/Hybrid 和 entropy/target-entropy controls 仍是后续安全层与排他性消融；未另行冻结前，不插入 Linear/Quadratic/Exp 核心顺序比较，也不预设优胜。

任何新增实验必须先说明它补哪一个 claim、是否替代现有实验、是否进入本文档。

---

<!-- HANDOFF-DELTA-BLOCK:section_end:v52-execution-order-override:START -->
11. **v52 执行覆盖：** 当锁定路线进入 E8-MECH 时，执行 `EXT-C-E8-V4.3` 而不是 V4.2；当前只完成注册和代码实现，真实 Qwen/CUDA pilot 仍为 not_run。
<!-- HANDOFF-DELTA-BLOCK:section_end:v52-execution-order-override:END -->
<!-- HANDOFF-DELTA-BLOCK:section_end:v55-du1-e6-semantic-gap-execution-order:START -->
12. **v55 执行覆盖：** Semantic-Gap 正式结果已闭环，不再等待该 successor 的 delivery。下一项仍不是直接运行 E6-TAPER，而是先冻结其 semantic remoteness coordinate、paired method protocol、全新 untouched held-out seeds，并实现独立 formal runner；完成用户审阅和 registry activation 前禁止启动。
<!-- HANDOFF-DELTA-BLOCK:section_end:v55-du1-e6-semantic-gap-execution-order:END -->
<!-- HANDOFF-DELTA-BLOCK:section_end:v56-e6-parent-closure-execution-order:START -->
13. **v56 执行覆盖：** E6 父 claim 已关闭，`D-U1-E6-TAPER-01` 改为可选非门禁 future study；当前直接进入已实现且 registry 为 ready/active 的 `EXT-H-E7-Q2`（E7-MECH）。E7-Q2 仍为 not_run，必须先完成正式运行、终态审计、打包与交付；其后才允许冻结并实施 `EXT-H-E7-BENCH-01`。E8-MECH/V4.3 与 E8-SCALE 的相对顺序不变。
<!-- HANDOFF-DELTA-BLOCK:section_end:v56-e6-parent-closure-execution-order:END -->
<!-- HANDOFF-DELTA-BLOCK:section_end:v57-e8-offline-bank-execution-order:START -->
14. **v57 执行覆盖：** v56 的 formal 顺序不变，`EXT-H-E7-Q2` 仍是下一正式实验。用户批准的 V4.4 作为 single-seed focused pilot 可独立执行，但必须先完成自身 best/terminal audit 与结果交付，才允许讨论 online off-policy successor；不得一次性同时改变 negative-bank 密度和数据在线刷新机制。
<!-- HANDOFF-DELTA-BLOCK:section_end:v57-e8-offline-bank-execution-order:END -->
<!-- HANDOFF-DELTA-BLOCK:section_end:v59-e8-offline-tuning-execution-order:START -->
15. **v59 执行覆盖：** formal 顺序仍由 v56/v58 控制，E7-Q2 优先级不变。V4.5 可作为独立 pilot 执行，但必须复用并校验 V4.4 frozen inputs，按 Stage A alpha、Stage B lambda、untouched-seed confirmation 顺序完成；test 只能在 selection 冻结后运行，结果必须 best/terminal 与三类事件分报后再交付。
<!-- HANDOFF-DELTA-BLOCK:section_end:v59-e8-offline-tuning-execution-order:END -->
<!-- HANDOFF-DELTA-BLOCK:section_end:v60-e4-taper-local-execution-order:START -->
16. **v60 E4-TAPER 内部执行序列：** 本条只登记 TAPER track 的依赖顺序，不自动改写 v56--v59 的全局外部实验路线。选择推进该 track 时，必须按 `NEAR-RETENTION-01 -> BUDGET-MATCH-01 -> CONV-01 -> CONFIRM-01` 逐项完成 protocol freeze、实现、正式运行、终态审计、打包和交付；下一项不得在前一项交付前启动。四项目前均 blocked，第一步是另行冻结 Near-retention protocol，而不是直接运行或延长旧 E4-TAPER。
<!-- HANDOFF-DELTA-BLOCK:section_end:v60-e4-taper-local-execution-order:END -->
<!-- HANDOFF-DELTA-BLOCK:section_end:v61-e4-taper-near-retention-execution-order:START -->
17. **v61 E4-TAPER 内部执行覆盖：** `NEAR-RETENTION-01` 已从 blocked 迁移为 implemented + ready + active + not_run，允许作为当前 TAPER track 的下一项正式运行。运行必须使用 hardened guard、正式 seeds 90--109、development-only coefficient calibration 和每 5 seeds checkpoint index；raw-complete 后仍需终态审计、canonical packaging 与交付。`BUDGET-MATCH-01` 在该交付之前继续 blocked，Long-run 与 Confirmation 顺序不变。
<!-- HANDOFF-DELTA-BLOCK:section_end:v61-e4-taper-near-retention-execution-order:END -->
<!-- HANDOFF-DELTA-BLOCK:section_end:v62-countdown-online-offpolicy-execution-order:START -->
18. **v62 Countdown 执行覆盖：** formal 主顺序继续由 v56/v58/v61 控制；`EXT-H-E7-Q2` 优先级不变。V4.6 允许作为独立 guarded pilot 执行，顺序固定为 predecessor/input hash audit -> 四 cell paired training -> 全部训练结束后 test evaluation -> 2×2 paired effect/interaction -> terminal audit -> canonical artifact delivery。任何 online phase 都必须保留 collector manifest、round JSONL、fresh/stale mix 与实际 selected-bank diagnostics；smoke 或单 seed 不得称实验结果。
<!-- HANDOFF-DELTA-BLOCK:section_end:v62-countdown-online-offpolicy-execution-order:END -->
<!-- HANDOFF-DELTA-BLOCK:section_end:v63-e4-taper-closure-execution-order:START -->
18. **v63 E4-TAPER 内部执行覆盖：** `NEAR-RETENTION-01` 已完成正式矩阵并沉淀为有限训练步数验证；当前下一项是已冻结且已实现的 `BUDGET-MATCH-01`，正式 seeds 固定为 110--129，只允许按每一步 Adam 之前的 raw negative-gradient L2 norm 做 paired budget matching。`CONV-01` 与 `CONFIRM-01` 虽已完整登记输入输出契约、shortlist 规则和 untouched seeds，但继续 blocked；必须等待 Budget-Match 正式结果完成终态审计、打包、交付并冻结 shortlist 后，才允许实现和启动 Convergence，Confirmation 仍为最后一步。
<!-- HANDOFF-DELTA-BLOCK:section_end:v63-e4-taper-closure-execution-order:END -->
<!-- HANDOFF-DELTA-BLOCK:section_end:v66-e4-taper-budget-match-execution-order:START -->
19. **v66 E4-TAPER 内部执行覆盖：** `BUDGET-MATCH-01` 已完成正式矩阵、终态审计和闭环交付，禁止无新登记重跑。`CONV-01` 仍不允许启动：先用既定规则在独立更新中生成并校验 `FROZEN_CONVERGENCE_SHORTLIST.json`，再实现读取 run_003 exact actor + Adam optimizer state 的 continuation runner；只有这两项通过并另行 activation 后才能运行。`CONFIRM-01` 与 seeds `130--149` 继续保持最后一道防火墙。
<!-- HANDOFF-DELTA-BLOCK:section_end:v66-e4-taper-budget-match-execution-order:END -->
<!-- HANDOFF-DELTA-BLOCK:section_end:v70-du1-e6-cartesian-taper-execution-order:START -->
10. **v70 D-U1 successor 覆盖：** `D-U1-E6-CARTESIAN-TAPER-01` 作为一个联合 formal experiment 执行，顺序固定为 environment/preflight audit → E6-Cartesian mechanism methods → preregistered TAPER methods → paired aggregation → 2× terminal audit → hardened packaging/delivery。禁止先查看正式机制结果后修改 taper family、retention、seeds 或阈值；原 `D-U1-E6-TAPER-01` 不再作为独立 runnable experiment。
<!-- HANDOFF-DELTA-BLOCK:section_end:v70-du1-e6-cartesian-taper-execution-order:END -->
<!-- HANDOFF-DELTA-BLOCK:section_end:v72-du1-e6-shared-rarity-repair-execution-order:START -->
11. **v72 D-U1 执行覆盖：**禁止直接运行 protocol revision 1 或未冻结的 protocol revision 2 formal。下一步固定为 protocol-revision-2 environment audit → development seeds `0--4` 的预登记校准 → 独立 formal-freeze 更新 → 才允许 guarded formal。校准前不得使用 held-out seeds `200--219`，不得根据 Countdown 或旧 pilot 预设 taper winner。
<!-- HANDOFF-DELTA-BLOCK:section_end:v72-du1-e6-shared-rarity-repair-execution-order:END -->
<!-- HANDOFF-DELTA-BLOCK:section_end:v73-e8-taper-execution-order:START -->
12. **v73 Countdown E8-TAPER 执行覆盖：**应用本更新并通过 trusted normalization 后，先执行 one-click preflight、reference gate、独立 replay/calibration freeze，再按相同 sampler plan 并行运行 6 methods × 3 paired seeds。全部训练完成前禁止读取 test；结果必须同时提交 best、terminal、final-window slopes、surprisal-bin allocation 和三类失败分报。
<!-- HANDOFF-DELTA-BLOCK:section_end:v73-e8-taper-execution-order:END -->
<!-- HANDOFF-DELTA-BLOCK:section_end:v74-du1-e6-rev4-execution-order:START -->
> **v74 唯一执行顺序覆盖：** 先应用并验证本 revision-4 freeze 更新；随后运行 `D-U1-E6-CARTESIAN-TAPER-01` 的 120 个 formal runs（6 methods × seeds `200--219`），每 5 seeds 形成持久 checkpoint；raw complete 后执行两窗口终态审计、三类崩溃分报加 environment-validity 分报、正式 artifact 打包与仓库结果登记。正式包交付前不得启动下一个依赖该结论的实验，也不得把 development pilot 宣称为正式排名。
<!-- HANDOFF-DELTA-BLOCK:section_end:v74-du1-e6-rev4-execution-order:END -->
<!-- HANDOFF-DELTA-BLOCK:section_end:v75-countdown-execution-order:START -->
13. **v75 Countdown full-bank diagnostic 后续门禁：**若需要把该外部机制诊断从 pilot 升级为正式证据，下一步不是调参，而是补跑多个独立 SFT/offline seeds 的同一 `probe_gradients` 导出并做 seed-level bootstrap。现有 single-seed 结果可用于论文候选图的 pilot 数值与口径设计，但禁止写成 multi-seed formal、方法排名或 scale-up 结论；E7 同口径逐样本统计仍需另行重导出。
<!-- HANDOFF-DELTA-BLOCK:section_end:v75-countdown-execution-order:END -->

# 7. 变量治理

## 7.1 原体系保留的核心变量

`s, a, pi_theta, theta, A, mu, sigma/Sigma, alpha, distance d, score function grad_theta log pi_theta`。

## 7.2 Exponential-family 核心定理所需新增符号

仅使用该理论无法避免的 `eta, T(a), psi(eta)`；首次出现时完整定义，并明确映射回 Gaussian 的 `mu/sigma` 与 categorical logits。

## 7.3 共同负对数概率

允许定义一次 `D_theta(s,a) = -log pi_theta(a|s)`；正文主要称“负对数概率”，括号注明 surprisal。它用于连续与离散的共同陈述，但不取代 Gaussian 距离变量。

## 7.4 撤出主体系的符号

已撤出的绘图归一化符号、`p/n/q` 简写、重复的 signed target、`kappa(D)`、与 discount factor 冲突的 `gamma` 等不进入主理论。局部证明需要时必须当场定义且不跨章节复用。

---

# 8. v4-v10 版本审计与恢复策略

| 版本 | 主要新增 | 是否保持累加 | 当前处理 |
|---|---|---|---|
| v4 | 锁定共识、两套环境职责、因果链、稳定外推计划、完整 related work、论文重构与文件索引 | 是 | 全部保留 |
| v5 | fixed-advantage 最小机制设定；明确未来统一环境是必做重构 | 是 | 全部保留 |
| v6 | sigma 方向修正；expected Fisher 纠错；均值—方差联合分析；原始代码诊断 | 是 | 全部保留并标记修正原论文 |
| v7 | 所谓统一 benchmark、三个 protocols、正式结果与代码索引 | 大体累加，但“统一环境”命名不准确 | 作为恢复底稿；相关结果降为旧分离环境开发证据 |
| v8 | 内容压缩、categorical 结果、方法筛选 | 否；发生破坏性删减 | 有效新增合入；删除动作不继承 |
| v9 | exponential-family 统一、自审、神经网络/critic 边界 | 在 v8 不完整底稿上新增 | 理论作为核心 patch 合入；过度完成声明撤回 |
| v10 | Hopper learned-critic probe | 独立报告 | 合入外部验证状态；明确 600-step 限制 |
| saturation audit | 部分稳定外推子实验长程复核 | 独立报告 | 合入收敛规范与修正结果；不得冒充全部实验审计 |

## 8.1 为什么 v8 删除是不合理的

当时将“压缩重复”错误实现为“删除历史环境、related work、实验 provenance 和多版计划”。这些内容对于论文定位、避免重复劳动和追踪结论来源都是必要的。正确方式应是：

- 正文只突出当前主结果；
- 历史内容移动到明确附录；
- 错误结论保留替代记录；
- related work 不得从研究主文档删除；
- performance/结果表必须保留来源、环境、训练步数和状态标签。

本恢复版以 v7 全量内容为底稿，后续补丁只增加或标记替代，不继承 v8 的破坏性删除。

---

# 9. 原审稿问题与新论文的硬约束

上一轮审稿的主要问题必须逐项关闭：

1. 杜绝 placeholder/hallucinated references，所有引用逐条核验；
2. 明确 fixed-advantage diagnostic 与完整 off-policy RL 的区别；
3. 不再宣称仅由 advantage sign 即可推出普遍发散；
4. Gaussian 特例必须由 exponential-family 核心定理和 categorical 分支支持，同时说明网络推广边界；
5. 自定义模拟器必须有完整公开代码、数据生成和训练细节；
6. 增加公共 D4RL/Countdown/推荐外部验证；
7. hard filtering 不再被包装成唯一生存条件；方法贡献转向受控 repulsive influence；
8. 所有动力学图必须覆盖达到稳态或明确 runaway 的全过程，而不是只展示早期数百步。

---

# 10. 当前新环境原型的审计结果

```json
{
  "negative_reward_equal_across_distance": true,
  "negative_advantage_equal_across_distance": true,
  "negative_distance_order_fraction": 1.0,
  "positive_advantage_fraction": 1.0,
  "negative_advantage_fraction": 1.0,
  "max_negative_reward_range_per_state": 2.38e-07,
  "max_negative_advantage_range_per_state": 2.38e-07,
  "mean_gap_to_unseen_optimum": 0.70,
  "mean_negative_offset": 0.50,
  "far_near_initial_distance_ratio": 2.22
}
```

该结果只证明环境几何满足预设，不代表 E1-E4 已完成。

---


# 11. v13 正式执行日志

## 11.1 C-U1 单 seed 回归（2026-06-24）

**状态：pilot / 回归通过。** 统一环境所有几何不变量通过。Positive-only 在测试状态上收敛到 `a_plus`：`归一化外推位移=-0.0001`、`mu_to_plus=0.0023`、`sigma=0.1904`。短程 signed update 随 alpha 增大出现更强方差收缩和反向漂移，方向与预登记理论一致；该扫描不作为正式 E3 结果。

代码与结果：`/mnt/data/drpo_experiments/c_u1_unified.py`；`/mnt/data/drpo_experiments/runs/c_u1_regression_seed0/`。

## 11.2 E1：统一环境中的瞬时梯度来源隔离（2026-06-24）

**状态：正式 20-seed 机制识别完成。** 使用 held-out seeds 10–29；每 seed 4096/4096 train/test states，并在 128 个 probe states 上计算逐样本全参数梯度。负 advantage 跨距离保持 `1.000000×`，每状态最大数值范围低于 `2.7e-7`。

| 阶段 | score far/near | 单样本全参数负梯度 far/near | 聚合负梯度 far/near |
|---|---:|---:|---:|
| 初始化 | 5.375 [5.290, 5.462] | 5.383 [5.296, 5.473] | 5.370 [5.220, 5.527] |
| Positive-only 收敛后 | 7.569 [7.562, 7.575] | 9.093 [9.012, 9.171] | 10.072 [9.959, 10.190] |

20/20 seeds 的远场单样本和聚合梯度比均大于 1。结论：在真正统一的 C-U1 中，badness 严格不变时，policy-relative remoteness 仍独立产生数量级更大的负梯度；主体来自 score geometry，聚合方向一致性使倍率进一步增强。该结果复现并迁移了历史 Product 环境的锁定结论，但不把 9×/10× 写成普适常数。

代码与结果：`/mnt/data/drpo_experiments/run_e1_formal.py`；`/mnt/data/drpo_experiments/runs/e1_formal/`。


## 11.3 E2：Positive-only 完整动力学与终态审计（2026-06-24）

**状态：已长期验证。** 20/20 held-out seeds 通过 `stable_plateau_2x_confirmed`。原始轨迹使用 2000-step minibatch Adam；随后对同一正样本目标做全数据 stationary-point 审计，并继续等长 2× 正样本训练。该终态审计不用于 E3/E4 初始化，只用于排除固定步数假平台。

| 指标 | 均值 | 95% bootstrap CI |
|---|---:|---:|
| 测试 reward | 0.646923 | [0.646898, 0.646946] |
| 沿隐藏最优方向的归一化外推位移 | 0.000033 | [-0.000011, 0.000074] |
| 到 `a_plus` 距离 | 0.001311 | [0.001272, 0.001351] |
| 学习到的 sigma | 0.190419 | [0.190407, 0.190431] |
| 终态 exact phantom negative gradient norm | 18.504 | [18.298, 18.718] |
| 终态 phantom far/near | 7.568 | [7.567, 7.568] |

解析条件残差给出的内部方差目标为 `sigma=0.190394`，实测与其一致且没有使用方差 clamp。固定负样本的标准化距离增长 **3.003×**，exact aggregate phantom negative gradient 增长 **28.927×**；与此同时 `归一化外推位移≈0`，策略均值停在 `a_plus` 而没有接近未见最优 `a_star`。

结论：Positive-only 稳定但存在明确 imitation ceiling；仅拟合正样本就会把固定负样本推入更深远场，并在尚未应用任何负更新时预先产生近 29× 的 phantom-gradient 放大。该结果排除了 critic/value 漂移作为这一前兆的必要条件。

代码与结果：`/mnt/data/drpo_experiments/c_u1_unified.py`；`/mnt/data/drpo_experiments/runs/e2_formal/`。

## 11.4 E3：C-U1 统一 Adam Near/Far 因果干预（2026-06-25，当前论文结果）

**状态：已长期验证。** Experiment ID 为 `C-U1-E3-ADAM-RERUN`；run commit 为 `ac286a46b8ffad898dfad0e7e9188b1d2e81052a`；正式 held-out seeds 为 30--49。环境使用 4096 train / 4096 test states、每状态 4 个正动作和 8 个等 advantage 负动作，E3 各方法从同 seed 的 2000-step positive-only Adam checkpoint 初始化。测试状态与训练状态同分布，只能称 held-out-context generalization。

### 固定方差：远场路径传导为任务性能崩溃

`sigma=0.1903943276, alpha=1.4, Adam lr=1e-4, 2000 steps`。

| 方法 | 终态 reward mean [95% CI] | 任务性能崩溃 | 支持边界 | NaN/Inf |
|---|---:|---:|---:|---:|
| Baseline | 0.000002 [0.000002, 0.000002] | 20/20 | 0/20 | 0/20 |
| Near-zero | 0.000002 [0.000002, 0.000002] | 20/20 | 0/20 | 0/20 |
| Far-zero | 0.739362 [0.738902, 0.739837] | 0/20 | 0/20 | 0/20 |
| Far-cap | 0.733072 [0.732594, 0.733563] | 0/20 | 0/20 | 0/20 |
| Global-scale | 0.599057 [0.598634, 0.599475] | 0/20 | 0/20 | 0/20 |
| Far-to-near | 0.875323 [0.874994, 0.875616] | 0/20 | 0/20 | 0/20 |

主文最短因果链只使用 Baseline、Near-zero、Far-zero、Far-cap：Baseline 崩溃；删除近场而保留远场仍崩溃；删除或截断远场则 20/20 救援。Global-scale 与 Far-to-near 保留为附录控制，不从本环境推出跨任务方法排名。

### 可学习方差：远场路径提前触发支持收缩

`alpha=0.15, Adam lr=5e-4, max 2000 steps, no variance clamp`。边界审计覆盖完整 4096 个训练状态，并以第一次负向越界作为科学事件。

| 方法 | 首个事件或终态 reward mean [95% CI] | support contraction | onset median | unexpected expansion | NaN/Inf |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.603254 [0.593897, 0.612491] | 20/20 | 73 | 0/20 | 0/20 |
| Near-zero | 0.601992 [0.592785, 0.611119] | 20/20 | 73 | 0/20 | 0/20 |
| Far-zero | 0.652887 [0.651945, 0.653841] | 0/20 | — | 0/20 | 0/20 |
| Far-cap | 0.652625 [0.651661, 0.653619] | 0/20 | — | 0/20 | 0/20 |
| Global-scale | 0.642867 [0.641859, 0.643927] | 0/20 | — | 0/20 | 0/20 |

该分支的第一事件是支持/方差收缩，不是“方差爆炸”，也不是 NaN/Inf。Adam 下未复现旧 plain-SGD 的正向一步过冲：100 个 learnable method-seed runs 中 unexpected expansion 为 0；固定与可学习方差合计 220 个 method-seed runs 中 NaN/Inf 为 0。

### 因果结论、论文位置与边界

E3 在同一个 C-U1 环境中闭合两条互补路径：固定方差下，远场异常负影响传导为均值漂移和任务性能崩溃；可学习方差下，它更早传导为支持收缩。Near-zero 在两条路径上均不能救援，Far-zero 与 Far-cap 均能救援。因此该结果可以进入论文的受控因果实验：fixed-variance 四方法对照进主文，learnable-variance 进互补 panel 或附录。

该实验不回答显式状态分布偏移，不得称 OOD；不证明所有真实任务都由该机制崩溃；不证明 Distance、Global-scale 或 Far-to-near 跨任务必然更优。任务性能崩溃、support/variance boundary 与 NaN/Inf 必须继续分开报告。

结果索引：`outputs/cu1_e3_adam/RESULT_SUMMARY.md`、`fixed_variance_aggregate.csv`、`learnable_variance_aggregate.csv`、`TERMINAL_AUDIT.md`、`ARTIFACT_INDEX.json`。完整 raw trajectories 与 checkpoints 位于已交付 artifact `DRPO_CU1_E3_ADAM_AC286A4_FINAL.zip`，SHA-256 为 `2b8bfdbe6f33ed1db9dc1e59f6e9fbdb6c224c7b31b1326a7f2fbaeeaaaf522b`。

### 来源与聚合审计说明

Scientific run 使用 exact committed runner blob，runner SHA-256 为 `502c345289d2b5b7c34832246478b64c33a1789e80ddcab7f6194cb09b0eac6f`。启动环境因 shell DNS 无法访问 GitHub，没有本地 Git object；该来源限制保留在 artifact provenance 中。最终聚合遇到 tuple 经 JSON 序列化为 list 后的 resume 比较问题，仅使用表示归一化 workaround 完成汇总；没有改变 seed、配置、优化器、梯度、阈值、轨迹或数值结果。

## 11.4 E3：C-U1 长期 Near/Far 因果干预（2026-06-24，历史 SGD 证据）

> **v29 历史覆盖说明（已由 v31 替代）：** 本节数值与 SGD 配置仅保留作历史 provenance。v29 当时将 `C-U1-E3-ADAM-RERUN` 记为“尚未运行”；当前状态以本节之前的 v31 Adam 结果为准。旧 SGD 结果不得覆盖或混入当前论文表。

**历史状态：当时登记为正式 20-seed 因果实验完成。** 为避免开发 seed 泄漏，正式使用 held-out seeds 30–49。固定方差与可学习方差分成两个互补分支，严格区分“任务效果崩溃但数值仍有限”和“任务阈值前先发生精度/支持坍缩”。

### 固定方差任务崩溃分支

`sigma=0.190394, alpha=1.4, lr=1e-4, 2000 steps`。Baseline 最终 reward `0.1591 [0.1539,0.1645]`，18/20 达到任务崩溃阈值、其余 2 个仍持续低 reward 漂移；Near-zero 最终 `0.1743 [0.1690,0.1797]`，13/20 达阈值、其余 7 个仍持续低 reward 漂移。两者均无 NaN/Inf。Far-zero、Far-cap、Global-scale 为 0/20 任务崩溃，最终 reward 分别为 `0.6934`、`0.6469`、`0.6469`。Far-zero/Far-cap/Global 分别在 20/20 配对 seeds 中胜过 Baseline。

Far-to-near 将被截断的远场预算转移给方向可靠的近场负梯度，最终 reward `0.8095 [0.8073,0.8115]`，20/20 胜过 Baseline。这一结果说明巨大负更新并非单独充分致害；方向可靠的负梯度可以有益，危险来自远场异常影响与低/错误方向效用的结合。

### 可学习方差精度坍缩分支

`alpha=0.15, lr=5e-4`。Baseline 与 Near-zero 均在 20/20 seeds 中先触发 log-sigma 数值边界，中位 onset 分别为 140 与 144 steps；当时 reward 尚未跌破任务崩溃阈值，因此正式标记为 `numerical_collapse_before_task_threshold`，不能写成任务与数值同时崩溃。Far-zero、Far-cap、Global-scale 均为 20/20 stable/bounded、0/20 数值崩溃。

**因果结论：** 删除近场不能切断固定方差的长期任务漂移，也不能阻止可学习方差的早期支持坍缩；删除或截断远场则同时救援两条分支。全局缩放同样稳定，进一步支持异常负梯度幅度是直接中介；Distance 与 global 的泛化优劣交由 E4。

代码与结果：`/mnt/data/drpo_experiments/runs/E3_RESULTS.md`；`/mnt/data/drpo_experiments/runs/e3_formal_fixed/`；`/mnt/data/drpo_experiments/runs/e3_formal_learn/`。


## 11.4 E3 历史 SGD 协议冻结补充：均值任务崩溃与可学习方差提前失稳

开发 seeds 0–4 显示，在统一 C-U1 的可学习方差 actor 中，远场负样本的二次 log-variance score 会先于明显 reward 失效把 `log_sigma` 推过数值边界。这与第 12.6 节“方差边界早于均值临界点”的理论一致，但单独不能完成“任务效果崩溃”的识别。因此 E3 正式报告分为两个互补分支，环境、数据与 Near/Far 划分完全相同：

1. **Fixed-variance causal branch（主任务崩溃识别）：** 将 sigma 固定在 E2 的解析正样本稳态 `0.190394`，冻结 `alpha=1.4`、SGD learning rate `1e-4`、2000 steps；1000-step 为首次分类点，继续至 2×=2000 steps 检查结论不反转。开发 seeds 0–4 中 Baseline/Near-zero 产生数值有限的任务崩溃，而 Far-zero/Far-cap/Global/Far-to-near 保持稳定或有界。该分支隔离远场对均值与 task reward 的因果传导。
2. **Learnable-variance branch（提前精度/支持失稳）：** 冻结 `alpha=0.15`、SGD learning rate `5e-4`、2000-step 上限；稳定方法以 1000→2000 的 2× 延长检查，失稳方法记录精确数值边界步。开发 seeds 0–4 中 Baseline/Near-zero 均先触发方差数值边界，Far-zero/Far-cap/Global 稳定。该分支验证远场负梯度对可学习方差的更早失稳路径。

正式 E3 使用严格未查看 seeds 30–49；seed 10 仅保留为实现 smoke，不进入正式统计。Fixed 与 learnable 两分支不是两个环境，也不混淆历史 Product/Collapse 实验；它们只改变 Gaussian variance 是否作为可训练参数。



## 11.7 C-U1 论文成熟度与剩余漏洞审计（2026-06-24）

### 当前判断

- E1：机制识别成熟度高；主文可用，但需在重跑包中补 full-parameter gradient 的 Jacobian/direction 分解或 same-ray control，避免把全参数倍率全部归因于距离。
- E2：科学成熟度高；可入正文或附录。它证明 positive-only 平台、有限方差稳态和 phantom-gradient 前兆；仍需恢复逐步分解曲线，区分均值远离与 sigma 收缩各自贡献。
- E3 fixed variance：**已长期验证，主文可用。** 20-seed 统一 Adam 因果链完整：Baseline/Near-zero 20/20 任务崩溃，Far-zero/Far-cap 0/20；动态轨迹、阈值判据与终态审计已进入交付 artifact。
- E3 learnable variance：**已长期验证，适合作为主文互补 panel 或附录。** Baseline/Near-zero 20/20 首先发生支持收缩，远场控制 0/20；完整轨迹、全状态边界审计和无 clamp/无 NaN/Inf 结果已保留。其职责是支持收缩机制，不替代 fixed-variance 的任务崩溃识别。
- E4 fixed variance：解析—实验闭环成熟度高；主文核心结果候选。图表需拆开有限固定点和 runaway。
- E4 learnable variance：机制证据中高；可入附录或主文补充。性能曲线与稳定边界必须分开，不允许用未稳态截面 reward 排名。
- 方向诊断：仅作 sanity check/附录，不构成跨环境主结论。

### 需要优先补强的漏洞

1. 当前 train/test state 来自同一分布，只能称 held-out context generalization，不应直接称 OOD；若论文要写 OOD，需另加分布偏移测试。
2. E1 的等 reward 轮廓改变动作方向；输出层 Gaussian score norm 只依赖距离，但全参数梯度还受网络 Jacobian 方向影响。需补 same-ray radial probe 或 Jacobian gain 分解。
3. 可学习方差分支必须对 log-standard-deviation 下界、参数化方式和精度做敏感性检查，排除人为 floor 决定 onset。
4. 动态 standardized-distance near/far 划分会受 sigma 收缩影响；必须同时报告 raw distance、standardized distance 和 near/far 样本占比随时间变化。
5. Far-cap 相对等预算 global 的优势依赖本环境的方向构造；只能作为 C-U1 方法识别，外部一般性由 Hopper/其他任务验证。
6. 统一源码与 raw trajectories 未进入当前持久化包，是目前最大的投稿级工程缺口。
# Part II. v7 全量累积研究记录（原文保留）

> **v15 术语覆盖提示：** 本 Part 为不可破坏的历史原文。其中凡把 C-U1/旧同分布测试直接称为 OOD 的句子，均为历史措辞，已由第 0.2 节正式替代；不得直接复制进新论文。真正的 OOD claim 只允许来自未来显式 distribution-shift protocol。


> 以下为 v7 全文转换内容，作为 v4-v7 累加研究历史。保留是为了避免任何 locked conclusion、related work、旧实验、路线与文件索引丢失。其“统一 benchmark”措辞已在 Part I 中标记为需要修正；具体内容不直接删除。

机制分解、因果干预、DRPO 理论支撑与贡献定位

━━━━━━━━━━━━━━━━━━━━━━━━

内部研究文档｜2026 年 6 月 23 日｜v7 统一非线性 benchmark 与论文级 P0
结果

**核心结论**

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th><p><strong>结论锁定</strong></p>
<p>在负优势幅度与策略距离严格解耦的环境中，远场负样本仍产生数十倍更大的全参数负梯度；基线训练中该异常梯度先于策略径向漂移与性能崩溃出现。删除近场负梯度几乎不改变崩溃，而删除或截断当前远场负梯度可在
20/20 个 held-out seeds 中阻止崩溃。因此，在本受控环境内，“远场几何 →
异常负梯度 → OOD 漂移 → 性能崩溃”已经形成较完整的因果闭环。</p></th>
</tr>
</thead>
<tbody>
</tbody>
</table>

*用途：论文理论补强、实验章节设计、novelty 边界与投稿叙事*

本次增补（v7 / 统一 benchmark 交接版）：在 v6
联合均值—方差修订基础上，完成一个可配置的 state-conditioned Gaussian
policy、统一训练/诊断代码栈与三个严格区分的 identification
protocols：来源隔离、collapse 因果干预、稳定外推。正式结果覆盖来源实验
20 个 held-out seeds、因果干预完整统一代码重跑 20 seeds、可学习方差外推
20 seeds、固定方差支持实验 10 seeds，并加入等预算 global
control、distance
cap、架构附录稳健性、原始曲线、置信区间、配对检验、矢量图和可复现代码。v4-v6
的已锁定结论与方差理论修正继续有效。

# 文档结构

- 10\. 从推荐扩展到通用 off-policy policy optimization

- 11\. 连续—离散统一的 Repulsive Surprisal Dynamics

- 12\. 为什么负梯度并不总是有害：局部泛化与远场失稳

- 13\. WAPO、STARE、Mu-GRPO、ASymPO、TOPR 等工作的机制对照

- 14\. 下一阶段理论与实验增强计划

- 1\. 执行摘要与最终判断

- 2\. 理论命题与需要闭合的因果链

## 2.1 机制分析的 fixed-advantage naive-PG 假设

本研究首先分析固定离线样本及固定 advantage label 下的最小
policy-gradient 动力学。令 D={(sᵢ,aᵢ,Aᵢ)} 为静态数据集，Aᵢ
在训练前计算并冻结；训练不包含 value/Q network、importance sampling、PPO
ratio/clipping 或其他 θ-dependent reweighting。于是更新为：

gᵢ(θ) = Aᵢ ∇θ log πθ(aᵢ \| sᵢ), ∂Aᵢ/∂θ = 0.

该设定不是为了逼近所有 offline RL
细节，而是为了识别一个最小充分机制：固定的样本质量信号与动态变化的
policy-score geometry 相乘，即可产生远场放大和 repeated
repulsion。若在此最干净设定中已出现 drift/collapse，则 critic error
或动态 advantage 不是该机制成立的必要条件。

论文主文只需明确这一研究范围；更一般的 gᵢ=wᵢ(θ)Aᵢ(θ)∇logπθ
形式可在附录用一小段说明其会移动临界点或改变增长率，但不作为当前理论和实验的主线，也无需在
P0 中新增动态 critic 实验。

- 3\. 实验一：badness–distance 严格解耦下的梯度来源分解

- 4\. 实验二：非线性 Gaussian actor 中的因果干预

- 5\. 证据强度、结论边界与仍未证明的内容

- 6\. 对 DRPO 论文贡献的含义

- 7\. 与相关工作的关系及 novelty 定位

- 8\. 建议写入论文的核心 claims、图表与段落结构

- 9\. 旧阶段路线回顾与更新后的执行顺序

- 附录：数值结果、实现验证与稳健性检查

# 0. 新会话交接：必须继承的共识与阅读顺序

本节是新会话的最小可靠上下文。后续讨论应先继承本节，再继续设计实验；不要把方法实现细节、探索性消融和已经锁定的机制结论混为一谈。

## 0.1 两套环境必须严格区分

环境 A（乘积流形机制环境）只回答“大梯度从哪里来”。质量/advantage
只依赖角度 θ，距离只依赖半径 r；同一 advantage
向量沿所有半径精确复制，因此 badness 与 distance
是结构独立，而非仅相关系数接近 0。该环境得到初始化 16×、训练后单样本
24.95×、聚合 29.13× 的远/近负梯度倍率。

环境 B（非线性 Gaussian
因果环境）只回答“远场大梯度是否传导成崩溃”。它使用共享非线性 MLP、二维
Gaussian mean、可学习方差和严格笛卡尔积负样本；开发 seeds 0–4 固定
α=0.1、距离阈值 d=2，正式检验使用 held-out seeds 10–29。该环境得到全参数
far/near 中位比 56.62×，并通过 Near-zero、Far-zero、Far-cap
等定点干预闭合因果链。

## 0.2 已锁定、未锁定与后续可变结论

| **层级** | **结论**                                                                                                       | **状态**                                    |
|----------|----------------------------------------------------------------------------------------------------------------|---------------------------------------------|
| 已锁定   | 在 badness 与 distance 严格解耦时，远场仍产生数量级更大的负梯度；主体来自 score geometry，而不是远场样本更差。 | 不得因后续方法实验而推翻                    |
| 已锁定   | 在当前非线性 Gaussian 环境中，远场异常负梯度是 OOD 漂移与 collapse 的主要自然传导路径。                        | Near-zero 无效；Far-zero/Far-cap 20/20 救援 |
| 高度支持 | 负梯度是直接中介，远场几何是异常幅度的自然生成机制；全局 α 与距离控制都可稳定。                                | 方法优劣仍待泛化任务决定                    |
| 尚未锁定 | 随训练步数严格指数增长、Distance 必然优于 α、所有真实任务仅由该机制崩溃。                                      | 必须继续验证，不能提前宣称                  |

## 0.3 三个容易误判、但不应再反复争论的问题

- Positive-only：它删除全部负向 repulsion，近似 advantage-filtered
  imitation；不发生远场负梯度 runaway
  是预期结果。它不是“远场理论的反例”，真正待检验的是其 imitation ceiling
  与受控负梯度的泛化收益。

- Detach：是否对 distance weight stop-gradient
  决定了“纯重权”还是“可微距离正则”的方法定义；它不影响环境 A
  中大梯度来源的结论，也不应被用来否定远场动力学。

- Shuffled
  distance：早期探索性打乱会同时改变实际梯度预算和方向，不是干净反证；正式因果证据应以动态
  Near/Far 定点干预和严格预算匹配为准。

## 0.4 当前论文主线与执行优先级

论文主线应回答三个问题：为什么负优势有用（突破 positive-only 的
imitation
ceiling）；为什么同一负优势会在远场变得有害；如何用距离/Surprisal-aware
控制保留局部泛化并阻断 runaway。

| **优先级** | **工作**                          | **验收目标**                                                                           |
|------------|-----------------------------------|----------------------------------------------------------------------------------------|
| P0（第一） | 负梯度稳定外推与泛化实验          | 直接证明受控近场负梯度把策略越过最佳正样本支持并改善 OOD；扫描到相变，再由距离控制恢复 |
| P0-并行    | 一维闭式理论与通用 surprisal 定理 | 给出稳定固定点、临界条件、二阶余项与连续动力学                                         |
| P1（第二） | Categorical bandit 严格隔离       | 把连续距离与离散 surprisal 统一，并复刻 rare/common 因果干预                           |
| P2         | 小型 Transformer 与真实 RLVR      | 验证共享参数、token interference、staleness 和 support suppression                     |
| P3         | D4RL / 推荐 / 机器人外部验证      | 验证真实数据中 badness-distance 耦合及跨任务方法收益                                   |

## 0.5 方法族的稳定解释

统一 α/SNA2C 是全局控制：整体降低负向推力；Distance/Surprisal cap
是选择性控制：优先抑制远场或 rare-negative；Advantage-only taper 控制
severity；Joint influence 则依据 \|A\|×score risk 同时控制 severity 与
geometry。在 D4RL 等现实数据中，差样本与远样本通常耦合，因此 advantage
weighting 也可能间接抑制远场，这不反驳距离机制。

方法路线：DRPO 的 hard filtering
可视为把危险远场权重直接置零的极限形式；下一阶段更值得验证的是 soft
distance decline / cap，目标不是删除全部负优势，而是在保留近场 boundary
shaping 与泛化推力的同时，让远场 influence 平滑衰减。

## 0.6 P0 的固定 advantage 假设与未来环境统一（必须继承）

- **P0 主研究对象。**采用静态离线数据上的 naive policy gradient，不引入
  value/Q network，不使用 importance sampling、PPO clipping 或其他随
  learner policy 变化的样本权重。

- **固定 advantage。**每个样本的 advantage label Aᵢ 在训练前由固定
  reward 与固定 baseline 得到，随后冻结，满足
  ∂Aᵢ/∂θ=0。训练过程中唯一随策略变化的核心量是 score ∇θ log
  πθ(aᵢ\|sᵢ)，因此可将 observed dynamics 归因于 policy-score geometry 与
  repeated off-policy update，而不是 critic/value feedback。

- **原 DRPO 论文核对。**原稿第 2.2 节已采用静态数据集
  D={(sᵢ,aᵢ,Rᵢ)}，定义 Â(s,a)=R(s,a)−b(s)；第 3 节单样本动力学又把 Â
  写为固定常数 Cbase，APG 实现也明确省略 importance-sampling
  correction。因而固定 advantage
  是原理论的隐含工作假设，但原稿没有清楚声明 baseline/advantage
  在机制分析期间冻结，修订稿必须显式补上。

- **结论边界。**固定 advantage 不是对所有 RL
  算法的普遍断言，而是机制隔离的有意设定：它证明即使不存在 value/Q
  估计漂移，仅靠固定负信号与动态 score geometry
  也足以产生稳定外推、过度外推与 runaway。动态 critic、importance ratio
  和自适应权重仅作为扩展讨论，不进入当前 P0 主实验。

- **未来环境统一是必做项。**当前环境 A、B 及新建 P0 环境 C
  继续分别承担来源识别、collapse
  因果识别和稳定外推识别；在论文定稿前，应统一为“一个解析模型 +
  一个可配置非线性
  benchmark”的总框架，并把三项任务写成同一母环境下的三个
  protocol。统一工作不得反向混淆或重写 A、B 已锁定结论。

## 0.7 Gaussian 方差动力学修正与联合稳态（必须继承）

- 完整稳态。可学习 Gaussian policy 的稳态必须同时满足 μ̇=0 与 ξ̇=0（ξ=log
  σ）；仅均值停止而 σ 继续变化，不是完整策略稳态。

- 条件残差。Positive-only
  是否具有有限方差稳态，不取决于全局正样本是否分散，而取决于网络拟合状态后仍剩余的条件残差
  Var(a\|s)。若该残差为正，则 σ
  可稳定在有限值；若每个状态对应唯一动作且网络可完全插值，则 Gaussian
  MLE 仍推动 σ→0。

- 远场符号。负优势始终把均值推离该动作，但对方差的作用由标准化距离
  z=(a−μ)/σ 决定：\|z\|\<1 时负更新扩大 σ/entropy；\|z\|\>1 时负更新收缩
  σ/entropy。远场危险链是“均值排斥 + 支持收缩”，而不是 μ 与 σ 同向扩张。

- 原始代码复核。gradient-explode 的 good-only 实验记录的是对
  pre-normalized mean logits 与 log σ 的梯度 norm，而不是实际 μ、σ
  参数值；实际 σ 在 good-only 下从约 0.606 收缩到约
  0.177，在正负混合更新下进一步逼近数值下限。Figure 2(b)
  只能支持梯度敏感度增长，不能支持“Both μ and σ expand”。

- 矩阵理论修正。expected Fisher/expected Hessian
  可描述策略信息几何，但不能直接充当固定 off-policy signed-gradient
  动力学的稳定矩阵。通用对象应改为 F(θ)=E_D\[A∇θlogπθ\] 及其 Jacobian
  J_F(θ\*)=∇θF(θ\*)；局部稳定由 Re λ_i(J_F)\<0 或离散情形 ρ(I+ηJ_F)\<1
  判定。

- 结论边界。原稿“advantage 符号单独决定联合参数所有方向
  expansion/contraction”的定理需要实质性替换；固定方差均值排斥、远场
  score 放大、positive contraction 引起的 OOD fragility，以及环境 A/B
  的已锁定因果结论不因此反转。

## 0.8 统一 benchmark 的新锁定结论（必须继承）

- 统一不等于混淆。三个 protocol 使用同一 Gaussian
  actor、训练循环与诊断工具，但来源隔离仍只回答“大梯度从哪里来”，因果干预仍只回答“远场异常梯度是否传导为
  collapse”，稳定外推只回答“负梯度何时有益、何时过量或失稳”。

- 来源隔离正式结果。20 个 held-out seeds 中，\|A\| 的 far/near 比严格为
  1.000；初始化时 score、单样本负梯度与聚合负梯度 far/near 比分别为
  45.13、47.78、61.56；正样本预训练后分别为
  38.02、38.64、82.08。统一非线性 actor 独立复现了 score geometry
  来源结论。

- 因果干预完整统一代码重跑。Baseline 19/20 崩溃、Near-zero
  18/20；Far-zero、Far-cap、Global-scale 均为 0/20。Far-zero/Far-cap 在
  20/20 配对 seeds 中胜过 Baseline，Near-zero 与 Baseline
  无显著差异。该结果精确复现已锁定因果结论。

- 稳定外推。Positive-only 停在最佳正样本支持附近，held-out reward 约
  0.085；固定方差 α=0.5 得到 β=0.897、reward=0.837；可学习方差 α=0.5
  得到 β=0.782、reward=0.709。更强负推力依次出现过度外推、慢漂移和
  collapse。

- 联合均值—方差相变。可学习方差在 α≤0.5 时 20/20 稳定，α=0.65
  进入混合慢漂移区，α=0.68 时 16/20 方差坍缩，α≥0.70 时 20/20
  方差坍缩；该边界显著早于固定方差均值临界点 α≈1。

- 控制方法。原始 α=0.9 时不受控 reward≈0；global scale、等预算 global 与
  distance cap 分别恢复至 0.719、0.725、0.747。Distance cap 相对等预算
  global 的配对增益为 +0.021，95% CI \[0.019,0.023\]，20/20
  胜出；这只是在本 benchmark
  中支持选择性距离控制，不能升级为跨任务必然优于全局 α。

- 论文级代码与材料已生成：完整 zip 包含 raw curves、per-seed
  tables、bootstrap CI、Wilcoxon paired tests、PNG/PDF 图、LaTeX
  tables、manifest、unit tests 及一键正式重跑入口。

# 1. 执行摘要与最终判断

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th><p><strong>一句话判断</strong></p>
<p>今天的实验显著增强了 DRPO
的理论可信度和论文贡献，但它强化的是“远场负样本通过 policy-score
geometry 自然生成异常大的排斥梯度，并成为 off-policy
崩溃的主传导路径”这一机制；它不能自动证明论文中的每一个更强命题，例如严格的时间指数律、DRO
形式的唯一最优性或所有真实任务崩溃都只有这一原因。</p></th>
</tr>
</thead>
<tbody>
</tbody>
</table>

综合判断：DRPO 仍然具有明确且可防守的
contribution。今天的结果不是简单增加一个 toy
曲线，而是补上了原论文最容易受到质疑的一环：负优势与距离在真实数据里通常耦合，过去很难判断大梯度究竟来自“样本更差”还是“样本更远”；现在通过严格乘积构造与定点干预，可以将两者拆开。

论文可以“大书特书”，但应大写正确的部分：机制识别、因果闭环、方法含义与现实耦合解释，而不是写成“此前没有人研究负优势”“任何
off-policy collapse 都由远场唯一导致”或“纯 Distance 必然优于所有 α
方法”。

| **判断对象**                                | **当前结论**                                        | **可信度** |
|---------------------------------------------|-----------------------------------------------------|------------|
| 远场负样本会产生异常大的负梯度              | 在严格解耦环境中已直接证明                          | 很高       |
| 大梯度主要来自距离而非更差的 advantage      | 乘积流形中 16×→24.95×；非线性 actor 中中位数 56.62× | 很高       |
| 远场大梯度是当前受控环境崩溃的主传导路径    | 近场删除无效；远场删除/截断 20/20 救活              | 高         |
| 统一 α 能否达到与 Distance 相同或更好的效果 | 当前 global-scale 可稳定，但跨任务优劣未定          | 未定       |
| 负梯度随训练步数严格指数增长                | 观察到自增强，尚需时间律拟合与解析递推              | 中等       |
| 所有真实 offline RL 崩溃都由该机制导致      | 尚无依据作普遍唯一性断言                            | 不可声称   |

# 2. 理论命题与需要闭合的因果链

单个负优势样本的 policy-gradient 项可写为：

gᵢ⁻ = Aᵢ ∇θ log πθ(aᵢ \| sᵢ), Aᵢ \< 0

其范数由两个因子共同决定：

‖gᵢ⁻‖ = \|Aᵢ\| · ‖∇θ log πθ(aᵢ \| sᵢ)‖

第一项是样本有多差（advantage severity）；第二项是当前策略对该动作的
score geometry。对于 Gaussian 均值参数，score 含有 (a−μ)/σ²；对于
log-variance
参数，远场部分近似含有标准化距离的平方。因此，距离远本身就可能放大负梯度，且反向排斥会进一步拉大距离。

远场 → score norm 放大 → 负向排斥增强 → 策略继续远离 → 更强 score

*该闭环是 DRPO“repulsive optimization”故事的核心动力学。*

要让理论从“合理解释”升级为“强因果证据”，必须分别证明：

1.  来源：在 \|A\| 与距离解耦后，远场负梯度仍显著更大。

2.  时间顺序：异常远场梯度先于策略漂移与 reward collapse。

3.  必要传导路径：保留远场而删除近场时仍崩溃。

4.  定点救援：只删除或截断远场异常梯度即可阻止崩溃。

5.  中介变量：统一缩放负梯度也能阻止崩溃，说明“异常梯度幅度”是直接传导量。

## 2.2 从 expected Fisher 改为 signed off-policy 动力场 Jacobian

为了保留原理论的 general matrix 形式，同时避免把 on-policy expected
Fisher 与固定离线样本动力学混淆，定义：

F(θ) = E\_(s,a)~D \[ A(s,a) ∇θ log πθ(a\|s) \], θ̇ = F(θ).

联合固定点 θ\* 满足 F(θ\*)=0。在其邻域令 δθ=θ−θ\*，则：

δθ̇ = J_F(θ\*) δθ, J_F(θ\*) = ∂F/∂θ \|\_(θ\*).

在 fixed-advantage 设定下，J_F(θ\*) = E_D\[A(s,a) ∇²θ log
πθ\*(a\|s)\]。连续时间局部稳定要求所有特征值实部为负；离散更新要求
ρ(I+ηJ_F)\<1。

Fisher I(θ)=E\_(a~πθ)\[∇logπ∇logπᵀ\] 仍用于描述 score
geometry、自然梯度与远场敏感度，但它不包含离线数据分布、advantage
符号和正负样本质量，因此不能单独判定实际训练稳定性。

# 3. 实验一：badness–distance 严格解耦下的梯度来源分解

## 3.1 乘积流形构造

机制环境采用乘积空间 \[0,R\] × S¹。样本质量和 advantage 只依赖角度
θ，径向距离只依赖 r；同一套 θ/advantage
向量被原样复制到所有半径上。因此这不是“Pearson 相关接近
0”，而是数据生成上的结构独立：

p(\|A\| \| r = r₁) = p(\|A\| \| r = r₂), ∀ r₁,r₂

在负样本子集内同样成立，因为负样本 mask 只由 θ 决定。训练改变 policy
score，但不改变 reward、baseline 或 advantage 的跨半径分布。

## 3.2 梯度分解结果

| **阶段**     | **\|A\| 远/近** | **score norm 远/近** | **单样本负梯度远/近** | **聚合负梯度远/近** |
|--------------|-----------------|----------------------|-----------------------|---------------------|
| 初始化       | 1.000×          | 16.000×              | 16.000×               | 16.000×             |
| 正优势训练后 | 1.000×          | 24.047×              | 24.950×               | 29.129×             |

训练后单样本比例可写为 24.950 = 24.047 × 1.038：主体是径向 score
scale，剩余是 score 与角度质量维度的轻微交互；不是远场 advantage
更差。聚合比例进一步乘以约 1.167 的方向一致性因子，达到 29.129×。

<img src="/mnt/data/master_recovery/media/media/image1.png"
style="width:6.45in;height:1.81406in" />

**图 1　乘积流形中的负梯度因子分解：advantage 分布保持不变，score
geometry 随距离放大。**

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th><p><strong>实验一锁定的结论</strong></p>
<p>“远场负梯度更大”不是因为远场样本平均更差。只要 policy score
的范数随策略相对距离增长，距离就构成独立的梯度放大因子；advantage
severity 是另一项可与其相乘的风险。</p></th>
</tr>
</thead>
<tbody>
</tbody>
</table>

# 4. 实验二：非线性 Gaussian actor 中的因果干预

## 4.1 环境与预注册式选择

为了验证“大梯度是否真的传导成崩溃”，进一步构建非线性 contextual offline
bandit：共享 MLP trunk、二维 Gaussian mean、可学习方差；质量坐标决定
reward，径向坐标只承担 policy-relative remoteness。负样本的 (state,
quality action, advantage)
逐项复制到六个半径上，形成精确笛卡尔积。正样本停留在径向锚点附近，提供
attraction。

开发 seeds 0–4 用于固定 α=0.1 与标准化径向距离阈值
d=2.0；正式检验使用未参与调参的 held-out seeds 10–29，共 20 个，训练 500
steps。

## 4.2 五类关键干预

| **干预**     | **操作**                                                | **要回答的问题**                 |
|--------------|---------------------------------------------------------|----------------------------------|
| Baseline     | 所有负梯度乘统一 α=0.1                                  | 自然训练是否崩溃                 |
| Near-zero    | 删除当前近场负梯度，保留远场                            | 近场是否是主要致因               |
| Far-zero     | 删除当前远场负梯度，保留近场                            | 切断远场路径能否救援             |
| Far-cap      | 保留远场方向，仅截断异常幅度                            | 是否无需删除样本，只去除放大即可 |
| Global-scale | 统一缩放全部负梯度，使总 norm 与 Far-cap 相同           | 异常梯度幅度是否为中介           |
| Far-to-near  | 截断远场后，把预算人为转移给近场并恢复 baseline 总 norm | 巨大负更新本身是否也有害         |

## 4.3 初始机制诊断

| **方法**     | **原始 far/near 中位比** | **负/正更新比** | **近场系数** | **远场系数** |
|--------------|--------------------------|-----------------|--------------|--------------|
| Baseline     | 56.62×                   | 47.07×          | 0.100        | 0.100        |
| Near-zero    | 56.62×                   | 46.99×          | 0.000        | 0.100        |
| Far-zero     | 56.62×                   | 0.81×           | 0.100        | 0.000        |
| Far-cap      | 56.62×                   | 4.86×           | 0.100        | 0.023        |
| Global-scale | 56.62×                   | 4.86×           | 0.010        | 0.010        |
| Far-to-near  | 56.62×                   | 47.07×          | 5.848        | 0.023        |

全参数空间中的原始远场/近场负梯度中位比为 56.62×。即使负样本整体已经乘
α=0.1，Baseline 的负/正更新比仍约为 47.07×。Near-zero
几乎不改变这一比例；Far-zero 将其降至 0.81×，Far-cap 与等预算
Global-scale 将其降至约 4.86×。

## 4.4 20-seed held-out 主结果

| **方法**      | **最终 reward（95% CI）** | **保持率** | **崩溃** | **最终 \|μᵣ\|** |
|---------------|---------------------------|------------|----------|-----------------|
| Baseline      | 0.201 \[0.165, 0.239\]    | 26.3%      | 19/20    | 5.760           |
| Near-zero     | 0.195 \[0.161, 0.231\]    | 25.4%      | 18/20    | 5.778           |
| Far-to-near   | 0.285 \[0.202, 0.373\]    | 37.1%      | 13/20    | 0.894           |
| Far-zero      | 0.618 \[0.597, 0.639\]    | 80.5%      | 0/20     | 0.173           |
| Far-cap       | 0.666 \[0.653, 0.680\]    | 86.7%      | 0/20     | 0.533           |
| Global-scale  | 0.763 \[0.753, 0.773\]    | 99.3%      | 0/20     | 0.210           |
| Positive-only | 0.782 \[0.771, 0.794\]    | 101.8%     | 0/20     | 0.011           |

<img src="/mnt/data/master_recovery/media/media/image2.png"
style="width:6.45in;height:3.94167in" />

**图 2　20 个 held-out seeds 的 reward 曲线。Baseline 与 Near-zero
几乎重合；Far-zero 与 Far-cap 稳定。**

## 4.5 配对因果效应

| **配对比较**            | **均值差** | **95% bootstrap CI** | **胜出 seeds** | **Wilcoxon p** |
|-------------------------|------------|----------------------|----------------|----------------|
| far_zero − baseline     | +0.417     | \[+0.372, +0.461\]   | 20/20          | 1.9e-06        |
| far_cap − baseline      | +0.465     | \[+0.423, +0.505\]   | 20/20          | 1.9e-06        |
| near_zero − baseline    | -0.006     | \[-0.020, +0.003\]   | 11/20          | 0.62           |
| global_scale − baseline | +0.562     | \[+0.519, +0.603\]   | 20/20          | 1.9e-06        |
| far_to_near − baseline  | +0.084     | \[-0.009, +0.187\]   | 11/20          | 0.45           |

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th><p><strong>核心因果判别</strong></p>
<p>删除近场而保留远场：18/20 仍崩溃，且 reward 与 Baseline
无显著差异。删除或截断远场：0/20 崩溃，分别在 20/20 seeds 中胜过
Baseline。因而在该环境中，远场负梯度不是仅与崩溃相关，而是可被定点干预的主要传导路径。</p></th>
</tr>
</thead>
<tbody>
</tbody>
</table>

## 4.6 时序中介与 OOD 漂移

Baseline 的中位时序为：约第 50 步径向漂移 \|μᵣ\|\>0.5，约第 70 步
\|μᵣ\|\>1，约第 80 步 reward 保持率低于 45%。远场梯度异常在 step 0
即已存在，先于漂移和崩溃。

<img src="/mnt/data/master_recovery/media/media/image3.png"
style="width:6.45in;height:3.94167in" />

**图 3　Baseline 中的传导链：异常远场负梯度先出现，随后径向漂移并伴随
reward collapse。**

## 4.7 稳健性与实现验证

- 阈值稳健性：标准化距离阈值 1.5、2.5、3.0 下，Far-zero 与 Far-cap 均为
  0/5 崩溃；Near-zero 均为 5/5 崩溃。

- 开发集稳定区间：α=0.075 和 0.1 均表现出 Baseline/Near-zero
  崩溃、Far-zero 稳定；α=0.05 属于较弱崩溃区。

- 梯度实现：逐样本全参数梯度聚合与标准 autograd 的相对误差为 2.75×10⁻⁸。

- 预算匹配：Global-scale 与 Far-cap 的 post-negative-gradient norm
  完全相同；Far-to-near 与 Baseline 的匹配误差低于 2×10⁻⁷。

# 5. 证据强度、结论边界与仍未证明的内容

## 5.1 当前可以强声明的内容

- 在质量严重程度与策略距离严格解耦时，远场仍会自然生成异常大的负梯度。

- 异常大梯度的主体来自 policy-score geometry，而不是远场样本平均更差。

- 在当前非线性 Gaussian offline
  环境中，保留远场而删除近场不能阻止崩溃；只处理远场即可阻止崩溃。

- 大梯度幅度是直接中介；统一 α/全局缩放可通过降低该幅度稳定训练。

- 因此“远场几何是异常负梯度的自然来源；异常负梯度是崩溃的直接传导量”在本环境内已经具有较强因果证据。

## 5.2 仍不可直接声称的内容

- 严格指数律：目前证明了显著放大和动态自增强，但未完成随训练步数的指数、线性、二次模型比较。

- 普遍唯一性：该机制是充分且主导的受控反例/机制环境，但真实 D4RL、LLM RL
  或推荐任务还可能存在 critic error、support mismatch、entropy collapse
  等其他机制。

- Distance 必然优于 α：本实验的 Global-scale
  表现很好，说明统一缩放也能控制中介变量；二者在保留近场负信息、泛化与跨任务鲁棒性上的优劣仍需验证。

- DRPO 的 DRO 最优性：今天的实验支撑 repulsion/collapse 理论，不单独验证
  hard filtering 是某一现实任务中的唯一最优解。

# 6. 对 DRPO 论文贡献的含义

今天的结果对 DRPO
的价值不是“把一个已知现象再跑一次”，而是为其理论提供了此前缺失的识别性证据。原始论文已经提出
repulsive optimization、负梯度强度爆炸以及 off-policy collapse
的统一解释；新的实验进一步回答了最关键的质疑：

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th><p><strong>关键质疑与回答</strong></p>
<p>质疑：大负梯度是不是仅仅因为远场样本更差？回答：不是。badness 与
distance 精确解耦后，远场仍出现 16×→24.95×、在非线性 actor 中中位数
56.62×
的放大。质疑：这些大梯度是否真的造成崩溃？回答：在受控环境中，删除近场无效，删除/截断远场
20/20 救援，因果链闭合。</p></th>
</tr>
</thead>
<tbody>
</tbody>
</table>

## 6.1 可形成的论文贡献组合

- 理论：将负优势更新解释为具有自增强性的 repulsive
  dynamics，而非普通的“降低坏动作概率”。

- 几何：把负梯度风险分解为 advantage severity × policy-score
  geometry，明确远场是独立放大因子。

- 识别：构造 badness–distance
  的精确笛卡尔积解耦，排除“远场只是更差”的混杂。

- 因果：通过 Near-zero、Far-zero、Far-cap、Global-scale、Far-to-near
  的定点干预，识别崩溃路径与中介变量。

- 方法解释：统一 α、Distance clipping/filtering 都可被理解为控制异常
  repulsive influence；它们的差异在于是否选择性保留近场负信息。

## 6.2 对原论文 claim 的增强与修正

| **原叙事风险**          | **建议升级后的叙事**                                                                                  |
|-------------------------|-------------------------------------------------------------------------------------------------------|
| 低质量数据导致 collapse | 负优势 severity 与 policy remoteness 共同决定 repulsive influence；远场可在质量独立时单独制造异常梯度 |
| 负梯度会爆炸            | 展示 score-level 解析结构、25×/56×实测分解与动态传导                                                  |
| DRPO 过滤坏样本         | 强调其切断 divergence-inducing repulsive path，而不只是在做数据清洗                                   |
| 硬过滤优于软权重        | 谨慎写成特定 DRO 下的解；经验上进一步比较 α、Distance 与 joint influence                              |

# 7. 与相关工作的关系及 novelty 定位

截至 2026 年 6 月 22 日的 primary-source
检索显示：不能声称“此前没人研究负优势、负梯度主导、罕见失败或
positive-only
filtering”。但可以主张一个更具体、也更强的空缺：此前工作通常将样本质量、罕见性/低概率和
off-policy shift 混合讨论；我们未检索到 DRPO
之前的工作同时完成“badness–distance
精确解耦、全参数梯度放大分解、近远场等控制因果干预与动态 collapse
传导”这一完整证据链。该表述应保留为审慎的文献检索结论，而非绝对 first
claim。

| **工作与时间**                       | **已研究内容**                                                                       | **与本工作的重叠**                         | **本工作的可防守差异**                                                                    |
|--------------------------------------|--------------------------------------------------------------------------------------|--------------------------------------------|-------------------------------------------------------------------------------------------|
| A-LoL（2023）                        | 丢弃负 advantage，仅模仿正优势数据以提高离线稳定性                                   | 承认负优势数据可能有害                     | 未研究远场 score amplification，也未区分近场/远场负信息                                   |
| BAPO（2025-10）                      | 负优势样本主导 policy gradient、存在梯度爆炸风险；自适应 clipping 平衡正负更新       | 与“负梯度主导导致不稳定”高度相邻           | 重点是正负 imbalance 与 entropy/clipping；未展示 badness–distance 解耦及远场因果干预      |
| DRPO（2026-02）                      | repulsive optimization、强度爆炸、off-policy collapse、optimistic DRO/hard filtering | 原始理论主体                               | 今天实验为其补充几何来源分解和因果闭环                                                    |
| Delightful PG（2026-03）             | 用 advantage × surprisal 抑制 rare negative actions，改善更新方向                    | 与“罕见/低概率负样本危险”高度相关          | 发布时间晚于 DRPO；理论重点为 surprisal、方向准确性与上下文预算，不同于连续动作远场动力学 |
| Delightful Distributed PG（2026-03） | stale/mismatched actors 下 high-surprisal failures 主导更新                          | 与 off-policy surprising failures 直接接近 | 晚于 DRPO；当前实验更强调严格解耦、Gaussian score 几何和定点因果干预                      |

## 7.1 推荐 novelty 表述

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th><p><strong>可防守的版本</strong></p>
<p>To our knowledge, prior work has studied negative-advantage
dominance, positive-only filtering, and rare-failure suppression, but
has not isolated policy remoteness from sample quality and causally
established the pathway from far-field score amplification to off-policy
collapse. We provide an exact product construction, full-parameter
gradient decomposition, and targeted near/far interventions that close
this mechanism-level causal chain.</p></th>
</tr>
</thead>
<tbody>
</tbody>
</table>

不建议写：“We are the first to observe negative gradients are harmful.”
BAPO、A-LoL、DPO/GRPO 负梯度分析等都会直接构成反例。

# 8. 建议写入论文的核心 claims、图表与段落结构

## 8.1 建议的核心 claims

6.  Repulsive influence factorization：负样本更新强度由 advantage
    severity 与 policy-score geometry 的乘积共同决定。

7.  Far-field amplification：在 badness 与 distance
    严格独立时，远场仍产生数量级更大的负梯度。

8.  Self-amplifying dynamics：负向排斥扩大策略相对距离，继而放大
    score，形成正反馈。

9.  Causal collapse pathway：只处理远场异常负梯度即可阻止 OOD 漂移和
    reward collapse；删除近场无效。

10. Stability–generalization
    interpretation：负梯度并非应被全部删除，关键是防止其异常 scale
    压倒正样本 attraction；统一 α 与 distance-aware control
    是不同的稳定化实现。

## 8.2 建议的实验章节结构

- Mechanism isolation：乘积流形，展示 16×→24.95× 与聚合 29.13×。

- Nonlinear actor validation：全参数梯度中位
  far/near=56.62×，并报告负/正更新比。

- Causal intervention：Baseline / Near-zero / Far-zero / Far-cap /
  Global-scale / Far-to-near。

- Temporal mediation：远场梯度、μᵣ 漂移、负样本整体远场化、reward
  collapse 的时间顺序。

- Robustness：距离阈值、α 区间、20 held-out seeds、autograd
  与预算匹配单元测试。

- External validity：D4RL 或真实推荐数据中报告 \|A\|–distance
  相关、二维分桶与方法比较。

## 8.3 论文中需要避免的过度表述

- 避免把 25× 或 56×
  写成普适常数；普适的是放大机制，倍率依赖距离范围、σ、网络 Jacobian
  和参数化。

- 避免把当前结果写成严格指数时间律；先写 self-amplifying / rapidly
  growing，指数律完成拟合后再升级。

- 避免把 positive-only 当作理论反例或主竞争方法；它删除全部
  repulsion，只是极端稳定参考线。

- 避免写 Distance 必然优于 α；当前实验反而显示 global scaling
  可以有效控制中介变量。

- 避免写所有 offline collapse 的唯一原因；应写 sufficient and dominant
  pathway in the controlled environment。

- 避免继续使用“Both μ and σ
  expand”。原代码与新实验表明，远场负优势对实际 σ
  的直接作用通常是收缩；可以写 mean/log-std gradient sensitivity
  grows，或 mean repulsion and support contraction jointly amplify
  standardized distance。

- 避免用 expected Fisher 的 SPD 直接推出固定 off-policy
  样本在联合参数空间所有方向扩张。论文应以完整 signed-gradient field 的
  Jacobian 和真实固定点为稳定性对象。

# 9. 旧阶段路线回顾与更新后的执行顺序

今天的受控机制实验已经足以支撑论文理论章节的大幅强化。为了将其从“强机制论文”进一步推向“广泛经验结论”，建议按优先级推进：

P0（第一优先级）：负梯度稳定外推与泛化实验。先证明 positive-only 的
imitation ceiling、受控近场负梯度的 OOD 收益、负推力相变与
distance-aware recovery。

P0-并行理论：完成一维 Gaussian 稳定外推闭式模型、通用 surprisal
increment identity、增长律与步长条件；指数律检查作为该理论的一部分。

P1（第二优先级）：Categorical bandit 严格隔离，复制相同 advantage 到不同
token surprisal，验证离散 support suppression 与 rare/common 定点干预。

P2：小型 Transformer 序列实验，固定 context、token identity、advantage
与长度，只改变 learner-relative surprisal；随后进入真实 RLVR。

P3：D4RL、推荐与机器人外部验证；报告 \|A\|–distance
耦合、二维梯度分桶、critic noise、方法比较与跨参数化稳健性。

# 10. 从推荐扩展到通用 Off-Policy Policy Optimization

当前 DRPO 的理论对象不应再被限制为推荐系统中的 generative
policy。推荐只是一个重要应用；真正的研究对象是：当固定或陈旧数据被当前策略重复优化时，负优势更新如何形成
repulsive dynamics，以及如何在保留有效负信息的同时避免远场失稳。

建议将论文主问题重写为“通用 off-policy policy optimization
中的排斥诅咒”，并把推荐实验保留为真实应用验证之一。理论和机制实验覆盖连续
Gaussian policy；下一阶段补充 categorical/softmax policy
后，可自然延展到 LLM RLVR、离散控制、diffusion/flow policy extraction
和其他生成式决策模型。

## 建议的总框架：

- 统一对象：任意可微策略 πθ(a\|s)，而不是某一种推荐架构。

- 统一风险：负优势严重程度 × policy-relative score geometry ×
  重复更新次数 × 梯度方向一致性。

- 统一问题：哪些负更新提供局部边界信息，哪些负更新因远场放大而成为破坏性排斥。

- 统一方法族：全局 α、advantage-based tapering、distance/surprisal-aware
  capping、joint influence control。

- 统一实验版图：机制环境 → continuous control / D4RL → recommendation →
  categorical bandit → LLM RLVR。

一个可考虑的总标题方向是：Breaking the Curse of Repulsion: Unified
Repulsive Dynamics in Off-Policy Policy
Optimization。若保留原题，可将推荐降为副标题或应用章节，以避免理论影响力被领域标签限制。

# 11. 连续—离散统一的 Repulsive Surprisal Dynamics

## 11.1 与动作空间无关的一阶动力学恒等式

定义样本 z=(s,a) 在当前策略下的 surprisal：Dθ(z) = −log πθ(a\|s)。对
A(z)\<0 的样本，标准 policy-gradient ascent 等价于沿负 score 方向更新：

θ⁺ = θ − η \|A(z)\| ∇θ log πθ(a\|s).

对 Dθ(z) 做一阶 Taylor 展开，可得：

Dθ⁺(z) − Dθ(z) = η \|A(z)\| ‖∇θ log πθ(a\|s)‖² + O(η²).

这一定理给出一个跨连续与离散策略都成立的核心事实：负优势更新必然使被更新样本在当前策略下变得更罕见；其变罕见速度由
advantage severity 和 score norm 的平方共同决定。重复使用同一批
off-policy 数据时，动力学可写成：

Dₜ₊₁ = Dₜ + η \|A\| κθ(Dₜ) + O(η²), κθ(D)=‖∇θ log πθ(a\|s)‖².

动作空间与策略参数化的差异，主要体现在 amplification law κθ(D)
的形状；统一理论的核心不是宣称所有策略都有相同的爆炸速度，而是识别“负更新
→ surprisal 增长 → 后续影响改变”的共同递推结构。

## 11.2 连续 Gaussian：梯度幅度自增强

对固定方差的 Gaussian mean，surprisal 的远场主项满足 D≈‖a−μ‖²/(2σ²)，而
mean-score norm² 满足 ‖∇μ logπ‖²≈2D/σ²，因此 κ(D) 随 D
线性增长。若同时学习 log σ，方差 score 的远场主项近似随 D
增长，其平方可达到 O(D²)。因此连续场景可能出现真正的 gradient-amplitude
runaway：距离越大，下一步排斥越强，进而继续增加距离。

## 11.3 离散 softmax：持续排斥与支持集坍缩

对 categorical softmax，采样 token y 的 logit score 为
e_y−p。其欧氏范数随 1−p_y 增长，但在 p_y→0
时有界。因此离散场景不应机械复制“单 token
欧氏梯度无界爆炸”的说法。更准确的动力学是：进入低概率区后，负更新仍保持非消失的持续排斥，使
log-odds 近似线性下降，而 token probability 近似指数衰减。

z_y−z_j ≈ −ct ⇒ p_y(t) ≈ exp(−ct).

在 Fisher / natural-policy geometry 中，categorical score 的内禀范数²为
1/p_y−1=eᴰ−1，随 rarity 无界增长。这为 Gaussian Mahalanobis distance 与
categorical surprisal 提供了统一的信息几何解释；但论文必须区分 intrinsic
norm 与 vanilla SGD 的实际参数梯度，后者还受到 Transformer Jacobian 和跨
token 梯度耦合影响。

| **策略类型**                          | **远场变量**                     | **κ(D) 的典型行为**  | **重复负更新的主要失稳形式**       |
|---------------------------------------|----------------------------------|----------------------|------------------------------------|
| Gaussian mean                         | Mahalanobis distance / surprisal | 约随 D 线性增长      | 梯度幅度自增强、均值漂移           |
| Gaussian log-variance                 | Mahalanobis distance / surprisal | 远场可达 O(D²)       | 方差收缩/膨胀与更强 runaway        |
| Categorical softmax（logit 欧氏几何） | token surprisal −log p           | 增长后饱和为常数量级 | 持续 suppression；概率近似指数衰减 |
| Categorical softmax（Fisher 几何）    | token surprisal −log p           | eᴰ−1，无界           | 内禀 policy distance 快速扩大      |

## 11.4 统一理论的潜在贡献

- 把 Gaussian far-field distance 与 LLM token surprisal 统一为
  policy-relative remoteness。

- 用 surprisal increment identity 解释负更新为何天然具有自我排斥性。

- 用 κ(D) 区分连续场景的 amplitude runaway 与离散场景的
  support-suppression runaway。

- 把 staleness、低概率 token、rare failure、entropy collapse 和连续
  policy drift 放入同一递推框架。

- 为 α、distance cap、surprisal reweighting、negative veto
  等方法提供同一控制论解释。

# 12. 为什么负梯度并不总是有害：局部泛化与远场失稳

论文需要同时回答两个看似矛盾的事实：负优势更新能够造成
collapse，但大量实证又表明，完全删除负样本可能损失性能、数据效率、探索与
OOD 泛化。最有解释力的理论不是“负梯度有害或无害”的二分法，而是负梯度存在
informativeness–amplification trade-off。

## 12.1 正优势提供 attraction，但存在 imitation ceiling

正优势更新把策略拉向已观察到的成功行为，类似 advantage-filtered
imitation。它能够快速学习已知好模式，但当 πθ 已贴近这些样本时，score 与
attraction
逐渐减弱；同时，正样本本身并不告诉模型哪些相邻模式是错误的，也难以直接消除数据中未覆盖的坏
mode。

## 12.2 受控负优势提供 boundary shaping 与 mode suppression

负优势并不是凭空把模型“推向正确的未知领域”。它首先降低已知坏行为的概率，并把释放出的概率质量按照模型已有的表示几何与先验重新分配到其他候选。若负样本仍位于当前策略的局部支持附近，它通常具有三类价值：

- 边界信息：区分“好样本附近哪些方向不能走”，形成比单纯模仿更大的决策
  margin。

- 坏 mode 抑制：在高精度、长时程和多模态任务中，仅提高好 mode
  并不必然消除竞争性的坏 mode。

- 组合与探索：释放的概率质量由预训练先验和共享表示重新分配，可能激活训练集中未直接展示、但模型已经潜在掌握的替代路径。

因此，“超越数据集”应被严谨表述为：负更新可以通过排除已知错误、重整概率质量和组合已有行为片段，使策略在未直接示范的区域获得更好表现；它并不保证创造训练前完全不存在的新能力。

## 12.3 为什么远场负梯度会从有益变为有害

近场负样本更接近当前 on-policy 分布，其方向仍能近似局部
policy-improvement
信号；随着样本被重复推远，它对当前局部决策边界的相关性可能下降，但 score
amplification 或持续 suppression
并不会同步下降。于是出现核心错配：信息价值随距离衰减，优化影响却随距离增长或保持不消失。

Useful local repulsion → boundary shaping / mode removal /
generalization

Far-field repulsion → low relevance × excessive influence → drift /
collapse

这正好解释 stability–generalization
trade-off：完全删除负梯度最稳定但可能落入 imitation
ceiling；适度保留近场负梯度有助于泛化；未受控的远场负梯度则压倒正向
attraction 并形成 runaway。

## 12.4 建议的形式化方向

- 定义 negative influence：I(z)=(-A(z))₊ ‖∇θlogπθ(z)‖。

- 定义 directional utility：U(z)=〈g⁻(z), g\*〉/‖g⁻(z)‖，其中 g\*
  为真实或高质量近似的 policy-improvement 方向。

- 研究 U(d) 是否随距离下降，而 I(d)
  随距离上升，从而产生可证明的安全半径或最优控制区间。

- 把远场风险写成“影响大小与方向可靠性的乘积”，而不是仅用 advantage 或
  surprisal 单变量判断。

- 预言负梯度强度存在倒 U 型效果：0 为 imitation
  ceiling，中等强度取得最好泛化，过大或过远导致 collapse。

## 12.5 一维 Gaussian 的稳定外推闭式模型

该模型把“为什么负梯度有用”和“为什么同一负梯度会失控”写进同一个动力学。设固定方差
Gaussian policy 的均值为 μ；训练中有最佳已观测正样本 a₊、正优势强度
p\>0，以及位于其另一侧的负样本 a₋\<a₊、负优势强度 n\>0，负信号系数为 α。

J(μ) = p log πμ(a₊) − α n log πμ(a₋).

其连续时间均值动力学为： μ̇ = \[p(a₊−μ) − αn(a₋−μ)\] / σ²。

当 p\>αn 时存在稳定平衡点： μ\* = (p a₊ − αn a₋)/(p−αn) = a₊ +
\[αn/(p−αn)\](a₊−a₋) \> a₊。

因此正样本模仿最多把策略拉向
a₊，而适度负推力可以把策略稳定地推到最佳正样本支持之外；这给出了
negative-gradient-induced extrapolation 的闭式证明。

当 αn→p⁻ 时，稳定点快速远移；当 αn=p 时出现持续漂移；当 αn\>p
时固定点消失并产生发散趋势。这给出 bounded extrapolation → critical
drift → divergence 的相变，并自然导出距离衰减
α(d)：近场保留推力，远场逐步衰减。

多维场景还需加入方向可靠性：只有与真实 improvement direction
正对齐的局部负梯度才具有泛化价值；论文应测量 cosine
alignment，并验证距离增大时 utility 下降而 influence 上升。

## 12.6 可学习方差下的联合均值—方差稳态

令 ξ=log σ，正、负样本分布的均值与条件方差分别为 (m₊,v₊) 与
(m₋,v₋)，正信号总强度为 p，负信号有效强度为 q=αn。定义：

M₊(μ)=v₊+(μ−m₊)², M₋(μ)=v₋+(μ−m₋)².

联合 population-gradient flow 为：

μ̇ = \[p(m₊−μ) − q(m₋−μ)\] / σ²,

ξ̇ = \[pM₊(μ) − qM₋(μ)\] / σ² − (p−q).

当 p\>q 时，均值候选固定点为：

μ\* = (p m₊ − q m₋)/(p−q).

将 μ\* 代回方差方程，可得内部正方差固定点：

σ²\* = \[pM₊(μ\*) − qM₋(μ\*)\]/(p−q).

因此完整联合稳态需要两个条件同时成立：（1）p\>q，保证均值恢复斜率为正；（2）pM₊(μ\*)\>qM₋(μ\*)，保证
σ²\*\>0。第二个条件通常更严格，因为远场负样本的 M₋ 含平方距离项。

在该内部固定点处，动力学 Jacobian 简化为：

J_F(θ\*) = diag(−(p−q)/σ²\*, −2(p−q)).

只要上述两个存在条件成立，两个特征值均为负，联合均值—方差稳态局部稳定。令
K\*=−J_F(θ\*)，则 K\*≻0，仍可保留原论文希望使用的 SPD contraction
表达，但 SPD 来自真实净动力学在联合固定点处的恢复曲率，而不是 on-policy
expected Fisher。

α 的作用。减小 α 会同时降低 q 与 qM₋：它既把均值分支从 expansion 拉回
contraction，也减弱远场负样本对 log σ
的收缩压力，使正样本条件残差提供的方差恢复力重新占优。α
不能阻止确定性正样本自身的 MLE 方差坍缩；该情形仍需正样本非零条件
spread、固定方差、entropy/KL 正则或 σ 下界。

| **Advantage** | **标准化位置**     | **对 σ / entropy 的方向** |
|---------------|--------------------|---------------------------|
| A\>0          | \|z\|\<1（正近场） | σ↓，entropy↓              |
| A\>0          | \|z\|\>1（正远场） | σ↑，entropy↑              |
| A\<0          | \|z\|\<1（负近场） | σ↑，entropy↑              |
| A\<0          | \|z\|\>1（负远场） | σ↓，entropy↓              |

该四象限与 WAPO/STARE 等离散 entropy 分析具有一致结构：entropy
变化不能只由 advantage 符号判断，必须同时考虑样本在当前策略下是
peak/near 还是 valley/far。

# 13. 最新相关工作的机制对照

这些工作不能简单归类为“只做方法 trick”。WAPO、STARE、Mu-GRPO
等都包含有价值的局部梯度或熵分析；但它们大多解释某个局部切面，并未建立从
repeated negative update 到 policy-relative remoteness 再到 collapse
的统一动力系统。DRPO
的机会是把这些分散观察统一起来，而不是否认它们的理论贡献。

| **工作**                            | **它解释了什么**                                                                              | **如何处理负信号**                                                           | **相对 DRPO 仍缺少什么**                                                      |
|-------------------------------------|-----------------------------------------------------------------------------------------------|------------------------------------------------------------------------------|-------------------------------------------------------------------------------|
| WAPO (2606.16154)                   | peak/valley × advantage sign 的局部概率与熵变化；不同负 token 可导致高熵或低熵 collapse       | 最终采用 positive-only：屏蔽所有非正 advantage completion                    | 承认部分负 token 可能有用，但选择问题超出范围；无重复 off-policy 自增强动力学 |
| STARE (2606.19236)                  | advantage–surprisal 四象限和 entropy near-criticality                                         | 熵低于目标时，放大高-surprisal 正 token，削弱高-surprisal 负 token；其余保持 | 重点是闭环熵控制；没有 badness–rarity 严格解耦和 drift/collapse 因果链        |
| Mu-GRPO (2605.17570)                | 高 staleness 下 prefix support mismatch；危险更新集中在越过 off-support trigger 后的负 suffix | relaxed clipping 保留有用 stale gradient；NAV 只 veto trigger 后负 suffix    | 非常接近离散 repulsion，但未给出统一 surprisal increment 和连续—离散动力学    |
| ASymPO (2606.03070)                 | stale 正负 response 在当前 NLL scale 上失衡                                                   | 按每条 response 的当前平均 NLL stop-gradient 归一化；保留非零负信号          | 控制 loss scale，不直接分析 score geometry、自增强或方向可靠性                |
| TOPR (2503.14286)                   | naive off-policy negative objective 无界；正负样本的有效比例决定性能                          | 对负样本使用 tapered importance sampling，稳定利用正负样本                   | 证明“负样本有价值”，但没有 far-field score dynamics 与因果隔离                |
| Negative Reinforcement (2506.01347) | 负样本单独训练可提高 pass@k 和多样性；抑制错误后由模型先验重新分配概率                        | 提高 NSR 权重，而非删除负梯度                                                | 直接支持负梯度的泛化价值；尚未解释近场有益—远场有害的转折                     |
| OGPO (2605.03065)                   | 高精度/长时程生成控制中，positive-only 无法可靠抑制坏 mode                                    | 保留 clipped negative-advantage extraction                                   | 连续生成策略中的外部支持，但未研究 far-field 自增强边界                       |

## 13.1 WAPO 的真正含义

WAPO 并不是“选择性保留有益负梯度”的方法。它的 peak–valley taxonomy
说明负更新的局部作用依赖当前 token 分布：Neg-peak 倾向提高
entropy，Neg-valley 倾向降低
entropy；两者都可能在不同条件下失稳。论文明确承认部分负优势 token
可能含有有效信号，但在 coarse sequence reward
下难以可靠选择，因此采用最保守的 coarse filter：只更新正优势
completion。

## 13.2 STARE 比 WAPO 更接近“打压危险部分、保留其余”

STARE 不删除全部负梯度。它在 entropy 低于目标时，使用 batch 内 surprisal
quantile 找出高-surprisal token：增强正优势高-surprisal
token，减弱负优势高-surprisal token；低-surprisal负 token
保持原始权重，entropy 恢复后则关闭干预并退回
GRPO。因此它已经隐含了“负梯度并非一律有害”，但判断标准服务于 entropy
regulation，而不是远场 repulsive dynamics。

## 13.3 Mu-GRPO 已经触及重复排斥，但尚未统一

Mu-GRPO 的 diagnosis 很重要：高 staleness 下，负优势 trajectory 一旦某个
prefix 跨越当前 policy 的支持边界，后续 suffix 仍可获得显著更新，形成
localized instability。它仅 veto 触发点后的负 suffix，并通过 relaxed
clipping 保留触发前的 stale learning
signal。这是目前最接近“局部负信号有用、off-support
负信号有害”的离散工作之一，应作为重点竞争与支持文献。

## 13.4 现有证据已经支持负梯度的泛化价值

- TOPR：同时利用正负样本提高准确率和数据效率，并观察到最佳有效正样本比例并非
  100%。

- Negative Reinforcement：negative-only 可提升整个 pass@k
  曲线；positive-only 提高 pass@1 却可能降低高 k 多样性。

- Good Actions Succeed, Bad Actions Generalize：失败轨迹片段可通过
  experience stitching 支持未见组合任务。

- OGPO：高精度和长时程任务中，negative advantage 对抑制坏 mode 很重要。

这些结果已经否定“负梯度只是不得不忍受的噪声”。我们需要新增的理论贡献，是解释其收益为何集中在局部/受控区域，以及为何同一排斥机制在远场会从
generalization signal 转化为 collapse driver。

## 13.5 OGPO：强外部证据，但尚未闭合负梯度外推机制

**OGPO（Off-policy Generative Policy Optimization）**面向 diffusion/flow
generative control policy，使用 off-policy critic 产生 group-relative
advantage，并通过 PPO 风格目标对完整生成过程进行 full-policy
finetuning。它明确同时利用正、负 advantage
gradient，并在机器人操控任务中展示了相对于 behavior cloning、steering 和
residual correction 更强的策略改进能力。

**OGPO 已经建立的事实：**（1）BC/Best-of-N+SFT
容易局限于已有动作支持；（2）OGPO 的 Q-guided full-policy update
可使动作流形向离线动作分布支持之外扩展；（3）no-negative ablation
表明，在高精度与长时程任务中，仅模仿高价值动作不足以压制竞争性坏
mode，负优势梯度具有实际价值。

**OGPO 尚未证明的关键链条：**它没有把 action-manifold expansion
单独归因于负梯度；没有固定正优势、critic、采样与网络共享效应后，只改变负梯度；没有研究负样本相对距离与梯度尺度；也没有给出“有益外推—临界失稳—runaway
collapse”的动力学和 distance-aware 控制。因此 OGPO 可以作为“负梯度有助于
support expansion / bad-mode
suppression”的强现实证据，但不能替代我们计划中的严格机制实验。

| **问题**                   | **OGPO 已覆盖**                                            | **本工作需要补齐**                                               |
|----------------------------|------------------------------------------------------------|------------------------------------------------------------------|
| 正样本模仿是否存在支持上限 | 通过 BC/QC 对照与动作流形可视化给出强经验支持              | 给出可解析的 imitation ceiling 与稳定外推闭式解                  |
| 负优势是否有用             | no-negative ablation：部分高精度、长时程任务明显依赖负优势 | 隔离负梯度对 support extrapolation 的直接因果贡献                |
| 负梯度为什么会从有益变有害 | 主要归因于 critic over-exploitation 与任务难度             | 建立距离增长、score amplification、固定点消失与发散临界条件      |
| 如何控制负梯度             | PPO clipping、成功样本 BC 正则、保守 advantage 等          | 距离/Surprisal aware attenuation：保留近场外推，抑制远场 runaway |

## 13.6 建议重点深化的三条前序工作线

- TOPR（2025）：最适合作为方法层面的前序基础。它已经证明 off-policy
  场景中正确利用正负样本优于丢弃负样本，并通过 tapered importance
  sampling 控制不稳定更新。我们的深化点是解释“为什么必须
  taper、危险性为何与 policy-relative distance
  相关，以及何时负推力越过稳定阈值”。

- Low-Probability Tokens / BAPO（2025）：最适合作为稳定性现象与 rarity
  证据。前者直接测量低概率 token 的较大梯度，后者展示负 token
  数量、长度、loss contribution 与 clipping imbalance。我们的深化点是将
  badness、数量、长度与 rarity 严格解耦，并建立 repeated negative update
  的跨时间动力学。

- Negative Reinforcement + Good Actions Succeed, Bad Actions
  Generalize（2025）：最适合作为“负信号促进多样性与泛化”的前序基础。它们分别从概率质量重分配和失败轨迹
  experience stitching
  解释负数据价值。我们的深化点是证明负梯度何时直接推动策略越过正样本支持，以及为什么这一机制在远场会反转为
  collapse。

**OGPO 的角色：**它发表于 DRPO
之后，更适合作为机器人连续控制中的独立后续证据，而不是时间意义上的前序基础。论文叙事可以写成：OGPO
观察到 full-policy RL 的 support expansion 与负优势价值；DRPO 提供更早的
repulsive dynamics
理论，并进一步给出负梯度外推收益、远场失稳与距离控制的统一解释。

## 13.7 2026 年新增相关工作与 novelty 风险检查

| **工作**                                  | **核心观察/方法**                                                                         | **与本工作的关系**                                                              |
|-------------------------------------------|-------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| REAL (2602.05630)                         | 把 reward 视为分类标签，形成有界、单调的梯度分配，缓解少量负样本主导                      | 支持“负梯度预算需要受控”；未研究距离递推与外推价值                              |
| POPO (2605.06650)                         | 只用正 rollout，通过概率归一化产生隐式负梯度，并用 siamese/momentum 稳定训练              | 是 positive-only 路线的重要反例；需通过泛化任务说明显式负优势何时仍不可替代     |
| Mu-GRPO (2605.17570)                      | off-support prefix 后的负 suffix 更新导致局部失稳；veto 危险后缀，保留其余 stale gradient | 最接近离散版“近场有用、越界有害”；未给出统一 surprisal increment 与距离放大律   |
| WAPO / STARE (2606)                       | 分别采取 winner-only 与 surprisal-aware entropy regulation                                | 说明负 token 不能仅按 advantage sign 判断；本工作需要解释跨时间的信息—放大错配  |
| Gradient Gap / RLVR Dynamics (2510.08539) | 从 trajectory/token gradient gap 分析 RLVR 的优化动态                                     | 属于广义动力学邻域，应在定理和实验层面对比其研究变量是否覆盖 repeated repulsion |

**稳定的 novelty 边界：**不能再声称“首次发现负优势危险”“首次发现低概率
token
梯度更大”或“首次提出削弱负梯度”。更有防御力的贡献是：统一解释负优势为何既能带来
support extrapolation / mode suppression，又会因 repeated repulsion 与
policy-relative remoteness 形成自增强失稳；并以距离/Surprisal
控制在同一理论下保留收益、阻断崩溃。

## 13.8 BAPO 与 Low-Probability Tokens：证据边界必须写清

BAPO 的主要证据是：staleness sweep 下 reward/entropy/gradient norm
失稳；正负 token 数量、长度与 loss contribution 统计；token probability
与 importance ratio 的分桶关系；以及放宽正/负 clipping
边界的干预。它说明低概率负 token 在真实 LLM off-policy RL
中会主导更新，但没有固定 advantage、长度和数量后单独改变
rarity，也没有复制同一样本到多个概率桶。

“Do Not Let Low-Probability Tokens Over-Dominate”更直接：按 token
probability 分位测量全参数梯度，并做低概率/高概率 token
的选择性单步更新。它证明低概率 token 的 score
相对更大且会通过共享参数主导更新，但 softmax logit score 在 p→0
时有界；−log p→∞ 不等于单 token logit 梯度无界。

我们的新增价值不是重复“低概率 token 梯度较大”，而是：（1）将
badness、数量、长度与 rarity 严格解耦；（2）建立 repeated negative
update → surprisal/distance 增长 → 后续 influence 改变的递推；（3）通过
rare/common 或 near/far 定点干预闭合 drift/collapse 因果路径。

# 14. 下一阶段理论与实验增强计划

## 14.1 理论并行任务：完成连续—离散统一定理

11. 正式证明通用 surprisal increment
    identity，并给出二阶余项和步长条件。

12. 推导 Gaussian mean、Gaussian variance、categorical logits 和 Fisher
    geometry 的 κ(D) 闭式或上下界。

13. 区分 gradient-amplitude runaway、persistent support suppression 和
    intrinsic-distance expansion 三种动力学。

14. 建立 repeated off-policy update
    的稳定/发散条件，明确何时为线性、几何或超线性增长。

15. 尝试证明 informativeness–amplification mismatch：方向可靠性随
    distance 下降而 influence 随 distance 上升。

## 14.2 P1：Categorical bandit 严格隔离实验（第二优先级）

- 构造 Cartesian product：reward/advantage 只依赖 quality
  coordinate，初始 token surprisal 只依赖 rarity coordinate。

- 对完全相同的 A、token 数量和上下文结构，复制到多个初始 p(y)
  桶，排除“低概率 token 更差/更多/更长”的混杂。

- 同时测量 logit Euclidean score、全参数 score、Fisher score、surprisal
  增量和多步 pₜ 衰减。

- 比较 Baseline、rare-negative cap、common-negative cap、global
  α、等预算 rare-to-common transfer。

- 验证离散版因果链：rare negative → persistent suppression /
  shared-parameter drift → support or entropy collapse。

## 14.3 P0：验证负梯度为何能超越 positive-only（第一优先级）

设计专门的 generalization benchmark，而不是只比较训练
reward。训练正样本仅覆盖有限成功模式，负样本覆盖决策边界与坏
mode；测试集包含未见状态—目标组合、未展示成功路径或 OOD 动作区域。

- Positive-only：测量 imitation ceiling 与正样本 attraction 饱和。

- Near-negative：检验局部边界塑形、坏 mode 抑制和 OOD 泛化提升。

- Far-negative：检验过量排斥、OOD 漂移与 collapse。

- Scaled all-negative：寻找 stability–generalization 的倒 U 型最优点。

- 核心指标：in-domain reward、OOD success、coverage/pass@k、mode
  count、entropy、梯度 alignment 与距离分桶。

最强预期结果不是“负梯度越多越好”，而是：Positive-only
稳定但泛化受限；受控近场负梯度取得最佳
OOD；远场或过强负梯度导致性能反转和 collapse。

## 14.4 P2：小型 Transformer 序列实验

- 固定 context、token identity、advantage、sequence length，只通过受控
  logit bias 或 stale checkpoint 改变 learner-relative surprisal。

- 测量单 token 与全参数梯度、跨 token interference、direction coherence
  和概率随重复负更新的动力学。

- 复刻 near/far 定点干预：仅压 rare-negative 与仅压 common-negative。

- 再扩展到真实数学 RLVR，报告 advantage × surprisal 二维分桶、staleness
  轨迹与 collapse onset。

## 14.5 论文重构建议

- 理论主线：Repulsive Surprisal Dynamics，而非推荐专属 DRO 叙事。

- 核心发现：negative signal 的 usefulness 由 local information 与
  dynamical amplification 共同决定。

- 实验主线：continuous isolation + causal intervention + categorical
  replication + generalization trade-off。

- 方法主线：SNA2C/α 是全局控制，distance/surprisal cap
  是选择性控制，joint influence 是统一形式。

- 应用主线：推荐作为原始真实场景，LLM RLVR
  作为高影响力离散验证，D4RL/控制作为跨领域中间层。

## 14.6 更新后的实验优先级（v4）

| **优先级** | **实验**                             | **必须回答的核心问题**                                                              |
|------------|--------------------------------------|-------------------------------------------------------------------------------------|
| P0         | 负梯度稳定外推与泛化实验             | 负梯度能否直接把策略推出正样本支持并改善 OOD；何时由有益外推转为 runaway            |
| P1         | Categorical bandit 严格解耦实验      | 连续距离与离散 surprisal 能否落入统一 repulsive dynamics                            |
| P2         | 小型 Transformer 序列实验            | 共享参数、token interference、staleness 与 support suppression 是否复现 bandit 结论 |
| P3         | D4RL / 机器人 / 推荐 / RLVR 外部验证 | 理论机制在真实耦合数据中的解释力与方法收益                                          |

## 14.7 P0 实验：负梯度的稳定外推—临界失稳—距离控制

16. 阶段 A：Imitation ceiling。正优势训练集只包含边界内的最佳动作
    a₊，真实最优 a\* 位于正样本支持之外。验证 positive-only 收敛到
    a₊附近，无法达到 a\*。

17. 阶段 B：Controlled extrapolation。在 a₊另一侧放置近场负样本 a₋，匹配
    advantage 与梯度预算。验证适度负推力使策略均值越过 a₊并向未见的
    a\*移动，从而提高 OOD reward、coverage 或 success。

18. 阶段 C：Phase transition。系统扫描
    α、\|A⁻\|、距离、负样本比例与重复更新次数，验证稳定固定点在负推力接近正向吸引时远移并最终消失，出现
    bounded extrapolation → persistent drift → divergence。

19. 阶段 D：Distance-aware recovery。比较 positive-only、全局
    α、advantage-only、distance decay、joint influence。理想结果是
    distance-aware 方法既超过 positive-only 的泛化上限，又避免
    unweighted negative update 的崩溃。

20. 阶段 E：多维方向可靠性。报告负梯度与真实 improvement direction 的
    cosine，验证 near-negative 的 directional utility 较高，而
    far-negative 出现“信息价值下降、影响规模上升”的
    information–amplification mismatch。

**最低充分证据标准：**不能只展示 Joint \>
Positive-only。必须同时展示：（1）策略越过最佳正样本支持；（2）测试收益来自未见区域而非训练拟合；（3）负推力存在可复现的倒
U
型或相变；（4）距离控制保留有益外推并阻止远场崩溃；（5）等预算与方向对照排除“只是梯度更小”的解释。

## 14.8 理论升级清单

- 定理 A（通用排斥）：对任意可微策略，负优势单步更新使该样本 surprisal
  增长，首阶增量为 η\|A\|‖∇logπ‖²，并给出二阶余项和步长条件。

- 定理 B（稳定外推）：在一维 Gaussian
  正负锚点模型中，若负推力小于正向吸引，则存在位于最佳正样本之外的稳定固定点；负推力接近临界值时固定点远移，越过临界后出现无界漂移。

- 定理 C（距离放大）：给出 Gaussian mean/variance、categorical logit 与
  Fisher 几何下 score norm 随 distance/surprisal 的增长律或上下界。

- 定理 D（信息—放大错配）：在局部方向可靠性随距离下降、score influence
  随距离上升的条件下，存在最优安全半径或最优负梯度强度区间。

- 推论（方法原则）：全局 α 控制总负推力；distance/surprisal weighting
  选择性控制远场；joint influence 依据 \|A\|×score risk 同时控制
  severity 与 geometry。

- 定理 E（联合 Gaussian 稳态）：对具有非零条件 spread
  的正负样本分布，推导 μ\* 与 σ²\*
  的闭式解、内部稳态存在条件和真实动力学
  Jacobian；区分均值临界边界与更早的方差临界边界。

- 原定理替换：删除“negative advantage 仅凭符号使联合参数所有方向扩张”的
  expected-Fisher 证明；保留固定方差均值排斥作为特例，并以 signed
  off-policy field 的谱条件统一高维和神经网络版本。

## 14.9 论文重构后的主线

- 问题一：为什么负优势有用？——突破 positive-only imitation ceiling，进行
  boundary shaping、bad-mode suppression 和 support extrapolation。

- 问题二：为什么同一负优势又会有害？——重复 off-policy
  排斥提高距离/Surprisal，使信息相关性下降而优化影响增长，最终触发 drift
  与 collapse。

- 问题三：如何平衡？——用 distance/Surprisal-aware attenuation
  保留局部负信号，控制远场 repulsive influence。

- 统一定位：从“off-policy generative recommendation 的 DRO
  方法”升级为“跨连续与离散策略的 Repulsive Policy Dynamics 与
  stability–generalization
  trade-off”。推荐保留为重要真实应用，而不再作为理论边界。

## 14.10 未来统一实验环境：论文定稿前的必做重构

当前按研究识别需求保留三套环境是合理的：环境 A 隔离梯度来源，环境 B 闭合
collapse 因果链，环境 C 验证 imitation
ceiling、稳定外推与相变。但论文最终不应把它们呈现为三个互不相关的 toy
world。

最终结构建议压缩为两层：第一层是一维 fixed-variance Gaussian
解析模型，用于给出固定点、最优外推点和临界条件；第二层是一个统一的
nonlinear Gaussian benchmark，具有共享的 state/action/reward/advantage
定义，通过配置切换三种 protocol：（I）gradient-source
isolation；（II）causal-collapse
intervention；（III）stable-extrapolation and generalization。

执行原则：当前先在环境 C 中快速、干净地完成 P0
机制识别；结论稳定后，再把 Protocol III 迁移到统一非线性
benchmark。环境统一属于论文工程与叙事的必做项，但不应阻塞当前
P0，也不得为了表面统一而牺牲 strict decoupling、定点干预或已锁定证据。

扩展层安排：主文仅用一个明确假设框声明 fixed advantage、wᵢ=1 和无
value/Q network；附录再给出 wᵢ(θ)Aᵢ(θ)
的一般形式及其可能移动稳定边界的说明。除非审稿人明确要求，不在主文展开动态
critic、importance sampling 或复杂 reweighting 的额外实验。

统一实现中的方差分支：统一 nonlinear Gaussian benchmark 应支持 fixed-σ
与 learnable-σ 两种配置。主文先用 fixed-σ 展示最简均值相变，再用
learnable-σ 作为一项关键扩展，验证 variance boundary 可先于 mean
boundary
失稳；不需要将该变量与所有网络宽度、激活函数和数据分布做笛卡尔积。

# 15. 最终结论

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th><p><strong>关于“DRPO 是否仍然 solid、是否有
contribution”</strong></p>
<p>答案是肯定的。今天的实验显著强化了 DRPO
最核心的科学贡献：它不仅描述“负样本多会崩”，而是揭示 off-policy
负更新中一个可解析、可隔离、可干预的远场排斥机制。严格解耦证明异常梯度不是样本更差的伪影；因果干预证明远场异常梯度在受控环境中是崩溃的主要自然传导路径。</p></th>
</tr>
</thead>
<tbody>
</tbody>
</table>

但 novelty 应当精确化：负优势有害、正负不平衡和 rare failure suppression
已有相关研究，尤其 BAPO 在 DRPO 之前已经讨论负优势主导与梯度爆炸；DPG
系列在 DRPO 之后给出了高度相邻的 surprisal 视角。DRPO
最有防御力的贡献不是“第一个说负梯度危险”，而是 repulsive divergence
理论、连续策略远场几何、badness–distance 识别，以及今天补齐的因果
collapse 闭环。

因此论文完全值得围绕这套结果重写和强化理论实验章节；最合理的目标是把它写成一个清晰、可复现、可被后续方法统一解释的机制贡献，而不是依赖夸张的绝对
first 或唯一原因叙事。

综合定位：DRPO
下一阶段最值得强化的不是再增加一个“压低负优势”的经验技巧，而是建立一个跨连续与离散动作空间的
Repulsive Surprisal Dynamics，并用 stability–generalization trade-off
解释负梯度为何既能提供边界、mode suppression
与组合泛化，又会在远场因影响—信息错配而触发 collapse。若 categorical
bandit 与小型 Transformer
的严格隔离实验复现连续场景的因果链，论文将从推荐领域方法显著升级为通用
off-policy policy optimization 的机制理论。

v4 交接结论：OGPO 已提供 full-policy RL 突破 BC 支持和负优势压制坏 mode
的强外部证据，但没有隔离负梯度的直接外推贡献，也没有解释从有益外推到远场失稳的临界转变。下一阶段应优先完成负梯度泛化实验，并以
TOPR、Low-Probability Tokens/BAPO、Negative Reinforcement/Good Actions
三条前序工作线构建 related work；categorical bandit
作为第二优先级，用于把连续距离与离散 surprisal 统一到 Repulsive
Surprisal Dynamics。 新会话必须优先继承第 0
节的两套环境区分、锁定结论与实验优先级。

# 附录 A　主要数值结果

| **方法**      | **最终 reward** | **保持率** | **崩溃率** | **最终径向漂移** |
|---------------|-----------------|------------|------------|------------------|
| Baseline      | 0.201           | 26.3%      | 95%        | 5.760            |
| Near-zero     | 0.195           | 25.4%      | 90%        | 5.778            |
| Far-to-near   | 0.285           | 37.1%      | 65%        | 0.894            |
| Far-zero      | 0.618           | 80.5%      | 0%         | 0.173            |
| Far-cap       | 0.666           | 86.7%      | 0%         | 0.533            |
| Global-scale  | 0.763           | 99.3%      | 0%         | 0.210            |
| Positive-only | 0.782           | 101.8%     | 0%         | 0.011            |

# 附录 B　参考文献与相关工作

\[1\] Baheti et al. (2023/2024). Leftover Lunch: Advantage-based Offline
Reinforcement Learning for Language Models. arXiv:2305.14718.
[<u>链接</u>](https://arxiv.org/abs/2305.14718)

\[2\] Xi et al. (2025). BAPO: Stabilizing Off-Policy Reinforcement
Learning for LLMs via Balanced Policy Optimization with Adaptive
Clipping. arXiv:2510.18927.
[<u>链接</u>](https://arxiv.org/abs/2510.18927)

\[3\] Jiang, Huo et al. (2026). Breaking the Curse of Repulsion:
Optimistic Distributionally Robust Policy Optimization for Off-Policy
Generative Recommendation. arXiv:2602.10430.
[<u>链接</u>](https://arxiv.org/abs/2602.10430)

\[4\] Osband (2026). Delightful Policy Gradient. arXiv:2603.14608.
[<u>链接</u>](https://arxiv.org/abs/2603.14608)

\[5\] Osband (2026). Delightful Distributed Policy Gradient.
arXiv:2603.20521. [<u>链接</u>](https://arxiv.org/abs/2603.20521)

\[6\] Peng et al. (2019). Advantage-Weighted Regression: Simple and
Scalable Off-Policy Reinforcement Learning. arXiv:1910.00177.
[<u>链接</u>](https://arxiv.org/abs/1910.00177)

\[7\] Kostrikov et al. (2021/2022). Offline Reinforcement Learning with
Implicit Q-Learning. arXiv:2110.06169.
[<u>链接</u>](https://arxiv.org/abs/2110.06169)

\[8\] YSS et al. (2026). A Gradient Perspective on RLVR Stability and
Winner Advantage Policy Optimization (WAPO). arXiv:2606.16154.
[<u>链接</u>](https://arxiv.org/abs/2606.16154)

\[9\] Luo et al. (2026). STARE: Surprisal-Guided Token-Level Advantage
Reweighting for Policy Entropy Stability. arXiv:2606.19236.
[<u>链接</u>](https://arxiv.org/abs/2606.19236)

\[10\] Tian, Xie, Wei (2026). How Off-Policy Can GRPO Be? Mu-GRPO for
Efficient LLM Reinforcement Learning. arXiv:2605.17570.
[<u>链接</u>](https://arxiv.org/abs/2605.17570)

\[11\] Liu et al. (2026). ASymPO: Asymmetric-Scale Policy Optimization
for Asynchronous LLM Post-Training Without Behavior Information.
arXiv:2606.03070. [<u>链接</u>](https://arxiv.org/abs/2606.03070)

\[12\] Le Roux et al. (2025). Tapered Off-Policy REINFORCE: Stable and
Efficient Reinforcement Learning for LLMs. arXiv:2503.14286.
[<u>链接</u>](https://arxiv.org/abs/2503.14286)

\[13\] Zhu et al. (2025). The Surprising Effectiveness of Negative
Reinforcement in LLM Reasoning. arXiv:2506.01347.
[<u>链接</u>](https://arxiv.org/abs/2506.01347)

\[14\] Song (2025). Good Actions Succeed, Bad Actions Generalize: A Case
Study on Why RL Generalizes Better. arXiv:2503.15693.
[<u>链接</u>](https://arxiv.org/abs/2503.15693)

\[15\] OGPO authors (2026). OGPO: Sample Efficient Full-Finetuning of
Generative Control Policies. arXiv:2605.03065.
[<u>链接</u>](https://arxiv.org/abs/2605.03065)

\[16\] Luo et al. (2025/2026). CE-GPPO: Coordinating Entropy via
Gradient-Preserving Clipping Policy Optimization. arXiv:2509.20712.
[<u>链接</u>](https://arxiv.org/abs/2509.20712)

\[17\] Qi et al. (2025). Do Not Let Low-Probability Tokens Over-Dominate
in RL for LLMs. arXiv:2505.12929.
[<u>链接</u>](https://arxiv.org/abs/2505.12929)

**文献定位说明：**上述 novelty 判断基于截至 2026-06-22 的
arXiv/primary-source
检索，不等同于穷尽全部未公开稿件、会议匿名投稿或所有领域文献。正式投稿前应再执行一次系统检索与逐篇
related-work 对照。

\[18\] Zhai et al. (2026). Rewards as Labels: Revisiting RLVR from a
Classification Perspective. arXiv:2602.05630.
[<u>arXiv</u>](https://arxiv.org/abs/2602.05630)

\[19\] Xu and Fang (2026). Beyond Negative Rollouts: Positive-Only
Policy Optimization with Implicit Negative Gradients. arXiv:2605.06650.
[<u>arXiv</u>](https://arxiv.org/abs/2605.06650)

\[20\] Authors (2025). On the Optimization Dynamics of RLVR: Gradient
Gap and Token-Level Dynamics. arXiv:2510.08539.
[<u>arXiv</u>](https://arxiv.org/abs/2510.08539)

# 附录 C　新会话核心文件索引

| **文件**       | **路径**                                                                 | **用途**                                    |
|----------------|--------------------------------------------------------------------------|---------------------------------------------|
| 当前交接文档   | /mnt/data/Far_Field_Negative_Gradient_DRPO_Research_Note_v4_Handoff.docx | 新会话应首先阅读                            |
| 乘积流形代码   | /mnt/data/product_manifold_gradient_decomposition.py                     | badness–distance 严格解耦与 16×→24.95× 分解 |
| 乘积流形结果   | /mnt/data/pmgd_check_1024/near_far_summary.csv                           | 近远场数值、score 与 coherence 分解         |
| 因果干预代码   | /mnt/data/causal_farfield_solid/causal_farfield_intervention.py          | 动态 near/far 干预与预算匹配                |
| 因果结果汇总   | /mnt/data/causal_farfield_solid/summary_with_ci.csv                      | 20-seed CI、崩溃率与方法结果                |
| 完整因果实验包 | /mnt/data/causal_farfield_solid_bundle.zip                               | 代码、曲线、逐 seed 结果与 README           |

# 附录 D　P0 可学习方差：闭式稳态与快速实验验证

实验采用 fixed-advantage naive PG，无 value/Q network、importance
sampling 或动态重权。正样本分布均值 0、标准差 1.2；负样本分布均值
−1、标准差 0.2；未见评估最优动作 a\*=1。

## D.1 直接参数化 (μ,log σ) 的解析—数值一致性

稳定区内数值解与 μ\*、σ²\* 闭式解的最大误差约为 2×10⁻¹⁵。均值临界点为
ρ_mean=αN/P=1，但方差内部固定点在 ρ_var≈0.586187 处先消失。

| **α=ρ** | **最终 μ** | **最终 σ** | **判定**               |
|---------|------------|------------|------------------------|
| 0.00    | 0.000      | 1.200      | Positive-only 联合稳态 |
| 0.25    | 0.333      | 1.209      | 稳定外推               |
| 0.50    | 1.000      | 0.917      | 到达未见最优点         |
| 0.56    | 1.273      | 0.574      | 稳定但过度外推         |
| 0.58    | 1.381      | 0.292      | 接近方差临界点         |
| 0.60    | 1.500      | ≈10⁻⁵      | 均值仍有限，但方差坍缩 |

## D.2 距离移动方差临界边界

将负锚点距离从 0.5 增大到 2.0，预测的方差临界点从 0.849 降至
0.263，而均值临界点保持为 1；每个理论边界上下的 spot simulation
均与预测一致。该结果直接说明 distance 通过二阶残差 M₋
提前压缩安全负梯度区间。

| **负样本距离 d** | **ρ_var** | **ρ_mean** |
|------------------|-----------|------------|
| 0.50             | 0.849     | 1.000      |
| 0.75             | 0.715     | 1.000      |
| 1.00             | 0.586     | 1.000      |
| 1.25             | 0.476     | 1.000      |
| 1.50             | 0.388     | 1.000      |
| 2.00             | 0.263     | 1.000      |

## D.3 非线性参数化与控制方法

单状态双输出 MLP 在 3 个初始化下复现相同转变：α≤0.58
跟随解析稳态，α=0.60 时 3/3 seeds 出现方差坍缩。原始 α=0.8
的不受控训练坍缩；将有效 ρ 全局缩放到 0.5，或使用 detached
standardized-distance cap，均恢复至 μ≈1、σ≈0.917、评估
reward≈1。该结果证明 α 可通过降低 qM₋ 恢复联合稳态，distance control
则对二次远场项进行选择性控制。

## D.4 原始 gradient-explode 代码诊断

原代码的 good-only 配置只应用正样本更新，坏样本仅用于 phantom gradient
monitor。复现结果显示坏/好总梯度比约 72.3×、log-σ 分支梯度比约
55.5×；但实际 σ 从约 0.606 收缩到约 0.177。打开正负混合训练后，σ
进一步收缩到约 0.008。故原 Figure 2(b)
应解释为远场梯度敏感度扩张，而不是实际 σ 扩张。

## D.5 论文写作结论

修正后的主线是：负优势在均值分支产生位置排斥；在方差分支，近场负样本扩大支持、远场负样本收缩支持。远场中的
d↑ 与 σ↓ 共同放大标准化距离和
score，形成比固定方差更强的自增强。完整策略稳定性由联合固定点存在条件与
signed off-policy dynamics Jacobian 决定。

# 附录 E　统一非线性 benchmark：论文级正式结果

统一 benchmark 的目的不是把不同识别问题混成一个实验，而是让三个 protocol
共用同一策略类、环境接口、训练循环、梯度诊断和统计管线。正式代码已使用
seeds 10–29 完成来源与因果实验，并对 P0 稳定外推及可学习方差相变进行
held-out 检验。

## E.1 Protocol A：严格来源隔离

| **阶段**            | **\|A\| 远/近** | **score 远/近（95% CI）** | **单样本梯度** | **聚合梯度** |
|---------------------|-----------------|---------------------------|----------------|--------------|
| initialization      | 1.000           | 45.13 \[43.30,46.95\]     | 47.78          | 61.56        |
| positive_pretrained | 1.000           | 38.02 \[37.11,38.96\]     | 38.64          | 82.08        |

解释：advantage magnitude 在所有半径上结构相同，统一 actor
中远场梯度仍出现数量级放大。聚合比高于单样本比，说明方向一致性进一步放大净更新。

## E.2 Protocol B：远场 collapse 因果干预

| **方法**      | **最终 reward（95% CI）** | **保持率** | **崩溃** | **最终 \|μr\|** |
|---------------|---------------------------|------------|----------|-----------------|
| baseline      | 0.201 \[0.165,0.239\]     | 26.3%      | 19/20    | 5.760           |
| near_zero     | 0.195 \[0.162,0.232\]     | 25.4%      | 18/20    | 5.778           |
| far_to_near   | 0.285 \[0.202,0.374\]     | 37.1%      | 13/20    | 0.894           |
| far_zero      | 0.618 \[0.596,0.639\]     | 80.5%      | 0/20     | 0.173           |
| far_cap       | 0.666 \[0.653,0.680\]     | 86.7%      | 0/20     | 0.533           |
| global_scale  | 0.763 \[0.753,0.773\]     | 99.3%      | 0/20     | 0.210           |
| positive_only | 0.782 \[0.771,0.793\]     | 101.8%     | 0/20     | 0.011           |

配对结论：Far-zero − Baseline = +0.417，Far-cap − Baseline =
+0.465，Global-scale − Baseline = +0.562，均为 20/20 胜出且 Wilcoxon
p=1.91×10⁻⁶；Near-zero − Baseline = −0.006，p=0.62。

<img src="/mnt/data/master_recovery/media/media/image4.png"
style="width:6.4in;height:3.85433in" />

*图 E1　统一代码 20-seed 因果干预曲线：Baseline 与 Near-zero
重合，Far-zero/Far-cap/Global-scale 稳定。*

## E.3 Protocol C：稳定外推与联合均值—方差相变

| **方差**   | **α** | **β**  | **held-out reward（95% CI）** | **状态计数**                        |
|------------|-------|--------|-------------------------------|-------------------------------------|
| 固定方差   | 0.00  | -0.000 | 0.085 \[0.084,0.086\]         | 稳定:10                             |
| 固定方差   | 0.50  | 0.897  | 0.837 \[0.823,0.849\]         | 稳定:10                             |
| 固定方差   | 0.75  | 2.753  | 0.002 \[0.001,0.003\]         | 稳定:6 / 慢漂移:4                   |
| 固定方差   | 1.00  | —      | —                             | 均值发散:9 / 慢漂移:1               |
| 可学习方差 | 0.00  | -0.000 | 0.085 \[0.085,0.085\]         | 稳定:20                             |
| 可学习方差 | 0.50  | 0.782  | 0.709 \[0.700,0.718\]         | 稳定:20                             |
| 可学习方差 | 0.65  | 1.392  | 0.258 \[0.247,0.269\]         | 稳定:13 / 慢漂移:7                  |
| 可学习方差 | 0.68  | —      | —                             | 方差坍缩:16 / 均值发散:3 / 慢漂移:1 |
| 可学习方差 | 0.70  | —      | —                             | 方差坍缩:20                         |

关键结果：Positive-only 的 held-out reward 约
0.085；适度负梯度将策略推过正样本边界并提升 OOD
reward；固定方差先出现稳定过度外推，再在 α≈1
附近失去均值稳态；可学习方差则在 α≈0.65–0.68 已进入联合失稳区。

<img src="/mnt/data/master_recovery/media/media/image5.png"
style="width:6.4in;height:3.89475in" />

*图 E2　held-out reward 的倒 U
型：适度负梯度有益，过强负梯度先过度外推、后崩溃。*

<img src="/mnt/data/master_recovery/media/media/image6.png"
style="width:6.4in;height:3.77652in" />

*图 E3　可学习方差的经验相变早于固定方差的均值临界边界。*

## E.4 控制方法与等预算识别

| **方法**   | **β**  | **reward（95% CI）**  | **σ** | **平均负权重** |
|------------|--------|-----------------------|-------|----------------|
| 不受控     | 29.466 | 0.000 \[0.000,0.000\] | 1.006 | 1.000          |
| 仅正样本   | -0.000 | 0.085 \[0.084,0.085\] | 1.200 | 0.000          |
| 全局缩放   | 0.791  | 0.719 \[0.710,0.728\] | 1.374 | 0.556          |
| 等预算全局 | 0.798  | 0.725 \[0.716,0.735\] | 1.374 | 0.554          |
| 距离截断   | 0.827  | 0.747 \[0.737,0.756\] | 1.376 | 0.563          |

Distance cap 相对 budget-matched global 的配对增益为 +0.021
\[0.019,0.023\]，20/20 胜出，p=1.91×10⁻⁶。该对照排除了“distance
仅仅因为总梯度更小”这一解释，但结论仍限定于当前受控 benchmark。

<img src="/mnt/data/master_recovery/media/media/image7.png"
style="width:6.5in;height:2.54873in" />

*图 E4　全局与距离控制均恢复有限有益稳态；等预算对照下 distance cap
在本环境中仍有小幅稳定优势。*

## E.5 论文呈现与代码索引

- 主文最低充分图组：来源分解图；20-seed Near/Far 因果曲线与最终
  CI；稳定外推 reward
  相图；联合方差相变图；不受控/global/budget-matched/distance 控制图。

- 主文 claim：远场 score geometry
  是异常负梯度的独立来源；在受控环境中该异常梯度是 OOD drift/collapse
  的主要传导路径；适度负梯度突破 imitation
  ceiling；可学习方差引入更早的稳定边界；distance cap
  在等预算下可选择性保留外推。

- 禁止升级的 claim：所有真实任务仅由该机制崩溃；Distance 普遍优于 global
  α；当前倍率是普适常数；组合泛化已经得到证明。

- 完整复现包：/mnt/data/unified_repulsive_dynamics/results/Unified_Repulsive_Dynamics_Paper_Results.zip

- 论文级摘要：PAPER_READY_SUMMARY.md；LaTeX
  表格：paper_tables.tex；正式入口：python run_paper.py --mode paper
  --rerun-collapse；安装检查：python run_paper.py --mode smoke。


---

# Part III. v9 Exponential-Family 核心理论补丁（完整保留）

> 本节保留 v9 理论正文。它是原 DRPO repulsive dynamics 的统一数学升级，不是另起一套与原理论无关的框架。涉及过多符号的部分在后续论文精简时调整，但研究主文档不删除。

# 2. 大一统理论：Repulsive Signed-Moment Dynamics

## 2.1 研究对象、记号与结论层级

对每个状态 s 条件化后，把正优势和负优势样本分别写成加权分布 P₊(a\|s)、P₋(a\|s)。令正质量 p(s)=E\[A₊\|s\]，负质量 q(s)=E\[(-A)₊\|s\]；全局 α、样本权重或方法控制均被吸收到 q 和 P₋ 中。基础理论先假设 actor step 内 advantage stop-gradient，随后再讨论 value/Q 随时间变化。

$$
J(\theta)=\mathbb{E}_{\mathcal D}[A(s,a)\log\pi_\theta(a\mid s)],\qquad F(\theta)=\nabla_\theta J(\theta)=\mathbb{E}_{\mathcal D}[A\nabla_\theta\log\pi_\theta(a\mid s)]
$$

理论分成三层：第一层是任意可微策略都成立的单样本 surprisal 递推；第二层是在正则最小指数族中成立的 signed-moment 平衡定理；第三层才是 Gaussian、categorical、神经网络与具体控制方法的分叉推论。这样既保留 general 形式，也避免把 expected Fisher 当成固定样本动力学。

## 2.2 定理 1：单个负优势更新必然提高该样本 surprisal

令 z=(s,a)，Sθ(z)=−logπθ(a\|s)，gθ(z)=∇θlogπθ(a\|s)。对固定负优势 A(z)=−c\<0，单样本梯度上升为：

$$
\theta^+=\theta-hc\,g_\theta(z),\qquad h>0
$$

对 Sθ 做二阶 Taylor 展开，存在位于 θ 与 θ⁺ 之间的 θ̃，使：


$$
S_{\theta^{+}}(z)-S_\theta(z)=hc\lVert g_\theta(z)\rVert^2+\frac12 h^2c^2 g_\theta(z)^\top\!\left[\nabla^2 S_{\tilde\theta}(z)\right]g_\theta(z)
$$


若该线段上 ‖∇²S‖op≤L，则：


$$
S_{\theta^{+}}-S_\theta\ge hc\lVert g_\theta\rVert^2\left(1-\frac12hcL\right)
$$


因此当 hcL\<2 时，surprisal 严格增加。连续时间梯度流 θ̇=−c gθ 下更有精确恒等式：


$$
\frac{dS_\theta(z)}{dt}=c\lVert g_\theta(z)\rVert^2\ge 0
$$


这一定理是连续与离散的共同主干：负更新不是“静态降低概率”，而是把同一样本沿当前策略的 score geometry 持续推向更低支持。

## 2.3 批量更新：自项、跨样本干涉与方向一致性

单样本单调性不能无条件提升为“batch 中每个负样本 surprisal 都单调增加”。令 batch field F=ΣⱼAⱼgⱼ，则样本 i 的首阶变化为：


$$
\Delta S_i=-h g_i^\top F+O(h^2)=h|A_i|\lVert g_i\rVert^2-h\sum_{j\ne i}A_j\langle g_i,g_j\rangle+O(h^2)
$$


第一项是负样本自身的确定性排斥；第二项是正负样本共享参数带来的 interference。远场风险因此不仅取决于单样本 scale，还取决于梯度方向是否相干。本文实验中的 aggregate amplification 正是在单样本 score 放大之外叠加了 coherence。


$$
\text{Repulsive influence}\approx\text{negative mass}\times\text{score scale}\times\text{directional coherence}\times\text{repeated reuse}
$$


## 2.4 定理 2：正则最小指数族中的 signed-moment 平衡

考虑固定状态下的正则最小指数族：


$$
\pi_\eta(a)=h(a)\exp\!\left\{\eta^\top T(a)-\psi(\eta)\right\}
$$


令 t₊=E\_{P₊}\[T(a)\]、t₋=E\_{P₋}\[T(a)\]，w=p−q。则 signed policy objective 可精确写为：


$$
J(\eta)=(p t_+-q t_-)^\top\eta-(p-q)\psi(\eta)+C=w\left[\tau^\top\eta-\psi(\eta)\right]+C
$$



$$
\tau=\frac{p t_+-q t_-}{p-q}
$$


其梯度和 Hessian 为：

$$
\nabla_\eta J=w[\tau-m(\eta)],\qquad m(\eta)=\mathbb E_{\pi_\eta}[T(a)]
$$


$$
\nabla_\eta^2J=-w\,\operatorname{Cov}_{\pi_\eta}[T(a)]
$$


由此得到统一结论：

- 若 w\>0 且 signed target τ 位于指数族 mean-parameter domain 的内部，则存在唯一有限平衡 η\*，满足 m(η\*)=τ；在可识别子空间上 Hessian 负定，平衡局部渐近稳定。

- 若 τ 位于 mean-domain 边界，最优分布只能在边界上实现，通常需要自然参数趋于无穷；这对应 Gaussian 的零方差边界或 categorical 的零概率支持。

- 若 τ 落在可行域之外，或 w≤0，则不存在有限内部平衡；目标可能无界，或动力学向参数/分布边界逃逸，具体表型由策略族决定。

- 离散 Euler 更新在平衡附近的充分步长条件是 ρ(I+hJ)\<1；指数族自然参数下可写为 h \< 2/\[w λmax(Covπ\*\[T\])\]。

这个定理把“稳定外推”和“崩溃”统一成一个几何问题：负优势把正样本的 moment target 沿远离负样本的方向外推；只要外推后的 signed target 仍位于可行 moment 域内，就存在稳定解；一旦越界，内部固定点消失。

| **策略族**                   | **充分统计 T(a)** | **mean-domain** | **越界表型**                                                   |
|------------------------------|-------------------|-----------------|----------------------------------------------------------------|
| 固定方差 Gaussian            | a                 | 整个实数空间    | p≤q 时均值漂移或 runaway                                       |
| 可学习方差 Gaussian          | (a, a²)           | m₂\>m₁²         | signed variance≤0，σ→0 或联合失稳                              |
| full softmax categorical     | one-hot eₐ        | 概率单纯形      | 某些 signed probability≤0，logit gap→∞                         |
| feature / energy categorical | 动作特征 φ(a)     | 特征凸包内部    | 目标 feature moment 越界或贴边，support / temperature collapse |

## 2.5 Gaussian 推论 A：固定方差下的稳定外推与均值相变

对 π=N(μ,σ²)，固定 σ。设正负动作均值为 m₊、m₋，有效质量为 p、q。均值动力学为：


$$
\dot\mu=\frac{p(m_+-\mu)-q(m_--\mu)}{\sigma^2}
$$


当 p\>q 时存在稳定点：

$$
\mu^*=\frac{pm_+-qm_-}{p-q},\qquad \mu^*-m_+=\frac{q(m_+-m_-)}{p-q}
$$

若 m₋\<m₊，负样本位于正样本另一侧，则 μ\*\>m₊：负梯度把策略稳定推到最佳正样本支持之外。若真实最优为 a\*\>m₊，使 μ\*=a\* 的最优负质量为：


$$
q_{\mathrm{opt}}=p\frac{a^*-m_+}{a^*-m_-}<p
$$


因此任务最优点严格位于动力学临界点 qcrit=p 之前。离散更新的误差满足：


$$
\mu_{t+1}-\mu^*=\left[1-\frac{h(p-q)}{\sigma^2}\right](\mu_t-\mu^*)
$$


稳定步长要求 0\<h(p−q)/σ²\<2。q=p 时若 m₊≠m₋，吸引与排斥曲率抵消，出现持续漂移；q\>p 时均值固定点失去稳定性并产生 runaway。

## 2.6 Gaussian 推论 B：可学习方差的联合稳态与提前失稳

令 ξ=logσ，正负条件方差分别为 v₊、v₋，并定义 M±(μ)=v±+(μ−m±)²。精确动力学为：


$$
\dot\mu=\frac{p(m_+-\mu)-q(m_--\mu)}{\sigma^2}
$$



$$
\dot\xi=\frac{pM_+(\mu)-qM_-(\mu)}{\sigma^2}-(p-q)
$$


联合内部固定点为：


$$
\mu^*=\frac{p m_+-q m_-}{p-q}
$$



$$
\sigma^{2*}=\frac{pM_+(\mu^*)-qM_-(\mu^*)}{p-q}
$$


将其化成 signed variance 可得到更清晰的可行性条件。令 Δ=m₊−m₋：


$$
\sigma^{2*}=\frac{p v_+-q v_-}{p-q}-\frac{pq\Delta^2}{(p-q)^2}
$$


因此联合稳态需要 p\>q 且 σ²\*\>0。第二个条件通常更严格，使方差边界早于均值边界。令 C=v₊+v₋+Δ²，v₋\>0 时较小正根为：


$$
q_{\mathrm{var}}=p\frac{C-\sqrt{C^2-4v_+v_-}}{2v_-}
$$


若 v₋=0，则极限为：


$$
q_{\mathrm{var}}=p\frac{v_+}{v_++\Delta^2}
$$


在联合固定点处，(μ,ξ) 动力学 Jacobian 恰好对角化：


$$
J_F(\mu^*,\xi^*)=\operatorname{diag}\!\left(-\frac{p-q}{\sigma^{2*}},-2(p-q)\right)
$$


所以只要内部解存在且 p\>q，均值和 log-std 都局部稳定；实验中观察到的“方差先坍缩”不是固定点不稳定，而是 signed target 先离开 Gaussian 可行 moment 域，使有限固定点直接消失。

## 2.7 Gaussian 推论 C：方差四象限、单样本 MLE 与远场幅度放大

$$
\frac{\partial\log\pi}{\partial\xi}=z^2-1,\qquad z=\frac{a-\mu}{\sigma}
$$

| **advantage** | **\|z\|\<1**          | **\|z\|\>1**            |
|---------------|-----------------------|-------------------------|
| A\>0          | σ下降：集中到近正样本 | σ上升：覆盖远正样本     |
| A\<0          | σ上升：摊薄近负样本   | σ下降：压缩远负样本支持 |

单个确定性正样本的 Gaussian log-likelihood 没有有限最大值：μ→a 后仍有 logπ(a)=−logσ+C→+∞，故 σ→0。只有拟合均值后仍存在非零条件残差，或加入 entropy/KL/σ-min，positive-only 才有有限方差稳态。

原 sign-only Hessian 论证的问题在此处最清楚。固定样本的 negative-log-likelihood Hessian 为：


$$
H_{\mathrm{sample}}=\begin{bmatrix}\sigma^{-2}&2(a-\mu)\sigma^{-2}\\2(a-\mu)\sigma^{-2}&2(a-\mu)^2\sigma^{-2}\end{bmatrix}
$$


$$
\det(H_{\mathrm{sample}})=-\frac{2(a-\mu)^2}{\sigma^4}<0\qquad(a\ne\mu)
$$

它是不定矩阵；只有对 a~π 取期望后才得到 Fisher / expected Hessian diag(σ⁻²,2)≻0。因此不能由 expected SPD 推出固定 off-policy 样本在 (μ,ξ) 每个方向都统一扩张。正确结论是：负样本始终排斥均值，但方差方向由 z²−1 决定。

远场幅度分叉仍然成立。Gaussian score 为：

$$
g_\mu=\frac{a-\mu}{\sigma^2}=\frac{z}{\sigma},\qquad g_\xi=z^2-1
$$


$$
\lVert g\rVert^2=\frac{z^2}{\sigma^2}+(z^2-1)^2
$$


固定 σ 且只重复一个负样本时，δₜ=μₜ−a 满足精确递推 δₜ₊₁=(1+hc/σ²)δₜ，故均值距离和 mean-score 关于训练步数几何增长。可学习方差时，远场负样本同时使 μ 远离、σ 收缩，通常进一步放大标准化距离；但不应再无条件声称 μ 与 σ 都“expand”。

## 2.7A Gaussian 远场负梯度的二次临界衰减定理（v30，已解析证明）

本节只证明同一 Gaussian 标准化距离上的控制阶数，不把 surprisal 替换成距离，也不把任务 reward 排名写进定理。考虑动作维数为 `D` 的 isotropic Gaussian：

$$
\pi_\theta(a\mid s)=\mathcal N(\mu_\theta(s),\sigma_\theta(s)^2I_D),\qquad \xi_\theta(s)=\log\sigma_\theta(s).
$$

定义当前 C-U1 Near/Far 使用的标准化距离

$$
d=d_\theta(s,a)=\frac{\lVert a-\mu_\theta(s)\rVert_2}{\sigma_\theta(s)}.
$$

固定一个负优势样本 `A(s,a)=-c<0`，并令 policy-output 坐标为 `y=(mu,xi)`。其负梯度幅度等于 `c` 乘以 Gaussian score 幅度。精确公式为

$$
\nabla_\mu\log\pi_\theta(a\mid s)=\frac{a-\mu}{\sigma^2},\qquad
\frac{\partial\log\pi_\theta(a\mid s)}{\partial\xi}=d^2-D,
$$

从而

$$
\boxed{
\lVert g_y^-(s,a)\rVert_2^2
=c^2\left[\frac{d^2}{\sigma^2}+(d^2-D)^2\right].
}
$$

### 定理 3（pre-boundary 区域中的二次远场界）

假设负优势幅度满足 `0<c_min<=c<=c_max<infinity`，并且只在 support/variance boundary 之前的正则区域讨论，即存在 `sigma_min>0` 使 `sigma_theta(s)>=sigma_min`。令 `d>=d_0=max{1,sqrt(2D)}`，则存在与 `d` 无关的正常数 `C_1,C_2`，使

$$
C_1d^2\le \lVert g_y^-(s,a)\rVert_2\le C_2d^2.
$$

可取

$$
C_1=\frac{c_{\min}}{2},\qquad
C_2=c_{\max}\sqrt{1+\sigma_{\min}^{-2}}.
$$

**证明。** 当 `d^2>=2D` 时，`d^2-D>=d^2/2`，故 log-scale 分支直接给出

$$
\lVert g_y^-\rVert_2\ge c|d^2-D|\ge\frac{c_{\min}}2d^2.
$$

另一方面，`0<=d^2-D<=d^2`，且 `d>=1`、`sigma>=sigma_min`，因此

$$
\frac{d^2}{\sigma^2}\le\frac{d^4}{\sigma_{\min}^2},\qquad(d^2-D)^2\le d^4.
$$

代回精确范数式，得到

$$
\lVert g_y^-\rVert_2^2\le c_{\max}^2d^4(1+\sigma_{\min}^{-2}).
$$

开平方即得上界，故 `||g_y^-||=Theta(d^2)`。证毕。

该定理说的是**固定时刻、同一标准化距离上的单样本输出梯度**。它不等同于 advantage 自身二次增长，也不声称神经网络全参数梯度无条件具有严格二次下界。

### 定理 4（reciprocal-polynomial 的二次临界阶）

令距离权重 stop-gradient，并定义

$$
w_p(d)=\frac{1}{1+\lambda(d/d_{\mathrm{ref}})^p},\qquad \lambda>0,\ p\ge0.
$$

在定理 3 的条件下，

$$
\boxed{\lVert w_p(d)g_y^-(s,a)\rVert_2=\Theta(d^{2-p}).}
$$

因此

$$
p<2\Rightarrow \lVert w_pg_y^-\rVert_2\to\infty,
$$

$$
p=2\Rightarrow 0<\liminf_{d\to\infty}\lVert w_pg_y^-\rVert_2\le\limsup_{d\to\infty}\lVert w_pg_y^-\rVert_2<\infty,
$$

$$
p>2\Rightarrow \lVert w_pg_y^-\rVert_2\to0.
$$

**证明。** 当 `p=0` 时，`w_0(d)=1/(1+lambda)` 为正常数，结论直接由定理 3 得到。以下设 `p>0`。对充分大的 `d`，`lambda(d/d_ref)^p>=1`，于是

$$
\frac{d_{\mathrm{ref}}^p}{2\lambda}d^{-p}
\le w_p(d)\le
\frac{d_{\mathrm{ref}}^p}{\lambda}d^{-p}.
$$

与定理 3 的 `C_1d^2<=||g_y^-||<=C_2d^2` 相乘，即得两侧同阶界 `Theta(d^{2-p})`；三种极限由 `2-p` 的符号立即得到。证毕。

**直接推论。**

$$
w_{\mathrm{lin}}(d)=\frac{1}{1+\lambda d/d_{\mathrm{ref}}}\quad\Rightarrow\quad \lVert w_{\mathrm{lin}}g_y^-\rVert=\Theta(d),
$$

仍然无界；

$$
w_{\mathrm{quad}}(d)=\frac{1}{1+\lambda(d/d_{\mathrm{ref}})^2}\quad\Rightarrow\quad \lVert w_{\mathrm{quad}}g_y^-\rVert=\Theta(1),
$$

所以二次 reciprocal 是该正值平滑多项式族中保证有界的最低阶。对

$$
w_{\exp}(d)=e^{-\lambda d/d_{\mathrm{ref}}},
$$

由 `d^2e^{-lambda d/d_ref}->0`，加权影响趋零。

### 命题 5（同一参考衰减下的有限距离选择性）

固定 `rho in (0,1)`，令 `lambda=rho^{-1}-1`、`u=d/d_ref`，则

$$
w_{\mathrm{lin}}(u)=\frac1{1+\lambda u},\qquad
w_{\mathrm{quad}}(u)=\frac1{1+\lambda u^2},
$$

并且二者均满足 `w(1)=rho`。因为 `u^2<u` 当 `0<u<1`，而 `u^2>u` 当 `u>1`，所以

$$
0<u<1\Rightarrow w_{\mathrm{quad}}(u)>w_{\mathrm{lin}}(u),
$$

$$
u=1\Rightarrow w_{\mathrm{quad}}(u)=w_{\mathrm{lin}}(u)=\rho,
$$

$$
u>1\Rightarrow w_{\mathrm{quad}}(u)<w_{\mathrm{lin}}(u).
$$

因此在相同 `d_ref` 和参考强度下，二次方法同时具有“近场保留更多、远场压制更强”的解析性质。该命题不需要渐近极限，但仍不推出任务 reward 必然更高。

### 对角 Gaussian 推论

对一般 diagonal Gaussian，令

$$
z_j=\frac{a_j-\mu_j}{\sigma_j},\qquad d=\lVert z\rVert_2,
$$

则每个 log-scale 分量为

$$
\frac{\partial\log\pi}{\partial\xi_j}=z_j^2-1.
$$

由 Cauchy--Schwarz 不等式，

$$
\frac{d^2}{\sqrt D}\le \lVert z^{\odot2}\rVert_2\le d^2.
$$

因此

$$
\frac{d^2}{\sqrt D}-\sqrt D
\le
\lVert z^{\odot2}-\mathbf 1\rVert_2
\le
 d^2+\sqrt D.
$$

当 `d^2>=2D` 时，log-scale 联合分支被上下界为常数倍 `d^2`。若各维 `sigma_j>=sigma_min>0`，mean 分支也至多为 `O(d^2)`，故 diagonal Gaussian 的联合 policy-output 梯度仍为 `Theta(d^2)`，定理 4 的临界阶 `p=2` 不变。该推论覆盖后续 state-conditioned diagonal Gaussian 外部验证；tanh-squashed actor 必须在 frozen inverse-squash/base-Gaussian 坐标中使用此距离。

### 神经网络 pullback、例外与可声明边界

令 `J_theta(s)=partial(mu_theta,xi_theta)/partial theta`，则

$$
g_\theta^-=J_\theta(s)^\top g_y^-.
$$

若研究区域内 `||J_theta(s)||_op<=M`，则

$$
\lVert w_pg_\theta^-\rVert_2\le M\lVert w_pg_y^-\rVert_2.
$$

故 `p=2` 对实际全参数单样本影响给出充分有界性；但要把 `p=2` 写成全参数空间的必要临界阶，还需 Jacobian 在 log-scale score 方向有统一非退化下界。正式 C-U1 实验直接测量实际全参数梯度，正是为了验证该 pullback 是否保留理论排序。

边界条件必须同时保留：

1. 固定方差时 log-scale 分支不存在，mean score 仅为一次阶，临界多项式阶降为 `p=1`。
2. `p=2` 的**有界上界**只需要 `|A|<=c_max`；“`p<2` 必然无界”的必要性结论还要求沿所讨论的远场序列存在 `|A|>=c_min>0`。C-U1 的等 advantage 设计正是用来隔离这一条件。
3. 若优势幅度本身满足 `|A(d)|=Theta(d^q)`，则总输出梯度阶变为 `Theta(d^{2+q})`，reciprocal-polynomial 的临界阶相应变为 `p=2+q`；当前定理的主情形是 `q=0`。
4. 若允许 `sigma->0` 且不在 pre-boundary 区域设置任何 `sigma_min`，标准化距离的二次 taper只能直接保证 log-scale 分支，不能无条件给出总 mean 分支的统一界；因此 support/variance boundary 必须单独报告。
5. 若权重不 stop-gradient，会额外出现 `grad_theta w`，本定理不适用。
6. `[1-lambda d]_+` 等 clipped-linear 在有限阈值后严格为零，属于 compact-support hard cutoff，不属于 reciprocal-linear 尾部，必须另行分析。
7. 本定理证明控制强度和有界性，不证明 Quadratic、Exp 或其他方法的任务性能排名。


## 2.8 Categorical 推论 A：有界单步 score 仍可把策略推到 simplex 边界

对 K 类 full-softmax，logits 为 z，π=softmax(z)。单独重复负更新动作 j，A=−c：


$$
\dot z=c(\pi-e_j)
$$



$$
\frac{d[-\log\pi_j]}{dt}=c\lVert e_j-\pi\rVert^2
$$


direct-logit score 有界：‖eⱼ−π‖≤√2。因此 categorical 不具备 Gaussian 式的单 token 欧氏梯度无界爆炸。但一旦 πⱼ≤ε，Cauchy 不等式给出：


$$
\lVert e_j-\pi\rVert^2\ge\frac{K}{K-1}(1-\varepsilon)^2>0
$$


所以该 token 的 surprisal 至少线性增长，概率至多指数衰减；logit gap 可以趋于无穷，分布被推到概率单纯形边界。动作集合有限并不能阻止 support collapse。

full-softmax 也是指数族，T(a)=eₐ。signed target 为：


$$
\pi^*=\frac{p r_+-q r_-}{p-q}
$$


若某个分量为 0，有限 logits 无法达到，只能令对应 logit→−∞；若某个分量为负，则 target 已离开 simplex，不存在内部解。由此得到离散版的精确 support-feasibility 边界。

Entropy 不是这一动力学的充分统计量：抑制高概率负动作时 entropy 可以先升高，抑制低概率负动作时 entropy 可直接下降；两种路径都可能最终损伤任务支持。因此 entropy control 是必要 baseline，但不能替代对具体危险负更新的选择性诊断。

## 2.9 Categorical 推论 B：未见动作外推为何需要语义结构，而不需要“动作有序”

对完全饱和的独立 logits，训练中从未出现的动作在经验 signed target 中通常为 0；纯最大似然/负强化不会凭空知道应把概率放到哪个未见动作。方向性外推必须来自共享参数、预训练先验或动作特征，而不是 token ID 顺序。

更一般地，令动作拥有任意编号和语义特征 φ(a)，使用 energy policy：


$$
\pi_\eta(a\mid s)\propto\exp\!\left\{\eta(s)^\top\phi(a)\right\}
$$


它仍是指数族，稳定点满足：


$$
\mathbb E_{\pi^*}[\phi(a)]=\frac{p\mathbb E_+[\phi(a)]-q\mathbb E_-[\phi(a)]}{p-q}
$$


负样本把目标 feature moment 推离坏动作特征；指数族的最大熵投影会把概率重新分配给具有相似语义、但可能未在正样本中出现的动作。若随机打乱 feature 与 reward 的对应关系，这种 task gain 应消失，而 support suppression 仍然存在。于是“结构破坏”对照不是为有序动作辩护，而是区分两个命题：通用的支持压制不需要结构；有益的未见动作外推需要可泛化结构。

一维 ordinal catalogue 仅保留为可解析的 T=(x,x²) 桥梁；generic categorical 的主要证据应使用随机动作 ID + semantic embedding，而不是人为数轴。

## 2.10 神经网络共享参数：指数族输出场的 pullback

令网络输出自然参数 ηθ(s)，Jacobian 为 Jθ(s)=∂ηθ(s)/∂θ。输出空间残差为 r_s(η)=p_s t₊(s)−q_s t₋(s)−(p_s−q_s)m(η)。参数场为：


$$
F_\theta=\mathbb E_s\!\left[J_\theta(s)^\top r_s(\eta_\theta(s))\right]
$$


若存在可实现的 moment-matching 解，使每个相关状态 r_s=0，则网络二阶项在固定点消失，局部 Jacobian 为：


$$
J_F(\theta^*)=-\mathbb E_s\!\left[(p_s-q_s)J_\theta(s)^\top\operatorname{Cov}_{\pi^*}[T]J_\theta(s)\right]
$$


在 p_s\>q_s 且聚合 feature-Fisher 对可训练参数子空间满秩时，该矩阵负定，得到局部稳定性。若多个状态的 signed targets 不能被同一网络同时实现，或固定点残差不为零，网络二阶项重新出现；此时只能使用一般 signed-field Jacobian，而不能声称全局凸性或唯一解。

这一推导说明矩阵形式完全可以保留：真正 general 的对象是 signed off-policy field Jacobian，而不是把 on-policy expected Fisher 直接当作固定样本转移矩阵。

## 2.11 方法推论：Global α、Exp-remoteness 与 stability budget

Global α 只改变总负质量 q，简单、稳定，但会无差别削弱近场有用信息。选择性方法令负样本权重依赖当前 policy-relative remoteness。定义连续/离散统一的 remoteness：

$$
S_i=-\log\pi_\theta(a_i\mid s_i),\qquad c_\lambda(S_i)=\exp\{-\lambda(S_i-S_0)_+\}
$$

实现时对 cλ stop-gradient，保证它是纯重权而不是额外可微正则。单负样本的首阶 surprisal 速度变为：

$$
\frac{dS}{dt}=|A|c_\lambda(S)\kappa(S),\qquad \kappa(S)=\lVert\nabla\log\pi\rVert^2
$$

若远场 κ(S) 至多多项式增长，或更一般满足 κ(S)≤Cexp(βS)，则 λ\>β 时加权 influence 有界并在远场衰减。固定方差 Gaussian 的 κ 为 O(S)，含 log-variance 的标准化远场为 O(S²)；direct-logit categorical 的 κ≤2。因此 Exp-remoteness 有一个比“梯度关于距离指数增长”更准确的故事：指数 taper 支配有限阶 score growth，并统一为 categorical 中的 π(a)^λ。

更强的 stability-budget 方法直接使用定理 2 的可行性：先经 cλ 重权得到有效 q_c 与 t₋,c，再选择最大的 batch 系数 γ∈\[0,1\]，使 signed target 保持在 mean-domain 的安全内点。

$$
\gamma^*=\max\{\gamma\in[0,1]:p-\gamma q_c\ge\varepsilon_{\mathrm{mass}},\ \operatorname{dist}(\tau(\gamma),\partial\mathcal M)\ge\varepsilon_{\mathrm{geom}}\}
$$

Gaussian 中可用 p−γq_c\>0 与 σ²\*(γ)≥σ²min 两个闭式条件，计算只需 batch reductions；full-softmax 可约束所有 signed probabilities≥ε。一般 feature policy 的凸包距离较难精确计算，因此 SBRC-Lite 只能使用 score/moment proxy，理论保证相应减弱。

## 2.12 Learned critic / value network：瞬时适用与移动目标

在 DRPO-Q、IQL 或一般 actor-critic 中，A_t=Qφ(s,a)−Vψ(s) 会随 critic 更新。只要 actor step 使用 A_t.detach()，上述理论对每一步的瞬时 signed field 仍成立；但整个系统变成非自治动力学，不能把固定 advantage 的全局固定点直接照搬。

若每一时刻都存在内部目标 η\*(t)，局部收缩率下界为 m\>0，且目标漂移速度 ‖η̇\*(t)‖≤v，则标准移动平衡分析给出 tracking error 的量级：


$$
\limsup_t\lVert\eta(t)-\eta^*(t)\rVert\le\frac{v}{m}
$$


因此 critic 越慢、稳定裕度越大，actor 越能跟踪；但任何梯度控制都不能修复 critic 给错 advantage 符号的问题，只能限制错误信号被 score geometry 放大的破坏。

## 2.13 自我审查：反例挑战、修正与最终可声明边界

| **挑战**                                       | **审查结果**                                                     | **最终处理**                                                            |
|------------------------------------------------|------------------------------------------------------------------|-------------------------------------------------------------------------|
| 单负样本 surprisal 是否在 batch 中仍必增？     | 否；跨样本 Gram 项可反转。                                       | 定理限定为单样本/隔离更新；batch 使用 interference 分解。               |
| expected Fisher SPD 能否证明固定样本联合扩张？ | 不能；pointwise Hessian 一般不定。                               | 以 signed field Jacobian 和指数族 Hessian 取代。                        |
| Gaussian 负样本是否总使 σ 增大？               | 否；far negative 使 σ下降，near negative 使 σ上升。              | 保留 z²−1 四象限，删除 both μ and σ expand。                            |
| 正样本非确定是否自动保证有限 σ？               | 仅当拟合状态后仍有非零条件残差。                                 | 把条件残差或 entropy/KL/σ-min 写成必要来源。                            |
| 有限 categorical 是否不会发散？                | 动作有限，但 logit gap 无界，概率可到 simplex 边界。             | 区分 amplitude runaway 与 support runaway。                             |
| rare token 的 direct-logit score 是否无界？    | 否，范数≤√2。                                                    | 只声称持续 suppression；Fisher 内禀范数与 SGD 梯度分开。                |
| 负优势是否必然带来未见动作泛化？               | 否；无结构 independent logits 不知道往哪里分配。                 | 外推需共享表示/动作特征；加入结构破坏对照。                             |
| entropy 是否等价于 support quality？           | 否；同一 entropy 可对应不同任务支持。                            | entropy control 仅作为 baseline，不作为机制替代。                       |
| Exp 是否由“距离指数增长”直接推出？             | 不完全；score 对距离多为线性/二次。                              | 改为指数 taper 支配多项式 score growth 的有界性论证。                   |
| 指数族全局结论能否直接套神经网络？             | 不能；共享网络可能不可实现，且非凸。                             | 只在 realizable fixed point 给 pullback 局部稳定；其余用一般 Jacobian。 |
| Adam / PPO / importance ratio 是否被定理覆盖？ | 当前定理直接覆盖 gradient flow / Euler 和 detached reweighting。 | 其他优化器、ratio clipping 作为经验扩展，不写成严格推论。               |
| information 随距离下降是否已证明？             | 尚未；需要任务结构和方向可靠性假设。                             | 保留为可检验 hypothesis，不列为已证定理。                               |
| 边界/低熵是否必然导致任务 reward collapse？    | 不必；若边界动作恰为最优可提升。                                 | 区分 support collapse 与 task collapse，后者需环境因果实验。            |

自审结论：目前没有发现会推翻主框架的逻辑缺口。可以严格成立的是“单样本排斥恒等式 + 指数族 signed-moment 可行性 + Gaussian/categorical 分叉 + 局部神经网络 pullback”。仍不能升级为定理的是“方向信息必随距离单调下降”“任意真实任务都由该机制唯一导致 collapse”以及“某一种控制在所有任务上必胜”。

---

# Part IV. v10 Hopper Learned-Critic 外部验证记录（完整保留，状态降级）

> 以下结果保留为有限训练步数的 learned-critic mechanism probe。600 optimization steps 未达到长期收敛，不能作为最终动力学或方法结论。

# DRPO / SNA2C 统一理论与实验备忘录 v9

> Markdown 公式修复版。所有核心公式均改写为标准 LaTeX；正文、表格和图片来源于 v9 DOCX。


本版本替代 v8 作为新会话首读文档。

目标：在保留已锁定实验结论的基础上，用可证明的 signed-moment / surprisal 动力学替换旧的 sign-only Hessian 叙事，并明确神经网络、critic 与方法设计的适用边界。

| **状态**                | **结论**                                                    |
|-------------------------|-------------------------------------------------------------|
| 连续受控机制            | 已完成并达到论文级别                                        |
| 离散 categorical bandit | 小环境、理论边界、多状态统一环境与 20-seed 因果干预均已完成 |
| 理论修正                | 原 expected-Fisher SPD / “μ 与 σ 同时扩张”表述已撤回并替换  |
| 下一主任务              | 方法创新 + 外部有效性；小型 Transformer/token 验证次之      |

# 0. 新会话必须继承的锁定结论

| **【锁定】乘积流形/来源隔离回答“大梯度从哪里来”；非线性因果干预回答“这些梯度是否导致漂移与崩溃”。两类问题不能混淆。** |
|-----------------------------------------------------------------------------------------------------------------------|

## 0.1 连续 Gaussian

- 远场异常负梯度的主要来源是 policy-score geometry，而不是远场样本 advantage 更差。

- 在连续非线性因果环境中，far-zero / far-cap 稳定救援，near-zero 无效；远场异常负梯度是受控环境中 OOD 漂移与 collapse 的主导传导路径。

- Positive-only 是无排斥参考：稳定但停在最佳正样本支持附近，存在 imitation ceiling。

- 适度负梯度能够形成数据支持之外的有益稳态；更强负梯度先造成稳定过度外推，再造成动力学失稳。

- 可学习方差引入更早的联合稳定边界：远场负样本导致支持收缩，precision 进一步放大均值与方差梯度。

- Global α 与 distance-aware control 都是有效稳定机制；不能宣称 distance 在所有任务上必然优于 α。

## 0.2 离散 categorical

- 直接 softmax logit 的单样本 score norm 有界；离散通用病理是 surprisal 持续增长与支持集/熵坍缩，而不是连续 Gaussian 同形式的无界幅度爆炸。

- 负优势对 entropy 的影响由当前概率/rarity 决定：打压高概率负动作最初提高熵，打压低概率负动作降低熵。

- 在具有有序 action catalogue 与参数共享的 categorical energy policy 中，负梯度同样可突破 positive-only 支持上限；任意独立 logits 不自动具备这一性质。

- 离散 signed-moment 可行边界 α≈0.585；α=0.58 稳定，α=0.62 在 20/20 seeds 中 temperature collapse。

- 保留 far negatives、删除 near negatives 仍在 20/20 seeds 中造成 task + support collapse；far-zero / far-cap 在 20/20 seeds 中同时阻止两类 collapse。

## 0.3 明确不可声称

- 远场机制是所有真实任务的唯一 collapse 原因。

- distance control 普遍优于 global α。

- 当前 held-out state 实验等于组合泛化。

- categorical energy policy 的结论可无条件推广到任意独立 softmax logits。

- 当前结果已经验证 Transformer / LLM token dynamics。

# 1. 文档审计与结论固化

## 1.1 v7 的主要问题

- 内容按研究时间顺序堆叠：旧路线、开发实验、正式实验和文献笔记并列，核心证据被淹没。

- C1/C2/V0 等小环境与统一 benchmark 同时保留为主结果，造成数值重复和 claim 层级不清。

- 原 DRPO 的 expected-Fisher SPD 证明、方差“扩张”表述与修正后的联合动力学并存，存在内部冲突。

- “下一阶段计划”重复多版，包含已经完成的 P0 与 categorical bandit。

## 1.2 v8 的删除与替换规则

| **内容**                          | **v8 处理**                                          |
|-----------------------------------|------------------------------------------------------|
| C1 标量、C2 MLP、V0/V1 小实验     | 保留为开发 sanity check；不再作为正文并行主证据      |
| 旧乘积流形与旧因果环境详细数值    | 仅保留职责与历史复现说明；正式主表改用统一 benchmark |
| “μ 与 σ 同时扩张”                 | 删除；改为“均值排斥 + 远场支持收缩”                  |
| sign-only joint expansion theorem | 删除；改为 signed off-policy field Jacobian          |
| 多版路线规划                      | 压缩成第 7 节唯一待办列表                            |
| 逐篇 related-work 长笔记          | 移出核心 handoff；正文只保留统一解释位置             |

| **【边界】旧小环境的原始代码和结果仍应归档，作用是调试、闭式验证和审稿 rebuttal 备份；论文主文不重复展示。** |
|--------------------------------------------------------------------------------------------------------------|

# 2. 大一统理论：Repulsive Signed-Moment Dynamics

## 2.1 研究对象、记号与结论层级

对每个状态 s 条件化后，把正优势和负优势样本分别写成加权分布 P₊(a\|s)、P₋(a\|s)。令正质量 p(s)=E\[A₊\|s\]，负质量 q(s)=E\[(-A)₊\|s\]；全局 α、样本权重或方法控制均被吸收到 q 和 P₋ 中。基础理论先假设 actor step 内 advantage stop-gradient，随后再讨论 value/Q 随时间变化。

$$
J(\theta)=\mathbb{E}_{\mathcal D}[A(s,a)\log\pi_\theta(a\mid s)],\qquad F(\theta)=\nabla_\theta J(\theta)=\mathbb{E}_{\mathcal D}[A\nabla_\theta\log\pi_\theta(a\mid s)]
$$

理论分成三层：第一层是任意可微策略都成立的单样本 surprisal 递推；第二层是在正则最小指数族中成立的 signed-moment 平衡定理；第三层才是 Gaussian、categorical、神经网络与具体控制方法的分叉推论。这样既保留 general 形式，也避免把 expected Fisher 当成固定样本动力学。

## 2.2 定理 1：单个负优势更新必然提高该样本 surprisal

令 z=(s,a)，Sθ(z)=−logπθ(a\|s)，gθ(z)=∇θlogπθ(a\|s)。对固定负优势 A(z)=−c\<0，单样本梯度上升为：

$$
\theta^+=\theta-hc\,g_\theta(z),\qquad h>0
$$

对 Sθ 做二阶 Taylor 展开，存在位于 θ 与 θ⁺ 之间的 θ̃，使：


$$
S_{\theta^{+}}(z)-S_\theta(z)=hc\lVert g_\theta(z)\rVert^2+\frac12 h^2c^2 g_\theta(z)^\top\!\left[\nabla^2 S_{\tilde\theta}(z)\right]g_\theta(z)
$$


若该线段上 ‖∇²S‖op≤L，则：


$$
S_{\theta^{+}}-S_\theta\ge hc\lVert g_\theta\rVert^2\left(1-\frac12hcL\right)
$$


因此当 hcL\<2 时，surprisal 严格增加。连续时间梯度流 θ̇=−c gθ 下更有精确恒等式：


$$
\frac{dS_\theta(z)}{dt}=c\lVert g_\theta(z)\rVert^2\ge 0
$$


这一定理是连续与离散的共同主干：负更新不是“静态降低概率”，而是把同一样本沿当前策略的 score geometry 持续推向更低支持。

## 2.3 批量更新：自项、跨样本干涉与方向一致性

单样本单调性不能无条件提升为“batch 中每个负样本 surprisal 都单调增加”。令 batch field F=ΣⱼAⱼgⱼ，则样本 i 的首阶变化为：


$$
\Delta S_i=-h g_i^\top F+O(h^2)=h|A_i|\lVert g_i\rVert^2-h\sum_{j\ne i}A_j\langle g_i,g_j\rangle+O(h^2)
$$


第一项是负样本自身的确定性排斥；第二项是正负样本共享参数带来的 interference。远场风险因此不仅取决于单样本 scale，还取决于梯度方向是否相干。本文实验中的 aggregate amplification 正是在单样本 score 放大之外叠加了 coherence。


$$
\text{Repulsive influence}\approx\text{negative mass}\times\text{score scale}\times\text{directional coherence}\times\text{repeated reuse}
$$


## 2.4 定理 2：正则最小指数族中的 signed-moment 平衡

考虑固定状态下的正则最小指数族：


$$
\pi_\eta(a)=h(a)\exp\!\left\{\eta^\top T(a)-\psi(\eta)\right\}
$$


令 t₊=E\_{P₊}\[T(a)\]、t₋=E\_{P₋}\[T(a)\]，w=p−q。则 signed policy objective 可精确写为：


$$
J(\eta)=(p t_+-q t_-)^\top\eta-(p-q)\psi(\eta)+C=w\left[\tau^\top\eta-\psi(\eta)\right]+C
$$



$$
\tau=\frac{p t_+-q t_-}{p-q}
$$


其梯度和 Hessian 为：

$$
\nabla_\eta J=w[\tau-m(\eta)],\qquad m(\eta)=\mathbb E_{\pi_\eta}[T(a)]
$$


$$
\nabla_\eta^2J=-w\,\operatorname{Cov}_{\pi_\eta}[T(a)]
$$


由此得到统一结论：

- 若 w\>0 且 signed target τ 位于指数族 mean-parameter domain 的内部，则存在唯一有限平衡 η\*，满足 m(η\*)=τ；在可识别子空间上 Hessian 负定，平衡局部渐近稳定。

- 若 τ 位于 mean-domain 边界，最优分布只能在边界上实现，通常需要自然参数趋于无穷；这对应 Gaussian 的零方差边界或 categorical 的零概率支持。

- 若 τ 落在可行域之外，或 w≤0，则不存在有限内部平衡；目标可能无界，或动力学向参数/分布边界逃逸，具体表型由策略族决定。

- 离散 Euler 更新在平衡附近的充分步长条件是 ρ(I+hJ)\<1；指数族自然参数下可写为 h \< 2/\[w λmax(Covπ\*\[T\])\]。

这个定理把“稳定外推”和“崩溃”统一成一个几何问题：负优势把正样本的 moment target 沿远离负样本的方向外推；只要外推后的 signed target 仍位于可行 moment 域内，就存在稳定解；一旦越界，内部固定点消失。

| **策略族**                   | **充分统计 T(a)** | **mean-domain** | **越界表型**                                                   |
|------------------------------|-------------------|-----------------|----------------------------------------------------------------|
| 固定方差 Gaussian            | a                 | 整个实数空间    | p≤q 时均值漂移或 runaway                                       |
| 可学习方差 Gaussian          | (a, a²)           | m₂\>m₁²         | signed variance≤0，σ→0 或联合失稳                              |
| full softmax categorical     | one-hot eₐ        | 概率单纯形      | 某些 signed probability≤0，logit gap→∞                         |
| feature / energy categorical | 动作特征 φ(a)     | 特征凸包内部    | 目标 feature moment 越界或贴边，support / temperature collapse |

## 2.5 Gaussian 推论 A：固定方差下的稳定外推与均值相变

对 π=N(μ,σ²)，固定 σ。设正负动作均值为 m₊、m₋，有效质量为 p、q。均值动力学为：


$$
\dot\mu=\frac{p(m_+-\mu)-q(m_--\mu)}{\sigma^2}
$$


当 p\>q 时存在稳定点：

$$
\mu^*=\frac{pm_+-qm_-}{p-q},\qquad \mu^*-m_+=\frac{q(m_+-m_-)}{p-q}
$$

若 m₋\<m₊，负样本位于正样本另一侧，则 μ\*\>m₊：负梯度把策略稳定推到最佳正样本支持之外。若真实最优为 a\*\>m₊，使 μ\*=a\* 的最优负质量为：


$$
q_{\mathrm{opt}}=p\frac{a^*-m_+}{a^*-m_-}<p
$$


因此任务最优点严格位于动力学临界点 qcrit=p 之前。离散更新的误差满足：


$$
\mu_{t+1}-\mu^*=\left[1-\frac{h(p-q)}{\sigma^2}\right](\mu_t-\mu^*)
$$


稳定步长要求 0\<h(p−q)/σ²\<2。q=p 时若 m₊≠m₋，吸引与排斥曲率抵消，出现持续漂移；q\>p 时均值固定点失去稳定性并产生 runaway。

## 2.6 Gaussian 推论 B：可学习方差的联合稳态与提前失稳

令 ξ=logσ，正负条件方差分别为 v₊、v₋，并定义 M±(μ)=v±+(μ−m±)²。精确动力学为：


$$
\dot\mu=\frac{p(m_+-\mu)-q(m_--\mu)}{\sigma^2}
$$



$$
\dot\xi=\frac{pM_+(\mu)-qM_-(\mu)}{\sigma^2}-(p-q)
$$


联合内部固定点为：


$$
\mu^*=\frac{p m_+-q m_-}{p-q}
$$



$$
\sigma^{2*}=\frac{pM_+(\mu^*)-qM_-(\mu^*)}{p-q}
$$


将其化成 signed variance 可得到更清晰的可行性条件。令 Δ=m₊−m₋：


$$
\sigma^{2*}=\frac{p v_+-q v_-}{p-q}-\frac{pq\Delta^2}{(p-q)^2}
$$


因此联合稳态需要 p\>q 且 σ²\*\>0。第二个条件通常更严格，使方差边界早于均值边界。令 C=v₊+v₋+Δ²，v₋\>0 时较小正根为：


$$
q_{\mathrm{var}}=p\frac{C-\sqrt{C^2-4v_+v_-}}{2v_-}
$$


若 v₋=0，则极限为：


$$
q_{\mathrm{var}}=p\frac{v_+}{v_++\Delta^2}
$$


在联合固定点处，(μ,ξ) 动力学 Jacobian 恰好对角化：


$$
J_F(\mu^*,\xi^*)=\operatorname{diag}\!\left(-\frac{p-q}{\sigma^{2*}},-2(p-q)\right)
$$


所以只要内部解存在且 p\>q，均值和 log-std 都局部稳定；实验中观察到的“方差先坍缩”不是固定点不稳定，而是 signed target 先离开 Gaussian 可行 moment 域，使有限固定点直接消失。

## 2.7 Gaussian 推论 C：方差四象限、单样本 MLE 与远场幅度放大

$$
\frac{\partial\log\pi}{\partial\xi}=z^2-1,\qquad z=\frac{a-\mu}{\sigma}
$$

| **advantage** | **\|z\|\<1**          | **\|z\|\>1**            |
|---------------|-----------------------|-------------------------|
| A\>0          | σ下降：集中到近正样本 | σ上升：覆盖远正样本     |
| A\<0          | σ上升：摊薄近负样本   | σ下降：压缩远负样本支持 |

单个确定性正样本的 Gaussian log-likelihood 没有有限最大值：μ→a 后仍有 logπ(a)=−logσ+C→+∞，故 σ→0。只有拟合均值后仍存在非零条件残差，或加入 entropy/KL/σ-min，positive-only 才有有限方差稳态。

原 sign-only Hessian 论证的问题在此处最清楚。固定样本的 negative-log-likelihood Hessian 为：


$$
H_{\mathrm{sample}}=\begin{bmatrix}\sigma^{-2}&2(a-\mu)\sigma^{-2}\\2(a-\mu)\sigma^{-2}&2(a-\mu)^2\sigma^{-2}\end{bmatrix}
$$


$$
\det(H_{\mathrm{sample}})=-\frac{2(a-\mu)^2}{\sigma^4}<0\qquad(a\ne\mu)
$$

它是不定矩阵；只有对 a~π 取期望后才得到 Fisher / expected Hessian diag(σ⁻²,2)≻0。因此不能由 expected SPD 推出固定 off-policy 样本在 (μ,ξ) 每个方向都统一扩张。正确结论是：负样本始终排斥均值，但方差方向由 z²−1 决定。

远场幅度分叉仍然成立。Gaussian score 为：

$$
g_\mu=\frac{a-\mu}{\sigma^2}=\frac{z}{\sigma},\qquad g_\xi=z^2-1
$$


$$
\lVert g\rVert^2=\frac{z^2}{\sigma^2}+(z^2-1)^2
$$


固定 σ 且只重复一个负样本时，δₜ=μₜ−a 满足精确递推 δₜ₊₁=(1+hc/σ²)δₜ，故均值距离和 mean-score 关于训练步数几何增长。可学习方差时，远场负样本同时使 μ 远离、σ 收缩，通常进一步放大标准化距离；但不应再无条件声称 μ 与 σ 都“expand”。

## 2.8 Categorical 推论 A：有界单步 score 仍可把策略推到 simplex 边界

对 K 类 full-softmax，logits 为 z，π=softmax(z)。单独重复负更新动作 j，A=−c：


$$
\dot z=c(\pi-e_j)
$$



$$
\frac{d[-\log\pi_j]}{dt}=c\lVert e_j-\pi\rVert^2
$$


direct-logit score 有界：‖eⱼ−π‖≤√2。因此 categorical 不具备 Gaussian 式的单 token 欧氏梯度无界爆炸。但一旦 πⱼ≤ε，Cauchy 不等式给出：


$$
\lVert e_j-\pi\rVert^2\ge\frac{K}{K-1}(1-\varepsilon)^2>0
$$


所以该 token 的 surprisal 至少线性增长，概率至多指数衰减；logit gap 可以趋于无穷，分布被推到概率单纯形边界。动作集合有限并不能阻止 support collapse。

full-softmax 也是指数族，T(a)=eₐ。signed target 为：


$$
\pi^*=\frac{p r_+-q r_-}{p-q}
$$


若某个分量为 0，有限 logits 无法达到，只能令对应 logit→−∞；若某个分量为负，则 target 已离开 simplex，不存在内部解。由此得到离散版的精确 support-feasibility 边界。

Entropy 不是这一动力学的充分统计量：抑制高概率负动作时 entropy 可以先升高，抑制低概率负动作时 entropy 可直接下降；两种路径都可能最终损伤任务支持。因此 entropy control 是必要 baseline，但不能替代对具体危险负更新的选择性诊断。

## 2.9 Categorical 推论 B：未见动作外推为何需要语义结构，而不需要“动作有序”

对完全饱和的独立 logits，训练中从未出现的动作在经验 signed target 中通常为 0；纯最大似然/负强化不会凭空知道应把概率放到哪个未见动作。方向性外推必须来自共享参数、预训练先验或动作特征，而不是 token ID 顺序。

更一般地，令动作拥有任意编号和语义特征 φ(a)，使用 energy policy：


$$
\pi_\eta(a\mid s)\propto\exp\!\left\{\eta(s)^\top\phi(a)\right\}
$$


它仍是指数族，稳定点满足：


$$
\mathbb E_{\pi^*}[\phi(a)]=\frac{p\mathbb E_+[\phi(a)]-q\mathbb E_-[\phi(a)]}{p-q}
$$


负样本把目标 feature moment 推离坏动作特征；指数族的最大熵投影会把概率重新分配给具有相似语义、但可能未在正样本中出现的动作。若随机打乱 feature 与 reward 的对应关系，这种 task gain 应消失，而 support suppression 仍然存在。于是“结构破坏”对照不是为有序动作辩护，而是区分两个命题：通用的支持压制不需要结构；有益的未见动作外推需要可泛化结构。

一维 ordinal catalogue 仅保留为可解析的 T=(x,x²) 桥梁；generic categorical 的主要证据应使用随机动作 ID + semantic embedding，而不是人为数轴。

## 2.10 神经网络共享参数：指数族输出场的 pullback

令网络输出自然参数 ηθ(s)，Jacobian 为 Jθ(s)=∂ηθ(s)/∂θ。输出空间残差为 r_s(η)=p_s t₊(s)−q_s t₋(s)−(p_s−q_s)m(η)。参数场为：


$$
F_\theta=\mathbb E_s\!\left[J_\theta(s)^\top r_s(\eta_\theta(s))\right]
$$


若存在可实现的 moment-matching 解，使每个相关状态 r_s=0，则网络二阶项在固定点消失，局部 Jacobian 为：


$$
J_F(\theta^*)=-\mathbb E_s\!\left[(p_s-q_s)J_\theta(s)^\top\operatorname{Cov}_{\pi^*}[T]J_\theta(s)\right]
$$


在 p_s\>q_s 且聚合 feature-Fisher 对可训练参数子空间满秩时，该矩阵负定，得到局部稳定性。若多个状态的 signed targets 不能被同一网络同时实现，或固定点残差不为零，网络二阶项重新出现；此时只能使用一般 signed-field Jacobian，而不能声称全局凸性或唯一解。

这一推导说明矩阵形式完全可以保留：真正 general 的对象是 signed off-policy field Jacobian，而不是把 on-policy expected Fisher 直接当作固定样本转移矩阵。

## 2.11 方法推论：Global α、Exp-remoteness 与 stability budget

Global α 只改变总负质量 q，简单、稳定，但会无差别削弱近场有用信息。选择性方法令负样本权重依赖当前 policy-relative remoteness。定义连续/离散统一的 remoteness：

$$
S_i=-\log\pi_\theta(a_i\mid s_i),\qquad c_\lambda(S_i)=\exp\{-\lambda(S_i-S_0)_+\}
$$

实现时对 cλ stop-gradient，保证它是纯重权而不是额外可微正则。单负样本的首阶 surprisal 速度变为：

$$
\frac{dS}{dt}=|A|c_\lambda(S)\kappa(S),\qquad \kappa(S)=\lVert\nabla\log\pi\rVert^2
$$

若远场 κ(S) 至多多项式增长，或更一般满足 κ(S)≤Cexp(βS)，则 λ\>β 时加权 influence 有界并在远场衰减。固定方差 Gaussian 的 κ 为 O(S)，含 log-variance 的标准化远场为 O(S²)；direct-logit categorical 的 κ≤2。因此 Exp-remoteness 有一个比“梯度关于距离指数增长”更准确的故事：指数 taper 支配有限阶 score growth，并统一为 categorical 中的 π(a)^λ。

更强的 stability-budget 方法直接使用定理 2 的可行性：先经 cλ 重权得到有效 q_c 与 t₋,c，再选择最大的 batch 系数 γ∈\[0,1\]，使 signed target 保持在 mean-domain 的安全内点。

$$
\gamma^*=\max\{\gamma\in[0,1]:p-\gamma q_c\ge\varepsilon_{\mathrm{mass}},\ \operatorname{dist}(\tau(\gamma),\partial\mathcal M)\ge\varepsilon_{\mathrm{geom}}\}
$$

Gaussian 中可用 p−γq_c\>0 与 σ²\*(γ)≥σ²min 两个闭式条件，计算只需 batch reductions；full-softmax 可约束所有 signed probabilities≥ε。一般 feature policy 的凸包距离较难精确计算，因此 SBRC-Lite 只能使用 score/moment proxy，理论保证相应减弱。

## 2.12 Learned critic / value network：瞬时适用与移动目标

在 DRPO-Q、IQL 或一般 actor-critic 中，A_t=Qφ(s,a)−Vψ(s) 会随 critic 更新。只要 actor step 使用 A_t.detach()，上述理论对每一步的瞬时 signed field 仍成立；但整个系统变成非自治动力学，不能把固定 advantage 的全局固定点直接照搬。

若每一时刻都存在内部目标 η\*(t)，局部收缩率下界为 m\>0，且目标漂移速度 ‖η̇\*(t)‖≤v，则标准移动平衡分析给出 tracking error 的量级：


$$
\limsup_t\lVert\eta(t)-\eta^*(t)\rVert\le\frac{v}{m}
$$


因此 critic 越慢、稳定裕度越大，actor 越能跟踪；但任何梯度控制都不能修复 critic 给错 advantage 符号的问题，只能限制错误信号被 score geometry 放大的破坏。

## 2.13 自我审查：反例挑战、修正与最终可声明边界

| **挑战**                                       | **审查结果**                                                     | **最终处理**                                                            |
|------------------------------------------------|------------------------------------------------------------------|-------------------------------------------------------------------------|
| 单负样本 surprisal 是否在 batch 中仍必增？     | 否；跨样本 Gram 项可反转。                                       | 定理限定为单样本/隔离更新；batch 使用 interference 分解。               |
| expected Fisher SPD 能否证明固定样本联合扩张？ | 不能；pointwise Hessian 一般不定。                               | 以 signed field Jacobian 和指数族 Hessian 取代。                        |
| Gaussian 负样本是否总使 σ 增大？               | 否；far negative 使 σ下降，near negative 使 σ上升。              | 保留 z²−1 四象限，删除 both μ and σ expand。                            |
| 正样本非确定是否自动保证有限 σ？               | 仅当拟合状态后仍有非零条件残差。                                 | 把条件残差或 entropy/KL/σ-min 写成必要来源。                            |
| 有限 categorical 是否不会发散？                | 动作有限，但 logit gap 无界，概率可到 simplex 边界。             | 区分 amplitude runaway 与 support runaway。                             |
| rare token 的 direct-logit score 是否无界？    | 否，范数≤√2。                                                    | 只声称持续 suppression；Fisher 内禀范数与 SGD 梯度分开。                |
| 负优势是否必然带来未见动作泛化？               | 否；无结构 independent logits 不知道往哪里分配。                 | 外推需共享表示/动作特征；加入结构破坏对照。                             |
| entropy 是否等价于 support quality？           | 否；同一 entropy 可对应不同任务支持。                            | entropy control 仅作为 baseline，不作为机制替代。                       |
| Exp 是否由“距离指数增长”直接推出？             | 不完全；score 对距离多为线性/二次。                              | 改为指数 taper 支配多项式 score growth 的有界性论证。                   |
| 指数族全局结论能否直接套神经网络？             | 不能；共享网络可能不可实现，且非凸。                             | 只在 realizable fixed point 给 pullback 局部稳定；其余用一般 Jacobian。 |
| Adam / PPO / importance ratio 是否被定理覆盖？ | 当前定理直接覆盖 gradient flow / Euler 和 detached reweighting。 | 其他优化器、ratio clipping 作为经验扩展，不写成严格推论。               |
| information 随距离下降是否已证明？             | 尚未；需要任务结构和方向可靠性假设。                             | 保留为可检验 hypothesis，不列为已证定理。                               |
| 边界/低熵是否必然导致任务 reward collapse？    | 不必；若边界动作恰为最优可提升。                                 | 区分 support collapse 与 task collapse，后者需环境因果实验。            |

自审结论：目前没有发现会推翻主框架的逻辑缺口。可以严格成立的是“单样本排斥恒等式 + 指数族 signed-moment 可行性 + Gaussian/categorical 分叉 + 局部神经网络 pullback”。仍不能升级为定理的是“方向信息必随距离单调下降”“任意真实任务都由该机制唯一导致 collapse”以及“某一种控制在所有任务上必胜”。

# 3. 连续统一 benchmark：正式论文级证据

## 3.1 Protocol A：来源隔离

| **阶段**            | **\|A\| far/near** | **score** | **单样本梯度** | **聚合梯度** |
|---------------------|--------------------|-----------|----------------|--------------|
| initialization      | 1.000              | 45.13×    | 47.78×         | 61.56×       |
| positive_pretrained | 1.000              | 38.02×    | 38.64×         | 82.08×       |

advantage 与 quality coordinate 沿半径严格复制；独立性检查在 20/20 seeds 中误差为 0。远场放大来自 score geometry，方向一致性进一步放大聚合梯度。

<img src="media/image1.png" style="width:5.9in;height:3.79382in" />

*图 1　统一连续 benchmark 中正样本预训练后的远场/近场梯度分解。*

## 3.2 Protocol B：因果 collapse 干预

| **方法**      | **最终 reward** | **95% CI**       | **collapse** |
|---------------|-----------------|------------------|--------------|
| baseline      | 0.201           | \[0.165, 0.239\] | 19/20        |
| near_zero     | 0.195           | \[0.162, 0.232\] | 18/20        |
| far_zero      | 0.618           | \[0.596, 0.639\] | 0/20         |
| far_cap       | 0.666           | \[0.653, 0.680\] | 0/20         |
| global_scale  | 0.763           | \[0.753, 0.773\] | 0/20         |
| positive_only | 0.782           | \[0.771, 0.793\] | 0/20         |

Far-zero 与 Far-cap 对 baseline 的 paired improvement 均为 20/20，Wilcoxon p=1.91×10⁻⁶；Near-zero 与 baseline 无显著差异（p=0.62）。

<img src="media/image2.png" style="width:5.9in;height:3.50264in" />

*图 2　连续因果干预的 20-seed 最终性能与置信区间。*

| **【边界】这证明受控环境中的主导传导路径，不证明所有真实任务中只有远场负梯度这一种原因。** |
|--------------------------------------------------------------------------------------------|

## 3.3 Protocol C：稳定外推与联合相变

| **方差设置**       | **α** | **β**     | **held-out reward** | **σ** | **entropy** |
|--------------------|-------|-----------|---------------------|-------|-------------|
| fixed_variance     | 0.0   | -0.000    | 0.085               | 1.200 | 3.203       |
| fixed_variance     | 0.5   | 0.897     | 0.837               | 1.200 | 3.203       |
| fixed_variance     | 0.75  | 2.753     | 0.002               | 1.200 | 3.203       |
| learnable_variance | 0.0   | -0.000    | 0.085               | 1.200 | 3.203       |
| learnable_variance | 0.5   | 0.782     | 0.709               | 1.376 | 3.460       |
| learnable_variance | 0.68  | -3433.588 | 0.080               | 1.267 | -1.998      |
| learnable_variance | 0.7   | 1.080     | 0.099               | 0.942 | -3.529      |

Positive-only 的 β≈0；固定方差 α=0.5 达到 β≈0.897、reward≈0.837；可学习方差 α=0.5 达到 β≈0.782、reward≈0.709。可学习方差在 α=0.65–0.68 附近进入过渡，并从 α=0.70 起 20/20 方差坍缩，早于固定方差约 α=1 的均值边界。

<img src="media/image3.png" style="width:5.9in;height:3.59047in" />

*图 3　连续环境中的 imitation ceiling、有益外推、过度外推和性能反转。*

<img src="media/image4.png" style="width:5.9in;height:3.48148in" />

*图 4　可学习方差使联合稳定边界提前。*

## 3.4 控制方法与等预算识别

| **方法**              | **reward** | **β**  | **σ** | **平均负权重** |
|-----------------------|------------|--------|-------|----------------|
| distance_cap          | 0.747      | 0.827  | 1.376 | 0.563          |
| budget_matched_global | 0.725      | 0.798  | 1.374 | 0.554          |
| global_scale          | 0.719      | 0.791  | 1.374 | 0.556          |
| positive_only         | 0.085      | -0.000 | 1.200 | 0.000          |
| uncontrolled          | 0.000      | 29.466 | 1.006 | 1.000          |

Distance cap 相对等预算 global control 提升 0.021，95% CI \[0.019, 0.023\]，20/20 paired seeds 胜出。该差异统计上稳定但数值较小，只支持“该 benchmark 中选择性控制更优”。

<img src="media/image5.png" style="width:5.9in;height:2.31346in" />

*图 5　连续不稳定设置中的 global α、等预算 global 与 distance cap。*

# 4. 离散 categorical benchmark：理论与正式结果

## 4.1 小环境：direct softmax 的精确结论

| **负动作初态**            | **p₀** | **p_T**  | **H₀** | **H峰值** | **H_T**  | **max score** |
|---------------------------|--------|----------|--------|-----------|----------|---------------|
| high_probability_negative | 0.8991 | 4.06e-12 | 0.386  | 0.906     | 6.72e-06 | 1.414213      |
| low_probability_negative  | 0.0038 | 1.90e-20 | 0.292  | 0.292     | 4.51e-09 | 1.414214      |

高概率负动作被抑制时，entropy 先上升后下降；低概率负动作被抑制时，entropy 从一开始就下降。两者 surprisal 都持续增加，score norm 均不超过 √2。

<img src="media/image6.png" style="width:5.8in;height:3.92903in" />

*图 6　离散负更新的 entropy 方向取决于当前动作概率。*

## 4.2 来源隔离：rarity 使梯度更大但幅度有界

| **阶段**            | **advantage** | **分布参数 score** | **全参数梯度** | **surprisal** |
|---------------------|---------------|--------------------|----------------|---------------|
| initialization      | 1.000         | 2.61×              | 2.79×          | 1.55×         |
| positive_pretrained | 1.000         | 3.30×              | 3.65×          | 1.66×         |

离散 far/near 放大约 3–4×，明显小于连续 Gaussian 的 38–82×，与有限 catalogue 和有界 direct-logit score 一致。

<img src="media/image7.png" style="width:5.8in;height:3.80625in" />

*图 7　categorical rarity-source isolation。*

## 4.3 结构化 categorical 外推：解析桥梁与无序语义动作

| **温度设置**  | **α** | **β**  | **reward** | **τ** | **entropy** |
|---------------|-------|--------|------------|-------|-------------|
| fixed_tau     | 0.0   | -0.002 | 0.246      | 1.200 | 2.981       |
| fixed_tau     | 0.5   | 0.894  | 0.327      | 1.200 | 2.949       |
| fixed_tau     | 0.9   | 3.487  | 0.000      | 1.200 | 0.063       |
| learnable_tau | 0.0   | -0.002 | 0.246      | 1.200 | 2.981       |
| learnable_tau | 0.5   | 0.894  | 0.408      | 0.909 | 2.704       |
| learnable_tau | 0.58  | 1.242  | 0.619      | 0.251 | 1.423       |
| learnable_tau | 0.62  | 1.470  | 0.339      | 0.050 | 0.206       |

一维 ordinal catalogue 只作为 T=(x,x²) 的解析桥梁：其 signed-moment 可行边界为 α≈0.585，实验中 α=0.58 稳定而 α=0.62 temperature collapse。generic categorical 的主要外推证据应来自随机动作 ID + semantic embedding；无结构 independent logits 仅用于验证 support suppression。

<img src="media/image8.png" style="width:5.8in;height:3.83692in" />

图 8　结构化 categorical 中的 support extrapolation、过度外推和性能反转（ordinal bridge）。

<img src="media/image9.png" style="width:5.8in;height:3.83692in" />

图 9　signed-moment 可行边界与 empirical entropy/temperature transition；无序语义动作实验用于排除“动作编号有序”的人为设定。

## 4.4 categorical near/far 因果干预

| **方法**      | **reward** | **entropy** | **task collapse** | **support collapse** |
|---------------|------------|-------------|-------------------|----------------------|
| baseline      | 0.000      | 0.000       | 20/20             | 20/20                |
| near_zero     | 0.001      | 0.001       | 20/20             | 20/20                |
| far_zero      | 0.250      | 3.095       | 0/20              | 0/20                 |
| far_cap       | 0.388      | 2.686       | 0/20              | 0/20                 |
| global_scale  | 0.252      | 0.142       | 0/20              | 20/20                |
| positive_only | 0.246      | 2.981       | 0/20              | 0/20                 |

Baseline 与 Near-zero（保留 far negatives）在 task 和 support 两个指标上均 20/20 collapse；Far-zero 与 Far-cap 均 0/20。Global scale 避免 task collapse，但 20/20 support collapse，说明缩小总负质量与选择性控制 far negatives 并不等价。

<img src="media/image10.png" style="width:5.8in;height:3.83692in" />

*图 10　categorical far-negative targeted interventions。*

## 4.5 控制方法：反对“distance 永远更优”的过度 claim

| **方法**              | **reward** | **β**  | **entropy** |
|-----------------------|------------|--------|-------------|
| global_scale          | 0.408      | 0.894  | 2.704       |
| budget_matched_global | 0.386      | 0.827  | 2.771       |
| distance_cap          | 0.377      | 0.852  | 2.800       |
| positive_only         | 0.246      | -0.002 | 2.981       |
| uncontrolled          | 0.000      | 3.467  | 0.000       |

在单负分布的 categorical 外推任务中，global scale 比 distance cap 高 0.031 reward；这是有价值的反证。稳健结论应是：α 控制总 repulsive mass，distance/far-selective control 在 rare negatives 主导时更能保护 support；具体 reward 排名取决于任务。

<img src="media/image11.png" style="width:5.8in;height:3.83692in" />

*图 11　categorical 单负分布下的恢复控制。*

# 5. 连续—离散统一解释

| **维度**       | **连续 Gaussian**               | **离散 categorical**                               |
|----------------|---------------------------------|----------------------------------------------------|
| 基本危险量     | 距离 d 与 precision 1/σ²        | 动作 surprisal / 低概率支持                        |
| 单样本 score   | 可随 d/σ²、d²/σ² 无界放大       | direct-logit score 有界                            |
| 负更新通用结果 | 均值远离负样本                  | 被选动作 surprisal 单调增加                        |
| 熵/尺度分支    | 远场负样本压缩 σ                | rare negative 压低概率与 entropy                   |
| 稳定外推条件   | 正负均值/二阶残差平衡           | 结构化 catalogue 的 signed moment 可行             |
| 失稳形态       | 幅度 runaway、variance collapse | temperature/support collapse、catalogue saturation |
| 控制           | global α、distance cap          | global α、surprisal/far cap                        |

统一对象不是“梯度 norm 在两种空间都必然爆炸”，而是 repulsive surprisal dynamics：重复负更新持续降低被拒样本在当前策略下的支持；连续分布可把这种 rarity 转化为无界 score amplitude，离散 direct logits 通常表现为支持集坍缩。


$$
\text{Repulsive risk}\approx\text{negative mass}\times\text{policy-relative rarity/geometry}\times\text{directional coherence}\times\text{repeated updates}
$$


# 6. 论文贡献与推荐 claim 结构

## 6.1 当前已经形成的贡献栈

1.  机制来源：严格解耦 advantage/quality 与 distance/rarity，证明异常负梯度来自 policy-relative geometry。

2.  因果传导：连续与离散均使用 near/far 定点干预闭合 far-field → drift/support collapse → performance failure。

3.  稳定—泛化理论：从 positive-only ceiling 到稳定外推、过度外推、临界失稳。

4.  联合尺度理论：连续方差与离散温度均可由 signed second moment / residual balance 推导稳定边界。

5.  统一解释：global α、positive-only、hard filtering、distance/surprisal control 都可视为 repulsive-gain control 的不同形式。

6.  方法设计依据：不再是先提出 heuristic 再寻找解释，而是从稳定边界反推控制策略。

## 6.2 论文主文建议

| **模块**             | **主文内容**                                                                      | **附录内容**                      |
|----------------------|-----------------------------------------------------------------------------------|-----------------------------------|
| Theory               | signed field Jacobian；Gaussian 联合稳态；categorical surprisal 与 signed moments | 完整推导、离散时间步长条件        |
| Controlled benchmark | 连续 A/B/C + categorical direct/phase/causal                                      | 架构稳健性、旧小环境 sanity check |
| Method               | 统一 repulsive-gain control；连续 distance 与离散 surprisal 版本                  | 更多权重函数和超参                |
| External validation  | 至少一个连续真实/标准任务 + 一个 token/序列任务                                   | 额外数据集与消融                  |

## 6.3 推荐用语

- “We identify a dominant far-field pathway in controlled off-policy policy-gradient dynamics.”

- “Negative feedback is locally informative but becomes destabilizing as policy-relative rarity grows.”

- “Continuous policies exhibit amplitude/precision amplification; categorical policies exhibit persistent surprisal growth and support collapse.”

- “Distance-aware control outperforms a budget-matched global control in the continuous benchmark, while categorical results show that no universal ranking should be claimed.”

# 7. 接下来唯一保留的任务清单

## P0：统一理论已完成；转入论文 LaTeX 化与定理精简

- 重写原 DRPO Section 3：删除 sign-only SPD theorem 与 “μ、σ 同时扩张”；改为 signed field Jacobian + 精确 Gaussian/categorical corollaries。

- 把 fixed-advantage assumption、时间重参数化和离散时间步长条件写入 theorem assumptions。

- 将统一 benchmark 的正式表格与图直接迁入论文草稿；旧简单环境只留 appendix provenance。

## P1：方法创新与外部有效性（下一实验主线）

**Countdown 模型梯度：**0.5B 仅用于快速调通和超参粗筛；3B 作为主 arena（效率与基础能力平衡）；7B 只对冻结后的前两名方法做最终确认。任务本身不要求 7B，但方法比较要求基线具有足够非零成功率和改进空间。

- 设计统一的 repulsive-gain controller：连续使用 standardized distance / surprisal，离散使用 token/action surprisal；允许全局 α 作为基线与退化特例。

- 在一个连续标准任务（优先 D4RL/推荐环境）验证 stability–generalization trade-off。

- 在一个离散序列任务或小型 Transformer 上验证 rare-negative suppression、entropy collapse 与 selective control。

## P2：当前不做或降级为 future work

- 组合泛化。

- 大规模架构笛卡尔积。

- 动态 critic/value feedback。

- 所有相关方法的完整复现。

| **【下一步】连续与 categorical 受控机制实验已完成。除非论文审稿风险明确指出缺口，不再继续堆叠同类 toy robustness。** |
|----------------------------------------------------------------------------------------------------------------------|

# 8. 复现入口与文件索引

| **内容**                   | **路径/命令**                                             |
|----------------------------|-----------------------------------------------------------|
| 连续 formal results        | unified_repulsive_dynamics/results/paper_run/             |
| categorical formal results | unified_repulsive_dynamics/results/categorical_paper_run/ |
| 连续一键复现               | python run_paper.py --mode paper --rerun-collapse         |
| categorical 一键复现       | python run_categorical.py --mode paper                    |
| 两类统一复现               | python run_all.py --mode paper --rerun-collapse           |
| 连续结果包                 | Unified_Repulsive_Dynamics_Paper_Results.zip              |
| categorical 结果包         | Categorical_Repulsive_Dynamics_Paper_Results.zip          |

正式随机种子：10–29。连续架构稳健性：30–34。所有 formal result 包含 raw curves、per-seed finals、bootstrap 95% CI、paired Wilcoxon、PNG/PDF figures、配置与 invariant tests。

# 附录 A. 旧实验的保留规则

下列实验不再占用正文篇幅，但不得删除源文件：

- C1 标量固定方差：验证闭式均值相变。

- C2 单状态/多状态 MLP：排除直接参数化偶然性。

- V0/V1 可学习方差：发现并验证早期方差边界。

- 原 gradient-explode：证明 phantom gradient growth，但同时暴露“梯度 norm ≠ 参数扩张”的记录歧义。

- 旧 product-manifold 与 causal_farfield：作为统一 benchmark 的独立历史复现与机制 provenance。

论文主文所有数值优先引用统一 formal benchmark；旧结果只用于 appendix、代码审计或 rebuttal。

# 附录 B. 最终研究判断

| **【锁定】当前工作已经从“一个负优势加权方法”升级为机制论文：来源隔离、因果传导、稳定—泛化相变、连续—离散统一和方法设计原则均有理论与 formal controlled evidence。方法创新仍然重要，但不再需要单独承担整篇论文的贡献。** |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

最合理的论文结构是：修正后的动力学理论 → 统一连续/离散受控 benchmark → 由理论导出的 repulsive-gain controller → 外部任务验证。

# 附录 C. 公式自检与机器验证

v9 理论完成后执行独立数值/自动微分自检：Gaussian 联合固定点残差、解析 Jacobian、方差临界根、固定样本 Hessian 行列式、categorical surprisal 速率、softmax score 上界与 Taylor 首阶误差均通过。可复现脚本：drpo_theory_v9/theory_self_check.py。

---

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

# Part V. Bandit 稳定外推子实验的收敛审计（完整保留）

> 本审计只覆盖有解析参照的稳定外推子实验。它修正了短训练终值，但没有完成 E2、E3、E6、E7 的完整长期审计。

# DRPO Bandit Saturation Re-audit (v1)

## 结论摘要

这次审计先计算可解析的 ground truth，再把原来的短训练延长到 **5,000–20,000 steps**，关键临界点补到 10,000 steps，并检查末段斜率、梯度范数与 moment error。

1. **原稳定外推实验明显低估了有限稳态的位置与性能。** 主要原因是没有训练到饱和。
2. **相变结构没有消失，反而与解析 ground truth 更一致。** 有限固定点、近临界慢收敛和无固定点 runaway 被清楚分开。
3. 连续可学习方差的 20-seed 解析临界点为 **0.6645 ± 0.0063**，范围 **[0.6475, 0.6751]**。
4. Categorical learnable-temperature 的平均 moment 临界点为 **0.5846 ± 0.0001**；第一个 state 失去正方差的边界约为 **0.5801**。
5. **无序 semantic categorical 的 120-step 方法排名未饱和，正式降级为 pilot。** 不能进入论文最终方法表。

## 1. Ground truth

### Continuous, fixed variance

对 `p=n=1, a_+=0, a_-=-1`：

```math
\beta^*(\alpha)=\frac{\alpha}{1-\alpha},\qquad \alpha<1.
```

`alpha >= 1` 时没有有限均值固定点。固定方差分支不应使用 signed variance 是否为正来判断均值固定点；旧审计脚本在这一点上有分类 bug，本报告已修正。

### Continuous, learnable variance

均值固定点仍为上式；对 diagonal variance，第 `j` 维的 signed residual variance 为

```math
\sigma_j^{2*}=\frac{\tau_+^2-\alpha\tau_-^2}{1-\alpha}-\frac{\alpha}{(1-\alpha)^2}\mathbb E[d_j^2].
```

只有所有维度均为正时，联合均值–方差内部固定点才存在。

### Categorical energy policy

固定 temperature 时，有限内部解需要匹配 signed first moment；learnable temperature 时还要匹配 signed second moment。signed variance 失去正性后，有限 temperature 解消失，策略走向 catalogue/simplex 边界。

## 2. 重新训练后的关键结果（seeds 10–12）

### Continuous fixed variance

| alpha | steps | theory beta* | train beta | test beta | tail slope / step | 判定 |
|---:|---:|---:|---:|---:|---:|---|
| 0.50 | 20,000 | 1.000 | 0.9838 | 0.9776 | 1.07e-06 | 有限稳态，近饱和 |
| 0.75 | 10,000 | 3.000 | 2.9794 | 2.9781 | 3.37e-06 | 有限但严重过外推，近饱和 |
| 0.90 | 10,000 | 9.000 | 8.9476 | 8.9450 | 1.06e-05 | 远场有限稳态，近饱和 |
| 1.00 | 3,000 | none | 98.5 | 97.8 | 3.48e-02 | 持续漂移，无有限固定点 |

原 2,200-step 的 `alpha=0.5` test beta/reward 约为 `0.897/0.837`；重跑后为 **0.978/0.954**。旧绝对数值必须替换。

### Continuous learnable variance

| alpha | steps | theory beta* | train beta | sigma_min | theory sigma_min | 判定 |
|---:|---:|---:|---:|---:|---:|---|
| 0.50 | 10,000 | 1.000 | 0.9626 | 1.1874 | 1.1796 | 联合固定点，近饱和 |
| 0.60 | 10,000 | 1.500 | 1.4757 | 0.9158 | 0.9075 | 联合固定点，近饱和 |
| 0.64 | 10,000 | 1.778 | 1.7659 | 0.5957 | 0.5919 | 近临界但有限，接近解析点 |
| 0.65 | 10,000 | 1.857 | 1.8466 | 0.4426 | 0.4390 | 近临界窄方差稳态 |
| 0.68 | <=750 | no joint point | NaN/collapse | NaN | signed variance < 0 | 内部固定点不存在，collapse |

**关键修正：** `alpha=0.65` 在这些 seeds 上不是未解释的失稳，而是慢收敛到窄方差有限固定点；`alpha=0.68` 才因 signed variance 变负而失去内部解。

### Categorical

| variant | alpha | steps | train beta | target beta | tau | target tau | 判定 |
|---|---:|---:|---:|---:|---:|---:|---|
| fixed tau | 0.50 | 5,000 | 0.9969 | ~0.996 | 1.2 | fixed | first-moment equilibrium saturated |
| fixed tau | 0.75 | 5,000 | 2.9928 | ~2.993 | 1.2 | fixed | far finite mean equilibrium saturated |
| fixed tau | 0.90 | 5,000 | 3.8034 | outside effective catalogue | 1.2 | fixed | latent parameter runaway / support boundary |
| learn tau | 0.50 | 5,000 | 0.9964 | ~0.996 | 0.9082 | 0.9040 | joint moment equilibrium saturated |
| learn tau | 0.58 | 5,000 | 1.3753 | ~1.377 | 0.2476 | 0.2476 | joint moment equilibrium saturated |
| learn tau | 0.62 | 5,000 | 1.6406 | infeasible | 0.05 floor | none | support/temperature collapse |

Categorical `0.58 stable / 0.62 collapse` 的夹逼经 10 倍训练长度后仍成立，并与解析 moment boundary `~0.5846` 对齐。

## 3. 对旧结论的处理

- **替换绝对数值：** continuous fixed `alpha=0.5`、learnable `alpha=0.5/0.6/0.65` 的 500–2,200-step 终值。
- **改写状态分类：** continuous learnable `alpha=0.65` 从“慢漂移/可能失稳”改为“seed-dependent boundary 内的慢收敛窄方差稳态”。
- **保留并增强：** `alpha=0.68` 无联合固定点并 collapse；categorical `0.58/0.62` 相变；fixed variance `alpha<1` 有有限均值点、`alpha=1` 持续漂移。
- **撤回为 pilot：** unordered semantic categorical 120-step 的 Global/Exp/SBRC 排名与显著性。

## 4. 哪些实验不以训练收敛为验收条件

- 乘积流形 source-isolation 回答瞬时梯度从哪里来，不是稳态实验。
- Near/Far causal intervention 回答固定 horizon 内切断路径是否救援；可以作为有限时域因果结果，但不能写成无限时域稳定定理。
- 若要回应长期动力学批评，causal protocol 仍需单独做 horizon extension；本次审计主要修复稳定外推/相变实验。

## 5. 现在可防守的 solid 结论

1. Positive-only ceiling 与适度负梯度外推的固定点可以由 ground truth 直接计算，长训练与其一致。
2. 增大负梯度强度产生的是 **有限稳态远移 → 近临界慢收敛 → 内部固定点消失/边界 runaway**，不是负优势一出现就必然发散。
3. 可学习方差/temperature 的可行域边界早于均值边界，并由 signed second moment 的正性决定。
4. 旧短训练低估了稳定外推，没有推翻相变；修正后理论–实验对齐更强。
5. 任何方法排名必须在目标指标末段斜率接近零，或明确到达无固定点边界后再报告。


---

# Part VI. 论文重写大纲（当前草案，后续须与实验状态同步）

# 论文重写大纲 v1：从推荐专属 DRPO 到通用 Repulsive Policy Dynamics

> 目标：把原稿从“生成式推荐 + Optimistic DRO + hard filtering”重构为一篇面向通用 off-policy policy optimization 的机制—理论—方法论文。本文档细化到段落，并在每一节注明“改什么、为什么改、对应哪类审稿意见”。

## 0. 一句话定位与非谈不可的重写原则

### 建议主标题

**Repulsive Policy Dynamics: Stable Extrapolation and Far-Field Collapse in Off-Policy Policy Optimization**

备选标题：

1. **When Negative Advantages Generalize—and When They Collapse Off-Policy Learning**
2. **Breaking the Curse of Repulsion: Signed-Moment Stability in Off-Policy Policy Optimization**
3. **Repulsive Surprisal Dynamics in Continuous and Discrete Policies**

### 一句话主张

负优势并非天然有害：适度、局部且方向可靠的负更新可以突破 positive-only 的模仿上限；但固定或陈旧数据被重复复用后，负更新会持续提高样本 surprisal，并在 policy-relative far field 中形成失稳。该机制在 Gaussian 策略中表现为 score 幅度与方差耦合的 runaway，在 categorical 策略中表现为 logit gap 发散和支持集坍缩。

### 必须完成的根本重写

- **从“推荐问题”改为“通用 off-policy signed policy optimization 问题”。** 推荐系统降为应用验证，不再承担理论动机的全部重量。
- **从“负优势必然爆炸”改为“共同排斥主干 + 策略族特有失稳分叉”。** 避免审稿人用 softmax score 有界或 Gaussian Hessian 反例推翻全文。
- **从“hard filtering 是必要且唯一方案”改为“负更新存在稳定—泛化折中，方法应控制 policy-relative remoteness”。** Hard filtering 仅保留为零负权重极限、旧方法或强基线。
- **理论核心从 expected-Fisher SPD 改为 surprisal increment + signed-moment feasibility + 动力场 Jacobian。** 这是技术上最重要的纠错。
- **实验核心从单一自建 RecSim 改为：受控因果环境 + D4RL 公共数据 + token-level Countdown + 可选推荐应用。**
- **彻底清理引用。** 所有参考文献必须能在 DOI、OpenReview、会议官网或 arXiv 中逐条核验；任何占位符和不确定引用不得进入主稿。

---

## 1. 旧审稿意见到新稿设计的逐项映射

| 旧审稿核心问题 | 新稿中的结构性响应 | 不允许仅靠文字解释的部分 |
|---|---|---|
| 存在 hallucinated / placeholder references | 建立引用白名单；主文只引用已核验 primary sources；附录给出 reference audit | 必须删除 `Lastname et al.`、错误 OneRec 占位符等；投稿前自动检查 BibTeX |
| 只在自建模拟器上验证 | 增加 D4RL Hopper medium / replay / expert；增加 Countdown token arena；公开统一受控环境 | 必须有公共数据、公开代码和可复现实验日志 |
| 理论过度强调 advantage sign | 将风险分解为 advantage mass × score geometry × coherence × repeated reuse；证明 sign 只决定自排斥项，不单独决定全局稳定性 | 需要 badness–distance 解耦和 near/far 因果干预 |
| Gaussian 特例被过度泛化 | 用指数族 signed-moment 定理统一；Gaussian 与 categorical 分叉；神经网络只给局部 pullback 结论 | 不能写任意深网全局收敛或任意策略都梯度无界 |
| 离线 PG 目标缺少 off-policy 条件说明 | 在 Preliminaries 明确：这是数据分布下的 signed log-likelihood actor update，不宣称无校正地等于真实 on-policy policy gradient；advantage 在 actor step 中 stop-gradient | 需要把“分析对象”与“无偏 policy gradient 定理”分开 |
| Figure 只显示梯度变大，没证明与负优势或因果相关 | Protocol A 精确解耦 advantage 与距离；Protocol B near/far 定点删除和等预算干预；报告时间顺序 | 不能只给 phantom curve；必须给 intervention |
| Hard filtering 过于简单、创新性弱 | 主方法改为 policy-relative surprisal taper，或其上加 stability budget；hard filtering 作为极端退化形式 | 必须和 global α、entropy、clipping、AWR/IQL、已有 surprisal 方法对比 |
| 缺少实验细节、指标和规模 | 独立 Reproducibility section + 主表写清数据量、seeds、CI、模型、训练步数、选择协议 | 不能将关键设置仅藏在未公开代码中 |
| 缺少推荐基线，理论又与推荐脱节 | 新稿不再以推荐为主标题；若保留推荐实验，则必须使用公共数据和现代 backbone | 不再用“生成式推荐独有问题”作为理论前提 |
| 没有 limitations | 主文单设 Limitations and Scope；列出 critic error、optimizer、non-realizable network、reward-collapse 非必然等边界 | 不得用“inevitable / universally / necessary for survival”式绝对措辞 |

---

## 2. 摘要：建议按 7 句话写成一个紧凑段落

### 句 1：通用问题

说明 off-policy actor updates 会反复使用固定或陈旧的负优势样本；这些样本既包含边界信息，也可能 destabilize training。

**修改原因：** 不再从推荐长尾数据切入，以免理论和题目脱节。

### 句 2：核心悖论

Positive-only 更新稳定但可能停在行为支持内；保留负更新可促进 mode suppression 和外推，却可能在 far field 造成崩溃。

**修改原因：** 原稿只强调“负样本有毒”，无法解释我们后续观察到的稳定泛化收益。

### 句 3：统一理论

提出 Repulsive Signed-Moment Dynamics：单负样本更新提高其 surprisal；聚合正负信号把策略推向 signed moment target；目标位于可行 moment 域内部时存在稳定外推，越界时内部稳态消失。

### 句 4：策略族分叉

Gaussian 中越界表现为均值 runaway、方差收缩与无界 score amplification；categorical 中 direct-logit score 虽有界，概率仍可指数衰减并逼近 simplex 边界。

### 句 5：因果证据

概括受控实验：advantage 与 distance 严格解耦仍出现 far-field amplification；只删除远场而非近场负更新可阻止 OOD drift 和 collapse；适度负更新出现倒 U 型泛化收益。

### 句 6：方法

提出 policy-relative remoteness control：以 surprisal 为统一变量，对负更新做指数 taper；可选增加仅在稳定裕度不足时触发的 batch safety budget。

### 句 7：外部验证

说明在 D4RL continuous-control 数据与 Countdown token policy 上验证机制和方法，并公开代码、数据处理及逐 seed 结果。

**注意：** 摘要中不再出现“hard filtering is mathematically necessary”“exactly solves all noise”或“SOTA generative recommendation”等无法由新证据直接支持的表述。

---

## 3. Introduction：建议 8 个段落

### P1：从 off-policy actor learning 的普遍结构切入

写任何使用日志、replay、stale rollouts 或固定偏好数据的 actor update，都可能出现：当前策略已经变化，但旧数据仍被重复赋予正负 advantage。列举 offline RL、off-policy generative control、RLHF/RLVR、推荐等场景，但不在此处展开相关工作。

**目的：** 建立 general paper 的对象。

### P2：解释负优势的双重价值

正优势把策略拉向已观察到的成功行为；负优势提供坏 mode 抑制和边界塑形。完全删除负优势接近 advantage-filtered imitation，因此稳定，却可能存在 support / imitation ceiling。

**目的：** 主动回应“为什么不直接过滤坏样本”；避免方法被看成简单 top-k 数据清洗。

### P3：提出真正的未解问题

同一个负更新为何会从有益的局部信号变为破坏性远场排斥？现有解释常混合三件事：样本有多差、当前策略认为它多罕见、以及负样本数量/长度多大。仅观察大梯度无法识别因果来源。

**目的：** 对应旧 reviewer 对 Figure 2 和 advantage sign 过度解释的批评。

### P4：机制概览

定义 policy-relative remoteness 为 surprisal 或相应几何距离。给出主循环：


a fixed negative sample → surprisal increases → support becomes more remote → future repulsive influence changes → drift or support collapse.

强调共同主干是 repeated repulsion，不是“所有策略的梯度范数都无界”。

### P5：理论贡献概览

先介绍单样本 surprisal increment identity，再介绍指数族 signed-moment target。用一句话解释稳定外推与崩溃是“目标在可行 moment 域内/外”的同一几何相变。

### P6：证据链概览

列出三个 protocol，但不在 Introduction 堆数值：

1. source isolation：质量与距离严格解耦；
2. causal collapse：near/far 定点干预；
3. stable extrapolation：positive-only ceiling → 最优负推力 → 过度外推 → collapse。

再说明 categorical 使用无序 action IDs + semantic embeddings，避免人为有序动作的质疑。

### P7：方法概览

说明方法不是依据 raw reward 或欧氏距离静态过滤，而是对当前策略的 surprisal 进行负更新 taper；必要时再用轻量 batch stability budget。强调只使用已有 forward-pass 量，复杂度近似线性，不计算 Hessian。

### P8：贡献列表

建议仅列四项：

1. **Unified theory：** surprisal 与 signed-moment feasibility；
2. **Causal identification：** badness–remoteness 解耦及 near/far intervention；
3. **Stability–generalization law：** 负更新的倒 U 型与联合均值—方差边界；
4. **Practical control and external validation：** remoteness taper + D4RL / token experiments。

**删除：** “首次发现负梯度有害”“hard filtering 唯一最优”“推荐 SOTA”之类贡献。

---

## 4. Related Work：建议 4 个小节，每节 2–3 段

### 4.1 Offline policy optimization and distribution shift

P1：AWR、CRR、IQL、BPPO、PPO-style off-policy variants 等如何约束 actor update。

P2：强调本工作研究的是 signed actor field 的动力学，不替代 critic conservatism 或 OOD value estimation。

### 4.2 Negative-advantage and low-probability update dynamics

P1：讨论 positive-only、negative filtering、BAPO、低概率 token、staleness / off-support suffix 等工作。

P2：明确已有工作已经发现负优势主导、低概率 token 风险或 entropy collapse；我们的差异是严格解耦、跨时间递推、连续—离散统一和因果干预。

### 4.3 Negative data for generalization and mode suppression

讨论 negative reinforcement、failure trajectory learning、OGPO、TOPR 等表明负信号可能改善多样性、pass@k、坏 mode 抑制和支持外推的工作。

**目的：** 让“负优势有益”成为有文献支撑的出发点，而不是只为了我们自己的结果临时改变叙事。

### 4.4 Entropy and support control

P1：entropy bonus、target entropy、KL/reference、temperature control。

P2：解释总体 entropy 不是 task-relevant support 的充分统计量，因此正文必须包含 entropy-matched baseline。

### 引用治理规则

- 每篇文献必须记录：官方标题、作者、年份、会议/arXiv ID、URL/DOI、与本文关系。
- 禁止引用无法核验的内部简称。
- 投稿前运行 BibTeX key 与正文引用自动一致性检查。
- 原稿中的 placeholder / hallucinated references 全部删除，不做“修补式保留”。

---

## 5. Problem Setup and Scope：建议 5 个段落

### P1：分析对象

定义静态数据分布 \(\mathcal D\) 上的 actor objective：

\[
J(\theta)=\mathbb E_{(s,a)\sim\mathcal D}[\widehat A(s,a)\log\pi_\theta(a\mid s)].
\]

明确它是许多 off-policy actor regression / approximate policy improvement 步骤的抽象。

### P2：不要把它伪装成无偏 policy-gradient theorem

明确：当数据不是当前策略采样且没有 importance correction 时，上式一般不等于真实 on-policy return gradient。本文研究的是该实际使用更新的稳定性与表示几何，而不是声称其无偏。

**直接回应 reviewer VKfL 的 Eq. 2 批评。**

### P3：advantage 条件

Actor step 中 \(\widehat A\) stop-gradient；它可来自 trajectory return、Q−V、group-relative reward 或固定标签。理论首先条件于给定 advantage，critic 联合训练作为移动目标在后文讨论。

### P4：正负质量与条件分布

定义 \(p,q,P_+,P_-\)，以及 global α、sample weighting 如何吸收到有效负质量中。

### P5：claim 层级

明确三层：任意可微策略的单样本结论；指数族输出分布的全局/局部几何结论；深网络参数空间的局部 pullback。读者从此处就知道哪些结论 general，哪些不是。

---

## 6. Theory：正文建议 4–5 页，完整证明放附录

### 6.1 Theorem 1：Single-sample repulsive surprisal dynamics

**段落 1：** 定义 \(S_\theta=-\log\pi_\theta\) 与负更新。

**段落 2：** 给连续时间精确恒等式：

\[
\frac{dS_\theta(z)}{dt}=|A(z)|\|\nabla_\theta\log\pi_\theta(z)\|^2.
\]

**段落 3：** 给离散步 Taylor 余项和步长充分条件。

**段落 4：** 解释该定理只保证被更新样本自身的 surprisal 增加；batch 中还存在交叉项，不能把单样本结论无条件扩展到每个样本。

### 6.2 Batch interference and directional coherence

**段落 1：** 推导 \(\Delta S_i\) 中 self-term 与 Gram cross-term。

**段落 2：** 定义 repulsive influence 的四因子：negative mass、score scale、coherence、reuse。

**段落 3：** 给出实验对应：单样本 far/near ratio 与聚合 ratio 的差异来自 coherence，而非额外 advantage。

### 6.3 Theorem 2：Signed-moment equilibrium in exponential families

**段落 1：** 写正则最小指数族和 signed objective。

**段落 2：** 定义 signed target \(\tau\)；推导梯度与 Hessian。

**段落 3：** 主定理三种情况：内部唯一稳态、边界解、域外无内部解。

**段落 4：** 解释该定理统一了稳定外推与 collapse，而不是把二者写成两个不相关故事。

**段落 5：** 给离散 Euler 局部步长条件，避免“连续时间稳定 = 任意学习率都稳定”的误解。

### 6.4 Gaussian branch

#### P1：fixed variance mean equilibrium

推导 \(\mu^*\)、\(q_{opt}\)、\(q_{crit}=p\)，建立 imitation ceiling → bounded extrapolation → persistent drift → runaway。

#### P2：learnable variance joint equilibrium

推导 \(\sigma^{2*}\) 和 signed variance feasibility。强调方差临界可早于均值临界。

#### P3：variance four quadrants

明确 A sign 与 standardized distance 共同决定 \(\sigma\) 的方向；删除原稿“both μ and σ expand”。

#### P4：fixed-sample Hessian correction

正文简要指出 pointwise Hessian 不定，expected Fisher SPD 不能证明固定样本联合扩张；详细矩阵放附录。

#### P5：far-field amplitude law

固定 \(\sigma\) 时 mean score 随距离线性、log-std score 随标准化平方距离增长；重复负更新可使距离对时间几何增长。不要写“梯度对原始距离指数增长”。

### 6.5 Categorical branch

#### P1：direct-logit score boundedness

证明 \(\|e_j-\pi\|\le\sqrt2\)。主动给出这一“反直觉限制”，避免审稿人指出后被动修改。

#### P2：support boundary dynamics

证明当 \(\pi_j\) 很小时 surprisal 仍以非零速率增长，因此 logit gap 可发散、概率可指数衰减到 simplex 边界。

#### P3：categorical signed target

把 full softmax 视为指数族，说明某分量为零/负分别对应边界/域外。

#### P4：semantic feature policy

解释有益未见动作外推需要 action feature / shared representation，而不是 token ID 有序。无序 ID + semantic embedding 是正式实验设定。

### 6.6 Neural-network pullback and scope

P1：定义输出自然参数 Jacobian \(J_\theta(s)\)。

P2：在 realizable fixed point 且残差为零附近，参数 Jacobian 是输出 covariance 的 pullback，给局部稳定性。

P3：明确非凸深网的全局收敛、Adam 动力学和多状态不可实现情形不在定理覆盖范围。

### 6.7 Moving critics and stale advantages

P1：每个 detached actor step 可用固定 advantage 理论分析。

P2：critic 更新让 signed target 移动；可给 tracking error bound 或定性讨论。

P3：方法只能限制错误 advantage 的破坏幅度，不能保证 critic 符号正确。

---

## 7. Method：先保留两个候选，外部实验后只选一个主方法

### 7.1 Candidate A：Exponential Remoteness Taper

统一定义：

\[
S_i=-\log\pi_\theta(a_i\mid s_i),\qquad
w_i^- = \exp[-\lambda(S_i-S_0)_+].
\]

**段落 1：** 只作用于负优势，正优势不变；\(S_i\) 和权重 stop-gradient。

**段落 2：** Gaussian 中 surprisal 含 Mahalanobis squared distance 与 log variance；categorical 中就是 token surprisal。

**段落 3：** 理论故事不是“原梯度指数增长，所以用 exp 抵消”，而是“指数 taper 支配任何有限阶 score growth，使加权 far-field influence 有界”。

**段落 4：** 计算开销只来自已有 log-prob，无第二次 backward。

### 7.2 Candidate B：Safety-only Stability-Budgeted Taper

先用 Candidate A 得到 sample weight，再计算 batch-level positive recovery 与 weighted negative budget；仅当稳定裕度不足时施加全局 \(\gamma_t<1\)。

**关键设计：** 不做完整 Hessian/Jacobian；只用 batch reductions。正常稳定区 \(\gamma_t=1\)，避免双重过度抑制。

### 7.3 主方法选择规则

- 若 Exp 在 D4RL + Countdown 中稳定领先或与 SBRC 持平，正文只保留 Exp，SBRC 放附录。
- 若 SBRC 在跨任务上显著降低超参敏感性并提升最差 seed，则正文采用 SBRC-Exp，Exp 作为核心 ablation。
- 不允许根据单个 toy 环境选择复杂方法。

### 7.4 Hard filtering / DRPO 的新位置

- 作为 \(w_i^-=0\) 的极端 conservative limit；
- 作为历史方法和 positive-only / top-k baseline；
- Optimistic DRO 的 closed form 仅在明确给定 uncertainty set 下成立，不再宣称现实任务的唯一最优方案；
- 原 DRPO 名称是否保留取决于最终主方法。若主方法不再是 DRO hard filtering，建议论文和方法改名，避免名实不符。

---

## 8. Controlled Experiments：正文机制证据

### 8.1 Protocol A：Source isolation

**问题：** 大梯度来自差样本还是 policy-relative remoteness？

**设计：** 质量 coordinate 与距离 coordinate 精确笛卡尔积；相同 advantage 复制到各距离。

**报告：** advantage ratio、score ratio、single-sample full-gradient ratio、aggregate ratio、coherence。

**文字边界：** 数十倍是该设置下效应量，不是普适常数。

### 8.2 Protocol B：Causal collapse

**问题：** far-field gradient 是相关变量还是传导路径？

**设计：** baseline、near-zero、far-zero、far-cap、global equal-budget、far-to-near。

**报告：** reward、OOD drift、collapse rate、时间顺序、paired CI、Wilcoxon。

**必须写清：** 乘积流形实验不回答 collapse；该非线性环境才回答因果传导。

### 8.3 Protocol C：Stable extrapolation and phase transition

**问题：** 为什么不直接 positive-only？

**设计：** 真实最优在最佳正样本支持之外；扫描负质量。

**报告：** positive-only ceiling、held-out beta、test reward、mean boundary、variance boundary、倒 U 曲线。

### 8.4 Generic categorical with unordered IDs

**设计：** 随机 action IDs + semantic embeddings；reward 由语义结构决定，不使用人为数轴。

**对照：** 打乱 reward–embedding 对应关系。预期 support suppression 仍存在，但有益方向外推消失。

### 8.5 Entropy-matched controls

比较 entropy bonus、target entropy、temperature floor 与 remoteness control。调节系数使最终 entropy 相近，再比较 task reward 和正确低概率支持保留率。

**目的：** 证明方法不是简单“增加随机性”。

---

## 9. External Validation：至少两类公共任务

### 9.1 D4RL / Hopper

**范围锁定：Hopper 不重复理想环境的全部实验，也不替代 C-U1。** Hopper 没有可直接观测的逐状态真实最优动作，因此不能复刻 E4 的 ground-truth 支持外推。它只重复可识别的子链：advantage-matched near/far 梯度来源、Positive-only 后固定负样本的 phantom 动力学、以及少量 signed/Near-zero/Far-zero/Global 干预。完整方法效果由标准 offline RL + environment rollout 单独回答。

#### Mechanism subsection

- Hopper-medium：保守、较窄分布上的外部机制证据；
- Hopper-medium-replay：更宽 replay mixture，验证自然 near/far；
- Hopper-medium-expert：明显质量混合，用于方法效果和 stable extrapolation。

分析 protocol：用正 advantage 训练 actor；固定负样本只测 phantom gradient；按 \(|A|\) 分桶匹配 near/far；报告 standardized distance、Gaussian score、full-parameter gradient。

#### E7-Q2：Gaussian 二次 log-scale 分支的独立外部验证

**新增 claim。** C-U1 已在受控 Gaussian 输出空间中解析并数值核对：mean-score 分支随距离一次增长，校正后的 log-scale-score 分支随标准化距离平方增长。E7-Q2 不重新证明这一恒等式，而检验真实 D4RL Hopper 数据、learned critic、状态条件 actor 与自然 near/far negatives 是否实际进入该二次分支显著作用的远场区域，以及该分支是否传导至 full-parameter gradient 和长期动力学。

**坐标与解析量。** 若 actor 为 tanh-squashed diagonal Gaussian，对数据动作使用冻结 inverse-squash 坐标：

\[
u=\operatorname{atanh}(\operatorname{clip}(a,-1+\epsilon,1-\epsilon)),
\qquad
z_j=\frac{u_j-\mu_j(s)}{\sigma_j(s)},
\qquad
r=\lVert z\rVert_2.
\]

Gaussian base-distribution 的输出分支 score 为：

\[
g_{\mu,j}=\frac{u_j-\mu_j}{\sigma_j^2},
\qquad
g_{\xi,j}=z_j^2-1,
\qquad \xi_j=\log\sigma_j.
\]

因此 component-wise 校正关系为 `g_xi,j+1=z_j^2`，聚合校正量为：

\[
Q_\xi=\sum_j(g_{\xi,j}+1)=\lVert z\rVert_2^2=r^2.
\]

`Q_xi` 用于检验二次标准化距离律；实际优化风险必须同时报告未校正的 `||g_xi||`、`||g_mu||`、joint output-score 与 full-parameter gradient norm。raw action distance 和 pre-squash distance 继续报告，但理论检验以 Gaussian base-coordinate 的 standardized residual 为准，避免 tanh 边界压缩造成表面斜率失真。

**预注册分析。**

1. 使用 learned critic 产生 advantage；只在负 advantage 内进行 near/far 比较，并在 `|A|` 上匹配或分层校正，避免把样本更差误当成距离效应。
2. 按 standardized distance 分桶，分别报告 mean、raw log-scale、corrected `Q_xi`、joint output 与 full-parameter gradient；同时报告 `log-scale/mean` contribution ratio。
3. 检验 mean 分支相对距离的一次增长，以及 `Q_xi` 相对 `r` 的二次增长；使用解析分解与 output-tensor autograd 交叉检查。
4. 沿训练时间报告二次分支贡献、sigma/support、mean saturation、actor loss、normalized return 和所有非有限事件；任务性能崩溃、支持/方差边界事件与 NaN/Inf 数值崩溃必须分开。
5. 通过 Far-zero、Far-cap 与等预算 Global control 检验：抑制远场影响后，full-parameter gradient、support contraction 和长期任务动力学是否缓解。机制表与标准 offline-RL 方法效果表分开呈现。
6. 使用预登记 paired seeds、置信区间与终态审计；旧 600-step probe 不能升级为 E7-Q2 正式结果。

**独立验证判据。** 只有同时满足以下条件，才称为 Gaussian 二次 log-scale 远场机制在 Hopper 中的独立外部验证：真实数据自然产生足够大的 standardized distance；二次 log-scale 分支相对 mean 分支显著增强；该增强对实际 full-parameter gradient 或长期支持/性能动力学具有可测贡献；定点远场控制产生相应缓解；结果通过 paired seeds 与终态审计。若只验证解析 score 与 autograd 一致，则仅为实现一致性检查。

**结论边界。** E7-Q2 不研究神经网络 pullback 的阶数，不声称全参数梯度对距离严格二次；也不预设 Exp、Linear、Global α、SBRC 或 Hybrid 的方法排名。Hopper 只提供外部有效性，不能替代 C-U1 的受控机制识别。


#### Method subsection

在 IQL 或既有 offline RL actor 上插入 Exp/SBRC 负优势控制；比较 normalized return、多 seeds、critic error sensitivity。不能只做 phantom analysis 就宣称方法提高 D4RL performance。

### 9.2 Countdown token arena

**历史入口（v12 登记，已由 v22 替换，仅作 provenance，不得执行）：** `/mnt/data/countdown_qwen_arena_onefile_v3.py`。

以下 v3 命令仅保留历史记录，不是当前执行入口：

```bash
python countdown_qwen_arena_onefile_v3.py run \
  --model_path /ABS/PATH/TO/QWEN-INSTRUCT \
  --work_dir /ABS/PATH/TO/COUNTDOWN_RUN \
  --gpu 0 --preset auto --memory_mode auto
```

上述 v3 流程中的强制 SFT、只评最佳 checkpoint、八方法 arena 和自动 QLoRA 选择均已被 v22 覆盖；该段只作 provenance。当前实验仍未完成真实 Qwen/CUDA/BF16-LoRA 端到端运行。

#### v28 当前协议覆盖：v4.2.0 一键 BF16-LoRA pilot

当前唯一推荐代码入口：

```bash
python3 scripts/run_countdown_pilot.py \
  --model_path /ABS/PATH/TO/QWEN2.5-0.5B-INSTRUCT \
  --work_dir /ABS/PERSISTENT/PATH/TO/COUNTDOWN_RUN
```

该入口自动使用冻结的 `preset=0.5b`、`memory_mode=bf16`、`seed=1234` 和四方法集合；默认选择全部可见 GPU（最多 8 张），无需本地 AI决定是否 SFT、如何分配方法、评测哪些 checkpoint 或何时打包。底层独立子命令继续保留用于测试和故障定位，但不得由本地 AI 临时拼接成另一套正式流程。

1. Base-first、SFT fallback、matched-pair、Park-inspired family holdout、四方法、checkpoint 与终态审计均沿用 v21。
2. 机制 probe 的 near/far advantage 仍固定为 `A=-1`，不乘方法训练的共同尺度。
3. 方法训练不再固定 `alpha=0.7`。先在固定 training calibration subset 上计算共同 `beta=G_pos_rms/G_neg_uncontrolled_rms`，再对三个含负优势方法冻结。
4. 同一 calibration 同时计算 `global_matched` 的 `gamma=G_neg_controlled_rms/G_neg_uncontrolled_rms`；validation/test task metric 不参与 `beta` 或 `gamma`。
5. 当前冻结 pilot 规模、优化器、LoRA 配置和 seed 见文档顶部 v22 增量记录及 registry；任何修改都需要新的版本登记。
6. 当前 runner 不实现 full FT。LoRA pilot 出现可复现信号后，另行登记 0.5B full-FT confirmation 的显存、优化器、步数、seeds 与 checkpoint 策略，再写代码。
7. `presence` 与 `success` 分开：unseen-structure success 必须 verifier 正确；per-pattern precision 分别报告 greedy 与 sampled 的 attempts/correct/precision，零尝试记为 `null`。
8. 非有限失败保存精确 optimizer-step 前的 trainable-adapter 状态，并记录 `failure_detected_at_step` 与 `last_finite_step`，不再使用最近一次验证 checkpoint 代替。
9. 顶层 `pilot` / `engineering_smoke` 标签必须传入 SFT 和 method manifests；直接子命令默认 `standalone_unclassified`。
10. 正式 pilot 必须通过 `scripts/run_countdown_pilot.py` 进入 hardened guard；guard 自动绑定当前完整 commit、监督前台进程并在成功/失败时生成持久 artifact。
11. 安全多 GPU 调度不得改变随机数据生成：`build_offline` 继续单 GPU；机制/calibration、方法训练和 checkpoint evaluation 才允许并行。
12. 成功门禁要求 `RUN_COMPLETE.json`、`terminal_audit.json` 与 `arena_summary.csv` 同时存在；本地只生成 CSV 不构成完成。

#### v21 历史协议：v4.1 审计式 BF16-LoRA pilot（由 v22 覆盖 alpha 与配置登记）

本小节覆盖 v18 中“单个 oracle signature 拆分、三方法比较、只保留最佳 checkpoint”的执行细节；v18 保留作 provenance。当前执行 ID 为 `EXT-C-E8-V4.1`，状态为“尚未运行”。

1. 先评测未经 Countdown 训练的 Qwen Instruct 0.5B Base；仅在既有能力门禁失败时执行最小 SFT fallback。
2. `positive_only`、`controlled_negative`、`uncontrolled_negative`、`global_matched` 统一使用 BF16 LoRA；QLoRA 只允许标注为工程 smoke fallback，不进入方法排名。
3. 使用 canonical-pattern-first、容量审计的近似平衡生成与 held-out pattern-family 拆分；普通 verifier success 只作为任务性能，结构泛化以 family coverage 与 per-pattern precision 报告。
4. Held-out family 不得作为训练 positive 或 near/far negative completion 出现。
5. Near/far 均为合法、使用全部数字、reward=0 的错误表达式并固定 `A=-1`；追加采样仍无法匹配则丢弃，主训练与 probe 只读取 matched pair。
6. `global_matched` 在固定 calibration 数据上匹配 Controlled 的 RMS 负梯度 norm，并冻结 global gamma；test 不参与校准。
7. 权重仅保存在服务器本地：正常结束保留 best+terminal，非有限失败保留 best+last-finite；不复制 foundation model，不默认保存 optimizer state。
8. 分别评测共同起点、best、terminal/last-finite，记录 stop reason；任务性能、结构/支持退化和 NaN/Inf 分开报告。
9. 单 dev seed 只标记 pilot；正式升级要求 paired held-out seeds、终态审计和持久 artifact。
10. LoRA pilot 出现可复现信号后，才在 0.5B 上做统一 full-FT 确认。

#### v18 历史协议：Base-first 0.5B 最小实验（由 v21/v22 覆盖）

本小节覆盖下方 v12 的“先 SFT、3B 主 arena、八方法比较”计划；旧计划不删除，仅作为 provenance。用户已明确授权先运行 EXT-C / E8，因此本地 0.5B 验证可在不改变 D-U1 职责的前提下先行。

1. **零训练 Base 门禁：**先直接评测原始 0.5B Instruct checkpoint。验证集 `greedy_success>=0.15` 且 `valid_rate>=0.80` 时跳过 SFT；所有方法从同一未训练 LoRA adapter 开始。未过门槛才进入最小 SFT fallback，SFT 后 `greedy_success>=0.15` 才继续。
2. **结构拆分：**训练、验证、测试的 canonical operator-tree signatures 两两不重叠。离线正答案不得把验证/测试结构重新引入训练支持。
3. **机制 probe：**从冻结参考策略构造 reward 同为 0 的合法 near/far 错误表达式；匹配 token 长度差 `<=2`、树深差 `<=1`、数值误差比 `<=4`，surprisal gap 默认至少 `0.5`。正式 probe 至少 16 个匹配 pair，默认报告 32 个 pair。
4. **固定负优势：**near/far 均使用 `A=-1`；报告 trainable-adapter gradient norm、target surprisal suppression、correct-answer collateral change。categorical direct-logit score 有界，不能把该结果写成 Gaussian 式无界梯度爆炸。
5. **最小方法：**`positive_only`、`controlled_negative`、`uncontrolled_negative`。Controlled 保留 near 分支，对 far token 使用 detached surprisal taper；不预设其必然优于其他方法。
6. **配对规则：**三组方法共享初始化、离线数据、训练 seed、验证题和 generation seed。Base checkpoint 与共同初始 checkpoint 都只评测一次，不视为额外训练方法。
7. **主指标：**greedy verifier success、pass@k、valid rate、greedy/pass@k unseen-structure success；unique correct structures、entropy/weights 和数值状态作为诊断。任务性能退化、有效支持/结构覆盖退化和 NaN/Inf 分开报告。
8. **规模策略：**0.5B 是当前主实验；3B 仅在 0.5B 基础能力不足或需要关键结论复验时运行；7B 不阻塞当前论文结论。
9. **结果状态：**代码测试不构成 E8 结果。真实模型运行前状态保持“尚未运行”；单 seed 只能标记 pilot，多 seed 配对且满足预登记门禁后才能升级。

#### 历史模型阶梯（v12，已由 v22 替换，不得执行）

- 0.5B：pipeline 和超参快速筛选；
- 3B Instruct：正式主 arena；
- 7B：冻结方法后的最终确认，而不是全部网格搜索。

#### 历史正式流程（v12，已由 v22 替换，不得执行）

SFT 达到至少 15%–20% greedy verifier success → 冻结 checkpoint → 同一模型采样正/near-negative/far-negative 轨迹 → 各方法从同一 checkpoint 训练。

#### 主指标

Verifier greedy success、pass@k、valid rate、token surprisal、正确低概率 token 保留、错误 token suppression、entropy、有效 support、\(\gamma_t\) 与平均权重。

#### 历史 Baselines（v12，已由 v22 替换，不得执行）

Positive-only、uncontrolled、global α、Exp、entropy bonus、target entropy、SBRC/Hybrid，以及适用的现有 low-probability / surprisal-aware 方法。

### 9.3 Recommendation application（可选但有价值）

若继续保留推荐实验，必须至少一个公共数据集 + 现代 backbone，例如 SASRec 或 generative retrieval backbone。旧 RecSim 可放附录作为工业形态 stress test，不能继续作为唯一主实验。

如果短期无法完成公共推荐实验，则主文不再声称“生成式推荐 SOTA”；推荐只作为 motivating application 和未来工作。

---

## 10. Results section 的段落顺序

### P1：先回答机制是否存在

Protocol A + Hopper phantom：distance-matched / advantage-matched far negatives 有更大 score 和全参数梯度。

### P2：再回答是否因果导致 collapse

Protocol B near/far intervention，给最强 paired effect。

### P3：回答负梯度是否有益

Protocol C 倒 U：positive-only ceiling、中等负推力最佳、过强后失稳。

### P4：回答连续—离散是否统一

共同 surprisal / signed target，表型分叉：amplitude versus support。

### P5：回答方法是否不仅仅维持 entropy

entropy-matched comparison。

### P6：回答是否外部有效

D4RL normalized return + Countdown verifier success。

### P7：超参和计算开销

报告 λ、S0、γ 的敏感性；训练时间、额外显存、是否需要第二次 backward。

---

## 11. Discussion and Limitations：主文必须单设

### 段落 1：机制边界

受控环境证明 far-field path 是充分且主导的传导路径，不代表真实任务的唯一 collapse 原因。

### 段落 2：方向可靠性

理论控制 influence，但“信息价值随 distance 下降”仍主要由实验和任务结构支撑，尚非无条件定理。

### 段落 3：critic 与 optimizer

固定 advantage 理论不保证 critic 正确；Adam、PPO clipping、importance ratio 等改变具体动力学。

### 段落 4：神经网络全局性

指数族输出结论不等于非凸深网全局收敛；参数共享可产生跨样本干涉。

### 段落 5：reward 与 support

support collapse 不必然导致 reward collapse；需要任务中有价值动作被压制的因果连接。

### 段落 6：方法边界

Exp taper 可能过度削弱真正有用的 rare negative；SBRC 可能受 batch estimate 噪声影响。需要跨任务验证。

---

## 12. Reproducibility and Ethics

### Reproducibility

- 主文列明数据版本、模型版本、seed 划分、开发/最终 untouched seeds、训练步数、batch、学习率、硬件。
- 公布统一代码、配置、逐 seed CSV、bootstrap 脚本、图表源数据。
- 所有主图可由一个 `run_all.sh` 或单一入口重建。
- 受控环境与外部数据 loader 分开，但共享训练和诊断接口。

### Reference integrity

- 投稿包中加入脚本检查 BibTeX 是否含 placeholder 字符串、空 venue、虚构 arXiv ID。
- 每个相关工作论断对应 primary source 页码或定理。
- 不再引用“看起来像论文”的未核验材料。

### Ethics

- 删除原稿中未经证据支持的 Green AI 和工业 ROI 强断言。
- 说明负反馈控制可能错误压制少数但有价值的行为；方法需要 reward/critic quality 和公平性审计。

---

## 13. 主图与主表蓝图

### Figure 1：统一机制示意图

左：local negative shaping；中：signed target 仍在 moment domain 内，稳定外推；右：target 越界，Gaussian/categorical 分叉失稳。

### Figure 2：Source isolation

advantage 相同、distance 变化、score / gradient amplification。

### Figure 3：Causal intervention

near-zero 与 baseline 重合；far-zero/far-cap 救援；同时显示 OOD drift。

### Figure 4：Stable extrapolation phase diagram

固定方差与可学习方差的 reward、mean、sigma 边界，叠加理论临界线。

### Figure 5：Categorical support dynamics

direct-logit score 有界但 probability/logit gap 到边界；semantic shuffle 对照。

### Figure 6：External validation

Hopper medium/replay/expert mechanism + method；Countdown verifier success 与 entropy-matched comparison。

### Table 1：理论对象与分叉

Gaussian / categorical / semantic energy policy 的 mean-domain、临界条件和失稳表型。

### Table 2：受控因果结果

20-seed paired effects。

### Table 3：D4RL 方法结果

normalized return，含 IQL/AWR/positive-only/global/Exp/SBRC。

### Table 4：Countdown arena

3B 主结果，7B 核心确认；greedy、pass@k、valid、entropy/support。

---

## 14. 原稿内容的删除、降级和复用清单

### 完全删除

- `Lastname et al.` 等虚构/占位引用；
- “negative advantages inevitably cause exponential explosion” 的无条件表述；
- expected Fisher SPD 推导固定样本所有方向扩张；
- “Both μ and σ expand”；
- “hard filtering is mathematically necessary for survival”；
- “simulation alone proves general offline RL superiority”；
- 未定义的 Soft-Base 术语。

### 降级到附录或历史背景

- Optimistic DRO hard-filtering 完整推导；
- 旧 RecSim 大部分结果；
- C1/C2/V0 开发环境；
- offline-to-online 推荐 curriculum，除非重新做公共可复现验证。

### 直接复用但需重写叙事

- Gaussian score 几何；
- phantom gradient 诊断，但必须报告真实 sigma 轨迹和梯度符号；
- 远场/近场干预；
- positive-only 对照；
- 原 DRPO 的“repulsive optimization”术语，可作为历史起点。

---

## 15. 写正文前的决策门槛

1. **方法门槛：** Exp 与 Safety-only SBRC 至少在受控 continuous、semantic categorical、D4RL、Countdown 四类中完成筛选，只留一个主方法。
2. **Countdown 门槛：** SFT greedy verifier success ≥15%，主 arena 使用 3B；0.5B 不承担最终结论。
3. **D4RL 门槛：** medium-replay 至少完成多 seed mechanism；方法结果需要 environment rollout 和 normalized return。
4. **引用门槛：** 所有引用核验完毕，自动审计无 placeholder。
5. **claim 门槛：** 正文每个强 claim 对应定理、受控干预或公共 benchmark 中至少一种直接证据。
6. **reproducibility 门槛：** 代码与配置在新环境中从零运行一次。

---

## 16. 建议的最终章节目录

1. Introduction
2. Related Work
3. Problem Setup and Scope
4. Repulsive Signed-Moment Dynamics
   - 4.1 Single-sample surprisal dynamics
   - 4.2 Batch interference
   - 4.3 Signed-moment equilibrium
   - 4.4 Gaussian instability
   - 4.5 Categorical support collapse
   - 4.6 Neural-network pullback and moving critics
5. Remoteness-Controlled Negative Updates
6. Controlled Mechanism Experiments
7. External Validation
   - 7.1 D4RL continuous control
   - 7.2 Countdown token policy
   - 7.3 Recommendation application（可选）
8. Discussion and Limitations
9. Conclusion

Appendix:

- Full proofs
- Formula self-checks
- Additional seeds and architecture robustness
- Entropy-matched sweeps
- D4RL data details
- Countdown data/verifier
- Optimistic DRO and original DRPO connection
- Reference audit and full reproducibility checklist


---

# 附录 A：v11 恢复记录（原第 0 节，完整保留）

# 0. 本次恢复的明确结论

## 0.1 “大一统连续环境”当前是否已经实现

**没有。** 当前代码实际上包含两个连续 contextual-bandit 数据生成器：

- Product 环境：用于质量—距离解耦的瞬时梯度来源与 Near/Far 因果干预；
- Extrapolation 环境：用于 positive-only 上限、稳定外推及均值—方差平衡。

二者虽然共享 6 维状态、2 维动作、Gaussian actor 和部分统计代码，但不是同一个环境，也不是同一批状态—动作几何。因此 v7 中“统一 benchmark”的命名高估了完成度。Categorical 因动作空间不同而使用独立环境是合理的。

## 0.2 Product 与 Extrapolation 能否真正合并

**能，没有根本技术障碍。** 过去分开的主要原因是：

1. Product 环境追求“同 reward/advantage、只改变策略距离”的严格变量隔离；
2. Extrapolation 环境追求“最佳已观察正样本、未见真实最优动作、反方向负样本”的解析结构，并使用分布期望降低采样噪声；
3. 分开实现更快、更容易获得单项结果。

这属于实现便利和局部变量隔离，不是必须分离的数学限制。此前没有向用户说明并取得同意，是需求执行错误。

## 0.3 真正统一的解决方案：等奖励轮廓构造

对每个 6 维状态 `s`，在 2 维动作空间中定义：

- `a_plus(s)`：训练中最佳已观察正样本支持；
- `a_star(s)`：训练中未展示的任务最优动作；
- `a_minus(s)`：位于 `a_plus` 另一侧、能够提供有益排斥方向的近场负样本；
- 一条以 `a_star` 为圆心、经过 `a_minus` 的等奖励轮廓。

在该轮廓上复制多个负动作。由于它们到 `a_star` 的距离完全相同，所以 reward 与 advantage 完全相同；但它们到 `a_plus`（也是 positive-only 预训练策略的初始均值）的距离不同。因此同一个标准 contextual bandit 环境同时满足：

1. **来源隔离**：badness/advantage 严格相同，只改变 policy-relative distance；
2. **Positive-only 动力学**：策略拟合 `a_plus`，同时监测固定负样本梯度；
3. **Near/Far 因果干预**：同一批负样本按当前策略距离动态分组；
4. **稳定外推**：`a_minus` 在 `a_plus` 的反方向，适量排斥可把策略推向 `a_star`；
5. **唯一真实最优动作**：ground-truth reward 直接由到 `a_star` 的二维距离定义；
6. **均值—方差联合动力学**：同一个可学习方差 Gaussian actor 完成全部连续实验。

![统一连续环境单状态几何](master_recovery/unified_environment_geometry.png)

本构造已经写成环境原型并通过以下不变量检查：

- 负样本跨距离复制后的 reward 相等；
- advantage 相等；
- 初始策略距离严格递增；
- 正样本 advantage 为正，负样本 advantage 为负；
- `a_plus -> a_star` 与 `a_minus -> a_plus` 的任务方向一致。

原型代码：`/mnt/data/drpo_unified_continuous_environment_v1.py`。当前只完成数据几何与不变量审计，尚未将四类训练 protocol 全部接入并正式重跑。

## 0.4 从昨日下午到现在是否“白忙活”

必须分开评价：

- **对于用户明确要求的“把连续小环境真正合并为一个环境并完整重跑”任务：基本没有完成。** 合并的是代码接口和部分 protocol，不是环境；因此不能把此前重跑算作该交付的完成。
- **并非所有工作都没有价值：** Gaussian 方差方向修正、expected-Fisher 证明纠错、exponential-family 理论、categorical 实验、Hopper 探针和训练收敛审计提供了新信息。但这些工作不能替代大一统环境任务，也不能用于掩盖其未完成状态。
- 已在旧分离环境上重复运行的结果仍可作为开发证据和回归基线，但新论文的统一连续主结果必须在真正合并后的环境重新运行。

---

# 附录：本恢复版之后的变更协议

每次更新必须在文档开头加入一条 changelog，至少包含：

- 新增内容；
- 修正内容；
- 拟删除内容（默认无）；
- 理论 claim 变化；
- 实验状态变化；
- 新增变量及必要性；
- 受影响的论文段落。

若需要删除，必须在生成新版本前向用户展示逐项删除清单并取得确认。

## 11.6 用户审阅后的 E4 变量治理与结论边界修正（2026-06-24）

- **D-U1/E6 暂停。** 按用户要求，在 C-U1 结果、术语和复现代码完成审计前不启动 categorical smoke 或长程训练。
- **β 使用不合规。** E4 报告中的 β 只是 `a_plus -> a_star` 方向的归一化投影位移，不是模型参数；但第 7.4 节已将绘图归一化 β 撤出主变量体系。将其重新作为主表符号违反变量治理规则。后续不新增替代希腊符号，统一使用文字指标“沿隐藏最优方向的归一化投影位移”，代码字段为 `normalized_extrapolation_displacement`。
- **α 的边界。** `alpha_local` 仅为代码配置名，映射到既有核心变量 α；E4 中它乘在方向可靠的局部负梯度组上。论文仍使用 α，并显式说明所作用的负样本子集。
- **状态分类补强。** E4 必须分开：稳定良好固定点、稳定坏固定点、数值有限的持续漂移/runaway、数值/支持边界事件。主判据为 reward、均值相对 `a_plus/a_star` 的原始距离、归一化净动力场残差、位移窗口斜率、更新 norm、sigma/log-sigma 及 2× horizon 是否反转。
- **方差解释修正。** sigma 收缩不是任务 reward 下降的充分条件。它表示完整联合稳态可能更早消失，并放大标准化距离和梯度敏感度；任务是否失效仍由均值所在 reward 区域和正负净梯度平衡决定。E4 数据本身显示 sigma 显著收缩阶段 reward 仍可提高。
- **方向效用降级。** 当前方向诊断由环境几何有意构造，保留为 sanity check/附录解释，不作为跨环境主要结论。一般性方向—距离规律需多几何或外部任务验证。
- **复现性卡点。** Master 引用的统一源代码和多数 raw run 目录在当前会话文件系统中缺失，仅有汇总报告与三张 E4 图。已生成 `C_U1_REPRODUCIBILITY_AUDIT.md`；在恢复原代码或完成严格 reimplementation 重跑前，不宣称已有可下载的精确复现包。


# v15 附录：上传环境代码兼容性与一键重建登记

## A. 上传 `drpo_unified_continuous_environment_v1 (1).py` 与正式 C-U1 的差异

该文件可作为几何骨架来源，但不是正式 E1-E4 环境，差异包括：

1. 状态分布为 `Uniform[-1,1]`，正式协议为 `N(0,I_6)`；
2. 状态数为 1024 train / 2048 test，正式为 4096 / 4096；
3. 每状态 2 个带高斯噪声正动作，正式为 4 个等 reward、质心精确等于 `a_plus` 的正动作；
4. 负动作数量为 6，正式为 8，且正式 index 0 为 `a_minus`、index 4 为最远动作；
5. reward width / baseline 为 0.80 / 0.50，正式为 0.75 / 0.40；
6. 文件只有环境与 invariant audit，没有共享两层 MLP actor、positive-only 饱和训练、动态标准化 Near/Far、E1-E4 driver、终态审计、逐 seed 轨迹和控制预算匹配。

因此不能原样运行来复现既有结果。

## B. 重建代码状态

- 文件：`/mnt/data/drpo_cu1_e1_e4_oneclick.py`
- 默认命令：`python drpo_cu1_e1_e4_oneclick.py`
- 不需要编辑源码或传入超参数；正式配置全部冻结在 `Protocol` dataclass。
- 自动选择 CUDA/CPU，固定结果目录，支持中断后自动跳过已完成 seed。
- 输出包括环境审计、manifest/hash、E1-E4 逐 seed JSON/CSV、逐步轨迹、bootstrap CI、相图、方差边界阈值/学习率稳健性和 reference regression。
- 当前状态：`python -m py_compile` 通过；开发用 CPU smoke 全流程通过；正式 20-seed 结果尚未运行，因此不能提前声称数值已复现。

## C. 重新登记的任务失效判据

由于旧瞬时 driver 未持久化，旧 E3 报告中的任务阈值精确值不可审计。重建代码明确预注册：held-out-context reward 低于同 seed positive-only reference 的 45%，且连续 3 个评估点满足，记为任务失效事件。最终结论同时报告连续漂移、终态 reward、2× horizon 和数值/支持状态，不让单一阈值承担主要结论。
