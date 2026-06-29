# Global research core and governance boundaries

> Generated Stage 4 minimal-context shadow module. Do not edit manually.
> `docs/handoff.md` and `experiments/registry.yaml` remain authoritative.

- Module ID: `global_core_governance`
- Responsibility: Preserve the unique-master rule, terminology, scientific scope, and non-destructive governance constraints.
- Source hash: `9fe36353a77f8888d9d3fbd71dea78d712ca15f20ceeb41574c0c621d69ffbdb`

## Source 1: docs/handoff.md: # 0. 研究与执行原则（每次新会话首先阅读） -> # 1. 论文最终目标与两条主工作线

# 0. 研究与执行原则（每次新会话首先阅读）

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
