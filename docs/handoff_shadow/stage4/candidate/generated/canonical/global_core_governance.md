# Global research core and governance boundaries

> Stage 4B lossless source-promotion shadow candidate.
> Generated from `docs/handoff.md`; do not edit while the manual handoff remains authoritative.

- Owner type: `canonical_module`
- Owner ID: `global_core_governance`
- Responsibility: Preserve the unique-master rule, terminology, scientific scope, and non-destructive governance constraints.
- Dependencies: none
- Content-contract topics: `unique_master_document`, `document_before_experiment`, `non_destructive_history`, `terminal_audit_governance`, `controlled_external_validity_boundary`
- Owned source blocks: 20
- Registry references are pointers only; `experiments/registry.yaml` remains the sole editable registry source.

## Registry references

- `experiments`: none
- `development_experiment_registrations`: none

## Owned source blocks

<!-- STAGE4B-SOURCE-BLOCK:B000020:START -->
# 0. 研究与执行原则（每次新会话首先阅读）
<!-- STAGE4B-SOURCE-BLOCK:B000020:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000021:START -->
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
<!-- STAGE4B-SOURCE-BLOCK:B000021:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000022:START -->
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
<!-- STAGE4B-SOURCE-BLOCK:B000022:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000023:START -->
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
<!-- STAGE4B-SOURCE-BLOCK:B000023:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000024:START -->

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

<!-- STAGE4B-SOURCE-BLOCK:B000024:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000025:START -->
## 0.1 当前执行门禁

- C-U1 E1/E2/E3：现有正式状态保留。`C-U1-E4-ADAM-RERUN` 保留“有限训练步数验证”；`C-U1-E4-CONV-01` 经用户明确审阅，在保留原 18/20 门禁失败事实的前提下，按 15/20、16/20、15/20 目标状态、0/60 明确相反终态与 60/60 长程科学角色不反转，闭合为“已长期验证”。`C-U1-E4-TAPER-01` 已完成 `220/220` 正式 runs、终态审计与交付；20/20 paired seeds 支持 Quadratic 在 anchor-normalized protocol 下比 Linear 更强抑制远场负梯度，但 200 controlled/positive runs 未形成稳定候选，故科学状态为 **有限训练步数验证**，不得称 long-run validated 或形成 universal method ranking。
- D-U1：E5 已长期闭环。E6 pilot/focused development 保持 development evidence；`D-U1-E6-SEMANTIC-LONGRUN-01` 已完成 `360/360` formal runs、2x 终态审计与 durable delivery，科学状态为 **long-run validated**。`D-U1-E6-TAPER-01` 的 predecessor delivery 已满足，但其距离坐标、paired protocol、新 untouched seeds 和独立 runner 尚未冻结/实现，仍是 review-required + blocked。
- Hopper/D4RL：`EXT-H-E7-Q2` 是 E7-MECH，runner/config 已实现但 formal launch 仍等待受控 taper 阶段交付；`EXT-H-E7-BENCH-01` 是 D4RL MuJoCo locomotion 9-task 方法效果表，等待 E7-MECH 与受控方法 shortlist 冻结。
- Countdown：`EXT-C-E8-V4.2` 是当前 E8-MECH/pilot；`EXT-C-E8-V4.1` 仅保留 provenance；`EXT-C-E8-SCALE-01` 是更大固定数据与模型规模验证，等待 E8-MECH 和 E7-BENCH。

<!-- STAGE4B-SOURCE-BLOCK:B000025:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000026:START -->
<!-- HANDOFF-DELTA-BLOCK:section_end:v52-countdown-current-gate-override:START -->
- **Countdown v52 覆盖：** `EXT-C-E8-V4.3` 取代 V4.2 成为当前 E8-MECH/focused pilot；V4.2 只保留 matched-pair mechanism provenance。`EXT-C-E8-SCALE-01` 继续等待 V4.3 与 E7-BENCH，不因本次实现自动解锁。
<!-- HANDOFF-DELTA-BLOCK:section_end:v52-countdown-current-gate-override:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000026:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000027:START -->
<!-- HANDOFF-DELTA-BLOCK:section_end:v55-du1-e6-semantic-gap-current-gate:START -->
- **D-U1 v55 覆盖：** `D-U1-E6-SEMANTIC-GAP-LONGRUN-01` 已完成 `100/100` 正式 runs、2× horizon 与终态审计，科学状态为 **有限训练步数验证**；45/100 plateau、55/100 persistent-drift-or-inconclusive，禁止稳态方法排名或无新登记重跑。`D-U1-E6-TAPER-01` 的 successor-delivery 条件已满足，但其四项协议/实现门禁仍未完成，继续 review-required + blocked。
<!-- HANDOFF-DELTA-BLOCK:section_end:v55-du1-e6-semantic-gap-current-gate:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000027:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000028:START -->
<!-- HANDOFF-DELTA-BLOCK:section_end:v56-e6-parent-closure-current-gate:START -->
- **v56 E6 父 claim 关闭覆盖：** E6 的论文核心 claim 现已范围受限关闭；主 long-run 与两个 gap 子实验的原科学状态分别保持 `long_run_validated / finite_step_validated / finite_step_validated`。`D-U1-E6-TAPER-01` 保留为可选非门禁未来工作。当前下一正式 route item 为 `EXT-H-E7-Q2`，registry 状态为 **implemented + ready + active + not_run**；启动后仍须走 canonical hardened guard，且在 raw-complete、终态审计、打包和交付前不得声称 E7 完成。
<!-- HANDOFF-DELTA-BLOCK:section_end:v56-e6-parent-closure-current-gate:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000028:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000029:START -->
<!-- HANDOFF-DELTA-BLOCK:section_end:v57-countdown-offline-bank-current-gate:START -->
- **Countdown v57 覆盖：** `EXT-C-E8-V4.4-OFFLINE-BANK` 是用户批准的当前离线 focused pilot；V4.3 保留为 fixed-pair predecessor。V4.4 只改变固定负样本覆盖与 current-policy near/far reselection，不引入在线数据刷新。`EXT-H-E7-Q2` 仍是下一正式 route item，`EXT-C-E8-SCALE-01` 继续 blocked。
<!-- HANDOFF-DELTA-BLOCK:section_end:v57-countdown-offline-bank-current-gate:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000029:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000030:START -->
<!-- HANDOFF-DELTA-BLOCK:section_end:v59-countdown-offline-bank-tuning-current-gate:START -->
- **Countdown v59 覆盖：** `EXT-C-E8-V4.5-OFFLINE-BANK-TUNING` 是当前用户批准的离线 focused successor；V4.4 作为 frozen-bank predecessor 保留。V4.5 只调 calibrated global negative multiplier 与 exponential taper lambda，禁止在线刷新、方向筛选或模型规模同时变化。`EXT-H-E7-Q2` 仍是下一 formal route item，`EXT-C-E8-SCALE-01` 继续 blocked。
<!-- HANDOFF-DELTA-BLOCK:section_end:v59-countdown-offline-bank-tuning-current-gate:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000030:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000031:START -->
<!-- HANDOFF-DELTA-BLOCK:section_end:v60-e4-taper-current-gate:START -->
- **E4-TAPER v60 覆盖：** `C-U1-E4-TAPER-01` 仍为 finite-step validated。四个后续 ID 已获用户批准并登记，但全部保持 blocked：先冻结并实现 `NEAR-RETENTION-01`，交付后才允许冻结 `BUDGET-MATCH-01`；二者交付并冻结 shortlist 后才允许 `CONV-01`；最后才用 untouched seeds 执行 `CONFIRM-01`。原实验禁止自动延长，几何 robustness 不作为当前门禁。
<!-- HANDOFF-DELTA-BLOCK:section_end:v60-e4-taper-current-gate:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000031:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000032:START -->
<!-- HANDOFF-DELTA-BLOCK:section_end:v61-e4-taper-near-retention-current-gate:START -->
- **E4-TAPER v61 覆盖：** `C-U1-E4-TAPER-NEAR-RETENTION-01` 已完成协议冻结、独立 runner、formal-channel 登记和工程 smoke，registry 为 **implemented + ready + active + not_run**。允许下一步启动该实验的 canonical guarded formal run，但 smoke/单元测试不构成科学结果。`BUDGET-MATCH-01` 仍必须等待 Near-Retention 的 raw-complete、终态审计、打包与交付；不得提前实现为可运行状态或并行启动。
<!-- HANDOFF-DELTA-BLOCK:section_end:v61-e4-taper-near-retention-current-gate:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000032:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000033:START -->
<!-- HANDOFF-DELTA-BLOCK:section_end:v62-countdown-online-offpolicy-current-gate:START -->
- **Countdown v62 覆盖：** `EXT-C-E8-V4.6-ONLINE-OFFPOLICY-REPLAY` 是当前用户批准并已实现的 Countdown focused successor，状态为 **implemented + not_run**。执行前必须提供完整 V4.5 `RUN_COMPLETE.json`/`terminal_audit.json` 及其指向的 V4.4 frozen inputs；runner fail-closed 校验输入与 reference adapter。它可作为独立 pilot 启动，但不改变 `EXT-H-E7-Q2` 的 formal 优先级，也不自动解锁 `EXT-C-E8-SCALE-01`。
<!-- HANDOFF-DELTA-BLOCK:section_end:v62-countdown-online-offpolicy-current-gate:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000033:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000034:START -->
<!-- HANDOFF-DELTA-BLOCK:section_end:v63-e4-taper-closure-current-gate:START -->
- **E4-TAPER v63 覆盖：** `C-U1-E4-TAPER-NEAR-RETENTION-01` 已完成 `280/280` method-seed runs 与终态审计，科学状态沉淀为 **有限训练步数验证**。主保持率 `0.75` 下，Reciprocal-Quadratic、current Exponential、Squared-distance Exponential 相对 Reciprocal-Linear 的 held-out-context reward 配对均值差分别为 `+0.012002 / +0.015619 / +0.036134`，三者均为 `20/20` seeds 正差；Squared-distance Exponential 的 harmful-far retention 为 `0.010382`，低于 Reciprocal-Linear 的 `0.055886`。该结果只支持当前冻结矩阵中的有限步函数形状信号；`260/280` runs 在 8000 steps 时未获严格终态解析，禁止稳态、普遍方法排名或 OOD 表述。
- 三类事件继续严格分报：task-performance collapse `13/280`、support/variance boundary `20/280`、NaN/Inf `0/280`；前两类全部来自 unweighted control。v63 仓库只保存 compact result deposition；本次构建会话没有原始 280-run raw-complete artifact 及其哈希，禁止伪造，归档发布前必须从原交付包恢复。
- `C-U1-E4-TAPER-BUDGET-MATCH-01` 在 v63 冻结并实现为下一项 **implemented + ready + active + not_run**。唯一 primary budget coordinate 是每一步、Adam 之前的 raw negative-gradient L2 norm；paired Reciprocal-Linear actor 生成冻结目标 schedule，其他 Distance families 与 non-selective Global stepwise scale 使用 detached scalar 精确匹配该 norm。Adam 实际 parameter-update norm 只记录、不声称匹配。正式 seeds 固定为 `110--129`；seeds `130--149` 继续 untouched，专属最终 confirmation。
- `C-U1-E4-TAPER-CONV-01` 与 `C-U1-E4-TAPER-CONFIRM-01` 的 seed firewall、输入输出契约、shortlist 冻结规则、32000-step 长程上限、continuous Adam-state 要求、2× terminal audit 与确认分析计划已预登记，但二者继续 blocked。Budget-Match terminal-audited、packaged、delivered 之前不得生成 shortlist 或启动 Convergence；Convergence 交付且 confirmation config 哈希冻结前不得访问 seeds `130--149`。
<!-- HANDOFF-DELTA-BLOCK:section_end:v63-e4-taper-closure-current-gate:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000034:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000035:START -->
<!-- HANDOFF-DELTA-BLOCK:section_end:v66-e4-taper-budget-match-current-gate:START -->
- **E4-TAPER v66 覆盖：** `C-U1-E4-TAPER-BUDGET-MATCH-01` 已完成 `140/140` 正式 runs、逐步 raw-negative-gradient budget audit 与 terminal audit，科学状态为 **有限训练步数验证**。相同 Adam 前 raw negative-gradient L2 budget 下，三种 selective candidates 相对 Reciprocal-Linear 均在 `20/20` paired seeds 上提高 held-out-context reward 并降低 harmful-far retention；Global stepwise scale 则在 `0/20` seeds 上提高 reward，且保留更多 harmful-far influence。Terminal useful-near retention 因零分母不可评估，不得补写为已证明。
- 原 guard 只在计算结束后的 required-output/package 阶段失败：return code `0`、provenance 未受损、正式结果和原 failed tree 均保留。v66 修复 runner 漏写 `scientific_run_manifest.json`，并通过 compact deposition + explicit full-raw sidecar 完成交付；不得把 packaging failure 称为实验数值失败。
- `CONV-01` 仍 blocked；下一项是 Budget-Match 交付后的独立 shortlist-freeze 更新和 continuation runner 实现。不得直接延长 run_003，也不得访问 confirmation seeds `130--149`。
<!-- HANDOFF-DELTA-BLOCK:section_end:v66-e4-taper-budget-match-current-gate:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000035:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000036:START -->

<!-- STAGE4B-SOURCE-BLOCK:B000036:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000037:START -->
## 0.2 C-U1 泛化术语覆盖规则（v15 锁定）

1. C-U1 的训练状态与测试状态使用同一生成分布 `s ~ N(0,I_6)`，只在样本身份上独立；因此 E1-E4 报告的是 **同分布 held-out-context generalization（未见状态泛化）**。
2. 允许使用的表述：`held-out-context reward`、`unseen-context generalization`、`同分布测试状态`、`未见状态上的函数泛化`。
3. 禁止用于当前 C-U1 的表述：`OOD reward`、`OOD generalization`、`distribution-shift generalization`。
4. `a_star(s)` 是训练中未作为正样本展示的隐藏最优动作。策略接近它可称为“越过正样本支持并接近隐藏最优动作”，不能仅凭此称为 OOD。
5. “策略漂移到低 reward 区域”与“数据分布 OOD”严格区分。前者是策略相对任务最优的几何漂移，不意味着测试状态来自分布外。
6. Part II 历史记录中的 OOD 旧措辞不删除，但全部由本节覆盖。新论文若需要 OOD 结论，必须另外登记并运行显式状态分布偏移实验。

<!-- STAGE4B-SOURCE-BLOCK:B000037:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000038:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000038:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000039:START -->
## 0.4 Registry 执行状态一致性（v42 锁定）

1. `execution_gate.state` 表示科学/依赖门禁，`formal_execution.activation_state` 表示 operational 启动状态；两者不得相互矛盾。
2. `execution_gate.state=ready` 且 `entrypoint_status=implemented` 时，`activation_state` 必须为 `active`。
3. `activation_state=active` 时，不得存在 `execution_gate.state=blocked`。
4. 任何 `blocked` 状态都必须登记非空依赖或 blocking reason；禁止无依据的陈旧 blocked 标记。
5. `entrypoint_status=planned`、`implementation_state=not_implemented` 的正式实验允许保持 blocked，但不得因此绕过 claim、职责和后续 protocol-freeze 登记。
6. `scripts/validate_formal_execution_channel.py` 对 canonical experiments 与 development registrations 中的 formal 条目执行 fail-closed 校验；registry 更新和 `drpo-update` 测试必须运行它。

<!-- STAGE4B-SOURCE-BLOCK:B000039:END -->
