# Global research core and governance boundaries

> Generated Stage 4 minimal-context shadow module. Do not edit manually.
> `docs/handoff.md` and `experiments/registry.yaml` remain authoritative.

- Module ID: `global_core_governance`
- Responsibility: Preserve the unique-master rule, terminology, scientific scope, and non-destructive governance constraints.
- Content contract topics: `unique_master_document`, `document_before_experiment`, `non_destructive_history`, `terminal_audit_governance`, `controlled_external_validity_boundary`
- Deduplicated overlapping source chunks: 0
- Source hash: `77ae9674fc2e5ef75d5d3b2827ba990988ea59afa8976ceff1fc8e9289fd2359`

## Content contract evidence

| Topic | Required semantic responsibility | Authoritative source | Matched phrase |
|---|---|---|---|
| unique_master_document | Keep docs/handoff.md as the unique research master and reject competing status sources. | docs/handoff.md: # 0. 研究与执行原则（每次新会话首先阅读） -> # 1. 论文最终目标与两条主工作线 | 唯一 Master 文档是任务轴 |
| document_before_experiment | Require claim, environment, data, metrics, convergence, and result placement before execution. | docs/handoff.md: # 0. 研究与执行原则（每次新会话首先阅读） -> # 1. 论文最终目标与两条主工作线 | 文档先于实验 |
| non_destructive_history | Preserve historical content and record replacement conclusions without destructive deletion. | docs/handoff.md: # 0. 研究与执行原则（每次新会话首先阅读） -> # 1. 论文最终目标与两条主工作线 | 不得破坏性删除 |
| terminal_audit_governance | Require terminal-state evidence for convergence, collapse, and method ranking claims. | docs/handoff.md: # 0. 研究与执行原则（每次新会话首先阅读） -> # 1. 论文最终目标与两条主工作线 | 动力学必须做终态审计 |
| controlled_external_validity_boundary | Keep Hopper and Countdown external validity distinct from C-U1 and D-U1 controlled identification. | docs/handoff.md: # 0. 研究与执行原则（每次新会话首先阅读） -> # 1. 论文最终目标与两条主工作线 | 外部实验不能替代理想识别 |

## Source 1: docs/handoff.md: # 0. 研究与执行原则（每次新会话首先阅读） -> # 1. 论文最终目标与两条主工作线

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
