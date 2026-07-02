# Paper rewrite and presentation plan

> Stage 4B lossless source-promotion shadow candidate.
> Generated from `docs/handoff.md`; do not edit while the manual handoff remains authoritative.

- Owner type: `canonical_module`
- Owner ID: `paper_rewrite`
- Responsibility: Provide the current manuscript outline, claim hierarchy, figures, tables, limitations, and reproducibility plan.
- Dependencies: `global_core_governance`, `theory_methods_related_work`, `terminal_audit`, `continuous_mechanism_e1_e3`, `continuous_e4_taper`, `categorical_e5_mechanism`, `categorical_e6_generalization`, `hopper_e7`, `countdown_e8`
- Content-contract topics: none
- Owned source blocks: 149
- Registry references are pointers only; `experiments/registry.yaml` remains the sole editable registry source.

## Registry references

- `experiments`: none
- `development_experiment_registrations`: none

## Owned source blocks

<!-- STAGE4B-SOURCE-BLOCK:B000001:START -->
# DRPO / SNA2C 远场负梯度动力学研究主文档 v68（Hopper E7-Q2 长程机制结果闭环版）
<!-- STAGE4B-SOURCE-BLOCK:B000001:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000016:START -->
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
<!-- STAGE4B-SOURCE-BLOCK:B000016:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000019:START -->

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

<!-- STAGE4B-SOURCE-BLOCK:B000019:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000039:START -->
# 1. 论文最终目标与两条主工作线

<!-- STAGE4B-SOURCE-BLOCK:B000039:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000040:START -->
## 1.1 论文目标

以原 DRPO（arXiv:2602.10430）为起点，重写为面向一般 off-policy policy optimization 的论文，而非推荐专属论文。原推荐实验作为应用验证保留，不再承担理论合理性的唯一证据。

<!-- STAGE4B-SOURCE-BLOCK:B000040:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000041:START -->
## 1.2 理论修改主线

1. 保留原始 repulsive dynamics 主干：正优势吸引、负优势排斥、固定离线样本随策略移动进入远场、score function 放大、正负梯度失衡与 collapse。
2. 修正 Gaussian 方差方向：远场负样本导致均值排斥并收缩方差，而不是均值与方差同时扩张。
3. 修正数学工具：固定 off-policy 样本不能用 expected Fisher 的 SPD 性质证明联合扩张；改用精确更新式与总体 signed gradient field 的 Jacobian。
4. 将 exponential-family 统一作为核心 contribution：在不抛弃原变量体系的前提下，使用指数族必要符号给出 Gaussian 与 categorical 的共同平衡/边界条件。
5. 解释负梯度的双重作用：受控负梯度可突破 positive-only 的模仿上限，远场异常负梯度则导致失稳。

<!-- STAGE4B-SOURCE-BLOCK:B000041:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000042:START -->
## 1.3 新实验主线

1. 一个真正统一的连续 contextual-bandit 环境，完成四个连续实验块；
2. 一个独立 categorical 环境，完成两个离散实验块；
3. Hopper、Countdown、推荐作为外部验证层；
4. 所有涉及动力学终态、稳态或方法排名的实验必须达到预定义收敛标准。

---

<!-- STAGE4B-SOURCE-BLOCK:B000042:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000043:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000043:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000058:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000058:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000073:START -->
## 3.9 E6--E8 方法迁移与规模验证路线（v42 锁定）

1. **E6：** pilot 与 focused development 已完成 development seeds 0--4 的 105/105 与 165/165 runs；`D-U1-E6-SEMANTIC-LONGRUN-01` 已在 untouched seeds 10--29 上完成 360/360 formal runs并通过 2x 终态审计。结果支持 positive-only ceiling、受控 local negative 的同分布 held-out-context / unseen-action 收益、过强压力反转、任务与支持事件分离以及语义置乱排他性。
2. **`D-U1-E6-TAPER-01`（E6-TAPER）：** 在 E6 long-run 冻结的同一个 semantic remoteness coordinate 上比较 reciprocal-linear、reciprocal-quadratic 与 exponential，并包含 positive-only、uncontrolled 和 global-alpha controls。该实验验证控制思想跨策略族迁移，不声称 categorical policy 具有 Gaussian 的二次梯度临界界。
3. **`EXT-H-E7-Q2`（E7-MECH）：** Hopper learned-critic 深度机制 runner 已实现，但 formal launch 继续 blocked，直到 E6-TAPER 交付。该实验回答真实数据是否进入 Gaussian log-scale 二次主导区、是否传导到 full-parameter gradient/长期动力学；不承担大规模方法排名。
4. **`EXT-H-E7-BENCH-01`（E7-BENCH）：** 公共大规模连续控制主表固定为 D4RL MuJoCo locomotion suite：Hopper、Walker2d、HalfCheetah × medium、medium-replay、medium-expert，共 9 tasks。方法 shortlist 与超参从 E4/E6-TAPER 冻结，不得在 D4RL 上按任务重新选择方法族；主报 normalized return、多 seed 区间、跨任务平均排名、最差 seed 与三类失效事件。AntMaze/Kitchen/Adroit 不属于本主表，可另行登记 stress test。
5. **`EXT-C-E8-V4.2`（E8-MECH）：** 0.5B Countdown/Qwen 继续承担 Transformer 固定负优势 near/far probe、pipeline 与小规模方法信号，不承担最终规模结论。
6. **`EXT-C-E8-SCALE-01`（E8-SCALE）：** 在方法 shortlist 冻结后，使用更大固定 Countdown offline dataset；3B 为正式主模型，7B 只做冻结配置确认，不在规模实验重新筛选方法族。
7. **执行顺序：** `E4-TAPER -> E6 -> E6-TAPER -> E7-MECH -> E7-BENCH -> E8-MECH -> E8-SCALE`。E4-TAPER 已以 finite-step status 交付；E6 long-run 已以 long-run validated 状态交付。当前下一阶段是先审阅、冻结并实现 `D-U1-E6-TAPER-01`，不能直接运行其 planned registration；每个正式 ID 必须先完成 terminal audit、packaging 和 delivery，下一正式 ID 才可启动。

---

<!-- STAGE4B-SOURCE-BLOCK:B000073:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000076:START -->
<!-- HANDOFF-DELTA-BLOCK:section_end:v59-e8-offline-tuning-route:START -->
9. **v59 E8 内部路线覆盖：** V4.4 fixed-bank 之后先运行 V4.5 validation-only α×λ 调参，检验当前 dynamic 方法是否只是控制强度偏保守。只有调参仍不能产生稳定收益时，才进入另行登记的 online off-policy successor；不得用 test 反复挑选参数，也不得把 V4.5 变成无界 HPO。
<!-- HANDOFF-DELTA-BLOCK:section_end:v59-e8-offline-tuning-route:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000076:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000077:START -->
<!-- HANDOFF-DELTA-BLOCK:section_end:v62-e8-online-offpolicy-route:START -->
10. **v62 E8 内部路线覆盖：** V4.5 已完成其“alpha/lambda 是否未调到位”的职责后，不再扩大 frozen-bank HPO。V4.6 用全新 paired seeds 执行 frozen/online × positive/dynamic 2×2；只有 online negative 相对 online Positive-only 的 paired 增量与 refresh×negative interaction 才能支持“动态负样本有额外价值”。若 online 两个 cells 都提高但彼此持平，收益归因于数据刷新；若 online dynamic 仍不占优，不得继续用 bank staleness 解释。
<!-- HANDOFF-DELTA-BLOCK:section_end:v62-e8-online-offpolicy-route:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000077:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000078:START -->

<!-- STAGE4B-SOURCE-BLOCK:B000078:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000097:START -->
# 7. 变量治理

<!-- STAGE4B-SOURCE-BLOCK:B000097:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000098:START -->
## 7.1 原体系保留的核心变量

`s, a, pi_theta, theta, A, mu, sigma/Sigma, alpha, distance d, score function grad_theta log pi_theta`。

<!-- STAGE4B-SOURCE-BLOCK:B000098:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000099:START -->
## 7.2 Exponential-family 核心定理所需新增符号

仅使用该理论无法避免的 `eta, T(a), psi(eta)`；首次出现时完整定义，并明确映射回 Gaussian 的 `mu/sigma` 与 categorical logits。

<!-- STAGE4B-SOURCE-BLOCK:B000099:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000100:START -->
## 7.3 共同负对数概率

允许定义一次 `D_theta(s,a) = -log pi_theta(a|s)`；正文主要称“负对数概率”，括号注明 surprisal。它用于连续与离散的共同陈述，但不取代 Gaussian 距离变量。

<!-- STAGE4B-SOURCE-BLOCK:B000100:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000101:START -->
## 7.4 撤出主体系的符号

已撤出的绘图归一化符号、`p/n/q` 简写、重复的 signed target、`kappa(D)`、与 discount factor 冲突的 `gamma` 等不进入主理论。局部证明需要时必须当场定义且不跨章节复用。

---

<!-- STAGE4B-SOURCE-BLOCK:B000101:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000104:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000104:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000105:START -->
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


<!-- STAGE4B-SOURCE-BLOCK:B000105:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000106:START -->
# 11. v13 正式执行日志

<!-- STAGE4B-SOURCE-BLOCK:B000106:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000107:START -->
## 11.1 C-U1 单 seed 回归（2026-06-24）

**状态：pilot / 回归通过。** 统一环境所有几何不变量通过。Positive-only 在测试状态上收敛到 `a_plus`：`归一化外推位移=-0.0001`、`mu_to_plus=0.0023`、`sigma=0.1904`。短程 signed update 随 alpha 增大出现更强方差收缩和反向漂移，方向与预登记理论一致；该扫描不作为正式 E3 结果。

代码与结果：`/mnt/data/drpo_experiments/c_u1_unified.py`；`/mnt/data/drpo_experiments/runs/c_u1_regression_seed0/`。

<!-- STAGE4B-SOURCE-BLOCK:B000107:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000108:START -->
## 11.2 E1：统一环境中的瞬时梯度来源隔离（2026-06-24）

**状态：正式 20-seed 机制识别完成。** 使用 held-out seeds 10–29；每 seed 4096/4096 train/test states，并在 128 个 probe states 上计算逐样本全参数梯度。负 advantage 跨距离保持 `1.000000×`，每状态最大数值范围低于 `2.7e-7`。

| 阶段 | score far/near | 单样本全参数负梯度 far/near | 聚合负梯度 far/near |
|---|---:|---:|---:|
| 初始化 | 5.375 [5.290, 5.462] | 5.383 [5.296, 5.473] | 5.370 [5.220, 5.527] |
| Positive-only 收敛后 | 7.569 [7.562, 7.575] | 9.093 [9.012, 9.171] | 10.072 [9.959, 10.190] |

20/20 seeds 的远场单样本和聚合梯度比均大于 1。结论：在真正统一的 C-U1 中，badness 严格不变时，policy-relative remoteness 仍独立产生数量级更大的负梯度；主体来自 score geometry，聚合方向一致性使倍率进一步增强。该结果复现并迁移了历史 Product 环境的锁定结论，但不把 9×/10× 写成普适常数。

代码与结果：`/mnt/data/drpo_experiments/run_e1_formal.py`；`/mnt/data/drpo_experiments/runs/e1_formal/`。


<!-- STAGE4B-SOURCE-BLOCK:B000108:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000109:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000109:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000110:START -->
## 11.4 E3：C-U1 统一 Adam Near/Far 因果干预（2026-06-25，当前论文结果）

**状态：已长期验证。** Experiment ID 为 `C-U1-E3-ADAM-RERUN`；run commit 为 `ac286a46b8ffad898dfad0e7e9188b1d2e81052a`；正式 held-out seeds 为 30--49。环境使用 4096 train / 4096 test states、每状态 4 个正动作和 8 个等 advantage 负动作，E3 各方法从同 seed 的 2000-step positive-only Adam checkpoint 初始化。测试状态与训练状态同分布，只能称 held-out-context generalization。

<!-- STAGE4B-SOURCE-BLOCK:B000110:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000111:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000111:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000114:START -->
### 来源与聚合审计说明

Scientific run 使用 exact committed runner blob，runner SHA-256 为 `502c345289d2b5b7c34832246478b64c33a1789e80ddcab7f6194cb09b0eac6f`。启动环境因 shell DNS 无法访问 GitHub，没有本地 Git object；该来源限制保留在 artifact provenance 中。最终聚合遇到 tuple 经 JSON 序列化为 list 后的 resume 比较问题，仅使用表示归一化 workaround 完成汇总；没有改变 seed、配置、优化器、梯度、阈值、轨迹或数值结果。

<!-- STAGE4B-SOURCE-BLOCK:B000114:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000115:START -->
## 11.4 E3：C-U1 长期 Near/Far 因果干预（2026-06-24，历史 SGD 证据）

> **v29 历史覆盖说明（已由 v31 替代）：** 本节数值与 SGD 配置仅保留作历史 provenance。v29 当时将 `C-U1-E3-ADAM-RERUN` 记为“尚未运行”；当前状态以本节之前的 v31 Adam 结果为准。旧 SGD 结果不得覆盖或混入当前论文表。

**历史状态：当时登记为正式 20-seed 因果实验完成。** 为避免开发 seed 泄漏，正式使用 held-out seeds 30–49。固定方差与可学习方差分成两个互补分支，严格区分“任务效果崩溃但数值仍有限”和“任务阈值前先发生精度/支持坍缩”。

<!-- STAGE4B-SOURCE-BLOCK:B000115:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000116:START -->
### 固定方差任务崩溃分支

`sigma=0.190394, alpha=1.4, lr=1e-4, 2000 steps`。Baseline 最终 reward `0.1591 [0.1539,0.1645]`，18/20 达到任务崩溃阈值、其余 2 个仍持续低 reward 漂移；Near-zero 最终 `0.1743 [0.1690,0.1797]`，13/20 达阈值、其余 7 个仍持续低 reward 漂移。两者均无 NaN/Inf。Far-zero、Far-cap、Global-scale 为 0/20 任务崩溃，最终 reward 分别为 `0.6934`、`0.6469`、`0.6469`。Far-zero/Far-cap/Global 分别在 20/20 配对 seeds 中胜过 Baseline。

Far-to-near 将被截断的远场预算转移给方向可靠的近场负梯度，最终 reward `0.8095 [0.8073,0.8115]`，20/20 胜过 Baseline。这一结果说明巨大负更新并非单独充分致害；方向可靠的负梯度可以有益，危险来自远场异常影响与低/错误方向效用的结合。

<!-- STAGE4B-SOURCE-BLOCK:B000116:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000117:START -->
### 可学习方差精度坍缩分支

`alpha=0.15, lr=5e-4`。Baseline 与 Near-zero 均在 20/20 seeds 中先触发 log-sigma 数值边界，中位 onset 分别为 140 与 144 steps；当时 reward 尚未跌破任务崩溃阈值，因此正式标记为 `numerical_collapse_before_task_threshold`，不能写成任务与数值同时崩溃。Far-zero、Far-cap、Global-scale 均为 20/20 stable/bounded、0/20 数值崩溃。

**因果结论：** 删除近场不能切断固定方差的长期任务漂移，也不能阻止可学习方差的早期支持坍缩；删除或截断远场则同时救援两条分支。全局缩放同样稳定，进一步支持异常负梯度幅度是直接中介；Distance 与 global 的泛化优劣交由 E4。

代码与结果：`/mnt/data/drpo_experiments/runs/E3_RESULTS.md`；`/mnt/data/drpo_experiments/runs/e3_formal_fixed/`；`/mnt/data/drpo_experiments/runs/e3_formal_learn/`。


<!-- STAGE4B-SOURCE-BLOCK:B000117:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000118:START -->
## 11.4 E3 历史 SGD 协议冻结补充：均值任务崩溃与可学习方差提前失稳

开发 seeds 0–4 显示，在统一 C-U1 的可学习方差 actor 中，远场负样本的二次 log-variance score 会先于明显 reward 失效把 `log_sigma` 推过数值边界。这与第 12.6 节“方差边界早于均值临界点”的理论一致，但单独不能完成“任务效果崩溃”的识别。因此 E3 正式报告分为两个互补分支，环境、数据与 Near/Far 划分完全相同：

1. **Fixed-variance causal branch（主任务崩溃识别）：** 将 sigma 固定在 E2 的解析正样本稳态 `0.190394`，冻结 `alpha=1.4`、SGD learning rate `1e-4`、2000 steps；1000-step 为首次分类点，继续至 2×=2000 steps 检查结论不反转。开发 seeds 0–4 中 Baseline/Near-zero 产生数值有限的任务崩溃，而 Far-zero/Far-cap/Global/Far-to-near 保持稳定或有界。该分支隔离远场对均值与 task reward 的因果传导。
2. **Learnable-variance branch（提前精度/支持失稳）：** 冻结 `alpha=0.15`、SGD learning rate `5e-4`、2000-step 上限；稳定方法以 1000→2000 的 2× 延长检查，失稳方法记录精确数值边界步。开发 seeds 0–4 中 Baseline/Near-zero 均先触发方差数值边界，Far-zero/Far-cap/Global 稳定。该分支验证远场负梯度对可学习方差的更早失稳路径。

正式 E3 使用严格未查看 seeds 30–49；seed 10 仅保留为实现 smoke，不进入正式统计。Fixed 与 learnable 两分支不是两个环境，也不混淆历史 Product/Collapse 实验；它们只改变 Gaussian variance 是否作为可训练参数。



<!-- STAGE4B-SOURCE-BLOCK:B000118:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000119:START -->
## 11.7 C-U1 论文成熟度与剩余漏洞审计（2026-06-24）

<!-- STAGE4B-SOURCE-BLOCK:B000119:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000120:START -->
### 当前判断

- E1：机制识别成熟度高；主文可用，但需在重跑包中补 full-parameter gradient 的 Jacobian/direction 分解或 same-ray control，避免把全参数倍率全部归因于距离。
- E2：科学成熟度高；可入正文或附录。它证明 positive-only 平台、有限方差稳态和 phantom-gradient 前兆；仍需恢复逐步分解曲线，区分均值远离与 sigma 收缩各自贡献。
- E3 fixed variance：**已长期验证，主文可用。** 20-seed 统一 Adam 因果链完整：Baseline/Near-zero 20/20 任务崩溃，Far-zero/Far-cap 0/20；动态轨迹、阈值判据与终态审计已进入交付 artifact。
- E3 learnable variance：**已长期验证，适合作为主文互补 panel 或附录。** Baseline/Near-zero 20/20 首先发生支持收缩，远场控制 0/20；完整轨迹、全状态边界审计和无 clamp/无 NaN/Inf 结果已保留。其职责是支持收缩机制，不替代 fixed-variance 的任务崩溃识别。
- E4 fixed variance：解析—实验闭环成熟度高；主文核心结果候选。图表需拆开有限固定点和 runaway。
- E4 learnable variance：机制证据中高；可入附录或主文补充。性能曲线与稳定边界必须分开，不允许用未稳态截面 reward 排名。
- 方向诊断：仅作 sanity check/附录，不构成跨环境主结论。

<!-- STAGE4B-SOURCE-BLOCK:B000120:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000121:START -->
### 需要优先补强的漏洞

1. 当前 train/test state 来自同一分布，只能称 held-out context generalization，不应直接称 OOD；若论文要写 OOD，需另加分布偏移测试。
2. E1 的等 reward 轮廓改变动作方向；输出层 Gaussian score norm 只依赖距离，但全参数梯度还受网络 Jacobian 方向影响。需补 same-ray radial probe 或 Jacobian gain 分解。
3. 可学习方差分支必须对 log-standard-deviation 下界、参数化方式和精度做敏感性检查，排除人为 floor 决定 onset。
4. 动态 standardized-distance near/far 划分会受 sigma 收缩影响；必须同时报告 raw distance、standardized distance 和 near/far 样本占比随时间变化。
5. Far-cap 相对等预算 global 的优势依赖本环境的方向构造；只能作为 C-U1 方法识别，外部一般性由 Hopper/其他任务验证。
6. 统一源码与 raw trajectories 未进入当前持久化包，是目前最大的投稿级工程缺口。
<!-- STAGE4B-SOURCE-BLOCK:B000121:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000300:START -->
# Part VI. 论文重写大纲（当前草案，后续须与实验状态同步）

<!-- STAGE4B-SOURCE-BLOCK:B000300:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000301:START -->
# 论文重写大纲 v1：从推荐专属 DRPO 到通用 Repulsive Policy Dynamics

> 目标：把原稿从“生成式推荐 + Optimistic DRO + hard filtering”重构为一篇面向通用 off-policy policy optimization 的机制—理论—方法论文。本文档细化到段落，并在每一节注明“改什么、为什么改、对应哪类审稿意见”。

<!-- STAGE4B-SOURCE-BLOCK:B000301:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000302:START -->
## 0. 一句话定位与非谈不可的重写原则

<!-- STAGE4B-SOURCE-BLOCK:B000302:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000303:START -->
### 建议主标题

**Repulsive Policy Dynamics: Stable Extrapolation and Far-Field Collapse in Off-Policy Policy Optimization**

备选标题：

1. **When Negative Advantages Generalize—and When They Collapse Off-Policy Learning**
2. **Breaking the Curse of Repulsion: Signed-Moment Stability in Off-Policy Policy Optimization**
3. **Repulsive Surprisal Dynamics in Continuous and Discrete Policies**

<!-- STAGE4B-SOURCE-BLOCK:B000303:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000304:START -->
### 一句话主张

负优势并非天然有害：适度、局部且方向可靠的负更新可以突破 positive-only 的模仿上限；但固定或陈旧数据被重复复用后，负更新会持续提高样本 surprisal，并在 policy-relative far field 中形成失稳。该机制在 Gaussian 策略中表现为 score 幅度与方差耦合的 runaway，在 categorical 策略中表现为 logit gap 发散和支持集坍缩。

<!-- STAGE4B-SOURCE-BLOCK:B000304:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000305:START -->
### 必须完成的根本重写

- **从“推荐问题”改为“通用 off-policy signed policy optimization 问题”。** 推荐系统降为应用验证，不再承担理论动机的全部重量。
- **从“负优势必然爆炸”改为“共同排斥主干 + 策略族特有失稳分叉”。** 避免审稿人用 softmax score 有界或 Gaussian Hessian 反例推翻全文。
- **从“hard filtering 是必要且唯一方案”改为“负更新存在稳定—泛化折中，方法应控制 policy-relative remoteness”。** Hard filtering 仅保留为零负权重极限、旧方法或强基线。
- **理论核心从 expected-Fisher SPD 改为 surprisal increment + signed-moment feasibility + 动力场 Jacobian。** 这是技术上最重要的纠错。
- **实验核心从单一自建 RecSim 改为：受控因果环境 + D4RL 公共数据 + token-level Countdown + 可选推荐应用。**
- **彻底清理引用。** 所有参考文献必须能在 DOI、OpenReview、会议官网或 arXiv 中逐条核验；任何占位符和不确定引用不得进入主稿。

---

<!-- STAGE4B-SOURCE-BLOCK:B000305:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000306:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000306:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000307:START -->
## 2. 摘要：建议按 7 句话写成一个紧凑段落

<!-- STAGE4B-SOURCE-BLOCK:B000307:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000308:START -->
### 句 1：通用问题

说明 off-policy actor updates 会反复使用固定或陈旧的负优势样本；这些样本既包含边界信息，也可能 destabilize training。

**修改原因：** 不再从推荐长尾数据切入，以免理论和题目脱节。

<!-- STAGE4B-SOURCE-BLOCK:B000308:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000309:START -->
### 句 2：核心悖论

Positive-only 更新稳定但可能停在行为支持内；保留负更新可促进 mode suppression 和外推，却可能在 far field 造成崩溃。

**修改原因：** 原稿只强调“负样本有毒”，无法解释我们后续观察到的稳定泛化收益。

<!-- STAGE4B-SOURCE-BLOCK:B000309:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000310:START -->
### 句 3：统一理论

提出 Repulsive Signed-Moment Dynamics：单负样本更新提高其 surprisal；聚合正负信号把策略推向 signed moment target；目标位于可行 moment 域内部时存在稳定外推，越界时内部稳态消失。

<!-- STAGE4B-SOURCE-BLOCK:B000310:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000311:START -->
### 句 4：策略族分叉

Gaussian 中越界表现为均值 runaway、方差收缩与无界 score amplification；categorical 中 direct-logit score 虽有界，概率仍可指数衰减并逼近 simplex 边界。

<!-- STAGE4B-SOURCE-BLOCK:B000311:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000312:START -->
### 句 5：因果证据

概括受控实验：advantage 与 distance 严格解耦仍出现 far-field amplification；只删除远场而非近场负更新可阻止 OOD drift 和 collapse；适度负更新出现倒 U 型泛化收益。

<!-- STAGE4B-SOURCE-BLOCK:B000312:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000313:START -->
### 句 6：方法

提出 policy-relative remoteness control：以 surprisal 为统一变量，对负更新做指数 taper；可选增加仅在稳定裕度不足时触发的 batch safety budget。

<!-- STAGE4B-SOURCE-BLOCK:B000313:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000314:START -->
### 句 7：外部验证

说明在 D4RL continuous-control 数据与 Countdown token policy 上验证机制和方法，并公开代码、数据处理及逐 seed 结果。

**注意：** 摘要中不再出现“hard filtering is mathematically necessary”“exactly solves all noise”或“SOTA generative recommendation”等无法由新证据直接支持的表述。

---

<!-- STAGE4B-SOURCE-BLOCK:B000314:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000315:START -->
## 3. Introduction：建议 8 个段落

<!-- STAGE4B-SOURCE-BLOCK:B000315:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000316:START -->
### P1：从 off-policy actor learning 的普遍结构切入

写任何使用日志、replay、stale rollouts 或固定偏好数据的 actor update，都可能出现：当前策略已经变化，但旧数据仍被重复赋予正负 advantage。列举 offline RL、off-policy generative control、RLHF/RLVR、推荐等场景，但不在此处展开相关工作。

**目的：** 建立 general paper 的对象。

<!-- STAGE4B-SOURCE-BLOCK:B000316:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000317:START -->
### P2：解释负优势的双重价值

正优势把策略拉向已观察到的成功行为；负优势提供坏 mode 抑制和边界塑形。完全删除负优势接近 advantage-filtered imitation，因此稳定，却可能存在 support / imitation ceiling。

**目的：** 主动回应“为什么不直接过滤坏样本”；避免方法被看成简单 top-k 数据清洗。

<!-- STAGE4B-SOURCE-BLOCK:B000317:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000318:START -->
### P3：提出真正的未解问题

同一个负更新为何会从有益的局部信号变为破坏性远场排斥？现有解释常混合三件事：样本有多差、当前策略认为它多罕见、以及负样本数量/长度多大。仅观察大梯度无法识别因果来源。

**目的：** 对应旧 reviewer 对 Figure 2 和 advantage sign 过度解释的批评。

<!-- STAGE4B-SOURCE-BLOCK:B000318:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000319:START -->
### P4：机制概览

定义 policy-relative remoteness 为 surprisal 或相应几何距离。给出主循环：


a fixed negative sample → surprisal increases → support becomes more remote → future repulsive influence changes → drift or support collapse.

强调共同主干是 repeated repulsion，不是“所有策略的梯度范数都无界”。

<!-- STAGE4B-SOURCE-BLOCK:B000319:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000320:START -->
### P5：理论贡献概览

先介绍单样本 surprisal increment identity，再介绍指数族 signed-moment target。用一句话解释稳定外推与崩溃是“目标在可行 moment 域内/外”的同一几何相变。

<!-- STAGE4B-SOURCE-BLOCK:B000320:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000321:START -->
### P6：证据链概览

列出三个 protocol，但不在 Introduction 堆数值：

1. source isolation：质量与距离严格解耦；
2. causal collapse：near/far 定点干预；
3. stable extrapolation：positive-only ceiling → 最优负推力 → 过度外推 → collapse。

再说明 categorical 使用无序 action IDs + semantic embeddings，避免人为有序动作的质疑。

<!-- STAGE4B-SOURCE-BLOCK:B000321:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000322:START -->
### P7：方法概览

说明方法不是依据 raw reward 或欧氏距离静态过滤，而是对当前策略的 surprisal 进行负更新 taper；必要时再用轻量 batch stability budget。强调只使用已有 forward-pass 量，复杂度近似线性，不计算 Hessian。

<!-- STAGE4B-SOURCE-BLOCK:B000322:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000323:START -->
### P8：贡献列表

建议仅列四项：

1. **Unified theory：** surprisal 与 signed-moment feasibility；
2. **Causal identification：** badness–remoteness 解耦及 near/far intervention；
3. **Stability–generalization law：** 负更新的倒 U 型与联合均值—方差边界；
4. **Practical control and external validation：** remoteness taper + D4RL / token experiments。

**删除：** “首次发现负梯度有害”“hard filtering 唯一最优”“推荐 SOTA”之类贡献。

---

<!-- STAGE4B-SOURCE-BLOCK:B000323:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000324:START -->
## 4. Related Work：建议 4 个小节，每节 2–3 段

<!-- STAGE4B-SOURCE-BLOCK:B000324:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000325:START -->
### 4.1 Offline policy optimization and distribution shift

P1：AWR、CRR、IQL、BPPO、PPO-style off-policy variants 等如何约束 actor update。

P2：强调本工作研究的是 signed actor field 的动力学，不替代 critic conservatism 或 OOD value estimation。

<!-- STAGE4B-SOURCE-BLOCK:B000325:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000326:START -->
### 4.2 Negative-advantage and low-probability update dynamics

P1：讨论 positive-only、negative filtering、BAPO、低概率 token、staleness / off-support suffix 等工作。

P2：明确已有工作已经发现负优势主导、低概率 token 风险或 entropy collapse；我们的差异是严格解耦、跨时间递推、连续—离散统一和因果干预。

<!-- STAGE4B-SOURCE-BLOCK:B000326:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000327:START -->
### 4.3 Negative data for generalization and mode suppression

讨论 negative reinforcement、failure trajectory learning、OGPO、TOPR 等表明负信号可能改善多样性、pass@k、坏 mode 抑制和支持外推的工作。

**目的：** 让“负优势有益”成为有文献支撑的出发点，而不是只为了我们自己的结果临时改变叙事。

<!-- STAGE4B-SOURCE-BLOCK:B000327:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000328:START -->
### 4.4 Entropy and support control

P1：entropy bonus、target entropy、KL/reference、temperature control。

P2：解释总体 entropy 不是 task-relevant support 的充分统计量，因此正文必须包含 entropy-matched baseline。

<!-- STAGE4B-SOURCE-BLOCK:B000328:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000329:START -->
### 引用治理规则

- 每篇文献必须记录：官方标题、作者、年份、会议/arXiv ID、URL/DOI、与本文关系。
- 禁止引用无法核验的内部简称。
- 投稿前运行 BibTeX key 与正文引用自动一致性检查。
- 原稿中的 placeholder / hallucinated references 全部删除，不做“修补式保留”。

---

<!-- STAGE4B-SOURCE-BLOCK:B000329:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000330:START -->
## 5. Problem Setup and Scope：建议 5 个段落

<!-- STAGE4B-SOURCE-BLOCK:B000330:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000331:START -->
### P1：分析对象

定义静态数据分布 \(\mathcal D\) 上的 actor objective：

\[
J(\theta)=\mathbb E_{(s,a)\sim\mathcal D}[\widehat A(s,a)\log\pi_\theta(a\mid s)].
\]

明确它是许多 off-policy actor regression / approximate policy improvement 步骤的抽象。

<!-- STAGE4B-SOURCE-BLOCK:B000331:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000332:START -->
### P2：不要把它伪装成无偏 policy-gradient theorem

明确：当数据不是当前策略采样且没有 importance correction 时，上式一般不等于真实 on-policy return gradient。本文研究的是该实际使用更新的稳定性与表示几何，而不是声称其无偏。

**直接回应 reviewer VKfL 的 Eq. 2 批评。**

<!-- STAGE4B-SOURCE-BLOCK:B000332:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000333:START -->
### P3：advantage 条件

Actor step 中 \(\widehat A\) stop-gradient；它可来自 trajectory return、Q−V、group-relative reward 或固定标签。理论首先条件于给定 advantage，critic 联合训练作为移动目标在后文讨论。

<!-- STAGE4B-SOURCE-BLOCK:B000333:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000334:START -->
### P4：正负质量与条件分布

定义 \(p,q,P_+,P_-\)，以及 global α、sample weighting 如何吸收到有效负质量中。

<!-- STAGE4B-SOURCE-BLOCK:B000334:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000335:START -->
### P5：claim 层级

明确三层：任意可微策略的单样本结论；指数族输出分布的全局/局部几何结论；深网络参数空间的局部 pullback。读者从此处就知道哪些结论 general，哪些不是。

---

<!-- STAGE4B-SOURCE-BLOCK:B000335:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000336:START -->
## 6. Theory：正文建议 4–5 页，完整证明放附录

<!-- STAGE4B-SOURCE-BLOCK:B000336:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000337:START -->
### 6.1 Theorem 1：Single-sample repulsive surprisal dynamics

**段落 1：** 定义 \(S_\theta=-\log\pi_\theta\) 与负更新。

**段落 2：** 给连续时间精确恒等式：

\[
\frac{dS_\theta(z)}{dt}=|A(z)|\|\nabla_\theta\log\pi_\theta(z)\|^2.
\]

**段落 3：** 给离散步 Taylor 余项和步长充分条件。

**段落 4：** 解释该定理只保证被更新样本自身的 surprisal 增加；batch 中还存在交叉项，不能把单样本结论无条件扩展到每个样本。

<!-- STAGE4B-SOURCE-BLOCK:B000337:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000338:START -->
### 6.2 Batch interference and directional coherence

**段落 1：** 推导 \(\Delta S_i\) 中 self-term 与 Gram cross-term。

**段落 2：** 定义 repulsive influence 的四因子：negative mass、score scale、coherence、reuse。

**段落 3：** 给出实验对应：单样本 far/near ratio 与聚合 ratio 的差异来自 coherence，而非额外 advantage。

<!-- STAGE4B-SOURCE-BLOCK:B000338:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000339:START -->
### 6.3 Theorem 2：Signed-moment equilibrium in exponential families

**段落 1：** 写正则最小指数族和 signed objective。

**段落 2：** 定义 signed target \(\tau\)；推导梯度与 Hessian。

**段落 3：** 主定理三种情况：内部唯一稳态、边界解、域外无内部解。

**段落 4：** 解释该定理统一了稳定外推与 collapse，而不是把二者写成两个不相关故事。

**段落 5：** 给离散 Euler 局部步长条件，避免“连续时间稳定 = 任意学习率都稳定”的误解。

<!-- STAGE4B-SOURCE-BLOCK:B000339:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000340:START -->
### 6.4 Gaussian branch

<!-- STAGE4B-SOURCE-BLOCK:B000340:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000341:START -->
#### P1：fixed variance mean equilibrium

推导 \(\mu^*\)、\(q_{opt}\)、\(q_{crit}=p\)，建立 imitation ceiling → bounded extrapolation → persistent drift → runaway。

<!-- STAGE4B-SOURCE-BLOCK:B000341:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000342:START -->
#### P2：learnable variance joint equilibrium

推导 \(\sigma^{2*}\) 和 signed variance feasibility。强调方差临界可早于均值临界。

<!-- STAGE4B-SOURCE-BLOCK:B000342:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000343:START -->
#### P3：variance four quadrants

明确 A sign 与 standardized distance 共同决定 \(\sigma\) 的方向；删除原稿“both μ and σ expand”。

<!-- STAGE4B-SOURCE-BLOCK:B000343:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000344:START -->
#### P4：fixed-sample Hessian correction

正文简要指出 pointwise Hessian 不定，expected Fisher SPD 不能证明固定样本联合扩张；详细矩阵放附录。

<!-- STAGE4B-SOURCE-BLOCK:B000344:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000345:START -->
#### P5：far-field amplitude law

固定 \(\sigma\) 时 mean score 随距离线性、log-std score 随标准化平方距离增长；重复负更新可使距离对时间几何增长。不要写“梯度对原始距离指数增长”。

<!-- STAGE4B-SOURCE-BLOCK:B000345:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000346:START -->
### 6.5 Categorical branch

<!-- STAGE4B-SOURCE-BLOCK:B000346:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000347:START -->
#### P1：direct-logit score boundedness

证明 \(\|e_j-\pi\|\le\sqrt2\)。主动给出这一“反直觉限制”，避免审稿人指出后被动修改。

<!-- STAGE4B-SOURCE-BLOCK:B000347:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000348:START -->
#### P2：support boundary dynamics

证明当 \(\pi_j\) 很小时 surprisal 仍以非零速率增长，因此 logit gap 可发散、概率可指数衰减到 simplex 边界。

<!-- STAGE4B-SOURCE-BLOCK:B000348:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000349:START -->
#### P3：categorical signed target

把 full softmax 视为指数族，说明某分量为零/负分别对应边界/域外。

<!-- STAGE4B-SOURCE-BLOCK:B000349:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000350:START -->
#### P4：semantic feature policy

解释有益未见动作外推需要 action feature / shared representation，而不是 token ID 有序。无序 ID + semantic embedding 是正式实验设定。

<!-- STAGE4B-SOURCE-BLOCK:B000350:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000351:START -->
### 6.6 Neural-network pullback and scope

P1：定义输出自然参数 Jacobian \(J_\theta(s)\)。

P2：在 realizable fixed point 且残差为零附近，参数 Jacobian 是输出 covariance 的 pullback，给局部稳定性。

P3：明确非凸深网的全局收敛、Adam 动力学和多状态不可实现情形不在定理覆盖范围。

<!-- STAGE4B-SOURCE-BLOCK:B000351:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000352:START -->
### 6.7 Moving critics and stale advantages

P1：每个 detached actor step 可用固定 advantage 理论分析。

P2：critic 更新让 signed target 移动；可给 tracking error bound 或定性讨论。

P3：方法只能限制错误 advantage 的破坏幅度，不能保证 critic 符号正确。

---

<!-- STAGE4B-SOURCE-BLOCK:B000352:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000353:START -->
## 7. Method：先保留两个候选，外部实验后只选一个主方法

<!-- STAGE4B-SOURCE-BLOCK:B000353:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000354:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000354:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000355:START -->
### 7.2 Candidate B：Safety-only Stability-Budgeted Taper

先用 Candidate A 得到 sample weight，再计算 batch-level positive recovery 与 weighted negative budget；仅当稳定裕度不足时施加全局 \(\gamma_t<1\)。

**关键设计：** 不做完整 Hessian/Jacobian；只用 batch reductions。正常稳定区 \(\gamma_t=1\)，避免双重过度抑制。

<!-- STAGE4B-SOURCE-BLOCK:B000355:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000356:START -->
### 7.3 主方法选择规则

- 若 Exp 在 D4RL + Countdown 中稳定领先或与 SBRC 持平，正文只保留 Exp，SBRC 放附录。
- 若 SBRC 在跨任务上显著降低超参敏感性并提升最差 seed，则正文采用 SBRC-Exp，Exp 作为核心 ablation。
- 不允许根据单个 toy 环境选择复杂方法。

<!-- STAGE4B-SOURCE-BLOCK:B000356:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000357:START -->
### 7.4 Hard filtering / DRPO 的新位置

- 作为 \(w_i^-=0\) 的极端 conservative limit；
- 作为历史方法和 positive-only / top-k baseline；
- Optimistic DRO 的 closed form 仅在明确给定 uncertainty set 下成立，不再宣称现实任务的唯一最优方案；
- 原 DRPO 名称是否保留取决于最终主方法。若主方法不再是 DRO hard filtering，建议论文和方法改名，避免名实不符。

---

<!-- STAGE4B-SOURCE-BLOCK:B000357:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000358:START -->
## 8. Controlled Experiments：正文机制证据

<!-- STAGE4B-SOURCE-BLOCK:B000358:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000359:START -->
### 8.1 Protocol A：Source isolation

**问题：** 大梯度来自差样本还是 policy-relative remoteness？

**设计：** 质量 coordinate 与距离 coordinate 精确笛卡尔积；相同 advantage 复制到各距离。

**报告：** advantage ratio、score ratio、single-sample full-gradient ratio、aggregate ratio、coherence。

**文字边界：** 数十倍是该设置下效应量，不是普适常数。

<!-- STAGE4B-SOURCE-BLOCK:B000359:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000360:START -->
### 8.2 Protocol B：Causal collapse

**问题：** far-field gradient 是相关变量还是传导路径？

**设计：** baseline、near-zero、far-zero、far-cap、global equal-budget、far-to-near。

**报告：** reward、OOD drift、collapse rate、时间顺序、paired CI、Wilcoxon。

**必须写清：** 乘积流形实验不回答 collapse；该非线性环境才回答因果传导。

<!-- STAGE4B-SOURCE-BLOCK:B000360:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000361:START -->
### 8.3 Protocol C：Stable extrapolation and phase transition

**问题：** 为什么不直接 positive-only？

**设计：** 真实最优在最佳正样本支持之外；扫描负质量。

**报告：** positive-only ceiling、held-out beta、test reward、mean boundary、variance boundary、倒 U 曲线。

<!-- STAGE4B-SOURCE-BLOCK:B000361:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000362:START -->
### 8.4 Generic categorical with unordered IDs

**设计：** 随机 action IDs + semantic embeddings；reward 由语义结构决定，不使用人为数轴。

**对照：** 打乱 reward–embedding 对应关系。预期 support suppression 仍存在，但有益方向外推消失。

<!-- STAGE4B-SOURCE-BLOCK:B000362:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000363:START -->
### 8.5 Entropy-matched controls

比较 entropy bonus、target entropy、temperature floor 与 remoteness control。调节系数使最终 entropy 相近，再比较 task reward 和正确低概率支持保留率。

**目的：** 证明方法不是简单“增加随机性”。

---

<!-- STAGE4B-SOURCE-BLOCK:B000363:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000364:START -->
## 9. External Validation：至少两类公共任务

<!-- STAGE4B-SOURCE-BLOCK:B000364:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000365:START -->
### 9.1 D4RL / Hopper

**范围锁定：Hopper 不重复理想环境的全部实验，也不替代 C-U1。** Hopper 没有可直接观测的逐状态真实最优动作，因此不能复刻 E4 的 ground-truth 支持外推。它只重复可识别的子链：advantage-matched near/far 梯度来源、Positive-only 后固定负样本的 phantom 动力学、以及少量 signed/Near-zero/Far-zero/Global 干预。完整方法效果由标准 offline RL + environment rollout 单独回答。

<!-- STAGE4B-SOURCE-BLOCK:B000365:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000366:START -->
#### Mechanism subsection

- Hopper-medium：保守、较窄分布上的外部机制证据；
- Hopper-medium-replay：更宽 replay mixture，验证自然 near/far；
- Hopper-medium-expert：明显质量混合，用于方法效果和 stable extrapolation。

分析 protocol：用正 advantage 训练 actor；固定负样本只测 phantom gradient；按 \(|A|\) 分桶匹配 near/far；报告 standardized distance、Gaussian score、full-parameter gradient。

<!-- STAGE4B-SOURCE-BLOCK:B000366:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000367:START -->
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


<!-- STAGE4B-SOURCE-BLOCK:B000367:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000368:START -->
#### Method subsection

在 IQL 或既有 offline RL actor 上插入 Exp/SBRC 负优势控制；比较 normalized return、多 seeds、critic error sensitivity。不能只做 phantom analysis 就宣称方法提高 D4RL performance。

<!-- STAGE4B-SOURCE-BLOCK:B000368:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000369:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000369:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000370:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000370:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000371:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000371:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000372:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000372:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000373:START -->
#### 历史模型阶梯（v12，已由 v22 替换，不得执行）

- 0.5B：pipeline 和超参快速筛选；
- 3B Instruct：正式主 arena；
- 7B：冻结方法后的最终确认，而不是全部网格搜索。

<!-- STAGE4B-SOURCE-BLOCK:B000373:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000374:START -->
#### 历史正式流程（v12，已由 v22 替换，不得执行）

SFT 达到至少 15%–20% greedy verifier success → 冻结 checkpoint → 同一模型采样正/near-negative/far-negative 轨迹 → 各方法从同一 checkpoint 训练。

<!-- STAGE4B-SOURCE-BLOCK:B000374:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000375:START -->
#### 主指标

Verifier greedy success、pass@k、valid rate、token surprisal、正确低概率 token 保留、错误 token suppression、entropy、有效 support、\(\gamma_t\) 与平均权重。

<!-- STAGE4B-SOURCE-BLOCK:B000375:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000376:START -->
#### 历史 Baselines（v12，已由 v22 替换，不得执行）

Positive-only、uncontrolled、global α、Exp、entropy bonus、target entropy、SBRC/Hybrid，以及适用的现有 low-probability / surprisal-aware 方法。

<!-- STAGE4B-SOURCE-BLOCK:B000376:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000377:START -->
### 9.3 Recommendation application（可选但有价值）

若继续保留推荐实验，必须至少一个公共数据集 + 现代 backbone，例如 SASRec 或 generative retrieval backbone。旧 RecSim 可放附录作为工业形态 stress test，不能继续作为唯一主实验。

如果短期无法完成公共推荐实验，则主文不再声称“生成式推荐 SOTA”；推荐只作为 motivating application 和未来工作。

---

<!-- STAGE4B-SOURCE-BLOCK:B000377:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000378:START -->
## 10. Results section 的段落顺序

<!-- STAGE4B-SOURCE-BLOCK:B000378:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000379:START -->
### P1：先回答机制是否存在

Protocol A + Hopper phantom：distance-matched / advantage-matched far negatives 有更大 score 和全参数梯度。

<!-- STAGE4B-SOURCE-BLOCK:B000379:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000380:START -->
### P2：再回答是否因果导致 collapse

Protocol B near/far intervention，给最强 paired effect。

<!-- STAGE4B-SOURCE-BLOCK:B000380:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000381:START -->
### P3：回答负梯度是否有益

Protocol C 倒 U：positive-only ceiling、中等负推力最佳、过强后失稳。

<!-- STAGE4B-SOURCE-BLOCK:B000381:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000382:START -->
### P4：回答连续—离散是否统一

共同 surprisal / signed target，表型分叉：amplitude versus support。

<!-- STAGE4B-SOURCE-BLOCK:B000382:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000383:START -->
### P5：回答方法是否不仅仅维持 entropy

entropy-matched comparison。

<!-- STAGE4B-SOURCE-BLOCK:B000383:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000384:START -->
### P6：回答是否外部有效

D4RL normalized return + Countdown verifier success。

<!-- STAGE4B-SOURCE-BLOCK:B000384:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000385:START -->
### P7：超参和计算开销

报告 λ、S0、γ 的敏感性；训练时间、额外显存、是否需要第二次 backward。

---

<!-- STAGE4B-SOURCE-BLOCK:B000385:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000386:START -->
## 11. Discussion and Limitations：主文必须单设

<!-- STAGE4B-SOURCE-BLOCK:B000386:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000387:START -->
### 段落 1：机制边界

受控环境证明 far-field path 是充分且主导的传导路径，不代表真实任务的唯一 collapse 原因。

<!-- STAGE4B-SOURCE-BLOCK:B000387:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000388:START -->
### 段落 2：方向可靠性

理论控制 influence，但“信息价值随 distance 下降”仍主要由实验和任务结构支撑，尚非无条件定理。

<!-- STAGE4B-SOURCE-BLOCK:B000388:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000389:START -->
### 段落 3：critic 与 optimizer

固定 advantage 理论不保证 critic 正确；Adam、PPO clipping、importance ratio 等改变具体动力学。

<!-- STAGE4B-SOURCE-BLOCK:B000389:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000390:START -->
### 段落 4：神经网络全局性

指数族输出结论不等于非凸深网全局收敛；参数共享可产生跨样本干涉。

<!-- STAGE4B-SOURCE-BLOCK:B000390:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000391:START -->
### 段落 5：reward 与 support

support collapse 不必然导致 reward collapse；需要任务中有价值动作被压制的因果连接。

<!-- STAGE4B-SOURCE-BLOCK:B000391:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000392:START -->
### 段落 6：方法边界

Exp taper 可能过度削弱真正有用的 rare negative；SBRC 可能受 batch estimate 噪声影响。需要跨任务验证。

---

<!-- STAGE4B-SOURCE-BLOCK:B000392:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000393:START -->
## 12. Reproducibility and Ethics

<!-- STAGE4B-SOURCE-BLOCK:B000393:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000394:START -->
### Reproducibility

- 主文列明数据版本、模型版本、seed 划分、开发/最终 untouched seeds、训练步数、batch、学习率、硬件。
- 公布统一代码、配置、逐 seed CSV、bootstrap 脚本、图表源数据。
- 所有主图可由一个 `run_all.sh` 或单一入口重建。
- 受控环境与外部数据 loader 分开，但共享训练和诊断接口。

<!-- STAGE4B-SOURCE-BLOCK:B000394:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000395:START -->
### Reference integrity

- 投稿包中加入脚本检查 BibTeX 是否含 placeholder 字符串、空 venue、虚构 arXiv ID。
- 每个相关工作论断对应 primary source 页码或定理。
- 不再引用“看起来像论文”的未核验材料。

<!-- STAGE4B-SOURCE-BLOCK:B000395:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000396:START -->
### Ethics

- 删除原稿中未经证据支持的 Green AI 和工业 ROI 强断言。
- 说明负反馈控制可能错误压制少数但有价值的行为；方法需要 reward/critic quality 和公平性审计。

---

<!-- STAGE4B-SOURCE-BLOCK:B000396:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000397:START -->
## 13. 主图与主表蓝图

<!-- STAGE4B-SOURCE-BLOCK:B000397:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000398:START -->
### Figure 1：统一机制示意图

左：local negative shaping；中：signed target 仍在 moment domain 内，稳定外推；右：target 越界，Gaussian/categorical 分叉失稳。

<!-- STAGE4B-SOURCE-BLOCK:B000398:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000399:START -->
### Figure 2：Source isolation

advantage 相同、distance 变化、score / gradient amplification。

<!-- STAGE4B-SOURCE-BLOCK:B000399:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000400:START -->
### Figure 3：Causal intervention

near-zero 与 baseline 重合；far-zero/far-cap 救援；同时显示 OOD drift。

<!-- STAGE4B-SOURCE-BLOCK:B000400:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000401:START -->
### Figure 4：Stable extrapolation phase diagram

固定方差与可学习方差的 reward、mean、sigma 边界，叠加理论临界线。

<!-- STAGE4B-SOURCE-BLOCK:B000401:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000402:START -->
### Figure 5：Categorical support dynamics

direct-logit score 有界但 probability/logit gap 到边界；semantic shuffle 对照。

<!-- STAGE4B-SOURCE-BLOCK:B000402:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000403:START -->
### Figure 6：External validation

Hopper medium/replay/expert mechanism + method；Countdown verifier success 与 entropy-matched comparison。

<!-- STAGE4B-SOURCE-BLOCK:B000403:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000404:START -->
### Table 1：理论对象与分叉

Gaussian / categorical / semantic energy policy 的 mean-domain、临界条件和失稳表型。

<!-- STAGE4B-SOURCE-BLOCK:B000404:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000405:START -->
### Table 2：受控因果结果

20-seed paired effects。

<!-- STAGE4B-SOURCE-BLOCK:B000405:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000406:START -->
### Table 3：D4RL 方法结果

normalized return，含 IQL/AWR/positive-only/global/Exp/SBRC。

<!-- STAGE4B-SOURCE-BLOCK:B000406:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000407:START -->
### Table 4：Countdown arena

3B 主结果，7B 核心确认；greedy、pass@k、valid、entropy/support。

---

<!-- STAGE4B-SOURCE-BLOCK:B000407:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000408:START -->
## 14. 原稿内容的删除、降级和复用清单

<!-- STAGE4B-SOURCE-BLOCK:B000408:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000409:START -->
### 完全删除

- `Lastname et al.` 等虚构/占位引用；
- “negative advantages inevitably cause exponential explosion” 的无条件表述；
- expected Fisher SPD 推导固定样本所有方向扩张；
- “Both μ and σ expand”；
- “hard filtering is mathematically necessary for survival”；
- “simulation alone proves general offline RL superiority”；
- 未定义的 Soft-Base 术语。

<!-- STAGE4B-SOURCE-BLOCK:B000409:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000410:START -->
### 降级到附录或历史背景

- Optimistic DRO hard-filtering 完整推导；
- 旧 RecSim 大部分结果；
- C1/C2/V0 开发环境；
- offline-to-online 推荐 curriculum，除非重新做公共可复现验证。

<!-- STAGE4B-SOURCE-BLOCK:B000410:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000411:START -->
### 直接复用但需重写叙事

- Gaussian score 几何；
- phantom gradient 诊断，但必须报告真实 sigma 轨迹和梯度符号；
- 远场/近场干预；
- positive-only 对照；
- 原 DRPO 的“repulsive optimization”术语，可作为历史起点。

---

<!-- STAGE4B-SOURCE-BLOCK:B000411:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000412:START -->
## 15. 写正文前的决策门槛

1. **方法门槛：** Exp 与 Safety-only SBRC 至少在受控 continuous、semantic categorical、D4RL、Countdown 四类中完成筛选，只留一个主方法。
2. **Countdown 门槛：** SFT greedy verifier success ≥15%，主 arena 使用 3B；0.5B 不承担最终结论。
3. **D4RL 门槛：** medium-replay 至少完成多 seed mechanism；方法结果需要 environment rollout 和 normalized return。
4. **引用门槛：** 所有引用核验完毕，自动审计无 placeholder。
5. **claim 门槛：** 正文每个强 claim 对应定理、受控干预或公共 benchmark 中至少一种直接证据。
6. **reproducibility 门槛：** 代码与配置在新环境中从零运行一次。

---

<!-- STAGE4B-SOURCE-BLOCK:B000412:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000413:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000413:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000414:START -->
# 附录 A：v11 恢复记录（原第 0 节，完整保留）

<!-- STAGE4B-SOURCE-BLOCK:B000414:END -->
