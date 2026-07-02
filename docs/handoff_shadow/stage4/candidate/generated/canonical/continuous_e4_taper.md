# Continuous C-U1 E4 taper-family follow-up track

> Stage 4B lossless source-promotion shadow candidate.
> Generated from `docs/handoff.md`; do not edit while the manual handoff remains authoritative.

- Owner type: `canonical_module`
- Owner ID: `continuous_e4_taper`
- Responsibility: Cover taper-family mechanism comparisons, near-retention and budget fairness controls, long-run resolution, confirmatory evidence, and the frozen follow-up order.
- Dependencies: `continuous_e4_extrapolation`
- Content-contract topics: none
- Owned source blocks: 14
- Registry references are pointers only; `experiments/registry.yaml` remains the sole editable registry source.

## Registry references

- `experiments`: `C-U1-E4-TAPER-01`, `C-U1-E4-TAPER-NEAR-RETENTION-01`, `C-U1-E4-TAPER-BUDGET-MATCH-01`, `C-U1-E4-TAPER-CONV-01`, `C-U1-E4-TAPER-CONFIRM-01`
- `development_experiment_registrations`: none

## Owned source blocks

<!-- STAGE4B-SOURCE-BLOCK:B000012:START -->
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
<!-- STAGE4B-SOURCE-BLOCK:B000012:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000013:START -->
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
<!-- STAGE4B-SOURCE-BLOCK:B000013:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000017:START -->
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
<!-- STAGE4B-SOURCE-BLOCK:B000017:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000063:START -->
## 3.8 C-U1 共享实现与二次阶方法实验 `C-U1-E4-TAPER-01`

<!-- STAGE4B-SOURCE-BLOCK:B000063:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000064:START -->
### 3.8.1 代码单一来源

C-U1 的环境与 actor 不再允许嵌入新实验文件。唯一共享实现为 `src/drpo/cu1_core.py`，包含 state-to-geometry 映射、正/负轮廓、`Split/Environment`、Gaussian actor、log-probability、标准化距离和输出 score 分解。`drpo_cu1_e1_e4_oneclick.py` 只保留冻结 protocol、训练、干预、审计与报告；`cu1_e1_componentwise_rerun.py` 和 taper runner 只导入共享实现。重构必须用确定性张量、actor 初始化、log-probability、环境不变量和 smoke run 做等价回归，不能以“代码更整洁”为由改变任何冻结科学变量。

<!-- STAGE4B-SOURCE-BLOCK:B000064:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000065:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000065:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000066:START -->
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


<!-- STAGE4B-SOURCE-BLOCK:B000066:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000067:START -->
### 3.8.4 环境连续性、质量匹配与方向效用边界（v44 澄清）

1. **连续环境与有限离线支持必须区分。** C-U1 的动作空间是 `R^2`，reward 对任意动作连续可计算；负样本集合来自以 `a_star(s)` 为圆心、半径 1.20 的连续等值圆周。正式数据每状态只取 8 个均匀角度，是有限 offline dataset 的支持设计，不是分段或不连续 reward。
2. **等 reward/advantage 是人为控制变量。** 它不是行为策略自然采样后的经验巧合。这样设计是为了排除“far 样本梯度更大只是因为 reward 更低或 `|A|` 更大”的混杂，使 near/far 差异主要来自当前 policy score geometry 与方向。
3. **质量解耦不等于方向效用解耦。** 对负样本，均值分支更新方向与 `mu-a` 同向；其相对真实 improvement direction `a_star-mu` 的 cosine 决定局部 utility。当前圆周含 `a_minus=a_plus-0.50u`，排斥该近场点朝向 hidden optimum；圆周另一侧的远点排斥方向可与 hidden optimum 相反。因此相同 advantage 可以具有不同 directional utility。
4. **允许的机制表述。** 当前环境展示一种受控且现实相关的结构：局部负样本仍可能提供 boundary shaping，随着 policy-relative remoteness 增大，方向相关性可能下降或反转，而 Gaussian score influence 仍增长。Distance taper 处理的是这种 informativeness--amplification mismatch。
5. **禁止的普遍化。** 不得写成“near negative 必然有益”“far negative 必然有害”或“distance 在任何任务中都是 oracle utility”。真实任务中的 utility--distance 关系必须由多几何稳健性和 Hopper/Countdown/推荐外部验证测量。
6. **未来透明化材料。** 论文附录至少报告：负 advantage 对 distance 的水平匹配；未加权 score/influence 随 distance 的变化；负更新与 oracle improvement direction 的 cosine；各 taper 后的有效 `utility x influence`。这属于解释与审计，不改变 v43 的冻结结果。

<!-- STAGE4B-SOURCE-BLOCK:B000067:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000068:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000068:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000069:START -->
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
<!-- STAGE4B-SOURCE-BLOCK:B000069:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000070:START -->
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
<!-- STAGE4B-SOURCE-BLOCK:B000070:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000071:START -->
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
<!-- STAGE4B-SOURCE-BLOCK:B000071:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000072:START -->
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
<!-- STAGE4B-SOURCE-BLOCK:B000072:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000073:START -->

<!-- STAGE4B-SOURCE-BLOCK:B000073:END -->
