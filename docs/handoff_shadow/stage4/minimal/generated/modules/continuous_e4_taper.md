# Continuous C-U1 E4 taper-family follow-up track

> Generated Stage 4 minimal-context shadow module. Do not edit manually.
> `docs/handoff.md` and `experiments/registry.yaml` remain authoritative.

- Module ID: `continuous_e4_taper`
- Responsibility: Cover taper-family mechanism comparisons, near-retention and budget fairness controls, long-run resolution, confirmatory evidence, and the frozen follow-up order.
- Content contract topics: none
- Deduplicated overlapping source chunks: 4
- Source hash: `3547ad0867e1954a83b448af85d904790a5b15ae06e733d2396013556f7de744`

## Source 1: docs/handoff.md: ## 3.8 C-U1 共享实现与二次阶方法实验 `C-U1-E4-TAPER-01` -> ## 3.9 E6--E8 方法迁移与规模验证路线（v42 锁定）

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

## Source 2: docs/handoff.md: HANDOFF-DELTA-BLOCK after_heading:v60-e4-taper-utility-registration

### Delta block `after_heading:v60-e4-taper-utility-registration`

> **v60 增量登记：E4-TAPER 负样本净效用理论与四项公平/终态实验路线（不删除 v59 及更早内容）**
>
> - `C-U1-E4-TAPER-01` 的 220/220 正式结果和 **finite-step validated** 状态保持不变；本版不重跑、不延长原 8000-step protocol，也不把有限步排序升级为长期或普遍方法排名。
> - 正式引入负样本 alignment utility、orthogonal nuisance cost 与 net utility。条件经验假设只要求：离开局部信息区后，负样本净效用随 policy-relative distance 总体不增，并可能趋零或转负；**不假设效用按指数速度下降**，也不把该关系声明为普遍定理。
> - 澄清 Quadratic 与 Exponential 的理论职责：Quadratic 权重本身趋零，但与 learnable-log-scale 的 `Theta(d^2)` 原始影响相乘后一般只得到 bounded nonzero influence；Exponential 或任何 `o(d^-2)` 尾部进一步保证 vanishing influence。Quadratic 是最低充分有界阶，Exponential 是平滑 vanishing-tail 候选而非唯一解。
> - 当前 E4 历史公式 `exp(-lambda*u)` 不变。`exp(-beta*u^2)` 仅登记为近场一阶导数为零、远场指数趋零的候选，必须在新实验 protocol freeze 中显式批准，不能追溯性替换旧结果。
> - 用户批准登记四项后续：`C-U1-E4-TAPER-NEAR-RETENTION-01`、`C-U1-E4-TAPER-BUDGET-MATCH-01`、`C-U1-E4-TAPER-CONV-01`、`C-U1-E4-TAPER-CONFIRM-01`。四项当前均为 **not_run + not_implemented + blocked**，不得因登记而直接启动。
> - E4-TAPER 内部顺序冻结为 near-retention matching -> negative-budget matching -> long-run terminal resolution -> untouched-seed confirmation。Long-run 继续推迟到前两项冻结方法公式和超参数之后；几何 robustness extension 保持低优先级、非当前门禁。
> - 本更新只修改理论、registry、Stage 3 delta 与治理测试；没有运行新的科学实验，也不预设 Linear、Quadratic、Exponential、Global alpha 或 squared-distance exponential 的最终赢家。

## Source 3: docs/handoff.md: HANDOFF-DELTA-BLOCK after_heading:v61-e4-taper-near-retention-implementation

### Delta block `after_heading:v61-e4-taper-near-retention-implementation`

> **v61 增量登记：`C-U1-E4-TAPER-NEAR-RETENTION-01` 协议冻结、独立 runner 与执行解锁（不删除 v60 及更早内容）**
>
> - `C-U1-E4-TAPER-01` 的 220/220 结果、有限训练步数验证状态、anchor-normalized 结论与所有公平性边界保持不变；本版不重跑、不延长旧实验。
> - 第一项后续 `C-U1-E4-TAPER-NEAR-RETENTION-01` 已冻结：near 区域为 frozen 2000-step positive-only checkpoint 上的标准化距离 `d<=5`；匹配目标为 development seeds 0--4 上 pooled `E[w(d)|near]`；正式 paired seeds 为 90--109。
> - 保持率层级冻结为主层级 `0.75` 与敏感性层级 `0.50/0.25`。每个 family 只通过确定性单调二分求一个系数，系数在正式 seeds 和全部训练步中固定；formal/confirmatory seeds 严禁参与校准。
> - 候选函数冻结为 reciprocal-linear、reciprocal-quadratic、历史 current exponential `exp(-c u)` 与新批准的 squared-distance exponential `exp(-c u^2)`。后者只属于本新实验，不能追溯替换旧 E4-TAPER exponential。
> - 新增独立 formal runner `src/drpo/cu1_taper_near_retention_formal.py`，复用共享 C-U1 环境/actor 与原 positive checkpoint；报告 near useful retention、far harmful influence、全参数 far/near 比、distance-bin utility、同分布 held-out-context reward、sigma/support 和三类失效事件。
> - 本实验不匹配总负梯度预算，科学状态上限为 finite-step validated；长期 shortlist 与稳态排名继续由后续 `CONV-01` 负责。当前仅完成实现与 smoke，正式多 seed 尚未启动。
> - `BUDGET-MATCH-01`、`CONV-01`、`CONFIRM-01` 继续 blocked；只有 Near-Retention 正式结果完成终态审计、打包并交付后，才允许冻结下一项。

## Source 4: docs/handoff.md: HANDOFF-DELTA-BLOCK after_heading:v66-e4-taper-budget-match-closure

### Delta block `after_heading:v66-e4-taper-budget-match-closure`

> **v66 增量登记：`C-U1-E4-TAPER-BUDGET-MATCH-01` 正式结果、收尾故障审计与闭环交付（不删除 v65 及更早内容）**
>
> - 正式运行绑定 clean `main` commit `1faea3a92f74af5d11409779d96b9ed21fe846ad`，使用冻结 paired seeds `110--129`、7 个条件、每个最多 8000 steps，完成 `140/140` method-seed runs。逐步 Adam 前 raw negative-gradient L2 budget 的最大相对误差为 `2.12e-16`，通过 `1e-6` 门槛；Adam parameter-update norm 仅记录、未匹配。
> - 以 Reciprocal-Linear 为 reference，Reciprocal-Quadratic、current Exponential、Squared-distance Exponential 的 held-out-context reward 配对均值差分别为 `+0.016011 / +0.088189 / +0.130616`，均为 `20/20` seeds 正差；harmful-far retention 配对差分别为 `-0.012528 / -0.053566 / -0.055866`，均为 `20/20` 更低。Non-selective Global stepwise scale 的 reward 差为 `-0.006883`（`0/20` 正差），harmful-far retention 差为 `+0.007659`（`0/20` 更低）。这支持“相同 raw negative-gradient 总预算下，选择性 taper 的远场分配而非仅总预算大小会改变有限步任务结果”。
> - Terminal near-useful retention 在非 Positive-only 方法上因 raw positive-projection denominator 为零而为 undefined/NaN，因此本实验不能独立声称 candidate 把更多预算保留给 useful-near；该部分仍由 Near-Retention predecessor 承担。当前 Budget-Match 的强证据是 harmful-far suppression 与 held-out-context reward 的 paired 一致性。
> - 三类事件严格分报：task-performance collapse `13/140`、support/variance boundary `20/140`、NaN/Inf `0/140`；前两类全部来自 unweighted boundary。所有 matched/controlled 方法三类事件均为 0。固定 8000-step horizon 不证明稳态，科学状态只能是 **有限训练步数验证**；禁止 steady-state winner、universal winner、OOD generalization、跨任务优越或“Adam update 已匹配”表述。
> - 计算本身 return code 为 0，coverage、budget 与 terminal audit 全部通过；hardened guard 在收尾阶段标记 failed，因为 runner 漏写已登记的 `scientific_run_manifest.json`，且默认 25 MiB 主包超限。该故障不改变数值输出或 provenance。原 failed guard tree 完整保留；闭环包加入 runner manifest 修复、compact repository deposition 与完整 raw sidecar，不重跑正式 seeds。
> - `C-U1-E4-TAPER-CONV-01` 继续 blocked。Budget-Match 交付后，下一动作必须是独立的 deterministic shortlist-freeze 更新，再实现 exact actor+Adam-state continuation runner；本版不提前生成 shortlist，不自动启动 Convergence。Seeds `130--149` 继续禁止访问。

## Source 5: docs/handoff.md: HANDOFF-DELTA-BLOCK after_heading:v69-e7-bench-parallel-pilot

### Delta block `after_heading:v69-e7-bench-parallel-pilot`

> **v69 增量登记：`EXT-H-E7-BENCH-01` 两数据集并行 Pilot 与正式并行拓扑（不删除 v68 及更早内容）**
>
> - 本版不新增顶层实验 ID；在既有 `EXT-H-E7-BENCH-01` 下登记一个 **pilot** 子阶段。Pilot 只检查数据加载、learned-critic/actor/rollout 链路、连续 taper 实现、运行成本、artifact 体积、断点恢复及初步 paired direction，不得填入正式 9-task 主表，不得据此更换方法族、按任务调参或升级正式科学状态。
> - Pilot development seeds 冻结为 `200, 201, 202, 203`。方法冻结为 `Positive-only`、`Signed`、`Global alpha=0.75`、`Reciprocal-Linear`、`Reciprocal-Quadratic`、`Exponential`。三种 taper 沿用 `C-U1-E4-TAPER-NEAR-RETENTION-01` development seeds `0--4` 的冻结系数：`0.4362580032734791`、`0.5520268617673281`、`0.374162511054291`；标准化距离 reference/near boundary 均为 `5.0`，禁止 D4RL 后验重调。
> - 两个上传数据单元必须按真实 provenance 区分：`hopper-medium-expert-v2` 是 legacy D4RL-v2 HDF5，使用 Hopper-v4 与 D4RL-v2 normalized return；上传的 `mujoco/hopper/medium-v0` metadata 明确属于 **Minari Hopper-v5**，不是 D4RL `hopper-medium-v2`，因此只作为 pilot/plumbing cell、只报告 raw return，不能计入正式 D4RL 9-task 主表。正式 Hopper-medium cell 仍需另行冻结精确 D4RL 版本。
> - Pilot 固定预算为：每数据集一个 canonical critic `20k` optimizer steps、每 `(dataset, seed)` Positive-only `20k` steps、其余每个 method branch `40k` steps；只有 NaN/Inf 可提前终止。固定 horizon 不等于收敛，仍需分开报告任务性能崩溃、support/variance boundary、NaN/Inf 与 persistent/slow drift。
> - 为使用 384 核 CPU，执行器冻结为三阶段并行：`2` 个 dataset critic workers 并行；`8` 个 `(dataset, seed)` Positive-only workers 并行；`40` 个 `(dataset, seed, method)` branch workers 并行。线程预算分别为 `64/32/8`，峰值 `320` threads，保留系统和 I/O 余量。seed 与 method 均禁止顶层串行；每个 branch 从对应的同一 Positive-only checkpoint 分叉，输出目录隔离，resume 粒度为 `dataset_seed_method`。
> - 正式 9-task E7-BENCH 同步登记为 staged resource-pool 并行，branch scheduling unit 为 `task_seed_method`，禁止 serial seed loop 与 serial method loop；但正式 exact D4RL versions、formal seeds、offline-RL base、optimizer 和 full budget 尚未冻结，故 formal activation 继续 blocked。Pilot ready 不等于 formal ready。
> - 新入口为 `src/drpo/e7_bench.py`、`scripts/run_e7_bench.py`，配置为 `configs/e7_bench_pilot.yaml`，协议说明为 `docs/e7_bench_pilot.md`。当前仅完成实现、静态/单元、真实数据 loader 与 canonical critic 短程 smoke；当前环境缺少 `gymnasium`，因此 actor/rollout 短程 smoke 未执行。该限制不等于 Pilot 已运行，更不支持任何方法优于 Positive-only。正式启动时 runner 会在长程 critic 之前预检 384 核线程预算、Gymnasium/MuJoCo 环境及数据—环境维度一致性。

## Source 6: docs/handoff.md: HANDOFF-DELTA-BLOCK section_end:v60-e4-taper-current-gate

### Delta block `section_end:v60-e4-taper-current-gate`

- **E4-TAPER v60 覆盖：** `C-U1-E4-TAPER-01` 仍为 finite-step validated。四个后续 ID 已获用户批准并登记，但全部保持 blocked：先冻结并实现 `NEAR-RETENTION-01`，交付后才允许冻结 `BUDGET-MATCH-01`；二者交付并冻结 shortlist 后才允许 `CONV-01`；最后才用 untouched seeds 执行 `CONFIRM-01`。原实验禁止自动延长，几何 robustness 不作为当前门禁。

## Source 7: docs/handoff.md: HANDOFF-DELTA-BLOCK section_end:v61-e4-taper-near-retention-current-gate

### Delta block `section_end:v61-e4-taper-near-retention-current-gate`

- **E4-TAPER v61 覆盖：** `C-U1-E4-TAPER-NEAR-RETENTION-01` 已完成协议冻结、独立 runner、formal-channel 登记和工程 smoke，registry 为 **implemented + ready + active + not_run**。允许下一步启动该实验的 canonical guarded formal run，但 smoke/单元测试不构成科学结果。`BUDGET-MATCH-01` 仍必须等待 Near-Retention 的 raw-complete、终态审计、打包与交付；不得提前实现为可运行状态或并行启动。

## Source 8: docs/handoff.md: HANDOFF-DELTA-BLOCK section_end:v63-e4-taper-closure-current-gate

### Delta block `section_end:v63-e4-taper-closure-current-gate`

- **E4-TAPER v63 覆盖：** `C-U1-E4-TAPER-NEAR-RETENTION-01` 已完成 `280/280` method-seed runs 与终态审计，科学状态沉淀为 **有限训练步数验证**。主保持率 `0.75` 下，Reciprocal-Quadratic、current Exponential、Squared-distance Exponential 相对 Reciprocal-Linear 的 held-out-context reward 配对均值差分别为 `+0.012002 / +0.015619 / +0.036134`，三者均为 `20/20` seeds 正差；Squared-distance Exponential 的 harmful-far retention 为 `0.010382`，低于 Reciprocal-Linear 的 `0.055886`。该结果只支持当前冻结矩阵中的有限步函数形状信号；`260/280` runs 在 8000 steps 时未获严格终态解析，禁止稳态、普遍方法排名或 OOD 表述。
- 三类事件继续严格分报：task-performance collapse `13/280`、support/variance boundary `20/280`、NaN/Inf `0/280`；前两类全部来自 unweighted control。v63 仓库只保存 compact result deposition；本次构建会话没有原始 280-run raw-complete artifact 及其哈希，禁止伪造，归档发布前必须从原交付包恢复。
- `C-U1-E4-TAPER-BUDGET-MATCH-01` 在 v63 冻结并实现为下一项 **implemented + ready + active + not_run**。唯一 primary budget coordinate 是每一步、Adam 之前的 raw negative-gradient L2 norm；paired Reciprocal-Linear actor 生成冻结目标 schedule，其他 Distance families 与 non-selective Global stepwise scale 使用 detached scalar 精确匹配该 norm。Adam 实际 parameter-update norm 只记录、不声称匹配。正式 seeds 固定为 `110--129`；seeds `130--149` 继续 untouched，专属最终 confirmation。
- `C-U1-E4-TAPER-CONV-01` 与 `C-U1-E4-TAPER-CONFIRM-01` 的 seed firewall、输入输出契约、shortlist 冻结规则、32000-step 长程上限、continuous Adam-state 要求、2× terminal audit 与确认分析计划已预登记，但二者继续 blocked。Budget-Match terminal-audited、packaged、delivered 之前不得生成 shortlist 或启动 Convergence；Convergence 交付且 confirmation config 哈希冻结前不得访问 seeds `130--149`。

## Source 9: docs/handoff.md: HANDOFF-DELTA-BLOCK section_end:v66-e4-taper-budget-match-current-gate

### Delta block `section_end:v66-e4-taper-budget-match-current-gate`

- **E4-TAPER v66 覆盖：** `C-U1-E4-TAPER-BUDGET-MATCH-01` 已完成 `140/140` 正式 runs、逐步 raw-negative-gradient budget audit 与 terminal audit，科学状态为 **有限训练步数验证**。相同 Adam 前 raw negative-gradient L2 budget 下，三种 selective candidates 相对 Reciprocal-Linear 均在 `20/20` paired seeds 上提高 held-out-context reward 并降低 harmful-far retention；Global stepwise scale 则在 `0/20` seeds 上提高 reward，且保留更多 harmful-far influence。Terminal useful-near retention 因零分母不可评估，不得补写为已证明。
- 原 guard 只在计算结束后的 required-output/package 阶段失败：return code `0`、provenance 未受损、正式结果和原 failed tree 均保留。v66 修复 runner 漏写 `scientific_run_manifest.json`，并通过 compact deposition + explicit full-raw sidecar 完成交付；不得把 packaging failure 称为实验数值失败。
- `CONV-01` 仍 blocked；下一项是 Budget-Match 交付后的独立 shortlist-freeze 更新和 continuation runner 实现。不得直接延长 run_003，也不得访问 confirmation seeds `130--149`。

## Source 10: experiments/registry.yaml: experiments[C-U1-E4-TAPER-01, C-U1-E4-TAPER-NEAR-RETENTION-01, C-U1-E4-TAPER-BUDGET-MATCH-01, C-U1-E4-TAPER-CONV-01, C-U1-E4-TAPER-CONFIRM-01]

collection: experiments
entries:
- id: C-U1-E4-TAPER-01
  execution_gate:
    state: ready
    depends_on_delivered_experiment: C-U1-E4-CONV-01
    closure_decision: user_confirmed_on_2026_06_26
    reason: C-U1-E4-CONV-01 is long-run validated for its scoped phase claim by explicit user evidence review. The original
      18/20 gate failure remains documented and no method ranking is assumed.
  environment: C-U1
  name: standardized_distance_taper_order_formal_comparison
  status: finite_step_validated
  parent_experiment: E4
  depends_on_delivered_experiments:
  - C-U1-E3-ADAM-RERUN
  - C-U1-E4-ADAM-RERUN
  registration_base_commit: ac286a46b8ffad898dfad0e7e9188b1d2e81052a
  claim: With the same detached standardized Gaussian distance, matched reference attenuation, fixed advantages, actor, initialization,
    and minibatch stream, reciprocal-quadratic tapering suppresses far-field negative gradients more strongly than reciprocal-linear
    tapering. Whether this stronger suppression improves held-out-context task reward is an empirical secondary claim and
    is not assumed.
  role: controlled_method_order_validation
  execution_class: formal
  formal_execution:
    channel_ref: hardened-v1
    activation_state: active
    entrypoint_status: implemented
    entrypoint: src/drpo/cu1_distance_taper_formal.py
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
  code_entrypoint: src/drpo/cu1_distance_taper_formal.py
  formal_launch_template: python3 scripts/run_experiment_guard_hardened.py --experiment-id C-U1-E4-TAPER-01 --repo-root .
    --output-root experiments/results/C-U1-E4-TAPER-01/run_001 --artifact-output artifacts/C-U1-E4-TAPER-01_RAW_COMPLETE.zip
    --heartbeat-seconds 60 --stale-seconds 900 --fail-on-stale --progress-glob per_seed_runs_partial.csv --required-output
    RUN_COMPLETE.json --required-output scientific_run_manifest.json --required-output terminal_audit.json --required-output
    per_seed_runs.csv --required-output aggregate.csv --required-output paired_primary_summary.json --source-file src/drpo/cu1_distance_taper_formal.py
    --source-file src/drpo/cu1_core.py --source-file src/drpo/drpo_cu1_e1_e4_oneclick.py --run-class formal --expected-commit
    "$(git rev-parse HEAD)" --require-origin-main-match -- python3 src/drpo/cu1_distance_taper_formal.py --output-dir experiments/results/C-U1-E4-TAPER-01/run_001
    --base-commit "$(git rev-parse HEAD)"
  does_not_replace:
  - C-U1-E1
  - C-U1-E2
  - C-U1-E3
  - C-U1-E4
  theory:
    status: analytically_proven
    distance: d_theta(s,a)=norm(a-mu_theta(s),2)/sigma_theta(s)
    pre_boundary_assumption: sigma_theta(s)>=sigma_min>0
    fixed_or_bounded_negative_advantage: true
    output_gradient_order: Theta(d^2)
    reciprocal_polynomial_weight: w_p(d)=1/(1+lambda*(d/d_ref)^p)
    weighted_output_gradient_order: Theta(d^(2-p))
    critical_polynomial_order: 2
    full_parameter_extension: sufficient_under_bounded_output_jacobian
    full_parameter_necessity_requires: nondegenerate_log_scale_pullback
    utility_extension:
      status: conditional_empirical_hypothesis
      negative_update: g_minus=A_minus*grad_theta_log_pi_with_A_minus_negative
      oracle_improvement_direction: g_star
      alignment_utility: U_align(d)=E[cos(g_minus,g_star)|d]
      orthogonal_nuisance_cost: N_perp(d)=E[||Proj_perp_g_star(g_minus)||^2|d]
      net_utility: U_net(d)=U_align(d)-kappa*N_perp(d)
      hypothesis: Outside the local informative region, expected net utility is generally non-increasing with policy-relative
        distance and may approach zero or become negative. No exponential decay rate is assumed.
      interpretation: Far negatives may lose first-order alignment while retaining nonzero gradient magnitude, variance, curvature,
        support, or optimizer costs.
      universal_law_claimed: false
      exact_decay_rate_claimed: false
    influence_control_levels:
      raw_learnable_log_scale_order: Theta(d^2)
      quadratic_reciprocal_weight_order: Theta(d^-2)
      quadratic_weight_itself_vanishes: true
      quadratic_weighted_influence_limit: bounded_nonzero_constant_in_general
      bounded_influence_condition: w(d)=O(d^-2)
      vanishing_influence_condition: w(d)=o(d^-2)
      exponential_tail_property: d^k*exp(-beta*d)->0_for_every_fixed_finite_k
      exponential_role: smooth_vanishing_tail_candidate_not_unique_solution
      squared_distance_exponential_candidate: exp(-beta*(d/d_ref)^2)
      squared_distance_exponential_near_field_property: first_derivative_at_zero_is_zero
      current_E4_exponential_formula_unchanged: true
      no_universal_method_winner_assumed: true
  environment_identification_boundary:
    action_space_continuous: true
    reward_function_continuous: true
    negative_support_construction: eight_uniform_angles_selected_from_a_continuous_equal_reward_contour_per_state
    equal_reward_and_advantage_are_controlled_by_design: true
    quality_magnitude_decoupled_from_policy_relative_distance_within_negative_set: true
    directional_utility_decoupled_from_distance: false
    directional_utility_note: The present 2D geometry intentionally allows local negative repulsion to align with the hidden-optimum
      direction and far-side repulsion to become misaligned. This is a controlled informativeness-amplification mismatch,
      not a universal near-good/far-bad law.
    not_an_ood_protocol: true
  comparison_fairness_boundary:
    matched:
    - distance_coordinate
    - reference_weight_at_d_ref
    - negative_alpha
    - fixed_advantage
    - actor_initialization
    - minibatch_index_stream
    not_matched:
    - reference_point_slope
    - mean_near_negative_retention
    - total_negative_gradient_norm
    - cumulative_negative_optimizer_update
    allowed_claim: mechanism_order_under_anchor_normalization
    forbidden_claims:
    - best_tuned_quadratic_always_beats_best_tuned_linear
    - exponential_is_universally_best
    - distance_taper_always_beats_global_alpha
  followup_evidence_requirements:
    authorization_state: user_approved_first_successor_implemented_remaining_sequence_gated
    geometry_robustness:
    - continuous_angle_sampling
    - random_phase_per_state
    - negative_contour_resolution_sweep
    - thin_annulus_radial_jitter
    - reward_bin_matching
    fair_family_comparison:
    - matched_near_negative_retention
    - matched_stepwise_or_cumulative_negative_gradient_budget
    - equal_hyperparameter_search_budget
    - new_confirmatory_seeds
    - pareto_frontier_near_retention_far_risk_reward
    terminal_resolution:
    - original_adam_long_horizon
    - frozen_two_times_continuation
    - separate_full_batch_stationary_solution_audit_if_registered
    global_alpha_comparison: required_before_claiming_selective_distance_control_superiority
    geometry_robustness_priority: low_optional_not_a_current_gate
    registered_followup_experiments:
    - C-U1-E4-TAPER-NEAR-RETENTION-01
    - C-U1-E4-TAPER-BUDGET-MATCH-01
    - C-U1-E4-TAPER-CONV-01
    - C-U1-E4-TAPER-CONFIRM-01
    local_execution_order: near_retention_then_budget_match_then_convergence_then_confirmation
    no_automatic_execution: true
  control_role_notes:
    unweighted: internal_negative_control_and_runner_regression_anchor_not_a_new_scientific_claim
    positive_only: zero_negative_reference; E4 already owns the imitation_ceiling claim
  shared_implementation:
    core: src/drpo/cu1_core.py
    base_runner: src/drpo/drpo_cu1_e1_e4_oneclick.py
    componentwise_runner: src/drpo/cu1_e1_componentwise_rerun.py
    experiment_entrypoint: src/drpo/cu1_distance_taper_formal.py
    duplicated_environment_or_actor_allowed: false
  protocol:
    initialization_source: positive_only_adam_2000_step_checkpoint
    e2_post_2000_terminal_audit_checkpoint_used: false
    shared_across_methods: true
    state_distribution: Normal(0,I_6)
    train_states: 4096
    test_states: 4096
    positive_actions_per_state: 4
    equal_advantage_negative_actions_per_state: 8
    actor: shared_two_layer_MLP_isotropic_Gaussian_learnable_log_scale
    distance_reference: 5.0
    negative_alpha: 1.0
    distance_and_weight_stop_gradient: true
    primary_reference_weight_rho: 0.25
    sensitivity_reference_weights:
    - 0.5
    - 0.75
    development_seeds:
    - 0
    - 1
    - 2
    - 3
    - 4
    formal_held_out_seeds:
    - 70
    - 71
    - 72
    - 73
    - 74
    - 75
    - 76
    - 77
    - 78
    - 79
    - 80
    - 81
    - 82
    - 83
    - 84
    - 85
    - 86
    - 87
    - 88
    - 89
    optimizer: Adam
    learning_rate: 0.0005
    state_minibatch_size: 256
    evaluation_interval_steps: 100
    minimum_steps_before_stationarity: 1000
    maximum_steps: 8000
    stable_windows: 10
    normalized_slope_threshold: 0.0001
    normalized_field_residual_threshold: 0.002
    positive_only_absolute_gradient_threshold: 0.001
    terminal_extension: two_times_first_stable_candidate
  methods:
  - positive_only
  - unweighted
  - reciprocal_linear
  - reciprocal_quadratic
  - exponential
  formulas:
    normalized_distance: u=d/d_ref
    reciprocal_linear: w=1/(1+(rho^-1-1)*u)
    reciprocal_quadratic: w=1/(1+(rho^-1-1)*u^2)
    exponential: w=exp(log(rho)*u)
    common_reference_alignment: w(u=1)=rho
    negative_gradient_budget_matching: false
  linear_baseline_boundary: Reciprocal-linear is the internally defined p=1 control in the present same-distance taper theory.
    It is not the distributionally robust linear weighting from the original DRPO paper and does not require an external prior-work
    formula lock. Clipped-linear or any other family would be a different method and requires separate registration.
  primary_metrics:
  - initial_and_terminal_full_parameter_far_near_negative_gradient_ratio
  - initial_and_terminal_far_field_loglog_slope
  - output_mean_and_log_scale_components
  - held_out_context_reward
  - normalized_extrapolation_displacement
  - distance_to_a_plus
  - distance_to_a_star
  - sigma_trajectory
  - normalized_field_residual
  reporting_separation:
  - task_performance_collapse_event
  - support_or_variance_boundary_event
  - nan_inf_numerical_event
  primary_inference:
    comparison: reciprocal_quadratic_minus_reciprocal_linear_at_rho_0_25
    paired_bootstrap: true
    theory_predicts_lower_far_near_ratio: true
    theory_predicts_higher_reward: false
  terminal_audit: required
  formal_run_status: delivered
  execution:
    state: delivered
    run_id: cu1_e4_taper_054c2e2_run002
    completion_mode: hardened_guard_with_fresh_output_root
    start_utc: '2026-06-26T16:18:42.497876+00:00'
    end_utc: '2026-06-26T17:29:38.645563+00:00'
    elapsed_seconds: 4256.147687
    process_exit_code: 0
    runtime: cpu
    first_attempt_failure_preserved: true
  evidence:
    raw_complete: true
    terminal_audited: true
    terminal_audit_all_checks_passed: false
    terminal_scientific_acceptance_passed: false
    expected_runs: 220
    actual_runs: 220
    expected_primary_pairs: 20
    actual_primary_pairs: 20
    maximum_steps_unresolved_runs: 200
    support_or_variance_boundary_events: 20
    task_performance_collapse_events: 10
    nan_inf_numerical_events: 0
    stable_plateau_2x_confirmed_runs: 0
    checkpoint_packages_created: 3
    package_created: true
    package_filename: DRPO_CU1_E4_TAPER_054C2E2_FINAL_RESULTS_AND_UPDATE.zip
    package_sha256: null
    package_checksum_note: ZIP self-hash is reported at delivery; internal files are covered by SHA256SUMS.txt
    raw_complete_package_filename: C-U1-E4-TAPER-01_RUN002_RAW_COMPLETE.zip
    raw_complete_package_sha256: 18ce26dfd9762f645095035ec24d544e4ec832e05e167402db04acb972c20b16
    failed_attempt_package_filename: C-U1-E4-TAPER-01_RAW_COMPLETE.zip
    failed_attempt_package_sha256: caabbc0c38dded4b33c49d6a7b00b7dd4bfa8a4f38c56bdbe595424e966d3376
    delivered_to_user: true
    applied_commit: null
    compact_result_path: outputs/cu1_e4_taper
    scientific_status: finite_step_validated
  provenance:
    run_commit: 054c2e275cfd36e07e8883cb65d0b8df00460361
    repository_closure_base_commit: 054c2e275cfd36e07e8883cb65d0b8df00460361
    source_mode: exact_git_bundle_checkout
    source_bundle_filename: 8598d4f4-716d-41fa-8bb0-24ddc0a9a1d1.bundle
    source_bundle_sha256: e04bb1b7761921dc4591b1728dc982999e6ba9d4935e1c72d6749af505a37066
    git_bundle_verify_passed: true
    clean_worktree_at_launch: true
    clean_worktree_at_exit: true
    provenance_compromised: false
    cuda_available: false
    device: cpu
  result_summary:
    primary_rho: 0.25
    paired_seeds: 20
    quadratic_suppression_wins: 20
    quadratic_reward_wins: 20
    quadratic_minus_linear_reward_mean: 0.011371958255767822
    quadratic_minus_linear_reward_ci95:
    - 0.010951273515820504
    - 0.011826263964176178
    quadratic_minus_linear_far_near_ratio_mean: -1.6013766842192854
    quadratic_minus_linear_far_near_ratio_ci95:
    - -1.617527360639786
    - -1.5868169221681996
    linear_terminal_far_near_ratio_mean: 2.2746997012172607
    quadratic_terminal_far_near_ratio_mean: 0.6733230169979755
    linear_reward_mean: 0.6333347022533417
    quadratic_reward_mean: 0.6447066605091095
    exponential_rho_0_25_reward_mean: 0.6505336821079254
    exponential_rho_0_25_terminal_far_near_ratio_mean: 0.29548881279823996
    positive_only_reward_mean: 0.6467910826206207
  paper_use:
    suitable_for_finite_horizon_mechanism_order_claim: true
    suitable_for_terminally_stable_method_ranking: false
    allowed:
    - reciprocal_quadratic_stronger_far_field_suppression_than_reciprocal_linear_at_rho_0_25
    - paired_reward_advantage_for_quadratic_over_linear_in_this_C_U1_horizon
    - separate_reporting_of_task_support_and_numerical_events
    prohibited_claims:
    - long_run_validated
    - stable_fixed_point_ranking
    - OOD_generalization
    - universal_method_ranking
    - exponential_universal_winner
  next_gate:
    automatic_horizon_extension_forbidden: true
    convergence_resolution_requires_new_registration: true
    project_route: EXT-H-E7-Q2_ready_active
  artifact_budget:
    main_package_hard_limit_mib: 25
    checkpoint_policy: persistent_local_index
    checkpoint_every_formal_seeds: 5
- id: C-U1-E4-TAPER-NEAR-RETENTION-01
  execution_gate:
    state: ready
    blocked_by: []
    blocking_reason: resolved_by_completed_formal_run_and_v62_compact_repository_deposition
    depends_on_delivered_experiment: C-U1-E4-TAPER-01
    protocol_freeze: v61_near_retention_matching
    reason: The frozen 280-run matrix completed and was terminal-audited. The result is deposited as finite-step validated;
      260/280 runs remained terminally unresolved at 8000 steps, so no steady-state ranking is allowed. The original raw-complete
      artifact is not embedded in this repository update and its hash must be restored before archival publication.
  environment: C-U1
  name: taper_family_near_negative_retention_matched_comparison
  status: finite_step_validated
  scientific_status: finite_step_validated
  parent_experiment: C-U1-E4-TAPER-01
  registration_base_commit: 22161a91c0863278765b0d604ea82401d481b5aa
  implementation_base_commit: ce5964a0c16b12626ceb81fa9813fff14893c612
  claim: Under the shared C-U1 geometry and matched pooled average retention over the preregistered near region, test whether
    taper shape changes useful-near preservation, harmful-far influence, full-parameter far/near gradient allocation, and
    held-out-context task behavior. Reward superiority and a universal family winner are not assumed.
  role: controlled_taper_shape_fairness_validation
  execution_class: formal
  implementation_state: implemented
  formal_execution:
    channel_ref: hardened-v1
    activation_state: active
    entrypoint_status: implemented
    entrypoint: src/drpo/cu1_taper_near_retention_formal.py
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
  code_entrypoint: src/drpo/cu1_taper_near_retention_formal.py
  formal_launch_template: python3 scripts/run_experiment_guard_hardened.py --experiment-id C-U1-E4-TAPER-NEAR-RETENTION-01
    --repo-root . --output-root experiments/results/C-U1-E4-TAPER-NEAR-RETENTION-01/run_001 --artifact-output artifacts/C-U1-E4-TAPER-NEAR-RETENTION-01_RAW_COMPLETE.zip
    --heartbeat-seconds 60 --stale-seconds 900 --fail-on-stale --progress-glob per_seed_runs_partial.csv --required-output
    RUN_COMPLETE.json --required-output scientific_run_manifest.json --required-output formal_protocol_freeze.json --required-output
    calibration.json --required-output terminal_audit.json --required-output per_seed_runs.csv --required-output aggregate.csv
    --required-output paired_summary.json --source-file src/drpo/cu1_taper_near_retention_formal.py --source-file src/drpo/cu1_distance_taper_formal.py
    --source-file src/drpo/cu1_core.py --source-file src/drpo/drpo_cu1_e1_e4_oneclick.py --run-class formal --expected-commit
    "$(git rev-parse HEAD)" --require-origin-main-match -- python3 src/drpo/cu1_taper_near_retention_formal.py --output-dir
    experiments/results/C-U1-E4-TAPER-NEAR-RETENTION-01/run_001 --base-commit "$(git rev-parse HEAD)"
  matching_contract:
    target: pooled_E_weight_given_initial_standardized_distance_le_5
    retention_levels:
    - 0.75
    - 0.5
    - 0.25
    near_region_boundary: standardized_distance_d_le_5_at_frozen_positive_checkpoint
    primary_retention_level: 0.75
    sensitivity_retention_levels:
    - 0.5
    - 0.25
    coefficient_calibration_data: development_seeds_0_4_only
    coefficient_solver: deterministic_monotone_bisection
    calibration_tolerance: 1.0e-06
    coefficient_application: fixed_for_all_formal_seeds_and_all_training_steps
    confirmatory_seed_access_during_calibration: forbidden
    total_negative_gradient_budget_matched: false
  candidate_families:
  - reciprocal_linear
  - reciprocal_quadratic
  - current_exponential
  - squared_distance_exponential
  optional_family_requires_explicit_freeze: []
  formulas:
    normalized_distance: u=d/5
    reciprocal_linear: w=1/(1+c*u)
    reciprocal_quadratic: w=1/(1+c*u^2)
    current_exponential: w=exp(-c*u)
    squared_distance_exponential: w=exp(-c*u^2)
    coefficient_rule: one_development_calibrated_c_per_family_and_retention_level
    distance_and_weight_stop_gradient: true
  utility_diagnostics:
    oracle_direction: hidden_optimum_parameter_improvement_direction
    useful_near_definition: d_le_5_and_negative_update_alignment_positive
    harmful_far_definition: d_gt_5_and_negative_update_alignment_negative
    near_useful_retention: weighted_positive_projection_mass_over_raw_positive_projection_mass
    far_harmful_influence: weighted_negative_projection_mass_and_retention
    normalized_net_utility: cosine_minus_kappa_times_orthogonal_fraction_squared
    normalized_utility_kappa: 1.0
    distance_bins:
    - 0_to_2_5
    - 2_5_to_5
    - 5_to_7_5
    - 7_5_to_10
    - 10_to_infinity
    dimensional_net_utility_universal_claimed: false
  protocol:
    initialization_source: positive_only_adam_2000_step_checkpoint
    shared_C_U1_environment_and_actor: true
    development_seeds:
    - 0
    - 1
    - 2
    - 3
    - 4
    formal_paired_seeds:
    - 90
    - 91
    - 92
    - 93
    - 94
    - 95
    - 96
    - 97
    - 98
    - 99
    - 100
    - 101
    - 102
    - 103
    - 104
    - 105
    - 106
    - 107
    - 108
    - 109
    seeds_70_89_role: predecessor_evidence_only_not_confirmation
    seeds_110_and_above: untouched_for_successor_protocol_freeze
    optimizer: Adam
    learning_rate: 0.0005
    negative_alpha: 1.0
    state_minibatch_size: 256
    evaluation_interval_steps: 100
    minimum_steps_before_stationarity: 1000
    maximum_steps: 8000
    stable_windows: 10
    normalized_slope_threshold: 0.0001
    normalized_field_residual_threshold: 0.002
    positive_only_absolute_gradient_threshold: 0.001
    terminal_candidate_audit: complete_two_times_when_it_fits_inside_8000
    checkpoint_every_formal_seeds: 5
  controls:
  - positive_only
  - unweighted_negative
  primary_metrics:
  - calibrated_near_region_mean_weight
  - near_useful_gradient_retention
  - far_harmful_influence_retention
  - far_harmful_weighted_projection
  - full_parameter_far_near_negative_gradient_ratio
  - net_utility_by_distance_bin
  - held_out_context_reward
  - sigma_trajectory
  - task_performance_collapse_event
  - support_or_variance_boundary_event
  - nan_inf_numerical_event
  seed_policy:
    seeds_70_89_reuse_as_confirmation: forbidden
    exact_development_seeds:
    - 0
    - 1
    - 2
    - 3
    - 4
    exact_formal_seeds:
    - 90
    - 91
    - 92
    - 93
    - 94
    - 95
    - 96
    - 97
    - 98
    - 99
    - 100
    - 101
    - 102
    - 103
    - 104
    - 105
    - 106
    - 107
    - 108
    - 109
    seeds_110_and_above: untouched_for_successor_protocol_freeze
  primary_inference:
    paired_bootstrap: true
    primary_retention_level: 0.75
    reference_family: reciprocal_linear
    candidate_minus_linear:
    - reciprocal_quadratic
    - current_exponential
    - squared_distance_exponential
    reward_directional_hypothesis: none
    universal_family_winner_hypothesis: none
  terminal_audit:
    required: true
    unresolved_terminal_states_do_not_upgrade_to_long_run: true
    scientific_status_cap: finite_step_validated
    long_run_shortlist_resolution_owned_by: C-U1-E4-TAPER-CONV-01
    failure_types_separate:
    - task_performance_collapse
    - support_or_variance_boundary
    - nan_inf_numerical_failure
  required_outputs:
  - RUN_COMPLETE.json
  - scientific_run_manifest.json
  - formal_protocol_freeze.json
  - calibration.json
  - environment_audit.json
  - terminal_audit.json
  - per_seed_runs.csv
  - aggregate.csv
  - paired_summary.json
  - utility_bins.csv
  no_method_winner_assumed: true
  evidence:
    implementation_tests_passed: true
    smoke_test_passed: true
    formal_run_started: true
    run_started: true
    raw_complete_reported_by_completed_formal_run: true
    terminal_audited: true
    terminal_resolution_complete: false
    expected_runs: 280
    completed_runs: 280
    unresolved_at_maximum_steps: 260
    task_performance_collapse_events: 13
    support_or_variance_boundary_events: 20
    nan_inf_numerical_events: 0
    compact_repository_deposition_created: true
    original_raw_artifact_embedded: false
    original_raw_artifact_hash_available: false
    raw_complete: false
    package_created: false
  next_gate:
    experiment_id: C-U1-E4-TAPER-BUDGET-MATCH-01
    state: ready_after_v62_application
    automatic_activation_authorized_by_user: true
    convergence_and_confirmation_remain_blocked: true
    automatic_activation_forbidden: true
  result_deposition:
    compact_result_path: outputs/cu1_e4_taper_near_retention
    run_commit: 69c8f532570b5c4377a0cd35ff42f0bcb77afef0
    expected_runs: 280
    completed_runs: 280
    raw_rows_embedded_in_repository: false
    raw_complete_artifact_available_in_v62_build_session: false
    raw_artifact_hash_known_in_v62_build_session: false
    archival_recovery_required: true
  result_summary:
    primary_retention_level: 0.75
    calibration_maximum_absolute_error: 1.11e-16
    calibration_tolerance: 1.0e-06
    paired_seeds: 20
    reciprocal_quadratic_minus_linear_reward_mean: 0.012002
    current_exponential_minus_linear_reward_mean: 0.015619
    squared_distance_exponential_minus_linear_reward_mean: 0.036134
    candidate_positive_reward_delta_seeds:
      reciprocal_quadratic: 20
      current_exponential: 20
      squared_distance_exponential: 20
    reciprocal_linear_harmful_far_retention: 0.055886
    squared_distance_exponential_harmful_far_retention: 0.010382
    interpretation: Strong finite-horizon evidence that faster tail decay changes harmful-far allocation after initial average
      near retention is matched; no steady-state or universal winner claim.
  paper_use:
    suitable_for_finite_horizon_near_retention_matched_shape_claim: true
    suitable_for_terminally_stable_method_ranking: false
    allowed:
    - near_retention_matching_passed_within_tolerance
    - faster_tail_candidates_outperformed_reciprocal_linear_on_20_of_20_paired_seeds_at_rho_0_75
    - squared_distance_exponential_is_the_strongest_candidate_in_this_frozen_finite_horizon_matrix
    - task_support_and_numerical_events_reported_separately
    prohibited_claims:
    - long_run_validated
    - stable_fixed_point_ranking
    - OOD_generalization
    - universal_method_ranking
    - cross_task_superiority
    - Adam_optimizer_update_budget_was_matched
- id: C-U1-E4-TAPER-BUDGET-MATCH-01
  execution_gate:
    state: blocked
    blocked_by:
    - completed_formal_execution_no_rerun_without_new_registration
    blocking_reason: completed_formal_execution_terminal_audit_and_delivery_closure
    depends_on_delivered_experiment: C-U1-E4-TAPER-NEAR-RETENTION-01
    protocol_freeze: v62_stepwise_raw_negative_gradient_l2_budget_match
    reason: The frozen 140-run Budget-Match matrix completed at run commit 1faea3a92f74af5d11409779d96b9ed21fe846ad and passed
      coverage, budget, and terminal audits. The scientific state is finite-step validated. The original guard lifecycle failed
      only after successful computation because the runner omitted scientific_run_manifest.json and the default package size
      was exceeded; the original failed tree is preserved and the closure package repairs delivery without rerunning formal
      seeds. A rerun requires a new registration.
  environment: C-U1
  name: taper_family_negative_update_budget_matched_comparison
  status: finite_step_validated
  scientific_status: finite_step_validated
  parent_experiment: C-U1-E4-TAPER-01
  predecessor: C-U1-E4-TAPER-NEAR-RETENTION-01
  run_commit: 1faea3a92f74af5d11409779d96b9ed21fe846ad
  compact_result_path: outputs/cu1_e4_taper_budget_match/RESULT_SUMMARY.json
  registration_base_commit: 22161a91c0863278765b0d604ea82401d481b5aa
  claim: At the same per-step raw negative-gradient L2 norm before Adam, test whether distance-selective tapering allocates
    more update mass to useful near negatives and less to harmful far negatives than reciprocal-linear and a non-selective
    global stepwise scale. Adam parameter-update norms are logged but are not claimed to be matched.
  role: controlled_selectivity_vs_global_scale_validation
  execution_class: formal
  implementation_state: implemented
  formal_execution:
    channel_ref: hardened-v1
    activation_state: blocked
    entrypoint_status: implemented
    entrypoint: src/drpo/cu1_taper_budget_match_formal.py
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
  budget_contract:
    primary_mode: stepwise_raw_negative_gradient_l2_before_Adam
    reference_method: reciprocal_linear_at_near_retention_0_75
    reference_schedule: paired_seed_reciprocal_linear_actor_same_minibatch_indices
    matching_rule: detached_scalar_target_norm_divided_by_current_method_negative_gradient_norm
    budget_relative_tolerance: 1.0e-06
    matched_methods:
    - reciprocal_linear
    - reciprocal_quadratic
    - current_exponential
    - squared_distance_exponential
    - global_stepwise_scale
    global_control: unweighted_negative_gradient_direction_scaled_each_step_to_reference_norm
    boundary_controls:
    - positive_only
    - unweighted_boundary
    same_initialization_and_minibatch_stream: true
    only_negative_gradient_direction_and_near_far_allocation_may_differ: true
    Adam_parameter_update_norm_matched: false
    Adam_parameter_update_norm_logged: true
    forbidden_reinterpretation: Do not call this cumulative Adam optimizer-update matching.
    required_controls:
    - global_alpha
    - reciprocal_linear
    - reciprocal_quadratic
    - vanishing_tail_candidate
    candidate_matching_modes:
    - stepwise_negative_gradient_norm
    - cumulative_negative_optimizer_update
    exact_primary_mode: stepwise_raw_negative_gradient_l2_before_Adam
    only_selective_distance_allocation_may_differ: true
  primary_metrics:
  - maximum_stepwise_negative_gradient_budget_matching_error
  - near_useful_gradient_retention
  - far_harmful_influence_retention
  - near_and_far_negative_budget_fraction
  - held_out_context_reward
  - cumulative_raw_negative_gradient_norm
  - realized_Adam_total_parameter_update_norm_secondary
  - sigma_and_support_trajectory
  - task_performance_collapse_event
  - support_or_variance_boundary_event
  - nan_inf_numerical_event
  no_method_winner_assumed: true
  evidence:
    implementation_tests_passed: true
    engineering_smoke_passed: true
    formal_run_started: true
    run_started: true
    raw_complete: true
    terminal_audited: true
    terminal_audit_passed: true
    completed_runs: 140
    expected_runs: 140
    formal_seed_count: 20
    maximum_budget_relative_error: 2.1179505308403659e-16
    task_performance_collapse_events: 13
    support_or_variance_boundary_events: 20
    nan_inf_numerical_events: 0
    compact_repository_deposition_created: true
    original_guard_failed_after_compute: true
    original_guard_failure_reason:
    - missing_scientific_run_manifest
    - default_main_package_exceeded_25_MiB
    compute_returncode: 0
    provenance_compromised: false
    package_created: true
    package_delivered: true
    full_raw_archive_sha256: b635eabe3482fee92225f274292a78b21b8e09bf161dc42f4c151a549df6485d
  code_entrypoint: src/drpo/cu1_taper_budget_match_formal.py
  formal_launch_template: python3 scripts/run_experiment_guard_hardened.py --experiment-id C-U1-E4-TAPER-BUDGET-MATCH-01 --repo-root
    . --output-root experiments/results/C-U1-E4-TAPER-BUDGET-MATCH-01/run_001 --artifact-output artifacts/C-U1-E4-TAPER-BUDGET-MATCH-01_RAW_COMPLETE.zip
    --heartbeat-seconds 60 --stale-seconds 900 --fail-on-stale --progress-glob per_seed_runs_partial.csv --required-output
    RUN_COMPLETE.json --required-output scientific_run_manifest.json --required-output formal_protocol_freeze.json --required-output
    calibration.json --required-output budget_audit.json --required-output terminal_audit.json --required-output per_seed_runs.csv
    --required-output aggregate.csv --required-output paired_summary.json --source-file src/drpo/cu1_taper_budget_match_formal.py
    --source-file src/drpo/cu1_taper_near_retention_formal.py --source-file src/drpo/cu1_distance_taper_formal.py --source-file
    src/drpo/cu1_core.py --source-file src/drpo/drpo_cu1_e1_e4_oneclick.py --run-class formal --expected-commit "$(git rev-parse
    HEAD)" --require-origin-main-match -- python3 src/drpo/cu1_taper_budget_match_formal.py --output-dir experiments/results/C-U1-E4-TAPER-BUDGET-MATCH-01/run_001
    --base-commit "$(git rev-parse HEAD)"
  protocol:
    initialization_source: positive_only_adam_2000_step_checkpoint
    coefficient_source: development_seeds_0_4_near_retention_0_75_calibration
    development_seeds:
    - 0
    - 1
    - 2
    - 3
    - 4
    formal_paired_seeds:
    - 110
    - 111
    - 112
    - 113
    - 114
    - 115
    - 116
    - 117
    - 118
    - 119
    - 120
    - 121
    - 122
    - 123
    - 124
    - 125
    - 126
    - 127
    - 128
    - 129
    confirmation_seeds_130_149_access: forbidden
    optimizer: Adam
    learning_rate: 0.0005
    negative_alpha_before_budget_rescaling: 1.0
    state_minibatch_size: 256
    evaluation_interval_steps: 100
    minimum_steps_before_stationarity: 1000
    maximum_steps: 8000
    stable_windows: 10
    normalized_slope_threshold: 0.0001
    normalized_field_residual_threshold: 0.002
    positive_only_absolute_gradient_threshold: 0.001
    checkpoint_every_formal_seeds: 5
    scientific_status_cap: finite_step_validated
  required_outputs:
  - RUN_COMPLETE.json
  - scientific_run_manifest.json
  - formal_protocol_freeze.json
  - calibration.json
  - environment_audit.json
  - budget_audit.json
  - terminal_audit.json
  - per_seed_runs.csv
  - aggregate.csv
  - paired_summary.json
  terminal_audit:
    required: true
    equal_finite_horizon: 8000
    fixed_horizon_does_not_imply_convergence: true
    scientific_status_cap: finite_step_validated
    long_run_resolution_owned_by: C-U1-E4-TAPER-CONV-01
    failure_types_separate:
    - task_performance_collapse
    - support_or_variance_boundary
    - nan_inf_numerical_failure
  result_summary:
    primary_budget_coordinate: stepwise_raw_negative_gradient_l2_before_Adam
    maximum_budget_relative_error: 2.1179505308403659e-16
    reciprocal_quadratic_reward_delta_vs_linear: 0.016011109948158263
    exponential_reward_delta_vs_linear: 0.08818890154361725
    squared_distance_exponential_reward_delta_vs_linear: 0.13061619997024537
    global_stepwise_scale_reward_delta_vs_linear: -0.006882542371749878
    reciprocal_quadratic_reward_positive_seeds_vs_linear: 20
    exponential_reward_positive_seeds_vs_linear: 20
    squared_distance_exponential_reward_positive_seeds_vs_linear: 20
    global_stepwise_scale_reward_positive_seeds_vs_linear: 0
    reciprocal_quadratic_far_harmful_lower_seeds_vs_linear: 20
    exponential_far_harmful_lower_seeds_vs_linear: 20
    squared_distance_exponential_far_harmful_lower_seeds_vs_linear: 20
    global_stepwise_scale_far_harmful_lower_seeds_vs_linear: 0
    interpretation: Under exact finite-horizon raw-negative-gradient budget matching, selective taper shape changed harmful-far
      allocation and held-out-context reward; terminal useful-near retention was undefined, and this does not establish steady-state
      or universal ranking.
  paper_use:
    suitable_for_finite_horizon_budget_matched_selectivity_claim: true
    suitable_for_terminally_stable_method_ranking: false
    prohibited_claims:
    - long_run_validated
    - steady_state_ranking
    - OOD_generalization
    - universal_method_ranking
    - cross_task_superiority
    - Adam_parameter_update_norm_was_matched
  next_gate:
    experiment_id: C-U1-E4-TAPER-CONV-01
    state: blocked_until_shortlist_frozen_and_continuation_runner_implemented
    automatic_activation_forbidden: true
- id: C-U1-E4-TAPER-CONV-01
  execution_gate:
    state: blocked
    blocked_by:
    - frozen_shortlist_json_generated_from_preregistered_rule
    - separately_implemented_continuation_runner
    blocking_reason: Budget-Match is delivered as finite-step validated. The next action is a separate, deterministic shortlist-freeze
      update using the preregistered rule, followed by implementation of a continuation runner that consumes exact actor and
      Adam optimizer checkpoints. No old 8000-step run may be extended ad hoc.
  environment: C-U1
  name: taper_frozen_shortlist_long_run_terminal_resolution
  status: not_run
  scientific_status: not_run
  parent_experiment: C-U1-E4-TAPER-01
  predecessors:
  - C-U1-E4-TAPER-NEAR-RETENTION-01
  - C-U1-E4-TAPER-BUDGET-MATCH-01
  registration_base_commit: 22161a91c0863278765b0d604ea82401d481b5aa
  claim: With the method shortlist and hyperparameters frozen by prior fairness studies, determine whether the finite-horizon
    ordering persists under the original Adam dynamics through a registered terminal candidate and complete two-times continuation.
  role: controlled_taper_long_run_terminal_resolution
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
  terminal_contract:
    continuation_seeds:
    - 110
    - 111
    - 112
    - 113
    - 114
    - 115
    - 116
    - 117
    - 118
    - 119
    - 120
    - 121
    - 122
    - 123
    - 124
    - 125
    - 126
    - 127
    - 128
    - 129
    input_checkpoint_owner: C-U1-E4-TAPER-BUDGET-MATCH-01
    actor_state_continuity: required
    Adam_optimizer_state_continuity: required
    reference_budget_schedule_continuation: reciprocal_linear_continues_first_then_other_methods_use_its_frozen_stepwise_norm_schedule
    starting_step: 8000
    maximum_total_steps: 32000
    evaluation_interval_steps: 100
    minimum_total_steps_before_new_terminal_candidate: 10000
    stable_windows: 10
    normalized_slope_threshold: 0.0001
    normalized_field_residual_threshold: 0.002
    complete_two_times_continuation: required_when_candidate_times_two_is_at_most_32000
    persistent_drift_or_runaway_is_a_resolved_terminal_class_if_explicitly_audited: true
    fixed_horizon_extension_without_registration: forbidden
    full_batch_stationary_audit: optional_separate_registered_diagnostic_not_a_replacement
    separate_failure_reporting:
    - task_performance_collapse
    - support_or_variance_boundary
    - nan_inf_numerical_failure
    optimizer: original_Adam_dynamics
    optimizer_state_continuity: required
    provisional_candidate_rule: pending_protocol_freeze
  no_method_winner_assumed: true
  evidence:
    run_started: false
    raw_complete: false
    terminal_audited: false
    package_created: false
  shortlist_freeze_rule:
    always_include:
    - positive_only
    - unweighted_boundary
    - reciprocal_linear
    - global_stepwise_scale
    selective_candidate_pool:
    - reciprocal_quadratic
    - current_exponential
    - squared_distance_exponential
    eligibility_all_required:
    - near_retention_primary_result_positive_reward_delta_vs_reciprocal_linear_on_at_least_18_of_20_seeds
    - budget_match_far_harmful_retention_lower_than_global_stepwise_scale_on_at_least_18_of_20_seeds
    - budget_match_reward_delta_vs_reciprocal_linear_nonnegative_on_at_least_18_of_20_seeds
    - no_more_nan_inf_events_than_reciprocal_linear
    maximum_selective_candidates: 2
    tie_break_order:
    - budget_match_mean_held_out_context_reward_descending
    - budget_match_mean_far_harmful_retention_ascending
    - family_name_lexicographic
    freeze_artifact_required: FROZEN_CONVERGENCE_SHORTLIST.json_with_sha256
    result_dependent_manual_override: forbidden_without_new_user_approved_registration
  confirmation_seed_access:
    seeds_130_149: forbidden
- id: C-U1-E4-TAPER-CONFIRM-01
  execution_gate:
    state: blocked
    blocked_by:
    - C-U1-E4-TAPER-CONV-01_delivered
    - frozen_confirmation_config_with_hash
    - separately_implemented_confirmation_runner
    blocking_reason: Confirmation is last. Seeds 130-149 are reserved now and may not be accessed by Budget-Match or Convergence.
      The final shortlist, coefficients, long-run horizon, terminal rules, and primary claims must be frozen after Convergence
      delivery and before any confirmation access.
  environment: C-U1
  name: taper_frozen_protocol_independent_seed_confirmation
  status: not_run
  scientific_status: not_run
  parent_experiment: C-U1-E4-TAPER-01
  predecessor: C-U1-E4-TAPER-CONV-01
  registration_base_commit: 22161a91c0863278765b0d604ea82401d481b5aa
  claim: On untouched confirmatory seeds, test whether the fully frozen fairness and long-run conclusions replicate without
    seed selection or post-result hyperparameter adjustment.
  role: controlled_taper_confirmatory_replication
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
  confirmation_contract:
    exact_untouched_seeds:
    - 130
    - 131
    - 132
    - 133
    - 134
    - 135
    - 136
    - 137
    - 138
    - 139
    - 140
    - 141
    - 142
    - 143
    - 144
    - 145
    - 146
    - 147
    - 148
    - 149
    earlier_seed_roles:
      '70_89': predecessor_TAPER_evidence_only
      '90_109': near_retention_formal_evidence_only
      '110_129': budget_and_continuation_evidence_only
    seed_access_before_frozen_confirmation_config: forbidden
    hyperparameter_retuning_after_confirmation_start: forbidden
    primary_claim_change_after_confirmation_start: forbidden
    method_formula_change_after_confirmation_start: forbidden
    registered_terminal_audit: required
    maximum_total_steps_inherit_from_frozen_convergence_config: true
    same_distribution_held_out_context_only: true
    OOD_claim_allowed: false
    seeds_70_89: development_evidence_not_confirmatory
  primary_metrics:
  - held_out_context_reward
  - near_useful_retention
  - far_harmful_influence
  - terminal_state_classification
  - task_performance_collapse_event
  - support_or_variance_boundary_event
  - nan_inf_numerical_event
  no_method_winner_assumed: true
  evidence:
    run_started: false
    raw_complete: false
    terminal_audited: false
    package_created: false
  primary_analysis_plan:
    mechanism_confirmation_report_separately:
    - near_useful_retention_noninferiority
    - far_harmful_retention_improvement_vs_reciprocal_linear
    - far_harmful_retention_improvement_vs_global_stepwise_scale
    task_confirmation_report_separately:
    - paired_held_out_context_reward_delta_vs_reciprocal_linear
    - paired_held_out_context_reward_delta_vs_global_stepwise_scale
    terminal_confirmation_report_separately:
    - terminal_state_classification
    - classification_reversal
    - persistent_drift_or_runaway
    - task_performance_collapse
    - support_or_variance_boundary
    - nan_inf_numerical_failure
    minimum_directional_consistency_seeds: 16
    paired_bootstrap_confidence_interval: 0.95
    steady_state_ranking_requires_terminal_resolution: required
    failure_of_task_superiority_does_not_erase_mechanism_result: true
