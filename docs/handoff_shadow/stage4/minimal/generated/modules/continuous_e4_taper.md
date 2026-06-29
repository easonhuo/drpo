# Continuous C-U1 E4 extrapolation and taper track

> Generated Stage 4 minimal-context shadow module. Do not edit manually.
> `docs/handoff.md` and `experiments/registry.yaml` remain authoritative.

- Module ID: `continuous_e4_taper`
- Responsibility: Cover stable extrapolation, phase transition, taper-family comparisons, fairness controls, and the frozen E4 follow-up order.
- Source hash: `158b04774f4940adae6a07349fa7b05dc760c90eb91847435d31174f29799241`

## Source 1: docs/handoff.md: ### 3.6.2 E4：稳定外推—相变—远场控制 -> ## 3.7 D-U1 / E6 开发配置登记（E4 已完成；用户已批准与 E4-TAPER 并行）

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

## Source 2: docs/handoff.md: ## 3.8 C-U1 共享实现与二次阶方法实验 `C-U1-E4-TAPER-01` -> ## 3.9 E6--E8 方法迁移与规模验证路线（v42 锁定）

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

## Source 3: docs/handoff.md: HANDOFF-DELTA-BLOCKs matching 'C-U1-E4-TAPER'

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

### Delta block `section_end:v60-e4-taper-current-gate`

- **E4-TAPER v60 覆盖：** `C-U1-E4-TAPER-01` 仍为 finite-step validated。四个后续 ID 已获用户批准并登记，但全部保持 blocked：先冻结并实现 `NEAR-RETENTION-01`，交付后才允许冻结 `BUDGET-MATCH-01`；二者交付并冻结 shortlist 后才允许 `CONV-01`；最后才用 untouched seeds 执行 `CONFIRM-01`。原实验禁止自动延长，几何 robustness 不作为当前门禁。

### Delta block `section_end:v61-e4-taper-near-retention-current-gate`

- **E4-TAPER v61 覆盖：** `C-U1-E4-TAPER-NEAR-RETENTION-01` 已完成协议冻结、独立 runner、formal-channel 登记和工程 smoke，registry 为 **implemented + ready + active + not_run**。允许下一步启动该实验的 canonical guarded formal run，但 smoke/单元测试不构成科学结果。`BUDGET-MATCH-01` 仍必须等待 Near-Retention 的 raw-complete、终态审计、打包与交付；不得提前实现为可运行状态或并行启动。

### Delta block `section_end:v60-e4-taper-utility-theory-and-followups`

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

### Delta block `section_end:v61-e4-taper-near-retention-protocol`

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

## Source 4: experiments/registry.yaml: experiments[C-U1-E4-ADAM-RERUN, C-U1-E4-CONV-01, C-U1-E4-TAPER-01, C-U1-E4-TAPER-NEAR-RETENTION-01, C-U1-E4-TAPER-BUDGET-MATCH-01, C-U1-E4-TAPER-CONV-01, C-U1-E4-TAPER-CONFIRM-01]

collection: experiments
entries:
- id: C-U1-E4-ADAM-RERUN
  environment: C-U1
  name: cu1_stable_extrapolation_phase_transition_unified_adam
  status: finite_step_validated
  claim: Test whether controlled local negative gradients improve beyond the positive-only ceiling and whether excessive negative
    pressure causes support contraction, continuing drift, or task-performance collapse under one Adam training pipeline.
  role: controlled_generalization_and_phase_transition
  execution_class: historical_formal
  historical_formal_execution:
    channel_status: grandfathered_completed_run
    future_rerun_requires_channel: hardened-v1
  depends_on_delivered_experiment: C-U1-E3-ADAM-RERUN
  code_entrypoint: src/drpo/drpo_cu1_e1_e4_oneclick.py
  command:
  - python
  - src/drpo/drpo_cu1_e1_e4_oneclick.py
  - --stage
  - e4
  - --output-root
  - outputs/cu1_e4_adam
  initialization:
    source: positive_only_adam_2000_step_checkpoint
    e2_terminal_audit_checkpoint_used: false
    shared_across_methods: true
  optimizer:
    name: Adam
    betas:
    - 0.9
    - 0.999
    eps: 1.0e-08
    lr: 0.0005
    lbfgs_parameter_updates_in_e4: false
  data:
    train_states: 4096
    test_states: 4096
    positive_actions_per_state: 4
    negative_actions_per_state: 8
    terminology: held_out_context_generalization
    fixed_advantage: true
  development_seeds:
  - 5
  - 6
  - 7
  - 8
  - 9
  held_out_seeds:
  - 50
  - 51
  - 52
  - 53
  - 54
  - 55
  - 56
  - 57
  - 58
  - 59
  - 60
  - 61
  - 62
  - 63
  - 64
  - 65
  - 66
  - 67
  - 68
  - 69
  fixed_variance_alpha_grid:
  - 0.0
  - 0.25
  - 0.5
  - 0.75
  - 1.0
  - 1.25
  - 1.5
  - 1.75
  learnable_variance_alpha_grid:
  - 0.0
  - 0.1
  - 0.2
  - 0.3
  - 0.35
  - 0.38
  - 0.4
  - 0.5
  main_story:
  - positive_only_ceiling
  - beneficial_controlled_negative_region
  - excessive_negative_pressure_support_contraction_or_task_collapse
  appendix_controls:
  - uncontrolled_all
  - far_cap
  - budget_matched_global
  terminal_audit:
    finite_internal_solution:
    - 200_step_adam
    - full_data_residual_audit_1
    - 200_step_adam_continuation
    - full_data_residual_audit_2
    no_internal_solution:
    - long_run_adam
    - drift_or_first_boundary_audit
    integrity_checks_all_passed: true
    scientific_terminal_acceptance_passed: false
    failure_reason: The finite-horizon beneficial branch is reproducible, but no beneficial alpha passes both frozen full-data
      residual audits in 20/20 seeds. Fixed-variance alpha=1.00 passes both audits in only 3/20 seeds.
  metrics:
  - held_out_context_reward
  - normalized_extrapolation_displacement
  - distance_to_a_plus_and_a_star
  - normalized_field_residual
  - support_contraction_onset
  - raw_total_gradient_norm
  - adam_parameter_update_norm
  - parameter_log_sigma_sigma_output_finiteness
  method_ranking_pre_registered: false
  controls_note: Raw-gradient matched controls are not called Adam-update matched.
  formal_run_status: delivered
  execution:
    state: delivered
    run_id: cu1_e4_adam_d699bb6_run001
    completion_mode: hardened_foreground_supervisor
    start_utc: '2026-06-26T03:55:06.139980+00:00'
    end_utc: '2026-06-26T04:25:23.841907+00:00'
    last_heartbeat_utc: '2026-06-26T04:25:21.671266+00:00'
    process_exit_code: 0
    elapsed_seconds: 1817.702
  evidence:
    raw_complete: true
    terminal_audited: true
    terminal_audit_integrity_all_checks_passed: true
    terminal_scientific_acceptance_passed: false
    expected_fixed_rows: 160
    actual_fixed_rows: 160
    expected_learnable_rows: 160
    actual_learnable_rows: 160
    expected_control_rows: 60
    actual_control_rows: 60
    expected_variance_robustness_rows: 45
    actual_variance_robustness_rows: 45
    missing_required_files: 0
    checkpoint_packages_created: 4
    package_created: true
    package_filename: DRPO_CU1_E4_ADAM_D699_FINAL_EVIDENCE.zip
    package_sha256: c2fbc594891b594652338b8937d02d4b283e75caa7cd475572ca7307f6f08673
    raw_complete_package_filename: DRPO_CU1_E4_ADAM_RAW_COMPLETE_D699_RUN001.zip
    raw_complete_package_sha256: daf7d133692335db477a5c5b42706b96d245e13696b6ea181f0e2895ee2387e8
    delivered_to_user: true
    applied_commit: null
    compact_result_path: outputs/cu1_e4_adam
  provenance:
    run_commit: d699bb6b1d0093d8a9b935fd6c67f049fc3c3df0
    repository_closure_base_commit: d699bb6b1d0093d8a9b935fd6c67f049fc3c3df0
    source_mode: exact_git_bundle_checkout
    source_bundle_filename: DRPO_MAIN_d699bb6b1d00.bundle
    source_bundle_sha256: d9c5dfa5b914d4e17e224849ebed6b400c6cadc26cc41ad60d6fcc510b0bc5bb
    git_bundle_verify_passed: true
    clean_worktree_at_launch: true
    clean_worktree_at_exit: true
    provenance_compromised: false
    cuda_available: false
    device: cpu
  result_summary:
    positive_only_ceiling:
      fixed_alpha_0_reward_mean: 0.646987646818161
    finite_horizon_beneficial_region:
      fixed_alpha_0_25_reward_mean: 0.7144076138734817
      fixed_alpha_0_50_reward_mean: 0.8040281385183334
      fixed_alpha_0_75_reward_mean: 0.9153103202581405
      fixed_alpha_1_00_reward_mean: 0.9917025655508042
      fixed_alpha_1_00_reward_ci95:
      - 0.99136316254735
      - 0.9920345157384872
      fixed_alpha_1_00_normalized_displacement_mean: 1.007807719707489
      each_alpha_0_25_through_1_00_beats_alpha_0_paired_seeds: 20
      stable_fixed_point_claim_terminally_validated: false
      fixed_alpha_1_00_both_residual_audits_passed: 3
    excessive_fixed_pressure:
      fixed_alpha_1_50_task_collapse_count: 20
      fixed_alpha_1_75_task_collapse_count: 20
      fixed_alpha_1_75_continuing_runaway_count: 20
      nan_inf_count: 0
    learnable_variance:
      alpha_0_40_support_contraction_count: 18
      alpha_0_40_support_onset_median: 434.5
      alpha_0_50_support_contraction_count: 20
      alpha_0_50_support_onset_median: 83.0
      unexpected_support_expansion_count: 0
      nan_inf_count: 0
    variance_robustness:
      alpha_0_38_cross_minus_8_count: 0
      alpha_0_40_cross_minus_8_count: 15
      alpha_0_40_cross_minus_12_count: 11
      alpha_0_50_cross_minus_14_count: 15
      total_rows: 45
    controls_4000_steps:
      uncontrolled_all_reward_mean: 0.0
      uncontrolled_all_task_failure_count: 20
      far_cap_reward_mean: 0.9952240824699402
      far_cap_task_failure_count: 0
      budget_matched_global_reward_mean: 0.502925130724907
      budget_matched_global_task_failure_count: 0
      nonfinite_count: 0
  paper_use:
    suitable_for_finite_horizon_phase_figure: true
    suitable_for_stable_fixed_point_claim: false
    allowed:
    - positive_only_ceiling_and_finite_horizon_benefit
    - excessive_fixed_pressure_task_collapse
    - learnable_variance_support_contraction
    - long_run_controls_as_appendix_evidence
    prohibited_claims:
    - terminally_stable_beneficial_fixed_point
    - OOD_generalization
    - universal_method_ranking
    - NaN_Inf_collapse
  next_gate:
    convergence_experiment: C-U1-E4-CONV-01
    convergence_status: long_run_validated_user_confirmed_closure
    taper_status: ready
    reason: The original 18/20 consensus did not pass, but the user explicitly accepted the preserved 15/20, 16/20, and 15/20
      expected-state counts together with 0/60 explicit opposite states and 60/60 non-reversed scientific roles as sufficient
      long-horizon E4 closure. TAPER may start only after this closure update is applied and committed.
  scientific_status: finite_step_validated
  preserved_history: true
- id: C-U1-E4-CONV-01
  environment: C-U1
  name: cu1_e4_fixed_variance_long_horizon_terminal_confirmation
  status: long_run_validated
  scientific_status: long_run_validated
  claim: Confirm that fixed-variance alpha=0.75 and alpha=1.00 remain bounded beneficial extrapolation rather than transient
    optimum crossings, and that alpha=1.25 is a stable over-extrapolated state rather than slow runaway.
  role: controlled_long_horizon_terminal_confirmation
  execution_class: formal
  formal_execution:
    channel_ref: hardened-v1
    activation_state: active
    entrypoint_status: implemented
    entrypoint: src/drpo/drpo_cu1_e1_e4_oneclick.py
    launch_mode: canonical_guard
    artifact_owner: canonical_channel
    guard_entrypoint: scripts/run_experiment_guard_hardened.py
    package_entrypoint: scripts/package_experiment_hardened.py
    verify_entrypoint: scripts/verify_experiment_package_hardened.py
    hardened_core: scripts/artifact_protocol_hardened.py
    artifact_protocol: docs/formal_experiment_artifact_protocol.md
    inherit_default_artifact_budget: true
    runner_archive_policy:
      mode: legacy_exception
      exception_id: CU1-RECOVERY-CHECKPOINT-LEGACY-01
  depends_on_delivered_experiment: C-U1-E4-ADAM-RERUN
  blocks_experiment: C-U1-E4-TAPER-01
  code_entrypoint: src/drpo/drpo_cu1_e1_e4_oneclick.py
  stage: e4_convergence
  command:
  - python3
  - src/drpo/drpo_cu1_e1_e4_oneclick.py
  - --stage
  - e4_convergence
  - --output-root
  - experiments/results/C-U1-E4-CONV-01/run_001
  guard_entrypoint: scripts/run_experiment_guard_hardened.py
  formal_launch_template: python3 scripts/run_experiment_guard_hardened.py --run-class formal --expected-commit "$(git rev-parse
    HEAD)" --experiment-id C-U1-E4-CONV-01 --repo-root . --output-root experiments/results/C-U1-E4-CONV-01/run_001 --artifact-output
    artifacts/C-U1-E4-CONV-01_RAW_COMPLETE.zip -- python3 src/drpo/drpo_cu1_e1_e4_oneclick.py --stage e4_convergence --output-root
    experiments/results/C-U1-E4-CONV-01/run_001
  scope:
    fixed_variance_only: true
    alphas:
    - 0.75
    - 1.0
    - 1.25
    positive_only_additional_run: false
    positive_only_terminal_evidence_owner: C-U1-E2
    rerun_full_e4: false
    rerun_learnable_variance: false
    rerun_controls: false
    rerun_alpha_1_50_or_1_75: false
  initialization:
    source: positive_only_adam_2000_step_checkpoint
    e2_terminal_audit_checkpoint_used: false
    restart_from_same_e2_initialization: true
    continue_from_old_400_step_checkpoint: false
    shared_across_alphas: true
  optimizer:
    name: Adam
    betas:
    - 0.9
    - 0.999
    eps: 1.0e-08
    lr: 0.0005
    fixed_sigma: 0.1903943276465978
    batch_or_rng_changes_allowed: false
  data:
    train_states: 4096
    test_states: 4096
    positive_actions_per_state: 4
    negative_actions_per_state: 8
    terminology: held_out_context_generalization
    fixed_advantage: true
  held_out_seeds:
  - 50
  - 51
  - 52
  - 53
  - 54
  - 55
  - 56
  - 57
  - 58
  - 59
  - 60
  - 61
  - 62
  - 63
  - 64
  - 65
  - 66
  - 67
  - 68
  - 69
  training:
    max_steps: 4000
    full_state_audit_steps:
    - 400
    - 800
    - 1600
    - 2400
    - 3200
    - 4000
    terminal_window_1:
    - 2000
    - 3000
    terminal_window_2:
    - 3000
    - 4000
    checkpoint_every_formal_seeds: 5
    silent_horizon_extension_allowed: false
  terminal_classification:
    residual_threshold_2e_3_is_hard_gate: false
    normalized_field_residual_retained_as_diagnostic: true
    stable_platform:
      window_2_absolute_displacement_change_max: 0.02
      window_2_absolute_reward_change_max: 0.01
      window_2_over_window_1_raw_gradient_median_ratio_max: 1.25
      window_2_over_window_1_adam_update_median_ratio_max: 1.25
      scientific_role_step_2000_to_4000_must_not_reverse: true
    continuing_runaway:
      window_1_displacement_increase_required: true
      window_2_displacement_change_min: 0.05
      raw_gradient_or_adam_update_ratio_min_exclusive: 1.25
    otherwise: terminal_state_inconclusive
  expected_terminal_states:
    alpha_0_75: stable_beneficial_extrapolation
    alpha_1_00: stable_beneficial_extrapolation
    alpha_1_25: stable_over_extrapolation
  aggregate_acceptance:
    minimum_expected_state_seeds_per_alpha: 18
    remaining_seeds_allowed_state: terminal_state_inconclusive
    explicit_opposite_terminal_state_allowed: false
  metrics:
  - held_out_context_reward
  - normalized_extrapolation_displacement
  - distance_to_a_plus_and_a_star
  - full_data_raw_total_gradient_norm
  - minibatch_raw_total_gradient_norm
  - adam_parameter_update_norm
  - normalized_field_residual_diagnostic
  - task_performance_collapse
  - support_or_variance_boundary_event
  - nan_inf_numerical_failure
  reporting_separation:
  - task_performance_collapse
  - support_or_variance_boundary
  - nan_inf_numerical_failure
  method_ranking_pre_registered: false
  formal_run_status: delivered
  execution:
    state: delivered
    run_id: cu1_e4_conv_c869df8_run002
    completion_mode: scientific_child_completed_wrapper_packaging_recovered
    start_utc: '2026-06-26T06:37:47.424130+00:00'
    end_utc: '2026-06-26T06:48:25.998924+00:00'
    process_exit_code: 0
    elapsed_seconds: 635.0746870040894
    first_attempt_failure_preserved: true
    wrapper_required_output_mismatch_preserved: true
  evidence:
    raw_complete: true
    terminal_audited: true
    package_created: true
    package_filename: DRPO_CU1_E4_CONV_C869_RUN002_RAW_COMPLETE.zip
    package_sha256: 98214c2f09f7cd6ba75472bfc489771cb2ac439031e9f3636a8472a6c2a06b13
    failed_attempt_package_filename: DRPO_CU1_E4_CONV_RUN001_FAILED_C869.zip
    failed_attempt_package_sha256: 765c40e3b2df1e980ef786cdf5f6dddd912d1d149d111207d822b098cf2a99ff
    guard_wrapper_failure_package_filename: DRPO_CU1_E4_CONV_C869_RUN002_GUARD_FAILED.zip
    guard_wrapper_failure_package_sha256: bef2e57cf0646b3aa3556156fcb710a611cebb397847af3ea033f59316e2a802
    expected_rows: 60
    actual_rows: 60
    checkpoint_packages_created: 4
    delivered_to_user: true
    applied_commit: null
    compact_result_path: outputs/cu1_e4_convergence
  paper_use:
    allowed:
    - long_horizon_beneficial_extrapolation_for_alpha_0_75_and_1_00_with_exact_counts
    - long_horizon_stable_over_extrapolation_for_alpha_1_25_with_exact_counts
    - no_explicit_opposite_terminal_state_in_60_runs
    - all_60_scientific_roles_not_reversed_from_step_2000_to_4000
    - aggregate_displacement_and_reward_remain_near_stationary
    prohibited:
    - registered_18_of_20_terminal_gate_passed
    - all_20_seeds_fixed_point_certified
    - every_seed_strictly_stationary
    - OOD_generalization
    - universal_method_ranking
  user_confirmed_closure:
    confirmed: true
    decision_date: '2026-06-26'
    decision_type: post_run_explicit_user_evidence_review
    original_pre_registered_consensus_gate_passed: false
    per_alpha_expected_state_counts:
      alpha_0_75: 15
      alpha_1_00: 16
      alpha_1_25: 15
    total_explicit_opposite_terminal_states: 0
    total_scientific_roles_not_reversed: 60
    accepted_scientific_scope: 'The 4000-step evidence closes the E4 long-horizon phase claim: alpha=0.75 and 1.00 retain
      bounded beneficial extrapolation without role reversal, while alpha=1.25 is stable over-extrapolation rather than slow
      runaway.'
    excluded_scope: This decision does not certify 20/20 fixed points, does not claim that the original 18/20 gate passed,
      and does not relabel inconclusive seed-alpha rows.
  preserved_history: true
  terminal_audit:
    integrity_checks_all_passed: true
    scientific_terminal_acceptance_passed: false
    pre_registered_18_of_20_gate_passed: false
    user_confirmed_scoped_scientific_closure_passed: true
    consensus_min: 18
    alpha_0_75_expected_state_count: 15
    alpha_0_75_inconclusive_count: 5
    alpha_1_00_expected_state_count: 16
    alpha_1_00_inconclusive_count: 4
    alpha_1_25_expected_state_count: 15
    alpha_1_25_inconclusive_count: 5
    explicit_opposite_terminal_state_count: 0
    scientific_role_not_reversed_count: 60
    task_performance_collapse_count: 0
    support_or_variance_boundary_count: 0
    nan_inf_count: 0
    failure_reason: No alpha reached the frozen 18/20 expected-state consensus. All remaining runs were inconclusive and no
      explicit opposite terminal state occurred.
    post_hoc_diagnostic_only: Fourteen seed-alpha rows were inconclusive because a raw-gradient or Adam-update W2/W1 ratio
      exceeded 1.25; displacement and reward windows remained within frozen stability bounds. This does not override the original
      gate.
    closure_note: The user explicitly accepted the preserved majority-state evidence and absence of opposite states as sufficient
      for the scoped long-horizon E4 phase claim; the original gate remains recorded as failed.
  provenance:
    run_commit: c869df8b203f13eb8389d1d300b33f1928502871
    source_mode: exact_reconstructed_git_commit_object
    parent_commit: 5b4671780c09d434d6482a71eada9422f885f10f
    reconstructed_commit_sha_matched_github_full_sha: true
    git_worktree_clean_at_launch: true
    git_worktree_clean_at_exit: true
    provenance_compromised: false
    origin_main_ls_remote_authoritative: false
    origin_main_resolution_error: container DNS could not resolve github.com
    device: cpu
    cuda_available: false
  result_summary:
    alpha_0_75_reward_mean: 0.9206409364938736
    alpha_0_75_displacement_mean: 0.566690868139267
    alpha_0_75_expected_state_count: 15
    alpha_1_00_reward_mean: 0.9982822149991989
    alpha_1_00_displacement_mean: 1.02889803647995
    alpha_1_00_expected_state_count: 16
    alpha_1_25_reward_mean: 0.6388135522603988
    alpha_1_25_displacement_mean: 2.012682521343231
    alpha_1_25_expected_state_count: 15
    explicit_opposite_count: 0
    scientific_acceptance_all_passed: false
  next_gate:
    state: ready_after_repository_closure
    unblocked_experiment: C-U1-E4-TAPER-01
    reason: The user explicitly closed the scoped E4 long-horizon claim after reviewing the unchanged 4000-step evidence.
      The original 18/20 gate remains failed and no threshold, horizon, optimizer, seed, or classification was changed.
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
    blocking_reason: resolved_by_v61_protocol_freeze_and_implemented_runner
    depends_on_delivered_experiment: C-U1-E4-TAPER-01
    protocol_freeze: v61_near_retention_matching
    reason: The near region, retention levels, family formulas, development/formal seed firewall, deterministic coefficient
      calibration, runner, metrics, and terminal audit are frozen. Formal execution is active but has not started; Budget-Match
      remains blocked until this result is delivered.
  environment: C-U1
  name: taper_family_near_negative_retention_matched_comparison
  status: not_run
  scientific_status: not_run
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
    formal_run_started: false
    run_started: false
    raw_complete: false
    terminal_audited: false
    package_created: false
  next_gate:
    experiment_id: C-U1-E4-TAPER-BUDGET-MATCH-01
    state: blocked_until_near_retention_delivered
    automatic_activation_forbidden: true
- id: C-U1-E4-TAPER-BUDGET-MATCH-01
  execution_gate:
    state: blocked
    blocked_by:
    - C-U1-E4-TAPER-NEAR-RETENTION-01_delivered
    - frozen_negative_budget_definition
    - frozen_global_alpha_control
    - frozen_optimizer_level_matching_rule
    - separately_implemented_formal_runner
    blocking_reason: This experiment starts only after the near-retention result is delivered and the exact stepwise or cumulative
      negative-update budget is frozen.
  environment: C-U1
  name: taper_family_negative_update_budget_matched_comparison
  status: not_run
  scientific_status: not_run
  parent_experiment: C-U1-E4-TAPER-01
  predecessor: C-U1-E4-TAPER-NEAR-RETENTION-01
  registration_base_commit: 22161a91c0863278765b0d604ea82401d481b5aa
  claim: At matched total negative-update budget, test whether distance-selective tapering improves the allocation of update
    mass between useful near and harmful far negatives relative to reciprocal-linear and Global alpha.
  role: controlled_selectivity_vs_global_scale_validation
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
  budget_contract:
    required_controls:
    - global_alpha
    - reciprocal_linear
    - reciprocal_quadratic
    - vanishing_tail_candidate
    candidate_matching_modes:
    - stepwise_negative_gradient_norm
    - cumulative_negative_optimizer_update
    exact_primary_mode: pending_protocol_freeze
    same_initialization_and_minibatch_stream: required
    only_selective_distance_allocation_may_differ: true
  primary_metrics:
  - matched_total_negative_update_budget
  - near_budget_fraction
  - far_budget_fraction
  - far_harmful_influence
  - held_out_context_reward
  - sigma_and_support_trajectory
  - task_performance_collapse_event
  - support_or_variance_boundary_event
  - nan_inf_numerical_event
  no_method_winner_assumed: true
  evidence:
    run_started: false
    raw_complete: false
    terminal_audited: false
    package_created: false
- id: C-U1-E4-TAPER-CONV-01
  execution_gate:
    state: blocked
    blocked_by:
    - C-U1-E4-TAPER-NEAR-RETENTION-01_delivered
    - C-U1-E4-TAPER-BUDGET-MATCH-01_delivered
    - frozen_method_shortlist_and_hyperparameters
    - frozen_long_horizon_and_terminal_windows
    - separately_implemented_formal_runner
    blocking_reason: Long-run execution is intentionally deferred until the fairness studies freeze the method formulas and
      hyperparameters. The original 8000-step experiment may not be extended in place.
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
    optimizer: original_Adam_dynamics
    optimizer_state_continuity: required
    fixed_horizon_extension_without_registration: forbidden
    provisional_candidate_rule: pending_protocol_freeze
    complete_two_times_continuation: required
    full_batch_stationary_audit: optional_separate_registered_diagnostic_not_a_replacement
    separate_failure_reporting:
    - task_performance_collapse
    - support_or_variance_boundary
    - nan_inf_numerical_failure
  no_method_winner_assumed: true
  evidence:
    run_started: false
    raw_complete: false
    terminal_audited: false
    package_created: false
- id: C-U1-E4-TAPER-CONFIRM-01
  execution_gate:
    state: blocked
    blocked_by:
    - C-U1-E4-TAPER-CONV-01_delivered
    - frozen_untouched_confirmatory_seeds
    - frozen_primary_claim_and_analysis_plan
    - frozen_all_method_formulas_and_hyperparameters
    - separately_implemented_formal_runner
    blocking_reason: Confirmation is last. It uses untouched seeds only after all formulas, coefficients, terminal rules,
      and primary metrics are frozen, with no result-dependent retuning.
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
    seeds_70_89: development_evidence_not_confirmatory
    exact_untouched_seeds: pending_protocol_freeze_before_any_access
    hyperparameter_retuning_after_confirmation_start: forbidden
    primary_claim_change_after_confirmation_start: forbidden
    registered_terminal_audit: required
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
