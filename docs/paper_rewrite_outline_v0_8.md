# DRPO 论文重写 v0.8：Canonical 完整大纲、修正账本与下一阶段施工清单

**状态：已否定并由 v0.7 恢复版本替代；仅保留为 reverse-alignment 错误的历史 provenance。**
**作用：历史审计材料，不再是 active manuscript contract；`docs/handoff.md` 仍是研究状态唯一权威 Master。**
**迁移基线：GitHub `main` commit `3738d09c6cf912ecb85b751fe313e4c79e5974e9`。**
**本次性质：文档与研究规划沉淀，不构成任何新实验结果。**
**Introduction 段落级施工图：** `docs/paper_rewrite_intro_blueprint_v0_2.md`。
**历史版本：** v0.7 与 v0.1 文件保留，仅作 provenance，不再作为 active manuscript contract。

---

## 0. 为什么需要这份文件

本文件从本次完整会话中沉淀三类内容：

1. 已审阅通过的论文 v0.7 完整结构；
2. 从 v0.1 到 v0.7 反复讨论后锁定的纠错与禁区；
3. 写正文、证明、绘图、引用审计和新增 Online 实验之前必须执行的任务。

它的首要目的不是再造一个研究 Master，而是防止后续写作重新犯以下错误：脱离原 DRPO 历史、扩大未经证明的 claim、凭空创造效用函数或安全半径、把 fixed-advantage 实验条件冒充普适 RL 假设、混淆 Gaussian 与 categorical 的失稳形式、滥用 OOD/support collapse、预设候选方法排名，以及用过量双栏图挤占八页正文。

---

# 1. 标题与总体定位

## 1.1 推荐标题

**Breaking the Curse of Repulsion: Distributionally Robust Policy Optimization for Off-Policy Learning**

保留 `Breaking the Curse of Repulsion` 的理由：

- 与原 DRPO arXiv 论文和方法名称保持连续；
- `repulsion` 精确指向全文核心机制；
- 辨识度高于普通的 stable/off-policy learning 标题。

必须加的边界：

- `curse` 指 fixed/stale data 被持续复用时未经控制的 far-field repulsion；
- 不表示所有负优势天然有害；
- 受控负更新可以提供 bad-mode suppression、boundary information，并突破 Positive-only 的性能上限。

## 1.2 一句话主张

Negative-advantage updates are not intrinsically harmful: controlled repulsion can move a policy beyond the positive-only performance limit, whereas repeated optimization of fixed or stale far-field negatives can eliminate finite equilibria and induce divergence. DRPO controls this transition through exponential distance- or probability-aware weighting.

## 1.3 与原论文的继承关系

新版不是另写一篇无关论文，而是把原稿“负更新导致 collapse”的半条理论补全为：

> Positive-only performance limit → stable extrapolation → approach to the feasible boundary → no finite equilibrium / divergence.

保留并重写：

- Repulsive optimization / Repulsive Dynamics；
- Gaussian score geometry；
- phantom-gradient 诊断；
- near/far intervention；
- Positive-only 对照；
- Optimistic DRO 与 DRPO 名称的正式联系。

必须修正或降级：

- 无条件 “negative advantages inevitably cause exponential explosion”；
- expected Fisher SPD 推出固定样本联合参数所有方向扩张；
- “Both mean and variance expand”；
- hard filtering 是唯一生存条件；
- 自建模拟器足以证明通用 offline RL 优越性；
- placeholder / hallucinated references；
- 推荐系统专属主叙事。

---

# 2. 全文叙事边界

## 2.1 单步几何与长期 off-policy 失稳

不需要长篇解释审稿人已知的 off-policy 定义。正文只保留一句：

> Far-field score geometry is defined per sample, while persistent amplification arises when stale or offline samples remain in the training distribution as the policy moves.

含义：给定同一个 `(s,a,A)`，单次梯度由 advantage 与当前 score 决定；真正形成长期自增强的是旧样本在 learner 改变后仍被保留和重复使用。

## 2.2 Objective stationarity 与 off-policy mismatch 必须分开

禁止写：

> The more off-policy the setting is, the more fixed the objective becomes.

正确表述：

- staleness/reuse 决定 data-policy mismatch 是否持续；
- stationarity 决定 empirical actor objective 是否固定；
- fully offline dataset + frozen advantage labels + fixed sample coefficients 时，objective 在 actor 训练期间完全 stationary；
- replay buffer 或 stale actors 可能高度 off-policy，但 buffer 组成与权重仍可变化，因此静态理论最多是局部近似。

## 2.3 理论条件与实验控制的不同角色

主定理需要在分析窗口内使用 stationary empirical actor objective，才能形成 autonomous dynamical system。该条件应作为定理前的数学 Assumption，而不是 Preliminaries 中高调宣传的 RL setting。

C-U1/D-U1 的 fixed advantage 则属于实验控制：

- 在训练前计算并冻结 advantage；
- 隔离 critic error、advantage relabeling 与 policy-dependent weighting；
- 证明只靠固定负信号与变化的 score geometry 已足以产生稳定外推、漂移或失稳。

D4RL 当前 actor pipeline 同样在训练 actor 前冻结 critic-derived advantage；Countdown 固定 verifier labels 和 rollout bank。因此当前论文没有实证支持 raw `p_t/q_t` 随 actor 更新迁移。

## 2.4 不再展开动态 critic / moving equilibrium

删除正文和附录中的 moving-equilibrium 推导。原因：该理论没有当前实验对应，且最初动机来自对 pipeline 的误判。

Limitations 只写一句：

> The equilibrium analysis assumes a stationary empirical actor objective; global stability under jointly evolving actors, critics, or online data distributions is left for future work.

---

# 3. 术语与符号纪律

## 3.1 优先复用 RL / optimization 常用术语

| 不再作为核心术语 | 改用 |
|---|---|
| fixed signed update measure | stationary empirical actor objective |
| policy-relative remoteness | Gaussian: standardized/Mahalanobis distance or negative log-density; categorical: negative log-probability / surprisal |
| negative influence | negative-gradient magnitude / weighted negative update |
| Repulsive Dynamics Trichotomy | Equilibrium and Divergence of Repulsive Policy Updates |
| probability-boundary dynamics | probabilities saturate toward the simplex boundary |
| Far-Field Influence Suppression | Bounded Far-Field Gradient under Exponential Weighting |

允许保留的核心术语：

- Repulsive Dynamics；
- far field / far-field negative samples；
- attraction / repulsion；
- stable extrapolation。

`far field` 首次出现时直接定义：continuous policy 下具有较大 standardized distance 的动作，或 categorical policy 下当前概率很低的动作。

## 3.2 Exponential family 只是证明工具

正文只需说明 regular minimal exponential family 包含 Gaussian 与 categorical policy。只引入不可避免的：

- natural parameter `η`；
- sufficient statistic `T(a)`；
- log-partition function `ψ(η)`。

必须映射回已有的 `μ, Σ` 与 logits，不大篇幅宣传“指数族”本身，也不制造第二套平行变量体系。

## 3.3 结果与失败事件的固定口径

C-U1 当前只能称：

- same-distribution held-out-context generalization；
- unseen-context generalization；
- generalization to unseen states。

不得称 OOD，除非另行登记并执行 state-distribution shift protocol。

始终分开报告：

1. task-performance collapse；
2. covariance / probability / support boundary event；
3. NaN/Inf numerical failure。

概率趋近 0 或 1 本身不自动等于 harmful collapse；只有压制任务相关动作且任务指标下降时才使用 harmful probability collapse。

---

# 4. 论文完整版大纲 v0.8

## 推荐标题

**Breaking the Curse of Repulsion: Distributionally Robust Policy Optimization for Off-Policy Learning**

---

## 1. Introduction — 约 2.25 columns

Introduction 的唯一段落合同如下。施工图必须使用完全相同的 Paragraph ID、标题与顺序，只能展开，不能自行拆分、合并、增删或重排。

<!-- MANUSCRIPT:BEGIN INTRO-P01 -->
## [INTRO-P01] Background and Motivation

说明 policy optimization 为什么同时利用正、负 advantage，以及该问题为何同时关联 offline RL、replay-based learning、asynchronous policy optimization 和 verifier-guided language-model training。本段只建立研究背景，不提前引入 far field、平衡定理或 DRPO。
<!-- MANUSCRIPT:END INTRO-P01 -->

<!-- MANUSCRIPT:BEGIN INTRO-P02 -->
## [INTRO-P02] Far-Field Negative-Gradient Mechanism

提出全文核心异常：固定 negative advantage 不意味着固定 negative gradient，因为当前 policy 改变时，同一历史动作的 score 也会改变。首次定义 far-field negative action：Gaussian policy 下具有较大 standardized distance，或 categorical policy 下具有很低当前概率的负动作。Gaussian 中，反复使用的负动作可能形成 distance–score feedback，使后续负梯度幅度增大；categorical direct-logit score 有界，因此只主张持续概率压制，不主张同类无界梯度放大。
<!-- MANUSCRIPT:END INTRO-P02 -->

<!-- MANUSCRIPT:BEGIN INTRO-P03 -->
## [INTRO-P03] Persistence under Off-Policy Data Reuse

解释单步几何为何在 off-policy learning 中成为长期问题：负动作进入 far field 后仍被保留和重复优化。Fully offline、固定数据与冻结 advantage 的 actor stage 是理论精确覆盖的 stationary empirical objective；replay buffer、stale behavior policy 和 frozen rollout bank 是更广泛的数据复用场景，objective 未必全局固定，但历史样本可以在有限窗口内持续。明确理论使用 fully offline case 是为了精确分析，不表示论文只研究 fully offline；Online/replay、D4RL 与 Countdown 承担更广泛 off-policy 外部验证。
<!-- MANUSCRIPT:END INTRO-P03 -->

<!-- MANUSCRIPT:BEGIN INTRO-P04 -->
## [INTRO-P04] Existing Controls and the Remaining Gap

用两到三句提纲挈领地概括已有负更新控制：删除或统一缩放、clipping/trust-region、以及按低概率、staleness 或 data quality 选择性加权。先承认其稳定化价值，再指出共同缺口：这些方法没有同时解释固定负动作为何可能随 policy 远离而增强后续梯度、局部负更新为何仍可能有益，以及聚合动力学何时失去有限平衡。具体方法、引用与逐项技术差异全部留给 Related Work。
<!-- MANUSCRIPT:END INTRO-P04 -->

<!-- MANUSCRIPT:BEGIN INTRO-P05 -->
## [INTRO-P05] Why Negative Updates Cannot Simply Be Removed

解释为何不能把稳定化等同于 Positive-only。Positive-only 将 policy 拉向已观察正行为，但正行为未必覆盖真正最优区域；适度负更新可能从另一侧推动策略越过 positive-only solution，形成 stable extrapolation。因此本文目标不是消灭所有负梯度，而是保留可能有用的局部负更新，同时控制被反复复用的 far-field negative gradients。不得声称所有 near-field negatives 必然有用或 Positive-only 普遍较差。
<!-- MANUSCRIPT:END INTRO-P05 -->

<!-- MANUSCRIPT:BEGIN INTRO-P06 -->
## [INTRO-P06] Equilibrium and Divergence of Repulsive Policy Updates

说明 Repulsive Dynamics 相比单样本 distance–score mechanism 新增什么：在 stationary empirical actor objective 下，给出聚合正负更新的有限平衡位置、相对于 positive-only solution 的位移、平衡存在条件、mean-parameter feasible-set boundary、no-finite-equilibrium 条件，以及离散更新的 spectral-radius 稳定条件。理论不能退化成“负更新小则好、大则坏”，也不能把数学边界直接等同于任务崩溃。Gaussian 与 categorical 共享平衡结构，但 failure mode 必须分开表述。
<!-- MANUSCRIPT:END INTRO-P06 -->

<!-- MANUSCRIPT:BEGIN INTRO-P07 -->
## [INTRO-P07] DRPO

提出最终方法 DRPO：依据 Gaussian standardized distance / calibrated negative log-density 或 categorical surprisal，对负更新施加 exponential weighting。Exp 是控制 weighted far-field gradient 的优化包络，不是负样本 utility 的指数模型，不假设真实安全半径，也不预设对 Global、Linear 或 Hard baseline 的方法排名。方法目标是削弱 distance–score feedback，同时避免删除全部负反馈。
<!-- MANUSCRIPT:END INTRO-P07 -->

<!-- MANUSCRIPT:BEGIN INTRO-P08 -->
## [INTRO-P08] Evidence Chain and Contributions

按职责而非环境堆叠组织证据：产品流形/解析实验回答远场大梯度来源；C-U1 非线性 Gaussian 环境回答是否传导为 drift、boundary event 与 task-performance degradation；D-U1 回答 categorical persistent suppression；near/far 与 common/rare interventions 闭合因果链；Online/replay 检验刷新与 stale reuse；D4RL 和 Countdown 提供外部有效性。最后总结四项贡献：far-field mechanism 与 equilibrium theory、受控因果识别、Exp DRPO、以及带终态审计的外部验证。不得提前写未运行实验的结果或排名。
<!-- MANUSCRIPT:END INTRO-P08 -->

---

## 2. Related Work — 约 0.75 column

正文最多三段，不设主文对比表：

1. negative-advantage control；
2. low-probability and stale-policy updates；
3. learning from failures and robust data selection。

必须承认 Positive-only、负梯度控制和低概率 token 风险已有前序工作。Novelty 不能写成 “first to observe negative gradients are harmful”。可防守差异是：stable extrapolation → boundary → divergence 的完整动力学、continuous/categorical 统一、badness-distance 精确隔离、定点因果干预与 distance/probability-aware control。

完整 related-work comparison table 放附录。

---

## 3. Preliminaries and Problem Setup — 约 1.0 column

本章只定义一般问题，不设置 `Fixed-advantage mechanism setting` 独立小节。

### 3.1 Policy-gradient objective

\[
\mathbf F(\theta)=\mathbb E_{(s,a,\widehat A)\sim\nu}
[\widehat A(s,a)\nabla_\theta\log\pi_\theta(a\mid s)].
\]

### 3.2 Positive and negative updates

\[
A^+=\max(\widehat A,0),\qquad A^-=\max(-\widehat A,0),
\]

\[
\mathbf F(\theta)=\mathbf F^+(\theta)-\mathbf F^-(\theta),
\]

\[
p=\mathbb E_\nu[w^+A^+],\qquad q=\mathbb E_\nu[w^-A^-].
\]

`p,q` 表示当前 empirical actor objective 中正、负 advantage 的总权重；在当前 fixed-advantage actor 训练中是常量。

### 3.3 Policy-relative distance and surprisal

- Gaussian：standardized distance、Mahalanobis distance、negative log-density；
- categorical：negative log-probability / surprisal。

统一符号 `r_θ(s,a)` 仅在方法公式中作为 family-specific 非负 statistic，不作为新术语宣传。

---

## 4. Repulsive Dynamics — 约 3.75 columns

### 4.1 Far-field score geometry

\[
\|g_i^-\|=A_i^-\|\nabla_\theta\log\pi_\theta(a_i\mid s_i)\|.
\]

Advantage magnitude 与 score geometry 必须分开；相同 advantage 的动作仍可因 distance/probability 不同产生完全不同的梯度幅度。

### 4.2 Exponential-family formulation

\[
\pi_{\boldsymbol\eta}(a)=h(a)\exp\{\boldsymbol\eta^\top\mathbf T(a)-\psi(\boldsymbol\eta)\},
\]

\[
\mathbf F(\boldsymbol\eta)=p\mathbf m_+-q\mathbf m_--(p-q)\nabla\psi(\boldsymbol\eta).
\]

#### Assumption 1：Stationary empirical actor objective

分析窗口内：

- empirical dataset 固定；
- advantage labels 固定；
- 基础样本系数固定；
- 不重新 rollout 或重新估计 advantage。

该条件在 fully offline actor optimization with frozen advantages 中精确成立，在 changing replay/stale buffer 中仅为局部近似，不覆盖 jointly evolving actor–critic、online collection 或 policy-dependent importance weights 的全局动力学。

### 4.3 Theorem 1：Equilibrium and Divergence of Repulsive Policy Updates

#### Case I：Finite interior equilibrium

若 `p>q`，且

\[
\mathbf m^\star=\frac{p\mathbf m_+-q\mathbf m_-}{p-q}
\]

位于 mean-parameter space 内部，则存在唯一有限平衡：

\[
\nabla\psi(\boldsymbol\eta^\star)=\mathbf m^\star.
\]

Jacobian：

\[
\mathbf J_F(\boldsymbol\eta^\star)=-(p-q)\nabla^2\psi(\boldsymbol\eta^\star).
\]

离散 transition matrix：

\[
\mathbf M=\mathbf I-\beta(p-q)\nabla^2\psi(\boldsymbol\eta^\star),
\]

局部稳定条件：

\[
\rho(\mathbf M)<1.
\]

#### Case II：Approach to the feasible-set boundary

`p>q` 但 signed target 位于 mean-parameter space 边界时，不存在有限自然参数，只能通过参数趋无穷或退化分布逼近。该事件不自动等于任务性能崩溃。

#### Case III：No finite equilibrium

- `p=q` 且 `m_+ != m_-`：log-partition 恢复项消失，形成非零常量漂移；必须保留 `m_+=m_-` 的抵消例外。
- `p<q`：恢复方向反转，在适当正则条件下存在 expansion direction，并满足对应 spectral-radius 失稳条件。

正式证明必须检查存在性、唯一性、自然参数域、离散步长和边界条件。

### 4.4 Gaussian policy corollary

固定 covariance：

\[
\boldsymbol\mu^\star=\frac{p\boldsymbol\mu_+-q\boldsymbol\mu_-}{p-q}.
\]

- `q=0`：Positive-only performance limit；
- `0<q<p`：finite stable extrapolation；
- `q→p^-`：equilibrium 远移；
- `q>=p`：finite mean equilibrium disappears。

Learnable covariance 必须同时满足 mean feasibility、second-moment feasibility 和 positive covariance。远场负样本通常表现为 mean repulsion + covariance contraction；不得恢复 “mean and variance both expand”。Gaussian score 可随 standardized distance 无界增长。

### 4.5 Categorical policy corollary

Mean-parameter space 为 probability simplex：

- interior target 对应 finite logits；
- boundary target 对应概率趋 0/1；
- direct-logit score 有界；
- repeated negative updates 仍可持续扩大 logit gap，使 probabilities saturate toward the simplex boundary。

### 4.6 Interpretation of existing methods

只用一到两段：

- Positive-only：`q=0`；
- Global scaling：统一减小负权重；
- AWR：non-negative weighted regression；
- clipping：限制单步 update，不保证 finite equilibrium；
- hard filtering：删除部分负样本及 moment；
- linear/Exp：按 distance/probability 选择性减小负更新。

### Figure 2：Equilibrium and Divergence Regimes

单栏，可在理论足够清楚时进一步缩小或移附录。横轴为 effective negative-update strength，显示 finite equilibrium、boundary approach、no finite equilibrium。底部只注 Gaussian unbounded score growth 与 categorical bounded but persistent logit updates。

---

## 5. Distributionally Robust Policy Optimization — 约 1.75 columns

### 5.1 Optimistic distributional formulation

只保留 DRPO 命名所需的正式基础：density-ratio ambiguity set、high-return sub-distribution 与 hard filtering 作为原 stated Optimistic-DRO objective 的 exact endpoint。完整 CVaR/DRO 推导移附录。

禁止把 soft Exp 写成原 Optimistic-DRO 问题的精确解，除非后续另有正式推导。

### 5.2 Exponential DRPO

最终论文主方法冻结为 Exp，正文统一称 DRPO，表格可写 `DRPO (Exp)`。

\[
w_i^-=\exp(-\lambda r_i),
\]

\[
\mathbf F_{\mathrm{DRPO}}=\mathbb E[A^+\nabla\log\pi-A^-e^{-\lambda r}\nabla\log\pi].
\]

Family-specific `r`：

- Gaussian：standardized distance 或 calibrated negative log-density；
- categorical：negative log-probability；
- sequence：normalized completion/token NLL。

Normalization/clipping 只用于尺度校准，不解释为真实安全半径或效用边界。

### 5.3 Why exponential weighting?

不假设负样本信息效用随距离指数衰减，也不定义凭空存在的局部决策区域 `r_0`。

若：

\[
\|\nabla\log\pi\|\le C(1+r)^k,
\]

则：

\[
A^-e^{-\lambda r}\|\nabla\log\pi\|\le A^-Ce^{-\lambda r}(1+r)^k,
\]

且：

\[
\lim_{r\to\infty}e^{-\lambda r}(1+r)^k=0.
\]

#### Proposition 2：Bounded Far-Field Gradient under Exponential Weighting

Exponential weighting dominates finite-order score growth and drives the weighted far-field negative gradient toward zero。

### 5.4 Relation to baselines

- Uncontrolled：`w^-=1`；
- Positive-only：`w^-=0`；
- Global：`w^-=α`；
- Linear：`w^-=max(0,1-λr)`；
- Hard：threshold 外为 0；
- DRPO：`w^-=exp(-λr)`。

SBRC 和 Hybrid 退出最终论文候选。Hybrid 的历史含义为 Exp selective taper × global stability-budget coefficient；只保留 provenance，不进入正文方法族。

---

## 6. Experiments — 约 5.5 columns

### 6.1 Experimental setup

- Product-manifold：只回答大梯度来源；
- nonlinear Gaussian causal environment：只回答远场大梯度是否传导为 drift/collapse；
- C-U1/D-U1：controlled mechanism and ground truth；
- Online、D4RL、Countdown：external validity / scope boundary。

Fixed advantage 只在这里说明：C-U1/D-U1 冻结 labels 以排除 critic error、relabeling 和 policy-dependent weighting，不把它包装为一般 RL setting。

### 6.2 RQ1：Where Do Large Negative Gradients Come From?

Continuous 固定 advantage/reward/state/sample count，只改变 standardized distance；categorical 固定 context/advantage/action semantics/count，只改变 action probability。报告 score norm、full-parameter gradient、far/near ratio 和 probability change。

#### Figure 3：Source Isolation

单栏 2×2：

- Gaussian score vs distance；
- Gaussian full gradient vs distance；
- categorical score vs action probability；
- categorical full gradient / probability change。

数据：controlled real data ready。

### 6.3 RQ2：When Are Negative Updates Useful or Harmful?

Continuous：Positive-only、weak negative、stable extrapolation、critical drift、covariance boundary、runaway。
Categorical：Positive-only、controlled negative、excessive negative、semantic-shuffled control。

#### Figure 4：Effect of Negative-Update Strength

单栏上下 2×1：continuous 与 categorical。主轴只承载 performance/useful-to-harmful transition；covariance、entropy、bad-action probability 和 terminal slope 移附录。

### 6.4 RQ3：Are Far-Field Negative Gradients the Causal Path?

Continuous：Uncontrolled、Near-zero、Far-zero、Far-cap、Global equal-budget、Far-to-near。
Categorical：Uncontrolled、common-negative、rare-negative、global equal-budget、rare-to-common。

#### Table 1：Controlled Mechanism Results

分 continuous/categorical 两个 block，严格分报 task-performance collapse、boundary event 和 NaN/Inf。

### 6.5 RQ4：How Does Online Exploration Affect Negative Gradients?

新增实验已获概念批准，但必须在下一更新包登记 protocol、seeds、thresholds、stopping/terminal audit 后才能运行。

探索强度 low/medium/high-or-heavy-tail；方法 Uncontrolled/Global/Linear/DRPO；保留 fixed-offline reference。必须测 standardized distance 和实际 score，不能把普通 Gaussian variance 直接等同于有效 far-field risk。

#### Figure 5：Online Exploration and Negative-Gradient Magnitude

单栏，数值当前 `TBD`，禁止填写预期结果。

### 6.6 RQ5：Does DRPO Improve D4RL Performance?

优先完整 9 个 locomotion datasets：halfcheetah/hopper/walker2d × medium/medium-replay/medium-expert；若后续资源不足，范围变化必须另行冻结。

方法：BC、CQL、IQL、Positive-only、Uncontrolled、Global、Linear、DRPO。

主指标：normalized return、mean±SE、best/terminal checkpoint、task-performance collapse rate。

当前 actor 阶段 frozen advantages，因此 raw `p,q` 只做附录数据统计。正文动态诊断优先：

\[
\|\mathbf F_t^-\|/\|\mathbf F_t^+\|,
\]

average negative distance、near/far gradient ratio、mean Exp weight、weighted negative-gradient magnitude、terminal drift。

#### Table 2：D4RL Normalized Return

宽表，dataset 为行、method 为列；best bold、second underline；reported/reproduced baseline 分开。当前数值 `TBD`。

### 6.7 RQ6：Does DRPO Improve Countdown Policy Learning?

定位：**controlled high-staleness off-policy stress test**。

现实动机：asynchronous rollout generation、stale behavior policies、expensive verifier-labeled trajectory reuse、historical reasoning replay。不得声称所有 LLM RL 完全 offline。

构造：

1. frozen SFT policy generates rollouts；
2. verifier assigns labels；
3. fixed mixed-quality rollout bank；
4. current learner repeatedly optimizes the bank。

DRPO 不替代 SFT；SFT 是 initialization / Positive-only reference，DRPO 是 frozen mixed-quality bank 上的 off-policy policy-improvement stage。

方法：SFT initialization、Uncontrolled、Positive-only、Global、Linear、DRPO。
3B main arena，7B top-method confirmation；same adapter/bank/seeds。

#### Table 3：Countdown Results

Greedy、pass@k、valid、best、terminal、collapse；当前数值 `TBD`。

---

## 7. Discussion, Limitations, and Conclusion — 约 1.0 column

### Scope

- 理论针对 stationary empirical actor objective；
- changing replay composition、online collection、joint actor–critic 只可局部诊断，不给全局保证；
- 受控主导路径不等于现实任务唯一失败原因。

### What DRPO controls

DRPO 针对 distance/low-probability-induced growth of negative gradients，不声称解决所有 critic error、distribution shift、exploration failure 或所有 negative-gradient imbalance。

### Utility versus gradient magnitude

不假设负样本效用指数衰减。Exp 只控制 weighted gradient magnitude；方向价值随距离如何变化仍是独立研究问题。

### Continuous versus categorical

共同：positive/negative competition、finite equilibrium、boundary approach、persistent updates。
不同：Gaussian score 可无界；categorical direct-logit score 有界但更新持续。

### Conclusion

1. 负优势可以产生有益稳定外推；
2. repeated optimization of fixed/stale far-field negatives can eliminate finite equilibria；
3. DRPO 使用 exponential distance/probability-aware weighting 控制 far-field negative gradients。

---

# 5. Main-paper visual plan

| Item | Layout | Core role | Current data status |
|---|---|---|---|
| Figure 1 | double-column, horizontal 3 panels | full-paper teaser | concept frozen; academic redraw pending |
| Figure 2 | single-column | equilibrium/boundary/divergence regimes | theorem proof pending |
| Figure 3 | single-column 2×2 | source isolation | real controlled data ready |
| Figure 4 | single-column 2×1 | useful-to-harmful transition | controlled data largely ready |
| Figure 5 | single-column | Online scope experiment | TBD; protocol not yet registered |
| Table 1 | compact controlled table | causal results | real data ready |
| Table 2 | wide D4RL table | normalized-return comparison | TBD |
| Table 3 | wide Countdown table | sequence task comparison | TBD |

原则：除 Figure 1 外，不默认使用双栏图。双栏只在信息结构确实横向且单栏不可读时使用，不因“看起来更重要”而占双栏。

## 5.1 视觉规范

- 论文双栏版式、serif 字体；
- panel label `(a)`, `(b)`；
- booktabs 表格，无竖线；
- 主方法实线，reference/uncontrolled 虚线，其他 baseline 点划线；
- 必须支持黑白打印，不只依赖颜色；
- 每张图最多一个主结论；
- mean curve 配 CI；
- best 与 terminal 分报；
- 结果图和表只允许真实数字或 `TBD`，禁止预期数字。

---

# 6. 八页正文预算

| Section | Columns |
|---|---:|
| Introduction | 2.25 |
| Related Work | 0.75 |
| Preliminaries | 1.00 |
| Repulsive Dynamics | 3.75 |
| DRPO | 1.75 |
| Experiments | 5.50 |
| Discussion + Conclusion | 1.00 |
| **Total** | **16.00** |

如果超页，优先压缩/移附录：Figure 2、Figure 5、Table 1 的非核心列、Gaussian/categorical 额外诊断。D4RL 与 Countdown 两张任务主表原则上保留。

---

# 7. 会话纠错账本：后续不得重复犯错

## 7.1 结构和叙事

- 不把 Related Work 放到论文末尾；最多半页到 0.75 column，只做定位。
- 不把论文写成与原 DRPO 无关的新工作；保留标题、Repulsive Dynamics 和 DRPO 名称连续性。
- Introduction 不能一上来罗列结论；必须按背景—问题—已有解法—局限—本文认识—方法—贡献展开。
- 不按 continuous/categorical 分成两套重复实验章节；同一 RQ 下 paired reporting，但 family-specific 现象不强求完全对称。
- 不把 recommendation 作为本轮主实验；主文关注 controlled、Online、D4RL、Countdown。

## 7.2 理论

- 不以单样本 sign-only theorem 作为全文主线；单样本 identity 只可作为 proof device/appendix。
- 不用 expected Fisher 的 SPD 证明 fixed off-policy sample 在联合参数空间所有方向 expansion。
- 不声称负优势必然指数爆炸；区分有限平衡、边界与无有限平衡。
- 不创造 mass–scale–coherence 作为正文核心概念；directional coherence 最多放附录/实验解释。
- repeated reuse 不是单步梯度的第四个乘法因子，而是让相同/相近 gradient field 跨时间持续存在。
- 不把 support/probability boundary 自动等同于 reward collapse。
- 不展开 neural-policy Jacobian 作为独立正文小节；本论文没有相应指标和全局深网定理。

## 7.3 Exp 方法

- 不定义未经证明的局部决策区域或真实安全半径 `r_0`。
- 不假设负样本 utility 随距离指数衰减。
- 不从 raw `p/q` 的时间变化推导 Exp；当前 pipeline 的 raw advantage mass 基本固定。
- Exp 的正式理论动机是 exponential control envelope 支配有限阶 score growth。
- 不宣称 Exp 必然优于 Global/Linear/Hard；结果必须由正式实验决定。
- SBRC 与 Hybrid 已退出最终论文候选，但不得破坏性删除历史实验/代码记录。

## 7.4 实验和 claim

- Product manifold 只回答来源；nonlinear Gaussian 只回答传导；C-U1/D-U1 是受控机制；D4RL/Countdown 是外部有效性。
- Fixed advantage 是机制控制，不是所有 RL 的普适现实假设。
- Online 实验尚未登记，不得直接开跑。
- D4RL/Countdown 缺结果时只做布局与 `TBD`，不得画预期数值。
- Countdown 是 controlled high-staleness off-policy stress test，不是 SFT 替代方案，也不代表所有 LLM RL 都 offline。
- Smoke/static/pilot 不得写成正式结果；稳态、collapse、排名必须 terminal audit。

## 7.5 术语和引用

- 复用 RL 常见术语，能不用新名词就不用。
- Positive-only 不是本文发明，必须引用前序工作。
- 不写 “first to discover negative gradients are harmful”。
- 删除 `Lastname et al.` 等 placeholder；每个 citation 与论断必须逐条核验 primary source。
- 不使用 OOD 描述当前 C-U1 held-out contexts。

---

# 8. 下一阶段唯一工作清单

1. **Formal theorem proof**：Assumptions、existence/uniqueness、三种 case、continuous/discrete dynamics。
2. **Symbol audit**：`p,q,η,T,ψ,β/r/λ` 冲突、首次定义、Gaussian/categorical 映射。
3. **Spectral-radius audit**：Jacobian 符号、离散步长、`p=q` 抵消例外、`p<q` expansion 条件。
4. **Gaussian corollary**：fixed covariance 与 learnable covariance 的 second-moment feasibility。
5. **Categorical corollary**：simplex interior/boundary、bounded direct-logit score、persistent logit-gap growth。
6. **Exp proposition**：Bounded Far-Field Gradient under Exponential Weighting；明确 advantage 矩条件和 family-specific `r`。
7. **Citation audit**：原稿、AWR/IQL、Positive-only、BAPO/low-probability token、TOPR、failure learning、stale-policy 等逐条 primary-source 核验；自动扫描 placeholder。
8. **Visual specification**：统一字体、线型、CI、caption、单/双栏尺寸和黑白可读性。
9. **Figure 1 academic redraw**：保留三阶段故事，明确 uncontrolled harm 与 DRPO recovery。
10. **Real-data figures**：使用已有 controlled raw data 生成 Figure 3、Figure 4 和 Table 1；不画预期数据。
11. **Online registration**：下一更新包登记 experiment ID、claim、environment、exploration/tail protocol、methods、seeds、metrics、stopping、terminal audit 和 outputs。
12. **Paragraph-level construction plan**：Introduction v0.1 已落库到 `docs/paper_rewrite_intro_blueprint_v0_1.md`；其余章节继续按相同模板记录问题、topic sentence、supporting argument、公式/图表、citation、字数、claim 强度、证据状态和禁止表述。
13. **正文写作顺序**：施工图通过后，先写 Introduction、Related Work、Preliminaries、Theory；D4RL/Countdown/Online 结果段保留正式占位。
14. **External results**：后续执行并回填 D4RL Table 2、Countdown Table 3、Online Figure 5；不得以当前 planning 状态冒充结果。

---

# 9. Paragraph-level construction plan 模板

每个正文段落必须登记：

| Field | Required content |
|---|---|
| Section / paragraph ID | 唯一编号 |
| Question answered | 本段解决什么读者问题 |
| Topic sentence | 第一主句 |
| Supporting logic | 2–4 个核心论点 |
| Evidence | theorem / controlled result / external result / citation |
| Equation / visual | 公式、Figure、Table 引用 |
| Citations | 已核验 primary sources |
| Target words / lines | 与 column 预算一致 |
| Claim strength | observation / supported / proven / limitation |
| Evidence status | ready / conditional / TBD |
| Forbidden overclaim | 本段不得出现的升级表述 |

施工图完成前，不进入大段自由写作。

---

# 10. 附录计划

A. Notation and assumptions
B. Full proof of Equilibrium and Divergence of Repulsive Policy Updates
C. Discrete transition-matrix and spectral-radius conditions
D. Gaussian covariance feasibility
E. Categorical simplex-boundary analysis
F. Bounded Far-Field Gradient under Exponential Weighting
G. Original Optimistic-DRO and CVaR derivation
H. DRPO algorithm and distance/probability calibration
I. C-U1 protocols and terminal audits
J. D-U1 protocols and terminal audits
K. Online experiment protocol
L. Complete D4RL results and critic details
M. Countdown construction and checkpoint audit
N. Related-work comparison table
O. Citation and reproducibility audit

不再包含 moving-equilibrium / dynamic-critic 全套理论附录。
