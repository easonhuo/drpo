# Paper rewrite and presentation plan

> Generated Stage 4 minimal-context shadow module. Do not edit manually.
> `docs/handoff.md` and `experiments/registry.yaml` remain authoritative.

- Module ID: `paper_rewrite`
- Responsibility: Provide the current manuscript outline, claim hierarchy, figures, tables, limitations, and reproducibility plan.
- Content contract topics: none
- Deduplicated overlapping source chunks: 0
- Source hash: `84d4e665db0ac281faf2026664ada2b092e9c492488adbf984ca4ddffd6126d2`

## Source 1: docs/handoff.md: # Part VI. 论文重写大纲（当前草案，后续须与实验状态同步） -> # 0. 本次恢复的明确结论

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
