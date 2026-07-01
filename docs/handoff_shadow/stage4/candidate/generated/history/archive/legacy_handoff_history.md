# Immutable legacy handoff history

> Stage 4B lossless source-promotion shadow candidate.
> Generated from `docs/handoff.md`; do not edit while the manual handoff remains authoritative.

- Owner type: `history_record`
- Owner ID: `legacy_handoff_history`
- Responsibility: Preserve byte-exact historical source payloads outside promoted module ownership.
- Dependencies: none
- Content-contract topics: none
- Owned source blocks: 161
- Registry references are pointers only; `experiments/registry.yaml` remains the sole editable registry source.

## Registry references

- `experiments`: none
- `development_experiment_registrations`: none

## Owned source blocks

<!-- STAGE4B-SOURCE-BLOCK:B000120:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000120:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000121:START -->
# 文档结构

- 10\. 从推荐扩展到通用 off-policy policy optimization

- 11\. 连续—离散统一的 Repulsive Surprisal Dynamics

- 12\. 为什么负梯度并不总是有害：局部泛化与远场失稳

- 13\. WAPO、STARE、Mu-GRPO、ASymPO、TOPR 等工作的机制对照

- 14\. 下一阶段理论与实验增强计划

- 1\. 执行摘要与最终判断

- 2\. 理论命题与需要闭合的因果链

<!-- STAGE4B-SOURCE-BLOCK:B000121:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000122:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000122:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000123:START -->
# 0. 新会话交接：必须继承的共识与阅读顺序

本节是新会话的最小可靠上下文。后续讨论应先继承本节，再继续设计实验；不要把方法实现细节、探索性消融和已经锁定的机制结论混为一谈。

<!-- STAGE4B-SOURCE-BLOCK:B000123:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000124:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000124:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000125:START -->
## 0.2 已锁定、未锁定与后续可变结论

| **层级** | **结论**                                                                                                       | **状态**                                    |
|----------|----------------------------------------------------------------------------------------------------------------|---------------------------------------------|
| 已锁定   | 在 badness 与 distance 严格解耦时，远场仍产生数量级更大的负梯度；主体来自 score geometry，而不是远场样本更差。 | 不得因后续方法实验而推翻                    |
| 已锁定   | 在当前非线性 Gaussian 环境中，远场异常负梯度是 OOD 漂移与 collapse 的主要自然传导路径。                        | Near-zero 无效；Far-zero/Far-cap 20/20 救援 |
| 高度支持 | 负梯度是直接中介，远场几何是异常幅度的自然生成机制；全局 α 与距离控制都可稳定。                                | 方法优劣仍待泛化任务决定                    |
| 尚未锁定 | 随训练步数严格指数增长、Distance 必然优于 α、所有真实任务仅由该机制崩溃。                                      | 必须继续验证，不能提前宣称                  |

<!-- STAGE4B-SOURCE-BLOCK:B000125:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000126:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000126:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000127:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000127:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000128:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000128:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000129:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000129:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000130:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000130:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000131:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000131:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000132:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000132:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000133:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000133:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000134:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000134:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000135:START -->
# 3. 实验一：badness–distance 严格解耦下的梯度来源分解

<!-- STAGE4B-SOURCE-BLOCK:B000135:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000136:START -->
## 3.1 乘积流形构造

机制环境采用乘积空间 \[0,R\] × S¹。样本质量和 advantage 只依赖角度
θ，径向距离只依赖 r；同一套 θ/advantage
向量被原样复制到所有半径上。因此这不是“Pearson 相关接近
0”，而是数据生成上的结构独立：

p(\|A\| \| r = r₁) = p(\|A\| \| r = r₂), ∀ r₁,r₂

在负样本子集内同样成立，因为负样本 mask 只由 θ 决定。训练改变 policy
score，但不改变 reward、baseline 或 advantage 的跨半径分布。

<!-- STAGE4B-SOURCE-BLOCK:B000136:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000137:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000137:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000138:START -->
# 4. 实验二：非线性 Gaussian actor 中的因果干预

<!-- STAGE4B-SOURCE-BLOCK:B000138:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000139:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000139:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000140:START -->
## 4.2 五类关键干预

| **干预**     | **操作**                                                | **要回答的问题**                 |
|--------------|---------------------------------------------------------|----------------------------------|
| Baseline     | 所有负梯度乘统一 α=0.1                                  | 自然训练是否崩溃                 |
| Near-zero    | 删除当前近场负梯度，保留远场                            | 近场是否是主要致因               |
| Far-zero     | 删除当前远场负梯度，保留近场                            | 切断远场路径能否救援             |
| Far-cap      | 保留远场方向，仅截断异常幅度                            | 是否无需删除样本，只去除放大即可 |
| Global-scale | 统一缩放全部负梯度，使总 norm 与 Far-cap 相同           | 异常梯度幅度是否为中介           |
| Far-to-near  | 截断远场后，把预算人为转移给近场并恢复 baseline 总 norm | 巨大负更新本身是否也有害         |

<!-- STAGE4B-SOURCE-BLOCK:B000140:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000141:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000141:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000142:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000142:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000143:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000143:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000144:START -->
## 4.6 时序中介与 OOD 漂移

Baseline 的中位时序为：约第 50 步径向漂移 \|μᵣ\|\>0.5，约第 70 步
\|μᵣ\|\>1，约第 80 步 reward 保持率低于 45%。远场梯度异常在 step 0
即已存在，先于漂移和崩溃。

<img src="/mnt/data/master_recovery/media/media/image3.png"
style="width:6.45in;height:3.94167in" />

**图 3　Baseline 中的传导链：异常远场负梯度先出现，随后径向漂移并伴随
reward collapse。**

<!-- STAGE4B-SOURCE-BLOCK:B000144:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000145:START -->
## 4.7 稳健性与实现验证

- 阈值稳健性：标准化距离阈值 1.5、2.5、3.0 下，Far-zero 与 Far-cap 均为
  0/5 崩溃；Near-zero 均为 5/5 崩溃。

- 开发集稳定区间：α=0.075 和 0.1 均表现出 Baseline/Near-zero
  崩溃、Far-zero 稳定；α=0.05 属于较弱崩溃区。

- 梯度实现：逐样本全参数梯度聚合与标准 autograd 的相对误差为 2.75×10⁻⁸。

- 预算匹配：Global-scale 与 Far-cap 的 post-negative-gradient norm
  完全相同；Far-to-near 与 Baseline 的匹配误差低于 2×10⁻⁷。

<!-- STAGE4B-SOURCE-BLOCK:B000145:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000146:START -->
# 5. 证据强度、结论边界与仍未证明的内容

<!-- STAGE4B-SOURCE-BLOCK:B000146:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000147:START -->
## 5.1 当前可以强声明的内容

- 在质量严重程度与策略距离严格解耦时，远场仍会自然生成异常大的负梯度。

- 异常大梯度的主体来自 policy-score geometry，而不是远场样本平均更差。

- 在当前非线性 Gaussian offline
  环境中，保留远场而删除近场不能阻止崩溃；只处理远场即可阻止崩溃。

- 大梯度幅度是直接中介；统一 α/全局缩放可通过降低该幅度稳定训练。

- 因此“远场几何是异常负梯度的自然来源；异常负梯度是崩溃的直接传导量”在本环境内已经具有较强因果证据。

<!-- STAGE4B-SOURCE-BLOCK:B000147:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000148:START -->
## 5.2 仍不可直接声称的内容

- 严格指数律：目前证明了显著放大和动态自增强，但未完成随训练步数的指数、线性、二次模型比较。

- 普遍唯一性：该机制是充分且主导的受控反例/机制环境，但真实 D4RL、LLM RL
  或推荐任务还可能存在 critic error、support mismatch、entropy collapse
  等其他机制。

- Distance 必然优于 α：本实验的 Global-scale
  表现很好，说明统一缩放也能控制中介变量；二者在保留近场负信息、泛化与跨任务鲁棒性上的优劣仍需验证。

- DRPO 的 DRO 最优性：今天的实验支撑 repulsion/collapse 理论，不单独验证
  hard filtering 是某一现实任务中的唯一最优解。

<!-- STAGE4B-SOURCE-BLOCK:B000148:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000149:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000149:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000150:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000150:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000151:START -->
## 6.2 对原论文 claim 的增强与修正

| **原叙事风险**          | **建议升级后的叙事**                                                                                  |
|-------------------------|-------------------------------------------------------------------------------------------------------|
| 低质量数据导致 collapse | 负优势 severity 与 policy remoteness 共同决定 repulsive influence；远场可在质量独立时单独制造异常梯度 |
| 负梯度会爆炸            | 展示 score-level 解析结构、25×/56×实测分解与动态传导                                                  |
| DRPO 过滤坏样本         | 强调其切断 divergence-inducing repulsive path，而不只是在做数据清洗                                   |
| 硬过滤优于软权重        | 谨慎写成特定 DRO 下的解；经验上进一步比较 α、Distance 与 joint influence                              |

<!-- STAGE4B-SOURCE-BLOCK:B000151:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000152:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000152:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000153:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000153:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000154:START -->
# 8. 建议写入论文的核心 claims、图表与段落结构

<!-- STAGE4B-SOURCE-BLOCK:B000154:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000155:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000155:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000156:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000156:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000157:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000157:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000158:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000158:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000159:START -->
# 10. 从推荐扩展到通用 Off-Policy Policy Optimization

当前 DRPO 的理论对象不应再被限制为推荐系统中的 generative
policy。推荐只是一个重要应用；真正的研究对象是：当固定或陈旧数据被当前策略重复优化时，负优势更新如何形成
repulsive dynamics，以及如何在保留有效负信息的同时避免远场失稳。

建议将论文主问题重写为“通用 off-policy policy optimization
中的排斥诅咒”，并把推荐实验保留为真实应用验证之一。理论和机制实验覆盖连续
Gaussian policy；下一阶段补充 categorical/softmax policy
后，可自然延展到 LLM RLVR、离散控制、diffusion/flow policy extraction
和其他生成式决策模型。

<!-- STAGE4B-SOURCE-BLOCK:B000159:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000160:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000160:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000161:START -->
# 11. 连续—离散统一的 Repulsive Surprisal Dynamics

<!-- STAGE4B-SOURCE-BLOCK:B000161:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000162:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000162:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000163:START -->
## 11.2 连续 Gaussian：梯度幅度自增强

对固定方差的 Gaussian mean，surprisal 的远场主项满足 D≈‖a−μ‖²/(2σ²)，而
mean-score norm² 满足 ‖∇μ logπ‖²≈2D/σ²，因此 κ(D) 随 D
线性增长。若同时学习 log σ，方差 score 的远场主项近似随 D
增长，其平方可达到 O(D²)。因此连续场景可能出现真正的 gradient-amplitude
runaway：距离越大，下一步排斥越强，进而继续增加距离。

<!-- STAGE4B-SOURCE-BLOCK:B000163:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000164:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000164:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000165:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000165:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000166:START -->
# 12. 为什么负梯度并不总是有害：局部泛化与远场失稳

论文需要同时回答两个看似矛盾的事实：负优势更新能够造成
collapse，但大量实证又表明，完全删除负样本可能损失性能、数据效率、探索与
OOD 泛化。最有解释力的理论不是“负梯度有害或无害”的二分法，而是负梯度存在
informativeness–amplification trade-off。

<!-- STAGE4B-SOURCE-BLOCK:B000166:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000167:START -->
## 12.1 正优势提供 attraction，但存在 imitation ceiling

正优势更新把策略拉向已观察到的成功行为，类似 advantage-filtered
imitation。它能够快速学习已知好模式，但当 πθ 已贴近这些样本时，score 与
attraction
逐渐减弱；同时，正样本本身并不告诉模型哪些相邻模式是错误的，也难以直接消除数据中未覆盖的坏
mode。

<!-- STAGE4B-SOURCE-BLOCK:B000167:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000168:START -->
## 12.2 受控负优势提供 boundary shaping 与 mode suppression

负优势并不是凭空把模型“推向正确的未知领域”。它首先降低已知坏行为的概率，并把释放出的概率质量按照模型已有的表示几何与先验重新分配到其他候选。若负样本仍位于当前策略的局部支持附近，它通常具有三类价值：

- 边界信息：区分“好样本附近哪些方向不能走”，形成比单纯模仿更大的决策
  margin。

- 坏 mode 抑制：在高精度、长时程和多模态任务中，仅提高好 mode
  并不必然消除竞争性的坏 mode。

- 组合与探索：释放的概率质量由预训练先验和共享表示重新分配，可能激活训练集中未直接展示、但模型已经潜在掌握的替代路径。

因此，“超越数据集”应被严谨表述为：负更新可以通过排除已知错误、重整概率质量和组合已有行为片段，使策略在未直接示范的区域获得更好表现；它并不保证创造训练前完全不存在的新能力。

<!-- STAGE4B-SOURCE-BLOCK:B000168:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000169:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000169:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000170:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000170:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000171:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000171:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000172:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000172:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000173:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000173:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000174:START -->
## 13.1 WAPO 的真正含义

WAPO 并不是“选择性保留有益负梯度”的方法。它的 peak–valley taxonomy
说明负更新的局部作用依赖当前 token 分布：Neg-peak 倾向提高
entropy，Neg-valley 倾向降低
entropy；两者都可能在不同条件下失稳。论文明确承认部分负优势 token
可能含有有效信号，但在 coarse sequence reward
下难以可靠选择，因此采用最保守的 coarse filter：只更新正优势
completion。

<!-- STAGE4B-SOURCE-BLOCK:B000174:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000175:START -->
## 13.2 STARE 比 WAPO 更接近“打压危险部分、保留其余”

STARE 不删除全部负梯度。它在 entropy 低于目标时，使用 batch 内 surprisal
quantile 找出高-surprisal token：增强正优势高-surprisal
token，减弱负优势高-surprisal token；低-surprisal负 token
保持原始权重，entropy 恢复后则关闭干预并退回
GRPO。因此它已经隐含了“负梯度并非一律有害”，但判断标准服务于 entropy
regulation，而不是远场 repulsive dynamics。

<!-- STAGE4B-SOURCE-BLOCK:B000175:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000176:START -->
## 13.3 Mu-GRPO 已经触及重复排斥，但尚未统一

Mu-GRPO 的 diagnosis 很重要：高 staleness 下，负优势 trajectory 一旦某个
prefix 跨越当前 policy 的支持边界，后续 suffix 仍可获得显著更新，形成
localized instability。它仅 veto 触发点后的负 suffix，并通过 relaxed
clipping 保留触发前的 stale learning
signal。这是目前最接近“局部负信号有用、off-support
负信号有害”的离散工作之一，应作为重点竞争与支持文献。

<!-- STAGE4B-SOURCE-BLOCK:B000176:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000177:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000177:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000178:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000178:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000179:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000179:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000180:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000180:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000181:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000181:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000182:START -->
# 14. 下一阶段理论与实验增强计划

<!-- STAGE4B-SOURCE-BLOCK:B000182:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000183:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000183:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000184:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000184:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000185:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000185:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000186:START -->
## 14.4 P2：小型 Transformer 序列实验

- 固定 context、token identity、advantage、sequence length，只通过受控
  logit bias 或 stale checkpoint 改变 learner-relative surprisal。

- 测量单 token 与全参数梯度、跨 token interference、direction coherence
  和概率随重复负更新的动力学。

- 复刻 near/far 定点干预：仅压 rare-negative 与仅压 common-negative。

- 再扩展到真实数学 RLVR，报告 advantage × surprisal 二维分桶、staleness
  轨迹与 collapse onset。

<!-- STAGE4B-SOURCE-BLOCK:B000186:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000187:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000187:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000188:START -->
## 14.6 更新后的实验优先级（v4）

| **优先级** | **实验**                             | **必须回答的核心问题**                                                              |
|------------|--------------------------------------|-------------------------------------------------------------------------------------|
| P0         | 负梯度稳定外推与泛化实验             | 负梯度能否直接把策略推出正样本支持并改善 OOD；何时由有益外推转为 runaway            |
| P1         | Categorical bandit 严格解耦实验      | 连续距离与离散 surprisal 能否落入统一 repulsive dynamics                            |
| P2         | 小型 Transformer 序列实验            | 共享参数、token interference、staleness 与 support suppression 是否复现 bandit 结论 |
| P3         | D4RL / 机器人 / 推荐 / RLVR 外部验证 | 理论机制在真实耦合数据中的解释力与方法收益                                          |

<!-- STAGE4B-SOURCE-BLOCK:B000188:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000189:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000189:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000190:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000190:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000191:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000191:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000192:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000192:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000193:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000193:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000194:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000194:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000195:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000195:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000196:START -->
# 附录 C　新会话核心文件索引

| **文件**       | **路径**                                                                 | **用途**                                    |
|----------------|--------------------------------------------------------------------------|---------------------------------------------|
| 当前交接文档   | /mnt/data/Far_Field_Negative_Gradient_DRPO_Research_Note_v4_Handoff.docx | 新会话应首先阅读                            |
| 乘积流形代码   | /mnt/data/product_manifold_gradient_decomposition.py                     | badness–distance 严格解耦与 16×→24.95× 分解 |
| 乘积流形结果   | /mnt/data/pmgd_check_1024/near_far_summary.csv                           | 近远场数值、score 与 coherence 分解         |
| 因果干预代码   | /mnt/data/causal_farfield_solid/causal_farfield_intervention.py          | 动态 near/far 干预与预算匹配                |
| 因果结果汇总   | /mnt/data/causal_farfield_solid/summary_with_ci.csv                      | 20-seed CI、崩溃率与方法结果                |
| 完整因果实验包 | /mnt/data/causal_farfield_solid_bundle.zip                               | 代码、曲线、逐 seed 结果与 README           |

<!-- STAGE4B-SOURCE-BLOCK:B000196:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000197:START -->
# 附录 D　P0 可学习方差：闭式稳态与快速实验验证

实验采用 fixed-advantage naive PG，无 value/Q network、importance
sampling 或动态重权。正样本分布均值 0、标准差 1.2；负样本分布均值
−1、标准差 0.2；未见评估最优动作 a\*=1。

<!-- STAGE4B-SOURCE-BLOCK:B000197:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000198:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000198:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000199:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000199:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000200:START -->
## D.3 非线性参数化与控制方法

单状态双输出 MLP 在 3 个初始化下复现相同转变：α≤0.58
跟随解析稳态，α=0.60 时 3/3 seeds 出现方差坍缩。原始 α=0.8
的不受控训练坍缩；将有效 ρ 全局缩放到 0.5，或使用 detached
standardized-distance cap，均恢复至 μ≈1、σ≈0.917、评估
reward≈1。该结果证明 α 可通过降低 qM₋ 恢复联合稳态，distance control
则对二次远场项进行选择性控制。

<!-- STAGE4B-SOURCE-BLOCK:B000200:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000201:START -->
## D.4 原始 gradient-explode 代码诊断

原代码的 good-only 配置只应用正样本更新，坏样本仅用于 phantom gradient
monitor。复现结果显示坏/好总梯度比约 72.3×、log-σ 分支梯度比约
55.5×；但实际 σ 从约 0.606 收缩到约 0.177。打开正负混合训练后，σ
进一步收缩到约 0.008。故原 Figure 2(b)
应解释为远场梯度敏感度扩张，而不是实际 σ 扩张。

<!-- STAGE4B-SOURCE-BLOCK:B000201:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000202:START -->
## D.5 论文写作结论

修正后的主线是：负优势在均值分支产生位置排斥；在方差分支，近场负样本扩大支持、远场负样本收缩支持。远场中的
d↑ 与 σ↓ 共同放大标准化距离和
score，形成比固定方差更强的自增强。完整策略稳定性由联合固定点存在条件与
signed off-policy dynamics Jacobian 决定。

<!-- STAGE4B-SOURCE-BLOCK:B000202:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000203:START -->
# 附录 E　统一非线性 benchmark：论文级正式结果

统一 benchmark 的目的不是把不同识别问题混成一个实验，而是让三个 protocol
共用同一策略类、环境接口、训练循环、梯度诊断和统计管线。正式代码已使用
seeds 10–29 完成来源与因果实验，并对 P0 稳定外推及可学习方差相变进行
held-out 检验。

<!-- STAGE4B-SOURCE-BLOCK:B000203:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000204:START -->
## E.1 Protocol A：严格来源隔离

| **阶段**            | **\|A\| 远/近** | **score 远/近（95% CI）** | **单样本梯度** | **聚合梯度** |
|---------------------|-----------------|---------------------------|----------------|--------------|
| initialization      | 1.000           | 45.13 \[43.30,46.95\]     | 47.78          | 61.56        |
| positive_pretrained | 1.000           | 38.02 \[37.11,38.96\]     | 38.64          | 82.08        |

解释：advantage magnitude 在所有半径上结构相同，统一 actor
中远场梯度仍出现数量级放大。聚合比高于单样本比，说明方向一致性进一步放大净更新。

<!-- STAGE4B-SOURCE-BLOCK:B000204:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000205:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000205:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000206:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000206:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000207:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000207:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000208:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000208:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000230:START -->
# Part IV. v10 Hopper Learned-Critic 外部验证记录（完整保留，状态降级）

> 以下结果保留为有限训练步数的 learned-critic mechanism probe。600 optimization steps 未达到长期收敛，不能作为最终动力学或方法结论。

<!-- STAGE4B-SOURCE-BLOCK:B000230:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000231:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000231:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000232:START -->
# 0. 新会话必须继承的锁定结论

| **【锁定】乘积流形/来源隔离回答“大梯度从哪里来”；非线性因果干预回答“这些梯度是否导致漂移与崩溃”。两类问题不能混淆。** |
|-----------------------------------------------------------------------------------------------------------------------|

<!-- STAGE4B-SOURCE-BLOCK:B000232:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000233:START -->
## 0.1 连续 Gaussian

- 远场异常负梯度的主要来源是 policy-score geometry，而不是远场样本 advantage 更差。

- 在连续非线性因果环境中，far-zero / far-cap 稳定救援，near-zero 无效；远场异常负梯度是受控环境中 OOD 漂移与 collapse 的主导传导路径。

- Positive-only 是无排斥参考：稳定但停在最佳正样本支持附近，存在 imitation ceiling。

- 适度负梯度能够形成数据支持之外的有益稳态；更强负梯度先造成稳定过度外推，再造成动力学失稳。

- 可学习方差引入更早的联合稳定边界：远场负样本导致支持收缩，precision 进一步放大均值与方差梯度。

- Global α 与 distance-aware control 都是有效稳定机制；不能宣称 distance 在所有任务上必然优于 α。

<!-- STAGE4B-SOURCE-BLOCK:B000233:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000234:START -->
## 0.2 离散 categorical

- 直接 softmax logit 的单样本 score norm 有界；离散通用病理是 surprisal 持续增长与支持集/熵坍缩，而不是连续 Gaussian 同形式的无界幅度爆炸。

- 负优势对 entropy 的影响由当前概率/rarity 决定：打压高概率负动作最初提高熵，打压低概率负动作降低熵。

- 在具有有序 action catalogue 与参数共享的 categorical energy policy 中，负梯度同样可突破 positive-only 支持上限；任意独立 logits 不自动具备这一性质。

- 离散 signed-moment 可行边界 α≈0.585；α=0.58 稳定，α=0.62 在 20/20 seeds 中 temperature collapse。

- 保留 far negatives、删除 near negatives 仍在 20/20 seeds 中造成 task + support collapse；far-zero / far-cap 在 20/20 seeds 中同时阻止两类 collapse。

<!-- STAGE4B-SOURCE-BLOCK:B000234:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000235:START -->
## 0.3 明确不可声称

- 远场机制是所有真实任务的唯一 collapse 原因。

- distance control 普遍优于 global α。

- 当前 held-out state 实验等于组合泛化。

- categorical energy policy 的结论可无条件推广到任意独立 softmax logits。

- 当前结果已经验证 Transformer / LLM token dynamics。

<!-- STAGE4B-SOURCE-BLOCK:B000235:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000236:START -->
# 1. 文档审计与结论固化

<!-- STAGE4B-SOURCE-BLOCK:B000236:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000237:START -->
## 1.1 v7 的主要问题

- 内容按研究时间顺序堆叠：旧路线、开发实验、正式实验和文献笔记并列，核心证据被淹没。

- C1/C2/V0 等小环境与统一 benchmark 同时保留为主结果，造成数值重复和 claim 层级不清。

- 原 DRPO 的 expected-Fisher SPD 证明、方差“扩张”表述与修正后的联合动力学并存，存在内部冲突。

- “下一阶段计划”重复多版，包含已经完成的 P0 与 categorical bandit。

<!-- STAGE4B-SOURCE-BLOCK:B000237:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000238:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000238:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000239:START -->
# 2. 大一统理论：Repulsive Signed-Moment Dynamics

<!-- STAGE4B-SOURCE-BLOCK:B000239:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000240:START -->
## 2.1 研究对象、记号与结论层级

对每个状态 s 条件化后，把正优势和负优势样本分别写成加权分布 P₊(a\|s)、P₋(a\|s)。令正质量 p(s)=E\[A₊\|s\]，负质量 q(s)=E\[(-A)₊\|s\]；全局 α、样本权重或方法控制均被吸收到 q 和 P₋ 中。基础理论先假设 actor step 内 advantage stop-gradient，随后再讨论 value/Q 随时间变化。

$$
J(\theta)=\mathbb{E}_{\mathcal D}[A(s,a)\log\pi_\theta(a\mid s)],\qquad F(\theta)=\nabla_\theta J(\theta)=\mathbb{E}_{\mathcal D}[A\nabla_\theta\log\pi_\theta(a\mid s)]
$$

理论分成三层：第一层是任意可微策略都成立的单样本 surprisal 递推；第二层是在正则最小指数族中成立的 signed-moment 平衡定理；第三层才是 Gaussian、categorical、神经网络与具体控制方法的分叉推论。这样既保留 general 形式，也避免把 expected Fisher 当成固定样本动力学。

<!-- STAGE4B-SOURCE-BLOCK:B000240:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000241:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000241:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000242:START -->
## 2.3 批量更新：自项、跨样本干涉与方向一致性

单样本单调性不能无条件提升为“batch 中每个负样本 surprisal 都单调增加”。令 batch field F=ΣⱼAⱼgⱼ，则样本 i 的首阶变化为：


$$
\Delta S_i=-h g_i^\top F+O(h^2)=h|A_i|\lVert g_i\rVert^2-h\sum_{j\ne i}A_j\langle g_i,g_j\rangle+O(h^2)
$$


第一项是负样本自身的确定性排斥；第二项是正负样本共享参数带来的 interference。远场风险因此不仅取决于单样本 scale，还取决于梯度方向是否相干。本文实验中的 aggregate amplification 正是在单样本 score 放大之外叠加了 coherence。


$$
\text{Repulsive influence}\approx\text{negative mass}\times\text{score scale}\times\text{directional coherence}\times\text{repeated reuse}
$$


<!-- STAGE4B-SOURCE-BLOCK:B000242:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000243:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000243:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000244:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000244:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000245:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000245:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000246:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000246:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000247:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000247:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000248:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000248:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000249:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000249:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000250:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000250:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000251:START -->
## 2.12 Learned critic / value network：瞬时适用与移动目标

在 DRPO-Q、IQL 或一般 actor-critic 中，A_t=Qφ(s,a)−Vψ(s) 会随 critic 更新。只要 actor step 使用 A_t.detach()，上述理论对每一步的瞬时 signed field 仍成立；但整个系统变成非自治动力学，不能把固定 advantage 的全局固定点直接照搬。

若每一时刻都存在内部目标 η\*(t)，局部收缩率下界为 m\>0，且目标漂移速度 ‖η̇\*(t)‖≤v，则标准移动平衡分析给出 tracking error 的量级：


$$
\limsup_t\lVert\eta(t)-\eta^*(t)\rVert\le\frac{v}{m}
$$


因此 critic 越慢、稳定裕度越大，actor 越能跟踪；但任何梯度控制都不能修复 critic 给错 advantage 符号的问题，只能限制错误信号被 score geometry 放大的破坏。

<!-- STAGE4B-SOURCE-BLOCK:B000251:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000252:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000252:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000253:START -->
# 3. 连续统一 benchmark：正式论文级证据

<!-- STAGE4B-SOURCE-BLOCK:B000253:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000254:START -->
## 3.1 Protocol A：来源隔离

| **阶段**            | **\|A\| far/near** | **score** | **单样本梯度** | **聚合梯度** |
|---------------------|--------------------|-----------|----------------|--------------|
| initialization      | 1.000              | 45.13×    | 47.78×         | 61.56×       |
| positive_pretrained | 1.000              | 38.02×    | 38.64×         | 82.08×       |

advantage 与 quality coordinate 沿半径严格复制；独立性检查在 20/20 seeds 中误差为 0。远场放大来自 score geometry，方向一致性进一步放大聚合梯度。

<img src="media/image1.png" style="width:5.9in;height:3.79382in" />

*图 1　统一连续 benchmark 中正样本预训练后的远场/近场梯度分解。*

<!-- STAGE4B-SOURCE-BLOCK:B000254:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000255:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000255:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000256:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000256:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000257:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000257:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000258:START -->
# 4. 离散 categorical benchmark：理论与正式结果

<!-- STAGE4B-SOURCE-BLOCK:B000258:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000259:START -->
## 4.1 小环境：direct softmax 的精确结论

| **负动作初态**            | **p₀** | **p_T**  | **H₀** | **H峰值** | **H_T**  | **max score** |
|---------------------------|--------|----------|--------|-----------|----------|---------------|
| high_probability_negative | 0.8991 | 4.06e-12 | 0.386  | 0.906     | 6.72e-06 | 1.414213      |
| low_probability_negative  | 0.0038 | 1.90e-20 | 0.292  | 0.292     | 4.51e-09 | 1.414214      |

高概率负动作被抑制时，entropy 先上升后下降；低概率负动作被抑制时，entropy 从一开始就下降。两者 surprisal 都持续增加，score norm 均不超过 √2。

<img src="media/image6.png" style="width:5.8in;height:3.92903in" />

*图 6　离散负更新的 entropy 方向取决于当前动作概率。*

<!-- STAGE4B-SOURCE-BLOCK:B000259:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000260:START -->
## 4.2 来源隔离：rarity 使梯度更大但幅度有界

| **阶段**            | **advantage** | **分布参数 score** | **全参数梯度** | **surprisal** |
|---------------------|---------------|--------------------|----------------|---------------|
| initialization      | 1.000         | 2.61×              | 2.79×          | 1.55×         |
| positive_pretrained | 1.000         | 3.30×              | 3.65×          | 1.66×         |

离散 far/near 放大约 3–4×，明显小于连续 Gaussian 的 38–82×，与有限 catalogue 和有界 direct-logit score 一致。

<img src="media/image7.png" style="width:5.8in;height:3.80625in" />

*图 7　categorical rarity-source isolation。*

<!-- STAGE4B-SOURCE-BLOCK:B000260:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000261:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000261:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000262:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000262:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000263:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000263:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000264:START -->
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


<!-- STAGE4B-SOURCE-BLOCK:B000264:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000265:START -->
# 6. 论文贡献与推荐 claim 结构

<!-- STAGE4B-SOURCE-BLOCK:B000265:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000266:START -->
## 6.1 当前已经形成的贡献栈

1.  机制来源：严格解耦 advantage/quality 与 distance/rarity，证明异常负梯度来自 policy-relative geometry。

2.  因果传导：连续与离散均使用 near/far 定点干预闭合 far-field → drift/support collapse → performance failure。

3.  稳定—泛化理论：从 positive-only ceiling 到稳定外推、过度外推、临界失稳。

4.  联合尺度理论：连续方差与离散温度均可由 signed second moment / residual balance 推导稳定边界。

5.  统一解释：global α、positive-only、hard filtering、distance/surprisal control 都可视为 repulsive-gain control 的不同形式。

6.  方法设计依据：不再是先提出 heuristic 再寻找解释，而是从稳定边界反推控制策略。

<!-- STAGE4B-SOURCE-BLOCK:B000266:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000267:START -->
## 6.2 论文主文建议

| **模块**             | **主文内容**                                                                      | **附录内容**                      |
|----------------------|-----------------------------------------------------------------------------------|-----------------------------------|
| Theory               | signed field Jacobian；Gaussian 联合稳态；categorical surprisal 与 signed moments | 完整推导、离散时间步长条件        |
| Controlled benchmark | 连续 A/B/C + categorical direct/phase/causal                                      | 架构稳健性、旧小环境 sanity check |
| Method               | 统一 repulsive-gain control；连续 distance 与离散 surprisal 版本                  | 更多权重函数和超参                |
| External validation  | 至少一个连续真实/标准任务 + 一个 token/序列任务                                   | 额外数据集与消融                  |

<!-- STAGE4B-SOURCE-BLOCK:B000267:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000268:START -->
## 6.3 推荐用语

- “We identify a dominant far-field pathway in controlled off-policy policy-gradient dynamics.”

- “Negative feedback is locally informative but becomes destabilizing as policy-relative rarity grows.”

- “Continuous policies exhibit amplitude/precision amplification; categorical policies exhibit persistent surprisal growth and support collapse.”

- “Distance-aware control outperforms a budget-matched global control in the continuous benchmark, while categorical results show that no universal ranking should be claimed.”

<!-- STAGE4B-SOURCE-BLOCK:B000268:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000269:START -->
# 7. 接下来唯一保留的任务清单

<!-- STAGE4B-SOURCE-BLOCK:B000269:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000270:START -->
## P0：统一理论已完成；转入论文 LaTeX 化与定理精简

- 重写原 DRPO Section 3：删除 sign-only SPD theorem 与 “μ、σ 同时扩张”；改为 signed field Jacobian + 精确 Gaussian/categorical corollaries。

- 把 fixed-advantage assumption、时间重参数化和离散时间步长条件写入 theorem assumptions。

- 将统一 benchmark 的正式表格与图直接迁入论文草稿；旧简单环境只留 appendix provenance。

<!-- STAGE4B-SOURCE-BLOCK:B000270:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000271:START -->
## P1：方法创新与外部有效性（下一实验主线）

**Countdown 模型梯度：**0.5B 仅用于快速调通和超参粗筛；3B 作为主 arena（效率与基础能力平衡）；7B 只对冻结后的前两名方法做最终确认。任务本身不要求 7B，但方法比较要求基线具有足够非零成功率和改进空间。

- 设计统一的 repulsive-gain controller：连续使用 standardized distance / surprisal，离散使用 token/action surprisal；允许全局 α 作为基线与退化特例。

- 在一个连续标准任务（优先 D4RL/推荐环境）验证 stability–generalization trade-off。

- 在一个离散序列任务或小型 Transformer 上验证 rare-negative suppression、entropy collapse 与 selective control。

<!-- STAGE4B-SOURCE-BLOCK:B000271:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000272:START -->
## P2：当前不做或降级为 future work

- 组合泛化。

- 大规模架构笛卡尔积。

- 动态 critic/value feedback。

- 所有相关方法的完整复现。

| **【下一步】连续与 categorical 受控机制实验已完成。除非论文审稿风险明确指出缺口，不再继续堆叠同类 toy robustness。** |
|----------------------------------------------------------------------------------------------------------------------|

<!-- STAGE4B-SOURCE-BLOCK:B000272:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000273:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000273:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000274:START -->
# 附录 A. 旧实验的保留规则

下列实验不再占用正文篇幅，但不得删除源文件：

- C1 标量固定方差：验证闭式均值相变。

- C2 单状态/多状态 MLP：排除直接参数化偶然性。

- V0/V1 可学习方差：发现并验证早期方差边界。

- 原 gradient-explode：证明 phantom gradient growth，但同时暴露“梯度 norm ≠ 参数扩张”的记录歧义。

- 旧 product-manifold 与 causal_farfield：作为统一 benchmark 的独立历史复现与机制 provenance。

论文主文所有数值优先引用统一 formal benchmark；旧结果只用于 appendix、代码审计或 rebuttal。

<!-- STAGE4B-SOURCE-BLOCK:B000274:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000275:START -->
# 附录 B. 最终研究判断

| **【锁定】当前工作已经从“一个负优势加权方法”升级为机制论文：来源隔离、因果传导、稳定—泛化相变、连续—离散统一和方法设计原则均有理论与 formal controlled evidence。方法创新仍然重要，但不再需要单独承担整篇论文的贡献。** |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

最合理的论文结构是：修正后的动力学理论 → 统一连续/离散受控 benchmark → 由理论导出的 repulsive-gain controller → 外部任务验证。

<!-- STAGE4B-SOURCE-BLOCK:B000275:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000276:START -->
# 附录 C. 公式自检与机器验证

v9 理论完成后执行独立数值/自动微分自检：Gaussian 联合固定点残差、解析 Jacobian、方差临界根、固定样本 Hessian 行列式、categorical surprisal 速率、softmax score 上界与 Taylor 首阶误差均通过。可复现脚本：drpo_theory_v9/theory_self_check.py。

---

<!-- STAGE4B-SOURCE-BLOCK:B000276:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000284:START -->
# Part V. Bandit 稳定外推子实验的收敛审计（完整保留）

> 本审计只覆盖有解析参照的稳定外推子实验。它修正了短训练终值，但没有完成 E2、E3、E6、E7 的完整长期审计。

<!-- STAGE4B-SOURCE-BLOCK:B000284:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000285:START -->
# DRPO Bandit Saturation Re-audit (v1)

<!-- STAGE4B-SOURCE-BLOCK:B000285:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000286:START -->
## 结论摘要

这次审计先计算可解析的 ground truth，再把原来的短训练延长到 **5,000–20,000 steps**，关键临界点补到 10,000 steps，并检查末段斜率、梯度范数与 moment error。

1. **原稳定外推实验明显低估了有限稳态的位置与性能。** 主要原因是没有训练到饱和。
2. **相变结构没有消失，反而与解析 ground truth 更一致。** 有限固定点、近临界慢收敛和无固定点 runaway 被清楚分开。
3. 连续可学习方差的 20-seed 解析临界点为 **0.6645 ± 0.0063**，范围 **[0.6475, 0.6751]**。
4. Categorical learnable-temperature 的平均 moment 临界点为 **0.5846 ± 0.0001**；第一个 state 失去正方差的边界约为 **0.5801**。
5. **无序 semantic categorical 的 120-step 方法排名未饱和，正式降级为 pilot。** 不能进入论文最终方法表。

<!-- STAGE4B-SOURCE-BLOCK:B000286:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000287:START -->
## 1. Ground truth

<!-- STAGE4B-SOURCE-BLOCK:B000287:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000288:START -->
### Continuous, fixed variance

对 `p=n=1, a_+=0, a_-=-1`：

```math
\beta^*(\alpha)=\frac{\alpha}{1-\alpha},\qquad \alpha<1.
```

`alpha >= 1` 时没有有限均值固定点。固定方差分支不应使用 signed variance 是否为正来判断均值固定点；旧审计脚本在这一点上有分类 bug，本报告已修正。

<!-- STAGE4B-SOURCE-BLOCK:B000288:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000289:START -->
### Continuous, learnable variance

均值固定点仍为上式；对 diagonal variance，第 `j` 维的 signed residual variance 为

```math
\sigma_j^{2*}=\frac{\tau_+^2-\alpha\tau_-^2}{1-\alpha}-\frac{\alpha}{(1-\alpha)^2}\mathbb E[d_j^2].
```

只有所有维度均为正时，联合均值–方差内部固定点才存在。

<!-- STAGE4B-SOURCE-BLOCK:B000289:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000290:START -->
### Categorical energy policy

固定 temperature 时，有限内部解需要匹配 signed first moment；learnable temperature 时还要匹配 signed second moment。signed variance 失去正性后，有限 temperature 解消失，策略走向 catalogue/simplex 边界。

<!-- STAGE4B-SOURCE-BLOCK:B000290:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000291:START -->
## 2. 重新训练后的关键结果（seeds 10–12）

<!-- STAGE4B-SOURCE-BLOCK:B000291:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000292:START -->
### Continuous fixed variance

| alpha | steps | theory beta* | train beta | test beta | tail slope / step | 判定 |
|---:|---:|---:|---:|---:|---:|---|
| 0.50 | 20,000 | 1.000 | 0.9838 | 0.9776 | 1.07e-06 | 有限稳态，近饱和 |
| 0.75 | 10,000 | 3.000 | 2.9794 | 2.9781 | 3.37e-06 | 有限但严重过外推，近饱和 |
| 0.90 | 10,000 | 9.000 | 8.9476 | 8.9450 | 1.06e-05 | 远场有限稳态，近饱和 |
| 1.00 | 3,000 | none | 98.5 | 97.8 | 3.48e-02 | 持续漂移，无有限固定点 |

原 2,200-step 的 `alpha=0.5` test beta/reward 约为 `0.897/0.837`；重跑后为 **0.978/0.954**。旧绝对数值必须替换。

<!-- STAGE4B-SOURCE-BLOCK:B000292:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000293:START -->
### Continuous learnable variance

| alpha | steps | theory beta* | train beta | sigma_min | theory sigma_min | 判定 |
|---:|---:|---:|---:|---:|---:|---|
| 0.50 | 10,000 | 1.000 | 0.9626 | 1.1874 | 1.1796 | 联合固定点，近饱和 |
| 0.60 | 10,000 | 1.500 | 1.4757 | 0.9158 | 0.9075 | 联合固定点，近饱和 |
| 0.64 | 10,000 | 1.778 | 1.7659 | 0.5957 | 0.5919 | 近临界但有限，接近解析点 |
| 0.65 | 10,000 | 1.857 | 1.8466 | 0.4426 | 0.4390 | 近临界窄方差稳态 |
| 0.68 | <=750 | no joint point | NaN/collapse | NaN | signed variance < 0 | 内部固定点不存在，collapse |

**关键修正：** `alpha=0.65` 在这些 seeds 上不是未解释的失稳，而是慢收敛到窄方差有限固定点；`alpha=0.68` 才因 signed variance 变负而失去内部解。

<!-- STAGE4B-SOURCE-BLOCK:B000293:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000294:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000294:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000295:START -->
## 3. 对旧结论的处理

- **替换绝对数值：** continuous fixed `alpha=0.5`、learnable `alpha=0.5/0.6/0.65` 的 500–2,200-step 终值。
- **改写状态分类：** continuous learnable `alpha=0.65` 从“慢漂移/可能失稳”改为“seed-dependent boundary 内的慢收敛窄方差稳态”。
- **保留并增强：** `alpha=0.68` 无联合固定点并 collapse；categorical `0.58/0.62` 相变；fixed variance `alpha<1` 有有限均值点、`alpha=1` 持续漂移。
- **撤回为 pilot：** unordered semantic categorical 120-step 的 Global/Exp/SBRC 排名与显著性。

<!-- STAGE4B-SOURCE-BLOCK:B000295:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000296:START -->
## 4. 哪些实验不以训练收敛为验收条件

- 乘积流形 source-isolation 回答瞬时梯度从哪里来，不是稳态实验。
- Near/Far causal intervention 回答固定 horizon 内切断路径是否救援；可以作为有限时域因果结果，但不能写成无限时域稳定定理。
- 若要回应长期动力学批评，causal protocol 仍需单独做 horizon extension；本次审计主要修复稳定外推/相变实验。

<!-- STAGE4B-SOURCE-BLOCK:B000296:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000297:START -->
## 5. 现在可防守的 solid 结论

1. Positive-only ceiling 与适度负梯度外推的固定点可以由 ground truth 直接计算，长训练与其一致。
2. 增大负梯度强度产生的是 **有限稳态远移 → 近临界慢收敛 → 内部固定点消失/边界 runaway**，不是负优势一出现就必然发散。
3. 可学习方差/temperature 的可行域边界早于均值边界，并由 signed second moment 的正性决定。
4. 旧短训练低估了稳定外推，没有推翻相变；修正后理论–实验对齐更强。
5. 任何方法排名必须在目标指标末段斜率接近零，或明确到达无固定点边界后再报告。


---

<!-- STAGE4B-SOURCE-BLOCK:B000297:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000413:START -->
# 0. 本次恢复的明确结论

<!-- STAGE4B-SOURCE-BLOCK:B000413:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000414:START -->
## 0.1 “大一统连续环境”当前是否已经实现

**没有。** 当前代码实际上包含两个连续 contextual-bandit 数据生成器：

- Product 环境：用于质量—距离解耦的瞬时梯度来源与 Near/Far 因果干预；
- Extrapolation 环境：用于 positive-only 上限、稳定外推及均值—方差平衡。

二者虽然共享 6 维状态、2 维动作、Gaussian actor 和部分统计代码，但不是同一个环境，也不是同一批状态—动作几何。因此 v7 中“统一 benchmark”的命名高估了完成度。Categorical 因动作空间不同而使用独立环境是合理的。

<!-- STAGE4B-SOURCE-BLOCK:B000414:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000415:START -->
## 0.2 Product 与 Extrapolation 能否真正合并

**能，没有根本技术障碍。** 过去分开的主要原因是：

1. Product 环境追求“同 reward/advantage、只改变策略距离”的严格变量隔离；
2. Extrapolation 环境追求“最佳已观察正样本、未见真实最优动作、反方向负样本”的解析结构，并使用分布期望降低采样噪声；
3. 分开实现更快、更容易获得单项结果。

这属于实现便利和局部变量隔离，不是必须分离的数学限制。此前没有向用户说明并取得同意，是需求执行错误。

<!-- STAGE4B-SOURCE-BLOCK:B000415:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000416:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000416:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000417:START -->
## 0.4 从昨日下午到现在是否“白忙活”

必须分开评价：

- **对于用户明确要求的“把连续小环境真正合并为一个环境并完整重跑”任务：基本没有完成。** 合并的是代码接口和部分 protocol，不是环境；因此不能把此前重跑算作该交付的完成。
- **并非所有工作都没有价值：** Gaussian 方差方向修正、expected-Fisher 证明纠错、exponential-family 理论、categorical 实验、Hopper 探针和训练收敛审计提供了新信息。但这些工作不能替代大一统环境任务，也不能用于掩盖其未完成状态。
- 已在旧分离环境上重复运行的结果仍可作为开发证据和回归基线，但新论文的统一连续主结果必须在真正合并后的环境重新运行。

---

<!-- STAGE4B-SOURCE-BLOCK:B000417:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000418:START -->
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

<!-- STAGE4B-SOURCE-BLOCK:B000418:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000419:START -->
## 11.6 用户审阅后的 E4 变量治理与结论边界修正（2026-06-24）

- **D-U1/E6 暂停。** 按用户要求，在 C-U1 结果、术语和复现代码完成审计前不启动 categorical smoke 或长程训练。
- **β 使用不合规。** E4 报告中的 β 只是 `a_plus -> a_star` 方向的归一化投影位移，不是模型参数；但第 7.4 节已将绘图归一化 β 撤出主变量体系。将其重新作为主表符号违反变量治理规则。后续不新增替代希腊符号，统一使用文字指标“沿隐藏最优方向的归一化投影位移”，代码字段为 `normalized_extrapolation_displacement`。
- **α 的边界。** `alpha_local` 仅为代码配置名，映射到既有核心变量 α；E4 中它乘在方向可靠的局部负梯度组上。论文仍使用 α，并显式说明所作用的负样本子集。
- **状态分类补强。** E4 必须分开：稳定良好固定点、稳定坏固定点、数值有限的持续漂移/runaway、数值/支持边界事件。主判据为 reward、均值相对 `a_plus/a_star` 的原始距离、归一化净动力场残差、位移窗口斜率、更新 norm、sigma/log-sigma 及 2× horizon 是否反转。
- **方差解释修正。** sigma 收缩不是任务 reward 下降的充分条件。它表示完整联合稳态可能更早消失，并放大标准化距离和梯度敏感度；任务是否失效仍由均值所在 reward 区域和正负净梯度平衡决定。E4 数据本身显示 sigma 显著收缩阶段 reward 仍可提高。
- **方向效用降级。** 当前方向诊断由环境几何有意构造，保留为 sanity check/附录解释，不作为跨环境主要结论。一般性方向—距离规律需多几何或外部任务验证。
- **复现性卡点。** Master 引用的统一源代码和多数 raw run 目录在当前会话文件系统中缺失，仅有汇总报告与三张 E4 图。已生成 `C_U1_REPRODUCIBILITY_AUDIT.md`；在恢复原代码或完成严格 reimplementation 重跑前，不宣称已有可下载的精确复现包。


<!-- STAGE4B-SOURCE-BLOCK:B000419:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000420:START -->
# v15 附录：上传环境代码兼容性与一键重建登记

<!-- STAGE4B-SOURCE-BLOCK:B000420:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000421:START -->
## A. 上传 `drpo_unified_continuous_environment_v1 (1).py` 与正式 C-U1 的差异

该文件可作为几何骨架来源，但不是正式 E1-E4 环境，差异包括：

1. 状态分布为 `Uniform[-1,1]`，正式协议为 `N(0,I_6)`；
2. 状态数为 1024 train / 2048 test，正式为 4096 / 4096；
3. 每状态 2 个带高斯噪声正动作，正式为 4 个等 reward、质心精确等于 `a_plus` 的正动作；
4. 负动作数量为 6，正式为 8，且正式 index 0 为 `a_minus`、index 4 为最远动作；
5. reward width / baseline 为 0.80 / 0.50，正式为 0.75 / 0.40；
6. 文件只有环境与 invariant audit，没有共享两层 MLP actor、positive-only 饱和训练、动态标准化 Near/Far、E1-E4 driver、终态审计、逐 seed 轨迹和控制预算匹配。

因此不能原样运行来复现既有结果。

<!-- STAGE4B-SOURCE-BLOCK:B000421:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000422:START -->
## B. 重建代码状态

- 文件：`/mnt/data/drpo_cu1_e1_e4_oneclick.py`
- 默认命令：`python drpo_cu1_e1_e4_oneclick.py`
- 不需要编辑源码或传入超参数；正式配置全部冻结在 `Protocol` dataclass。
- 自动选择 CUDA/CPU，固定结果目录，支持中断后自动跳过已完成 seed。
- 输出包括环境审计、manifest/hash、E1-E4 逐 seed JSON/CSV、逐步轨迹、bootstrap CI、相图、方差边界阈值/学习率稳健性和 reference regression。
- 当前状态：`python -m py_compile` 通过；开发用 CPU smoke 全流程通过；正式 20-seed 结果尚未运行，因此不能提前声称数值已复现。

<!-- STAGE4B-SOURCE-BLOCK:B000422:END -->
<!-- STAGE4B-SOURCE-BLOCK:B000423:START -->
## C. 重新登记的任务失效判据

由于旧瞬时 driver 未持久化，旧 E3 报告中的任务阈值精确值不可审计。重建代码明确预注册：held-out-context reward 低于同 seed positive-only reference 的 45%，且连续 3 个评估点满足，记为任务失效事件。最终结论同时报告连续漂移、终态 reward、2× horizon 和数值/支持状态，不让单一阈值承担主要结论。
<!-- STAGE4B-SOURCE-BLOCK:B000423:END -->
