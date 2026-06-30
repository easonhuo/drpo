# DRPO Introduction 段落级施工图 v0.2

**状态：已否定并由 v0.3 替代；本文件错误地反向驱动了大纲改写，仅保留历史 provenance。**
**上位大纲（历史错误版本）：** `docs/paper_rewrite_outline_v0_8.md`。
**Live hierarchy：** 已移除；当前 active blueprint 由 `docs/manuscript/hierarchy.yaml` 指向 v0.3。
**研究状态唯一权威 Master：** `docs/handoff.md`。
**迁移基线：** GitHub `main` commit `3738d09c6cf912ecb85b751fe313e4c79e5974e9`。
**证据边界：** planning、静态检查、smoke 或 pilot 不得升级为正式实验结果。

## 使用规则

- 本文件的 Paragraph ID、标题、数量和顺序必须与 canonical outline 完全一致。
- 每个 block 的 `Parent-Outline-SHA256` 绑定对应 outline block；上位段落变化后必须重新审阅并更新本段。
- 施工图只能展开 topic sentence、论证、证据、引用、字数和禁区，不得自行拆段、合段、增删、重排或改写段落职责。
- 正文尚未建立，因此 live hierarchy 当前只注册 outline 与 blueprint 两层。

**预计 Introduction 长度：** 950–1100 English words，约 2.0–2.25 双栏 columns。

---
<!-- MANUSCRIPT:BEGIN INTRO-P01 -->
## [INTRO-P01] Background and Motivation
Parent-Outline-SHA256: `6f37aed94cabd014988f8cebfa7ced726216a11652e2162f08285f9f72a9253b`

### 回答的问题
为什么 signed policy optimization 及其中的 negative feedback 值得研究？

### Topic sentence
> Policy optimization improves a policy by both reinforcing actions that outperform a reference value and suppressing actions that underperform it.

### 论证顺序
1. Policy optimization 使用正、负 advantage，而不是只模仿成功动作。
2. 该范式存在于 offline RL、replay-based RL、asynchronous policy learning 和 verifier-guided language-model optimization。
3. 失败行为可能提供 behavior cloning / positive-only fitting 之外的改进信号。
4. 本段以研究背景收束，不提前解释 far-field mechanism。

### 引用目标
- policy-gradient / actor–critic 基础工作；
- advantage-weighted policy learning；
- representative offline / replay / verifier-guided policy learning。

### 目标长度
90–110 words。

### 禁止内容
- negative samples are inherently harmful；
- positive-only methods cannot learn；
- all RL systems are off-policy；
- far field、equilibrium 或 Exp 的提前展开。
<!-- MANUSCRIPT:END INTRO-P01 -->

<!-- MANUSCRIPT:BEGIN INTRO-P02 -->
## [INTRO-P02] Far-Field Negative-Gradient Mechanism
Parent-Outline-SHA256: `d937b1eb154b3267d6b3f50e7ddc94ac091e976229d83e7accd7e7dbb320f2e0`

### 回答的问题
本文最核心、最先需要让审稿人理解的异常是什么？

### Topic sentence
> A fixed negative advantage does not imply a fixed negative gradient, because the score of the same historical action changes as the current policy moves.

### 论证顺序
1. 首次定义 far-field negative action：Gaussian 下 standardized distance 较大；categorical 下当前 action probability 很低。
2. 强调 far field 是相对于 current policy 的动态状态，不是样本永久标签。
3. 对 Gaussian，mean score 随 standardized residual 增长；在反复使用负动作时，policy 被推远可能使该动作的后续 score 与梯度幅度继续增大。
4. 给出最简链条：negative update → larger distance → larger score → potentially stronger subsequent negative gradient。
5. 对 categorical 只说明 direct-logit score 有界但负更新可以持续压低概率，不宣称 Gaussian 式无界放大。

### 证据与公式
- Introduction 只给直觉链条；
- 解析递推和 covariance 细节留在 Section 4；
- 产品流形实验作为来源证据，不在本段展开实验表格。

### 目标长度
120–145 words。

### 禁止内容
- advantage 本身“膨胀”；
- 复杂共享网络下每个负样本必然单调放大；
- categorical gradient explosion；
- safe radius、utility horizon 或新造同义术语。
<!-- MANUSCRIPT:END INTRO-P02 -->

<!-- MANUSCRIPT:BEGIN INTRO-P03 -->
## [INTRO-P03] Persistence under Off-Policy Data Reuse
Parent-Outline-SHA256: `2c5562e7ece52857edcc71bdd2f53d142939375e21ba4b58fb864bf3cc61aed1`

### 回答的问题
单样本几何为何会在 off-policy learning 中积累为长期问题，论文为何又能覆盖 replay？

### Topic sentence
> The feedback becomes persistent when a negative action remains in the training data after the learner has moved away from it.

### 论证顺序
1. Fully offline frozen-data actor stage 中，dataset、advantage labels 和 base weights 固定，empirical actor objective 精确 stationary。
2. Replay buffer / stale policy 下 objective 未必全局 stationary，但旧样本会跨多个 update 被复用，因此 persistence 仍可出现。
3. Fresh on-policy collection 更快刷新历史样本，但不否认单步 far-field gradient。
4. 明确理论选择 stationary case 是为了得到 exact equilibrium result，而不是把 fully offline 当作 off-policy 的定义。
5. Online/replay、D4RL、Countdown 负责验证更广泛 data reuse 与 staleness。

### 必须出现的 scope patch
> We derive exact results for a stationary frozen-data actor objective and evaluate the same persistence mechanism in broader replay-based and stale-policy settings.

### 目标长度
115–140 words。

### 禁止内容
- the more off-policy, the more fixed the objective；
- 论文只研究 fully offline；
- replay 与 fixed offline 完全等价；
- dynamic critic / moving-equilibrium 作为当前主线。
<!-- MANUSCRIPT:END INTRO-P03 -->

<!-- MANUSCRIPT:BEGIN INTRO-P04 -->
## [INTRO-P04] Existing Controls and the Remaining Gap
Parent-Outline-SHA256: `23fe5b2cd0b8baa511068b00a797e69d794ea72e57b077582f1b2ed7a4eb559a`

### 回答的问题
已有研究已经解决了什么，Introduction 与 Related Work 如何避免重复？

### Topic sentence
> Existing methods control harmful negative updates by removing or scaling them, constraining policy changes, or selectively downweighting low-probability, stale, or low-quality data.

### 论证顺序
1. 只用一组概括句覆盖三类控制：remove/global attenuation；clipping/trust region；selective weighting/filtering。
2. 明确认可这些方法提供了有效稳定化工具。
3. 只指出一个共同缺口：尚未联合解释 fixed negative action 的 distance–score feedback、局部负更新价值和有限平衡消失。
4. 立即转入本文问题，不逐项评价方法。

### 与 Related Work 的硬边界
- Introduction：三类概括 + 一句共同缺口；
- Related Work：方法名、primary citations、objective、技术差异和公平比较。

### 目标长度
70–90 words。

### 禁止内容
- 五类以上方法枚举；
- 每类方法逐项批评；
- existing work ignores negative gradients；
- 把本段写成缩略文献综述。
<!-- MANUSCRIPT:END INTRO-P04 -->

<!-- MANUSCRIPT:BEGIN INTRO-P05 -->
## [INTRO-P05] Why Negative Updates Cannot Simply Be Removed
Parent-Outline-SHA256: `5acf4b4bdb07c65fe26a6afa6351f2a6bf71b1e332a60cb2f35c0f8a7c84f329`

### 回答的问题
既然远场负更新危险，为什么不直接删除所有 negative updates？

### Topic sentence
> Eliminating all negative updates avoids repulsion, but can also discard information needed to improve beyond the observed positive behavior.

### 论证顺序
1. Positive-only 主要把 policy 拉向 observed positive behavior。
2. Positive data 未必覆盖真正最优动作或最优区域。
3. 适度负更新可以从另一侧提供排斥，使 policy 越过 positive-only solution，形成 stable extrapolation。
4. 因而方法目标是保留可能有用的 local negative updates，控制被反复复用的 far-field updates。

### 证据边界
- 使用“can / may”，不声称 universally beneficial；
- 该 claim 由受控机制实验和 equilibrium theory 支持，不从定义直接推出任务提升。

### 目标长度
90–115 words。

### 禁止内容
- near-field negatives are always useful；
- Positive-only universally underperforms；
- negative feedback is necessary in every task；
- 把所有近场 mass imbalance 纳入 DRPO 责任。
<!-- MANUSCRIPT:END INTRO-P05 -->

<!-- MANUSCRIPT:BEGIN INTRO-P06 -->
## [INTRO-P06] Equilibrium and Divergence of Repulsive Policy Updates
Parent-Outline-SHA256: `e9338a01357d7540251d907a2771ff5255a4ce1f166e741f1de4c7e5355fb0f7`

### 回答的问题
单样本 gradient mechanism 已经解释了“为什么会变强”，Repulsive Dynamics 还能给出什么非显然结果？

### Topic sentence
> The distance–score mechanism explains how a negative update can strengthen; Repulsive Dynamics characterizes the resulting aggregate equilibrium and when that equilibrium ceases to exist.

### 论证顺序
1. 正、负 weighted updates 聚合成 stationary actor dynamics。
2. 给出 positive-only equilibrium 和加入 controlled repulsion 后的有限平衡位置。
3. 说明负更新可以把 equilibrium 推过 positive-only solution，而不是只说“正负相互竞争”。
4. 给出 finite interior equilibrium、feasible-set boundary 和 no finite equilibrium 三类结果。
5. 给出离散 transition matrix 与 spectral-radius / step-size local stability。
6. 区分 Gaussian unbounded score growth 与 categorical bounded-but-persistent logit updates。

### 理论输出必须明确
- location；
- existence；
- boundary；
- no-finite-equilibrium / divergence；
- step-size-dependent local stability。

### 目标长度
130–160 words。

### 禁止内容
- “positive force dominates / negative force dominates”作为全部结论；
- small is good, large is bad；
- 定理直接证明 task-performance collapse；
- boundary event 与任务失败混为一谈。
<!-- MANUSCRIPT:END INTRO-P06 -->

<!-- MANUSCRIPT:BEGIN INTRO-P07 -->
## [INTRO-P07] DRPO
Parent-Outline-SHA256: `e9b6c4430f9b3c3113bc2ceef385162b8f2441b4b20a147e6baeacb728746fb9`

### 回答的问题
DRPO 如何直接切断核心机制，为什么选择 Exp？

### Topic sentence
> DRPO attenuates negative updates according to standardized distance or action surprisal, weakening far-field gradients without discarding all negative feedback.

### 论证顺序
1. 正 advantage update 保持不变；负 update 使用 `w_λ(r)=exp(-λr)`。
2. Gaussian 的 `r` 使用 calibrated standardized distance / negative log-density；categorical 使用 negative log-probability / surprisal。
3. Exp 不建模 negative-sample utility，也不定义真实安全半径。
4. 方法理论只证明 exponential tail 可支配 finite-order score growth，使 weighted far-field gradient 最终下降。
5. Global、Linear、Hard 是对照，不提前写方法排名。
6. Optimistic DRO 历史联系只用一句，完整推导留在方法章/附录。

### 目标长度
105–130 words。

### 禁止内容
- utility decays exponentially；
- Exp is theoretically optimal；
- DRPO solves all negative-gradient problems；
- DRPO replaces SFT；
- soft Exp 已被证明是原 DRO 的 exact solution。
<!-- MANUSCRIPT:END INTRO-P07 -->

<!-- MANUSCRIPT:BEGIN INTRO-P08 -->
## [INTRO-P08] Evidence Chain and Contributions
Parent-Outline-SHA256: `af0ba0a522a75ddcc66819e9160555c55c91ae7c80cb2a1f8f2b195c498c8dd4`

### 回答的问题
如何用分层证据证明来源、传导、因果路径与外部适用性？

### Topic sentence
> We test the proposed account through source isolation, causal intervention, and progressively broader off-policy policy-learning settings.

### 证据顺序
1. 产品流形 / analytic source isolation：只回答 far-field large gradient 来源。
2. C-U1 非线性 Gaussian 环境：回答是否传导为 drift、boundary event 和 task degradation。
3. D-U1：回答 categorical persistent probability suppression。
4. near/far 与 common/rare interventions：闭合因果路径。
5. Online/replay：检验 data refresh 与 stale reuse 边界。
6. D4RL：continuous-control external validity。
7. Countdown：controlled high-staleness sequence-policy stress test。

### Contributions 的固定四项
1. Far-field gradient mechanism and aggregate equilibrium theory；
2. Controlled causal identification；
3. Exponential DRPO；
4. External validation with terminal audits。

### 治理边界
- C-U1 只称 held-out-context / unseen-state generalization；
- task-performance collapse、boundary event、NaN/Inf 分开；
- Online、D4RL、Countdown 未完成前不提前写结果幅度或排名。

### 目标长度
150–180 words。
<!-- MANUSCRIPT:END INTRO-P08 -->
