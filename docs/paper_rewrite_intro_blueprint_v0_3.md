# DRPO Introduction 段落级施工图 v0.3

**状态：active；严格从用户审阅通过的 canonical outline v0.7 逐段派生。**
**上位大纲：** `docs/paper_rewrite_outline_v0_7.md`。
**Live hierarchy：** `docs/manuscript/hierarchy.yaml`。
**研究状态唯一权威 Master：** `docs/handoff.md`。
**证据边界：** planning、静态检查、smoke 或 pilot 不得升级为正式实验结果。

## 使用规则

- 本文件必须与 v0.7 Introduction 的七个 Paragraph ID、标题、数量和顺序完全一致。
- 每个 block 的 `Parent-Outline-SHA256` 绑定对应 v0.7 outline block。
- 本文件只能展开 topic sentence、论证、证据、引用、字数和禁区，不得反向改写 canonical outline。
- 若在施工图审阅中发现内容优化，必须先重新检查对应 outline：outline 本身错误时先修 outline 并级联；outline 正确时只修 blueprint 及其下游。
- “施工图与大纲不一致”本身不是大纲错误证据，禁止为消除 mismatch 而反向修改大纲。

**预计 Introduction 长度：** 950–1150 English words，约 2.0–2.25 双栏 columns。

---

<!-- MANUSCRIPT:BEGIN INTRO-P01 -->
## [INTRO-P01] 背景与重要性

Parent-Outline-SHA256: `16d73cdd078b9e07b98460b459dd53af243f01f854fc591c400869f1fcad5bba`

### 回答的问题
为什么 policy optimization 同时利用成功与失败行为，这一问题为何跨越 offline RL、replay、continuous control 与 verifier-guided language-model training？

### Topic sentence
> Policy optimization improves a policy by learning not only from successful behavior, but also from actions that underperform a reference value.

### 论证顺序
1. Policy optimization 使用 positive and negative advantage，而不是只拟合成功行为。
2. 该范式存在于 offline RL、replay-based policy learning、asynchronous policy optimization、continuous control 和 verifier-guided language-model training。
3. 与纯 behavior cloning 相比，利用 signed feedback 具有超越行为数据平均质量的潜力。
4. 本段只建立背景，不提前引入 far field、平衡定理或 DRPO。

### 引用目标
- policy-gradient / actor–critic foundations；
- advantage-weighted policy learning；
- representative offline / replay / verifier-guided policy learning。

### 目标长度
110–140 words。

### 禁止内容
- negative samples are inherently harmful；
- positive-only methods cannot learn；
- all modern RL systems are off-policy；
- 提前展开 equilibrium、Exp 或实验结果。
<!-- MANUSCRIPT:END INTRO-P01 -->

<!-- MANUSCRIPT:BEGIN INTRO-P02 -->
## [INTRO-P02] 正负 advantage 的不同作用

Parent-Outline-SHA256: `a67ea57bfc30849950ff944850392d7dcf43ed5662fc29df6b949520f3165340`

### 回答的问题
正、负 advantage 分别提供什么作用，为什么 negative updates 既不能被简单删除，也不能不受控制地重复使用？

### Topic sentence
> Positive and negative advantages induce qualitatively different policy updates: attraction toward successful behavior and repulsion away from unsuccessful behavior.

### 论证顺序
1. Positive advantage 提高高价值行为的概率，形成 attraction。
2. Negative advantage 降低低价值行为的概率，形成 repulsion。
3. 受控负更新可能抑制竞争性坏行为、提供局部决策边界、重新分配概率质量，并帮助策略越过 Positive-only 的性能上限。
4. 风险不在于 advantage 必然变大，而在于固定负样本被策略推远后，其 current-policy score 与负梯度幅度可能变化。
5. 本段只提出“双重作用”和核心异常；far field 的正式定义与理论条件留给后续段落。

### 引用目标
- positive-only / winner-only / asymmetric weighting；
- learning from failures / useful negative feedback；
- low-probability negative-gradient risk。

### 目标长度
120–150 words。

### 禁止内容
- near-field negatives are always useful；
- far-field negatives are always harmful；
- advantage explosion；
- 负更新单独决定任务性能。
<!-- MANUSCRIPT:END INTRO-P02 -->

<!-- MANUSCRIPT:BEGIN INTRO-P03 -->
## [INTRO-P03] 为什么 fixed/stale off-policy data 特别危险

Parent-Outline-SHA256: `6d3eb4da942bf2ae7a92c6fbde842b8fd5a5fc023062173c75522d15a0e682be`

### 回答的问题
为什么 fixed or stale off-policy data 会把单步负更新转化为可持续的长期动力学问题，论文又如何覆盖 replay？

### Topic sentence
> The risk becomes persistent when historical negative actions remain in the training distribution after the current policy has moved away from them.

### 论证顺序
1. Fresh on-policy data 会随当前策略刷新，旧动作通常不会以同样方式长期驻留。
2. Fixed offline datasets、replay buffers、stale actors、frozen rollout banks 和 historical trajectory reuse 会让旧样本继续参与更新。
3. Fully offline + frozen advantages + fixed base weights 时，empirical actor objective 在 actor stage 精确 stationary。
4. Replay/stale 情形不保证 objective 全局固定，但 behavior–learner mismatch 可以在有限更新窗口内持续。
5. 理论使用 stationary case 获得精确结论；Online/replay、D4RL 和 Countdown 检验更广泛 off-policy data reuse。

### 必须出现的边界句
> We derive exact results for a stationary frozen-data actor objective and evaluate the same persistence mechanism in broader replay-based and stale-policy settings.

### 目标长度
135–165 words。

### 禁止内容
- the more off-policy, the more fixed the objective；
- 论文只研究 fully offline；
- replay 与 fixed offline 完全等价；
- dynamic critic / moving equilibrium 作为当前主线。
<!-- MANUSCRIPT:END INTRO-P03 -->

<!-- MANUSCRIPT:BEGIN INTRO-P04 -->
## [INTRO-P04] 已有解决方法

Parent-Outline-SHA256: `c6a5971b1c6a044dca0bce326c957490b7b1a6f13b18a52bbfde59404d175577`

### 回答的问题
已有方法如何处理负更新风险，Introduction 如何提纲挈领而不重复 Related Work？

### Topic sentence
> Existing approaches reduce harmful negative updates through deletion or global attenuation, constrained policy changes, and probability-, staleness-, or quality-aware selection.

### 论证顺序
1. 用一组概括句覆盖 positive-only / winner-only、asymmetric or global scaling、policy-ratio clipping、low-probability or surprisal-aware weighting、hard filtering。
2. 明确认可这些方法已经识别并缓解 negative-update domination 与 low-probability risk。
3. 本段不逐项评价方法，只为下一段的共同缺口做铺垫。
4. 方法名、primary citations、objective 和逐项技术差异全部留在 Related Work。

### 目标长度
75–95 words。

### 禁止内容
- 把本段写成缩略文献综述；
- 五类方法逐项展开优缺点；
- existing work ignores negative gradients；
- clipping is ineffective。
<!-- MANUSCRIPT:END INTRO-P04 -->

<!-- MANUSCRIPT:BEGIN INTRO-P05 -->
## [INTRO-P05] 共同缺口

Parent-Outline-SHA256: `ce00882a6be6bad2c007f52573ed453ad4f616bff41f3fab35bf6388a068541b`

### 回答的问题
现有研究尚未统一回答哪些问题，far field 在本文中具体指什么？

### Topic sentence
> What remains unclear is when negative updates improve a policy, when they become destructive in the far field, and how this transition changes the existence of a finite equilibrium.

### 论证顺序
1. 首次定义 far-field negative sample：continuous policy 下 standardized distance 较大，或 categorical policy 下 current probability 很低的负动作。
2. 强调 far field 是相对于 current policy 的动态状态，不是样本的永久标签。
3. 提出五个缺口：负更新为何有益；为何在 far field 反转；有限平衡何时存在或消失；Gaussian/categorical 如何共享结构但保留差异；如何选择性保留有用负更新。
4. 现有工作通常分别研究 sign、magnitude、ratio、probability、staleness、entropy 或 data quality，缺少上述统一解释。

### 目标长度
115–145 words。

### 禁止内容
- no prior work studies negative gradients；
- distance is the only source of instability；
- far-field samples contain no useful information；
- Gaussian 与 categorical 具有完全相同的 score behavior。
<!-- MANUSCRIPT:END INTRO-P05 -->

<!-- MANUSCRIPT:BEGIN INTRO-P06 -->
## [INTRO-P06] 本文理论与方法

Parent-Outline-SHA256: `ee49f6357dea2ac8c070578f29747a611016a88ae5e88efabf7db4a38873abb0`

### 回答的问题
本文的 Repulsive Dynamics 和 DRPO 分别解决什么问题，为什么两者在同一段中形成完整回应？

### Topic sentence
> Repulsive Dynamics characterizes when controlled negative updates yield a finite, improved equilibrium and when persistent far-field gradients eliminate it; DRPO is designed to prevent that transition without discarding all negative feedback.

### 论证顺序
1. 先说明固定 negative advantage 不意味着固定 negative gradient：current-policy score 会随策略位置变化，Gaussian 中可随 standardized distance 无界增长；categorical direct-logit score 有界但负更新可持续累积。
2. Repulsive Dynamics 给出 finite equilibrium、stable extrapolation、boundary approach 和 no finite equilibrium / divergence，并保留 transition-matrix 与 spectral-radius 条件。
3. 理论输出必须是 equilibrium location、existence、boundary、no-finite-equilibrium 与 step-size local stability，不能退化成“负更新小则好、大则坏”。
4. 随后提出 DRPO：对负更新使用 exponential distance/surprisal weighting，削弱 weighted far-field gradient，同时保留局部负更新。
5. Exp 是 gradient-control envelope，不是假设 negative-sample utility 指数衰减，也不定义真实安全半径。
6. Global、Linear、Hard 作为对照，不提前写方法排名；Optimistic DRO 历史联系只用一句。

### Figure 1 引用
> Figure 1 summarizes the transition from the Positive-only limit, to useful controlled repulsion, and finally to persistent far-field instability and DRPO recovery.

### 目标长度
180–220 words。

### 禁止内容
- advantage 本身“膨胀”；
- categorical gradient explosion；
- utility decays exponentially；
- Exp is theoretically optimal；
- 定理直接证明 task-performance collapse；
- boundary event 与任务失败混为一谈。
<!-- MANUSCRIPT:END INTRO-P06 -->

<!-- MANUSCRIPT:BEGIN INTRO-P07 -->
## [INTRO-P07] 实验版图与贡献

Parent-Outline-SHA256: `20e24ba114226347faf21aafb693f0507673b0a30c04de3ddd0d80f24f8aa495`

### 回答的问题
论文通过哪些分层实验验证来源、传导、因果路径和外部有效性，最终贡献如何概括？

### Topic sentence
> We evaluate the proposed account through controlled continuous and categorical mechanisms, an Online data-refresh boundary, and external off-policy benchmarks.

### 论证顺序
1. Product-manifold / analytic experiments 只回答 far-field large gradients 的来源。
2. Nonlinear Gaussian / C-U1 环境回答远场异常负梯度是否传导为 drift、boundary event 与 task-performance degradation。
3. D-U1 回答 categorical policy 中的 persistent probability suppression；near/far 与 common/rare interventions 闭合因果链。
4. Online 检验 fresh-data refresh 与 stale replay 的边界；D4RL 和 Countdown 提供外部有效性。
5. 最后按 v0.7 固定四项贡献收束：Repulsive Dynamics Theory、Mechanism Identification、DRPO、Empirical Validation。

### Contributions 固定表述职责
- 理论：finite interior equilibrium、boundary approach、no finite equilibrium/divergence、transition matrix 和 spectral radius；
- 机制：解耦 advantage magnitude 与 distance/probability，并做定点干预；
- 方法：Exp DRPO 保留局部负更新并控制 far-field gradients；
- 实证：Online、D4RL、Countdown 的任务效果、外部有效性与 terminal audit。

### 目标长度
170–210 words。

### 禁止内容
- C-U1 称 OOD generalization；
- task-performance collapse、boundary event、NaN/Inf 混为一谈；
- Countdown 代表所有 LLM RL；
- 未运行实验的性能幅度、排名或正式结论。
<!-- MANUSCRIPT:END INTRO-P07 -->

---

## 段落对照表

| ID | Canonical outline responsibility | Blueprint responsibility | 状态 |
|---|---|---|---|
| INTRO-P01 | 背景与重要性 | 背景、适用场景、研究动机 | PASS |
| INTRO-P02 | 正负 advantage 的不同作用 | attraction/repulsion 双重作用与核心异常 | PASS |
| INTRO-P03 | 为什么 fixed/stale off-policy data 特别危险 | persistence、stationarity 与 replay 边界 | PASS |
| INTRO-P04 | 已有解决方法 | 提纲挈领概括，细节留 Related Work | PASS |
| INTRO-P05 | 共同缺口 | far-field 定义与统一未解问题 | PASS |
| INTRO-P06 | 本文理论与方法 | Repulsive Dynamics 与 Exp DRPO 的完整回应 | PASS |
| INTRO-P07 | 实验版图与贡献 | 分层证据链与四项贡献 | PASS |

任何一行职责发生变化，必须先修改 canonical outline，再重新派生本施工图。
