# DRPO Introduction 段落级施工图 v0.5

**状态：** active Introduction blueprint after application；由 v0.9.1 canonical outline 派生并通过稳定 Guidance 与 DRPO strategy 审查。

**上位大纲：** `docs/paper_rewrite_outline_v0_9_1.md`。

**上位策略：** `docs/manuscript/DRPO_MANUSCRIPT_STRATEGY.md`。

**稳定写作标准：** `docs/manuscript/RL_PAPER_WRITING_GUIDANCE.md`。

**Live hierarchy：** `docs/manuscript/hierarchy.yaml`。

**研究状态唯一权威：** `docs/handoff.md`。

**基线：** GitHub `main` commit `445a2e6d129994b2dd48f7c87050206dc705b838`。

## 使用规则

- Paragraph ID、标题、数量、顺序与 v0.9.1 outline 完全一致。
- `Parent-Outline-SHA256` 绑定对应 outline block；父 block 修改后必须级联更新。
- 本文件只展开 topic sentence、论证顺序、证据、引用目标、长度和禁区，不得重新决定论文战略。
- DRPO 名称与原论文谱系的关系必须正面、简洁地表达，不得把名称写成待辩护的偶然选择。
- fixed advantage 只可在受控实验设置中作为混杂控制出现。
- Product manifold 不进入 Introduction 的环境清单；历史来源只在 appendix/provenance 中交代。
- 实时实验状态、seed 队列和执行计划不写入段落结构；正式结果由 handoff/review 注入。

**预计长度：** 900–1050 English words，约 2.0 双栏 columns。

---

<!-- MANUSCRIPT:BEGIN INTRO-P01 -->
## [INTRO-P01] Negative Feedback as a Policy-Improvement Resource
Parent-Outline-SHA256: `547c99224fbe00be26fa0391dfa19c7ea53f4079fa0c9ae627c1aaa788bd46ea`

### 回答的问题
为什么 negative feedback 是 policy improvement 的资源，而不是首先被定义为噪声？

### Topic sentence
> Policy improvement relies on both attraction toward successful behavior and repulsion from known failures, and their balance can move the policy beyond the target induced by positive data alone.

### 论证顺序
1. Positive updates reinforce observed successful behavior。
2. Negative updates suppress known bad modes and provide a directional counterforce。
3. Positive-only is a stable and informative reference, but its target is determined by observed positive behavior。
4. Balanced repulsion can shift the policy equilibrium beyond that target。
5. 收束为核心问题：如何保留这种 improvement resource，同时避免 repulsion 随训练变得过强？

### 证据与引用目标
- policy-gradient / actor–critic 基础；
- advantage-weighted learning 与 learning from failures；
- 只使用 primary sources；
- 不在本段放本文实验数字。

### 目标长度
120–140 words。

### 禁止内容
- negative updates are inherently harmful；
- Positive-only cannot learn；
- probability mass “必然”流向正确未知动作；
- fixed advantage、far-field 公式或 theorem assumptions。
<!-- MANUSCRIPT:END INTRO-P01 -->

<!-- MANUSCRIPT:BEGIN INTRO-P02 -->
## [INTRO-P02] Historical Reuse Turns Local Feedback into Persistent Repulsion
Parent-Outline-SHA256: `5ef4a7f689f42febc3d9d35a93e4a11d42291a2ad3e4f557ebe7c5fc2638eabd`

### 回答的问题
同一个最初有用的负反馈为什么会随 historical reuse 转化为破坏性排斥？

### Topic sentence
> Negative feedback becomes dynamically hazardous when a historical action remains in optimization after the learner has moved far away from it.

### 论证顺序
1. 初次出现的负动作可能位于当前策略附近，提供局部 boundary information。
2. Offline logs、replay、stale actors 和 asynchronous trajectories 让该动作在 learner 改变后继续出现。
3. far field 是相对 current policy 的动态关系，不是样本永久标签。
4. Gaussian standardized distance 可放大 score；categorical direct-logit score 有界但可持续压低概率。
5. 收束为 local useful feedback → historical reuse → far-field excessive influence。

### 证据与引用目标
- off-policy/replay/staleness；
- low-probability updates；
- Introduction 只讲机制直觉，不放 proof。

### 目标长度
145–170 words。

### 禁止内容
- objective stationarity 或 global guarantee 防御；
- categorical Euclidean gradient explosion；
- 把 offline 等同于 frozen advantage。
<!-- MANUSCRIPT:END INTRO-P02 -->

<!-- MANUSCRIPT:BEGIN INTRO-P03 -->
## [INTRO-P03] The Missing Link: Separating Badness from Distance
Parent-Outline-SHA256: `4f041058dc84a906983364d3000022cfe0bc761e3a656db8cf07de934890ab79`

### 回答的问题
如何排除“far samples 梯度更大只是因为它们更差”这一核心替代解释？

### Topic sentence
> Identifying far-field amplification requires matching sample badness and changing policy-relative distance, rather than measuring the two where they are naturally correlated.

### 论证顺序
1. 现实数据中 reward、negative advantage、rarity 和 distance 常常耦合。
2. 因而普通分桶只显示相关性，无法说明 distance 独立贡献。
3. 介绍 paper-facing matched control：匹配 context、quality/semantics、reward、`|A|`、count 和 base coefficient，只改变 distance/rarity。
4. 预告 continuous 与 categorical source isolation，以及 near/far/common/rare targeted intervention。
5. 结论口径：distance/rarity is an independent amplifier, not the only cause。

### 证据
- C-U1 E1 paper-facing protocol；
- D-U1 quality/semantics/count matching audit；
- full-parameter claim only after same-state/same-ray or registered Jacobian-gain closure；
- 历史 Product-manifold 只留 appendix provenance。

### 目标长度
150–180 words。

### 禁止内容
- Product manifold 作为第三个环境；
- distance is the only factor；
- correlation 写成 causal isolation；
- 未核验倍率。
<!-- MANUSCRIPT:END INTRO-P03 -->

<!-- MANUSCRIPT:BEGIN INTRO-P04 -->
## [INTRO-P04] Repulsive Dynamics Explains Stable Extrapolation and Equilibrium Loss
Parent-Outline-SHA256: `49a40cbbabd951f512efc076b62068e4ad0973e495c4af2f1b3a5dd7a74b5c70`

### 回答的问题
per-sample far-field influence 如何转化为“先提高、后失稳”的 aggregate policy dynamics？

### Topic sentence
> Repulsive Dynamics reveals a phase transition in which negative updates first create a stable equilibrium beyond the Positive-only target and then, as their aggregate contribution grows, drive that equilibrium to the policy boundary and eliminate it.

### 论证顺序
1. Positive-only targets the positive moment。
2. Moderate aggregate negative contribution produces a finite stable extrapolation。
3. Stronger or more outward contribution moves the signed target toward the feasible boundary。
4. Target crossing or loss of restoring mass removes a finite equilibrium and leaves persistent drift。
5. Gaussian and categorical share the aggregate structure but express different boundary dynamics。

### 证据与公式边界
- Introduction 只给 phase sequence；
- `m*` 公式留 theory section；
- boundary、task collapse、NaN/Inf 不合并；
- 不出现旧 expected-Fisher 证明。

### 目标长度
150–180 words。

### 禁止内容
- 把 Theorem 1 写成所有 RL 的收敛定理；
- unrelated guarantee disclaimer；
- fixed empirical field / analysis window 叙事。
<!-- MANUSCRIPT:END INTRO-P04 -->

<!-- MANUSCRIPT:BEGIN INTRO-P05 -->
## [INTRO-P05] DRPO Controls the Destabilizing Far-Field Term
Parent-Outline-SHA256: `97f565852d9d97a09afdd9f246658df8095118d150e9f77796b6511f01c5e921`

### 回答的问题
DRPO 如何同时成为 Theorem 1 的方法后果和原 DRPO 研究谱系的延续？

### Topic sentence
> DRPO acts on the aggregate negative term identified by Repulsive Dynamics, smoothly attenuating its far-field component while retaining the local repulsion that enables improvement beyond Positive-only learning.

### 论证顺序
1. Theorem 1 中 `q m_-` 同时负责 useful equilibrium shift 与 excessive boundary push。
2. DRPO 用 `exp(-λr)` 形成 `q_λ m_{-,λ}`。
3. 近场负反馈保留，weighted far-field gradient 在 finite-order growth 下消失。
4. 原 DRPO 的 Optimistic-DRO/hard filtering 是 distributional-selection endpoint；新 smooth control 是同一方法族的选择性扩展。
5. 因而 DRPO 名称体现研究连续性，而不是给新 heuristic 套旧名字。

### 证据与引用目标
- 原 DRPO arXiv:2602.10430；
- theorem–method equation chain；
- main text 只用一句 lineage bridge，详细历史放 method/appendix。

### 目标长度
145–170 words。

### 禁止内容
- hard filtering is the only solution；
- Exp 必然优于其他 control；
- utility exponentially decays；
- 把名称变成防御性讨论。
<!-- MANUSCRIPT:END INTRO-P05 -->

<!-- MANUSCRIPT:BEGIN INTRO-P06 -->
## [INTRO-P06] Evidence Chain and Contributions
Parent-Outline-SHA256: `a05d9704a6884bc46424e1ad36805118f4978b036c47e883d91784804408d303`

### 回答的问题
如何用四个 research questions 展示完整而不碎片化的证据链？

### Topic sentence
> We validate the account through an external–controlled–external evidence chain that separates occurrence, source and causal transmission, phase behavior and method control, and final task improvement.

### 论证顺序
1. RQ1：Hopper/Countdown formal terminal-audited external occurrence。
2. RQ2：C-U1/D-U1 matched-badness source isolation + near/far/common/rare causal interventions。
3. RQ3：Theorem 1 phase map、aggregate negative term measurement、matched-budget DRPO comparison。
4. RQ4：D4RL/Countdown external task closure。
5. 四项贡献：Repulsive Dynamics、quality–distance/rarity identification、lineage-preserving DRPO、multi-level terminal-audited validation。

### 状态纪律
- Introduction 不保存 seed/horizon/queue 等实时执行信息；
- 未形成 formal result 的外部数字不预填；
- C-U1 只称 held-out-context/unseen-state generalization；
- task collapse、boundary、NaN/Inf 分开；
- Product manifold 不进入环境清单。

### 目标长度
155–185 words。

### 禁止内容
- 六个碎片化 RQ；
- 环境 inventory 代替 contribution；
- unregistered Online RQ；
- 预设方法排名。
<!-- MANUSCRIPT:END INTRO-P06 -->
