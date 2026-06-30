# DRPO Introduction 段落级施工图 v0.4

**状态：active Introduction blueprint；已获用户批准、接入 manuscript cascade live hierarchy，并通过 `docs/manuscript/RL_PAPER_WRITING_GUIDANCE.md` 审阅。**

**上位大纲：** `docs/paper_rewrite_outline_v0_9.md`。

**Live hierarchy：** `docs/manuscript/hierarchy.yaml`。

**研究状态唯一权威 Master：** `docs/handoff.md`。

**基线：** GitHub `main` commit `84edc2aa0b2f258033ddf2ef9aaf98e7a89a6edd`。

**证据边界：** planning、静态检查、smoke、pilot 和未完成外部实验不得升级为正式结果。

## 使用规则

- Paragraph ID、标题、数量、顺序必须与 v0.9 canonical outline 一致。
- `Parent-Outline-SHA256` 绑定对应 outline block；上位 block 改动后必须重新审阅并更新 hash。
- 每段只展开 topic sentence、论证顺序、证据、引用目标、长度和禁区，不得自行改变段落职责。
- fixed advantage 不进入理论叙事；只可在受控实验设置中作为混杂控制出现。
- Product manifold 不作为第三个主环境；正文使用 `C-U1 E1 quality–distance factorized source-isolation protocol`。
- Introduction 每一段必须通过 guidance hard gates G01–G14。

**预计长度：** 900–1050 English words，约 2.0 双栏 columns。

---

<!-- MANUSCRIPT:BEGIN INTRO-P01 -->
## [INTRO-P01] Negative Feedback as a Policy-Improvement Resource
Parent-Outline-SHA256: `68c103770f8127229c43cea44839ce93ce9afa67640e2cb634011bb81b02ff19`

### 回答的问题
为什么 negative feedback 是 policy optimization 的资源，而不是首先被定义为噪声？

### Topic sentence
> Policy improvement relies not only on reinforcing successful actions, but also on suppressing failures that reveal which modes and directions should lose probability.

### 论证顺序
1. Policy optimization 同时利用 positive attraction 与 negative repulsion。
2. Positive updates 学习已经观察到的成功行为；negative updates 抑制坏 mode、塑造边界并释放概率质量。
3. Positive-only 是重要稳定基线，但其目标由 observed positive behavior 决定，可能停在最佳已观察正样本附近。
4. 以核心矛盾收束：negative feedback 有价值，因此正确问题不是是否删除它，而是如何防止其动力学失控。

### 证据与引用目标
- policy-gradient / actor–critic 基础；
- advantage-weighted learning；
- learning from failures / negative reinforcement；
- 只引用可核验 primary sources。

### 目标长度
115–135 words。

### 禁止内容
- negative updates are inherently harmful；
- Positive-only cannot learn；
- fixed advantage；
- far-field 数学、Theorem 1 或 Exp 细节。
<!-- MANUSCRIPT:END INTRO-P01 -->

<!-- MANUSCRIPT:BEGIN INTRO-P02 -->
## [INTRO-P02] Historical Reuse Turns Local Feedback into Persistent Repulsion
Parent-Outline-SHA256: `5238c14bf861a8f37287be4750f6e50c059a4cd18039bd4367d1ee70aa9811c1`

### 回答的问题
一个最初有用的负更新为什么会随训练变成持续的破坏性排斥？

### Topic sentence
> Negative feedback becomes dynamically hazardous when a historical action remains in optimization after the learner has moved far away from it.

### 论证顺序
1. 初次出现的负动作可能位于当前策略附近，提供局部边界信息。
2. Offline logs、replay、stale rollout 或 asynchronous trajectories 使同一动作在 learner 改变后继续被复用。
3. far field 是相对于 current policy 的动态关系，不是数据永久标签。
4. Gaussian 中 standardized distance 增长可放大 score；categorical direct-logit score 有界，但更新可持续降低 action probability。
5. 收束为：local feedback → historical reuse → far-field repulsion → excessive aggregate negative contribution。

### 证据与引用目标
- stale/off-policy reuse；
- low-probability updates；
- Introduction 只给直觉，不放 proof 或限制清单。

### 目标长度
145–170 words。

### 禁止内容
- objective stationarity / global convergence 防御；
- categorical gradient explosion；
- complex shared networks 下逐样本必然单调；
- 把 offline 等同于 frozen advantage。
<!-- MANUSCRIPT:END INTRO-P02 -->

<!-- MANUSCRIPT:BEGIN INTRO-P03 -->
## [INTRO-P03] The Missing Link: Separating Badness from Distance
Parent-Outline-SHA256: `5ef4c5f4cdad8f4558e6811ca766f953b237f17971c61e435c2aae339d9a32c6`

### 回答的问题
如何排除“远场梯度更大只是因为远场样本更差”这一关键替代解释？

### Topic sentence
> To identify the source of the gradient gap, sample badness must be separated from policy-relative distance rather than measured where the two are naturally correlated.

### 论证顺序
1. 现实数据中 reward、negative advantage、rarity、distance 常常耦合。
2. 因此仅观察 far samples 梯度更大不能证明 distance 独立起作用。
3. 介绍 paper-facing C-U1 E1 isolation：匹配 context、reward、advantage severity、sample count 和 base coefficient，只改变 policy-relative distance。现有 formal E1 已闭合 `|A|`-matched output-score claim；由于等 reward 轮廓同时改变动作方向，full-parameter 归因还需 same-state/same-ray radial probe 或 Jacobian-gain decomposition。
4. 预告结论：`|A|` far/near 匹配时，policy-output score 随距离增大；只有在 same-ray/Jacobian closure 完成后，才把完整 full-parameter ratio 归因于 distance，而不是方向或网络 Jacobian。
5. 明确解释：distance 是独立 amplifier，而不是唯一决定因素。

### 证据
- 使用 formal C-U1 E1 output-score source-isolation result，并把 full-parameter closure 状态写清；
- 正文可给 far/near `|A|=1` 的关键控制与 gradient ratio，但数字必须与 handoff/正式 artifact 一致；
- 历史 Product-manifold 只在 appendix/provenance 提及。

### 目标长度
145–175 words。

### 禁止内容
- Product manifold 作为第三主环境；
- distance is the only cause；
- 将 correlation 写成 causal isolation；
- 使用未核对的历史倍率。
<!-- MANUSCRIPT:END INTRO-P03 -->

<!-- MANUSCRIPT:BEGIN INTRO-P04 -->
## [INTRO-P04] Repulsive Dynamics Explains Stable Extrapolation and Equilibrium Loss
Parent-Outline-SHA256: `6e95e0720edd2066ed2ecb9218ada7d5a0e4049e6821a26e82b1782088d34a25`

### 回答的问题
单样本梯度放大如何转化为“先有益、后失稳”的聚合策略动力学？

### Topic sentence
> Repulsive Dynamics reveals a phase transition in which negative updates first create a stable equilibrium beyond the positive-only target and then, as their aggregate contribution grows, drive that equilibrium to the policy boundary and eliminate it.

### 论证顺序
1. Positive-only 对应 positive target 的有限平衡。
2. 适度 aggregate negative contribution 将平衡点推到 positive target 之外，形成 stable extrapolation。
3. 继续增强 negative contribution，使 signed target 接近 mean-parameter feasible boundary。
4. target 越界后不存在有限平衡；恢复项消失时形成 persistent drift。
5. Gaussian 与 categorical 共享聚合结构，但分别表现为 distance-amplified mean/support dynamics 与 bounded-but-persistent probability suppression。

### 证据与公式边界
- Introduction 只出现 phase sequence，不放完整 `m*` 推导；
- 不把 boundary 自动等同于 task collapse；
- 不出现 fixed empirical field、future work 或 global guarantee。

### 目标长度
150–180 words。

### 禁止内容
- Theorem 1 变成所有 RL 的收敛定理；
- “small negative good, large negative bad”作为全部贡献；
- boundary、task failure、NaN 混写；
- expected Fisher SPD 旧证明。
<!-- MANUSCRIPT:END INTRO-P04 -->

<!-- MANUSCRIPT:BEGIN INTRO-P05 -->
## [INTRO-P05] DRPO Controls the Destabilizing Far-Field Term
Parent-Outline-SHA256: `67afcc3c0b2123f4db7d255c41fed9c742b79031d680e9fe3c0136ca0af83512`

### 回答的问题
DRPO 为什么是理论的直接方法后果，而不是额外的 heuristic？

### Topic sentence
> DRPO acts on the same aggregate negative term identified by Repulsive Dynamics, attenuating its far-field component while retaining stronger local repulsion.

### 论证顺序
1. Theorem 1 中 `q m_-` 既推动 stable extrapolation，也在过强时推动 boundary crossing。
2. DRPO 用 `exp(-lambda r)` 将其替换为 weighted `q_lambda m_{-,lambda}`。
3. Gaussian 用 registered standardized distance/calibrated NLL；categorical 用 surprisal；sequence setting 用 registered normalized NLL。
4. Exp 在 finite-order score growth 下使 weighted far-field gradient 衰减到零。
5. 它不删除全部负反馈，也不从假设的 utility decay 推导。

### 目标长度
130–155 words。

### 禁止内容
- hard filtering is the only solution；
- Exp 必然优于 Global/Linear/Hard/SBRC/Hybrid；
- utility exponentially decays；
- 重新引入独立于 Theorem 1 的方法故事。
<!-- MANUSCRIPT:END INTRO-P05 -->

<!-- MANUSCRIPT:BEGIN INTRO-P06 -->
## [INTRO-P06] Evidence Chain and Contributions
Parent-Outline-SHA256: `c7eb68d3a7bfb86e7b7f821a99ff5e9d296c33c232b9b8a48d986d20a77569ad`

### 回答的问题
论文如何用最少的 contribution bullets 展示完整、可信且不越界的证据链？

### Topic sentence
> We validate this account through a reality-anchor, controlled-identification, and reality-closure evidence chain that separates external occurrence, geometric source, causal transmission, phase behavior, and method control.

### 论证顺序
1. 以 Hopper/Countdown 的 terminal-audited mechanism signature 作为 external reality anchor；在正式结果可用前，相关结论和数字保持 `TBD`。
2. C-U1 E1：badness–distance isolation，回答“大梯度为什么变大”；明确 output-score claim 已完成，full-parameter same-ray/Jacobian closure 待登记执行。
3. C-U1/D-U1 interventions：回答 far/rare negative contribution 是否传导为 drift、boundary event 与 task degradation。
4. E4/E6：映射 Theorem 1 的 Positive-only、stable extrapolation、boundary、persistent drift regimes。
5. 已完成 Budget-Match 证明同 raw negative-gradient budget 下 allocation matters；shortlist-freeze、TAPER-CONV、CONFIRM 决定能否形成 terminally stable ranking。
6. 回到 Hopper/Countdown 的 task performance 形成 external closure。
7. 四项贡献：theory、quality–distance causal identification、DRPO、terminal-audited multi-level validation。

### 状态纪律
- 不预填 Hopper/Countdown 数字；
- 不把 pilot 或有限步数升级；
- C-U1 只能称 held-out-context/unseen-state generalization；
- Product manifold 只作 provenance。

### 目标长度
155–185 words。

### 禁止内容
- 环境清单代替 contribution；
- unregistered Online RQ；
- 预设方法排名；
- 模糊使用 collapse。
<!-- MANUSCRIPT:END INTRO-P06 -->
