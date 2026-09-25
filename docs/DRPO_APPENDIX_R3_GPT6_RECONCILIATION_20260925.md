# DRPO Appendix R3 / GPT-6 复核反馈（全文通读后版本）

## 0. 复核基准

本轮不是对 GPT-6 意见做局部逐条接受，而是先完整通读当前论文，再从正文 claim、附录定义、真实实验协议、结果 provenance 四个层面交叉核对。

- Repository / branch: `easonhuo/drpo / main`
- 审查基准 commit: `e4976edaa4577b248646bfbe9892cd0c108985a2`
- 当前审查基准 manuscript blob: `ffd987fb4358f92ef98eb32ad6a3c2cb40774fc4`
- GPT-6 反馈对象: `DRPO_Appendix_R3_Review_Feedback_20260925.md`
- R3 原方案对象: `DRPO_ICLR2027_实验附录论文职责审计_R3_20260925.md`

我已经从 Abstract 到 Appendix 最后一节完整通读 `paper/iclr2027/manuscript_source.tex`，并额外核对了当前 `experiments/registry.yaml`、C-U1 E3 runner / formal outputs、E8 gradient-probe 实现与配置、D4RL development-closure 文档和 figure manifests。

总体判断：

> GPT-6 对 R3 的六个问题中，**第 1、3、6 项基本成立；第 4、5 项方向成立但实际问题比其描述更深；第 2 项“发现冲突”成立，但它给出的具体 fixed-variance 修法不能直接采用。**
>
> 更重要的是，全文通读后又发现了几项 GPT-6 没有指出、但优先级更高的结果-provenance 问题。这些问题如果不先处理，仅仅把 appendix “写得更像论文”反而会把尚未闭合的结果包装得更正式。

---

## 1. 对 GPT-6 六项意见的逐项结论

### 1.1 C-U1 距离定义和 5.0 阈值：同意，已经修

**结论：同意 GPT-6。**

R3 原来的整段替换确实会把

[
d_	heta(s,a)=rac{|a-mu_	heta(s)|_2}{sigma_	heta(s)}
]

以及 near/far threshold (5.0) 一并删掉。这个不是内部 experiment bookkeeping，而是实际 intervention definition。

当前代码 `src/drpo/drpo_cu1_e1_e4_oneclick.py` 也确认 E3 使用动态 standardized distance，阈值固定为 5.0。

**已在 draft manuscript 中处理：**

- 删除 E1/E2/E3/E4 内部编号映射；
- 保留标准化距离公式；
- 保留固定阈值 5.0；
- 改成 source-isolation / Positive-only reference / causal-transmission / taper comparison 的论文职责叙述。

这一点 GPT-6 的修改方向正确。

---

### 1.2 Fixed variance vs. learnable variance：同意发现冲突，但不同意 GPT-6 给出的直接修法

**结论：问题成立；建议修法不能直接采用。**

GPT-6 正确发现两处文字冲突：

- C-U1 环境写“main continuous mechanism results use the fixed-variance branch unless explicitly labeled”；
- Causal-Transmission 又写“policy mean and variance evolve jointly”。

但它进一步建议直接把 Figure 6.3 绑定为 fixed-variance branch，这一步不安全。

原因有两层。

#### 第一层：当前 Figure 6.3 底层 CSV 本身不像 current fixed-variance formal result

当前 Figure 6.3 rescue 数据来自：

`results/FIGURE2_CONTROLLED_SOURCE_TRANSMISSION/fig_6_3_2_causal_summary.csv`

其中不仅存在 `final_sigma_r` 的变化，而且数值与当前正式 C-U1 E3 long-run result 不一致。例如旧 figure summary 中：

- baseline final reward 约 0.201，task collapse 19/20；
- far-zero 约 0.618；
- far-cap 约 0.666；
- far-to-near 约 0.285，collapse 13/20。

而当前正式 `C-U1-E3-ADAM-RERUN` / `outputs/cu1_e3_adam/` 中 fixed-variance branch 是：

- baseline / near-zero 约 (2	imes10^{-6})，20/20 task collapse；
- far-zero 约 0.739；
- far-cap 约 0.733；
- global 约 0.599；
- far-to-near 约 0.875，0/20 task collapse。

learnable-variance branch则承担 support contraction 结果。

因此，**不能只靠文字逻辑把现有 Figure 6.3 声明成 fixed-variance formal branch**；先要重新绑定 figure provenance。

#### 第二层：当前 formal C-U1 E3 本来就有两个互补 branch

更准确的论文职责是：

- **fixed-variance branch**：隔离 mean drift 与 task-performance collapse / rescue；
- **learnable-variance branch**：检查同一 far-field pathway 是否传导为 support / variance contraction；
- 两者都不能和 NaN/Inf numerical failure 混为一类。

**已在 draft manuscript 中处理：**把 C-U1 环境和 Causal-Transmission 改成这两个 branch 的明确分工。

**没有做的事情：**没有把现有 Figure 6.3 强行标成 fixed-variance，因为当前 figure artifact 与 formal E3 provenance 尚未统一。

所以对 GPT-6 的反馈应当是：

> 你指出了真实冲突，但“Figure 6.3 = fixed variance”不是仅凭当前 manuscript 就能推出的安全修复。需要先对齐 figure source 与 current formal E3 result。

---

### 1.3 SG 统计定义：同意，而且问题比 GPT-6 所写更强

**结论：同意。**

GPT-6 正确区分了三个维度：

1. 训练 horizon：1,200 optimizer updates；
2. 每个 coefficient 的时间维统计：late-window Pass@8；
3. coefficient 轴上的五点窗口汇总。

这三者不能混写。

当前 E8 文档进一步确认：

- late-window 是 updates **800, 900, 1000, 1100, 1200** 五个 evaluation points 的均值；
- main table 的“contiguous five-point window”是 **(lambda) 轴上的五个相邻 coefficient points**，不是训练时间窗口；
- 当前 coefficient-response evidence 是 **staged held-out evaluation response curves**：后续 coefficient range 曾在看过前一轮 held-out curves 后扩展；
- 因此不能把整个 combined response curve 包装成“一次性 untouched confirmatory test”。

**已在 draft manuscript 中处理：**

- 明确 late-window = 800/900/1000/1100/1200 的均值；
- 去掉若干 `registered` 之类内部治理措辞；
- 将 1,200-step curves 明确写成 fixed-horizon measurements，不把 endpoint 解释为 convergence / steady state。

**仍未直接改 main SG performance table 的原因：**

这不是再补一句定义就能闭合。当前 repo 对这些 response curves 的科学状态仍保留 pilot / staged-heldout 边界，而 main table 现在把它们压成 task-level DRPO 分数并继续做跨方法计数。这个属于 **result-provenance / claim-strength 问题**，必须单独处理，不能靠 paperization 掩盖。

---

### 1.4 Controlled Protocols：总体同意，但实际定义需要按真实实现重写

#### 1.4.1 Far-to-near：同意 GPT-6，而且已经找到精确定义

GPT-6 说正文报告了 far-field budget transfer，但 appendix 没定义；这一点成立。

当前 C-U1 E3 runner 给出了精确实现。令当前 batch 的 near/far aggregate negative gradients 为

[
mathbf G_N,quad mathbf G_F.
]

Far-cap 不是 manuscript 目前写的逐样本 (C_{m near}/I(z)) cap；真实 formal implementation 是 aggregate gradient cap：

[
gamma_{m cap}
=
minleft(
1,,
0.05rac{|mathbf G_N|}{|mathbf G_F|+epsilon}
ight).
]

Far-cap 使用

[
mathbf G_N+gamma_{m cap}mathbf G_F.
]

Global-matched 匹配的是这个 selective intervention 的 **raw aggregate gradient norm**。

Far-to-near 则在保留同一个 far cap 后，解一个非负 near multiplier (c_N)：

[
|c_Nmathbf G_N+gamma_{m cap}mathbf G_F|
=
|mathbf G_N+mathbf G_F|,
]

即把被移除的 far-field raw-gradient budget 转回 near component。

这也说明 GPT-6 只要求“定义 (C_{m near})”仍然不够：**当前 manuscript 的 Far-cap 数学形式本身和 formal implementation 不一致。**

**已在 draft manuscript 中处理：**把 Far-cap、Global-matched、Far-to-near 全部改成真实 aggregate-gradient 版本，并明确“匹配 raw gradient norm，不匹配 Adam parameter update”。

#### 1.4.2 Task-collapse event：同意，可以精确定义

当前 C-U1 formal E3 定义是：

- 每 100 optimizer updates evaluation；
- held-out-context reward 低于 branch-initial reference 的 45%；
- 连续 3 次 evaluation 满足才记作 task-performance collapse。

learnable-variance branch 的 finite support boundary 是 (logsigma<-12)。

**已在 draft manuscript 中补齐。**

#### 1.4.3 Reward retention / policy shift：GPT-6 指出缺口是对的，但不能直接补一个“看起来合理”的定义

这里必须区分：

- 旧 Figure 6.3 CSV 中的 reward retention；
- 当前 formal E3；
- Figure 6.4.1 phase-transition 中的 policy shift。

当前 phase-transition figure 的 repository source 仍是 template（见第 2 节的新发现），因此现在去“补 policy shift 定义”会把 placeholder 进一步论文化。

C-U1 runner 内确实已有 `normalized_extrapolation_displacement` 的明确实现，可以作为未来正式 phase-transition figure 的候选 metric，但在 figure provenance 未闭合前，不应该假装当前图已经使用了这个定义。

所以这里的结论是：

> GPT-6 正确发现“指标定义不足”，但修复顺序应是先确认当前图的真实数据来源，再写定义，而不是反过来替 placeholder 猜定义。

#### 1.4.4 Near-field-retention matching：同意症状，但 GPT-6 还没抓到最深层问题

GPT-6 指出：如果 (mathcal N) 完全处于 shared DRPO formula 的 untapered region (Dle	au)，那么所有 selective taper 都有 (ho_{m near}=1)，matching 变成恒等式。这个判断正确。

实际 current formal C-U1 near-retention experiment 使用的是另一套明确参数化：

- coordinate: direct standardized distance (d)；
- reference / near boundary: (d_{m ref}=5.0)；
- main target average near retention: **0.75**；
- sensitivity targets: 0.50 / 0.25；
- calibration tolerance: **(10^{-6})**；
- calibration set: development seeds 0--4 的 Positive-only-2000 checkpoint 上 (dle5) 的 pooled near negatives；
- calibration 后 coefficients freeze，再进入 formal seeds。

也就是说，C-U1 controlled taper comparison **不是**当前 shared-method subsection 中那个带 near-field plateau 的
(x=[(D-	au)/c]_+) 公式直接实例化。

因此这里不能只补“target=0.75 / tolerance=1e-6”。真正需要的是：

> 把 “controlled C-U1 taper-shape experiment” 和 “shared external DRPO deployment formula” 的 coordinate / taper parameterization 明确分开。

这属于科学映射问题，不是措辞补洞。我没有在本轮 draft 中擅自重写这一整块，因为需要先确认论文究竟希望把 C-U1 作为“family-shape mechanism comparison”还是“DRPO exact formula reproduction”。

---

### 1.5 D4RL 参数说明：同意冲突，但不能按 GPT-6 的方式直接补表

**结论：同意 GPT-6 对文字冲突的判断；反对在没有 provenance 闭环的情况下把 development-grid 直接补成 formal table。**

GPT-6 指出的两个局部问题都存在：

1. “threshold and scale kept fixed across methods” 与 task-specific (c) grids 冲突；
2. (alpha_{m Exp}=1) 在当前 shared method formulas 中没有明确角色。

**已在 draft manuscript 中做的安全修复：**

- 改成“共享的是 dimension-normalized remoteness coordinate；固定 coordinate settings 与 tuned weighting parameters 分开说明”；
- 删除多余的内部 “registered/config populated” 语言；
- 把 D4RL protocol-matched comparison 压成真正有论文职责的一句。

**但我没有直接把 (alpha_{m Exp}) 猜测性重命名为 (lambda)，也没有把 task-specific grid candidates 猜填进表。**

原因是全文 + registry 复核发现了更大的 blocker：

- `EXT-H-E7-BENCH-01` 当前 formal status 仍是 **not_run / blocked**；
- registry 明确写 formal nine-task benchmark 尚未冻结 exact versions / formal seeds / base algorithm / optimizer / budget；
- D4RL GLQ coarse/refinement 是 development **pilot**；
- closure 明确写：
  - held-out seeds untouched；
  - all selected cells fixed-horizon inconclusive；
  - `cross_method_ranking_allowed: false`；
  - `formal_d4rl9_table_population_allowed: false`；
  - `held_out_confirmation_claim_allowed: false`。

更直接的是：当前 manuscript 的 DRPO D4RL task values（包括 total 698.8）在当前 repo code/document search 中只出现在 manuscript / replacement manuscript，而没有找到一份与它们一一对应的当前 result artifact。

因此，当前最优先的问题不是“table 里 grid 写得不够详细”，而是：

> **当前 main paper 的 D4RL result table 是否有足够的正式结果 provenance。**

在这个问题闭合之前，把 development grid 写得更漂亮只会让一张尚未绑定正式 result source 的表看起来更可信。

---

### 1.6 Structured-generation gradient diagnostic：同意，而且已经找到精确测量定义

GPT-6 的三个问题都成立：

- gradient scalar；
- fixed probe checkpoint；
- 95% CI。

当前实现可以精确回答。

#### Gradient scalar

Countdown probe 和 E8 multitask P0 都对 **mean completion-token log probability** 求 trainable-parameter gradient：

[
G(x,y)
=
left|

abla_{	heta_{m train}}
rac1Tsum_tlogpi_	heta(y_tmid x,y_{<t})
ight|_2.
]

不是 summed sequence log probability。

#### Checkpoint

八个非-Countdown multitask probes：

- task-specific Positive-only warm-start；
- 100 optimizer updates；
- 256 negative responses / task；
- matched absolute advantage = 1.0。

Countdown：

- seed 100 full-bank probe；
- 6,000 puzzles，near/far 各一条，共 12,000 responses；
- SFT/reference adapter；
- absolute negative coefficient = 1.0；
- probe 不执行 optimizer update。

#### CI

八个 multitask probes：

- prompt-within-task cluster bootstrap；
- 2,000 replicates；
- seed 161803。

Countdown：

- puzzle-id cluster bootstrap；
- 400 replicates；
- single-seed full-bank diagnostic。

**已在 draft manuscript 中全部补齐。**

此外全文通读又发现一个 GPT-6 没指出的直接矛盾：

当前 main Figure 1 的 canonical plotting script `scripts/figures/plot_figure1_external_gradient.py` 的 panel (b) 是 **Countdown only**，但 manuscript caption 之前写成 “structured-generation tasks in (b)”。

**已在 draft manuscript 中修成：panel (b) = Countdown full-bank probe；其他八个 task-level curves 放 appendix nine-panel。**

---

## 2. 全文通读后新增的高优先级问题

这些不是 GPT-6 六项里的措辞问题，而是当前 paper claim 与 repository evidence 的直接一致性问题。

### 2.1 Figure 6.4.1 phase transition 当前是 template，却被正文当作结果解释

Repository 的 `scripts/figures/README.md` 和 `paper/figures/FIGURE_INDEX.md` 明确记录：

- Figure 3 / `fig_6_4_1_phase_transition`：template/layout；
- formal phase-scan aggregates pending；
- 不应把 template numbers 当成 formal experimental results。

对应 CSV：

`results/FIGURE3_PHASE_TRANSITION/fig_6_4_1_phase_transition_template.csv`

每一行都有 `is_placeholder=1`。

但当前正文写：

> Figure ... shows a non-monotone transition ...

并进一步解释 task collapse / boundary events。

**这是当前全文中比 appendix “文档感”更严重的问题。**

建议：在真实 aggregate 替换前，不应继续以 empirical-result 口吻引用这张图。

### 2.2 Controlled taper Figure 当前也是 template，却被正文/附录当作 empirical comparison

同一 README 明确：

- `fig_6_4_2_leftfig_bigtext_legend_protocol` = template/layout；
- formal 6.4.2/6.4.3 aggregates pending。

对应 CSV：

`results/FIGURE4_TAPER_CONTROL_TRANSFER/fig_6_4_2_leftfig_template.csv`

也是 template。

但 manuscript 现在写：

> Under both matched comparisons, exponential tapering ...

这已经是结果 claim。

这必须在最终 paper 前替换成真实 formal aggregate，或者删除/降级为 protocol illustration；不能通过 appendix paperization 把它继续包装成结果。

### 2.3 Figure 6.3 当前 artifact 与 current formal E3 不一致

如 1.2 所述，当前 Figure 6.3 rescue CSV 与 `C-U1-E3-ADAM-RERUN` formal result 在 baseline、far-to-near、support dynamics 等处都有明显差异。

因此 Figure 6.3 必须先回答：

> 这张图究竟展示哪个 experiment / branch / commit 的结果？

在 provenance 重新绑定之前，不能只改 caption。

### 2.4 D4RL main table 与当前 formal registry 状态冲突

当前 main manuscript 报告 DRPO D4RL-9 total 698.8，并将其作为 task-level result。

但当前 registry 的正式 D4RL-9 benchmark `EXT-H-E7-BENCH-01` 仍是 not_run / blocked；development tuning closure 明确禁止 formal table population 和 cross-method ranking。

因此必须找到 698.8 的独立合法 result source，或者重新界定/移除当前 table claim。仅仅完善 Appendix D4RL 参数表不能解决这一点。

### 2.5 SG main table 的 “window-max task score” 也需要重新定义证据强度

当前 SG coefficient-response 文档明确：

- fixed 1,200-step finite-horizon；
- response curves 是 staged held-out evaluation；
- 多轮 coefficient range expansion 在早期 held-out curves 已被观察之后发生；
- pilot evidence 不支持 convergence / steady-state / formal method ranking。

因此 main table 把每任务 (lambda)-curve 压成一个 maximum contiguous-five-point-window mean 后，再写“DRPO exceeds AsymRE on six / TOPR on eight”，至少需要在论文中明确它是何种 descriptive summary，而不能让读者理解为 untouched confirmatory test 上的独立 HPO + final test result。

### 2.6 C-U1 controlled taper 与 shared DRPO formula 当前不是同一参数化

如 1.4.4 所述，formal C-U1 near-retention comparison 使用 direct standardized distance (d) 上的 taper family；shared external method 则使用 thresholded normalized excess remoteness / surprisal coordinate。

这两者可以承担互补职责，但不能在 Appendix 里写成同一公式的无缝复用。

---

## 3. 对 R3 原清理方向的最终判断

R3 的大方向仍然正确：

- 删除 RunSpec、artifact fields、internal package/provenance bookkeeping；
- 去掉 E1--E6 这种只服务内部 roadmap 的编号；
- 把 shared-condition checklist 改成正常论文 prose；
- 删除 TOPR 的 internal pilot identity sentence；
- 删除 finite-measure 末尾孤立的 quality-selection 比较；
- 保留环境定义、真实 intervention、matching design 和三类 failure outcome 的区分。

但 R3 不能“一键整段替换”，因为至少有三类科学内容不能被 paperization 吞掉：

1. 实际 protocol 数值与公式（例如 C-U1 (d_	heta)、5.0 threshold、E3 far-cap/far-to-near）；
2. result-statistic 定义（time window、coefficient-window、bootstrap unit）；
3. result provenance / evidence strength（formal vs pilot vs template）。

“删掉内部状态词”不等于“结果自动变成论文结果”。

---

## 4. 本轮已经实际修改的 manuscript 内容

在 draft branch `dev/iclr-appendix-r3-reconcile` 中，已完成以下范围受限修改：

- 保留 C-U1 standardized distance 与 threshold 5.0，同时去掉 E1--E4 内部编号；
- 把 C-U1 fixed/learnable variance 改成两个互补 causal branches；
- 去掉 D-U1 E5/E6-A/B/C 内部编号映射；
- 把 Shared Experimental Invariants checklist 改成 reader-facing controls prose；
- 用 current formal implementation 重写 Far-cap / Global-matched / Far-to-near；
- 补 C-U1 task-collapse 45% × 3 consecutive evaluations 与 (logsigma<-12) boundary 定义；
- 精确定义 SG gradient scalar、probe checkpoint、sample count、bootstrap unit / replicates；
- 修正 main Figure 1 panel (b) 实际是 Countdown full-bank probe，而非 nine-task aggregate；
- 明确 SG late-window = updates 800/900/1000/1100/1200；
- 去掉多处 `registered` / `paper-facing` / RunSpec / internal provenance prose；
- 删除 finite-measure 末尾孤立的 quality-selection comparison；
- 将 D4RL “所有 scale 都固定”改成“coordinate convention 统一；fixed settings 与 tuned weighting parameters 分开”。

这些修改不改变任何实验数据、seed、threshold、训练 horizon 或方法结果。

---

## 5. 本轮刻意没有自动修改的内容

以下不是“忘了改”，而是因为当前 evidence 还不足以安全落笔：

1. **没有把 Figure 6.3 标成 fixed-variance**：figure artifact 与 formal E3 还没对齐。
2. **没有替 Figure 6.4.1/6.4.2 猜正式结果**：repo 明确说它们是 template。
3. **没有给 D4RL task-specific grid 猜 candidate values，也没有把 (alpha_{m Exp}) 猜改成 (lambda)**：formal D4RL benchmark provenance 未闭合。
4. **没有把 D4RL 698.8 或 SG window-max table 重新包装成 formal ranking**：当前 registry / result docs 的 evidence-status 与这种写法存在冲突。
5. **没有强行把 C-U1 controlled taper formula 改成 shared DRPO formula**：两者当前真实实现的 coordinate / taper parameterization 不同。

---

## 6. 希望 GPT-6 重点复核的五个问题

请下一轮不要只看修改后的 Appendix prose，而重点判断下面五项：

1. 是否同意：Figure 6.3 必须先完成 artifact-to-formal-E3 provenance 对齐，才能决定 caption 里的 variance branch。
2. 是否同意：Figure 6.4.1 / 6.4.2 在 repo 明确标为 template 的情况下，当前正文不能继续把它们当正式 empirical result。
3. 是否能从当前 repository 找到 D4RL 698.8 的可审计正式 result source；若找不到，是否同意它不能继续按 formal D4RL-9 task table 叙述。
4. 是否同意：SG staged-heldout response curve 的 five-(lambda)-point maximum mean 应当作为 descriptive response-curve summary，而不能暗示 untouched confirmatory HPO/test。
5. 是否同意：C-U1 near-retention formal comparison与 shared thresholded DRPO formula 应明确写成两个相关但不同职责的 parameterization，而不是把一个公式覆盖到另一个实验上。

---

## 7. 最终立场

GPT-6 这次 review 的价值是确实抓住了 R3 “删内部文档时把科学定义一起删掉”的风险。这个核心判断我同意。

但全文通读后的结论是：**现在最大的风险已经不只是 Appendix 写法，而是几张主图和主表的 provenance / evidence-status 与论文叙述不一致。**

因此正确顺序应当是：

> 先把 result source、branch identity、statistic definition 和 evidence strength 对齐；再做 paperization 压缩。

不能反过来先把半闭合结果写成漂亮论文语言，再让文字掩盖底层不一致。
