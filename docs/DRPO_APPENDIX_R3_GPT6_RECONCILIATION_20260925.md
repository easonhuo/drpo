# DRPO Appendix R3 / GPT-6 复核反馈（全文通读后版本）

## 0. 复核基准

本轮不是对 GPT-6 意见做局部逐条接受，而是先完整通读当前论文，再从正文 claim、附录定义、真实实验协议、结果 provenance 四个层面交叉核对。

- Repository / branch：easonhuo/drpo / main
- 审查基准 commit：e4976edaa4577b248646bfbe9892cd0c108985a2
- 审查基准 manuscript blob：ffd987fb4358f92ef98eb32ad6a3c2cb40774fc4
- GPT-6 反馈对象：DRPO_Appendix_R3_Review_Feedback_20260925.md
- R3 原方案对象：DRPO_ICLR2027_实验附录论文职责审计_R3_20260925.md

我已经从 Abstract 到 Appendix 最后一节完整通读 paper/iclr2027/manuscript_source.tex，并额外核对 experiments/registry.yaml、C-U1 E3 runner / formal outputs、E8 gradient-probe 实现与配置、D4RL development-closure 文档以及 figure manifests。

总体判断：

> GPT-6 对 R3 的六项主要意见中，第 1、3、6 项基本成立；第 4、5 项方向成立但真实问题比其描述更深；第 2 项“发现冲突”成立，但其建议的“直接把 Figure 6.3 写成 fixed-variance branch”不能安全采用。
>
> 更重要的是，全文通读后又发现了几项 GPT-6 没指出、但优先级更高的结果 provenance / evidence-status 问题。若不先解决这些问题，只把 Appendix 写得更像论文，会把尚未闭合的结果包装得更正式。

---

## 1. 对 GPT-6 六项意见的逐项结论

### 1.1 C-U1 距离定义和 5.0 阈值：同意，已修

R3 原来的整段替换会同时删掉真正的实验定义：

~~~
d_theta(s,a) = ||a - mu_theta(s)||_2 / sigma_theta(s)
near/far threshold = 5.0
~~~

这不是内部 bookkeeping，而是 causal intervention 的必要定义。当前 src/drpo/drpo_cu1_e1_e4_oneclick.py 也确认 E3 使用动态 standardized distance，阈值固定为 5.0。

已在 draft manuscript 中：
- 去掉 E1/E2/E3/E4 内部编号映射；
- 保留 standardized-distance 定义；
- 保留 5.0 threshold；
- 改成 source-isolation / Positive-only reference / causal-transmission / taper comparison 的论文职责叙述。

这一项同意 GPT-6。

### 1.2 Fixed variance vs. learnable variance：同意发现冲突，不同意直接指定 Figure 6.3 = fixed variance

GPT-6 正确发现两处文字冲突：

- C-U1 环境写 main continuous mechanism results 默认 fixed variance；
- Causal-Transmission 又写 mean and variance jointly evolve。

但不能因此直接把 Figure 6.3 声明为 fixed-variance branch。

原因是当前 Figure 6.3 底层结果与 current formal E3 不一致。

当前 figure source：
results/FIGURE2_CONTROLLED_SOURCE_TRANSMISSION/fig_6_3_2_causal_summary.csv

其中大致是：
- baseline final reward ≈ 0.201，task collapse 19/20；
- far-zero ≈ 0.618；
- far-cap ≈ 0.666；
- far-to-near ≈ 0.285，collapse 13/20；
- CSV 还记录了随方法变化的 final_sigma_r。

而 current formal C-U1-E3-ADAM-RERUN / outputs/cu1_e3_adam 中 fixed-variance branch 是：
- baseline / near-zero ≈ 2e-6，20/20 collapse；
- far-zero ≈ 0.739；
- far-cap ≈ 0.733；
- global ≈ 0.599；
- far-to-near ≈ 0.875，0/20 collapse。

learnable-variance branch则承担 support contraction 结果。

因此问题不是一句 caption 能解决，而是 Figure 6.3 必须先重新绑定到明确的 experiment / branch / result source。

当前 formal E3 更准确的职责是：
- fixed-variance branch：隔离 mean drift 与 task-performance collapse / rescue；
- learnable-variance branch：检查同一 far-field pathway 是否传导为 support / variance contraction；
- NaN/Inf 始终单独报告。

已在 draft manuscript 中把环境和 causal protocol 改成上述双分支职责；但没有把现有 Figure 6.3 强行标成 fixed variance。

对 GPT-6 的反馈是：发现的问题成立，但建议的具体修法不安全。

### 1.3 SG 统计定义：同意，而且问题比“补一句定义”更强

GPT-6 正确区分了三种不同轴：

1. 训练长度：1,200 optimizer updates；
2. 每个 coefficient 的时间维统计：late-window Pass@8；
3. coefficient 轴上的 contiguous five-point window。

当前 E8 文档进一步确认：

- late-window = updates 800, 900, 1000, 1100, 1200 的均值；
- main table 的 five-point window 是 lambda 轴上的五个相邻 coefficient points，不是训练时间上的五个 checkpoint；
- 当前 coefficient-response 是 staged held-out response curve：后续 coefficient range 曾在看过前一轮 held-out curves 后扩展；
- 因此 combined response curve 不能写成一次性 untouched confirmatory test。

已在 draft manuscript 中：
- 明确 late-window = 800/900/1000/1100/1200；
- 将 1,200-step curves 明确为 fixed-horizon measurements；
- 不再把 endpoint 写成 convergence / steady-state 证据；
- 去掉若干内部 registered / governance 措辞。

但 main SG table 的证据强度问题没有用一句话掩盖：它仍需单独决定应该作为 descriptive response-curve summary，还是需要真正独立的 final confirmation。

### 1.4 Controlled Protocols：同意缺定义，但真实 formal implementation 必须优先

#### 1.4.1 Far-cap / Global-matched / Far-to-near

GPT-6 正确指出正文报告了 far-field budget transfer，而 Appendix 没有定义。

进一步核对 current formal E3 runner 后发现，现稿中的 Far-cap 数学形式本身也不是 current implementation。

真实实现先构造当前 minibatch aggregate negative gradients：

~~~
G_N = aggregate near-field negative gradient
G_F = aggregate far-field negative gradient
~~~

Far-cap：

~~~
gamma_cap = min(1, 0.05 * ||G_N|| / (||G_F|| + eps))
G_far_cap = G_N + gamma_cap * G_F
~~~

Global-matched：

~~~
alpha_match =
    ||G_far_cap|| / (||G_N + G_F|| + eps)

G_global =
    alpha_match * (G_N + G_F)
~~~

Far-to-near：

~~~
choose c_N >= 0 such that

|| c_N * G_N + gamma_cap * G_F ||
    =
|| G_N + G_F ||
~~~

也就是说，Far-to-near 把被 far cap 移掉的 raw-gradient budget 转给 near component。

这里匹配的是 raw aggregate gradient norm，不是 Adam parameter-update norm。

已在 draft manuscript 中按 current formal implementation 重写这三项。

#### 1.4.2 Task-performance collapse / support boundary

当前 formal C-U1 E3 可精确定义：

- evaluation cadence = every 100 optimizer updates；
- task-performance collapse：held-out-context reward 持续低于 branch-initial reference 的 45%；
- sustained = 连续 3 次 evaluation；
- learnable-variance support boundary：log sigma < -12；
- NaN/Inf numerical failure 单独报告。

已在 draft manuscript 中补齐。

#### 1.4.3 Reward retention / policy shift

GPT-6 指出定义不足是对的，但不能先“猜一个合理定义”再写论文。

原因是：
- Figure 6.3 结果 provenance 尚未对齐 current formal E3；
- Figure 6.4.1 phase transition 在 repo 中仍明确标为 template / placeholder。

C-U1 runner 内确实已有 normalized_extrapolation_displacement，可作为 future phase-transition figure 的候选 policy-shift metric；但在 figure source 未闭合前，不应该假装当前图已经使用该定义。

因此正确顺序应是：先绑定真实 result source，再写 metric definition。

#### 1.4.4 Near-field-retention matching

GPT-6 指出一个真实逻辑问题：如果 near set N 完全处于 shared DRPO formula 的 untapered region，那么所有 selective taper 在 N 上权重都等于 1，near-retention matching 会退化成恒等式。

但实际 formal C-U1 near-retention experiment 使用另一套明确 parameterization：

- coordinate：direct standardized distance d；
- reference / near boundary：d_ref = 5.0；
- main target average near retention：0.75；
- sensitivity targets：0.50 / 0.25；
- calibration tolerance：1e-6；
- calibration set：development seeds 0--4 的 Positive-only-2000 checkpoint 上 d <= 5 的 pooled near negatives；
- calibration 后 coefficients freeze，再进入 formal seeds。

所以 C-U1 controlled taper comparison 并不是 current shared-method subsection 里 thresholded excess-remoteness formula 的直接实例化。

这里真正要修的是：
> 明确区分 “controlled C-U1 taper-shape mechanism comparison” 和 “shared external DRPO deployment formula”。

我没有在本轮 draft 中擅自把两套 parameterization 强行合并，因为这是科学映射，不是文字清理。

### 1.5 D4RL 参数说明：同意局部冲突，但更大的问题是正式结果 provenance

GPT-6 指出的两个局部问题成立：

1. “threshold and scale kept fixed across methods” 与 task-specific coefficient grids 冲突；
2. alpha_Exp = 1 在当前 shared method formulas 中没有明确角色。

已在 draft manuscript 中做的安全修复：
- 改成 remoteness coordinate convention 统一；
- fixed coordinate settings 与 tuned weighting parameters 分开说明；
- 删除多余的 registered / config-population 内部语言；
- 简化 protocol-matched comparison prose。

没有做的事情：
- 没把 alpha_Exp 猜改成 lambda；
- 没把 development grid 猜填成 formal protocol table。

因为 current registry 显示更大的 blocker：

EXT-H-E7-BENCH-01 当前：
- formal status = not_run；
- execution gate = blocked；
- exact D4RL versions / formal seeds / base algorithm / optimizer / full budgets 仍待 formal lock。

development GLQ tuning closure 明确写：
- scientific_status = pilot；
- held-out seeds untouched；
- selected cells fixed-horizon inconclusive；
- cross_method_ranking_allowed = false；
- formal_d4rl9_table_population_allowed = false；
- held_out_confirmation_claim_allowed = false。

另外，当前 manuscript 的 DRPO D4RL task values（包括 total 698.8）在 current repo code/document search 中只找到 manuscript / replacement manuscript，并没有找到一份一一对应的正式 result artifact。

因此当前最优先的问题不是“把 grid table 写完整”，而是：
> 当前 main paper 的 D4RL 698.8 table 是否有可审计的正式 result provenance。

在这个问题闭合前，把 development grid 写得更完整会让尚未闭合的 result table 看起来更正式，但不会让证据变强。

### 1.6 Structured-generation gradient diagnostic：同意，已按真实实现补齐

GPT-6 提出的三个缺口都成立：

- 梯度到底对哪个 scalar 求导；
- fixed probe checkpoint 是什么；
- 95% CI 怎么算。

当前实现可以精确回答。

Gradient scalar：

~~~
mean_log_prob(x,y)
    = (1/T) * sum_t log pi_theta(y_t | x, y_<t)

G_SG(x,y)
    = || grad_{trainable parameters} mean_log_prob(x,y) ||_2
~~~

因此不是 summed sequence log probability 的 gradient。

八个非-Countdown multitask probes：
- task-specific Positive-only warm-start checkpoint；
- 100 optimizer updates；
- 256 negative responses / task；
- matched absolute advantage = 1.0；
- 10 equal-count surprisal bins；
- prompt-within-task cluster bootstrap；
- 2,000 replicates；
- bootstrap seed 161803。

Countdown full-bank probe：
- seed 100；
- 6,000 puzzles；
- near/far 各一条，共 12,000 responses；
- SFT/reference adapter；
- absolute negative coefficient = 1.0；
- 10 equal-count bins；
- puzzle-id cluster bootstrap；
- 400 replicates；
- single-seed external diagnostic；
- probe 不执行 optimizer update。

已在 draft manuscript 中补齐上述定义。

全文通读还发现一个 GPT-6 没指出的直接错误：
- canonical main Figure 1 plotting script 的 panel (b) 实际是 Countdown full-bank；
- 旧 manuscript caption 却写成 “structured-generation tasks in (b)”。

已在 draft manuscript 中修正：
- main panel (b) = Countdown full-bank probe；
- 其余 structured-generation task-level curves在 appendix panel 中报告。

---

## 2. 全文通读后新增的高优先级问题

这些问题比“Appendix 有没有文档味”更严重，因为它们直接涉及 paper claim 与 repository evidence 是否一致。

### 2.1 Figure 6.4.1 phase transition 当前是 template，却被正文按 empirical result 解读

Repository 的 scripts/figures/README.md 和 paper/figures/FIGURE_INDEX.md 明确记录：

- fig_6_4_1_phase_transition = template/layout；
- formal phase-scan aggregates pending；
- template numbers 不应被解释成正式实验结果。

对应数据：
results/FIGURE3_PHASE_TRANSITION/fig_6_4_1_phase_transition_template.csv

其中每行都有 is_placeholder = 1。

但当前正文写：
> Figure ... shows a non-monotone transition ...

并进一步解释 task collapse / boundary events。

这个问题优先级高于 Appendix paperization。最终论文必须用真实 aggregate 替换，或者把图降级为 protocol illustration；不能继续把 placeholder 当 empirical result。

### 2.2 Controlled taper Figure 同样是 template，却被正文/附录当作结果

当前：
results/FIGURE4_TAPER_CONTROL_TRANSFER/fig_6_4_2_leftfig_template.csv

以及 repository figure docs 都明确把它标为 template/layout，formal 6.4.2 / 6.4.3 aggregates pending。

但 manuscript 现在写：
> Under both matched comparisons, exponential tapering ...

这已经是结果 claim。

同样必须先替换为真实 formal aggregate，或者降级为 protocol illustration。

### 2.3 Figure 6.3 artifact 与 current formal E3 不一致

如 1.2 所述，Figure 6.3 当前 CSV 与 current formal C-U1-E3-ADAM-RERUN 在 baseline、far-to-near、support dynamics 等处存在实质差异。

因此必须先回答：
> Figure 6.3 展示的是哪个 experiment / branch / commit？

在 provenance 对齐前，仅修改 caption 不够。

### 2.4 D4RL main table 与 current formal registry 状态冲突

Current manuscript 报告 DRPO D4RL-9 total 698.8。

但 current formal D4RL-9 benchmark EXT-H-E7-BENCH-01 仍是 not_run / blocked；development tuning closure 明确禁止 formal table population 和 cross-method ranking。

因此必须：
- 找到 698.8 对应的独立、可审计、允许进入论文正式表格的 result source；
- 或重新界定 / 移除当前 D4RL formal-table claim。

完善 Appendix 参数表不能替代这一步。

### 2.5 SG main table 的 window-max task score 需要重新界定 evidence strength

Current SG coefficient-response evidence：
- fixed 1,200-step finite horizon；
- staged held-out response curves；
- coefficient range 曾在观察早期 held-out curves 后扩展；
- current docs 不允许把它包装成 convergence / steady-state / untouched confirmatory ranking。

因此，把每任务 lambda-response curve 压成一个 maximum contiguous-five-point-window mean 后再进行跨方法计数，至少应明确为 descriptive response-curve summary，而不能让读者理解为独立 untouched confirmatory HPO + final test。

### 2.6 C-U1 controlled taper 与 shared DRPO formula 不是同一 parameterization

Formal C-U1 near-retention comparison：
- direct standardized distance d；
- d_ref = 5；
- matched average near retention。

Shared external DRPO method：
- thresholded normalized excess remoteness / surprisal coordinate；
- explicit untapered near field。

二者可以承担互补职责，但不能在 Appendix 中写成同一个公式的无缝复用。

---

## 3. 对 R3 清理方向的最终判断

R3 的大方向仍然正确：

- 删除 RunSpec、artifact-field、internal package/provenance bookkeeping；
- 去掉只服务内部 roadmap 的 E1--E6 编号；
- 将 shared-condition checklist 改成论文 prose；
- 删除 TOPR internal pilot identity sentence；
- 删除 finite-measure 末尾孤立的 quality-selection 比较；
- 保留环境定义、真实 intervention、matching design 和三类 failure outcome 的区分。

但 R3 不能整段机械替换，因为以下三类内容不能被 paperization 吞掉：

1. 真实 protocol 公式和数值；
2. result-statistic 定义；
3. result provenance / evidence strength。

“删掉内部状态词”不等于“结果自动变成论文结果”。

---

## 4. 本轮已经实际修改的 manuscript 内容

Draft branch：dev/iclr-appendix-r3-reconcile

已经修改：

- 保留 C-U1 standardized distance 与 threshold 5.0，同时去掉 E1--E4 内部编号；
- 把 C-U1 fixed/learnable variance 改成互补 causal branches；
- 去掉 D-U1 E5/E6-A/B/C 内部编号映射；
- 把 Shared Experimental Invariants checklist 改成 reader-facing controls prose；
- 按 current formal implementation 重写 Far-cap / Global-matched / Far-to-near；
- 补 C-U1 task-collapse 45% × 3 consecutive evaluations 和 log-sigma < -12 support-boundary definition；
- 精确定义 SG gradient scalar、probe checkpoint、sample count、bootstrap unit / replicates；
- 修正 main Figure 1 panel (b) 实际为 Countdown full-bank，而不是 nine-task SG aggregate；
- 明确 SG late-window = updates 800/900/1000/1100/1200；
- 去掉多处 registered / paper-facing / RunSpec / internal provenance prose；
- 删除 finite-measure 末尾孤立的 quality-selection comparison；
- 将 D4RL “所有 scale 都固定”改成“coordinate convention 统一；fixed settings 与 tuned weighting parameters 分开”。

这些修改不改变任何实验数据、seed、threshold、训练 horizon 或方法结果。

---

## 5. 本轮刻意没有自动修改的内容

以下不是遗漏，而是当前 evidence 不足以安全落笔：

1. 没有把 Figure 6.3 标成 fixed-variance：figure artifact 与 formal E3 还没对齐。
2. 没有给 Figure 6.4.1 / controlled taper template 猜正式结果。
3. 没有给 D4RL task-specific grid 猜 candidate values，也没有把 alpha_Exp 猜改成 lambda。
4. 没有把 D4RL 698.8 或 SG window-max table 重新包装成 formal ranking。
5. 没有强行把 C-U1 controlled taper formula 改成 shared DRPO formula。

---

## 6. 请 GPT-6 下一轮重点复核的五个问题

1. 是否同意：Figure 6.3 必须先完成 artifact-to-formal-E3 provenance 对齐，才能决定 caption 中的 variance branch。
2. 是否同意：Figure 6.4.1 / controlled taper figure 在 repository 明确标为 template 的情况下，当前正文不能继续把它们当正式 empirical result。
3. 是否能从 current repository 找到 D4RL 698.8 的可审计正式 result source；如果找不到，是否同意它不能继续按 formal D4RL-9 task table 叙述。
4. 是否同意：SG staged-heldout response curve 的 five-lambda-point maximum mean 应被界定为 descriptive response-curve summary，而不能暗示 untouched confirmatory HPO/test。
5. 是否同意：C-U1 near-retention formal comparison 与 shared thresholded DRPO formula 应明确写成两个相关但职责不同的 parameterization，而不是用一个公式覆盖另一个实验。

---

## 7. 最终立场

GPT-6 这次 review 的核心价值是抓住了 R3 “删内部文档时把科学定义一起删掉”的风险。这个判断我同意。

但全文通读后的结论是：

> 当前最大的风险已经不只是 Appendix 写法，而是几张主图和主表的 provenance / evidence-status 与论文叙述不一致。

因此正确顺序应当是：

> 先对齐 result source、branch identity、statistic definition 和 evidence strength，再做 paperization 压缩。

不能先把半闭合结果写成漂亮论文语言，再让文字掩盖底层不一致。
