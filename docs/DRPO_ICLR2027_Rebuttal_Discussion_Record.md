# DRPO ICLR 2027：四位审稿人原始意见、中文回复稿与核对记录

版本：v35  
更新日期：2026-11-15（北京时间）  
论文：Breaking the Curse of Repulsion in Off-Policy Reinforcement Learning  
投稿：ICLR 2027，Submission 30229  
仓库：easonhuo/drpo  
本轮读取的 main 基线：`b949e60e68d1f1832a6e06f94a3c38a32b43d7b2`  
本次协作分支：`dev/iclr2027-rebuttal-record`  
仓库文件：`docs/DRPO_ICLR2027_Rebuttal_Discussion_Record.md`

**本文件的用途：供多个 session 共用审稿原文、已确认的中文回复和讨论进度；先完成完整中文回复，再统一翻译成英文。它是审稿回复工作文件，不替代唯一研究主文档 `docs/handoff.md`，也不是已向 OpenReview 提交的回复。**

**完成范围：**YPds 的 W1 四个方面、W2 的机制证据与收益／不确定性，以及 Q1 的优势刷新和重要性修正已有完整中文回复；外部 q/p 轨迹和匹配总负梯度的 Q2 仍待补充结果。Uwf4 已整合记号呈现、研究动机、TOPR／IS 方法与理论区别、行为概率估计、Positive-only 收益及统计补充的完整中文回复，待用户检查。aYhA 的完整中文回复已经用户确认并整合；iW2z 的表 2 不确定性回复与两张表格已确认并整合，所提供的 SD 已补齐并重算九任务汇总；D4RL 表现与方法实际价值的完整中文回复也已确认并整合；Q1–Q7 的完整中文回复也已全部确认并整合；其余 Weaknesses 由对应 session 继续逐条处理。文末核对记录和历史存档不进入审稿回复。

## 四位审稿人的原始意见

来源：用户上传的 OpenReview 粘贴文本 `Pasted text.txt`。以下按 YPds、Uwf4、iW2z、aYhA 保留四份 review 的 Summary、评分、Strengths、Weaknesses、Questions、Rating 和 Confidence；仅去掉页面导航及 review 之间的 Add 按钮文本。**该粘贴文本的部分行内公式／希腊字母为空白，以下保留这些空白，不根据猜测补写审稿人的原话。**需要辨认缺失符号时，应对照投稿 PDF 的对应位置，不将推测写成原文。

<a id="review-ypds"></a>

### Reviewer YPds：原始 review 全文

> Official Review of Submission30229 by Reviewer YPds
> Official Reviewby Reviewer YPds03 Nov 2026, 22:53 (modified: 06 Nov 2026, 20:27)EveryoneRevisions
> Paper Summary:
> This paper studies how repeatedly reusing negative-advantage samples can push a policy away from historical actions and destabilize learning. It proposes DRPO, which downweights negative updates when the corresponding actions are already unlikely under the current policy.
>
> Presentation: 2: Generally understandable, but requires substantial effort in places.
> Informativeness: 1: I learned little or nothing new from this paper.
> Soundness: 2: I have minor or fixable concerns about the paper’s soundness.
> Critical Strengths:
> S1: The analysis clearly separates advantage magnitude from policy-score magnitude and distinguishes Gaussian gradient amplification from bounded categorical suppression.
>
> S2: The controlled interventions support the proposed mechanism.
>
> Critical Weaknesses:
> W1: The theory assumes fixed data and advantages and analyzes direct policy-coordinate updates without importance correction. Its relevance to practical off-policy learning with changing critics, shared neural parameters, and data refresh remains unclear.
>
> W2: The external experiments do not directly establish that the proposed mechanism causes training instability. Gradient probes show a near-to-far pattern, but the causal interventions are confined to controlled environments. The modest structured-generation gains also lack uncertainty estimates in the main table.
>
> Questions For Authors:
> Q1: Does the predicted instability persist when advantages are refreshed or off-policy corrections are applied? What are the observed (q/p) trajectories in the external experiments?
>
> Q2: Does removing or capping far-field negative updates rescue D4RL or structured-generation training when total negative-gradient magnitude is matched?
>
> Flag For Ethics Review: No
> Rating: 2: Weak rejection
> Confidence: 3: I am familiar with this topic and its relevant literature.

<a id="review-uwf4"></a>

### Reviewer Uwf4：原始 review 全文

> Official Review of Submission30229 by Reviewer Uwf4
> Official Reviewby Reviewer Uwf401 Nov 2026, 13:16 (modified: 06 Nov 2026, 20:27)EveryoneRevisions
> Paper Summary:
> This work studies off-policy RL in which the same historical negative-advantage samples are reused repeatedly as the policy changes.
>
> The authors introduce “remoteness” (the negative log-probability of the action under the current parametric policy). They argue (assuming frozen data and fixed advantages) that repeated negative updates on a fixed sample make it increasingly "remote" without weakening its gradient, creating a self perpetuating loop that pushes gradients arbitrarily large for Gaussian policies, drives suppressed action probabilities toward zero in categorical policies.
>
> To address this, the authors propose Dynamic Remoteness Aware Policy Optimization (DRPO). DRPO leaves positive updates and low-remoteness negative updates unchanged, and exponentially decays negative updates whose remoteness under the current policy exceeds a given threshold.
>
> Empirically, synthetic environments separate remoteness from advantage magnitude and show that removing far field (but not near field) negative updates prevents collapse due to remote samples. On task performance, DRPO achieves the highest scores (in aggregate) among protocol-matched negative-weighting variants (e.g Positive only, Global 
> ) on D4RL and on structured generation with Qwen2.5-0.5B-Instruct, and is competitive with published offline-RL baselines on D4RL.
>
> Presentation: 1: Difficult to follow despite careful reading.
> Informativeness: 1: I learned little or nothing new from this paper.
> Soundness: 3: I have no substantive concerns about the paper’s soundness.
> Critical Strengths:
> None
>
> Critical Weaknesses:
> (Bad presentation) Paper was hard to follow. Too much notation was introduced, even for simple things, and never reused later. Motivation of the method was highly unclear, and the relationship to existing literature under-emphasised.
>
> (Insufficient novelty) Eq. 1 cites the policy-gradient literature (e.g Sutton et al. 2000), but it defined on off policy samples without importance weights, which would omit a crucial correction term. TOPR (Le Roux et al. 2025) is explicitly reintroduces this correction term on negative advantages (which is what DRPO does as well). The TOPR paper also explores the instability of off-policy negative updates (which would be the key theoretical contribution here).
>
> (Insufficient empirical performance) In practice, the methods seems to yield marginal gains compared to positive advantage only (see Tables 1 and 2 and Figure 6). No confidence intervals in the tables preclude statements about the significance of the results.
>
> Questions For Authors:
> Could the authors clarify whether the far-field instability is distinct from the omission of importance correction?
>
> If it is not distinct, is the justification for DRPO that the data-generating policy is not always known? Alternatively, is there a setting where remoteness-based weighting is preferable to ratio-based weighting even when the data-generating policy is known?
>
> If the justification is that the data-generating policy is not known, wouldn't estimating it from the dataset (e.g behaviour cloning in D4RL, and the frozen initial policy in structured generation), and using it to do clipped importance correction (approximately) achieve the desired result?
>
> Flag For Ethics Review: No
> Rating: 1: Clear rejection
> Confidence: 2: I have broad familiarity with this field but little or no direct research experience on this topic.

<a id="review-iw2z"></a>

### Reviewer iW2z：原始 review 全文

> Official Review of Submission30229 by Reviewer iW2z
> Official Reviewby Reviewer iW2z22 Oct 2026, 05:50 (modified: 06 Nov 2026, 20:27)EveryoneRevisions
> Paper Summary:
> This paper identifies an instability issue that arises from offline / off-policy policy gradient update schemes, where negative examples can induce unbounded logits. The paper then proposes a method of tapering the updates for negative examples with an exponential in the policy log probabilities to alleviate this issue. This results in a practical algorithm which is deployed on D4RL and a structured sequence generation task, achieving strong performance.
>
> Presentation: 1: Difficult to follow despite careful reading.
> Informativeness: 1: I learned little or nothing new from this paper.
> Soundness: 1: I have significant concerns about the paper’s soundness that do not appear easy to address.
> Critical Strengths:
> The paper studies an important and timely problem, that being practical policy optimization algorithms that consume only offline data, as well as the particularly interesting phenomenon that occurs as a byproduct of policy gradient optimization. The proposed solution is simple to implement, and appears to achieve very strong structured generation performance when applied to a small language model.
>
> Critical Weaknesses:
> The mathematical content of the paper is effectively incomprehensible due to terms that aren't defined properly – see the Questions section. In addition, I also don't understand the setting of the mathematical analysis, particularly where advantages are kept frozen over the course of policy optimization, which in principle impacts the optimization objective.
>
> Moreover, the experimental results leave a bit to be desired; results aren't shown with confidence intervals for instance. Moreover, the D4RL result doesn't look particularly impressive (indeed, I'd be curious to see a proper comparison, like via rliable for example), and the structured generation result is only on a small language model. I get that this was meant to propose some theory / new algorithms and that empirical results are not the main focus, however I have two concerns with this:
>
> This issue of instability in off-policy policy gradient settings was already discussed in the paper introducing TOPR (Le Roux et al., 2025). This paper wasn't really mentioned in any of the background.
> While TOPR also proposed a (in fact similar) method of addressing this issue, the paper doesn't compare against TOPR. For the structured generation experiment, the paper compares against a strange variant of TOPR that I haven't seen in the literature, and the D4RL experiment doesn't compare to TOPR at all.
> So, overall, the theoretical issue is something that is already understood to some degree, the mathematical contributions are extremely difficult to parse, and the experimental results leave much room for improvement.
>
> Questions For Authors:
> At the beginning of section 4.1, it says "To isolate action-side geometry, fix context 
>  and write 
>  for its policy-output coordinates". What does this mean? Particularly, I don't know what action-side geometry means, and I don't know what policy-output coordinates are. As such, I really don't know how to interpret 
> . What's the difference between 
>  and policy parameters? Likewise, towards the end of this paragaph, what is "output-space update magnitude"?
>
> I am confused about what the "signed coefficient" 
>  represents. I would have expected it to be the advantage of a state-action pair for the policy being learned. But then in section 4.2, you say that 
>  remains fixed over the reuse window. Why? What is the interpretation of equation 4 then?
>
> In Theorem 4.5, the categorical case introduces a new assumption that the logits sum to 1. But the gradient update schemes analyzed previously don't preserve this property. So what should one conclude from this part of Theorem 4.5?
>
> I don't understand at all what 
>  are. They are described as the "positive-only mean-parameter target" and the "magnitude-weighted negative sample moment", respectively, but these are not precise terms and I genuinely don't know how to interpret them (or even what space they live in).
>
> On line 236, it refers to "Gaussian policies with finitely many negative atoms". What does this mean? Gaussian policies are atomless.
>
> In Corollary 5.3, what is 
> ? Above, this symbol is described as "a bounded region representing historical data coverage or desirable task performance", but these things don't have the same "type" as each other, nor is it even clear to me what either of those are. Moreover, Corollay 5.3 asserts that 
> . While again I don't understand what that term is, I don't see how "positive-only mealn parameter target" is an element of type "historical data" or "desirable task performance".
>
> What is "Joint fitted reference 
>  TOPR"? I see some description of this in the appendix (not mentioned in the main text as far as I can tell). But I don't understand why you compare against this bespoke TOPR variant rather than TOPR itself.
>
> Flag For Ethics Review: No
> Rating: 1: Clear rejection
> Confidence: 3: I am familiar with this topic and its relevant literature.

<a id="review-ayha"></a>

### Reviewer aYhA：原始 review 全文

> Official Review of Submission30229 by Reviewer aYhA
> Official Reviewby Reviewer aYhA21 Oct 2026, 01:32 (modified: 06 Nov 2026, 20:27)EveryoneRevisions
> Paper Summary:
> The paper works in the setting of off-policy reinforcement learning, where a distribution 
> , different from the policy distribution, is used to compute the expected reward and score function. It notes that as the policy changes, the same negative sample from 
>  can be revisited under a different current policy. For a score function used has the form 
> , it shows that revisiting an action with a negative advantage-signal (
> ) makes it increasingly improbable under the current policy (they introduce the remoteness 
>  to quantify this), while the size of updates from the negative action does not decrease. There is also a contrast with positive actions, for which the size of the updates decreases monotonically under reuse. Thus repetition sustains or amplifies repulsion from negative actions, causing instability in training. The paper proves that: For Gaussian policy, negative feedback dominating positive feedback can make the policy move indefinitely farther away from the stationary point, with unbounded score response from every fixed action. For Categorical policy, the score response remains bounded, but the probabilities can still be pushed toward zero. The learning process need not settle at a finite policy parameter. To mitigate these issues, the authors introduce Dynamic Remoteness-Aware Policy Optimization (DRPO), which multiplies negative actions by a thresholded exponential taper to attenuate their effect. The weight is a function of the remoteness 
>  of the action, and decays exponentially for large values of 
> . The theoretical results prove that such a tapering scheme can mitigate the instability for both Gaussian and Categorical policies.
>
> The experimental section shows the instability in training in practice. Figure 1 and 2 show that large updates from negative samples occur in practice, and establish that they arise from the gradients 
>  rather than from larger advantages. Figure 3 shows the effect of negative dominance on the held-out reward. The authors also perform comparisons with other algorithms, notably the closely related 
> -TOPR on various tasks. DRPO shows competitive overall benchmark results.
>
> Presentation: 2: Generally understandable, but requires substantial effort in places.
> Informativeness: 3: I gained a substantially deeper understanding of the problem or question.
> Soundness: 3: I have no substantive concerns about the paper’s soundness.
> Critical Strengths:
> The theoretical results in section 4 present a complete picture of the issue of self amplification under reuse of negative samples, the effect of the dominance of negative samples on the learning dynamics, and the control implication (form of the taper required to mitigate the issues). The issue of self amplification of negative samples has been addressed in literature before, but this is a novel contribution which analyzes the issue theoretically and proves why an exponential taper is chosen for mitigating the issue.
> The experiments in Section 6.1 demonstrate the issue in practice. The controlled experiments are especially instructive to understand the source of the large updates from negative samples. Also, the controlled experiments in the bottom panel of Figure 2 show that the far-field negative updates, not negative feedback generally, causes the instability in C-U1 experiments.
> Critical Weaknesses:
> The issue that concerns me the most is the comparison between DRPO and 
> -TOPR. Looking at appendix E.8, in particular, the formula for 
> , it seems that the only difference between DRPO and 
> -TOPR is the extra threshold parameter 
>  (note that 
>  and 
>  can be absorbed into the parameter 
> ). Looking at Table 2 (b), we see that DRPO only provides a minor improvement over 
> -TOPR. As such, I see this paper as being weak methodologically, even though the theory portion is strong.
> Questions For Authors:
> Clarification: Could the authors explain intuitively why DRPO shows a slight improvement over 
>  TOPR even though they have very similar functional forms of the tapering?
> Flag For Ethics Review: No
> Rating: 3: Weak acceptance
> Confidence: 2: I have broad familiarity with this field but little or no direct research experience on this topic.

---

## 多 session 阅读入口与当前进度

| Reviewer／问题 | 当前状态 | 可复用内容／剩余事项 |
| --- | --- | --- |
| YPds W1：固定数据、固定优势／critic、共享神经参数、无显式 IS | 中文回复已有 | 见下方 W1 四个子问题；保持已确认文字 |
| YPds W2：外部机制证据 | 中文回复已有 | 受控因果设计与补充 Hopper case；事件类型分报 |
| YPds W2：收益幅度与不确定性 | 中文回复及五方法九任务统计已更新，后续独立运行仍待补 | 按 W2 原文聚焦结构化生成；补齐 Positive-only／DPO；区分 seed SD 与配置 SD |
| YPds Q1：优势刷新、off-policy correction | 中文回复已有 | 外部原始／有效 q/p 轨迹待结果 |
| YPds Q2：总负梯度匹配的 far-zero／far-cap | 方案已讨论；结果待补 | 与新的 36-run Hopper 方案对应，不用旧 case 冒充新匹配实验结果 |
| Uwf4 W1（1）：记号呈现 | 中文回复已确认并保留 | 本条已完成讨论；论文修改与回复整理分别跟踪 |
| Uwf4 W1／W2、Q1–Q3：动机、TOPR／IS 与理论增量 | 完整中文回复已整合；待用户检查 | 保留直观比较、并列权重公式、平衡公式、两条解决路线、已知／未知行为策略及拟合参考对照 |
| Uwf4 W3：收益幅度与统计补充 | 完整中文回复已整合；待用户检查 | 保留两项结构化生成收益、两项历史 D4RL 扫描、Positive-only／DRPO 统计表及图 6；宏平均和平均 SD 在同一行分别标注 |
| iW2z：不确定性报告 | 回复文字及两张表格已整合，所提供 SD 已补齐 | Table 2(a) 按本轮数据更新并新增 BC；Table 2(b) 补齐 DPO／Countdown；重算九任务汇总 |
| iW2z：D4RL 表现与方法实际价值 | 完整中文回复已确认并整合 | 保留已有代表性路线与不同控制路径两句；说明简洁性、跨场景适用性、行为概率信息需求及三类组合空间 |
| iW2z：Q1–Q7 | 完整中文回复均已确认并整合 | 动作侧几何／输出坐标／更新幅度、固定优势与公式（4）、logit 零和约定、正负样本中心、负原子、有界区域及拟合参考 TOPR |
| iW2z：其余 Weaknesses | 原文已收录；继续逐条核对 | 复用已确认内容；canonical TOPR 直接比较及其余实验请求仍单独跟踪 |
| aYhA：方法区别及性能解释 | 完整中文回复已确认并整合 | 已覆盖阈值／参数合并、行为／参考概率依赖、动力学贡献及有益／过度外推解释 |

共享文件仍为这一份，不按 W1／W2 或 reviewer 拆成多个文件。多个 session 可按 reviewer 分工；写回前读取该分支最新文件，仅合并本 session 负责的段落，保留其他已确认回复和历史。讨论稿、实验待办与可给审稿人的完整中文文字分别标明状态。

Hopper 补充实验的实现讨论位于已有分支 `dev/e7-rebuttal-pq-is`、[PR #424](https://github.com/easonhuo/drpo/pull/424)。36-run 方案 ID 为 `EXT-H-E7-REBUTTAL-PQ-IS-01`；本次只核实该分支／PR 已存在，未审计其最新实现或训练完成情况。本文件不升级实验结果状态。main 中既有 `EXT-H-E7-Q2` 的 long-run closure 与这轮新补充实验不是同一次运行。

---

## 中文回复正文

感谢审稿人认真审阅本文，并就理论分析与实际训练的联系、外部机制证据以及实验结果的不确定性提出具体意见。下面我们分别回应这些问题。

### W1：理论设定与实际 off-policy 学习的相关性

> The theory assumes fixed data and advantages and analyzes direct policy-coordinate updates without importance correction. Its relevance to practical off-policy learning with changing critics, shared neural parameters, and data refresh remains unclear.

感谢您指出需要进一步说明分析设定与实际训练之间的联系。本文研究的是：当前策略不断变化时，历史负反馈被反复复用，如何通过当前策略下的 remoteness 影响更新，并进一步形成排斥、漂移或支持收缩。固定反馈的理论分析、真实神经策略的梯度诊断、受控干预及外部任务评估，分别提供可解析机制、实际梯度证据、因果证据和任务效果。针对您列出的四个方面，我们具体说明如下。

#### （1）固定数据与数据刷新

感谢您关注固定数据设定与实际训练的联系。固定历史数据具有明确的应用需求：连续动作的 Offline RL 本来就从既有数据中学习策略，语言模型的 SFT 也通常基于预先收集的数据进行训练。在线收集一批反馈、随后离线进行多轮优化，同样属于历史数据复用。因此，固定数据训练是本文主动研究的实际应用场景。

以 SFT 为例，常规训练主要模仿选定的目标响应，没有显式利用错误响应提供的负反馈。本文结构化生成中的 Positive-only 对照只学习正确 completion，其目标对应 SFT 式监督似然学习。从训练形式上看，本文研究的更新是在监督似然学习中引入 reward／advantage 提供的正负反馈，并进一步按当前 remoteness 控制负反馈的影响。当只保留正样本并赋予统一正权重时，得到正样本上的监督似然训练；保留经过控制的负权重，则可以进一步利用错误响应中的排斥信息。我们与 Positive-only 的实验比较为这种利用负反馈改善学习的路线提供了支持。论文结论也已提出，用经过 remoteness 控制的 verified negative completions 扩展语言模型 SFT，是一个值得进一步研究的应用方向。

固定数据还对应一个有意义的极限设置。On-policy 学习要求采样策略与当前策略相匹配，及时刷新数据是维持这种匹配的方式。随着历史数据保留和复用时间延长，这些样本可能在策略已经发生较大变化后继续参与更新。沿着数据刷新频率这一维度，完全不刷新就是零刷新极限，代表一种极端的 off-policy 复用情形。本文选择这一设置，是为了研究历史负反馈持续复用时的极端表现，让其累计作用充分显现，并检验在不依赖数据刷新帮助的情况下能否控制这条失稳路径。原稿引言和结论均已明确这一 zero-refresh limit 定位。

数据固定时，策略仍然持续变化。同一个历史动作在当前策略下的概率、remoteness 和梯度响应都会随更新改变。一个负样本被排斥后可能变得更远，但其更新影响不一定相应减弱，反复复用因而可能形成持续排斥。固定数据使我们能够在保持样本质量和反馈构成不变的条件下，识别这一机制及其累计后果。

DRPO 的控制原则也不依赖数据始终固定：每次更新都依据样本在当前策略下的 remoteness，调节当时负反馈的影响。无论样本来自固定数据集、持续刷新的 replay buffer，还是较早策略采集的数据，同一条控制规则都可以使用。因此，零刷新设定是本文充分暴露机制并建立理论分析的研究情形；按当前 remoteness 控制远场负反馈的原则，也适用于有限复用、replay 和 stale-policy 等一般 off-policy 训练场景。论文结论已明确说明，remoteness-aware control 并不限于静态离线数据。

#### （2）固定优势与 critic 变化

感谢您关注固定优势分析与 critic 变化的联系。DRPO 使用的 remoteness 与优势的估计方式解耦：对于给定样本和当前策略，remoteness 直接由当前策略计算，不依赖 critic 的输出或优势的数值。Critic 等反馈来源负责提供学习信号，DRPO 则根据当前 remoteness 调节负反馈的更新影响。因此，DRPO 不要求使用特定的 critic，也可以用于不使用 learned critic 的训练。

更重要的是，远场机制不以 critic 的估计误差为前提。即使优势完全准确，远负样本的排斥机制仍可能存在。固定优势分析的目的，是保持反馈系数不变，单独研究策略变化带来的几何效应：同一个历史负样本被反复使用时，可能被逐步推远，而其梯度响应不一定随之减弱。这样可以将优势估计的变化与 remoteness 的作用区分开来。

论文已有实验也提供了具体佐证。D4RL 实验使用 critic 提供的优势，其中外部梯度诊断先拟合 critic，再计算并冻结优势；结构化生成则直接使用 verifier 提供的正负反馈，不训练 learned value critic。两类设置说明，DRPO 可以与 critic 产生的优势结合，也可以用于 critic-free 的反馈训练。相关反馈来源和训练设置已在论文实验部分及相应附录中说明。

如果 critic 更新后重新计算了优势，actor 接收到的反馈系数及其正负号可能改变。DRPO 可以直接使用刷新后的反馈，继续根据当前 remoteness 调节负分支的影响。固定优势用于隔离并分析机制，算法本身可以与优势刷新结合使用。

#### （3）直接策略坐标与共享神经参数

直接策略坐标用于得到可解析的几何与动力学关系。实际神经策略训练使用完整的参数梯度。令 \(u_\theta(s)\) 为网络输出的均值或 logits，链式法则给出

\[
\nabla_\theta\log\pi_\theta(a\mid s)
=
J_\theta(s)^\top
\nabla_u\log\pi_u(a).
\]

论文中的 full-parameter gradient 诊断测量的正是包含网络 Jacobian 的完整梯度。D4RL 诊断匹配负样本的绝对优势，结构化生成诊断固定负反馈系数，从而检查真实神经策略中梯度响应是否仍随 remoteness 呈现 near-to-far 模式。受控干预则进一步检查远场更新是否传导为策略漂移和训练失稳。

DRPO 的作用也直接延伸到完整参数梯度。按照论文第 5 节的 stop-gradient 定义，对同一个负样本，有精确关系

\[
g^-_{\theta,\mathrm{DRPO}}
=
\omega(D_\theta)\,
g^-_{\theta,\mathrm{raw}},
\qquad
g^-_{\theta,\mathrm{raw}}
=
-\widehat A^-\nabla_\theta\log\pi_\theta(a\mid s).
\]

因此，DRPO 衰减的是包含共享网络 Jacobian 的完整样本梯度。输出坐标分析提供机制，神经梯度诊断和干预检查其实际表现；这几部分共同建立了理论与神经策略训练之间的联系。

#### （4）没有显式重要性修正

感谢您关于重要性修正的提问。使用 off-policy 数据进行策略优化，并不强制要求每一种 actor 更新都显式乘以 importance ratio。利用历史数据拟合策略，与从历史数据无偏估计当前策略梯度，是不同的优化任务。AWR 提供了直接的同类先例：它采用优势加权的策略拟合目标

\[
\max_\theta
\mathbb E_{(s,a)\sim\mathcal D}
\left[
\exp\!\left(\frac{\widehat A(s,a)}{\beta}\right)
\log\pi_\theta(a\mid s)
\right],
\]

而非在 actor 更新中显式乘以 \(\pi_\theta/\mu\)。

本文第 3.1 节将 Eq. (1) 明确定义为 empirical signed actor field。固定历史分布 \(\nu\) 与反馈 \(\widehat A\) 时，它对应经验 signed log-likelihood 目标

\[
\mathcal J_\nu(\theta)
=
\mathbb E_\nu
\left[
\widehat A(s,a)\log\pi_\theta(a\mid s)
\right]
\]

的梯度。因而，不含显式 ratio 本身并不构成该经验更新定义中的遗漏。这个分析对象使我们能够直接研究历史正、负反馈反复复用所形成的 actor 动力学。

IS 与本文方法的关系也已在原稿第 2 节、第 5 节末尾及附录 C 中讨论。IS 依据当前策略与行为策略的相对概率进行分布修正；DRPO 依据样本在当前策略下的 remoteness，选择性减弱远场负反馈。两者都能改变样本影响，但控制依据不同。DRPO 的控制不需要取得或估计行为策略密度。

原始 IS 可能面临高方差问题。当行为策略对某个动作的概率（或密度）很低，而当前策略对其赋予较高概率（或密度）时，单步 importance ratio 就可能很大；长轨迹上的概率比连乘还可能进一步放大方差。这种高方差可能增加梯度估计的波动，使训练更不稳定，也是研究不依赖显式 ratio 的优化路线的实际动机之一。

如我们在 W1 第（1）点中所述，本文研究数据完全不刷新的 zero-refresh limit，即历史数据持续复用的极端 off-policy 设置。Levine 等人（2020）在论文《[Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems](https://arxiv.org/pdf/2005.01643)》第 3.5 节中明确讨论了重要性采样在完全离线训练中的困难：当学习策略与行为策略差异过大时，重要性权重会退化，使回报或梯度估计的方差过大而难以有效使用。因此，该论文指出，重要性采样估计最适合学习策略仅有限偏离行为策略的情形。这为本文在上述设置下研究不依赖显式重要性采样的控制方式提供了直接的文献依据。

本文因此从当前策略几何出发研究另一种控制方式，直接根据 remoteness 衰减远场负更新。至于加入适当重要性修正后，未经修正时的失稳路径会如何改变，我们在 Q1 中直接说明。

### W2：外部机制证据、收益幅度与不确定性

> The external experiments do not directly establish that the proposed mechanism causes training instability. Gradient probes show a near-to-far pattern, but the causal interventions are confined to controlled environments. The modest structured-generation gains also lack uncertainty estimates in the main table.

感谢您对证据范围和结果报告的建议。我们分别回答外部机制证据、收益所支持的贡献，以及不确定性估计。

#### （1）为什么使用受控干预，以及外部任务中的补充证据

感谢您对实验设计与外部机制证据的关注。我们选择受控实验，是为了区分“样本远”和“样本差”这两个因素：自然数据中的远样本也可能质量较差，仅观察删除后的性能变化，难以判断是哪一个因素起作用。引言（第 1 节）讨论了这一混杂问题，第 6.1 节进一步介绍了在相同奖励和固定优势下改变距离的比较设计。

正文第 6.1 节进一步说明了这一设计：C-U1 的负动作共享奖励等值线和固定优势，但相对当前策略的标准化距离不同；相应干预分支仅改变近场／远场负更新的处理。附录 E.2 展开了源头隔离的构造与比较协议。外部梯度诊断用于检查相应的 near-to-far 模式是否出现在真实神经策略中；受控干预进一步识别该因素是否会传导为失稳。二者回答不同层次的问题。我们同意，外部梯度诊断本身不能替代外部训练中的因果干预。

为进一步回应这一点，我们在本回复中补充一组 Hopper 外部干预结果。在同一外部设置下，比较保留全部负更新、删除近场负更新和删除远场负更新，终态结果如下。

| 干预方式 | 终态平均归一化回报 | 任务性能崩溃次数 |
|---|---:|---:|
| 保留全部负更新（Signed） | 0.99 | 10/10 |
| 删除近场负更新（Near-zero） | 1.13 | 10/10 |
| 删除远场负更新（Far-zero） | 22.53 | 3/10 |

表中“崩溃”专指任务性能崩溃，各分支均未出现 NaN/Inf 数值崩溃。该外部干预设置使用 Hopper-v4 兼容评估环境与 D4RL-v2 参考分数归一化。

删除近场负更新没有减少任务崩溃，而删除远场负更新后，崩溃次数由 10/10 降至 3/10，终态平均归一化回报也明显提高。在对应梯度诊断中，far/near 的绝对优势均值比约为 0.99977，而完整参数负梯度的均值比约为 4.21。这说明所观察到的梯度差异不能简单归因于远样本具有更大的负优势系数。

与受控环境相比，真实 Hopper 数据中的距离与样本质量等因素更难被完全分离。在这种难以进行同等严格因素隔离的外部条件下，我们仍然观察到相应的梯度模式，以及删除远场负更新后回报提高、任务崩溃减少的现象。这为受控实验识别的远场机制提供了外部支持。

#### （2）如何理解结构化生成中的收益幅度

关于收益幅度，本文的核心贡献是识别 remoteness 在 off-policy 训练崩溃中的关键作用，并提出按 remoteness 衰减负更新的控制策略。同一控制原则覆盖连续 Gaussian 策略（如连续控制任务）和离散 categorical 策略（如语言模型的结构化生成任务）。仅通过这一简单控制，我们就在两类任务中取得了有竞争力的结果，展示了这一思路跨策略族的适用性。

该控制只依赖当前策略的 remoteness，无需取得或估计行为策略密度，也不需要额外参考策略。例如，canonical TOPR 的负样本权重 \(\min\{\pi_\theta/\mu,1\}\) 需要行为策略 \(\mu\) 对样本的概率；DPO 则在偏好目标中使用固定参考策略的响应概率。DRPO 的负反馈衰减不需要上述信息，便于接入已有的策略更新流程。它还可以与 baseline 调整等方法组合：以 AsymRE 的 \(r-V\) 作为优势输入，在不衰减边界 \(\omega\equiv1\) 恢复其原始更新系数；启用衰减则进一步调节远场负样本的影响。两种调节分别作用于反馈构造和当前策略下的样本影响，可以组合使用。

因此，这些性能结果的意义在于：结合理论分析及受控干预，验证了一种通用、简洁且可组合的控制思路——根据当前策略下的 remoteness 调节历史负反馈的影响。跨连续和离散任务的竞争性结果为这一思路提供了外部有效性证据，也为后续方法改进提供了基础。

#### （3）补充不确定性估计

感谢您指出主表缺少不确定性估计。我们补充报告九项结构化生成任务中 Positive-only、AsymRE、β-TOPR、DPO 与 DRPO 的均值和标准差，结果如下。

| 任务 | Positive-only | AsymRE | β-TOPR | DPO | DRPO |
|---|---:|---:|---:|---:|---:|
| Countdown | 13.98 ± 0.54 | 14.38 ± 0.24 | 14.41 ± 0.12 | 8.48 ± 0.82 | **15.72 ± 0.48** |
| Word Sorting | 62.06 ± 3.43 | 61.72 ± 1.99 | 59.61 ± 0.78 | 34.38 ± 12.85 | **63.38 ± 1.39** |
| Spiral Matrix | **100.00 ± 0.00** | 99.92 ± 0.11 | **100.00 ± 0.00** | **100.00 ± 0.00** | 99.94 ± 0.14 |
| Mini Sudoku | 95.91 ± 0.50 | **97.58 ± 0.33** | 96.88 ± 0.88 | 91.41 ± 2.21 | 97.06 ± 0.32 |
| Maze | 76.74 ± 0.87 | 77.81 ± 0.66 | 75.86 ± 0.55 | **79.69 ± 0.87** | 77.66 ± 0.00 |
| Word Ladder | 10.68 ± 1.25 | 12.50 ± 0.23 | 11.41 ± 0.45 | 4.69 ± 2.12 | **13.38 ± 0.82** |
| Knights & Knaves | 67.53 ± 2.00 | **70.16 ± 0.88** | 68.59 ± 1.77 | 58.20 ± 1.98 | 69.53 ± 0.00 |
| Graph Coloring | 98.67 ± 0.58 | 98.91 ± 0.22 | 99.37 ± 0.45 | 96.88 ± 0.62 | **99.88 ± 0.13** |
| WikiSQL | 85.65 ± 0.96 | 85.78 ± 0.44 | 82.89 ± 1.88 | 78.91 ± 1.02 | **86.69 ± 0.95** |
| **整体汇总（九任务）** | **67.91（平均任务 SD：1.13）** | **68.75（平均任务 SD：0.57）** | **67.67（平均任务 SD：0.76）** | **61.40（平均任务 SD：2.50）** | **69.25（平均任务 SD：0.47）** |

表注：分数为 Pass@8 百分比，SD 单位为百分点。Positive-only 的 Countdown 使用两个训练 seed，其余任务使用六个训练 seed。AsymRE 和 β-TOPR 的八项迁移任务 SD 来自两个训练 seed。DRPO 的 Countdown 使用所选配置的两个训练 seed；其余八项报告所选五个相邻 λ 配置之间的样本标准差，每个配置的分数为晚期五次评估的均值。整体汇总的分数为九任务宏平均，括号内为九个任务各自 SD 的等权平均。

DRPO 的八项迁移任务平均配置标准差为 0.47 个百分点，八项中有七项低于 1 个百分点，说明在所报告的相邻 λ 配置间，任务分数波动较小、表现平稳。我们将在修订稿中补充上述均值、标准差及其计算说明。

### Q1：优势刷新、重要性修正与外部 q/p 轨迹

> Does the predicted instability persist when advantages are refreshed or off-policy corrections are applied? What are the observed (q/p) trajectories in the external experiments?

感谢您提出这一问题。优势刷新和重要性修正改变更新的方式不同，因此需要分别判断它们对本文机制的影响。

#### （1）优势刷新后，不稳定性是否仍可能存在？

感谢您关于优势刷新的问题。重新计算优势后，各样本优势的大小和正负号可能改变，因此正反馈总强度 \(p\) 和负反馈总强度 \(q\) 也会改变。这里，\(p\) 汇总正优势，\(q\) 汇总负优势的绝对值，两者都包含样本频率和优势大小。

即使 critic 完全准确，历史数据上的负反馈仍可能占优。优势反映历史动作相对于当前策略的表现，而这些动作本来就可能整体较差，因此准确估计同样可能得到 \(q>p\)。Critic 的准确性与正负反馈的平衡，是两个不同的问题。

优势刷新改变了反馈系数，但历史负样本与当前策略之间的距离效应仍然存在。对于刷新后仍为负的样本，持续复用仍可能将其进一步推远，而其负更新影响未必相应减弱。因此，刷新后的反馈平衡与持续复用过程，仍可能形成本文研究的远场失稳路径。DRPO 可以直接使用刷新后的优势，再根据当前 remoteness 调节远场负更新。

#### （2）加入 off-policy correction 后，不稳定性是否仍会持续？

感谢您关于加入 off-policy correction 后失稳是否仍会出现的提问。适当的重要性修正可以缓解，甚至阻断本文分析的远场失稳路径。这与本文识别的机制是一致的：减弱远负样本的有效更新影响，可以抑制相应的排斥与失稳。

重要性采样通过重加权，改变历史样本在更新中的有效占比。对同一个历史负样本，在行为策略概率固定且非零时，当前策略赋予它的概率越低，其重要性权重 \(\pi_\theta/\mu\) 就越小，从而可以减弱该样本的负更新影响。DRPO 则直接根据当前 remoteness 衰减远负样本的更新。因此，两者虽然采用不同的控制依据，却可以起到相近的作用：降低远负样本对更新的影响。这一联系也与论文第 5 节及附录 C 中的讨论一致。

正如我们在对 W1 第（4）点的回复中所解释的，原始 IS 在减弱部分样本影响的同时，也可能产生过大的重要性权重，带来高方差并增加训练波动。

DRPO 对负分支使用 \([0,1]\) 内的衰减系数，只减弱相应样本的负更新，不会引入由过大 importance ratio 造成的额外梯度放大，也无需获取或估计行为策略概率。因此，DRPO 提供了一种直接控制远场负影响的方式，同时避免了原始 IS 中过大概率比带来的放大问题。

#### （3）外部实验中观察到的 q/p 轨迹是什么？

**【待完成：已讨论统计口径并形成 Hopper 补充实验方案；实际数值与曲线待补，当前没有可直接提交的外部轨迹回复。】**

### Q2：匹配总负梯度大小后的外部干预

> Does removing or capping far-field negative updates rescue D4RL or structured-generation training when total negative-gradient magnitude is matched?

**【待完成：已讨论 far-zero／far-cap 与总负梯度匹配对照，并纳入 36-run Hopper 方案；实际比较结果与最终回复仍待补。W2 的旧 Hopper case 不替代这一新匹配实验。】**

### Reviewer Uwf4：W1（1）正文记号的呈现

> Too much notation was introduced, even for simple things, and never reused later.

感谢您指出记号呈现带来的阅读负担。重新检查正文后，我们认同部分辅助记号和重复定义可以精简。我们将在修订稿中合并重复定义、将仅用于承接一次表达的中间变量内联，并统一步长记号，使表达更加紧凑。

本文主要记号用于区分机制分析中不同的作用因素。例如，\(D\) 表示样本在当前策略下的 remoteness，\(R\) 表示相应的梯度响应，\(p,q\) 则表示正负反馈的总强度。这些量在后续定理和推导中反复使用，其区分使我们能够分别分析反馈大小、策略几何及其聚合动力学。因此，我们会保留这些必要记号，同时精简辅助记号，并改善首次引入时的解释，降低读者的理解负担。


### Reviewer Uwf4：研究动机、理论增量及与已有方法的关系

感谢您对研究动机及相关工作关系的关注。本文希望回答的核心问题是：**历史负反馈在反复复用时，什么时候能够帮助学习，什么时候会造成过度排斥乃至失稳，以及如何保留有益作用、控制有害作用。** 因此，我们不仅研究单个负样本如何影响策略，还分析正负反馈共同作用时的平衡位置、稳定性与任务表现，并据此提出 DRPO。下面结合论文已有内容进一步解释。

#### （1）本文研究的失稳，是否只是遗漏重要性采样修正造成的？

论文式（1）分析的是历史数据上的 signed actor 更新，即利用正反馈提高相应行为的概率，利用负反馈降低相应行为的概率。固定数据与反馈系数时，它对应一个明确的经验优化目标。本文研究的是这一更新规则在历史样本反复复用时的动力学，而不是将其视为当前策略期望回报的无偏梯度估计。

重要性修正确实可以减弱这一失稳来源。当同一个负样本在当前策略下的概率相对于历史生成概率持续下降时，重要性比率也会下降，从而降低它继续参与更新的影响。**这一点与本文识别的机制相容：负样本已经远离当前策略后，持续保留较强的排斥作用，可能导致过度外推或失稳。** 但重要性采样并非解决这一问题的唯一路线。

本文采用零数据刷新的离线设定，考察历史反馈持续复用的情形。在这一设定下，策略可以不断变化，而数据无法随之刷新。[Levine 等人（2020）的论文 *Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems*](https://arxiv.org/abs/2005.01643) 第 3.5 节明确指出，当学习策略与行为策略相差过大时，重要性权重可能退化，导致回报或梯度估计具有过高的方差；这类估计更适合策略相对行为策略仅有限偏离的情况。

这种困难也不只来自长轨迹上的概率比连乘：即使单步比率中，行为概率很低、当前概率较高，也可能产生很大的权重，使少数样本主导梯度估计，增加训练波动。因此，本文研究另一种控制路线：直接根据负样本在当前策略下的 remoteness，衰减其远场更新影响。论文第 5 节末及附录 C 已讨论这一控制依据与行为策略比率的区别。

#### （2）DRPO 与 TOPR 的方法和理论区别是什么？

直观而言，**DRPO 关注“当前策略有多不可能产生这个负样本”，TOPR 关注“当前策略产生这个负样本的概率，相比行为策略降低了多少”**。两者都通过逐样本加权调节负更新，但决定权重的量不同。

为比较两种权重的结构，采用同一概率定义，记
\[
D_\theta=-\log\pi_\theta,\qquad D_\mu=-\log\mu,
\]
并将 DRPO 的指数衰减写在这一坐标下，则
\[
\boxed{
w^-_{\mathrm{TOPR}}
=\exp\!\left\{-[D_\theta-D_\mu]_+\right\},
\qquad
w^-_{\mathrm{DRPO}}
=\exp\!\left\{-k[D_\theta-\tau]_+\right\}
}
\]
其中 \(k=\lambda/c>0\)。TOPR 的表达式由其 canonical negative weight \(\min\{\pi_\theta/\mu,1\}\) 严格改写得到。**TOPR 的衰减基准由行为概率决定；DRPO 的阈值 \(\tau\) 设置当前 remoteness 的近远场边界，独立于行为概率。**（[Le Roux 等人，2025，TOPR](https://arxiv.org/abs/2503.14286)，式（1）及 Algorithm 1。）

这一差异对应两种解决路线。**DRPO 从负更新的梯度动力学出发，直接控制远场排斥。** 历史负样本被反复使用时，当前策略可能越来越不容易产生它，但它对策略的排斥作用仍可能持续，甚至增强。因此，我们根据当前 remoteness 降低远负样本的更新权重。判断样本现在有多远，只需要当前模型，不需要知道它过去以多大概率生成。

**TOPR 则利用重要性重加权，模拟负样本越来越少被采到的效果。** 如果持续从当前策略采样，一个负样本的概率降低后，后续自然更少遇到它；固定历史数据却仍会反复提供这个样本。TOPR 用当前概率与历史概率的比率，降低相应负更新的权重。这项换算需要行为策略概率。

在理论上，TOPR 已指出 signed log-likelihood 的负分支可能无界，并为其不对称加权提供了理论依据。本文进一步研究的是：**当正反馈的吸引和负反馈的排斥共同作用时，策略最终被推向哪里，什么时候能够稳定，以及稳定是否就意味着表现良好。**

论文第 4.3 节量化了这一演化规律。令 \(p,q\) 分别表示正、负反馈的总强度，包含样本权重和反馈幅度，记 \(\rho=q/p\)。在论文的指数族分析中，以 \(\mathbf m_+\)、\(\mathbf m_-\) 分别表示吸引中心与排斥中心。当 \(0<\rho<1\)、目标可达时，平衡位置为
\[
\boxed{
\mathbf m^\star(\rho)
=
\mathbf m_+
+
\frac{\rho}{1-\rho}
\bigl(\mathbf m_+-\mathbf m_-\bigr)
}
\]
这一平衡在连续动力学及足够小步长的离散更新下局部稳定。固定两端中心时，增强负反馈会使平衡沿排斥方向进一步外移。论文还分析了临界平衡下的持续漂移，以及负反馈占优时有限平衡的失稳条件。

这也说明，**负反馈产生排斥，并不意味着整体训练必然失稳；正负反馈可以形成新的稳定平衡。反过来，保持稳定也不意味着任务表现一定良好。** 在论文第 6.2 节的受控任务中，适度排斥能够把策略从正样本中心推向更好的位置，提高未见 context 上的收益；继续增强排斥，又可能越过有益区域，使收益下降，即使此时策略仍然稳定。

这一分析为 DRPO 提供了控制依据：通过逐样本负更新的保留比例，调节整体吸引与排斥的合力。阈值决定从哪里开始衰减，衰减强度决定远场影响减弱得多快，从而保留近场负反馈可能提供的有益推动，并减弱远场负反馈造成的过度排斥。**这套关于演化、平衡、稳定性及过度外推的分析，是本文区别于仅观察负分支无界或提出重加权规则的理论贡献。**

#### （3）DRPO 是否也在做重要性采样？

两者可以产生相似的效果，例如都能减弱某些历史负样本的更新影响，但权重的依据与用途不同。

重要性采样利用目标分布与采样分布的密度比，将历史数据上的期望换算为目标分布下的期望。DRPO 根据当前 remoteness 直接设置负更新的保留比例，控制远场负反馈对 actor 的作用。

上面的两个公式也明确展示了它们的联系：取 \(k=1\)，并为每个样本设置 \(\tau=D_\mu\)，就能恢复 TOPR 的负权重。但 DRPO 使用独立于行为概率的近远场边界，因此其一般控制规则并不等同于行为策略比率修正。

#### （4）DRPO 的提出，是否仅仅因为行为策略未知？

不是。行为策略未知时无需额外拟合它，是 DRPO 的实际便利；其研究动机还包括前述动力学问题，以及直接控制当前远场负影响的设计思路。

**即使行为概率完全已知，“当前样本有多远”与“当前概率相对行为概率降低了多少”仍然是两个不同的问题。** 因此，获得行为概率不会使 DRPO 的分析与控制依据失去意义。

#### （5）行为策略未知时，为什么不先估计它，再使用 clipped IS？

估计行为策略后使用 clipped IS 是一种可选方案。但即使行为策略估计完全准确，它得到的仍然是行为相对比率，而不是 DRPO 所使用的当前 remoteness。

对于来自不同策略、且未保存生成概率的混合历史数据，拟合行为分布还会引入额外的建模工作与估计误差。DRPO 直接利用当前模型计算样本远度，因而无需这一步。选择这条路线的理由，既包括信息需求上的便利，也包括控制对象本身的不同。

对于您提到的结构化生成中的冻结初始策略，本文的 replay bank 包含 oracle 解及模型无关构造的负响应，并非全部由初始学习策略直接采样。因此，冻结初始模型可以提供一个参考策略，但不自动等于这些数据的真实生成分布。论文也已经报告 joint fitted-reference \(\beta\)-TOPR，作为拟合参考策略后进行比率加权的对照，具体定义见附录的结构化生成方法部分。

#### （6）行为策略已知时，为什么仍然值得采用 remoteness 控制？

因为行为相对比率不直接决定一个样本当前有多远。例如，一个负样本在当前策略下的概率已经很低，但它在行为策略下的概率更低，此时 \(\pi_\theta/\mu>1\)，TOPR 的负权重仍为 1；DRPO 则会在当前 remoteness 超过阈值后衰减它。

这说明，行为概率已知并不能让两种控制规则自动得到相同结果。**当我们希望直接抑制当前远场负影响时，DRPO 的权重明确对应这一控制目标：根据当前负样本的近远程度，确定每个负更新保留多少，并据此调节整体外推与稳定性。** 这一路线的价值不取决于行为策略是否未知。

#### （7）为什么不直接使用 Positive-only？

Positive-only 可以避开负反馈造成的过度排斥，但也会舍弃负反馈中有用的信息。正反馈告诉模型应该接近哪些行为，负反馈则提供了应该远离哪些行为的信息。两者结合时，策略有可能超越仅拟合正样本所得的解。

本文的平衡分析给出了这一变化的明确表达：Positive-only 对应目标 \(\mathbf m_+\)；加入负反馈后，平衡沿远离负样本中心的方向移动。受控实验进一步展示，适度移动可以改善未见 context 上的表现，而过强移动又可能降低收益。因此，我们希望保留有益的负反馈作用，并控制其中的远场过度排斥。

外部实验也支持保留负反馈的价值。D4RL 的平均分由 Positive-only 的 **70.1** 提高至 DRPO 的 **77.7**；SG-9 的宏平均由 **67.91** 提高至 **69.25**。具体到单项任务，Word Ladder 从 **10.68%** 提高至 **13.38%**，相对提升 **25.3%**；Countdown 从 **13.98%** 提高至 **15.72%**，相对提升 **12.4%**。总体上，九项结构化生成任务中有八项超过 Positive-only。

作为补充，历史开发阶段的 D4RL 参数扫描还提供了两个提升较明显的例子：

| 数据集 | Positive-only | DRPO 扫描最优点 | 相对提升 |
|---|---:|---:|---:|
| Walker2d-medium-replay | 52.99 | **62.77** | **18.5%** |
| Hopper-medium-replay | 50.86 | **57.53** | **13.1%** |

这两项报告的是两个开发 seed 的后段平均分，属于历史参数扫描的补充观察，与上述论文主表结果分别报告。它们进一步展示了保留并控制负反馈可以获得的收益。

TOPR 原文同样报告了保留负样本相较 Positive-only 的收益（第 4.3 节）。本文进一步提供的是对这类作用的动力学刻画，以及基于当前 remoteness 的选择性控制方式。


### Reviewer Uwf4：W3——不确定性报告与参数稳健性

感谢您指出主表缺少不确定性信息。我们补充报告 Positive-only 与 DRPO 的均值和标准差，结果如下。

| 任务 | Positive-only：均值 ± seed SD | DRPO：均值 ± SD（来源见表注） |
|---|---:|---:|
| Countdown | 13.98 ± 0.54 | **15.72 ± 0.48** |
| Word Sorting | 62.06 ± 3.43 | **63.38 ± 1.39** |
| Spiral Matrix | **100.00 ± 0.00** | 99.94 ± 0.14 |
| Mini Sudoku | 95.91 ± 0.50 | **97.06 ± 0.32** |
| Maze | 76.74 ± 0.87 | **77.66 ± 0.00** |
| Word Ladder | 10.68 ± 1.25 | **13.38 ± 0.82** |
| Knights & Knaves | 67.53 ± 2.00 | **69.53 ± 0.00** |
| Graph Coloring | 98.67 ± 0.58 | **99.88 ± 0.13** |
| WikiSQL | 85.65 ± 0.96 | **86.69 ± 0.95** |
| **整体汇总（九任务）** | **67.91（平均任务 SD：1.13）** | **69.25（平均任务 SD：0.47）** |

表注：均值为 Pass@8 百分比，SD 单位为百分点。Positive-only 的 Countdown 使用两个训练 seed，其余任务使用六个训练 seed。DRPO 的 Countdown 使用所选配置的两个训练 seed；其余八项报告所选五个相邻 λ 配置之间的样本标准差。汇总行的分数为九任务宏平均，括号内为九个任务各自 SD 的等权平均。

DRPO 在八个迁移任务中的平均配置标准差为 **0.47 个百分点**，其中七个任务低于 **1 个百分点**，说明在所报告的相邻配置下，整体表现较为平稳。附录 E.9 的图 6 进一步显示，八个任务均存在多个相邻参数配置超过 Positive-only，另一任务的曲线能够达到基线的 **100% 天花板水平**。这些结果为所报告收益在一定参数范围内的稳健性提供了补充支持。我们将在修订稿中补充上述统计结果及计算说明。


### Reviewer iW2z：逐条回复（部分整合）

本节整合已确认的不确定性回复、两张统计表及九任务汇总，以及 D4RL 表现与方法实际价值的完整中文回复。Q1–Q7 的完整中文回复现已确认并收录；其余 Weaknesses 继续逐条核对，不据此标记本 reviewer 全部回复完成。

#### Weaknesses：表 2 的不确定性报告

感谢您指出表 2 缺少不确定性报告。我们补充报告 D4RL 与结构化生成任务的均值和标准差，并将在修订稿中加入相应表格及统计说明。

**Table 2(a)：D4RL-9 normalized return**

| 数据集 | BC | TD3+BC | CQL | IQL | DRPO |
|---|---:|---:|---:|---:|---:|
| halfcheetah-medium | 43.2 ± 0.6 | 48.3 ± 0.3 | 47.0 ± 0.5 | 47.4 ± 0.2 | 47.3 ± 0.3 |
| hopper-medium | 54.1 ± 3.8 | 59.3 ± 4.2 | 53.0 ± 28.5 | 66.2 ± 5.7 | 57.3 ± 10.1 |
| walker2d-medium | 70.9 ± 11.0 | 83.7 ± 2.1 | 73.3 ± 17.7 | 78.3 ± 8.7 | 85.5 ± 2.2 |
| halfcheetah-medium-replay | 37.6 ± 2.1 | 44.6 ± 0.5 | 45.5 ± 0.7 | 44.2 ± 1.2 | 43.3 ± 0.5 |
| hopper-medium-replay | 16.6 ± 4.8 | 60.9 ± 18.8 | 88.7 ± 12.9 | 94.7 ± 8.6 | 87.8 ± 8.8 |
| walker2d-medium-replay | 20.3 ± 9.8 | 81.8 ± 5.5 | 81.8 ± 2.7 | 73.8 ± 7.1 | 78.1 ± 6.1 |
| halfcheetah-medium-expert | 44.0 ± 1.6 | 90.7 ± 4.3 | 75.6 ± 25.7 | 86.7 ± 5.3 | 91.1 ± 2.5 |
| hopper-medium-expert | 53.9 ± 4.7 | 98.0 ± 9.4 | 105.6 ± 12.9 | 91.5 ± 14.3 | 98.0 ± 10.2 |
| walker2d-medium-expert | 90.1 ± 13.2 | 110.1 ± 0.5 | 107.9 ± 1.6 | 109.6 ± 1.0 | 110.4 ± 1.2 |
| **D4RL-9 总分（均值之和）** | **430.7** | **677.4** | **678.4** | **692.4** | **698.8** |
| **整体汇总（九任务）** | **47.86（平均任务 SD：5.73）** | **75.27（平均任务 SD：5.07）** | **75.38（平均任务 SD：11.47）** | **76.93（平均任务 SD：5.79）** | **77.64（平均任务 SD：4.66）** |

表注：每个任务报告均值 ± SD，DRPO 的原表结果基于 10 次运行。总分为九项任务均值之和；整体汇总分别报告九任务分数的宏平均，以及九个任务各自 SD 的等权平均。汇总按上表显示值计算，保留两位小数。

**Table 2(b)：Structured-generation Pass@8**

| 任务 | Positive-only | AsymRE | β-TOPR | DPO | DRPO |
|---|---:|---:|---:|---:|---:|
| Countdown | 13.98 ± 0.54 | 14.38 ± 0.24 | 14.41 ± 0.12 | 8.48 ± 0.82 | **15.72 ± 0.48** |
| Word Sorting | 62.06 ± 3.43 | 61.72 ± 1.99 | 59.61 ± 0.78 | 34.38 ± 12.85 | **63.38 ± 1.39** |
| Spiral Matrix | **100.00 ± 0.00** | 99.92 ± 0.11 | **100.00 ± 0.00** | **100.00 ± 0.00** | 99.94 ± 0.14 |
| Mini Sudoku | 95.91 ± 0.50 | **97.58 ± 0.33** | 96.88 ± 0.88 | 91.41 ± 2.21 | 97.06 ± 0.32 |
| Maze | 76.74 ± 0.87 | 77.81 ± 0.66 | 75.86 ± 0.55 | **79.69 ± 0.87** | 77.66 ± 0.00 |
| Word Ladder | 10.68 ± 1.25 | 12.50 ± 0.23 | 11.41 ± 0.45 | 4.69 ± 2.12 | **13.38 ± 0.82** |
| Knights & Knaves | 67.53 ± 2.00 | **70.16 ± 0.88** | 68.59 ± 1.77 | 58.20 ± 1.98 | 69.53 ± 0.00 |
| Graph Coloring | 98.67 ± 0.58 | 98.91 ± 0.22 | 99.37 ± 0.45 | 96.88 ± 0.62 | **99.88 ± 0.13** |
| WikiSQL | 85.65 ± 0.96 | 85.78 ± 0.44 | 82.89 ± 1.88 | 78.91 ± 1.02 | **86.69 ± 0.95** |
| **整体汇总（九任务）** | **67.91（平均任务 SD：1.13）** | **68.75（平均任务 SD：0.57）** | **67.67（平均任务 SD：0.76）** | **61.40（平均任务 SD：2.50）** | **69.25（平均任务 SD：0.47）** |

表注：分数为 Pass@8 百分比，SD 单位为百分点。Positive-only 的 Countdown 使用两个训练 seed，其余任务使用六个训练 seed。AsymRE 和 β-TOPR 的八项迁移任务 SD 来自两个训练 seed。DRPO 的 Countdown 使用所选配置的两个训练 seed；其余八项报告所选五个相邻 λ 配置之间的样本标准差，每个配置的分数为晚期五次评估的均值。整体汇总的分数为九任务宏平均，括号内为九个任务各自 SD 的等权平均；本轮新增的 DPO 与 Countdown SD 已填入对应单元格。

DRPO 的八项迁移任务平均配置标准差为 0.47 个百分点，八项中有七项低于 1 个百分点，说明所报告的相邻 λ 配置间任务分数波动较小。上述统计补充了原主表的不确定性信息，我们将在修订稿中一并报告。

#### Weaknesses：D4RL 表现与方法实际价值

感谢您对 D4RL 实验表现与方法实际价值的关注。关于收益幅度，本文的核心贡献是识别 remoteness 在 off-policy 训练崩溃中的关键作用，并提出按 remoteness 衰减负更新的控制策略。动力学分析与受控实验解释了这一控制的依据，D4RL 实验则检验其在实际连续控制任务中的有效性。

行为策略约束和保守价值处理是离线强化学习中的代表性控制路线。我们提供了一条不同于行为策略约束和保守价值处理的控制路径，直接在策略更新中调节远场负反馈。正如正文第 6.3 节所述，我们在既有的优势加权策略更新中，仅加入负分支上的 remoteness 衰减，无需额外引入显式行为策略约束或保守价值惩罚，就在 D4RL-9 上取得了有竞争力的结果。仅通过这一简单改动获得这样的表现，说明直接控制远场负反馈是一条具有实际价值的策略优化路径。

同一控制原则覆盖连续 Gaussian 策略（如连续控制任务）和离散 categorical 策略（如语言模型的结构化生成任务）。跨连续和离散任务的竞争性结果为这一思路提供了外部有效性证据，也为后续方法改进提供了基础。

该控制只依赖当前策略的 remoteness，无需取得或估计行为策略密度，也不需要额外参考策略。例如，canonical TOPR 的负样本权重 \(\min\{\pi_\theta/\mu,1\}\) 需要行为策略 \(\mu\) 对样本的概率。对于来自不同策略、且未保存生成概率的混合历史数据，拟合行为分布还会引入额外的建模工作与估计误差。DRPO 直接利用当前模型计算样本远度，因而无需这一步，便于接入已有的策略更新流程。

DRPO 还具有与 AsymRE 的反馈调整、IQL 的价值学习模块以及 Trust Region 式策略移动约束结合的空间。本文对 remoteness 机制及其控制有效性的研究，为后续方法的组合与改进提供了基础。

#### Q1：动作侧几何、策略输出坐标与更新幅度

感谢您指出第 4.1 节中解释不够清楚的概念。我们按照您提出的几个具体问题分别说明。

**（1）“动作侧几何”是什么意思？**

我们关注的是一个样本产生多强的更新作用，以及这种作用如何随当前策略变化。固定状态 \(s\)，是为了在同一个输入下，考察历史动作在当前策略中的概率及其梯度响应。“动作侧几何”指的就是这种给定状态下，动作的对数概率与梯度响应之间的关系，也就是保持输入不变，考察一个历史动作在当前模型下有多不可能，以及它会产生多强的更新信号。

**（2）“策略输出坐标” \(u\) 和 \(\pi_u\) 是什么意思？**

\(u\) 表示当前策略在状态 \(s\) 下，用于确定动作分布的那组输出量。\(\pi_u\) 表示由这组输出确定的动作分布，\(\pi_u(a)\) 就是动作 \(a\) 在其中的概率或概率密度。由于状态已经固定，这里简化记号，省略了对 \(s\) 的显式标注。给定当前输入，\(u\) 是模型算出的一组数，这组数确定了模型如何选择动作，对应的动作分布记作 \(\pi_u\)。

**（3）\(u\) 与网络参数 \(\theta\) 有什么区别？**

\(\theta\) 是网络内部的权重，\(u=u_\theta(s)\) 是这些权重针对状态 \(s\) 计算得到的输出。第 4.1 节先直接分析对 \(u\) 的梯度更新，以看清单个样本产生的作用。实际训练通过反向传播，将这一信号传回网络权重，得到对 \(\theta\) 的梯度。因此，\(\theta\) 是模型内部参与计算的权重，\(u\) 是针对这一次输入算出的结果；训练实际改动的是 \(\theta\)，输出 \(u\) 随之改变。

**（4）“输出空间更新幅度”是什么意思？**

更新幅度就是单个样本产生的更新向量的长度，表示这个样本推动策略的力度。这里“输出空间”明确了它衡量的是对 \(u\) 的更新信号。具体而言，样本的更新信号为 \(\widehat A(s,a)\nabla_u\log\pi_u(a)\)，其幅度为
\[
|\widehat A(s,a)|\,\|\nabla_u\log\pi_u(a)\|_2
=
|\widehat A(s,a)|\sqrt{R_u(a)}.
\]
其中，\(R_u(a)\) 是对数概率梯度范数的平方。对于负样本，上述幅度衡量其排斥性更新的强度。直接做普通梯度更新时，该量乘以学习率，就是这个样本造成的一步 \(u\) 的移动幅度；文中的表达省略了共同的学习率。这里，加权后的梯度方向表示往哪里调整，其范数表示这个样本提供的更新信号有多强；多个样本共同训练时，各自的更新贡献再合成整步更新。

我们会在第 4.1 节的对应位置补充上述解释，使这些概念的含义及其与实际训练的关系更加明确。

#### Q2：带符号系数、固定优势与公式（4）

感谢您关于反馈系数及其固定方式的提问。我们分别说明这个系数的含义、保持固定的原因，以及公式（4）所对应的训练过程。

**（1）“带符号的系数”是什么？**

它就是第 3.1 节定义的 \(\widehat A(s,a)\)，即 actor 更新所使用的优势估计或同类正负反馈。“带符号”表示它可以为正，也可以为负：正值用于提高该动作的概率，负值用于降低该动作的概率，其绝对值决定反馈的力度。这里没有引入另一个独立于优势估计的新系数。

**（2）为什么在复用窗口内保持固定？**

本文关注的是 remoteness 如何影响样本的更新作用。对于给定样本和当前策略，remoteness 直接由当前策略计算，不需要使用优势值。因此，我们先保持反馈系数不变，单独考察策略变化带来的梯度效应：即使反馈力度没有增加，同一个历史负样本产生的排斥作用仍可能持续或增强。

这一设定也对应实际使用的训练流程。在以 PPO 为代表的一类 policy-based 方法中，可以先根据一批数据计算优势，再使用这批已经算好的优势进行多轮 actor 更新。PPO 原论文的 Algorithm 1 就先计算优势估计，再执行多轮优化。在这个复用窗口内，策略不断变化，actor 使用的优势估计仍然是先前计算的数值。[^ppo]

我们从常见的优势加权策略梯度更新形式出发，研究历史反馈持续复用所产生的累计作用。固定的是更新时使用的估计反馈，并不表示该动作相对于每一步当前策略的真实优势始终不变。

如果重新计算优势，DRPO 可以直接使用刷新后的反馈，并继续依据当前策略计算 remoteness。如果样本仍为负优势，就根据其当前 remoteness 调节负更新；如果变为正优势，就按正分支处理。优势刷新改变反馈，remoteness 则继续用于判断相应负更新应当保留多少。

优势刷新也可能强化原始的排斥作用。例如，采用 \(\widehat A=G-b\) 时，如果历史回报 \(G\) 固定而基线 \(b\) 上移，原本为负的优势会变得更负，负反馈系数的绝对值随之增大，可能进一步增强排斥。因此，优势变化不会自动消除本文研究的远场机制；固定反馈分析用于识别该机制，而算法本身可以与优势刷新结合使用。

公式（4）描述的就是复用同一个历史样本及其已计算反馈的递推更新过程：反馈数值保持固定，但每一步都根据更新后的当前策略重新计算梯度，因此同一个样本产生的更新作用仍然可以随策略变化。

我们将在第 4.2 节补充上述说明，明确固定估计优势所对应的训练流程及其分析目的。

#### Q3：Theorem 4.5 的 logit 约定与更新一致性

感谢您对定理 4.5 中坐标约定与更新一致性的关注。这里需要澄清，定理采用的是 logits 总和为零的约定，即 \(\sum_j z_j=0\)，而非总和为一。这只是为同一个策略选择统一的表示：给所有 logits 同时减去它们的均值，不会改变任何动作的 softmax 概率，因此不会限制策略能够表达的动作分布。对于文中分析的直接 logit 梯度更新，每个样本的梯度分量之和为 \(1-\sum_j\pi_z(j)=0\)，乘以优势系数并对样本聚合后仍然为零。因此，从零和表示出发，连续和离散梯度更新都会保持这一约定，并不需要额外的投影或约束步骤。附录证明已经明确给出了聚合梯度分量之和为零的计算。我们将在定理附近补充这一约定的含义及其保持性，使它与前述更新过程的对应关系更加清楚。

#### Q4：正负样本加权中心的定义、作用与所在空间

感谢您指出这两个量的定义与所在空间需要进一步说明。它们可以理解为正、负样本各自的“反馈加权中心”，计算方式类似于求物体的重心：样本的位置对应物体各部分的位置，反馈力度对应重量，再通过加权平均概括这一组样本的整体位置。

**（1）这两个量如何计算？**

在固定状态下，令 \(T(a)\) 表示理论推导采用的动作表示。正样本按正优势大小加权，负样本按负优势的绝对值加权，分别计算平均值。附录中的定义为
\[
\mathbf m_+
=
\frac{\mathbb E_\nu[\widehat A^+T(a)]}{p},
\qquad
\mathbf m_-
=
\frac{\mathbb E_\nu[\widehat A^-T(a)]}{q},
\]
其中 \(p=\mathbb E_\nu[\widehat A^+]>0\)，第二个定义用于 \(q=\mathbb E_\nu[\widehat A^-]>0\) 的情形，\(\widehat A^-\) 表示负优势的绝对值。因此，反馈力度越大、出现频率越高的样本，对相应中心的贡献越大。

**（2）为什么使用这两个中心？**

每个样本的位置和反馈力度不同，这两个中心分别汇总正、负反馈集中在哪里。\(\mathbf m_+\) 是只使用正反馈时，模型希望拟合的平均特征，因此称为 positive-only target；\(\mathbf m_-\) 概括负反馈所抑制行为的平均位置，用于描述排斥作用，并不是让模型去拟合的另一个目标。结合正、负反馈总强度 \(p,q\)，我们就能分别描述反馈集中在哪里、作用有多强，进而分析它们合成后的更新与平衡。

**（3）它们属于什么空间？**

它们与 \(T(a)\) 位于同一个向量空间：若 \(T(a)\in\mathbb R^d\)，则 \(\mathbf m_+,\mathbf m_-\in\mathbb R^d\)。策略自身的平均特征 \(\mathbb E_{\pi_\xi}[T(a)]\)，也就是这里的均值参数，同样使用这套坐标。因此，比较的是样本的平均特征与策略的平均特征；这两个中心并非神经网络权重。某个目标能否由有限的策略参数实现，则由定理中的可达性条件判断。

附录已经给出了上述计算定义。我们将在正文首次引入这两个量的位置补充其加权中心含义、计算方式与空间说明，使读者能够直接理解它们在后续分析中的作用。

#### Q5：高斯策略与有限负原子的含义

感谢您指出这里的表述歧义。高斯策略本身确实没有原子。文中的“有限个负原子”指历史数据中有限个带负反馈的样本点，每个点按相应权重计入更新。策略分布是连续的，而用于更新它的历史样本可以只有有限个，两者并不矛盾。原句没有清楚说明这些原子属于历史数据，容易造成误解。我们将在正文中将其明确表述为“对于固定协方差的高斯策略，考虑历史负反馈由有限个加权样本点表示的情形”，以消除这一歧义。

#### Q6：有界区域的定义、作用与包含条件

感谢您关于 \(\mathcal K\) 及其与正样本中心关系的提问。我们分别说明以下三点。

**（1）\(\mathcal K\) 是什么？**

\(\mathcal K\) 是在策略平均特征所在的空间中，预先划定的一块固定、有界区域。\(\mathbf m_+\) 是同一空间中的正样本加权中心。因此，\(\mathcal K\) 是一片范围，\(\mathbf m_+\) 是一个点，两者使用同一套坐标，可以直接讨论这个点是否位于该范围内。文中“历史数据覆盖或期望任务表现”是在说明该区域可能的用途，区域本身并不是历史数据集或任务得分。

**（2）为什么引入 \(\mathcal K\)？**

它用于说明负反馈能把平衡位置推离多远。**前面的外推结论表明：无论预先划定的有限范围有多大，随着负反馈相对强度增加，即使正反馈仍占主导，稳定平衡点也可以被推到范围之外；如果策略先无法表示被推远的目标，则有限平衡点先消失。** 范围大小会影响越界所需的负反馈强度，但结论不依赖某个特别选取的边界。这说明，模型能够稳定下来，并不保证它停下来的位置仍然靠近正样本中心。

**（3）为什么 \(\mathbf m_+\) 位于 \(\mathcal K\) 内部？**

Corollary 5.3 写的是“如果 \(\mathbf m_+\in\operatorname{int}(\mathcal K)\)”，并未声称它属于任意区域。这个条件表示：我们选定的区域包含正样本中心，而且中心周围还留有一定空间。在推论的其余条件下，衰减后的综合目标不断趋近于这个中心，因此最终会进入该区域。中心位于区域内部是前提，综合目标最终进入区域才是结论。

我们将在正文中补充引入 \(\mathcal K\) 的目的，并将其数学定义与用途说明区分清楚。

#### Q7：Joint fitted-reference β-TOPR 的定义与采用理由

感谢您关于这一变体的定义及选择理由的提问。附录已给出其目标函数和更新步骤，我们在此进一步说明它如何工作，以及为什么采用这一适配。

**（1）Joint fitted-reference \(\beta\)-TOPR 是什么？**

我们另外训练一个参考模型，同时拟合训练数据中的正确回答和错误回答。参考模型的拟合目标给正、负两个分支各一半总权重；策略每更新一步，参考模型也更新一步，这就是“joint fitted-reference”的含义。

对于负样本，其更新权重为
\[
w_\beta(x,y)
=
\min\left\{
\left(\frac{\pi_\theta(y\mid x)}
{\mu_{\mathrm{ref}}(y\mid x)}\right)^\beta,1
\right\},
\]
其中，比值使用完整回答的概率计算，\(\beta\) 调节衰减力度。如果当前策略生成某个错误回答的概率已经低于参考模型，就减弱继续压低它的力度。计算策略梯度时，参考模型输出和该权重均作为固定数值处理。

**（2）为什么采用拟合参考版本，而不直接使用原始 TOPR？**

原始 TOPR 的比值分母是行为策略概率，即当初生成该回答的策略赋予它的概率。然而，在一般离线数据和经验回放中，这一信息未必可用。**例如，持续刷新的 replay buffer 可以同时保留多个历史策略产生的数据，还可能混入其他来源的数据，因此未必对应某一个单独的 old-policy 模型。若没有保留逐样本的生成策略概率，就不能仅凭这些样本还原原版比值所需的分母。**

拟合参考模型，是为这类来源混合或行为概率不可得的数据建立统一、可计算的概率参照。参考模型在策略优化过程中持续拟合训练数据，从而提供计算衰减权重所需的参考概率。

本实验的固定数据也是这一情形的具体实例：正确回答来自 oracle，错误回答经过程序构造、验证和筛选，没有现成的语言模型行为策略概率。初始模型虽然可以给这些回答打分，但它并不是生成这些构造数据的策略，因此不能直接把它的概率视为真实行为策略概率。

**采用这一适配，是为了使 TOPR 式的概率比衰减机制能够用于缺少已知行为概率的离线数据，并在与其他方法相同的数据上进行比较。** 我们因此将其明确标为“joint fitted-reference \(\beta\)-TOPR”变体：其分母来自拟合参考模型，而非已知的原始行为策略。

我们将在正文中补充上述定义、一般适用情形及本实验的适配理由，并保留附录中的完整目标函数和训练细节。

[^ppo]: Schulman et al. (2017), *Proximal Policy Optimization Algorithms*, Algorithm 1（PDF 第 5 页）：先计算优势估计，再用该批数据进行 K 个 epochs 的优化。[论文 PDF](https://arxiv.org/pdf/1707.06347)。对应参考实现可见 [OpenAI Spinning Up PPO 源码](https://spinningup.openai.com/en/latest/_modules/spinup/algos/pytorch/ppo/ppo.html)。

### Reviewer aYhA：方法区别、理论贡献与性能解释

感谢您对本文理论分析和受控实验的肯定。关于 DRPO 与 β-TOPR 的区别，以及两者采用相似衰减形式却产生不同效果的原因，我们进一步说明如下。

直观而言，**DRPO 关注“当前策略有多不可能产生这个负样本”，TOPR 关注“当前策略产生这个负样本的概率，相比行为策略降低了多少”**。两者都通过逐样本加权调节负更新，但决定权重的量不同。

为比较两种权重的结构，采用同一概率定义，记
\[
D_\theta=-\log\pi_\theta,\qquad D_\mu=-\log\mu,
\]
并将 DRPO 的指数衰减写在这一坐标下，则
\[
\boxed{
w^-_{\mathrm{TOPR}}
=\exp\!\left\{-[D_\theta-D_\mu]_+\right\},
\qquad
w^-_{\mathrm{DRPO}}
=\exp\!\left\{-k[D_\theta-\tau]_+\right\}
}
\]
其中 \(k=\lambda/c>0\)。TOPR 的表达式由其 canonical negative weight \(\min\{\pi_\theta/\mu,1\}\) 严格改写得到。**TOPR 的衰减基准由行为概率决定；DRPO 的阈值 \(\tau\) 设置当前 remoteness 的近远场边界，独立于行为概率。**（[Le Roux 等人，2025，TOPR](https://arxiv.org/abs/2503.14286)，式（1）及 Algorithm 1。）您引用的附录 E.8 中，β-TOPR 的比率分母来自拟合参考策略。合并衰减强度参数并不能消除这一参考概率，因此两种方法的区别不只是增加阈值。

这一差异对应两种解决路线。**DRPO 从负更新的梯度动力学出发，直接控制远场排斥。** 历史负样本被反复使用时，当前策略可能越来越不容易产生它，但它对策略的排斥作用仍可能持续，甚至增强。因此，我们根据当前 remoteness 降低远负样本的更新权重。判断样本现在有多远，只需要当前模型，不需要知道它过去以多大概率生成。

**TOPR 则利用重要性重加权，模拟负样本越来越少被采到的效果。** 如果持续从当前策略采样，一个负样本的概率降低后，后续自然更少遇到它；固定历史数据却仍会反复提供这个样本。TOPR 用当前概率与历史概率的比率，降低相应负更新的权重。这项换算需要行为策略概率。

在理论上，TOPR 已指出 signed log-likelihood 的负分支可能无界，并为其不对称加权提供了理论依据。本文进一步研究的是：**当正反馈的吸引和负反馈的排斥共同作用时，策略最终被推向哪里，什么时候能够稳定，以及稳定是否就意味着表现良好。**

论文第 4.3 节量化了这一演化规律。令 \(p,q\) 分别表示正、负反馈的总强度，包含样本权重和反馈幅度，记 \(\rho=q/p\)。在论文的指数族分析中，以 \(\mathbf m_+\)、\(\mathbf m_-\) 分别表示吸引中心与排斥中心。当 \(0<\rho<1\)、目标可达时，平衡位置为
\[
\boxed{
\mathbf m^\star(\rho)
=
\mathbf m_+
+
\frac{\rho}{1-\rho}
\bigl(\mathbf m_+-\mathbf m_-\bigr)
}
\]
这一平衡在连续动力学及足够小步长的离散更新下局部稳定。固定两端中心时，增强负反馈会使平衡沿排斥方向进一步外移。论文还分析了临界平衡下的持续漂移，以及负反馈占优时有限平衡的失稳条件。

这也说明，**负反馈产生排斥，并不意味着整体训练必然失稳；正负反馈可以形成新的稳定平衡。反过来，保持稳定也不意味着任务表现一定良好。** 在论文第 6.2 节的受控任务中，适度排斥能够把策略从正样本中心推向更好的位置，提高未见 context 上的收益；继续增强排斥，又可能越过有益区域，使收益下降，即使此时策略仍然稳定。

TOPR 根据行为／参考相对概率调整历史负样本的影响；本文则把负反馈引起的排斥和外推本身作为分析与控制对象。这一分析为 DRPO 提供了控制依据：通过逐样本负更新的保留比例，调节整体吸引与排斥的合力。阈值决定从哪里开始衰减，衰减强度决定远场影响减弱得多快，从而保留近场负反馈可能提供的有益推动，并减弱远场负反馈造成的过度排斥。**这套关于演化、平衡、稳定性及过度外推的分析，是本文区别于仅观察负分支无界或提出重加权规则的理论贡献。**

---

## 待完成事项与核对记录（不进入审稿回复）

### 1. 尚未完成的实质性问题

- Q1 的外部 \(q/p\) 轨迹：统计与 Hopper 补充实验方案已讨论，critic／优势刷新诊断见第 2.1 节；实际数值与曲线待补，本条尚未完成。
- Q2 的总负梯度匹配比较：方案已讨论并纳入 36-run Hopper 计划，实际结果与回复待补。历史 Hopper 分支结果仍在下方存档，不把旧预算口径当作新方案的实测结果。
- Reviewer Uwf4 的完整中文回复已在 v29 整合，待用户检查；aYhA 的完整中文回复已确认并在 v30 整合；iW2z 的 Q1–Q7 中文回复已确认并在 v35 整合，剩余 Weaknesses 继续逐条处理。
- Reviewer iW2z 的不确定性回复与 Table 2(a)/(b) 已更新：A 表按本轮提供的数据补齐各任务 SD、加入 BC；B 表补齐 DPO 与 Countdown 的 SD。九任务宏平均和平均任务 SD 已重算；独立训练／配置波动的来源保持分别标注。
- Reviewer iW2z 关于 D4RL 表现与方法实际价值的回复已由用户确认并在 v34 原样整合；其他尚未讨论完成的问题继续分别处理。
- Reviewer iW2z 的 Q1–Q7 完整中文回复已由用户逐条确认并在 v35 整合。回复中的正文补充承诺尚待论文修改落实；canonical TOPR 直接比较等实验请求不因本轮文字整合而标为完成。

### 2. 用户已确认保留的补实验待办

- [ ] 有时间和计算资源后补充独立训练结果，优先补齐三种方法可直接比较的跨-seed波动。
- [ ] 具体任务、方法、seed 数量和配置届时确认；本轮没有启动新实验。
- [ ] 新结果齐备后，更新均值、样本 SD、实际 seed 数及相应回复。

当前使用 AsymRE、β-TOPR 的现有 \(n=2\) 结果；不将它们模糊改写为未说明数量的“多个 seed”。DRPO 八项迁移任务现有配置 SD 不等于独立训练 SD。上述待办不因本轮完成中文措辞而关闭。

### 2.1. Hopper 补充实验：critic 变化与优势刷新诊断

**状态：用户已确认纳入补充实验方案；v28 已核实实现分支和 PR #424 存在，未核实最新训练完成情况。**本项并入已讨论的 Hopper 固定优势／刷新优势两组及 IS 对照，回应 W1 对 changing critics 的关注，以及 Q1 对优势刷新与外部 \(q/p\) 轨迹的提问。首个场景为 \(\texttt{hopper-medium-replay-v2}\)。

两组从同一训练骨架下匹配的初始 actor／critic 状态出发。固定优势组计算一次优势并冻结；刷新组按对应训练协议继续更新 critic、重新计算优势。各方法在同一设置内使用相同数据、训练预算和配对 seeds。本项不另增 critic 网络规模或预训练时长的扫描。

在同一固定诊断样本集上，定期同步记录：

- **优势变化。**与训练起点及上一诊断时刻相比的优势变化量，以及正转负、负转正的比例，直接展示刷新过程带来的反馈变化。
- **反馈平衡。**原始 \(p_t=\mathbb E_\nu[\widehat A_t^+]\)、\(q_t=\mathbb E_\nu[\widehat A_t^-]\) 与 \(q_t/p_t\)，并分别记录方法加权后的有效正、负反馈强度。固定数据和固定优势组的原始 \(p,q\) 为常数；动态权重造成的变化另列为有效反馈强度，不混称原始 \(q/p\) 变化。
- **远场影响与任务表现。**结合负样本 remoteness、近／远场负梯度及其贡献、方法衰减效果和回报曲线，观察刷新使远场影响增强、减弱还是消失，以及各方法如何改变结果。任务性能退化、支持／方差边界事件和 NaN/Inf 分开记录。

critic 的预训练及验证误差曲线作为辅助信息。现有 Hopper 冻结优势协议的 100,000 次 critic 更新预算可作为预训练方案的起点；各方法共享兼容的预训练起点，刷新阶段沿用对应的交替更新频率，不在每次 actor 更新前重复完成一轮预训练。具体实现和训练设置在后续代码修改文档中落实。

本项增加的是原计划训练中的诊断记录，不以预期方向替代实际结果。外部 \(q/p\) 的数值、曲线及最终审稿回复仍待原始数据整理或补充实验完成后回填。

### 3. 正式提交前仍需落实的材料事项

- 本轮仅更新审稿回复文档；记号精简和统计表等稿件修改另行落实，不将其列为尚未回答的审稿问题。

- 本轮各任务均值及 SD 采用用户补充表；Hopper-medium-expert 的 DRPO 经用户进一步确认是 98.0 ± 10.2。β-TOPR 的 Graph Coloring 统一为用户最新表中的 99.37 ± 0.45。
- SG-9 宏平均为 Positive-only 67.91、AsymRE 68.75、β-TOPR 67.67、DPO 61.40、DRPO 69.25。新增 Countdown SD 后，汇总中的平均任务 SD 统一覆盖九个任务，分别为 1.13、0.57、0.76、2.50、0.47；此前八任务口径保留在历史核对记录中。宏平均分和平均任务 SD 分别标注。
- Hopper 补充表使用已有 EXT-H-E7-Q2 的终态汇总，不是本轮新跑实验。任务性能崩溃、支持／方差边界事件、NaN/Inf 继续分开。Far-zero 的支持／方差边界事件仍为 10/10，不能把任务性能救援写成所有边界事件消失。
- 论文已有依据已按当前可取得的稿件核对：§3.1 的 empirical signed actor field、§5 的 stop-gradient 控制、附录 C 的 IS/TOPR 定位、附录 E.2 的受控设计、附录 E.8 的基线定义与统计设置。
- v28 于 2026-11-15 通过 GitHub API 实际确认 easonhuo/drpo 的 main@b949e60e68d1f1832a6e06f94a3c38a32b43d7b2，并读取 AGENTS、handoff 第 0 节及 registry。当前可取得稿件与 2026-10-06 投稿 PDF 的节号、图表号仍需在最终提交前对应；本轮未重新取得该次投稿 PDF。
- v28 按用户要求整理四位审稿人的原始意见、已确认回复和本轮讨论，并通过 dev/iclr2027-rebuttal-record 同步到仓库。仅修改此回复工作文件；论文、算法代码、handoff、registry 和实验设置保持原样。本会话未运行训练，也未向审稿人发送内容。实际上传／PR 状态以交付时的 GitHub 返回结果为准。

### 4. 本轮从头核对后的修正

1. **明确文档用途。**已讨论的内容须写成面向审稿人的完整中文回复，包括致谢、直接回答、论证、证据和必要的承接；后续统一翻译。
2. **补齐 W1。**四个子问题均保留，加入原讨论中的实际场景、动态控制公式、完整梯度关系、AWR 目标、经验目标定义以及 IS/DRPO 的区别。
3. **补齐 W2。**受控设计理由、Hopper 补充表和解释放在同一回复中；收益与不确定性继续分成两个子问题，不重复展开原主表均分和任务胜出数量。
4. **补齐 Q1 已讨论部分。**优势刷新从讨论结论变为完整回复，加入准确优势下反馈平衡的单步例子；IS 部分正面回答可能缓解／阻断，并用原稿固定协方差 Gaussian score 关系解释衰减如何抵消放大。该例子没有被写成任意共享网络的长期稳定定理。
5. **纠正“完整”的含义。**v12 的 1.7 节只是简略整合，不能再称为已完成的完整回复；本版完成的是已讨论部分的中文表述，未讨论的 Q1 后半和 Q2 仍明确留空。
6. **保留历史依据。**v12 原始讨论、数字、统计口径、来源和待办完整保留在下方存档，便于后续核对。本版正文为当前采用的中文回复表述。

### 5. v14：固定数据回复的补充

本轮只扩展 W1 第（1）点“固定数据与数据刷新”，并更新版本标记与本条记录。W1 其余三点、W2、Q1、未完成问题和既有补实验待办均保持不变。

- 补入 SFT 的具体应用：固定数据上的正样本似然学习，可扩展为利用 reward／advantage 正负反馈并控制负更新的训练；与 Positive-only 的比较支持这一方向，不改写成对任意标准 SFT 的普遍优越性保证。
- 明确零刷新沿数据刷新维度构成一种极端的 off-policy 复用情形，用于研究历史负反馈持续作用的极端表现；不把零刷新等同于任意分布距离下的最大策略失配。
- 说明当前 remoteness 控制规则可直接用于一般 off-policy 训练；固定数据的长期理论保证不自动变为任意刷新过程的稳定性定理。该区分用于确定表述范围，不新增实验义务。

### 6. v15：critic 与反馈控制的关系

本轮核对确认，W1 第（1）点已完整包含上一轮的 SFT、零刷新极限和一般 off-policy 适用性论证，保持原文不变。本轮重写第（2）点，并更新版本标记与本条记录；其余回复、待办和历史存档不变。

- 删除“不能将本文的应用范围限定为必须依赖持续变化 critic 的设置”，改为中性解释 DRPO 不要求特定的反馈估计方式，不将审稿人没有提出的主张加到其意见中。
- 用论文已有两类设置说明有无 learned critic 的适用性：D4RL 使用 critic 产生的优势；结构化生成使用 verifier 提供的 signed feedback。结构化生成没有 learned critic，不等于 actor 没有优势类反馈系数。
- 将 critic 更新、优势刷新、DRPO 更新系数连接起来：critic 通过更新后的优势影响 actor；DRPO 依据当前 remoteness 调节该反馈的负分支。保留原动态系数公式，使 Q1 中已有回指仍然对应。
- 上述两类实验支持不同反馈来源下的方法适用性，不冒充持续刷新 critic 的直接稳定性验证。D4RL 所用优势的冻结事实已在回复中说明；优势刷新后的机制问题仍由 Q1 回答。

### 7. v16：直接说明 remoteness 与优势估计的解耦

本轮按用户确认的表述重写 W1 第（2）点：给定样本与当前策略，remoteness 不依赖 critic 输出或优势数值；保留“远场机制不以 critic 的估计误差为前提”，并继续用 D4RL 与结构化生成说明不同反馈来源下的适用性。

本节不再重复动态系数公式，改用直接说明连接 critic 更新、优势变化与 DRPO 负分支控制。Q1 中“如 W1 中给出的动态系数公式所示”的过时回指同步删除，Q1 的论证内容不变。其余回复、待办及 v12 存档保持不变。

这里的解耦指给定当前策略下的计算与控制设计，不表示 critic 与 actor 的长期轨迹统计独立，也不声称优势估计错误或变号的影响必然很小。这些范围说明保留在核对记录，不额外扩展为新的审稿问题或实验义务。

### 8. v17：保持 critic 回复的主题集中

按用户意见，删除 W1 第（2）点末尾转向 Q1 失稳问题的回指。本节聚焦 critic、优势构造与 DRPO 控制的关系；Q1 仍在自己的问题下独立回答。本轮仅作上述删改、版本标记与本条记录更新，其余回复和待办不变。

### 9. v18：补充 IS 高方差的来源与训练影响

本轮仅扩展 W1 第（4）点的原始 IS 方差段落：单步 ratio 也可能因行为概率很低而很大；轨迹连乘并非高方差的唯一来源；高方差可能增加梯度估计波动，使训练更不稳定。保留“可能”的限定，不改写为必然失稳，也不据此推断 DRPO 在任意任务中一定优于 IS。其余回复和待办不变。

本轮核对的一手依据：

- [Dudík、Langford 与 Li，ICML 2011，Doubly Robust Policy Evaluation and Learning](https://icml.cc/2011/papers/554_icmlpaper.pdf)，§2.1：单步 IPS 的方差问题会在行为动作概率更小时加剧。
- [Wang 等，ICLR 2017，Sample Efficient Actor-Critic with Experience Replay](https://arxiv.org/pdf/1611.01224)，§3：讨论潜在无界的重要性权重及连乘的高方差，并指出偶发的大权重仍可能造成不稳定。

### 10. v19：Hopper 补充证据的呈现与依赖项对照

面向审稿人的正文改为“补充一组 Hopper 外部干预结果”，不使用“已有”这一内部时间背景。补充解释强调：真实外部数据更难达到受控环境中同等严格的因素隔离，但仍观察到相应梯度模式和远场干预后的任务性能改善，为受控结论提供外部支持。不将因素隔离较难写成实验本身“不严谨”，也不改变原有结果与崩溃统计口径。

依赖项对照补入 canonical TOPR 与 DPO：前者计算负样本权重需要行为策略概率，后者的偏好目标使用固定参考策略概率。DPO 的 reference policy 不被改称为真实行为策略；本段说明的是控制所需信息，不据此声称未测量的计算成本或调参成本优势。

本轮核对的一手依据：

- [Le Roux 等，NeurIPS 2025，Tapered Off-Policy REINFORCE](https://proceedings.neurips.cc/paper_files/paper/2025/file/6274d57365d7a6be06e58cad30d1b9da-Paper-Conference.pdf)，Algorithm 1 与 Eq. (1)：canonical TOPR 使用生成数据的策略概率作为负样本比率的分母。
- [Rafailov 等，Direct Preference Optimization](https://arxiv.org/html/2305.18290v3)，§4：偏好目标使用当前策略与参考策略的响应概率比。

本轮只修改上述三处正文及版本标记与本条记录。Hopper 与 SG 结果表、其他回复、待办和历史存档均保持不变。

### 11. v20：补入受控设计动机的正文依据

本轮核对确认，这一动机不只出现在附录。第 1 节 Introduction 的贡献列表前一段，明确指出 observational data 中 remoteness 与 advantage magnitude、data quality 混杂，并说明通过匹配这些因素、独立改变 remoteness 和分别干预 near／far 更新来研究。第 6.1 节进一步说明 C-U1 的负动作具有相同奖励等值线与固定优势、不同标准化距离，以及干预分支的共享条件。

W2 第（1）点现先引用这些正文位置，再解释“远”和“差”应如何分开，避免把设计动机写成 rebuttal 才提出的新解释。原审稿 W2 对外部因果证据范围的观察与设计动机是否已交代是不同判断；正文依据不用于主观认定审稿人未认真阅读。Hopper 补充结果继续在下文回应外部证据问题。

本轮仅改上述开头论证、版本标记及本条记录；其余回复、实验数字、统计口径和待办保持不变。

### 12. v21：采用用户确认的礼貌开头

按用户明确确认的草稿，替换 W2 第（1）点此前的开头两段：先感谢对实验设计与外部机制证据的关注，再解释“远”与“差”的区分，并自然指向正文第 1 节和第 6.1 节。后续设计细节、Hopper 补充结果、其他回复与待办不变。

后续文档修改先在对话中提供拟改文字，得到用户确认后再写入；不将未经确认的措辞视为定稿。

### 13. v22：写入用户确认的完整不确定性回复

本轮按用户确认的完整草稿（含表格）替换 W2 第（3）点：保留八任务原数值，新增任务 SD 的平均值 0.61、0.84、0.47，并保留各列的实际波动来源、样本 SD 算法和修订稿补充承诺。平均 SD 使用各任务未舍入的 SD 计算；DRPO 的平均值为 0.46906553 个百分点，八项中七项低于 1 个百分点。

删除的正文仅限本节原先表后重复说明与防御性声明。此前针对其他段落的删改建议尚未获得确认，本轮未执行。Q1 的优势刷新解释暂留原文，等待用户先讨论、确认措辞；其余回复、补实验待办与历史存档不变。本轮未启动训练或修改论文。

### 14. v23：写入用户确认的优势刷新回复

按用户明确确认的完整草稿，仅替换 Q1 第（1）点：先说明刷新会改变优势与正负反馈总强度，再说明准确估计也可得到负反馈占优，最后连接刷新后的反馈、持续复用与 remoteness 控制。原先关于不同分布下零均值及单步例子的展开不再放在这段正式回复中。

新正文与用户确认文字一致。Q1 第（2）点、外部 q/p 轨迹待办、Q2、W1、W2 及已确认的不确定性结果均不变。历史讨论和依据继续保留备查。

### 15. v24：写入用户确认的简明重要性修正回复

按用户明确确认的完整草稿，仅替换 Q1 第（2）点：先回答适当修正可以缓解甚至阻断远场失稳，再通过样本的有效更新占比说明 IS 与 DRPO 的共同作用；高方差段落采用用户确认的短句，呼应 W1 第（4）点；最后说明 DRPO 的衰减系数与行为策略信息需求。

原先 Gaussian 远场衰减的展开推导不再放在本段正式回复中，相关历史讨论与依据保留。其余回复、表格、补实验待办与历史存档不变。本轮未启动训练或修改论文。

### 16. v25：将 critic 影响与优势刷新诊断纳入 Hopper 计划

按用户确认，在内部补实验待办中新增第 2.1 节：固定／刷新优势对照同时记录优势变化与双向变号比例、原始及有效反馈强度、远场负梯度和回报；critic 预训练曲线作为辅助记录，不增加独立 critic 扫描。同步将内部 Q1 轨迹待办更新为“方案已讨论、结果待补”。

正式中文回复、已有结果表、其他待办及历史存档保持不变。本轮仅记录方案，没有修改代码或启动实验。


### 17. v28：四 reviewer 协作归档与 TOPR／动力学讨论

以下保留 v28 时点的讨论状态；Uwf4 的当前完整中文回复已在 v29 写入前部，以下“尚未整理”等表述不再作为当前回复进度。

本轮仅整理回复文档，不修改论文和实验。文首四份 review 按用户上传文本保留，原文数学符号缺失没有猜补。已有 YPds 回复、Uwf4 记号回复和 v12 历史存档保留；仅刷新明确过时的待办状态和“未上传 GitHub”说明。

**方法区别的已确认段落：**见上方 Uwf4 TOPR 定位回复。后续精简时保留“直观而言”的两问和两个并列公式，不再仅保留抽象标签、删掉公式对照。论文已有定位在 §5 末和 Appendix C；当前可核对 PDF 的相关位置为正文第 6–7 页及 Appendix C 第 23 页。

**理论贡献的已讨论主线（尚未整理成最终 reviewer 回复）：**

1. 核心是重复复用下的动力学：个体负更新的持续／增强、正负反馈的竞争、平衡位置与局部稳定性、临界漂移及失稳，不只是“负梯度可能很大”。
2. 大负梯度不必然等于任务崩溃；稳定平衡也不等于高任务性能。在正向占优且 signed target 可达等论文条件下，负反馈可以把稳定平衡从 positive-only target 向外推移。受控 utility 构造展示适度负反馈改善 held-out-context reward、继续增强后发生过度偏移和收益下降。
3. 量化依据在 §4.3：令 ρ=q/p，在正向占优、目标可达等条件下，稳定 signed target 为 m*(ρ)=m+ + ρ/(1−ρ)(m+−m−)。固定两端 moments 时，ρ 增大使平衡沿排斥方向外移；ρ=1 且两端不同时可形成漂移，ρ>1 时有限驻点不稳定。类别策略还可能先遇到可达性边界。这里量化的是平衡和稳定性，任务性能崩溃、边界事件与数值失败按各自指标判断。
4. 允许并保留有益的局部排斥，同时控制远场负影响，是方法与动力学之间的连接。收益先升后降是当前受控构造和相应实验中的结果，不写成任意任务上的普遍泛化定律。§5.2–5.3 的控制条件针对加权远场响应 ω(D)I(D)；指数衰减可以压住文中有限阶增长的响应，因此固定有限 τ（包括零）不需要行为概率参照。τ 调整近场保留范围，τ=0 不自动意味着任务性能最优。
5. C-U1 的测试是同分布 held-out-context／未见状态泛化。策略相对正样本中心的外推不等同于 OOD 测试；任务性能崩溃、支持／方差边界与 NaN/Inf 数值失败分开。
6. TOPR 也有负目标无界、正负作用的 KL 分解与替代目标有界性分析。后续对比应具体指出本论文进一步给出的轨迹、平衡、稳定性与受控 utility 结果，不写成“TOPR 没有理论解释”。

**当前 IS 追问的核对结论（尚未形成单独最终回复）：**

- “只要两策略有 gap，IS／TOPR 就惩罚”不成立。原始 ratio π/μ 可大于或小于 1；canonical TOPR 仅在 π<μ 时衰减负分支，在 π≥μ 时负权重为 1，正权重也为 1。
- DRPO 的准确表述是“保留近场负反馈、衰减远场负反馈”，不是给近样本额外放大奖励。近场可能 π<μ，远场也可能 π≥μ，因此当前远度与行为相对 ratio 不决定相同的样本排序。
- IS 校正基于旧数据的估计，不以把当前策略拉回行为策略或使两策略完全一致为目标。不能把“我们允许有益外推”与“IS 不允许策略偏离”对立起来。

**后续回复中保持的已核对事实：**

- 当前概率由当前模型计算；不把 remoteness 说成脱离模型的样本固有“客观属性”。行为概率已知时可直接使用，未知时才需额外估计。
- 混合来源在数学上仍可定义生成分布，不说“混合策略不存在合法概率”。实际优势针对行为概率缺失／难以可靠估计的历史数据。
- “单一行为策略＋准确行为概率”不保证 TOPR 与 DRPO 权重相同。在同一概率坐标下，k=1 且逐样本 τ=Dμ 是一个严格相同的参数映射；一般统一 τ 与逐样本 Dμ 并不相同。这里比较的是权重，不自动推出完整训练结果相同。
- TOPR 的负权重已截断在 [0,1]，不能把原始 IS 大 ratio 放大的问题直接作为 canonical TOPR 的缺点。DRPO 不做 IS 截断，也不因此自动成为无偏当前策略梯度估计或“无精度损失”。
- canonical TOPR 的 ratio 权重在形式上可用连续策略密度定义；原论文聚焦 LLM 不等于数学上只能用于离散策略。本文同时提供连续／离散验证是实证覆盖，不是排他适用性证明。
- 本文实际外部比较是 joint fitted-reference β-TOPR，不是 canonical TOPR；针对 iW2z／aYhA 时尤其需要对照 Appendix E.8 的真实定义。

36-run 补充实验方案已经包含固定／刷新优势、原始 IS，以及 frozen 组新增的 far-zero／far-cap 和相应全参数负梯度范数匹配 Global 对照；未加入 clipped-IS 对照。实现、实际运行和科学结果分别核实，不根据 PR 标题或聊天计划宣称训练完成。

### 18. v29：Uwf4 完整中文回复整合与已确认统计呈现

**用户授权：**本轮用户确认将已讨论的 Uwf4 回复一次性写入共享 GitHub 文档，保留此前确认的记号回复，并采用“宏平均（平均 SD）”的同一行汇总写法，供随后检查。本轮不修改论文、代码、实验协议、handoff 或 registry，也不运行训练或模型评估。

**本轮写入范围：**

- 保留原已确认的记号呈现回复。
- 整合七个解释性子问题：经验 signed 更新与 IS、TOPR 的方法／理论区别、DRPO 与 IS 的关系、行为策略未知是否为唯一动机、估计行为策略及冻结初始模型、行为概率已知时的控制价值，以及 Positive-only 的信息损失与收益证据。这七个子问题是作者组织回复的方式，不冒充审稿人原文的七条独立提问。
- 保留两个并列权重公式和正负合力的平衡位置公式，区分稳定性与任务效果，不将梯度响应、任务崩溃、支持／方差边界及 NaN/Inf 混为一谈。
- Positive-only 收益解释保留 Word Ladder 与 Countdown 的相对提升；按用户确认不再将 Knights & Knaves 作为“大幅提升”举例。
- 两个 D4RL 例子明确为历史开发扫描、两个开发 seed 的后段均值、扫描最优点，与当前主表的 70.1／77.7 分别报告；不升级为新的正式结果或独立确认的方法排名。
- 统计表保留全部九任务、实际 SD 来源及样本数。整体汇总按用户确认写为“Positive-only：67.91（平均 seed SD：1.20）；DRPO：69.25（平均配置 SD：0.47）”。前两个分数为 SG-9 宏平均，括号内为八个迁移任务的平均 SD，不写为宏平均分的“±”或置信区间。
- 平均 SD 使用未舍入值计算：Positive-only 为 1.1995007235，DRPO 为 0.4690655300 个百分点。前者为六 seed 统计，后者为五个相邻配置统计；其含义保持分别标注，正文结论为参数稳健性，不改写成跨 seed 稳定性排名。
- 图 6 的核对结论为：八项任务存在多个相邻已测参数点超过 Positive-only，另外一项的曲线可达到 100% 基线。没有写成所有参数都更好，也没有声称每条曲线都呈现完整的内部最优。
- 中文回复覆盖原审稿意见的相应条目，不表示已经获得审稿人认可或已经计算出新的 95% 置信区间。已有的跨 seed SD、配置 SD 与置信区间保持区分。

**核对来源：**

1. 当前 ICLR 论文源稿：main@b949e60e68d1f1832a6e06f94a3c38a32b43d7b2 的 paper/iclr2027/manuscript_source.tex；公式、§4.3／§5、Appendix C、结构化生成数据构造、joint fitted-reference β-TOPR、主表和 Appendix E.9 图 6。
2. Positive-only 八个迁移任务的六 seed 记录：[CURVE_POINTS](https://github.com/easonhuo/drpo/tree/2a19fa1cd23e25dcd3c319e58d286e47a4e10dac/experiments/results/e8_multitask_exp_lambda_joint_20260902/CURVE_POINTS)。本轮重新读取并计算，重现各任务的均值和样本 SD。
3. Countdown 的两 seed 记录：experiments/results/e8_paper_aligned_linear_scan_round1_pilot/PER_CELL_SUMMARY.csv。Positive-only 为 13.98 ± 0.54；DRPO 所选点的两个 late-window 分数均为 15.72，seed SD 为 0.00。本轮不将其替换为五点配置均值 15.29。
4. DRPO 参数响应及配置 SD：当前主分支 paper/iclr2027/figures/data/e8_multitask_exp_response/ 的九项 CSV，以及已保存的五点窗口逐值统计。
5. 历史 D4RL 例子：
   - [Positive-only 原始任务汇总](https://github.com/easonhuo/drpo-results/blob/3060a3c90da9fd1daccd1738da6d68f7d57b8bbb/runs/e7/E7_BENCH_JOINT_GAE_P1_FULL_20260719_02/files/outputs/e7/bench_joint_gae_p1_full_002/aggregate/p1_task_summary.csv)。
   - [P2 参数响应任务汇总](https://github.com/easonhuo/drpo-results/blob/3060a3c90da9fd1daccd1738da6d68f7d57b8bbb/runs/e7/E7_BENCH_JOINT_GAE_P2_LEFT_FULL_20260721_01/files/outputs/e7/p2left_verify_autotune_001/aggregate/p2_task_summary.csv)，并核对同目录 p2_paired_deltas.csv 与 terminal_audit.json。
   - Walker2d-medium-replay：52.9874194175 → 62.7744263987，提升 18.4704%；Hopper-medium-replay：50.8642747935 → 57.5313198655，提升 13.1075%。对应两个开发 seed 200、201；原始审计保留 pilot／formal_evidence_allowed=false 状态。
6. 一手文献：[Levine 等（2020），§3.5](https://arxiv.org/html/2005.01643v3#S3.SS5)；[Le Roux 等（2025），TOPR](https://arxiv.org/html/2503.14286v2)，式（1）、Algorithm 1、§2.1／§3.1／§4.3。

**v28 方法段原文归档：**

以下仅保留旧版本原文供追溯；当前采用前部 v29 的完整回复。

```markdown
### Reviewer Uwf4：W1（2）方法动机中的 TOPR 定位（部分完成）

**状态：以下为已讨论形成的完整中文段落，供方法区别及相关质疑复用；它尚不等于 Uwf4 的完整 motivation／novelty 回复。**对应原始意见中的“Motivation of the method was highly unclear”，以及后续对 DRPO／TOPR 关系和行为概率估计的追问。

感谢您对 DRPO 与 TOPR 关系的关注。论文第 5 节末及附录 C 已明确区分两者的控制依据，下面进一步解释这一差异。

直观而言，**DRPO 关注“当前策略有多不可能产生这个负样本”，TOPR 关注“当前策略产生这个负样本的概率，相比行为策略降低了多少”**。两者都通过逐样本加权调节负更新，但决定权重的量不同。为便于比较，在采用同一概率定义时，记 \(D_\theta=-\log\pi_\theta\)、\(D_\mu=-\log\mu\)，则
\[
w^-_{\mathrm{TOPR}}
=\exp\!\left\{-[D_\theta-D_\mu]_+\right\},
\qquad
w^-_{\mathrm{DRPO}}
=\exp\!\left\{-k[D_\theta-\tau]_+\right\},
\]
其中 \(k=\lambda/c>0\)。**TOPR 的衰减基准由行为概率决定；DRPO 的阈值 \(\tau\) 用于设置当前 remoteness 的近远场边界，独立于行为概率。**

**DRPO 的思路是：找到负更新持续排斥的原因，再直接控制它。** 本文的梯度动力学分析表明：反复使用同一个负样本，会使当前策略越来越不可能产生它，但它对策略的排斥作用仍可能持续，甚至增强。也就是说，样本已经被推得越来越远，却仍在持续推动策略远离它。因此，我们直接根据当前 remoteness 降低远负样本的更新权重，减少对它的继续排斥。判断一个样本现在有多远，只需要当前模型，不需要知道它过去由什么策略、以多大概率生成。

**TOPR 的思路则是：通过重加权，模拟这个负样本越来越少被采到的效果。** 如果不断从当前策略重新采样，一个负样本被压低概率后，后面自然就更少遇到它；但固定旧数据仍会反复提供这个样本。TOPR 因此借助重要性比率，根据当前概率相对于历史生成概率的变化，决定降低多少负更新权重。它通过减小每次更新的权重，模拟减少该样本参与更新的次数。这项换算需要历史生成概率，也就是行为策略概率。

这带来的直接区别是：**DRPO 可以由当前模型逐样本计算远度和控制权重，无需获取或拟合行为概率。** TOPR 在行为概率缺失时，则需要额外估计。尤其对于来自多个历史模型或不同来源、且没有保存生成概率的混合数据，行为概率需要通过额外建模获得，估计误差也会进入控制权重。DRPO 无需完成这一步，因此避免了这部分信息依赖和估计误差来源。
```

### 19. v30：写入用户确认的 aYhA 完整中文回复

用户已确认本轮完整回复覆盖第四位审稿人的全部质疑，并明确授权写入现有 GitHub 共享文档。本轮复用 Uwf4 已确认的 TOPR 方法／理论段落，保留并列权重公式和平衡公式，合并“是否只增加阈值”的说明，补入已确认的附录 E.8 拟合参考概率及外推控制衔接；不另加实验分数段。正文按本轮用户确认版本写入，不再自行润色。同步更新 aYhA 的完成范围、进度行和待办状态；其他 reviewer 的回复、原始 review、实验待办及历史记录保持原样。本轮未修改论文、代码、handoff 或 registry，未运行训练或向 OpenReview 提交。

以下保留 v29 的 aYhA 占位说明，仅作为历史状态：

```text
### Reviewer aYhA：逐条回复（待讨论）

完整原始意见见文首。尚未形成独立回复；其问题针对附录 E.8 的 joint fitted-reference β-TOPR 和性能差异，后续须按该实际基线作答，不能直接用 canonical TOPR 的行为策略假设替代变体定义。
```

### 20. v31：写入用户确认的 iW2z 不确定性回复及待补表格

用户已确认本轮完整回复和 Table 2(a)/(b)，并明确授权写入现有 GitHub 文档。本轮按确认文本原样整合：A 表保留论文均值、SD 和 95% CI 留待用户补充；B 表保留已确认的 SD，缺项标注待补，不将配置 SD 改称置信区间或跨 seed 统计。同步更新 iW2z 的对应进度，未将其余问题标为完成。其他 reviewer 回复、原始 review、实验待办及历史记录保持原样；本轮未修改论文、代码、handoff 或 registry，也未运行实验。

以下保留 v30 的 iW2z 占位说明，仅作为历史状态：

```text
### Reviewer iW2z：逐条回复（待讨论）

完整原始意见见文首。尚未形成独立回复；不把其他 reviewer 的回答直接标为本 reviewer 已完成。缺失的行内公式保留来源说明；审稿人原话中的“logits sum to 1”也按原文保留，后续须对照稿件实际 gauge 条件判断。
```

### 21. v32：补齐用户提供的 SD 并重算九任务汇总

用户授权将本轮补充表写入现有 GitHub 回复文档，并要求重新计算汇总。用户随后明确更正：Hopper-medium-expert 的 DRPO 是 **98.0 ± 10.2**；最初的“10.2+9.8”不采用。

本轮只更新统计表、对应表注及进度记录，保留四位审稿人的原始意见、理论／方法回复与历史存档；未修改论文、代码、handoff 或 registry，也未运行训练或模型评估。

- D4RL 表加入 BC，逐项采用本轮均值和 SD。CQL 与 IQL 的部分基线均值随用户新表更新，对应总分分别重算为 678.4 与 692.4；BC、TD3+BC、DRPO 总分分别为 430.7、677.4、698.8。
- D4RL-9 宏平均／平均任务 SD：BC **47.86／5.73**；TD3+BC **75.27／5.07**；CQL **75.38／11.47**；IQL **76.93／5.79**；DRPO **77.64／4.66**。
- SG-9 宏平均／平均任务 SD：Positive-only **67.91／1.13**；AsymRE **68.75／0.57**；β-TOPR **67.67／0.76**；DPO **61.40／2.50**；DRPO **69.25／0.47**。
- 汇总统一按九任务计算：宏平均为九个任务均值的等权平均，平均任务 SD 为九个任务所报 SD 的等权平均。使用用户本轮表中的显示精度进行十进制运算，最后保留两位小数。这一平均任务 SD 与先按 seed 求宏平均再计算的跨 seed SD 是不同统计量，表格采用“宏平均（平均任务 SD）”分别标注。
- 同步更新 YPds 的不确定性表与 Uwf4 的对应统计；DRPO Countdown SD 由旧占位 0.00 改为 0.48，DPO 的九项 SD 已填齐。DRPO 八项迁移任务的配置 SD 来源保持原样，未改称独立 seed SD，也未据此生成 95% CI。
- 旧版八任务平均 SD 及其已确认记录原样留档。本轮已知完成动作是文档更新与算术复核，不将新增表格统计写成新跑实验。

### 22. v33：按 YPds 原始问题补齐结构化生成的五方法九任务表

用户要求先核对第一位审稿人对不确定性报告的范围，再补齐相应表格。YPds 的 W2 原文为：“The modest structured-generation gains also lack uncertainty estimates in the main table.” 这项不确定性质疑明确针对结构化生成；Q2 中的 D4RL／structured-generation 则是总负梯度匹配的外部干预问题，分别处理。

本轮复用 iW2z 已核对的完整结构化生成表，放入 YPds W2 第（3）点。表格覆盖 Countdown 加八项迁移任务，共九任务，并列出 Positive-only、AsymRE、β-TOPR、DPO、DRPO 五种方法；全部均值、SD 和九任务汇总与 iW2z 表保持一致。v32 已含 Countdown，本轮补齐原三方法表中缺少的 Positive-only 与 DPO。

仅修改本段统计表、相应表注、进度与版本记录；D4RL 表及其他 reviewer 回复、四份原始意见和历史存档原样保留。本轮未修改论文、代码、实验设置、handoff 或 registry，也未运行实验。

### 23. v34：原样写入用户确认的 iW2z D4RL 表现回复

用户已确认完整中文回复，并明确授权加入现有 GitHub 文档。本轮将确认稿原样放入 iW2z 的独立子节“D4RL 表现与方法实际价值”，同步更新完成范围、进度和版本记录。

确认稿同时保留“行为策略约束和保守价值处理是离线强化学习中的代表性控制路线”及“我们提供了一条不同于行为策略约束和保守价值处理的控制路径，直接在策略更新中调节远场负反馈”两句；不以新增定位句替换原句。可组合性采用已确认的简短表述，分别列出 AsymRE 的反馈调整、IQL 的价值学习模块和 Trust Region 式策略移动约束，不扩写单一方法，也不将未来组合空间写成已验证收益。

除上述新增回复及对应记录外，现有回复、两张统计表、四位审稿人的原始意见及历史存档均保持原样。本轮未修改论文、代码、handoff、registry 或实验设置，未运行实验，也未向 OpenReview 提交内容。

### 24. v35：整合用户确认的 iW2z Q1–Q7 中文回复

用户已逐条确认第三位审稿人 iW2z 的七条 Questions 回复，并明确授权加入现有 GitHub 文档。本轮按 Q1–Q7 原顺序整合至该 reviewer 的回复节，沿用现有标题层级，保留最终确认稿的正文用语；PPO 引文脚注随回复迁入。

七条回复分别覆盖动作侧几何、策略输出坐标与更新幅度；反馈系数、固定优势及公式（4）；logit 零和约定；正负样本加权中心；有限负原子；有界区域的定义、引入目的及包含条件；拟合参考版本 TOPR 的定义及一般离线／回放适配理由。Q6 保留“无论预先划定的有限范围有多大”的核心解释及目标可达性分支；Q7 保留混合历史策略的 replay buffer 与逐样本行为概率不可得的条件。

核对依据：[定理原文](https://github.com/easonhuo/drpo/blob/b949e60e68d1f1832a6e06f94a3c38a32b43d7b2/paper/iclr2027/manuscript_source.tex#L504)；[附录中的约束保持性证明](https://github.com/easonhuo/drpo/blob/b949e60e68d1f1832a6e06f94a3c38a32b43d7b2/paper/iclr2027/manuscript_source.tex#L1929)。

本轮同步更新完成范围、进度表和待办。其他 reviewer 的回复、iW2z 已有的不确定性表格与 D4RL 表现回复、四份原始 review 及历史存档均保持原样。正文补充承诺与 canonical TOPR 直接比较等实验请求仍分别跟踪；本轮未修改论文、代码、handoff、registry 或实验设置，未运行实验，也未向 OpenReview 提交内容。

---

## 历史存档：v12 原始讨论记录（不进入审稿回复）

以下为本轮修订前的原文，完整保留；其中“当前”“尚未定稿”等表述对应 v12 当时状态。后续使用本版前部“中文回复正文”，不再把历史整合工作稿当作最终回复。

<!-- BEGIN_V12_ARCHIVE -->

# DRPO ICLR 2027：Reviewer YPds 逐条讨论记录

版本：v12  
更新日期：2026-11-13（北京时间）  
论文：Breaking the Curse of Repulsion in Off-Policy Reinforcement Learning

当前进度：W1 的四个方面已讨论，1.7 节整合回复已按四个子问题拆分；W2 的已确认主线、统计口径及补实验待办保留。Q1 前半已分别讨论优势刷新与重要性修正：优势刷新部分的回应主线已明确；与 W1 去重后的重要性修正回复草稿见 3.3.2，只正面回答加入 IS 后的影响，研究设定与方法区别回指 W1。Q1 后半的外部 q/p 轨迹仍待讨论。

W1、W2、Q1 在本文件内分别成节，各自保存审稿原文、讨论结论、回应依据、工作稿及待办。后续继续在本文件对应章节内更新。

## 1. W1：理论设定与实际 off-policy 学习的相关性

### 1.1. 审稿原文

> The theory assumes fixed data and advantages and analyzes direct policy-coordinate updates without importance correction. Its relevance to practical off-policy learning with changing critics, shared neural parameters, and data refresh remains unclear.

这是审稿人列在 Critical Weaknesses 下的负面评价，不是单纯的中立追问。回应应维护论文研究设定的实际意义，并解释理论、方法及实验之间的连接。

### 1.2. 已确定的回应主线

| 方面 | 结论 | 状态 |
|---|---|---|
| 固定数据 | 固定历史数据是实际研究场景；数据固定时策略与样本 remoteness 仍会变化 | 已讨论 |
| 固定优势 | critic 并非必需；固定反馈分析解释机制，不限制算法永远冻结反馈 | 已讨论 |
| 策略坐标 | 说明输出坐标机制与实际参数梯度的联系，结合已有神经梯度证据回应 | 已讨论 |
| 重要性修正 | Off-policy 优化不强制使用显式 IS；以 AWR 和原稿已有论述回应 | 已讨论 |

严格按审稿原文回应，优先引用论文已有内容；不把讨论中的延伸改成新的研究义务，也不将此前总 review 中未经本轮讨论的判断自动视为已达成结论。

### 1.3. W1—固定数据

#### 1.3.1 已确定结论

固定数据是具有实际需求的研究场景，不能仅凭“没有数据刷新”否定研究的相关性。

- 连续 Offline RL 本来就从既有静态数据中学习；研究价值不依赖 LLM 的存在。
- 离散场景不限于语言模型。论文用 LLM 作为真实神经离散策略的典型案例。
- SFT 使用收集好的示范数据训练；固定偏好或 verifier 标注数据上的训练，也具有实际对应。
- 在线采集反馈与随后离线批量训练并不矛盾。训练期间 policy 持续变化，即使数据不变，同一历史样本的 remoteness 和梯度也会变化。

固定数据不等于静态训练。我们分析的正是 policy 改变时，历史负反馈被重复使用所产生的动态影响。

#### 1.3.2 与 SFT 的联系

标准 SFT 通常模仿期望响应，没有显式利用失败响应的排斥信号。本文 SG 的 Positive-only 对照只学习正确 completion，其目标对应监督似然学习。

应表述为：理论解释负反馈作用的机制，实验支持适当控制的负反馈能够改善表现。不要改写成“任意负样本只要保留就一定有益”，也不要说所有 SFT 都在字面上执行“删除负样本”这一步。

#### 1.3.3 回应立场

先说明这是主动研究的实际场景，不将其写成推导方便所作的无奈让步。数据刷新会影响历史样本的复用频率和累计作用，但不使按当前 remoteness 控制负更新的规则失去定义或意义。

### 1.4. W1—固定优势与 critic

#### 1.4.1 已确定结论

critic 是反馈的一种来源，不是 DRPO 的必备组件。优势或 signed signal 也可以来自 verifier、固定 baseline 或其他反馈估计。

用户提及 GRPO 的重点是：LLM 训练不必使用 learned value critic，固定优势的批内复用有真实算法对应。该例子不是在提议“把本文改成基于 GRPO 的方法”。

对于使用 critic 的方法，可靠的优势估计是共同要求。假设 critic 准确，是合法的机制分析起点；critic 估错不是远场几何问题成立的必要条件。

#### 1.4.2 需要保持的区分

- critic 的估计误差可以减小，但当前策略对应的真实优势仍可能随 actor 改变。
- 在单次求梯度时 detach advantage，与整个复用窗口内优势数值固定，是不同的设定。
- 固定优势的长期聚合定理有其分析范围；这不能直接改写成“算法只有永远冻结 critic 才有效”。

这些区分用于准确说明范围，不作为新增的方法缺陷或额外实验义务。

#### 1.4.3 优势变化不与控制规则冲突

按当前信号，规则可以写成：

$$
\widetilde A_t
=\widehat A_t^+
-\omega(D_{\theta_t})\widehat A_t^-.
$$

当前为负，按当前距离衰减；当前变为正，就进入正分支。优势变化或变号本身并不是距离控制规则的反例。

本轮关于“critic 更新可能增强负系数”“多步 RL 优势可以双向变号”的例子，是解释性延伸，不是审稿人提出的新缺点，也不列为需要逐条反驳的额外问题。

#### 1.4.4 核心机制的表述

论文关注的是负反馈的几何暴露：更新幅度大，不等于对策略改进的价值大；已经很远的负样本不应仅凭几何放大而主导更新。

Gaussian 中分析幅度放大，categorical 中分析有界但持续的抑制。这里的“远样本价值可能较低”指其边际改进价值，不能直接等同于其 reward 必然较低。

### 1.5. W1—直接策略坐标与共享神经参数

#### 1.5.1 已确定结论

要求解释输出坐标机制如何传到神经参数，是可以回应的连接性问题。不能仅凭采用输出坐标分析，就将论文判定为缺乏实际意义；也不必为此承担任意共享神经网络的完整长期动力学证明。

审稿原文写的是 shared neural parameters，没有明确要求一个完整 Jacobian 定理。Jacobian 是本轮用于解释这项连接的数学对象，不能反过来扩大审稿人的要求。

#### 1.5.2 实际梯度已经包含 Jacobian

令 uθ(s) 为网络输出的均值或 logits：

$$
\nabla_\theta\log\pi_\theta(a\mid s)
=J_\theta(s)^\top\nabla_u\log\pi_u(a).
$$

反向传播计算的是完整参数梯度。论文中的 full-parameter gradient 诊断也包含这一影响。

对 detached DRPO 权重，有逐样本的精确关系：

$$
g^-_{\theta,\mathrm{DRPO}}
=\omega(D_\theta)g^-_{\theta,\mathrm{raw}}.
$$

因此，权重作用在包含 Jacobian 的完整样本梯度上，算法没有将共享网络假装成直接 logits 来更新。

#### 1.5.3 可选的简短 Gaussian 桥接结果

如果均值使用含可训练 bias 的线性输出头：

$$
\mu_\theta(s)=Wh_\phi(s)+b,
$$

且协方差固定，则

$$
\nabla_b\log\pi_\theta(a\mid s)
=\Sigma^{-1}(a-\mu_\theta(s)),
$$

从而

$$
\|\nabla_\theta\log\pi_\theta(a\mid s)\|_2
\ge
\|\Sigma^{-1}(a-\mu_\theta(s))\|_2.
$$

这表明在该架构条件下，神经参数化不会自动消除 Gaussian 输出侧的远场放大。

状态：这是本轮的辅助推导，尚未加入论文；使用前应确认对应输出头条件。它是单样本强度的桥接，不是整个共享网络聚合稳定性的证明，也不直接套用于所有 LoRA 参数化。不将它列为必须完成的新研究任务。

#### 1.5.4 文献校准的结论

- DDPG 显式使用链式法则；策略梯度理论本身直接处理可微参数梯度。
- TRPO 在策略分布层面建立改进界，并在实践中使用 KL/Fisher 几何。
- PPO、GRPO、TOPR 并没有统一承担任意神经网络完整训练动力学的证明；TOPR 附录 B 还直接分析 trajectory logits。
- CQL 附录有 critic 的 NTK 分析，不能声称它完全没有考虑网络几何。
- Neural PPO/TRPO 等专门理论工作，在特定网络与条件下研究神经参数化收敛。

因此，回应应以“理论层次、简洁桥接、已有神经实验证据”组织，不以“别人也没做”为唯一理由。

### 1.6. W1—没有显式重要性修正

#### 1.6.1 已确定的主要反驳

Off-policy 策略优化不强制使用显式重要性采样。AWR 是直接、同类且可核验的先例：它使用优势加权的策略拟合，而不是在 actor 更新中显式乘 πθ/μ。

$$
\max_\theta
\mathbb E_{(s,a)\sim\mathcal D}
\left[e^{\widehat A(s,a)/\beta}\log\pi_\theta(a\mid s)\right].
$$

利用 off-policy 数据，与从这些数据无偏估计当前策略梯度，是不同问题。AWR 证明存在不依赖显式 ratio 的优化路线；它不等于证明任意无 ratio 更新都是无偏估计。

我们的 Eq. (1) 定义为 empirical signed actor field。固定 ν 和 Â 时，它对应经验 signed log-likelihood surrogate 的梯度。没有显式 IS，本身不足以判定该定义“漏项”。

#### 1.6.2 选例约定

**此处以 AWR 为主要先例，不使用 CQL 作同类 actor 更新的反例。**CQL 的核心在保守价值学习，不是本条正在讨论的 signed actor 路线。IQL 的优势加权策略提取可以作为补充。

CQL 仍可用于说明静态数据的 Offline RL 问题具有实际意义；不要将两个论证位置混在一起。

#### 1.6.3 原稿已经讨论的关联

| 原稿位置 | 已有内容 |
|---|---|
| §2 Related Work | importance weighting、Retrace、V-trace；分布修正与远场负反馈控制的定位 |
| §3.1 Problem Setup | Eq. (1) 明确称 empirical signed actor field |
| §5 末尾 | 按 learner-relative remoteness 控制样本影响，区别于 behavior-policy ratio，并指向附录 |
| Appendix C | 同一 sample-influence 分类下列出 IS、TOPR、DRPO；给出 canonical TOPR ratio 及其控制坐标 |
| Appendix E.8 | 定义所用 β-TOPR 变体的 ratio 与训练目标 |

回复应先指出原稿已有节号、公式和说明，不写成“我们现在才意识到需要解释”。不默认承认原稿完全没有相关定位，也无需在 rebuttal 中从头讲授 IS。

审稿人的 W1 质疑实际相关性，并没有字面声称论文完全没提 IS。回应应针对他的真实评价，不将“他没仔细读”当作已证事实。

#### 1.6.4 关系怎样准确表述

- IS 依据 target/behavior 的相对概率进行分布修正。
- DRPO 依据当前策略的 remoteness 控制远场负反馈。
- 固定某个样本的 behavior denominator 时，IS ratio 随当前策略概率增加而增加；跨样本比较时，各自 denominator 不同，因此不能把 IS 直接定义为“越近权重越大”的距离函数。
- 两者都改变样本影响，并有数学联系，但不等同于同一个控制标准。

原始 IS，特别是轨迹 ratio 连乘时，存在高方差困难。这可以作为采用其他路线的动机。

**不把“clipped IS/TOPR 也一定高方差”作为本条反驳命题。**审稿人没有提出必须证明这个结论。此前对此的延伸不列入当前待办。

YPds 的 Q1 确实问 correction 后不稳定性是否仍存在；另一位 Uwf4 还问估计 behavior 后的 clipped correction 是否足够。这些直接问题留待对应条目，不混入 W1 的一般必要性讨论。

### 1.7. W1 中文整合工作稿

以下根据已讨论结论，按四个子问题组织，尚未逐句定稿：

#### 子问题一：固定数据与数据刷新

固定历史数据与反馈是本文主动研究的实际训练场景。连续 Offline RL 本来就从既有数据中学习；离散策略优化也包含固定偏好或 verifier 反馈上的训练，LLM 只是本文的一类典型应用。数据固定时，policy 仍然持续变化，因此同一历史负样本的 remoteness 和更新影响也会变化。

#### 子问题二：固定优势与 critic 变化

本文的固定反馈分析用于识别这一机制，不要求实际算法永远冻结数据或 critic。反馈可以来自 critic，也可以直接来自 verifier 等来源。

#### 子问题三：直接策略坐标与共享神经参数

对于神经参数化，实际训练及 full-parameter gradient 诊断都包含网络 Jacobian。输出坐标理论提供可解析的机制，神经梯度诊断和受控干预检查其在实际参数更新中的出现及传导。采用这一分析层次，不等于忽略了实际网络的反向传播。

#### 子问题四：没有显式重要性修正

Off-policy 优化并不强制采用显式 importance ratio。AWR 等策略拟合方法已经提供同类先例。本文 Eq. (1) 定义的是经验 signed actor 更新；与 IS/TOPR 的关系及区别，也已在 §2、§5 末尾和 Appendix C 中讨论。因此，不能仅凭没有显式 ratio 就将该更新判定为漏项。

### 1.8. 来源与版本

- 审稿材料：用户提供的 Pasted text.txt，Reviewer YPds。
- 论文：当前可取得的 39 页 ICLR 格式 PDF，以及 paper/iclr2027/manuscript_source.tex。
- 读取基准：easonhuo/drpo，main@b949e60e68d1f1832a6e06f94a3c38a32b43d7b2；这里记录此前的读取基准，不声称本次重新核验了最新 HEAD。

投稿页面标注曾在 2026-10-06 修改，目前尚未取得该次提交原始 PDF。正式回复前，节号、图号和表号需对应实际提交版本。

### 关键一手文献

- [AWR 原论文](https://xbpeng.github.io/projects/AWR/AWR_2019.pdf)：Algorithm 1、§3.1–3.2，优势加权策略拟合与 replay 路线。
- [Offline RL 综述](https://arxiv.org/pdf/2005.01643)：§3.5，原始 IS 在强失配、长轨迹下的方差困难。
- [DeepSeekMath / GRPO](https://arxiv.org/html/2402.03300v3)：§4.1、Algorithm 1、§5.2，critic-free 相对优势与更新解释。
- [PPO](https://arxiv.org/pdf/1707.06347)：参数化 surrogate 与神经实验。
- [TRPO](https://proceedings.mlr.press/v37/schulman15.pdf)：分布层面的改进分析与实际参数更新。
- [DDPG](https://arxiv.org/pdf/1509.02971)：actor 更新中的链式法则。
- [TOPR](https://arxiv.org/html/2503.14286v2)：负分支 ratio；Appendix B 的 logits 分析。
- [CQL](https://arxiv.org/pdf/2006.04779)：静态数据问题；Appendix D.2 的 critic NTK 分析。此处不作为无 IS actor 路线的反例。
- [IQL](https://arxiv.org/pdf/2110.06169)：expectile/value-learning 与优势加权策略提取。
- [Neural PPO/TRPO 理论](https://proceedings.neurips.cc/paper/2019/file/227e072d131ba77451d8f27ab9afdfb7-Paper.pdf)：特定神经参数化下的收敛分析。

## 2. W2：外部机制证据与结构化生成结果

### 2.1. 审稿原文

> The external experiments do not directly establish that the proposed mechanism causes training instability. Gradient probes show a near-to-far pattern, but the causal interventions are confined to controlled environments. The modest structured-generation gains also lack uncertainty estimates in the main table.

### 2.2. 讨论进度

| 部分 | 内容 | 状态 |
|---|---|---|
| 前半 | 外部梯度诊断是否建立了训练失稳的因果机制；为什么采用受控干预 | 已讨论，回应主线已确定 |
| 前半的补充证据 | 在回复中加入已有 Hopper 外部干预 case | 已确定补充方向，提交材料尚未定稿 |
| 后半 | Structured-generation 改进幅度与主表不确定性 | 完整中文回复见 2.4.8；现有波动结果见 2.4.6，后续补实验待办见 2.4.7 |

本记录只保存已讨论的立场，不提前判断尚未讨论的部分。优先解释论文已有的实验分工，并清楚区分原稿证据与已有但尚未展示的补充结果。

### 2.3. W2 前半—外部诊断与受控因果识别

#### 2.3.1 已确定结论

当前稿件的分工是明确且有理由的：

| 实验 | 主要职责 |
|---|---|
| 外部梯度诊断，Figure 1 | 真实神经策略中是否出现 near-to-far 梯度模式 |
| 受控实验，Figure 2 等 | 排除其他因素后，远场负更新是否传导为漂移、崩溃；干预能否救援 |
| 外部任务评估，Table 1–2 及系数响应曲线 | 控制方法在实际任务上是否有效 |

当前稿件没有展示与受控实验相同的完整“外部失稳—定点干预—救援”链条。审稿人对已展示证据范围的观察有依据；这本身不是算法被证伪。

#### 2.3.2 为什么采用受控实验：保留直白表述

**如果远样本碰巧也更差，删掉后效果变好，我们怎么知道是因为它们“远”，还是因为它们“差”？所以受控实验要让近、远负样本的奖励和优势相同，再检查距离的作用。**

这就是应在回复中强调的理由。Appendix E.2 已说明，外部离线数据通常难以提供任务质量精确匹配、同时能够独立设定相对当前策略距离的反事实比较。

外部诊断已经匹配或固定负信号幅度；受控环境进一步提供已知任务几何下的识别。无需猜测“业界更接受哪种解释”，只需说明替代解释需要被排除。

不要将这段写成抽象的“改变样本质量、错误类型及反馈构成”堆叠。使用上面的具体问题，让读者立即理解。

外部 near/far masking 本身可以实施。受控构造的理由是更干净地识别作用来源，不写成“外部实验原则上无法做任何干预”。

#### 2.3.3 已确定：在回复中加入已有 Hopper case

用户明确同意，将已有 Hopper 外部干预结果作为补充材料纳入回复，增强对 W2 前半的回应。

材料对应实验登记：EXT-H-E7-Q2。它是单独的 Hopper learned-critic 外部机制设置，不是 D4RL-9 方法主表，也不是 SG-9 的因果实验。

当前已核对的是实验登记汇总。已有结果应整理成对应的外部干预表/曲线，不把它登记成待运行的新实验，也不声称当前投稿已经展示了它。

**W2 前半回复优先使用的三行结果：**

| 干预 | 终态平均归一化回报 | 任务崩溃 |
|---|---:|---:|
| 保留全部负更新（Signed） | 0.99 | 10/10 |
| 删除近场负更新（Near-zero） | 1.13 | 10/10 |
| 删除远场负更新（Far-zero） | 22.53 | 3/10 |

能够支持的陈述：删除近场负更新没有减少任务崩溃；删除远场负更新提高回报并降低崩溃发生率，为真实 Hopper 任务中的远场负更新作用提供补充证据。

不要写成“删除远场后完全消除崩溃”：登记结果仍有 3/10。

**完整备用汇总，供后续 Q2 核对：**

| 分支 | 终态平均归一化回报 | 任务崩溃 |
|---|---:|---:|
| Positive-only | 32.23 | 0/10 |
| Signed | 0.99 | 10/10 |
| Near-zero | 1.13 | 10/10 |
| Far-zero | 22.53 | 3/10 |
| Far-cap | 11.47 | 7/10 |
| Dynamic budget-matched Global | 15.77 | 6/10 |

far/near 的绝对优势均值比约为 0.99977，full-parameter negative-gradient 的 far/near 均值比约为 4.206。以上来自登记汇总，不表示本轮重新运行或独立复核了原始实验包。

表中的“任务崩溃”不等于 NaN/Inf 数值崩溃，也不等于支持/方差边界事件：各分支 NaN/Inf 均为 0；Far-zero 仍有 10/10 出现过支持/方差边界事件。该设置使用 Hopper-v4 兼容环境与冻结的 D4RL-v2 参考分数归一化，不能当作原版环境排行榜的直接复现。

这些材料的展示与解释尚待落实；尤其匹配总梯度后的比较属于 Q2，不能因备份了这张表就将 Q2 标为完成。

#### 2.3.4 W2 前半中文工作稿

我们选择受控实验，是为了区分“样本远”和“样本差”这两个因素。如果自然数据中的远样本也更差，那么删除这些样本后，即使性能提高，也无法单凭这个结果判断改善来自哪个因素。因此，我们在 C-U1 中构造奖励和优势相同、但相对当前策略距离不同的负样本，再分别干预近场和远场负更新。这一设计理由已在附录 E.2 中说明。

外部梯度诊断用于检查同样的 near-to-far 模式是否出现在真实神经策略中。我们还补充已有 Hopper 外部干预结果：保留全部负更新和删除近场负更新时，任务崩溃均为 10/10；删除远场负更新后，崩溃降至 3/10，终态平均归一化回报从 0.99 提高至 22.53。这为远场负更新在真实 Hopper 任务中传导为训练失稳提供了补充证据，与受控实验的机制识别相互支持。

工作稿状态：回应主线和保留数字已讨论，提交措辞及补充表/曲线尚未定稿。

### 2.4. W2 后半—收益幅度、不确定性与后续补充

#### 2.4.1 已讨论的回应立场

主表缺少不确定性估计，是需要补充的信息。审稿人原文没有指定必须补跑多 seed，不把重跑训练当作回应前提。在现有时间和计算条件下，优先利用已经保存的结果。

原表 SG-9 宏平均：Positive-only 67.91、AsymRE 68.75、β-TOPR 67.67、DRPO 69.25。DRPO 相对 Positive-only 提高 1.34 个百分点，相对 AsymRE 提高 0.50 个百分点；不将外部收益描述为大幅领先。

补充方向：对已有高分参数邻域报告均值、标准差或范围，明确波动来自不同参数配置；不把它改称 seed 标准差或 95% 置信区间。AsymRE、β-TOPR 可以做同类分析，但仍需各自的逐参数记录。

2026-11-12 更新：用户提供了 AsymRE、β-TOPR 八个 transfer 任务的 seed 4000、5000 原始汇总截图。现有两次独立训练结果可以计算样本 SD；当前采用“跨随机种子的均值 ± 样本标准差”，在表注明确 n=2。用户同意先按这一口径写，并明确后续有时间仍要补实验。以下保留原先配置邻域分析，新增 seed 结果和当前写法另见 2.4.6；这不表示已完成更多 seed 的实验。

#### 2.4.2 计算口径与结果

本节原先仅重新汇总已有 CSV；2026-11-12 另根据用户提供的两-seed结果截图计算基线 SD。以上均未运行训练或模型评估。

对每项任务，按 λ 排序，使用均值最高的连续五点窗口，不将分散的最高分点拼在一起。每个配置值已经是晚期五次评估的均值；对五个配置值计算样本 SD（分母为 5−1），单位为百分点。

八个 transfer 任务每个 λ 主要只有 seed 4000，因此这里报告配置间波动。八项任务的五点均值均重现原表。Countdown 原表 15.72 来自单个参数点的两-seed均值；其统一五点统计另列，保留原分数以免混淆。

下表保留此前汇总：AsymRE、β-TOPR 两列是论文原分数，DRPO 另列五点配置邻域统计。新提供的两-seed基线 SD 见 2.4.6，不能把它与配置邻域 SD 当作同一种波动来源。

| 任务 | AsymRE 原分数 | β-TOPR 原分数 | DRPO 原分数 | DRPO 五点邻域均值 ± 配置 SD |
|---|---:|---:|---:|---:|
| Countdown | 14.38 | 14.41 | 15.72 | 15.29 ± 0.42 |
| Word Sorting | 61.72 | 59.61 | 63.38 | 63.38 ± 1.39 |
| Spiral Matrix | 99.92 | 100.00 | 99.94 | 99.94 ± 0.14 |
| Mini Sudoku | 97.58 | 96.88 | 97.06 | 97.06 ± 0.32 |
| Maze | 77.81 | 75.86 | 77.66 | 77.66 ± 0.00 |
| Word Ladder | 12.50 | 11.41 | 13.38 | 13.38 ± 0.82 |
| Knights & Knaves | 70.16 | 68.59 | 69.53 | 69.53 ± 0.00 |
| Graph Coloring | 98.91 | 99.37 | 99.88 | 99.88 ± 0.13 |
| WikiSQL | 85.78 | 82.89 | 86.69 | 86.69 ± 0.95 |
| **SG-9 宏平均** | **68.75** | **67.67** | **69.25** | **69.20**；宏平均 SD 未估计 |

69.20 是九项五点窗口均值的等权平均，比原报告 69.25 低约 0.05 个百分点，差别主要来自 Countdown 汇总口径。没有把九项 SD 的平均冒充宏平均 SD；本轮不改论文原表。

#### 2.4.3 五点窗口的实际观测值

| 任务 | λ 首—末 | 五个晚期 Pass@8 均值（%） |
|---|---|---|
| Countdown | 1.3862944—2.9957323 | 14.88、15.34、15.72、14.84、15.68 |
| Word Sorting | 33.140519—52.913597 | 62.8125、65.78125、62.96875、62.1875、63.125 |
| Spiral Matrix | 15.44621—34.538776 | 100、100、100、99.6875、100 |
| Mini Sudoku | 157.51641—220 | 97.34375、96.71875、97.34375、96.71875、97.1875 |
| Maze | 517.0163—943.24996 | 77.65625、77.65625、77.65625、77.65625、77.65625 |
| Word Ladder | 4.3784041—5.6643625 | 12.5、13.90625、12.96875、14.53125、12.96875 |
| Knights & Knaves | 230.78793—421.05191 | 69.53125、69.53125、69.53125、69.53125、69.53125 |
| Graph Coloring | 1.3862944—2.6 | 99.6875、100、99.84375、99.84375、100 |
| WikiSQL | 111.55624—172.82958 | 88.125、86.09375、85.625、86.71875、86.875 |

Maze 和 Knights & Knaves 的五个晚期均值相同，故观测配置 SD 为 0；不据此声称训练或评估完全没有随机性。

#### 2.4.4 另列保存的 seed 波动

以下从历史逐 seed CSV 重算，属于不同于配置间 SD 的统计量：

| 任务 | seed 数 | Positive-only 均值 ± seed SD |
|---|---:|---:|
| Countdown | 2 | 13.98 ± 0.54 |
| Word Sorting | 6 | 62.06 ± 3.43 |
| Spiral Matrix | 6 | 100.00 ± 0.00 |
| Mini Sudoku | 6 | 95.91 ± 0.50 |
| Maze | 6 | 76.74 ± 0.87 |
| Word Ladder | 6 | 10.68 ± 1.25 |
| Knights & Knaves | 6 | 67.53 ± 2.00 |
| Graph Coloring | 6 | 98.67 ± 0.58 |
| WikiSQL | 6 | 85.65 ± 0.96 |

Countdown：DRPO 选定参数点的两次晚期均值均为 15.72%，对应样本 seed SD 为 0.00；AsymRE 选定 δᵥ=−1 的两次均值为 14.52%、14.24%，对应 14.38 ± 0.20（seed SD）。不把跨扫描重复的 AsymRE 边界控制行重复计数。

#### 2.4.5 首版结果与剩余事项

DRPO 五点邻域的 SG-9 宏平均为 69.20，对照原报告 AsymRE 68.75、β-TOPR 67.67；这里只作描述性比较。

Countdown、Word Sorting、Graph Coloring 的五个观测值均高于两条基线的原报告分数。Word Ladder 窗口的低端接近 AsymRE；WikiSQL 包含略低于 AsymRE 的观测值。Mini Sudoku、Maze、Knights & Knaves 的 DRPO 邻域均值仍低于 AsymRE。不将结果写成逐任务全面领先。

**历史待补记录（2026-11-10）：当时 AsymRE、β-TOPR 的参数邻域波动结果暂留空。2026-11-12 已收到的是两-seed结果，见 2.4.6；参数邻域 SD 仍不由这些数据替代。**

| 方法 | 参数邻域 SD | 状态 |
|---|---:|---|
| AsymRE | — | 参数邻域 SD 未补；现有 seed SD 已另列 |
| β-TOPR | — | 参数邻域 SD 未补；现有 seed SD 已另列 |

“—”表示该统计量尚未提供，不表示 SD 为 0。当前采用现有 seed 结果，不要求为了填这两格而另开参数扫描。后续补实验仍保留为待办，具体安排届时确认；当前没有启动新实验。

预算匹配的外部干预属于 Q2，尚未逐条讨论。

#### 2.4.6 当前采用的两-seed结果与写法（2026-11-12）

**状态：现有结果先行使用；后续有时间仍需补实验。**

AsymRE、β-TOPR 的 seed 4000、5000 各提供一个任务汇总值。按 n=2 计算均值和样本 SD：s = |x₄₀₀₀ − x₅₀₀₀| / √2，分母采用 n−1。下表基线数值由截图中保留两位小数的结果计算，正式回填时优先使用未舍入原始值。DRPO 数值沿用 2.4.2 的五点 λ 邻域统计。

| 任务 | AsymRE：均值 ± seed SD（n=2） | β-TOPR：均值 ± seed SD（n=2） | DRPO：均值 ± λ 邻域 SD（5 个配置） |
|---|---:|---:|---:|
| Word Sorting | 61.72 ± 1.99 | 59.61 ± 0.78 | 63.38 ± 1.39 |
| Spiral Matrix | 99.92 ± 0.11 | 100.00 ± 0.00 | 99.94 ± 0.14 |
| Mini Sudoku | 97.58 ± 0.33 | 96.88 ± 0.88 | 97.06 ± 0.32 |
| Maze | 77.81 ± 0.66 | 75.86 ± 0.55 | 77.66 ± 0.00 |
| Word Ladder | 12.50 ± 0.23 | 11.41 ± 0.45 | 13.38 ± 0.82 |
| Knights & Knaves | 70.16 ± 0.88 | 68.59 ± 1.77 | 69.53 ± 0.00 |
| Graph Coloring | 98.91 ± 0.22 | 99.38 ± 0.45 | 99.88 ± 0.13 |
| WikiSQL | 85.78 ± 0.44 | 82.89 ± 1.88 | 86.69 ± 0.95 |

单位：均值为百分比，SD 为百分点。β-TOPR 的 Graph Coloring 均值按截图已舍入值计算为 99.38，原论文为 99.37；正式回填时用未舍入结果确认末位舍入。两种写法下九任务宏平均均为 67.67。

八任务均分分别为 AsymRE 75.55、β-TOPR 74.33、DRPO 75.94。加回论文原报告 Countdown 14.38、14.41、15.72 后，九任务均分分别为 68.75、67.67、69.25，与论文主表一致。此处沿用论文 Countdown 原分数，不将其替换为 λ 邻域均值 15.29。

**拟用于补充表的正文句：**

We additionally report mean ± sample standard deviation across training seeds for AsymRE and β-TOPR.

**对应表注：**

AsymRE and β-TOPR: mean ± sample SD across two training seeds (n=2). DRPO: mean ± sample SD across five adjacent λ configurations, reflecting local parameter sensitivity.

这两句是当前拟用表述，不表示已经修改或提交论文。seed SD 与配置 SD 分别标注；现有比较不用于断言三种方法的跨-seed稳定性排名。宏平均分的 SD 尚未估计，不用任务 SD 的平均或平均方差替代。

#### 2.4.7 后续补实验待办（用户已确认，保持开放）

- [ ] 有时间和计算资源后补充独立训练结果，优先补齐三种方法可直接比较的跨-seed波动信息。
- [ ] 具体任务、方法、seed 数量及配置安排届时确认；当前不预设新增实验已完成。
- [ ] 新结果齐备后，重新汇总对应的均值、样本 SD 和实际 seed 数，并更新补充表及回复措辞。

当前阶段使用现有 n=2 结果，不以“已经能够计算 SD”为理由关闭后续补实验待办，也不把计划中的更多 seed 写成已有结果。

#### 2.4.8 W2 后半完整中文回复（当前采用稿）

对应审稿原文：

> The modest structured-generation gains also lack uncertainty estimates in the main table.

以下按“收益幅度与贡献”和“不确定性估计”两个子问题组织拟用于回复的完整文字；提交时附 2.4.6 的补充表及其统计口径说明。两部分均属于 W2 后半，不混入 W1 或 W2 前半的回复。

##### 子问题一：收益幅度与贡献

关于收益幅度，本文的核心贡献是识别 remoteness 在 off-policy 训练崩溃中的关键作用，并提出按 remoteness 衰减负更新的控制策略。同一控制原则覆盖连续 Gaussian 策略（如连续控制任务）和离散 categorical 策略（如语言模型的结构化生成任务）。仅通过这一简单控制，我们就在两类任务中取得了有竞争力的结果，展示了这一思路跨策略族的适用性。

该控制只依赖当前策略的 remoteness，无需取得或估计行为策略密度，也不需要额外参考策略，易于接入已有的策略更新流程。它还可以与 baseline 调整等方法组合：以 AsymRE 的 r−V 作为优势输入，关闭距离衰减即可恢复其原始更新系数；启用衰减则进一步调节远场负样本的影响。

这些结果与理论分析及受控干预共同验证了一种通用、简洁且可组合的控制思路的可行性：根据当前策略下的 remoteness 调节历史负反馈的影响。这为后续方法改进提供了基础，也是这些性能结果所支持的主要贡献。

##### 子问题二：不确定性估计

感谢审稿人指出主表缺少不确定性估计。我们同意应补充波动统计。基于已有独立训练结果，我们报告 AsymRE 和 β-TOPR 在八个迁移任务上的均值 ± 样本标准差，并在表注明确使用两个训练 seed（n=2）。对于 DRPO，我们另行报告五个相邻 λ 配置之间的均值 ± 样本标准差，反映高分参数邻域内的波动；补充表明确区分这两种统计量的来源。

在所报告的五点高分 λ 邻域内，DRPO 在八个迁移任务上的配置 SD 为 0.00–1.39 个百分点，表明这些邻域内的性能波动较小。这项补充提供了参数敏感性的信息，超出了原主表的平均性能比较。

**当前状态：**上述回复基于已汇总结果起草，不表示已经修改或提交论文。更多独立 seed 的实验继续按 2.4.7 保留为待办；当前结果不用于声称统计显著或三种方法的跨-seed稳定性排名。

#### 2.4.9 关于 AsymRE 的原稿依据与本轮表述结论

- 第 2 节 Related Work 已区分 AsymRE 的 baseline-based 正负反馈再平衡与 DRPO 的远场负反馈衰减。第 5 节及附录 C 进一步说明 DRPO 在单样本影响层面依据当前策略的 remoteness 动态加权。
- 附录 E.8 定义论文中比较的 AsymRE 为 A=r−V，并明确它不使用 remoteness。对 verifier 奖励相同的两个负样本，AsymRE 的反馈系数相同；DRPO 则可以根据它们当前的 remoteness 赋予不同的衰减权重。这是在区分系数，不是声称两个样本的实际梯度相同。
- 第 6.2 节的有用排斥与匹配控制实验支持选择性控制的设计，第 6.3 节报告与 AsymRE 的直接任务比较。这些是原稿已有依据；不能把受控 Global-α 对照直接称作 AsymRE 的专门消融。
- 本轮确定的表达重点：定位 remoteness 的关键作用；仅按 remoteness 衰减负更新就取得有竞争力的结果；验证控制思路的可行性，为后续迭代提供基础。用这一贡献主线解释收益的价值；实际性能数字保留在原表与 2.4.2、2.4.6，完整回复不再展开复述。
- 2026-11-13 用户确认进一步强调通用性、简洁性与可组合性。连续 Gaussian 策略后注明“如连续控制任务”，离散 categorical 策略后注明“如语言模型的结构化生成任务”，使策略族与应用场景的对应更直观。
- 兼容性的严格关系：将 A_V=r−V 作为优势输入，并将不衰减边界 λ=0 纳入候选时，ω≡1，因此 Ã=A_V^+−A_V^−=r−V。这里恢复的是 AsymRE 的更新系数；现稿中的 τ 是 remoteness 阈值，与 reward baseline V 不同。联合调节 baseline 与衰减提供进一步优化空间，不据此保证每个任务的测试性能一定不低于 AsymRE；当前没有将联合调参实验写成已完成。
- 简洁性的表述限定于控制本身：无需行为策略密度或额外参考策略，易于叠加到现有策略更新。DPO 的 reference policy 不统一称为预估 behavior policy；不将 TOPR 的 LLM 实验范围写成它只能用于离散空间。现有结果不用于声称调参成本更低或跨-seed方差显著更小；有关 SD 的区分与后续补实验安排保持不变。
- 本轮用户确认删除完整回复中重复均分和任务胜出数量的“具体而言”段落。新增实证信息集中在 SD：基线报告 seed 波动，DRPO 报告所选五点高分 λ 邻域的配置波动；0.00–1.39 个百分点仅概括八个迁移任务的这些配置 SD，不用于推断跨-seed稳定性排名。

### 2.5. 来源与版本

- 审稿材料：用户提供的 Pasted text.txt，Reviewer YPds。
- 论文：当前可取得的 39 页 ICLR 格式 PDF，以及 paper/iclr2027/manuscript_source.tex。
- 读取基准：easonhuo/drpo，main@b949e60e68d1f1832a6e06f94a3c38a32b43d7b2；这里记录此前的读取基准，不声称本次重新核验了最新 HEAD。

投稿页面标注曾在 2026-10-06 修改，目前尚未取得该次提交原始 PDF。正式回复前，节号、图号和表号需对应实际提交版本。

- Hopper 补充材料：experiments/registry.yaml 中 EXT-H-E7-Q2 的结果汇总。

- 本轮 SG 图数据：main@b949e60e68d1f1832a6e06f94a3c38a32b43d7b2 的 paper/iclr2027/figures/data/e8_multitask_exp_response/ 下九项任务 CSV。
- Countdown 逐 seed 原始结果：experiments/results/e8_paper_aligned_linear_scan_round1_pilot/PER_CELL_SUMMARY.csv。
- Positive-only 六-seed来源：drpo@2a19fa1cd23e25dcd3c319e58d286e47a4e10dac 的 experiments/results/e8_multitask_exp_lambda_joint_20260902/CURVE_POINTS/。
- Countdown AsymRE：experiments/results/e8_asymre_deltav_joint_closure_20260722/JOINT_RESPONSE_CURVE.csv。
- 2026-11-12 用户提供的 AsymRE、β-TOPR 八任务两-seed结果截图（seed 4000、5000）；2.4.6 中新增基线 SD 由该截图计算。

## 3. Q1：优势刷新、重要性修正与外部 q/p 轨迹

### 3.1. 审稿原文与讨论范围

> Does the predicted instability persist when advantages are refreshed or off-policy corrections are applied? What are the observed (q/p) trajectories in the external experiments?

Q1 前半包含优势刷新与 off-policy correction 两个更新设置；后半另问外部实验的 q/p 轨迹。本轮先讨论前半，后半尚未展开。

### 3.2. 优势刷新：已明确的回应主线

优势刷新指对已有历史样本重新计算优势，不要求数据本身持续更新。审稿人在 W1 中分别列出 changing critics 与 data refresh，不能将 Q1 的 advantages are refreshed 自动等同于数据刷新。

本文机制不以 critic 估计误差为前提。即使优势完全准确，历史数据上的正、负反馈总量仍可定义为 p_t=Eν[(A^{π_t})^+]、q_t=Eν[(A^{π_t})^-]，也仍可能出现 q_t>p_t。它们表达给定历史分布与当前策略下的反馈平衡，不是由估计误差才产生，也不是训练中永远不变的固有常数。

在固定一个状态／提示的单步奖励例子中，真实优势 A_t(a)=r(a)−V^{π_t}，因此 p_t−q_t=Eν[r]−V^{π_t}。当当前策略的平均表现超过历史数据平均表现时，即使优势准确，仍可能出现历史分布上的负反馈主导。该等式是这个例子的直接关系，不将其不加区分地套到一般多步 Q^{π_t} 的所有情形。

已确认的回答：优势刷新会改变 p_t、q_t，却不自动消除 remoteness 的几何作用，因此刷新后这类失稳机制仍可能存在。固定优势长期定理不直接原封不动移用于任意刷新过程；也不将某一时刻 q_t>p_t 等同于后续必然崩溃。

### 3.3. 重要性修正：已讨论立场与中文回复草稿

#### 3.3.1 已确定立场

- 标准 IS 通常通过当前策略／行为策略的概率比对样本重加权，不等于必须减少某类样本的抽样次数。
- 对同一个历史样本，在行为概率固定且非零时，当前概率趋近零会使其重要性权重趋近零；适当的修正可能控制本文分析的远场失稳路径。不要求加入 IS 后原有失稳现象仍然必然持续。
- 不加 IS 的基准分析用于识别未经修正的历史负反馈复用动力学；该更新在论文 §3.1 中定义为 empirical signed actor field。功能相近支持并列分析两种控制方式，不作为“因此可以忽略 IS”的理由。
- IS 的控制坐标是 target/behavior ratio，DRPO 的控制坐标是当前策略的 remoteness。两者在削弱远负样本影响方面可有共同功能，不能据此认为跨样本权重排序或完整更新目标相同。
- 原稿 §2、§5 末尾及 Appendix C 已有相关定位。以这些内容为回应依据，不把联系写成此前完全遗漏、到 rebuttal 才补上的解释。
- 当前草稿是机制层面的回答，不将 β-TOPR 任务分数改称修正后仍崩溃的直接实验，也不新增“证明 clipped IS/TOPR 一定高方差”的任务。
- 与 W1 的分工：W1 回答为什么无显式 IS 的研究设定合理、以及 IS 与 DRPO 的区别；Q1 回答加入 IS 后原来的失稳路径是否仍然持续。当前回复仅用一句话回指 W1，不重复展开上述两项内容。

#### 3.3.2 重要性修正部分的中文回复草稿（待用户审阅）

加入适当的重要性修正后，本文分析的远场失稳路径可能得到缓解，甚至被阻断。标准 IS 按当前策略与行为策略的概率比 πθ/μ 调整样本权重。对同一个历史负样本，在行为概率固定且非零时，πθ 趋近零会使重要性权重也趋近零，从而可以抑制相应的远场负更新。

因此，加入 IS 后，不一定继续出现未经修正时的失稳现象。重要性修正本身可以发挥削弱远负样本影响的控制作用，这与本文识别出的 remoteness 机制及相应控制思路是一致的。

关于不含 IS 的研究设定及 IS 与 DRPO 的区别，请参见对 W1 的回复和论文附录 C。

### 3.4. Q1 后半：外部实验的 q/p 轨迹

状态：尚未逐条讨论，尚未形成对应回复。前半的机制讨论不替代外部 q/p 轨迹的直接观测。

## 4. 后续尚未逐条展开的意见

- Q2：总负梯度大小匹配后，外部 near/far 干预是否救援。已有 Hopper 结果见 2.3.3 节，尚未完成本条讨论。
- 其他 reviewers：尚未进入本轮逐条讨论。

<!-- END_V12_ARCHIVE -->
