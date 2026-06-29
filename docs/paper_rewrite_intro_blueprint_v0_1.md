# DRPO Introduction 段落级施工图 v0.1

**状态：用户批准进入段落级施工图阶段；本文件是 Introduction 写作约束，不是最终正文。**
**上位蓝图：** `docs/paper_rewrite_outline_v0_7.md`。
**研究状态唯一权威 Master：** `docs/handoff.md`。
**落库基线：** GitHub `main` commit `2054f51719bfd53fe1103a20cfb522b244ad25e0`。
**证据边界：** 不把 planning、静态检查、smoke 或 pilot 升级为正式实验结果。

---

## 0. Introduction 总目标

Introduction 必须形成以下叙事闭环：

> policy optimization 为什么同时使用成功与失败行为
> → 负反馈为什么既有价值又可能失稳
> → fixed/stale data reuse 为什么让远场作用持续
> → 现有方法解决了什么、遗漏了什么
> → Repulsive Dynamics 提供什么新解释
> → DRPO 为什么使用 exponential distance/probability-aware weighting
> → 理论和实验分别贡献什么

Introduction 不提前展开完整证明，不把受控实验条件写成一般 RL 假设，也不把所有负梯度问题都归因于远场机制。

**预计长度：** 950–1150 English words，约 2.0–2.25 双栏 columns。

---

# Paragraph 1 — Background: policy optimization uses successes and failures

## Question answered

为什么负样本和负 advantage 值得研究，而不是只使用正样本？

## Topic sentence

> Policy optimization improves a policy not only by reinforcing successful actions, but also by suppressing actions that perform worse than a reference value.

## Supporting logic

1. Policy-based learning 广泛使用 signed feedback。
2. Positive advantage 提高高价值行为概率。
3. Negative advantage 降低低价值行为概率。
4. 该机制存在于 offline RL、replay-based RL、asynchronous policy learning 和 verifier-guided language-model optimization。
5. 相比只拟合成功行为，利用负反馈具有超越 behavior cloning / positive-only performance limit 的潜力。

## Preferred terminology

- policy optimization
- positive and negative advantages
- behavior cloning
- offline reinforcement learning
- replay
- verifier-guided policy optimization

## Terms deferred to later paragraphs

- Repulsive Dynamics
- far-field divergence
- equilibrium disappearance
- exponential weighting

## Citation targets

- policy gradient / actor–critic foundations
- AWR / IQL or closely related advantage-weighted policy learning
- representative offline or verifier-guided policy learning

Temporary citation slots:

- `[POLICY-GRADIENT-CITATION]`
- `[AWR-IQL-CITATION]`
- `[OFFLINE-OR-VERIFIER-POLICY-LEARNING-CITATION]`

## Target length

110–140 words.

## Forbidden overclaims

- negative samples are inherently harmful
- positive-only methods cannot learn
- all modern RL systems are off-policy
- negative feedback is unique to DRPO

---

# Paragraph 2 — The dual role of negative updates

## Question answered

为什么不能简单删除所有 negative advantage？

## Topic sentence

> Negative-advantage updates play a dual role: they can provide useful contrastive information, yet the same updates may become destabilizing when their gradient contribution is excessively large or repeatedly reinforced.

## Supporting logic

1. Positive updates create attraction toward high-value behavior.
2. Negative updates create repulsion away from low-value behavior.
3. Controlled repulsion may suppress bad actions, redistribute probability mass, provide local decision-boundary information, and move the policy beyond the positive-only performance limit.
4. Excessive or persistent negative updates may push the policy past a high-return region, induce drift, produce abnormal continuous-policy gradients, or repeatedly suppress categorical actions.
5. The central question is not whether to use negative advantages, but when and how strongly to use them.

## Preferred terminology

- positive update
- negative update
- attraction
- repulsion
- gradient magnitude
- policy improvement
- positive-only performance limit

## Forbidden assumptions

- a true local utility radius
- a universally safe local region
- exponential decay of negative-sample utility

## Citation targets

- positive-only / winner-only / asymmetric negative weighting
- learning from failures / useful negative feedback
- low-probability negative-gradient risk

## Target length

120–145 words.

## Forbidden overclaims

- local negative samples are always useful
- far negative samples are always harmful
- positive-only necessarily collapses
- repulsion alone determines task performance

---

# Paragraph 3 — Why fixed or stale data make the problem persistent

## Question answered

为什么单次负更新会成为长期动力学问题？

## Topic sentence

> The challenge becomes particularly acute when historical actions remain in the training distribution as the current policy moves away from them.

## Supporting logic

1. A negative sample's gradient magnitude depends on its negative advantage magnitude and current-policy score.
2. As the policy moves, standardized distance or action surprisal of the fixed sample changes.
3. If the sample leaves the training distribution, its effect need not persist.
4. The mismatch persists under fully offline datasets, replay buffers, stale behavior policies, frozen rollout banks, and repeated optimization of historical trajectories.
5. In the fully offline actor optimization analyzed here, the dataset, advantage labels, and base sample weights are frozen, so the empirical actor objective is stationary.
6. Replay or asynchronous learning may be nonstationary globally, but stale behavior–learner mismatch can persist over multiple updates.

## Required boundary patch

> Greater data reuse does not by itself define the degree of off-policy mismatch, but it makes behavior–policy mismatch more persistent.

> In our fully offline analysis, the empirical actor objective is stationary once the dataset, advantage labels, and base sample weights are frozen.

This distinguishes objective stationarity from the degree of off-policy mismatch.

## Preferred terminology

- behavior–policy mismatch
- stale data / stale behavior policy
- replay
- stationary empirical actor objective
- repeated optimization

Avoid `fixed signed update measure` in the main text; it may remain in proof notes only.

## Target length

135–165 words.

## Forbidden overclaims

- the more off-policy, the more fixed the objective
- all offline RL critics are fixed
- on-policy learning cannot exhibit large negative gradients
- off-policy is defined by repeated data reuse

---

# Paragraph 4 — Existing approaches to controlling negative updates

## Question answered

现有研究已经解决了哪些部分？

## Topic sentence

> Existing methods mitigate harmful negative updates through deletion, global attenuation, clipping, low-probability control, or robust data selection.

## Supporting logic

1. Positive-only / winner-only removes negative advantages, improving stability while potentially discarding useful negative information.
2. Global asymmetric weighting scales all negative updates without distinguishing policy-relative geometry.
3. Policy-ratio clipping or trust-region methods constrain individual policy changes but do not directly characterize repeated far-field pressure.
4. Low-probability / surprisal-aware methods identify risks from rare negative actions or tokens.
5. Filtering and robust data-selection methods remove low-quality samples or select a high-value sub-distribution.

## Writing stance

- First acknowledge what prior work solves.
- Do not claim that earlier methods ignore negative gradients.
- Locate the gap in unified explanation and selective control, not in total absence of prior awareness.

## Citation targets

At least one primary source for each major category:

- positive-only / asymmetric weighting
- PPO/TRPO or related policy-ratio control
- low-probability negative-gradient control
- AWR/IQL/robust filtering
- learning from failures

## Target length

125–155 words.

## Forbidden overclaims

- existing methods ignore negative gradients
- clipping is ineffective
- positive-only always sacrifices performance
- low-probability methods are equivalent to DRPO

---

# Paragraph 5 — The unresolved theoretical gap

## Question answered

为什么还需要 Repulsive Dynamics？

## Topic sentence

> What remains unclear is not merely how to reduce negative updates, but when repulsion improves a policy, when it becomes destructive, and how this transition depends on policy geometry.

## Supporting logic

The paragraph should pose five questions:

1. Why can negative updates improve a policy rather than merely destabilize it?
2. Why can positive-only optimization stop near observed positive behavior?
3. Why can samples with identical negative advantages produce different gradients under different current policies?
4. When does a finite stable equilibrium exist, approach the feasible boundary, or disappear?
5. Do Gaussian and categorical policies share the same attraction–repulsion structure despite different score behavior?

Prior work often studies advantage sign, action probability, update magnitude, staleness, or data quality separately. The missing object is the competition between positive and negative policy updates under a stationary off-policy actor objective.

## First definition of far field

> Far-field negative samples are actions with large standardized distance under a continuous policy, or low current probability under a categorical policy.

Do not introduce parallel umbrella terms such as `policy-relative remoteness`, `repulsive frontier`, or `utility horizon`.

## Target length

115–140 words.

## Forbidden overclaims

- no prior work studies negative gradients
- distance is the only source of instability
- far-field samples contain no useful information
- Gaussian and categorical policies have identical score behavior

---

# Paragraph 6 — Repulsive Dynamics and nonlinear policy behavior

## Question answered

本文发现了什么新的理论结构？

## Topic sentence

> We show that positive attraction and negative repulsion can form a finite equilibrium, and that increasing far-field negative gradients can move this equilibrium toward the feasible-set boundary or eliminate it altogether.

## Supporting logic

1. Under a stationary empirical actor objective, positive and negative advantage updates define a competing gradient field.
2. When positive restoring strength dominates, a finite equilibrium can exist.
3. Controlled repulsion can move the policy beyond the positive-only performance limit, producing stable extrapolation.
4. As negative-update strength increases, the equilibrium moves outward, approaches the mean-parameter feasible-set boundary, or ceases to exist.
5. Gaussian policies have nonlinear mean/covariance score geometry; standardized-distance growth can yield unbounded finite-order score growth and gradient-amplitude runaway.
6. Categorical direct-logit scores are bounded, but repeated updates can continue widening logit gaps and drive probabilities toward the simplex boundary.
7. The theory therefore gives a common equilibrium structure with policy-family-specific failure modes.

## How to mention nonlinearity

The paragraph must make clear that:

- the paper is not limited to a linear one-parameter bandit;
- C-U1 uses a shared nonlinear Gaussian actor;
- Gaussian mean and covariance scores are nonlinear functions of parameters and standardized distance;
- DRPO's exponential weighting is a nonlinear attenuation rule.

Do not claim that nonlinearity alone causes divergence or that neural networks amplify every far-field sample.

## Figure reference

> Figure 1 summarizes the transition from the positive-only limit, to useful repulsion, and finally to persistent far-field instability.

## Target length

155–185 words.

## Forbidden overclaims

- the theorem proves task-performance collapse
- every boundary approach is harmful
- Gaussian variance always expands
- categorical score norm diverges
- all policy families have identical dynamics

---

# Paragraph 7 — DRPO and exponential weighting

## Question answered

DRPO 如何针对远场问题，而不是删除全部负更新？

## Topic sentence

> Motivated by this analysis, DRPO selectively attenuates negative updates according to standardized distance or action surprisal.

## Supporting logic

1. DRPO preserves the unmodified positive-advantage update.
2. Negative advantages receive an exponential weight:
   \[
   w_\lambda(r)=\exp(-\lambda r).
   \]
3. The weight is not a model of negative-sample utility.
4. The method does not assume exponential utility decay or a true safe radius.
5. It is a gradient-control envelope: exponential weighting dominates finite-order score growth and makes the weighted far-field negative gradient vanish under the proposition's boundedness/moment conditions.
6. Global weighting scales all negatives equally; linear taper uses fixed-rate decay and may reach hard truncation; exponential weighting provides smooth, positive, stronger tail attenuation.

## Connection to Optimistic DRO

Use only one sentence in the Introduction:

> DRPO retains the optimistic distributional-selection view of the original formulation while replacing its conservative hard endpoint with a smooth distance- or probability-aware negative weighting rule.

The full DRO/CVaR derivation belongs in the Method section and appendix.

## Target length

130–160 words.

## Forbidden overclaims

- negative utility decays exponentially
- exponential weighting is theoretically optimal
- DRPO solves every form of negative-gradient imbalance
- DRPO replaces SFT
- DRPO is guaranteed to outperform Global, Linear, or Hard filtering

---

# Paragraph 8 — Evidence map and contributions

## Question answered

理论与方法如何通过分层证据验证？

## Topic sentence

> We evaluate the proposed account through a sequence of controlled mechanism tests and external policy-learning benchmarks.

## Supporting logic

1. **C-U1:** nonlinear continuous Gaussian controlled environment; identifies gradient source and causal far-field transmission. Its generalization result is same-distribution held-out-context / unseen-state generalization, never OOD without a registered shift.
2. **D-U1:** categorical semantic controlled environment; studies rare negative actions, semantic sharing, and persistent probability suppression.
3. **Online:** tests whether fresh data collection mitigates persistent accumulation; it does not redefine the controlled mechanism claim.
4. **D4RL:** continuous-control external validity, including normalized return, best checkpoint, terminal checkpoint, and separate failure audits.
5. **Countdown:** controlled high-staleness off-policy stress test using a frozen verifier-labeled rollout bank; it is not claimed to represent every LLM RL workflow.

## Contribution statements

**First**, derive equilibrium, boundary, and divergence conditions for repulsive policy updates under a stationary empirical actor objective.

**Second**, causally isolate standardized distance and low action probability from advantage magnitude in continuous and categorical controlled environments.

**Third**, propose DRPO, an exponential distance-/probability-aware weighting rule that controls far-field negative gradients without discarding all negative feedback.

**Fourth**, evaluate mechanism and method across Online, D4RL, and Countdown settings, with terminal audits separating task-performance failure, distribution-boundary events, and NaN/Inf numerical failure.

## Target length

170–210 words.

## Forbidden overclaims

- C-U1 demonstrates OOD generalization
- controlled experiments prove D4RL performance
- Countdown represents all LLM RL
- pilot or smoke results are formal evidence
- task failure, support/variance boundary, and numerical failure are interchangeable

---

# 9. Paragraph transitions

## P1 → P2

> This makes negative feedback potentially more informative than pure imitation, but also introduces a qualitatively different optimization force.

## P2 → P3

> The risk becomes persistent when the same negative actions remain in the training distribution after the policy has moved away from them.

## P3 → P4

> A growing body of work has therefore sought to reduce or reshape negative updates.

## P4 → P5

> These approaches improve stability, but they do not fully characterize the transition from useful repulsion to destructive far-field dynamics.

## P5 → P6

> We address this gap by analyzing the equilibrium structure induced by competing positive and negative policy updates.

## P6 → P7

> This analysis suggests controlling negative updates according to their current policy geometry rather than their sign alone.

## P7 → P8

> We test both the mechanism and the resulting controller across increasingly realistic settings.

---

# 10. Global consistency constraints

## Theory scope

- The main theorem analyzes a stationary empirical actor objective.
- Fully offline actor optimization with fixed data, frozen advantages, and fixed base weights satisfies the condition exactly.
- Replay, changing buffers, online collection, or jointly evolving actor–critic systems are outside the theorem's global guarantee; at most the theorem is a local diagnostic there.
- Fixed advantage is not a standalone general RL setting in Preliminaries; it is explained as a controlled experimental design in C-U1/D-U1.

## Experimental scope

- Product-manifold experiments identify the source of large far-field gradients.
- Nonlinear Gaussian intervention experiments test causal transmission into drift and collapse.
- C-U1/D-U1 are controlled mechanism environments.
- D4RL/Countdown provide external validity.
- Online studies the fresh-data boundary and must be registered before execution.

## Method scope

- Exponential weighting controls weighted gradient magnitude; it does not model utility.
- No real local safety region or threshold radius is assumed.
- The paper does not claim to solve every critic, distribution-shift, exploration, or negative-gradient issue.
- Exp method ranking remains empirical until formal D4RL/Countdown/Online results exist.

## Terminology scope

Prefer established terms:

- stationary empirical actor objective
- behavior–policy mismatch
- standardized / Mahalanobis distance
- negative log-probability / surprisal
- policy-ratio clipping
- advantage-weighted regression
- replay / staleness
- simplex boundary
- task-performance collapse
- numerical failure

Retain only necessary paper concepts:

- Repulsive Dynamics
- far-field negative samples
- attraction and repulsion
- stable extrapolation

Avoid in the main text:

- policy-relative remoteness
- utility radius
- repulsive frontier
- probability-boundary dynamics
- signed update measure
- support collapse unless strictly defined and separated from task performance

---

# 11. Evidence status

## Supported by existing theory or controlled evidence

- fixed-advantage score-geometry mechanism
- Gaussian far-field gradient growth
- categorical persistent suppression
- positive-only performance limit
- useful-to-harmful transition
- near/far causal interventions

## Formal work still required

- complete proof of equilibrium/boundary/divergence theorem
- discrete spectral-radius condition
- Gaussian covariance corollary
- categorical simplex-boundary corollary
- bounded far-field gradient proposition for Exp
- citation-by-citation primary-source audit
- Figure 1 academic redraw

## Formal external results still pending

- Online
- D4RL
- Countdown

Until those results are complete, the Introduction may say `we study` or `we evaluate`, but it must not state unobserved improvements, rankings, or effect sizes.

---

# 12. Recommended drafting order

Do not draft the Introduction mechanically from Paragraph 1 to Paragraph 8. Use this order:

1. Paragraph 5 — research gap
2. Paragraph 6 — theory
3. Paragraph 7 — method
4. Paragraph 8 — evidence and contributions
5. Paragraphs 1–4 — background and prior approaches
6. Final pass — opening, transitions, terminology, citation density, and page fit

This order fixes the paper's actual claims first and prevents the background from becoming longer or broader than the contribution.
