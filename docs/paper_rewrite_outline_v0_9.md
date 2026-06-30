# DRPO 论文重写 v0.9：Guidance-reviewed canonical outline

**状态：active canonical outline；已获用户批准并通过 `docs/manuscript/RL_PAPER_WRITING_GUIDANCE.md` 审阅。**

**作用：** 论文结构的 active contract；`docs/handoff.md` 仍是研究状态、冻结协议与实验结论的唯一权威 Master。

**基线：** GitHub `main` commit `84edc2aa0b2f258033ddf2ef9aaf98e7a89a6edd`。

**本次性质：** 文档与论文结构更新，不构成任何新实验结果，不改变 seeds、阈值、数据规模、训练配置或实验职责。

**Introduction 施工图：** `docs/paper_rewrite_intro_blueprint_v0_4.md`。

**历史版本：** v0.7/v0.3 及更早文件完整保留；v0.8/v0.2 的 invalid reverse-alignment 记录继续保留为 superseded provenance。

---

# 0. v0.9 的总原则

## 0.1 全文唯一主线

> Negative feedback enables policy improvement beyond positive-only learning, but repeated optimization of far-field negative actions can amplify repulsion until the policy loses a finite equilibrium. DRPO preserves useful negative feedback while suppressing the destructive far-field tail.

对应中文主线：

> 负反馈能够帮助策略突破 Positive-only 上限；但历史负动作被反复复用并进入远场后，排斥贡献会被异常放大，使稳定平衡接近边界并最终消失。DRPO 保留有用的局部负反馈，同时压制破坏性的远场尾部。

## 0.2 全文证据链


a. negative feedback has value；

b. terminal-audited Hopper/Countdown results provide a compact external phenomenon anchor when available；

c. quality–distance factorization proves distance is an independent amplifier；

d. targeted near/far and common/rare interventions identify causal transmission；

e. Theorem 1 explains Positive-only → stable extrapolation → boundary → no finite equilibrium；

f. DRPO attenuates the same aggregate negative term, and external task results close the loop。

## 0.3 Product manifold 的正式位置

`Product manifold` 是历史开发阶段用于质量–距离解耦的机制构造，不作为新版论文的第三个主环境。

正文统一写为：

> **C-U1 E1 quality–distance factorized source-isolation protocol**

历史 Product-manifold 代码、结果与来源关系保留在 appendix/provenance，用于说明该识别思想的演进。主要受控环境仍只有 C-U1 与 D-U1。

## 0.4 必须正面回答的逻辑漏洞

审稿人最自然的替代解释是：

> 远场样本的梯度更大，是否只是因为远场样本更差、advantage 绝对值更大？

主文必须明确：paper-facing source isolation 需要在相同 state/context、quality coordinate / action semantics、reward、advantage severity、sample count 和 base coefficient 下，只改变 policy-relative distance。现有 C-U1 E1 已正式证明 `|A|` 跨距离匹配并在 policy-output score 上得到 distance amplification；由于当前等 reward 轮廓点同时改变动作方向，full-parameter 归因必须由同射线 radial probe 或 Jacobian-gain decomposition 补齐并在计算前登记。因而论文最终要建立：

> **Far-field distance is an independent gradient amplifier; the far/near gradient gap cannot be reduced to far samples having worse advantages.**

不得升级为“distance 是唯一因素”；advantage severity、方向一致性与样本数量仍是独立因子。

## 0.5 fixed advantage 的位置

fixed advantage 仅是 C-U1/D-U1 受控识别中的实验控制，用于排除 critic feedback、relabeling 与 policy-dependent weighting。它不是 far-field 理论的前提，也不作为论文 scope 反复解释。

---

# 1. 标题、摘要与总体定位

## 1.1 推荐标题

**Breaking the Curse of Repulsion: Distributionally Robust Policy Optimization for Off-Policy Learning**

## 1.2 一句话 contribution

> We identify a useful-to-destructive transition in negative policy updates: controlled repulsion creates stable improvement beyond positive-only learning, whereas far-field amplification can remove the finite equilibrium; DRPO directly attenuates the far-field component of the aggregate negative term that drives this transition.

## 1.3 Abstract move sequence

1. **Problem:** negative feedback is necessary for policy improvement but becomes unstable under repeated historical-data reuse.
2. **Missing mechanism:** prior controls do not explain the transition from useful local repulsion to destructive far-field repulsion.
3. **Theory:** characterize Positive-only, stable extrapolation, boundary approach, and loss of finite equilibrium.
4. **Identification:** isolate sample badness from policy-relative distance and causally intervene on far versus near negatives.
5. **Method:** DRPO exponentially attenuates the far-field component of the theorem’s aggregate negative term.
6. **Evidence:** a reality anchor from formal Hopper/Countdown diagnostics, controlled continuous/categorical identification, and external task closure, using only terminal-audited results available at writing time.
7. **Implication:** the correct target is far-field negative contribution, not negative feedback as a whole.

No limitations paragraph, fixed-advantage discussion, or unverified external result appears in the abstract.

---

# 2. Introduction — 约 2.0 columns

Introduction uses exactly six paragraph contracts.

<!-- MANUSCRIPT:BEGIN INTRO-P01 -->
## [INTRO-P01] Negative Feedback as a Policy-Improvement Resource

Establish that policy optimization learns from both success and failure. Positive updates reinforce observed successful behavior; negative updates suppress bad modes, shape boundaries, and may reallocate probability toward better alternatives. Positive-only learning is a stable reference but can stop at the observed positive target. This paragraph motivates the scientific value of negative feedback without discussing fixed advantage, far field, or theorem assumptions.
<!-- MANUSCRIPT:END INTRO-P01 -->

<!-- MANUSCRIPT:BEGIN INTRO-P02 -->
## [INTRO-P02] Historical Reuse Turns Local Feedback into Persistent Repulsion

Introduce the core failure mechanism. A negative action may initially provide relevant local feedback, but after the policy moves away, the same historical action can remain in offline data, replay, stale rollouts, or asynchronous trajectories. Its learner-relative distance or rarity grows, while the update continues. Gaussian scores can grow with standardized distance; categorical direct-logit scores are bounded but can persistently suppress probability. The paragraph ends with the useful-to-destructive transition, not with a scope disclaimer.
<!-- MANUSCRIPT:END INTRO-P02 -->

<!-- MANUSCRIPT:BEGIN INTRO-P03 -->
## [INTRO-P03] The Missing Link: Separating Badness from Distance

State the central identification problem: real data often couples low reward, large negative advantage, rarity, and distance, so a large far-field gradient could be dismissed as a consequence of worse samples. Present the paper’s decisive control: match context, semantics, reward/advantage severity, count, and base coefficients, varying only policy-relative distance or rarity. Preview the result that distance independently amplifies negative gradients. This paragraph makes quality–distance isolation a first-class contribution rather than an implementation detail.
<!-- MANUSCRIPT:END INTRO-P03 -->

<!-- MANUSCRIPT:BEGIN INTRO-P04 -->
## [INTRO-P04] Repulsive Dynamics Explains Stable Extrapolation and Equilibrium Loss

Present the theoretical advance. Positive-only optimization converges to the positive target; moderate negative repulsion shifts a finite stable equilibrium beyond that target; increasing aggregate negative contribution moves the target toward the feasible boundary and can eliminate a finite equilibrium. Gaussian and categorical policies share this aggregate phase structure while expressing different boundary dynamics. Do not expand proof details, spectral conditions, or defensive claims.
<!-- MANUSCRIPT:END INTRO-P04 -->

<!-- MANUSCRIPT:BEGIN INTRO-P05 -->
## [INTRO-P05] DRPO Controls the Destabilizing Far-Field Term

Introduce DRPO as the direct method consequence of Theorem 1. The theorem identifies the aggregate negative moment as the term that enables extrapolation but eventually pushes the equilibrium toward the boundary. DRPO replaces its far-field component with exponential distance/surprisal weighting, retaining stronger local negative feedback while making weighted far-field gradients vanish under finite-order score growth. Exp is a gradient-control envelope, not a model of sample utility.
<!-- MANUSCRIPT:END INTRO-P05 -->

<!-- MANUSCRIPT:BEGIN INTRO-P06 -->
## [INTRO-P06] Evidence Chain and Contributions

Summarize the final evidence chain by claim: terminal-audited Hopper/Countdown diagnostics first anchor the phenomenon in external policy learning; C-U1 E1 then separates distance from badness; C-U1/D-U1 interventions identify far-field transmission; E4/E6 test the useful-to-destructive phase transition; matched-budget comparisons test DRPO; external task results close the loop. Until the external diagnostics are formal, keep their claims and numbers explicitly `TBD`. End with four contributions: Repulsive Dynamics theory, quality–distance causal identification, DRPO, and multi-level validation with terminal audits. Do not present Product manifold as a separate primary environment or predeclare method rankings.
<!-- MANUSCRIPT:END INTRO-P06 -->

---

# 3. Related Work — 约 0.75 column

Three methodological paragraphs:

1. **Learning from negative or suboptimal behavior.** Positive-only, advantage-weighted learning, failure learning, and negative reinforcement establish that negative feedback can be useful. The unresolved question is when useful repulsion becomes dynamically destructive.
2. **Off-policy, stale, and low-probability updates.** Clipping, importance correction, low-probability-token control, and stale-policy methods regulate update scale or support. The unresolved link is repeated policy-relative far-field movement and finite-equilibrium loss.
3. **Robust offline policy learning.** CQL, IQL, TD3+BC/ReBRAC, AWR-like actor fitting, and data filtering control value extrapolation, support, or data quality. DRPO instead targets the far-field component of signed actor updates.

Novelty is the complete bridge: useful negative feedback → quality–distance isolation → far-field causal transmission → equilibrium loss → selective control. Never claim first discovery of harmful negative gradients or low-probability risk.

---

# 4. Problem Setup — 约 0.75 column

## 4.1 Signed policy update

\[
\mathbf F(\theta)=
\mathbb E_{\nu}
[\widehat A(s,a)\nabla_\theta\log\pi_\theta(a\mid s)].
\]

Define:

\[
A^+=\max(\widehat A,0),\qquad
A^-=\max(-\widehat A,0),
\]

\[
\mathbf F(\theta)=\mathbf F^+(\theta)-\mathbf F^-(\theta).
\]

For a negative sample:

\[
\|g_i^-\|=A_i^-\|\nabla_\theta\log\pi_\theta(a_i\mid s_i)\|.
\]

This factorization separates advantage severity from score geometry.

## 4.2 Policy-relative far field

- Gaussian: large standardized/Mahalanobis distance or calibrated negative log-density;
- categorical: low current action probability / high surprisal;
- sequence models: normalized token/completion NLL where registered.

Far field is a dynamic relation to the current learner, not a permanent data label.

---

# 5. Repulsive Dynamics — 约 2.5 columns

## 5.1 Per-sample far-field amplification

Gaussian mean score:

\[
\nabla_\mu\log\pi(a\mid s)=\Sigma^{-1}(a-\mu).
\]

Distance can independently enlarge the score. With learned covariance, standardized distance couples mean repulsion and support contraction.

Categorical score:

\[
\nabla_z\log\pi(a\mid s)=e_a-\pi(\cdot\mid s).
\]

Its Euclidean logit score is bounded, but repeated negative updates can continuously increase logit gaps and push probabilities toward the simplex boundary.

## 5.2 Aggregate positive–negative competition

For a regular minimal exponential-family policy:

\[
\pi_\eta(a)=h(a)\exp\{\eta^\top T(a)-\psi(\eta)\}.
\]

Let `p,q` be aggregate positive and negative update masses, and `m_+,m_-` their sufficient-statistic targets. Then:

\[
\mathbf F(\eta)=p\mathbf m_+-q\mathbf m_--(p-q)\nabla\psi(\eta).
\]

These quantities define the mathematical objective of the theorem. No fixed-advantage scope discussion is introduced.

## 5.3 Theorem 1: Stable Extrapolation and Loss of Finite Equilibrium

### Positive-only

For `q=0`:

\[
\nabla\psi(\eta^\star)=\mathbf m_+.
\]

### Stable extrapolation

For `0<q<p`, if

\[
\mathbf m^\star=
\frac{p\mathbf m_+-q\mathbf m_-}{p-q}
\]

lies in the interior of the mean-parameter space, a unique finite equilibrium exists, and

\[
\mathbf m^\star-\mathbf m_+
=
\frac{q}{p-q}(\mathbf m_+-\mathbf m_-).
\]

Negative repulsion therefore shifts the stable target beyond the Positive-only target.

### Boundary and no finite equilibrium

As the magnitude or outward position of `q m_-` increases, `m*` approaches the feasible boundary. A boundary target requires a degenerate limiting distribution or diverging natural parameter; a target outside the feasible region has no finite equilibrium. If `p=q` and the signed targets do not cancel, the restoring term disappears and persistent drift remains.

The local Jacobian at an interior equilibrium is:

\[
J_F(\eta^\star)=-(p-q)\nabla^2\psi(\eta^\star).
\]

Full proof, discrete step-size bounds, exceptional cancellation cases, and covariance details go to the appendix.

## 5.4 Testable predictions and experiment mapping

| Regime | Prediction | Primary evidence |
|---|---|---|
| Positive-only | stable target near positive behavior, but no repulsive extrapolation | C-U1 E4 / D-U1 E6 |
| Stable extrapolation | controlled negatives move the terminal policy beyond Positive-only and improve held-out-context/unseen-state performance | C-U1 E4 / D-U1 E6 |
| Boundary approach | stronger/farther aggregate negative contribution drives covariance or probabilities toward a boundary | C-U1 E3/E4; D-U1 E5/E6 |
| No finite equilibrium | terminal slope/residual remains nonzero or the policy runs away | C-U1 E3/E4 terminal audit |
| DRPO recovery | far-field attenuation restores a finite stable regime while retaining useful negative updates | E4 taper/budget-match; D-U1 controls |

Task collapse, boundary events, and NaN/Inf remain separately measured.

## 5.5 Corollaries

- **Gaussian:** finite mean/covariance interior; mean escape or covariance boundary; distance-amplified score.
- **Categorical:** finite logits interior; probabilities approach 0/1 boundary; bounded but persistent suppression.

---

# 6. Distributionally Robust Policy Optimization — 约 1.5 columns

## 6.1 DRPO update

For negative samples:

\[
w_i^-=\exp(-\lambda r_i),
\]

\[
\mathbf F_{\mathrm{DRPO}}
=
\mathbb E[A^+\nabla\log\pi-A^-e^{-\lambda r}\nabla\log\pi].
\]

Use family-appropriate registered `r`: Gaussian standardized distance/calibrated NLL; categorical surprisal; sequence-model normalized NLL.

## 6.2 Direct bridge to Theorem 1

Uncontrolled negative moment:

\[
q\mathbf m_-=\mathbb E[A^-T(a)].
\]

DRPO negative moment:

\[
q_\lambda\mathbf m_{-,\lambda}
=
\mathbb E[A^-e^{-\lambda r_\theta(s,a)}T(a)].
\]

The controlled equilibrium target becomes:

\[
\mathbf m_\lambda^\star
=
\frac{p\mathbf m_+-q_\lambda\mathbf m_{-,\lambda}}
{p-q_\lambda}.
\]

This is the central theory–method bridge: DRPO modifies the same aggregate negative term that both enables stable extrapolation and drives boundary crossing.

## 6.3 Proposition 2: Vanishing weighted far-field gradient

If

\[
\|\nabla\log\pi\|\le C(1+r)^k,
\]

then

\[
e^{-\lambda r}\|\nabla\log\pi\|
\le Ce^{-\lambda r}(1+r)^k\to0.
\]

Therefore the far-field tail cannot retain polynomially growing gradient magnitude after exponential weighting.

## 6.4 Method family

- Uncontrolled: `w=1`;
- Positive-only: `w=0`;
- Global: `w=alpha`;
- Linear taper;
- Hard cap/filter;
- DRPO Exp;
- SBRC/Hybrid only when registered and budget-matched.

No ranking is prespecified. Optimistic DRO and historical hard filtering are retained as method lineage and appendix material, not as the only possible solution.

---

# 7. Experiments — 约 7.75 columns

The final paper follows **external anchor → controlled explanation → external closure**. Every subsection follows: claim → rival explanation → control/intervention → metric → verdict → status.

## 7.1 Environments and evidence roles

| Environment / protocol | Role | What it can establish |
|---|---|---|
| Hopper/D4RL | external continuous control with learned critic and real offline data | external mechanism signature and task effect after formal execution |
| Countdown/Qwen | external sequence policy with shared Transformer parameters | rare-negative/shared-parameter signature and task effect after formal execution |
| C-U1 E1 quality–distance factorized protocol | controlled continuous source isolation | whether distance independently amplifies score/negative gradients at matched badness |
| C-U1 E2–E4 | nonlinear Gaussian dynamics and interventions | temporal transmission, boundary events, task collapse, stable extrapolation, method recovery |
| D-U1 E5–E6 | shared categorical representation | persistent suppression, probability boundary, useful/harmful negative feedback |
| Historical Product-manifold construction | development provenance / appendix | the earlier source-isolation construction; not a third primary environment |

C-U1 test states are independently sampled from the same distribution as training states. Use only held-out-context/unseen-state generalization language.

## 7.2 RQ1: Does the phenomenon appear in external policy learning?

This section is the **reality anchor**. It appears before the controlled environments so that the paper does not ask a simulator to establish the phenomenon’s existence.

### Hopper

Report distance-binned negative-gradient magnitude, far/near and negative/positive ratios, gradient-anomaly/drift/return timing, learned-critic provenance, and terminal audit.

### Countdown

Report surprisal-binned negative token/completion gradients, rare/common contribution, target-probability change under replay/staleness, greedy success, pass@k, valid rate, and terminal degradation.

### Verdict discipline

External tasks establish that disproportionate far-field or rare-negative influence occurs in realistic policy learning. They do not isolate whether the effect comes from worse advantages, distance, critic error, staleness, or shared parameters; RQ2–RQ4 perform that identification. Hopper and Countdown remain `TBD` until their formal terminal-audited results exist.

## 7.3 RQ2: Is distance an independent source of large negative gradients?

### Claim

At matched sample badness, far-field distance independently amplifies the negative gradient.

### Rival explanation

Far samples have worse rewards or larger negative advantages.

### Control

Within the paper-facing C-U1 E1 closure, match state/context, reward, advantage severity, sample count, base coefficient, and—where full-parameter attribution is claimed—action direction/semantics; vary only radius/current-policy distance. Report the exact far/near advantage ratio alongside output-score and full-parameter gradient ratios.

The current formal E1 result already closes the `|A|`-matched **policy-output score** claim. Its equal-reward contour changes action direction, so the complete full-parameter ratio cannot yet be attributed solely to distance. Before the final paper makes that stronger claim, register and complete either:

1. a same-state, same-ray radial probe that varies only radius; or
2. a Jacobian-gain decomposition separating output-score amplification from network-Jacobian/directional effects.

This is a diagnostic extension inside C-U1 E1, not a new environment.

### Metrics

- reward/advantage equality error;
- far/near `|A|` ratio;
- output-score ratio;
- per-sample and aggregate full-parameter negative-gradient ratio;
- direction/coherence and Jacobian-gain diagnostics.

### Verdict language

Distance is an independent amplifier; the effect is not explained by far samples being worse. Do not claim distance is the sole factor.

### Categorical analogue

Audit D-U1 so common/rare comparisons match advantage severity, semantics, and count. Register only the missing diagnostic if the existing protocol is insufficient.

## 7.4 RQ3: Do far-field negative gradients causally transmit into instability?

C-U1 interventions:

- Uncontrolled/Baseline;
- Near-zero;
- Far-zero;
- Far-cap;
- equal-budget Global;
- registered transfer controls.

D-U1 interventions:

- common-negative suppression;
- rare-negative suppression;
- equal-budget global control;
- registered transfer controls.

Report onset order, raw and weighted near/far contributions, optimizer update norm, policy drift, held-out-context/unseen-state reward, support/variance/probability boundary, and NaN/Inf separately.

## 7.5 RQ4: Does training follow Theorem 1’s phase transition?

Map negative-strength scans to:

1. Positive-only target;
2. stable extrapolation beyond Positive-only;
3. approach to covariance/probability boundary;
4. persistent drift or no finite equilibrium.

Required metrics:

- terminal policy/equilibrium location;
- performance relative to Positive-only;
- terminal slope/field residual;
- boundary contact;
- 2× horizon/long-run audit where registered;
- separate failure-event categories.

E4’s beneficial branch must not be upgraded beyond its actual certification status. Use the handoff’s current finite-step/long-run labels.

## 7.6 RQ5: Does DRPO preserve useful repulsion while preventing instability?

Compare only registered methods with paired seeds, identical initialization/data, matched or reported negative-gradient budgets, best/terminal split, and terminal audit.

Primary questions:

1. Does the method exceed Positive-only?
2. Does it retain near-field negative contribution?
3. Does it suppress far-field contribution?
4. Does it remain in a finite stable regime?
5. How do task collapse, boundary event, and NaN/Inf change?

Current status and remaining gates:

- `C-U1-E4-TAPER-BUDGET-MATCH-01` is completed and finite-step validated. It supports a finite-horizon matched-raw-negative-gradient selectivity claim, not a steady-state ranking.
- Next, freeze the deterministic shortlist in a separate update, implement exact actor+Adam-state continuation, run `C-U1-E4-TAPER-CONV-01`, and only then use untouched seeds `130–149` for confirmation.
- Terminal useful-near retention was undefined in Budget-Match; do not retroactively claim it was measured.

## 7.7 RQ6: Does DRPO improve external tasks?

This is the **reality closure**.

### D4RL/Hopper family

Use the registered external route and datasets. Report normalized return, mean ± uncertainty, best checkpoint, terminal checkpoint, task-collapse rate, and mechanism diagnostics.

### Countdown

Use same initialization, rollout/replay bank, seeds, verifier, and selection protocol. Report greedy success, pass@k, valid rate, best/terminal, and rare-negative diagnostics.

Do not write results until formal execution and terminal audit are complete.

## 7.8 Experiment supplementation required by the new framework

No new primary environment is added. The new paper architecture requires the following evidence closures:

1. **C-U1 E1 full-parameter attribution closure:** register a same-ray radial probe or Jacobian-gain decomposition; the existing formal E1 output-score result remains valid.
2. **E4 terminal method closure:** Budget-Match is already complete; proceed through deterministic shortlist freeze, exact actor+Adam continuation, `TAPER-CONV`, and untouched-seed confirmation before steady-state ranking.
3. **External reality anchor and closure:** complete Hopper E7 and Countdown formal routes, separating mechanism signatures from task performance.
4. **D-U1 isolation audit:** verify whether common/rare comparisons already match badness, semantics, and count; register only the missing diagnostic.
5. **Theory–method observable:** report an empirical proxy for the aggregate negative term before/after DRPO—prefer distance-binned weighted contribution or signed-target shift from existing outputs; register any new computation before execution.

These are evidence closures within existing experiment responsibilities, not new scientific claims or a license to change frozen protocols.

---

# 8. Implications and Conclusion — 约 0.75 column

## 8.1 Transferable principle

Negative feedback is a policy-improvement resource with a dynamical failure mode. The correct control target is the far-field negative contribution, not the sign of feedback itself.

## 8.2 Continuous–categorical synthesis

Gaussian policies express persistent repulsion through distance-amplified scores and mean/covariance movement; categorical policies express it through bounded but persistent probability suppression. Both can move an interior equilibrium toward a feasibility boundary.

## 8.3 Three-sentence conclusion

1. Negative feedback can move a policy beyond the Positive-only target.
2. Repeated far-field repulsion can eliminate the finite equilibrium that makes this improvement stable.
3. DRPO preserves useful negative feedback while suppressing the far-field tail responsible for instability.

No standalone defensive scope list appears in the main conclusion.

---

# 9. Main-paper visual plan

| Item | Core claim | Status rule |
|---|---|---|
| Figure 1 | Full arc: external symptom → badness-matched distance amplification → stable extrapolation/boundary → DRPO recovery | concept only until academic redraw |
| Figure 2 | Hopper/Countdown external mechanism anchor | `TBD` until formal terminal-audited results |
| Figure 3 | C-U1/D-U1 source isolation: badness matched while distance/rarity changes | full-parameter distance attribution requires the registered E1 closure |
| Figure 4 | targeted near/far and common/rare causal interventions | use only closure-qualified results |
| Figure 5 | continuous/categorical phase diagram aligned to Theorem 1 | separate boundary and task failure |
| Figure 6 | DRPO modifies aggregate negative contribution and closes external tasks | method fairness and terminal audit required |
| Table 1 | environment/evidence roles | Product manifold listed only as historical provenance |
| Table 2 | controlled claim–evidence results | status column mandatory |
| Table 3 | D4RL/Hopper external results | `TBD` until formal |
| Table 4 | Countdown external results | `TBD` until formal |

Every caption states control, visible pattern, supported conclusion, and result status.

---

# 10. Column budget

| Section | Columns |
|---|---:|
| Introduction | 2.00 |
| Related Work | 0.75 |
| Problem Setup | 0.75 |
| Repulsive Dynamics | 2.50 |
| DRPO | 1.50 |
| Experiments | 7.75 |
| Implications + Conclusion | 0.75 |
| **Total** | **16.00** |

---

# 11. Guidance review summary

The v0.9 rewrite addresses the structural problems identified in v0.7:

- removes fixed-advantage/stationarity defense from the theory and Introduction;
- replaces the eight-paragraph Introduction with a six-move causal story;
- promotes quality–distance isolation to the central empirical identification contribution;
- removes Product manifold as a third active environment while preserving provenance;
- gives Theorem 1 one proportional role and a direct experiment map;
- makes DRPO modify the exact aggregate negative term in Theorem 1;
- removes an unregistered standalone Online RQ and uses registered Hopper/Countdown routes;
- converts Discussion from a disclaimer inventory into a transferable principle;
- adds a guidance review gate before future outline, blueprint, prose, or figure-plan activation.
