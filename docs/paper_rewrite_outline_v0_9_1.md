# DRPO 论文重写 v0.9.1：strategy-separated canonical outline

**状态：** active canonical outline after application；已获用户授权，派生自 v0.9，并由稳定 Guidance 与 DRPO Manuscript Strategy 分层审查。

**上位文件：**

- scientific authority: `docs/handoff.md`；
- stable writing standard: `docs/manuscript/RL_PAPER_WRITING_GUIDANCE.md`；
- project-specific story: `docs/manuscript/DRPO_MANUSCRIPT_STRATEGY.md`。

**作用：** 规定当前论文的 section、paragraph、theory–method–experiment mapping、图表职责和篇幅预算。它不保存实时实验状态、下一步 seeds 或执行队列；这些继续由 handoff、registry 与 review dependency 记录管理。

**基线：** GitHub `main` commit `445a2e6d129994b2dd48f7c87050206dc705b838`。

**Introduction 施工图：** `docs/paper_rewrite_intro_blueprint_v0_5.md`。

**历史保留：** v0.9/v0.4 及更早 artifact 不删除，作为 manuscript provenance。

---

# 0. Paper identity and structural contract

## 0.1 Why this paper remains DRPO

This manuscript is the theoretical reconstruction and generalization of the original paper:

> **Breaking the Curse of Repulsion: Optimistic Distributionally Robust Policy Optimization for Off-Policy Generative Recommendation** (arXiv:2602.10430).

The original DRPO introduced repulsive policy updates, off-policy collapse, and an Optimistic-DRO/hard-filtering response in generative recommendation. The revised paper preserves that research identity while:

- generalizing from recommendation to off-policy policy optimization;
- correcting the Gaussian variance and expected-Fisher arguments;
- separating sample badness from policy-relative distance/rarity;
- explaining why negative feedback is first useful and later destructive;
- unifying continuous and categorical boundary dynamics;
- extending hard filtering into a selective smooth far-field control family;
- adding controlled causal identification and external validation.

DRPO is therefore a lineage commitment, not a new name attached to an unrelated method. The main text should state this continuity positively and briefly; it should not turn the name into a defensive issue.

## 0.2 One central tension

> Negative feedback can shift a policy beyond the Positive-only target, but historical negative actions become increasingly remote from the learner and can exert excessive far-field influence. DRPO preserves useful local repulsion while attenuating the far-field component that drives the policy toward a feasibility boundary and the loss of a finite equilibrium.

## 0.3 Full paper arc

\[
\text{negative feedback as a resource}
\rightarrow
\text{stable extrapolation}
\rightarrow
\text{historical reuse and far-field movement}
\rightarrow
\text{excessive aggregate negative contribution}
\rightarrow
\text{boundary / no finite equilibrium}
\rightarrow
\text{DRPO recovery}.
\]

## 0.4 The decisive logical control: separate badness from distance

The main rival explanation is:

> Far-field gradients are larger only because far-field samples are worse and have larger negative advantages.

The paper-facing isolation must match state/context, reward, negative-advantage severity, action quality coordinate or semantic role, sample count, base coefficient, and policy stage, while changing only learner-relative distance or rarity. The paper’s claim is:

> **Policy-relative distance/rarity is an independent amplifier of negative influence; the far/near gap cannot be reduced to far samples having worse advantages.**

Do not upgrade independence to exclusivity. Advantage severity, direction coherence, count, and network Jacobian remain independent factors.

## 0.5 Environment discipline

The two primary controlled environments remain **C-U1** and **D-U1**. Hopper/D4RL and Countdown are external-validity environments. The historical Product-manifold construction is appendix/provenance for the source-isolation idea; it is not a main-paper environment and does not appear in the main environment table.

Fixed advantage belongs only to controlled identification as a confound-removal device. It is not a premise of the far-field mechanism or a paper-wide scope statement.

---

# 1. Title, abstract, and positioning

## 1.1 Recommended title

**Breaking the Curse of Repulsion: Distributionally Robust Policy Optimization for Off-Policy Learning**

## 1.2 One-sentence contribution

> We identify a useful-to-destructive transition in negative policy updates: controlled repulsion creates stable improvement beyond Positive-only learning, whereas policy-relative far-field amplification can eliminate the finite equilibrium; DRPO selectively attenuates the far-field component of the same aggregate negative term while preserving useful local feedback.

## 1.3 Abstract move sequence

1. **Resource and problem:** policy improvement needs negative feedback, yet historical negative actions can remain in optimization after the learner has moved away.
2. **Missing link:** sample badness and policy remoteness are normally confounded, so the origin and causal role of large negative updates remain unresolved.
3. **Theory:** Repulsive Dynamics characterizes Positive-only, stable extrapolation, boundary approach, and loss of finite equilibrium.
4. **Method:** DRPO reweights the theorem’s aggregate negative term, preserving local repulsion and suppressing its far-field tail; the smooth form extends the original Optimistic-DRO/hard-filtering lineage.
5. **Identification:** matched-badness source isolation and targeted near/far or common/rare interventions.
6. **Evidence:** controlled continuous/categorical phase tests and terminal-audited external validation.
7. **Implication:** control policy-relative far-field influence rather than deleting negative feedback as a whole.

No fixed-advantage scope paragraph, unrelated guarantee, or unfinished quantitative result enters the abstract.

---

# 2. Introduction — approximately 2.0 columns

The Introduction uses exactly six paragraph contracts.

<!-- MANUSCRIPT:BEGIN INTRO-P01 -->
## [INTRO-P01] Negative Feedback as a Policy-Improvement Resource

Establish the positive role of negative feedback. Positive updates reinforce observed successful behavior; negative updates suppress known bad modes and, when balanced against positive attraction, can shift the policy equilibrium beyond the Positive-only target. Positive-only is a stable and important reference, but its target is determined by observed positive behavior. End with the paper’s central question: how can policy optimization preserve the improvement supplied by negative feedback without allowing that repulsion to become dynamically excessive?
<!-- MANUSCRIPT:END INTRO-P01 -->

<!-- MANUSCRIPT:BEGIN INTRO-P02 -->
## [INTRO-P02] Historical Reuse Turns Local Feedback into Persistent Repulsion

Explain how the failure emerges. A negative action may initially provide relevant local boundary information, but offline logs, replay, stale actors, and asynchronous trajectories can continue to reuse it after the learner has moved away. Its policy-relative distance or rarity grows while its negative update persists. Gaussian scores can increase with standardized distance; categorical direct-logit scores are bounded but repeated updates can persistently suppress probability. Conclude with the useful-local-feedback to destructive-far-field-repulsion transition, without adding scope disclaimers.
<!-- MANUSCRIPT:END INTRO-P02 -->

<!-- MANUSCRIPT:BEGIN INTRO-P03 -->
## [INTRO-P03] The Missing Link: Separating Badness from Distance

State the decisive identification gap. In realistic data, low reward, large negative advantage, rarity, and distance are correlated; observing larger far-field gradients therefore does not establish that distance independently matters. Introduce the matched control that holds context, quality/semantics, reward, advantage severity, sample count, and base coefficient fixed while changing only policy-relative distance or rarity. Preview the source-isolation and targeted-intervention evidence. Make clear that distance is an independent amplifier, not the only factor.
<!-- MANUSCRIPT:END INTRO-P03 -->

<!-- MANUSCRIPT:BEGIN INTRO-P04 -->
## [INTRO-P04] Repulsive Dynamics Explains Stable Extrapolation and Equilibrium Loss

Present the theoretical contribution. Positive-only optimization targets the positive moment; moderate aggregate negative repulsion moves a finite stable equilibrium beyond that target; stronger or more outward negative contribution moves the signed target toward the feasible boundary and can eliminate a finite equilibrium. Gaussian and categorical policies share this aggregate phase structure while expressing different score and boundary dynamics. Keep proof details, step-size conditions, and secondary cases out of the Introduction.
<!-- MANUSCRIPT:END INTRO-P04 -->

<!-- MANUSCRIPT:BEGIN INTRO-P05 -->
## [INTRO-P05] DRPO Controls the Destabilizing Far-Field Term

Connect theory, lineage, and method. Theorem 1 identifies the aggregate negative moment as the term that both enables extrapolation and, when excessive, drives boundary crossing. DRPO replaces its far-field component with exponential policy-distance/surprisal weighting, retaining stronger local feedback while making weighted far-field gradients vanish under finite-order score growth. Briefly state that this smooth selective control extends the original DRPO Optimistic-DRO/hard-filtering formulation rather than introducing an unrelated algorithm.
<!-- MANUSCRIPT:END INTRO-P05 -->

<!-- MANUSCRIPT:BEGIN INTRO-P06 -->
## [INTRO-P06] Evidence Chain and Contributions

Summarize four research questions: external occurrence; controlled source and causal identification; theorem phase transition plus DRPO control; external task closure. Name C-U1 and D-U1 as the two controlled environments, and Hopper/D4RL and Countdown as external validation. Emphasize matched badness–distance isolation, targeted interventions, aggregate-negative-term measurement, fair budget comparisons, and terminal audits. End with four contributions: Repulsive Dynamics theory, quality–distance/rarity causal identification, the lineage-preserving DRPO method, and multi-level external–controlled–external validation. Do not list Product manifold as an environment or predeclare a method ranking.
<!-- MANUSCRIPT:END INTRO-P06 -->

---

# 3. Related Work — approximately 0.75 column

Organize three conceptual lines.

## 3.1 Learning from negative or suboptimal behavior

Positive-only filtering, advantage-weighted learning, failure learning, and negative reinforcement establish that negative feedback can be useful. The unresolved question is the dynamical transition from useful repulsion to destructive far-field influence.

## 3.2 Off-policy, stale, and low-probability updates

Clipping, importance correction, stale-policy methods, and low-probability-token controls regulate scale, support, or mismatch. The missing bridge is repeated learner-relative movement, aggregate negative influence, and finite-equilibrium loss.

## 3.3 Robust offline policy learning

CQL/IQL-style approaches, behavior regularization, TD3+BC/ReBRAC, AWR-like actor fitting, and data filtering control value extrapolation, support, or data quality. DRPO instead targets the far-field component of signed actor updates while preserving useful local negatives.

Novelty language should describe the complete bridge rather than deny prior negative-update work:

> useful negative feedback → badness–distance isolation → far-field causal transmission → equilibrium transition → selective DRPO control.

---

# 4. Problem Setup — approximately 0.75 column

## 4.1 Signed actor update

\[
\mathbf F(\theta)
=
\mathbb E_{\nu}
[\widehat A(s,a)\nabla_\theta\log\pi_\theta(a\mid s)].
\]

Define

\[
A^+=\max(\widehat A,0),\qquad
A^-=\max(-\widehat A,0),
\]

\[
\mathbf F(\theta)=\mathbf F^+(\theta)-\mathbf F^-(\theta).
\]

For a negative sample,

\[
\|g_i^-\|
=
A_i^-\,\|\nabla_\theta\log\pi_\theta(a_i\mid s_i)\|.
\]

Interpretation:

- `A_i^-` is sample badness/severity;
- the score norm is learner-relative policy geometry;
- count and direction coherence determine aggregation.

This factorization motivates the isolation experiment.

## 4.2 Policy-relative far field

Define one common remoteness variable `r_θ(s,a)` with family-specific realization:

- Gaussian: standardized/Mahalanobis distance or calibrated negative log-density;
- categorical: action surprisal `-log π_θ(a|s)`;
- sequence policy: registered normalized token/completion NLL.

Far field is dynamic relative to the current learner, not a permanent sample label.

---

# 5. Repulsive Dynamics — approximately 2.5 columns

## 5.1 Per-sample far-field amplification

### Gaussian

\[
\nabla_\mu\log\pi(a\mid s)=\Sigma^{-1}(a-\mu).
\]

Mean-score magnitude grows with standardized distance. With learnable covariance, the sign of variance change depends on standardized location; the corrected dangerous chain is mean repulsion plus support contraction in the far field, not universal mean-and-variance expansion.

### Categorical

\[
\nabla_z\log\pi(a\mid s)=e_a-\pi(\cdot\mid s).
\]

The direct-logit score is bounded, but repeated negative updates continue to reduce the action’s log-odds and can push probability toward the simplex boundary. Do not describe this as unbounded Euclidean logit-gradient explosion.

### Shared principle

Continuous and categorical policies differ in the amplification law but share persistent repulsion toward a policy feasibility boundary.

## 5.2 Aggregate positive–negative competition

For a regular minimal exponential-family policy,

\[
\pi_\eta(a)=h(a)\exp\{\eta^\top T(a)-\psi(\eta)\}.
\]

Let `p,q` be aggregate positive/negative update mass and `m_+,m_-` their sufficient-statistic moments. Then

\[
\mathbf F(\eta)
=
p\mathbf m_+-q\mathbf m_--(p-q)\nabla\psi(\eta).
\]

These variables define the theorem’s signed objective. Do not narrate them as an “analysis window,” a frozen-advantage paper assumption, or a limitation on realistic actor–critic training.

## 5.3 Theorem 1: Stable Extrapolation and Loss of Finite Equilibrium

### Positive-only limit

For `q=0`,

\[
\nabla\psi(\eta^\star)=\mathbf m_+.
\]

The policy targets the positive moment.

### Stable extrapolation

When `0<q<p` and

\[
\mathbf m^\star
=
\frac{p\mathbf m_+-q\mathbf m_-}{p-q}
\]

lies in the interior of the mean-parameter space, there is a unique finite equilibrium satisfying

\[
\nabla\psi(\eta^\star)=\mathbf m^\star.
\]

Moreover,

\[
\mathbf m^\star-\mathbf m_+
=
\frac{q}{p-q}(\mathbf m_+-\mathbf m_-),
\]

so negative repulsion can shift the equilibrium beyond Positive-only.

### Boundary approach and loss of finite equilibrium

Increasing the magnitude or outward location of `q m_-` moves `m*` toward the feasible boundary. A boundary target requires a degenerate distribution or diverging natural parameter; a target outside the feasible region has no finite equilibrium. When `p=q` and the positive/negative moments do not cancel, the restoring term disappears and persistent drift remains.

### Local stability

At an interior equilibrium,

\[
J_F(\eta^\star)=-(p-q)\nabla^2\psi(\eta^\star).
\]

State the local result compactly; place proof, spectral step-size details, exact cancellation cases, and second-moment derivations in the appendix.

## 5.4 Testable predictions and experiment mapping

| Theory regime | Controlled intervention | Required observable |
|---|---|---|
| Positive-only target | remove negative updates | finite stable platform near positive target |
| stable extrapolation | add controlled negative influence | platform beyond Positive-only with low terminal residual |
| boundary approach | increase negative strength/remoteness | support/variance/probability approaches registered boundary |
| no finite equilibrium | continue increasing influence | persistent parameter drift or non-vanishing field residual |
| DRPO recovery | selectively attenuate far field | finite stable regime with retained local-negative benefit |

E4 and E6 are the primary phase tests. RQ2 source/causal experiments explain why far-field influence changes the aggregate term. RQ3 measures the aggregate term itself.

## 5.5 Family corollaries

- **Gaussian:** finite mean/covariance, mean escape, covariance/support boundary, and distance-amplified score.
- **Categorical:** finite logits, probability approaching `0/1`, bounded but persistent direct-logit suppression.

Use one compact comparison table. Detailed derivations go to the appendix.

---

# 6. Distributionally Robust Policy Optimization — approximately 1.75 columns

## 6.1 Lineage from original DRPO

Briefly recap the original Optimistic-DRO formulation: selecting a high-quality subdistribution yields a hard-filtering endpoint that removes destructive negative influence. The revised formulation retains this distributional-control view but uses a smooth policy-relative envelope to preserve useful local negative feedback. This establishes continuity without re-running the entire old paper in the main text.

## 6.2 DRPO update

\[
w_i^-=\exp(-\lambda r_i),
\]

\[
\mathbf F_{\mathrm{DRPO}}
=
\mathbb E
\left[
A^+\nabla\log\pi
-
A^-e^{-\lambda r}\nabla\log\pi
\right].
\]

The registered family-specific remoteness measure must match the experiment protocol.

## 6.3 Direct bridge to Theorem 1

Uncontrolled negative moment:

\[
q\mathbf m_-
=
\mathbb E[A^-\mathbf T(a)].
\]

DRPO-controlled moment:

\[
q_\lambda\mathbf m_{-,\lambda}
=
\mathbb E[A^-e^{-\lambda r_\theta(s,a)}\mathbf T(a)].
\]

Controlled equilibrium target:

\[
\mathbf m_\lambda^\star
=
\frac{p\mathbf m_+-q_\lambda\mathbf m_{-,\lambda}}
{p-q_\lambda}.
\]

Central interpretation:

> Theorem 1 identifies the aggregate negative moment that first enables extrapolation and later drives boundary crossing; DRPO selectively attenuates the far-field component of that same moment.

## 6.4 Proposition 2: Vanishing weighted far-field gradient

If

\[
\|\nabla\log\pi\|\le C(1+r)^k,
\]

then

\[
e^{-\lambda r}\|\nabla\log\pi\|
\le Ce^{-\lambda r}(1+r)^k
\rightarrow 0.
\]

The exponential form is a gradient-tail envelope, not an assumed utility-decay law.

## 6.5 Method family and ablations

- Uncontrolled: `w^-=1`;
- Positive-only: `w^-=0`;
- Global: `w^-=α`;
- Linear: `max(0,1-λr)`;
- Hard: threshold endpoint inherited from the original distributional-selection view;
- DRPO-Exp: `exp(-λr)`;
- other registered selective controls only when their protocol and budgets are frozen.

Do not predict a winner. The scientific comparison is which control preserves useful local influence while producing a terminally stable regime under fair budgets.

---

# 7. Experiments — approximately 7.5 columns

## 7.1 Environments and evidence roles

| Environment | Design features | Paper responsibility |
|---|---|---|
| Hopper/D4RL | public offline continuous control, learned critic, long-horizon dynamics | external continuous occurrence and task effect |
| Countdown | shared Transformer parameters, verifier feedback, stale/replay trajectories | external sequence/categorical occurrence and task effect |
| C-U1 | 6D numerical context, 2D continuous action, nonlinear Gaussian actor, known optimum, near/far interventions | continuous source isolation, causal transmission, phase transition, controlled method test |
| D-U1 | 6D context, finite unordered semantic actions, shared categorical actor, common/rare interventions | categorical rarity isolation, support boundary, semantic held-out-context generalization |

### Required environment description

For C-U1 and D-U1, state:

- state/action construction and ground-truth target;
- how train and held-out contexts are sampled;
- how positive and negative actions are constructed;
- which variables are matched in source isolation;
- how near/far or common/rare status is defined relative to the current policy;
- why the intervention supports the assigned inference.

Use **held-out-context generalization** or **unseen-state generalization** for C-U1, not OOD generalization.

Historical Product-manifold provenance is described only in the appendix.

## 7.2 RQ1 — Does repulsive instability appear in realistic policy learning?

### Hopper/D4RL

Report:

- negative influence by current-policy distance bin;
- far/near negative ratio and negative/positive ratio;
- temporal ordering of gradient anomaly, policy drift, boundary event, and return degradation;
- best and terminal state;
- task collapse, boundary event, and NaN/Inf separately.

### Countdown

Report:

- negative token/completion influence by surprisal bin;
- rare/common ratio and stale-replay probability change;
- greedy success, pass@k, valid rate, best and terminal state;
- support/entropy boundary and numerical failure separately.

### Verdict

This RQ establishes external occurrence and relevance. It does not replace controlled source or causal identification. Only formal terminal-audited results enter the main text.

## 7.3 RQ2 — Why does it occur? Source isolation and causal transmission

### RQ2a: Is distance/rarity an independent source?

**Claim:** with badness and base weight matched, farther/rarer negatives exert greater policy influence.

**Rival explanation:** the far/rare group is worse, more numerous, semantically different, or assigned a larger coefficient.

**Continuous control (C-U1 E1):** same context and quality construction; match reward, `|A|`, count, and base coefficient; vary only policy-relative radius/distance. The final full-parameter attribution requires a same-state/same-ray radial probe or a registered Jacobian-gain decomposition so action direction is not changed with radius.

**Categorical control (D-U1):** match context, badness, semantics, count, and coefficient across common/rare probability conditions.

**Metrics:** output score, all-parameter score/gradient, gradient direction coherence, far/near or rare/common ratio, matching diagnostics.

**Verdict language:** distance/rarity is an independent amplifier; do not claim it is the only cause.

### RQ2b: Does far-field influence causally transmit into instability?

C-U1 interventions:

- Uncontrolled;
- Near-zero;
- Far-zero;
- Far-cap;
- equal-budget Global control;
- registered budget-transfer control.

D-U1 analogues:

- common-negative suppression;
- rare-negative suppression;
- equal-budget global control;
- registered rare-to-common transfer.

Report early-time influence, onset order, terminal reward, parameter drift, support/variance/probability boundary, and NaN/Inf separately. The decisive pattern is whether treating far/rare negatives changes the outcome while treating matched near/common negatives does not.

## 7.4 RQ3 — When does useful repulsion become destructive, and can DRPO control the transition?

### RQ3a: Theorem 1 phase map

Scan the registered negative-influence strength/control variable and identify:

1. Positive-only target;
2. stable extrapolation beyond Positive-only;
3. boundary approach;
4. persistent drift/no finite equilibrium.

Report:

- terminal policy/equilibrium location;
- performance relative to Positive-only;
- field residual or terminal slope;
- covariance/support/probability boundary;
- registered horizon-extension audit;
- three failure types separately.

### RQ3b: Measure the theorem-level aggregate term

For each method and phase, report an empirical proxy for

\[
\widehat{\mathbf M}_t^-
=
\sum_i A_i^-w_i\mathbf T(a_i),
\]

including:

- raw and weighted norm;
- near/far or common/rare decomposition;
- direction relative to positive attraction and observed displacement;
- relationship to equilibrium shift, boundary approach, or terminal drift.

This metric is mandatory: theory identifies the term, DRPO modifies it, and experiments must measure it.

### RQ3c: Controlled method comparison

Compare registered Uncontrolled, Positive-only, Global, Linear, Hard, DRPO-Exp, and any activated selective controls under:

- paired seeds and identical data/initialization;
- matched or explicitly measured raw negative-gradient budgets;
- near-field retention and far-field suppression diagnostics;
- best and terminal performance;
- terminal audit.

The main criterion is not only peak reward, but whether a method exceeds Positive-only while remaining in a finite stable regime and preserving useful local-negative influence.

## 7.5 RQ4 — Does DRPO improve external tasks?

### D4RL/Hopper family

Use the registered locomotion matrix and report normalized return, uncertainty, best checkpoint, terminal checkpoint, task-collapse rate, and mechanism diagnostics. Preserve dataset-specific results rather than hiding instability in a global average.

### Countdown

Use the registered common initialization, rollout/replay bank, seeds, verifier, and selection protocol. Report greedy success, pass@k, valid rate, best/terminal split, rare-negative diagnostics, support event, and numerical failure.

### Closure

The final evidence chain is:

\[
\text{external occurrence}
\rightarrow
\text{controlled explanation}
\rightarrow
\text{phase and method control}
\rightarrow
\text{external improvement}.
\]

---

# 8. Implications and Conclusion — approximately 0.75 column

## 8.1 Transferable principle

Negative feedback is not the enemy. The failure arises when historical negatives remain influential after becoming policy-relative far-field samples. The correct control target is their excessive far-field contribution, not their sign alone.

## 8.2 Continuous–categorical synthesis

Gaussian policies express repulsion through distance-amplified scores and mean/support dynamics; categorical policies express it through bounded but persistent probability suppression. Both can move an interior equilibrium toward a feasibility boundary.

## 8.3 Lineage conclusion

DRPO began as an Optimistic-DRO/hard-filtering response to repulsive collapse in generative recommendation. The revised paper provides the broader dynamics, corrected theory, selective smooth control, and external validation needed to establish DRPO as a general off-policy policy-optimization framework.

## 8.4 Three-sentence close

1. Negative feedback can move a policy beyond the Positive-only target.
2. Repeated far-field repulsion can eliminate the finite equilibrium that makes this improvement stable.
3. DRPO preserves useful negative feedback while suppressing the far-field tail responsible for instability.

---

# 9. Main-paper visual and table plan

| Item | Single primary claim |
|---|---|
| Figure 1 | complete arc: Positive-only → stable extrapolation → far-field boundary/drift → DRPO recovery, with badness–distance matching visible |
| Figure 2 | formal external distance/surprisal-binned repulsive signature |
| Figure 3 | C-U1/D-U1 matched-badness source isolation and targeted causal intervention |
| Figure 4 | Theorem 1 continuous/categorical phase map plus aggregate-negative-term trajectory |
| Figure 5 | matched-budget DRPO control and external task closure, split if density requires |
| Table 1 | four environment roles and controls |
| Table 2 | controlled results with task/boundary/numerical failures separated |
| Table 3 | D4RL external performance, best and terminal |
| Table 4 | Countdown external performance, best and terminal |

Every caption states setup/control, visible pattern, supported claim, and evidence status.

---

# 10. Column budget

| Section | Columns |
|---|---:|
| Introduction | 2.00 |
| Related Work | 0.75 |
| Problem Setup | 0.75 |
| Repulsive Dynamics | 2.50 |
| DRPO | 1.75 |
| Experiments | 7.50 |
| Implications and Conclusion | 0.75 |
| **Total** | **16.00** |

---

# 11. Guidance and strategy review summary

This version makes six structural refinements over v0.9:

1. stable writing Guidance is separated from the evolving DRPO strategy;
2. the original DRPO lineage and reason for retaining the name are explicit;
3. live experiment status and execution details are removed from the structural outline;
4. Product manifold is removed from the main environment table;
5. six RQs are consolidated into four paper-level questions;
6. the empirical aggregate negative term is elevated to a mandatory theory–method–experiment observable.

All unresolved experiment dependencies remain in `docs/handoff.md`, `experiments/registry.yaml`, and the associated manuscript review record rather than being embedded as outline structure.
