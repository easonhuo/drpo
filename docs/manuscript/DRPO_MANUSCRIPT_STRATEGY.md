# DRPO Manuscript Strategy v1.0

**Status:** active project-specific manuscript strategy.

**Relationship to other files:**

- `docs/handoff.md` is the authority for scientific conclusions, experiment state, frozen protocols, and execution order.
- `docs/manuscript/RL_PAPER_WRITING_GUIDANCE.md` supplies stable writing principles and review gates.
- This file defines the **current DRPO paper story** and may evolve when the scientific interpretation or evidence architecture changes.
- The versioned outline, blueprint, prose, figures, and tables derive from this strategy.

---

## 1. The paper’s lineage: why it remains DRPO

This manuscript is not a newly named algorithm or an unrelated follow-up. It is the theoretical reconstruction and generalization of the original DRPO paper:

> **Breaking the Curse of Repulsion: Optimistic Distributionally Robust Policy Optimization for Off-Policy Generative Recommendation** (arXiv:2602.10430).

The original paper established the research identity around three ideas:

1. negative-advantage policy updates act as repulsion rather than ordinary imitation;
2. repeated off-policy use of low-quality historical behavior can destabilize the actor;
3. distributional selection/control can suppress destructive repulsion, with Optimistic DRO and hard filtering providing the original formulation.

The revised paper preserves that research object and method family while repairing and extending the scientific account:

- recommendation-specific framing becomes general off-policy policy optimization;
- sample badness is separated from policy-relative distance/rarity;
- the old Gaussian variance direction and expected-Fisher argument are corrected;
- negative feedback is shown to have a useful-to-destructive transition rather than being uniformly harmful;
- continuous Gaussian and categorical policies receive a common equilibrium/boundary account;
- hard filtering becomes an endpoint in a broader selective-control family, while the practical DRPO form preserves useful local negatives and attenuates the far-field tail;
- C-U1/D-U1 provide controlled identification, and Hopper/Countdown provide external validity.

Therefore **DRPO is retained by design**. The new manuscript is the mature form of the same “curse of repulsion” program, not a branding choice made after the fact. Global negative scaling/SNA2C and other tapers are comparison or control families inside this research program; they do not replace the DRPO paper identity.

---

## 2. Central tension and one-sentence contribution

### 2.1 Central tension

Negative feedback is a policy-improvement resource: it suppresses known bad modes and, when balanced against positive attraction, can shift the policy beyond the Positive-only target. Yet historical negative actions remain in offline data, replay, stale rollouts, and asynchronous trajectories after the learner has moved away from them. Their policy-relative distance or rarity increases while their negative update persists, turning useful local repulsion into destructive far-field influence.

### 2.2 One-sentence contribution

> We identify a useful-to-destructive transition in negative policy updates: controlled repulsion creates stable improvement beyond Positive-only learning, whereas policy-relative far-field amplification can remove the finite equilibrium; DRPO preserves useful negative feedback by selectively attenuating the far-field component of the aggregate negative term.

### 2.3 Full causal arc

\[
\text{useful negative feedback}
\rightarrow
\text{stable extrapolation}
\rightarrow
\text{historical reuse and far-field movement}
\rightarrow
\text{excessive aggregate negative contribution}
\rightarrow
\text{boundary / loss of finite equilibrium}
\rightarrow
\text{DRPO recovery}.
\]

Every main-text section and figure must advance this arc.

---

## 3. The decisive identification claim: separate badness from distance

The central rival explanation is:

> Far-field negative gradients may be larger only because far samples are worse and carry larger negative advantages.

The paper must answer this directly. The source-isolation protocol matches, as applicable:

- state/context;
- reward and negative-advantage magnitude;
- action quality coordinate or semantic role;
- sample count;
- base coefficient;
- training stage and policy parameters;

and changes only policy-relative distance or rarity.

The claim is:

> **Policy-relative distance/rarity is an independent amplifier of negative influence; the far/near gap cannot be reduced to far samples having worse advantages.**

Do not claim distance is the only factor. Advantage severity, count, direction coherence, and network Jacobian remain independent contributors.

### Environment placement

- The formal paper-facing continuous protocol is **C-U1 E1 quality–distance factorized source isolation**.
- The historical Product-manifold construction is provenance for the idea and belongs in the appendix/history, not the main environment table and not as a third primary controlled environment.
- The categorical analogue is performed in D-U1 with quality/semantics/count/base-coefficient matching across common/rare or low/high-surprisal actions.

---

## 4. Theory role

### 4.1 Per-sample mechanism

For a negative sample,

\[
\|g_i^-\|
=
A_i^-
\|\nabla_\theta\log\pi_\theta(a_i\mid s_i)\|.
\]

This factorization separates badness (`A_i^-`) from policy-score geometry. Gaussian standardized distance can increase score magnitude; categorical logit scores are bounded but repeated negative updates can persistently suppress probability.

### 4.2 Theorem 1 has one job

**Theorem 1: Stable Extrapolation and Loss of Finite Equilibrium** characterizes the aggregate competition between positive attraction and negative repulsion:

1. Positive-only has a finite target at the positive moment;
2. moderate negative contribution shifts the finite equilibrium beyond that target;
3. stronger or more outward negative contribution moves the signed target toward the feasible boundary;
4. beyond the feasible region—or when the restoring mass disappears—a finite equilibrium is lost.

The theorem explains why the same negative signal can first help and later destabilize. It is not presented as a generic convergence theorem, and the paper does not volunteer unrelated global-guarantee disclaimers.

### 4.3 Testable phase sequence

\[
\text{Positive-only limit}
\rightarrow
\text{stable extrapolation}
\rightarrow
\text{boundary approach}
\rightarrow
\text{persistent drift / no finite equilibrium}.
\]

E4 and E6 are the primary controlled tests of this sequence. Task-performance collapse, support/variance/probability boundary events, and NaN/Inf remain separate outcomes.

---

## 5. Direct theory–method bridge

For an exponential-family policy, write the uncontrolled aggregate negative moment as

\[
q\mathbf m_-
=
\mathbb E[A^-\mathbf T(a)].
\]

Theorem 1 identifies this term as both:

- the source of useful equilibrium displacement beyond Positive-only; and
- the term that can push the signed target to or beyond the feasible boundary.

DRPO applies policy-relative distance/surprisal weighting:

\[
w^-_\lambda(s,a)=e^{-\lambda r_\theta(s,a)},
\]

and replaces the negative moment by

\[
q_\lambda\mathbf m_{-,\lambda}
=
\mathbb E[A^-e^{-\lambda r_\theta(s,a)}\mathbf T(a)].
\]

This is the central method sentence:

> DRPO modifies the same aggregate negative term identified by Theorem 1, retaining substantial local repulsion while attenuating the far-field component that drives boundary crossing.

Under finite-order score growth, exponential weighting makes the weighted far-field gradient vanish. The exponential envelope is justified as gradient-tail control; no assumed exponential decay of sample utility is required.

### DRPO and the original Optimistic DRO formulation

The original DRPO used Optimistic DRO to derive hard selection of a high-quality subdistribution. In the revised framework:

- hard filtering remains a distributionally robust endpoint that removes selected negative influence;
- exponential distance/surprisal weighting is the smooth selective-control realization used to preserve useful local negatives;
- the paper must present this as an evolution within the same DRPO family, not as an unrelated heuristic appended to an old name.

The method section must include a concise formulation that connects distributional reweighting, the hard endpoint, and the smooth far-field envelope. Detailed historical derivations may move to the appendix, but the lineage may not disappear.

---

## 6. Evidence architecture

Use four research questions.

### RQ1 — Does repulsive instability appear in realistic policy learning?

Hopper/D4RL and Countdown provide the external occurrence anchor once terminal-audited formal results exist. Report distance/surprisal-binned negative influence, positive/negative imbalance, temporal ordering, and separate failure events. Until formal results exist, values and verdicts remain `TBD`.

### RQ2 — Why does it occur?

Combine two controlled steps:

1. **source isolation:** match badness and change only distance/rarity;
2. **causal transmission:** near/far and common/rare targeted interventions, including equal-budget controls where registered.

C-U1 answers continuous source and causal questions. D-U1 answers categorical source and support-boundary questions. Historical Product-manifold evidence appears only as provenance/appendix support.

### RQ3 — When does useful repulsion become destructive, and can DRPO control the transition?

Map Positive-only, stable extrapolation, boundary approach, and persistent drift to Theorem 1. Compare registered methods under paired seeds and matched/explicit raw negative-gradient budgets. Measure not only reward and terminal state, but the theorem-level object:

\[
\widehat{\mathbf M}_t^-
=
\sum_i A_i^- w_i\mathbf T(a_i)
\]

or a family-specific empirical proxy. Report its unweighted/weighted norm, direction, distance bins, and relationship to equilibrium displacement or terminal drift.

Do not predeclare Exp, Linear, Global, Hard, SBRC, or Hybrid as the winner. Rankings require the registered terminal audit.

### RQ4 — Does the resulting control improve external tasks?

Return to D4RL/Hopper and Countdown with the same initialization/data/selection rules, paired seeds, best and terminal reporting, and mechanism diagnostics. This is the reality closure.

---

## 7. Environment responsibilities

| Environment | Responsibility | Not its responsibility |
|---|---|---|
| C-U1 | controlled continuous source isolation, near/far causality, stable-extrapolation transition, method control | external validity; categorical support claims |
| D-U1 | controlled categorical rarity isolation, common/rare causality, probability/support boundary, semantic generalization | Gaussian gradient-amplitude claims |
| Hopper/D4RL | external continuous RL occurrence and task effect with learned critic | exact badness–distance causal isolation |
| Countdown | external shared-parameter sequence/categorical occurrence and task effect | replacing D-U1’s controlled causal identification |
| Historical Product manifold | provenance and appendix support for the original factorization idea | a primary paper environment |

Current C-U1 test reporting uses **held-out-context generalization** or **unseen-state generalization**, not OOD generalization.

---

## 8. Main-text story and visuals

### Introduction moves

1. negative feedback is a policy-improvement resource;
2. historical reuse turns it into persistent repulsion;
3. badness and distance must be separated;
4. Repulsive Dynamics explains stable extrapolation and equilibrium loss;
5. DRPO controls the destabilizing far-field term while preserving lineage to original DRPO;
6. external → controlled → external evidence closes the story.

### Figure 1

Figure 1 should show:

- Positive-only stopping at the positive target;
- controlled repulsion shifting to a better stable equilibrium;
- historical reuse moving negatives into the far field and enlarging aggregate negative influence;
- boundary/persistent-drift regime;
- DRPO attenuating the far-field tail and restoring a finite stable regime.

Quality–distance matching must be visible or explicitly called out, because it closes the paper’s central logical loophole.

---

## 9. Novelty boundary and language

Defensible novelty:

> Prior work studies negative-feedback value, negative-gradient domination, stale or low-probability updates, and data/support control. This paper separates policy remoteness from sample badness, causally traces far-field influence into instability, characterizes the useful-to-destructive equilibrium transition, and controls the same aggregate negative term with DRPO.

Do not claim:

- first discovery that negative gradients can be harmful;
- first observation that low-probability actions/tokens receive large or persistent updates;
- distance is the only cause of failure;
- all real-task collapse has one cause;
- current C-U1 is OOD generalization;
- any taper family is universally superior before terminal evidence.

These are claim-calibration rules, not invitations to add defensive paragraphs.

---

## 10. Activation rule

A new outline or blueprint becomes active only when:

1. it is consistent with this strategy and the scientific record;
2. it passes all stable guidance gates;
3. the outline/blueprint block IDs and parent hashes align;
4. live experiment statuses are not embedded as structural content—status remains in the handoff/registry/review dependency list;
5. superseded manuscript artifacts remain preserved.
