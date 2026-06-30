# DRPO Manuscript Strategy v1.1

**Status:** active project-specific strategy.

This file defines the current paper story. It may evolve only through an explicitly authorized manuscript version. Scientific conclusions and experiment status remain governed by `docs/handoff.md` and `experiments/registry.yaml`; stable writing principles remain governed by `RL_PAPER_WRITING_GUIDANCE.md`.

---

## 1. Manuscript identity: one paper under rewrite

This repository is rewriting the same DRPO manuscript. It is not producing an “old DRPO paper” followed by a new sequel, and the main text must never use that narrative.

The current manuscript must be self-contained. Valid theory, method, and evidence from earlier drafts are incorporated directly into the current paper; invalid statements are corrected in place or recorded in the appendix correction ledger. Research history is provenance, not a two-paper story.

The name DRPO is retained because this is the DRPO manuscript itself. The paper does not need a defensive explanation for retaining its own name.

### Hard prohibition

Do not write:

- “the original paper did X, while the revised paper does Y” as the current scientific narrative;
- “we extend the old algorithm with a new method”;
- “to avoid rewriting the old paper”;
- a mathematical hard-to-smooth relationship merely to justify the name.

---

## 2. Central tension

> Negative feedback is a policy-improvement resource: controlled repulsion can move a policy beyond the Positive-only target. Historical reuse can make the same negative actions increasingly remote from the current learner, causing excessive far-field influence, boundary approach, and loss of finite equilibrium. DRPO preserves useful local repulsion while attenuating the destructive far-field component.

---

## 3. Precise missing link

Existing methods regulate negative sign, global scale, clipping, support, staleness, probability, or data quality. They do not jointly explain:

1. why negative feedback can improve a policy;
2. why the same feedback becomes destructive after learner-relative movement;
3. whether far-field influence is independent of sample badness;
4. how the aggregate transition changes equilibrium existence;
5. which term a method should control.

The decisive rival explanation is that far samples merely have worse reward or larger negative advantage. The paper answers it by matching context, quality or semantic role, reward, advantage severity, count, coefficient, and policy stage while changing only distance or rarity.

Distance/rarity is an independent amplifier, not the only cause of negative influence.

---

## 4. Full paper arc

\[
\text{negative feedback as a resource}
\rightarrow
\text{historical reuse and far-field movement}
\rightarrow
\text{existing-control and identification gap}
\rightarrow
\text{stable extrapolation to equilibrium loss}
\rightarrow
\text{DRPO control of the same aggregate term}
\rightarrow
\text{external--controlled--external closure}.
\]

---

## 5. Theory role

### Per-sample mechanism

\[
\|g_i^-\|=A_i^-\|\nabla_\theta\log\pi_\theta(a_i\mid s_i)\|.
\]

This separates sample severity from learner-relative score geometry.

### Theorem 1

Theorem 1 has one job: characterize Positive-only, stable extrapolation, boundary approach, and loss of finite equilibrium under aggregate positive--negative competition.

It must be followed by testable predictions and an explicit method bridge.

### Policy-family distinction

- Gaussian: distance-amplified mean score and corrected mean--support dynamics;
- categorical: bounded direct-logit score but persistent probability suppression;
- shared: movement toward a feasibility boundary under excessive aggregate repulsion.

---

## 6. Method role

DRPO is presented as the current paper's single self-contained method.

The method controls the aggregate negative term identified by Theorem 1:

\[
q\mathbf m_-
\rightarrow
q_\lambda\mathbf m_{-,\lambda}
=
\mathbb E[A^-e^{-\lambda r_\theta}T(a)].
\]

The exponential envelope is justified by far-field gradient-tail control. It does not assume exponential utility decay.

Quality-based selection and policy-remoteness control are distinct axes. The paper may compare hard quality filtering, hard distance thresholds, global scaling, and smooth distance tapers, but it must not assert equivalence or limit relations without derivation.

---

## 7. Evidence architecture

Use four research questions.

### RQ1 — External occurrence

Hopper/D4RL and Countdown test whether remote negative influence becomes disproportionately large in realistic training. Results remain TBD until terminal-audited formal closure.

### RQ2 — Source and causal transmission

C-U1 and D-U1 match badness while changing distance/rarity, then apply near/far or common/rare interventions and equal-budget controls.

### RQ3 — Phase transition and DRPO control

E4/E6 map the theorem regimes. Controlled method comparisons measure performance, terminal state, local and remote retention, and a validated empirical proxy for the aggregate negative term.

### RQ4 — External task closure

D4RL and Countdown test task benefit using common initialization, data, seeds, selection, and terminal reporting.

---

## 8. Environment responsibilities

| Environment | Responsibility | Not its responsibility |
|---|---|---|
| C-U1 | controlled continuous source isolation, near/far causality, stable-extrapolation phase, method control | external validity; categorical support claims |
| D-U1 | controlled categorical rarity isolation, common/rare causality, probability boundary, semantic generalization | Gaussian unbounded-score claims |
| Hopper/D4RL | external continuous occurrence and task effect with learned critic | exact badness--distance isolation |
| Countdown | external shared-parameter sequence occurrence and task effect | replacing D-U1 causal identification |
| Historical Product-manifold work | provenance and appendix support for source-factorization development | a primary main-paper environment |

C-U1 uses held-out-context or unseen-state generalization, never OOD generalization without a separately registered distribution shift.

---

## 9. Introduction contracts

1. negative feedback as a resource;
2. historical reuse turns local feedback into persistent repulsion;
3. existing controls and the missing badness--distance identification link;
4. Repulsive Dynamics explains stable extrapolation and equilibrium loss;
5. DRPO controls the same far-field aggregate term;
6. external--controlled--external evidence and contributions.

P03 must contain both prior-method positioning and the matched isolation. Neither may replace the other.

---

## 10. Activation rule

A new artifact becomes active only when:

- it derives from the explicitly approved parent version;
- every substantive delta is listed and authorized;
- no same-paper/rewrite block is framed as a sequel;
- Guidance gates pass;
- stable node IDs and parent hashes align;
- live experiment status stays in handoff/registry rather than structural prose;
- the Overleaf project compiles and the PDF is inspected.
