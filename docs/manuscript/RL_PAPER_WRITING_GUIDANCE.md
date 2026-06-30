# DRPO RL Paper Writing Guidance v1.0

**Status:** active manuscript quality gate.

**Authority:** this guide governs the presentation of the paper outline, paragraph blueprints, prose, figures, tables, and claim–evidence reviews. `docs/handoff.md` remains the sole authority for scientific conclusions, experiment status, frozen protocols, and execution order. When this guide conflicts with the handoff, the handoff wins.

**Purpose:** prevent technically correct material from being arranged as a defensive report instead of a persuasive research paper. Every manuscript artifact must tell one coherent story, make each claim testable, and connect theory, method, and evidence through the same mathematical object.

---

## 1. Source corpus and extraction policy

This guidance was synthesized from two sources:

1. a curated set of high-impact RL papers spanning theory-led, method-led, offline, online, and sequence-model work;
2. popular open-source paper-writing skills and editorial workflows.

The external skills were reviewed as untrusted read-only references. Their executable instructions and dependencies are **not vendored or run** in this repository. Only independently reviewed writing principles are incorporated here.
Primary paper pages/PDFs and skill repositories were inspected in a temporary research workspace. Copyrighted PDFs and third-party skill code are not committed; the tables below retain stable primary-source links and the specific writing move extracted from each source.

### 1.1 RL paper corpus

| Paper | Type | Writing pattern extracted |
|---|---|---|
| [Trust Region Policy Optimization](https://proceedings.mlr.press/v37/schulman15.html) (ICML 2015) | theory-led method | begin with one optimization failure; derive the surrogate/constraint that the algorithm directly implements; use theory to justify the design variable rather than decorate the paper |
| [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) (2017) | method-led | reduce the value proposition to a memorable operational trade-off; keep the algorithmic story simple enough to restate in one sentence |
| [Soft Actor-Critic](https://proceedings.mlr.press/v80/haarnoja18b.html) (ICML 2018) | theory + method | define an idealized policy-iteration object, derive the practical algorithm from it, and let experiments test the practical system without repeatedly apologizing for the abstraction |
| [Maximum a Posteriori Policy Optimisation](https://arxiv.org/abs/1806.06920) (ICLR 2018) | derivation-led policy optimization | separate policy improvement from projection, then make each algorithmic step correspond to one part of the derivation |
| [Advantage-Weighted Regression](https://arxiv.org/abs/1910.00177) (2019) | simple off-policy actor learning | present a familiar supervised-learning form as the practical consequence of an RL objective; use simplicity and scale as part of the contribution |
| [Behavior Regularized Offline Reinforcement Learning](https://arxiv.org/abs/1911.11361) (2019/2020) | offline-RL framework | define one family of constraints around a concrete distribution-shift failure, then compare instantiations under a shared evaluation protocol |
| [Conservative Q-Learning](https://proceedings.neurips.cc/paper/2020/hash/0d2b2061826a5df3221116a5085a6052-Abstract.html) (NeurIPS 2020) | theory + offline RL | identify one concrete failure, prove one property that counters it, and make the method optimize the exact term appearing in that property |
| [Pessimism in the Face of Uncertainty](https://arxiv.org/abs/2012.15085) (2020/2021) | theory-led offline RL | state the statistical obstacle first, make the pessimistic principle the theorem-level answer, and keep finite-sample guarantees tied to the algorithmic object |
| [AWAC](https://arxiv.org/abs/2006.09359) (2020/2021) | method + real-world motivation | anchor relevance in a real deployment bottleneck, then explain why existing objectives cannot cross the offline-to-online transition |
| [A Minimalist Approach to Offline RL / TD3+BC](https://proceedings.neurips.cc/paper/2021/hash/a8166da05c5a094f7dc03724b41886e5-Abstract.html) (NeurIPS 2021) | minimalist method | attack unnecessary complexity with a deliberately simple design; let broad, fair experiments provide the defense |
| [Decision Transformer](https://proceedings.neurips.cc/paper/2021/hash/7f489f642a0ddb10272b5c31057f0663-Abstract.html) (NeurIPS 2021) | reframing method | lead with a surprising but simple reframing, maintain that framing consistently through method, experiments, and title |
| [Implicit Q-Learning](https://openreview.net/forum?id=68n2s9ZJWF8) (ICLR 2022) | theory-informed method | ask one sharp question, build every component around it, and avoid expanding the paper into all adjacent offline-RL problems |
| [Direct Preference Optimization](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html) (NeurIPS 2023) | derivation-led method | turn an implicit RL objective into a directly optimized loss; the derivation and implementation share the same variable and equation chain |
| [ReBRAC](https://proceedings.neurips.cc/paper_files/paper/2023/hash/5d2017d07b3b55c0c4e30f95f532b8d0-Abstract-Conference.html) (NeurIPS 2023) | empirical method | establish credibility through scale, ablations, and sensitivity analysis; distinguish core algorithmic changes from implementation choices |
| [Cal-QL](https://arxiv.org/abs/2303.05479) (NeurIPS 2023) | theory-motivated offline-to-online method | identify the exact failure of an otherwise successful principle, introduce the smallest corrective property, and validate the correction at the transition where the failure matters |
| [DreamerV3](https://www.nature.com/articles/s41586-025-08744-2) (Nature 2025 / arXiv lineage) | general method | emphasize a single general recipe and demonstrate breadth without changing the core method across domains |

### 1.2 Open-source writing-skill corpus

| Repository / skill | Useful principle retained |
|---|---|
| [Master-cai/Research-Paper-Writing-Skills](https://github.com/Master-cai/Research-Paper-Writing-Skills) | paragraph flow, claim–evidence alignment, section-specific review, reviewer-mindset self-audit |
| [NousResearch/Hermes research-paper-writing](https://github.com/NousResearch/Hermes-Agent/tree/main/skills/research-paper-writing) | one-sentence contribution; paper as a story rather than an experiment inventory; every experiment serves an explicit claim; citations must be verified |
| [SNL-UCSB/paper-writing-skill](https://github.com/SNL-UCSB/paper-writing-skill) | Brainstorm → Draft 0 → Evaluate → Write → Compress; rhetorical moves and compression are separate stages; the author keeps responsibility for claims and evidence |
| [Orchestra-Research AI-Research-SKILLs / ML Paper Writing](https://github.com/Orchestra-Research/AI-Research-SKILLs) | What–Why–So What discipline; Figure 1 as a story carrier; methodological grouping in Related Work; citation verification |
| [hzwer/WritingAIPaper](https://github.com/hzwer/WritingAIPaper) | identify whether the paper contributes insight, performance, or capability; foreground one or two memorable ideas; treat experimental gains as evidence for the idea rather than the story itself |

### 1.3 Structured close-reading synthesis

The corpus was not used as a list of famous papers. For each paper, the review compared six manuscript moves: opening problem, named missing link, theorem or derivation role, method consequence, experiment architecture, and conclusion. The recurring patterns were:

| Manuscript move | Recurring pattern in strong RL papers | DRPO rule derived from it |
|---|---|---|
| Opening | state one operational or conceptual failure before introducing machinery | open with useful negative feedback becoming destructive under historical reuse |
| Gap | name one unresolved causal or optimization link | separate sample badness from policy-relative distance and identify the useful-to-destructive transition |
| Theory | prove the property that determines the method’s design variable | Theorem 1 identifies the aggregate negative term and the equilibrium transition |
| Method | modify the same object named by theory | DRPO maps `q m_-` to `q_lambda m_{-,lambda}` |
| Evaluation | organize experiments around rival explanations and decisive controls | matched badness/distance isolation, targeted near/far intervention, budget matching, terminal audit |
| Conclusion | leave one transferable principle, not a disclaimer inventory | control far-field negative contribution rather than deleting negative feedback |

The open-source skills were downloaded or inspected in a temporary research workspace. Their strongest additions were process controls: topic-sentence scaffolding, claim–evidence maps, introduction-twice revision, section-specific rhetorical moves, compression as a separate stage, and mandatory reviewer-style audits. No third-party executable skill code is committed or invoked by this project.

### 1.4 Living-corpus rule

This is a living guidance document, not a claim that the corpus is exhaustive. The initial corpus deliberately spans classic theory-led policy optimization, modern offline RL, derivation-led objectives, minimalist empirical methods, and broad generalist agents. Add a paper or skill only when it changes a concrete rule below. Do not expand the bibliography merely to appear comprehensive.

---

## 2. The central doctrine

### 2.1 One paper, one central tension

The paper must be restatable as one causal and methodological arc:

> Negative feedback enables policy improvement beyond positive-only learning, but repeated optimization of far-field negative actions can amplify repulsion until the policy loses a finite equilibrium. DRPO preserves useful negative feedback while suppressing the destructive far-field tail.

Any paragraph, theorem, experiment, or figure that does not advance this arc belongs in an appendix, provenance record, or separate paper.

### 2.2 One-sentence contribution test

Before drafting any section, write the paper contribution in one sentence. The section passes only if its role in that sentence is explicit.

- **What:** a theory of the useful-to-destructive transition in repulsive policy updates and a method that controls the far-field negative term.
- **Why:** quality–distance isolation and targeted interventions identify the source and causal transmission of the instability.
- **So what:** negative feedback need not be removed; it can be retained safely for improvement beyond positive-only learning.

### 2.3 Attack with a precise missing link

Do not attack prior work with broad claims such as “negative gradients were ignored.” Attack the unresolved link:

1. prior work shows that negative updates can help or hurt;
2. prior work controls sign, scale, probability, ratio, staleness, or data quality;
3. the missing explanation is when useful local repulsion becomes destructive far-field repulsion, how this changes equilibrium existence, and which term a method should control.

### 2.4 Evidence is the primary defense

The main text should state what is true, why it matters, and how the evidence rules out alternatives. Do not pre-emptively enumerate everything the paper does not solve.

A limitation belongs in the main text only when omitting it would change the interpretation of a result. Mathematical conditions belong in theorem statements. Experiment status belongs in tables, captions, and provenance. Venue-required limitations belong in the designated limitations section, not throughout the story.

Forbidden defensive drift includes:

- “we do not claim global convergence of every actor–critic method” when no such claim is needed;
- “the mechanism is not excluded in broader settings” instead of positively stating the observed mechanism;
- repeated reminders that a controlled environment is not the whole world;
- lists of unrelated RL problems that DRPO does not solve.

---

## 3. DRPO-specific story architecture

### 3.1 Required causal chain

Every active artifact must preserve this order:

1. **Negative feedback has value:** it suppresses bad modes and can move a policy beyond the positive-only target.
2. **Historical reuse creates persistence:** negative actions remain in optimization as the learner moves.
3. **Quality–distance isolation identifies the source:** at matched reward/advantage, semantics, contexts, and sample count, distance alone increases the score and negative-gradient magnitude.
4. **Targeted interventions identify transmission:** retaining far-field negatives preserves failure; removing or capping them prevents drift/boundary events/task collapse in the controlled environment.
5. **Theorem 1 explains the phase transition:** positive-only → stable extrapolation → boundary approach → loss of finite equilibrium.
6. **DRPO controls the same term:** it attenuates the far-field component of the aggregate negative moment appearing in Theorem 1.
7. **External tasks close the loop:** Hopper and Countdown test whether the signature and method benefit survive realistic critic, replay, and shared-parameter conditions.

### 3.2 Non-negotiable quality–distance isolation statement

The paper must state, prominently and repeatedly enough to be unmistakable, that sample badness and policy-relative distance are experimentally separated.

The controlled source-isolation protocol must hold fixed or match:

- state/context;
- action semantics or direction;
- reward and advantage severity;
- sample count and base coefficient;
- actor architecture and optimizer state at comparison;

while varying only policy-relative radius/distance or rarity.

The conclusion must be phrased precisely:

> Far-field distance is an independent gradient amplifier; the observed far/near gradient gap cannot be reduced to far samples having worse advantages.

Do **not** strengthen this into “distance is the only cause.” Advantage severity and direction remain separate factors.

### 3.3 Environment-role discipline

The active paper has two primary controlled environments:

- **C-U1:** continuous Gaussian mechanism, causal transmission, stable extrapolation, and held-out-context generalization;
- **D-U1:** categorical shared-representation mechanism, persistent suppression, boundary behavior, and unseen-state generalization.

Hopper and Countdown provide external validity.

“Product manifold” is a historical development construction for source isolation, not a third primary paper environment. In the active manuscript, refer to the current experiment as the **quality–distance factorized source-isolation protocol in C-U1 E1**. Preserve Product-manifold results only as provenance or appendix development evidence.

### 3.4 Failure-event discipline

Always report separately:

1. task-performance collapse;
2. covariance/support/probability boundary events;
3. NaN/Inf numerical failure.

Never let “collapse” silently switch meanings within a paragraph, figure, or table.

---

## 4. Theory–method–experiment bridge

### 4.1 Same object rule

A theory section earns main-text space only when the method changes the same mathematical object that the theorem identifies.

For DRPO:

- Theorem 1 identifies the aggregate negative contribution `q m_-` as the term that shifts the equilibrium beyond the positive-only solution and can move it to the feasibility boundary.
- DRPO replaces it with a distance/surprisal-weighted contribution `q_lambda m_{-,lambda}`.
- Proposition 2 shows the weighted far-field gradient vanishes under finite-order score growth.
- Experiments measure the corresponding near/far contribution, equilibrium/terminal state, and recovery.

If any of these links is absent, the manuscript fails the bridge gate.

### 4.2 Theorem role rule

Theorem 1 has one job:

> characterize when positive–negative competition yields the positive-only equilibrium, a stable extrapolated equilibrium, a boundary target, or no finite equilibrium.

Proof machinery that does not change this interpretation goes to the appendix. The theorem must be followed immediately by testable predictions and their experiment IDs.

### 4.3 No experiment-setting leakage into theory

Fixed advantage is an experimental control used to isolate score geometry from critic feedback. It is not a premise of the far-field mechanism and must not be presented as the scope of the theory.

The theorem may define fixed positive/negative masses and target distributions because those quantities define its mathematical objective. Do not narrate this as “freezing the empirical update field over an analysis window.”

---

## 5. Experiment design and presentation

### 5.1 Claim-first experiment template

Every experiment subsection must include, in this order:

1. **Claim:** the exact proposition being tested.
2. **Alternative explanation:** the confound or rival interpretation.
3. **Control/intervention:** what is held fixed and what is changed.
4. **Metrics:** which observable would distinguish the explanations.
5. **Verdict:** what the result establishes and what remains unresolved.
6. **Status:** formal, long-run, finite-step, pilot, or not run.

A compact authoring table should be maintained:

| Claim | Rival explanation | Intervention/control | Metric | Acceptance criterion | Experiment ID | Status |
|---|---|---|---|---|---|---|

### 5.2 Evidence order

The final paper should use a **reality anchor → controlled explanation → reality closure** structure:

1. environment and evidence-role table;
2. a compact external mechanism anchor from Hopper and/or Countdown, once formal terminal-audited results exist;
3. quality–distance source isolation;
4. causal near/far or common/rare intervention;
5. phase-transition validation of Theorem 1;
6. controlled method comparison and budget matching;
7. external task-performance closure.

The external anchor establishes that the phenomenon is not created by the controlled environment. The controlled environments then answer the questions that external tasks cannot isolate. The final external comparison shows that the resulting control improves a real task. Until Hopper/Countdown results are formal, preserve their section and evidence role as `TBD`; never substitute a pilot or planning result merely to keep the order.

### 5.3 No new environment by default

A narrative gap is not automatically an experiment gap. Prefer a new diagnostic or registered analysis in an existing environment over another toy environment.

The current framework requires no additional primary controlled environment. Remaining evidence work is completion of already registered gates and external validation, not proliferation of simulators.

### 5.4 Fair-comparison rule

Method comparisons involving Global, Linear, Hard, Exp, SBRC, or Hybrid require:

- paired seeds;
- matched or explicitly reported raw negative-gradient budgets;
- identical initialization and data;
- best and terminal results reported separately;
- terminal-state audit;
- no prespecified winner in prose or captions.

### 5.5 Terminal audit rule

Any claim about equilibrium, divergence, collapse, or ranking requires a terminal audit. A finite-step curve is not a steady-state result. Record slope/residual, 2×-horizon consistency where registered, clamp/floor contact, and all three failure-event types.

---

## 6. Section contracts

### 6.1 Title and abstract

- Title names the central phenomenon and method, not every domain.
- Abstract follows: problem → missing mechanism → theory → method → evidence → implication.
- Every result in the abstract must already be formal and traceable.
- Do not use the abstract for caveats, literature defense, or implementation detail.

### 6.2 Introduction

A strong Introduction performs six moves:

1. negative feedback is valuable;
2. historical reuse turns it into persistent repulsion;
3. existing controls miss the useful-to-destructive transition;
4. the theory explains the transition;
5. DRPO controls the destabilizing term;
6. the evidence chain and contributions close the story.

Keep the Introduction attack focused. Do not explain theorem assumptions, full environment construction, or all related work there.

### 6.3 Related Work

Group by methodological line, not paper chronology. Each paragraph should end with the exact unresolved link the paper addresses. Avoid “paper A did X, paper B did Y” catalogs.

### 6.4 Theory

State the physical interpretation before the formal statement. After the theorem, give testable predictions. Put long algebra, exceptional cases, and routine local-stability derivations in the appendix unless they alter the core conclusion.

### 6.5 Method

Start from the destabilizing term identified by theory, then show the modification. Every design choice must answer “why this term, why this shape, why not deleting all negatives?” Ablations belong in Experiments.

### 6.6 Experiments

Lead with evidence roles and research questions. Environment descriptions must explain why each environment can answer its assigned question. Never present a simulator as external validity.

### 6.7 Discussion and conclusion

Conclude with transferable insight, not a list of disclaimers. State the principle readers should remember:

> control far-field negative contribution, not negative feedback as a whole.

---

## 7. Figures, tables, and captions

### 7.1 Figure 1 rule

Figure 1 must communicate the whole paper without requiring the theory section:

- positive-only stops at observed positive behavior;
- controlled repulsion creates stable extrapolation;
- far-field reuse creates excessive negative contribution and boundary crossing;
- DRPO suppresses the far-field tail and restores a stable regime.

### 7.2 One visual, one claim

Each main figure has one primary claim. Captions must be standalone and state:

- setup/control;
- visible pattern;
- conclusion supported;
- status or uncertainty when relevant.

### 7.3 Tables as contracts

Main tables should expose fairness and terminal status, not just peak reward. Include budget, best/terminal split, and separate failure-event columns when applicable.

---

## 8. Language and rhetorical discipline

### 8.1 Preferred verbs

Use verbs that identify evidence strength:

- derive / prove for mathematics;
- isolate for controlled source attribution;
- causally identify for targeted interventions;
- observe / validate for external signatures;
- improve for paired task metrics;
- suggest only when evidence is incomplete.

### 8.2 Avoid self-sabotaging phrases

Do not write:

- “our theory does not cover real actor–critic systems”;
- “we make no global guarantee” unless the paper has invited that interpretation;
- “the mechanism is not ruled out”;
- “toy environment” in the main text;
- “distance is the only factor”;
- “OOD generalization” for current C-U1 results.

### 8.3 Compression rule

After scientific review passes, run a separate compression pass:

- remove repeated caveats;
- replace noun chains with direct verbs;
- delete sentences that merely announce structure;
- keep one definition per term;
- move proof details and provenance to appendices.

Compression must not remove experiment status, controls, or evidence boundaries.

---

## 9. Mandatory guidance review gate

Every new or modified outline, blueprint, prose section, main figure plan, or result narrative requires a guidance review record under `docs/manuscript/reviews/` before it becomes active.

### 9.1 Hard gates

A review is `PASS` only if all gates below pass:

| Gate | Question |
|---|---|
| G01 Thesis | Can the artifact's contribution be stated in one sentence consistent with the handoff? |
| G02 Central tension | Does it advance the useful-negative-feedback versus destructive-far-field-repulsion tension? |
| G03 No defensive drift | Does it avoid unnecessary disclaimers and self-imposed adjacent problems? |
| G04 Quality–distance isolation | Is the badness-versus-distance confound explicitly controlled wherever the source claim appears? |
| G05 Theory role | Does Theorem 1 have a single clear job and proportional main-text space? |
| G06 Theory–method bridge | Does the method modify the same aggregate negative term identified by theory? |
| G07 Claim–experiment map | Does each major claim have a registered experiment, metric, acceptance criterion, and status? |
| G08 Environment roles | Are C-U1, D-U1, Hopper, Countdown, and historical Product-manifold evidence assigned correctly? |
| G09 Failure separation | Are task collapse, boundary events, and NaN/Inf separated? |
| G10 Fairness and terminal audit | Are budget matching, paired comparisons, best/terminal reporting, and terminal audit preserved? |
| G11 Related-work positioning | Is novelty expressed as a missing link rather than denial of prior work? |
| G12 Citation and result integrity | Are citations verifiable and are no pilot/planning results promoted? |
| G13 Figure-1 story | Does the visual plan carry the full paper arc? |
| G14 Cascade integrity | Are outline, blueprint, prose, and review hashes aligned? |

### 9.2 Severity

- **Blocker:** violates the handoff, changes a frozen protocol, overclaims result status, breaks the central story, or fails a hard gate.
- **Major:** theory/method/evidence link is unclear, a key confound is not controlled, or the section reads as defense rather than contribution.
- **Minor:** local wording, order, compression, or caption issue that does not change interpretation.

Any Blocker or Major finding prevents activation.

### 9.3 Required review record

The review record must contain:

- artifact path and SHA-256;
- base commit;
- reviewer date;
- one-sentence thesis;
- gate-by-gate pass/fail with evidence;
- Blocker/Major/Minor findings;
- unresolved experiment dependencies;
- final verdict.

---

## 10. Pre-activation checklist

Before changing `docs/manuscript/hierarchy.yaml`:

1. read `docs/handoff.md` §0 and relevant experiment entries;
2. verify the current GitHub `main` commit;
3. run the manuscript cascade validator;
4. run the guidance review and record artifact hashes;
5. confirm Product manifold is not introduced as a third primary environment;
6. confirm quality–distance isolation is prominent in Introduction, experiments, and Figure 1 plan;
7. confirm Theorem 1, DRPO, and experiments share the same aggregate negative term;
8. confirm no unfinished external experiment is written as a result;
9. confirm all three failure types are separate;
10. only then activate the new outline/blueprint pair.
