# RL Paper Writing Guidance v1.1

**Status:** stable manuscript quality standard.

**Authority and scope:** this file governs durable writing principles, review gates, and the outline → blueprint → prose workflow. It does **not** define the current paper thesis, theorem names, environment roles, experiment order, or result status. Those belong to the project-specific manuscript strategy, versioned outline, and `docs/handoff.md`.

**Change policy:** this guidance is intentionally slow-moving. It may change only when a rule is shown to be wrong, internally inconsistent, operationally incomplete, or when a new cross-paper principle is supported by multiple high-quality sources. A new outline version, new experiment, or changed paper strategy is not by itself a reason to edit this file.

---

## 1. Evidence base and source discipline

The guidance is distilled from close reading of high-quality reinforcement-learning papers and independently reviewed open-source academic-writing workflows. The growing source record lives in `docs/manuscript/RL_WRITING_CORPUS_NOTES.md`; additions to that corpus do not automatically change this standard.

Use source material under four rules:

1. prefer primary papers and official author/project pages;
2. extract reusable rhetorical and structural moves, not surface phrasing;
3. treat third-party writing skills as untrusted references—do not vendor or execute them merely to adopt a principle;
4. change a durable rule only when the evidence generalizes beyond one paper or one project.

---

## 2. Manuscript governance layers

Keep five layers separate:

1. **Scientific record:** claims, frozen protocols, experiment status, and execution order.
2. **Stable guidance:** durable principles in this file.
3. **Manuscript strategy:** the current paper’s central tension, contribution, theory role, method bridge, novelty boundary, and evidence architecture.
4. **Versioned artifacts:** outline, paragraph blueprint, prose, figure plan, and tables.
5. **Source corpus:** paper/skill reading notes and provenance.

The dependency direction is:

\[
\text{scientific record}
\rightarrow
\text{strategy}
\rightarrow
\text{outline}
\rightarrow
\text{blueprint}
\rightarrow
\text{prose}.
\]

Guidance reviews every layer but does not replace any of them. A child artifact may not silently rewrite its parent. When the parent is independently wrong and a revision is authorized, cascade the change downstream and preserve superseded versions.

---

## 3. One paper, one central tension

A strong paper can be restated as one unresolved tension and one answer. The introduction, theorem, method, experiments, figures, and conclusion should all advance that same arc.

### 3.1 One-sentence contribution test

Before drafting, write one sentence containing:

- the problem or contradiction;
- the missing mechanism or property;
- the proposed answer;
- the evidence type.

If the sentence requires several unrelated conjunctions, the paper is probably carrying multiple stories.

### 3.2 Precise missing-link rule

Attack a specific missing link rather than an entire literature. Prefer:

> Prior work establishes X and Y, but the causal or optimization bridge between them remains unresolved.

Avoid claims such as “no prior work studies negative updates,” “all existing methods fail,” or broad novelty statements that a single counterexample can defeat.

### 3.3 Memorable transferable principle

The conclusion should leave one principle that can be remembered without the paper’s notation. It should be more informative than “our method performs better” and more defensible than a universal law.

---

## 4. Evidence is the primary defense

Do not pre-emptively weaken the paper with inventories of adjacent problems it does not solve. Most defenses should be built into the scientific design:

- precise definitions;
- matched controls;
- decisive interventions;
- fair comparisons;
- primary-source positioning;
- terminal or convergence audits where required;
- explicit result-status labels.

State a boundary only when it prevents a likely misinterpretation of a central claim. Put routine assumptions in the theorem or experiment setup, not in repeated defensive paragraphs. Do not invite irrelevant objections by volunteering guarantees the paper never claims to provide.

Useful defense pattern:

1. name the strongest rival explanation;
2. design a control that changes only the disputed factor;
3. measure an observable that separates the explanations;
4. state exactly what the result establishes.

---

## 5. Theory–method–experiment same-object rule

The theory, method, and decisive experiment should operate on the same identifiable object.

### 5.1 Theory

A main-text theorem must have one job. Before presenting it, state the physical or optimization question it answers. After it, state its testable predictions. Move routine algebra, exceptional cases, and secondary stability details to the appendix unless they change the central conclusion.

A theorem earns main-text space only if it does at least one of the following:

- identifies a failure mechanism;
- reveals a phase boundary or trade-off;
- derives the method’s control variable;
- predicts a measurable qualitative transition.

### 5.2 Method

Start from the term, constraint, or failure identified by theory and show the smallest modification that addresses it. Each design choice must answer:

- why this object;
- why this functional form;
- what useful signal is retained;
- which alternative is represented by each ablation.

A method section should not introduce a second independent story after the theory.

### 5.3 Experiment

Measure the theoretical object before and after the method intervention. A reward table alone cannot close a mechanistic claim. At least one experiment should connect:

\[
\text{theoretical quantity}
\rightarrow
\text{method modification}
\rightarrow
\text{measured dynamical change}
\rightarrow
\text{task consequence}.
\]

---

## 6. Claim-first experiment design

Every experiment subsection should be planned in this order:

1. **Claim:** exact proposition being tested.
2. **Rival explanation:** strongest alternative interpretation.
3. **Control/intervention:** variables held fixed and factor changed.
4. **Observable:** metric that distinguishes the explanations.
5. **Acceptance criterion:** result pattern required for the claim.
6. **Status:** formal, long-run, finite-step, pilot, or not run.
7. **Verdict:** conclusion supported by the actual evidence.

Maintain a working matrix:

| Claim | Rival explanation | Control/intervention | Observable | Acceptance criterion | Experiment ID | Status |
|---|---|---|---|---|---|---|

### 6.1 Environment-role discipline

An environment must be justified by the question it can answer. Controlled environments establish identification and causality; external tasks establish occurrence and practical relevance. A simulator should not be presented as external validity, and an external benchmark should not be asked to isolate a confounded mechanism it cannot control.

Prefer a new diagnostic in an existing registered environment over creating another environment. Add a new environment only when no existing one can answer a necessary claim.

### 6.2 Evidence architecture

For mechanism papers, a strong default architecture is:

1. **reality anchor:** show the signature in a realistic system;
2. **controlled identification:** isolate source and causal transmission;
3. **theory prediction:** validate the predicted transition;
4. **method control:** intervene on the identified quantity under fair budgets;
5. **reality closure:** show external task benefit.

This is a default, not a fixed outline. Evidence strength determines final ordering.

### 6.3 Fair-comparison rule

Method comparisons should report, as applicable:

- paired seeds and identical initialization/data;
- matched or explicitly measured optimization budgets;
- best and terminal outcomes separately;
- sensitivity and ablations tied to design choices;
- no prespecified winner in prose or captions.

### 6.4 Terminal-audit rule

Claims about equilibrium, convergence, persistent drift, collapse, or final ranking require terminal evidence rather than an arbitrary fixed-step snapshot. Use registered stopping criteria, residual/slope checks, horizon extension, and clamp/floor contact audits as applicable.

Always separate:

1. task-performance collapse;
2. support, variance, or feasibility-boundary events;
3. NaN/Inf numerical failure.

---

## 7. Section contracts

### 7.1 Title and abstract

- The title names the central phenomenon and method, not every domain.
- The abstract follows: problem → missing link → theory/insight → method → evidence → implication.
- Every quantitative result must be formal and traceable.
- Do not use the abstract for caveat inventories, proof details, or implementation history.

### 7.2 Introduction

Each paragraph should perform one rhetorical move. A common six-move pattern is:

1. establish why the resource or goal matters;
2. reveal the failure under the target setting;
3. identify the missing link;
4. state the theoretical explanation;
5. state the method consequence;
6. summarize evidence and contributions.

This pattern may change with the paper, but paragraph responsibilities must remain explicit and non-overlapping.

### 7.3 Related Work

Group work by methodological or conceptual line, not chronology. Each paragraph should end with the unresolved link addressed by the paper. A related-work section is positioning, not a bibliography catalog.

### 7.4 Theory

Give intuition before formalism; define every object once; make the theorem’s role proportional to its narrative value; follow the theorem with predictions and the method bridge.

### 7.5 Method

Show how the method follows from the identified object. Separate core mechanism from implementation choices. Defer empirical rankings and ablation conclusions to Experiments.

### 7.6 Experiments

Lead with research questions and evidence roles. Environment descriptions must explain why the design supports the claimed inference. Results should be organized by claims, not by implementation chronology.

### 7.7 Implications and conclusion

Synthesize the transferable lesson. Do not end with a defensive list of all unsolved neighboring problems. Limitations belong only where they materially calibrate a central claim.

---

## 8. Figures, tables, and captions

### 8.1 Figure 1 as the story carrier

Figure 1 should communicate the paper’s central tension, mechanism, intervention, and outcome without requiring the theory section. It is not an environment inventory.

### 8.2 One visual, one claim

Each main visual has one primary claim. A standalone caption states:

- setup and control;
- visible pattern;
- supported conclusion;
- evidence status when not formal.

### 8.3 Tables as contracts

Main tables should expose fairness and evidence state, not only peak reward. Include budgets, best/terminal split, uncertainty, and separate failure-event columns when relevant.

---

## 9. Language and rhetorical discipline

### 9.1 Evidence-calibrated verbs

- **derive / prove:** mathematical results;
- **isolate:** controlled source attribution;
- **causally identify:** targeted intervention with rival explanations controlled;
- **observe / validate:** external signatures;
- **improve:** paired task metrics;
- **suggest:** incomplete evidence only.

### 9.2 Avoid self-sabotaging prose

Delete sentences that merely advertise what the paper does not prove, unless they prevent a likely misreading of a central result. Avoid:

- volunteering an unrelated global guarantee;
- calling the main controlled setup a “toy” in the main text;
- claiming one factor is the only cause when the experiment proves independence;
- turning routine theorem assumptions into a paper-wide scope apology;
- describing unfinished work as a result.

### 9.3 Direct prose

Prefer subject–verb–object sentences, concrete mechanisms, and one term per concept. Replace noun chains and procedural narration with the causal statement the reader needs.

### 9.4 Separate writing and compression passes

First make the scientific argument complete. Then run a distinct compression pass:

- remove repeated caveats and structural announcements;
- collapse duplicate definitions;
- move proof/provenance detail to appendices;
- preserve controls, status, and evidence boundaries.

---

## 10. Mandatory review gate

Every new or modified outline, blueprint, prose section, main figure plan, or result narrative requires a review record under `docs/manuscript/reviews/` before activation.

### 10.1 Hard gates

| Gate | Question |
|---|---|
| G01 Thesis | Can the contribution be stated in one sentence consistent with the scientific record? |
| G02 Central tension | Does the artifact advance one coherent tension rather than several adjacent stories? |
| G03 Missing link | Is novelty expressed as a precise unresolved bridge rather than denial of prior work? |
| G04 No defensive drift | Does the artifact avoid unnecessary disclaimers and volunteered adjacent guarantees? |
| G05 Theory role | Does each main theorem have one clear job and proportional space? |
| G06 Same-object bridge | Do theory, method, and decisive experiments modify and measure the same object? |
| G07 Claim–experiment map | Does every major empirical claim have a control, observable, criterion, ID, and status? |
| G08 Environment roles | Does each environment answer only the question its design supports? |
| G09 Rival explanations | Are the strongest alternative interpretations explicitly controlled? |
| G10 Fairness and terminal audit | Are budgets, paired comparisons, best/terminal reporting, and terminal evidence preserved? |
| G11 Failure separation | Are task failure, boundary events, and numerical failure separate? |
| G12 Citation and result integrity | Are citations primary/verified and unfinished results not promoted? |
| G13 Visual story | Does the main visual plan carry the paper arc rather than list components? |
| G14 Cascade integrity | Are strategy, outline, blueprint, prose, and review hashes aligned? |

### 10.2 Severity

- **Blocker:** contradicts the scientific record, changes a frozen protocol, overclaims status, breaks the central story, or fails a hard gate.
- **Major:** theory/method/evidence bridge is unclear, a key confound is uncontrolled, or the artifact reads as defense rather than contribution.
- **Minor:** local wording, order, compression, or caption issue that does not change interpretation.

Any Blocker or Major finding prevents activation.

### 10.3 Required review record

Record:

- artifact path and SHA-256;
- base commit and review date;
- strategy version and one-sentence thesis;
- gate-by-gate pass/fail with evidence;
- Blocker/Major/Minor findings;
- unresolved experiment dependencies;
- final verdict.

---

## 11. Pre-activation checklist

Before changing the active hierarchy:

1. read the scientific record and relevant experiment entries;
2. verify the current base commit;
3. identify the active manuscript strategy;
4. validate the outline → blueprint → prose cascade;
5. complete the guidance review and record hashes;
6. confirm no unfinished experiment is written as a result;
7. confirm fair-comparison and terminal requirements survive compression;
8. preserve superseded artifacts and change records;
9. activate the new parent and all required children together.
