# RL Paper Writing Guidance v1.2

**Status:** stable manuscript quality standard.

**Authority and scope:** this file contains durable writing and review principles. New corpus entries do not automatically change this standard. It does not define the current DRPO thesis, theorem names, environment responsibilities, experiment status, paragraph order, or result values. Those belong to `docs/handoff.md`, `experiments/registry.yaml`, `docs/manuscript/DRPO_MANUSCRIPT_STRATEGY.md`, and versioned manuscript artifacts.

**Change policy:** this file is deliberately slow-moving. It may change only when a rule is wrong, internally inconsistent, operationally incomplete, or supported as a cross-paper principle by multiple high-quality sources. A new outline version, a changed experiment, or a new paper result is not by itself a reason to edit Guidance.

---

## 1. Authority, authorization, and version identity

### G01. Scientific authority wins

Scientific claims, experiment status, terminology, frozen variables, and execution order come from `docs/handoff.md` and `experiments/registry.yaml`. Writing files may reorganize those facts but may not upgrade, weaken, rename, or invent them.

### G02. An approved artifact is a frozen content baseline

When the user approves a specific outline, blueprint, or prose version, that exact content becomes the baseline for the named version. A later review may propose changes, but it may not silently regenerate a different artifact under the same version number.

The required flow is:

\[
\text{approved artifact}
\rightarrow
\text{exact repository version}
\rightarrow
\text{new proposals under a new version}.
\]

### G03. Review permission is not rewrite permission

A request to review, audit, or apply Guidance authorizes diagnosis and proposed deltas. It does not authorize wholesale regeneration. Every substantive change must be either:

1. explicitly requested;
2. explicitly approved after being proposed; or
3. a mechanical downstream consequence of an approved upstream change.

### G04. Preserve manuscript identity

Before writing, establish whether the task is a revision of the same manuscript, a journal extension, or a genuinely new follow-up. Never convert one into another through rhetorical framing. A revision must remain a self-contained current paper; it must not be written as “the old paper” plus “our new sequel” unless that is the actual publication relationship.

### G05. Version names must reflect content boundaries

If content changes after a user-approved version, create a new version. Do not reuse an earlier version label for a newly generated artifact. Every version change must have a human-readable delta and a machine-verifiable parent.

---

## 2. One paper, one central tension

### G06. One-sentence contribution test

The paper must be restatable as one sentence containing:

- the object of study;
- the unresolved tension or failure;
- the new explanation or property;
- the method consequence;
- the evidence type.

Every main-text paragraph, theorem, method component, figure, and experiment must advance that sentence. Material that does not belong moves to the appendix, provenance record, or another paper.

### G07. Attack a precise missing link

Do not claim that an entire topic was ignored. Strong papers identify a missing causal, optimization, or representational link after acknowledging what prior work already established.

A good gap statement has four parts:

1. what existing methods control;
2. what they do not explain;
3. the rival explanation that remains possible;
4. the decisive control or theorem that resolves it.

### G08. Leave one transferable principle

The conclusion should leave a principle that readers can reuse, not an inventory of limitations or a restatement of every experiment.

---

## 3. Theory, method, and experiment must share an object

### G09. Theory has one job per theorem

A theorem must answer one named question that changes the paper's logic. The main statement contains the qualitative result; proof details, secondary cases, and routine step-size algebra move to the appendix unless they are essential to understand the contribution.

After every main theorem, provide:

1. a plain-language interpretation;
2. qualitative regimes or cases;
3. testable predictions;
4. the method variable affected by the result.

### G10. The method modifies the theorem's object

The method section must form an unbroken equation chain:

\[
\text{failure term}
\rightarrow
\text{theorem-level object}
\rightarrow
\text{method modification}
\rightarrow
\text{implemented update}
\rightarrow
\text{ablation controls}.
\]

If the practical algorithm changes a different quantity from the one named by theory, the bridge is incomplete.

### G11. Experiments measure the same object

At least one experiment must measure an empirical proxy for the theorem-level object before and after the method intervention. Performance alone cannot validate a mechanism claim.

### G12. Mathematical continuity may not be invented for narrative convenience

A shared research name or history does not prove that two objectives are mathematical limits, relaxations, or closed-form descendants of one another. Such relations require an explicit derivation. Research continuity and mathematical equivalence are separate claims.

---

## 4. Evidence is the primary defense

### G13. Use evidence, not anticipatory apology

The main text should state what the paper establishes. Do not fill the Introduction or Discussion with unsolicited statements about unrelated guarantees the paper never attempted to prove.

A legitimate boundary belongs in the main text only when it prevents a likely misreading of the central claim. Otherwise place it in the theorem assumptions, experiment responsibility table, appendix, or reviewer-objection map.

### G14. Design against the strongest rival explanation

For each major claim, maintain an internal objection map:

| Claim | Strongest rival explanation | Decisive control | Observable | Evidence status |
|---|---|---|---|---|

The paper presents the decisive comparison, not the entire private objection list.

### G15. Claim status and execution status remain separate

Analytically proven, long-run validated, finite-step validated, pilot, not run, and rejected/superseded are scientific statuses. Registered, running, raw-complete, terminal-audited, packaged, delivered, and applied are execution states. Neither may be silently promoted.

### G16. Negative, inconclusive, and failed results are evidence

Do not remove a result because it weakens a ranking. Report when a metric is undefined, a method has no stable candidate, a terminal ranking reverses, or a run fails. Explain the scientific consequence rather than hiding the event.

---

## 5. Claim-first experiment architecture

### G17. Every experiment has one primary claim

Before execution or writing, specify:

- claim;
- rival explanation;
- manipulated variable;
- matched variables;
- methods and controls;
- metrics;
- seeds and uncertainty;
- terminal criterion;
- allowed conclusion;
- prohibited overclaim.

### G18. Controlled and external environments have different duties

Controlled environments isolate source, direction, causality, or ground truth. External environments establish relevance, task effect, and robustness under realistic confounds. Neither substitutes for the other.

### G19. Fair comparison is part of the scientific claim

Where relevant, use paired seeds, matched update budgets, common initialization, common data, common checkpoint selection, and common terminal audits. Do not assume a proposed method or taper is superior before the registered comparison.

### G20. Terminal audit is mandatory for dynamic claims

Claims about convergence, persistent drift, collapse, or method ranking require terminal evidence. A fixed number of steps is not convergence. Report best and terminal checkpoints when selection can hide degradation.

### G21. Distinguish failure types

Task-performance collapse, support or variance-boundary events, and NaN/Inf numerical failure are different outcomes and must be reported separately.

---

## 6. Section contracts

### G22. Abstract

The abstract follows a compact sequence:

1. resource or practical need;
2. failure or missing link;
3. theory or diagnosis;
4. method consequence;
5. evidence architecture;
6. transferable implication.

No unfinished result, historical narration, or paper-wide disclaimer enters the abstract.

### G23. Introduction

Each paragraph performs one rhetorical move. A common six-move structure is:

1. establish the useful resource or opportunity;
2. expose the mechanism that turns it into a problem;
3. position existing controls and state the missing identification link;
4. present the theoretical answer;
5. present the method consequence;
6. preview evidence and contributions.

Paragraph count follows the story, not a template; the stable rule is one move per paragraph and an explicit transition.

### G24. Related Work

Organize by conceptual or methodological lines, not by paper chronology. Each line ends with a precise relationship to the current paper. Credit prior findings before stating the remaining bridge.

### G25. Theory

Use the order intuition -> formal object -> theorem -> plain-language interpretation -> predictions -> method bridge. Keep notation minimal and stable.

### G26. Method

Begin from the theoretical control object, derive the practical update, define every weight and stop-gradient choice, explain computational cost, and map ablations to design alternatives.

### G27. Experiments

Introduce environment construction and evidence roles before results. Organize result sections by research question, not by a chronology of runs. Each result paragraph states the decisive comparison, uncertainty, mechanism observable, and claim supported.

### G28. Appendix

The appendix is part of the evidence package, not a dumping ground. It contains proofs, environment details, protocols, full results, negative results, implementation details, citation verification, and correction ledgers that would interrupt the main arc.

---

## 7. Paragraph and prose discipline

### G29. One paragraph, one claim

A paragraph has:

1. a topic sentence that states the claim;
2. a short logical development;
3. evidence, equation, or citation;
4. a takeaway or transition.

Do not begin with background and reveal the point only in the final sentence.

### G30. Put old information before new information

Within a sentence and across sentences, begin from the concept already active in the reader's mind and end with the new or emphasized information.

### G31. Use calibrated verbs

- theorem/derivation: prove, derive, imply;
- decisive controlled evidence: identify, causally support, isolate;
- external observation: observe, find, associate;
- finite-step evidence: support, indicate;
- hypothesis: suggest, motivate.

### G32. Prefer concrete actors and verbs

Name the policy, update, sample, theorem, or method. Avoid empty abstractions such as “the framework facilitates enhancement” when a concrete actor and action can be named.

### G33. Separate drafting from compression

First obtain scientific completeness and correct logic. Then run a separate compression pass for redundancy, notation, sentence length, and page budget. Compression may not delete the premise required by a later conclusion.

---

## 8. Figures, tables, citations, and LaTeX

### G34. Figure 1 carries the paper's causal story

A reader should understand the central tension, missing link, and method intervention from Figure 1 and its caption without reading the full paper.

### G35. One visual, one primary claim

Axes, legends, controls, uncertainty, and status must be readable without searching the prose. Do not use a performance plot as evidence for a mechanism unless the mechanism observable is shown.

### G36. Tables are comparison contracts

Rows and columns must encode the scientific comparison: matched budgets, best versus terminal, mechanism versus task metrics, or failure taxonomy. Avoid decorative tables that merely repeat prose.

### G37. Captions are mini-arguments

A strong caption states:

1. what is manipulated;
2. what is measured;
3. the decisive pattern;
4. the claim the pattern supports.

### G38. Verify every citation

Never create a BibTeX entry from memory. Prefer the published version over an arXiv duplicate, verify title/authors/year/venue, and ensure every citation supports the exact sentence containing it.

### G39. The Overleaf project must compile from a clean checkout

The repository stores the venue template, generated sections, figures, tables, bibliography, build script, and compiled review PDF. Missing images, undefined references, and bibliography errors block activation.

---

## 9. Cascade and automation

### G40. The hierarchy is bidirectional but conflict-intolerant

The registered layers are:

\[
\text{strategy}
\leftrightarrow
\text{outline}
\leftrightarrow
\text{blueprint}
\leftrightarrow
\text{prose}
\leftrightarrow
\text{TeX/appendix/figures}.
\]

Stable node IDs bind the layers. A change in one layer propagates to all linked layers. If two layers change the same node independently, the pipeline must stop and require an explicit preference; it may not guess which change is authoritative.

### G41. Semantic changes require structured metadata

Free-form prose cannot be reliably reverse-engineered into a new scientific claim. Every prose block therefore carries a structured claim field. Changing the claim through any layer updates the upstream and downstream representations automatically; changing wording without changing the claim leaves the structural parents intact.

### G42. Generated artifacts are reproducible projections

The manuscript graph stores stable node content and provenance. Outline, blueprint, prose, TeX, appendix, and Overleaf packages are generated projections. Direct edits are allowed only through registered blocks and must be imported by the synchronization command.

---

## 10. Mandatory activation gate

A manuscript artifact may become active only when all applicable checks pass:

1. **AUTH:** content matches the user-approved version or has explicit new-version authorization;
2. **IDENTITY:** the manuscript relationship (revision versus follow-up) is correct;
3. **STORY:** one central tension and contribution sentence remain intact;
4. **GAP:** existing work and the precise missing link are both present;
5. **OBJECT:** theory, method, and experiment share a named object;
6. **CONTROL:** strongest rival explanation has a decisive control;
7. **STATUS:** every result uses its registered scientific status;
8. **FAIRNESS:** comparison protocol is matched or differences are explicit;
9. **TERMINAL:** dynamic claims have terminal evidence;
10. **FAILURE:** task, boundary, and numerical failures are separated;
11. **CASCADE:** stable IDs, parent hashes, and all downstream projections align;
12. **CITATION:** citations and BibTeX are verified;
13. **LATEX:** clean compilation succeeds and the PDF is visually inspected;
14. **DELTA:** the version diff is complete, human-readable, and authorized.

Critical failure of AUTH, IDENTITY, OBJECT, STATUS, CASCADE, or LATEX blocks activation.
