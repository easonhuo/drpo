# RL Paper Writing Playbook v1.0

**Status:** active operational handbook.

This Playbook turns the stable rules in `RL_PAPER_WRITING_GUIDANCE.md` into executable writing moves. It may grow as useful techniques are extracted from high-quality RL papers and open-source writing skills, but every addition must remain consistent with Guidance and the scientific record.

The Playbook is not a source of scientific truth. It is a construction and review manual.

---

## 1. End-to-end workflow

Use ten passes. Do not collapse them into one “write the paper” prompt.

### Pass 1 — Evidence inventory

Create or update:

- claim-evidence matrix;
- experiment status table;
- citation ledger;
- figure/table inventory;
- unresolved scientific questions;
- prohibited claims.

No prose is written before the main claims have evidence owners.

### Pass 2 — Story and missing link

Write:

1. one-sentence contribution;
2. one central tension;
3. strongest rival explanation;
4. decisive control;
5. transferable conclusion.

Reject the story if it requires more than one independent paper-level tension.

### Pass 3 — Outline

For every stable node, specify:

- reader question;
- claim;
- rhetorical role;
- required evidence;
- must-include points;
- must-avoid points;
- dependency on earlier nodes.

An outline is a structural contract, not shorthand notes.

### Pass 4 — Paragraph blueprint

For every node, specify:

- topic sentence;
- logical moves in order;
- equation/citation/result use;
- likely reviewer objection;
- transition to the next node;
- target length.

Blueprints should make prose generation nearly mechanical.

### Pass 5 — Complete prose

Write for scientific completeness, not brevity. Use the blueprint order. Leave explicit `TBD` markers when evidence is unavailable; never fill gaps with plausible-sounding results.

### Pass 6 — Theory-method-object audit

Trace every central symbol through:

- problem setup;
- theorem;
- method update;
- code/config;
- mechanism metric;
- ablation.

Breaks in the chain are resolved before stylistic editing.

### Pass 7 — Reviewer simulation

Run separate reviews from four perspectives:

- theory reviewer: assumptions, theorem value, missing cases;
- mechanism reviewer: confounds, causality, proxy validity;
- empirical reviewer: baselines, budgets, seeds, terminal audit;
- narrative reviewer: central tension, paragraph purpose, unsupported framing.

The review report uses Critical / Important / Minor severity.

### Pass 8 — Scientific writing quality

Run sequential passes:

1. clutter removal;
2. topic-sentence and information-flow check;
3. verb and claim calibration;
4. notation and terminology consistency;
5. paragraph transitions;
6. abstract and Introduction rewrite after results are known.

### Pass 9 — Compression and page budget

Cut only after the scientific chain is complete. Prefer:

- moving proofs and exhaustive settings to the appendix;
- merging repeated explanations;
- replacing prose lists with one table;
- removing historical narration;
- shortening formula commentary that is already visible in the equation.

Do not cut the rival explanation, decisive control, or theorem-to-method bridge.

### Pass 10 — Submission package

Verify:

- clean LaTeX build;
- PDF visual render;
- references and labels;
- figure legibility;
- anonymization mode;
- appendix inclusion;
- repository and artifact provenance;
- Overleaf ZIP opens and compiles without local-only paths.

---

## 2. Claim-evidence engineering

Maintain `docs/manuscript/claim_evidence_matrix.yaml` or an equivalent graph representation.

Each claim record contains:

```yaml
id: CLAIM-EXAMPLE
claim: concise falsifiable statement
type: theorem | mechanism | causal | task_effect | robustness
status: analytically_proven | long_run_validated | finite_step_validated | pilot | not_run
owner:
  theorem: THM-1
  experiment_ids: [C-U1-E1]
rival_explanation: ...
decisive_control: ...
observable: ...
allowed_language: ...
forbidden_language: ...
```

A paragraph may cite a claim only if its allowed language matches the current status.

### Claim ladder

Use the weakest statement sufficient for the contribution:

1. **Observation:** variable X is larger in condition Y.
2. **Isolation:** X differs when rival factors are matched.
3. **Causal support:** targeted intervention on X changes outcome Z.
4. **Mechanism closure:** source, timing, intervention, and mediator align.
5. **External validation:** the signature and task effect appear in realistic tasks.
6. **General theorem:** the property follows under stated assumptions.

Do not skip levels by changing verbs.

---

## 3. Abstract construction

Target six moves and approximately 150–220 words unless the venue requires otherwise.

### Move A — Resource or practical need

State why the object is valuable before saying it fails.

Bad: “Negative gradients are dangerous.”

Better: “Negative feedback suppresses bad behavior and can improve a policy beyond positive-only learning.”

### Move B — Failure and missing link

Name the condition that turns the resource into a problem and the confound that prevents existing evidence from identifying it.

### Move C — Theory

State one theorem-level transition or property. Do not list all lemmas.

### Move D — Method

Name the term or distribution changed by the method. Avoid feature lists.

### Move E — Evidence

Name the evidence architecture: matched isolation, intervention, controlled phase test, external closure. Include numbers only after formal freeze.

### Move F — Implication

End with the transferable design principle.

### Abstract audit

- Can a reviewer identify the missing link?
- Does the method modify the object named by theory?
- Are unfinished results absent?
- Is there any history of “old method” and “new method” that belongs elsewhere?

---

## 4. Introduction paragraph recipes

### P1 — Establish the resource

Template:

> X is useful because A and B. Existing positive-only or imitation approaches capture A but miss B. This raises the central question: how can we retain X while preventing failure Y?

End with a question, not a literature survey.

### P2 — Explain how the failure emerges

Use a temporal or causal chain. Define the key state change before introducing a new method.

### P3 — Existing controls plus missing identification link

This paragraph must do both jobs.

1. Group existing solutions by what they control.
2. Credit their success.
3. State what transition they do not explain.
4. State the confound or rival explanation.
5. Preview the decisive isolation.

Do not replace prior-work positioning with the new experiment; do not list prior work without the identification gap.

### P4 — Theory answer

State the qualitative regimes. Avoid assumptions, proof details, Jacobians, and unrelated guarantees.

### P5 — Method consequence

Use the sequence:

> The theorem identifies term X. The method changes X to X'. Near/local behavior is retained; far/dangerous behavior is attenuated. Proposition Y gives the key property.

Do not narrate the current paper as an old algorithm plus a new add-on.

### P6 — Evidence and contributions

Use research questions, not experiment chronology. Contributions should be conceptual nouns plus one-line consequences, not marketing labels.

### Introduction-twice rule

Draft the Introduction after the outline, then rewrite it after the main results and figures are finalized. The second draft is not cosmetic; it aligns the promised evidence with what the paper actually establishes.

---

## 5. Related Work construction

### Method-line grouping

Group papers by the control object or scientific question, for example:

- learning from negative/suboptimal data;
- off-policy, stale, and rare updates;
- conservative or behavior-regularized offline RL.

For each group:

1. state the common contribution;
2. identify the closest papers;
3. describe overlap accurately;
4. end with the missing bridge addressed here.

### Citation discipline

- Cite the published venue version when available.
- Do not use one citation to support multiple distinct claims.
- Never write “all prior work” unless the literature search supports it.
- Avoid adversarial adjectives; the gap should emerge from object and evidence differences.

---

## 6. Theory construction

### 6.1 The theorem-value test

Before keeping a theorem in the main text, answer:

- What question becomes answerable only after this theorem?
- Which method choice follows from it?
- Which experiment maps to its cases?
- What would the paper lose if it moved to the appendix?

If the answer is “it adds rigor” but not a new decision or prediction, compress or move it.

### 6.2 Theorem presentation template

1. **Physical question.** One paragraph of intuition.
2. **Minimal formal object.** Define only required symbols.
3. **Statement.** Put qualitative cases in the theorem.
4. **Translation.** Explain the result in ordinary language.
5. **Predictions.** Table mapping cases to observables.
6. **Method bridge.** Identify the controlled term.
7. **Appendix proof.** Full algebra, secondary cases, technical conditions.

### 6.3 Assumption placement

Put an assumption next to the theorem it enables. Do not turn a local analytical specialization into the declared scope of the entire paper.

### 6.4 Notation budget

For every new symbol, record:

- why an existing symbol cannot serve;
- first definition;
- section lifetime;
- continuous/categorical mapping;
- collision check.

Delete symbols used once when prose is clearer.

### 6.5 Theory-experiment mapping table

Use columns:

| Theoretical regime | Intervention | Observable | Terminal criterion | Failure type |
|---|---|---|---|---|

This table belongs immediately after the central theorem or at the start of experiments.

---

## 7. Method construction

### 7.1 Equation chain

Write equations in this order:

1. raw objective/update;
2. isolated dangerous or useful term;
3. method transformation;
4. final training loss/update;
5. implementation-specific detach, normalization, and clipping;
6. complexity.

Every line should answer “what changed and why?”

### 7.2 Method identity

A method section must be self-contained for the current manuscript. Research history may be recorded internally or in a correction ledger, but the reader must not need an earlier version to understand the method.

### 7.3 Component audit

For each component, fill:

| Component | Mathematical role | Code path | Hyperparameter | Ablation | Failure if removed |
|---|---|---|---|---|---|

If a component has no independent role or ablation, consider removing it.

### 7.4 Baseline family

Include simple endpoints and matched controls:

- no control;
- delete all negative feedback;
- global scale;
- selective scale;
- hard threshold on the same coordinate;
- quality control on a distinct coordinate;
- proposed method.

Never claim that a quality filter is the limit of a distance taper without a derivation.

---

## 8. Experiment and results construction

### 8.1 Environment paragraph

Before results, explain:

- state/context and action spaces;
- ground-truth optimum or verifier;
- how positives and negatives are generated;
- which variables are matched;
- how near/far or common/rare is defined;
- train/test relationship;
- what the environment can and cannot establish.

Write the final sentence positively: “The controlled design isolates X; external tasks test whether the signature persists under Y.”

### 8.2 Research-question structure

For each RQ:

1. question;
2. rival explanation;
3. decisive comparison;
4. metric and uncertainty;
5. result;
6. mechanism diagnostic;
7. conclusion allowed;
8. link to next RQ.

### 8.3 Result paragraph template

> To test CLAIM, we compare A with B under MATCHED CONDITIONS. A changes METRIC by RESULT (UNCERTAINTY), while CONTROL C does/does not change it. The mechanism observable X changes before or together with outcome Y. This supports CLAIM and rules out RIVAL EXPLANATION within the registered environment.

Do not mix two independent RQs in one paragraph.

### 8.4 Negative and inconclusive results

Use explicit language:

- “The run completed but did not reach a terminal classification.”
- “The metric is undefined because the denominator is zero; no retention claim is made.”
- “Best-checkpoint gains reverse at the terminal checkpoint.”
- “The comparison does not support a stable ranking.”

### 8.5 Best versus terminal

Best performance answers selection quality. Terminal performance answers dynamical stability. Report both when the paper discusses drift, collapse, or equilibrium.

### 8.6 Mechanism before ranking

A proposed method may lose a task metric yet still validate a mechanism, or win a metric for an unrelated reason. Separate:

- mechanism result;
- task result;
- method ranking;
- robustness result.

---

## 9. Figures and tables

### 9.1 Figure 1 storyboard

Use 3–5 panels:

1. useful baseline resource;
2. missing transition or confound;
3. theoretical regimes;
4. method intervention;
5. evidence architecture.

The caption should be understandable before the reader knows all notation.

### 9.2 Plot selection

- trajectories for dynamics and onset ordering;
- scatter or binned plots for source isolation;
- paired lines/differences for paired seeds;
- phase diagrams for theorem regimes;
- tables for multi-dataset performance;
- failure matrices for task/boundary/numerical separation.

### 9.3 Figure quality gate

- labels readable at final two-column size;
- uncertainty visible;
- no color-only semantics;
- controls ordered consistently across figures;
- formal versus pilot/TBD clearly marked;
- axes and normalization defined in caption;
- no cropped legends or rasterized tiny text.

### 9.4 Caption template

> (Setup) We vary X while matching Y. (Measure) The panels report Z. (Pattern) Condition A changes before/relative to B. (Claim) This isolates/supports C.

### 9.5 Table templates

**Mechanism table:** method, negative budget, near retention, far retention, drift, boundary, task collapse, NaN/Inf.

**External table:** dataset, method, best, terminal, uncertainty, collapse rate, mechanism diagnostic.

**Environment table:** environment, control/realism, claim owner, non-responsibility.

---

## 10. Appendix construction

Plan the appendix with the outline, not after main-text overflow.

Required modules for a theory-plus-method RL paper:

1. full theorem proofs;
2. family-specific derivations;
3. environment generation and invariants;
4. hyperparameters and seeds;
5. terminal audit definitions;
6. complete per-seed results;
7. negative/inconclusive results;
8. implementation details and pseudocode;
9. citation and artifact provenance;
10. correction ledger for superseded scientific statements.

The appendix may preserve provenance, but it must not reframe a revision as a sequel.

---

## 11. Scientific prose passes

### Pass A — Clutter extraction

Remove:

- “It is worth noting that”;
- “in order to” when “to” works;
- repeated nouns in adjacent sentences;
- adjective chains without measurable content;
- restatements of the displayed equation.

### Pass B — Strong subjects and verbs

Prefer:

- “Far-zero prevents boundary crossing”

over
- “Boundary crossing is shown to be prevented by Far-zero.”

Use passive voice when the procedure or object, not the actor, is the topic.

### Pass C — Stress position

Put the new or important information at the end of the sentence and paragraph.

### Pass D — Logical connectors

Use connectors only when the relation is real:

- however = contrast;
- therefore = deduction;
- consequently = causal consequence;
- in contrast = paired comparison;
- specifically = refinement.

### Pass E — Claim calibration

Replace universal language unless the theorem or evidence supports it. Verify every “proves,” “causes,” “general,” “robust,” “stable,” “converged,” and “state of the art.”

### Pass F — Compression

Target one main idea per sentence and one claim per paragraph. Preserve equations, rival explanations, and decisive controls.

---

## 12. Multi-reviewer audit

Run reviews independently before synthesis.

### Theory reviewer

Checks:

- assumptions match statement;
- theorem not larger than its role;
- counterexamples and boundary cases;
- notation collisions;
- method actually follows.

### Empirical reviewer

Checks:

- strongest baselines;
- matched budgets and seeds;
- selection leakage;
- terminal evidence;
- variance and uncertainty;
- missing negative results.

### Mechanism reviewer

Checks:

- badness versus distance confounds;
- proxy validity;
- source versus transmission;
- intervention specificity;
- direction and aggregation.

### Narrative reviewer

Checks:

- one central tension;
- prior work acknowledged;
- paragraph moves;
- absence of defensive drift;
- no old-paper-plus-sequel framing;
- title/abstract/conclusion consistency.

### Clarity reviewer

Checks:

- topic sentences;
- information order;
- long sentences;
- undefined acronyms;
- figure and table readability;
- standalone captions.

The synthesis report lists Critical issues first and does not average away a single fatal objection.

---

## 13. Automated manuscript pipeline

The repository pipeline uses stable node IDs and a manuscript graph.

### 13.1 Canonical objects

- graph: `docs/manuscript/paper_graph.yaml`;
- outline projection;
- blueprint projection;
- prose projection;
- generated TeX sections and appendices;
- figures/tables;
- pipeline state with per-node hashes;
- compiled PDF and Overleaf package.

### 13.2 Downward synchronization

An outline edit updates the graph and regenerates the linked blueprint, prose, TeX, appendix references, and build package. The default deterministic backend creates a conservative scaffold; a trusted generator command may provide higher-quality rewriting.

### 13.3 Upward synchronization

A blueprint or prose block carries a structured claim field. Changing that field updates the corresponding outline claim and all other layers. Pure wording edits preserve the structural claim.

### 13.4 Conflict rule

If two layers change the same node since the last state snapshot, synchronization stops. The user must specify the preferred layer. The system never resolves a scientific conflict by guessing.

### 13.5 Commands

```bash
python3 scripts/paper_pipeline.py render --repo-root .
python3 scripts/paper_pipeline.py sync --repo-root .
python3 scripts/paper_pipeline.py validate --repo-root .
python3 scripts/paper_pipeline.py compile --repo-root .
python3 scripts/paper_pipeline.py package-overleaf --repo-root .
python3 scripts/paper_pipeline.py all --repo-root .
```

For a trusted external generator:

```bash
python3 scripts/paper_pipeline.py sync \
  --repo-root . \
  --generator-cmd 'trusted-paper-generator'
```

The command receives a node JSON object on stdin and returns updated `blueprint` and `prose` fields. No third-party skill code is executed by default.

---

## 14. Techniques retained from reviewed open-source skills

The following techniques were retained after manual review:

- claim-evidence engineering rather than prose-first generation;
- Brainstorm -> Draft 0 -> Evaluate -> Write -> Compress;
- introduction-twice revision;
- every experiment serves a named claim;
- Figure 1 as a story carrier;
- paragraph rhetorical-move templates;
- five-pass scientific writing audit;
- citation verification and duplicate BibTeX removal;
- LaTeX hardening and clean-build checks;
- independent specialist reviewers;
- deterministic figure specifications where possible;
- evidence-backed submission package with build notes.

Techniques rejected or constrained:

- autonomous invention of claims or results;
- direct execution of unreviewed third-party scripts;
- multi-agent voting as a substitute for scientific authority;
- automatic citation generation from model memory;
- free-form reverse synchronization without structured node metadata;
- rewriting an approved artifact under the same version label.
