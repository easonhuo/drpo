# Paper Writing Optimization Iteration Log

**Status:** active long-term iteration record.

**Current initiative:** `PAPER-WRITING-LOGIC-FIRST-01`.

**Authority boundary:** this document records manuscript-process incidents, design decisions, implementation plans, cost budgets, validation criteria, and observed outcomes. It is not a scientific source of truth, does not replace `docs/handoff.md` or `experiments/registry.yaml`, and does not override the stable rules in `RL_PAPER_WRITING_GUIDANCE.md`. Historical entries are append-only; later revisions may supersede a design decision but must not silently erase the original record.

---

## 1. Purpose

The repository already contains four distinct manuscript layers:

1. `RL_PAPER_WRITING_GUIDANCE.md` defines slow-moving cross-paper quality and authorization rules.
2. `RL_PAPER_WRITING_PLAYBOOK.md` provides operational writing and review techniques.
3. `paper_graph.yaml`, the approved outline, paragraph blueprint, and manuscript specifications store the current manuscript structure and evidence bindings.
4. The paper pipelines compile, validate, typeset, and package the manuscript.

What was missing was a durable place to record how this system itself should improve after real writing failures. This log fills that gap. It should answer:

- What failure or inefficiency was observed?
- Which existing layer failed to prevent it?
- What design change is proposed?
- What user and compute cost is acceptable?
- What is implemented, validated, rejected, or still uncertain?
- Which measurements determine whether the change should remain?

This document is therefore the manuscript-process analogue of an engineering improvement log, not a second writing handbook.

---

## 2. Relationship to the existing manuscript system

The intended division of responsibility is:

```text
scientific authority
  docs/handoff.md + experiments/registry.yaml
        ↓
stable writing constitution
  RL_PAPER_WRITING_GUIDANCE.md
        ↓
operational writing knowledge
  RL_PAPER_WRITING_PLAYBOOK.md + approved skill corpus
        ↓
logic-first authoring controller
  required artifacts, approval states, scope authorization, stale propagation
        ↓
structured manuscript representation
  outline + paragraph blueprint + paper_graph/spec
        ↓
existing paper pipeline
  evidence binding, prose realization, validation, LaTeX, PDF, release
```

The new controller is not another independent paper pipeline. It is a constrained front-end for the existing hierarchy:

```text
approved outline -> paragraph blueprint -> prose -> TeX/PDF
```

Its job is to make already-documented rules unavoidable at execution time. The skills explain how to reason; the controller determines which reasoning artifact must exist before the next stage may run; the existing pipeline performs downstream compilation and release.

---

## 3. Incident record: Introduction rewrite loop

### 3.1 Observed failure

During the July 2026 Introduction revision, the repository already contained rules equivalent to:

```text
story -> outline -> paragraph blueprint -> prose -> review -> compression
```

However, the first revision attempts did not actually use a chapter-level logic map or a paragraph-level logic map. The result was repeated broad rewriting, inconsistent preservation of approved prose, and several rounds of correction. The useful version emerged only after the following sequence became explicit:

```text
chapter logic map
  -> paragraph responsibilities
  -> sentence-level logical moves
  -> constrained prose revision
  -> redundancy and clarity review
```

The failure was therefore not primarily a lack of writing knowledge. It was an enforcement failure:

- the relevant skill existed but could be skipped;
- prose generation did not depend on an approved logic artifact;
- review permission could expand into unauthorized rewriting;
- no sentence-level `KEEP / TRIM / REVISE / MOVE / ADD` mapping constrained the delta;
- no gate failed when an upstream artifact was missing.

### 3.2 Root cause

The current system is strong after structured artifacts already exist: it has stable IDs, parent hashes, conflict rejection, evidence bindings, and deterministic release checks. It is weaker before prose generation, where model compliance with planning skills is still largely voluntary.

The missing component is a fail-closed, incremental authoring control layer between the skill system and the manuscript compiler.

### 3.3 Decision

The user approved development of a logic-first incremental gate, subject to one hard usability requirement:

> The constraint must prevent unplanned rewriting without turning every wording interaction into a full multi-stage rebuild.

The accepted target is an average local-interaction overhead of no more than roughly 15%, with full chapter replanning reserved for genuine chapter-level changes.

---

## 4. Approved design: `PAPER-WRITING-LOGIC-FIRST-01`

### 4.1 Design principles

1. **Logic before prose.** No substantive prose generation without the required approved logic artifact.
2. **Reuse before rewrite.** Existing approved sentences are reused unless a recorded logical defect requires change.
3. **Authorization before delta.** Every substantive modification must map to an approved node and operation.
4. **Incremental invalidation.** Upstream changes invalidate only affected descendants, not the whole manuscript by default.
5. **Fail closed on missing state.** The system must reject a skipped stage rather than silently synthesize it.
6. **Review is auditable.** Every accepted edit records the original text, replacement text, reason, and claim-strength impact.
7. **No forced churn.** A review pass that finds no defect records `PASS`; it must not edit text merely to demonstrate activity.

### 4.2 Required artifacts

#### A. Section logic map

A section-level artifact records:

- central reader question;
- entry point and exit point;
- ordered causal or argumentative chain;
- paragraph responsibilities;
- dependencies on earlier sections;
- theory--method--evidence bridge;
- allowed and forbidden claims;
- approval status and parent fingerprint.

Example shape:

```yaml
section_id: INTRO
central_question: How can useful negative feedback be retained without excessive far-field repulsion?
chain:
  - signed policy updates
  - repeated historical reuse
  - learner-relative remoteness feedback
  - sample-level far-field amplification
  - aggregate Repulsive Dynamics
  - same-coordinate DRPO control
  - layered empirical evidence
status: approved
```

#### B. Paragraph logic map

Each paragraph records:

- one rhetorical responsibility;
- topic claim;
- ordered sentence roles;
- explicit dependency on the previous paragraph;
- transition to the next paragraph;
- evidence/citation/theorem bindings;
- reviewer objection and response;
- allowed conclusion and prohibited overclaim;
- approval status.

The current blueprint `sentence_plan` is the natural integration point. The design should strengthen it rather than create a competing representation.

#### C. Source mapping and edit authorization

Before revising approved prose, every planned sentence node receives one operation:

```text
KEEP -> TRIM -> REVISE -> MOVE -> ADD
```

`KEEP` has the highest priority. `ADD` is permitted only when the approved logical node cannot be realized from existing text.

Example:

```yaml
paragraph_id: INTRO-P02
source_version: DRPO_REWRITTEN_DRAFT_20260709
mapping:
  - sentence_node: P2.S1
    action: KEEP
    source_text: "Against this background, we identify ..."
  - sentence_node: P2.S3
    action: ADD
    reason: "Approved logic gap: policy geometry must connect remoteness to update influence."
```

The prose generator may modify only authorized nodes. A new claim without a corresponding approved node fails validation.

### 4.3 State machine

The proposed minimum state machine is:

```text
SOURCE_LOCKED
  -> SECTION_MAP_DRAFT
  -> SECTION_MAP_APPROVED
  -> PARAGRAPH_MAP_DRAFT
  -> PARAGRAPH_MAP_APPROVED
  -> SOURCE_MAPPING_COMPLETE
  -> PROSE_DRAFT
  -> AUDIT_PASSED
  -> USER_APPROVED
  -> FINAL_LOCKED
```

Rules:

- prose generation is forbidden before the required parent states are approved;
- changing a section map marks affected paragraph maps, source mappings, and prose as `stale`;
- changing one paragraph map invalidates that paragraph and relevant transitions, not unrelated paragraphs;
- wording-only edits do not automatically invalidate an approved section map;
- an approved prose version becomes a frozen source baseline for the next revision;
- a stale child may not be promoted to a release artifact.

### 4.4 Three execution levels

The controller must classify a requested change before deciding which gates to run.

#### Level 0 — Wording-only change

Examples: typo, grammar, concision, local clarity without changing claim, paragraph role, evidence, or order.

Required path:

```text
existing approved paragraph map
  -> local source mapping
  -> wording delta
  -> semantic-scope check
```

No section-map reapproval.

#### Level 1 — Paragraph-logic change

Examples: missing causal bridge, reordered sentence moves, added boundary sentence, changed paragraph transition.

Required path:

```text
paragraph map revision
  -> user approval when substantive
  -> local source mapping
  -> regenerate affected paragraph and adjacent transitions
```

No full-section rebuild unless the paragraph responsibility changes the section chain.

#### Level 2 — Section-logic change

Examples: changed central question, paragraph responsibility, theory--method bridge, evidence architecture, section ordering, or major claim.

Required path:

```text
section map revision
  -> approval
  -> affected paragraph maps marked stale
  -> incremental descendant rebuild
```

Only Level 2 triggers chapter-level replanning.

### 4.5 Review sequence

The prose audit should use separate passes rather than one generic request to "review several times":

1. **Logic coverage:** every approved node is realized; no unauthorized node appears.
2. **Paragraph responsibility:** each paragraph performs one main move; adjacent paragraphs do not duplicate duties.
3. **Scientific boundary:** claim verbs, evidence status, terminology, and theory/method/experiment responsibility remain valid.
4. **Compression:** repeated conclusions, duplicated setup, unnecessary summaries, and equation-visible prose are removed.
5. **Language and citation:** directness, antecedents, transitions, citation support, and author voice are checked.

Every proposed change records:

```text
problem
original text
replacement text
reason
scientific meaning changed: yes/no
approval required: yes/no
```

### 4.6 Generator contract

A prose generator should receive a constrained packet, not a generic `rewrite this section` request. The packet should include:

- approved section map;
- approved paragraph map;
- source version and source hash;
- source mapping operations;
- frozen `KEEP` sentences;
- authorized nodes;
- forbidden claims and terms;
- evidence and citation bindings;
- target length and transition contract.

Output validation should reject:

- changes to frozen sentences;
- sentences with no approved node;
- missing required nodes;
- claim-strength upgrades;
- paragraph merge, split, rename, reorder, or silent omission;
- use of stale parent artifacts.

---

## 5. Cost and usability budget

The design is accepted only as an incremental system. It must not make every interaction run the complete manuscript pipeline.

### 5.1 Expected per-task overhead

| Task type | Expected additional latency | Target relative overhead |
|---|---:|---:|
| typo / grammar / local wording | 10--30 seconds | about 5% |
| paragraph polish without logic change | 1--3 minutes | about 10--20% |
| paragraph logic change | 3--8 minutes | about 20--40% |
| new or restructured section | 15--30 minutes of planning | intentionally higher upfront cost |
| formal release | 1--3 minutes beyond existing checks | small relative to compilation and rendering |

These values are design estimates, not measured performance claims.

### 5.2 System-level targets

- average local authoring overhead: `<= 15%`;
- no section-map reapproval for verified Level 0 edits;
- no whole-manuscript invalidation for a local paragraph change;
- reduction in rejected or reverted prose rounds: target `30--60%` for major sections;
- no increase in unauthorized claim changes;
- logic checks remain separate from expensive LaTeX/PDF release checks;
- if ordinary copyedits repeatedly trigger Level 2 gates, the implementation is considered too heavy and must be simplified.

### 5.3 Cost interpretation

The intended trade is:

```text
small, bounded planning cost
  in exchange for
fewer uncontrolled rewrites and fewer full-section correction rounds
```

For high-value sections such as the Abstract, Introduction, Theory bridge, and main experimental narrative, increased upfront planning is acceptable. For routine copyediting, the gate must remain nearly invisible.

---

## 6. Implementation plan

The design is approved; implementation has not started as of the initial entry below.

### Phase A — Minimal viable enforcement

Estimated engineering cost: 2--4 days.

Deliverables:

- schema fields for section/paragraph approval and stale state;
- source mapping with `KEEP / TRIM / REVISE / MOVE / ADD`;
- task-level classifier with explicit override;
- fail-closed prevention of prose generation when required parents are missing;
- validator for frozen sentences and unauthorized nodes;
- unit tests for Level 0, Level 1, and Level 2 paths.

### Phase B — Incremental integration

Estimated cumulative engineering cost: 5--8 days.

Deliverables:

- integration with existing blueprint `sentence_plan` and stable IDs;
- descendant invalidation and adjacent-transition handling;
- CLI commands for approve, mark-stale, map-source, generate, and audit;
- deterministic delta reports;
- migration path for existing approved manuscript artifacts;
- rollback to the prior Core pipeline without data migration.

### Phase C — Observability and optimization

Implement only after Phase A/B show value.

Deliverables:

- gate-trigger and bypass logs;
- per-level latency;
- rejected-delta counts;
- number of user correction rounds before approval;
- false-positive escalation rate;
- percentage of approved source sentences preserved;
- optional visualization of section and paragraph logic maps.

A UI, autonomous literature search, semantic merge engine, or full database migration is explicitly out of scope for the first implementation.

---

## 7. Acceptance and rollback criteria

### 7.1 Acceptance tests

The first implementation should pass at least these scenarios:

1. A typo fix preserves all logic approvals and changes only one authorized span.
2. A paragraph-level causal bridge invalidates only the paragraph and adjacent transition.
3. A section-level change invalidates all and only dependent descendants.
4. A `KEEP` sentence modified by the generator is rejected.
5. A new unsupported claim is rejected even when the prose is fluent.
6. A review with no identified defect produces `PASS` and no diff.
7. Existing scientific terminology and evidence gates remain intact.
8. The system can reproduce the approved Introduction logic chain without broad unauthorized rewriting.

### 7.2 Runtime acceptance

- median Level 0 overhead should remain under 30 seconds excluding model response time;
- median Level 1 orchestration overhead should remain under 3 minutes excluding user deliberation;
- full section planning should not be triggered by wording-only changes;
- the existing release pipeline remains independently usable.

### 7.3 Rollback

The first implementation must be additive. The approved outline, blueprint, prose, and paper graph remain readable without the new controller. Rollback consists of reverting the controller/schema integration commit and returning to the existing Core pipeline. No scientific record, approved prose, or evidence artifact may depend on an irreversible migration.

---

## 8. Metrics to maintain over time

Each substantial paper-writing cycle should append a compact observation:

| Metric | Meaning |
|---|---|
| task level | Level 0, 1, or 2 |
| planning latency | time spent before prose generation |
| generation rounds | candidate drafts before approval |
| user rejection rounds | revisions rejected for logic, scope, or unauthorized rewriting |
| preserved-sentence ratio | fraction of approved source sentences retained |
| unauthorized-node count | generated claims or moves with no approved node |
| false escalation | local request incorrectly promoted to a higher gate |
| release defects | unresolved citations, labels, figures, or build errors |
| net outcome | faster, neutral, or slower than the prior workflow |

The target is not maximal gate activity. A gate that rarely catches anything and materially slows routine work should be simplified or removed.

---

## 9. Iteration ledger

### 2026-07-14 — Initial logic-first design approved

- **Trigger:** repeated Introduction revisions produced broad and sometimes incorrect rewriting before the chapter and paragraph logic maps were made explicit.
- **Finding:** Guidance and Playbook already contained the right hierarchy, but they were advisory rather than an unavoidable generation dependency.
- **Decision:** introduce an incremental logic-first controller with section maps, paragraph maps, source mapping, approval state, and stale propagation.
- **Cost decision:** estimated average local overhead of 10--20% is acceptable; target average remains `<= 15%` after implementation and tuning.
- **Scope:** documentation and design only in this iteration; no pipeline behavior changes yet.
- **Next decision gate:** approve a concrete Phase A implementation scope, schema, rollback plan, and tests before modifying pipeline code.

---

## 10. Maintenance rules

1. Append a dated ledger entry after each major writing-process incident, design change, implementation milestone, rollback, or measured evaluation.
2. Distinguish `proposed`, `approved design`, `implemented`, `validated`, `rejected`, and `superseded` states.
3. Do not convert estimated costs or expected quality gains into measured claims without logs.
4. Do not duplicate stable cross-paper rules from Guidance; link them and record only the process implication or implementation gap.
5. Do not duplicate current scientific status from the handoff or registry.
6. Preserve rejected approaches and their failure reasons.
7. Update `docs/manuscript/README.md` when the active initiative or canonical process entry point changes.
