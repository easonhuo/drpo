# DRPO Development Workflow Optimization Project

**Claim:** `GOV-DEV-WORKFLOW-OPTIMIZATION-BENCHMARK-01`  
**Status:** staged disposable-prototype implementation active on the development branch; no default-route change is authorized  
**Repository base:** `main@7d0ecfbee3b9e44bbad97fb806c8806b604f75f6`

## 1. Purpose

This directory is the durable project hub for improving the DRPO repository-development and scientific-pilot integration workflow. It records:

- the problems that motivated each workflow mechanism;
- what the existing mechanisms already solve;
- remaining failure and friction classes;
- proposed optimizations and rejected alternatives;
- historical and controlled timing evidence;
- validation, adoption, rollback, and stop conditions;
- the current authorized next step.

A later session must be able to recover the full engineering context from repository documents rather than chat history.

## 2. Authority and reading order

This project document is subordinate to:

1. `AGENTS.md` for repository-wide operating rules;
2. `docs/handoff.md` for the unique research master and scientific execution order;
3. `experiments/registry.yaml` for experiment registration state;
4. the accepted contracts of each workflow component.

It is not a second research master. It does not alter experiments, scientific claims, seeds, thresholds, budgets, horizons, result statuses, or priorities.

A future session working on workflow optimization must read, in order:

1. this document;
2. `REPLAY_BENCHMARK_PROTOCOL.md`;
3. `IMPLEMENTATION_PLAN.md`;
4. the current scope for the active optimization iteration;
5. the relevant component contracts;
6. related incident and transition records.

## 3. Optimization objective

The objective is not maximal automation or maximal gate coverage. It is:

> Reduce end-to-end human and model effort while preserving or improving every existing correctness, provenance, authority, and scientific-safety guarantee.

A workflow feature is valuable only when the recurring loss it removes materially exceeds its implementation and maintenance cost.

The project therefore distinguishes four quantities:

- **risk avoided:** invalid runs, incorrect authority changes, provenance loss, or false scientific claims prevented;
- **time saved:** controlled wall time and active operation time reduced;
- **complexity added:** production code, tests, dependencies, state, routing, and maintenance burden;
- **coverage:** the task and failure classes for which the feature actually helps.

No optimization may be justified by architectural elegance alone.

## 4. Why this project exists

The repository contains several individually useful mechanisms, but repeated local fixes can become a patch stack: a new rule, script, gate, temporary workflow, or recovery convention is added for each incident. A locally correct feature can still make the global system slower and harder to maintain.

The recurring concern is not that every existing mechanism is useless. The concern is that the project previously lacked a common method for answering:

- Which real problems did this mechanism prevent?
- How often did those problems occur?
- How much time or compute did it save?
- Did it make any task materially slower?
- Is its maintenance cost lower than its benefit?
- Should it remain universal, become selective, or be removed?

Historical paired replay is introduced as the common validation framework for those questions.

## 5. Development history

This chronology preserves the reason for each major mechanism. Detailed incidents remain in their original records.

### 5.1 V1 dev-branch integration transaction

The accepted V1 transaction established one auditable path from a reviewed dev snapshot to a local `READY` candidate:

```text
plan → prepare → normalize → gate → finalize → READY
```

It owns source locking, scope/provenance audit, isolated candidate construction, handoff authority invocation, required gates, freshness checks, diagnostics, and terminal transaction state. It intentionally excludes automatic push, PR creation, approval, and merge.

This solved correctness and authority problems, but it left multiple operator-facing commands and intermediate-file placement steps.

### 5.2 Pilot-registration fastpath

The July 14 pilot-registration merge incident showed that manually assembling registration inputs and repeatedly reconstructing candidates created avoidable friction. The fastpath adapter was added to compile one reviewer-authored `DEV_PILOT_REGISTRATION_SPEC.yaml` into deterministic `PREPARED_INPUTS` while reusing V1 rather than creating a second integration engine.

The adapter merged through PR #62 and was activated as the default preparation route through PR #84. Activation did not change V1, scientific state, CI defaults, publication behavior, or merge authority.

### 5.3 First real safety observation

During the E7 frozen-critic trajectory-GAE pilot, real-data preparation stopped before any of 192 actor branches started and exposed two integration defects:

- canonical RunSpec placeholder handling;
- a mixed-precision validation mismatch between float32 storage and float64 reference computation.

No held-out seeds were accessed and no scientific result was claimed. This is real evidence that preflight and fail-closed checks can avoid expensive invalid execution. It does not prove that the overall workflow is faster.

### 5.4 Remaining adoption and composition friction

Recent E7/E8 work continued to use temporary source-export or importer paths, stale or rebuilt candidates, and manual stage coordination. In one E8 chain:

- PR #94 was a temporary source export;
- PR #95 closed without an effective change;
- PR #96 produced the final candidate;
- PR #97 later removed a duplicated execution stack from that candidate; no experiment had run from the removed stack.

This sequence must not be attributed to one cause without replay evidence. It does demonstrate that final success can conceal substantial intermediate work and that duplicate implementation is a real maintenance risk.

## 6. Initial historical timing snapshot

GitHub timestamps provide historical operational context even where local stage reports are missing.

| Historical item | Observed PR window |
|---|---:|
| PR #84 — fastpath activation | 34 min 23 sec |
| PR #86 — E7 result-evidence closure | 27 min 33 sec |
| PR #93 — evidence-locator contract | 39 min 15 sec |
| PR #96 — final E8 paper-aligned candidate | 36 min 42 sec |
| PR #94 → final merge of PR #96 | 2 hr 08 min 03 sec |

The last row includes temporary export, an empty/closed PR, rebuild, CI, review, and other real operational delays. It is not a controlled baseline and must not be directly compared with a clean replay run.

Historical time answers, “How difficult was the real past workflow?” Controlled paired replay answers, “How much does the candidate optimization itself change time?” Both are reported, but never merged into one estimate.

## 7. Existing component ownership

The following components remain independent owners of their current responsibilities:

| Component | Sole responsibility relevant here |
|---|---|
| Connected GitHub route | dev branch, Draft PR, exact-head CI, review, explicit approval, publication |
| Pilot-registration fastpath | reviewer spec → deterministic `PREPARED_INPUTS` |
| V1 transaction | reviewed snapshot → local `READY` transaction state |
| Handoff authority | schema-v3 delta normalization and materialized handoff/registry state |
| RunSpec/lane runner | registered execution identity and supervised launch |
| Results delivery | durable append-only result transport |
| Evidence locator | immutable discovery of delivered result evidence |

A future optimization may coordinate these owners. It may not duplicate or replace their domain logic.

## 8. What existing mechanisms solve

The existing safety and correctness kernel addresses:

- exact source and implementation identity;
- reviewer and approval binding;
- changed-path and provenance scope;
- stale main/dev references;
- deterministic registration input preparation;
- authoritative handoff and registry materialization;
- required local and GitHub gates;
- fail-closed transaction state;
- durable execution and result identity.

Its continuing value must be monitored independently by recording real defects detected, stage, severity, avoided cost, and false positives.

## 9. Problems not yet solved

The current evidence indicates unresolved coordination and adoption problems:

- manual transfer of prepared overlay and registration inputs;
- multiple commands and stage-specific recovery knowledge;
- temporary source-export or importer workflows;
- empty, stale, or rebuilt PRs;
- long-lived development branches used as final integration candidates;
- incomplete proof that the default fastpath was actually used;
- operational separation between pre-run registration and post-run closure;
- no established measurement of time saved by workflow changes.

These must remain separate from:

- scientific-design or implementation bugs;
- missing models, data, credentials, or dependencies;
- GPU, CPU, memory, or network failures;
- task-performance collapse;
- support or variance-boundary events;
- NaN/Inf numerical failure;
- weak method performance.

A coordination layer cannot be credited for problems outside its declared coverage.

## 10. Current architectural hypothesis

The working hypothesis is:

> The required domain components exist, but the safe path lacks one low-friction executable composition entry point across their boundaries.

A possible thin orchestration layer would only:

- invoke existing commands in the accepted order;
- place already generated deterministic inputs at required locations;
- read existing transaction states to continue, stop, or require a new attempt;
- maintain one persistent workspace per case;
- derive a rebuildable status and timing summary from existing records.

It would not:

- own scientific design;
- create a second authority or transaction state machine;
- change registry or handoff semantics;
- select scientific variables;
- schedule experiments or GPUs;
- weaken gates;
- push, create PRs, approve, or merge.

A disposable prototype of this hypothesis is now authorized only on the active development branch under `IMPLEMENTATION_PLAN.md`. Adoption, default-route activation, and method success remain unproven.

## 11. Evidence-before-code policy

Before a candidate optimizer may be evaluated or adopted:

1. identify repeated incidents or measurable loss;
2. define affected and unaffected task classes;
3. freeze a representative historical case inventory before candidate results;
4. freeze correctness and efficiency acceptance criteria;
5. freeze the replay toolchain and environment policy;
6. set production-code and maintenance budgets;
7. define rollback and stop conditions.

The reusable case validator, recorder, and equivalence test harness may be implemented before the final historical inventory because they do not encode candidate results or select cases. The candidate orchestrator may not be historically evaluated until the inventory is frozen.

The implementation remains a disposable prototype on the active dev branch. It may enter `main` only after passing the paired replay protocol and receiving a separate explicit merge approval.

## 12. Historical replay validation

Each case freezes:

- historical task base and source artifacts;
- frozen implementation identity;
- one shared reviewer-approved input/specification;
- expected paths and semantic outcome;
- the benchmark toolchain used by both A and B;
- required gates;
- environment and cache policy.

The two arms are:

- **A — accepted baseline:** the existing component path;
- **B — candidate optimization:** the proposed wrapper, orchestrator, or other change while calling the same components.

The candidate must first produce equivalent repository, authority, provenance, and gate outcomes. Only then is time compared.

The detailed protocol is `REPLAY_BENCHMARK_PROTOCOL.md`.

## 13. No-regression and adoption policy

The desired result is that every in-scope case becomes faster. Measurement noise is handled as a tie tolerance, not as an allowed slowdown.

For universal adoption:

- correctness, safety, and provenance have zero regression tolerance;
- no case may show a material time regression greater than `max(60 seconds, 5% of baseline median controlled-replay time)`;
- median controlled wall time must decrease by at least 30%;
- mean controlled wall time must also decrease;
- median active operation time must decrease by at least 30%;
- explicit command count must decrease by at least 60%;
- manual intermediate-file copies must be zero;
- temporary workflows and temporary PRs must be zero for covered cases;
- implementation must remain within its frozen complexity budget.

A narrow task-class route is allowed only when its scope and routing are predeclared, deterministic, and simpler than a universal path. A poorly performing case may not be excluded after results are known.

## 14. Complexity limits

For the first orchestration prototype:

- target production code: 300–450 lines;
- mandatory redesign review above 500 production lines;
- no new third-party dependency;
- no changes to V1 core, handoff authority, registry schema, scientific code, or GitHub merge behavior;
- no automatic push, PR creation, approval, or merge;
- no new domain state beyond append-only raw events and derived, rebuildable summaries;
- no E7- or E8-specific scientific branches.

Crossing a boundary triggers redesign or cancellation, not silent scope expansion.

## 15. Bounded iteration model

Each optimization is one finite iteration:

1. documentation, architecture, and threshold freeze;
2. case validator and replay-harness implementation;
3. correctness-equivalence and failure-injection tests;
4. historical replay-case inventory freeze;
5. accepted-path baseline replay;
6. disposable candidate orchestration;
7. candidate replay and independent comparison;
8. at least three real production observations after adoption;
9. retain, narrow, redesign, or roll back.

Each numbered implementation step has its own goal, changed-path scope, focused tests, review, and stop decision. A later step may not hide an earlier defect with cross-module special cases.

A later optimization starts a new iteration. It may reuse the benchmark protocol but must not continuously expand one orchestrator to absorb unrelated responsibilities.

## 16. Required engineering evidence

Each real use of an existing mechanism should record, when available:

- issue detected or blocked;
- detecting component and stage;
- severity and avoided cost;
- false-positive status;
- safe route used or bypassed;
- manual fallback reason;
- temporary workflow or PR use;
- final terminal state.

Each benchmark iteration retains only the frozen `CASE_INVENTORY.yaml`, append-only `RAW_RESULTS.jsonl`, derived `PAIRED_COMPARISON.json`, and reviewed `DECISION.md`. These are engineering artifacts and do not enter the scientific experiment registry.

## 17. Current state and next permitted action

As of the implementation authorization:

- the existing safety and registration components remain active and unchanged;
- a unified lifecycle orchestrator does not yet exist;
- staged disposable-prototype implementation is authorized on `dev/gov-dev-workflow-optimization-benchmark-01`;
- Step 0 architecture and scope documentation is in progress;
- Step 1 may implement only the strict case model and static validation after Step 0 review;
- no candidate historical results may be inspected before the case inventory is frozen;
- no E7/E8 scientific experiment execution is authorized;
- no default-route change, Ready-for-review transition, or merge is authorized by the implementation approval.

## 18. Related records

- `docs/development_workflow_optimization/IMPLEMENTATION_PLAN.md`
- `docs/dev_branch_integration_protocol.md`
- `docs/dev_pilot_registration_fastpath.md`
- `docs/development_workflow_incident_and_improvement_log.md`
- `docs/development_workflow_incidents/README.md`
- `docs/development_workflow_incidents/DEVOPT-2026-07-14-PILOT-REGISTRATION-MERGE-01.md`
- `docs/development_workflow_transitions/GOV-DEV-PILOT-REGISTRATION-FASTPATH-TRANSITION-01.md`
- `docs/scopes/GOV-DEV-PILOT-REGISTRATION-FASTPATH-01.md`
- `docs/scopes/GOV-DEV-PILOT-REGISTRATION-FASTPATH-ACTIVATION-01.md`

## 19. Update discipline

This document is append-preserving. Later decisions must retain earlier problem statements, incidents, benchmark results, rejected alternatives, and rollback records. Do not rewrite history to make a later solution appear inevitable.