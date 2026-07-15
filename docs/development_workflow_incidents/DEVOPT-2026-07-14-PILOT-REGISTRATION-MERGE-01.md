# 2026-07-14 — DEVOPT-2026-07-14-PILOT-REGISTRATION-MERGE-01

**Parent ledger:** `docs/development_workflow_incident_and_improvement_log.md`  
**Implementation claim:** `GOV-DEV-PILOT-REGISTRATION-FASTPATH-01`  
**Implementation plan:** `docs/dev_pilot_registration_fastpath.md`  
**Scientific experiment impact:** none

## Context

A review of recent code-first pilot pull requests found that ordinary GitHub merge operations were not the main source of delay. The dominant cost came before merge: registration assembly, generated authority state, repeated exact-head CI, micro-commits, stale-base recovery, RunSpec/provenance synchronization, and branch reconstruction.

Observed pull requests:

| PR | State at review | Commits | Changed files | Relevant observation |
|---|---|---:|---:|---|
| #48 `EXT-H-E7-PPO-W0-EXP-GRID-01` | merged | 67 | 28 | successful final integration, but excessive branch history and repeated synchronization |
| #52 `EXT-H-E7-W0-HIGHC-ACTOR-01` | merged | 22 | 26 | fast final dependency-ordered merge, but registration and temporary-workflow history remained fragmented |
| #49 `EXT-C-E8 continuous EXP` | open Draft | 30 | 17 | scientifically scoped pilot remained unmerged while its base aged and registration after-images became harder to refresh |

These counts are evidence of workflow amplification, not evidence that the scientific implementations or final merged results were invalid.

## Intended change and expected complexity

The intended optimization is a thin preparation adapter over the accepted V1 dev-integration transaction. It should reduce manual input assembly and remote trial-and-error without changing scientific authority or final merge gates.

Expected implementation complexity: medium. The core source-lock, scope audit, registry mutation, schema-v3 normalization, gate selection, rollback, and local `READY` behavior already exist in V1 and must be reused.

## Observed elapsed-time pattern

The exact session duration was not consistently recorded for every PR, which is itself a telemetry gap. Available timestamps show:

- PR #48 remained open for approximately five hours before merge;
- PR #52 merged roughly thirty-two minutes after creation and immediately after its parent dependency;
- the final merge actions themselves occurred quickly once exact heads, dependencies, and checks were settled;
- PR #49 remained open into the next day and accumulated a stale-base burden.

The critical path was therefore not GitHub's merge API. It was the repeated preparation required to reach a trustworthy merge candidate.

## Chronological incident record

### 1. Code-first implementation and registration were interleaved

Implementation SHA, RunSpec pins, registry content, delta inputs, and materialized authority outputs were not consistently separated into stable milestones. A later implementation edit invalidated previously generated registration identities and forced regeneration.

### 2. CI was used as a generator and remote debugger

Temporary or specialized workflows were used to construct or diagnose registration state. Each correction created another remote cycle and sometimes another workflow edit. CI should independently verify an already prepared candidate, not be the primary place where the candidate is assembled.

### 3. Per-file writes amplified commit and CI counts

GitHub Contents API writes naturally create one commit per file. Without a local-finalization or atomic tree step, logically single changes expanded into many branch commits. Each new head could retrigger exact-head checks and increase reviewer/provenance ambiguity.

### 4. Full checks were repeatedly paid during exploratory synchronization

The repository already computes a tiered test plan, but the active PR workflow still ran the full repository checks after ordinary head changes. This preserved safety but increased feedback time and made small registration corrections expensive.

### 5. Stale main amplified generated-state work

While a registration branch was being repaired, `main` continued to advance. Authority-generated after-images and parent-sensitive provenance then required reconstruction on a newer base. Continuing to patch an old generated image had diminishing value.

### 6. Final merge was not the bottleneck

Once PR #48 and PR #52 had exact approved heads and passing checks, their merges occurred in close succession. This separates merge execution from merge preparation and focuses the optimization on the correct layer.

## Root causes

### RC-1: no canonical implementation-freeze boundary

The workflow did not consistently declare one immutable implementation commit before registration preparation began.

### RC-2: no compact high-level input for routine pilot registration

The accepted V1 inputs are safe but verbose. Repeated manual assembly increased transcription, hash-binding, and consistency work.

### RC-3: candidate generation and independent verification were mixed

Remote CI cycles performed both construction and validation, so a generation defect paid network, queue, and full-gate cost.

### RC-4: transport granularity leaked into logical history

Per-file API writes became logical commits instead of being treated as transport intermediates followed by a clean finalization boundary.

### RC-5: risk-selected test planning was not yet operational in edit feedback

The selector existed, but ordinary PR synchronizations still paid unconditional full-suite cost.

### RC-6: workflow performance telemetry was incomplete

The repository lacked consistent per-PR records for preparation time, queue time, gate time, rebuild count, first blocker, and unique blocker value. This limits evidence-based gate reduction.

## Impact

Engineering impact:

- longer time from reviewed implementation to a trustworthy candidate;
- repeated CI and reviewer work;
- increased stale-main probability;
- higher chance of confusing implementation, registration, launched, and merged SHAs;
- unnecessary temporary-workflow and branch-reconstruction work.

Scientific impact:

- no scientific conclusion is changed by this incident record;
- no experiment state is upgraded or downgraded;
- task-performance collapse, support/boundary events, and NaN/Inf classifications remain separate;
- pilot evidence remains pilot evidence.

## Immediate fixes already completed

- the accepted V1 transaction already provides exact source locking, scope audit, optional reviewer-bound registration inputs, trusted normalization, risk-selected required gates, final freshness checks, rollback diagnostics, and local `READY`;
- Stage-5 merge-history verification was separately repaired for normal merge-commit integration;
- recent successful integrations demonstrated the correct implementation-commit then immutable-registration/result-commit layout;
- the repository now has a chronological development-workflow incident ledger.

These fixes establish safe primitives but do not yet make the routine pilot-registration path concise.

## Proposed systemic optimization

### P0 — thin Pilot Spec adapter

Compile one strict reviewer-authored Pilot Registration Spec into existing V1 request, review, intent, and approval inputs. Do not create a new authority, state machine, or publisher.

### P0 — deterministic preparation and preflight

Validate all locally decidable identities, paths, add/replace semantics, before-image hashes, reviewer bindings, and generated hashes before V1 or CI begins. Publish the preparation directory atomically.

### P0 — logical commit discipline

Freeze implementation first. Treat transport micro-commits as intermediate and present a clean logical implementation boundary followed by one immutable registration/result boundary.

### P1 — tiered CI and telemetry

After PR-A is accepted, separately authorize operational edit/standard/full tiers. Preserve a final exact-head full merge gate and provide immediate rollback to unconditional full checks.

### P2 — data-driven gate reduction

Only after enough real PR observations should duplicated gates be merged or removed. Removal requires evidence that the gate adds no unique blocker value and a documented rollback.

## Acceptance criteria

The implementation plan must demonstrate:

1. one high-level spec generates existing V1-compatible inputs deterministically;
2. no authority, registry, handoff, or scientific semantic logic is duplicated;
3. no registration approval is invented;
4. add/replace, target scope, before-image, path, and SHA errors fail before publication;
5. failed preparation publishes no partial candidate;
6. an identical rerun is idempotent and a conflicting output is rejected;
7. a historical code-plus-registration case can be replayed;
8. PR-A changes no workflow default and removes no gate;
9. exact-head repository checks pass;
10. PR-B is separately reviewed after PR-A evidence.

## Cross-incident problem map

| Problem family | This incident | Earlier squared-EXP incident | Planned owner |
|---|---|---|---|
| implementation/registration coupling | yes | yes | PR-A adapter and freeze contract |
| command-contract gap | indirect | yes | existing DEVOPT-B / experiment-specific gates |
| micro-commit and repeated CI | yes | yes | logical finalization; later PR-B |
| authority rejects but does not guide construction | yes | yes | V1 reuse plus deterministic preflight |
| missing dependency profile | observed historically | yes | later PR-B or separate maintenance |
| incomplete gate-value telemetry | yes | yes | later PR-B |

## Prioritized backlog

| Priority | Item | Status | Authorization boundary |
|---|---|---|---|
| P0 | PR-A Pilot Spec compiler and preflight | implementing | current claim |
| P0 | historical replay and V1 compatibility tests | implementing | current claim |
| P1 | operational tiered CI with final full gate | proposed | separate PR-B review required |
| P1 | gate timing and unique-blocker telemetry | proposed | separate PR-B review required |
| P2 | evidence-based gate consolidation | observed only | requires real telemetry and explicit approval |

## Remaining uncertainties

- whether PR #49's historical branch can serve as a complete immutable replay source or requires a faithful repository fixture;
- the measured reduction in wall-clock time after PR-A;
- whether any current full gate is genuinely redundant;
- whether transport-level commit batching should remain an operator convention or later receive a dedicated GitHub publication adapter.

## Status

```text
Incident documented
PR-A plan linked
PR-A implementation authorized and started
No workflow default changed
No scientific state changed
PR-B not authorized
```
