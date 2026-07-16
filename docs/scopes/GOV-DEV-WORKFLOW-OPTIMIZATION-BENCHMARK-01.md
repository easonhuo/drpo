# GOV-DEV-WORKFLOW-OPTIMIZATION-BENCHMARK-01 Scope

## Identity

- Claim: `GOV-DEV-WORKFLOW-OPTIMIZATION-BENCHMARK-01`
- Base: `main@7d0ecfbee3b9e44bbad97fb806c8806b604f75f6`
- Branch: `dev/gov-dev-workflow-optimization-benchmark-01`
- Phase: documentation and validation-design freeze
- User authorization: explicit instruction to document the workflow-optimization project, preserve its history, define a reusable validation framework, and complete review before implementation

## Objective

Create a durable documentation system that lets later sessions understand:

- recurring development and integration problems;
- what fastpath, V1, authority, RunSpec, and result delivery already solve;
- what remains unsolved;
- why a possible orchestration layer is a coordination hypothesis rather than a replacement system;
- how workflow changes must be tested through historical paired replay;
- how time reduction, per-case non-regression, complexity, and rollback determine adoption.

## Authorized files

- `docs/development_workflow_optimization/README.md`
- `docs/development_workflow_optimization/REPLAY_BENCHMARK_PROTOCOL.md`
- `docs/scopes/GOV-DEV-WORKFLOW-OPTIMIZATION-BENCHMARK-01.md`

A later reviewed revision may add links from an existing workflow-document index. It may not change `AGENTS.md` or a default executable policy in this phase without a separate explicit review of the exact diff.

## Explicit exclusions

This claim does not authorize:

- an orchestration script or library;
- a telemetry database, dashboard, service, or blocking gate;
- changes to fastpath, V1, handoff authority, registry schema, RunSpec, lane runner, result delivery, or evidence locator;
- GitHub workflow changes;
- automatic push, PR creation, approval, or merge;
- changes to scientific code, configs, data, seeds, thresholds, budgets, horizons, result statuses, or priorities;
- execution of E7/E8 experiments;
- retrospective claims that current fastpath has already reduced time;
- merge without separate explicit user approval.

## Design constraints

The documentation must:

1. preserve the distinction between the existing safety/correctness kernel and a possible coordination layer;
2. prohibit a second authority or state machine;
3. require evidence before code;
4. use representative historical cases and paired A/B replay;
5. separate historical task identity from the shared benchmark toolchain;
6. require at least two opposite-order measured pairs per case;
7. require correctness equivalence before efficiency analysis;
8. require no material per-case regression for universal adoption;
9. report every repetition, every case, mean, and median;
10. separate historical real wall time from controlled replay time;
11. include implementation and maintenance cost in ROI;
12. use minimal benchmark artifacts and hard stop conditions to prevent framework expansion.

## Frozen first-iteration thresholds

A future candidate may be recommended as the universal default only when:

- all correctness and safety checks pass;
- no in-scope case is slower by more than `max(60 seconds, 5% of baseline median controlled-replay time)`;
- median case-level controlled wall time decreases by at least 30%;
- mean controlled wall time also decreases;
- median active operation time decreases by at least 30%;
- command count decreases by at least 60%;
- manual intermediate-file copies and temporary workflow/PR use fall to zero;
- production code stays within the 250–450 line target, with mandatory redesign review above 500 lines;
- no V1 core, authority, registry schema, scientific code, or merge automation change is required.

These thresholds and the case inventory must be frozen before candidate results are inspected.

## Benchmark boundary

The primary controlled comparison uses:

- historical task inputs and expected outcomes;
- one benchmark toolchain SHA shared by A and B;
- Arm A as the accepted manual/component path;
- Arm B as the candidate optimization;
- the same environment, cache policy, source blobs, gates, and terminal-state definition.

Historical PR and Actions time is reported separately as operational context. It cannot replace the same-environment A/B baseline.

## Minimal evidence set

A benchmark iteration may retain only:

- `CASE_INVENTORY.yaml`;
- `RAW_RESULTS.jsonl`;
- `PAIRED_COMPARISON.json`;
- `DECISION.md`.

Large raw transaction directories may remain persistent-local with hashes and locators. This scope does not authorize a new long-lived measurement platform.

## Review plan and record

### Review 1 — document hierarchy

**Pass after revision.** The project hub is subordinate to repository authorities and links to, rather than replacing, component contracts and incident records.

### Review 2 — architecture and ownership

**Pass.** The possible orchestration layer is only a hypothesis and may coordinate existing owners without taking scientific, authority, transaction, execution, publication, or merge responsibility.

### Review 3 — benchmark validity

**Pass after revision.** Historical task identity and benchmark toolchain are separated. Each case requires at least two measured pairs in opposite order, shared cache policy, event-based active-time capture, and predeclared invalidation rules.

### Review 4 — ROI and no-regression

**Pass after revision.** The protocol reports every repetition and case, uses per-arm medians, blocks material per-case regressions, accounts for implementation/maintenance cost, and reports break-even task count.

### Review 5 — anti-framework and future-session continuity

**Pass after revision.** Artifacts were reduced to a minimal four-file set; no database, dashboard, service, or blocking CI is authorized. The project history, current state, next action, and stop conditions are recoverable without chat history.

## Completion condition

This documentation phase is complete only when:

- all three authorized files exist on one reviewable branch;
- the diff contains no code, workflow, registry, handoff, or scientific changes;
- the five review passes remain recorded;
- exact-head repository checks applicable to documentation pass;
- no implementation begins before a separate user decision.

## Remaining uncertainty

The historical replay inventory is not yet frozen. GitHub wall-clock and Actions timing are available for many cases, while local V1 stage timing is incomplete. The benchmark protocol supplements missing timing through repeated same-environment replay rather than reconstructing unsupported historical active time.
