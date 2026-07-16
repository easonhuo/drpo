# GOV-DEV-WORKFLOW-OPTIMIZATION-BENCHMARK-01 Scope

## Identity

- Claim: `GOV-DEV-WORKFLOW-OPTIMIZATION-BENCHMARK-01`
- Base: `main@7d0ecfbee3b9e44bbad97fb806c8806b604f75f6`
- Branch: `dev/gov-dev-workflow-optimization-benchmark-01`
- Phase: documentation and validation-design freeze
- User authorization: explicit instruction to document the workflow-optimization project, preserve its history, define a reusable validation framework, and complete review before implementation

## Objective

Create a durable documentation system that lets later sessions understand:

- the recurring development/integration problems;
- what the existing fastpath, V1, authority, RunSpec, and result-delivery components already solve;
- what remains unsolved;
- why a possible orchestration layer is a separate coordination hypothesis rather than a replacement system;
- how any workflow optimization must be tested through historical paired replay;
- how time reduction, no-regression, complexity, and rollback determine adoption.

## Authorized files

- `docs/development_workflow_optimization/README.md`
- `docs/development_workflow_optimization/REPLAY_BENCHMARK_PROTOCOL.md`
- `docs/scopes/GOV-DEV-WORKFLOW-OPTIMIZATION-BENCHMARK-01.md`

A later reviewed revision may add links from an existing workflow-document index. It may not change `AGENTS.md` or a default executable policy in this phase without a separate explicit review of the exact diff.

## Explicit exclusions

This claim does not authorize:

- an orchestration script or library;
- machine telemetry implementation;
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
2. prohibit replacing existing component owners with a second authority or state machine;
3. require evidence before code;
4. use representative historical cases and paired A/B replay;
5. require correctness equivalence before efficiency analysis;
6. require no material per-case regression for universal adoption;
7. report mean, median, and every individual case;
8. separate historical real wall time from controlled replay time;
9. include implementation and maintenance cost in ROI;
10. define hard stop conditions that prevent framework expansion.

## Frozen first-iteration thresholds

A future candidate may be recommended as the universal default only when:

- all correctness and safety checks pass;
- no in-scope case is slower by more than `max(60 seconds, 5% of baseline)`;
- median controlled wall time decreases by at least 30%;
- mean controlled wall time also decreases;
- median active operation time decreases by at least 30%;
- command count decreases by at least 60%;
- manual intermediate-file copies and temporary workflow/PR use fall to zero;
- production code stays within the 250–450 line target, with mandatory redesign review above 500 lines;
- no V1 core, authority, registry schema, scientific code, or merge automation change is required.

These thresholds must be frozen before candidate results are inspected.

## Review plan

### Review 1 — document hierarchy

Confirm that the new project hub is subordinate to the repository authorities and links rather than duplicates existing component and incident documents.

### Review 2 — architecture and ownership

Confirm that the proposed orchestration hypothesis only coordinates existing owners and cannot become a parallel state machine, authority, scientific planner, or publication system.

### Review 3 — benchmark validity

Check paired inputs, representative sampling, cache/order control, timing boundaries, replayability classes, correctness equivalence, and anti-cherry-picking rules.

### Review 4 — ROI and no-regression

Check time-reduction formulas, per-case visibility, material-regression tolerance, complexity accounting, break-even analysis, and adoption thresholds.

### Review 5 — scope and future-session continuity

Check that a later session can recover the full problem history, current state, next permitted action, and stop conditions without relying on chat history.

## Completion condition

This documentation phase is complete only when:

- all three authorized files exist on one reviewable branch;
- the diff contains no code, workflow, registry, handoff, or scientific changes;
- the five review passes are recorded in the PR description or review record;
- exact-head repository checks applicable to documentation pass;
- no implementation begins before a separate user decision.

## Remaining uncertainty

The historical replay inventory is not yet frozen. Historical GitHub wall-clock and Actions timing are available for many cases, while local V1 stage timing is incomplete. The benchmark protocol resolves this by preserving historical context separately and measuring baseline/candidate under the same controlled replay environment.
