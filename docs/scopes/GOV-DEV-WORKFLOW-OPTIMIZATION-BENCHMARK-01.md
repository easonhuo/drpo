# GOV-DEV-WORKFLOW-OPTIMIZATION-BENCHMARK-01 Scope

## Identity

- Claim: `GOV-DEV-WORKFLOW-OPTIMIZATION-BENCHMARK-01`
- Base: `main@7d0ecfbee3b9e44bbad97fb806c8806b604f75f6`
- Branch: `dev/gov-dev-workflow-optimization-benchmark-01`
- Phase: staged disposable-prototype implementation
- Initial user authorization: document the workflow-optimization project, preserve its history, define a reusable validation framework, and complete review before implementation
- Implementation authorization: explicit instruction on 2026-07-16 to continue development and iteration on this branch, use modular steps with independent goals, implement the project, self-test it, and evaluate the result

## Objective

Build and evaluate the smallest repository-local composition path that can reduce workflow time and active operation effort while preserving all existing correctness, provenance, authority, gate, and scientific-safety behavior.

The implementation must let later sessions understand:

- recurring development and integration problems;
- what fastpath, V1, authority, RunSpec, and result delivery already solve;
- what remains unsolved;
- why the candidate is a coordination layer rather than a replacement system;
- how workflow changes are tested through historical paired replay;
- how time reduction, per-case non-regression, complexity, and rollback determine adoption.

## Authorized paths

Documentation and benchmark records:

- `docs/development_workflow_optimization/README.md`
- `docs/development_workflow_optimization/REPLAY_BENCHMARK_PROTOCOL.md`
- `docs/development_workflow_optimization/IMPLEMENTATION_PLAN.md`
- `docs/development_workflow_optimization/benchmarks/**`
- `docs/scopes/GOV-DEV-WORKFLOW-OPTIMIZATION-BENCHMARK-01.md`

Disposable prototype and tests:

- `src/drpo/workflow_replay/**`
- `scripts/run_workflow_replay.py`
- `tests/test_workflow_replay_*.py`
- `tests/fixtures/workflow_replay/**`

A step may use fewer files. Adding another production path requires a scope review before the file is created. `AGENTS.md`, default workflow policy, and existing component cores remain outside this implementation scope.

## Explicit exclusions

This claim does not authorize:

- changes to `scripts/prepare_dev_pilot_registration.py`, V1 core, handoff authority, registry schema, RunSpec, lane runner, result delivery, or evidence locator;
- a telemetry database, dashboard, service, daemon, queue, scheduler, or blocking gate;
- GitHub workflow changes;
- automatic push, PR creation, approval, or merge;
- changes to scientific code, configs, data, seeds, thresholds, budgets, horizons, result statuses, or priorities;
- execution of E7/E8 scientific experiments;
- task-specific E7/E8 branches in general workflow code;
- retrospective claims that the current fastpath has already reduced time;
- default-route activation or merge without separate explicit user approval.

## Architecture constraints

The prototype must:

1. preserve the distinction between the existing safety/correctness kernel and the candidate coordination layer;
2. invoke existing owners rather than duplicate validators, authority semantics, transaction state, or gates;
3. use independently testable modules for case validation, execution recording, correctness equivalence, and paired comparison;
4. use representative historical cases and paired A/B replay;
5. separate historical task identity from the shared benchmark toolchain;
6. require at least two opposite-order measured pairs per case;
7. require correctness equivalence before efficiency analysis;
8. require no material per-case regression for universal adoption;
9. report every repetition, every case, mean, and median;
10. separate historical real wall time from controlled replay time;
11. include implementation and maintenance cost in ROI;
12. use minimal artifacts and hard stop conditions to prevent framework expansion.

## Frozen complexity budget

- target production code: 300–450 lines;
- mandatory redesign review above 500 production lines;
- no new third-party dependency;
- no new domain state beyond append-only raw events and derived, rebuildable summaries;
- test and fixture volume may exceed production code because equivalence and failure evidence are primary;
- a simpler solution must replace a larger prototype when both solve the same measured problem.

## Frozen first-iteration thresholds

A candidate may be recommended as the universal default only when:

- all correctness and safety checks pass;
- no in-scope case is slower by more than `max(60 seconds, 5% of baseline median controlled-replay time)`;
- median case-level controlled wall time decreases by at least 30%;
- mean controlled wall time also decreases;
- median active operation time decreases by at least 30%;
- command count decreases by at least 60%;
- manual intermediate-file copies and temporary workflow/PR use fall to zero;
- production code stays within the complexity budget;
- no existing component core, scientific code, or merge behavior changes.

These thresholds and the case inventory must be frozen before candidate results are inspected.

## Staged execution

### Step 0 — scope and architecture

Authorized now. Freeze module boundaries, paths, line budget, stop conditions, and per-step review gates. No behavior-changing code.

### Step 1 — case model and static validation

Authorized after Step 0 review. Implement strict case-manifest validation and positive/negative fixtures. No subprocess or repository mutation.

### Step 2 — execution recorder and dry-run adapter

Requires Step 1 focused tests. Record deterministic command plans, monotonic timing, active-operation events, environment identity, and diagnostics using fixture execution first.

### Step 3 — correctness-equivalence verifier

Requires Step 2 review. Detect tree, semantic, authority, gate, terminal-state, and provenance mismatch before any efficiency output.

### Step 4 — thin candidate orchestrator

Requires Steps 1–3 to pass. Compose the accepted fastpath/V1 stages without modifying or duplicating their core behavior.

### Step 5 — failure injection and end-to-end fixture replay

Cover main drift, before-image mismatch, gate failure, interruption, output conflict, and mutated reviewer approval. Full repository tests and Ruff are required at exact head.

### Step 6 — historical inventory and baseline

Freeze 6–10 representative cases before candidate replay. Preserve incomplete-artifact limitations and forbid post-hoc case removal.

### Step 7 — paired candidate replay and decision

Run identical case inputs in opposite orders, preserve every repetition, inspect every regression, and record `ADOPT`, `NARROW`, `REDESIGN`, or `REJECT`.

The full module and gate plan is in `docs/development_workflow_optimization/IMPLEMENTATION_PLAN.md`.

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

Large raw transaction directories may remain persistent-local with hashes and locators. This scope does not authorize a long-lived measurement platform.

## Per-step review requirements

Every step must record:

- one explicit goal;
- one frozen changed-path scope;
- focused tests tied to the goal;
- changed-path and line-count review;
- confirmation that accepted behavior did not regress;
- a stop decision before the next step.

A later step may not hide an earlier defect with task-specific special cases.

## Stop and redesign conditions

Stop implementation when:

- production code exceeds 500 lines;
- a new service, dependency, database, scheduler, queue, dashboard, or state machine appears necessary;
- an existing component core would need modification;
- automatic publication or merge appears necessary to demonstrate benefit;
- correctness equivalence cannot be evaluated independently of timing;
- historical fixture reconstruction becomes larger than the optimization itself;
- a smaller command wrapper or documentation change solves the same measured problem.

## Current completion condition

The implementation iteration is complete only when:

- all staged module and failure-injection tests pass at the exact head;
- the historical case inventory was frozen before candidate results;
- both arms use identical inputs, environment, gates, and expected outcomes;
- correctness equivalence is audited before time results;
- every case and repetition is reported;
- complexity and break-even cost are reviewed;
- the candidate receives one evidence-backed decision;
- no default-route change or merge occurs without separate explicit user approval.

## Remaining uncertainty

The historical replay inventory is not yet frozen. GitHub wall-clock and Actions timing are available for many cases, while local V1 stage timing and some historical intermediate inputs are incomplete. The benchmark supplements missing timing through repeated same-environment replay; incomplete cases must be classified rather than silently reconstructed beyond available evidence.