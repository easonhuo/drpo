# Development Workflow Optimization Implementation Plan

**Claim:** `GOV-DEV-WORKFLOW-OPTIMIZATION-BENCHMARK-01`  
**Branch:** `dev/gov-dev-workflow-optimization-benchmark-01`  
**Base:** `main@7d0ecfbee3b9e44bbad97fb806c8806b604f75f6`  
**Status:** staged disposable-prototype implementation authorized on the development branch; no default-route change

## 1. Engineering-size estimate

This is a medium repository-tool project. It is not a service, platform, daemon, workflow engine, or replacement authority.

Estimated active engineering effort:

| Workstream | Estimate |
|---|---:|
| Scope, architecture, and case-contract freeze | 1–2 h |
| Core replay/orchestration implementation | 6–8 h |
| Unit, integration, and failure-injection tests | 4–6 h |
| Historical case reconstruction and repeated replay | 4–8 h |
| Review, simplification, and documentation sync | 2–3 h |
| **Total active effort** | **15–24 h** |

Unattended test and replay execution may add several hours of elapsed machine time. Missing historical local artifacts can increase case-reconstruction effort but must not justify expanding the core architecture.

Frozen size budget:

- production code target: 300–450 lines;
- mandatory redesign review above 500 production lines;
- test and fixture code may be larger because correctness equivalence and failure replay are the main evidence burden;
- no new third-party dependency;
- no database, dashboard, service, daemon, queue, scheduler, or blocking CI.

## 2. Architecture

The implementation is split into independently testable modules. No module may assume ownership of another component's domain logic.

### Module A — case contract and fixture loader

Responsibility:

- parse and strictly validate one immutable replay-case manifest;
- bind historical task identity, shared benchmark toolchain identity, expected outputs, gates, cache policy, and replayability class;
- reject unknown keys, mutable paths, post-hoc exclusions, or ambiguous terminal states.

Does not:

- execute V1;
- interpret scientific results;
- mutate repository state.

### Module B — arm execution adapter

Responsibility:

- run Arm A or Arm B using the same frozen inputs and environment;
- invoke existing fastpath and V1 commands rather than reimplementing them;
- record monotonic stage timestamps, exit status, terminal state, commands, file-placement events, and diagnostics;
- support isolated workspaces and deterministic reruns.

Does not:

- auto-repair failures;
- weaken gates;
- push, create PRs, approve, or merge.

### Module C — correctness-equivalence verifier

Responsibility:

- compare successful outcomes by tree, changed paths, file modes, registry semantic hash, handoff/materialization state, delta semantics, authority result, selected gates, gate conclusions, terminal state, and provenance;
- compare failed outcomes by safety boundary, diagnostics, lack of partial mutation, and recovery class;
- block all efficiency claims until equivalence passes.

Does not:

- redefine authority semantics;
- accept approximate scientific equivalence.

### Module D — timing and operation recorder

Responsibility:

- emit append-only raw events for wall time and active operation time;
- distinguish operator/model-active intervals from unattended command execution;
- record cache policy, order, repetition, invalidation reason, and environment fingerprint;
- preserve every raw repetition.

Does not:

- create a telemetry service;
- persist a database;
- hide outliers.

### Module E — paired comparison and decision

Responsibility:

- aggregate per-arm medians across opposite-order repetitions;
- trigger a third repetition when variation exceeds the frozen threshold;
- calculate per-case and aggregate time reduction, active-time reduction, command reduction, and complexity cost;
- emit `ADOPT`, `NARROW`, `REDESIGN`, or `REJECT` only from frozen rules.

Does not:

- change the default route;
- treat a tie as an improvement;
- exclude a poor case after results are known.

## 3. Staged development plan

### Step 0 — implementation scope and skeleton

Goal:

- record implementation authorization;
- freeze module boundaries, file layout, line budget, and stage gates;
- create no behavior-changing code.

Exit gate:

- documents are internally consistent;
- no scientific, authority, V1-core, registry-schema, workflow, or merge behavior changes.

### Step 1 — case model and static validation

Goal:

- implement Module A;
- define a minimal case-manifest schema;
- create representative positive and negative fixture manifests.

Exit gate:

- focused unit tests pass;
- unknown keys, unsafe paths, invalid SHAs, missing expected outcomes, ambiguous scope, and post-hoc exclusions fail closed;
- no subprocess or repository mutation occurs.

### Step 2 — execution recorder and dry-run adapter

Goal:

- implement Modules B and D in dry-run/fixture mode first;
- produce deterministic command plans and append-only raw event records;
- prove Arm A and Arm B receive identical frozen inputs.

Exit gate:

- no existing component core is modified;
- command planning is deterministic and idempotent;
- interrupted fixture runs remain diagnosable and do not claim terminal success.

### Step 3 — correctness-equivalence verifier

Goal:

- implement Module C;
- support both successful and fail-closed replay outcomes;
- verify protected semantic equality before timing analysis.

Exit gate:

- injected tree, registry, delta, authority, gate, terminal-state, and provenance mismatches are detected;
- efficiency output is impossible when correctness equivalence fails.

### Step 4 — thin candidate orchestrator

Goal:

- implement the smallest Arm B composition path;
- connect existing preparation and V1 stages without duplicating their validators or state;
- remove manual intermediate placement for covered cases.

Exit gate:

- production code remains at or below 450 lines unless redesign is explicitly approved;
- no new third-party dependency;
- no V1 core, authority, registry schema, scientific code, GitHub workflow, publication, or merge change;
- focused integration tests pass.

### Step 5 — failure-injection and end-to-end fixture replay

Goal:

- cover main drift, before-image mismatch, gate failure, interruption, output conflict, and mutated reviewer approval;
- run opposite-order paired fixture replays;
- validate raw-result and decision artifacts.

Exit gate:

- zero correctness/safety regression;
- every failure remains fail closed;
- no partial authority/scientific mutation;
- full repository tests and Ruff pass at exact head.

### Step 6 — historical replay inventory and baseline

Goal:

- freeze 6–10 representative cases before candidate results;
- reconstruct immutable inputs and classify replayability;
- run accepted Arm A baselines under the shared toolchain.

Exit gate:

- case inventory is fixed;
- no poor or difficult case is removed post hoc;
- raw repetitions, invalidations, and missing-artifact limitations are preserved.

### Step 7 — candidate replay, evaluation, and review

Goal:

- run Arm B on the identical case inventory;
- perform repeated `A→B` and `B→A` comparisons;
- evaluate all frozen adoption and no-regression thresholds;
- conduct architecture, correctness, ROI, and anti-framework review.

Exit gate:

- one evidence-backed decision is recorded;
- a failing candidate is narrowed, redesigned, or rejected rather than rationalized;
- no merge or default-route activation occurs without a new explicit user approval.

## 4. Expected repository shape

The preferred implementation shape is:

```text
src/drpo/workflow_replay/
  model.py
  execute.py
  compare.py
scripts/run_workflow_replay.py
tests/
  test_workflow_replay_model.py
  test_workflow_replay_execute.py
  test_workflow_replay_compare.py
  fixtures/workflow_replay/
```

This is a target, not permission to fill every file with a separate framework. Modules may be collapsed when that reduces total code and state while preserving test isolation.

## 5. Per-step review rule

Every step must have:

1. one explicit goal;
2. one frozen changed-path scope;
3. focused tests tied to that goal;
4. a changed-path and line-count review;
5. confirmation that no prior accepted behavior regressed;
6. a stop decision before the next step.

A later step may not silently repair an earlier step by adding cross-module special cases. The earlier module must instead be corrected and retested.

## 6. Stop and redesign conditions

Stop implementation and review the architecture when any of the following occurs:

- production code exceeds 500 lines;
- a new dependency, service, database, queue, scheduler, state machine, or dashboard appears necessary;
- V1 core or handoff authority would need modification;
- task-specific E7/E8 branches appear in general workflow code;
- the candidate needs automatic push, PR creation, approval, or merge to show benefit;
- correctness equivalence cannot be stated independently of timing;
- historical fixture reconstruction becomes larger than the workflow optimization itself;
- a simpler documentation or command-wrapper change solves the same measured problem.

## 7. Current next action

The next action is Step 0 completion followed by Step 1 only. Step 2 does not begin until the case contract and focused tests are reviewed. No scientific experiment execution is part of this plan.