# Candidate 01 — V1 One-Click Integration Plan

**Project:** DRPO development-workflow optimization  
**Candidate:** `Candidate 01 -- V1 One-Click Integration`  
**Parent claim:** `GOV-DEV-WORKFLOW-OPTIMIZATION-BENCHMARK-01`  
**Development branch:** `dev/gov-dev-workflow-optimization-benchmark-01`  
**Historical implementation checkpoint:** `4acdc5855cfe3d110d466166174dac0bf2d93e5a`  
**Roadmap split base:** `5aba3eaf3fe705bc5306e2d187622577add38e4d`  
**Status:** implemented prototype; candidate evaluation not started; no default-route change, merge, or adoption authorized

## 1. Document role

This document is the durable candidate-specific plan for the first workflow optimization measured by the DRPO A/B Replay Engine.

It separates Candidate 01 from ReplayAB Core:

- ReplayAB Core owns case contracts, evidence normalization, correctness and safety evaluation, paired comparison, confidence grading, and reports.
- Candidate 01 owns only the proposed V1 command-and-file-placement composition being evaluated.
- V1, pilot-registration preparation, handoff authority, registry authority, GitHub publication, and scientific execution remain independent owners.
- A future execution backend may run the candidate, but it may not change the candidate definition or ReplayAB verdict rules.

The original combined `IMPLEMENTATION_PLAN.md` remains immutable historical provenance for Stages 0--4. This plan governs Candidate 01 prospectively and does not retroactively rename or invalidate historical checkpoints.

## 2. Problem being addressed

The accepted code-first pilot-registration route is safe but operator-heavy. A reviewed implementation must currently move through:

```text
pilot-registration preparation
-> repository-overlay placement
-> V1 plan
-> V1 prepare
-> transaction-input placement when applicable
-> V1 normalize
-> V1 gate
-> V1 finalize
-> local READY or safe non-READY terminal state
```

The underlying components already own correctness, provenance, authority, and safety. The remaining candidate problem is coordination friction:

- multiple explicit commands;
- manual placement of deterministic intermediate files;
- stage-order knowledge held by the operator;
- repeated inspection to determine which command comes next;
- avoidable opportunity for wrong-path, missing-input, or wrong-order errors;
- active waiting and cross-session handoff burden.

Candidate 01 does not claim that these are the only development-workflow problems, and it cannot receive credit for scientific-design bugs, implementation bugs inside a scientific method, missing data or credentials, hardware failures, weak method performance, task-performance collapse, support or variance-boundary events, or NaN/Inf numerical failure.

## 3. Candidate hypothesis

The candidate hypothesis is:

> A thin one-click composition of the already accepted preparation and V1 owners can reduce operator-active work and command count while preserving the exact accepted repository, authority, provenance, gate, and terminal-state behavior of the existing path.

The candidate may coordinate existing owners. It may not replace or reinterpret them.

## 4. Compared paths

### 4.1 Arm A — accepted multi-step baseline

Arm A runs the accepted path without Candidate 01:

1. invoke `scripts/prepare_dev_pilot_registration.py`;
2. place the exact prepared repository overlay;
3. invoke V1 `plan`;
4. invoke V1 write-path `prepare`;
5. place the exact transaction inputs when required;
6. invoke V1 `normalize`;
7. invoke the existing finalize gate;
8. invoke V1 `finalize`.

No artificial delay, redundant command, deliberate mistake, or weakened recovery behavior may be added to make Arm A look worse.

### 4.2 Arm B — Candidate 01

Arm B invokes the same accepted owners exactly once and in the same required order through the existing thin candidate entry point:

```text
scripts/run_workflow_replay.py
src/drpo/workflow_replay/orchestrate.py
```

The candidate automates only:

- orchestration of the accepted commands;
- placement of already generated deterministic repository-overlay files;
- placement of already reviewed transaction inputs when required;
- fail-closed propagation of child-command outcomes;
- a derived status and timing summary from the run evidence.

It does not own any child component's validation or state transition.

## 5. Frozen ownership boundaries

Candidate 01 must not:

- implement a second V1 transaction;
- reproduce handoff or registry authority logic;
- replace the pilot-registration preparation adapter;
- skip, weaken, reinterpret, or duplicate a required gate;
- run full repository scans merely to create telemetry;
- retry, auto-repair, silently rebase, or select a recovery action unless a later case-specific protocol explicitly authorizes the same behavior for both arms;
- create or modify scientific variables, seeds, thresholds, budgets, horizons, experiment responsibilities, statuses, or execution order;
- push, create or approve a PR, merge, publish, or activate a default route;
- become a service, daemon, database, dashboard, queue, scheduler, or general workflow engine;
- embed E7-, E8-, Hopper-, Countdown-, or other scientific-domain branches.

## 6. Current implementation identity

The historical Stage-4 implementation is preserved at:

- final Stage-4 checkpoint: `4acdc5855cfe3d110d466166174dac0bf2d93e5a`;
- implementation checkpoint identified in the Stage-4 record: `f0d7ceee103970bd2c12b0a32b7de3b457a47378`.

The candidate-specific implementation consists primarily of:

- `src/drpo/workflow_replay/orchestrate.py`;
- the Candidate 01 command surface in `scripts/run_workflow_replay.py`;
- Candidate 01 integration tests and fixtures.

The reusable case, execution-recording, and comparison primitives are ReplayAB Core starting points, not candidate ownership.

The combined Stage-4 development record counted 767 production lines across Core plus candidate. That historical budget remains provenance. Future Candidate 01 work must report candidate-only churn separately from ReplayAB Core churn.

## 7. Required ReplayAB modes and confidence target

Candidate 01 currently requires two ReplayAB comparison modes:

### 7.1 `exact_artifact`

Use for successful deterministic cases in which both arms must produce the same protected repository and authority outcome.

Required equality includes, as applicable:

- final tree or protected semantic hashes;
- changed paths and file modes;
- handoff and registry materialization;
- authority result;
- selected gate plan and gate conclusions;
- terminal state;
- source-lock and provenance identities.

Metadata-only differences may be allowed only when declared before the run and excluded by a stable rule.

### 7.2 `failure_boundary`

Use when the correct behavior is to stop safely.

Required equivalence includes:

- the correct blocking condition;
- no false `READY`;
- no partial authority or scientific-state mutation;
- complete diagnostics;
- the same safe recovery class;
- no hidden retry or automatic repair.

### 7.3 Confidence target

The minimum evidence target for the current deterministic candidate claim is ReplayAB **C1 -- deterministic replay** across a predeclared case bank.

Candidate 01 may not claim that it changes coding-agent behavior, reduces model implementation errors, or improves stochastic repair probability from C1 evidence. Any such claim would require an isolated live backend and at least C3 evidence, and is outside the current candidate scope.

## 8. Predeclared candidate failure hypotheses

Candidate evaluation must attempt to falsify the candidate, not only confirm the happy path.

### 8.1 Ordering and invocation

- a child stage runs out of order;
- a required stage is skipped;
- a child stage runs more than once;
- the candidate continues after a nonzero child exit;
- the candidate reports `READY` before finalize completes;
- the candidate summary disagrees with the underlying V1 terminal state.

### 8.2 Input placement

- repository overlay is placed at the wrong lifecycle boundary;
- transaction inputs are placed too early, too late, or in the wrong location;
- optional transaction inputs are treated as mandatory or silently ignored;
- an existing conflicting file is overwritten;
- a partial placement remains after failure;
- a path escape or symlink changes the intended write target.

### 8.3 Identity and freshness

- prepared input identity does not match the frozen implementation or reviewer input;
- main or source identity drifts after preparation;
- a stale workspace is reused;
- an existing transaction belongs to another case or attempt;
- cached or previously green evidence is accepted after relevant input changes.

### 8.4 Diagnostics and recovery

- malformed or missing child output is interpreted as success;
- an interruption leaves an ambiguous terminal state;
- a failure loses the original child diagnostics;
- the candidate prescribes a different or less safe recovery class than Arm A;
- rerunning the entry point silently mutates or resumes an incompatible attempt.

### 8.5 Cost and complexity

- candidate self-overhead materially offsets operator savings;
- the candidate introduces duplicate scans, gates, or network work;
- active operation decreases but controlled end-to-end time materially regresses;
- candidate-specific branches or maintenance burden exceed the benefit;
- the wrapper becomes another state-machine owner.

Coverage claims must remain limited to the failure dimensions actually exercised.

## 9. Case-bank requirements

Candidate 01 evaluation requires a predeclared 6--10-case deterministic bank before candidate results are inspected.

The bank must cover at least:

- one code-only integration;
- two new experiment registrations;
- one replacement or protocol update;
- one result closure or evidence-locator update;
- one stale-main or failed-attempt recovery;
- one deterministic gate failure;
- both an E7-derived and an E8-derived workflow case;
- at least one case with optional transaction inputs;
- at least one case with an interrupted or conflicting intermediate state.

Cases may be complete, reconstructed, or explicitly partial according to `REPLAY_BENCHMARK_PROTOCOL.md`. Candidate inconvenience, slowness, or failure is not a valid reason to remove a frozen case.

## 10. Controlled execution requirements

For every case:

- Arm A and Arm B use the same historical task input;
- both use the same benchmark toolchain, validators, gates, dependencies, machine class, network policy, environment, and cache policy;
- workspaces are isolated;
- order is balanced with measured `A -> B` and `B -> A` pairs;
- all measured repetitions, failures, interruptions, and invalidations remain in raw evidence;
- a third repetition is triggered only by the predeclared variance or environmental invalidation rule;
- candidate failure or slowness is never discarded as measurement noise;
- wall time, child-command time, candidate self-overhead, and active operation time remain separate.

## 11. Correctness and safety gate

Efficiency evidence is blocked until both arms independently satisfy the case contract and pass pair comparability.

Zero tolerance applies to:

- correctness regression;
- safety regression;
- provenance or source-lock regression;
- authority regression;
- gate weakening or omission;
- false `READY`;
- unauthorized changed paths;
- partial mutation after a failed case.

If Arm B is faster but fails any of these conditions, the candidate is rejected or redesigned rather than adopted narrowly after the fact.

## 12. Efficiency and operation metrics

After correctness release, report for every case and in aggregate:

- controlled wall time;
- active operation time;
- underlying child-component time;
- candidate self-overhead;
- explicit commands initiated;
- manual file-placement actions;
- workspaces and attempts created;
- recovery decisions requiring interpretation;
- temporary workflows or PRs;
- full-path restarts;
- local and external gate executions;
- unique blocking errors;
- candidate implementation and test size;
- candidate-only maintenance burden.

Historical real PR time may be shown separately as operational context but may not be mixed with the controlled causal estimate.

## 13. Frozen adoption thresholds

The candidate inherits the existing first-iteration thresholds unless a new document changes them before any new candidate result is inspected.

Universal adoption requires:

- all cases pass correctness, safety, authority, and provenance evaluation;
- zero in-scope cases have a material wall-time regression greater than `max(60 seconds, 5% of the Arm-A median)`;
- median controlled wall-time reduction is at least 30%;
- mean controlled wall time also improves;
- median active-operation-time reduction is at least 30%;
- explicit command count falls by at least 60%;
- manual intermediate-file copies are zero;
- temporary workflows and temporary PRs are zero for covered cases;
- candidate complexity remains within its reviewed budget.

A difference inside the no-regression tolerance is a tie, not an improvement.

A narrow route is allowed only when its task-class boundary and routing rule were declared before results, are decided from immutable input metadata, preserve an explicit Arm-A fallback, and are simpler than the recurring work they save.

## 14. Decision outcomes

ReplayAB may support one of four candidate decisions:

- `ADOPT`: all universal-adoption gates pass;
- `NARROW`: a predeclared task class passes while the explicit baseline remains for excluded classes;
- `REDESIGN`: the hypothesis remains useful but candidate defects, overhead, or complexity require bounded redesign;
- `REJECT`: correctness, safety, ownership, or ROI evidence does not justify retention.

ReplayAB maturity and Candidate 01 adoption are separate decisions. A calibrated ruler does not approve the candidate automatically, and a useful candidate does not prove the ruler is generally mature.

## 15. Rollback and preservation

Before any future activation, rollback must be demonstrated as removal of the candidate route while leaving existing owners and the accepted Arm-A route intact.

Rollback must not delete:

- historical Stage-4 commits and reports;
- frozen case inventories;
- failed or interrupted replay evidence;
- paired comparisons;
- adoption or rejection decisions;
- post-adoption observations if adoption later occurs.

No state migration should be required because Candidate 01 may not own durable domain state.

## 16. Current state

At this plan freeze:

- the thin Candidate 01 prototype is implemented on the historical development branch;
- the candidate has not completed its frozen failure-injection matrix;
- the historical Arm-A inventory is not frozen under the separated plan;
- no calibrated R1 ReplayAB Core has yet produced the required C1 candidate evidence;
- no Candidate 01 adoption decision exists;
- no default route, PR readiness, merge, publication, or scientific execution is authorized;
- old Stage 5--7 labels do not automatically start future work.

## 17. Next permitted candidate action

Candidate 01 remains paused while ReplayAB completes the separately authorized Core roadmap gates.

After calibrated deterministic ReplayAB Core is available, Candidate 01 may proceed only after a fresh explicit authorization to:

1. freeze the candidate case inventory and evaluator identities;
2. execute candidate-specific failure-boundary cases;
3. execute balanced Arm-A and Arm-B deterministic replays;
4. review every mismatch, slowdown, and invalidation;
5. issue an evidence-backed `ADOPT`, `NARROW`, `REDESIGN`, or `REJECT` decision.

No coding-agent Regeneration Runner is required for the current deterministic orchestration claim. It becomes relevant only if a later candidate claim concerns stochastic agent generation or repair behavior.

## 18. Maintenance discipline

Every update to this candidate plan must:

- preserve earlier statements and record supersession non-destructively;
- identify the new evidence or requirement;
- keep candidate logic outside ReplayAB Core;
- preserve V1 and authority ownership;
- audit false-pass, false-rejection, partial-mutation, and recovery risks;
- report candidate-only code and runtime cost;
- state whether the update changes the deterministic claim or introduces a new stochastic claim;
- avoid changing scientific or default-route authority;
- require explicit approval before a new evaluation or activation phase.
