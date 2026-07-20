# ReplayAB R2 Gap Audit

**Project:** DRPO A/B Replay Engine  
**Claim:** `GOV-DEV-WORKFLOW-OPTIMIZATION-BENCHMARK-01`  
**Audit stage:** `R2 gap audit`  
**Base:** `main@bb637503e1289f24f7a28e587f50665afb20e0de`  
**Branch:** `dev/replayab-r2-gap-audit-01`  
**Decision:** `DEFER`  
**Scientific impact:** none

## 1. Executive decision

Do not implement ReplayAB R2 now.

R2 remains a legitimate future capability: it would let ReplayAB accept two different
implementations when both independently satisfy one frozen semantic acceptance contract.
However, the current repository has no approved near-term consumer that requires that
capability.

The first measured workflow candidate, `Candidate 01 -- V1 One-Click Integration`, is a
deterministic orchestration candidate. Its accepted comparison modes are
`exact_artifact` and `failure_boundary`, and its minimum confidence target is C1. The
merged R1 capability already addresses that requirement.

Starting R2 now would therefore add evaluator schema, tolerance semantics, hidden-evaluator
identity, per-arm acceptance states, and a new calibration bank before there is a concrete
decision that needs them. That would violate the roadmap's fresh ROI review and
anti-framework constraints.

`DEFER` means:

- R2 is not rejected as a concept;
- no R2 behavior code is authorized;
- ReplayAB remains C1 for the implemented deterministic modes;
- no C2 language is allowed;
- Candidate 01 may proceed through a separate C1 evaluation-readiness decision without R2.

## 2. Objects inspected

This audit inspected the current-main versions of:

- repository-root `AGENTS.md`;
- `docs/handoff.md`, including Section 0 and current workflow-governance overrides;
- `experiments/registry.yaml`;
- `docs/development_workflow_optimization/REPLAYAB_ENGINE_ROADMAP.md`;
- `docs/development_workflow_optimization/V1_SUBMISSION_WORKFLOW_OPTIMIZATION_PLAN.md`;
- `docs/development_workflow_optimization/R1_IMPLEMENTATION_CONTRACT.md`;
- `src/drpo/workflow_replay/model.py`;
- `src/drpo/workflow_replay/compare.py`;
- `src/drpo/workflow_replay/evidence.py`;
- the R1 calibration inventory, expected verdicts, fixtures, and focused tests.

The audit found no current implementation of `semantic_acceptance` or an
`AcceptanceContract` in ReplayAB Core.

## 3. What R1 now provides

Merged R1 provides a bounded C1 deterministic ruler for `exact_artifact` and
`failure_boundary` cases. It can:

- validate a schema-v2 deterministic case contract;
- ingest content-addressed real repository evidence;
- bind case, run, arm, pair, repetition, order, backend, evidence, timing, and report identities;
- verify paths, parent symlinks, regular-file status, byte sizes, and SHA-256 digests;
- validate event-journal identity, sequence, start, and terminal state;
- derive invalid execution and workspace mutation rather than trusting a caller flag;
- compare each arm to a frozen deterministic contract before pair comparison;
- reject both arms when both are identically wrong;
- block efficiency release after invalid or incorrect outcomes;
- retain the schema-v1 path unchanged.

R1 intentionally does not answer whether two different trees are both correct.

## 4. What R2 would have to add

The roadmap defines R2 as semantic acceptance rather than approximate tree equality.
A legitimate R2 implementation would need all of the following.

### 4.1 Frozen acceptance contract

A semantic case must freeze, before arm results are inspected:

- evaluator identity and digest;
- required behaviors;
- forbidden behaviors and regressions;
- completeness requirements;
- protected variables, paths, and APIs;
- numerical, performance, or compatibility tolerances when applicable;
- required diagnostics;
- acceptance and rejection rules;
- treatment/evaluator visibility policy;
- supported and unsupported claims.

A semantic contract cannot be generated or edited after seeing an arm merely to make the
arm pass.

### 4.2 Independent per-arm acceptance

R2 must separate at least three questions:

1. Is Arm A independently accepted?
2. Is Arm B independently accepted?
3. Are the accepted arms comparable for the declared efficiency or complexity analysis?

`A tree != B tree` must not be a failure when both arms pass the frozen contract.
Likewise, `A tree == B tree` must not be success when both violate it.

### 4.3 Semantic evaluator evidence

ReplayAB would need to ingest a frozen evaluator report that proves:

- which evaluator ran;
- which subject artifact it evaluated;
- the evaluator input and result digests;
- mandatory-check outcomes;
- forbidden-regression outcomes;
- tolerance measurements and thresholds;
- completeness and provenance conclusions;
- evaluator terminal and diagnostic class.

R2 Core should judge authenticated evaluator evidence. It should not reimplement pytest,
GitHub Actions, project-specific scientific validators, or a general command platform.

### 4.4 Semantic pair report

The report vocabulary must distinguish:

- accepted and accepted, with different implementations;
- accepted and rejected;
- rejected and accepted;
- both rejected for the same reason;
- both rejected for different reasons;
- execution invalidity;
- evaluator invalidity;
- pair incomparability despite two accepted arms.

The existing deterministic `equivalent: bool` is insufficient by itself for these states.

### 4.5 Independent calibration truth

C2 requires a pre-reviewed calibration bank containing at least:

- two intentionally different correct implementations;
- one correct and one incorrect implementation in both arm orders;
- two implementations sharing the same hidden defect;
- a public-test pass that fails frozen hidden acceptance;
- a correct large change rejected by an over-broad gate, exposing false rejection;
- a small change that deletes required behavior;
- evaluator identity mutation;
- tolerance boundary cases immediately inside and outside the frozen threshold.

No such frozen bank currently exists.

## 5. Current gap matrix

| R2 requirement | Current status | Consequence |
|---|---|---|
| `AcceptanceContract` | absent | semantic obligations are not machine-frozen |
| semantic comparison mode | rejected by current R1 validator | R1 correctly fails closed instead of pretending support |
| independent per-arm acceptance | absent | current pair logic centers on deterministic contract equality |
| mandatory/forbidden behavior schema | absent | no general semantic verdict can be justified |
| tolerance representation | absent | boundary and comparison semantics are undefined |
| evaluator-result artifact schema | absent | evaluator claims cannot be independently authenticated |
| hidden evaluator visibility policy | conceptual only | C2 independence and leakage claims are unsupported |
| semantic pair-state vocabulary | absent | accepted-but-different cannot be represented cleanly |
| different-correct calibration cases | absent | no evidence that false rejection is controlled |
| R1 regression protection | present | any future R2 must leave deterministic modes unchanged |
| immediate approved consumer | absent | implementation ROI is currently unproven |

## 6. Consumer and ROI audit

### 6.1 Candidate 01

Candidate 01 explicitly requires:

- `exact_artifact` for deterministic successful cases;
- `failure_boundary` for safe-stop cases;
- C1 deterministic evidence across its predeclared bank.

It does not require two different correct implementations. Candidate 01 therefore does not
justify R2 implementation.

### 6.2 Scientific workflows

C-U1, D-U1, Hopper, and Countdown have their own scientific validators, formal execution
channel, terminal audits, and artifact protocols. ReplayAB R2 must not absorb those domain
rules. No registered scientific experiment currently names ReplayAB semantic acceptance as
a launch prerequisite.

### 6.3 Other workflow candidates

No second workflow candidate with a frozen semantic-acceptance need has been registered.
Building R2 for hypothetical future candidates would create infrastructure before demand.

### 6.4 Cost signal

The merged R1 deterministic evidence module is already at the frozen 400-line yellow-zone
boundary. Adding R2 directly to it would trigger redesign pressure. Creating a new Python
module would also require explicit human approval of the exact path and responsibility under
the current new-Python-file gate.

These constraints do not make R2 impossible. They mean R2 should begin only when a real use
case can justify the added contract, evaluator, tests, fixtures, and maintenance burden.

## 7. Main risks of implementing R2 prematurely

### 7.1 Rebuilding test and CI infrastructure

A broad implementation could turn ReplayAB into a second pytest/CI runner. The roadmap
forbids that. Existing component owners must continue to execute their own validators.

### 7.2 Self-authored oracle bias

If the same development step creates the candidate outputs, evaluator, and expected
verdicts, passing tests may only prove internal consistency. The evaluator and calibration
truth must be frozen independently before results.

### 7.3 Ambiguous tolerance policy

A generic tolerance field is unsafe without domain semantics. Absolute, relative, per-item,
aggregate, stochastic, and numerical-stability tolerances are not interchangeable.

### 7.4 Hidden-evaluator leakage overclaim

An evaluator digest proves identity, not that a generator was unable to inspect the hidden
evaluator. Strong leakage claims require an execution backend and isolation evidence beyond
R2's repository-level judgment layer.

### 7.5 Pair-state conflation

Treating two accepted implementations as `equivalent` can hide material implementation,
complexity, safety, or performance differences. R2 needs accepted/comparable vocabulary,
not a weakened equality check.

### 7.6 Regression to the deterministic ruler

Changing existing R1 contract or result semantics to accommodate R2 could weaken the exact
modes that are already calibrated. R2 must be additive and mode-explicit.

## 8. Future reopening conditions

R2 implementation may be reconsidered only when all of these conditions are met:

1. a named candidate or task class requires different correct implementations to be accepted;
2. exact-artifact comparison would cause a demonstrated false rejection for that decision;
3. at least two intentionally different correct artifacts and relevant wrong artifacts exist;
4. an independent acceptance contract and expected verdicts can be frozen before evaluation;
5. evaluator ownership and leakage limits are explicit;
6. a fresh code-size, file-count, runtime, adapter, and maintenance budget is approved;
7. the implementation can reuse R1 identities and evidence locators without modifying R1 verdict semantics;
8. any proposed new Python path receives explicit human approval before creation.

Until then, R2 remains `deferred_not_started` and ReplayAB remains C1.

## 9. Minimum legitimate future implementation boundary

This section is a future boundary, not current authorization.

A reopened R2 should begin with evidence ingestion, not a universal execution framework:

- one frozen semantic acceptance-contract schema;
- one authenticated evaluator-result artifact schema;
- one per-arm acceptance result;
- one pair comparator that separates acceptance from comparability;
- one pre-reviewed calibration bank;
- unchanged R1 exact-artifact and failure-boundary behavior.

The first implementation must not include:

- a generic plugin registry;
- a service, database, queue, scheduler, or dashboard;
- a replacement for pytest or GitHub Actions;
- a hidden-test runner owned by ReplayAB Core;
- coding-agent execution or Regeneration Runner;
- stochastic probability claims;
- Candidate 01 adoption logic;
- scientific-domain rules.

## 10. Decision

The R2 gap-audit verdict is **`DEFER`**.

Repository facts supporting the decision:

- R1/C1 deterministic evidence hardening is merged;
- current R1 intentionally rejects semantic mode;
- Candidate 01 requires only deterministic C1 modes;
- no approved near-term consumer requires semantic acceptance;
- no frozen different-correct calibration bank exists;
- R1 is already at its reviewed yellow-zone size boundary;
- anti-framework and new-Python-file gates make speculative expansion especially costly.

This is not evidence that semantic acceptance is unimportant. It is evidence that developing
it now is not the highest-value next action.

## 11. Next permitted ReplayAB action

The next higher-value action is a bounded **Candidate 01 C1 evaluation-readiness audit**:

- inspect the existing candidate implementation against its frozen plan;
- determine whether its 6--10-case deterministic bank can be frozen from current artifacts;
- identify missing adapters or evidence without modifying ReplayAB Core;
- decide `GO / NARROW / REDESIGN / STOP` before running Candidate 01 evaluation.

That action remains separate from this R2 audit and requires its own explicit authorization.

## 12. Authorization boundary

This audit changes documentation only. It does not authorize:

- R2 behavior implementation;
- a new Python file;
- semantic evaluator execution;
- Candidate 01 evaluation or adoption;
- R3 or later roadmap work;
- handoff or registry changes;
- workflow or default-route activation;
- scientific execution or result-status changes;
- merge of this audit without explicit user approval.
