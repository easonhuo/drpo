# DRPO A/B Replay Engine Roadmap

**Project:** DRPO A/B Replay Engine  
**Claim:** `GOV-DEV-WORKFLOW-OPTIMIZATION-BENCHMARK-01`  
**Document role:** durable roadmap for ReplayAB Core only  
**Branch at roadmap freeze:** `dev/gov-dev-workflow-optimization-benchmark-01`  
**Repository state reviewed:** `main@d042a60e6e665fc7f8761e97d41fa0a621f78b87`, branch head `5ce421fae810b0ac30cf8f8e14f5672308077952`  
**Status:** documentation-only planning record; no new implementation stage, benchmark run, default-route change, merge, or scientific execution is authorized

## 1. Executive decision

ReplayAB and the first workflow optimization must be maintained as separate objects.

- **ReplayAB Core is the measuring instrument.** It freezes cases, records runs, evaluates correctness and safety, compares paired outcomes, and reports confidence and efficiency.
- **A workflow optimization is a measured candidate.** The current thin V1 orchestration path is the first candidate, not part of ReplayAB's general definition.
- **An execution runner is a backend.** A future Regeneration Runner may create isolated coding-agent runs, but it must not own ReplayAB's acceptance rules or decision logic.

The original Stage 0--7 plan mixed these responsibilities:

- original Stages 1--3 largely created reusable ReplayAB Core primitives;
- original Stage 4 created Candidate 01, the V1 one-click integration path;
- original Stages 5--7 planned candidate-specific fault injection, historical baseline replay, and paired evaluation.

That history remains valid provenance and must not be destructively rewritten. From this roadmap onward, ReplayAB Core has its own capability roadmap, calibration gates, and definition of done.

## 2. Authority and document relationships

This roadmap is subordinate to:

1. repository-root `AGENTS.md`;
2. `docs/handoff.md`, the unique research master;
3. `experiments/registry.yaml` for scientific experiment state;
4. accepted contracts of V1, handoff authority, GitHub routing, formal execution, and other existing component owners.

This roadmap:

- does not become a second research master;
- does not alter scientific variables, seeds, thresholds, budgets, horizons, statuses, or experiment order;
- does not authorize implementation merely because a capability appears in a later roadmap stage;
- does not authorize a service, daemon, database, dashboard, queue, scheduler, automatic merge, or default-route activation.

Related documents retain distinct roles:

- `README.md`: history, motivation, existing workflow owners, and the optimization project's durable hub;
- `REPLAY_BENCHMARK_PROTOCOL.md`: the first exact-artifact historical replay protocol;
- `IMPLEMENTATION_PLAN.md`: non-destructive history of the combined ReplayAB-plus-Candidate-01 development sequence;
- `REPLAYAB_ENGINE_ROADMAP.md`: future capability and calibration plan for ReplayAB Core;
- a future candidate-specific plan: lifecycle, risks, and adoption evidence for Candidate 01 or another optimization.

Where the old combined plan and this roadmap use different decomposition, the old plan remains historical provenance while this document controls future ReplayAB Core planning. No existing commit, report, or stage result is retroactively renamed.

## 3. Problem statement

A repository workflow is partly deterministic and partly stochastic.

Deterministic segments include:

- input materialization;
- file placement;
- command order;
- gate execution;
- authority normalization;
- terminal transaction state;
- final repository artifacts.

Stochastic segments include:

- coding-agent implementation choices;
- model mistakes and repair behavior;
- human or model recovery decisions;
- timing variation and external load;
- which failure mode appears on a particular task.

A useful A/B engine must therefore answer more than "did two final trees match?" It must distinguish:

1. whether both arms received a valid and comparable task;
2. whether each arm was executed under the declared treatment and controls;
3. whether each result is independently correct and safe;
4. whether the two results are comparable even when their code differs;
5. whether the observed difference is stable across tasks and repetitions;
6. whether the measurement process itself introduced bias, leakage, or excessive cost.

The current implementation is strongest on deterministic exact-artifact checks and incomplete on stochastic generation, semantic acceptance, full trajectories, unknown-regression detection, and multi-task statistical inference.

## 4. Stable terminology and ownership

### 4.1 ReplayAB Core

The reusable measurement layer. It may own:

- immutable case contracts;
- execution-plan identity;
- append-only run evidence;
- normalized run artifacts;
- correctness and safety evaluators;
- paired comparison and confidence grading;
- derived decision reports.

It must not own:

- V1 logic;
- handoff or registry authority;
- coding-agent implementation behavior;
- scientific experiment design;
- GitHub publication or merge;
- candidate-specific domain rules hidden inside the core.

### 4.2 Candidate optimization

One proposed workflow change, such as:

- V1 one-click integration;
- targeted versus full test selection;
- a review gate;
- package generation versus direct GitHub integration;
- a different recovery or registration path.

Candidate-specific commands, adapters, failure hypotheses, and adoption thresholds must remain outside generic core logic.

### 4.3 Execution backend

A mechanism that produces one arm's real run evidence under a frozen contract. Planned backend classes are:

- historical-artifact adapter;
- deterministic local-command adapter;
- isolated coding-agent or Regeneration Runner adapter.

A backend executes. ReplayAB judges.

### 4.4 Evaluator

An independent mechanism that transforms a run into correctness, safety, completeness, and diagnostic evidence. The evaluator must be frozen before candidate results and unavailable to a generator when hidden evaluation is required.

### 4.5 Case bank

A reviewed collection of frozen cases covering task classes, known incidents, boundary conditions, and adversarial or fault-injection scenarios. It is not a cherry-picked success set.

### 4.6 Run Artifact

The normalized evidence object for one arm, one case, and one repetition. It includes final outcome and the trajectory required to interpret that outcome.

## 5. Current implementation inventory

The current branch contains the following reusable primitives.

### 5.1 Implemented core primitives

- `src/drpo/workflow_replay/model.py`
  - strict immutable manifest validation;
  - frozen task, benchmark toolchain, expected terminal state, paths, hashes, gates, environment, and cache policy;
  - fail-closed validation for unknown keys, unsafe paths, malformed hashes, and ambiguous outcomes.

- `src/drpo/workflow_replay/execute.py`
  - deterministic Arm A and Arm B command plans;
  - plan identity hashing;
  - append-only fixture event journals;
  - separation of child-command time from ReplayAB self-overhead;
  - interruption and nonzero-exit terminal recording.

- `src/drpo/workflow_replay/compare.py`
  - strict `OutcomeSnapshot` validation;
  - comparison against the frozen manifest and against the paired arm;
  - terminal state, safety boundary, paths, modes, hashes, authority, gates, provenance, diagnostics, partial mutation, and recovery class;
  - efficiency release only after strict equivalence.

### 5.2 Candidate-specific implementation

- `src/drpo/workflow_replay/orchestrate.py`
- `scripts/run_workflow_replay.py`

The current orchestration path composes the preparation adapter and V1 stages for Candidate 01. It is useful code, but its V1-specific composition is not proof that ReplayAB Core is complete or general.

### 5.3 Existing evidence

The branch includes matched historical code-bloat repair pilot documents. They show that historical replay can expose a gate that rejects both bad and correct outcomes, and can verify a repaired rule on a frozen matched case.

Those pilots were assembled from frozen historical evidence. They are not proof that the current Core automatically ingested or executed historical artifacts end to end.

They do not establish:

- a randomized live same-model A/B result;
- a probability that a coding agent will repair itself;
- general performance across task classes;
- complete isolation of treatment and evaluator;
- universal validity of the current exact-artifact comparator.

## 6. What ReplayAB can currently measure

### 6.1 Strong current use cases

The current core is appropriate when:

- both arms should receive the same immutable input identity;
- command plans can be frozen explicitly;
- terminal state is `READY`, `BLOCKED`, or `STALE`;
- success requires exact paths, modes, protected hashes, gates, authority, and provenance;
- failure requires the same safety boundary, diagnostics, no partial mutation, and equivalent recovery class;
- efficiency is meaningful only after exact correctness equivalence.

Representative examples:

- deterministic registration materialization;
- wrapper replacement where final repository semantics must be unchanged;
- V1 lifecycle composition with the same underlying owners;
- stale-input or gate-failure boundary replay;
- comparison of existing historical artifacts after explicit normalization into frozen outcome evidence.

### 6.2 Conditional current use cases

ReplayAB can analyze matched historical coding outcomes if the conclusion is narrow:

- it can compare final code size, churn, paths, gates, and known acceptance evidence;
- it can diagnose whether a proposed gate accepts or rejects those frozen outcomes;
- it cannot infer how often a fresh agent would generate either outcome.

### 6.3 Unsupported or weak current use cases

The current core cannot yet support strong claims about:

- two different but semantically correct implementations;
- live stochastic coding-agent behavior;
- first-attempt quality and repair trajectories;
- treatment effects on model behavior;
- newly introduced failure classes outside the frozen evaluator;
- multi-task error rates and uncertainty;
- evaluator leakage or cross-arm information contamination;
- token, message, tool-call, and model-version controls;
- automated opposite-order repeated paired runs;
- general use outside DRPO without project-specific adapters.

## 7. Validity model: when is the ruler trustworthy?

ReplayAB evidence is trustworthy only when four validity layers pass.

### 7.1 Case validity

The case must bind:

- exact base and source identities;
- one frozen public task packet;
- treatment definition;
- allowed and forbidden changes;
- acceptance contract;
- evaluator identity;
- environment and cache policy;
- budgets and repetition policy;
- predeclared exclusions.

### 7.2 Execution validity

The run must prove:

- each arm used its declared path;
- both arms received identical controls except the treatment;
- workspaces were isolated where required;
- no cross-arm communication occurred;
- no hidden evaluator or treatment assignment leaked;
- failures, timeouts, and partial runs were retained;
- first-attempt and final states were not conflated.

### 7.3 Evaluator validity

The evaluator must:

- be frozen before outcomes;
- be independent of the implementation author where hidden evaluation is required;
- accept multiple correct implementations when the task permits them;
- reject both arms when both violate the task contract;
- detect partial mutation and unauthorized behavior;
- expose false rejection rather than treating rejection alone as safety success.

### 7.4 Statistical validity

For stochastic workflows:

- one paired run is an observation, not a probability estimate;
- task selection must precede results;
- order and cache effects must be balanced;
- all failed and timed-out trajectories remain in the denominator;
- per-task and aggregate results must both be visible;
- uncertainty and sample-size limits must accompany adoption claims.

## 8. Comparison modes

ReplayAB must explicitly select one comparison mode per case.

### 8.1 `exact_artifact`

Use when the accepted output is deterministic.

Required checks may include:

- tree or protected semantic hashes;
- changed paths and file modes;
- authority and gate results;
- terminal state;
- provenance identity.

Different artifact outcomes are a correctness failure unless the case contract explicitly allows metadata-only variation.

### 8.2 `semantic_acceptance`

Use when multiple implementations can be correct.

Each arm is evaluated independently against one frozen acceptance contract. The two trees need not match. Pair comparison begins only after each arm is classified independently.

Required checks may include:

- hidden or independent functional tests;
- mandatory behaviors and forbidden regressions;
- task completeness;
- protected variables and paths;
- API, performance, or numerical tolerances;
- safety and provenance constraints.

### 8.3 `failure_boundary`

Use when the correct outcome is to stop.

The evaluator checks:

- correct blocking condition;
- no unauthorized mutation;
- complete diagnostics;
- correct recovery class;
- no false `READY` claim.

### 8.4 `stochastic_generation`

Use when the treatment is expected to change agent behavior.

This mode requires an isolated execution backend and repeated live runs. It evaluates outcome distributions, not one fixed tree.

## 9. Target architecture

```text
Frozen Case Contract
        |
        v
Replay Controller ----------------------+
        |                               |
        v                               v
Execution Backend A              Execution Backend B
        |                               |
        v                               v
Append-only Run Evidence         Append-only Run Evidence
        |                               |
        +---------------+---------------+
                        v
                Run Artifact Normalizer
                        v
                 Independent Evaluator
                        v
             Paired Comparator + Aggregator
                        v
          Evidence Report + Confidence Grade
```

Core responsibilities remain small:

1. validate frozen contracts;
2. record or ingest complete evidence;
3. normalize backend outputs;
4. invoke frozen evaluators;
5. compare and aggregate;
6. emit a reviewable report.

Git, pytest, CI, V1, authority, and coding-agent logic remain external owners.

## 10. Required normalized evidence objects

### 10.1 Case Contract

Future schema evolution must support:

- comparison mode;
- public input identity;
- treatment definition;
- control variables;
- acceptance contract locator and digest;
- evaluator locator and digest;
- backend requirements;
- resource and attempt budgets;
- isolation and leakage policy;
- repetition and order policy;
- predeclared metrics and decision rules.

### 10.2 Run Artifact

One run artifact should minimally contain:

- case, arm, repetition, and run IDs;
- base, task packet, toolchain, evaluator, environment, and backend identities;
- treatment identity without leaking hidden assignment to the generator;
- complete event and command locators;
- first complete attempt commit or artifact;
- every repair attempt and feedback class;
- final commit or artifact;
- terminal state and failure class;
- evaluator results and diagnostics;
- changed paths, modes, hashes, and provenance;
- wall, child, self-overhead, active-operation, token, message, tool-call, and CI cost when available;
- timeout, interruption, invalidation, and partial-mutation evidence.

### 10.3 Pair Report

A pair report should separate:

- case validity;
- execution validity;
- per-arm acceptance;
- pair comparability;
- efficiency and complexity;
- uncertainty and exclusions;
- supported and unsupported claims.

## 11. Calibration suite

ReplayAB must be tested against cases with pre-reviewed expected verdicts.

Minimum calibration matrix:

| Calibration case | Expected behavior |
|---|---|
| A and B are byte-identical and correct | accept both; equivalent |
| A and B differ but are independently correct | accept both in semantic mode |
| A correct, B incorrect | reject B; no efficiency claim |
| A incorrect, B correct | reject A; report candidate improvement without pretending the pair is exact-equivalent |
| A and B fail in the same unauthorized way | reject both; do not label equivalent success |
| Public tests pass but hidden acceptance fails one arm | reject the failing arm |
| Correct large change is rejected by an over-broad gate | record false rejection |
| Small change deletes required behavior | reject despite low churn |
| Mid-run interruption after partial write | detect partial mutation and retain trajectory |
| Main drift or before-image mismatch | stop at the frozen safety boundary |
| Evaluator or approval mutation | detect identity mismatch |
| Generator can read treatment or hidden evaluator | mark execution invalid |
| Same case run A->B and B->A | expose order or cache sensitivity |
| Worker timeout or unrepaired run | retain in denominator and report terminal class |

Calibration reports must compare ReplayAB verdicts with an independent reviewed expected verdict. A green unit test count alone is not calibration evidence.

## 12. Confidence grades

Every ReplayAB conclusion should carry one grade.

- **C0 -- schema or fixture only:** static validation and unit fixtures; no real task.
- **C1 -- deterministic replay:** real frozen task or artifact under exact-artifact or failure-boundary evaluation.
- **C2 -- independently accepted semantic replay:** different implementations evaluated against a frozen independent acceptance contract.
- **C3 -- isolated live paired run:** real isolated workers, hidden treatment/evaluator controls, complete trajectories, at least one paired task.
- **C4 -- repeated multi-task paired evidence:** predeclared case bank, balanced order, retained failures, uncertainty reporting, and independent review.
- **C5 -- post-adoption observation:** C4 evidence plus monitored real production uses and rollback evidence.

A conclusion may not use a higher grade's language when only a lower grade was achieved.

## 13. Roadmap

### R0 -- Identity split and planning freeze

Goal:

- separate ReplayAB Core from Candidate 01 and execution backends;
- preserve the old combined Stage history;
- freeze terminology, comparison modes, validity layers, confidence grades, and anti-framework constraints.

Deliverables:

- this roadmap;
- a current implementation-to-capability map;
- explicit unsupported claims;
- candidate-specific planning separated before further candidate evaluation.

Exit gate:

- no contradiction with `AGENTS.md`, handoff authority, registry, or existing component ownership;
- no code or default-route change;
- no false claim that the current engine is production-ready or live-A/B validated.

### R1 -- Exact-artifact core hardening

Goal:

- finish and calibrate deterministic exact-artifact and failure-boundary replay as an independent core capability.

Required work:

- validate real command or artifact ingestion rather than fixture callbacks alone;
- standardize run identity and evidence locators;
- automate paired opposite-order repetitions for deterministic backends;
- execute known-good and known-bad calibration cases;
- preserve strict efficiency release after correctness.

Exit gate:

- all R1 calibration verdicts match independent review;
- interrupted and partial runs cannot claim success;
- no candidate-specific V1 rule enters generic core;
- ReplayAB overhead remains within frozen runtime guardrails.

### R2 -- Semantic acceptance

Goal:

- support different but correct implementations without requiring identical trees.

Required work:

- add a frozen `AcceptanceContract` concept;
- separate per-arm acceptance from pair equivalence;
- support mandatory behaviors, forbidden regressions, tolerances, and hidden evaluator identity;
- preserve exact-artifact mode unchanged for deterministic cases.

Exit gate:

- calibration accepts two intentionally different correct implementations;
- calibration rejects both-same-wrong outcomes;
- evaluator identity is frozen before results;
- no efficiency result is released for an unaccepted arm.

### R3 -- Complete trajectory and Run Artifact

Goal:

- make final outcomes interpretable by retaining how each result was produced.

Required work:

- record first complete attempt separately from repairs;
- retain feedback classes, attempts, failures, timeouts, and interruptions;
- normalize final and intermediate artifacts across backends;
- distinguish environment invalidation from candidate failure;
- include active operation and resource accounting where observable.

Exit gate:

- every final verdict is traceable to immutable raw evidence;
- failed and unrepaired runs remain visible;
- no last-success-only reporting;
- trajectory ingestion does not require a database or service.

### R4 -- Fault injection and unknown-regression coverage

Goal:

- test more than historical known failures and improve the chance of detecting candidate-introduced regressions.

Required work:

- systematic fault matrix for path escape, symlinks, conflicts, stale identities, malformed child output, gate false pass, gate false rejection, interruption, duplicate execution, partial state, evaluator mutation, and leakage;
- unauthorized changed-path and behavior audit;
- property or metamorphic checks where exact outputs are inappropriate;
- independent calibration verdicts.

Exit gate:

- all predeclared faults receive the expected terminal and diagnostic class;
- false rejection is measured, not hidden;
- newly introduced behavior outside the allowed contract is visible;
- coverage claims remain limited to exercised fault dimensions.

### R5 -- Pluggable execution backends

Goal:

- let ReplayAB consume real outcomes without coupling core judgment to one runner.

Backends:

1. historical-artifact adapter;
2. deterministic local-command adapter;
3. isolated Regeneration Runner adapter.

Regeneration Runner requirements:

- independent workspaces or containers;
- identical public task packet and base;
- neutral worker identities;
- hidden assignment and evaluator;
- equal model, tool, time, token, and attempt budgets as far as externally controllable;
- no cross-arm reading;
- first-attempt freeze;
- controlled feedback routing;
- complete event, message, tool, commit, and failure evidence;
- no evaluator or decision logic inside the worker.

Exit gate:

- backend outputs normalize to the same Run Artifact contract;
- core evaluator and pair comparator are backend-independent;
- backend-specific failures cannot be mistaken for candidate correctness failures;
- external model-build or server-side randomness limits are stated explicitly.

### R6 -- Stochastic paired evaluation

Goal:

- measure how a treatment changes the distribution of coding-agent outcomes.

Required design:

- predeclared diverse case bank;
- same public task and controls in both arms;
- balanced A/B execution order;
- repeated live runs sufficient for the declared claim;
- complete failure and timeout retention;
- task-level and aggregate reporting;
- independent reviewer audit.

Primary metrics may include:

- final task success and completeness;
- unsafe-pass and false-rejection rates;
- first-attempt success;
- repair-at-1 and repair-at-2;
- regression and unauthorized-change classes;
- production churn and duplication;
- wall, active-operation, token, tool, and CI cost;
- operator interventions.

Exit gate:

- all validity layers pass;
- uncertainty and task heterogeneity are reported;
- no single-case result is generalized into a probability claim;
- no claim of exact equivalence to an internal platform A/B is made.

### R7 -- Adoption and maintenance

Goal:

- determine whether ReplayAB itself is useful enough to retain and whether a measured candidate should be adopted.

ReplayAB retention evidence:

- at least two materially different optimization projects use the same core without copying core internals;
- candidate-specific adapter code remains thin;
- verdicts agree with independent review on the calibration bank;
- measurement cost is materially below the decision value it provides;
- defects, false positives, and missed regressions are recorded.

Candidate adoption remains a separate decision using that candidate's frozen thresholds. ReplayAB passing R7 does not automatically approve any candidate.

## 14. Metrics for the ruler itself

ReplayAB must report its own quality, not only candidate outcomes.

Required ruler metrics:

- verdict agreement with independent reviewed calibration truth;
- unsafe-pass rate;
- false-rejection rate;
- invalid-run detection rate;
- partial-trajectory retention rate;
- evaluator/treatment leakage detection rate;
- reproducibility of deterministic verdicts;
- adapter code and maintenance burden per new project;
- self-overhead and total benchmark cost;
- proportion of conclusions limited by missing evidence;
- number and severity of post-decision reversals.

A lower false-rejection rate may not be purchased by weakening unsafe-pass protection, and vice versa. Both must be visible.

## 15. Anti-framework constraints

ReplayAB must remain a thin repository-local measurement engine.

It must not:

- become a general workflow platform;
- maintain a database or always-on service;
- create a dashboard before repeated use proves that a static report is insufficient;
- reimplement Git, pytest, GitHub Actions, V1, authority, or coding-agent runtimes;
- encode E7-, E8-, Hopper-, Countdown-, or candidate-specific science in the core;
- create automatic push, PR, approval, merge, or scientific execution;
- add retries or repair behavior without a case-level frozen protocol;
- hide failed runs, outliers, or inconvenient cases;
- weaken validation to satisfy a line budget.

Complexity is evaluated by:

- production lines and files;
- dependencies;
- persistent state;
- adapter size;
- execution overhead;
- review burden;
- break-even number of future decisions.

Every roadmap stage requires a fresh size and ROI review before implementation authorization.

## 16. Candidate 01 relationship

The current thin V1 orchestration implementation is reclassified prospectively as:

> **Candidate 01 -- V1 One-Click Integration**

It aims to reduce manual command sequencing and file placement while preserving existing component ownership and gates.

This prospective classification does not alter its historical Stage-4 checkpoint or code. Before Candidate 01 can be recommended:

1. ReplayAB's required comparison mode and confidence target must be declared;
2. candidate-specific failure hypotheses and adoption metrics must be frozen in a separate plan;
3. exact-artifact and failure-boundary evaluation must use calibrated ReplayAB Core;
4. any claim about changed coding-agent behavior requires the isolated live backend and stochastic evidence;
5. no further candidate stage starts automatically from the old combined plan.

## 17. Current status and next permitted action

At this roadmap freeze:

- R0 remains `active`: this roadmap is committed and reviewed, while the separate Candidate 01 plan and any required cross-document pointers remain pending before the R0 exit gate;
- existing Stage 1--3 code is an implementation starting point for R1, not automatic proof that R1 passed;
- existing Stage 4 code remains Candidate 01;
- historical repair pilots remain narrow matched observations;
- no live Regeneration Runner result exists in ReplayAB Core;
- no multi-task stochastic adoption evidence exists;
- no default workflow change is authorized.

The next implementation action, if separately approved, should be a bounded R1 gap audit before new code:

1. map current tests and code to R1 exit gates;
2. identify the minimum missing deterministic ingestion, pairing, and calibration work;
3. estimate code and runtime cost;
4. decide `GO`, `NARROW`, `REDESIGN`, or `STOP` before implementation.

R2--R7 remain unstarted until their preceding gates and explicit authorizations are satisfied.

## 18. Definition of done

ReplayAB Core is mature for controlled external coding-workflow A/B only when all of the following hold:

1. deterministic exact-artifact and failure-boundary calibration passes;
2. semantic acceptance correctly handles different valid implementations;
3. complete first-attempt, repair, failure, timeout, and final trajectories are retained;
4. fault injection covers declared known and boundary risks;
5. at least one isolated live worker backend preserves treatment and evaluator separation;
6. a predeclared multi-task paired study reports uncertainty and all failures;
7. ruler-quality metrics are independently reviewed;
8. two distinct optimization projects reuse the core with thin adapters;
9. measured value exceeds implementation and maintenance cost;
10. rollback and evidence-preservation behavior is proven.

Even then, ReplayAB should be described as a controlled external repository-workflow A/B engine. It must not claim full equivalence to a platform-internal model A/B system whose server-side model build, routing, sampling state, or hidden infrastructure cannot be externally controlled.

## 19. Maintenance and review discipline

For every roadmap update:

1. preserve previous statements and record supersession non-destructively;
2. state which evidence or requirement changed;
3. keep Core, candidate, backend, and evaluator responsibilities separate;
4. audit supported versus unsupported claims;
5. audit whether a proposed capability requires real worker evidence;
6. audit false-pass and false-rejection risks;
7. audit code, runtime, adapter, and maintenance budgets;
8. verify no scientific or default-route authority changed;
9. update status only at a reviewed gate;
10. keep chat as discussion, never as the sole durable authority.

## 20. Roadmap freeze review record

The first roadmap freeze received four post-write review passes.

### Pass 1 -- Architecture and ownership

Result: `PASS`.

- ReplayAB Core, Candidate 01, execution backend, evaluator, and case bank have separate responsibilities.
- Regeneration Runner is treated as a backend rather than the definition of ReplayAB.
- V1, authority, GitHub, and scientific owners remain external.

### Pass 2 -- Evidence and claim boundary

Result: `PASS after correction`.

- The historical code-bloat pilots are now explicitly described as frozen matched evidence, not proof of automatic end-to-end historical ingestion.
- Deterministic, semantic, and stochastic claims are separated.
- The roadmap does not claim production readiness, live randomized validation, or equivalence to a platform-internal A/B system.

### Pass 3 -- Engineering feasibility and anti-framework control

Result: `PASS`.

- The roadmap favors adapters and normalized artifacts over reimplementation.
- Services, databases, dashboards, schedulers, automatic publication, and scientific execution remain unauthorized.
- Every implementation stage requires a fresh size, runtime, and ROI review.

### Pass 4 -- Status and cross-document consistency

Result: `PASS after correction`.

- R0 remains active rather than being incorrectly marked complete solely because this file was committed.
- Existing Stage 1--3 code is treated as an R1 starting point, not accepted R1 evidence.
- Existing Stage 4 remains Candidate 01, and no old Stage automatically authorizes the next action.

No code, scientific configuration, registry entry, handoff authority state, default route, PR readiness, or merge state was changed by this roadmap review.
