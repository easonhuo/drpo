# ReplayAB R1 Gap Audit

**Project:** DRPO A/B Replay Engine  
**Claim:** `GOV-DEV-WORKFLOW-OPTIMIZATION-BENCHMARK-01`  
**Audit stage:** `R1 gap audit`  
**Authoritative base:** `main@96f09f7fe00719392cb52e0534e79c7c7d1ab0cd`  
**Audit branch:** `dev/replayab-r1-gap-audit-01`  
**Scope:** documentation and code inspection only  
**Decision:** `NARROW`  
**Implementation status:** not authorized by this audit

## 1. Decision

The R1 implementation should proceed only as a bounded **artifact-first deterministic hardening slice**.

The current Core is a useful C0 prototype, not a calibrated C1 replay engine. Its strict manifest validator, deterministic plan construction, fixture event journal, exact-outcome comparator, and correctness-first efficiency brake should be retained. They do not need to be replaced.

R1 must not expand directly into a general runner, semantic evaluator, repair-trajectory system, Regeneration Runner, stochastic benchmark service, or Candidate 01 evaluation. The smallest defensible next implementation is:

1. strengthen the frozen contract for `exact_artifact` and `failure_boundary` cases;
2. introduce stable run identity and content-addressed evidence locators;
3. normalize real deterministic run artifacts rather than accepting only manually constructed snapshots;
4. create a balanced `A -> B` / `B -> A` deterministic schedule;
5. run a pre-reviewed calibration bank containing real frozen artifacts plus adversarial variants;
6. bind correctness, timing, and the final report to the same run and evidence identities.

This is a `NARROW`, not a full `GO`, because the roadmap's broad R1 wording could otherwise pull command execution, backend generalization, trajectory recording, and fault-platform work into one iteration. Those responsibilities belong to later roadmap stages or to a thin external adapter.

## 2. Authorization and non-goals

This audit does not authorize behavior code, benchmark execution, Candidate 01 evaluation, or default-route activation.

It does not authorize:

- modifications to V1, pilot-registration preparation, handoff authority, registry authority, GitHub workflows, or scientific code;
- `semantic_acceptance` implementation;
- complete first-attempt and repair trajectories;
- a generic backend plugin framework;
- Regeneration Runner or live coding-agent workers;
- stochastic A/B claims;
- databases, services, daemons, queues, schedulers, dashboards, or automatic publication;
- changes to scientific variables, seeds, thresholds, budgets, experiment roles, results, or execution order.

The current merged Candidate 01 files may remain in the repository as provenance. A future R1 diff must leave `orchestrate.py` and Candidate 01 behavior unchanged. “Pure Core branch” means a branch whose new diff is Core-only, not destructive removal of already merged historical candidate files.

## 3. Inspected authority and implementation

The audit inspected the following at the exact base commit:

- repository-root `AGENTS.md`;
- Section 0 of `docs/handoff.md`;
- `experiments/registry.yaml`;
- `REPLAYAB_R0_CLOSURE.md`;
- `REPLAYAB_ENGINE_ROADMAP.md`;
- `V1_SUBMISSION_WORKFLOW_OPTIMIZATION_PLAN.md`;
- the original scope and historical implementation plan;
- `src/drpo/workflow_replay/model.py`;
- `src/drpo/workflow_replay/execute.py`;
- `src/drpo/workflow_replay/compare.py`;
- `src/drpo/workflow_replay/__init__.py`;
- `scripts/run_workflow_replay.py`;
- the three focused Core test modules and fixtures;
- `GOV-CODE-PAIRED-REPAIR-01`, including `scripts/paired_repair_report.py` and its tests.

No scientific experiment was run. No benchmark case was executed. No result status changed.

## 4. Current evidence grade

The current Core remains **C0 -- schema or fixture only**.

It has strong focused unit evidence for:

- strict manifest parsing and fail-closed unknown-key rejection;
- immutable in-memory case contracts;
- deterministic single-arm and paired plan construction;
- duplicate-command rejection;
- append-only creation of one fixture event journal;
- nonzero child status blocking;
- interruption recording without a false success event;
- exact expected-path, protected-hash, gate, authority, terminal, and provenance comparisons when fully populated snapshots are supplied;
- rejection of two arms that are identically wrong relative to the frozen manifest;
- blocking efficiency payload release after a detected mismatch.

These tests do not constitute C1 because they do not ingest or execute a real frozen task or artifact end to end, do not establish independent calibration verdicts, and do not preserve a complete identity chain from case through run evidence to pair report.

## 5. R1 requirement-to-capability map

| R1 requirement | Current status | Evidence | Gap |
|---|---|---|---|
| Strict frozen case contract | `partial` | `model.py` schema v1 and negative tests | exact-artifact and failure-boundary semantics remain conflated and under-specified |
| Real command or artifact ingestion | `missing` | `run_fixture_plan` accepts an injected integer-returning callback | no strict real artifact loader; no generic real command result contract |
| Stable run identity | `partial` | case, arm, input, environment, cache, plan, and caller-supplied `run_id` are journaled | no pair ID, repetition, order position, backend, workspace, evaluator, or evidence-root identity |
| Evidence locators | `missing` | event path is supplied by caller | no typed locator, digest verification, relative-root policy, or cross-artifact binding |
| Deterministic plan identity | `partial` | `plan_sha256` is deterministic | plan omits base, toolchain, evaluator, backend, workspace, and repetition/order identity |
| Append-only run evidence | `partial` | new-file JSONL journal with sequence numbers and flush | no content digest, monotonic-order validation, terminal uniqueness validation, stdout/stderr locators, or sealing record |
| Interrupted run cannot claim success | `partial` | fixture interruption emits `run_interrupted` and re-raises | `OutcomeSnapshot` cannot represent `INTERRUPTED`; no normalized invalid-run verdict |
| Partial mutation cannot claim success | `missing` | comparator rejects `partial_mutation=True` only when a caller supplies it | no before/after workspace identity or independent mutation derivation |
| Exact-artifact comparison | `partial` | expected paths/hashes and pair fields are compared | schema does not require final tree identity or expected file modes; both arms can share an under-specified wrong outcome |
| Failure-boundary comparison | `partial` | expected boundary, non-empty diagnostics, recovery class, and no declared partial mutation are checked | expected diagnostic class, failing gate, authority state, and before/after no-mutation proof are not frozen |
| Outcome normalization | `missing` | tests construct `OutcomeSnapshot` directly | no strict loader from immutable run output to `OutcomeSnapshot` |
| Opposite-order paired repetitions | `missing` | `build_paired_plans` creates one A plan and one B plan | no pair schedule, repetition IDs, order balance, or raw-repetition retention |
| Independent calibration | `missing` | focused unit tests assert expected behavior | no pre-reviewed expected-verdict artifact and no real frozen case |
| Correctness-first efficiency release | `implemented with binding gap` | `release_efficiency_payload` blocks non-equivalent reports | arbitrary payload is not bound to run IDs, evidence digests, or report identity |
| Core/candidate separation | `pass for current three modules` | model, execute, and compare contain no V1 stage rules | current CLI is Candidate 01-only; R1 needs a separate thin Core surface if a CLI is required |
| Runtime guardrails | `partial` | manifest and planning unit timing thresholds exist | no real-artifact ingestion, normalization, scheduling, or report overhead measurement |

## 6. Module findings

### 6.1 `model.py`

Strengths:

- exact key allowlists reject silent schema drift;
- SHA, digest, case ID, task class, terminal state, path, cache, and replayability syntax are validated;
- path traversal and manifest symlink use are rejected;
- READY and non-READY outcomes have different basic shape requirements;
- the accepted object is recursively frozen.

R1 gaps:

1. Schema v1 has no explicit comparison mode. The field named `expected_final_tree_or_semantic_hashes` mixes exact and semantic concepts even though semantic acceptance is a later capability.
2. A READY case is not required to freeze `final_tree_sha`. Arbitrary semantic digests can satisfy the non-empty hash requirement.
3. Expected file modes are not part of the frozen contract. Pair equality detects different modes between arms, but both arms can share the same wrong mode unless a final tree SHA is present and trusted.
4. Failure cases do not freeze the expected failing gate, authority state, diagnostic class, recovery class, or before/after workspace identity.
5. The contract does not identify an evaluator, evidence schema, backend requirement, repetition policy, order policy, or decision/report schema.
6. The schema has no backward-compatible evolution rule beyond `schema_version == 1`.

R1 implication:

- introduce a backward-compatible deterministic case-contract revision rather than weakening v1;
- retain v1 loading for historical fixtures;
- require explicit `exact_artifact` or `failure_boundary` mode in the new revision;
- require a final tree identity or an equally strict declared artifact identity for READY exact-artifact cases;
- require explicit no-mutation and failure-class expectations for failure-boundary cases.

### 6.2 `execute.py`

Strengths:

- command plans are deterministic and content-addressed;
- duplicate names and duplicate argv are rejected;
- A and B share the same manifest digest;
- event journals use exclusive creation and sequential JSONL records;
- child time and engine overhead are separated;
- nonzero status blocks immediately;
- exceptions and keyboard interruptions cannot emit `run_finished`.

R1 gaps:

1. The only executor is explicitly a fixture callback returning an integer.
2. The event journal does not retain stdout, stderr, output artifact, or command-result locators.
3. The journal does not bind base SHA, toolchain SHA, evaluator identity, backend identity, workspace identity, pair ID, repetition, or order position.
4. The caller chooses an arbitrary run ID; uniqueness and deterministic derivation are not enforced.
5. No timeout or external invalidation class exists.
6. No before/after workspace identity is observed or required.
7. No validator reloads a completed journal and proves contiguous sequence, monotonic timestamps, one terminal event, and terminal/summary consistency.
8. `INTERRUPTED` exists only in the fixture summary/event layer and cannot flow into `OutcomeSnapshot`.
9. `build_paired_plans` is not an opposite-order scheduler.

R1 implication:

- preserve fixture execution for unit tests;
- add a strict real-evidence path rather than silently renaming fixture execution as production execution;
- define a small command-result/evidence contract and a balanced schedule;
- keep subprocess policy and domain commands outside the evaluator and comparator.

### 6.3 `compare.py`

Strengths:

- each arm is checked against the manifest before pair equality is considered;
- two equally wrong arms do not pass;
- protected paths, hashes, gate plan, provenance, authority, terminal state, partial mutation, and recovery fields are represented;
- efficiency is withheld after any detected mismatch.

R1 gaps:

1. Snapshots are manually supplied Python objects, not normalized immutable evidence.
2. Exactness is only as strong as the manifest. The comparator cannot repair missing expected modes, missing final tree identity, or missing failure-class expectations.
3. For non-READY outcomes, the exact expected authority state, failing gate result, diagnostic code, and recovery class are not compared to frozen expectations.
4. `INTERRUPTED`, `TIMEOUT`, and `INVALIDATED` cannot be represented as run-execution terminal classes.
5. The report does not include arm IDs, run IDs, pair/repetition identities, evidence digests, comparison mode, evaluator identity, or its own digest.
6. Timing payload release is not cryptographically or structurally bound to the compared runs.
7. Pair-field equality is appropriate for R1 exact mode, but must not become the future semantic-acceptance implementation.

R1 implication:

- normalize evidence into snapshots through one strict loader;
- introduce an execution-validity layer before outcome equivalence;
- bind the pair report and efficiency record to exact run/evidence identities;
- leave semantic per-arm acceptance for R2.

### 6.4 Current CLI and Candidate 01

`scripts/run_workflow_replay.py` is a Candidate 01 command surface. It imports `orchestrate.py` and runs the V1 composition path. It is not a generic Core calibration or ingestion command.

R1 must not add generic evaluator rules inside Candidate 01 or make Candidate 01 imports a prerequisite for Core use. A separate small deterministic Core command is acceptable only if the implementation plan proves that a library-only calibration entry is insufficient.

## 7. Paired-repair reuse audit

`GOV-CODE-PAIRED-REPAIR-01` provides useful adjacent evidence:

- full Git commit resolution;
- base -> A0 -> B1 ancestry checks;
- first-attempt freeze;
- same-worker identity;
- gate snapshot identity;
- feedback digest and source;
- A0/B1 validation records;
- production/test churn and changed-file metrics;
- a durable pair JSON and comparison markdown;
- explicit wording that it is not a causal two-worker A/B result.

It should be reused by **artifact ingestion**, not imported as ReplayAB judgment logic.

Do not:

- copy its Git diff/metrics implementation into Core;
- treat `reviewer_correctness` as a generic evaluator contract;
- treat A0 -> B1 as Arm A -> Arm B randomization;
- import `scripts/paired_repair_report.py` directly into Core, because it is a governance workflow script with its own Git and scope dependencies;
- make paired-repair availability mandatory for deterministic R1 cases.

Recommended future reuse:

- define generic evidence locators and run identities that can reference paired-repair `PAIR.json`, `VALIDATION.json`, and feedback digest;
- add a thin paired-repair adapter in R3 or R5 after the generic Run Artifact contract exists;
- use one paired-repair record as an ingestion compatibility fixture, not as the primary R1 calibration case or as proof of stochastic behavior.

## 8. Minimum R1 implementation selected by this audit

### 8.1 Contract revision

Add a backward-compatible deterministic contract revision with:

- explicit `comparison_mode` restricted to `exact_artifact` or `failure_boundary` for R1;
- case-contract digest;
- expected final tree/artifact identity for exact success;
- expected file modes when final tree identity is unavailable;
- expected authority state;
- expected gate-result class;
- expected diagnostic and recovery class for failure cases;
- required before/after workspace identity rule;
- evaluator and evidence-schema digests;
- fixed order policy of two opposite-order pairs for the initial R1 calibration.

Historical schema-v1 fixtures remain readable and C0-only unless explicitly normalized through a reviewed compatibility rule.

### 8.2 Run identity and evidence locators

Introduce immutable validated objects equivalent to:

```text
RunIdentity
- case_id
- arm
- pair_id
- repetition
- order_position
- run_id
- backend_id

EvidenceLocator
- kind
- relative_path
- sha256
- byte_size
```

The implementation must reject:

- absolute paths;
- `..` traversal;
- symlinks in the evidence path;
- digest or size mismatch;
- duplicate run identities;
- evidence outside the declared root;
- mismatched case, arm, pair, repetition, backend, environment, toolchain, evaluator, or plan identity.

### 8.3 Deterministic Run Artifact v1

The minimum R1 artifact should bind:

- case-contract digest;
- run identity;
- base, task input, toolchain, evaluator, environment, cache, backend, and plan identities;
- event-log locator;
- outcome locator when an outcome exists;
- before/after workspace or artifact identity;
- execution terminal class;
- command count and timing summary;
- partial-mutation and invalidation evidence;
- producer identity and schema version.

R1 execution terminal classes must distinguish at least:

- `READY`;
- `BLOCKED`;
- `STALE`;
- `INTERRUPTED`;
- `INVALIDATED`.

Only the first three can enter the existing outcome comparator. Interrupted or invalidated runs remain visible and automatically block efficiency release.

### 8.4 Real deterministic evidence path

R1 should validate one narrow deterministic local evidence producer or pre-existing artifact producer. It must not build a generic backend framework.

The preferred implementation is:

1. retain `run_fixture_plan` for unit fixtures;
2. define a strict command-result artifact contract;
3. use a thin external local-command adapter to run declared argv without shell interpolation;
4. store stdout/stderr or their explicit omission class as content-addressed evidence;
5. collect before/after workspace identity through an existing owner or a thin adapter;
6. normalize the resulting artifact through Core;
7. keep Git, V1, pytest, authority, and domain semantics outside Core.

A pre-existing artifact path is also valid, but it must carry producer identity and immutable locators. Artifact ingestion alone must not be described as proof that ReplayAB controlled the original execution.

### 8.5 Opposite-order schedule

Generate two deterministic measured pairs per case:

```text
pair-0: A -> B
pair-1: B -> A
```

The schedule must freeze:

- pair ID;
- arm order;
- run ID;
- repetition number;
- environment and cache policy;
- expected evidence root;
- invalidation handling.

Every attempted run remains in raw evidence. No third repetition is allowed except under a predeclared environmental invalidation rule.

### 8.6 Calibration bank

Before R1 implementation results are inspected, freeze a calibration inventory containing at least:

1. one real frozen READY exact-artifact case;
2. one real frozen BLOCKED or STALE failure-boundary case with no mutation;
3. one one-arm protected-tree or hash mismatch;
4. one both-arms-same-wrong outcome;
5. one wrong file-mode or unauthorized-path outcome;
6. one interrupted run with no outcome artifact;
7. one blocked run with a changed after-tree or declared partial mutation;
8. one evidence-digest or identity mismatch;
9. one balanced-order schedule check;
10. one timing/report binding mismatch.

At least the first two cases must be real frozen repository artifacts. The adversarial cases may be derived from those artifacts, but their mutations and expected verdicts must be frozen before execution.

Store independent expected verdicts separately from generated ReplayAB reports and bind them by SHA-256. A unit test expectation embedded only in Python is insufficient as the sole calibration authority.

### 8.7 Report and efficiency binding

The R1 report must include:

- case-contract digest;
- pair and run identities;
- all evidence locators and digests;
- execution-validity verdicts;
- per-arm outcome verdicts;
- pair-equivalence verdict;
- mismatch codes;
- timing locator and timing identity;
- supported confidence grade;
- explicit unsupported claims;
- report schema version and report digest.

Efficiency is released only when:

- both runs are execution-valid;
- both outcomes independently match the frozen R1 contract;
- pair comparability passes;
- timing belongs to the same run identities and evidence set.

## 9. Proposed implementation scope

A future implementation should prefer the following small scope:

- add `src/drpo/workflow_replay/evidence.py`;
- update `src/drpo/workflow_replay/model.py` for a backward-compatible deterministic contract revision;
- update `src/drpo/workflow_replay/execute.py` for run identity, command results, journal validation, and balanced scheduling;
- update `src/drpo/workflow_replay/compare.py` only for execution validity and report/evidence binding;
- update `src/drpo/workflow_replay/__init__.py` exports;
- add one thin deterministic calibration command only if required;
- add focused tests and `tests/fixtures/workflow_replay/r1/**` calibration artifacts;
- add a frozen R1 calibration inventory and expected-verdict file under the workflow-optimization documentation tree.

Do not modify:

- `src/drpo/workflow_replay/orchestrate.py`;
- Candidate 01 tests or plan except for a later cross-reference if necessary;
- V1 or pilot-registration owners;
- paired-repair implementation;
- GitHub workflows;
- handoff or registry;
- scientific files.

## 10. Size, effort, and runtime budget recommendation

### 10.1 Current size

The current three Core implementation files contain 521 physical lines in total before `__init__.py`; the historical Stage-4 review counted approximately 447 production lines for Core logic. The focused Core tests contain 513 physical lines.

The old 500-line hard stop governed the original combined disposable prototype and was later superseded by a reviewed Stage-4 budget revision. It must not be silently reused or silently relaxed for R1. R1 needs a fresh incremental budget before code is written.

### 10.2 R1 incremental production budget

Recommended budget:

- preferred: `240--340` new or changed production lines;
- yellow review: `341--400`;
- hard redesign trigger: more than `400` production lines for the R1 diff;
- no new third-party dependency;
- no network work or full repository scan inside Core;
- test and calibration volume may exceed production volume.

Expected allocation:

| Area | Preferred lines |
|---|---:|
| deterministic contract revision | 45--75 |
| run identity, locators, artifact loader/normalizer | 110--160 |
| journal validation and balanced schedule | 45--75 |
| report/efficiency binding | 25--50 |
| optional thin calibration command and exports | 15--35 |

Expected R1 Core total after the change: approximately `700--850` counted production lines. This remains compatible with the longer-term Core target only if later stages reuse these objects instead of introducing parallel schemas.

Expected tests, calibration fixtures, and reviewed expected verdicts: `500--900` additional lines or equivalent structured artifacts.

### 10.3 Effort estimate

- implementation and focused tests: `12--18` active hours;
- calibration artifact preparation and independent expected-verdict review: `4--8` active hours;
- CI, defect repair, size review, and closure: `3--6` active hours;
- expected total: `19--32` active hours.

Entering the upper end requires a fresh ROI and architecture review before further expansion.

### 10.4 Runtime guardrails

For evidence sets up to the frozen calibration size limit:

- contract validation, artifact loading, normalization, scheduling, and report generation target median `<=250 ms` and p95 `<=1 s`;
- Core self-overhead excluding child command time targets median `<=1 s`;
- self-overhead above `max(2 s, 2% of the deterministic Arm-A median)` enters yellow review;
- self-overhead above `5 s`, duplicate full scans, duplicate gates, or Core network work triggers redesign;
- event count and evidence byte limits must be frozen before implementation to prevent unbounded ingestion.

## 11. R1 exit gate after the narrowed implementation

R1 may close only when all of the following pass at exact head:

1. every frozen calibration verdict matches the independently reviewed expected verdict;
2. both real frozen cases are ingested and compared end to end;
3. interrupted, invalidated, and partial-mutation runs cannot produce equivalent success or release timing;
4. both-same-wrong artifacts are rejected;
5. wrong mode, path, digest, identity, gate class, authority class, diagnostic class, or recovery class is detected where frozen;
6. two opposite-order pairs are generated and all attempts are retained;
7. report and timing identities match the exact compared runs;
8. schema-v1 historical fixtures remain readable or receive an explicit reviewed migration classification;
9. Candidate 01 and paired-repair behavior remain unchanged;
10. no V1, authority, registry, handoff, GitHub workflow, default route, or scientific behavior changes;
11. focused tests, full repository pytest, Ruff, authority, formal-channel, and governance checks pass;
12. runtime and code-size budgets pass or receive an explicit yellow-zone review;
13. the resulting claim is limited to C1 deterministic exact-artifact and failure-boundary replay.

## 12. Stop and redesign conditions

Stop the R1 implementation when any of the following becomes necessary:

- a generic plugin registry or backend service;
- a database or durable mutable state store;
- Core-owned Git, V1, pytest, authority, or scientific logic;
- Candidate 01-specific fields in generic contracts;
- semantic acceptance or hidden evaluator logic;
- repair-attempt orchestration;
- live worker isolation;
- automatic publication or merge;
- more than 400 production lines in the R1 diff without a smaller rejected alternative and explicit new approval;
- evidence producer trust cannot be stated precisely;
- partial mutation can only be asserted by the candidate without an independent before/after identity;
- calibration cases are selected or removed after candidate outcomes are inspected.

## 13. Remaining limitations after successful R1

Even a successful narrowed R1 will not establish that ReplayAB:

- accepts multiple different correct implementations;
- captures complete coding-agent attempts and repairs;
- discovers every unknown regression;
- isolates worker contexts or hides treatment;
- measures stochastic coding-agent error probabilities;
- controls server-side model build, routing, cache, or sampling state;
- is equivalent to a platform-internal A/B system;
- validates Candidate 01 for adoption;
- is a general cross-repository product without adapters.

Those claims remain assigned to R2--R6.

## 14. Branch and delivery recommendation

This audit branch should remain documentation-only.

After explicit user approval of the narrowed design, create a new implementation branch from the then-current exact `main`, tentatively:

```text
dev/replayab-core-r1-exact-artifact-01
```

Before writing behavior code, freeze:

- exact allowed paths;
- schema revision;
- calibration inventory and expected-verdict digests;
- event and artifact size limits;
- incremental production-code budget;
- runtime guardrails;
- focused and repository-wide test commands;
- rollback plan.

## 15. Final audit verdict

**Verdict: `NARROW`.**

The current Core has the correct small-module direction and several valuable fail-closed primitives. It should not be discarded. It is nevertheless not yet a calibrated C1 ruler.

Proceed only with the artifact-first deterministic slice defined here. Do not start Candidate 01 evaluation, semantic acceptance, complete trajectory work, backend generalization, or Regeneration Runner in the same implementation iteration.
