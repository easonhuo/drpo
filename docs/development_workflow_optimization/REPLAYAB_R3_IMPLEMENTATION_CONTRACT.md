# ReplayAB R3 Implementation Contract

Claim: `GOV-DEV-WORKFLOW-OPTIMIZATION-BENCHMARK-01`

Work ID: `REPLAYAB-R3-IMPLEMENTATION-CONTRACT-01`

Contract base: `main@18440ae79a155be1d8aa357ff0fdf3c238b018bb`

Gap audit: `docs/development_workflow_optimization/REPLAYAB_R3_GAP_AUDIT.md`

Calibration inventory: `docs/development_workflow_optimization/REPLAYAB_R3_CALIBRATION_INVENTORY.yaml`

Status: `PROPOSED; IMPLEMENTATION_NOT_AUTHORIZED`

Scientific impact: none

## 1. Decision

This contract specifies R3 as a narrow evidence-model extension only. It must preserve the complete ordered attempt trajectory for one ReplayAB run and bind that trajectory to immutable evidence without changing R2 semantic acceptance.

The implementation must make the following coexist without contradiction:

```text
final_acceptance: PASS
initial_terminal: FAILED
repair_count: 2
candidate_failure_count: 2
final_attempt_terminal: SUCCEEDED
trajectory_complete: true
```

Earlier failures are process facts. They do not retroactively change the final R2 verdict.

This contract does not authorize code changes. Implementation may begin only after the repository owner explicitly approves this contract and the exact future Python paths listed below.

## 2. Frozen responsibility

R3 answers one question:

> Can every final arm verdict be traced through one complete ordered attempt trajectory to immutable raw evidence, including unsuccessful attempts that occurred before the final result?

R3 must:

- preserve exactly one initial attempt at ordinal zero;
- preserve every later repair attempt in order;
- preserve failed, timed-out, interrupted, invalidated, and unrepaired attempts;
- distinguish candidate failure from environment invalidation and insufficient evidence;
- bind repairs to their parent attempt and feedback evidence;
- bind intermediate and final candidate artifacts independently;
- bind the trajectory to the final outcome and R2 acceptance evidence when available;
- expose a deterministic trajectory summary without rewriting R2 correctness;
- reject incomplete, reordered, misbound, or tampered trajectories fail-closed.

## 3. Explicit non-responsibility

R3 must not:

- execute an evaluator;
- execute a coding agent;
- create a worker, queue, service, daemon, database, or network protocol;
- add a backend plugin framework;
- add sandbox or container orchestration;
- perform stochastic repeated A/B analysis;
- infer population-level agent quality;
- adopt Candidate 01;
- modify scientific code, configs, seeds, thresholds, budgets, results, handoff, or registry;
- change R1 exact-artifact or failure-boundary behavior;
- change R2 acceptance, pair verdict, or efficiency-release behavior;
- report only the last successful attempt.

R4, R5, and R6 remain separate stages.

## 4. Attempt boundary

An attempt is frozen as:

> one complete candidate submission followed by its declared evaluation boundary and terminal evidence.

An attempt is not:

- one tool call;
- one command;
- one file edit;
- one model message;
- an arbitrary time slice selected after results are known.

The initial attempt is ordinal `0` and kind `INITIAL`.

Every later attempt is kind `REPAIR`, has ordinal `n > 0`, and must identify the immediately preceding attempt as its parent. Independent retries are represented as repair attempts with feedback class `NONE`; they still retain parent linkage.

## 5. Frozen enums

### 5.1 Attempt kind

```text
INITIAL
REPAIR
```

### 5.2 Attempt terminal

```text
SUCCEEDED
FAILED
TIMED_OUT
INTERRUPTED
INVALIDATED
```

### 5.3 Disposition

```text
NONE
CANDIDATE
ENVIRONMENT
INSUFFICIENT_EVIDENCE
```

Compatibility rules:

| terminal | allowed disposition |
|---|---|
| `SUCCEEDED` | `NONE` |
| `FAILED` | `CANDIDATE`, `INSUFFICIENT_EVIDENCE` |
| `TIMED_OUT` | `CANDIDATE`, `ENVIRONMENT`, `INSUFFICIENT_EVIDENCE` |
| `INTERRUPTED` | `CANDIDATE`, `ENVIRONMENT`, `INSUFFICIENT_EVIDENCE` |
| `INVALIDATED` | `ENVIRONMENT`, `INSUFFICIENT_EVIDENCE` |

An environment invalidation must never increment candidate-failure count.

### 5.4 Feedback class

```text
NONE
EVALUATOR
AUTHORITY
EXECUTION
OPERATOR
```

Rules:

- initial attempts must use `NONE` and have no feedback locator;
- repairs may use `NONE` only for an independent retry;
- repairs using any non-`NONE` class must provide exactly one feedback locator;
- the feedback payload remains opaque to R3 Core;
- R3 validates identity and content binding but does not interpret natural language.

### 5.5 Final acceptance

```text
PASS
REJECTED
NOT_AVAILABLE
```

`PASS` and `REJECTED` are existing R2 outcomes and require immutable acceptance-evidence binding.

`NOT_AVAILABLE` is permitted only when execution did not reach a valid R2 acceptance boundary, such as interruption or environment invalidation.

## 6. Frozen resource vocabulary

The bounded initial vocabulary is:

```text
command_count
active_ns
retained_bytes
tool_operation_count
token_count
message_count
monetary_microunits
```

Each backend declares, once per run, every dimension as exactly one of:

```text
OBSERVED
UNAVAILABLE
```

Rules:

- no dimension may be omitted;
- no dimension may be both observed and unavailable;
- unavailable values must not be fabricated as zero;
- attempt records contain values only for observed dimensions;
- observed values are non-negative integers;
- run-level values must equal the frozen sum of attempt-level values;
- `attempt_count` and `repair_count` are derived and are not backend-reported resource fields;
- monetary values use integer microunits; floating-point currency is forbidden.

Fixture normalization must observe at least `command_count`, `active_ns`, and `retained_bytes`.

Historical reconstruction may declare dimensions unavailable when the raw evidence cannot support them.

## 7. Frozen normalized schemas

Exact Python representation may use frozen dataclasses, but serialized payloads must contain exactly the following concepts.

### 7.1 AttemptRecord

```text
attempt_id
ordinal
kind
parent_attempt_id
terminal
disposition
input_artifact_locator
output_artifact_locator
event_journal_locator
feedback_class
feedback_locator
diagnostic_codes
observed_resources
attempt_sha256
```

Required invariants:

- `attempt_id = canonical_sha256({run_id, ordinal})`;
- ordinal zero exists exactly once;
- ordinals are contiguous and stored in increasing order;
- ordinal zero is `INITIAL`, has no parent and no feedback;
- every later attempt is `REPAIR` and its parent is the immediately preceding attempt;
- diagnostic codes are sorted and unique;
- locators use the existing `EvidenceLocator` contract;
- output artifact may be absent only when terminal evidence proves no candidate artifact was produced;
- event journal is required for every attempt;
- non-`NONE` feedback requires a locator;
- attempt digest covers the complete canonical payload excluding `attempt_sha256` itself.

### 7.2 RunArtifact

```text
schema_version
run_identity
base_sha
toolchain_sha
environment_id
cache_policy
backend_id
resource_capabilities
attempts
first_attempt_id
final_attempt_id
final_outcome_locator
final_acceptance
acceptance_evidence_locator
aggregate_observed_resources
run_artifact_sha256
```

Required invariants:

- schema version is integer `1`;
- `run_identity` uses the existing deterministic `RunIdentity` contract;
- backend ID equals `run_identity.backend_id`;
- attempts are non-empty and bounded;
- `first_attempt_id` identifies ordinal zero;
- `final_attempt_id` identifies the last stored attempt;
- no attempt may be omitted between first and final;
- final outcome must bind to the same case, arm, run, and final attempt;
- `PASS` or `REJECTED` requires an acceptance-evidence locator;
- `NOT_AVAILABLE` forbids an acceptance-evidence locator;
- aggregate resources equal the validated attempt sum;
- run artifact digest covers the complete canonical payload excluding `run_artifact_sha256` itself.

### 7.3 TrajectorySummary

The report summary is derived, not user-supplied.

It contains exactly:

```text
final_acceptance
initial_terminal
repair_count
candidate_failure_count
timeout_count
interruption_count
invalidation_count
final_attempt_terminal
trajectory_complete
```

`trajectory_complete` is true only after all schema, lineage, locator, digest, final-pointer, and resource checks pass.

A caller must never receive a partial summary from an invalid artifact.

## 8. Evidence and size limits

The initial bounded implementation freezes:

- maximum attempts per run: `32`;
- maximum locators per attempt: `8`;
- maximum serialized normalized RunArtifact size: `262144` bytes;
- maximum individual evidence item: existing `EvidenceLocator` limit;
- canonical UTF-8 JSON with sorted keys and compact separators;
- no symlinks, path escapes, absolute paths, backslashes, or mutable external URLs;
- no secrets or unsanitized arbitrary workspace scraping.

A limit breach is invalid evidence, not candidate failure.

## 9. Public API contract

The initial implementation must expose these public names from the proposed R3 module:

```text
AttemptRecord
RunArtifact
TrajectorySummary
TrajectoryError
validate_attempt_record
validate_r3_run_artifact
load_r3_run_artifact
summarize_trajectory
```

Required behavior:

- validation is deterministic and side-effect free except bounded evidence reads;
- loader rejects symlinks, unsafe paths, oversized JSON, malformed UTF-8, unknown keys, and digest mismatch;
- all invalid calibration artifacts raise `TrajectoryError` with the frozen calibration error code available as structured data or a stable message prefix;
- no API performs network access or executes child processes;
- no R3 API mutates R1 or R2 objects.

The fixture adapter must expose one narrow function in the existing execution layer:

```text
normalize_fixture_attempt
```

It converts already-produced fixture journal and locator evidence into one AttemptRecord-compatible payload. It does not run commands and does not assemble multi-attempt policy logic.

Historical evidence is accepted only through a complete explicit normalized RunArtifact payload plus immutable locators. No heuristic log scraping is authorized.

## 10. Exact proposed implementation paths

### 10.1 New Python paths requiring explicit owner approval

```text
src/drpo/workflow_replay/trajectory.py
tests/test_workflow_replay_r3.py
```

Status: `PROPOSED_NEW_PYTHON_PATHS; NOT_YET_APPROVED_FOR_CREATION`

These exact paths must be named in the repository-owner implementation approval before either file is created.

No other new Python path is permitted.

### 10.2 Existing Python paths permitted for bounded modification

```text
src/drpo/workflow_replay/evidence.py
src/drpo/workflow_replay/execute.py
tests/test_workflow_replay_execute.py
```

`src/drpo/workflow_replay/compare.py` is frozen and must not change in the initial implementation. R2 comparison and efficiency-release semantics remain untouched.

### 10.3 Non-Python calibration fixtures permitted

```text
tests/fixtures/workflow_replay/r3/**
```

### 10.4 Result and review documents permitted after implementation

```text
docs/development_workflow_optimization/REPLAYAB_R3_IMPLEMENTATION_RESULT.md
docs/development_workflow_optimization/REPLAYAB_R3_CLOSURE_REVIEW.md
```

No handoff or registry modification is authorized by this contract.

## 11. Code budget

### 11.1 Production Python

| path | net-addition target | hard ceiling |
|---|---:|---:|
| `trajectory.py` | 300–450 | 500 |
| `evidence.py` | 0–120 | 160 |
| `execute.py` | 40–100 | 120 |
| **total production Python** | **450–700** | **800** |

Crossing 700 lines requires an explicit review note explaining why reuse was insufficient.

Crossing 800 lines blocks implementation acceptance and requires a contract amendment before continuing.

### 11.2 Python tests

| path | target | hard ceiling |
|---|---:|---:|
| `test_workflow_replay_r3.py` | 550–800 | 900 |
| `test_workflow_replay_execute.py` additions | 50–150 | 180 |
| **total Python tests** | **600–950** | **1080** |

### 11.3 Fixture and documentation budget

- calibration fixture data: target 200–400 lines, hard ceiling 600;
- implementation-result document: target at most 350 lines;
- no generated large logs or binary artifacts committed to Git.

## 12. Two-PR implementation sequence

### R3-A: Core schema and validator

Allowed work:

- create `trajectory.py` after exact path approval;
- define frozen enums, records, hashing, validation, loading, and summary;
- create `test_workflow_replay_r3.py` after exact path approval;
- materialize the 16 frozen calibration cases as deterministic fixtures;
- pass cases R3-C01 through R3-C16 with the frozen expected verdicts;
- add no fixture-execution integration beyond what is needed to construct test evidence.

R3-A exit gate:

- all 16 calibration expected verdicts match exactly;
- R1 and R2 focused tests remain unchanged and pass;
- full repository gates pass;
- no unapproved path exists;
- production-code budget remains within contract.

### R3-B: Fixture normalization and reporting integration

Allowed work:

- add `normalize_fixture_attempt` to `execute.py`;
- add bounded locator helpers to `evidence.py` only when necessary;
- extend existing execution tests;
- demonstrate one first-attempt success, one repair-success trajectory, one unrepaired failure, one interruption, and one environment invalidation through the same normalized contract;
- expose derived summary without modifying R2 acceptance or compare logic;
- demonstrate historical explicit-payload and fixture normalization compatibility.

R3-B exit gate:

- no last-success-only report path;
- fixture and historical evidence normalize to the same RunArtifact schema;
- R1/R2 exact behavior remains unchanged;
- full repository gates pass;
- exact-candidate terminal audit succeeds.

R3 closure requires a later closure review and immutable closure record. Completion of R3-A or R3-B alone does not close R3.

## 13. Frozen calibration contract

`REPLAYAB_R3_CALIBRATION_INVENTORY.yaml` is authoritative for:

- case IDs;
- valid versus rejected ingestion;
- stable expected error codes;
- expected summary values;
- the rule that cases cannot be weakened or removed after implementation results.

Valid cases:

```text
R3-C01 through R3-C07
R3-C16
```

Fail-closed cases:

```text
R3-C08 through R3-C15
```

Implementation code must not special-case case IDs. The cases exercise general invariants.

## 14. Required non-regression gates

Before either implementation PR may merge:

- focused R3 calibration tests pass;
- all existing `tests/test_workflow_replay_*.py` tests pass;
- R1 exact-artifact calibration remains unchanged;
- R1 failure-boundary behavior remains unchanged;
- R2 semantic acceptance remains unchanged;
- R2 pair verdict and efficiency-release gating remain unchanged;
- full repository pytest passes;
- Ruff passes;
- Python compilation passes;
- shell syntax checks pass where applicable;
- handoff authority passes;
- formal execution channel passes;
- governance inventory passes;
- governance stage status passes;
- Evidence Locator Gate passes;
- new-Python-path oral-approval gate passes for the exact two proposed paths.

Smoke tests, static checks, or a partial calibration subset are not formal R3 results.

## 15. Terminal audit contract

Any claim that R3 is implemented, stable, or ready to close requires an exact-candidate terminal audit that records:

- base SHA;
- implementation head SHA;
- synthetic merge candidate SHA;
- changed-file inventory;
- production and test line-budget accounting;
- all 16 calibration results;
- focused ReplayAB results;
- full repository pytest count;
- Ruff, compile, authority, formal, inventory, stage, and locator gate exits;
- R1/R2 non-regression statement;
- artifact ID and digest;
- explicit `r3_close_permitted` boolean.

No audit tied only to a branch head may substitute for the exact merge candidate.

## 16. Stop conditions

Implementation must stop and return to design review when any of the following occurs:

- a third new Python path appears necessary;
- production Python exceeds 800 net new lines;
- a database, service, worker, network fetch, backend registry, or plugin abstraction appears necessary;
- attempt boundaries cannot be defined without result-dependent judgment;
- historical evidence requires heuristic log scraping;
- any frozen calibration case must be weakened to pass;
- R1 or R2 behavior changes;
- resource values would need to be fabricated;
- scientific code, handoff, or registry would need modification.

## 17. Approval required before implementation

Implementation authorization must explicitly approve:

```text
REPLAYAB-R3-IMPLEMENTATION-CONTRACT-01
src/drpo/workflow_replay/trajectory.py
tests/test_workflow_replay_r3.py
```

The approval may authorize R3-A first while withholding R3-B.

Ordinary merge approval remains separate from implementation authorization.

Until that approval is recorded, the only permitted action is review or amendment of this contract and calibration inventory.
