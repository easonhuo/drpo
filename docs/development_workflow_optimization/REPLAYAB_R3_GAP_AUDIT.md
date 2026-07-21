# ReplayAB R3 Gap Audit

Claim: `GOV-DEV-WORKFLOW-OPTIMIZATION-BENCHMARK-01`

Work ID: `REPLAYAB-R3-TRAJECTORY-RUN-ARTIFACT-GAP-AUDIT-01`

Audit base: `main@d3f7d046f948108a3d837bdcff617eed5146a2f0`

Predecessor closure: `docs/development_workflow_optimization/REPLAYAB_R2_CLOSURE.md`

Status: `NARROW_IMPLEMENTATION_RECOMMENDED; IMPLEMENTATION_NOT_AUTHORIZED`

Scientific impact: none

## 1. Decision summary

ReplayAB R3 should be implemented as a narrow evidence-model extension that preserves complete attempt trajectories and binds them into one immutable, backend-normalized `RunArtifact`.

R3 must not replace or reinterpret the R2 semantic-acceptance verdict. The intended relationship is:

- R2 answers whether the final submitted outcome satisfies the frozen acceptance contract;
- R3 records how that outcome was produced, including unsuccessful initial attempts, repairs, feedback, timeouts, interruptions, invalidations, intermediate artifacts, and observable resource use;
- a run may therefore have final acceptance `PASS` while its R3 trajectory truthfully contains earlier failed attempts;
- those earlier failures are trajectory facts, not a retroactive change of the final R2 verdict.

The audit verdict is `NARROW_IMPLEMENTATION_RECOMMENDED` because the repository already contains useful identity, evidence-location, event-journal, terminal-state, outcome-snapshot, and timing primitives. The missing work is material but bounded. A database, service, live coding-agent worker, execution backend framework, evaluator plugin system, or stochastic benchmark is not required for R3.

This document authorizes no implementation.

## 2. Authority and stage boundary

R2 is formally closed at `main@d3f7d046f948108a3d837bdcff617eed5146a2f0` as a bounded deterministic semantic-acceptance evidence ruler.

The R2 closure permits only a documentation-first R3 gap audit and design review. It does not authorize:

- R3 production-code changes;
- evaluator execution inside ReplayAB Core;
- live coding-agent workers;
- hidden evaluator services;
- new execution backends;
- stochastic repeated A/B experiments;
- Candidate 01 adoption;
- handoff or registry changes;
- scientific execution.

A separate reviewed R3 implementation contract and explicit repository-owner authorization are required before code work.

## 3. R3 question and non-question

### 3.1 Question R3 must answer

For every final arm verdict, can ReplayAB reconstruct a complete, immutable chain from the final decision back through all attempts and their raw evidence without hiding failures that occurred before the final result?

### 3.2 Question R3 must not answer

R3 does not decide whether the final implementation is semantically correct. R2 remains the correctness ruler.

R3 also does not estimate population-level agent quality, benchmark generalization, or stochastic superiority. It exposes process evidence needed for later controlled comparisons.

## 4. Repository surfaces inspected

The audit inspected the current ReplayAB roadmap and R2 closure boundary together with the implementation surfaces that currently carry execution and evidence semantics:

- `docs/development_workflow_optimization/REPLAYAB_ENGINE_ROADMAP.md`;
- `docs/development_workflow_optimization/REPLAYAB_R2_CLOSURE.md`;
- `src/drpo/workflow_replay/model.py`;
- `src/drpo/workflow_replay/execute.py`;
- `src/drpo/workflow_replay/evidence.py`;
- `src/drpo/workflow_replay/compare.py`;
- existing ReplayAB tests under `tests/test_workflow_replay_*.py`.

No existing R3 gap-audit document or implemented `RunArtifact` / attempt-trajectory schema was found at the audit base.

## 5. Existing reusable capability

### 5.1 Deterministic run identity

`RunIdentity` already binds case, arm, pair, repetition, order position, and backend identity into a deterministic run identifier. This should remain the root identity for the R3 artifact rather than introducing a competing run-ID system.

### 5.2 Immutable evidence locators

`EvidenceLocator` already provides:

- a typed evidence kind;
- repository-relative path validation;
- byte-size binding;
- SHA-256 binding;
- symlink and path-escape rejection;
- bounded evidence reads.

R3 should compose these locators instead of embedding arbitrarily large raw logs inside the normalized artifact.

### 5.3 Append-only event journal

`run_fixture_plan` already creates a new event file exclusively and writes ordered JSONL events with:

- run identity;
- monotonic sequence;
- monotonic timestamp;
- event type;
- event payload;
- flush after every event.

It records run start, command start, command finish, blocked terminal state, interrupted terminal state, and successful terminal state.

This is a useful raw-evidence source, but it currently describes one fixture-plan execution, not a multi-attempt repair trajectory.

### 5.4 Terminal and diagnostic evidence

Existing evidence code recognizes terminal events including ready, blocked, stale, interrupted, and invalidated forms. `OutcomeSnapshot` already carries final terminal state, safety boundary, changed paths, file modes, output hashes, authority and gate results, provenance, diagnostic codes, partial-mutation status, and recovery class.

These final-state objects should be referenced by R3, not duplicated or weakened.

### 5.5 Timing separation

Fixture execution already separates total elapsed time, child-command time, and ReplayAB self-overhead. R3 can reuse these quantities at attempt level and aggregate them at run level.

### 5.6 Existing R1/R2 evidence bindings

R1 and R2 already fail closed on identity and evidence mismatches, including run identity, evaluator identity, evidence schema, acceptance contract, case, and final outcome content. R3 should extend this chain backward to intermediate attempts without changing those closed contracts.

## 6. Material gaps

### 6.1 No explicit attempt model

The current repository has no normalized object for:

- the first complete attempt;
- a repair attempt;
- attempt ordinal and parent linkage;
- attempt-level terminal state;
- attempt-level output artifact;
- attempt-level feedback input;
- attempt-level diagnostics and resources.

Without an attempt model, a final successful snapshot cannot prove whether it was produced immediately or after multiple failed repairs.

### 6.2 No first-attempt freeze

There is no invariant requiring exactly one immutable initial attempt at ordinal zero. A later successful state could therefore be presented without proving that earlier attempts were retained.

### 6.3 No repair and feedback lineage

There is no binding from a repair attempt to:

- the prior attempt it repairs;
- the feedback evidence that triggered the repair;
- the class of feedback;
- the intermediate candidate artifact used as repair input.

Consequently, the repository cannot yet distinguish independent retries, feedback-driven repairs, and replacement runs.

### 6.4 No normalized complete `RunArtifact`

Existing evidence types are useful components but do not form one object that binds:

- run identity and frozen environment;
- the ordered complete attempt sequence;
- intermediate and final artifacts;
- feedback and diagnostics;
- final outcome / acceptance evidence;
- resource totals;
- a canonical artifact digest.

### 6.5 No enforceable failed-run retention

The current contracts do not prove that failed, timed-out, interrupted, invalidated, or unrepaired runs remain visible. There is no fail-closed rule against publishing only the last successful attempt.

### 6.6 Incomplete environment-versus-candidate separation

The evidence layer recognizes interruption and invalidation concepts, but the normalized final outcome and fixture runner do not yet provide one complete trajectory taxonomy that reliably distinguishes:

- candidate failure;
- environment invalidation;
- timeout;
- interruption;
- malformed or missing evidence;
- successful completion.

This distinction must be explicit at attempt level and at overall-run level.

### 6.7 No cross-backend normalization contract

Historical artifacts, fixture execution, and future runners may expose different raw formats. There is no R3 ingestion contract requiring each backend adapter to produce the same normalized attempt and run objects while preserving backend-specific raw evidence through locators.

### 6.8 Resource evidence is partial

Elapsed child time and ReplayAB overhead are available for fixture commands, but the repository has no normalized fields for observable counts such as:

- attempt count;
- command or tool-operation count;
- active execution time;
- feedback count;
- bytes retained;
- externally supplied token, message, or monetary usage.

R3 must not fabricate unavailable resource values. Required identity and attempt counts can be strict; backend-dependent resource fields must be explicitly observed or explicitly unavailable.

### 6.9 No trajectory-level tamper checks

Existing digest checks protect individual evidence items, but no contract currently rejects:

- missing initial attempt;
- duplicate attempt ordinal;
- reordered attempts;
- broken parent linkage;
- final-attempt pointer mismatch;
- feedback bound to the wrong attempt;
- an omitted failed attempt;
- aggregate resources inconsistent with attempt resources.

## 7. Recommended narrow capability

### 7.1 Attempt record

The implementation contract should define one immutable attempt record with the minimum concepts below. Exact field names remain to be frozen in the implementation contract.

Required concepts:

- deterministic attempt identity derived from run identity and ordinal;
- ordinal beginning at zero with no gaps;
- kind: initial or repair;
- optional parent attempt, forbidden for the initial attempt and required for repairs;
- terminal class;
- candidate-versus-environment disposition;
- input candidate/artifact locator where applicable;
- output candidate/artifact locator where applicable;
- event-journal locator;
- feedback locator and feedback class for repairs;
- diagnostics;
- observed resource accounting;
- canonical attempt digest.

A failed attempt remains a valid evidence record. Failure of the candidate is not failure of trajectory ingestion.

### 7.2 Terminal and disposition separation

A narrow contract should avoid one overloaded status field. At minimum it must distinguish:

- whether an attempt completed, failed, timed out, was interrupted, or was invalidated;
- whether the cause is attributable to candidate behavior, environment validity, or insufficient evidence.

The exact enum should be frozen only after calibration cases are written. The design must prevent an environment-invalidated attempt from being counted as candidate failure.

### 7.3 Feedback classes

The implementation should begin with a small frozen feedback taxonomy sufficient for deterministic calibration, for example:

- evaluator or test feedback;
- authority or safety-boundary feedback;
- execution or infrastructure feedback;
- human/operator feedback;
- no feedback / independent retry.

Feedback content should remain external immutable evidence referenced by locator. R3 should not interpret natural-language feedback or execute an evaluator.

### 7.4 Run artifact

One immutable `RunArtifact` should bind:

- schema version;
- deterministic `RunIdentity`;
- base, toolchain, environment, cache, and backend identities already required by earlier stages;
- ordered attempt records;
- the immutable first-attempt identity;
- optional final-attempt identity;
- final `OutcomeSnapshot` / R1 or R2 acceptance evidence locators;
- overall terminal classification;
- aggregate observed resources;
- explicit unavailable-resource fields or capability declaration;
- canonical run-artifact digest.

The final-attempt pointer may identify a successful or unsuccessful final attempt. A run with no accepted result must still produce a valid artifact when evidence is complete.

### 7.5 Final verdict versus trajectory summary

The report layer should present both without conflation:

- `final_acceptance`: the existing R2 result when available;
- `trajectory_summary`: initial-attempt result, number of repairs, failed-attempt count, timeout/interruption/invalidation counts, final-attempt identity, and complete-evidence status.

Example:

```text
final_acceptance: PASS
initial_attempt: FAILED
repair_attempts: 2
failed_attempts: 2
final_attempt: SUCCEEDED
trajectory_complete: true
```

This is the precise interpretation of “R2 passes while R3 shows earlier failures.”

### 7.6 No database or service

The normalized artifact should be a canonical bounded JSON-compatible object plus immutable evidence locators. JSONL may remain the raw event format. Validation and ingestion must work from a directory tree or artifact bundle without a database, daemon, network request, or mutable registry.

## 8. Preliminary implementation boundary

This audit does not freeze exact files, but a narrow implementation should prefer extending existing ReplayAB modules and existing test files rather than creating a platform or broad new package tree.

Likely responsibilities:

- evidence schema, parsing, canonical hashing, and validation in the existing evidence layer;
- fixture-journal normalization in the existing execution layer;
- report exposure without changing R2 acceptance semantics;
- calibration and regressions in existing ReplayAB test surfaces where practical.

A subsequent implementation contract must specify exact paths and line-budget expectations before code changes.

## 9. Required calibration matrix before implementation results

The following cases should be frozen before evaluating implementation output:

1. initial attempt succeeds with no repair;
2. initial candidate failure followed by one successful repair;
3. multiple candidate failures followed by success;
4. all attempts fail and remain visible;
5. timeout followed by a valid repair, with timeout retained;
6. interruption with no repair;
7. environment invalidation that is not counted as candidate failure;
8. missing initial attempt;
9. duplicate or gapped attempt ordinal;
10. reordered attempts or broken parent linkage;
11. feedback locator bound to the wrong repair;
12. tampered intermediate artifact or event journal;
13. final-attempt pointer inconsistent with final outcome;
14. aggregate resources inconsistent with attempt records;
15. backend omits an unavailable-resource declaration;
16. complete artifact containing a final rejected R2 result.

Cases 1-7 and 16 are valid artifacts with distinct summaries. Cases 8-15 must fail closed.

No failed or inconvenient calibration case may be removed after implementation results are observed.

## 10. Proposed exit gates

R3 may close only when all of the following are demonstrated on a frozen deterministic calibration bank:

1. every final verdict is traceable through the `RunArtifact` to immutable raw evidence;
2. exactly one first complete attempt is retained at ordinal zero;
3. every repair is linked to its parent and feedback evidence;
4. failed and unrepaired runs produce valid visible artifacts when their evidence is complete;
5. no report path can reduce a multi-attempt run to last-success-only evidence;
6. timeout, interruption, environment invalidation, candidate failure, and insufficient evidence remain distinct;
7. missing, duplicate, reordered, gapped, or tampered attempts fail closed;
8. intermediate and final artifacts are independently content-bound;
9. aggregate resources equal the sum or declared aggregation of observed attempt resources;
10. unavailable backend-dependent resource fields are explicit and never fabricated;
11. fixture and historical evidence normalize to the same contract without a database or service;
12. R1 exact-artifact and failure-boundary behavior remains unchanged;
13. R2 semantic acceptance and efficiency-release behavior remains unchanged;
14. repository-wide tests, Ruff, compilation, authority, formal-channel, inventory, and governance gates pass;
15. a terminal audit is tied to the exact merge candidate before closure.

## 11. Risks and controls

### 11.1 Attempt-boundary ambiguity

Risk: a backend may define an “attempt” opportunistically after seeing results.

Control: freeze the attempt-boundary rule before calibration results. For the initial narrow implementation, an attempt should be a complete candidate submission followed by its declared evaluation boundary, not an arbitrary tool call or file edit.

### 11.2 Evidence volume

Risk: embedding logs, messages, and artifacts makes the normalized object large and unstable.

Control: store bounded metadata and immutable locators; keep raw evidence external and digest-bound.

### 11.3 Secret or sensitive material

Risk: raw tool logs or messages may contain credentials or unrelated private data.

Control: R3 must not require ingestion of secrets. Backend adapters remain responsible for producing sanitized evidence artifacts; the core validates locators and digests rather than scraping arbitrary workspaces.

### 11.4 Backend coupling

Risk: fields are designed around GitHub Actions or one coding agent.

Control: freeze backend-neutral concepts; retain backend-specific material as typed raw evidence rather than core schema fields.

### 11.5 Scope expansion into R5

Risk: trajectory normalization grows into worker execution, sandboxing, or plugin infrastructure.

Control: R3 accepts already-produced evidence. Executing real workers and building pluggable execution backends remain R5 responsibilities.

### 11.6 Changed interpretation of R2

Risk: earlier failed attempts cause an accepted final outcome to be relabeled as incorrect.

Control: keep final semantic acceptance and trajectory summary as separate report dimensions. R3 adds process truth but does not rewrite R2 correctness.

## 12. ROI assessment

### Value

High for the parent workflow-optimization claim. Without R3, a workflow that succeeds only after repeated repair can look identical to a first-attempt success, and last-success-only evidence can hide operational regressions. R3 enables later comparison of repair burden, failure modes, and active cost without weakening correctness gates.

### Cost

Moderate if kept narrow. The repository already has most identity, digest, locator, journal, snapshot, and timing primitives. The principal work is schema composition, invariants, adapters for existing evidence, reporting, and calibration.

### Complexity risk

Acceptable only under an implementation contract that forbids database/service work, live workers, evaluator execution, broad backend abstraction, and stochastic inference.

### Verdict

`NARROW_IMPLEMENTATION_RECOMMENDED`.

## 13. Remaining uncertainties to resolve in the implementation contract

1. the exact attempt-boundary rule for historical reconstructed evidence;
2. whether feedback class belongs in the core enum or a frozen per-contract vocabulary;
3. how a backend declares observable versus unavailable resource dimensions;
4. whether final outcome and acceptance objects are embedded canonical payloads or digest-bound locators;
5. the maximum attempt count, evidence-locator count, and normalized JSON size;
6. whether `INTERRUPTED` and `INVALIDATED` remain terminal classes or are represented by terminal plus disposition fields;
7. the smallest existing-file implementation plan that preserves R1/R2 compatibility;
8. the exact calibration inventory and expected verdict document paths.

These uncertainties do not block the audit verdict. They must be frozen before implementation.

## 14. Recommended next step

After this audit is reviewed and merged, prepare a separate R3 implementation contract that:

- freezes the exact schema and enums;
- freezes the calibration inventory and expected verdicts before implementation results;
- names exact modified paths and code budget;
- preserves R1 and R2 behavior;
- defines liveness and terminal-audit gates;
- explicitly excludes R4, R5, R6, Candidate 01 adoption, handoff/registry changes, and scientific execution.

No R3 code should be written until that implementation contract receives explicit repository-owner approval.
