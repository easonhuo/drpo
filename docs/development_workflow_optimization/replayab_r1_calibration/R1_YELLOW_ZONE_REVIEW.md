# ReplayAB R1 Yellow-Zone Architecture Review

**Claim:** `GOV-DEV-WORKFLOW-OPTIMIZATION-BENCHMARK-01`  
**Initial base:** `main@2f677f4b00954ea71d0efa7def552a1ea3daa565`  
**Current integration base:** `main@531506b4be6967012c5d01aa4d112deaaccf1c49`  
**Branch:** `dev/replayab-core-r1-exact-artifact-01`  
**Decision:** `GO_WITHIN_YELLOW`  
**R1 status:** implementation candidate; C1 not yet granted

## Measured size

The narrowed R1 production delta is one new module:

- `src/drpo/workflow_replay/evidence.py`: **400 nonblank, non-comment lines**.

No existing production module is modified. This is the upper boundary of the frozen `341--400` yellow review range and does not exceed the `>400` hard redesign trigger. Any further production addition returns the decision to `REDESIGN` unless code is first reduced within the same frozen scope.

## Why the yellow-zone implementation is retained

The module contains one cohesive deterministic evidence boundary:

1. schema-v2 R1 contract validation while delegating unchanged base fields to the existing schema-v1 validator;
2. deterministic run identity and two-opposite-pair scheduling;
3. content-addressed evidence locators with root, traversal, parent-symlink, regular-file, size, role, and digest checks;
4. Run Artifact normalization, terminal/journal consistency, workspace-mutation derivation, and real result-artifact verification;
5. strict per-arm frozen-contract checks before pair equivalence;
6. pair-report, Run Artifact, evidence, run-ID, and timing binding.

Moving these responsibilities into `model.py`, `execute.py`, or `compare.py` would not reduce total production logic and would increase regression exposure to accepted C0 paths.

## Smaller alternatives reviewed and rejected

- **Keep caller-built `OutcomeSnapshot`:** does not prove artifact ingestion or evidence identity.
- **Trust locator metadata without reading files:** permits digest or size substitution.
- **Verify a file and reopen it later:** leaves a verify-to-use replacement window; R1 now parses the exact verified bytes.
- **Bind run IDs but not timing:** permits arbitrary timing substitution.
- **Trust a mutable pair-report dataclass:** permits report-field mutation; efficiency release now rechecks the report digest.
- **Use only pair equality:** permits both arms to be identically wrong.
- **Import paired-repair judgment logic:** violates component ownership and does not provide randomized A/B.
- **Build a generic backend framework:** is outside R1 and a hard redesign condition.

## Risk review

Remaining risks are bounded but nonzero:

- ingested artifacts prove what ReplayAB read, not that ReplayAB controlled their original producer;
- producer, environment, and workspace identity values still depend on the declared evidence producer trust boundary;
- R1 supports only exact artifacts and expected failure boundaries;
- semantic equivalence, complete repair trajectories, worker isolation, and stochastic conclusions remain outside R1.

## Validation before exact-head CI

Executed in the isolated implementation copy:

```text
21 passed in tests/test_workflow_replay_r1.py
Python compileall: PASS
legacy schema-v1 / paired-plan / fixture-run / exact-comparison smoke: PASS
production count: 400
```

The 21 focused checks include the 10 frozen calibration verdicts plus authority-digest, real failure-source, role-kind, invalid expected-state, report-digest, source-substitution, symlink/path, compatibility, determinism, and runtime checks.

Local Ruff was unavailable. Exact-head GitHub CI must run Ruff, full pytest, handoff authority, formal-channel, governance, and evidence-locator checks before R1 can close.

## Main-drift review

`main` advanced after the initial freeze through PR #119, which changed only RunSpec environment-prefix compatibility files. The R1 branch incorporated that exact current main without changing the frozen R1 contract, calibration inventory, expected verdicts, ReplayAB production scope, or scientific state.

## Closure rule

`GO_WITHIN_YELLOW` permits the implementation candidate to remain in Draft PR review. It does not grant C1, authorize merge, activate Candidate 01, or start R2. A failed frozen verdict, regression, identity/timing loophole, production growth beyond 400 lines, or scope expansion returns the decision to `REDESIGN`.
