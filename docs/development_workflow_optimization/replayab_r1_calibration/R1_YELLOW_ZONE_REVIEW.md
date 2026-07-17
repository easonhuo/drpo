# ReplayAB R1 Yellow-Zone Architecture Review

**Claim:** `GOV-DEV-WORKFLOW-OPTIMIZATION-BENCHMARK-01`  
**Base:** `main@2f677f4b00954ea71d0efa7def552a1ea3daa565`  
**Branch:** `dev/replayab-core-r1-exact-artifact-01`  
**Decision:** `GO_WITHIN_YELLOW`  
**R1 status:** implementation candidate; C1 not yet granted

## Measured size

The narrowed R1 production delta is one new module:

- `src/drpo/workflow_replay/evidence.py`: **399 nonblank, non-comment lines**.

No existing production module is modified. This is inside the frozen `341--400` yellow review range and does not exceed the `>400` hard redesign trigger.

## Why the yellow-zone implementation is retained

The module contains one cohesive deterministic evidence boundary:

1. schema-v2 R1 contract validation while delegating unchanged base fields to the existing schema-v1 validator;
2. deterministic run identity and two-opposite-pair scheduling;
3. content-addressed evidence locators with root, traversal, symlink, size, and digest checks;
4. Run Artifact normalization, terminal/journal consistency, workspace-mutation derivation, and real result-artifact verification;
5. strict per-arm comparison against the frozen contract before pair equivalence;
6. pair-report, Run Artifact, evidence, and timing binding.

Removing any of these responsibilities would leave one of the pre-frozen R1 calibration verdicts unimplemented. Moving them into `model.py`, `execute.py`, or `compare.py` would not reduce total production churn and would increase regression exposure to the accepted C0 paths.

## Smaller alternatives reviewed and rejected

- **Keep caller-built `OutcomeSnapshot`:** rejected because it does not prove artifact ingestion or evidence identity.
- **Trust locator metadata without reading files:** rejected because digest/size substitution remains possible.
- **Bind run IDs but not timing:** rejected after review found that a caller could attach arbitrary timing to otherwise valid identities.
- **Use only pair equality:** rejected because both arms can be identically wrong.
- **Import paired-repair judgment logic:** rejected because it is a separate governance workflow and is not randomized A/B.
- **Build a generic backend framework:** rejected as out of R1 scope and a hard redesign condition.
- **Modify existing C0 modules to save a file:** rejected because total logic does not shrink and compatibility risk increases.

## Risk review

Remaining risks are bounded:

- ingested artifacts prove what ReplayAB read, not that ReplayAB controlled their original producer;
- producer, environment, and workspace identities remain claims whose trust boundary must be stated;
- R1 supports only exact artifacts and expected failure boundaries;
- semantic equivalence, complete repair trajectories, worker isolation, and stochastic conclusions remain outside R1.

These limitations prevent an overclaim but do not invalidate the C1 deterministic target.

## Pre-PR validation

Executed in the isolated implementation copy:

```text
15 passed in tests/test_workflow_replay_r1.py
Python compileall: PASS
legacy schema-v1 / paired-plan / fixture-run / exact-comparison smoke: PASS
```

Local Ruff was not available. Exact-head GitHub CI must run Ruff, full pytest, authority, formal-channel, and governance checks before R1 can close.

## Closure rule

`GO_WITHIN_YELLOW` permits the implementation candidate to enter Draft PR review. It does not grant C1, authorize merge, activate Candidate 01, or start R2. Any production addition beyond 400 lines, failed calibration verdict, regression, or identity/timing loophole returns the decision to `REDESIGN`.
