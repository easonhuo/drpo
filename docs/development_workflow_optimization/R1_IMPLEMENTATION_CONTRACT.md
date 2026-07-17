# ReplayAB R1 Narrow Implementation Contract

**Project:** DRPO A/B Replay Engine  
**Claim:** `GOV-DEV-WORKFLOW-OPTIMIZATION-BENCHMARK-01`  
**Stage:** narrowed R1 — artifact-first deterministic hardening  
**Base:** `main@2f677f4b00954ea71d0efa7def552a1ea3daa565`  
**Branch:** `dev/replayab-core-r1-exact-artifact-01`  
**Authorization:** explicit user approval after the merged `R1_GAP_AUDIT.md` verdict `NARROW`  
**Scientific impact:** none

## 1. Objective

Upgrade ReplayAB from C0 schema/fixture evidence to a bounded C1 deterministic ruler for
`exact_artifact` and `failure_boundary` cases.

R1 succeeds only when ReplayAB can ingest immutable real repository artifacts, bind every
verdict and timing record to exact run/evidence identities, reject adversarial mutations,
and reproduce all independently frozen calibration verdicts.

R1 does not claim that ReplayAB controlled the historical production of an ingested artifact.

## 2. Allowed paths

Behavior and exports:

- `src/drpo/workflow_replay/model.py`
- `src/drpo/workflow_replay/execute.py`
- `src/drpo/workflow_replay/compare.py`
- `src/drpo/workflow_replay/evidence.py`
- `src/drpo/workflow_replay/__init__.py`

Focused tests and calibration evidence:

- `tests/test_workflow_replay_r1.py`
- `tests/fixtures/workflow_replay/r1/**`
- `docs/development_workflow_optimization/replayab_r1_calibration/**`
- this contract

No other production path is authorized. `orchestrate.py`, Candidate 01, V1, paired-repair,
handoff authority, registry, workflows, scientific code, and default-route behavior remain unchanged.

## 3. Frozen implementation slice

The implementation may add only:

1. backward-compatible schema-v2 deterministic case contracts;
2. explicit `exact_artifact` and `failure_boundary` modes;
3. immutable `RunIdentity`, content-addressed `EvidenceLocator`, and deterministic Run Artifact;
4. strict evidence-root, path, symlink, size, digest, and identity validation;
5. append-only event-journal validation;
6. normalized outcome loading from immutable evidence;
7. balanced two-pair scheduling: `A -> B`, then `B -> A`;
8. execution-validity and evidence-bound pair reports;
9. efficiency release bound to the exact compared run and evidence identities;
10. the frozen calibration bank.

No generic backend registry, service, database, semantic evaluator, repair orchestrator,
Regeneration Runner, stochastic repetition engine, or Candidate 01 benchmark is allowed.

## 4. Frozen deterministic contracts

### 4.1 Case schema v2

Schema v1 remains readable and retains C0 compatibility status.

Schema v2 must explicitly freeze:

- comparison mode;
- expected exact artifact/tree identity;
- expected file modes when applicable;
- expected authority result;
- expected gate results;
- expected diagnostic and recovery classes for failure boundaries;
- workspace mutation rule;
- evaluator SHA-256;
- evidence-schema SHA-256;
- order policy `two_opposite_pairs`.

### 4.2 Run identity

A run identity contains:

- `case_id`;
- arm `A` or `B`;
- pair ID;
- repetition;
- order position;
- backend ID;
- deterministically derived run ID.

### 4.3 Evidence locator

A locator contains:

- evidence kind;
- repository/evidence-root-relative path;
- SHA-256;
- byte size.

Absolute paths, traversal, root escape, symlinks, size mismatch, digest mismatch, and duplicate
run identities must fail closed.

### 4.4 Run Artifact

The minimum Run Artifact binds:

- case-contract digest;
- run identity;
- base, input, toolchain, evaluator, environment, cache, backend, and plan identities;
- event and outcome locators;
- before/after workspace identities;
- execution terminal class;
- timing summary;
- producer/schema identity.

Execution terminal classes are limited to:

- `READY`;
- `BLOCKED`;
- `STALE`;
- `INTERRUPTED`;
- `INVALIDATED`.

Only the first three may enter exact outcome comparison. The latter two remain visible and
must block efficiency release.

## 5. Frozen calibration authority

The calibration inventory and expected verdicts are frozen in:

- `replayab_r1_calibration/INVENTORY.yaml`;
- `replayab_r1_calibration/EXPECTED_VERDICTS.yaml`.

The bank must include:

1. real committed READY artifact ingestion;
2. real committed failure-boundary input/artifact ingestion;
3. one-arm artifact/hash mismatch;
4. both-arms-same-wrong;
5. wrong file mode or unauthorized path;
6. interrupted run;
7. failure with changed workspace / partial mutation;
8. evidence digest or identity mismatch;
9. balanced-order schedule;
10. timing/report binding mismatch.

Expected verdicts are independent inputs. Generated ReplayAB reports may not rewrite them.

## 6. Complexity and runtime budgets

Production-code delta counting covers nonblank, non-comment changed Python lines in the allowed
production paths.

- preferred: `240--340`;
- yellow architecture review: `341--400`;
- hard redesign trigger: `>400`;
- no new third-party dependency;
- no Core network work;
- no Core-owned repository-wide scan.

Evidence limits:

- maximum one locator: `1 MiB`;
- maximum event journal: `1000` events and `1 MiB`;
- maximum one Run Artifact JSON: `256 KiB`;
- paths must stay under the declared evidence root.

Runtime targets on the frozen CI environment:

- contract validation, artifact loading, normalization, scheduling, and report generation:
  median `<=250 ms`, p95 `<=1 s`;
- Core self-overhead excluding child execution: median `<=1 s`;
- `>5 s`, duplicate validators/gates, or network work triggers redesign.

## 7. Validation plan

Focused:

```bash
python3 -m pytest \
  tests/test_workflow_replay_model.py \
  tests/test_workflow_replay_execute.py \
  tests/test_workflow_replay_compare.py \
  tests/test_workflow_replay_r1.py -q
```

Repository-wide:

```bash
python3 -m pytest -q
python3 -m ruff check .
python3 scripts/handoff_authority.py verify --repo-root .
python3 scripts/validate_formal_execution_channel.py --repo-root .
python3 scripts/validate_governance_pipeline_stage_status.py --repo-root .
```

R1 closes only when every frozen calibration verdict matches, exact-head CI passes, Candidate 01
and paired-repair after-images are unchanged, and the code/runtime budgets pass.

## 8. Risk controls

Known risks and controls:

- **schema regression:** preserve schema-v1 tests and behavior;
- **self-authored oracle bias:** expected verdicts are frozen before behavior code;
- **artifact substitution:** verify size and SHA-256 at load time;
- **path/symlink escape:** resolve under a declared root and reject symlinks;
- **two-arms-same-wrong:** compare each arm to the frozen contract before pair equivalence;
- **partial mutation:** derive from before/after workspace identity rather than trusting a caller flag;
- **timing mix-up:** bind efficiency records to report run IDs and evidence digests;
- **scope growth:** stop above 400 changed production lines or when a generic backend/service becomes necessary.

## 9. Rollback

Revert the R1 implementation PR. There is no database, migration, default-route activation,
or persistent mutable state to unwind. Historical R0 and audit documents remain provenance.

## 10. Authorization boundary

This contract authorizes only the bounded implementation and calibration described above.
It does not authorize merge, Candidate 01 evaluation, R2, Regeneration Runner, live coding-agent
A/B, scientific execution, or any result/status change.
