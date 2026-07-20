# ReplayAB R2 Semantic-Acceptance Gap Audit

Parent claim: `GOV-DEV-WORKFLOW-OPTIMIZATION-BENCHMARK-01`

R2 work ID: `REPLAYAB-R2-SEMANTIC-ACCEPTANCE-01`

Audit base: `main@b01b7715c8a703905a03403e54dfce258e69ba1e`

R1 status: `CLOSED`

Decision: `NARROW`

Scientific impact: none

## Finding

R2 can be implemented as a bounded extension of the closed R1 core. R1 already provides immutable run and evidence identity, digest and path validation, opposite-order pairing, invalid-execution detection, evidence-bound reports, and correctness-first efficiency release.

The missing capability is limited to:

1. a frozen `AcceptanceContract`;
2. an evidence-bound per-arm acceptance result;
3. independent A and B acceptance decisions;
4. a semantic pair report that allows efficiency comparison only when both arms are accepted.

Different implementation trees must be allowed in semantic mode. Schema-v2 R1 `exact_artifact` and `failure_boundary` behavior must remain unchanged.

## Frozen narrow design

The acceptance contract contains only:

- mandatory behavior IDs;
- forbidden-regression IDs;
- finite inclusive metric bounds;
- protected repository paths;
- evaluator SHA-256;
- evidence-schema SHA-256;
- deterministic opposite-order policy.

ReplayAB ingests immutable evaluator-result evidence and recomputes acceptance. It does not execute evaluator code.

A run is accepted only when execution is valid, all mandatory checks pass, no forbidden regression is detected, all metrics lie within their frozen bounds, protected paths pass, and all identities match.

Efficiency release requires both arms accepted plus matching report, run, evidence, and timing identities.

## Allowed paths

Production changes:

- `src/drpo/workflow_replay/evidence.py`;
- `src/drpo/workflow_replay/__init__.py` only if exports are needed.

Focused test changes:

- `tests/test_workflow_replay_compare.py`;
- `tests/test_workflow_replay_r1.py`.

Fixtures and calibration records may be added under:

- `tests/fixtures/workflow_replay/r2/**`;
- `docs/development_workflow_optimization/replayab_r2_calibration/**`.

No new Python file is authorized or required.

## Forbidden scope

R2 may not add evaluator execution, plugin registries, dynamic imports, sandboxing, services, network work, live workers, full repair trajectories, stochastic aggregation, Candidate 01 rules, V1 rules, authority changes, handoff changes, registry changes, or scientific changes.

## Budget

Changed production Python lines:

- preferred: `280--360`;
- yellow review: `361--450`;
- redesign trigger: `>450`;
- hard stop: `>600`.

No new dependency, Python file, network work, or Core-owned repository-wide scan.

## Minimum calibration

The frozen bank must cover:

- different implementations, both accepted;
- one arm missing a mandatory behavior;
- one arm containing a forbidden regression;
- both arms identically wrong;
- lower and upper tolerance boundaries;
- one tolerance violation;
- evaluator and contract digest mismatch;
- wrong run binding;
- protected-path failure;
- efficiency blocked for any rejected arm;
- complete R1 non-regression.

## Next action

Proceed under a separate R2 implementation contract. Any need for evaluator execution, workers, sandboxing, plugins, trajectories, or a broad R1 loader rewrite changes the verdict to `REDESIGN` and requires separate approval.
