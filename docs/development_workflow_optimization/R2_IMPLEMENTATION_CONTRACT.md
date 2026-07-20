# ReplayAB R2 Narrow Implementation Contract

Parent claim: `GOV-DEV-WORKFLOW-OPTIMIZATION-BENCHMARK-01`

Work ID: `REPLAYAB-R2-SEMANTIC-ACCEPTANCE-01`

Base: `main@b01b7715c8a703905a03403e54dfce258e69ba1e`

Audit verdict: `NARROW`

Scientific impact: none

## Objective

Add bounded semantic acceptance so two different implementations may both be accepted under one frozen contract without requiring identical trees.

## Required objects

### AcceptanceContract

The contract must freeze:

- case ID;
- mandatory behavior IDs;
- forbidden-regression IDs;
- named inclusive numeric bounds;
- protected repository paths;
- evaluator SHA-256;
- evidence-schema SHA-256;
- opposite-order policy.

Lists and mappings must be strict, sorted, unique, and canonicalized into a contract SHA-256. Numeric bounds must be finite and at least one bound must be present per metric.

### AcceptanceResult

The immutable evaluator-result artifact must bind:

- case ID;
- run ID;
- acceptance-contract SHA-256;
- evaluator SHA-256;
- exact mandatory-result keys;
- exact forbidden-result keys;
- exact tolerance-value keys;
- protected-path result;
- diagnostics.

ReplayAB must derive acceptance from these primitive results. It must not trust a caller-provided final verdict.

### Semantic run and pair report

Each arm must expose execution validity and independent acceptance. Pair reporting must separate:

- A acceptance;
- B acceptance;
- acceptance pattern;
- pair comparability;
- efficiency-release permission;
- run, evidence, timing, and report identities.

Different output hashes or implementation trees are permitted when both arms satisfy the contract.

## Acceptance rules

An arm is accepted only if:

1. execution is valid;
2. all mandatory results are true;
3. all forbidden-regression detections are false;
4. all finite metric values lie within inclusive frozen bounds;
5. protected paths pass;
6. evaluator, contract, case, and run identities match.

Efficiency may be released only when both arms are accepted and all report bindings match.

## Compatibility

- Existing R1 schema-v2 validation remains unchanged.
- Existing R1 exact-artifact and failure-boundary loaders and reports remain unchanged.
- Existing R1 calibration and live C1 anchors must remain green.
- No semantic case may silently fall back to exact-artifact comparison.

## Allowed files

Production:

- `src/drpo/workflow_replay/evidence.py`;
- `src/drpo/workflow_replay/__init__.py` only for public exports.

Tests:

- `tests/test_workflow_replay_compare.py`;
- `tests/test_workflow_replay_r1.py`.

Non-Python fixtures and records:

- `tests/fixtures/workflow_replay/r2/**`;
- `docs/development_workflow_optimization/replayab_r2_calibration/**`;
- this contract and `R2_GAP_AUDIT.md`.

No new Python path is authorized.

## Forbidden work

Do not add evaluator execution, commands, subprocesses, plugins, dynamic imports, services, network work, sandboxing, live workers, repair trajectories, stochastic aggregation, Candidate 01 logic, V1 logic, authority changes, handoff changes, registry changes, or scientific changes.

## Budget

- preferred changed production Python lines: `280--360`;
- yellow review: `361--450`;
- redesign above `450`;
- hard stop above `600`;
- no new dependency;
- no new Python file;
- no Core network work;
- no Core repository-wide scan.

## Frozen calibration expectations

The bank must predeclare and test:

1. both different implementations accepted;
2. A accepted and B rejected for missing mandatory behavior;
3. A rejected for forbidden regression and B accepted;
4. both identically wrong and both rejected;
5. lower-bound equality accepted;
6. upper-bound equality accepted;
7. out-of-bound value rejected;
8. evaluator mismatch rejected;
9. contract mismatch rejected;
10. wrong run binding rejected;
11. protected-path failure rejected;
12. efficiency release blocked unless both accepted;
13. R1 full non-regression.

Expected verdicts must be frozen before implementation results.

## Validation

Focused:

```bash
python3 -m pytest tests/test_workflow_replay_compare.py tests/test_workflow_replay_r1.py -q
```

Repository-wide:

```bash
python3 -m pytest -q
python3 -m ruff check .
python3 scripts/handoff_authority.py verify --repo-root .
python3 scripts/validate_formal_execution_channel.py --repo-root .
python3 scripts/validate_governance_pipeline_stage_status.py --repo-root .
```

## Stop conditions

Stop and request redesign review if implementation requires:

- a new Python file;
- evaluator execution inside Core;
- a plugin or backend registry;
- worker isolation or trajectory capture;
- broad modification of the closed R1 loader;
- production Python above the frozen redesign threshold.

Passing implementation and calibration permit R2 closure review only. They do not authorize R3, live workers, stochastic A/B, Candidate 01 default adoption, or scientific execution.
