# Candidate 01 C1 Adapter Yellow Review

Claim: `GOV-DEV-WORKFLOW-OPTIMIZATION-BENCHMARK-01`  
Base: `main@b65882993eaf674390989bb9082be2b79f1f1e44`  
Reviewed head: `3084033f896c01d5407d6421c9a5e59f81f8267c`  
Verdict: `GO_TO_EXACT_HEAD_CI_WITHIN_YELLOW`

## Budget

The contract metric is nonblank, non-comment changed Python lines in the two allowed production files. The implementation changes only `scripts/run_workflow_replay.py` and totals exactly 300 lines. `src/drpo/workflow_replay/execute.py` is unchanged. Any further production addition requires an equal or larger reduction; otherwise stop and redesign.

## Boundary

No new Python path exists. `evidence.py`, `orchestrate.py`, V1 owners, workflows, handoff, registry and science are unchanged. R1 remains the judge; the new CLI produces evidence. Candidate 01 remains the measured Arm-B treatment. Execution uses local Git objects and performs no network fetch.

## Limitations

- The adapter reuses frozen Candidate placement and payload-validation helpers. Independent frozen R1 truth remains mandatory; pair agreement alone is insufficient.
- Arm-B placement events preserve paths and counts but are recorded after `run_candidate` returns, so original intra-command placement timestamps are unavailable.
- Scope is only the C01 READY and C06 stale-main liveness cases. Full nine-case execution is not authorized.
- Local checks were Python compile, Ruff and local-Git helper smoke. Full repository pytest was not run locally because the runtime could not clone GitHub.
- Real C01/C06 liveness has not run. Unit tests and CI are not adoption evidence.

## Decision

Proceed only to Draft PR and exact-head CI. Do not proceed to real liveness, merge, adoption, default activation, R2 or scientific execution without later gates and approvals.