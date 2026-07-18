# Candidate 01 C1 Adapter Refinement

Claim: `GOV-DEV-WORKFLOW-OPTIMIZATION-BENCHMARK-01`  
Base: `main@b65882993eaf674390989bb9082be2b79f1f1e44`  
Scientific impact: none

Code inspection fixes four implementation assumptions:

1. Real runs use a workspace checked out at the frozen ReplayAB toolchain SHA. Historical main/dev commits are exposed as run-local local-Git refs to the existing V1 commands. An old-main checkout cannot run the current Candidate and preparation entrypoints.
2. V1 `plan` reports a stale-main SHA mismatch as terminal `BLOCKED`, diagnostic `SOURCE_DRIFT`, phase `source_lock`. The case remains a stale-main failure-boundary test; the adapter must not relabel the actual terminal as `STALE`.
3. Request/review placement in the tool workspace is input staging, not protected-target mutation. Failure-boundary equality is evaluated on the source/integration target. A changed transaction integration repository or false READY remains invalid.
4. Volatile commit and report timestamps remain in the append-only journal. Exact-artifact truth uses deterministic tree, changed paths, modes, authority and required-gate states.

Boundary remains unchanged: no new Python file; modify only `scripts/run_workflow_replay.py` and, if needed, `src/drpo/workflow_replay/execute.py`; do not modify Candidate 01, R1 judgment, V1 owners, workflow, handoff, registry or science; production change cap is 300 nonblank non-comment lines; exact-head CI precedes real liveness.