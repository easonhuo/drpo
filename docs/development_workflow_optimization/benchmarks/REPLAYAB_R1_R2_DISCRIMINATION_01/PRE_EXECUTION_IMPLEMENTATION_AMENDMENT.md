# ReplayAB R1-versus-R2 Discrimination Benchmark Pre-execution Amendment

Parent claim: `GOV-DEV-WORKFLOW-OPTIMIZATION-BENCHMARK-01`

Work ID: `REPLAYAB-R1-R2-DISCRIMINATION-01`

Base protocol commit: `198fca463949c5de5b2e9aa40618ff7a7f4d6257`

Date: 2026-07-21

Status: `APPROVED_BEFORE_EXECUTION`

## Exact approval

After the benchmark protocol, 16-case inventory, independent labels, metrics, and success gate were frozen—and before any judge output or aggregate result was generated—the user explicitly approved creation of exactly this new Python test path:

`tests/test_workflow_replay_r1_r2_discrimination.py`

Its sole responsibility is to execute and verify the frozen R1-versus-R2 controlled discrimination benchmark using the already merged ReplayAB comparators and contracts.

## Narrow supersession

This amendment supersedes only the protocol statements that prohibited a new Python path or required extension of `tests/test_workflow_replay_r1.py`.

The approved test path may:

- load the frozen inventory and expected verdicts;
- construct controlled outcome evidence from the frozen facts;
- invoke the merged R1 exact-artifact and R2 semantic-acceptance judges;
- calculate the frozen arm, pair, and efficiency-release metrics;
- verify `RAW_RESULTS.jsonl`, `PAIRED_COMPARISON.json`, and `DECISION.md` after execution.

It may not:

- change R1 or R2 production semantics;
- special-case a case ID inside production code;
- alter the frozen cases, labels, orientation, metrics, or success gate;
- execute a live coding agent;
- modify handoff, registry, governance authority, scientific code, or scientific status;
- create any other Python path.

All other protocol boundaries remain unchanged. This amendment does not predeclare a favorable result and does not authorize merge or R2 closure.
