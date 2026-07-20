# ReplayAB R1 Formal Calibration Execution Plan

Claim: `GOV-DEV-WORKFLOW-OPTIMIZATION-BENCHMARK-01`  
Calibration: `REPLAYAB-R1-C1-CALIBRATION-01`  
Frozen implementation/toolchain: `main@dd46727c1efefd2e6d4cdf6f3b204ec1fc58fca3`  
Execution branch: `exec/replayab-r1-calibration-01`  
Scientific impact: none

## Purpose

Execute the already frozen R1 deterministic calibration bank against the exact merged
ReplayAB toolchain. This run calibrates the ruler; it does not rank scientific methods,
adopt Candidate 01 as a default, start R2, or modify V1, authority, handoff, registry, or
scientific code.

## Frozen authority

The following existing files define the calibration before this execution:

- `INVENTORY.yaml`: frozen ten-case inventory and no-removal rule;
- `EXPECTED_VERDICTS.yaml`: independently predeclared expected verdicts;
- `R1_IMPLEMENTATION_CONTRACT.md`: evidence and runtime contract;
- `tests/test_workflow_replay_r1.py`: executable calibration implementation;
- `src/drpo/workflow_replay/evidence.py`: R1 evidence loader and comparator.

No case may be removed, reclassified, or assigned a different expected verdict after
results are observed. Failed, skipped, interrupted, and invalid executions remain visible.

## Calibration matrix

| Case | Expected verdict | Efficiency release |
|---|---|---|
| `R1-CAL-READY-EXACT` | equivalent | allowed |
| `R1-CAL-FAILURE-BOUNDARY` | equivalent expected stop | allowed |
| `R1-CAL-ONE-ARM-HASH-MISMATCH` | reject B | blocked |
| `R1-CAL-BOTH-SAME-WRONG` | reject both | blocked |
| `R1-CAL-WRONG-MODE` | reject wrong mode | blocked |
| `R1-CAL-INTERRUPTED` | execution invalid | blocked |
| `R1-CAL-PARTIAL-MUTATION` | invalid partial mutation | blocked |
| `R1-CAL-EVIDENCE-DIGEST-MISMATCH` | evidence invalid | blocked |
| `R1-CAL-ORDER-BALANCE` | exactly A→B then B→A | not applicable |
| `R1-CAL-TIMING-BINDING-MISMATCH` | efficiency remains blocked | blocked |

## Execution protocol

1. Check out this execution branch only to obtain the execution plan and workflow.
2. Create a detached worktree at the exact frozen toolchain SHA above.
3. Verify that branch-only changes are limited to this plan and the temporary workflow.
4. Install the repository development dependencies from the detached toolchain.
5. Run the complete `tests/test_workflow_replay_r1.py` module without deselection.
6. Produce JUnit, stdout, environment, digest, and structured verdict reports.
7. Upload all reports even when pytest fails.

The existing post-merge C06 run `29693652154` and READY A/B run `29693991085` are
supplemental live anchors. They are not substituted for any of the ten frozen calibration
cases and are not rerun by this execution plan.

## Frozen acceptance and guardrails

R1 is eligible for closure review only when all conditions hold:

- verdict agreement is `10/10`;
- covered unsafe-pass count is `0`;
- covered false-rejection count is `0`;
- all evidence identity, journal, subject, and efficiency-binding checks pass;
- the existing runtime guardrail test passes: median load/compare time no more than
  `0.250 s` and p95 no more than `1.000 s` over its frozen 100-iteration procedure;
- no R1 production file differs from the frozen toolchain;
- full failures and diagnostics are retained.

A passing run means `READY_FOR_R1_CLOSURE_REVIEW`, not automatic R1 closure. Closure,
merging any corrective code, or starting R2 remains separately gated.

## Stop conditions

Stop and retain evidence if any case disagrees with its predeclared verdict, if a runtime
guardrail fails, or if satisfying calibration would require modifying V1, Stage 5
authority, handoff, registry, scientific code, or a frozen expected verdict.
