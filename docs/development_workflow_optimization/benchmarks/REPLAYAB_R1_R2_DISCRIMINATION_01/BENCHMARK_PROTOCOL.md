# ReplayAB R1-versus-R2 Discrimination Benchmark Protocol

Parent claim: `GOV-DEV-WORKFLOW-OPTIMIZATION-BENCHMARK-01`

Work ID: `REPLAYAB-R1-R2-DISCRIMINATION-01`

Base: `main@b18aea9186d7e3ccc5d43b456719cafc23761e03`

Evidence target: `C2 -- independently accepted semantic replay`

Status: `PROTOCOL_FROZEN_NOT_RUN`

Scientific impact: none

## 1. Question under test

Determine whether ReplayAB R2 provides a controlled judge-level advantage over R1 on tasks that permit more than one correct implementation.

The primary question is not whether R2 is universally better. It is whether, on one frozen balanced bank:

1. R2 rejects fewer independently correct but non-canonical implementations than R1 exact-artifact mode;
2. R2 does not increase acceptance of independently incorrect implementations;
3. R2 attributes mixed pairs to the correct arm rather than reporting only non-equivalence;
4. R2 releases efficiency comparison for more truly comparable pairs without releasing it for an incorrect pair.

## 2. What this benchmark can and cannot prove

A passing result may support only this statement:

> On the frozen controlled discrimination bank, R2 is a strict judge-level capability extension over R1 exact-artifact mode: it preserves rejection of predeclared incorrect outcomes while reducing false rejection caused solely by implementation non-identity.

It does not establish:

- population-level error rates for arbitrary coding tasks;
- live coding-agent improvement;
- hidden evaluator isolation;
- first-attempt or repair-trajectory effects;
- stochastic repeated A/B effects;
- Candidate 01 time or quality improvement;
- replacement of R1 for deterministic exact-output tasks;
- R3, R4, R5, or R6 completion.

## 3. Compared judges

### Judge R1

Use the merged `exact_artifact` comparator. Each task family freezes one canonical path and artifact hash. An arm is classified as accepted only when no arm-specific R1 mismatch is emitted.

R1 pair efficiency is considered releasable only when the exact pair report is equivalent.

### Judge R2

Use the merged `AcceptanceContract`, primitive `AcceptanceResult` recomputation, and semantic pair comparator. Each arm is classified independently from mandatory behavior, forbidden regression, tolerance, protected-path, execution, and identity evidence.

R2 pair efficiency is considered releasable only when both arms are accepted and the semantic report is comparable.

The benchmark may not special-case expected case IDs inside either production comparator.

## 4. Independent ground truth

The case inventory freezes arm-level labels before benchmark implementation results:

- `correct`: the arm satisfies all predeclared semantic requirements;
- `incorrect`: the arm violates at least one predeclared mandatory, forbidden, tolerance, or protected-path condition.

The ground-truth labels are not inferred from R1 or R2 output. The benchmark implementation must derive R2 primitive acceptance from the frozen facts and must compare both judges against the same labels.

## 5. Case-bank design

The bank contains 16 pairs across four responsibility-derived task families:

1. API compatibility;
2. numeric tolerance;
3. protected-path safety;
4. serialization semantics.

Each family contributes exactly four pair classes:

- identical correct;
- different but both correct;
- one correct and one incorrect;
- both incorrect.

Canonical-arm orientation alternates across task families. Post-result case removal, relabeling, orientation changes, and threshold changes are forbidden.

The task families are controlled abstractions anchored to current repository responsibilities. They are not claimed to be full historical code re-executions.

## 6. Primary metrics

All metrics are computed from all retained cases.

### 6.1 Correct-arm false-rejection rate

`correct_arm_frr = rejected_ground_truth_correct_arms / ground_truth_correct_arms`

### 6.2 Incorrect-arm false-acceptance rate

`incorrect_arm_far = accepted_ground_truth_incorrect_arms / ground_truth_incorrect_arms`

### 6.3 Arm-label accuracy

Fraction of all arms whose accepted/rejected classification matches ground truth.

### 6.4 Pair-label accuracy

A pair prediction is correct only when both arm labels match ground truth.

### 6.5 Efficiency-release precision and coverage

A ground-truth efficiency-eligible pair is one in which both arms are correct.

- precision: released pairs that are truly eligible divided by all released pairs;
- coverage: truly eligible pairs released divided by all truly eligible pairs.

## 7. Frozen success gate

The benchmark decision is `PASS_CONTROLLED_ADVANTAGE` only when all conditions hold:

1. R2 correct-arm false-rejection rate is at least 0.15 lower than R1;
2. R2 incorrect-arm false-acceptance rate is exactly 0;
3. R2 incorrect-arm false-acceptance rate is not greater than R1;
4. R2 arm-label accuracy is 1.0;
5. R2 pair-label accuracy is 1.0;
6. R2 efficiency-release precision is 1.0;
7. R2 efficiency-release coverage exceeds R1 by at least 0.40;
8. R1 accepts every identical-correct control pair;
9. both judges reject every both-incorrect control arm;
10. the closed R1 terminal non-regression suite remains green.

Any false acceptance by R2 produces `FAIL_SAFETY`. A lower false-rejection rate accompanied by false acceptance is not an advantage.

## 8. Frozen expected aggregate counts

Before execution, the bank structure implies only these ground-truth denominators:

- pairs: 16;
- arms: 32;
- correct arms: 20;
- incorrect arms: 12;
- ground-truth efficiency-eligible pairs: 8.

Judge outputs and derived rates remain unobserved until execution.

## 9. Evidence set

The durable evidence set is limited to:

- `BENCHMARK_PROTOCOL.md`;
- `CASE_INVENTORY.yaml`;
- `EXPECTED_VERDICTS.yaml`;
- append-only `RAW_RESULTS.jsonl` after execution;
- derived `PAIRED_COMPARISON.json` after execution;
- reviewed `DECISION.md` after execution.

No database, service, dashboard, queue, scheduler, live worker, or new Python path is authorized.

## 10. Implementation boundary

Authorized implementation changes after this protocol-freeze commit:

- extend existing `tests/test_workflow_replay_r1.py` only;
- add non-Python fixtures only if strictly necessary under `tests/fixtures/workflow_replay/`;
- add the three post-run evidence files under this benchmark directory.

Forbidden:

- change R1 or R2 production semantics;
- add a Python file;
- execute an evaluator inside ReplayAB Core;
- alter the frozen inventory, labels, metrics, or success gate after results;
- use existing R2 calibration counts as the new benchmark result;
- modify handoff, registry, governance authority, scientific code, or scientific status.

## 11. Validation

Required before a decision:

```bash
python3 -m pytest tests/test_workflow_replay_r1.py -q
python3 -m pytest \
  tests/test_workflow_replay_model.py \
  tests/test_workflow_replay_execute.py \
  tests/test_workflow_replay_compare.py \
  tests/test_workflow_replay_r1.py \
  -q
python3 -m pytest -q
python3 -m ruff check .
python3 -m compileall -q src scripts tests
python3 scripts/handoff_authority.py verify --repo-root .
python3 scripts/validate_formal_execution_channel.py --repo-root .
python3 scripts/validate_governance_pipeline_stage_status.py --repo-root .
```

## 12. Relationship to R2 closure

This benchmark does not itself close R2 and does not rewrite the existing R2 closure review. A pass may be cited as additional controlled discrimination evidence. A fail requires diagnosis and blocks any stronger R2-advantage statement, but it does not silently redefine the already frozen R2 implementation contract.
