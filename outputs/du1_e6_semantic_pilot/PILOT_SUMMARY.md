# D-U1 E6 shared-semantic pilot review

**Experiment:** `D-U1-E6-SEMANTIC-PILOT-01`
**Status:** pilot / raw complete / terminal audited; **not** a formal long-run result.
**Expected GitHub commit:** `e8b62dde518f593ff8325c7da94c41406311ca45`
**Execution:** CPU, development seeds `0--4`, 105/105 runs, 2000 steps per run, exit code 0.

## Integrity

- Environment invariants passed for aligned and shuffled catalogues across all five development seeds.
- 105/105 method-seed runs completed.
- NaN/Inf numerical failures: **0/105**.
- Support/temperature-boundary events: **56/105**.
- Task-performance collapse events under the preregistered paired-positive-only threshold: **0/105**.
- Formal 2x extension was not performed; method ranking remains forbidden.

## Development observations

### Protocol A: fixed concentration, local-negative alpha scan

| alpha | reward mean | hidden-optimal probability | delta vs positive-only | normalized extrapolation | support events | plateau seeds |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.856817 | 0.144089 | +0.000000 | -0.178619 | 0/5 | 0/5 |
| 0.10 | 0.868275 | 0.165712 | +0.021623 | -0.004493 | 0/5 | 0/5 |
| 0.25 | 0.882730 | 0.198287 | +0.054199 | 0.320618 | 0/5 | 0/5 |
| 0.50 | 0.888228 | 0.216125 | +0.072037 | 0.981432 | 0/5 | 0/5 |
| 0.75 | 0.849997 | 0.145232 | +0.001143 | 1.619084 | 0/5 | 0/5 |
| 1.00 | 0.760937 | 0.053274 | -0.090814 | 2.075849 | 0/5 | 0/5 |

The scanned development peak occurs at `alpha=0.5`: reward and hidden-optimal probability both improve relative to positive-only, while `alpha=0.75` and `1.0` show over-extrapolation and performance reversal. This is only a pilot observation. Every Protocol-A branch remains `persistent_drift_or_inconclusive` at 2000 steps, so the current horizon cannot be frozen as a converged formal horizon.

### Protocol B: learnable concentration and near/far controls

| method | reward mean | hidden-optimal probability | effective support | concentration mean | support events |
|---|---:|---:|---:|---:|---:|
| positive_only | 0.867217 | 0.154083 | 7.810 | 9.807 | 0/5 |
| local_only | 0.939141 | 0.473215 | 1.280 | 155.287 | 5/5 |
| uncontrolled | 0.934280 | 0.401783 | 1.270 | 153.984 | 5/5 |
| near_zero | 0.911884 | 0.235886 | 1.271 | 155.190 | 5/5 |
| far_zero | 0.939141 | 0.473215 | 1.280 | 155.287 | 5/5 |
| far_cap | 0.939511 | 0.453791 | 1.322 | 140.480 | 5/5 |
| budget_matched_global | 0.924146 | 0.321844 | 1.316 | 134.273 | 5/5 |

Positive-only avoids the registered support boundary. Every aligned branch containing negative pressure reaches the support/temperature boundary in 5/5 seeds, usually between steps 200 and 400. Reward can remain high while support collapses, directly confirming that task performance and support-boundary events must be reported separately. The current learnable-concentration setting (`alpha=0.5`, far pressure `1.0`) is therefore unsuitable for unchanged formal freezing.

### Protocol C: semantic alignment control

The aligned catalogue gives much larger hidden-optimal probability and reward than the policy-side shuffled catalogue for every method. This supports the intended shared-semantic mechanism at pilot level: the apparent movement toward the hidden optimum depends on the policy embedding being aligned with reward semantics. It is not an OOD result and does not yet establish a formal method ranking.

## Gate decision

`D-U1-E6-SEMANTIC-LONGRUN-01` remains blocked. The pilot does **not** justify automatic parameter freezing because:

1. no fixed-concentration branch reached the preregistered provisional plateau within 2000 steps;
2. the current learnable-concentration negative branches overwhelmingly hit support boundaries;
3. the formal 2x terminal extension and untouched held-out seeds have not been executed.

The next scientifically clean step is a focused development extension using only the already registered variables: extend the fixed-concentration candidate region around `alpha in {{0.25, 0.5, 0.75}}`, and re-pilot lower negative-pressure settings for the learnable-concentration branch before freezing the formal method matrix, horizon, thresholds, and held-out seeds. No new variable or regularizer is introduced by this recommendation.
