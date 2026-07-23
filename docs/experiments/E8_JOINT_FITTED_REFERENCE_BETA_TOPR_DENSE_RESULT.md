# E8 Joint Fitted-Reference beta-TOPR Dense Pilot Result

## 1. Identity and evidence level

- Experiment: `EXT-C-E8-ORACLE-OFFLINE-V2-JOINT-FITTED-REFERENCE-BETA-TOPR-DENSE-0.5B-01`
- RunSpec result: `E8_JOINT_FITTED_REFERENCE_BETA_TOPR_DENSE_20260723_01`
- Durable result repository commit: `712c36c5e858182de4e93c48dfe917bd42198c67`
- Reported local run commit: `3733ee28cc1517b67ad235afa10f2e855f2dde33`
- Evidence status: `pilot`
- Cells: `16/16`
- Failed cells: `0`
- NaN/Inf numerical failures: `0`
- Evaluation split: structurally disjoint held-out `val.jsonl`
- Test split used: `false`
- Terminal audit: `PASS`

The reported local run commit is not currently resolvable from the authoritative remote `drpo` repository. The protected source SHA-256 inventory and durable result package are therefore part of the required provenance record. This limitation prevents direct-commit provenance closure but does not erase the deposited trajectories.

This method is **Joint Fitted-Reference beta-TOPR**. The reference LoRA is trained jointly with a fixed `0.5/0.5` positive/negative branch target. The bank does not contain logged behavior-policy probabilities. The result must not be described as canonical frozen-behavior TOPR.

Countdown is an external-validity environment. This pilot does not replace controlled causal identification in C-U1, D-U1, or the nonlinear Gaussian environment.

## 2. Frozen protocol

The dense scan used:

```text
beta = [0, 0.01, 0.02, 0.04, 0.08, 0.125, 0.25, 0.5]
seed offsets = [4000, 5000]
steps = 1200
evaluation every = 100 steps
Pass@64 every = 200 steps
late window = [800, 900, 1000, 1100, 1200]
```

The negative weight was:

```text
exp(beta * min(sum_logpi - sum_logmu, 0))
```

where `mu` is the jointly fitted reference adapter. Best checkpoints are supplementary only. The paper-facing policy is fixed late-window plus terminal reporting.

## 3. Dense-scan results

All values below are held-out validation values. Percentages are displayed for readability.

| beta | late Pass@8 seed 4000 | late Pass@8 seed 5000 | late Pass@8 mean | terminal Pass@8 mean | terminal Pass@64 mean | terminal valid mean | best Pass@8 mean |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2.04% | 2.96% | 2.50% | 2.70% | 4.10% | 27.10% | 7.40% |
| 0.01 | 12.28% | 12.24% | 12.26% | 12.70% | 18.50% | 78.90% | 16.10% |
| 0.02 | 11.28% | 12.12% | 11.70% | 11.60% | 19.50% | 73.50% | 17.40% |
| 0.04 | 11.72% | 12.76% | 12.24% | 12.40% | 17.80% | 90.10% | 15.00% |
| 0.08 | 13.04% | 13.44% | 13.24% | 12.90% | 18.20% | 98.20% | 16.80% |
| 0.125 | 14.68% | 13.48% | 14.08% | 14.60% | 20.10% | 99.40% | 16.50% |
| 0.25 | 13.68% | 13.72% | 13.70% | 14.00% | 20.70% | 99.30% | 18.50% |
| 0.5 | 14.40% | 14.48% | 14.44% | 14.60% | 22.20% | 99.10% | 18.90% |

The aggregate source is:

```text
runs/e8/E8_JOINT_FITTED_REFERENCE_BETA_TOPR_DENSE_20260723_01/
files/outputs/e8/joint_fitted_reference_beta_topr_dense_001/
aggregate/per_cell_summary.csv
```

The late-window values are the arithmetic mean of the five registered checkpoints, not best-checkpoint values.

## 4. Repeated anchors versus the predecessor scan

The predecessor result is bound to `drpo-results` commit `68ea4980ed9c8ebb79e02f7d2b40a7e2a8ee0461`. The dense scan repeated beta `0`, `0.25`, and `0.5` as internal anchors.

| beta | predecessor late Pass@8 | dense late Pass@8 | change | predecessor terminal Pass@8 | dense terminal Pass@8 | change | predecessor terminal valid | dense terminal valid |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2.12% | 2.50% | +0.38 pp | 2.40% | 2.70% | +0.30 pp | 25.20% | 27.10% |
| 0.25 | 15.12% | 13.70% | -1.42 pp | 16.50% | 14.00% | -2.50 pp | 99.50% | 99.30% |
| 0.5 | 13.58% | 14.44% | +0.86 pp | 13.30% | 14.60% | +1.30 pp | 99.40% | 99.10% |

The repeated evidence supports two robust qualitative findings:

1. beta `0` repeatedly produces severe task-performance and valid-expression degradation without NaN/Inf;
2. positive beta values enter a broad high-validity performance plateau.

The point ordering is not robust. Beta `0.25` was stronger in the predecessor scan, while beta `0.5` is stronger on the dense scan's registered late-window metric. The evidence therefore does not support a sharp or universal optimum.

## 5. Task, structure, and numerical events

The three event classes must remain separate.

### 5.1 Task-performance degradation

Beta `0` has late-window Pass@8 `2.50%`, compared with approximately `12%--14%` for positive beta values. This task-performance degradation reproduced across both scans.

### 5.2 Valid-expression or structure degradation

Terminal valid rate changes in stages:

```text
beta 0      -> 27.10%
beta 0.01   -> 78.90%
beta 0.02   -> 73.50%
beta 0.04   -> 90.10%
beta 0.08   -> 98.20%
beta 0.125+ -> approximately 99%
```

A formal support boundary was not registered or instrumented. `valid_rate` is therefore a structure/output-validity diagnostic, not a formal support-recovery claim.

### 5.3 NaN/Inf numerical failure

No NaN/Inf numerical failure was observed in any of the 16 cells. The beta `0` failure is a task and output-structure collapse, not a NaN/Inf collapse.

## 6. Mechanism-facing diagnostics

The terminal diagnostics show a threshold-like control transition.

For seed offset `4000`:

| beta | terminal mean log ratio | terminal median weight | terminal raw policy-gradient norm | terminal valid rate |
|---:|---:|---:|---:|---:|
| 0 | -2591.00 | 1.0 | 46.00 | 28.8% |
| 0.01 | -1504.39 | 3.44e-8 | 259.54 | 77.8% |
| 0.02 | -516.27 | 9.08e-6 | 151.45 | 67.6% |
| 0.04 | -283.35 | 5.61e-6 | 5.13 | 88.2% |
| 0.08 | -162.03 | 1.14e-6 | 6.21 | 97.8% |
| 0.125 | -125.37 | 1.62e-7 | 5.18 | 99.2% |
| 0.25 | -85.06 | 3.93e-10 | 5.24 | 99.6% |
| 0.5 | -65.47 | 6.93e-14 | 4.75 | 99.8% |

These diagnostics support the following bounded interpretation:

- no ratio taper continues full-strength repulsion even after the negative branch is extremely far from the fitted reference;
- very small positive beta values recover much of the task metric, but beta `0.01--0.02` can still show unstable output validity and large trajectory-dependent gradients;
- the main structure-stability transition lies approximately between beta `0.04` and `0.08` in this pilot;
- beta `0.08--0.5` forms a broad stable plateau rather than a clearly ordered narrow optimum.

This is a closed-loop fitted-reference result. Terminal weight means are not expected to be monotone in beta because changing beta changes the learned policy trajectory and hence the log-ratio distribution itself.

## 7. Horizon interpretation

The fixed 1200-step horizon is not convergence evidence.

The predecessor beta `0.25` trajectories ended at their local best values, which initially suggested terminal truncation risk. That pattern did not robustly repeat in the dense scan. Across positive beta values, the registered step `800--1200` trajectories look more like a noisy broad plateau than a method-wide sustained rise.

The allowed statement is:

> The 1200-step budget is insufficient to prove convergence or saturation, but the stable positive-beta region does not exhibit a robust method-wide continuing ascent at the terminal checkpoint.

No steady-state ranking is allowed.

## 8. Closure decision and frozen comparison configuration

Scientific exploration on this TOPR beta line is closed. No further large beta scan or local decimal-point search is required.

For the later fair comparison against the main model, freeze:

```text
method = joint_fitted_reference_topr
beta = 0.5
selection evidence = highest dense-scan registered late-window Pass@8 mean
selection split = held-out validation only
future retuning after main-model or test observation = forbidden
```

Beta `0.5` is selected because it has the largest dense-scan late-window Pass@8 mean (`14.44%`) and remains in the approximately `99%` valid-rate plateau. This does **not** establish that beta `0.5` is significantly, universally, or mechanistically optimal. It is a transparent frozen baseline-selection rule.

A future final comparison may evaluate this frozen configuration once under the same budget, seeds, checkpoint policy, and evaluation protocol as the main model. It must not reopen beta tuning.

## 9. Prohibited claims

This pilot does not establish:

- canonical frozen-behavior TOPR;
- logged behavior-policy correction;
- convergence, saturation, or steady state;
- a statistically significant best beta;
- formal support recovery;
- a formal cross-method or cross-task ranking;
- replacement of controlled causal evidence by Countdown.

Detailed provenance limitations are recorded in `E8_JOINT_FITTED_REFERENCE_BETA_TOPR_DENSE_PROVENANCE_AUDIT.json`.
