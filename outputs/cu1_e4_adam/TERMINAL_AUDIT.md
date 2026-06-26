# C-U1 E4 unified-Adam terminal audit

- Experiment ID: `C-U1-E4-ADAM-RERUN`
- Base/run commit: `d699bb6b1d0093d8a9b935fd6c67f049fc3c3df0`
- Integrity/provenance gates: **PASS**
- Scientific terminal acceptance for the full stable-extrapolation claim: **FAIL / partial evidence only**
- Scientific status: **finite-step validated（有限训练步数验证）**
- Terminology: same-distribution held-out-context generalization; not OOD

## Why the full terminal gate did not pass

The finite-horizon reward curve is strong, but the registered stationary gate requires both full-data residual audits to pass. No beneficial alpha passes that gate in 20/20 seeds. Fixed-variance `alpha=1.00` reaches reward about `0.9917`, but only 3/20 seeds pass both residual audits; therefore this result cannot be called a terminally stable fixed point.

## Fixed-variance phase scan

| alpha | Reward mean [95% CI] | Normalized displacement | Task collapse | Both residual audits | Terminal interpretation |
|---:|---:|---:|---:|---:|---|
| 0.00 | 0.646988 [0.646228, 0.647727] | 0.000749 | 0/20 | 0/20 | finite_continuing_drift_or_runaway:20 |
| 0.25 | 0.714408 [0.713818, 0.715007] | 0.122173 | 0/20 | 0/20 | finite_continuing_drift_or_runaway:20 |
| 0.50 | 0.804028 [0.803322, 0.804756] | 0.294212 | 0/20 | 0/20 | finite_continuing_drift_or_runaway:20 |
| 0.75 | 0.915310 [0.914907, 0.915704] | 0.556315 | 0/20 | 1/20 | finite_continuing_drift_or_runaway:19, stable_beneficial_extrapolation:1 |
| 1.00 | 0.991703 [0.991363, 0.992035] | 1.007808 | 0/20 | 3/20 | finite_continuing_drift_or_runaway:17, stable_beneficial_extrapolation:3 |
| 1.25 | 0.650072 [0.648429, 0.651699] | 1.963490 | 0/20 | 8/20 | finite_continuing_drift_or_runaway:12, stable_over_extrapolated_fixed_point:8 |
| 1.50 | 0.000537 [0.000499, 0.000576] | 5.325676 | 20/20 | 15/20 | finite_continuing_drift_or_runaway:5, stable_bad_fixed_point:15 |
| 1.75 | 0.000000 [0.000000, 0.000000] | 523331.290625 | 20/20 | 0/20 | finite_continuing_drift_or_runaway:20 |

Finite-horizon paired evidence is reproducible: `alpha=0.25`, `0.50`, `0.75`, and `1.00` each outperform `alpha=0` in 20/20 paired seeds. Excessive pressure reverses the gain: `alpha=1.50` and `1.75` are below the registered task threshold in 20/20 seeds, while all parameters remain finite.

## Learnable-variance phase scan

| alpha | Reward at event/terminal [95% CI] | Support contraction | Median onset | Unexpected expansion | NaN/Inf |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.646997 [0.646204, 0.647727] | 0/20 | — | 0/20 | 0/20 |
| 0.10 | 0.671650 [0.671094, 0.672178] | 0/20 | — | 0/20 | 0/20 |
| 0.20 | 0.699399 [0.698854, 0.699963] | 0/20 | — | 0/20 | 0/20 |
| 0.30 | 0.730730 [0.730181, 0.731297] | 0/20 | — | 0/20 | 0/20 |
| 0.35 | 0.747854 [0.747279, 0.748448] | 0/20 | — | 0/20 | 0/20 |
| 0.38 | 0.759039 [0.758043, 0.759992] | 0/20 | — | 0/20 | 0/20 |
| 0.40 | 0.756429 [0.748028, 0.764426] | 18/20 | 434.5 | 0/20 | 0/20 |
| 0.50 | 0.773876 [0.762808, 0.783437] | 20/20 | 83.0 | 0/20 | 0/20 |

The robustness audit confirms that this is support contraction rather than a single arbitrary cutoff: at `alpha=0.50`, all 15 development-seed/learning-rate runs cross log-sigma thresholds `-8`, `-10`, `-12`, and `-14`; at `alpha=0.38`, none of the 15 runs crosses `-8`. No positive-boundary event or nonfinite parameter occurs.

## Long-run controls (4000 Adam steps)

| Method | Reward mean [95% CI] | Task failure | Nonfinite |
|---|---:|---:|---:|
| uncontrolled_all | 0.000000 [0.000000, 0.000000] | 20/20 | 0/20 |
| far_cap | 0.995224 [0.995023, 0.995416] | 0/20 | 0/20 |
| budget_matched_global | 0.502925 [0.501994, 0.503900] | 0/20 | 0/20 |

These controls show that far-field capping can rescue the registered controlled environment, whereas the raw-gradient budget-matched global control does not match the same held-out reward. This is an appendix control, not a pre-registered universal method ranking; Adam parameter-update budgets were not matched.

## Gate summary

- PASS — all_expected_raw_files_exist: missing=0
- PASS — fixed_rows_160: 160
- PASS — learnable_rows_160: 160
- PASS — control_rows_60: 60
- PASS — variance_robustness_rows_45: 45
- PASS — fixed_alpha_grid_exact: [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
- PASS — learnable_alpha_grid_exact: [0.0, 0.1, 0.2, 0.3, 0.35, 0.38, 0.4, 0.5]
- PASS — all_optimizer_fields_adam: True
- PASS — reference_regression_passed: True
- PASS — environment_audit_passed: True
- PASS — preflight_self_tests_passed: True
- PASS — no_unexpected_support_expansion: 0
- PASS — no_nan_inf_numerical_failure: True
- PASS — finite_horizon_benefit_beyond_positive_only_ceiling: fixed alpha 0.25/0.50/0.75/1.00 each beats alpha=0 in 20/20 paired seeds
- FAIL — beneficial_branch_terminal_stationarity: No beneficial alpha passes both registered residual audits in 20/20 seeds; alpha=1.00 passes 3/20.
- PASS — excessive_fixed_pressure_task_collapse: alpha=1.50 20/20; alpha=1.75 20/20
- PASS — learnable_variance_support_contraction: alpha=0.40 18/20; alpha=0.50 20/20
- PASS — variance_boundary_not_single_threshold_artifact: alpha=0.50 crosses log-sigma -8/-10/-12/-14 in 15/15 development-seed/lr runs; alpha=0.38 crosses none.
- PASS — far_control_long_run_rescue_without_universal_ranking_claim: uncontrolled_all failure 20/20; far_cap 0/20

## Decision

E4 is accepted as **finite-step validated**, not long-run validated. The phase-transition and failure-side evidence is strong, but the beneficial branch does not satisfy the frozen stationary residual gate. `C-U1-E4-TAPER-01` remains blocked until a convergence-resolution protocol is separately registered and approved, or the paper explicitly limits E4 to finite-horizon evidence.
