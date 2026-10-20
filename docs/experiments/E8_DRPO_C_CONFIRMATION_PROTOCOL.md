# E8 DRPO c-confirmation protocol

Experiment ID: `EXT-C-E8-ORACLE-OFFLINE-V2-DRPO-C-CONFIRMATION-0.5B-01`

## Question

Among the four coefficients already motivated by completed development pilots,

\[
c\in\{1.897119985,3,5,8\},
\]

which coefficient has the strongest mean late-window Pass@8 on four new paired development seed offsets under an otherwise unchanged paper-aligned DRPO training pipeline?

This protocol confirms a DRPO hyperparameter. It does not compare DRPO formally against AsymRE or TOPR, does not access an independent test split, and does not establish convergence.

## Fixed execution matrix

| coefficient | seed 17000 | seed 18000 | seed 19000 | seed 20000 |
|---:|:---:|:---:|:---:|:---:|
| 1.897119985 | run | run | run | run |
| 3.0 | run | run | run | run |
| 5.0 | run | run | run | run |
| 8.0 | run | run | run | run |

Total: 16 cells.

The four seed offsets must be shared across all four coefficients. A failed cell is preserved and reported; it is not silently replaced by another seed.

## Frozen implementation and data

The run must reuse:

- `src/drpo/countdown_e8_alpha1_c_scan_trainer.py`;
- `src/drpo/countdown_e8_alpha1_c_scan_runtime.py`;
- `src/drpo/countdown_e8_alpha1_highc_scan_runtime.py`;
- `scripts/run_countdown_e8_oracle_offline_v2_alpha1_highc_scan_auto.py`;
- `scripts/run_countdown_e8_oracle_offline_v2_alpha1_highc_scan_auto_one_click.sh`;
- the same Qwen2.5-0.5B model identity, offline bank, base config, structurally disjoint held-out validation split, LoRA parameterization, optimizer, denominator, and evaluation implementation used by the paper-aligned linear scan.

The negative weight remains

\[
w=\exp(-cu),\qquad u=\frac{\text{current sequence surprisal}}{2},
\]

with detached remoteness and `alpha=1.0`. No extra square, hidden negative scale, gradient-budget matching, or weight-sum normalization is allowed.

## Liveness gate

Before the full matrix, run the existing two-step liveness path with representative `c=5.0` and eight validation examples. The gate must verify:

- model, bank, validation, base-config, and grid-config identity;
- successful forward/backward/update execution;
- terminal step equals 2;
- no numerical failure;
- no test data used.

Liveness passing is implementation evidence only.

## Training and evaluation

- fixed horizon: 1200 steps;
- early stopping: disabled;
- Greedy and Pass@8 evaluation: every 100 steps;
- Pass@64 evaluation: every 200 steps;
- late window: steps 800, 900, 1000, 1100, 1200;
- test split: forbidden.

For coefficient `c` and seed `s`, compute

\[
L_{c,s}=\frac{1}{5}\sum_{t\in\{800,900,1000,1100,1200\}}
\operatorname{Pass@8}_{c,s,t}.
\]

The primary aggregate is the mean of the four seed-level values `L_{c,s}`. The statistical sample count is four seeds, not twenty checkpoints.

## Decision rule

1. Complete or terminally account for all 16 cells.
2. Report every seed-level late-window Pass@8, terminal Pass@8, late-window Greedy, late-window valid-expression rate, and numerical status.
3. Rank the four coefficients by mean seed-level late-window Pass@8.
4. Use terminal Pass@8 and validity only as secondary diagnostics; best-checkpoint values remain supplementary.
5. Freeze no coefficient until terminal audit and durable result delivery are complete.
6. Do not change the candidate set, seed set, horizon, window, or primary metric after observing results.

Prior results on seeds `4000`, `5000`, `9000`, `10000`, `11000`, and `12000` are contextual evidence only and are not pooled into this confirmation aggregate.

## Terminal audit and reporting separation

The audit must separately report:

- task-performance outcome: Greedy, Pass@8, and Pass@64 trajectories;
- structure/validity outcome: valid-expression trajectory and any registered structure event;
- numerical outcome: NaN/Inf, failed updates, missing checkpoints, and process failures.

A low-performing finite trajectory, including a recurrence similar to historical seed `11000`, is not a numerical failure and may not be excluded post hoc.

The fixed 1200-step horizon may be described only as a finite-step pilot. It must not be called convergence, steady state, significance, or a formal method ranking.

## Outputs

Expected work directory:

`outputs/e8/drpo_c_confirmation_001`

Required durable text evidence includes:

- `SWEEP_PLAN.json`;
- `SMOKE_GATE.json`;
- `SWEEP_STATUS.json` and `SWEEP_COMPLETE.json`;
- `aggregate/per_cell_summary.csv`;
- `aggregate/terminal_audit.json`;
- every cell's `summary.json`, `manifest.json`, and `metrics.csv`;
- logs and source/config hashes;
- package and delivery manifests required by the existing E8 results-repository path.
