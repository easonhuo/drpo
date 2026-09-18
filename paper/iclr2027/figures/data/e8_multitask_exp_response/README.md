# E8 multitask EXP response plot-ready data

This directory is the checked-in plot-ready input for
`scripts/figures/plot_e8_multitask_exp_response_curves.py`.

Scientific status remains `pilot_response_shape_only`. These files do not
upgrade convergence, significance, steady-state, or method-ranking claims.

## Provenance

The authoritative locator is:

`experiments/results/e8_multitask_exp_response_20260914/RESULT_LOCATOR.json`

For the eight non-Countdown transfer tasks, the plot-ready rows are derived
from the locator's standard plot composition:

- historical joint source at
  `easonhuo/drpo@2a19fa1cd23e25dcd3c319e58d286e47a4e10dac`,
  `experiments/results/e8_multitask_exp_lambda_joint_20260902/CURVE_POINTS/`;
- Wide-RightTail-03 source at
  `easonhuo/drpo-results@aea817b94acd9f87a9251cfd46c362bfed14c44b`,
  `e8/EXT-C-E8-MULTITASK-EXP-LAMBDA-WIDE-RIGHTTAIL-03_pilot/CURVE_POINTS_COMPACT.csv`
  where that task participated.

Rows are grouped only by identical `lambda`; no missing observation is
interpolated or fabricated in these CSV files. The plotting script performs
visual smoothing only after loading the stored observations.

Countdown uses the approved 22-point paper-aligned Exp./DRPO coefficient curve
recovered from the registered Countdown coefficient-response evidence chain.
Its Positive-only reference is `13.98%` late-window Pass@8. Countdown's plot
band is a visualization-only local-variation proxy because this compact
22-point mean series does not store per-point seed standard deviations.

## Columns

- `task`: task identifier;
- `lambda`: observed coefficient;
- `mean_pass8_pct`: observed late-window Pass@8 mean, in percent;
- `std_pass8_pct`: observed same-lambda seed standard deviation when available;
- `positive_only_pct`: pooled Positive-only reference, in percent;
- `point_count`: number of rows aggregated at that exact lambda when retained;
- `source`: compact provenance label.

Do not infer formal evidence status from the smoothed rendering. Use the
registered result documents and locator for scientific interpretation.
