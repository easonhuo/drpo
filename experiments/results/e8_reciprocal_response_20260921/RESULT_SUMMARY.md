# E8 Reciprocal 400-cell merged result summary

## Coverage

The merged response dataset contains exactly **400 cells** from four runs:

- `EXT-C-E8-MULTITASK-RECIPROCAL-MATRIX-01`: 128
- `EXT-C-E8-MULTITASK-RECIPROCAL-RIGHTTAIL-01`: 104
- `EXT-C-E8-MULTITASK-RECIPROCAL-RIGHTTAIL-CLOSURE-01`: 108
- `EXT-C-E8-MULTITASK-RECIPROCAL-LINEAR-GRAPHCOLOR-CLOSURE-01`: 60

The compact CSV has 400 unique `(task, method, lambda, seed)` rows and exactly seeds `4000/5000`.

## Response-curve reading

The added right-tail and Graph Color low-range points substantially close the previously open descriptive response-shape questions. Reciprocal Linear and Reciprocal Quadratic remain separate families; neither family is promoted to a universal cross-task winner.

For peak comparisons, isolated single-point maxima should not be treated as the primary summary. Use local multi-point smoothing when a peak-region summary is needed so Graph Color / Maze-like spikes do not dominate interpretation.

The merged evidence remains descriptive finite-horizon response evidence only; it is not a convergence, steady-state, significance, or universal method-ranking result.

## Table 1 SG-9 aggregate

For the compact external-transfer table, Countdown is now aggregated with the eight transfer tasks into a nine-task structured-generation macro (SG-9). The corresponding Pass@8 values are:

| Scope | Positive-only | Global-α | Reciprocal Linear | Reciprocal Quadratic | DRPO / Exp |
| --- | ---: | ---: | ---: | ---: | ---: |
| SG-9 macro | 67.91 | pending | 68.08 | 68.17 | 69.25 |

The paper-facing compact table rounds these to one decimal place: `67.9 / -- / 68.1 / 68.2 / 69.2`. For Reciprocal Linear/Quadratic, the SG-9 macro uses the current ICLR Table 1 displayed Countdown values (`13.6/14.5`) together with the four-decimal transfer-task summaries; one decimal is therefore the intended paper-facing precision. The Global-α aggregate is intentionally blank because the protocol-matched eight-transfer-task Global-α sweep has not yet been run. The derived task-level inputs and macro are stored in `TABLE1_SG9_TAPER_AGGREGATE.csv`.

This row is a descriptive finite-horizon aggregate; it does not convert the reciprocal sweeps into a formal universal method-ranking result.

## Durable paths

- Canonical compact CSV: `experiments/results/e8_reciprocal_response_20260921/RECIPROCAL_400_CURVE_POINTS.csv`
- Machine locator: `experiments/results/e8_reciprocal_response_20260921/RESULT_LOCATOR.json`
- Figure 6-aligned plot-ready mirror: `paper/iclr2027/figures/data/e8_reciprocal_response_20260921/RECIPROCAL_400_CURVE_POINTS.csv`
- Plot code: `scripts/figures/plot_e8_multitask_exp_response_curves.py --reciprocal`
- Checked-in working figure: `paper/iclr2027/figures/fig_e8_reciprocal_400_response_eight_panel.svg`
- Generated PNG default: `paper/iclr2027/figures/fig_e8_reciprocal_400_response_eight_panel.png`
- Generated paper-facing PDF default: `paper/overleaf/figures/fig_app_structured8_reciprocal_400_response.pdf`

The raw four ZIP result packages are separate provenance objects and remain intended for archival in `drpo-results`.
