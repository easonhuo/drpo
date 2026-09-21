# E8 Reciprocal 400-cell merged response evidence

This directory stores the compact, repository-local evidence for the merged E8 Reciprocal response curves assembled on 2026-09-21.

## Source experiments

1. `EXT-C-E8-MULTITASK-RECIPROCAL-MATRIX-01` — 128 cells
2. `EXT-C-E8-MULTITASK-RECIPROCAL-RIGHTTAIL-01` — 104 cells
3. `EXT-C-E8-MULTITASK-RECIPROCAL-RIGHTTAIL-CLOSURE-01` — 108 cells
4. `EXT-C-E8-MULTITASK-RECIPROCAL-LINEAR-GRAPHCOLOR-CLOSURE-01` — 60 cells

Total: **400 cells**.

The merged cell table has exactly 400 unique `(task, method, lambda, seed)` rows, seeds `4000/5000`, all source rows marked complete, and no NaN/Inf numerical failures.

## Files

- `RECIPROCAL_400_CURVE_POINTS.csv` — compact 400-row cell-level curve data used by the figure.
- `reciprocal_merged_400_9grid.svg` — repository-viewable 3x3 vector figure.
- `PLOT_CODE.md` — exact Python plotting recipe used to reconstruct the SVG from the CSV.
- `RESULT_SUMMARY.md` — concise interpretation and evidence boundary.

## Figure coordinate

The plot overlays both Reciprocal families on a matched taper-strength x-axis:

- Reciprocal Linear: `x = lambda`
- Reciprocal Quadratic: `x = sqrt(lambda)`

The y-axis is two-seed mean late-window Pass@8.

## Evidence boundary

This is **finite-horizon / pilot response-curve evidence**. It supports descriptive curve-shape and localization statements only. It does not establish convergence, steady state, statistical significance, or universal/cross-task method ranking.

The original raw ZIP packages remain separate provenance artifacts intended for `easonhuo/drpo-results`; this compact copy does not replace them.
