# E8 Reciprocal 400-cell merged response evidence

This directory stores the compact, repository-local evidence for the merged E8 Reciprocal response curves assembled from four runs.

## Source experiments

1. `EXT-C-E8-MULTITASK-RECIPROCAL-MATRIX-01` — 128 cells
2. `EXT-C-E8-MULTITASK-RECIPROCAL-RIGHTTAIL-01` — 104 cells
3. `EXT-C-E8-MULTITASK-RECIPROCAL-RIGHTTAIL-CLOSURE-01` — 108 cells
4. `EXT-C-E8-MULTITASK-RECIPROCAL-LINEAR-GRAPHCOLOR-CLOSURE-01` — 60 cells

Total: **400 cells**.

The merged cell table has exactly 400 unique `(task, method, lambda, seed)` rows with seeds `4000/5000`.

## Canonical compact evidence

- `RECIPROCAL_400_CURVE_POINTS.csv` — 400-row cell-level curve data.
- `RESULT_LOCATOR.json` — machine-readable locator for data, renderer, and figure assets.
- `RESULT_SUMMARY.md` — concise interpretation and evidence boundary.

## Figure 6-aligned paper layout

The reciprocal figure now follows the same repository convention as the existing Structured-9 Figure 6 pipeline:

- plot-ready mirror: `paper/iclr2027/figures/data/e8_reciprocal_response_20260921/RECIPROCAL_400_CURVE_POINTS.csv`
- renderer: `scripts/figures/plot_e8_multitask_exp_response_curves.py --reciprocal`
- checked-in working SVG: `paper/iclr2027/figures/fig_e8_reciprocal_400_response_eight_panel.svg`
- renderer defaults also materialize:
  - `paper/iclr2027/figures/fig_e8_reciprocal_400_response_eight_panel.png`
  - `paper/overleaf/figures/fig_app_structured8_reciprocal_400_response.pdf`

The PNG/PDF are generated outputs; the current PR checks in the SVG working preview and the exact plot-ready observations.

## Figure coordinate

The plot overlays both Reciprocal families on one matched taper-strength x-axis:

- Reciprocal Linear: `x = lambda`
- Reciprocal Quadratic: `x = sqrt(lambda)`

The y-axis is two-seed mean late-window Pass@8. Countdown was not part of these reciprocal sweeps, so the 3×3 layout contains eight task panels plus one explanatory panel.

## Evidence boundary

This is **finite-horizon / pilot response-curve evidence**. It supports descriptive curve-shape and localization statements only. It does not establish convergence, steady state, statistical significance, or universal/cross-task method ranking.

The original raw ZIP packages remain separate provenance artifacts intended for `easonhuo/drpo-results`; this compact copy does not replace them.
