# E8 Reciprocal 400-cell merged result summary

## Coverage

The merged response dataset contains exactly **400 cells** from four runs:

- `EXT-C-E8-MULTITASK-RECIPROCAL-MATRIX-01`: 128
- `EXT-C-E8-MULTITASK-RECIPROCAL-RIGHTTAIL-01`: 104
- `EXT-C-E8-MULTITASK-RECIPROCAL-RIGHTTAIL-CLOSURE-01`: 108
- `EXT-C-E8-MULTITASK-RECIPROCAL-LINEAR-GRAPHCOLOR-CLOSURE-01`: 60

The compact CSV has 400 unique `(task, method, lambda, seed)` rows, exactly seeds `4000/5000`, and no duplicated cells. Across the four source packages all scheduled rows were complete and NaN/Inf numerical failure count was zero.

## Response-curve reading

The added right-tail and Graph Color low-range points substantially close the previously open response-shape questions. Reciprocal Linear and Reciprocal Quadratic should remain reported as separate families. The expanded curves show that neither family has a stable cross-task advantage over the other.

For peak comparisons, isolated single-point maxima should not be treated as the primary summary. Use a local multi-point smoothing window when describing peak regions so Graph Color / Maze-like spikes do not dominate interpretation.

The merged evidence is descriptive finite-horizon response evidence only; it is not a convergence, steady-state, significance, or universal method-ranking result.

## Durable paths

- CSV: `experiments/results/e8_reciprocal_response_20260921/RECIPROCAL_400_CURVE_POINTS.csv`
- Figure: `experiments/results/e8_reciprocal_response_20260921/reciprocal_merged_400_9grid.svg`
- Plot code: `experiments/results/e8_reciprocal_response_20260921/PLOT_CODE.md`
- This summary: `experiments/results/e8_reciprocal_response_20260921/RESULT_SUMMARY.md`

The raw four ZIP result packages are separate provenance objects and remain intended for archival in `easonhuo/drpo-results`.
