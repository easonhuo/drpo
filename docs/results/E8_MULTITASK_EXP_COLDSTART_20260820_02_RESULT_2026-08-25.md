# EXT-C-E8-MULTITASK-EXP-COLDSTART-01 — 208-cell pilot result closure

**Run ID:** `E8_MULTITASK_EXP_COLDSTART_20260820_02`
**Scientific status:** `pilot`
**Repository closure base:** `e35573be30a5df53dd6201f112fb57fedabd8d15`

## Closure summary

The frozen cold-start sweep completed all `208/208` scheduled cells. The terminal audit reports zero missing cells, zero incomplete cells, zero NaN/Inf numerical failures, Countdown protocol diagnostic `PASS`, and no access to the test partition. The run used a fixed 1200-update horizon; this is **not** evidence of convergence or steady state.

This closure preserves the run as a historical response-curve anchor. The eight transfer-task Exponential curves are single-seed response-shape localization, while the transfer Positive-only baseline uses four seeds. Fresh-seed confirmation is therefore required before any winner claim. Cross-method ranking, statistical-significance claims, convergence/steady-state claims, and OOD-generalization claims are not supported by this run.

## Task-local response localization

| Task | Positive-only late-window Pass@8 | Selected coefficient | Selected Exp late-window Pass@8 | Grid edge | All Exp below PO |
|---|---:|---:|---:|:---:|:---:|
| Countdown | 0.1398 | 2.995732274 | 0.1542 | yes | no |
| Word Sorting | 0.620703125 | 1.7 | 0.5640625 | no | yes |
| Spiral Matrix | 1.0 | 6.907755279 | 0.996875 | yes | yes |
| Mini Sudoku | 0.9609375 | 6.0 | 0.9625 | no | no |
| Maze | 0.76875 | 0.9 | 0.7671875 | no | yes |
| Word Ladder | 0.105078125 | 2.0794415416798357 | 0.1109375 | no | no |
| Knights & Knaves | 0.672265625 | 0.6931471805599453 | 0.68125 | no | no |
| Graph Coloring | 0.98671875 | 2.6 | 1.0 | no | no |
| WikiSQL | 0.856640625 | 3.8 | 0.8171875 | no | yes |

“Selected coefficient” is only the within-grid localization point under the frozen pilot selection rule; it is not a formal method winner.

## Durable curve anchor

The repository copy preserves the per-seed projection of all 208 cells needed to concatenate a later curve-completion sweep without rerunning these cells:

- `experiments/results/e8_multitask_exp_coldstart_20260820_02/CURVE_ANCHOR.csv` — 208 rows with task, method, historical coefficient, seed, late-window Pass@8, and late-window greedy.
- `experiments/results/e8_multitask_exp_coldstart_20260820_02/TASK_SUMMARY.csv` — exact task-level localization summary from the run artifact.
- `RUN_COMPLETE.json`, `TERMINAL_AUDIT.json`, and `COUNTDOWN_PROTOCOL_DIAGNOSTIC.json` — exact compact execution/audit evidence.

The original raw artifact remains hash-bound: `plot_curve_points.csv` SHA-256 `220087bab4439bdc6bc3a771a828c2e80962d3d2daf01c61c2dbf3ec78f42e5d` and `all_cells.csv` SHA-256 `779ccd31f04ec47ce47c4e0cee540ed4c7c920ee3e02992177a4c814a45c4c29`. Historical `rho` fields are intentionally omitted from the repository curve anchor and remain only in the original raw provenance; this closure does **not** freeze or prescribe the parameterization of any successor experiment.

## Provenance limitation

The run artifact records source/base commit `01471868f5f057cc033c4434265e4adb32f36ae8`. On 2026-08-25 that SHA could not be resolved from the authoritative `easonhuo/drpo` GitHub repository. The frozen run config hash is `777b3a276f4125fe3012b5eca1e7dceb7d4a9baa3a678069e8919cad671b9125`, the aggregate-summary SHA-256 is `64a55c088564259e3721811104d48e313dc5f095ab67f8808493da897ffa2bd1`, and the terminal-audit SHA-256 is `24bdb096071351353ebc873eea697e810001cadc8fedbb46cce35771c417c9da`.

Accordingly, this repository closure records a complete **pilot result and historical curve anchor with an explicit unresolved-source-commit limitation**. It does not upgrade the run to formal confirmatory evidence.
