# E8 Multitask EXP Lambda Completion Joint Result Closure — 2026-09-02

## Scope and status

- This closure deposits two previously unlanded multitask successor runs and concatenates them with the already-landed cold-start anchor for response-curve analysis only.
- `EXT-C-E8-MULTITASK-EXP-LAMBDA-COMPLETION-01`: `199/199` cells raw-complete, terminal audit `PASS`, no test access, and zero NaN/Inf numerical failures.
- `EXT-C-E8-MULTITASK-EXP-LAMBDA-CURVE-COMPLETION-02`: scientific workload `140/140` raw-complete, terminal audit `PASS`, no test access, and zero NaN/Inf numerical failures. The outer supervisor returned code `2` only in the post-scientific hardened-packaging phase because the generated main package was `42.271 MiB` against a `25 MiB` limit; `missing_required_outputs=[]` and the scientific workload had already completed.
- Both remain `pilot` response-shape evidence. Fixed `1,200` updates are not convergence or steady state; transfer-task Exp curves use a single tuning seed and therefore do not authorize significance or formal method ranking.
- The run source SHAs `831b3ab5b24bf42464e79abfaf7e1077b03addf0` and `71d792ab6dc9ba5c7086852f94db043563740d9e` could not be resolved from the authoritative GitHub repository at closure time. Their run manifests also record `origin_matches_local=false`; this provenance limitation is preserved and forbids upgrading either run to authoritative formal evidence.
- Countdown was not rerun in either successor. Countdown values digitized from a presentation image are deliberately excluded from repository scientific evidence.
- Task-performance collapse, validity/support-structure diagnostics, and NaN/Inf numerical failure remain separate reporting categories; this closure does not retroactively invent an unregistered task-collapse threshold.

## Joint response summary

| Task | Pooled PO | Best Exp | Best λ | Rightmost Exp | Rightmost λ | Unique Exp λ |
|---|---:|---:|---:|---:|---:|---:|
| word_sorting | 62.057% | 65.781% | 37.253 | 63.281% | 120 | 65 |
| spiral_matrix | 100.000% | 100.000% | 8.44711 | 100.000% | 34.5388 | 28 |
| mini_sudoku | 95.911% | 97.656% | 30.919 | 97.188% | 220 | 65 |
| maze | 76.745% | 77.812% | 8.57117 | 77.188% | 100 | 65 |
| word_ladder | 10.677% | 14.531% | 5.31119 | 12.500% | 120 | 65 |
| knights_knaves | 67.526% | 70.000% | 24.1567 | 68.125% | 110 | 65 |
| graph_color | 98.672% | 100.000% | 0.08 | 97.969% | 210 | 65 |
| wikisql | 85.651% | 88.125% | 111.556 | 86.094% | 240 | 65 |
| countdown | 13.980% | 15.420% | 2.99573 | 15.420% | 2.99573 | 6 |

The pooled Positive-only values above combine all available historical-anchor and completion-01 Positive-only rows within each task. They are plotting/reference summaries, not fresh-seed significance estimates.

## Boundary interpretation and next-design scope

- **Word Sorting:** the sampled right tail is not conclusively closed; the curve reaches `65.781%` at λ≈`37.25` and remains `63.281%` at λ=`120`, above the pooled PO reference.
- **Mini Sudoku:** remains on a high plateau; the right edge λ=`220` is `97.188%` with no clear sustained right-tail decline.
- **Maze:** shows a broad plateau without a sufficiently clear decline; the right edge λ=`100` remains near the pooled PO reference.
- **Knights & Knaves:** decline hints exist, but the right edge λ=`110` remains slightly above pooled PO; the boundary is not treated as conclusively closed.
- **WikiSQL:** the best sampled high-λ point reaches `88.125%` at λ≈`111.56`, above pooled PO; the right edge λ=`240` remains slightly above pooled PO, so the sampled right-tail boundary is not treated as conclusively closed.
- **Word Ladder** and **Graph Coloring** show clearer peak/right-tail decline and are not prioritized for the next right-tail completion. **Spiral Matrix** is at ceiling. **Countdown** stays on its separate paper-aligned external-validity evidence chain.
- The user-approved design direction for the next round is therefore five tasks — Word Sorting, Mini Sudoku, Maze, Knights & Knaves, and WikiSQL — with `20` points per task and wider spacing than the 140-cell round. Exact λ values and a successor experiment ID are intentionally **not frozen in this result closure**; a new documented protocol is required before launch.

## Durable files

- Historical anchor: `experiments/results/e8_multitask_exp_coldstart_20260820_02/CURVE_ANCHOR.csv`.
- Completion-01 compact evidence: `experiments/results/e8_multitask_exp_lambda_completion_20260827_01/`.
- Curve-Completion-02 compact evidence: `experiments/results/e8_multitask_exp_lambda_curve_completion_02/`.
- Joint task-partitioned curve projection: `experiments/results/e8_multitask_exp_lambda_joint_20260902/CURVE_MANIFEST.json` plus the nine CSV parts under `experiments/results/e8_multitask_exp_lambda_joint_20260902/CURVE_POINTS/`; concatenating them in manifest order reconstructs all `547` rows.
