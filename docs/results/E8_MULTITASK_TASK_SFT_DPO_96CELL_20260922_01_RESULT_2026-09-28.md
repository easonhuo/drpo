# E8 Task-SFT DPO 96-cell result deposition

**Experiment:** `EXT-C-E8-MULTITASK-TASK-SFT-DPO-96CELL-01`  
**Run:** `E8_MULTITASK_TASK_SFT_DPO_96CELL_20260922_01`  
**Repository closure base:** `888423f94d78b18a7c9e84097de9685861787375`  
**Repository scientific status:** **pilot / raw-complete response evidence with unresolved provenance**

## What is deposited

The uploaded raw-complete artifact contains all `96/96` configured cells for the eight transfer tasks. Training/evaluation completion and the run's terminal contract are complete, the test partition was not accessed, and the run reports zero NaN/Inf cells. The run's own workload manifests label the completed run `finite_step_validated`; however, the recorded source commit `905375a374647ea3d4691eae1d66022d2be103f4` cannot be resolved in the authoritative GitHub repository as of 2026-09-28. This repository deposition therefore preserves the evidence at **pilot** status and does not upgrade it to authoritative formal evidence.

The current run contains no Countdown cell. Historical Countdown DPO evidence is separate provenance and is not merged into this 96-cell run.

## Protocol identity preserved in the result

The DPO sweep is `beta in {0, 0.05, 0.1, 0.2, 0.5, 1.0}`. `beta=0` is the deterministic task-specific SFT-only control with no DPO optimizer update. Its two configured DPO seed labels duplicate the same SFT control, so it contributes one independent static-control replicate per task. Positive-beta points use two independent DPO-stage training replicates per task and beta while sharing the materialized task-SFT initialization.

The primary paper-facing response coordinate in this artifact is validation late-window Pass@8 averaged over updates `800, 900, 1000, 1100, 1200` for positive-beta cells; `beta=0` is evaluated at the SFT-only initialization.

## Descriptive response summary

| Task | SFT-only beta=0 Pass@8 (%) | Best positive beta | Best positive-beta Pass@8 (%) | Sweep-best beta incl. 0 | Sweep-best Pass@8 (%) |
|---|---:|---:|---:|---:|---:|
| `word_sorting` | 34.38 | 1 | 29.38 | 0 | 34.38 |
| `spiral_matrix` | 100.00 | 0.5 | 14.30 | 0 | 100.00 |
| `mini_sudoku` | 91.41 | 0.05 | 0.00 | 0 | 91.41 |
| `maze` | 79.69 | 1 | 75.78 | 0 | 79.69 |
| `word_ladder` | 4.69 | 0.5 | 0.16 | 0 | 4.69 |
| `knights_knaves` | 49.22 | 1 | 58.20 | 1 | 58.20 |
| `graph_color` | 96.88 | 1 | 78.12 | 0 | 96.88 |
| `wikisql` | 78.91 | 1 | 55.16 | 0 | 78.91 |

Across the eight transfer tasks, the fixed-beta macro late-window Pass@8 values are `7.40`, `9.14`, `11.67`, `19.88`, and `38.13` percent for positive `beta=0.05, 0.1, 0.2, 0.5, 1.0`, respectively; the SFT-only `beta=0` macro is `66.89` percent. The taskwise sweep-best macro including `beta=0` is `68.02` percent, while the taskwise best over positive beta only is `38.89` percent.

Five of eight tasks attain their best **positive-beta** value at the right boundary `beta=1.0`. Therefore this artifact does not close the positive-DPO right tail and does not justify claiming a universal best beta. Seven of eight taskwise sweep-best entries include the SFT-only `beta=0` control rather than a positive-beta DPO continuation; only `knights_knaves` has its sweep-best at a positive beta.

## Event and claim boundaries

Task-performance collapse is not adjudicated because this experiment has no registered collapse threshold. Greedy/sample validity is retained as a separate structure diagnostic only. NaN/Inf numerical failure count is `0`. Fixed `1200`-update continuation is not convergence or steady state.

No method ranking, significance claim, convergence claim, or test-set claim is authorized by this deposition. The unresolved source-commit provenance must remain explicit until the recorded source tree can be independently resolved or reproduced.

## Compact evidence

- `RESULT_CLOSURE.json`: repository interpretation and provenance boundary.
- `RUN_COMPLETE.json`: exact workload run-complete record from the uploaded artifact.
- `TERMINAL_AUDIT.json`: exact workload terminal-audit record.
- `CURVE_ANCHOR.csv`: compact per-cell paper-facing response anchor, preserving both configured beta-zero rows and all positive-beta seed rows.
- `GROUPED_CURVE.csv`: independent-replicate grouped response curve used for the descriptive summaries.
- `TASK_RESPONSE_SUMMARY.csv`: compact taskwise beta summary derived from the grouped curve.
- `BETA_MACRO_RESPONSE.csv`: compact eight-task macro response by beta.

The uploaded raw-complete package SHA-256 is `2cd42c723df4172c5fcff555223f51a32f405c8751c6e20d8626233dd832ccd4`.
