# E8 Multitask Reciprocal Matrix 01

Experiment ID: `EXT-C-E8-MULTITASK-RECIPROCAL-MATRIX-01`

RunSpec ID: `E8_MULTITASK_RECIPROCAL_MATRIX_20260914_01`

Status: `not_run`

Execution class: `formal`

Capability base: `1830e672395ed4b881b147ad3c3a42f1b36074bf`

Execution branch: `dev/e8-multitask-exp-coldstart-01`

## Responsibility

Finite-horizon external-validity comparison of the two already-implemented reciprocal taper families on the exact eight P0 transfer tasks. Countdown launches no new scientific cells; historical Countdown reciprocal evidence is reused.

## Frozen design

Tasks: `word_sorting`, `spiral_matrix`, `mini_sudoku`, `maze`, `word_ladder`, `knights_knaves`, `graph_color`, `wikisql`.

Seeds: `[4000, 5000]`.

Training remains 1,200 optimizer updates per cell with no early stopping and no test-partition access.

Detached remoteness coordinate:

`x = relu(current_mean_completion_token_surprisal / 2 - 0.125)`.

Reciprocal-Linear:

`w = 1 / (1 + lambda * sqrt(x))`, with `lambda = [1, 3, 7, 19]`.

Reciprocal-Quadratic:

`w = 1 / (1 + lambda * x)`, with `lambda = [1, 9, 49, 361]`.

The paired grids use equal half-weight radius, so each quadratic coefficient is the square of the paired linear coefficient.

Matrix size: `8 tasks x 2 seeds x 2 methods x 4 strengths = 128 cells`.

Execution remains 16 concurrent cells (`8 GPUs x 2 slots`), with the seed-4000 batch completed before seed-5000. This gives 8 nominal waves.

## Reporting

The fixed 1,200-step horizon is finite-step evidence, not convergence or steady state. No method winner may be reported before terminal audit. Task performance, valid/structure boundary diagnostics, and NaN/Inf numerical failures must remain separate.

## RunSpec

The RunSpec is a durable execution snapshot. It uses deferred registration with `closure_required: true`, declares `formal_evidence_allowed: true`, and requires automatic delivery to `easonhuo/drpo-results` on `ingest/e8`.

This protocol freezes the execution identity and scientific matrix only. It does not launch training.
