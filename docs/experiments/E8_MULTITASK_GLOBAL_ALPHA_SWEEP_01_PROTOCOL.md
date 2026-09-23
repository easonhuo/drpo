# E8 Multitask Global-Alpha Sweep Protocol

Experiment ID: `EXT-C-E8-MULTITASK-GLOBAL-ALPHA-SWEEP-01`

Status: protocol frozen for implementation; scientific run not started.

## Scientific question

On the exact eight structured-generation transfer tasks used by the current E8
multitask external-transfer pipeline, measure the finite-horizon response to a
uniform negative-sample weight

$
w(z)=\alpha
$

without learner-relative remoteness. This fills the missing Global-`alpha`
transfer response needed for the SG-9 external-transfer aggregate.

This experiment is external-validity evidence only. It does not replace C-U1 or
D-U1 controlled mechanism identification and does not establish convergence,
steady state, significance, or a universal method ranking.

## Frozen tasks and data

The active scientific cells cover exactly these eight P0 transfer tasks:

- `word_sorting`
- `spiral_matrix`
- `mini_sudoku`
- `maze`
- `word_ladder`
- `knights_knaves`
- `graph_color`
- `wikisql`

Countdown is retained only as the historical SG-9 anchor and receives no new
scientific cell in this sweep. The existing model-independent banks, train /
validation split identities, all-unique-negative consumer, and task interfaces
remain unchanged. The test partition remains forbidden during tuning.

## Global-alpha grid

The eight nonzero Global-`alpha` values are frozen to

$
\alpha \in
\left\{
\frac{1}{128},\frac{1}{64},\frac{1}{32},\frac{1}{16},
\frac{1}{8},\frac{1}{4},\frac{1}{2},1
\right\}.
$

Decimal values:

`[0.0078125, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0]`.

`alpha=0` is exactly the Positive-only endpoint and is not retrained. Existing
protocol-matched Positive-only results remain the left endpoint for response
plots and SG-9 aggregation.

Global-`alpha` uses the canonical E8 paper runtime with exponential coefficient
`c=0`, so the old kernel reduces exactly to `alpha * exp(0) = alpha`.
No loss, optimizer, bank, evaluator, or trainer formula is reimplemented.

## Seeds and matrix size

Paired development seeds are frozen to:

`[4000, 5000]`.

The scientific matrix is therefore:

$
8\text{ tasks} \times 8\text{ alpha values} \times 2\text{ seeds}
= 128\text{ cells}.
$

No Countdown cell and no Positive-only rerun is included in the 128 cells.

## Frozen training and evaluation contract

The sweep inherits the current protocol-matched E8 transfer settings:

- Qwen2.5-0.5B-Instruct, revision
  `7ae557604adf67be50417f59c2c2f167def9a775`;
- fresh LoRA from the base model for every cell;
- LoRA rank 32, alpha 64, dropout 0.05;
- 1200 optimizer updates, no early stopping;
- micro-batch 1, gradient accumulation 8;
- learning rate `5e-5`, weight decay `0.01`, warmup ratio `0.03`;
- max gradient norm `1.0`;
- evaluation every 100 updates;
- late-window primary summary over updates
  `[800, 900, 1000, 1100, 1200]`;
- transfer-task Pass@8 on the frozen 128-prompt validation subset;
- the same task-local max-length, max-new-token, and evaluation-batch settings
  as the reciprocal/DRPO transfer sweeps;
- two cells per GPU on GPU 0--7, maximum concurrency 16.

## Response and reporting

For every task and alpha, report the paired-seed mean late-window Pass@8 and the
terminal Pass@8, together with Greedy/valid-rate diagnostics available from the
canonical evaluator. The response curve must retain all eight alpha points.

Task-performance degradation, valid-structure/support diagnostics, and NaN/Inf
numerical failure remain separate outcome classes. Fixed 1200-update evidence is
finite-horizon evidence and must not be called convergence or steady state.

The sweep exists to complete the Global-`alpha` transfer response. Any later
choice of a paper-facing per-task Global-`alpha` operating point must use the
same validation-only selection semantics as the other E8 transfer controls and
must not use the test partition.
