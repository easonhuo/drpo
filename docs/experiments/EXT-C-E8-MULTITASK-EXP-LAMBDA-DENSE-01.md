# EXT-C-E8-MULTITASK-EXP-LAMBDA-DENSE-01

## Status and role

- Status: implementation candidate / not run
- Scientific status after execution: development pilot
- Role: task-local response-shape refinement after
  `EXT-C-E8-MULTITASK-EXP-TUNING-01`
- Causal authority: unchanged; D-U1 remains the categorical
  controlled-identification environment
- Parent result:
  `easonhuo/drpo-results@26cf09102e4eaea9b58844fac158dc3cfc33314d`

This stage maps the lambda response more densely before spending a larger budget on
fresh-seed confirmation. It does not replace replication, establish significance, or
authorize a universal method-ranking claim.

## Task decision

Countdown is not rerun because its existing scenario evidence is already sufficient
for this development question. Spiral Matrix is excluded because Positive-only and all
seven predecessor Exp points are exactly 100%, so another response sweep has no
identifiable task-performance signal.

The remaining seven tasks are:

1. word sorting;
2. Mini Sudoku;
3. maze;
4. word ladder;
5. knights and knaves;
6. graph coloring;
7. WikiSQL.

Graph coloring is retained only as a saturation control. Its Positive-only late-window
Pass@8 is 99.6875%, so the broad lambda scan must not be described as local peak
identification.

## Frozen 112-cell matrix

Each task owns one complete 16-cell wave. All cells use the predecessor tuning seed
`2026072904`; no Positive-only cell is rerun. Fifteen points per task add local or
boundary resolution, and one point exactly repeats the predecessor-selected lambda as
a report-only bridge.

| Wave | Task | Scientific role | Lambda range |
|---:|---|---|---|
| 1 | word sorting | local peak refinement | 1.05–1.95 |
| 2 | Mini Sudoku | strong-taper boundary closure | 1.80–8.00 |
| 3 | maze | local peak refinement | 0.10–0.70 |
| 4 | word ladder | noisy peak refinement | 0.95–2.05 |
| 5 | knights and knaves | local peak refinement | 0.65–1.65 |
| 6 | graph coloring | saturation-control broad scan | 0.10–6.00 |
| 7 | WikiSQL | strong-taper boundary closure | 1.40–5.20 |

The exact values are authoritative only in
`configs/e8_multitask_exp_lambda_dense.yaml`.

## Inherited and rerun components

The pipeline fails closed unless the predecessor work directory matches the immutable
delivered hashes for its plan, split manifest, train-only reference manifest, and
aggregate summary. It then:

- reuses the exact seven predecessor train/validation/test partitions without opening
  test rows;
- reuses the exact seven train-only 100-update reference adapters;
- inherits the predecessor Positive-only and seven-point Exp response as plotting and
  comparison anchors;
- reruns task-specific initial-gradient calibration for all 16 new lambda values;
- runs a fresh two-update adapter-change/reload liveness gate;
- trains seven task-local waves;
- aggregates both the 112 new cells and the combined 168-point response;
- writes a terminal audit before RunSpec packaging and delivery.

Reference reuse is a cost optimization, not a new scientific result. Calibration is
rerun because its frozen gradient-matching multiplier depends on the candidate lambda.

## Preserved training contract

The parent contract remains unchanged:

- Qwen2.5-0.5B-Instruct with LoRA rank 32, alpha 64, dropout 0.05;
- 1200 optimizer updates per cell;
- AdamW learning rate `5e-5`, weight decay `0.01`, cosine schedule, 3% warmup;
- microbatch 1 and gradient accumulation 8;
- evaluation every 100 updates;
- late window at updates 800, 900, 1000, 1100, and 1200;
- dynamically reselected current-near/current-far negatives;
- task-specific initial negative-gradient target of 1/32 the positive-gradient norm;
- no early stopping and no test access.

## Selection and reporting

The primary development metric remains validation late-window Pass@8. Terminal
Pass@8, late-window greedy, terminal greedy, and weaker suppression break exact ties.
The report must also record:

- whether the selected point lies on either grid edge;
- whether the strong-taper edge remains unclosed;
- the bridge delta against the predecessor same-lambda cell;
- the inherited Positive-only delta;
- task-performance, validity/structure, and NaN/Inf separately.

The bridge is an implementation-continuity diagnostic, not a second statistical seed.
The 112-cell run is still single-seed shape discovery. Any later claim that a selected
Exp setting exceeds Positive-only requires a separately frozen fresh-seed confirmation
stage and untouched test evaluation.

## Execution and delivery

The reviewed one-click entrypoint is:

```bash
E8_DENSE_EXPECTED_COMMIT=<implementation-sha> \
E8_DENSE_PARENT_OUTPUT_ROOT=<completed-parent-output> \
E8_DENSE_BASE_MODEL_PATH=<qwen-model-path> \
bash scripts/run_e8_multitask_exp_lambda_dense.sh full
```

The associated deferred E8 RunSpec is responsible for packaging the text-first
outputs and automatically delivering them to
`easonhuo/drpo-results@ingest/e8`. Checkpoints and adapter weights remain excluded.
