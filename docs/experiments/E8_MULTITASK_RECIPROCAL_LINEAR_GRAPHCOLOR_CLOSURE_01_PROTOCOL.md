# E8 Multitask Reciprocal Linear + Graph Color Closure 01 Protocol

## Experiment identity

- Experiment ID: `EXT-C-E8-MULTITASK-RECIPROCAL-LINEAR-GRAPHCOLOR-CLOSURE-01`
- Scientific role: finite-horizon external-validity response-curve follow-up.
- Parent evidence: the predecessor reciprocal matrix, 104-cell right-tail extension, and 108-cell right-tail closure rounds.
- This protocol does not authorize convergence, steady-state, significance, asymptotic, universal-superiority, or method-ranking claims.

## Locked scientific questions

This successor answers exactly two follow-up questions from the reviewed reciprocal response curves:

1. **Reciprocal-Linear right-tail closure.** Continue only the still-open Linear right tails for Word Sorting, Spiral Matrix, Mini Sudoku, and Word Ladder.
2. **Graph Color early-range densification.** Densify both Reciprocal-Linear and Reciprocal-Quadratic in the low-strength region where the existing sparse points show an apparent non-monotonic spike structure.

The Graph Color densification tests whether the apparent early-range structure persists under denser sampling. It does not pre-assume that the structure is caused by faster convergence; convergence-speed interpretation remains a post-run trajectory-analysis question.

No other task or method arm receives new scientific cells in this experiment.

## Frozen task-by-method grid

### Reciprocal-Linear right-tail extension

| Task | New Linear lambda values |
| --- | --- |
| Word Sorting | `601, 901, 1351, 2025, 3037` |
| Spiral Matrix | `181, 271, 401, 601, 901` |
| Mini Sudoku | `601, 901, 1351, 2025, 3037` |
| Word Ladder | `601, 901, 1351, 2025, 3037` |

Spiral Matrix starts at `181` because its existing Linear curve ends at `121`; the other three tasks already have completed Linear points through `401`.

### Graph Color early-range densification

The densified Graph Color points preserve the existing matched-strength relation `lambda_Q = lambda_L^2`:

| New point | Linear lambda | Quadratic lambda |
| ---: | ---: | ---: |
| A | 2 | 4 |
| B | 4 | 16 |
| C | 5 | 25 |
| D | 10 | 100 |
| E | 14 | 196 |

These points densify the previously sparse `1 -> 3 -> 7 -> 19` Linear interval and its squared Quadratic counterpart. Existing completed points are not rerun.

## Frozen cell count

Per transfer seed:

- Linear right-tail cells: `4 tasks x 5 = 20`
- Graph Color Linear densification: `5`
- Graph Color Quadratic densification: `5`
- Total per seed: `30`

With paired seeds `[4000, 5000]`, the formal matrix contains exactly **60 scientific cells**.

## Frozen training and evaluation contract

All scientific variables other than the new task/method coefficient lists are inherited unchanged from the current canonical reciprocal cold-start family, including:

- `Qwen/Qwen2.5-0.5B-Instruct` at the frozen model revision;
- fresh deterministic zero-update LoRA initialization;
- no warm start and no external adapter;
- 1200 optimizer updates per cell;
- the same optimizer, learning rate, LoRA, batching, evaluation cadence, late window, and generation settings;
- the same model-independent task banks and split hashes;
- 16 unique negatives per prompt and the existing canonical negative consumer;
- current-policy surprisal recomputed each update;
- detached reciprocal excess-remoteness coordinate `relu(current_sequence_surprisal/2 - 0.125)`;
- no extra square and no gradient-RMS matching;
- test partition unopened;
- paired seeds `4000` and `5000` with the existing hard seed-batch barrier;
- 16 concurrent cells over 8 GPUs with two slots per GPU;
- fixed 1200 updates are finite-horizon evidence, not convergence or steady-state evidence.

## Execution geometry

- Total cells: `60`
- Cells per seed: `30`
- Maximum concurrent cells: `16`
- Waves per seed: `ceil(30 / 16) = 2`
- Total waves with hard seed-batch barrier: `4`

## Reporting and terminal audit

Task performance, valid/structure diagnostics, and NaN/Inf numerical failure remain separate report dimensions. Terminal audit remains required before any collapse or method-ranking claim.

For the four Linear-tail tasks, the post-run question is whether the response curve reaches a plateau, continues rising at the tested boundary, or turns downward at the fixed 1200-update horizon.

For Graph Color, the post-run question is whether the apparent early-range spike/non-monotonic structure remains after densification. Any explanation in terms of faster convergence must be supported by the existing per-evaluation trajectories rather than inferred from endpoint Pass@8 alone.

A successful run may support finite-step response-curve evidence only.
