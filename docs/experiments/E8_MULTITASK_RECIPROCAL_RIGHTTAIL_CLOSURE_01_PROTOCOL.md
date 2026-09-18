# E8 Multitask Reciprocal Right-Tail Closure 01 Protocol

## Experiment identity

- Experiment ID: `EXT-C-E8-MULTITASK-RECIPROCAL-RIGHTTAIL-CLOSURE-01`
- Scientific role: finite-horizon external-validity response-curve closure for the canonical Reciprocal-Linear and Reciprocal-Quadratic tapers.
- Parent evidence: the predecessor reciprocal matrix and 104-cell right-tail extension.
- This protocol does not authorize convergence, steady-state, significance, universal-superiority, or method-ranking claims.

## Locked scientific question

Extend only the still-insufficiently-closed right tails of the reciprocal response curves. The goal is not to retune the algorithm or introduce a new method. It is to obtain longer right-tail coverage so that rise, plateau, peak, and decline behavior can be distinguished more solidly at the fixed 1200-update horizon.

Spiral Matrix launches no new cells because its existing right tail is already sufficiently closed for this purpose. Countdown also launches no new cells and remains reused historical evidence only.

## Frozen matched-strength construction

Matched reciprocal strength continues to use the existing relation

- Reciprocal-Linear coefficient: `lambda_L`
- Reciprocal-Quadratic coefficient: `lambda_Q = lambda_L^2`

The newly introduced common right-tail levels are:

| matched level | Linear lambda | Quadratic lambda |
| ---: | ---: | ---: |
| 9 | 181 | 32761 |
| 10 | 271 | 73441 |
| 11 | 401 | 160801 |

Existing levels 5--8 are reused only where a task/method arm previously stopped before them; no already-completed cell is rerun.

## Frozen task-by-method grid

| Task | Reciprocal-Linear new lambda values | Reciprocal-Quadratic new lambda values |
| --- | --- | --- |
| Word Sorting | `181, 271, 401` | `32761, 73441, 160801` |
| Spiral Matrix | none | none |
| Mini Sudoku | `181, 271, 401` | `32761, 73441, 160801` |
| Maze | `181, 271, 401` | `32761, 73441, 160801` |
| Word Ladder | `181, 271, 401` | `32761, 73441, 160801` |
| Knights & Knaves | `181, 271, 401` | `961, 2401, 6561, 14641, 32761, 73441, 160801` |
| Graph Color | `31, 49, 81, 121, 181, 271, 401` | `961, 2401, 6561, 14641, 32761, 73441, 160801` |
| WikiSQL | `181, 271, 401` | `32761, 73441, 160801` |
| Countdown | none | none |

The grid contains exactly 54 cells per seed. With paired seeds `[4000, 5000]`, the formal matrix contains exactly 108 cells.

## Frozen training and evaluation contract

All scientific variables other than the new task/method coefficient lists are inherited unchanged from `configs/e8_multitask_reciprocal_righttail_104.yaml`, including:

- `Qwen/Qwen2.5-0.5B-Instruct` at the frozen model revision;
- fresh deterministic zero-update LoRA initialization;
- no warm start and no external adapter;
- 1200 optimizer updates per cell;
- the same optimizer, LR, LoRA, batching, evaluation cadence, late window, and generation settings;
- the same model-independent task banks and split hashes;
- 16 unique negatives per prompt and the existing canonical negative consumer;
- current-policy surprisal recomputed each update;
- detached reciprocal excess-remoteness coordinate `relu(current_sequence_surprisal/2 - 0.125)`;
- no extra square and no gradient-RMS matching;
- test partition unopened;
- paired seeds `4000` and `5000` with the existing hard seed-batch barrier;
- 16 concurrent cells over 8 GPUs with two slots per GPU;
- fixed 1200 updates are not convergence or steady-state evidence.

## Execution geometry

- Total cells: `108`
- Cells per seed: `54`
- Maximum concurrent cells: `16`
- Waves per seed: `ceil(54 / 16) = 4`
- Total waves with hard seed-batch barrier: `8`

## Reporting and terminal audit

Task performance, valid/structure diagnostics, and NaN/Inf numerical failure remain separate report dimensions. Terminal audit remains required before any method-ranking or collapse claim. A successful fixed-horizon run may support finite-step response-curve evidence only; it does not establish convergence or steady state.
