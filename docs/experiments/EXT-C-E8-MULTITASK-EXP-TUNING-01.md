# EXT-C-E8-MULTITASK-EXP-TUNING-01

## Status and role

- Status: implementation candidate / not run
- Role: external-validity development tuning across nine discrete tasks
- Causal authority: unchanged; D-U1 remains the categorical controlled-identification environment
- Parent implementation: `EXT-C-E8-MULTITASK-P0-01`
- Formal registration: deferred until the implementation SHA is frozen and reviewed

This stage trains task-specific negative-update methods. It is separate from the P0 occurrence and implemented-gradient diagnostic and must not retroactively change the P0 claim.

## Frozen tuning question

For each of nine discrete tasks, what exponential-remoteness retention produces the strongest validation trajectory under one fixed training contract, relative to a single paired Positive-only baseline?

The pilot may reveal task-specific response trends and candidate settings. It does not establish convergence, statistical significance, cross-task superiority, or a formal method ranking.

## Tasks and inherited inputs

The suite is:

1. Countdown;
2. word sorting;
3. spiral matrix;
4. Mini Sudoku;
5. maze;
6. word ladder;
7. knights and knaves;
8. graph coloring;
9. WikiSQL.

The eight new tasks reuse the exact qualified P0 banks and exact 100-update task-positive LoRA warm starts. Countdown remains a separate archived external task and is supplied through explicit bank and reference-adapter paths. The runner fails closed if any bank, adapter, source, split, or configuration identity changes.

## Frozen 72-cell tuning matrix

Each task has seven exponential cells and one Positive-only cell:

```text
rho = [0.9, 0.75, 0.6, 0.5, 0.35, 0.25, 0.125]
methods = [positive_only, exponential]
9 tasks x (1 + 7) = 72 cells
16 concurrent slots -> exactly 5 scheduling waves
```

The exponential controller uses detached learner-relative mean-token surprisal:

```text
u = sqrt(relu(surprisal - tau) / scale)
w(u) = exp(-lambda * u)
lambda = -log(rho)
```

`tau` and `scale` are calibrated once per task from the frozen reference policy and frozen calibration prompts. `tau` is the current-near median and `scale` is current-far median minus current-near median, so `u=0` at the near anchor and `u=1` at the far anchor. A degenerate or non-finite scale fails closed.

All seven exponential points share the same task-specific initial negative-gradient RMS target. A per-`rho` frozen multiplier is calibrated at initialization so the grid changes remoteness shape rather than initial total negative-gradient magnitude. Positive-only has no negative branch and is not part of this budget matching.

## Data split and leakage boundary

For every 6000-row task bank, a deterministic prompt-ID hash partition creates:

```text
train = 5000 prompts
validation = 500 prompts
test = 500 prompts
```

The split manifest is frozen before training. Tuning and candidate selection may use only validation. The test partition must not be opened by this pilot. Bank rows are never relabeled with permanent near/far identities; current near/far are reselected from each prompt's frozen 16-negative bank under the current learner.

## Training contract

Each cell starts from the same task-specific reference adapter and trains a fresh child LoRA copy. The candidate contract is:

- Qwen2.5-0.5B-Instruct backbone;
- inherited LoRA rank 32, alpha 64, dropout 0.05;
- fixed 1200 optimizer updates;
- AdamW, learning rate `2e-4`, weight decay `0.01`;
- cosine schedule with 5% warmup;
- microbatch 2 and gradient accumulation 32;
- maximum gradient norm 1.0;
- no early stopping and no checkpoint-based stopping;
- evaluation every 100 updates;
- terminal adapter always saved;
- validation-selected best checkpoint retained as supplementary tuning evidence only.

Positive-only and all exponential cells use the same task, seed, prompt order, optimizer settings, update count, and evaluation schedule. Their only objective difference is whether the selected current near/far negative branches enter the loss.

## Objective

Each optimizer batch contains oracle-positive completions and dynamically selected current-near/current-far verifier-wrong completions from the same frozen prompts. Selection is stop-gradient.

```text
positive_only:
    loss = -mean(log pi(positive))

exponential:
    loss = -mean(log pi(positive))
           + negative_scale * mean(w(u_near) * log pi(negative_near))
           + negative_scale * mean(w(u_far)  * log pi(negative_far))
```

The implementation preserves the existing unique-negative denominator and does not normalize by the sum of taper weights. The negative sign convention is audited by a two-step liveness test that must show finite, nonzero repulsive updates.

## Evaluation and selection

Validation evaluation reports:

- greedy verifier success;
- Pass@8;
- valid-format rate;
- mean completion length;
- late-window mean over updates `800, 900, 1000, 1100, 1200`;
- terminal metrics at update 1200;
- best checkpoint and best step as supplementary diagnostics.

For each task, the selected `rho` is the maximum validation late-window Pass@8 subject to finite execution and terminal valid-format rate at least 0.95. Ties use, in order, terminal Pass@8, late-window greedy success, terminal greedy success, then the larger `rho` as the conservative weaker-taper tie-break. Positive-only anchors the response curve but cannot be selected as an Exp hyperparameter.

If all Exp points are below Positive-only, the report must say so. If the selected Exp point lies at `rho=0.125`, the strong-taper boundary is not closed and only that task may later receive an explicitly registered boundary extension. No task may be silently dropped or assigned a post-hoc private grid.

## Scheduling

The deterministic plan contains five waves with at most 16 cells each. The first three waves contain the four coarse anchors plus one Positive-only per task; the last two waves contain the three refinement points. The aggregate plan still contains all 72 cells and every result is keyed by task, method, `rho`, and seed.

## Reporting boundary

The pilot must separately report:

1. task-performance degradation or improvement;
2. valid-format or other registered structure diagnostics;
3. NaN/Inf numerical failure.

A fixed 1200-update horizon is not convergence. P0 remains the occurrence diagnostic; this experiment is task-performance tuning only; D-U1 remains categorical causal authority.
