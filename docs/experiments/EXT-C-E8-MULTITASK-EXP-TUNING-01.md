# EXT-C-E8-MULTITASK-EXP-TUNING-01

## Status and role

- Status: implementation candidate / not run
- Role: external-validity development tuning across nine discrete tasks
- Causal authority: unchanged; D-U1 remains the categorical controlled-identification environment
- Parent implementation: `EXT-C-E8-MULTITASK-P0-01`
- Formal registration: deferred until the implementation SHA is frozen and reviewed

This stage trains task-specific negative-update methods. It is separate from the P0 occurrence and implemented-gradient diagnostic and must not retroactively change the P0 claim.

## Frozen tuning question

For each of nine discrete tasks, what exponential-remoteness retention produces the strongest validation trajectory under one fixed optimization contract, relative to a single paired Positive-only baseline?

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

The eight new tasks reuse the exact qualified P0 banks, pinned task sources, generators, and verifiers. They do **not** reuse the P0 100-update adapters, because those diagnostic adapters were trained on all 6000 bank prompts and would expose later validation/test prompts. After the deterministic 5000/500/500 split is frozen, this experiment creates one new train-only 100-update positive reference per P0 task. The reference manifest must record `validation_rows_seen=0` and `test_rows_seen=0`.

Countdown remains a separate archived external task and is supplied through explicit train-bank, structurally disjoint validation, and reference-adapter paths. Its test file is not an input. The runner fails closed if any bank, adapter, source, split, base model, or configuration identity changes.

## Frozen 72-cell tuning matrix

Each task has seven exponential cells and one Positive-only cell:

```text
rho = [0.9, 0.75, 0.6, 0.5, 0.35, 0.25, 0.125]
methods = [positive_only, exponential]
9 tasks x (1 + 7) = 72 cells
16 concurrent slots -> exactly 5 scheduling waves
```

The four coarse anchors are `0.9, 0.6, 0.35, 0.125`; the three predetermined refinement points are `0.75, 0.5, 0.25`. Refinement does not depend on which coarse point looks best. Together they form one seven-point response curve rather than a winner-only rerun.

The exponential controller uses detached learner-relative mean-token surprisal:

```text
u = sqrt(relu(surprisal - tau) / scale)
w(u) = exp(-lambda * u)
lambda = -log(rho)
```

`tau` and `scale` are calibrated once per task from the frozen train-only reference policy and frozen training-only calibration prompts. `tau` is the current-near median and `scale` is current-far median minus current-near median, so `u=0` at the near anchor and `u=1` at the far anchor. A degenerate or non-finite scale fails closed.

All seven exponential points share the same task-specific initial negative-gradient target. For each `rho`, a frozen multiplier matches the combined `0.5/0.5` near/far raw negative-gradient norm to `1/32` of that task's initial positive-gradient norm. Thus the grid primarily changes remoteness shape instead of initial total pressure. Positive-only has no negative branch and is not part of this budget matching.

## Data split and leakage boundary

For each of the eight P0 tasks, the qualified 6000-row bank is partitioned by a deterministic prompt-ID hash:

```text
train = 5000 prompts
validation = 500 prompts
test = 500 prompts
```

Countdown preserves its historical structural split: the tuning runner hash-selects 5000 rows from the frozen train bank and reads 500 rows from the separately supplied frozen validation file.

The split manifest is frozen before reference preparation, calibration, or method training. P0 train-only references, calibration, and all optimizer updates may open only train rows. Tuning and candidate selection may open only validation rows. P0 test partitions are written for identity separation but are never opened by reference preparation, training, evaluation, aggregation, or selection. Bank rows are never relabeled with permanent near/far identities; current near/far are reselected from each prompt's frozen 16-negative bank under the current learner.

## Reference preparation

Before calibration, each P0 task receives one new reference adapter:

- fresh Qwen2.5-0.5B-Instruct LoRA initialized under the inherited P0 LoRA contract;
- exactly 100 optimizer updates;
- only the task's 5000 train oracle completions;
- fixed train-only NLL audit;
- no validation/test metric, early stopping, or method response used for checkpoint selection;
- terminal adapter selected only by the fixed update count.

Countdown uses its supplied archived reference adapter and structurally disjoint validation protocol. Reference preparation is an engineering prerequisite, not method evidence.

## Method-training contract

Each cell starts from the same task-specific reference adapter and loads an independent writable copy of its LoRA parameters. The candidate method-training contract inherits the established Countdown offline-RL budget where applicable:

- Qwen2.5-0.5B-Instruct backbone;
- LoRA rank 32, alpha 64, dropout 0.05 for the eight new task references;
- fixed 1200 optimizer updates;
- AdamW, learning rate `5e-5`, weight decay `0.01`;
- cosine schedule with 3% warmup;
- microbatch 1 and gradient accumulation 8;
- maximum gradient norm 1.0;
- no early stopping and no checkpoint-based stopping;
- evaluation every 100 updates;
- terminal adapter always saved;
- validation-selected best checkpoint retained as supplementary tuning evidence only.

Positive-only and all exponential cells use the same task, tuning seed, prompt order, optimizer settings, update count, and evaluation schedule. Their only objective difference is whether dynamically selected negative branches enter the loss.

## Objective

Each optimizer microbatch contains oracle-positive completions and dynamically selected current-near/current-far verifier-wrong completions from the same frozen prompts. Selection is stop-gradient.

```text
positive_only:
    loss = -mean(log pi(positive))

exponential:
    loss = -mean(log pi(positive))
           + negative_scale * 0.5 * mean(w(u_near) * log pi(negative_near))
           + negative_scale * 0.5 * mean(w(u_far)  * log pi(negative_far))
```

The positive term is minimized to increase oracle likelihood. The signed negative terms are minimized to decrease wrong-completion likelihood. The implementation preserves the unique-negative denominator and does not normalize by the sum of taper weights.

A real two-step liveness gate must show finite nonzero positive loss, finite nonzero repulsive scalar, finite nonzero raw gradient, changed adapter weights, and successful fresh-process reload before the full matrix is launched. These are engineering gates, not scientific evidence.

## Evaluation and selection

Every 100 updates, validation evaluation reports:

- greedy verifier success over 500 prompts;
- Pass@8 over a deterministic 128-prompt subset;
- greedy and sampled valid-format rates;
- mean completion length;
- late-window mean over updates `800, 900, 1000, 1100, 1200`;
- terminal metrics at update 1200;
- best checkpoint and best step as supplementary diagnostics.

For each task, the selected `rho` is the maximum validation late-window Pass@8 subject to finite execution and terminal greedy valid-format rate at least 0.95. Ties use, in order, terminal Pass@8, late-window greedy success, terminal greedy success, then the larger `rho` as the conservative weaker-taper tie-break. Positive-only anchors the response curve but cannot be selected as an Exp hyperparameter.

If all Exp points are below Positive-only, the report must say so. If the selected Exp point lies at `rho=0.125`, the strong-taper boundary is not closed and only that task may later receive an explicitly registered boundary extension. No task may be silently dropped or assigned a post-hoc private grid.

## Execution order

```text
prepare
-> reference          # eight train-only P0-task references
-> calibrate          # nine task-specific remoteness/budget calibrations
-> liveness           # one real two-step Exp cell and reload gate
-> run-wave 1..5
-> aggregate
-> audit
```

No full wave may start before all required references and calibrations are complete and liveness passes.

## Scheduling

The deterministic plan contains five waves with at most 16 cells each. The first three waves contain all four coarse anchors plus one Positive-only per task; the last two waves contain the three refinement points. The aggregate plan contains all 72 cells and every result is keyed by task, method, `rho`, and seed.

The default hardware topology is eight visible GPUs with two independent 0.5B LoRA cells per GPU. This is scheduling only and does not alter a scientific variable. Identity-checked completed cells resume; an incomplete or mismatched cell fails closed rather than being silently reused.

## Reporting boundary

The pilot must separately report:

1. task-performance degradation or improvement;
2. valid-format or other registered structure diagnostics;
3. NaN/Inf numerical failure.

No task-collapse threshold is registered in this tuning pilot, so collapse is reported as `not_adjudicated` rather than inferred post hoc. Valid-format rate remains a structure diagnostic, not a formal support-boundary event. A fixed 1200-update horizon is not convergence. P0 remains the occurrence diagnostic; this experiment is task-performance tuning only; D-U1 remains categorical causal authority.
