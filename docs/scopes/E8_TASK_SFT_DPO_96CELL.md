# EXT-C-E8-MULTITASK-TASK-SFT-DPO-96CELL-01

## Status

- implementation state: in development
- result status: `not_run`
- execution class: formal candidate
- environment role: external-validity DPO initialization / beta-response study
- Countdown role: historical external-validity context only; no new Countdown cells

No scientific run has started. This document freezes the owner-approved scientific intent before implementation.

## Claim

On the exact eight P0 transfer tasks and the same frozen model-independent qualified banks used by the current E8 multitask baseline work, measure canonical DPO response after a task-specific train-only positive SFT initialization. The scan varies only DPO beta after the SFT checkpoint and uses two paired seeds. It is finite-horizon external-validity evidence and does not establish convergence, statistical significance, or a universal method ranking.

## Frozen task matrix

Transfer tasks:

1. `word_sorting`
2. `spiral_matrix`
3. `mini_sudoku`
4. `maze`
5. `word_ladder`
6. `knights_knaves`
7. `graph_color`
8. `wikisql`

DPO beta grid for every task:

```text
[0.0, 0.05, 0.1, 0.2, 0.5, 1.0]
```

Paired seed offsets:

```text
[4000, 5000]
```

Matrix size:

```text
8 tasks × 6 beta values × 2 seeds = 96 cells
```


## Task-specific SFT initialization

Each transfer task receives its own train-only positive SFT adapter before DPO. The implementation must reuse the existing P0 positive-warm-start trainer and its frozen SFT recipe rather than reimplement SFT:

- checkpoint kind: `task_positive_warmstart_100`
- optimizer updates: 100
- LoRA rank: 32
- LoRA alpha: 64
- LoRA dropout: 0.05
- micro batch: 2
- gradient accumulation: 32
- learning rate: `2.0e-4`
- weight decay: `0.01`
- warmup ratio: `0.05`
- max grad norm: `1.0`
- max length: 512
- train split only; validation/test rows seen: 0

The task-specific SFT adapter is the DPO policy initialization. Immediately before DPO update 1, an exact copy of that initialized policy adapter becomes the permanently frozen DPO reference adapter.

## Beta-zero control

`beta=0.0` is the SFT-only initialization control. It must perform **zero DPO optimizer updates**.

This special handling is required because the DPO optimizer uses AdamW with nonzero weight decay. Running nominal DPO optimizer steps at beta zero would still change parameters through weight decay and therefore would not be an SFT-only control.

The beta-zero cell keeps the task-specific SFT adapter unchanged, evaluates it on the same held-out validation protocol at the nominal evaluation checkpoints needed by aggregation, records zero DPO optimizer updates, and marks itself explicitly as the SFT-only / no-DPO-update control.

## Canonical DPO semantics for beta > 0

The positive-beta cells preserve the historical PR #268 canonical DPO semantics already ported into the multitask runner:

- chosen completion: oracle completion;
- rejected completions: every first-occurrence unique verifier-wrong completion from the same prompt;
- full-completion **summed** token log probability;
- exact frozen initial-policy reference;
- pair margin:
  `(log pi(y+|x)-log pi(y-|x))-(log ref(y+|x)-log ref(y-|x))`;
- sigmoid / softplus DPO loss;
- zero label smoothing;
- prompt-balanced mean over unique rejected completions;
- no hard-negative mining;
- no near/far selection;
- no distance taper;
- no value network.

Only the initialization source and beta grid differ from the current fresh-LoRA multitask DPO baseline.

## DPO training and evaluation

For beta > 0:

- optimizer updates: 1200;
- micro batch: 1;
- gradient accumulation: 8;
- learning rate: `5.0e-5`;
- weight decay: `0.01`;
- warmup ratio: `0.03`;
- max grad norm: `1.0`;
- early stopping: disabled;
- validation evaluation every 100 updates;
- paper-facing late window: `[800, 900, 1000, 1100, 1200]`;
- test split access: forbidden.

The same qualified 16-negative-per-prompt banks remain fixed. SFT initialization must not rebuild or retune the bank.

## Reporting

Report, separately:

- task-performance metrics;
- validity / structure diagnostics;
- NaN/Inf numerical failure.

For DPO cells also retain pair-margin, preference-accuracy, logit-saturation, raw-gradient norm, and optimizer-update norm diagnostics. Beta zero must be clearly identified as `sft_only_no_dpo_update`.

Any comparison across beta values must use terminal / late-window audit data and must not be described as convergence or universal method ranking.

## Implementation scope

Expected implementation touches:

- one new YAML config for this 96-cell experiment;
- existing DPO config validation;
- existing task-positive warm-start preparation / attachment plumbing;
- existing canonical DPO transfer trainer;
- existing cold-start shell orchestration;
- focused regression tests.

No new Python file is required. The canonical DPO loss, pair construction, bank construction, task verifiers, and evaluation semantics must not be rewritten.
