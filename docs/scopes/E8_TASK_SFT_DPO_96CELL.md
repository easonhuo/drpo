# EXT-C-E8-MULTITASK-TASK-SFT-DPO-96CELL-01

## Status

- implementation state: in development
- result status: `not_run`
- execution class: `formal` (not launched)
- environment role: external-validity DPO initialization / beta-response study
- Countdown role: historical external-validity context only; no new Countdown cells

No scientific run has started. This document freezes the owner-approved scientific intent before implementation.

## Launch identity and expected outputs

- scientific config: `configs/e8_multitask_task_sft_dpo_96cell.yaml`
- existing runner: `scripts/run_e8_multitask_exp_coldstart.sh`
- formal invocation after review/merge: set `E8_COLDSTART_CONFIG=configs/e8_multitask_task_sft_dpo_96cell.yaml` and bind `E8_COLDSTART_EXPECTED_COMMIT` to the reviewed full launch SHA before calling the existing runner with `full`
- data/environment: the exact eight qualified P0 transfer-task banks and held-out validation splits materialized by the existing E8 multitask input pipeline; no new Countdown cell
- development/run seeds: paired DPO seed offsets `[4000, 5000]`
- separately held-out seed set: none registered for this finite-horizon beta-response experiment; the test partition remains forbidden
- expected per-cell output: `workload/cells/<cell_key>/cell_manifest.json` plus training/evaluation diagnostics
- expected aggregate outputs: `workload/aggregate/plot_curve_points.csv`, `workload/terminal_audit.json`, and `workload/RUN_COMPLETE.json`
- durable delivery: the existing guarded formal-run packaging path under the cold-start runtime

## Claim

On the exact eight P0 transfer tasks, using the same frozen qualified P0 source banks and the unchanged canonical reference-remoteness training-bank derivation used by the current E8 multitask baseline work, measure canonical DPO response after a task-specific train-only positive SFT initialization. Positive-beta cells use the two paired DPO seed offsets; beta zero keeps both execution labels only for the frozen 96-cell geometry and contributes one independent static-control replicate per task. It is finite-horizon external-validity evidence and does not establish convergence, statistical significance, or a universal method ranking.

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

The beta-zero cell keeps the task-specific SFT adapter unchanged, performs one held-out validation evaluation at step 0, records zero DPO optimizer updates, and marks itself explicitly as the SFT-only / no-DPO-update control. Because the terminal policy is bitwise the same policy state as the initial SFT adapter, the aggregate reuses that single measured validation value for the nominal late-window/terminal summary fields instead of rerunning the identical model at every nominal DPO checkpoint. The manifest records this reuse explicitly. Beta zero materializes the terminal adapter only; it must not also write a duplicate supplementary-best adapter containing the identical SFT policy.

The 96-cell geometry still contains beta-zero rows under both DPO seed labels. Those two rows for a task share the same materialized task SFT adapter and the same evaluation seed, so they are duplicate control rows for matrix geometry, not two independent SFT replications and not evidence for seed uncertainty at beta zero. Both task-local and final aggregate plot tables must therefore mark one beta-zero row as the independent representative and the other as a duplicate control label; grouped statistics must expose an effective independent-seed count of 1 and never use the duplicate row to create seed uncertainty or error bars.

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
- model max length: 512;
- max new tokens: 128;
- greedy validation prompts: 500;
- Pass@8 validation prompts: 128;
- evaluation batch size: 16;
- auxiliary Pass@64: disabled for the transfer tasks;
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

The qualified P0 source banks remain fixed and SFT initialization must not rebuild or retune them. As in the current E8 baseline, the canonical training input then deterministically derives a fixed 16-negative-per-prompt reference-remoteness bank from those source rows using all deterministic verifier-wrong candidates, the frozen zero-update reference policy, source P0 error-class sequence, and within-class reference-rank spread. Reference rank is provenance/diagnostic only and does not enter the DPO training weight. The canonical reference-remoteness-bank and task-SFT-DPO training base seed remains `2026070803`; each positive-beta DPO cell uses that base plus its configured DPO seed offset. The task SFT trainer independently inherits the existing P0 warm-start base seed `2026072900` (plus the existing deterministic task offset), and both DPO seed labels for a task reuse that same materialized task SFT adapter. Thus the positive-beta pair measures DPO-stage stochasticity conditional on one shared SFT initialization, not end-to-end SFT+DPO pipeline variability. These seed roles must be represented separately in machine-readable configuration rather than overloading one `initialization.seed` field. Historical fresh-LoRA/shared-SFT DPO paths keep their existing base-config seed semantics.

## Input scope

This successor experiment has no new Countdown scientific cell. Its active suite and parameter grid therefore contain only the eight P0 transfer tasks; `suite.external_tasks` is empty for this run, and no empty Countdown beta entry is retained merely for compatibility. Countdown remains historical external-validity context in reporting, but the 96-cell run must not require a Countdown bank, Countdown validation file, Countdown split materialization, Countdown adapter, or Countdown grid entry merely to prepare or execute these transfer-task cells.

## Reporting

Report, separately:

- task-performance metrics;
- validity / structure diagnostics;
- NaN/Inf numerical failure.

For positive-beta DPO cells also retain pair-margin, preference-accuracy, logit-saturation, raw-gradient norm, and optimizer-update norm diagnostics. Beta zero must be clearly identified as `sft_only_no_dpo_update`; its pair-margin, raw-gradient, and optimizer-step probes are not fabricated and are recorded as not run because no DPO objective/backward/optimizer step is executed, while exact policy/reference state-hash equality establishes the initialization copy.

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
