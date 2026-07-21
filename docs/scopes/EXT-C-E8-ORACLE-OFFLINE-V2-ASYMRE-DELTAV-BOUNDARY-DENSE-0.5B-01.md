# EXT-C-E8-ORACLE-OFFLINE-V2-ASYMRE-DELTAV-BOUNDARY-DENSE-0.5B-01

## Status

- result status: `pilot`
- registration state: `dev_code_first_unregistered`
- execution state: `not_run`
- environment role: Countdown external validity only

## Claim

Resolve the previously unobserved AsymRE interval between `delta_v=-1.0` and
`delta_v=-0.5` on the frozen E8 V2 bank. The experiment asks whether any
weak-negative interior point exceeds the `delta_v=-1.0` zero-negative boundary
on fixed late-window Pass@8.

This experiment does not test whether AsymRE is superior to the registered taper
methods. It does not establish formal method ranking, significance, convergence,
steady state, cross-model generalization, or OOD generalization.

## Frozen scientific contract

- model: `Qwen2.5-0.5B-Instruct`
- initialization: pretrained base plus fresh LoRA for every cell
- response bank: frozen E8 V2 model-independent bank
- negative set: all unique negatives per prompt
- objective: `(1-delta_v) * positive_lp - (1+delta_v) * negative_lp`
- value network: none
- distance/remoteness control: none
- denominator: unique negative count per prompt
- optimizer, scheduler, learning rate, LoRA configuration, data geometry, and
  evaluation cadence: inherited unchanged from the predecessor AsymRE pilot
- training horizon: fixed 1200 steps, no early stopping

## Matrix

Eight explicit parameter points:

`[-1.0, -0.95, -0.9, -0.85, -0.8, -0.7, -0.6, -0.5]`

Paired development seed offsets:

`[4000, 5000]`

Exact matrix:

`8 parameter points x 2 paired seeds = 16 cells`

The two predecessor boundary points, `-1.0` and `-0.5`, are rerun as internal
anchors. Historical predecessor values may be shown for provenance but may not
replace the internally rerun anchors.

## Evaluation and reporting

- held-out file: `val.jsonl`
- role: structurally disjoint held-out evaluation
- training-loss access: forbidden
- separate `test.jsonl` access: false
- primary metric: mean Pass@8 over steps `[800, 900, 1000, 1100, 1200]`
- secondary metric: terminal step-1200 Pass@8
- additional metrics: greedy accuracy, Pass@64, valid-expression rate
- best validation checkpoint: supplementary diagnostic/recovery evidence only

Every declared point must be reported. An interior-point benefit claim requires
both the two-seed mean and paired per-seed differences against `delta_v=-1.0`.
No point may be selected from best-checkpoint performance.

Task-performance degradation, valid-expression or structure-boundary behavior,
and NaN/Inf numerical failure must be reported separately. Fixed 1200-step
training is not convergence.

## Liveness and execution

A two-step AsymRE smoke cell at `delta_v=-0.9` is required before the full
matrix. Smoke verifies runtime liveness only and is not scientific evidence.

The full matrix uses GPU 0--7 with two cells per GPU, for one 16-slot wave under
the existing E8 slot/runtime contract. Identity-checked resume and text-first
result delivery remain required.

## Expected outputs

- `SWEEP_PLAN.json`
- `RUNTIME_SELECTION.json`
- `RUNTIME_SLOTS.json`
- `SWEEP_STATUS.json`
- per-cell `summary.json`, `metrics.csv`, and diagnostics
- aggregate CSV/JSON summaries
- `aggregate/terminal_audit.json`
- `SWEEP_COMPLETE.json`
- durable text-first delivery to `easonhuo/drpo-results@ingest/e8`

## Gate

No full run is authorized merely by the existence of this scope or code. The
implementation SHA must be frozen, the RunSpec must pin that SHA, relevant tests
and static checks must pass, and launch requires explicit user approval.
