# E8 AsymRE Delta-V Scan Protocol

Experiment ID:
`EXT-C-E8-ORACLE-OFFLINE-V2-ASYMRE-DELTAV-SCAN-0.5B-01`

Status: code-first development pilot; not yet run and not formal evidence.

## Purpose

Measure the held-out task-performance response curve produced by canonical AsymRE's
additive baseline offset in the frozen E8 V2 one-step offline setting. The experiment
isolates `delta_v`; it does not add a critic or a distance-dependent controller.

## Objective

The branch-balanced signed objective uses

```text
A = R - delta_v
R_positive = +1
R_negative = -1
```

and therefore

```text
positive coefficient = 1 - delta_v
negative-repulsion coefficient = 1 + delta_v
```

The empirical prompt-level baseline is zero under the equal-total-mass positive and
negative branches. This is a controlled zero-baseline specialization of the AsymRE
objective, not a behavior-group reproduction of every data-collection detail in the
original paper.

## Frozen matrix

```text
delta_v: [-1.0, -0.5, -0.3, -0.2, -0.1, -0.05, 0.0, 0.1]
seed offsets: [4000, 5000]
parameter points: 8
cells: 16
```

`delta_v=-1` is the zero-negative-repulsion boundary. The first scan does not cross
below `-1` because that would flip wrong-response coefficients from repulsion to
imitation.

## Inherited training contract

- frozen model-independent E8 V2 bank;
- all unique negatives, first-occurrence expression deduplication;
- pretrained Qwen2.5-0.5B-Instruct plus fresh LoRA per cell;
- unchanged AdamW, scheduler, learning rate, accumulation, and clipping;
- 1200 fixed steps, no early stopping;
- unchanged unique-negative denominator and no weight-sum normalization;
- GPU `0-7`, two cells per GPU;
- identity-checked resume and terminal audit;
- no value network, learned baseline, taper, entropy bonus, or on-policy replay.

## Held-out evaluation contract

The runtime reads `val.jsonl` only as a structurally disjoint held-out evaluation split.
It does not enter the optimizer loss. Train/evaluation structure families and
`(numbers, target)` keys remain disjoint under the frozen split generator.

The response curve reports all eight declared points. Primary reporting is the mean of
Pass@8 over steps `800, 900, 1000, 1100, 1200`; step-1200 terminal Pass@8 is secondary.
Greedy, Pass@64, valid rate, structure diagnostics, and terminal-state diagnostics are
also preserved.

The trainer may save a `best_pass8_adapter`, but that checkpoint is supplementary only.
It is not used to construct the paper-facing `delta_v` curve and does not replace the
late-window or terminal result.

The separate `test.jsonl` file is not accessed. This is recorded as a file-access fact,
not as an absence of held-out evaluation.

## Execution and delivery

The pilot must run through RunSpec
`E8_ASYMRE_DELTAV_SCAN_20260721_01` on lane `e8`. Deferred registration requires later
registry/handoff closure. On successful completion, the canonical lane automatically
exports text-first evidence to `easonhuo/drpo-results`, branch `ingest/e8`.

The review package includes runtime plans, smoke and sweep status, cell summaries,
metrics, diagnostics, aggregate CSV/JSON, terminal audit, and logs. It excludes model,
adapter, checkpoint, optimizer, and other model-like files.

## Claim boundary

The pilot may support only a finite-horizon held-out coefficient-response observation.
It cannot establish statistical significance, convergence, steady state, formal method
ranking, OOD generalization, cross-model generalization, or universal AsymRE/DRPO
superiority. Task collapse, structure/support boundary events, and NaN/Inf numerical
failure remain separate.
