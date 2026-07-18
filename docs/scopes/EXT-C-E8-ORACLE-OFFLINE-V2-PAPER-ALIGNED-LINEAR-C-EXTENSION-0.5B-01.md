# EXT-C-E8-ORACLE-OFFLINE-V2-PAPER-ALIGNED-LINEAR-C-EXTENSION-0.5B-01

## Development status

- Lifecycle: code-first development pilot; deferred registration with closure required.
- Result status: `pilot / not_run`.
- Scientific role: Countdown external-validity coefficient localization only.
- Direct predecessor: `EXT-C-E8-ORACLE-OFFLINE-V2-PAPER-ALIGNED-LINEAR-SCAN-0.5B-01`.
- This pilot cannot establish formal method ranking, convergence, steady state,
  controlled mechanism identification, or OOD generalization.

## Authorized scientific delta

Only the explicit `c` point list changes. The weight remains

```text
w = alpha * exp(-c * u)
u = current_sequence_surprisal / 2
```

with detached `u` and `w`. `alpha` remains exactly `1.0` for every new cell.
The trainer, bank, fresh-LoRA initialization, optimizer, scheduler, 1200-step
horizon, evaluation cadence, denominator, runtime slots, resume logic, and
terminal audit remain unchanged.

## Frozen 16-cell matrix

New coefficient points:

```text
0.01
0.025
0.04
3.506557897
4.605170186
5.298317367
6.907755279
9.210340372
```

Paired development seed offsets remain `4000,5000`.

```text
8 c points x 2 seed offsets = 16 cells
```

The first three points fill the left edge below the previous minimum
`c=0.051293294`. The remaining five extend the previous right boundary
`c=2.995732274` and search for the descending branch toward the Positive-only
limit.

## Historical Positive-only reference

Positive-only is not rerun. The paired reference is taken only from completed
run `E8_PAPER_ALIGNED_LINEAR_SCAN_20260716_01`, result manifest
`24635fbb634b23450cdfb560fd7b16a2dc0fe4a6d0586f10e1cf385e58bab333`, on the
same seed offsets `4000,5000`. Its late-window Pass@8 values are `0.1360` and
`0.1436`, mean `0.1398`.

The extension results may be combined with that immutable Round-1 evidence for
curve plotting and same-seed comparisons. Other historical Positive-only runs
must not be silently pooled into this comparison.

## Runtime and delivery

- GPUs `0-7`;
- two cells per GPU;
- 16 concurrent cells;
- exactly one full-run wave after the required non-scientific smoke gate;
- validation-only tuning; test split forbidden;
- automatic text-first result delivery to `easonhuo/drpo-results`, branch
  `ingest/e8`;
- model/checkpoint/optimizer artifacts remain excluded.

## Reporting separation

Report separately:

1. task-performance degradation or improvement;
2. valid-structure/support proxy events;
3. NaN/Inf numerical failure.

Fixed 1200 steps are not convergence or steady state. Best-checkpoint metrics
remain supplementary to late-window and terminal reporting.

## Stop conditions

Stop and request a new protocol before changing `alpha`, adding `tau` or
`scale_c`, changing the bank, changing seeds, changing the horizon, changing the
selection metric, adding normalization, or accessing test data.
