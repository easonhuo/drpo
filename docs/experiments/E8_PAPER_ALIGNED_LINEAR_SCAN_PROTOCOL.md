# E8 Paper-Aligned Linear-Coordinate Scan Protocol

Experiment ID:
`EXT-C-E8-ORACLE-OFFLINE-V2-PAPER-ALIGNED-LINEAR-SCAN-0.5B-01`

Status: code-first development pilot; not yet run and not formal evidence.

## Current authority

This document and the experiment scope supersede the previously registered
18-cell calibrated-lambda launch protocol **for new execution only**. The old
protocol, its handoff record, and its Git history remain preserved as historical
planning evidence. No result was produced by the removed implementation.

The current round isolates one scientific correction in the already tested E8
alpha1-c execution lineage:

```text
old: w = alpha * exp(-c * u^2)
new: w = alpha * exp(-c * u)
u   = current_sequence_surprisal / 2
```

`u` and `w` remain detached. This round does not simultaneously introduce a new
threshold, scale calibration, trainer, evaluator, scheduler, or aggregation
system. A calibrated `tau/scale_c` protocol may only be reopened as a separate,
explicitly registered later stage.

## Frozen matrix

- one Positive-only point: `(alpha=0,c=0)`;
- fifteen `alpha=1` points, including exact `c=0` uncontrolled-negative endpoint;
- coefficient grid:
  `0, 0.051293294, 0.105360516, 0.162518929, 0.223143551,
   0.287682072, 0.430782916, 0.693147181, 0.916290732,
   1.203972804, 1.386294361, 1.609437912, 1.897119985,
   2.302585093, 2.995732274`;
- paired development seed offsets: `4000,5000`;
- total: `16 parameter points x 2 seeds = 32 cells`.

The internal method label for `(alpha=1,c=0)` may remain `global` for backward
compatibility; scientifically it is the exact uncontrolled-negative endpoint.
It is not Positive-only.

## Inherited execution contract

The following are inherited rather than reimplemented:

- frozen model-independent V2 bank and first-occurrence unique-negative rule;
- fresh-LoRA initialization;
- optimizer, scheduler, learning rate and gradient accumulation;
- 1200-step fixed horizon with no early stopping;
- validation-only tuning and test-split prohibition;
- Pass@8 every 100 steps and Pass@64 every 200 steps;
- checkpointing, last-finite guard, identity-checked resume and aggregation;
- terminal audit separating task performance, valid-structure/support proxy and
  NaN/Inf numerical failure;
- GPU 0-7 with two cells per GPU, total concurrency 16.

All eight GPUs must pass the existing resource selector. Selection of fewer than
eight GPUs fails closed rather than silently changing the intended two-wave
schedule. The two-step smoke gate is required and is not scientific evidence.

## Selection rule

Primary metric: mean validation Pass@8 at steps
`800,900,1000,1100,1200`.

Secondary metric: terminal validation Pass@8 at step `1200`.

Pass@64, Greedy, valid-rate trajectories, best checkpoint, numerical status and
last-finite status remain mandatory diagnostics. Best-checkpoint performance is
supplementary and cannot replace late-window or terminal reporting.

Round 1 only localizes a coefficient region. It cannot establish convergence,
steady state, a universal method ranking, controlled causal identification or
OOD generalization. Countdown remains external-validity evidence.

## Next-round decision

After all 32 cells and the terminal audit are complete:

- best coefficient on the left boundary: extend left;
- best coefficient on the right boundary: extend right;
- best coefficient in the interior with stable paired-seed ordering: refine
  locally around neighboring coefficients;
- unstable or disordered paired-seed ordering: add fresh development seeds
  before opening another scientific variable;
- coherent curve but no advantage over Positive-only: report non-improvement
  under this fixed-coordinate successor; do not retroactively invalidate the
  historical squared-surprisal observation.

The decision may be computed from the preserved per-cell `metrics.csv` files;
no test-set information may enter the decision.
