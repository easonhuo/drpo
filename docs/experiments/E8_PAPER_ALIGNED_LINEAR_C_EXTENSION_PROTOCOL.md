# E8 Paper-Aligned Linear-C Extension Protocol

Experiment ID:
`EXT-C-E8-ORACLE-OFFLINE-V2-PAPER-ALIGNED-LINEAR-C-EXTENSION-0.5B-01`

Status: code-first development pilot; not yet run and not formal evidence.

## Purpose

Round 1 already produced a dense two-seed curve from the uncontrolled endpoint
through `c=2.995732274`, including the paired Positive-only reference. This
extension does not repeat that matrix. It adds only eight previously unmeasured
coefficients:

```text
0.01, 0.025, 0.04,
3.506557897, 4.605170186, 5.298317367, 6.907755279, 9.210340372
```

The left three improve plot resolution below the previous minimum nonzero
coefficient. The right five test whether performance eventually falls back
toward Positive-only as exponential negative weights vanish.

## Frozen matrix

- `alpha=1.0` for every cell;
- seed offsets `4000,5000`;
- no Positive-only or uncontrolled rerun;
- 8 parameter points;
- 16 total cells;
- GPU `0-7`, two cells per GPU, one full-run wave.

The completed predecessor result is the only comparison source:

- run ID: `E8_PAPER_ALIGNED_LINEAR_SCAN_20260716_01`;
- result manifest SHA-256:
  `24635fbb634b23450cdfb560fd7b16a2dc0fe4a6d0586f10e1cf385e58bab333`;
- paired Positive-only late-window Pass@8: `0.1360,0.1436`;
- paired Positive-only mean: `0.1398`.

## Inherited execution contract

The following remain byte-for-byte or semantically inherited:

- frozen model-independent V2 bank and first-occurrence unique-negative rule;
- pretrained base plus fresh LoRA initialization;
- objective formula `alpha*exp(-c*u)`, `u=current_sequence_surprisal/2`;
- detached remoteness and weights;
- optimizer, scheduler, learning rate, gradient accumulation and denominator;
- 1200 fixed steps, no early stopping;
- validation-only evaluation; test split prohibited;
- Pass@8 every 100 steps and Pass@64 every 200 steps;
- checkpoint, identity-checked resume, aggregation and terminal audit;
- 16-slot resource policy and required two-step smoke gate.

No new trainer or launcher is introduced. The existing paper-aligned adapter
selects the frozen profile from the grid config.

## Selection and plotting

Primary metric remains mean validation Pass@8 over steps
`800,900,1000,1100,1200`. Secondary is terminal Pass@8 at step 1200. Pass@64,
Greedy, valid rate, numerical status and best-checkpoint metrics are mandatory
auxiliary diagnostics.

The plot may concatenate Round-1 and extension points because both use the same
formula, model/bank/training contract and seed offsets. Plot source rows must
retain their run IDs and manifest identities; the two executions must not be
presented as one physical sweep.

## Next decision

- a clear right-side fall toward `0.1398`: localize the peak using a later small
  refinement round;
- `c=9.210340372` still materially above Positive-only: document the observed
  persistence and decide whether one further right sentinel is justified;
- noisy or seed-reversed right-side behavior: do not open `alpha`, `tau`, or
  `scale_c`; first review terminal trajectories and paired differences;
- after the c-curve decision, `tau`/`scale_c` and any narrow-alpha scan require a
  separate frozen experiment.

Fixed-horizon results do not establish convergence, steady state, significance,
or a universal ranking.
