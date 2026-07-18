# E8 paper-aligned tau c-range extension protocol

## Frozen scientific delta

Relative to `E8_PAPER_ALIGNED_TAU_CURVE_20260717_01`, only the coefficient set
changes. The tau grid, seed offset, model, bank, LoRA initialization, objective,
training horizon, evaluation cadence, runtime topology, and reporting separation
remain unchanged.

```text
c = 1.386294361, 1.609437912, 4.605170186, 5.298317367
tau = 0, 0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.25
seed_offset = 4000
```

The configuration lists all 32 points explicitly in `c`-major, `tau`-minor order.
The explicit list is authoritative.

## Bridge logic

- `c=1.609437912` repeats the prior lower boundary.
- `c=4.605170186` repeats the prior upper boundary.
- `c=1.386294361` extends one previously used c-only coefficient step downward.
- `c=5.298317367` extends one previously used c-only coefficient step upward.

The bridge curves must be reported against the predecessor raw curves before the
outer curves are interpreted. Large bridge disagreement is a run-batch warning,
not evidence for a new c effect.

## Gates

1. Static validation proves exactly 32 unique `(c,tau,seed)` identities.
2. No Python file is added.
3. `tau=0` remains exactly the existing linear weight.
4. A two-step liveness cell at `c=5.298317367,tau=0.5` passes before the sweep.
5. Liveness is engineering evidence only.
6. The full run is validation-only; test access remains forbidden.
7. Fixed 1200 steps are not convergence.
8. All cells, raw trajectories, logs, aggregate files, and terminal audit are
   delivered durably before scientific review.

## Primary analysis

For each fixed c, compute

```text
mean Pass@8 at steps 800, 900, 1000, 1100, 1200
```

Show all eight raw tau points. Do not smooth away reversals. Also report the
paired bridge differences between this run and the predecessor run at the same
`c,tau,seed_offset` identities.

Predeclared interpretations:

- bridge curves agree coarsely and outer curves show moderate-tau benefit followed
  by high-tau degradation: range-extension support;
- bridge curves disagree by the scale of the claimed outer effect: inconclusive
  because run-batch variation is not separated;
- outer curves remain flat within bridge variation: no useful range extension;
- outer curves improve monotonically through tau=1.25: the current high-tau
  degradation pattern does not extend to that coefficient;
- irregular isolated peaks: inconclusive single-seed pilot.

## Event separation

Task performance, valid-rate structure proxy, NaN/Inf numerical failure, and
infrastructure failure must be reported independently. A valid-rate drop is not a
numerical collapse, and an OOM is not a NaN/Inf event.

## Claim boundary

This is still one seed. It can strengthen or weaken the observed response-surface
trend and audit run-batch bridges, but it cannot establish cross-seed robustness,
statistical significance, an exact optimum, a formal method ranking, convergence,
steady state, or OOD generalization.
