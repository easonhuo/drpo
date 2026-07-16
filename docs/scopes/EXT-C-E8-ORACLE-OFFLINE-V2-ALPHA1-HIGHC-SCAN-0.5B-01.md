# EXT-C-E8-ORACLE-OFFLINE-V2-PAPER-ALIGNED-LINEAR-SCAN-0.5B-01

## Development status

- Lifecycle: code-first development pilot, not yet authoritatively registered.
- Result status: `pilot / not_run`.
- Scientific role: Countdown external-validity tuning only.
- Direct predecessor: commit `929142930a3e2efaa7cafc8e4afe3866600027a5`.
- Server execution may begin only after the real liveness gate passes.
- This pilot cannot support formal method ranking, convergence, steady state,
  controlled mechanism identification, or OOD claims.

## Authorized code delta

The existing trainer, runtime, evaluator, checkpointing, resume, aggregation,
and 8-GPU x 2-slot scheduler remain unchanged. The only objective change is:

```diff
- w = alpha * exp(-c * u**2)
+ w = alpha * exp(-c * u)
```

where `u=current_sequence_surprisal/2` remains detached. No calibration module,
new trainer, new scheduler, or new execution stack is permitted.

## Frozen Round-1 matrix

- Positive-only: one point.
- `alpha=1`: fifteen `c` values, including exact `c=0` uncontrolled endpoint.
- Development seed offsets: `{4000,5000}`.
- Total: `16 parameter points x 2 seeds = 32 cells`.
- Runtime: GPUs `0-7`, two cells per GPU, 16 concurrent cells, two waves.

This round localizes the useful coefficient region. It does not by itself
confirm superiority over Positive-only or Global. The historical
squared-surprisal trend remains historical directional evidence whether the
linear successor transfers, is inconclusive, or does not transfer.

## Inherited protocol

The model, frozen bank, fresh-LoRA initialization, unique-negative denominator,
optimizer, learning rate, 1200-step horizon, evaluation cadence, validation
split, test prohibition, and terminal reporting are inherited unchanged. Every
first-occurrence unique negative participates. Near/far selection, hidden
scaling, gradient-budget matching, weight-sum normalization, dynamic alpha,
SBRC, Hybrid, entropy bonus, SFT warmstart, on-policy sampling, replay refresh,
and test access remain forbidden.

## Selection and reporting

1. primary: mean Pass@8 over steps `800,900,1000,1100,1200`;
2. secondary: terminal Pass@8 at step `1200`;
3. auxiliary: Pass@64 and Greedy;
4. validity gate: valid-rate trajectory;
5. numerical audit: NaN/Inf and last-finite checkpoint.

Best-checkpoint Pass@8 is supplementary. Fixed 1200 steps are not convergence
or steady state. Task performance, valid-structure/support diagnostics, and
NaN/Inf numerical failure must be reported separately.
