# EXT-C-E8-ORACLE-OFFLINE-V2-ALPHA1-C-SCAN-0.5B-01

## Development status

- Lifecycle: authoritatively registered development pilot; the server run started earlier from launch commit `a54dc74b849561c15f6195336fca446ed36f0640`.
- Result status: `pilot / not_run`.
- Scientific role: Countdown external-validity tuning only.
- Server execution is running under the validation-only pilot protocol. It is not a formal RunSpec and cannot support a method-ranking, convergence, steady-state, or OOD claim.

## Question

Fix `alpha=1` and scan larger exponential-quadratic decay coefficients to test whether the two-parameter controller

`w = alpha * exp(-c * u^2)`

can be reduced, on this E8 development environment, to the one-parameter family

`w = exp(-c * u^2)`.

The scan must retain Positive-only and the previous development winner `alpha=0.5,c=1.0` on the same new seeds. Without both same-seed controls, the alpha=1 family cannot be judged.

## Frozen matrix

Parameter points:

1. Positive-only: `(alpha=0,c=0)`.
2. Previous best control: `(alpha=0.5,c=1.0)`.
3. Alpha-one scan: `c in {1.5,2.0,2.25,2.5,3.0,4.0}`.

Development seed offsets: `{5000,6000,7000,8000}`.

Total: `8 parameter points x 4 seeds = 32 cells`.

## Inherited scientific protocol

The following remain unchanged from the continuous-EXP grid pilot:

- Qwen2.5-0.5B-Instruct with fresh LoRA initialization per cell;
- frozen E8 V2 model-independent offline bank;
- every first-occurrence unique negative participates;
- `d = -stopgrad(sequence_mean_logprob)`, `u=d/2`;
- loss denominator is the unique-negative count per prompt, never the weight sum;
- 1200 optimizer steps, no early stopping;
- Greedy and Pass@8 every 100 steps, Pass@64 every 200 steps;
- validation only; test split access is forbidden;
- no near/far selection, budget matching, hidden scaling, dynamic alpha, SBRC, Hybrid, or entropy bonus.

## Runtime-only inheritance

The prior completed 62-cell run established that two workers per 80-GiB GPU were viable and faster than one worker. This branch cleanly incorporates the same parent-controller two-slot scheduler:

- 8 candidate GPUs;
- 2 runtime slots per selected GPU;
- 16 concurrent cells;
- 32 cells therefore occupy two full scheduling waves;
- runtime slot count does not alter any scientific matrix field;
- clean exact checkout and identity-checked resume remain mandatory.

## Required reporting

Report separately:

1. task performance: terminal and steps 800--1200 late-window metrics;
2. structure/support proxy: validation valid rate;
3. numerical failure: NaN/Inf and last-finite checkpoint state.

A fixed 1200-step horizon is not convergence or steady state. The alpha=1 family is considered competitive only if its best `c` is stable across the four paired seeds, preserves valid rate, and matches or exceeds the same-seed `alpha=0.5,c=1.0` control rather than merely exceeding Positive-only.
