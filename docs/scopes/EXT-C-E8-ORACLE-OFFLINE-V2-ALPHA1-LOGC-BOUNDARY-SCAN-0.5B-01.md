# EXT-C-E8-ORACLE-OFFLINE-V2-ALPHA1-LOGC-BOUNDARY-SCAN-0.5B-01

## Development status

- Lifecycle: code-first development pilot, not yet authoritatively registered.
- Result status: `pilot / not_run`.
- Scientific role: Countdown external-validity tuning only.
- Registration closure must follow the temporary exact-commit fastpath rule active on `main`; no ad hoc registry/handoff construction is permitted.
- This scope authorizes implementation, command-contract validation, the existing real liveness gate, and validation-only server execution after a clean implementation SHA is frozen. It does not authorize a formal method ranking, convergence, steady-state, controlled-mechanism, or OOD claim.

## Question

The preceding alpha-one scans found a noisy but useful plateau through approximately `c=3--10` and a possible right-edge decline at `c=12`. They did not exclude a second useful region at substantially larger coefficients.

This pilot asks:

> Does the fixed-alpha family `w=exp(-c*u^2)` show a second late-window or terminal validation-performance peak for `c>12`, or do `c=16--128` remain on a flat/declining path toward the Positive-only endpoint?

The experiment does not assume that a second peak exists, that larger `c` is better, or that the previous `c=8` numerical winner is stable.

## Frozen matrix

Parameter points:

1. Same-seed strong control: `(alpha=0.5,c=1.0)`.
2. Previous anchors: `(alpha=1,c=8.0)` and `(alpha=1,c=12.0)`.
3. Logarithmic boundary extension: `(alpha=1,c in {16,24,32,64,128})`.

Development seed offsets: `{13000,14000,15000,16000}`.

Total: `8 parameter points x 4 seeds = 32 cells`.

Positive-only is intentionally not rerun. Historical Positive-only results remain unpaired context and may not be used as a paired control for this scan.

## Inherited scientific protocol

The following remain unchanged from the preceding E8 continuous-EXP pilots:

- Qwen2.5-0.5B-Instruct with fresh LoRA initialization per cell;
- frozen E8 V2 model-independent offline bank;
- every first-occurrence unique negative participates;
- `d=-stopgrad(sequence_mean_logprob)`, `u=d/2`;
- `w=alpha*exp(-c*u^2)`;
- loss denominator is the unique-negative count per prompt, never the weight sum;
- 1200 optimizer steps, no early stopping;
- Greedy and Pass@8 every 100 steps, Pass@64 every 200 steps;
- validation only; test split access is forbidden;
- no near/far selection, hidden scaling, budget matching, dynamic alpha, SBRC, Hybrid, entropy bonus, SFT warmstart, on-policy sampling, or replay refresh.

## Runtime-only inheritance

- 8 candidate GPUs;
- 2 runtime slots per selected GPU;
- up to 16 concurrent cells;
- 32 cells therefore occupy two full scheduling waves;
- exact clean checkout and identity-checked resume are mandatory;
- a real two-step representative liveness run at `alpha=1,c=32` must pass before the sweep;
- runtime selection does not change any scientific matrix field.

## Frozen evaluation and stopping interpretation

Selection order is fixed before execution:

1. primary: mean validation Pass@8 over steps `800,900,1000,1100,1200`;
2. secondary: terminal validation Pass@8 at step `1200`;
3. auxiliary: Pass@64 and Greedy;
4. structure proxy: valid-rate trajectory;
5. numerical audit: NaN/Inf and last-finite checkpoint.

Best-checkpoint Pass@8 is supplementary and may not select `c`.

A larger-`c` follow-up is scientifically motivated only if a candidate:

- improves late-window Pass@8 by at least `0.01` absolute over the same-seed `alpha=0.5,c=1` control or the `c=8` anchor; and
- has the same paired direction on at least 3 of 4 seeds; and
- does not introduce a valid-rate or numerical failure.

If the high-`c` points merely exchange rankings within seed noise, or all fail the above gate, this boundary search closes without automatic further extension. A fixed 1200-step horizon is not convergence or steady state.

## Required reporting

Report separately:

1. task performance: terminal and registered late-window metrics;
2. structure/support proxy: validation valid rate, explicitly not a formal support-boundary audit;
3. numerical failure: NaN/Inf, finite gradient spikes, and last-finite checkpoint state.

Countdown remains external-validity evidence and does not replace C-U1, D-U1, or D-Diag controlled mechanism identification.
