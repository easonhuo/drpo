# EXT-C-E8-ORACLE-OFFLINE-V2-ALPHA1-HIGHC-SCAN-0.5B-01

## Development status

- Lifecycle: code-first development pilot, not yet authoritatively registered.
- Result status: `pilot / not_run`.
- Scientific role: Countdown external-validity tuning only.
- The implementation is a non-destructive successor to
  `EXT-C-E8-ORACLE-OFFLINE-V2-ALPHA1-C-SCAN-0.5B-01`.
- Server execution may begin only after the real liveness gate passes. This
  pilot cannot support formal method ranking, convergence, steady-state,
  controlled mechanism, or OOD claims.

## Question

Fix `alpha=1` and extend the exponential-quadratic coefficient into the high-c
region. Determine whether late-window and terminal validation performance keeps
improving, forms a plateau, or returns toward the Positive-only endpoint.

## Frozen matrix

Parameter points:

1. Strong same-seed control: `(alpha=0.5,c=1.0)`.
2. Previous alpha-one anchors: `(alpha=1,c=3.0)` and `(alpha=1,c=4.0)`.
3. High-c extension: `(alpha=1,c in {5,6,8,10,12})`.

Development seed offsets: `{9000,10000,11000,12000}`.

Total: `8 parameter points x 4 seeds = 32 cells`.

Positive-only is not rerun in this tuning stage. Historical Positive-only runs
may be shown only as unpaired context and cannot be used for paired inference.

## Inherited scientific protocol

The model, frozen bank, fresh-LoRA initialization, objective, unique-negative
denominator, optimizer, learning rate, 1200-step horizon, evaluation cadence,
validation split, and test prohibition are inherited unchanged from the
predecessor. Every first-occurrence unique negative participates. Near/far
selection, hidden scaling, gradient-budget matching, weight-sum normalization,
dynamic alpha, SBRC, Hybrid, entropy bonus, SFT warmstart, on-policy sampling,
replay refresh, and test access remain forbidden.

## Selection and reporting

The selection order is frozen before seeing results:

1. primary: mean Pass@8 over steps `800,900,1000,1100,1200`;
2. secondary: terminal Pass@8 at step `1200`;
3. auxiliary: Pass@64 and Greedy;
4. validity gate: valid-rate trajectory;
5. numerical audit: NaN/Inf and last-finite checkpoint.

Best-checkpoint Pass@8 is supplementary and must not select `c`. Fixed 1200
steps are not convergence or steady state. Task performance, valid-rate
structure proxy, and NaN/Inf numerical failure must be reported separately.
