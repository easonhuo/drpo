# EXT-C-E8-ORACLE-OFFLINE-V2-PAPER-ALIGNED-TAU-CURVE-0.5B-01

## Status

- Lifecycle: code-first development pilot; authoritative registration deferred.
- Result status: `pilot / completed_pending_registration`.
- Scientific role: Countdown external-validity response-curve localization.
- The completed server result remains single-seed pilot evidence only.

## Question

After the paper-aligned linear coefficient curve has identified a broad useful
`c` region, does increasing a nonnegative near-retention threshold `tau` produce
an interpretable response trend at fixed `c`?

The frozen weight is

```text
u = current_sequence_surprisal / 2
w = alpha * exp(-c * max(u - tau, 0))
```

`u` and `w` remain detached. The clamp prevents weights above `alpha`.
`tau=0` is exactly the completed paper-aligned linear taper.

## Frozen matrix

- `alpha = 1`.
- `c = {1.609437912, 1.897119985, 2.995732274, 4.605170186}`.
- `tau = {0, 0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.25}`.
- development seed offset: `{4000}`.
- total: `4 c x 8 tau x 1 seed = 32 cells`.
- fixed horizon: `1200` steps; no early stopping.

The four `c` anchors cover the observed high-performance ridge and its right
shoulder. The dense single-seed design estimates curve shape efficiently; it
does not replace later cross-seed confirmation.

## Inherited controls

The model, frozen bank, fresh-LoRA initialization, unique-negative denominator,
optimizer, learning rate, scheduler, training horizon, evaluation cadence,
validation split, test prohibition, GPU selection, checkpointing, resume,
aggregation, and delivery path remain unchanged.

Positive-only is not rerun. Its completed two-seed result is plot context only.
The completed c-extension result remains immutable predecessor evidence.

## Reporting boundary

Report four fixed-`c` tau-response curves, raw points, late-window Pass@8,
terminal Pass@8, Pass@64, valid-rate diagnostics, and numerical failures.
Task-performance degradation, valid-structure proxy degradation, and NaN/Inf
failure must remain separate.

A monotone decline, rise-then-fall curve, or coherent cross-`c` interaction is a
valid trend result. An irregular single-seed surface must be reported as
inconclusive rather than repaired by post-hoc tau changes. No exact optimum,
formal method ranking, convergence, steady state, significance, or OOD claim is
allowed.

## Implementation boundary

No Python file is added. Existing paper-aligned common/runtime files receive the
minimal tau-compatible extension; the existing trainer, evaluator, scheduler,
launcher, and one-click shell entrypoint are reused. Test coverage is added to
the existing high-c scan test module.
