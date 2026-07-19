# EXT-H-E7-SQEXP-GAE-REFACTOR-EQUIVALENCE-01

## Purpose

This is a read-only engineering equivalence gate for the E7 joint-critic TD/GAE refactor. It is not a scientific experiment and cannot upgrade any E7 result status.

## Compared implementations

- Historical joint-critic implementation: `2fe97cdcff0e8361b33193dd2a7be8cf63c44a3b`.
- Refactored implementation: the exact reviewed head of this gate branch, based on `main@85b0a68d77ed085a7f6e67771fb0f7672c43da09`.

The historical SHA must remain remotely resolvable. A failure to acquire either implementation is a hard gate failure, not permission to substitute another commit.

## Scientific boundary

The gate tests implementation equivalence only. It does not test method quality, convergence, steady state, universal GAE superiority, task-performance collapse, support/variance boundary events, or NaN/Inf collapse rates. It does not touch held-out seeds and does not authorize a formal D4RL launch.

## Layer 1: deterministic synthetic lockstep

The mandatory CI layer uses a fixed eight-transition ordered replay, a deterministic actor and critic state, SGD optimizers with fixed state, and a fixed nine-update transition-ID sequence. The nine updates cross snapshot refreshes at updates 1, 5, and 9.

The frozen matrix is:

- estimators: one-step TD and GAE with lambda 0.95;
- controls: Positive-only and squared-remoteness exponential control with `c=128`, `w(0)=1`, reference distance 2, and canonical alpha 0.11;
- canonical batch size for the synthetic replay: 2;
- cases: 4;
- updates per case: 9.

The gate records, before training and after every update:

- actor state SHA-256;
- critic state SHA-256;
- actor optimizer state SHA-256;
- critic optimizer state SHA-256;
- active advantage-table SHA-256;
- snapshot count and snapshot critic hashes;
- actor loss, critic loss, positive/negative fractions, and negative factor mean.

## PASS criteria

All four old/new cases must have identical:

- initial actor and critic state hashes;
- per-update actor and critic state hashes;
- per-update optimizer state hashes;
- per-update advantage-table hashes;
- snapshot counts, refresh positions, and snapshot critic hashes.

Recorded scalar diagnostics must agree within absolute tolerance `1e-12`. Any mismatch is a hard FAIL and must identify the first case, update, and field that diverged.

## Layer 2: real-data liveness

The real-data layer is deferred to the existing D4RL execution host because GitHub Actions does not possess the frozen Hopper HDF5 and canonical runtime tree. It must use one development seed only, Hopper medium-expert, TD and GAE, `c=128`, and exactly one full snapshot interval plus one update. It is not required before the synthetic lockstep result is known.

No complete 1M-step branch, additional ordinary seed, held-out seed, or nine-task sweep is part of this gate.

## Outputs

The deterministic gate writes:

- `OLD_TRACE.json`;
- `NEW_TRACE.json`;
- `SOURCE_IDENTITY.json`;
- `EQUIVALENCE_AUDIT.json`.

A PASS permits treating the refactor as update-semantic equivalent for the tested deterministic path. It does not prove bitwise equivalence of environment evaluation rollouts or make a scientific claim.
