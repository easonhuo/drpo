# E8 Linear-C Extension Implementation Note

This implementation reuses the existing paper-aligned trainer, evaluator,
runtime scheduler and one-click launcher. No new training stack is introduced.
The existing adapter now selects one of two frozen profiles from the declared
grid config:

- completed Round 1: 16 points x 2 seeds = 32 cells;
- extension: 8 new c points x the same 2 seeds = 16 cells.

The extension profile is unavailable unless the config contains the exact
experiment ID, point list, predecessor run ID, predecessor result-manifest hash,
seed offsets, no-Positive-only declaration and unchanged evaluation contract.

The RunSpec binds execution to the implementation commit preceding the RunSpec
commit and enables deferred registration closure plus automatic text-first
result delivery to `easonhuo/drpo-results` on `ingest/e8`.
