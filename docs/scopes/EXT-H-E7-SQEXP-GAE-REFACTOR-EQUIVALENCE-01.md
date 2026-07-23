# EXT-H-E7-SQEXP-GAE-REFACTOR-EQUIVALENCE-01

Read-only engineering gate comparing historical joint-critic commit `2fe97cdcff0e8361b33193dd2a7be8cf63c44a3b` with the refactored branch based on `main@85b0a68d77ed085a7f6e67771fb0f7672c43da09`.

Frozen synthetic matrix: TD/GAE × Positive-only/squared-EXP `c=128`; fixed eight-transition replay; nine updates; snapshot refreshes at updates 1, 5, and 9.

PASS requires exact old/new actor, critic, optimizer, advantage-table, and snapshot hashes at every update; scalar diagnostics use absolute tolerance `1e-12`.

This is not a scientific experiment, touches no held-out seed, changes no scientific variable, and cannot support convergence, steady-state, collapse, or method-ranking claims. Real-D4RL liveness remains a separate optional host-side check; no 1M-step rerun belongs to this gate.
