# E8 source-snapshot gate simplification

Status: approved engineering simplification; no scientific experiment is launched by this change.

## Decision

For the E8 multitask cold-start / baseline-matrix runner, per-file `--source-file` snapshotting is not a launch or packaging requirement. The E8 runner must rely on commit-level provenance instead:

- exact full Git commit SHA;
- canonical repository identity;
- clean checkout at launch;
- tracked config availability at the launch commit;
- config hash and run/scientific manifests;
- terminal audit and package verification.

The generic hardened artifact tooling may continue to support optional `--source-file` snapshots for other callers. This scope only removes E8's use of that optional mechanism; it does not weaken exact-commit or clean-worktree checks.

## Rationale

A clean checkout bound to one full commit already fixes the complete repository tree. Maintaining a second hand-written list of source files duplicates that identity, creates maintenance drift when modules are split or renamed, and can fail or appear incomplete without changing the code actually executed by the run.

## Scientific boundary

This change does not alter tasks, data, banks, seeds, method definitions, hyperparameters, optimizer settings, training horizon, DPO semantics, scheduler seed barriers, evaluation policy, terminal-audit criteria, or result status. `EXT-C-E8-MULTITASK-BASELINE-MATRIX-01` remains `not_run`.
