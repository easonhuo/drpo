# E8 Multitask Lambda-Only Successor Transport Protocol

Status: engineering precondition only; no successor scientific grid, seed set, cell count, or experiment ID is frozen by this note.

Base main commit: `9ad875a2f013ab9bab80849333831078d27e6121`.
Historical source experiment: `EXT-C-E8-MULTITASK-EXP-COLDSTART-01`.

## Purpose

The successor E8 multitask experiment will expose the exponential taper through the paper parameter `lambda` only. The paper scale is fixed to `c = 1` at the successor protocol level. The historical 208-cell COLDSTART artifacts and their raw `rho` provenance are immutable and are not rewritten by this change.

## Scientific invariants

The canonical paper trainer, taper implementation, negative consumer, optimizer, data, initialization, evaluation protocol, and frozen historical source blobs must not be changed by the parameter-transport cleanup.

For the canonical COLDSTART execution path, `src/drpo/e8_multitask_exp_tuning.py` already sends `cell.lambda_value` directly to the locked paper runtime/trainer as its exponential coefficient. Therefore the successor cleanup is limited to making a lambda-only `Cell` representable and routing that same `lambda_value` directly to the same canonical trainer. It must not reimplement the taper or loss.

The legacy `nine_task_rho_v1` path remains readable and executable for historical provenance. Removing `rho` from the successor interface does not authorize destructive deletion of legacy rho support.

## Equivalence gate

A formal successor run is forbidden until the lambda-only transport equivalence gate passes. The gate must establish all of the following:

1. For every lambda value admitted by the eventual frozen successor grid, the old-style cell representation and the lambda-only cell representation resolve to the exact same canonical exponential coefficient.
2. With fixed inputs and the same canonical paper implementation, both representations produce the same taper weights, loss contribution, and gradients; the test should require exact equality wherever the deterministic backend permits it and must report any backend limitation rather than silently widening tolerance.
3. The canonical paper source/config blob identities remain unchanged.

This gate proves parameter-transport equivalence only. It is not a scientific result and does not establish convergence, significance, method ranking, or external-validity claims.

## Not frozen here

This note intentionally does not freeze the successor experiment ID, lambda grid, seed offsets, task-specific cell counts, stopping criteria, or selection rule. Those scientific variables require their own documented protocol decision before a formal run.
