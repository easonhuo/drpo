# ReplayAB R2 Post-Implementation Review

Work ID: `REPLAYAB-R2-SEMANTIC-ACCEPTANCE-01`

Review base: `6368009ce718f1f87bea4f8a637f40aab1a55242`

Status: `NARROW_FIX_REQUIRED`

## Finding

The first bounded implementation binds an AcceptanceResult to the case ID, run ID, acceptance-contract digest, and evaluator digest. The immutable Run Artifact also contains the outcome locator and the evaluator-result locator.

However, the AcceptanceResult does not directly name the exact outcome evidence digest that it evaluated. A substituted outcome under the same deterministic run identity would therefore rely on co-location inside the Run Artifact rather than an explicit evaluator-to-outcome content binding.

## Required correction

Add an `outcome_sha256` field to the AcceptanceResult schema and require it to match the exact outcome evidence locator SHA-256 before semantic acceptance is loaded.

Add a regression test that changes only this binding and requires fail-closed evidence rejection.

Synchronize the two semantic fixture AcceptanceResults and their Run Artifact result locators.

## Scope

This correction:

- does not alter the frozen twelve-case expected-verdict bank;
- adds one post-review regression test outside that original count;
- does not execute an evaluator;
- does not add a Python file, plugin, backend, sandbox, worker, trajectory, network call, authority change, handoff change, registry change, or scientific change;
- must remain within the preferred production-code budget.

R2 closure review remains blocked until the corrected exact-head focused tests, repository gate, calibration audit, and R1 non-regression audit pass.
