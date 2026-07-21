# PAPER-CODE-VALIDATION-01 D4RL-9 Correctness Re-audit

Date: 2026-07-21

Canonical branch: `dev/paper-code-reference-01`

Audited pre-fix head: `610f1d97c31f1901f11bfa09c395cc7d60fd8c52`

Correctness-fix commit: `7dd83ef3d664089c9988358dc1c7625bf4a4acde`

Scientific-status impact: none.

## Scope

This audit covers the existing D4RL-9 reviewer task-performance backend. It remains separate from the Hopper E7-Q2 mechanism runner. No real dataset sweep, MuJoCo execution, formal seeds, convergence claim, or method ranking was produced.

## Evidence

Existing differential tests bind the migrated implementation to the selected canonical vendor code for the nine-task catalog, model initialization, forward values, the first Adam update, a fixed three-step trajectory, locomotion data transforms, unresolved-dataset fail-closed behavior, and use of one shared trainer across all tasks.

## Defect and repair

A helper promised detached control factors but direct calls could retain an autograd path through the distance input. The active training path already called it under `torch.no_grad()`, so prior losses and updates were unaffected.

Commit `7dd83ef3d664089c9988358dc1c7625bf4a4acde` detaches both helper inputs after validation and extends the existing control test to verify gradient-free outputs for all reviewer methods while preserving the exact formulas.

Only these files changed:

- `paper_code/src/drpo_reference/experiments/d4rl.py`
- `paper_code/tests/test_controls.py`

No formula, coefficient, method profile, optimizer, dataset transform, training schedule, output schema, or scientific coordinate changed.

## Tests

- `PYTHONPATH=src python3 -m pytest -q tests/test_controls.py`: 18 passed.
- Python compilation of both modified files: passed.
- A broader local test command timed out and is not claimed as complete.
- Ruff was unavailable locally and is not claimed as executed.
- Exact-head GitHub checks remain required.

## Decision

D4RL-9 reviewer backend engineering correctness is accepted for the migrated scope after the repair.

This does not establish real dataset identity for unresolved artifacts, real MuJoCo compatibility, a nine-task sweep, a frozen final protocol, convergence, terminal scientific acceptance, or method ranking.

## Next slice

Proceed to Countdown correctness acceptance. Do not implement resume or launch real Qwen/CUDA execution.
