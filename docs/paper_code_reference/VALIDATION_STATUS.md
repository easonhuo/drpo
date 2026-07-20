# PAPER-CODE-VALIDATION-01 Current Validation Status

**Document role:** additive task-local validation status synchronized with the
existing migration documents.  
**Not a research master:** `docs/handoff.md` remains the unique research source
of truth.  
**Parent claim:** `PAPER-CODE-REFERENCE-01`.  
**Scientific-status impact:** none.

This file does not replace or delete
`docs/paper_code_reference/CURRENT_STATUS.md`. It supplies the live validation
phase that follows the migration snapshot recorded there.

## 1. Locked planning coordinates

- repository: `easonhuo/drpo`;
- default branch: `main`;
- current main observed during planning:
  `4b718e7439cf78a04f4affa1987ac15582d702d1`;
- validation branch: `dev/paper-code-reference-01`;
- validation planning base head:
  `8b81b4b72e38538a0b2ea4b50595059a67838d63`;
- Draft PR: `#149`, open, Draft, unmerged;
- baseline exact-head Evidence Locator and PR Gate: passed;
- integration-freshness audit against current main: pending.

These coordinates are audit facts. Resolve both branch heads again before any
execution or code repair.

## 2. Validation authorities

Read in this order after the mandatory repository startup protocol:

1. `docs/paper_code_reference/CURRENT_STATUS.md`;
2. this file;
3. `docs/paper_code_reference/ACCEPTANCE_MATRIX.yaml`;
4. `docs/paper_code_reference/VALIDATION_RUNBOOK.md`;
5. `docs/paper_code_reference/VALIDATION_MATRIX.yaml`;
6. `docs/paper_code_reference/SOURCE_MIGRATION_MAP.md`;
7. actual branch files, differential oracles, and exact-head CI.

The runbook owns validation order and pass/fail semantics. The validation matrix
owns live per-gate status. The original acceptance matrix continues to own
migration acceptance and scientific/non-scientific boundaries.

## 3. Current phase

Code migration is frozen for deadline-critical validation. Broad refactoring,
code-size reduction, and readability-only restructuring are out of scope.
Implementation may reopen only for a concrete correctness or real-environment
compatibility defect exposed by a validation gate.

The validation campaign stages are:

- V0: baseline and documentation freeze;
- V1: clean package, installation, full tests, and public entry points;
- V2: fixed-input function and first-update equivalence;
- V3: fixed-seed short-trajectory and checkpoint equivalence;
- V4: actual CPU, HDF5/MuJoCo, and Qwen/PEFT/CUDA liveness;
- V5: registered scientific reproduction and terminal review.

## 4. Live gate status at this write

- V0 planning coordinates: passed;
- V0 runbook: passed;
- V0 machine-readable validation matrix: passed;
- V0 additive validation-status synchronization: passed by this file;
- V0 original acceptance-matrix migration facts: unchanged;
- V0 exact-head CI after all validation-document commits: pending;
- V0 exact v0.1 package inventory: pending;
- V1--V5: not started by this documentation slice.

No experiment, smoke run, real liveness, or scientific reproduction was launched
by these documentation commits.

## 5. Existing engineering evidence and remaining gates

Existing migration evidence includes formula and detachment tests, C-U1 and D-U1
differential tests, Hopper/D4RL differential tests, Countdown fake-HF lifecycle
tests, first-update checks, compile, full pytest, Ruff, and public CLI tests.
Those records are prior engineering evidence. The validation campaign reruns the
required commands on its exact head and records them under one evidence contract.

Remaining deadline-critical gates are:

1. exact-head CI for the validation-document head;
2. exact `paper_code` v0.1 extraction, source commit, and file-hash manifest;
3. clean-package install, compile, full pytest, Ruff, and CLI checks;
4. focused fixed-input differential suites;
5. fixed short trajectories;
6. C-U1 and D-U1 real CPU liveness;
7. Hopper registered HDF5/Gymnasium/MuJoCo liveness when the data is available;
8. real Qwen/PEFT/CUDA Countdown liveness after a non-scientific schema-1
   liveness config is frozen;
9. registered full reproduction and terminal review only for final
   manuscript-required experiments.

D4RL-9 remains blocked for formal reproduction by unresolved dataset identities,
final method controls, formal seeds, budgets, checkpoint policy, and manuscript
role. Its code-level differential gate remains in V2.

## 6. Evidence and claim boundary

Smoke, fake-HF, short-trajectory, and liveness evidence are engineering evidence
only. Fixed-horizon completion is not convergence. Task-performance collapse,
support/variance/probability-boundary events, NaN/Inf numerical failure,
environment invalidity, and unresolved terminal state remain separate.

C-U1 remains same-distribution held-out-context or unseen-state generalization,
not OOD generalization. Hopper and Countdown remain external-validity evidence
and do not replace C-U1 or D-U1 controlled mechanism identification.

## 7. Immediate next gate

The next gate is V0 exact-head CI after the documentation slice. After it passes,
freeze and inventory the exact `paper_code` v0.1 package, then execute V1 exactly
as specified in `VALIDATION_RUNBOOK.md`.

Any code repair creates a new exact head and reruns every affected earlier gate.
Draft PR #149 remains unmerged until a separate explicit user instruction.