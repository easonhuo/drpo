# PAPER-CODE-VALIDATION-01 Execution Report — 2026-07-20

**Parent:** `PAPER-CODE-REFERENCE-01`  
**Scientific-status impact:** none  
**Draft PR:** #149, open, Draft, unmerged  
**Planning base:** `8b81b4b72e38538a0b2ea4b50595059a67838d63`

This additive report records the first executed validation gates. It does not
replace `CURRENT_STATUS.md`, `VALIDATION_STATUS.md`, or the research handoff.

## 1. Validation documentation

Added and synchronized:

- `VALIDATION_RUNBOOK.md`;
- `VALIDATION_MATRIX.yaml`;
- `VALIDATION_STATUS.md`;
- `VALIDATION_ACCEPTANCE_DELTA.yaml`;
- `VALIDATION_MATRIX_STATUS_DELTA.yaml`.

At documentation head `0201b9cde2b1f6e9b8d754fc7b7aacaa580c3f3e`:

- Evidence Locator run `29738490031`: passed;
- PR Gate run `29738490048`, job `88339432508`: passed;
- install, compile, shell syntax, handoff authority, formal execution channel,
  governance inventory/stage, full pytest, and Ruff: passed.

The exact v0.1 package inventory is still pending.

## 2. Isolated package and public entry points

`paper_code/tests/test_cli.py` now copies only the public package files into a
temporary directory, installs that copy with local build dependencies and no
dependency resolution, verifies that `drpo_reference` loads from the isolated
target, and checks help for the root command plus C-U1, D-U1, Hopper, D4RL, and
Countdown.

At head `6bb89fce283e508906f62bccbf32aa844e90528b`:

- Evidence Locator run `29739126054`: passed;
- PR Gate run `29739126098`, job `88341481345`: passed;
- isolated package test, full pytest, and Ruff: passed.

This passes package isolation and public-entrypoint loading. V1 remains in
progress until the exact package manifest and all runbook-listed clean-package
commands, including the explicit format check, are recorded.

## 3. Function and short-trajectory validation

The exact-head full pytest includes all currently registered shared, C-U1, D-U1,
Hopper, D4RL, and Countdown differential suites and short-path tests. They passed
at heads `6bb89fce...` and `67ad662f...`.

This is engineering evidence for formulas, masks, detachment, losses, raw
gradients, first updates, events, fixed short trajectories, checkpoint behavior,
and the controlled fake-HF Countdown lifecycle. It is not scientific
reproduction, convergence evidence, or method ranking.

## 4. C-U1 CPU CLI liveness

`paper_code/tests/test_cu1_suite.py` now invokes the actual public `cli.main` path
for C-U1 source-stage CPU smoke and verifies terminal audit, non-formal evidence
status, aggregate CSV, seed output, and Positive-only initialization checkpoint.

At head `67ad662f01c169167167269a3743e0e01ac65033`:

- Evidence Locator run `29739812639`: passed;
- PR Gate run `29739812652`, job `88343702063`: passed;
- the real C-U1 CLI smoke, full pytest, and Ruff: passed.

C-U1 CPU liveness is passed as non-scientific engineering smoke. Registered full
CPU reproduction and terminal scientific review remain pending.

## 5. D-U1 CPU evidence boundary

The same exact-head full pytest reran the existing D-U1 six-method real CPU smoke
runner, which passed. The separate D-U1 CLI parsing/dispatch test also passed.
Two attempts to add a new combined full-CLI test were rejected by the connector
safety layer and produced no branch change.

Therefore the status is deliberately split:

- D-U1 six-method CPU runner liveness: passed;
- D-U1 CLI parsing and dispatch: passed;
- one combined `cli.main -> six-method smoke -> terminal audit` gate: pending.

The two separate facts are not reported as a completed combined gate.

## 6. Remaining immediate gates

1. exact `paper_code` v0.1 extraction, source commit, and SHA-256 manifest;
2. explicit remaining V1 clean-package commands and format check;
3. combined D-U1 CLI CPU smoke;
4. Hopper registered HDF5/Gymnasium/MuJoCo liveness;
5. Countdown real Qwen/PEFT/CUDA liveness after freezing a visibly
   non-scientific schema-1 liveness config;
6. registered full reproduction and terminal review only for experiments used by
   the final manuscript.

D4RL-9 formal validation remains blocked by unresolved dataset identities, final
methods and coefficients, formal seeds, budget, checkpoint policy, and
manuscript role.

No formal experiment was launched. No scientific result status, method ranking,
or manuscript value changed.