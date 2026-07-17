# GOV-RUNSPEC-ENV-PREFIX-COMPAT-01

## Status

- Change class: compatibility fix.
- Base commit: `2f677f4b00954ea71d0efa7def552a1ea3daa565`.
- Scientific experiment status: unchanged.
- Triggering observation: the E8 linear-c extension RunSpec could not enter the governed lane because its valid shell-style leading environment assignments were interpreted as a script path.

## Root cause

The RunSpec contract accepted a command string, but the validator assumed that
the first token was the executable while the executor passed every token directly
to `subprocess.run(..., shell=False)`. A command of the form:

```text
WORK_DIR=... GRID_CONFIG=... bash scripts/run.sh
```

therefore failed script validation. Merely skipping the assignments during
validation would still leave execution trying to launch `WORK_DIR=...` as a
program.

## Authorized fix

1. Parse only consecutive leading `NAME=value` tokens as literal environment
   assignments.
2. Validate the checked-in script from the remaining argv.
3. Execute the remaining argv with a copied process environment plus the parsed
   assignments.
4. Keep `shell=False`; do not add arbitrary shell evaluation.
5. Apply the same behavior to the initial command and bounded recovery command.
6. Fail before claim when assignments are not followed by an executable.
7. Add end-to-end recovery tests covering quoted values and the fail-closed case.

## Reuse and boundaries

The change reuses the existing RunSpec validator, executor, recovery state
machine, artifact packager, results uploader, and scoped wrappers. It introduces
no launcher, uploader, scheduler, lane, repository, result schema, scientific
configuration, or registration authority.

It does not change:

- E7/E8 scientific variables, seeds, matrices, thresholds, budgets, or horizons;
- the completed E8 result or its result-repository commit;
- automatic-delivery policy;
- handoff or registry state;
- experiment launch authorization.

## Acceptance

- a command with two leading assignments and a quoted value validates and runs;
- the checked-in script remains the object protected and validated;
- initial and recovery attempts receive their declared values;
- assignment-only commands fail before claim;
- Python compilation, focused tests, full pytest, Ruff, handoff authority,
  formal execution channel, and governance validation pass on the exact PR head.

## Rollback

Revert the parser/executor, focused tests, documentation, and authorization
record together. Preserve all existing scientific outputs and result-repository
commits. RunSpecs requiring environment prefixes must not be launched after
rollback until rewritten with a separately reviewed compatible entrypoint.
