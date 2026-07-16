# Runtime-resource acceptance server correction 01

**Claim:** `GOV-RUNTIME-RESOURCE-ACCEPTANCE-HARNESS-01`

**Failed acceptance source:** exact harness commit
`1f264c1ff09f9339a1b269ecca16de014fb44f3f`.

## Status of the first server run

The first target-server run remains an immutable engineering failure record.
It is not replaced or reinterpreted as a pass.

- Stage 0 topology/preflight: `PASS`.
- Stage 1 explicit resource pool: `PASS`.
- Stage 2 E7 measured-CPU path: `FAIL` after successful measured selection,
  because the PPO auto-plan wrapper expected `RUN_IDENTITY.json` even though the
  delegated plan command only materialized `EXECUTION_PLAN.json`.
- Stage 3 GPU placement: `FAIL` before the hardware probe because the outer
  resource-pool wrapper was launched with only the detached GPU worktree on
  `PYTHONPATH`, so it could not import the harness-owned resource-pool module.
- Stage 4 E8 thread scan: `BLOCKED` by Stage 3.
- Stage 5 concurrent pool isolation: `BLOCKED` by Stages 2 and 3.

There was no complete scientific sweep, OOM, NaN/Inf numerical failure,
orphan process, repository modification, or scientific result.

## Correction scope

### E7 plan identity

The PPO measured-CPU auto wrapper must materialize `RUN_IDENTITY.json` from the
just-written `EXECUTION_PLAN.json` when the delegated plan command has not
created the identity itself. The identity hash must use the same stable-plan
rule as the delegated run path: exclude only `created_utc`, then compute the
canonical JSON SHA-256. The selected worker count and immutable selection
digest are then bound into that identity.

This correction must not:

- run the scientific matrix during plan;
- invoke a second representative probe or throughput grid;
- change selected workers, selector arithmetic, seeds, steps, thresholds,
  configs, data, or scientific variables;
- weaken run-time identity or revalidation checks.

### Detached GPU worktree environment

The outer resource-pool wrapper must import code from the harness checkout.
The delegated GPU selection-only command must then receive a child-specific
`PYTHONPATH` whose first entry is the detached GPU worktree `src` directory.
The environment boundary is applied through the delegated command, after the
outer wrapper has activated and recorded the resource pool.

This correction must not:

- bypass the resource-pool wrapper;
- change the GPU pool or delegated `--gpus` validation;
- access the E8 test split;
- enter the complete E8 sweep;
- change thread candidates, placement arithmetic, phase envelopes, thresholds,
  model, bank, validation rows, or calibration.

## Required regression coverage

- A plan that writes only `EXECUTION_PLAN.json` creates a run identity with the
  canonical stable-plan digest and exact runtime-selection binding.
- The generated identity is accepted by the existing run identity gate.
- The outer GPU wrapper environment resolves the harness source tree.
- The delegated GPU command explicitly switches to the detached GPU source
  tree while preserving the declared physical `--gpus` argument.
- Thread-scan and concurrent E8 paths use the same split environment boundary.
- Existing no-test-split, timeout cleanup, text-only packaging, and affinity
  isolation tests remain passing.

## Acceptance gate

The correction is not accepted by unit tests alone. It requires:

1. exact-head GitHub Actions success;
2. a fresh target-server one-click run using the reviewed profile;
3. terminal audit of all stages;
4. no orphan, OOM, NaN/Inf, repository modification, or full scientific sweep;
5. explicit review before any merge or activation.

All resulting evidence is engineering evidence only.