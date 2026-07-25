# GOV-RUNSPEC-DEFAULT-RESULT-DELIVERY-01

## Status

- Change class: governance default-policy change.
- Scientific experiment status: unchanged.
- Base commit: `d042a60e6e665fc7f8761e97d41fa0a621f78b87`.
- Target lanes: `e7`, `e8`.
- Execution status: implementation review only; no experiment is started by this change.

## Problem

The results-repository uploader is already implemented and verified, but E7/E8
RunSpecs that omit `delivery` are interpreted as local-only. This creates a
silent failure mode: training may finish while the durable result handoff is
never requested.

Direct execution of an experiment one-click launcher also bypasses RunSpec state,
artifact packaging, upload, and deferred registration closure.

## Authorized behavior

1. When a claimed RunSpec has lane `e7` or `e8` and has no `delivery` key, the
   executor materializes the existing results-repository V1 contract:
   - `enabled: true`;
   - `auto: true`;
   - repository `easonhuo/drpo-results`;
   - branch `ingest/<lane>`;
   - profile `manifest_text_v1`;
   - 30 MB total and 10 MB per-file review-package limits.
2. The materialized contract is persisted before the RunSpec enters `running`,
   so `done` state, retry upload, and later audits use the same policy.
3. An explicit `delivery` block remains authoritative. In particular,
   `delivery.enabled: false` remains an explicit local-only choice.
4. The canonical operator entry is the existing scoped RunSpec wrapper.
   Underlying one-click launchers remain training-only and are not modified.
5. The current E8 paper-aligned linear scan uses deferred registration with
   closure required.

## Reuse and anti-duplication audit

Reused unchanged:

- `scripts/agent/runspec_results_delivery.py`;
- `deliver_completed_run`;
- results-repository append-only and idempotency logic;
- scoped server wrappers;
- existing RunSpec state machine, packaging, retry uploader, and locator closure.

No new uploader, launcher, scheduler, results schema, repository, lane, or
registration authority is introduced.

## Excluded scope

- automatic experiment start after GitHub merge;
- direct one-click upload behavior;
- scientific variable, seed, matrix, horizon, threshold, or result changes;
- handoff or registry mutation;
- experiment-status upgrade;
- modification of credentials or server wrapper installation.

## Acceptance

- missing delivery defaults correctly for both E7 and E8;
- explicit disabled delivery is preserved;
- the effective contract is persisted in the completed RunSpec state;
- automatic delivery is invoked and its immutable locator fields are returned;
- current E8 RunSpec validates as deferred;
- focused tests, Python compilation, Ruff, governance validation, and full pytest
  pass on the exact PR head.

## Rollback

Revert the executor default, focused tests, E8 registration adjustment, and
operator clarification together. Existing delivered results and locator records
must remain untouched. Operators may continue using explicit delivery blocks.
