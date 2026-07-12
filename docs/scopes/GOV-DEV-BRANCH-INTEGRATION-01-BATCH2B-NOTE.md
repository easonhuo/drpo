# GOV-DEV-BRANCH-INTEGRATION-01 — Batch 2B implementation note

**Claim:** `GOV-DEV-BRANCH-INTEGRATION-01`  
**Batch:** `2B` — trusted normalization, required gates, and local `READY` closure  
**Development base:** `51fdfe194647cf58543701591bf9a0e7c2278330`  
**Development branch:** `dev/gov-dev-branch-integration-01-batch2b`  
**Approval record:** user authorized continuation after Batch 2A acceptance and merge.

## Purpose

Batch 2B completes the local write path defined by the lightweight V1 contract. It consumes a Batch 2A `PREPARED` transaction, optionally applies one separately reviewer-approved registry/handoff registration intent, invokes the existing trusted schema-v3 authority, runs the existing required-gate selector, and records a local `READY` commit.

This batch is an orchestration layer. It does not replace the registry semantic validator, handoff renderer, handoff authority, formal experiment channel, test selector, or reviewer scientific judgment.

## Approved implementation scope

- add a Batch 2B transaction adapter with `normalize`, `gate`, and `finalize` commands;
- preserve the Batch 2A source commit and the pre-normalization authored commit under local transaction refs;
- support code-only normalization as a verified authority no-op;
- support one optional, hash-bound `REGISTRATION_INTENT.yaml` plus matching `REGISTRATION_APPROVAL.yaml`;
- mutate exactly one experiment entity through a YAML AST/source-span adapter, without reserializing unrelated registry content;
- generate one schema-v3 authoritative delta using the existing renderer and history-preservation checks;
- call the trusted current-main `handoff_authority.py normalize`, then atomically amend all materialized outputs into the same local candidate commit;
- execute explicit invariant gates plus the existing `select_update_tests.py` plan;
- write `NORMALIZATION_REPORT.json`, `GATE_REPORT.json`, `READY_COMMIT.json`, logs, and stable diagnostics;
- perform the third main/dev freshness check before `READY`;
- add deterministic unit and local-Git fixture tests;
- update the protocol and optional input templates.

## Explicit non-scope

- no automatic push, PR creation, CI polling, PR refresh, or merge;
- no change to the default GitHub development route;
- no experiment launch or result interpretation;
- no autonomous evidence upgrade, method ranking, convergence claim, or collapse classification;
- no arbitrary registry edit language and no experiment deletion;
- no direct edit of `docs/handoff.md`;
- no modification of Stage 1, Stage 2, or Stage 5 protected files or responsibilities;
- no modification of scientific variables, datasets, seeds, thresholds, budgets, convergence standards, experiment duties, or execution order.

## Required invariants

1. The transaction must still bind the exact Batch 1 request, review, main SHA, dev SHA, and Batch 2A source commit.
2. Optional registration must be explicitly bound to the locked reviewer, request hash, review hash, and exact intent bytes.
3. The registration target experiment must be present in the original reviewed request scope.
4. Registry mutation may add or replace one exact experiment entity; unrelated top-level fields and experiment entities must remain semantically unchanged.
5. Registry changes require exactly one new schema-v3 delta and the existing authority must validate complete semantic coverage.
6. Source-authored deltas may not self-author materialization reports or generated views.
7. The final local commit must have the locked current-main commit as its unique parent.
8. Delta, materialization report, handoff after-image, registry after-image, and generated views must first become reachable together in the final amended commit.
9. Required gate selection must use the trusted current-main impact map; fast mode may not override a fail-closed full selection.
10. A passing gate report must be hash-bound before finalization.
11. Main or dev drift marks the attempt `STALE`; other failures produce `BLOCKED` plus `DIAGNOSTIC.json`.
12. V1 still ends at local `READY` and performs no network-side publication.

## Acceptance for this batch

Batch 2B becomes a merge candidate only after:

- targeted Batch 2B tests pass;
- Python compile and Ruff pass;
- handoff authority verification passes as a code-only no-op for this implementation PR;
- governance stage and formal-channel validators pass;
- full repository pytest passes;
- the PR diff contains only the approved files;
- no research handoff, registry, experiment configuration, or result-status change appears in the implementation PR.

Passing this batch does not complete V1. Batch 3 must still run a real code-only shadow, a real code-plus-registry/delta/pilot-summary shadow, and a rollback rehearsal.

## Rollback

Before V1 activation, rollback is a normal revert of the Batch 2B implementation commit. Local transaction directories and worktrees may be removed. Scope records, failed-attempt diagnostics, immutable handoff deltas, materialization reports, and historical governance evidence must not be destructively deleted.

## Remaining uncertainties

- whether real long-lived dev branches expose Git path or mode cases not covered by local fixtures;
- whether the current gate selector creates material duplicate runtime in full mode;
- whether the local authored-commit provenance ref should later be published by a separate publish adapter;
- whether registry replacement is needed frequently enough to retain after the two real shadows.
