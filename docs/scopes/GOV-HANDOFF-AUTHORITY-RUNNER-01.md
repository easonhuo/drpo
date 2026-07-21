# GOV-HANDOFF-AUTHORITY-RUNNER-01 — path-scoped read-only PR gate

**Base:** `main@d1b163b6fc9eb58158cc5841f9bc1d4a719d3f69`  
**Approval:** repository owner explicitly approved installing the workflow on 2026-07-21.  
**Scientific impact:** none. No experiment is launched and no scientific variable, seed, threshold, result, claim, or execution order changes.

## Problem

A previous handoff/registry closure used a temporary GitHub Actions workflow as a remote
execution environment. Missing dependencies and an over-broad test boundary caused repeated
workflow edits, waits, and reruns. The repository already has a trusted Stage 5 authority
engine, but it does not have a permanent PR gate dedicated to authority-controlled paths.

## Authorized implementation

This change adds:

1. `.github/workflows/handoff-authority.yml`, a pull-request workflow that runs only when
   authority-controlled paths change;
2. `scripts/run_handoff_authority_gate.sh`, the single read-only command used by both CI and
   local checkouts;
3. static contract coverage in the existing `tests/test_pr_gate_log_workflow.py`;
4. this scope and one closed-stage reopen authorization record.

The gate uses the exact pull-request base and head SHAs to classify the diff. When an authority
path changed, it verifies:

- exactly one newly added schema-v3 delta accompanies committed handoff, registry, or generated-view changes;
- a new delta is not left unmaterialized;
- the committed handoff authority is valid;
- Stage 4A generated views are current;
- the governance stage ledger is valid.

## Trigger boundary

The full gate is limited to the handoff/registry authority surface, including the authoritative
delta tree, handoff and registry after-images, Stage 4A generated views, Stage 5 authority
engine/configuration, and this workflow's own governed files.

Ordinary training code, experiment scripts, tests, paper files, figures, and unrelated
documentation do not trigger this dedicated workflow. They continue to use the existing PR
gates.

## Explicit exclusions

- no branch write permission;
- no commit, push, PR creation, PR refresh, merge, or auto-merge;
- no remote materialization or automatic mutation of handoff/registry;
- no replacement of `scripts/handoff_authority.py`;
- no duplicate renderer, registry validator, or integration transaction engine;
- no weakening, removal, skipping, or reordering of existing PR gates;
- no new Python path;
- no scientific experiment execution or status upgrade.

Materialization remains an authored/prepared step performed through the existing trusted
normalization route. This workflow verifies the committed result; it does not create the result.

## Acceptance

The change is acceptable only when:

1. the workflow has `contents: read` and no write permission;
2. the workflow uses exact PR base/head SHAs and full history;
3. path filtering prevents ordinary code-only PRs from starting this dedicated job;
4. the shell command is syntactically valid and contains no normalize, commit, or push action;
5. static contract tests pass;
6. the new workflow passes on its own Draft PR;
7. existing full PR gates remain unchanged and pass.

## Rollback

Revert the single squash-merge commit for this claim. That removes the workflow, shell gate,
tests, scope, and authorization while preserving all historical handoff deltas, registry
records, scientific code, results, and existing PR gates.

## Remaining uncertainty

The first real Draft PR is required to confirm GitHub path filtering, dependency setup, exact
base/head availability, and Stage 5 verification behavior on the merge candidate. Until that
run passes, the workflow is installed only on the development branch and must not be described
as active on `main`.
