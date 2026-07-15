# DRPO Code-First Pilot Registration Fastpath — Implementation Plan

**Claim:** `GOV-DEV-PILOT-REGISTRATION-FASTPATH-01`  
**Phase:** PR-A — thin preparation adapter and deterministic preflight  
**Base:** `c0ff38b51b0062b26a20771421f62b08eaaa0d12`  
**Scope contract:** `docs/scopes/GOV-DEV-PILOT-REGISTRATION-FASTPATH-01.md`  
**Incident ledger:** `docs/development_workflow_incident_and_improvement_log.md`  
**Incident annex:** `docs/development_workflow_incidents/DEVOPT-2026-07-14-PILOT-REGISTRATION-MERGE-01.md`

## 1. Document placement decision

The chronological incident ledger records what happened, evidence, root causes, and optimization status. This separate plan freezes implementation boundaries, interfaces, failure semantics, acceptance tests, and rollout. The plan and incident annex both link to the parent ledger; none of these files replaces `docs/handoff.md`, `experiments/registry.yaml`, or the accepted V1 integration specification.

## 2. Objective

Optimize the interval from a frozen reviewed implementation snapshot to an auditable V1 input set and, after the existing V1 stages run, a local `READY` commit.

```text
reviewed implementation SHA
→ reviewer-authored Pilot Registration Spec
→ deterministic preparation and preflight
→ existing V1 transaction
→ local READY
```

Success means:

- fewer manually assembled files and remote generation attempts;
- one frozen implementation identity;
- one deterministic registration candidate;
- no experiment-specific temporary workflow;
- unchanged scientific, provenance, authority, and final merge gates;
- locally decidable defects rejected before authority normalization or CI.

Time-reduction percentages remain hypotheses until measured on real PRs.

## 3. Existing capabilities to reuse

The accepted V1 already provides source locking, scope/provenance audit, isolated source-commit construction, optional reviewer-bound registration inputs, one-experiment add/replace mutation, trusted schema-v3 normalization, risk-selected gates, final parent/tree/freshness checks, diagnostics, rollback, and local `READY`.

PR-A must not reimplement those capabilities. It only compiles a compact spec into their existing inputs and performs static consistency checks.

## 4. CLI and input contract

```bash
python3 scripts/prepare_dev_pilot_registration.py \
  --repo-root . \
  --spec /path/to/DEV_PILOT_REGISTRATION_SPEC.yaml \
  --output-root /persistent/path/drpo-pilot-preparations \
  --json
```

The strict spec contains:

- exact repository/main/dev/result identity;
- one experiment subject and optional governance claims;
- exact V1 file operations with blob SHAs and modes;
- explicit reviewer identity, decision token, decision, limitations, and unresolved issues;
- registration mode `none`, `add_experiment`, or `replace_experiment`;
- for registration, complete reviewer-authored experiment mapping, handoff operations, registry changes, update ID, and replacement before-image hash when applicable.

The tool never invents a claim, experiment entity, result status, handoff text, registry declaration, reviewer identity, approval token, or scientific interpretation.

The checked-in template is deliberately non-approved. A runnable preparation requires the reviewer-authored spec to contain `approved: true` and `code_integration_eligible: true`; registration inputs are never emitted from an unapproved spec.

## 5. Correct output layout

Review found that V1 resolves `REVIEW_DECISION.yaml` from a repository-relative path. Therefore a flat external output directory would not be directly compatible. PR-A publishes two distinct layers:

```text
<output-root>/<preparation-id>/
  repository_overlay/
    docs/integrations/<preparation-id>/
      INTEGRATION_REQUEST.yaml
      REVIEW_DECISION.yaml
  transaction_inputs/
    REGISTRATION_INTENT.yaml       # registration mode only
    REGISTRATION_APPROVAL.yaml     # registration mode only
  PREPARATION_MANIFEST.json
  PREPARATION_REPORT.json
```

Usage is explicit:

1. place the exact `repository_overlay/` bytes at the corresponding repository-relative paths in the reviewed worktree, tracked or untracked according to the existing V1 operating procedure;
2. run V1 plan and Batch 2A using that exact request/review pair;
3. after the attempt reaches `PREPARED`, copy the exact `transaction_inputs/` bytes into the attempt directory;
4. run the existing V1 normalize, gate, and finalize stages.

The overlay is not silently applied by PR-A. The tool does not mutate the repository or transaction attempt.

`REGISTRATION_APPROVAL.yaml` is only a cryptographic binding record generated from an explicitly approved reviewer-authored spec. It does not authenticate a human or turn a pending review into approval.

Code-only mode omits `transaction_inputs/`.

## 6. Validation and atomic publication

Validation order:

1. parse strict YAML and reject unknown/missing keys;
2. validate IDs, full SHAs, modes, and normalized paths;
3. compile V1 request/review in memory;
4. invoke existing request, reviewer-decision, and provenance validators;
5. reject protected dev-import paths;
6. validate registration mode, target, reviewer approval, entity mapping, and registry-change coverage;
7. read the current registry only to verify add absence or replace presence and exact semantic before-image hash;
8. render all files deterministically and verify request/review/intent/approval hashes;
9. write to a temporary sibling directory;
10. validate the written overlay through the existing V1 validators;
11. atomically publish only after all checks pass.

Identical reruns reuse an existing output only when its deterministic manifest and every prepared input hash match. A conflicting directory fails closed. A failure publishes no partial preparation directory and may write only a structured engineering diagnostic outside tracked scientific evidence.

## 7. Architecture constraints

### One authority

Schema-v3 rendering and registry-event semantics remain owned by `scripts/handoff_authority.py`. PR-A neither renders handoff state nor materializes registry changes.

### One transaction

V1 remains the only integration state machine. PR-A has no `NORMALIZED`, gate, or `READY` state; its sole success state is `PREPARED_INPUTS`.

### One test selector

PR-A adds no impact map and changes no workflow. V1 continues using the existing selector. Operational CI tiering is deferred to separately reviewed PR-B.

### No publication responsibility

PR-A never performs Git network operations, commit publication, PR creation/refresh, Actions polling, merge, or branch deletion.

## 8. Commit model

PR-A is reviewed as two logical changes:

1. documentation freeze: scope, plan, incident annex, and template;
2. implementation: script and tests.

GitHub transport may create intermediate branch commits, but the eventual merge should be a clean squash commit.

For later scientific integrations, the desired history remains:

```text
implementation commit
→ immutable registration/result commit
→ reviewed merge commit
```

V1's existing atomic amend already makes registry, delta, materialized handoff, report, and generated views first reachable together. PR-A only prepares its inputs and does not create another atomic-registration engine.

## 9. Rollout

### PR-A1 — compiler core

- freeze scope, plan, incident annex, and spec template;
- implement strict parsing, existing-validator reuse, deterministic rendering, and atomic publish;
- add positive and negative unit tests;
- change no workflow file.

### PR-A2 — compatibility and replay

- validate generated overlay through the existing V1 validators;
- verify registration intent/approval hash binding;
- replay one immutable historical case or a faithful repository fixture;
- compare semantic preparation identities, not scientific outcomes;
- run exact-head repository checks.

PR-A may merge only after both parts pass review.

### PR-B — future, separately reviewed

- operational edit/standard/full CI tiers;
- final exact-head full merge gate retained;
- gate-duration and unique-blocker telemetry;
- immediate rollback to unconditional full CI.

PR-B is not authorized by this branch.

## 10. Acceptance matrix

Positive cases:

- code-only preparation;
- add-experiment preparation;
- replace-experiment preparation with exact before-image hash;
- byte-identical idempotent rerun;
- explicit reviewer-approved fixture accepted by existing validators;
- E8-style historical replay fixture preserving `pilot / not_run` semantics.

Negative cases:

- unknown key or missing required key;
- malformed/abbreviated SHA;
- unsafe, escaping, protected, duplicate, or case-fold-colliding path;
- unapproved review;
- dirty formal/closure evidence;
- registration target mismatch;
- add target already present;
- replace target absent or stale before-image hash;
- hash/reviewer binding mismatch;
- conflicting existing output;
- injected write failure leaving no published partial directory.

Repository acceptance:

- Python compile and Ruff;
- focused pytest;
- handoff authority verification;
- formal execution-channel validation;
- governance inventory/stage validation;
- selector-required repository tests;
- final changed-path and exact-head review.

## 11. PR-A telemetry

`PREPARATION_REPORT.json` records only preparation-local data: start/end, elapsed time, input/output hashes, mode, operation/file counts, idempotent reuse, and explicit `network_used=false` / `repository_modified=false`.

PR-A does not claim measured CI or merge savings. Cross-PR timing and unique-blocker telemetry belongs to PR-B or external GitHub analysis.

## 12. Rollback

The existing manual V1 input path remains valid. Rollback requires no migration:

1. stop invoking the new script;
2. preserve diagnostics as engineering evidence;
3. continue assembling existing V1 inputs manually;
4. revert only PR-A if necessary;
5. leave transaction, handoff, registry, scientific artifact, and closed-stage history unchanged.

## 13. Review record

### Review 1 — authority and scope

**Pass.** No handoff/registry edit, scientific change, protected-file modification, workflow change, or publication automation is authorized.

### Review 2 — interface and V1 compatibility

**Pass after correction.** The original flat-output idea was rejected because V1 resolves the review through a repository-relative path. The corrected repository-overlay plus transaction-input layout is explicit and testable.

### Review 3 — architecture and complexity

**Pass after narrowing.** The design is a compiler/preflight over existing V1 inputs, not a second authority, transaction, registry renderer, publisher, or test map.

### Review 4 — failure and rollback

**Pass.** Validation precedes output; temporary-directory publication, manifest-based idempotency, conflict rejection, structured diagnostics, faithful replay, and zero-migration rollback are required.

### Review 5 — efficiency-claim discipline

**Pass.** Expected gains are separated from measured evidence, and no existing final scientific or governance gate is removed.

## 14. Implementation decision

PR-A implementation may proceed. A need to modify protected authority, formal CI defaults, publication behavior, V1 core responsibilities, or scientific state triggers the scope stop conditions and requires separate authorization.
