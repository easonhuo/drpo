# GOV-DEV-BRANCH-INTEGRATION-01 V1 scope, authorization, and rollback

## Purpose

Authorize development of a lightweight local transaction framework that converts a reviewer-approved dev-branch snapshot into a current-main-based local ready commit while preserving DRPO's existing scientific and governance authorities.

This claim does not authorize a new research experiment, a scientific result, an automatic merge path, or a change to any closed governance Stage responsibility.

## Base and approval

- Base commit: `d94eb5d7231653f557e66c6ae0b1cc4fa008ef27`
- Claim: `GOV-DEV-BRANCH-INTEGRATION-01`
- Phase: `v1_local_transaction`
- Approval record: `user_approved_2026_07_11_finalize_lightweight_v1_documents_then_begin_development`
- Research experiment impact: none
- Scientific variable impact: none
- Governance classification: standalone new tooling adjacent to, but not part of, the closed Stage 1 package pipeline

The user explicitly approved finalizing the lightweight design documents, committing them to the repository, and then beginning development.

## Authority boundary

V1 remains subordinate to:

- `AGENTS.md`;
- `docs/handoff.md`;
- `experiments/registry.yaml`;
- `docs/governance_pipeline_stage_status.yaml`;
- the production schema-v3 handoff authority;
- the canonical formal experiment channel;
- the existing update-package producer/verifier and test selector.

V1 may invoke these authorities through documented public entry points. It may not duplicate, replace, weaken, or silently alter them.

## Authorized documentation changes

- add `docs/pipeline_handoffs/DEV_BRANCH_INTEGRATION_PIPELINE_V1_SPEC.md`;
- add `docs/pipeline_handoffs/README.md` as an index;
- add this scope/authorization/rollback record;
- retain `DEV_BRANCH_INTEGRATION_PIPELINE_BUILD_BRIEF.md` as the long-term blueprint and historical failure analysis.

## Authorized implementation scope

Development may add or modify only the following initial V1 surfaces, subject to reviewer approval of the exact diff:

- `docs/dev_branch_integration_protocol.md`;
- `docs/templates/dev_integration_request.yaml`;
- `docs/templates/dev_integration_reviewer_decision.yaml`;
- `docs/integrations/README.md`;
- `scripts/integrate_dev_branch.py`;
- `scripts/validate_dev_integration.py`;
- `tests/test_dev_branch_integration.py`;
- narrowly scoped fixture files under `tests/fixtures/dev_branch_integration/` if needed.

The implementation may call existing public scripts but may not modify their protected bytes in V1.

## Authorized V1 capabilities

- strict request and reviewer-decision validation;
- authoritative source-ref resolution with explicit failure when unavailable;
- immutable source locking;
- diff, scope, path, blob, mode, rename, delete, and provenance audit;
- local clean worktree creation;
- allowlist-only selective import;
- registry updates through a structured YAML path rather than string insertion;
- trusted handoff normalization through the existing authority entry point;
- source commit plus atomic amend;
- existing required-gate selection and execution;
- local ready-commit output;
- machine-readable transaction, gate, ready, and diagnostic records;
- idempotent phase commands and new-attempt semantics after input drift;
- local fixtures, fault injection, and two real shadow observations.

## Explicitly excluded

- direct edits to `docs/handoff.md`;
- direct edits to `experiments/registry.yaml` outside the trusted integration transaction being shadowed;
- modifications to Stage 1, Stage 2, or Stage 5 protected files;
- modifications to `AGENTS.md` or the default repository workflow;
- a new blocking GitHub Actions workflow;
- automatic push, pull-request creation, pull-request refresh, CI polling, merge, or branch deletion;
- branch-protection changes;
- automatic scientific review, evidence upgrading, convergence claims, or method ranking;
- modification of experiment code, configs, datasets, seeds, thresholds, stopping rules, result status, or execution order as part of this governance claim;
- databases, services, Web UI, workflow-language plugins, multi-repository support, or task scheduling;
- destructive deletion or rewriting of historical integration, experiment, handoff, delta, report, or closure records.

## Development sequence

### Documentation gate

The V1 implementation contract and this scope record must be committed before production code is written.

### Batch 1: read-only core

Implement schemas, source lock, scope/path/provenance audit, status and diagnostics, plus deterministic local Git fixtures.

Entry to Batch 2 requires targeted tests and reviewer diff approval for Batch 1.

### Batch 2: local write path

Implement selective import, structured registry mutation, trusted normalization, atomic amend, required gates, and local ready commit.

Entry to Batch 3 requires local integration fixtures to pass, including the historical stale-main/old-registry/non-atomic-report failure modes.

### Batch 3: shadow hardening

Run one code-only and one code-plus-registration shadow transaction. The tool must not push, open a PR, or merge during shadow.

Any publish/PR/merge automation requires a separate user-approved scope after shadow evidence is reviewed.

## Acceptance

V1 is acceptable only when:

1. exact main/dev/result SHAs are recorded and drift fails closed;
2. input and machine-generated records cannot overwrite each other;
3. unapproved files, unsafe paths, symlinks, gitlinks, case collisions, undeclared file modes, deletes, or renames are rejected;
4. dev-side stale handoff, registry, delta, and generated views cannot be imported by default;
5. registry mutations are structurally validated;
6. trusted normalization is reused rather than reimplemented;
7. delta/report/materialized outputs satisfy the existing atomic-history authority checks;
8. the existing test selector and governance validators determine required gates;
9. every failure creates a durable structured diagnostic;
10. old attempts remain immutable when main/dev inputs drift;
11. two real shadow observations and a rollback rehearsal pass;
12. no scientific result or status is upgraded by the tool;
13. no closed Stage responsibility or protected after-image changes.

## Rollback

Before default adoption:

- stop using the CLI and discard its local worktree/transaction directories;
- revert the single implementation PR or the specific implementation commits;
- preserve this scope, authorization, tests, shadow reports, diagnostics, and historical failure evidence;
- keep all existing Stage closures, formal experiment authorities, handoff deltas, materialization reports, registry history, and scientific results unchanged;
- never repair a failed transaction by deleting or rewriting its recorded attempt; create a new attempt instead.

No data migration or scientific rollback is required because V1 does not own authoritative research data and is not a default execution path.

## Remaining uncertainties

- authoritative Git ref resolution must be validated both in local Git and in the available GitHub-connected execution environment;
- the smallest safe structured registry mutation interface may require one extra design adjustment after Batch 1 fixtures;
- real full-gate duration and unique blocker rate are unknown until the two shadow cases run;
- publish/PR automation may prove unnecessary if V1 reduces manual integration to an acceptable level.
