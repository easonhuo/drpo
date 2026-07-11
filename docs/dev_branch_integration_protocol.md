# DRPO dev-branch integration protocol

**Claim:** `GOV-DEV-BRANCH-INTEGRATION-01`  
**Current implementation phase:** Batch 1, read-only planning and audit  
**Authoritative scope:** `docs/scopes/GOV-DEV-BRANCH-INTEGRATION-01.md`  
**V1 contract:** `docs/pipeline_handoffs/DEV_BRANCH_INTEGRATION_PIPELINE_V1_SPEC.md`

## Purpose

The tool turns a reviewer-approved dev-branch snapshot into a machine-audited integration transaction. Batch 1 does not modify the source worktree, create an integration branch, update the registry, normalize the handoff, run repository gates, push, open a pull request, or merge.

The tool is an orchestration layer. It does not replace the schema-v3 handoff authority, the formal experiment channel, the update-package pipeline, the test selector, or reviewer scientific judgment.

## Batch 1 commands

Validate request and reviewer decision without resolving remote refs:

```bash
python3 scripts/validate_dev_integration.py \
  --repo-root . \
  --request docs/integrations/<integration-id>/INTEGRATION_REQUEST.yaml \
  --json
```

Lock remote sources and audit the exact reviewed diff:

```bash
python3 scripts/integrate_dev_branch.py plan \
  --repo-root . \
  --request docs/integrations/<integration-id>/INTEGRATION_REQUEST.yaml \
  --transaction-root /persistent/path/drpo-integration-transactions \
  --json
```

Read a transaction:

```bash
python3 scripts/integrate_dev_branch.py status \
  --transaction-dir /persistent/path/drpo-integration-transactions/<integration-id>/attempt-0001 \
  --json
```

`--transaction-root` should be persistent and outside tracked repository paths. Batch 1 creates an isolated bare audit repository inside the attempt and removes it after a successful audit unless `--keep-audit-repo` is supplied.

## Inputs

### `INTEGRATION_REQUEST.yaml`

The request binds:

- exact remote main SHA;
- exact reviewed dev SHA;
- optional result-producing commit and dirty status;
- experiment IDs and/or governance claims;
- exact file operations with blob SHA and Git mode;
- a repository-relative reviewer decision path;
- requested future gate tier.

The schema is strict: unknown fields fail. `add` and `modify` use the same source/destination path; `rename` names both paths; `delete` omits destination, new blob, and new mode. `expected_old_blob_sha` is optional but recommended for modify/delete/rename.

### `REVIEW_DECISION.yaml`

The reviewer, not the tool, supplies:

- code integration eligibility;
- evidence level and result status;
- claim support level;
- terminal-audit state;
- task-performance collapse;
- support/boundary event;
- NaN/Inf numerical failure;
- limitations and unresolved issues.

These three event classes remain separate. Formal or closure evidence requires a clean result commit that is an ancestor of the reviewed dev SHA. Dirty evidence cannot be classified as formal or closure.

## Machine records

A successful Batch 1 attempt contains:

- `SOURCE_LOCK.json`: immutable resolved refs and input hashes;
- `SCOPE_AUDIT.json`: actual diff, operation/blob/mode audit, provenance relation, and reviewer decision;
- `TRANSACTION.json`: current state and next action.

A failed attempt contains:

- `DIAGNOSTIC.json`: stable error code, phase, message, source SHAs, and recovery guidance;
- `TRANSACTION.json`: `BLOCKED` state.

Every `plan` invocation creates a new numbered attempt. Existing `SOURCE_LOCK.json` and `SCOPE_AUDIT.json` files are never rewritten.

## Safety rules

The plan fails closed for:

- main or dev ref drift;
- unapproved, absent, copied, or unsupported changes;
- mismatched blob SHA or Git mode;
- symlinks, gitlinks/submodules, unsafe paths, control bytes, or case-fold collisions;
- direct dev imports of handoff, registry, handoff deltas, or generated views;
- missing or non-approving reviewer decisions;
- scientifically inconsistent provenance classifications.

Batch 1 reads the configured Git remote through native Git. If authoritative ref resolution is unavailable, it reports `SOURCE_UNRESOLVED`; it does not substitute cached chat or document SHAs.

## Not implemented yet

The following commands and states belong to Batch 2 and must not be inferred from Batch 1 success:

- `prepare` and clean integration worktree;
- selective file materialization;
- registry AST mutation;
- source commit;
- trusted normalize and atomic amend;
- required gates;
- local ready commit.

Push, pull-request, CI-polling, and merge automation are outside V1.
