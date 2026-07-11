# DRPO dev-branch integration protocol

**Claim:** `GOV-DEV-BRANCH-INTEGRATION-01`  
**Current implementation phase:** Batch 2A, local source-commit preparation  
**Authoritative scope:** `docs/scopes/GOV-DEV-BRANCH-INTEGRATION-01.md`  
**V1 contract:** `docs/pipeline_handoffs/DEV_BRANCH_INTEGRATION_PIPELINE_V1_SPEC.md`

## Purpose and boundary

The transaction layer converts one reviewer-approved dev snapshot into a locally auditable integration candidate. It is deliberately split into narrow stages:

1. Batch 1 locks source refs, reviewer decisions, provenance, and exact file operations.
2. Batch 2A constructs an exact local source commit from those immutable records.
3. Batch 2B will handle registry mutation, schema-v3 normalization, required gates, and the final local ready commit.
4. Real shadow observations follow only after Batch 2B passes its own acceptance.

The tool does not design experiments, classify evidence autonomously, replace reviewer judgment, or replace the existing handoff authority, formal experiment channel, update-package tooling, or test selector. V1 still excludes automatic push, pull-request creation, CI polling, and merge.

## Batch 1: plan and audit

Validate request and reviewer decision:

```bash
python3 scripts/validate_dev_integration.py \
  --repo-root . \
  --request docs/integrations/<integration-id>/INTEGRATION_REQUEST.yaml \
  --json
```

Lock remote refs and audit the exact dev diff:

```bash
python3 scripts/integrate_dev_branch.py plan \
  --repo-root . \
  --request docs/integrations/<integration-id>/INTEGRATION_REQUEST.yaml \
  --transaction-root /persistent/path/drpo-integration-transactions \
  --json
```

A successful Batch 1 attempt reaches `REVIEWED` and contains:

- `SOURCE_LOCK.json`;
- `SCOPE_AUDIT.json`;
- `TRANSACTION.json`.

## Batch 2A: local source commit

Run the write-path adapter on one successful Batch 1 attempt:

```bash
python3 scripts/dev_integration_write_path.py \
  --transaction-dir /persistent/path/drpo-integration-transactions/<integration-id>/attempt-0001 \
  --json
```

Batch 2A performs the following sequence:

1. rechecks the request and reviewer-decision byte hashes;
2. cross-checks request operations against the stored scope audit;
3. resolves remote `main` and dev refs again;
4. fetches the exact locked commits into a temporary local repository;
5. re-audits every approved main/dev tree entry and blob;
6. builds the candidate tree in an isolated Git index using only approved blob SHAs and modes;
7. creates a local source commit whose unique parent is the locked main SHA;
8. checks out the commit and verifies a clean worktree and exact committed diff;
9. writes `PREPARE_REPORT.json` and advances the transaction to `PREPARED`.

The isolated-index construction is intentional. The tool does not copy a whole dev worktree and does not execute arbitrary manifest commands. Add, modify, delete, and rename operations are represented as exact Git tree changes.

## Batch 2A records

A successful attempt adds:

- `integration-repo/`: a local Git repository checked out at the source commit;
- `PREPARE_REPORT.json`: parent, tree, source commit, exact committed changes, and input hashes;
- updated `TRANSACTION.json`: state `PREPARED` and source-commit identifiers.

Re-running Batch 2A on an intact `PREPARED` attempt is idempotent. If the repository and report were completed but the transaction update was interrupted, the adapter verifies both and safely repairs only the transaction state. Partial or inconsistent prepared artifacts fail closed.

## Safety rules

Batch 2A rejects:

- main or dev ref drift after Batch 1;
- request or reviewer-decision mutation;
- operation drift between the request and scope audit;
- unapproved or incomplete changed-path coverage;
- direct dev import of handoff, registry, deltas, generated views, or `.git` paths;
- path traversal, control bytes, target collisions, symlinks, gitlinks, and unsupported Git modes;
- blob or mode mismatches against the locked source trees;
- Git LFS pointers and imported blobs larger than 10 MiB;
- a transaction directory placed inside the source repository;
- a source commit with any parent other than the locked main commit;
- worktree, HEAD, tree, report, or committed-scope drift on an idempotent rerun.

Every failure writes `DIAGNOSTIC.json` and moves the transaction to `BLOCKED`, except source drift, which moves it to `STALE`. A failed or stale attempt is preserved; the normal recovery is a new Batch 1 attempt.

## Concurrency and hooks

One advisory file lock serializes Batch 2A executions for an attempt. Git hooks are disabled in the isolated repository, terminal credential prompts are disabled, and Git LFS smudge is skipped. The adapter creates no network-side branch, pull request, or merge.

## Current acceptance boundary

Batch 1 is accepted and merged. Batch 2A acceptance requires its targeted tests and the repository PR gates to pass. Even after Batch 2A is accepted, the complete V1 remains incomplete until Batch 2B and both registered shadow cases pass.

## Not implemented yet

- registry AST mutation;
- compact result-summary registration;
- schema-v3 delta generation and validation;
- trusted normalization and atomic amend;
- required test selection and gate execution;
- final local ready commit;
- code-only and code-plus-evidence shadow closure.

Push, pull-request, CI-polling, and merge automation remain outside V1.
