# DRPO dev-branch integration protocol

**Claim:** `GOV-DEV-BRANCH-INTEGRATION-01`  
**Current implementation phase:** Batch 2B, trusted normalization and local `READY` closure  
**Authoritative scope:** `docs/scopes/GOV-DEV-BRANCH-INTEGRATION-01.md`  
**Batch 2B scope:** `docs/scopes/GOV-DEV-BRANCH-INTEGRATION-01-BATCH2B-NOTE.md`  
**V1 contract:** `docs/pipeline_handoffs/DEV_BRANCH_INTEGRATION_PIPELINE_V1_SPEC.md`

## Purpose and boundary

The transaction layer converts one reviewer-approved dev snapshot into a locally auditable integration candidate. It is deliberately split into narrow stages:

1. Batch 1 locks source refs, reviewer decisions, provenance, and exact file operations.
2. Batch 2A constructs an exact local source commit from those immutable records.
3. Batch 2B invokes trusted normalization, runs required gates, and records a local `READY` commit.
4. Batch 3 performs two real shadow observations and a rollback rehearsal before V1 acceptance.

The tool does not design experiments, classify evidence autonomously, replace reviewer judgment, or replace the existing handoff authority, formal experiment channel, update-package tooling, or test selector. V1 excludes automatic push, pull-request creation, CI polling, PR refresh, and merge.

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

Batch 2A rechecks the locked inputs and refs, builds the candidate tree in an isolated Git index using only approved blobs and modes, creates a source commit whose unique parent is locked main, verifies its exact diff and clean checkout, and advances the transaction to `PREPARED`.

A successful attempt adds:

- `integration-repo/`;
- `PREPARE_REPORT.json`;
- source-commit identifiers in `TRANSACTION.json`.

## Optional registration inputs

Code-only integration requires no additional inputs. When the reviewed change must also register or update one experiment, copy the templates into the untracked attempt directory:

```text
REGISTRATION_INTENT.yaml
REGISTRATION_APPROVAL.yaml
```

The two files must appear together. The approval binds:

- the exact intent SHA-256;
- the locked Batch 1 request SHA-256;
- the locked reviewer-decision SHA-256;
- the same reviewer ID and decision token already recorded by Batch 1.

The intent may add or replace exactly one experiment entity. Replacement requires the semantic SHA-256 of the exact existing experiment mapping. The target experiment must already appear in the original reviewed request. Experiment deletion and arbitrary registry path edits are not supported.

The intent carries complete schema-v3 handoff operations and registry change declarations. These are inputs to the existing authority; the Batch 2B adapter does not implement a second renderer or registry semantic validator.

## Batch 2B: normalize

```bash
python3 scripts/dev_integration_finalize.py normalize \
  --transaction-dir /persistent/path/drpo-integration-transactions/<integration-id>/attempt-0001 \
  --json
```

The normalizer stage:

1. rechecks the immutable request, review, Batch 1 records, Batch 2A report, source commit, and remote main/dev refs;
2. creates a clean detached trusted-current-main worktree;
3. in registration mode, applies one source-span registry mutation and creates one schema-v3 delta;
4. preserves local refs to the Batch 2A source commit and the pre-normalization authored commit;
5. invokes trusted current-main `scripts/handoff_authority.py normalize`;
6. restricts uncommitted normalizer outputs to the handoff after-image, one materialization report, and Stage 4A generated views;
7. amends the candidate so authored inputs and trusted materialized outputs first become reachable together;
8. verifies a clean one-parent commit and committed authority state;
9. writes `NORMALIZATION_JOURNAL.json` and `NORMALIZATION_REPORT.json`;
10. advances the transaction to `NORMALIZED`.

Code-only normalization must be a verified authority no-op and may not change final scope. A hard interruption while the transaction remains `PREPARED` is recovered by restoring the isolated repository to the immutable Batch 2A source commit before retrying. A normal validation failure closes the attempt and restores that source checkout for diagnosis.

## Batch 2B: required gates

```bash
python3 scripts/dev_integration_finalize.py gate \
  --transaction-dir /persistent/path/drpo-integration-transactions/<integration-id>/attempt-0001 \
  --json
```

The gate stage performs a fresh main/dev check, verifies the normalized commit is clean, and records all outcomes instead of stopping after the first independent failure. It runs:

- Git diff whitespace checks;
- compile and Ruff checks for changed Python files;
- committed handoff-authority verification;
- governance-stage validation;
- formal-execution-channel validation;
- the trusted current-main `select_update_tests.py` plan and selected commands.

The trusted impact map determines fast versus full mode. A requested fast mode cannot override a high-risk or unknown-path full selection. Every command receives a complete log under `gate-logs/`. The aggregate result is written to `GATE_REPORT.json`; success advances the transaction to `REQUIRED_GATES_PASSED`.

## Batch 2B: finalize

```bash
python3 scripts/dev_integration_finalize.py finalize \
  --transaction-dir /persistent/path/drpo-integration-transactions/<integration-id>/attempt-0001 \
  --json
```

Finalization performs the third remote main/dev freshness check, rechecks the normalized commit, unique parent, report hashes, passing gates, clean worktree, and committed handoff authority. It writes `READY_COMMIT.json` and advances the transaction to local `READY`.

`READY` means the exact local candidate passed the transaction gates. It does not push, open or refresh a PR, poll CI, or merge. Publication remains a separately reviewed manual action outside V1.

## Batch 2B records

A completed local transaction may contain:

```text
SOURCE_LOCK.json
SCOPE_AUDIT.json
TRANSACTION.json
PREPARE_REPORT.json
NORMALIZATION_JOURNAL.json
NORMALIZATION_REPORT.json
gate-logs/
GATE_REPORT.json
READY_COMMIT.json
integration-repo/
trusted-main/
```

Optional reviewer inputs remain separate from machine records:

```text
REGISTRATION_INTENT.yaml
REGISTRATION_APPROVAL.yaml
```

The transaction hash-binds the normalization, gate, and ready records. Local Git refs preserve the Batch 2A source commit and pre-normalization authored commit even though the final candidate is produced by amend.

## Safety rules

The pipeline rejects:

- main or dev ref drift at plan, prepare, normalize, gate, or finalize boundaries;
- request, reviewer decision, registration intent, or approval mutation;
- registration targets outside the reviewed experiment scope;
- operation drift or unapproved changed-path coverage;
- direct dev import of handoff, registry, deltas, generated views, or `.git` paths;
- path traversal, control bytes, target collisions, symlinks, gitlinks, unsupported modes, LFS pointers, and imported blobs larger than 10 MiB;
- destructive experiment removal or unrelated registry entity mutation;
- a registry change without exactly one new schema-v3 delta;
- source-authored materialization reports or generated views;
- unexpected normalizer outputs;
- a candidate commit whose unique parent is not locked main;
- dirty worktrees, report drift, HEAD drift, tree drift, failed gates, or stale authority state.

Every failure writes `DIAGNOSTIC.json`. Source drift moves the transaction to `STALE`; other failures move it to `BLOCKED`. Failed and stale attempts are preserved. Normal recovery is a new immutable plan attempt after the underlying source or implementation issue is corrected.

## Concurrency and hooks

An advisory attempt lock serializes Batch 2A and Batch 2B operations. Git hooks are disabled in the isolated repository, terminal credential prompts are disabled, and Git LFS smudge is skipped. The adapters create no network-side branch, pull request, or merge.

## Current acceptance boundary

Batch 1 and Batch 2A are accepted and merged. Batch 2B requires targeted tests plus repository PR gates before merge. Even after Batch 2B is accepted, the complete V1 remains incomplete until both real Batch 3 shadows and the rollback rehearsal pass.
