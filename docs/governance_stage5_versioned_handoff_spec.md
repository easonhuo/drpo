# Stage 5 — Versioned Handoff Write-Authority Candidate

**Governance claim:** `GOV-HANDOFF-AUTHORITY-CUTOVER-01`
**Implementation state:** hardened candidate; ready for pre-cutover acceptance; no authority cutover
**Original design base:** `4ad8b09ca80bc4b98aebffc6540f9be29440ba28`
**Pre-cutover hardening base:** `dacddafbf3e3caf560e87c145ab92d35b8d7fef1`

## 1. Purpose

Stage 5 promotes the validated Stage 3 append-oriented delta engine into a future handoff write authority without changing the current read contract. `docs/handoff.md` remains the unique research master read by users and tools. On the current `main` branch it also remains the manual write authority.

After a separately authorized cutover, handoff changes are reconstructed from an immutable checkpoint plus immutable schema-v3 `HANDOFF_DELTA.yaml` files. A trusted normalizer materializes `docs/handoff.md` and refreshes the Stage 4A minimal generated views.

## 2. Current candidate boundary

The checked-in authority state remains:

```yaml
mode: manual
human_handoff_write_authority: true
direct_handoff_edit_forbidden: false
authority_cutover_allowed: false
```

The hardened candidate implements the lifecycle commands and acceptance tests, but it does not execute a transition on `main`. A real cutover still requires a separate, already-committed `stage_transition` authorization whose scope explicitly permits the manual-to-delta transaction and names the checkpoint ID.

This work does not:

- activate schema-v3 production updates on `main`;
- event-source `experiments/registry.yaml`;
- make Stage 4B a runtime-generated view;
- compact `AGENTS.md`;
- modify scientific claims, experiments, seeds, thresholds, gates, or results.

## 3. Future delta-authority model

```text
immutable checkpoint
        +
immutable schema-v3 deltas
        ↓
trusted normalizer
        ├── docs/handoff.md
        └── Stage 4A minimal generated views
```

The normalizer uses the implementation and policy from the pre-integration current-main checkout. Ordinary content packages may not modify the authority config, checkpoint, accepted v3 delta, accepted materialization report, normalizer, delta policy, updater normalization hook, Stage 4A taxonomy, stage ledger, authorization records, or `AGENTS.md`.

## 4. Schema v3

A future authoritative delta has:

```yaml
schema_version: 3
update_id: EXAMPLE-2026-07-01
mode: authoritative
base:
  commit: <40-character SHA>
  handoff_sha256: <64-character SHA-256>
  registry_sha256: <64-character SHA-256>
renderer_version: 1
operations: []
registry:
  mode: unchanged  # or expected_after
  exact_base_after_sha256: null
  changes: []
expected:
  exact_base_candidate_sha256: <64-character SHA-256>
```

At least one handoff operation or one declared registry change is required. The existing append-oriented renderer remains the sole renderer core. Arbitrary replacement and destructive deletion remain unsupported.

## 5. Package classification and normalization

Delta authority does not mean that every repository update is a research-authority update. The trusted normalizer first classifies the source package:

```text
handoff unchanged + registry unchanged + authority/control plane unchanged
→ verify current authority state
→ deterministic normalization no-op
→ ordinary code, test, or documentation update may continue

handoff or registry semantic state changes
→ require exactly one newly added immutable schema-v3 delta
→ validate exact-base intent
→ materialize on current state

control-plane or authority state changes
→ reject as an ordinary content package
→ require the separately authorized lifecycle/governance path
```

A code-only no-op never creates an empty delta or a fake handoff version record. A direct registry edit without a v3 declaration fails closed.

## 6. Stale-base behavior

The trusted normalizer validates two distinct facts:

1. **Exact-base intent:** the delta renders to its declared handoff hash on its own base and its registry declaration matches the source patch on that base.
2. **Current-state application:** applying the same operation to current main is conflict-free, preserves history, and yields a deterministic current handoff.

Independent appends with different block IDs must commute. A repeated block ID, incompatible heading rename, stale path after rename, undeclared registry mutation, non-ancestor base, direct handoff edit, or control-plane/content mixed package fails closed.

Registry remains directly maintained in this lifecycle version. Git-clean registry merges are semantically checked; Git textual conflicts are rejected rather than guessed.

## 7. Trusted updater integration

In delta mode, `drpo-update` performs:

```text
source bundle/patch verification
→ isolated source integration commit
→ trusted handoff normalization (materialization or verified no-op)
→ Stage 4A minimal-view refresh when authority content changed
→ MATERIALIZATION_REPORT.json generation for a real delta
→ amend integration commit
→ verify normalized state
→ select and run tests on the normalized commit
→ repository gates
→ ff-only main
```

Any normalization or test failure leaves main unchanged. The acceptance suite includes a real bundle-backed stale-base v3 package through the actual `drpo-update` entry point. A child-process marker prevents that self-test from recursively invoking itself when selected by the updater's own impact map.

## 8. Stage 4 lifecycle model

Stage 4A has two different kinds of state and they must not be conflated:

**Permanently frozen acceptance inputs and contracts**

- builder and validator implementation under their governance authorization;
- schema and module taxonomy;
- dependency definitions;
- semantic contracts;
- the historical Stage 4A acceptance report and after-image as evidence of that acceptance event.

**Dynamically validated current-source outputs**

- `docs/handoff_shadow/stage4/minimal/generated/**`.

The historical after-image retains the original generated hashes as immutable audit evidence. Current governance validation does not require future generated content to equal those historical bytes. Instead, it runs the Stage 4A builder in check mode and requires the tracked outputs to be the deterministic result of the current handoff and registry. Stage 4B remains frozen cutover-audit evidence and is not refreshed on ordinary updates.

## 9. Checkpoint contract

The cutover command creates checkpoint manifest schema v2. Verification requires all of the following:

- checkpoint handoff bytes exactly equal the activation source parent's handoff;
- registry provenance hash exactly equals the activation source parent's registry;
- Stage 3 Full Acceptance report exists, is a real `PASS` full-tier report, and its content/hash matches the source parent;
- that selected Stage 3 report covers every real handoff-delta observation present at the activation source parent, leaving zero uncovered observations at cutover;
- Stage 4B acceptance report and historical after-image are valid and source-parent identical;
- a newly generated cutover audit binds those reports to the checkpoint handoff and registry hashes;
- the separately committed cutover authorization is valid, source-parent identical, and immutable;
- the activation commit changes neither handoff nor registry and contains no first production v3 delta;
- checkpoint, audit, and authority assets remain immutable after activation;
- repository-containment checks compare canonicalized paths so platform aliases such as macOS `/var` and `/private/var` do not reject a valid in-repository checkpoint, while actual symlink escapes remain forbidden.

The first production v3 delta must therefore be a later, independently auditable commit. Requiring a source-parent-complete Stage 3 Full Acceptance report also prevents the first valid v3 update from inheriting a nearly exhausted pre-cutover acceptance interval.

## 10. Cutover and rollback commands

Preparation commands write a complete transaction into the worktree but never commit it automatically:

```bash
python3 scripts/handoff_authority.py cutover \
  --repo-root . \
  --checkpoint-id <registered-checkpoint-id> \
  --authorization-record docs/governance_stage_authorizations/<cutover-authorization>.yaml

git diff --check
git add docs/handoff_versions docs/governance_pipeline_stage_status.yaml
git commit -m "Activate delta handoff authority"
python3 scripts/handoff_authority.py verify --repo-root .
```

The cutover authorization must already be committed and must be a separate Stage 5 `stage_transition` authorization. The current hardening authorization explicitly excludes executing cutover and cannot satisfy this gate.

Rollback preserves the currently materialized handoff and registry bytes, records the accepted authoritative update IDs and previous checkpoint provenance, and returns authority to manual mode:

```bash
python3 scripts/handoff_authority.py rollback \
  --repo-root . \
  --rollback-id <registered-rollback-id> \
  --reason "<reason>"

git diff --check
git add docs/handoff_versions docs/governance_pipeline_stage_status.yaml
git commit -m "Rollback handoff authority to manual"
python3 scripts/handoff_authority.py verify --repo-root .
```

Rollback does not destructively delete checkpoints, deltas, reports, or materialized research history.

## 11. Acceptance boundary

The two previously confirmed blockers are closed by:

- verified no-op normalization for packages with no handoff/registry authority change;
- current-source deterministic validation for dynamic Stage 4A generated outputs instead of permanent equality to historical generated hashes.

The hardened candidate also contains lifecycle preparation, strong checkpoint provenance checks, cutover/first-delta separation, rollback simulation, and real updater-path coverage. Formal authority activation remains forbidden until an independent pre-cutover acceptance is recorded and a separate user-approved cutover authorization is committed.
