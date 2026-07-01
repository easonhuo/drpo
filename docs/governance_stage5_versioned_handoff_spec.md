# Stage 5 — Versioned Handoff Write-Authority Candidate

**Governance claim:** `GOV-HANDOFF-AUTHORITY-CUTOVER-01`
**Implementation state:** candidate only; no authority cutover
**Design base:** `4ad8b09ca80bc4b98aebffc6540f9be29440ba28`

## 1. Purpose

Stage 5 promotes the validated Stage 3 append-oriented delta engine into a future handoff write authority without changing the current read contract. `docs/handoff.md` remains the unique research master read by users and tools. During candidate implementation it also remains the human-written authority.

After a separately authorized cutover, the only human-written source for handoff changes will be an immutable checkpoint plus immutable schema-v3 `HANDOFF_DELTA.yaml` files. A trusted normalizer will materialize `docs/handoff.md` and refresh the Stage 4A minimal generated views.

## 2. Candidate boundary

This implementation adds and tests the future production path but keeps:

```yaml
mode: manual
human_handoff_write_authority: true
direct_handoff_edit_forbidden: false
authority_cutover_allowed: false
```

It does not:

- activate schema-v3 production updates;
- create the final cutover checkpoint;
- event-source `experiments/registry.yaml`;
- make Stage 4B a runtime dependency;
- compact `AGENTS.md`;
- modify any scientific experiment state, seed, threshold, or result.

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

The normalizer uses the implementation and policy from the pre-integration current main checkout. Ordinary content packages may not modify the authority config, checkpoint, accepted v3 delta, accepted materialization report, normalizer, delta policy, updater normalization hook, Stage 4A taxonomy, stage ledger, authorization records, or `AGENTS.md`.

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

## 5. Stale-base behavior

The trusted normalizer validates two distinct facts:

1. **Exact-base intent:** the delta renders to its declared handoff hash on its own base and its registry declaration matches the source patch on that base.
2. **Current-state application:** applying the same operation to current main is conflict-free, preserves history, and yields a deterministic current handoff.

Independent appends with different block IDs must commute. A repeated block ID, incompatible heading rename, stale path after rename, undeclared registry mutation, non-ancestor base, direct handoff edit, or control-plane/content mixed package fails closed.

Registry remains directly maintained in v1. Git-clean registry merges are semantically checked; Git textual conflicts are rejected rather than guessed.

## 6. Trusted updater integration

In future delta mode, `drpo-update` performs:

```text
source bundle/patch verification
→ isolated source integration commit
→ trusted handoff normalization
→ Stage 4A minimal-view refresh
→ MATERIALIZATION_REPORT.json generation
→ amend integration commit
→ verify normalized state
→ select and run tests on normalized commit
→ ff-only main
```

Any normalization failure leaves main unchanged.

## 7. Stage 4 roles

- Stage 4A taxonomy, dependencies, builders, validators, and semantic contracts remain protected.
- Stage 4A minimal generated outputs become dynamic, deterministic views of the current handoff and registry.
- Stage 4B remains frozen cutover-audit evidence and is not refreshed on ordinary updates.

## 8. Cutover and rollback gates

A later cutover requires a new explicit user authorization and all of:

- current Stage 3 Full Acceptance;
- exact checkpoint reconstruction;
- Stage 4B audit bound to checkpoint bytes;
- real stale-base A→B and B→A updater replay;
- conflict and tamper rejection;
- failure atomicity;
- full repository tests;
- rollback simulation preserving identical handoff bytes.

Until then, `docs/handoff.md` remains the manual authority.
