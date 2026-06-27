# HANDOFF Delta Shadow Protocol

**Governance claim:** `GOV-HANDOFF-INDEX-01`
**Current phase:** Stage 3 shadow mode. `docs/handoff.md` remains the only authoritative research master.

## 1. Purpose and authority boundary

Every update that changes `docs/handoff.md` or `experiments/registry.yaml` must also add exactly one immutable delta under:

```text
docs/handoff_deltas/<update_id>/HANDOFF_DELTA.yaml
```

The delta is replayed against its exact base commit to generate a candidate handoff. During shadow mode the candidate is compared with the hand-written handoff but never replaces it. A later authority cutover requires a separate user-approved governance transition.

## 2. Version-1 delta envelope

```yaml
schema_version: 1
update_id: GOV-EXAMPLE-001
mode: shadow
base:
  commit: <40-character SHA>
  handoff_sha256: <64 hex>
  registry_sha256: <64 hex>
renderer_version: 1
operations: []
registry:
  mode: unchanged  # or expected_after
  expected_after_sha256: null
  transitions: []
expected:
  candidate_sha256: <64 hex>
  manual_sha256: <64 hex>
```

Unknown fields and unknown operations are rejected. Version 1 is intentionally append-oriented: it supports heading replacement, insertion immediately after a heading, and append-only blocks at the end of a section. It does not support arbitrary text replacement or destructive section deletion.

## 3. Supported operations

### `replace_heading`

Renames one uniquely identified Markdown heading without changing its level. Replaying the same operation after the new heading already exists is a no-op. A missing old/new target or an ambiguous heading path is a conflict.

### `insert_after_heading`

Adds a marker-delimited block immediately after a uniquely identified heading. Blocks at the same location are ordered by `block_id`, making independent insertions deterministic. Reusing a block ID with different content is a conflict.

### `append_to_section`

Adds a marker-delimited block at the end of a uniquely identified section, before the next heading of the same or higher level. Blocks are ordered by `block_id`.

A heading is addressed by its full exact path, for example:

```yaml
heading_path:
- DRPO / SNA2C 远场负梯度动力学研究主文档 v50（Stage 3 Shadow Mode 启动版）
- 0. 研究与执行原则
- 0.4 Registry 执行状态一致性（v42 锁定）
```

## 4. Registry assertions

`registry.mode: unchanged` requires the worktree registry to remain byte-identical to the base. `expected_after` requires an exact after-image SHA-256 and a transition assertion for every changed registered field. Each assertion names an experiment, nested field path, state machine, old value, new value, and evidence paths. A transition not present in `docs/handoff_delta_state_machines.yaml` is rejected.

## 5. Deterministic safety gates

The blocking gate is local and deterministic. It does not call a network or an LLM. It verifies:

- exact base commit and base document hashes;
- schema and operation allowlists;
- unique heading paths and block IDs;
- deterministic rendering and replay idempotence;
- exact candidate/manual equality during shadow mode;
- preservation of historical experiment and governance identifiers;
- registry after-image and declared state transitions;
- the five-second Fast Gate performance target and 15-second hard limit.

An LLM may produce a non-blocking review warning during acceptance, but it is never the sole pass/fail oracle.

## 6. Test tiers and fixed triggers

- **Fast:** every relevant handoff, registry, delta, policy, renderer, or state-machine update. Target p95: five seconds; hard timeout: 15 seconds.
- **Standard:** every schema, renderer, state-machine, conflict-rule, parser/index, or operation change. Target: 60 seconds.
- **Full:** before shadow activation, before authority cutover, after a schema major upgrade, after an architecture-level change, after 20 successful relevant updates, after seven days with at least one relevant update, or after repairing a critical semantic mismatch. Target: 15 minutes.

After cutover, the manual-vs-candidate dual path is retired. Deterministic schema, provenance, idempotence, conflict, history-preservation, state-machine, replay, and mutation tests remain permanent.

## 7. Retention

A successful normal update keeps the delta, `SHADOW_REPORT.json`, hashes, and affected selectors. It does not retain a duplicate full candidate. A failed comparison, full acceptance case, or golden replay case retains the full candidate for diagnosis.
