# HANDOFF Delta Shadow Protocol

**Governance claim:** `GOV-HANDOFF-INDEX-01`
**Current phase:** Stage 3 shadow mode. `docs/handoff.md` remains the only authoritative research master.

## 1. Purpose and authority boundary

Every update that changes `docs/handoff.md` or `experiments/registry.yaml` must also add exactly one immutable delta under:

```text
docs/handoff_deltas/<update_id>/HANDOFF_DELTA.yaml
```

The delta is replayed against its exact base commit to generate a candidate handoff. During shadow mode the candidate is compared with the hand-written handoff but never replaces it. A later authority cutover requires a separate user-approved governance transition.

## 2. Current schema and legacy compatibility

New deltas use schema version 2. The three pre-v2 shadow deltas
remain valid schema-version-1 historical records and are replayed through an
explicit allowlist in `docs/handoff_delta_policy.yaml`; schema 1 may not be used
for a new update ID.

### 2.1 Version-2 delta envelope

```yaml
schema_version: 2
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
  changes: []
expected:
  candidate_sha256: <64 hex>
  manual_sha256: <64 hex>
```

Unknown fields and unknown operations are rejected. The renderer remains
append-oriented: it supports heading replacement, insertion immediately after
a heading, and append-only blocks at the end of a section. It does not support
arbitrary text replacement or destructive section deletion.

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

## 4. Registry assertions and complete change coverage

`registry.mode: unchanged` requires the worktree registry to remain byte-identical
to the base. Schema version 2 `expected_after` requires an exact after-image
SHA-256 and complete semantic coverage of the registry diff. Supported change
kinds are:

- `transition`: an experiment field governed by a registered state machine;
- `add_entity`: a newly registered experiment; destructive entity removal is
  forbidden;
- `update_field`: an exact non-state before/after update with a reason and
  evidence.

Every actual added entity and changed field must be declared exactly once; an
undeclared change, an assertion without an actual diff, or a removed experiment
fails closed. The three legacy schema-1 deltas retain partial transition
assertions but their exact registry after-image remains hash-bound and is marked
as legacy partial coverage in replay reports.

## 5. Deterministic safety gates and durable reports

The blocking gate is local and deterministic. It does not call a network or an LLM. It verifies:

- exact base commit and base document hashes;
- schema and operation allowlists;
- unique heading paths and block IDs;
- deterministic rendering and replay idempotence;
- exact candidate/manual equality during shadow mode;
- preservation of historical experiment and governance identifiers;
- registry after-image and declared state transitions;
- the five-second Fast Gate performance target and 15-second hard limit.

Every committed delta must have a sibling `SHADOW_REPORT.json`. `auto-check`
does not trust that report blindly: it performs a fresh deterministic replay and
compares all semantic report fields. The one-delta-per-update limit applies to
new or modified `HANDOFF_DELTA.yaml` files; a report-only backfill for a
pre-v2 historical observation is replayed against that observation's repository
after-image and does not masquerade as a second new delta. Runtime timing and the validation-worktree
commit are excluded from that equality check. New reports use
`validation_worktree_head`; the ambiguous legacy `head_commit` field is accepted
read-only and is interpreted as a validation-worktree commit, never as the final
repository commit. The actual repository commit is derived from Git history by
the observation audit.

During update authoring, the report is created with:

```bash
python3 scripts/handoff_delta_shadow.py record-report \
  --repo-root . --delta docs/handoff_deltas/<update_id>/HANDOFF_DELTA.yaml
```

An LLM may produce a non-blocking review warning during acceptance, but it is never the sole pass/fail oracle.

## 6. Test tiers and fixed triggers

- **Fast:** every relevant handoff, registry, delta, policy, renderer, or state-machine update. Target p95: five seconds; hard timeout: 15 seconds.
- **Standard:** every schema, renderer, state-machine, conflict-rule, parser/index, or operation change. Target: 60 seconds.
- **Full:** before shadow activation, before authority cutover, after a schema major upgrade, after an architecture-level change, after 20 successful relevant updates, after seven days with at least one relevant update, or after repairing a critical semantic mismatch. Target: 15 minutes.

The count and seven-day triggers are machine-evaluated from immutable delta
directories, stored reports, Git history, and the coverage list in the latest
successful Full Acceptance report. The ordinary Fast Gate scans this immutable
metadata without replaying the entire historical corpus; it fully replays only
the delta/report artifacts touched by the current update. `corpus-check` and the
Full tier remain responsible for complete historical replay. This keeps normal
submission cost proportional to the affected update rather than to the lifetime
size of the handoff history.

`acceptance-status --require-current` and normal `auto-check` fail when Full
Acceptance is overdue. A Full run must persist its report and record the exact
update IDs it covers. Version-2 Full reports are themselves validated: covered
IDs must be real observations already present in Git history, counts and the
observation fingerprint must agree, all command outcomes must pass without a
timeout, and the corpus audit must report successful stored-report replay.

```bash
python3 scripts/run_handoff_delta_acceptance.py \
  --repo-root . --tier full \
  --reason architecture_level_renderer_or_schema_change \
  --report docs/handoff_deltas/<update_id>/FULL_ACCEPTANCE_REPORT.json
```

After cutover, the manual-vs-candidate dual path is retired. Deterministic schema, provenance, idempotence, conflict, history-preservation, state-machine, replay, and mutation tests remain permanent.

## 7. Retention

A successful normal update keeps the delta, `SHADOW_REPORT.json`, hashes, and affected selectors. It does not retain a duplicate full candidate. A failed comparison, full acceptance case, or golden replay case retains the full candidate for diagnosis.

## 8. Observation and corpus audit

The Git history plus immutable delta/report pairs form the observation ledger;
there is no manually incremented counter that can drift. The audit derives the
repository commit that first introduced each delta, distinguishes bootstrap from
real observations, replays each observation against its historical repository
after-image, verifies its stored report, and computes performance p95 and Full
Acceptance trigger status.

```bash
python3 scripts/handoff_delta_shadow.py corpus-check --repo-root . --json
python3 scripts/handoff_delta_shadow.py acceptance-status --repo-root . --json
```
