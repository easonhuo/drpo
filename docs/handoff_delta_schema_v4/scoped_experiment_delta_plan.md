# DRPO schema-v4 scoped experiment delta plan

## Purpose

This document defines the narrow first version of schema-v4 scoped deltas for
DRPO handoff/registry updates. The goal is not to redesign handoff authority.
The goal is only to remove one high-frequency false conflict class:

```text
Two update packages modify different experiment IDs, but the later package is
rejected because the global handoff/registry hash changed.
```

schema-v4 scoped experiment deltas are an iteration on the existing delta
system. schema-v3 remains the default exact-base safety mode.

## Current schema-v3 behavior

schema-v3 binds a delta to a global authority base:

```text
base.commit
base.handoff_sha256
base.registry_sha256
registry.exact_base_after_sha256
expected.exact_base_candidate_sha256
```

This is safe and diagnostic, but it is not a local merge mechanism. If another
unrelated handoff/registry update lands first, the global hashes can change and
the older delta can fail even when it touches a different experiment.

## Narrow schema-v4 objective

The first schema-v4 version supports only:

```text
schema_version: 4
delta_kind: scoped_experiment_update
scope.type: experiment
operation.type: register_or_update_experiment
```

It is only for experiment-level registry/handoff updates. Two deltas that touch
different experiment IDs should commute. Two deltas that touch the same
experiment ID must fail closed.

## Non-goals

The first version explicitly does not support:

- arbitrary handoff text merge;
- registry schema migration;
- control-plane file edits;
- normalizer semantic rewrites outside the scoped delta path;
- generated view patching as authority input;
- multi-scope deltas;
- automatic merge of two edits to the same experiment ID;
- fixing ordinary Git patch conflicts;
- fixing experiment/runtime bugs.

## Safety principles

1. Different experiment scopes may merge.
2. The same experiment scope must fail closed.
3. Generated views are derived artifacts and must be regenerated.
4. Missing or ambiguous scope metadata must fail closed.
5. False rejection is acceptable in v1; false merge is not.
6. schema-v3 behavior must remain unchanged.
7. schema-v4 must be explicitly enabled until it has enough history.

## Scope hash

The scoped preimage hash is computed from the current repository authority state
for exactly one experiment ID:

```text
sha256(canonical_json({
  "schema": "drpo-scoped-experiment-v1",
  "experiment_id": "...",
  "registry_entry": <registry entry or "<MISSING>">,
  "handoff_entry": <handoff authority material or "<MISSING>">
}))
```

The first analyzer implementation is intentionally conservative. It extracts:

1. the matching `experiments/registry.yaml` entry by `id`;
2. handoff lines that mention the experiment ID, with local context windows.

The analyzer is not a merge engine. If the handoff scope cannot be extracted
with high confidence, later merge-enabling code must fail closed rather than
using a loose match.

## Proposed schema-v4 shape

```yaml
schema_version: 4
delta_kind: scoped_experiment_update
operation:
  type: register_or_update_experiment
  experiment_id: EXT-C-E8-ONPOLICY-UNPOLISHED-0.5B-01
scope:
  type: experiment
  id: EXT-C-E8-ONPOLICY-UNPOLISHED-0.5B-01
preimage:
  scope_exists: false
  experiment_scope_sha256: "..."
payload:
  registry_entry:
    id: EXT-C-E8-ONPOLICY-UNPOLISHED-0.5B-01
    ...
  handoff_entry:
    ...
```

The first version allows exactly one experiment ID per delta.

## Merge rules

### Allowed

A schema-v4 scoped delta may be applied on top of the current main if:

- it declares exactly one experiment ID;
- the current scope hash for that experiment ID equals the delta preimage hash;
- the payload references no other experiment ID;
- no control-plane paths are modified by the package;
- generated views are regenerated, not trusted from the package.

### Rejected

A schema-v4 scoped delta must be rejected if:

- the current scope hash differs from the preimage hash;
- the delta references multiple experiment IDs;
- the scope hash is missing;
- the experiment scope cannot be extracted;
- the registry schema is changed;
- a control-plane path is changed;
- arbitrary handoff text is changed outside the experiment scope;
- a generated view patch is presented as authority.

## Development rounds

### Round 0: documentation

Add this plan to the repository. No behavior changes.

### Round 1: read-only scope analyzer

Add a read-only analyzer that can:

- parse schema-v3 and proposed schema-v4 deltas;
- infer or read the affected experiment ID;
- compute current registry/handoff scope material;
- produce a deterministic scope hash;
- compare it to a schema-v4 preimage hash when present;
- fail closed on ambiguous or malformed deltas;
- never modify the repository.

### Round 2: standalone schema-v4 generator, disabled by default

Add a standalone generator under `docs/handoff_delta_schema_v4/` that writes
schema-v4 scoped experiment delta candidates and immediately validates them with
the Round-1 analyzer. This deliberately does not modify `scripts/package_update.py`,
`drpo-update`, or the normalizer.

The generator is a low-risk bridge: it proves that scoped delta files can be
created deterministically from current repository scope material and a single
experiment payload, while schema-v3 remains the only production merge path.
Formal package-generation integration is deferred until the scoped format has
shadow-mode evidence.

### Round 2 acceptance checks

The standalone generator must:

- require exactly one `experiment_id`;
- require a payload mapping;
- reject payloads that mention another experiment ID;
- compute `preimage.experiment_scope_sha256` from the current repository;
- write a schema-v4 YAML delta only to the requested output path;
- re-run the Round-1 analyzer on the generated delta;
- report that schema-v4 merge remains disabled.

### Round 3: normalizer-shadow decision mode

Add a schema-v4 normalizer-shadow decision helper under
`docs/handoff_delta_schema_v4/handoff_delta_normalizer_shadow.py`. The helper
computes the conservative `would_merge` / `would_reject` decision that a future
trusted normalizer path would use, but it must not materialize a schema-v4 delta
or modify `docs/handoff.md`, `experiments/registry.yaml`, generated views,
`scripts/handoff_authority.py`, `drpo-update`, or `scripts/package_update.py`.

Round 3 deliberately remains outside the production normalizer. The purpose is
to gather shadow-mode evidence and align decision semantics with the Round-1
analyzer before any control-plane integration.

### Round 4: guarded enablement

Allow actual scoped merge only for `scoped_experiment_update`, guarded by an
explicit feature flag.

### Round 5: preflight integration

Extend update-package preflight reports so a scoped conflict is rejected before a
runnable package is emitted.

## Rollback

Rollback is simple because schema-v3 remains default:

```text
1. Disable schema-v4 feature flag.
2. Continue generating schema-v3 deltas.
3. Keep the analyzer as a read-only diagnostic tool or remove it in a follow-up.
4. Already-normalized handoff/registry files remain ordinary repository state.
```

## Success criteria for the first full schema-v4 version

- Different experiment IDs can be merged in either order.
- The same experiment ID is rejected on the second merge.
- schema-v3 packages behave exactly as before.
- generated views are regenerated rather than trusted.
- failure reports name the conflicting experiment ID.

## Round 2 tool usage

Round 2 adds `docs/handoff_delta_schema_v4/handoff_delta_scope_generator.py`. Example usage:

```bash
python docs/handoff_delta_schema_v4/handoff_delta_scope_generator.py \
  --repo . \
  --experiment-id EXT-C-E8-EXAMPLE-01 \
  --payload /path/to/payload.yaml \
  --output /tmp/HANDOFF_DELTA.yaml \
  --update-id EXT-C-E8-EXAMPLE-01-2026-07-07 \
  --json
```

The payload must be a YAML mapping. It may contain `registry_entry` and
`handoff_entry` keys, but the generator treats their structure as candidate
payload data only. It does not apply the payload to `experiments/registry.yaml`
or `docs/handoff.md`.

Rollback remains trivial: remove the generator and continue using schema-v3.

## Round 3 tool usage

Round 3 adds `docs/handoff_delta_schema_v4/handoff_delta_normalizer_shadow.py`. Example usage:

```bash
python docs/handoff_delta_schema_v4/handoff_delta_normalizer_shadow.py   --repo .   --delta /tmp/HANDOFF_DELTA.yaml   --json
```

The result reports `decision: would_merge` only when all conservative checks pass:

- `schema_version: 4`;
- `delta_kind: scoped_experiment_update`;
- `operation.type: register_or_update_experiment`;
- exactly one experiment scope;
- `scope.id` exactly matches `operation.experiment_id`;
- `preimage.experiment_scope_sha256` matches the current repository scope;
- the Round-1 analyzer also returns `PASS`.

All malformed, multi-scope, stale-preimage, schema-v3, or unsupported deltas are
reported as `would_reject`. The helper is read-only and always reports
`schema_v4_merge_enabled: false`. Rollback remains trivial: remove the Round-3
helper and continue using schema-v3.
