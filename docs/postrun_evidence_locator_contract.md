# Post-run evidence locator contract

**Policy ID:** `GOV-POSTRUN-EVIDENCE-LOCATOR-01`  
**Schema:** `evidence_locator.schema_version: 1`

## Purpose

`docs/handoff.md` and `experiments/registry.yaml` intentionally keep scientific conclusions compact. Detailed text-first evidence lives in the append-only private repository `easonhuo/drpo-results`. A completed registration therefore needs an immutable machine-readable pointer from the registry entry to the exact delivered evidence.

The handoff remains the research summary. It should name the experiment ID and primary run ID; the registry entry is the canonical locator index.

## Required registry block

A changed or newly added experiment is treated as delivered when any of these markers is present:

- `execution.state: delivered`;
- `formal_run_status: delivered`;
- `evidence.delivered_to_user: true`;
- legacy result-repository locator fields appear under `evidence`.

Such an experiment must include:

```yaml
# Copy and replace docs/templates/EVIDENCE_LOCATOR.yaml.
evidence_locator:
  schema_version: 1
  primary_run_id: E8_EXAMPLE_20260716_01
  records:
    - run_id: E8_EXAMPLE_20260716_01
      lane: e8
      source_commit: <40-character DRPO source commit>
      results_repository: easonhuo/drpo-results
      results_branch: ingest/e8
      results_commit: <40-character results-repository commit>
      result_path: runs/e8/E8_EXAMPLE_20260716_01
      manifest_sha256: <64-character RESULT_MANIFEST SHA-256>
      export_profile: manifest_text_v1
```

The `primary_run_id` must name the last record. A later rerun appends a new record and may make it primary. Existing records may not be edited, reordered, or removed.

## Closure procedure

1. Run the READY RunSpec through the scoped lane wrapper.
2. Require `delivery_status=PASS` or an idempotent `ALREADY_DELIVERED` retry.
3. Lock the returned `run_id`, source commit, results commit, result path, and manifest SHA-256.
4. Audit the delivered files and terminal state before assigning a scientific status.
5. Add the exact `evidence_locator` block to the reviewer-authored registry after-image.
6. In the handoff delta, summarize the result and name the experiment ID plus `primary_run_id`; do not duplicate raw tables in the handoff.
7. Trusted normalization must materialize the handoff delta into `docs/handoff.md` and create the sibling `MATERIALIZATION_REPORT.json` in the same integration image.
8. The PR gate compares the base and head registries and rejects a changed delivered experiment without a valid locator. It also rejects a newly added authoritative schema-v3 handoff delta whose materialization report or declared handoff after-image is missing.
9. For project handoff purposes, a result closure remains pending integration until that materialized integration commit is merged to authoritative `main`; a result document, delta-only branch, open PR, local READY state, or CI pass on an unmerged branch is not itself authoritative closure.

### Result-closure materialization hard gate

`GOV-RESULT-CLOSURE-MATERIALIZATION-GATE-01` closes the historical half-closure failure mode in which a valid result closure and `HANDOFF_DELTA.yaml` existed on a branch but never reached the unique research master.

For each authoritative schema-v3 delta newly added by a pull request, the blocking transition validator requires:

- sibling `MATERIALIZATION_REPORT.json` in the PR head;
- a changed `docs/handoff.md` when the delta contains handoff operations;
- the declared replacement heading for every `replace_heading` operation;
- canonical `HANDOFF-DELTA-BLOCK` start/end markers for every `insert_after_heading` or `append_to_section` operation.

Failure is reported as `HANDOFF_MATERIALIZATION_MISSING`. Code-only changes that add no authoritative schema-v3 delta are unaffected. Registry-only authoritative deltas still require the materialization report but do not require a handoff-byte change when they contain no handoff operations.

This gate does not change scientific workload-completion semantics, experiment status, terminal-audit semantics, or Stage-5 write authority. It only makes repository result-closure integration fail closed when the authoritative handoff after-image is absent.

## Historical compatibility

Existing untouched historical entries are grandfathered and reported by current-tree validation; they are not silently rewritten. When a delivered historical experiment is next changed, it must receive a valid locator. No historical conclusion, package record, or provenance field may be deleted during backfill.

The first version covers canonical RunSpec delivery to `easonhuo/drpo-results`. Legacy ZIP-only evidence and persistent-local sidecars remain explicit historical provenance but do not masquerade as a results-repository locator.

The materialization hard gate is transition-aware and therefore does not destructively rewrite or retroactively reject already-landed historical handoff state. It applies when a pull request newly adds an authoritative schema-v3 delta.

## Commands

Current-tree audit, including grandfathered entries:

```bash
python scripts/validate_evidence_locator.py --repo-root . --json
```

Pull-request transition gate, including result-closure materialization completeness:

```bash
python scripts/validate_evidence_locator.py \
  --repo-root . \
  --base <base-sha> \
  --head <head-sha> \
  --json
```

## Rollback

Remove the materialization check from the transition validator if this gate is explicitly rolled back. Do not remove any locator, handoff delta, materialization report, or result evidence already registered: those records are provenance and remain append-only. The Stage-5 authority remains unchanged.
