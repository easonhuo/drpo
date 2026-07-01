# Stage 4B — Lossless Module Source Promotion

**Policy / claim:** `GOV-HANDOFF-INDEX-01`
**Base authorized for implementation:** `cf775893b9885ba893278437556abb4d1d5dd1a8`

## Authority boundary

`docs/handoff.md` remains the unique research master and the only human-editable research source. `experiments/registry.yaml` remains the only experiment-state source. Every Stage 4B output is a generated shadow candidate; authority cutover is forbidden.

## Frozen contracts

1. Inherit the accepted Stage 4A module IDs, order, responsibilities, dependencies, semantic contracts, and registry mappings.
2. Partition every byte of `docs/handoff.md` into stable source blocks. Every block has exactly one content owner: one canonical module or immutable history.
3. Resolve overlaps with the explicit, frozen `owner_priority` in `STAGE4B_CONFIG.yaml`; preserve non-owner candidates as references. No automatic taxonomy change is allowed.
4. Current uncovered blocks use the explicit `current_fallback_owner`; post-boundary uncovered blocks belong to immutable history. Both decisions are auditable in `OWNERSHIP.yaml`.
5. Reconstruct the compatibility handoff only from owner files, in source order, and require byte identity with `docs/handoff.md`.
6. Registry data is referenced, never copied as a second editable state source.
7. Generated files are non-editable and validated by exact content/hash manifests.
8. An unchanged build must be a no-op. A local source change may regenerate only its owner plus shared manifests and compatibility output; unrelated canonical modules must be reused.
9. Stage 4B acceptance may only make Stage 4C ready for separate authorization. It may not start Stage 4C or Stage 5.

## Acceptance blockers

Acceptance fails on any unmapped byte, duplicate block ID, unresolved owner, missing history, reconstruction mismatch, stale/tampered generated file, registry ownership duplication, Stage 4A taxonomy/order drift, source-provenance loss, authority change, generated-file edit, or fault-injection case that does not fail closed.

## Rollback

Delete the Stage 4B candidate and acceptance evidence, restore the ledger to Stage 4A accepted / Stage 4B ready for authorization, and keep `docs/handoff.md` authoritative. No research content or experiment state is changed by rollback.
