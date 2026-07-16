# DRPO A/B Replay Engine — Stage 4 Code-Budget Revision

**Claim:** `GOV-DEV-WORKFLOW-OPTIMIZATION-BENCHMARK-01`  
**Authorization:** explicit user approval on 2026-07-16  
**Applies from:** Stage 4  
**Stage-3 checkpoint:** `644a36169a89522ed6fcff88889686b2df73a342`  
**Stage-4 implementation checkpoint:** `f0d7ceee103970bd2c12b0a32b7de3b457a47378`

## Non-destructive supersession

The Stage-0 budget remains historical provenance:

- preferred: 350–450 counted production lines;
- yellow review: 451–500;
- hard stop: more than 500.

At the end of Stage 3, the accepted implementation contained 447 counted nonblank,
non-comment Python production lines. Review found that the bulk of those lines implements
necessary fail-closed manifest validation, append-only execution evidence, timing separation,
and correctness equivalence. Removing them merely to preserve the original estimate would
weaken accepted behavior or compress unrelated responsibilities.

The user therefore explicitly authorized increasing the hard production-code limit to 1000
lines and proceeding to Stage 4. This authorization changes only the engineering complexity
budget. It does not authorize a broader scientific claim, a default-route change, publication,
merge, or modification of existing component cores.

## Active budget from Stage 4

Production-code counting continues to cover nonblank, non-comment Python lines under
`src/drpo/workflow_replay/**` and `scripts/run_workflow_replay.py`.

- preferred completion range: **550–800** lines;
- yellow architecture-review range: **801–1000** lines;
- hard stop: **more than 1000** lines;
- Stage 4 itself should remain a thin composition layer and should not consume the remaining
  budget merely because it is available.

The limit may not be met through minification, hidden dynamic execution, weaker validation,
or moving Python behavior into unrelated file types.

## Stage-4 measured size

The same nonblank, non-comment counting rule gives:

- accepted Stage 1–3 production code: **447** lines;
- `src/drpo/workflow_replay/orchestrate.py`: **241** lines;
- `scripts/run_workflow_replay.py`: **79** lines;
- Stage-4 production increment: **320** lines;
- cumulative Replay Engine production code: **767** lines.

The cumulative result remains inside the revised 550–800 preferred completion range.

## Constraints that did not change

The implementation must still stop or redesign if it introduces:

- a database, service, daemon, queue, scheduler, dashboard, or new third-party dependency;
- duplicate validators, authority logic, registry semantics, gates, full scans, or network work;
- changes to V1 core, fastpath core, handoff authority, registry schema, scientific code, or
  GitHub workflows;
- automatic push, PR creation, approval, merge, or default-route activation;
- candidate self-overhead above the frozen runtime thresholds;
- a materially slower in-scope replay case;
- task-specific E7/E8 behavior in the general workflow layer.

## Stage authorization

This record authorized **Stage 4 only**: implement the smallest Arm-B composition path that
invokes the existing pilot-registration preparation adapter and V1 stages in their accepted
order, while automating exact intermediate-file placement.

Stage 4 is now implemented and exact-head repository CI has passed. Stage 5 is technically
eligible for a later explicit start instruction, but has not begun. Stage 5 must cover failure
injection and fixture end-to-end replay before any historical benchmark or adoption claim.
