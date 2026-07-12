# DRPO Dev Branch → Main Integration Pipeline V1 Closure

**Claim:** `GOV-DEV-BRANCH-INTEGRATION-01`  
**Status:** `accepted_operational`  
**Implementation closure base:** `ff7484b914a58b7a86ab466607393acf5091819c`  
**Acceptance date:** `2026-07-12`  
**Scientific experiment impact:** none

## 1. Closure decision

The lightweight V1 transaction framework is accepted for operational use within its documented boundary.

V1 converts one reviewer-approved, immutable dev snapshot into a current-main-based local `READY` commit through:

```text
source/ref lock
→ exact scope and provenance audit
→ isolated selective import
→ optional reviewer-bound registry/delta inputs
→ trusted handoff normalization
→ required risk-selected gates
→ final freshness and authority audit
→ local READY
```

The implementation remains deliberately local. It does not push, create or refresh pull requests, poll CI, merge, delete branches, classify scientific evidence autonomously, or replace reviewer judgment.

## 2. Accepted implementation lineage

- Batch 1 planner and audit: `a61e5b11af1e09ea880f1738991cb9e1b0e3ca1a`
- Batch 2A source-commit path: `51fdfe194647cf58543701591bf9a0e7c2278330`
- Batch 2B normalization/gates/READY: `83ae2545406cc17cdfaa9fa5f240f0dddd7e2d04`
- Batch 3 real shadows and rollback rehearsal: `ff7484b914a58b7a86ab466607393acf5091819c`

Historical development commits and PR discussions remain evidence, but the squash commits above are the authoritative merged lineage.

## 3. Batch 3 acceptance evidence

Batch 3 used immutable real repository refs rather than synthetic-only fixtures.

### Code-only shadow

- locked main: `17a7975c4fd0b0fb7058fd44bd6e725c6c1559ae`
- locked dev/result: `83ae2545406cc17cdfaa9fa5f240f0dddd7e2d04`
- source scope: the nine already-reviewed Batch 2B files
- result: local `READY`
- trusted normalization: verified code-only no-op outside the approved source scope

### Code + registry + delta + pilot-summary shadow

- locked main: `ead84d39c7df8c77de82e17d6fde27028582ff15`
- locked dev/result: `17a7975c4fd0b0fb7058fd44bd6e725c6c1559ae`
- source scope: seven already-reviewed E7 pilot/provenance files
- local-only registration target: existing `EXT-H-E7-BENCH-01`
- mutation: one `batch3_shadow_observation` field
- result: local `READY`
- preservation checks: every unrelated experiment and every pre-existing target field remained unchanged
- scientific status: unchanged; no ranking, convergence, collapse, support-boundary, or NaN/Inf claim
- publication: none

### Rollback rehearsal

The rehearsal used a valid reviewed target replacement and handoff operation together with an intentionally unsupported registry-change declaration. Trusted authority rejected the candidate after authored mutation began.

Required recovery checks passed:

- the attempt became `BLOCKED`, not `READY`;
- `DIAGNOSTIC.json` was preserved;
- the isolated repository returned to the immutable Batch 2A source commit;
- the restored worktree was clean;
- no successful normalization report was created;
- all real shadow refs remained unchanged.

## 4. Machine and CI evidence

The exact accepted Batch 3 PR head was:

`3d70983d593d02eeb4c203c13a5e4a04454eee3d`

GitHub Actions run `29178509613`, job `86611979699`, passed:

- tiered test planning;
- Python compilation;
- shell syntax;
- handoff authority;
- formal execution channel;
- governance inventory;
- governance stage status;
- full pytest;
- Ruff.

The branch-scoped acceptance harness emitted a machine-readable `BATCH3_SHADOW_SUMMARY_JSON` record containing stage and gate durations. Its asserted aggregate outcome was:

```yaml
valid_shadow_count: 2
ready_shadow_count: 2
expected_fault_interception_count: 1
false_positive_count: 0
unexpected_blocker_count: 0
scientific_state_upgrades: 0
published_candidates: 0
remote_refs_unchanged: true
rollback_restored: true
```

The two-shadow sample is sufficient for V1 acceptance but not sufficient to remove gates based on empirical interception rates. Per-stage timing and gate outcomes must continue to be retained in normal transaction records during the initial operational period.

## 5. Operational adoption boundary

V1 is the default integration route for a reviewed dev snapshot when any of the following is true:

- the branch is long-lived or its original base is no longer current main;
- selective import is safer than a whole-branch merge;
- the change touches scientific code, experiment configuration, registry state, handoff deltas, generated authority views, or result provenance;
- more than one dev agent or concurrent main change can affect integration correctness;
- the reviewer requires a durable source lock, scope audit, gate report, or local ready commit.

A simple direct PR remains permitted when the change is fresh-main, isolated, low-risk, and does not need registry/handoff normalization or scientific provenance classification. Direct PRs remain subject to `AGENTS.md`, reviewer approval, exact-head CI, and the existing merge gate.

This risk boundary prevents the transaction framework from becoming a mandatory heavyweight wrapper around trivial changes while preserving it where stale-base, scope, provenance, or scientific-governance risk is material.

## 6. Gate policy

Always-retained invariants:

- exact main/dev/result SHA lock;
- immutable request and reviewer binding;
- exact operation/path/blob/mode audit;
- system-forbidden path and filesystem safety checks;
- registry entity preservation;
- trusted handoff normalization and authority verification;
- final parent/tree/changed-path audit;
- final main/dev freshness check;
- durable diagnostics and immutable-attempt semantics.

Test intensity is risk-selected through the existing selector. Fast mode may be used only when the selector permits it. Unknown, scientific, governance, or high-impact paths remain fail-closed to full mode.

No gate is removed at closure. Future gate reduction requires observed evidence that a check adds no unique blocker value, together with a documented change and rollback path.

## 7. Remaining exclusions

V1 acceptance does not authorize:

- automatic push, PR creation, PR refresh, CI polling, merge, or branch deletion;
- new or self-modifying GitHub Actions workflows;
- automatic scientific evidence upgrading;
- automatic method ranking, convergence, steady-state, or collapse conclusions;
- databases, services, Web UI, workflow-language plugins, multi-repository scheduling, or task queues;
- modification of Stage 1, Stage 2, or Stage 5 protected responsibilities.

Any such expansion requires a new claim and explicit user authorization.

## 8. Maintenance and rollback

Operational failures remain fail-closed. A failed or stale attempt is preserved; recovery uses a new attempt unless the documented crash-recovery path can verify a completed durable stage report and repair only the transaction state.

Rollback of V1 operational use requires no data migration:

1. stop invoking the transaction CLI;
2. preserve transaction records and diagnostics;
3. continue with the direct reviewed PR path;
4. revert only the defective implementation commit if necessary;
5. leave scientific history, registry history, handoff deltas, materialization reports, and closed governance Stages unchanged.

## 9. Closure statement

`GOV-DEV-BRANCH-INTEGRATION-01` V1 development and shadow acceptance are complete. The framework enters maintenance mode inside the boundary above. Future work should focus on measured gate efficiency and real integration defects, not speculative platform expansion.
