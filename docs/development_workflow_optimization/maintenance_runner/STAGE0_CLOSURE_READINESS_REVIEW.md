# M0 Stage 0 closure-readiness review

**Claim:** `GOV-DRPO-MAINTENANCE-RUNNER-REPLAYAB-01`  
**Controlling plan:** `docs/development_workflow_optimization/M0_ATOMIC_DEV_TRANSACTION_PLAN.md`  
**Current plan version:** `1.3-stage0-review-C`  
**Current main verified:** `ad9bda80796dcf5c48976f5d64ffd79a006c70d5`  
**Review decision:** `HOLD_STAGE0_NOT_CLOSED`  
**Architecture decision:** `STABLE_M0_WITH_BOUNDED_MEASUREMENT_ADAPTER`  
**Scientific impact:** none

## 1. Objects reviewed

- controlling M0 plan;
- preserved superseded M2 plan;
- Review 03 redesign record;
- R-A ownership/capability review;
- R-B security/governance review;
- R-C ReplayAB-validity review;
- M0 transaction specification;
- validation-profile matrix;
- stage-packet and transaction-evidence templates;
- candidate eight-case inventory;
- evaluator bindings;
- claim scope record;
- current GitHub connector capabilities and current-main repository owners.

## 2. Architecture verdict

The implementation direction is now stable enough that Stage 1 should not rediscover a
new product architecture.

Frozen decisions:

1. M2 patch/apply/test/commit/push runner is rejected as duplicative.
2. M0 candidate has zero production code and standardizes the existing GitHub atomic
   blob/tree/commit/ref route.
3. M0 begins only after complete reviewed after-images exist.
4. V1, Candidate 01, ReplayAB Core, E7/E8 code, handoff authority, formal execution, and
   PR gates remain independent owners.
5. M1 is not authorized and may be designed only from a specific measured publisher gap.
6. Remote GitHub qualification and controlled local-git timing are separate evidence
   layers.
7. Existing ReplayAB Core is reusable, but its real-pair CLI is Candidate-01-specific.
8. The only predeclared executable work is a bounded in-place
   `scripts/run_workflow_replay.py git-object-pair` measurement command.
9. No new Python file, workflow, backend, dependency, E7/E8 adapter, or publisher is
   permitted.
10. New branches appear directly at the final audited commit; PREOBJECT checks occur
    before Git blob storage.

These decisions resolve the causes that would otherwise have forced a mid-Stage-1
redesign.

## 3. Review-pass disposition

### R-A — ownership and necessity

**Resolved in plan.** M0 no longer duplicates V1 or Candidate 01. A non-publishing
connector check demonstrated that the current GitHub App can construct a tree update
from an exact commit base without sequential Contents-API commits.

### R-B — security and governance

**Resolved in plan.** PREOBJECT review, pre-ref freshness, direct final-commit branch
creation, approval binding, local-only failure qualification, and separate adoption
governance are now explicit.

### R-C — Replay validity and stability

**Resolved architecturally, not yet implemented.** The exact existing-file adapter path,
responsibility, internal Stage 1 order, and line budgets are frozen. Adapter correctness
remains a Stage 1 acceptance item and is not being claimed now.

## 4. Why Stage 0 cannot close yet

The eight-case inventory is deliberately marked `candidate_inventory_not_frozen`.
The following immutable facts are incomplete:

- per-file after-image SHA-256 values;
- Git modes for all selected files;
- case-specific historical Arm-A operation reconstruction;
- proof of treatment contrast for each success case;
- exact new-Python approval provenance for broad historical payloads;
- final case-specific scientific-invariant facts;
- failure-fixture and diagnostic identities;
- final evaluator SHA-256 bindings;
- exact controlling-plan file SHA-256;
- a fresh branch lineage from then-current `main` for Stage 1.

Closing Stage 0 without these facts would allow post-result case replacement or evaluator
changes and would undermine ReplayAB validity.

## 5. Candidate-case review

The inventory contains four E7-derived and four E8-derived candidates with explicit
blockers. None is represented as frozen.

Particular risks retained rather than hidden:

- reconstructed config/RunSpec subsets may not preserve a real Arm-A treatment contrast;
- stacked PR payloads require exact base-object availability;
- broad integration-heavy cases may exceed the intended M0 task class;
- new-Python historical payloads require durable approval provenance;
- failure cases require fixed non-executable local fixtures.

A case may be replaced before freeze only for missing immutable artifacts, absent
contrast, scope exclusion, or inability to evaluate independently. The reason and old
candidate remain preserved in Git history.

## 6. Validation-command review

The validation matrix binds the current repository-owned commands from PR Gate Log and
the current impact-map policy. Full exact-head pytest and Ruff remain the acceptance
gate; focused tests supplement rather than replace them.

The command matrix is sufficiently specific for Stage 0, but each final case must bind
its exact focused pytest targets and expected scientific invariants before freeze.

## 7. Governance and branch state

- Stage 0 changed documentation only.
- No workflow, runner, ReplayAB code, scientific code, handoff, registry, authority, or
  governance implementation changed.
- No experiment ran.
- No default route or merge is authorized.
- The current design branch remains based on an old merge base and is behind current
  `main`; no Stage 1 code may begin from it.
- Before Stage 1, create a clean current-main-based implementation branch and carry only
  the reviewed controlling documents and approved Stage 1 scope.

## 8. Final decision

```text
architecture: STABLE
Stage 0: HOLD
Stage 1: BLOCKED_NOT_STARTED
M2: REJECTED
M1: NOT_AUTHORIZED
```

The next permitted work remains Stage 0 evidence freeze:

1. reconstruct candidate after-images from immutable Git objects;
2. compute per-file SHA-256 and modes;
3. validate treatment contrast and replace ineligible candidates before freeze;
4. bind exact evaluators and failure fixtures;
5. record exact plan SHA-256;
6. rerun closure review against then-current `main`.

Only a later all-pass closure report plus explicit user approval can authorize Stage 1.
