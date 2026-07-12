# GOV-DEV-BRANCH-INTEGRATION-01 — Batch 3 shadow acceptance note

**Claim:** `GOV-DEV-BRANCH-INTEGRATION-01`  
**Phase:** Batch 3 / shadow hardening  
**Authoritative V1 contract:** `docs/pipeline_handoffs/DEV_BRANCH_INTEGRATION_PIPELINE_V1_SPEC.md`  
**Implementation base:** `83ae2545406cc17cdfaa9fa5f240f0dddd7e2d04`

## Purpose

Batch 3 validates the already-merged V1 transaction implementation against real repository history and real GitHub refs. It does not introduce a new integration architecture, change the default merge route, launch a scientific experiment, or alter any registered scientific result.

The acceptance runner is invoked by the existing full-pytest PR gate only on the dedicated Batch 3 branch. No GitHub Actions workflow is added or modified.

## Real shadow sources

### A. Code-only shadow

- main ref: `refs/heads/shadow/gov-dev-integration-01-code-main`
- locked main SHA: `17a7975c4fd0b0fb7058fd44bd6e725c6c1559ae`
- dev ref: `refs/heads/shadow/gov-dev-integration-01-code-dev`
- locked dev/result SHA: `83ae2545406cc17cdfaa9fa5f240f0dddd7e2d04`
- source change: the already-reviewed and merged Batch 2B implementation
- expected changed paths: exactly the nine files changed by commit `83ae2545…`
- registration mode: code-only; trusted handoff normalization must be a no-op

### B. Code + registry + delta + pilot-summary shadow

- main ref: `refs/heads/shadow/gov-dev-integration-01-reg-main`
- locked main SHA: `ead84d39c7df8c77de82e17d6fde27028582ff15`
- dev ref: `refs/heads/shadow/gov-dev-integration-01-reg-dev`
- locked dev/result SHA: `6e39b48dd18f273a30a737b39baf80af0093410a`
- source change: the already-reviewed E7 EXP coefficient/horizon pilot implementation and its pilot protocol summary
- expected changed paths: exactly the seven files changed by commit `6e39b48d…`
- registration target: `GOV-DEV-INTEGRATION-SHADOW-PILOT-01`
- local-only registry status: `pilot`
- scientific meaning: governance pipeline shadow only; no Hopper method ranking, convergence, task-performance, support/boundary, or numerical-collapse claim
- publication: forbidden; the resulting candidate is never pushed or merged

### C. Rollback rehearsal

A second registration transaction uses the same locked real refs but supplies a reviewer-bound delta with a deliberately nonexistent handoff heading. The expected outcome is a fail-closed normalization error after authored mutation begins. Acceptance requires:

- `DIAGNOSTIC.json` is written;
- the transaction becomes `BLOCKED`, not `READY`;
- the isolated integration repository is restored to the immutable Batch 2A source commit;
- its worktree is clean;
- the real shadow main/dev refs remain unchanged;
- the failed attempt is preserved and not rewritten into a successful attempt.

## Runtime request construction

The acceptance runner derives blob SHA and Git mode fields mechanically from the two locked commit pairs only after verifying that the actual diff status and exact path set match this document. This mechanism is restricted to Batch 3 acceptance. Production integrations continue to require a normal explicit reviewed `INTEGRATION_REQUEST.yaml`.

Reviewer decisions are committed under `docs/integrations/` and hash-bound by the normal Batch 1 source lock. Registration intent and approval are written into the untracked attempt directory after Batch 2A, exactly as required by the Batch 2B protocol.

## Required outputs

The runner must emit one machine-readable summary containing:

- locked refs and SHAs;
- stage durations for plan, prepare, normalize, gate, and finalize;
- resulting transaction states and candidate SHAs;
- gate count, failures, selector mode, and gate durations;
- expected-fault interception count;
- unexpected blocker count;
- false-positive count across the two valid shadows;
- rollback restoration checks;
- confirmation that no scientific state was upgraded.

## Acceptance criteria

Batch 3 passes only when:

1. both valid real-ref shadows reach local `READY`;
2. the code-only normalizer is a verified no-op;
3. the registration candidate contains exactly the approved source paths plus registry, one schema-v3 delta, its materialization report, handoff after-image, and approved generated views;
4. committed handoff authority and all required gates pass;
5. the registration remains explicitly pilot/governance-only and creates no scientific ranking or convergence claim;
6. the injected invalid-heading transaction is blocked and restored to `PREPARED` source state;
7. real remote refs remain at their locked SHAs;
8. all attempts and diagnostics remain auditable;
9. no Stage 1, 2, or 5 protected responsibility is changed;
10. no new workflow, auto-push, auto-PR, CI polling, or auto-merge capability is introduced.

Passing Batch 3 permits a separate closure update. It does not by itself change `AGENTS.md` or make the transaction tool the default route.
