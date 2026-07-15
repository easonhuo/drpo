# GOV-DEV-PILOT-REGISTRATION-FASTPATH-01 — Real V1 Shadow Note

**Claim:** `GOV-DEV-PILOT-REGISTRATION-FASTPATH-01`  
**Phase:** PR-A2 real code-plus-registration shadow  
**Scientific experiment impact:** none  
**Publication:** forbidden

## 1. Purpose

Validate that the PR-A preparation adapter, not a hand-authored request/intent pair, can drive the accepted V1 transaction through:

```text
Pilot Registration Spec
→ PREPARED_INPUTS
→ V1 plan
→ V1 prepare
→ V1 normalize
→ V1 gate
→ V1 finalize
→ local READY
```

This is an integration-mechanics observation only. It does not create or upgrade scientific evidence and does not publish the READY candidate.

## 2. Immutable real sources

Reuse the already accepted Batch 3 registration-shadow refs:

- main ref: `refs/heads/shadow/gov-dev-integration-01-reg-main`;
- locked main SHA: `ead84d39c7df8c77de82e17d6fde27028582ff15`;
- dev ref: `refs/heads/shadow/gov-dev-integration-01-reg-dev`;
- locked dev/result SHA: `17a7975c4fd0b0fb7058fd44bd6e725c6c1559ae`;
- target experiment: existing `EXT-H-E7-BENCH-01`;
- approved source scope: the seven E7 coefficient/horizon pilot files previously used by Batch 3.

The test must first verify that both real refs still resolve to the locked SHAs and that the exact diff/path/mode/blob set matches the frozen scope.

## 3. Fastpath-authored local mutation

The reviewer-authored Pilot Registration Spec replaces only the target entity by adding one field:

```text
fastpath_real_shadow_observation
```

The field must state:

- local unpublished governance shadow;
- scientific state unchanged;
- no method ranking;
- no convergence claim;
- task-performance collapse not assessed;
- support/boundary event not assessed;
- NaN/Inf numerical failure not assessed;
- publication forbidden.

Every pre-existing target field and every non-target registry entity must remain byte/semantic equivalent after normalization.

## 4. Adapter/V1 boundary under test

The acceptance test must use `scripts/prepare_dev_pilot_registration.py` to generate:

- repository-overlay request/review files;
- registration intent;
- registration approval binding;
- preparation manifest and report.

It must not reconstruct those four V1 inputs manually after the adapter runs.

The test may mechanically derive operation blob SHAs and modes from the locked real commits, matching the earlier Batch 3 acceptance restriction. Production use still requires an explicit reviewer-authored spec.

## 5. Required assertions

The shadow passes only when:

1. the adapter returns `PREPARED_INPUTS` with `network_used=false` and `repository_modified=false`;
2. the generated request, review, intent, and approval hashes remain exact through V1 ingestion;
3. V1 reaches `PREPARED`, `NORMALIZED`, `REQUIRED_GATES_PASSED`, and `READY`;
4. the READY commit has exactly the locked historical main as its sole parent;
5. the integration worktree is clean;
6. the target registry entity differs only by the approved observation field;
7. all non-target entities remain unchanged;
8. one schema-v3 delta and its materialization report exist;
9. handoff and generated-view changes remain within authority-approved output scope;
10. task collapse, support/boundary, numerical failure, ranking, convergence, and experiment status are not changed;
11. both real remote refs remain at their locked SHAs;
12. no candidate is pushed, no PR is opened, and no merge occurs.

## 6. Execution mechanism

The existing full-pytest PR gate invokes the branch-scoped shadow test only when:

```text
GITHUB_HEAD_REF=dev/gov-dev-pilot-registration-fastpath-01
```

No GitHub Actions workflow is added or modified. On other branches and on `main`, the historical acceptance test skips.

## 7. Failure handling

Any failure blocks PR-A completion. Preserve the failing V1 attempt and diagnostic in the CI workspace/log long enough to identify the failing phase. Do not weaken V1, authority, provenance, or gate checks to make the shadow pass.

## 8. Acceptance boundary

A passing real shadow proves compatibility of the adapter with one real registration path. It does not prove measured wall-clock savings, activate the fastpath on `main`, authorize CI tiering, or remove any existing gate.
