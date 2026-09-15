# GOV-RESULT-CLOSURE-MATERIALIZATION-GATE-01

## Problem

A scientific result can be fully reviewed and a schema-v3 `HANDOFF_DELTA.yaml` can be prepared while the corresponding pull request never completes trusted materialization and merge. In that half-closed state, the result exists on a dev branch but the unique research master on authoritative `main`, `docs/handoff.md`, still lacks the closure. A later session that correctly follows the startup protocol can therefore miss the result.

The motivating historical incident is PR #342 (`EXT-C-E8-MULTITASK-EXP-LAMBDA-COMPLETION-01` + `EXT-C-E8-MULTITASK-EXP-LAMBDA-CURVE-COMPLETION-02`): the result closure and authoritative schema-v3 handoff delta were prepared, but the PR remained unmerged and the handoff block never reached `main`.

## Approval

- Claim: `GOV-RESULT-CLOSURE-MATERIALIZATION-GATE-01`
- Approval record: `user_approved_2026_09_14_add_result_closure_handoff_materialization_hard_gate`
- Base commit: `1830e672395ed4b881b147ad3c3a42f1b36074bf`
- Change class: maintenance bugfix / fail-closed completion check
- Scientific experiment impact: none
- Scientific variable impact: none

The repository owner explicitly approved adding a hard gate after the missing-materialization failure mode was identified.

## Hard-gate contract

For a pull request that newly adds an authoritative schema-v3 handoff delta under `docs/handoff_deltas/<update_id>/HANDOFF_DELTA.yaml`:

1. the sibling `MATERIALIZATION_REPORT.json` must exist in the PR head;
2. if the delta contains handoff operations, `docs/handoff.md` must differ from the PR base;
3. every `insert_after_heading` / `append_to_section` operation must have its canonical generated block markers in the head handoff;
4. every `replace_heading` operation must have its declared new heading in the head handoff;
5. otherwise the blocking pull-request transition check fails closed with `HANDOFF_MATERIALIZATION_MISSING`.

A code-only change that adds no authoritative schema-v3 delta is unaffected. A registry-only authoritative delta still requires its sibling materialization report but does not require a handoff-byte change when it has no handoff operations.

The scientific meaning of a result is unchanged by this gate. The gate only prevents a delta-only or partially materialized integration from being accepted as a completed authoritative closure.

## Result-closure status rule

A result closure is not authoritative merely because its scientific workload finished, its result document exists, its handoff delta exists, a PR is open, or CI on an unmaterialized branch passes. For project handoff purposes it remains pending integration until the trusted materialization is present in the integration commit and that commit is merged to authoritative `main`.

This rule does not redefine experiment execution completion. Scientific workload completion, artifact delivery, terminal audit, and repository result closure remain distinct states.

## Implementation scope

Authorized changes are limited to:

- extend `scripts/validate_evidence_locator.py` with the transition-aware handoff-materialization completeness check;
- extend `tests/test_evidence_locator.py` with regression coverage for the missing-materialization and valid-materialization cases;
- clarify the closure procedure in `docs/postrun_evidence_locator_contract.md`;
- this scope record.

The existing blocking `.github/workflows/evidence-locator-gate.yml` already invokes the validator with exact PR base/head SHAs, so no new workflow or new Python file is required.

## Explicitly unchanged

- no scientific experiment is launched;
- no experiment result, scientific status, claim, method ranking, seed, threshold, data size, training horizon, or stopping criterion changes;
- no direct edit to `docs/handoff.md` or `experiments/registry.yaml` is made by this implementation change;
- no modification is made to Stage-5 protected authority files such as `scripts/handoff_authority.py`;
- no new handoff authority, result authority, database, service, scheduler, or merge automation is introduced;
- no historical result or failed integration record is deleted or rewritten;
- no existing unmerged PR is automatically merged or closed.

## Acceptance

The change is acceptable only when:

1. a synthetic PR transition that adds an authoritative schema-v3 delta but omits the sibling materialization report fails with `HANDOFF_MATERIALIZATION_MISSING`;
2. a delta with a report but unchanged handoff fails when handoff operations are present;
3. a correctly materialized delta with the expected canonical block markers passes;
4. transitions that add no authoritative schema-v3 delta retain the existing evidence-locator behavior;
5. the existing Evidence Locator Gate, repository tests, Ruff, handoff-authority verification, and governance-stage validation pass on the exact PR head.

## Rollback

Revert the validator, focused tests, and documentation clarification together. Preserve all historical handoff deltas, materialization reports, result evidence, and scientific records. Stage-5 authority and its protected implementation remain unchanged.
