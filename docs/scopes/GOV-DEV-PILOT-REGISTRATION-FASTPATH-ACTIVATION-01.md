# GOV-DEV-PILOT-REGISTRATION-FASTPATH-ACTIVATION-01 — Scope and Rollout

**Status:** authorized for Draft PR; activation pending review and merge  
**Base commit:** `cd3f293f1aa5d53696ed7a95f4d5a525bbbfe2bd`  
**Implementation claim:** `GOV-DEV-PILOT-REGISTRATION-FASTPATH-01`  
**Implementation merge:** PR #62, commit `21968670018c430d5e5129b802fbae21a474c574`  
**Authorization record:** `docs/governance_stage_authorizations/GOV-DEV-PILOT-REGISTRATION-FASTPATH-ACTIVATION-01.yaml`  
**Scientific experiment impact:** none

## 1. Purpose

Make the merged and reviewed pilot-registration fastpath the default registration-preparation route for new or modified code-first E7, E8, and other scientific pilots after their implementation SHA is frozen.

The permanent default is:

```text
scientific design and scope freeze
→ implementation
→ command-contract validation
→ already-authorized liveness
→ implementation-SHA freeze
→ reviewer-authored DEV_PILOT_REGISTRATION_SPEC.yaml
→ scripts/prepare_dev_pilot_registration.py
→ PREPARED_INPUTS
→ existing V1 plan / prepare / normalize / gate / finalize
→ local READY
→ normal reviewed GitHub publication and explicit merge approval
```

Activation changes which already-reviewed preparation route is the default. It does not create a new authority, transaction engine, publisher, or scientific execution permission.

## 2. Evidence supporting activation

PR #62 established all of the following before activation:

- the adapter is merged on `main`;
- strict parsing, provenance checks, before-image validation, output isolation, locking, atomic publication, idempotency, conflict rejection, and negative tests passed;
- a real E7 code-plus-registration shadow traversed `PREPARED_INPUTS → local READY` through the existing V1 transaction;
- exact-head compile, handoff authority, formal execution channel, governance validation, full pytest, and Ruff passed;
- the adapter performs no Git network operation, push, PR creation, merge, authority normalization, or candidate publication.

The first real E7 and E8 production registrations remain rollout observations for measured efficiency and failure data. They are not prerequisites for selecting the safer reviewed route as the default.

## 3. Applicability

The default applies when a code-first task needs a new or modified authoritative scientific-pilot registration that may affect:

- `experiments/registry.yaml`;
- `docs/handoff_deltas/**`;
- the materialized `docs/handoff.md` after-image;
- Stage 4A generated views;
- registration or result provenance tied to a newly frozen implementation commit.

Already registered experiments remain governed by their existing protocol. Code-only changes that require no registration do not invoke the registration portion of this route.

## 4. Manual V1 fallback

The existing manual V1 input path remains available as an explicit fallback. Before selecting it, the session must record:

1. why the merged fastpath is unavailable or unsuitable;
2. the exact frozen implementation SHA;
3. the intended logical commit structure;
4. the review plan;
5. the rollback plan;
6. explicit user approval for the fallback.

Convenience, schedule pressure, historical habit, or unfamiliarity with the fastpath are not sufficient reasons.

## 5. Boundaries

This activation does not:

- authorize an unregistered formal experiment launch;
- alter any E7/E8 matrix, method, dataset, seed, threshold, horizon, stopping rule, result status, or priority;
- edit `docs/handoff.md` or `experiments/registry.yaml` in this PR;
- bypass reviewer approval, schema-v3 authority, V1 gates, exact-head CI, terminal audit, artifact delivery, or explicit merge approval;
- enable automatic push, PR creation, merge, or publication;
- activate tiered CI, telemetry enforcement, or a test-impact policy;
- implement `GOV-DEV-POSTRUN-CLOSURE-DEPENDENCY-GATE-01`.

Code-first means registration is deferred until the implementation identity is stable; it does not mean registration is optional. Post-run terminal audit, durable packaging, compact result deposition, and result closure remain mandatory under their existing rules.

## 6. Transition succession

`docs/development_workflow_transitions/GOV-DEV-PILOT-REGISTRATION-FASTPATH-TRANSITION-01.md` is retained as historical evidence and marked `superseded` after this activation merges. The temporary section in `AGENTS.md` is replaced rather than silently removed.

## 7. Rollback

Rollback is low migration risk because the accepted manual V1 route remains intact:

1. restore the temporary `AGENTS.md` transition wording from base commit `cd3f293f1aa5d53696ed7a95f4d5a525bbbfe2bd` through a reviewed PR;
2. mark this activation scope reverted while preserving the file and authorization history;
3. use the existing manual V1 inputs and all existing gates;
4. preserve every completed registration and scientific artifact unchanged.

No rollback step permits direct handoff editing, authority bypass, or destructive deletion.

## 8. Acceptance criteria

The Activation PR is acceptable only when:

- its diff is limited to `AGENTS.md`, the historical transition document, this scope record, and its authorization record;
- `main` freshness is rechecked at the final head;
- handoff and registry are unchanged;
- the governance-stage validator accepts the explicit reopen authorization and rollback plan;
- handoff authority and formal execution channel checks pass;
- full pytest and Ruff pass;
- no unresolved review thread remains;
- merge occurs only after a separate explicit user approval.

## 9. Rollout observations after merge

The next E7 and E8 registrations should record:

- implementation-SHA-freeze time;
- fastpath start and `PREPARED_INPUTS` time;
- V1 local-`READY` time;
- manual files edited;
- commit count;
- CI runs and reruns;
- stale-main rebuilds;
- registration regenerations;
- any fallback reason or blocker.

These observations decide whether PR-B tiered CI, telemetry, or additional enforcement is justified. They do not retroactively weaken this activation's safety gates.
