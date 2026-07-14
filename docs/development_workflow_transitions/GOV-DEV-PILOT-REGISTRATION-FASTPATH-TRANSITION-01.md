# GOV-DEV-PILOT-REGISTRATION-FASTPATH-TRANSITION-01

**Status:** proposed temporary adoption rule  
**Base commit:** `f387391b39502e700e7f780cd8b4a1fd9c7eca7c`  
**Parent incident ledger:** `docs/development_workflow_incident_and_improvement_log.md`  
**Implementation plan under review:** `docs/dev_pilot_registration_fastpath.md` in PR #62  
**Implementation PR:** https://github.com/easonhuo/drpo/pull/62  
**Scientific experiment impact:** none

## 1. Purpose

PR #62 implements a deterministic preparation adapter for code-first pilot registration, but the adapter is not yet present on `main`. New sessions read `main`, so without an explicit transition rule they may continue the older ad hoc registration path and recreate the same micro-commit, stale-base, remote-generation, and repeated-CI problems that motivated the fastpath.

This document defines a temporary adoption boundary while PR #62 completes real V1 shadow validation and review.

## 2. Applicability

The rule applies when a new or continuing development task would require **new or modified** authoritative registration for an E7, E8, or other scientific pilot, including any update that would change:

- `experiments/registry.yaml`;
- `docs/handoff_deltas/**`;
- a materialized `docs/handoff.md` after-image;
- Stage 4A generated views;
- registration/result provenance bound to a newly frozen implementation commit.

The rule does not alter experiment priority, scientific variables, seeds, thresholds, stopping criteria, result status, or evidence classification.

## 3. Temporary default

Until the fastpath is activated on `main`, a new code-first pilot may proceed through:

```text
scientific design and scope freeze
→ implementation
→ command-contract validation
→ real-data liveness when already authorized
→ implementation-SHA freeze
```

After the implementation SHA is frozen, the session must **not** construct a new authoritative registration through an ad hoc manual sequence of per-file writes, temporary workflows, repeated remote generation, or unreviewed branch reconstruction.

Registration closure must use one of these routes:

1. a reviewed PR #62 fastpath shadow explicitly pinned to an exact tool commit and labelled as shadow/non-default; or
2. wait until the fastpath is merged and activated on `main`; or
3. an explicit user-approved exception that records why the existing manual V1 path is required, the exact frozen implementation SHA, the expected logical commit structure, and the additional review/rollback plan.

Silence, convenience, schedule pressure, or the fact that an older session used the manual route are not exceptions.

## 4. Work that is not blocked

This transition does not block:

- execution of an experiment that is already authoritatively registered and otherwise allowed by the handoff;
- code-only bug fixes that do not change scientific semantics or registration;
- command-contract tests, unit tests, smoke tests, or real-data liveness already permitted by the current protocol;
- implementation and review work up to the frozen implementation commit;
- result analysis that does not claim an unregistered status transition.

An already registered experiment remains governed by its existing handoff/registry protocol. This transition must not be used to postpone required terminal audit, packaging, or result deposition.

## 5. Prohibited interpretations

This rule does not:

- make PR #62 production authority before merge;
- permit a dev-branch tool to be treated as the repository default;
- relax document-before-experiment requirements;
- permit an unregistered formal experiment launch;
- authorize direct editing of `docs/handoff.md`;
- bypass schema-v3 authority, reviewer approval, exact-head gates, or explicit merge approval;
- convert a fastpath shadow, smoke test, or preparation report into scientific evidence.

## 6. Session startup behavior

A session that may touch E7/E8 scientific code or registration must state:

- whether the target experiment is already authoritatively registered;
- whether the task changes registration or only implementation;
- the exact implementation/base SHA currently frozen;
- whether the transition rule applies;
- which of the three permitted registration-closure routes will be used.

If the session cannot establish these facts, it must stop before registration mutation.

## 7. Review and rollback

This is a documentation-only adoption guard. It adds no authority engine, workflow, validator, or scientific state.

Rollback is immediate:

1. remove the temporary startup instruction from `AGENTS.md` through a reviewed PR;
2. preserve this document as historical process evidence;
3. continue using the accepted V1 manual inputs and gates.

Rollback does not alter any experiment or previously completed registration.

## 8. Expiration and succession

The temporary rule remains active until all of the following are true:

1. PR #62 has been rebuilt on the then-current `main`;
2. a real code-plus-registration V1 shadow reaches local `READY` without scientific-state publication;
3. PR #62 passes exact-head review and is merged with explicit user approval;
4. a separate activation change on `main` defines the permanent fastpath default and fallback semantics.

After activation, this document is marked `superseded` but is not deleted. The temporary `AGENTS.md` section is replaced by the permanent activation rule rather than silently disappearing.

## 9. Current uncertainties

- the real V1 registration shadow may expose additional adapter/V1 interface defects;
- PR #62 is currently stale relative to `main` and its previous CI result is not a final merge gate;
- operational CI tiering and enforcement remain PR-B work and are not activated by this transition.
