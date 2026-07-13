# Runtime Resource Policy V1 — implementation plan and cost estimate

**Claim:** `GOV-RUNTIME-RESOURCE-POLICY-INTEGRATION-01`  
**Status:** design candidate.  
**Estimate base:** current E7/E8 selectors are reused; they are not rewritten.

## 1. Delivery strategy

The implementation is split by risk boundary rather than delivered as one large
change. Each phase has an independent rollback and may stop without requiring the
next phase.

```text
A. design and authorization
B. portable core + standalone plan
C. E7/E8 shadow equivalence
D. formal-channel compatibility and cutover
E. optional providers/backends/throughput work
```

The first three phases do not change the formal execution default. Phase D is the
only phase that requires a Stage-2 reopen authorization.

## 2. Scope controls that keep the implementation light

The following choices are deliberate line-count and maintenance controls:

- no execution-backend abstraction in V1 core;
- no Ray, Dask, Submitit, Slurm, or Kubernetes dependency;
- no dynamic Python plugin discovery or entry-point scanning;
- adapters are supplied through an explicit in-process registry mapping;
- no Pydantic dependency;
- JSON Schema is shipped for interoperability, while a small strict standard-
  library parser handles production validation;
- one authoritative `RUNTIME_SELECTION.json`, not separate overlapping decision,
  snapshot, and selection sources of truth;
- one conditional resume revalidation artifact;
- current E7/E8 measurement and selection functions are wrapped, not copied;
- no migration of historical RunSpecs in the first implementation;
- no throughput search, elastic scaling, batch tuning, or multi-node support.

Any proposal that adds one of these excluded capabilities must update the claim and
cost estimate before implementation.

## 3. Proposed implementation files

### 3.1 Portable project-neutral core

```text
src/runtime_resource_policy/__init__.py
src/runtime_resource_policy/model.py
src/runtime_resource_policy/contract.py
src/runtime_resource_policy/identity.py
src/runtime_resource_policy/engine.py
```

Responsibilities:

- immutable models and result codes;
- strict common-contract parsing;
- canonical JSON and SHA-256 identity;
- fresh/plan/resume state machine;
- atomic selection and revalidation artifacts;
- explicit adapter registry lookup.

The core accepts machine snapshots, workload fingerprints, scientific fingerprints,
and adapters from the caller. It has no DRPO import.

### 3.2 DRPO integration

```text
src/drpo/runtime_resource_policy_integration.py
scripts/plan_runtime_resources.py
```

Small updates are expected in:

```text
src/drpo/runtime_resource_adapters.py
scripts/run_e7_canonical_exp_horizon_joint_auto.py
scripts/run_countdown_e8_oracle_offline_v2_taper_auto.py
```

The dedicated wrappers become compatibility callers of the common integration;
they are not deleted during V1.

### 3.3 Tests

```text
tests/test_runtime_resource_policy_contract.py
tests/test_runtime_resource_policy_identity.py
tests/test_runtime_resource_policy_engine.py
tests/test_runtime_resource_policy_drpo_integration.py
tests/test_runtime_resource_policy_shadow_equivalence.py
```

Phase D adds focused RunSpec/formal-channel tests only after authorization.

## 4. Production line-count estimate

Line counts exclude blank lines, comments-only documentation, schemas, and tests.
They are estimates, not quotas, but significant overruns trigger architecture
review.

| Component | Estimated production LOC | Notes |
|---|---:|---|
| `model.py` | 80–120 | enums, frozen dataclasses, structured errors |
| `contract.py` | 100–150 | strict parser and mode validation |
| `identity.py` | 60–100 | canonicalization, finite-value checks, hashes |
| `engine.py` | 180–260 | state machine and atomic artifacts |
| package API | 15–30 | exports and protocol version |
| **Portable core subtotal** | **435–660** | target ceiling: 700 |
| DRPO policy integration | 120–190 | registry, fingerprints, provider bridge |
| standalone plan CLI | 70–120 | load inputs and emit result |
| E7/E8 adapter wrapping | 80–160 | reuse current selectors |
| compatibility wrapper edits | 30–80 | no scientific changes |
| **Phase B/C production total** | **735–1,210** | portable core plus DRPO shadow path |
| RunSpec compatibility integration | 80–150 | Phase D only |
| formal guard/preflight hook | 100–180 | Phase D only |
| cutover validation/reporting | 40–90 | Phase D only |
| **End-state V1 production total** | **955–1,630** | before optional providers/backends |

### Core-size conclusion

The genuinely reusable core should be approximately **500–650 lines** and must
remain below **700 production lines** unless design review approves a larger
surface. The complete DRPO integration through formal cutover is expected to be
approximately **1,100–1,500 production lines**.

This is materially smaller than adopting a scheduler framework or building a
backend abstraction, while still providing a real cross-project contract,
identity, and resume state machine.

## 5. Test and documentation size

Tests are expected to exceed production code because most risk is in negative
state transitions.

| Area | Estimated LOC |
|---|---:|
| portable core unit tests | 550–850 |
| DRPO adapter/integration tests | 350–600 |
| shadow-equivalence tests | 200–350 |
| Phase-D formal/RunSpec tests | 350–650 |
| **Total tests** | **1,450–2,450** |
| schema and design/operations docs | 500–900 |

The larger test-to-core ratio is intentional. It protects resume identity,
fail-closed behavior, and scientific isolation without increasing runtime
architecture.

## 6. Phase plan

## Phase A — design-only

Deliverables:

- scope;
- architecture;
- JSON Schema draft;
- implementation and cost plan;
- Stage-2 reopen authorization draft.

Acceptance:

- no executable files changed;
- design boundaries and line budget approved;
- review confirms the core is project-neutral.

Estimated effort: **0.5–1 engineer-day**.

## Phase B — portable core and standalone plan

Deliverables:

- neutral standard-library package;
- explicit adapter protocol and registry;
- strict contract parser;
- identity and artifacts;
- fresh auto/fixed/exempt and resume state machine;
- standalone DRPO plan command using current machine provider;
- no formal or RunSpec enforcement.

Required tests:

- all schema modes and malformed combinations;
- canonical identity stability;
- non-finite value rejection;
- atomic-write failure;
- unknown adapter/version;
- existing-selection conflict;
- missing/malformed resume identity;
- resume never calls auto selection;
- unsafe resume blocks;
- scientific fingerprint drift blocks.

Estimated effort: **2–3 engineer-days**.

## Phase C — E7/E8 shadow equivalence

Deliverables:

- current E7 and E8 selectors wrapped as adapters;
- dedicated wrappers call or compare against the common path;
- plan-only shadow reports;
- real server comparison packages.

Acceptance:

- E7 selected worker count and limiting constraints match the validated dedicated
  path under the same snapshot and policy;
- E8 selected device IDs and host/VRAM constraints match;
- selection identities are stable;
- no full sweep or scientific run is launched;
- no orphan probe processes;
- `scientific_matrix_changed=false`.

Estimated effort: **1.5–2.5 engineer-days**, plus **0.5 day** of server acceptance
and review latency.

## Phase D — RunSpec/formal integration

Precondition:

- explicit user approval of the Stage-2 reopen authorization;
- Phase-C shadow acceptance complete;
- rollback tested.

Deliverables:

- optional RunSpec field during compatibility window;
- formal guard preflight hook;
- exact-resume resource identity enforcement;
- compatibility telemetry for historical RunSpecs;
- separate enforcement cutover after observation.

Acceptance:

- unmanaged historical RunSpecs retain existing behavior during compatibility;
- managed fresh runs cannot bypass policy preflight;
- managed resume cannot recalculate selection;
- rollback disables enforcement without altering experiment history;
- full governance and repository gates pass.

Estimated effort: **2–3.5 engineer-days**, plus at least **one observation cycle**
before default enforcement.

## Phase E — optional enhancements

Independent estimates:

| Enhancement | Extra production LOC | Effort |
|---|---:|---:|
| optional `psutil` provider | 80–140 | 0.5–1 day |
| optional NVML provider | 100–170 | 0.5–1 day + GPU test |
| Submitit/Slurm consumer | 180–320 | 1.5–3 days |
| Ray consumer | 250–450 | 2–4 days |
| E7 throughput-knee policy | 250–450 | 2–4 days + server runs |
| E8 phase-peak probe | 220–400 | 2–4 days + GPU runs |

None is required for V1 portability or formal integration.

## 7. Overall development cost

For design through controlled formal integration, excluding optional Phase E:

- **engineering effort:** approximately **6–9 focused engineer-days**;
- **calendar duration:** approximately **1–2 weeks**, mainly due to PR review,
  governance authorization, server shadow runs, and observation rather than coding;
- **production code:** approximately **1,100–1,500 lines** at the likely midpoint;
- **portable core:** approximately **500–650 lines**;
- **tests:** approximately **1,500–2,300 lines**.

A coding agent can reduce typing time, but it does not remove server acceptance,
identity review, Stage-2 authorization, or observation requirements. The project
should therefore plan by review and validation steps rather than raw coding hours.

## 8. Risk assessment

| Risk | Level | Mitigation |
|---|---|---|
| resume identity or selection drift | high | immutable identity, negative tests, fail closed |
| formal-channel anti-bypass integration | high | separate Phase D and Stage-2 reopen |
| RunSpec backward compatibility | medium-high | optional compatibility window, no history rewrite |
| adapter output-shape mismatch | medium | common validation and E7/E8 shadow equivalence |
| core becoming framework-specific | medium | no project imports and no backend dependency |
| duplicated provenance files | low-medium | one authoritative selection plus one identity |
| schema over-expansion | low-medium | opaque adapter parameters and strict common fields |

## 9. Stop/go gates

- **After Phase A:** stop if the portable core cannot remain under the 700-line
  ceiling without hiding workload logic in the core.
- **After Phase B:** stop if the standalone state machine cannot reproduce exact
  identity and resume behavior with deterministic tests.
- **After Phase C:** stop if E7/E8 require core-level scientific special cases.
- **Before Phase D:** require explicit Stage-2 authorization and successful rollback
  rehearsal.
- **Before enforcement:** require observation showing no unmanaged launch paths are
  unintentionally blocked or bypassing policy.

## 10. Recommended decision

Proceed with the architecture as designed. It provides the long-term cross-project
foundation, but keeps the reusable core near 500–650 lines by excluding execution
orchestration and optional system libraries. Implement Phase B and Phase C as
separate PRs; do not authorize Phase D merely by approving this design.
