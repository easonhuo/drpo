# Runtime Resource Policy V1 — implementation plan and cost estimate

**Claim:** `GOV-RUNTIME-RESOURCE-POLICY-INTEGRATION-01`  
**Status:** refined design candidate.  
**Estimate base:** current E7/E8 selectors and machine discovery are reused, not
rewritten.

## 1. Review-driven delivery strategy

The implementation is split at responsibility and governance boundaries:

```text
A. design and review
B. portable create/verify/revalidate core
C. E7/E8 plan-only shadow equivalence
D. separately authorized formal compatibility integration
E. optional providers, schedulers, or throughput work
```

Phase B/C do not change the formal execution default. Phase D is the only phase that
may touch the closed Stage-2 channel and requires a later explicit authorization.

## 2. Decisions that keep V1 efficient and non-redundant

V1 adopts the following hard constraints:

- only two state-changing core operations: create and revalidate;
- `plan` is a project CLI command, not a third core state;
- one authoritative `RUNTIME_SELECTION.json` with embedded identity;
- one conditional attempt-local `RUNTIME_REVALIDATION.json`;
- both artifact layouts are explicitly versioned public formats;
- no separate identity sidecar;
- no backend, machine-provider, or dynamic-plugin abstraction in the core;
- caller supplies plain machine observation and an explicit adapter mapping;
- caller supplies workload and scientific fingerprints independently of the adapter;
- adapter owns bounded probes, measurement cache, fallback, and resource arithmetic;
- core validates common shape, freezes identity, and writes artifacts;
- current E7/E8 selectors are wrapped rather than copied;
- no mandatory third-party dependency;
- no historical RunSpec migration in the first release;
- no throughput search, online resizing, batch tuning, multi-node support, or
  scheduler lifecycle.

Any implementation that violates one of these controls requires a design amendment
and updated line-count estimate before review.

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

- immutable result models and common error codes;
- strict contract parsing and canonical defaults;
- canonical JSON and embedded SHA-256 identity;
- `create_selection`, `verify_selection`, and `revalidate_selection`;
- versioned selection/revalidation serialization;
- common adapter-output validation;
- atomic selection and revalidation writes;
- explicit caller-supplied adapter mapping.

The core has no DRPO import and no knowledge of RunSpec, E7/E8, providers, cache
formats, subprocesses, or schedulers.

### 3.2 DRPO integration

```text
src/drpo/runtime_resource_policy_integration.py
scripts/plan_runtime_resources.py
```

Focused changes are expected in:

```text
src/drpo/runtime_resource_adapters.py
scripts/run_e7_canonical_exp_horizon_joint_auto.py
scripts/run_countdown_e8_oracle_offline_v2_taper_auto.py
```

The existing wrappers remain available as rollback paths and may first compare their
current output with the common path before internally delegating to it.

### 3.3 Tests

```text
tests/test_runtime_resource_policy_contract.py
tests/test_runtime_resource_policy_identity.py
tests/test_runtime_resource_policy_engine.py
tests/test_runtime_resource_policy_drpo_integration.py
tests/test_runtime_resource_policy_shadow_equivalence.py
```

Phase D adds formal/RunSpec tests only after Stage-2 authorization.

## 4. Refined production line-count estimate

Line counts exclude blank lines, documentation, schemas, and tests.

| Component | Estimated production LOC | Notes |
|---|---:|---|
| `model.py` | 60–90 | frozen values and structured errors |
| `contract.py` | 80–120 | common modes and deterministic normalization |
| `identity.py` | 50–80 | canonicalization and embedded digest |
| `engine.py` | 140–210 | create/verify/revalidate and atomic writes |
| package API | 15–25 | exports and protocol version |
| **Portable core subtotal** | **345–525** | preferred target 400–500; hard ceiling 550 |
| DRPO policy integration | 100–170 | fingerprints, provider bridge, adapter map |
| standalone plan CLI | 60–100 | load inputs and call create |
| E7/E8 adapter wrapping | 60–130 | reuse current selectors and caches |
| compatibility wrapper edits | 20–60 | no scientific changes |
| **Phase B/C production total** | **585–985** | core plus DRPO shadow path |
| RunSpec compatibility integration | 70–130 | Phase D only |
| formal guard/preflight hook | 90–160 | Phase D only |
| cutover validation/reporting | 35–75 | Phase D only |
| **End-state V1 production total** | **780–1,350** | before optional Phase E |

### Size conclusion

The genuinely reusable core is now expected to be **400–500 production lines**, with
a hard review ceiling of **550**. The likely complete DRPO V1 through controlled
formal integration is **950–1,200 production lines**.

The reduction comes from four concrete removals, not from hiding complexity:

1. no separate plan state;
2. no identity sidecar;
3. no generic cache engine;
4. no provider/backend protocol layer.

Resource-specific complexity remains visible in thin project adapters, where it can
be tested against the existing E7/E8 implementations.

## 5. Refined test and documentation estimate

Most risk remains in negative transitions, so tests still exceed core code.

| Area | Estimated LOC |
|---|---:|
| portable core unit tests | 420–700 |
| DRPO adapter/integration tests | 300–500 |
| shadow-equivalence tests | 180–300 |
| Phase-D formal/RunSpec tests | 300–550 |
| **Total tests** | **1,200–2,050** |
| schema and design/operations docs | 500–900 |

Tests must emphasize identity stability, artifact-version compatibility, unsafe
resume, scientific drift, atomic-write failure, and adapter-output validation rather
than producing one test per internal helper.

## 6. Phase plan

### Phase A — design-only

Deliverables:

- scope and architecture;
- strict contract schema;
- packaging decision;
- implementation/cost estimate;
- draft Stage-2 reopen boundary.

Acceptance:

- no executable file or default behavior changes;
- five-pass architecture review completed;
- responsibility, identity, artifact, cache, and integration boundaries agree;
- CI and governance gates pass.

Estimated effort: **0.5–1 engineer-day**.

### Phase B — portable core and standalone plan

Deliverables:

- neutral standard-library package;
- explicit adapter protocol and caller-supplied mapping;
- deterministic parser and embedded identity;
- create/verify/revalidate operations;
- versioned one-file selection and attempt-local revalidation formats;
- standalone DRPO plan CLI using the current provider;
- no RunSpec or formal enforcement.

Required tests:

- all valid and invalid contract combinations;
- omitted-field normalization and schema-default non-reliance;
- stable identity under key order and irrelevant provenance changes;
- identity changes for selected resources, fingerprints, adapter version, or declared
  resource binding;
- selection/revalidation format version rejection and compatibility;
- non-finite value rejection;
- unknown adapter/version;
- invalid adapter output;
- existing-selection conflict;
- atomic-write failure and no partial authority;
- missing or malformed resume selection;
- resume never calls auto selection;
- unsafe resume blocks without mutating the original;
- workload/scientific fingerprint drift blocks;
- isolated import without DRPO or scientific dependencies.

Estimated effort: **1.5–2.5 engineer-days**.

### Phase C — E7/E8 shadow equivalence

Deliverables:

- existing E7 and E8 selectors wrapped as adapters;
- cache and fallback remain in existing adapter logic;
- dedicated and common paths compared under the same inputs;
- plan-only real-server shadow evidence.

Acceptance:

- E7 worker count and limiting constraints match the validated path;
- E8 device IDs and host/VRAM constraints match;
- selected resources are identical even when evidence route is probe versus cache;
- only declared resource binding affects identity;
- one authoritative selection file is produced;
- no full sweep or scientific run starts;
- no orphan probe process remains;
- `scientific_matrix_changed=false`.

Estimated effort: **1–2 engineer-days**, plus approximately **0.5 day** of server
acceptance and review latency.

### Phase D — RunSpec/formal compatibility integration

Preconditions:

- explicit Stage-2 reopen approval;
- Phase-C real-hardware shadow accepted;
- rollback rehearsal passes;
- compatibility behavior for historical RunSpecs approved.

Deliverables:

- optional `runtime_resource` during the compatibility window;
- formal preflight hook consuming a verified selection result;
- embedded identity and exact-resource resume enforcement;
- compatibility telemetry;
- separate enforcement cutover after observation.

Acceptance:

- historical unmanaged RunSpecs retain approved behavior;
- managed fresh launches cannot bypass preflight;
- managed resume cannot recalculate selection;
- formal guard does not own provider, probe, cache, or E7/E8 logic;
- rollback restores the old path without changing experiment history;
- full repository and governance gates pass.

Estimated effort: **2–3 engineer-days**, plus at least one observation cycle before
default enforcement.

### Phase E — optional enhancements

These remain independent claims:

| Enhancement | Extra production LOC | Effort |
|---|---:|---:|
| optional `psutil` integration provider | 60–110 | 0.5–1 day |
| optional NVML integration provider | 80–140 | 0.5–1 day + GPU test |
| Submitit/Slurm consumer | 160–280 | 1–2.5 days |
| Ray consumer | 220–380 | 1.5–3.5 days |
| E7 throughput-knee policy | 250–450 | 2–4 days + server runs |
| E8 phase-peak probe | 220–400 | 2–4 days + GPU runs |

None is required for V1 portability or formal integration.

## 7. Overall development cost

For design through controlled formal integration, excluding Phase E:

- **focused engineering effort:** approximately **5–8 engineer-days**;
- **calendar duration:** approximately **1–2 weeks**, dominated by review, server
  shadow acceptance, Stage-2 authorization, and observation;
- **likely production code:** approximately **950–1,200 lines**;
- **portable core:** approximately **400–500 lines**;
- **tests:** approximately **1,200–2,050 lines**.

A coding agent may reduce typing time but not identity review, real-hardware
acceptance, governance authorization, or observation time.

## 8. Risk assessment

| Risk | Level | Mitigation |
|---|---|---|
| resume identity or resource drift | high | embedded digest, immutable selection, negative tests |
| formal-channel anti-bypass | high | separate Phase D and narrow reopen |
| RunSpec backward compatibility | medium-high | compatibility window and no history rewrite |
| adapter judges its own science identity | medium-high | caller supplies scientific fingerprint independently |
| identity over-binding to machines/evidence | medium | minimal stable payload and explicit resource binding |
| duplicated cache or provenance logic | medium | adapter-owned cache; one selection authority |
| artifact-format drift | medium | public version fields and compatibility tests |
| framework or provider leakage | medium | plain data inputs and no plugin/provider framework |
| schema expansion | low-medium | opaque adapter parameters and strict common envelope |

## 9. Stop/go gates

- **After Phase A:** stop if the core cannot remain under 550 production lines
  without hiding workload logic.
- **After Phase B:** stop if identity is not stable under irrelevant provenance
  changes, artifact formats are not versioned, or resume can reach auto selection.
- **After Phase C:** stop if E7/E8 require scientific special cases in the core or
  common cache logic duplicates the dedicated selectors.
- **Before Phase D:** require explicit Stage-2 authorization and rollback rehearsal.
- **Before enforcement:** require an observation window showing neither bypass nor
  accidental blocking of approved historical paths.

## 10. Recommended decision

Proceed with Phase B only after this refined design is approved. Phase B and C remain
separate PRs. Approval of this design does not authorize Stage-2 integration or
default-policy cutover.