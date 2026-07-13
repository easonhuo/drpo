# Runtime Resource Policy Architecture V1

**Claim:** `GOV-RUNTIME-RESOURCE-POLICY-INTEGRATION-01`  
**Status:** design candidate; no executable behavior is authorized by this file.  
**Design target:** generic across projects, efficient in the common path, and small
enough that identity and resume behavior remain reviewable.

## 1. Executive decision

V1 is a **project-neutral resource-selection preflight library**. It is not a
scheduler, process supervisor, machine-discovery framework, or execution backend.

The complete boundary is:

```text
project integration
  supplies policy + fingerprints + machine observation + adapter
        |
        v
portable policy core
  validates -> selects/revalidates -> freezes one authoritative artifact
        |
        v
existing runner / formal guard / Slurm or Ray consumer
```

The core exposes only two state-changing operations:

```text
create_selection
revalidate_selection
```

A command named `plan` is a project CLI operation that calls `create_selection`; it
is not a third core state. This removes a redundant fresh/plan distinction and keeps
all creation semantics in one place.

## 2. Why this boundary is both generic and light

A full orchestrator would need to own subprocess lifecycle, retries, logging,
preemption, multi-node placement, scheduler submission, and recovery. That would
duplicate existing runners, Slurm, Ray, or Kubernetes.

A workload-specific wrapper is initially smaller, but it duplicates contract
validation, immutable identity, resume rules, artifact semantics, and failure codes
for every project.

V1 therefore owns only stable project-independent semantics:

- strict common-contract parsing and normalization;
- `auto`, `fixed`, and `exempt` creation modes;
- adapter resolution through a caller-supplied mapping;
- canonical identity construction and verification;
- common adapter-output validation;
- atomic creation of one immutable selection artifact;
- attempt-local resume revalidation records;
- fail-closed structured results.

The project integration owns everything that changes with a workload or machine:

- machine discovery and provider fallback;
- workload and scientific fingerprint construction;
- adapter context and resource-selection algorithms;
- bounded probes, measurement cache, and fallback policy;
- translating selected resources into runner arguments;
- lifecycle, heartbeat, packaging, terminal audit, and scheduling.

This division is the main anti-redundancy rule: **the core owns semantics; adapters
own evidence and resource arithmetic; runners own execution.**

## 3. Portable source boundary

The initial neutral implementation lives under:

```text
src/runtime_resource_policy/
```

It must not import `drpo`, E7, E8, Hopper, Countdown, a project registry, a machine
provider, or a runner.

Proposed modules:

```text
runtime_resource_policy/
  __init__.py   public API and protocol version
  model.py      immutable values and structured errors
  contract.py   strict parsing and normalization
  identity.py   canonical JSON and SHA-256 verification
  engine.py     create/revalidate operations and atomic artifacts
```

The core uses the Python standard library only. JSON Schema is published for
interoperability, but the production parser does not require `jsonschema`, Pydantic,
`psutil`, NVML, Ray, Dask, Submitit, or another framework.

## 4. Inputs and public API

### 4.1 Caller-supplied inputs

The project integration supplies explicit data:

```text
normalized policy contract
adapter mapping
adapter context
workload fingerprint
scientific-invariant fingerprint
machine observation
source provenance
work directory / attempt directory
```

`adapter_context` is opaque to the core. The core neither defines a generic
`ProjectRequest` hierarchy nor interprets project paths or scientific fields.

Machine observation is also opaque except for canonical JSON safety checks. Provider
selection and fallback occur before the core is called and are recorded as
provenance.

### 4.2 Minimal API

Conceptually:

```python
def create_selection(
    *,
    contract,
    adapters,
    adapter_context,
    workload_fingerprint,
    scientific_fingerprint,
    machine_observation,
    provenance,
    output_path,
) -> SelectionResult: ...


def revalidate_selection(
    *,
    selection_path,
    adapters,
    adapter_context,
    workload_fingerprint,
    scientific_fingerprint,
    machine_observation,
    provenance,
    attempt_output_path,
) -> RevalidationResult: ...
```

A pure `verify_selection(document)` helper may be public. It verifies schema,
canonical identity, and common invariants without performing selection or I/O.

No backend, provider, plugin-discovery, or scheduler protocol is part of the V1
core API.

## 5. Policy contract

The RunSpec-facing contract remains intentionally small:

```yaml
runtime_resource:
  schema_version: 1
  mode: auto                 # auto | fixed | exempt
  adapter: e7_cpu_v1         # required for auto/fixed
  adapter_version: "1"       # optional exact pin
  policy:
    profile: conservative
    parameters: {}           # adapter-owned and adapter-validated
  fixed:                     # fixed mode only
    request: {}
  exempt:                    # exempt mode only
    reason: "..."
```

The production parser, not JSON Schema defaults, canonicalizes omitted optional
fields. For example, omitted `policy` becomes the explicit canonical value:

```json
{"profile":"conservative","parameters":{}}
```

### 5.1 `auto`

For a new selection, the adapter inspects current resources and may reuse compatible
measurement evidence or run a bounded probe. It returns one safe selection.

### 5.2 `fixed`

The caller supplies an adapter-specific fixed request. The adapter validates current
capacity before the selection is frozen. Fixed mode is not a bypass around safety.

### 5.3 `exempt`

The caller supplies a non-empty reason and optional classification. Exempt mode does
not resolve an adapter and cannot carry adapter, policy, or fixed fields. The core
writes an auditable no-resource-selection document.

### 5.4 Adapter version pinning

`adapter_version` is an optional requested exact pin. The resolved exact adapter
version is always recorded in the selection. A supplied pin that does not match the
resolved adapter fails closed.

## 6. Minimal adapter protocol

The adapter protocol contains resource-specific behavior only:

```python
class ResourceAdapter(Protocol):
    adapter_id: str
    adapter_version: str

    def select_auto(
        self,
        *,
        adapter_context: object,
        machine_observation: Mapping[str, object],
        policy_profile: str,
        policy_parameters: Mapping[str, object],
    ) -> AdapterSelection: ...

    def validate_fixed(
        self,
        *,
        adapter_context: object,
        machine_observation: Mapping[str, object],
        fixed_request: Mapping[str, object],
    ) -> AdapterSelection: ...

    def revalidate(
        self,
        *,
        adapter_context: object,
        machine_observation: Mapping[str, object],
        selected_resources: Mapping[str, object],
        resource_binding: Mapping[str, object],
    ) -> AdapterRevalidation: ...
```

The adapter does **not** construct workload or scientific fingerprints. Those are
project-governance inputs supplied independently, which prevents adapter code from
becoming the sole judge of its own scientific identity.

The adapter returns a common envelope:

```text
selected_resources
resource_binding
limits
limiting_factor
measurement_evidence
cache_status
fallback_status
limitations
scientific_matrix_changed = false
```

Only `selected_resources`, `resource_binding`, and common safety flags participate
in generic identity or validation. Measurement, cache, fallback, and limitation
fields are recorded provenance owned by the adapter.

## 7. Creation and resume semantics

### 7.1 Create: auto or fixed

```text
parse and normalize contract
  -> reject conflicting existing immutable selection
  -> resolve adapter and optional version pin
  -> validate caller fingerprints and machine observation as canonical JSON
  -> call select_auto or validate_fixed
  -> validate common output and scientific_matrix_changed=false
  -> construct stable identity payload
  -> atomically write RUNTIME_SELECTION.json
  -> return ALLOW
```

### 7.2 Create: exempt

```text
parse explicit exemption
  -> construct auditable no-selection document
  -> atomically write RUNTIME_SELECTION.json
  -> return ALLOW_WITHOUT_RESOURCE_SELECTION
```

### 7.3 Resume revalidation

```text
load RUNTIME_SELECTION.json
  -> verify embedded identity digest
  -> verify adapter, workload, and scientific fingerprints
  -> resolve the recorded adapter version
  -> call adapter.revalidate with the original selected resources
       -> safe: atomically write attempt-local RUNTIME_REVALIDATION.json; ALLOW
       -> unsafe: write structured block record; BLOCK
```

Resume never:

- calls `select_auto`;
- changes worker or device count;
- rewrites the original selection;
- treats cache as authority;
- silently changes adapter or resource binding.

A deliberate resource-identity change creates a new run or uses a separately
registered recovery protocol; it is not ordinary resume.

## 8. Identity model

### 8.1 Stable identity inputs

The immutable digest binds only facts that define the selected resource contract:

- normalized requested policy contract;
- core protocol and selection schema versions;
- resolved adapter ID and exact version;
- workload fingerprint;
- scientific-invariant fingerprint;
- selected resources;
- adapter-defined `resource_binding` when same-host/device semantics are required.

### 8.2 Provenance excluded from stable identity

The following remain fully recorded but do not change the digest:

- current free memory, utilization, and load;
- machine/provider observation details not declared in `resource_binding`;
- probe logs and measured peaks;
- cache hit/miss and fallback route;
- timestamps;
- repository commit and dirty-state provenance.

This avoids two failure modes:

1. equivalent hosts cannot resume solely because a hostname changed;
2. the same selected resources receive different identities because one path used a
   probe and another reused compatible evidence.

Adapters may declare a minimal `resource_binding`, such as a host class, GPU UUID
set, or topology fingerprint, only when changing that value would alter execution
semantics. Same-host enforcement is therefore explicit rather than globally assumed.

### 8.3 Canonicalization

Identity uses UTF-8 JSON, sorted keys, compact separators, finite numbers only, and
no implicit path normalization. Canonicalization is versioned by the core protocol.

## 9. Artifact model

V1 has one authoritative immutable creation artifact:

```text
RUNTIME_SELECTION.json
```

Its top-level layout is a **versioned public format**. Phase B must publish and test
that format before another project vendors the core. It contains both selection and
embedded identity:

```json
{
  "selection_schema_version": 1,
  "identity": {
    "algorithm": "sha256",
    "canonicalization_version": 1,
    "digest": "..."
  },
  "requested_contract": {},
  "resolved_adapter": {},
  "workload_fingerprint": {},
  "scientific_fingerprint": {},
  "selected_resources": {},
  "resource_binding": {},
  "machine_observation": {},
  "measurement_evidence": {},
  "cache_status": {},
  "fallback_status": {},
  "limitations": [],
  "provenance": {},
  "scientific_matrix_changed": false
}
```

The digest is computed from the documented identity payload, not from the full file
and not from the digest field itself. Independent verification recomputes it from
the selection document.

There is no separate `RUNTIME_SELECTION_IDENTITY.json`; a sidecar would duplicate
authority and create split-brain repair cases.

Resume produces one attempt-local, versioned artifact:

```text
RUNTIME_REVALIDATION.json
```

It records the original selection digest, current observation, adapter result,
provenance, and structured allow/block reasons. It never mutates the original file.

Adapter-owned probe logs may remain under `_runtime_resource_probe/`. Checkpoints and
model payloads are not policy artifacts.

## 10. Cache and fallback semantics

Cache is an adapter implementation detail and evidence reuse, not a core state or
scheduling authority.

Each adapter defines its measurement-cache key and compatibility checks using the
inputs relevant to that workload. It must still inspect current dynamic capacity
before returning a selection. The core records cache and fallback evidence but does
not implement a second generic cache that duplicates E7/E8 logic.

A fallback is valid only when the adapter returns it as an explicit safe selection
with structured evidence. No core or integration error may silently fall back from
managed policy to an unmanaged launcher.

## 11. Scientific isolation

The project supplies the scientific fingerprint independently of the adapter. For
DRPO it covers all frozen workload fields, including data, method matrix, seeds,
coefficients, batch, horizon, optimizer, evaluation, and stopping rules.

The integration layer compares the fingerprint before and after applying selected
runtime fields. Any drift blocks execution.

V1 does not claim that concurrency can never affect floating-point ordering or wall
clock. It claims that declared scientific configuration is unchanged and the exact
runtime resource schedule is explicit provenance.

## 12. DRPO integration and migration

### Phase B: portable core

- implement the neutral library and a DRPO `plan` CLI;
- reuse current machine discovery as integration code;
- do not change existing wrappers, RunSpecs, or formal guards.

### Phase C: E7/E8 shadow equivalence

- wrap current E7 and E8 selectors without copying their algorithms;
- compare selected resources and limiting constraints under identical inputs;
- compare one-file artifact semantics;
- run plan/selection only, not full scientific sweeps.

### Phase D: formal compatibility

Only after separate Stage-2 authorization:

- accept an optional `runtime_resource` block;
- invoke project preflight before managed formal launch;
- verify embedded selection identity and attempt-local resume revalidation;
- mark historical RunSpecs without the block as `legacy_unmanaged` during the
  approved compatibility window;
- enforce defaults only after a separate cutover decision.

The formal guard consumes policy results. It does not discover machines, run probes,
or embed E7/E8 arithmetic.

## 13. Failure codes

The minimum common codes are:

```text
INVALID_CONTRACT
UNKNOWN_ADAPTER
ADAPTER_VERSION_MISMATCH
INVALID_FIXED_REQUEST
EXEMPT_REASON_REQUIRED
EXISTING_SELECTION_CONFLICT
MISSING_RESUME_SELECTION
IDENTITY_MISMATCH
WORKLOAD_FINGERPRINT_DRIFT
SCIENTIFIC_FINGERPRINT_DRIFT
NO_SAFE_CAPACITY
RESUME_CAPACITY_UNSAFE
ADAPTER_FAILURE
INVALID_ADAPTER_OUTPUT
ARTIFACT_WRITE_FAILURE
```

Adapter-specific reasons are nested under a common `details` field and do not expand
the portable error enum for every workload condition.

## 14. Deliberately deferred features

Separate claims are required for:

- execution backends or scheduler submission;
- dynamic plugin discovery;
- generic machine-provider abstractions;
- throughput-knee search;
- online resizing or preemption;
- multi-node placement;
- automatic batch or gradient-accumulation changes;
- distributed cache services;
- a standalone package release before a second real consumer exists.

## 15. Acceptance criteria

The architecture is accepted only if implementation demonstrates:

- the portable core imports without DRPO or scientific dependencies;
- public behavior is expressible through create, verify, and revalidate operations;
- one selection file is the only creation authority;
- selection and revalidation formats are explicitly versioned and tested;
- auto, fixed, and exempt contracts normalize deterministically;
- resume cannot invoke auto selection or mutate resources;
- identity is stable across irrelevant provenance changes;
- adapter-defined resource binding can enforce genuinely required host/device
  identity;
- cache and measurement remain outside the core without losing provenance;
- E7/E8 shadow choices match their validated dedicated paths;
- the core remains below the reviewed line-count ceiling;
- no scientific variable or default formal behavior changes during Phase B/C.

## 16. Rollback

Before formal cutover, rollback is simply to stop invoking the new preflight and use
the unchanged dedicated launchers. All selection, revalidation, probe, and failure
evidence is preserved. An already started run is never reinterpreted under a new
resource identity.