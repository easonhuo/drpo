# Runtime Resource Policy Architecture V1

**Claim:** `GOV-RUNTIME-RESOURCE-POLICY-INTEGRATION-01`  
**Status:** design candidate; no executable behavior is authorized by this file.  
**Design target:** generic and cross-project, with the smallest stable core that
preserves scientific and execution identity.

## 1. Executive decision

V1 will be a **preflight policy library**, not a scheduler and not an execution
framework. It receives a versioned policy contract plus project-supplied workload
and machine fingerprints, invokes one registered workload adapter, and emits an
immutable resource selection. The existing project runner consumes that selection.

The minimum architecture is:

```text
project RunSpec/config
        |
        v
portable policy core  <---- project adapter registry
        |                       |
        |                       +-- E7 CPU adapter
        |                       +-- E8 GPU adapter
        |                       +-- another project's adapters
        v
RUNTIME_SELECTION.json + identity
        |
        v
existing local runner / formal guard / Slurm wrapper / other consumer
```

The core deliberately does **not** launch tasks. This removes the need for a
mandatory local/Submitit/Ray backend abstraction in V1, keeps the core small, and
lets every project preserve its existing execution and failure semantics.

## 2. Why this is the optimal boundary

A full resource orchestrator would need to own subprocess lifecycle, logs,
retries, preemption, cluster submission, multi-node placement, and recovery. That
would duplicate Slurm, Ray, Kubernetes, or existing project guards and would make
the integration substantially heavier.

A one-off wrapper per workload would be smaller initially, but would duplicate
fresh/resume rules, identity, cache semantics, and provenance. It would not be a
cross-project foundation.

V1 therefore owns only the project-independent control plane:

- policy parsing and strict validation;
- `auto`, `fixed`, and `exempt` state semantics;
- fresh versus resume transitions;
- adapter lookup and version checks;
- stable fingerprint and identity construction;
- immutable artifact creation;
- cache acceptance rules;
- fail-closed errors and machine-readable reasons.

Projects own:

- workload fingerprint construction;
- resource measurement and selection algorithms;
- scientific-invariant fingerprinting;
- applying a valid selection to their existing runner;
- formal lifecycle, heartbeat, packaging, and terminal audit.

## 3. Portability model

### 3.1 Neutral package

The portable implementation is proposed as a neutral Python package under:

```text
src/runtime_resource_policy/
```

It must not import `drpo`, E7, E8, Hopper, Countdown, a project registry, or a
project runner. Another repository can vendor or package this directory without
bringing DRPO scientific code.

Initial modules:

```text
runtime_resource_policy/
  __init__.py      public API and version
  model.py         enums and immutable dataclasses
  contract.py      strict parser and schema-facing validation
  identity.py      canonical JSON and SHA-256 identity
  engine.py        state machine and artifact transaction
```

A standalone CLI is project integration code, not part of the minimum reusable
library, because CLI arguments and RunSpec loading differ by project.

### 3.2 No mandatory third-party dependency

The reference core uses the Python standard library only. The repository ships a
JSON Schema for interoperability, but production validation does not require the
`jsonschema` package. CI may optionally validate the schema with `jsonschema`.

System probes remain provider implementations outside the portable core:

- current procfs/cgroup and `nvidia-smi` path: default in DRPO;
- `psutil`: optional host provider;
- NVML / `nvidia-ml-py`: optional GPU provider;
- cloud or scheduler APIs: future optional providers.

Submitit, Ray, Dask, Slurm, and Kubernetes are execution consumers or future
integrations. They are not V1 core dependencies.

## 4. Policy contract

The RunSpec-facing contract is intentionally small:

```yaml
runtime_resource:
  schema_version: 1
  mode: auto                 # auto | fixed | exempt
  adapter: e7_cpu_v1         # required for auto/fixed
  policy:
    profile: conservative
    parameters: {}           # adapter-owned and adapter-validated
  fixed: null                # object only for a fresh fixed request
  exempt: null               # object only for exempt
```

The portable core validates only common fields and mode combinations. Adapter
parameters are opaque to the core and validated by the selected adapter.

### 4.1 `auto`

For a fresh run, the adapter measures or inspects resources and returns a safe
selection. The result is frozen before any scientific runner starts.

### 4.2 `fixed`

For a fresh run, the contract supplies an adapter-specific fixed request. The
adapter must verify that the request is currently safe before the core freezes it.
A fixed request is not an escape hatch from resource validation.

### 4.3 `exempt`

The contract gives a non-empty reason and optional classification. The core emits
an auditable no-selection decision. Exemption is explicit; the engine does not
infer that a task is small.

### 4.4 Resume override

Invocation context is supplied separately from the policy contract:

```text
fresh | resume | plan
```

For `resume`, the effective mode is always `fixed_existing`, regardless of the
original requested mode. The original selection and identity are required.
The engine never calls `select_auto` during resume.

## 5. Minimal adapter protocol

The adapter API is intentionally narrower than a general scheduler plugin:

```python
class ResourceAdapter(Protocol):
    adapter_id: str
    adapter_version: str

    def workload_fingerprint(self, request: ProjectRequest) -> Mapping[str, object]: ...

    def select_auto(
        self,
        request: ProjectRequest,
        machine_snapshot: Mapping[str, object],
        work_dir: Path,
        policy_parameters: Mapping[str, object],
    ) -> AdapterSelection: ...

    def validate_fixed(
        self,
        request: ProjectRequest,
        machine_snapshot: Mapping[str, object],
        fixed_request: Mapping[str, object],
    ) -> AdapterSelection: ...

    def revalidate(
        self,
        request: ProjectRequest,
        machine_snapshot: Mapping[str, object],
        existing_selection: Mapping[str, object],
    ) -> RevalidationResult: ...
```

The adapter may run a bounded probe inside `select_auto`; this preserves the
already validated E7 implementation and avoids building a generic measurement
workflow prematurely. The adapter must return structured evidence and is
responsible for cleanup of workload-specific probe payloads.

The core enforces common shape and identity requirements on the returned value:

```text
selected_resources
limits
limiting_factor
measurement_evidence
fallback
cache
limitations
scientific_matrix_changed = false
```

## 6. Machine provider boundary

The core receives a machine snapshot as data; it does not discover the machine.
A project integration chooses a provider and records:

```text
provider_id
provider_version
snapshot
static_identity
observed_utc
```

This makes the core portable and keeps provider fallback logic out of the policy
state machine. DRPO V1 reuses the current `discover_machine` implementation.
Provider plugins may be introduced without changing the policy contract.

## 7. State machine

### 7.1 Fresh auto

```text
validate contract
  -> reject existing immutable identity in a new-run directory
  -> resolve adapter/version
  -> snapshot machine
  -> compute workload + scientific fingerprints
  -> inspect compatible measurement cache
  -> adapter.select_auto
  -> validate selection shape and scientific_matrix_changed=false
  -> atomically write selection and identity
  -> return ALLOW
```

### 7.2 Fresh fixed

```text
validate contract
  -> resolve adapter
  -> snapshot machine
  -> adapter.validate_fixed
  -> freeze exact fixed selection
  -> return ALLOW
```

### 7.3 Fresh exempt

```text
validate explicit reason
  -> write exempt selection and identity
  -> return ALLOW_WITHOUT_RESOURCE_SELECTION
```

### 7.4 Resume

```text
load original immutable selection + identity
  -> verify identity digest
  -> verify workload/scientific/adapter fingerprints
  -> snapshot current machine
  -> adapter.revalidate original selection
       -> safe: write attempt-local revalidation record; ALLOW unchanged
       -> unsafe: BLOCK
```

Resume must not:

- recalculate a smaller or larger worker/device count;
- rewrite the original selection;
- treat cache as authority;
- silently change adapter or provider semantics.

A deliberate resource-identity change is a new recovery decision under a separate
project policy, not an ordinary resume.

## 8. Identity model

### 8.1 Immutable selection identity inputs

The SHA-256 identity binds:

- policy schema version and normalized common contract;
- portable core protocol version;
- adapter ID and adapter version;
- workload fingerprint;
- scientific-invariant fingerprint;
- policy profile and adapter parameters;
- machine static identity;
- resolved provider ID/version;
- selected resources;
- stable measurement/fallback evidence fingerprint;
- application-facing selection schema version.

The full repository commit and dynamic load/free-memory snapshot are recorded as
provenance but are not part of the stable selection identity. Including the whole
repository commit would block resume after unrelated documentation changes;
including dynamic load would make every revalidation a different identity.
Projects may add protected implementation hashes to the workload or scientific
fingerprint when stronger binding is required.

### 8.2 Canonicalization

Identity uses UTF-8 JSON with sorted keys, compact separators, finite numbers only,
and no platform-dependent path normalization beyond project-supplied fingerprints.
The canonicalization algorithm is part of the core protocol version.

## 9. Artifacts

V1 keeps the artifact set deliberately small.

### 9.1 Required immutable files

```text
RUNTIME_SELECTION.json
RUNTIME_SELECTION_IDENTITY.json
```

`RUNTIME_SELECTION.json` is the single authoritative document and contains:

```text
schema/core/adapter/provider versions
requested and effective mode
invocation context at creation
normalized policy contract
workload and scientific fingerprints
machine snapshot and static identity
cache/fallback status
measurement evidence
selected resources
limits and limiting factor
known limitations
source provenance
scientific_matrix_changed=false
created_utc
```

`RUNTIME_SELECTION_IDENTITY.json` contains the canonical identity payload hash and
only the minimum fields needed for independent verification.

### 9.2 Conditional resume artifact

```text
RUNTIME_REVALIDATION.json
```

This is written in the current launch/attempt directory, not by mutating the
original selection. It records current dynamic resources, validation result, and
block reasons.

### 9.3 Adapter evidence

Bounded probe logs remain under an adapter-owned directory such as:

```text
_runtime_resource_probe/
```

Large model payloads and checkpoints are not resource-policy artifacts.

## 10. Cache semantics

Cache is evidence reuse, not scheduling authority.

A measurement cache key includes:

- adapter ID/version;
- workload fingerprint;
- scientific-invariant fingerprint;
- policy profile and parameters;
- machine static identity;
- provider ID/version;
- measurement policy fingerprint.

A cache hit may reuse measured per-worker/per-device evidence. The adapter must
still evaluate current CPU load, available memory, cgroup state, GPU visibility,
GPU utilization, and free VRAM before selecting resources.

The engine rejects cache entries that are malformed, identity-incompatible,
unsafe under current capacity, or produced by an unknown adapter version.

## 11. Scientific isolation

Each project supplies a `scientific_invariant_fingerprint`. For DRPO this must
cover all frozen fields relevant to the workload, including data, method matrix,
seeds, coefficients, batch, horizon, optimizer, evaluation, and stopping rules.

The adapter returns only declared runtime resource fields. The integration layer
compares the scientific fingerprint before and after applying the selection. Any
change blocks execution.

V1 does not claim that runtime concurrency can never affect floating-point order
or wall-clock behavior. It claims only that the declared scientific configuration
is unchanged and that the runtime schedule is explicit provenance.

## 12. Integration with RunSpec and formal execution

### 12.1 Shadow period

The RunSpec loader may accept an optional `runtime_resource` block. A standalone
preflight command generates a selection, but existing runners and guards remain
unchanged. E7/E8 results are compared with the current dedicated entrypoints.

### 12.2 Compatibility period

After Stage-2 authorization, new or explicitly migrated RunSpecs require the
contract. Historical RunSpecs without it preserve existing behavior and are marked
`legacy_unmanaged` in validation output; they are not silently assigned `auto`.

### 12.3 Enforcement period

After a separately approved cutover:

- new registered workloads with an adapter require an explicit policy;
- resume requires the original immutable selection;
- unknown adapters or missing required policy block formal launch;
- historical executions retain their recorded path and are not rewritten.

The formal guard calls preflight and consumes `ALLOW`, `ALLOW_WITHOUT_RESOURCE_SELECTION`,
or `BLOCK`. It does not embed E7/E8 selection logic.

## 13. Open-source reuse

V1 directly reuses standards and optional libraries where they reduce code without
controlling project semantics:

- JSON Schema: portable contract documentation and CI validation;
- `psutil`: optional host snapshot provider;
- NVML / `nvidia-ml-py`: optional GPU provider;
- Submitit: future Slurm submission consumer;
- Ray: future distributed execution consumer.

V1 does not adopt Ray or Dask as the policy core because that would require
rewriting existing runners as framework tasks and would couple identity semantics
to a particular scheduler runtime.

## 14. Failure behavior

All failures produce a structured code and human-readable reason. Minimum codes:

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
CACHE_REJECTED
NO_SAFE_CAPACITY
RESUME_CAPACITY_UNSAFE
ADAPTER_FAILURE
ARTIFACT_WRITE_FAILURE
```

No failure may silently fall back from managed policy to an unmanaged launcher.
A project may explicitly choose a registered fixed fallback inside an adapter, and
that fallback must be recorded in the immutable selection.

## 15. Rollback and compatibility

The core is additive through shadow and compatibility phases. Rollback is:

1. stop invoking the policy preflight;
2. keep existing fixed launchers and formal guard behavior;
3. preserve all generated selection/revalidation evidence;
4. revert the integration files without deleting experiment history;
5. do not reinterpret an already started run under a different resource identity.

## 16. Deferred features

The following require separate claims and are not hidden V1 requirements:

- throughput-knee or global throughput optimization;
- online scale-up/down;
- batch or gradient-accumulation autotuning;
- multi-node placement and gang scheduling;
- preemption, migration, or elastic recovery;
- same-GPU multi-process packing;
- mandatory `psutil`, NVML, Submitit, Ray, Dask, Slurm, or Kubernetes;
- cross-project hosted service or central resource database.

## 17. Architecture acceptance criteria

The architecture is acceptable when:

- another project can implement an adapter without importing DRPO;
- E7/E8 can wrap their current validated selectors rather than rewrite them;
- the core has no runner lifecycle responsibility;
- resume cannot invoke auto selection;
- the artifact identity is independently verifiable;
- scientific fields are outside the writable selection surface;
- no mandatory external dependency is introduced;
- the Phase-B core remains under approximately 700 production lines, excluding
  project adapters, tests, schema, and documentation.
