# Runtime Resource Policy V1 — cross-project packaging decision

**Claim:** `GOV-RUNTIME-RESOURCE-POLICY-INTEGRATION-01`  
**Status:** refined design candidate; no packaging or dependency change is authorized
here.

## Decision

V1 is **source-portable and contract-portable first**. It is not published as a
second distribution or repository during the initial implementation.

The neutral source lives under:

```text
src/runtime_resource_policy/
```

It has no `drpo` import and no mandatory dependency outside the Python standard
library. Another project may vendor that directory together with the public
contract schema, versioned artifact-format documentation, and portable compatibility
tests.

The existing `drpo` distribution includes Torch, Gymnasium, Minari, Matplotlib, and
other research dependencies. Cross-project consumers must not install the full DRPO
package merely to use the small policy core.

## Why extraction is deferred

Creating a standalone wheel or repository before a second real consumer exists
would add release versioning, security maintenance, duplicate CI, synchronization,
and compatibility promises that have not yet been exercised outside DRPO.

None of those costs improves the V1 contract, identity, resume, or E7/E8 shadow
validation. Premature extraction would make the system heavier without increasing
functionality.

## Reusable boundary

The vendorable V1 boundary is:

```text
src/runtime_resource_policy/
docs/schemas/runtime_resource_policy_v1.schema.json
versioned RUNTIME_SELECTION.json and RUNTIME_REVALIDATION.json format documentation
portable contract / identity / engine tests
```

Project-specific code remains outside:

```text
machine discovery and provider fallback
adapter mapping and adapter context
workload and scientific fingerprints
measurement cache, probes, and fallback policy
RunSpec loader and CLI
formal guard integration
runner or scheduler application logic
```

The public Python API is deliberately limited to creation, verification, and
revalidation. There is no public provider, backend, cache, or plugin framework to
extract later.

## Artifact portability

`RUNTIME_SELECTION.json` is the sole immutable creation authority and embeds its
SHA-256 identity metadata. A separate identity sidecar is not part of the public
format.

Portable identity binds normalized policy, core/schema version, resolved adapter,
workload/scientific fingerprints, selected resources, and explicit resource binding.
Machine observations, provider details, probe evidence, cache/fallback route,
timestamps, and repository commit remain recorded provenance.

This distinction lets another project reproduce identity semantics without copying
DRPO machine or cache implementations.

## Extraction trigger

A standalone lightweight distribution should be created only after at least one
condition becomes real:

1. a second repository adopts the core and source vendoring creates update friction;
2. an external team needs independent versioning or ownership;
3. release cadence diverges from DRPO;
4. multiple projects share a generic integration consumer.

A future Submitit, Slurm, or Ray consumer alone does not require moving scheduling
into the core; it may remain a separate integration package.

## Expected later extraction cost

Because V1 forbids DRPO imports and repository-relative resources, later extraction
is estimated at:

- **production/packaging code:** approximately 30–80 lines;
- **packaging and release tests:** approximately 80–180 lines;
- **engineering effort:** approximately 0.5–1 focused engineer-day;
- **calendar risk:** first external-consumer and release-automation validation.

This deferred work is not included in the Phase-B line budget.

## Compatibility rules

- all core behavior is determined by explicit inputs;
- no runtime access to DRPO package metadata or repository-relative files;
- canonicalization and serialized identity semantics are protocol-versioned;
- extraction may not change old selection digests without a version bump;
- adapters remain project integrations rather than being bundled into the neutral
  package;
- source vendoring must preserve the exact portable tests for the adopted protocol
  version.

## Rejected alternatives

### Install the full DRPO wheel elsewhere

Rejected because it couples unrelated projects to scientific dependencies and DRPO's
release lifecycle.

### Publish a separate package before Phase-C acceptance

Rejected because the API has not yet been exercised by both validated adapters
through the refined core.

### Keep the reusable core under `src/drpo/`

Rejected because project imports would gradually leak into identity and state
semantics, making clean extraction harder.

### Publish a generic scheduler/provider framework

Rejected because V1 consumes plain machine observations and leaves execution to the
existing project. A framework would increase surface area without helping the core
contract.

## Acceptance

Phase B satisfies this decision only when the neutral directory can be copied into
an isolated environment, imported without `drpo` or scientific dependencies, and
passes portable contract, identity, create, verify, and revalidate tests.