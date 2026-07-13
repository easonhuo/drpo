# Runtime Resource Policy V1 — cross-project packaging decision

**Claim:** `GOV-RUNTIME-RESOURCE-POLICY-INTEGRATION-01`  
**Status:** design candidate; no packaging or dependency change is authorized here.

## Decision

V1 will be **source-portable and contract-portable first**, not published as a
separate package in the initial implementation.

The neutral implementation lives under:

```text
src/runtime_resource_policy/
```

It has no `drpo` import and no mandatory dependency outside the Python standard
library. Another project may vendor that directory together with the public schema
and compatibility tests.

The DRPO repository's existing Python distribution remains named `drpo` and has
research dependencies such as Torch, Gymnasium, Minari, and Matplotlib. Another
project must not be told to install the full `drpo` distribution merely to obtain
the policy core. The package boundary in source code is therefore neutral even
though the first host repository remains DRPO.

## Why not create a standalone distribution immediately

Publishing a second wheel or repository before there is a real second consumer
would add:

- an additional release/version lifecycle;
- dependency and security maintenance;
- duplicate CI and packaging metadata;
- compatibility promises that have not yet been tested outside DRPO;
- synchronization or subtree-release tooling.

None of these is needed to validate the core contract, identity semantics, or E7/E8
integration. Premature extraction would make the system heavier without improving
current function.

## Cross-project boundary in V1

The reusable boundary consists of:

```text
src/runtime_resource_policy/
docs/schemas/runtime_resource_policy_v1.schema.json
portable core tests selected for vendoring
serialized selection/identity artifact schemas
```

Project-specific code remains outside that boundary:

```text
machine providers
adapter registry
workload and scientific fingerprints
RunSpec loader
formal guard integration
runner application logic
```

The public Python API and artifact schema version must be stable enough that a
second project can implement its own integration without copying DRPO adapters.

## Extraction trigger

A standalone lightweight distribution should be created only when at least one of
the following is true:

1. a second repository adopts the core and source vendoring creates real update
   friction;
2. an external team needs an independently versioned dependency;
3. release cadence or ownership differs from DRPO;
4. a Submitit, Slurm, or other generic integration is shared by multiple projects.

At that point the preferred path is a clean subtree extraction or dedicated package
repository, preserving the existing import name and artifact protocol.

## Expected extraction cost

The later extraction is estimated at:

- **production/packaging code:** approximately 40–100 lines;
- **packaging and release tests:** approximately 100–220 lines;
- **engineering effort:** approximately 0.5–1.5 focused engineer-days;
- **calendar risk:** dominated by release automation and first external-consumer
  compatibility testing.

This cost is intentionally deferred and is not included in the Phase-B core line
budget.

## Compatibility rule

The source-portable V1 may not use DRPO-relative resource files, dynamic repository
imports, or package metadata at runtime. Contract and identity behavior must be
fully determined by explicit inputs. Extraction later must not change previously
serialized identity semantics without a protocol-version bump.

## Rejected alternatives

### Install the full DRPO wheel in other projects

Rejected because it pulls unnecessary scientific dependencies and couples unrelated
projects to DRPO's release lifecycle.

### Publish a separate package before Phase-C acceptance

Rejected as premature. The API has not yet been exercised by the two validated
adapters through the new core.

### Keep the core under `src/drpo/`

Rejected because project imports would gradually leak into the reusable layer and
make later extraction harder.

## Acceptance

This decision is satisfied when Phase B demonstrates that the neutral directory can
be copied into an isolated test environment, imported without `drpo` or scientific
dependencies, and passes its portable contract, identity, and state-machine tests.
