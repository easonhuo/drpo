# GOV-RUNTIME-RESOURCE-POLICY-INTEGRATION-01 — design scope

**Approval:** user explicitly authorized design on 2026-07-13.  
**Design base:** `448936a7d61cb8871457078ef196e371fbce380c`.  
**Phase:** design-only.  
**Scientific experiment impact:** none.  
**Execution-policy impact:** none in this phase.  
**Stage-2 status:** not reopened by this document.

## Goal

Design a small, versioned runtime-resource policy layer that is reusable across
projects while preserving DRPO's scientific and execution-governance guarantees.
The design must reuse the validated E7/E8 selectors, keep workload-specific logic in
thin adapters, and avoid becoming a scheduler or provider framework.

The refined boundary is:

```text
project integration supplies explicit data and adapter
  -> portable create/verify/revalidate core
  -> one immutable RUNTIME_SELECTION.json
  -> existing runner or formal guard
```

## Design principles

1. **Generic core, project-owned adapters.** The core may not import E7, E8,
   Hopper, Countdown, DRPO scientific configs, providers, registries, or runners.
2. **Semantics rather than orchestration.** The core validates policy, freezes
   identity, and revalidates resume. It does not launch, schedule, retry, or monitor.
3. **Two state-changing operations.** Creation and revalidation are the only core
   transitions. `plan` is a project CLI command, not a third state.
4. **One creation authority.** Selection and its digest live in one immutable
   `RUNTIME_SELECTION.json`; there is no identity sidecar.
5. **Independent scientific identity.** The project supplies workload and scientific
   fingerprints separately from the resource adapter.
6. **Adapter-owned evidence.** Probes, measurement cache, fallback, provider logic,
   and resource arithmetic remain in adapters/integration rather than being
   duplicated by the core.
7. **Minimal stable identity.** Dynamic capacity, cache route, and measurement logs
   are provenance, not identity. Required host/device affinity is explicit through
   adapter-defined `resource_binding`.
8. **Immutable resume.** Resume uses the original selected resources unchanged or
   blocks; it never invokes auto selection.
9. **Small dependency surface.** The core is Python-standard-library only. Optional
   system libraries and scheduler consumers remain outside it.
10. **Incremental cutover.** E7/E8 plan-only shadow equivalence precedes any formal
    integration or default-policy change.

## Five-pass design review record

The design was re-audited independently for:

1. **responsibility cohesion:** removed provider/backend protocols and the redundant
   plan state from the core;
2. **identity minimality:** removed machine observations and measurement route from
   stable identity, retaining explicit resource binding only when required;
3. **artifact authority:** embedded the digest in the single selection document and
   removed the identity sidecar;
4. **cache and fallback ownership:** kept both in existing adapters to avoid a second
   generic cache system;
5. **integration and cost:** reduced the portable core target from 500–650 to
   400–500 production lines and tightened the hard ceiling to 550.

These refinements reduce code and state without weakening fail-closed behavior,
provenance, scientific isolation, or resume safety.

## Authorized design files

- `docs/scopes/GOV-RUNTIME-RESOURCE-POLICY-INTEGRATION-01.md`
- `docs/runtime_resource_policy_architecture_v1.md`
- `docs/runtime_resource_policy_implementation_plan_v1.md`
- `docs/runtime_resource_policy_packaging_decision_v1.md`
- `docs/schemas/runtime_resource_policy_v1.schema.json`
- `docs/governance_stage_authorizations/GOV-RUNTIME-RESOURCE-POLICY-INTEGRATION-01-STAGE2-REOPEN-DRAFT.md`

## Explicitly excluded from this phase

- Python implementation or modification of current resource code;
- changes to `docs/handoff.md` or `experiments/registry.yaml`;
- changes to RunSpec validators, agents, formal guards, packagers, or runners;
- a Stage-2 reopen or any default-policy change;
- migration of existing RunSpecs;
- execution of E7, E8, Hopper, Countdown, or any scientific experiment;
- adding mandatory third-party dependencies;
- provider or execution-backend frameworks in the portable core;
- generic cache services, throughput-knee search, online resizing, multi-node
  scheduling, preemption, migration, or scientific-parameter changes.

## Required implementation phases after design approval

### Phase B — portable core, no execution integration

Implement strict contract normalization, create/verify/revalidate operations,
caller-supplied adapter lookup, embedded identity, common output validation, atomic
artifacts, and a DRPO `plan` CLI. Machine discovery, cache, and selection arithmetic
remain project integration code. Existing launchers remain unchanged.

### Phase C — E7/E8 shadow integration

Wrap the validated E7/E8 selectors and compare selected resources, constraints, and
one-file artifacts against the dedicated paths in plan-only mode. Do not launch full
experiments.

### Phase D — Stage-2 integration and controlled cutover

Only after a separate explicit user approval and accepted authorization record,
integrate verified preflight results into the formal channel. Use compatibility and
enforcement windows rather than rewriting historical RunSpecs.

### Phase E — optional consumers and throughput policy

Submitit/Slurm, Ray, optional providers, and throughput-aware search require separate
claims. They are not prerequisites for the V1 core.

## Acceptance for this design phase

- architecture defines a minimal stable core and explicit ownership boundaries;
- schema covers `auto`, `fixed`, and `exempt` without workload fields;
- creation and resume transitions are deterministic and fail closed;
- one authoritative selection artifact and its identity payload are defined;
- cache, fallback, provider, and scheduler responsibilities are non-duplicative;
- cross-project source boundary is concrete;
- Stage-2 reopen and rollback boundaries are explicit;
- refined line-count and engineering-cost estimates include uncertainty ranges;
- no current runtime, formal, scientific, or experiment behavior changes.

## Rollback

Close the design PR without merging, or revert only the authorized design files.
No executable behavior, scientific configuration, result, or experiment directory
is affected by this design-only phase.