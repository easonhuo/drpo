# GOV-RUNTIME-RESOURCE-POLICY-INTEGRATION-01 — design scope

**Approval:** user explicitly authorized design on 2026-07-13.  
**Design base:** `448936a7d61cb8871457078ef196e371fbce380c`.  
**Phase:** design-only.  
**Scientific experiment impact:** none.  
**Execution-policy impact:** none in this phase.  
**Stage-2 status:** not reopened by this document.

## Goal

Design a small, versioned runtime-resource policy layer that is usable across
projects while preserving DRPO's scientific and execution-governance guarantees.
The design must reuse the already validated E7/E8 resource-probe implementation,
keep workload-specific logic behind thin adapters, and avoid becoming a cluster
scheduler.

The target is a portable control plane for this sequence:

```text
policy contract
  -> machine snapshot
  -> workload adapter
  -> resource decision
  -> immutable selection identity
  -> existing execution backend
```

## Design principles

1. **Generic core, project-owned adapters.** The core may not import E7, E8,
   Hopper, Countdown, DRPO scientific configs, or experiment registries.
2. **Small dependency surface.** The reference implementation uses the Python
   standard library and current procfs/cgroup/`nvidia-smi` providers. `psutil`,
   NVML, Submitit, and Ray are optional integrations, not mandatory dependencies.
3. **Preflight, not orchestration.** The layer decides and records resources before
   launch; it does not replace Slurm, Kubernetes, Ray, the existing runner, or the
   hardened formal channel.
4. **Immutable resume identity.** A resume revalidates the original selection and
   either uses it unchanged or blocks. It never silently recalculates concurrency.
5. **Scientific isolation.** The policy layer may change only declared runtime
   resource fields. It may not modify data, seeds, methods, coefficients, batch,
   horizon, optimization, evaluation, stopping, or result status.
6. **Fail closed at the governance boundary.** Unknown adapters, malformed
   contracts, resource-identity drift, or unsafe resume capacity block execution.
7. **Incremental cutover.** New behavior is shadowed and compared against the
   already validated E7/E8 paths before any formal-channel or default-policy
   integration.
8. **Portable artifacts.** The serialized contract and decision artifacts are
   project-neutral and stable enough for reuse by another repository.

## Authorized design files

- `docs/scopes/GOV-RUNTIME-RESOURCE-POLICY-INTEGRATION-01.md`
- `docs/runtime_resource_policy_architecture_v1.md`
- `docs/runtime_resource_policy_implementation_plan_v1.md`
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
- throughput-knee search, online resizing, multi-node scheduling, preemption,
  migration, or automatic scientific-parameter changes.

## Required implementation phases after design approval

### Phase B — portable core, no execution integration

Implement schema validation, state transitions, adapter/provider/backend protocols,
identity generation, cache validation, and a standalone `plan` command. Existing
launchers remain unchanged.

### Phase C — E7/E8 shadow integration

Route the already validated E7/E8 adapters through the portable core in plan-only
mode and compare decisions/artifacts against the current dedicated paths. Do not
launch full experiments.

### Phase D — Stage-2 integration and controlled cutover

Only after a separate explicit user approval and accepted authorization record,
integrate the resource preflight into the formal channel. Introduce compatibility
and enforcement windows rather than changing all historical RunSpecs at once.

### Phase E — optional execution backends and throughput policy

Add Submitit/Slurm, Ray, or throughput-aware search only under separate claims.
These are not prerequisites for the core policy layer.

## Acceptance for this design phase

- architecture names a minimal stable core and explicit extension boundaries;
- schema covers `auto`, `fixed`, and `exempt` without embedding workload-specific
  fields in the core;
- fresh, resume, cache, fallback, and unsafe-capacity transitions are specified;
- serialized artifacts and their identity inputs are defined;
- cross-project integration path is concrete;
- Stage-2 reopen boundaries and rollback are explicit;
- implementation line-count and engineering-cost estimates are included with
  uncertainty ranges;
- no current runtime or formal behavior changes.

## Rollback

Close the design PR without merging, or revert only the authorized design files.
No executable behavior, scientific configuration, result, or experiment directory
is affected by this design-only phase.
