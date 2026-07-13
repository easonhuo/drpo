# DRAFT — Stage-2 reopen authorization for runtime-resource preflight

**Claim:** `GOV-RUNTIME-RESOURCE-POLICY-INTEGRATION-01`  
**Status:** draft only; this file does not reopen Stage 2.  
**Activation condition:** a later explicit user approval after Phase-B core and
Phase-C E7/E8 shadow acceptance.

## Proposed authorization

Temporarily reopen the closed-maintenance-only canonical formal experiment channel
for the narrow purpose of adding a project-neutral runtime-resource preflight hook.
The hook may consume an already validated immutable resource selection or invoke
the approved policy engine before the scientific runner starts.

The proposed responsibility change is limited to:

```text
formal launch request
  -> existing provenance and safety checks
  -> runtime-resource preflight
  -> ALLOW / ALLOW_WITHOUT_RESOURCE_SELECTION / BLOCK
  -> existing guarded runner lifecycle
```

The formal channel does not gain responsibility for measuring workload resources,
choosing workload-specific resource values, scheduling tasks, or recovering an
unsafe resume. Those remain in the portable policy core and project adapters.

## Preconditions

All must be satisfied before this draft can be promoted to an active authorization:

1. portable core merged and tested independently of the formal channel;
2. E7 CPU and E8 GPU shadow decisions accepted on real hardware;
3. immutable selection and resume revalidation artifacts verified;
4. dedicated wrappers remain available as rollback paths;
5. compatibility behavior for historical RunSpecs reviewed;
6. rollback rehearsal passes in a clean checkout;
7. user explicitly approves the active Stage-2 reopen and cutover scope.

## Proposed authorized files

The active authorization should use the smallest confirmed set after implementation
review. The expected upper bound is:

- `scripts/run_experiment_guard_hardened.py`
- `scripts/validate_formal_execution_channel.py`
- `docs/formal_experiment_artifact_protocol.md`
- one runtime-policy/formal integration module under `src/drpo/`
- focused formal-channel and RunSpec tests
- `docs/governance_pipeline_stage_status.yaml` only to record authorization and
  refreshed protected hashes after acceptance

The packager, verifier, hardened artifact core, handoff authority, experiment
registry, and scientific runners are excluded unless a separate reviewed amendment
proves they are necessary.

## Allowed changes

- invoke the policy preflight before runner launch;
- verify `RUNTIME_SELECTION.json` and its immutable identity;
- require attempt-local resume revalidation for managed runs;
- record managed, exempt, compatibility, and blocked status in launch provenance;
- add compatibility-window validation and anti-bypass tests;
- update protocol documentation and protected hashes after acceptance.

## Prohibited changes

- scientific config, methods, seeds, data, coefficients, batch, horizon, optimizer,
  evaluation, stopping, or result status;
- automatic batch or gradient-accumulation changes;
- dynamic resizing after runner launch;
- changing an existing resume selection;
- silently falling back from managed policy to an unmanaged launcher;
- bulk rewriting historical RunSpecs;
- replacing the formal guard with Ray, Dask, Submitit, Slurm, or another scheduler;
- weakening clean-worktree, provenance, heartbeat, packaging, terminal-audit, or
  anti-bypass requirements.

## Compatibility and cutover

### Compatibility window

- historical RunSpecs without `runtime_resource` preserve existing behavior;
- their validation output records `legacy_unmanaged`;
- new or explicitly migrated RunSpecs may opt into managed policy;
- no repository-wide default changes in the first compatibility release.

### Enforcement window

A separate explicit cutover approval is required to enforce policy for new formal
RunSpecs. Enforcement may not reinterpret or invalidate historical completed runs.

## Acceptance

- managed fresh runs cannot start without a valid immutable selection;
- managed resume never invokes automatic reselection;
- unsafe resume blocks with a structured reason;
- unknown adapters and identity drift block;
- exempt runs produce auditable evidence;
- historical compatibility behavior matches the approved matrix;
- anti-bypass tests cover direct and agent-mediated formal entrypoints;
- full repository, governance, formal-channel, and protected-hash gates pass;
- one real managed fresh shadow launch and one resume preflight are accepted before
  enforcement;
- no scientific experiment result is claimed from engineering acceptance.

## Rollback

1. disable the formal preflight hook through the approved compatibility switch;
2. restore the previous protected formal-channel files from the pre-reopen commit;
3. continue using existing fixed/dedicated launchers;
4. preserve all selection, revalidation, launch, failure, and experiment artifacts;
5. do not resume a run under a different resource identity;
6. re-run formal-channel, governance, and protected-hash validation;
7. close the reopen authorization with a rollback record.

## Non-activation statement

Merging this draft as design documentation does not authorize implementation,
reopen Stage 2, modify a protected file, or change default execution policy.
