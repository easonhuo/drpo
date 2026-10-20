# DRAFT — Stage-2 reopen authorization for runtime-resource preflight

**Claim:** `GOV-RUNTIME-RESOURCE-POLICY-INTEGRATION-01`  
**Status:** draft only; this file does not reopen Stage 2.  
**Activation condition:** later explicit user approval after Phase-B core and Phase-C
E7/E8 shadow acceptance.

## Proposed authorization

Temporarily reopen the closed-maintenance-only formal experiment channel only to add
a narrow verification hook for managed runtime-resource selections.

```text
formal launch request
  -> existing provenance and safety checks
  -> project runtime-resource preflight
  -> verify RUNTIME_SELECTION.json or RUNTIME_REVALIDATION.json
  -> ALLOW / ALLOW_WITHOUT_RESOURCE_SELECTION / BLOCK
  -> existing guarded runner lifecycle
```

The formal guard may call a project integration function that creates or revalidates
the selection, but it must not contain machine-provider logic, probes, cache policy,
fallback arithmetic, E7/E8 special cases, or scheduler behavior.

## Preconditions

All must hold before this draft can become an active authorization:

1. portable create/verify/revalidate core merged and independently tested;
2. E7 CPU and E8 GPU shadow decisions accepted on real hardware;
3. one-file embedded identity and resume revalidation verified;
4. dedicated wrappers remain available as rollback paths;
5. historical RunSpec compatibility behavior reviewed;
6. rollback rehearsal passes in a clean checkout;
7. user explicitly approves the active reopen scope.

A separate later approval is still required for default enforcement.

## Proposed authorized files

The active authorization should use the smallest confirmed set after implementation
review. Expected upper bound:

- `scripts/run_experiment_guard_hardened.py`
- `scripts/validate_formal_execution_channel.py`
- `docs/formal_experiment_artifact_protocol.md`
- one runtime-policy/formal integration module under `src/drpo/`
- focused formal-channel and RunSpec tests
- `docs/governance_pipeline_stage_status.yaml` only to record authorization and
  refreshed protected hashes after acceptance

The packager, verifier, hardened artifact core, handoff authority, experiment
registry, scientific runners, machine providers, and adapters are excluded unless a
separate reviewed amendment proves they are required.

## Allowed changes

- invoke the project preflight before a managed runner starts;
- verify the embedded identity in `RUNTIME_SELECTION.json`;
- require attempt-local `RUNTIME_REVALIDATION.json` for managed resume;
- record managed, exempt, compatibility, and blocked status in launch provenance;
- add compatibility and anti-bypass tests;
- update protocol documentation and protected hashes after acceptance.

## Prohibited changes

- scientific config, methods, seeds, data, coefficients, batch, horizon, optimizer,
  evaluation, stopping, or result status;
- machine discovery, probe execution, measurement cache, or resource-selection
  arithmetic inside the formal guard;
- automatic batch or gradient-accumulation changes;
- dynamic resizing after runner launch;
- changing an existing resume selection;
- silently falling back from managed policy to an unmanaged launcher;
- bulk rewriting historical RunSpecs;
- replacing the guard with Ray, Dask, Submitit, Slurm, or another scheduler;
- adding a second identity or selection authority;
- weakening clean-worktree, provenance, heartbeat, packaging, terminal-audit, or
  anti-bypass requirements.

## Compatibility and cutover

### Compatibility window

- historical RunSpecs without `runtime_resource` preserve approved behavior;
- validation records them as `legacy_unmanaged`;
- new or explicitly migrated RunSpecs may opt into managed policy;
- no repository-wide default changes in the first compatibility release.

### Enforcement window

A separate explicit cutover approval is required to enforce policy for new formal
RunSpecs. Enforcement may not reinterpret completed or historical runs.

## Acceptance

- managed fresh runs cannot start without a valid one-file selection;
- the guard recomputes and verifies the embedded digest;
- managed resume never invokes auto reselection and cannot mutate resources;
- unsafe resume blocks with a structured reason;
- unknown adapters, version mismatch, and fingerprint drift block;
- exempt runs produce auditable evidence without adapter fields;
- historical compatibility matches the approved matrix;
- anti-bypass tests cover direct and agent-mediated formal entrypoints;
- the guard contains no provider, probe, cache, or workload arithmetic;
- full repository, governance, formal-channel, and protected-hash gates pass;
- one real managed fresh shadow launch and one resume preflight are accepted before
  enforcement;
- engineering acceptance is not reported as a scientific experiment result.

## Rollback

1. disable the formal preflight hook through the approved compatibility switch;
2. restore protected formal-channel files from the pre-reopen commit;
3. continue using existing fixed/dedicated launchers;
4. preserve all selection, revalidation, launch, failure, and experiment artifacts;
5. do not resume a run under a different resource identity;
6. re-run formal-channel, governance, and protected-hash validation;
7. close the reopen authorization with a rollback record.

## Non-activation statement

Merging this draft as design documentation does not authorize implementation,
reopen Stage 2, modify a protected file, or change default execution policy.