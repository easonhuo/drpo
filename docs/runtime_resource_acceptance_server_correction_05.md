# Runtime-resource acceptance server correction 05

## Identity and authority

- Claim: `GOV-RUNTIME-RESOURCE-ACCEPTANCE-HARNESS-01`.
- User clarification: 2026-07-16.
- Pre-correction harness head:
  `3385e9fb4d6f1b69bad609c4f555858475611740`.
- Scientific impact: none.
- Experiment-status impact: none.
- Scientific variables, datasets, seeds, steps, thresholds, model settings, and
  evaluation protocols: unchanged.

## Corrected problem statement

The target server is a shared host. ResearchBench, AIDE, and joblib/loky workers are
permanent external workloads. The acceptance goal is not to guarantee exclusive CPUs
for E7 or E8 and is not to prove that external processes cannot run on the same CPUs.

The actual engineering goal is narrower:

1. impose an explicit hard upper bound on CPU/GPU resources that DRPO-owned processes
   may use;
2. measure the currently available CPU, RAM, GPU, and VRAM capacity inside those bounds;
3. select only a safe worker or slot count below the configured caps;
4. revalidate before the bounded liveness launch;
5. stop only DRPO-owned process groups and audit cleanup.

A configured CPU pool is therefore a containment ceiling on DRPO-owned processes, not a
reservation or exclusivity claim.

## Superseded readiness assumptions

The following conditions are not required by the shared-host route:

- ResearchBench, AIDE, or joblib/loky process counts reaching zero;
- a cgroup v2 cpuset partition;
- exclusive ownership of the E7/E8 CPU-pool union;
- proof that external workloads have disjoint CPU affinity;
- a fixed guarantee of 192 E7 CPUs or 144 E8 CPUs.

Correction 04 remains immutable historical provenance for the exclusive-partition
alternative, but its partition requirement is superseded as the target-server default.
The partitioned entrypoint remains an optional diagnostic for a compatible cgroup v2
host and is not a prerequisite for shared-host acceptance.

## Shared-host capacity contract

The default one-click route now uses the following contract.

### Hard limits

- Every E7-owned process inherits exactly the configured E7 CPU affinity.
- Every E8-owned process inherits exactly the configured E8 CPU affinity.
- E7 and E8 CPU pools must be non-overlapping.
- E7 worker count remains capped by `e7.max_workers` and the configured growth bound.
- E8 placement remains capped by `e8.max_devices` and `e8.max_slots_per_gpu`.
- Existing BLAS/OpenMP/PyTorch thread environments remain part of the reviewed workload
  inputs or bounded thread-envelope checks.

These limits prevent a DRPO launcher from escaping to every CPU visible to the server.
They do not prevent unrelated workloads from sharing the declared CPUs.

### Dynamic capacity selection

- E7 planning executes inside the E7 affinity pool. CPU accounting reads only the CPUs
  in that active affinity and combines measured external occupancy, measured
  per-worker demand, RAM headroom, and configured caps.
- E8 selection executes inside the E8 affinity pool and combines measured CPU/RAM,
  GPU/VRAM envelopes, GPU utilization, and configured slot caps.
- Existing revalidation remains fail-closed. If the previously selected count is no
  longer safe, the launch is blocked rather than oversubscribed.
- Safe-capacity shortage is `BLOCKED`, not a code failure and not a request to kill
  permanent workloads.

### External-process observation

Configured process-name patterns remain useful provenance. Stage 0 records matching
external workloads, but their presence alone no longer blocks the run. Their real load
is reflected by pool-local capacity measurement in later stages.

True Stage-0 blockers remain:

- wrong or dirty checkout;
- missing external inputs;
- requested CPUs outside inherited affinity;
- malformed or overlapping resource pools;
- other identity or topology errors.

## Default operator route

The unchanged operator command is:

```bash
bash scripts/run_runtime_resource_acceptance_one_click.sh \
  --profile /absolute/path/runtime_resource_acceptance_server.json
```

The one-click shell now dispatches to the shared-host runner. No process-count-zero gate
and no partition gate may be added by the operator prompt.

## Status and evidence boundary

- `PASS`: required engineering stages completed under the configured hard limits.
- `INCONCLUSIVE`: only a single worker/slot could be exercised for a multi-worker claim.
- `BLOCKED`: current safe measured capacity is unavailable or a prerequisite is absent.
- `FAIL`: identity, implementation, process supervision, cleanup, OOM, or numerical
  failure.

A `BLOCKED` capacity result is a terminal environment observation for that run. It must
not be converted into repeated code changes or a claim that the scientific experiment
is impossible.

All output remains engineering evidence only. It cannot establish task performance,
method ranking, convergence, steady state, controlled mechanism identification, or OOD
generalization.
