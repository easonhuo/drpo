# GOV-RUNSPEC-RESOURCE-ORCHESTRATOR-01 — unified RunSpec resource boundary

**Approval:** user explicitly authorized implementation on 2026-07-17.  
**Base:** `main@4544005bd7df69c53bad70a9dcac846af01285e4`.  
**Dependencies:** `GOV-RUNTIME-RESOURCE-POOL-01`, `GOV-RUNTIME-RESOURCE-AUTOTUNE-CPU-V2-01`, `EXT-H-E7-DEFERRED-CAPACITY-ADMISSION-01`.  
**Scientific impact:** none.  
**Experiment execution:** none in this change.

## Problem

The governed RunSpec command and the existing resource tools are separate. An operator
currently has to know which experiment entrypoint already performs capacity handling. The
resource boundary can therefore be omitted even when the RunSpec itself is correct.

## Objective

Extend the existing RunSpec execution path so one foreground command can:

1. claim the selected RunSpec;
2. apply a declared Linux CPU-affinity pool before the experiment entrypoint starts;
3. measure available CPU capacity inside that affinity and visible cgroup quota domains;
4. wait until a declared minimum capacity is observed;
5. export an optional maximum-worker ceiling for compatible entrypoints;
6. continue through the existing execution, recovery, packaging, and delivery lifecycle.

## Operator interface

```text
--cpu-pool <linux-cpu-list>
--resource-cpu-fraction <0,1]
--minimum-available-cpu-cores <positive number>
--resource-wait-timeout-seconds <negative for unlimited>
--resource-poll-seconds <positive number>
--resource-sample-seconds <positive number>
--max-workers <positive integer, optional>
```

Omitting both `--cpu-pool` and `--max-workers` preserves the current path. These values are
runtime placement policy rather than scientific hyperparameters. They are recorded below
the local RunSpec log directory and are not inserted into the tracked READY specification.

## Capacity semantics

The outer measurement answers whether the selected CPU pool currently has the declared
minimum available capacity. It reuses the existing affinity/cgroup counters. It does not
infer an arbitrary program's per-worker CPU or memory demand.

`--max-workers` is exported as `DRPO_RUNTIME_MAX_WORKERS`. Compatible entrypoints may use it
as a ceiling. Existing workload-specific auto entrypoints may still perform their detailed
representative-worker measurement after inheriting the restricted pool. Fixed or sequential
liveness entrypoints may ignore this value because they have no parallel width to select.

## Implemented code surface

Existing files only:

- `scripts/agent/run_lane.py` exposes the shared command-line options;
- `scripts/agent/run_claimed_runspec.py` validates and records the request, activates the
  existing pool, measures pool-local capacity, delegates waiting to the existing foreground
  wait function, exports the worker ceiling, and binds the report to the RunSpec lifecycle;
- `tests/test_agent_runspec.py` covers compatibility, validation, identity reuse, delegation,
  and environment inheritance;
- `runspecs/README.md` documents the operator command.

No new Python file or replacement resource/RunSpec framework is introduced.

## Fail-closed rules

- requested CPU IDs must belong to inherited affinity;
- effective affinity must exactly equal the requested pool;
- resource values must be finite and within their declared ranges;
- an impossible minimum capacity is rejected immediately;
- missing or contradictory CPU/cgroup evidence is fatal;
- a changed request or pool identity for the same run ID is rejected;
- waiting remains in the foreground;
- running workers are not resized after entrypoint launch.

## Explicit exclusions

- automatic NUMA/socket CPU-ID selection;
- changing unrelated workload placement;
- generic inference of per-worker memory or CPU demand;
- rewriting experiment commands to insert workload-specific flags;
- changing scientific configuration, data, seeds, methods, optimizer, horizon, or evaluation;
- changing `docs/handoff.md` or `experiments/registry.yaml`;
- launching an experiment or authorizing merge.

## Acceptance

1. Existing execution without resource options remains compatible.
2. One `run_lane.py --once` command applies the pool before entrypoint execution.
3. Measurements are limited to the effective affinity and visible quota domains.
4. Wait and identity evidence are written below the run log directory.
5. Invalid values, impossible floors, and conflicting identities fail closed.
6. Recovery attempts inherit the same process affinity and worker environment.
7. `DRPO_RUNTIME_MAX_WORKERS` reaches the entrypoint without command rewriting.
8. No duplicate resource or RunSpec subsystem is introduced.
9. Focused tests, compilation, Ruff, full pytest, and governance checks pass on the exact PR
   head.

## Rollback

Revert the implementation commits. The previous RunSpec command and standalone resource
wrapper remain available. Runtime resource records are engineering provenance, not
scientific evidence.
