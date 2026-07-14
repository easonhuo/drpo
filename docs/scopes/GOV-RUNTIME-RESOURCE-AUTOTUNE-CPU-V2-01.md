# GOV-RUNTIME-RESOURCE-AUTOTUNE-CPU-V2-01 — measured CPU capacity maintenance scope

**Approval:** user explicitly authorized implementation after approving and merging the design ledger on 2026-07-14.  
**Implementation base:** `f387391b39502e700e7f780cd8b4a1fd9c7eca7c`.  
**Design authority:** `docs/runtime_resource_autotune_evolution.md`.  
**Scientific experiment impact:** none.  
**Default-policy impact:** none; existing fixed launchers remain available and unchanged.

## Objective

Replace the E7 CPU autotune path's raw-load-average capacity arithmetic with the
measured-CPU V2 contract frozen in the design ledger. The implementation must also
eliminate plan/run self-feedback: `plan` is the only operation that may probe and
select a worker count, while `run` verifies and consumes the immutable selection or
fails closed.

The maintenance changes only the active independent subprocess count. It must not
change any scientific branch, method, seed, coefficient, optimizer, batch, thread
environment, horizon, stopping rule, evaluation protocol, or result status.

## Authorized production surface

- `src/drpo/runtime_cpu_capacity.py` — one cohesive Linux CPU observation and pure
  capacity-arithmetic module;
- `src/drpo/runtime_resource_autotune.py` — shared machine/resource selection and
  artifact integration;
- `src/drpo/runtime_resource_adapters.py` — E7 adapter wiring and revalidation;
- `src/drpo/e7_ppo_w0_runtime_autotune.py` — shared PPO-family probe, throughput, and
  plan/run behavior;
- existing thin E7 auto wrappers that delegate to the shared selector, limited to
  identity/version plumbing and selection consumption;
- `scripts/probe_runtime_resources.py` when required to expose the new diagnostic
  observation fields;
- `docs/runtime_resource_autotuning_v1.md` or a versioned successor usage document;
- `docs/runtime_resource_autotune_evolution.md`, append-only history plus current
  status update;
- this scope file;
- focused tests for the shared CPU model, adapters, affected auto wrappers, and
  lifecycle semantics.

Before modifying a thin wrapper, the implementation must confirm from code that it
actually delegates to the shared E7 autotune path. No unrelated runner may be edited
for speculative consistency.

## Required implementation

### CPU observation

- bind capacity to the exact `sched_getaffinity` CPU set;
- resolve the current cgroup from `/proc/self/cgroup`;
- discover every finite CPU quota domain from the current cgroup through the
  controller mount root;
- support cgroup v2 `cpu.max` and cgroup v1 quota/period files when visible;
- sample affinity-scoped `/proc/stat` execution occupancy over aligned monotonic
  intervals;
- sample CPU usage for every finite quota domain over the same interval;
- record load averages as diagnostic provenance only;
- fail closed on malformed quota/controller evidence, changing affinity, missing CPU
  rows, or non-positive tick/usage deltas.

### Worker demand and capacity

- measure representative process-tree CPU seconds and peak RSS in the same bounded
  resource probe;
- calculate reserved CPU demand using an explicit safety factor and minimum floor;
- apply affinity and every finite quota domain as independent constraints;
- retain the existing host/cgroup memory gate and bounded growth cap;
- retain existing workload-specific throughput candidate search rather than adding a
  new generic throughput engine;
- require every candidate to satisfy CPU, memory, completion, timeout, and cleanup
  gates before it may be selected.

### Plan/run lifecycle

- `plan` owns all automatic probing, candidate benchmarking, and creation of one
  immutable `RUNTIME_SELECTION.json`;
- an existing valid selection is not silently replanned in the same work directory;
- `run` never invokes automatic selection, representative probes, or throughput
  candidates;
- `run` verifies source/workload/scientific/adapter/policy identity and the exact
  selected worker count;
- `run` confirms no conflicting worker from the work directory is alive;
- `run` performs three consecutive one-second CPU/domain samples plus current memory
  validation, using the most conservative observed pressure for each constraint;
- `run` either starts the exact planned worker count or writes a block record and
  returns `RUNTIME_CAPACITY_CHANGED_REPLAN_REQUIRED`;
- silent downshift from a planned value such as 112 to 80, 20, or 1 is forbidden.

### Artifacts and compatibility

- introduce an explicit measured-CPU selector policy version and incompatible
  selection schema version;
- invalidate raw-load-average selections and caches;
- keep stable selection identity separate from dynamic machine evidence;
- write attempt-local
  `_runtime_resource_attempts/<attempt_id>/RUNTIME_REVALIDATION.json` without mutating
  the authoritative selection;
- preserve failed probes, candidate logs, blocked revalidation evidence, and process
  cleanup evidence;
- keep fixed launchers and prior scientific artifacts unchanged.

## Explicitly excluded

- modifying `docs/handoff.md` or `experiments/registry.yaml`;
- modifying the closed formal execution channel or RunSpec governance;
- changing any scientific variable, matrix, seed, threshold, horizon, evaluation, or
  result status;
- modifying GPU placement PR `#53` or attempting CPU/GPU policy unification here;
- a scheduler, supervisor, provider/backend abstraction, generic cache service, or
  cross-project portable package;
- dynamic resizing, worker migration, CPU affinity tuning, NUMA tuning, multi-node
  placement, Slurm/Kubernetes/Ray/Dask integration;
- automatic batch, thread, dataloader, precision, or scientific configuration tuning;
- claims of throughput optimality or scientific evidence from resource probes.

## Deterministic acceptance

Tests must cover at least:

1. affinity and finite current/ancestor quota combinations, including fractional
   quota;
2. unlimited, malformed, zero-period, contradictory, and escaping cgroup paths;
3. high load average with low measured execution occupancy;
4. low load average with high measured execution occupancy;
5. `iowait` exclusion and no guest-time double counting;
6. same-cgroup and ancestor-domain sibling usage;
7. aligned worker/system/domain intervals and exact-once probe subtraction;
8. process-tree CPU demand and descendant cleanup;
9. CPU-, memory-, task-, configured-cap-, and growth-bound cases;
10. throughput candidates never exceed measured capacity and resource-invalid fast
    candidates are rejected;
11. immutable selection digest and old-policy invalidation;
12. run never calls probe or throughput selection;
13. load average raised by plan cannot change the frozen worker count;
14. true CPU/RAM/quota changes block without silent downshift;
15. three-sample revalidation uses the conservative pressure for each constraint;
16. conflicting live process, identity drift, or missing selection blocks;
17. all affected thin wrappers delegate to shared arithmetic rather than copying it;
18. fixed launchers and scientific matrices remain unchanged.

Before real-server shadow, the exact PR head must pass Python compilation, focused
tests, full pytest, Ruff, handoff-authority no-op verification, formal execution
channel validation, governance inventory, governance-stage validation, and exact diff
review for scientific isolation.

## Real CPU shadow gate

The exact reviewed commit must be tested in a new work directory on the target E7
server. The shadow is engineering evidence only and must not start the full scientific
sweep.

It must record affinity, every finite quota domain, load diagnostics, `/proc/stat`
busy cores, quota-domain usage, representative worker CPU/RSS demand, safe ceiling,
every throughput candidate, selected workers, selection digest, and revalidation
evidence. It must exercise a candidate above one when capacity permits, prove that
plan-induced load-average elevation cannot change the frozen selection, prove that
run launches no second candidate grid, and leave no orphan process group.

A separate approved small real-data liveness using the exact selected worker count is
required before resuming the stopped 150-branch Stage A workload. Static tests or a
single resource probe are not sufficient for runtime readiness.

## Rollback

1. Stop invoking the affected E7 `*_auto.py` entrypoints.
2. Use the unchanged fixed launcher or the last separately verified fixed schedule.
3. Preserve selections, revalidation records, failed work directories, candidate
   summaries, and logs.
4. Revert the measured-CPU implementation as one reviewed change.
5. Do not delete or reinterpret the prior `112 -> 1` failure evidence.
6. Do not treat any resource shadow as a scientific result.
