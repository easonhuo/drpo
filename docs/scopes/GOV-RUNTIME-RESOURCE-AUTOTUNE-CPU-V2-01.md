# GOV-RUNTIME-RESOURCE-AUTOTUNE-CPU-V2-01 — measured CPU capacity maintenance scope

**Approval:** user explicitly authorized implementation after approving and merging the design ledger on 2026-07-14.  
**Implementation base:** `f387391b39502e700e7f780cd8b4a1fd9c7eca7c`.  
**Design authority:** `docs/runtime_resource_autotune_evolution.md`.  
**Usage document:** `docs/runtime_resource_autotuning_v2.md`.  
**Implementation PR:** Draft PR `#65`.  
**Scientific experiment impact:** none.  
**Default-policy impact:** none; fixed launchers remain available and unchanged.

## Objective

Replace the E7 CPU autotune path's raw-load-average capacity arithmetic with the
measured-CPU V2 contract. Eliminate plan/run self-feedback: `plan` is the only
operation that may probe and select a worker count, while `run` verifies and consumes
that immutable selection or fails closed.

The maintenance changes only active independent subprocess count. It must not change
scientific branches, methods, seeds, coefficients, optimizer, batch, thread
environment, horizon, stopping rule, evaluation, or result status.

## Implemented production surface

- `src/drpo/runtime_cpu_capacity.py` — Linux affinity/cgroup observation, aligned CPU
  accounting, process-tree demand, and pure capacity arithmetic;
- `src/drpo/runtime_resource_adapters.py` — canonical E7 V2 selection/revalidation;
- `src/drpo/e7_ppo_w0_runtime_autotune.py` — shared PPO-family resource probe,
  resource-valid throughput candidates, immutable selection, and revalidation;
- `src/drpo/e7_w0_highc_runtime_autotune.py` and
  `src/drpo/e7_squared_exp_night_runtime_autotune.py` — thin representative identity,
  fingerprint, and implementation-identity adapters;
- `scripts/run_e7_canonical_exp_horizon_joint_auto.py`;
- `scripts/run_e7_ppo_w0_grid_pilot_auto.py`;
- `scripts/run_e7_w0_highc_actor_auto.py`;
- `scripts/run_e7_squared_exp_night_auto.py`;
- `docs/runtime_resource_autotuning_v2.md`;
- this scope and focused deterministic tests.

The fixed launchers, scientific runners, configurations, `docs/handoff.md`,
`experiments/registry.yaml`, formal execution channel, and GPU PR `#53` are unchanged.

## CPU observation contract

- bind capacity to the exact `sched_getaffinity(0)` CPU set;
- resolve current cgroup membership from `/proc/self/cgroup`;
- discover every visible finite quota from current cgroup through controller root;
- support cgroup v2 `cpu.max`/`cpu.stat` and direct cgroup v1
  quota/period/`cpuacct.usage` files;
- reject paths escaping the configured mount root;
- accept a namespaced mount-root fallback only when `cgroup.procs` or `tasks`
  explicitly lists the current PID;
- sample affinity-scoped `/proc/stat` and every finite quota domain over aligned
  monotonic intervals;
- count user, nice, system, irq, softirq, and steal as execution;
- treat idle and iowait as non-executing capacity and avoid guest double counting;
- record load averages as diagnostics only;
- fail closed on malformed or unresolved cgroup evidence, changing affinity, missing
  CPU rows, or invalid counter deltas.

## Worker demand and capacity contract

- measure representative process-tree CPU seconds and peak RSS in one bounded probe;
- reserve CPU using an explicit safety factor and minimum floor;
- apply affinity and every finite quota domain as independent constraints;
- retain host/cgroup memory, task, configured-cap, and bounded-growth constraints;
- retain only workload-specific throughput search already present in PPO-family paths;
- require candidate completion, no timeout/controller cleanup/orphan, CPU validity in
  affinity and every quota domain, and aggregate RSS validity;
- never select a resource-invalid candidate because it is faster;
- canonical exp-horizon selects the measured safe ceiling and records the absence of a
  throughput-knee search as a limitation.

Default policy introduced by the opt-in V2 entrypoints:

```text
cpu_fraction = 0.85
memory_headroom_fraction = 0.15
per_worker_memory_safety_factor = 1.20
per_worker_cpu_safety_factor = 1.25
minimum_cpu_cores_per_worker = 1.0
revalidation_samples = 3
revalidation_sample_seconds = 1.0
```

## Plan/run lifecycle contract

### Plan

- owns representative probing and optional bounded throughput candidates;
- creates one `RUNTIME_SELECTION.json` with schema and selector policy version 2;
- records implementation hashes, exact CPU binding, measured CPU/RSS evidence, limits,
  selected workers, and stable selection digest;
- refuses to replace an existing selection in the same work directory;
- delegates to the existing scientific runner's plan command and binds worker count
  plus selection digest into `RUN_IDENTITY.json`.

### Run

- requires both immutable selection and plan-created run identity;
- validates selected workers and digest before dynamic capacity sampling;
- never calls automatic selection, representative probe, or throughput candidates;
- verifies schema, policy, adapter, implementation, source, resource fingerprint,
  binding, and selection digest;
- confirms no conflicting process referring to the work directory is alive;
- performs three consecutive CPU/domain samples and uses the conservative maximum in
  each independent domain;
- validates projected selected-worker CPU and current host/cgroup memory;
- writes `_runtime_resource_attempts/<attempt_id>/RUNTIME_REVALIDATION.json`;
- starts the exact selected count or fails
  `RUNTIME_CAPACITY_CHANGED_REPLAN_REQUIRED`;
- never silently changes a planned count such as 112 to 80, 20, or 1;
- never mutates the original selection during allow or block decisions.

## Compatibility and artifact contract

- raw-load-average E7 selection/cache semantics are incompatible with V2;
- stable identity and dynamic revalidation evidence remain separate;
- selections include `scientific_matrix_changed: false`;
- failed probes, candidate logs/summaries, blocked revalidations, and cleanup evidence
  are preserved;
- fixed launchers and historical artifacts remain valid rollback/provenance paths;
- Stage A KL integration remains separate until its development branch is synchronized
  after shared-core acceptance.

## Explicitly excluded

- handoff or registry changes;
- formal execution-channel or RunSpec governance changes;
- scientific-variable, matrix, seed, threshold, horizon, evaluation, or result-status
  changes;
- modification of GPU placement PR `#53` or automatic CPU/GPU policy unification;
- scheduler, supervisor, provider/backend abstraction, generic cache service, or
  cross-project portable package;
- online resizing, migration, CPU-affinity/NUMA tuning, multi-node or external
  scheduler integration;
- batch, thread, dataloader, precision, or scientific configuration tuning;
- throughput-optimality or scientific claims from resource probes.

## Deterministic acceptance

Coverage includes:

1. current and ancestor finite quotas, fractional and unlimited quota;
2. malformed, zero-period, contradictory, escaping, unresolved, and namespaced paths;
3. high load average with low measured execution and real high measured pressure;
4. iowait exclusion, steal inclusion, and no guest-time double counting;
5. same-cgroup and ancestor sibling usage;
6. aligned worker/system/domain intervals and exact-once probe subtraction;
7. process-tree CPU demand and cleanup;
8. CPU-, memory-, task-, configured-cap-, and growth-bound selection;
9. resource-invalid fast candidate rejection;
10. immutable digest and old-policy invalidation;
11. run does not call resource probe or candidate benchmark;
12. plan-induced load average cannot change selected workers;
13. genuine CPU/RAM/quota changes block without downshift;
14. conservative multi-sample revalidation;
15. live-process, identity, missing-selection, or missing-run-identity blocking;
16. thin wrapper use of shared arithmetic and wrapper implementation identity;
17. fixed launcher and scientific-matrix preservation.

At implementation head `7b1369221a26da993888f8eaf6df7bdb3d35be77`, GitHub Actions run
`29326666713` passed tiered test plan, Python compilation, shell syntax, handoff
authority, formal execution-channel validation, governance inventory, governance
stage validation, full pytest, and Ruff. Any subsequent documentation or code commit
requires a new exact-head CI result before shadow.

## Real CPU shadow gate

The exact final reviewed commit must be tested in a new work directory on the target
E7 server. The shadow is engineering evidence only and must not start a full
scientific sweep.

It must record affinity, every finite quota domain, load diagnostics, `/proc/stat`
busy cores, quota-domain use, representative worker CPU/RSS demand, safe ceiling,
every throughput candidate, selected workers, digest, and revalidation evidence. It
must execute a candidate above one when capacity permits, prove that plan-induced
load-average elevation does not change the selection, prove run launches no second
probe/grid, and leave no orphan process group.

A separately approved small real-data liveness with the exact selected count remains
required before the stopped 150-branch Stage A workload can resume. Static tests,
CI, or one resource probe are not runtime-readiness evidence.

## Current status

```text
implementation: complete on Draft PR #65
focused/full CI: passed at implementation head 7b1369221a26da993888f8eaf6df7bdb3d35be77
documentation synchronization: in progress after that head
real CPU shadow: not run
small real-data liveness: not run
merge/default cutover: not approved
scientific result: none
```

## Rollback

1. Stop using affected E7 `*_auto.py` entrypoints.
2. Use the unchanged fixed launcher or a separately verified fixed schedule.
3. Preserve selections, revalidations, candidate summaries, logs, and failed work
   directories.
4. Revert the measured-CPU implementation as one reviewed change.
5. Preserve and do not reinterpret the prior `112 -> 1` failure evidence.
6. Never treat a resource shadow as a scientific result.
