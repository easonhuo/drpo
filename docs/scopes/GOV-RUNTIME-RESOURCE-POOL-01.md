# GOV-RUNTIME-RESOURCE-POOL-01 — explicit per-experiment resource pools

**Approval:** user explicitly authorized implementation on 2026-07-14.  
**Stacked implementation base:** `6afaa58803516d46979287fa40921680aac4f183` from Draft PR `#65`.  
**Dependency:** `GOV-RUNTIME-RESOURCE-AUTOTUNE-CPU-V2-01`.  
**Scientific experiment impact:** none.  
**Default-policy impact:** none; omission of explicit pool arguments preserves inherited process affinity and existing GPU visibility.

## Objective

Prevent independently launched E7 and E8 workloads from treating the complete server
as their private resource domain. Add an explicit, auditable resource-pool contract so
that each experiment is limited to declared CPU IDs and, where applicable, GPU IDs
before any resource discovery, probe, plan, or scientific subprocess starts.

The resource pool is a runtime placement boundary only. It must not change datasets,
methods, seeds, model settings, batch/thread settings, training horizon, evaluation,
or scientific matrix membership.

## Incident evidence

A concurrent E7 and E8 server run showed that both workloads were allowed to see the
same 384-CPU host. Sixteen E8 workers consumed roughly 206 CPU cores under the
existing unbounded PyTorch thread environment while E7 consumed additional CPU.
Runnable-thread pressure drove load average far above the CPU count and roughly
doubled E8 wall time. This is engineering contention evidence, not a scientific
result and not evidence that a fixed 80-core E8 pool is sufficient.

## V1 implementation scope

- parse explicit Linux CPU-list syntax such as `0-31,64-95`;
- reject empty, malformed, descending, duplicate, negative, or unavailable CPU IDs;
- require the requested CPU set to be a subset of the caller's inherited affinity;
- apply the CPU pool with `os.sched_setaffinity(0, ...)` before machine discovery;
- verify the effective affinity exactly matches the requested pool;
- represent inherited affinity explicitly when no pool is requested;
- parse and normalize optional GPU ID lists without changing CUDA semantics;
- emit a stable resource-pool identity and SHA-256 digest;
- include the pool identity in E7 resource fingerprints, immutable selections,
  run identities, and revalidation evidence;
- require plan and run to use the same explicit/inherited pool identity;
- preserve child-process inheritance of CPU affinity and existing CUDA/GPU arguments;
- provide a small command wrapper for controlled E8/fixed-launcher execution without
  duplicating scientific launch logic;
- add deterministic tests for parsing, subset validation, exact application,
  identity stability, plan/run mismatch, and subprocess inheritance.

## Supported initial entrypoints

Direct `--cpu-pool` integration:

```text
scripts/run_e7_canonical_exp_horizon_joint_auto.py
scripts/run_e7_ppo_w0_grid_pilot_auto.py
scripts/run_e7_w0_highc_actor_auto.py
scripts/run_e7_squared_exp_night_auto.py
```

Generic controlled wrapper for E8 and fixed launchers:

```text
scripts/run_with_resource_pool.py
```

The wrapper accepts a command after `--`, applies the declared CPU pool, optionally
sets `CUDA_VISIBLE_DEVICES` from a declared GPU pool, writes one provenance document,
and then `exec`s the existing launcher. It does not inspect or alter scientific
arguments.

## Identity contract

A pool document contains:

```text
schema_version
source = explicit_cli | inherited_affinity
requested_cpu_ids
effective_cpu_ids
cpu_count
requested_gpu_ids
effective_cuda_visible_devices
pool_digest
```

CPU IDs are canonical sorted integers. GPU IDs are canonical ordered unique strings.
The digest excludes timestamps and mutable runtime observations.

For E7 plan/run:

- pool activation occurs before `discover_machine()`;
- the pool identity enters the resource fingerprint and selection digest;
- `RUN_IDENTITY.json` records pool digest and exact IDs;
- run validates the requested/inherited pool against both selection and run identity
  before CPU/RAM revalidation;
- a changed pool fails closed and never triggers automatic replanning or worker-count
  downshift.

## Safety and failure contract

Fail closed on:

- unavailable `sched_getaffinity` / `sched_setaffinity` for an explicit pool;
- requested CPU outside inherited affinity;
- effective affinity different from requested affinity;
- malformed or duplicate CPU/GPU IDs;
- plan/run pool digest mismatch;
- selection/run-identity pool mismatch;
- wrapper provenance path already occupied by a different pool identity;
- child process that does not inherit the expected CPU affinity in deterministic
  acceptance tests.

The implementation must not kill, pause, migrate, or change affinity of unrelated
processes.

## Explicit exclusions

- automatic selection of which CPU IDs belong to E7 or E8;
- automatic NUMA/socket optimization;
- central scheduler, reservation daemon, lock server, or global resource database;
- dynamic resizing, preemption, migration, or cross-run negotiation;
- Slurm/Kubernetes/Ray integration;
- automatic thread-count selection;
- changing OMP, MKL, PyTorch, dataloader, tokenizer, or verifier thread settings;
- claiming that E8 needs 80 cores before a separate exact-head thread/capacity shadow;
- merging PR `#65`, GPU PR `#53`, or any scientific PR as part of this claim;
- modifying `docs/handoff.md`, `experiments/registry.yaml`, scientific configs, or
  formal execution-channel governance.

## Acceptance

Deterministic acceptance requires:

1. exact CPU-list parsing and canonicalization;
2. subset-only affinity application and exact post-application verification;
3. inherited-affinity behavior when no explicit pool is supplied;
4. stable pool digest independent of timestamps;
5. explicit and inherited identities are distinguishable;
6. subprocesses inherit the exact pool;
7. E7 plan selection includes the pool and run rejects a different pool before probe
   or revalidation;
8. E7 run with the same pool preserves selection digest and worker count;
9. generic wrapper sets CPU affinity and optional `CUDA_VISIBLE_DEVICES` without
   altering the delegated command;
10. existing fixed launchers and no-pool paths remain compatible;
11. focused tests, full pytest, Ruff, Python compile, and governance gates pass on the
   exact stacked head.

Real-server acceptance additionally requires a topology audit and a controlled
concurrent E7/E8 shadow using non-overlapping pools. The audit must report socket/NUMA
mapping, exact CPU IDs, GPU IDs, thread counts, CPU usage, step/evaluation throughput,
and no orphan processes. It must not start or reinterpret a formal scientific sweep.

## Rollback

1. Stop passing `--cpu-pool` or stop using `run_with_resource_pool.py`.
2. Use the existing inherited-affinity launch path.
3. Preserve pool provenance, selections, revalidations, logs, and contention evidence.
4. Revert this stacked claim independently after PR `#65` if necessary.
5. Never reinterpret a resource-pool shadow as a scientific result.
