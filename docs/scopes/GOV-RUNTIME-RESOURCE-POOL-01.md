# GOV-RUNTIME-RESOURCE-POOL-01 — explicit per-experiment resource pools

**Approval:** user explicitly authorized implementation on 2026-07-14.  
**Stacked implementation base:** `6afaa58803516d46979287fa40921680aac4f183` from Draft PR `#65`.  
**Dependency:** `GOV-RUNTIME-RESOURCE-AUTOTUNE-CPU-V2-01`.  
**Scientific experiment impact:** none.  
**Default-policy impact:** none; omitting the wrapper preserves inherited process affinity and existing GPU visibility.

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
- apply the CPU pool with `os.sched_setaffinity(0, ...)` before delegated startup;
- verify the effective affinity exactly matches the requested pool;
- represent inherited affinity explicitly when no pool is requested;
- parse and normalize optional GPU ID lists without changing CUDA semantics;
- emit a stable resource-pool identity and SHA-256 digest;
- preserve child-process inheritance of CPU affinity and the delegated command;
- require a declared physical GPU pool to match one delegated `--gpus` argument, or
  explicitly use `CUDA_VISIBLE_DEVICES` enforcement for compatible commands;
- create or exactly revalidate one immutable `RESOURCE_POOL.json` shared by plan and
  run;
- provide a small command wrapper for E7, E8, and fixed launchers without duplicating
  their scientific launch logic;
- add deterministic tests for parsing, subset validation, exact application,
  immutable identity, GPU enforcement, exact delegation, and subprocess inheritance.

## Supported initial entrypoint

```text
scripts/run_with_resource_pool.py
```

The wrapper can delegate to the four measured-CPU E7 auto entrypoints supplied by PR
`#65`, existing E8 launchers, or fixed launchers. It applies the pool before the
existing entrypoint performs machine discovery or launches children, writes one
immutable provenance document, and then `exec`s the exact delegated command.

V1 intentionally does not add duplicate `--cpu-pool` parsing to every workload
launcher. The wrapper is the single pool-activation boundary.

## Identity contract

A pool document contains:

```text
schema_version
source = explicit_cli | inherited_affinity
inherited_cpu_ids
requested_cpu_ids
effective_cpu_ids
cpu_count
requested_gpu_ids
gpu_enforcement = none | launcher_argument | cuda_visible
pool_digest
```

CPU IDs are canonical sorted integers. GPU IDs are canonical ordered unique strings.
The digest excludes timestamps, command text, and mutable runtime observations.
Creation uses exclusive file creation. Reuse is permitted only when the complete JSON
identity matches exactly; a concurrent or later different identity fails closed.

The wrapper exports:

```text
DRPO_RESOURCE_POOL_DIGEST
DRPO_CPU_POOL
DRPO_GPU_POOL        # only when declared
CUDA_VISIBLE_DEVICES # only in cuda_visible mode
```

## Relationship to E7 measured-CPU V2

This stacked claim does not modify E7 capacity arithmetic or the four E7 auto
entrypoints. Instead:

1. the wrapper applies the explicit CPU affinity before the E7 process starts;
2. PR `#65` discovers that restricted `sched_getaffinity(0)` set;
3. the exact effective affinity is recorded in `RUNTIME_SELECTION.json` and enters the
   immutable selection digest;
4. the selected worker count and digest are bound into `RUN_IDENTITY.json`;
5. run revalidation rejects changed affinity before scientific execution;
6. the shared `RESOURCE_POOL.json` independently ensures the operator invokes plan
   and run with the same explicit or inherited pool identity.

Thus a changed pool is blocked either by immutable pool identity or by the measured
CPU binding contract. It never triggers automatic replanning or silent worker-count
downshift.

## Relationship to E8 GPU placement

For E8 launchers that already accept physical GPU IDs, the recommended mode is:

```text
gpu_enforcement = launcher_argument
```

The wrapper requires the declared ordered GPU IDs to exactly match one delegated
`--gpus` value. It does not reinterpret physical IDs or modify GPU placement.

`cuda_visible` mode sets `CUDA_VISIBLE_DEVICES` and is valid only for commands whose
GPU identity is defined by CUDA visibility. It must not wrap a physical-ID launcher
without separate validation.

## Safety and failure contract

Fail closed on:

- unavailable `sched_getaffinity` or `sched_setaffinity` for an explicit pool;
- requested CPU outside inherited affinity;
- effective affinity different from requested affinity;
- malformed or duplicate CPU/GPU IDs;
- declared GPU pool without an enforcement mode;
- missing, repeated, or mismatched delegated `--gpus` in launcher-argument mode;
- existing pool identity that differs in source, CPU IDs, GPU IDs, or enforcement;
- child process that does not inherit the expected CPU affinity in deterministic
  acceptance tests.

The implementation must not kill, pause, migrate, reserve, or change affinity of
unrelated processes.

## Explicit exclusions

- automatic selection of which CPU IDs belong to E7 or E8;
- automatic NUMA/socket optimization;
- direct changes to every E7/E8 launcher;
- central scheduler, reservation daemon, lock server, or global resource database;
- dynamic resizing, preemption, migration, or cross-run negotiation;
- Slurm/Kubernetes/Ray integration;
- automatic thread-count selection;
- changing OMP, MKL, PyTorch, dataloader, tokenizer, or verifier thread settings;
- claiming that E8 needs 80 cores before a separate exact-head thread/capacity shadow;
- merging PR `#65`, GPU PR `#53`, or any scientific PR as part of this claim;
- modifying `docs/handoff.md`, `experiments/registry.yaml`, scientific configs, or
  formal execution-channel governance.

## Deterministic acceptance

1. Exact CPU-list parsing and canonicalization.
2. Subset-only affinity application and exact post-application verification.
3. Inherited-affinity behavior when no explicit pool is supplied.
4. Stable pool digest independent of timestamps and delegated command text.
5. Explicit and inherited identities are distinguishable.
6. Actual delegated subprocess inherits the exact CPU pool.
7. Immutable identity accepts exact reuse and rejects a different plan/run pool.
8. Launcher-argument GPU enforcement requires one exact ordered `--gpus` value.
9. CUDA-visible enforcement exports only the declared GPU visibility.
10. Wrapper preserves the delegated executable and argument vector exactly.
11. Existing no-wrapper and fixed-launcher paths remain unchanged.
12. Focused tests, full pytest, Ruff, Python compile, and governance gates pass on the
    exact stacked head.

## Real-server acceptance

The exact final head requires:

- server socket/NUMA/CPU/GPU topology audit;
- explicit non-overlapping E7 and E8 CPU pools chosen from that topology;
- dry-run validation of both complete commands;
- a controlled concurrent engineering shadow showing child affinity remains inside
  each pool;
- E7 selection/revalidation evidence produced inside its pool;
- E8 CPU use, thread count, training throughput, pass@64 throughput, and GPU utilization
  recorded inside its pool;
- no process from either workload executing outside its assigned affinity;
- no orphan process group;
- no formal scientific sweep or scientific-result claim.

A separate E8 thread-count scan remains required before assigning a permanent pool
size. The observed unbounded run does not establish that OMP/MKL=4 or an 80-core pool
is sufficient.

## Rollback

1. Stop using `run_with_resource_pool.py`.
2. Use the existing inherited-affinity launch path.
3. Preserve pool provenance, selections, revalidations, logs, and contention evidence.
4. Revert this stacked claim independently after PR `#65` if necessary.
5. Never reinterpret a resource-pool shadow as a scientific result.
