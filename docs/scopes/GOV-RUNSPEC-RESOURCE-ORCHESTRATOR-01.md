# GOV-RUNSPEC-RESOURCE-ORCHESTRATOR-01 — unified RunSpec resource boundary

**Approval:** user explicitly authorized implementation on 2026-07-17.  
**Base:** `main@4544005bd7df69c53bad70a9dcac846af01285e4`.  
**Dependencies:** `GOV-RUNTIME-RESOURCE-POOL-01`, `GOV-RUNTIME-RESOURCE-AUTOTUNE-CPU-V2-01`, `EXT-H-E7-DEFERRED-CAPACITY-ADMISSION-01`.  
**Scientific impact:** none.  
**Experiment execution:** none in this change.

## Problem

The governed RunSpec command is currently `python scripts/agent/run_lane.py --once`, while
CPU-pool activation and capacity waiting live behind separate workload-specific commands.
An operator therefore has to know whether a particular RunSpec entrypoint already calls an
autotuner. That defeats the intended one-command RunSpec interface and makes the resource
boundary easy to omit.

## Objective

Extend the existing RunSpec execution chain so one foreground command can:

1. claim the selected RunSpec;
2. apply an explicit Linux CPU-affinity pool before the scientific entrypoint starts;
3. measure current CPU availability only inside that inherited pool and its cgroup quota
   domains;
4. wait in the foreground until an operator-declared minimum available CPU capacity is
   observed;
5. export one standard maximum-worker ceiling for compatible canonical runners;
6. execute, recover, package, and deliver through the unchanged RunSpec lifecycle.

The implementation must reuse the existing pool parser/identity, measured CPU counters,
foreground wait loop, RunSpec state machine, recovery, packaging, and delivery owners.

## Operator interface

The existing `run_lane.py` and direct claimed-RunSpec entrypoint receive opt-in runtime
arguments:

```text
--cpu-pool <linux-cpu-list>
--resource-cpu-fraction <0,1]
--minimum-available-cpu-cores <positive number>
--resource-wait-timeout-seconds <negative for unlimited>
--resource-poll-seconds <positive number>
--resource-sample-seconds <positive number>
--max-workers <positive integer, optional>
```

Omitting `--cpu-pool` preserves the current RunSpec path exactly. Resource arguments are
runtime placement policy, not scientific hyperparameters, and are recorded under the local
RunSpec log directory rather than inserted into the tracked READY specification.

## Capacity semantics

The generic outer probe answers only whether the selected CPU pool currently has the
operator-declared minimum available CPU capacity. It computes availability from the
restricted affinity and every visible finite cgroup quota domain. It does not infer an
arbitrary program's per-worker CPU or memory demand.

`--max-workers` is an explicit ceiling. It is exported as
`DRPO_RUNTIME_MAX_WORKERS`; canonical E7 runners may consume it as their default executor
width. A workload-specific auto runner may still perform its existing representative-worker
probe inside the already restricted affinity and choose a lower safe width. An explicit
runner argument remains authoritative.

A fixed or sequential liveness runner may ignore the worker ceiling while still receiving
the CPU pool and capacity wait. This is correct because such a runner has no parallel worker
width to tune.

## Planned code surface

Modify existing files only:

- `scripts/agent/run_lane.py` — expose the opt-in one-command resource arguments;
- `scripts/agent/run_claimed_runspec.py` — bind the runtime request to the run lifecycle;
- `scripts/agent/runspec_lib.py` — normalize, activate, record, and prepare resources;
- `src/drpo/runtime_capacity_wait.py` — add one public CPU-pool availability wait using
  existing CPU measurement primitives;
- `src/drpo/e7_canonical_sweep.py` — consume the standard worker ceiling only when no
  explicit `--max-workers` is supplied;
- existing focused test files — cover no-resource compatibility, exact pool binding,
  pool-local waiting, identity reuse, invalid options, and worker-ceiling propagation.

No new Python file, scheduler, daemon, queue, database, launcher family, scientific runner,
trainer, or recovery system is permitted.

## Fail-closed rules

- requested CPU IDs must be available within inherited affinity;
- the effective affinity must exactly equal the requested pool;
- resource values must be finite and within their declared ranges;
- a minimum capacity larger than the pool/cgroup policy budget is rejected immediately;
- CPU or cgroup evidence errors are fatal;
- non-capacity RunSpec failures keep their existing classification;
- a changed immutable pool identity for the same run ID is rejected;
- resource waiting stays in the foreground and launches no background controller;
- the running worker count is never resized after entrypoint launch.

## Explicit exclusions

- automatic NUMA/socket CPU-ID selection;
- changing affinity of unrelated E7 or E8 processes;
- guaranteeing isolation unless E7 and E8 are both launched in non-overlapping pools;
- generic inference of per-worker memory or CPU demand;
- silent command rewriting or insertion of workload-specific `--max-workers` flags;
- dynamic scale-up/down after launch;
- scientific configuration, dataset, seed, method, optimizer, horizon, or evaluation changes;
- modification of `docs/handoff.md` or `experiments/registry.yaml`;
- experiment launch, result interpretation, or merge authorization.

## Acceptance

1. Existing RunSpec execution without resource options is byte-for-byte behavior compatible.
2. One `run_lane.py --once` command applies the declared pool before entrypoint execution.
3. Capacity measurements contain only CPUs in the effective pool and honor quota domains.
4. Waiting records deterministic JSON/JSONL state under the run log directory.
5. Timeout, impossible floor, malformed pool, and contradictory evidence fail closed.
6. Recovery attempts inherit the same pool and worker ceiling.
7. Canonical E7 defaults consume `DRPO_RUNTIME_MAX_WORKERS`; an explicit CLI value wins.
8. No new Python file or duplicate resource/RunSpec subsystem is introduced.
9. Focused tests, Python compilation, Ruff, full pytest, handoff authority, formal execution,
   governance inventory, and governance-stage validation pass on the exact PR head.

## Rollback

Revert the implementation commit. Existing standalone resource wrappers and existing
RunSpec execution remain the rollback path. Preserve any local resource identity and wait
records as engineering provenance; they are not scientific evidence.
