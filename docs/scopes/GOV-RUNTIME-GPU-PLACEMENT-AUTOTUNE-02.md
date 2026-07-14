# GOV-RUNTIME-GPU-PLACEMENT-AUTOTUNE-02

**Approval:** user explicitly authorized implementation on 2026-07-13.  
**Implementation base:** `9c49824558b1eb7f697f299b246a135ff35a2017`.  
**Scientific experiment impact:** none.  
**Default-policy impact:** none; the new path remains explicit opt-in.

## Objective

Extend the existing E8 runtime-resource path from idle-device selection to automatic
single-GPU task placement. For a workload whose task topology is already fixed as
one GPU per independent task, the selector must automatically determine a safe
`slots_per_gpu` value instead of assuming one process per GPU or requiring an
operator-written constant.

The cold probe target is approximately ten minutes with a hard wall-clock budget.
The selected placement is runtime provenance only and must not change model, data,
method, seed, batch, sequence length, training horizon, evaluation protocol, or the
scientific cell matrix.

## Corrective finding from the first H20 shadow

The first real-H20 shadow of commit
`8466115e89fb639063fa07728493b98b42a86e06` showed that a timed liveness window was
not a sufficient resource-envelope test. Eight workers stayed alive without OOM,
but they had not reached a training update or the maximum-shape evaluation path.
The resulting `slots_per_gpu=8` therefore did not establish full-workload capacity.

This is a code-contract defect, not a scientific result. The old success rule is
superseded. A process that merely loads a model or stays alive must never be accepted
as a safe concurrency candidate.

## V1 scope

- static eligible-GPU filtering from visibility, utilization, free VRAM, host RAM,
  cgroup memory, current CPU load, and configured candidate IDs;
- homogeneous selected GPU pool only;
- one dedicated resource-equivalent workload probe preserving the real model,
  training micro-batch, negative-bank shape, sequence limits, evaluation batch size,
  maximum configured pass@k, and generation length while reducing only outer
  repetition;
- mandatory phase contract:
  `model_loaded`, `training_peak_completed`, `evaluation_peak_completed`, and
  `probe_complete`;
- one representative single-worker envelope measuring incremental peak VRAM and
  process-tree host RSS across all required phases;
- automatic CPU, host-memory, VRAM, task-count, and safety-ceiling capacity limits;
- bounded same-GPU validation of a small candidate set using the same phase contract;
- automatic `slots_per_gpu`, total slot count, and per-GPU placement output;
- fail-closed cleanup of all probe processes and preservation of logs and phase
  evidence;
- exact runtime-selection provenance, cache revalidation, and unchanged scientific
  configuration;
- integration into the opt-in Countdown E8 taper auto entrypoint;
- the historical fixed one-process-per-GPU runtime remains available and unchanged.

## Explicitly excluded

- selecting or tuning DDP, tensor parallelism, FSDP, ZeRO, pipeline parallelism, or
  any other distributed-training strategy;
- heterogeneous-GPU packing or changing the number of GPUs required by one task;
- multi-node placement, topology-aware collectives, Slurm, Kubernetes, Ray, or Dask;
- automatic batch size, gradient accumulation, precision, sequence length, or
  scientific evaluation changes;
- CPU thread affinity, NUMA placement, or dataloader-worker autotuning;
- online migration or dynamic slot changes after the scientific run starts;
- claims of global throughput optimality;
- modifying `docs/handoff.md`, `experiments/registry.yaml`, the formal execution
  channel, or any closed-stage protected file.

## Phase-aware selection contract

The workload adapter declares a fixed topology of one GPU per independent task and
provides a dedicated resource-envelope command. The selector:

1. records the machine snapshot and eligible homogeneous devices;
2. runs one worker and requires all phase markers before accepting the measured
   single-worker peak;
3. fails closed when the single worker does not complete training and maximum-shape
   evaluation inside the bounded window;
4. derives a candidate from measured VRAM/host usage, CPU/load capacity, safety
   factors, remaining task count, and an operator safety ceiling;
5. validates the candidate on one GPU and requires every worker to complete every
   required phase;
6. projects measured host RSS across the selected device pool;
7. backs off through a bounded candidate sequence when validation fails;
8. writes `RUNTIME_SELECTION.json` before the scientific scheduler starts.

The maximum-slot setting is only a search ceiling. It is never the selected value by
itself. If no candidate above one passes the complete phase contract, the already
validated single-worker envelope is used. If even the single-worker envelope is
incomplete, selection fails and no runtime schedule is emitted.

## Cache contract

The phase-aware selector uses probe contract version 2. Cached selections from the
older liveness-only implementation are invalid. Reuse requires the exact workload,
probe implementation, selector policy, machine identity, required phase list, and a
phase-complete record for the selected concurrency.

## Acceptance

- deterministic tests cover measured capacity derivation, candidate backoff,
  timeout/fallback, OOM/nonzero rejection, host-memory and CPU limits,
  heterogeneous-device rejection, per-GPU slot expansion, old-cache invalidation,
  phase-incomplete rejection, phase-evidence preservation, CLI compatibility, and
  scientific configuration preservation;
- Python compilation, Ruff, full pytest, handoff authority verification, formal
  execution-channel validation, governance inventory, and governance-stage validation
  pass on the exact PR head;
- a new real E8 H20 shadow must show that the accepted candidate completed both the
  training and maximum-shape evaluation phases, left no orphan workers, and did not
  start a full sweep;
- no real-hardware readiness or merge approval is inferred from the superseded first
  shadow or from CI alone.

## Rollback

1. Stop invoking the GPU-placement auto entrypoint.
2. Continue using the unchanged fixed E8 runtime or the previous one-process-per-GPU
   auto path.
3. Preserve `RUNTIME_SELECTION.json`, probe logs, phase evidence, and failed probes.
4. Revert this claim's files as one reviewed change if the feature is removed.
5. Never delete scientific outputs, completed cells, or historical failure evidence.
