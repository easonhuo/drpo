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

## Corrective finding from the second H20 shadow

The second real-H20 shadow of commit
`a28a3bde4cde3785fb0e037e8f486e2d86b12c84` established that the dedicated worker
completed the full phase contract and measured both the training and maximum-shape
evaluation envelopes. It did not validate multi-worker placement, however. The
server exposed eight idle homogeneous H20 devices, but the selector interpreted the
one-minute load average (`387.5`) as already-consumed worker capacity on a
384-logical-CPU host. The formula
`floor(logical_cpu_count * cpu_fraction - load_average_1m)` collapsed the total CPU
worker limit to one, reduced the selected GPU pool to one device, and never launched
a candidate above one.

Linux load average includes runnable and uninterruptible tasks and is not a measured
count of CPU cores unavailable to this workload. The second shadow therefore passes
the single-worker resource-envelope sub-gate only; it does not establish automatic
multi-worker placement. Its `slots_per_gpu=1` is not a hardware capacity conclusion.

The same shadow also recorded `worker_returncodes=[null]` because the controller
accepted the completed phase markers before requiring normal worker termination. A
phase-complete process must exit zero and leave no process-group descendants before
its result is accepted. Controller-forced termination after `probe_complete` is a
failed candidate, not a clean success.

## V1 scope

- static eligible-GPU filtering from visibility, utilization, free VRAM, host RAM,
  cgroup memory, and configured candidate IDs;
- homogeneous selected GPU pool only;
- one dedicated resource-equivalent workload probe preserving the real model,
  training micro-batch, negative-bank shape, sequence limits, evaluation batch size,
  maximum configured pass@k, and generation length while reducing only outer
  repetition;
- mandatory phase contract:
  `model_loaded`, `training_peak_completed`, `evaluation_peak_completed`, and
  `probe_complete`;
- mandatory clean worker termination: every accepted worker exits zero and its
  process group disappears without controller intervention;
- one representative single-worker envelope measuring incremental peak VRAM,
  process-tree host RSS, process-tree CPU seconds, and system CPU occupancy across
  all required phases;
- CPU capacity derived from measured worker CPU demand plus measured `/proc/stat`
  external CPU occupancy; one-minute load average is retained only as diagnostic
  provenance and is never subtracted as a worker count;
- automatic CPU, host-memory, VRAM, task-count, and safety-ceiling capacity limits;
- bounded same-GPU validation of a small candidate set using the same phase and exit
  contracts;
- automatic `slots_per_gpu`, total slot count, and per-GPU placement output;
- fail-closed cleanup of all probe process groups and preservation of logs and phase
  evidence;
- exact runtime-selection provenance, cache revalidation, and unchanged scientific
  configuration;
- integration into the opt-in Countdown E8 taper auto entrypoint;
- the historical fixed one-process-per-GPU runtime and earlier selector modules
  remain available and unchanged for provenance.

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

## Measured-CPU phase-aware selection contract

The workload adapter declares a fixed topology of one GPU per independent task and
provides a dedicated resource-envelope command. Selector policy version 2:

1. records the machine snapshot and eligible homogeneous devices without using raw
   load average to truncate the GPU pool;
2. measures current process-visible system CPU occupancy from per-CPU `/proc/stat`
   accounting; `iowait` is not charged as busy compute time;
3. runs one worker and requires all phase markers, measured VRAM/RSS/CPU demand,
   normal exit code zero, and disappearance of the complete process group;
4. fails closed when the single worker does not complete training and maximum-shape
   evaluation or does not exit cleanly inside the bounded window;
5. estimates external CPU occupancy by subtracting measured probe-worker CPU demand
   from concurrent system busy-core accounting;
6. derives a candidate from measured VRAM, host RSS, worker CPU demand, external CPU
   occupancy, safety factors, remaining task count, and an operator safety ceiling;
7. validates the candidate on one GPU and requires every worker to complete every
   phase and exit zero without controller cleanup;
8. projects candidate host RSS and worker CPU demand across the full selected device
   pool, adding the concurrently measured external CPU occupancy;
9. backs off through a bounded candidate sequence when validation fails;
10. writes `RUNTIME_SELECTION.json` before the scientific scheduler starts.

The maximum-slot setting is only a search ceiling. It is never the selected value by
itself. If measured resources permit a candidate above one, at least one such
candidate must be executed before multi-worker readiness can be claimed. If no
candidate above one passes, the selector may emit one slot only as a conservative
runtime placement, but the corresponding hardware shadow is not evidence that the
multi-worker optimization is ready. If even the single-worker envelope is
incomplete, selection fails and no runtime schedule is emitted.

## Cache contract

The phase-aware worker contract remains version 2. The selector uses policy version
2 and `RUNTIME_SELECTION.json` schema version 3. Cached selections from the
liveness-only selector or the raw-load-average selector are invalid. Reuse requires
the exact workload, probe implementation, selector implementation, selector policy,
machine identity, required phase list, a phase-complete and clean-exit record for the
selected concurrency, and current revalidation of GPU, host-memory, and measured
system CPU capacity.

## Acceptance

- deterministic tests cover measured capacity derivation, candidate backoff,
  timeout/fallback, OOM/nonzero rejection, host-memory and measured CPU limits,
  high-load-average/non-busy-CPU separation, projected CPU occupancy,
  heterogeneous-device rejection, per-GPU slot expansion, old-cache invalidation,
  phase-incomplete rejection, phase-evidence preservation, clean-exit enforcement,
  process-group cleanup, CLI compatibility, and scientific configuration
  preservation;
- Python compilation, Ruff, full pytest, handoff authority verification, formal
  execution-channel validation, governance inventory, and governance-stage
  validation pass on the exact PR head;
- a third real E8 H20 shadow must show that a candidate above one was actually
  launched when measured VRAM/RAM/CPU permit it, and that every accepted worker
  completed both the training and maximum-shape evaluation phases, exited zero,
  left no process-group descendants, and did not start a full sweep;
- the third shadow must separately report load average, measured system busy cores,
  estimated external busy cores, measured worker CPU cores, projected full-pool CPU
  demand, and every candidate/backoff decision;
- no real-hardware readiness or merge approval is inferred from either superseded
  shadow or from CI alone.

## Rollback

1. Stop invoking the GPU-placement auto entrypoint.
2. Continue using the unchanged fixed E8 runtime or the previous one-process-per-GPU
   path.
3. Preserve `RUNTIME_SELECTION.json`, probe logs, phase evidence, and failed probes.
4. Revert this claim's files as one reviewed change if the feature is removed.
5. Never delete scientific outputs, completed cells, or historical failure evidence.
