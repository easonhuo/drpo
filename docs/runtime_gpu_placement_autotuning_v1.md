# Runtime GPU placement autotuning V1

## Purpose

The historical E8 resource path can identify visible, idle, sufficiently free GPUs,
but it cannot determine how many independent single-GPU tasks should share each
GPU. This opt-in path closes that gap without becoming a distributed-training
framework.

The task launcher declares the topology. V1 supports only:

```text
one independent task -> one GPU
one GPU -> one or more independent tasks
```

DDP, tensor parallelism, FSDP, multi-node placement, and heterogeneous-GPU packing
remain outside this selector.

## Corrective history

The first H20 shadow of commit
`8466115e89fb639063fa07728493b98b42a86e06` exposed a defect in the original
liveness-only probe: workers could remain alive through model loading without ever
reaching a training update or maximum-shape evaluation, yet still be accepted. That
success rule is superseded.

The second H20 shadow of commit
`a28a3bde4cde3785fb0e037e8f486e2d86b12c84` verified the full single-worker training
and evaluation envelope, but exposed two remaining defects:

- raw one-minute load average was subtracted as though it were an exact count of
  occupied CPU workers, so a `387.5` load on a 384-logical-CPU server reduced eight
  idle H20 devices to one selected GPU and prevented any multi-worker candidate;
- completed phase markers were accepted while the worker return code was still
  `null`, after which controller cleanup terminated the process.

The current selector keeps load average only as diagnostic provenance, measures CPU
occupancy and worker demand directly, and requires clean zero exits.

## Phase and exit contract

Probe contract version 2 requires every worker to emit all of:

```text
model_loaded
training_peak_completed
evaluation_peak_completed
probe_complete
```

A process that merely starts, loads the model, or remains alive is not capacity
evidence. Missing phases make the candidate fail. After `probe_complete`, every
worker must exit with code zero within a bounded grace interval, and its entire
process group must disappear. SIGTERM/SIGKILL cleanup after phase completion is
recorded as controller intervention and makes the candidate fail.

## Resource-equivalent Countdown envelope

The Countdown adapter uses a dedicated engineering-only worker. It preserves the
registered resource-driving shapes:

- the exact model, dtype, LoRA parameterization, and maximum sequence length;
- the offline training micro-batch and full gradient-accumulation count;
- the populated 16-negative bank and current-policy near/far selection path;
- one real optimizer update, including first-step optimizer-state allocation;
- the registered evaluation batch size;
- the maximum configured pass@k (`64`) and generation length;
- the same generation implementation and temporary generation context.

It reduces only outer repetition: the evaluation envelope contains exactly one full
registered batch rather than the complete 500-example validation set. This keeps the
peak-producing tensor shape while fitting the approximately ten-minute probe budget.
The probe never reads the test split and never writes a scientific result.

Training and post-hoc evaluation are measured as separate lifecycle phases. The
training model is checkpointed and released before a fresh non-trainable evaluation
model executes pass@64, matching the hardened runtime more closely than retaining the
optimizer during evaluation.

## Measured CPU capacity

Linux load average is not used as CPU capacity. It includes runnable and
uninterruptible tasks and cannot identify how many compute cores are actually
available to the GPU workload.

Selector policy version 2 records three separate quantities:

1. **Worker CPU demand.** Process-tree user and system CPU seconds are sampled during
   the phase-complete envelope and divided by elapsed wall time to obtain average CPU
   cores used by the worker or candidate.
2. **System CPU occupancy.** Per-CPU `/proc/stat` deltas are measured over the same
   window for the process-visible CPU affinity set. `iowait` is treated as available
   compute capacity rather than busy CPU execution.
3. **External occupancy.** Candidate/worker CPU demand is subtracted from concurrent
   system busy-core accounting, producing an estimate of CPU demand external to the
   probe.

The reserved CPU demand per worker is:

```text
reserved_cpu_cores = max(minimum_cpu_cores_per_worker,
                         measured_cpu_cores_per_worker
                         * per_worker_cpu_safety_factor)
```

The CPU capacity ceiling is `logical_cpu_count * cpu_fraction`. Worker capacity is
computed after subtracting measured external busy cores. During candidate validation,
measured candidate CPU demand is projected over the complete selected GPU pool and
added back to concurrently measured external occupancy. A candidate is rejected when
that projected total exceeds the CPU ceiling.

A short `/proc/stat` sample is also used for cache revalidation. One-minute load
average remains in `RUNTIME_SELECTION.json` only for diagnosis.

## VRAM and host-memory capacity

For one eligible GPU, the selector records:

- free VRAM before and during the worker;
- worker-reported CUDA peak allocated/reserved memory for completed phases;
- process-tree host RSS;
- process-tree CPU seconds and average CPU cores;
- process-visible system busy cores and estimated external busy cores;
- effective host/cgroup memory after headroom.

The single-worker VRAM input is the maximum of parent-observed incremental VRAM and
the worker-reported phase peak. The initial slot candidate is bounded by VRAM,
measured host memory, measured CPU capacity, task count, and
`--max-slots-per-gpu`. The maximum-slot argument is a search ceiling, never the
selected value.

## Candidate validation

Every candidate runs the same resource-equivalent phase worker concurrently on one
GPU. A candidate fails on any of:

- missing required phase from any worker;
- worker nonzero exit or controller-forced process-group cleanup;
- lingering process-group descendants;
- CUDA OOM signature;
- free-VRAM floor violation;
- projected host RSS above effective host/cgroup capacity;
- projected full-pool CPU demand above measured CPU capacity;
- global probe-budget exhaustion.

The selector backs off through a small descending candidate sequence. If no
candidate above one passes, the already phase-complete single-worker envelope can be
used as a conservative one-slot runtime schedule. Such a result does not validate
the multi-worker optimization. The selector does not claim a global throughput
optimum.

## Budget and defaults

```text
--single-probe-seconds 240
--validation-probe-seconds 300
--probe-budget-seconds 600
--max-slots-per-gpu 8
--cpu-fraction 0.85
--per-worker-cpu-safety-factor 1.5
--minimum-cpu-cores-per-worker 1.0
```

The phase worker exits as soon as its envelope and cleanup finish; the probe
durations are upper bounds. Candidate validation is also bounded by the remaining
global deadline.

## Runtime artifacts

`RUNTIME_SELECTION.json` schema version 3 records:

- source, workload, selector-policy, and machine identity;
- selector policy version and probe contract version;
- required and completed phases for every worker;
- worker return codes and controller-intervention status;
- preserved phase-state files and worker logs;
- parent-observed and worker-reported VRAM peaks;
- host RSS and measured worker/system CPU accounting;
- VRAM, host, CPU, task, and configured capacity limits;
- candidate outcomes, backoff decisions, and selected `slots_per_gpu`;
- `scientific_matrix_changed: false`.

A failed single-worker envelope records `mode: failed` with `selection: null` and
never starts the scientific scheduler. Cached liveness-only or raw-load-average
selections are invalid because selector policy version, schema, policy fingerprint,
and placement-selector source hash differ.

## Limitations

- single node and homogeneous selected GPU pool only;
- one GPU required per independent task;
- measured CPU is a capacity gate, not affinity or NUMA tuning;
- no automatic DDP/TP/FSDP selection;
- no dynamic scaling after launch;
- capacity-oriented, not throughput-optimality evidence;
- a new exact-head H20 shadow with a real candidate above one is required before
  merge or default use.
