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

## Corrective phase contract

The first H20 shadow of commit
`8466115e89fb639063fa07728493b98b42a86e06` exposed a defect in the original
liveness-only probe: workers could remain alive through model loading without ever
reaching a training update or maximum-shape evaluation, yet still be accepted.
That success rule is superseded.

Probe contract version 2 requires every worker to emit all of:

```text
model_loaded
training_peak_completed
evaluation_peak_completed
probe_complete
```

A process that merely starts, loads the model, or remains alive is not capacity
evidence. Missing phases make the candidate fail. Missing phases in the
single-worker envelope fail closed and no runtime schedule is emitted.

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

## Capacity derivation

For one eligible GPU, the selector records:

- free VRAM before and during the worker;
- worker-reported CUDA peak allocated/reserved memory for completed phases;
- process-tree host RSS;
- process-visible CPU count and current load;
- effective host/cgroup memory after headroom.

The single-worker VRAM input is the maximum of parent-observed incremental VRAM and
the worker-reported phase peak. The initial slot candidate is bounded by VRAM,
measured host memory, CPU/load capacity, task count, and
`--max-slots-per-gpu`. The maximum-slot argument is a search ceiling, never the
selected value.

## Candidate validation

Every candidate runs the same resource-equivalent phase worker concurrently on one
GPU. A candidate fails on any of:

- missing required phase from any worker;
- CUDA OOM signature;
- nonzero exit or launch failure;
- free-VRAM floor violation;
- projected host RSS above effective host/cgroup capacity;
- total worker count above CPU process capacity;
- global probe-budget exhaustion.

The selector backs off through a small descending candidate sequence. If no
candidate above one passes, the already phase-complete single-worker envelope is
used. It does not claim a global throughput optimum.

## Budget and defaults

```text
--single-probe-seconds 240
--validation-probe-seconds 300
--probe-budget-seconds 600
--max-slots-per-gpu 8
```

The phase worker exits as soon as its envelope finishes; these durations are upper
bounds. Candidate validation is also bounded by the remaining global deadline.

## Runtime artifacts

`RUNTIME_SELECTION.json` schema version 2 records:

- source, workload, selector-policy, and machine identity;
- probe contract version and required phases;
- completed phases for every worker;
- preserved phase-state files and worker logs;
- parent-observed and worker-reported VRAM peaks;
- measured host RSS and capacity limits;
- candidate outcomes and selected `slots_per_gpu`;
- `scientific_matrix_changed: false`.

A failed single-worker envelope records `mode: failed` with `selection: null` and
never starts the scientific scheduler. Cached liveness-only selections are invalid
because the probe contract and resource-probe fingerprint changed.

## Limitations

- single node and homogeneous selected GPU pool only;
- one GPU required per independent task;
- CPU is a process-count gate, not affinity or NUMA tuning;
- no automatic DDP/TP/FSDP selection;
- no dynamic scaling after launch;
- capacity-oriented, not throughput-optimality evidence;
- a new exact-head H20 shadow is required before merge or default use.
