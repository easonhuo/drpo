# Runtime GPU placement autotuning V1

## Purpose

The existing E8 resource path can identify visible, idle, sufficiently free GPUs,
but it cannot determine how many independent single-GPU tasks should share each
GPU. V1 closes that gap without becoming a distributed-training framework.

The task launcher remains responsible for declaring its topology. V1 supports only:

```text
one independent task -> one GPU
one GPU -> one or more independent tasks
```

DDP, tensor parallelism, FSDP, and other multi-GPU strategies remain fixed by the
workload and are outside this selector.

## Cold-probe budget

The default target is under ten minutes:

1. static machine snapshot: seconds;
2. one-worker representative probe: bounded interval;
3. one or more bounded candidate validations;
4. final selection and cleanup.

The probe runner enforces a hard deadline. It never continues searching after the
budget expires. A timed-out or inconclusive search falls back to the largest already
validated candidate, or one slot per GPU when no higher candidate is validated.

## Capacity derivation

For one eligible GPU, V1 records free memory before the representative probe and the
minimum free memory observed during the probe. Their difference is the measured
incremental peak VRAM for one worker.

The reserved memory per worker is:

```text
ceil(measured_peak_vram * per_worker_safety_factor)
```

The initial candidate is bounded by:

- free VRAM after configured headroom;
- measured reserved memory per worker;
- host-memory capacity;
- remaining task count;
- configured maximum slots per GPU.

The maximum-slot setting is a safety ceiling, not the chosen value. The chosen value
must come from the measured capacity and a successful concurrent validation.

## Candidate validation

The selector validates the derived candidate on one GPU by launching that many
isolated representative workers. A candidate fails when any of the following occurs:

- an OOM signature is present;
- a worker exits nonzero;
- a worker cannot start;
- free VRAM crosses the safety floor;
- the candidate does not remain live for the required validation interval;
- the global probe deadline is exhausted before validation completes.

On failure, V1 backs off through a bounded descending sequence. It does not enumerate
every possible concurrency and does not claim to find a global throughput optimum.

## Runtime output

`RUNTIME_SELECTION.json` records:

- source and machine identity;
- eligible and rejected devices;
- measured single-worker peak VRAM;
- candidate validation records;
- selected `slots_per_gpu`;
- total runtime slots;
- per-GPU slot expansion;
- elapsed probe time and timeout/fallback reason;
- `scientific_matrix_changed: false`.

The scientific runner receives the original frozen configuration. Runtime placement
is supplied separately to the parent scheduler.

## Resume and reuse

A completed selection may be reused only when workload fingerprints, machine static
identity, candidate GPU identities, and selector policy match. Dynamic availability
must still be revalidated before launch. V1 does not reuse a selection across model,
configuration, GPU model, or scientific-input changes.

## Limitations

- single-node only;
- one GPU required per independent task;
- capacity-oriented, not a global throughput-knee search;
- no automatic DDP/TP/FSDP selection;
- no dynamic scaling after launch;
- real-hardware acceptance remains required.
