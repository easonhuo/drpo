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
workload and are outside this selector. V1 also requires a homogeneous selected GPU
pool; heterogeneous-device packing is deferred rather than guessed.

## Cold-probe budget

The default target is under ten minutes:

1. static machine snapshot: seconds;
2. one-worker representative probe: bounded interval;
3. one or more bounded candidate validations;
4. final selection and cleanup.

The probe runner enforces a hard deadline. It never continues searching after the
budget expires. A timed-out or inconclusive search falls back to one slot per GPU
unless a higher candidate has already been validated.

## Capacity derivation

For one eligible GPU, V1 records:

- free VRAM before and during the representative worker;
- process-tree host RSS for that worker;
- process-visible CPU count and current load;
- effective host/cgroup memory available after headroom.

The reserved per-worker capacities are:

```text
reserved_vram = ceil(measured_peak_vram * vram_safety_factor)
reserved_host = max(configured_host_floor,
                    ceil(measured_peak_host_rss * host_safety_factor))
```

The initial slot candidate is the minimum of:

- VRAM capacity after headroom;
- measured host-memory capacity;
- process-count CPU capacity after current load;
- remaining task count per selected device;
- configured maximum slots per GPU.

The maximum-slot setting is a bounded-search safety ceiling, not the chosen value.
The chosen value must come from measured capacity and a successful concurrent probe.

## Candidate validation

The selector validates the derived candidate on one GPU by launching that many
isolated representative workers. It records concurrent VRAM and process-tree host
RSS, then projects that host usage across the selected homogeneous GPU pool.

A candidate fails when any of the following occurs:

- an OOM signature is present;
- a worker exits nonzero or cannot start;
- free VRAM crosses the safety floor;
- projected host RSS exceeds effective host/cgroup capacity;
- total worker count exceeds the CPU process-count capacity;
- the candidate does not remain live for the required interval;
- the global probe deadline is exhausted.

On failure, V1 backs off through a small descending candidate sequence. It does not
enumerate every concurrency and does not claim a global throughput optimum.

## Runtime output

`RUNTIME_SELECTION.json` records:

- source, workload, selector-policy, and machine identity;
- measured single-worker peak VRAM and host RSS;
- reserved per-worker VRAM and host-memory values;
- CPU, host-memory, VRAM, task, and configured capacity limits;
- candidate validation and projected-host checks;
- selected devices, `slots_per_gpu`, total runtime slots, and expanded slot list;
- elapsed probe time and timeout/fallback reason;
- `scientific_matrix_changed: false`.

The scientific runner receives the original frozen configuration. Runtime placement
is supplied separately to the parent scheduler.

## Resume and reuse

A completed selection may be reused only when workload fingerprint, machine static
identity, selected device identity, and selector policy match. Before reuse, V1
rechecks current GPU free memory, effective host capacity, and CPU process capacity.
It does not reuse a selection across model, configuration, GPU profile, or
scientific-input changes.

## Limitations

- single-node only;
- homogeneous GPU pool only;
- one GPU required per independent task;
- CPU is a process-count gate, not thread-affinity or NUMA autotuning;
- capacity-oriented, not a global throughput-knee search;
- no automatic DDP/TP/FSDP selection;
- no dynamic scaling after launch;
- real-hardware acceptance remains required.
