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

## V1 scope

- static eligible-GPU filtering from visibility, utilization, free VRAM, host RAM,
  cgroup memory, current CPU load, and configured candidate IDs;
- homogeneous selected GPU pool only;
- one representative single-worker probe measuring incremental peak VRAM and
  process-tree host RSS;
- automatic CPU, host-memory, VRAM, task-count, and safety-ceiling capacity limits;
- bounded same-GPU validation of a small candidate set derived from measured capacity;
- automatic `slots_per_gpu`, total slot count, and per-GPU placement output;
- fail-closed cleanup of all probe processes and preservation of probe logs;
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
  evaluation changes;
- CPU thread affinity, NUMA placement, or dataloader-worker autotuning;
- online migration or dynamic slot changes after the scientific run starts;
- claims of global throughput optimality;
- modifying `docs/handoff.md`, `experiments/registry.yaml`, the formal execution
  channel, or any closed-stage protected file.

## Selection contract

The workload adapter declares a fixed topology of one GPU per independent task and
provides a representative worker command. The selector:

1. records the machine snapshot and eligible homogeneous devices;
2. runs one worker for a bounded interval and measures incremental peak VRAM plus
   process-tree host RSS;
3. derives a candidate from measured VRAM/host usage, CPU/load capacity, safety
   factors, remaining task count, and an operator safety ceiling;
4. validates the candidate on one GPU and projects measured host RSS across the
   selected device pool;
5. backs off through a bounded candidate sequence when validation fails;
6. writes `RUNTIME_SELECTION.json` before the scientific scheduler starts.

The probe budget is bounded. If no candidate above one is validated within the
budget, selection falls back to one slot per GPU with an explicit reason. A probe is
engineering evidence only.

## Acceptance

- deterministic tests cover measured capacity derivation, candidate backoff,
  timeout/fallback, OOM/nonzero rejection, host-memory and CPU limits,
  heterogeneous-device rejection, per-GPU slot expansion, cache revalidation, CLI
  compatibility, and scientific configuration preservation;
- Python compilation, Ruff, full pytest, handoff authority verification, formal
  execution-channel validation, governance inventory, and governance-stage validation
  pass on the exact PR head;
- a real E8 server shadow must confirm selection-file creation, no orphan workers,
  no immediate OOM, and the expected per-GPU process count before default use;
- no real-hardware result is claimed by code or CI alone.

## Rollback

1. Stop invoking the GPU-placement auto entrypoint.
2. Continue using the unchanged fixed E8 runtime or the previous one-process-per-GPU
   auto path.
3. Preserve `RUNTIME_SELECTION.json`, probe logs, and failed probe evidence.
4. Revert this claim's files as one reviewed change if the feature is removed.
5. Never delete scientific outputs, completed cells, or historical failure evidence.
