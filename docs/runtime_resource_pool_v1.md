# Runtime Resource Pool V1

**Claim:** `GOV-RUNTIME-RESOURCE-POOL-01`  
**Status:** stacked implementation on Draft PR dependency `#65`; server acceptance pending  
**Scientific impact:** none  
**Default-policy impact:** none

## Purpose

E7 and E8 may run concurrently on the same server. Without an explicit placement
boundary, each launcher observes the full host and can consume CPUs also needed by
the other workload. Runtime autotuning then estimates capacity inside a resource
domain that is not actually private.

Resource Pool V1 adds an explicit Linux CPU-affinity boundary and an auditable GPU
pool identity before the existing launcher starts. It does not replace autotuning:
E7 measured-CPU V2 still selects worker count, but it now measures only inside the
pool inherited from the wrapper.

## Components

```text
src/drpo/runtime_resource_pool.py
scripts/run_with_resource_pool.py
```

The core module only parses, applies, verifies, and identifies resource pools. The
wrapper delegates to existing launchers without changing their arguments.

## CPU pool syntax

```text
0-31
0-31,64-95
3,7,11-15
```

Rules:

- CPU IDs are non-negative integers;
- ranges are inclusive and ascending;
- empty tokens, descending ranges, malformed IDs, and duplicates fail closed;
- an explicit pool must be a subset of the wrapper process's inherited affinity;
- the wrapper calls `os.sched_setaffinity(0, ...)` and verifies exact equality;
- child processes inherit the resulting affinity;
- omitting `--cpu-pool` records the existing inherited affinity without widening it.

The wrapper never changes affinity of unrelated processes.

## GPU pool contract

GPU placement already belongs to the delegated launcher in most E8 paths. The
wrapper supports two enforcement modes:

```text
launcher_argument
cuda_visible
```

`launcher_argument` is the default. A declared `--gpu-pool` requires the delegated
command to contain one identical `--gpus` value. This is appropriate for launchers
that use physical GPU IDs and inspect `nvidia-smi`.

`cuda_visible` sets `CUDA_VISIBLE_DEVICES` before delegation. It is only appropriate
for commands whose GPU identity is defined entirely through CUDA visibility. It must
not be used around a launcher that expects physical `nvidia-smi` IDs unless its
arguments are adjusted and separately validated.

The wrapper does not auto-select GPUs.

## Immutable identity

Every invocation uses one shared identity path, normally:

```text
<work_dir>/RESOURCE_POOL.json
```

The identity contains:

```text
schema_version
source = explicit_cli | inherited_affinity
inherited_cpu_ids
requested_cpu_ids
effective_cpu_ids
cpu_count
requested_gpu_ids
gpu_enforcement
pool_digest
```

The document has no timestamp or command-specific fields. Plan and run may reuse it
only when the entire identity matches exactly. A different CPU set, source, GPU set,
or enforcement mode is rejected.

The wrapper also exports:

```text
DRPO_RESOURCE_POOL_DIGEST
DRPO_CPU_POOL
DRPO_GPU_POOL        # only when a GPU pool is declared
```

## E7 usage

Use the same pool identity and CPU list for plan and run:

```bash
python scripts/run_with_resource_pool.py \
  --cpu-pool 0-95,192-287 \
  --pool-identity /path/to/e7_work/RESOURCE_POOL.json \
  -- \
  python scripts/run_e7_ppo_w0_grid_pilot_auto.py plan \
    --contract /path/to/contract.json \
    --run-spec /path/to/run_spec.json \
    --grid /path/to/grid.json \
    --work-dir /path/to/e7_work

python scripts/run_with_resource_pool.py \
  --cpu-pool 0-95,192-287 \
  --pool-identity /path/to/e7_work/RESOURCE_POOL.json \
  -- \
  python scripts/run_e7_ppo_w0_grid_pilot_auto.py run \
    --contract /path/to/contract.json \
    --run-spec /path/to/run_spec.json \
    --grid /path/to/grid.json \
    --work-dir /path/to/e7_work \
    --resume
```

Measured-CPU V2 records the exact effective affinity in `RUNTIME_SELECTION.json`.
The selection digest is bound into `RUN_IDENTITY.json`, and run revalidation rejects
a changed affinity. `RESOURCE_POOL.json` adds the explicit operator-facing identity
that prevents accidentally invoking plan and run with different wrapper settings.

## E8 usage

For an E8 launcher that already accepts physical GPU IDs:

```bash
python scripts/run_with_resource_pool.py \
  --cpu-pool 96-191,288-367 \
  --gpu-pool 0,1,2,3,4,5,6,7 \
  --gpu-enforcement launcher_argument \
  --pool-identity /path/to/e8_work/RESOURCE_POOL.json \
  -- \
  python <existing_e8_launcher.py> \
    <existing arguments> \
    --gpus 0,1,2,3,4,5,6,7
```

The CPU IDs above are examples only. The real allocation must follow the server's
socket/NUMA topology and measured E7/E8 demand. V1 does not prescribe a split.

## Dry run

The wrapper can validate pool syntax, affinity, GPU-argument agreement, and immutable
identity without starting the delegated command:

```bash
python scripts/run_with_resource_pool.py \
  --cpu-pool <cpu-list> \
  --gpu-pool <gpu-list> \
  --pool-identity <path>/RESOURCE_POOL.json \
  --dry-run \
  -- \
  python <launcher.py> --gpus <same-gpu-list> ...
```

## Thread settings

Resource pooling and thread tuning are separate.

The contention incident showed that the current E8 process can create a very large
PyTorch/OpenMP thread population. A CPU pool prevents those threads from stealing CPU
execution from E7, but it does not guarantee that E8 is efficient inside its pool.
OMP/MKL/PyTorch thread limits require a separate controlled throughput scan. V1 does
not set or tune them.

## Failure and rollback

The wrapper exits before delegation on malformed IDs, unavailable CPUs, affinity
application mismatch, GPU argument mismatch, or pool-identity mismatch.

Rollback is immediate: stop using the wrapper and return to the existing inherited
process affinity. Preserve `RESOURCE_POOL.json`, runtime selections, revalidations,
and contention evidence. Resource-pool tests or shadows are engineering evidence,
not scientific results.
