# Runtime resource autotuning V1: E7/E8 opt-in usage

## Status and boundary

The runtime-resource path is opt-in and does not replace fixed launchers or change
scientific variables.

- **E7 CPU:** measures one representative process-tree RSS and selects active
  subprocess count from CPU, load, host/cgroup memory, task count, and configured
  safety limits.
- **Historical E8 GPU path (`GOV-RUNTIME-RESOURCE-AUTOTUNE-01`):** filters visible,
  idle, sufficiently free devices and remains the conservative one-process-per-GPU
  implementation.
- **Measured-CPU phase-aware E8 placement
  (`GOV-RUNTIME-GPU-PLACEMENT-AUTOTUNE-02`):** runs a resource-equivalent workload
  envelope and accepts `slots_per_gpu` only after every concurrent worker completes
  the required training and maximum-shape evaluation phases, exits zero, and leaves
  no process-group descendants.

Neither path selects DDP, tensor parallelism, FSDP, batch size, precision, or
multi-node topology. Existing scientific configs and fixed launchers remain
unchanged.

## E8 usage

Use a new work directory:

```bash
python scripts/run_countdown_e8_oracle_offline_v2_taper_auto.py \
  --model_path /path/to/Qwen2.5-0.5B-Instruct \
  --work_dir outputs/e8/v2_taper_auto_run_001 \
  --bank /path/to/bank.jsonl \
  --val /path/to/validation.jsonl \
  --test /path/to/test.jsonl \
  --global_calibration /path/to/global_calibration.json \
  --base_config configs/countdown_e8_base_rl_replay_0p5b.yaml \
  --sweep_config configs/countdown_e8_oracle_offline_v2_taper_sweep_0p5b.yaml \
  --gpus 0,1,2,3,4,5,6,7
```

Default controls:

```text
--required-free-gpu-memory-gib 8
--required-host-memory-gib-per-worker 4
--gpu-memory-headroom-fraction 0.12
--host-memory-headroom-fraction 0.15
--per-worker-host-memory-safety-factor 1.25
--cpu-fraction 0.85
--per-worker-cpu-safety-factor 1.5
--minimum-cpu-cores-per-worker 1.0
--per-worker-vram-safety-factor 1.25
--max-slots-per-gpu 8
--single-probe-seconds 240
--validation-probe-seconds 300
--probe-budget-seconds 600
--probe-free-floor-gib 4
```

`--required-host-memory-gib-per-gpu` remains a compatibility alias. The configured
host value is a minimum; measured process-tree RSS with safety factor may impose a
larger reservation.

## Phase-complete and clean-exit probe

The dedicated Countdown resource worker preserves the real training and generation
shapes while reducing only outer repetition. Every worker must emit:

```text
model_loaded
training_peak_completed
evaluation_peak_completed
probe_complete
```

The training phase performs one full registered optimizer update with the frozen
micro-batch, complete gradient accumulation, sequence length, and negative-bank
path. The evaluation phase releases the training model and optimizer, loads a fresh
non-trainable adapter model, and performs one full registered evaluation batch with
the maximum configured pass@k and generation length. The test split is not used by
the probe.

Model loading or timed liveness alone is insufficient. A phase-incomplete
single-worker probe records a failed selection and stops before the scientific
scheduler. After `probe_complete`, each worker must exit zero inside the bounded
grace interval and its full process group must disappear. Controller SIGTERM/SIGKILL
cleanup is recorded as a failed candidate rather than a clean exit.

The parent observes total GPU free-memory changes, while each worker also records
CUDA peak allocated/reserved memory for its completed phases. Capacity derivation
uses the larger single-worker value, preventing a short peak between `nvidia-smi`
polls from being ignored.

## Measured CPU capacity

Raw one-minute load average is retained as diagnostic provenance only. It is not
subtracted as an already-consumed worker count.

The selector measures:

- process-tree CPU seconds and average worker CPU cores during the real envelope;
- process-visible system busy cores from per-CPU `/proc/stat` deltas;
- estimated external CPU occupancy after subtracting the probe worker's demand;
- projected worker demand across the complete selected GPU pool during candidate
  validation.

`iowait` is not charged as busy compute execution. A candidate passes the CPU gate
only when measured external occupancy plus projected full-pool worker demand remains
within `logical_cpu_count * cpu_fraction`. The same measured capacity model is used
for cache revalidation.

## Artifacts

A phase-aware work directory contains:

```text
RUNTIME_SELECTION.json
RUNTIME_SLOTS.json
_runtime_resource_probe/e8_gpu_placement/
```

The selection records schema version 3, selector policy version 2, probe contract,
required/completed phases, worker return codes, controller-intervention status,
archived phase-state files, VRAM/host-memory/CPU measurements, capacity limits,
candidate outcomes, exact placement, and `scientific_matrix_changed: false`.

Selections created by either the old liveness-only probe or the raw-load-average
selector are not reusable because the selector implementation hash, policy version,
policy fingerprint, and workload fingerprint differ.

## Failure and rollback

The auto entrypoint fails closed on malformed identity, missing phases, unclean
worker exit, invalid placement, missing resources, heterogeneous GPUs, OOM, or unsafe
VRAM/RAM/CPU capacity. It does not change methods, seeds, model settings, batch
shapes, horizons, or scientific evaluation.

Rollback is immediate: stop invoking the auto entrypoint and use the unchanged fixed
runtime or historical conservative path. Preserve all probe evidence.

## Validation commands

```bash
python -m py_compile \
  src/drpo/runtime_gpu_placement_autotune.py \
  src/drpo/runtime_gpu_placement_autotune_v2.py \
  src/drpo/countdown_e8_oracle_offline_v2_taper_resource_probe.py \
  src/drpo/countdown_e8_oracle_offline_v2_taper_slot_runtime.py \
  scripts/run_countdown_e8_oracle_offline_v2_taper_auto.py

pytest -q \
  tests/test_runtime_gpu_placement_autotune.py \
  tests/test_runtime_gpu_placement_autotune_v2.py \
  tests/test_countdown_e8_taper_resource_probe.py \
  tests/test_e8_gpu_placement_auto_cli.py
```

Tests and CI are engineering evidence only. Exact-head real-H20 shadow acceptance
with an actual candidate above one is still required before merge or default use.
