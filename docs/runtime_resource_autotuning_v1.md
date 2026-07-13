# Runtime resource autotuning V1: E7/E8 opt-in usage

## Status and boundary

The runtime-resource path is opt-in and does not replace fixed launchers or change
scientific variables.

- **E7 CPU:** runs one isolated representative branch for a bounded interval,
  measures process-tree peak RSS, then selects active subprocess count from CPU,
  measured host RAM, cgroup limits, task count, a configured cap, and a bounded
  growth factor.
- **E8 GPU, historical `GOV-RUNTIME-RESOURCE-AUTOTUNE-01`:** selects visible and
  sufficiently idle GPU devices that satisfy configured VRAM and host-RAM gates. It
  remains available as the conservative one-process-per-GPU implementation.
- **E8 GPU placement, `GOV-RUNTIME-GPU-PLACEMENT-AUTOTUNE-02`:** adds a bounded
  representative GPU probe and automatically selects how many independent
  one-GPU tasks may share each selected GPU. The maximum-slots argument is a safety
  ceiling, not the selected value.

Neither path is a general distributed scheduler. The GPU-placement V1 does not
select DDP, tensor parallelism, FSDP, batch size, precision, or multi-node topology.
It is a bounded capacity and liveness selector, not a global throughput-optimality
claim. Detailed GPU placement semantics are documented in
`docs/runtime_gpu_placement_autotuning_v1.md`.

Existing fixed E7/E8 scientific code and configs remain unchanged.

## Inspect the machine

```bash
python scripts/probe_runtime_resources.py \
  --output /tmp/drpo_machine_snapshot.json
```

The snapshot includes process-visible CPU count, host memory, effective cgroup
memory limit/current usage, swap, and visible GPU memory/utilization.

## E7 opt-in run

Use a **new work directory**. Do not point this entrypoint at the current fixed-60
run directory.

```bash
python scripts/run_e7_canonical_exp_horizon_joint_auto.py plan \
  --contract /root/d4rl2/configs/e7_canonical_contract_9task.json \
  --run-spec /root/d4rl2/configs/e7_canonical_9task_full_grid_run_spec_v1.json \
  --grid configs/e7_canonical_exp_horizon_joint_grid_v1.json \
  --work-dir outputs/e7/exp_horizon_joint_auto_run_001

python scripts/run_e7_canonical_exp_horizon_joint_auto.py run \
  --contract /root/d4rl2/configs/e7_canonical_contract_9task.json \
  --run-spec /root/d4rl2/configs/e7_canonical_9task_full_grid_run_spec_v1.json \
  --grid configs/e7_canonical_exp_horizon_joint_grid_v1.json \
  --work-dir outputs/e7/exp_horizon_joint_auto_run_001
```

The first command normally performs a bounded two-minute memory probe. The second
validates and reuses the exact cached selection. The dedicated probe seed is
`990001` by default and is not part of the scientific branch matrix.

The CLI keeps `--probe-steps` as an operator-requested floor, then derives an
effective isolated probe horizon of at least two frozen trainer evaluation
intervals. Wall-clock execution remains bounded by `--probe-seconds`; formal branch
horizons and evaluation rules are unchanged.

Important E7 controls:

```text
--fallback-workers 60
--probe-steps 20000
--probe-seconds 120
--cpu-fraction 0.85
--memory-headroom-fraction 0.15
--per-worker-safety-factor 1.20
--max-growth-factor 3.0
--max-workers <optional hard cap>
```

## E8 automatic single-GPU task placement

Use a new work directory. The original frozen sweep config is still passed to
calibration and scientific worker subprocesses.

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

Default placement controls:

```text
--required-free-gpu-memory-gib 8
--required-host-memory-gib-per-worker 4
--gpu-memory-headroom-fraction 0.12
--host-memory-headroom-fraction 0.15
--per-worker-vram-safety-factor 1.25
--max-slots-per-gpu 8
--single-probe-seconds 90
--validation-probe-seconds 120
--probe-budget-seconds 600
--probe-free-floor-gib 4
```

The selector first rejects unavailable, busy, or insufficient-memory devices. It
then runs a representative one-worker probe, derives a concurrency candidate from
measured VRAM and host-memory capacity, and validates a bounded descending candidate
set on one GPU. The cold-probe hard budget defaults to ten minutes. If no candidate
above one is validated, it records the reason and falls back to one slot per GPU.

`--max-slots-per-gpu` only bounds the search. It does not force that number of
workers. The chosen value is written to `RUNTIME_SELECTION.json` and must come from
probe evidence.

## Artifacts

Each GPU-placement work directory contains:

```text
RUNTIME_SELECTION.json
RUNTIME_SLOTS.json
_runtime_resource_probe/e8_gpu_placement/
```

`RUNTIME_SELECTION.json` records:

- adapter, selector policy, and workload fingerprint;
- source commit/worktree state when available;
- CPU/RAM/cgroup/GPU snapshot;
- measured single-worker incremental peak VRAM;
- candidate validation results and fallback reason;
- selected devices, `slots_per_gpu`, expanded slot list, and total concurrency;
- `scientific_matrix_changed: false`.

`RUNTIME_SLOTS.json` records the exact placement consumed by the parent scheduler.
The scientific runner continues to own cells, summaries, checkpoints, resume
semantics, and experiment outputs.

## Failure and rollback

The auto entrypoints fail closed on malformed identity, invalid placement, missing
resources, or unsafe probe outcomes. They do not silently change batch size,
methods, seeds, model settings, or evaluation.

Rollback is immediate: stop invoking the auto entrypoint and use the unchanged fixed
E8 runtime or the historical conservative path. Preserve selection files and failed
probe logs.

## Validation commands

```bash
python -m py_compile \
  src/drpo/runtime_resource_autotune.py \
  src/drpo/runtime_resource_adapters.py \
  src/drpo/runtime_gpu_placement_autotune.py \
  src/drpo/countdown_e8_oracle_offline_v2_taper_slot_runtime.py \
  scripts/probe_runtime_resources.py \
  scripts/run_e7_canonical_exp_horizon_joint_auto.py \
  scripts/run_countdown_e8_oracle_offline_v2_taper_auto.py

pytest -q \
  tests/test_runtime_resource_autotune.py \
  tests/test_runtime_resource_adapters.py \
  tests/test_runtime_gpu_placement_autotune.py \
  tests/test_e7_runtime_resource_auto.py \
  tests/test_e7_canonical_exp_horizon_joint.py \
  tests/test_countdown_e8_oracle_offline_v2_taper_sweep.py
```

Unit tests and CI are engineering evidence only. E7/E8 real-hardware readiness
requires an actual isolated server/GPU shadow run; it is not inferred from tests.
