# Runtime resource autotuning V1: E7/E8 opt-in usage

## Status and boundary

V1 is an opt-in runtime-capacity guard for two existing DRPO workloads. It does
not replace the fixed launchers and does not change any scientific variable.

- **E7:** runs one isolated representative branch for a bounded interval, measures
  process-tree peak RSS, then selects the active subprocess count from CPU,
  measured host RAM, cgroup limits, task count, a configured cap, and a bounded
  growth factor.
- **E8:** selects visible and sufficiently idle GPU devices that satisfy a
  configured free-VRAM floor. Host RAM also limits the number of simultaneous GPU
  workers. V1 keeps exactly one process per selected GPU.

V1 is deliberately conservative. E7 V1 is a capacity model, not a multi-point
throughput-knee search. E8 V1 uses a configured VRAM floor plus `nvidia-smi`
snapshot; it does not yet measure training/evaluation phase peaks automatically.
Those limitations are written into every `RUNTIME_SELECTION.json`.

Existing fixed E7/E8 code and configs remain unchanged.

## Inspect the machine

```bash
python scripts/probe_runtime_resources.py \
  --output /tmp/drpo_machine_snapshot.json
```

The snapshot includes the process-visible CPU count, host memory, effective
cgroup memory limit/current usage, swap, and visible GPU memory/utilization.

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
command validates and reuses the exact cached selection. The dedicated probe seed
is `990001` by default and is not part of the scientific branch matrix.

The CLI keeps `--probe-steps` as an operator-requested floor, then derives an
effective isolated probe horizon of at least two frozen trainer evaluation
intervals. This prevents a fast probe from finishing before the trainer has any
evaluation history. The actual wall-clock probe remains bounded by
`--probe-seconds`; formal branch horizons and evaluation rules are unchanged. The
CLI reports both requested and effective probe steps.

The convenience shell entrypoint uses the same defaults:

```bash
bash scripts/run_e7_canonical_exp_horizon_joint_auto_one_click.sh
```

Important runtime controls:

```text
--fallback-workers 60
--probe-steps 20000                 # requested floor; may be raised for safety
--probe-seconds 120
--cpu-fraction 0.85
--memory-headroom-fraction 0.15
--per-worker-safety-factor 1.20
--max-growth-factor 3.0
--max-workers <optional hard cap>
```

The actual E7 branch configs, seeds, methods, coefficients, horizons, evaluation,
and thread environment remain those of the existing registered runner.

## E8 opt-in run

Use a new work directory. The original frozen sweep config remains unchanged and
is still passed to calibration and worker subprocesses.

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
  --gpus 0,1,2,3,4,5,6,7 \
  --required-free-gpu-memory-gib 8 \
  --required-host-memory-gib-per-gpu 4
```

The VRAM and host-RAM floors are runtime safety policy, not scientific
hyperparameters. Set them from a verified prior peak or a conservative operator
limit. V1 will reject busy, invisible, or insufficient-memory devices and will
never place two training processes on one GPU.

The wrapper changes only the parent scheduler's active slot count in memory. The
original config file and all worker/calibration subprocess inputs remain frozen.

## Artifacts

Each opt-in work directory contains:

```text
RUNTIME_SELECTION.json
_runtime_resource_probe/       # E7 only; small probe evidence
```

`RUNTIME_SELECTION.json` records:

- adapter and workload fingerprint;
- exact source commit/worktree state when available;
- CPU/RAM/cgroup/GPU snapshot;
- selected schedule and rejected constraints;
- probe or free-memory-gate evidence;
- fallback and known limitations;
- `scientific_matrix_changed: false`.

The normal E7/E8 runner continues to own branch identities, summaries, resume
semantics, and scientific artifacts.

## Failure and rollback

The auto entrypoints fail closed if no safe schedule can be selected. They do not
silently change batch size, methods, seeds, or model settings.

Rollback is immediate: stop invoking the auto entrypoints and use the unchanged
fixed E7/E8 launchers. Do not delete the selection artifact or failed probe log.

## Validation commands

```bash
python -m py_compile \
  src/drpo/runtime_resource_autotune.py \
  src/drpo/runtime_resource_adapters.py \
  scripts/probe_runtime_resources.py \
  scripts/run_e7_canonical_exp_horizon_joint_auto.py \
  scripts/run_countdown_e8_oracle_offline_v2_taper_auto.py

pytest -q \
  tests/test_runtime_resource_autotune.py \
  tests/test_runtime_resource_adapters.py \
  tests/test_e7_runtime_resource_auto.py \
  tests/test_e7_canonical_exp_horizon_joint.py \
  tests/test_countdown_e8_oracle_offline_v2_taper_sweep.py
```

Unit tests and CI are engineering evidence only. E7/E8 real-hardware readiness
requires an actual isolated server/GPU shadow run; it is not inferred from tests.
