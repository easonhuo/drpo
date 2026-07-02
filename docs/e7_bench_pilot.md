# EXT-H-E7-BENCH-01 Parallel Pilot Protocol

## Scope

This is a pilot substage under the existing `EXT-H-E7-BENCH-01` ID. It does not create a second top-level experiment. The pilot checks the benchmark implementation, runtime, artifact volume, paired method branching, rollout evaluation, and task/support/numerical audit plumbing.

The formal benchmark remains the registered nine-cell D4RL MuJoCo suite. Pilot output is never formal evidence and may not be used to select a new taper family or tune a task-specific coefficient.

## Pilot matrix

- Four development seeds: `200, 201, 202, 203`.
- Six methods: Positive-only, Signed, Global alpha, Reciprocal-Linear, Reciprocal-Quadratic, and Exponential.
- Taper coefficients are copied without D4RL retuning from the frozen C-U1 near-retention calibration.
- Two uploaded data cells:
  - `hopper-medium-minari-v0`: the exact uploaded `mujoco/hopper/medium-v0` file. Metadata identifies it as Minari/Hopper-v5, not D4RL Hopper-medium-v2. It is therefore plumbing-only and is not eligible for the formal nine-cell table.
  - `hopper-medium-expert-v2`: the exact uploaded D4RL-v2 legacy HDF5 cell.

The version distinction is deliberate. The code refuses to relabel the Minari medium file as D4RL and reports raw Hopper-v5 return for that pilot cell. A formal Hopper-medium result still requires a separately frozen exact D4RL dataset version.

## Parallel execution

The coordinator uses three fail-fast subprocess stages:

1. Two dataset-level canonical critics run concurrently.
2. Eight `(dataset, seed)` Positive-only checkpoints run concurrently.
3. Forty `(dataset, seed, method)` branches run concurrently from the corresponding identical Positive-only checkpoint.

This structure does not serialize either seeds or method branches. It preserves paired initialization because every branch verifies and loads the same Positive-only checkpoint for its dataset and seed. It also uses isolated outputs, so one failed method branch can be retried without rerunning the other 39 branches.

The default CPU allocation is stage-specific: 2 critic workers × 64 threads, 8 Positive-only workers × 32 threads, and 40 branch workers × 8 threads. Peak allocation is 320 threads, leaving operating-system and I/O headroom on a 384-core server. Resume granularity is `(dataset, seed, method)`.

Before starting the 20k-step critics, the coordinator performs one SHA-256 pass per dataset, checks that the machine exposes at least the registered 320 peak CPU threads, opens both Gymnasium/MuJoCo evaluation environments, and verifies dataset/environment observation and action dimensions. Worker processes reuse the coordinator's immutable dataset manifest and check path, size, and mtime rather than launching 40 simultaneous hashes of the same HDF5 files.

The formal nine-cell registration freezes the same topology: `task_seed_method` is the branch parallel unit, both serial seed and serial method loops are forbidden, and every branch starts from an identical per-task-seed Positive-only checkpoint. Formal launch remains fail-closed until exact formal seeds, D4RL versions, base algorithm, optimizer, and full budgets are registered.

## Budgets and interpretation

The pilot uses 20k critic steps, 20k Positive-only steps, and 40k steps per branch. These are pilot budgets, not convergence claims. Only NaN/Inf can stop a branch early. Terminal audit still separates:

- task-performance collapse;
- support or variance-boundary events;
- NaN/Inf numerical failure;
- persistent/slow drift or unresolved fixed-horizon state.

A pilot method may look promising, but the result cannot establish a stable terminal ranking, a universal controller winner, or superiority over Positive-only.

## Commands

The runtime environment must provide the E7-Q2 PyTorch/HDF5 stack plus `gymnasium` with MuJoCo support for `Hopper-v4` and `Hopper-v5`. The runner fails before long critic training when this rollout dependency is unavailable.

After extracting the registered dataset bundle into one directory:

```bash
python3 scripts/run_e7_bench.py inspect \
  --dataset-root "$DATASET_ROOT" \
  --config configs/e7_bench_pilot.yaml

python3 scripts/run_e7_bench.py run \
  --mode pilot \
  --dataset-root "$DATASET_ROOT" \
  --work-dir "$WORK_DIR" \
  --config configs/e7_bench_pilot.yaml \
  --resume
```

`DATASET_ROOT` and `WORK_DIR` are runtime environment variables chosen by the operator; the repository stores exact internal relative paths and SHA-256 values, not machine-specific absolute paths.
