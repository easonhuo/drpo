# EXT-H-E7-BENCH-01 Long-Budget Parallel Pilot Protocol

## Scope

This is a `pilot` substage under the existing `EXT-H-E7-BENCH-01` ID. It does not create a second top-level experiment. The pilot checks the benchmark implementation, runtime, artifact volume, paired method branching, rollout evaluation, terminal-state audit, and method-specific parameter sensitivity under a D4RL-scale training budget.

The formal benchmark remains the registered nine-cell D4RL MuJoCo suite. Pilot output is not formal nine-task evidence and may not be used to select a new method family, tune a task-specific coefficient, or populate a formal method ranking. A later freeze update is required before any selected scalar can be promoted to a formal benchmark setting.

## Correction of the superseded short-budget design

The first implementation registered `20k` critic steps, a `20k` Positive-only checkpoint, and `40k` non-positive branches. That design was suitable only as an engineering smoke test. It had two scientific defects:

1. every stage was one fifth of the already audited E7-Q2 long-run budget; and
2. Positive-only stopped after `20k`, while the other methods received another `40k`, so the baseline and controlled methods did not have equal actor horizons.

No scientific pilot result was produced under that design. This protocol replaces it with the E7-Q2 long-run scale and an equal-horizon comparison:

- canonical critic: `100k` optimizer steps per dataset;
- shared Positive-only warm-start: `100k` steps per `(dataset, seed)`;
- method continuation: `200k` steps for **all six methods**, including Positive-only;
- total actor horizon: `300k` steps for every compared method.

Only NaN/Inf numerical failure may stop a worker early. A fixed horizon is not treated as convergence.

## Pilot parameter-sweep matrix

- Two development seeds: `200, 201`. The seed count is intentionally reduced from four to two so the same server budget can cover more method-specific scalar settings.
- Six method families remain fixed: Positive-only, Signed, Global alpha, Reciprocal-Linear, Reciprocal-Quadratic, and Exponential.
- The follow-up pilot now expands these into 21 method variants: one Positive-only, one Signed, four Global-alpha scalars, four stronger Reciprocal-Linear coefficients, four stronger Reciprocal-Quadratic coefficients, and seven Exponential coefficients. The prior stronger-taper pilot showed Exponential around `c=8` had the clearest usable signal, while Reciprocal and Global-alpha also improved as their control strength increased; this matrix therefore keeps all non-positive families alive instead of prematurely reducing the paper comparison to Exponential-only.
- The old C-U1 near-retention scalars remain recorded in the config as provenance, but the active follow-up grid is a pilot sensitivity search centered on stronger taper settings; it does not authorize per-task D4RL retuning or formal coefficient promotion.
- Latest pilot interpretation is recorded only as tuning rationale: Exponential currently has the strongest stable signal, but Reciprocal and Global-alpha are still under-explored at stronger settings. No method ranking, formal D4RL table entry, or locked conclusion is promoted by this pilot update.
- Two uploaded data cells:
  - `hopper-medium-minari-v0`: the exact uploaded `mujoco/hopper/medium-v0` file. Metadata identifies it as Minari/Hopper-v5, not D4RL Hopper-medium-v2. It is plumbing/pilot-only and is not eligible for the formal nine-cell table.
  - `hopper-medium-expert-v2`: the exact uploaded D4RL-v2 legacy HDF5 cell.

The version distinction is deliberate. The code refuses to relabel the Minari medium file as D4RL and reports raw Hopper-v5 return for that pilot cell. A formal Hopper-medium result still requires a separately frozen exact D4RL dataset version.

## Parallel execution

The coordinator uses three fail-fast subprocess stages:

1. Two dataset-level canonical critics run concurrently: `2 × 64 = 128` CPU threads.
2. Four `(dataset, seed)` shared Positive-only warm-starts run concurrently: `4 × 64 = 256` CPU threads.
3. Ninety-two `(dataset, seed, method_variant)` equal-horizon continuations run concurrently: `92 × 4 = 368` CPU threads.

The third stage includes Positive-only itself. Every method variant loads the same 100k warm-start for its dataset and seed, creates a fresh optimizer, and receives the same 200k continuation budget. This removes the former 20k-versus-60k comparison asymmetry while allowing method-specific scalar search.

The taper definitions are fixed and explicit. For standardized Gaussian distance
`u = d / 5`, Reciprocal-Linear uses `1 / (1 + c u)`, Reciprocal-Quadratic uses
`1 / (1 + c u^2)`, and Exponential uses `exp(-c u)`. “Quadratic” here refers to
the squared standardized distance, which is the Gaussian surprisal-order proxy;
it is not the quartic form that would arise from reciprocal-squared-surprisal.

The peak registered allocation is 368 threads, leaving 16 threads of headroom on a 384-core server. Seeds and method variants are never executed by a top-level serial loop. Every worker has an isolated output directory, and shared aggregate files are written only by the coordinator.

The formal nine-cell registration freezes the same topology: `task_seed_method` is the continuation parallel unit, both serial seed and serial method loops are forbidden, Positive-only is an equal-horizon continuation branch, and each method starts from an identical per-task-seed Positive-only warm-start. Formal launch remains fail-closed until exact formal seeds, D4RL versions, base algorithm, optimizer, and full budgets are registered.

## Resume and stale-output protection

Resume identity is not based only on dataset, seed, and method names. Every run and worker binds:

- exact Pilot config SHA-256;
- exact E7-Q2 base-config SHA-256;
- runner and protocol versions;
- dataset SHA-256;
- stage-specific training budget;
- method variant identity and all scalar/taper parameters.

A work directory from the superseded `20k/20k/40k` design cannot be resumed under this protocol. The coordinator fails closed and requires a new work directory. Within a matching run identity, incomplete worker directories are preserved under a `_stale_worker_outputs` archive before that worker alone is retried.

If one worker fails, the coordinator terminates active peer subprocesses instead
of waiting for dozens of 200k-step workers to finish. Canonical critics and shared
warm-starts must complete their full frozen budgets before downstream branching.
Method continuations may stop early only for NaN/Inf; summaries record scheduled
and actually executed actor steps separately, so a numerical failure cannot be
misreported as a completed 300k path.

## Interpretation

The long budget makes this a meaningful scientific pilot rather than a plumbing smoke, but it still does not establish:

- a stable finite terminal state;
- a universal method ranking;
- superiority over Positive-only;
- formal nine-task D4RL performance;
- permission to change method families, tune per-task coefficients, or promote a scalar into the formal benchmark without a separate freeze update.

The terminal audit continues to report separately:

- task-performance collapse;
- support or variance-boundary events;
- NaN/Inf numerical failure;
- persistent/slow drift or unresolved fixed-horizon state.

## Commands

The runtime environment must provide the E7-Q2 PyTorch/HDF5 stack plus `gymnasium` with MuJoCo support for `Hopper-v4` and `Hopper-v5`. The runner fails before the 100k critics when this rollout dependency is unavailable. The shared warm-start worker is intentionally method-agnostic: method validation belongs to branch continuations only, because `warmstart-worker` does not accept a `--method` argument.

After extracting the registered dataset bundle into one directory, use a **new** work directory for this long-budget protocol:

```bash
python3 scripts/run_e7_bench.py inspect \
  --dataset-root "$DATASET_ROOT" \
  --config configs/e7_bench_pilot.yaml

python3 scripts/run_e7_bench.py run \
  --mode pilot \
  --dataset-root "$DATASET_ROOT" \
  --work-dir "$WORK_DIR" \
  --config configs/e7_bench_pilot.yaml
```

For interruption recovery, rerun the same command with `--resume`. `DATASET_ROOT` and `WORK_DIR` are runtime environment variables chosen by the operator; the repository stores exact internal relative paths and SHA-256 values, not machine-specific absolute paths.

## Micro-global and high-taper follow-up rationale from 2026-07-06 pilot

The preceding 23-variant pilot is still pilot evidence only and does not establish a formal method ranking. Its practical tuning signal was: reciprocal-linear and reciprocal-quadratic entered usable regions only after stronger coefficients, exponential remained the broadest stable family, and global-alpha values down to `0.05` still task-collapsed with support/variance-boundary symptoms.

The next follow-up therefore keeps the same pilot budget while shifting search regions again: global-alpha probes the micro range `0.001--0.03`, reciprocal-linear probes `40--80`, reciprocal-quadratic probes `32--64`, and exponential probes the higher range `14--24`. With two datasets and two seeds, the resulting pilot matrix remains `2 × 2 × 23 = 92` equal-horizon branch continuations.

This update remains a parameter-sensitivity pilot. It may guide the next run but may not promote any scalar, method family, or ranking into the formal D4RL benchmark without a separate freeze update and terminal audit.
