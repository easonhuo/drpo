# EXT-H-E7-BENCH-01 Long-Budget Parallel Pilot Protocol

## Scope

This is a `pilot` substage under the existing `EXT-H-E7-BENCH-01` ID. It does not create a second top-level experiment. The pilot checks the benchmark implementation, runtime, artifact volume, direct-from-seed method comparison, rollout evaluation, terminal-state audit, and method-specific parameter sensitivity under a D4RL-scale training budget.

The formal benchmark remains the registered nine-cell D4RL MuJoCo suite. Pilot output is not formal nine-task evidence and may not be used to select a new method family, tune a task-specific coefficient, or populate a formal method ranking. A later freeze update is required before any selected scalar can be promoted to a formal benchmark setting.

## Correction of the superseded short-budget design

The first implementation registered `20k` critic steps, a `20k` Positive-only checkpoint, and `40k` non-positive branches. That design was suitable only as an engineering smoke test. It had two scientific defects:

1. every stage was one fifth of the already audited E7-Q2 long-run budget; and
2. Positive-only stopped after `20k`, while the other methods received another `40k`, so the baseline and controlled methods did not have equal actor horizons.

No scientific pilot result was produced under that design. This protocol was later corrected again after audit: the shared Positive-only warm-start changed the scientific question into a continuation-from-Positive-only protocol. The current pilot therefore uses direct method training from the same actor initialization stream for each `(dataset, seed, method)`:

- canonical critic: `100k` optimizer steps per dataset;
- shared Positive-only warm-start: `0` steps; no warm-start stage is scheduled;
- method training: `500k` steps for **every** method variant, including Positive-only;
- total actor horizon: `500k` steps for every compared method.

Only NaN/Inf numerical failure may stop a worker early. A fixed horizon is not treated as convergence.

## Pilot parameter-sweep matrix

- Two development seeds: `200, 201`. The seed count is intentionally reduced from four to two so the same server budget can cover more method-specific scalar settings.
- Six method families remain fixed: Positive-only, Signed, Global alpha, Reciprocal-Linear, Reciprocal-Quadratic, and Exponential.
- The follow-up pilot now expands these into 21 method variants: one Positive-only, one Signed, four Global-alpha scalars, four stronger Reciprocal-Linear coefficients, four stronger Reciprocal-Quadratic coefficients, and seven Exponential coefficients. The prior stronger-taper pilot showed Exponential around `c=8` had the clearest usable signal, while Reciprocal and Global-alpha also improved as their control strength increased; this matrix therefore keeps all non-positive families alive instead of prematurely reducing the paper comparison to Exponential-only.
- The old C-U1 near-retention scalars remain recorded in the config as provenance, but the active follow-up grid is a pilot sensitivity search centered on stronger taper settings; it does not authorize per-task D4RL retuning or formal coefficient promotion.
- Latest pilot interpretation is recorded only as tuning rationale: Exponential currently has the strongest stable signal, but Reciprocal and Global-alpha are still under-explored at stronger settings. No method ranking, formal D4RL table entry, or locked conclusion is promoted by this pilot update.
- Two uploaded legacy D4RL-v2 data cells are used in the current pilot:
  - `hopper-medium-replay-v2`: the exact uploaded D4RL-v2 medium-replay HDF5 cell.
  - `hopper-medium-expert-v2`: the exact uploaded D4RL-v2 medium-expert HDF5 cell.

The earlier Minari/Hopper-v5 medium file is no longer part of this E7-BENCH pilot config. That earlier cell remains historical plumbing evidence only and must not be relabeled as D4RL medium or medium-replay.

## Parallel execution

The coordinator uses two fail-fast subprocess stages:

1. Two dataset-level canonical critics run concurrently: `2 × 64 = 128` CPU threads.
2. Ninety-two `(dataset, seed, method_variant)` direct-from-seed actor trainings run concurrently: `92 × 4 = 368` CPU threads.

The actor stage includes Positive-only itself. Every method variant constructs a policy from the same dataset/seed initialization stream, creates its own optimizer, and receives the same 500k actor budget. This avoids the superseded protocol where all methods inherited a Positive-only warm-start before method-specific continuation.

The taper definitions are fixed and explicit. For standardized Gaussian distance
`u = d / 5`, Reciprocal-Linear uses `1 / (1 + c u)`, Reciprocal-Quadratic uses
`1 / (1 + c u^2)`, and Exponential uses `exp(-c u)`. “Quadratic” here refers to
the squared standardized distance, which is the Gaussian surprisal-order proxy;
it is not the quartic form that would arise from reciprocal-squared-surprisal.

The peak registered allocation is 368 threads, leaving 16 threads of headroom on a 384-core server. Seeds and method variants are never executed by a top-level serial loop. Every worker has an isolated output directory, and shared aggregate files are written only by the coordinator.

The formal nine-cell registration remains fail-closed. If promoted later, the relevant topology is `task_seed_method` with both serial seed and serial method loops forbidden, and methods must not silently inherit a Positive-only warm-start unless a separate warm-start-continuation protocol is explicitly registered.

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
of waiting for dozens of 500k-step workers to finish. Canonical critics must complete
their full frozen budget before actor training. Method workers may stop early only
for NaN/Inf; summaries record scheduled and actually executed actor steps separately,
so a numerical failure cannot be misreported as a completed 500k path.

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

The runtime environment must provide the E7-Q2 PyTorch/HDF5 stack plus `gymnasium` with MuJoCo support for `Hopper-v4`. The runner fails before the 100k critics when this rollout dependency is unavailable. The branch worker validates its method variant directly and does not require a `--warmstart-dir` argument.

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

## Direct-from-seed 500k correction

The previous warm-start continuation design is superseded for method comparison. It remains useful only as a separate diagnostic question: whether controlled negative updates can improve after a Positive-only policy has already been trained. The current pilot instead answers the cleaner comparison question: under the same critic, dataset cell, actor seed, network profile, optimizer, and 500k actor budget, how do the method variants behave when trained directly from the same actor initialization stream?

This update also replaces the historical Minari/Hopper-v5 plumbing cell with the uploaded D4RL-v2 `hopper-medium-replay-v2` file. The current pilot cells are therefore `hopper-medium-replay-v2` and `hopper-medium-expert-v2`. The run remains a pilot: two seeds are not enough for a formal method ranking, and fixed 500k training remains a horizon requiring terminal audit rather than a proof of convergence.

## Reusable canonical critic cache

The pilot supports a reusable canonical critic cache. This is an engineering feature for the same frozen-advantage protocol, not a new scientific result. It prevents repeated `critic_steps` spending when only actor seeds, method variants, branch horizon, rollout episodes, or sweep coefficients change.

The cache is optional but enabled by default in `configs/e7_bench_pilot.yaml` with root `~/.cache/drpo/e7_bench/critics`. It can be overridden at launch time via `--critic-cache-dir`. A cache entry is valid only when its identity matches all critic-defining inputs:

- experiment ID and runner/cache schema version;
- dataset ID, dataset SHA256, format, env ID, dataset family, score protocol, and D4RL normalization references;
- uploaded base E7-Q2 config SHA256;
- canonical critic seed;
- critic optimizer-step budget and eval interval;
- recovered 2×256 ReLU orthogonal network profile and log-std clamp.

The identity deliberately excludes actor seeds, method variants, the sweep config SHA, branch actor horizon, and rollout-evaluation episode counts. Therefore a cached critic remains reusable across direct-from-seed actor branches and future actor-only sensitivity sweeps, but a dataset, critic-seed, critic-budget, base-config, network, or runner/cache-schema change invalidates it.

The runner copies cached critic artifacts into the work directory rather than symlinking them. Cache hits write `CRITIC_CACHE_USED.json`; newly trained critics are copied back to the cache with `CRITIC_CACHE_STORED.json`. A missing, stale, or malformed cache entry fails closed instead of silently using an unchecked artifact.

This does not implement joint actor-critic training. It preserves the existing frozen-advantage comparison while making longer or repeated critic pretraining practical. Critic-seed robustness, if needed for paper-facing evidence, should be a separate top-method audit instead of coupling every actor seed to a new critic seed.
