# E7 canonical two-dataset 1M shortlist pilot

## Scope

This protocol is a fixed two-dataset pilot under `EXT-H-E7-BENCH-01`.
It uses the recovered canonical actor-critic backbone to compare a small,
predeclared negative-control shortlist. It is external-validity evidence only:
it does not replace C-U1/D-U1 controlled mechanism identification and it is not
the formal nine-task D4RL benchmark.

The historical 300k taper pilot remains preserved. This protocol does not
reinterpret that run as a 1M result and does not delete or overwrite it.

## Fixed execution matrix

- Datasets:
  - `hopper-medium-replay-v2`
  - `hopper-medium-expert-v2`
- Dataset SHA-256:
  - replay: `e121c5f7c9857a307baa9edc6a2c3b48e85fedb9ac316ecddd0f48ca7ef4e39b`
  - expert: `9d51ad87f8c905be3880d84c6140bcdb7fbf39a19e046a237f238ba34fec9e26`
- Paired seeds: `200, 201, 202, 203`
- Training updates per branch: `1,000,000`
- Evaluation interval: `50,000` updates
- Evaluation episodes: `10`
- Evaluation points per branch: `20`
- Default parallel branch workers: `40`
- OpenMP threads per worker: `2`
- Expected methods per dataset-seed cell: `7`
- Expected total branches: `2 x 4 x 7 = 56`

The launcher uses the existing resumable branch subprocess executor. The
official entry point validates the complete scientific matrix before launch and
rejects any attempt to override seeds, steps, alpha, tau, temperature, learning
rate, batch size, grid, canonical source, or target class. Only dataset paths,
the exact registered dataset checksums, the output directory, `--resume`, and
the resource-only `E7_MAX_WORKERS` setting are accepted.

Changing the worker count must not change the scientific matrix. A same-named
HDF5 file with a different SHA-256 is rejected before branch launch.

## Fixed shortlist

| Reporting ID | Internal implementation | Effective negative coefficient | Role |
|---|---|---:|---|
| `original_exp_rank_mr` | unchanged passthrough `SNA2C_IQLV_ExpRankAgent` | rank-dependent, maximum 0.11 | recovered strong baseline |
| `positive_only` | `positive_only`, scale 0 | 0 | imitation-ceiling control |
| `global_neg_0p11` | `canonical_signed`, scale 1 | 0.11 | constant canonical-alpha anchor |
| `global_neg_0p011` | `global`, scale 0.1 | 0.011 | maximum-coefficient-matched global anchor |
| `reciprocal_linear_max0p011` | reciprocal-linear, scale 0.1 | at most 0.011 | distance-selective taper |
| `reciprocal_quadratic_max0p011` | reciprocal-quadratic, scale 0.1 | at most 0.011 | distance-selective taper |
| `exponential_max0p011` | exponential, scale 0.1 | at most 0.011 | distance-selective taper |

The two global anchors are intentionally different by a factor of ten. The
`0.011` global branch matches the **maximum coefficient** of the three distance
tapers. It is not a batch-gradient-norm, total-negative-weight, or optimizer-
update budget match. Therefore it supports a maximum-coefficient comparison but
must not be described as a strict gradient-budget-matched causal control.

This update reuses the coefficient values and standardized-distance reference
already present in the canonical two-dataset adapter configuration. It does not
introduce new D4RL-specific retuning:

- reference distance: `2.0`
- reciprocal-linear coefficient: `0.4362580032734791`
- reciprocal-quadratic coefficient: `0.5520268617673281`
- exponential coefficient: `0.374162511054291`

## Provenance gate

The full 56-branch pilot must start from a clean repository commit equal to the
authoritatively resolved `origin/main`. The canonical source root, module name,
agent and trainer relative paths, target class, return contract, Python tree
fingerprint, individual source fingerprints, and both dataset SHA-256 values are
validated before launch. The launcher records start and end repository
provenance and fails if the worktree is dirty, `origin/main` cannot be resolved,
HEAD differs from `origin/main`, or HEAD changes during execution.

Development-branch smoke or liveness checks are separate non-result gates and
must not be represented as the full pilot.

## Evaluation and terminal audit

The primary late window is fixed before execution to:

`750k, 800k, 850k, 900k, 950k, 1000k`.

The dedicated runner automatically writes `TERMINAL_AUDIT.json` after all 56
branches complete. It verifies the zero-exit branch manifest, exact trainer
metadata, and all 20 registered evaluation points, then reports at minimum:

- branch-level late-window mean, population standard deviation, minimum, and maximum;
- final score;
- best score and best step;
- best-to-final and best-to-late-mean drops;
- terminal slope per 100k updates;
- dataset-method aggregates across the four paired seeds;
- paired late-window differences versus Positive-only;
- task-performance collapse, support/variance-boundary events, and NaN/Inf numerical failure as separate fields.

No task-score threshold or stationarity tolerance is currently registered.
Consequently, the audit records threshold-dependent task collapse as
`not_classified`, records the unavailable support/variance boundary metric as
`not_available`, and assigns `fixed_horizon_inconclusive` rather than inventing
a convergence rule. Terminal slope is diagnostic only. A fixed 1M horizon is
not automatically a convergence claim, and steady-state ranking remains
prohibited without a separately registered terminal rule.

A zero process exit plus finite evaluation history is reported only as
`nan_inf_numerical_failure: not_observed`; it is not upgraded to a claim that a
separate internal NaN/Inf counter was measured when the unchanged canonical
trainer does not expose one.

## Launch

```bash
bash scripts/run_e7_canonical_two_dataset_shortlist_1m.sh \
  --data-dir /absolute/path/to/d4rl_hdf5 \
  --work-dir /absolute/path/to/e7_shortlist_1m
```

Optional resource override:

```bash
E7_MAX_WORKERS=24 bash scripts/run_e7_canonical_two_dataset_shortlist_1m.sh \
  --data-dir /absolute/path/to/d4rl_hdf5 \
  --work-dir /absolute/path/to/e7_shortlist_1m
```

The output directory must be absolute and new, or belong to the exact same run
identity when `--resume` is used. Do not point this launcher at the historical
300k pilot output directory.
