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
- Paired seeds: `200, 201, 202, 203`
- Training updates per branch: `1,000,000`
- Evaluation interval: `50,000` updates
- Evaluation episodes: `10`
- Evaluation points per branch: `20`
- Default parallel branch workers: `40`
- OpenMP threads per worker: `2`
- Expected methods per dataset-seed cell: `7`
- Expected total branches: `2 x 4 x 7 = 56`

The launcher uses the existing resumable `ThreadPoolExecutor` branch scheduler.
The default runs up to 40 independent subprocess branches concurrently. Set
`E7_MAX_WORKERS` to another integer greater than or equal to 2 only for resource
capacity reasons; changing worker count must not change the scientific matrix.

## Fixed shortlist

| Reporting ID | Internal implementation | Effective negative coefficient | Role |
|---|---|---:|---|
| `original_exp_rank_mr` | unchanged passthrough `SNA2C_IQLV_ExpRankAgent` | rank-dependent, maximum 0.11 | recovered strong baseline |
| `positive_only` | `positive_only`, scale 0 | 0 | imitation-ceiling control |
| `global_neg_0p11` | `canonical_signed`, scale 1 | 0.11 | constant canonical-alpha anchor |
| `global_neg_0p011` | `global`, scale 0.1 | 0.011 | magnitude-matched global control |
| `reciprocal_linear_max0p011` | reciprocal-linear, scale 0.1 | at most 0.011 | distance-selective taper |
| `reciprocal_quadratic_max0p011` | reciprocal-quadratic, scale 0.1 | at most 0.011 | distance-selective taper |
| `exponential_max0p011` | exponential, scale 0.1 | at most 0.011 | distance-selective taper |

The two global controls are intentionally different by a factor of ten. The
`0.011` global branch matches the maximum coefficient of the three distance
tapers and distinguishes selective distance decay from simply shrinking all
negative gradients.

This update reuses the coefficient values and standardized-distance reference
already present in the canonical two-dataset adapter configuration. It does not
introduce new D4RL-specific retuning:

- reference distance: `2.0`
- reciprocal-linear coefficient: `0.4362580032734791`
- reciprocal-quadratic coefficient: `0.5520268617673281`
- exponential coefficient: `0.374162511054291`

## Evaluation and terminal audit

The primary late window is fixed before execution to:

`750k, 800k, 850k, 900k, 950k, 1000k`.

Report at minimum:

- late-window mean, standard deviation, minimum, and maximum;
- final score;
- best score and best step;
- best-to-final and best-to-late-mean drops;
- fraction of late-window evaluations above the registered threshold;
- terminal slope or explicit persistent-drift/inconclusive label.

Best checkpoint score is diagnostic and must not be the sole ranking metric.
A fixed 1M horizon is not automatically a convergence claim. Method ranking or
steady-state language requires the registered terminal audit.

Task-performance collapse, support/variance-boundary events, and NaN/Inf
numerical failure must be reported separately.

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

The output directory must be new or belong to the exact same run identity when
`--resume` is used. Do not point this launcher at the historical 300k pilot
output directory.
