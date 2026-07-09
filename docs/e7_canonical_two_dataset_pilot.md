# EXT-H/E7 canonical-agent two-dataset pilot

## Status and scope

This is a **pilot-only** external-validity workflow for `EXT-H-E7-BENCH-01`.
It does not replace the controlled C-U1/D-U1 mechanism experiments, and it does
not populate a formal D4RL-9 ranking table.

The workflow validates two Hopper cells first:

- `hopper-medium-replay-v2`
- `hopper-medium-expert-v2`

The goal is to separate two questions that were conflated in the frozen-critic
E7 scaffold:

1. Can the historical `agent.py` / `train_sna2c_variant.py` backbone recover the
   old ExpRank_MR performance scale on the current server and D4RL data?
2. Conditional on (1), does changing only the negative-advantage multiplier add
   stability or performance on top of the same strong backbone?

## Why the old backbone is vendored

The runner now uses a self-contained source snapshot under:

```text
src/drpo/e7_canonical_vendor/d4rl/
```

It includes only source files needed for the historical trainer:

- `agents.py`
- `train_sna2c_variant.py`
- `d4rl_common/*`
- `refs/d4rl_infos.py`

The wrapper fingerprints this vendored source tree before every run.  GLM does
not need to pass `--canonical-root /path/to/d4rl` for normal execution.  An
external `--canonical-root` remains available only for lineage audits.

## Execution profiles

The first question does **not** require running the entire taper grid.  It does
require the old training horizon if the result is to be compared to the old
ExpRank_MR scale.

| Profile | Default steps | Branches with default 2 datasets × 4 seeds | Purpose |
|---|---:|---:|---|
| `smoke` | 20k | 8 | Liveness only; not a performance result. |
| `reproduce` | 1M | 8 | Original ExpRank_MR passthrough reproduction. |
| `taper-pilot` | 300k | 56 | Small control/taper pilot after reproduction is sane. |
| `full-grid` | 1M | 120 | Broad exploratory grid; launch only after review. |

Therefore: **the original ExpRank_MR reproduction should keep the historical 1M
step budget**, but taper exploration should not start with a 120-branch 1M sweep.
Use the 300k `taper-pilot` profile first. The default pilot seeds are now `[200, 201, 202, 203]`; override `--seeds` only for liveness/debugging, not for interpreting method trends.

## One-click commands

Assuming the two HDF5 files are under `/data/d4rl_hdf5`:

```bash
scripts/run_e7_canonical_two_dataset.sh smoke \
  --data-dir /data/d4rl_hdf5 \
  --work-dir outputs/e7_canonical_two_dataset_smoke

scripts/run_e7_canonical_two_dataset.sh reproduce \
  --data-dir /data/d4rl_hdf5 \
  --work-dir outputs/e7_canonical_two_dataset_reproduce

scripts/run_e7_canonical_two_dataset.sh taper-pilot \
  --data-dir /data/d4rl_hdf5 \
  --work-dir outputs/e7_canonical_two_dataset_taper_pilot
```

The expected HDF5 filenames are:

```text
hopper-medium-replay-v2.hdf5
hopper-medium-expert-v2.hdf5
```

Explicit paths may be supplied instead:

```bash
python scripts/run_e7_canonical_two_dataset.py run \
  --profile reproduce \
  --hopper-medium-replay-hdf5 /abs/path/hopper-medium-replay-v2.hdf5 \
  --hopper-medium-expert-hdf5 /abs/path/hopper-medium-expert-v2.hdf5 \
  --work-dir outputs/e7_canonical_two_dataset_reproduce
```

## Interpretation gates

1. `smoke` only checks imports, dataset access, runner wiring, checkpoint/log
   creation, and early numerical liveness.
2. `reproduce` checks whether the old ExpRank_MR backbone can recover the old
   score scale on the two selected Hopper cells.  If this fails, investigate
   data, dependencies, normalization, env versions, and source lineage before
   evaluating taper.
3. `taper-pilot` is only meaningful after `reproduce` is sane.  It compares the
   original passthrough branch to a small set of injected controls on the same
   backbone.
4. No profile here is a formal D4RL-9 result.  Formal method ranking requires a
   registered expanded protocol, terminal-state audit, complete logs, raw curves,
   checkpoint evaluation policy, and provenance bound to the launch commit.

## Branch definitions

`reproduce` uses no injected branches.  Each dataset/seed has only:

- `original_exp_rank_mr`: unchanged `SNA2C_IQLV_ExpRankAgent` with
  `--variant iqlv_exp_rank --alpha 0.11 --tau 0.5 --temp 5.0`.

`taper-pilot` adds six injected branches per dataset/seed:

- `positive_only`
- `canonical_signed`
- `global` at scale `0.01`
- `reciprocal_linear` at scale `0.1`
- `reciprocal_quadratic` at scale `0.1`
- `exponential` at scale `0.1`

The injection preserves the old trainer/network/critic/update loop and replaces
only the negative-advantage multiplier in the configured agent class.
