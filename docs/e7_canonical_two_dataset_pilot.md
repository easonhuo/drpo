# EXT-H-E7-BENCH-01 canonical-agent two-dataset pilot

## Purpose

This is a **pilot-only** development path for the D4RL external-validity layer.
It is not a replacement for C-U1/D-U1 controlled mechanism evidence and it does
not populate the formal D4RL-9 table.

The goal is to recover the user's older D4RL `agent.py` / `train_sna2c_variant.py`
backbone before doing any new method comparison.  The current frozen-critic
`e7_bench.py` scaffold remains useful for mechanism diagnostics, but it should
not be treated as the canonical D4RL performance backbone.

## Branch and scope

Recommended dev branch:

```bash
git checkout -b dev/e7-canonical-agent-taper origin/main
```

Allowed scope for this branch:

- fingerprint the old D4RL source tree;
- validate the old `SNA2C_IQLV_ExpRankAgent` trainer on two Hopper cells;
- run the unchanged ExpRank_MR passthrough branch;
- run injected negative-control branches that preserve the old trainer/network
  and replace only the negative-advantage multiplier;
- save per-branch command, source fingerprints, logs, trainer outputs, and
  completion/failure markers.

Forbidden scope:

- changing the old canonical source after fingerprinting;
- changing actor/critic network sizes, optimizer, learning rate, TD target,
  value expectile, batch size, or D4RL normalization outside the wrapper args;
- editing `src/drpo/e7_bench.py` for this canonical-agent pilot;
- interpreting pilot output as a formal method ranking.

## First validation cells

The first pilot is intentionally limited to two cells:

- `hopper-medium-replay-v2`;
- `hopper-medium-expert-v2`.

Default seeds are `200, 201`.  Default trainer horizon is `1_000_000` updates,
with 50k evaluation/checkpoint intervals.  These defaults can be overridden for
smoke or liveness gates, but such runs remain smoke/pilot evidence only.

## Commands

Prepare source fingerprints and run spec without training:

```bash
python scripts/run_e7_canonical_two_dataset.py prepare \
  --canonical-root /ABS/PATH/TO/OLD_D4RL_SOURCE/d4rl \
  --hopper-medium-replay-hdf5 /ABS/PATH/hopper-medium-replay-v2.hdf5 \
  --hopper-medium-expert-hdf5 /ABS/PATH/hopper-medium-expert-v2.hdf5 \
  --work-dir /ABS/PATH/outputs/e7_canonical_two_dataset/run_001
```

Plan all branches:

```bash
python scripts/run_e7_canonical_two_dataset.py plan \
  --canonical-root /ABS/PATH/TO/OLD_D4RL_SOURCE/d4rl \
  --hopper-medium-replay-hdf5 /ABS/PATH/hopper-medium-replay-v2.hdf5 \
  --hopper-medium-expert-hdf5 /ABS/PATH/hopper-medium-expert-v2.hdf5 \
  --work-dir /ABS/PATH/outputs/e7_canonical_two_dataset/run_001 \
  --max-workers 4
```

Run or resume:

```bash
python scripts/run_e7_canonical_two_dataset.py run \
  --canonical-root /ABS/PATH/TO/OLD_D4RL_SOURCE/d4rl \
  --hopper-medium-replay-hdf5 /ABS/PATH/hopper-medium-replay-v2.hdf5 \
  --hopper-medium-expert-hdf5 /ABS/PATH/hopper-medium-expert-v2.hdf5 \
  --work-dir /ABS/PATH/outputs/e7_canonical_two_dataset/run_001 \
  --max-workers 4

python scripts/run_e7_canonical_two_dataset.py run \
  --canonical-root /ABS/PATH/TO/OLD_D4RL_SOURCE/d4rl \
  --hopper-medium-replay-hdf5 /ABS/PATH/hopper-medium-replay-v2.hdf5 \
  --hopper-medium-expert-hdf5 /ABS/PATH/hopper-medium-expert-v2.hdf5 \
  --work-dir /ABS/PATH/outputs/e7_canonical_two_dataset/run_001 \
  --max-workers 4 --resume
```

If dataset SHA-256 values are already known, pass them explicitly with
`--hopper-medium-replay-sha256` and `--hopper-medium-expert-sha256`; otherwise the
wrapper computes and stores them in the concrete run spec.

## Methods in the default grid

For each `(dataset, seed)`, the wrapper produces:

- one unchanged passthrough branch: `original_exp_rank_mr`;
- injected anchors: `positive_only`, `canonical_signed`;
- global scales: `0.005`, `0.01`;
- reciprocal-linear scales: `0.03`, `0.1`, `0.3`;
- reciprocal-quadratic scales: `0.03`, `0.1`, `0.3`;
- exponential scales: `0.03`, `0.1`, `0.3`, `1.0`.

The injected branches preserve the old trainer and replace only the negative
advantage multiplier in the contracted `SNA2C_IQLV_ExpRankAgent` class.  Positive
advantages and the full-batch loss denominator are unchanged.

## Acceptance for the first GLM run

Before scaling beyond two datasets, GLM must deliver:

- `canonical_contract.json`;
- `run_spec.json`;
- `EXECUTION_PLAN.json`;
- per-branch `BRANCH_IDENTITY.json`, `LAUNCH.json`, `branch_manifest.json`,
  `stdout_stderr.log`, and trainer outputs;
- `RUN_SUMMARY.json`;
- a concise report comparing unchanged `original_exp_rank_mr` against the
  injected branches on both cells.

The report must separate task-performance collapse, support/boundary events, and
NaN/Inf numerical failures if those diagnostics are available.  A fixed 1M
horizon is not convergence; method ranking still requires terminal audit and a
separate formal protocol.
