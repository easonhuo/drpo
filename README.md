# DRPO

Research codebase for DRPO experiments on offline reinforcement learning datasets.

## Status

This repository is newly scaffolded. The first target is to support lightweight D4RL / Minari Hopper experiments, including:

- dataset loading from Minari folders and legacy D4RL HDF5 files
- Gaussian MLP policy training
- near/far advantage matching
- per-sample gradient diagnostics
- result aggregation and plotting

## Project Layout

```text
DRPO/
  configs/          Experiment configs
  data/             Local datasets or symlinks, ignored by git
  outputs/          Experiment outputs, ignored by git
  scripts/          Command-line entry points
  src/drpo/         Python package
  tests/            Unit tests
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Next Steps

1. Define the exact DRPO objective and baselines.
2. Add D4RL / Minari dataset loaders.
3. Implement the policy model and training loop.
4. Add gradient diagnostic scripts for single-seed Hopper validation.
5. Add reproducible configs for multi-task and multi-seed runs.
