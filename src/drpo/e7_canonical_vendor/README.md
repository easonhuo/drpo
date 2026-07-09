# E7 canonical D4RL vendor snapshot

This directory contains the minimal source snapshot needed to run the user's
historical `train_sna2c_variant.py` / `SNA2C_IQLV_ExpRankAgent` backbone for the
EXT-H/E7 two-dataset canonical-agent pilot.

The wrapper fingerprints this source tree before every run.  Do not edit these
files to change scientific behavior; add a new fingerprinted source snapshot or
explicit adapter if the old code must be revised.

Included files:

- `d4rl/agents.py`
- `d4rl/train_sna2c_variant.py`
- `d4rl/d4rl_common/*`
- `d4rl/refs/d4rl_infos.py`

This vendor snapshot is source code only.  It does not include D4RL HDF5 data,
checkpoints, model weights, logs, or generated experiment results.
