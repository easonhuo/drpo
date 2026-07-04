# E7 recovered DRRL/D4RL agent reference

This directory preserves a user-recovered historical DRRL/D4RL `agents.py`
reference for E7-taper implementation alignment review.

## Scope

The file is a provenance/reference artifact only. It is intentionally saved as
`.py.txt` so it is not imported as active experiment code.

The recovered source includes many historical algorithms and network presets.
For the current E7-taper framework, the immediate use is to align the recovered
network structure and relevant hyperparameters, especially the SNA2C/IQLV family,
without automatically switching the current runner to Q-based IQL or changing the
scientific role of E7.

## Important boundary

Some recovered agents use Q networks or full IQL-style Q/V updates. Current
E7-taper work must not blindly copy those implementations unless a separate
experiment registration explicitly changes the runner responsibility. In the
current interpretation, the reference supports canonical-agent-aligned network
and hyperparameter review, not full legacy trainer-lineage recovery.

## Preserved file

- `recovered_drrl_agents_20260704.py.txt`
- `recovered_drrl_agents_20260704.sha256`

The SHA-256 file records the exact preserved text artifact.
