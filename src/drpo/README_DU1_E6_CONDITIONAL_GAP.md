# D-U1 E6 Structured Conditional-Gap Protocol

This protocol extends, but does not overwrite,
`D-U1-E6-SEMANTIC-LONGRUN-01`. The completed dense semantic E6 remains a
benign-coverage control. The new protocol asks whether a large structured
conditional support gap permits wrong state-action generalization and
 task-performance collapse.

## What is held out

Train and test contexts are independent draws from the same marginal state
distribution. This is **not** state-distribution OOD.

The 256 actions are partitioned into eight semantic groups of 32 actions. For
exactly half of contexts, the entire optimal action group is absent from every
logged role. Each context logs only three of eight groups, so 62.5% of
conditional action-group blocks are absent. Every action group still appears in
other contexts, which isolates unseen state-action composition rather than a new
action-ID problem.

A paired control uses the same context coordinates and exposes the optimal group.
Random action-ID permutation prevents numeric action IDs from revealing group
structure.

## Reward and failure reporting

- correct group: reward scale 1.0;
- proxy positive group: reward scale 0.65;
- all other groups, including the trap group: reward 0.0;
- fixed concentration 8.0 isolates task overgeneralization from learnable-support dynamics.

The task-collapse rule is normalized between paired positive-only performance
and the random-policy reference. Task-performance collapse,
support/concentration boundary events, and NaN/Inf numerical failure are always
reported separately.

## Development checks

```bash
PYTHONPATH=src python src/drpo/du1_e6_conditional_gap.py \
  --config configs/du1_e6_conditional_gap_dev.yaml \
  --stage invariants --output-root /tmp/du1_e6_gap_invariants --device cpu

PYTHONPATH=src python src/drpo/du1_e6_conditional_gap.py \
  --config configs/du1_e6_conditional_gap_dev.yaml \
  --stage smoke --output-root /tmp/du1_e6_gap_smoke --device cpu
```

The development pilot uses only seeds 0--1 and cannot produce a formal result.

## Formal launch

After applying and committing this update on `main`, run from a clean checkout:

```bash
python scripts/run_du1_e6_conditional_gap_longrun.py \
  --work-dir /ABSOLUTE/PERSISTENT/PATH/D-U1-E6-CONDITIONAL-GAP-01/run_001 \
  --device cpu
```

The wrapper resolves the current commit, requires a clean worktree and
`origin/main` match, and launches the scientific runner through the canonical
hardened guard. Formal seeds are 130--149; development seeds are rejected from
formal aggregation.
