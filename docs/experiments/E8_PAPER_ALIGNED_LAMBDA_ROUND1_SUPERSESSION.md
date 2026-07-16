# Supersession of the 18-Cell Calibrated-Lambda Round 1 Plan

The previously registered experiment
`EXT-C-E8-ORACLE-OFFLINE-V2-PAPER-ALIGNED-LAMBDA-ROUND1-0.5B-01`
was not run. Its duplicate execution stack was removed from `main` before any
CUDA liveness or scientific sweep.

For new execution, it is superseded by:

`EXT-C-E8-ORACLE-OFFLINE-V2-PAPER-ALIGNED-LINEAR-SCAN-0.5B-01`

The successor isolates the correction `u^2 -> u` in the existing alpha1-c
lineage, uses 16 parameter points on two paired development seeds, and preserves
the canonical GPU 0-7 x two-slots-per-GPU runtime.

The old protocol, registry entry, handoff delta and Git history are preserved as
historical planning records. They must not be used as launch instructions, and
they must not be interpreted as completed evidence. The current protocol and
operator instructions are:

- `docs/experiments/E8_PAPER_ALIGNED_LINEAR_SCAN_PROTOCOL.md`;
- `docs/experiments/E8_PAPER_ALIGNED_LINEAR_SCAN_OPERATOR.md`;
- `runspecs/ready/E8_PAPER_ALIGNED_LINEAR_SCAN_20260716_01.yaml`.

A later calibrated `tau/scale_c` experiment requires a new explicit protocol and
cannot be inferred from this supersession.
