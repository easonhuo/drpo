# EXT-C-E8-ORACLE-OFFLINE-V2-PAPER-ALIGNED-LAMBDA-MINIMAL-0.5B-01

## Status

Code-first development pilot. Not run. Not formal evidence.

## Canonical lineage

- predecessor branch: `dev/e8-alpha1-highc-scan-pilot`
- predecessor commit: `929142930a3e2efaa7cafc8e4afe3866600027a5`
- inherited unchanged: trainer, optimizer, evaluator, checkpointing, resume,
  aggregation, terminal audit, GPU selector, and 8-GPU x 2-slot scheduler
- runtime capacity: 16 concurrent cells; 32 cells complete in two scheduling waves

The predecessor files are protected by exact Git blob hashes in config and tests.
A changed predecessor blob fails the focused test. This successor is forbidden
from implementing another trainer, scheduler, or evaluator.

## Only authorized scientific delta

```text
D = negative mean completion-token log probability
z = relu((D - tau) / scale_c)
w = alpha * exp(-lambda * z)
```

`D`, `z`, and `w` are detached. EOS is included in the completion mean. No extra
surprisal square is permitted. `tau` and `scale_c` are frozen once from the
shared fresh-LoRA initialization without task, validation, or test metrics.

The predecessor `Cell.c` argument is retained only as a compatibility carrier
for paper `lambda`; outputs add an explicit `lambda` field.

## Round-1 matrix

- Positive-only: one point;
- paper lambda: 15 points including exact `lambda=0` uncontrolled endpoint;
- paired development seed offsets: `4000,5000`;
- total: `16 points x 2 seeds = 32 cells`;
- runtime: GPU `0-7`, two cells per GPU, 16-way concurrency.

`alpha=1`, `tau`, and `scale_c` remain frozen. Global is reused as historical
context and is not rerun. Test access is forbidden. This round localizes a
region; it does not establish method ranking, convergence, or steady state.

## Launch

```bash
bash scripts/run_countdown_e8_paper_aligned_lambda_minimal_one_click.sh
```

The launcher plans the existing two-slot runtime, calibrates on the first
selected GPU, exports the frozen calibration identity, runs the inherited smoke
gate, and then invokes the inherited 16-slot runtime. Identity-checked resume is
inherited.
