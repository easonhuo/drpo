# C-U1 E3 unified-Adam causal intervention result

## Identity

- Experiment ID: `C-U1-E3-ADAM-RERUN`
- Scientific status: `long_run_validated`
- Run commit: `ac286a46b8ffad898dfad0e7e9188b1d2e81052a`
- Runner SHA-256: `502c345289d2b5b7c34832246478b64c33a1789e80ddcab7f6194cb09b0eac6f`
- Result artifact: `DRPO_CU1_E3_ADAM_AC286A4_FINAL.zip`
- Result artifact SHA-256: `2b8bfdbe6f33ed1db9dc1e59f6e9fbdb6c224c7b31b1326a7f2fbaeeaaaf522b`
- Optimizer: Adam, `betas=(0.9,0.999)`, `eps=1e-8`
- Data: 4096 train states / 4096 test states, 4 positive and 8 equal-advantage negative actions per state
- Held-out seeds: 30--49
- Terminology: same-distribution held-out-context generalization; not OOD

## Fixed-variance causal branch

| Method | Final reward mean [95% CI] | Task collapse | Support event | NaN/Inf |
|---|---:|---:|---:|---:|
| Baseline | 0.000002 [0.000002, 0.000002] | 20/20 | 0/20 | 0/20 |
| Near-zero | 0.000002 [0.000002, 0.000002] | 20/20 | 0/20 | 0/20 |
| Far-zero | 0.739362 [0.738902, 0.739837] | 0/20 | 0/20 | 0/20 |
| Far-cap | 0.733072 [0.732594, 0.733563] | 0/20 | 0/20 | 0/20 |
| Global-scale | 0.599057 [0.598634, 0.599475] | 0/20 | 0/20 | 0/20 |
| Far-to-near | 0.875323 [0.874994, 0.875616] | 0/20 | 0/20 | 0/20 |

Baseline and Near-zero collapse in every seed, while Far-zero and Far-cap prevent task collapse in every paired seed. This is the shortest paper-facing causal chain: retaining the far-field path is sufficient for failure in this controlled environment; deleting only near-field negative gradients is not a rescue; deleting or capping far-field influence is a rescue.

## Learnable-variance branch

| Method | Recorded reward at first event or terminal [95% CI] | Support contraction | Median onset | Unexpected expansion | NaN/Inf |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.603254 [0.593897, 0.612491] | 20/20 | 73 | 0/20 | 0/20 |
| Near-zero | 0.601992 [0.592785, 0.611119] | 20/20 | 73 | 0/20 | 0/20 |
| Far-zero | 0.652887 [0.651945, 0.653841] | 0/20 | — | 0/20 | 0/20 |
| Far-cap | 0.652625 [0.651661, 0.653619] | 0/20 | — | 0/20 | 0/20 |
| Global-scale | 0.642867 [0.641859, 0.643927] | 0/20 | — | 0/20 | 0/20 |

The first registered event for Baseline and Near-zero is support/variance contraction, not task collapse and not variance expansion. Adam removes the earlier plain-SGD one-step positive overshoot artifact without removing the contraction mechanism. Far-zero, Far-cap, and Global-scale avoid the support boundary in all seeds.

## Paper-use decision

This result is suitable for the controlled C-U1 causal experiment in the paper.

- Main text: use the fixed-variance Baseline / Near-zero / Far-zero / Far-cap comparison.
- Complementary panel or appendix: use the learnable-variance support-contraction result.
- Appendix controls: Global-scale and Far-to-near.
- Required reporting separation: task-performance collapse, support/variance contraction, and NaN/Inf numerical collapse.
- Prohibited inference: do not call the held-out test states OOD and do not infer a universal method ranking from this environment.

## Provenance and closure boundary

The scientific trajectories and terminal audit are bound to the exact committed runner at `ac286a46...`. The launch environment did not contain the Git commit object locally because shell DNS could not reach GitHub; source identity was instead recorded using the exact committed runner blob, SHA-256, and committed handoff/registry snapshots. Final aggregation used a JSON tuple/list normalization workaround after the raw result directory was restored; it changed no seed, configuration, optimizer, gradient, threshold, trajectory, or result value.

The full raw trajectories and checkpoints remain in the externally delivered result artifact. This repository directory contains only compact summaries and the artifact index.
