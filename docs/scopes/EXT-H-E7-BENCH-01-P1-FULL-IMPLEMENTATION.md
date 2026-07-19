# EXT-H-E7-BENCH-01 P1 full-run implementation

This stacked development branch starts from the reviewed P1 matrix implementation at `0d47ab23492138b3b4469567b2049d7582157e7c`. It implements the full 198-branch development run, task-balanced aggregation, and terminal audit after the real three-arm Hopper liveness passed. It remains a development hyperparameter screen, not formal scientific evidence or a method-ranking result.

The branch must not change the frozen nine-task matrix, seeds `200,201`, one-million-step horizon, GAE lambda `0.95`, taper lambda `1`, `tau=0`, or common `c` grid. Held-out seeds `204--207` remain untouched. Task-performance collapse, support/variance boundary, rollout failure, and NaN/Inf numerical failure remain separate.

The implementation reuses existing Python files only; temporary export and patch-transport scaffolding is excluded from the final stacked diff and from every scientific provenance identity. The full run remains explicitly gated and is not launched by this branch.
