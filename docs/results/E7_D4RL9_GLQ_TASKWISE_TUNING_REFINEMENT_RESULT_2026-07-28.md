# E7 D4RL-9 GLQ coarse tuning and refinement closure

The coarse and refinement development pilots each completed `540/540`
branches with zero failures. The combined audit freezes 27
task/controller selections from ten candidates per cell using the
registered 750k--1M late-window mean.

This evidence remains **pilot**. Development seeds are `200--203`;
held-out seeds `204--207` are untouched. Every selected cell is
`fixed_horizon_inconclusive`. No convergence, steady-state,
held-out-confirmation, cross-method-ranking, or formal D4RL-9-table
claim is allowed. Positive-only and Exponential were not rerun.

Provenance is bound to coarse code/result commits
`a0e4be818cbd780ac6ac36e0a56fa44de89493bf` /
`088b703c6df98e2fa5807d471260d3c7241c7614` and refinement
code/result commits `23cc332ce7e0a1156d752a3354782142ef57d470` /
`a9f40ce65a45456a3be57cf0205f8ed37e320180`. Draft PRs `#288`
and `#293` remain unmerged.

The exact 27 selected cells are stored in
`experiments/results/e7_d4rl9_glq_tuning_refinement_20260728/TASKWISE_SELECTION.csv`.
