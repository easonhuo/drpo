# E7 Frozen-Critic TD/GAE Diagnostic — 192-Branch Result Archive

**Archive date:** 2026-07-18  
**Repository documentation base:** `bb637503e1289f24f7a28e587f50665afb20e0de`  
**Evidence class:** superseded frozen-critic / precomputed-advantage development diagnostic  
**Current authoritative joint-critic experiment:** `EXT-H-E7-SQEXP-GAE-01` remains `not_run`  
**External result package SHA-256:** `01ac39461248391e668f9808428b3d66157ea9ea28ac5411d0b07b3a725c04d0`  
**Package run-identity SHA-256:** `ed24dd90262823422368e3d3fb991743c1c2e8c67d8da2188c118ac75c6b5719`

> This document preserves a completed result that was produced by the earlier
> frozen-critic implementation. It is not the result of the subsequently merged
> joint-critic implementation. The package contains no resolvable repository
> source commit, so it must not be promoted to formal evidence or used to change
> the current joint-critic experiment status.

## 1. Identity and matrix

The archived package reports `scientific_status=frozen_critic_trajectory_gae_development_pilot_only`. All 192 branch manifests set `critic_frozen=true` and consume a prepared advantage manifest. The matrix is:

- datasets: Hopper medium-expert, Walker2d medium, Walker2d medium-replay;
- development seeds: `200,201,202,203`; held-out seeds `204--207` are untouched;
- advantage estimators: one-step TD and GAE with `lambda=0.95`;
- actor updates: canonical A2C and PPO clip K=4;
- controls: Positive-only and squared EXP `c={64,128,256}`;
- horizon: 1,000,000 actor updates, evaluated every 50,000 updates;
- total: `3 × 4 × 2 × 2 × 4 = 192` branches.

Run completion and terminal audit:

| Item | Value |
|---|---:|
| Completed branches | 192/192 |
| Failed branches | 0 |
| Critic-audit failures | 0 |
| NaN/Inf numerical failures | 0 |
| Support/variance-boundary events | 0 |
| Held-out seeds touched | no |
| Formal evidence allowed | no |
| Fixed 1M treated as convergence | no |
| Universal GAE claim allowed | no |

Task-performance collapse was not adjudicated because the run did not register a task-collapse threshold. This is separate from the zero numerical-failure and zero support/variance-boundary counts.

## 2. Score definition

Every score below is the mean over seeds `200--203` and is read directly from the package aggregate:

- `500k`: normalized score at update 500,000;
- `BEST`: per-seed best normalized score over the 1M trajectory, then averaged;
- `FINAL`: normalized score at update 1,000,000;
- `LATE`: per-seed mean over the 800k--1M window, then averaged.

Positive-only exists separately for both estimators: TD Positive-only keeps only positive one-step-TD advantages, whereas GAE Positive-only keeps only positive GAE advantages.

## 3. Full TD/GAE score tables

### Hopper medium-expert

#### A2C

| Control | TD 500k | TD BEST | TD FINAL | TD LATE | GAE 500k | GAE BEST | GAE FINAL | GAE LATE | GAE−TD LATE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Positive-only | 24.30 | 84.84 | 62.02 | 42.30 | 64.38 | 93.00 | 79.67 | 72.97 | +30.67 |
| c=64 | 45.73 | 75.45 | 50.42 | 36.52 | 56.42 | 106.31 | 80.78 | 57.50 | +20.98 |
| c=128 | 53.34 | 78.14 | 60.41 | 55.19 | 61.73 | 102.14 | 83.76 | 67.63 | +12.44 |
| c=256 | 71.69 | 91.20 | 35.43 | 41.71 | 71.60 | 103.46 | 91.86 | 81.58 | +39.87 |

#### PPO clip K=4

| Control | TD 500k | TD BEST | TD FINAL | TD LATE | GAE 500k | GAE BEST | GAE FINAL | GAE LATE | GAE−TD LATE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Positive-only | 43.59 | 96.44 | 60.27 | 47.86 | 76.45 | 97.64 | 72.16 | 73.49 | +25.64 |
| c=64 | 53.10 | 84.30 | 30.79 | 39.52 | 50.12 | 95.02 | 86.79 | 76.69 | +37.18 |
| c=128 | 39.04 | 83.94 | 57.52 | 50.32 | 70.51 | 94.25 | 78.77 | 70.98 | +20.66 |
| c=256 | 70.45 | 87.35 | 47.52 | 49.70 | 76.97 | 95.83 | 82.68 | 80.59 | +30.89 |

### Walker2d medium

#### A2C

| Control | TD 500k | TD BEST | TD FINAL | TD LATE | GAE 500k | GAE BEST | GAE FINAL | GAE LATE | GAE−TD LATE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Positive-only | 70.06 | 82.13 | 69.42 | 65.14 | 81.83 | 82.93 | 75.00 | 74.26 | +9.11 |
| c=64 | 63.86 | 80.63 | 63.56 | 63.35 | 77.36 | 83.69 | 79.48 | 73.62 | +10.27 |
| c=128 | 67.73 | 81.91 | 66.58 | 69.78 | 79.56 | 83.65 | 78.01 | 74.00 | +4.23 |
| c=256 | 57.68 | 81.49 | 63.48 | 61.91 | 76.43 | 84.26 | 78.00 | 76.31 | +14.40 |

#### PPO clip K=4

| Control | TD 500k | TD BEST | TD FINAL | TD LATE | GAE 500k | GAE BEST | GAE FINAL | GAE LATE | GAE−TD LATE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Positive-only | 61.69 | 81.08 | 54.74 | 61.67 | 78.12 | 83.42 | 73.14 | 75.50 | +13.83 |
| c=64 | 64.14 | 80.47 | 61.51 | 66.78 | 75.67 | 82.82 | 72.13 | 72.05 | +5.27 |
| c=128 | 59.90 | 82.52 | 62.54 | 56.97 | 75.32 | 83.12 | 80.35 | 72.98 | +16.01 |
| c=256 | 56.07 | 80.23 | 57.06 | 60.09 | 80.60 | 83.25 | 77.21 | 76.51 | +16.41 |

### Walker2d medium-replay

#### A2C

| Control | TD 500k | TD BEST | TD FINAL | TD LATE | GAE 500k | GAE BEST | GAE FINAL | GAE LATE | GAE−TD LATE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Positive-only | 59.18 | 74.40 | 57.63 | 60.91 | 46.80 | 79.41 | 62.77 | 63.09 | +2.19 |
| c=64 | 57.63 | 75.97 | 61.14 | 57.77 | 59.91 | 76.81 | 60.34 | 60.32 | +2.55 |
| c=128 | 69.64 | 79.14 | 49.16 | 48.83 | 59.74 | 76.17 | 72.19 | 64.52 | +15.68 |
| c=256 | 56.71 | 77.21 | 55.20 | 55.97 | 53.91 | 81.48 | 60.74 | 63.05 | +7.08 |

#### PPO clip K=4

| Control | TD 500k | TD BEST | TD FINAL | TD LATE | GAE 500k | GAE BEST | GAE FINAL | GAE LATE | GAE−TD LATE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Positive-only | 66.23 | 74.05 | 52.28 | 59.58 | 61.88 | 79.26 | 66.39 | 63.68 | +4.11 |
| c=64 | 60.75 | 80.69 | 53.77 | 56.58 | 54.37 | 82.58 | 54.54 | 61.23 | +4.65 |
| c=128 | 64.09 | 78.24 | 55.83 | 60.33 | 67.17 | 76.57 | 58.52 | 60.62 | +0.29 |
| c=256 | 60.82 | 78.08 | 64.07 | 62.02 | 60.41 | 79.81 | 55.50 | 59.38 | -2.64 |

## 4. Positive-only isolation

This comparison removes every negative-advantage actor term and therefore isolates the effect of changing the retained positive labels from one-step TD to GAE.

| Dataset | Actor | TD Positive-only LATE | GAE Positive-only LATE | Difference |
|---|---|---:|---:|---:|
| Hopper medium-expert | A2C | 42.30 | 72.97 | +30.67 |
| Hopper medium-expert | PPO clip K=4 | 47.86 | 73.49 | +25.64 |
| Walker2d medium | A2C | 65.14 | 74.26 | +9.11 |
| Walker2d medium | PPO clip K=4 | 61.67 | 75.50 | +13.83 |
| Walker2d medium-replay | A2C | 60.91 | 63.09 | +2.19 |
| Walker2d medium-replay | PPO clip K=4 | 59.58 | 63.68 | +4.11 |

GAE Positive-only exceeds TD Positive-only in all six dataset–actor cells. This shows that the observed GAE signal is not created only by a particular negative-control coefficient.

## 5. Descriptive aggregate findings

- GAE has the higher 800k--1M mean in **23/24** dataset–actor–control groups.
- Across paired seeds, GAE has the higher late-window score in **82/96** cells.
- Pooled descriptive means are: GAE−TD `LATE=+14.24`, `FINAL=+17.02`, and `BEST=+5.71` normalized-score points.
- GAE reduces BEST-to-FINAL drop by an average of **11.31** points; the drop is smaller in **66/96** paired cells.
- The only grouped late-window exception is Walker2d medium-replay with PPO K=4 and `c=256`, where GAE−TD is `-2.64`.

Dataset-specific patterns:

- **Hopper medium-expert:** the strongest GAE signal. GAE `c=256` reaches LATE `81.58` under A2C and `80.59` under PPO K=4; GAE Positive-only is already `72.97/73.49`.
- **Walker2d medium:** GAE is consistently stronger than TD, but the best negative-control setting adds only about 1–2 points beyond GAE Positive-only.
- **Walker2d medium-replay:** the GAE signal is weaker and control-dependent. A2C is best at GAE `c=128` (`64.52` LATE), while PPO K=4 is best at GAE Positive-only (`63.68`).

At Walker2d medium-replay, pooled GAE−TD is negative at 500k but positive in the late window:

| Actor | Mean GAE−TD at 500k | Mean GAE−TD at 800k--1M |
|---|---:|---:|
| A2C | -5.70 | +6.88 |
| PPO clip K=4 | -2.01 | +1.60 |

Therefore a 500k-only decision would mischaracterize the later trajectory in this dataset.

## 6. Advantage diagnostics

| Dataset | TD std | GAE std | GAE/TD std | Pearson | Spearman | Sign-flip fraction |
|---|---:|---:|---:|---:|---:|---:|
| Hopper medium-expert | 1.082 | 3.833 | 3.54× | 0.347 | 0.288 | 39.1% |
| Walker2d medium-replay | 2.064 | 5.848 | 2.83× | 0.351 | 0.309 | 38.6% |
| Walker2d medium | 1.138 | 3.352 | 2.95× | 0.345 | 0.296 | 38.9% |

GAE is not merely a low-noise copy of one-step TD in this package: its standard deviation is roughly 2.8–3.5× larger, the TD/GAE correlation is modest, and about 39% of transitions change advantage sign. Thus GAE materially changes which samples are encouraged or suppressed and their gradient magnitude.

## 7. Interpretation boundary

The archived run supports a strong **candidate signal** that trajectory-aware advantages can improve finite-horizon external-task performance, especially on Hopper. It also suggests an interaction between GAE's heavier advantage tails and far-field negative-gradient control.

It does **not** establish:

- that GAE will retain the same gains when the critic evolves jointly;
- a universal best `c`; the best setting varies by dataset and actor;
- PPO superiority over A2C;
- convergence, steady state, or a universal method ranking;
- controlled causal identification;
- OOD generalization.

The merged successor implementation refreshes matched TD/GAE tables from a changing critic and keeps the critic update active. `EXT-H-E7-SQEXP-GAE-01` therefore remains `not_run` until that 96-branch joint-critic protocol is executed and terminal-audited.

## 8. Committed compact evidence

- `experiments/results/e7_sqexp_gae_frozen_critic_diagnostic_20260718/RESULT_SUMMARY.json`
- `experiments/results/e7_sqexp_gae_frozen_critic_diagnostic_20260718/group_summary.csv`
- `experiments/results/e7_sqexp_gae_frozen_critic_diagnostic_20260718/gae_vs_td.csv`
- `experiments/results/e7_sqexp_gae_frozen_critic_diagnostic_20260718/advantage_diagnostics_summary.csv`
- `experiments/results/e7_sqexp_gae_frozen_critic_diagnostic_20260718/terminal_audit.json`
- `experiments/results/e7_sqexp_gae_frozen_critic_diagnostic_20260718/PACKAGE_SHA256.txt`

The full external ZIP is not committed; its SHA-256 above binds the compact archive to the reviewed package.
