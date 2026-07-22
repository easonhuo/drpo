# E8 AsymRE delta-v joint result closure

## Scope

This document closes the two linked development pilots `EXT-C-E8-ORACLE-OFFLINE-V2-ASYMRE-DELTAV-SCAN-0.5B-01` and `EXT-C-E8-ORACLE-OFFLINE-V2-ASYMRE-DELTAV-BOUNDARY-DENSE-0.5B-01`.
It does not merge their scientific code PRs, start another scan, or promote either run to
formal method-ranking evidence.

## Frozen protocol

Both runs use Qwen2.5-0.5B-Instruct with fresh LoRA, the same 6000-row oracle-offline v2
bank, the same structurally disjoint `val.jsonl` held-out evaluation split, seed offsets
`4000,5000`, fixed 1200 training steps, and the objective

`(1-delta_v) * positive_lp - (1+delta_v) * negative_lp`.

No value network, learned baseline, remoteness taper, early stopping, or separate
`test.jsonl` evaluation entered either response curve. Paper-facing reporting is the mean
Pass@8 over steps 800, 900, 1000, 1100, and 1200 plus the step-1200 terminal metric.
Validation-selected best checkpoints remain supplementary diagnostics only.

## Completeness

The first scan completed 16/16 cells and the dense boundary scan completed 16/16 cells.
Both terminal audits passed. Across 32 cells, NaN/Inf numerical failures were 0 and test
data were not used.

## Joint response

| Run / delta_v | Mean late-window Pass@8 | Terminal Pass@8 | Terminal valid |
|---|---:|---:|---:|
| first / -1.00 | 0.1438 | 0.1430 | 0.9960 |
| dense / -0.95 | 0.1034 | 0.1010 | 0.9830 |
| dense / -0.90 | 0.1038 | 0.1030 | 0.9910 |
| dense / -0.85 | 0.1000 | 0.0980 | 0.8520 |
| dense / -0.80 | 0.1054 | 0.1050 | 0.8620 |
| dense / -0.70 | 0.0870 | 0.0840 | 0.8860 |
| dense / -0.60 | 0.0990 | 0.0980 | 0.9040 |
| first / -0.50 | 0.1040 | 0.1090 | 0.8860 |
| dense / -0.50 | 0.1090 | 0.1090 | 0.8720 |
| first / -0.30 | 0.1040 | 0.1000 | 0.8700 |
| first / -0.20 | 0.0864 | 0.0910 | 0.7920 |
| first / -0.10 | 0.0732 | 0.0810 | 0.6170 |
| first / -0.05 | 0.0482 | 0.0580 | 0.2580 |
| first / 0.00 | 0.0264 | 0.0260 | 0.2830 |
| first / 0.10 | 0.0180 | 0.0200 | 0.2480 |

The `delta_v=-1` boundary removes the negative term and has mean late-window Pass@8
`0.1438`. Every tested active-negative point is lower on both paired seeds. The strongest
observed active-negative aggregate is the dense `delta_v=-0.5` point at `0.1090`, a
difference of `-0.0348` from the zero-negative boundary.

At `delta_v=-0.95` and `-0.90`, terminal valid rate remains about 98--99% while late-window
Pass@8 has already fallen by about four percentage points. Therefore the first visible
effect is task-performance degradation, not a NaN/Inf event or an already-established
formal structure boundary. From approximately `delta_v=-0.85` onward, valid-rate
instability becomes material; the original scan's `-0.20` through `0.10` region combines
continued task degradation with severe valid-expression loss.

## Event separation

- Task performance: reported through the fixed late window and terminal metrics.
- Support/structure behavior: no formal boundary threshold was registered; valid rate is
  retained as a diagnostic and is not relabeled as a formal support event.
- Numerical failure: 0/32 NaN/Inf events.

## Reproducibility and provenance

The first run is bound to resolvable source commit
`396601fb3b041cd09743dec4f9a1f925ea71bae1`, scientific implementation
`1ff3628baf720adbf14754e7ed7835fefee099d0`, results commit
`2f9e6d366596f609a91d46f2db014c82e57c04a2`, and READY manifest SHA-256
`4384eacdea314c7f13f7bad500a1860c2aa601b53ffcb0ffa51f48193a66976c`.

The dense package records source commit
`600a2c3400bd6109d80e566c7a5d5eeaf64518d8`, which was not resolvable from GitHub at
closure time. The run remains bound to results commit
`ecdda6392ab9bcca224f15482c245c6d7c06597a`, READY manifest SHA-256
`696d4f72c0bf13e67b0744397d330c27689f163eaf414ed81d92e4f4e097babb`, source artifact
ZIP SHA-256 `703485c06d1dacc236ff5801ba94bc6e5522a988a1d9a6d240c5258a09a6275a`,
reviewed implementation commit `d865b5ce41034db037ea80804bb8d35989549eda`, and Draft PR #229 head
`2eddd6139eb07754cb9355fad6a8a37ad59d3d4e`. This explicit source-resolution limitation
prevents formal-evidence promotion but does not erase the delivered pilot observations.

## Locked interpretation and next priority

The tested global AsymRE coefficient reallocation does not reveal an interior point that
beats the zero-negative boundary. Additional search inside `[-1,-0.95]` is low priority
and would answer only a narrow onset-threshold diagnostic, not the current main method
question. The next scientific line may move to TORR/TOPR review after this repository
closure.

This evidence remains `pilot`: two development seeds do not support statistical
significance, fixed 1200 steps do not establish convergence or steady state, and these
runs alone do not establish a formal comparison against remoteness-aware taper or a
cross-task/model ranking.
