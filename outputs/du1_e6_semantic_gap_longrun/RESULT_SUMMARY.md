# D-U1 E6 Semantic-Gap Long-Run Result Summary

## Identity and status

- Experiment: `D-U1-E6-SEMANTIC-GAP-LONGRUN-01`
- Scientific run commit: `0907c3c0e76fc836c2bf2b752abf554c17f79f22`
- Repository-closure base: `fa225510e3e3e4616f36d8f586611aa6af79bf6e`
- Formal held-out seeds: `150--169`; sandbox seeds `900--909` are absent from formal aggregation.
- Completed: `100/100` registered method-seed runs with all required outputs and terminal audits accepted.
- Scientific status: **finite-step validated**. Only `45/100` runs are formal terminal plateaus; `55/100` remain persistent-drift-or-inconclusive, so no steady-state method ranking is allowed.
- Event separation: task-performance collapse `0/100`; support/temperature boundary `0/100`; NaN/Inf numerical failure `0/100`.
- Terminology: same-distribution held-out-context generalization with a structured state-action support gap, **not state-distribution OOD generalization**.

## Registered-horizon result

| Alpha | 4k reward | 8k reward | 16k reward | 24k reward | 32k reward | 32k paired difference vs Positive-only | 32k wins/losses | Terminal plateau |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.755119 | 0.749738 | 0.746320 | 0.743028 | 0.741309 | 0.000000 | reference | 20/20 |
| 0.25 | 0.794441 | 0.786909 | 0.777792 | 0.770590 | 0.766269 | +0.024960 | 20/0 | 20/20 |
| 0.50 | 0.827152 | 0.811720 | 0.790033 | 0.776419 | 0.765975 | +0.024666 | 20/0 | 5/20 |
| 0.75 | 0.825493 | 0.798410 | 0.767609 | 0.752034 | 0.739330 | -0.001978 | 9/11 | 0/20 |
| 1.00 | 0.759061 | 0.735997 | 0.707152 | 0.689802 | 0.680224 | -0.061085 | 0/20 | 0/20 |

The frozen trajectory supports the registered qualitative claim without supporting a full steady-state ranking. `alpha=0.25` and `alpha=0.50` beat Positive-only on all 20 paired seeds at 32k. The unsuppressed `alpha=1` branch changes from approximately tied at 4k to progressively worse gaps at 8k, 16k, 24k, and 32k (`-0.013741`, `-0.039167`, `-0.053227`, `-0.061085`), losing all 20 paired seeds from 8k onward. `alpha=0.75` shows a finite-horizon reversal: it is beneficial through 24k on average but is approximately tied/slightly worse by 32k and remains non-terminal in all seeds.

## Scientific interpretation and limits

The result strengthens the controlled categorical evidence that a Positive-only policy can have a lower finite-horizon overall-reward ceiling, moderate retained negative pressure can improve held-out-context reward, and leaving the original negative gradient unsuppressed can cause progressively larger task-reward degradation. It does **not** establish that `alpha=0.25` is a universal optimum, that all methods have reached steady state, that the support-gap protocol is OOD generalization, or that categorical policies obey the Gaussian quadratic far-field law.

The raw-complete ZIP is immutable experiment evidence and is intentionally rejected by `drpo-update`. Repository closure uses the compact files in this directory; the 33.6 MB trajectory file remains indexed at its registered persistent-local location rather than copied into Git.
