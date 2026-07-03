# D-U1 E6 separation-grid development pilot

## Scope

- Experiment: `D-U1-E6-CARTESIAN-TAPER-01`
- Status: development calibration / pilot only
- Base repository commit recorded by the recovered revision-3 work: `828b7db5fcdf8ee7ad9b0d87693955081e39c27e`
- Development seeds: `0,1,2,3,4`
- Formal seeds `200--219`: not accessed
- Horizon: 8000 optimizer steps per run
- Active methods: Positive-only, All-negative, Global matched, Reciprocal-linear, Reciprocal-quadratic, Exponential-quadratic
- Quartic: excluded from the active matrix; historical evidence was not deleted
- Grid: `negative_alpha ∈ {0.25,0.5}`, `rarity_logit_anchor_coefficient ∈ {0.25,0.1}`, `rho=0.25`
- Total: 120 runs

## Main result

The useful separation point is:

```text
negative_alpha = 0.5
rarity_logit_anchor_coefficient = 0.25
reference_rare_retention = 0.25
```

| Method | Mean reward | Delta vs Positive-only | Wins vs Positive-only | Hidden-optimal prob. | Prototype support | Rare mass |
|---|---:|---:|---:|---:|---:|---:|
| Positive-only | 0.598032 | 0.000000 | — | 0.011789 | 12.6349 | 0.032725 |
| All-negative | 0.577771 | -0.020261 | 0/5 | 0.000314 | 7.4648 | 0.000730 |
| Global matched | 0.594923 | -0.003110 | 0/5 | 0.008039 | 11.3710 | 0.019221 |
| Reciprocal-linear | 0.596025 | -0.002007 | 1/5 | 0.007908 | 10.9195 | 0.017835 |
| Reciprocal-quadratic | 0.598047 | +0.000015 | 3/5 | 0.008977 | 10.9080 | 0.020458 |
| Exponential-quadratic | 0.600682 | +0.002649 | 4/5 | 0.010476 | 10.9993 | 0.024591 |

At this point:

- Exp minus Quadratic reward: `+0.002634`, Exp wins `5/5` seeds.
- Exp minus Global reward: `+0.005759`, Exp wins `5/5` seeds.
- Exp minus Positive-only reward: `+0.002649`, Exp wins `4/5`; the remaining seed is effectively tied (`-0.000055`).
- All 30 runs in this cell reached `terminal_plateau`.
- Environment validity failures: `0`.
- Support-boundary events: `0`.
- NaN/Inf failures: `0`.

This doubles the Exp-vs-Quadratic separation relative to the original `alpha=0.25, anchor=0.25` point (`+0.001255` to `+0.002634`) without creating a boundary or numerical failure.

## What did not work

Reducing the anchor to `0.1` was not the right way to create separation:

- At `alpha=0.25, anchor=0.1`, every controlled method remained below Positive-only.
- At `alpha=0.5, anchor=0.1`, All-negative hit the support boundary in all `5/5` seeds; Exp recovered most performance but still remained slightly below Positive-only on average.

Therefore the practical conclusion is:

> Increase negative pressure from `0.25` to `0.5`, but retain the `0.25` rarity anchor. Do not weaken the anchor to `0.1`.

## Full grid mean rewards

| Alpha | Anchor | Positive | All-negative | Global | Linear | Quadratic | Exp |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 0.25 | 0.598032 | 0.589708 | 0.596317 | 0.597187 | 0.598157 | 0.599411 |
| 0.25 | 0.10 | 0.598032 | 0.583358 | 0.590598 | 0.591609 | 0.593558 | 0.596533 |
| 0.50 | 0.25 | 0.598032 | 0.577771 | 0.594923 | 0.596025 | 0.598047 | 0.600682 |
| 0.50 | 0.10 | 0.598032 | 0.577621 | 0.587055 | 0.588517 | 0.591756 | 0.597039 |

## Integrity and scientific boundary

- Registered runs present: `120/120`.
- Terminal classes: `115 terminal_plateau`, `5 support_boundary`.
- The five boundary outcomes are exactly the five All-negative runs at `alpha=0.5, anchor=0.1`.
- Environment validity failures: `0`.
- NaN/Inf failures: `0`.
- Task-collapse events under the registered `0.2 × Positive-only` threshold: `0`.
- Minimum dynamic oracle utility-sign validity: `1.0`.
- Maximum stepwise Global budget-match error: `4.440892098500626e-16`.
- Quartic active runs: `0`.

This remains development evidence. It supports choosing a development parameter point, not a formal method ranking. A separate documented formal freeze and user approval would still be required before accessing held-out formal seeds.
