# C-U1 E3 Adam terminal audit

- Base commit: `ac286a46b8ffad898dfad0e7e9188b1d2e81052a`
- All checks passed: **True**
- Missing files: **0**

## Aggregate results

| Branch | Method | n | Reward mean [95% CI] | Task collapse | Support contraction | Unexpected expansion | NaN/Inf | Median support onset |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_variance | baseline | 20 | 0.000002 [0.000002, 0.000002] | 20/20 | 0/20 | 0/20 | 0/20 | None |
| fixed_variance | near_zero | 20 | 0.000002 [0.000002, 0.000002] | 20/20 | 0/20 | 0/20 | 0/20 | None |
| fixed_variance | far_zero | 20 | 0.739362 [0.738902, 0.739837] | 0/20 | 0/20 | 0/20 | 0/20 | None |
| fixed_variance | far_cap | 20 | 0.733072 [0.732594, 0.733563] | 0/20 | 0/20 | 0/20 | 0/20 | None |
| fixed_variance | global_scale | 20 | 0.599057 [0.598634, 0.599475] | 0/20 | 0/20 | 0/20 | 0/20 | None |
| fixed_variance | far_to_near | 20 | 0.875323 [0.874994, 0.875616] | 0/20 | 0/20 | 0/20 | 0/20 | None |
| learnable_variance | baseline | 20 | 0.603254 [0.593897, 0.612491] | 0/20 | 20/20 | 0/20 | 0/20 | 73.0 |
| learnable_variance | near_zero | 20 | 0.601992 [0.592785, 0.611119] | 0/20 | 20/20 | 0/20 | 0/20 | 73.0 |
| learnable_variance | far_zero | 20 | 0.652887 [0.651945, 0.653841] | 0/20 | 0/20 | 0/20 | 0/20 | None |
| learnable_variance | far_cap | 20 | 0.652625 [0.651661, 0.653619] | 0/20 | 0/20 | 0/20 | 0/20 | None |
| learnable_variance | global_scale | 20 | 0.642867 [0.641859, 0.643927] | 0/20 | 0/20 | 0/20 | 0/20 | None |

## Gates

- PASS — all expected JSON and trajectory files exist
- PASS — fixed rows 120
- PASS — learnable rows 100
- PASS — all outputs use Adam
- PASS — trajectory support onset matches summaries
- PASS — fixed baseline: task collapse in >=18/20
- PASS — fixed near_zero: task collapse in >=18/20
- PASS — fixed far_zero: zero task collapse
- PASS — fixed far_cap: zero task collapse
- PASS — learnable baseline: support contraction in 20/20
- PASS — learnable near_zero: support contraction in 20/20
- PASS — learnable far_zero: zero support events
- PASS — learnable far_cap: zero support events
- PASS — learnable global_scale: zero support events
- PASS — no unexpected support expansion in learnable branch
- PASS — no NaN/Inf numerical collapse
- PASS — far controls outperform fixed baseline

## Interpretation

- Fixed-variance Baseline/Near-zero are evaluated as task-performance collapse/drift, not as variance or numerical collapse.
- Learnable-variance Baseline/Near-zero are evaluated by the first full-state support-contraction boundary.
- NaN/Inf, support-boundary events, and task collapse are reported separately.
- No result is described as OOD; the test split is same-distribution held-out context.
