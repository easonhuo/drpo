# EXT-H-E7-Q2 scientific closure

## Status

- Experiment: `EXT-H-E7-Q2`
- Role: Hopper learned-critic external mechanism validation
- Scientific status: **long_run_validated**
- Formal run: `formal-20260630T105458Z`
- Run commit: `c5c638b47c945f5a3ecb8243f679caa31a129f9e`
- Seeds: `100--109` (`10/10`)
- Dataset: `hopper-medium-replay-v2`, SHA-256 `e121c5f7c9857a307baa9edc6a2c3b48e85fedb9ac316ecddd0f48ca7ef4e39b`
- Evaluation: Gymnasium `Hopper-v4` compatibility environment with frozen D4RL-v2 normalization; not an exact legacy `mujoco-py` leaderboard reproduction.

## Accepted mechanism result

The formal fixed-budget run completed the shared 100k-step critic, the 100k-step Positive-only initialization, and all five 200k-step branches for every paired seed. After matching negative-advantage magnitude, natural far negatives had a mean standardized-distance ratio of **3.845x**, corrected quadratic log-scale quantity ratio of **14.547x**, and full-parameter negative-gradient ratio of **4.206x**. The corrected log-scale quantity followed the expected radius-squared law with mean log-log slope **2.000000000019**. The matched absolute-advantage ratio was **0.999770x**.

`Signed` and `Near-zero` both produced task collapse in **10/10** seeds, sigma at the frozen lower boundary, and nearly complete mean-action boundary saturation. Removing near-field negatives therefore did not rescue the dynamics. `Far-zero`, `Far-cap`, and dynamic budget-matched Global each exceeded Signed in **10/10 paired seeds**; their mean terminal normalized-return gains over Signed were **21.546**, **10.484**, and **14.779**, respectively.

This closes the E7 mechanism responsibility: in this external Hopper learned-critic setting, anomalous far-field negative gradients are a major transmission path into support contraction and task-performance failure. The near/far split is an identification intervention, not the final algorithmic form; continuous taper design and method benefit remain separate experiments.

## Event separation

| Method | Mean normalized return | Task collapse | Boundary event | Mean boundary fraction | NaN/Inf |
|---|---:|---:|---:|---:|---:|
| Positive-only | 32.231 | 0/10 | 9/10 | 0.1123 | 0/10 |
| Signed | 0.987 | 10/10 | 10/10 | 0.9997 | 0/10 |
| Near-zero | 1.125 | 10/10 | 10/10 | 0.9999 | 0/10 |
| Far-zero | 22.532 | 3/10 | 10/10 | 0.1215 | 0/10 |
| Far-cap | 11.471 | 7/10 | 10/10 | 0.1821 | 0/10 |
| Dynamic Global | 15.766 | 6/10 | 10/10 | 0.1898 | 0/10 |

The binary boundary flag is intentionally not treated as severity. Far-zero's mean boundary fraction is close to Positive-only, whereas Signed and Near-zero are approximately fully saturated.

## Claim boundary

This result does **not** authorize a universal method ranking, a finite steady-state claim, a claim that the current controls outperform Positive-only, or a claim that far-field dynamics are the unique cause of all real-task failures. Positive-only is retained as a deletion-all-negative stability reference; the primary mechanism baseline is Signed.
