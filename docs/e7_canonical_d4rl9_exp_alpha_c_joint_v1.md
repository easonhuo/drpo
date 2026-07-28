# E7 D4RL-9 Exponential joint negative-scale and distance-coefficient tuning

## Status and scope

- Experiment ID: `EXT-H-E7-D4RL9-EXP-ALPHA-C-JOINT-TUNE-01`
- Source experiment: `EXT-H-E7-BENCH-01`
- Predecessor tuning experiment: `EXT-H-E7-D4RL9-EXP-3WAVE-TUNE-01`
- Scientific status before execution: `not_run`
- Execution class: development `pilot`
- Role: external-validity hyperparameter tuning on the nine public D4RL locomotion tasks
- This protocol does not provide controlled causal identification, cross-method ranking, steady-state evidence, or held-out confirmation.

The predecessor experiment fixed the Exponential controller's global negative-scale multiplier at `1.0` and tuned only the distance coefficient `c`. Its development result closed the local `c` neighborhood for eight tasks, while `walker2d-medium-replay-v2` retained a lower-bound winner. The Global control sweep also showed that the preferred constant negative scale varies substantially by task. This successor therefore tunes the Exponential controller in the two-dimensional form

\[
w_{\mathrm{neg}}(d)
=
0.11\,s\,
\exp\!\left[-c\left(\frac{d}{2}\right)\right],
\]

where `s=negative_scale` controls the overall negative-gradient magnitude and `c=exponential_coefficient` controls distance decay. Positive advantages remain unchanged. The standardized-distance geometry is detached.

## Frozen canonical training contract

The experiment reuses the same canonical one-step-TD joint actor--critic path and exact source RunSpec as the GLQ and predecessor Exp development sweeps.

Frozen trainer settings:

- variant: `iqlv_exp_rank`;
- trainer alpha: `0.11`;
- expectile `tau=0.5`;
- trainer temperature `5.0`;
- `1,000,000` policy updates;
- batch size `256`;
- learning rate `3e-4`;
- evaluation every `50,000` updates;
- `10` evaluation episodes;
- reference distance `R_ref=2`;
- development seeds `200,201,202,203`;
- held-out seeds `204,205,206,207` remain untouched.

The trainer alpha remains `0.11`; the injected near-field negative coefficient is `0.11 * negative_scale`. Setting `negative_scale=0` is an exact zero-negative/Positive-only-equivalent anchor even though the injected method identity remains `exponential`.

## Matrix

The matrix is frozen at

\[
9\ \text{tasks}
\times
15\ (s,c)\ \text{candidates}
\times
4\ \text{development seeds}
=
540\ \text{branches}.
\]

Every task has exactly fifteen unique `(negative_scale, exponential_coefficient)` pairs.

### Eight tasks with a closed predecessor `c` neighborhood

For these tasks, the search is alpha-primary with a local `c` recheck:

1. one exact zero-negative anchor `(s=0,c=c_star)`;
2. four nonzero task-specific scales crossed with `c in {0.5*c_star, c_star, 2*c_star}`;
3. two additional points at the task's priority scale with `c in {0.75*c_star, 1.5*c_star}`.

| Dataset | predecessor `c_star` | four nonzero scales | priority scale |
|---|---:|---|---:|
| `hopper-medium-v2` | 0.15 | 0.02, 0.065, 0.2, 1.0 | 0.065 |
| `hopper-medium-replay-v2` | 0.122666666667 | 0.3, 0.6, 1.0, 2.0 | 1.0 |
| `hopper-medium-expert-v2` | 0.425 | 0.003, 0.01, 0.1, 1.0 | 0.01 |
| `walker2d-medium-v2` | 0.2 | 0.001, 0.003, 0.03, 1.0 | 0.003 |
| `walker2d-medium-expert-v2` | 0.64 | 0.03, 0.1, 0.3, 1.0 | 0.1 |
| `halfcheetah-medium-v2` | 0.0416666666666 | 0.005, 0.02, 0.1, 1.0 | 0.02 |
| `halfcheetah-medium-replay-v2` | 0.0046875 | 0.003, 0.01, 0.05, 1.0 | 0.01 |
| `halfcheetah-medium-expert-v2` | 2.3 | 0.01, 0.03, 0.1, 1.0 | 0.03 |

The `(s=1,c=c_star)` predecessor winner is included for every task.

### `walker2d-medium-replay-v2` lower-`c` boundary closure

This task uses an explicit matrix because the predecessor winner `c=0.0096` was the lowest combined candidate:

- `s=1.0`, `c in {0, 0.0006, 0.0012, 0.0024, 0.0048, 0.0072, 0.0096}`;
- `s=0.003`, `c in {0, 0.0024, 0.0096}`;
- `s=0.03`, `c in {0, 0.0024, 0.0096}`;
- `(s=0.1,c=0.0024)`;
- exact zero-negative anchor `(s=0,c=0.0096)`.

Here `c=0` is the exact Global-shape limit of the Exponential controller, not a cross-method Global rerun.

## Selection rule

Selection is per dataset over the fifteen development candidates:

1. maximize mean normalized return over evaluations at `750k,800k,850k,900k,950k,1M`;
2. maximize the minimum seed-level late-window mean;
3. minimize mean best-to-late-window drop;
4. choose the smaller `negative_scale`;
5. choose the smaller `exponential_coefficient`;
6. choose the earlier frozen candidate index.

The final two rules are deterministic tie-breakers only. The experiment does not assume that smaller scale or smaller `c` is scientifically superior.

## Terminal audit and event separation

Every branch must expose exactly twenty evaluations from `50k` through `1M`, complete with finite values and zero process exit status. The root audit records:

- late-window mean, minimum, maximum, and standard deviation;
- final score;
- best score and step;
- best-to-final and best-to-late drops;
- terminal slope per `100k` updates;
- per-seed branch rows and per-candidate aggregates;
- exact bindings to the generated execution plan and candidate matrix.

The fixed one-million-update horizon is not convergence. Without a registered task-collapse threshold, task-performance collapse remains unclassified. The unchanged trainer does not expose a support/variance boundary metric. NaN/Inf numerical failure is reported separately through finite-history and zero-exit checks.

## Parallel execution

The launcher generates `135` one-dataset/one-pair units. Each unit runs four seeds with four workers. Up to fifteen units execute concurrently, for at most sixty workers.

The launcher supports:

```bash
bash scripts/run_e7_canonical_d4rl9_exp_alpha_c_joint_one_click.sh --validate-only
```

and a plan-generation check that does not start training:

```bash
CONTRACT=/root/d4rl2/configs/e7_canonical_contract_9task.json \
SOURCE_RUN_SPEC=/root/d4rl2/configs/e7_canonical_9task_full_grid_run_spec_v1.json \
bash scripts/run_e7_canonical_d4rl9_exp_alpha_c_joint_one_click.sh --prepare-only
```

The actual development run uses the same command without `--prepare-only`.

## Outputs

Default output root:

`outputs/e7/d4rl9_exp_alpha_c_joint_run_001`

Required root outputs:

- `EXECUTION_PLAN.json`;
- `CANDIDATE_MATRIX.json`;
- `TERMINAL_AUDIT.json`;
- `TASKWISE_SELECTION.json`;
- generated one-unit RunSpecs and grids;
- per-branch completion manifests, logs, and trainer summaries.

## Prohibited claims

This experiment alone cannot support:

- a formal D4RL-9 table;
- cross-method ranking;
- held-out-seed confirmation;
- convergence or steady-state claims;
- a universal best alpha or `c`;
- a claim that reducing negative scale must improve every task;
- a claim that GAE or global alpha is the unique cause of historical Hopper behavior.

The output is development tuning evidence only. Parameters must be frozen before any separately registered held-out confirmation using seeds `204--207`.
