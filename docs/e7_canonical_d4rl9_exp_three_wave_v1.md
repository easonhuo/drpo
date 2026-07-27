# E7 D4RL-9 Exp Three-Wave Task-Specific Tuning Protocol

## Status and identity

- Experiment ID: `EXT-H-E7-D4RL9-EXP-3WAVE-TUNE-01`
- Source experiment: `EXT-H-E7-BENCH-01`
- Role: external-validity development hyperparameter tuning
- Status before execution: `not_run`
- Execution class: pilot
- Code lineage: stacked on the D4RL-9 GLQ implementation in PR #293
- Result authority: none until the code-first schema-v3 registration transaction is completed and a terminal-audited result package is delivered

This experiment does not establish a formal cross-method ranking, convergence, or a paper-table result.

## Claim

The experiment asks whether Exponential taper performance remains below the
task-specific Global, Reciprocal-Linear, and Reciprocal-Quadratic development
results when Exp is evaluated on the same canonical one-step-TD D4RL runner and
receives an equally dense task-specific coefficient search.

The experiment changes only the Exponential coefficient \(c\). It does not
change the actor/critic implementation, data, training budget, evaluation
cadence, seeds, canonical negative coefficient, distance normalization, or
method-level negative multiplier.

## Frozen method

For negative-advantage samples,

\[
u = d / R_{\mathrm{ref}},
\qquad
w_{\mathrm{Exp}}(u)=\exp(-cu),
\]

with the following fixed values:

- canonical actor coefficient: `canonical_alpha = 0.11`;
- method-level negative multiplier: `negative_scale = 1`;
- effective negative coefficient at zero remoteness: `0.11`;
- reference distance: `R_ref = 2`;
- positive advantages unchanged;
- detached remoteness geometry.

The source trainer argument `--tau=0.5` remains unchanged because it belongs to
the canonical trainer configuration. It is not introduced or retuned as an
additional Exp remoteness threshold.

## Frozen training and evaluation configuration

The source nine-task RunSpec must retain:

- variant: `iqlv_exp_rank`;
- policy updates per branch: `1,000,000`;
- batch size: `256`;
- learning rate: `3e-4`;
- trainer alpha: `0.11`;
- trainer tau: `0.5`;
- temperature: `5.0`;
- evaluation every `50,000` updates;
- `10` evaluation episodes;
- `OMP_NUM_THREADS=1`;
- `MKL_NUM_THREADS=1`.

The nine D4RL-v2 datasets and their registered SHA-256 identities are inherited
unchanged from the canonical source RunSpec.

## Seeds and access boundary

- Development seeds: `200, 201, 202, 203`
- Held-out seeds: `204, 205, 206, 207`

The held-out seeds are not loaded by any of the three tuning waves.

## Matrix

Each wave runs:

\[
9\ \text{tasks}
\times 5\ c\text{-candidates per task}
\times 4\ \text{development seeds}
=180\ \text{branches}.
\]

Three sequential waves therefore run exactly:

\[
3\times180=540\ \text{branches}.
\]

Only Exp is rerun. Global, Reciprocal-Linear, Reciprocal-Quadratic,
Positive-only, and uncontrolled branches are not part of this matrix.

## Three-wave selection protocol

### Wave 1

Wave 1 uses five frozen, task-specific, broad candidates per dataset. Historical
Exp results are used only as prior localization information for the search
range; they are not imported as evidence and they do not determine the current
result.

### Wave 2

For each task, Wave 1 candidates are ranked using the registered primary metric.
Five new candidates are generated around the Wave 1 winner by the frozen
`wave2_multiplier_pool`. Existing candidates are excluded.

### Wave 3

For each task, the combined ten Wave 1+2 candidates are ranked using the same
rule. Five additional candidates are generated around the combined winner by
the frozen `wave3_multiplier_pool`. Existing candidates are excluded.

Wave 2 and Wave 3 generation is deterministic and recorded in each wave's
`WAVE_GRID.json`. Manual post-result insertion, deletion, or replacement of
candidate values is forbidden.

## Selection metric

The primary metric is the mean normalized D4RL score over evaluations at

`750k, 800k, 850k, 900k, 950k, 1M`

policy updates, averaged across the four development seeds.

Tie-breaking is frozen as:

1. maximize cross-seed late-window mean;
2. maximize cross-seed minimum;
3. minimize mean best-to-late-window drop;
4. choose the smaller numeric \(c\).

Each task selects its own Exp coefficient from the final fifteen candidates.
This task-specific selection is required because empirical remoteness
distributions differ across datasets, so one numerical \(c\) does not imply one
common effective retention regime.

## Execution

```bash
bash scripts/run_e7_canonical_d4rl9_exp_three_wave_one_click.sh
```

The launcher runs at most fifteen units in parallel, with four branch workers per
unit, preserving the registered sixty-worker ceiling. Re-execution is resumable.

Configuration-only validation:

```bash
bash scripts/run_e7_canonical_d4rl9_exp_three_wave_one_click.sh --validate-only
```

## Expected outputs

- `outputs/e7/d4rl9_exp_three_wave_run_001/wave_01/WAVE_GRID.json`
- `outputs/e7/d4rl9_exp_three_wave_run_001/wave_01/TERMINAL_AUDIT.json`
- `outputs/e7/d4rl9_exp_three_wave_run_001/wave_02/WAVE_GRID.json`
- `outputs/e7/d4rl9_exp_three_wave_run_001/wave_02/TERMINAL_AUDIT.json`
- `outputs/e7/d4rl9_exp_three_wave_run_001/wave_03/WAVE_GRID.json`
- `outputs/e7/d4rl9_exp_three_wave_run_001/wave_03/TERMINAL_AUDIT.json`
- `outputs/e7/d4rl9_exp_three_wave_run_001/TERMINAL_AUDIT.json`
- `outputs/e7/d4rl9_exp_three_wave_run_001/TASKWISE_SELECTION.json`

Raw branch outputs, generated unit grids, generated one-dataset RunSpecs,
execution plans, logs, completion manifests, and trainer summaries remain under
the corresponding wave directory.

## Terminal audit and reporting boundaries

The terminal audit must report separately:

- task-performance collapse: not classified without a registered threshold;
- support or variance-boundary events: unavailable in the unchanged canonical
  trainer summary;
- NaN/Inf numerical failure: audited through zero exits and finite evaluation
  histories.

A fixed one-million-update horizon is not convergence. Development winners must
be frozen before any separately registered held-out confirmation on seeds
`204--207`.
