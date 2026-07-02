# D-U1 E6 utility × surprisal Cartesian + TAPER

Formal experiment: `D-U1-E6-CARTESIAN-TAPER-01`

This successor preserves the historical E6 result and fixes its identification
confound. The old environment used semantic `local/far` roles, so directional
utility and policy rarity were not an exact Cartesian product. The new environment
uses 32 semantic prototypes, each duplicated into a common and rare categorical
action. A replica pair has identical reward embedding, directional utility, and
fixed negative advantage. A frozen copy of the initialized semantic policy is
subtracted from the trainable semantic logits, so the initial useful/unhelpful
probabilities are exactly matched within each rarity level; only the fixed
common/rare action-logit bias differs.

The four negative cells are:

- `useful_common`
- `useful_rare`
- `unhelpful_common`
- `unhelpful_rare`

Each context contains one sample in every cell. All four negative advantages are
exactly `-1`, and sample counts are matched.

`common` and `rare` are learner-relative roles, not permanent action labels. At
every update the higher-probability member of each utility-matched replica pair is
the current common action and the lower-probability member is the current rare
action. The discrete role selection is stop-gradient. When mechanism methods zero
cells, the remaining cells keep their original one-quarter coefficient; they are
not renormalized upward.

## Scientific blocks

The mechanism block compares Positive-only, each cell alone, each utility/rareness
marginal, and all negatives. It asks whether rarity is harmful at fixed utility and
whether utility is beneficial at fixed rarity.

The method block uses the same data, initialization, batch stream, and seeds. It
compares:

- `global_matched`
- `reciprocal_linear`
- `reciprocal_quadratic`
- `exponential`

The calibration constants are estimated once per seed before training, while the
surprisal itself is recomputed from the current learner at every update:

```text
u = relu((surprisal - common_median) / (rare_median - common_median))
```

The coordinate is then frozen across paired methods. All selective tapers retain
weight `1` at `u=0` and weight `0.25` at `u=1`. The matched-global control freezes a
single scalar whose initial raw negative-gradient norm matches the exponential
method.

## Formal launch

After this update is applied and committed on clean `main`:

```bash
python3 scripts/run_experiment_guard_hardened.py \
  --experiment-id D-U1-E6-CARTESIAN-TAPER-01 \
  --repo-root . \
  --output-root experiments/results/D-U1-E6-CARTESIAN-TAPER-01/run_001 \
  --artifact-output artifacts/D-U1-E6-CARTESIAN-TAPER-01_RAW_COMPLETE.zip \
  --run-class formal \
  --expected-commit "$(git rev-parse HEAD)" \
  --require-origin-main-match \
  --required-output RUN_COMPLETE.json \
  --required-output terminal_audit.json \
  --required-output aggregate_summary.json \
  --required-output mechanism_summary.json \
  --required-output taper_summary.json \
  --required-output per_run_summary.csv \
  --required-output formal_protocol_freeze.json \
  --source-file src/drpo/du1_e6_cartesian_taper.py \
  --source-file configs/du1_e6_cartesian_taper.yaml \
  --progress-glob 'checkpoints/*/CHECKPOINT_COMPLETE.json' \
  -- python3 src/drpo/du1_e6_cartesian_taper.py \
  --config configs/du1_e6_cartesian_taper.yaml \
  --output-root experiments/results/D-U1-E6-CARTESIAN-TAPER-01/run_001 \
  --stage formal \
  --device cpu
```

Do not call the formal runner directly. Unit tests and smoke runs are engineering
evidence only and do not change the experiment status from `not_run`.
