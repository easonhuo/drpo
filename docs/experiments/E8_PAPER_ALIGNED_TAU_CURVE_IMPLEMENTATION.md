# E8 paper-aligned tau curve implementation note

## Reuse decision

No Python file is added. The implementation extends the existing paper-aligned
profile registry and runtime adapter in place on the current-main-synchronized
linear-c extension stack.

The inherited trainer still calls the same `continuous_exp_weights` symbol. A
worker-local active tau is selected before the inherited runtime imports the
trainer. Each subprocess receives its cell tau explicitly through `--tau`, so
parallel cells cannot share mutable tau state.

## Backward compatibility

- Round-1 and c-extension cells default to `tau=0`.
- A zero-tau cell keeps the historical cell name.
- At `tau=0`, `max(u-tau,0)=u` because `u` is nonnegative; the linear weight is
  therefore exactly unchanged.
- Nonzero tau is added to the cell name and run identity.

## Unchanged components

No change is made to the base trainer, evaluator, optimizer, scheduler, model
loading, bank loading, loss denominator, checkpoint policy, aggregation core,
autotuned launcher, one-click shell entrypoint, or current RunSpec delivery
implementation.

## Current-main integration

The parent linear-c branch was synchronized with
`main@bb637503e1289f24f7a28e587f50665afb20e0de` before this successor branch was
created. The synchronization preserves the merged RunSpec environment-prefix
compatibility and results-repository delivery behavior.

All test coverage is folded into the existing
`tests/test_countdown_e8_oracle_offline_v2_alpha1_highc_scan.py`; no new `.py`
path is introduced under the current Python-file approval policy.
