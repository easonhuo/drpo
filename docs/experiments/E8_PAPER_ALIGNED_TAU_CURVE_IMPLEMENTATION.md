# E8 paper-aligned tau curve implementation note

## Reuse decision

No Python file is added. The implementation extends the existing paper-aligned
profile registry and runtime adapter in place.

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
autotuned launcher, or one-click shell entrypoint.

## Code-size gate

Relative to source commit `23f88c34e8bf7d8b3056324d0a55989c65d42456`,
production Python changes are limited to the existing common and runtime adapter
files. The final diff must remain at or below 100 changed production-Python
lines. Any new Python file requires separate user approval.
