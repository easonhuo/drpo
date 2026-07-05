# Experiments to run for Section 6.4.1

## Purpose

Section 6.4.1 answers the missing question after the causal rescue result:

> If far-field negative feedback is harmful, why not remove all negative feedback?

The required evidence is a controlled strength sweep showing a useful
negative-feedback regime. Small-to-moderate effective negative strength moves
the policy beyond the positive-only support ceiling and improves
held-out-context reward. Larger strength produces over-extrapolation and then
task-performance collapse or boundary events.

## Main experiment

Recommended experiment ID:

```text
C-U1-E4-STRENGTH-SWEEP-FINAL
```

Use the existing C-U1 / E4 controlled environment and frozen protocol settings.
Do not silently change data construction, hidden optimum, seeds,
initialization, optimizer, horizon, task-collapse threshold, boundary
definitions, or terminal audit rules.

## Independent variable

Sweep only the effective negative-repulsion budget relative to positive
attraction:

```text
q/p
```

If implementation uses `alpha`, `negative_scale`, or another knob, export both
the raw knob and the normalized `q/p`.

## Required arms

1. Positive-only baseline: `q/p = 0`.
2. Weak controlled negative feedback: begins moving beyond the positive-only target.
3. Moderate controlled negative feedback: exceeds positive-only held-out-context reward.
4. Strong controlled negative feedback: over-extrapolation, falling reward, larger policy shift.
5. Very strong controlled negative feedback: task-performance collapse and/or boundary events.

Use the registered E4 sweep grid if it already exists. If a denser grid is
needed for the final figure, register it as a new final sweep rather than
silently changing the old protocol.

## Required seed-level metrics

```text
experiment_id
env_id
run_seed
strength_q_over_p
raw_negative_scale
heldout_reward
task_reward
positive_only_ceiling
policy_shift
distance_to_hidden_optimum
over_extrapolation_score
task_collapse
boundary_event
nan_inf
terminal_audit_pass
```

## Aggregated CSV schema for the figure

```text
strength_q_over_p
heldout_reward
heldout_ci_low
heldout_ci_high
policy_shift
positive_only_ceiling
task_collapse_count
boundary_event_count
nan_inf_count
n_seeds
phase
```

## Metric definitions

- **Held-out reward**: held-out-context / unseen-state reward, not OOD reward.
- **Positive-only ceiling**: performance of the positive-only baseline.
- **Policy shift**: normalized displacement of the learned policy from the
  positive-only target. It visualizes how far negative feedback pushes the
  policy beyond the imitation/positive-support solution.
- **Task coll.**: number of seeds with task-performance collapse.
- **Boundary**: number of seeds with support or variance-boundary events.
- **NaN/Inf**: number of seeds with numerical NaN/Inf failure.

Report task-performance collapse, boundary events, and NaN/Inf separately.

## Optional categorical confirmation

If time permits, run the same strength-sweep logic in the D-U1 categorical
control environment:

```text
D-U1-E6-STRENGTH-SWEEP-FINAL
```

This is useful for appendix confirmation or for a two-panel C-U1/D-U1 version.
The main 6.4.1 figure can be built from the C-U1 controlled strength sweep if
space is limited.

## Updating the figure

Replace:

```text
data/fig_6_4_1_phase_transition_template.csv
```

with the real aggregated CSV using the same schema, then run:

```bash
python scripts/plot_6_4_1_phase_transition.py \
  --data data/fig_6_4_1_phase_transition_template.csv \
  --out figures/fig_6_4_1_phase_transition
```
