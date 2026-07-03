# D-U1 E6 utility × rarity Cartesian + TAPER

Experiment: `D-U1-E6-CARTESIAN-TAPER-01`

Protocol revision 4 is the formal freeze that follows the revision-3 development
calibration. The experiment remains a controlled categorical mechanism study and
uses independently sampled train/test contexts from the same standard-normal
state distribution. Its claim is held-out-context generalization, not OOD
generalization.

## Why revision 3 replaced revision 2

Revision 2 produced an exact observed common/rare replica lattice, but deleting a
rare replica could be almost task-neutral because its common twin carried the same
reward. Its fixed geometric utility label could also cease to match the actual
reward derivative after the policy moved. Revision 3 therefore added two fail-closed
mechanism gates:

- sixteen evaluation-only hidden actions share the rare coordinate and contain the
  hidden high-reward support, so global rare-support contraction has a measurable
  task cost;
- useful/unhelpful labels are audited against the current expected-reward derivative
  throughout training, and a run is marked environment-invalid if the sign contract
  fails.

The four observed negative cells remain an exact `utility × learner-relative rarity`
Cartesian product. Common/rare counterparts have the same semantic prototype,
reward, fixed advantage, and sample count. There is no trainable per-action bias.
The positive semantic-family objective is exactly neutral to the shared rarity head.

## Development calibration

Development seeds `0--4` ran a 120-run grid at 8000 optimizer steps:

- `negative_alpha ∈ {0.25, 0.5}`;
- `rarity_logit_anchor_coefficient ∈ {0.25, 0.1}`;
- `reference_rare_retention = 0.25`;
- methods: Positive-only, All-negative, Global matched, Reciprocal-linear,
  Reciprocal-quadratic, and Exponential-quadratic.

Quartic was explored historically but was removed from the active formal matrix
before formal freeze because it did not have a distinct experimental responsibility.
Its historical code and results are preserved.

The frozen point is:

```text
negative_alpha = 0.5
rarity_logit_anchor_coefficient = 0.25
reference_rare_retention = 0.25
```

The selection rule was not “make Exp win.” It was the highest registered negative
pressure that passed the environment-validity, support-boundary, numerical, and
terminal gates. Reducing the anchor to `0.1` was rejected because All-negative hit
the support boundary in all five development seeds at `alpha=0.5`.

Development evidence is pilot-only. It selects a formal protocol and is not a
formal method ranking.

## Formal revision-4 protocol

Formal execution uses held-out seeds `200--219`, never development seeds, with six
methods and 8000 steps per run:

1. `positive_only`
2. `all_negative`
3. `global_matched`
4. `reciprocal_linear_distance`
5. `reciprocal_quadratic_distance`
6. `exponential_quadratic_distance`

This yields `6 × 20 = 120` formal runs. All parameters, data geometry, seeds,
thresholds, terminal windows, and method formulas are frozen before formal seed
access. No retuning is allowed after launch, and no method winner is assumed.

## Taper definitions

Let

```text
u = relu((current_surprisal - initial_common_median)
         / (initial_rare_median - initial_common_median))
```

The active selective methods are:

- Reciprocal-linear distance: `1 / (1 + λ sqrt(u))`
- Reciprocal-quadratic distance: `1 / (1 + λ u)`
- Exponential-quadratic distance: `exp(-λ u)`

All retain weight `1` at `u=0` and `0.25` at `u=1`. `global_matched` recomputes a
detached scalar on every optimizer step so that its raw negative-gradient norm
matches the exponential method on the same current model and minibatch.

## Required reporting

Every formal run reports separately:

- task-performance collapse;
- prototype-support or rarity-mass boundary events;
- NaN/Inf numerical failure;
- environment-validity failure;
- terminal plateau versus persistent drift/inconclusive.

A completed fixed horizon is not automatically a steady state. Formal claims require
all registered runs, the two-window terminal audit, durable packaging, and binding
to the exact Git commit used for execution.
