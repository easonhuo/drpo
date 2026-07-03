# D-U1 E6 utility × surprisal Cartesian + TAPER

Experiment: `D-U1-E6-CARTESIAN-TAPER-01`

Protocol revision 2 repairs the protocol-revision-1 pilot environment before any
formal run. The historical E6 result is preserved. The old Cartesian pilot used
two semantically identical action IDs separated by a trainable per-action bias.
That construction did match reward and utility, but rarity mostly acted through a
private bias coordinate, Positive-only itself widened the replica gap, and an
action-ID support loss could merely delete a redundant copy. Those pilot results
remain engineering provenance and are not eligible for method ranking.

## Repaired Cartesian environment

There are 32 semantic prototypes and two categorical replicas per prototype. Each
common/rare pair has exactly the same reward, fixed advantage, and ground-truth
directional utility. Rarity is now an orthogonal **shared contextual coordinate**:

- the policy contains no trainable per-action bias;
- a shared state-conditioned rarity head shifts all common/rare pairs;
- the head is zero-initialized around a fixed initial logit gap;
- rare/common negative updates therefore act through shared parameters and may
  affect other actions rather than only their own private bias.

The four negative cells are:

- `useful_common`
- `useful_rare`
- `unhelpful_common`
- `unhelpful_rare`

All four advantages equal `-1`, all cell counts match, and common/rare roles are
reassigned from the current pair probabilities at every update. The role assignment
and taper weight are stop-gradient.

Positive training is replica-neutral. It maximizes the probability of the semantic
family by taking `logsumexp` over both replicas, so Positive-only does not create a
common/rare preference by construction.

## Mandatory environment gates

Before training, every seed must pass:

1. no trainable per-action bias;
2. positive-family loss has zero rarity-head gradient within tolerance;
3. positive-family probability is invariant to a shared rarity shift;
4. rare/common negative samples have matched reward, utility, advantage, and count;
5. rare negative updates produce a materially larger gradient on the shared rarity
   head than common negative updates.

Evaluation reports both action-ID support and semantic-prototype support, plus total
common/rare probability mass. Task collapse, prototype-support boundary,
rarity-mass boundary, and NaN/Inf failure remain separate events.

A quadratic trust-region anchor on the shared rarity residual gives the
development protocol a finite output-level objective without changing the initial
common/rare gap. The anchor is zero at initialization and grows quadratically,
whereas repeated negative log-probability pressure grows only linearly in the
rarity coordinate. This replaces the earlier forward-KL draft, whose restoring
force was too weak when the reference rare mass was already small. The anchor
coefficient, negative alpha, rarity retention, horizon, and terminal tolerances
are not yet formally frozen.

## Taper definitions

Let

```text
S = relu((current_surprisal - initial_common_median)
         / (initial_rare_median - initial_common_median))
```

The method names follow their continuous-distance interpretation, using
`surprisal ∝ distance²`:

- `reciprocal_linear_distance`: `1 / (1 + lambda * sqrt(S))`
- `reciprocal_quadratic_distance`: `1 / (1 + lambda * S)`
- `reciprocal_quartic_distance`: `1 / (1 + lambda * S²)`
- `exponential_quadratic_distance`: `exp(-lambda * S)`

All methods retain weight `1` at `S=0` and the configured reference retention at
`S=1`. `global_matched` recomputes a detached scalar every optimizer step so that
its raw negative-gradient L2 norm matches the exponential method on the same
current model and minibatch. Adam parameter-update norms are recorded but are not
claimed to be matched.

## Current execution gate

Formal execution is intentionally blocked. Development seeds `0--4` must first run
the registered calibration over candidate negative alpha, reference rarity
retention, and rarity-logit-anchor coefficient. Formal seeds `200--219` must remain
untouched until an independent formal-freeze update records the selected values,
horizon, and terminal thresholds.

Smoke and unit tests are engineering evidence only. They neither select a method nor
change the experiment status from `not_run`.
## Protocol-revision-2 engineering acceptance

The repaired environment was checked on development seeds `0,1,2` with six core arms for 8000 steps. This is an engineering diagnostic, not a scientific result or method ranking. The checks observed exact Positive-only neutrality on the rarity coordinate, zero family-likelihood shift error, a minimum rare/common shared-rarity gradient ratio of `54.60x`, stepwise Global raw-gradient matching error below `8.9e-16`, 18/18 terminal plateaus, and no prototype-support, rarity-mass, or NaN/Inf event. Formal execution remains blocked until the registered development calibration and a separate formal freeze are complete.
