# E8 paper-aligned tau response-curve protocol

## Frozen scientific delta

The only objective change relative to the completed linear-`c` runs is the
thresholded coordinate

```text
max(u - tau, 0)
```

inside the existing detached exponential weight. All other scientific and
runtime variables are inherited.

## Matrix and ordering

The configuration lists 32 explicit points in `c`-major, `tau`-minor order:

- `c`: `1.609437912`, `1.897119985`, `2.995732274`, `4.605170186`;
- `tau`: `0`, `0.125`, `0.25`, `0.375`, `0.5`, `0.75`, `1.0`, `1.25`;
- seed offset: `4000`.

The explicit list is the authority. Do not regenerate it from observed results,
change its spacing after seeing a partial curve, substitute another seed, or add
Positive-only cells.

## Gates

1. Static validation must prove 32 unique `(c, tau, seed)` identities.
2. `tau=0` must be exactly equal to the existing linear weight.
3. A representative two-step liveness cell must pass before the full sweep.
4. Liveness is not scientific evidence.
5. The full run must remain validation-only and must not access test data.
6. Fixed 1200 steps are not convergence.
7. Terminal audit and durable results-repository delivery are required.

## Primary analysis

For each fixed `c`, plot the eight-point function

```text
tau -> mean Pass@8 over steps 800, 900, 1000, 1100, 1200
```

The terminal step-1200 Pass@8 is secondary. Pass@64, Greedy, weight diagnostics,
and valid rate are auxiliary.

Predeclared trend interpretations are:

- rise then fall: evidence for an interior near-retention region;
- coherent decline: evidence that additional retention does not help at these
  anchors;
- coherent curve crossing: evidence of a `c x tau` interaction;
- irregular, nonreproducible-looking single-seed points: inconclusive pilot.

The analysis must show raw values and must not smooth away reversals. A single
highest cell is not the scientific target.

## Historical context

The completed Positive-only late-window Pass@8 mean (`0.1398`) may be drawn as a
historical context line. It is not a same-run paired control for this single-seed
matrix. The completed c-curve establishes why the four anchors were selected;
it must not be rewritten as a formal optimum claim.

## Event separation

Report separately:

1. task-performance behavior;
2. valid-structure/support proxy behavior;
3. NaN/Inf numerical failure.

An OOM or infrastructure interruption is also not a NaN/Inf event and must be
preserved as execution failure evidence.

## Follow-up boundary

No second-seed confirmation is automatically authorized. After terminal audit,
a separately documented decision may select sparse tau anchors for seed 5000.
That decision must be based on predeclared curve-shape rules, not on choosing a
favorable isolated cell.
