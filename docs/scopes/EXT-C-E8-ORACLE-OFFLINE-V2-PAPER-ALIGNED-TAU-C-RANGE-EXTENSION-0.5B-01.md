# EXT-C-E8-ORACLE-OFFLINE-V2-PAPER-ALIGNED-TAU-C-RANGE-EXTENSION-0.5B-01

## Status

- Lifecycle: code-first development pilot; authoritative registration deferred.
- Result status: `pilot / not_run`.
- Scientific role: Countdown external-validity response-surface extension.
- No training is authorized by this code change alone.

## Question

Does the previously observed tau-response pattern extend one coefficient step below
and above the current bracket while the two old boundary curves provide direct
run-batch bridges?

The frozen weight remains

```text
u = current_sequence_surprisal / 2
w = alpha * exp(-c * max(u - tau, 0))
```

`u` and `w` remain detached. No objective, trainer, optimizer, scheduler, model,
bank, denominator, evaluation split, or horizon changes.

## Frozen matrix

- `alpha = 1`.
- bridge and extension coefficients:
  - lower outer: `c = 1.386294361`;
  - lower bridge: `c = 1.609437912`;
  - upper bridge: `c = 4.605170186`;
  - upper outer: `c = 5.298317367`.
- `tau = {0, 0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.25}`.
- development seed offset: `{4000}`.
- total: `4 c x 8 tau x 1 seed = 32 cells`.
- fixed horizon: `1200` steps; no early stopping.

The repeated `c=1.609437912` and `c=4.605170186` curves are mandatory bridge
controls. They are not redundant search points: they estimate whether the new
run batch can be joined to the completed tau-curve run. The two outer coefficients
extend only one previously used coefficient step in each direction.

## Predecessor evidence

- predecessor experiment:
  `EXT-C-E8-ORACLE-OFFLINE-V2-PAPER-ALIGNED-TAU-CURVE-0.5B-01`;
- predecessor run: `E8_PAPER_ALIGNED_TAU_CURVE_20260717_01`;
- durable results commit:
  `easonhuo/drpo-results@8baae3f728043fc85f14b56c246091e64aeb9dfe`;
- predecessor state: 32/32 cells delivered and terminal-audited, single-seed pilot.

The predecessor's recorded local source commit is not currently resolvable on the
remote repository. The new run must use the remote-resolvable implementation SHA
from this branch and must not copy that unresolved source identity.

## Analysis contract

Primary metric: mean validation Pass@8 over steps `800,900,1000,1100,1200`.
Terminal Pass@8 is secondary. Greedy, Pass@64, valid rate, weight diagnostics,
raw-gradient norm, and optimizer-update norm are auxiliary.

Report separately:

1. task-performance behavior;
2. valid-structure/support proxy behavior;
3. NaN/Inf numerical failure;
4. infrastructure interruption.

The main questions are whether the two bridge curves reproduce their coarse
shape, whether moderate tau retention remains useful at the outer coefficients,
and whether large tau values continue to degrade performance. No exact optimum,
cross-seed ranking, convergence, steady state, significance, or OOD claim is
allowed.

## Implementation boundary

- Add no Python file.
- Extend the existing paper-aligned profile registry only.
- Reuse the existing runtime, trainer, evaluator, launcher, one-click shell script,
  aggregation, terminal audit, and results-repository delivery path.
- Do not overwrite the predecessor output directory or results package.
