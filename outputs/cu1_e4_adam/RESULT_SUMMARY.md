# C-U1 E4 unified-Adam result report

## Identity

- Experiment ID: `C-U1-E4-ADAM-RERUN`
- Run/base commit: `d699bb6b1d0093d8a9b935fd6c67f049fc3c3df0`
- Formal seeds: `50--69`
- Optimizer: Adam, frozen E4 protocol
- Runtime: 30.25 minutes on CPU
- Raw rows: fixed variance 160, learnable variance 160, controls 60, variance robustness 45
- Scientific status: **finite-step validated**
- Scope: same-distribution held-out-context generalization; not OOD

## Main result

The finite-horizon curve supports the proposed non-monotonic story. Positive-only (`alpha=0`) reaches reward `0.646988`. Moderate fixed-variance negative pressure improves reward in every paired seed, peaking on the registered grid at `alpha=1.00` with reward `0.991703` and normalized displacement `1.007808`. Stronger pressure reverses the gain: `alpha=1.50` and `1.75` show task-performance collapse in 20/20 seeds.

The learnable-variance branch independently shows support contraction: `alpha=0.40` reaches the negative log-sigma boundary in 18/20 seeds, and `alpha=0.50` in 20/20. No branch produces NaN/Inf or unexpected positive support expansion.

## Critical limitation

The beneficial branch is **not terminally validated** under the frozen residual audit. Fixed-variance `alpha=1.00` passes both stationary audits in only 3/20 seeds. The finite-horizon reward therefore cannot be called a stable fixed-point result. This is the reason the experiment status is `finite_step_validated`, not `long_run_validated`.

## Controls

- `uncontrolled_all`: reward `0.000000`, task failure 20/20.
- `far_cap`: reward `0.995224`, task failure 0/20.
- `budget_matched_global`: reward `0.502925`, task failure 0/20.

The control comparison is descriptive and mechanistic. Method ranking was not pre-registered, and raw-gradient matching is not Adam-update matching.

## Paper-use boundary

The current result can support a finite-horizon phase-transition figure and the separation of task collapse, support contraction, and numerical failure. It cannot yet support the sentence that moderate negative gradients converge to a stable beneficial fixed point under the registered Adam protocol.
