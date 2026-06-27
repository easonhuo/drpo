# D-U1 E6 Semantic Long-Run Result Summary

## Identity and status

- Experiment: `D-U1-E6-SEMANTIC-LONGRUN-01`
- Scientific run commit: `eb5e12626026854f44f4698dbc8ed8829e74e0b0`
- Repository-closure base: `a1672d95653139964debdd5c1baf00173722c071`
- Formal held-out seeds: `10--29`; development seeds `0--4` are absent from formal aggregation.
- Completed: `360/360` registered runs, all terminal audits present and accepted.
- Scientific status: **long-run validated** under the frozen 8000-step / formal 2x protocol.
- Event separation: task-performance collapse `0/360`; support/temperature boundary `120/360`; NaN/Inf numerical failure `0/360`.
- Terminology: this is same-distribution held-out-context generalization and unseen-action semantic extrapolation, **not OOD generalization**.

## E6-A: controlled local negatives and the positive-only ceiling

With fixed concentration `8.0`, controlled local-negative pressure is non-monotonic:

| Method | Mean reward | Mean hidden-optimum probability | Mean normalized extrapolation | Support boundary |
|---|---:|---:|---:|---:|
| positive-only | 0.851449 | 0.131065 | -0.197115 | 0/20 |
| local-only alpha=0.25 | 0.872986 | 0.173032 | 0.345313 | 0/20 |
| local-only alpha=0.50 | 0.867436 | 0.173790 | 0.943832 | 0/20 |
| local-only alpha=0.75 | 0.820420 | 0.112169 | 1.484809 | 0/20 |

Relative to positive-only, `alpha=0.25` improves reward by `+0.021538` with paired bootstrap 95% CI `[+0.020573,+0.022449]` and wins `20/20` seeds. `alpha=0.50` improves reward by `+0.015987` with CI `[+0.013682,+0.018125]`, also `20/20`. `alpha=0.75` extrapolates farther but lowers reward by `-0.031028` with CI `[-0.034218,-0.027886]`, losing `20/20` reward comparisons. Thus the formal result supports a positive-only imitation ceiling, useful controlled local repulsion, and a reversal under excessive pressure; it does not support “more negative pressure is always better.”

## E6-B: reward and support state must be reported separately

For learnable concentration, `far_zero` raises mean reward from `0.864392` to `0.885318` while avoiding the registered support/temperature boundary (`0/20` for both positive-only and far-zero). However, only `5/20` far-zero runs are terminal plateaus; `15/20` remain persistent-drift-or-inconclusive under the frozen terminal classifier.

`uncontrolled`, `near_zero`, `far_cap`, and `budget_matched_global` each trigger support/temperature boundary in `20/20` seeds even though their mean rewards remain high (`0.892346--0.907836`). In particular, far-cap versus uncontrolled has a negligible paired reward difference of `+0.000138`, CI `[-0.000158,+0.000460]`, and budget-matched global versus far-cap is `+0.000076`, CI `[-0.000332,+0.000452]`. The run therefore does **not** establish Far-cap, Global alpha, or uncontrolled training as a safe winner. High task reward can coexist with a support boundary.

## E6-C: semantic-alignment exclusion control

Holding the reward-side catalogue and task fixed while shuffling only policy-side semantic alignment sharply reduces performance. For every registered method, aligned reward exceeds shuffled reward in `20/20` paired seeds. Mean aligned-minus-shuffled reward differences are:

- positive-only: `+0.336245`;
- far-zero: `+0.354120`;
- uncontrolled: `+0.372520`;
- far-cap: `+0.372657`.

The same `20/20` direction holds for hidden-optimum probability. This supports policy-side shared semantic alignment as necessary for the observed unseen-action benefit in this controlled D-U1 environment. It does not establish Transformer external validity or cross-task method superiority.

## Paper-use boundary

Allowed:

- controlled local negative gradients can beat positive-only on held-out contexts and unseen actions in aligned D-U1;
- excessive local pressure reverses the benefit;
- support/temperature boundary events can occur without task-performance or numerical collapse;
- shuffling policy-side semantic alignment removes the structured benefit.

Not allowed:

- calling the protocol OOD generalization;
- treating support boundary as NaN/Inf collapse;
- claiming far-field pressure is the sole cause of every D-U1 failure;
- claiming Far-cap, Global alpha, Distance, Exp, SBRC, Hybrid, or any other family is a universal winner;
- transferring this controlled categorical result directly to Hopper or Countdown.
