# D-U1 E6 Structured Conditional-Gap Formal Result Summary

## Identity and status

- Experiment: `D-U1-E6-CONDITIONAL-GAP-01`
- Scientific run commit: `7a70278f3d6061379c81f33e82d93ead86484908`
- Formal held-out seeds: `130--149`; development seeds `0--1` are absent from formal aggregation.
- Completed: `200/200` frozen method-seed runs; four five-seed recovery checkpoints and all required artifacts are present.
- Scientific status after terminal review: **finite-step validated**. The 8000-step formal horizon and registered extension windows were executed, but only `49/200` runs are terminal plateaus; `151/200` remain persistent-drift-or-inconclusive.
- Event separation: task-performance collapse `77/200`; support/temperature boundary `0/200`; NaN/Inf numerical failure `0/200`. Task collapse therefore occurred without support-boundary or numerical failure in this fixed-concentration protocol.
- Terminology: this is a same-distribution structured state-action support gap, not state-distribution OOD generalization.

## The large conditional gap is behaviorally consequential

Under positive-only training, withholding the complete optimal action group on half of states lowers mean gap reward from `0.731293` in the paired covered control to `0.580133` in the structured-gap condition. The paired difference is `-0.151160`, bootstrap 95% CI `[-0.155861,-0.146638]`, with the gap condition lower in `20/20` seeds. Correct-group probability falls by `-0.453066`, also `20/20`. The policy instead concentrates on the observed proxy-positive group (`0.614637` mean probability). Thus the structured missing block is not a cosmetic holdout: it creates a large conditional-generalization failure while keeping train/test state marginals matched.

## Controlled local repulsion repairs the withheld block, but not the whole task

For structured-gap states, local negative pressure `alpha=0.5` raises mean gap reward from `0.580133` to `0.763974` (`+0.183842`, CI `[+0.180143,+0.187496]`, `20/20` wins) and raises correct-group probability from `0.197383` to `0.609623` (`+0.412241`, `20/20`). This is direct evidence that controlled local negative information can overcome the positive-only conditional imitation ceiling.

The benefit is localized: covered-state reward drops from `0.720214` to `0.434405`, so overall expected reward is `0.050984` lower than structured-gap positive-only in `20/20` seeds. The result supports repair of the withheld block, not a claim that `alpha=0.5` is the globally best policy.

## Excessive local pressure and far pressure produce task-performance collapse

Local `alpha=1.5` collapses task performance in `20/20` structured-gap runs and also `20/20` paired covered-control runs. On gap states its mean reward is `0.075519`, `-0.504614` below positive-only, while trap-group probability rises by `+0.716897` to `0.732509`. Because the covered control also collapses, excessive local pressure is harmful independently of the structured gap; the gap is not the sole cause of this branch.

Adding uncontrolled far pressure (`lambda=4.0`) to local `alpha=0.5` lowers mean gap reward from `0.763974` to `0.178149` (`-0.585825`, `20/20`) and produces task collapse in `20/20` runs. Removing local updates while retaining far pressure (`near-zero`) still collapses `16/20` runs. This supports far pressure as a dominant harmful path in the registered stress condition, while showing that local information is protective rather than causal for that failure.

## Far controls rescue collapse, but no steady-state method winner is established

Far-cap improves gap reward over uncontrolled by `+0.122380` (CI `[+0.115541,+0.130316]`, `20/20`) and reduces collapse from `20/20` to `1/20`. Raw-budget-matched global scaling improves gap reward by `+0.131577` (CI `[+0.123238,+0.139857]`, `20/20`) and reduces collapse to `0/20`. Within this frozen benchmark, Global exceeds Far-cap in gap reward by `+0.009197`, but the CI barely excludes zero (`[+0.000085,+0.017882]`), it wins only `14/20`, and both conditions remain persistent-drift-or-inconclusive in `20/20` runs. This cannot be promoted to a universal method ranking or a stable fixed-point result.

## Paper-use boundary

Allowed:

- a large structured conditional support gap causes a reproducible positive-only failure despite matched state marginals;
- controlled local negative information can strongly improve the withheld optimal group;
- excessive local pressure and uncontrolled far pressure can cause task-performance collapse without support-boundary or NaN/Inf failure;
- targeted far control and matched global scaling substantially rescue the registered far-pressure stress condition.

Not allowed:

- calling this state-distribution OOD generalization;
- claiming the structured gap is necessary for every collapse, since covered `alpha=1.5` also collapses;
- treating task collapse as support collapse or numerical collapse;
- claiming a terminal steady state for the 151 persistent/inconclusive runs;
- claiming Far-cap, Global alpha, Distance, Exp, SBRC, Hybrid, or any family is a universal winner;
- transferring this controlled categorical result directly to Countdown or other Transformer tasks.
