# D-U1-E5-LONGRUN-RERUN result summary

- Run commit: `22c5823d66169eb90c256de342e27c5391e464c3`
- Formal scope: 6 methods × 20 seeds × 20000 steps = 120/120 runs
- Scientific status: **long-run validated**, pending application of this repository closure package
- Historical qualitative joint-class match: **120/120**
- NaN/Inf: **0/120**

## Direct-softmax

| case | p0 | pT | entropy pattern | tail surprisal slope/step | max direct-logit score |
|---|---:|---:|---|---:|---:|
| high-probability negative | 0.8991 | 3.70436e-12 | rise then fall | 0.001999998 | 1.414213272 |
| low-probability negative | 0.0038 | 1.91726e-20 | nonincreasing | 0.002000000 | 1.414213562 |

The score remains bounded by `sqrt(2)`, while target surprisal and logit gap continue to grow and probability approaches the simplex boundary.

## D-U1 causal reconstruction

| method | task collapse | support boundary | NaN/Inf | terminal class | mean reward |
|---|---:|---:|---:|---|---:|
| positive_only | 0/20 | 0/20 | 0/20 | stable_bounded 20/20 | 0.275255 |
| baseline | 20/20 | 20/20 | 0/20 | support_boundary 20/20 | 0.000698 |
| near_zero | 20/20 | 20/20 | 0/20 | support_boundary 20/20 | 0.005860 |
| far_zero | 0/20 | 0/20 | 0/20 | stable_bounded 20/20 | 0.267381 |
| far_cap | 0/20 | 0/20 | 0/20 | stable_bounded 20/20 | 0.297148 |
| global_scale | 0/20 | 20/20 | 0/20 | support_boundary 20/20 | 0.956783 |

Global scaling preserves task performance in this environment but does not preserve support, so task failure and support-boundary failure must not be conflated.
