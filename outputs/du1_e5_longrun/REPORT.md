# D-U1-E5-LONGRUN-RERUN report

This runner reconstructs the missing historical E5 code from locked handoff records.
It does not claim byte-identical reproduction of the uncommitted legacy runner.

## Direct-softmax diagnostic

| case | p0 | pT | H0 | Hmax | HT | max score |
|---|---:|---:|---:|---:|---:|---:|
| high_probability_negative | 0.8991 | 3.70436e-12 | 0.386 | 0.908703 | 6.45127e-06 | 1.41421 |
| low_probability_negative | 0.0038 | 1.91726e-20 | 0.292 | 0.292 | 4.67918e-09 | 1.41421 |

## Long-run near/far causal reconstruction

| method | task collapse | support collapse | NaN/Inf | historical class match | mean reward | mean entropy |
|---|---:|---:|---:|---:|---:|---:|
| positive_only | 0/20 | 0/20 | 0/20 | 20/20 | 0.275255 | 2.981075 |
| baseline | 20/20 | 20/20 | 0/20 | 20/20 | 0.000698 | 0.386919 |
| near_zero | 20/20 | 20/20 | 0/20 | 20/20 | 0.005860 | 0.396318 |
| far_zero | 0/20 | 0/20 | 0/20 | 20/20 | 0.267381 | 3.074917 |
| far_cap | 0/20 | 0/20 | 0/20 | 20/20 | 0.297148 | 3.002567 |
| global_scale | 0/20 | 20/20 | 0/20 | 20/20 | 0.956783 | 0.664717 |

## Reporting boundary

Task-performance collapse, support/temperature boundary, and NaN/Inf failure are separate events.
The historical numbers are comparison references, not acceptance targets that may be tuned after seeing results.
E5 does not test unseen-action semantic generalization; that remains E6's responsibility.

Maximum causal steps: 20000; audit windows: (10000, 15000) and (15000, 20000).
