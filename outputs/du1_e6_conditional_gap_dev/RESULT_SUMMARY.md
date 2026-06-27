# D-U1 E6 Conditional-Gap Development Pilot

**Experiment:** `D-U1-E6-CONDITIONAL-GAP-DEV-01`
**Scientific status:** pilot
**Formal result:** no

This pilot used development seeds 0--1 for 1000 updates. It validates the new
same-distribution structured state-action support-gap environment and selects a
formal comparison matrix. It is not a long-run result and must not be cited as
formal evidence.

## Environment audit

- Train and test states are independent samples from the same paired standard-normal marginal.
- Exactly 50% of states are gap states.
- Each state logs 3 of 8 action groups; 62.5% of action-group blocks are absent.
- On gap states, the entire 32-action optimal group is absent from all logged roles.
- Every action group remains observed elsewhere, and random action IDs remove ordinal shortcuts.
- The within-group reward factor is bounded in `[0.85, 1.00]` as frozen.
- All environment invariants passed for both development seeds and both coverage modes.

## Pilot observations

The random-policy gap reward is approximately `0.1908`.

| Coverage | Method | Gap reward mean | Correct-group mass | Trap-group mass | Task collapse |
|---|---|---:|---:|---:|---:|
| structured gap | positive-only | 0.5152 | 0.2606 | 0.0630 | 0/2 |
| structured gap | local alpha 0.5 | 0.6363 | 0.5090 | 0.1723 | 0/2 |
| structured gap | local alpha 1.5 | 0.1859 | 0.1896 | 0.4725 | 2/2 |
| structured gap | uncontrolled, far 4.0 | 0.2991 | 0.3027 | 0.4769 | 0/2 |
| structured gap | near-zero, far 4.0 | 0.3438 | 0.3440 | 0.4607 | 0/2 |
| structured gap | far-cap, far 4.0 | 0.4508 | 0.4249 | 0.3862 | 0/2 |
| structured gap | budget-matched global | 0.4908 | 0.4494 | 0.3504 | 0/2 |

The pilot exhibits the intended qualitative separation: moderate local
repulsion improves the held-out optimal group, excessive local pressure moves
mass into the trap group and reaches the task-collapse rule, and far-pressure
controls partially rescue the uncontrolled condition. These are development
observations only; formal seeds 130--149 remain untouched.

## Reporting boundary

This protocol is **not** a state-distribution OOD experiment. It tests
same-distribution held-out contexts with a large, structured conditional
state-action support gap. Task-performance collapse, support/concentration
boundary events, and NaN/Inf numerical failure remain separate outcomes.
