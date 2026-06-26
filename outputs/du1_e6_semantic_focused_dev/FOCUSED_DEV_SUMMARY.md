# D-U1 E6 focused development extension

**Experiment:** `D-U1-E6-SEMANTIC-FOCUSED-DEV-01`
**Status:** focused development pilot complete; not a formal long-run result.
**Base GitHub commit:** `2e04f6dba6d4e87f61920bedb1c464656906bf2b`

## Integrity

- Phase 1: 55/55 runs; Phase 2: 110/110 runs; development seeds 0--4 only.
- NaN/Inf: **0/165**. Task-performance collapse: **0/165**.
- Support/temperature boundary: **78/165**; reported separately from task reward.
- 4000 steps is the registered 2x development horizon relative to the 2000-step pilot; this is not the formal held-out 2x audit.

## Blocker 1: fixed-concentration terminal state

All four fixed-concentration branches reached the registered focused terminal plateau in 5/5 seeds at 4000 steps. `alpha=0.25` and `0.5` remain beneficial relative to positive-only; `alpha=0.75` remains over-extrapolated and reverses part of the benefit. The prior failure was caused by an unsuitable absolute-zero stochastic-gradient criterion, not continuing task drift.

| alpha | reward | hidden-optimal probability | normalized extrapolation | plateau | support events |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.856666 | 0.142259 | -0.180374 | 5/5 | 0/5 |
| 0.25 | 0.881503 | 0.192782 | 0.356074 | 5/5 | 0/5 |
| 0.50 | 0.880620 | 0.197180 | 0.991665 | 5/5 | 0/5 |
| 0.75 | 0.840156 | 0.131064 | 1.569457 | 5/5 | 0/5 |

## Blocker 2: learnable-concentration pressure

The preregistered descending rule selected **local alpha 0.1**. It is the largest candidate with 0/5 support events, 0/5 numerical failures, 5/5 reward wins, 5/5 hidden-probability wins, and 5/5 focused terminal plateaus. `alpha=0.2` entered the boundary transition with 3/5 support events and only 1/5 plateaus.

At `local alpha=0.1`, far pressure shows a sharp transition. `lambda=0.01` is safe in 5/5 seeds. At `lambda=0.02`, uncontrolled, far-cap, and budget-matched global all hit support boundary in 5/5 seeds, while far-only remains safe. At `lambda=0.05`, far-only also hits the boundary in 5/5 seeds. Thus far pressure can independently trigger support contraction, but the registered ratio-1 far cap and matched-global control do not rescue the transition; no method superiority is claimed.

## Gate decision

The two development blockers are resolved sufficiently to propose a formal freeze, but automatic activation remains prohibited. `formal_freeze_recommendation.json` proposes held-out seeds 10--29, an 8000-step formal horizon, fixed alpha {0,0.25,0.5,0.75}, learnable local alpha 0.1, and far-pressure stress lambda 0.05. The formal runner and canonical activation must wait for explicit user approval.
