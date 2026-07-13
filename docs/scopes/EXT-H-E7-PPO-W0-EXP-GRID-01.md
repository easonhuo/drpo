# EXT-H-E7-PPO-W0-EXP-GRID-01 — implementation scope

## Objective

Implement the full 186-branch, 500k PPO screening pilot for the direct parameterization `w(d)=w(0) exp(-c d/2)` and integrate automatic CPU/RAM/throughput selection.

## Authorized scientific matrix

- datasets: hopper-medium-expert, walker2d-medium, walker2d-medium-replay;
- development seeds: 200, 201;
- held-out seeds 204--207 forbidden;
- `w(0)`: 0, 0.025, 0.05, 0.11, 0.25, 0.5, 1.0;
- `c`: 0, 0.25, 0.5, 1.0, 1.5;
- Positive-only deduplicated to one point;
- PPO clip only;
- 500,000 updates;
- 50,000-step evaluation interval and 10 episodes;
- 186 branches.

## Authorized runtime changes

- inspect CPU affinity/count, current load, host and cgroup memory;
- measure representative process-tree peak RSS;
- benchmark bounded concurrency candidates;
- select only active subprocess count;
- write runtime provenance and freeze selected workers into run identity.

## Excluded

- changing network, critic, advantage, optimizer, learning rate, batch, datasets, evaluation, PPO clip epsilon, or old-policy cadence;
- A2C comparisons;
- 1M continuation or four-seed validation;
- held-out confirmation;
- automatic candidate promotion;
- scientific launch from a dev branch;
- convergence or steady-state claims;
- universal PPO superiority;
- treating task degradation, support/variance boundary, and NaN/Inf as one event.

## Acceptance before launch

- targeted unit tests and Python/shell syntax pass;
- CI and governance checks pass;
- exact reviewed commit is pinned in a ready RunSpec;
- authoritative handoff/registry registration is materialized;
- short real-data server liveness confirms the bootstrap and resource selector;
- reviewer explicitly accepts the launch commit.
