# PR notes

## Implemented

- direct `w(0)` × EXP coefficient grid with 31 unique cells;
- 3 tasks × 2 development seeds × 31 cells = 186 PPO branches;
- 500k screening horizon;
- public configs and finalized diagnostics use `w(0)` and `c`, not experiment-facing scale/alpha;
- exact historical equivalence test for `w(0)=0.11` versus old alpha 0.11 × scale 1;
- CPU/load/cgroup-memory discovery, peak-RSS safety ceiling, bounded throughput grid, and 97%-retained-peak worker selection;
- one-click plan/run/resume path;
- finite-step aggregation and terminal audit with separate task, support/variance, and NaN/Inf reporting.

## Still blocked

- authoritative handoff/registry registration;
- real-data server liveness;
- ready RunSpec promotion;
- full CI/governance acceptance;
- merge approval and scientific launch.
