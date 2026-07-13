# EXT-H-E7-PPO-W0-EXP-GRID-01

## Status

- scientific class: external-validity development screening pilot;
- implementation: code-first dev branch;
- registration: pending authoritative handoff/registry delta;
- result status: `not_run`;
- RunSpec: blocked template;
- held-out seeds `204--207`: untouched and forbidden.

## Claim

Determine which regions of the direct negative-weight parameterization

\[
u=d/2,\qquad w(d)=w(0)\exp(-cu)
\]

remain viable under the canonical E7 PPO actor update across Hopper-medium-expert, Walker2d-medium, and Walker2d-medium-replay. This is a 500k screening experiment; it does not establish convergence, a steady-state ranking, or universal PPO superiority.

## Frozen matrix

- datasets: `hopper-medium-expert-v2`, `walker2d-medium-v2`, `walker2d-medium-replay-v2`;
- development seeds: `200`, `201`;
- `w(0)`: `0`, `0.025`, `0.05`, `0.11`, `0.25`, `0.5`, `1.0`;
- `c`: `0`, `0.25`, `0.5`, `1.0`, `1.5`;
- `w(0)=0` appears once as Positive-only;
- unique parameter points: `31`;
- actor mode: PPO clip only;
- horizon: `500,000` optimizer updates;
- total branches: `31 × 3 × 2 = 186`.

The historical scale-1 point is represented directly by `w(0)=0.11`. The public config, branch identity, manifest, and finalized diagnostics must not expose `negative_scale`, `canonical_alpha`, or `effective_alpha` as experiment coordinates.

## Preserved components

- canonical D4RL source fingerprint contract;
- actor and critic architecture;
- critic target and expectile update;
- advantage definition and full-batch normalization;
- dataset files and hashes;
- batch size `256`;
- learning rate `0.0003`;
- formal evaluation interval `50,000` and `10` episodes;
- PPO clip epsilon `0.2` and four updates per old-policy snapshot;
- no KL penalty, target-KL stop, entropy bonus, actor gradient clipping, or value clipping.

## Automatic resource selection

Before materializing the 186-branch run identity, the launcher:

1. reads process-visible CPUs, current load, host/cgroup memory, and memory availability;
2. runs one isolated representative PPO branch to measure process-tree peak RSS;
3. computes a CPU/RAM safety ceiling;
4. benchmarks a bounded concurrency grid near 50%, the verified fallback, 75%, and the ceiling;
5. selects the smallest successful concurrency reaching 97% of measured peak aggregate updates/s;
6. writes `RUNTIME_SELECTION.json` and freezes the selected worker count into the run identity.

This runtime selection changes only active subprocess count. It is not a scientific variable and is not a proof of a continuous global throughput optimum.

## Reporting

Report branch-wise BEST, 500k FINAL, BEST-to-FINAL drop, the 400k--500k mean and slope, two-seed spread, PPO ratio/clipping diagnostics, and numerical failures. Keep the following separate:

- task-performance degradation;
- support or variance-boundary events;
- NaN/Inf numerical failures.

The current pilot does not instrument a registered support/variance boundary threshold; that field must be reported as unavailable rather than inferred from task performance.

## Execution gate

Code review and CI may proceed immediately. Scientific launch remains blocked until:

- the experiment is registered through the authoritative handoff/registry mechanism;
- the RunSpec is pinned to an exact reviewed commit and promoted from template to ready;
- targeted tests and repository governance checks pass;
- a short real-data liveness run confirms the direct-w(0) bootstrap and resource selector on the server.
