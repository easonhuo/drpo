# EXT-H-E7-PPO-W0-EXP-GRID-01 — code-first implementation scope

## Temporary status

This branch follows the user-approved code-first sequence: push the runnable pilot implementation first, then complete the authoritative handoff/registry/RunSpec process before merge or scientific launch.

- base commit: `c52f40907b44091ec5548dc6cf16d23137920ca7`;
- experiment ID: `EXT-H-E7-PPO-W0-EXP-GRID-01`;
- branch matrix: 31 unique `(w(0), c)` points × 3 datasets × 2 development seeds = 186 branches;
- horizon: 500,000 optimizer updates;
- actor mode: PPO clip only;
- held-out seeds `204--207`: forbidden;
- formal evidence: forbidden;
- merge and scientific launch: blocked until the documentation, registry, RunSpec, real-data liveness gate, and review steps are complete.

## Frozen pilot matrix

- datasets: `hopper-medium-expert-v2`, `walker2d-medium-v2`, `walker2d-medium-replay-v2`;
- development seeds: `200`, `201`;
- `w(0)`: `0`, `0.025`, `0.05`, `0.11`, `0.25`, `0.5`, `1.0`;
- `c`: `0`, `0.25`, `0.5`, `1.0`, `1.5`;
- `w(0)=0` is stored once as Positive-only;
- formula: `u=d/2`, `w(d)=w(0) exp(-c u)`;
- PPO: clip epsilon `0.2`, four updates per old-policy snapshot, no KL penalty, entropy bonus, gradient clipping, or value clipping;
- evaluation: every 50k updates, 10 episodes.

## Code-first boundary

The pushed code may be reviewed and tested immediately, but must not be described as registered, ready, merged, or scientifically launched. The subsequent governance commit must register the experiment without altering this matrix.
