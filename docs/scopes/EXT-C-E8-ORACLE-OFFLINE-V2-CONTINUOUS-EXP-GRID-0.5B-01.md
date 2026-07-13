# EXT-C-E8-ORACLE-OFFLINE-V2-CONTINUOUS-EXP-GRID-0.5B-01 scope

## Authorized registered-pilot scope

The server-runnable code-first implementation is now authoritatively registered as a `pilot / not_run` experiment on its dedicated dev branch. Repository-wide review gates and real-server liveness remain subsequent requirements.

Authorized scientific coordinates:

- the frozen E8 V2 oracle-offline bank and base Qwen2.5-0.5B fresh-LoRA initialization;
- all unique negatives per prompt, without extreme selection;
- `u = current_sequence_surprisal / 2`;
- `w = alpha * exp(-c * u^2)`;
- `alpha = [0, 0.025, 0.05, 0.11, 0.25, 0.5, 1.0]`;
- `c = [0, 0.25, 0.5, 1.0, 1.5]`;
- two fresh development seed offsets, `3000` and `4000`;
- 31 unique points, 62 cells;
- 1200 fixed steps, no early stop;
- validation-only tuning evaluation.

Authorized engineering work:

- one-click plan/smoke/run entrypoint;
- identity-checked cell resume;
- GPU/host-memory autotune that changes only active GPU slots;
- actual representative liveness before the full grid;
- compact metrics, progress, and terminal-audit files;
- focused deterministic tests.

## Explicit exclusions

- no direct handoff edit outside the accepted schema-v3 delta materialization;
- no reviewed READY RunSpec yet;
- no merge to `main` without review and explicit user approval;
- no test-split access during tuning;
- no method ranking, convergence, steady-state, or OOD claim;
- no near/far selection, binary mixture, hidden scale, budget matching, weight renormalization, dynamic alpha, SBRC, Hybrid, entropy, SFT warmstart, on-policy, or replay changes;
- no modification or deletion of predecessor experiment history.

## Acceptance boundary for server trial

A dev-branch server trial may start only after the representative liveness gate succeeds. A successful trial remains dev pilot evidence bound to the exact dev HEAD and does not substitute for CI, governance, terminal audit, fresh-seed confirmation, or reviewer gates. Handoff/registry registration is complete, but no run result is registered.
