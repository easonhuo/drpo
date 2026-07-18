# PAPER-CODE-REFERENCE-01 Hopper Rollout Handoff Delta

**Date:** 2026-07-18  
**Claim:** `PAPER-CODE-REFERENCE-01`  
**Document role:** append-only task-local continuation delta  
**Research status impact:** none

This file must be read immediately after:

```text
docs/paper_code_reference/SESSION_HANDOFF.md
```

It supersedes only the stale implementation-status statements in Sections 2, 4.4, 6, and 9 of that file. It does not replace `docs/handoff.md`, alter any scientific status, or weaken the locked single-branch rule.

## 1. Repository snapshot

- repository: `easonhuo/drpo`;
- default branch observed before this slice: `main@e99489e7435bc26e2a7e30cd8d1a3aa10f4fc67a`;
- only active development branch: `dev/paper-code-reference-01`;
- rollout implementation exact head before this delta: `fa71435fc3259f70e2961c601cab668df3b75eac`;
- persistent cumulative Draft PR: `#149`;
- overall acceptance state: `in_development`.

The development branch remains separate from `main`. PR `#149` remains Draft and must not be merged without a separate explicit user instruction.

## 2. Human-approved new Python paths

The repository owner explicitly approved exactly these paths after their responsibilities and nearest-file insufficiency were stated:

```text
paper_code/src/drpo_reference/external/hopper_rollout.py
paper_code/tests/test_hopper_rollout_differential.py
```

The durable approval record is PR `#149` conversation comment `5011057103`.

That approval does not authorize any public-runner, aggregation, regeneration, Countdown, formal-execution, branch, or merge path.

## 3. Hopper rollout slice now durably implemented

The consolidated development branch now contains:

- the frozen `gymnasium_mujoco` backend identity;
- the frozen offline dataset identity `hopper-medium-replay-v2`;
- the frozen evaluation environment identity `Hopper-v4`;
- process-isolated preflight with the registered 120-second timeout;
- the registered 2000-step preflight safety bound;
- environment/package diagnostics and worker/canonical JSON reports;
- explicit native-signal, timeout, missing-report, Python-exception, required-failure, and optional-unavailable outcomes;
- no import or fallback through `d4rl` or `mujoco_py`;
- reset compatibility for Gymnasium and legacy four-tuple test doubles;
- five-tuple terminated/truncated handling;
- action-space clipping and action-shape checks;
- observation/action dimension checks against the offline dataset contract;
- deterministic actor evaluation using the training observation normalizer;
- episode seed derivation `seed + episode`;
- raw-return mean/std and frozen D4RL-v2 normalized-score calculation;
- explicit rollout diagnostics and fail-closed required evaluation;
- a protocol wrapper that passes only frozen Hopper rollout coordinates.

The following frozen fields were added to `HopperProtocol`:

```text
rollout_backend = gymnasium_mujoco
process_isolated_preflight = true
rollout_preflight_timeout_seconds = 120
rollout_preflight_max_steps = 2000
rollout_required = true
```

## 4. Differential and repository validation

Exact-head `fa71435fc3259f70e2961c601cab668df3b75eac` passed:

- Evidence Locator Gate;
- Python compilation;
- shell syntax;
- handoff authority;
- formal execution channel;
- governance inventory;
- governance stage validation;
- full repository pytest;
- Ruff.

The focused differential coverage includes:

- normalization values and failure messages versus the authoritative legacy runner;
- four-tuple and five-tuple preflight identity on controlled fake environments;
- reset seeds, actions, clipping, steps, returns, normalized returns, and close behavior;
- required preflight failure persistence before raising;
- optional preflight failure remaining unavailable rather than being reported as success;
- SIGSEGV and timeout diagnostics in the isolated-process parent;
- deterministic actor rollout endpoint identity versus the authoritative runner;
- disabled-rollout identity;
- wrong backend/dataset/environment rejection;
- explicit proof that the open path imports only `gymnasium`;
- exact protocol-to-preflight argument binding.

These checks are engineering migration evidence only. They did not install or exercise a real MuJoCo runtime in CI, did not load the registered D4RL HDF5 artifact, did not launch a formal experiment, and did not change Hopper scientific status or authorize a method ranking.

## 5. Hopper implementation status after this delta

Durably migrated:

1. protocol and HDF5 data contract;
2. episode handling, normalization, and splits;
3. value critic, selected checkpoint, and frozen advantages;
4. squashed-Gaussian actor and six weighting modes;
5. fixed-budget actor training and terminal classification;
6. advantage-matched near/far mechanism diagnostics;
7. Positive-only preparation and exact six-branch per-seed suite;
8. Gymnasium/MuJoCo rollout adapter and process-isolated preflight.

Still missing:

1. public Hopper CLI/runner;
2. canonical critic plus per-seed suite wiring through that public runner;
3. formal/smoke/seed-subset argument validation;
4. multi-seed aggregation and root terminal audit;
5. registered-input compact regeneration;
6. paper-facing table/figure-data binding;
7. clean-checkout real-environment liveness;
8. any newly authorized registered-data fixed-budget reproduction.

## 6. Exact next implementation slice

The next slice is **Hopper public runner and multi-seed aggregation**.

It must compose existing migrated components rather than copy their implementation. At minimum it must:

1. verify dataset basename and SHA-256 before loading;
2. construct or strictly reuse one canonical critic/frozen-advantage artifact;
3. run the registered seed set or an explicitly non-formal subset;
4. invoke process-isolated rollout preflight before required evaluations;
5. pass one rollout evaluator into Positive-only and all six branch stages;
6. preserve per-seed and per-method output isolation;
7. aggregate paired seeds, mechanism diagnostics, terminal states, and rollout availability;
8. report task-performance collapse, support/variance-boundary events, and NaN/Inf numerical failure separately;
9. mark smoke/subset execution as non-formal and prohibit method-ranking claims;
10. write a root completion marker and terminal audit only after all required artifacts are verified.

This next slice must not yet perform compact paper regeneration or launch a registered formal rerun.

No exact new Python path for the public-runner slice is authorized by this delta. A continuation session must inspect current files, name each proposed path and responsibility, explain why extending the nearest existing file is insufficient, and obtain explicit human approval before creating any new `.py` file.

## 7. Remaining uncertainties

- The exact real Gymnasium/MuJoCo dependency set has not been exercised by this engineering CI environment.
- Full Hopper reproduction still requires the registered HDF5 artifact and compatible MuJoCo runtime.
- The development branch remains diverged from newer unrelated work on `main`; integration freshness must be reviewed before future writes and before any eventual merge decision.
- The acceptance matrix and source migration map still lag the current implementation state.
- Countdown final manuscript-facing protocol and result remain unresolved and blocked from migration.
