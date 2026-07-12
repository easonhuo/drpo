# EXT-H-E7-PPO-STABILITY-01 execution protocol

## Status

- scientific class: external-validity development pilot;
- implementation: merged PPO actor implementation plus this execution follow-up;
- result status: `not_run`;
- full pilot state: blocked until the real-data smoke gate passes;
- held-out seeds `204--207`: untouched and forbidden in this pilot.

This document supplements `docs/experiments/EXT-H-E7-PPO-STABILITY-01.md`. It is not a second research master; `docs/handoff.md` remains authoritative.

## Ordered execution

The E7 lane must execute the following single-use RunSpecs in order:

1. `E7_PPO_STABILITY_SMOKE_20260712_01`
2. reviewer inspection of the smoke gate;
3. `E7_PPO_STABILITY_PILOT_20260712_01`

The full pilot entrypoint fails closed when the smoke gate is absent, failed, or bound to different protected implementation hashes. A local executor must not recreate the smoke command, alter the matrix, or bypass the gate.

## Real-data smoke gate

The smoke run is deliberately excluded from scientific aggregation. It executes one matched pair:

- dataset: `walker2d-medium-v2`;
- development seed: `200`;
- negative control: EXP scale `1.0`, coefficient `1.5`;
- actor updates: historical A2C-style and PPO clipped;
- optimizer steps: `20,000`;
- PPO diagnostics interval: `1,000`.

Required gate checks:

- both branches finish at 20k with finite task scores;
- PPO block position 1 has old/new ratio approximately one;
- positions 2--4 show nonzero ratio movement;
- ratio statistics and positive/negative objective clip fractions are finite and in `[0,1]`;
- old policy refresh count is exactly consistent with four updates per snapshot;
- actor gradient norm and parameter-update norm are finite;
- PPO diagnostics JSONL has the expected number of records;
- the smoke gate records SHA-256 fingerprints for every protected implementation file.

The smoke may report zero or low clipping. That is not automatically a failure when the ratio moves but stays within the clip interval. Conversely, a high clip fraction only proves that the proximal constraint is active; it does not establish scientific benefit.

Outputs:

```text
outputs/e7/ppo_stability_smoke_001/
  SMOKE_GATE.json
  RUN_SUMMARY.json
  branches/*
```

## PPO-specific runtime capacity selection

After the smoke gate passes, the full pilot performs a short representative PPO branch probe before planning the 96 branches. The resource fingerprint includes compute-relevant fields:

- canonical source and PPO implementation hashes;
- batch size and optimizer learning rate;
- old-policy refresh cadence;
- diagnostics cadence;
- evaluation load;
- BLAS/OpenMP thread environment;
- representative workload family.

The fingerprint intentionally ignores scientific sweep coordinates that do not change the branch compute graph:

- seed values;
- EXP coefficients;
- Positive-only versus EXP labels;
- one-million-step horizon;
- method and branch counts, except that total branch count caps selected workers.

The selector writes `RUNTIME_SELECTION.json` and chooses the active subprocess count from:

- currently available logical CPUs;
- current one-minute load;
- measured representative-branch peak RSS;
- host-memory headroom;
- configured safety and growth limits;
- the 96-branch task ceiling.

This V1 is a conservative capacity guard. It does **not** claim to locate the globally throughput-optimal knee. The selected worker count is runtime provenance, not scientific identity. Once `RUN_IDENTITY.json` exists, changing workers for the same work directory is forbidden.

## Frozen full pilot

The full matrix remains exactly:

```text
3 datasets x 4 development seeds x 4 negative-control settings x 2 actor modes
= 96 branches
```

Datasets:

- `hopper-medium-expert-v2`
- `walker2d-medium-v2`
- `walker2d-medium-replay-v2`

Development seeds:

- `200`
- `201`
- `202`
- `203`

Controls:

- Positive-only;
- EXP `c=0.5`;
- EXP `c=1.0`;
- EXP `c=1.5`.

Actor modes:

- historical A2C-style surrogate;
- PPO clipped surrogate with `epsilon=0.2` and four optimizer updates per old-policy snapshot.

All branches use 1M optimizer steps. No 2M continuation, KL penalty, target-KL stop, entropy bonus, actor gradient clipping, value clipping, learning-rate change, network change, critic change, advantage change, or EXP-normalization change is authorized.

## Terminal reporting

The aggregate must report:

- branch-wise BEST;
- 1M FINAL;
- BEST-to-FINAL drop;
- FINAL across-seed standard deviation;
- paired PPO-minus-A2C differences;
- paired EXP-minus-Positive-only differences;
- ratio and sign-aware clipping diagnostics.

The following remain separate:

- task-performance degradation;
- support or variance-boundary events;
- NaN/Inf numerical failure.

A fixed 1M endpoint is not convergence. This pilot cannot establish a steady-state ranking or universal PPO superiority.
