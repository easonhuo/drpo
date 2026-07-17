# EXT-H-E7-SQEXP-GAE-01

## Status and scope

This is the first complete code implementation for a development pilot. No
scientific run has been started. Hopper and Walker are external-validity tasks;
this pilot does not replace C-U1 or D-U1 controlled identification.

The implementation compares frozen one-step TD and behavior-trajectory GAE
advantages under the same frozen critic. It preserves the frozen matrix:

- datasets: `hopper-medium-expert-v2`, `walker2d-medium-v2`, and
  `walker2d-medium-replay-v2`;
- development seeds: `200--203`; held-out seeds `204--207` are rejected;
- estimators: one-step TD and GAE with `gamma=0.99`, `lambda=0.95`;
- actor modes: canonical A2C and PPO clipping with epsilon `0.2` and fixed
  old-policy cadence `K=4`;
- controls: Positive-only and squared-remoteness EXP at `c=64,128,256`;
- one frozen critic for every dataset/seed cell, trained for 100,000 updates;
- every actor branch receives 1,000,000 updates;
- exact expansion: 12 critic/preparation jobs and 192 actor branches.

A fixed 1M horizon is not convergence or steady-state evidence.

## Boundary and numerical contract

For each ordered behavior trajectory:

1. a true terminal uses no value bootstrap and stops GAE carry;
2. a timeout bootstraps from the stored next observation and stops carry;
3. a final stored nonterminal row also bootstraps and stops carry;
4. terminal and timeout flags may not overlap;
5. `lambda=0` must reproduce one-step TD exactly;
6. advantages are neither normalized nor clipped and are exposed to actor code as
   `float32`.

The preparation worker computes in `float64`, compares the main reverse-scan GAE
against an independent episode-slice implementation, and reports storage
quantization separately from implementation disagreement.

## Shared-critic contract

Each dataset/seed critic identity binds the dataset SHA, source RunSpec SHA,
seed, critic budget, gamma, expectile, batch size, learning rate, and network
profile. A preparation manifest binds both advantage arrays to the frozen critic
checkpoint. Actor workers verify all hashes, never load the critic into the actor
optimizer, and re-hash the critic checkpoint after training.

## Execution

Plan without loading data:

```bash
python scripts/run_e7_sqexp_gae.py plan \
  --config configs/e7_sqexp_gae_v1.yaml \
  --work-dir /tmp/e7_sqexp_gae_plan
```

Run or resume against the existing exact-dataset RunSpec:

```bash
python scripts/run_e7_sqexp_gae.py run \
  --config configs/e7_sqexp_gae_v1.yaml \
  --run-spec /root/d4rl2/configs/e7_canonical_9task_full_grid_run_spec_v1.json \
  --work-dir outputs/e7/sqexp_gae_run_01 \
  --critic-workers 12 \
  --actor-workers 64 \
  --resume
```

The run is two-stage and resumable: all 12 critic/preparation workers complete
before actor workers launch. Existing output is accepted only under the same run
identity.

## Terminal reporting

Every branch reports task-performance collapse, support/variance boundary events,
and NaN/Inf numerical failure separately. Because this task does not freeze a new
task-collapse threshold, task collapse is explicitly `not_adjudicated` rather
than inferred. Aggregation uses the 800k--1M late window and paired GAE-minus-TD
rows; failed or non-finite cells are excluded without imputation.

No universal estimator or actor-mode ranking, convergence, steady state, causal
identification, OOD generalization, or paper-facing formal result may be claimed
from implementation checks or a future development pilot.
