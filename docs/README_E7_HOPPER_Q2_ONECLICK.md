# EXT-H-E7-Q2 Hopper one-command runner

This entrypoint implements the preregistered Hopper external-mechanism experiment. It does not replace C-U1, and it is not a standard D4RL method-ranking table.

## Scientific flow

For each paired seed, the runner automatically:

1. verifies the exact `hopper_medium_replay-v2.hdf5` basename and SHA-256;
2. splits complete episodes into critic train/validation/test partitions;
3. trains a return-value critic and performs the registered optimization-terminal audit;
4. freezes the critic and materializes one immutable TD-residual advantage array;
5. trains Positive-only to an audited terminal candidate and completes the 2× continuation gate;
6. matches near/far negative transitions in absolute frozen advantage;
7. reports Gaussian base-coordinate components:
   - standardized radius `r=||z||`;
   - mean-score norm;
   - raw log-scale-score norm;
   - corrected `Q_xi=||z||²`;
   - joint output-score norm;
   - full-parameter gradient norm;
   - analytic/output-autograd consistency;
8. branches `positive_only`, `signed`, `near_zero`, `far_zero`, `far_cap`, and `budget_matched_global` from the same checkpoint and minibatch order;
9. reports normalized-return trajectories when the registered D4RL environment is available;
10. writes per-seed summaries, raw curves, terminal-state audits, checkpoint hashes, and root-level aggregate/audit files.

The scientific runner does **not** create ZIP or TAR files. The canonical hardened guard owns supervision, failure recovery, checksums, size policy, source snapshots, and the uploadable result artifact.

## Dataset identity

The primary dataset is fixed to:

```text
basename: hopper_medium_replay-v2.hdf5
sha256: e121c5f7c9857a307baa9edc6a2c3b48e85fedb9ac316ecddd0f48ca7ef4e39b
```

A mismatched file fails before training.

## Install

Install the repository dependencies. Formal normalized-return evaluation additionally requires a working legacy D4RL Hopper environment registered as `hopper-medium-replay-v2`.

```bash
python3 -m pip install -e .
# Install the project-approved MuJoCo/Gym/D4RL stack on the training machine.
```

## Pilot

Run after the update has been applied and committed. Use a new persistent output directory:

```bash
python3 scripts/run_e7_hopper_q2.py \
  --run-class pilot \
  --dataset-path /absolute/path/hopper_medium_replay-v2.hdf5 \
  --work-dir /absolute/persistent/path/e7_q2_pilot \
  --device cuda
```

The pilot uses seed `42`, at most 10,000 transitions, and short horizons. It is implementation evidence only.

## Formal run

Formal mode requires a clean committed worktree. The wrapper binds the exact current commit and launches the canonical hardened guard:

```bash
python3 scripts/run_e7_hopper_q2.py \
  --run-class formal \
  --dataset-path /absolute/path/hopper_medium_replay-v2.hdf5 \
  --work-dir /absolute/persistent/path/e7_q2_formal \
  --device cuda
```

Formal seeds are `100-109`. The default artifact is written beside the work directory as:

```text
<work-dir>_EXT-H-E7-Q2_formal.zip
```

Upload that ZIP unchanged for scientific audit and repository closure. Large critic/actor checkpoints and the dataset stay on persistent local storage and are indexed by path, size, role, and SHA-256 rather than copied into the main artifact.

## Output contract

The guarded run requires these root files:

```text
RUN_COMPLETE.json
terminal_audit.json
aggregate_summary.json
per_seed_summary.csv
run_manifest.json                 # owned by hardened guard
scientific_run_manifest.json      # owned by scientific runner
logs/supervised_run.log           # owned by hardened guard
```

Per-seed directories retain critic metrics, frozen-advantage manifests, matched component tables, distance-bin tables, method curves, terminal audits, and checkpoint manifests.

## Reporting boundary

Task-performance collapse, support/variance-boundary events, and NaN/Inf numerical failure are always reported separately. Passing only the analytic/autograd identity is an implementation check, not independent external validation. The formal claim also requires natural far-field samples, measurable log-scale/full-parameter contribution, paired-seed targeted-control mitigation, and terminal-state audit.
