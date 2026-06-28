# EXT-H-E7-Q2 Hopper one-command runner

This entrypoint implements the preregistered Hopper external-mechanism experiment. It does not replace C-U1, and it is not a standard D4RL method-ranking table.

## What changed after the first engineering pilot

The first one-seed/100-step pilot correctly remained `pilot`, but exposed three implementation issues that are now fixed:

1. the critic used to be retrained independently inside every actor seed;
2. rollout failures were collapsed into `rollout_unavailable=1` without a traceback;
3. a pilot could show `independent_validation_gate_all_seeds=true` even though paired-seed evidence and formal review were absent;
4. the legacy `gym.make("hopper-medium-replay-v2")` path imported `mujoco_py` and could terminate the entire process with SIGSEGV.

Protocol v4.1 trains or verifies **one canonical critic artifact per run**, freezes one advantage array, shares both across every actor seed, uses a process-isolated Gymnasium `Hopper-v4` preflight before critic training, and separates engineering completion, mechanism subchecks, paired evidence, formal prerequisites, and post-run scientific review.

## Scientific flow

The one-command runner automatically:

1. verifies the exact `hopper_medium_replay-v2.hdf5` basename and SHA-256;
2. launches an isolated subprocess that constructs Gymnasium `Hopper-v4`, resets it, executes a real step, runs one random episode, and checks manual D4RL-v2 reference normalization;
3. writes the full rollout environment/version report, subprocess exit code, native signal, stdout/stderr, and traceback before any critic training if preflight fails;
4. creates one episode-level critic train/validation/test split using the registered canonical critic seed;
5. trains one return-value critic to the registered optimization-terminal candidate, completes the 2× continuation, and reconfirms the terminal criteria at the continuation endpoint;
6. selects the **terminal extension checkpoint** for the experiment; the best transient validation checkpoint is retained only as a diagnostic;
7. freezes the critic, materializes one immutable TD-residual advantage array, hashes both, and shares them across all actor seeds;
8. trains each actor seed's Positive-only initialization to its own terminal audit;
9. matches near/far negative transitions in absolute frozen advantage;
10. reports Gaussian base-coordinate components and full-parameter gradient ratios;
11. branches `positive_only`, `signed`, `near_zero`, `far_zero`, `far_cap`, and `budget_matched_global` from the same actor checkpoint and minibatch stream within each seed;
12. performs deterministic environment rollout evaluation at registered checkpoints;
13. writes per-seed raw curves, terminal audits, root-level summaries, and a canonical critic reference.

The scientific runner does **not** create ZIP or TAR files. The canonical hardened guard owns supervision, failure recovery, checksums, size policy, source snapshots, and the uploadable result artifact.

## Critic interpretation

The canonical critic is required to be optimization-converged under the registered stopping rule before a formal actor run can begin. This removes dynamic critic drift and actor--critic feedback from the actor mechanism stage:

- critic parameters never update during actor training;
- advantage labels are computed once and never recomputed;
- minibatches never renormalize advantage;
- all actor seeds and methods use the same critic checkpoint, episode split, normalizers, and frozen advantage file.

This does **not** claim that the learned critic is a ground-truth value function. Held-out MSE, R², and Pearson remain reported so residual statistical approximation error is visible.

## Dataset identity

```text
basename: hopper_medium_replay-v2.hdf5
sha256: e121c5f7c9857a307baa9edc6a2c3b48e85fedb9ac316ecddd0f48ca7ef4e39b
```

A mismatched file fails before environment interaction or training.

## Install

```bash
python3 -m pip install -e .
python3 -m pip install 'gymnasium[mujoco]'
```

The rollout path does not import `d4rl` or `mujoco_py` and has no automatic legacy fallback. The offline HDF5 file remains D4RL Hopper medium-replay-v2 data; only policy interaction uses the local Gymnasium `Hopper-v4` compatibility environment.

Normalized return is computed manually with the frozen D4RL-v2 Hopper medium-replay references:

```text
min = -20.272305
max = 3234.3
score = (raw_return - min) / (max - min) * 100
```

Reports must describe this as a Gymnasium `Hopper-v4` compatibility evaluation with D4RL-v2 reference normalization, not as an exact legacy `mujoco-py` leaderboard reproduction.

CPU is supported and is acceptable for this small-network experiment. Use the same device policy across paired methods and retain the device in the run manifest.

## Pilot

Use a new persistent output directory:

```bash
python3 scripts/run_e7_hopper_q2.py \
  --run-class pilot \
  --dataset-path /absolute/path/hopper_medium_replay-v2.hdf5 \
  --work-dir /absolute/persistent/path/e7_q2_pilot \
  --device cpu
```

The pilot uses seed `42`, at most 10,000 transitions, and short horizons. It is implementation evidence only. The process-isolated rollout preflight is mandatory for pilot as well as formal execution; a Python exception, timeout, or native signal such as SIGSEGV stops the run before critic training and is packaged by the hardened guard.


## Dataset and rollout version boundary

| Component | Frozen choice |
|---|---|
| Offline training data | `hopper-medium-replay-v2` HDF5 |
| Local interaction environment | Gymnasium `Hopper-v4` |
| MuJoCo binding | modern `mujoco` package |
| Normalization | D4RL-v2 min/max reference constants |
| Legacy D4RL/mujoco-py fallback | forbidden |

This split is intentional: dataset version and simulator API version are different objects. `Hopper-v4` is not a v4 dataset.

## Reuse an exact canonical critic artifact

A completed canonical artifact can be reused only when its dataset, loaded transition count, run class, config hash, runner version, dimensions, and canonical seed exactly match:

```bash
python3 scripts/run_e7_hopper_q2.py \
  --run-class pilot \
  --dataset-path /absolute/path/hopper_medium_replay-v2.hdf5 \
  --work-dir /absolute/persistent/path/e7_q2_pilot_reuse \
  --critic-artifact /absolute/persistent/path/e7_q2_pilot/canonical_critic \
  --device cpu
```

Every artifact file is rehashed before reuse. A nonterminal critic artifact is rejected in formal mode.

## Formal run

`EXT-H-E7-Q2` is now the next registered formal route item: its scientific gate is `ready` and its operational activation state is `active`. Its scientific status remains `not_run`; this route release is permission to launch the frozen protocol, not evidence that Hopper results already exist.

Formal execution uses:

```bash
python3 scripts/run_e7_hopper_q2.py \
  --run-class formal \
  --dataset-path /absolute/path/hopper_medium_replay-v2.hdf5 \
  --work-dir /absolute/persistent/path/e7_q2_formal \
  --device cpu
```

Formal seeds remain `100-109`; the canonical critic seed is `100` and is used once for the shared critic artifact, not retrained ten times.

## Key output structure

```text
<work-dir>/
  ROLLOUT_PREFLIGHT.json
  CANONICAL_CRITIC_REFERENCE.json
  canonical_critic/
    canonical_critic_manifest.json
    episode_split.npz
    training/
      terminal_critic.pt
      best_validation_critic.pt
      critic_metrics.csv
      critic_terminal_audit.json
    frozen_advantage/
      frozen_advantages.npz
      advantage_manifest.json
  seeds/seed_<id>/
    positive_only_initialization/
    probes/
    methods/<method>/
    rollouts/<stage>/
  per_seed_summary.csv
  aggregate_summary.json
  terminal_audit.json
  RUN_COMPLETE.json
```

## Audit semantics

Task performance is three-state:

- `available`: normalized return was observed; collapse is `true` or `false`;
- `unavailable`: rollout was attempted but failed; collapse is `null`;
- `not_evaluated` or `disabled`: no task conclusion is permitted; collapse is `null`.

Root audit fields are intentionally separated:

- `engineering_pipeline_complete`;
- `mechanism_subchecks_passed_for_completed_seeds`;
- `paired_seed_evidence_complete`;
- `formal_evidence_prerequisites_complete`;
- `formal_scientific_gate_passed`.

A pilot can complete engineering and mechanism subchecks, but `formal_scientific_gate_passed` is always false. Even a complete formal raw run requires post-run scientific review before any claim upgrade. The historical `independent_validation_gate_all_seeds` field is retained only as a compatibility alias and is false for pilot runs.

## Reporting boundary

Task-performance collapse, support/variance-boundary events, and NaN/Inf numerical failure are always reported separately. Passing only the analytic/autograd identity is an implementation check, not independent external validation. Hopper provides external validity and cannot replace C-U1 controlled causal identification.
