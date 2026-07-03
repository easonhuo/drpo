# E7 canonical D4RL lineage recovery and small-negative-weight pilot

## Scope

This update does not convert the current `e7_bench.py` frozen-advantage Hopper
runner into IQL, DRPO, or SNA2C. It leaves E7-Q2 unchanged and adds a separate,
fail-closed adapter that executes the user's original D4RL source tree.

The adapter deliberately imports the old `Actor`, `Critic`, optimizers, dataset
loop, rollout evaluator, initialization, and trainer from that source tree. It
changes only the negative-advantage multiplier in the historical signed TD/V
agent update. A Python-tree SHA-256, agent-module SHA-256, and trainer SHA-256
must match before any branch starts.

**Experiment:** `EXT-H-E7-BENCH-01`, lineage-recovery weight-sweep pilot.

**Claim answered by this pilot:** after restoring the old network and training
skeleton, determine whether the previous Hopper collapse was caused by a
negative coefficient that was still too large, and locate a nonzero transition
region between Positive-only and the canonical signed update.

This pilot cannot be reported as the formal nine-task D4RL table. The formal
protocol remains blocked until its exact versions, seeds, base algorithms,
optimizer settings, budgets, and authority registration are frozen.

## What is preserved

For a contracted `signed_td_v_v1` agent, the adapter preserves:

- the original actor and value network objects;
- their original initialization and constructor hyperparameters;
- the original Adam optimizers and learning rates;
- the original TD target and expectile value regression;
- actor-before-critic update order;
- the original trainer, replay/data sampling, evaluation, checkpointing, and
  rollout code.

The only replacement is:

```text
negative advantage
    -> canonical alpha (expected 0.11)
    -> branch negative-scale
    -> optional detached distance taper
```

Positive advantages always retain factor `1`. Every branch uses the full batch
mean. Positive-only therefore zeroes negative terms without changing the
positive-gradient denominator.

## Branches

`configs/e7_canonical_weight_grid_v1.json` creates 31 injected branches for
each `(dataset, seed)`:

- Positive-only anchor: scale `0`;
- canonical signed anchor: scale `1` at canonical alpha `0.11`;
- Global: `1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3`;
- Reciprocal-linear, reciprocal-quadratic, and exponential: `1e-3, 3e-3,
  1e-2, 3e-2, 0.1, 0.3, 1.0`.

The effective negative coefficient is

```text
0.11 × negative_scale × taper(distance)
```

Quartic is not included. Optional unchanged IQL/DRPO/SNA2C baselines can be
added as `passthrough_variants` in the run spec; they execute the exact old
trainer without class injection.

## Required canonical-source contract

First fingerprint the actual old source tree. Example names below must be
replaced with the real paths and class from that tree:

```bash
python scripts/run_e7_canonical_sweep.py fingerprint \
  --canonical-root /absolute/path/to/original_d4rl_source \
  --agents-relpath agents.py \
  --trainer-relpath train_d4rl.py \
  --module-name agents \
  --target-class SNA2C_IQLV_DistAgent \
  --expected-canonical-alpha 0.11 \
  --output /absolute/path/to/canonical_contract.json
```

Review and retain the generated contract with the experiment artifacts. Any
source edit after fingerprinting causes a fail-closed SHA mismatch.

The target class must satisfy the historical `signed_td_v_v1` interface:

```text
actor(s) -> (mu, log_std)
critic(s) -> V(s)
a_opt, c_opt, gamma, tau, alpha
update(s, a, r, ns, done, ep_ret=None)
```

The package does not guess an actor width, activation, critic implementation,
or trainer CLI when the original source is absent.

## Run-spec format

Create a JSON file that maps the generic launcher to the unchanged trainer:

```json
{
  "run_kind": "pilot",
  "datasets": [
    {
      "id": "hopper-medium-expert-v2",
      "path": "/absolute/path/hopper-medium-expert-v2.hdf5",
      "sha256": "9d51ad87f8c905be3880d84c6140bcdb7fbf39a19e046a237f238ba34fec9e26"
    }
  ],
  "seeds": [200, 201, 202, 203],
  "trainer_argv_template": [
    "--dataset", "{dataset_path}",
    "--seed", "{seed}",
    "--agent", "{agent_selector}",
    "--output_dir", "{output_dir}"
  ],
  "injected_template_values": {
    "agent_selector": "SNA2C_IQLV_DistAgent"
  },
  "passthrough_variants": [
    {
      "id": "iql",
      "template_values": {"agent_selector": "IQL"}
    },
    {
      "id": "drpo",
      "template_values": {"agent_selector": "DRPO"}
    },
    {
      "id": "sna2c",
      "template_values": {"agent_selector": "SNA2C_IQLVAgent"}
    }
  ],
  "environment": {
    "OMP_NUM_THREADS": "7",
    "MKL_NUM_THREADS": "7"
  }
}
```

The selector strings are examples. They must exactly match the original
trainer. Omit `passthrough_variants` until those selectors are confirmed.

Validate without training:

```bash
python scripts/run_e7_canonical_sweep.py plan \
  --contract /absolute/path/to/canonical_contract.json \
  --run-spec /absolute/path/to/run_spec.json \
  --grid configs/e7_canonical_weight_grid_v1.json \
  --work-dir /absolute/path/to/e7_canonical_pilot \
  --max-workers 40
```

Run and resume:

```bash
python scripts/run_e7_canonical_sweep.py run \
  --contract /absolute/path/to/canonical_contract.json \
  --run-spec /absolute/path/to/run_spec.json \
  --grid configs/e7_canonical_weight_grid_v1.json \
  --work-dir /absolute/path/to/e7_canonical_pilot \
  --max-workers 40

python scripts/run_e7_canonical_sweep.py run \
  --contract /absolute/path/to/canonical_contract.json \
  --run-spec /absolute/path/to/run_spec.json \
  --grid configs/e7_canonical_weight_grid_v1.json \
  --work-dir /absolute/path/to/e7_canonical_pilot \
  --max-workers 40 --resume
```

Each branch has its own identity, command, logs, source fingerprints, output
directory, completion marker, and failure marker.

## Mandatory checks before interpreting results

1. `canonical_signed` must reproduce the original signed agent for the same
   seed and batch.
2. Positive-only and every controlled method must share the same initial
   checkpoint and full-batch positive-gradient normalization.
3. Task-performance collapse, support/variance boundary events, and NaN/Inf
   numerical failure must remain separate outputs.
4. Fixed horizon is not convergence. Any claimed ranking requires terminal
   audit and horizon extension.
5. This grid is used to locate a viable scale interval. Choosing a winning
   scale on these same pilot cells and then calling it a formal D4RL result is
   forbidden.
