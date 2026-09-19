# DRPO paper reference code

This directory contains the compact paper-facing DRPO reference implementation.
It is intentionally separate from the repository's historical experiment
drivers and research-governance machinery.

C-U1 and D-U1 both use independent train and held-out contexts drawn from the
same distribution. Their result is **same-distribution held-out-context
generalization**, not OOD generalization.

## Reviewer-facing scope

This package is a compact reference implementation of the paper algorithms.
It keeps the numerical training logic and the parameters needed to reproduce the
experiments, while avoiding repository-internal provenance, checkpoint,
eligibility, and workflow machinery.

Scores may vary across seeds, hardware, MuJoCo/Gymnasium versions, and numerical
libraries. The intended invariant is the algorithmic update sequence and
experiment definition.

## Install and test

Core training and tests:

```bash
cd paper_code
python -m pip install -e '.[test]'
python -m pytest
```

Install the optional Gymnasium/MuJoCo rollout dependencies when real environment
evaluation is required:

```bash
python -m pip install -e '.[test,rollout]'
```

Install the optional Countdown Transformer runtime without changing the base
CPU-oriented package:

```bash
python -m pip install -e '.[test,countdown]'
```

Use `countdown-4bit` instead of `countdown` only for an explicitly configured
bitsandbytes/CUDA run.

## C-U1

C-U1 uses same-distribution train and held-out contexts. The four public stages
correspond directly to the paper experiments:

- `source`: equal-advantage near/far gradient amplification;
- `causal`: near/far interventions and drift/collapse transmission;
- `phase`: negative-strength scans and far-pressure controls;
- `taper`: remoteness-aware taper comparison.

```bash
python -m drpo_reference cu1 --stage source --output outputs/cu1_source
python -m drpo_reference cu1 --stage causal --output outputs/cu1_causal
python -m drpo_reference cu1 --stage phase --output outputs/cu1_phase
python -m drpo_reference cu1 --stage taper --output outputs/cu1_taper
```

Each command runs the original numerical experiment and writes one compact JSON
result instead of checkpoints, manifests, and terminal-gate artifacts.

## D-U1 revision 4

D-U1 is the controlled categorical utility×rarity environment. The six methods
are Positive-only, All-negative, matched-global, reciprocal-linear distance,
reciprocal-quadratic distance, and exponential-quadratic distance. Every method
starts from the same per-seed model and Adam state.

```bash
python -m drpo_reference du1 \
  --output outputs/du1_rev4 \
  --device cpu
```

The runner preserves the original warm start, minibatch sequence, losses,
optimizer updates, evaluation metrics, and paired Positive-only comparison. It
writes a single `results.json` containing trajectories and summaries.

## D4RL-9 locomotion performance

D4RL uses one SNA2C-IQLV actor/critic implementation across HalfCheetah, Hopper,
and Walker2d with medium, medium-replay, and medium-expert datasets. The core
path is dataset preparation → actor/critic update → optional MuJoCo rollout.

ExpRank is the default:

```bash
drpo-reference d4rl \
  --dataset-root /ABS/PATH/TO/D4RL_V2_HDF5 \
  --tasks hopper-medium-replay-v2 \
  --seeds 200,201 \
  --steps 100000 \
  --eval-episodes 10 \
  --output outputs/d4rl_hopper_medium_replay
```

The same loop can expose the historical control methods directly:

```bash
drpo-reference d4rl \
  --dataset-root /ABS/PATH/TO/D4RL_V2_HDF5 \
  --tasks hopper-medium-replay-v2 \
  --methods exprank,positive_only,signed,global,reciprocal_linear,reciprocal_quadratic,exponential \
  --seeds 200,201 \
  --steps 100000 \
  --eval-episodes 10 \
  --output outputs/d4rl_hopper_medium_replay_methods
```

Omit `--tasks` to run all nine tasks. `--eval-episodes 0` runs training only.
The runner writes one `results.json`; checkpoints, SHA/provenance gates,
completion manifests, and failure-state files are intentionally omitted from
the reference implementation.

## Countdown Transformer runtime

`drpo_reference.categorical.countdown` remains the dependency-light algorithm
core. `drpo_reference.experiments.countdown` adds the approved reviewer-facing
Transformers/PEFT lifecycle without moving those dependencies into the core.

The runtime implements explicit JSON configuration, model/tokenizer/LoRA loading,
replay and independent calibration validation, prompt-balanced paired training,
per-seed model-backed calibration, AdamW with cosine warmup, gradient
accumulation and clipping, raw/update norms, non-finite guards, best/last-finite/
terminal adapter checkpoints, delayed test access, Greedy/Pass@k generation,
completion/failure records, and method/seed mean/std aggregation.

Run it with:

```bash
drpo-reference countdown \
  --config /ABS/PATH/countdown-reviewer.json \
  --output outputs/countdown-reviewer
```

The JSON object must explicitly provide these sections and coordinates:

- `schema_version: 2` for the canonical v79 coordinate; schema 1 is retained only for custom reviewer runs;
- `model`: path, optional initial adapter, device, dtype, 4-bit flag, gradient
  checkpointing flag, and LoRA rank/alpha/dropout/target modules;
- `data`: replay, calibration, validation, and optional test JSONL paths;
- `methods` and `seeds`;
- `training`: max length, steps, micro-batch, accumulation, learning rate, weight
  decay, warmup ratio, clipping norm, evaluation cadence, checkpoint cadence;
- `calibration`: prompt count/seed, minimum scale, inherited exponential
  coefficient, search range/steps/tolerance, and nondegenerate gates;
- `evaluation`: batch size, example count, generation length, Pass@k, seed,
  selection metric, temperature, and top-p.

The command writes calibration, training, evaluation, and adapter outputs for
each method/seed run. Countdown remains an external-validity task and is separate
from the controlled D-U1 mechanism experiment.
