# DRPO paper reference code

This directory contains a compact reference implementation of the DRPO
experiments and comparison methods used in the paper.

C-U1 and D-U1 both use independent train and held-out contexts drawn from the
same distribution. Their result is **same-distribution held-out-context
generalization**, not OOD generalization.

## Scope

The package focuses on the experiment definitions, model updates, task adapters,
and evaluation code needed to understand and run the paper experiments.

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

Install the optional Structured Generation Transformer runtime without changing
the base CPU-oriented package:

```bash
python -m pip install -e '.[test,structured-generation]'
```

Use `structured-generation-4bit` only for an explicitly configured
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

Each command runs the corresponding numerical experiment and writes one compact
JSON result.

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

D4RL uses one shared Gaussian actor/critic implementation across HalfCheetah,
Hopper, and Walker2d with medium, medium-replay, and medium-expert datasets. The
reviewer-facing runner contains only the DRPO exponential negative taper. Its
core path is dataset preparation → DRPO actor/critic update → optional MuJoCo
rollout.

Example:

```bash
drpo-reference d4rl \
  --dataset-root /ABS/PATH/TO/D4RL_V2_HDF5 \
  --tasks hopper-medium-replay-v2 \
  --seeds 200,201 \
  --steps 1000000 \
  --eval-episodes 10 \
  --output outputs/d4rl_hopper_medium_replay
```

Omit `--tasks` to run all nine tasks. `--eval-episodes 0` runs training only.
The compact reviewer runner evaluates after training; the full paper protocol's
intermediate evaluation schedule is documented in the manuscript. The runner
writes one `results.json`.

## Structured Generation

Structured Generation uses one common path for all nine tasks:
Countdown, Word Sorting, Spiral Matrix, Mini Sudoku, Maze, Word Ladder,
Knights & Knaves, Graph Coloring, and WikiSQL. A task adapter owns only source
instance construction, output canonicalization, verification, and
model-independent negative mutations. From bank construction onward, Countdown
uses the same code path as the other eight tasks.

The shared path is:

```text
task adapter
  -> verified oracle + verifier-incorrect candidate generation
  -> canonical deduplication and deterministic 16-negative selection
  -> 5,000 train / 500 validation / 500 reserved test split
  -> common completion-only likelihood computation
  -> Positive-only objective
  -> greedy verifier success / Pass@8 evaluation
```

The frozen replay rows contain task data, verifier outputs, and the selected
negative completions. The reviewer-facing Structured Generation runner exposes
only the common task/data path and Positive-only training objective.

The bundled JSON runtime configuration is
`configs/structured_generation_0p5b.json`. Reasoning Gym
and WikiSQL source checkouts are supplied under `DRPO_STRUCTURED_SOURCES_ROOT`
as `reasoning-gym/` and `wikisql/`.

Run the reviewer-facing path on all nine tasks with:

```bash
drpo-reference structured-generation \
  --config configs/structured_generation_0p5b.json \
  --output outputs/structured-generation
```

Or select a task subset without changing the implementation:

```bash
drpo-reference structured-generation \
  --config configs/structured_generation_0p5b.json \
  --tasks countdown,wikisql \
  --output outputs/structured-generation
```

Structured Generation is external-validity evidence; C-U1 and D-U1 remain the
controlled mechanism environments.
