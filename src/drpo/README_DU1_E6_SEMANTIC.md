# D-U1 E6 shared-semantic categorical pilot

`D-U1-E6-SEMANTIC-PILOT-01` prepares the controlled shared-semantic categorical
experiment that is separate from the completed E5 support-boundary mechanism.
It asks whether controlled local negative gradients can improve probability on a
hidden optimal catalogue action for independently sampled held-out contexts, and
whether uncontrolled far-negative pressure destroys that benefit.

## Scientific boundary

- This is a **pilot** using development seeds `0--4`; it is not a formal result.
- Train and test contexts are sampled independently from the same `N(0,I_6)`
  distribution. Use *held-out-context* or *unseen-state* generalization, not OOD.
- Task-performance collapse, support/temperature boundary events, and NaN/Inf
  numerical failure are separate outputs.
- E6-C shuffles only the policy-side action embeddings. Reward semantics, hidden
  optimal actions, demonstrations, negative labels, and state geometry remain fixed.
- Raw-gradient budget matching is not an Adam parameter-update match.
- The formal experiment ID `D-U1-E6-SEMANTIC-LONGRUN-01` is reserved but blocked
  until the pilot produces a user-reviewed parameter-freeze record.

## Registered development geometry

- 6D state, 64 unordered actions, 4D unit semantic embeddings.
- `t_star = normalize(t_plus + 0.45 d)` and
  `t_minus = normalize(t_plus - 0.45 d)`.
- The hidden optimal action is the reward argmax nearest `t_star` and is excluded
  from four positive demonstrations near `t_plus`.
- One local negative is selected near `t_minus`; four far negatives are selected
  near `-t_plus`. All negative advantages are exactly `-1` and frozen.
- A shared MLP emits a semantic direction. Fixed-concentration runs isolate
  semantic extrapolation; learnable concentration runs expose support/temperature
  dynamics without an upper clamp.

## Protocols

- **E6-A:** positive-only versus local-negative alpha scan.
- **E6-B:** positive-only, local-only/Far-zero, uncontrolled, Near-zero, Far-cap,
  and raw-gradient-budget-matched global scaling.
- **E6-C:** aligned versus shuffled policy-side semantic embeddings.

The YAML values are development settings, not frozen formal hyperparameters.

## Commands

Invariant-only preflight:

```bash
PYTHONPATH=src python src/drpo/du1_e6_semantic.py \
  --config configs/du1_e6_semantic_pilot.yaml \
  --stage invariants \
  --output-root outputs/du1_e6_invariants \
  --device cpu
```

Engineering smoke:

```bash
PYTHONPATH=src python src/drpo/du1_e6_semantic.py \
  --config configs/du1_e6_semantic_pilot.yaml \
  --stage smoke \
  --output-root outputs/du1_e6_smoke \
  --device cpu
```

Development pilot on the available accelerator:

```bash
PYTHONPATH=src python src/drpo/du1_e6_semantic.py \
  --config configs/du1_e6_semantic_pilot.yaml \
  --stage pilot \
  --output-root experiments/results/D-U1-E6-SEMANTIC-PILOT-01/run_001 \
  --device auto
```

Every attempt must use a new or empty output directory. The runner writes ordinary
JSON/JSONL/CSV/YAML files only; it does not create an experiment artifact archive.
Formal execution must later use a separately registered hardened-guard command.

## Output interpretation

`pilot_freeze_recommendation.json` summarizes development candidates but sets
`automatic_freeze_allowed=false`. The user must review the development evidence
before alpha, concentration, optimizer, stopping rules, thresholds, and untouched
held-out seeds are registered for the long-run experiment.

## Focused blocker-resolution extension

`configs/du1_e6_semantic_focused_dev.yaml` registers
`D-U1-E6-SEMANTIC-FOCUSED-DEV-01`. It keeps development seeds 0--4 and the
original D-U1 geometry, doubles the pilot horizon to 4000 steps, and uses only
existing alpha/far-pressure variables. Phase 1 verifies fixed-concentration
terminal behavior and screens lower learnable-concentration local pressure. A
phase-2 far-pressure scan is allowed only after the pre-registered phase-1
selection rule chooses a safe local alpha. The focused terminal classification
uses window-mean stability and gradient/update growth ratios; it does not require
a stochastic raw gradient to vanish and does not constitute formal acceptance.
