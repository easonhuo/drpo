# D-U1 E6 shared-semantic categorical experiments

`D-U1-E6-SEMANTIC-PILOT-01` prepares the controlled shared-semantic categorical
experiment that is separate from the completed E5 support-boundary mechanism.
It asks whether controlled local negative gradients can improve probability on a
hidden optimal catalogue action for independently sampled held-out contexts, and
whether uncontrolled far-negative pressure destroys that benefit.

## Scientific boundary

- Pilot and focused-development runs use seeds `0--4` and remain **pilot** evidence.
  The separately frozen formal long-run uses untouched seeds `10--29`.
- Train and test contexts are sampled independently from the same `N(0,I_6)`
  distribution. Use *held-out-context* or *unseen-state* generalization, not OOD.
- Task-performance collapse, support/temperature boundary events, and NaN/Inf
  numerical failure are separate outputs.
- E6-C shuffles only the policy-side action embeddings. Reward semantics, hidden
  optimal actions, demonstrations, negative labels, and state geometry remain fixed.
- Raw-gradient budget matching is not an Adam parameter-update match.
- The formal experiment `D-U1-E6-SEMANTIC-LONGRUN-01` is now frozen and active.
  It uses untouched seeds `10--29` and must launch through the hardened guard.

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

The pilot/focused YAML values remain development settings. The separate long-run YAML is frozen and validator-enforced.

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
Formal execution uses the separately registered one-click hardened-guard command below.

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

## Frozen formal long-run

The user approved the focused-development freeze on 2026-06-27. The formal
configuration is `configs/du1_e6_semantic_longrun.yaml`; development seeds 0--4
are forbidden, held-out seeds 10--29 are fixed, and the runner rejects any change
to the frozen optimizer, alpha grid, far pressure, method matrix, horizon, event
thresholds, or terminal windows.

Validate the formal config without consuming held-out seeds:

```bash
PYTHONPATH=src python src/drpo/du1_e6_semantic_longrun.py \
  --config configs/du1_e6_semantic_longrun.yaml \
  --output-root /tmp/e6-formal-check-unused \
  --check-only
```

Launch the complete formal run once, from a clean `main` checkout whose HEAD
matches `origin/main`:

```bash
python3 scripts/run_du1_e6_semantic_longrun.py \
  --work-dir experiments/results/D-U1-E6-SEMANTIC-LONGRUN-01/run_001 \
  --device cpu
```

The one-click launcher binds the exact Git commit, invokes the canonical hardened
guard in the foreground, requires the frozen result files, and creates the durable
raw-complete artifact. The guard owns `run_manifest.json`; the scientific runner
writes `scientific_run_manifest.json` and refuses unguarded formal execution.
Because the run is expected to exceed 30 minutes, compact persistent-local
checkpoint manifests are written after each five-seed block (`10--14`, `15--19`,
`20--24`, `25--29`). If the child fails, the guard packages all completed partial
evidence. The scientific runner never creates ZIP/TAR archives. Task-performance
collapse, support/temperature boundary events, and NaN/Inf numerical failure remain
separate even when a method reaches a high reward.
