# PAPER-CODE-REFERENCE-01 Source Migration Map

**Base:** `4544005bd7df69c53bad70a9dcac846af01285e4`

This map selects authoritative paper-facing source paths. Historical and superseded implementations remain in the repository but are not copied into `paper_code`.

## Shared controls and engineering utilities

| Responsibility | Authoritative source for characterization | Reference target | Decision |
|---|---|---|---|
| deterministic seeding | C-U1, D-U1 v4, Hopper E7-Q2, Countdown taper runners | `common/seeding.py` | one small implementation |
| JSON/CSV writes | same runners | `common/io.py` | one implementation; no artifact-packaging logic |
| event separation | C-U1 E3/E4, D-U1 v4, Hopper E7-Q2, Countdown taper | `common/events.py` | task, boundary, numerical, environment-invalid remain distinct |
| terminal records | registered runners | family-local audit plus shared schemas | common schema; family-specific metric computation stays local |
| taper formulas | `cu1_taper_near_retention_formal.py`, `du1_e6_cartesian_taper_v4.py`, final Countdown runner | `controls/weights.py` | one distance-coordinate implementation where formulas are identical |
| hard selection | C-U1 E3 and Hopper E7-Q2 | `controls/selection.py` | one detached mask implementation |
| budget matching | C-U1 fairness controls, D-U1 v4, Hopper E7-Q2, Countdown taper | `controls/budget.py` | one validated scalar matcher; task modules own audit subsets |

## C-U1

### Include

- `src/drpo/cu1_core.py`
- `src/drpo/drpo_cu1_e1_e4_oneclick.py`
- `src/drpo/cu1_distance_taper_formal.py`
- `src/drpo/cu1_taper_near_retention_formal.py`
- `src/drpo/cu1_taper_budget_match_formal.py`
- frozen paper-facing configurations and compact delivered outputs associated with the selected claims

### Reference split

- `continuous/gaussian.py`: Gaussian log probability, standardized distance, score diagnostics;
- `continuous/cu1.py`: environment, policy, objectives, evaluation, boundary metrics;
- `experiments/cu1.py`: thin selection of E1/E2/E3/E4 and taper protocols.

### Exclude

- historical recovered copies;
- runner-owned ZIP creation and governance checks;
- superseded convergence or tuning paths not used by the final paper;
- plotting code that does not generate manuscript data.

## D-U1

### Include

- `src/drpo/du1_e6_cartesian_taper_v4.py`
- `configs/du1_e6_cartesian_taper_v4.yaml`
- the delivered protocol-revision-4 compact outputs if selected by the manuscript

### Reference split

- `categorical/surprisal.py`: normalized excess surprisal and distance conversion;
- `categorical/du1.py`: revision-4 environment, policy, cell assignment, metrics;
- `experiments/du1.py`: six-method formal path.

### Exclude

- `du1_e6_cartesian_taper.py` revisions 1/2;
- revision-3 development grids;
- reciprocal-quartic historical method;
- semantic-gap predecessors unless the final manuscript cites a result from them explicitly.

## D4RL locomotion

The manuscript uses two scientifically distinct external profiles:

1. Hopper E7-Q2 is mechanism validation with a frozen canonical critic, frozen advantages, actor-only updates, matched near/far diagnostics, and targeted interventions;
2. D4RL-9 is task-performance validation over HalfCheetah, Hopper, and Walker2d under medium, medium-replay, and medium-expert datasets.

A source audit showed that these profiles do **not** currently have an identical full training backend. The Hopper mechanism policy uses a transformed squashed-Gaussian likelihood with an inverse-tanh and Jacobian correction, while the legacy D4RL benchmark candidate uses the canonical SNA2C/IQLV agent family with dynamic TD advantages and joint actor/critic updates. They must not be forced into one scientifically incorrect trainer merely to reduce filenames.

### Contracts that may be shared

- D4RL-v2 task and dataset identities;
- HDF5 locomotion field validation where the schema is identical;
- observation/action shape checks;
- Gymnasium/MuJoCo reset, step, termination, truncation, seeding, and action-boundary handling;
- D4RL reference-score normalization;
- common I/O, provenance, and event taxonomy;
- one task catalog for the nine manuscript coordinates.

### Contracts that remain backend-specific

- actor likelihood and action-density semantics;
- critic lifecycle and objective;
- advantage estimator and whether it is frozen or recomputed;
- actor/critic optimizer scheduling;
- method family and weight transformation;
- terminal-audit rules tied to that training protocol.

### Include

- Hopper E7-Q2 authoritative mechanism sources:
  - `src/drpo/e7_hopper_q2.py`
  - `configs/e7_hopper_q2.yaml`
  - `scripts/run_e7_hopper_q2.py`
  - delivered compact E7-Q2 outputs and registered dataset identity;
- the audited D4RL performance-backend candidate sources:
  - `src/drpo/e7_canonical_vendor/d4rl/agents.py`
  - `src/drpo/e7_canonical_vendor/d4rl/train_sna2c_variant.py`
  - `src/drpo/e7_canonical_shortlist_protocol.py`;
- the final D4RL-9 backend only after its algorithm, methods, datasets, seeds, budget, and terminal audit are explicitly frozen;
- D4RL-9 task identities only with verified provenance.

### Reference split

- existing `external/hopper_*` modules: Hopper E7-Q2 mechanism backend; historical filenames do not make them the universal D4RL-9 performance trainer;
- `external/d4rl_tasks.py`: task specifications and provenance state for the nine manuscript coordinates;
- `experiments/hopper.py`: Hopper E7-Q2 mechanism profile;
- `experiments/d4rl.py`: D4RL-9 matrix, audited backend boundary, formal blockers, and dispatch to one selected performance backend across all nine tasks.

### Compatibility rule

- current Hopper behavior must remain differential-test equivalent;
- HalfCheetah, Hopper, and Walker2d performance tasks must use one selected D4RL performance backend, not separate per-task trainer copies;
- the Hopper mechanism runner must not be reused as the performance runner unless a later scientific audit proves exact contract equivalence and freezes that change;
- unresolved dataset SHA values remain unresolved and block formal use;
- pilot-only canonical shortlist code is provenance and a backend candidate, not a frozen D4RL-9 formal protocol;
- task-performance collapse, support/variance-boundary events, rollout failures, and NaN/Inf numerical failures remain separately reported.

### Exclude

- copied `halfcheetah_*`, `hopper_*`, or `walker2d_*` performance trainers;
- an invented hybrid that silently combines Hopper mechanism likelihoods with the canonical D4RL optimizer/advantage lifecycle;
- unmerged GAE development work;
- historical stopping logic or parameter sweeps not selected by the final manuscript;
- table/figure-generation machinery inside training code.

## Countdown

### Stable sources that may be characterized now

- `src/drpo/countdown_qwen_arena_onefile.py` for currently shared verifier/model/data behavior;
- `src/drpo/countdown_e8_taper.py` for already registered continuous-surprisal controls;
- the current paper-aligned runner only after its protocol/result is merged and frozen on `main`.

### Reference split

- `categorical/countdown.py`: verifier, masking, sequence log-probability, training and evaluation;
- `experiments/countdown.py`: final selected protocol only.

### Exclude

- historical one-file orchestration as a public entry point;
- superseded fixed-pair and squared-surprisal tuning stacks;
- dirty-worktree or single-seed pilot claims as final evidence.

Countdown migration remains blocked at the final experiment-entry layer until the manuscript-facing protocol and result are frozen. Shared task-independent control functions are not blocked.

## Result reporting

The final reference package needs only a minimal result summary and protocol audit sufficient to recompute the manuscript's selected scientific conclusions. It does not need a general paper table/figure-generation framework, byte-identical stochastic-output verifier, manuscript-writing pipeline, governance validator, or historical result-discovery system.
