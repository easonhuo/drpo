# PAPER-CODE-REFERENCE-01 Source Migration Map

**Base:** `4544005bd7df69c53bad70a9dcac846af01285e4`

This map selects authoritative paper-facing source paths. Historical and superseded implementations remain in the repository but are not copied into `paper_code`.

## Shared controls and engineering utilities

| Responsibility | Authoritative source for characterization | Reference target | Decision |
|---|---|---|---|
| deterministic seeding | C-U1, D-U1 v4, Hopper E7-Q2, Countdown taper runners | `common/seeding.py` | one small implementation |
| JSON/CSV writes | same runners | `common/io.py` | one implementation; no artifact-packaging logic |
| event separation | C-U1 E3/E4, D-U1 v4, Hopper E7-Q2, Countdown taper | `common/events.py` | task, boundary, numerical, environment-invalid remain distinct |
| terminal records | registered runners | `common/audit.py` | common schema; family-specific metric computation stays local |
| taper formulas | `cu1_taper_near_retention_formal.py`, `du1_e6_cartesian_taper_v4.py`, final Countdown runner | `controls/weights.py` | one distance-coordinate implementation |
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

## Hopper

### Include

- `src/drpo/e7_hopper_q2.py`
- `configs/e7_hopper_q2.yaml`
- `scripts/run_e7_hopper_q2.py`
- delivered compact E7-Q2 outputs and registered dataset identity

### Reference split

- `continuous/hopper.py`: dataset, critic, frozen advantage, Gaussian actor, matching, rollout and audits;
- `experiments/hopper.py`: delivered E7-Q2 method matrix and commands.

### Exclude

- `src/drpo/e7_bench.py` unless the final manuscript uses E7-BENCH;
- unmerged GAE development work;
- historical v4.2 stopping logic;
- general D4RL benchmark framework work unrelated to the final paper.

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

## Paper aggregation

The final package will contain only scripts that transform selected reference-run outputs into the exact controlled/external table inputs and figure-data files. It will not contain manuscript-writing pipelines, governance validators, or historical result discovery logic.
