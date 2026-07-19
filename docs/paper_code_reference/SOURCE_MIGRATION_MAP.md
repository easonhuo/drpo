# PAPER-CODE-REFERENCE-01 Source Migration Map

**Base:** `4544005bd7df69c53bad70a9dcac846af01285e4`  
**Live status authority:** `docs/paper_code_reference/CURRENT_STATUS.md`

This map selects authoritative paper-facing source paths and records inclusion,
exclusion, and responsibility boundaries. Historical and superseded
implementations remain in the repository as provenance references. Early
conceptual target names preserved in Git history are not the live file layout.

## Reviewer-facing boundary

The public `paper_code` package must provide readable algorithms, explicit data
and environment identities, runnable training and rollout evaluation when a
protocol is selected, checkpoints, lightweight completion/failure records, and
simple summaries.

It does not duplicate internal registry/handoff governance, formal-evidence
promotion, full scientific terminal adjudication, or manuscript table-cell and
artifact-hash binding. Those remain internal repository responsibilities.

## Shared controls and engineering utilities

| Responsibility | Authoritative source for characterization | Live reference target | Decision |
|---|---|---|---|
| deterministic seeding | C-U1, D-U1 v4, Hopper E7-Q2, canonical D4RL | `common/seeding.py` plus family-local generators | preserve family-specific seed contracts |
| JSON/CSV writes | registered runners | `common/io.py` | one implementation; no artifact-packaging logic |
| event separation | C-U1, D-U1 v4, Hopper E7-Q2, Countdown taper | `common/events.py` plus family-local records | task, boundary, numerical, and environment-invalid remain distinct |
| lightweight public completion | reviewer-facing runners | family-local runner records | command completion only; not formal scientific terminal audit |
| taper formulas | selected C-U1/D-U1/Countdown sources | `controls/weights.py` plus sequence-specific core | share only mathematically identical controls |
| hard selection | C-U1 and Hopper E7-Q2 | `controls/selection.py` | one detached-mask implementation |
| budget matching | controlled and Hopper paths | `controls/budget.py` | shared scalar matcher; family modules own audit subsets |

## C-U1

### Include

- `src/drpo/cu1_core.py`;
- `src/drpo/drpo_cu1_e1_e4_oneclick.py`;
- `src/drpo/cu1_distance_taper_formal.py`;
- `src/drpo/cu1_taper_near_retention_formal.py`;
- `src/drpo/cu1_taper_budget_match_formal.py`;
- frozen paper-facing configurations and selected compact outputs.

### Live reference split

- `continuous/gaussian.py`: Gaussian log probability, standardized distance,
  and score primitives;
- `continuous/cu1.py`: controlled environment, policy objectives, evaluation,
  and boundary metrics;
- `continuous/cu1_training.py`: Positive-only and shared training lifecycle;
- `continuous/cu1_source_causal.py` and `continuous/cu1_mechanism.py`: source
  isolation and causal transmission;
- `continuous/cu1_phase.py`, `continuous/cu1_control.py`, and
  `continuous/cu1_phase_taper.py`: negative-strength scans and controls;
- `continuous/cu1_taper.py`: taper-family execution;
- `continuous/cu1_suite.py`, `continuous/cu1_public_protocol.py`,
  `continuous/cu1_public_audit.py`, and `continuous/cu1_artifacts.py`: public
  execution and artifacts;
- `cli.py`: public command dispatch.

### Exclude

- historical recovered copies;
- runner-owned ZIP creation and governance checks;
- superseded convergence or tuning paths not used by the final paper;
- plotting code that does not generate selected manuscript data.

## D-U1 revision 4

### Include

- `src/drpo/du1_e6_cartesian_taper_v4.py`;
- `configs/du1_e6_cartesian_taper_v4.yaml`;
- selected protocol-revision-4 compact outputs when formally accepted.

### Live reference split

- `categorical/du1_environment.py`: utility × rarity environment and dynamic
  common/rare roles;
- `categorical/du1_policy.py`: categorical policy;
- `categorical/du1_controls.py`: the six frozen revision-4 controls and
  calibration;
- `categorical/du1_training.py`: shared-start training and updates;
- `categorical/du1_metrics.py`: task and mechanism metrics;
- `categorical/du1_protocol.py`: revision-4 coordinate;
- `categorical/du1_suite.py`, `categorical/du1_public.py`, and
  `categorical/du1_reports.py`: public execution, aggregation, and reports;
- `cli.py`: public command dispatch.

### Exclude

- `du1_e6_cartesian_taper.py` revisions 1/2;
- revision-3 development grids;
- reciprocal-quartic historical method;
- semantic-gap predecessors unless the final manuscript explicitly selects one
  of their results.

## D4RL locomotion

The manuscript uses two scientifically distinct external profiles:

1. Hopper E7-Q2 is mechanism validation with a frozen canonical critic, frozen
   advantages, actor-only updates, matched near/far diagnostics, and targeted
   interventions.
2. D4RL-9 is task-performance validation over HalfCheetah, Hopper, and Walker2d
   under medium, medium-replay, and medium-expert datasets.

The repository owner selected `SNA2C_IQLV_ExpRankAgent` as the D4RL-9
performance backend. All nine performance tasks use one migrated implementation.
Hopper E7-Q2 remains a separate mechanism trainer.

### Contracts that may be shared

- D4RL-v2 task and dataset identities;
- HDF5 locomotion field validation where the schema is identical;
- observation/action shape checks;
- Gymnasium/MuJoCo reset, step, termination, truncation, episode seeding, and
  action-boundary handling;
- D4RL reference-score normalization;
- common I/O and lightweight failure records;
- one task catalog for the nine manuscript coordinates.

### Contracts that remain backend-specific

- actor likelihood and action-density semantics;
- critic lifecycle and objective;
- advantage estimator and whether it is frozen or recomputed;
- actor/critic optimizer scheduling;
- method family and weight transformation;
- formal experiment protocol and internal scientific terminal review.

### Include

- Hopper E7-Q2 authoritative mechanism sources:
  - `src/drpo/e7_hopper_q2.py`;
  - `configs/e7_hopper_q2.yaml`;
  - `scripts/run_e7_hopper_q2.py`;
  - registered dataset identity and selected compact outputs;
- selected canonical D4RL performance oracles:
  - `src/drpo/e7_canonical_vendor/d4rl/agents.py`;
  - `src/drpo/e7_canonical_vendor/d4rl/train_sna2c_variant.py`;
  - `src/drpo/e7_canonical_vendor/d4rl/d4rl_common/train_loop.py`;
  - `src/drpo/e7_canonical_vendor/d4rl/d4rl_common/normalize.py`;
- D4RL-9 task identities with explicit verified or unresolved provenance.

### Live reference split

- `external/hopper_*`: Hopper E7-Q2 mechanism implementation;
- `experiments/hopper.py`: Hopper E7-Q2 public mechanism runner;
- `external/d4rl_tasks.py`: D4RL-9 task catalog, dataset provenance state,
  rollout identities, and reference-score constants;
- `experiments/d4rl.py`: canonical D4RL actor, critic, ExpRank update, locomotion
  preparation, deterministic minibatch trainer, checkpoint contract, and legacy
  differential boundary;
- `experiments/__init__.py`: reviewer-facing D4RL task/seed orchestration, HDF5
  loading, training, direct Gymnasium/MuJoCo rollout evaluation, lightweight
  completion/failure records, and simple seed aggregation;
- `cli.py`: `drpo-reference d4rl` public command;
- `pyproject.toml`: optional `rollout` dependency extra.

### Compatibility rules

- current Hopper behavior remains differential-test equivalent;
- HalfCheetah, Hopper, and Walker2d performance tasks use the same migrated
  trainer, not per-task copies;
- legacy canonical code remains the differential oracle, not a runtime
  dependency of the paper package;
- initialization, forward values, first Adam update, and a fixed short trajectory
  match the legacy oracle;
- the Hopper mechanism trainer is not reused as the D4RL performance trainer;
- the D4RL reviewer evaluator may use shared low-level Gymnasium semantics but
  remains a direct readable evaluator rather than importing Hopper's formal
  mechanism audit lifecycle;
- unresolved dataset SHA values remain unresolved and block formal use without
  blocking explicitly non-formal reviewer execution;
- reviewer-code migration does not freeze the formal method matrix, ten-run
  seeds, budgets, coefficients, checkpoint policy, or result status;
- environment failure is reported as rollout unavailability and is never
  converted into a task-performance score.

### Exclude

- copied `halfcheetah_*`, `hopper_*`, or `walker2d_*` performance trainers;
- an invented hybrid combining Hopper mechanism likelihoods with the canonical
  D4RL optimizer/advantage lifecycle;
- unmerged GAE development work;
- historical stopping logic or parameter sweeps not selected by the manuscript;
- registry/handoff mutation from reviewer commands;
- internal formal-evidence promotion and manuscript table/figure binding inside
  training code.

## Countdown

### Stable source oracles

The current stable-core characterization uses:

- `src/drpo/countdown_qwen_arena_onefile.py` for expression cleaning, exact
  verifier semantics, chat prompt rendering, completion masking, sequence
  likelihood/statistics, and response metrics;
- `src/drpo/countdown_e8_alpha1_c_scan_common.py` for first-occurrence
  unique-negative handling and denominator semantics;
- `src/drpo/countdown_e8_alpha1_highc_scan_common.py` for the corrected
  paper-aligned linear-surprisal envelope;
- `docs/experiments/E8_PAPER_ALIGNED_LINEAR_SCAN_PROTOCOL.md` and
  `docs/experiments/E8_PAPER_ALIGNED_LINEAR_C_EXTENSION_PROTOCOL.md` for the
  stable formula and the boundary between completed development scans and an
  unfrozen final manuscript protocol.

These legacy files remain differential/provenance oracles. They are not imported
at runtime by the reviewer package.

### Approved live reference path

- `categorical/countdown.py`: stable expression/verifier, prompt/completion
  encoding, completion-only likelihood/statistics, unique-bank deduplication,
  normalized sequence surprisal, detached `alpha * exp(-c*u)` weights, and
  lightweight response metrics;
- `categorical/__init__.py`: exports stable primitives without creating an
  experiment command;
- existing `tests/test_common.py`: formula, mask, verifier, bank, and response
  aggregation characterization.

The exact path `paper_code/src/drpo_reference/categorical/countdown.py` and this
limited responsibility were explicitly approved by the user and preserved in
Draft PR #149 comment `5016309623` before creation.

### Contracts intentionally not migrated yet

- model or LoRA loading;
- GPU/resource scheduling;
- training-loop and checkpoint selection;
- coefficient or method-matrix selection;
- development-to-confirmatory seed promotion;
- validation/test access policy for the final experiment;
- RunSpec, artifact delivery, or internal terminal adjudication;
- a public Countdown experiment entry point or CLI command.

The nearest D-U1 modules remain unsuitable for these sequence primitives because
D-U1 models unordered categorical actions rather than autoregressive completion
tokens. The stable-core module therefore remains separate while sharing no
scientific responsibility with D-U1.

### Exclude

- historical one-file orchestration as the public entry point;
- superseded fixed-pair and squared-surprisal tuning stacks;
- current development coefficient grids as a silent final default;
- dirty-worktree, smoke, or two-seed pilot claims as final evidence;
- any claim that Countdown replaces controlled D-U1 causal identification.

Countdown stable-core migration is implemented. The final experiment-entry layer
remains blocked until the manuscript-facing protocol and result are frozen and a
separate exact Python path is proposed and approved.

## Result reporting

The reviewer-facing package needs only commands for selected protocols,
checkpoints where applicable, lightweight completion/failure metadata, task
scores, and simple summaries. It does not need a general paper table/figure
framework, byte-identical stochastic-output verifier, manuscript-writing
pipeline, governance validator, historical result-discovery system, or internal
formal scientific audit.
