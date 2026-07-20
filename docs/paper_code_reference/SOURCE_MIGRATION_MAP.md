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

### Differential and provenance oracles

- `src/drpo/countdown_qwen_arena_onefile.py`: expression cleaning, verifier,
  prompting, masking, sequence statistics, model/LoRA loading, generation, and
  historical end-to-end lifecycle;
- `src/drpo/countdown_e8_alpha1_c_scan_common.py`: unique-negative handling,
  bank flattening, per-prompt denominator, and no weight-sum normalization;
- `src/drpo/countdown_e8_alpha1_c_scan_trainer.py`: Positive-only skipping,
  joint objective, AdamW update semantics, diagnostics, and checkpoint lifecycle;
- `src/drpo/countdown_e8_alpha1_highc_scan_common.py`: corrected
  linear-surprisal envelope;
- `src/drpo/countdown_e8_taper.py` and the registered active-tail protocol:
  current-policy active-tail remoteness, calibration, method weights, training,
  checkpoint, and evaluation responsibilities;
- Round-1 and active-tail protocol documents: scientific-variable provenance and
  the distinction between development pilots and an unfrozen final coordinate.

Legacy files remain differential/provenance oracles. The reviewer package does
not import them at runtime.

### Approved algorithm-core path

The exact path
`paper_code/src/drpo_reference/categorical/countdown.py` was approved before
creation and recorded in Draft PR #149 comment `5016309623`.

It owns:

- stable expression/verifier and sequence encoding/statistics;
- first-occurrence unique banks and per-prompt denominators;
- historical Round-1 linear-surprisal compatibility;
- registered active-tail coordinate and six method-weight formulas;
- deterministic detached current weights and the exact joint objective;
- prompt-balanced sampling and model-backed raw-gradient calibration;
- diagnostics, parameter-update norm, and response aggregation.

`categorical/__init__.py` exports these primitives. Existing
`tests/test_common.py` contains formula, masking, objective, two-forward,
calibration, and first-update characterization.

The core must remain free of Transformers/PEFT lifecycle responsibilities.

### Approved reviewer-runtime path

The exact path
`paper_code/src/drpo_reference/experiments/countdown.py` was separately approved
before creation and recorded in Draft PR #149 comment `5019085196`.

It owns:

- explicit JSON configuration with no hidden model, method, seed, budget,
  coefficient, checkpoint, or test defaults;
- lazy Transformers/PEFT loading, optional bitsandbytes, tokenizer, base model,
  initial adapter or fresh LoRA;
- replay/calibration/validation checks, delayed optional test access, hashes, and
  prompt-disjointness;
- per-seed model-backed calibration and shared-initialization digest checks;
- prompt-balanced paired training, AdamW/cosine scheduling, accumulation,
  clipping, raw/update norms, and non-finite rollback;
- best, last-finite, and terminal adapters;
- Greedy/Pass@k best-versus-terminal evaluation;
- lightweight per-run/root records and simple seed aggregation.

Related existing paths:

- `cli.py`: `drpo-reference countdown --config ... --output ...` dispatch;
- `pyproject.toml`: optional `countdown` and `countdown-4bit` extras;
- `tests/test_cli.py`: command-dispatch characterization;
- `README.md`: install, explicit-config contract, output, and evidence boundary.

### Compatibility and evidence rules

- the historical Round-1 and active-tail APIs remain separate;
- Positive-only skips both negative forwards;
- method models must match the calibrated initial trainable-state digest;
- test input is not read or hashed before all method training finishes;
- fixed horizon is not convergence;
- task collapse, support/probability boundary, NaN/Inf, invalid evaluation input,
  checkpoint-evaluation failure, and incomplete terminal review remain distinct;
- reviewer completion never upgrades scientific status or authorizes ranking;
- Countdown remains external validity and does not replace D-U1 controlled
  identification.

### Still not migrated or not closed

- interrupted-run resume with optimizer and scheduler state;
- real Qwen/PEFT/CUDA liveness and any fixes it reveals;
- final manuscript-facing model/initialization, methods, coefficients, seeds,
  budget, checkpoint rule, validation/test protocol, and result values;
- internal RunSpec, hardened artifact delivery, scientific terminal adjudication,
  registry/handoff mutation, and manuscript binding;
- human-only protected-environment confirmation and integration-freshness review.

### Exclude

- importing the historical one-file orchestration as the public runtime;
- superseded fixed-pair or squared-surprisal tuning stacks;
- current development grids as silent final defaults;
- dirty-worktree, smoke, fake-HF, or limited-seed results as final evidence;
- any claim that Countdown replaces controlled D-U1 causal identification.

The algorithm core and reviewer lifecycle are implemented. Real Qwen/CUDA
liveness, final protocol freeze, resume, protected human review, terminal
scientific review, and integration freshness remain open gates.

## Result reporting

The reviewer-facing package needs only commands for selected protocols,
checkpoints where applicable, lightweight completion/failure metadata, task
scores, and simple summaries. It does not need a general paper table/figure
framework, byte-identical stochastic-output verifier, manuscript-writing
pipeline, governance validator, historical result-discovery system, or internal
formal scientific audit.
