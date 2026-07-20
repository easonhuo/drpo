# PAPER-CODE-REFERENCE-01 Current Status

**Document role:** canonical task-local current snapshot and continuation index.  
**Not a research master:** `docs/handoff.md` remains the unique research source of truth.  
**Claim:** `PAPER-CODE-REFERENCE-01`  
**Scientific-status impact:** none.  
**Last engineering-validated code head:** `1cc7bab43f2fc598c813e069655b778156e68e47`.

This file consolidates the current engineering state without deleting historical
records. It supersedes stale *current-status* and *next-slice* statements in the
older task-local handoff and delta files. Those older files remain provenance for
the order in which the migration was implemented and reviewed.

## 1. Mandatory continuation order

A new session must read and verify, in this order:

1. `AGENTS.md` from current `main`;
2. Section 0 of `docs/handoff.md` from current `main`;
3. `experiments/registry.yaml` from current `main`;
4. this file;
5. `docs/paper_code_reference/ACCEPTANCE_MATRIX.yaml`;
6. `docs/paper_code_reference/SOURCE_MIGRATION_MAP.md` for source ownership and
   historical inclusion/exclusion decisions;
7. `docs/paper_code_reference/IMPLEMENTATION_PLAN.md` as the original architecture
   and acceptance plan, not as a live file inventory;
8. the actual `dev/paper-code-reference-01` branch, Draft PR `#149`, changed files,
   legacy differential oracles, and exact-head CI.

The older `SESSION_HANDOFF*` files are append-only historical implementation
records. They are no longer the preferred way to discover the current next step.

## 2. Repository and branch snapshot

At the audit immediately before this document:

- repository: `easonhuo/drpo`;
- default branch: `main`;
- current `main`: `dd46727c1efefd2e6d4cdf6f3b204ec1fc58fca3`;
- task base and current merge base: `4544005bd7df69c53bad70a9dcac846af01285e4`;
- only active paper-code development branch: `dev/paper-code-reference-01`;
- engineering-validated code head: `1cc7bab43f2fc598c813e069655b778156e68e47`;
- persistent cumulative Draft PR: `#149`;
- PR state: open, Draft, unmerged;
- overall task state: `in_development`.

The SHA values above are audit facts, not reusable assumptions. Every continuation
session must resolve both heads again. The branch remains separate from `main`
until the user explicitly authorizes a merge decision.

The branch is materially behind newer unrelated `main` work and currently
diverges from it. This does not by itself prove a content conflict, but it requires
an integration-freshness audit before a final merge proposal. Do not silently
rebase, merge `main`, or reinterpret newer scientific registrations as part of
this task.

## 3. Reviewer-facing code boundary

The public `paper_code` package provides readable algorithms, explicit dataset and
environment identities, runnable training/evaluation when a selected protocol is
frozen, checkpoints, lightweight completion/failure records, and simple
summaries.

It does not duplicate registry/handoff mutation, formal-evidence promotion, full
scientific terminal adjudication, manuscript artifact binding, or internal result
promotion. Lightweight command completion is not a scientific terminal audit.

Training and evaluation scores may vary across hardware, dependencies, and random
seeds. Reviewer-facing reproducibility means the algorithm and stated protocol
are readable and runnable; it does not promise byte-identical stochastic scores.

## 4. Two-axis status model

Do not conflate scientific status registered on `main` with `paper_code`
reproduction status.

| Component | Existing scientific status | `paper_code` migration status | Remaining reviewer-code gate |
|---|---|---|---|
| Shared utilities and controls | no independent scientific claim | implementation complete; engineering validated | final integration review |
| C-U1 | experiment-specific statuses remain authoritative in handoff/registry | implementation complete | registered reproduction remains internal scientific work |
| D-U1 revision 4 | `not_run` for the active formal matrix | implementation complete | formal run remains internal scientific work |
| Hopper E7-Q2 | existing learned-critic external mechanism result is `long_run_validated` | implementation complete | real registered-data reproduction through the new runner |
| D4RL-9 / `EXT-H-E7-BENCH-01` | historical archive is pilot provenance only; no formal ranking | reviewer algorithm, multi-method training, rollout, and aggregation complete | real HDF5/MuJoCo liveness |
| Countdown / `EXT-C-E8-TAPER-0.5B-01` | registered active-tail experiment remains pilot and `not_run` | stable algorithm core complete and engineering validated | selected model/runtime, full lifecycle, public entry, and real Qwen/CUDA liveness |

A smoke, static check, first update, or fixed short trajectory is engineering
evidence only. It does not change scientific status.

## 5. Current implementation ownership

### 5.1 C-U1

The live implementation is split by responsibility across `continuous/cu1*.py`,
with public dispatch in `cli.py`. It covers source, causal, phase/control, taper,
artifacts, aggregation, and audit for the controlled continuous environment.

### 5.2 D-U1 revision 4

The live implementation is split across `categorical/du1_*.py`, with public
dispatch in `cli.py`. It covers the revision-4 environment, categorical policy,
six frozen controls, shared-start training, metrics, public execution, and
reports. Revisions 1/2/3 and the historical reciprocal-quartic method remain
excluded.

### 5.3 Hopper E7-Q2 mechanism profile

The Hopper mechanism implementation includes data, models, critic, frozen
advantages, actor training, diagnostics, rollout, suite execution, public runner,
aggregation, and terminal records. It remains scientifically distinct from the
D4RL-9 task-performance backend.

### 5.4 D4RL-9 performance profile

Implemented reviewer-facing code includes the exact nine-task catalog, one
canonical SNA2C-IQLV actor/critic lifecycle, reviewer-selectable historical
controls behind an explicit pilot profile, selected-task or all-nine-task
training, direct Gymnasium/MuJoCo rollout evaluation, checkpoints, lightweight
completion/failure records, and task × method × seed aggregation.

No additional D4RL actor, critic, multi-method training-loop, rollout, or simple
aggregation migration is currently required. Real liveness and the internal
formal experiment lifecycle are execution/protocol gates rather than missing
reviewer code.

### 5.5 Countdown stable algorithm core

The approved path `categorical/countdown.py` owns reusable Countdown behavior
below the final experiment-entry layer.

Stable sequence and historical Round-1 compatibility:

- canonical expression cleaning, exact arithmetic verification, and mutually
  exclusive verifier categories;
- chat prompt rendering with thinking disabled when supported;
- prompt/completion encoding, EOS inclusion, prompt-label masking, and padding;
- completion-only sequence log-probability, entropy, and bounded direct-logit
  score;
- first-occurrence unique-negative bank encoding and flattened batching;
- explicit per-prompt unique-negative denominators, never weight-sum
  normalization;
- detached historical coordinate `u=-log P(y|x)/2` and
  `alpha * exp(-c*u)` weights;
- historical joint objective retained as an explicit compatibility layer;
- bank/weight diagnostics, parameter-update norm, and lightweight Greedy/Pass@k
  aggregation.

Registered E8-TAPER active-tail core:

- frozen method catalog: Positive-only, Uncontrolled negative, Global matched,
  Reciprocal-linear, Exponential, and Squared-distance exponential;
- detached learner-relative coordinate
  `S=max(-log P(y|x)-tau,0)/surprisal_scale`, `d=sqrt(S)`;
- exact method weights, including
  `exp(-lambda*d^2)=exp(-lambda*S)` rather than a quartic taper;
- independent-calibration median scale, common-half-median `tau`, active-tail
  diagnostics, prompt-balanced sampling, coefficient bracket scan/bisection, and
  nondegenerate fail-closed gate;
- exact active-tail joint objective
  `-(mean positive lp - shared_negative_scale * mean_per_prompt(weight*negative lp))`;
- deterministic current weights from an `eval()` plus `no_grad()` negative pass,
  followed by a separate gradient-bearing negative forward after restoring model
  mode;
- Positive-only skips both negative forwards;
- model-backed full-trainable-parameter raw gradient-L2 calibration for positive,
  uncontrolled negative, inherited exponential target, Global matched,
  Reciprocal-linear, and Squared-distance exponential;
- frozen `shared_negative_scale=positive_gradient_l2/uncontrolled_negative_gradient_l2`;
- calibration output explicitly records that confirmation/test metrics were not
  used and coefficients were frozen before method training;
- first clipped AdamW update identity for the active-tail objective.

The historical linear objective and active-tail objective remain separate APIs.
The migration does not silently reinterpret completed Round-1 linear-surprisal
pilots as v79 active-tail runs.

The module imports neither Transformers/PEFT nor the historical one-file runner.
It does not select a model path, LoRA initialization, optimizer/scheduler,
training horizon, checkpoint rule, seed set, test protocol, or final manuscript
coordinate.

Focused local validation passed 19 tests. Exact-head CI on
`1cc7bab43f2fc598c813e069655b778156e68e47` passed Evidence Locator, Python
compile, shell syntax, handoff authority, formal execution channel, governance
inventory/stage, full pytest, and Ruff.

## 6. Countdown remaining work

### 6.1 Algorithm-core status

The protocol-independent algorithm core is now closed for the currently
registered active-tail lineage. The remaining missing pieces are lifecycle and
runtime code, not absent weight/objective/calibration formulas.

### 6.2 Lifecycle/runtime code still absent

- Transformers tokenizer/model loading;
- PEFT/LoRA adapter creation, loading, and identity checks;
- optional bf16/4-bit/gradient-checkpointing runtime boundaries;
- replay-file and calibration-file loading/validation for the selected public
  protocol;
- optimizer and scheduler construction;
- gradient accumulation, clipping, non-finite guards, and the complete multi-step
  loop;
- checkpoint persistence, resume, best/last-finite/terminal selection;
- real greedy and sampled generation evaluation;
- selected-checkpoint and terminal evaluation;
- public run manifests, completion/failure records, and simple seed aggregation;
- `paper_code/src/drpo_reference/experiments/countdown.py`;
- a `drpo-reference countdown` CLI command;
- real Qwen/CUDA liveness.

### 6.3 Scientific/protocol items still unresolved

- final manuscript-facing Countdown result and public reproduction coordinate;
- final model scale and initialization;
- final comparison arms and default coefficient source;
- formal/confirmatory seeds and budget;
- checkpoint-selection rule;
- validation/test access policy and final evaluation protocol;
- manuscript values or method-ranking claims.

No new Python path for the final experiment entry has been approved. Do not create
it by inference from the stable-core approval.

## 7. Exact next sequence

1. Treat the Countdown stable algorithm core as frozen pending review.
2. Audit which selected model-loading, optimizer/scheduler, checkpoint, and
   evaluation lifecycle belongs in reviewer code versus internal governance.
3. Propose the exact final experiment-entry Python path and responsibility before
   creating it.
4. After separate path approval, migrate the selected lifecycle and public CLI.
5. Execute controlled real Qwen/CUDA liveness before any larger reproduction.
6. Perform integration-freshness audit against current `main` before any merge
   proposal.

No formal experiment was launched by this migration. No method ranking,
scientific status, model scale, seed, threshold, training horizon, default
coefficient, checkpoint rule, or manuscript value changed.
