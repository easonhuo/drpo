# PAPER-CODE-REFERENCE-01 Current Status

**Document role:** canonical task-local current snapshot and continuation index.  
**Not a research master:** `docs/handoff.md` remains the unique research source of truth.  
**Claim:** `PAPER-CODE-REFERENCE-01`  
**Scientific-status impact:** none.  
**Last engineering-validated code head:** `69e60c5bc944b47ab403b6432f1efdb118bc6fa4`.

This file consolidates the current engineering state without deleting historical records. It supersedes stale *current-status* and *next-slice* statements in the older task-local handoff and delta files. Those older files remain provenance for the order in which the migration was implemented and reviewed.

## 1. Mandatory continuation order

A new session must read and verify, in this order:

1. `AGENTS.md` from current `main`;
2. Section 0 of `docs/handoff.md` from current `main`;
3. `experiments/registry.yaml` from current `main`;
4. this file;
5. `docs/paper_code_reference/ACCEPTANCE_MATRIX.yaml`;
6. `docs/paper_code_reference/SOURCE_MIGRATION_MAP.md` for source ownership and historical inclusion/exclusion decisions;
7. `docs/paper_code_reference/IMPLEMENTATION_PLAN.md` as the original architecture and acceptance plan, not as a live file inventory;
8. the actual `dev/paper-code-reference-01` branch, Draft PR `#149`, changed files, legacy differential oracles, and exact-head CI.

The older `SESSION_HANDOFF*` files are append-only historical implementation records. They are no longer the preferred way to discover the current next step.

## 2. Repository and branch snapshot

At the audit immediately before this document:

- repository: `easonhuo/drpo`;
- default branch: `main`;
- current `main`: `939533e2d5933e06a441f6b0f8c2f9e58ce952e4`;
- task base and current merge base: `4544005bd7df69c53bad70a9dcac846af01285e4`;
- only active development branch: `dev/paper-code-reference-01`;
- development head before this document: `d6e46f2eae9f3cca71b00b7cedf80fe603f03aae`;
- persistent cumulative Draft PR: `#149`;
- PR state: open, Draft, unmerged;
- overall task state: `in_development`.

The SHA values above are audit facts, not reusable assumptions. Every continuation session must resolve both heads again. The branch remains separate from `main` until the user explicitly authorizes a merge decision.

The branch is materially behind newer unrelated `main` work and currently diverges from it. This does not by itself prove a content conflict, but it requires an integration-freshness audit before a final merge proposal. Do not silently rebase, merge `main`, or reinterpret newer scientific registrations as part of this task.

## 3. Reviewer-facing code boundary

This boundary governs all remaining migration work.

The public `paper_code` package is reviewer-facing reference code. Its primary obligations are:

- readable algorithm implementation;
- explicit dataset and environment identities;
- runnable training entry points when the selected protocol is frozen;
- runnable evaluation;
- checkpoints and basic run metadata for public runners;
- a lightweight completion/failure record;
- simple multi-seed or response-level summaries.

It is **not** required to duplicate the repository's internal scientific-governance platform. The following remain internal responsibilities and are not hard requirements for reviewer-facing migration closure:

- registry or handoff mutation;
- formal-evidence eligibility decisions;
- full task × method × seed completeness governance;
- selected-versus-terminal checkpoint scientific adjudication;
- manuscript table-cell and artifact-hash binding;
- internal collapse-taxonomy adjudication and formal result promotion.

The public code still fails clearly on missing files, invalid shapes, non-finite inputs or training, unavailable rollout environments, and incomplete commands. Lightweight completion fields are normal software robustness, not an internal formal audit.

Training and evaluation scores may vary across hardware, dependency versions, and random seeds. Reviewer-facing reproducibility means the algorithm and stated protocol are readable and runnable; it does not promise byte-identical single-run scores on every machine.

## 4. Two-axis status model

Do not conflate the scientific status registered in the main repository with the reproduction status of `paper_code`.

| Component | Existing scientific status | `paper_code` migration status | Remaining reviewer-code gate |
|---|---|---|---|
| Shared utilities and controls | no independent scientific claim | implementation complete; engineering validated | final integration review |
| C-U1 | experiment-specific statuses remain authoritative in `docs/handoff.md` and the registry | implementation complete | registered reproduction remains an internal scientific task |
| D-U1 revision 4 | `not_run` for the active formal matrix | implementation complete | formal run remains an internal scientific task |
| Hopper E7-Q2 | `long_run_validated` for the existing learned-critic external mechanism result | implementation complete | real registered-data reproduction through the new runner |
| D4RL-9 / `EXT-H-E7-BENCH-01` | historical archive is pilot provenance only; no formal ranking | reviewer algorithm, multi-method training, rollout, and aggregation complete | real HDF5/MuJoCo liveness |
| Countdown | paper-aligned scans remain development-pilot evidence; final manuscript protocol/result not frozen | stable sequence/verifier/objective/evaluation core implemented and engineering validated | freeze final protocol/result before adding the experiment entry point |

A smoke, static check, first update, or fixed short trajectory is engineering evidence only. It does not change scientific status.

## 5. Current implementation ownership

### 5.1 C-U1

The live implementation is split by responsibility across `continuous/cu1*.py`, with public dispatch in `cli.py`. It covers source, causal, phase/control, taper, artifacts, aggregation, and audit for the controlled continuous environment.

### 5.2 D-U1 revision 4

The live implementation is split across `categorical/du1_*.py`, with public dispatch in `cli.py`. It covers the revision-4 environment, categorical policy, six frozen controls, shared-start training, metrics, public execution, and reports. Revisions 1/2/3 and the historical reciprocal-quartic method remain excluded.

### 5.3 Hopper E7-Q2 mechanism profile

The Hopper mechanism implementation includes data, models, critic, frozen advantages, actor training, diagnostics, rollout, suite execution, public runner, aggregation, and terminal records. It remains scientifically distinct from the D4RL-9 task-performance backend.

### 5.4 D4RL-9 performance profile

Implemented reviewer-facing code:

- exact nine-task catalog, dataset basenames, environment IDs, reference-score constants, and fail-closed SHA state in `external/d4rl_tasks.py`;
- `CanonicalActor`, `CanonicalCritic`, and the shared `SNA2CIQLVExpRankAgent` lifecycle;
- dynamic TD advantages, expectile critic regression, and actor-then-critic Adam updates;
- reviewer-selectable `exprank`, `positive_only`, `signed`, `global`, `reciprocal_linear`, `reciprocal_quadratic`, and `exponential` actor-side controls;
- ExpRank as the only implicit default;
- explicit `legacy-pilot-v1` acknowledgement before any historical non-ExpRank control may run;
- historical pilot coefficients carried as provenance, with `profile_is_final=false` and `final_method_matrix_frozen=false`;
- one shared trainer across every task, method, and seed, with no per-task trainer copies and no post-hoc per-task method selection;
- locomotion reward normalization, episode-aware returns, action clipping, deterministic minibatch training, and legacy-compatible ExpRank checkpoints;
- public `drpo-reference d4rl` CLI with optional `--methods` and `--method-profile`;
- selected-task or all-nine-task training, method × seed output roots, final checkpoints, and lightweight completion/failure records;
- optional direct Gymnasium/MuJoCo evaluation in `HalfCheetah-v4`, `Hopper-v4`, and `Walker2d-v4`;
- raw return and D4RL normalized score per episode;
- episode mean/std and task × method mean/std across seed means;
- differential and controlled tests for formulas, detached distance, legacy ExpRank compatibility, method-profile gates, execution paths, rollout semantics, and aggregation.

No additional D4RL actor, critic, multi-method training-loop, rollout, or simple aggregation migration is currently required. Real liveness and the internal formal experiment lifecycle are execution/protocol gates rather than missing reviewer code.

### 5.5 Countdown stable core

The approved path `categorical/countdown.py` now owns only the stable Countdown primitives that are independent of the final manuscript-facing experiment protocol:

- canonical expression cleaning;
- exact arithmetic verifier and mutually exclusive verifier categories;
- chat prompt rendering with thinking disabled when supported;
- prompt/completion encoding, EOS inclusion, and prompt-label masking;
- padded completion batches;
- completion-only sequence log-probability, entropy, and bounded direct-logit score;
- weighted sequence log-probability without weight-sum renormalization;
- detached normalized current sequence surprisal `u=-log P(y|x)/2`;
- detached paper-aligned linear-surprisal envelope `alpha * exp(-c*u)`;
- first-occurrence unique-negative bank handling;
- lightweight Greedy, Pass@k, valid-rate, and verifier-category aggregation.

The module imports neither Transformers/PEFT nor the historical one-file trainer. It does not own model loading, GPU resource selection, training scheduling, checkpoint selection, RunSpec execution, formal artifact delivery, or manuscript result binding.

The new path was explicitly approved by the user and the approval is preserved in Draft PR #149 comment `5016309623`. The first stable-core slice passed Python compilation, full pytest, Ruff, handoff authority, formal execution-channel validation, governance inventory, and governance stage checks at `69e60c5bc944b47ab403b6432f1efdb118bc6fa4`.

## 6. Countdown remaining work

### 6.1 What is no longer blocked

Stable expression, verifier, masking, sequence-likelihood, remoteness-weight, bank-deduplication, and response-metric primitives no longer depend on the final coefficient or experiment selection. Their migration is implemented and engineering validated.

### 6.2 What remains blocked

The following are intentionally absent until the manuscript-facing Countdown protocol and result are frozen:

- `paper_code/src/drpo_reference/experiments/countdown.py`;
- a `drpo-reference countdown` CLI command;
- final model scale and initialization;
- final comparison arms and default coefficient;
- formal or confirmatory seeds;
- training horizon and checkpoint-selection rule;
- validation/test access and final evaluation protocol;
- public run manifests tied to the final experiment coordinate;
- manuscript-facing result values or method-ranking claims.

No new Python path for the final experiment entry has been approved. Do not create it by inference from the stable-core approval.

## 7. Items that are not missing stable-core code

The following remain provenance, protocol, resource, execution, or internal review gates:

- D4RL dataset identities, liveness, final formal matrix, seeds, budgets, and manuscript values;
- Countdown final coefficient/profile selection after development-pilot review;
- Countdown final model scale, common comparison matrix, seeds, budget, checkpoint rule, and test protocol;
- real Transformer/GPU liveness for any future public runner;
- full execution, internal terminal scientific review, and final manuscript values.

## 8. Exact next sequence

1. Keep D4RL reviewer code frozen; treat real liveness as a separate execution gate.
2. Review the completed Countdown stable-core diff and its exact-head CI.
3. Wait for the final Countdown manuscript-facing protocol/result freeze before proposing the exact final experiment-entry Python path.
4. After that separate approval, migrate only the selected training/evaluation lifecycle and add the public command.
5. Perform the integration-freshness audit against current `main` before any final merge proposal.

No formal experiment was launched by the Countdown stable-core migration. No method ranking, scientific status, model scale, seed, threshold, training horizon, default coefficient, checkpoint rule, or manuscript value was changed.
