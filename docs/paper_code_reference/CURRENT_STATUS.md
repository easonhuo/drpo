# PAPER-CODE-REFERENCE-01 Current Status

**Document role:** canonical task-local current snapshot and continuation index.  
**Not a research master:** `docs/handoff.md` remains the unique research source of truth.  
**Claim:** `PAPER-CODE-REFERENCE-01`  
**Scientific-status impact:** none.  
**Last engineering-validated head:** `d234f0f589134bb959a98bfafa50cb47784ff04c`.

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
- current `main`: `dd46727c1efefd2e6d4cdf6f3b204ec1fc58fca3`;
- task base and current merge base: `4544005bd7df69c53bad70a9dcac846af01285e4`;
- only active development branch: `dev/paper-code-reference-01`;
- development head before this document: `d234f0f589134bb959a98bfafa50cb47784ff04c`;
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

It is **not** required to duplicate the repository's internal scientific-governance platform. Registry or handoff mutation, formal-evidence eligibility, full matrix completeness governance, selected-versus-terminal checkpoint adjudication, manuscript artifact binding, and scientific result promotion remain internal responsibilities.

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
| Countdown | paper-aligned scans remain development-pilot evidence; final manuscript protocol/result not frozen | stable sequence, frozen-bank, objective, first-update, and response-metric core implemented | freeze final protocol/result before adding the experiment entry point |

A smoke, static check, first update, or fixed short trajectory is engineering evidence only. It does not change scientific status.

## 5. Current implementation ownership

### 5.1 C-U1

The live implementation is split by responsibility across `continuous/cu1*.py`, with public dispatch in `cli.py`. It covers source, causal, phase/control, taper, artifacts, aggregation, and audit for the controlled continuous environment.

### 5.2 D-U1 revision 4

The live implementation is split across `categorical/du1_*.py`, with public dispatch in `cli.py`. It covers the revision-4 environment, categorical policy, six frozen controls, shared-start training, metrics, public execution, and reports. Revisions 1/2/3 and the historical reciprocal-quartic method remain excluded.

### 5.3 Hopper E7-Q2 mechanism profile

The Hopper mechanism implementation includes data, models, critic, frozen advantages, actor training, diagnostics, rollout, suite execution, public runner, aggregation, and terminal records. It remains scientifically distinct from the D4RL-9 task-performance backend.

### 5.4 D4RL-9 performance profile

Implemented reviewer-facing code includes the exact nine-task catalog, one canonical SNA2C-IQLV actor/critic lifecycle, reviewer-selectable historical controls behind an explicit pilot profile, selected-task or all-nine-task training, direct Gymnasium/MuJoCo rollout evaluation, checkpoints, lightweight completion/failure records, and task × method × seed aggregation.

No additional D4RL actor, critic, multi-method training-loop, rollout, or simple aggregation migration is currently required. Real liveness and the internal formal experiment lifecycle are execution/protocol gates rather than missing reviewer code.

### 5.5 Countdown stable training core

The approved path `categorical/countdown.py` owns the Countdown primitives that are independent of the final manuscript-facing experiment coordinate:

- canonical expression cleaning, exact arithmetic verification, and mutually exclusive verifier categories;
- chat prompt rendering with thinking disabled when supported;
- prompt/completion encoding, EOS inclusion, prompt-label masking, and padding;
- completion-only sequence log-probability, entropy, and bounded direct-logit score;
- first-occurrence unique-negative bank encoding and flattened batching;
- explicit per-prompt unique-negative counts and raw-bank counts;
- weighted sequence log-probability without weight-sum renormalization;
- detached normalized current sequence surprisal `u=-log P(y|x)/2`;
- detached paper-aligned linear-surprisal envelope `alpha * exp(-c*u)`;
- exact joint objective `-(mean positive log-probability - mean weighted negative log-probability)`;
- Positive-only execution that skips the negative-bank model forward;
- stable bank/weight diagnostics and parameter-update norm measurement;
- lightweight Greedy, Pass@k, valid-rate, and verifier-category aggregation.

Controlled tests cover frozen-bank collation, the unique-negative denominator, the exact objective, Positive-only bank skipping, diagnostics, and equality of the first clipped AdamW update against the authoritative legacy formula.

The module imports neither Transformers/PEFT nor the historical one-file trainer. It does not own model loading, GPU resource selection, optimizer/scheduler selection, multi-step training scheduling, checkpoint selection, RunSpec execution, formal artifact delivery, or manuscript result binding.

The new path was explicitly approved by the user and the approval is preserved in Draft PR #149 comment `5016309623`. That approval does not extend to a final experiment-entry path.

Exact-head validation at `d234f0f589134bb959a98bfafa50cb47784ff04c` passed Evidence Locator, Python compile, shell syntax, handoff authority, formal execution channel, governance inventory, governance stage, full pytest, and Ruff.

## 6. Countdown remaining work

### 6.1 What is no longer blocked

Stable expression, verifier, masking, sequence-likelihood, frozen-bank batching, remoteness weighting, exact objective, first optimizer-update semantics, diagnostics, and response metrics no longer depend on the final coefficient or experiment selection. Their migration is implemented and engineering validated.

### 6.2 What remains blocked

The following are intentionally absent until the manuscript-facing Countdown protocol and result are frozen:

- `paper_code/src/drpo_reference/experiments/countdown.py`;
- a `drpo-reference countdown` CLI command;
- model and LoRA loading;
- optimizer/scheduler and full multi-step training-loop selection;
- checkpoint persistence, resume, selection, and terminal evaluation;
- final model scale and initialization;
- final comparison arms and default coefficient;
- formal or confirmatory seeds;
- training horizon and checkpoint-selection rule;
- validation/test access and final evaluation protocol;
- public run manifests tied to the final experiment coordinate;
- manuscript-facing result values or method-ranking claims.

No new Python path for the final experiment entry has been approved. Do not create it by inference from the stable-core approval.

## 7. Items that are not missing stable-training-core code

The following remain provenance, protocol, resource, execution, or internal review gates:

- D4RL dataset identities, liveness, final formal matrix, seeds, budgets, and manuscript values;
- Countdown final coefficient/profile selection after development-pilot review;
- Countdown final model scale, common comparison matrix, seeds, budget, checkpoint rule, and test protocol;
- real Transformer/GPU liveness for any future public runner;
- full execution, internal terminal scientific review, and final manuscript values.

## 8. Exact next sequence

1. Keep D4RL reviewer code frozen; treat real liveness as a separate execution gate.
2. Treat the Countdown stable training core as the reusable implementation boundary below the final runner.
3. Wait for the final Countdown manuscript-facing protocol/result freeze before proposing the exact final experiment-entry Python path.
4. After separate path approval, migrate only the selected model-loading, optimizer/scheduler, multi-step/checkpoint, and evaluation lifecycle and add the public command.
5. Perform the integration-freshness audit against current `main` before any final merge proposal.

No formal experiment was launched by the Countdown stable-training-core migration. No method ranking, scientific status, model scale, seed, threshold, training horizon, default coefficient, checkpoint rule, or manuscript value was changed.
