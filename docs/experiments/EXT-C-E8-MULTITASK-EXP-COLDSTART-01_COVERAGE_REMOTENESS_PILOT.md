# EXT-C-E8-MULTITASK-EXP-COLDSTART-01 coverage-remoteness pilot variant

## Status and scope

This document registers a pilot protocol variant of `EXT-C-E8-MULTITASK-EXP-COLDSTART-01` before execution. It changes only the non-Countdown transfer-task negative-bank selection rule. It does not alter the scientific training kernel, optimizer, model, data sizes, task splits, seeds, lambda grids, evaluation budgets, scheduler, recovery semantics, or terminal reporting rules of the current cold-start configuration.

The current/running cold-start result remains a separate config-hash/run identity and is not overwritten or reused as evidence for this variant.

## Claim tested

The pilot asks whether preserving task-specific error-direction coverage while also spanning frozen-reference policy remoteness produces a more informative 16-negative bank than globally spreading 16 candidates only by reference surprisal.

No method-superiority result is assumed. Coverage is treated as a data-selection mechanism hypothesis, not as a sufficient condition for negative feedback to outperform Positive-only.

## Frozen inputs and training

- Base implementation: latest cold-start orchestration in `src/drpo/e8_multitask_exp_tuning.py`.
- Countdown: unchanged old paper bank, paper worker, seeds, coefficients, and evaluation.
- Transfer tasks: unchanged P0 task generators, mutation candidates, verifiers, prompt splits, model/runtime overrides, 20 task-local Exp coefficients, four Positive-only seed streams, and single Exp response-shape seed.
- Cell matrix: unchanged 208 cells.
- Training: unchanged 1200 optimizer updates, micro-batch 1, gradient accumulation 8, AdamW/paper trainer settings, evaluation every 100 updates, no early stopping.
- Scientific kernel: unchanged import-only Countdown paper kernel; no loss reimplementation.
- Training remoteness: unchanged current-policy sequence surprisal recomputed on every update. Static reference ranks never enter the training weight.
- Test split: sealed and unavailable for selection.

## Negative-bank selection change

The source candidate universe remains `all_deterministic_verified_wrong_mutations`, with canonical deduplication and verifier rejection exactly as in the current cold-start code.

For each transfer-task prompt:

1. Reconstruct all deterministic unique verifier-wrong mutation candidates.
2. Score every candidate once under the unchanged zero-update base plus fresh-LoRA reference policy using mean completion-token surprisal.
3. Group candidates by the existing task adapter `error_class`.
4. Allocate exactly 16 slots across error classes using the July-29 P0 coverage-first round-robin rule: sorted class names receive one slot per round; an exhausted class leaves the rotation and the remaining budget is redistributed among non-empty classes.
5. Within each error class, sort candidates by reference surprisal with the existing stable-hash tie break.
6. For a class quota of two or more, choose evenly spaced within-class ranks including both class-local surprisal extremes. For a quota of one, choose the deterministic middle rank because one slot cannot represent both extremes.
7. Interleave the selected class-local queues using the same coverage-first round-robin order, producing exactly 16 unique negatives.

Thus the two bank-selection dimensions have separate responsibilities:

- `error_class` controls directional/error-type coverage;
- frozen-reference surprisal controls near-to-far coverage within each available error direction.

The frozen-reference score is selection/provenance only. The paper trainer continues to compute the actual taper from current-policy surprisal during training.

## Diagnostics

For every prompt the derived-bank audit must retain the existing overall candidate/selected surprisal summaries and additionally record:

- candidate counts by error class;
- selected counts by error class;
- selected distinct error-class count;
- class-local candidate and selected surprisal summaries;
- class-local selected ranks and global reference ranks.

No new numerical coverage threshold is introduced. Narrow within-class support is reported rather than silently rejected.

## Interpretation boundary

This pilot can test whether combining error-direction coverage with policy-remoteness coverage changes transfer-task response curves. It cannot by itself establish that coverage is sufficient for useful negative feedback, that Exp is universally superior, or that fixed-horizon task-local snapshots constitute a final method ranking. The final 208-cell aggregate remains the reporting authority for this run identity, and task-performance collapse, support/format boundary behavior, and NaN/Inf numerical failure remain separately reported.
