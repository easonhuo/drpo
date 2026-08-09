# E8 multitask warm-started pilot result closure — 2026-08-09

This record preserves three linked E8 multitask development pilots and records a post-run initialization-protocol correction. It does not delete or rewrite the delivered results.

## 1. P0 implemented-gradient diagnostic

- Experiment: `EXT-C-E8-MULTITASK-P0-01`
- Run: `E8_MULTITASK_P0_FIG1_20260729_01`
- Durable result: `easonhuo/drpo-results@a5541e6487d74988f07a381a837972bdc5d3282c`
- Eight new tasks, 256 diagnostic points per task, 2,048 total points.
- Task-equal relative implemented actor-gradient curve from lowest to highest surprisal bin:
  `1.000, 1.111, 1.239, 1.554, 2.028, 2.389, 3.100, 2.949, 3.226, 3.638`.
- Highest-bin 95% bootstrap CI: `[3.256, 4.095]`.

The diagnostic intentionally used a task-specific Positive-only LoRA prepared for exactly 100 optimizer updates before the full-parameter gradient probe. That preparation created a nontrivial reference policy for the gradient diagnostic. Therefore the curve is a diagnostic **at the 100-step Positive-only reference policy**, not at untouched Qwen initialization. This is a scope condition of P0, not by itself a defect in the diagnostic.

## 2. Exp multitask tuning

- Experiment: `EXT-C-E8-MULTITASK-EXP-TUNING-01`
- Run: `E8_MULTITASK_EXP_TUNING_20260729_01`
- Durable result: `easonhuo/drpo-results@26cf09102e4eaea9b58844fac158dc3cfc33314d`
- Completed: `72/72` cells; NaN/Inf events `0`; separate test partition unused.
- One tuning seed; fixed 1,200-update horizon, which is not convergence.

| Task | selected rho | late-window Pass@8 | Positive-only late-window Pass@8 | boundary note |
|---|---:|---:|---:|---|
| Countdown | 0.35 | 0.184375 | 0.176563 | closed in tested grid |
| Word Sorting | 0.25 | 0.734375 | 0.728125 | closed in tested grid |
| Spiral Matrix | 0.90 | 1.000000 | 1.000000 | performance ceiling |
| Mini Sudoku | 0.125 | 0.979688 | 0.982813 | all tested Exp below PO; strong-taper boundary unclosed |
| Maze | 0.75 | 0.742188 | 0.735938 | closed in tested grid |
| Word Ladder | 0.25 | 0.178125 | 0.168750 | closed in tested grid |
| Knights & Knaves | 0.35 | 0.693750 | 0.679688 | closed in tested grid |
| Graph Color | 0.25 | 1.000000 | 0.996875 | performance ceiling |
| WikiSQL | 0.125 | 0.837500 | 0.817188 | strong-taper boundary unclosed |

Post-run review found that method training inherited the P0 `train_only_task_positive_warmstart_100` reference. That warm start was introduced for the gradient diagnostic; carrying it into downstream method training was not required by the historical Countdown cold-start protocol. The Exp response remains descriptive **conditional on this warmstarted initialization**, but it is not a fresh-LoRA/cold-start reproduction.

## 3. Fitted-reference TOPR / AsymRE multitask tuning

- Experiment: `EXT-C-E8-MULTITASK-TOPR-ASYMRE-TUNING-01`
- Run: `E8_MULTITASK_TOPR_ASYMRE_TUNING_20260731_01`
- Durable result: `easonhuo/drpo-results@6479320012da4a83120d0700a3df71525cc28aef`
- Current durable delivery: **71/80 cells**, waves 1--4 partial; no final complete aggregate is claimed.
- Eight new tasks only; Countdown was not rerun in this matrix.
- TOPR grid: `beta={0,0.04,0.08,0.25,0.5}`; `beta=0` is the uncontrolled-negative ratio boundary.
- AsymRE grid: `delta_v={-1,-0.9,-0.7,-0.5,0}`; `delta_v=-1` is the zero-negative boundary.

These cells explicitly inherited `train_only_task_positive_warmstart_100`. This is a **warm-start carryover protocol error for method-training comparison**: the initialization choice from the P0 diagnostic was reused as if it were a neutral training default. The delivered per-cell trends are retained as warmstarted development-pilot evidence, but they are not protocol-matched cold-start reproductions of the historical Countdown TOPR/AsymRE experiments.

## 4. Replacement interpretation

The old Countdown DRPO/Positive-only/AsymRE/TOPR comparison used pretrained Qwen plus a fresh LoRA per method cell; the later multitask method-training runs above used a 100-step Positive-only reference policy first. Therefore old Countdown and new multitask method curves must not be combined into a strict apples-to-apples nine-task method comparison.

A future cold-start rerun is required before these multitask training curves become the authoritative protocol-matched method comparison. Until then:

- P0 remains a valid warmstarted-reference implemented-gradient occurrence diagnostic;
- Exp remains a complete but warmstarted single-seed development tuning curve;
- TOPR/AsymRE remain partial warmstarted development-pilot evidence;
- no convergence, significance, universal method ranking, or categorical causal-identification claim is authorized;
- D-U1 remains the categorical causal-identification environment;
- task performance, valid/structure behavior, and NaN/Inf numerical failure remain separate.

## Durable correction notes in `drpo-results`

- P0 note commit: `c4c23860ed9f52732e96e7b6571a87bd53ea3e69`
- Exp note commit: `30e3388571e54c63cdcf8fd7062cb866c8407a6a`
- TOPR/AsymRE note commit: `7a13fcfe648d9b31bb09e122076183280511944f`
