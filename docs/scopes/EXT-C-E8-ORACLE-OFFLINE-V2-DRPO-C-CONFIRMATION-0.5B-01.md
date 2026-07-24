# Scope: EXT-C-E8-ORACLE-OFFLINE-V2-DRPO-C-CONFIRMATION-0.5B-01

## Scientific responsibility

This pilot confirms one DRPO exponential taper coefficient from the frozen set

\[
c\in\{1.897119985,3,5,8\}
\]

using four new paired development seed offsets. Countdown is an external-validity environment. This experiment does not replace controlled causal evidence from C-U1 or D-U1.

## Status

- result status: `not run` / code-first pilot implementation;
- registration state: `dev_code_first_unregistered` until the implementation SHA is frozen and the normal registration transaction is completed;
- test split: forbidden;
- formal cross-method ranking: forbidden;
- convergence or steady-state claim: forbidden.

## Frozen matrix

- model: `Qwen2.5-0.5B-Instruct`;
- initialization: pretrained base plus fresh LoRA for every cell;
- alpha: `1.0` for every cell;
- coefficients: `1.897119985`, `3.0`, `5.0`, `8.0`;
- seed offsets: `17000`, `18000`, `19000`, `20000`;
- total: 4 coefficients x 4 paired seeds = 16 cells;
- horizon: 1200 steps, no early stopping;
- evaluation cadence: Greedy and Pass@8 every 100 steps; Pass@64 every 200 steps;
- primary metric: seed-level mean Pass@8 over steps 800, 900, 1000, 1100, and 1200;
- secondary metric: terminal Pass@8;
- best-checkpoint values: supplementary only.

The five checkpoints within one seed are correlated measurements and must not be counted as five independent samples.

## Selection rule

For each seed and coefficient, first average Pass@8 over the frozen late window. Then aggregate those four seed-level values. The coefficient with the highest mean late-window Pass@8 is the candidate for freezing. All paired seed values, dispersion, terminal values, valid-expression diagnostics, and failures must be reported before the freeze decision.

No candidate may be added, removed, rerun selectively, or retuned after observing the new-seed matrix.

## Historical evidence role

Prior seed offsets `4000`, `5000`, `9000`, `10000`, `11000`, and `12000` motivated the candidate set but are excluded from the confirmation aggregate. In particular, seed `11000` remains a valid low-performance trajectory; it may not be discarded absent a registered execution or numerical failure.

## Allowed changes

- add the frozen profile to the existing paper-aligned E8 profile registry;
- add one YAML config, this scope, one protocol, one RunSpec, and tests in the existing test file;
- reuse the existing trainer, runtime, launcher, one-click script, bank, and validation split.

## Forbidden changes

- any loss, optimizer, learning-rate, bank, model, LoRA, horizon, evaluation, or denominator change;
- AsymRE, TOPR, Positive-only, or Global reruns;
- test-split access;
- post-hoc seed exclusion or candidate substitution;
- claims of significance, convergence, steady state, or universal method ranking;
- conflating task-performance degradation, valid-structure events, and NaN/Inf numerical failure.

## Required evidence

- liveness gate before the full matrix; liveness is not scientific evidence;
- exact 16-cell plan and identity-checked resume;
- raw per-checkpoint metrics and per-cell summaries;
- terminal audit with task performance, structure/validity, and NaN/Inf reported separately;
- durable result delivery bound to the implementation commit;
- explicit closure before any paper-facing parameter freeze.
