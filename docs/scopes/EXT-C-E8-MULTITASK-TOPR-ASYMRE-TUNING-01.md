# Scope contract: EXT-C-E8-MULTITASK-TOPR-ASYMRE-TUNING-01

## Approved new Python path

The repository owner explicitly approved adding exactly one new Python file:

```text
src/drpo/e8_multitask_baseline_tuning.py
```

Its sole responsibility is to execute the eight-new-task development tuning pilot for joint fitted-reference beta-TOPR and AsymRE while inheriting the exact split, leakage-safe train-only references, model/training contract, validation evaluator, and terminal statistics from `EXT-C-E8-MULTITASK-EXP-TUNING-01`.

No shared-core Python file and no additional Python path are approved.

## Frozen scientific matrix

- Tasks: the exact eight P0 tasks; Countdown is not rerun.
- TOPR beta: `0, 0.04, 0.08, 0.25, 0.5`.
- AsymRE delta_v: `-1.0, -0.9, -0.7, -0.5, 0.0`.
- One tuning seed inherited from the Exp development contract.
- Exact total: `8 tasks x (5 + 5) = 80` cells.
- Execution topology: 16 cells per wave, exactly five waves.

## Method identity boundaries

The TOPR arm is the existing **joint fitted-reference beta-TOPR variant**. It is not a canonical frozen-behavior TOPR reproduction. `beta=0` is an uncontrolled-negative ratio boundary and cannot be selected as an active TOPR parameter.

For AsymRE, `delta_v=-1` has zero negative-repulsion coefficient. It is a zero-negative boundary and cannot be reported as an active AsymRE victory.

## Required shared contract

The runner must fail closed unless it receives the exact Exp tuning artifacts showing:

- deterministic `5000/500/500` train/validation/test split for every task;
- test access disabled;
- train-only 100-update positive reference adapters;
- zero validation and test rows seen by reference preparation;
- exact prompt-hash and adapter identity matches;
- the same optimizer, learning rate, update horizon, evaluation cadence, generation settings, late window, and validation-selection contract used by Exp.

## Exclusions

This scope must not:

- rerun Countdown or Positive-only;
- add Exp, Linear, Quadratic, Global, Hybrid, SBRC, DPO, or any other method;
- use current near/far selection or any remoteness taper in AsymRE;
- describe fitted-reference beta-TOPR as canonical TOPR;
- let the TOPR reference adapter receive ratio gradients;
- access test partitions during tuning, aggregation, or selection;
- change the declared parameter grids after observing results;
- claim convergence, significance, cross-task superiority, or categorical causal identification;
- register, launch, merge, or upgrade scientific status automatically.

Task performance, validity/structure diagnostics, and NaN/Inf numerical failures remain separate reports. D-U1 remains the categorical causal-identification authority.
