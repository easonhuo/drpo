# E8 AsymRE Config-Driven ReplayAB Arm B

## Status

- claim: `GOV-DEV-WORKFLOW-OPTIMIZATION-BENCHMARK-01`
- case: `E8-ASYMRE-CONFIG-DRIVEN-REPLAYAB-01`
- arm: `B`
- scientific evidence: `false`
- full sweep authorization: `false`
- terminal boundary for this replay: repository plan-ready with exact-head gates

## Purpose

Replay the historical PR #229 E8 AsymRE boundary-dense preparation task through
the config-driven runtime introduced by PR #250. The benchmark asks whether the
same launch semantics can be prepared without changing experiment Python,
registering a concrete grid in `_PROFILES`, or adding a parameter-specific test.

## Frozen semantic target

- model: `Qwen2.5-0.5B-Instruct`
- initialization: pretrained base plus fresh LoRA per cell
- objective: `(1-delta_v) * positive_lp - (1+delta_v) * negative_lp`
- distance/remoteness control: none
- delta_v: `[-1.0, -0.95, -0.9, -0.85, -0.8, -0.7, -0.6, -0.5]`
- paired seed offsets: `[4000, 5000]`
- exact matrix: `8 x 2 = 16` cells
- training horizon: fixed 1200 steps, no early stopping
- held-out evaluation: structurally disjoint `val.jsonl`, excluded from training loss
- reporting: late-window and terminal Pass@8; best checkpoint supplementary only
- task-performance, structure/support boundary, and NaN/Inf failure remain separate

## Arm-B scope contract

The routine task may add only this benchmark's reviewed config, scope, and
RunSpec. It must not modify any `.py` file, hard-coded parameter tuple,
`_PROFILES`, expected-count constant, parameter-specific regression test,
launcher, trainer, registry, handoff, or completed result.

The implementation identity is the frozen PR #250 head
`c4ec718ea426f91a4b616a83106706ced8b8e028`. A later PR #250 repair changes the
toolchain and invalidates this arm until its RunSpec is rebased deliberately.

## ReplayAB measurement

Correctness is evaluated before efficiency. The normalized launch semantics must
match historical arm A. Process evidence records commits, changed paths, exact-head
CI, repairs, candidate failures, wall-clock timestamps, and any temporary workflow.
Unavailable historical active-time or token fields remain `UNAVAILABLE`.

No CUDA liveness or 16-cell scientific sweep is performed by this repository-only
ReplayAB arm. Those remain separate execution gates and are not counted as passed.
