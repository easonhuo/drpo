# E8 Multitask Lambda Wide Right-Tail 03 — Cold-Start Execution Binding

Experiment ID: `EXT-C-E8-MULTITASK-EXP-LAMBDA-WIDE-RIGHTTAIL-03`

RunSpec ID: `E8_MULTITASK_LAMBDA_WIDE_RIGHTTAIL_03_20260904_01`.

Status: `not_run`.

Execution class: `pilot` only.

This document is an execution-binding addendum to `docs/experiments/E8_MULTITASK_LAMBDA_WIDE_RIGHTTAIL_03_PROTOCOL.md`. It does not modify the frozen scientific matrix, lambdas, seeds, data, optimizer, training horizon, evaluation contract, or claim boundary.

## Post-merge execution authority

PR #343 merged the reviewed protocol, config, and READY RunSpec to `main` at `26c6b6831ff8cae8c138d2f617d57ef683d376ce`.

The canonical offline execution branch for this pilot is:

`dev/e8-multitask-exp-coldstart-01`

The previous task-development branch `dev/e8-multitask-lambda-wide-righttail-03` is no longer the execution target after main integration. Any earlier launch text that names that task-development branch is superseded by this addendum for post-merge execution.

Before launch, `dev/e8-multitask-exp-coldstart-01` must be fast-forward synchronized to the final reviewed `main` integration containing this addendum and the updated READY RunSpec. The executor must run from a clean checkout of that exact cold-start branch head. The bootstrap/run manifests must record the exact resolved source commit and target ref.

The preferred launch path is the READY RunSpec:

```bash
python3 scripts/agent/run_lane.py \
  --repo-root . \
  --lane e8 \
  --run-id E8_MULTITASK_LAMBDA_WIDE_RIGHTTAIL_03_20260904_01 \
  --once
```

The RunSpec must set:

- `E8_COLDSTART_TARGET_REF=refs/heads/dev/e8-multitask-exp-coldstart-01`;
- `E8_COLDSTART_RUN_CLASS=pilot`;
- `E8_COLDSTART_REQUIRE_ORIGIN_MAIN=0`;
- `E8_COLDSTART_CONFIG=configs/e8_multitask_exp_lambda_wide_righttail_03.yaml`;
- `E8_COLDSTART_RUN_ID=E8_MULTITASK_LAMBDA_WIDE_RIGHTTAIL_03_20260904_01`.

The existing bootstrap remains fail-closed: a full-mode launch is rejected if the selected source checkout HEAD does not equal the authoritative cold-start target-ref commit. No source switching or scientific-parameter mutation is authorized to bypass that check.

## Scientific boundary unchanged

The workload remains exactly 100 new Exp cells: 20 each for Word Sorting, Mini Sudoku, Maze, Knights & Knaves, and WikiSQL, with seed 4000 and 1200 optimizer updates. Countdown, Spiral Matrix, Word Ladder, and Graph Coloring receive zero new scientific cells. The test partition remains unopened.

This is still pilot response-shape / curve-boundary evidence only. It does not authorize convergence, significance, steady-state, asymptotic, or universal method-ranking claims. Task-performance degradation, valid/structure-boundary events, and NaN/Inf numerical failures must remain separate report dimensions.
