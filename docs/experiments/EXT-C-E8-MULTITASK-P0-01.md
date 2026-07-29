# EXT-C-E8-MULTITASK-P0-01

## Status and role

- Status: **implemented candidate / not run**
- Role: external discrete-task occurrence and implemented-gradient diagnostic
- Formal registration: deferred until the implementation SHA is frozen and reviewed
- Causal-identification authority: unchanged; categorical causal identification remains D-U1

Unit tests, source acquisition, bank qualification, fixed-step warm-start liveness, and
integration smoke tests are not scientific results. P0 trains only one fixed 100-update
oracle-positive reference adapter per task. It does not train negative-update methods or rank
methods.

## Claim

Test whether policy-relative remoteness and the implemented negative actor-gradient
diagnostic can be measured under one frozen protocol across distinct discrete output
structures. The protocol permits growth followed by saturation: categorical selected-logit
scores are bounded, so unbounded monotone gradient explosion is not a pass criterion.

## Tasks

The frozen candidate suite contains eight new tasks:

1. Reasoning Gym word sorting;
2. Reasoning Gym spiral matrix;
3. Reasoning Gym Mini Sudoku;
4. Reasoning Gym maze shortest-path length;
5. Reasoning Gym word ladder;
6. Reasoning Gym knights and knaves;
7. Reasoning Gym graph coloring;
8. WikiSQL official JSON logical forms.

Countdown is not rerun or included in the task-equal aggregate. Its existing SFT/reference-policy
full-bank diagnostic remains separate external evidence under its archived protocol. The
`CountdownAdapter` is retained only for backward-compatible engineering checks and is not part of
the default task list.

Reasoning Gym is pinned to
`49b07130b3fcd12f2d064bba7c43869543a0e7e7`. WikiSQL is pinned to
`7857cfd8aefcc9823245c370f9e39ecd55745ea6`; the pipeline records the bundled
`data.tar.bz2` SHA-256 after acquisition.

No task is silently replaced, simplified, or skipped. In particular, the official Maze task
returns one scalar shortest-path length. Format-valid wrong integers are classified by the sign of
their oracle-relative residual as `path_length_underestimate` or
`path_length_overestimate`; the signed and absolute residuals are model-independent verifier
diagnostics, not learner-relative near/far labels. One-sided scalar banks still fail closed.

## Frozen stage contract

```text
prepare -> qualify -> warmstart -> diagnose -> aggregate
```

### `prepare`

- acquires the two exact public-source commits;
- extracts the pinned WikiSQL archive safely;
- generates task instances;
- verifies every oracle;
- constructs 16 unique verifier-wrong negatives per retained prompt;
- writes model-independent banks and build audits.

The bank may contain oracle identity, mutations, verifier score, binary correctness,
format validity, error class, response length, and edit distance. It must not contain
near/far labels, current-policy probabilities, surprisal, distance, or taper weights.

### `qualify`

The task gate checks:

- exact requested row count;
- oracle verification rate;
- per-prompt negative count;
- per-prompt and per-task error-class diversity;
- canonical uniqueness;
- format-valid fraction;
- completion-length boundary;
- absence of verifier-correct negatives.

Qualification is fail closed. The audit preserves every failed prompt and gate.

### `warmstart`

- starts each task from the same supplied base model with a fresh LoRA adapter;
- uses only that task's verified oracle-positive completions;
- executes exactly 100 optimizer updates with effective batch size 64;
- uses LoRA rank 32, alpha 64, dropout 0.05, AdamW at `2e-4`, cosine decay, 5% warmup,
  and gradient clipping at 1.0;
- disables early stopping and checkpoint selection;
- audits oracle NLL on a deterministic fixed subset before and after the 100 updates;
- fails closed on non-finite loss, gradient, parameters, incomplete updates, or failure of the
  predeclared NLL-direction audit;
- writes one terminal task adapter and exact identity manifest.

The checkpoint is selected only by the fixed update count. Negative-gradient correlation,
surprisal bins, downstream method performance, and visual curve quality cannot select or reject a
checkpoint. The warm start is reference-policy preparation, not evidence that a negative-update
method is effective.

### `diagnose`

For a deterministic prompt-balanced point sample, each task's frozen 100-update reference policy
records:

- task, prompt ID, negative ID, seed, and checkpoint kind;
- mean-token surprisal;
- mean direct-logit score norm;
- matched absolute negative advantage;
- raw full-parameter gradient norm;
- implemented actor-gradient norm;
- verifier score, error class, and response length.

The diagnostic loads the task-specific terminal LoRA adapter, merges it into a fresh copy of the
shared base model, and then computes the full-parameter diagnostic. The frozen bank is not
relabelled near/far; remoteness is computed only at this stage. Every point is flushed
incrementally, and resume requires exact bank/config/model identity.

### `aggregate`

- rank and bin points within each task by current mean-token surprisal;
- use ten task-internal quantile bins;
- normalize each curve by that task's lowest-surprisal bin;
- aggregate tasks with equal weight;
- obtain 95% intervals by prompt-level bootstrap within task;
- preserve per-task panels and the task-equal aggregate separately.

Pooling raw points across tasks is forbidden because output length and task frequency would
otherwise dominate the curve.

## Entry points

Run the full protocol:

```bash
scripts/run_e8_multitask_p0.sh \
  --work-dir /absolute/path/to/e8_multitask_p0 \
  all \
  --model-path /absolute/path/to/Qwen2.5-0.5B-Instruct
```

Run stages separately:

```bash
scripts/run_e8_multitask_p0.sh \
  --work-dir /absolute/path/to/e8_multitask_p0 \
  prepare

scripts/run_e8_multitask_p0.sh \
  --work-dir /absolute/path/to/e8_multitask_p0 \
  qualify

scripts/run_e8_multitask_p0.sh \
  --work-dir /absolute/path/to/e8_multitask_p0 \
  warmstart \
  --model-path /absolute/path/to/Qwen2.5-0.5B-Instruct

scripts/run_e8_multitask_p0.sh \
  --work-dir /absolute/path/to/e8_multitask_p0 \
  diagnose \
  --model-path /absolute/path/to/Qwen2.5-0.5B-Instruct \
  --tasks word_sorting spiral_matrix

scripts/run_e8_multitask_p0.sh \
  --work-dir /absolute/path/to/e8_multitask_p0 \
  aggregate \
  --tasks word_sorting spiral_matrix
```

Run a non-scientific source/API smoke:

```bash
scripts/run_e8_multitask_p0.sh \
  --work-dir /absolute/path/to/e8_multitask_p0_smoke \
  --smoke-rows 2 \
  --smoke-negatives 4 \
  prepare
```

The smoke override changes the data contract and may not be reported as P0 evidence.

## Expected outputs

```text
source_manifest.json
bank_manifest.json
qualification_audit.json
banks/<task>.jsonl
banks/<task>.build_audit.json
warmstart/warmstart_manifest.json
warmstart/<task>/task_manifest.json
warmstart/<task>/adapter/
warmstart/<task>/training_metrics.jsonl
diagnostics/diagnostic_identity.json
diagnostics/<task>.jsonl
diagnostics/diagnostic_manifest.json
aggregate/per_task_bins.csv
aggregate/task_equal_aggregate.csv
aggregate/aggregate_summary.json
```

Large raw banks and diagnostic points belong in the result/artifact channel, not the Git
source repository.

## Explicit non-claims

This stage does not establish:

- causal transmission from far-field gradients to drift or collapse;
- method effectiveness or ranking;
- Countdown replication under the new eight-task protocol;
- convergence or steady state;
- cross-model-family generalization;
- unbounded categorical score growth;
- any formal result before implementation review, registration, exact-head execution,
  terminal audit, packaging, and durable delivery.

After the implementation commit is frozen, formal/pilot activation must use the existing
schema-v3 delta and code-first registration route. `docs/handoff.md` and
`experiments/registry.yaml` are not edited directly by this code-only implementation PR.
