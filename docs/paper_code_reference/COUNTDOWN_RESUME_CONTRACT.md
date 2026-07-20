# PAPER-CODE-VALIDATION-01 Countdown interrupted-run resume contract

## 1. Identity and boundary

- Claim: `PAPER-CODE-VALIDATION-01`
- Parent claim: `PAPER-CODE-REFERENCE-01`
- Component: reviewer-facing Countdown runtime
- Scientific status impact: none
- Formal experiment authorization: none
- Method-ranking authorization: none
- New scientific variables: none
- New Python paths: none

This contract closes the engineering gap recorded as
`interrupted_optimizer_scheduler_resume_not_implemented`. It does not run Qwen,
PEFT, CUDA, a pilot, or a formal experiment. It does not alter the registered
active-tail coordinate, model, adapter, methods, coefficients, seeds, update
budget, checkpoint cadence, validation/test protocol, or result interpretation.

Only these existing files may change in the implementation slice:

- `paper_code/src/drpo_reference/experiments/countdown.py`;
- `paper_code/src/drpo_reference/cli.py`;
- `paper_code/tests/test_common.py`;
- `paper_code/tests/test_cli.py`;
- task-local paper-code documentation and validation workflow test inventory if
  required to execute the new existing-file tests.

## 2. User-facing command

Fresh execution remains unchanged:

```text
drpo-reference countdown --config <explicit-json> --output <new-or-empty-dir>
```

Explicit resume adds one flag:

```text
drpo-reference countdown --config <same-explicit-json> \
  --output <existing-run-dir> --resume
```

Rules:

- without `--resume`, the output must remain new or empty;
- with `--resume`, the output must already exist and contain a compatible
  `RUN_MANIFEST.json` plus durable per-method resume state;
- `--resume` never infers or changes a config value;
- a completed suite returns its preserved completion and summary rather than
  retraining;
- a missing, partial, stale, or incompatible resume state fails closed.

## 3. Durable boundary

Resume is supported only at a completed optimizer-step boundary. A committed
resume checkpoint contains:

1. trainable adapter and tokenizer files;
2. optimizer state;
3. scheduler state;
4. Python, NumPy, Torch CPU, and available CUDA RNG states;
5. completed optimizer step and deterministic sampler-plan offset;
6. initial trainable-state digest;
7. method, seed, coefficient, shared negative scale, tau, and surprisal scale;
8. best step/value, last-finite step, and accumulated metric rows;
9. config/input/model/reference-adapter identity digests;
10. schema and runner versions plus `formal_result_claim=false`.

The state is written to a temporary sibling and atomically promoted only after
all required files are complete. A stale temporary directory is ignored and
removed on resume. The committed state is never assembled from partial files.

The first durable state is written after step-0 validation and before training.
Later states are written at the existing `checkpoint_every` cadence and at the
final finite step. An interruption after a durable state may replay work after
that state, but it must restore RNG, sampler offset, optimizer, and scheduler so
that the resumed trajectory starts from the exact committed boundary.

No claim of bit-identical GPU kernels is made. The engineering invariant is
exact restored state and coordinate, not a new scientific reproducibility
claim.

## 4. Identity checks

Before any resume-side model loading, training, or test access, the runtime must
recompute and compare:

- config SHA-256 and parsed explicit coordinate;
- replay, calibration, validation, and structure-reference SHA-256 values;
- model identity;
- reference-adapter identity and file hashes;
- experiment ID, protocol ID, runner-compatible resume schema, seed, and method;
- calibration identity and initial trainable-state digest;
- expected sampler offset
  `completed_step * grad_accum * micro_batch`;
- optimizer and scheduler step compatibility with the requested total budget.

A mismatch produces a dedicated fail-closed resume error. The runtime must not
silently delete output, restart from scratch, accept a different coordinate, or
reuse a state from another method or seed.

A state created before this resume schema exists is not resumable. Existing
historical reviewer outputs remain readable evidence but are never upgraded by
inference.

## 5. Failure semantics

Resume is for process interruption, not for reclassifying a completed failure.

- `nan_inf_numerical_failure` remains a numerical failure and is not silently
  resumed;
- task-performance collapse remains separate from numerical or support events;
- support/probability boundary remains separate;
- invalid input/model/environment remains separate;
- a method with an explicit terminal `RUN_FAILED.json` is not automatically
  resumed;
- a method with a valid durable state but no terminal completion record is
  treated as interrupted/incomplete and may resume;
- stale temporary state is not considered a committed checkpoint;
- the original failure and partial files remain preserved.

## 6. Paired-method and test-access semantics

The existing paired coordinate remains unchanged:

- the per-seed calibration record is reused only after exact validation;
- every method must still originate from the calibrated initial trainable-state
  digest;
- completed method summaries may be reused after validation;
- incomplete methods continue from their own method/seed state;
- test input is neither read nor hashed until every requested method/seed has
  completed training;
- an interruption during test evaluation may reuse completed training and
  evaluate only missing best/terminal test records;
- best and terminal/last-finite checkpoints remain distinct;
- fixed horizon remains not convergence;
- completion remains reviewer-run evidence with
  `formal_result_claim=false` and `method_ranking_claim_allowed=false`.

## 7. Acceptance tests

The implementation must pass deterministic fake-stack tests covering:

1. fresh run behavior remains unchanged without `--resume`;
2. CLI dispatch forwards `resume=False` and `resume=True` explicitly;
3. a durable state round-trip restores model, optimizer, scheduler, RNG,
   sampler offset, best/last-finite metadata, and metric rows;
4. uninterrupted training and interrupted-then-resumed training reach the same
   deterministic CPU parameters, scheduler state, metric rows, and summary;
5. mismatched config/input/model/adapter/method/seed fails before training;
6. corrupted or partial state fails closed;
7. stale temporary state is removed but never promoted;
8. an explicit numerical or runtime failure is not silently resumed;
9. completed methods are not retrained;
10. test data remains inaccessible until all training is complete;
11. resume during the test phase does not retrain completed methods;
12. every resumed record remains non-formal and preserves the event taxonomy.

The exact-head package, full repository pytest, Ruff, Ruff format, handoff
authority, formal execution channel, and public CLI checks must pass after the
slice.

## 8. Non-goals

This slice does not provide:

- distributed or multi-process checkpoint coordination;
- migration of arbitrary historical optimizer files;
- resumption in the middle of gradient accumulation or an optimizer step;
- automatic retry of explicit failures;
- altered checkpoint cadence;
- changed early stopping or convergence logic;
- real Qwen/CUDA liveness;
- scientific terminal audit or formal result promotion;
- main-branch merge authorization.
