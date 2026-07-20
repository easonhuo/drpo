# PAPER-CODE-REFERENCE-01 Current Status

**Document role:** canonical task-local current snapshot and continuation index.  
**Not a research master:** `docs/handoff.md` remains the unique research source of truth.  
**Claim:** `PAPER-CODE-REFERENCE-01`  
**Scientific-status impact:** none.  
**Latest implementation head before this document:** `5791f279e5177542d38cc2c4e235f4fddbc04fde`.

This file consolidates the current engineering state without deleting historical
records. It supersedes stale current-status and next-slice statements in older
task-local handoff files. Those older files remain provenance.

## 1. Mandatory continuation order

A continuation session must resolve and read, in order:

1. current-main `AGENTS.md`;
2. Section 0 of current-main `docs/handoff.md`;
3. current-main `experiments/registry.yaml`;
4. this file;
5. `docs/paper_code_reference/ACCEPTANCE_MATRIX.yaml`;
6. `docs/paper_code_reference/SOURCE_MIGRATION_MAP.md`;
7. the actual `dev/paper-code-reference-01` branch, Draft PR `#149`, changed
   files, differential oracles, approval records, and exact-head CI.

`docs/handoff.md` remains the research authority. This file is an engineering
continuation index only.

## 2. Repository and branch snapshot

At the audit immediately before this document:

- repository: `easonhuo/drpo`;
- default branch: `main`;
- current `main`: `dd46727c1efefd2e6d4cdf6f3b204ec1fc58fca3`;
- task base and merge-base lineage: `4544005bd7df69c53bad70a9dcac846af01285e4`;
- development branch: `dev/paper-code-reference-01`;
- implementation head before this document: `5791f279e5177542d38cc2c4e235f4fddbc04fde`;
- persistent cumulative Draft PR: `#149`;
- PR state: open, Draft, unmerged;
- overall task state: `in_development`.

These SHA values are audit facts, not reusable assumptions. Resolve both heads
again before any further work. The branch diverges from newer `main`; an
integration-freshness audit is required before a merge proposal. Do not silently
merge or rebase scientific registrations from `main` into this task.

## 3. Reviewer-facing code boundary

The `paper_code` package provides readable algorithms, explicit input and model
identities, runnable training/evaluation, checkpoints, lightweight
completion/failure records, and simple summaries.

It does not duplicate registry/handoff mutation, formal-evidence promotion,
scientific terminal adjudication, manuscript binding, or internal artifact
promotion. A successful public command, fake backend, smoke test, or finite
training horizon is not a scientific terminal result.

## 4. Two-axis status model

Do not conflate scientific status registered on `main` with `paper_code`
migration status.

| Component | Existing scientific status | `paper_code` migration status | Remaining gate |
|---|---|---|---|
| Shared utilities and controls | no independent claim | implementation complete | final integration review |
| C-U1 | handoff/registry remain authoritative | implementation complete | registered reproduction and terminal review |
| D-U1 revision 4 | active matrix remains `not_run` | implementation complete | registered formal run and terminal review |
| Hopper E7-Q2 | existing mechanism result remains `long_run_validated` | implementation complete | registered-data reproduction through reviewer runner |
| D4RL-9 / `EXT-H-E7-BENCH-01` | archive is pilot provenance; no formal ranking | reviewer algorithm, multi-method training, rollout, and aggregation complete | real HDF5/MuJoCo liveness and final protocol |
| Countdown / `EXT-C-E8-TAPER-0.5B-01` | active-tail experiment remains **pilot / `not_run`** | **reviewer-code migration closed** | real Qwen/CUDA liveness, scientific execution, terminal review |

No migration activity in this branch changes a scientific result status.

## 5. Current implementation ownership

### 5.1 C-U1, D-U1, Hopper, and D4RL

- C-U1 lives under `continuous/` with dispatch in `cli.py`.
- D-U1 revision 4 lives under `categorical/du1_*.py` with dispatch in `cli.py`.
- Hopper E7-Q2 lives in `experiments/hopper.py` plus `external/hopper_*.py`.
- D4RL-9 uses one reviewer-facing backend in `experiments/d4rl.py`, orchestration
  in `experiments/__init__.py`, and task contracts in `external/d4rl_tasks.py`.

Their scientific execution and terminal-review gates remain separate from
reviewer-code completeness.

### 5.2 Countdown algorithm core

The user-approved path
`paper_code/src/drpo_reference/categorical/countdown.py` owns dependency-light,
protocol-independent Countdown behavior. Approval was recorded before creation
in Draft PR #149 comment `5016309623`.

It includes:

- expression cleaning, exact arithmetic verification, and verifier categories;
- chat prompting, completion masking/EOS, padding, completion-only likelihood,
  entropy, and direct-logit score;
- first-occurrence unique-negative banks and per-prompt denominators, never
  weight-sum normalization;
- historical Round-1 `u=-log P/2` compatibility;
- registered active-tail `S=max(-log P-tau,0)/scale`, `d=sqrt(S)`;
- Positive-only, Uncontrolled, Global matched, Reciprocal-linear, Exponential,
  and Squared-distance exponential weights;
- deterministic `eval()` plus `no_grad()` current weights followed by a separate
  gradient-bearing negative forward;
- exact joint objective, prompt-balanced sampling, model-backed raw-gradient
  calibration, diagnostics, and response aggregation.

The core imports neither Transformers nor PEFT.

### 5.3 Countdown reviewer runtime

The user separately approved the exact path
`paper_code/src/drpo_reference/experiments/countdown.py`. The approval and path
responsibility were recorded before creation in Draft PR #149 comment
`5019085196`.

This runtime owns:

- lazy Transformers/PEFT loading and optional bitsandbytes boundary;
- tokenizer, base model, fresh LoRA or explicitly supplied initial adapter;
- explicit JSON runtime configuration with no hidden paper defaults;
- replay, independent calibration, validation, and delayed optional test input;
- input hashes and prompt-disjointness checks;
- per-seed calibration and shared-initialization digest checks;
- prompt-balanced paired training;
- AdamW, cosine warmup, gradient accumulation, clipping, raw/update norms, and
  non-finite guards;
- rollback to the last finite parameter state when an optimizer step becomes
  non-finite;
- best, last-finite, and terminal adapter checkpoints;
- Greedy/Pass@k generation and best-versus-terminal evaluation;
- per-run/root completion and failure records plus simple seed aggregation.

The public command is:

```text
drpo-reference countdown --config <explicit-json> --output <new-or-empty-dir>
```

The base package remains Transformer-free; optional dependencies are declared as
`countdown` and `countdown-4bit` extras.

### 5.4 Failure and evidence semantics

The runtime keeps distinct:

- training/runtime failure;
- NaN/Inf numerical failure;
- invalid or unavailable evaluation input;
- checkpoint-evaluation failure;
- task-performance collapse;
- support/probability-boundary events;
- incomplete terminal scientific review.

Task-collapse and support-boundary fields remain unset because no reviewer-facing
threshold has been frozen. Fixed-horizon completion is explicitly not
convergence. Every output keeps `formal_result_claim: false` and forbids a method
ranking claim.

## 6. Engineering validation

Completed before this status-document update:

- stable-core differential tests and active-tail first-update tests;
- controlled local fake-HF end-to-end integration covering configuration,
  calibration, two methods, multi-step updates, clipping, checkpoint lifecycle,
  delayed test access, generation/evaluation, and aggregation;
- local Python compile and Ruff for the runtime;
- exact-head full CI for the initial approved runtime commit
  `ded2ffe15e7f09fc358657eea553b57ec33e297d`;
- CLI dispatch test added to existing `paper_code/tests/test_cli.py`.

The final cumulative head after this document still requires its own exact-head
Evidence Locator, compile, governance, full pytest, and Ruff confirmation.
Controlled fake-HF execution is engineering evidence only and is not real
Qwen/CUDA liveness.

## 7. Remaining Countdown work

Reviewer runtime gaps:

- resume of an interrupted optimizer/scheduler/run state;
- real Qwen/PEFT/CUDA liveness, including an explicitly selected 4-bit path when
  applicable;
- any runtime fixes revealed by real model/data execution.

Scientific/protocol gates:

- final manuscript-facing model and initialization;
- final comparison arms and coefficient source;
- formal or confirmatory seeds and training budget;
- checkpoint-selection and terminal-audit policy;
- validation/test access protocol and final evaluation coordinate;
- manuscript values and any method-ranking claim;
- registered real execution and scientific terminal review.

Governance/integration gates:

- human-only protected environment `large-code-change-approval` must be observed;
- exact-head cumulative CI must pass;
- integration-freshness audit against current `main`;
- explicit user merge instruction.

## 8. Exact next sequence

1. Synchronize the acceptance matrix and source migration map with this runtime.
2. Run exact-head full CI and inspect the protected human-review status.
3. Do not launch a scientific experiment from this migration branch.
4. Execute a separately authorized real Qwen/CUDA liveness only after the exact
   runtime coordinate and available hardware/data are resolved.
5. Audit integration freshness before any merge proposal.

No formal experiment was launched by this migration. No method ranking,
scientific status, model scale, seed, threshold, training horizon, default
coefficient, checkpoint rule, or manuscript value changed.


## 9. Countdown reviewer-code closure

The canonical reviewer configuration is
`paper_code/configs/countdown_e8_taper_0p5b.json`, bound to
`EXT-C-E8-TAPER-0.5B-01-v79`. Schema version 2 validates the registered
result-affecting coordinate fail-closed: Qwen2.5-0.5B-Instruct identity, LoRA
`r=32/alpha=64/dropout=0.05` and target modules, structured data and reference
contracts, natural replay construction, six-method order, seeds
`9234/10234/11234`, 1200-update AdamW coordinate, active-tail calibration,
Greedy/Pass@8 evaluation, `seed+700000` paired evaluation, `0.002` selection
delta, best plus terminal/last-finite reporting, delayed test access, and
structure-aware held-out metrics. Model, adapter, input identities, and canonical
row counts are checked at runtime. Schema version 1 remains a custom reviewer
coordinate and does not claim canonical-v79 identity.

**Countdown reviewer-code migration is closed.** Remaining real Qwen/PEFT/CUDA
liveness, interrupted-run resume, protected human review, integration freshness,
registered execution, and scientific terminal review are not missing migration
code. No experiment was launched and the scientific status remains **pilot /
`not_run`**.
