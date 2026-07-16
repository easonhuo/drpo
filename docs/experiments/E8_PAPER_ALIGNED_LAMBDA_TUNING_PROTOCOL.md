# E8 Paper-Aligned Lambda Tuning Protocol

Experiment ID:
`EXT-C-E8-ORACLE-OFFLINE-V2-PAPER-ALIGNED-LAMBDA-ROUND1-0.5B-01`

Status: registered development pilot; not formal evidence and not yet run.

## 1. Locked scientific scope

Countdown supplies external-validity evidence only. It does not replace C-U1
or D-U1 causal identification.

The historical squared-surprisal sweep is retained as
`locked_directional_evidence`: it observed a non-monotone relationship between
negative-control strength and task performance. The current experiment asks a
narrow successor question:

> Does the paper-aligned linear excess-surprisal taper produce a reproducible
> useful lambda region while preserving near-field negative feedback?

Failure to reproduce this signature may establish non-transfer to the
paper-aligned formula. It does not retroactively declare the historical sweep
unobserved or invalid.

## 2. Formula contract

For a Countdown completion with valid completion-token count `L`:

```text
D = -(1/L) * sum_t log pi_theta(y_t | x, y_<t)
z = relu((D - tau) / scale_c)
w = alpha * exp(-lambda * z)
```

The prompt and padding are excluded. The EOS token is included in `L`, matching
`arena.encode_prompt_completion` and `arena.completion_stats`. `D`, `z`, and the
weight are stop-gradient. No extra square is permitted.

Parameter roles are frozen:

- `alpha`: global negative-update strength;
- `tau`: the surprisal threshold below which the negative weight is unchanged;
- `scale_c`: a data-derived surprisal unit;
- `lambda`: the dimensionless decay strength per normalized excess-surprisal
  unit;
- `lambda / scale_c`: the effective slope on the raw surprisal axis above
  `tau`.

Golden cases:

```text
D <= tau             -> w = alpha
D = tau + scale_c    -> w = alpha * exp(-lambda)
D = tau + 2*scale_c  -> w = alpha * exp(-2*lambda), never exp(-4*lambda)
```

As `lambda -> 0`, the taper cell approaches uncontrolled negative training at
`alpha=1`. As `lambda -> infinity`, far-tail negatives vanish but near-field
negatives remain at full weight. The high-lambda endpoint is therefore
**near-field-only negative training, not Positive-only**. Positive-only remains
an independent baseline.

## 3. Calibration contract

Before training, a frozen deterministic sample of 256 bank prompts is scored by
the current initialization. No reward, validation metric, or test metric is
used.

```text
tau     = median(D)
scale_c = median(upper half of D) - median(lower half of D)
```

The calibration sample may also appear in the training bank; this is a
pre-training unsupervised coordinate calibration, not a held-out performance
split. The calibration JSON is hashed into every cell identity. A degenerate
scale or active-tail fraction below 0.25 fails closed.

## 4. Baseline reuse

Positive-only and Global do not depend on the erroneous historical squared
surprisal formula.

- Positive-only is included with three paired development seeds to solidify its
  current baseline level. Historical summaries place it roughly in the
  0.13--0.15 Pass@8 range depending on best/terminal aggregation; the new round
  must report best, late-window, and terminal separately.
- Historical Global x1/32 is reused as context and is not rerun in Round 1. Its
  earlier best improvement and terminal reversal must remain visible rather
  than being summarized by the best checkpoint alone.

A reused baseline is not automatically pooled with the new paired cells. Exact
pooling requires matching provenance, seeds, evaluator, and training budget.

## 5. Bounded tuning sequence

### Round 0 — equation and liveness gate

Run formula golden tests, emit token-level and completion-level diagnostics,
create frozen calibration, and execute a two-step representative smoke run.
This round is not scientific evidence.

### Runtime resource default

Round 1 defaults to **two auto-selected GPU slots** with one training process per GPU. The launcher chooses the two eligible devices from the configured candidate pool using free-memory and utilization gates. `--max-devices` is an explicit operator override; changing the number of slots changes only wall-clock scheduling, never the scientific matrix, seeds, formulas, or cell identities.

## Round 1 — lambda localization

Freeze `alpha=1`, calibrated `tau`, and calibrated `scale_c`. Run Positive-only
and five lambda values on three paired development seeds:

| lambda | retained tail weight at `z=1` |
|---:|---:|
| 0.105360516 | 0.90 |
| 0.287682072 | 0.75 |
| 0.693147181 | 0.50 |
| 1.386294361 | 0.25 |
| 2.302585093 | 0.10 |

This is six parameter points and eighteen cells. Validation Pass@8 is the
selection metric; Pass@64, valid rate, late-window behavior, and terminal
reversal are mandatory diagnostics. Test data are forbidden.

### Round 2 — boundary extension or local refinement

- Best lambda at the left boundary: extend left before interpretation.
- Best lambda at the right boundary: extend right before interpretation.
- Best lambda inside the grid: add a small log-spaced refinement around the two
  neighboring points.
- Disordered seed rankings: add paired seeds before opening another parameter.

### Round 3 — optional alpha check

`alpha` remains 1 unless the lambda curve is coherent but globally misplaced
relative to Positive-only and historical Global. Only then may a narrow,
registered alpha check be opened around the best lambda. A large alpha-by-lambda
grid is forbidden.

### Round 4 — tau sensitivity

Only after the lambda region is localized may the threshold be checked at:

```text
tau0 - 0.5*scale_c, tau0, tau0 + 0.5*scale_c
```

This is a sensitivity check, not permission to tune on the test set.

### Round 5 — fresh-seed confirmation

Freeze `alpha`, `lambda`, `tau`, and `scale_c`; use fresh seeds not involved in
selection. Only this stage can support a transfer-confirmation claim.

## 6. Unexpected-result decision tree

1. Wrong endpoint behavior, all weights near one/zero, or non-monotone weights:
   implementation/calibration failure; repair and rerun the same registered
   cells.
2. Best point on a lambda boundary: incomplete coverage; extend that boundary.
3. Internal shape but unstable seed order: increase paired-seed evidence.
4. Coherent curve but no gain over Positive-only: consider the registered
   narrow alpha check, then tau sensitivity.
5. Only after all gates pass and fresh seeds fail to reproduce the target may
   the result be labeled `non_transfer_under_successor_protocol`.

At no point may a successor non-transfer result be rewritten as "all previous
Countdown trends were wrong."

## 7. Execution and audit

Canonical launcher:

```bash
bash scripts/run_countdown_e8_paper_aligned_lambda_one_click.sh
```

The launcher performs plan, calibration, liveness, and the resumable sweep.
Completed cells are reused only when model, bank, validation set, config,
calibration, code, and cell identity hashes match.

Required outputs include `RUNTIME_SELECTION.json`, `SWEEP_PLAN.json`,
`TAPER_CALIBRATION.json`, `SMOKE_GATE.json`, per-cell summaries,
`aggregate/lambda_summary.csv`, `aggregate/ROUND1_DECISION.json`,
`aggregate/terminal_audit.json`, and `SWEEP_COMPLETE.json`.
