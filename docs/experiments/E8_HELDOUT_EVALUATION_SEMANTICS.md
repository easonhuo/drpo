# Countdown E8 held-out evaluation semantics

## Authoritative clarification target

For the current Countdown E8 coefficient-response experiments, `val.jsonl` is the **structurally disjoint held-out evaluation split**. It carries the test-like scientific role for the reported coefficient-response evidence; the separately materialized `test.jsonl` is an additional split that was not accessed.

This clarification applies to the completed EXP/Linear/Tau and Reciprocal response curves and to the pending AsymRE delta-v response curve, subject to each experiment's existing provenance and claim limits.

## Why the split is held out

The split is separate from the offline training bank:

- canonical structure families do not overlap with the training bank;
- `(numbers, target)` problem keys do not overlap with the training bank;
- held-out rows never enter the training loss or optimizer update.

The filename `val.jsonl` is therefore an implementation name rather than the determinant of scientific identity. Its role in these experiments is held-out task-performance evaluation.

## Reporting rule

Paper-facing coefficient-response evidence reports every declared parameter point using:

- the fixed late-window Pass@8 summary;
- the fixed terminal Pass@8 checkpoint/horizon.

A held-out-selected `best_pass8_adapter` may be saved, but it is supplementary local diagnostic and recovery evidence only. It must not replace the late-window or terminal response curve and must not be presented as the source of the paper-facing coefficient response.

## Separate `test.jsonl` access fact

The separate `test.jsonl` remains unused. Statements such as `test_data_used: false`, `test split unused`, or historical `test access forbidden` refer only to that separate file. They do **not** mean:

- that held-out evaluation was absent;
- that task performance was measured on the training bank;
- that the existing response curves must be rerun solely because `test.jsonl` was unused.

This is a split-role clarification for the registered E8 evidence, not a general claim that validation and test are universally interchangeable.

## Adaptivity and claim boundary

Some later coefficient ranges were added after earlier held-out curves were inspected. The combined evidence must therefore be called **staged held-out evaluation response curves**, not an untouched one-shot confirmatory test.

This clarification does not upgrade any result. The experiments remain pilots and do not establish statistical significance, convergence, steady state, formal method ranking, OOD generalization, cross-task/model generalization, or universal exponential superiority.

Historical evaluator RNG contamination for the early EXP scans remains a separate provenance limitation and is not repaired or erased by this split-semantics clarification.

Task-performance outcomes, support/valid-structure boundary events, and NaN/Inf numerical failures remain separate reports.
