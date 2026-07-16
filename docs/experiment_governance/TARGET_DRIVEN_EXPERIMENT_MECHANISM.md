# Target-Driven Experiment Mechanism

Status: locked governance mechanism for development-stage scientific searches.

## Purpose

A target-driven experiment starts from a prespecified scientific signature and
uses each round to localize, stress-test, or confirm that signature. It is not a
license to hide contrary evidence, change metrics after inspection, or force a
preferred conclusion. It prevents the opposite failure: running unstructured
sweeps whose interpretation silently changes across sessions.

Every target-driven experiment MUST register five items before execution:

1. **Target signature.** The expected qualitative curve or mechanism.
2. **Historical evidence boundary.** Which earlier observations remain valid
   under their original protocol, and which successor claim is still pending.
3. **Diagnostic decision tree.** What to inspect before interpreting an
   unexpected result as scientific non-transfer.
4. **Bounded search sequence.** The exact order in which parameters may be
   opened, with stop and extension rules.
5. **Confirmation boundary.** Development data select the region; fresh seeds
   confirm the frozen setting. Test data may not select parameters.

## Evidence-preservation rule

A successor experiment with a changed mathematical protocol does not
retroactively erase an earlier observation. A non-replication may establish
that the earlier trend **did not transfer to the successor protocol**. It may
not be summarized as "all previous trends were wrong" unless an independent
audit establishes corruption, leakage, evaluator failure, or another defect in
the original evidence itself.

Each registered claim therefore has one of these statuses:

- `locked_directional_evidence`: an observation under its registered historical
  protocol;
- `pending_transfer_confirmation`: a successor formula or environment is being
  tested;
- `confirmed_transfer`: the frozen successor setting reproduces the target
  signature on fresh seeds;
- `not_yet_confirmed`: coverage or statistical power is insufficient;
- `non_transfer_under_successor_protocol`: a fully audited successor protocol
  does not reproduce the historical signature.

Only explicit user approval may change a locked directional-evidence status.

## Unexpected-result order of operations

Unexpected results MUST be handled in this order:

1. **Equation and implementation audit:** formula fingerprint, sign, detach,
   denominator, mask, and endpoint golden cases.
2. **Calibration audit:** active fraction, weight quantiles, nondegenerate
   scale, and endpoint behavior.
3. **Search-coverage audit:** determine whether the best point lies on a search
   boundary and extend only that boundary when required.
4. **Variance audit:** paired seeds, deterministic evaluation seeds,
   late-window and terminal results, and checkpoint-selection stability.
5. **Limited secondary-parameter check:** only parameters authorized by the
   registered sequence may be opened.
6. **Scientific interpretation:** only after the previous gates pass may a
   result be labeled non-transfer.

A smoke test, liveness run, static check, or one-seed pilot cannot satisfy the
scientific interpretation step.

## Reporting discipline

Task-performance degradation, support/variance or valid-structure boundary
events, and NaN/Inf numerical failure are always reported separately. A fixed
training horizon is not convergence. Development selection is not formal
ranking. All results, including target-inconsistent results, remain visible.
