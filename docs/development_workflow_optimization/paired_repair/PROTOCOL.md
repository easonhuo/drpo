# Paired Repair Workflow

Claim: `GOV-CODE-PAIRED-REPAIR-01`

Frozen candidate feedback snapshot:
`7826f5d60c83d8a58a11dc526b487cc09078d818`.

## Purpose

Keep one real first implementation as an untreated baseline (`A0`), then let the same
worker perform one feedback-driven repair (`B1`). This answers the practical question:
"Did the new feedback improve this implementation?" without creating a second worker
or a regeneration platform.

It is not a randomized Arm-A/Arm-B experiment and must not be reported as one.

## Preconditions

- the task is a real approved development task;
- A0 is complete enough for ordinary review and has a full commit SHA;
- A0 was committed before structured repair feedback was delivered;
- the same worker/session identity can perform the repair;
- the task's scientific variables and responsibilities are already frozen where
  applicable;
- required tests and liveness are known before B1 is judged.

## Step 1: Freeze A0

Use an external temporary evidence directory so evidence files do not accidentally
enter the B1 code commit.

```bash
python scripts/paired_repair_report.py freeze-a0 \
  --repo-root . \
  --base <TASK_BASE_SHA> \
  --a0 <A0_SHA> \
  --claim <CLAIM_OR_EXPERIMENT_ID> \
  --worker '<STABLE_WORKER_LABEL>' \
  --gate-snapshot 7826f5d60c83d8a58a11dc526b487cc09078d818 \
  --record-dir /tmp/drpo-paired/<TASK_ID>
```

The command verifies `base -> A0` ancestry and records base-to-A0 code metrics.

## Step 2: Deliver structured feedback

The feedback must be recorded before B1 and should contain:

- verdict: accept as-is or repair required;
- duplicated or unnecessary responsibilities;
- nearest existing modules/symbols that should be reused;
- concrete required repair;
- explicit statement that scientific variables and task responsibilities must not
  change.

A timestamped PR comment is the preferred source. Save the exact feedback text to a
file and retain the PR comment ID or another durable source identifier.

The reviewer applies the frozen candidate code-change-budget review at the snapshot
listed above. The workflow reuses that reviewed rubric instead of copying or expanding
the gate implementation.

## Step 3: Produce B1

The same worker receives the recorded feedback and may perform one bounded repair.
B1 must be a descendant of A0. Do not rewrite A0 history, squash away A0, or expand the
scientific scope during repair.

Run the same required checks on both versions where applicable. Record evidence using
`VALIDATION_TEMPLATE.json`.

## Step 4: Close the pair

```bash
python scripts/paired_repair_report.py close-b1 \
  --repo-root . \
  --record-dir /tmp/drpo-paired/<TASK_ID> \
  --b1 <B1_SHA> \
  --worker '<STABLE_WORKER_LABEL>' \
  --feedback-file /tmp/gate-feedback.md \
  --feedback-source 'pr-comment:<COMMENT_ID>' \
  --validation-file /tmp/paired-validation.json
```

The command verifies `A0 -> B1`, copies the durable feedback and validation evidence,
and writes:

- `PAIR.json`;
- `GATE_FEEDBACK.md`;
- `VALIDATION.json`;
- `COMPARISON.md`.

## Interpretation

Possible evidence verdicts:

- `B1_ELIGIBLE_AND_SMALLER`: B1 preserved A0-passing checks, passed reviewer
  correctness, and reduced production churn or new production files;
- `B1_ELIGIBLE_NO_SIZE_GAIN`: B1 remained eligible but produced no measured size
  benefit;
- `B1_INELIGIBLE_RETAIN_A0_OR_REPAIR`: B1 lost correctness or required evidence.

These are evidence classifications, not merge decisions. A human still selects A0,
B1, or neither.

## Correctness and reporting rules

- never prefer a smaller incomplete implementation;
- a check that passed for A0 must also pass for B1;
- required liveness may be `not_applicable`, but must not be silently omitted;
- task-performance collapse, support/boundary events, and NaN/Inf remain separate
  whenever the underlying task owns those diagnostics;
- do not call this a causal A/B result;
- preserve unsuccessful B1 repairs instead of deleting them.

## Observation storage

After B1 is frozen, copy the external evidence directory into a task-specific location
under:

```text
docs/development_workflow_optimization/paired_repair/observations/<TASK_ID>/
```

Commit evidence separately from the B1 implementation commit so code metrics remain
unambiguous.
