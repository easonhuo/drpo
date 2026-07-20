# New Python File Human-Approval Policy

Policy ID: `GOV-NEW-PYTHON-FILE-HUMAN-APPROVAL-01`

## Mandatory rule

A new tracked Python-file path may not enter `main` without explicit human approval.

This is a hard repository-governance rule. AI agents and automation must not treat a
successful test run, reviewer model verdict, code-size gate, paired-repair report, or
ordinary PR approval as a substitute for the dedicated human-approval environment.

## What counts as a new Python file

A destination path is governed when all of the following are true:

1. the path exists at the pull-request head;
2. the same path does not exist at the pull-request base;
3. the path suffix is `.py`, compared case-insensitively.

The rule therefore covers:

- a newly added Python file;
- a copied Python file at a new destination;
- a rename into a new Python destination;
- an upper- or mixed-case suffix such as `.PY` or `.Py`.

The rule is path-based for every contributor. GitHub commit metadata is not a reliable
oracle for deciding whether a change was produced by AI.

## Approval channel

The canonical source-controlled gate is `.github/workflows/code-change-budget.yml`.
It runs through `pull_request_target`, checks out the trusted base commit, evaluates the
base-to-head diff with `scripts/check_new_python_file_gate.sh`, and sends every governed
change to the existing `large-code-change-approval` environment.

The approval job name remains `Approve large code change` so an existing required-check
binding is not silently replaced. A push to the PR head reruns the workflow and invalidates
the prior run in accordance with the workflow concurrency policy.

## Self-protection

The following paths are part of the hard gate and any change to them also requires
human approval:

- `.github/workflows/code-change-budget.yml`;
- `AGENTS.md`;
- `docs/governance_new_python_file_approval.md`;
- `scripts/check_new_python_file_gate.sh`.

A pull request cannot bypass self-protection by editing its own workflow or detector,
because `pull_request_target` executes the versions from the base branch.

## Agent behavior

Before creating, copying, or renaming a file to a new `.py` path, an AI agent must:

1. identify the exact proposed path and why extending an existing Python file is
   insufficient;
2. obtain explicit human authorization for the new path;
3. preserve that authorization in the PR discussion or another durable repository
   record;
4. allow the GitHub environment approval to complete before merge.

An agent must not split one proposed module into several files to evade review, change
letter case, use a rename, modify the gate itself, or push directly to `main` as a
workaround.

## Existing Python files

This policy does not by itself forbid modifying an existing `.py` path. Existing change
budget, scope, scientific-variable, test, liveness, and merge gates continue to apply.

## External repository-settings requirement

Source-controlled workflows cannot by themselves prohibit a repository administrator
from bypassing pull requests or required checks. Absolute enforcement also requires
GitHub settings that:

- require pull requests for `main`;
- require the `Code Change Budget / Approve large code change` check when applicable;
- configure `large-code-change-approval` with a human required reviewer;
- prevent automation and administrators from bypassing those rules unless an explicit
  emergency policy says otherwise.

The current connected GitHub interface does not expose those settings. Their state must
be checked in GitHub Settings and must not be inferred from this file.

## Exceptions and rollback

There is no automatic exception. Emergency creation, weakening, removal, or rollback
requires explicit human approval naming the exact affected paths and reason. Historical
gate evidence must be preserved.