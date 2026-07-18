# GOV-NEW-PYTHON-FILE-HUMAN-APPROVAL-01

## Status

- class: repository governance default-policy change;
- repository: `easonhuo/drpo`;
- base branch: `main`;
- base commit: `4544005bd7df69c53bad70a9dcac846af01285e4`;
- development branch: `dev/gov-new-python-file-human-approval-01`;
- scientific experiment status impact: none;
- user authorization: explicit request on 2026-07-18 that every newly created Python file require human approval;
- implementation state: documentation-first, not yet merged.

## Claim

No AI agent, automation, or ordinary repository-development path may introduce a new
tracked Python-file path without the repository's human-approval gate.

For this policy, a new Python-file path is any changed destination path whose name ends
with `.py`, case-insensitively, that does not exist at the pull request base commit and
does exist at the pull request head. This includes additions, copies, and renames into a
new Python path.

The policy is path-based rather than author-identity-based because GitHub cannot
reliably infer whether a commit was authored by AI. Therefore the machine gate applies
to every pull request, including human-authored changes.

## Authorized scope

1. Add an explicit root-agent rule forbidding creation of a new `.py` path without
   human approval.
2. Strengthen the existing `Code Change Budget` workflow so every new Python path
   enters the existing human-review environment before merge.
3. Detect `.py` suffixes case-insensitively and include additions, copies, and renames.
4. Protect the hard-gate workflow, detector, root agent rule, and policy document from
   silent modification: changes to those paths also require human approval.
5. Preserve the existing approval-job name and environment so current required-check
   and environment configuration are not silently replaced.
6. Add shell-only deterministic tests. This governance implementation must not add a
   Python file.

## Explicit non-scope

- no scientific code, experiment configuration, seed, threshold, budget, result, or
  execution-order change;
- no change to `docs/handoff.md` or `experiments/registry.yaml`;
- no prohibition on editing an existing `.py` file beyond pre-existing change-budget
  rules;
- no automatic approval, merge, or human-identity inference;
- no claim that a repository file alone can replace GitHub branch protection or a
  required-reviewer configuration.

## Enforcement boundary

The repository-controlled enforcement point is a `pull_request_target` workflow that
runs trusted code from the base branch. A pull request cannot bypass the gate by
modifying its own copy of the workflow or detector because the base-branch version is
executed.

For merge prevention to be absolute, GitHub must also require the existing approval
check on `main`, require pull requests, and configure the referenced environment with a
human required reviewer. Those repository-settings controls are external to source
files and are not exposed by the current GitHub App interface; their state must be
verified separately rather than assumed.

## Acceptance

- modifying only an existing Python file does not trigger this specific new-file reason;
- adding `x.py` triggers human approval;
- adding `x.PY` triggers human approval;
- copying or renaming into a new `.py` destination triggers human approval;
- deleting a Python file does not trigger this specific new-file reason, though other
  existing structural-change rules may still require approval;
- modifying a protected gate-policy path triggers human approval;
- unresolved commits, malformed diff records, or inconsistent head paths fail closed;
- the exact-head full test suite, Ruff, handoff authority, formal-channel validator,
  governance validator, and shell syntax checks pass.

## Rollback

Rollback is one explicit human-approved revert that restores the previous workflow and
root `AGENTS.md`, and removes the policy, detector, shell tests, scope record, and
scoped authorization. Historical PR discussions and CI evidence remain preserved.

Do not weaken or remove this gate through an ordinary AI-generated change. A rollback
or exception requires a new explicit human authorization for the exact paths and
reason.