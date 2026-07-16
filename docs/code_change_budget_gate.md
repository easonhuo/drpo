# Code Change Budget Gate

Claim: `GOV-CODE-CHANGE-BUDGET-01`

This gate is the immediate repository-level brake against experiment-by-copy
code growth. It is intentionally smaller than a general architecture policy.

## Automatic path

A pull request may proceed without human code-budget approval only when its
complete base-to-head Python diff satisfies both conditions:

1. additions plus deletions are at most 100 lines; and
2. no Python file is added, deleted, copied, or renamed.

The calculation is cumulative across the entire pull request, not per commit.
Exactly 100 changed Python lines remain automatic; 101 do not.

## Human-approval path

A pull request requires explicit human approval when either condition is true:

- cumulative Python churn exceeds 100 lines;
- any `.py` file is added, deleted, copied, or renamed, even if it is one line.

Before approval, the PR body must contain a JSON object between these markers:

```text
<!-- DRPO_CODE_CHANGE_JUSTIFICATION_START -->
{
  "purpose": "...",
  "nearest_existing_modules": ["src/drpo/existing.py"],
  "reuse_attempt": "...",
  "why_existing_code_is_insufficient": "...",
  "why_change_cannot_be_under_100_lines": "...",
  "file_responsibilities": {
    "src/drpo/existing.py": "...",
    "tests/test_existing.py": "..."
  },
  "new_python_files": {
    "src/drpo/new_module.py": {
      "closest_existing_module": "src/drpo/existing.py",
      "why_new_module_required": "...",
      "intended_reuse_by": "..."
    }
  },
  "tests": ["python -m pytest -q tests/test_existing.py"]
}
<!-- DRPO_CODE_CHANGE_JUSTIFICATION_END -->
```

The machine precheck verifies that the stated existing modules really exist at
the PR base, every changed Python path has one responsibility, every new Python
file has a separate justification, and the test inventory is non-empty. This
precheck verifies factual completeness and referenced-path existence; it does
not pretend to prove the architectural necessity of the change or replace
human judgment.

The human approver then runs the `Approve Large Code Change` workflow manually,
providing the PR number, the exact reviewed head SHA, and an approval reason.
Approval is bound to both the exact head commit and the SHA-256 of the exact
justification JSON. A new commit or edited justification invalidates it.

## Required repository setting

The workflow cannot make itself merge-blocking without repository rules. After
this change reaches `main`, configure the `main` branch ruleset to:

- require a pull request before merging;
- require status check `drpo/code-change-budget`;
- block direct and force pushes;
- disallow bypass for the actors used by coding agents.

Until that status is required by the branch ruleset, the evaluator reports the
correct decision but a repository administrator can still override it. Do not
claim the gate is non-bypassable before this one-time setting is active.

## Security boundary

The evaluator uses `pull_request_target`, checks out only the trusted base, and
fetches the PR head as Git objects without executing it. A PR cannot weaken the
gate by modifying its own copy of the workflow or validator. The manual
approval workflow runs only from the default branch and is not automatically
dispatched by coding agents.

## Rollback

Revert the single merge commit that introduces the two workflows, validator,
tests, this document, and its authorization record. Remove the required status
from the branch ruleset only after that revert is complete, so an in-flight PR
cannot exploit a temporary unprotected interval.
