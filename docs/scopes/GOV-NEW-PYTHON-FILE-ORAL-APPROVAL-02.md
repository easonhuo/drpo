# GOV-NEW-PYTHON-FILE-ORAL-APPROVAL-02

## Status

- class: repository governance default-policy correction;
- repository: `easonhuo/drpo`;
- base branch: `main`;
- base commit: `b18aea9186d7e3ccc5d43b456719cafc23761e03`;
- development branch: `dev/gov-new-python-file-oral-approval-02`;
- scientific experiment status impact: none;
- user authorization: explicit instruction on 2026-07-21 that oral approval in the active conversation is sufficient and must not require a second GitHub Environment click;
- predecessor: `GOV-NEW-PYTHON-FILE-HUMAN-APPROVAL-01` remains preserved as historical provenance.

## Claim

A new tracked Python-file path still requires explicit human approval of the exact path and responsibility before creation. Once the repository owner gives that approval in the active conversation, the approval is complete. The AI must preserve it in the pull-request discussion, and the repository gate must verify that durable record. GitHub Environment approval is not a second authorization and is removed from this policy path.

## Authorized scope

1. Preserve repository-wide detection of additions, copies, renames, and case-insensitive `.py` destinations.
2. Preserve the requirement to state the exact path, responsibility, and why extending the nearest existing Python file is insufficient before creation.
3. Treat the repository owner's explicit oral approval in the active conversation as the authorization itself.
4. Require the AI to copy the approval into a durable PR comment; this record is evidence of the already-granted approval, not a second approval request.
5. Verify that every current new Python destination is covered by an owner-authored approval record. Any additional path not covered by the record fails closed.
6. Support a structured version-2 approval record and a narrow legacy compatibility path for already-open PRs whose comments contain the predecessor policy ID, an explicit human-approval statement, the exact backticked path, and its responsibility.
7. Remove the `large-code-change-approval` Environment from the new-Python-file approval path while retaining the existing approval-job display name for required-check compatibility.
8. Apply the same oral-approval-plus-durable-record rule to changes to the hard-gate workflow, detector, root agent rule, policy, scope, authorization, and regression test.
9. Add only shell/workflow/document changes; do not create a Python file.

## Approval record format

The preferred PR comment format is:

```text
DRPO-ORAL-APPROVAL: GOV-NEW-PYTHON-FILE-ORAL-APPROVAL-02
DRPO-APPROVED-REASON: new_python_file
DRPO-APPROVED-PATH: path/to/file.py
DRPO-RESPONSIBILITY: path/to/file.py :: concise responsibility
DRPO-REUSE-RATIONALE: path/to/file.py :: why the nearest existing file is insufficient
```

For a hard-gate policy change or a large existing-Python change, the record also names the applicable reason and includes a non-empty `DRPO-APPROVED-SCOPE:` line.

The record is path- and responsibility-scoped, not commit-SHA-scoped. Normal edits to an already approved path do not require repeated approval. Adding another Python path or materially changing the approved responsibility requires a new explicit approval and updated durable record.

## Explicit non-scope

- no scientific code, experiment configuration, seed, threshold, budget, result, execution order, or scientific status change;
- no change to `docs/handoff.md` or `experiments/registry.yaml`;
- no automatic creation of unapproved Python paths;
- no permission for an AI to invent, infer, or fabricate user approval;
- no automatic merge or replacement of the repository's separate final merge-approval rule;
- no claim that PR comments cryptographically distinguish a human keystroke from an action performed through the connected GitHub account.

## Enforcement boundary

The machine gate verifies durable repository evidence and exact path coverage. The behavioral authority remains the user's explicit approval in the conversation. Under this revised policy, the repository intentionally accepts the AI-authored durable transcription of that approval and no longer requires an independent Environment click.

The migration PR itself is evaluated by the predecessor `pull_request_target` workflow from its base branch, so it cannot prove the new after-image through its own live approval job. The after-image must be covered by deterministic shell tests and exact-head CI; the first subsequent qualifying PR provides the production liveness observation.

## Acceptance

- modifying only an existing Python file does not trigger the new-file path requirement;
- a new `.py`, `.PY`, copied, or renamed destination is detected;
- a complete structured owner approval record passes;
- a missing path, wrong author, missing responsibility, or missing reuse rationale fails closed;
- adding an extra unapproved Python destination fails closed;
- the narrow predecessor-format compatibility accepts the already-recorded approval on PR #218 without requiring the user to repeat approval;
- a hard-gate policy change requires a durable oral-approval scope record but no Environment click;
- the existing approval-job display name remains stable;
- shell syntax, focused regression tests, full pytest, Ruff, handoff authority, formal execution channel, and governance-stage validation pass.

## Rollback

Rollback requires a new explicit user instruction. Revert the workflow, detector, regression tests, `AGENTS.md`, and canonical policy together; remove this scope and authorization only in the same reviewed rollback; preserve the predecessor scope, both authorization histories, PR discussions, and CI evidence.
