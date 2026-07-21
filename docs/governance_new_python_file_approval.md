# New Python File Human-Approval Policy

Policy ID: `GOV-NEW-PYTHON-FILE-ORAL-APPROVAL-02`

Predecessor: `GOV-NEW-PYTHON-FILE-HUMAN-APPROVAL-01` remains preserved as historical provenance. This revision changes only the approval channel: it removes the duplicate GitHub Environment click and treats the repository owner's explicit oral approval as the authorization.

## Mandatory rule

A new tracked Python-file path may not enter `main` without explicit human approval of the exact path and its stated responsibility.

The repository owner's explicit approval in the active conversation is sufficient. The AI must preserve that approval in the pull-request discussion or another durable repository record. That record is evidence of the already-granted approval; it is not a second approval request.

A successful test run, reviewer-model verdict, code-size report, paired-repair output, or generic task instruction is not a substitute for explicit approval of the exact new Python path.

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

The rule is path-based for every contributor. GitHub commit metadata is not used to infer whether a change was produced by AI.

## Approval channel

The approval sequence is:

1. before creation, the AI names the exact proposed path, its responsibility, and why extending the nearest existing Python file is insufficient;
2. the repository owner explicitly approves that path in the active conversation;
3. the AI transcribes the approval into the PR discussion;
4. `.github/workflows/code-change-budget.yml` verifies that every current new Python destination is covered by the durable record;
5. the ordinary explicit merge-approval rule remains separate.

No GitHub Environment click is required by this policy. The workflow job keeps the display name `Approve large code change` only to avoid silently breaking an existing required-check binding; the job now verifies the durable oral-approval record rather than waiting for an Environment deployment.

A push to the PR head reruns the workflow. Approval is path- and responsibility-scoped rather than commit-SHA-scoped: normal edits to an already approved path do not require repeated approval, but adding another Python path or materially changing the approved responsibility does.

## Preferred durable record

The preferred owner-authored PR record is:

```text
DRPO-ORAL-APPROVAL: GOV-NEW-PYTHON-FILE-ORAL-APPROVAL-02
DRPO-APPROVED-REASON: new_python_file
DRPO-APPROVED-PATH: path/to/file.py
DRPO-RESPONSIBILITY: path/to/file.py :: concise responsibility
DRPO-REUSE-RATIONALE: path/to/file.py :: why the nearest existing file is insufficient
```

Repeat the path, responsibility, and reuse-rationale lines for every approved new Python destination.

For a hard-gate policy change or an independently large/structural change to existing Python files, use the matching reason and include a non-empty scope line:

```text
DRPO-APPROVED-REASON: hard_gate_policy_change
DRPO-APPROVED-SCOPE: exact approved governance change
```

or:

```text
DRPO-APPROVED-REASON: large_or_structural_python_change
DRPO-APPROVED-SCOPE: exact approved existing-code change
```

## Legacy approval compatibility

Already-open PRs do not require the user to repeat an approval that was validly given before this revision. For `new_python_file` only, the verifier accepts an owner-authored predecessor-format comment when the same comment contains:

- `Human approval record`;
- `GOV-NEW-PYTHON-FILE-HUMAN-APPROVAL-01`;
- the exact governed path in backticks;
- a statement of that path's responsibility.

This compatibility is narrow and does not waive exact path coverage. Any additional current Python destination still fails closed until explicitly approved and durably recorded.

## Self-protection

The following paths are part of the hard gate and any change to them requires explicit oral approval plus a durable scope record:

- `.github/workflows/code-change-budget.yml`;
- `AGENTS.md`;
- `docs/governance_new_python_file_approval.md`;
- `docs/scopes/GOV-NEW-PYTHON-FILE-HUMAN-APPROVAL-01.md`;
- `docs/scopes/GOV-NEW-PYTHON-FILE-ORAL-APPROVAL-02.md`;
- `docs/governance_stage_authorizations/GOV-NEW-PYTHON-FILE-HUMAN-APPROVAL-01.yaml`;
- `docs/governance_stage_authorizations/GOV-NEW-PYTHON-FILE-ORAL-APPROVAL-02.yaml`;
- `scripts/check_new_python_file_gate.sh`;
- `scripts/check_human_approval_record.sh`;
- `tests/test_new_python_file_gate.sh`;
- `tests/test_human_approval_record_gate.sh`.

A pull request cannot bypass self-protection by editing its own workflow or verifier because `pull_request_target` executes the versions from the base branch.

## Agent behavior

Before creating, copying, or renaming a file to a new `.py` path, an AI agent must:

1. identify the exact proposed path;
2. state its responsibility;
3. explain why extending the nearest existing Python file is insufficient;
4. obtain explicit human authorization for that path;
5. preserve the approval in the PR discussion using the preferred record or an equally explicit durable record.

The AI must not invent, infer, or fabricate approval; split one proposed module into several files to evade review; change letter case; use a rename; add an unrecorded path after approval; weaken the gate as part of an unrelated task; or push directly to `main` as a workaround.

## Existing Python files

This policy does not by itself forbid modifying an existing `.py` path. Existing scope, scientific-variable, test, liveness, code-budget, and merge gates continue to apply. When the workflow classifies an existing-Python change as large or structural, the same oral-approval-plus-durable-scope-record rule applies; no Environment click is required.

## Enforcement boundary

The machine gate verifies durable repository evidence, owner-account authorship, stated scope, and exact path coverage. The behavioral authority remains the user's explicit approval in the conversation.

This policy intentionally accepts the AI-authored durable transcription of the user's approval. It does not claim that a PR comment cryptographically proves who pressed each key, and it does not use GitHub Environment approval as a second identity channel.

Source-controlled workflows also cannot absolutely prevent a repository administrator from bypassing pull requests or required checks. Repository settings should still require the relevant checks and normal PR review where available, but no Environment reviewer is required by this policy.

## Migration boundary

The PR that introduces this revision is itself evaluated by the predecessor `pull_request_target` workflow from its base branch and therefore cannot live-test its own after-image. Deterministic shell tests and exact-head CI validate the implementation; the first later qualifying PR is the production liveness observation for the revised approval-record path.

## Exceptions and rollback

There is no automatic exception for an unapproved Python path. Emergency creation, weakening, removal, or rollback requires a new explicit human instruction naming the affected paths and reason. Historical scope, authorization, PR discussion, and CI evidence must be preserved.
