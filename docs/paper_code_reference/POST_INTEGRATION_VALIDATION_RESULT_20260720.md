# PAPER-CODE-VALIDATION-01 post-integration validation result

## 1. Identity

- Repository: `easonhuo/drpo`
- Branch: `dev/paper-code-reference-01`
- Claim: `PAPER-CODE-VALIDATION-01`
- Parent claim: `PAPER-CODE-REFERENCE-01`
- Audited main incorporated: `cd3271f844bcaf2550beb6247451dd3104258d0b`
- Integration merge commit: `5b4c1a7d138c2ba78778ca0e0a67be17bce536ac`
- Exact validation head: `479b1dadef168c9e42a0fd67cc60c66842e8f799`
- Current main re-resolved after validation: `cd3271f844bcaf2550beb6247451dd3104258d0b`
- Scientific status impact: none
- Formal experiment launched: no
- Draft PR merged: no

This record closes the engineering integration gate only. It is not a real-stack
Hopper, Countdown, or D4RL liveness result; it is not V5 scientific
reproduction; and it does not authorize any method ranking, convergence,
steady-state, OOD, or universal-superiority claim.

## 2. Integration execution

The bounded integration workflow merged the exact audited main commit into the
development branch and passed:

- workflow run: `29750341987`;
- job: `88378873573`;
- result: `success`.

Before pushing the merge commit, the job verified:

1. exact first and second parents;
2. no integration-side change to `paper_code/`,
   `docs/paper_code_reference/`, or the paper-code validation workflow;
3. exact blob identity with audited main for `AGENTS.md`, `docs/handoff.md`,
   `experiments/registry.yaml`, and the governance inventory/stage records;
4. self-deletion of the temporary integration workflow;
5. a clean resulting worktree.

The earlier failed integration workflow run `29750225521` produced no repository
commit. Its failed check was over-broad because it scanned whitespace across all
inherited main-side history rather than isolating integration changes. The
successful run retained all source-tree, parent, and authority-blob invariants.

## 3. Exact-head repository gates

At `479b1dadef168c9e42a0fd67cc60c66842e8f799`:

### Evidence Locator

- run: `29750416921`;
- result: `success`.

### PR Gate

- run: `29750417148`;
- job: `88379208707`;
- result: `success`.

Passed steps:

- tiered test-plan shadow;
- Python compile;
- shell syntax;
- handoff authority;
- formal execution channel;
- governance inventory;
- governance stage status;
- full repository pytest;
- Ruff.

This full-pytest pass includes the repository-internal differential and
short-trajectory suites after the expanded current-main Countdown provenance
oracle was incorporated. Therefore the post-merge regression requirement in
`INTEGRATION_FRESHNESS_AUDIT_20260720.md` is satisfied at this exact head.

## 4. Exact-head reviewer-package gates

Paper Code Validation:

- run: `29750419114`;
- job: `88379171954`;
- result: `success`.

Every declared step passed:

- build exact-head reviewer package;
- verify package manifest;
- install the extracted package in isolation;
- compile extracted source and tests;
- shared self-contained tests;
- public CLI self-contained tests;
- C-U1 self-contained tests;
- D-U1 CLI tests;
- D-U1 public smoke tests;
- D-U1 public CLI CPU liveness for all six methods;
- Ruff check;
- Ruff format check;
- all public CLI entry points;
- validated reviewer-package upload.

These are engineering checks. The D-U1 CPU route remains explicitly
non-formal; smoke completion is not a registered full reproduction.

## 5. Validated package

GitHub Actions artifact:

- artifact ID: `8464318725`;
- artifact name:
  `drpo-reference-v0.1-479b1dadef168c9e42a0fd67cc60c66842e8f799`;
- workflow artifact digest:
  `sha256:430048d3d0fbe280423625478703f11a9241c90dd67654052e5aa40f1693d28c`;
- expiry: `2026-08-19T14:25:20Z`.

Independent download verification performed in this review session:

- outer archive downloaded successfully;
- embedded `drpo-reference-v0.1.zip.sha256` check: passed;
- `SOURCE_COMMIT.txt`: exact validation head;
- `PACKAGE_MANIFEST.json` source commit: exact validation head;
- manifest file count excluding manifest: `68`;
- actual file count excluding manifest: `68`;
- every manifest path, SHA-256, and size matched the extracted files.

## 6. Gate result

```text
INTEGRATION_FRESHNESS = PASSED_AT_MAIN_cd3271f844bcaf2550beb6247451dd3104258d0b
POST_INTEGRATION_ENGINEERING_VALIDATION = PASSED
VALIDATED_REVIEWER_PACKAGE = AVAILABLE
SCIENTIFIC_REPRODUCTION = NOT_RUN
SCIENTIFIC_STATUS_CHANGE = NONE
PR_149_MERGE_READINESS = STILL_BLOCKED
```

The engineering package is now clean and integrated with the audited main
snapshot. That does not satisfy the task's scientific ready definition because
real external dependencies and registered full reproductions remain open.

## 7. Remaining gates

### Human and merge governance

- protected human review for all new Python paths is not confirmed by the
  connector;
- Draft PR #149 must remain Draft and unmerged without a separate explicit user
  merge instruction.

### Countdown

- real Qwen/Transformers/PEFT/CUDA liveness has not run;
- interrupted-run resume preserving optimizer and scheduler state is not
  implemented;
- final manuscript-facing execution, terminal review, and selected-conclusion
  report remain open;
- Countdown is external validity and does not replace controlled D-U1.

### Hopper E7-Q2

- registered HDF5 identity plus real Gymnasium/MuJoCo liveness has not run;
- registered full-budget rerun and terminal result review remain open;
- Hopper E7-Q2 remains external mechanism validation and does not replace C-U1.

### D4RL-9

- all nine dataset identities are not verified;
- final methods, coefficients, seeds, budgets, and checkpoint policy are not
  frozen;
- one-backend nine-task real liveness and formal execution remain blocked;
- no method-ranking claim is authorized.

### Controlled environments and V5

- C-U1 and D-U1 registered full reproductions and scientific terminal reviews
  remain separate pending gates;
- C-U1 terminology remains same-distribution held-out-context / unseen-state
  generalization, not OOD;
- task-performance collapse, support/probability/variance boundaries, NaN/Inf,
  environment invalidity, and incomplete terminal state remain separately
  reported.

## 8. Next engineering task

The next code-completeness item that can be addressed without external data or
GPU access is the Countdown interrupted-run resume gap. Any implementation must
be documented first, preserve the explicit configuration and paired-method
coordinate, restore model/optimizer/scheduler/RNG/progress identity, fail closed
on mismatch, and remain reviewer-run engineering functionality rather than a
scientific-status promotion.

This result-record commit changes only task-local documentation. Its own CI is a
non-self-referential regression guard; the evidence recorded above remains bound
to the exact validation head named in section 1.
