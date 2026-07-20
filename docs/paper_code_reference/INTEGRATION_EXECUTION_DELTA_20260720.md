# PAPER-CODE-VALIDATION-01 integration execution delta

## Identity

- Claim: `PAPER-CODE-VALIDATION-01`
- Parent claim: `PAPER-CODE-REFERENCE-01`
- Pre-integration audit: `INTEGRATION_FRESHNESS_AUDIT_20260720.md`
- Audited main: `cd3271f844bcaf2550beb6247451dd3104258d0b`
- Pre-merge development head: `268f92e87e46c9023828f9750ea8e36cafc18e05`
- Integration merge commit: `5b4c1a7d138c2ba78778ca0e0a67be17bce536ac`
- Integration workflow run: `29750341987`
- Integration workflow job: `88378873573`
- Scientific status impact: none
- Formal experiment launched: no

## Execution result

The audited current-main commit was merged into
`dev/paper-code-reference-01` without rewriting branch history. The bounded
integration job passed.

The job verified before pushing:

1. the merge commit had the development head as first parent and the exact
   audited main commit as second parent;
2. `paper_code/`, `docs/paper_code_reference/`, and
   `.github/workflows/paper-code-validation.yml` were unchanged by the main-side
   merge;
3. `AGENTS.md`, `docs/handoff.md`, `experiments/registry.yaml`,
   `docs/governance_pipeline_stage_status.yaml`, and
   `docs/governance_rule_inventory.yaml` had exactly the same Git blobs as the
   audited main commit;
4. the temporary integration workflow was removed from the resulting merge
   commit;
5. the resulting worktree was clean before push.

The first integration attempt, run `29750225521`, produced no repository commit.
It failed inside an over-broad validation command that checked whitespace across
all 226 inherited main-side commits. That check was removed because it did not
isolate defects introduced by this merge. The second attempt retained the
source-tree and authority-blob checks above and passed.

## Current gate

The integration merge itself is complete, but the engineering gate is not closed
until the merged tree passes all exact-head checks and produces a newly validated
reviewer package.

Required checks:

- Evidence Locator Gate;
- PR Gate compile, shell, handoff-authority, formal-channel, governance, full
  pytest, and Ruff checks;
- Paper Code Validation package manifest, isolated install, self-contained
  tests, C-U1 and D-U1 tests, D-U1 public CPU liveness, Ruff and format checks,
  every public CLI entry point, and artifact upload;
- post-merge Countdown differential coverage against the expanded current-main
  high-c oracle.

The direct API commit that creates this delta is the first ordinary post-merge
branch commit and exists to trigger those exact-head checks. Its SHA and artifact
identity must be recorded only after the runs complete.

## Remaining boundaries

- Draft PR #149 remains unmerged.
- Protected human review for new Python files is not declared passed here.
- Hopper HDF5/MuJoCo, Countdown Qwen/PEFT/CUDA, and D4RL-9 real-stack liveness
  remain separate unresolved gates.
- Countdown interrupted optimizer/scheduler resume remains unimplemented.
- V5 scientific reproduction and terminal scientific review remain pending.
- No method ranking, convergence, steady-state, OOD, or universal superiority
  claim is created by this integration.
