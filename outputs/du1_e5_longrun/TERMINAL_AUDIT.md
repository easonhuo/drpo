# D-U1-E5-LONGRUN-RERUN terminal audit

## Integrity

- Expected/actual method-seed runs: 120/120
- Direct reference checks: passed
- All causal runs classified: yes
- Historical qualitative classes matched: 120/120
- NaN/Inf: 0/120
- Run commit: `22c5823d66169eb90c256de342e27c5391e464c3`
- Worktree clean at launch and exit: yes
- Scientific process return code: 0

## Scientific classification

- `positive_only`, `far_zero`, and `far_cap`: 20/20 `stable_bounded`.
- `baseline`, `near_zero`, and `global_scale`: 20/20 `support_boundary`.
- Task collapse occurs for `baseline` and `near_zero` only, 20/20 each.
- `global_scale` demonstrates support-boundary failure without task-performance collapse.

## Closure decision

The registered long-run checks pass. E5 is `long_run_validated` for the controlled D-Diag/D-U1 categorical mechanism reconstruction. This does not upgrade the missing legacy runner to byte-identical reproduction and does not replace E6 semantic generalization.
