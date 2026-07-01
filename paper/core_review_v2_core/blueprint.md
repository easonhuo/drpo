# Executable blueprint: PAPER-PIPELINE-V2-CORE-01

Snapshot: `34ba99a5e3cd25d370d3984c51146bb64f4a39d582f602dfdea6338d06c58cd5`

## EXP-P04-A - Fixed-variance causal intervention

- Reader question: Does retaining the far-field negative path cause task collapse in the controlled C-U1 environment?
- Paragraph claim: Baseline and Near-zero collapse in all paired seeds, whereas Far-zero and Far-cap prevent collapse and retain high terminal reward.
- Sentence plan:
  1. State the matched four-way intervention and the 20 paired held-out seeds.
  2. Report Baseline and Near-zero terminal reward with task-collapse counts.
  3. Report Far-zero and Far-cap terminal reward with confidence intervals and collapse counts.
  4. Report Global-scale and Far-to-near as registered budget controls, not as a universal ranking.
  5. Conclude only that the far-field path is the dominant causal transmission path in this controlled environment.
- Reviewer objection: The rescue may reflect removing all negative information.
- Response: Near-zero removes the near component yet does not rescue; Far-cap retains bounded far influence and rescues.
- Budget controls: Global-scale and Far-to-near are included from the fixed-variance registered controls.
- Figure/table: `cu1_e3_fixed_reward.pdf`, `cu1_e3_results.tex`.

## EXP-P04-B - Learnable-variance boundary audit

- Reader question: Is the learnable-variance failure task collapse, a support boundary, or numerical failure?
- Paragraph claim: Baseline and Near-zero reach support contraction in all seeds near step 73, while Far-zero and Far-cap avoid that event; no method produces NaN/Inf.
- Sentence plan:
  1. Name the first registered event as support/variance contraction.
  2. Report the event counts and onset without relabeling it as task or numerical collapse.
  3. Report the absence of NaN/Inf separately.
  4. State the bounded controlled-environment conclusion.
- Reviewer objection: The boundary classification may be a numerical artifact.
- Response: The terminal audit records finite parameters and zero NaN/Inf events.

## METHOD-P03 - Proposition 2

- Reader question: Why does exponential remoteness weighting control a polynomially growing far-field score?
- Claim: Exponential decay dominates every finite polynomial order.
- Assumption: The unweighted score-times-advantage norm is at most `C(1+r)^k` for finite `C,k`.
- Conclusion ceiling: The weighted contribution vanishes; this does not model or assume exponential sample utility.

## Exact metric paths

- `methods.baseline.fixed_variance.reward`
- `methods.baseline.fixed_variance.reward_ci95`
- `methods.baseline.fixed_variance.task_collapse_count`
- `methods.baseline.fixed_variance.nan_inf_count`
- `methods.baseline.learnable_variance.support_boundary_count`
- `methods.baseline.learnable_variance.nan_inf_count`
- `methods.near_zero.fixed_variance.reward`
- `methods.near_zero.fixed_variance.reward_ci95`
- `methods.near_zero.fixed_variance.task_collapse_count`
- `methods.near_zero.fixed_variance.nan_inf_count`
- `methods.near_zero.learnable_variance.support_boundary_count`
- `methods.near_zero.learnable_variance.nan_inf_count`
- `methods.far_zero.fixed_variance.reward`
- `methods.far_zero.fixed_variance.reward_ci95`
- `methods.far_zero.fixed_variance.task_collapse_count`
- `methods.far_zero.fixed_variance.nan_inf_count`
- `methods.far_zero.learnable_variance.support_boundary_count`
- `methods.far_zero.learnable_variance.nan_inf_count`
- `methods.far_cap.fixed_variance.reward`
- `methods.far_cap.fixed_variance.reward_ci95`
- `methods.far_cap.fixed_variance.task_collapse_count`
- `methods.far_cap.fixed_variance.nan_inf_count`
- `methods.far_cap.learnable_variance.support_boundary_count`
- `methods.far_cap.learnable_variance.nan_inf_count`
- `methods.global_scale.fixed_variance.reward`
- `methods.global_scale.fixed_variance.reward_ci95`
- `methods.global_scale.fixed_variance.task_collapse_count`
- `methods.global_scale.fixed_variance.nan_inf_count`
- `methods.global_scale.learnable_variance.support_boundary_count`
- `methods.global_scale.learnable_variance.nan_inf_count`
- `methods.far_to_near.fixed_variance.reward`
- `methods.far_to_near.fixed_variance.reward_ci95`
- `methods.far_to_near.fixed_variance.task_collapse_count`
- `methods.far_to_near.fixed_variance.nan_inf_count`
